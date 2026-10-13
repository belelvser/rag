"""Лёгкий SDK-клиент и JSON-контракт; здесь нет импортов RAG или ML-библиотек."""

import asyncio
import logging
import math
import os
from typing import Literal

os.environ["OTEL_SDK_DISABLED"] = "true"

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.client.errors import A2AClientTimeoutError, AgentCardResolutionError
from a2a.helpers import new_data_message
from a2a.types import Message, Role, SendMessageRequest
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict, ParseError
from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse

AGENT_URL = os.getenv("RAG_AGENT_URL", "http://127.0.0.1:9999").rstrip("/")
AGENT_PUBLIC_URL = os.getenv("RAG_AGENT_PUBLIC_URL", AGENT_URL).rstrip("/")
SDK_VERSION = "1.2.2"
PROTOCOL_VERSION = "1.0"
A2A_TIMEOUT = 320
ERROR_STATUS = {"invalid_input": 422, "busy": 429, "unavailable": 503, "timeout": 504, "internal": 500}
logger = logging.getLogger("uvicorn.error")


class AgentCallError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class SearchData(SearchRequest):
    operation: Literal["search"]


class AskData(AskRequest):
    operation: Literal["ask"]


class ApplicationError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["invalid_input", "busy", "unavailable", "timeout", "internal"]
    message: str


def parse_input(data):
    """Protobuf Value хранит JSON-числа как double; целые top_k восстанавливаем явно."""
    if not isinstance(data, dict):
        raise ValueError("Data-part должен содержать JSON-объект.")
    data = dict(data)
    number = data.get("top_k")
    if isinstance(number, float) and math.isfinite(number) and number.is_integer():
        data["top_k"] = int(number)
    if data.get("operation") == "search":
        return SearchData.model_validate(data)
    if data.get("operation") == "ask":
        return AskData.model_validate(data)
    raise ValueError("operation должна быть search или ask.")


def message_data(message):
    if not isinstance(message, Message) or len(message.parts) != 1 or not message.parts[0].HasField("data"):
        raise ValueError("Ожидался Message с одной структурированной data-part.")
    if message.parts[0].media_type != "application/json":
        raise ValueError("Data-part должна иметь mediaType application/json.")
    return MessageToDict(message.parts[0].data)


def validate_result(envelope, operation, question, top_k=3):
    """Неверный ответ агента — 502; ошибки приложения приходят внутри data-part."""
    if not isinstance(envelope, dict) or type(envelope.get("ok")) is not bool:
        raise ValueError("В ответе агента отсутствует boolean ok.")
    if not envelope["ok"]:
        if set(envelope) != {"ok", "error"}:
            raise ValueError("Неверный формат ошибки приложения.")
        error = ApplicationError.model_validate(envelope["error"])
        raise AgentCallError(ERROR_STATUS[error.kind], error.message)
    if set(envelope) != {"ok", "result"}:
        raise ValueError("Неверный формат успешного ответа.")
    model = SearchResponse if operation == "search" else AskResponse
    value = envelope["result"]
    if not isinstance(value, dict) or set(value) != set(model.model_fields):
        raise ValueError("Поля результата не соответствуют пользовательской схеме.")
    value = dict(value)
    if operation == "search":
        value["results"] = [dict(item) for item in value["results"]]
        for item in value["results"]:
            rank = item.get("rank")
            if isinstance(rank, float) and rank.is_integer():
                item["rank"] = int(rank)
    result = model.model_validate(value, strict=True)
    sources = result.results if operation == "search" else result.sources
    if result.question != question or len(sources) != top_k:
        raise ValueError("В ответе другой вопрос или неверное количество источников.")
    if operation == "ask" and not result.answer.strip():
        raise ValueError("Агент вернул пустой answer.")
    if not math.isfinite(result.retrieval_seconds) or result.retrieval_seconds < 0:
        raise ValueError("Неверное время retrieval.")
    if operation == "ask" and (not math.isfinite(result.generation_seconds) or result.generation_seconds < 0):
        raise ValueError("Неверное время генерации.")
    for i, source in enumerate(sources, 1):
        if not source.text or not source.chunk_id or not source.doc_id or not math.isfinite(source.score):
            raise ValueError("Неверные поля источника.")
        if operation == "search" and source.rank != i:
            raise ValueError("Неверный порядок результатов.")
    return result.model_dump()


def transport_error(exc, discovery=False):
    logger.error("Ошибка A2A: %s", exc, exc_info=exc)
    cause = exc
    while cause is not None:
        if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
            return AgentCallError(503, "RAG-агент недоступен. Проверьте его запуск и RAG_AGENT_URL.")
        if isinstance(cause, (httpx.TimeoutException, A2AClientTimeoutError)):
            return AgentCallError(503 if discovery else 504, "Истекло время ожидания RAG-агента.")
        if isinstance(cause, httpx.HTTPStatusError):
            return AgentCallError(503 if cause.response.status_code in (404, 502, 503, 504) else 502,
                                  "Ошибка HTTP-транспорта RAG-агента; см. лог сервера.")
        cause = cause.__cause__ or cause.__context__
    if isinstance(exc, (A2AError, ValueError, TypeError, KeyError, ParseError, ValidationError)):
        return AgentCallError(502, "Некорректный ответ или ошибка протокола A2A; см. лог сервера.")
    return AgentCallError(500, "Внутренняя ошибка A2A-клиента; см. лог сервера.")


class AgentConnection:
    def __init__(self, http_client=None):
        self.http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(A2A_TIMEOUT, connect=3, pool=3), trust_env=False)
        self.client = None
        self.card = None
        self.connect_lock = asyncio.Lock()

    async def connect(self):
        async with self.connect_lock:
            if self.client is None:
                try:
                    card = await A2ACardResolver(self.http, AGENT_URL).get_agent_card(http_kwargs={"timeout": 3})
                    if not any(i.protocol_binding == "JSONRPC" and i.protocol_version == PROTOCOL_VERSION
                               for i in card.supported_interfaces):
                        raise ValueError("В Agent Card нужен JSONRPC 1.0.")
                    if not {"search_knowledge", "answer_question"} <= {s.id for s in card.skills}:
                        raise ValueError("Agent Card не объявляет нужные skills.")
                    self.client = await create_client(card, client_config=ClientConfig(
                        streaming=False, polling=False, httpx_client=self.http,
                        supported_protocol_bindings=["JSONRPC"], accepted_output_modes=["application/json"]))
                    self.card = card
                except Exception as exc:
                    raise transport_error(exc, discovery=True) from exc
        return self.client

    async def exchange(self, data, trace=None):
        client = await self.connect()
        # SDK создаёт UUID messageId. Новые запросы не содержат taskId/contextId/истории.
        request = SendMessageRequest(message=new_data_message(data, media_type="application/json", role=Role.ROLE_USER))
        response = None
        try:
            async for event in client.send_message(request):
                if response is not None or not event.HasField("message") or event.message.role != Role.ROLE_AGENT:
                    raise ValueError("Ожидался один завершённый Message-ответ агента.")
                response = event.message
            if response is None:
                raise ValueError("Пустой ответ A2A.")
            envelope = message_data(response)
            if trace is not None:
                trace.update(request=MessageToDict(request), response=MessageToDict(response))
            return envelope
        except Exception as exc:
            # Не повторяем вычисление. Следующий независимый запрос заново получает Card.
            # SDK.close закрыл бы общий httpx, поэтому здесь сбрасываем только ссылку.
            self.client = self.card = None
            raise transport_error(exc) from exc

    async def call(self, operation, question, top_k=3):
        data = {"operation": operation, "question": question}
        if operation == "search":
            data["top_k"] = top_k
        envelope = await self.exchange(data)
        try:
            return validate_result(envelope, operation, question, top_k)
        except AgentCallError:
            raise
        except Exception as exc:
            raise transport_error(exc) from exc

    async def ready(self):
        await self.connect()
        try:
            response = await self.http.get(AGENT_URL + "/ready", timeout=3)
            response.raise_for_status()
            body = response.json()
            if body.get("ready") is not True:
                raise AgentCallError(503, "Зависимости RAG-агента не готовы.")
            return {"ready": True, "dependencies": {"rag_agent": "ok"}}
        except Exception as exc:
            self.client = self.card = None
            raise AgentCallError(503, "RAG-агент или его зависимости не готовы; см. лог сервера.") from exc

    async def close(self):
        if self.client is not None:
            await self.client.close()
        await self.http.aclose()
