"""JSON-RPC A2A-сервер. Только этот процесс владеет MiniLM, RAG и ask-lock."""

import asyncio
import gc
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import httpx
from a2a.helpers import new_data_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.errors import UnsupportedOperationError
from ollama import ResponseError
from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.a2a_client import AGENT_PUBLIC_URL, PROTOCOL_VERSION, message_data, parse_input
from app.schemas import AskResponse, SearchResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rag import MODEL, OLLAMA_URL, get_rag
from retrieval import COLLECTION, POINT_COUNT, QDRANT_URL, VECTOR_SIZE, get_retriever

logger = logging.getLogger("uvicorn.error")


def failure(kind, message):
    return {"ok": False, "error": {"kind": kind, "message": message}}


def dependency_failure(exc):
    logger.error("Ошибка RAG-агента", exc_info=exc)
    cause = exc
    while cause is not None:
        if isinstance(cause, (httpx.ConnectError, httpx.TimeoutException, ConnectionError)):
            return failure("unavailable", "Qdrant или Ollama недоступны. Проверьте Docker-сервисы.")
        if isinstance(cause, (UnexpectedResponse, ResponseError)) and cause.status_code == 404:
            return failure("unavailable", "Нужная модель или коллекция отсутствует.")
        cause = cause.__cause__ or cause.__context__
    return failure("internal", "Внутренняя ошибка агента; причина записана в лог сервера.")


class KnowledgeExecutor(AgentExecutor):
    def __init__(self):
        self.retriever = self.rag = None
        self.lock = Lock()
        self.workers = set()
        self.ready_client = None

    def compute(self, payload):
        try:
            self.retriever.validate_question(payload.question)
        except ValueError as exc:
            return failure("invalid_input", str(exc))
        except Exception as exc:
            return dependency_failure(exc)
        try:
            if payload.operation == "search":
                started = time.perf_counter()
                results = self.retriever.search(payload.question, payload.top_k)
                result = SearchResponse(question=payload.question, results=results,
                                        retrieval_seconds=time.perf_counter() - started)
            else:
                raw = self.rag.ask(payload.question, top_k=3)
                if raw["status"] == "error":
                    logger.error("Ошибка генерации: %s", raw.get("error"))
                    return failure(raw.get("error_kind", "internal"), {
                        "timeout": "Ollama не завершила генерацию за 300 секунд.",
                        "unavailable": "Ollama недоступна или модель отсутствует.",
                    }.get(raw.get("error_kind"), "Внутренняя ошибка генерации; см. лог сервера."))
                if raw["status"] == "context_error" and raw["generation_seconds"] == 0:
                    return failure("invalid_input", raw["error"])
                if raw["status"] != "ok":
                    logger.error("Некорректный результат RAG: %r", raw)
                    return failure("internal", "Модель вернула неполный или некорректный ответ.")
                result = AskResponse.model_validate(raw)
            return {"ok": True, "result": result.model_dump()}
        except Exception as exc:
            return dependency_failure(exc)

    def compute_ask(self, payload):
        try:
            return self.compute(payload)
        finally:
            # Освобождение принадлежит реальному вычислению, не ожидающему клиенту.
            self.lock.release()
            logger.info("Генерация завершилась; ask-lock освобождена.")

    async def execute(self, context, event_queue):
        try:
            payload = parse_input(message_data(context.message))
        except (ValueError, ValidationError) as exc:
            logger.info("Неверный data-контракт: %s", exc)
            result = failure("invalid_input", "Неверные operation/question/top_k или формат data-part.")
        else:
            if payload.operation == "ask":
                if not self.lock.acquire(blocking=False):
                    result = failure("busy", "Генерация агента уже выполняется. Повторите запрос позже.")
                else:
                    work = asyncio.create_task(run_in_threadpool(self.compute_ask, payload))
                    self.workers.add(work)
                    work.add_done_callback(self.workers.discard)
                    # Отмена HTTP-ожидания не отменяет рабочий поток и не освобождает lock.
                    result = await asyncio.shield(work)
            else:
                result = await run_in_threadpool(self.compute, payload)
        await event_queue.enqueue_event(new_data_message(
            result, media_type="application/json", context_id=context.context_id))

    async def cancel(self, context, event_queue):
        raise UnsupportedOperationError("Отмена Task не поддерживается в Message-only обмене.")

    def readiness(self):
        dependencies = {}
        for name, url in (("ollama", OLLAMA_URL + "/api/tags"),
                          ("qdrant", f"{QDRANT_URL}/collections/{COLLECTION}")):
            try:
                response = self.ready_client.get(url)
                response.raise_for_status()
                body = response.json()
                if name == "ollama":
                    if not any(m["name"] == MODEL for m in body["models"]):
                        raise RuntimeError("Выбранная модель отсутствует.")
                else:
                    info = body["result"]
                    vectors = info["config"]["params"]["vectors"]
                    if (info["status"] != "green" or info["points_count"] != POINT_COUNT
                            or vectors.get("size") != VECTOR_SIZE or vectors.get("distance") != "Cosine"):
                        raise RuntimeError("Коллекция должна содержать 666 points 384/COSINE и быть green.")
                dependencies[name] = "ok"
            except Exception as exc:
                logger.warning("Agent ready: %s", name, exc_info=exc)
                dependencies[name] = "Недоступна или несовместима зависимость; см. лог агента."
        return {"ready": all(v == "ok" for v in dependencies.values()), "dependencies": dependencies}


executor = KnowledgeExecutor()
card = AgentCard(
    name="Mini Helpdesk Knowledge Agent", version="0.1.0",
    description="CPU-поиск по учебному корпусу и локальные ответы Qwen по top-3. Независимые запросы без истории.",
    supported_interfaces=[AgentInterface(url=AGENT_PUBLIC_URL + "/a2a", protocol_binding="JSONRPC",
                                         protocol_version=PROTOCOL_VERSION)],
    default_input_modes=["application/json"], default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[
        AgentSkill(id="search_knowledge", name="Search knowledge", tags=["search", "rag"],
                   description='Data-part: operation="search", question, optional top_k=1..5. Поиск без генерации.',
                   input_modes=["application/json"], output_modes=["application/json"],
                   examples=['{"operation":"search","question":"What Guinness record was set during the filming of Spectre?","top_k":3}']),
        AgentSkill(id="answer_question", name="Answer question", tags=["answer", "rag"],
                   description='Data-part: operation="ask", question. Фиксированный top-3, одна активная генерация.',
                   input_modes=["application/json"], output_modes=["application/json"],
                   examples=['{"operation":"ask","question":"What is the typical temperature range for a box cooker?"}']),
    ],
)
handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card,
                                validate_input_modes=True)


@asynccontextmanager
async def lifespan(app):
    executor.ready_client = httpx.Client(timeout=2, trust_env=False)
    try:
        executor.retriever = await run_in_threadpool(get_retriever)
        executor.rag = await run_in_threadpool(get_rag)
        logger.info("AGENT pid=%s: MiniLM instances=1, CPU; retriever_cache=%s; Qwen generation at startup=0",
                    os.getpid(), get_retriever.cache_info())
        yield
    finally:
        if executor.workers:
            await asyncio.gather(*executor.workers, return_exceptions=True)
        # SDK 1.2.2 требует закрыть реестр/очереди обработки в lifespan.
        await handler.aclose()
        if executor.rag is not None:
            await run_in_threadpool(executor.rag.llm._client._client.close)
            await executor.rag.llm._async_client._client.aclose()
        if executor.retriever is not None:
            await run_in_threadpool(executor.retriever.client.close)
        executor.ready_client.close()
        executor.rag = executor.retriever = None
        get_rag.cache_clear()
        get_retriever.cache_clear()
        gc.collect()
        logger.info("Клиенты агента закрыты; Docker-сервисы не остановлены.")


async def health(request):
    return JSONResponse({"status": "ok"})


async def ready(request):
    body = await run_in_threadpool(executor.readiness)
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


app = Starlette(lifespan=lifespan, routes=[
    *create_agent_card_routes(card),
    *create_jsonrpc_routes(handler, rpc_url="/a2a", enable_v0_3_compat=False),
    Route("/health", health), Route("/ready", ready),
])
