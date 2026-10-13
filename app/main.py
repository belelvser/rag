"""Пользовательский HTTP API. Поиск и генерацию выполняет только A2A-агент."""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.a2a_client import AgentCallError, AgentConnection
from app.schemas import (AskRequest, AskResponse, ErrorResponse, HealthResponse,
                         ReadyResponse, SearchRequest, SearchResponse)

logger = logging.getLogger("uvicorn.error")
ERRORS = {
    422: {"model": ErrorResponse, "description": "Неверный вопрос, токеновый предел или параметры."},
    503: {"model": ErrorResponse, "description": "Недоступный агент или его зависимость."},
    502: {"model": ErrorResponse, "description": "Неверный ответ агента или ошибка протокола A2A."},
    504: {"model": ErrorResponse, "description": "Таймаут вычисления или ожидания A2A."},
    500: {"model": ErrorResponse, "description": "Неожиданная внутренняя ошибка; причина в логе сервера."},
}


@asynccontextmanager
async def lifespan(app):
    app.state.agent = AgentConnection()
    logger.info("API pid=%s import audit: torch=%s sentence_transformers=%s rag=%s retrieval=%s",
                os.getpid(), "torch" in sys.modules, "sentence_transformers" in sys.modules,
                "rag" in sys.modules, "retrieval" in sys.modules)
    try:
        yield
    finally:
        await app.state.agent.close()
        logger.info("Async A2A/HTTP-клиент API закрыт.")


app = FastAPI(
    title="Учебный локальный RAG", version="0.2.0", lifespan=lifespan,
    description="Пользовательский HTTP API поверх отдельного A2A-агента. "
                "Источники — переданный модели контекст, не автоматически проверенные доказательства.",
)


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.error("Неожиданная ошибка %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка API; причина в логе сервера."})


async def call_agent(request, operation, question, top_k=3):
    try:
        return await request.app.state.agent.call(operation, question, top_k)
    except AgentCallError as exc:
        logger.warning("A2A HTTP %s: %s", exc.status, exc.message)
        headers = {"Retry-After": "5"} if exc.status == 429 else None
        raise HTTPException(exc.status, exc.message, headers=headers) from exc


@app.get("/health", response_model=HealthResponse, summary="Жив ли API",
         description="Быстрый ответ без обращения к агенту или моделям.")
async def health():
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}},
         summary="Готов ли RAG-агент", description="Agent Card через SDK и короткая проверка готовности агента. Без генерации.")
async def ready(request: Request):
    try:
        return await request.app.state.agent.ready()
    except AgentCallError as exc:
        logger.warning("Ready: %s", exc, exc_info=True)
        return JSONResponse(status_code=503, content={"ready": False, "dependencies": {"rag_agent": exc.message}})


@app.post("/search", response_model=SearchResponse, responses=ERRORS, summary="Поиск фрагментов",
          description="A2A operation=search, top_k 1–5. Без генерации; score — сходство.")
async def search(body: SearchRequest, request: Request):
    return await call_agent(request, "search", body.question, body.top_k)


@app.post("/ask", response_model=AskResponse,
          responses={**ERRORS, 429: {"model": ErrorResponse, "description": "Генерация занята в агенте."}},
          summary="Ответ по найденному контексту", description="A2A operation=ask, фиксированный top-3 и prompt. "
          "Контроль одной генерации — в агенте. Недостаток информации — нормальный ответ 200.")
async def ask(body: AskRequest, request: Request):
    return await call_agent(request, "ask", body.question)
