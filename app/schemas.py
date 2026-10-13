"""Входные и выходные JSON-схемы API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXAMPLE = {"question": "What Guinness record was set during the filming of Spectre?"}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})
    question: str = Field(strict=True, min_length=1, max_length=2000,
                          description="Вопрос; пробелы по краям удаляются. Есть дополнительный предел токенов MiniLM.")

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class SearchRequest(AskRequest):
    top_k: int = Field(default=3, strict=True, ge=1, le=5)


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    score: float = Field(description="Сходство COSINE, не вероятность правильного ответа.")
    text: str = Field(description="Текст источника; сам по себе не является проверенным доказательством ответа.")


class SearchResult(Source):
    rank: int


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
    retrieval_seconds: float


class AskResponse(BaseModel):
    question: str
    answer: str = Field(description="Исходный ответ модели без исправлений.")
    sources: list[Source] = Field(description="Фрагменты, переданные модели. Не автоматическая проверка ответа.")
    retrieval_seconds: float
    generation_seconds: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    dependencies: dict[str, str]


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]]
