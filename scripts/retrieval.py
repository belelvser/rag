"""Общие настройки Qdrant и один CPU-поисковик на процесс."""

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CACHE = ROOT / ".cache" / "huggingface"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
COLLECTION = "mini_helpdesk"
VECTOR_SIZE = 384
POINT_COUNT = 666
METADATA_FIELDS = ("chunk_id", "doc_id", "title", "source_split", "chunk_index", "start_char", "end_char")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def load_manifest():
    manifest = json.loads((DATA / "embeddings_manifest.json").read_text(encoding="utf-8"))
    require(manifest["matrix"]["shape"] == [POINT_COUNT, VECTOR_SIZE], "В manifest нужна матрица [666, 384].")
    require(manifest["matrix"]["normalized"] is True, "Ожидались нормализованные embeddings.")
    return manifest


def qdrant_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise RuntimeError(
                f"Коллекция {COLLECTION} не найдена. Выполните scripts/index_qdrant.py."
            ) from exc
        raise RuntimeError(f"Qdrant вернул HTTP {exc.status_code}: {exc}") from exc
    except ResponseHandlingException as exc:
        raise RuntimeError(
            f"Qdrant недоступен по {QDRANT_URL}: {exc}. "
            "Проверьте Docker Desktop и docker compose up -d qdrant."
        ) from exc


def connect_qdrant():
    client = QdrantClient(url=QDRANT_URL, timeout=10, prefer_grpc=False, trust_env=False)
    info = qdrant_call(client.info)
    return client, info.version


def check_collection(client, expected_count=None):
    info = qdrant_call(client.get_collection, COLLECTION)
    vectors = info.config.params.vectors
    require(isinstance(vectors, models.VectorParams)
            and vectors.size == VECTOR_SIZE and vectors.distance == models.Distance.COSINE,
            f"Коллекция {COLLECTION} несовместима: нужен один unnamed dense-вектор 384 с COSINE. "
            "Коллекция не удаляется автоматически.")
    count = qdrant_call(client.count, COLLECTION, exact=True).count
    if expected_count is not None:
        require(count == expected_count,
                f"В коллекции {count} points, ожидалось {expected_count}. Проверьте импорт готовых векторов.")
    return count


def validate_search_input(question, top_k):
    require(isinstance(question, str) and bool(question.strip()), "Вопрос не должен быть пустым.")
    require(type(top_k) is int and top_k > 0, "top_k должен быть положительным целым числом.")


class Retriever:
    """Соединение, модель и vector store, переиспользуемые между вопросами."""

    def __init__(self):
        self.manifest = load_manifest()
        self.client, self.server_version = connect_qdrant()
        check_collection(self.client, POINT_COUNT)
        os.environ["HF_HOME"] = str(CACHE)
        os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
        os.environ["HF_ASSETS_CACHE"] = str(CACHE / "assets")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        require(torch.version.cuda is None, "Нужна CPU-сборка PyTorch.")
        started = time.perf_counter()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.manifest["model"]["id"], cache_folder=str(CACHE / "hub"),
            model_kwargs={"revision": self.manifest["model"]["revision"], "device": "cpu",
                          "local_files_only": True, "trust_remote_code": False,
                          "model_kwargs": {"use_safetensors": True}},
            encode_kwargs={"normalize_embeddings": True, "device": "cpu", "batch_size": 16,
                           "convert_to_numpy": True},
            show_progress=False, multi_process=False,
        )
        self.model_load_seconds = time.perf_counter() - started
        # В зафиксированной langchain-huggingface модель SentenceTransformer хранится в _client.
        # Используем её токенизатор, чтобы не загружать вторую копию модели.
        model = self.embeddings._client
        require(model.device.type == "cpu" and model.get_embedding_dimension() == VECTOR_SIZE,
                "Модель должна быть на CPU и выдавать векторы размерности 384.")
        self.tokenizer = model.tokenizer
        self.input_limit = min(int(model.max_seq_length), int(model.tokenizer.model_max_length),
                               int(model[0].auto_model.config.max_position_embeddings))
        require(self.input_limit == self.manifest["model"]["effective_input_limit"],
                "Фактический предел модели отличается от embeddings_manifest.json.")
        self.store = QdrantVectorStore(
            client=self.client, collection_name=COLLECTION, embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE, distance=models.Distance.COSINE,
            content_payload_key="page_content", metadata_payload_key="metadata",
            # Размерность/коллекцию проверили выше; исключаем встроенный encode dummy_text.
            validate_embeddings=False, validate_collection_config=False,
        )

    def validate_question(self, question):
        """Проверка до поиска; API использует её для ответа HTTP 422."""
        validate_search_input(question, 1)
        # HuggingFaceEmbeddings заменяет переносы строк пробелами перед embed_query.
        encoded_text = question.replace("\n", " ")
        length = len(self.tokenizer.encode(encoded_text, add_special_tokens=True, truncation=False, verbose=False))
        require(length <= self.input_limit,
                f"Вопрос содержит {length} токенов со специальными токенами, предел {self.input_limit}. "
                "Сократите вопрос; автоматического обрезания нет.")

    def search(self, question, top_k=5):
        validate_search_input(question, top_k)
        self.validate_question(question)
        matches = qdrant_call(self.store.similarity_search_with_score, question, k=top_k)
        results = []
        for rank, (document, score) in enumerate(matches, 1):
            metadata = document.metadata
            results.append({"rank": rank, "score": float(score), "chunk_id": metadata["chunk_id"],
                            "doc_id": metadata["doc_id"], "title": metadata["title"],
                            "start_char": metadata["start_char"], "end_char": metadata["end_char"],
                            "text": document.page_content})
        return results


@lru_cache(maxsize=1)
def get_retriever():
    """Одна копия модели и один vector store на процесс."""
    return Retriever()


def search(question, top_k=5):
    validate_search_input(question, top_k)
    return get_retriever().search(question, top_k)
