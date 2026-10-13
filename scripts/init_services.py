"""Одноразовая подготовка: кеш моделей и импорт уже рассчитанных векторов."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from index_qdrant import load_ready_vectors, run as index_vectors
from retrieval import CACHE, QDRANT_URL, load_manifest
from rag import MODEL, OLLAMA_URL, TOKENIZER_ID, TOKENIZER_REVISION, get_rag

WAIT_SECONDS = 180


def request(url, data=None, timeout=5):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    return urllib.request.urlopen(urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body else {}), timeout=timeout)


def wait_for_services():
    pending = {"Qdrant": QDRANT_URL + "/", "Ollama": OLLAMA_URL + "/api/version"}
    deadline = time.monotonic() + WAIT_SECONDS
    while pending and time.monotonic() < deadline:
        for name, url in list(pending.items()):
            try:
                with request(url, timeout=3) as response:
                    response.read()
                print(f"Доступен {name} по {url}", flush=True)
                del pending[name]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Ожидаем {name}: {exc}", flush=True)
        if pending:
            time.sleep(2)
    if pending:
        raise RuntimeError(f"Сервисы не готовы за {WAIT_SECONDS} с: {', '.join(pending)}")


def ensure_qwen():
    with request(OLLAMA_URL + "/api/tags") as response:
        models = json.load(response)["models"]
    if any(model["name"] == MODEL for model in models):
        print(f"Qwen {MODEL} уже на диске: используем повторно, не загружаем в RAM.", flush=True)
        return
    print(f"Скачиваем веса {MODEL}. Это не обучение; скачивание может занять время.", flush=True)
    # 600 с — таймаут между данными HTTP-потока, не общий deadline скачивания.
    with request(OLLAMA_URL + "/api/pull", {"model": MODEL, "stream": True}, timeout=600) as response:
        previous = None
        for line in response:
            item = json.loads(line)
            if item.get("error"):
                raise RuntimeError(item["error"])
            total, completed = item.get("total", 0), item.get("completed", 0)
            progress = int(100 * completed / total) if total else None
            current = (item.get("status"), progress)
            if current != previous:
                print(f"Ollama pull: {current[0]}" + (f" {progress}%" if progress is not None else ""), flush=True)
                previous = current
    with request(OLLAMA_URL + "/api/tags") as response:
        if not any(m["name"] == MODEL for m in json.load(response)["models"]):
            raise RuntimeError("После pull выбранная Qwen не найдена.")


def prepare_cache():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model = load_manifest()["model"]
    print(f"Кеш MiniLM: {model['id']} revision={model['revision']}; {CACHE / 'hub'}", flush=True)
    def snapshot(repo, revision, allowed, required):
        try:
            path = snapshot_download(repo, revision=revision, cache_dir=str(CACHE / "hub"),
                                     allow_patterns=allowed, local_files_only=True)
            if all((Path(path) / name).is_file() for name in required):
                print(f"Готовый кеш {repo} используется без сетевого скачивания.", flush=True)
                return
        except LocalEntryNotFoundError:
            pass
        print(f"Подготавливаем отсутствующие файлы {repo}; без общего лимита времени скачивания.", flush=True)
        snapshot_download(repo, revision=revision, cache_dir=str(CACHE / "hub"), allow_patterns=allowed)

    mini_files = ["config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json",
                  "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.txt", "1_Pooling/config.json"]
    snapshot(model["id"], model["revision"], mini_files, mini_files)
    print(f"Кеш только токенизатора Qwen: {TOKENIZER_ID} revision={TOKENIZER_REVISION}", flush=True)
    qwen_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    snapshot(TOKENIZER_ID, TOKENIZER_REVISION,
             qwen_files + ["special_tokens_map.json", "vocab.json", "merges.txt", "chat_template*"], qwen_files)
    # Проверка загрузки из готового кеша; encode и генерации здесь нет.
    embedding = SentenceTransformer(model["id"], revision=model["revision"], cache_folder=str(CACHE / "hub"),
                                    device="cpu", local_files_only=True, trust_remote_code=False,
                                    model_kwargs={"use_safetensors": True})
    if embedding.get_embedding_dimension() != 384 or embedding.max_seq_length != model["effective_input_limit"]:
        raise RuntimeError("Кеш MiniLM несовместим с готовыми embeddings.")
    AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION,
                                  cache_dir=str(CACHE / "hub"), local_files_only=True, trust_remote_code=False)
    print("Кеш проверен: CPU MiniLM 384/256 и токенизатор Qwen; документов не кодировали.", flush=True)


def main():
    started = time.perf_counter()
    try:
        wait_for_services()
        print("Проверяем существующие артефакты до скачивания и записи в Qdrant.", flush=True)
        load_ready_vectors()
        ensure_qwen()
        prepare_cache()
        print("Импортируем существующую embeddings.npy через index_qdrant.py: UUID5/upsert без дубликатов.", flush=True)
        index_vectors(False)
        # Проверяет GGUF/template/tokenizer, но не вызывает chat и не загружает Qwen.
        rag = get_rag()
        rag.llm._client._client.close()
        print(f"INIT SUCCESS: 666 points, кеш и Qwen готовы; {time.perf_counter() - started:.3f} с.", flush=True)
        return 0
    except Exception as exc:
        print(f"INIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
