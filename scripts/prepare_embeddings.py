"""Точные фрагменты корпуса и локальные MiniLM embeddings на CPU, без обучения."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CACHE = ROOT / ".cache" / "huggingface"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30
BATCH_SIZE = 16
DIMENSION = 384
NORM_TOLERANCE = 1e-4
SEPARATORS = ["\n\n", "\n", " ", ""]
CHUNK_FIELDS = {
    "chunk_id", "doc_id", "title", "source_split", "chunk_index",
    "start_char", "end_char", "text", "token_count",
}

# Задаём пути до импорта Hugging Face: веса и служебные кеши остаются в проекте.
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_ASSETS_CACHE"] = str(CACHE / "assets")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            require(isinstance(row, dict), f"{path.name}:{number}: нужен JSON-объект.")
            rows.append(row)
    return rows


def load_corpus():
    rows = read_jsonl(DATA / "corpus.jsonl")
    require(len(rows) == 500, f"В корпусе должно быть 500 документов, получено {len(rows)}.")
    require(len({d["doc_id"] for d in rows}) == 500, "В корпусе повторяются doc_id.")
    for document in rows:
        text = document["text"]
        require(isinstance(text, str) and bool(text.strip()), "В корпусе есть пустой текст.")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == document["doc_id"],
                f"doc_id не совпадает с SHA-256 текста: {document['doc_id']}.")
    return sorted(rows, key=lambda d: d["doc_id"])


def load_model(local_only=False):
    import torch
    from sentence_transformers import SentenceTransformer

    require(torch.version.cuda is None, "Нужна CPU-сборка PyTorch, установлена сборка с CUDA.")
    print(f"Загружаем {MODEL_ID}, revision={MODEL_REVISION}, device=cpu…", flush=True)
    model = SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cpu", cache_folder=str(CACHE / "hub"),
        local_files_only=local_only, trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    require(model.device.type == "cpu", "Embedding-модель должна работать на CPU.")
    require(model.get_embedding_dimension() == DIMENSION, "Ожидалась размерность 384.")
    limits = {
        "sentence_transformer_max_seq_length": int(model.max_seq_length),
        "tokenizer_model_max_length": int(model.tokenizer.model_max_length),
        "transformer_max_position_embeddings": int(model[0].auto_model.config.max_position_embeddings),
    }
    limit = min(limits.values())
    require(limit > 0, "Не удалось определить предел входа модели.")
    print(f"Пределы входа: {limits}; используемый предел={limit}. Лимиты не меняются.", flush=True)
    return model, limits, limit


def token_count(tokenizer, text, special_tokens=True):
    # Явно без truncation: слишком длинный текст должен обнаруживаться проверкой.
    return len(tokenizer.encode(text, add_special_tokens=special_tokens, truncation=False))


def make_chunk_id(identifier, start, end, text):
    content = json.dumps([identifier, start, end, text], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_documents(corpus, tokenizer):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: token_count(tokenizer, text, special_tokens=False),
        separators=SEPARATORS, keep_separator="end", strip_whitespace=False,
        is_separator_regex=False,
    )
    chunks = []
    for document in corpus:
        original = document["text"]
        previous_start, previous_end = -1, 0
        pieces = splitter.split_text(original)
        require(bool(pieces), f"Документ остался без фрагментов: {document['doc_id']}.")
        for index, text in enumerate(pieces):
            # Ищем следующую точную подстроку, которая продолжает покрытие документа.
            # Перекрытие задано в токенах: вычитать 30 из символьной позиции нельзя.
            search_from = max(0, previous_start + 1, previous_end - len(text) + 1)
            start = original.find(text, search_from)
            end = start + len(text)
            require(start >= 0 and start <= previous_end and end > previous_end,
                    f"Не удалось восстановить непрерывные границы: {document['doc_id']}, chunk={index}.")
            require(original[start:end] == text, "Фрагмент не совпадает с исходным срезом.")
            chunks.append({
                "chunk_id": make_chunk_id(document["doc_id"], start, end, text),
                "doc_id": document["doc_id"], "title": document["title"],
                "source_split": document["source_split"], "chunk_index": index,
                "start_char": start, "end_char": end, "text": text,
                "token_count": token_count(tokenizer, text),
            })
            previous_start, previous_end = start, end
        require(previous_end == len(original), f"Потерян конец документа {document['doc_id']}.")
    print(f"Разбиение: {len(corpus)} документов → {len(chunks)} фрагментов.", flush=True)
    return chunks


def check_chunks(corpus, chunks, tokenizer, limit):
    require(bool(chunks), "Список фрагментов пуст.")
    require(len({c["chunk_id"] for c in chunks}) == len(chunks), "chunk_id должны быть уникальны.")
    documents = {d["doc_id"]: d for d in corpus}
    require({c["doc_id"] for c in chunks} == set(documents),
            "Фрагменты должны представлять все 500 документов без посторонних doc_id.")
    order = [(c["doc_id"], c["chunk_index"]) for c in chunks]
    require(order == sorted(order), "Нарушен порядок документов или chunk_index.")
    by_doc = defaultdict(list)
    for chunk in chunks:
        require(set(chunk) == CHUNK_FIELDS, "Нарушена схема chunks.jsonl.")
        document = documents[chunk["doc_id"]]
        start, end, text = chunk["start_char"], chunk["end_char"], chunk["text"]
        require(isinstance(text, str) and bool(text.strip()), f"Пустой фрагмент {chunk['chunk_id']}.")
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(document["text"]),
                f"Неверные границы {chunk['chunk_id']}.")
        require(document["text"][start:end] == text, f"Исходный срез не совпадает: {chunk['chunk_id']}.")
        require(chunk["chunk_id"] == make_chunk_id(chunk["doc_id"], start, end, text),
                "chunk_id не совпадает с SHA-256 документа, границ и текста.")
        require(chunk["title"] == document["title"] and chunk["source_split"] == document["source_split"],
                "Метаданные фрагмента отличаются от исходного документа.")
        actual_count = token_count(tokenizer, text)
        require(chunk["token_count"] == actual_count, "Сохранённый token_count неверен.")
        require(actual_count <= limit,
                f"Фрагмент {chunk['chunk_id']} содержит {actual_count} токенов, предел модели {limit}; "
                "encode не запускается, чтобы избежать truncation.")
        require(token_count(tokenizer, text, special_tokens=False) <= CHUNK_SIZE,
                f"Фактическая длина текста фрагмента превысила chunk_size={CHUNK_SIZE}.")
        by_doc[chunk["doc_id"]].append(chunk)
    for identifier, group in by_doc.items():
        require([c["chunk_index"] for c in group] == list(range(len(group))), "chunk_index должны идти от 0 подряд.")
        require(group[0]["start_char"] == 0 and group[-1]["end_char"] == len(documents[identifier]["text"]),
                "Фрагменты не покрывают начало или конец документа.")
        require(all(a["start_char"] < b["start_char"] <= a["end_char"] < b["end_char"]
                    for a, b in zip(group, group[1:])), "В покрытии документа есть пропуск или нарушение порядка.")
    print(f"OK: {len(chunks)} фрагментов; ID, 500 документов, границы, порядок и длины проверены.", flush=True)
    return {
        "unique_chunk_ids": True, "all_500_documents_represented": True,
        "exact_source_substrings": True, "stable_order_and_indexes": True,
        "full_document_character_coverage": True, "metadata_and_chunk_hashes": True,
        "token_counts_without_truncation": True, "within_model_limit": True,
        "within_content_chunk_size": True,
    }


def check_dev(corpus, chunks):
    documents = {d["doc_id"]: d for d in corpus}
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)
    questions = read_jsonl(DATA / "dev.jsonl")
    require(len(questions) == 20, "Ожидались 20 dev-вопросов.")
    coverage, missing = [], []
    for question in questions:
        identifier = question["expected_doc_id"]
        require(identifier in documents, "Правильный dev-документ отсутствует в корпусе.")
        answers, starts = question["answers"], question["answer_starts"]
        require(bool(answers) and len(answers) == len(starts), "Нарушена разметка dev-ответов.")
        match = None
        for answer, start in zip(answers, starts):
            end = start + len(answer)
            require(bool(answer.strip()) and start >= 0 and documents[identifier]["text"][start:end] == answer,
                    f"Неверный answer span dev-вопроса {question['question_id']}.")
            for chunk in by_doc[identifier]:
                if chunk["start_char"] <= start and end <= chunk["end_char"]:
                    match = {"chunk_id": chunk["chunk_id"], "answer_start_char": start, "answer_end_char": end}
                    break
            if match is not None:
                break
        coverage.append({"question_id": question["question_id"], "expected_doc_id": identifier,
                         "covered": match is not None, "matched_span": match})
        if match is None:
            missing.append({"question_id": question["question_id"], "question": question["question"],
                            "answers": answers, "answer_starts": starts,
                            "chunk_ranges": [[c["start_char"], c["end_char"]] for c in by_doc[identifier]]})
    print(f"Dev answer spans: покрыто {len(questions) - len(missing)}/{len(questions)} вопросов.", flush=True)
    require(not missing, "Ответ разрезан границами фрагментов: " + json.dumps(missing[:1], ensure_ascii=False)
            + ". Ни один фрагмент не содержит полный эталонный span; вопросы и ответы не изменяются.")
    return {"total_questions": 20, "covered_questions": 20, "all_covered": True,
            "checked_using_answer_positions": True, "details": coverage}


def check_matrix(matrix, count):
    import numpy as np

    require(matrix.shape == (count, DIMENSION), f"Неверная форма матрицы: {matrix.shape}, нужна {(count, DIMENSION)}.")
    require(matrix.dtype == np.float32, f"Нужен dtype float32, получен {matrix.dtype}.")
    require(bool(np.isfinite(matrix).all()), "Матрица содержит NaN или infinity.")
    norms = np.linalg.norm(matrix, axis=1)
    require(bool((norms > 0).all()), "Обнаружен нулевой вектор.")
    require(bool(np.allclose(norms, 1.0, atol=NORM_TOLERANCE, rtol=0)), "Нормы векторов отличаются от 1.")
    print(f"OK: матрица {matrix.shape}, float32; конечные ненулевые векторы; "
          f"нормы [{norms.min():.8f}, {norms.max():.8f}].", flush=True)
    return {"shape": True, "dtype_float32": True, "finite_values": True, "nonzero_vectors": True,
            "unit_norms": True, "norm_tolerance": NORM_TOLERANCE,
            "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def order_digest(chunks):
    identifiers = "".join(c["chunk_id"] + "\n" for c in chunks)
    return hashlib.sha256(identifiers.encode("utf-8")).hexdigest()


def show_examples(chunks, matrix, tokenizer):
    for first, second in zip(chunks, chunks[1:]):
        if first["doc_id"] == second["doc_id"] and second["start_char"] < first["end_char"]:
            overlap_length = first["end_char"] - second["start_char"]
            overlap = second["text"][:overlap_length]
            require(first["text"][-overlap_length:] == overlap, "Тексты перекрытия не совпадают.")
            print(f"Соседние фрагменты документа {first['title']} ({first['doc_id']}):")
            print(f"  #{first['chunk_index']}: [{first['start_char']}, {first['end_char']}), "
                  f"{first['token_count']} токенов; хвост: {first['text'][-overlap_length - 80:]!r}")
            print(f"  #{second['chunk_index']}: [{second['start_char']}, {second['end_char']}), "
                  f"{second['token_count']} токенов; начало: {second['text'][:overlap_length + 80]!r}")
            print(f"  Перекрытие: {overlap_length} символов, "
                  f"{token_count(tokenizer, overlap, special_tokens=False)} токенов текста: {overlap!r}")
            break
    else:
        print("В текущем корпусе соседние фрагменты с ненулевым перекрытием не найдены.")
    print(f"Первые 8 чисел embedding строки 0 (chunk_id={chunks[0]['chunk_id']}): {matrix[0, :8].tolist()}")


def check_existing_artifacts():
    manifest_path = DATA / "embeddings_manifest.json"
    artifacts = ("chunks.jsonl", "embeddings.npy")
    if not manifest_path.exists():
        require(not any((DATA / name).exists() for name in artifacts),
                "Найдены артефакты без embeddings_manifest.json; они не будут перезаписаны.")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("corpus.jsonl", "dev.jsonl", *artifacts):
        require(sha256(DATA / name) == manifest["files"][name]["sha256"],
                f"{name} изменён относительно embeddings_manifest.json; перезапись отменена.")
    return manifest


def check_saved_files(corpus, model, limit, manifest=None):
    import numpy as np

    chunks = read_jsonl(DATA / "chunks.jsonl")
    matrix = np.load(DATA / "embeddings.npy", allow_pickle=False)
    chunk_checks = check_chunks(corpus, chunks, model.tokenizer, limit)
    matrix_checks = check_matrix(matrix, len(chunks))
    dev_coverage = check_dev(corpus, chunks)
    if manifest is not None:
        require(order_digest(chunks) == manifest["row_order"]["ordered_chunk_ids_sha256"],
                "Нарушен порядок chunk_id по manifest.")
        require(list(matrix.shape) == manifest["matrix"]["shape"], "Форма матрицы отличается от manifest.")
        require(limit == manifest["model"]["effective_input_limit"], "Предел модели отличается от manifest.")
        require(manifest["model"]["revision"] == MODEL_REVISION, "Ревизия модели отличается от manifest.")
    return chunks, matrix, chunk_checks, matrix_checks, dev_coverage


def write_bytes(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def run(validate_only):
    import numpy as np

    previous = check_existing_artifacts()
    require(not validate_only or previous is not None, "Для проверки нужны готовые артефакты и manifest.")
    # test.jsonl не читается: для проверки answer spans используется только dev.
    source_hashes = {name: sha256(DATA / name) for name in ("corpus.jsonl", "dev.jsonl", "manifest.json")}
    corpus = load_corpus()
    model, limits, limit = load_model(local_only=validate_only)
    if validate_only:
        print("Повторная проверка сохранённых файлов, без encode и сетевой загрузки модели:")
        chunks, matrix, _, _, _ = check_saved_files(corpus, model, limit, previous)
        show_examples(chunks, matrix, model.tokenizer)
        return

    chunks = split_documents(corpus, model.tokenizer)
    print("Проверки перед encode:")
    chunk_checks = check_chunks(corpus, chunks, model.tokenizer, limit)
    dev_coverage = check_dev(corpus, chunks)
    lengths = [chunk["token_count"] for chunk in chunks]
    length_statistics = {"min": min(lengths), "mean": sum(lengths) / len(lengths), "max": max(lengths),
                         "includes_special_tokens": True}
    print(f"Длины фрагментов: {length_statistics}")
    print(f"Считаем embeddings на CPU, batch_size={BATCH_SIZE}; заголовки не кодируются…", flush=True)
    started = time.perf_counter()
    matrix = model.encode(
        [chunk["text"] for chunk in chunks], batch_size=BATCH_SIZE, device="cpu",
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    )
    encode_seconds = time.perf_counter() - started
    matrix_checks = check_matrix(matrix, len(chunks))
    print(f"Расчёт embeddings занял {encode_seconds:.3f} с (без скачивания и разбиения).", flush=True)

    serialized = "".join(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n"
                         for chunk in chunks).encode("utf-8")
    write_bytes(DATA / "chunks.jsonl", serialized)
    temporary_matrix = DATA / "embeddings.npy.tmp"
    with temporary_matrix.open("wb") as target:
        np.save(target, matrix, allow_pickle=False)
    temporary_matrix.replace(DATA / "embeddings.npy")
    print("Повторное чтение сохранённых файлов:")
    saved_chunks, saved_matrix, saved_chunk_checks, saved_matrix_checks, saved_dev = check_saved_files(corpus, model, limit)
    require(saved_chunks == chunks and np.array_equal(saved_matrix, matrix),
            "Данные или порядок строк изменились после сохранения.")
    require(all(sha256(DATA / name) == digest for name, digest in source_hashes.items()),
            "Исходные файлы этапа 2 изменились во время обработки.")
    print("OK: данные и порядок строк после записи совпадают; исходные файлы не изменены.")
    files = {name: {"sha256": sha256(DATA / name), "bytes": (DATA / name).stat().st_size}
             for name in ("corpus.jsonl", "dev.jsonl", "chunks.jsonl", "embeddings.npy")}
    manifest = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
                  "device": "cpu", "input_limits": limits, "effective_input_limit": limit,
                  "special_tokens_per_text": model.tokenizer.num_special_tokens_to_add(pair=False)},
        "versions": {"python": platform.python_version(), **{name: version(name) for name in
                     ("sentence-transformers", "torch", "numpy", "langchain-text-splitters",
                      "transformers", "tokenizers", "huggingface-hub")}},
        "splitter": {"class": "RecursiveCharacterTextSplitter", "chunk_size": CHUNK_SIZE,
                     "chunk_overlap": CHUNK_OVERLAP, "length_unit": "model tokenizer content tokens",
                     "length_function_add_special_tokens": False, "separators": SEPARATORS,
                     "keep_separator": "end", "strip_whitespace": False, "is_separator_regex": False,
                     "offset_method": "progressive exact substring search, verified slices and full coverage",
                     "token_count_add_special_tokens": True, "token_count_truncation": False},
        "encode": {"batch_size": BATCH_SIZE, "device": "cpu", "normalize_embeddings": True,
                   "convert_to_numpy": True, "show_progress_bar": True, "encoded_field": "text"},
        "document_count": len(corpus), "chunk_count": len(chunks),
        "chunk_token_statistics": length_statistics,
        "matrix": {"shape": list(matrix.shape), "dimension": DIMENSION, "dtype": str(matrix.dtype),
                   "normalized": True},
        "row_order": {"rule": "zero-based matrix row i corresponds to JSONL row i",
                      "sort": "doc_id ascending, then chunk_index ascending",
                      "ordered_chunk_ids_sha256": order_digest(chunks)},
        "files": files, "previous_stage_manifest_sha256": source_hashes["manifest.json"],
        "preparation_script_sha256": sha256(Path(__file__)), "encode_seconds": encode_seconds,
        "checks": {"before_encode": chunk_checks, "matrix": matrix_checks,
                   "after_reading": {"chunks": saved_chunk_checks, "matrix": saved_matrix_checks,
                                     "equal_to_computed_data_and_order": True},
                   "source_files_unchanged": True, "dev_answer_spans": saved_dev},
    }
    write_bytes(DATA / "embeddings_manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for name in ("chunks.jsonl", "embeddings.npy", "embeddings_manifest.json"):
        print(f"{name}: {(DATA / name).stat().st_size} байт")
    show_examples(saved_chunks, saved_matrix, model.tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Проверить артефакты с моделью из кеша, без encode.")
    args = parser.parse_args()
    try:
        run(args.validate_only)
    except Exception as exc:
        print(f"Ошибка подготовки embeddings: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Проверьте входные данные, CPU-зависимости и кеш модели. Исходные данные не исправляются автоматически.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
