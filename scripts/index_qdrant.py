"""Импорт готовой матрицы в серверный Qdrant без кодирования документов."""

import argparse
import hashlib
import json
import sys
import uuid

import numpy as np
from qdrant_client import models

from retrieval import (
    COLLECTION, DATA, METADATA_FIELDS, POINT_COUNT, VECTOR_SIZE,
    check_collection, connect_qdrant, load_manifest, qdrant_call, read_jsonl, require, sha256,
)

BATCH_SIZE = 64


def point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload(chunk):
    return {"page_content": chunk["text"],
            "metadata": {field: chunk[field] for field in METADATA_FIELDS}}


def load_ready_vectors():
    manifest = load_manifest()
    for name, metadata in manifest["files"].items():
        require(sha256(DATA / name) == metadata["sha256"], f"SHA-256 {name} отличается от embeddings_manifest.json.")
    chunks = read_jsonl(DATA / "chunks.jsonl")
    matrix = np.load(DATA / "embeddings.npy", allow_pickle=False)
    require(len(chunks) == POINT_COUNT and matrix.shape == (POINT_COUNT, VECTOR_SIZE), "Нужны 666 фрагментов и матрица [666, 384].")
    require(matrix.dtype == np.float32 and bool(np.isfinite(matrix).all()), "Матрица должна быть float32 без NaN/infinity.")
    require(bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4, rtol=0)), "Векторы должны быть нормализованы.")
    identifiers = "".join(chunk["chunk_id"] + "\n" for chunk in chunks)
    require(hashlib.sha256(identifiers.encode("utf-8")).hexdigest() == manifest["row_order"]["ordered_chunk_ids_sha256"],
            "Порядок фрагментов отличается от manifest.")
    order = [(chunk["doc_id"], chunk["chunk_index"]) for chunk in chunks]
    require(order == sorted(order), "Нарушен порядок doc_id/chunk_index.")
    require(len({point_id(c["chunk_id"]) for c in chunks}) == POINT_COUNT, "Повторяются UUID points.")
    documents = {d["doc_id"]: d for d in read_jsonl(DATA / "corpus.jsonl")}
    for chunk in chunks:
        document = documents[chunk["doc_id"]]
        require(document["text"][chunk["start_char"]:chunk["end_char"]] == chunk["text"], "Текст фрагмента отличается от корпуса.")
    print("OK: SHA-256, [666, 384], float32, нормы, UUID, порядок и исходные тексты проверены.")
    return chunks, matrix


def check_existing_ids(client, chunks):
    expected = {point_id(chunk["chunk_id"]) for chunk in chunks}
    offset = None
    while True:
        records, offset = qdrant_call(client.scroll, COLLECTION, limit=128, offset=offset,
                                      with_payload=False, with_vectors=False)
        require(all(str(record.id) in expected for record in records),
                "В коллекции есть посторонние point ID. Импорт отменён; существующие points не удаляются.")
        if offset is None:
            break


def verify_points(client, chunks, matrix):
    count = check_collection(client, POINT_COUNT)
    indices = [0, 1, POINT_COUNT // 3, 2 * POINT_COUNT // 3, POINT_COUNT - 1]
    expected = {point_id(chunks[i]["chunk_id"]): i for i in indices}
    records = qdrant_call(client.retrieve, COLLECTION, ids=list(expected), with_payload=True, with_vectors=True)
    require(len(records) == len(expected), "Не все контрольные points найдены.")
    max_difference = 0.0
    for record in records:
        index = expected[str(record.id)]
        require(record.payload == payload(chunks[index]), f"Payload не совпадает: {record.id}.")
        actual = np.asarray(record.vector, dtype=np.float32)
        require(actual.shape == (VECTOR_SIZE,) and bool(np.allclose(actual, matrix[index], atol=1e-6, rtol=1e-5)),
                f"Вектор point {record.id} отличается от исходной строки матрицы.")
        max_difference = max(max_difference, float(np.max(np.abs(actual - matrix[index]))))
    print(f"OK: {count} points; размерность 384, COSINE; тексты, метаданные и векторы пяти points совпадают "
          f"(максимальное отклонение {max_difference:.9g}).")


def run(verify_only):
    chunks, matrix = load_ready_vectors()
    client, server_version = connect_qdrant()
    print(f"Сервер Qdrant {server_version}; коллекция {COLLECTION}.")
    if verify_only:
        verify_points(client, chunks, matrix)
        return
    if not qdrant_call(client.collection_exists, COLLECTION):
        qdrant_call(client.create_collection, COLLECTION,
                    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
        print("Создана коллекция с одним dense-вектором 384/COSINE.")
    count_before = check_collection(client)
    check_existing_ids(client, chunks)
    print(f"До импорта: {count_before} points. Используется только готовая embeddings.npy.")
    for start in range(0, POINT_COUNT, BATCH_SIZE):
        end = min(start + BATCH_SIZE, POINT_COUNT)
        points = [models.PointStruct(id=point_id(chunks[i]["chunk_id"]), vector=matrix[i].tolist(),
                                     payload=payload(chunks[i])) for i in range(start, end)]
        qdrant_call(client.upsert, COLLECTION, points=points, wait=True)
        print(f"Upsert завершён: {end}/{POINT_COUNT}", flush=True)
    verify_points(client, chunks, matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Только проверка базы, без create/upsert.")
    args = parser.parse_args()
    try:
        run(args.verify_only)
    except Exception as exc:
        print(f"Ошибка индексации: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
