"""Ручной dense-поиск по существующей коллекции Qdrant."""

import argparse
import sys
import time

from retrieval import get_retriever, validate_search_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        validate_search_input(args.question, args.top_k)
        retriever = get_retriever()
        print(f"Первоначальная загрузка модели: {retriever.model_load_seconds:.3f} с")
        started = time.perf_counter()
        results = retriever.search(args.question, args.top_k)
        search_seconds = time.perf_counter() - started
        print(f"Поиск после загрузки модели: {search_seconds:.3f} с (проверка длины + embedding вопроса + Qdrant)")
        print("Score — исходное сходство COSINE, не вероятность правильного ответа.")
        for result in results:
            print(f"\n#{result['rank']} score={result['score']:.6f} | {result['title']}")
            print(f"doc_id={result['doc_id']}\nchunk_id={result['chunk_id']} "
                  f"[{result['start_char']}, {result['end_char']})")
            print(result["text"])
    except Exception as exc:
        print(f"Ошибка поиска: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
