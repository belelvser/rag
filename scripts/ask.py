"""Один локальный RAG-вопрос через retrieval и Qwen."""

import argparse
import sys

from rag import error_message, get_rag
from retrieval import validate_search_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        validate_search_input(args.question, args.top_k)
        result = get_rag().ask(args.question, args.top_k)
        print(f"Ответ модели: {result['answer']}")
        print(f"Статус: {result['status']}; обрезание: {result['truncated']}; пустой ответ: {result['empty_answer']}")
        print(f"Retrieval: {result['retrieval_seconds']:.3f} с; генерация: {result['generation_seconds']:.3f} с")
        print(f"Промпт Qwen: оценка {result['estimated_prompt_tokens']}, "
              f"prompt_eval_count={result['ollama_metrics'].get('prompt_eval_count')}/{2048}")
        print(f"Метрики Ollama: {result['ollama_metrics']}")
        if result["error"]:
            print(f"Ошибка: {result['error']}")
        for warning in result["warnings"]:
            print(f"Предупреждение: {warning}")
        print("Источники — переданные модели документы; они не доказывают автоматически каждое утверждение.")
        for source in result["sources"]:
            print(f"\n[{source['source_id']}] {source['title']} score={source['score']} "
                  f"doc_id={source['doc_id']} chunk_id={source['chunk_id']}")
            print(source["text"])
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        print(f"Ошибка RAG: {error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
