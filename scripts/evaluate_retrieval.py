"""Оценка top-5 dense retrieval только на 20 dev-вопросах."""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

from retrieval import COLLECTION, DATA, ROOT, get_retriever, read_jsonl, require, sha256

RESULT_PATH = ROOT / "results" / "retrieval_dev.json"
TOP_K = 5


def matching_span(question, results, top_k):
    for result in results[:top_k]:
        if result["doc_id"] != question["expected_doc_id"]:
            continue
        for answer, start in zip(question["answers"], question["answer_starts"]):
            end = start + len(answer)
            if result["start_char"] <= start and end <= result["end_char"]:
                return {"rank": result["rank"], "chunk_id": result["chunk_id"], "answer": answer,
                        "answer_start_char": start, "answer_end_char": end}
    return None


def evaluate_question(question, results, seconds):
    # Не удаляем дубли doc_id: rank относится к строке выдачи, то есть к фрагменту.
    correct_ranks = [r["rank"] for r in results if r["doc_id"] == question["expected_doc_id"]]
    first_rank = min(correct_ranks) if correct_ranks else None
    spans = {str(k): matching_span(question, results, k) for k in (3, 5)}
    return {
        **question, "search_seconds": seconds, "results": results,
        "first_correct_document_rank": first_rank,
        "reciprocal_rank_at_5": 1 / first_rank if first_rank is not None else 0.0,
        "doc_hits": {str(k): first_rank is not None and first_rank <= k for k in (1, 3, 5)},
        "answer_span_coverage": {k: span is not None for k, span in spans.items()},
        "matched_answer_spans": spans,
    }


def summarize(records):
    total = len(records)
    summary = {}
    for k in (1, 3, 5):
        count = sum(row["doc_hits"][str(k)] for row in records)
        summary[f"DocHit@{k}"] = {"count": count, "total": total, "value": count / total}
    reciprocal_sum = sum(row["reciprocal_rank_at_5"] for row in records)
    summary["MRR@5"] = {"reciprocal_rank_sum": reciprocal_sum, "total": total,
                        "value": reciprocal_sum / total}
    for k in (3, 5):
        count = sum(row["answer_span_coverage"][str(k)] for row in records)
        summary[f"AnswerSpanCoverage@{k}"] = {"count": count, "total": total, "value": count / total}
    return summary


def load_dev_inputs(manifest):
    # Этот скрипт не открывает test.jsonl.
    input_names = ("corpus.jsonl", "chunks.jsonl", "embeddings.npy", "embeddings_manifest.json", "dev.jsonl")
    hashes = {name: sha256(DATA / name) for name in input_names}
    for name, metadata in manifest["files"].items():
        require(hashes[name] == metadata["sha256"], f"SHA-256 {name} отличается от embeddings_manifest.json.")
    questions = read_jsonl(DATA / "dev.jsonl")
    require(len(questions) == 20 and len({q["question_id"] for q in questions}) == 20,
            "Нужны ровно 20 dev-вопросов с уникальными ID.")
    documents = {d["doc_id"]: d for d in read_jsonl(DATA / "corpus.jsonl")}
    chunks = {c["chunk_id"]: c for c in read_jsonl(DATA / "chunks.jsonl")}
    for question in questions:
        require(bool(question["question"].strip()) and question["source_split"] == "train", "Некорректный dev-вопрос.")
        answers, starts = question["answers"], question["answer_starts"]
        require(bool(answers) and len(answers) == len(starts), "Некорректная разметка dev-ответов.")
        text = documents[question["expected_doc_id"]]["text"]
        require(all(bool(answer.strip()) and type(start) is int and start >= 0
                    and text[start:start + len(answer)] == answer for answer, start in zip(answers, starts)),
                f"Answer spans вопроса {question['question_id']} не совпадают с корпусом.")
    return questions, chunks, hashes


def check_returned_sources(results, chunks):
    require(len(results) == TOP_K, f"Ожидалось {TOP_K} результатов.")
    require([r["rank"] for r in results] == list(range(1, TOP_K + 1)), "Нарушены ранги фрагментов.")
    require(all(a["score"] >= b["score"] for a, b in zip(results, results[1:])), "Выдача не упорядочена по score.")
    for result in results:
        require(result["chunk_id"] in chunks, "Qdrant вернул неизвестный chunk_id.")
        original = chunks[result["chunk_id"]]
        require(all(result[field] == original[field] for field in
                    ("doc_id", "title", "start_char", "end_char", "text")),
                "Содержимое найденного фрагмента отличается от исходного chunks.jsonl.")


def run():
    retriever = get_retriever()
    questions, chunks, hashes = load_dev_inputs(retriever.manifest)
    print(f"Первоначальная загрузка модели: {retriever.model_load_seconds:.3f} с")
    records = []
    for index, question in enumerate(questions, 1):
        started = time.perf_counter()
        results = retriever.search(question["question"], TOP_K)
        seconds = time.perf_counter() - started
        check_returned_sources(results, chunks)
        record = evaluate_question(question, results, seconds)
        records.append(record)
        print(f"Dev {index}/20: correct_doc_rank={record['first_correct_document_rank']}, "
              f"span@5={record['answer_span_coverage']['5']}, search={seconds:.4f} с", flush=True)
    summary = summarize(records)
    times = [row["search_seconds"] for row in records]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"id": retriever.manifest["model"]["id"], "revision": retriever.manifest["model"]["revision"],
                  "device": "cpu", "normalize_embeddings": True, "effective_input_limit": retriever.input_limit},
        "qdrant": {"server_version": retriever.server_version, "collection": COLLECTION,
                   "points": 666, "vector_size": 384, "distance": "COSINE"},
        "versions": {name: version(name) for name in
                     ("qdrant-client", "langchain-qdrant", "langchain-huggingface", "langchain-core",
                      "sentence-transformers", "torch", "numpy", "transformers", "huggingface-hub")},
        "input_sha256": hashes, "evaluation_split": "dev", "top_k": TOP_K,
        "ranking_unit": "chunk; repeated doc_id are not removed",
        "timings": {"model_load_seconds": retriever.model_load_seconds,
                    "search_mean_seconds": statistics.mean(times), "search_median_seconds": statistics.median(times),
                    "search_min_seconds": min(times), "search_max_seconds": max(times),
                    "search_total_seconds": sum(times),
                    "search_scope": "input validation + question embedding + Qdrant HTTP search; model already loaded"},
        "summary": summary, "questions": records,
    }
    require(all(sha256(DATA / name) == digest for name, digest in hashes.items()), "Входные данные изменились во время оценки.")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    require(saved == report and summarize(saved["questions"]) == saved["summary"], "Отчёт изменился при сохранении.")
    print("\nDocHit@k: число вопросов с правильным документом в первых k фрагментах / 20.")
    print("MRR@5: сумма 1/rank первого правильного фрагмента (нет в top-5 → 0) / 20.")
    print("AnswerSpanCoverage@k: число вопросов с полным эталонным span в правильном фрагменте среди первых k / 20.")
    for name, values in summary.items():
        if name == "MRR@5":
            print(f"{name}: {values['reciprocal_rank_sum']:.4f}/{values['total']} = {values['value']:.4f}")
        else:
            print(f"{name}: {values['count']}/{values['total']} = {100 * values['value']:.1f}%")
    print(f"Поиск без загрузки модели: среднее {statistics.mean(times):.4f} с, "
          f"минимум {min(times):.4f} с, максимум {max(times):.4f} с.")
    print(f"Подробные результаты сохранены и проверены: {RESULT_PATH}")


def main():
    try:
        run()
    except Exception as exc:
        print(f"Ошибка оценки retrieval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
