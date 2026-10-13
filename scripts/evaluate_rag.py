"""Последовательное сравнение oracle/retrieved; только dev, без обучения."""

import argparse
import json
import re
import statistics
import string
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

from rag import (
    GENERATION_PARAMETERS, REFUSAL, SYSTEM_PROMPT, TOKENIZER_ID, TOKENIZER_REVISION,
    TOP_K, USER_TEMPLATE, error_message, get_rag,
)
from retrieval import COLLECTION, DATA, ROOT, read_jsonl, require, sha256

SLOW_CALL_SECONDS = 240


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(reference, prediction):
    gold, predicted = normalize_answer(reference).split(), normalize_answer(prediction).split()
    if not gold or not predicted:
        return float(gold == predicted)
    common = sum((Counter(gold) & Counter(predicted)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def answer_scores(answer, references):
    # Так же, как официальный SQuAD v2 evaluator: максимум по непустым после нормализации эталонам.
    references = [r for r in references if normalize_answer(r)] or [""]
    return {"em": max(int(normalize_answer(r) == normalize_answer(answer)) for r in references),
            "f1": max(token_f1(r, answer) for r in references)}


def span_coverage(question, sources):
    for source in sources:
        if source["doc_id"] != question["expected_doc_id"]:
            continue
        for answer, start in zip(question["answers"], question["answer_starts"]):
            end = start + len(answer)
            if source["start_char"] <= start and end <= source["end_char"]:
                relative = start - source["start_char"]
                if source["text"][relative:relative + len(answer)] == answer:
                    return {"present": True, "source_id": source["source_id"], "chunk_id": source["chunk_id"],
                            "answer_start_char": start, "answer_end_char": end}
    return {"present": False}


def docker_snapshot():
    snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "kind": "single sample, not peak"}
    try:
        result = subprocess.run(["docker", "stats", "--no-stream", "--format", "json"],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        require(result.returncode == 0, result.stderr.strip())
        snapshot["containers"] = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        for container in snapshot["containers"]:
            print(f"docker stats: {container['Name']}: {container['MemUsage']}, CPU {container['CPUPerc']}", flush=True)
    except Exception as exc:
        snapshot["error"] = str(exc)
        print(f"Не удалось снять docker stats: {exc}", flush=True)
    return snapshot


def summarize(rows):
    completed = [row for row in rows if row.get("oracle") is not None and row.get("retrieved") is not None]
    summary = {"completed_pairs": len(completed)}
    for mode in ("oracle", "retrieved"):
        calls = [row[mode] for row in rows if row.get(mode) is not None]
        scored = [row[mode] for row in completed]
        summary[mode] = {
            "calls": len(calls), "successful_calls": sum(c["status"] == "ok" for c in calls),
            "errors": sum(c["status"] in ("error", "context_error", "empty_answer") for c in calls),
            "truncated_answers": sum(c["truncated"] for c in calls),
            "empty_answers": sum(c["empty_answer"] for c in calls),
            "near_context_limit_calls": sum(c["near_context_limit"] for c in calls),
            "scored_questions": len(scored),
            "em_percent": 100 * statistics.mean(c["scores"]["em"] for c in scored) if scored else None,
            "f1_percent": 100 * statistics.mean(c["scores"]["f1"] for c in scored) if scored else None,
            "mean_generation_seconds": statistics.mean(c["generation_seconds"] for c in calls) if calls else None,
            "mean_retrieval_seconds": statistics.mean(c["retrieval_seconds"] for c in calls) if calls else None,
        }
    calls = [row[mode] for row in rows for mode in ("oracle", "retrieved") if row.get(mode) is not None]
    if calls:
        summary["first_call"] = {"question": calls[0]["question"], "generation_seconds": calls[0]["generation_seconds"],
                                 "ollama_metrics": calls[0]["ollama_metrics"]}
        summary["mean_generation_seconds_all_calls"] = statistics.mean(c["generation_seconds"] for c in calls)
        summary["mean_generation_seconds_after_first"] = statistics.mean(c["generation_seconds"] for c in calls[1:]) if len(calls) > 1 else None
        counts = [c["ollama_metrics"].get("prompt_eval_count") for c in calls]
        counts = [c for c in counts if isinstance(c, int)]
        summary["prompt_eval_count_max"] = max(counts) if counts else None
    return summary


def save_report(path, report):
    report["summary"] = summarize(report["questions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    temporary.replace(path)


def load_inputs(smoke):
    questions = read_jsonl(DATA / "dev.jsonl")
    require(len(questions) == 20, "Нужны 20 исходных dev-вопросов.")
    documents = {d["doc_id"]: d for d in read_jsonl(DATA / "corpus.jsonl")}
    for question in questions:
        document = documents[question["expected_doc_id"]]
        require(question["answers"] and len(question["answers"]) == len(question["answer_starts"]), "Неверные dev-ответы.")
        require(all(document["text"][start:start + len(answer)] == answer for answer, start in
                    zip(question["answers"], question["answer_starts"])), "Разметка dev не совпадает с корпусом.")
    if smoke:
        selected = []
        for fragment in ("Spectre", "box cooker", "ISCII"):
            matches = [q for q in questions if fragment in q["question"]]
            require(len(matches) == 1, f"Нужен ровно один исходный dev-вопрос с {fragment}.")
            selected.append(matches[0])
        questions = selected
    hashes = {name: sha256(DATA / name) for name in
              ("dev.jsonl", "corpus.jsonl", "chunks.jsonl", "embeddings.npy", "embeddings_manifest.json")}
    return questions, documents, hashes


def oracle_source(question, documents):
    document = documents[question["expected_doc_id"]]
    return {"chunk_id": None, "doc_id": document["doc_id"], "title": document["title"], "score": None,
            "start_char": 0, "end_char": len(document["text"]), "text": document["text"]}


def evaluate_call(rag, mode, question, documents):
    try:
        if mode == "oracle":
            # Полный документ; retrieval здесь не вызывается.
            result = rag.generate(question["question"], [oracle_source(question, documents)])
        else:
            result = rag.ask(question["question"], TOP_K)
    except Exception as exc:
        result = {"question": question["question"], "answer": "", "sources": [], "context": "",
                  "retrieval_seconds": 0.0, "generation_seconds": 0.0, "status": "error",
                  "error": error_message(exc), "truncated": False, "empty_answer": False,
                  "near_context_limit": False, "ollama_metrics": {}, "response_metadata": {}}
    result["scores"] = answer_scores(result["answer"], question["answers"])
    result["reference_span_in_context"] = span_coverage(question, result["sources"])
    return result


def run_evaluation(smoke, resume=False):
    questions, documents, hashes = load_inputs(smoke)
    path = ROOT / "results" / ("rag_smoke.json" if smoke else "rag_dev.json")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "scope": "three control questions" if smoke else "20 answerable dev questions",
        "expected_pairs": len(questions), "expected_llm_calls": 2 * len(questions),
        "generation_parameters": GENERATION_PARAMETERS, "top_k": TOP_K,
        "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "weights_downloaded": False},
        "evaluator_source": "https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py",
        "input_sha256": hashes,
        "versions": {name: version(name) for name in ("langchain-ollama", "ollama", "langchain-core",
                     "langchain-huggingface", "langchain-qdrant", "qdrant-client", "transformers", "torch")},
        "slow_call_stop_threshold_seconds": SLOW_CALL_SECONDS,
        "questions": [], "memory_snapshots": {"before_generation": docker_snapshot()},
    }
    if resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        require(previous["input_sha256"] == hashes and previous["generation_parameters"] == GENERATION_PARAMETERS
                and previous["system_prompt"] == SYSTEM_PROMPT and previous["user_template"] == USER_TEMPLATE,
                "Checkpoint не совпадает с текущими данными, prompt или параметрами; продолжение отменено.")
        report = previous
        report["status"] = "running"
        report.pop("termination_reason", None)
        report.setdefault("resumed_at_utc", []).append(datetime.now(timezone.utc).isoformat())
        print(f"Продолжение checkpoint: сохранено {summarize(report['questions'])['completed_pairs']} пар.", flush=True)
    save_report(path, report)
    try:
        rag = get_rag()
        report["ollama_version"] = rag.ollama_version
        report["ollama_chat_template_sha256"] = rag.template_sha256
        for index, question in enumerate(questions, 1):
            row = next((r for r in report["questions"] if r["question_id"] == question["question_id"]), None)
            if row is None:
                row = {**question, "oracle": None, "retrieved": None}
                report["questions"].append(row)
            if resume and row["oracle"] is not None and row["retrieved"] is not None:
                continue
            print(f"\nПара {index}/{len(questions)}: {question['question']!r}", flush=True)
            print(f"Эталоны (только для оценки): {question['answers']}", flush=True)
            for mode in ("oracle", "retrieved"):
                if resume and row[mode] is not None:
                    continue
                print(f"  {mode}: начинаем независимый вызов…", flush=True)
                result = evaluate_call(rag, mode, question, documents)
                row[mode] = result
                save_report(path, report)
                print(f"  {mode}: answer={result['answer']!r}; статус={result['status']}; "
                      f"EM={result['scores']['em']}; F1={result['scores']['f1']:.3f}; "
                      f"generation={result['generation_seconds']:.3f} с; "
                      f"prompt_eval_count={result['ollama_metrics'].get('prompt_eval_count')}", flush=True)
                if result["error"]:
                    print(f"  Ошибка: {result['error']}", flush=True)
                if result["status"] in ("error", "context_error", "empty_answer"):
                    raise RuntimeError(f"Оценка остановлена на {mode}: {result['error'] or result['status']}. "
                                       "Полученные ответы сохранены.")
                if result["generation_seconds"] >= SLOW_CALL_SECONDS:
                    raise RuntimeError("Слишком длительный вызов; оценка остановлена. Проверьте память и журнал Ollama.")
            print("  Источники retrieved: " + "; ".join(
                f"{s['source_id']}: {s['title']} ({s['chunk_id']})" for s in row["retrieved"]["sources"]), flush=True)
            save_report(path, report)
        require(all(sha256(DATA / name) == digest for name, digest in hashes.items()), "Входные данные изменились.")
        report["status"] = "complete"
    except KeyboardInterrupt:
        report.update(status="stopped", termination_reason="Остановлено пользователем; полученные ответы сохранены.")
    except Exception as exc:
        report.update(status="stopped", termination_reason=error_message(exc))
    finally:
        report["memory_snapshots"]["after_generation"] = docker_snapshot()
        save_report(path, report)
    saved = json.loads(path.read_text(encoding="utf-8"))
    require(saved["summary"] == summarize(saved["questions"]), "Сводка сохранённого JSON не совпала с пересчётом.")
    print(f"\nОтчёт: {path}; статус {report['status']}")
    for mode in ("oracle", "retrieved"):
        values = report["summary"][mode]
        print(f"{mode}: EM={values['em_percent']}, F1={values['f1_percent']}; "
              f"успешных вызовов={values['successful_calls']}/{values['calls']}; "
              f"ошибок={values['errors']}, обрезаний={values['truncated_answers']}")
    if report.get("termination_reason"):
        print(report["termination_reason"])
    return 0 if report["status"] == "complete" else 1


def run_insufficient_context():
    # Отдельный пример: не входит в EM/F1 двадцати ответных вопросов.
    context = "The fictional project Orivexa-47 was launched in 2021. Its support email is help@orivexa.example."
    source = {"chunk_id": None, "doc_id": "fictional-orivexa-47", "title": "Orivexa-47 (fictional)",
              "score": None, "start_char": 0, "end_char": len(context), "text": context}
    result = get_rag().generate("Who founded Orivexa-47?", [source])
    report = {"scope": "single insufficient-context check, excluded from dev metrics",
              "generation_parameters": GENERATION_PARAMETERS, "system_prompt": SYSTEM_PROMPT,
              "expected_message": REFUSAL, "result": result,
              "matches_required_message": result["answer"].strip() == REFUSAL}
    path = ROOT / "results" / "rag_insufficient_context.json"
    path.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"Orivexa: {result['answer']!r}, статус={result['status']}")
    print(f"Сообщение совпало: {report['matches_required_message']}; отдельный отчёт: {path}")
    return 0 if result["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Сначала три точных dev-вопроса: Spectre, box cooker, ISCII.")
    group.add_argument("--insufficient-context", action="store_true", help="Отдельный пример Orivexa, без dev-метрик.")
    parser.add_argument("--resume", action="store_true", help="Продолжить checkpoint при тех же данных и настройках.")
    args = parser.parse_args()
    try:
        require(not (args.resume and args.insufficient_context), "--resume не применяется к отдельной проверке Orivexa.")
        return run_insufficient_context() if args.insufficient_context else run_evaluation(args.smoke, args.resume)
    except Exception as exc:
        print(f"Ошибка оценки RAG: {error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
