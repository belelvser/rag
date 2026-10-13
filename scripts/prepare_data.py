"""Подготовка учебного корпуса SQuAD v2 и независимых dev/test-наборов."""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "huggingface"
DATASET_ID = "rajpurkar/squad_v2"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DOCUMENT_COUNTS = {"train": 300, "validation": 200}
QUESTION_COUNTS = {"train": 20, "validation": 30}
FILENAMES = ("corpus.jsonl", "dev.jsonl", "test.jsonl")
DOCUMENT_FIELDS = {"doc_id", "title", "text", "source_split"}
QUESTION_FIELDS = {
    "question_id", "question", "answers", "answer_starts", "expected_doc_id", "source_split"
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def doc_id(context):
    # Ни strip(), ни другая нормализация контекста не применяются.
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_key(question):
    # Только ключ сравнения: исходный вопрос в JSONL остаётся неизменённым.
    return " ".join(question.split()).casefold()


def valid_answers(texts, starts, context):
    if not isinstance(texts, list) or not isinstance(starts, list):
        return False
    if not texts or len(texts) != len(starts):
        return False
    return all(
        nonempty_text(text)
        and type(start) is int
        and start >= 0
        and context[start:start + len(text)] == text
        for text, start in zip(texts, starts)
    )


def collect_split(rows, split):
    """Сначала учитываем исходные контексты, затем проверяем вопрос и разметку."""
    documents = {}
    titles_by_doc = defaultdict(set)
    questions = defaultdict(list)
    stats = Counter(rows=len(rows), empty_answers=0, invalid_annotation=0)
    seen_question_ids = set()
    for row in rows:
        context, title = row.get("context"), row.get("title")
        if not nonempty_text(context) or not nonempty_text(title):
            stats["invalid_annotation"] += 1
            continue
        identifier = doc_id(context)
        if identifier in documents:
            require(documents[identifier]["text"] == context, "Обнаружена коллизия SHA-256 контекстов.")
        else:
            documents[identifier] = {
                "doc_id": identifier, "title": title, "text": context, "source_split": split
            }
        titles_by_doc[identifier].add(title)
        answers = row.get("answers")
        if not isinstance(answers, dict):
            stats["invalid_annotation"] += 1
            continue
        texts, starts = answers.get("text"), answers.get("answer_start")
        if texts == [] and starts == []:
            stats["empty_answers"] += 1
            continue
        question_id, question = row.get("id"), row.get("question")
        if (
            not nonempty_text(question_id)
            or question_id in seen_question_ids
            or not nonempty_text(question)
            or not valid_answers(texts, starts, context)
        ):
            stats["invalid_annotation"] += 1
            continue
        seen_question_ids.add(question_id)
        questions[identifier].append({
            "question_id": question_id,
            "question": question,
            "answers": texts,
            "answer_starts": starts,
            "expected_doc_id": identifier,
            "source_split": split,
        })
    stats["unique_contexts"] = len(documents)
    stats["unique_titles"] = len(set().union(*titles_by_doc.values()))
    stats["valid_answerable_questions"] = sum(map(len, questions.values()))
    stats["contexts_with_multiple_titles"] = sum(len(titles) > 1 for titles in titles_by_doc.values())
    print(
        f"{split}: {stats['rows']} строк, {stats['unique_contexts']} уникальных контекстов; "
        f"пропущено пустых ответов: {stats['empty_answers']}, "
        f"некорректных записей: {stats['invalid_annotation']}.", flush=True
    )
    return documents, titles_by_doc, questions, dict(stats)


def prepare_samples(groups, seed):
    train_docs, train_titles, train_questions, _ = groups["train"]
    validation_docs, validation_titles, validation_questions, _ = groups["validation"]
    shared_docs = set(train_docs) & set(validation_docs)
    all_train_titles = set().union(*train_titles.values())
    all_validation_titles = set().union(*validation_titles.values())
    shared_titles = all_train_titles & all_validation_titles
    train_q = [q for group in train_questions.values() for q in group]
    validation_q = [q for group in validation_questions.values() for q in group]
    shared_ids = {q["question_id"] for q in train_q} & {q["question_id"] for q in validation_q}
    shared_questions = {question_key(q["question"]) for q in train_q} & {
        question_key(q["question"]) for q in validation_q
    }
    overlap = {
        "context_texts": len(shared_docs), "article_titles": len(shared_titles),
        "answerable_question_ids": len(shared_ids),
        "answerable_question_comparison_keys": len(shared_questions),
        "policy": "Exclude shared contexts/titles and shared question IDs/text keys on both sides.",
    }
    print(
        f"Пересечения исходных частей: контексты={len(shared_docs)}, "
        f"названия статей={len(shared_titles)}, ID вопросов={len(shared_ids)}, "
        f"тексты вопросов (ключ сравнения)={len(shared_questions)}.", flush=True
    )
    rng = random.Random(seed)
    corpus, selected_questions, selection_stats = [], {}, {}
    for split in ("train", "validation"):
        documents, titles_by_doc, questions, _ = groups[split]
        excluded_docs = {
            identifier for identifier, titles in titles_by_doc.items()
            if identifier in shared_docs or titles & shared_titles or len(titles) != 1
        }
        eligible_questions = {}
        excluded_shared_questions = 0
        for identifier, candidates in questions.items():
            filtered = [q for q in candidates if q["question_id"] not in shared_ids
                        and question_key(q["question"]) not in shared_questions]
            excluded_shared_questions += len(candidates) - len(filtered)
            if identifier not in excluded_docs and filtered:
                eligible_questions[identifier] = sorted(filtered, key=lambda q: q["question_id"])
        candidates = sorted(eligible_questions)
        doc_count, question_count = DOCUMENT_COUNTS[split], QUESTION_COUNTS[split]
        require(len(candidates) >= doc_count,
                f"{split}: нужно {doc_count} документов, после исключений доступно {len(candidates)}.")
        selected_docs = sorted(rng.sample(candidates, doc_count))
        require(len(selected_docs) >= question_count,
                f"{split}: недостаточно разных документов для {question_count} вопросов.")
        question_docs = sorted(rng.sample(selected_docs, question_count))
        corpus.extend(documents[identifier] for identifier in selected_docs)
        selected_questions[split] = sorted(
            (rng.choice(eligible_questions[identifier]) for identifier in question_docs),
            key=lambda q: q["question_id"],
        )
        selection_stats[split] = {
            "documents_excluded_for_overlap_or_ambiguous_title": len(excluded_docs),
            "questions_excluded_for_shared_id_or_text": excluded_shared_questions,
            "eligible_documents": len(candidates),
            "selected_documents": doc_count, "selected_questions": question_count,
        }
        print(f"{split}: доступно {len(candidates)} документов; выбрано {doc_count} документов "
              f"и {question_count} вопросов по одному на документ.", flush=True)
    return sorted(corpus, key=lambda d: d["doc_id"]), selected_questions["train"], selected_questions["validation"], overlap, selection_stats


def validate_samples(corpus, dev, test):
    """Проверяем все требования; не выводим содержимое test."""
    checks = {}

    def check(name, condition, message):
        require(condition, f"Нарушено требование: {message}.")
        checks[name] = True
        print(f"OK: {message}")

    check("exact_counts", (len(corpus), len(dev), len(test)) == (500, 20, 30),
          "ровно 500 документов, 20 dev-вопросов и 30 test-вопросов")
    check("document_schema", all(set(d) == DOCUMENT_FIELDS for d in corpus),
          "корпус содержит только doc_id, title, text, source_split")
    check("document_text_and_hash", all(
        nonempty_text(d["title"]) and nonempty_text(d["text"]) and d["doc_id"] == doc_id(d["text"])
        for d in corpus), "тексты и названия непустые; doc_id совпадает с SHA-256 текста")
    check("unique_documents", len({d["doc_id"] for d in corpus}) == len(corpus)
          and len({d["text"] for d in corpus}) == len(corpus),
          "doc_id уникальны, дубликатов текстов нет")
    check("document_split_counts", Counter(d["source_split"] for d in corpus) == DOCUMENT_COUNTS,
          "в корпусе 300 документов train и 200 validation")
    documents = {d["doc_id"]: d for d in corpus}
    questions = dev + test
    check("question_schema", all(set(q) == QUESTION_FIELDS for q in questions),
          "схема вопросов верна; копий полного контекста в вопросах нет")
    check("unique_question_ids", all(nonempty_text(q["question_id"]) for q in questions)
          and len({q["question_id"] for q in questions}) == len(questions),
          "все 50 question_id непустые и уникальные")
    check("nonempty_questions", all(nonempty_text(q["question"]) for q in questions),
          "все вопросы непустые")
    check("expected_documents_exist", all(q["expected_doc_id"] in documents for q in questions),
          "каждый expected_doc_id существует в корпусе")
    check("answer_alignment", all(valid_answers(q["answers"], q["answer_starts"],
          documents[q["expected_doc_id"]]["text"]) for q in questions),
          "ответы непустые, списки согласованы, все позиции точно совпадают с текстом")
    check("question_split_provenance", all(
        q["source_split"] == split and documents[q["expected_doc_id"]]["source_split"] == split
        for subset, split in ((dev, "train"), (test, "validation")) for q in subset),
          "dev происходит из train, test из validation; документы имеют ту же исходную часть")
    check("one_question_per_document", all(
        len({q["expected_doc_id"] for q in subset}) == len(subset) for subset in (dev, test)),
        "по одному вопросу на правильный документ в каждом наборе")
    check("dev_test_question_disjoint", not (
        {q["question_id"] for q in dev} & {q["question_id"] for q in test})
        and not ({question_key(q["question"]) for q in dev} & {question_key(q["question"]) for q in test}),
        "между dev и test нет общих ID и текстов вопросов")
    check("dev_test_document_disjoint", not (
        {q["expected_doc_id"] for q in dev} & {q["expected_doc_id"] for q in test}),
        "между dev и test нет общих правильных документов")
    check("dev_test_title_disjoint", not (
        {documents[q["expected_doc_id"]]["title"] for q in dev}
        & {documents[q["expected_doc_id"]]["title"] for q in test}),
        "между dev и test нет общих названий статей")
    check("corpus_split_title_disjoint", not (
        {d["title"] for d in corpus if d["source_split"] == "train"}
        & {d["title"] for d in corpus if d["source_split"] == "validation"}),
        "названия статей не пересекаются даже между полными группами 300/200 документов")
    return checks


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            require(bool(line.strip()), f"{path.name}:{line_number}: пустая строка в JSONL.")
            row = json.loads(line)
            require(isinstance(row, dict), f"{path.name}:{line_number}: нужен JSON-объект.")
            rows.append(row)
    return rows


def jsonl_bytes(rows):
    # Явные LF и UTF-8 дают одинаковые байты на Windows и Linux.
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")


def write_jsonl(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def read_manifest(output_dir):
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["source"]["dataset_id"] == DATASET_ID, "В папке находится manifest другого источника.")
    return manifest


def verify_files(output_dir, manifest):
    for filename in FILENAMES:
        require(file_sha256(output_dir / filename) == manifest["files"][filename]["sha256"],
                f"{filename}: SHA-256 должен совпадать с manifest.")
    checks = validate_samples(*(read_jsonl(output_dir / filename) for filename in FILENAMES))
    print("OK: контрольные суммы всех трёх JSONL совпадают с manifest.")
    return checks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="Полный 40-символьный commit SHA источника на Hugging Face.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--validate-only", action="store_true", help="Проверить сохранённые файлы без сети.")
    return parser.parse_args()


def run(args):
    output_dir = args.output_dir.resolve()
    previous = None
    if (output_dir / "manifest.json").exists():
        previous = read_manifest(output_dir)
        verify_files(output_dir, previous)
    elif any((output_dir / name).exists() for name in FILENAMES):
        raise ValueError("Найдены JSONL без manifest. Выберите другую --output-dir, чтобы не затереть чужие данные.")
    if args.validate_only:
        require(previous is not None, "Для --validate-only нужны готовые JSONL и manifest.json.")
        return
    revision = args.revision or (previous["source"]["revision"] if previous else None)
    if revision is not None:
        require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "--revision должна быть полным commit SHA, а не веткой main.")

    # Переменные кеша задаются до импорта библиотек Hugging Face.
    os.environ.setdefault("HF_HOME", str(CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import datasets
    from huggingface_hub import HfApi

    print("Определяем точную ревизию публичного источника…", flush=True)
    info = HfApi().dataset_info(DATASET_ID, revision=revision or "main")
    require(bool(info.sha and re.fullmatch(r"[0-9a-f]{40}", info.sha)), "Источник не вернул полный commit SHA.")
    require(revision is None or info.sha == revision, "Ревизия источника отличается от запрошенной.")
    revision = info.sha
    print(f"Источник: {DATASET_ID}; revision={revision}; datasets={datasets.__version__}", flush=True)
    print("Загружаем официальные train/validation через datasets…", flush=True)
    source = datasets.load_dataset(DATASET_ID, revision=revision, cache_dir=str(CACHE / "datasets"))
    require("train" in source and "validation" in source, "В источнике отсутствуют train или validation.")
    groups = {split: collect_split(source[split], split) for split in ("train", "validation")}
    corpus, dev, test, overlap, selection = prepare_samples(groups, args.seed)
    checks = validate_samples(corpus, dev, test)
    contents = {name: jsonl_bytes(rows) for name, rows in zip(FILENAMES, (corpus, dev, test))}
    if previous and previous["source"]["revision"] == revision and previous["selection"]["seed"] == args.seed:
        require(all(hashlib.sha256(contents[name]).hexdigest() == previous["files"][name]["sha256"]
                    for name in FILENAMES),
                "Повторная подготовка изменила SHA-256 при той же ревизии и seed. Старые файлы сохранены.")
        print("OK: повторный запуск дал те же SHA-256 всех трёх JSONL.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        write_jsonl(output_dir / filename, content)
    # Дополнительно проверяем именно прочитанные с диска данные.
    disk_checks = validate_samples(*(read_jsonl(output_dir / name) for name in FILENAMES))
    files = {
        name: {"sha256": file_sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size,
               "records": count}
        for name, count in zip(FILENAMES, (500, 20, 30))
    }
    card_data = info.card_data.to_dict() if info.card_data is not None else {}
    manifest = {
        "schema_version": 1,
        "source": {"dataset_id": DATASET_ID, "url": DATASET_URL, "revision": revision,
                   "revision_url": f"{DATASET_URL}/tree/{revision}",
                   "license_from_dataset_card": card_data.get("license")},
        "selection": {"seed": args.seed, "documents_by_source_split": DOCUMENT_COUNTS,
                      "questions_by_source_split": QUESTION_COUNTS,
                      "one_question_per_document": True,
                      "document_order": "full SHA-256 ascending before random.sample and serialization",
                      "question_order": "original question_id ascending before random.choice and serialization",
                      "question_comparison_key": "casefold and collapsed whitespace; stored question unchanged"},
        "environment": {"python": platform.python_version(), "datasets": datasets.__version__,
                        "dependencies": {name: version(name) for name in
                                         ("huggingface-hub", "pyarrow", "numpy", "pandas", "fsspec")}},
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "source_statistics": {split: groups[split][3] for split in groups},
        "source_overlaps": overlap,
        "selection_statistics": selection,
        "files": files,
        "checks": {"before_save": checks, "after_reading_saved_files": disk_checks},
        "reproducibility_note": "Seed alone is insufficient. Use the recorded source commit, script, "
                                "Python/datasets environment and compare output SHA-256 hashes.",
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    print(f"Данные и manifest сохранены: {output_dir}")
    for name, metadata in files.items():
        print(f"{name}: {metadata['records']} записей, {metadata['bytes']} байт, SHA-256={metadata['sha256']}")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Ошибка подготовки данных: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Проверьте входные файлы, зависимости и доступ к Hugging Face. Размеры подвыборок не уменьшаются.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
