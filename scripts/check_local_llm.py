"""Три независимых запроса к локальному Ollama без сторонних библиотек."""

import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
CONTEXT = (
    "The fictional project Orivexa-47 was launched in 2021. "
    "Its support email is help@orivexa.example."
)
SYSTEM_PROMPT = (
    "Answer only using the provided context. Answer briefly in English. "
    "If the context does not contain the requested information, explicitly say "
    "that the information is not available in the context. Do not guess or use "
    "outside knowledge."
)
OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 64}
KEEP_ALIVE = "5m"
TIMEOUT_SECONDS = 300
RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "step1_check.json"
QUESTIONS = (
    ("A", "In what year was Orivexa-47 launched?"),
    ("B", "In what year was Orivexa-47 launched?"),
    ("C", "Who founded Orivexa-47?"),
)


def request_chat(question):
    # Новый список сообщений для каждого запроса: общей истории нет.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{CONTEXT}\n\nQuestion: {question}"},
        ],
        "stream": False,
        "options": OPTIONS,
        "keep_alive": KEEP_ALIVE,
    }
    request = Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Локальный запрос не должен уходить через системный HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "model" in detail.lower() and "not found" in detail.lower():
            raise RuntimeError(
                f"Модель {MODEL} отсутствует. Выполните: "
                f"docker compose exec ollama ollama pull {MODEL}. Ответ: {detail}"
            ) from exc
        raise RuntimeError(f"Ollama вернула HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Таймаут HTTP-запроса ({TIMEOUT_SECONDS} с). "
            "Проверьте docker compose logs --tail 50 ollama и доступную память."
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeError(f"Таймаут HTTP-запроса ({TIMEOUT_SECONDS} с).") from exc
        raise RuntimeError(
            f"Сервер Ollama недоступен по {URL}: {exc.reason}. "
            "Проверьте Docker Desktop, docker compose up -d и docker compose ps."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Ollama вернула некорректный JSON: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Ошибка соединения с локальным Ollama: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Ожидался JSON-объект в ответе Ollama.")
    if data.get("error"):
        raise RuntimeError(f"Ошибка Ollama: {data['error']}")
    message = data.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("В ответе Ollama отсутствует текст message.content.")
    if not message["content"].strip():
        raise RuntimeError("Ollama вернула пустой ответ. Проверьте журнал сервиса.")
    return data


def seconds_from_ns(value):
    if isinstance(value, (int, float)) and value >= 0:
        return value / 1_000_000_000
    return None


def extract_metrics(data):
    load_seconds = seconds_from_ns(data.get("load_duration"))
    eval_seconds = seconds_from_ns(data.get("eval_duration"))
    count = data.get("eval_count")
    speed = None
    if isinstance(count, int) and count >= 0 and eval_seconds is not None and eval_seconds > 0:
        speed = count / eval_seconds
    return {
        "load_duration_seconds": load_seconds,
        "eval_count": count,
        "eval_duration_seconds": eval_seconds,
        "tokens_per_second": speed,
    }


def format_number(value):
    return "недоступно" if value is None else f"{value:.3f}"


def save_report(report):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": URL,
        "model": MODEL,
        "context": CONTEXT,
        "system_prompt": SYSTEM_PROMPT,
        "options": OPTIONS,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "http_timeout_seconds": TIMEOUT_SECONDS,
        "calls": [],
    }
    for label, question in QUESTIONS:
        print(f"\n{label}. {question}", flush=True)
        started = time.perf_counter()
        call = {"label": label, "question": question}
        try:
            data = request_chat(question)
            call.update(
                status="ok",
                elapsed_seconds=time.perf_counter() - started,
                answer=data["message"]["content"],
                metrics=extract_metrics(data),
                raw_response=data,
            )
            print(f"Ответ модели: {call['answer']}")
            metrics = call["metrics"]
            print(f"load_duration: {format_number(metrics['load_duration_seconds'])} с")
            print(f"eval_count: {metrics['eval_count'] if metrics['eval_count'] is not None else 'недоступно'}")
            print(f"eval_duration: {format_number(metrics['eval_duration_seconds'])} с")
            print(f"Скорость генерации: {format_number(metrics['tokens_per_second'])} токенов/с")
        except RuntimeError as exc:
            call.update(status="error", elapsed_seconds=time.perf_counter() - started, error=str(exc))
            print(f"Ошибка: {exc}")
        print(f"Время HTTP-вызова: {call['elapsed_seconds']:.3f} с", flush=True)
        report["calls"].append(call)
        try:
            save_report(report)
        except OSError as exc:
            print(f"Не удалось сохранить {RESULT_PATH}: {exc}", file=sys.stderr)
            return 1
    print(f"\nОтветы и измерения сохранены: {RESULT_PATH}")
    return 0 if all(call["status"] == "ok" for call in report["calls"]) else 1


if __name__ == "__main__":
    sys.exit(main())
