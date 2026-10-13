"""Небольшая реальная HTTP-проверка API; только стандартная библиотека."""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECTRE = "What Guinness record was set during the filming of Spectre?"
BOX_COOKER = "What is the typical temperature range for a box cooker?"


def request(base_url, path, payload=None, timeout=310):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    query = urllib.request.Request(base_url + path, data=data,
                                   headers={"Content-Type": "application/json"} if data else {})
    started = time.perf_counter()
    result = {"path": path, "method": query.get_method(), "request": payload}
    try:
        try:
            response = urllib.request.urlopen(query, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc  # HTTP 422/429 — измеряемый ответ, не сбой самого скрипта.
        with response:
            raw = response.read()
            result.update(status=response.code, content_type=response.headers.get("Content-Type"),
                          response_bytes=len(raw))
            if "application/json" in (result["content_type"] or ""):
                result["body"] = json.loads(raw)
            else:
                result["body_preview"] = raw.decode("utf-8", errors="replace")[:300]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.update(status=None, error=str(exc))
    result["seconds"] = time.perf_counter() - started
    return result


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(base_url):
    path = ROOT / "results" / "api_check.json"
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
              "status": "running", "checks": [], "note": "Real HTTP; model answers are not replaced with references."}

    def record(name, result, expected, extra_ok=True):
        result.update(name=name, expected_status=expected,
                      passed=result.get("status") == expected and bool(extra_ok))
        report["checks"].append(result)
        save(report, path)
        print(f"{name}: HTTP {result.get('status')}, {result['seconds']:.3f} с; passed={result['passed']}", flush=True)
        return result

    health = record("health", request(base_url, "/health", timeout=5), 200)
    if not health["passed"]:
        report["status"] = "failed"
        save(report, path)
        return 1
    record("ready", request(base_url, "/ready", timeout=10), 200)
    record("docs", request(base_url, "/docs", timeout=5), 200)
    specification = request(base_url, "/openapi.json", timeout=5)
    spec = specification.get("body", {})
    valid_openapi = str(spec.get("openapi", "")).startswith("3.") and all(
        p in spec.get("paths", {}) for p in ("/health", "/ready", "/search", "/ask"))
    errors = spec.get("paths", {}).get("/ask", {}).get("post", {}).get("responses", {})
    valid_openapi = valid_openapi and all(str(code) in errors for code in (422, 429, 503, 504, 500))
    record("openapi", specification, 200, valid_openapi)
    result = request(base_url, "/search", {"question": "  " + SPECTRE + "  "})
    body = result.get("body", {})
    record("search_spectre", result, 200,
           body.get("question") == SPECTRE and len(body.get("results", [])) == 3)

    invalid = [
        ("missing_question", "/ask", {}),
        ("empty_question", "/ask", {"question": ""}),
        ("whitespace_question", "/ask", {"question": "  \t\n "}),
        ("top_k_zero", "/search", {"question": SPECTRE, "top_k": 0}),
        ("top_k_six", "/search", {"question": SPECTRE, "top_k": 6}),
        ("top_k_string", "/search", {"question": SPECTRE, "top_k": "3"}),
        ("top_k_bool", "/search", {"question": SPECTRE, "top_k": True}),
        ("unknown_field", "/ask", {"question": SPECTRE, "top_k": 3}),
        ("over_character_limit", "/search", {"question": "x" * 2001}),
        ("over_token_limit_search", "/search", {"question": "word " * 300}),
        ("over_token_limit_ask", "/ask", {"question": "word " * 300}),
    ]
    for name, endpoint, payload in invalid:
        record(name, request(base_url, endpoint, payload, timeout=10), 422)

    # Один настоящий /ask в фоне. Главный поток проверяет /health и конкурирующий /ask.
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, base_url, "/ask", {"question": "  " + SPECTRE + "  "})
        time.sleep(0.2)
        active_before_health = not first.done()
        result = request(base_url, "/health", timeout=5)
        record("health_during_ask", result, 200, active_before_health and not first.done() and result["seconds"] < 2)
        active_before_second = not first.done()
        result = request(base_url, "/ask", {"question": BOX_COOKER}, timeout=5)
        record("second_ask_busy", result, 429, active_before_second and not first.done() and result["seconds"] < 2)
        record("ask_spectre", first.result(), 200)
    # HTTP 200 после первого вызова доказывает освобождение lock; это второй реальный ответ.
    record("ask_box_cooker_lock_released", request(base_url, "/ask", {"question": BOX_COOKER}), 200)
    report["status"] = "complete" if all(c["passed"] for c in report["checks"]) else "failed"
    save(report, path)
    # Повторно читаем JSON, убеждаемся в сохранности исходных HTTP-ответов.
    assert json.loads(path.read_text(encoding="utf-8")) == report
    print(f"Результат: {report['status']}; {len(report['checks'])} проверок. Файл: {path}")
    for c in report["checks"]:
        if c["name"] in ("ask_spectre", "ask_box_cooker_lock_released"):
            print(f"{c['name']}: {c.get('body', {}).get('answer')!r}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    raise SystemExit(run(args.base_url.rstrip("/")))
