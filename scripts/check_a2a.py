"""SDK-клиент и небольшая HTTP-проверка. Старые api/dev-отчёты не меняются."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from a2a.client import A2ACardResolver
from a2a.helpers import new_text_message
from a2a.types import SendMessageRequest, Role
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict

from app.a2a_client import (AGENT_URL, A2A_TIMEOUT, SDK_VERSION, PROTOCOL_VERSION,
                          AgentConnection, validate_result)

API = "http://127.0.0.1:8000"
SPECTRE = "What Guinness record was set during the filming of Spectre?"
BOX = "What is the typical temperature range for a box cooker?"
PATH = ROOT / "results" / "a2a_check.json"


def save(report):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(PATH)


async def run(phase):
    report = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sdk_version": SDK_VERSION,
        "protocol_version": PROTOCOL_VERSION, "agent_url": AGENT_URL,
        "checks": [], "scope": "Real SDK/HTTP, independent public questions; no test/dev evaluation."}
    report["status"] = "running"
    wire = []
    previous_wire = report.get("wire_exchanges", [])

    async def capture_request(request):
        if request.method == "POST" and request.url.path == "/a2a":
            wire.append({"request": json.loads(request.content), "url": str(request.url)})

    async def capture_response(response):
        if response.request.method == "POST" and response.request.url.path == "/a2a":
            await response.aread()
            # Один клиент может иметь несколько запросов; сопоставление по JSON-RPC id.
            body = response.json()
            for item in wire:
                if item["request"].get("id") == body.get("id"):
                    item["response"] = body
                    item["http_status"] = response.status_code
                    break

    sdk_http = httpx.AsyncClient(timeout=httpx.Timeout(A2A_TIMEOUT, connect=3), trust_env=False,
                               event_hooks={"request": [capture_request], "response": [capture_response]})
    connection = AgentConnection(sdk_http)

    def record(name, value, passed):
        item = {"phase": phase, "name": name, "passed": bool(passed), **value}
        report["checks"].append(item)
        save(report)
        print(f"{name}: passed={passed}; status={value.get('status')}; seconds={value.get('seconds')}", flush=True)
        if not passed:
            raise RuntimeError(f"Проверка не прошла: {name}")
        return item

    async def http(client, path, data=None, base=API, timeout=330):
        started = time.perf_counter()
        response = await (client.get(base + path, timeout=timeout) if data is None
                          else client.post(base + path, json=data, timeout=timeout))
        value = {"status": response.status_code, "seconds": time.perf_counter() - started,
                 "request": data, "path": path}
        if "json" in response.headers.get("content-type", ""):
            value["body"] = response.json()
        else:
            value["response_bytes"] = len(response.content)
        return value

    async def direct(name, data, expected_ok=True, expected_kind=None):
        trace = {}
        started = time.perf_counter()
        envelope = await connection.exchange(data, trace)
        passed = envelope.get("ok") is expected_ok
        if expected_ok:
            validated = validate_result(envelope, data["operation"], data["question"].strip(), data.get("top_k", 3))
            passed = passed and bool(validated)
        else:
            passed = passed and envelope.get("error", {}).get("kind") == expected_kind
        return record(name, {"request": data, "body": envelope, "standard_exchange": trace,
                             "seconds": time.perf_counter() - started}, passed)

    async def check_protocol():
        await connection.connect()
        protocol_error = None
        try:
            async for _ in connection.client.send_message(SendMessageRequest(message=new_text_message(
                "Unsupported text request", media_type="text/plain", role=Role.ROLE_USER))):
                pass
        except A2AError as exc:
            protocol_error = {"type": type(exc).__name__, "message": str(exc)}
        record("standard_protocol_error", {"error": protocol_error},
               protocol_error is not None and protocol_error["type"] == "ContentTypeNotSupportedError")

    try:
        async with httpx.AsyncClient(trust_env=False) as public:
            if phase in ("before-agent", "agent-down"):
                for name, endpoint, payload, code in (
                    ("api_health_without_agent", "/health", None, 200),
                    ("api_docs_without_agent", "/docs", None, 200),
                    ("api_ready_without_agent", "/ready", None, 503),
                    ("api_search_without_agent", "/search", {"question": SPECTRE}, 503),
                    ("api_ask_without_agent", "/ask", {"question": SPECTRE}, 503),
                ):
                    value = await http(public, endpoint, payload, timeout=10)
                    record(name, value, value["status"] == code)
            elif phase == "protocol":
                await check_protocol()
            elif phase == "recovered":
                for name, endpoint, payload in (
                    ("api_ready_after_agent_restore", "/ready", None),
                    ("api_search_after_agent_restore", "/search", {"question": SPECTRE}),
                    ("api_health_after_agent_restore", "/health", None),
                ):
                    value = await http(public, endpoint, payload, timeout=10)
                    record(name, value, value["status"] == 200)
            else:
                await connection.connect()
                card = MessageToDict(connection.card)
                report["agent_card"] = card
                record("sdk_agent_card", {"body": card},
                       card["name"] == "Mini Helpdesk Knowledge Agent" and not connection.card.capabilities.streaming
                       and not connection.card.capabilities.push_notifications)
                # Сначала только поиск: сверка /api/ps до и после, не вызывает chat.
                before = await http(public, "/api/ps", base="http://127.0.0.1:11434", timeout=5)
                await direct("direct_search_spectre", {"operation": "search", "question": SPECTRE, "top_k": 3})
                after = await http(public, "/api/ps", base="http://127.0.0.1:11434", timeout=5)
                record("search_did_not_load_qwen", {"before": before["body"], "after": after["body"]},
                       before["body"]["models"] == [] and after["body"]["models"] == [])
                await direct("direct_ask_spectre", {"operation": "ask", "question": SPECTRE})
                for endpoint in ("/health", "/ready", "/docs", "/openapi.json"):
                    value = await http(public, endpoint, timeout=10)
                    record("api" + endpoint, value, value["status"] == 200)
                value = await http(public, "/search", {"question": SPECTRE})
                record("api_search_spectre", value, value["status"] == 200)

                # Общее ограничение: первый запрос через API, конкурирующие через SDK и API.
                first = asyncio.create_task(http(public, "/ask", {"question": SPECTRE}))
                await asyncio.sleep(0.15)
                for base in (API, AGENT_URL):
                    value = await http(public, "/health", base=base, timeout=5)
                    record("health_during_ask_" + ("api" if base == API else "agent"), value,
                           value["status"] == 200 and value["seconds"] < 2 and not first.done())
                refreshed = await A2ACardResolver(sdk_http, AGENT_URL).get_agent_card(http_kwargs={"timeout": 3})
                record("card_during_ask", {"body": MessageToDict(refreshed)}, not first.done())
                await direct("direct_busy_during_api_ask", {"operation": "ask", "question": BOX}, False, "busy")
                value = await http(public, "/ask", {"question": BOX}, timeout=5)
                record("api_busy_429", value, value["status"] == 429 and value["seconds"] < 2 and not first.done())
                value = await first
                record("api_ask_spectre", value, value["status"] == 200)
                value = await http(public, "/ask", {"question": BOX})
                record("api_ask_box_lock_released", value, value["status"] == 200)

                for name, endpoint, payload in (
                    ("empty_question", "/ask", {"question": "  "}),
                    ("missing_question", "/ask", {}),
                    ("invalid_top_k", "/search", {"question": SPECTRE, "top_k": 6}),
                    ("unknown_field", "/ask", {"question": SPECTRE, "top_k": 3}),
                    ("over_token_limit", "/ask", {"question": "word " * 300}),
                    ("over_char_limit", "/search", {"question": "x" * 2001}),
                ):
                    value = await http(public, endpoint, payload, timeout=10)
                    record(name, value, value["status"] == 422)
                await direct("direct_empty_question", {"operation": "ask", "question": ""}, False, "invalid_input")
                await direct("direct_ask_forbids_top_k", {"operation": "ask", "question": SPECTRE, "top_k": 3}, False, "invalid_input")

                # Отмена ожидания реальным HTTP-клиентом; вычисление агента продолжается.
                started = time.perf_counter()
                try:
                    await public.post(API + "/ask", json={"question": BOX}, timeout=0.15)
                    disconnected = False
                except httpx.ReadTimeout:
                    disconnected = True
                record("real_client_disconnect", {"seconds": time.perf_counter() - started,
                       "scope": "Actual HTTP read timeout; result deliberately not received."}, disconnected)
                value = await http(public, "/ask", {"question": "word " * 300}, timeout=5)
                record("busy_after_client_disconnect", value, value["status"] == 429)
                # Некорректный вопрос не запустит ещё одну генерацию после освобождения lock.
                probes = []
                deadline = time.perf_counter() + 60
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.3)
                    value = await http(public, "/ask", {"question": "word " * 300}, timeout=5)
                    probes.append({"status": value["status"], "seconds": value["seconds"]})
                    if value["status"] != 429:
                        break
                record("lock_released_after_actual_completion", {"probes": probes},
                       bool(probes) and probes[-1]["status"] == 422)

                # Неверный стандартный тип content: ошибку протокола выдаёт SDK.
                await check_protocol()

                report["wire_exchanges"] = previous_wire + wire
                report["wire_note"] = "Actual SDK-generated JSON-RPC; random IDs, public corpus, no auth/private data. No handcrafted RPC."
        report["status"] = "complete" if all(c["passed"] for c in report["checks"]) else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if wire:
            report["wire_exchanges"] = previous_wire + wire
        await connection.close()
        save(report)
    assert json.loads(PATH.read_text(encoding="utf-8")) == report
    print(f"Фаза {phase}: {report['status']}; файл {PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("core", "before-agent", "agent-down", "recovered", "protocol"), default="core")
    asyncio.run(run(parser.parse_args().phase))
