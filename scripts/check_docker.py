"""HTTP-проверка с Windows; --internal выполняет прямой SDK search внутри Compose."""

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECTRE = "What Guinness record was set during the filming of Spectre?"
BOX = "What is the typical temperature range for a box cooker?"


def http(path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request("http://127.0.0.1:8000" + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=330)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read()
        value = {"path": path, "request": body, "status": response.status,
                 "seconds": time.perf_counter() - started}
        if "json" in response.headers.get("Content-Type", ""):
            value["body"] = json.loads(raw)
        else:
            value["response_bytes"] = len(raw)
    return value


async def internal():
    sys.path.insert(0, str(ROOT))
    from google.protobuf.json_format import MessageToDict
    from app.a2a_client import AGENT_URL, AgentConnection
    connection = AgentConnection()
    try:
        await connection.connect()
        trace = {}
        envelope = await connection.exchange({"operation": "search", "question": SPECTRE, "top_k": 3}, trace)
        from app.a2a_client import validate_result
        result = validate_result(envelope, "search", SPECTRE, 3)
        card = MessageToDict(connection.card)
        assert card["supportedInterfaces"][0]["url"] == "http://agent:9999/a2a"
        assert AGENT_URL == "http://agent:9999"
        print(json.dumps({"passed": True, "agent_url": AGENT_URL, "card": card,
                          "result": result, "standard_exchange": trace}, ensure_ascii=False))
    finally:
        await connection.close()


def run(phase):
    path = ROOT / "results" / "docker_check.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    cases = [("/health", None), ("/ready", None), ("/docs", None), ("/openapi.json", None),
             ("/search", {"question": SPECTRE})]
    if phase == "initial":
        cases += [("/ask", {"question": SPECTRE}), ("/ask", {"question": BOX})]
    report["status"] = "running"
    for endpoint, body in cases:
        value = http(endpoint, body)
        value.update(phase=phase, passed=value["status"] == 200)
        report["checks"].append(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(f"{phase} {endpoint}: HTTP {value['status']}, {value['seconds']:.3f} с", flush=True)
        if not value["passed"]:
            raise RuntimeError(f"Проверка {endpoint} не прошла; ответ сохранён.")
        if endpoint == "/ask":
            print("Реальный answer:", value["body"]["answer"], flush=True)
    report["status"] = "complete"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == report
    print(f"Проверки сохранены в {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--phase", choices=("initial", "restart", "recreated", "repeat"), default="initial")
    args = parser.parse_args()
    asyncio.run(internal()) if args.internal else run(args.phase)
