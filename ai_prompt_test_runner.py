#!/usr/bin/env python3
"""Small, rate-limited runner for authorized AI chat test cases.

Input JSONL: one JSON object per line, e.g. {"id":"case-1","prompt":"..."}
Payload template: JSON object containing {{PROMPT}} where the prompt belongs.
The runner records HTTP status, latency, and response text for human review.
It does not decide whether an AI response is safe or vulnerable.

Use only on a system you own or are explicitly authorized to test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any


def replace_prompt(value: Any, prompt: str) -> Any:
    """Recursively substitute {{PROMPT}} in a JSON-compatible template."""
    if isinstance(value, str):
        return value.replace("{{PROMPT}}", prompt)
    if isinstance(value, list):
        return [replace_prompt(item, prompt) for item in value]
    if isinstance(value, dict):
        return {key: replace_prompt(item, prompt) for key, item in value.items()}
    return value


def load_cases(path: str, max_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(case, dict) or not isinstance(case.get("prompt"), str):
                raise ValueError(f"Line {line_number} must be an object with a string 'prompt' field")
            cases.append(case)
            if len(cases) > max_cases:
                raise ValueError(f"Test file contains more than the allowed {max_cases} cases")
    if not cases:
        raise ValueError("No test cases found")
    return cases


def self_test() -> None:
    template = {"prompt": "{{PROMPT}}", "nested": ["prefix {{PROMPT}}"]}
    result = replace_prompt(template, "synthetic test")
    assert result["prompt"] == "synthetic test"
    assert result["nested"][0] == "prefix synthetic test"
    print("SELF-TEST PASSED: prompt-template substitution")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rate-limited AI endpoint test runner for authorized environments.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorized", action="store_true", help="Confirm that you have permission to test this endpoint.")
    parser.add_argument("--url")
    parser.add_argument("--cases", help="JSONL test cases file")
    parser.add_argument("--template", help="JSON payload template containing {{PROMPT}}")
    parser.add_argument("--output", default="ai_prompt_test_results.jsonl")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests; minimum 1 second.")
    parser.add_argument("--max-requests", type=int, default=10, help="Maximum 20 requests per run.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.authorized:
        parser.error("Pass --authorized only after confirming written permission to test the endpoint")
    if not args.url or not args.cases or not args.template:
        parser.error("--url, --cases, and --template are required")
    if args.delay < 1.0:
        parser.error("--delay must be at least 1 second")
    if not 1 <= args.max_requests <= 20:
        parser.error("--max-requests must be between 1 and 20")

    try:
        import requests
    except ImportError:
        print("Install HTTP support with: python -m pip install requests", file=sys.stderr)
        return 1

    try:
        with open(args.template, "r", encoding="utf-8") as handle:
            template = json.load(handle)
        if not isinstance(template, dict):
            raise ValueError("Payload template must be a JSON object")
        cases = load_cases(args.cases, args.max_requests)
    except Exception as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    headers = {"Content-Type": "application/json"}
    token = os.getenv("AI_TEST_BEARER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with requests.Session() as session, open(args.output, "w", encoding="utf-8") as output:
        for index, case in enumerate(cases):
            request_body = replace_prompt(template, case["prompt"])
            started = time.monotonic()
            record: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case.get("id", f"case-{index + 1}"),
                "prompt": case["prompt"],
            }
            try:
                response = session.post(args.url, headers=headers, json=request_body, timeout=30)
                record["http_status"] = response.status_code
                record["latency_seconds"] = round(time.monotonic() - started, 3)
                # Keep response text bounded; avoid accidentally collecting large/private responses.
                record["response_excerpt"] = response.text[:4000]
            except requests.RequestException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["latency_seconds"] = round(time.monotonic() - started, 3)

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index + 1}/{len(cases)}] {record['case_id']}: {record.get('http_status', record.get('error'))}")
            if index + 1 < len(cases):
                time.sleep(args.delay)

    print(f"Saved results to {args.output}. Review responses manually against your test policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
