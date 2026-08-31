"""Unified tool execution: classify failures, retry, degrade, structured traces."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ToolBudgetExceeded(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ToolLoopDetected(RuntimeError):
    def __init__(self, signature: str) -> None:
        super().__init__(f"duplicate tool call: {signature}")
        self.signature = signature


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(exc: BaseException) -> dict[str, Any]:
    msg = str(exc)
    name = type(exc).__name__.lower()
    low = msg.lower()

    if isinstance(exc, FuturesTimeout) or "timeout" in low or "timed out" in low:
        return {
            "error_type": "timeout",
            "message": msg,
            "retryable": True,
            "suggested_action": "retry",
        }
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return {
            "error_type": "rate_limit",
            "message": msg,
            "retryable": True,
            "suggested_action": "retry",
        }
    if any(x in low for x in ("connection", "temporarily", "unavailable", "502", "503", "504")):
        return {
            "error_type": "transient",
            "message": msg,
            "retryable": True,
            "suggested_action": "retry",
        }
    if any(x in low for x in ("permission", "unauthorized", "forbidden", "401", "403")):
        return {
            "error_type": "permission",
            "message": msg,
            "retryable": False,
            "suggested_action": "stop",
        }
    if any(x in low for x in ("invalid", "validation", "schema", "unsupported")):
        return {
            "error_type": "invalid_argument",
            "message": msg,
            "retryable": False,
            "suggested_action": "stop",
        }
    if "empty" in low or isinstance(exc, ToolLoopDetected):
        return {
            "error_type": "empty_or_loop",
            "message": msg,
            "retryable": False,
            "suggested_action": "rewrite",
        }
    return {
        "error_type": name or "error",
        "message": msg,
        "retryable": False,
        "suggested_action": "fallback",
    }


def action_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    blob = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def summarize_result(result: Any, *, max_len: int = 240) -> tuple[int, str]:
    if result is None:
        return 0, ""
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return (1 if result.strip() else 0), result[:max_len]
    else:
        data = result
    if isinstance(data, dict):
        for key in ("papers", "rag_hits", "results", "citing_papers", "memory", "snippets"):
            if isinstance(data.get(key), list):
                n = len(data[key])
                return n, json.dumps({key: n, "keys": list(data.keys())[:8]}, ensure_ascii=False)[:max_len]
        return 1, json.dumps({"keys": list(data.keys())[:8]}, ensure_ascii=False)[:max_len]
    if isinstance(data, list):
        return len(data), f"list(len={len(data)})"
    text = str(data)
    return (1 if text.strip() else 0), text[:max_len]


class ToolRuntime:
    """Wraps tool callables with budget, dedupe, retry, and traces."""

    def __init__(
        self,
        *,
        max_tool_calls: int = 24,
        tool_timeout_s: float = 30.0,
        max_retries: int = 2,
        max_empty_streak: int = 3,
        allowed_tools: Optional[set[str]] = None,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.tool_timeout_s = float(os.getenv("GAP_TOOL_TIMEOUT_S", str(tool_timeout_s)))
        self.max_retries = int(os.getenv("GAP_TOOL_MAX_RETRIES", str(max_retries)))
        self.max_empty_streak = int(os.getenv("GAP_MAX_EMPTY_STREAK", str(max_empty_streak)))
        self.allowed_tools = allowed_tools
        self.tool_call_count = 0
        self.empty_streak = 0
        self.visited_actions: list[str] = []
        self.tool_traces: list[dict[str, Any]] = []
        self.retry_counts: dict[str, int] = {}
        self.warnings: list[str] = []
        self._round_seen: set[str] = set()

    def begin_round(self) -> None:
        self._round_seen.clear()

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        fn: Callable[[], Any],
        *,
        tool_call_id: Optional[str] = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            err = {
                "error_type": "unauthorized_tool",
                "message": f"tool {tool_name} not in whitelist",
                "retryable": False,
                "suggested_action": "stop",
            }
            trace = self._base_trace(tool_name, arguments, tool_call_id)
            trace.update(
                {
                    "status": "rejected",
                    "finished_at": _utcnow(),
                    "error_type": err["error_type"],
                    "error_message": err["message"],
                }
            )
            self.tool_traces.append(trace)
            self.warnings.append(err["message"])
            raise PermissionError(err["message"])

        if self.tool_call_count >= self.max_tool_calls:
            raise ToolBudgetExceeded("BUDGET_EXCEEDED:max_tool_calls")

        sig = action_signature(tool_name, arguments or {})
        if sig in self._round_seen or sig in set(self.visited_actions):
            raise ToolLoopDetected(sig)
        self._round_seen.add(sig)

        last_err: Optional[dict[str, Any]] = None
        for attempt in range(self.max_retries + 1):
            self.tool_call_count += 1
            if self.tool_call_count > self.max_tool_calls:
                raise ToolBudgetExceeded("BUDGET_EXCEEDED:max_tool_calls")

            trace = self._base_trace(tool_name, arguments, tool_call_id)
            trace["retry_index"] = attempt
            started = time.time()
            try:
                result = self._call_with_timeout(fn)
                duration = int((time.time() - started) * 1000)
                count, summary = summarize_result(result)
                status = "success" if count > 0 else "empty"
                if status == "empty":
                    self.empty_streak += 1
                else:
                    self.empty_streak = 0
                trace.update(
                    {
                        "finished_at": _utcnow(),
                        "duration_ms": duration,
                        "status": status,
                        "result_count": count,
                        "result_summary": summary,
                        "result_reference": {"signature": sig, "count": count},
                    }
                )
                self.tool_traces.append(trace)
                self.visited_actions.append(sig)
                if self.empty_streak >= self.max_empty_streak:
                    self.warnings.append("consecutive empty tool results reached limit")
                return result, trace
            except ToolBudgetExceeded:
                raise
            except ToolLoopDetected:
                raise
            except Exception as exc:  # noqa: BLE001
                duration = int((time.time() - started) * 1000)
                classified = classify_error(exc)
                last_err = classified
                self.retry_counts[tool_name] = self.retry_counts.get(tool_name, 0) + 1
                trace.update(
                    {
                        "finished_at": _utcnow(),
                        "duration_ms": duration,
                        "status": "timeout" if classified["error_type"] == "timeout" else "error",
                        "error_type": classified["error_type"],
                        "error_message": classified["message"][:400],
                    }
                )
                self.tool_traces.append(trace)
                if not classified.get("retryable") or attempt >= self.max_retries:
                    raise
                sleep_s = (2**attempt) * 0.25 + random.random() * 0.2
                time.sleep(sleep_s)

        raise RuntimeError(last_err or {"error_type": "unknown", "message": "tool failed"})

    def _call_with_timeout(self, fn: Callable[[], Any]) -> Any:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn)
            return fut.result(timeout=self.tool_timeout_s)

    def _base_trace(
        self, tool_name: str, arguments: dict[str, Any], tool_call_id: Optional[str]
    ) -> dict[str, Any]:
        return {
            "trace_id": str(uuid4()),
            "tool_call_id": tool_call_id or "",
            "tool_name": tool_name,
            "arguments": arguments or {},
            "started_at": _utcnow(),
            "finished_at": "",
            "duration_ms": 0,
            "status": "error",
            "retry_index": 0,
            "result_count": 0,
            "result_summary": "",
            "result_reference": None,
            "error_type": None,
            "error_message": None,
        }

    def export_state_fields(self) -> dict[str, Any]:
        return {
            "tool_call_count": self.tool_call_count,
            "visited_actions": list(self.visited_actions),
            "tool_traces": list(self.tool_traces),
            "retry_counts": dict(self.retry_counts),
            "warnings": list(self.warnings),
        }
