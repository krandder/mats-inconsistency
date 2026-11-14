#!/usr/bin/env python3
"""Minimal stdio MCP server exposing the as-of tools with a fixed cutoff.

The cutoff comes only from $ASOF_T (set by the runner, not by the model); every
call goes through AsOf.call_tool, which enforces it in SQL.  Each call and its
result are appended to $ASOF_LOG (JSONL) so the runner can record what the agent saw.

Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout (MCP stdio).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.asof import DEFAULT_STORE, TOOL_SCHEMAS, AsOf, iso, to_ms  # noqa: E402

MAX_RESULT_CHARS = 60_000


def tools_list() -> list[dict]:
    return [{"name": s["function"]["name"], "description": s["function"]["description"],
             "inputSchema": s["function"]["parameters"]} for s in TOOL_SCHEMAS]


def main() -> None:
    t = to_ms(os.environ["ASOF_T"])
    store = AsOf(os.environ.get("ASOF_STORE") or DEFAULT_STORE)
    log_path = os.environ.get("ASOF_LOG")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # notification
            continue
        if method == "initialize":
            ver = (msg.get("params") or {}).get("protocolVersion") or "2025-06-18"
            result = {"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "manifold-asof", "version": "1"},
                      "instructions": f"Manifold data as it stood just before {iso(t, minutes=True)}. "
                                      "Nothing at or after that time is available."}
        elif method == "tools/list":
            result = {"tools": tools_list()}
        elif method == "tools/call":
            params = msg.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            try:
                out = store.call_tool(name, args, t)
                text = json.dumps(out, ensure_ascii=False, default=str)
                if len(text) > MAX_RESULT_CHARS:
                    text = text[:MAX_RESULT_CHARS] + " …[truncated]"
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            except Exception as exc:  # report to the model, keep serving
                out = {"error": f"{type(exc).__name__}: {exc}"}
                result = {"content": [{"type": "text", "text": json.dumps(out)}], "isError": True}
            if log_path:
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"time": time.time(), "tool": name, "args": args, "result": out},
                                        ensure_ascii=False, default=str) + "\n")
        elif method == "ping":
            result = {}
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                         "error": {"code": -32601, "message": f"unknown method {method}"}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
