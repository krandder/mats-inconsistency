#!/usr/bin/env python3
"""Minimal stdio MCP server with one tool, exa_search, for forecasts made "as of" a past date.

The cutoff is enforced here, not in the prompt: every query goes through the MiniBench bot's strict
backtest path (mb/research.py): Exa endPublishedDate = cutoff - 36 h, then leak_filter(strict=True)
(no date, after cutoff, date-only stamp within 36 h, excluded domains, live pages, text naming a later
date) and verify_dates (an independent page or URL date; dropped if none or if published or modified after
the cutoff). Only results that pass both checks reach the model.

Every query is appended to --log (one JSON line: the call, the query, its filters, every returned result
with its date, title and highlights, and why each dropped one was dropped). At most --max-queries per process
(one process per forecast call); after that the tool answers with an error and no search.

Protocol: JSON-RPC 2.0, one message per line on stdin/stdout (initialize, notifications/initialized,
tools/list, tools/call, ping). EXA_API_KEY comes from the environment (or `secret get [removed]`).

Usage (from an --mcp-config): exa_mcp_server.py --cutoff 2026-08-01T00:00:00Z --call-id ID --log PATH
    [--max-queries 4] [--num 5] [--highlight-chars 1000] [--cost-path PATH]
"""
import argparse
import fcntl
import hashlib
import json
import sys
import time
from datetime import timedelta

sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/minibench/bot")
from mb import llm, paths, research  # noqa: E402

PROTOCOL = "2025-06-18"
TOOL = {
    "name": "exa_search",
    "description": ("Search the web (news, articles, reports) as of today. Only pages published before today are "
                    "returned; each result shows its publication date, site, title and short excerpts. "
                    "Write a specific search query. You can make a limited number of searches per question."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "days_back": {"type": "integer", "minimum": 1, "maximum": 3650,
                          "description": "Optional: only return pages published within this many days before today."},
        },
        "required": ["query"],
    },
}


def append_jsonl(path, rec):
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def slim(r, kept):
    out = {"url": r.get("url"), "title": r.get("title"), "publishedDate": r.get("publishedDate"),
           "author": r.get("author"), "highlights": r.get("highlights"), "kept": kept}
    if r.get("_drop"):
        out["drop"] = r["_drop"]
    if "_verify" in r:
        out["verify"] = r["_verify"]
    return out


class Server:
    def __init__(self, a):
        self.a = a
        self.cutoff = paths.parse_iso(a.cutoff)
        self.n = 0
        research.NUM_RESULTS = a.num
        research.HIGHLIGHT_CHARS = a.highlight_chars
        llm.BUDGET.cap = 1e9  # the run script enforces the experiment cap; this process only logs

    def search(self, args):
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: empty query.", True
        if self.n >= self.a.max_queries:
            append_jsonl(self.a.log, {"ts": paths.utc_iso(), "call_id": self.a.call_id, "cutoff": self.a.cutoff,
                                      "n": self.n + 1, "query": query, "refused": "max_queries"})
            return (f"Search limit reached ({self.a.max_queries} searches). Give your answer now from what you "
                    "have."), True
        self.n += 1
        days = args.get("days_back")
        start = None
        if isinstance(days, (int, float)) and days >= 1:
            start = self.cutoff - timedelta(days=int(days))
        t0 = time.time()
        rec = {"ts": paths.utc_iso(), "call_id": self.a.call_id, "cutoff": self.a.cutoff, "n": self.n,
               "query": query, "days_back": days}
        try:
            res, body = research.exa_search(query, self.cutoff, start=start, num=self.a.num, strict=True,
                                            qid=self.a.call_id, run_id="linked-pairs-20261001",
                                            cost_path=self.a.cost_path)
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", latency_s=round(time.time() - t0, 2))
            append_jsonl(self.a.log, rec)
            return "Search failed (temporary error). Try again or answer from what you have.", True
        kept, dropped = research.leak_filter(res, self.cutoff, strict=True)
        kept, vdrop = research.verify_dates(kept, self.cutoff)
        text = research.render(kept)
        rec.update(filters={k: v for k, v in body.items() if k not in ("query", "_cost_usd")},
                   exa_cost_usd=body.get("_cost_usd"), exa_request_id=None,
                   results=[slim(r, True) for r in kept] + [slim(r, False) for r in dropped + vdrop],
                   n_returned=len(res), n_kept=len(kept),
                   shown_sha256=hashlib.sha256(text.encode()).hexdigest(), shown_chars=len(text),
                   latency_s=round(time.time() - t0, 2))
        append_jsonl(self.a.log, rec)
        if not text:
            return "No results published before today passed the date checks. Try a different query.", False
        return text, False

    def handle(self, msg):
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            return None  # notification
        if method == "initialize":
            pv = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": pv, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "exa", "version": "1.0"}}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}}
        if method == "tools/call":
            p = msg.get("params") or {}
            if p.get("name") != "exa_search":
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
            text, is_err = self.search(p.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}],
                                                            "isError": is_err}}
        if method in ("resources/list", "prompts/list"):
            key = method.split("/")[0]
            return {"jsonrpc": "2.0", "id": mid, "result": {key: []}}
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cost-path", required=True)
    ap.add_argument("--max-queries", type=int, default=4)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--highlight-chars", type=int, default=1000)
    s = Server(ap.parse_args())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        out = s.handle(msg)
        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
