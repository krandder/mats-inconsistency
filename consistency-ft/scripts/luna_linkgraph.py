#!/usr/bin/env python3
"""Link graph over the consistency-ft events, proposed by GPT-6 Luna (Kelvin's 2026-09-27 design).

An external model says which events are linked; the graph only decides which joint questions a later step
asks Qwen3-32B.  No values are injected: Luna returns ids and a one-word relation type, nothing else, and
the parser drops anything else it writes.  Train, validation and test are linked separately (links never
cross a split).

Steps (all outputs under /home/agent/data/luna-linkgraph-20260927/<split>/):
  nodes     events = every binary market and multiple-choice option of the split's parents
            (jev-disjoint-v1-filtered-20260919), resolved YES or NO; text, open/close/resolution
            times from the as-of store.  Candidates per event: up to 40 same-split events whose open
            windows overlap by >= 1 day, ranked by BGE-small cosine plus a same-group bonus
            (at most 15 options of the same parent).  Needs torch+transformers (run with --python-ok).
  run       Luna sessions, N focus events per session, resumable (one JSON file per session).
  edges     merge proposals into undirected edges.
  stats     degree distribution, components, triangles, max clique, treewidth upper bound.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import asof  # noqa: E402

DATA = Path("/home/agent/data")
SPLITS_ROOT = DATA / "jev-disjoint-v1-filtered-20260919"
OUT = DATA / "luna-linkgraph-20260927"
SPLITS = ("train", "validation", "test")
CODEX = "/home/agent/.local/bin/codex-any"
MODEL = "gpt-6-luna"
EMB_MODEL, EMB_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DAY = 86_400_000
N_CAND, MAX_SIBLINGS, MIN_OVERLAP = 40, 15, DAY
GROUP_BONUS, GENERIC_GROUP_SHARE = 0.06, 0.10
MAX_LINKS = 5
TYPES = ("same-quantity", "same-event-different-date", "alternatives", "cause/effect", "related-topic")
TRANSIENT = ("429", "529", "503", "502", "500", "rate limit", "overloaded", "stream disconnected", "account_unavailable",
             "Service Unavailable", "timed out", "Too Many Requests", "capacity", "ECONNRESET", "connection")

RULE = ("You label which prediction-market questions are related, for a research dataset. Use no tools. "
        "Reply with the requested JSON only: ids and relation types, nothing else.")

PROMPT = """You are building a link graph between prediction-market questions for a research dataset.

For each FOCUS question below, pick up to {k} questions from ITS OWN candidate list that are most logically or
causally connected to it: the ones a careful forecaster would most want to consider together with it when
judging its chances. Pick fewer (or none) if fewer are genuinely connected.

For each link give exactly one relation type:
- same-quantity: both ask about one underlying number or measure (different thresholds, ranges or variants)
- same-event-different-date: the same event with a different deadline or date
- alternatives: competing options for the same slot (who wins, which one happens first, ...)
- cause/effect: one event could plausibly bring about or influence the other
- related-topic: same topic, actors or process, a weaker connection

Rules:
- Judge only from the question texts shown here. Do not use, mention or hint at how any event turned out.
- Output only ids and relation types: no probabilities, no outcomes, no direction of influence, no explanations.
- Use only ids that appear in that focus question's candidate list.

Output exactly one JSON object and nothing else:
{{"links": {{"<focus id>": [["<candidate id>", "<relation type>"], ...], ...}}}}
Include every focus id as a key (an empty list if nothing is connected).

QUESTION TABLE (id | closes | question)
{table}

FOCUS QUESTIONS
{focus}
"""


# --- nodes --------------------------------------------------------------------

_VERSION_AT = """SELECT question, description, close, group_slugs FROM market_versions
WHERE mid = ? AND valid_from < ? AND valid_to >= ? ORDER BY valid_from DESC LIMIT 1"""
_ANSWER_TEXT_AT = """SELECT text FROM answer_versions WHERE aid = ? AND valid_from < ? AND valid_to >= ?
ORDER BY valid_from DESC LIMIT 1"""


def parents_of(split: str) -> set[str]:
    return {json.loads(line)["parent_id"] for line in (SPLITS_ROOT / split / "propositions.jsonl").read_text().splitlines()
            if line.strip()}


def build_events(db, split: str, stats: Counter) -> list[dict]:
    manifest_ids = {json.loads(line)["id"] for line in (SPLITS_ROOT / split / "propositions.jsonl").read_text().splitlines()
                    if line.strip()}
    events = []
    for pid in sorted(parents_of(split)):
        m = db.execute("SELECT mid, id, created, close, outcome_type, resolution, resolution_time, visibility, url "
                       "FROM markets WHERE id = ?", (pid,)).fetchone()
        if m is None:
            stats["parent_missing"] += 1
            continue
        mid, cid, created, _close, otype, mres, mres_t, vis, url = m
        if otype == "BINARY":
            items = [(cid, None, created, mres, mres_t)]
        else:
            items = [(f"{cid}:{a[0]}", a[1], max(created, a[2]), a[3], a[4]) for a in
                     db.execute("SELECT id, aid, created, resolution, resolution_time FROM answers WHERE mid = ? "
                                "ORDER BY coalesce(idx, 1e9), created, id", (mid,)).fetchall()]
        for eid, aid, open_t, res, res_t in items:
            if res not in ("YES", "NO") or res_t is None:
                stats[f"drop_resolution_{res}"] += 1
                continue
            # text and close in force just before the event stopped being open-and-unresolved
            v = db.execute(_VERSION_AT, (mid, res_t, res_t)).fetchone()
            if v is None:
                stats["drop_no_version"] += 1
                continue
            close_t = v[2]
            end_t = min(x for x in (close_t, res_t) if x is not None)
            vt = db.execute(_VERSION_AT, (mid, end_t, end_t)).fetchone() or v
            answer = None
            if aid is not None:
                row = db.execute(_ANSWER_TEXT_AT, (aid, end_t, end_t)).fetchone() or \
                    db.execute(_ANSWER_TEXT_AT, (aid, res_t, res_t)).fetchone()
                answer = row[0] if row else db.execute("SELECT text FROM answers WHERE aid = ?", (aid,)).fetchone()[0]
            if end_t - open_t < DAY:
                stats["drop_open_under_1d"] += 1
                continue
            events.append({
                "id": eid, "parent_id": cid, "split": split, "outcome_type": otype,
                "question": vt[0], "answer_text": answer, "criteria_text": vt[1] or "",
                "groups": json.loads(vt[3]) if vt[3] else [],
                "open_time": open_t, "close_time": close_t, "resolution_time": res_t, "end_time": end_t,
                "open_date": asof.iso(open_t), "close_date": asof.iso(close_t),
                "resolution_date": asof.iso(res_t), "end_date": asof.iso(end_t),
                "resolution": res, "visibility": vis, "url": url, "in_pair_manifest": eid in manifest_ids,
            })
            stats["kept"] += 1
    return events


def title(e: dict, n: int = 110) -> str:
    t = asof.clip(e["question"], n)
    return t + (" :: " + asof.clip(e["answer_text"], 60) if e["answer_text"] is not None else "")


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL, revision=EMB_REV)
    model = AutoModel.from_pretrained(EMB_MODEL, revision=EMB_REV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            v = model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def candidates(events: list[dict], vec: np.ndarray) -> list[list[dict]]:
    n = len(events)
    sim = vec @ vec.T
    op = np.array([e["open_time"] for e in events])
    en = np.array([e["end_time"] for e in events])
    overlap = np.minimum(en[:, None], en[None, :]) - np.maximum(op[:, None], op[None, :])
    ok = overlap >= MIN_OVERLAP
    np.fill_diagonal(ok, False)
    gcount = Counter(g for e in events for g in set(e["groups"]))
    generic = {g for g, c in gcount.items() if c > GENERIC_GROUP_SHARE * n}
    gsets = [set(e["groups"]) - generic for e in events]
    out = []
    for i in range(n):
        idx = np.nonzero(ok[i])[0]
        score = sim[i, idx] + GROUP_BONUS * np.array([bool(gsets[i] & gsets[j]) for j in idx], dtype=np.float32)
        order = idx[np.argsort(-score, kind="stable")]
        sc = dict(zip(idx.tolist(), score.tolist()))
        picked, sib = [], 0
        for j in order.tolist():
            if events[j]["parent_id"] == events[i]["parent_id"]:
                if sib >= MAX_SIBLINGS:
                    continue
                sib += 1
            picked.append({"j": j, "score": round(sc[j], 4), "cos": round(float(sim[i, j]), 4),
                           "shared_group": bool(gsets[i] & gsets[j]), "overlap_days": round(overlap[i, j] / DAY, 1)})
            if len(picked) >= N_CAND:
                break
        out.append(picked)
    return out, sorted(generic)


def cmd_nodes(a) -> None:
    store = asof.AsOf()
    for split in a.splits:
        stats = Counter()
        events = build_events(store.db, split, stats)
        for k, e in enumerate(events):
            e["key"] = f"e{k:04d}"
        texts = [(e["question"] or "") + (" " + e["answer_text"] if e["answer_text"] else "") for e in events]
        t0 = time.time()
        vec = embed(texts)
        cands, generic = candidates(events, vec)
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "nodes.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        with open(d / "candidates.jsonl", "w") as f:
            for e, c in zip(events, cands):
                f.write(json.dumps({"id": e["id"], "key": e["key"],
                                    "candidates": [{**x, "id": events[x["j"]]["id"], "key": events[x["j"]]["key"]}
                                                   for x in c]}) + "\n")
        np.save(d / "embeddings.npy", vec)
        info = {"split": split, "events": len(events), "stats": dict(stats), "generic_groups": generic,
                "embed_seconds": round(time.time() - t0, 1),
                "cand_len": dict(Counter(min(len(c), N_CAND) // 10 * 10 for c in cands)),
                "by_type": dict(Counter(e["outcome_type"] for e in events)),
                "yes_rate": round(sum(e["resolution"] == "YES" for e in events) / max(1, len(events)), 4),
                "in_pair_manifest": sum(e["in_pair_manifest"] for e in events),
                "config": {"n_cand": N_CAND, "max_siblings": MAX_SIBLINGS, "min_overlap_days": MIN_OVERLAP / DAY,
                           "group_bonus": GROUP_BONUS, "generic_group_share": GENERIC_GROUP_SHARE,
                           "embedding": {"model": EMB_MODEL, "revision": EMB_REV, "pooling": "cls", "max_length": 128}}}
        (d / "nodes-info.json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps({k: info[k] for k in ("split", "events", "stats", "by_type", "yes_rate", "cand_len")}))


# --- sessions -----------------------------------------------------------------

def load_split(split: str, cand_file: str = "candidates.jsonl"):
    d = OUT / split
    nodes = [json.loads(line) for line in (d / "nodes.jsonl").read_text().splitlines()]
    cands = [json.loads(line) for line in (d / cand_file).read_text().splitlines()]
    return nodes, cands


def make_batches(nodes, cands, size: int, only: list[int] | None = None) -> list[list[int]]:
    """Greedy: seed with the first unassigned event, fill from its candidates, so batches share table rows."""
    todo = list(range(len(nodes))) if only is None else list(only)
    allowed = set(todo)
    left = dict.fromkeys(todo)
    batches = []
    while left:
        seed = next(iter(left))
        batch = [seed]
        del left[seed]
        frontier = [c["j"] for c in cands[seed]["candidates"]]
        while len(batch) < size:
            nxt = next((j for j in frontier if j in left and j in allowed), None)
            if nxt is None:
                nxt = next(iter(left), None)
                if nxt is None:
                    break
            batch.append(nxt)
            del left[nxt]
            frontier += [c["j"] for c in cands[nxt]["candidates"]]
        batches.append(sorted(batch))
    return batches


def render(nodes, cands, batch: list[int]) -> str:
    rows = sorted({j for i in batch for j in [i] + [c["j"] for c in cands[i]["candidates"]]})
    table = "\n".join(f"{nodes[j]['key']} | {nodes[j]['close_date']} | {title(nodes[j])}" for j in rows)
    focus = []
    for i in batch:
        e = nodes[i]
        crit = asof.clip(e["criteria_text"], 400)
        focus.append(f"FOCUS {e['key']}\nQuestion: {e['question']}" +
                     (f"\nOption: {e['answer_text']}" if e["answer_text"] is not None else "") +
                     f"\nOpen {e['open_date']} to {e['close_date']}" + (f"\nCriteria: {crit}" if crit else "") +
                     "\nCandidates: " + " ".join(nodes[c["j"]]["key"] for c in cands[i]["candidates"]))
    return PROMPT.format(k=MAX_LINKS, table=table, focus="\n\n".join(focus))


def parse(text: str | None, nodes, cands, batch) -> tuple[dict, dict]:
    """Keep only (focus key -> [(candidate key, type)]) with valid ids/types; everything else is dropped."""
    info = Counter()
    if not text:
        return {}, {"status": "empty"}
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0) if m else text)
    except (ValueError, AttributeError):
        return {}, {"status": "invalid_json"}
    links = obj.get("links") if isinstance(obj, dict) else None
    if not isinstance(links, dict):
        return {}, {"status": "no_links"}
    info["extra_top_keys"] = len(set(obj) - {"links"})
    out = {}
    for i in batch:
        key = nodes[i]["key"]
        allowed = {nodes[c["j"]]["key"] for c in cands[i]["candidates"]}
        got = links.get(key)
        if got is None:
            info["focus_missing"] += 1
            continue
        kept, seen = [], set()
        for item in got if isinstance(got, list) else []:
            if isinstance(item, dict):
                item = [item.get("id"), item.get("type")]
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                info["bad_item"] += 1
                continue
            ck, typ = str(item[0]).strip(), str(item[1]).strip().lower()
            if len(item) > 2:
                info["extra_fields_stripped"] += 1
            if ck not in allowed or ck in seen:
                info["bad_or_dup_id"] += 1
                continue
            if typ not in TYPES:
                info["bad_type"] += 1
                continue
            seen.add(ck)
            kept.append([ck, typ])
        if len(kept) > MAX_LINKS:
            info["over_limit_cut"] += len(kept) - MAX_LINKS
            kept = kept[:MAX_LINKS]
        out[key] = kept
    info["extra_focus_keys"] = len(set(links) - {nodes[i]["key"] for i in batch})
    return out, {"status": "ok" if len(out) == len(batch) else "partial", **{k: v for k, v in info.items() if v}}


class Gate:
    """Shared back-off: 429/529 from any worker pauses all; proxy claims in fleet-now pause all; PAUSE file too."""

    def __init__(self, pause_file: Path):
        self.until = 0.0
        self.lock = threading.Lock()
        self.pause_file = pause_file
        self.last_claim_check, self.claim_until = 0.0, 0.0

    def backoff(self, seconds: float) -> None:
        with self.lock:
            self.until = max(self.until, time.time() + seconds)

    def proxy_claim(self) -> float:
        now = time.time()
        if now - self.last_claim_check > 60:
            self.last_claim_check = now
            try:
                rows = json.loads(subprocess.run(["fleet-now", "grep", "claim", "--json"], capture_output=True,
                                                 text=True, timeout=30).stdout or "[]")
                until = [dt.datetime.fromisoformat(r["until"].replace("Z", "+00:00")).timestamp() for r in rows
                         if r.get("kind") == "claim" and "proxy" in (r.get("tags") or []) and r.get("until")]
                self.claim_until = max([u for u in until if u > now], default=0.0)
            except Exception:
                pass
        return self.claim_until

    def wait(self, log) -> None:
        while True:
            now = time.time()
            reasons = []
            if self.pause_file.exists():
                reasons.append("PAUSE file")
            if self.until > now:
                reasons.append(f"backoff {self.until - now:.0f}s")
            if self.proxy_claim() > now:
                reasons.append("proxy claim in fleet-now")
            if not reasons:
                return
            log(f"waiting: {', '.join(reasons)}")
            time.sleep(30)


def call_luna(prompt: str, work: Path, effort: str, timeout: float) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "XAI_", "ZAI_", "ZHIPU_", "CODEX_API_"))}
    env.pop("TMUX", None)
    env.update(FLEET_ACTOR="consistency-luna-linkgraph", FUTARCHY_AGENT_NAME="consistency-luna-linkgraph",
               AI_AGENT_NAME="consistency-luna-linkgraph")
    with tempfile.TemporaryDirectory(dir=work) as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        out = Path(td) / "final.txt"
        env["TMPDIR"] = td
        cmd = [CODEX, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
               "-C", str(empty), "--model", MODEL, "-c", "model_reasoning_effort=" + json.dumps(effort),
               "-c", 'service_tier="default"', "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
               "-c", "features.multi_agent_v2=false", "-c", "features.shell_tool=false",
               "-c", "developer_instructions=" + json.dumps(RULE), "-o", str(out), "-"]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=timeout, cwd=empty)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, rc = "", "timed out", "timeout"
        text = out.read_text() if out.exists() else None
    usage, errors, tools = None, [], 0
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage")
        elif ev.get("type") in ("error", "turn.failed"):
            errors.append(json.dumps(ev)[:400])
        elif ev.get("type") == "item.completed":
            it = ev.get("item", {})
            if it.get("type") not in ("agent_message", "reasoning", "error"):
                tools += 1
    return {"text": text, "usage": usage, "errors": errors, "tool_items": tools, "rc": rc,
            "stderr_tail": stderr[-600:], "elapsed": round(time.time() - t0, 1)}


def run_session(sid: str, batch, nodes, cands, sdir: Path, gate: Gate, effort: str, log) -> dict:
    prompt = render(nodes, cands, batch)
    attempts, delay = [], 60.0
    links, pinfo, res = {}, {"status": "not_run"}, None
    reminder = ""
    for attempt in range(8):
        gate.wait(log)
        res = call_luna(prompt + reminder, sdir, effort, timeout=1200)
        err = " ".join(res["errors"]) + " " + res["stderr_tail"]
        attempts.append({k: res[k] for k in ("rc", "elapsed", "usage", "errors", "tool_items")})
        if "model_not_found" in err or "not supported" in err or "does not exist" in err:
            raise SystemExit(f"model {MODEL} rejected, stopping (no substitution): {err[:300]}")
        if res["usage"] is None or res["text"] is None:
            if any(t.lower() in err.lower() for t in TRANSIENT) or res["rc"] != 0:
                if any(c in err for c in ("429", "529", "Too Many", "overloaded", "capacity")):
                    gate.backoff(delay)
                log(f"{sid} transient (attempt {attempt}): {err.strip()[:160]}")
                time.sleep(delay * random.uniform(0.5, 1.0))
                delay = min(delay * 2, 900)
                continue
            break
        links, pinfo = parse(res["text"], nodes, cands, batch)
        if pinfo["status"] in ("ok",) and res["tool_items"] == 0:
            break
        log(f"{sid} parse {pinfo} tools={res['tool_items']} (attempt {attempt})")
        reminder = ("\n\nReminder: reply with exactly one JSON object {\"links\": {...}} containing every focus id, "
                    "only candidate ids from that focus question's list and one of the five relation types.")
        if attempt >= 2:
            break
    rec = {"session": sid, "model": MODEL, "effort": effort, "batch": [nodes[i]["id"] for i in batch],
           "keys": [nodes[i]["key"] for i in batch], "links": links, "parse": pinfo,
           "tool_items": res["tool_items"] if res else None, "raw": res["text"] if res else None,
           "attempts": attempts, "prompt_chars": len(prompt), "finished_at": time.time()}
    (sdir / f"{sid}.json").write_text(json.dumps(rec) + "\n")
    return rec


def cmd_run(a) -> None:
    if a.pass2:
        nodes, cands = load_split(a.split, "candidates-pass2.jsonl")
        sdir, a.plan = OUT / a.split / "raw-pass2", "pass2"
    else:
        nodes, cands = load_split(a.split)
        sdir = OUT / a.split / "raw"
    sdir.mkdir(parents=True, exist_ok=True)
    only = [i for i, c in enumerate(cands) if c["candidates"]] if a.pass2 else None
    if a.limit:
        rng = random.Random(a.seed)
        only = sorted(rng.sample(range(len(nodes)), min(a.limit, len(nodes))))
    # batches are fixed once (plan file) so reruns resume the same sessions
    plan_path = OUT / a.split / (f"plan-{a.plan}.json")
    if plan_path.exists():
        batches = json.loads(plan_path.read_text())
    else:
        done_ids = set()
        for f in sdir.glob("*.json"):
            r = json.loads(f.read_text())
            if r["parse"].get("status") in ("ok", "partial"):
                done_ids |= set(r["keys"]) if r["parse"]["status"] == "ok" else set(r["links"])
        todo = [i for i in (only if only is not None else range(len(nodes))) if nodes[i]["key"] not in done_ids]
        batches = make_batches(nodes, cands, a.batch, todo)
        plan_path.write_text(json.dumps(batches))
    logf = open(OUT / a.split / f"run-{a.plan}.log", "a")
    lock = threading.Lock()

    def log(msg):
        with lock:
            logf.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")
            logf.flush()

    gate = Gate(OUT / "PAUSE")
    todo = []
    for k, b in enumerate(batches):
        sid = f"{a.plan}-{k:04d}"
        f = sdir / f"{sid}.json"
        if f.exists() and json.loads(f.read_text())["parse"].get("status") == "ok":
            continue
        todo.append((sid, b))
    if a.max_batches:
        todo = todo[:a.max_batches]
    log(f"start plan={a.plan} batches={len(batches)} todo={len(todo)} workers={a.workers} effort={a.effort}")
    t0, n_done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {}
        for k, (sid, b) in enumerate(todo):
            futs[pool.submit(run_session, sid, b, nodes, cands, sdir, gate, a.effort, log)] = sid
        for fut in as_completed(futs):
            rec = fut.result()
            n_done += 1
            u = (rec["attempts"][-1].get("usage") or {}) if rec["attempts"] else {}
            log(f"{rec['session']} {rec['parse'].get('status')} links={sum(len(v) for v in rec['links'].values())} "
                f"in={u.get('input_tokens')} out={u.get('output_tokens')} t={rec['attempts'][-1]['elapsed'] if rec['attempts'] else None} "
                f"[{n_done}/{len(todo)} {time.time() - t0:.0f}s]")
    log("done")


# --- edges and stats ----------------------------------------------------------

def cmd_edges(a) -> None:
    raw, out, proposer = ("raw-pass2", "edges-pass2.jsonl", "luna-pass2") if a.pass2 else ("raw", "edges.jsonl", "luna")
    for split in a.splits:
        d = OUT / split
        nodes, _ = load_split(split)
        by_key = {e["key"]: e for e in nodes}
        known = set()
        if a.pass2:
            known = {(e["u"], e["v"]) for e in map(json.loads, (d / "edges.jsonl").read_text().splitlines())}
        props = defaultdict(dict)  # (u, v) -> {proposer id: type}
        sessions = defaultdict(set)
        proposed_for = set()
        n_sess = 0
        for f in sorted((d / raw).glob("*.json")):
            r = json.loads(f.read_text())
            n_sess += 1
            for fk, lst in r["links"].items():
                proposed_for.add(fk)
                for ck, typ in lst:
                    u, v = sorted((by_key[fk]["id"], by_key[ck]["id"]))
                    if (u, v) in known:
                        continue
                    props[(u, v)][by_key[fk]["id"]] = typ
                    sessions[(u, v)].add(r["session"])
        with open(d / out, "w") as fh:
            for (u, v), p in sorted(props.items()):
                types = sorted(set(p.values()))
                fh.write(json.dumps({"u": u, "v": v, "type": Counter(p.values()).most_common(1)[0][0] if len(types) == 1
                                     else sorted(p.values(), key=lambda t: TYPES.index(t))[0],
                                     "types": types, "proposed_by": {k: p[k] for k in sorted(p)},
                                     "mutual": len(p) == 2, "model": MODEL, "proposer": proposer,
                                     "sessions": sorted(sessions[(u, v)])}) + "\n")
        print(json.dumps({"split": split, "sessions": n_sess, "events_linked_for": len(proposed_for),
                          "events": len(nodes), "edges": len(props),
                          "mutual": sum(len(p) == 2 for p in props.values())}))


def cmd_stats(a) -> None:
    import networkx as nx
    from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in
    summary = {}
    for split in a.splits:
        d = OUT / split
        nodes, _ = load_split(split)
        g = nx.Graph()
        g.add_nodes_from(e["id"] for e in nodes)
        edges = [json.loads(line) for line in (d / "edges.jsonl").read_text().splitlines()]
        if (d / "edges-pass2.jsonl").exists():  # final graph = pass 1 + pass 2
            edges += [json.loads(line) for line in (d / "edges-pass2.jsonl").read_text().splitlines()]
        for e in edges:
            g.add_edge(e["u"], e["v"])
        deg = [x for _, x in g.degree()]
        comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        tri = sum(nx.triangles(g).values()) // 3
        maxc = max((len(c) for c in nx.find_cliques(g)), default=0)
        tw_md, _ = treewidth_min_degree(g)
        tw_mf, _ = treewidth_min_fill_in(g)
        big = g.subgraph(max(nx.connected_components(g), key=len)) if g.number_of_edges() else g
        s = {"events": g.number_of_nodes(), "edges": g.number_of_edges(),
             "mutual_edges": sum(e["mutual"] for e in edges),
             "edges_by_proposer": dict(Counter(e.get("proposer", "luna") for e in edges)),
             "edge_types": dict(Counter(e["type"] for e in edges)),
             "same_parent_edges": sum(e["u"].split(":")[0] == e["v"].split(":")[0] for e in edges),
             "degree": {"mean": round(float(np.mean(deg)), 2), "median": float(np.median(deg)), "max": max(deg),
                        "isolated": deg.count(0),
                        "hist": {str(k): v for k, v in sorted(Counter(min(x, 20) for x in deg).items())}},
             "components": {"count": len(comps), "largest": comps[:5], "singletons": comps.count(1),
                            "size_hist": {str(k): v for k, v in sorted(Counter(comps).items())}},
             "triangles": tri, "max_clique": maxc, "degeneracy": max(nx.core_number(g).values(), default=0),
             "treewidth_upper_min_degree": tw_md, "treewidth_upper_min_fill_in": tw_mf,
             "largest_component_edges": big.number_of_edges(),
             "clustering_avg": round(nx.average_clustering(g), 4)}
        (d / "graph-stats.json").write_text(json.dumps(s, indent=1) + "\n")
        summary[split] = s
        print(split, json.dumps({k: s[k] for k in ("events", "edges", "triangles", "max_clique",
                                                   "treewidth_upper_min_fill_in", "degeneracy")}), s["degree"])
    (OUT / "graph-stats.json").write_text(json.dumps(summary, indent=1) + "\n")


def _components(n: int, pairs: list[tuple[int, int]]):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if not pairs:
        return n, np.arange(n)
    u, v = np.array(pairs).T
    g = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    return connected_components(g, directed=False)


def _paths(n: int, pairs: list[tuple[int, int]]) -> dict:
    """Diameter and mean shortest path (all pairs, BFS) of a connected graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    u, v = np.array(pairs).T
    g = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    d = shortest_path(g, directed=False, unweighted=True)
    off = d[~np.eye(n, dtype=bool)]
    fin = off[np.isfinite(off)]
    return {"nodes": n, "edges": len(set(map(tuple, map(sorted, pairs)))), "diameter": int(fin.max()),
            "avg_shortest_path": round(float(fin.mean()), 3), "unreachable_pairs": int((~np.isfinite(off)).sum()),
            "path_len_hist": {str(int(k)): int(c) for k, c in zip(*np.unique(fin, return_counts=True))}}


def _comp_summary(n, pairs) -> dict:
    k, lab = _components(n, pairs)
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    return {"count": int(k), "largest": sizes[:6], "singletons": sizes.count(1)}


ENT_STOP = frozenset("""will the what which who when where how does did is are can yes no end before after by in on
of and or for with new first next last any more most least than this that these those per total top over under
january february march april may june july august september october november december jan feb mar apr jun jul aug
sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday polymarket manifold market markets
question short fuse add responses other none may resolves resolution""".split())


def entities(e: dict) -> set[str]:
    text = (e["question"] or "") + " " + (e["answer_text"] or "")
    toks = re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9&'.-]{2,}", text)
    return {t.lower().strip(".'-") for t in toks} - ENT_STOP


def market_components(nodes, pairs):
    par = [e["parent_id"] for e in nodes]
    markets = sorted(set(par))
    mix = {m: k for k, m in enumerate(markets)}
    mp = [(mix[par[i]], mix[par[j]]) for i, j in pairs if par[i] != par[j]]
    k, lab = _components(len(markets), mp)
    return markets, lab, mp


def cmd_cands2(a) -> None:
    """Second-pass candidates (coordinator 2026-09-27, after Kelvin rejected embedding bridges): for every event of a
    market outside the giant market-level component, up to 80 same-split events of OTHER markets with open windows
    overlapping >= 1 day: same non-generic Manifold group (<= 25), shared rare capitalised names (<= 25), then
    nearest by BGE cosine; shown in cosine order.  Luna then links exactly as in pass 1."""
    for split in a.splits:
        d = OUT / split
        nodes, _ = load_split(split)
        n = len(nodes)
        ix = {e["id"]: k for k, e in enumerate(nodes)}
        vec = np.load(d / "embeddings.npy")
        sim = vec @ vec.T
        op = np.array([e["open_time"] for e in nodes])
        en = np.array([e["end_time"] for e in nodes])
        ok = (np.minimum(en[:, None], en[None, :]) - np.maximum(op[:, None], op[None, :])) >= MIN_OVERLAP
        par = np.array([e["parent_id"] for e in nodes])
        pairs = [(ix[e["u"]], ix[e["v"]]) for e in map(json.loads, (d / "edges.jsonl").read_text().splitlines())]
        markets, lab, _ = market_components(nodes, pairs)
        giant = np.argmax(np.bincount(lab))
        in_giant = {m for m, c in zip(markets, lab) if c == giant}
        targets = [i for i in range(n) if par[i] not in in_giant]
        gcount = Counter(g for e in nodes for g in set(e["groups"]))
        generic = {g for g, c in gcount.items() if c > GENERIC_GROUP_SHARE * n}
        gsets = [set(e["groups"]) - generic for e in nodes]
        ents = [entities(e) for e in nodes]
        ecount = Counter(t for s_ in ents for t in s_)
        ents = [{t for t in s_ if ecount[t] <= max(3, 0.03 * n)} for s_ in ents]
        out = [[] for _ in range(n)]
        src_count = Counter()
        for i in targets:
            idx = np.nonzero(ok[i] & (par != par[i]))[0]
            order = idx[np.argsort(-sim[i, idx], kind="stable")].tolist()
            grp = [j for j in order if gsets[i] & gsets[j]][:25]
            ent = sorted((j for j in order if ents[i] & ents[j]), key=lambda j: (-len(ents[i] & ents[j]), -sim[i, j]))[:25]
            chosen = dict.fromkeys(grp + ent)
            for j in order:
                if len(chosen) >= 80:
                    break
                chosen.setdefault(j)
            src = {j: ("group" if j in grp else "") + ("+name" if j in ent else "") or "embedding" for j in chosen}
            src_count.update(src.values())
            out[i] = [{"j": j, "cos": round(float(sim[i, j]), 4), "source": src[j].lstrip("+")}
                      for j in sorted(chosen, key=lambda j: -sim[i, j])]
        with open(d / "candidates-pass2.jsonl", "w") as f:
            for e, c in zip(nodes, out):
                f.write(json.dumps({"id": e["id"], "key": e["key"],
                                    "candidates": [{**x, "id": nodes[x["j"]]["id"], "key": nodes[x["j"]]["key"]}
                                                   for x in c]}) + "\n")
        print(json.dumps({"split": split, "markets_outside_giant": len(markets) - len(in_giant),
                          "target_events": len(targets), "mean_cands": round(float(np.mean([len(out[i]) for i in targets])), 1)
                          if targets else 0, "sources": dict(src_count)}))


def cmd_connectivity(a) -> None:
    """Components before/after pass 2 (event and market level), and path lengths within the giant component."""
    summary = {}
    for split in a.splits:
        d = OUT / split
        nodes, _ = load_split(split)
        n = len(nodes)
        ix = {e["id"]: k for k, e in enumerate(nodes)}
        p1 = [(ix[e["u"]], ix[e["v"]]) for e in map(json.loads, (d / "edges.jsonl").read_text().splitlines())]
        f2 = d / "edges-pass2.jsonl"
        p2 = [(ix[e["u"]], ix[e["v"]]) for e in map(json.loads, f2.read_text().splitlines())] if f2.exists() else []
        s = {"events": n, "edges_pass1": len(p1), "edges_pass2_new": len(p2)}
        for tag, pairs in (("pass1", p1), ("pass1+2", p1 + p2)):
            markets, lab, mp = market_components(nodes, pairs)
            ks = np.bincount(lab)
            k_ev, lab_ev = _components(n, pairs)
            ks_ev = np.bincount(lab_ev)
            cross = {m for pr in mp for m in pr}
            g_ev = np.nonzero(lab_ev == np.argmax(ks_ev))[0]
            remap = {int(v): k for k, v in enumerate(g_ev)}
            ev_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
            g_m = np.nonzero(lab == np.argmax(ks))[0]
            mremap = {int(v): k for k, v in enumerate(g_m)}
            m_pairs = [(mremap[i], mremap[j]) for i, j in mp if i in mremap and j in mremap]
            s[tag] = {"event": _comp_summary(n, pairs), "market": _comp_summary(len(markets), mp),
                      "markets": len(markets), "markets_without_cross_market_link": len(markets) - len(cross),
                      "giant_paths_event": {k: v for k, v in _paths(len(g_ev), ev_pairs).items() if k != "path_len_hist"},
                      "giant_paths_market": {k: v for k, v in _paths(len(g_m), m_pairs).items() if k != "path_len_hist"}}
        (d / "connectivity.json").write_text(json.dumps(s, indent=1) + "\n")
        summary[split] = s
        print(split, json.dumps(s))
    (OUT / "connectivity.json").write_text(json.dumps(summary, indent=1) + "\n")


def cmd_show(a) -> None:
    """Print neighbourhoods (for the doc)."""
    nodes, _ = load_split(a.split)
    by_id = {e["id"]: e for e in nodes}
    adj = defaultdict(list)
    for line in (OUT / a.split / "edges.jsonl").read_text().splitlines():
        e = json.loads(line)
        adj[e["u"]].append((e["v"], e))
        adj[e["v"]].append((e["u"], e))
    for eid in a.ids:
        e = by_id[eid]
        print(f"* {title(e, 140)} [{e['id']}, open {e['open_date']} to {e['end_date']}]")
        for nb, ed in adj[eid]:
            who = "both" if ed["mutual"] else ("this" if eid in ed["proposed_by"] else "the neighbour")
            print(f"    - {ed['type']}: {title(by_id[nb], 110)} [{nb}; proposed by {who}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nodes"); p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p = sub.add_parser("run")
    p.add_argument("--split", required=True); p.add_argument("--plan", default="full")
    p.add_argument("--batch", type=int, default=10); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--effort", default="medium"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="run only the first N pending sessions (pilot)")
    p.add_argument("--pass2", action="store_true", help="second pass: widened candidates (candidates-pass2.jsonl)")
    p = sub.add_parser("edges"); p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--pass2", action="store_true")
    p = sub.add_parser("stats"); p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p = sub.add_parser("cands2"); p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p = sub.add_parser("connectivity"); p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p = sub.add_parser("show"); p.add_argument("--split", required=True); p.add_argument("ids", nargs="+")
    a = ap.parse_args()
    {"nodes": cmd_nodes, "run": cmd_run, "edges": cmd_edges, "stats": cmd_stats, "cands2": cmd_cands2, "connectivity": cmd_connectivity,
     "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
