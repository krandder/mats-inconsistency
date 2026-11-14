#!/usr/bin/env python3
"""Audit gpt-6-sol market-game traces: per-question rationales, leaks the model reported
(articles it excluded as post-date or outcome-revealing), and a filtered trace file.

Outputs in runs/market-game/sol-train-n16/audit/:
  per-question.jsonl   one row per traded question: rationale, target, resolution, leak phrases,
                       cited excluded articles, Jev flags of those articles
  missed-leaks.jsonl   excluded articles Sol named, with Jev's scores (the misses are flags all false)
  traces-clean.jsonl   rationale+trade rows kept for distillation
  report.md
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "runs/market-game/sol-train-n16"
POOL = Path("/home/agent/data/polymarket-evidence-exa-20260922")
OUT = RUN / "audit"

LEAK_RE = re.compile(
    r"(exclud|ignor|disregard|set aside|discount|do not use|don't use|won't use|not us)\w*[^.]{0,80}?"
    r"(after|later|post[- ]?(date|match|game|event|election|result)|leak|outcome|result|as-of|date rule|future)",
    re.I)
ART_RE = re.compile(r"(?:articles?\s+|\[)\s*([\d][\d,\s–\-and\]\[]*)", re.I)
EXCL_CLAUSE_RE = re.compile(r"(exclud|ignor|disregard|set aside|discount)\w*([^.;]*)", re.I)
REFUSE_RE = re.compile(r"(can(’|')?t|cannot|won(’|')?t|will not|not able to)\s+(provide|assign|price|trade|give|estimate)", re.I)


def articles_in(text: str) -> list[int]:
    out = []
    for m in ART_RE.finditer(text):
        for part in re.split(r"[,\s\[\]]+|and", m.group(1)):
            part = part.strip()
            if re.fullmatch(r"\d+[–-]\d+", part):
                a, b = map(int, re.split(r"[–-]", part)); out += list(range(a, b + 1))
            elif part.isdigit():
                out.append(int(part))
    return sorted(set(out))


def split_rationales(text: str, n: int) -> dict[int, str]:
    """Map q-index -> rationale sentence(s), from '**q3 — 0.06:** ...' or '- q3: ...' lines."""
    body = text.split("```")[0]
    chunks: dict[int, str] = {}
    pat = re.compile(r"(?:^|\n)\s*[-*]*\s*\**\s*q(\d{1,2})\b[^\n]*", re.I)
    ms = list(pat.finditer(body))
    for i, m in enumerate(ms):
        k = int(m.group(1)); end = ms[i + 1].start() if i + 1 < len(ms) else len(body)
        if 1 <= k <= n:
            chunks[k] = (chunks.get(k, "") + " " + body[m.start():end]).strip()
    return chunks


def evidence_articles(prompt: str) -> dict[int, dict]:
    arts = {}
    for m in re.finditer(r"\[(\d+)\]\s*(.*?)\s*\(published on ([^ ]+) by ([^)]*)\)", prompt):
        arts[int(m.group(1))] = {"title": m.group(2)[:200], "published": m.group(3), "source": m.group(4)}
    return arts


def main() -> None:
    OUT.mkdir(exist_ok=True)
    questions = {json.loads(l)["question_id"]: json.loads(l) for l in open(RUN / "questions.jsonl") if l.strip()}
    last = {}
    for l in open(RUN / "sessions.jsonl"):
        if l.strip():
            r = json.loads(l); last[r["session_id"]] = r
    sessions = [r for r in last.values() if r.get("status") == "completed"]

    # Jev flags keyed by (question id, title prefix) is fragile; key by (id, url) and by id -> list in prompt order
    flags_by_q = collections.defaultdict(list)
    for l in open(POOL / "leak-flags.jsonl"):
        f = json.loads(l); flags_by_q[f["id"]].append(f)
    pool_titles = {}
    for l in open(POOL / "pool.jsonl"):
        p = json.loads(l)
        for res in p.get("results") or []:
            pool_titles[(p["id"], (res.get("title") or "")[:60])] = res.get("url")

    rows, missed = [], []
    stats = collections.Counter()
    for s in sessions:
        ids = s["question_ids"]; text = s.get("raw_response") or ""
        rats = split_rationales(text, len(ids))
        traded = {}
        for e in s.get("executed") or []:
            if not e["trade"].get("context"):
                traded[e["trade"]["variable"]] = e["post"]
        for k, qid in enumerate(ids, start=1):
            q = questions.get(qid, {})
            rat = rats.get(k, "")
            if qid not in traded:
                stats["untraded"] += 1
                if REFUSE_RE.search(text):
                    stats["untraded_refusal_session"] += 1
                continue
            stats["traded"] += 1
            leak_hits = [m.group(0) for m in LEAK_RE.finditer(rat)]
            arts = evidence_articles(q.get("prompt", ""))
            excluded = []
            if leak_hits:
                stats["leak_reported"] += 1
                cited = set()
                for cm in EXCL_CLAUSE_RE.finditer(rat):
                    cited.update(articles_in(cm.group(0)))
                # "Article 1's January comment is excluded": article named just before the verb
                for cm in re.finditer(r"(?:article\s+(\d+)|\[(\d+)\])[^.;]{0,60}?(is|are)\s+(excluded|ignored|disregarded)", rat, re.I):
                    cited.add(int(cm.group(1) or cm.group(2)))
                for hit in leak_hits[:1]:
                    for a in sorted(cited):
                        if a in arts:
                            url = pool_titles.get((qid, arts[a]["title"][:60]))
                            jev = next((f for f in flags_by_q.get(qid, []) if f.get("url") == url), None)
                            item = {"question_id": qid, "question": q.get("text"), "article": a, **arts[a], "url": url,
                                    "today": q.get("today"), "jev_p": jev and jev["p"], "jev_flags": jev and jev["flags"],
                                    "sol_says": hit[:200]}
                            excluded.append(item)
                            missed.append(item)
            p = traded[qid]; y = int(q.get("resolution", 0))
            rows.append({"question_id": qid, "session_id": s["session_id"], "q": k, "target": p, "resolution": y,
                         "brier": (p - y) ** 2, "rationale": rat[:2000], "leak_reported": bool(leak_hits),
                         "excluded_articles": excluded})

    # which reported leaks did Jev miss (no flag set on that snippet)
    with_jev = [m for m in missed if m["jev_flags"] is not None]
    jev_missed = [m for m in with_jev if not any(m["jev_flags"].values())]
    no_match = [m for m in missed if m["jev_flags"] is None]

    def brier(rs): return sum(r["brier"] for r in rs) / len(rs) if rs else float("nan")
    leak_rows = [r for r in rows if r["leak_reported"]]; clean_rows = [r for r in rows if not r["leak_reported"]]
    # suspicious confidence: extreme and correct, a leak signature even when not reported
    extreme = [r for r in rows if r["target"] <= 0.05 or r["target"] >= 0.95]
    extreme_right = [r for r in extreme if (r["target"] >= 0.5) == bool(r["resolution"])]

    with open(OUT / "per-question.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(OUT / "missed-leaks.jsonl", "w") as f:
        for m in missed: f.write(json.dumps(m) + "\n")
    kept = 0
    with open(OUT / "traces-clean.jsonl", "w") as f:
        for r in clean_rows:
            if len(r["rationale"]) >= 40:
                f.write(json.dumps({k: r[k] for k in ("question_id", "session_id", "q", "target", "rationale")}) + "\n"); kept += 1

    rep = [
        "# Sol trace audit", "",
        f"Completed sessions {len(sessions)}; traded questions {stats['traded']}; untraded {stats['untraded']}.",
        f"Questions where Sol said it set evidence aside as later or outcome-revealing: {len(leak_rows)} "
        f"({100*len(leak_rows)/max(1,len(rows)):.1f}%).",
        f"Excluded articles Sol named: {len(missed)}; matched to a Jev row {len(with_jev)}; "
        f"**Jev set no flag on {len(jev_missed)}** ({100*len(jev_missed)/max(1,len(with_jev)):.0f}% of matched); unmatched {len(no_match)}.", "",
        "| subset | n | Brier |", "|---|---:|---:|",
        f"| all traded | {len(rows)} | {brier(rows):.4f} |",
        f"| Sol reported a leak | {len(leak_rows)} | {brier(leak_rows):.4f} |",
        f"| no leak reported | {len(clean_rows)} | {brier(clean_rows):.4f} |",
        f"| extreme targets (≤0.05 or ≥0.95) | {len(extreme)} | {brier(extreme):.4f} |", "",
        f"Extreme targets on the correct side: {len(extreme_right)}/{len(extreme)}.",
        f"Clean traces written for distillation: {kept} (leak-reported questions and empty rationales dropped).", "",
        "## Examples Jev missed", ""]
    for m in jev_missed[:12]:
        rep.append(f"- **{m['question']}** (as of {m['today']}): article {m['article']} “{m['title']}”, published {m['published']} "
                   f"({m['source']}); Jev p = {m['jev_p']}; Sol: “{m['sol_says']}”")
    (OUT / "report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep[:16]))


if __name__ == "__main__":
    main()
