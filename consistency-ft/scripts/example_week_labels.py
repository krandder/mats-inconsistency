"""Short (<=40 character) display labels for the example-week events, written by Qwen3-32B (thinking off).

Display only: the labels never enter a forecasting prompt.  One call per event; results go to
labels.jsonl in the week directory, and events whose answer is too long or missing keep the
heuristic label from example_week.py.
Usage: example_week_labels.py --out /home/agent/data/example-week-20260927 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precutoff_client as pc  # noqa: E402
import worldmodel_clusters as wc  # noqa: E402

PROMPT = ("Write a label of at most 40 characters for this forecasting question, for a node in a graph. "
          "Keep the key entity, the event and the date if it matters. No question mark, no quotes. "
          "Reply with the label only.\n\nQuestion: {q}")


def clean(t: str | None) -> str:
    return (t or "").strip().strip('"').strip("'").rstrip("?").strip()


def fit40(raws: list, fallback: str) -> str:
    """The model's shortest answer, cut at a word boundary to <=40 characters when it overshot."""
    ts = sorted((clean(t) for t in raws if clean(t) and "\n" not in clean(t)), key=len)
    if not ts:
        return fallback
    t = ts[0]
    if len(t) <= 40:
        return t
    cut = t[:39].rsplit(" ", 1)[0].rstrip(",:;-")
    return cut + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    events = [json.loads(l) for l in (a.out / "events.jsonl").read_text().splitlines()]
    k = subprocess.run(["secret", "get", a.key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {a.key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[wc.MODEL] = ("dashscope", wc.MODEL, {"enable_thinking": False, "temperature": 0}, wc.PRICE_IN, wc.PRICE_OUT)

    def one(e):
        q = e["parent_question"] + (f" -- option: {e['answer_text']}" if e.get("answer_text") else "")
        r = pc.chat(wc.MODEL, [{"role": "user", "content": PROMPT.format(q=q)}], max_tokens=30)
        if r["served_model"] and r["served_model"] != wc.MODEL:
            raise RuntimeError(f"served model {r['served_model']} != {wc.MODEL}")
        t = (r["text"] or "").strip().strip('"').strip("'").rstrip("?").strip()
        cost, raw = wc.cost_of(r["usage"]), [r["text"]]
        if not (0 < len(t) <= 40) and r["text"]:  # one retry: the model often overshoots by a few characters
            r = pc.chat(wc.MODEL, [{"role": "user", "content": PROMPT.format(q=q)}, {"role": "assistant", "content": r["text"]},
                                   {"role": "user", "content": f"That is {len(t)} characters. Shorten it to at most 34 characters."}],
                        max_tokens=30)
            t = (r["text"] or "").strip().strip('"').strip("'").rstrip("?").strip()
            cost += wc.cost_of(r["usage"]); raw.append(r["text"])
        ok = 0 < len(t) <= 40 and "\n" not in t
        return {"id": e["id"], "label": t if ok else fit40(raw, e["short"]), "raw": raw, "model_ok": ok, "cost": cost}

    with ThreadPoolExecutor(min(a.workers, 8)) as pool:
        rows = list(pool.map(one, events))
    (a.out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} labels, {sum(r['model_ok'] for r in rows)} from the model, cost ${sum(r['cost'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
