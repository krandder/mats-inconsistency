"""Extra statistics for the Luna ceiling, computed the same way for Qwen3-32B.

Coherence: corr(p(P), p(NOT P)) over pairs (a coherent forecaster gives -1),
mean |p(P) + p(NOT P) - 1|, SD of p(AND) forms.  Recall screen, per unique
proposition (the model's literal P/Q answers averaged over the pairs where the
proposition appears): AUC, sharpness, share of confident calls (<=0.1 or >=0.9)
and their hit rate versus the confidence they claim, split by binary vs
multiple-choice option and by closing month.  A calibrated forecaster's 0.9
calls hit about 90% of the time; near-perfect hit rates on questions whose
base rate is near 0.5 point to recall of the outcome.

Usage: analyze_luna_ceiling.py --data-root D --pred luna.jsonl [--compare name=pred.jsonl ...] --out extra.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]; neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def coherence(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["pair_id"]][r["form"]] = r["p"]
    xs, ys, gaps = [], [], []
    for f in by.values():
        for a, b in (("P", "NOT P"), ("Q", "NOT Q")):
            if f.get(a) is not None and f.get(b) is not None:
                if a == "P":
                    xs.append(f[a]); ys.append(f[b])
                gaps.append(abs(f[a] + f[b] - 1))
    ands = [r["p"] for r in rows if "AND" in r["form"] and r["p"] is not None]
    ors = [r["p"] for r in rows if "OR" in r["form"] and r["p"] is not None]
    return {"corr_pP_pNOTP": corr(xs, ys), "n_pairs": len(xs), "mean_abs_complement_gap": st.mean(gaps) if gaps else None,
            "sd_p_AND": st.pstdev(ands) if ands else None, "mean_p_AND": st.mean(ands) if ands else None,
            "sd_p_OR": st.pstdev(ors) if ors else None, "mean_p_OR": st.mean(ors) if ors else None}


def proposition_answers(rows):
    """Per proposition: mean model p(YES) from literal forms (P, Q, and 1 - NOT P, 1 - NOT Q)."""
    vals, labels = defaultdict(list), {}
    for r in rows:
        if r["p"] is None or r["form"] not in ("P", "Q", "NOT P", "NOT Q"):
            continue
        pid = r["p_id"] if "P" in r["form"] else r["q_id"]
        neg = r["form"].startswith("NOT")
        vals[pid].append(1 - r["p"] if neg else r["p"])
        labels[pid] = 1 - r["label"] if neg else r["label"]
    return {pid: (st.mean(v), labels[pid]) for pid, v in vals.items()}


def screen(items):
    if not items:
        return {"n": 0}
    ps = [p for p, _ in items]; ys = [y for _, y in items]
    conf = [(p, y) for p, y in items if p <= 0.1 or p >= 0.9]
    hits = [(p >= 0.5) == bool(y) for p, y in conf]
    claimed = [max(p, 1 - p) for p, _ in conf]
    return {"n": len(items), "yes_rate": st.mean(ys), "auc": auc(ps, ys), "accuracy": st.mean((p >= 0.5) == bool(y) for p, y in items),
            "sharpness_mean_abs_p_minus_half": st.mean(abs(p - 0.5) for p in ps),
            "confident_share": len(conf) / len(items), "confident_n": len(conf),
            "confident_hit_rate": st.mean(hits) if hits else None, "confident_claimed": st.mean(claimed) if claimed else None,
            "confident_yes_calls": sum(p >= 0.9 for p, _ in conf),
            "confident_yes_hit_rate": st.mean([y for p, y in conf if p >= 0.9]) if any(p >= 0.9 for p, _ in conf) else None}


def recall_screen(rows, props):
    ans = proposition_answers(rows)
    out = {"all": screen(list(ans.values()))}
    for kind in ("binary", "option"):
        out[kind] = screen([v for pid, v in ans.items() if bool(props[pid].get("answer_text")) == (kind == "option")])
    months = defaultdict(list)
    for pid, v in ans.items():
        months[props[pid]["closing_week"][:7]].append(v)
    out["by_closing_month"] = {m: screen(v) for m, v in sorted(months.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {"luna": args.pred, **dict(c.split("=", 1) for c in args.compare)}
    loaded = {k: [json.loads(l) for l in Path(v).read_text().splitlines()] for k, v in runs.items()}
    split = loaded["luna"][0]["pair_id"].split("-pair-")[0]
    props = {p["id"]: p for p in map(json.loads, (Path(args.data_root) / split / "propositions.jsonl").read_text().splitlines())}
    res = {}
    for name, rows in loaded.items():
        res[name] = {"coherence": coherence(rows), "recall_screen": recall_screen(rows, props)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        c, s = r["coherence"], r["recall_screen"]
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()})
        for kind in ("binary", "option"):
            k = s[kind]
            print(f"  {kind}: n={k['n']} yes={k['yes_rate']:.2f} auc={k['auc']:.3f} acc={k['accuracy']:.3f} "
                  f"confident {k['confident_share']:.1%} hit {k['confident_hit_rate']} vs claimed {k['confident_claimed']}")


if __name__ == "__main__":
    main()
