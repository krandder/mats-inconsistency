"""Apply the pre-registered selection rule (PREREG.md) to results/summary.json (VALIDATION).

Eligible: real-world-valid without conditions (not refpost, not reordered evidence = stages=perm / arm perm),
full sweep (5 N) with >= 3 paired seeds at every N. Score S_small (lower = better); champion = lowest S_small
with S_all < 0 (else lowest S_small, flagged). Strict plain baseline = best plain config (k0 / k0cell family).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
summ = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "summary.json")))
NS = ["100", "300", "1000", "3000", "10000"]


def full(e):
    return all(n in e["per_n"] and len(e["per_n"][n]["paired_seeds"]) >= 3 for n in NS)


def perm(t, e):
    return e["arm"] == "perm" or "stages=perm" in (e.get("set") or "")


def plain(e):
    return e["arm"] in ("k0", "k0cell") and not any(k in (e.get("set") or "") for k in ("stages=", "td=", "mart=", "masked=", "bn=", "rbtd="))


rows = []
for t, e in summ.items():
    elig = e["rwv"] and not perm(t, e) and full(e) and e["arm"] != "refpost"
    why = [] if elig else [x for x, c in (("not rwv", not e["rwv"]), ("reordered evidence", perm(t, e)), ("no full sweep x3", not full(e)), ("reference", e["arm"] == "refpost")) if c]
    rows.append((t, e.get("S_small"), e.get("S_all"), elig, ",".join(why), plain(e)))
rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0))
print("| rank | tag | S_small | S_all | eligible | why not | plain |")
print("|---|---|---|---|---|---|---|")
for i, (t, s, a, el, why, pl) in enumerate(rows, 1):
    f = lambda x: f"{x:.3f}" if x is not None else ""
    print(f"| {i} | {t} | {f(s)} | {f(a)} | {'yes' if el else 'no'} | {why} | {'yes' if pl else ''} |")
cands = [r for r in rows if r[3] and r[1] is not None]
champ = next((r for r in cands if r[2] is not None and r[2] < 0), cands[0] if cands else None)
plains = [r for r in rows if r[5] and r[1] is not None and full(summ[r[0]])]
print("\nchampion:", champ[0] if champ else None, "(S_small %.3f, S_all %.3f)" % (champ[1], champ[2]) if champ and champ[2] is not None else "")
print("strict plain baseline:", plains[0][0] if plains else None)
