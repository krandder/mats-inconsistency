"""Compare every numeric leaf of the committed T1 outputs with a clean rerun.

Usage: python3 compare_rerun.py RERUN_ROOT [--out FILE]
RERUN_ROOT holds w1/out, w2/results, w3/out produced by rerunning the scripts in a copy
(see THEORY.md, "Lead check"). Timing fields (names containing 'sec', 'time', 'wall') and
sha/hash strings are skipped. Reports, per file, the number of numeric leaves compared,
the max relative difference, and whether all leaves agree to rel 1e-9 (abs 1e-12).
"""
import json, math, os, sys, csv

T1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("sec", "time", "wall", "sha", "hash", "elapsed")


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if any(s in k.lower() for s in SKIP):
                continue
            yield from leaves(v, p + "." + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    elif isinstance(x, bool) or x is None or isinstance(x, str):
        yield p, x
    elif isinstance(x, (int, float)):
        yield p, float(x)


def load(path):
    if path.endswith(".json"):
        return dict(leaves(json.load(open(path))))
    rows = list(csv.reader(open(path)))
    out = {}
    for i, r in enumerate(rows[1:]):
        for j, c in enumerate(r):
            try:
                out["r%d.%s" % (i, rows[0][j])] = float(c)
            except ValueError:
                out["r%d.%s" % (i, rows[0][j])] = c
    return out


def compare(a, b):
    n, worst, worst_key, missing, mism = 0, 0.0, None, [], []
    for k, va in a.items():
        if k not in b:
            missing.append(k); continue
        vb = b[k]
        if isinstance(va, float) and isinstance(vb, float):
            n += 1
            if math.isnan(va) and math.isnan(vb):
                continue
            d = abs(va - vb)
            rel = 0.0 if d <= 1e-12 else d / max(abs(va), abs(vb), 1e-300)
            if rel > worst:
                worst, worst_key = rel, k
        elif va != vb:
            mism.append(k)
    return n, worst, worst_key, missing, mism


def main():
    root = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    pairs = []
    for sub in ("w1/out", "w2/results", "w3/out"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".json", ".csv")) and os.path.exists(os.path.join(T1, sub, f)):
                pairs.append((sub + "/" + f, os.path.join(T1, sub, f), os.path.join(d, f)))
    res = {}
    for name, orig, new in pairs:
        n, worst, wk, missing, mism = compare(load(orig), load(new))
        res[name] = dict(n_numeric=n, max_rel_diff=worst, worst_key=wk, missing_in_rerun=len(missing),
                         nonnumeric_mismatch=mism[:5], agree=(worst <= 1e-9 and not missing and not mism))
        print("%-32s n=%6d max_rel=%.3g agree=%s %s" % (name, n, worst, res[name]["agree"], wk or ""))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
