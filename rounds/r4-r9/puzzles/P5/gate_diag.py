"""Gate diagnostics (used when G0 decides the run): accuracy on 9x9 validation by S-T path length for the gate runs
and the diagnostic runs, at the validation-best iteration. Prints JSON; imported by summarize_gate.py."""
import glob
import json
import os
import sys

import numpy as np

import mazes

HERE = os.path.dirname(os.path.abspath(__file__))
VM = os.path.join(HERE, "results", "vm")
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 200)]


def by_dist(name, sname="val9"):
    r = json.load(open(f"{VM}/eval/{name}.json"))
    z = np.load(f"{VM}/eval/{name}.npz")
    s = mazes.load_set(sname)
    y = (s["y"] * s["mask"]).sum((1, 2)) > 0.5
    d = s["dist"]
    it = r.get("best_it") or max(int(k) for k in r["sets"][sname]["it"])
    lg = z[f"{sname}@{it}"].astype(np.float32)
    c = (lg > 0) == y[:len(lg)]
    d = d[:len(lg)]
    out = dict(best_it=it, acc=float(c.mean()), n=int(len(c)), buckets={})
    for lo, hi in BUCKETS:
        m = (d >= lo) & (d <= hi)
        if m.any():
            out["buckets"][f"{lo}-{hi}"] = dict(acc=float(c[m].mean()), n=int(m.sum()))
    return out


def main():
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{VM}/eval/g0_*.json") + glob.glob(f"{VM}/eval/diag_*.json"))
    res = {}
    for n in names:
        try:
            res[n] = {sn: by_dist(n, sn) for sn in ("val9", "val13") if os.path.exists(f"{VM}/eval/{n}.npz") and f"{sn}@64" in np.load(f"{VM}/eval/{n}.npz")}
        except Exception as ex:
            res[n] = dict(error=str(ex))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
