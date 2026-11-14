"""Sensitivity check (post hoc, PREREG Addendum I): at maze size 99 some stored float16 values overflowed to inf
(realized K and head outputs above 65504, mostly at H=64 and cap). The true value is >= 65504, so replacing inf by
65504 gives the other bound on every halting decision (the stored inf never stops there; 65504 stops earliest).
Reruns the size-99 maze analysis with that replacement and writes the result next to the main one.

    python sens_f16.py --maze results/vm/trader-cat-p1-7/maze --out results/sensitivity_f16_n99.json --nboot 200 --procs 5
"""
import argparse, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze
from acore import load_pool as _load_pool

F16MAX = np.float16(65504)


def load_pool_capped(dirn, name):
    O, E, M, info = _load_pool(dirn, name)
    for D in (O, E):
        for k, v in D.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float16:
                D[k] = np.where(np.isposinf(v), F16MAX, v)
    return O, E, M, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    analyze.PROCS = max(1, min(6, a.procs))
    analyze.load_pool = load_pool_capped
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "maze"))
        for f in ("obs_eval_n99.npz", "eval_eval_n99.npz", "obs_dev_n99.npz", "eval_dev_n99.npz"):
            os.symlink(os.path.abspath(os.path.join(a.maze, f)), os.path.join(td, "maze", f))
        r, _ = analyze.analyze_maze(td, a.nboot, np.random.default_rng(a.seed))
    r["note"] = "size 99 only; float16 +inf replaced by 65504 in stored head outputs and realized K; " \
                f"nboot {a.nboot}"
    json.dump(r, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
