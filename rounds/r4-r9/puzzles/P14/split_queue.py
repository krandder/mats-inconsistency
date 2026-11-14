"""Split a queue across VMs by estimated cost (longest job first, greedy onto the least-loaded VM).
Cost of a line = steps x single-thread s/step from results/logs/timing.log (+ a fixed probe/eval overhead).
Usage: split_queue.py QUEUE N OUT_PREFIX [--overhead-s 300]  -> OUT_PREFIX_{0..N-1}.txt, longest jobs first."""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("queue")
ap.add_argument("n", type=int)
ap.add_argument("prefix")
ap.add_argument("--overhead-s", type=float, default=300.0)
a = ap.parse_args()
tim = {}
for l in open(os.path.join(HERE, "results", "logs", "timing.log")):
    if l.startswith("{"):
        d = json.loads(l)
        tim[d["arm"]] = d
lines = [l.rstrip("\n") for l in open(a.queue) if l.strip()]


def cost(l):
    tok = l.split()
    arm = tok[tok.index("--arm") + 1]
    mode = tok[tok.index("--mode") + 1]
    t = tim[arm]
    steps = t["steps_to_B"] if arm not in ("ORACLE", "P-LOCO", "P-W") else 6000
    return steps * t["s_per_step"] + a.overhead_s * (2 if mode == "eval" else 1)


jobs = sorted(lines, key=cost, reverse=True)
load = [0.0] * a.n
parts = [[] for _ in range(a.n)]
for l in jobs:
    i = min(range(a.n), key=lambda k: load[k])
    parts[i].append(l)
    load[i] += cost(l)
for i, p in enumerate(parts):
    fn = f"{a.prefix}_{i}.txt"
    open(fn, "w").write("\n".join(p) + "\n")
    print(fn, len(p), "jobs", f"{load[i] / 60:.0f} single-thread min")
