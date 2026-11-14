"""Bracketing queue: for every arm whose tuned value sits at a grid edge (results/hp_select_{task}.json), add the
two 3x extension values on that side (select_hp.py admits the second only if the first is chosen).
Usage: bracket_queue.py --task reach|dist --seeds 100,101,102 [--grace 500] > queue file"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_queue import fmt, line, order  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="reach")
ap.add_argument("--seeds", default="100,101,102")
ap.add_argument("--grace", type=int, default=500)
a = ap.parse_args()
sel = json.load(open(os.path.join(HERE, "results", f"hp_select_{a.task}.json")))
seeds = [int(s) for s in a.seeds.split(",")]
ch = sel["choice"]
out = []
for arm, c in ch.items():
    if not c.get("edge") or c.get("extensions", 0) >= 2:
        continue
    p = [k for k in ("fee", "lam", "tau") if k in c][0]
    v = c[p]
    f = 1 / 3 if c["edge"] == "low" else 3.0
    vals = [float("%.3g" % (v * f)), float("%.3g" % (v * f * f))] if c.get("extensions", 0) == 0 else \
        [float("%.3g" % (v * f))]
    for x in vals:
        for s in seeds:
            kw = {p: x}
            if arm == "BP-L1-D-cm":
                kw["lam"] = ch["BP-L1-cm"]["lam"]
                tag = f"c{fmt(x)}-lam{fmt(kw['lam'])}"
            elif arm == "STG-L1-cm":
                kw["tau"] = ch["STG"]["tau"]
                tag = f"lam{fmt(x)}-tau{fmt(kw['tau'])}"
            else:
                tag = {"fee": "c", "lam": "lam", "tau": "tau"}[p] + fmt(x)
            out.append(line(arm, s, task=a.task, grace=a.grace, tag=tag, **kw))
    print(f"# {arm} {p}={v:g} edge={c['edge']} -> {vals}", file=sys.stderr)
print("\n".join(order(out)))
