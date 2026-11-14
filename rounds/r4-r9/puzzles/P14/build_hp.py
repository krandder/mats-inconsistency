"""Assemble results/hp.json (the format make_queue.py reads) from results/hp_select_reach.json and, if present,
results/hp_select_dist.json. hp[arm] = {param: value}; hp["BP*"]; hp["endpoint_switch"]; hp["tdist"][arm].
Usage: build_hp.py [--allow-edge]  (refuses while a chosen value sits at a grid edge, unless --allow-edge)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
sel = json.load(open(os.path.join(R, "hp_select_reach.json")))
hp = {}
edges = []
for arm, ch in sel["choice"].items():
    p = [k for k in ("fee", "lam", "tau") if k in ch][0]
    hp[arm] = {p: ch[p]}
    if ch.get("edge"):
        edges.append(f"reach {arm} {p}={ch[p]:g} ({ch['edge']})")
hp["BP*"] = sel["BP*"]
hp["endpoint_switch"] = sel["endpoint_switch"]
fd = os.path.join(R, "hp_select_dist.json")
if os.path.exists(fd):
    seld = json.load(open(fd))
    hp["tdist"] = {}
    for arm, ch in seld["choice"].items():
        p = [k for k in ("fee", "lam", "tau") if k in ch][0]
        hp["tdist"][arm] = {p: ch[p]}
        if ch.get("edge"):
            edges.append(f"dist {arm} {p}={ch[p]:g} ({ch['edge']})")
if edges:
    print("edge choices:", "; ".join(edges))
    if "--allow-edge" not in sys.argv:
        raise SystemExit("edge choices remain: bracket them first (or pass --allow-edge after two extensions)")
json.dump(hp, open(os.path.join(R, "hp.json"), "w"), indent=1)
print(json.dumps(hp, indent=1))
