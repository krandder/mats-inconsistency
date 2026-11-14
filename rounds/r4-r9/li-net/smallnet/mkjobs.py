"""Make job files for a wave of experiments and split them over VMs by estimated cost (LPT).

  python3 mkjobs.py --wave W1 --vms 6 --spec 'E01-k0:k0' 'E06-masked:masked:masked=1.0' ... [--ns 100,300,1000] [--seeds 0,1,2] [--test]

Each --spec is TAG:ARM[:key=val,key=val]. Writes jobs/W1-vm{i}.txt and prints the estimated busy minutes per VM.
Cost model (seconds per run on one L4 with 6 processes sharing it) comes from cost_model.json (smoke-run fit).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def est_sec(arm, n, sets, cm):
    base = cm.get("per_arm", {}).get(arm, cm.get("per_arm", {}).get("k0", 1.0))  # seconds per training step under load
    steps = cm.get("steps", {}).get(str(n), 2000)
    mult = 1.0
    for kv in sets:
        k, v = kv.split("=")
        if k in ("mask_R", "chain_R"):
            mult *= float(v)
        if k == "rbtd":
            mult *= max(1.0, float(v) / 4)
        if k == "rbtd_R":
            mult *= float(v)
        if k == "stages" and v in ("perm", "all"):
            mult *= 7 / 4
        if k == "masked" and arm == "rbtd":
            mult *= 1.25
    return base * steps * mult + cm.get("eval_overhead", {}).get(arm, 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--vms", type=int, default=1)
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--ns", default="100,300,1000,3000,10000")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--extra", default="", help="extra key=val for every job (comma list)")
    a = ap.parse_args()
    cm = json.load(open(os.path.join(HERE, "cost_model.json"))) if os.path.exists(os.path.join(HERE, "cost_model.json")) else {}
    jobs = []
    for sp in a.spec:
        parts = sp.split(":")
        tag, arm = parts[0], parts[1]
        sets = [s for s in (parts[2].split(",") if len(parts) > 2 and parts[2] else [])]
        sets += [s for s in a.extra.split(",") if s]
        for n in [int(x) for x in a.ns.split(",")]:
            for s in [int(x) for x in a.seeds.split(",")]:
                out = f"/root/runs/smallnet/{a.wave}/{tag}/{tag}-n{n}-s{s}.json"
                line = f"--arm {arm} --n {n} --seed {s} --tag {tag}" + (f" --set {' '.join(sets)}" if sets else "") + (" --test" if a.test else "") + f" --out {out}"
                jobs.append((est_sec(arm, n, sets, cm), line))
    jobs.sort(key=lambda x: -x[0])
    bins = [[0.0, []] for _ in range(a.vms)]
    for c, line in jobs:
        b = min(bins, key=lambda x: x[0])
        b[0] += c
        b[1].append(line)
    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    par = cm.get("par", 6)
    for i, (c, lines) in enumerate(bins):
        p = os.path.join(HERE, "jobs", f"{a.wave}-vm{i}.txt")
        with open(p, "w") as f:
            f.write(f"# wave {a.wave} vm {i}: {len(lines)} jobs, est {c/60:.0f} process-minutes, ~{c/60/par:.0f} wall-minutes at par {par}\n")
            f.write("\n".join(lines) + "\n")
        print(p, len(lines), "jobs", f"~{c/60/par:.0f} wall-min")
    print("total jobs", len(jobs), f"est process-hours {sum(j[0] for j in jobs)/3600:.1f}")


if __name__ == "__main__":
    main()
