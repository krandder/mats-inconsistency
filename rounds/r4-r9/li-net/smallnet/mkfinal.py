"""Final-phase jobs for one or more loop tags (final protocol: es_boards=3000, patience 40 at N>=3000, --test).

  python3 mkfinal.py --wave F2 --vms 4 --tags E21-rbtd+td,E16-k0wd5 [--n30k E21-rbtd+td]
Reads each tag's arm and --set overrides from its loop result JSONs (/tmp/smallnet/runs/W*/TAG/*.json) and writes
jobs/<wave>-vm<i>.txt with tags F-<tag>, seeds 0-4, N = 100..10000 (+ N=30000, seeds 0-2, for --n30k tags).
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def cfg_of(tag):
    p = sorted(glob.glob(f"/tmp/smallnet/runs/W*/{tag}/*.json"))[0]
    r = json.load(open(p))
    return r["arm"], [s for s in r.get("set", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--vms", type=int, default=4)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--n30k", default="")
    a = ap.parse_args()
    cm = json.load(open(os.path.join(HERE, "cost_model.json")))
    jobs = []
    for tag in a.tags.split(","):
        arm, sets = cfg_of(tag)
        mult = 1.0
        for kv in sets:
            k, v = kv.split("=")
            if k in ("rbtd_R", "mask_R"):
                mult *= float(v)
            if k == "rbtd":
                mult *= max(1.0, float(v) / 4)
            if k == "teacher" and v == "1":
                mult *= 1.5
            if k == "stages" and v in ("all", "perm"):
                mult *= 7 / 4
        ns = [(n, s) for s in range(5) for n in (100, 300, 1000, 3000, 10000)]
        if tag in a.n30k.split(","):
            ns += [(30000, s) for s in range(3)]
        for n, s in ns:
            st = sets + ["es_boards=3000"] + (["patience=40"] if n >= 3000 else [])
            ft = f"F-{tag}"
            out = f"/root/runs/smallnet/F/{ft}/{ft}-n{n}-s{s}.json"
            line = f"--arm {arm} --n {n} --seed {s} --tag {ft} --set {' '.join(st)} --test --out {out}"
            steps = cm["steps"][str(min(n, 10000))] * (1.6 if n >= 3000 else 1.2) * (2 if n > 10000 else 1)
            jobs.append((cm["per_arm"].get(arm, 0.1) * mult * steps, line))
    jobs.sort(key=lambda x: -x[0])
    bins = [[0.0, []] for _ in range(a.vms)]
    for c, l in jobs:
        b = min(bins, key=lambda x: x[0])
        b[0] += c
        b[1].append(l)
    for i, (c, ls) in enumerate(bins):
        p = os.path.join(HERE, "jobs", f"{a.wave}-vm{i}.txt")
        open(p, "w").write(f"# final {a.wave}: {len(ls)} jobs, ~{c/3600:.1f} process-h\n" + "\n".join(ls) + "\n")
        print(p, len(ls), f"{c/3600:.1f} process-h; longest job {max(x[0] for x in jobs)/60:.0f} min")


if __name__ == "__main__":
    main()
