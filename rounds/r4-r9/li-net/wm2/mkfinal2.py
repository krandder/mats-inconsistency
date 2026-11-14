"""Final-phase part 2 job files (after the champion is fixed on validation; PREREG "Final TEST protocol").

  python3 mkfinal2.py --champ WMr-D-rbtd8 [--small-vms 6] [--name F2]
T1: F-<champ> bundles (its WM + its forecaster), seeds 0-4, all 5 N, --test.
T2: one twin WM per (N, seed) (wm_twin=1: one board network, an L0 probe network and a board-blind one; PREREG deviation
18:00Z), with the champion's forecaster and the dream-only forecaster on each twin (tags F-<WM>-<arm>, F-<WM>b-<arm>).
Layout: each seed's N = 10000 bundles (T1 and T2) get their own rental (jobs/<name>-big-vm<seed>.txt); the N <= 3000
bundles of both tasks are spread over --small-vms rentals (jobs/<name>-small-vm<i>.txt), longest first.
(F-B-k0 and F-B-rbtd8 for both tasks run in final part 1.)
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WMSPEC = {"WMc": [], "WMs": ["wm_split=1"], "WMr": ["wm_split=1", "wm_board=rbtd", "wm_rbtd=8"], "WMe": ["wm_split=1"]}
ENS = {"WMe": 3}
NS = [100, 300, 1000, 3000, 10000]


def proto(n):
    return ["es_boards=3000", "max_steps=40000", f"patience={25 if n <= 1000 else 40}"]


def parse(tag):
    wm, arm = tag.split("-", 1)
    sets = []
    if arm.endswith("-rho0.75"):
        arm, sets = arm[: -len("-rho0.75")], ["rho=0.75"]
    return wm, arm, sets


def bundle(task, wm, arms, n, s, twin):
    """Lines and a cost guess (process-minutes) for one (task, N, seed) bundle."""
    W, root = "F", f"/root/runs/wm2/F/{task}"
    wt = f"F-{wm}"
    k = ENS.get(wm, 1)
    wls, cks, ckbs = [], [], []
    for e in range(k):
        suf = f"-e{e}" if k > 1 else ""
        ck = f"/root/ckpt/{W}-{task}-{wt}{suf}-n{n}-s{s}.pt"
        sets = WMSPEC[wm] + ([f"init_seed={e}"] if k > 1 else []) + (["wm_twin=1"] if twin else []) + proto(n)
        wls.append(f"--task {task} wm --n {n} --seed {s} --wmtag {wt}{suf} --set {' '.join(sets)} --ckpt {ck} --out {root}/{wt}{suf}/{wt}{suf}-n{n}-s{s}.json")
        cks.append(ck)
        ckbs.append(ck + ".blind.pt")
    fls = []
    for arm, asets in arms:
        suf = "-" + "-".join(x.replace("=", "") for x in asets) if asets else ""
        for b, cl in ((False, cks), (True, ckbs)) if twin else ((False, cks),):
            t = f"{wt}{'b' if b else ''}-{arm}{suf}"
            fls.append(f"--task {task} fc --n {n} --seed {s} --arm {arm} --tag {t} --wm-ckpt {','.join(cl)} --set {' '.join(asets + proto(n))} --test --out {root}/{t}/{t}-n{n}-s{s}.json")
    cm = {100: 60, 300: 80, 1000: 150, 3000: 220, 10000: 300}[n] * k * (1.3 if twin else 1.0) * (2.5 if wm == "WMr" else 1.0)
    cm += sum({100: 15, 300: 25, 1000: 60, 3000: 120, 10000: 200}[n] * (2.5 if a == "D-rbtd8" else 0.6) for a, _ in arms) * (2 if twin else 1)
    return wls, fls, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ", required=True)
    ap.add_argument("--small-vms", type=int, default=6)
    ap.add_argument("--name", default="F2")
    a = ap.parse_args()
    wm, arm, sets = parse(a.champ)
    if wm not in WMSPEC:
        raise SystemExit(f"champion {a.champ} is not a world-model arm (pass the best WM arm explicitly)")
    t1_arms = [(arm, sets)]
    t2_arms = [(arm, sets)] + ([] if (arm == "D-only" and not sets) else [("D-only", [])])
    big, small = {}, []
    for s in range(5):
        for task, arms, twin in (("t1", t1_arms, False), ("t2", t2_arms, True)):
            for n in NS:
                wls, fls, c = bundle(task, wm, arms, n, s, twin)
                if n == 10000:
                    big.setdefault(s, []).append((wls, fls))
                else:
                    small.append((c, n, wls, fls))
    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    out = []
    for s, bs in big.items():
        lines = [l for w, _ in bs for l in w] + [l for _, f in bs for l in f]
        out.append((f"{a.name}-big-vm{s}", lines))
    small.sort(key=lambda x: -x[0])
    bins = [[0.0, [], []] for _ in range(a.small_vms)]
    for c, n, wls, fls in small:
        b = min(bins, key=lambda x: x[0])
        b[0] += c
        b[1] += [(n, l) for l in wls]
        b[2] += [(n, l) for l in fls]
    for i, (c, wl, fl) in enumerate(bins):
        lines = [l for _, l in sorted(wl, key=lambda x: -x[0])] + [l for _, l in sorted(fl, key=lambda x: -x[0])]
        out.append((f"{a.name}-small-vm{i}", lines))
    for name, lines in out:
        p = os.path.join(HERE, "jobs", name + ".txt")
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        with open(p, "w") as fh:
            fh.write(f"# final part 2 ({a.champ}): {len(lines)} jobs\n" + "\n".join(lines) + "\n")
        print(p, len(lines))
    json.dump({"champion": a.champ, "files": [n for n, _ in out]}, open(os.path.join(HERE, "jobs", f"{a.name}-manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
