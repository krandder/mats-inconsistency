"""Make wm2 job files for a wave and split them over VMs (LPT on estimated process-minutes; a WM and the forecasters
that use it always share a VM, because the checkpoint stays on the VM).

  python3 mkjobs2.py --wave W1 --task t1 --vms 6 \
      --base 'B-k0=k0' 'B-rbtd8=rbtd:rbtd=8' \
      --wm 'WMc=' --fc D-only D-mix50 'D-rbtd8' [--ns 100,300,1000,3000,10000] [--seeds 0,1,2] [--test] [--par 6]

Protocol keys added to every job (PREREG): es_boards=3000, max_steps=40000, patience 25 (N <= 1000) / 40 (N >= 3000).
Tags: baselines as given; WM bundles WMTAG (the WM's JSON) and WMTAG-ARM (forecasters). Writes jobs/<wave>-vm<i>.txt.
Per-VM priority order: WM jobs, baselines, forecasters, each largest N first (the VM's ready-queue starts a forecaster as
soon as its WM checkpoint exists).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NS_ALL = "100,300,1000,3000,10000"


def proto(n):
    return ["es_boards=3000", "max_steps=40000", f"patience={25 if n <= 1000 else 40}"]


def _get(d, kind, arm, default):
    return d.get(f"{kind}:{arm}", d.get(kind, default))


def est(kind, arm, n, cm):
    """Estimated process-minutes on a shared L4 (cost_model.json, fitted on the smoke rental): steps x sec/step + overhead."""
    steps = _get(cm["steps"], kind, arm, {}).get(str(n), 3000)
    sps = _get(cm["sec_per_step"], kind, arm, 0.1)
    return (steps * sps + _get(cm.get("overhead_sec", {}), kind, arm, 60)) / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--task", required=True, choices=["t1", "t2"])
    ap.add_argument("--vms", type=int, default=1)
    ap.add_argument("--base", nargs="*", default=[], help="TAG=ARM[:k=v,k=v] (round-1 smallnet arms)")
    ap.add_argument("--wm", nargs="*", default=[], help="WMTAG=[k=v,k=v][|ARM;ARM:k=v,...] (world-model configs, optional own forecaster list)")
    ap.add_argument("--fc", nargs="*", default=[], help="ARM[:k=v,...] forecaster arms run on every WM")
    ap.add_argument("--ns", default=NS_ALL)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--prefix", default="", help="tag prefix (e.g. F- for final runs)")
    ap.add_argument("--name", default="", help="job-file name (default: the wave); never overwrites an existing job file")
    a = ap.parse_args()
    cm = json.load(open(os.path.join(HERE, "cost_model.json")))
    ns = [int(x) for x in a.ns.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    W, T, P = a.wave, a.task, a.prefix
    root = f"/root/runs/wm2/{W}/{T}"
    tst = " --test" if a.test else ""
    units = []   # (cost, [wm lines], [base lines], [fc lines], label)
    for spec in a.base:
        tag, rest = spec.split("=", 1)
        arm, _, sets = rest.partition(":")
        sets = [s for s in sets.split(",") if s]
        for n in ns:
            for s in seeds:
                t = P + tag
                line = f"--task {T} sn --arm {arm} --n {n} --seed {s} --tag {t} --set {' '.join(sets + proto(n))}{tst} --out {root}/{t}/{t}-n{n}-s{s}.json"
                units.append((est("sn", arm, n, cm), [], [line], [], f"{t} n{n} s{s}"))
    for spec in a.wm:
        wtag, _, rest = spec.partition("=")
        wsets, bar, farms = rest.partition("|")      # WMTAG=k=v,k=v|ARM,ARM:k=v;... (per-WM forecaster arms; default --fc)
        wsets = [x for x in wsets.split(",") if x]
        fc_list = [f for f in farms.split(";") if f] if bar else a.fc
        ens = 1
        for x in list(wsets):
            if x.startswith("ens="):       # ensemble of K WMs (init_seed 0..K-1) on the same boards
                ens = int(x.split("=")[1])
                wsets.remove(x)
        for n in ns:
            for s in seeds:
                wt = P + wtag
                cks, wls = [], []
                for e in range(ens):
                    suf = f"-e{e}" if ens > 1 else ""
                    ck = f"/root/ckpt/{W}-{T}-{wt}{suf}-n{n}-s{s}.pt"
                    es_ = [f"init_seed={e}"] if ens > 1 else []
                    wls.append((n, f"--task {T} wm --n {n} --seed {s} --wmtag {wt}{suf} --set {' '.join(wsets + es_ + proto(n))} --ckpt {ck} --out {root}/{wt}/{wt}{suf}-n{n}-s{s}.json"))
                    cks.append(ck)
                ck = ",".join(cks)
                c = est("wm", wtag, n, cm) * ens
                fls = []
                for f in fc_list:
                    arm, _, fsets = f.partition(":")
                    fsets = [x for x in fsets.split(",") if x]
                    t = f"{wt}-{arm}" + ("-" + "-".join(x.replace("=", "") for x in fsets) if fsets else "")
                    fls.append(((n, est("fc", arm, n, cm)), f"--task {T} fc --n {n} --seed {s} --arm {arm} --tag {t} --wm-ckpt {ck} --set {' '.join(fsets + proto(n))}{tst} --out {root}/{t}/{t}-n{n}-s{s}.json"))
                    c += est("fc", arm, n, cm)
                units.append((c, wls, [], fls, f"{wt} n{n} s{s}"))
    units.sort(key=lambda u: -u[0])
    bins = [[0.0, [], [], []] for _ in range(a.vms)]
    for c, wl, bl, fl, _ in units:
        b = min(bins, key=lambda x: x[0])
        b[0] += c
        b[1] += wl
        b[2] += bl
        b[3] += fl
    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    tot = 0
    for i, (c, wl, bl, fl) in enumerate(bins):
        # priority order for the VM's ready-queue (gpu/remote_sched.py): WMs (largest N first), baselines (largest N first),
        # forecasters (largest N first; each starts as soon as its WM checkpoint exists)
        nkey = lambda l: -int(l.split("--n ")[1].split()[0])
        lines = [l for _, l in sorted(wl, key=lambda x: -x[0])] + sorted(bl, key=nkey) + [l for _, l in sorted(fl, key=lambda x: (-x[0][0], -x[0][1]))]
        tot += len(lines)
        p = os.path.join(HERE, "jobs", f"{a.name or W}-{T}-vm{i}.txt")
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p} (use --name)")
        with open(p, "w") as fh:
            fh.write(f"# wave {W} task {T} vm {i}: {len(lines)} jobs, est {c:.0f} process-minutes, ~{c / a.par:.0f} wall-minutes at par {a.par}\n")
            fh.write("\n".join(lines) + "\n")
        print(p, len(lines), "jobs", f"~{c / a.par:.0f} wall-min (est {c:.0f} process-min)")
    print("total jobs", tot, f"est process-hours {sum(b[0] for b in bins) / 60:.1f}")


if __name__ == "__main__":
    main()
