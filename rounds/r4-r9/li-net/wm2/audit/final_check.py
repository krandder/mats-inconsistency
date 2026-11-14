"""wm2 audit, phase 2: recompute the TEST tables and decisions from the raw F JSONs (own code; does not import
analyze2.py or final2.py). Refuses to read any TEST field until results/test-t1-final.json exists (the one TEST analysis).

  PYTHONDONTWRITEBYTECODE=1 python3 audit/final_check.py

[F0] completeness: every (tag, N) has seeds 0-4, each run's stop reason, protocol tuple, --test scored on 5000 boards.
[F1] T1: KL x1e3 per tag and N (mean +- 95% t half-width), paired differences champion - B-rbtd8 and - B-k0
     (t-interval, 5 seeds), the PREREG decision, outcome log-loss differences; comparison with results/test-t1-final.json.
[F2] T2: the same per tag; the selection-unaware reference KL on the TEST boards (from the data cache); rule (b)(i) per
     arm (KL minus reference, 95% t); rule (b)(ii) paired vs the board-blind twin; rule (a) location information of the
     final L0 WMs (validation diagnostic, 5 seeds); comparison with results/test-t2-final.json.
[F3] compute per board (all phases, to the best checkpoint) for the final runs.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

R = os.environ.get("AUDIT_F", "/tmp/wm2/runs/F")
RES = os.environ.get("AUDIT_RES", "/home/agent/workspaces/eurisko-study/li-net/wm2/results")
NS = [100, 300, 1000, 3000, 10000]
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
PROTO = ["lr", "ema", "eval_every", "max_steps", "patience", "es_boards", "B", "warmup", "wd", "clip"]


def tci(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(TQ[len(x)] * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def kl_bern(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def load(task):
    fc, wm = [], []
    for p in sorted(glob.glob(os.path.join(R, task, "*", "*.json"))):
        r = json.load(open(p))
        r["_path"] = p
        (wm if r.get("kind") == "wm" else fc).append(r)
    return fc, wm


def table(fc):
    t = defaultdict(dict)
    for r in fc:
        tag = r["tag"][2:] if r["tag"].startswith("F-") else r["tag"]
        assert r["seed"] not in t[(tag, r["n"])], f"duplicate {tag} {r['n']} {r['seed']}"
        t[(tag, r["n"])][r["seed"]] = r
    return t


def paired(t, a, b, n, key="kl"):
    A, B = t.get((a, n), {}), t.get((b, n), {})
    s = sorted(set(A) & set(B))
    return [A[x]["test"][key] - B[x]["test"][key] for x in s], s


def per_tag(t, tags):
    out = {}
    for tag in tags:
        cells = []
        for n in NS:
            v = t.get((tag, n), {})
            if not v:
                cells.append("-")
                continue
            m, h = tci([r["test"]["kl"] for r in v.values()])
            cells.append(f"{1e3 * m:.3f} +- {1e3 * h:.3f} [{len(v)}]")
            out[(tag, n)] = (m, h, sorted(v))
        print(f"   {tag:16s} " + " | ".join(cells))
    return out


def main():
    if not os.path.exists(os.path.join(RES, "test-t1-final.json")):
        raise SystemExit("results/test-t1-final.json does not exist yet: TEST fields stay closed (phase 1)")
    ok_all = True
    for task in ("t1", "t2"):
        fc, wm = load(task)
        t = table(fc)
        print(f"\n[F0] {task}: {len(fc)} forecaster/baseline JSONs, {len(wm)} WM JSONs")
        tags = sorted({k[0] for k in t})
        for tag in tags:
            for n in NS:
                v = t.get((tag, n), {})
                if sorted(v) != [0, 1, 2, 3, 4]:
                    ok_all = False
                    print(f"   INCOMPLETE {tag} N={n}: seeds {sorted(v)}")
        stops = defaultdict(int)
        for r in fc:
            stops[(r["tag"], r["stop_reason"])] += 1
        print("   stop reasons:", dict(stops))
        nonp = [(r["tag"], r["n"], r["seed"], r["stop_reason"], r["best_step"], r["steps_run"]) for r in fc if r["stop_reason"] != "patience"]
        print("   forecasters/baselines not stopped by patience:", nonp)
        wnonp = [(r["wmtag"], r["n"], r["seed"], r["stop_reason"], r["best_step"]) for r in wm if r["stop_reason"] != "patience/patience"]
        print(f"   WMs not stopped by patience on both networks: {len(wnonp)} {sorted(wnonp)[:12]}")
        proto = defaultdict(set)
        for r in fc + wm:
            proto[r["n"]].add(tuple(r["cfg"].get(k) for k in PROTO))
        print("   protocol tuples per N:", {n: sorted(v) for n, v in proto.items()})
        nb = {r["test"]["nb"] for r in fc if "test" in r}
        miss = [r["_path"] for r in fc if "test" not in r]
        print(f"   TEST boards per run: {nb}; runs without a test field: {len(miss)}")
        devs = defaultdict(int)
        for r in fc:
            devs[r.get("gpu")] += 1
        print(f"   devices: {dict(devs)}")
        if task == "t1":
            print("\n[F1] T1 TEST KL x1e3 (mean +- 95% t half-width [seeds])")
            pt = per_tag(t, ["WMr-D-rbtd8", "B-rbtd8", "B-k0"])
            print("   paired differences (champion minus base), KL x1e3, 95% t (df 4); geometric-mean ratio; outcome log-loss x1e4:")
            dec = {}
            for base in ("B-rbtd8", "B-k0"):
                below = []
                for n in NS:
                    d, s = paired(t, "WMr-D-rbtd8", base, n)
                    m, h = tci(d)
                    dl, _ = paired(t, "WMr-D-rbtd8", base, n, "logloss")
                    ml, hl = tci(dl)
                    rr = [math.log(t[("WMr-D-rbtd8", n)][x]["test"]["kl"] / t[(base, n)][x]["test"]["kl"]) for x in s]
                    mr, hr = tci(rr)
                    below.append(m + h < 0)
                    print(f"    vs {base} N={n}: {1e3 * m:+.3f} +- {1e3 * h:.3f} (upper {1e3 * (m + h):+.3f}; per seed {[round(1e3 * x, 3) for x in d]}) "
                          f"ratio x{math.exp(mr):.2f} [{math.exp(mr - hr):.2f}, {math.exp(mr + hr):.2f}]; dLL x1e4 {1e4 * ml:+.2f} +- {1e4 * hl:.2f}; seeds {s}")
                dec[base] = below
            print(f"   decision (PREREG 2): beats rbtd8 at every N on TEST = {all(dec['B-rbtd8'])}; fails at N = "
                  f"{[n for n, b in zip(NS, dec['B-rbtd8']) if not b]}; beats k0 at every N = {all(dec['B-k0'])}")
            ref = json.load(open(os.path.join(RES, "test-t1-final.json")))
            diffs = []
            for tag in ("WMr-D-rbtd8", "B-rbtd8", "B-k0"):
                for n in NS:
                    e = ref["per_tag"].get(tag, {}).get(str(n))
                    if e and (tag, n) in pt:
                        diffs.append(abs(e["kl_mean"] - pt[(tag, n)][0]) + abs(e["kl_ci95"] - pt[(tag, n)][1]))
            print(f"   results/test-t1-final.json vs this recomputation: max |difference| in mean + half-width {max(diffs) if diffs else float('nan'):.2e} over {len(diffs)} cells; "
                  f"its decision {ref.get('decision')}")
        else:
            print("\n[F2] T2 TEST KL x1e3 to the selection-aware posterior (mean +- 95% t half-width [seeds])")
            tags2 = ["WMr-D-rbtd8", "WMrb-D-rbtd8", "WMr-D-only", "WMrb-D-only", "B-rbtd8", "B-k0"]
            pt = per_tag(t, tags2)
            d = np.load("/tmp/wm2/data-t2/test-n5000.post.npz")
            k = kl_bern(d["post"], d["post_unaware"])
            ref_kl = float(k.mean())
            print(f"   selection-unaware exact posterior on the TEST boards: KL {1e3 * ref_kl:.3f}e-3 (per stage {[round(1e3 * x, 2) for x in k.mean((0, 2))]})")
            for tag in ("WMr-D-rbtd8", "WMr-D-only", "WMrb-D-rbtd8", "WMrb-D-only"):
                cells = []
                for n in NS:
                    v = [r["test"]["kl"] - ref_kl for r in t.get((tag, n), {}).values()]
                    if len(v) >= 2:
                        m, h = tci(v)
                        cells.append(f"N={n}: {1e3 * m:+.2f} +- {1e3 * h:.2f}{' below' if m + h < 0 else ' NOT below'}")
                print(f"   (b)(i) {tag} minus reference: " + "; ".join(cells))
            for a_, b_ in (("WMr-D-rbtd8", "WMrb-D-rbtd8"), ("WMr-D-only", "WMrb-D-only")):
                cells = []
                for n in NS:
                    dd, s = paired(t, a_, b_, n)
                    if len(dd) >= 2:
                        m, h = tci(dd)
                        cells.append(f"N={n}: {1e3 * m:+.2f} +- {1e3 * h:.2f}{' below' if m + h < 0 else ' NOT below'}")
                print(f"   (b)(ii) {a_} - {b_} (paired): " + "; ".join(cells))
            # rule (a): location information of the final L0 WMs (validation diagnostic)
            li = defaultdict(dict)
            for w in wm:
                li[(w["wmtag"], w["n"])][w["seed"]] = w["val"].get("loc_info_nats_per_episode")
            for wt in ("F-WMr", "F-WMrb"):
                cells = []
                for n in NS:
                    v = [x for x in li.get((wt, n), {}).values() if x is not None]
                    if len(v) >= 2:
                        m, h = tci(v)
                        cells.append(f"N={n}: {m:.3f} +- {h:.3f} [{len(v)}]{' >0' if m - h > 0 else ''}")
                print(f"   (a) location information, {wt}: " + "; ".join(cells))
            # extra: champion pipeline vs rbtd8 on T2
            for base in ("B-rbtd8",):
                cells = []
                for n in NS:
                    dd, s = paired(t, "WMr-D-rbtd8", base, n)
                    if len(dd) >= 2:
                        m, h = tci(dd)
                        cells.append(f"N={n}: {1e3 * m:+.2f} +- {1e3 * h:.2f}")
                print(f"   WMr-D-rbtd8 - {base} on T2 (paired, not a pre-registered rule): " + "; ".join(cells))
            p2 = os.path.join(RES, "test-t2-final.json")
            if os.path.exists(p2):
                ref = json.load(open(p2))
                diffs = []
                for tag in tags2:
                    for n in NS:
                        e = ref["per_tag"].get(tag, {}).get(str(n))
                        if e and (tag, n) in pt:
                            diffs.append(abs(e["kl_mean"] - pt[(tag, n)][0]) + abs(e["kl_ci95"] - pt[(tag, n)][1]))
                print(f"   results/test-t2-final.json vs this recomputation: max |difference| {max(diffs) if diffs else float('nan'):.2e} over {len(diffs)} cells; "
                      f"its reference {ref.get('unaware_exact_kl_test')}; its decision_t2 keys {list(ref.get('decision_t2', {}))}")
        print(f"\n[F3] {task}: FLOPs per training board to the best checkpoint, all phases, seed mean (x k0)")
        for tag in sorted({k[0] for k in t}):
            cells = []
            for n in NS:
                v = t.get((tag, n), {})
                k0 = t.get(("B-k0", n), {})
                if v and k0:
                    a_ = np.mean([r["flops_per_board_at_best"] for r in v.values()])
                    b_ = np.mean([r["flops_per_board_at_best"] for r in k0.values()])
                    cells.append(f"{a_:.2e} ({a_ / b_:.0f}x)")
            print(f"   {tag:16s} " + " / ".join(cells))
    print("\nall (tag, N) complete with seeds 0-4:", ok_all)


if __name__ == "__main__":
    main()
