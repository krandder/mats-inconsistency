"""Analyse the synced P5 results (results/vm): per-seed accuracy at the validation-chosen iteration, seed-level Welch
95% CIs for the pre-registered differences, maze-level paired bootstrap, McNemar 6400 vs 64,000, depth bins, ledger,
credit-quality and write-structure diagnostics, memory against D, FLOPs; checks the success and falsification criteria.
Writes results/summary.json and results/chart.json."""
import glob
import json
import math
import os

import numpy as np
from scipy import stats

import mazes

HERE = os.path.dirname(os.path.abspath(__file__))
VM = os.path.join(HERE, "results", "vm")
REC = [64, 128, 256, 640, 1280, 2560, 6400]
TESTS = ["test9", "test13", "test19", "test33", "test59", "long19", "long33", "long59", "lat30", "lat100"]
BIN_NAMES = ["<=1x", "1-3x", "3-10x", "10-30x", "30x+"]
MAIN = ["A1", "A1w", "A2", "A2t", "A3", "A4", "A5", "A6", "A7", "A8", "A10", "A7p", "A7f"]
LEDGER_KEYS = ["capped", "roy_scaled", "subsidy", "roy_to_writers", "gini", "neg_wealth", "roy_onpath_share", "cons_err"]
_Y = {}


def jl(p):
    return json.load(open(p))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return dict(diff=d, lo=d, hi=d, na=len(a), nb=len(b))
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return dict(diff=d, lo=d - h, hi=d + h, na=len(a), nb=len(b))


def mean_ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return None
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return dict(mean=m, lo=m - h, hi=m + h, n=len(v), per_seed=[float(x) for x in v])


def ytrue(sname):
    if sname not in _Y:
        s = mazes.load_set(sname)
        _Y[sname] = ((s["y"] * s["mask"]).sum((1, 2)) > 0.5, s["dist"])
    return _Y[sname]


def runs(arm, sfx):
    out = []
    for p in sorted(glob.glob(f"{VM}/eval/{arm}{sfx}_s[0-9].json")):
        r = jl(p)
        if r["arm"] == arm and not r["args"].get("dense"):
            out.append((os.path.basename(p)[:-5], r))
    return out


def correct(name, r, sname, it):
    """Per-maze correctness of one run on one set at iteration it (from the float16 logits)."""
    z = np.load(f"{VM}/eval/{name}.npz")
    key = f"{sname}@{it}"
    if key not in z:
        return None
    lg = z[key].astype(np.float32)
    y, dist = ytrue(sname)
    return (lg > 0) == y[:len(lg)], dist[:len(lg)]


def it_get(ent, it):
    return ent["it"].get(str(it)) or ent["it"].get(it)


def main():
    st = jl(f"{VM}/driver_state.json") if os.path.exists(f"{VM}/driver_state.json") else {}
    meta = jl(os.path.join(mazes.DATA, "meta.json"))
    L = meta["L_train"]
    sfx = "c" if st.get("g0", {}).get("mode") == "curriculum" else ""
    S = dict(driver_state=st, L_train=L, sfx=sfx, arms={}, diffs={}, bootstrap={}, stability={}, bins={}, validity={},
             ledger={}, credit={}, writes={}, occupancy={}, probes={}, memdepth={}, flops=None, criteria={})
    if os.path.exists(os.path.join(HERE, "results", "tests.json")):
        S["tests"] = jl(os.path.join(HERE, "results", "tests.json"))
    # ------------------------------------------------ per-arm, per-seed accuracy
    per = {}  # arm -> set -> list over seeds (acc at best_it)
    corr_cache = {}
    for arm in MAIN:
        rs = runs(arm, sfx)
        if not rs:
            continue
        A = dict(seeds=[], best_it=[], sets={}, sets_at_max={}, nll={}, ece={})
        for name, r in rs:
            A["seeds"].append(r["seed"])
            bi = r.get("best_it") or 64
            A["best_it"].append(bi)
            for sname in TESTS + ["val9", "val13"]:
                if sname not in r["sets"]:
                    continue
                ent = r["sets"][sname]
                m = it_get(ent, bi)
                A["sets"].setdefault(sname, []).append(m["acc"])
                A["nll"].setdefault(sname, []).append(m["nll"])
                A["ece"].setdefault(sname, []).append(m["ece"])
                last = max(int(k) for k in ent["it"])
                A["sets_at_max"].setdefault(sname, []).append(it_get(ent, last)["acc"])
            # pooled depth bins over all test sets at best_it
            cs, ds = [], []
            for sname in TESTS:
                if sname in r["sets"]:
                    cd = correct(name, r, sname, bi)
                    if cd is not None:
                        cs.append(cd[0]); ds.append(cd[1])
                        corr_cache[(name, sname)] = cd[0]
            if cs:
                c, d = np.concatenate(cs), np.concatenate(ds)
                bins = np.digitize(d / L, [1, 3, 10, 30], right=True)
                A.setdefault("bins", []).append([float(c[bins == k].mean()) if (bins == k).any() else None for k in range(5)])
                A.setdefault("bins_n", []).append([int((bins == k).sum()) for k in range(5)])
                hi = bins >= 3
                A.setdefault("bins_10x_plus", []).append(float(c[hi].mean()) if hi.any() else None)
            # stability
            if "stab33" in r["sets"]:
                ent = r["sets"]["stab33"]
                a64k, a6k = it_get(ent, 64000), it_get(ent, 6400)
                if a64k and a6k:
                    A.setdefault("stab_6400", []).append(a6k["acc"])
                    A.setdefault("stab_64000", []).append(a64k["acc"])
                    c1, _ = correct(name, r, "stab33", 6400)
                    c2, _ = correct(name, r, "stab33", 64000)
                    b_, c_ = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
                    p = float(stats.binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0
                    A.setdefault("mcnemar", []).append(dict(seed=r["seed"], right_to_wrong=b_, wrong_to_right=c_, p=p,
                                                            drop=a6k["acc"] - a64k["acc"]))
            # extras
            exr = r.get("extras", {})
            if exr:
                A.setdefault("extras", []).append(exr)
            if "test19" in r["sets"] and "bank_stats" in r["sets"]["test19"]:
                A.setdefault("bank_stats", []).append(r["sets"]["test19"]["bank_stats"])
        A["summary"] = {s: mean_ci(v) for s, v in A["sets"].items()}
        A["summary_at_max"] = {s: mean_ci(v) for s, v in A["sets_at_max"].items()}
        A["nll_mean"] = {s: float(np.mean(v)) for s, v in A["nll"].items()}
        A["ece_mean"] = {s: float(np.mean(v)) for s, v in A["ece"].items()}
        if "bins" in A:
            A["bins_summary"] = [mean_ci([b[k] for b in A["bins"]]) for k in range(5)]
            A["bins_10x_plus_summary"] = mean_ci(A["bins_10x_plus"])
        if "stab_6400" in A:
            A["stab_drop"] = mean_ci(np.array(A["stab_6400"]) - np.array(A["stab_64000"]))
            A["unstable_seeds"] = [m for m in A["mcnemar"] if m["drop"] > 0.05 and m["p"] < 0.05]
        if "extras" in A:
            keys = sorted({k for e in A["extras"] for k in e})
            A["extras_mean"] = {k: float(np.mean([e[k] for e in A["extras"] if e.get(k) is not None]))
                                for k in keys if any(e.get(k) is not None for e in A["extras"])}
        if "bank_stats" in A:
            keys = sorted({k for e in A["bank_stats"] for k in e})
            A["bank_stats_mean"] = {k: float(np.mean([e[k] for e in A["bank_stats"] if k in e])) for k in keys}
        # training logs: ledger, cosine, TD, cross writes
        tl = []
        for name, r in rs:
            p = f"{VM}/ckpt/{name}.train.json"
            if os.path.exists(p):
                tl.append(jl(p))
        if tl:
            A["train_s_per_it"] = float(np.mean([t["s_per_it"] for t in tl]))
            A["params"] = tl[0]["params"]
            A["train_acc_last"] = mean_ci([t["train_acc_last"] for t in tl])

            def late(key, frac=0.25):
                v = []
                for t in tl:
                    lg = [e for e in t["log"] if key in e]
                    if lg:
                        k0 = int(len(lg) * (1 - frac))
                        v.append(float(np.mean([e[key] for e in lg[k0:]])))
                return mean_ci(v) if v else None

            def whole(key):
                v = [float(np.mean([e[key] for e in t["log"] if key in e])) for t in tl if any(key in e for e in t["log"])]
                return mean_ci(v) if v else None
            A["ledger"] = {k: late(k) for k in LEDGER_KEYS if late(k)}
            A["ledger_max_cons_err"] = max([e.get("cons_err", 0) for t in tl for e in t["log"]] or [0])
            A["cos_A2t"] = whole("cos_A2t")
            A["cos_A2t_late"] = late("cos_A2t")
            A["td"] = late("td")
            A["cross_write_share_train"] = late("cross_write_share")
        per[arm] = A
        S["arms"][arm] = {k: v for k, v in A.items() if k not in ("extras", "bank_stats")}
    # ------------------------------------------------ dense validity
    for arm in ("A1", "A3"):
        v, ex = [], []
        for p in sorted(glob.glob(f"{VM}/eval/{arm}dense_s[0-9].json")):
            r = jl(p)
            t = r["sets"].get("testdense9")
            if t:
                m128 = it_get(t, 128)
                bi = r.get("best_it") or 128
                v.append(it_get(t, bi)["acc"])
                ex.append(m128["exact_maze"])
        S["validity"][f"{arm}_dense_cell_acc_best"] = mean_ci(v)
        S["validity"][f"{arm}_dense_exact_maze_at128"] = mean_ci(ex)
    if "tests" in S:
        S["validity"]["memory_need_logistic"] = S["tests"].get("memory_need")
    # ------------------------------------------------ pre-registered differences (seed-level Welch)
    pairs = [("A7", "A4"), ("A2", "A7"), ("A7", "A6"), ("A5", "A2t"), ("A4", "A3"), ("A7", "A2t"), ("A2", "A2t"),
             ("A7", "A1"), ("A7", "A1w"), ("A7", "A2"), ("A6", "A2t"), ("A1w", "A1"), ("A2", "A1"), ("A8", "A5"),
             ("A10", "A5"), ("A7p", "A7"), ("A7f", "A7")]
    for a, b in pairs:
        if a in per and b in per:
            d = {}
            for sname in TESTS:
                if sname in per[a]["sets"] and sname in per[b]["sets"]:
                    d[sname] = welch(per[a]["sets"][sname], per[b]["sets"][sname])
            if per[a].get("bins_10x_plus") and per[b].get("bins_10x_plus"):
                d["bins_10x_plus"] = welch(per[a]["bins_10x_plus"], per[b]["bins_10x_plus"])
                for k in (2, 3, 4):
                    va = [x[k] for x in per[a]["bins"] if x[k] is not None]
                    vb = [x[k] for x in per[b]["bins"] if x[k] is not None]
                    d[f"bin_{BIN_NAMES[k]}"] = welch(va, vb)
            if "stab_drop" in per[a] and "stab_drop" in per[b]:
                d["stab_drop"] = welch(per[a]["stab_drop"]["per_seed"], per[b]["stab_drop"]["per_seed"])
            S["diffs"][f"{a}-{b}"] = d
            # maze-level paired bootstrap (seed-averaged correctness per maze)
            bt = {}
            rng = np.random.default_rng(0)
            for sname in ("test9", "test13", "test19", "test33", "test59"):
                ca = [corr_cache[(n, sname)] for n, _ in runs(a, sfx) if (n, sname) in corr_cache]
                cb = [corr_cache[(n, sname)] for n, _ in runs(b, sfx) if (n, sname) in corr_cache]
                if ca and cb:
                    k = min(min(len(x) for x in ca), min(len(x) for x in cb))
                    diff = np.mean([x[:k] for x in ca], 0) - np.mean([x[:k] for x in cb], 0)
                    idx = rng.integers(0, k, size=(5000, k))
                    bs_ = diff[idx].mean(1)
                    bt[sname] = dict(diff=float(diff.mean()), lo=float(np.percentile(bs_, 2.5)), hi=float(np.percentile(bs_, 97.5)))
            S["bootstrap"][f"{a}-{b}"] = bt
    # ------------------------------------------------ memory against D, FLOPs
    for p in sorted(glob.glob(f"{VM}/ckpt/mem_*_D*.train.json")):
        t = jl(p)
        S["memdepth"][os.path.basename(p)[4:-11]] = dict(arm=t["arm"], D=t["args"]["D"], peak_mem_mb=t["peak_mem_mb"],
                                                           s_per_it=t["s_per_it_after_first"])
    if os.path.exists(f"{VM}/flops.json"):
        S["flops"] = jl(f"{VM}/flops.json")
    # ------------------------------------------------ criteria
    C = S["criteria"]
    D = S["diffs"]

    def g(arm, sname):
        s = S["arms"].get(arm, {}).get("summary", {}).get(sname)
        return s["mean"] if s else None

    def dd(k, sname):
        return (D.get(k) or {}).get(sname)
    g0 = st.get("g0", {})
    C["gate_G0_passed"] = g0.get("passed")
    C["validity"] = dict(
        A3_dense_ge_90=(S["validity"]["A3_dense_cell_acc_best"] or {}).get("mean", 0) >= 0.90 if S["validity"].get("A3_dense_cell_acc_best") else None,
        A3_sparse19_le_60=(g("A3", "test19") <= 0.60) if g("A3", "test19") is not None else None,
        A1_dense_exact128_ge_95=(S["validity"]["A1_dense_exact_maze_at128"] or {}).get("mean", 0) >= 0.95 if S["validity"].get("A1_dense_exact_maze_at128") else None,
        logistic_le_52=(S.get("tests", {}).get("memory_need", {}) or {}).get("pass"),
        kill_A3_sparse19_ge_80=(g("A3", "test19") >= 0.80) if g("A3", "test19") is not None else None)
    s = {}
    s["1_A7_9x9_ge_85"] = (g("A7", "test9") or 0) >= 0.85
    x = dd("A7-A4", "test19")
    s["2_A7_minus_A4_19_ge15_lo_gt5"] = bool(x and x["diff"] >= 0.15 and x["lo"] > 0.05)
    xs = [dd("A2-A7", n) for n in ("test19", "test33")]
    s["3_A2_minus_A7_hi_le5_at19_33"] = bool(all(v and v["hi"] <= 0.05 for v in xs))
    md = S["memdepth"]
    try:
        a7 = {v["D"]: v["peak_mem_mb"] for v in md.values() if v["arm"] == "A7"}
        a2 = {v["D"]: v["peak_mem_mb"] for v in md.values() if v["arm"] == "A2"}
        s["3b_A7_mem_D256_le_1.2x_D32"] = a7[256] <= 1.2 * a7[32]
        s["3b_A2_mem_grows"] = a2[256] > 1.5 * a2[32]
    except Exception:
        pass
    xs = [dd("A2-A7", f"bin_{BIN_NAMES[k]}") for k in (2, 3)]
    s["4_A7_within5_of_A2_bins_3-30x"] = bool(all(v and v["hi"] <= 0.05 for v in xs))
    C["online_prices_work"] = s
    C["online_prices_work_all"] = all(v for k, v in s.items() if not k.startswith("3b_A2"))
    x19 = dd("A7-A4", "test19")
    C["fail_A7_minus_A4_hi_lt5_at19"] = bool(x19 and x19["hi"] < 0.05)
    C["fail_A2_minus_A7_lo_gt10_at19_and_33"] = bool(all((dd("A2-A7", n) or {}).get("lo", -1) > 0.10 for n in ("test19", "test33")))
    ev = [dd("A7-A6", n) for n in ("test33",)] + [dd("A7-A6", "bins_10x_plus")]
    C["econ_beyond_DNI"] = bool(any(v and v["diff"] >= 0.03 and v["lo"] > 0 for v in ev))
    a67 = [dd("A7-A6", n) for n in TESTS[:5]]
    C["A7_minus_A6_within_3pp_everywhere"] = bool(all(v and v["lo"] >= -0.03 and v["hi"] <= 0.03 for v in a67))
    led = S["arms"].get("A7", {}).get("ledger", {}).get("capped")
    C["A7_ledger_binding_rate"] = led["mean"] if led else None
    C["ledger_inert_lt1pct"] = (led["mean"] < 0.01) if led else None
    best = [a for a in ("A1", "A1w", "A2") if a in S["arms"]]
    lw = {}
    for a in best:
        for k in ("test59", "bins_10x_plus"):
            v = dd(f"A7-{a}", k)
            lw[f"{a}:{k}"] = v
    C["long_shot"] = dict(detail=lw, won=bool(best) and any(
        all((dd(f"A7-{a}", k) or {}).get("diff", -1) >= 0.05 and (dd(f"A7-{a}", k) or {}).get("lo", -1) > 0 for a in best)
        for k in ("test59", "bins_10x_plus")))
    a5 = [dd("A5-A2t", n) for n in ("test9", "test13", "test19", "test33")]
    C["offline_A5_eq_A2t_within_3pp"] = bool(all(v and v["lo"] >= -0.03 and v["hi"] <= 0.03 for v in a5))
    cs = S["arms"].get("A5", {}).get("cos_A2t")
    C["A5_cos_A2t_mean"] = cs["mean"] if cs else None
    x = dd("A4-A3", "test19")
    C["prediction_A4_within5_of_A3_at19"] = bool(x and abs(x["diff"]) <= 0.05)
    C["investigate_A4_beats_A3_by_10"] = bool(x and x["diff"] > 0.10)
    C["stability"] = {a: dict(drop=S["arms"][a].get("stab_drop"), unstable=S["arms"][a].get("unstable_seeds"))
                      for a in S["arms"] if "stab_drop" in S["arms"][a]}
    json.dump(S, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1, default=float)
    # ------------------------------------------------ chart JSON
    arms = [a for a in MAIN if a in S["arms"]]
    chart = dict(
        title="P5: MP-sparse accuracy at T (seed mean, 95% CI) by test size and by path length / L_train",
        by_size=dict(x=["test9", "test13", "test19", "test33", "test59"], series={
            a: [S["arms"][a]["summary"].get(n) and dict(mean=S["arms"][a]["summary"][n]["mean"], lo=S["arms"][a]["summary"][n]["lo"],
                                                        hi=S["arms"][a]["summary"][n]["hi"]) for n in ["test9", "test13", "test19", "test33", "test59"]]
            for a in arms}),
        by_depth_bin=dict(x=BIN_NAMES, series={a: [b and dict(mean=b["mean"], lo=b["lo"], hi=b["hi"]) for b in S["arms"][a].get("bins_summary", [])]
                                                for a in arms}),
        stability=dict(x=["6400", "64000"], series={a: [float(np.mean(S["arms"][a]["stab_6400"])), float(np.mean(S["arms"][a]["stab_64000"]))]
                                                     for a in arms if "stab_6400" in S["arms"][a]}),
        memdepth=S["memdepth"])
    json.dump(chart, open(os.path.join(HERE, "results", "chart.json"), "w"), indent=1, default=float)
    print(json.dumps(C, indent=1, default=float))


if __name__ == "__main__":
    main()
