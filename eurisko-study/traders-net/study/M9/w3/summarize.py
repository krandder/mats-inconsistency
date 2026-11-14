"""M9(c) w3 summary: python3 summarize.py -> results/SUMMARY.csv, results/LOTTERY.csv, results/VERDICT.json, kq.png"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or"]
KS = [1, 4, 16, 64]
SERIES = {"FS": {1: 8, 4: 8, 16: 8, 64: 8}, "FT": {1: 128, 4: 32, 16: 8, 64: 2}}
runs = {}
for f in glob.glob(f"{R}/vm/*-s*.json"):
    d = json.load(open(f)); runs[(d["arm"], d["K"], d["hw"], d["seed"])] = d


def ms(v):
    v = [x for x in v if x is not None]
    if not v: return (None, None, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def priced(d, f):
    b = [x for x, g in zip(d["q_brier"], d["q_fam"]) if g == f]
    return float(np.mean(np.array(b) < 0.1))


rows, lot, verdict = [], [], {"series": {}}
for arm in ["B", "A"]:
    for ser, hwmap in SERIES.items():
        pts = {}
        for K in KS:
            ds = [runs[k] for k in sorted(runs) if k[:3] == (arm, K, hwmap[K])]
            if not ds: continue
            pts[K] = ds
            base = {"arm": arm, "series": ser, "K": K, "hw": hwmap[K], "seeds": len(ds),
                    "n_layer_params": ds[0]["n_layer_params"], "n_net_params": ds[0]["n_net_params"],
                    "flops_step": ds[0]["flops_step"]}
            mets = {"learn_bce": [d["score"]["learn_bce"] for d in ds],
                    "humility": [d["score"]["humility"] for d in ds],
                    "lfsr_learnt": [d["score"]["lfsr_learnt"] for d in ds],
                    "viol_all": [d["score"]["viol_all"] for d in ds],
                    "eff_traders_per_q": [d["eff_traders_per_q"] for d in ds],
                    "rolling_brier_final": [d["rolling_brier_final"] for d in ds]}
            for f in FAM:
                mets[f"bce_{f}"] = [d["score"]["fam"][f]["bce"] for d in ds]
                mets[f"priced_{f}"] = [priced(d, f) for d in ds]
                mets[f"flops_to_b01_{f}"] = [(d["fam_pos_to_brier01"][f] / d["B"] * d["flops_step"])
                                            if d["fam_pos_to_brier01"][f] is not None else None for d in ds]
                mets[f"n_hit_b01_{f}"] = [int(d["fam_pos_to_brier01"][f] is not None) for d in ds]
            for m, v in mets.items():
                mu, sd, n = ms(v)
                rows.append({**base, "metric": m, "mean": mu, "sd": sd, "n": n,
                             "values": "/".join("NA" if x is None else f"{x:.5g}" for x in v)})
        if len(pts) < 4: continue
        # stop rule: trend = |mean(K1)-mean(K64)| > 2 pooled sd and monotone
        L = {K: [d["score"]["learn_bce"] for d in pts[K]] for K in KS}
        mu = [np.mean(L[K]) for K in KS]; psd = math.sqrt(np.mean([np.var(L[K], ddof=1) for K in KS]))
        diffs = np.diff(mu)
        mono = bool(np.all(diffs <= 0) or np.all(diffs >= 0))
        trend = bool(abs(mu[0] - mu[-1]) > 2 * psd and mono)
        # lottery fit per family: priced(K) = 1-(1-pi)^K over 4 K x seeds
        fits = {}
        for f in FAM:
            x = np.array([K for K in KS for d in pts[K]], float)
            y = np.array([priced(d, f) for K in KS for d in pts[K]])
            grid = np.linspace(1e-4, 1 - 1e-4, 20000)
            sse = [np.sum((y - (1 - (1 - p) ** x)) ** 2) for p in grid]
            j = int(np.argmin(sse)); sst = np.sum((y - y.mean()) ** 2)
            r2 = float(1 - sse[j] / sst) if sst > 0 else None
            rho = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else None
            fits[f] = {"pi": float(grid[j]), "r2": r2, "corr_K_priced": rho,
                       "priced_by_K": [float(np.mean([priced(d, f) for d in pts[K]])) for K in KS]}
            lot.append({"arm": arm, "series": ser, "family": f, **{k: v for k, v in fits[f].items() if k != "priced_by_K"},
                        **{f"priced_K{K}": fits[f]["priced_by_K"][i] for i, K in enumerate(KS)}})
        law_fams = [f for f in FAM if fits[f]["r2"] is not None and fits[f]["r2"] > 0.5
                    and (fits[f]["corr_K_priced"] or 0) > 0]
        verdict["series"][f"{arm}-{ser}"] = {"learn_bce_by_K": dict(zip(map(str, KS), map(float, mu))),
                                            "pooled_sd": psd, "monotone": mono, "trend": trend,
                                            "lottery_r2_gt_0.5_families": law_fams}
for arm in ["B", "A"]:
    s = verdict["series"].get(f"{arm}-FT")
    if s: verdict[f"law_present_{arm}_FT"] = bool(s["trend"] and len(s["lottery_r2_gt_0.5_families"]) >= 2)
import csv
with open(f"{R}/SUMMARY.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if lot:
    with open(f"{R}/LOTTERY.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lot[0])); w.writeheader(); w.writerows(lot)
json.dump(verdict, open(f"{R}/VERDICT.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))

# plot: learnable BCE vs K, small multiples by arm; FS vs FT as two series
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"FS": "#2a78d6", "FT": "#eb6834"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ai, arm in enumerate(["B", "A"]):
        ax = axs[ai]
        for ser, hwmap in SERIES.items():
            xs, m_, s_ = [], [], []
            for K in KS:
                v = [runs[k]["score"]["learn_bce"] for k in runs if k[:3] == (arm, K, hwmap[K])]
                if v: xs.append(K); m_.append(np.mean(v)); s_.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            ax.errorbar(xs, m_, yerr=s_, color=C[ser], lw=2, marker="o", ms=8, capsize=3,
                        label={"FS": "fixed trader size (hw=8)", "FT": "fixed layer params (K*hw=128)"}[ser])
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("traders per question K"); ax.set_ylabel("learnable BCE (nats), mean ± sd, 3 seeds")
        ax.set_title(f"arm {arm}: " + ("z = l0 + Σd (anchored)" if arm == "B" else "z = Σd (plain)"), fontsize=10)
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        if ai == 0: ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    for ser, hwmap in SERIES.items():
        xs, m_ = [], []
        for K in KS:
            v = [np.mean([priced(runs[k], f) for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or"]])
                 for k in runs if k[:3] == ("B", K, hwmap[K])]
            if v: xs.append(K); m_.append(np.mean(v))
        ax.plot(xs, m_, color=C[ser], lw=2, marker="o", ms=8)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_ylim(0, 1)
    ax.set_xlabel("traders per question K"); ax.set_ylabel("share of learnable questions with Brier < 0.1")
    ax.set_title("arm B: priced share (soup's K/Q axis)", fontsize=10)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "kq.png"), dpi=110)
except Exception as e:
    print("plot failed", e)

# ex-LFSR16 learnable BCE (LFSR16 is learnt or not by seed luck, M6.md) and humility trend test
ex = {}
for arm in ["B", "A"]:
    for ser, hwmap in SERIES.items():
        for K in KS:
            ds = [runs[k] for k in sorted(runs) if k[:3] == (arm, K, hwmap[K])]
            v = [float(np.mean([d["score"]["fam"][f]["bce"] for f in ["periodic", "ca110", "prime", "and", "or"]])) for d in ds]
            h = [d["score"]["humility"] for d in ds]
            ex[f"{arm}-{ser}-K{K}"] = {"learn_bce_ex_lfsr": ms(v)[:2], "values": v, "humility": ms(h)[:2], "humility_values": h}
json.dump(ex, open(f"{R}/EXLFSR.json", "w"), indent=1)
for k, v in ex.items():
    print(k, "exLFSR %.4f±%.4f" % tuple(v["learn_bce_ex_lfsr"]), "hum %.4f±%.4f" % tuple(v["humility"]))

# figure 2: ex-LFSR16 learnable BCE and humility vs K (both arms), paired-seed means
try:
    C = {"FS": "#2a78d6", "FT": "#eb6834"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, ax, lab in [("learn_bce_ex_lfsr", axs[0], "learnable BCE excl. LFSR16 (nats)"),
                         ("humility", axs[1], "noise excess max(BCE − ln2) on prng/ca30")]:
        for arm, ls in [("B", "-"), ("A", "--")]:
            for ser in SERIES:
                m_ = [ex[f"{arm}-{ser}-K{K}"][key][0] for K in KS]; s_ = [ex[f"{arm}-{ser}-K{K}"][key][1] for K in KS]
                ax.errorbar(KS, m_, yerr=s_, color=C[ser], ls=ls, lw=2, marker="o", ms=7, capsize=3,
                            label=f"arm {arm}, {'fixed trader size' if ser == 'FS' else 'fixed layer params'}")
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("traders per question K"); ax.set_ylabel(lab + ", mean ± sd", fontsize=9)
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "kq2.png"), dpi=110)
except Exception as e:
    print("plot2 failed", e)
