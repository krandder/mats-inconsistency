"""M9(b): does wealth-net price LFSR16/CA110, and compute-to-Brier net vs soup (1 VI = 1 op).
Read-only on wealth-net, S0 w1 npz, logical-soup runs/phase3. Writes CSV/JSON/PNG next to this file."""
import csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ES = "/home/agent/workspaces/eurisko-study"
WN = f"{ES}/wealth-net/results"
SOUP = f"{ES}/logical-soup/runs/phase3"
NPZ = "/tmp/traders-net/S0/w1"
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "pi", "and", "or"]
THR = [0.2, 0.1, 0.05]
ROLL = 50
Q, W = 64, 32


def net_fwd_flops(B, Q=64, W=32, d=64, H=4, L=2, mlp=4):  # same formula as S0 panel.net_forward_flops
    per_tok = W * d + L * (3 * d * d + 2 * Q * d + d * d + 2 * mlp * d * d) + d
    return 2 * B * Q * per_tok


def first_below(x, thr):
    idx = np.nonzero(x <= thr)[0]
    return int(idx[0]) if len(idx) else None


def first_sustained(x, thr, hold=500):
    """first i with x[i] <= thr and mean(x[i:i+hold]) <= thr (secondary, amendment 1)."""
    x = np.nan_to_num(x, nan=9.0)
    c = np.cumsum(np.insert(x, 0, 0.0)); n = len(x)
    for i in np.nonzero(x <= thr)[0]:
        j = min(n, i + hold)
        if (c[j] - c[i]) / (j - i) <= thr:
            return int(i)
    return None


def roll(x, n=ROLL):
    c = np.cumsum(np.insert(x, 0, 0.0))
    r = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, np.nan), r])


rows = []      # arm, cell, seed, family, thr, reached, compute, compute_per_q, data_per_q
finals = []    # arm, seed, family, brier_final, brier_log_final
curves = {}    # (arm, family) -> list of (compute_per_q array, brier array)

# ---- wealth-net (multi-epoch; held-out eval every 100 steps)
step_flops = 3 * net_fwd_flops(128)
for s in (1, 2, 3):
    snaps = sorted(glob.glob(f"{WN}/train-s{s}/snapshots/step-*.json"), key=lambda p: int(p.split("-")[-1][:-5]))
    steps, br, brl = [], [], []
    for p in snaps:
        d = json.load(open(p))
        steps.append(d["step"]); br.append(d["ev_brier"]); brl.append(d["ev_brier_log"])
    steps, br, brl = np.array(steps), np.array(br), np.array(brl)
    comp = steps * step_flops
    train_pos = int(24000 * 0.8)
    for fi, f in enumerate(FAM):
        finals.append(dict(arm="WN", seed=s, family=f, step=int(steps[-1]), brier_final=br[-1, fi],
                           brier_log_final=brl[-1, fi],
                           brier_max_last5k=float(br[steps >= steps[-1] - 5000, fi].max())))
        curves.setdefault(("WN", f), []).append((comp / Q, br[:, fi], steps * 128.0))
        for t in THR:
            i = first_below(br[:, fi], t)
            c = comp[i] if i is not None else comp[-1]
            rows.append(dict(arm="WN", cell="B128", seed=s, family=f, thr=t, reached=i is not None,
                             compute=c, compute_per_q=c / Q,
                             data_per_q=0, sust_reached=i is not None, sust_compute_per_q=c / Q, sust_data_per_q=0,
                             total_compute_per_q=comp[-1] / Q))

# ---- S0 prequential P4 / P16 (trade-time logits)
for arm, B in (("P4", 4), ("P16", 16)):
    for s in (1, 2, 3):
        d = np.load(f"{NPZ}/{arm}-s{s}.npz")
        z, y, fam = d["z"].astype(np.float64), d["y"].astype(np.float64), d["fam"]
        p = 1 / (1 + np.exp(-z)); pb = 1 / (1 + np.exp(-d["zb"].astype(np.float64)))
        n = z.shape[0]
        steps_at = np.arange(n) // B + 1          # training steps completed at/after predicting pos
        comp = (np.arange(n) // B) * 3 * net_fwd_flops(B)  # compute spent BEFORE this prediction
        for fi, f in enumerate(FAM):
            m = fam == fi
            b = ((p[:, m] - y[:, m]) ** 2).mean(1)
            bl = ((pb[:, m] - y[:, m]) ** 2).mean(1)
            rb = roll(b)
            finals.append(dict(arm=arm, seed=s, family=f, step=int(steps_at[-1]),
                               brier_final=float(b[51200:].mean()), brier_log_final=float(bl[51200:].mean()),
                               brier_max_last5k=float(np.nanmax(roll(b, 1000)[-5000:]))))
            sub = np.arange(ROLL - 1, n, 50)
            curves.setdefault((arm, f), []).append((comp[sub] / Q, rb[sub], sub + 1.0))
            for t in THR:
                i = first_below(np.nan_to_num(rb, nan=9.0), t)
                c = comp[i] if i is not None else comp[-1]
                k = first_sustained(rb, t)
                rows.append(dict(arm=arm, cell=f"B{B}", seed=s, family=f, thr=t, reached=i is not None,
                                 compute=c, compute_per_q=c / Q, data_per_q=(i + 1) if i is not None else n,
                                 sust_reached=k is not None, sust_compute_per_q=(comp[k] if k is not None else comp[-1]) / Q,
                                 sust_data_per_q=(k + 1) if k is not None else n, total_compute_per_q=(n // B) * 3 * net_fwd_flops(B) / Q))

# ---- soup phase3 cells
CELLS = ["k12-f8", "k12-f10", "k12-f12", "k14-f8", "k14-f10", "k14-f12", "k16-f12"]
for cell in CELLS:
    for s in (1, 2, 3):
        path = f"{SOUP}/{cell}-s{s}/metrics.csv"
        with open(path) as fh:
            rd = list(csv.DictReader(fh))
        cfg = json.load(open(f"{SOUP}/{cell}-s{s}/config.json"))
        Qs = 2 ** cfg["logq"]
        ops = np.array([float(r["cum_ops"]) for r in rd])
        for f in FAM:
            b = np.array([float(r["brier_" + ("lfsr" if f == "lfsr16" else f)]) for r in rd]) * 1e-6
            rb = roll(b)
            finals.append(dict(arm="SOUP", cell=cell, seed=s, family=f, step=len(rd),
                               brier_final=float(b[-50:].mean()), brier_log_final=float("nan"),
                               brier_max_last5k=float("nan")))
            sub = np.arange(ROLL - 1, len(rd), 10)
            curves.setdefault((f"SOUP {cell}", f), []).append((ops[sub] / Qs, rb[sub], sub + 1.0))
            for t in THR:
                i = first_below(np.nan_to_num(rb, nan=9.0), t)
                c = ops[i] if i is not None else ops[-1]
                k = first_sustained(rb, t)
                rows.append(dict(arm="SOUP", cell=cell, seed=s, family=f, thr=t, reached=i is not None,
                                 compute=c, compute_per_q=c / Qs, data_per_q=(i + 1) if i is not None else len(rd),
                                 sust_reached=k is not None, sust_compute_per_q=(ops[k] if k is not None else ops[-1]) / Qs,
                                 sust_data_per_q=(k + 1) if k is not None else len(rd), total_compute_per_q=ops[-1] / Qs))

# fix WN data_per_q: examples seen = steps*128 (positions of all Q), i.e. outcomes per question
for r in rows:
    if r["arm"] == "WN":
        r["data_per_q"] = r["compute"] / step_flops * 128
        r["sust_data_per_q"] = r["data_per_q"]

with open(f"{HERE}/compute_to_brier_raw.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
fk = ["arm", "cell", "seed", "family", "step", "brier_final", "brier_log_final", "brier_max_last5k"]
with open(f"{HERE}/final_brier_raw.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fk); w.writeheader()
    for r in finals: w.writerow({k: r.get(k, "") for k in fk})

# ---- summary: per arm/cell/family/thr
summ = []
keys = sorted({(r["arm"], r["cell"], r["family"], r["thr"]) for r in rows}, key=str)
for k in keys:
    rs = [r for r in rows if (r["arm"], r["cell"], r["family"], r["thr"]) == k]
    reached = [r for r in rs if r["reached"]]
    lc = np.log10([r["compute_per_q"] for r in reached]) if reached else np.array([])
    ld = np.log10([r["data_per_q"] for r in reached]) if reached else np.array([])
    tot = float(np.mean([r["total_compute_per_q"] for r in rs]))
    sr = [r for r in rs if r["sust_reached"]]
    sc = np.log10([r["sust_compute_per_q"] for r in sr]) if sr else np.array([])
    sd_ = np.log10([r["sust_data_per_q"] for r in sr]) if sr else np.array([])
    summ.append(dict(arm=k[0], cell=k[1], family=k[2], thr=k[3], reached=f"{len(reached)}/{len(rs)}",
                     log10_flops_per_q_mean=round(float(lc.mean()), 3) if len(lc) else "",
                     log10_flops_per_q_sd=round(float(lc.std(ddof=1)), 3) if len(lc) > 1 else "",
                     log10_outcomes_per_q_mean=round(float(ld.mean()), 3) if len(ld) else "",
                     log10_outcomes_per_q_sd=round(float(ld.std(ddof=1)), 3) if len(ld) > 1 else "",
                     log10_total_compute_per_q=round(float(np.log10(tot)), 3),
                     sust_reached=f"{len(sr)}/{len(rs)}",
                     sust_log10_flops_per_q_mean=round(float(sc.mean()), 3) if len(sc) else "",
                     sust_log10_flops_per_q_sd=round(float(sc.std(ddof=1)), 3) if len(sc) > 1 else "",
                     sust_log10_outcomes_per_q_mean=round(float(sd_.mean()), 3) if len(sd_) else "",
                     sust_log10_outcomes_per_q_sd=round(float(sd_.std(ddof=1)), 3) if len(sd_) > 1 else ""))
with open(f"{HERE}/compute_to_brier_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

fs = []
for arm in ["WN", "P4", "P16"] + [f"SOUP {c}" for c in CELLS]:
    for f in FAM:
        rs = [r for r in finals if r["arm"] == arm.split()[0] and (arm == r["arm"] or r.get("cell") == arm.split()[-1])]
        rs = [r for r in rs if r["family"] == f]
        v = np.array([r["brier_final"] for r in rs]); vl = np.array([r["brier_log_final"] for r in rs])
        fs.append(dict(arm=arm, family=f, n=len(v), brier_mean=round(float(v.mean()), 4), brier_sd=round(float(v.std(ddof=1)), 4),
                       per_seed="/".join(f"{x:.4f}" for x in v), priced_le_0p1=f"{int((v <= 0.1).sum())}/{len(v)}",
                       logistic_mean=round(float(np.nanmean(vl)), 4) if not np.all(np.isnan(vl)) else ""))
with open(f"{HERE}/final_brier_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fs[0].keys())); w.writeheader(); w.writerows(fs)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SER = [("P4", "net prequential B=4", "#2a78d6"), ("P16", "net prequential B=16", "#eb6834"),
       ("WN", "wealth-net multi-epoch B=128 (held-out)", "#1baf7a"), ("SOUP k14-f8", "soup K/Q=64 (16k/256)", "#eda100"),
       ("SOUP k16-f12", "soup K/Q=16 (64k/4096)", "#e87ba4"), ("SOUP k12-f12", "soup K/Q=1 (4k/4096)", "#008300")]
PF = ["periodic", "prime", "ca110", "lfsr16"]
fig, ax = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
for row, xi, xl in ((0, 0, "compute per question (net: training FLOPs/Q; soup: VIs/Q, 1 VI = 1 op)"), (1, 2, "outcomes observed per question (WN: examples incl. repeats)")):
    for j, f in enumerate(PF):
        a = ax[row, j]
        for key, lab, col in SER:
            for si, tr in enumerate(curves.get((key, f), [])):
                x = tr[xi]; ok = x > 0
                a.plot(x[ok], tr[1][ok], color=col, lw=1.6, alpha=0.85, label=lab if si == 0 else None)
        a.axhline(0.1, color="#52514e", lw=0.8, ls="--"); a.axhline(0.25, color="#b0afa8", lw=0.8, ls=":")
        a.set_xscale("log"); a.set_ylim(-0.01, 0.32); a.grid(alpha=0.25, lw=0.5)
        a.set_title(f"{f}", fontsize=11)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        if j == 0: a.set_ylabel("Brier (rolling 50; WN held-out)")
    fig.text(0.5, 0.49 if row == 0 else 0.015, xl, ha="center", fontsize=10)
ax[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("M9(b): Brier vs compute and vs data, net (3 seeds) vs soup (3 seeds per cell). Dashed: Brier 0.1; dotted: 0.25 (coin)", fontsize=12)
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.subplots_adjust(hspace=0.35)
fig.savefig(f"{HERE}/flops_to_brier.png", dpi=110)
print("step_flops_WN", step_flops, "P4", 3 * net_fwd_flops(4), "P16", 3 * net_fwd_flops(16))
