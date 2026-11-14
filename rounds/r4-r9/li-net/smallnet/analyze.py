"""Aggregate smallnet runs (eurisko 2026-10-10).

  python3 analyze.py [--runs /tmp/smallnet/runs] [--baseline E01-k0] [--plot TAG,TAG,...] [--split val|test]

Writes results/runs.csv (one row per run), results/summary.json and results/summary.md (per experiment tag:
seed-mean single-cell KL per N, paired log-ratio vs the baseline per N, the loop's selection score), and,
with --plot, results/curve_kl_vs_boards.png and results/curve_kl_vs_compute.png.
Selection score (PREREG.md): S_small = mean over N in {100, 300, 1000} of the seed-paired mean of
log(KL_arm / KL_baseline) on VALIDATION; negative = better than k0. Needs >= 3 paired seeds at each N.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [100, 300, 1000, 3000, 10000]
SMALL = [100, 300, 1000]


def load(runs, split="val"):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "**", "*.json"), recursive=True)):
        if "/status/" in p:
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if split not in r:
            continue
        v = r[split]
        row = {
            "path": os.path.relpath(p, runs), "tag": r.get("tag", r["arm"]), "arm": r["arm"], "n": r["n"], "seed": r["seed"],
            "set": " ".join(r.get("set", [])), "kl": v["kl"], "logloss": v["logloss"], "ece": v["ece"],
            "joint_ll": v.get("joint_ll"), "joint_ll_ar": v.get("joint_ll_ar"), "joint_ll_exact": v.get("joint_ll_exact"),
            "steps_run": r["steps_run"], "best_step": r["best_step"], "max_steps": r["cfg"]["max_steps"],
            "train_sec": r["train_sec"], "flops_at_best": r["flops_at_best"], "flops_per_board": r["flops_per_board_at_best"],
            "flops_total": r["flops_total"], "rwv": r.get("real_world_valid", True), "gpu": r.get("gpu"),
            "marghead_kl": (r.get(split + "_marghead") or r.get("val_marghead") or {}).get("kl") if split == "val" else None,
            "bn_M1_kl": (r.get("val_bn_M1") or {}).get("kl") if split == "val" else None,
        }
        for i, x in enumerate(v["kl_stage"]):
            row[f"kl_s{i}"] = x
        rows.append(row)
    return rows


def by(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def paired_logratio(rows, tag, base, n):
    a = {r["seed"]: r["kl"] for r in rows if r["tag"] == tag and r["n"] == n}
    b = {r["seed"]: r["kl"] for r in rows if r["tag"] == base and r["n"] == n}
    s = sorted(set(a) & set(b))
    return [math.log(a[k] / b[k]) for k in s], s


def tci(x):
    """mean and 95% t-interval half-width."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(x), 2.0)
    return float(x.mean()), float(tq * x.std(ddof=1) / math.sqrt(len(x)))


def n_to_reach(curve, target):
    """Log-linear interpolation of KL(N) (seed-mean, increasing N) to the N where KL first reaches target."""
    pts = sorted(curve.items())
    if pts[0][1] <= target:
        return float(pts[0][0])
    for (n0, k0), (n1, k1) in zip(pts, pts[1:]):
        if k1 <= target:
            f = (math.log(k0) - math.log(target)) / (math.log(k0) - math.log(k1))
            return float(math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))))
    return float("inf")


def decision(rows, tag, base, nboot=2000, seed=0):
    """PREREG rule pieces: (a) N_arm needed to reach base's 10k KL, as a fraction of 10k, with a seed bootstrap;
    (b) per-N paired differences (arm - base) with t-intervals."""
    rng = np.random.default_rng(seed)
    A = {(r["n"], r["seed"]): r["kl"] for r in rows if r["tag"] == tag}
    B = {(r["n"], r["seed"]): r["kl"] for r in rows if r["tag"] == base}
    ns = [n for n in NS if any(k[0] == n for k in A) and any(k[0] == n for k in B)]
    if 10000 not in ns:
        return {"note": "needs both arms at N=10000"}
    sa = {n: [A[k] for k in sorted(A) if k[0] == n] for n in ns}
    sb10 = [B[k] for k in sorted(B) if k[0] == 10000]
    curve = {n: float(np.mean(sa[n])) for n in ns}
    frac = n_to_reach(curve, float(np.mean(sb10))) / 10000
    boots = []
    for _ in range(nboot):
        c = {n: float(np.mean(rng.choice(sa[n], len(sa[n])))) for n in ns}
        t = float(np.mean(rng.choice(sb10, len(sb10))))
        boots.append(n_to_reach(c, t) / 10000)
    boots = np.array(boots)
    boots = np.where(np.isfinite(boots), boots, 1e3)    # never reached within the sweep -> report as > 1 (1000)
    per_n = {}
    for n in ns:
        s = sorted({k[1] for k in A if k[0] == n} & {k[1] for k in B if k[0] == n})
        d = [A[(n, k)] - B[(n, k)] for k in s]
        m, h = tci(d)
        per_n[n] = {"diff_mean": m, "ci95_half": h, "seeds": s, "base_kl": float(np.mean([B[(n, k)] for k in s])) if s else None}
    return {"frac_boards_to_reach_base10k": frac, "frac_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "rule_a_pass": bool(np.percentile(boots, 97.5) <= 1 / 3),
            "per_n": per_n, "rule_b_pass": all(v["diff_mean"] + v["ci95_half"] < 0 for v in per_n.values() if not math.isnan(v["ci95_half"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/tmp/smallnet/runs")
    ap.add_argument("--baseline", default="E01-k0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--plot", default="")
    ap.add_argument("--label", default="", help="tags that get a direct label (default: first 4 plotted)")
    ap.add_argument("--title", default="")
    ap.add_argument("--decision", default="", help="comma list of tags to test against the baseline with the PREREG rule")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--tables", default="", help="comma list of tags: write results/<prefix>tables-<split>.md")
    a = ap.parse_args()
    if a.tables:
        os.makedirs(a.out, exist_ok=True)
        print(md_tables(a.runs, a.split, a.tables.split(","), a.baseline, os.path.join(a.out, a.prefix + f"tables-{a.split}.md")))
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.runs, a.split)
    pre = a.prefix
    with open(os.path.join(a.out, pre + "runs.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [k for k in ("kl_s0", "kl_s1", "kl_s2", "kl_s3") if k not in rows[0]])
            w.writeheader()
            for r in rows:
                w.writerow(r)
    summ = {}
    for (tag,), rs in sorted(by(rows, "tag").items()):
        e = {"arm": rs[0]["arm"], "set": rs[0]["set"], "rwv": rs[0]["rwv"], "per_n": {}}
        for (n,), rn in sorted(by(rs, "n").items()):
            kl = [r["kl"] for r in rn]
            lr, seeds = paired_logratio(rows, tag, a.baseline, n)
            m, h = tci(lr) if lr else (float("nan"), float("nan"))
            e["per_n"][n] = {
                "seeds": sorted(r["seed"] for r in rn), "kl_mean": float(np.mean(kl)), "kl_sd": float(np.std(kl, ddof=1)) if len(kl) > 1 else None,
                "kl_stage_mean": [float(np.mean([r[f"kl_s{i}"] for r in rn])) for i in range(4)],
                "logloss_mean": float(np.mean([r["logloss"] for r in rn])), "ece_mean": float(np.mean([r["ece"] for r in rn])),
                "joint_ll_mean": float(np.mean([r["joint_ll"] for r in rn if r["joint_ll"] is not None])) if any(r["joint_ll"] is not None for r in rn) else None,
                "joint_ll_ar_mean": float(np.mean([r["joint_ll_ar"] for r in rn if r["joint_ll_ar"] is not None])) if any(r["joint_ll_ar"] is not None for r in rn) else None,
                "flops_per_board_mean": float(np.mean([r["flops_per_board"] for r in rn])),
                "flops_at_best_mean": float(np.mean([r["flops_at_best"] for r in rn])),
                "best_step_mean": float(np.mean([r["best_step"] for r in rn])), "hit_max_steps": sum(r["steps_run"] >= r["max_steps"] and r["best_step"] > 0.8 * r["max_steps"] for r in rn),
                "train_sec_mean": float(np.mean([r["train_sec"] for r in rn])),
                "logratio_vs_base": m, "logratio_ci95_half": h, "paired_seeds": seeds,
                "marghead_kl_mean": float(np.mean([r["marghead_kl"] for r in rn if r["marghead_kl"] is not None])) if any(r["marghead_kl"] is not None for r in rn) else None,
            }
        sm = [e["per_n"][n]["logratio_vs_base"] for n in SMALL if n in e["per_n"] and len(e["per_n"][n]["paired_seeds"]) >= 3]
        e["S_small"] = float(np.mean(sm)) if len(sm) == len(SMALL) else None
        sa = [e["per_n"][n]["logratio_vs_base"] for n in NS if n in e["per_n"] and len(e["per_n"][n]["paired_seeds"]) >= 3]
        e["S_all"] = float(np.mean(sa)) if len(sa) == len(NS) else None
        summ[tag] = e
    if a.decision:
        for t in a.decision.split(","):
            if t in summ:
                summ[t]["decision_vs_base"] = decision(rows, t, a.baseline)
    json.dump(summ, open(os.path.join(a.out, pre + "summary.json"), "w"), indent=1)
    # markdown table
    lines = [f"Split: {a.split}. Baseline: {a.baseline}. KL = single-cell KL(P*||q), nats per cell, mean over 4 stages, seed mean (x1000). "
             "ratio = seed-paired geometric-mean KL ratio vs baseline. S_small = mean log-ratio over N=100,300,1000.", "",
             "| tag | rwv | " + " | ".join(f"N={n}: KL x1e3 (ratio)" for n in NS) + " | S_small | S_all | FLOPs/board @N=1k |",
             "|---|---|" + "---|" * len(NS) + "---|---|---|"]
    order = sorted(summ, key=lambda t: (summ[t]["S_small"] is None, summ[t]["S_small"] if summ[t]["S_small"] is not None else 0))
    for t in order:
        e = summ[t]
        cells = []
        for n in NS:
            if n in e["per_n"]:
                pn = e["per_n"][n]
                r = math.exp(pn["logratio_vs_base"]) if not math.isnan(pn["logratio_vs_base"]) else float("nan")
                cells.append(f"{1000*pn['kl_mean']:.2f} ({r:.2f}, {len(pn['seeds'])}s)")
            else:
                cells.append("")
        fb = e["per_n"].get(1000, {}).get("flops_per_board_mean")
        f3 = lambda x: f"{x:.3f}" if x is not None else ""
        lines.append(f"| {t} | {'yes' if e['rwv'] else 'NO'} | " + " | ".join(cells) +
                     f" | {f3(e['S_small'])} | {f3(e['S_all'])} | {f'{fb:.3g}' if fb else ''} |")
    open(os.path.join(a.out, pre + "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.plot:
        plot(rows, summ, a.plot.split(","), a.out, pre, a.split,
             label_tags=a.label.split(",") if a.label else None, title=a.title or None)


SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # dataviz reference order, light mode
REF_GRAY, INK, INK2, SURFACE, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


ENTITY_SLOT = {"k0": 0, "k0cell": 1, "k1": 2, "k2": 3, "masked": 4, "td": 5, "rbtd8": 6}   # same colour in every figure


def _entity(t):
    import re as _re
    return _re.sub(r"^(F-)?(E\d+-)?", "", t)


def _series(tags):
    """Fixed colour per entity across figures (never by rank); evaluation-only references get a dashed gray line."""
    out, used = [], set()
    for t in tags:
        e = _entity(t)
        if "refpost" in e:
            out.append((t, REF_GRAY, "--"))
            continue
        k = ENTITY_SLOT.get(e)
        if k is None or k in used:
            k = next(i for i in range(len(SLOTS)) if i not in used and i not in ENTITY_SLOT.values())
        used.add(k)
        out.append((t, SLOTS[k], "-"))
    return out


def plot(rows, summ, tags, out, pre, split, label_tags=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label_tags = set(label_tags or tags[:4])
    for kind in ("boards", "compute"):
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        _style(ax, fig)
        ends = []
        for t, c, ls in _series(tags):
            rs = [r for r in rows if r["tag"] == t]
            if not rs:
                continue
            g = by(rs, "n")
            ns = sorted(n for (n,) in g)
            xs = [n if kind == "boards" else float(np.mean([r["flops_at_best"] for r in g[(n,)]])) for n in ns]
            m = [float(np.mean([r["kl"] for r in g[(n,)]])) for n in ns]
            ax.plot(xs, m, ls, color=c, lw=2, marker="o", ms=5.5, label=t, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
            if kind == "boards":
                lo = [min(r["kl"] for r in g[(n,)]) for n in ns]
                hi = [max(r["kl"] for r in g[(n,)]) for n in ns]
                ax.fill_between(xs, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
            if t in label_tags:
                ends.append((xs[-1], m[-1], t))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training boards (one observed probe trajectory each)" if kind == "boards"
                      else "training compute to the early-stopping point (FLOPs, forward + backward)", color=INK2, fontsize=10)
        ax.set_ylabel(f"single-cell KL to the exact posterior\n({split}, nats per cell, seed mean)", color=INK2, fontsize=10)
        if title:
            ax.set_title(title + (" - vs boards" if kind == "boards" else " - vs compute"), color=INK, fontsize=11, loc="left")
        # direct labels (neutral ink) in a column right of the last point, spread to >= 0.07 decades apart
        if ends:
            ends.sort(key=lambda e: e[1])
            ys = [math.log10(e[1]) for e in ends]
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + 0.07)
            xr = max(e[0] for e in ends) * 1.25
            for (x, y, t), ly in zip(ends, ys):
                ax.annotate(t, (x, y), xytext=(xr, 10 ** ly), textcoords="data", va="center", fontsize=8.5, color=INK2,
                            arrowprops=dict(arrowstyle="-", color="#b9b8b2", lw=0.8, shrinkA=0, shrinkB=3),
                            annotation_clip=False)
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, labelcolor=INK)
        fig.tight_layout(rect=(0, 0, 0.88, 1))
        fig.savefig(os.path.join(out, pre + f"curve_kl_vs_{kind}.png"), dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------------------------- report tables
def md_tables(runs, split, tags, base, out_path, ns=NS):
    """Markdown tables for RESULTS.md: KL per N with 95% t-intervals, paired differences vs base, per-stage
    KL at N=1000, secondary metrics at N=1000 and the largest N."""
    rows = load(runs, split)
    L = []
    L.append(f"KL x 1e3 on {split.upper()} (mean over seeds, 95% t-interval half-width; seeds in brackets):\n")
    L.append("| arm | " + " | ".join(f"N={n}" for n in ns) + " |")
    L.append("|---|" + "---|" * len(ns))
    for t in tags:
        cells = []
        for n in ns:
            v = [r["kl"] for r in rows if r["tag"] == t and r["n"] == n]
            if v:
                m, h = tci(v)
                cells.append(f"{1000*m:.3f} +- {1000*h:.3f} [{len(v)}]" if not math.isnan(h) else f"{1000*m:.3f} [{len(v)}]")
            else:
                cells.append("")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append(f"\nPaired difference vs {base} (KL x 1e3, arm minus base, same seeds = same training boards; 95% t-interval) and geometric-mean ratio:\n")
    L.append("| arm | " + " | ".join(f"N={n}" for n in ns) + " |")
    L.append("|---|" + "---|" * len(ns))
    for t in tags:
        if t == base:
            continue
        cells = []
        for n in ns:
            A = {r["seed"]: r["kl"] for r in rows if r["tag"] == t and r["n"] == n}
            B = {r["seed"]: r["kl"] for r in rows if r["tag"] == base and r["n"] == n}
            s = sorted(set(A) & set(B))
            if len(s) >= 2:
                d = [A[k] - B[k] for k in s]
                m, h = tci(d)
                ratio = math.exp(np.mean([math.log(A[k] / B[k]) for k in s]))
                cells.append(f"{1000*m:+.3f} +- {1000*h:.3f} (x{ratio:.2f})")
            else:
                cells.append("")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    n1 = 1000
    L.append(f"\nKL x 1e3 per stage at N={n1} ({split}; stage = probes seen 0 / 2 / 4 / 6):\n")
    L.append("| arm | 0 | 2 | 4 | 6 | mean |")
    L.append("|---|---|---|---|---|---|")
    for t in tags:
        rs = [r for r in rows if r["tag"] == t and r["n"] == n1]
        if rs:
            L.append(f"| {t} | " + " | ".join(f"{1000*np.mean([r[f'kl_s{i}'] for r in rs]):.3f}" for i in range(4)) + f" | {1000*np.mean([r['kl'] for r in rs]):.3f} |")
    for nn in (1000, max(ns)):
        L.append(f"\nSecondary metrics at N={nn} ({split}; seed means): outcome log-loss per cell, ECE, joint log-lik of the true board per board-stage (independent / net / autoregressive), training FLOPs per board to the best checkpoint, steps to the best checkpoint.\n")
        L.append("| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |")
        L.append("|---|---|---|---|---|---|---|")
        for t in tags:
            rs = [r for r in rows if r["tag"] == t and r["n"] == nn]
            if not rs:
                continue
            jl = np.mean([r["joint_ll"] for r in rs if r["joint_ll"] is not None]) if any(r["joint_ll"] is not None for r in rs) else float("nan")
            ja = np.mean([r["joint_ll_ar"] for r in rs if r["joint_ll_ar"] is not None]) if any(r["joint_ll_ar"] is not None for r in rs) else float("nan")
            L.append(f"| {t} | {np.mean([r['logloss'] for r in rs]):.5f} | {np.mean([r['ece'] for r in rs]):.4f} | {jl:.2f} | "
                     + (f"{ja:.2f}" if not math.isnan(ja) else "") + f" | {np.mean([r['flops_per_board'] for r in rs]):.3g} | {np.mean([r['best_step'] for r in rs]):.0f} |")
        ex = [r["joint_ll_exact"] for r in rows if r["n"] == nn and r["joint_ll_exact"] is not None]
        if ex:
            L.append(f"\nExact posterior joint log-lik of the true board: {np.mean(ex):.2f} per board-stage (the floor any joint model can reach).")
    open(out_path, "w").write("\n".join(L) + "\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
