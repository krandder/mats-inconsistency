"""Aggregate wm2 runs (eurisko 2026-10-11).

  python3 analyze2.py --task t1 [--runs /tmp/wm2/runs] [--split val|test] [--waves W1,W2] [--plot TAG,...] [--prefix P]

Reads every forecaster / baseline run JSON (and the world-model JSONs) under RUNS/<wave>/<task>/, writes
results/<prefix><task>-<split>-runs.csv, -summary.json, -summary.md, -wm.md and optional curves.

Selection (PREREG.md; VALIDATION OUTCOMES ONLY): per N, D_N = seed-paired mean of (outcome log-loss of the arm
minus that of B-rbtd8) on the validation boards' true cells; G_N = max(seed-paired mean (log-loss B-k0 minus
B-rbtd8), 5e-5). S_out = mean over the 5 N of D_N / G_N (0 = rbtd8, -1 = as far beyond rbtd8 as rbtd8 is beyond
k0, +1 = as bad as k0). Eligible: real-world-valid, L0, not flagged (arm 4), full sweep with >= 3 seeds paired with
both baselines. Champion = lowest S_out. KL columns are reported, never used for selection.
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
RB, K0 = "B-rbtd8", "B-k0"
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(TQ.get(len(x), 2.0) * x.std(ddof=1) / math.sqrt(len(x)))


def load(runs, task, split, waves=None, strip_prefix=""):
    rows, wms = [], []
    pat = os.path.join(runs, "*", task, "*", "*.json")
    for p in sorted(glob.glob(pat)):
        wave = p[len(runs):].strip("/").split("/")[0]
        if waves and wave not in waves:
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("kind") == "wm":
            v = r.get("val", {})
            wms.append({"wave": wave, "wmtag": r["wmtag"], "n": r["n"], "seed": r["seed"], "val_nll": v.get("total"),
                        "board": v.get("board"), "loc": v.get("loc"), "read": v.get("read"), "loc_info": v.get("loc_info_nats_per_episode"),
                        "repeat": v.get("repeat_mass_per_step"), "best_step": r["best_step"], "steps_run": r["steps_run"],
                        "stop": r["stop_reason"], "train_sec": r["train_sec"], "flops_per_board": r["flops_per_board_at_best"],
                        "dream_valid": r.get("dreams", {}).get("valid_frac"), "dream_occ7": r.get("dreams", {}).get("occ7_frac"),
                        "diag_kl": r.get("diag_post", {}).get("wm_post_valid_kl"), "diag_kl_stage": r.get("diag_post", {}).get("wm_post_valid_kl_stage"),
                        "l0": r.get("l0", True)})
            continue
        if split not in r:
            continue
        v = r[split]
        tag = r.get("tag", r["arm"])
        if strip_prefix and tag.startswith(strip_prefix):
            tag = tag[len(strip_prefix):]
        fpb = r.get("flops_per_board_at_best")
        row = {"wave": wave, "tag": tag, "arm": r["arm"], "n": r["n"], "seed": r["seed"], "kl": v["kl"], "logloss": v["logloss"],
               "ece": v["ece"], "joint_ll": v.get("joint_ll"), "joint_ll_ar": v.get("joint_ll_ar"),
               "kl_unaware": v.get("kl_vs_unaware_post"),
               "best_step": r["best_step"], "steps_run": r["steps_run"], "stop": r.get("stop_reason"), "train_sec": r["train_sec"],
               "flops_per_board": fpb, "flops_per_board_total": r.get("flops_per_board_total", r.get("flops_total", 0) / r["n"]),
               "rwv": r.get("real_world_valid", True), "l0": r.get("l0", True), "flag": r.get("flag", ""), "wm_tag": r.get("wm_tag"),
               "n_dreams": r.get("n_dreams", 0), "set": " ".join(r.get("set", []))}
        for i, x in enumerate(v["kl_stage"]):
            row[f"kl_s{i}"] = x
        rows.append(row)
    return dedup(rows, ("tag", "n", "seed")), dedup(wms, ("wmtag", "n", "seed"))


def dedup(recs, key):
    """One record per (tag, N, seed) (PREREG deviation 17:21Z): a run that stopped by patience or the step cap wins over
    one stopped at the rental deadline; otherwise the original wave (W*, F) wins over a requeue wave (RQ*)."""
    best = {}
    for r in recs:
        k = tuple(r[x] for x in key)
        rank = ("deadline" in str(r.get("stop", "")), str(r.get("wave", "")).startswith("RQ"))
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return [v[1] for v in best.values()]


def paired(rows, tag, base, n, key):
    a = {r["seed"]: r[key] for r in rows if r["tag"] == tag and r["n"] == n}
    b = {r["seed"]: r[key] for r in rows if r["tag"] == base and r["n"] == n}
    s = sorted(set(a) & set(b))
    return [a[k] for k in s], [b[k] for k in s], s


def summarize(rows):
    tags = sorted({r["tag"] for r in rows})
    out = {}
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        e = {"rwv": all(r["rwv"] for r in rs), "l0": all(r["l0"] for r in rs), "flag": next((r["flag"] for r in rs if r["flag"]), ""),
             "arm": rs[0]["arm"], "wm_tag": rs[0]["wm_tag"], "per_n": {}}
        for n in NS:
            rn = [r for r in rs if r["n"] == n]
            if not rn:
                continue
            pn = {"seeds": sorted(r["seed"] for r in rn), "kl_mean": float(np.mean([r["kl"] for r in rn])),
                  "logloss_mean": float(np.mean([r["logloss"] for r in rn])),
                  "kl_stage_mean": [float(np.mean([r[f"kl_s{i}"] for r in rn])) for i in range(4)],
                  "flops_per_board_mean": float(np.mean([r["flops_per_board"] or 0 for r in rn])),
                  "flops_per_board_total_mean": float(np.mean([r["flops_per_board_total"] or 0 for r in rn])),
                  "best_step_mean": float(np.mean([r["best_step"] for r in rn])), "train_sec_mean": float(np.mean([r["train_sec"] for r in rn])),
                  "stops": sorted({r["stop"] for r in rn})}
            if any(r["kl_unaware"] is not None for r in rn):
                pn["kl_unaware_mean"] = float(np.mean([r["kl_unaware"] for r in rn if r["kl_unaware"] is not None]))
            for base, nm in ((RB, "rb"), (K0, "k0")):
                a, b, s = paired(rows, t, base, n, "logloss")
                if s:
                    m, h = tci(np.array(a) - np.array(b))
                    pn[f"dll_vs_{nm}"], pn[f"dll_vs_{nm}_ci"], pn[f"paired_{nm}"] = m, h, s
                ak, bk, s2 = paired(rows, t, base, n, "kl")
                if s2:
                    pn[f"klratio_vs_{nm}"] = float(math.exp(np.mean(np.log(np.array(ak) / np.array(bk)))))
                    m2, h2 = tci(np.array(ak) - np.array(bk))
                    pn[f"dkl_vs_{nm}"], pn[f"dkl_vs_{nm}_ci"] = m2, h2
            e["per_n"][n] = pn
        out[t] = e
    # outcome-only selection score
    gap = {}
    for n in NS:
        a, b, s = paired(rows, K0, RB, n, "logloss")
        gap[n] = max(float(np.mean(np.array(a) - np.array(b))), 5e-5) if s else None
    for t, e in out.items():
        terms, full = [], True
        for n in NS:
            pn = e["per_n"].get(n)
            if pn is None or "dll_vs_rb" not in pn or gap[n] is None or len(pn.get("paired_rb", [])) < 3 or len(pn.get("paired_k0", [])) < 3:
                full = False
                continue
            terms.append(pn["dll_vs_rb"] / gap[n])
        e["S_out"] = float(np.mean(terms)) if full and terms else None
        e["S_out_partial"] = float(np.mean(terms)) if terms else None
        e["full_sweep"] = full
        e["beats_rb_every_n_val"] = bool(full and all(e["per_n"][n]["dll_vs_rb"] < 0 for n in NS))
        lr = [math.log(e["per_n"][n]["klratio_vs_k0"]) for n in (100, 300, 1000) if n in e["per_n"] and "klratio_vs_k0" in e["per_n"][n]]
        e["S_small_kl_vs_k0"] = float(np.mean(lr)) if len(lr) == 3 else None   # round-1 style, reported only
        e["eligible"] = bool(e["rwv"] and e["l0"] and not e["flag"] and full and t not in (RB, K0))
    return out, gap


def champion(summ):
    c = [(e["S_out"], t) for t, e in summ.items() if e["eligible"] and e["S_out"] is not None]
    return min(c)[1] if c else None


def fmt(x, k=3):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{k}f}"


def md_summary(summ, gap, task, split):
    L = [f"Task {task}, split {split}. KL x1e3 = single-cell KL to the exact posterior (seed mean; reported, never used to select). "
         "dLL = seed-paired outcome log-loss difference vs B-rbtd8 x1e4 (negative = better than rbtd8). "
         "S_out = mean over N of dLL / (k0's log-loss gap to rbtd8) (selection score; 0 = rbtd8, -1 = twice rbtd8's gain over k0, +1 = k0).", "",
         "| tag | eligible | " + " | ".join(f"N={n}: KL x1e3 [dLL x1e4]" for n in NS) + " | S_out | KL ratio vs rbtd8 (geo, 5 N) | FLOPs/board @1k |",
         "|---|---|" + "---|" * len(NS) + "---|---|---|"]
    order = sorted(summ, key=lambda t: (summ[t]["S_out_partial"] is None, summ[t]["S_out_partial"] if summ[t]["S_out_partial"] is not None else 0))
    for t in order:
        e = summ[t]
        cells, rr = [], []
        for n in NS:
            pn = e["per_n"].get(n)
            if not pn:
                cells.append("")
                continue
            d = pn.get("dll_vs_rb")
            cells.append(f"{1e3 * pn['kl_mean']:.2f} [{'' if d is None else f'{1e4 * d:+.2f}'}] ({len(pn['seeds'])}s)")
            if "klratio_vs_rb" in pn:
                rr.append(math.log(pn["klratio_vs_rb"]))
        why = "yes" if e["eligible"] else ("flagged" if e["flag"] else ("not L0" if not e["l0"] else ("baseline" if t in (RB, K0) else "partial")))
        fb = e["per_n"].get(1000, {}).get("flops_per_board_mean")
        sc = fmt(e['S_out'] if e['S_out'] is not None else e['S_out_partial']) + ('' if e['S_out'] is not None else '*')
        ratio = f"{math.exp(np.mean(rr)):.2f}" if len(rr) == len(NS) else ""
        fbs = f"{fb:.2g}" if fb else ""
        L.append(f"| {t} | {why} | " + " | ".join(cells) + f" | {sc} | {ratio} | {fbs} |")
    L.append("")
    L.append("Gap k0 - rbtd8 in outcome log-loss x1e4 per N (the S_out scale): " + ", ".join(f"{n}: {fmt(1e4 * g, 2) if g else ''}" for n, g in gap.items()))
    L.append("* = partial sweep (not all 5 N with 3 paired seeds).")
    return "\n".join(L)


def md_wm(wms):
    L = ["| WM | N | seeds | val NLL (board / loc / read) nats per episode | location info | repeat mass | dream valid | WM posterior KL x1e3 (valid boards, eval-only) | best step | FLOPs/board |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(w["wmtag"], w["n"]) for w in wms})
    for wt, n in keys:
        ws = [w for w in wms if w["wmtag"] == wt and w["n"] == n]
        f = lambda k: float(np.mean([w[k] for w in ws if w[k] is not None])) if any(w[k] is not None for w in ws) else float("nan")
        L.append(f"| {wt} | {n} | {len(ws)} | {f('val_nll'):.3f} ({f('board'):.3f} / {f('loc'):.3f} / {f('read'):.3f}) | {f('loc_info'):+.4f} | {f('repeat'):.4f} | "
                 f"{f('dream_valid'):.3f} | {1e3 * f('diag_kl'):.3f} | {f('best_step'):.0f} | {f('flops_per_board'):.2g} |")
    return "\n".join(L)


SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DESC = {"B-k0": "k0, plain 0/1 targets", "B-rbtd8": "rbtd8, round-1 champion",
        "WMr-D-rbtd8": "TD-trained WM + rbtd8 on dreams and real (champion)", "WMs-D-rbtd8": "split WM + rbtd8 on dreams and real",
        "WMe-D-rbtd8": "3-WM ensemble + rbtd8 on dreams and real", "WMr-D-only": "TD-trained WM, dreams only",
        "WMs-D-only": "split WM, dreams only", "WMc-D-only": "joint WM, dreams only", "WMc-S-only": "joint WM shared with the forecaster",
        "WMs-D-post": "split WM, own-posterior targets (arm 4, FLAGGED)", "WMsb-D-only": "board-blind WM, dreams only (not L0)",
        "WMrb-D-only": "board-blind TD WM, dreams only (not L0)", "WMrb-D-rbtd8": "board-blind TD WM + rbtd8 (not L0)"}
REF_GRAY, INK, INK2, SURFACE, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def plot(rows, tags, out_png, split, title, kind="boards", ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fixed = {K0: 0, RB: 1}
    used = set(fixed.values())
    ends = []
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        if not rs:
            continue
        k = fixed.get(t)
        if k is None:
            k = next(i for i in range(len(SLOTS)) if i not in used)
            used.add(k)
        c = SLOTS[k % len(SLOTS)]
        ls = "--" if ("D-post" in t or t.startswith("WMsb") or t.startswith("WMrb")) else "-"
        ns = sorted({r["n"] for r in rs})
        xs = [n if kind == "boards" else float(np.mean([r["flops_per_board"] * r["n"] for r in rs if r["n"] == n])) for n in ns]
        m = [float(np.mean([r["kl"] for r in rs if r["n"] == n])) for n in ns]
        ax.plot(xs, m, ls, color=c, lw=2, marker="o", ms=5.5, label=f"{t}: {DESC[t]}" if t in DESC else t, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.0)
        if kind == "boards":
            lo = [min(r["kl"] for r in rs if r["n"] == n) for n in ns]
            hi = [max(r["kl"] for r in rs if r["n"] == n) for n in ns]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
        ends.append((xs[-1], m[-1], t))
    if ref is not None and kind == "boards":
        ax.axhline(ref[1], color=REF_GRAY, ls="--", lw=1.2, zorder=2)
        ax.annotate(ref[0], (min(NS), ref[1]), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training boards (one observed episode each)" if kind == "boards" else "training compute to the early-stopping point, all phases (FLOPs)", color=INK2, fontsize=10)
    ax.set_ylabel(f"single-cell KL to the exact posterior\n({split}, nats per cell, seed mean)", color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ends:
        ends.sort(key=lambda e: e[1])
        ys = [math.log10(e[1]) for e in ends]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + 0.07)
        xr = max(e[0] for e in ends) * 1.25
        for (x, y, t), ly in zip(ends, ys):
            ax.annotate(t, (x, y), xytext=(xr, 10 ** ly), textcoords="data", va="center", fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color="#b9b8b2", lw=0.8, shrinkA=0, shrinkB=3), annotation_clip=False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/tmp/wm2/runs")
    ap.add_argument("--task", default="t1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--waves", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--ref", default="", help="LABEL=VALUE horizontal reference line (boards plot)")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--strip", default="", help="strip this tag prefix (e.g. F- for final runs)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    waves = [w for w in a.waves.split(",") if w] or None
    rows, wms = load(a.runs, a.task, a.split, waves, a.strip)
    base = os.path.join(a.out, f"{a.prefix}{a.task}-{a.split}")
    if rows:
        with open(base + "-runs.csv", "w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    summ, gap = summarize(rows)
    ch = champion(summ)
    json.dump({"summary": summ, "gap_k0_minus_rb_logloss": gap, "champion_by_S_out": ch}, open(base + "-summary.json", "w"), indent=1)
    txt = md_summary(summ, gap, a.task, a.split) + f"\n\nChampion by the pre-registered rule (lowest S_out among eligible): **{ch}**\n"
    open(base + "-summary.md", "w").write(txt)
    if wms:
        open(base + "-wm.md", "w").write(md_wm(wms) + "\n")
    print(txt)
    if wms:
        print(md_wm(wms))
    if a.plot:
        tags = a.plot.split(",")
        for kind in ("boards", "compute"):
            ref = (a.ref.split("=")[0], float(a.ref.split("=")[1])) if a.ref else None
            plot(rows, tags, base + f"-curve_kl_vs_{kind}.png", a.split, (a.title or f"{a.task} {a.split}") + f" - vs {kind}", kind, ref=ref)


if __name__ == "__main__":
    main()
