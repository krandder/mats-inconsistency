"""Build results/index.html (and summary.json) from whatever snapshots exist. Safe to rerun."""
import glob, io, json, os, re, sys, time, html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "pi", "and", "or"]
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
INK = "#123456"      # placeholder, becomes currentColor
GRID = "#654321"     # placeholder, becomes var(--grid)
GRAN = ["out", "head", "neuron", "comp"]
GNAME = {"out": "output readouts (per question)", "head": "attention heads", "neuron": "MLP neurons", "comp": "layer components"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID, "axes.facecolor": "none",
                     "figure.facecolor": "none", "savefig.facecolor": "none", "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "svg.fonttype": "none", "legend.frameon": False})


_N = [0]


def svg(fig):
    if os.environ.get("PNGDIR"):
        _N[0] += 1
        fig.savefig(os.path.join(os.environ["PNGDIR"], f"fig{_N[0]}.png"), dpi=80, facecolor="white")
    b = io.StringIO(); fig.savefig(b, format="svg", bbox_inches="tight"); plt.close(fig)
    s = b.getvalue()
    s = s[s.find("<svg"):]
    s = s.replace(INK, "currentColor").replace(GRID, "var(--grid)")
    for i, c in enumerate(LIGHT):
        s = s.replace(c, f"var(--s{i+1})")
    s = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', 'width="100%"', s, count=1)
    return f'<div class="fig">{s}</div>'


def unit_names(g, n, L=2, H=4):
    if g == "out":
        return [f"out:q{i}" for i in range(n)]
    if g == "head":
        return [f"L{i // H}.head{i % H}" for i in range(n)]
    if g == "neuron":
        N = n // L
        return [f"L{i // N}.n{i % N}" for i in range(n)]
    return ["emb.window", "emb.qid"] + [f"L{i // 2}.{'attn' if i % 2 == 0 else 'mlp'}" for i in range(n - 2)]


def load(res):
    runs = {}
    for d in sorted(glob.glob(os.path.join(res, "*", "snapshots"))):
        tag = os.path.basename(os.path.dirname(d))
        files = sorted(glob.glob(os.path.join(d, "step-*.json")))
        snaps = []
        for f in files:
            try:
                snaps.append(json.load(open(f)))
            except Exception:
                pass
        latest = None
        try:
            latest = json.load(open(os.path.join(res, tag, "latest.json")))
        except Exception:
            latest = snaps[-1] if snaps else None
        meta = None
        try:
            meta = json.load(open(os.path.join(res, tag + ".json")))
        except Exception:
            pass
        if snaps:
            runs[tag] = {"snaps": snaps, "latest": latest, "meta": meta}
    return runs


def gini_pos(x):
    x = np.sort(np.clip(x, 0, None))
    return 0.0 if x.sum() <= 0 else float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum()))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


def run_stats(r):
    """Per-run numbers for RESULTS.md (all seeds, all modes)."""
    sn, lat = r["snaps"], r["latest"]
    st = {}
    ncut = len(sn[0]["cut"]); Lh = (ncut - 1) // 2
    bidx = np.array([-1] + sum([[2 * l] * 3 + [2 * l + 1] * 3 for l in range(Lh)], []) + [2 * Lh, 2 * Lh])
    c, b = np.array(lat["cut"]), np.array(lat["bias"])
    st["cons_resid_cum"] = [float((lat["R_total"] - c[k] - b[bidx >= k].sum()) / abs(lat["R_total"])) for k in range(ncut)]
    st["frozen_attn_resid_max"] = float(max(np.abs(s["frozen_attn_resid_rel"]).max() for s in sn))
    st["sum_resid_max"] = float(max(s["sum_resid_max"] for s in sn))
    rho = {}
    for g in GRAN:
        gn = np.array(lat["gradnorm_cum"][g]).ravel()
        if gn.sum() > 0:
            w = np.array(lat["wealth"][g]); rho[g] = spearman(gn, w); rho[g + "_abs"] = spearman(gn, np.abs(w))
    st["rho"] = rho
    half = sn[len(sn) // 2:]
    par = {}
    for g in ["head", "neuron", "out"]:
        Wm = np.array([s["wealth"][g] for s in half]); par[g] = np.where((Wm < 0).mean(0) >= 0.9)[0].tolist()
    st["parasites"] = {g: len(v) for g, v in par.items()}
    act = np.array(lat.get("act_per_example_cum") or []).ravel()
    if act.size and act.sum() > 0:
        st["parasite_neuron_act_share"] = float(act[par["neuron"]].sum() / act.sum())
        st["parasite_neuron_frac"] = len(par["neuron"]) / act.size
        st["dead_neurons"] = int((act == 0).sum())
    tv = [s["turnover_income_rank_corr"] for s in sn[len(sn) // 2:]]
    st["income_rank_corr_2nd_half_mean"] = {g: float(np.nanmean([np.nan if x[g] is None else x[g] for x in tv])) for g in GRAN}
    wv = [s["wealth_rank_corr_vs_1000_steps_ago"] for s in sn[len(sn) // 2:]]
    st["wealth_rank_corr_2nd_half_mean"] = {g: float(np.nanmean([np.nan if x[g] is None else x[g] for x in wv])) for g in GRAN}
    st["gini_first"] = sn[0]["gini_pos"]
    st["R_per_step"] = lat["R_total"] / max(1, lat["step"])
    st["ev_bce"] = lat["ev_bce"]; st["ev_bce_log"] = lat["ev_bce_log"]
    st["wealth_by_family_out"] = lat["wealth_by_family"]["out"]
    st["comp_wealth"] = lat["wealth"]["comp"]
    return st


def fmt(x, n=3):
    return "n/a" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))


def main(res):
    runs = load(res)
    trains = [t for t in runs if t.startswith("train")]
    main_tag = "train-s1" if "train-s1" in runs else (trains[0] if trains else None)
    out = []
    summ = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    if main_tag is None:
        open(os.path.join(res, "index.html"), "w").write("<p>No snapshots yet.</p>")
        return
    R = runs[main_tag]
    snaps, lat = R["snaps"], R["latest"]
    steps = np.array([s["step"] for s in snaps])
    for t, r in runs.items():
        s = r["latest"]
        summ["runs"][t] = {"step": s["step"], "elapsed_s": s["elapsed_s"], "loss": s["loss"], "R_total": s["R_total"],
                           "gini_pos": s["gini_pos"], "n_negative": s["n_negative"],
                           "ev_bce_mean": float(np.mean(s["ev_bce"])), "ev_bce_log_mean": float(np.mean(s["ev_bce_log"]))}

    # ---------- status ----------
    rows = "".join(f"<tr><td>{t}</td><td>{r['latest']['mode']}</td><td>{r['latest']['step']}</td><td>{r['latest']['elapsed_s']:.0f}</td>"
                   f"<td>{r['latest']['loss']:.4f}</td><td>{np.mean(r['latest']['ev_bce']):.4f}</td><td>{np.mean(r['latest']['ev_bce_log']):.4f}</td>"
                   f"<td>{r['latest']['R_total']:.1f}</td><td>{'done' if os.path.exists(os.path.join(res, t + '.json')) and r['meta'] and r['meta'].get('steps_done') == r['latest']['step'] else 'running or partial'}</td></tr>"
                   for t, r in runs.items())
    out.append(f"<h2>Runs</h2><p class=note>Checked at {summ['checked_at']} by reading every snapshot under <code>{html.escape(os.path.abspath(res))}</code>. "
               f"Snapshot cadence: every 100 steps. Main run shown in the charts: <b>{main_tag}</b>.</p>"
               f"<div class=tw><table><tr><th>run</th><th>mode</th><th>step</th><th>seconds</th><th>train loss (last 100)</th><th>eval BCE net</th><th>eval BCE logistic</th><th>total wealth R</th><th>state</th></tr>{rows}</table></div>")

    # ---------- loss by family ----------
    fam_rows = "".join(f"<tr><td>{f}</td><td>{lat['ev_bce'][i]:.3f}</td><td>{lat['ev_bce_log'][i]:.3f}</td><td>{lat['ev_brier'][i]:.3f}</td><td>{lat['ev_brier_log'][i]:.3f}</td></tr>" for i, f in enumerate(FAM))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, f in enumerate(FAM):
        ax.plot(steps, [s["ev_bce"][i] for s in snaps], color=LIGHT[i], label=f)
    ax.set_xlabel("step"); ax.set_ylabel("held-out BCE (nats)"); ax.legend(ncol=4, fontsize=8)
    out.append(f"<h2>What the model learns</h2><p>Held-out cross-entropy per family for {main_tag} (later 20% of each stream). 0.693 = coin flip.</p>{svg(fig)}"
               f"<div class=tw><table><tr><th>family</th><th>BCE net</th><th>BCE logistic</th><th>Brier net</th><th>Brier logistic</th></tr>{fam_rows}</table></div>")

    # ---------- trajectories ----------
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for ax, g in zip(axs.ravel(), GRAN):
        Wm = np.array([s["wealth"][g] for s in snaps])     # [T, n]
        names = unit_names(g, Wm.shape[1])
        order = np.argsort(Wm[-1])
        k = min(4, max(1, Wm.shape[1] // 2))
        pick = list(order[-k:][::-1]) + list(order[:k])
        for j, u in enumerate(pick):
            c = LIGHT[0] if j < k else LIGHT[1]
            ax.plot(steps, Wm[:, u], color=c, alpha=1.0 - 0.18 * (j % k))
            ax.annotate(names[u], (steps[-1], Wm[-1, u]), fontsize=7, xytext=(3, 0), textcoords="offset points", va="center")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(f"{GNAME[g]}: top {k} (blue), bottom {k} (orange)", fontsize=9)
        ax.set_xlabel("step"); ax.set_ylabel("cumulative wealth")
    fig.tight_layout()
    out.append(f"<h2>Wealth trajectories</h2><p>Wealth = cumulative revenue per example, where revenue = activation x d(settlement)/d(activation). "
               f"Richest and poorest units at the latest snapshot, traced back over training.</p>{svg(fig)}")

    # ---------- concentration & turnover ----------
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for i, g in enumerate(GRAN):
        axs[0].plot(steps, [s["gini_pos"][g] for s in snaps], color=LIGHT[i], label=g)
        tv = [s["turnover_income_rank_corr"][g] for s in snaps]
        axs[1].plot(steps, [np.nan if v is None else v for v in tv], color=LIGHT[i], label=g, lw=1)
        wv = [s["wealth_rank_corr_vs_1000_steps_ago"][g] for s in snaps]
        axs[2].plot(steps, [np.nan if v is None else v for v in wv], color=LIGHT[i], label=g)
    axs[0].set_title("Gini of positive wealth", fontsize=9); axs[1].set_title("income rank corr, window vs previous window", fontsize=9)
    axs[2].set_title("wealth rank corr vs 1000 steps earlier", fontsize=9)
    for a in axs: a.set_xlabel("step")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    negs = " ".join(f"{g}: {lat['n_negative'][g]}/{len(lat['wealth'][g])}" for g in GRAN)
    out.append(f"<h2>Concentration and turnover</h2><p>Gini is computed on max(wealth, 0) because wealth can be negative. "
               f"Units with negative wealth now: {negs}. Income rank correlation near 1 means the same units earn every 100 steps; "
               f"near 0 means income is reshuffled.</p>{svg(fig)}")

    # ---------- wealth vs gradient norm ----------
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    rho = {}
    for ax, g in zip(axs, GRAN):
        w = np.array(lat["wealth"][g]); gn = np.array(lat["gradnorm_cum"][g]).ravel() / max(1, lat["step"])
        if gn.sum() == 0:
            ax.set_title(f"{g}: no training", fontsize=9); continue
        ax.scatter(gn, w, s=10 if g == "neuron" else 22, color=LIGHT[0], alpha=0.6, edgecolors="none")
        rho[g] = spearman(gn, w); rho[g + "_abs"] = spearman(gn, np.abs(w))
        ax.set_xscale("log"); ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(f"{g}: rho={rho[g]:.2f}, |W| rho={rho[g+'_abs']:.2f}", fontsize=9); ax.set_xlabel("mean loss-gradient norm")
    axs[0].set_ylabel("wealth")
    fig.tight_layout()
    summ["runs"][main_tag]["rho_wealth_gradnorm"] = rho
    out.append(f"<h2>Wealth vs loss-gradient norm</h2><p>Spearman rho between a unit's mean loss-gradient norm (its parameters) and its wealth, "
               f"and between gradient norm and |wealth|.</p>{svg(fig)}")

    # ---------- wealth by family ----------
    comp = np.array(lat["comp_by_family"]); head = np.array(lat["head_by_family"]).reshape(-1, 8)
    M = np.vstack([comp, head]); names = unit_names("comp", comp.shape[0]) + unit_names("head", head.shape[0])
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(names) + 1))
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels(FAM, rotation=30); ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.grid(False)
    for (i, j), x in np.ndenumerate(M):
        ax.text(j, i, f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(x) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fam_tot = np.array(lat["wealth_by_family"]["out"])
    out.append(f"<h2>Which families fund whom</h2><p>Wealth of layer components and heads split by the question family whose settlement paid it "
               f"(blue positive, red negative). Total wealth paid out per family at the output: "
               + ", ".join(f"{f} {fam_tot[i]:.0f}" for i, f in enumerate(FAM)) + f".</p>{svg(fig)}")

    # ---------- conservation ----------
    ncut = len(snaps[0]["cut"]); Lh = (ncut - 1) // 2
    bidx = np.array([-1] + sum([[2 * l] * 3 + [2 * l + 1] * 3 for l in range(Lh)], []) + [2 * Lh, 2 * Lh])

    def cum_resid(sn):
        c, b = np.array(sn["cut"]), np.array(sn["bias"])
        return np.array([(sn["R_total"] - c[k] - b[bidx >= k].sum()) / abs(sn["R_total"]) for k in range(ncut)])
    cons = np.array([cum_resid(s) for s in snaps]); fz = np.array([s["frozen_attn_resid_rel"] for s in snaps])
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.8))
    for k in range(cons.shape[1]):
        axs[0].plot(steps, cons[:, k], color=LIGHT[k % 8], label=f"cut {k}")
        axs[1].plot(steps, np.abs(fz[:, k]) + 1e-12, color=LIGHT[k % 8], label=f"cut {k}")
    axs[0].set_title("live books, cumulative: (R - cut - biases below)/|R|", fontsize=9); axs[0].legend(fontsize=7, ncol=3)
    axs[1].set_yscale("log"); axs[1].set_title("frozen-attention check: |residual|/|R| (float32)", fontsize=9)
    for a in axs: a.set_xlabel("step")
    fig.tight_layout()
    cut = np.array(lat["cut"]); bias = np.array(lat["bias"])
    sr_max = max(s["sum_resid_max"] for s in snaps)
    cons_tab = "".join(f"<tr><td>cut {k} ({'embedding' if k == 0 else ('after L%d.%s' % ((k - 1) // 2, 'attn' if k % 2 else 'mlp'))})</td><td>{cut[k]:.2f}</td>"
                       f"<td>{lat['R_total']:.2f}</td><td>{cum_resid(lat)[k]:.4f}</td></tr>" for k in range(len(cut)))
    summ["runs"][main_tag]["cons_resid_cum"] = cum_resid(lat).tolist()
    out.append(f"<h2>Do the books balance?</h2><p>Each residual-stream cut is a complete layer of suppliers, so its cumulative revenue plus the revenue of "
               f"biases below it should equal the total settlement R. With attention patterns frozen the identity is exact (right panel, float32 noise), "
               f"which checks the code. With live attention the softmax is not homogeneous, so part of each attention layer's revenue is not passed to "
               f"its suppliers (left panel). Sum rule (layer = sum of its heads or neurons + its output bias): worst per-step residual {sr_max:.2e}.</p>{svg(fig)}"
               f"<div class=tw><table><tr><th>cut</th><th>cumulative cut revenue</th><th>total R</th><th>(R - cut - biases below)/|R|</th></tr>{cons_tab}</table></div>")

    # ---------- rich & poor table ----------
    allw, alln, allf = [], [], []
    for g in ["out", "head", "neuron"]:
        allw += list(lat["wealth"][g]); alln += unit_names(g, len(lat["wealth"][g]))
    fam_by = {}
    for g, key in (("head", "head_by_family"), ("neuron", "neuron_by_family")):
        arr = np.array(lat.get(key) or []).reshape(-1, 8) if lat.get(key) is not None else None
        if arr is not None:
            for n, row in zip(unit_names(g, arr.shape[0]), arr):
                fam_by[n] = row
    fams = np.array(lat["fam"]) if "fam" in lat else None
    allw = np.array(allw); order = np.argsort(allw)

    def famstr(n, w):
        if n.startswith("out:q"):
            q = int(n[5:])
            return "own question only"
        row = fam_by.get(n)
        if row is None:
            return ""
        tot = np.abs(row).sum() or 1
        top = np.argsort(-np.abs(row))[:3]
        return ", ".join(f"{FAM[i]} {row[i] / tot * 100:+.0f}%" for i in top)
    act = np.array(lat.get("act_per_example_cum") or []).ravel()
    trows = ""
    for lab, idx in (("richest", order[::-1][:10]), ("poorest", order[:10])):
        for i in idx:
            trows += f"<tr><td>{lab}</td><td>{alln[i]}</td><td>{allw[i]:.2f}</td><td>{famstr(alln[i], allw[i])}</td></tr>"
    out.append(f"<h2>Ten richest and ten poorest units</h2><p>Across readouts, heads and neurons (components excluded). Family shares are signed shares of "
               f"the unit's absolute revenue from the three biggest families.</p><div class=tw><table><tr><th></th><th>unit</th><th>wealth</th><th>funded by</th></tr>{trows}</table></div>")

    # ---------- parasites ----------
    half = snaps[len(snaps) // 2:]
    par = []
    for g in ["head", "neuron", "out"]:
        Wm = np.array([s["wealth"][g] for s in half])
        neg = (Wm < 0).mean(0)
        nm = unit_names(g, Wm.shape[1])
        for u in np.where(neg >= 0.9)[0]:
            par.append((g, nm[u], float(Wm[-1, u]), float(neg[u]), u))
    ptxt = ""
    if act.size and lat["step"]:
        tot_act = act.sum()
        pn = [p for p in par if p[0] == "neuron"]
        share = sum(act[p[4]] for p in pn) / tot_act if tot_act else 0
        dead = int((act == 0).sum())
        ptxt = (f"Neuron parasites use {share * 100:.1f}% of all neuron activations (a neuron 'activates' when its ReLU output is positive). "
                f"{dead} of {act.size} neurons never activated.")
        summ["runs"][main_tag]["parasite_neuron_act_share"] = share
    summ["runs"][main_tag]["parasites"] = {g: sum(1 for p in par if p[0] == g) for g in ["head", "neuron", "out"]}
    prow = "".join(f"<tr><td>{p[1]}</td><td>{p[2]:.2f}</td><td>{p[3] * 100:.0f}%</td></tr>" for p in sorted(par, key=lambda p: p[2])[:25])
    out.append(f"<h2>Parasites</h2><p>A parasite here is a unit whose cumulative wealth was negative in at least 90% of the snapshots in the second half "
               f"of the run so far. Count: " + ", ".join(f"{g} {summ['runs'][main_tag]['parasites'][g]}" for g in ["head", "neuron", "out"]) +
               f". {ptxt}</p><div class=tw><table><tr><th>unit</th><th>wealth</th><th>share of snapshots negative</th></tr>{prow}</table></div>")

    # ---------- ablation ----------
    abl = os.path.join(res, "ablation.csv")
    if os.path.exists(abl):
        import csv
        A = list(csv.DictReader(open(abl)))
        for r_ in A:
            for k in ("seed", "step", "final", "k_removed", "n_units", "n_negative_wealth"):
                r_[k] = int(r_[k])
            for k in list(r_):
                if k.startswith(("bce", "brier", "frac")):
                    r_[k] = float(r_[k])
        seeds = sorted({r_["seed"] for r_ in A})
        CR = ["wealth", "recent", "random", "gradnorm", "actmag"]
        CL = {"wealth": LIGHT[1], "recent": LIGHT[4], "random": LIGHT[0], "gradnorm": LIGHT[2], "actmag": LIGHT[6]}
        fig, axs = plt.subplots(2, len(seeds), figsize=(3.4 * len(seeds), 5.2), squeeze=False)
        for j, sd in enumerate(seeds):
            for i, g in enumerate(["neuron", "head"]):
                ax = axs[i, j]
                for c in CR:
                    pts = sorted((r_["frac"], r_["bce"]) for r_ in A if r_["seed"] == sd and r_["final"] and r_["granularity"] == g and r_["criterion"] == c)
                    if pts:
                        ax.plot([100 * p_[0] for p_ in pts], [p_[1] for p_ in pts], color=CL[c], marker="o", ms=3, label=c,
                                ls="--" if c == "random" else "-")
                ax.set_title(f"seed {sd}, {g}s (final)", fontsize=9); ax.set_xlabel("% removed"); ax.set_ylabel("held-out BCE")
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout()
        # summary table: BCE increase at 10% and 20% neurons, 25%(2 of 8) heads, all checkpoints
        trs = ""
        base = {(r_["seed"], r_["step"], r_["granularity"]): r_["bce"] for r_ in A if r_["frac"] == 0 and r_["criterion"] == "random"}
        for key in sorted({(r_["seed"], r_["step"], r_["granularity"]) for r_ in A}):
            sd, st_, g = key
            for fr in ([0.05, 0.1, 0.2, 0.4] if g == "neuron" else [0.2, 0.4]):
                cells = []
                for c in CR:
                    v = [r_ for r_ in A if (r_["seed"], r_["step"], r_["granularity"]) == key and r_["criterion"] == c and abs(r_["frac"] - fr) < 1e-9]
                    cells.append(f"{v[0]['bce'] - base[key]:+.4f}" if v else "")
                nneg = [r_["n_negative_wealth"] for r_ in A if (r_["seed"], r_["step"], r_["granularity"]) == key][0]
                k_ = [r_["k_removed"] for r_ in A if (r_["seed"], r_["step"], r_["granularity"]) == key and abs(r_["frac"] - fr) < 1e-9][0]
                trs += f"<tr><td>{sd}</td><td>{st_}</td><td>{g}</td><td>{nneg}</td><td>{fr * 100:.0f}% ({k_})</td><td>{base[key]:.4f}</td>" + "".join(f"<td>{c_}</td>" for c_ in cells) + "</tr>"
        out.append(f"<h2>Does deleting the poorest traders help?</h2><p>Units are zeroed in order of each criterion and held-out BCE is measured on a fresh "
                   f"stream segment (4096 positions after everything used in training). Random is the control (mean of 5 draws, dashed). "
                   f"Curves: final checkpoint. Table: change in BCE from the intact model, all checkpoints; lower is better. Data: <code>ablation.csv</code>.</p>{svg(fig)}"
                   f"<div class=tw><table><tr><th>seed</th><th>step</th><th>units</th><th>negative wealth</th><th>removed</th><th>intact BCE</th>"
                   + "".join(f"<th>{c}</th>" for c in CR) + f"</tr>{trs}</table></div>")

    # ---------- decayed wealth ----------
    from decay_section import section as _decay_section
    _d = _decay_section(res, svg, LIGHT)
    if _d:
        out.append(_d)

    from settle_section import section as _settle_section
    _s = _settle_section(res, svg, LIGHT)
    if _s:
        out.append(_s)

    from blocklocal_section import section as _bl_section
    _b = _bl_section(res, svg, LIGHT)
    if _b:
        out.append(_b)

    # ---------- controls ----------
    ctl = [t for t in runs if not t.startswith("train")]
    if ctl:
        crow = ""
        for t in sorted(runs):
            s = runs[t]["latest"]
            ns = max(1, s["step"])
            crow += (f"<tr><td>{t}</td><td>{s['step']}</td><td>{s['R_total'] / ns:.4f}</td>"
                     + "".join(f"<td>{s['gini_pos'][g]:.2f}</td>" for g in GRAN) + f"<td>{np.mean(s['ev_bce']):.3f}</td></tr>")
        out.append(f"<h2>Controls</h2><p>frozen = the trained weights of the same seed, no updates, books kept for 500 fresh steps; "
                   f"random = a fresh initialisation, no updates, 500 steps.</p><div class=tw><table><tr><th>run</th><th>steps</th><th>R per step</th>"
                   + "".join(f"<th>Gini {g}</th>" for g in GRAN) + f"<th>eval BCE</th></tr>{crow}</table></div>")

    for t, r in runs.items():
        summ["runs"][t].update(run_stats(r))
    json.dump(summ, open(os.path.join(res, "summary.json"), "w"), indent=1)
    css_l = "".join(f"--s{i+1}:{c};" for i, c in enumerate(LIGHT)); css_d = "".join(f"--s{i+1}:{c};" for i, c in enumerate(DARK))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wealth net books</title><meta http-equiv="refresh" content="120">
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--grid:#d9d8d2;--line:#e6e5df;{css_l}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#f4f4f1;--muted:#c3c2b7;--grid:#3d3d3a;--line:#2e2e2c;{css_d}}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#f4f4f1;--muted:#c3c2b7;--grid:#3d3d3a;--line:#2e2e2c;{css_d}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px}}
main{{max-width:1100px;margin:24px auto}} h1{{font-size:22px}} h2{{font-size:17px;margin-top:32px;border-top:1px solid var(--line);padding-top:16px}}
.note,p{{color:var(--muted)}} .fig{{color:var(--fg);max-width:100%;overflow:hidden}} .fig svg{{height:auto;max-width:100%}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}} td,th{{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}}
code{{font-size:12px}}
</style></head><body><main>
<h1>Wealth accounting on a small transformer</h1>
<p>Every unit is a trader. The output settles like an LMSR market: revenue = logit contribution x (y - p). Money flows backward along the chain rule,
activation x d(downstream revenue)/d(activation). Nothing is taxed or rerouted; these are only books kept on ordinary training.
This page rebuilds from the latest snapshots about every 2 minutes while the run goes.</p>
{''.join(out)}
</main></body></html>"""
    open(os.path.join(res, "index.html"), "w").write(page)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
