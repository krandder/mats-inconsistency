"""Phase 3 analysis (RESULTS.md section 17): the ablation, the Bansal DT-recall baseline, the
regularizer arms and the L3200 / extreme-depth re-evaluations.

Reads results/vm/trader-mresnet-{10..25}/p3/ (10-15 phase 3; 16-18 phase 3b, the audit's controls; 19-25 re-evaluation
only, added 21:50 UTC; a file on several VMs is taken from the highest-numbered one) and, for the two phase-2 reference arms
(econ_cov_voc, bptt_cov_dscov), the latest VM copy under results/vm/*/p2/.
Writes results/p3_summary.json, charts/p3_*.json, and prints the markdown tables.

Metric: exact-string accuracy (every output bit's sign right), one task (prefix-sum parity).
Per arm: mean over seeds with a 95% t-interval, and the strings pooled over seeds with a 95%
Wilson interval (k of n strings exactly right).
"""
import glob, json, math, os, re
from scipy import stats

D0 = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D0, "results")
P3 = [os.path.join(R, "vm", f"trader-mresnet-{i}", "p3") for i in range(10, 26)]   # (not used by main; find() globs all VMs)
SEEDS = [0, 1, 2]


def vmnum(p):
    return int(re.search(r"trader-mresnet-(\d+)", p).group(1))


def find(name, sub="p3"):
    """latest VM copy of results/vm/*/<sub>/<name>"""
    c = sorted(glob.glob(os.path.join(R, "vm", "*", sub, name)), key=vmnum)
    return c[-1] if c else None


def load(arm, s):
    old = arm in ("econ_cov_voc", "bptt_cov_dscov")
    p = find(f"{arm}_s{s}.json", "p2" if old else "p3")
    return json.load(open(p)) if p else None


def load_re(arm, s):
    old = arm in ("econ_cov_voc", "bptt_cov_dscov")
    p = find(f"re_{'old_' if old else ''}{arm}_s{s}.json")
    return json.load(open(p)) if p else None


def ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return m, stats.t.ppf(0.975, len(xs) - 1) * sd / len(xs) ** 0.5, len(xs)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def ev(r, L, D, key="acc"):
    for e in (r or {}).get("eval", []):
        if e["L"] == L and e["D"] == D and e.get("pick", "roundrobin") in ("roundrobin", "-"):
            return e[key]
    return None


def rrow(r, L, D, clamp=0.0):
    for x in (r or {}).get("rows", []):
        if x["L"] == L and x["D"] == D and x["clamp"] == clamp:
            return x
    return None


def fmt(m, h, n=None, d=2):
    if m is None:
        return "-"
    s = f"{m:.{d}f}" if abs(m) < 1000 else f"{m:,.0f}"
    if h is not None and h > 0:
        s += f" +- {h:.{d}f}" if abs(h) < 1000 else f" +- {h:,.0f}"
    if n is not None and n < 3:
        s += f" (n={n})"
    return s


def fnll(m, h=None, n=None):
    if m is None:
        return "-"
    if m >= 100:
        s = f"{m:,.0f}"
    elif m >= 1:
        s = f"{m:.1f}"
    else:
        s = f"{m:.3f}"
    if n is not None and n < 3:
        s += f" (n={n})"
    return s


ABL = [("econ_cov_voc", "economy, random order (phase 2)"),
       ("econ_cov", "economy without VOC markets"),
       ("econ_cov_voc_rr", "economy, round-robin order in training"),
       ("bptt_cov_dscov", "BPTT full graph, round-robin (phase 2)"),
       ("bptt_cov_dscov_rand", "BPTT full graph, random order"),
       ("bptt_cov_dscov_tb1", "BPTT truncated to 1 step, round-robin"),
       ("bptt_cov_dscov_tb1_rand", "BPTT truncated to 1 step, random order"),
       ("bptt_cov_dscov_voc_rand", "BPTT full graph, random order, v on the VOC targets"),
       ("bptt_cov_dscov_tb1_voc_rand", "BPTT truncated to 1 step, random order, v on the VOC targets"),
       ("bptt_cov_dscov_tb1_voc_rep_rand", "the economy's gradient as BPTT(1): + per-trader Adam/clip, bid head"),
       ("econ_cov_online", "economy, online update, no VOC (the 4.2 GB configuration)"),
       ("bptt_cov_progcov_a0", "BPTT loss at depth 24 only + covariance targets (alpha 0)"),
       ("bptt_cov_progcov_a05", "BPTT progressive loss alpha 0.5 + covariance targets"),
       ("bptt_cov_progcov", "BPTT progressive loss alpha 1 + covariance targets"),
       ("dtr_eqflop", "DT-recall width 96, equal FLOPs (3090 it)"),
       ("dtr_eqtime", "DT-recall width 400, equal time (700 s)"),
       ("dtr_full", "DT-recall width 400, published schedule (12000 it)")]
REG = [("econ_cov_voc", "unregularized (phase 2)"),
       ("econ_cov_voc_floor", "eps floor 0.01"),
       ("econ_cov_voc_l2b", "L2 fee on belief logits 1e-3"),
       ("econ_cov_voc_bcap", "logit cap +-10 (bounded liquidity)"),
       ("econ_cov_voc_dbcap", "trade-size cap 1 (tanh)"),
       ("econ_cov_voc_liqt", "liquidity b_t = t"),
       ("econ_cov_voc_liqsqrt", "liquidity b_t = sqrt(t)"),
       ("bptt_cov_dscov_tb1_voc_rand", "no market: the economy's BPTT(1) twin, uncapped"),
       ("bptt_cov_dscov_tb1_voc_bcap_rand", "no market: the BPTT(1) twin with the same cap +-10"),
       ("bptt_cov_dscov_bcap", "no market: full BPTT, round-robin, with the cap +-10"),
       ("dtr_full", "Bansal DT-recall, published schedule"),
       ("dtr_eqflop", "Bansal DT-recall, equal FLOPs")]
POINTS = [(32, 32, "1x"), (320, 320, "10x"), (3200, 3200, "100x")]


def main():
    out = {"ablation": {}, "reg": {}, "reeval": {}, "deep": {}, "early": {}, "train": {}}
    arms = list(dict.fromkeys([a for a, _ in ABL] + [a for a, _ in REG]))
    J = {a: {s: load(a, s) for s in SEEDS} for a in arms}
    RE = {a: {s: load_re(a, s) for s in SEEDS} for a in arms}

    # training cost
    for a in arms:
        secs = [J[a][s].get("train_sec") for s in SEEDS if J[a][s]]
        its = [J[a][s].get("train_iters", J[a][s]["args"].get("iters")) for s in SEEDS if J[a][s]]
        res = [J[a][s].get("resumed") for s in SEEDS if J[a][s]]
        out["train"][a] = {"train_sec": ci(secs)[:2], "iters": its, "resumed": res, "seeds": len(secs)}

    def table(spec, title):
        print(f"\n### {title}: exact-string accuracy (eval_depths test sets, n per seed 1000 / 200 / 40) and log loss, nats per bit\n")
        print("| arm | seeds | train s | 1x: L32 D32 | 10x: L320 D320 | 100x: L3200 D3200 | nll 10x | nll 100x |")
        print("|---|---|---|---|---|---|---|---|")
        for a, lab in spec:
            row = {}
            cells = []
            for L, D, tag in POINTS:
                m, h, n = ci([ev(J[a][s], L, D) for s in SEEDS])
                nl = ci([ev(J[a][s], L, D, "nll_bit") for s in SEEDS])
                row[tag] = {"acc": m, "acc_ci": h, "n_seeds": n, "nll": nl[0],
                            "per_seed": [ev(J[a][s], L, D) for s in SEEDS]}
                cells.append(fmt(m, h))
            ts = out["train"][a]["train_sec"][0]
            ns = out["train"][a]["seeds"]
            print(f"| {a} ({lab}) | {ns} | {ts:.0f} |" if ts else f"| {a} ({lab}) | {ns} | - |", " | ".join(cells),
                  f"| {fnll(row['10x']['nll'])} | {fnll(row['100x']['nll'])} |")
            out["ablation" if spec is ABL else "reg"][a] = row

    table(ABL, "Ablation and Bansal baseline")
    table(REG, "Regularizer arms")

    # re-evaluation at L3200 (fresh 200 strings per seed), clamp 0 and 20
    print("\n### L3200 re-evaluation: 200 fresh strings per seed, depth 3200; strings pooled over seeds (Wilson 95%)\n")
    print("| arm | acc (seed mean) | pooled k/n [Wilson] | per seed | nll | nll_raw | nll clip 10 | clamp 20: acc | clamp 20: nll | wrong bits per failed string (median) | wrong-bit abs logit (median) | share of wrong bits past abs logit 100 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        rows = {s: rrow(RE[a][s], 3200, 3200) for s in SEEDS}
        rc = {s: rrow(RE[a][s], 3200, 3200, 20.0) for s in SEEDS}
        have = [s for s in SEEDS if rows[s]]
        if not have:
            continue
        k = sum(sum(rows[s]["per_string_ok"]) for s in have); n = sum(rows[s]["n"] for s in have)
        lo, hi = wilson(k, n)
        m, h, ns = ci([rows[s]["acc"] for s in have])
        mc = ci([rc[s]["acc"] for s in SEEDS if rc[s]])
        nlc = ci([rc[s]["nll_bit"] for s in SEEDS if rc[s]])
        g = lambda key: ci([rows[s][key] for s in have if rows[s].get(key) is not None])[0]
        out["reeval"][a] = {"acc": m, "acc_ci": h, "k": k, "n": n, "wilson": [lo, hi],
                            "per_seed": [rows[s]["acc"] if rows[s] else None for s in SEEDS],
                            "nll": g("nll_bit"), "nll_raw": g("nll_raw"), "nll_clip10": g("nll_clip10"),
                            "clamp20_acc": mc[0], "clamp20_nll": nlc[0],
                            "wrong_bits_per_fail_median": g("wrong_bits_per_fail_median"),
                            "wrong_abs_logit_median": g("wrong_abs_logit_median"),
                            "wrong_share_gt100": g("wrong_share_abs_logit_gt100"),
                            "right_abs_logit_median": g("right_abs_logit_median"),
                            "depths": {D: ci([rrow(RE[a][s], 3200, D)["acc"] for s in have if rrow(RE[a][s], 3200, D)])[0]
                                       for D in (320, 1600, 3200)}}
        o = out["reeval"][a]
        ps = ", ".join("-" if x is None else f"{x:.2f}" for x in o["per_seed"])
        print(f"| {a} | {fmt(m, h, ns)} | {k}/{n} [{lo:.2f}, {hi:.2f}] | {ps} | {fnll(o['nll'])} | {fnll(o['nll_raw'])} | "
              f"{fnll(o['nll_clip10'])} | {fmt(mc[0], mc[1], mc[2]) if mc[2] else '-'} | {fnll(nlc[0])} | "
              f"{fmt(o['wrong_bits_per_fail_median'], None, d=0)} | {fnll(o['wrong_abs_logit_median'])} | "
              f"{fmt(o['wrong_share_gt100'], None)} |")

    # extreme depths
    print("\n### Extreme depths (no clamp): L32 at D 24 / 240 / 2400 / 24000 (n=200 per seed), L320 at D 320 / 3200 / 32000 (n=50 per seed)\n")
    cols = [(32, 24), (32, 240), (32, 2400), (32, 24000), (320, 320), (320, 3200), (320, 32000)]
    print("| arm | " + " | ".join(f"L{L} D{D}" for L, D in cols) + " | nll L32 D24000 | nll L320 D32000 |")
    print("|---|" + "---|" * (len(cols) + 2))
    for a in arms:
        if not any(RE[a][s] for s in SEEDS):
            continue
        cells, row = [], {}
        for L, D in cols:
            rs = [rrow(RE[a][s], L, D) for s in SEEDS]
            rs = [r for r in rs if r]
            m, h, n = ci([r["acc"] for r in rs])
            row[f"L{L}_D{D}"] = {"acc": m, "acc_ci": h, "n_seeds": n, "nll": ci([r["nll_bit"] for r in rs])[0],
                                 "k": sum(sum(r["per_string_ok"]) for r in rs), "n": sum(r["n"] for r in rs)}
            cells.append(fmt(m, h, n) if n else "-")
        out["deep"][a] = row
        print(f"| {a} | " + " | ".join(cells) + f" | {fnll(row['L32_D24000']['nll'])} | {fnll(row['L320_D32000']['nll'])} |")

    # early information: accuracy against steps at small depth
    print("\n### Early steps: exact-string accuracy against depth (seed mean)\n")
    for L, Ds in [(32, [2, 4, 6, 8, 12, 16, 24, 32, 64]), (320, [6, 8, 12, 16, 24, 32, 64, 160, 320, 640]),
                  (3200, [8, 12, 16, 24, 32, 64, 320, 1600, 3200])]:
        print(f"\nL{L}: | arm | " + " | ".join(f"D{D}" for D in Ds) + " |")
        print("|---|" + "---|" * len(Ds))
        for a in [x for x, _ in REG] + ["bptt_cov_dscov", "econ_cov", "econ_cov_voc_rr"]:
            vals = [ci([ev(J[a][s], L, D) for s in SEEDS])[0] for D in Ds]
            out["early"].setdefault(a, {})[L] = dict(zip(Ds, vals))
            print(f"| {a} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in vals) + " |")

    # decision rule of the audit (section 5): the detached random-order BPTT arms against the economy, L3200 fresh set
    print("\n### Audit decision rule: L3200, 200 fresh strings per seed, depth 3200, pooled over seeds\n")
    ref = out["reeval"].get("econ_cov_voc")
    for a in ["bptt_cov_dscov_tb1_rand", "bptt_cov_dscov_tb1_voc_rand", "bptt_cov_dscov_tb1_voc_rep_rand",
              "bptt_cov_dscov_rand", "bptt_cov_dscov_voc_rand", "econ_cov", "econ_cov_voc_rr"]:
        o = out["reeval"].get(a)
        if o and ref:
            print(f"- {a}: {o['k']}/{o['n']} = {o['k'] / o['n']:.3f} [{o['wilson'][0]:.2f}, {o['wilson'][1]:.2f}] "
                  f"against econ_cov_voc {ref['k']}/{ref['n']} = {ref['k'] / ref['n']:.3f} [{ref['wilson'][0]:.2f}, {ref['wilson'][1]:.2f}]")

    # the logit cap with and without the market (phase 3c controls)
    ref = out["reeval"].get("econ_cov_voc_bcap")
    for a in ["bptt_cov_dscov_tb1_voc_bcap_rand", "bptt_cov_dscov_bcap"]:
        o = out["reeval"].get(a)
        if o and ref:
            print(f"- {a}: {o['k']}/{o['n']} = {o['k'] / o['n']:.3f} [{o['wilson'][0]:.2f}, {o['wilson'][1]:.2f}] "
                  f"against econ_cov_voc_bcap {ref['k']}/{ref['n']} = {ref['k'] / ref['n']:.3f} [{ref['wilson'][0]:.2f}, {ref['wilson'][1]:.2f}]")

    # memory at fixed L = 32 (memdepth4.py, seed-0 VM, alone on the GPU)
    p = find("memdepth4.json")
    if p:
        md = json.load(open(p)); out["memdepth4"] = md
        Ds = sorted({r["D"] for r in md})
        print("\n### Memory and time of one training step at fixed L = 32, batch 16, one L4 (memdepth4.py): peak GB, seconds\n")
        print("| mode | " + " | ".join(f"D={D}" for D in Ds) + " |")
        print("|---|" + "---|" * len(Ds))
        for mode in dict.fromkeys(r["mode"] for r in md):
            cells = []
            for D in Ds:
                r = next((r for r in md if r["mode"] == mode and r["D"] == D), None)
                cells.append("-" if r is None else (f"{r['peak_gb']:.3f} GB, {r['sec']:.1f} s" if r["ok"] else f"out of memory ({r['peak_gb']:.1f} GB)"))
            print(f"| {mode} | " + " | ".join(cells) + " |")

    # halting at L64 only and on the mixed set, against an entropy threshold (halt64.py, 3 seeds of econ_cov_voc)
    H = {s: (json.load(open(find(f"halt64_s{s}.json"))) if find(f"halt64_s{s}.json") else None) for s in SEEDS}
    if any(H.values()):
        out["halt64"] = {}
        for st in ("L64", "mix"):
            print(f"\n### Halting, {'every string of length 64' if st == 'L64' else 'mixed lengths 4 to 64 (section 10 set)'}: exact-string accuracy at matched mean steps (n = 500 per seed, step cap 96)\n")
            print("| cost | steps (VOC) | VOC halting | entropy threshold | fixed depth | random (same histogram) | cap 96 |")
            print("|---|---|---|---|---|---|---|")
            hs = [H[s]["sets"][st]["rows"] for s in SEEDS if H[s]]
            for i, c in enumerate([r["cost"] for r in hs[0]]):
                g = {k: ci([h[i][k] for h in hs]) for k in ("steps", "voc_acc", "entropy_acc", "fixed_acc", "random_acc", "cap_acc")}
                out["halt64"].setdefault(st, []).append({"cost": c, **{k: v[:2] for k, v in g.items()}, "n_seeds": len(hs)})
                print(f"| {c:g} | {fmt(*g['steps'][:2], d=1)} | " + " | ".join(fmt(*g[k][:2], d=3) for k in ("voc_acc", "entropy_acc", "fixed_acc", "random_acc", "cap_acc")) + " |")

    fl = json.load(open(find("flops.json"))) if find("flops.json") else {}
    out["flops"] = fl
    json.dump(out, open(os.path.join(R, "p3_summary.json"), "w"), indent=1, default=str)

    # charts
    os.makedirs(os.path.join(D0, "charts"), exist_ok=True)
    for L, arms_c, fn, title in [
            (3200, [a for a, _ in ABL], "p3_depth_L3200_ablation.json", "L3200 (100x training length): exact-string accuracy against depth, ablation and Bansal baseline"),
            (3200, [a for a, _ in REG], "p3_depth_L3200_reg.json", "L3200: exact-string accuracy against depth, regularizer arms"),
            (32, [a for a, _ in REG], "p3_depth_L32_reg.json", "L32 (training range): exact-string accuracy against depth, regularizer arms (early dilution)")]:
        Ds = sorted({e["D"] for a in arms_c for s in SEEDS if J[a][s] for e in J[a][s].get("eval", []) if e["L"] == L})
        ser = []
        for a in arms_c:
            pts = [[D, ci([ev(J[a][s], L, D) for s in SEEDS])[0]] for D in Ds]
            ser.append({"name": a, "points": [p for p in pts if p[1] is not None]})
        json.dump({"title": title, "x_label": "depth (steps), log scale", "y_label": "exact-string accuracy (mean over seeds)",
                   "series": ser}, open(os.path.join(D0, "charts", fn), "w"), indent=1)


if __name__ == "__main__":
    main()
