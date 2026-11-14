#!/usr/bin/env python3
"""Stage 0 report: Gate 0 verdict per model, measured decoding speed, concurrency slowdown, and the re-costed plan.

Reads runs/gate0-q06/gate0.json, runs/gate0-q17/gate0.json, runs/gate0-*/timing_train/steps.jsonl and
runs/stage0-conc-{a,b}/steps.jsonl (synced from the Stage 0 VM). Writes stage0_summary.json next to this file.
Gate 0 (PROPOSAL.md section 6): >= 80% of chains give all claims, and the mean informativeness per claim is above 0
(lower end of the 95% CI above 0), pooled over the gate cells (L = 64 and 256).
Re-costing uses measured seconds per chain (gate cells, at the batch sizes the full runs use) and measured seconds
per training step; the assumed decode step was 0.035 s (0.6B) / 0.045 s (1.7B) in cost_estimate.json.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs")
RATE = 0.87
ASSUMED_TAU = {"Qwen/Qwen3-0.6B": 0.035, "Qwen/Qwen3-1.7B": 0.045}


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def pooled_info_ci(name, n_boot=4000):
    """Per-claim informativeness E_bayes[log q(y)/p0(y)] over all gate-cell claims (both L), bootstrap 95% CI."""
    import sys
    sys.path.insert(0, HERE)
    from pool import Board
    from refocus import make_board
    from env.shiphunt import PRESETS
    recs = [r for r in rows(os.path.join(R, name, "gate0_chains.jsonl")) if r.get("cell") == "gate0" and r.get("c")]
    if not recs:
        return None
    B = Board(PRESETS["medium"])
    bay = {}
    vals = []
    for r in recs:
        if r["seed"] not in bay:
            bay[r["seed"]] = B.bayes(make_board(B.cfg, r["seed"])) @ B.F
        b = bay[r["seed"]]
        for j, (q, _) in zip(r["t"], r["c"]):
            p0 = B.p0_marg[j]
            vals.append(b[j] * np.log(q / p0) + (1 - b[j]) * np.log((1 - q) / (1 - p0)))
    v = np.array(vals)
    rng = np.random.default_rng(7)
    m = v[rng.integers(len(v), size=(n_boot, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))]


def gate(name):
    p = os.path.join(R, name, "gate0.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    cells = g["cells"]
    ch = sum(c["chains"] for c in cells.values())
    parsed = sum(c["parse_rate"] * c["chains"] for c in cells.values())
    n = sum(c["n_claims"] for c in cells.values())
    info = sum(c["info_per_claim"] * c["n_claims"] for c in cells.values() if c["n_claims"]) / max(n, 1)
    lows = [c["info_ci"][0] for c in cells.values() if c["n_claims"]]
    pooled_ci = pooled_info_ci(name)
    passed = (parsed / max(ch, 1) >= 0.8) and pooled_ci is not None and pooled_ci[1] > 0
    tau = {L: c["s_per_decode_step"] for L, c in cells.items()}
    per_chain = {L: c["gen_s"] / c["chains"] for L, c in cells.items()}
    steps = rows(os.path.join(R, name, "timing_train", "steps.jsonl"))
    return {"model": g["model"], "prompt_dev": g.get("prompt_dev"), "prompt_chosen": g.get("prompt_chosen"),
            "parse_rate_pooled": parsed / max(ch, 1), "info_per_claim_pooled": info, "info_ci_lower_per_cell": lows,
            "info_pooled_mean_ci_n": pooled_ci,
            "gate_pass": bool(passed), "s_per_decode_step": tau, "s_per_chain": per_chain,
            "brier_vs_bayes": {L: c["brier_vs_bayes"] for L, c in cells.items()},
            "train_steps": [{k: s.get(k) for k in ("step", "L", "t_step", "t_gen", "t_loss", "decode_steps", "gen_s")} for s in steps],
            "cuda_max_mem_gb": g.get("cuda_max_mem_gb")}


def plan(gt, arms_trained=("S1", "S2", "S3"), s0=True, seeds=3, boards=100, steps=150, grid="full", long_arms=("S0", "S1"),
         rr=True, setup_h=0.25, step_s=None):
    """L4-hours from measured per-chain seconds (gate cells, batch ~115-160) and measured training step seconds.
    grid 'full': L=256: 64 round-1 chains + rounds 2-3 for n in {1,4,16}; L=64: 64 + rounds 2-3 for n in {1,4,16,64}.
    grid 'cutB': L=256: 64 round-1 chains + rounds 2-3 for n in {4,16}; L=64: 64 round-1 chains only (R = 1)."""
    pc = {int(L): v for L, v in gt["s_per_chain"].items()}
    st = [s_["t_step"] for s_ in gt["train_steps"] if s_.get("t_step")]
    t_step = step_s or (float(np.mean(st)) if st else float("nan"))
    if grid == "full":
        per_board = {256: 64 + 2 * 21, 64: 64 + 2 * 85}
    else:
        per_board = {256: 64 + 2 * 20, 64: 64}
    eval_h = sum(boards * k * pc[L] for L, k in per_board.items()) / 3600
    long_h = boards * (pc[256] / 352) * sum(b + 192 for b in (256, 1024, 4096)) * 1.5 / 3600 * (128 / 40)
    rr_h = boards * 16 * 3 * pc[256] / 3600 + (boards * 16 * 3 * pc[64] / 3600 if grid == "full" else 0)
    small_h = 4 * 100 * 16 * (pc[64] + pc[256]) / 2 / 3600
    train_h = steps * t_step / 3600 + small_h
    n_ckpt = len(arms_trained) * seeds + (seeds if s0 else 0)
    total = len(arms_trained) * seeds * train_h + n_ckpt * eval_h + len(long_arms) * seeds * long_h + (seeds * rr_h if rr else 0)
    vms = int(np.ceil(total / 3.3)) + 1
    total += vms * setup_h
    return {"t_step_s": t_step, "train_h_per_arm_seed": train_h, "eval_h_per_checkpoint": eval_h, "long_h_per_checkpoint": long_h,
            "roundrobin_h_per_seed": rr_h, "vm_count_est": vms, "total_L4h": total, "total_usd": total * RATE,
            "settings": {"arms_trained": list(arms_trained), "s0": s0, "seeds": seeds, "boards": boards, "steps": steps,
                         "grid": grid, "long_arms": list(long_arms), "roundrobin": rr}}


def gate_detail(name):
    """Per cell (dev-A, dev-B, gate0 L=64, gate0 L=256): parse rate, informativeness, rank correlation of claims with
    exact Bayes, Brier vs Bayes of each pool (LL-fit / D-fit with stakes cross-fitted over boards, 2 folds), Brier on
    statements no chain was asked about (stale prices), and per-chain credit: settlement at the D-fit prices against
    oracle informativeness (within-board Spearman)."""
    import sys
    sys.path.insert(0, HERE)
    from pool import Board, Claims, fit_stakes, price
    from refocus import make_board, open_mask
    from settle import settle as settle_fn, spearman
    from env.shiphunt import PRESETS
    recs = rows(os.path.join(R, name, "gate0_chains.jsonl"))
    if not recs:
        return {}
    B = Board(PRESETS["medium"])
    cache = {}

    def truth(seed):
        if seed not in cache:
            env = make_board(B.cfg, seed)
            z = B.truth(env)
            cache[seed] = (B.F[z].copy(), B.bayes(env) @ B.F, open_mask(B, env), z)
        return cache[seed]
    out = {}
    for cell, L in sorted({(r["cell"], r["L"]) for r in recs}):
        by = {}
        for r in recs:
            if r["cell"] == cell and r["L"] == L:
                by.setdefault(r["seed"], []).append(r)
        eps, qs, bs, info = [], [], [], []
        for seed, lst in sorted(by.items()):
            y, bay, om, z = truth(seed)
            ch, st, q, w = [], [], [], []
            for r in lst:
                if r["c"]:
                    for j, (p, c) in zip(r["t"], r["c"]):
                        ch.append(r["cid"]); st.append(j); q.append(p); w.append(c)
                        p0 = B.p0_marg[j]
                        qs.append(p); bs.append(bay[j]); info.append(bay[j] * np.log(p / p0) + (1 - bay[j]) * np.log((1 - p) / (1 - p0)))
            eps.append({"claims": Claims(np.array(ch, int), np.array(st, int), np.array(q), np.array(w)), "y": y, "zstar": z,
                        "open": om, "bayes": bay, "recs": lst})
        fits = {}
        for fold in (0, 1):
            tr = [e for k, e in enumerate(eps) if k % 2 != fold]
            fits[fold] = fit_stakes(B, tr, "stmt") if sum(len(e["claims"].q) for e in tr) else {"kappa": 1.0, "alpha": 1.0}
        se = {m: [] for m in ("prior", "mean", "vote", "D-avg", "LL-fit", "D-fit")}
        stale = {m: [] for m in ("prior", "mean", "D-fit")}
        rhos, pays, infos = [], [], []
        for k, e in enumerate(eps):
            asked = np.zeros(B.J, bool)
            for r in e["recs"]:
                asked[r["t"]] = True
            o = e["open"]
            stm = o & ~asked
            prices = {}
            for m in se:
                par = fits[k % 2] if m in ("LL-fit", "D-fit") else None
                prices[m], _ = price(B, e["claims"], m, par)
                se[m].append(float(np.mean((prices[m][o] - e["bayes"][o]) ** 2)))
            if stm.any():
                for m in stale:
                    stale[m].append(float(np.mean((prices[m][stm] - e["bayes"][stm]) ** 2)))
            pay, inf = [], []
            for r in e["recs"]:
                if not r["c"]:
                    continue
                cl = Claims(np.zeros(len(r["t"]), int), np.array(r["t"]), np.array([c[0] for c in r["c"]]), np.array([c[1] for c in r["c"]]))
                pay.append(float(settle_fn(B, cl, np.ones(len(cl.q)), prices["D-fit"], e["y"], n_chains=1)["s"][0]))
                bj, p0 = e["bayes"][cl.stmt], B.p0_marg[cl.stmt]
                inf.append(float(np.sum(bj * np.log(cl.q / p0) + (1 - bj) * np.log((1 - cl.q) / (1 - p0)))))
            if len(pay) >= 3:
                rhos.append(spearman(pay, inf))
            pays += pay
            infos += inf
        out[f"{cell}|L{L}"] = {
            "chains": sum(len(v) for v in by.values()), "parse_rate": float(np.mean([r["c"] is not None for v in by.values() for r in v])),
            "claims": len(qs), "info_per_claim": float(np.mean(info)) if info else float("nan"),
            "spearman_claim_vs_bayes": spearman(qs, bs) if len(qs) > 2 else float("nan"),
            "mean_claim": float(np.mean(qs)) if qs else float("nan"), "mean_bayes_on_claimed": float(np.mean(bs)) if bs else float("nan"),
            "brier_vs_bayes": {m: float(np.mean(v)) for m, v in se.items()},
            "stale_brier_vs_bayes": {m: float(np.mean(v)) if v else float("nan") for m, v in stale.items()},
            "credit": {"spearman_pay_vs_info_within_board_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
                       "boards": len(rhos), "share_paid_positive": float(np.mean(np.array(pays) > 0)) if pays else float("nan"),
                       "pooled_spearman": spearman(pays, infos) if len(pays) > 2 else float("nan"),
                       "pay_quantiles_5_50_95": [float(x) for x in np.percentile(pays, [5, 50, 95])] if pays else []},
            "fitted_stakes_LL": {str(k): {"kappa": v["kappa"], "alpha": v["alpha"]} for k, v in fits.items()}}
    return out


def main():
    out = {"gates": {}, "plans": {}, "detail": {}}
    for name in ("gate0-q06", "gate0-q17"):
        d = gate_detail(name)
        if d:
            out["detail"][name] = d
            for cell, v in d.items():
                print(name, cell, "parse %.2f info %.3f rho(q,bayes) %+.3f Brier x1000: %s kappa %s" % (
                    v["parse_rate"], v["info_per_claim"], v["spearman_claim_vs_bayes"],
                    {m: round(1000 * x, 1) for m, x in v["brier_vs_bayes"].items()},
                    [round(f["kappa"], 3) for f in v["fitted_stakes_LL"].values()]))
        g = gate(name)
        if g:
            out["gates"][name] = g
            m = g["model"]
            out["gates"][name]["tau_vs_assumed"] = {L: v / ASSUMED_TAU[m] for L, v in g["s_per_decode_step"].items()}
    a, b = rows(os.path.join(R, "stage0-conc-a", "steps.jsonl")), rows(os.path.join(R, "stage0-conc-b", "steps.jsonl"))
    g06 = out["gates"].get("gate0-q06")
    if a and b and g06:
        single = {s["step"]: s["t_step"] for s in g06["train_steps"] if s.get("t_step")}
        ratios = [((x["t_step"] + y["t_step"]) / 2) / single[x["step"]] for x, y in zip(a, b) if x["step"] in single]
        out["concurrency_slowdown"] = {"per_step_ratio": ratios, "mean": float(np.mean(ratios)) if ratios else None,
                                       "throughput_gain_two_procs": 2 / float(np.mean(ratios)) if ratios else None}
    for name, g in out["gates"].items():
        if not g["s_per_chain"].get("64") or not g["s_per_chain"].get("256"):
            continue
        out["plans"][name + " | PROPOSAL full plan"] = plan(g)
        out["plans"][name + " | pre-registered cuts: no S2, long chains S1 only, 80 boards, 100 steps"] = plan(
            g, arms_trained=("S1", "S3"), long_arms=("S1",), boards=80, steps=100)
        out["plans"][name + " | cut B: S0+S1 only, smaller grid, long chains S1 only, 150 steps"] = plan(
            g, arms_trained=("S1",), long_arms=("S1",), grid="cutB")
        out["plans"][name + " | cut B with 100 steps"] = plan(g, arms_trained=("S1",), long_arms=("S1",), grid="cutB", steps=100)
    json.dump(out, open(os.path.join(HERE, "stage0_summary.json"), "w"), indent=1)
    for name, g in out["gates"].items():
        print(name, "prompt", g["prompt_chosen"], g["prompt_dev"], "parse %.3f info %.4f pooled %s pass %s tau %s"
              % (g["parse_rate_pooled"], g["info_per_claim_pooled"], g["info_pooled_mean_ci_n"],
                 g["gate_pass"], {k: round(v, 4) for k, v in g["s_per_decode_step"].items()}))
    print("concurrency", out.get("concurrency_slowdown"))
    for k, v in out["plans"].items():
        print(k, "| step %.0f s, train/arm-seed %.2f h, eval/ckpt %.2f h, total %.1f L4-h $%.1f" % (v["t_step_s"], v["train_h_per_arm_seed"], v["eval_h_per_checkpoint"], v["total_L4h"], v["total_usd"]))


if __name__ == "__main__":
    main()
