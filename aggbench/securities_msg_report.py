"""Render the securities-as-message tables (aggbench/README_securities_msg.md) from the result JSON. Nothing is
hand-copied: every number comes from the JSON files given on the command line (training runs and closed forms) and
the uncompressed anchors from compression_results/parity_learn*.json:
  lr  (leak-reduced primary): b5_lf_transformer / b5_lf_isab (8000 steps) when present, else lf_anchor / lf_isab
      (16000 steps, parity_learn_gate_lf.json; twice the arms' steps, so the gap is an upper bound)
  rr3 (readers random, 3 per bit): step4_rr_anchor / step4_rr_isab (parity_learn.json, 8000 steps)
Records without a spec_name are rr3 (the first sweep).

    python3 -m aggbench.securities_msg_report aggbench/compression_results/securities_msg*.json
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["oraclephi_market", "oraclephi_settle_market", "settle_market", "settle_mlp", "settledict_market",
         "settledict_mlp", "settlefree_market", "yonly_market", "yonly_mlp", "ysec_market", "ysec_mlp"]
SPEC_TITLE = {"lr": "leak-reduced primary Z9-p2 (readers_random, Dirichlet(1) bit weights, 6 readers per bit; "
                    "13 agents)",
              "rr3": "Z9-p2 readers random, 3 per bit (7 agents)"}
SETTLE_TAG = "[structure-informed: settlement observes all 6 latent bits]"
MARKET_TAG = "[decoder given Y(omega): the market readout hard-codes Y = XOR of bits 0-3]"


def tags(r):
    """Gate step 7: settlement regresses phi(omega) on the whole 6-bit world state, and the market readout is given
    the target function; both are side information the uncompressed anchor does not get."""
    t = []
    if r.get("settle", 0) > 0 and not r["arm"].startswith("ysec"):
        t.append(SETTLE_TAG)
    if r.get("readout") == "market":
        t.append(MARKET_TAG)
    return (" " + " ".join(t)) if t else ""


ANCHORS = {"lr": [("b5_lf_transformer", "b5_lf_isab"), ("lf_anchor", "lf_isab")],
           "rr3": [("step4_rr_anchor", "step4_rr_isab")]}


def load(paths):
    res, closed, commits = [], {}, defaultdict(set)
    for p in paths:
        d = json.load(open(p))
        c = d.get("code_commit") or ((d.get("harness") or {}).get("git_commit", "?")[:7])
        if d.get("closed_forms"):
            cf = d["closed_forms"]
            closed[cf.get("spec_name", "rr3")] = cf
        for r in d["results"]:
            r.setdefault("spec_name", "rr3")
            if r.get("pretrain") or r.get("power_norm"):      # opus-mats 2026-10-22: masked pretraining / fixed power
                r["arm"] = r["arm"] + ("+pretrain" if r.get("pretrain") else "") + ("+power" if r.get("power_norm") else "")
            commits[r["spec_name"]].add(r.get("code_commit") or c)
            res.append(r)
    seen = {}
    for r in res:                                                    # one record per (spec, arm, b, seed, steps)
        seen[(r["spec_name"], r["arm"], r["b"], r["seed"], r["steps"])] = r
    return list(seen.values()), closed, commits


def anchors(spec_name):
    recs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HERE, "compression_results", "parity_learn*.json"))):
        for r in json.load(open(p))["results"]:
            recs[r["name"]].append((r["seed"], r["regret"]["mean"], sum(n for _, n in r["stages"]),
                                    os.path.basename(p)))
    for tr, isab in ANCHORS[spec_name]:
        if recs.get(tr) and recs.get(isab):
            out = {}
            for kind, name in (("transformer", tr), ("ISAB", isab)):
                rows = sorted({s: (s, v, n, f) for s, v, n, f in recs[name]}.values())
                out[kind] = {"name": name, "mean": float(np.mean([v for _, v, _, _ in rows])),
                             "seeds": [v for _, v, _, _ in rows], "steps": rows[0][2], "file": rows[0][3]}
            return out
    return None


def info_on_structures(n, overrides):
    """I(Y; X) in nats on gsigma structures 0..n-1 (R of the empty security set): R's scale on the same structures."""
    from .analysis import gsigma as gs
    from .securities_msg import span_regret
    return float(np.mean([span_regret(gs.z9p2_cell(s, **overrides), np.zeros((64, 0))) for s in range(n)]))


def section(spec_name, res, closed, commits):
    from .securities_msg import SPECS
    L = [f"## {SPEC_TITLE.get(spec_name, spec_name)}", ""]
    anc = anchors(spec_name)
    best_unc = None
    if anc:
        best_unc = min(a["mean"] for a in anc.values())
        L.append("Uncompressed anchors (same task, fresh episodes, best-val): " + "; ".join(
            f"{k} {a['mean']:.4f} (seeds {', '.join(f'{x:.4f}' for x in a['seeds'])}; {a['steps']} steps; "
            f"`{a['name']}` in {a['file']})" for k, a in anc.items()) +
            f". gap = arm minus the best uncompressed reader ({best_unc:.4f}).")
        L.append("")
    L.append(f"Code commit(s) of the runs: {', '.join(sorted(commits)) or 'not recorded'}.")
    L.append("")
    if closed:
        L.append(f"Prior KL: {closed['prior_regret']:.3f}.")
        L.append("")
        L.append("| closed form | b | possible trade vectors (fixed-length bits) | symbol entropy (bits) | dev KL [95% CI] |")
        L.append("|---|---|---|---|---|")
        for r in closed["closed"]:
            g = r["regret"]
            L.append(f"| oracle bit securities, accuracy in {r['acc_levels']} level(s) {MARKET_TAG} | {r['b']} | "
                     f"{r['possible_trade_vectors']} ({r['fixed_length_bits']}) | {r['symbol_entropy_bits']:.2f} | "
                     f"{g['mean']:.4f} [{g['ci_low']:.4f}, {g['ci_high']:.4f}] |")
        L.append("")
        L.append("| fixed securities | R (nats) | parity-bit energy outside span | log-lik energy outside span |")
        L.append("|---|---|---|---|")
        for name, s in closed["securities"].items():
            L.append(f"| {name} | {s['R_mean_nats']:.4f} | {s['energy_outside_parity_bits']:.2f} | "
                     f"{s['energy_outside_loglik_mean']:.2f} |")
        L.append("")
    # summary per b (8000-step runs): the arm of each family with the lowest mean VALIDATION KL (best_val_kl), its
    # dev KL mean and seed range; arms are never chosen on the dev set they are scored on (gate step 7)
    fam = {f"oracle securities + settlement {SETTLE_TAG} {MARKET_TAG}": ["oraclephi_settle_market"],
           f"learned securities + settlement {SETTLE_TAG}": ["settle_market", "settle_mlp", "settledict_market",
                                                             "settledict_mlp", "settlefree_market"],
           "learned securities, Y only": ["yonly_market", "yonly_mlp"],
           f"oracle securities, Y only {MARKET_TAG}": ["oraclephi_market"], "Y security": ["ysec_market", "ysec_mlp"],
           f"oracle securities + settlement + masked pretraining + fixed power {SETTLE_TAG} {MARKET_TAG}":
               ["oraclephi_settle_market+pretrain+power"],
           f"learned securities + settlement + masked pretraining + fixed power {SETTLE_TAG}":
               ["settle_market+pretrain+power"],
           "learned securities, Y only + masked pretraining + fixed power": ["yonly_market+pretrain+power"]}
    mean, vmean = defaultdict(list), defaultdict(list)
    for r in res:
        if r["steps"] == 8000:
            mean[(r["arm"], r["b"])].append(r["regret"]["mean"])
            vmean[(r["arm"], r["b"])].append(r["best_val_kl"])
    bs = sorted({b for _, b in mean})
    cf = {r["b"]: r["regret"]["mean"] for r in closed["closed"]} if closed else {}
    L.append("Summary (8000 steps; per family the arm with the lowest mean VALIDATION KL, never chosen on dev; "
             "cell = its dev KL mean [seed min-max] (arm)):")
    L.append("")
    L.append("| family | " + " | ".join(f"b = {b}" for b in bs) + " |")
    L.append("|---|" + "---|" * len(bs))
    for name, arms in fam.items():
        cells = []
        for b in bs:
            c = [(np.mean(vmean[(a, b)]), a) for a in arms if mean.get((a, b))]
            if c:
                a = min(c)[1]
                v = mean[(a, b)]
                mk = " [decoder given Y(omega)]" if a.split("+")[0].endswith("_market") else ""
                cells.append(f"{np.mean(v):.3f} [{min(v):.3f}-{max(v):.3f}] ({a}){mk}")
            else:
                cells.append("n/a")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append(f"| oracle bit securities, closed form {MARKET_TAG} | " + " | ".join(f"{cf[b]:.4f}" if b in cf else "n/a" for b in bs)
             + " |")
    if best_unc is not None:
        L.append("| best uncompressed reader (no channel; Y labels only) | " + " | ".join(f"{best_unc:.4f}" for _ in bs) + " |")
    L.append("")
    L.append("| arm | b (id + amount bits) | steps | supervision | dev KL mean (per seed) | escapes | gap to best "
             "uncompressed | best-val step | symbol entropy, bits (id) | distinct symbols | R(phi) (random phi, same S) "
             "| R / I(Y;X) | parity-bit energy outside span(phi) | largest angle to parity bits (deg) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in res:
        groups[(r["arm"], r["b"], r["steps"])].append(r)
    I = info_on_structures(len(res[0]["securities"]["R_nats"]), SPECS[spec_name][1])
    keys = sorted(groups, key=lambda k: (k[2], ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))
    for arm, b, steps in keys:
        rs = sorted(groups[(arm, b, steps)], key=lambda r: r["seed"])
        m = [r["regret"]["mean"] for r in rs]
        sec = [r["securities"] for r in rs]
        Rm = np.mean([s["R_mean_nats"] for s in sec])
        L.append(
            f"| {arm} | {b} ({rs[0]['id_bits']} + {rs[0]['amount_bits']}) | {steps} | {rs[0]['supervision']}{tags(rs[0])} | "
            f"{np.mean(m):.3f} ({', '.join(f'{x:.3f}' for x in m)}) | "
            f"{sum(x < 0.9 * r['prior_regret'] for x, r in zip(m, rs))}/{len(rs)} | "
            f"{'n/a' if best_unc is None else f'{np.mean(m) - best_unc:+.3f}'} | "
            f"{', '.join(str(r['best_step']) for r in rs)} | "
            f"{np.mean([r['symbol_entropy_bits'] for r in rs]):.2f} ({np.mean([r['id_entropy_bits'] for r in rs]):.2f}) | "
            f"{', '.join(str(r['distinct_symbols']) for r in rs)} | "
            f"{Rm:.3f} ({np.mean([s.get('R_random_phi_mean_nats', np.nan) for s in sec]):.3f}) | {Rm / I:.3f} | "
            f"{np.mean([s['energy_outside_parity_bits'] for s in sec]):.2f} | "
            f"{np.mean([s['angles_parity_bits'][0] for s in sec]):.0f} |")
    L.append("")
    L.append(f"Prior KL: {res[0]['prior_regret']:.3f}. I(Y;X) on the {len(res[0]['securities']['R_nats'])} "
             f"structures used for R: {I:.3f}.")
    # span check (gate step 7): which arms have span(phi) within 2 degrees of the parity bits and R <= 0.01 at every
    # b >= 4; on lr, R is a Monte Carlo estimate over 6 fixed structures with unquantised all-security trades
    ok, bad = set(), []
    for (arm, b, steps), rs_ in groups.items():
        if b < 4 or steps != 8000 or not arm.startswith(("settle", "yonly")):
            continue
        ang = np.mean([r["securities"]["angles_parity_bits"][0] for r in rs_])
        Rv = np.mean([r["securities"]["R_mean_nats"] for r in rs_])
        if ang <= 2 and round(Rv, 2) <= 0.01:
            ok.add(arm)
        else:
            bad.append(f"{arm} b={b} ({ang:.0f} deg, R {Rv:.3f})")
    ok -= {a for a in ok if any(x.startswith(a + " ") for x in bad)}
    L.append("Span check (learned securities, 8000 steps, b >= 4): largest angle to the parity bits <= 2 deg and R <= "
             f"0.01 at every b holds for: {', '.join(sorted(ok)) or 'none'}; it fails for: {'; '.join(sorted(bad)) or 'none'}."
             " R on this spec is a Monte Carlo estimate (6 structures, unquantised trades).")
    L.append("")
    L.append("")
    return L


def render(paths):
    res, closed, commits = load(paths)
    L = ["# Securities as the message on Z9-p2: tables", "",
         f"Generated by `python3 -m aggbench.securities_msg_report "
         f"{' '.join(os.path.relpath(p, os.path.dirname(HERE)) for p in paths)}` from the JSON; do not edit by hand. "
         "Exploratory. Code: `aggbench/securities_msg.py` (its docstring defines every arm). KL to Bayes in nats on "
         "4000 fresh dev episodes, best-val checkpoint on a separate validation split, fresh training episodes "
         "(512 per step).", "",
         "Columns: escapes = seeds with dev KL below 0.9 x prior. R(phi) = exact additive-pool regret of the "
         "learned securities, gsigma's R: every agent trades its exact posterior (unquantised) on span(1, phi) and "
         "the trades are summed; nats, mean over 6 gsigma structures (exact enumeration for 7 agents, Monte Carlo "
         "with 20000 draws for 13). In brackets: R of a random phi with the same number of securities (the "
         "degeneracy control). R / I(Y;X) = R as a share of the Y-information on the same structures. "
         "Energy outside = share of the parity-bit span outside span(phi) (0: every parity bit is a combination of "
         "listed securities); the angle is the largest principal angle between the two spans. R can exceed I(Y;X): "
         "summed projections onto a span that misses part of the evidence can do worse than the prior. Reading: R "
         "near 0 with dev KL well above it means the securities are adequate and the loss is in the one quantised "
         "trade per agent and its readout.", ""]
    for spec_name in ("lr", "rr3"):
        rs = [r for r in res if r["spec_name"] == spec_name]
        if rs:
            L += section(spec_name, rs, closed.get(spec_name), commits[spec_name])
    return "\n".join(L) + "\n"


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or sorted(
        glob.glob(os.path.join(HERE, "compression_results", "securities_msg*.json")))
    out = os.path.join(HERE, "README_securities_msg.md")
    with open(out, "w") as f:
        f.write(render(paths))
    print(out)


if __name__ == "__main__":
    main()
