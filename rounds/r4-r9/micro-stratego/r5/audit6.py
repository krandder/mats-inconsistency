#!/usr/bin/env python3
"""R5-E2 audit (separate pass; does not import report6.py). Checks:
  A. PREREG-R5-E2.md matches its recorded sha256, and its hash time precedes the first r5e2 launch (gpu/launches.log)
     and the first r5e2 PENDING-CREATE line of the shared logical-soup/GPU-LOG.md.
  B. No oracle in training or selection: train6.py / model6.py / agents6.py never open an oracle file or name an oracle
     quantity; eval6.py opens oracle files only inside its scoring functions; the capacity rule reads only calibration
     summaries.
  C. Re-derives every headline number from the per-item CSVs with its own code and compares with the JSON summaries.
  D. Recomputes every rule verdict from the re-derived numbers with its own t-intervals, compares with report/verdicts.csv.
  E. The on-VM independent recomputation (audit_vm.json) of the D0 headline numbers.
Writes report/audit_rederived.csv and prints a markdown block for AUDIT-R5-E2.md.
   python3 audit6.py"""
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(H, "runs")
SEEDS = (101, 102, 103)
AGGS = ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3")
GPULOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
PRE = os.path.join(os.path.dirname(H), "PREREG-R5-E2.md")


def rd(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def js(seed, arch, part):
    return json.load(open(os.path.join(RUNS, f"s{seed}", arch, "eval", f"{part}.json")))


def ci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - h, m + h


def main():
    out = []
    say = out.append
    # A
    rec = open(PRE.replace(".md", ".sha256")).read().split()
    h = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    t_hash = rec[2] if len(rec) > 2 else "?"
    launches = [l for l in open(os.path.join(H, "gpu", "launches.log"))] if os.path.exists(os.path.join(H, "gpu", "launches.log")) else []
    first_launch = launches[0].split()[0] if launches else "none"
    pend = [l for l in open(GPULOG) if "vm=soup-gpu-r5e2-" in l and "PENDING-CREATE" in l]
    first_pend = pend[0].split()[1] if pend else "none"
    ok_a = (h == rec[0]) and t_hash < first_launch and t_hash[:16] <= first_pend
    say(f"A. prereg sha256 {h[:16]}... recorded {rec[0][:16]}... match={h == rec[0]}; hashed {t_hash}; first launch "
        f"{first_launch}; first r5e2 PENDING-CREATE {first_pend}; precedes={ok_a}")
    # B
    bad = []
    for f in ("train6.py", "model6.py", "agents6.py"):
        s = open(os.path.join(H, f)).read()
        for pat in (r"oracle_", r"pstar", r"qstar", r"q_star", r"posterior", r"constraint_marginals", r"msgame6"):
            for m in re.finditer(pat, s):
                line = s[:m.start()].count("\n") + 1
                bad.append(f"{f}:{line}:{pat}")
    ev = open(os.path.join(H, "eval6.py")).read()
    loads = [(ev[:m.start()].count("\n") + 1, m.group(0)) for m in re.finditer(r"oracle_[a-z0-9_]+\.npz", ev)]
    funcs = []
    for line, name in loads:
        head = ev.split("\n")[:line]
        fn = [l for l in head if l.startswith("def ")][-1]
        funcs.append(f"{name} in {fn.split('(')[0][4:]}")
    rp = open(os.path.join(H, "report6.py")).read()
    cap = rp[rp.index("# ---------------- capacity rule"):rp.index("# ---------------- D0")]
    sel_ok = "d0cal" in cap and '"d0"' not in cap
    say(f"B. oracle names in training/model/agent code: {bad if bad else 'none'}; eval6 oracle reads: {'; '.join(funcs)}; "
        f"capacity rule reads calibration summaries only: {sel_ok}")
    # C
    rows = []
    def chk(q, s, a, b):
        rows.append((q, s, a, b, abs(a - b)))
    for s in SEEDS:
        for arch in ("primary", "fallback"):
            e = os.path.join(RUNS, f"s{s}", arch, "eval")
            if not os.path.exists(os.path.join(e, "d0.json")):
                continue
            for split, part in (("test", "d0"), ("cal", "d0cal")):
                j = js(s, arch, part)
                chk(f"{part} kl0 {arch}", s, j["kl0_full_per_hidden_slot"], rd(os.path.join(e, f"d0_{split}_positions.csv"))["kl0_full"].mean())
                chk(f"{part} nll7 {arch}", s, j["nll_7known"], rd(os.path.join(e, f"d0_{split}_nll7.csv"))["nll7"].mean())
                chk(f"{part} battle {arch}", s, j["battle_acc_ply0"], rd(os.path.join(e, f"d0_{split}_battles.csv"))["correct"].mean())
            j = js(s, arch, "d0b")
            c = rd(os.path.join(e, "d0b_belief_positions.csv"))
            chk(f"d0b belief red {arch}", s, j["belief_kl_reduction"], 1 - c["kl16"].mean() / c["kl0"].mean())
            c = rd(os.path.join(e, "d0b_action_values.csv"))
            chk(f"d0b action red {arch}", s, j["action_mse_reduction"], 1 - c["se16"].mean() / c["se0"].mean())
            j = js(s, arch, "d1")
            c = rd(os.path.join(e, "d1_positions.csv"))
            for agg in AGGS:
                chk(f"d1 jsd_b {agg} {arch}", s, j[agg]["jsd_belief"], c[f"jsd_b_{agg}"].mean())
                chk(f"d1 jsd_v {agg} {arch}", s, j[agg]["jsd_value"], c[f"jsd_v_{agg}"].mean())
                chk(f"d1 klref {agg} {arch}", s, j[agg]["kl_ref_belief"], c[f"klref_b_{agg}"].mean())
                chk(f"d1 mse {agg} {arch}", s, j[agg]["mse_ref_value"], c[f"mse_v_{agg}"].mean())
            g = np.nanmean(np.stack([c["key_gap_L"], c["key_gap_R"]]), 0)
            chk(f"d1 key gap {arch}", s, j["key_gap"], float(np.nanmean(g)))
            j = js(s, arch, "d2")
            c = rd(os.path.join(e, "d2_rollouts.csv"))
            y = -c["kl"]
            chk(f"d2 spearman trust {arch}", s, j["spearman_trust"], stats.spearmanr(c["beta"], y).correlation)
            chk(f"d2 spearman stiff {arch}", s, j["spearman_stiff"], stats.spearmanr(-c["H"], y).correlation)
            order = np.argsort(c["H"], kind="stable")
            ranks = np.empty(len(order))
            ranks[order] = np.arange(len(order))
            q = np.quantile(c["H"], np.linspace(0, 1, 11))
            st = np.clip(np.searchsorted(q, c["H"], side="right") - 1, 0, 9)
            tt = np.nanmean([stats.spearmanr(c["beta"][st == b], y[st == b]).correlation for b in range(10)])
            ts = np.nanmean([stats.spearmanr(-c["H"][st == b], y[st == b]).correlation for b in range(10)])
            chk(f"d2 strata trust {arch}", s, j["strata_trust"], tt)
            chk(f"d2 strata stiff {arch}", s, j["strata_stiff"], ts)
            j = js(s, arch, "d3a")
            c = rd(os.path.join(e, "d3a_positions.csv"))
            rel = (c["q_miner_max"] - c["q_miner_std"]) >= 0.05
            chk(f"d3a shift rate {arch}", s, j["shift_rate_relevant"], (c["q_miner_miner"][rel] > c["q_miner_std"][rel]).mean())
            j = js(s, arch, "d3b")
            c = rd(os.path.join(e, "d3b_games.csv"))
            chk(f"d3b effective win {arch}", s, j["effective_win"], c["result"].mean())
    with open(os.path.join(H, "report", "audit_rederived.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "seed", "json", "rederived", "abs_diff"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.8g}", f"{r[3]:.8g}", f"{r[4]:.3g}"])
    mx = max((r[4] for r in rows if math.isfinite(r[4])), default=float("nan"))
    say(f"C. {len(rows)} numbers re-derived from per-item CSVs; largest |difference| {mx:.3g} (report/audit_rederived.csv)")
    # D: recompute verdicts for the reported architecture
    S = json.load(open(os.path.join(H, "report", "summary.json")))
    A = S["reported_arch"]
    get = lambda q: np.array([r[3] for r in rows if r[0] == f"{q} {A}"])
    v = {}
    kl = get("d0 kl0")
    v["D0"] = bool(get("d0 battle").mean() >= 0.97 and get("d0 nll7").mean() <= 0.05 and kl.mean() >= 0.05 and
                   (get("d0b belief red").mean() >= 0.10 or get("d0b action red").mean() >= 0.10) and S["selected_arch"] is not None)
    kg = get("d1 key gap")
    d1 = kg.mean() >= 0.25 and ci(kg)[1] > 0
    for comp in ("jsd_b", "jsd_v"):
        jm, ja = get(f"d1 {comp} mix"), get(f"d1 {comp} attn_L3")
        d1 = d1 and (jm / ja).mean() >= 2.5 and ci(jm - ja)[1] > 0
    for base in ("mix", "ataraxos_mmd", "kc_dedup"):
        for comp in ("klref", "mse"):
            d1 = d1 and ci(get(f"d1 {comp} {base}") - get(f"d1 {comp} attn_L3"))[1] > 0
    v["D1"] = bool(d1)
    st_, ss_ = get("d2 spearman trust"), get("d2 spearman stiff")
    tt_, ts_ = get("d2 strata trust"), get("d2 strata stiff")
    v["D2"] = bool(st_.mean() >= 0.25 and ci(st_)[1] > 0 and (st_ - ss_).mean() >= 0.20 and ci(st_ - ss_)[1] > 0 and
                   (tt_ - ts_).mean() >= 0.15 and ci(tt_ - ts_)[1] > 0)
    sr, ew = get("d3a shift rate"), get("d3b effective win")
    d3a_extra = all(js(s, A, "d3a")["param_hash_identical"] and js(s, A, "d3a")["belief_max_abs_change"] == 0 for s in SEEDS)
    v["D3"] = bool(sr.mean() >= 0.75 and ci(sr)[1] > 0.5 and d3a_extra and ew.mean() > 0.52 and ci(ew)[1] > 0.5)
    agree = {k: (v[k] == S["rules"][k]) for k in v}
    say(f"D. verdicts recomputed for {A}: {v}; report6 verdicts: {S['rules']}; agree: {agree}")
    # E
    for s in SEEDS:
        for arch in ("primary", "fallback"):
            p = os.path.join(RUNS, f"s{s}", arch, "eval", "audit_vm.json")
            if os.path.exists(p):
                a = json.load(open(p))
                say(f"E. s{s} {arch}: on-VM independent recomputation |diff| kl {a['abs_diff']['kl']:.2g}, nll7 "
                    f"{a['abs_diff']['nll7']:.2g}, battle {a['abs_diff']['battle']:.2g}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
