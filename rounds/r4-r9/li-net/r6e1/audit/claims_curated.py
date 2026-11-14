#!/usr/bin/env python3
"""R6-E1 audit, check 7: hand-mapped check of the measured numbers in RESULTS-R6-E1.md.

Every entry below was mapped BY HAND (auditor) from a number written in RESULTS-R6-E1.md to the quantity it claims to be; the
script only does the arithmetic: is the written number the correct rounding of the file value at the written precision?
Sources: analysis/results.json (3-seed means, seed-t CIs, per seed; computed from unrounded data on the VM), summary.json, and
the auditor's re-derivation (rederive-output.json) where results.json has no entry. Bars and design constants are checked against
PREREG-R6-E1.md section 3 / section 1. Prints WRONG rows and totals; writes claims-curated.csv.
Run: python3 claims_curated.py
"""
import csv
import json
import os
import re

import numpy as np

R6 = "/home/agent/workspaces/eurisko-study/li-net/r6e1"
OUT = os.path.dirname(os.path.abspath(__file__))
RJ = json.load(open(os.path.join(R6, "results", "main", "analysis", "results.json")))
ST, V, FIT, CMP = RJ["stats"], RJ["verdicts"], RJ["fit"], RJ["compute"]
RED = json.load(open(os.path.join(OUT, "rederive-output.json")))["rules"]["t_exact"]
SUM = {s: json.load(open(os.path.join(R6, "results", "main", f"s{s}", "summary.json"))) for s in (101, 102, 103)}
D21, D22, D3 = V["E1-D2.1"]["stats"], V["E1-D2.2"]["stats"], V["E1-D3"]["stats"]
K = ST["keys"]
LV = (0, 1, 2, 3, 4, 6)
rows = []


def add(line, written, ref, src, kind="measured", scale=1.0):
    """written: the number exactly as printed (string, may end in % or x); ref: the file value."""
    w = written.replace(",", "").replace("$", "").rstrip("%x")
    dec = len(w.split(".")[1]) if "." in w else 0
    val = round(ref * scale, dec)
    ok = abs(val - float(w)) < 1e-9
    dbl = (not ok) and abs(round(round(ref * scale, dec + 1), dec) - float(w)) < 1e-9
    rows.append({"line": line, "written": written, "kind": kind, "file_value": ref * scale, "correct_rounding": f"{val:.{dec}f}",
                 "status": "OK" if ok else ("WRONG (double rounding)" if dbl else "WRONG"), "source": src})


def ci3(line, ws, c, src, scale=1.0):
    for w, k in zip(ws, ("mean", "lo", "hi")):
        if w is not None:
            add(line, w, c[k], f"{src}.{k}", scale=scale)


def bar(line, written, value, src):
    add(line, written, value, src, kind="bar/design")


# ---------------- header and TL;DR (lines 1-57)
bar(4, "$2.50", 2.50, "PREREG header ceiling")
add(12, "0.375", 0.3749, "R5-E1 v2 results.json part_a.key_pearson.mean (quoted by the spec)")
add(24, "0.9994", ST["clause1_corr_std6"]["mean"], "results.stats.clause1_corr_std6.mean")
add(24, "0.9994", ST["clause1_corr_std6"]["lo"], "results.stats.clause1_corr_std6.lo")
add(24, "0.9995", ST["clause1_corr_std6"]["hi"], "results.stats.clause1_corr_std6.hi")
bar(24, "0.99", 0.99, "prereg E1-D0")
ci3(24, ("63.2%", "62.1", "64.2"), ST["fs_cut_scaled_mix"], "results.stats.fs_cut_scaled_mix", 100)
bar(24, "40%", 40, "prereg E1-D0")
ci3(25, ("0.863", "0.858", "0.868"), K["K4"]["pearson"], "results.stats.keys.K4.pearson")
add(25, "0.629", K["K4"]["gap"]["mean"], "results.stats.keys.K4.gap.mean")
add(25, "1.00", K["K4"]["spearman_levels_mean_curve"], "results.stats.keys.K4.spearman")
ci3(25, ("0.880", "0.876", "0.884"), K["K3"]["pearson"], "results.stats.keys.K3.pearson")
add(25, "0.967", K["K3"]["gap"]["mean"], "results.stats.keys.K3.gap.mean")
add(25, "1.00", K["K3"]["spearman_levels_mean_curve"], "results.stats.keys.K3.spearman")
for w, b in (("0.60", 0.60), ("0.50", 0.50), ("0.50", 0.50), ("0.94", 0.94)):
    bar(25, w, b, "prereg E1-D1")
add(26, "5.8x", D21["scaled_mix"]["jsd_ratio"]["mean"], "verdicts.E1-D2.1.scaled_mix.jsd_ratio.mean")
add(26, "6.5x", D21["mmd"]["jsd_ratio"]["mean"], "verdicts.E1-D2.1.mmd.jsd_ratio.mean")
add(26, "2.9x", D21["kc"]["jsd_ratio"]["mean"], "verdicts.E1-D2.1.kc.jsd_ratio.mean")
bar(26, "2.5x", 2.5, "prereg E1-D2.1")
ci3(27, ("0.9494", "0.9471", "0.9517"), D22["coverage_puct"], "verdicts.E1-D2.2.coverage_puct")
bar(27, "0.95", 0.95, "prereg E1-D2.2")
add(27, "0.828", D22["coverage_open"]["mean"], "verdicts.E1-D2.2.coverage_open.mean")
ci3(27, ("2.00x", "1.91", "2.09"), D22["jsd_ratio_open_over_puct"], "verdicts.E1-D2.2.jsd_ratio")
bar(27, "1.8x", 1.8, "prereg E1-D2.2")
add(27, "38%", D22["kl_cut_puct_vs_open"]["mean"], "verdicts.E1-D2.2.kl_cut.mean", scale=100)
bar(27, "20%", 20, "prereg E1-D2.2")
add(27, "0.82", D22["wave1_nw_share"]["mean"], "verdicts.E1-D2.2.wave1_nw_share.mean")
bar(27, "0.75", 0.75, "prereg E1-D2.2")
add(28, "1.08", D3["W16/W1_t0.3"]["mean"], "verdicts.E1-D3.W16/W1_t0.3.mean")
bar(28, "1.25", 1.25, "prereg E1-D3")
ci3(28, ("3.54", "3.28", "3.79"), D3["W8/W1_t2.5"], "verdicts.E1-D3.W8/W1_t2.5")
add(28, "4.52", D3["W16/W1_t2.5"]["mean"], "verdicts.E1-D3.W16/W1_t2.5.mean")
bar(28, "2.20", 2.20, "prereg E1-D3")
bar(28, "2.80", 2.80, "prereg E1-D3")
add(28, "38.5%", D3["msecut_vs_mmd_t0.3"]["mean"], "verdicts.E1-D3.msecut_vs_mmd_t0.3.mean", scale=100)
bar(28, "35%", 35, "prereg E1-D3")
ci3(28, ("10.3%", "7.7", "13.0"), D3["msecut_vs_kc_t2.5"], "verdicts.E1-D3.msecut_vs_kc_t2.5", 100)
bar(28, "15%", 15, "prereg E1-D3")
add(31, "0.88", K["K3"]["pearson"]["mean"], "results.stats.keys.K3.pearson.mean")
add(31, "0.01", K["K3"]["cos_by_overlap_mean"][0], "results.stats.keys.K3.cos_by_overlap_mean[0]")
add(32, "0.96", K["K3"]["cos_by_overlap_mean"][5], "results.stats.keys.K3.cos_by_overlap_mean[6 shared]")
add(32, "-0.012", K["K3"]["across"]["mean"], "results.stats.keys.K3.across.mean ('orthogonal')")
add(33, "0.86", K["K4"]["pearson"]["mean"], "results.stats.keys.K4.pearson.mean")
add(34, "0.58", K["K2"]["gap"]["mean"], "results.stats.keys.K2.gap.mean")
add(35, "0.40", K["K2"]["pearson"]["mean"], "results.stats.keys.K2.pearson.mean")
add(36, "0.38", K["K1"]["pearson"]["mean"], "results.stats.keys.K1.pearson.mean")
add(38, "0.0078", ST["fs_jsd_attn_K4"]["mean"], "results.stats.fs_jsd_attn_K4.mean")
add(39, "0.022", ST["fs_jsd_kc"]["mean"], "results.stats.fs_jsd_kc.mean (range low end)")
add(39, "0.051", ST["fs_jsd_mmd"]["mean"], "results.stats.fs_jsd_mmd.mean (range high end)")
add(40, "0.034", ST["fs_kl_attn_K4"]["mean"], "results.stats.fs_kl_attn_K4.mean")
add(40, "0.074", ST["fs_kl_kc"]["mean"], "results.stats.fs_kl_kc.mean (range low end)")
add(40, "0.120", ST["fs_kl_mmd"]["mean"], "results.stats.fs_kl_mmd.mean (range high end)")
add(40, "0.002", ST["fs_kl_union"]["mean"], "results.stats.fs_kl_union.mean")
add(43, "94.9%", D22["coverage_puct"]["mean"], "verdicts.E1-D2.2.coverage_puct.mean", scale=100)
add(43, "82.8%", D22["coverage_open"]["mean"], "verdicts.E1-D2.2.coverage_open.mean", scale=100)
add(44, "0.0261", ST["ap_open_jsd_attn_K4"]["mean"], "results.stats.ap_open_jsd_attn_K4.mean")
add(44, "0.0131", ST["ap_puct_jsd_attn_K4"]["mean"], "results.stats.ap_puct_jsd_attn_K4.mean")
add(44, "38%", D22["kl_cut_puct_vs_open"]["mean"], "verdicts.E1-D2.2.kl_cut.mean", scale=100)
add(44, "0.080", ST["ap_open_kl_attn_K4"]["mean"], "results.stats.ap_open_kl_attn_K4.mean")
add(44, "0.050", ST["ap_puct_kl_attn_K4"]["mean"], "results.stats.ap_puct_kl_attn_K4.mean")
add(45, "82%", D22["wave1_nw_share"]["mean"], "verdicts.E1-D2.2.wave1_nw_share.mean", scale=100)
add(46, "0.06", 100 * (0.95 - D22["coverage_puct"]["mean"]), "100 x (0.95 - coverage_puct.mean), percentage points")
add(48, "1.08x", D3["W16/W1_t0.3"]["mean"], "verdicts.E1-D3.W16/W1_t0.3.mean")
add(49, "2.00", D3["W8/W1_t1.5"]["mean"], "verdicts.E1-D3.W8/W1_t1.5.mean")
add(49, "3.54", D3["W8/W1_t2.5"]["mean"], "verdicts.E1-D3.W8/W1_t2.5.mean")
for w, t in (("3.33", "0.3"), ("2.16", "1.5"), ("1.50", "2.5")):
    add(51, w, float(np.mean([FIT[s]["beta_by_tau"][t] for s in FIT])), f"3-seed mean of results.fit.<seed>.beta_by_tau.{t}")
add(52, "10%", D3["msecut_vs_kc_t2.5"]["mean"], "verdicts.E1-D3.msecut_vs_kc_t2.5.mean", scale=100)
bar(52, "15%", 15, "prereg E1-D3")
add(54, "2,789", FIT["101"]["params"]["looped"], "results.fit.101.params.looped")
add(54, "39", FIT["101"]["params"]["attn_K4"] / FIT["101"]["params"]["looped"], "attn_K4 / looped params ('1/39')")
add(54, "2.0x", D21["looped_secondary"]["kl_ratio_to_attn"]["mean"], "verdicts.E1-D2.1.looped.kl_ratio.mean")
bar(55, "1.15x", 1.15, "prereg E1-D2.1 secondary")
add(55, "0.885", K["looped"]["pearson"]["mean"], "results.stats.keys.looped.pearson.mean")
add(56, "$1.10", 1.10, "ledger6.py total (GPU-LOG lines 2075, 2088)")
add(56, "$0.28", 0.28, "GPU-LOG line 2075 COST-AUDIT")
add(56, "$0.82", 0.82, "GPU-LOG line 2088 DELETE")
# ---------------- sections 2-3
add(75, "669,633", SUM[101]["theta_real_params"], "summary.json theta_real_params")
add(85, "2,789", FIT["101"]["params"]["looped"], "results.fit.101.params.looped")
for ln, ws in ((98, ("0.9994", "0.9994", "0.9995")),):
    ci3(ln, ws, ST["clause1_corr_std6"], "results.stats.clause1_corr_std6")
for w, x in zip(("0.9994", "0.9995", "0.9994"), ST["clause1_corr_std6"]["per_seed"]):
    add(99, w, x, "results.stats.clause1_corr_std6.per_seed")
add(101, "0.280", ST["kl_p0"]["mean"], "results.stats.kl_p0.mean")
add(102, "0.23", float(np.mean([FIT[s]["alpha"] for s in FIT])), "3-seed mean of results.fit.<seed>.alpha")
add(102, "0.103", ST["fs_kl_scaled_mix"]["mean"], "results.stats.fs_kl_scaled_mix.mean")
ci3(103, ("63.2%", "62.1", "64.2"), ST["fs_cut_scaled_mix"], "results.stats.fs_cut_scaled_mix", 100)
add(104, "0.0022", ST["fs_kl_union"]["mean"], "results.stats.fs_kl_union.mean")
# ---------------- section 4 keys table (lines 113-117) and line 119
table = {113: ("K1", "0.382", "0.950", "0.756", "0.194", ("0.74", "0.72", "0.77", "0.80", "0.87", "0.95"), "0.94"),
         114: ("K2", "0.403", "0.793", "0.213", "0.580", ("0.18", "0.14", "0.21", "0.28", "0.45", "0.79"), "0.94"),
         115: ("K3", "0.880", "0.955", "-0.012", "0.967", ("0.01", "0.27", "0.50", "0.67", "0.78", "0.96"), "1.00"),
         116: ("K4", "0.863", "0.943", "0.314", "0.629", ("0.31", "0.42", "0.54", "0.66", "0.76", "0.94"), "1.00"),
         117: ("looped", "0.885", "0.942", "0.321", "0.621", ("0.31", "0.42", "0.55", "0.67", "0.76", "0.94"), "1.00")}
for ln, (v, p, sm, ac, gp, curve, sp) in table.items():
    add(ln, p, K[v]["pearson"]["mean"], f"results.stats.keys.{v}.pearson.mean")
    add(ln, sm, K[v]["same"]["mean"], f"results.stats.keys.{v}.same.mean")
    add(ln, ac, K[v]["across"]["mean"], f"results.stats.keys.{v}.across.mean")
    add(ln, gp, K[v]["gap"]["mean"], f"results.stats.keys.{v}.gap.mean")
    for w, x, o in zip(curve, K[v]["cos_by_overlap_mean"], LV):
        add(ln, w, x, f"results.stats.keys.{v}.cos_by_overlap_mean[{o} shared]")
    add(ln, sp, K[v]["spearman_levels_mean_curve"], f"results.stats.keys.{v}.spearman_levels_mean_curve")
ci3(115, (None, "0.876", "0.884"), K["K3"]["pearson"], "results.stats.keys.K3.pearson")
ci3(115, (None, "0.936", "0.999"), K["K3"]["gap"], "results.stats.keys.K3.gap")
ci3(116, (None, "0.858", "0.868"), K["K4"]["pearson"], "results.stats.keys.K4.pearson")
ci3(116, (None, "0.627", "0.632"), K["K4"]["gap"], "results.stats.keys.K4.gap")
for w, x in zip(("0.880", "0.881", "0.878"), K["K3"]["pearson"]["per_seed"]):
    add(119, w, x, "results.stats.keys.K3.pearson.per_seed")
for w, x in zip(("0.863", "0.861", "0.865"), K["K4"]["pearson"]["per_seed"]):
    add(119, w, x, "results.stats.keys.K4.pearson.per_seed")
add(108, "276", 24 * 23 / 2, "pairs per bag (24 reads)")
# ---------------- section 5 P-FullSupport table (lines 135-142) and text
fst = {135: ("mix", "0.197", "0.0038"), 136: ("scaled_mix", "0.103", "0.0453"), 137: ("mmd", "0.120", "0.0509"),
       138: ("kc", "0.074", "0.0222"), 139: ("looped", "0.069", "0.0219"), 141: ("attn_K4", "0.034", "0.0078"),
       142: ("union", "0.002", "0.0000")}
for ln, (a, kl, js) in fst.items():
    add(ln, kl, ST[f"fs_kl_{a}"]["mean"], f"results.stats.fs_kl_{a}.mean")
    add(ln, js, ST[f"fs_jsd_{a}"]["mean"], f"results.stats.fs_jsd_{a}.mean")
for ln, b in ((136, "scaled_mix"), (137, "mmd"), (138, "kc")):
    ws = {"scaled_mix": (("5.82", "5.15", "6.49"), ("0.0695", "0.0658", "0.0732")),
          "mmd": (("6.54", "5.81", "7.26"), ("0.0863", "0.0830", "0.0897")),
          "kc": (("2.85", "2.48", "3.21"), ("0.0401", "0.0297", "0.0506"))}[b]
    ci3(ln, ws[0], D21[b]["jsd_ratio"], f"verdicts.E1-D2.1.{b}.jsd_ratio")
    ci3(ln, ws[1], D21[b]["kl_diff"], f"verdicts.E1-D2.1.{b}.kl_diff")
ci3(139, ("2.04", "1.84", "2.24"), D21["looped_secondary"]["kl_ratio_to_attn"], "verdicts.E1-D2.1.looped.kl_ratio")
for w, a in zip(("0.037", "0.038", "0.033"), ("attn_K1", "attn_K2", "attn_K3")):
    add(140, w, ST[f"fs_kl_{a}"]["mean"], f"results.stats.fs_kl_{a}.mean")
for w, a in zip(("0.0091", "0.0096", "0.0069"), ("attn_K1", "attn_K2", "attn_K3")):
    add(140, w, ST[f"fs_jsd_{a}"]["mean"], f"results.stats.fs_jsd_{a}.mean")
for b, lo_w, hi_w in (("scaled_mix", "5.67", "6.13"), ("mmd", "6.25", "6.84"), ("kc", "2.68", "2.96")):
    add(145, lo_w, min(D21[b]["jsd_ratio"]["per_seed"]), f"min per-seed {b} JSD ratio")
    add(145, hi_w, max(D21[b]["jsd_ratio"]["per_seed"]), f"max per-seed {b} JSD ratio")
add(148, "0.0038", ST["fs_jsd_mix"]["mean"], "results.stats.fs_jsd_mix.mean")
add(148, "0.197", ST["fs_kl_mix"]["mean"], "results.stats.fs_kl_mix.mean")
add(150, "2,789", FIT["101"]["params"]["looped"], "params.looped")
add(150, "108,228", FIT["101"]["params"]["attn_K4"], "params.attn_K4")
add(150, "2.04x", D21["looped_secondary"]["kl_ratio_to_attn"]["mean"], "verdicts looped kl_ratio.mean")
add(152, "0.033", min(ST[f"fs_kl_attn_{v}"]["mean"] for v in ("K1", "K2", "K3", "K4")), "min fs_kl over attn K1-K4")
add(152, "0.038", max(ST[f"fs_kl_attn_{v}"]["mean"] for v in ("K1", "K2", "K3", "K4")), "max fs_kl over attn K1-K4")
# ---------------- section 5 P-ActivePUCT table (lines 158-162) and text
ci3(158, ("0.828", "0.819", "0.836"), D22["coverage_open"], "verdicts.E1-D2.2.coverage_open")
ci3(158, ("0.9494", "0.9471", "0.9517"), D22["coverage_puct"], "verdicts.E1-D2.2.coverage_puct")
for w, x in zip(("0.9484", "0.9496", "0.9502"), D22["coverage_puct"]["per_seed"]):
    add(158, w, x, "coverage_puct.per_seed")
    add(165, w, x, "coverage_puct.per_seed")
add(159, "0.0261", ST["ap_open_jsd_attn_K4"]["mean"], "results.stats.ap_open_jsd_attn_K4.mean")
add(159, "0.0131", ST["ap_puct_jsd_attn_K4"]["mean"], "results.stats.ap_puct_jsd_attn_K4.mean")
ci3(159, ("2.00", "1.91", "2.09"), D22["jsd_ratio_open_over_puct"], "verdicts.E1-D2.2.jsd_ratio")
ci3(159, ("0.0131", "0.0126", "0.0135"), D22["jsd_diff_open_minus_puct"], "verdicts.E1-D2.2.jsd_diff")
add(160, "0.0803", ST["ap_open_kl_attn_K4"]["mean"], "results.stats.ap_open_kl_attn_K4.mean")
add(160, "0.0496", ST["ap_puct_kl_attn_K4"]["mean"], "results.stats.ap_puct_kl_attn_K4.mean")
ci3(160, ("38.2%", "35.7", "40.7"), D22["kl_cut_puct_vs_open"], "verdicts.E1-D2.2.kl_cut", 100)
add(161, "0.0179", ST["ap_open_jsd_union"]["mean"], "results.stats.ap_open_jsd_union.mean")
add(161, "0.042", ST["ap_open_kl_union"]["mean"], "results.stats.ap_open_kl_union.mean")
add(161, "0.0041", ST["ap_puct_jsd_union"]["mean"], "results.stats.ap_puct_jsd_union.mean")
add(161, "0.011", ST["ap_puct_kl_union"]["mean"], "results.stats.ap_puct_kl_union.mean")
add(162, "0.85", float(np.mean([SUM[s]["test"]["ap_open_pref_share_NW"] for s in SUM])), "summary ap_open_pref_share_NW (measured)")
ci3(162, ("0.824", "0.815", "0.833"), D22["wave1_nw_share"], "verdicts.E1-D2.2.wave1_nw_share")
add(164, "94.94%", D22["coverage_puct"]["mean"], "coverage_puct.mean", scale=100)
add(164, "0.06", 100 * (0.95 - D22["coverage_puct"]["mean"]), "shortfall in percentage points")
add(166, "12", 100 * float(np.mean([SUM[s]["test"]["ap_coverage_puct"] - SUM[s]["test"]["ap_coverage_open"] for s in SUM])),
    "coverage gain, points")
add(167, "4.4x", ST["ap_open_jsd_union"]["mean"] / ST["ap_puct_jsd_union"]["mean"], "union-call JSD open / puct")
add(167, "2.0x", D22["jsd_ratio_open_over_puct"]["mean"], "attn JSD ratio")
# ---------------- section 6 W table (lines 177-179) and text
for ln, t, ws, r8, r16 in ((177, "0.3", ("0.86", "0.96", "0.93", "0.90", "0.92"), "1.05", "1.08"),
                           (178, "1.5", ("0.56", "0.80", "0.99", "1.12", "1.23"), "2.00", "2.19"),
                           (179, "2.5", ("0.36", "0.58", "0.89", "1.26", "1.61"), "3.54", "4.52")):
    for w, N in zip(ws, ("1", "2", "4", "8", "16")):
        add(ln, w, D3[f"W_t{t}"][N]["mean"], f"verdicts.E1-D3.W_t{t}.N{N}.mean")
    add(ln, r8, D3[f"W8/W1_t{t}"]["mean"], f"verdicts.E1-D3.W8/W1_t{t}.mean")
    add(ln, r16, D3[f"W16/W1_t{t}"]["mean"], f"verdicts.E1-D3.W16/W1_t{t}.mean")
ci3(177, (None, "0.63", "1.52"), D3["W16/W1_t0.3"], "W16/W1_t0.3")
for w, x in zip(("1.09", "1.25", "0.89"), D3["W16/W1_t0.3"]["per_seed"]):
    add(177, w, x, "W16/W1_t0.3.per_seed")
ci3(179, (None, "3.28", "3.79"), D3["W8/W1_t2.5"], "W8/W1_t2.5")
ci3(179, (None, "3.95", "5.09"), D3["W16/W1_t2.5"], "W16/W1_t2.5")
add(178, "0.239", 0.15 + 0.7 * 0.127, "effective flip 0.15 + 0.7 x 0.127")
add(179, "0.346", 0.15 + 0.7 * 0.28, "effective flip 0.15 + 0.7 x 0.28")
add(181, "1.08", D3["W16/W1_t0.3"]["mean"], "W16/W1_t0.3.mean")
ci3(181, ("38.5%", "36.4", "40.7"), D3["msecut_vs_mmd_t0.3"], "msecut_vs_mmd_t0.3", 100)
add(183, "3.54", D3["W8/W1_t2.5"]["mean"], "W8/W1_t2.5.mean")
add(183, "3.28", D3["W8/W1_t2.5"]["lo"], "W8/W1_t2.5.lo")
add(184, "4.52", D3["W16/W1_t2.5"]["mean"], "W16/W1_t2.5.mean")
ci3(184, ("10.3%", "7.7", "13.0"), D3["msecut_vs_kc_t2.5"], "msecut_vs_kc_t2.5", 100)
add(185, "23%", D3["msecut_vs_mmd_t2.5"]["mean"], "msecut_vs_mmd_t2.5.mean", scale=100)
add(185, "47%", D3["msecut_vs_mix_t2.5"]["mean"], "msecut_vs_mix_t2.5.mean", scale=100)
for w, t in zip(("1.05", "2.00", "3.54"), ("0.3", "1.5", "2.5")):
    add(186, w, D3[f"W8/W1_t{t}"]["mean"], f"W8/W1_t{t}.mean")
# ---------------- sections 7, 8, 10
add(206, "52", float(np.mean([json.load(open(os.path.join(R6, "results", "main", f"s{s}", "rc.json")))["sec"] for s in SUM])) / 60,
    "rc.json sec / 60 (minutes)")
for w, x in (("2,553", min(CMP[s]["train_sec"] for s in CMP)), ("2,559", max(CMP[s]["train_sec"] for s in CMP)),
             ("499", min(CMP[s]["fit_sec"] for s in CMP)), ("517", max(CMP[s]["fit_sec"] for s in CMP)),
             ("39", min(CMP[s]["test_sec"] for s in CMP)), ("43", max(CMP[s]["test_sec"] for s in CMP)),
             ("3,106", min(CMP[s]["total_sec"] for s in CMP)), ("3,114", max(CMP[s]["total_sec"] for s in CMP))):
    add(227, w, x, "results.compute")
add(240, "0.86", K["K4"]["pearson"]["mean"], "K4 pearson")
add(240, "0.88", K["K3"]["pearson"]["mean"], "K3 pearson")
add(242, "0.008", ST["fs_jsd_attn_K4"]["mean"], "fs_jsd_attn_K4.mean")
add(242, "2.9", D21["kc"]["jsd_ratio"]["mean"], "kc JSD ratio (low end of '2.9-6.5x')")
add(242, "6.5", D21["mmd"]["jsd_ratio"]["mean"], "mmd JSD ratio (high end)")
add(246, "95%", D22["coverage_puct"]["mean"], "'Key-space PUCT reaches 95% coverage' vs coverage_puct.mean", scale=100)
add(248, "10%", D3["msecut_vs_kc_t2.5"]["mean"], "msecut_vs_kc_t2.5.mean", scale=100)
add(249, "2.0x", D21["looped_secondary"]["kl_ratio_to_attn"]["mean"], "looped KL ratio")
add(251, "5%", 100 * (1 - D22["coverage_puct"]["mean"]), "100 - coverage_puct (%)")

with open(os.path.join(OUT, "claims-curated.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
meas = [r for r in rows if r["kind"] == "measured"]
bad = [r for r in rows if r["status"] != "OK"]
print(f"{len(rows)} hand-mapped numbers ({len(meas)} measured, {len(rows) - len(meas)} bars/design); "
      f"{len(rows) - len(bad)} OK, {len(bad)} WRONG")
for r in bad:
    print(f"  line {r['line']}: written {r['written']!r}; correct {r['correct_rounding']} (file value {r['file_value']:.6f}); "
          f"{r['status']}; source {r['source']}")
