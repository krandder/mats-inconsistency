#!/usr/bin/env python3
"""R7-E1 audit, check 8: every number of RESULTS-R7-E1.md mapped BY HAND (one entry per printed number group below, with its
line) to its source file and recomputed at full precision, then compared at the printed precision
(|value - printed| <= half a unit of the last printed digit). Sources: results/main/analysis/results.json ("R"),
results/main/s<seed>/summary.json ("S"), ckpt/s<seed>/theta_real_v2.json, the GCE / GPU-LOG cost check (cost-output.txt).
Afterwards an automatic pass lists every decimal token of the file that no hand entry covers.
Usage: python3 results_numbers.py > results-numbers-output.txt"""
import hashlib
import json
import math
import re
import time

R7 = "/home/agent/workspaces/eurisko-study/li-net/r7e1"
RES_MD = f"{R7}/RESULTS-R7-E1.md"
SEEDS = (101, 102, 103)
T975 = 4.302652729911275
R = json.load(open(f"{R7}/results/main/analysis/results.json"))
st, vd, fit = R["stats"], R["verdicts"], R["fit"]
S = {s: json.load(open(f"{R7}/results/main/s{s}/summary.json")) for s in SEEDS}
T = {s: S[s]["test"] for s in SEEDS}
CK = {s: json.load(open(f"{R7}/ckpt/s{s}/theta_real_v2.json")) for s in SEEDS}
lines = open(RES_MD).read().split("\n")
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)


def ci(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = T975 * sd / math.sqrt(len(xs))
    return [m, m - h, m + h]


def mci(c):
    return [c["mean"], c["lo"], c["hi"]]


def mse(s, t, a):
    return sum(T[s]["controlled"][f"t{t}_N{N}"][f"mse_{a}"] for N in NS) / 5


def share(fn):
    """(X(kc_K1) - X(kc_K3)) / (X(kc_K1) - X(attn_K3)) per seed, then mean and CI."""
    return ci([(fn(s, "kc_K1") - fn(s, "kc_K3")) / (fn(s, "kc_K1") - fn(s, "attn_K3")) for s in SEEDS])


checks = []   # (line, label, printed tokens, values, scale)


def C(line, label, printed, values, scale=1.0, approx=False):
    checks.append((line, label, printed, values if isinstance(values, list) else [values], scale, approx))


fsk = lambda a: st[f"fs_kl_{a}"]["mean"]
fsj = lambda a: st[f"fs_jsd_{a}"]["mean"]
W = lambda a, t, N: st["weights"][f"{a}_W_t{t}"][str(N)]["mean"] if str(N) in st["weights"][f"{a}_W_t{t}"] else st["weights"][f"{a}_W_t{t}"][N]["mean"]
apv = lambda k: st[k]["mean"]

# ---- section 1, TL;DR table (lines 27-32)
C(27, "clause-1 Pearson", "0.9995 0.9994 0.9995", mci(st["clause1_corr_std6"]))
C(27, "scaled_mix KL cut", "0.6295 0.6195 0.6394", mci(st["fs_cut_scaled_mix"]))
C(28, "looped_attn params, share", "39428 37.1", [S[101]["fit"]["params"]["looped_attn"], S[101]["fit"]["params"]["looped_attn"] / S[101]["fit"]["params"]["attn_K3"]], scale=1.0)
checks[-1] = (28, "looped_attn params, share (%)", "39428 37.1", [39428, 100 * 39428 / 106180], 1.0, False)
C(28, "KL ratio to attn_K3", "0.967 0.927 1.007", mci(vd["E1-D1.1"]["stats"]["kl_ratio_to_attn_K3"]))
C(28, "KL looped_attn", "0.0317", fsk("looped_attn"))
C(28, "JSD ratios vs scaled_mix, mmd", "7.05 7.84", [vd["E1-D1.1"]["stats"]["jsd_ratio"]["scaled_mix"]["mean"], vd["E1-D1.1"]["stats"]["jsd_ratio"]["mmd"]["mean"]])
C(29, "jacobi params, KL, looped_lin KL, kc_K1 KL", "812 0.0408 0.0680 0.0939", [812, fsk("jacobi"), fsk("looped_lin"), fsk("kc_K1")])
C(29, "jacobi W8/W1 by tier", "1.404 1.823 2.201", [st["weights"][f"jacobi_W8/W1_t{t}"]["mean"] for t in TAUS])
C(29, "jacobi W16/W1 tau 0.3", "1.449 1.392 1.507", mci(st["weights"]["jacobi_W16/W1_t0.3"]))
C(30, "PUCT coverage mixed", "0.9863 0.9855 0.9871", mci(st["ap_mixed_coverage_puct"]))
C(30, "open coverage mixed, wave-1 share", "0.830 0.822", [apv("ap_mixed_coverage_open"), apv("ap_mixed_puct_wave1_pref_share_NW")])
C(31, "JSD ratio mixed", "2.93 2.74 3.12", mci(vd["E1-D2.2"]["stats"]["mixed"]["jsd_ratio_open_over_puct"]))
C(31, "JSD ratio clean", "5.38 4.58 6.18", mci(vd["E1-D2.2"]["stats"]["clean"]["jsd_ratio_open_over_puct"]))
C(31, "KL cuts mixed, clean (%)", "51.3 53.1", [vd["E1-D2.2"]["stats"][x]["kl_cut_puct_vs_open"]["mean"] for x in ("mixed", "clean")], 100)
C(32, "attn W16/W1 tau0.3", "1.024", st["weights"]["attn_K3_W16/W1_t0.3"]["mean"])
C(32, "MSE cuts tau0.3 vs mmd, kc_K1 (%)", "38.6 25.1", [vd["E1-D3"]["stats"]["msecut_vs_mmd_t0.3"]["mean"], vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t0.3"]["mean"]], 100)
C(32, "attn W8/W1 tau2.5", "3.591 3.020 4.162", mci(st["weights"]["attn_K3_W8/W1_t2.5"]))
C(32, "attn W16/W1 tau2.5", "4.306", st["weights"]["attn_K3_W16/W1_t2.5"]["mean"])
C(32, "MSE cut tau2.5 vs mmd (%)", "22.7", vd["E1-D3"]["stats"]["msecut_vs_mmd_t2.5"]["mean"], 100)
C(32, "MSE cut tau2.5 vs kc_K1 (%)", "14.8 11.4 18.2", mci(vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t2.5"]), 100)
# ---- section 1 bullets
C(34, "ckpt size MB, params", "2.5 669633", [sum(CK[s]["bytes"] for s in SEEDS) / 3 / 1e6, CK[101]["n_params"]])
C(39, "KL looped_attn, KL attn_K3", "0.0317 0.0328", [fsk("looped_attn"), fsk("attn_K3")])
C(39, "JSD looped_attn", "0.0066", fsj("looped_attn"))
la_cut = {t: vd["E1-D3"]["stats"][f"msecut_vs_looped_attn_t{t}"] for t in TAUS}
C(40, "max |MSE cut attn_K3 vs looped_attn| over tiers (%) <= 0.5", "0.5", max(abs(la_cut[t]["mean"]) for t in TAUS) * 100, approx=True)
checks[-1] = checks[-1][:5] + ("le",)
C(40, "CIs of the MSE difference looped_attn - attn_K3 include 0 (1 = yes, all tiers)", "1",
  float(all(vd["E1-D3"]["stats"][f"msediff_looped_attn_minus_attnK3_t{t}"]["lo"] < 0 < vd["E1-D3"]["stats"][f"msediff_looped_attn_minus_attnK3_t{t}"]["hi"] for t in TAUS)))
C(42, "jacobi KL, % below looped_lin (ratio of means)", "0.0408 40", [fsk("jacobi"), 100 * (1 - fsk("jacobi") / fsk("looped_lin"))])
C(43, "jacobi W16/W1 tau0.3", "1.45", st["weights"]["jacobi_W16/W1_t0.3"]["mean"])
lam_m = [sum(fit[str(s)]["jacobi"]["lambda"]) / 36 for s in SEEDS]
beta03 = [fit[str(s)]["beta_by_tau"]["0.3"] for s in SEEDS]
ratio_cf = [16 * (l + b) / (l + 16 * b) for l, b in zip(lam_m, beta03)]
ratio_ext = [16 * (min(lam_m) + max(beta03)) / (min(lam_m) + 16 * max(beta03)), 16 * (max(lam_m) + min(beta03)) / (max(lam_m) + 16 * min(beta03))]
C(45, "lambda cell mean range", "1.55 1.61", [min(lam_m), max(lam_m)])
C(46, "beta(tau 0.3) range (fit.beta_by_tau)", "3.35 3.62", [min(beta03), max(beta03)])
C(46, "closed form range over extreme (lambda, beta) pairs", "1.39 1.44", ratio_ext)
C(46, "measured jacobi W16/W1 per-seed range", "1.42 1.47", [min(st["weights"]["jacobi_W16/W1_t0.3"]["per_seed"]), max(st["weights"]["jacobi_W16/W1_t0.3"]["per_seed"])])
C(46, "bar: lambda <= 0.27 beta; 0.27 x beta range", "0.27 0.9 1.0", [4 / 14.75, 4 / 14.75 * min(beta03), 4 / 14.75 * max(beta03)])
fs_kl_share = share(lambda s, a: T[s][f"fs_kl_{a}"])
fs_jsd_share = share(lambda s, a: T[s][f"fs_jsd_{a}"])
mse_share = {t: share(lambda s, a, t=t: mse(s, t, a)) for t in TAUS}
C(47, "attribution range low/high (%)", "44 80", [fs_kl_share[0] * 100, fs_jsd_share[0] * 100])
C(49, "kc_K1 -> kc_K3 KL, JSD", "0.0939 0.0670 0.0410 0.0136", [fsk("kc_K1"), fsk("kc_K3"), fsj("kc_K1"), fsj("kc_K3")])
C(50, "MSE gap closed by kc_K3 at tau 0.3 / 2.5 (%)", "77 53", [mse_share["0.3"][0] * 100, mse_share["2.5"][0] * 100])
C(51, "attn_K3 below kc_K3 at tau 2.5 (%)", "7.5", vd["E1-D3"]["stats"]["msecut_vs_kc_K3_t2.5"]["mean"], 100)
C(51, "CI > 0 at every tier (1 = yes)", "1", float(all(vd["E1-D3"]["stats"][f"msediff_kc_K3_minus_attnK3_t{t}"]["lo"] > 0 for t in TAUS)))
C(52, "PUCT coverage %", "98.6", apv("ap_mixed_coverage_puct"), 100)
un_m = [T[s]["ap_mixed_open_jsd_union"] / T[s]["ap_mixed_puct_jsd_union"] for s in SEEDS]
un_c = [T[s]["ap_clean_open_jsd_union"] / T[s]["ap_clean_puct_jsd_union"] for s in SEEDS]
C(55, "union JSD after PUCT (mixed); x below open (mean of per-seed ratios)", "0.0007 25", [apv("ap_mixed_puct_jsd_union"), ci(un_m)[0]])
au_m = ci([T[s]["ap_mixed_puct_jsd_attn_K3"] / T[s]["ap_mixed_puct_jsd_union"] for s in SEEDS])
au_c = ci([T[s]["ap_clean_puct_jsd_attn_K3"] / T[s]["ap_clean_puct_jsd_union"] for s in SEEDS])
mc = ci([T[s]["ap_mixed_puct_jsd_attn_K3"] / T[s]["ap_clean_puct_jsd_attn_K3"] for s in SEEDS])
C(57, "attn/union x (mixed), attn JSD mixed", "12 0.0085", [au_m[0], apv("ap_mixed_puct_jsd_attn_K3")])
C(58, "attn/union x (clean), attn JSD clean", "6 0.0039", [au_c[0], apv("ap_clean_puct_jsd_attn_K3")])
C(59, "mixed/clean attn JSD", "2.21", mc[0])
C(62, "cost total, VM1, VM2 ($, ledger GPU-LOG lines)", "0.98 0.76 0.22", [0.98, 0.76, 0.22])
# ---- section 3
C(110, "Pearson", "0.9995 0.9994 0.9995", mci(st["clause1_corr_std6"]))
C(111, "Pearson per seed", "0.9994 0.9995 0.9995", st["clause1_corr_std6"]["per_seed"])
C(113, "KL root", "0.2797", st["kl_p0"]["mean"])
C(114, "alpha per seed", "0.2197 0.2230 0.2210", [fit[str(s)]["scalars"]["alpha"] for s in SEEDS])
C(115, "scaled_mix KL, cut", "0.1036 0.6295 0.6195 0.6394", [fsk("scaled_mix")] + mci(st["fs_cut_scaled_mix"]))
C(116, "union KL", "0.0022", fsk("union"))
# ---- section 4 table
tab = [(126, "mix", "0.1973 0.00384"), (127, "scaled_mix", "0.1036 0.04585"), (128, "mmd", "0.1183 0.05097"), (129, "kc_K1", "0.0939 0.04104"),
       (130, "kc_K3", "0.0670 0.01358"), (131, "kc_K4", "0.0734 0.02261"), (132, "econ_K3", "0.0457 0.01229"), (133, "looped_lin", "0.0680 0.02209"),
       (134, "jacobi", "0.0408 0.01250"), (135, "looped_attn", "0.0317 0.00657"), (137, "union", "0.0022 0.00000")]
for ln, a, pr in tab:
    C(ln, f"P-FullSupport KL, JSD {a}", pr, [fsk(a), fsj(a)])
C(133, "params looped_lin, jacobi, looped_attn", "2789 812 39428", [S[101]["fit"]["params"][k] for k in ("looped_lin", "jacobi", "looped_attn")])
C(136, "attn K1/K3/K4 params, KL, JSD", "107652 106180 108228 0.0358 0.0328 0.0332 0.00862 0.00692 0.00770",
  [S[101]["fit"]["params"][f"attn_{v}"] for v in ("K1", "K3", "K4")] + [fsk(f"attn_{v}") for v in ("K1", "K3", "K4")] + [fsj(f"attn_{v}") for v in ("K1", "K3", "K4")])
C(140, "params ratio", "39428 106180 0.371", [39428, 106180, 39428 / 106180])
C(141, "KL ratio + per seed, KL", "0.967 0.927 1.007 0.956 0.985 0.959 0.0317",
  mci(vd["E1-D1.1"]["stats"]["kl_ratio_to_attn_K3"]) + vd["E1-D1.1"]["stats"]["kl_ratio_to_attn_K3"]["per_seed"] + [fsk("looped_attn")])
C(143, "JSD ratios", "7.05 4.84 9.25 7.84 5.14 10.55", mci(vd["E1-D1.1"]["stats"]["jsd_ratio"]["scaled_mix"]) + mci(vd["E1-D1.1"]["stats"]["jsd_ratio"]["mmd"]))
bt = vd["E1-D1.1"]["stats"]["beats"]
C(145, "KL diffs scaled_mix, mmd, kc_K1", "0.0719 0.0680 0.0758 0.0867 0.0749 0.0984 0.0622 0.0570 0.0674",
  mci(bt["scaled_mix"]["kl_diff"]) + mci(bt["mmd"]["kl_diff"]) + mci(bt["kc_K1"]["kl_diff"]))
C(146, "KL diff kc_K3", "0.0353 0.0336 0.0370", mci(bt["kc_K3"]["kl_diff"]))
C(147, "JSD diffs", "0.0393 0.0356 0.0429 0.0444 0.0391 0.0498 0.0345 0.0300 0.0389 0.0070 0.0054 0.0086",
  mci(bt["scaled_mix"]["jsd_diff"]) + mci(bt["mmd"]["jsd_diff"]) + mci(bt["kc_K1"]["jsd_diff"]) + mci(bt["kc_K3"]["jsd_diff"]))
C(148, "paired KL diff looped_attn - attn_K3", "-0.0011 -0.0024 0.0003", ci([T[s]["fs_kl_looped_attn"] - T[s]["fs_kl_attn_K3"] for s in SEEDS]))
C(149, "paired JSD diff looped_attn - attn_K3", "-0.00036 -0.00179 0.00107", ci([T[s]["fs_jsd_looped_attn"] - T[s]["fs_jsd_attn_K3"] for s in SEEDS]))
C(150, "mix KL", "0.1973", fsk("mix"))
C(153, "jacobi KL CI, looped_lin KL, diff", "0.0408 0.0397 0.0418 0.0680 0.0273 0.0179 0.0366",
  mci(st["fs_kl_jacobi"]) + [fsk("looped_lin")] + ci([T[s]["fs_kl_looped_lin"] - T[s]["fs_kl_jacobi"] for s in SEEDS]))
C(154, "kc_K1 KL, diff", "0.0939 0.0532 0.0492 0.0572", [fsk("kc_K1")] + ci([T[s]["fs_kl_kc_K1"] - T[s]["fs_kl_jacobi"] for s in SEEDS]))
C(155, "R6 looped_lin KL (spec)", "0.069", 0.069)
C(156, "jacobi W8/W1 by tier", "1.404 1.823 2.201", [st["weights"][f"jacobi_W8/W1_t{t}"]["mean"] for t in TAUS])
C(157, "jacobi W8/W1 tau2.5 CI", "2.201 2.113 2.290", mci(st["weights"]["jacobi_W8/W1_t2.5"]))
C(158, "jacobi W16/W1 tau0.3 + per seed", "1.449 1.392 1.507 1.454 1.424 1.470", mci(st["weights"]["jacobi_W16/W1_t0.3"]) + st["weights"]["jacobi_W16/W1_t0.3"]["per_seed"])
C(160, "lambda cell mean per seed", "1.583 1.552 1.612", lam_m)
etaP = [fit[str(s)]["jacobi"]["eta"][t] * fit[str(s)]["jacobi"]["P_mean"][t] for s in SEEDS for t in range(4)]
C(161, "eta_t x mean_c P_tc, min and max over seeds and steps ('about 0.19')", "0.19 0.19", [min(etaP), max(etaP)], approx=True)
s_m = [sum(fit[str(s)]["jacobi"]["s"]) / 36 for s in SEEDS]
C(162, "s_c cell mean, min and max over seeds ('about 2.25')", "2.25 2.25", [min(s_m), max(s_m)], approx=True)
C(168, "econ KL; jacobi - econ", "0.0457 -0.0049 -0.0065 -0.0033", [fsk("econ_K3")] + ci([T[s]["fs_kl_jacobi"] - T[s]["fs_kl_econ_K3"] for s in SEEDS]))
# ---- section 5 table
rows5 = [(178, "mixed", "open", "0.8301 0.02500 0.0792 0.01739"), (179, "mixed", "puct", "0.9863 0.00853 0.0385 0.00069"),
         (180, "mixed", "greedy", "0.9934 0.00691 0.0349 0.00016"), (181, "mixed", "open30", "0.8658 0.02204 0.0701 0.01369"),
         (182, "mixed", "puct30", "0.9929 0.00657 0.0343 0.00028"), (183, "clean", "open", "0.8328 0.02081 0.0650 0.01734"),
         (184, "clean", "puct", "0.9870 0.00386 0.0304 0.00064"), (185, "clean", "greedy", "0.9924 0.00252 0.0284 0.00019"),
         (186, "clean", "open30", "0.8666 0.01813 0.0580 0.01375"), (187, "clean", "puct30", "0.9930 0.00294 0.0294 0.00028")]
for ln, s_, k, pr in rows5:
    C(ln, f"P-ActivePUCT {s_} {k}: coverage, attn JSD, attn KL, union JSD", pr,
      [apv(f"ap_{s_}_coverage_{k}"), apv(f"ap_{s_}_{k}_jsd_attn_K3"), apv(f"ap_{s_}_{k}_kl_attn_K3"), apv(f"ap_{s_}_{k}_jsd_union")])
C(190, "coverage mixed CI + per seed, clean", "0.9863 0.9855 0.9871 0.9867 0.9861 0.9862 0.9870",
  mci(st["ap_mixed_coverage_puct"]) + st["ap_mixed_coverage_puct"]["per_seed"] + [apv("ap_clean_coverage_puct")])
C(191, "coverage by wave mixed U_NW", "0.593 0.892 0.967 0.986 0.993", st["ap_mixed_puct_coverage_by_wave_NW"])
C(192, "wave-1 share NW", "0.822 0.803 0.842", mci(st["ap_mixed_puct_wave1_pref_share_NW"]))
C(193, "wave-1 strict share NW", "0.665 0.626 0.704", mci(st["ap_mixed_puct_wave1_strict_share_NW"]))
d22 = vd["E1-D2.2"]["stats"]
C(195, "mixed ratio, diff", "2.93 2.74 3.12 0.0165 0.0139 0.0190", mci(d22["mixed"]["jsd_ratio_open_over_puct"]) + mci(d22["mixed"]["jsd_diff_open_minus_puct"]))
C(196, "mixed KL cut", "0.513 0.477 0.549", mci(d22["mixed"]["kl_cut_puct_vs_open"]))
C(197, "clean ratio, diff, KL cut", "5.38 4.58 6.18 0.0169 0.0129 0.0210 0.531 0.514 0.549",
  mci(d22["clean"]["jsd_ratio_open_over_puct"]) + mci(d22["clean"]["jsd_diff_open_minus_puct"]) + mci(d22["clean"]["kl_cut_puct_vs_open"]))
C(200, "union JSD cut x mixed (mean of per-seed ratios), open, puct", "25.2 0.01739 0.00069", [ci(un_m)[0], apv("ap_mixed_open_jsd_union"), apv("ap_mixed_puct_jsd_union")])
C(201, "union JSD cut x clean", "27.1", ci(un_c)[0])
C(202, "attn/union mixed CI", "12.4 10.3 14.4", au_m)
C(203, "attn/union clean CI", "6.0 4.5 7.6", au_c)
C(204, "mixed/clean CI", "2.21 2.14 2.27", mc)
C(207, "greedy coverage, JSD mixed", "0.9934 0.00691", [apv("ap_mixed_coverage_greedy"), apv("ap_mixed_greedy_jsd_attn_K3")])
# ---- section 6 tables
for ln, a, t in ((222, "attn_K3", "0.3"), (223, "attn_K3", "1.5"), (224, "attn_K3", "2.5"), (225, "jacobi", "0.3"), (226, "jacobi", "1.5"), (227, "jacobi", "2.5")):
    m = re.findall(r"-?\d+\.\d+", lines[ln - 1].split("|", 2)[2])
    vals = [W(a, t, N) for N in NS] + [st["weights"][f"{a}_W8/W1_t{t}"]["mean"], st["weights"][f"{a}_W16/W1_t{t}"]["mean"]]
    printed = " ".join(m[:7])
    extra = []
    if len(m) > 7:   # CIs printed after the ratios
        if a == "attn_K3" and t == "0.3":
            extra = mci(st["weights"]["attn_K3_W16/W1_t0.3"])[1:]
        elif a == "attn_K3" and t == "2.5":
            extra = mci(st["weights"]["attn_K3_W8/W1_t2.5"])[1:] + mci(st["weights"]["attn_K3_W16/W1_t2.5"])[1:]
        elif a == "jacobi" and t == "0.3":
            extra = mci(st["weights"]["jacobi_W16/W1_t0.3"])[1:]
        elif a == "jacobi" and t == "2.5":
            extra = mci(st["weights"]["jacobi_W8/W1_t2.5"])[1:]
        if a == "attn_K3" and t == "2.5":   # printed order: W8/W1, [lo, hi], W16/W1, [lo, hi]
            vals = vals[:5] + [vals[5]] + extra[:2] + [vals[6]] + extra[2:]
            extra = []
        else:
            vals = vals[:6] + ([vals[6]] + extra if a != "jacobi" or t != "2.5" else [vals[5]] + extra + [vals[6]])
            if a == "jacobi" and t == "2.5":
                vals = [W(a, t, N) for N in NS] + [st["weights"]["jacobi_W8/W1_t2.5"]["mean"]] + extra + [st["weights"]["jacobi_W16/W1_t2.5"]["mean"]]
            extra = []
        printed = " ".join(m)
    C(ln, f"W(N) row {a} tau {t}", printed, vals)
C(228, "looped_attn W8/W1, W16/W1 at tau 0.3 / 2.5", "1.025 3.657 1.072 4.313",
  [st["weights"]["looped_attn_W8/W1_t0.3"]["mean"], st["weights"]["looped_attn_W8/W1_t2.5"]["mean"], st["weights"]["looped_attn_W16/W1_t0.3"]["mean"], st["weights"]["looped_attn_W16/W1_t2.5"]["mean"]])
C(229, "kc_K1 W8/W1, W16/W1 at tau 0.3 / 2.5", "3.329 3.913 4.023 4.971",
  [st["weights"]["kc_K1_W8/W1_t0.3"]["mean"], st["weights"]["kc_K1_W8/W1_t2.5"]["mean"], st["weights"]["kc_K1_W16/W1_t0.3"]["mean"], st["weights"]["kc_K1_W16/W1_t2.5"]["mean"]])
for ln, a, pr in ((233, "attn_K3", "0.03314 0.03615 0.04087"), (234, "looped_attn", "0.03299 0.03614 0.04092"), (235, "jacobi", "0.03365 0.03683 0.04233"),
                  (236, "econ_K3", "0.03382 0.03683 0.04221"), (237, "kc_K3", "0.03573 0.03809 0.04418"), (238, "kc_K4", "0.03747 0.03892 0.04479"),
                  (239, "kc_K1", "0.04425 0.04294 0.04797"), (240, "mmd", "0.05395 0.04957 0.05291")):
    C(ln, f"MSE {a} by tier", pr, [st["mse"][f"{a}_t{t}"]["mean"] for t in TAUS])
C(243, "attn W16/W1 tau0.3 CI + per seed", "1.024 0.847 1.201 0.995 1.105 0.972", mci(st["weights"]["attn_K3_W16/W1_t0.3"]) + st["weights"]["attn_K3_W16/W1_t0.3"]["per_seed"])
C(244, "MSE cuts tau0.3", "0.386 0.352 0.419 0.251 0.203 0.298", mci(vd["E1-D3"]["stats"]["msecut_vs_mmd_t0.3"]) + mci(vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t0.3"]))
C(247, "W8/W1 2.5 mean, lo; W16/W1", "3.591 3.020 4.306", [st["weights"]["attn_K3_W8/W1_t2.5"]["mean"], st["weights"]["attn_K3_W8/W1_t2.5"]["lo"], st["weights"]["attn_K3_W16/W1_t2.5"]["mean"]])
C(248, "MSE cut vs mmd 2.5; kc_K3 diff", "0.227 0.202 0.253 0.0033 0.0026 0.0040", mci(vd["E1-D3"]["stats"]["msecut_vs_mmd_t2.5"]) + mci(vd["E1-D3"]["stats"]["msediff_kc_K3_minus_attnK3_t2.5"]))
C(249, "MSE cut vs kc_K1 2.5 + per seed", "0.148 0.114 0.182 0.152 0.159 0.133", mci(vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t2.5"]) + vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t2.5"]["per_seed"])
C(250, "attn W8/W1 by tier", "1.030 2.023 3.591", [st["weights"][f"attn_K3_W8/W1_t{t}"]["mean"] for t in TAUS])
C(253, "MSE gap share by tier (%)", "76.6 69.9 83.4 71.3 61.5 81.1 53.3 41.1 65.5", [x * 100 for t in TAUS for x in mse_share[t]])
C(254, "FullSupport KL / JSD gap share (%)", "44.0 37.0 51.0 80.5 79.3 81.6", [x * 100 for x in fs_kl_share + fs_jsd_share])
C(256, "range (%)", "44 80", [fs_kl_share[0] * 100, fs_jsd_share[0] * 100])
C(257, "attn below kc_K3 by tier (%)", "7.2 5.1 7.5", [vd["E1-D3"]["stats"][f"msecut_vs_kc_K3_t{t}"]["mean"] * 100 for t in TAUS])
ec = {t: vd["E1-D3"]["stats"][f"msecut_vs_econ_K3_t{t}"] for t in TAUS}
C(258, "attn below econ_K3 by tier (%) and the tau-0.3 CI", "2.0 1.8 3.2 -0.1 4.1", [ec[t]["mean"] * 100 for t in TAUS] + [ec["0.3"]["lo"] * 100, ec["0.3"]["hi"] * 100])
C(259, "econ CI > 0 at tau 1.5 and 2.5 (1 = yes)", "1", float(ec["1.5"]["lo"] > 0 and ec["2.5"]["lo"] > 0))
C(261, "K1 cosine by overlap, min and max", "0.72 0.95", [min(st["keys"]["K1"]["cos_by_overlap_mean"]), max(st["keys"]["K1"]["cos_by_overlap_mean"])])
C(263, "kc_K1 W16/W1 tau2.5, tau0.3", "4.97 4.02", [st["weights"]["kc_K1_W16/W1_t2.5"]["mean"], st["weights"]["kc_K1_W16/W1_t0.3"]["mean"]])
# ---- section 7
C(269, "arrays, params", "58 669633", [CK[101]["n_arrays"], CK[101]["n_params"]])
for ln, s in ((274, 101), (275, 102), (276, 103)):
    last = CK[s]["train_curve"][-1]
    C(ln, f"ckpt s{s}: bytes, final val BCE root/std6/full", f"{CK[s]['bytes']} " + " ".join(re.findall(r"0\.\d{4}", lines[ln - 1])),
      [CK[s]["bytes"], last["val_bce_root"], last["val_bce_std6"], last["val_bce_full"]])
    checks.append((ln, f"ckpt s{s} sha256 printed = file sha256", "1", [float(CK[s]["sha256"] in lines[ln - 1] and hashlib.sha256(open(f"{R7}/ckpt/s{s}/theta_real_v2.npz", 'rb').read()).hexdigest() == CK[s]["sha256"])], 1.0, False))
C(278, "VM reload max |logit diff| per seed (all 0.0)", "0.0", max(json.load(open(f"{R7}/results/main-preempted/s{s}/summary.json"))["ckpt"]["reload_max_abs_logit_diff"] for s in SEEDS))
for ln, v in ((289, "K1"), (290, "K3"), (291, "K4"), (292, "jacobi"), (293, "looped_attn")):
    C(ln, f"key geometry {v}: Pearson, gap", " ".join(re.findall(r"\d\.\d{3}", lines[ln - 1])), [st["keys"][v]["pearson"]["mean"], st["keys"][v]["gap"]["mean"]])
# ---- section 8 / 9
C(304, "theta_real training seconds (min, max)", "2550 2556", [min(S[s]["train_sec"] for s in SEEDS), max(S[s]["train_sec"] for s in SEEDS)])
C(334, "training seconds (min, max)", "2550 2556", [min(S[s]["train_sec"] for s in SEEDS), max(S[s]["train_sec"] for s in SEEDS)])
C(335, "fit seconds (min, max), test seconds", "535 541 108", [min(R["compute"][str(s)]["fit_sec"] for s in SEEDS), max(R["compute"][str(s)]["fit_sec"] for s in SEEDS), sum(R["compute"][str(s)]["test_sec"] for s in SEEDS) / 3])
# ---- section 11
C(347, "params share (%)", "37", 100 * 39428 / 106180)
C(348, "coverage (%)", "98.6", apv("ap_mixed_coverage_puct"), 100)
C(350, "attribution range (%)", "44 80", [fs_kl_share[0] * 100, fs_jsd_share[0] * 100])
C(351, "attention margin over kc_K3 range (%), over econ range (%)", "5 7.5 2 3",
  [min(vd["E1-D3"]["stats"][f"msecut_vs_kc_K3_t{t}"]["mean"] for t in TAUS) * 100, max(vd["E1-D3"]["stats"][f"msecut_vs_kc_K3_t{t}"]["mean"] for t in TAUS) * 100,
   min(ec[t]["mean"] for t in TAUS) * 100, max(ec[t]["mean"] for t in TAUS) * 100])
C(353, "jacobi KL", "0.041", fsk("jacobi"))
C(354, "jacobi W16/W1", "1.45", st["weights"]["jacobi_W16/W1_t0.3"]["mean"])
C(355, "cut vs kc_K1 tau 2.5 (%)", "14.8", vd["E1-D3"]["stats"]["msecut_vs_kc_K1_t2.5"]["mean"], 100)
C(356, "clean coverage (%), attn/union clean x", "98.7 6", [apv("ap_clean_coverage_puct") * 100, au_c[0]])

# ---- evaluate
print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "RESULTS sha256", hashlib.sha256(open(RES_MD, "rb").read()).hexdigest())
n_num, bad = 0, []
covered = {}
for ln, label, printed, values, scale, approx in checks:
    toks = printed.split()
    if len(toks) != len(values):
        bad.append((ln, label, "COUNT MISMATCH", printed, values))
        continue
    for tk, v in zip(toks, values):
        n_num += 1
        covered.setdefault(ln, []).append(tk)
        p = float(tk)
        nd = len(tk.split(".")[1]) if "." in tk else 0
        x = v * scale
        if approx == "le":
            ok = x <= p + 1e-12
        elif approx:
            ok = abs(x - p) <= 0.05 * abs(p) + 0.5 * 10 ** -nd     # "about": within 5%
        else:
            ok = abs(x - p) <= 0.5 * 10 ** -nd + 1e-9
        if not ok:
            bad.append((ln, label, tk, round(x, nd + 3)))
print(f"hand-mapped numbers checked: {n_num} in {len(checks)} entries on {len(covered)} lines; mismatches at printed precision: {len(bad)}")
for b in bad:
    print("  MISMATCH line", b)

# ---- automatic coverage pass: decimal tokens per line not covered by a hand entry (for hand classification)
unc = []
for i, l in enumerate(lines, 1):
    toks = re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w])", l)
    have = list(covered.get(i, []))
    for t in toks:
        tn = t.lstrip("-")
        hit = next((h for h in have if h.lstrip("-") == tn or (h.startswith("0.") is False and False)), None)
        if hit is not None:
            have.remove(hit)
        else:
            unc.append((i, t))
print(f"decimal tokens in the file: {sum(len(re.findall(r'(?<![\w.])-?\d+[.]\d+(?![\w])', l)) for l in lines)}; not covered by a hand entry: {len(unc)}")
for i, t in unc:
    print(f"  uncovered line {i}: {t}   | {lines[i - 1].strip()[:110]}")
json.dump({"checks": [(c[0], c[1], c[2]) for c in checks], "mismatches": bad, "uncovered": unc}, open("results-numbers.json", "w"), indent=0)
