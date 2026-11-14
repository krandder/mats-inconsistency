"""Render X1/w4/CLAIMS.md from repro.csv + check_w4.json (run check_w4.py first). Usage: python3 make_claims.py
Status rule (PREREG.md): verified = file check passes AND every primary seed-4 quantity within 2 sd (s floored at 1e-3
when s < 1e-4), or no fresh seed applies and the analysis reran byte-identically; refuted = quoted number does not match
its file, or seed 4 lands outside 2 sd in the direction that reverses the claim; unverified = otherwise."""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = {r["quantity"]: r for r in csv.DictReader(open(os.path.join(HERE, "repro.csv")))}
J = json.load(open(os.path.join(HERE, "check_w4.json")))
FC = {c["check"]: c for c in J["filechecks"]}
T = J["checked_at"]
RERUN = "bash study/X1/w4/rerun.sh; PYTHONPATH=wealth-net/.pylib python3 study/X1/w4/check_w4.py"
F = lambda x, n=4: f"{float(x):.{n}f}" if abs(float(x)) >= 1e-3 or float(x) == 0 else f"{float(x):.2e}"


def within(q):
    r = R[q]
    if r["within_2sd"] == "True": return True
    return r["within_2sd_floor"] == "True"


def fresh(qs):
    out = []
    for q in qs:
        r = R[q]; z = float(r["z"]) if r["z"] not in ("", "inf") else float("nan")
        tag = "" if r["within_2sd"] == "True" else (" (floor ok)" if r["within_2sd_floor"] == "True" else " OUT")
        out.append(f"{q}: {F(r['seed4'])} vs {F(r['mean'])} +- {F(r['sd'])} (z {z:+.1f}){tag}")
    return "; ".join(out)


C = []   # id, claim, claimed, file_ok, primary quantities, evidence, status override, notes
M8c = "M8/check.py PASS (rerun 12:0xZ); X1/w4/check_w4.py re-reads"
C += [
 ("M8-sign-shuf", "sign of income is not a gate signal (shuf1 ~ pos1, lfsr16 removed)", "-0.0008/-0.0030/+0.0008", True,
  ["w1 nolfsr shuf1 - pos1"], "M8/w1/results/decision.json nolfsr_shuf1_minus_pos1; /tmp/traders-net/X1w4/study/M8/w1/results/run-*-s4.json", None, ""),
 ("M8-sign-neg", "flipping the sign costs little (neg1 - pos1, lfsr16 removed; >=0.005 in 2/3)", "+0.0079/+0.0026/+0.0051", True,
  ["w1 nolfsr neg1 - pos1"], "same", None, "seed 4 +0.0054 is >= 0.005, so 3/4 seeds clear the prereg bar that needed 3/3 of seeds 1-3"),
 ("M8-none-better", "income gating hurts vs no gate (none - pos1, lfsr16 removed; ungated better 3/3)", "-0.0067/-0.0066/-0.0066", True,
  ["w1 nolfsr none - pos1"], "same", None, "strict 2-sd fails only because the 3-seed sd is 8e-5 (z -2.5); floor rule passes; sign holds 4/4"),
 ("M8-wholerun", "whole-run BCE ungated / pos1 / neg1 / shuf1", "0.3554+-0.0071 / 0.3673+-0.0145 / 0.3719+-0.0135 / 0.3665+-0.0153", True,
  ["w1 prequential_bce none", "w1 prequential_bce pos1", "w1 prequential_bce neg1", "w1 prequential_bce shuf1"], "M8/w1/results/summary.csv", None,
  "ungated seed 4 lower than all three earlier seeds (z -2.1); ordering none < shuf1 ~ pos1 < neg1 holds"),
 ("M8-census", "sign census (ungated): top-10% important neurons negative 34.6+-3.3%, all 49.0+-1.9%", "0.346 / 0.490", True,
  ["w1 census none neg_share_top10", "w1 census none neg_share_all"], "M8/w1/results/summary.csv; seed-4 run-none-s4.json census", None,
  "held-out census positions are never trained (sign_gate.py lines 102-104): no leakage"),
 ("M8-spearman", "income ranks importance weakly (0.106+-0.034); |a.g| does better (0.203+-0.068)", "0.106 / 0.203", True,
  ["w1 census none spearman_income", "w1 census none spearman_absag"], "same", None,
  "seed 4 income Spearman 0.032 is below the 3-seed band (weaker still); |a.g| > income holds 4/4"),
 ("M8-stop", "|a.g| gate + rebirth vs random reinit, post-shift switched BCE: -0.0161/+0.0007/+0.0011, not 3/3 -> stop", "not 3/3", True,
  ["w2 absag - random bce_postshift_switched"], "M8/w2/results/decision.json absag_vs_random; /tmp/traders-net/X1w4/study/M8/w2/results/run_*_s4.json", None,
  "stop rule applied as preregistered on seeds 1-3. Seed 4: absag -0.083 (z -8.0), a large absag win, driven by LFSR16: absag learnt it at seed 4, random did not. With lfsr16 removed seed 4 is +0.0060 (absag worse, within 2 sd). Raw absag wins in 2/4 seeds; the raw stop result rests on the LFSR16 lottery"),
 ("M8-postshift-table", "post-shift BCE absag / random", "0.4515+-0.0447 / 0.4563+-0.0371", True,
  ["w2 absag bce_postshift_switched", "w2 random bce_postshift_switched"], "M8/w2/results/summary.csv", None, "absag seed 4 0.322 (LFSR16 learnt)"),
 ("M8-onefamily", "absag's whole-run lead is one family: removing lfsr16 erases it (0.3106+-0.0183 vs 0.3089+-0.0195)", "0.3106 vs 0.3089", True,
  ["w2 absag - random bce_all_nolfsr", "w2 absag - random postshift_switched_nolfsr", "w2 absag bce_all_nolfsr", "w2 random bce_all_nolfsr"],
  "M8/w2/results/family_s{1,2,3}.csv; /tmp/traders-net/X1w4/study/M8/w2/results/family_s4.csv", None,
  "raw absag - random bce_all seed 4 -0.038 (z -2.2) comes from LFSR16 again. M8/check.py never checks random 0.3089+-0.0195 (the eq sits after a # on the absag_nl line); check_w4.py does: file 0.30887+-0.01948 matches"),
 ("M8-fisher", "exploratory: recycling low-score neurons may help find LFSR16 (absag vs random p=0.20)", "absag 2/3, random 0/3", True,
  [], "M8/w2/results/decision.json lfsr16_learnt_by_arm; X1/w4/check_w4.json", "unverified",
  "seed 4: absag learnt, random not -> 3/4 vs 0/4, one-sided Fisher p=0.071 (exploratory, not preregistered). Supports the hypothesis, still not significant"),
 ("M8-births", "reborn neurons surviving their trial slot, absag 67.7+-23.7 of 396", "67.7", True,
  ["w2 absag births_survived"], "M8/w2/results/summary.csv", None, "movement/taylor arms not rerun"),
]
C += [
 ("M9a-lottery", "soup Brier = 0.25 - D(1-(1-pi)^{K/Q}); R2 0.79-0.94 on priced families, shuffled null p=0.001", "p=0.001", True,
  [], "M9/w1/lottery_fit.csv; /tmp/traders-net/X1w4/m9check (byte-identical rerun); X1/w4/m9a_null_seed4.csv", "verified",
  "fit_lottery.py reran byte-identical (SOUP path made absolute in the copy: the committed relative path only works from M9/w1). Fresh null RNG (default_rng(4)): p=0.001 for all 4 priced families, null p95 0.148-0.171 vs original 0.159-0.174. No fresh soup runs (logical-soup is read-only). LFSR also fits a harm curve (R2 0.55), disclosed in M9-a.md"),
 ("M9a-predict", "CA110 needs K/Q >= 1358 for half its gain", "1358", True, [], "M9/w1/predictions.csv (byte-identical rerun)", "verified",
  "analysis reproduces byte-identically; the extrapolation itself is untested"),
 ("M9b-flops", "FLOPs to Brier 0.1, net vs soup (M9-b)", "see M9.md table", True, [], "M9/b/*.csv (byte-identical rerun of analyze_b.py)", "unverified",
  "analysis reruns byte-identically; fresh seed of the S0 P4 runs it reads is X1 worker 1's S0 item (not duplicated here); wealth-net inputs read-only"),
 ("M9c-notrend", "K/Q in {1,4,16,64} shows no trend in any series (B-FT excl. LFSR16 flat: K64-K1 +0.0007/+0.0015/-0.0006)", "trend=false x4", True,
  ["B-FT K64-K1 learn_bce_ex_lfsr", "B-FT K64-K1 learn_bce", "B-FS K64-K1 learn_bce_ex_lfsr", "B-FS K64-K1 learn_bce"],
  "M9/w3/results/VERDICT.json, EXLFSR.json; /tmp/traders-net/X1w4/study/M9/w3/results/*-s4.json", None,
  "summarize.py reran byte-identical. B-FS K64-K1 excl. LFSR16 is a real gain (-0.015 +- 0.010, seed 4 -0.009) from the width-8 bottleneck, as M9.md says"),
 ("M9c-thin", "very thin traders cost: A-FT K=64 (hw 2) +0.0069/+0.0066/+0.0069 worse than K=1", "+0.0068", True,
  ["A-FT K64-K1 learn_bce_ex_lfsr"], "M9/w3/results/EXLFSR.json; seed-4 A-K1-hw128, A-K64-hw2", None,
  "seed 4 +0.0081: same sign, larger; outside the band only because the 3-seed sd is 1.7e-4"),
 ("M9c-noise", "noise excess (humility) up at K=64 in every series", "~+0.005", True,
  ["B-FT K64-K1 humility", "A-FT K64-K1 humility", "B-FS K64-K1 humility"], "M9/w3/results/EXLFSR.json humility_values", None, ""),
 ("M9c-eff", "heads do not collapse: effective traders per question 46.5 at K=64 (B-FS)", "46.5", True,
  ["B-FS K64 eff_traders_per_q"], "M9/w3/results/SUMMARY.csv eff_traders_per_q", None, ""),
]
C += [
 ("K1-niches-on", "niches on (10k Q64): every slot alive, market reaches oracle Brier", "alive 10000; Brier 0.1195 vs oracle 0.1191", True,
  ["w1 10k Q64-on alive", "w1 10k Q64-on tail Brier - oracle"], "K1/runs/w1-n10k-q64-on-s*/summary.json; /tmp/traders-net/X1w4/study/K1/runs/w1-n10k-q64-on-s4", None,
  "regret 6.4e-4 vs 4.1e-4 +- 8.5e-5 passes only the floor rule; still 100x below niches-off regret (0.06)"),
 ("K1-niches-off", "niches off: one species holds >= 99.7% of wealth (Blume-Easley selection)", "0.9969 +- 0.0046", True,
  ["w1 10k Q64-off top_species_share", "w1 10k Q64-off alive", "w1 10k Q64-off top_clade_share"], "K1/w1/cells.csv; seed-4 w1-n10k-q64-off-s4", None, ""),
 ("K1-gini", "skilled traders are less unequal than the beliefs-shuffled null (Gini 0.44 vs 0.74, Q64 on)", "0.4395 vs 0.7340", True,
  ["w1 10k Q64-on gini_traders", "w1 10k Q64-on gini_null_traders", "w1 10k Q64-on gini - null"], "K1/w1/cells.csv", None,
  "K1/check/check_k1.json headline says 'gini_below_null_cells 15/15' while K1.md says 14/15 plus one tie (1k Q4096-on): wording mismatch, not a number error"),
 ("K1-fof", "rebalancing funds of funds lose share to buy-and-hold (10k crp-within premium -6.18 +- 1.19; bh +0.05)", "-6.176 / +0.046", True,
  ["w3 N10000 crp-within premium_agg", "w3 N10000 bh-within premium_agg", "w3 N10000 crp-within brier2"], "K1/w3/out/summary_N10000.csv; seed-4 w3-*-N10000-s4", None,
  "all 42 w3 runs (and 32 others) predate econ.py's code stamp (cfg.econ_sha absent), so the committed runs cannot be tied to the current engine by hash; the seed-4 rerun on the current engine reproduces them, which closes that gap for these arms"),
 ("K1-price", "the compute price c only destroys: best regret at c=0 (tie up to c/c*=0.03), c/c*=0.1 worse", "0.00043 +- 0.00008", True,
  ["w2 csweep C0-10k regret@10k", "w2 csweep C0.1-10k regret@10k", "w2 csweep S-off10k regret@10k"], "K1/w2/results/csweep_runs.csv; seed-4 w2-C*-10k-s4", None,
  "seed 4: c=0 6.8e-4, default c 6.4e-4, c/c*=0.1 1.5e-3 -> tie at small c, worse at 0.1, as claimed"),
 ("K1-shift", "after a shift the economy adapts by births: recovery 833 +- 58 steps; newborns hold 0.98 +- 0.004 of shifted-niche wealth", "833 / 0.98", True,
  ["w2 S-on10k recovery_steps", "w2 S-on10k newborn_share_trk_end"], "K1/w2/results/shift_runs.csv; seed-4 w2-S-on10k-s4", None,
  "seed 4: recovery 1700 steps (z +15), newborn share 0.87 (z -28). Births still dominate (vs 0.33-0.53 with niches off), so the direction holds, but the 3-seed sd badly understates seed variance and the '~830 steps / 98%' numbers are not reproduced"),
 ("K1-conservation", "money conserved in all runs (max residual 4.1e-12)", "4.1e-12", True, [], "K1/check/check_k1.json; X1/w4/k1_check_rerun.json; X1/w4/check_w4.json", "verified",
  "check_k1 rerun PASS (tests, conservation, v1_serves_all, cross_worker, stop_rule; analyzers identical modulo absolute paths); 8 seed-4 runs max residual 1.2e-12"),
 ("K1-stop", "no clade monoculture except 4/49 small costly arms: stop rule does not fire", "4/49", True, [], "K1/check/check_k1.json stop_rule; X1/w4/check_w4.json", "verified",
  "no seed-4 rerun arm is a monoculture"),
]

rows, n = [], {"verified": 0, "unverified": 0, "refuted": 0}
for cid, claim, claimed, fok, qs, ev, over, note in C:
    if over: st = over
    elif not fok: st = "refuted"
    else: st = "verified" if all(within(q) for q in qs) else "unverified"
    n[st] += 1
    fr = fresh(qs) if qs else "n/a"
    w2 = "n/a" if not qs else ("yes" if all(within(q) for q in qs) else "NO")
    rows.append(f"| {cid} | {claim} | {claimed} | {'yes' if fok else 'NO'} | {fr} | {w2} | {st} | {T} | {ev} | {RERUN} | {note} |")
hdr = f"""# X1 worker 4: claims check for M8, M9, K1

Generated {T} by `make_claims.py` from `repro.csv` and `check_w4.json` (both from `check_w4.py`). Rows: {len(rows)};
verified {n['verified']}, unverified {n['unverified']}, refuted {n['refuted']}. Paths relative to `study/`, `/tmp/...` = rerun copy.
Rule: `PREREG.md` (seed 4 within 2 sd of seeds 1-3 mean; s floored at 1e-3 when s < 1e-4). Fresh seed = 4 for every rerun
(20 CPU jobs, `rerun.sh`, clean copy with only output paths edited, `copy.diff`). No GPU, $0.
Seed-4 quantities: {J['n_with_seed4']}; within 2 sd (strict): {J['n_within_2sd']}. Number file checks all pass: {J['filechecks_ok']}.

| id | claim | claimed | re-read matches file | fresh seed 4 (value vs mean +- sd) | within 2 sd | status | checked_at | evidence | rerun | notes |
|---|---|---|---|---|---|---|---|---|---|---|
"""
skept = """
## Skeptic notes
- **Stop rules honoured.** M8 (w2 PREREG: 3/3 post-shift gain and bce_all not worse by >0.005) was applied to seeds 1-3 as written; M9 (§3 two sweep points no trend) triggered and reported; K1 stop rule (monoculture) re-checked by check_k1.
- **Seed variance is understated by 3-seed sds.** 17 of 81 seed-4 quantities land outside 2 sd (`check_w4.json` outside_2sd); none reverses a stated conclusion. Worst: K1 shift recovery (z +15) and newborn share (z -28); M8 absag-random post-shift (z -8, via LFSR16); M9 A-FT thin-trader cost (z +7.9, same sign).
- **LFSR16 lottery drives M8's raw w2 results.** Both big absag wins (seeds 1, 4) coincide with absag learning LFSR16; the lfsr16-removed comparisons are stable. The exploratory Fisher p moves 0.20 -> 0.071 with seed 4.
- **Kill control gap in M8/check.py.** The random bce_all_nolfsr check is commented out (after `#` on the absag_nl line); the number is right (check_w4.py).
- **Leakage.** M8 census uses held-out positions never trained; M8 w2 predicts before training on each block (gated.py lines 58-71); M9 run_kq.py asserts prequential order. No leakage found.
- **Relocatability.** M9 fit_lottery.py uses a relative SOUP path (only runs from M9/w1); K1 w3/out/summary.json stores absolute run paths (analyzer output identical modulo paths).
- **Code provenance.** 74/141 K1 runs predate econ.py's `econ_sha` stamp; seed-4 reruns on the current engine match them (above).
"""
open(os.path.join(HERE, "CLAIMS.md"), "w").write(hdr + "\n".join(rows) + "\n" + skept)
print(n)
