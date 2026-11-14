You are gpt-6-astra, co-prover and red team, round 2, for the input-bits lane on random critical ReLU networks. Your round-1
answer is /home/agent/workspaces/mats/work/input-bits/coprover/astra_r1_answer.md. You may now read the whole lane directory
/home/agent/workspaces/mats/work/input-bits/ (read-only): FORMAL.md (theory v2, merging your round 1; v1 frozen before your
answer is coprover/FORMAL_v1_as_written.md), PREREG.md (Parts 1-3, hashes in prereg_hashes.txt), lib/ib.py and tests,
scratch/*.py (runs, scorers), runs/*.json (results), results/part1_scores.json (Part 1 scorecard), logs/. Single-threaded small
computations are fine. Be concrete and adversarial; say plainly what is wrong.

Tasks (number your answers T1..T6):
T1. Scorer audit. Check scratch/score_part1.py against PREREG Part 1 (P1.1-P1.20) rule by rule: any deviation from the
registered text (aggregation, bands, cells, layer indexing, the definition of "median" vs "per net"), any bug. Recompute two
or three scores yourself from runs/*.json. Then do the same for scratch/score_part2.py against Part 2 (H2.1-H2.6) and
scratch/score_part3.py against Part 3 (P3.1-P3.6), even if their runs are not finished (check the code).
T2. Estimator audit. lib/ib.py and scratch/part1a.py, part1b.py, part1c.py, part1chain.py, part2_noise.py, part2_input.py,
part2_mlmc.py, part3.py: are the estimators unbiased/consistent for what PREREG says they measure? In particular: the
U-statistics in var_split; univariate_shares and deg2_share (two halves); the Jansen total indices; the rotated first-order
share in part1b (the rotation is fitted on an independent pilot); the Gamma and sensitivity ratio in part1c (multi-reference
average, rescaling); the noise-regime window shares in part2_noise (noise regeneration, centring, the window definition
matching work/stationary/scratch/partB.py's RQMC coordinates); the f1 subtraction in part2_input (estimation noise of f1's
coefficients, the G_res_true correction); the multilevel pilot in part3 (same definitions as work/unrestricted/lib/mlmc.py)
and the quadratic control CVQ (is its mean exactly zero? is the pilot independent?).
T3. Theory follow-up. (a) The Part 1 data (results/part1_scores.json, runs/p1a_*, p1chain_*) show the chain's annealed V
falling with L at L >= n/2 although E V_amp is monotone (your round-1 bound): quantify the heavy tail of E_W V_amp at L ~ n
(e.g. the distribution of per-net V_amp, the moments of the chain's per-replica amplitude terms) and say which statistic
(mean, median, pooled ratio) a report should use. (b) Explain any registered miss in Part 1 you find substantive.
(c) Give the finite-width correction to the kernel direction share at L ~ n/4 to n/2 if you can.
T4. Synthesis design red team. Is G_pred = 1/(1 - S1 - (1 - rho_2(N, n)) W2_off) with rho_2 calibrated on exchangeable
quadratic forms (scratch/calib_rho2.py) a fair test of "gain set by the low-order share"? What would make it fail for
reasons unrelated to the question? Is the verdict rule H2.6 sound? Propose one additional measurement that would separate
"interaction order grows" from "dimension grows at fixed N" in the noise regime (cheap enough for CPU in under 30 minutes).
T5. Part 3 predictions. Independently of my registered bands, predict for each Part 3 method (MC, RBMC, RQMC, RQMC-RB,
RQMC-ROT, CVQ, RQMC-CVQ, ML2, MLD, ML2-RQMC) its gain over MC at 2^12 L mv and its exponent, at n = 64, 128, 256 and
L = 32, n/2, n, 2n. Which, if any, could give a gain that grows with width, and why?
T6. Anything else that would change a conclusion.

Format: markdown, numbered, each finding labelled (BUG, DEVIATION, CONCERN, OK, PROVEN, HEURISTIC), with file and line
references and the fix.
