# E-SUD round 2 pre-registration (written 2026-10-15 ~04:25Z, before any round-2 chain or score exists)

Source: gemini's follow-up list `/home/agent/workspaces/mats/outbox/overnight-followups-round-1.md`, Rank 1 (E-SUD-R2),
relayed by opus-mats (am #3627). Eval only on the saved round-1b checkpoints (150k steps, seeds 0-2; models in
`vmout/r1b-a/model_s6_seed0.pt`, `vmout/esud-f/model_s6_seed{1,2}.pt`). No training. Budget group overnight-sud,
cap $5 for this round. Same data (val 2,000 and test 2,000 6x6 boards) and the same code otherwise.

## Why chains are re-run at tau = 1

Round 1b saved only per-board scores, not the raw chain outputs (completions, log q, the model's own check) that a
new null-search likelihood needs. So `sud.py dump` re-runs the round-1b chains (P1 rand, P2 voi, P3 conf; K = 16;
same generator seed 1000 + seed and the same call order as round 1b's eval) and saves them. Reproduction check
(reported, not a rule): the old unfloored null-search and witness log losses at M = 8 recomputed from the dump vs
the round-1b `eval_s6_seed*.json` values (GPU sampling may differ slightly).

## Changes under test

1. Recall-floored null search. P(null | Solvable = 1) = (1 - eta_max) + eta_max (1 - C_M)^theta, so
   logit P(solvable | no witness) = logit p0 + log[(1 - eta_max) + eta_max (1 - C_M)^theta].
   Count control: (1 - eta_max) + eta_max exp(-kappa M). A witness (completion with own check >= 0.5) gives
   max(p0, best check) as before. C_M = summed q_sigma(z) of the distinct refuted completions among the first M.
   Fitting: per seed, rule and tau, (theta, eta_max) and (kappa, eta_max) minimise mean validation Solvable log loss
   pooled over M in {1, 2, 4, 8, 16}. Grids: theta in exp(linspace(ln 0.01, ln 1000, 101)); kappa in {0} and
   exp(linspace(ln 0.001, ln 10, 61)); eta_max in {0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1}.
   Test is never used for fitting.
2. Noisy proposal. Chains for P1 (rand) and P3 (conf) re-run on val and test with proposal temperature tau_prop = 2.0:
   p = softmax(logits / 2) is used for cell selection (P3 entropy) and for sampling, and log q_sigma(z) is the log
   probability of the sampled values under that tempered proposal. The check pass is unchanged.

## Decision rules (statistic as in round 1: per-seed relative cut 1 - LL_arm / LL_base on the subset, mean over 3
seeds, seed-level t 95% CI with df 2; hierarchical bootstrap reported alongside; PASS = mean >= threshold and
seed-t lower bound > 0)

- R2-D1a (tail-safe null search, tau = 1 chains, M = 8): for EACH of P1, P2, P3, the floored null search cuts
  Solvable log loss by >= 20% vs P0 on solvable boards, >= 20% vs P0 on unsolvable boards, and >= 20% vs
  witness-only on unsolvable boards. PASS only if all 9 comparisons pass.
- R2-D1b (mass vs count under a noisy proposal, tau = 2 chains, M = 4): floored mass update vs floored count
  control, Solvable log loss on all test boards: cut >= 5% with seed-t lower bound > 0. Reported separately for P1
  and P3; R2-D1b PASS only if both pass.
Also reported (no verdict): the same comparisons at every M; the unfloored arms; ECE; fitted parameters; the
fraction of boards with a witness at each M and tau.

A failed rule is a result. No refitting after looking at test.
