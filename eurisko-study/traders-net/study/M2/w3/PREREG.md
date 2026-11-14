# M2 worker 3 prereg: metrics (Spearman, additivity gap, pruning frontier)

Written 2026-09-28 ~02:00Z before any run. Item 3 of M2 (STUDY-PLAN.md §2 "M2", §3 row M2).
Inputs: worker 1's oracle (single-unit zero/mean ablations, coalitions) and worker 2's predictor
tables. Substrate: wealth-net checkpoints 8k / 20k / 30k / final (31.1k), seeds 1-3, read-only.

## Arms (predictors scored; whatever worker 2 delivers, at minimum)
books v1 (wealth-net cumulative wealth), books v2 (M1 books_v2: tangent / lmsr / meanbase),
accumulated |a.g|, movement, ||w_out||, gradient norm, IG / conductance, sampled Shapley
(heads and readouts), live grad x input and exact leave-one-out settlement if provided.
Controls: random (5 orders), magnitude (||w_out||), Taylor and movement (the tie references).
Reference ceiling: the single-unit oracle order itself (measured on a different segment than
the frontier, so it is a fair arm, not a tautology).

Sign convention: every score is oriented so that higher = more important (removing it hurts
more). Signed book wealth is scored as-is (a negative-wealth unit predicted "harmful"); |score|
variants reported separately where a predictor's sign is not meaningful.

## Metrics
1. Spearman(score, true dloss) per granularity (neuron / head / readout), per checkpoint, per
   seed, zero and mean ablation; reported as mean +- sd over 3 seeds. Plus top-10% recall of the
   oracle top 5% (neurons only).
2. Coalition additivity gap: for each coalition S, gap(S) = dloss(S) - sum_{i in S} dloss(i);
   report median |gap|/|dloss(S)| and Spearman(sum of singles, dloss(S)) (the additive
   ceiling); per predictor, Spearman(sum_{i in S} score_i, dloss(S)) over the 200 random and,
   separately, the 50 structured coalitions.
3. Pruning frontier: zero-ablate the lowest-importance k units by each predictor,
   fractions {0, 1, 2, 5, 10, 20, 40, 60}% (heads: every k in 0..8); held-out BCE on an
   independent segment, bit positions T+4096 .. T+8191 (T=24000), disjoint from the oracle
   segment T .. T+4095 used by wealth-net ablate.py and (expected) worker 1. Summary: area
   under the dBCE curve over fractions 0-40% (trapezoid), lower = better.

## Seeds and stop rule
3 seeds (the wealth-net seeds 1-3), mean +- sd. Stop rule (STUDY-PLAN §3): tie of the best
book predictor with Taylor or movement within seed sd at every granularity -> report the tie and
the pruning work stops. Operational: at the final checkpoint and zero ablation,
|mean_book - mean_ref| <= max(sd_book, sd_ref) for ref in {Taylor, movement}, at each of neuron /
head / readout. Also report whether any book beats the best of Taylor/movement by more than
that margin (a "win") or loses by more (a "loss"). Gate B's "coalition win": a book whose
coalition Spearman beats Taylor's by more than seed sd in 3/3 seeds on random coalitions.

## Compute
CPU only, at most 3 jobs x 1-2 threads, torch.set_num_threads(1-2), load checked with uptime.
Bulky intermediates in /tmp/traders-net/M2/w3; only CSV/JSON/PNG under study/M2/w3.

## Amendment 1 (2026-09-28 ~02:05Z, before any frontier run)
A full forward on 4096 positions takes ~8 s on one thread at box load ~40, so the frontier is
cut to: checkpoints 20k and final only (Spearman and additivity metrics keep all 4), eval
positions T+4096 .. T+6143 (2048, still disjoint from the oracle segment), and identical
pruned sets are evaluated once (memo). Nothing else changes.

## Amendment 2 (2026-09-28 ~02:10Z, frontier restarted after 1 min, no results read)
Worker 2's `_val` predictors use positions T+4096 .. T+6143. The frontier eval segment moves to
T+6144 .. T+8191 (2048 positions) so it is disjoint from the oracle (T..T+4095), from every
predictor's data, and from training.
