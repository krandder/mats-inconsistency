# E-SUD round 3 pre-registration (written ~09:50Z 2026-10-15, before any chain runs on the hard split)

Spec: gemini, `/home/agent/workspaces/mats/outbox/overnight-followups-round-3.md`, Rank 1 (E-SUD-R3), relayed by
opus-mats (am #3643, #3650). Kelvin (09:2xZ): "Please run new experiments with gemini; we didn't like the previous
results." Zero retraining: checkpoints `r2models/model_s6_seed{0,1,2}.pt` (the round-1b 150k-step nets). Budget
group overnight-sud, round-3 cap $5 (opus-mats #3650; the earlier $15 is superseded).

## Hard split (generated before this file; no model was run on it)

`code/r3hard.py` makes `data/r3/hard/s6_hard_{val,test}.npz`, 1,000 boards each, exactly 50/50, givens k in
[10, 15]; generator seeds 501 (val) and 502 (test); no board appears in both.
- Unsolvable boards: a random grid with k givens, then 1-2 givens changed. Each must have no direct peer clash and
  zero completions, and `prop_depth >= 3` (the spec's helper, verbatim). Test depths: 3 = 392, 4 = 69, 5 = 8,
  999 = 31.
- Solvable boards: 2-400 completions and `dead_end_rate >= 0.25`. The rate comes from 10 random-MRV rollouts: pick a
  minimum-candidate cell (ties at random), fill a uniformly random legal digit, and count a dead end when some empty
  cell has no legal digit. Mean rate 0.39; median 9 completions.
- Known imbalance, recorded and not fixed: mean clue count is 11.7 for solvable and 12.2 for unsolvable boards.

## Code change (the spec's chains() fix)

`chains()` now also accumulates `logp_bel` = sum of log softmax(logits)[sampled digit], the untempered believed
log-mass, next to the tempered proposal log-mass `logq` = sum of log softmax(logits / tau)[sampled digit]. Sampling
and P3's entropy ranking use the tempered p, as in round 2. The check pass is untempered.
- C_M^bel = min(1 - 1e-9, sum over distinct refuted completions among the first M of exp(logp_bel)).
- C_M^prop is the same with logq; it is reported as a secondary only.
- `sud.py dump` saves `<rule>_lb`. `ESUD_SPLIT_PREFIX=hard_` makes it read `s6_hard_{val,test}.npz`.

## Runs

`sud.py dump` with K = 16 chains, val and test, seeds 0-2, for:
- tau = 1.0: P1 rand, P2 voi, P3 conf;
- tau = 1.5 and 2.0: P1 rand, P3 conf.

The mixed bag is formed in the scorer: chains 1..M alternate P1 and P3 at the same tau (P1 chain 1, P3 chain 1,
P1 chain 2, ...).

## Arms (scorer `code/r3.py`)

- P0 = q(Solvable = 1 | board) from one pass.
- witness-only: found ? max(p0, best check) : p0, where found means some completion has own check >= 0.5.
- null_floor: likelihood (1 - eta) + eta (1 - C_M^bel)^theta.
- null_floor_prop: the same likelihood with C_M^prop (secondary).
- count_floor: likelihood (1 - eta) + eta exp(-kappa M).

(theta, eta) and (kappa, eta) are fitted per seed and condition on s6_hard_val Solvable outcomes only, minimising
mean log loss pooled over M in {1, 2, 4, 8, 16}. The grids are the same as round 2: theta in
exp(linspace(ln 0.01, ln 1000, 101)); kappa in {0} and exp(linspace(ln 0.001, ln 10, 61)); eta in {0, 0.5, 0.8, 0.9,
0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1}. Log loss is clipped at 1e-6.

## Decision rules (gemini's, made operational here)

Statistic, as before: per-seed relative cut 1 - LL_arm / LL_base, mean over 3 seeds, seed-t 95% CI (t = 4.303).
"CI excluding 0" means a lower bound > 0.

- **R3-D1a** (one-pass failure gate): the 3-seed mean P0 log loss on unsolvable test boards is >= 0.50 nats.
- **R3-D1b** (tau = 1, M = 8): for EACH of P1, P2 and P3, null_floor cuts unsolvable-board log loss by >= 50% vs P0
  and by >= 50% vs witness-only (lower bound > 0), and cuts solvable-board log loss by >= 20% vs P0 (lower bound > 0).
  PASS only if all three selectors pass.
- **R3-D1c** (believed mass vs count), per condition. There are 6 conditions: tau in {1.5, 2.0} x {P1, P3, mixed}.
  - Part 1: on test boards still without a witness at M = 1 and at M = 2, AUC(-C_M^bel; solvable vs unsolvable)
    has a 3-seed mean >= 0.80 at both M. The count M is constant across these boards, so its AUC is 0.5.
  - Part 2: null_floor total Solvable log loss is >= 10% lower than count_floor (mean cut >= 0.10, lower bound > 0)
    at M = 2 or at M = 4.
  - A condition passes if both parts pass. **R3-D1c PASS requires all 6 conditions to pass.** The number of passing
    conditions is also reported.

Also reported, with no verdict attached:
- per-chain witness recall on solvable boards for every condition;
- C_M^prop versions;
- the round-3 board-filter diagnostic (below);
- all curves over M.

## Diagnostic kept from the first round-3 step (opus-mats' superseded "20-60% recall" filter)

On an 1,800-board pool (k = 6-20, 50/50, seed 301), per-chain witness recall on solvable boards at tau = 1 is high at
every clue count: P1 rand 0.78 (k = 6) to 0.99 (k = 20), P3 conf at least 0.97 everywhere. The 20-60% range was
reached only by tempering the proposal: tau = 3 gives P1 0.53 and P3 0.60; tau = 4 gives about 0.2. The own check
was 100% accurate there. This is reported as a diagnostic only.

A failed rule is a result. Test is never used to fit anything.
