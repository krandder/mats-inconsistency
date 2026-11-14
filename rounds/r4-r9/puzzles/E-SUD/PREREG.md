# E-SUD pre-registration (frozen before any model is trained; sha256 in PREREG.sha256)

Design: `/home/agent/workspaces/mats/outbox/overnight-experiments-design-2026-10-15.md` as amended by design v2
(`overnight-experiments-design-v2-2026-10-15.md`), which adopts gemini's review in full. Authorised by Kelvin
2026-10-15 00:3xZ. Budget group `overnight-sud`, hard cap $50, GCP L4 only (MATS cluster full tonight).

## Standing rules and how the code meets them

- Training uses outcomes only. Targets: one completion drawn uniformly from each solvable board's completions, and the
  solvable bit. Training files (`data/s*_train.npz`) contain no exact marginals. Early stopping uses validation outcome
  loss only. The one-parameter nuisance terms (lambda, theta, kappa) are fitted on validation outcomes only.
- Nothing hand-coded about the domain in the model or the arms. The network sees cell tokens with learned position ids
  and a homogeneous Solvable token; it is not told rows, columns, boxes, N or the rule. No rule checker at test time:
  a "witness" is a completed board that the model's own Solvable token accepts (q >= 0.5). The Sudoku solver is used
  only to generate boards, to draw the training outcome and to compute exact marginals for scoring.
- Every arm runs on real-world questions: it needs only a model that answers q(variable | visible variables) for any
  subset, including the query variable (Solvable / "a plan exists").

## Data (generated once, seeds fixed; 6x6 primary, 4x4 smoke only)

- 6x6 (2x3 boxes): 10-20 givens from a random grid. Solvable boards have 2-400 completions (ambiguous on purpose, so
  marginals are real probabilities). Unsolvable boards: same generator and clue count, 1 or 2 givens changed so that
  there is no direct peer clash and zero completions (clue-count matched; the contradiction needs search).
  Train 200,000 boards, validation 2,000, test 2,000. About 42% solvable.
- 4x4: 3-8 givens, same construction; train 100,000; a 2-minute smoke only (288 grids, memorisable).
- Exact marginals: mean one-hot over all completions found by a bitmask DFS. Cross-checked before launch
  (`results/check_exact.json`): 4x4 against filtering the full 288-grid table (2,000 boards, max |diff| 3e-8); 6x6
  against an independent row-permutation enumerator on 300 test boards (max |diff| 3e-8); 0 unsolvable boards with a
  completion; every stored 6x6 outcome is a valid grid that agrees with the givens.

## Model and training

One pre-LN transformer, d = 128, 4 layers, 4 heads, 37 tokens on 6x6 (36 cells + Solvable). Any-order masked loss:
each state reveals k ~ Uniform{0..#empty} extra cells (from the drawn completion, or from the pre-corruption grid for
unsolvable boards, which keeps them unsolvable and makes the clash local when enough is revealed). On solvable boards
the Solvable token is revealed (= 1) half the time, otherwise masked with target 1; on unsolvable boards it is masked
with target 0 and there are no cell targets. AdamW, lr 1e-3 cosine, batch 512, 30,000 steps, best checkpoint by
validation outcome loss. 3 seeds (0, 1, 2) share the data; test boards are paired across seeds.

## Inference arms (frozen network)

- P0 direct: q(cell | board, Solvable = 1) and q(Solvable = 1 | board), one pass each.
- Chains: M chains per board (M in {1, 2, 4, 8, 16}, nested prefixes of 16), run under the hypothesis Solvable = 1,
  i.e. proposal q(z | board, X = 1). Each step grows S_t = S_{t-1} + {z_t} (sequential materialisation, review item 1);
  the value of z_t is sampled from q(z_t | S_{t-1}); a chain ends when the board is full. Selection rules:
  - P1 random order (= P4-Rand);
  - P2 stiff-gated VoI: among the 6 lowest-entropy masked cells, maximise sum_d q(z=d|S) sum_Y ||q(Y|S,z=d) - q(Y|S)||^2
    over the other masked cells;
  - P3 confidence-first (lowest entropy = stiffest arc; = P2-Stiff).
  Stiffness here is the model's own predicted entropy; E-SUD has no separate beta head.
- Cell estimate from M chains (primary, "checked"): (lambda p0 + sum_r w_r rb_r) / (lambda + sum_r w_r), rb_r(Y) = the
  conditional Y was sampled from in chain r, w_r = the model's own q(Solvable = 1 | board + completion r). Secondary:
  w_r = 1 ("nochk"). lambda per (rule, M) fitted on validation outcomes.
- Solvable from M chains: found = some completion with check >= 0.5; C_M = summed proposal mass q_sigma(z_r) of the
  distinct refuted completions.
  - witness-only (ignore null search): found ? max(p0, best check) : p0.
  - null search (full loop): found ? same : sigmoid(logit p0 + log(1 - eta(C_M))), eta(C) = 1 - (1 - C)^theta, so
    log(1 - eta) = theta log(1 - C). theta (one parameter per rule) fitted on validation outcomes pooled over M.
  - count control: found ? same : sigmoid(logit p0 - kappa M) (does refuted MASS matter beyond the refuted COUNT?).

## Metrics

- KL(exact || estimate) over the empty cells of solvable test boards, per arm, against M. Outcome log loss of the
  same estimates.
- Solvable log loss on solvable and unsolvable test boards separately and pooled, and ECE (10 bins), against M.
- Chain diagnostics (scorer-only): share of completions that are truly valid; accuracy, false-positive and
  false-negative rates of the model's own check.

## Decision rules (6x6; per v2)

- D0 gate, before the sweep (seed 0): KL(exact || P0) >= 0.05 nats averaged over test cells, while KL <= 0.02 on
  one-step cells (naked singles given the board, identified by the scorer only). If KL(P0) < 0.01, cut the network to
  2 layers before the sweep and record it. Recorded either way; a KL between 0.01 and 0.05 is recorded as "small gap"
  and the sweep proceeds unchanged.
- D4 (primary, rule P2 declared now; P1 and P3 reported alongside). At M = 8 against M = 0:
  (a) null search cuts Solvable log loss by >= 20% vs P0 on solvable boards AND on unsolvable boards (each
  separately), and (b) null search beats witness-only by >= 20% on unsolvable boards.
  Statistic: per seed, relative cut = 1 - LL_arm / LL_baseline over the subset; mean over 3 seeds with a seed-level
  t 95% CI (t = 4.303, df 2) and a hierarchical bootstrap (seeds, then boards; 10,000 draws) as a second interval.
  PASS if the mean cut >= 20% and the seed-t lower bound > 0. "Strong pass" if the lower bound is also >= 20%.
  Guard (reported, not a rule): pooled log loss of null search <= witness-only at M = 8, so a pass cannot come from
  shifting every board down.
- Secondary (descriptive, no verdict word): cell KL at M = 16 for P2 vs P0 (the v2 D2 bar of 30% is quoted for
  comparison), and P2 vs P1 at equal M.

A failed rule is a result. No tuning after looking at test. Any deviation is listed in RESULTS.md.
