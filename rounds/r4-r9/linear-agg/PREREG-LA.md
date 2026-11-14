# linear-agg pre-registration: linear-transformer aggregation of rollouts (gemini's Experiments A and B)

Written 2026-10-16 ~13:25Z by an Opus 5.5 subagent of opus-mats, before any training or evaluation run. Ask: Kelvin
2026-10-16, "gemini has two more experiments, let us run both, but have a subagent do it." Spec: gemini (Argon),
`/home/agent/workspaces/mats/outbox/gemini-linear-transformer-candidates-2026-10-15.md`, Sec. 3 (Experiments A and B,
"What we check"). Mechanism: Vladymyrov et al. 2024, arXiv 2402.14180 (linear transformers implement preconditioned GD
with momentum and in-context noise rescaling; Full / Diag / GD++ forms, AdaRR baseline).
Rules: outcomes only in training and pool fitting (Experiment A: the noiseless query target; Experiment B: the realised
query cell); the exact posterior (B) and the closed-form oracle (A) are used for scoring only; nothing hand-coded about
the domain in any aggregator; every check reported PASS/FAIL with numbers; no tuning after test; 3 seeds with paired 95%
t-intervals (2 df) across seeds unless stated. Changes after this file is hashed go to `PREREG-LA-AMENDMENTS.md` with
their time, never into this file. Before this file: generator design scans (generator only; no model was trained or
evaluated on them) and 300-1,000-step timing runs of train_a.py on training batches and its validation set (no test data).

## Experiment A: synthetic correlated-subspace rollouts + active VoI probes (`synth.py`, `lt.py`, `train_a.py`)

- **Generator** (`synth.py`, fixed): hidden w in R^16, 12 core chain variables with prior N(0, 1) and 4 noisy-TV variables
  with prior N(0, 9). K = 20 rollouts observe v_r = <k_r, w> + noise: 12 hub rollouts (4 near-duplicate unit keys around
  each of 3 hub directions over blocks 0-3, 4-7, 8-11, perturbation 0.35 inside the block), 6 links (e_i - e_j)/sqrt 2 on
  (1,2), (2,3), (3,4), (7,8), (8,9), (9,10), 2 noisy-TV rollouts (e_12 + e_13)/sqrt 2, (e_14 + e_15)/sqrt 2. Known depth
  m_r ~ U{1..6}; unknown episode noise sigma ~ U(0.1, 3.0); each rollout a bad branch w.p. 0.2 (noise variance x 9,
  unknown); noise variance sigma^2 m_r (x 9 if bad). Query k_q = (e_0 + e_11)/sqrt 2, target y_q = <k_q, w>. Median
  condition number of the core Gram 53 (unit keys; p10 24, p90 166). 8 probe candidates: 2 missing bridge links
  (e_0 - e_1, e_10 - e_11), 3 fresh hub copies (one per hub), 3 noisy-TV keys (e_12, e_14, (e_13 - e_15)/sqrt 2); a probe is
  a clean depth-1 observation.
- **Generator facts found before training (they shape check 3):** under the exact posterior, the expected-VoI top-2 is the
  bridge pair in only 42 / 52 / 41 / 26 / 20% of episodes at sigma = 0.1 / 0.5 / 1 / 2 / 3 (at high noise a fresh hub copy,
  whose raw cosine with the query is higher, is worth more), though the bridge pair is worth 0.98-0.85 of the best pair;
  raw cosine <k_q, k_z>^2 picks both bridges in 16%; uncertainty sampling (max predictive variance) picks noisy TV 100%.
  gemini's literal "selects the exact missing bridge probes" therefore disagrees with exact VoI here; check 3 is scored
  against exact VoI, and the literal bridge rate is reported for every selector.
- **Tokens:** every regression-type arm (linear transformers, AdaRR, oracle) uses depth-whitened tokens
  (x_r, y_r) = (k_r, v_r) / sqrt(m_r) (known relative noise scale; standard GLS weighting); for the transformers y is
  divided by a fixed YSCALE = 3 (std of the whitened values over training-distribution episodes, 2.97) and the
  prediction multiplied back. Query-like tokens (the query and the 8 candidates) have y = 0 and are never attended to.
- **Arms (learned):** linear transformers of 2402.14180, single head, paper eq. 2 / 10 / 11, readout -y of the query:
  **GD++** (2 scalars per layer), **Diag** (4 scalars per layer, omega_xx, omega_xy, omega_yx, omega_yy), **Full**
  (P, Q in R^{17x17} per layer, 578 parameters per layer), each at **L in {1, 2, 4, 6}**, seeds 0-2 (36 models).
  (gemini's parameter counts, 32 / 120 / 17k, do not match any single-head form of the paper; ours: GD++ 2L, Diag 4L,
  Full 578L.) Training (`train_a.py`): fresh episodes every step (sigma ~ U(0.1, 3)), squared error on y_q, Adam,
  lr 1e-3 (L <= 2), 5e-4 (L = 4), 3e-4 (L = 6), 500-step warmup then cosine decay to 0 over 20,000 steps, batch 512,
  gradient-norm clip 1, float64, init N(0, 1e-3^2); a batch with a non-finite loss is skipped; after 50 skips, or a
  validation MSE > 10 or non-finite, restart at lr / 3 (at most 2 restarts). Model selection: best validation MSE (fixed
  20,000-episode validation set, seed 7,000,001, every 500 steps).
- **Baselines:** 1-step scalar pools fitted by least squares on y_q over 100,000 training-distribution episodes per seed
  (seed 8,000,000 + seed), with per-rollout claims c_r = <k_q, k_r> v_r (unit keys, raw values) and precision proxy
  p_r = 1/m_r: **mix** y = u mean_r c_r; **sw_mix** y = u sum_r p_r^eta c_r / (lam + sum_r p_r^eta) (eta in {0, .5, 1, 2},
  lam in {0} U logspace(-3, 3, 25), u closed form); **dd** (1-step Gram dedup, as E-CA) y = u sum_r p_r c_r / crowd_r,
  crowd_r = sum_b max(0, <k_r, k_b>). Closed form: **AdaRR** (paper: OLS by pseudo-inverse on the whitened tokens,
  sigma^2 = RSS / (n - rank), ridge (X'X + sigma^2 I)^-1 X'y); **oracle** = exact posterior mean with the true prior and
  true per-rollout noise variances (reference; "adjusted MSE" = MSE - oracle MSE); prior mean 0 (MSE 1).
- **Test sets** (numpy generator): 20,000 episodes at each sigma in {0.1, 0.5, 1, 2, 3} (seeds 9,000,000 + i) and a pooled
  set of 50,000 with sigma ~ U(0.1, 3) (seed 9,100,000). Mechanism diagnostics use the first 5,000 pooled episodes.
- **A1 (gemini check 1).** (a) Each of Diag-4, Diag-6, Full-4, Full-6 has lower pooled test MSE than each of mix,
  sw_mix, dd and GD++ at the same L: 3-seed mean, paired 95% t-interval of MSE(baseline) - MSE(arm) above 0. (b) At
  L = 6, Diag-6 or Full-6 is within 5% of AdaRR or better (3-seed mean MSE <= 1.05 x AdaRR) at every one of the 5 noise
  levels. **PASS iff (a) for all four arms and (b).**
- **A2 (gemini check 2).** (a) For Diag-4 and Diag-6, every seed: median over episodes of kappa(Sigma^L) / kappa(Sigma^0)
  <= 0.1, kappa = largest / smallest eigenvalue of the context x-Gram Sigma^l = sum_j x_j^l x_j^l' restricted to the range
  of Sigma^0. (b) Every seed of Diag-4 and Diag-6 has at least one layer with omega_yy < 0 and one with omega_xy < 0.
  (c) Diag-6 downweights corrupted rollouts: influence ratio = mean |d yhat / d v_r| over bad-branch rollouts / the same
  over clean rollouts (pooled over episode-rollout pairs) <= 0.8, 3-seed mean, with the t-interval's upper end < 1.
  **PASS iff (a), (b) and (c).** Reported for reference: the same ratio for GD++-6, Full-6, AdaRR and the oracle.
- **A3 (gemini check 3).** One-pass score s_z = <x_q^L, x_z^L>^2 of the query and candidate tokens' x-parts after the last
  layer; the model "picks" its top 2. (a) Diag-6's VoI efficiency (exact value of the picked pair, i.e. the exact
  reduction of Var(y_q) from adding both probes, / that of the best of the 28 pairs, mean over the pooled test set)
  >= 0.90 (3-seed mean) and above the raw-cosine selector's efficiency (t-interval of the difference above 0). (b)
  Noisy-TV candidates are <= 5% of Diag-6's picks. **PASS iff (a) and (b).** Reported: efficiency, noisy-TV share and the
  both-bridges rate for all models and L and for the exact-VoI (marginal) top-2, raw cosine, uncertainty sampling and random
  selectors, pooled and per noise level.

## Experiment B: vector rollout tokens on the saved E-CA round-3 checkpoints (`b_rollouts.py`, `b_agg.py`)

- **Backbones:** the saved round-3 rule-110, 4-layer checkpoints, seeds 0-2 (`results/raw/r3/r110-L4-s{0,1,2}/model.pt`,
  md5 d7a45518 / f4c0b3fa / eb3d299d; backbone, stiffness head and v2 key head frozen; $0 backbone training). Data and
  query families exactly as round 3 (W = 12, T = 10, eps = 0.015, disjoint row-0 split). B1: rows 0 and 9 known, query in
  rows 4-5; B2: row 0 cols 0-5, row 9 and columns 0-1 known, query rows 4-5, cols 3-10.
- **Instances:** aggregator training: B1 and B2 queries on the 4,000 cal grids (family masks with new seeds 55,001 /
  55,002); early stopping: the 2,000 round-3 validation instances per family; test: the first 500 round-3 test instances
  per family (the instances behind round 3's published test numbers).
- **Rollouts** (gemini's spec): K = 16 per instance at temperatures 0.5, 1.0, 1.5, 2.0 (4 each), random order (round 3's
  best bridge selector, the one behind the cited pool numbers), M = 16 materialised cells, each value sampled from
  sigmoid(logit q / tau). Token z_r = (k_r, v_r, c_r): **k_r** (16-d, not gemini's 12-d: the checkpoint's own v2 key, the
  "attention-source fingerprint" of `ecanet.py`, of Y at S_M over the materialised cells); **v_r** = log-odds shift of
  Y vs the one pass, plus the shifts of the 3 hidden cells with the highest last-layer attention from Y at E0 (a
  materialised one is read leave-one-out); **c_r** = (beta_int(Y | S_M), conflict, trajectory log-probability, tau). The
  conflict score is the domain-free form of gemini's "boundary conflict": mean over evidence cells e of -log q(x_e | S_M
  minus e) (leave-one-out), not a hand-picked row 9. Query token: (k_q = Y's v2 key at E0, beta_int(Y | E0), conflict at
  E0, 0, 0; v = 0). A constant-1 coordinate is appended to every token.
- **Aggregators.** **LT-4** (primary, gemini's): Full linear transformer, 4 heads, 4 layers, D = 25 (20,000 parameters;
  gemini estimated 12.8k for 20-d tokens); logit mu = logit P0 + (readout of the query's Y slot). Ablations: **LT-1**
  (1 layer) and **LT-4 without the conflict feature**. Inputs standardised with cal-set statistics (v only scaled),
  clipped to [-5, 5]. Training: outcome log loss on cal instances (B1 + B2 pooled), Adam lr 1e-3, batch 256, up to 300
  epochs, early stopping on validation outcome log loss (patience 30 epochs), float64, init N(0, 1e-3^2), gradient clip 1,
  same non-finite / restart rule as A. One aggregator per backbone seed.
- **Scalar pools on the same rollouts** (round-3 definitions, `ecarun.py`): **mix** (mean of q_r(Y)), **sw_mix** (anchored
  at P0, weights beta_int^eta, (lam, eta) fitted on cal + val outcomes), **dd** (1-step Gram dedup with the v2 keys,
  anchored at the network's no-evidence prior, u fitted on cal + val outcomes). **Published round-3 values** (rand, M = 16,
  3-seed test means): B1 mix 0.2527, sw 0.2276, dd 0.2768; B2 mix 0.1982, sw 0.1981, dd 0.2231. Also a **replication** of
  round 3's protocol (rand chains, tau 1, K 16, M 16, F/B1/B2, pools fitted on the first 500 validation instances, test on
  the first 500) as a pipeline check, and P0.
- **Metric:** mean KL(P* || mu) over the 500 test instances per family (P* = exact posterior, scoring only).
- **B-main (gemini's question).** On each of B1 and B2: (i) LT-4's 3-seed mean test KL is below the best of mix, sw_mix
  and dd on the same rollouts (the one with the lowest 3-seed mean), with the paired t-interval of KL(best pool) - KL(LT-4)
  above 0; and (ii) below the published round-3 best pool (B1 0.2276, B2 0.1981). **PASS iff (i) and (ii) on both.**
- **B-a (mechanism a: rejecting conflicting rollouts).** Within each test instance, Spearman between the rollouts'
  conflict score and their influence |d out / d v_r[Y]| on LT-4's output; mean over instances (B1 and B2 pooled).
  **PASS iff the 3-seed mean <= -0.10 with the t-interval below 0.** The no-conflict ablation's KL is reported.
- **B-b (mechanism b: stitching across layers).** **PASS iff LT-4 beats LT-1 on B1 and on B2** (paired t-interval of
  KL(LT-1) - KL(LT-4) above 0 on each).

## Compute, cost and outputs

- GPU: eurisko's logical-soup infra, budget group linear-agg (cap $10), run ids la-*, spot L4 (on-demand if spot is
  unavailable or preempted), GPU_STARTUP=minimal. One VM for A's 36 trainings, one for B's rollouts; B's aggregators and all
  evaluation on CPU here (<= 4 cores). Expected < $2 at the $0.87/h ceiling. Ledger: `GPU-LOG.md` in this directory.
  Every VM deleted with gpu-down.sh --only NAME --no-sync and checked with gcloud.
- Analysis scripts are written before results are read; every number in RESULTS-LA.md is re-derivable from the saved
  arrays (`runs/`), and an independent audit section re-derives the headline numbers.
