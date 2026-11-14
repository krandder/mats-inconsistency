# linear-agg: linear-transformer aggregation of search rollouts (gemini's Experiments A and B): results

MATS study, 2026-10-16 (12:27-14:25Z). Written by an Opus 5.5 subagent of opus-mats. Asked by Kelvin ("gemini has two
more experiments, let us run both, but have a subagent do it."); spec by gemini (Argon),
`/home/agent/workspaces/mats/outbox/gemini-linear-transformer-candidates-2026-10-15.md`, Sec. 3. Mechanism paper:
Vladymyrov et al. 2024, "Linear Transformers are Versatile In-Context Learners" (arXiv 2402.14180).
Pre-registration: `PREREG-LA.md` (saved 12:48:03Z, sha256 9abce49f...534ef4, before any training or evaluation run);
amendments with times and reasons: `PREREG-LA-AMENDMENTS.md` (training-recipe changes made before any test-set evaluation,
Sec. 7). Numbers: `runs/a-eval/summary.json`, `runs/b-eval/summary.json`, re-derivable from the saved arrays
(`runs/a-eval/arrays.npz`, `runs/b-eval/arrays-s*.npz`; Sec. 9).

## 1. TL;DR

- **Both experiments fail their main pre-registered checks.** No linear-transformer aggregator beat round 3's one-step pools
  on the saved E-CA bridge queries. In the synthetic test, a 4-layer model pools well, but none of gemini's three
  mechanism claims held as stated.
- **A (synthetic, 36 models).** A 4-layer Diag linear transformer (16 learned scalars) pools redundant, noisy rollouts far
  better than any one-step pool (test MSE 0.782 against 0.898-0.925; exact-posterior floor 0.699). It also beats the
  paper's closed-form AdaRR on the pooled noise mix (0.808) and at every noise level except the lowest. But the
  pre-registered 4- and 6-layer comparison fails (**A1 FAIL**):
  - The 6-layer models were unstable: every Diag-6 run diverged at all three learning rates tried, and Full-6 in two of
    three seeds.
  - Diag-6, Full-4 and Full-6 each have a seed whose predictions explode on 1-2 of 50,000 test episodes (rare extreme
    inputs), which wrecks the 3-seed mean MSE.
  - At the lowest noise the 6-layer models are 5-7x worse than AdaRR.
  
  **A2 FAILS:** no model compresses the condition number (kappa ratio 0.6-3.6, bar 0.1). The learned omega_yy and
  omega_xy do turn negative, but corrupted rollouts are not down-weighted: influence ratio 1.01, against 0.19 for the exact
  posterior and 0.89 for AdaRR. **A3 FAILS:** the 6-layer Diag key score picks probes worse than plain cosine (VoI
  efficiency 0.75 vs 0.90), though it never picks a noisy-TV probe. GD++ keys at 2-6 layers do beat cosine (0.95).
- **B (E-CA round 3, rule 110, 3 frozen checkpoints).** The 4-layer aggregator over 24-number rollout tokens scores test
  KL 0.230 (B1) and 0.216 (B2). That is not below round 3's anchored mixture (0.228, 0.198): **B-main FAILS**. A 1-layer
  aggregator does as well (0.226, 0.198): **B-b FAILS**. The aggregator does give less weight to rollouts whose sampled
  cells conflict with the evidence (Spearman -0.43; **B-a PASSES**), but dropping that feature costs no measurable
  accuracy. gemini's mixed temperatures also make the rollouts worse for every pool. The pipeline reproduces round 3's
  numbers.
- **Cost:** $1.40 of GPU at the list-price ceiling (4 VMs, all deleted; cap $10), plus CPU on this box. The training
  recipe was amended before any test evaluation (Sec. 7): the deep models' instability forced it.

## 2. Verdicts (gemini's "What we check", as operationalised in PREREG-LA.md; 3 seeds, paired 95% t-intervals)

| check | verdict | numbers |
|---|---|---|
| A1: L = 4-6 Diag / Full beat mix, sw_mix, dd and GD++ (a), and are within 5% of AdaRR at every noise level at L = 6 (b) | **FAIL** | (a) Diag-4 passes: 0.782 vs mix 0.925, sw_mix 0.898, dd 0.900, GD++-4 0.898 (all differences +0.116 to +0.142, CIs above 0). Diag-6, Full-4 and Full-6 fail: 3-seed means 2.6e31, 1.20 and 0.93, each driven by one seed that explodes on 1-2 test episodes. (b) MSE / AdaRR at sigma 0.1 / 0.5 / 1 / 2 / 3: Diag-6 6.6 / 1.36 / 1.01 / 0.96 / explodes; Full-6 4.9 / 1.21 / 1.01 / 0.96 / explodes |
| A2: omega_xx compresses kappa(Sigma^l) toward 1 (a); omega_yy, omega_xy < 0 (b) down-weight corrupted rollouts (c) | **FAIL** | (a) median kappa(Sigma^L) / kappa(Sigma^0) per seed: Diag-4 0.61, 0.73, 1.04; Diag-6 1.73, 2.63, 3.61 (bar <= 0.1). (b) passes: every Diag-4 and Diag-6 seed has a layer with omega_yy < 0 and one with omega_xy < 0. (c) Diag-6 influence ratio, corrupted / clean: 1.01 [0.91, 1.11] (bar <= 0.8) |
| A3: the one-pass key product <k_q^L, k_z^L> picks the missing bridge probes (a) and avoids noisy TV (b) | **FAIL** | (a) Diag-6 VoI efficiency 0.747 (bar 0.90), below raw cosine's 0.895 (difference -0.149 [-0.222, -0.075]). (b) passes: noisy-TV share 0.000. gemini's literal both-bridges rate: 0.016 (exact VoI itself: 0.336) |
| B-main: LT-4 beats every one-step scalar pool on B1 and B2 | **FAIL** | B1: LT-4 0.2299 vs sw_mix on the same rollouts 0.2407 (difference 0.011 [-0.002, 0.024]) and the published 0.2276. B2: 0.2161 vs 0.2045 (-0.012 [-0.092, 0.069]) and 0.1981 |
| B-a: the aggregator rejects rollouts whose cells clash with the evidence | **PASS** | Spearman(conflict, influence) -0.430 [-0.499, -0.360]. Without the conflict feature, KL changes by +0.006 [-0.013, 0.025] (B1) and +0.038 [-0.033, 0.109] (B2): no reliable cost |
| B-b: stitching across layers (LT-4 beats LT-1) | **FAIL** | LT-1 minus LT-4: B1 -0.004 [-0.015, 0.008], B2 -0.018 [-0.114, 0.078] |

## 3. What ran (setup as run)

**Experiment A (synthetic; `synth.py`, `lt.py`, `train_a.py`, `eval_a.py`).**
- Hidden w in R^16: 12 core chain variables (prior N(0, 1)) and 4 noisy-TV variables (prior N(0, 9)). Each episode has
  K = 20 rollouts v_r = <k_r, w> + noise: 12 hub rollouts (4 near-duplicate keys around each of 3 hub directions;
  median condition number of the core Gram 53), 6 two-variable links (e_i - e_j)/sqrt 2 along the chain, 2 noisy-TV
  rollouts. Known depth m_r ~ U{1..6}; unknown episode noise sigma ~ U(0.1, 3); 20% bad branches (9x noise variance,
  unknown). Query (e_0 + e_11)/sqrt 2 (the two chain ends). 8 probe candidates: 2 missing bridge links, 3 fresh hub
  copies, 3 noisy-TV keys.
- Arms: the paper's single-head linear transformers, GD++ (2 scalars per layer), Diag (4 scalars per layer) and Full
  (578 parameters per layer), at L = 1, 2, 4, 6, 3 seeds each (36 models), trained on fresh episodes for 20,000 steps of
  512 episodes (10.2M episodes per model; the paper used 200,000 x 2,048), squared error on the query, best validation
  checkpoint. Tokens are depth-whitened (k_r, v_r)/sqrt(m_r) for every regression-type arm.
- Baselines: three 1-step scalar pools fitted on 100,000 training episodes (mix, sw_mix, dd, with per-rollout claims
  <k_q, k_r> v_r), the paper's AdaRR (closed form), and the exact posterior mean (oracle, reference).
- Test: 20,000 episodes at each sigma in {0.1, 0.5, 1, 2, 3} and 50,000 pooled episodes (sigma ~ U(0.1, 3)).

**Experiment B (saved E-CA round-3 checkpoints; `b_rollouts.py`, `b_agg.py`).**
- Backbones: round 3's rule-110, 4-layer checkpoints, seeds 0-2 (frozen; $0 backbone training). Grids 12 x 10, eps 0.015,
  zero row-0 overlap between splits. B1: rows 0 and 9 known, query in rows 4-5; B2: row 0 cols 0-5, row 9 and columns 0-1
  known, query rows 4-5, cols 3-10.
- Rollouts: 16 per instance (temperatures 0.5, 1, 1.5, 2; 4 each), random order, 16 materialised cells. Each rollout
  becomes a 24-d token: the checkpoint's 16-d v2 key of the query at the end of the rollout; 4 log-odds shifts (the query
  and the 3 cells the query attends to most at the start); stiffness beta_int, a leave-one-out evidence-conflict score,
  the trajectory log-probability and the temperature; plus a constant 1. About 690 (B1) and 860 (B2) forward passes per
  instance; 33M passes in all, on one L4.
- Instances: aggregator training on 4,000 calibration grids per family (outcomes only), early stopping on round 3's 2,000
  validation instances per family, test on the first 500 round-3 test instances per family (the instances behind round 3's
  published numbers).
- Aggregators: LT-4 (Full, 4 heads, 4 layers, 20,000 parameters; primary), LT-1 (1 layer) and LT-4 without the conflict
  feature, one per backbone seed; the round-3 scalar pools (mix, sw_mix, dd) fitted on the same rollouts (calibration +
  validation outcomes); a replication of round 3's own protocol (tau 1 random chains); the one-pass belief P0.
- Metric: mean KL(P* || pooled belief) on the test instances; P* is the exact posterior (scoring only).

## 4. Experiment A: synthetic correlated-subspace rollouts

IMG: runs/a-eval/a-adjusted-mse.png (MSE above the oracle by noise level and depth: 3-seed median, seed range, off-scale
seeds marked; drawn by `plot_a.py`); runs/a-eval/a-train-curves.png (validation MSE during training)

| arm | params | pooled MSE | adjusted (MSE - oracle) | sigma 0.1 | sigma 0.5 | sigma 1.0 | sigma 2.0 | sigma 3.0 |
|---|---|---|---|---|---|---|---|---|
| oracle (exact posterior; reference) | - | 0.6988 | 0 | 0.0748 | 0.4018 | 0.6594 | 0.8714 | 0.9244 |
| AdaRR (closed form) | - | 0.8078 | +0.1090 | 0.1074 | 0.5417 | 0.8109 | 0.9778 | 0.9984 |
| prior mean 0 | - | 0.9932 | +0.2944 | 1.0054 | 0.9961 | 1.0086 | 1.0230 | 0.9982 |
| mix (1-step) | 1-3 | 0.9245 [0.9245, 0.9246] | +0.2258 | 0.8715 | 0.8686 | 0.8970 | 0.9640 | 1.0445 |
| sw_mix (1-step) | 1-3 | 0.8983 [0.8983, 0.8983] | +0.1996 | 0.8295 | 0.8282 | 0.8587 | 0.9422 | 1.0495 |
| dd (1-step Gram dedup) | 1-3 | 0.8997 [0.8997, 0.8998] | +0.2010 | 0.8315 | 0.8298 | 0.8612 | 0.9421 | 1.0498 |
| GD++ L=1 | 2 | 0.8983 [0.8983, 0.8984] | +0.1996 | 0.8284 | 0.8273 | 0.8581 | 0.9424 | 1.0518 |
| Diag L=1 | 4 | 0.8983 [0.8983, 0.8984] | +0.1996 | 0.8284 | 0.8273 | 0.8581 | 0.9424 | 1.0518 |
| Full L=1 | 578 | 0.8984 [0.8980, 0.8989] | +0.1997 | 0.8286 | 0.8275 | 0.8582 | 0.9423 | 1.0513 |
| GD++ L=2 | 4 | 0.8980 [0.8980, 0.8980] | +0.1992 | 0.8271 | 0.8259 | 0.8570 | 0.9414 | 1.0534 |
| Diag L=2 | 8 | 0.8342 [0.8216, 0.8469] | +0.1355 | 0.5232 | 0.5979 | 0.7725 | 0.9509 | 1.4541 |
| Full L=2 | 1156 | 0.8393 [0.8323, 0.8462] | +0.1405 | 0.5902 | 0.6413 | 0.7790 | 0.9500 | 1.1817 |
| GD++ L=4 | 8 | 0.8980 [0.8980, 0.8980] | +0.1992 | 0.8270 | 0.8259 | 0.8570 | 0.9414 | 1.0535 |
| Diag L=4 | 16 | 0.7824 [0.7820, 0.7829] | +0.0837 | 0.2848 | 0.5256 | 0.7770 | 0.9347 | 0.9624 |
| Full L=4 | 2312 | 1.1973 [-0.4467, 2.8412] | +0.4985 | 0.3382 | 0.5418 | 0.7831 | 0.9344 | explodes (1.7e+08) |
| GD++ L=6 | 12 | 0.8980 [0.8980, 0.8980] | +0.1992 | 0.8270 | 0.8259 | 0.8570 | 0.9414 | 1.0535 |
| Diag L=6 | 24 | explodes (2.6e+31) [CI not meaningful] | explodes (2.6e+31) | 0.7109 | 0.7378 | 0.8203 | 0.9388 | explodes (1.2e+82) |
| Full L=6 | 3468 | 0.9262 [0.5492, 1.3031] | +0.2274 | 0.5221 | 0.6570 | 0.8193 | 0.9396 | explodes (2.4e+07) |

Per-seed pooled MSE of the exploding arms, and a post-hoc description computed after the results were read (not a rule;
`runs/a-eval/posthoc-explosions.json`): episodes with a prediction beyond +-10 (the target has standard deviation 1), and
the MSE without them.

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Diag-6 | 0.888 (0 exploded) | 0.860 (0) | 7.9e31 (1 of 50,000; 0.815 without it) |
| Full-4 | 0.819 (2; 0.794 without) | 1.961 (1; 0.793) | 0.811 (2; 0.787) |
| Full-6 | 0.910 (0) | 0.783 (0) | 1.085 (1; 0.795) |

At sigma = 3, predictions beyond +-10 occur in 11-27 of 20,000 episodes for each Full-4 seed, 0-8 for Full-6, 0-14 for
Diag-6, and 6-17 for Diag-2 and Full-2 (there they stay moderate: MSE at most 1.6). Diag-4 has none at any noise level, and
no model has any at sigma = 0.1.

**What the trained models do.**
- **Depth 4 is the sweet spot here.** Diag-4 (16 scalars) is the best learned pool: 0.782 pooled, with all three seeds
  within 0.0005. It beats AdaRR at sigma >= 0.5 (0.526 vs 0.542, 0.777 vs 0.811, 0.935 vs 0.978, 0.962 vs 0.998) but
  not at sigma = 0.1 (0.285 vs 0.107, oracle 0.075). Inverting a Gram matrix with condition number ~90 in 4 layers is
  too few steps when the noise is low, which is the paper's pattern as well (its transformers need >= 4-5 layers to
  match ridge at low noise). Full-4 matches Diag-4 where it does not explode (0.787-0.794).
- **GD++ never uses its extra layers.** Its best validation MSE is 0.9023 at every depth, and its test MSE (0.898)
  equals the one-step pools'. Its x-updates do precondition (kappa ratio 0.68-0.79) but the prediction does not benefit.
- **Six layers are not trainable under this recipe.** Every Diag-6 run diverged at all three learning rates (3e-4,
  1e-4, 3.3e-5), within 500-5,000 steps. Full-6 seed 0 diverged at all three and seed 2 at the first. The selected
  checkpoints have validation MSE 0.82-0.89 (Diag-6) and 0.79-0.92 (Full-6), against 0.791 for Diag-4. The data have
  heavy tails: corrupted rollouts at sigma = 3 carry noise of standard deviation up to 22, and a degree-3^L polynomial in
  the tokens amplifies them. The paper trained 4-12 hours per model (200,000 steps of 2,048) and
  lowered the learning rate for deep models; this study had 20,000 steps of 512 (Sec. 7).

**Mechanism (A2).**
- **No condition-number compression.** The median kappa of the context Gram is 89 at the input. GD++ lowers it by a
  quarter to a third (ratio 0.68-0.79), Diag-4 by 0-40% depending on seed (0.61-1.04), and Full and Diag-6 raise it
  1.5-3.6-fold. The learned omega_xx are small and mixed in sign (|omega_xx| <= 0.23); none drives kappa toward 1.
- **Negative omega_yy and omega_xy appear, as the paper describes.** For example, Diag-4 seed 0 layer 3 has omega_xy
  -0.30 and omega_yy -0.13 (all scalars in `runs/report-tables.md`).
- **The noise adaptivity is global, not per rollout.** The influence of corrupted rollouts relative to clean ones is
  1.01 for Diag-6, 0.95 for Diag-4, 0.95-1.27 for Full-6, 1.007 for GD++ (linear in the values) and 0.89 for AdaRR,
  against 0.19 for the exact posterior, which knows which rollouts are corrupted. omega_yy shrinks every residual by the
  same factor; it does not single out the clashing ones.

**Probe choice (A3; pooled test set, 50,000 episodes).**

| selector | VoI efficiency (picked pair / best pair) | noisy-TV share | both bridges picked |
|---|---|---|---|
| exact VoI (oracle, per-probe) | 0.974 | 0.000 | 0.336 |
| raw cosine <k_q, k_z>^2 | 0.895 | 0.000 | 0.162 |
| uncertainty sampling | 0.000 | 1.000 | 0.000 |
| random | 0.391 | 0.374 | 0.035 |
| GD++ L = 2 / 4 / 6 keys | 0.952 / 0.950 / 0.950 | 0.000 | 0.30 / 0.26 / 0.26 |
| Diag L = 2 / 4 / 6 keys | 0.949 / 0.928 / 0.747 | 0.000 | 0.26 / 0.27 / 0.016 |
| Full L = 2 / 4 / 6 keys | 0.853 / 0.911 / 0.838 | 0.001 / 0.015 / 0.004 | 0.21 / 0.68 / 0.35 |

(3-seed means; per-seed values in `runs/report-tables.md`.) The preconditioned keys of the shallower and the GD++ models
choose probes better than raw cosine (0.95 vs 0.90). The pre-registered arm (Diag-6), whose training failed, does worse.
The Diag and GD++ scores never pick a noisy-TV probe and the Full scores rarely do (at most 4% of picks), whereas
uncertainty sampling always does. Full-4 picks gemini's two bridges most often (0.68) but is less efficient than GD++:
in two thirds of episodes exact VoI's own top two include a hub copy.

## 5. Experiment B: vector rollout tokens on the saved E-CA round-3 checkpoints

IMG: runs/b-eval/b-kl.png (bars: 3-seed means; dots: seeds; red dotted lines: round 3's published pools)

Test KL(P* || pooled belief) on round 3's 500 test instances per family (3-seed mean [95% t-CI]):

| arm | B1 | B2 |
|---|---|---|
| one pass P0 | 0.3013 [0.1752, 0.4273] | 0.2559 [0.2376, 0.2742] |
| mix (same rollouts) | 0.2735 [0.2198, 0.3273] | 0.2164 [0.2005, 0.2323] |
| sw_mix (same rollouts) | 0.2407 [0.1856, 0.2957] | 0.2045 [0.1695, 0.2396] |
| dd (same rollouts) | 0.2987 [0.2253, 0.3721] | 0.2297 [0.2015, 0.2580] |
| mix of the 4 tau=1 rollouts | 0.3213 [0.2694, 0.3733] | 0.2720 [0.2370, 0.3070] |
| LT-4 (primary) | 0.2299 [0.1875, 0.2723] | 0.2161 [0.1634, 0.2688] |
| LT-1 (1 layer) | 0.2262 [0.1813, 0.2711] | 0.1980 [0.1550, 0.2411] |
| LT-4 without the conflict feature | 0.2356 [0.1786, 0.2927] | 0.2544 [0.2097, 0.2991] |
| replication: one pass | 0.3013 [0.1752, 0.4273] | 0.2559 [0.2376, 0.2742] |
| replication: R3 mix (tau 1) | 0.2529 [0.1731, 0.3328] | 0.2031 [0.1934, 0.2128] |
| replication: R3 sw_mix | 0.2233 [0.1607, 0.2858] | 0.1981 [0.1754, 0.2208] |
| replication: R3 dd | 0.2687 [0.1750, 0.3624] | 0.2358 [0.2093, 0.2623] |
| published round 3 (rand, M = 16): mix / sw_mix / dd | 0.2527 / 0.2276 / 0.2768 | 0.1982 / 0.1981 / 0.2231 |

- **The pipeline reproduces round 3.** The one-pass KL is identical to round 3's (B1 0.3013, B2 0.2559), and round 3's
  own protocol rerun here (tau-1 random chains, pools fitted on 500 validation instances) gives B1 mix / sw_mix / dd
  0.253 / 0.223 / 0.269 against the published 0.253 / 0.228 / 0.277, and B2 0.203 / 0.198 / 0.236 against 0.198 / 0.198 /
  0.223: within the chains' sampling noise.
- **gemini's mixed temperatures make the rollouts themselves worse.** The same three pools on the 16 mixed-temperature
  rollouts score B1 0.274 / 0.241 / 0.299 and B2 0.216 / 0.205 / 0.230, each worse than on round 3's tau-1 chains; the
  4 tau-1 rollouts alone (mix) score 0.321 / 0.272 (only 4 chains).
- **B-main FAILS.** On B1, LT-4 (0.230) is below every pool on the same rollouts (best: sw_mix 0.241; difference 0.011,
  95% t-interval [-0.002, 0.024], includes 0) but not below the published round-3 sw_mix (0.228). On B2, LT-4 (0.216)
  is worse than sw_mix on the same rollouts (0.205) and than the published 0.198.
- **B-b FAILS: one layer does as well as four.** LT-1 scores B1 0.226 and B2 0.198 (LT-1 minus LT-4: -0.004
  [-0.015, 0.008] and -0.018 [-0.114, 0.078]); the point estimates favour the 1-layer aggregator. Nothing here needs
  multi-layer "stitching".
- **B-a PASSES, but it buys no measurable accuracy.** Within a test instance, LT-4's sensitivity to a rollout's query
  claim falls with that rollout's evidence-conflict score: Spearman -0.430 [-0.499, -0.360] (3-seed mean; per seed
  -0.442, -0.449, -0.397). Post hoc (computed after reading the results, not a rule): within rollouts of the same
  temperature the correlation is still -0.32, so it is not only the temperature (which correlates +0.34 with conflict).
  But the ablation without the conflict feature is not reliably worse: KL change +0.006 [-0.013, 0.025] on B1 and +0.038
  [-0.033, 0.109] on B2.
- Aggregator training: early stopping after 35-57 epochs (best epoch 4-26, counting from 0), 15-96 s each on one CPU
  thread; no non-finite batch in any run. Validation loss stopped improving early (8,000 training instances for 20,000
  parameters; training loss was not logged, so over- versus under-fitting is not separated). Fitted round-3 pools on these
  rollouts: sw_mix lambda 3.2-5.6, eta 0.5; dd u 0.019-0.021.

## 6. What this means

- **Part of the paper's effect carries over to rollout-like data.** On synthetic rollouts with redundant coverage
  (condition number ~90), an unknown noise level and 20% corrupted rollouts, a 4-layer, 16-parameter linear transformer
  trained only on outcomes learns a pool far better than any one-step pool, and better than closed-form adaptive ridge at
  all but the lowest noise. That supports gemini's first point (multi-step beats one-step), at depth 4.
- **The mechanisms gemini named are not what the trained models did.** They did not compress the condition number toward
  1. They did not single out corrupted rollouts: their noise handling is a global shrinkage, as in the paper, not
  per-rollout rejection. Their last-layer keys did not choose probes as well as exact VoI. Deeper models (6 layers) were
  unstable to train on these heavy-tailed inputs. Every trained Diag or Full model of 2 or more layers except Diag-4
  makes occasional outlier predictions on rare extreme episodes, and at 4-6 layers some are astronomically large: a
  practical risk for an aggregator.
- **On the real E-CA bridges, vector tokens plus a trained linear transformer do not beat round 3's best one-step pool.**
  One attention layer does as well as four, so nothing here shows multi-step "stitching" across rollouts. The bridge gap
  seems limited by the rollouts (what the network samples) more than by the pooling: adding gemini's off-temperature
  rollouts made every pool worse, and nothing beat round 3's anchored mixture of tau-1 chains (0.228 / 0.198); LT-1
  matched it (0.226 / 0.198).
- **Not tested here:** the paper's training budget (about 40x more episodes per model) or float64 training of the deep
  models; deeper models with a stabiliser (normalisation, input clipping); B with tau-1 rollouts only, or with more
  aggregator training instances (8,000 here for 20,000 parameters); bridge cells chosen by anything other than one-pass
  attention.

## 7. Deviations, amendments and infrastructure events

- **Training recipe amended before any test evaluation** (`PREREG-LA-AMENDMENTS.md`; only training and validation logs
  had been read when each change was made).
  - A-1 (13:07Z, Experiment A) and A-3 (13:09Z, Experiment B) dropped the pre-registered restart rules. A validation
    evaluation with a non-finite or worse loss is never selected, and non-finite batches are skipped. Reason: the rule
    "validation MSE > 10 -> restart at lr / 3" fired in all three Full-L6 runs at every lr tried (3e-4, 1e-4, 3.3e-5), near
    step 2,000, while the training loss stayed near 0.9. A few validation episodes make the degree-3^L polynomial explode.
  - A-2 (13:07Z): Experiment A trained in float32 instead of float64 (float64 ran at 46-87 ms per step on the L4).
    Evaluation is float64.
  - A-4 (13:49Z): A-1 had also dropped, by mistake, the pre-registered "more than 50 skipped batches -> restart at lr / 3"
    rule. In float32 a gradient overflow can turn the parameters into NaN, after which every batch is skipped: two
    Diag-L6 runs never had a finite checkpoint. The rule was reinstated. It binds only in the five runs that skipped more
    than 50 batches (Diag-L6 x 3, Full-L6 seeds 0 and 2), which were retrained; the other 31 runs skipped at most 3
    batches and are unchanged.
  - Every A model was trained under the final recipe; old-recipe runs were killed or discarded and re-run, and the
    superseded logs are in `runs/a-superseded/`. The five retrains ran on this box's CPU (1 thread each), because no
    on-demand L4 could be created (stockout in 11 zones, run la-a4). torch's CPU and CUDA generators draw different (equally
    distributed) training episodes.
- **Pre-registration header time.** PREREG-LA.md says "written ~13:25Z"; it was saved at 12:48:03Z and hashed at
  12:48:10Z, before any run (erratum in the amendments file).
- **Check 3 operationalisation** (in the pre-registration, before training): in this generator exact VoI does not always
  pick the two "missing bridge" probes gemini named (it does in 20-52% of episodes depending on noise), so A3 is scored
  against exact VoI (efficiency of the picked pair) plus noisy-TV avoidance; the literal both-bridges rate is reported.
- **Spec details not in gemini's text, fixed in the pre-registration:** the rollout key is the checkpoint's 16-d v2 key
  (gemini wrote 12-d); the "boundary conflict" is a leave-one-out evidence-conflict score over all evidence cells (a
  hand-picked bottom row would hard-code the domain); the rollout selector is round 3's random order; parameter counts
  differ from gemini's estimates (A: GD++ 2L, Diag 4L, Full 578L scalars; B: 20,000 for LT-4).
- **Infrastructure.** (1) On the A VM the first six Diag/GD++-L6 jobs crashed at start (torch 2.14's Triton launcher needs
  Python.h); python3-dev was installed and the jobs re-queued. (2) A third rental for the L6 jobs (la-a2, spot) was
  preempted by GCE 1 s after creation (13:19:12Z); the jobs ran on an on-demand L4 instead (la-a3).
- B's aggregators were trained on calibration grids with family masks drawn with new seeds (55,001 / 55,002); round 3 never
  used calibration grids for queries.

## 8. Cost and clean-up

- GPU: **$1.40** at the $0.87/h list-price ceiling (`GPU-LOG.md`, rebuilt by `gpu/ledger.py` from the shared
  logical-soup log; budget group linear-agg, cap $10). Four VMs: la-a (spot, A training, 0.80 h, $0.70), la-b (spot, B
  rollouts, 0.47 h, $0.41), la-a2 (spot, preempted 1 s after creation, $0.01), la-a3 (on-demand, A's L6 jobs, 0.32 h,
  $0.28); la-a4 never existed (stockout). Spot is normally 60-70% below the ceiling, so true spend is lower.
- CPU (this box, at most 4 cores): B's aggregators (9 trainings, 15-96 s each), five A retrains (Sec. 7), all evaluation.
- Live linear-agg VMs at the final check (gcloud, name~soup-gpu-la-): none. Every VM was deleted by the launcher's
  gpu-down (`--only NAME --no-sync`, driver mode) or by GCE (preemption).

## 9. Self-audit (re-derived from the saved arrays with separate code: `audit_la.py`, `runs/audit.json`, `runs/audit-b-raw.json`)

- **A, MSEs:** every MSE in `runs/a-eval/summary.json` (36 models x 6 test sets, plus the oracle, AdaRR, prior and pools)
  recomputed from the saved per-episode predictions: largest difference 0. The test episodes were regenerated from their
  seeds and match the saved targets.
- **A, model forward:** a NumPy re-implementation of the paper's Diag update (eq. 10), run from the omegas saved in
  `train.json`, reproduces Diag-6's saved predictions to 7.5e-16. Changing one omega by 1e-6 moves them by 4e-6, so the
  check is sensitive.
- **A2:** kappa(Sigma^L) / kappa(Sigma^0) from the NumPy forward on 2,000 episodes: 1.72, 2.61, 3.55 (summary, 5,000
  episodes: 1.73, 2.63, 3.61). Influence ratio by central finite differences on 500 episodes: 1.054, 1.039, 0.971 (summary,
  autograd on 5,000: 1.053, 1.011, 0.974). Both verdicts are unchanged.
- **A1 and A3:** the t-intervals recomputed with scipy match; Diag-6's VoI efficiency, re-read from the per-episode arrays,
  is identical (0.779, 0.739, 0.722). On 2,000 episodes its picks show a noisy-TV share of 0.000 and a both-bridges rate of
  0.031, 0.014 and 0.010.
- **B:** every KL recomputed from the saved pooled probabilities and exact posteriors: largest difference 0. P0, mix and
  sw_mix were rebuilt from the raw rollout arrays with a separate implementation, with the exact posteriors read straight
  from round 3's cached evaluation file: largest difference 0. The t-intervals (scipy) and the conflict-influence
  Spearman (scipy, -0.442, -0.449, -0.397) match.
- **Limits of this audit:** the same agent wrote it, with separate code; it re-uses the trained models and the rollouts (no
  retraining, no fresh rollouts), and it does not re-check the exact posteriors beyond confirming they are round 3's
  (which round 3's audit checked against dense and brute-force code).

## 10. Files (all under /home/agent/workspaces/mats/work/linear-agg/)

- Pre-registration and changes: `PREREG-LA.md`, `PREREG-LA.sha256`, `PREREG-LA-AMENDMENTS.md`.
- Code: `synth.py` (A generator, oracle, exact VoI), `lt.py` (Full / Diag / GD++ linear transformers), `train_a.py`,
  `eval_a.py`, `b_rollouts.py` (B rollouts and tokens), `b_agg.py` (B aggregators, pools, replication), `audit_la.py`,
  `report_tables.py`, `plot_a.py`; round-3 code copied unchanged into `eca/`; checkpoints copied into `ckpt/` (md5 as in round 3).
- A outputs: `runs/a/<kind>-L<L>-s<seed>/` (model, training history), `runs/a-eval/summary.json`, `arrays.npz`
  (per-episode predictions), `mse.csv`, `tables.md`, `a-adjusted-mse.png`, `a-train-curves.png`; superseded logs of the
  five retrained runs in `runs/a-superseded/`.
- B outputs: `runs/b/b-seed{0,1,2}.npz` (all rollout tokens; md5 in `runs/b/npz.md5`), `runs/b-eval/summary.json`,
  `seed*.json`, `arrays-s*.npz` (pooled probabilities, exact posteriors, LT-4 influences), `kl.csv`, `tables.md`,
  `b-kl.png`, `b-train-curves.png`, `train-*.csv`, `lt4-s*.pt`.
- Audit: `runs/audit.json`, `runs/audit-b-raw.json`. GPU ledger: `GPU-LOG.md`; launch scripts and logs in `gpu/`.
