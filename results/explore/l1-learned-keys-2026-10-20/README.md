Serves: aggregation that approaches full Bayes without being told who knows what

# L1: learned keys from provenance, scalar reports (EXPLORATORY)

**EXPLORATORY** (PROTOCOL 10.1, track E). Not a finding; dev split only, one training seed, no test split touched.
Lane L1 of stage 1, run by an opus-mats subagent on 2026-10-20 (CPU, 1 thread per process, at most 4 processes).

## Question

With scalar reports (each agent sends its log-odds, or in E1 its posterior mean and precision), can an
aggregator that **learns** keys from the provenance metadata (aggbench keys at visibility none / noisy / full)
reach the reports-level ceiling `oracle_clearing` without being given the true structure, in E1, E2 and E3
(sybil, biased, collude)?

## Arms (code: `explore/l1_learned_keys/arms.py`, runner `run.py`, tables `table.py`)

All arms see only `batch.view(vis)` and are trained on the realized truth (no distillation), Adam with one-cycle
lr 3e-3, batch 256, 2500 steps on 40k train episodes (main run). Harness used unmodified (`import aggbench`).

- `lk_clear` (learned-key clearing). E2/E3: phi_i = MLP(key_i) in R^32, G = Phi Phi^T, output
  l0 + diag(G)^T (G + eps I)^{-1} (r - l0), eps learned. With phi_ik = sqrt(2) mu_k c_ik this *is* oracle_clearing
  for honest agents, so the clearing solve is built in and only the keys are learned. E1: h_i = P_i m_i and
  J_i = P_i - I satisfy h_i = J_i theta + noise with Cov(h_i) = J_i and Cov(h_i, h_j) = G_i G_j^T; the arm learns
  G_i = chol(J_i + eps I) Q_i (Q_i = M_i L_i^{-T} with L_i L_i^T = I + M_i^T M_i and M_i = MLP(key_i, J_i),
  which keeps the covariance PSD) and returns prec = I + J^T Sigma^{-1} J, info = J^T Sigma^{-1} h. With the true G_i this reproduces
  bayes to 1e-13 (checked before the run on 50 train episodes; not stored in JSON).
- `lk_clear_cal` (E2/E3 only): `lk_clear` plus a per-agent value calibrator r'_i = (r_i - l0) e^{a_i} + b_i, with
  (a_i, b_i) from an MLP of agent i's own key and report features (meant for the biased agent, whose key looks honest).
- `attn_pool`: generic 2-layer set-attention (width 32) over tokens [key_i, report features], no solve. E2/E3:
  l0 + sum_i w_i (r_i - l0) + mean_i c_i; E1: prec = I + sum_i alpha_i J_i, info = sum_i beta_i h_i.
- Harness baselines: `oracle_clearing`, `extremized_log_pool` (alpha fitted on train), `log_pool`, `linear_pool`,
  `sum_logodds` (in the JSONs), and `bayes`.
- Controls for every learned arm through the identical `predict` (`aggbench.controls.run_with_controls`):
  `control:untrained_init`, `control:random_network`.

## What was measured

Mean regret to `bayes` (nats/episode) with 95% bootstrap CIs over 4000 dev episodes (dev seeds 10^7 .. 10^7+3999),
and paired regret differences against `oracle_clearing` and `extremized_log_pool` (`metrics.paired_difference`).
One JSON per (env, visibility): `result-<env>-<vis>.json`, written by `aggbench.results.write_result` (validated
against `result_schema.json`, brute-force oracle check on 24 episodes, harness `git_dirty=false`). Follow-ups at
visibility full (same code, `lk_clear`; the E2 long-training run also has `lk_clear_cal` and visibility noisy): `long-training/` (10000 steps on the same 40k episodes) and
`more-data/` (160k train episodes, 10000 steps; `more-data/E2-640k/`: 640k, 20000 steps; `more-data/E1-160k/`).
Logs in `logs/` of each directory. Tables below are printed by `python3 -m explore.l1_learned_keys.table <dir>`
from the JSONs.

## Result in one line

EXPLORATORY: at visibility **full**, learned-key clearing (`lk_clear`) approaches `oracle_clearing` in E2, E3-sybil
and E3-collude (gap 0.013 / 0.013 / 0.007 nats at 40k episodes, 0.003 at 640k in E2, CIs exclude 0) but not in E1
(0.35 nats left at 160k) or E3-biased (keys cannot reveal the distortion; the calibrator halves the gap); at
**noisy** it only matches `extremized_log_pool`, and at **none** it reduces by construction to the log pool.

## Main run: 40k train episodes, 2500 steps, 4000 dev episodes

Regret to bayes (nats/episode, 95% bootstrap CI over dev episodes):

| env | vis | oracle_clearing | extremized_log_pool | lk_clear | lk_clear_cal | attn_pool |
|---|---|---|---|---|---|---|
| E1 | none | -0.000 [-0.000, 0.000] | 0.819 [0.779, 0.858] | 0.822 [0.783, 0.861] | n/a | 0.836 [0.795, 0.876] |
| E1 | noisy | -0.000 [-0.000, 0.000] | 0.819 [0.779, 0.858] | 0.784 [0.743, 0.823] | n/a | 0.915 [0.872, 0.960] |
| E1 | full | -0.000 [-0.000, 0.000] | 0.819 [0.779, 0.858] | 0.670 [0.631, 0.709] | n/a | 0.901 [0.858, 0.942] |
| E2 | none | 0.019 [0.014, 0.025] | 0.043 [0.034, 0.051] | 0.104 [0.093, 0.114] | 0.042 [0.033, 0.051] | 0.040 [0.031, 0.048] |
| E2 | noisy | 0.019 [0.014, 0.025] | 0.043 [0.034, 0.051] | 0.045 [0.036, 0.055] | 0.044 [0.035, 0.054] | 0.108 [0.090, 0.126] |
| E2 | full | 0.019 [0.014, 0.025] | 0.043 [0.034, 0.051] | 0.032 [0.025, 0.039] | 0.033 [0.026, 0.040] | 0.108 [0.091, 0.127] |
| E3-sybil | none | 0.019 [0.014, 0.025] | 0.075 [0.064, 0.087] | 0.118 [0.107, 0.129] | 0.070 [0.059, 0.081] | 0.042 [0.033, 0.050] |
| E3-sybil | noisy | 0.019 [0.014, 0.025] | 0.075 [0.064, 0.087] | 0.054 [0.044, 0.064] | 0.053 [0.043, 0.063] | 0.104 [0.088, 0.121] |
| E3-sybil | full | 0.019 [0.014, 0.025] | 0.075 [0.064, 0.087] | 0.032 [0.025, 0.039] | 0.033 [0.026, 0.040] | 0.097 [0.081, 0.116] |
| E3-biased | none | 0.019 [0.014, 0.025] | 0.073 [0.061, 0.084] | 0.084 [0.074, 0.094] | 0.053 [0.043, 0.062] | 0.051 [0.042, 0.060] |
| E3-biased | noisy | 0.019 [0.014, 0.025] | 0.073 [0.061, 0.084] | 0.079 [0.067, 0.091] | 0.060 [0.050, 0.071] | 0.116 [0.100, 0.133] |
| E3-biased | full | 0.019 [0.014, 0.025] | 0.073 [0.061, 0.084] | 0.074 [0.062, 0.087] | 0.047 [0.038, 0.056] | 0.126 [0.108, 0.145] |
| E3-collude | none | 0.019 [0.014, 0.024] | 0.031 [0.024, 0.037] | 0.042 [0.035, 0.049] | 0.031 [0.024, 0.038] | 0.027 [0.020, 0.034] |
| E3-collude | noisy | 0.019 [0.014, 0.024] | 0.031 [0.024, 0.037] | 0.035 [0.027, 0.042] | 0.035 [0.027, 0.042] | 0.080 [0.065, 0.095] |
| E3-collude | full | 0.019 [0.014, 0.024] | 0.031 [0.024, 0.037] | 0.026 [0.019, 0.032] | 0.027 [0.021, 0.034] | 0.088 [0.071, 0.106] |

Paired regret difference to oracle_clearing (arm minus oracle_clearing; > 0 means worse than the reports-level ceiling):

| env | vis | extremized_log_pool | lk_clear | lk_clear_cal | attn_pool |
|---|---|---|---|---|---|
| E1 | none | 0.819 [0.779, 0.858] | 0.822 [0.783, 0.861] | n/a | 0.836 [0.795, 0.876] |
| E1 | noisy | 0.819 [0.779, 0.858] | 0.784 [0.743, 0.823] | n/a | 0.915 [0.872, 0.960] |
| E1 | full | 0.819 [0.779, 0.858] | 0.670 [0.631, 0.709] | n/a | 0.901 [0.858, 0.942] |
| E2 | none | 0.023 [0.016, 0.030] | 0.084 [0.074, 0.094] | 0.022 [0.015, 0.029] | 0.020 [0.014, 0.027] |
| E2 | noisy | 0.023 [0.016, 0.030] | 0.026 [0.018, 0.033] | 0.024 [0.017, 0.033] | 0.089 [0.072, 0.106] |
| E2 | full | 0.023 [0.016, 0.030] | 0.013 [0.008, 0.017] | 0.013 [0.009, 0.018] | 0.088 [0.072, 0.107] |
| E3-sybil | none | 0.056 [0.045, 0.066] | 0.099 [0.088, 0.109] | 0.050 [0.040, 0.060] | 0.022 [0.015, 0.029] |
| E3-sybil | noisy | 0.056 [0.045, 0.066] | 0.035 [0.026, 0.044] | 0.033 [0.024, 0.043] | 0.085 [0.070, 0.100] |
| E3-sybil | full | 0.056 [0.045, 0.066] | 0.013 [0.008, 0.017] | 0.014 [0.009, 0.019] | 0.078 [0.063, 0.096] |
| E3-biased | none | 0.053 [0.044, 0.063] | 0.065 [0.055, 0.074] | 0.033 [0.026, 0.041] | 0.032 [0.024, 0.039] |
| E3-biased | noisy | 0.053 [0.044, 0.063] | 0.060 [0.049, 0.071] | 0.041 [0.032, 0.049] | 0.097 [0.082, 0.113] |
| E3-biased | full | 0.053 [0.044, 0.063] | 0.055 [0.044, 0.066] | 0.027 [0.021, 0.034] | 0.107 [0.091, 0.124] |
| E3-collude | none | 0.012 [0.007, 0.017] | 0.023 [0.017, 0.029] | 0.012 [0.007, 0.017] | 0.008 [0.004, 0.013] |
| E3-collude | noisy | 0.012 [0.007, 0.017] | 0.016 [0.011, 0.022] | 0.016 [0.011, 0.021] | 0.061 [0.047, 0.076] |
| E3-collude | full | 0.012 [0.007, 0.017] | 0.007 [0.003, 0.010] | 0.008 [0.004, 0.012] | 0.069 [0.054, 0.086] |

Controls (regret to bayes, mean only): trained / untrained_init / random_network:

| env | vis | lk_clear | lk_clear_cal | attn_pool |
|---|---|---|---|---|
| E1 | none | 0.822 / 0.822 / 3.244 | n/a | 0.836 / 7.987 / 14.704 |
| E1 | noisy | 0.784 / 0.821 / 2.224 | n/a | 0.915 / 4.619 / 4.722 |
| E1 | full | 0.670 / 0.820 / 1.736 | n/a | 0.901 / 4.574 / 4.861 |
| E2 | none | 0.104 / 0.106 / 0.336 | 0.042 / 0.106 / 0.319 | 0.040 / 1.090 / 0.984 |
| E2 | noisy | 0.045 / 0.146 / 0.403 | 0.044 / 0.146 / 0.404 | 0.108 / 5.889 / 0.717 |
| E2 | full | 0.032 / 0.137 / 0.378 | 0.033 / 0.137 / 0.381 | 0.108 / 5.772 / 0.719 |
| E3-sybil | none | 0.118 / 0.120 / 0.312 | 0.070 / 0.120 / 0.305 | 0.042 / 1.578 / 1.669 |
| E3-sybil | noisy | 0.054 / 0.174 / 0.384 | 0.053 / 0.174 / 0.386 | 0.104 / 9.582 / 0.531 |
| E3-sybil | full | 0.032 / 0.155 / 0.359 | 0.033 / 0.155 / 0.361 | 0.097 / 9.284 / 0.808 |
| E3-biased | none | 0.084 / 0.086 / 0.316 | 0.053 / 0.086 / 0.269 | 0.051 / 1.896 / 1.413 |
| E3-biased | noisy | 0.079 / 0.191 / 0.404 | 0.060 / 0.191 / 0.402 | 0.116 / 8.175 / 0.772 |
| E3-biased | full | 0.074 / 0.183 / 0.396 | 0.047 / 0.183 / 0.392 | 0.126 / 8.003 / 0.924 |
| E3-collude | none | 0.042 / 0.043 / 0.285 | 0.031 / 0.043 / 0.291 | 0.027 / 1.489 / 1.561 |
| E3-collude | noisy | 0.035 / 0.075 / 0.383 | 0.035 / 0.075 / 0.379 | 0.080 / 9.825 / 0.710 |
| E3-collude | full | 0.026 / 0.071 / 0.342 | 0.027 / 0.071 / 0.350 | 0.088 / 9.801 / 0.670 |

n dev = 4000 per cell, n train = 40000; harness commit `b94f9fe50b`, git_dirty=False, code sha256 `9a78862b8a42`.

## Follow-ups at visibility full (lk_clear; does the gap close with more steps or more data?)

| env | n train | steps | lk_clear regret | minus oracle_clearing | minus extremized_log_pool | random-network control | dir |
|---|---|---|---|---|---|---|---|
| E1 | 40000 | 2500 | 0.670 [0.631, 0.709] | 0.670 [0.631, 0.709] | -0.149 [-0.171, -0.127] | 1.736 | . |
| E1 | 40000 | 10000 | 0.540 [0.500, 0.583] | 0.540 [0.500, 0.583] | -0.279 [-0.321, -0.236] | 5.134 | long-training |
| E1 | 160000 | 10000 | 0.354 [0.326, 0.380] | 0.354 [0.326, 0.380] | -0.465 [-0.493, -0.436] | 2.041 | more-data/E1-160k |
| E2 | 40000 | 2500 | 0.032 [0.025, 0.039] | 0.013 [0.008, 0.017] | -0.011 [-0.017, -0.004] | 0.378 | . |
| E2 | 40000 | 10000 | 0.071 [0.059, 0.084] | 0.052 [0.041, 0.063] | 0.029 [0.017, 0.041] | 0.110 | long-training |
| E2 | 160000 | 10000 | 0.025 [0.019, 0.031] | 0.005 [0.003, 0.008] | -0.018 [-0.025, -0.011] | 0.108 | more-data |
| E2 | 640000 | 20000 | 0.022 [0.017, 0.028] | 0.003 [0.001, 0.005] | -0.020 [-0.027, -0.014] | 0.058 | more-data/E2-640k |
| E3-biased | 40000 | 2500 | 0.074 [0.062, 0.087] | 0.055 [0.044, 0.066] | 0.002 [-0.005, 0.008] | 0.396 | . |
| E3-collude | 40000 | 2500 | 0.026 [0.019, 0.032] | 0.007 [0.003, 0.010] | -0.005 [-0.011, 0.000] | 0.342 | . |
| E3-collude | 40000 | 10000 | 0.060 [0.048, 0.072] | 0.041 [0.031, 0.052] | 0.029 [0.018, 0.041] | 0.050 | long-training |
| E3-sybil | 40000 | 2500 | 0.032 [0.025, 0.039] | 0.013 [0.008, 0.017] | -0.043 [-0.053, -0.033] | 0.359 | . |
| E3-sybil | 40000 | 10000 | 0.070 [0.058, 0.084] | 0.051 [0.040, 0.063] | -0.005 [-0.020, 0.011] | 0.100 | long-training |
| E3-sybil | 160000 | 10000 | 0.025 [0.019, 0.031] | 0.005 [0.002, 0.008] | -0.050 [-0.061, -0.040] | 0.087 | more-data |

## Reading (EXPLORATORY; one training seed, dev split, no hyperparameter search)

- **full.** In E2 and E3-sybil `lk_clear` beats `extremized_log_pool` (paired CIs exclude 0; in E3-collude the
  paired difference is -0.005 [-0.011, 0.000]), and in all three its paired gap to `oracle_clearing` is small and shrinks with training data (E2: 0.013 at 40k, 0.005 at 160k, 0.003 at 640k), so the
  remaining gap looks like sample size with truth labels rather than architecture. More steps on the same 40k
  episodes made it worse (train loss fell below the Bayes log loss: overfitting), so steps are not the lever, data is.
  Sybil copies and colluders are handled because identical keys give identical rows of Phi and the solve shares their
  weight; the arm was never told which agents are copies.
- **E3-biased.** Keys describe information, not honesty, so `lk_clear` (0.074) ends at the extremized level; the
  per-agent calibrator (`lk_clear_cal`, 0.047) closes about half of the gap from report magnitude alone.
- **E1.** The learned-covariance clearing beats `extremized_log_pool` at noisy (-0.035 [-0.052, -0.018]) and full and improves steadily with
  data (0.670 -> 0.540 -> 0.354), but stays far from `oracle_clearing` (= bayes in E1). It has to learn the
  nonlinear map key -> G_i = A_i^T R_i^{-1} B_i; the 160k run had not converged. Not closed.
- **noisy.** `lk_clear` is level with or slightly worse than `extremized_log_pool` in E2, E3-collude and E3-biased
  (paired +0.003 to +0.006) and beats it only in E3-sybil (-0.021 [-0.028, -0.014]); it does not approach
  `oracle_clearing`. With 15% bit flips the learned Gram matrix is not good enough to beat extremizing; an open
  candidate is a key model that denoises using the other agents' keys and reports.
- **none.** With no key, Phi is constant, so `lk_clear` is exactly the (shrunk) log pool and cannot extremize
  (trained = untrained_init by construction; no claim). The generic `attn_pool` does best here, below
  `extremized_log_pool` in E3-sybil (0.042 vs 0.075), presumably by spotting identical reports.
- **attn_pool degrades when keys are added** (E2: 0.040 at none, 0.108 at noisy and full). Not investigated
  (optimization with 24 extra input dims at 2500 steps is the obvious suspect); treat it as a broken arm at noisy/full,
  not as evidence about attention.

## Controls

Every learned arm ran through `aggbench.controls.run_with_controls` (same `predict`): `control:untrained_init`
and `control:random_network` (scale-matched random weights). In the main run every trained `lk_clear`,
`lk_clear_cal` and `attn_pool` cell at noisy/full is below both controls. Exceptions, which void the
corresponding claim: (a) at visibility none `lk_clear` trained equals untrained_init (structural, see above);
(b) in `long-training/` E3-collude the random-network control (0.050) beats the trained arm (0.060), and in
the long-training E2 and E3-sybil cells the random-network control (0.100-0.113) is close to extremized; those
long-training cells carry no claim beyond "more steps on 40k episodes hurt".

## Harness notes (things that were uncertain or broken)

- `results.harness_info()` is computed when the JSON is written, from the shared working tree, so
  `harness.git_commit` is the repository HEAD at write time (other lanes committed during this run) and
  `git_dirty`/`sha256` describe the files on disk, not the code the process imported. Another lane committed an
  additive E4 environment to `aggbench/` at 22:54Z while the main run (launched 22:48Z at 26c02b8, harness as of
  718627f) was in progress; one JSON written during that edit (E1-none) recorded `git_dirty=true`. E1-none was
  rerun (identical numbers) so all 24 JSONs here have `git_dirty=false`; their `git_commit` fields differ for this
  reason. The E4 diff touches no E1-E3 code path (baseline numbers are identical across runs). `validate()` does not
  reject `git_dirty=true`; it probably should, and harness_info should be captured at import time.
- Under the machine's load (~50 on 16 cores) torch with 4 threads was ~10x slower per step than 1 thread, so the
  run used 1 thread per process and up to 4 processes.

## Files

- `result-<env>-<vis>.json` (15), `long-training/result-*.json` (5), `more-data/result-*.json` (2),
  `more-data/E2-640k/result-E2-full.json`, `more-data/E1-160k/result-E1-full.json`; logs under each `logs/`.
- Code: `explore/l1_learned_keys/` (arms.py, run.py, table.py). The runs used commit 26c02b8 for arms.py and run.py
  (`extra.code_sha256` in each JSON); table.py was added afterwards and only reads the JSONs.
