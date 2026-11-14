Serves: treating search rollouts as information sources, so one aggregation mechanism covers compute and data

# L9: rollouts of an imperfect model as information sources (EXPLORATORY)

**Track: EXPLORATORY.** Stage 1, lane L9, 2026-10-20. Dev split only. No test split was used. Recipe v2 was
added after recipe v1 had been seen (see Arms).

**Result.** Rollouts behave like information gathering with correlated sources. Each extra rollout lowers
the exact ceiling the way an extra observer does, but the ceiling stops at a floor set by the model's
shared misreading: about 0.029 nats here, where an exact model goes to 0. With N = 24 rollouts the imperfect
model's reports ceiling (0.042) falls between those of 2 and 4 exact-model rollouts. A simple attention
aggregator closes a quarter to a third of the 0.018-nat gap between extremized pooling and the reports
ceiling (keys full, recipe v2). At N = 6 its paired CIs against extremized pooling include 0 for all three
seeds. In the N sweep (seed 0) the gain is significant at N = 2, 4, 12 and 24.

## Question

Kelvin asked how much search rollouts act like gathering information. The question here: if the sources
are rollouts of one imperfect model rather than independent observers, does the same aggregation machinery
apply, which ceilings bound it, and how much does more compute (more rollouts) buy?

## Design: environment E4 (`aggbench`, `make_env("E4")`)

The world is E2's: Y ~ Bernoulli(0.5) and K = 8 evidence pieces x_k ~ N(+-mu_k, 1), with
mu_k ~ U(0.1, 0.6) and LLR lam_k = 2 mu_k x_k. All 8 pieces are the context (the evidence). The sources are
N rollouts of one imperfect any-order predictor. Rollout i conditions on L_i ~ U{2..6} pieces in its own
random order and reports

    r_i = logit0 + b + sum_{t=1..L_i} [ a gamma^(L_i - t) (lam_k(t) + e_k(t)) + nu_it ]

| model error | default | what it does |
|---|---|---|
| calibration a | 1.5 | over-confident on every piece |
| recency gamma | 0.7 | forgets early pieces, so the answer depends on the order (an exact model is order-consistent) |
| misreading e_k ~ N(0, 0.3^2) | per episode | the same for every rollout that reads piece k: shared model error, so errors are correlated |
| bias b ~ N(0, 0.3^2) | per episode | shared by all rollouts |
| step noise nu_it ~ N(0, 0.3^2) | per step | private sampling noise |

Keys are what a search log would record: which pieces the rollout read, how many steps before its end it
read each one, and coverage x mu. The control is the exact model (a = 1, gamma = 1, no noise). Its rollouts
are E2-style overlapping observers.

## Ceilings and how they are verified

Everything is linear-Gaussian given Y, so every ceiling is exact:

- **bayes**: the posterior given all 8 pieces. Regret is measured against it.
- **reports ceiling** (`oracle_clearing`): the posterior given the N reports and the true error structure
  (orders, a, gamma, noise sds): l0 + 2 (W m)^T Sigma^+ (r - l0), with m_k = 2 mu_k^2 and
  Sigma = W diag(2m) W^T + s_e^2 W W^T + s_b^2 11^T + s_nu^2 diag(L).
- **trace ceiling**: the posterior given every rollout's whole trace (start value plus per-step increments).
  This is the ceiling for vector messages.
- **infinite-rollouts floor**: the limit of both ceilings as N grows. The rollouts then reveal
  z_k = lam_k + e_k and b exactly, but never e_k itself, so the floor is l0 + sum_k 2 m_k z_k / (2 m_k + s_e^2).
  Table 3 is consistent with it: the trace ceiling at N = 24 is 0.029 against a floor of 0.028. This floor is
  derived and checked numerically against Table 3, but it is not brute-forced.

Verification:

- `verify.brute_force_rollouts` runs in every result's `oracle_check`. It recomputes bayes by enumerating Y
  over the raw piece densities. It rebuilds the reports ceiling latent by latent along each rollout's order
  and evaluates both class densities in full (pseudo-determinant when the covariance is singular). Max
  difference is 2e-15 on 24 smoke episodes per condition.
- `tests/aggbench/test_rollouts.py` (49 aggbench tests pass):
  - 400k draws from the environment's own sampler, with the structure fixed: logistic regression recovers
    the weights of the reports ceiling and of the trace ceiling;
  - the exact-model control reduces to E2's `report_posterior_logit`, and its trace ceiling to Bayes on the
    union;
  - a mutated ceiling (no recency) fails the brute-force check;
  - order holds: bayes < trace < reports ceiling (paired CIs).

Harness commits: `99859d8` (E4) and `42e894a` (singular case). Every JSON has the same harness sha256
`e50881934b0f...` and `git_dirty: false`. The recorded `git_commit` differs between files only because
other lanes committed in between.

## Arms

- Baselines from `all_baselines`, fitted on train, without keys.
- Learned: a one-layer attention aggregator. Tokens are [report, key], and the output is
  l0 + sum_i (w_i (r_i - l0) + c_i), with (w_i, c_i) read from each token after attention.
- Training is on Y labels with no distillation. Each arm has 3 seeds at N = 6 and seed 0 in the N sweep.
- Every trained arm runs through `controls.run_with_controls`, which adds the untrained-init and
  random-network controls.
- Training recipes:
  - **v1** (fixed before launch): 50k train episodes, 3000 Adam steps.
  - **v2** (added after v1): v1's keyed arms memorized train, with train regret 0.000 against dev 0.20 in a
    local diagnostic. v2 uses 200k train episodes, AdamW with weight decay 1e-4, and early stopping on 10k
    held-out train-split episodes. Dev is only scored.

## Results (5000 dev episodes; mean regret to bayes in nats, 95% bootstrap CI)

### Table 1. N = 6 rollouts, recipe v2 (ceilings and baselines are the same as in v1)

| method | imperfect model | exact model (control) |
|---|---|---|
| infinite-rollouts floor | 0.029 [0.023, 0.036] | 0.000 [-0.000, 0.000] |
| trace ceiling | 0.044 [0.036, 0.053] | 0.004 [0.001, 0.006] |
| reports ceiling | 0.083 [0.072, 0.093] | 0.011 [0.007, 0.014] |
| extremized_log_pool | 0.101 [0.089, 0.112] | 0.034 [0.027, 0.041] |
| log_pool | 0.111 [0.100, 0.121] | 0.065 [0.057, 0.073] |
| linear_pool | 0.137 [0.126, 0.148] | 0.092 [0.082, 0.101] |
| sum_logodds | 0.553 [0.501, 0.608] | 0.317 [0.280, 0.354] |
| attention v2, keys none (3 seeds averaged) | 0.099 [0.088, 0.110] | 0.031 [0.025, 0.038] |
| attention v2, keys noisy | 0.099 [0.088, 0.111] | not run |
| attention v2, keys full | 0.096 [0.085, 0.107] | 0.026 [0.020, 0.031] |
| attention v1, keys none / noisy / full | 0.099 / 0.187 / 0.178 (overfit) | 0.031 / - / 0.104 (overfit) |
| controls v2, untrained init (range over seeds and keys) | 0.22 to 2.66 | 0.18 to 2.80 |
| controls v2, random network (range over seeds and keys) | 0.16 to 0.95 | 0.13 to 0.93 |

Every control is far worse than the trained arm, so no claim here is one that a control also passes.

### Table 2. Paired differences at N = 6, recipe v2 (negative = first arm better)

| condition | keys | attn - extremized_log_pool (seeds 0 / 1 / 2) | attn - reports ceiling (seed 0) |
|---|---|---|---|
| imperfect | none | -0.002 [-0.004, 0.000] / -0.002 [-0.004, 0.000] / -0.002 [-0.004, -0.000] | 0.016 [0.010, 0.022] |
| imperfect | noisy | -0.000 [-0.004, 0.003] / -0.003 [-0.005, 0.000] / -0.001 [-0.005, 0.002] | 0.018 [0.012, 0.024] |
| imperfect | full | -0.006 [-0.012, 0.001] / -0.004 [-0.009, 0.001] / -0.004 [-0.009, 0.002] | 0.012 [0.007, 0.017] |
| exact | none | -0.003 [-0.006, -0.000] / -0.003 [-0.006, -0.001] / -0.003 [-0.006, -0.000] | 0.021 [0.015, 0.026] |
| exact | full | -0.009 [-0.014, -0.004] / -0.008 [-0.013, -0.002] / -0.009 [-0.014, -0.003] | 0.014 [0.009, 0.019] |

Ceiling gaps:

- imperfect: reports ceiling - extremized = -0.018 [-0.024, -0.012]; trace - reports ceiling = -0.038 [-0.045, -0.032].
- exact: reports ceiling - extremized = -0.024 [-0.029, -0.018]; trace - reports ceiling = -0.007 [-0.009, -0.004].

### Table 3. More rollouts (keys full, attention v2 seed 0)

| model | N | reports ceiling | trace ceiling | infinite-rollouts floor | extremized_log_pool | attention v2 | context covered |
|---|---|---|---|---|---|---|---|
| imperfect | 1 | 0.224 [0.210, 0.238] | 0.204 [0.191, 0.218] | 0.029 [0.023, 0.036] | 0.227 [0.212, 0.241] | 0.226 [0.211, 0.240] | 0.0% |
| imperfect | 2 | 0.163 [0.149, 0.177] | 0.134 [0.121, 0.146] | 0.035 [0.028, 0.041] | 0.172 [0.157, 0.185] | 0.168 [0.153, 0.181] | 8.4% |
| imperfect | 4 | 0.092 [0.080, 0.103] | 0.065 [0.056, 0.075] | 0.030 [0.023, 0.037] | 0.112 [0.099, 0.125] | 0.104 [0.091, 0.116] | 58.9% |
| imperfect | 6 | 0.083 [0.072, 0.093] | 0.044 [0.036, 0.053] | 0.029 [0.023, 0.036] | 0.101 [0.089, 0.112] | 0.095 [0.084, 0.106] | 88.1% |
| imperfect | 12 | 0.050 [0.041, 0.058] | 0.033 [0.026, 0.040] | 0.028 [0.022, 0.035] | 0.076 [0.066, 0.086] | 0.065 [0.056, 0.075] | 99.8% |
| imperfect | 24 | 0.042 [0.034, 0.051] | 0.029 [0.022, 0.036] | 0.028 [0.022, 0.035] | 0.064 [0.054, 0.073] | 0.056 [0.047, 0.065] | 100.0% |
| exact | 1 | 0.157 [0.145, 0.170] | 0.157 [0.145, 0.170] | 0.000 | 0.157 [0.145, 0.170] | 0.158 [0.146, 0.170] | 0.0% |
| exact | 2 | 0.082 [0.072, 0.091] | 0.072 [0.062, 0.081] | 0.000 | 0.090 [0.080, 0.100] | 0.088 [0.077, 0.097] | 8.4% |
| exact | 4 | 0.029 [0.023, 0.035] | 0.016 [0.011, 0.020] | 0.000 | 0.050 [0.042, 0.058] | 0.041 [0.034, 0.048] | 58.9% |
| exact | 6 | 0.011 [0.007, 0.014] | 0.004 [0.001, 0.006] | 0.000 | 0.034 [0.027, 0.041] | 0.025 [0.019, 0.031] | 88.1% |
| exact | 12 | 0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] | 0.000 | 0.015 [0.010, 0.020] | 0.007 [0.003, 0.011] | 99.8% |
| exact | 24 | 0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.000] | 0.000 | 0.008 [0.004, 0.011] | 0.007 [0.004, 0.010] | 100.0% |

Paired differences, attention v2 minus extremized_log_pool, in the N sweep (seed 0):

| model | N = 1 | N = 2 | N = 4 | N = 6 | N = 12 | N = 24 |
|---|---|---|---|---|---|---|
| imperfect | -0.001 [-0.003, 0.001] | -0.004 [-0.007, -0.001] | -0.008 [-0.013, -0.004] | -0.006 [-0.012, 0.001] | -0.010 [-0.015, -0.006] | -0.008 [-0.013, -0.003] |
| exact | 0.000 [-0.001, 0.001] | -0.003 [-0.006, 0.001] | -0.009 [-0.014, -0.004] | -0.009 [-0.014, -0.004] | -0.008 [-0.012, -0.005] | -0.000 [-0.005, 0.004] |

They are also in `json/*__v2.json` under `extra.paired_regret_differences`. The full v1 and v2 tables come from `report.py v1` and `report.py v2`.

## Reading

- **Compute as data.** More rollouts lower the ceiling the way more observers do, until the context is
  covered (N ~ 12). Past that point an exact model's ceiling is 0. The imperfect model keeps a floor of
  0.029 nats because its shared misreading e_k is identical in every rollout and never averages out. Shared
  model error is to rollouts what shared noise factors are to observers in E1.
- **Scalar reports lose more for rollouts than for observers.** The gap between the trace ceiling and the
  reports ceiling is 0.038 for imperfect rollouts and 0.007 for exact ones. A rollout's final answer mixes
  its order-dependent weighting with the shared bias, and the trace keeps them apart. This supports
  vector messages (lane L2's question) for search sources in particular.
- **Learned aggregation.** With keys and enough data (v2), the attention aggregator beats extremized pooling
  on the exact-model control at N = 4-12, and on imperfect rollouts at N = 2, 4, 12 and 24 (seed 0). It is still 0.012-0.018 nats
  above the reports ceiling at N = 6. v1 shows the failure mode: with continuous keys and 50k episodes, the
  aggregator memorizes episodes.

## Caveats

- Exploratory: one environment family and one default error mix. The parameters were tuned once, before the
  learned runs, so that the ceiling gaps are visible.
- Recipe v2 was chosen after seeing v1. Its early stopping uses train-split episodes, not dev.
- The N sweep has one seed per learned arm.
- The floor is derived, not brute-forced.
- The keys include true mu. A real search log would have to estimate it.

## Reproduce

```bash
cd /home/agent/workspaces/inconsistency-search
# forecasts (ran on spot L4 VMs at runner commits 963f107 (v1) and edfbe03 (v2), about 30 L4-minutes in total)
L9_RECIPE=v2 python3 results/explore/l9-rollouts-as-sources-20261020/run.py forecasts --out <dir>
# result JSONs (local, needs a clean harness with the same sha256), then tables
L9_RECIPE=v2 PYTHONPATH=.:/usr/lib/python3/dist-packages /home/agent/.venvs/trader-cat/bin/python3 \
  results/explore/l9-rollouts-as-sources-20261020/run.py results --forecasts <dir>
python3 results/explore/l9-rollouts-as-sources-20261020/report.py v2
```

Files:

- `json/<model>__N<n>__<keys>[__v2].json`: 30 validated `aggbench.result/v1` files, one per condition.
- `summary.json`, `summary__v2.json`: the rows behind the tables.
- `run.py`, `report.py`: the runner and the table script.
- The raw dev forecasts (npz) are not in git. They are under `/tmp/l4runs/l9-rollouts-b/` and
  `/tmp/l4runs/l9-rollouts-v2/`.
