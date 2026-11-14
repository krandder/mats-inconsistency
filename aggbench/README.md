# aggbench: stage-1 harness for learned aggregation

Question: can an attention-like aggregator learn to combine information from sources that overlap in
what they know and differ in reliability, and approach the Bayes-optimal aggregate given everything
they saw? This package gives every experiment the same environments, exact ceilings, baselines,
metrics and controls.

**The setup is rich messages.** Each agent runs a learned encoder over its own observation
(`batch.agent_inputs(visibility, alignment='unaligned')`, locality-checked) and sends a
high-dimensional VALUE and KEY; the aggregator sees messages only. Kelvin, 2026-10-20: "The output
(value) needs to be a high-dimensional." Headline results use rich messages, unaligned inputs, and
keys at `none`/`noisy`, and are compared with `full_context_transformer` (Kelvin's bar: no worse than a
transformer that reads all the raw inputs).

**Scalar reports are only a baseline.** The `reports` field below (each agent's exact local
posterior) is a legacy, impoverished interface kept for the closed-form baselines. A result that uses
it must be labelled "scalar-report baseline"; it is not the design.

## Environments (`aggbench/envs.py`, `make_env(name, **overrides)`)

| name | latent | agent i sees | reports |
|---|---|---|---|
| E1 | theta ~ N(0, I_d), d=4 | s_i = A_i theta + B_i eta + eps_i: 2 projections of 3 of the 4 coordinates (overlapping coverage), shared noise factors eta (correlated errors), private noise sd log-uniform in [0.3, 2] | posterior mean and precision |
| E2 | Y in {0,1}, P(Y=1)=0.5 | 2-6 of K=12 evidence pieces; piece k is x_k ~ N(+-mu_k, 1), mu_k ~ U(0.1, 0.6), conditionally independent given Y | posterior log-odds |
| E3-sybil | as E2 | plus 3 exact copies of one honest agent | as E2 |
| E3-biased | as E2 | one agent reports l0 + 3 (l - l0) + 1.5 instead of l | as E2 |
| E3-collude | as E2 | 3 agents pool their pieces and all report the pooled posterior | as E2 |
| E4 | as E2 with K=8; all K pieces are the evidence | each source is a ROLLOUT of one imperfect model: it reads 2-6 pieces in its own random order | the model's log-odds (errors correlated through shared model error) |
| Z1-Z9 | finite world states; Z8 linear-Gaussian (see "Structure zoo") | common causes, nonlinear readings, outliers, hypothesis samples, sub-questions, XOR, regime-dependent reliability, hidden provenance, k-bit parity | posterior log-odds |

Structure (coverage, strengths, noise loadings, adversary positions) is redrawn every episode and
agent order is shuffled, so position carries no information. Keys come at three visibility levels:
`none` (empty), `noisy` (E1: A_i, B_i, log sigma_i plus N(0, 0.3^2) noise; E2: coverage bits flipped with
probability 0.15, strengths jittered) and `full` (E1: A_i, B_i, log sigma_i; E2: coverage bits and
coverage x mu). Keys describe information, not honesty: a biased agent's key looks honest.

Each `Batch` holds truth, per-agent raw observations, each agent's exact local posterior, its
report, keys at all three levels, the exact global posterior `bayes` (given all agents'
observations, each E2 piece counted once) and `oracle` (the true structure). An aggregator gets only
`batch.view(visibility)`: reports, keys, public constants.

Two ceilings. `bayes` is the global posterior; regret is always measured against it.
`oracle_clearing` is the exact posterior given the reports plus the true structure, computed with the
clearing equation (K^T B K + Lambda) w = K^T B Delta_l. In E1 it equals `bayes` (each report is a lossless
summary because an agent observes at most as many scalars as it covers). In E2/E3 it equals `bayes` only
when the union's evidence is a linear function of the reports (`envs.union_identified`, rare at the
default config); otherwise the reports have lost information, and `oracle_clearing` is the best any
aggregator of reports can do. At visibility `full` in E2 it is a function of the view. The
`oracle_clearing` column of the table below (its regret to Bayes) is that information loss.

Splits are seed ranges: episode j of a split uses seed base + j, with bases train 0, dev 10^7,
test 2x10^7 and smoke 9x10^7 (`splits.verify_disjoint` checks a set of seed arrays). `sample("test", ...)` refuses
without `allow_test=True`.

## Run

Run every experiment from your OWN git worktree, never from the shared checkout
`/home/agent/workspaces/inconsistency-search` (other lanes edit files there, which stamps your results
`git_dirty=true`; `results.validate` now rejects those):

```bash
git -C /home/agent/workspaces/inconsistency-search fetch origin
git -C /home/agent/workspaces/inconsistency-search worktree add /tmp/my-lane -b my-lane origin/main
cd /tmp/my-lane          # work, commit, then rebase on origin/main and push from here
python3 -m pytest -q tests/aggbench                       # numpy tests; the torch tests skip
PYTHONPATH=/home/agent/.local/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:. \
  /home/agent/.venvs/trader-cat/bin/python3 -m pytest -q tests/aggbench   # all tests, with torch
python3 -m aggbench.report --write                        # recompute the table below (numpy baselines)
PYTHONPATH=/home/agent/.local/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:. \
  /home/agent/.venvs/trader-cat/bin/python3 -m aggbench.report --transformer --write  # + full_context_transformer
/home/agent/.venvs/trader-cat/bin/python3 -m aggbench.examples.smoke_attention --env E2   # SMOKE only
```

```python
from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.metrics import summarize, paired_difference
env = make_env("E3-collude", n_agents=8)          # overrides are recorded in the result JSON
# every environment with its family tag: aggbench.zoo.list_envs(); held-out ones need allow_heldout=True
tr, dv = env.sample("train", 2000), env.sample("dev", 2000)
arms = run_with_controls(make_my_aggregator, tr.view("full"), tr.truth, dv.view("full"))
summarize(arms["trained"], dv)["regret"]          # {mean, ci_low, ci_high, n}
```

A learned aggregator has `.module` (a torch module), `.fit(view, truth)` and `.predict(view)`, which
returns `{"logit": (n,)}` (E2/E3) or `{"mean": (n,d), "prec": (n,d,d)}` (E1).

## Baselines on the default config

`prior` = the public prior alone (its regret is what the agents' information is worth); `linear_pool` = mean
of probabilities (E1: moment-matched mixture); `log_pool` = mean of log-odds (E1: mean precision and
information); `sum_logodds` = independent product, sum of log-odds minus (N-1) prior log-odds;
`extremized_log_pool` = l0 + alpha (mean log-odds - l0), alpha fitted on train; `oracle_clearing` = the
reports ceiling (E1-E4 as above; zoo: exact posterior given the reports and the true structure). None of them
uses keys. Every environment, held-out ones included, one row each (rows were columns before 2026-10-21).

<!-- BASELINES:BEGIN -->
| env | family | prior | floor | floor_none | oracle_clearing | linear_pool | log_pool | sum_logodds | extremized_log_pool | transformer fresh (unaligned, provenance none) | transformer fresh (unaligned, provenance noisy) | transformer pool 20k (unaligned, provenance none) | transformer pool 20k (unaligned, provenance noisy) | Bayes log loss | alpha | reports ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | linear-gaussian | 4.448 [4.359, 4.535] | 0.000 [0.000, 0.000] | unknown | -0.000 [-0.000, 0.000] | 3.398 [3.324, 3.469] | 2.047 [1.988, 2.108] | 0.827 [0.770, 0.884] | 0.829 [0.771, 0.887] | 1.095 [1.035, 1.151] | 1.103 [1.043, 1.162] | 1.459 [1.381, 1.534] | 1.490 [1.409, 1.572] | 1.193 | 5.17 | reports lossless (always) |
| E2 | evidence-pieces | 0.433 [0.411, 0.454] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.017 [0.009, 0.024] | 0.152 [0.137, 0.166] | 0.112 [0.099, 0.125] | 0.132 [0.100, 0.167] | 0.045 [0.033, 0.056] | 0.035 [0.025, 0.045] | 0.037 [0.026, 0.048] | 0.051 [0.037, 0.066] | 0.072 [0.056, 0.087] | 0.260 | 2.17 | union identified 3.1% |
| E3-sybil | adversarial | 0.433 [0.411, 0.454] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.017 [0.009, 0.024] | 0.159 [0.143, 0.173] | 0.126 [0.111, 0.139] | 0.417 [0.345, 0.495] | 0.078 [0.062, 0.092] | 0.042 [0.031, 0.052] | 0.042 [0.030, 0.053] | 0.068 [0.054, 0.082] | 0.069 [0.055, 0.083] | 0.260 | 1.90 | union identified 3.1% |
| E3-biased | adversarial | 0.433 [0.411, 0.454] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.017 [0.009, 0.024] | 0.142 [0.128, 0.156] | 0.091 [0.077, 0.105] | 0.341 [0.278, 0.407] | 0.076 [0.060, 0.091] | 0.039 [0.028, 0.050] | 0.043 [0.031, 0.055] | 0.061 [0.046, 0.077] | 0.078 [0.061, 0.094] | 0.260 | 1.40 | union identified 3.1% |
| E3-collude | adversarial | 0.433 [0.411, 0.454] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.018 [0.010, 0.025] | 0.074 [0.063, 0.085] | 0.046 [0.036, 0.056] | 0.259 [0.206, 0.314] | 0.030 [0.020, 0.039] | 0.024 [0.015, 0.033] | 0.025 [0.015, 0.034] | 0.048 [0.036, 0.060] | 0.043 [0.031, 0.055] | 0.260 | 1.38 | union identified 11.8% |
| E4 | rollouts | 0.359 [0.334, 0.381] | 0.047 [0.033, 0.060] | 0.047 [0.033, 0.060] | 0.087 [0.070, 0.103] | 0.140 [0.121, 0.156] | 0.114 [0.096, 0.131] | 0.538 [0.455, 0.632] | 0.104 [0.085, 0.122] | 0.079 [0.062, 0.096] | 0.081 [0.063, 0.097] | 0.093 [0.074, 0.110] | 0.098 [0.079, 0.115] | 0.334 | 1.31 | reports ceiling (rollouts) |
| Z1 | common-cause | 0.533 [0.515, 0.551] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.274 [0.260, 0.288] | 0.185 [0.172, 0.198] | 0.053 [0.036, 0.072] | 0.034 [0.024, 0.045] | 0.013 [0.007, 0.019] | 0.015 [0.008, 0.021] | 0.022 [0.015, 0.030] | 0.032 [0.022, 0.041] | 0.160 | 3.94 | reports lossless 100.0% |
| Z2 | nonlinear | 0.482 [0.462, 0.499] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.016 [0.010, 0.022] | 0.262 [0.248, 0.276] | 0.137 [0.124, 0.150] | 0.039 [0.025, 0.054] | 0.025 [0.016, 0.034] | 0.001 [-0.001, 0.003] | 0.001 [-0.002, 0.004] | 0.011 [0.006, 0.018] | 0.019 [0.009, 0.030] | 0.211 | 4.27 | reports lossless 4.9% |
| Z4 | hypothesis-samples | 0.520 [0.498, 0.541] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.271 [0.253, 0.288] | 0.234 [0.216, 0.250] | 0.048 [0.030, 0.066] | 0.022 [0.013, 0.032] | 0.003 [-0.001, 0.007] | 0.002 [-0.002, 0.006] | 0.009 [0.003, 0.015] | 0.012 [0.005, 0.019] | 0.173 | 5.19 | reports lossless 100.0% |
| Z5 | multi-question | 0.421 [0.398, 0.443] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.159 [0.141, 0.176] | 0.149 [0.132, 0.166] | 0.518 [0.436, 0.599] | 0.136 [0.117, 0.154] | 0.003 [-0.000, 0.007] | 0.004 [0.000, 0.008] | 0.020 [0.011, 0.028] | 0.031 [0.021, 0.042] | 0.240 | 1.28 | reports lossless 100.0% |
| Z6 | complementary-xor | 0.247 [0.224, 0.271] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.247 [0.224, 0.271] | 0.247 [0.224, 0.271] | 0.247 [0.224, 0.271] | 0.247 [0.224, 0.271] | 0.247 [0.224, 0.271] | 0.003 [0.000, 0.005] | 0.003 [0.001, 0.006] | 0.012 [0.005, 0.018] | 0.013 [0.006, 0.020] | 0.447 | 0.06 | reports lossless 2.9% |
| Z8 | hidden-provenance | 0.457 [0.434, 0.480] | 0.000 [0.000, 0.000] | unknown | 0.024 [0.016, 0.032] | 0.164 [0.149, 0.179] | 0.118 [0.104, 0.131] | 0.095 [0.068, 0.122] | 0.035 [0.025, 0.045] | 0.036 [0.026, 0.048] | 0.037 [0.027, 0.048] | 0.084 [0.064, 0.104] | 0.075 [0.059, 0.091] | 0.236 | 2.39 | reports lossless 0.1% |
| Z9 | k-parity | 0.447 [0.419, 0.473] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.447 [0.419, 0.473] | 0.447 [0.419, 0.473] | 0.447 [0.419, 0.473] | 0.447 [0.419, 0.473] | 0.447 [0.419, 0.473] | 0.002 [-0.001, 0.005] | 0.002 [-0.001, 0.005] | 0.010 [0.004, 0.016] | 0.021 [0.012, 0.031] | 0.246 | 0.06 | reports lossless 0.0% |
| Z9-hard | k-parity | 0.082 [0.065, 0.098] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.082 [0.065, 0.098] | 0.082 [0.065, 0.098] | 0.082 [0.065, 0.098] | 0.082 [0.065, 0.098] | 0.082 [0.065, 0.098] | 0.006 [-0.000, 0.012] | 0.082 [0.065, 0.098] | 0.012 [0.002, 0.022] | 0.083 [0.066, 0.099] | 0.611 | 0.06 | reports lossless 0.0% |
| Z1-blind | common-cause | 0.550 [0.530, 0.567] | 0.000 [0.000, 0.000] | 0.012 [0.006, 0.019] | 0.000 [0.000, 0.000] | 0.264 [0.249, 0.277] | 0.175 [0.162, 0.187] | 0.075 [0.053, 0.099] | 0.041 [0.029, 0.053] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.025] | 0.021 [0.012, 0.031] | 0.040 [0.027, 0.053] | 0.143 | 3.56 | reports lossless 100.0% |
| Z4-blind | hypothesis-samples | 0.512 [0.489, 0.534] | 0.000 [0.000, 0.000] | 0.005 [-0.000, 0.010] | 0.000 [0.000, 0.000] | 0.272 [0.253, 0.289] | 0.241 [0.223, 0.258] | 0.051 [0.033, 0.069] | 0.025 [0.014, 0.036] | 0.005 [0.000, 0.010] | 0.005 [0.000, 0.011] | 0.007 [0.001, 0.012] | 0.013 [0.005, 0.021] | 0.181 | 5.59 | reports lossless 100.0% |
| Z3 (held out) | heavy-tail | 0.525 [0.506, 0.543] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.184 [0.172, 0.196] | 0.135 [0.123, 0.146] | 0.051 [0.035, 0.069] | 0.016 [0.010, 0.022] | 0.002 [-0.000, 0.004] | 0.001 [-0.001, 0.004] | 0.011 [0.006, 0.017] | 0.010 [0.004, 0.016] | 0.168 | 3.45 | reports lossless 100.0% |
| Z7 (held out) | context-reliability | 0.617 [0.599, 0.634] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.338 [0.323, 0.353] | 0.226 [0.211, 0.240] | 0.027 [0.016, 0.041] | 0.028 [0.016, 0.042] | 0.006 [-0.000, 0.012] | 0.006 [-0.002, 0.014] | 0.024 [0.012, 0.036] | 0.024 [0.013, 0.035] | 0.076 | 6.48 | reports lossless 100.0% |
| Z1-wide (held out) | common-cause | 0.493 [0.472, 0.512] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.278 [0.262, 0.294] | 0.208 [0.193, 0.223] | 0.218 [0.173, 0.266] | 0.061 [0.048, 0.074] | 0.004 [0.001, 0.008] | 0.005 [0.001, 0.009] | 0.015 [0.008, 0.023] | 0.021 [0.013, 0.029] | 0.200 | 3.32 | reports lossless 100.0% |
| Z5-majority (held out) | multi-question | 0.425 [0.403, 0.447] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.170 [0.152, 0.188] | 0.159 [0.141, 0.176] | 0.467 [0.397, 0.541] | 0.135 [0.116, 0.154] | 0.008 [0.003, 0.013] | 0.007 [0.002, 0.012] | 0.038 [0.028, 0.049] | 0.037 [0.024, 0.050] | 0.268 | 1.50 | reports lossless 100.0% |
| Z6-parity3 (held out) | complementary-xor | 0.157 [0.137, 0.177] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.157 [0.137, 0.177] | 0.157 [0.137, 0.177] | 0.157 [0.137, 0.177] | 0.157 [0.137, 0.177] | 0.157 [0.137, 0.177] | 0.003 [-0.001, 0.007] | 0.004 [0.000, 0.009] | 0.016 [0.008, 0.023] | 0.031 [0.020, 0.042] | 0.536 | 0.06 | reports lossless 7.1% |

Mean regret to the global Bayes posterior (nats per episode) with 95% bootstrap CIs over 2000 dev episodes; `prior` = the public prior alone (what the agents' information is worth). Generated by `python3 -m aggbench.report --write` at harness commit `4183f0a02b`, 2026-10-21T07:58:07+00:00; full numbers in `aggbench/baselines_default.json`; brute-force oracle checks: E1 3e-15, E2 2e-16, E3-sybil 1e-16, E3-biased 2e-16, E3-collude 2e-16, E4 1e-15, Z1 1e-15, Z2 7e-16, Z4 2e-15, Z5 2e-16, Z6 3e-16, Z8 2e-15, Z9 9e-16, Z9-hard 1e-15, Z1-blind 4e-16, Z4-blind 5e-15, Z3 1e-15, Z7 5e-16, Z1-wide 1e-15, Z5-majority 3e-16, Z6-parity3 4e-16.

`transformer fresh` (headline regime): `full_context_transformer` trained PFN-style on 3000 steps x 512 FRESH simulator episodes (no reuse, slot permutations resampled every minibatch), lr 0.001, weight decay 0.01, dropout 0.1; harness commit `0d94ef1483`, 2026-10-21T04:14:24+00:00; controls in `aggbench/baselines_transformer_fresh.json`. `transformer pool 20k` (secondary): a fixed pool of 20k train episodes with early stopping on 10% of them.

`full_context_transformer` (`aggbench/transformer.py`): 3-layer pre-norm transformer encoder, width 64, 4 heads, CLS readout, one token per agent = [raw observation, present mask, own sensor model, provenance key, report] from `agent_inputs(alignment=..., own_model=True)`; trained from outcomes on 20000 train episodes, 3000 AdamW steps of 512, OneCycle lr 0.001; evaluated on 2000 dev episodes with the harness controls (random-network and untrained-init regrets are in `aggbench/baselines_transformer.json`). Harness commit `23f3e4be07`, 2026-10-21T03:28:56+00:00. E3-biased: the biased agent's raw observation is honest, so the transformer sees through the bias there.
<!-- BASELINES:END -->

## E4: rollouts of an imperfect model as sources (added 2026-10-20, lane L9)

Kelvin asked how much search rollouts act like gathering information. In E4 the sources are not
independent observers but N=6 rollouts of one imperfect any-order predictor on one context. The world is
E2's: Y with prior 0.5, K=8 pieces x_k ~ N(+-mu_k, 1), piece LLR lam_k = 2 mu_k x_k, and the evidence is
all K pieces. Rollout i reads L_i ~ U{2..6} pieces in its own random order and reports

    r_i = logit0 + b + sum_{t=1..L_i} [ a gamma^(L_i - t) (lam_k(t) + e_k(t)) + nu_it ]

- a = 1.5 (`calibration`): the model is over-confident.
- gamma = 0.7 (`recency`): it forgets early evidence, so its answer depends on the order. An exact model
  gives the same answer in every order.
- e_k ~ N(0, 0.3^2) (`read_noise`): it misreads piece k the same way in every rollout that reads k. This
  is shared model error, and it makes the rollouts' errors correlated.
- b ~ N(0, 0.3^2) (`shared_bias`): an episode-level bias shared by all rollouts.
- nu_it ~ N(0, 0.3^2) (`step_noise`): private sampling noise of each rollout.

Fields: `local` is what an exact model reports with the same conditioning set. `obs` is each rollout's
trace (start value logit0 + b, then its per-step increments, NaN after step L_i); the trace is the
message for vector-message experiments. Keys are the coverage bits, recency (steps before the end at
which each piece was read, / K) and coverage x mu. At visibility `noisy`, bits flip with probability
0.15 and the values are jittered. `make_env("E4", calibration=1, recency=1, read_noise=0, shared_bias=0,
step_noise=0)` is the exact-model control: then the rollouts are E2-style overlapping observers.

Three exact ceilings, all Gaussian given Y:

- `bayes` is the posterior given all K pieces.
- `oracle_clearing` is the posterior given the N reports and the true error structure (orders, a, gamma,
  noise sds): l0 + 2 (W m)^T Sigma^+ (r - l0), with m_k = 2 mu_k^2 and
  Sigma = W diag(2m) W^T + s_e^2 W W^T + s_b^2 11^T + s_nu^2 diag(L).
- `envs.rollout_trace_posterior_logit` is the same posterior given every rollout's whole trace.

Checks: `verify.brute_force_rollouts` (run by `oracle_check`) recomputes `bayes` by enumerating Y over the
raw piece densities. It recomputes the reports ceiling from a latent-by-latent construction along each
rollout's order, with both class densities evaluated in full; it agrees to 1e-15. `tests/aggbench/test_rollouts.py`
also:

- fits a logistic regression on 400k simulated draws from the environment's own sampler with the structure
  fixed, and recovers the weights of the reports ceiling and the trace ceiling;
- checks that the exact-model control reduces to E2's formula, and that its trace ceiling reduces to Bayes
  on the union;
- checks that a mutated ceiling (no recency decay) fails the brute-force check;
- checks the order: mean regret of bayes < trace ceiling < reports ceiling.

## Rules for experiments that use this harness

1. Import `aggbench` unmodified, from your own git worktree (see Run). The result JSON records
   `harness.git_commit`, `harness.git_dirty` and `harness.sha256`; `results.validate` rejects
   `git_dirty=true` unless you pass `allow_dirty=True` (recorded as `harness.dirty_allowed`, never for a
   headline result). Change environments only through `make_env` overrides. A
   harness change is its own commit with its tests, before any experiment uses it.
2. Default architecture (2026-10-21): each agent runs an encoder over its own `batch.agent_inputs(...)` (its
   raw observation, its own sensor model, the provenance key at the run's visibility) and sends a message; the
   aggregator sees the messages and `view.public` only. An agent's inputs reach only its own encoder (prove it
   with `messages.check_locality`). Only the scalar-report baselines read `view.reports`. `truth` is a training
   label on the train split; fitting to `bayes` (distillation) must be declared. `oracle`, `bayes` and `local`
   never reach `predict`.
3. Fit on train, choose on dev. The test split is for confirmatory rounds only (PROTOCOL 10.2).
4. Report mean regret to Bayes with 95% bootstrap CIs (`metrics.summarize`), and paired differences
   (`metrics.paired_difference`) against `extremized_log_pool` and `oracle_clearing`.
5. Run every learned arm through `controls.run_with_controls` (random-network and untrained-init
   controls through the same `predict`). A claim the random-network control also passes is void.
6. Write `results/explore/<run-id>/` with a `README.md` (question, arms, what was measured, a one-line
   result, the EXPLORATORY label) and the JSON from `results.write_result`. It validates
   `result_schema.json`, requires a verified brute-force `oracle_check` (>= 20 episodes) and a control
   entry for every learned arm. Smoke runs use track SMOKE and give no verdicts.

## High-dimensional messages (added 2026-10-20, Kelvin: "The output (value) needs to be a high-dimensional.")

A scalar report loses information: an agent's log-odds cannot separate the evidence it shares with others from its private evidence, which is why `oracle_clearing` stays above `bayes` in E2. Experiments may therefore train per-agent encoders that send vector values (and keys) instead of scalars:

- `batch.agent_inputs(visibility, alignment=..., own_model=True)` returns `(obs, keys, public, own)`: each
  agent's own raw observation, the provenance key at `visibility`, and its own sensor model (see below).
- An encoder may map agent i's own inputs to agent i's message only. Nothing may mix agents before the messages are sent. Prove it with `aggbench.messages.check_locality(encode, obs, keys)` and record the result in the run's JSON.
- The aggregator sees messages only, never raw observations. Report regret to `bayes` and to `oracle_clearing`: beating `oracle_clearing` is the evidence that richer messages carry information a scalar loses.
- E3-biased with vector messages: the biased agent's raw observations are honest, so apply the environment's distortion to its message, or exclude E3-biased and say so.

### Aligned versus unaligned inputs (2026-10-21)

In E2/E3 `agent_inputs` gives each agent's observation as a K-vector indexed by evidence piece, so piece k sits
at coordinate k for every agent. That hands an aggregator the overlap structure for free: a slot-equivariant
model can dedup per coordinate and reach Bayes (`tests/aggbench/test_alignment.py`: per-position dedup has
regret 0 aligned, > 0.05 unaligned). Real sources do not come with aligned evidence slots.

`batch.agent_inputs(visibility, alignment="unaligned")` passes each agent's slot-indexed coordinates through an
agent-specific, episode-fixed random permutation drawn from the episode seed and not revealed: the observation
and the slot-indexed key blocks alike (E1: the m observed scalars and the rows of A_i, B_i in the key; E2/E3:
the K pieces, also in the coverage and coverage x mu keys; E4: the piece-indexed keys, the trace is already in
the rollout's own order; zoo: the identity of a shared parent, Z1 group and Z4 model). Unseen pieces (NaN) move
with their slot, so position says nothing. `mixing=True` (E1 only) uses a random orthogonal matrix instead.
`envs.alignment_transforms(batch)` returns the hidden transforms for ceilings and tests only. The default
stays `"aligned"` for backward compatibility. The ceilings are unchanged: the transform is invertible (tested),
and in E1 the exact posterior computed from the unaligned observations and full keys alone equals Bayes.

### Own sensor model versus provenance (2026-10-21)

Two different things used to share the key. What an agent knows about its OWN sensor or likelihood model (how
its own observation depends on the latent) and what the aggregator is told about provenance (which agents share
sources). A real forecaster knows its own sources, so `agent_inputs(..., own_model=True)` always returns the
agent's own model, under the same hidden transform as its observation, whatever the visibility;
`visibility` (none/noisy/full) only sets the provenance key. Without the own model, Bayes is out of reach at
visibility `none` (design review: E2 0.039 versus the scalar ceiling 0.017; Z6 0.149 versus the prior's 0.247).
Rule: the own model may describe the agent's own likelihood and nothing about which other agents share its
sources beyond what that likelihood implies (`test_own_model_is_the_likelihood_and_nothing_more` rebuilds it
from the oracle). Per environment:

- Copied provenance removed (2026-10-21 fix): E1's own model is (A_i, R_i = B_i B_i^T + sigma_i^2 I), not the
  loadings B_i, which name the shared noise factors; Z1 and Z4 drop the parent-group / model one-hot.
- Own model = likelihood, no provenance to hide: Z2, Z3, Z5, Z7 (no shared sources besides the latent), Z6 and
  Z5 (which bit or sub-question an agent reads IS its likelihood).
- Structural, `none` is not a test of provenance inference: in E2/E3 a shared piece is the same float in two
  observations and its strength mu is a continuous fingerprint; in E4 mu fingerprints pieces; in Z1 and Z4 the
  parent's rho or the model's mu does. While agents send content and know their own model, overlap can be read
  off by exact matching.
- Variants where `none` is a real test: Z8 (E2 with private read noise and three shared strength levels),
  Z1-blind (rho in {0.75, 0.9}) and Z4-blind (both models have strength 1.0). Neither content nor own models
  identify shared sources there (`test_z8_content_and_own_models_do_not_fingerprint`, Z1-blind test).

Floors. `floor`: the exact posterior given everything the agents know WITH full provenance: Bayes everywhere
except E4, where it is the trace ceiling (`envs.rollout_trace_posterior_logit`; Bayes there conditions on context
pieces no rollout read). `floor_none`: the exact posterior given what agents can send at provenance `none`
(observations and own models, source identities marginalized under the environment's prior;
`zoo.none_floor_logit`). It equals `floor` where provenance is recoverable (the structural cases above; tests
rebuild Bayes from unaligned inputs plus own models in E2, Z1 and Z6). It is computed exactly by enumeration for
Z1-blind and Z4-blind (tested: equal to Bayes with a single source, KL >= 0, calibrated), and shown as `unknown`
(not computed, not Bayes) for E1 (shared noise factors) and Z8 (piece overlap; a Monte Carlo mixture over
overlap structures would be needed).

Z8's key has a third block, the identity of the piece at each read coordinate. It is provenance, never in the own
model; at `full` it moves with the coordinate in unaligned mode, so the aggregator can group the readings of one
piece and rebuild Bayes (tested); `noisy` replaces an identity by a random one with probability 0.15.

**Headline results must use `alignment="unaligned"`, the own model, and provenance keys at `none` and
`noisy`.** Aligned or `full`-provenance runs are diagnostics.

## The full-context transformer bar (added 2026-10-21)

Kelvin: "Our aggregation mechanism cannot be worse than a transformer with self-attention that can read all
the info sequentially." and "all kinds of simple things we can imagine need to be learnable if they would be
learnable by a transformer that sees all the info sequentially and has to make a decision after."

`aggbench/transformer.py` provides `full_context_transformer`: a plain pre-norm transformer encoder (3 layers,
width 64, 4 heads, CLS readout, no positional encoding, no hand structure) over one token per agent, token =
[raw observation, present mask, own sensor model, provenance key, report] from
`agent_inputs(visibility, alignment, own_model=True)`. It is trained from
outcomes and run through `controls.run_with_controls`, in one of two training regimes
(`evaluate_full_context_transformer(..., regime=...)`; the same API for every learned arm):

- `regime="fresh"` (default; the HEADLINE regime, PFN-style; Kelvin: training speed does not matter): 3000 steps x
  512 FRESH simulator episodes, one new minibatch per step, never reused; every episode has its own seed, so slot
  permutations are resampled with every minibatch (`transformer.FreshStream`, built ahead by sampler processes).
- `regime="pool"` (secondary row): a fixed pool of 20k train episodes, up to 3000 steps, with early stopping on 10%
  of the train episodes (scored every 100 steps, best weights kept, stop after 1000 steps without improvement),
  never on dev. Without early stopping the transformer memorizes the pool: E2 dev regret 1.0, worse than its own
  untrained control (0.49) and than the prior.

Both use AdamW, OneCycle lr 1e-3, weight decay 0.01, dropout 0.1, batch 512.

**E1: bar far from Bayes** (diagnosed 2026-10-21; not a bug). The transformer's E1 regret (fresh 1.09 / 1.10 at
provenance none / noisy, pool 1.46 / 1.49) is an order of magnitude above the other environments.
(a) Scale: prior 4.45, untrained-init control 6.7 / 7.5, random-network control 7.8 / 7.6, sum_logodds and
extremized 0.83, Bayes log loss 1.19 nats. (b) Head: a Gaussian over the 4-d theta with full covariance (mean and the
Cholesky factor of the precision), the family of the exact posterior; regret is -log q(theta) minus Bayes' -log
p(theta) (`metrics.per_episode`), the same for every method. (c) No generalization gap in the fresh regime: the
training NLL on fresh minibatches at step 3000 (no dropout) is 2.18 versus Bayes 1.19, a gap of 0.98, and the dev
regret is 0.99. It is still falling at step 3000: 12000 steps give 0.93, and no dropout at 3000 steps gives 0.99.
(d) Inputs are standardized per feature from 4096 held-out train episodes (own-model A_i entries O(1), R_i up to
about 5, observations up to about +-5). E1 needs a matrix-valued combination (summing per-agent precision terms
A_i^T R_i^-1 A_i with the shared-factor corrections, then inverting), which this small transformer learns slowly.
Read E1 as "bar far from Bayes", not as a ceiling an aggregator must stay under. `evaluate_full_context_transformer(env, visibility=..., alignment=...)` returns
metrics, controls and the training log; the table below has it on every environment, unaligned, with the own
model, provenance keys at `none` and at `noisy`, fresh (headline) and pool (secondary)
(`python3 -m aggbench.report --fresh --transformer --write --job-dir DIR` retrains them; the fresh regime samples
1.5M episodes per cell, hours of CPU; finished cells are saved in DIR and a rerun resumes).

Kelvin's bar is what a transformer CAN learn, not what it learns on a small budget, so the data-scaling curve
below trains it on 20k, 80k and 320k episodes (steps 3000, 6000, 12000) on E2, Z8, Z1 and Z6 at the headline
setting (`python3 -m aggbench.report --scaling --write`). Compare an aggregator with the transformer at the same
training budget, and read the curve to see how far more data takes the transformer toward the floor.

<!-- SCALING:BEGIN -->
| env | 20k train, 3000 steps | 80k train, 6000 steps | 320k train, 12000 steps | floor (provenance none) | prior | extremized_log_pool |
|---|---|---|---|---|---|---|
| E2 | 0.051 [0.037, 0.066] | 0.042 [0.030, 0.055] | 0.039 [0.028, 0.050] | 0.000 | 0.433 | 0.045 |
| Z8 | 0.084 [0.064, 0.104] | 0.051 [0.038, 0.064] | 0.041 [0.029, 0.053] | unknown | 0.457 | 0.035 |
| Z1 | 0.022 [0.015, 0.030] | 0.015 [0.009, 0.021] | 0.012 [0.006, 0.018] | 0.000 | 0.533 | 0.034 |
| Z6 | 0.012 [0.005, 0.018] | 0.008 [0.003, 0.013] | 0.005 [0.000, 0.009] | 0.000 | 0.247 | 0.247 |

`full_context_transformer` regret to Bayes (nats, 95% bootstrap CI, 2000 dev episodes) versus training episodes, unaligned inputs with own models, provenance keys `none`; steps grow with the data. Generated by `python3 -m aggbench.report --scaling --write` at harness commit `23f3e4be07`, 2026-10-21T03:30:16+00:00; full numbers and controls in `aggbench/baselines_transformer_scaling.json`.
<!-- SCALING:END -->

The bar, at matched message bandwidth:

- When messages can carry each agent's full input (raw observation and key), an aggregator worse than
  `full_context_transformer` in the same setting (alignment, visibility, train data) fails the bar.
- When messages are narrower (a scalar report, a d-dimensional vector), compare with the transformer anyway and
  attribute the gap: information lost by the messages (the best any predictor can do from those messages,
  versus Bayes) and aggregation error (the aggregator versus that best-from-messages ceiling). The reports
  ceiling `oracle_clearing` is that ceiling for scalar reports; for learned vector messages estimate it with a
  high-capacity predictor trained on the messages (large data; distillation from `bayes` allowed, declared):
  `transformer.decompose_regret(forecast, msgs_train, msgs_eval, train_batch, eval_batch)` returns `total`,
  `information_loss` and `aggregation_error` (paired CI). Its predictor estimates the ceiling from below, so
  report its budget with the split.

## Structure zoo (added 2026-10-21, `aggbench/zoo.py`)

Kelvin: "it needs to converge near the full best bayes for all kinds of information structures we can think
of." E1-E4 cover overlapping evidence, shared noise, adversaries and rollouts. The zoo adds structures they
miss. Every zoo environment except Z8 (linear-Gaussian, below) is a finite world: a latent state w with a known prior, a binary target Y = h(w),
and agents that each observe one symbol and are conditionally independent given w (all dependence goes through
w). Observations come from a physical simulator (continuous noise then binned, Bernoulli draws); the likelihood
tables P(o_i | w) are closed forms of the same parameters. So all three quantities are exact finite sums:
`local` (what agent i reports: log-odds of Y given its symbol), `bayes` (given all symbols) and
`oracle_clearing` (given the N reports and the true structure: symbols with the same report are merged, then
the same sum). Same `Batch`/`View`/`agent_inputs` interface, `kind="bernoulli"`, keys at none/noisy/full
(noisy: one-hot bits flipped with probability 0.15, parameters jittered).

How each ceiling is verified (`tests/aggbench/test_zoo.py`, and `verify.brute_force_tabular` in every
`oracle_check`): (1) plain-Python enumeration over w of the local reports, Bayes and the reports ceiling
(agree to 1e-15); (2) the tables against 200k draws of the physical simulator: P(w), P(w, o_i) and P(w, o_i,
o_j) (conditional independence) within 5e-3, and a wrong table fails this check; (3) calibration of `bayes` on
60k simulated draws per decile of p; (4) closed forms: Z1 group sums, Z4 hypothesis reports E[q^2]/E[q], Z5
full joint enumeration over 8 x 2^5 cells, Z6 parity formula; (5) bayes <= reports ceiling <= every report
baseline.

- **Z1 common cause** (`common-cause`). Y -> Z_g -> agents: 3 latent parents, P(Z_g = Y) = rho_g in
  [0.6, 0.95]; each of 6 agents hangs off a random parent (or, with probability 0.2, directly off Y) and reads
  x ~ N(s_i (2 u - 1), 1), binned. Agents of a group are correlated through the shared parent, not through
  shared noise: no matter how many agree, they cannot push the posterior past log(rho/(1-rho)). Reports are
  invertible, so the reports ceiling equals Bayes; the loss is all in aggregation. Keys: parent one-hot
  (slot-indexed), direct flag, rho of the parent, s_i. Held-out variant `Z1-wide`: 10 agents, 2 groups.
- **Z2 nonlinear** (`nonlinear`). theta on a 40-point grid with a discretized N(0, 1) prior, Y = 1[theta > 0].
  Readings are linear (a theta + noise), thresholded (1[theta + noise > c]) or magnitude-only (|theta| +
  noise). A magnitude reading says nothing about the sign alone (its report is exactly the prior) but sharpens
  the others, so scalar reports lose information here (reports ceiling > Bayes).
- **Z3 heavy tail** (`heavy-tail`, held out). Same theta; every agent reads theta plus contaminated-Gaussian
  noise (1 - p_i) N(0, sd_i^2) + p_i N(0, (10 sd_i)^2), p_i in [0, 0.3]. Agents are conditionally independent
  given theta, not given Y; one wild reading can dominate a naive pool.
- **Z4 hypothesis samples** (`hypothesis-samples`). Two kinds of sources. A hypothesis agent attached to model m
  reports a sample h ~ Bernoulli(q_m) of the world, where q_m = P(Y | D_m) is model m's posterior given its
  private evidence D_m: mixture semantics, many samples of one model estimate q_m and do not add up. An evidence
  agent reads independent evidence about Y and should be added. 8 agents, 2 models, 60% hypothesis agents.
- **Z5 multi-question** (`multi-question`). Three binary sub-questions (A, B, C) with a joint redrawn every
  episode (an Ising model, in every agent's key); target T = A and (B or C). Each agent reads one sub-question
  with accuracy in [0.6, 0.95]. Held-out variant `Z5-majority`: T = majority(A, B, C).
- **Z6 complementary / XOR** (`complementary-xor`). Y = B_1 xor B_2 with uniform bits; each agent reads one bit.
  Every report equals the prior exactly, so the reports ceiling is the prior: no report-only aggregator can do
  anything, and the information is only in observations plus keys (messages). Held-out variant `Z6-parity3`:
  parity of 3 bits, 9 agents.
- **Z8 hidden provenance** (`hidden-provenance`). E2's evidence pool (K = 12 pieces, 5 agents reading 2-6 each),
  but agent i reads piece k through private noise, r_ik = x_k + 0.5 eps_ik, and strengths take the values 0.2,
  0.4, 0.6. Agents sharing a piece see different numbers and look like agents reading different pieces, so
  overlap must be inferred. Linear-Gaussian, not tabular: Bayes = l0 + sum_k 2 mu_k (sum of the readings of
  piece k) / (n_k + tau^2); the reports ceiling is the Gaussian posterior given the reports. Checked by the full
  joint Gaussian density of all readings and of the reports (`verify.brute_force_private_read`) and by a logistic
  regression on 300k simulated draws that recovers the ceiling's weights. Variants with the same property:
  `Z1-blind` and `Z4-blind` (see "Own sensor model versus provenance").
- **Z9 k-bit parity** (`k-parity`; added 2026-10-21 because a one-layer model solves Z6's 2-bit parity). Y = XOR
  of k uniform bits (k = 4; `make_env("Z9", n_bits=3 | 6)`), plus 2 irrelevant distractor bits. Every parity bit is
  read by 3 traders (`readers_per_bit`) with accuracy in [0.85, 0.97] (`accuracy_range`), and 2 more traders read
  distractor bits (k = 4: 14 traders). No single reading informs Y, and for k >= 3 no PAIR of readings does either
  (tested exactly from the tables), so only combining readings of all k bits helps: a test of multi-hop
  combination and of attention depth. Exact Bayes is the parity of the per-bit marginals,
  (1 - prod_j (1 - 2 p_j)) / 2 (tested against brute force for k = 3, 4, 6). Prior-to-Bayes gap (mean KL, the
  prior's regret): 0.50, 0.45, 0.37 nats at k = 3, 4, 6, so a model that solves parity is separable from one that
  outputs the prior. `Z9-hard` keeps the first, noisier setting (12 traders, the first k cover the bits, the rest
  read random bits or, with probability 0.2, distractors; accuracy in [0.65, 0.95]); its gap is 0.20, 0.08, 0.01 at
  k = 3, 4, 6, too small at k >= 4 to separate parity from the prior. Own model / key: one-hot of the bit read and
  the accuracy. Note: Z9 is open, so it overlaps the held-out `Z6-parity3` (both parity); the family-level held-out
  split is Z3 and Z7.
- **Z7 context-dependent reliability** (`context-reliability`, held out). A latent regime C (3 values); agent
  i's reading strength s_i[C] in [-0.5, 2] depends on it (good in one regime, useless or misleading in
  another); half the agents also read a noisy regime cue. Bayes infers the regime from everyone jointly.

Held-out structures (`zoo.HELDOUT`: whole families Z3 and Z7, plus `Z1-wide`, `Z5-majority`, `Z6-parity3`) are
for a confirmatory round that tests generalization ACROSS structures: develop on E1-E4 and the open zoo
environments, then evaluate once on the held-out ones. `make_env` refuses them without `allow_heldout=True`;
their baseline rows below are for reference, do not tune on them. `zoo.list_envs()` lists every environment
with its `family` and `heldout` tag.

Scope (Kelvin, 2026-10-20): "We need to solve all the issues that appear on these toy examples before we expand to more complicated settings."

## Sequential aggregation (added 2026-10-21, `aggbench/sequential.py`)

Kelvin: "Each of the agent observes, reports to the CEO sequentially. The CEO (aggregation) updates and gives a
"summary of the consensus" to the next agent. ... The only challenge is the info-bottleneck there the messages may
be smaller than the full observations maybe." On views: "I like the idea of the agents doing the queries, and the
consensus answering the queries."

**Protocol.** The consensus memory starts empty. Agents arrive in a random order, drawn from the episode seed
(`sample_orders(seeds, N, order_seed)`) and recorded in the trace. Agent t:

1. builds a query from its own inputs (`query(inputs, i, t)`);
2. receives a view of width `w_view` from the consensus (`view(state, query, t)`);
3. sends a message of width `w_msg` computed from its own inputs and its view only (`encode(inputs, i, view, t)`).

Then the CEO updates its memory (`update(state, message, t)`). After the last agent, `readout(state)` gives the
posterior (`{"logit"}` or `{"mean", "prec"}`). The modules are plain callables, so torch `nn.Module`s work. The
runner (`run_protocol`) enforces the declared widths. CEO-side modules never receive inputs. Agent-side modules
receive every agent's inputs plus the per-episode agent index `i`, and must read row `i` only. This is so the
locality check can perturb the other agents. Inputs for learned agents: `agent_inputs_dict(batch, visibility,
"unaligned")`. Headline runs use unaligned inputs with keys at `none` or `noisy`.

**Knobs.** `w_msg` is the message width (the information bottleneck). `w_view` is the view width. The mode is
**queried** or **broadcast**:

- Queried (the default): the agent supplies a query and the consensus answers it by attention, `AttentionView`
  (softmax(q K^T / sqrt(d_k)) V, then an optional projection to `w_view`).
- Broadcast (the comparison): `query=None`, and every agent gets the same summary (`MeanView`).

The transformer mapping:

| protocol | transformer |
|---|---|
| consensus memory | KV cache |
| agent's query | cross-attention read |
| message | new KV entry (`AppendKV(d_k)`: its first `d_k` columns are the key, the rest the value) |

`linear_protocol` is an untrained template of these modules. A real-valued message of width w is only a
bottleneck for smooth learned encoders: a float can hide many numbers. Quantize messages, or report widths
alongside the encoder class.

**Locality.** `check_sequential_locality(protocol, inputs, order)` runs two checks and raises on a violation:

1. Message and query locality. At a position t, every other agent's inputs are perturbed, and agent t's query and
   message are recomputed from the same recorded view. They must not change.
2. Causality. The agents after position t are perturbed. Every query, view and message at positions <= t must stay
   unchanged.

Record its result in the run's JSON.

**References** (none is learned):

- `exact_sequential_bayes`, on `exact_inputs(batch)`. Messages are unbounded: each agent sends its whole
  likelihood over a latent world that makes agents conditionally independent. In E1 that is the natural parameters
  over (theta, eta). In E2/E3 it is its pieces' LLRs, keyed by piece identity. In the zoo it is log P(o_i | w) for
  every w. The CEO keeps every message, and the readout combines them. It equals `bayes` in every order on E1,
  E2, E3 and the zoo (tested to 1e-9); a mutated readout fails the test. It assumes each agent knows its own
  observation model and its pieces' identities. It is a ceiling. E4 is not supported.
- `coarse_action_cascade`, a cascade-prone baseline after Bikhchandani-Hirshleifer-Welch. The broadcast view is
  the public log-odds. Each agent sends one action: the sign of public plus its own log-odds. The CEO updates by
  the action's likelihood ratio (`ActionModel`, fitted on train; it ignores overlap between agents).
- `scalar_report_sequential`, the **scalar-report baseline** (labelled; it is not the design): each agent sends
  its local log-odds and the CEO adds them. `oracle_clearing` stays the non-sequential scalar-report ceiling.

**Metrics.**

- `regret`: regret to `bayes`, with 95% bootstrap CIs.
- `cascade_curve(protocols, make_inputs, batch)`, the cascade metric. For each position t, the private input of
  the agent arriving at t is resampled given the latent world (`resample_private`), and every protocol is rerun in
  the same order:
  - In E1 the private input is its private noise. In E2/E3 it is the pieces only that agent sees. In the zoo it is
    its symbol, drawn from P(o | w).
  - `influence` is E|delta final log-odds|; in E1 it is sqrt(2 KL).
  - `retention` = influence / influence of the exact reference.
  - `message_retention` = the fraction of episodes in which the agent's own message changed, relative to the exact
    reference.
  - Also reported: KL and the regret increase.

  A cascade shows as retention falling toward 0 with position.
- `order_sensitivity`: the variance of the final posterior across random orders (0 for exact Bayes).
- `full_context_bar(env, visibility=..., alignment=..., n_train=...)`: Kelvin's bar. It calls
  `aggbench.transformer.evaluate_full_context_transformer` in the same setting as the protocol run (it trains).
  The reference table below does not include it.

`python3 -m aggbench.sequential --env E2 Z1 --n 2000` prints the reference table:

| env | protocol | w_msg | w_view | regret [95% CI] | order variance | retention, position 1..N | message retention, position 1..N |
|---|---|---|---|---|---|---|---|
| E2 | exact_sequential_bayes | 24 | 0 | 0.000 [-0.000, 0.000] | 2e-31 | 1.00 at every position | 1.00 at every position |
| E2 | coarse_action_cascade | 1 | 1 | 0.194 [0.166, 0.223] | 1.17 | 0.48 0.24 0.19 0.14 0.09 | 0.19 0.08 0.06 0.04 0.03 |
| E2 | scalar_report_sequential (scalar-report baseline) | 1 | 0 | 0.132 [0.099, 0.166] | 1e-30 | 1.00 at every position | 1.00 at every position |
| E2 | oracle_clearing (scalar-report ceiling, not sequential) | 1 | - | 0.017 [0.009, 0.024] | - | - | - |
| Z1 | exact_sequential_bayes | 16 | 0 | -0.000 [-0.000, 0.000] | 4e-30 | 1.00 at every position | 1.00 at every position |
| Z1 | coarse_action_cascade | 1 | 1 | 0.183 [0.160, 0.206] | 0.90 | 0.38 0.19 0.16 0.13 0.14 0.07 | 0.22 0.09 0.06 0.04 0.04 0.02 |
| Z1 | scalar_report_sequential (scalar-report baseline) | 1 | 0 | 0.053 [0.036, 0.072] | 1e-30 | 1.27-1.39 | 1.00 at every position |
| Z1 | oracle_clearing (scalar-report ceiling, not sequential) | 1 | - | 0.000 [0.000, 0.000] | - | - | - |

These are 2000 dev episodes; the coarse-action model is fitted on 2000 train episodes. Coarse actions lose half
of the first agent's information. Late agents' messages stop depending on what they saw (message retention falls
to 0.02-0.03), and the final answer depends on the order. This is the cascade the metric is built to show.

## Trader transformer (added 2026-10-21, `aggbench/trader_transformer.py`)

Kelvin: "(1) first, imagine that each information each 'trader' see is just one token. (2) second, imagine that what
it does is just the same as any transformer does when looking at a token. It also has self-attention over all
previous 'traders' (runs of the same network) over the previous tokens. (3) at the final layer of the last run, it
predicts something".

The model is a plain pre-norm decoder-only transformer. Each trader is one token, built by `transformer.full_context`
with the agent's own sensor model, unaligned, and the provenance key at `none`, `noisy` or `full`. Traders arrive
in a random order per episode (`sequential.sample_orders`, seeded and recorded in `TraderContext.order`). A causal
mask runs over traders. The prediction is read from the final layer at the last trader's position. There is no
positional encoding unless `pos="learned"`. Bernoulli environments only.

Knobs for Kelvin's questions (A), (B) and (C):

- `attn_layers` (A): the layers where traders read earlier traders: `"all"`, `"last"`, `"first"`, `"every_other"`
  (ending with the last layer), `"none"`, or a tuple of indices. The other layers are MLP-only on each trader's own
  residual stream.
- `loss` (B):
  - `"final"`: the last position only.
  - `"every_prefix"`: every position predicts the outcome, as in next-token training. The target is still the
    outcome.
  - `"distill"`: every position is fitted to the prefix Bayes posterior. It uses the oracle and is recorded as
    `uses_oracle`.
- `token_state` (C):
  - `"kv"`: causal self-attention over earlier traders.
  - `"recurrent"`: no cross-trader attention. A GRU running summary passes in the trader direction, as in the
    CEO-summary design.
  - `"feedback"`: Feedback Transformer (Fan et al. 2020). Each reading layer of trader t attends over one memory
    per earlier trader, a learned mix of all that trader's layer states.

- `layout`, Kelvin's 3-D version (depth, token index, trader index). His rule: "each info path goes through one
  specific layer only once".
  - `"trader"` (default): one token per trader.
  - The grid layouts give each trader one token per evidence piece it sees. They support E2/E3 only. Cell (t, r)
    is token t of the trader at position r. The computation is the standard transformer; only the mask changes.
  - `"grid"`: inside each attention layer, `token_heads` heads (default half) read along the token axis (same
    trader, t' <= t). The other heads read along the trader axis (same token index, r' <= r). Kelvin: "it senough
    to see either the t' <= t; or the r' <= r, even with different heads etc." A diagonal path takes two layers.
  - `"axial"` (comparison): alternate layers read along one axis.
  - `"grid_full"` (comparison): the full cone t' <= t, r' <= r.
  - Readout is at the last cell. Prefix predictions are read at each trader's last token slot.
  - Caveat: a trader-axis head links cells by the local piece rank t (t' == t), as Kelvin's rule states. With
    unaligned pieces, rank t says nothing about which piece it is: token t of two traders is usually a different
    piece. `grid_full` (the full cone) is the comparison that does not depend on rank.
- `trader_encoder` (layout `"trader"` only): `"linear"` (default) puts the trader's whole token through one Linear.
  `"deepsets"` (E2/E3) applies a shared encoder to each piece cell and sums over the trader's present pieces. Build
  the context with `trader_context(..., trader_encoder="deepsets")`.
- `device`: None means cuda when available, else cpu. `threads` (default 2) applies on cpu only. The module trains
  on the device and is back on the CPU after `fit`.

```python
from aggbench.trader_transformer import evaluate_trader_transformer, trader_context, TraderTransformer
rec = evaluate_trader_transformer(make_env("E2"), visibility="noisy", attn_layers="last", loss="every_prefix",
                                  token_state="kv", threads=2)
rec["methods"]      # trained arm and both controls, regret to Bayes
rec["prefix_kl"]    # mean KL(prefix Bayes || prediction) at each position
```

Tests (`tests/aggbench/test_trader_transformer.py`):

- Causal locality for every token state, attention set and position option: perturbing trader j > t leaves
  positions <= t unchanged.
- With `"none"`, nothing crosses between traders.
- The order is seeded and recorded.
- Prefix Bayes ends at `bayes`.
- The loss knob: only `every_prefix` puts gradient on early positions.
- Grid cone locality for all three grid layouts: perturbing cells outside {t' <= t, r' <= r} leaves cell (t, r)
  unchanged.
- In `grid` and `axial`, diagonal paths need two layers.
- The `token_heads` knob.
- A CPU smoke run on E2, for the trader layout and for `grid`.

Smoke runs give no verdicts. This section makes no regret claims.

## Literature methods (added 2026-10-21, `aggbench/external/`)

Kelvin: "we should seek to reimplement as little as possible and just apply the results directly to our problems."
So the established methods run here from their official code where it exists, with only a thin adapter to the
aggbench inputs and result schema, and from their published equations where no code exists. Provenance is in
`aggbench.external.SOURCES`.

| method | code | version / commit | licence | what is ours |
|---|---|---|---|---|
| TabPFN (Hollmann et al. 2025, doi:10.1038/s41586-024-08328-6; PFN, arXiv:2112.10510) | PriorLabs pip `tabpfn` | 9.2.0, `ModelVersion.V2` weights (the v2.5+ weights need a PriorLabs licence token; v2 downloads without one) | Apache-2.0 | row encoding, controls (`tabpfn_adapter.py`) |
| Set Transformer (Lee et al. 2019, arXiv:1810.00825) | authors' repo juho-lee/set_transformer, `modules.py` + `models.py`, vendored | `73432c640ac78140496d6738416c54d32c686d65` | MIT | token adapter, output head, training loop |
| Deep Sets (Zaheer et al. 2017, arXiv:1703.06114) | `DeepSet` class of the same repo (mean pooling) | same | MIT | same |
| TarMAC (Das et al. 2019, arXiv:1810.11187) | no official code (GitHub searched 2026-10-21); attention step = `MessageAggregator` of MATE (XuehaiPan/mate, `examples/tarmac/models.py`), vendored | `3e631c0c3b043990fc53ae5fc3a37b0f65f230c5` | MIT | encoder, GRU agent loop, rounds, readout (paper's description) |
| PIF Gaussian oracle (Satopaa et al., arXiv:1505.06472) | equations: Sec. 2.2.1 eq. (1), Prop. 2.2 iii, Sec. 2.2.2 revealed aggregator diag(Sigma)' Sigma^-1 (X - mu0) + mu0 | - | - | Sigma from keys (information overlap), or fitted from train reports alone at keys `none` |
| Winkler GLS (doi:10.1287/mnsc.27.4.479; Clemen & Winkler doi:10.1287/opre.33.2.427) | equations (w = S^-1 1 / 1' S^-1 1), S fitted on train; E1 only | - | - | formula from the literature map, not re-checked against the paper |
| Covariance intersection (Julier & Uhlmann 1997, doi:10.1109/ACC.1997.609105) | equations, N-source form of Noack et al. 2017 eq. (10), det criterion; E1 only | - | - | - |
| Inverse covariance intersection (Noack et al. 2017, doi:10.1016/j.automatica.2017.01.019) | equations, Theorem 7, eqs. (17)-(18), det criterion, sequential over agents; E1 only | - | - | - |
| Extremizing (Satopaa et al. 2014, doi:10.1016/j.ijforecast.2013.09.009) | `baselines.extremized` (already in aggbench) | - | - | - |
| Surprisingly popular (Prelec et al. 2017, doi:10.1038/nature21054) | equations, `closed_form.surprisingly_popular`, toy test only: aggbench has no meta-prediction field yet | - | - | - |

Not applied, with the reason:

- **AVICI** (Lorch et al. 2022, arXiv:2205.12934; pip `avici` 1.0.7, larslorch/avici, MIT). AVICI infers a causal
  graph over d observed variables from many i.i.d. samples of one fixed causal model. aggbench redraws the
  information structure every episode, so each structure yields one sample (one row of reports), and the sources
  are not linked by edges between them: they depend on each other through hidden shared evidence pieces and
  noise factors, a hidden-confounder structure outside the class AVICI is trained on. Applying it needs a
  fixed-structure variant with many episodes per structure; that is a separate probe, not a baseline here.
- **Neural enhanced BP** (Garcia Satorras & Welling 2021, arXiv:2003.01998): no public official code was found
  (the authors' GitHub has `hybrid-inference`, the code of the related arXiv:1906.02547). A later item.
- **PFNs codebase** (automl/PFNs, now SamuelGabriel/PFNs, Apache-2.0): training a PFN on the aggbench prior is what
  the learned rows do with the official set architectures; the codebase itself is not used.

**Inputs.** The learned methods read RICH inputs by default, exactly the tokens `full_context_transformer` reads:
`aggbench.transformer.full_context(batch, visibility, alignment)` (each agent's raw observation, present mask, own
sensor model, provenance key and report, one token per agent). Deep Sets' per-agent encoder and TarMAC's first-round
messages depend on the agent's own token only (`messages.check_locality`, recorded in each result JSON); the Set
Transformer and TabPFN read all agents' tokens jointly (TabPFN: one row per episode, agents sorted by report).
Headline settings are `alignment="unaligned"` with provenance keys `none` and `noisy`; `aligned` rows are a giveaway
control. E3-biased is left out of the rich rows. The scalar-report rows (reports only) are kept as baselines,
labelled as such; the closed-form rules are aggregation rules over reports by definition.

**Training regimes** (protocol owner's decision, 2026-10-21, for every arm in every lane). Every trained arm uses the
optimizer of `full_context_transformer` (AdamW lr 1e-3, weight decay 0.01, OneCycle with 5% warm-up, 3000 steps of
512, gradient clip 1.0); the official architectures are used as published (no dropout added).

- **fresh** (headline columns): `aggbench.transformer.FreshStream`, a new minibatch of simulator episodes every
  step, never reused (PFN-style). The minibatches of a cell are generated once and shared by its architectures.
- **pool** (secondary, `*_pool` columns): the fixed 20k train pool, with early stopping on a held-out 10% of TRAIN
  (scored every 100 steps, patience 1000), never on dev.
- TabPFN is in-context (no training): the first 10000 train episodes are its context in both regimes.

A first equal-budget run reused the 20k pool for 3000 steps with no early stopping. Every rich learned arm memorised
the pool, and so did `full_context_transformer` run unmodified (E2, noisy keys, unaligned): dev regret 1.004 against
0.49 for its untrained init and 0.41 for the random network, with training loss going from 0.20 to 0.0005. Deep
Sets on the same inputs had train regret -0.27 and dev regret 1.85. That run appears at the end of this section,
labelled "memorised; superseded".

**Check.** With exact (`full`) keys the PIF oracle equals `oracle_clearing` in E2, E3-sybil and E3-collude, and
equals Bayes in E1 with aligned and unaligned keys (agreement to 1e-8, `tests/aggbench/test_external.py`). It
differs in E3-biased, whose keys do not reveal the distortion.

**Run.** The environment is a uv venv: torch 2.14.1+cpu, tabpfn 9.2.0, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1 (uv 0.13.4, Python 3.12.3).

```bash
PY=/home/agent/.venvs/aggext/bin/python
$PY -m aggbench.external.report --all --part closed --out /tmp/aggext-runs --jobs 4    # numpy, CPU
$PY -m aggbench.external.report --all --part learned --out /tmp/aggext-runs --jobs 2   # torch + tabpfn, GPU
$PY -m aggbench.external.report --collect /tmp/aggext-runs --write                     # merges the parts
$PY -m pytest -q tests/aggbench/test_external.py tests/aggbench/test_external_learned.py
```

The learned parts ran on spot L4 GPUs (`/home/agent/workspaces/mats/work/run_on_l4.sh`), the closed-form parts on
General, all at harness commit `71eaeff` (before the rebase onto main; kept on branch `aggext`, harness sha256 in
each result). The `full_context_transformer` columns are read from main's caches at write time.

<!-- EXTERNAL:BEGIN -->
**Rich inputs, fresh regime (headline)** (`aggbench.transformer.full_context` tokens: each agent's raw observation, own sensor model, provenance key and report; `*-unaligned` is the headline, `*-aligned` a giveaway control):

| env | setting | extremized_log_pool | oracle_clearing | pif_oracle | full_context_transformer | tabpfn | deepsets | set_transformer | tarmac_r1 | tarmac_r3 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | none-unaligned | 0.827 [0.770, 0.884] | -0.000 [-0.000, 0.000] | n/a | n/a | 2.118 [2.031, 2.208] | 2.166 [2.089, 2.242] | 0.951 [0.893, 1.010] | 1.284 [1.217, 1.349] | 1.146 [1.084, 1.208] |
| E1 | noisy-unaligned | 0.827 [0.770, 0.884] | -0.000 [-0.000, 0.000] | 3.616 [3.332, 3.901] | n/a | 2.372 [2.282, 2.462] | 2.161 [2.083, 2.235] | 0.961 [0.900, 1.020] | 1.308 [1.240, 1.373] | 1.166 [1.100, 1.227] |
| E1 | none-aligned | 0.827 [0.770, 0.884] | -0.000 [-0.000, 0.000] | n/a | n/a | n/a | 2.168 [2.091, 2.245] | 0.949 [0.890, 1.006] | 1.295 [1.229, 1.358] | 1.148 [1.087, 1.209] |
| E1 | noisy-aligned | 0.827 [0.770, 0.884] | -0.000 [-0.000, 0.000] | 3.616 [3.332, 3.901] | n/a | n/a | 2.161 [2.085, 2.237] | 0.961 [0.899, 1.020] | 1.306 [1.239, 1.371] | 1.160 [1.097, 1.220] |
| E2 | none-unaligned | 0.045 [0.033, 0.056] | 0.017 [0.009, 0.024] | n/a | n/a | 0.043 [0.030, 0.056] | 0.039 [0.028, 0.050] | 0.035 [0.025, 0.045] | 0.034 [0.024, 0.044] | 0.034 [0.024, 0.043] |
| E2 | noisy-unaligned | 0.045 [0.033, 0.056] | 0.017 [0.009, 0.024] | 1.771 [0.923, 2.917] | n/a | 0.044 [0.031, 0.057] | 0.040 [0.029, 0.051] | 0.035 [0.025, 0.045] | 0.036 [0.026, 0.046] | 0.036 [0.026, 0.046] |
| E2 | none-aligned | 0.045 [0.033, 0.056] | 0.017 [0.009, 0.024] | n/a | n/a | n/a | 0.037 [0.026, 0.048] | 0.009 [0.004, 0.014] | 0.034 [0.024, 0.044] | 0.032 [0.022, 0.041] |
| E2 | noisy-aligned | 0.045 [0.033, 0.056] | 0.017 [0.009, 0.024] | 0.676 [0.490, 0.880] | n/a | n/a | 0.038 [0.027, 0.049] | 0.006 [-0.000, 0.011] | 0.034 [0.024, 0.044] | 0.035 [0.024, 0.045] |
| E3-sybil | none-unaligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | n/a | n/a | 0.045 [0.032, 0.058] | 0.047 [0.035, 0.058] | 0.040 [0.029, 0.050] | 0.059 [0.047, 0.072] | 0.040 [0.029, 0.050] |
| E3-sybil | noisy-unaligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | 2.311 [1.498, 3.432] | n/a | 0.047 [0.034, 0.060] | 0.047 [0.036, 0.058] | 0.039 [0.028, 0.049] | 0.051 [0.038, 0.063] | 0.041 [0.030, 0.051] |
| E3-sybil | none-aligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | n/a | n/a | n/a | 0.049 [0.037, 0.060] | 0.011 [0.006, 0.017] | 0.059 [0.046, 0.071] | 0.039 [0.029, 0.050] |
| E3-sybil | noisy-aligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | 3.656 [1.469, 8.490] | n/a | n/a | 0.048 [0.036, 0.059] | 0.006 [0.000, 0.012] | 0.049 [0.036, 0.062] | 0.038 [0.028, 0.048] |
| E3-collude | none-unaligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | n/a | n/a | 0.028 [0.018, 0.038] | 0.029 [0.019, 0.038] | 0.025 [0.015, 0.033] | 0.022 [0.014, 0.030] | 0.025 [0.016, 0.034] |
| E3-collude | noisy-unaligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | 3.827 [0.891, 9.182] | n/a | 0.029 [0.019, 0.039] | 0.029 [0.019, 0.038] | 0.025 [0.015, 0.034] | 0.026 [0.016, 0.034] | 0.024 [0.016, 0.033] |
| E3-collude | none-aligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | n/a | n/a | n/a | 0.029 [0.019, 0.038] | 0.008 [0.003, 0.014] | 0.024 [0.015, 0.032] | 0.024 [0.015, 0.032] |
| E3-collude | noisy-aligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | 2.000 [0.736, 3.804] | n/a | n/a | 0.028 [0.018, 0.037] | 0.008 [0.002, 0.014] | 0.025 [0.015, 0.034] | 0.025 [0.016, 0.033] |

Controls: regret of the random-network / untrained-init control (TabPFN: random network / shuffled-label context).

| env | setting | tabpfn | deepsets | set_transformer | tarmac_r1 | tarmac_r3 |
|---|---|---|---|---|---|---|
| E1 | none-unaligned | 1356.699 / 5.431 | 4.814 / 4.897 | 4.969 / 4.975 | 5.033 / 4.957 | 4.917 / 4.948 |
| E1 | noisy-unaligned | 1284.211 / 4.458 | 4.810 / 4.870 | 5.193 / 4.944 | 5.190 / 4.962 | 5.192 / 4.982 |
| E1 | none-aligned | n/a | 4.814 / 4.897 | 4.968 / 4.976 | 5.040 / 4.955 | 4.918 / 4.946 |
| E1 | noisy-aligned | n/a | 4.812 / 4.870 | 5.197 / 4.944 | 5.199 / 4.962 | 5.204 / 4.981 |
| E2 | none-unaligned | 0.435 / 0.441 | 0.433 / 0.433 | 0.437 / 0.432 | 0.433 / 0.439 | 0.434 / 0.439 |
| E2 | noisy-unaligned | 0.433 / 0.445 | 0.436 / 0.438 | 0.431 / 0.432 | 0.432 / 0.433 | 0.431 / 0.434 |
| E2 | none-aligned | n/a | 0.433 / 0.433 | 0.438 / 0.431 | 0.433 / 0.440 | 0.433 / 0.439 |
| E2 | noisy-aligned | n/a | 0.436 / 0.438 | 0.432 / 0.432 | 0.434 / 0.434 | 0.433 / 0.435 |
| E3-sybil | none-unaligned | 0.435 / 0.440 | 0.433 / 0.433 | 0.436 / 0.432 | 0.433 / 0.440 | 0.434 / 0.439 |
| E3-sybil | noisy-unaligned | 0.438 / 0.438 | 0.436 / 0.438 | 0.430 / 0.432 | 0.432 / 0.433 | 0.432 / 0.435 |
| E3-sybil | none-aligned | n/a | 0.433 / 0.433 | 0.438 / 0.431 | 0.432 / 0.440 | 0.433 / 0.439 |
| E3-sybil | noisy-aligned | n/a | 0.436 / 0.438 | 0.431 / 0.432 | 0.434 / 0.433 | 0.432 / 0.435 |
| E3-collude | none-unaligned | 0.433 / 0.436 | 0.433 / 0.433 | 0.436 / 0.431 | 0.433 / 0.441 | 0.433 / 0.441 |
| E3-collude | noisy-unaligned | 0.433 / 0.435 | 0.434 / 0.438 | 0.431 / 0.432 | 0.433 / 0.432 | 0.432 / 0.434 |
| E3-collude | none-aligned | n/a | 0.433 / 0.433 | 0.438 / 0.429 | 0.432 / 0.442 | 0.433 / 0.441 |
| E3-collude | noisy-aligned | n/a | 0.434 / 0.438 | 0.431 / 0.433 | 0.435 / 0.433 | 0.434 / 0.435 |

**Rich inputs, pool regime (secondary)** (20k train pool, early stopping on 10% of train):

| env | setting | full_context_transformer_pool | deepsets_pool | set_transformer_pool | tarmac_r1_pool | tarmac_r3_pool |
|---|---|---|---|---|---|---|
| E1 | none-unaligned | 1.459 [1.381, 1.534] | 2.336 [2.251, 2.416] | 1.576 [1.488, 1.663] | 1.706 [1.622, 1.787] | 1.610 [1.532, 1.690] |
| E1 | noisy-unaligned | 1.490 [1.409, 1.572] | 2.347 [2.263, 2.431] | 1.758 [1.656, 1.854] | 1.750 [1.668, 1.831] | 1.709 [1.625, 1.793] |
| E1 | none-aligned | n/a | 2.323 [2.239, 2.405] | 1.629 [1.532, 1.724] | 1.723 [1.633, 1.811] | 1.634 [1.550, 1.719] |
| E1 | noisy-aligned | n/a | 2.355 [2.269, 2.437] | 1.693 [1.599, 1.786] | 1.777 [1.691, 1.856] | 1.718 [1.642, 1.796] |
| E2 | none-unaligned | 0.051 [0.037, 0.066] | 0.055 [0.041, 0.069] | 0.060 [0.045, 0.074] | 0.052 [0.038, 0.066] | 0.053 [0.039, 0.067] |
| E2 | noisy-unaligned | 0.072 [0.056, 0.087] | 0.071 [0.055, 0.087] | 0.061 [0.047, 0.075] | 0.055 [0.041, 0.069] | 0.057 [0.043, 0.072] |
| E2 | none-aligned | n/a | 0.057 [0.043, 0.070] | 0.054 [0.040, 0.068] | 0.053 [0.040, 0.065] | 0.054 [0.041, 0.067] |
| E2 | noisy-aligned | n/a | 0.073 [0.056, 0.089] | 0.048 [0.036, 0.059] | 0.060 [0.046, 0.073] | 0.065 [0.050, 0.079] |
| E3-sybil | none-unaligned | 0.068 [0.054, 0.082] | 0.070 [0.055, 0.084] | 0.066 [0.050, 0.082] | 0.083 [0.069, 0.098] | 0.068 [0.053, 0.083] |
| E3-sybil | noisy-unaligned | 0.069 [0.055, 0.083] | 0.087 [0.069, 0.105] | 0.067 [0.052, 0.083] | 0.073 [0.056, 0.088] | 0.083 [0.065, 0.101] |
| E3-sybil | none-aligned | n/a | 0.082 [0.066, 0.098] | 0.056 [0.043, 0.069] | 0.080 [0.065, 0.095] | 0.067 [0.053, 0.080] |
| E3-sybil | noisy-aligned | n/a | 0.096 [0.078, 0.113] | 0.058 [0.043, 0.073] | 0.072 [0.057, 0.087] | 0.072 [0.056, 0.087] |
| E3-collude | none-unaligned | 0.048 [0.036, 0.060] | 0.037 [0.025, 0.048] | 0.042 [0.031, 0.054] | 0.041 [0.029, 0.053] | 0.042 [0.029, 0.053] |
| E3-collude | noisy-unaligned | 0.043 [0.031, 0.055] | 0.041 [0.029, 0.053] | 0.045 [0.034, 0.057] | 0.046 [0.033, 0.059] | 0.045 [0.032, 0.059] |
| E3-collude | none-aligned | n/a | 0.047 [0.034, 0.060] | 0.044 [0.032, 0.056] | 0.044 [0.032, 0.056] | 0.052 [0.039, 0.064] |
| E3-collude | noisy-aligned | n/a | 0.046 [0.033, 0.058] | 0.046 [0.034, 0.057] | 0.049 [0.036, 0.062] | 0.047 [0.035, 0.060] |

Controls: regret of the random-network / untrained-init control (TabPFN: random network / shuffled-label context).

| env | setting | deepsets_pool | set_transformer_pool | tarmac_r1_pool | tarmac_r3_pool |
|---|---|---|---|---|---|
| E1 | none-unaligned | 4.819 / 4.897 | 5.028 / 4.975 | 5.011 / 4.957 | 4.898 / 4.948 |
| E1 | noisy-unaligned | 4.829 / 4.870 | 5.251 / 4.944 | 5.210 / 4.962 | 5.158 / 4.982 |
| E1 | none-aligned | 4.813 / 4.897 | 5.015 / 4.976 | 5.022 / 4.955 | 4.891 / 4.946 |
| E1 | noisy-aligned | 4.835 / 4.870 | 5.250 / 4.944 | 5.174 / 4.962 | 5.129 / 4.981 |
| E2 | none-unaligned | 0.433 / 0.433 | 0.441 / 0.432 | 0.433 / 0.439 | 0.434 / 0.439 |
| E2 | noisy-unaligned | 0.435 / 0.438 | 0.430 / 0.432 | 0.433 / 0.433 | 0.432 / 0.434 |
| E2 | none-aligned | 0.433 / 0.433 | 0.441 / 0.431 | 0.433 / 0.440 | 0.434 / 0.439 |
| E2 | noisy-aligned | 0.435 / 0.438 | 0.430 / 0.432 | 0.434 / 0.434 | 0.433 / 0.435 |
| E3-sybil | none-unaligned | 0.433 / 0.433 | 0.439 / 0.432 | 0.433 / 0.440 | 0.434 / 0.439 |
| E3-sybil | noisy-unaligned | 0.436 / 0.438 | 0.429 / 0.432 | 0.433 / 0.433 | 0.432 / 0.435 |
| E3-sybil | none-aligned | 0.433 / 0.433 | 0.441 / 0.431 | 0.432 / 0.440 | 0.433 / 0.439 |
| E3-sybil | noisy-aligned | 0.435 / 0.438 | 0.430 / 0.432 | 0.434 / 0.433 | 0.432 / 0.435 |
| E3-collude | none-unaligned | 0.433 / 0.433 | 0.441 / 0.431 | 0.434 / 0.441 | 0.434 / 0.441 |
| E3-collude | noisy-unaligned | 0.435 / 0.438 | 0.430 / 0.432 | 0.433 / 0.432 | 0.432 / 0.434 |
| E3-collude | none-aligned | 0.433 / 0.433 | 0.442 / 0.429 | 0.433 / 0.442 | 0.434 / 0.441 |
| E3-collude | noisy-aligned | 0.435 / 0.438 | 0.430 / 0.433 | 0.435 / 0.433 | 0.433 / 0.435 |

**Scalar-report baselines** (reports only, keys none; `set_transformer` fresh, `_pool` pool):

| env | setting | extremized_log_pool | oracle_clearing | pif_oracle | winkler_gls | covariance_intersection | inverse_covariance_intersection | set_transformer | set_transformer_pool | tabpfn |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | scalar | 0.827 [0.770, 0.884] | -0.000 [-0.000, 0.000] | 2.419 [2.338, 2.498] | 3.358 [3.273, 3.440] | 1.884 [1.824, 1.945] | 1.294 [1.240, 1.349] | 0.923 [0.863, 0.981] | 1.382 [1.307, 1.455] | 2.011 [1.924, 2.099] |
| E2 | scalar | 0.045 [0.033, 0.056] | 0.017 [0.009, 0.024] | 0.060 [0.049, 0.071] | n/a | n/a | n/a | 0.043 [0.031, 0.054] | 0.047 [0.035, 0.060] | 0.043 [0.031, 0.056] |
| E3-sybil | scalar | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | 0.089 [0.075, 0.102] | n/a | n/a | n/a | 0.043 [0.031, 0.054] | 0.048 [0.035, 0.060] | 0.047 [0.034, 0.060] |
| E3-biased | scalar | 0.076 [0.061, 0.091] | 0.017 [0.009, 0.024] | 0.094 [0.072, 0.116] | n/a | n/a | n/a | 0.051 [0.039, 0.063] | 0.053 [0.039, 0.067] | 0.051 [0.037, 0.065] |
| E3-collude | scalar | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | 0.033 [0.023, 0.042] | n/a | n/a | n/a | 0.027 [0.018, 0.036] | 0.028 [0.018, 0.038] | 0.029 [0.018, 0.039] |

Controls: regret of the random-network / untrained-init control (TabPFN: random network / shuffled-label context).

| env | setting | set_transformer | set_transformer_pool | tabpfn |
|---|---|---|---|---|
| E1 | scalar | 5.107 / 5.054 | 5.135 / 5.054 | 1494.483 / 5.112 |
| E2 | scalar | 0.413 / 0.411 | 0.408 / 0.411 | 0.436 / 0.433 |
| E3-sybil | scalar | 0.414 / 0.411 | 0.410 / 0.411 | 0.435 / 0.436 |
| E3-biased | scalar | 0.415 / 0.414 | 0.415 / 0.414 | 0.434 / 0.433 |
| E3-collude | scalar | 0.410 / 0.408 | 0.408 / 0.408 | 0.438 / 0.434 |

Mean regret to the global Bayes posterior (nats per episode), 95% bootstrap CIs over 2000 dev episodes. Learned arms use the optimizer of full_context_transformer (AdamW lr 1e-3, weight decay 0.01, OneCycle, 3000 steps of 512), on an L4 GPU. Headline regime: fresh simulator episodes every step, never reused (the same episodes for every architecture of a cell). Pool regime: the 20000-episode train pool with early stopping on 10% of it. TabPFN is in-context (no training): v2 weights, the first 10000 train episodes as context, package-default ensemble (E1 regressors: 1 estimator, a cost cap). `full_context_transformer` is main's reference (aggbench/transformer.py) in the same regime, copied from aggbench/baselines_transformer_fresh.json / baselines_transformer.json when generated. n/a: not defined for that env/setting (closed-form Gaussian rules are E1 only; PIF needs keys or the scalar fit; rich TabPFN runs only unaligned). Generated by `python -m aggbench.external.report --collect DIR --write`; harness commits: closed parts `71eaeff79d`, learned parts `71eaeff79d`; full numbers, controls, early-stopping steps and locality checks in `aggbench/external/results_default.json`.
<!-- EXTERNAL:END -->

**Prior work that scores learned aggregation against an exact posterior** (search 2026-10-21: Semantic Scholar
citations of TarMAC (597) and of the PIF papers (arXiv:1505.06472: 13; JASA doi:10.1080/01621459.2015.1100621:
56), keyword-filtered, plus 5 Exa queries; ids checked on export.arxiv.org). None was found that scores a learned
aggregator or learned communication against the exact Bayes posterior. Closest: arXiv:1802.07107 (online learning
of forecast aggregation in partial-evidence environments, regret to the Bayes-optimal aggregate; theory, no
neural learner; cites the JASA PIF paper) and arXiv:2610.19215 (transformers and LLMs exchanging calibrated
beliefs, cites TarMAC; its theory uses the full-information posterior, but no KL or regret of the learned
agents to it was found).

**Appendix: memorised; superseded.** The equal-budget run without early stopping and with reused episodes, kept as
evidence for the regime change above. Do not cite these numbers as the methods' performance. Its commits are on
branch `aggext-memorised`. Only the E3 cells finished (one GPU VM was preempted and the rest was stopped when the
regime changed); the scalar E1/E2 learned parts are in `results_memorised.json`.

<!-- EXTERNAL-MEMORISED:BEGIN -->
| env | setting | extremized_log_pool | oracle_clearing | tabpfn | deepsets | set_transformer | tarmac_r1 | tarmac_r3 |
|---|---|---|---|---|---|---|---|---|
| E3-sybil | none-unaligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | 0.045 [0.032, 0.058] | 1.620 [1.430, 1.830] | 2.601 [2.264, 2.962] | 0.505 [0.431, 0.579] | 0.941 [0.812, 1.077] |
| E3-sybil | noisy-unaligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | 0.047 [0.034, 0.060] | 2.150 [1.905, 2.412] | 2.829 [2.450, 3.219] | 0.864 [0.748, 0.984] | 1.123 [0.979, 1.276] |
| E3-sybil | none-aligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | n/a | 2.010 [1.749, 2.263] | 2.509 [2.177, 2.891] | 0.564 [0.475, 0.655] | 0.988 [0.853, 1.125] |
| E3-sybil | noisy-aligned | 0.078 [0.062, 0.092] | 0.017 [0.009, 0.024] | n/a | 2.011 [1.762, 2.261] | 2.446 [2.085, 2.813] | 0.853 [0.740, 0.979] | 1.028 [0.892, 1.169] |
| E3-collude | none-unaligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | 0.028 [0.018, 0.038] | 1.544 [1.327, 1.751] | 2.887 [2.505, 3.263] | 0.576 [0.487, 0.668] | 1.089 [0.953, 1.236] |
| E3-collude | noisy-unaligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | 0.029 [0.019, 0.039] | 1.740 [1.526, 1.963] | 2.749 [2.341, 3.169] | 0.907 [0.772, 1.047] | 1.090 [0.945, 1.241] |
| E3-collude | none-aligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | n/a | 2.100 [1.831, 2.365] | 2.488 [2.121, 2.864] | 0.649 [0.553, 0.746] | 1.049 [0.917, 1.187] |
| E3-collude | noisy-aligned | 0.030 [0.020, 0.039] | 0.018 [0.010, 0.025] | n/a | 1.978 [1.724, 2.245] | 2.316 [1.993, 2.685] | 0.828 [0.715, 0.950] | 1.060 [0.915, 1.210] |

Controls (random network / untrained init; TabPFN: random network / shuffled labels):

| env | setting | tabpfn | deepsets | set_transformer | tarmac_r1 | tarmac_r3 |
|---|---|---|---|---|---|---|
| E3-sybil | none-unaligned | 0.435 / 0.443 | 0.433 / 0.433 | 0.451 / 0.432 | 0.545 / 0.440 | 0.538 / 0.439 |
| E3-sybil | noisy-unaligned | 0.438 / 0.439 | 0.454 / 0.438 | 0.432 / 0.432 | 0.549 / 0.433 | 0.469 / 0.435 |
| E3-sybil | none-aligned | n/a | 0.433 / 0.433 | 0.446 / 0.431 | 0.667 / 0.440 | 0.571 / 0.439 |
| E3-sybil | noisy-aligned | n/a | 0.446 / 0.438 | 0.434 / 0.432 | 0.623 / 0.433 | 0.503 / 0.435 |
| E3-collude | none-unaligned | 0.433 / 0.436 | 0.433 / 0.433 | 0.452 / 0.431 | 0.702 / 0.441 | 0.576 / 0.441 |
| E3-collude | noisy-unaligned | 0.433 / 0.435 | 0.440 / 0.438 | 0.433 / 0.432 | 0.607 / 0.432 | 0.490 / 0.434 |
| E3-collude | none-aligned | n/a | 0.434 / 0.433 | 0.452 / 0.429 | 0.819 / 0.442 | 0.613 / 0.441 |
| E3-collude | noisy-aligned | n/a | 0.444 / 0.438 | 0.435 / 0.433 | 0.719 / 0.433 | 0.551 / 0.435 |

Same tokens as the main table except the own sensor model (pre-20ee6a0 full key), 20k train pool reused for 3000 AdamW steps of 512 (lr 2e-3, wd 1e-4), no early stopping, 2000 dev episodes; harness commits: closed parts `95b031c823`, learned parts `b0efd45080`, `dc1c329d22`.
<!-- EXTERNAL-MEMORISED:END -->

## State of the compression challenge (2026-10-22, end of the overnight batches)

Kelvin's target: compressed learned aggregation (each agent writes b bits into the consensus) should learn what an
uncompressed transformer reading every raw token learns, on the same data, steps and labels. Benchmark: leak-reduced
Z9-p2 (13 agents x 2 readings of a 4-bit parity; prior 0.270 nats; a count code keeps 3.9% of the information and a
searched blind 4-bit code about 20%). Gap = arm minus the same-decoder transformer trained for the same total steps
(10000 steps 0.0050, 40000 0.0010, 80000 0.0009; anchors were still slowly improving, so gaps are lower bounds).
Fresh episodes, best-val checkpoints, sha256-checked harnesses; exploratory, mostly 3 seeds (one-seed cells marked).

| setting | b = 2 | b = 4 | b = 6 | b = 8 |
|---|---|---|---|---|
| plain, Y labels only, no pretraining (any channel, keys or not; 36/36 same-harness controls) | +0.265 | +0.265 | +0.265 | not run |
| STRUCTURE-KNOWN (decoder reads every agent's key outside b): masked pretraining (+ fixed power), 10000 steps | **+0.016** | **+0.004** | **+0.004** after sign fine-tune | not run |
| BLIND (no key reaches the decoder; keys are a pretraining reconstruction target) | nothing escapes (+0.25) | +0.165 (self-reconstruction + fixed power, fine-tune, 10000 steps) | **+0.051** (self-reconstruction + fixed power, fine-tune, 40000); FSQ queried +0.059 (80000) | **+0.017** (self-reconstruction + FSQ, one-shot, 40000) |
| blind closed form, decoder sees only messages (reference, not a bound) | 0.159 (searched 2-bit, joint-prior decoder) | 0.216 (searched 4-bit, per-agent decoder) | 0.006 at 5.49 bits (hand code) | 0.006 |

**Where it stands.** With the structure known to the decoder, compressed messages learn the aggregation as well as
the uncompressed transformer to within 0.004 nats at b = 4-6 (and 0.016 at b = 2). Blind, the gap is 0.017 at b = 8,
about 0.05 at b = 6, and nothing learns at b = 2, although a blind 5.49-bit hand code reaches 0.006: the information
fits; learning the code is the obstacle.

What worked:
- **Self-supervised pretraining of the code before Y**: masked-agent prediction (structure-known) or self-reconstruction
  of each agent's readings and var ids from all messages (blind). Without it no label-only arm leaves the prior at
  b <= 6; with it every structure-known arm escapes (18/18) and blind arms escape from b = 4.
- **FSQ** (4 levels per dimension, straight-through, usage term) for blind codes: b = 8 0.018 against 0.083 for the
  Gaussian fixed-power channel at the same steps.
- **Fixed power** (unit-power Gaussian messages): removes the vanishing-message failure (encoder gradient 1e-4 -> 0.1),
  helps blind b = 6 with the sign fine-tune, but fails alone in some seeds and breaks the learned-securities arm.
- **The query** (each agent reads the consensus of earlier b-bit messages before writing): helps only where the budget
  is tight, blind b = 6 (-0.034 Gaussian, -0.019 FSQ; the FSQ number is confounded by a usage-term bug); no gain or a
  small cost elsewhere (+0.007 at blind b = 8 with the bug fixed, one seed; +0.05 at structure-known b = 2).
- **Steps up to 4x** narrow the blind gaps; 8x does not help further (b = 8: 0.024 at 8x vs 0.018 at 4x).

What did not work: label-only training without pretraining at any finite b (Gaussian or Gumbel; the encoder sends
nothing and the decoder learns to ignore it); a channel-noise schedule; decoder keys through LayerNorm projections;
a factorised id/value message (no better at b = 6; b = 8 untested at matched steps); DeepSets (stays at the prior);
biased bits or the k curriculum without keys; learned securities with Y labels only; 8x steps.

Open: blind b <= 4 (the information exists, nothing learned reaches it); closing the last 0.017 at b = 8 needs a
better code (the gap does not shrink with steps), e.g. FSQ with more levels per dimension or distillation from the
hand code; queried arms with learned queries (gemini).

## Compression benchmark (added 2026-10-21, `aggbench/compression.py`)

Kelvin, 2026-10-21: "The important part, indeed, is a message smaller than the evidence read by each trader. We
need to know and measure how effective this 'compression' is, specifically for this purpose of the total
aggregation." Also: "We need to specifically use the available evidence *to improve* aggregated estimates, and this
needs to differ from how an agent would summarize information for itself to predict the thing based only on its own
evidence." Also: "it is essential for the consensus to query the agent or vice-versa before the compression
happens." Plan and baselines: `outbox/compressed-aggregation-literature-2026-10-21.md` (mats workspace).

**Budget.** b = bits per agent message. A message is an integer symbol in [0, 2^b), made by an agent-local encoder
from that agent's own `agent_inputs` (plus, in queried and chain protocols, what it received), and it passes a
`Channel(b)` that rejects anything else. A real-valued width without quantization is not a budget (one float can
hold ~1500 tokens). b = 0 is the prior, b = inf the uncompressed numbers. Torch training uses a straight-through
estimator (`ste_quantize` for scalar quantization, `gumbel_code` for a code of 2^b symbols); the eval channel is
hard. Two axes are reported: b per agent, and the total consensus size `consensus_bits` (N b for the append family,
S for a fixed-size state). Which one is primary is open (Kelvin to decide).

**Locality.** `check_message_locality(encode, obs, own, received)`: a message depends only on the sender's own
inputs and what it received (its query, or the chain state); perturbing other agents' observations and own models
must leave it unchanged, and perturbing agent i's received input must leave the others' messages unchanged.

**Metrics.** regret(b) = mean KL to Bayes (nats); `regret_to_floor` = mean log loss minus the floor's
(`report.achievable_floor`; Bayes except E4); rho(b) = (R(0) - R(b)) / (R(0) - R_full) with R_full the floor and,
as a second column in the JSON, the full-context transformer (fresh, unaligned, provenance none); b90 = the
smallest swept b with rho >= 0.9; gain(b) = regret(self-summary, b) - regret(aggregation-aware, b), paired, with a
bootstrap CI.

**Arms.** Literature rank in brackets. "Implemented" = closed form or trained here; "hook" = interface only.

| arm | what each agent sends at b bits | decoder | status |
|---|---|---|---|
| `llr_quantizer` [1] SELF-SUMMARY | its own local log-odds in 2^b cells (companded thresholds on its own-model scale or a pooled scale; family and scale tuned on train). Equals the queried arm with q = 0 (one-shot) | **ORACLE-STRUCTURE decoder**: exact Bayes given the cells and the episode's true structure (read from `batch.oracle`, which a learned decoder never sees): Gaussian box probabilities (Genz, 1024 Sobol points, common to both classes; matches Monte Carlo in the tests) in E2/E3/E4/Z8, enumeration in the tabular zoo. b = inf gives `oracle_clearing` | implemented |
| `queried_llr_qinf` (headline queried), `queried_llr_qb` (minor) | before compressing, agent t reads a query computed from the consensus state (only the earlier b-bit messages; read-only, a pure function of them): unbounded, or q = b bits. Reference rule: running log-odds estimate; the agent centres its quantizer at -kappa x query (kappa tuned, 0 allowed) | same ORACLE-STRUCTURE decoder (cells stay boxes) | reference implemented; learned query and encoder modules, and the parallel (first-round) mode, are hooks |
| `top_w` [3] | its w pieces of largest LLR: which ones (ceil(log2 C(n_i, w)) bits) plus each reading in q bits; w, q tuned on train | ORACLE-STRUCTURE: exact per-piece cell likelihoods over the distinct pieces sent (knows coverage); ignores the selection event | implemented, E2/E3 |
| `k_of_n` [6] | a scheduler that sees the agents' own models (not their readings) lets the most informative agents send all readings at q bits until N b total bits are spent | ORACLE-STRUCTURE: exact over the distinct pieces sent | implemented, E2/E3 |
| `extremized_q`, `trimmed_q` [7] | b-bit quantized local log-odds (cell centroids from train) | extremized mean / 20%-trimmed mean, alpha fit on train; b = inf equals `extremized_log_pool` | implemented |
| `coa_chain_scalar` [4] | Chain-of-Agents reference: one b-bit state that each agent reads and rewrites (running log-odds sum on a uniform grid, weight and range fit on train) | readout of the last state | closed-form reference implemented; learned chain = hook |
| `fixed_state` | fixed-size family at the append total: one S = N b bit state, as above. `run_n_sweep`: regret vs N at fixed S | as above | closed-form reference implemented |
| `dvib` [2] AGGREGATION-AWARE | per-agent MLP over its own token -> b-bit channel (2^b-symbol straight-through Gumbel code, or v2: b-dim Gaussian capacity channel, sign at eval) -> shared set-transformer decoder, end to end (`train_learned_pair`, `train_learned_v2`) | learned | implemented (torch, L4) |
| `self_summary_vib` | the same encoder trained to predict Y from its own message alone (per-agent head), frozen; then a set decoder on its codes | learned | implemented (torch, L4) |
| `raw_code` (tabular zoo) AGGREGATION-AWARE, closed form | its first k raw readings, k = min(p, floor(b / log2 A)); lossless once A^p <= 2^b | ORACLE-STRUCTURE: exact enumeration with the episode's tables (which bit each reading is of, accuracies) | implemented |
| `fair_llr` (E2/E3) FAIR | its own log-odds in 2^b cells, shared thresholds (public) | FAIR: messages + public constants + the structure PRIOR only: P(cells \| Y) = mean over 256 structures sampled from the prior of the Gaussian box probability | implemented |
| `fair_z9` (Z9-p2) FAIR, aggregation-aware | per reading a token: "distractor" or (which parity bit, reading, optional accuracy cell); the unordered pair (lossless for the task at 6 bits, plus 2 x q accuracy bits) or one token, mod 2^b; best code per b chosen on train | FAIR: exact posterior from a per-code table P(message \| w) summed over the per-agent structure prior (exact when the message reveals the vars) | implemented |
| `tree` [5] | b-bit merge net over log2 N levels | learned | hook |

**Environments.** E2, E3-sybil, E3-collude, E4, Z8 (agents read several pieces). The zoo gives each agent one
symbol, so `PiecesPerAgent` groups base readers p at a time: `make_cenv("Z9-p2")` = Z9's 14 readers as 7 agents
with 2 readings each (also Z1-p2, Z5-p2, Z6-p2). Bayes is unchanged; tables are products over symbol tuples, so
decoders stay exact. Z9-p2 is the case where the self-summary is useless (two readings of a 4-bit parity carry no
own signal: the local posterior is the prior) while the evidence matters jointly. E3-biased is excluded.
A native `pieces_per_agent` knob in `zoo.py` would need: the config field on Z1/Z5/Z9 (and Z9's reader-count
check), grouped tables in `TabularEnv.episode`, `slot_layout`/`own_model_from_full` support so that
`agent_inputs(alignment="unaligned")` works (the wrapper supports aligned inputs only; the learned arms build their
own tokens), new rows in `report.py`, and `verify.brute_force_tabular` over tuples.

Commands:

```bash
OMP_NUM_THREADS=2 python3 -m aggbench.compression --envs E2 E3-sybil --bits 1 2 4 8 --out closed.json
python3 -m aggbench.compression --learned --envs E2 E3-sybil Z9-p2 --steps 4000 --out-dir runs/x   # torch, GPU
```

### Results (2026-10-21; exploratory)

Closed forms on CPU (2 threads), 2000 train episodes for tuning, 2000 dev episodes scored, mean KL to Bayes in nats
with 95% bootstrap CIs (KL and the log-loss regret of the main table estimate the same quantity; the main table
shows the latter, e.g. E2 prior 0.433 there vs KL 0.418 here). Harness commits (clean): `765e274` (E2, E3-sybil, the
extra envs and the N sweep), `4f9feeb` (Z9-p2 with `raw_code`). JSON (with harness commits):
`aggbench/compression_results/closed_*.json`. b = inf is the uncompressed message. b90 = smallest swept b with rho >= 0.9 (rho
relative to the floor: 0, except E4 0.047).

What the numbers say:
Every number from `llr_quantizer`, `queried_*`, `top_w`, `k_of_n` and `raw_code` uses an ORACLE-STRUCTURE decoder (the episode's structure out of band). They are references, not targets a learned decoder can reach at the same b;
the FAIR closed forms are in the next subsection. `extremized_q`, `trimmed_q`, `coa_chain_scalar` and `fixed_state` use no episode structure.
- **Self-summary at b bits, oracle-structure decoder** (`llr_quantizer`): E2 0.076 / 0.033 / 0.018 / 0.016
  at b = 1 / 2 / 4 / 8, and 0.016 unquantized (= `oracle_clearing`). b90 = 2 bits per agent (10 bits of consensus).
  From b = 2 it beats unquantized extremizing (0.041). E3-sybil gives the same numbers: the exact decoder knows the
  copies, so the Sybils' bits are only wasted.
- **Extremizing of b-bit log-odds** (`extremized_q`): E2 0.104 / 0.056 / 0.042 / 0.041; E3-sybil 0.158 / 0.099 /
  0.076 / 0.074 (it double-counts the copies). This is the practitioner bar Kelvin wants beaten.
- **Sending pieces** (`top_w`, `k_of_n`) is poor at small b (E2 b = 2: 0.351 and 0.148) and reaches 0.037 / 0.035 at
  b = 8. Only the unbounded message reaches 0.
- **Queries with the reference rule do not help.** The unbounded read-only query (`queried_llr_qinf`) ties one-shot at
  every b: kappa = 0 was the train optimum except at b = 1, where the q = b arm is slightly worse (E2 0.083 vs
  0.076). Running-estimate centring is not a useful query here; learned queries are the open test (gemini).
- **Fixed-size state.** A scalar CoA state of b bits (`coa_chain_scalar`) is far worse than appending b bits per
  agent (E2 b = 2: 0.097 vs 0.033). One state of S = N b bits (`fixed_state`) is at 0.044 from S = 5 and flat at
  0.042: a running sum cannot remove overlap. N sweep at fixed S = 16 (E2, `n_agents` = 2, 4, 8): append-with-exact-
  decoder 0.012 / 0.018 / 0.026, fixed state 0.020 / 0.038 / 0.045. The N = 16 row is unreliable for the exact
  decoder (16 agents on K = 12 pieces make the covariance singular; the QMC estimate degrades: 0.309 with a wide CI).
- **Where the self-summary is useless (Z9-p2).** Two readings of a 4-bit parity carry no own signal, so the
  quantized local posterior is the prior at every b (0.457). Sending raw readings with an oracle-structure decoder
  (`raw_code`) gives 0.360 at b = 1 and 0 from b = 2; a fair decoder needs the structure in the message too
  (a hand code reaches 0.015 at 5.49 bits, below).

Closed-form arms, mean KL to Bayes (nats) [95% CI]:

**E2** (N = 5; prior 0.418, floor 0.000, full-context transformer 0.035). Consensus size = N b = 5, 10, 20, 40 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.418 [0.408, 0.429] | 0.076 [0.070, 0.083] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] | 0.016 [0.014, 0.018] | 2 |
| queried_llr_qinf | 0.418 | 0.076 [0.070, 0.083] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] |  | 2 |
| queried_llr_qb | 0.418 | 0.083 [0.077, 0.089] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] |  | 2 |
| extremized_q | 0.418 [0.408, 0.429] | 0.104 [0.097, 0.111] | 0.056 [0.052, 0.062] | 0.042 [0.038, 0.046] | 0.041 [0.038, 0.045] | 0.041 [0.037, 0.044] | 4 |
| trimmed_q | 0.418 [0.408, 0.429] | 0.116 [0.109, 0.125] | 0.076 [0.070, 0.083] | 0.060 [0.055, 0.065] | 0.059 [0.054, 0.064] | 0.059 [0.054, 0.064] | None |
| top_w | 0.418 [0.408, 0.429] | 0.418 [0.408, 0.429] | 0.351 [0.338, 0.364] | 0.120 [0.112, 0.128] | 0.037 [0.034, 0.040] | -0.000 [-0.000, 0.000] | 8 |
| k_of_n | 0.418 [0.408, 0.429] | 0.228 [0.215, 0.241] | 0.148 [0.137, 0.158] | 0.083 [0.076, 0.090] | 0.035 [0.032, 0.039] | -0.000 [-0.000, 0.000] | 8 |
| coa_chain_scalar | 0.418 [0.408, 0.429] | 0.177 [0.164, 0.192] | 0.097 [0.092, 0.103] | 0.049 [0.045, 0.053] | 0.042 [0.038, 0.045] | 0.042 [0.039, 0.045] | 8 |
| fixed_state | 0.418 [0.408, 0.429] | 0.044 [0.040, 0.048] | 0.042 [0.039, 0.045] | 0.042 [0.039, 0.045] | 0.042 [0.039, 0.045] | 0.042 [0.039, 0.045] | 2 |

**E3-sybil** (N = 8; prior 0.418, floor 0.000, full-context transformer 0.042). Consensus size = N b = 8, 16, 32, 64 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.418 [0.408, 0.429] | 0.076 [0.070, 0.083] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] | 0.016 [0.014, 0.018] | 2 |
| queried_llr_qinf | 0.418 | 0.076 [0.070, 0.083] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] |  | 2 |
| queried_llr_qb | 0.418 | 0.077 [0.071, 0.083] | 0.033 [0.030, 0.037] | 0.018 [0.016, 0.020] | 0.016 [0.014, 0.018] |  | 2 |
| extremized_q | 0.418 [0.408, 0.429] | 0.158 [0.148, 0.168] | 0.099 [0.091, 0.107] | 0.076 [0.070, 0.082] | 0.074 [0.068, 0.081] | 0.075 [0.069, 0.081] | None |
| trimmed_q | 0.418 [0.408, 0.429] | 0.174 [0.163, 0.185] | 0.125 [0.116, 0.133] | 0.106 [0.098, 0.114] | 0.105 [0.097, 0.114] | 0.104 [0.096, 0.112] | None |
| top_w | 0.418 [0.408, 0.429] | 0.418 [0.408, 0.429] | 0.351 [0.338, 0.364] | 0.120 [0.112, 0.128] | 0.037 [0.034, 0.040] | -0.000 [-0.000, 0.000] | 8 |
| k_of_n | 0.418 [0.408, 0.429] | 0.170 [0.159, 0.181] | 0.112 [0.103, 0.121] | 0.068 [0.061, 0.074] | 0.035 [0.031, 0.040] | -0.000 [-0.000, 0.000] | 8 |
| coa_chain_scalar | 0.418 [0.408, 0.429] | 0.167 [0.154, 0.180] | 0.111 [0.103, 0.120] | 0.096 [0.087, 0.105] | 0.081 [0.073, 0.088] | 0.078 [0.073, 0.084] | None |
| fixed_state | 0.418 [0.408, 0.429] | 0.081 [0.073, 0.088] | 0.078 [0.073, 0.084] | 0.078 [0.073, 0.084] | 0.078 [0.073, 0.084] | 0.078 [0.073, 0.084] | None |

**E3-collude** (N = 5; prior 0.418, floor 0.000, full-context transformer 0.024). Consensus size = N b = 5, 10, 20, 40 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.418 [0.408, 0.429] | 0.079 [0.073, 0.085] | 0.032 [0.030, 0.035] | 0.015 [0.014, 0.017] | 0.014 [0.013, 0.016] | 0.014 [0.013, 0.016] | 2 |
| queried_llr_qinf | 0.418 | 0.079 [0.073, 0.085] | 0.032 [0.030, 0.035] | 0.015 [0.014, 0.017] | 0.014 [0.013, 0.016] |  | 2 |
| queried_llr_qb | 0.418 | 0.084 [0.078, 0.090] | 0.032 [0.029, 0.035] | 0.016 [0.014, 0.018] | 0.014 [0.013, 0.016] |  | 2 |
| extremized_q | 0.418 [0.408, 0.429] | 0.096 [0.089, 0.103] | 0.047 [0.043, 0.051] | 0.030 [0.027, 0.033] | 0.029 [0.026, 0.032] | 0.029 [0.026, 0.032] | 4 |
| trimmed_q | 0.418 [0.408, 0.429] | 0.116 [0.108, 0.125] | 0.061 [0.056, 0.067] | 0.042 [0.038, 0.046] | 0.041 [0.037, 0.045] | 0.041 [0.036, 0.045] | 4 |
| top_w | 0.418 [0.408, 0.429] | 0.418 [0.408, 0.429] | 0.388 [0.376, 0.400] | 0.183 [0.174, 0.193] | 0.076 [0.071, 0.082] | -0.000 [-0.000, 0.000] | None |
| k_of_n | 0.418 [0.408, 0.429] | 0.264 [0.250, 0.277] | 0.142 [0.132, 0.151] | 0.079 [0.072, 0.086] | 0.034 [0.030, 0.038] | -0.000 [-0.000, 0.000] | 8 |
| coa_chain_scalar | 0.418 [0.408, 0.429] | 0.135 [0.125, 0.145] | 0.084 [0.080, 0.089] | 0.038 [0.035, 0.042] | 0.029 [0.026, 0.032] | 0.029 [0.026, 0.032] | 4 |
| fixed_state | 0.418 [0.408, 0.429] | 0.033 [0.030, 0.036] | 0.029 [0.026, 0.032] | 0.029 [0.026, 0.032] | 0.029 [0.026, 0.032] | 0.029 [0.026, 0.032] | 1 |

**Z8** (N = 5; prior 0.450, floor 0.000, full-context transformer 0.036). Consensus size = N b = 5, 10, 20, 40 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.450 [0.439, 0.460] | 0.075 [0.069, 0.081] | 0.034 [0.031, 0.037] | 0.018 [0.017, 0.020] | 0.017 [0.015, 0.019] | 0.017 [0.015, 0.019] | 2 |
| queried_llr_qinf | 0.450 | 0.075 [0.069, 0.081] | 0.034 [0.031, 0.037] | 0.018 [0.017, 0.020] | 0.017 [0.015, 0.019] |  | 2 |
| queried_llr_qb | 0.450 | 0.079 [0.073, 0.085] | 0.034 [0.031, 0.037] | 0.019 [0.017, 0.021] | 0.017 [0.015, 0.019] |  | 2 |
| extremized_q | 0.450 [0.439, 0.460] | 0.094 [0.087, 0.102] | 0.050 [0.046, 0.055] | 0.032 [0.029, 0.035] | 0.030 [0.028, 0.033] | 0.030 [0.028, 0.033] | 4 |
| trimmed_q | 0.450 [0.439, 0.460] | 0.106 [0.098, 0.114] | 0.063 [0.058, 0.068] | 0.049 [0.044, 0.053] | 0.047 [0.043, 0.051] | 0.047 [0.043, 0.051] | None |
| coa_chain_scalar | 0.450 [0.439, 0.460] | 0.189 [0.174, 0.204] | 0.099 [0.094, 0.105] | 0.037 [0.034, 0.040] | 0.030 [0.027, 0.033] | 0.030 [0.027, 0.033] | 4 |
| fixed_state | 0.450 [0.439, 0.460] | 0.034 [0.031, 0.037] | 0.030 [0.027, 0.033] | 0.030 [0.027, 0.033] | 0.030 [0.027, 0.033] | 0.030 [0.027, 0.033] | 1 |

**E4** (N = 6; prior 0.353, floor 0.047, full-context transformer 0.079). Consensus size = N b = 6, 12, 24, 48 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.353 [0.343, 0.363] | 0.129 [0.120, 0.138] | 0.094 [0.087, 0.101] | 0.082 [0.076, 0.089] | 0.080 [0.074, 0.087] | 0.080 [0.074, 0.086] | None |
| queried_llr_qinf | 0.353 | 0.129 [0.120, 0.138] | 0.094 [0.087, 0.101] | 0.082 [0.076, 0.089] | 0.080 [0.074, 0.087] |  | None |
| queried_llr_qb | 0.353 | 0.129 [0.120, 0.138] | 0.094 [0.087, 0.101] | 0.081 [0.075, 0.088] | 0.080 [0.074, 0.087] |  | None |
| extremized_q | 0.353 [0.343, 0.363] | 0.147 [0.139, 0.156] | 0.112 [0.105, 0.120] | 0.102 [0.095, 0.109] | 0.101 [0.094, 0.109] | 0.101 [0.094, 0.108] | None |
| trimmed_q | 0.353 [0.343, 0.363] | 0.156 [0.147, 0.165] | 0.125 [0.117, 0.133] | 0.112 [0.104, 0.120] | 0.111 [0.103, 0.118] | 0.111 [0.103, 0.118] | None |
| coa_chain_scalar | 0.353 [0.343, 0.363] | 0.245 [0.229, 0.263] | 0.151 [0.142, 0.161] | 0.118 [0.112, 0.125] | 0.103 [0.097, 0.110] | 0.103 [0.096, 0.110] | None |
| fixed_state | 0.353 [0.343, 0.363] | 0.105 [0.098, 0.111] | 0.103 [0.096, 0.110] | 0.103 [0.096, 0.110] | 0.103 [0.096, 0.110] | 0.103 [0.096, 0.110] | None |

| N (E2, S = 16 bits total) | b = S/N | llr_quantizer | queried_llr_qinf | extremized_q | fixed_state (S bits) | prior |
|---|---|---|---|---|---|---|
| 2 | 8 | 0.012 [0.010, 0.013] | 0.011 [0.010, 0.013] | 0.020 [0.018, 0.022] | 0.020 [0.019, 0.022] | 0.311 |
| 4 | 4 | 0.018 [0.016, 0.019] | 0.018 [0.016, 0.019] | 0.039 [0.036, 0.042] | 0.038 [0.035, 0.041] | 0.402 |
| 8 | 2 | 0.026 [0.023, 0.028] | 0.026 [0.023, 0.028] | 0.057 [0.052, 0.062] | 0.045 [0.041, 0.050] | 0.450 |
| 16 | 1 | 0.309 [0.166, 0.470] | 0.110 [0.087, 0.144] | 0.060 [0.055, 0.065] | 0.028 [0.025, 0.031] | 0.460 |

**Z9-p2** (N = 7; prior 0.457, floor 0.000, full-context transformer n/a). Consensus size = N b = 7, 14, 28, 56 bits for b = 1, 2, 4, 8.

| arm | b=0 | b=1 | b=2 | b=4 | b=8 | b=inf | b90 |
|---|---|---|---|---|---|---|---|
| llr_quantizer | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | None |
| raw_code | 0.457 [0.448, 0.465] | 0.360 [0.345, 0.373] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2 |

#### Fair closed forms (2026-10-22; the decoder sees no episode structure)

Gemini found, and the coordinator confirmed, that the closed-form decoders above read each episode's structure from `batch.oracle` (`_tabular_tables`, `gaussian_report_model`), while learned decoders see only messages. The fair closed forms decode from the messages, the public constants and the environment's generative prior over structures (sampled from reserved fresh seeds, never from the scored episodes); an agent that wants the decoder to know its structure spends bits on it. Tests run them on `blind(batch)`, whose oracle raises on any access, and check that they equal the oracle version once the message carries the structure (Z9 with fixed accuracy at b >= 6; E2 with the prior set to the true structure). Harness commit `ad8f7ab`; 2000 train / 2000 dev episodes; JSON `aggbench/compression_results/fair.json`.

| env | b = 1 | b = 2 | b = 4 | b = 8 |
|---|---|---|---|---|
| E2, `fair_llr` | 0.104 [0.097, 0.111] | 0.057 [0.052, 0.062] | 0.040 [0.036, 0.044] | 0.038 [0.035, 0.042] |
| E3-sybil, `fair_llr` | 0.128 [0.119, 0.137] | 0.068 [0.062, 0.075] | 0.074 [0.063, 0.089] | 0.077 [0.068, 0.087] |

| Z9-p2, `fair_z9`, b | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best code (chosen on train) | one, q=0 | one, q=0 | one, q=0 | one, q=1 | pair, q=0 | pair, q=0 | pair, q=0 | pair, q=1 | pair, q=1 | pair, q=2 | pair, q=2 | pair, q=2 |
| regret | 0.425 | 0.383 | 0.310 | 0.306 | 0.132 | 0.015 | 0.015 | 0.004 | 0.004 | 0.001 | 0.001 | 0.001 |

- **Z9-p2:** a hand code reaches 0.015 at 5.49 bits (6-bit fixed length): the unordered pair of (which of the 4 parity bits or "distractor", reading) tokens has 45 symbols; the minimum number of bits is unknown (5 of the 45 pairs carry no per-agent information and could share a symbol). The residual is the unknown accuracy; 2 accuracy bits per reading bring it to 0.001 at b = 10. Below 6 bits these hand codes with a PER-AGENT decoder (it ignores the coupling between agents) are poor: 0.425 / 0.383 / 0.306 at b = 1 / 2 / 4; with the joint-prior decoder (learnability section) the best blind codes found are 0.350 / 0.159 / <= 0.130.
- **E2 / E3-sybil:** the fair self-summary is 0.104 / 0.057 / 0.040 / 0.038 on E2 (oracle-structure: 0.076 / 0.033 / 0.018 / 0.016) and 0.128 / 0.068 / 0.074 / 0.077 on E3-sybil: not knowing the structure costs the decoder the overlap correction, and the copies now count. The E3-sybil numbers are NOT a bound: they rise from b = 2 to b = 8, so the 256-structure Monte Carlo decoder is not Bayes under its own prior at fine cells.

**Learned arms** (spot L4 runs `aggcomp-learned-20261021` at `f442270` and `aggcomp-learned-ws-20261021` at
`9643d1b`, $0.83 + $0.51; JSON: `aggbench/compression_results/learned_{run1,ws}.json`). 100k train episodes, 4000 AdamW steps of
512 per stage, 2000 dev episodes; the decoder sees only the b-bit codes (provenance none, no structure), so these
are not comparable with the structure-aware exact decoder above. Self-summary = encoder trained on own-evidence Y
prediction, frozen, then a set decoder. Aware from scratch = end to end. Aware warm start = the self-summary system
fine-tuned end to end. Gain = regret(self-summary) - regret(aware), paired 95% CI; positive means the aware code is
better. b = inf here is an unquantized 16-dim float message (diagnostic, not a budget).

| env | b | self-summary (learned) | aware, from scratch | aware, warm start | gain (scratch) | gain (warm start) | untrained control |
|---|---|---|---|---|---|---|---|
| E2 | 1 | 0.102 [0.096, 0.109] | 0.214 [0.203, 0.226] | 0.114 [0.106, 0.122] | -0.112 [-0.126, -0.099] | -0.011 [-0.019, -0.004] | 0.424 |
| E2 | 2 | 0.062 [0.057, 0.067] | 0.059 [0.054, 0.064] | 0.057 [0.052, 0.061] | 0.003 [-0.001, 0.008] | 0.005 [0.002, 0.009] | 0.449 |
| E2 | 4 | 0.046 [0.042, 0.049] | 0.052 [0.048, 0.057] | 0.051 [0.047, 0.055] | -0.007 [-0.010, -0.004] | -0.005 [-0.008, -0.003] | 0.414 |
| E2 | 8 | 0.045 [0.042, 0.050] | 0.044 [0.041, 0.048] | 0.047 [0.043, 0.051] | 0.001 [-0.002, 0.004] | -0.002 [-0.004, 0.001] | 0.446 |
| E2 | inf | 0.037 [0.034, 0.041] | 0.035 [0.032, 0.038] | n/a | 0.002 [-0.000, 0.005] | n/a | 0.415 |
| E3-sybil | 1 | 0.125 [0.117, 0.133] | 0.301 [0.289, 0.312] | 0.328 [0.317, 0.339] | -0.175 [-0.189, -0.161] | -0.203 [-0.217, -0.189] | 0.421 |
| E3-sybil | 2 | 0.066 [0.061, 0.071] | 0.130 [0.122, 0.139] | 0.093 [0.086, 0.100] | -0.064 [-0.074, -0.055] | -0.027 [-0.034, -0.020] | 0.453 |
| E3-sybil | 4 | 0.048 [0.044, 0.052] | 0.111 [0.103, 0.120] | 0.050 [0.046, 0.054] | -0.064 [-0.072, -0.056] | -0.002 [-0.005, 0.001] | 0.410 |
| E3-sybil | 8 | 0.053 [0.048, 0.058] | 0.050 [0.046, 0.054] | 0.056 [0.052, 0.061] | 0.003 [-0.001, 0.007] | -0.004 [-0.008, 0.000] | 0.447 |
| E3-sybil | inf | 0.038 [0.035, 0.041] | 0.040 [0.037, 0.044] | n/a | -0.003 [-0.006, 0.000] | n/a | 0.417 |
| E3-collude | 1 | 0.091 [0.084, 0.097] | 0.114 [0.106, 0.122] | n/a | -0.024 [-0.033, -0.015] | n/a | 0.425 |
| E3-collude | 2 | 0.047 [0.043, 0.051] | 0.047 [0.044, 0.051] | n/a | -0.000 [-0.003, 0.002] | n/a | 0.455 |
| E3-collude | 4 | 0.031 [0.028, 0.034] | 0.040 [0.037, 0.044] | n/a | -0.009 [-0.011, -0.007] | n/a | 0.402 |
| E3-collude | 8 | 0.032 [0.029, 0.036] | 0.033 [0.030, 0.036] | n/a | -0.001 [-0.003, 0.001] | n/a | 0.447 |
| E3-collude | inf | 0.025 [0.023, 0.028] | 0.027 [0.024, 0.031] | n/a | -0.002 [-0.005, 0.000] | n/a | 0.414 |
| Z8 | 1 | 0.095 [0.087, 0.103] | 0.172 [0.160, 0.183] | n/a | -0.077 [-0.091, -0.063] | n/a | 0.454 |
| Z8 | 2 | 0.048 [0.044, 0.053] | 0.051 [0.047, 0.056] | n/a | -0.003 [-0.006, 0.000] | n/a | 0.486 |
| Z8 | 4 | 0.037 [0.033, 0.040] | 0.047 [0.043, 0.052] | n/a | -0.011 [-0.014, -0.008] | n/a | 0.444 |
| Z8 | 8 | 0.036 [0.033, 0.040] | 0.041 [0.037, 0.045] | n/a | -0.005 [-0.007, -0.003] | n/a | 0.485 |
| Z8 | inf | 0.031 [0.028, 0.034] | 0.035 [0.031, 0.038] | n/a | -0.004 [-0.005, -0.001] | n/a | 0.446 |
| E4 | 1 | 0.135 [0.126, 0.143] | 0.288 [0.277, 0.298] | n/a | -0.153 [-0.166, -0.140] | n/a | 0.374 |
| E4 | 2 | 0.095 [0.088, 0.102] | 0.110 [0.103, 0.118] | n/a | -0.016 [-0.021, -0.010] | n/a | 0.351 |
| E4 | 4 | 0.089 [0.082, 0.096] | 0.093 [0.086, 0.100] | n/a | -0.004 [-0.007, -0.001] | n/a | 0.415 |
| E4 | 8 | 0.090 [0.083, 0.097] | 0.089 [0.082, 0.096] | n/a | 0.001 [-0.003, 0.005] | n/a | 0.400 |
| E4 | inf | 0.079 [0.073, 0.085] | 0.081 [0.075, 0.088] | n/a | -0.002 [-0.005, 0.000] | n/a | 0.427 |
| Z9-p2 | 1 | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.456 [0.448, 0.465] | 0.000 [-0.000, 0.000] | 0.000 [-0.000, 0.001] | 0.461 |
| Z9-p2 | 2 | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.000 [-0.000, 0.001] | -0.000 [-0.001, 0.000] | 0.463 |
| Z9-p2 | 4 | 0.458 [0.450, 0.466] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.001 [-0.000, 0.003] | 0.001 [-0.000, 0.003] | 0.465 |
| Z9-p2 | 8 | 0.459 [0.450, 0.468] | 0.457 [0.448, 0.465] | 0.457 [0.448, 0.465] | 0.002 [0.000, 0.004] | 0.002 [0.000, 0.004] | 0.488 |
| Z9-p2 | inf | 0.243 [0.230, 0.258] | 0.082 [0.073, 0.091] | n/a | 0.162 [0.148, 0.175] | n/a | 0.467 |

What the learned arms say: the straight-through Gumbel aggregation-aware arm does not beat the learned
self-summary anywhere with a real budget (the only positive CI is E2 b = 2 warm start, +0.005). It fails badly at
b <= 2 on E3-sybil (0.30 at b = 1). It learns nothing on Z9-p2 at any finite b, although a closed-form 2-bit code
reaches 0 there with an oracle-structure decoder (a blind hand code reaches 0.015 at 5.49 bits); with an unquantized float message it reaches 0.082 vs the self-summary's 0.243. So the learned
discrete-code training is the bottleneck, not the information. Next: train the codes better (deterministic STE on
argmax, temperature schedule, VQ, or distillation from `raw_code`), then rerun the gain. The learned decoders also
stay above the oracle-structure decoder at every b (E2 b = 2: 0.057 vs 0.033); the fair closed form at b = 2 is 0.057.

### Trainable learned arms (v2, 2026-10-21; revised after the gate `outbox/gate-e1gls-compv2-2026-10-21.md`)

**Headline (Y labels only).** End to end (aggregation-aware) vs the learned self-summary, Gaussian channel, mean of 3
seeds: Z9-p2 +0.065 / +0.108 / +0.228 nats at b = 1 / 2 / 4 (aware regret 0.392 / 0.349 / 0.229). Blind references at the same b
(no episode structure; review 2026-10-22): 0.350 / 0.159 / <= 0.130 (searched 1-, 2-, 3-bit codes with a
joint-prior decoder); a blind code needs 5.49 bits for 0.015. The oracle-structure raw code [provenance full]
reaches 0 at 2 bits but is not a target for a structure-blind decoder; E2 and E3-sybil: no gain (|gain| <= 0.007). On Z9 the
self-summary is null BY CONSTRUCTION: with k = 4 and two readings per agent, an agent's own evidence has zero mutual
information with Y, so the own-prediction target is independent of the input, the encoder collapses (1-2 distinct
symbols) and the self arm sits at the prior (0.457) in every cell. The Z9 "gain" is therefore aware minus prior; the
fair non-aware comparison is the random-frozen-encoder control below.

Changes (`train_learned_v2`, `run_learned_v2`, `run_z9_controls`; `python3 -m aggbench.compression --learned-v2`,
`--z9-controls`):
- **Gaussian capacity channel** (DIAL-style): the encoder emits b dimensions, tanh (|s| <= 1), plus N(0, 1/3) noise in
  training: 1/2 log2(1 + P / sigma^2) = 1 bit per dimension is an upper bound on capacity (with an amplitude limit it
  is lower). Train noisy, test quantised: the reported regret sends sign(.) through `Channel(b)` (exactly b bits);
  the noisy-channel regret is a column. `gumbel` = the earlier 2^b-symbol straight-through code.
- **Parity aids, aware arms only (asymmetry, stated):** an intermediate-parity auxiliary loss, decoder heads
  predicting the prefix parities b1, b1^b2, b1^b2^b3 of the simulator's LATENT bits (`oracle.w`), weight 1 annealed
  to 0 by half the steps. This is supervision beyond Y, so the arms that use it are tagged **[LATENT-LABEL
  SUPERVISION]** and are not headline results. And a k curriculum (Z9 with k = 2, then 3, then 4 parity bits; Y
  labels only, but it encodes the designer's knowledge that the task is a parity; 2 x 30k extra episodes). The
  self-summary got neither: its encoder is trained on own-evidence prediction, which carries no signal on Z9 at any
  k <= 4 with two readings, so the aids could not change it; the decoder stage was not given them either. Bit-bias
  annealing is not implemented (it needs a biased-bit option in `zoo.KParityEnv`).
- Decoders are learned and structure-blind (provenance none). Self-summary: encoder trained on own-evidence Y
  prediction (S steps), frozen, then the decoder (S steps). Aware: end to end (S steps). 3 seeds, b in {1, 2, 4};
  S = 4000 (E2/E3-sybil) or 8000 (Z9-p2); 100k train episodes; 2000 dev episodes; gains paired per seed.
- **Steps to escape** = the first evaluation (every 200 steps) at which validation KL on the k = 4 task falls below
  0.9 x the prior's (`ESCAPE_FRAC`; the runs stored a 0.5 threshold, now kept as `steps_to_escape_0p5_as_run`, and
  `steps_to_escape` is recomputed at 0.9 from the stored curves). Curriculum steps count from the start of k = 2
  (k = 4 starts at step 4000).
- Runs: spot L4 `aggcomp-v2-{e2e3,z9,z9b2,z9b4}-20261021` at harness commit `2b62d80` (tag
  `aggcomp-v2-runs-20261021`), $2.75; the Z9 VM was stopped after b = 1 and b = 2, 4 ran on their own VMs. Result
  JSON with the harness commit: `aggbench/compression_results/` (closed forms and learned v1 too).

E2 and E3-sybil (regret in nats, mean of 3 seeds; gain = self - aware, positive = aware better):

| env | b | channel | aware arm | self-summary | aware | gain, mean of 3 seeds | per-seed gain [95% CI] | aware noisy-channel regret | steps to escape: val KL < 0.9 prior (self / aware, per seed) |
|---|---|---|---|---|---|---|---|---|---|
| E2 | 1 | gauss | aware (Y labels only) | 0.104 | 0.107 | -0.003 | -0.004 [-0.009, -0.000]; -0.002 [-0.007, +0.002]; -0.002 [-0.006, +0.001] | 0.116 | 200/200, 200/200, 200/200 |
| E2 | 1 | gumbel | aware (Y labels only) | 0.105 | 0.227 | -0.122 | -0.121 [-0.136, -0.107]; -0.102 [-0.115, -0.088]; -0.144 [-0.157, -0.131] | n/a | 200/200, 200/200, 200/200 |
| E2 | 2 | gauss | aware (Y labels only) | 0.071 | 0.070 | +0.001 | -0.001 [-0.005, +0.003]; +0.003 [-0.001, +0.007]; +0.002 [-0.002, +0.006] | 0.080 | 200/200, 200/200, 200/200 |
| E2 | 2 | gumbel | aware (Y labels only) | 0.061 | 0.120 | -0.059 | -0.105 [-0.116, -0.094]; -0.016 [-0.021, -0.010]; -0.058 [-0.066, -0.049] | n/a | 0/200, 200/200, 0/200 |
| E2 | 4 | gauss | aware (Y labels only) | 0.052 | 0.051 | +0.001 | -0.001 [-0.004, +0.003]; +0.001 [-0.002, +0.004]; +0.001 [-0.002, +0.005] | 0.057 | 200/200, 0/200, 200/200 |
| E2 | 4 | gumbel | aware (Y labels only) | 0.046 | 0.054 | -0.008 | -0.003 [-0.006, -0.000]; -0.015 [-0.019, -0.012]; -0.007 [-0.010, -0.004] | n/a | 0/200, 200/200, 200/200 |
| E3-sybil | 1 | gauss | aware (Y labels only) | 0.129 | 0.135 | -0.006 | -0.006 [-0.011, -0.001]; -0.005 [-0.010, -0.001]; -0.007 [-0.012, -0.002] | 0.127 | 200/200, 200/200, 200/200 |
| E3-sybil | 1 | gumbel | aware (Y labels only) | 0.126 | 0.339 | -0.213 | -0.209 [-0.223, -0.195]; -0.200 [-0.214, -0.186]; -0.232 [-0.245, -0.219] | n/a | 200/200, 200/200, 200/200 |
| E3-sybil | 2 | gauss | aware (Y labels only) | 0.081 | 0.079 | +0.002 | -0.001 [-0.006, +0.005]; +0.006 [+0.001, +0.011]; +0.001 [-0.005, +0.006] | 0.087 | 200/200, 200/200, 200/200 |
| E3-sybil | 2 | gumbel | aware (Y labels only) | 0.066 | 0.237 | -0.170 | -0.047 [-0.055, -0.039]; -0.237 [-0.249, -0.224]; -0.227 [-0.240, -0.214] | n/a | 200/200, 200/200, 0/200 |
| E3-sybil | 4 | gauss | aware (Y labels only) | 0.058 | 0.058 | +0.000 | +0.002 [-0.002, +0.006]; -0.001 [-0.005, +0.003]; -0.001 [-0.006, +0.003] | 0.063 | 200/200, 0/200, 200/200 |
| E3-sybil | 4 | gumbel | aware (Y labels only) | 0.049 | 0.086 | -0.037 | -0.001 [-0.005, +0.002]; -0.106 [-0.117, -0.095]; -0.004 [-0.007, -0.000] | n/a | 0/200, 200/200, 200/200 |

Z9-p2 (two readings of a 4-bit parity per agent; blind references 0.350 / 0.159 / <= 0.130 at b = 1 / 2 / 4; the self-summary is at the prior by construction):

| env | b | channel | aware arm | self-summary | aware | gain, mean of 3 seeds | per-seed gain [95% CI] | aware noisy-channel regret | steps to escape: val KL < 0.9 prior (self / aware, per seed) |
|---|---|---|---|---|---|---|---|---|---|
| Z9-p2 | 1 | gauss | aware (Y labels only) | 0.457 | 0.392 | +0.065 | +0.098 [+0.085, +0.111]; +0.098 [+0.086, +0.111]; +0.000 [-0.001, +0.001] | 0.410 | None/4800, None/3800, None/None |
| Z9-p2 | 1 | gauss | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.457 | +0.000 | -0.000 [-0.000, +0.000]; +0.000 [-0.000, +0.001]; -0.000 [-0.000, +0.000] | 0.457 | None/None, None/None, None/None |
| Z9-p2 | 1 | gauss | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.388 | +0.068 | +0.100 [+0.087, +0.114]; +0.000 [-0.000, +0.000]; +0.105 [+0.091, +0.118] | 0.416 | None/4200, None/None, None/4200 |
| Z9-p2 | 1 | gumbel | aware (Y labels only) | 0.457 | 0.431 | +0.025 | +0.077 [+0.065, +0.090]; -0.000 [-0.001, +0.001]; -0.001 [-0.002, +0.000] | n/a | None/4800, None/None, None/None |
| Z9-p2 | 1 | gumbel | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.457 | -0.000 | +0.000 [-0.001, +0.001]; -0.000 [-0.001, +0.000]; -0.000 [-0.000, +0.000] | n/a | None/None, None/None, None/None |
| Z9-p2 | 1 | gumbel | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.439 | +0.018 | +0.053 [+0.043, +0.062]; +0.002 [-0.001, +0.005]; -0.000 [-0.000, +0.000] | n/a | None/6600, None/None, None/None |
| Z9-p2 | 2 | gauss | aware (Y labels only) | 0.457 | 0.349 | +0.108 | +0.099 [+0.087, +0.113]; +0.098 [+0.085, +0.112]; +0.127 [+0.111, +0.142] | 0.361 | None/3200, None/4000, None/4000 |
| Z9-p2 | 2 | gauss | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.457 | -0.000 | +0.000 [-0.000, +0.000]; -0.000 [-0.001, +0.001]; -0.000 [-0.000, +0.000] | 0.457 | None/None, None/None, None/None |
| Z9-p2 | 2 | gauss | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.215 | +0.242 | +0.193 [+0.177, +0.209]; +0.236 [+0.220, +0.252]; +0.297 [+0.282, +0.313] | 0.298 | None/4200, None/6800, None/4200 |
| Z9-p2 | 2 | gumbel | aware (Y labels only) | 0.457 | 0.390 | +0.067 | +0.106 [+0.092, +0.119]; -0.001 [-0.002, +0.000]; +0.095 [+0.082, +0.108] | n/a | None/5800, None/None, None/4400 |
| Z9-p2 | 2 | gumbel | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.441 | +0.015 | +0.000 [-0.001, +0.001]; +0.046 [+0.036, +0.056]; -0.000 [-0.001, +0.001] | n/a | None/None, None/6600, None/None |
| Z9-p2 | 2 | gumbel | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.324 | +0.133 | +0.299 [+0.282, +0.315]; +0.101 [+0.088, +0.115]; -0.000 [-0.001, +0.001] | n/a | None/4200, None/4200, None/None |
| Z9-p2 | 4 | gauss | aware (Y labels only) | 0.457 | 0.229 | +0.228 | +0.293 [+0.276, +0.311]; +0.098 [+0.085, +0.112]; +0.293 [+0.277, +0.309] | 0.257 | None/3800, None/3600, None/2800 |
| Z9-p2 | 4 | gauss | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.143 | +0.314 | +0.363 [+0.350, +0.377]; +0.293 [+0.278, +0.309]; +0.286 [+0.270, +0.301] | 0.226 | None/800, None/4000, None/1000 |
| Z9-p2 | 4 | gauss | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.119 | +0.338 | +0.372 [+0.359, +0.384]; +0.332 [+0.318, +0.345]; +0.311 [+0.295, +0.327] | 0.199 | None/4200, None/4800, None/4400 |
| Z9-p2 | 4 | gumbel | aware (Y labels only) | 0.457 | 0.424 | +0.033 | +0.002 [-0.001, +0.005]; +0.099 [+0.086, +0.113]; -0.001 [-0.002, +0.001] | n/a | None/None, None/4000, None/None |
| Z9-p2 | 4 | gumbel | aware_aux [LATENT-LABEL SUPERVISION] | 0.457 | 0.135 | +0.322 | +0.351 [+0.336, +0.366]; +0.260 [+0.243, +0.277]; +0.355 [+0.341, +0.370] | n/a | None/1000, None/1200, None/2600 |
| Z9-p2 | 4 | gumbel | aware_curr_aux [LATENT-LABEL SUPERVISION + k curriculum] | 0.457 | 0.216 | +0.241 | +0.340 [+0.325, +0.355]; +0.113 [+0.099, +0.128]; +0.271 [+0.254, +0.286] | n/a | None/4400, None/4200, None/4200 |

Z9-p2 controls (Y labels only; `run_z9_controls`, 8000 steps, same data and seeds; aware rerun alongside for paired gains):

Uncompressed full-context anchor (the same set decoder reading every agent's raw token, no channel): regret 0.068, 0.067, 0.056 (seeds 0, 1, 2; repeated on each VM, identical).

| b | channel | random frozen encoder | aware (rerun) | aware - random, mean of 3 seeds | per-seed gain [95% CI] |
|---|---|---|---|---|---|
| 1 | gauss | 0.457 | 0.392 | +0.065 | +0.098 [+0.086, +0.111]; +0.098 [+0.086, +0.111]; +0.000 [-0.001, +0.001] |
| 1 | gumbel | 0.457 | 0.431 | +0.025 | +0.077 [+0.065, +0.089]; +0.000 [-0.001, +0.001]; -0.001 [-0.002, +0.000] |
| 2 | gauss | 0.457 | 0.349 | +0.108 | +0.099 [+0.086, +0.113]; +0.098 [+0.085, +0.112]; +0.127 [+0.111, +0.142] |
| 2 | gumbel | 0.457 | 0.390 | +0.066 | +0.105 [+0.092, +0.118]; -0.000 [-0.001, +0.000]; +0.094 [+0.082, +0.108] |
| 4 | gauss | 0.457 | 0.229 | +0.227 | +0.292 [+0.275, +0.309]; +0.098 [+0.085, +0.112]; +0.292 [+0.277, +0.309] |
| 4 | gumbel | 0.456 | 0.424 | +0.032 | -0.000 [-0.003, +0.003]; +0.100 [+0.086, +0.114]; -0.002 [-0.005, +0.000] |

Controls read: a random frozen encoder (a non-collapsed, non-aware local code) is also at the prior, 0.457, at every b
and seed, so on Z9 the label-only gains above are gains of learned aggregation-aware coding over a fixed local code
too, not only over the collapsed self-summary. The aware reruns reproduce the v2 numbers seed for seed. The
uncompressed anchor (same decoder, data and steps, every raw token) reaches 0.056-0.068: the best label-only
compressed arm (b = 4, 0.229) is about 0.17 nats behind it. That anchor reuses a 100k pool for 41 epochs and
reports the final model (it memorizes, review B4); the fresh-episode anchor is in the learnability section. Recomputed
against BLIND references (the earlier 'the closed-form 2-bit code reaches 0, so the gap is learning, not bits' used
an oracle-structure decoder and is withdrawn): at b = 2 the best blind code found reaches 0.159 vs learned 0.349
(a learning gap of about 0.19); at b = 4 learned 0.229 vs a blind 3-bit code at 0.130; a blind code needs about
5.5 bits per agent to get near 0, so below that part of the gap to 0 is bits. (The main table's Z9 full-context transformer,
0.002, is a 3-layer model trained on fresh episodes; this anchor is the 2-layer compression decoder on the 100k pool.)
Runs: spot L4 `aggcomp-z9ctl-b{1r,2,4}-20261021` at `f98df6a` (tag `aggcomp-z9ctl-runs-20261021`; two earlier
attempts were preempted), about $1.0; JSON `aggbench/compression_results/z9_controls_b*.json`.

What it says:
- **E2 and E3-sybil: end to end does not beat the self-summary.** With the Gaussian channel the two tie within
  +-0.007 at every b. The only positive per-seed CI is +0.006 [0.001, 0.011] (E3-sybil b = 2, one seed of three).
  At E3-sybil b = 1 the self-summary is slightly AHEAD on all 3 seeds (gain -0.006, every per-seed CI below 0). The
  Gaussian channel fixes the Gumbel arm's collapse (E3-sybil b = 1: 0.135 vs 0.339). With a learned,
  structure-blind decoder neither comes near the oracle-structure decoder (E2 b = 2: 0.070 vs 0.033); the fair (blind) closed form is 0.057
  at b = 2, so the learning gap there is about 0.013.
  Duplicates cost the learned self-summary 0.006-0.025 nats over E2 (Gaussian channel).
- **Z9-p2, Y labels only:** plain end to end leaves the prior in 2 of 3 seeds at b = 1, in 3 of 3 at b = 2 and 4:
  +0.065 / +0.108 / +0.228 (Gaussian), +0.025 / +0.067 / +0.033 (Gumbel). Blind references at the same b: 0.350 / 0.159 / <= 0.130.
- **Z9-p2 with latent-label supervision** (not headline): curriculum + aux gives +0.068 / +0.242 / +0.338 (regret
  0.388 / 0.215 / 0.119); the aux loss alone helps only at b = 4 (escape at 800-1000 steps vs 2800-3800 without);
  the curriculum is what moves b = 2.

## Learnability review follow-up (2026-10-22, `aggbench/parity_learn.py`)

Review: `outbox/learnability-review-2026-10-22.md` (mats workspace; Fable-adjudicated). This section covers its step 0
(the reframed table), step 1, step 3, step 4, step 5 and bugs B1, B4, B6, B7, B8, B9, B10, B12; gemini owns B5, B11,
B13 and steps 2 and 6. Kelvin's target: the compressed learned arms should match what an uncompressed transformer
reading everything learns on the same data, steps and labels, so **gap(b) = regret(learned arm, b) - regret(anchor)
is the primary column**. Blind (provenance-paid) closed forms bound what b bits can carry; oracle-structure closed
forms [provenance full] are references only.

### Step 0: the reframed table

Anchors (the uncompressed full-context transformer, fresh episodes, best-val checkpoint): **Z9-p2 0.0036** (0.0037,
0.0033, 0.0038; `parity_learn` `anchor_fresh`, 8000 steps of 512 fresh episodes, the same decoder as the arms).
E2 0.035 and E3-sybil 0.042 are the main table's fresh-regime transformer (3-layer, 3000 x 512 fresh episodes). The
pool anchors are not used: they memorize their 100k pool and report the final model (review B4): Z9-p2 0.056-0.068,
E2 0.138-0.147, E3-sybil 0.122-0.156 (`compression_results/anchor.json`, `z9_controls_b*.json`).
**One Z9-p2 anchor: 0.0036.** Gemini's 0.2583 "FCT" column is `dvib_oneshot_q0` at b = inf (a per-agent encoder with
an unbounded float message, 4500 steps), substituted for the missing anchor (review B5; gemini relabels it); the
0.056-0.068 one is the pool anchor above. Steps, data and the decoder explain the three numbers.

| env | b | learned self-summary (v2, Gaussian) | learned aware, Y labels only | gap to anchor (aware) | symbols used per agent (of 2^b) | blind closed form (structure paid in bits) | oracle-structure closed form [provenance full] |
|---|---|---|---|---|---|---|---|
| E2 | 1 | 0.104 | 0.107 | +0.072 | 2 | 0.104 | 0.076 |
| E2 | 2 | 0.071 | 0.070 | +0.035 | 3 | 0.057 | 0.033 |
| E2 | 4 | 0.052 | 0.051 | +0.016 | 5 | 0.040 | 0.018 |
| E3-sybil | 1 | 0.129 | 0.135 | +0.093 | 2 | 0.128 | 0.076 |
| E3-sybil | 2 | 0.081 | 0.079 | +0.037 | 3 | 0.068 | 0.033 |
| E3-sybil | 4 | 0.058 | 0.058 | +0.016 | 5 | 0.074 (not a bound) | 0.018 |
| Z9-p2 | 1 | 0.457 | 0.392 (v2) | +0.388 | 2 | 0.350 (searched 1-bit) | 0.360 (`raw_code`) |
| Z9-p2 | 2 | 0.457 | 0.349 (v2); 0.183 with decoder keys + fixed-identity curriculum | +0.345; +0.180 | 2-4 (H 1.1-1.6 bits) | 0.159 (searched 2-bit; count code 0.162) | 0.000 (`raw_code`) |
| Z9-p2 | 4 | 0.457 | 0.229 (v2); 0.095 with keys + curriculum | +0.225; +0.091 | 2-8 (H 1.8-2.1 bits) | <= 0.130 (searched 3-bit) | 0.000 (`raw_code`) |

Notes. Learned E2/E3/Z9 v2 numbers are 3-seed means (pool-trained, final model; `learned_v2_*.json`); "symbols used"
is distinct eval symbols on 2000 dev episodes (B9: codes use far fewer than 2^b symbols; at b = 4 on E2 only 5 of 16,
about 2.3 bits). Blind Z9-p2 references (`blind_refs_z9p2.json`, `parity_learn.blind_references_tabulated`): the
decoder sees only the symbols, with P(Y | symbol multiset) tabulated from 2M fresh episodes, so it uses the joint
structure prior (exactly 3 readers per parity bit); a per-agent decoder (`compression.FairZ9Code`) misses that
coupling and is worse (count code 0.308 instead of 0.162). Raw readings without identity 0.316; bit identity plus
reading (5.49 bits) 0.0151. These reproduce the review's 0.31 / 0.160 / 0.151 / 0.0147 within Monte Carlo error.
Symbol entropy per arm is logged for every `parity_learn` run (table below); the v2 runs logged distinct symbols only.

### Steps 1, 3, 4, 5: results (fresh episodes, best-val checkpoint, 3 seeds, 8000 steps of 512)

Runs: `aggcomp-pl{c,s1-seed*,s35-seed*}-20261022` on L4 (spot, and on-demand after stockouts), harness commit
`ece51b7`, about $2.2 in total; JSON `aggbench/compression_results/parity_learn.json` (54 runs, with every curve of
validation and training KL). The tables below are generated from the JSON by `scripts/render_parity_learn.py --write`
(gap = dev KL minus the uncompressed reader trained on the same stages, else the best uncompressed reader on the same
spec). CAUTION (gate 2026-10-22): every "decoder keys = yes" row in `parity_learn.json` used concatenated unit-scale
keys that starve the encoder of gradient (~2e-4); those rows are superseded by the in_norm reruns
(`parity_learn_batch3.json`). The fixed-identity curriculum is STRUCTURE-INFORMED (its k = 2, 3 subtasks are built from
the true parity bits), not "Y labels only".

### Batch 3 (2026-10-22): key path, the leak-reduced primary benchmark, matched biased bits, fair Z8/E4

Gate on the steps above: `outbox/gate-learnability-steps0-5-2026-10-22.md`. Runs on L4 (on-demand after spot
stockouts): `aggcomp-rr*`, `aggcomp-gf-*`, `aggcomp-l0-*`, `aggcomp-l1-*`, `aggcomp-ks-*` (2026-10-22), about $6 in total;
JSON `compression_results/parity_learn_{rr,gate_lf,keyscale}.json` with the harness commit of each run. Every
learned arm: fresh episodes, best-val checkpoint, 3 seeds, 8000 steps of 512 (anchors 16000 where marked).

**Primary parity benchmark (from now on): leak-reduced Z9-p2**, `Z9Spec(readers_random=True, readers_alpha=1.0,
readers_per_bit=6)` (zoo: `make_env("Z9", readers_random=True, readers_alpha=1.0, readers_per_bit=6)` grouped p = 2):
13 agents with two readings each, per-episode bit weights ~ Dirichlet(1) so readers per bit are uneven. Prior KL
0.270. The count code (number of 1-readings among parity readings, joint-prior decoder) reaches 0.2595: it keeps 0.010
nats, 3.9% of the information. That holds for the count code only: a searched blind 4-bit code (per-agent decoder)
reaches 0.216, about 20% of the gap, so other cheap statistics remain (DeepSets control: batch 5) (fixed scheme: 0.29 of 0.45, 64%; readers random with 3 per bit: 0.053 of 0.308, 17%). The
leak cannot be removed exactly in this family: Y = (number of 1 bits) mod 2, so any equal-weight count of corrected
readings estimates it; uneven random weights only blur it, and Dirichlet weights smaller than 1 remove more of it but
leave bits unread (alpha 0.25: prior KL 0.015). So "leak-reduced", not "leak-free". Blind references on it
(`blind_refs_z9p2.json`): raw readings 0.263, searched 2-bit 0.254, searched 4-bit (per-agent decoder) 0.216,
bit identity + reading 0.006 at 5.49 bits. With decoder-side keys the decoder knows each agent's structure, so 2 raw
bits per agent are lossless: the information bound with keys is 0 from b = 2.
Anchor = best uncompressed reader: ISAB 0.0010 (16000 steps; transformer 0.003; transformer under the k curriculum
0.010).

**Item 1, the key path.** Diagnosis (logged encoder gradient and message sd): every Gaussian-channel arm on the fixed
Z9-p2, keys or not, has encoder gradients of 2e-4 to 1e-3 and message sd 0.59 (the channel noise alone: the encoder
sends nothing) and stays at the prior, 0/3 at b = 2 and b = 4. The keys anomaly is visible only with the float
message: no keys 3/3 escape (to the count level, 0.13-0.15); keys through separate LayerNorm'd projections plus 1000
steps keys-off: 1/3 (one seed to 0.018, encoder grad 0.7 vs 1e-3 in the stuck seeds); keys concatenated x 0.1 (the
gate's variant): 3/3 (0.006, 0.131, 0.134). So the LayerNorm projection does NOT fix it; down-weighting keys does, and
the gate's pass rule (keys escape at least as often as no keys at every b) holds for keys x 0.1 (b = 2: 0 vs 0,
b = 4: 0 vs 0, inf: 3 vs 3) and fails for the LayerNorm version at inf (1 vs 3). Gate on b481850: x0.1 gives 0.134 / 0.006 / 0.131, so
only 1 of 3 seeds goes past the count level and the other two equal the no-keys arm: scaling removes the harm, keys do
not help; the x0.1 run had no warm-up while the LayerNorm run had 1000 steps (unmatched; the matched rerun is in batch
5), and at b = 2 and 4 the pass rule is vacuous (0 vs 0). The keys arms of the batch-3 runs used the LayerNorm version.
Plain label-only training at finite b almost never escapes: all 54 leak-reduced plain runs stayed at the prior, and of
the fixed-scheme runs only one (step 1, no keys, b = 2, seed 0) left it, to 0.358 (above the count level). Every
finite-b run used the Gaussian channel, so channel and supervision are confounded.

**Item 2, the leak-reduced benchmark at b = 2, 4, 6** (gap to the best uncompressed reader, 0.0010; to the anchor
under the same curriculum, 0.010, in the table): label-only dvib (end to end), self-summary and random encoder sit at
the prior (0.270) in 54 of 54 runs, with and without keys. The structure-informed k curriculum with decoder keys
escapes: b = 2 0.178 (2/3; 0.075-0.248), b = 4 0.057 (3/3), b = 6 0.017 (3/3; 0.013-0.022, best step 7999). Gap at
b = 6: +0.007 against the transformer trained on the same curriculum (0.010; the curriculum makes the anchor worse, so
this flatters the arm) and +0.016 against ISAB (0.0010, 16000 steps, no curriculum). It used the LayerNorm key path;
with keys the information bound is 0 from b = 2, where the arm sits at 0.178.
Without keys the curriculum gets 0.267 / 0.234 / 0.203. Symbol entropy per agent (curriculum + keys): 0.8-1.6 bits at b = 2, 1.7-2.0 at b = 4,
2.5-3.2 at b = 6: the codes use well under b bits. Readers-random with 3 per bit (`parity_learn_rr.json`) gives the same picture: plain arms at
the prior, curriculum + keys 0.169 / 0.049 / 0.037.

**Item 3, biased-bit curriculum with keys matched** (P(bit = 1) 0.8 -> 0.65 -> 0.5, Y labels only; this ran on
readers-random with 3 per bit, NOT the leak-reduced spec; batch 5 reruns both arms on the leak-reduced spec):
with a float message it escapes fully with or without keys (0.002 / 0.003, equal to the anchor trained on the same
biased curriculum, 0.005, and to ISAB 0.001); at b = 4 it reaches 0.078 with keys and 0.249 without (3/3 each). The 0.249 sits on the blind count / 2-bit
plateau (0.256 / 0.253) and above the blind 4-bit code (0.135); the keys arm had no warm-up. So
low-order signal makes the parity learnable from Y alone; under a 4-bit channel the encoder still needs the decoder to
know the structure. In the review's terms: escape at b = inf, and at b = 4 only with keys, so compression (sending
identity) is the limit, not parity.

**Item 4, fair closed forms for Z8 and E4** (`compression_results/fair_z8e4.json`; the decoder sees messages and the
structure prior only): Z8 0.094 / 0.045 / 0.032 / 0.030 at b = 1 / 2 / 4 / 8 (oracle-structure 0.075 / 0.034 / 0.018 /
0.017); E4 0.147 / 0.116 / 0.100 / 0.099 (oracle-structure 0.129 / 0.094 / 0.082 / 0.080). Like E2 these are
Monte Carlo decoders over 256 sampled structures, achievable references rather than bounds.

Anchors (MF2): the fixed-scheme transformer anchor trained twice as long reaches 0.002 (best step 15800-15999, so
still improving); under the k curriculum 0.005. ISAB anchors also peak at the last step (0.001); every anchor is an
upper bound on what the uncompressed reader can reach. Gaps in the table use the uncompressed reader trained
on the same stages where one exists, else the best uncompressed reader on the same spec.

Tables generated from the JSON (`python3 scripts/render_parity_learn.py --write`):


### Batches 4 and 5 (2026-10-22): x0.1 keys, the biased-bit curve, what blocks 2-4 bit channels

Runs: `aggcomp-b4-*` (harness `89474da`) and `aggcomp-b5-*` (`d8e2cab`), L4, $2.59 for both; JSON
`compression_results/parity_learn_batch{4,5}.json`. Every merged JSON now records the commit and a CHECKED dirty flag:
each run's harness sha256 (computed on the VM) must equal the sha256 of that commit's `aggbench/` in git
(`render_parity_learn.merge`); all merged files pass. Spec: leak-reduced Z9-p2 unless noted; 3 seeds; 8000 steps.

**Same-data anchors (gate MF2/MF4).** Uncompressed readers trained 8000 steps on the arms' data: ISAB 0.003 (plain),
0.003 (k curriculum), 0.002 (biased-bit curriculum); transformer 0.006 (plain), 0.007 (biased curriculum). Gaps in the
tables use the reader trained on the same stages. **DeepSets on the leak-reduced spec stays at the prior (0.270, 0/3)**:
mean pooling finds no cheap statistic there, although a searched blind 4-bit per-agent code keeps about 20% of the gap
(0.216; README above).

**Item 1, x0.1 keys on the leak-reduced spec** (keys scaled by 0.1 and concatenated; the variant that removed the
harm). Structure-informed k curriculum + keys, b = 2 / 4 / 6: 0.129 / 0.023 / 0.019 without warm-up, 0.113 / 0.024 /
0.023 with 1000 steps keys-off, against 0.178 / 0.057 / 0.017 with the LayerNorm key path; gap to the same-curriculum
ISAB (0.003): +0.11 / +0.02 / +0.02. Biased bits + keys at b = 4: 0.051 (no warm-up), 0.037 (warm-up). Matched key
comparison on the fixed spec (both with 1000 steps warm-up): x0.1 keys with a float message escape in 3/3 seeds to
0.011 (0.007-0.012), past the count level that the no-keys arm reaches (0.13-0.15) and past the LayerNorm version
(1/3); at b = 2 and 4 both stay at the prior (0/3, the pass rule is vacuous there).

**Item 2, biased bits with Y labels only (no structure-informed labels), keys x0.1, b = 2 / 4 / 6 / 8:**
0.160 / 0.051 / 0.020 / 0.049 (b = 8: 0.108, 0.017, 0.022), symbol entropy 1.2 / 2.1 / 3.0 / 3.5 bits per agent;
gap to the ISAB trained on the same biased curriculum (0.002): +0.158 / +0.049 / +0.018 / +0.047. Without keys the same
curriculum stays on the blind 2-bit plateau at every b: 0.259 / 0.252 / 0.252 / 0.249 (blind count code 0.2595,
searched 2-bit 0.254). So biased bits with Y labels do not close the gap at any b tried: the best is b = 6, 0.018
nats above the anchor, and it needs decoder-side keys (the decoder is told which bit each agent read, outside the b
bits; with keys 2 raw bits per agent are already lossless, so the remaining gap is learning, not bits). The b = 8 runs
are worse than b = 6 in 1 of 3 seeds.

**Item 3, what blocks the 2-4 bit channels.** In every label-only run that stays at the prior, the encoder sends
nothing: its output u stays near 0, tanh(u) is about 0, and the message the decoder sees is the channel noise alone
(logged message sd 0.59 = the noise sd 0.577); the encoder gradient norm is 2e-4 to 1e-3, against 0.2-0.8 in float
runs that escape. A small signal inside N(0, 1/3) noise is invisible, so the decoder learns to ignore the message
slot; once it ignores it, the encoder gets no gradient; and the parity target gives no first-order signal to break the
symmetry (one agent's message is uncorrelated with Y until the others carry information). The noiseless float channel
lets a tiny signal through, which then grows. Tried fix: a channel-noise schedule on the plain label-only arm,
sigma^2 from 0.01 annealed geometrically to 1/3 over the first 4000 steps (b = 2 / 4 / 6, no keys, checkpoints chosen
only at the final noise): 0/9 escapes, all at the prior 0.270, while the codes do use their symbols (entropy 1.5-1.8 /
3.0-3.2 / 3.3-4.6 bits). Lower noise alone does not help on the leak-reduced spec; what does is low-order signal in
the task (biased bits or the k curriculum) together with decoder-side keys. FSQ and joint-learning remedies
(alternation, warm starts) are gemini's.

### Batch 6 (2026-10-22): fixed-power channel and masked-agent pretraining (Y labels only)

Runs `aggcomp-b6k{0,1}-seed*` (harness `a40442a`, sha256-checked clean), $1.97, JSON
`compression_results/parity_learn_batch6.json`. Leak-reduced Z9-p2, b = 2 / 4 / 6, keys off and x0.1 keys (1000 steps
warm-up), 3 seeds, fresh episodes, 8000 Y steps. Matched anchor: ISAB trained 8000 steps on the same data, 0.003.
- **Fixed power** (`power_norm`): the pre-channel message is standardized per dimension over the batch (BatchNorm
  without affine), so E[s^2] = 1 exactly: the power constraint enforced instead of "<= 1"; same N(0, 1/3) noise, so
  the Gaussian capacity is still 1 bit per dimension; eval sends the sign. Logged message power: 1.00 in every run
  (it was 0.01 with tanh at initialization).
- **Masked-agent pretraining** (`pretrain=2000`, no Y): one agent per episode is held out; its message is replaced by a
  query built from its own key, and the decoder predicts its two readings from the other agents' b-bit messages. The
  overlap (several readers per bit) makes the task informative. Then 8000 steps on Y with the same encoder and decoder.
  In keys-off arms the held-out agent's key is still used as the query during pretraining (it defines the task); the
  Y phase has no keys.

| arm | decoder side information | b = 2 | b = 4 | b = 6 |
|---|---|---|---|---|
| fixed power | none (blind) | 0.267 (0/3) | 0.238 (2/3) | 0.164 (3/3); 0.093 after sign fine-tune |
| pretraining | blind at inference; held-out agent's key used as the pretraining query | 0.254 (0/3) | 0.144 (3/3) | 0.130 (3/3) |
| both | blind at inference; held-out agent's key used as the pretraining query | 0.267 (0/3) | 0.228 (2/3); 0.175 after fine-tune | 0.122 (3/3); 0.059 after fine-tune |
| fixed power | STRUCTURE-KNOWN: decoder reads 14 uncompressed floats per agent (x0.1 keys) outside b | 0.174 (2/3) | 0.195 (1/3) | 0.121 (2/3) |
| pretraining | STRUCTURE-KNOWN: decoder reads 14 uncompressed floats per agent (x0.1 keys) outside b | 0.058 (3/3) | 0.009 (3/3) | 0.012 (3/3) |
| both | STRUCTURE-KNOWN: decoder reads 14 uncompressed floats per agent (x0.1 keys) outside b | 0.021 (3/3) | 0.017 (3/3); 0.009 after fine-tune | 0.017 (3/3); 0.009 after fine-tune |

Mean dev KL (escapes = seeds below 0.9 x the prior 0.270); "after sign fine-tune" = the decoder fine-tuned 1000 steps on
the sign messages it is scored on (encoder frozen, Y labels). Prior and earlier plain arms: 0.270.
- **Both fixes work, and pretraining is the stronger one.** With x0.1 keys, pretraining alone reaches 0.009 at b = 4
  (gap +0.006 to the 0.003 anchor) and 0.012 at b = 6. Pretraining plus fixed power reaches 0.021 already at b = 2
  (2 of 3 seeds at 0.009-0.010). These arms are STRUCTURE-KNOWN (the decoder reads every agent's key outside
  the b bits; with keys 2 raw bits are lossless), trained on Y with no curriculum, against 0.270 (prior) for the
  earlier plain arms (but those used the LayerNorm key path on another harness: the same-harness controls are in
  batch 7), 0.113-0.178 for the structure-informed curriculum at b = 2 and 0.160 for biased bits at b = 2.
- **The blind arm (no keys at inference; the held-out agent's key is the pretraining query) leaves the 0.25 plateau**
  from b = 4: pretraining 0.144 at b = 4 (3/3) and 0.130 at b = 6;
  both fixes with the sign fine-tune 0.059 at b = 6 (0.049-0.076). That beats every blind closed form below 5.49 bits
  (searched 4-bit code 0.216, count code 0.2595), but not the 5.49-bit hand code (0.006): the blind gap to the anchor
  is still +0.14 at b = 4 and +0.056 at b = 6 (after fine-tune). At b = 2 nothing blind moves (0.25-0.27; the blind
  2-bit references are 0.254).
- **Mechanism, from the logs.** Encoder gradient norm (median, steps < 3000): plain gauss arms 2e-4 to 1e-3 (batch 3);
  fixed power 3e-3 to 0.3; pretraining 0.005-0.1; both 0.09-0.32. Fixed power alone is not enough in every seed: the
  runs that stay at the prior keep an encoder gradient of 1e-4 to 4e-4 despite unit message power (the decoder learned
  to ignore the message channel anyway: 4 such seeds with keys, 2 without); pretraining removes those failures
  (18 of 18 x0.1-key runs escape with pretraining, against 5 of 9 with fixed power alone).
- **Symbol entropy:** fixed power uses most of the code (1.6-2.0 / 3.1-3.8 / 4.0-5.2 bits at b = 2 / 4 / 6);
  pretraining alone uses less (1.6-1.9 / 2.0-3.3 / 2.6-3.8 bits).

### Batch 7 (2026-10-22): same-harness controls, step-matched anchors, fully blind pretraining

Gate on batch 6: `outbox/gate-batch6-2026-10-22.md`. Runs `aggcomp-b7{c,ab}-seed*` (harness `17654b4`, sha256-checked
clean), $2.44, JSON `compression_results/parity_learn_batch7.json`. Leak-reduced Z9-p2, 3 seeds, fresh episodes.
Code fixes in this harness: the key warm-up now also applies inside masked pretraining (in batch 6 the keys were on
during pretraining although the record says key_warmup = 1000), symbol entropy is computed in eval mode, and the
renderer keys on pretraining (an arm with 2000 + 8000 steps is no longer compared as "same stages" with an 8000-step
anchor).

**Labels (gate item 1).** Arms with x0.1 keys are STRUCTURE-KNOWN: the decoder reads 14 uncompressed floats per agent
(each agent's key) outside the b bits, and with keys 2 raw bits per agent are lossless. The batch-6 "blind" pretraining
arms are blind at inference but used the held-out agent's key as the pretraining query (training only).

**Same-harness plain controls (gate item 2):** tanh channel, no pretraining, concatenated keys as in batch 6, at 8000
and at 10000 steps (the pretrained arms get 2000 + 8000): 36 of 36 runs stay at the prior (0.267-0.270), blind and
structure-known, at b = 2, 4, 6. So the batch-6 escapes come from the fixes, not from the key path, harness or steps.

**Step-matched anchors (gate item 3), primary = the same-decoder transformer** (CLS transformer, 2 layers, width 64,
reading every raw token): 10000 steps 0.005 (0.007, 0.004, 0.004); 11000 steps (for the fine-tune columns) 0.005;
with the same masked-agent pretraining (2000 + 8000) 0.006. ISAB: 0.002 at 10000, 0.0015 at 11000.

**Gaps to the same-decoder transformer (10000 steps; 11000 for the after-fine-tune numbers):**

| arm (batch) | side information | b = 2 | b = 4 | b = 6 |
|---|---|---|---|---|
| plain (7) | none or keys | +0.265 | +0.265 | +0.265 |
| masked pretraining (6) | structure-known (keys) | +0.053 | +0.004 | +0.007 |
| masked pretraining + fixed power (6) | structure-known (keys) | +0.016 | +0.012; +0.004 after fine-tune | +0.012; +0.004 after fine-tune |
| masked pretraining (6) | blind at inference, key as pretraining query | +0.249 | +0.139 | +0.125 |
| masked pretraining + fixed power (6) | blind at inference, key as pretraining query | +0.262 | +0.170 after fine-tune | +0.054 after fine-tune |
| self-reconstruction pretraining (7) | blind: keys are a pretraining reconstruction target, never an input | +0.265 | +0.184 | +0.148 |
| self-reconstruction + fixed power (7) | blind: keys are a pretraining reconstruction target, never an input | +0.255 | +0.219; +0.165 after fine-tune | +0.146; +0.076 after fine-tune |

Against ISAB (0.002 / 0.0015) every gap is about 0.003 larger. Every anchor was still improving at its last step
(best step = last eval in all seeds), so these gaps are LOWER BOUNDS; batch 8 trains the anchors 4x and 8x longer.

**Fully blind pretraining (gate item 5).** Self-reconstruction: each agent's message goes through the channel, and the
decoder predicts every agent's two readings and the var id of each reading at the agent's own slot from all agents'
messages. Keys are a pretraining reconstruction target (the agents' own var ids), never a decoder input. It still escapes at b = 4 and 6 in 3 of 3 seeds (0.189 and 0.153;
with fixed power and the sign fine-tune 0.081 at b = 6, seeds 0.120 / 0.071 / 0.053), but not at b = 2 (0/3). It is
weaker than the key-queried masked pretraining at b = 4 (0.189 vs 0.144) and close at b = 6 (0.153 vs 0.130), and it
beats every blind closed form below 5.49 bits (searched 4-bit 0.216, count 0.2595). Encoder gradient (median, steps
< 3000): 2e-6 to 1e-5 at b = 2 (stuck), 5e-3 to 2e-2 at b = 4-6, 0.03-0.23 with fixed power; message power 0.95-1.00.

**Where this leaves Kelvin's test** (compressed consensus learns the aggregation as well as an uncompressed
transformer on the same data): met within 0.004 nats at b = 4 when the decoder knows the structure (keys outside b,
masked pretraining, Y otherwise); not met blind: the best fully blind arm is 0.076 nats behind at b = 6 and 0.165 at
b = 4, and nothing blind escapes at b = 2.

### Batch 8 (2026-10-22): 4x steps, factorised message, queried encoder, securities with pretraining

Runs `aggcomp-b8{p,qb,qk,a,s}-*`: parity arms at harness `6118e08` (scale, factorised), `2ad41bc` (queried) and
`be95f22` (8x anchor), all sha256-checked clean; securities at `6118e08` (sha256 matches). $6.75 in total. JSON
`compression_results/parity_learn_batch8_{scale_factor,query,anchor8x}.json`, `securities_msg_lr_pretrain_power.json`.
Leak-reduced Z9-p2, 3 seeds, fresh episodes. Primary anchor = the same-decoder transformer, step-matched: 0.0050 at
10000 steps (2000 pretraining + 8000), 0.0048 at 11000 (with the sign fine-tune), 0.0010 at 40000 (4x); trained 8x
(80000 steps) it reaches 0.0009, so the 4x anchor is close to converged. ISAB 4x: 0.0004. The after-fine-tune gaps of the 4x
arms (41000 steps) use the 40000-step anchor; with the 80000-step anchor (0.0009) they change by under 0.0002.

| item | arm | b | dev KL (3 seeds) | gap to transformer | after sign fine-tune (gap) |
|---|---|---|---|---|---|
| 1, scale 4x | blind self-reconstruction + fixed power, 8000 + 32000 steps | 6 | 0.130 (0.108-0.147) | +0.129 | 0.052 (+0.051) |
| 1, scale 4x | same | 8 | 0.083 (0.074-0.090) | +0.082 | 0.025 (+0.024) |
| 1, reference 1x | same, 2000 + 8000 steps (batch 7) | 6 | 0.151 | +0.146 | 0.081 (+0.076) |
| 2, factorised | id bits (pretrained on the agent's own key) + value bits (masked prediction queried by the agent's own id bits), blind | 6 (4 + 2) | 0.143 (0.131-0.152) | +0.138 | 0.128 (+0.123) |
| 2, factorised | same | 8 (6 + 2) | 0.082 (0.065-0.094) | +0.077 | 0.061 (+0.056) |
| 2, factorised | same + fixed power | 6 | 0.192 | +0.187 | 0.119 (+0.114) |
| 2, factorised | same + fixed power | 8 | 0.133 (0.087-0.189) | +0.128 | 0.065 (+0.060) |

**Item 1.** Four times the steps (and an anchor trained as long) narrows the blind gap but does not close it: after the
sign fine-tune +0.051 at b = 6 and +0.024 at b = 8 (1x: +0.076 at b = 6). The information fits (a blind hand code
reaches 0.006 at 5.49 bits), so the remaining gap is learning.
**Item 2.** Like for like at b = 6 (fixed power, 2000 + 8000 steps) the factorised message is no better than plain
self-reconstruction: 0.192 vs 0.151. At b = 8 the comparison is untested at matched steps: the factorised arm at 10000
steps (0.082) equals the unfactorised arm at 40000 steps (0.083), and no unfactorised b = 8 run at 10000 steps exists.
Keys are a pretraining target of the id bits (each agent's own key, agent-local), never a decoder input.

**Queried encoder (Kelvin: the consensus queries the agent before compression).** Each agent, in a random order,
reads the consensus (all earlier agents' b-bit messages as received; unbounded, read-only; attention) before writing
its own b-bit message; the decoder reads only the messages (prefix locality is tested). Same pretraining, fixed power,
2000 + 8000 steps, one-shot rerun on the same harness. Query gain = queried minus one-shot (negative = queried better):

| arm | b = 2 | b = 4 | b = 6 |
|---|---|---|---|
| blind, self-reconstruction: one-shot / queried | 0.260 / 0.265 | 0.224 / 0.222 | 0.151 / 0.117 |
| query gain (after sign fine-tune) | +0.005 (+0.008) | -0.002 (-0.005) | **-0.034 (-0.025)** |
| structure-known (x0.1 keys), masked pretraining: one-shot / queried | 0.052 / 0.103 | 0.026 / 0.026 | 0.019 / 0.019 |
| query gain (after sign fine-tune) | +0.051 (+0.032) | 0.000 (+0.001) | 0.000 (+0.000) |

Gaps to the transformer (0.005): blind queried b = 6 +0.112 (after fine-tune 0.056, +0.051); structure-known one-shot
b = 4 +0.021 (after fine-tune 0.011, +0.006). The query helps only the blind arm and only at b = 6, in 2 of 3 seeds (queried 0.107 / 0.104 /
0.138 against one-shot 0.179 / 0.154 / 0.120, paired by seed), and hurts the structure-known arm at b = 2. With the reference sequential design the consensus adds little that the decoder
cannot already combine.

**Item 3, securities as the message + masked pretraining + fixed power** (securities_msg arms; the pretraining query is
the held-out agent's own key, training only; fixed power standardises the forecast that sets the trade):
oracle securities + settlement [structure-informed: settlement observes all 6 latent bits; the market readout is given
Y(omega)]: b = 6 0.014 (0.013-0.017; 0.037 without), b = 8 0.009 (0.010 without): pretraining closes most of the
quantised-trade loss at b = 6 (closed form 0.006 / 0.0018). Learned securities with settlement collapse to the prior
with pretraining + power (0.271 at b = 6 and 8; 0.139 / 0.112 without); Y-only learned securities stay at the prior.
Gap of the oracle-securities arm to the transformer: +0.009 at b = 6, +0.004 at b = 8 (it uses more side information
than the anchor).

### Batch 9 (2026-10-22): blind self-reconstruction + FSQ channel, 4x steps; one-shot and queried

Runs `aggcomp-b9{o,q}-seed*` (harness `ef8f14e`, sha256-checked clean; two queried VMs were preempted and rerun from
scratch, reproducing seed 1 b = 6 exactly), $6.07 with the securities split below. JSON
`compression_results/parity_learn_batch9_fsq.json`. Leak-reduced Z9-p2, 3 seeds, fresh episodes, blind (keys are a
pretraining reconstruction target of each agent's own message, never a decoder input), 8000 self-reconstruction +
32000 Y steps. FSQ channel (Mentzer et al. 2309.15505; gemini's 1-D 4-level variant): b / 2 dimensions with 4 levels
each, straight-through rounding, no noise, usage term 0.05 per dimension; exactly b bits (`Channel(b)` checks the
base-4 symbols). Anchor: the same-decoder transformer at 40000 steps, 0.0010 (80000 steps: 0.0009).

| arm | b | dev KL (seeds 0, 1, 2) | gap to transformer | after sign fine-tune | symbol entropy (bits) |
|---|---|---|---|---|---|
| one-shot, FSQ | 6 | 0.082 (0.081, 0.077, 0.088) | +0.081 | 0.076 | 5.0-5.2 |
| one-shot, FSQ | 8 | **0.018** (0.013, 0.0205, 0.019) | **+0.017** | 0.018 | 6.3-6.5 |
| queried, FSQ | 6 | 0.063 (0.063, 0.066, 0.059) | +0.062 | 0.062 | 5.3 |
| queried, FSQ | 8 | 0.022 (0.021, 0.023, 0.022) | +0.021 | 0.022 | 6.5-6.7 |
| Gaussian + fixed power (batch 8, 4x) | 6 / 8 | 0.130 / 0.083 | +0.129 / +0.082 | 0.052 / 0.025 | 4.9 / 5.3-5.6 |

- **FSQ is the best blind channel so far.** At b = 8 it reaches 0.018 (gap +0.017) without the sign fine-tune, against
  0.083 (+0.082) for the Gaussian fixed-power channel at the same steps; at b = 6, 0.082 against 0.130. It uses more of
  the code (6.3-6.5 of 8 bits). The blind hand code reaches 0.006 at 5.49 bits, so b = 8 is now within 0.012 of
  that closed form and 0.017 of the transformer.
- **Query gain (queried minus one-shot, same harness):** b = 6 -0.019 (a consistent direction in 3 of 3 seeds, sign test p = 0.125: -0.017, -0.011, -0.029);
  b = 8 +0.005 (consistent in 3 of 3 seeds, p = 0.125: +0.008, +0.002, +0.004). CAVEAT (gate on batch 9): in these
  queried runs the FSQ usage term covered only the LAST agent (one-shot: all agents), so both query gains
  (b = 6 and b = 8) are confounded; fixed in `34e5e03` and rerun in batch 10. The query helps when the budget is tight and costs a little
  when it is not. Encoder-gradient logs of the queried runs in this batch are invalid (they read the unused one-shot
  encoder; fixed in `f885a3b`).

**Learned-securities collapse, separated (settle_market, 1 seed, harness `ef8f14e`; JSON
`securities_msg_lr_collapse_split.json`):** masked pretraining alone 0.145 (b = 6) / 0.098 (b = 8), like the arm
without either (0.139 / 0.112); fixed power alone 0.270 / 0.270 (the prior). The collapse in batch 8 comes from the
fixed-power standardisation of the forecast that sets the trade, not from pretraining.

### Batch 10 (2026-10-22): 8x steps for the best blind arms; usage-term fix for the queried arm

Runs: one-shot b = 8 `aggcomp-b10o-seed*` (harness `137a30b`), queried b = 6 `agc-qf0` and queried b = 8 with the
usage fix `agc-q8` (harness `34e5e03`); sha256-checked clean. JSON `compression_results/parity_learn_batch10_{oneshot,
queried}.json`. Gate on batch 9: in the queried arm the FSQ usage term covered only the LAST agent (one-shot: all
agents); fixed in `34e5e03` (`test_queried_fsq_usage_term_averages_all_agents`). The first queried b = 6 launch used
the buggy commit and was stopped before training; all three reruns (`agc-qf{0,1,2}`) use the fix (checked: worktree
HEAD 34e5e03 on both VMs that survived an API outage). Cost of batch 10: $6.01. Anchor: transformer at 80000 steps 0.0009
(40000: 0.0010).

| arm | steps | dev KL (seeds) | best-val step | gap to transformer | after sign fine-tune |
|---|---|---|---|---|---|
| one-shot FSQ, b = 8 | 16000 + 64000 | 0.0236 (0.0226, 0.0243, 0.0239) | 43400-58400 | +0.023 | 0.0215 |
| one-shot FSQ, b = 8 (batch 9) | 8000 + 32000 | 0.0176 (0.0134, 0.0205, 0.0188) | 27800-31000 | +0.017 | 0.0175 |
| queried FSQ, b = 6, usage fix | 16000 + 64000 | 0.0603 (0.0674, 0.0681, 0.0454) | 28200-53000 | +0.059 | 0.0561 |
| queried FSQ, b = 6 (batch 9, usage bug) | 8000 + 32000 | 0.0626 (0.0633, 0.0660, 0.0586) | 24400-30800 | +0.062 | 0.0618 |
| queried FSQ, b = 8, usage fix (1 seed) | 8000 + 32000 | 0.0206 (seed 0) | 31800 | +0.020 | 0.0206 |

- **More training does not close the gap.** At 8x steps the one-shot b = 8 arm is slightly WORSE than at 4x (0.0236 vs
  0.0176; every 8x seed is above every 4x seed), and its best validation step falls at 43000-58000 of 64000: it
  plateaus. Queried b = 6 at 8x (0.060, with the usage fix) is about where it was at 4x (0.063, usage bug): seeds 0 and 1 are
  slightly worse, seed 2 better (0.045). The gap to the 5.49-bit blind hand
  code (0.006) stays at about 0.012-0.018 for b = 8, so what limits the blind arm is the code it learns, not the number
  of steps.
- **Query at b = 8 with the usage fix:** queried 0.0206 vs one-shot 0.0134 on seed 0, a cost of +0.007 as before (one
  seed; the batch-9 query gains at b = 6 and 8 were both confounded by the usage bug).

Tables generated from the JSON (all parity_learn files):

<!-- PARITY-LEARN:BEGIN -->
`parity_learn.json` (harness `ece51b77b6`, 54 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor_fresh | inf | none / transformer | no | Y labels only | 0.004 (0.004, 0.003, 0.004) | 3/3 | n/a | n/a | 0.0020 | n/a | n/a | 0.449 |
| step1_keys0 | 2 | gauss / transformer | no | Y labels only | 0.419 (0.358, 0.449, 0.449) | 1/3 | same stages | +0.415 | 0.0020 | 0.93, 1.46, 1.40 | 0.358, 0.450, 0.449 | 0.449 |
| step1_keys0 | 4 | gauss / transformer | no | Y labels only | 0.450 (0.450, 0.450, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 2.84, 2.25, 3.16 | 0.450, 0.450, 0.450 | 0.449 |
| step1_keys0 | inf | float / transformer | no | Y labels only | 0.103 (0.047, 0.130, 0.131) | 3/3 | same stages | +0.099 | 0.0020 | n/a | n/a | 0.449 |
| step1_keys1 | 2 | gauss / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 0.69, 0.00, 0.23 | 0.449, 0.450, 0.450 | 0.449 |
| step1_keys1 | 4 | gauss / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.450, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 3.51, 3.00, 3.56 | 0.382, 0.426, 0.441 | 0.449 |
| step1_keys1 | inf | float / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.449 (0.450, 0.449, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | n/a | n/a | 0.449 |
| step3_curr_aux | 4 | gauss / transformer | yes | latent-label supervision; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.036 (0.044, 0.039, 0.025) | 3/3 | same stages | +0.031 | 0.0020 | 1.93, 1.92, 1.93 | 0.040, 0.037, 0.016 | 0.449 |
| step3_curr_labelonly | 2 | gauss / transformer | yes | Y labels only; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.183 (0.215, 0.258, 0.077) | 3/3 | same stages | +0.179 | 0.0020 | 1.31, 1.14, 1.63 | 0.208, 0.249, 0.074 | 0.449 |
| step3_curr_labelonly | 4 | gauss / transformer | yes | Y labels only; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.095 (0.041, 0.057, 0.187) | 3/3 | same stages | +0.090 | 0.0020 | 2.14, 1.98, 1.78 | 0.038, 0.049, 0.186 | 0.449 |
| step4_fixed_deepsets | inf | none / deepsets | no | Y labels only | 0.136 (0.133, 0.138, 0.136) | 3/3 | n/a | n/a | 0.0020 | n/a | n/a | 0.449 |
| step4_fixed_isab | inf | none / isab | no | Y labels only | 0.090 (0.063, 0.081, 0.126) | 3/3 | n/a | n/a | 0.0020 | n/a | n/a | 0.449 |
| step4_rr_anchor | inf | none / transformer | no | Y labels only | 0.017 (0.006, 0.005, 0.040) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| step4_rr_deepsets | inf | none / deepsets | no | Y labels only | 0.264 (0.242, 0.242, 0.308) | 2/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| step4_rr_dvib_inf | inf | float / transformer | no | Y labels only | 0.104 (0.308, 0.003, 0.003) | 2/3 | same stages | +0.102 | 0.0012 | n/a | n/a | 0.308 |
| step4_rr_isab | inf | none / isab | no | Y labels only | 0.002 (0.002, 0.002, 0.003) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| step5_bias_curr | 4 | gauss / transformer | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.065 (0.065, 0.065, 0.066) | 3/3 | transformer, 8000 steps (`anchor_fresh`, 0.0036) | +0.062 | 0.0020 | 1.66, 1.62, 1.91 | 0.065, 0.065, 0.062 | 0.449 |
| step5_bias_curr | inf | float / transformer | no | Y labels only; biased-bit curriculum | 0.089 (0.130, 0.130, 0.006) | 3/3 | transformer, 8000 steps (`anchor_fresh`, 0.0036) | +0.085 | 0.0020 | n/a | n/a | 0.449 |

`parity_learn_batch10_oneshot.json` (harness `137a30bc27`, 3 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b10_oneshot_blind_recon_fsq_8x | 8 | fsq / transformer | no | Y labels only; pretraining 16000 steps (none: blind self-reconstruction) | 0.024 (0.023, 0.024, 0.024) | 3/3 | transformer, 80000 steps (`b8_anchor_transformer_8x`, 0.0009) | +0.023 | 0.0004 | 6.40, 6.29, 6.46 | 0.020, 0.021, 0.024 | 0.270 |

`parity_learn_batch10_queried.json` (harness `34e5e0307f`, 4 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b10_queried_blind_recon_fsq_4x_usagefix | 8 | fsq / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.021 (0.021) | 1/1 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.020 | 0.0004 | 6.61 | 0.021 | 0.270 |
| b10_queried_blind_recon_fsq_8x | 6 | fsq / transformer | no | Y labels only; pretraining 16000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.060 (0.067, 0.068, 0.045) | 3/3 | transformer, 80000 steps (`b8_anchor_transformer_8x`, 0.0009) | +0.059 | 0.0004 | 5.18, 5.15, 5.38 | 0.058, 0.067, 0.043 | 0.270 |

`parity_learn_batch4.json` (harness `89474da9c5`, 33 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b4_anchor_bias | inf | none / transformer | no | Y labels only; biased-bit curriculum | 0.007 (0.008, 0.007, 0.006) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b4_bias_ks01 | 2 | gauss / transformer | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.160 (0.161, 0.159, 0.160) | 3/3 | same stages | +0.158 | 0.0004 | 1.23, 1.19, 1.18 | 0.161, 0.158, 0.161 | 0.270 |
| b4_bias_ks01 | 4 | gauss / transformer | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.051 (0.066, 0.065, 0.021) | 3/3 | same stages | +0.049 | 0.0004 | 2.07, 2.11, 2.06 | 0.066, 0.058, 0.015 | 0.270 |
| b4_bias_ks01 | 6 | gauss / transformer | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.020 (0.021, 0.026, 0.013) | 3/3 | same stages | +0.018 | 0.0004 | 3.21, 2.55, 3.22 | 0.014, 0.019, 0.011 | 0.270 |
| b4_bias_ks01 | 8 | gauss / transformer | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.049 (0.108, 0.017, 0.022) | 3/3 | same stages | +0.047 | 0.0004 | 3.43, 3.88, 3.21 | 0.098, 0.012, 0.013 | 0.270 |
| b4_curr_ks01 | 2 | gauss / transformer | yes | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.129 (0.225, 0.055, 0.106) | 3/3 | same stages | +0.126 | 0.0004 | 1.09, 1.64, 1.44 | 0.201, 0.051, 0.095 | 0.270 |
| b4_curr_ks01 | 4 | gauss / transformer | yes | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.023 (0.025, 0.021, 0.022) | 3/3 | same stages | +0.020 | 0.0004 | 2.21, 2.68, 2.56 | 0.021, 0.014, 0.015 | 0.270 |
| b4_curr_ks01 | 6 | gauss / transformer | yes | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.019 (0.019, 0.015, 0.023) | 3/3 | same stages | +0.017 | 0.0004 | 3.67, 3.50, 3.36 | 0.013, 0.011, 0.014 | 0.270 |
| b4_noise_sched | 2 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.75, 1.49, 1.56 | 0.270, 0.270, 0.270 | 0.270 |
| b4_noise_sched | 4 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.95, 3.19, 2.98 | 0.270, 0.270, 0.270 | 0.270 |
| b4_noise_sched | 6 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 4.58, 3.34, 4.18 | 0.269, 0.270, 0.270 | 0.270 |

`parity_learn_batch5.json` (harness `d8e2cab166`, 48 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b5_bias_keys0 | 2 | gauss / transformer | no | Y labels only; biased-bit curriculum | 0.259 (0.257, 0.264, 0.257) | 0/3 | same stages | +0.257 | 0.0004 | 1.40, 1.38, 1.40 | 0.256, 0.264, 0.256 | 0.270 |
| b5_bias_keys0 | 4 | gauss / transformer | no | Y labels only; biased-bit curriculum | 0.252 (0.252, 0.252, 0.252) | 0/3 | same stages | +0.250 | 0.0004 | 2.10, 2.08, 2.08 | 0.251, 0.251, 0.251 | 0.270 |
| b5_bias_keys0 | 6 | gauss / transformer | no | Y labels only; biased-bit curriculum | 0.252 (0.252, 0.251, 0.252) | 0/3 | same stages | +0.250 | 0.0004 | 2.16, 2.11, 2.15 | 0.252, 0.251, 0.251 | 0.270 |
| b5_bias_keys0 | 8 | gauss / transformer | no | Y labels only; biased-bit curriculum | 0.249 (0.244, 0.252, 0.252) | 0/3 | same stages | +0.247 | 0.0004 | 2.37, 2.18, 2.09 | 0.244, 0.252, 0.251 | 0.270 |
| b5_bias_ks01_w1000 | 4 | gauss / transformer | yes (1000) | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.037 (0.025, 0.041, 0.044) | 3/3 | same stages | +0.035 | 0.0004 | 1.96, 2.11, 1.99 | 0.022, 0.031, 0.036 | 0.270 |
| b5_curr_ks01_w1000 | 2 | gauss / transformer | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.113 (0.216, 0.068, 0.056) | 3/3 | same stages | +0.111 | 0.0004 | 0.73, 1.69, 1.81 | 0.209, 0.063, 0.048 | 0.270 |
| b5_curr_ks01_w1000 | 4 | gauss / transformer | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.024 (0.023, 0.029, 0.021) | 3/3 | same stages | +0.022 | 0.0004 | 2.83, 2.11, 2.57 | 0.018, 0.022, 0.015 | 0.270 |
| b5_curr_ks01_w1000 | 6 | gauss / transformer | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.023 (0.022, 0.018, 0.029) | 3/3 | same stages | +0.020 | 0.0004 | 2.82, 3.39, 3.18 | 0.015, 0.012, 0.019 | 0.270 |
| b5_lf_deepsets | inf | none / deepsets | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b5_lf_isab | inf | none / isab | no | Y labels only | 0.003 (0.002, 0.003, 0.003) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b5_lf_isab_bias | inf | none / isab | no | Y labels only; biased-bit curriculum | 0.002 (0.002, 0.002, 0.002) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b5_lf_isab_curr | inf | none / isab | no | Y labels only; structure-informed | 0.003 (0.003, 0.002, 0.003) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b5_lf_transformer | inf | none / transformer | no | Y labels only | 0.006 (0.007, 0.006, 0.006) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b5_step1_ks01_w1000 | 2 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.450, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 1.28, 1.32, 1.63 | 0.450, 0.450, 0.450 | 0.449 |
| b5_step1_ks01_w1000 | 4 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 3.62, 3.27, 3.25 | 0.449, 0.449, 0.450 | 0.449 |
| b5_step1_ks01_w1000 | inf | float / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.011 (0.007, 0.012, 0.012) | 3/3 | same stages | +0.007 | 0.0020 | n/a | n/a | 0.449 |

`parity_learn_batch6.json` (harness `a40442affb`, 54 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b6_both_k0 | 2 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.267 (0.267, 0.268, 0.265) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.262 | 0.0004 | 1.98, 1.89, 1.96 | 0.260, 0.263, 0.262 | 0.270 |
| b6_both_k0 | 4 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.228 (0.212, 0.267, 0.205) | 2/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.223 | 0.0004 | 3.71, 3.30, 3.78 | 0.144, 0.246, 0.136 | 0.270 |
| b6_both_k0 | 6 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.122 (0.103, 0.145, 0.120) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.117 | 0.0004 | 4.95, 4.71, 5.03 | 0.049, 0.076, 0.052 | 0.270 |
| b6_both_k1 | 2 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.021 (0.010, 0.045, 0.009) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.016 | 0.0004 | 1.92, 1.96, 1.90 | 0.008, 0.026, 0.007 | 0.270 |
| b6_both_k1 | 4 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.017 (0.018, 0.017, 0.015) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.012 | 0.0004 | 3.85, 3.38, 2.88 | 0.009, 0.010, 0.008 | 0.270 |
| b6_both_k1 | 6 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.017 (0.013, 0.015, 0.024) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.012 | 0.0004 | 4.99, 5.13, 4.76 | 0.007, 0.007, 0.012 | 0.270 |
| b6_power_k0 | 2 | gauss + power / transformer | no | Y labels only | 0.267 (0.267, 0.270, 0.265) | 0/3 | same stages | +0.265 | 0.0004 | 1.91, 1.97, 1.84 | 0.256, 0.270, 0.262 | 0.270 |
| b6_power_k0 | 4 | gauss + power / transformer | no | Y labels only | 0.238 (0.232, 0.271, 0.211) | 2/3 | same stages | +0.235 | 0.0004 | 3.62, 3.63, 3.70 | 0.185, 0.270, 0.174 | 0.270 |
| b6_power_k0 | 6 | gauss + power / transformer | no | Y labels only | 0.164 (0.136, 0.195, 0.162) | 3/3 | same stages | +0.162 | 0.0004 | 5.01, 4.40, 4.88 | 0.065, 0.123, 0.091 | 0.270 |
| b6_power_k1 | 2 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.174 (0.119, 0.132, 0.270) | 2/3 | same stages | +0.171 | 0.0004 | 1.59, 1.83, 1.98 | 0.060, 0.066, 0.270 | 0.270 |
| b6_power_k1 | 4 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.195 (0.043, 0.270, 0.270) | 1/3 | same stages | +0.192 | 0.0004 | 3.13, 3.77, 3.63 | 0.015, 0.270, 0.270 | 0.270 |
| b6_power_k1 | 6 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.121 (0.069, 0.025, 0.270) | 2/3 | same stages | +0.119 | 0.0004 | 4.05, 4.63, 5.16 | 0.029, 0.014, 0.270 | 0.270 |
| b6_pretrain_k0 | 2 | gauss / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.254 (0.257, 0.255, 0.251) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.249 | 0.0004 | 1.88, 1.87, 1.91 | 0.254, 0.254, 0.251 | 0.270 |
| b6_pretrain_k0 | 4 | gauss / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.144 (0.142, 0.144, 0.146) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.139 | 0.0004 | 3.27, 3.18, 3.25 | 0.140, 0.141, 0.146 | 0.270 |
| b6_pretrain_k0 | 6 | gauss / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.130 (0.143, 0.109, 0.139) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.125 | 0.0004 | 3.33, 3.84, 3.34 | 0.139, 0.104, 0.137 | 0.270 |
| b6_pretrain_k1 | 2 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.058 (0.050, 0.051, 0.073) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.053 | 0.0004 | 1.59, 1.59, 1.59 | 0.050, 0.051, 0.072 | 0.270 |
| b6_pretrain_k1 | 4 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.009 (0.007, 0.011, 0.008) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.004 | 0.0004 | 2.24, 2.01, 2.10 | 0.006, 0.010, 0.008 | 0.270 |
| b6_pretrain_k1 | 6 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.012 (0.009, 0.009, 0.019) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.007 | 0.0004 | 3.46, 2.64, 3.20 | 0.008, 0.008, 0.018 | 0.270 |

`parity_learn_batch7.json` (harness `17654b40a8`, 69 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b7_anchor_isab_10000 | inf | none / isab | no | Y labels only | 0.002 (0.002, 0.002, 0.002) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b7_anchor_isab_11000 | inf | none / isab | no | Y labels only | 0.001 (0.001, 0.002, 0.002) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b7_anchor_transformer_10000 | inf | none / transformer | no | Y labels only | 0.005 (0.007, 0.004, 0.004) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b7_anchor_transformer_11000 | inf | none / transformer | no | Y labels only | 0.005 (0.006, 0.005, 0.004) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b7_anchor_transformer_pretrain | inf | none / transformer | no | Y labels only; pretraining 2000 steps (held-out agent's own key (training only)) | 0.006 (0.005, 0.006, 0.007) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b7_plain_k0_10000 | 2 | gauss / transformer | no | Y labels only | 0.270 (0.269, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 0.99, 1.24, 1.53 | 0.267, 0.270, 0.270 | 0.270 |
| b7_plain_k0_10000 | 4 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.67, 1.99, 1.50 | 0.265, 0.270, 0.270 | 0.270 |
| b7_plain_k0_10000 | 6 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.269 | 0.0004 | 4.21, 4.50, 3.06 | 0.270, 0.270, 0.270 | 0.270 |
| b7_plain_k0_8000 | 2 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.29, 1.48, 0.59 | 0.270, 0.270, 0.270 | 0.270 |
| b7_plain_k0_8000 | 4 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.267 | 0.0004 | 1.89, 2.32, 2.97 | 0.265, 0.270, 0.270 | 0.270 |
| b7_plain_k0_8000 | 6 | gauss / transformer | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 4.32, 3.58, 3.19 | 0.268, 0.270, 0.270 | 0.270 |
| b7_plain_k1_10000 | 2 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.269 (0.267, 0.270, 0.270) | 0/3 | same stages | +0.267 | 0.0004 | 1.00, 1.23, 1.37 | 0.267, 0.270, 0.270 | 0.270 |
| b7_plain_k1_10000 | 4 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.269, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.54, 3.03, 3.32 | 0.263, 0.270, 0.270 | 0.270 |
| b7_plain_k1_10000 | 6 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.269 | 0.0004 | 4.74, 3.82, 4.17 | 0.270, 0.270, 0.270 | 0.270 |
| b7_plain_k1_8000 | 2 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.58, 1.51, 1.39 | 0.270, 0.270, 0.270 | 0.270 |
| b7_plain_k1_8000 | 4 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.75, 2.38, 3.30 | 0.269, 0.270, 0.270 | 0.270 |
| b7_plain_k1_8000 | 6 | gauss / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 4.64, 3.62, 4.69 | 0.270, 0.270, 0.270 | 0.270 |
| b7_recon_k0 | 2 | gauss / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.270 (0.270, 0.270, 0.270) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.265 | 0.0004 | 1.99, 1.29, 2.00 | 0.270, 0.270, 0.270 | 0.270 |
| b7_recon_k0 | 4 | gauss / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.189 (0.209, 0.158, 0.200) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.184 | 0.0004 | 3.78, 3.36, 3.11 | 0.201, 0.154, 0.199 | 0.270 |
| b7_recon_k0 | 6 | gauss / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.153 (0.156, 0.157, 0.148) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.148 | 0.0004 | 3.51, 3.52, 3.82 | 0.154, 0.154, 0.142 | 0.270 |
| b7_recon_power_k0 | 2 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.260 (0.256, 0.260, 0.265) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.255 | 0.0004 | 1.87, 1.93, 1.98 | 0.250, 0.255, 0.260 | 0.270 |
| b7_recon_power_k0 | 4 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.224 (0.239, 0.223, 0.211) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.219 | 0.0004 | 3.74, 3.66, 3.74 | 0.168, 0.186, 0.155 | 0.270 |
| b7_recon_power_k0 | 6 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.151 (0.179, 0.154, 0.120) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.146 | 0.0004 | 4.90, 4.82, 5.02 | 0.120, 0.071, 0.053 | 0.270 |

`parity_learn_batch8_anchor8x.json` (harness `be95f227d7`, 3 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_anchor_transformer_8x | inf | none / transformer | no | Y labels only | 0.001 (0.001, 0.001, 0.001) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |

`parity_learn_batch8_query.json` (harness `2ad41bc141`, 36 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_oneshot_blind_recon_power | 2 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.260 (0.256, 0.260, 0.265) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.255 | 0.0004 | 1.87, 1.93, 1.98 | 0.250, 0.255, 0.260 | 0.270 |
| b8_oneshot_blind_recon_power | 4 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.224 (0.239, 0.223, 0.211) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.219 | 0.0004 | 3.74, 3.66, 3.74 | 0.168, 0.186, 0.155 | 0.270 |
| b8_oneshot_blind_recon_power | 6 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction) | 0.151 (0.179, 0.154, 0.120) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.146 | 0.0004 | 4.90, 4.82, 5.02 | 0.120, 0.071, 0.053 | 0.270 |
| b8_oneshot_known_masked_power | 2 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.052 (0.103, 0.022, 0.032) | 3/3 | same stages | +0.047 | 0.0004 | 1.58, 1.98, 2.00 | 0.066, 0.011, 0.011 | 0.270 |
| b8_oneshot_known_masked_power | 4 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.026 (0.027, 0.034, 0.016) | 3/3 | same stages | +0.020 | 0.0004 | 3.60, 3.71, 3.83 | 0.012, 0.013, 0.008 | 0.270 |
| b8_oneshot_known_masked_power | 6 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)) | 0.019 (0.021, 0.021, 0.016) | 3/3 | same stages | +0.014 | 0.0004 | 4.45, 4.79, 4.81 | 0.011, 0.013, 0.008 | 0.270 |
| b8_queried_blind_recon_power | 2 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.265 (0.270, 0.264, 0.262) | 0/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.260 | 0.0004 | 1.95, 1.96, 1.93 | 0.270, 0.264, 0.255 | 0.270 |
| b8_queried_blind_recon_power | 4 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.222 (0.225, 0.234, 0.207) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.217 | 0.0004 | 3.84, 3.52, 3.77 | 0.166, 0.190, 0.140 | 0.270 |
| b8_queried_blind_recon_power | 6 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.117 (0.107, 0.104, 0.138) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.111 | 0.0004 | 5.06, 5.07, 5.35 | 0.046, 0.056, 0.067 | 0.270 |
| b8_queried_known_masked_power | 2 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.103 (0.097, 0.086, 0.127) | 3/3 | same stages | +0.097 | 0.0004 | 1.90, 1.62, 1.78 | 0.045, 0.062, 0.075 | 0.270 |
| b8_queried_known_masked_power | 4 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.026 (0.029, 0.027, 0.021) | 3/3 | same stages | +0.020 | 0.0004 | 3.86, 3.76, 3.48 | 0.014, 0.014, 0.010 | 0.270 |
| b8_queried_known_masked_power | 6 | gauss + power / transformer | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b); pretraining 2000 steps (held-out agent's own key (training only)); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.019 (0.021, 0.014, 0.022) | 3/3 | same stages | +0.013 | 0.0004 | 4.82, 5.19, 4.76 | 0.011, 0.008, 0.013 | 0.270 |

`parity_learn_batch8_scale_factor.json` (harness `6118e08294`, 24 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_anchor_isab_4x | inf | none / isab | no | Y labels only | 0.000 (0.000, 0.000, 0.000) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b8_anchor_transformer_4x | inf | none / transformer | no | Y labels only | 0.001 (0.001, 0.001, 0.001) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| b8_factor | 6 | gauss / transformer | no | Y labels only; pretraining 2000 steps (none: blind (id bits local key reconstruction; value bits masked prediction queried by the agent's own id bits)) | 0.143 (0.131, 0.146, 0.152) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.138 | 0.0004 | 4.29, 4.56, 4.30 | 0.111, 0.133, 0.140 | 0.270 |
| b8_factor | 8 | gauss / transformer | no | Y labels only; pretraining 2000 steps (none: blind (id bits local key reconstruction; value bits masked prediction queried by the agent's own id bits)) | 0.082 (0.094, 0.087, 0.065) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.077 | 0.0004 | 4.83, 5.25, 5.57 | 0.080, 0.068, 0.036 | 0.270 |
| b8_factor_power | 6 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind (id bits local key reconstruction; value bits masked prediction queried by the agent's own id bits)) | 0.192 (0.203, 0.177, 0.195) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.187 | 0.0004 | 4.83, 4.80, 4.64 | 0.117, 0.113, 0.128 | 0.270 |
| b8_factor_power | 8 | gauss + power / transformer | no | Y labels only; pretraining 2000 steps (none: blind (id bits local key reconstruction; value bits masked prediction queried by the agent's own id bits)) | 0.133 (0.121, 0.189, 0.087) | 3/3 | transformer, 10000 steps (`b7_anchor_transformer_10000`, 0.0050) | +0.128 | 0.0004 | 5.60, 5.30, 5.54 | 0.039, 0.127, 0.030 | 0.270 |
| b8_recon_power_4x | 6 | gauss + power / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction) | 0.130 (0.147, 0.136, 0.108) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.129 | 0.0004 | 4.87, 4.86, 4.94 | 0.065, 0.045, 0.046 | 0.270 |
| b8_recon_power_4x | 8 | gauss + power / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction) | 0.083 (0.074, 0.086, 0.090) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.082 | 0.0004 | 5.29, 5.48, 5.64 | 0.029, 0.021, 0.026 | 0.270 |

`parity_learn_batch9_fsq.json` (harness `ef8f14e332`, 12 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b9_oneshot_blind_recon_fsq_4x | 6 | fsq / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction) | 0.082 (0.081, 0.077, 0.088) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.081 | 0.0004 | 5.00, 5.19, 5.16 | 0.068, 0.077, 0.084 | 0.270 |
| b9_oneshot_blind_recon_fsq_4x | 8 | fsq / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction) | 0.018 (0.013, 0.020, 0.019) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.017 | 0.0004 | 6.41, 6.32, 6.45 | 0.013, 0.020, 0.019 | 0.270 |
| b9_queried_blind_recon_fsq_4x | 6 | fsq / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.063 (0.063, 0.066, 0.059) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.062 | 0.0004 | 5.27, 5.30, 5.31 | 0.063, 0.063, 0.059 | 0.270 |
| b9_queried_blind_recon_fsq_4x | 8 | fsq / transformer | no | Y labels only; pretraining 8000 steps (none: blind self-reconstruction); QUERIED encoder (reads earlier agents' b-bit messages, unbounded, read-only) | 0.022 (0.021, 0.023, 0.022) | 3/3 | transformer, 40000 steps (`b8_anchor_transformer_4x`, 0.0010) | +0.021 | 0.0004 | 6.56, 6.52, 6.71 | 0.021, 0.023, 0.022 | 0.270 |

`parity_learn_gate_lf.json` (harness `03246e0f35`, 126 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate_anchor_curr | inf | none / transformer | no | Y labels only; structure-informed | 0.005 (0.005, 0.005, 0.004) | 3/3 | n/a | n/a | 0.0020 | n/a | n/a | 0.449 |
| gate_anchor_long | inf | none / transformer | no | Y labels only | 0.002 (0.002, 0.002, 0.002) | 3/3 | n/a | n/a | 0.0020 | n/a | n/a | 0.449 |
| gate_rr_anchor_bias | inf | none / transformer | no | Y labels only; biased-bit curriculum | 0.005 (0.005, 0.005, 0.005) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| gate_rr_anchor_curr | inf | none / transformer | no | Y labels only; structure-informed | 0.008 (0.006, 0.010, 0.009) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| gate_rr_isab_long | inf | none / isab | no | Y labels only | 0.001 (0.001, 0.001, 0.001) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| gate_step1_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.450 (0.450, 0.450, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 1.43, 1.47, 0.47 | 0.449, 0.450, 0.450 | 0.449 |
| gate_step1_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.449 (0.449, 0.449, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 2.78, 2.26, 3.04 | 0.450, 0.449, 0.449 | 0.449 |
| gate_step1_keys0 | inf | float / transformer (in_norm) | no | Y labels only | 0.137 (0.149, 0.131, 0.131) | 3/3 | same stages | +0.133 | 0.0020 | n/a | n/a | 0.449 |
| gate_step1_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.449 (0.450, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 1.70, 1.89, 1.70 | 0.450, 0.450, 0.450 | 0.449 |
| gate_step1_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.450, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 2.44, 3.27, 1.43 | 0.449, 0.449, 0.449 | 0.449 |
| gate_step1_keys1 | inf | float / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.306 (0.449, 0.018, 0.450) | 1/3 | same stages | +0.302 | 0.0020 | n/a | n/a | 0.449 |
| gate_step3_curr_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only; structure-informed | 0.247 (0.164, 0.357, 0.219) | 3/3 | same stages | +0.242 | 0.0020 | 1.52, 0.93, 1.65 | 0.164, 0.357, 0.216 | 0.449 |
| gate_step3_curr_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only; structure-informed | 0.243 (0.171, 0.395, 0.162) | 3/3 | same stages | +0.238 | 0.0020 | 1.88, 1.81, 1.52 | 0.168, 0.395, 0.162 | 0.449 |
| gate_step3_curr_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.271 (0.074, 0.398, 0.341) | 3/3 | same stages | +0.266 | 0.0020 | 1.65, 0.91, 0.85 | 0.071, 0.398, 0.341 | 0.449 |
| gate_step3_curr_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.072 (0.075, 0.072, 0.070) | 3/3 | same stages | +0.067 | 0.0020 | 1.61, 1.64, 1.53 | 0.072, 0.069, 0.068 | 0.449 |
| lf_anchor | inf | none / transformer | no | Y labels only | 0.003 (0.004, 0.003, 0.003) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| lf_anchor_curr | inf | none / transformer | no | Y labels only; structure-informed | 0.010 (0.013, 0.008, 0.010) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| lf_aware_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.26, 1.45, 1.15 | 0.270, 0.270, 0.270 | 0.270 |
| lf_aware_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.269, 0.270, 0.270) | 0/3 | same stages | +0.267 | 0.0004 | 2.27, 3.09, 3.30 | 0.268, 0.270, 0.270 | 0.270 |
| lf_aware_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.271, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 4.58, 2.33, 3.12 | 0.270, 0.270, 0.270 | 0.270 |
| lf_aware_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.96, 1.69, 1.36 | 0.270, 0.270, 0.270 | 0.270 |
| lf_aware_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.23, 3.11, 0.95 | 0.270, 0.270, 0.270 | 0.270 |
| lf_aware_keys1 | 6 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 3.92, 3.08, 4.44 | 0.270, 0.270, 0.270 | 0.270 |
| lf_curr_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only; structure-informed | 0.267 (0.270, 0.270, 0.261) | 0/3 | same stages | +0.264 | 0.0004 | 0.00, 0.56, 1.74 | 0.270, 0.270, 0.261 | 0.270 |
| lf_curr_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only; structure-informed | 0.234 (0.246, 0.244, 0.213) | 1/3 | same stages | +0.231 | 0.0004 | 2.00, 1.88, 2.74 | 0.243, 0.243, 0.208 | 0.270 |
| lf_curr_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only; structure-informed | 0.203 (0.147, 0.192, 0.269) | 2/3 | same stages | +0.200 | 0.0004 | 3.69, 3.16, 0.98 | 0.133, 0.183, 0.269 | 0.270 |
| lf_curr_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.178 (0.210, 0.075, 0.248) | 2/3 | same stages | +0.175 | 0.0004 | 0.93, 1.57, 0.82 | 0.210, 0.071, 0.231 | 0.270 |
| lf_curr_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.057 (0.054, 0.056, 0.061) | 3/3 | same stages | +0.055 | 0.0004 | 2.01, 1.80, 1.65 | 0.052, 0.052, 0.058 | 0.270 |
| lf_curr_keys1 | 6 | gauss / transformer (in_norm) | yes (1000) | Y labels only; structure-informed; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.017 (0.015, 0.013, 0.022) | 3/3 | same stages | +0.014 | 0.0004 | 3.19, 2.58, 2.45 | 0.013, 0.011, 0.019 | 0.270 |
| lf_isab | inf | none / isab | no | Y labels only | 0.001 (0.001, 0.001, 0.001) | 3/3 | n/a | n/a | 0.0004 | n/a | n/a | 0.270 |
| lf_random_enc_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.18, 1.26, 0.49 | 0.270, 0.270, 0.270 | 0.270 |
| lf_random_enc_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.29, 3.04, 2.92 | 0.270, 0.270, 0.270 | 0.270 |
| lf_random_enc_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.271) | 0/3 | same stages | +0.268 | 0.0004 | 4.26, 2.74, 3.68 | 0.270, 0.270, 0.270 | 0.270 |
| lf_random_enc_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 1.18, 1.26, 0.49 | 0.270, 0.270, 0.270 | 0.270 |
| lf_random_enc_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.29, 3.04, 2.92 | 0.270, 0.270, 0.270 | 0.270 |
| lf_random_enc_keys1 | 6 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.271) | 0/3 | same stages | +0.268 | 0.0004 | 4.26, 2.74, 3.68 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 0.48, 1.91, 0.00 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.39, 0.84, 1.42 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.84, 2.73, 3.52 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys1 | 2 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 0.48, 1.91, 1.25 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys1 | 4 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 2.19, 2.50, 0.32 | 0.270, 0.270, 0.270 | 0.270 |
| lf_self_keys1 | 6 | gauss / transformer (in_norm) | yes (1000) | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.270 (0.270, 0.270, 0.270) | 0/3 | same stages | +0.268 | 0.0004 | 3.71, 3.14, 2.15 | 0.270, 0.270, 0.270 | 0.270 |

`parity_learn_keyscale.json` (harness `69eff2a347`, 9 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyscale01_keys1 | 2 | gauss / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 1.21, 1.41, 1.16 | 0.450, 0.450, 0.449 | 0.449 |
| keyscale01_keys1 | 4 | gauss / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.449 (0.449, 0.449, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 3.60, 2.97, 3.40 | 0.449, 0.450, 0.449 | 0.449 |
| keyscale01_keys1 | inf | float / transformer | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.090 (0.134, 0.006, 0.131) | 3/3 | same stages | +0.087 | 0.0020 | n/a | n/a | 0.449 |

`parity_learn_rr.json` (harness `f729b218bc`, 108 runs):

| experiment | b | channel / decoder | keys (warm-up) | supervision / curriculum | dev KL, mean (per seed) | escapes | reference | gap | best uncompressed reader | symbol entropy (bits) | after sign fine-tune | prior KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyfix_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.449 (0.450, 0.449, 0.449) | 0/3 | same stages | +0.446 | 0.0020 | 1.58, 1.56, 0.56 | 0.451, 0.449, 0.449 | 0.449 |
| keyfix_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.450 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 3.29, 3.40, 3.64 | 0.449, 0.449, 0.450 | 0.449 |
| keyfix_keys0 | inf | float / transformer (in_norm) | no | Y labels only | 0.137 (0.149, 0.131, 0.131) | 3/3 | same stages | +0.133 | 0.0020 | n/a | n/a | 0.449 |
| keyfix_keys1 | 2 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.450 (0.449, 0.450, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 1.81, 1.19, 0.32 | 0.449, 0.450, 0.450 | 0.449 |
| keyfix_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.449 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | 1.98, 1.18, 1.42 | 0.449, 0.449, 0.450 | 0.449 |
| keyfix_keys1 | inf | float / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.449 (0.449, 0.449, 0.450) | 0/3 | same stages | +0.446 | 0.0020 | n/a | n/a | 0.449 |
| rr_anchor | inf | none / transformer | no | Y labels only | 0.017 (0.006, 0.005, 0.040) | 3/3 | n/a | n/a | 0.0012 | n/a | n/a | 0.308 |
| rr_aware_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 1.90, 1.72, 0.84 | 0.308, 0.308, 0.308 | 0.308 |
| rr_aware_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.305 | 0.0012 | 3.32, 3.11, 3.17 | 0.308, 0.308, 0.308 | 0.308 |
| rr_aware_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 4.97, 1.70, 3.83 | 0.308, 0.308, 0.308 | 0.308 |
| rr_aware_keys1 | 2 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 1.32, 1.18, 0.24 | 0.308, 0.309, 0.308 | 0.308 |
| rr_aware_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 1.98, 2.64, 1.01 | 0.308, 0.308, 0.308 | 0.308 |
| rr_aware_keys1 | 6 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 4.28, 3.22, 3.03 | 0.308, 0.309, 0.308 | 0.308 |
| rr_bias_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only; biased-bit curriculum | 0.249 (0.249, 0.249, 0.249) | 3/3 | same stages | +0.244 | 0.0012 | 1.89, 1.88, 1.89 | 0.249, 0.249, 0.249 | 0.308 |
| rr_bias_keys0 | inf | float / transformer (in_norm) | no | Y labels only; biased-bit curriculum | 0.002 (0.003, 0.002, 0.002) | 3/3 | same stages | -0.003 | 0.0012 | n/a | n/a | 0.308 |
| rr_bias_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.078 (0.066, 0.109, 0.060) | 3/3 | same stages | +0.073 | 0.0012 | 1.77, 1.54, 1.81 | 0.064, 0.109, 0.056 | 0.308 |
| rr_bias_keys1 | inf | float / transformer (in_norm) | yes | Y labels only; biased-bit curriculum; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.003 (0.002, 0.003, 0.004) | 3/3 | same stages | -0.002 | 0.0012 | n/a | n/a | 0.308 |
| rr_curr_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only; structure-informed (k curriculum) | 0.275 (0.307, 0.257, 0.261) | 2/3 | same stages | +0.267 | 0.0012 | 0.52, 1.89, 1.65 | 0.307, 0.255, 0.261 | 0.308 |
| rr_curr_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only; structure-informed (k curriculum) | 0.266 (0.308, 0.233, 0.257) | 2/3 | same stages | +0.257 | 0.0012 | 0.14, 2.63, 1.53 | 0.308, 0.230, 0.257 | 0.308 |
| rr_curr_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only; structure-informed (k curriculum) | 0.239 (0.234, 0.238, 0.245) | 3/3 | same stages | +0.231 | 0.0012 | 3.12, 2.98, 2.71 | 0.233, 0.238, 0.245 | 0.308 |
| rr_curr_keys1 | 2 | gauss / transformer (in_norm) | yes | Y labels only; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.169 (0.202, 0.222, 0.082) | 3/3 | same stages | +0.160 | 0.0012 | 0.98, 0.96, 1.69 | 0.202, 0.222, 0.078 | 0.308 |
| rr_curr_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.049 (0.024, 0.055, 0.068) | 3/3 | same stages | +0.041 | 0.0012 | 2.21, 2.05, 1.62 | 0.017, 0.047, 0.068 | 0.308 |
| rr_curr_keys1 | 6 | gauss / transformer (in_norm) | yes | Y labels only; structure-informed (k curriculum); STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.037 (0.045, 0.037, 0.030) | 3/3 | same stages | +0.029 | 0.0012 | 2.46, 2.97, 2.18 | 0.037, 0.030, 0.025 | 0.308 |
| rr_random_enc_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.309, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 1.22, 1.30, 0.58 | 0.308, 0.308, 0.308 | 0.308 |
| rr_random_enc_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.305 | 0.0012 | 2.29, 3.12, 2.92 | 0.308, 0.308, 0.308 | 0.308 |
| rr_random_enc_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 4.27, 2.88, 3.76 | 0.308, 0.308, 0.308 | 0.308 |
| rr_random_enc_keys1 | 2 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 1.22, 1.30, 0.58 | 0.308, 0.309, 0.308 | 0.308 |
| rr_random_enc_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 2.29, 3.12, 2.92 | 0.308, 0.309, 0.308 | 0.308 |
| rr_random_enc_keys1 | 6 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.309, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 4.27, 2.88, 3.76 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys0 | 2 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 0.08, 1.84, 0.00 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys0 | 4 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 2.39, 3.11, 2.09 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys0 | 6 | gauss / transformer (in_norm) | no | Y labels only | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 2.69, 2.87, 4.05 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys1 | 2 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 0.08, 1.84, 0.55 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys1 | 4 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.308, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 2.31, 1.78, 1.01 | 0.308, 0.308, 0.308 | 0.308 |
| rr_self_keys1 | 6 | gauss / transformer (in_norm) | yes | Y labels only; STRUCTURE-KNOWN (decoder reads every agent's key outside b) | 0.308 (0.308, 0.309, 0.308) | 0/3 | same stages | +0.306 | 0.0012 | 4.26, 2.87, 2.99 | 0.308, 0.309, 0.308 | 0.308 |
<!-- PARITY-LEARN:END -->

What each step said in the first pass (parity_learn.json; SUPERSEDED where batch 3 above differs: the keys arms there
had the unit-scale key bug, the curriculum is structure-informed, and the fixed-scheme anchor is 0.002):
- **Step 1 (decoder-side keys, no curriculum): keys alone do not help; they hurt.** With each agent's own model fed
  to the decoder outside the channel, the label-only arm stays at the prior at b = 2, 4 and even with an unbounded
  float message (9 of 9 runs at 0.449). Without keys the same fresh-episode setup escapes in 1 of 3 seeds at b = 2
  (0.358), 0 of 3 at b = 4, and with a float message reaches 0.047-0.131 (the 0.13 runs are at the count-leak level).
  So the gap was not only B1: encoder and channel learning is the problem (the review's branch "stuck: go to 2").
  Why keys make the plateau worse is not established; a guess is that 14 uninformative key features swamp a 2- to
  4-dimensional message at the decoder input.
- **B4 (anchor):** on fresh episodes with a best-val checkpoint the uncompressed transformer reaches 0.0036 on
  Z9-p2 (3 of 3 seeds); the review expected about 0.002. Train and validation KL are logged in every curve.
- **Step 3 (repairs + fixed-identity curriculum, with decoder keys):** the label-only curriculum now escapes at every
  seed: b = 2 0.183 (0.077-0.258), b = 4 0.095 (0.041-0.187). Seed 2 at b = 2 (0.077) is below every blind 2-bit
  reference (0.159), as decoder keys allow. Latent-label supervision (aux at b >= 4 only, B12) adds to it: b = 4
  0.036 (0.025-0.044). So intermediate (prefix-parity) labels help, but a Y-only curriculum without them already
  escapes, which the earlier role-flipping curriculum never did label-only. B6 (all-stage normalization, constant
  feature dropped), B7 (fixed identities), B10 (evaluation draws no global RNG; also fixed in compression v2) are in;
  B8 (decoder fine-tuned 1000 steps on the sign messages it is scored on) changes little: b = 2 0.183 -> 0.177,
  curr+aux 0.036 -> 0.031; at b = 4 without curriculum it moves the stuck decoder-key runs to 0.38-0.44.
- **Step 4 (count leak):** on the fixed Z9-p2, uncompressed DeepSets reaches 0.136 and ISAB 0.090 (0.063-0.126):
  both stall above the anchor, DeepSets just under the count-code level. With readers per bit randomised per episode
  (`zoo.KParityConfig.readers_random`; prior 0.308; the count code is worth less there, 0.256 vs 0.162 blind) ISAB
  reaches 0.002 in 3 of 3 seeds, label-only dvib with a float message 0.003 in 2 of 3 (one seed stays at the prior),
  the transformer anchor 0.005-0.040, while DeepSets learns little (0.24-0.31). So the leak was a trap: the arms
  that plateaued at the count level on the fixed scheme solve the leak-reduced one; mean pooling (DeepSets) is the
  exception.
- **Step 5 (biased-bit curriculum, P(bit = 1) = 0.8 -> 0.65 -> 0.5, `KParityConfig.bit_prob`):** at b = 4 (decoder
  keys) it escapes in 3 of 3 seeds to 0.065; with a float message (no decoder keys) 1 of 3 seeds reaches 0.006 and 2
  stay at the count level, 0.130. Confound: the b = 4 arm had decoder keys and the float arm did not (a design slip
  in `parity_learn.experiments`), so "escape at b = 4 but not at inf" cannot be read as the review's "compression is
  not the limit". What holds: low-order signal from biased bits gets Y-only training out of the prior at b = 4 in
  every seed, which plain Y-only training never did (step 1).

Overall: the best label-only compressed arms on Z9-p2 are now 0.077-0.258 at b = 2 and 0.041-0.187 at b = 4, against
an uncompressed anchor of 0.0036 (gap +0.07 to +0.25), with decoder keys and a curriculum. The large gaps of the
pool-trained v2 arms (+0.35 at b = 2) were partly optimization traps (the count leak, a role-flipping curriculum) and
partly the missing curriculum. Not done: step 2 and 6 (gemini), the float arm with decoder keys under the biased
curriculum, more seeds where seed spread is large.

## ForecastWorld (added 2026-10-22, `aggbench/forecast_world.py`)

Stage 2 of the Deger plan without LLMs: a latent theta in R^50 drives 20 binary questions (probit); 40 news-like
sources with Zipf popularity (exponent alpha = the overlap knob) give noisy linear readings; each forecaster reads k
sources and outputs its exact local forecasts on all 20 questions plus a key (sum of its sources' 16-d embeddings
plus N(0, sigma^2) noise). Two exact ceilings: (A) Bayes given every source read by anyone, and (B) Bayes given the
forecasts plus the exact source sets (probit inversion, then linear-Gaussian GLS). Tests:
`tests/aggbench/test_forecast_world.py`; runner `scripts/forecast_world_run.py`; results and generated tables:
`results/explore/forecast-world-20261022/README.md`.
