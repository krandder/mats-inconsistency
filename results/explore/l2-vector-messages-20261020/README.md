Serves: richer messages let aggregation recover what scalar forecasts lose, toward full Bayes

# L2: vector messages from local encoders (EXPLORATORY)

**EXPLORATORY** (PROTOCOL 10.1). These are development data, not findings. Dev split only; the test split was never sampled. One training seed per cell.

**Question.** Each agent runs a local encoder on its own observation and key only (`batch.agent_inputs("full")`) and sends a vector message: a key and a value, each in R^m. An attention aggregator sees only these messages. Can it beat `oracle_clearing` (the best possible from scalar reports plus the true structure) and approach full Bayes in E2, while matching Bayes in E1? How does regret fall as m grows?

**Arms.**
- `vecmsg_m{m}`: one encoder with weights shared across agents and applied to each agent separately (MLP 128-128 producing [key in R^m, value in R^m]), then an attention aggregator (embed, 2 pre-LN self-attention blocks with 4 heads at width 64, attention pooling with 4 learned seeds, then an MLP). Output: E2/E3 log-odds = prior + net; E1 mean plus precision I + L L^T. Trained end to end on the **truth labels** of 200k train episodes: 2500 AdamW steps, batch 512, lr 2e-3 one-cycle, checkpoint picked on 10k held-out train episodes. m in {1,2,4,8,16} for E2; {1,4,16} for E1 and E3 (to fit the CPU budget).
- Encoder inputs (local preprocessing, declared). E2/E3: own piece values, a seen mask, the own key (coverage bits and coverage x mu) and own piece LLRs mu*x. E1: own s_i and key, plus the agent's own local posterior in information form (A^T R^-1 s, A^T R^-1 A). A first E1 attempt without the information-form features got m=1 regret 3.13; it was stopped during m=2 and its log is kept in `log_E1_rawfeatures_killed.txt`.
- E3-biased: the distortion is applied on the agent side. The biased agent's piece LLRs become a*lam + b/n_i, so its own local log-odds equal the environment's distorted report exactly (checked: max error 4e-15). Its key stays honest. The aggregator never sees roles, a or b.
- Scalar-report baselines from `aggbench.baselines` (`oracle_clearing`, `extremized_log_pool`, `log_pool`, `sum_logodds`) on the same dev episodes. `linear_pool` is also in the JSON.
- Controls: `run_with_controls` runs the untrained-init and random-network controls through the same encoder→aggregator `predict`.
- Diagnostic (E2 only, **declared distillation**, README rule 2): the same pipeline trained on soft targets sigmoid(bayes) instead of 0/1 truth labels (`run_l2_distill.py`). It separates "bandwidth or architecture cannot carry Bayes" from "learning from noisy labels in this budget is the limit". It is not an answer to the main question.

**Locality.** `aggbench.messages.check_locality` was run on every trained encoder (500 dev episodes, 5 trials, atol 1e-9). It passed for all 22 trained encoders (17 main, 5 distillation) (`extra.locality_check` in each JSON). Messages are computed from each agent's own inputs only, and the aggregator never sees obs.

**Measured.** Regret to `bayes` in nats per episode, with 95% bootstrap CIs over 5000 dev episodes, and the paired differences against `oracle_clearing` and `extremized_log_pool` (`metrics.paired_difference`). Every number below comes from `python3 tables.py`, which reads the `result_*.json` files and validates them with `aggbench.results.validate`. For the scalar rows, "m" means one scalar report.

## Main runs (truth labels)

| env | arm | m (key+value floats) | regret to bayes | regret minus oracle_clearing | minus extremized_log_pool | random-network control | untrained-init control | locality |
|---|---|---|---|---|---|---|---|---|
| E1 | oracle_clearing (scalar reports) | 1 | -0.000 [-0.000, 0.000] | | | | | |
| E1 | extremized_log_pool (scalar reports) | 1 | 0.818 [0.785, 0.851] | | | | | |
| E1 | log_pool (scalar reports) | 1 | 2.062 [2.025, 2.100] | | | | | |
| E1 | sum_logodds (scalar reports) | 1 | 0.818 [0.785, 0.851] | | | | | |
| E1 | vector messages | 1+1 | 2.770 [2.715, 2.824] | 2.770 [2.715, 2.824] | 1.952 [1.904, 2.001] | 4.64 [4.58, 4.71] | 4.96 [4.89, 5.02] | pass |
| E1 | vector messages | 4+4 | 1.544 [1.502, 1.590] | 1.544 [1.502, 1.590] | 0.727 [0.693, 0.760] | 5.86 [5.76, 5.96] | 4.61 [4.55, 4.68] | pass |
| E1 | vector messages | 16+16 | 1.212 [1.171, 1.255] | 1.212 [1.171, 1.255] | 0.394 [0.369, 0.421] | 4.67 [4.61, 4.74] | 4.54 [4.48, 4.61] | pass |
| E2 | oracle_clearing (scalar reports) | 1 | 0.018 [0.013, 0.023] | | | | | |
| E2 | extremized_log_pool (scalar reports) | 1 | 0.045 [0.037, 0.053] | | | | | |
| E2 | log_pool (scalar reports) | 1 | 0.109 [0.100, 0.118] | | | | | |
| E2 | sum_logodds (scalar reports) | 1 | 0.141 [0.117, 0.163] | | | | | |
| E2 | vector messages | 1+1 | 0.032 [0.026, 0.039] | 0.014 [0.009, 0.019] | -0.013 [-0.017, -0.008] | 0.43 [0.41, 0.44] | 0.43 [0.42, 0.45] | pass |
| E2 | vector messages | 2+2 | 0.033 [0.026, 0.040] | 0.015 [0.009, 0.020] | -0.012 [-0.017, -0.007] | 0.44 [0.43, 0.46] | 0.43 [0.42, 0.44] | pass |
| E2 | vector messages | 4+4 | 0.032 [0.025, 0.038] | 0.014 [0.008, 0.019] | -0.013 [-0.018, -0.008] | 0.43 [0.42, 0.45] | 0.43 [0.41, 0.44] | pass |
| E2 | vector messages | 8+8 | 0.031 [0.024, 0.037] | 0.013 [0.007, 0.018] | -0.014 [-0.019, -0.009] | 0.43 [0.42, 0.45] | 0.46 [0.45, 0.48] | pass |
| E2 | vector messages | 16+16 | 0.031 [0.025, 0.038] | 0.013 [0.007, 0.019] | -0.014 [-0.019, -0.008] | 0.43 [0.41, 0.44] | 0.43 [0.41, 0.44] | pass |
| E3-sybil | oracle_clearing (scalar reports) | 1 | 0.018 [0.013, 0.023] | | | | | |
| E3-sybil | extremized_log_pool (scalar reports) | 1 | 0.079 [0.069, 0.089] | | | | | |
| E3-sybil | log_pool (scalar reports) | 1 | 0.124 [0.115, 0.134] | | | | | |
| E3-sybil | sum_logodds (scalar reports) | 1 | 0.439 [0.389, 0.488] | | | | | |
| E3-sybil | vector messages | 1+1 | 0.034 [0.027, 0.041] | 0.016 [0.010, 0.022] | -0.045 [-0.053, -0.037] | 0.43 [0.41, 0.44] | 0.43 [0.42, 0.45] | pass |
| E3-sybil | vector messages | 4+4 | 0.034 [0.027, 0.041] | 0.016 [0.010, 0.021] | -0.045 [-0.053, -0.037] | 0.43 [0.42, 0.45] | 0.43 [0.41, 0.44] | pass |
| E3-sybil | vector messages | 16+16 | 0.029 [0.023, 0.035] | 0.011 [0.006, 0.017] | -0.050 [-0.059, -0.042] | 0.43 [0.41, 0.44] | 0.43 [0.41, 0.44] | pass |
| E3-biased | oracle_clearing (scalar reports) | 1 | 0.018 [0.013, 0.023] | | | | | |
| E3-biased | extremized_log_pool (scalar reports) | 1 | 0.077 [0.067, 0.087] | | | | | |
| E3-biased | log_pool (scalar reports) | 1 | 0.089 [0.080, 0.099] | | | | | |
| E3-biased | sum_logodds (scalar reports) | 1 | 0.362 [0.318, 0.406] | | | | | |
| E3-biased | vector messages | 1+1 | 0.049 [0.040, 0.057] | 0.031 [0.024, 0.037] | -0.028 [-0.035, -0.021] | 0.43 [0.41, 0.44] | 0.43 [0.42, 0.45] | pass |
| E3-biased | vector messages | 4+4 | 0.046 [0.037, 0.054] | 0.027 [0.021, 0.034] | -0.031 [-0.038, -0.024] | 0.44 [0.42, 0.45] | 0.43 [0.41, 0.44] | pass |
| E3-biased | vector messages | 16+16 | 0.043 [0.035, 0.051] | 0.025 [0.019, 0.031] | -0.034 [-0.041, -0.026] | 0.43 [0.41, 0.44] | 0.43 [0.41, 0.44] | pass |
| E3-collude | oracle_clearing (scalar reports) | 1 | 0.018 [0.014, 0.023] | | | | | |
| E3-collude | extremized_log_pool (scalar reports) | 1 | 0.031 [0.025, 0.038] | | | | | |
| E3-collude | log_pool (scalar reports) | 1 | 0.044 [0.038, 0.051] | | | | | |
| E3-collude | sum_logodds (scalar reports) | 1 | 0.285 [0.248, 0.323] | | | | | |
| E3-collude | vector messages | 1+1 | 0.020 [0.015, 0.026] | 0.002 [-0.002, 0.006] | -0.011 [-0.015, -0.007] | 0.43 [0.42, 0.44] | 0.43 [0.42, 0.45] | pass |
| E3-collude | vector messages | 4+4 | 0.020 [0.015, 0.025] | 0.002 [-0.003, 0.006] | -0.011 [-0.015, -0.007] | 0.43 [0.42, 0.45] | 0.43 [0.41, 0.44] | pass |
| E3-collude | vector messages | 16+16 | 0.019 [0.014, 0.025] | 0.001 [-0.003, 0.005] | -0.012 [-0.016, -0.008] | 0.42 [0.41, 0.44] | 0.43 [0.41, 0.44] | pass |

## Distillation diagnostic (E2, declared, soft targets sigmoid(bayes))

| m (key+value floats) | regret to bayes | regret minus oracle_clearing | minus extremized_log_pool | random-network control | untrained-init control | locality |
|---|---|---|---|---|---|---|
| 1+1 | 0.029 [0.022, 0.035] | 0.011 [0.006, 0.016] | -0.016 [-0.021, -0.011] | 0.43 [0.41, 0.44] | 0.43 [0.42, 0.45] | pass |
| 2+2 | 0.023 [0.017, 0.028] | 0.004 [-0.000, 0.009] | -0.022 [-0.028, -0.017] | 0.46 [0.45, 0.48] | 0.43 [0.42, 0.44] | pass |
| 4+4 | 0.022 [0.017, 0.028] | 0.004 [-0.001, 0.009] | -0.023 [-0.028, -0.017] | 0.44 [0.42, 0.45] | 0.43 [0.41, 0.44] | pass |
| 8+8 | 0.019 [0.014, 0.024] | 0.001 [-0.004, 0.006] | -0.026 [-0.033, -0.020] | 0.43 [0.42, 0.45] | 0.46 [0.45, 0.48] | pass |
| 16+16 | 0.014 [0.009, 0.019] | -0.004 [-0.009, 0.001] | -0.031 [-0.038, -0.023] | 0.43 [0.41, 0.44] | 0.43 [0.41, 0.44] | pass |

Provenance:

- E1: harness edfbe03cdd dirty=False, oracle_check max diff 2.9e-15, train n=200000, dev n=5000, runner sha256 9020f5727463
- E2: harness 992bc2bdd8 dirty=False, oracle_check max diff 2.2e-16, train n=200000, dev n=5000, runner sha256 9020f5727463
- E3-sybil: harness 78a4f11a02 dirty=False, oracle_check max diff 1.1e-16, train n=200000, dev n=5000, runner sha256 9020f5727463
- E3-biased: harness 6dc805e492 dirty=False, oracle_check max diff 2.2e-16, train n=200000, dev n=5000, runner sha256 9020f5727463
- E3-collude: harness edfbe03cdd dirty=False, oracle_check max diff 2.2e-16, train n=200000, dev n=5000, runner sha256 9020f5727463
- result_E2-distill-m1.json: harness 1d48a0ada4 dirty=False, runner sha256 d1cd897fdfbf
- result_E2-distill-m2.json: harness 1d48a0ada4 dirty=False, runner sha256 d1cd897fdfbf
- result_E2-distill-m4.json: harness 1d48a0ada4 dirty=False, runner sha256 d1cd897fdfbf
- result_E2-distill-m8.json: harness 1d48a0ada4 dirty=False, runner sha256 d1cd897fdfbf
- result_E2-distill.json: harness 20ca7bbd08 dirty=False, runner sha256 9e673e092ff7

## Reading

- **E2, truth labels.** Regret is flat in m: 0.032 at m=1 and 0.031 at m=16. The learned arm beats `extremized_log_pool` by about 0.013, but it stays **above** `oracle_clearing` by 0.013 [0.007, 0.019] even at m=16. With this training budget, bandwidth is not the limit; learning from 0/1 labels is.
- **E2, distillation.** Regret falls steadily with m: 0.029 (m=1), 0.023, 0.022, 0.019, 0.014 (m=16). At m=16 it is 0.004 below `oracle_clearing`, with CI [-0.009, +0.001]. So the point estimate beats the scalar ceiling, but the CI includes zero, and validation loss was still falling at step 2500. The messages and the architecture can carry more than a scalar does; the truth-label pipeline does not extract it in 2500 steps.
- **E3.**
  - E3-collude reaches `oracle_clearing` level: +0.001 [-0.003, 0.005] at m=16.
  - E3-sybil and E3-biased fall a little with m but stay above `oracle_clearing`: +0.011 and +0.025 at m=16.
  - In every E3 variant the learned arm beats `extremized_log_pool` by a wide margin (0.011–0.050).
- **E1.** It does not match Bayes. Regret falls with m (2.77 at m=1, 1.54 at m=4, 1.21 at m=16; `oracle_clearing` = bayes = 0), but it stays worse than the scalar `sum_logodds`/`extremized_log_pool` (0.818). Computing the correlated-noise posterior from scratch with 2500 truth-label steps is far from solved.
- **Controls.** The random-network and untrained-init controls sit at about prior-only regret in every cell (E2/E3 about 0.43, E1 about 4.5–5.9), so the gains come from training.

## Caveats

- **Harness changed mid-run.** E4 was added to aggbench by commits 99859d8..cc8cf8d at 22:47–22:55Z. The E2/E3 processes had imported the harness at 718627f (about 22:40Z). `harness_info()` hashes the files when it writes, so every JSON records harness sha256 e50881934b0f… (the later files). `git diff 718627f cc8cf8d -- aggbench` is additive (E4 only), and the E1–E3 code paths are unchanged. `harness.git_commit` also varies between JSONs because HEAD moved with unrelated commits.
- **Runner sha.** The E2/E3 main runs started from runner commit 71e0785. The JSONs record runner sha256 9020f572…, which is the file at fab082b. The only difference is the E1 branch of `local_features`, which bernoulli envs never run. E1 ran from fab082b.
- One seed and one budget (2500 steps), with no hyperparameter search. `result_E2-distill.json` is the m=16 distillation run, whose run_id has no `-m16` suffix.

## Result

EXPLORATORY: with truth-label training, vector messages beat every scalar pool except `oracle_clearing` but do not beat it in E2, and regret is flat in m. Under declared distillation, regret falls with m and m=16 reaches `oracle_clearing` level (−0.004 [−0.009, +0.001]). E1 is far from Bayes. Next step: longer training at m=16 (truth labels and distillation) to see whether the point estimate's gap below `oracle_clearing` becomes significant.

Files: `run_l2.py` (main runner), `run_l2_distill.py` (diagnostic), `tables.py` (tables above), `result_*.json` (validated aggbench.result/v1), `log_*.txt`.
