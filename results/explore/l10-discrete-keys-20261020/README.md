Serves: forecasters that say what they know, in a form an aggregator (and a human) can read

# L10: forecasters with short discrete keys (EXPLORATORY)

**EXPLORATORY** (PROTOCOL 10.1, stage 1, lane L10). Dev split only; the test split was never sampled. No language models. This is not a finding.

**One-line result.** Short discrete keys help only when trained with deterministic straight-through bits. Then regret to Bayes falls steadily with key length (E2: 0.043 with no key, 0.040 at 2 bits, 0.033 at 16 bits; continuous R^16: 0.028), and in E3-collude 16 bits get within CI of `oracle_clearing`. Gumbel-sigmoid keys gave no gain at any length up to 24 bits. No learned key, discrete or continuous, says which evidence pieces the agent saw: at most 0.8 of about 11 bits.

## Question

Each agent is a small trained network that sees only its own observation (`batch.agent_inputs("full")`: its piece values, its coverage bits and the strengths of its pieces). It outputs a forecast and a short key: a description of what it knows, learned end to end with an aggregator. How does regret (to Bayes and to `oracle_clearing`) fall with key length and vocabulary? How do discrete keys compare with continuous keys, ground-truth provenance keys and no key? And do learned discrete keys line up with the true evidence pieces?

## Arms

Every arm uses the same pipeline: agent net (MLP 48-128-128, shared weights, applied to each agent separately) gives forecast f_i and key k_i; the aggregator (2 self-attention blocks + attention pooling) sees only [f_i, k_i]. The forecast is trained only as a forecaster (BCE of the agent's own log-odds against Y), and the aggregator's gradient is stopped at it. As a result f_i matches the agent's exact local posterior (mean KL 0.0003-0.0013 nats; `extra.arms.*.own_forecast_kl_to_local_posterior`), and the key is the only learned channel. Training uses train-split truth labels (no distillation), 200k train episodes, 4000 steps, batch 512, AdamW one-cycle. The checkpoint is chosen on a held-out slice of train, in eval mode (hard keys). Evaluation is on 5000 dev episodes.

| arm | key |
|---|---|
| `none` | no key |
| `bits<L>` | L bits, Gumbel-sigmoid straight-through (tau 1.0 to 0.25), hard threshold at eval |
| `bag<T>x16` | bag (token counts) of T tokens from a 16-token vocabulary, Gumbel-softmax straight-through |
| `stbits<L>` (follow-up) | L bits, deterministic straight-through: forward hard sign, backward sigmoid gradient, no noise |
| `wbits<L>` (follow-up) | L bits, warm start: soft sigmoid bits for the first half of training, then deterministic straight-through; checkpoint only from the hard phase |
| `cont<m>` | real vector in R^m (continuous reference) |
| `prov_bits` | ground-truth provenance: the agent's 12 coverage bits (not learned) |
| `prov_full` | ground truth with strengths: env key `full` (coverage bits and coverage x mu), everything `oracle_clearing` uses about the agent |
| `<arm>@12000` (follow-up) | the same arm trained 3x longer |

Main sweep: 21 arms x 2 seeds x {E2, E3-collude}. Follow-up (diagnosing why Gumbel keys did nothing): 14 arms x seed 0. E3-biased is excluded (lane scope). A second follow-up (ST keys at 24/32/64 bits, ST bags, a second seed for ST bits; `vm_followup2.sh`) never ran because its VM hit the 25-minute startup timeout, and the L4 budget was used up. So every `stbits`/`wbits`/`@12000` number below is a single seed.

## Results

Regret is mean (log loss of the arm) minus (log loss of the exact global Bayes posterior), in nats per episode, over 5000 dev episodes, with 95% bootstrap CIs (`aggbench.metrics`). "minus" columns are paired differences on the same episodes. Full table, every arm and both seeds: `tables.md`; all numbers: `result_E2.json`, `result_E3-collude.json`. The tables below come from those JSONs via `summary_tables.py`.

#### E2: regret to Bayes (nats/episode, seed 0, 95% CI) by key size

| key size L | bits, Gumbel-sigmoid ST | bits, deterministic ST | bits, warm start | continuous R^L |
|---|---|---|---|---|
| 0 (no key) | 0.043 [0.035, 0.051] | - | - | - |
| 1 | 0.043 [0.035, 0.051] | - | - | 0.032 [0.025, 0.039] |
| 2 | 0.042 [0.034, 0.050] | 0.040 [0.032, 0.048] | 0.038 [0.030, 0.045] | 0.032 [0.026, 0.039] |
| 3 | 0.042 [0.034, 0.050] | - | - | - |
| 4 | 0.043 [0.035, 0.051] | 0.039 [0.031, 0.047] | 0.036 [0.029, 0.043] | 0.029 [0.022, 0.035] |
| 6 | 0.042 [0.035, 0.050] | - | - | - |
| 8 | 0.042 [0.034, 0.050] | 0.037 [0.030, 0.044] | 0.035 [0.028, 0.042] | 0.031 [0.024, 0.037] |
| 12 | 0.043 [0.034, 0.050] | - | - | - |
| 16 | 0.042 [0.034, 0.049] | 0.033 [0.026, 0.039] | 0.033 [0.026, 0.040] | 0.028 [0.021, 0.034] |
| 24 | 0.042 [0.034, 0.050] | - | - | - |

#### E2: key arms against the no-key arm and oracle_clearing (paired, same dev episodes)

| arm | regret to Bayes | minus no-key arm | minus oracle_clearing |
|---|---|---|---|
| oracle_clearing (baseline, no learning) | 0.018 [0.013, 0.023] | - | - |
| extremized_log_pool (baseline, no learning) | 0.045 [0.037, 0.053] | - | - |
| none | 0.043 [0.035, 0.051] | - | +0.0251 [+0.0191, +0.0311] |
| bits16 | 0.042 [0.034, 0.049] | -0.0016 [-0.0028, -0.0004] | +0.0235 [+0.0173, +0.0296] |
| stbits16 | 0.033 [0.026, 0.039] | -0.0107 [-0.0148, -0.0070] | +0.0144 [+0.0088, +0.0199] |
| wbits8@12000 | 0.031 [0.024, 0.038] | -0.0123 [-0.0175, -0.0068] | +0.0128 [+0.0070, +0.0183] |
| cont4 | 0.029 [0.022, 0.035] | -0.0144 [-0.0189, -0.0099] | +0.0107 [+0.0054, +0.0157] |
| cont16 | 0.028 [0.021, 0.034] | -0.0154 [-0.0204, -0.0104] | +0.0096 [+0.0041, +0.0146] |
| cont4@12000 | 0.026 [0.020, 0.032] | -0.0170 [-0.0225, -0.0116] | +0.0081 [+0.0025, +0.0136] |
| prov_bits | 0.040 [0.033, 0.047] | -0.0033 [-0.0079, +0.0014] | +0.0217 [+0.0168, +0.0269] |
| prov_full | 0.036 [0.029, 0.042] | -0.0072 [-0.0118, -0.0025] | +0.0179 [+0.0134, +0.0223] |
| prov_full@12000 | 0.033 [0.025, 0.040] | -0.0104 [-0.0153, -0.0057] | +0.0147 [+0.0098, +0.0195] |

#### E2: what the key says (bits; 'x / y' = information / entropy of the target)

| arm | key entropy | I(key; which pieces seen), probe, of H | I(key; # pieces) | I(key; sign of own log-odds) | I(key; size of own log-odds, 8 bins) | I(key; Y) |
|---|---|---|---|---|---|---|
| bits16 | 3.93 | 0.15 / 11.04 | 0.17 / 2.32 | 0.96 / 1.00 | 1.38 / 3.00 | 0.30 / 1.00 |
| bag4x16 | 3.71 | 0.16 / 11.04 | 0.22 / 2.32 | 0.96 / 1.00 | 1.17 / 3.00 | 0.28 / 1.00 |
| stbits4 | 3.48 | 0.23 / 11.04 | 0.18 / 2.32 | 0.45 / 1.00 | 0.66 / 3.00 | 0.13 / 1.00 |
| stbits16 | 6.13 | 0.15 / 11.04 | 0.24 / 2.32 | 0.91 / 1.00 | 1.94 / 3.00 | 0.31 / 1.00 |
| wbits8 | 5.02 | 0.25 / 11.04 | 0.26 / 2.32 | 0.60 / 1.00 | 1.30 / 3.00 | 0.20 / 1.00 |
| wbits8@12000 | 6.20 | 0.69 / 11.04 | 0.22 / 2.32 | 0.47 / 1.00 | 0.67 / 3.00 | 0.17 / 1.00 |
| cont16 | - | 5.31 / 11.04 | - | - | - | - |
| prov_bits | 10.71 | 11.03 / 11.04 | 1.99 / 2.32 | 0.00 / 1.00 | 0.07 / 3.00 | 0.00 / 1.00 |

#### E3-collude: regret to Bayes (nats/episode, seed 0, 95% CI) by key size

| key size L | bits, Gumbel-sigmoid ST | bits, deterministic ST | bits, warm start | continuous R^L |
|---|---|---|---|---|
| 0 (no key) | 0.028 [0.022, 0.035] | - | - | - |
| 1 | 0.030 [0.023, 0.036] | - | - | 0.021 [0.016, 0.027] |
| 2 | 0.029 [0.023, 0.035] | 0.028 [0.022, 0.034] | 0.024 [0.018, 0.030] | 0.019 [0.013, 0.024] |
| 3 | 0.028 [0.022, 0.035] | - | - | - |
| 4 | 0.028 [0.022, 0.035] | 0.026 [0.020, 0.032] | 0.025 [0.019, 0.031] | 0.018 [0.013, 0.024] |
| 6 | 0.028 [0.022, 0.035] | - | - | - |
| 8 | 0.028 [0.022, 0.035] | 0.026 [0.020, 0.033] | 0.022 [0.016, 0.028] | 0.019 [0.014, 0.025] |
| 12 | 0.028 [0.022, 0.035] | - | - | - |
| 16 | 0.030 [0.024, 0.037] | 0.022 [0.016, 0.028] | 0.024 [0.019, 0.030] | 0.020 [0.015, 0.026] |
| 24 | 0.029 [0.023, 0.036] | - | - | - |

#### E3-collude: key arms against the no-key arm and oracle_clearing (paired, same dev episodes)

| arm | regret to Bayes | minus no-key arm | minus oracle_clearing |
|---|---|---|---|
| oracle_clearing (baseline, no learning) | 0.018 [0.014, 0.023] | - | - |
| extremized_log_pool (baseline, no learning) | 0.031 [0.025, 0.038] | - | - |
| none | 0.028 [0.022, 0.035] | - | +0.0098 [+0.0056, +0.0142] |
| bits16 | 0.030 [0.024, 0.037] | +0.0016 [+0.0001, +0.0033] | +0.0115 [+0.0071, +0.0161] |
| stbits16 | 0.022 [0.016, 0.028] | -0.0062 [-0.0096, -0.0028] | +0.0037 [-0.0007, +0.0080] |
| wbits8@12000 | 0.023 [0.017, 0.029] | -0.0056 [-0.0095, -0.0018] | +0.0042 [-0.0003, +0.0087] |
| cont4 | 0.018 [0.013, 0.024] | -0.0098 [-0.0136, -0.0061] | +0.0000 [-0.0042, +0.0043] |
| cont16 | 0.020 [0.015, 0.026] | -0.0082 [-0.0121, -0.0043] | +0.0016 [-0.0029, +0.0062] |
| cont4@12000 | 0.023 [0.017, 0.029] | -0.0053 [-0.0094, -0.0014] | +0.0045 [+0.0001, +0.0088] |
| prov_bits | 0.029 [0.023, 0.035] | +0.0009 [-0.0032, +0.0048] | +0.0107 [+0.0067, +0.0148] |
| prov_full | 0.026 [0.020, 0.032] | -0.0023 [-0.0061, +0.0015] | +0.0076 [+0.0043, +0.0108] |
| prov_full@12000 | 0.028 [0.022, 0.034] | -0.0004 [-0.0042, +0.0036] | +0.0095 [+0.0057, +0.0134] |

#### E3-collude: what the key says (bits; 'x / y' = information / entropy of the target)

| arm | key entropy | I(key; which pieces seen), probe, of H | I(key; # pieces) | I(key; sign of own log-odds) | I(key; size of own log-odds, 8 bins) | I(key; Y) |
|---|---|---|---|---|---|---|
| bits16 | 3.87 | 0.26 / 11.88 | 0.15 / 3.26 | 0.96 / 1.00 | 1.51 / 3.00 | 0.43 / 1.00 |
| bag4x16 | 4.15 | 0.79 / 11.88 | 0.50 / 3.26 | 0.87 / 1.00 | 1.29 / 3.00 | 0.42 / 1.00 |
| stbits4 | 3.01 | 0.59 / 11.88 | 0.30 / 3.26 | 0.37 / 1.00 | 1.09 / 3.00 | 0.09 / 1.00 |
| stbits16 | 5.94 | 0.61 / 11.88 | 0.42 / 3.26 | 0.94 / 1.00 | 1.94 / 3.00 | 0.45 / 1.00 |
| wbits8 | 4.77 | 0.58 / 11.88 | 0.34 / 3.26 | 0.84 / 1.00 | 1.43 / 3.00 | 0.43 / 1.00 |
| wbits8@12000 | 5.06 | 0.73 / 11.88 | 0.41 / 3.26 | 0.58 / 1.00 | 1.18 / 3.00 | 0.28 / 1.00 |
| cont16 | - | 4.10 / 11.88 | - | - | - | - |
| prov_bits | 11.14 | 11.88 / 11.88 | 2.38 / 3.26 | 0.12 / 1.00 | 0.56 / 3.00 | 0.12 / 1.00 |


### Reading the tables

- **Gumbel-sigmoid and Gumbel-softmax keys do nothing.** Every `bits<L>` and `bag<T>x16` arm sits within about 0.002 of the no-key arm in both environments, from 1 to 24 bits and up to 18.9 bits of bag capacity. Seed 1 agrees within 0.0015. Training 3x longer (`bits8@12000`) recovers part of the gain: E2 -0.0054 [-0.0086, -0.0023] against no key.
- **Deterministic or warm-start straight-through keys work, and regret falls with length.** E2, deterministic ST: 0.040 / 0.039 / 0.037 / 0.033 at 2 / 4 / 8 / 16 bits, against 0.043 with no key. 16 bits beat no key by -0.0107 [-0.0148, -0.0070] and recover about 70% of the gain of a 16-dim continuous key. In E3-collude, `stbits16` is 0.022, within CI of `oracle_clearing` (+0.0037 [-0.0007, +0.0080]). The noise in the Gumbel relaxation, not discreteness itself, is what blocked learning in the main sweep.
- **Continuous keys are best.** E2 `cont4@12000`: 0.026, still above `oracle_clearing` (+0.008 [+0.003, +0.014]). In E3-collude, `cont2`-`cont8` match `oracle_clearing` (+0.000 [-0.004, +0.004] for `cont4`). No arm beats `oracle_clearing` in either environment, so these keys never got past the information loss of a scalar report. That needs a richer value channel (lane L2), not a richer key.
- **Ground-truth provenance helps little with this aggregator.** E2: `prov_bits` -0.003 [-0.008, +0.001] against no key; `prov_full` -0.007, or -0.010 at 12k steps. E3-collude: neither helps (`prov_bits@12000` is slightly worse, +0.004 [+0.000, +0.008]). The keys that help most carry almost no provenance (next section). With true coverage and strengths in hand, the aggregator still does not learn the clearing computation (exact pseudo-inverse algebra over overlapping coverage) in 4000-12000 steps.

### Interpretability: do discrete keys align with the true evidence pieces?

They do not. Two measures, on dev agents of seed 0 (25000 agent-episodes):

1. A held-out MLP probe from the key to the agent's 12 seen-bits. Trained on the first half of dev episodes, scored on the second half, it gives a lower bound on sum_k I(key; seen_k). Learned discrete keys carry 0.0-0.8 bits about which pieces were seen, out of 11.0 (E2) or 11.9 (E3-collude). The ground-truth `prov_bits` key scores 11.03 / 11.04, which validates the probe; shuffled keys score about 0. Continuous keys carry more (`cont16`: 5.3 bits in E2), but spread over all dimensions.
2. Per-unit plug-in MI between each key bit (or vocabulary token, or continuous dimension) and each piece's seen-bit. Alignment is the best unit's MI divided by H(seen), averaged over pieces. It is at most 0.04 for every learned discrete key and at most 0.06 for continuous keys (1.00 for `prov_bits`). No learned unit is dedicated to a piece (`pieces_with_a_dedicated_unit` = 0 for every learned arm). The full unit x piece matrices are in the JSON (`extra.arms.<arm>.interpretability.unit_mi_seen_matrix_bits`).

What the learned discrete keys do encode (plug-in MI minus a shuffled-key estimate): mostly the agent's own forecast. That is up to 0.96 of the 1 bit of its sign (0.37-0.96 across arms), up to 1.9 of the 3 bits of its size, and 0.09-0.45 bits about Y. They carry only 0.15-0.5 bits about how many pieces the agent saw. The forecast is already sent exactly, so the useful part of a key must be what it carries beyond the forecast; this run did not isolate that part. In short, the learned keys are a coarse second copy of the forecast plus some evidence detail. They do not describe "which evidence I hold", so a human cannot read them as provenance. Nothing in the training objective asks them to; that would need a key that is forced to be about provenance (e.g. an auxiliary loss or a fixed vocabulary grounded in pieces).

## Checks

- **Controls** (`aggbench.controls.run_with_controls`, the arm's own `predict`): every random-network control has regret 0.42-0.44 and every untrained-init control 0.42-0.46, against 0.02-0.04 for trained arms. No claim here is passed by a control.
- **Locality**: `aggbench.messages.check_locality` passed for every trained agent net (500 dev episodes, 5 perturbation trials, atol 1e-9; `extra.arms.*.locality`). The message is [forecast, key] and depends only on the agent's own obs and key.
- **Oracle check**: brute-force Bayes check on 24 smoke episodes per environment, max abs diff 2.2e-16 (`oracle_check`).
- **Harness**: JSON written by `aggbench.results.write_result` (schema validated), `harness.git_dirty` false. The harness commit recorded at finalize (`e5113ba9`) includes `a7023f6`, a change made by another lane while training ran: `controls.random_weights` now handles GPU models. Training used the earlier `controls.py`. On the VM, the runner moves the trained module to CPU before the control, so both versions produce the same random-network arm.
- **Runner**: `run_l10.py`. Main sweep trained at commit 19fc996, follow-up at c3632a0; the `runner_sha256` in the JSON is the finalize version (which adds only analysis code). `extremized_log_pool`'s alpha was fitted on 20000 train episodes.

## Compute

Spot L4 (g2-standard-8) via `scripts/infra/gpu-run.sh --run-id` (the `run_on_l4.sh` wrapper has no `--run-id` and refuses while other lanes' VMs exist). There were five VM rentals totalling about 1.9 L4-hours, at most about $1.65 at the $0.87/h list ceiling. Two hit the 25-minute startup timeout and ran nothing; one stopped on a device bug in the first arm (fixed in 19fc996); the main sweep took 0.46 h and the follow-up 0.34 h. GPU-LOG lines are in `logs/`. Finalize (baselines, metrics, probes) ran on CPU on General.

## Files

`run_l10.py` (runner: `data`, `train`, `finalize`), `vm_run.sh`, `vm_followup.sh`, `vm_followup2.sh` (not run), `summary_tables.py`, `result_E2.json`, `result_E3-collude.json`, `tables.md`, `arms/` (per-arm JSON: locality, fit history, interpretability; NPZ: dev logits of trained and control arms, own forecasts, seed-0 dev keys), `logs/`.
