Serves: an aggregation mechanism that scales to many sources, as a general system must

# L8: scaling and transfer of learned aggregators (EXPLORATORY)

**Label: EXPLORATORY** (PROTOCOL 10.1). This is development data, not a finding. Dev split only. The test split was never touched.

**Question.** Train permutation-invariant aggregators on few agents and test them zero-shot on more agents, more evidence pieces and other overlap levels. Where does regret to Bayes and to `oracle_clearing` degrade? Does a looped or clearing-style aggregator transfer better than a single attention layer? The setup is aggbench E2 (binary Y, K evidence pieces, scalar log-odds reports), with keys at `full` and `noisy`.

**Training.** E2 with N = 4..8 agents (10,000 train episodes per N; the seed ranges for different N are disjoint), K = 12 pieces and 2-6 pieces per agent. Labels are `truth` (no distillation). Each model trains for 6000 AdamW steps at batch 256, with width 32 and 3 training seeds. Each visibility has its own models.

**Arms.** Each arm is invariant to agent and piece order and is defined for any N and K (code: `l8.py`).
- `attn1`: one attention layer over agents. An agent's token is its report plus a sum over its pieces of phi(key piece). The attention logits are a content term plus a per-head key overlap S_ij = sum_k f(key_ik)·g(key_jk). The unnormalised overlap mass sum_j S_ij is also an input. Readout: l0 + sum_i a_i (l_i - l0) + b.
- `attnloop`: the same layer looped 4 times with shared weights. `attnloop_T8` uses the same weights with 8 loops at test.
- `clearloop`: learned clearing. Piece estimates w_k start at 0. On each loop, each agent's residual (l_i - l0) - sum_k c_ik w_k is split over its claimed pieces by learned weights beta(key_ik) and averaged per piece (Cimmino style). It loops 16 times in training. `clearloop_T64` uses the same weights with 64 loops at test. With full keys it can represent `oracle_clearing`.
- References (from aggbench): `oracle_clearing`; `extremized_log_pool` with alpha fitted at the target config (a privileged reference); `extremized_log_pool@train_alpha` with alpha fitted at the training config (alpha = 2.26), which is a fair zero-shot baseline.
- Controls: every learned arm goes through `aggbench.controls.run_with_controls`, which adds an untrained-init arm and a scale-matched random-network arm. `fit()` is memoised, so each trained model is reused across test conditions. The lowest random-network regret in any condition is 0.33 nats or more. No claim below is matched by a control.

**Measured.** Mean regret to Bayes (nats per episode) over 2000 dev episodes for each of 13 test conditions, with 95% bootstrap CIs over episodes. Main arms are averaged over 3 seeds per episode. Each JSON also has paired differences against `extremized_log_pool` and `oracle_clearing`. Learning curves vary training episodes (1k/3k/10k/50k) and width (8 to 128; 228 to 201k parameters) and are evaluated at N8 and N64. One harness-validated JSON per (visibility, condition) is in `json/`, at harness commit da9a26b (clean). Each JSON has a brute-force oracle check on 24 episodes, with max difference 3e-16.

**One-line result (EXPLORATORY).** The clearing-style loop transfers almost perfectly. Trained on 4-8 agents, it stays within 0.006 nats of `oracle_clearing` at every tested N (up to 64), K (up to 48) and overlap with full keys. The single and looped attention layers match it roughly in distribution but break past about 16 agents or at high overlap: regret reaches 13 to 166 nats at N = 32-64 and over 1000 nats at N64/K48. Looping attention did not help transfer, and running more loops at test made it worse.

## Main observations
- **Number of agents.** With full keys, `clearloop` regret to Bayes is 0.012 at N8, 0.002 at N16 and 0.004 at N64. That is equal to `oracle_clearing` within its CI and better than extremized at train alpha (0.016 at N64). Both attention arms are already worse than extremized at N16 (`attnloop` 1.31) and diverge at N32 and N64. The cause is extrapolation: the overlap mass and the readout sum_i a_i r_i grow linearly in N beyond anything seen in training. Neither size nor data fixes this: every attention model in the learning curves has regret of at least 15 nats at N64.
- **Pieces and latent dimension (K).** At N = 8, every arm transfers to K = 24 and K = 48 at both fixed overlap fraction and lower overlap. `clearloop` is within 0.001 of `oracle_clearing` with full keys; `attnloop` degrades at K48 with 8-24 pieces per agent (0.064). E1's latent dimension d was not tested, because E1 keys are d-dependent matrices and no d-invariant arm was built.
- **Overlap.** At high overlap (10-12 of 12 pieces), `attn1` and `attnloop` fail (6.3 and 27.6 nats with full keys; 0.95 and 0.30 with noisy keys), while `clearloop` stays at 0.003-0.005 with full keys and 0.022 with noisy keys. At low overlap (1-2 pieces), all arms transfer.
- **Noisy keys.** In distribution, all three arms beat extremized by only 0.005-0.007 nats and stay 0.03 above `oracle_clearing`, which knows the true structure. `clearloop` still transfers in N, with regret to Bayes 0.005 at N64, but at high overlap it is 0.016 worse than extremized fitted at the target.
- **More loops at test.** `clearloop_T64` is equal to `clearloop` (T16) or slightly better, so its iteration has converged. `attnloop_T8` is worse than T4 everywhere, so the looped transformer did not learn a fixed point.
- **Learning curves.** `clearloop` is already near its plateau at 1k training episodes (N8 0.015 vs 0.012 at 50k with full keys). The attention arms need 10k or more episodes (attn1 at 1k: 6.0 nats; at 10k: 0.028). Their N64 transfer stays broken at every data size and width, so neither more data nor more parameters fixes it.
- **Seeds.** `clearloop` gives near-identical results across seeds (seed means differ by up to 0.002). The seed spread of the attention arms is in `summary.json`.

## Limitations
The attention arms are one design, with sum readout and unnormalised overlap mass. A design normalised by N might transfer better; this run does not test that. `clearloop` has the structure of the clearing equation built in. That it transfers shows the structure carries over, not that a generic learner discovers it. Only E2 is covered (no E3 adversaries, no E1). Results come from one run with 3 seeds and a single learning-curve seed, on the dev split.

## Tables

### Regret to Bayes, keys `full` (nats/episode, mean over 3 training seeds, 95% bootstrap CI over episodes)

| test condition | oracle_clearing | extremized (alpha fit at target) | extremized (train alpha) | attn1 | attnloop | attnloop T8 | clearloop | clearloop T64 | lowest random-net control (any arm) |
|---|---|---|---|---|---|---|---|---|---|
| N4 | 0.021 [0.014, 0.028] | 0.044 [0.033, 0.056] | 0.044 [0.033, 0.057] | 0.032 [0.023, 0.042] | 0.027 [0.019, 0.035] | 0.075 [0.060, 0.092] | 0.022 [0.015, 0.030] | 0.022 [0.015, 0.030] | 0.394 |
| N8 | 0.012 [0.006, 0.018] | 0.048 [0.034, 0.061] | 0.049 [0.035, 0.061] | 0.029 [0.020, 0.037] | 0.020 [0.012, 0.028] | 0.104 [0.082, 0.127] | 0.012 [0.006, 0.019] | 0.012 [0.006, 0.018] | 0.437 |
| N16 | 0.000 [0.000, 0.000] | 0.029 [0.019, 0.039] | 0.033 [0.023, 0.042] | 0.072 [0.058, 0.086] | 1.314 [1.179, 1.455] | 0.641 [0.482, 0.830] | 0.002 [-0.002, 0.005] | 0.001 [-0.001, 0.004] | 0.442 |
| N32 | 0.000 [-0.000, 0.000] | 0.019 [0.011, 0.026] | 0.024 [0.016, 0.032] | 13.392 [12.775, 14.014] | 40.079 [38.609, 41.533] | 24.296 [22.495, 26.194] | 0.003 [0.000, 0.006] | 0.002 [-0.000, 0.005] | 0.439 |
| N64 | 0.000 [-0.000, 0.000] | 0.008 [0.003, 0.014] | 0.016 [0.009, 0.022] | 115.716 [111.278, 119.879] | 135.878 [131.358, 140.416] | 166.022 [156.347, 175.807] | 0.004 [0.001, 0.007] | 0.004 [0.001, 0.006] | 0.435 |
| N8_K24_ppa4-12 | 0.020 [0.012, 0.028] | 0.033 [0.023, 0.043] | 0.033 [0.023, 0.043] | 0.027 [0.016, 0.037] | 0.025 [0.015, 0.035] | 0.045 [0.031, 0.059] | 0.020 [0.011, 0.029] | 0.021 [0.012, 0.029] | 0.652 |
| N8_K48_ppa8-24 | 0.004 [0.000, 0.007] | 0.008 [0.003, 0.013] | 0.008 [0.004, 0.013] | 0.006 [0.001, 0.011] | 0.064 [0.045, 0.085] | 0.040 [0.024, 0.057] | 0.004 [0.000, 0.008] | 0.004 [0.000, 0.008] | 1.013 |
| N8_K24_ppa2-6 | 0.020 [0.013, 0.027] | 0.043 [0.031, 0.055] | 0.074 [0.063, 0.085] | 0.029 [0.019, 0.039] | 0.028 [0.019, 0.037] | 0.088 [0.068, 0.110] | 0.022 [0.014, 0.029] | 0.021 [0.014, 0.029] | 0.582 |
| N8_K48_ppa2-6 | 0.010 [0.003, 0.017] | 0.027 [0.017, 0.038] | 0.080 [0.069, 0.090] | 0.013 [0.005, 0.021] | 0.012 [0.004, 0.019] | 0.061 [0.047, 0.075] | 0.011 [0.004, 0.019] | 0.011 [0.004, 0.019] | 0.664 |
| N8_ppa1-2 | 0.001 [-0.001, 0.003] | 0.047 [0.035, 0.060] | 0.086 [0.072, 0.100] | 0.023 [0.015, 0.031] | 0.012 [0.006, 0.019] | 0.070 [0.058, 0.082] | 0.004 [0.001, 0.007] | 0.004 [0.001, 0.007] | 0.326 |
| N8_ppa6-10 | 0.005 [0.002, 0.009] | 0.018 [0.011, 0.026] | 0.045 [0.030, 0.061] | 0.095 [0.078, 0.113] | 1.149 [1.020, 1.285] | 0.263 [0.211, 0.324] | 0.010 [0.005, 0.015] | 0.007 [0.003, 0.012] | 0.443 |
| N8_ppa10-12 | 0.000 [-0.000, 0.000] | 0.007 [0.004, 0.010] | 0.075 [0.053, 0.099] | 6.315 [5.987, 6.648] | 27.590 [26.452, 28.666] | 20.753 [19.699, 21.808] | 0.005 [0.002, 0.007] | 0.003 [0.000, 0.005] | 0.442 |
| N64_K48_ppa8-24 | -0.000 [-0.000, 0.000] | -0.001 [-0.003, 0.001] | 0.001 [-0.001, 0.003] | 1746.528 [1711.167, 1781.643] | 1692.902 [1655.533, 1729.638] | 1589.481 [1501.158, 1686.474] | 0.000 [-0.001, 0.001] | 0.000 [-0.001, 0.001] | 0.986 |

### Regret to oracle_clearing, keys `full` (paired, seed-averaged; < 0 means better than the reports-level ceiling)

| test condition | attn1 | attnloop | attnloop_T8 | clearloop | clearloop_T64 | clearloop regret to Bayes, min-max over seeds |
|---|---|---|---|---|---|---|
| N4 | 0.011 [0.006, 0.017] | 0.006 [0.002, 0.009] | 0.054 [0.041, 0.069] | 0.001 [-0.001, 0.003] | 0.001 [-0.001, 0.003] | 0.022-0.022 |
| N8 | 0.017 [0.009, 0.023] | 0.008 [0.003, 0.013] | 0.092 [0.071, 0.112] | 0.000 [-0.003, 0.003] | -0.000 [-0.002, 0.002] | 0.012-0.012 |
| N16 | 0.072 [0.058, 0.086] | 1.314 [1.179, 1.454] | 0.641 [0.482, 0.829] | 0.002 [-0.002, 0.005] | 0.001 [-0.001, 0.003] | 0.001-0.002 |
| N32 | 13.392 [12.775, 14.014] | 40.079 [38.609, 41.533] | 24.296 [22.495, 26.194] | 0.003 [0.000, 0.006] | 0.002 [-0.000, 0.005] | 0.002-0.004 |
| N64 | 115.716 [111.278, 119.879] | 135.878 [131.358, 140.416] | 166.022 [156.347, 175.807] | 0.004 [0.001, 0.007] | 0.004 [0.001, 0.006] | 0.003-0.006 |
| N8_K24_ppa4-12 | 0.007 [0.001, 0.013] | 0.005 [0.001, 0.009] | 0.025 [0.015, 0.035] | 0.000 [-0.002, 0.003] | 0.001 [-0.002, 0.003] | 0.020-0.020 |
| N8_K48_ppa8-24 | 0.002 [-0.001, 0.006] | 0.060 [0.042, 0.081] | 0.036 [0.021, 0.053] | 0.000 [-0.001, 0.001] | 0.000 [-0.001, 0.001] | 0.004-0.004 |
| N8_K24_ppa2-6 | 0.008 [0.002, 0.014] | 0.008 [0.003, 0.013] | 0.068 [0.050, 0.087] | 0.001 [-0.001, 0.004] | 0.001 [-0.001, 0.004] | 0.022-0.022 |
| N8_K48_ppa2-6 | 0.002 [-0.001, 0.006] | 0.001 [-0.001, 0.004] | 0.050 [0.038, 0.063] | 0.001 [-0.001, 0.003] | 0.001 [-0.001, 0.003] | 0.011-0.012 |
| N8_ppa1-2 | 0.022 [0.014, 0.029] | 0.011 [0.006, 0.017] | 0.069 [0.058, 0.081] | 0.003 [0.000, 0.006] | 0.003 [0.000, 0.005] | 0.004-0.004 |
| N8_ppa6-10 | 0.090 [0.073, 0.108] | 1.143 [1.014, 1.281] | 0.258 [0.207, 0.318] | 0.005 [0.002, 0.008] | 0.002 [-0.000, 0.004] | 0.010-0.010 |
| N8_ppa10-12 | 6.315 [5.987, 6.647] | 27.590 [26.452, 28.666] | 20.753 [19.699, 21.807] | 0.005 [0.002, 0.007] | 0.003 [0.000, 0.005] | 0.004-0.005 |
| N64_K48_ppa8-24 | 1746.528 [1711.167, 1781.643] | 1692.902 [1655.533, 1729.638] | 1589.481 [1501.158, 1686.474] | 0.000 [-0.001, 0.001] | 0.000 [-0.001, 0.001] | -0.000-0.000 |

### Learning curves, keys `full`, seed 0 (regret to Bayes; same 6000 steps for every model)

| arm | width | train episodes | params | N8 (in distribution) | N64 (zero-shot) | N64 random-net control |
|---|---|---|---|---|---|---|
| attn1 | 8 | 50000 | 1194 | 0.028 [0.019, 0.036] | 102.440 [98.532, 106.089] | 17.426 |
| attn1 | 16 | 50000 | 3626 | 0.028 [0.019, 0.036] | 225.831 [217.515, 233.653] | 1.644 |
| attn1 | 32 | 1000 | 13386 | 6.018 [4.696, 7.632] | 927.224 [766.607, 1095.557] | 118.899 |
| attn1 | 32 | 3000 | 13386 | 0.874 [0.691, 1.070] | 2199.637 [2103.284, 2300.903] | 87.672 |
| attn1 | 32 | 10000 | 13386 | 0.028 [0.018, 0.037] | 81.976 [78.811, 85.070] | 43.328 |
| attn1 | 32 | 50000 | 13386 | 0.029 (main, s0) | 127.646 (main, s0) | - |
| attn1 | 64 | 50000 | 51338 | 0.027 [0.018, 0.036] | 237.306 [228.402, 245.696] | 1.895 |
| attn1 | 128 | 50000 | 200970 | 0.028 [0.019, 0.036] | 289.555 [278.691, 300.126] | 17.229 |
| attnloop | 8 | 50000 | 1194 | 0.023 [0.014, 0.031] | 374.166 [360.673, 386.604] | 6.258 |
| attnloop | 16 | 50000 | 3626 | 0.023 [0.014, 0.031] | 936.065 [901.988, 968.295] | 0.115 |
| attnloop | 32 | 1000 | 13386 | 5.756 [4.778, 6.787] | 1900.756 [1561.474, 2260.650] | 86.838 |
| attnloop | 32 | 3000 | 13386 | 3.905 [3.262, 4.579] | 619.369 [508.623, 746.899] | 92.054 |
| attnloop | 32 | 10000 | 13386 | 0.027 [0.016, 0.038] | 123.382 [117.934, 128.634] | 45.968 |
| attnloop | 32 | 50000 | 13386 | 0.019 (main, s0) | 28.956 (main, s0) | - |
| attnloop | 64 | 50000 | 51338 | 0.020 [0.011, 0.028] | 56.216 [50.367, 62.349] | 7.028 |
| attnloop | 128 | 50000 | 200970 | 0.021 [0.013, 0.029] | 2124.203 [2041.278, 2209.925] | 15.739 |
| clearloop | 8 | 50000 | 228 | 0.013 [0.006, 0.019] | 0.000 [-0.000, 0.001] | 0.185 |
| clearloop | 16 | 50000 | 708 | 0.012 [0.006, 0.019] | 0.000 [-0.000, 0.001] | 0.170 |
| clearloop | 32 | 1000 | 2436 | 0.015 [0.007, 0.023] | 0.001 [-0.001, 0.004] | 0.124 |
| clearloop | 32 | 3000 | 2436 | 0.013 [0.007, 0.020] | 0.000 [-0.000, 0.001] | 0.153 |
| clearloop | 32 | 10000 | 2436 | 0.013 [0.006, 0.019] | 0.002 [-0.000, 0.004] | 0.165 |
| clearloop | 32 | 50000 | 2436 | 0.012 (main, s0) | 0.006 (main, s0) | - |
| clearloop | 64 | 50000 | 8964 | 0.012 [0.006, 0.019] | 0.000 [-0.000, 0.001] | 0.173 |
| clearloop | 128 | 50000 | 34308 | 0.012 [0.006, 0.019] | 0.001 [-0.000, 0.001] | 0.172 |

### Regret to Bayes, keys `noisy` (nats/episode, mean over 3 training seeds, 95% bootstrap CI over episodes)

| test condition | oracle_clearing | extremized (alpha fit at target) | extremized (train alpha) | attn1 | attnloop | attnloop T8 | clearloop | clearloop T64 | lowest random-net control (any arm) |
|---|---|---|---|---|---|---|---|---|---|
| N4 | 0.021 [0.014, 0.028] | 0.044 [0.033, 0.056] | 0.044 [0.033, 0.057] | 0.042 [0.031, 0.052] | 0.042 [0.032, 0.052] | 0.049 [0.035, 0.062] | 0.042 [0.032, 0.052] | 0.043 [0.033, 0.053] | 0.404 |
| N8 | 0.012 [0.006, 0.018] | 0.048 [0.034, 0.061] | 0.049 [0.035, 0.061] | 0.043 [0.031, 0.055] | 0.041 [0.030, 0.053] | 0.048 [0.034, 0.062] | 0.042 [0.030, 0.055] | 0.046 [0.033, 0.059] | 0.447 |
| N16 | 0.000 [0.000, 0.000] | 0.029 [0.019, 0.039] | 0.033 [0.023, 0.042] | 0.157 [0.136, 0.174] | 0.190 [0.167, 0.211] | 0.376 [0.339, 0.412] | 0.023 [0.014, 0.033] | 0.025 [0.014, 0.035] | 0.447 |
| N32 | 0.000 [-0.000, 0.000] | 0.019 [0.011, 0.026] | 0.024 [0.016, 0.032] | 14.297 [13.714, 14.860] | 16.590 [15.973, 17.202] | 21.253 [20.433, 22.061] | 0.014 [0.008, 0.021] | 0.015 [0.008, 0.022] | 0.444 |
| N64 | 0.000 [-0.000, 0.000] | 0.008 [0.003, 0.014] | 0.016 [0.009, 0.022] | 112.593 [108.518, 116.522] | 96.648 [93.209, 100.037] | 139.810 [134.960, 144.539] | 0.005 [-0.000, 0.009] | 0.004 [-0.000, 0.009] | 0.441 |
| N8_K24_ppa4-12 | 0.020 [0.012, 0.028] | 0.033 [0.023, 0.043] | 0.033 [0.023, 0.043] | 0.030 [0.019, 0.041] | 0.028 [0.017, 0.039] | 0.035 [0.021, 0.049] | 0.029 [0.018, 0.040] | 0.030 [0.019, 0.041] | 0.670 |
| N8_K48_ppa8-24 | 0.004 [0.000, 0.007] | 0.008 [0.003, 0.013] | 0.008 [0.004, 0.013] | 0.006 [0.002, 0.011] | 0.009 [0.004, 0.014] | 0.011 [0.005, 0.019] | 0.006 [0.002, 0.011] | 0.006 [0.001, 0.012] | 1.063 |
| N8_K24_ppa2-6 | 0.020 [0.013, 0.027] | 0.043 [0.031, 0.055] | 0.074 [0.063, 0.085] | 0.044 [0.033, 0.055] | 0.045 [0.034, 0.056] | 0.045 [0.033, 0.058] | 0.047 [0.036, 0.057] | 0.049 [0.038, 0.060] | 0.572 |
| N8_K48_ppa2-6 | 0.010 [0.003, 0.017] | 0.027 [0.017, 0.038] | 0.080 [0.069, 0.090] | 0.029 [0.019, 0.039] | 0.033 [0.023, 0.043] | 0.033 [0.022, 0.045] | 0.038 [0.028, 0.047] | 0.039 [0.029, 0.048] | 0.622 |
| N8_ppa1-2 | 0.001 [-0.001, 0.003] | 0.047 [0.035, 0.060] | 0.086 [0.072, 0.100] | 0.058 [0.047, 0.070] | 0.056 [0.045, 0.068] | 0.050 [0.040, 0.062] | 0.054 [0.043, 0.066] | 0.060 [0.048, 0.073] | 0.333 |
| N8_ppa6-10 | 0.005 [0.002, 0.009] | 0.018 [0.011, 0.026] | 0.045 [0.030, 0.061] | 0.039 [0.030, 0.048] | 0.027 [0.019, 0.035] | 0.044 [0.033, 0.056] | 0.028 [0.018, 0.038] | 0.030 [0.020, 0.041] | 0.469 |
| N8_ppa10-12 | 0.000 [-0.000, 0.000] | 0.007 [0.004, 0.010] | 0.075 [0.053, 0.099] | 0.952 [0.888, 1.016] | 0.301 [0.274, 0.327] | 0.621 [0.567, 0.676] | 0.022 [0.013, 0.033] | 0.021 [0.012, 0.031] | 0.472 |
| N64_K48_ppa8-24 | -0.000 [-0.000, 0.000] | -0.001 [-0.003, 0.001] | 0.001 [-0.001, 0.003] | 1850.443 [1814.416, 1885.735] | 1625.316 [1593.229, 1655.836] | 1861.843 [1823.681, 1896.920] | -0.001 [-0.003, 0.001] | -0.001 [-0.003, 0.002] | 1.048 |

### Regret to oracle_clearing, keys `noisy` (paired, seed-averaged; < 0 means better than the reports-level ceiling)

| test condition | attn1 | attnloop | attnloop_T8 | clearloop | clearloop_T64 | clearloop regret to Bayes, min-max over seeds |
|---|---|---|---|---|---|---|
| N4 | 0.021 [0.013, 0.028] | 0.021 [0.014, 0.028] | 0.028 [0.017, 0.038] | 0.021 [0.014, 0.028] | 0.022 [0.014, 0.029] | 0.042-0.042 |
| N8 | 0.031 [0.020, 0.041] | 0.029 [0.019, 0.039] | 0.036 [0.024, 0.049] | 0.030 [0.019, 0.041] | 0.034 [0.023, 0.046] | 0.042-0.042 |
| N16 | 0.156 [0.136, 0.174] | 0.189 [0.167, 0.211] | 0.376 [0.339, 0.412] | 0.023 [0.013, 0.033] | 0.024 [0.014, 0.035] | 0.023-0.023 |
| N32 | 14.297 [13.714, 14.860] | 16.590 [15.973, 17.202] | 21.253 [20.433, 22.061] | 0.014 [0.008, 0.021] | 0.015 [0.008, 0.022] | 0.014-0.015 |
| N64 | 112.593 [108.518, 116.522] | 96.648 [93.209, 100.037] | 139.810 [134.960, 144.539] | 0.005 [-0.000, 0.009] | 0.004 [-0.000, 0.009] | 0.004-0.005 |
| N8_K24_ppa4-12 | 0.010 [0.002, 0.018] | 0.008 [0.001, 0.016] | 0.015 [0.006, 0.025] | 0.009 [0.002, 0.016] | 0.010 [0.002, 0.017] | 0.029-0.029 |
| N8_K48_ppa8-24 | 0.003 [-0.001, 0.006] | 0.005 [0.001, 0.009] | 0.008 [0.002, 0.014] | 0.002 [-0.001, 0.007] | 0.002 [-0.002, 0.007] | 0.006-0.006 |
| N8_K24_ppa2-6 | 0.024 [0.015, 0.032] | 0.025 [0.016, 0.034] | 0.025 [0.015, 0.035] | 0.027 [0.018, 0.035] | 0.029 [0.020, 0.038] | 0.047-0.047 |
| N8_K48_ppa2-6 | 0.018 [0.011, 0.025] | 0.023 [0.015, 0.030] | 0.022 [0.014, 0.031] | 0.028 [0.020, 0.034] | 0.028 [0.021, 0.035] | 0.038-0.038 |
| N8_ppa1-2 | 0.057 [0.046, 0.069] | 0.055 [0.044, 0.066] | 0.049 [0.039, 0.061] | 0.053 [0.042, 0.065] | 0.059 [0.047, 0.072] | 0.054-0.055 |
| N8_ppa6-10 | 0.033 [0.025, 0.042] | 0.021 [0.014, 0.028] | 0.038 [0.029, 0.049] | 0.022 [0.014, 0.032] | 0.025 [0.016, 0.035] | 0.027-0.028 |
| N8_ppa10-12 | 0.952 [0.888, 1.015] | 0.301 [0.274, 0.327] | 0.621 [0.567, 0.676] | 0.022 [0.013, 0.033] | 0.021 [0.012, 0.031] | 0.022-0.023 |
| N64_K48_ppa8-24 | 1850.443 [1814.416, 1885.735] | 1625.316 [1593.229, 1655.836] | 1861.843 [1823.681, 1896.920] | -0.001 [-0.003, 0.001] | -0.001 [-0.003, 0.002] | -0.001--0.001 |

### Learning curves, keys `noisy`, seed 0 (regret to Bayes; same 6000 steps for every model)

| arm | width | train episodes | params | N8 (in distribution) | N64 (zero-shot) | N64 random-net control |
|---|---|---|---|---|---|---|
| attn1 | 8 | 50000 | 1194 | 0.042 [0.030, 0.054] | 129.780 [125.105, 134.296] | 14.576 |
| attn1 | 16 | 50000 | 3626 | 0.043 [0.031, 0.055] | 87.955 [84.810, 91.015] | 2.612 |
| attn1 | 32 | 1000 | 13386 | 5.956 [4.586, 7.505] | 2146.531 [2061.863, 2240.460] | 69.469 |
| attn1 | 32 | 3000 | 13386 | 0.419 [0.341, 0.501] | 924.200 [892.001, 955.967] | 66.292 |
| attn1 | 32 | 10000 | 13386 | 0.044 [0.032, 0.056] | 157.861 [152.280, 163.386] | 42.795 |
| attn1 | 32 | 50000 | 13386 | 0.043 (main, s0) | 158.918 (main, s0) | - |
| attn1 | 64 | 50000 | 51338 | 0.043 [0.031, 0.055] | 179.152 [172.800, 185.302] | 1.958 |
| attn1 | 128 | 50000 | 200970 | 0.043 [0.031, 0.055] | 250.635 [241.780, 259.222] | 19.204 |
| attnloop | 8 | 50000 | 1194 | 0.042 [0.030, 0.054] | 215.786 [208.225, 223.028] | 4.993 |
| attnloop | 16 | 50000 | 3626 | 0.041 [0.030, 0.053] | 72.842 [70.320, 75.326] | 0.097 |
| attnloop | 32 | 1000 | 13386 | 4.650 [3.976, 5.361] | 826.401 [784.770, 866.021] | 40.997 |
| attnloop | 32 | 3000 | 13386 | 2.554 [2.113, 3.018] | 565.356 [529.077, 602.622] | 59.423 |
| attnloop | 32 | 10000 | 13386 | 0.099 [0.074, 0.128] | 15.575 [12.821, 18.625] | 43.977 |
| attnloop | 32 | 50000 | 13386 | 0.041 (main, s0) | 115.360 (main, s0) | - |
| attnloop | 64 | 50000 | 51338 | 0.043 [0.031, 0.055] | 141.192 [136.176, 146.115] | 7.240 |
| attnloop | 128 | 50000 | 200970 | 0.043 [0.031, 0.055] | 152.516 [147.102, 157.718] | 17.946 |
| clearloop | 8 | 50000 | 228 | 0.043 [0.030, 0.056] | 0.005 [-0.000, 0.010] | 0.155 |
| clearloop | 16 | 50000 | 708 | 0.042 [0.030, 0.055] | 0.005 [-0.000, 0.009] | 0.157 |
| clearloop | 32 | 1000 | 2436 | 0.054 [0.039, 0.069] | 0.007 [0.001, 0.013] | 0.129 |
| clearloop | 32 | 3000 | 2436 | 0.047 [0.033, 0.060] | 0.004 [-0.001, 0.009] | 0.150 |
| clearloop | 32 | 10000 | 2436 | 0.043 [0.030, 0.056] | 0.006 [0.000, 0.012] | 0.146 |
| clearloop | 32 | 50000 | 2436 | 0.042 (main, s0) | 0.005 (main, s0) | - |
| clearloop | 64 | 50000 | 8964 | 0.042 [0.029, 0.055] | 0.005 [-0.000, 0.009] | 0.162 |
| clearloop | 128 | 50000 | 34308 | 0.042 [0.029, 0.055] | 0.004 [-0.000, 0.009] | 0.168 |

## Files and reproduction
- `l8.py` trains and evaluates; it ran on a spot L4 (g2-standard-8) for 1.48 h, about $1.29 (GPU-LOG). Run: `python l8.py --out DIR`, at commit da9a26b.
- `assemble.py` regenerates every dev batch locally, checks that the saved seeds and Bayes logits match exactly, scores the saved forecasts and writes `json/`, `summary.json` and `tables.md`. It ran in a clean worktree at da9a26b: `python3 assemble.py RUN_DIR`.
- `raw/meta.json` holds the training-loss logs, specs, alphas and parameter counts. `raw/run.log` is the VM log. The per-episode forecasts (16 MB npz) are archived in `/home/agent/data/archive/inconsistency-search/l8-scaling-transfer-20261020/`.
