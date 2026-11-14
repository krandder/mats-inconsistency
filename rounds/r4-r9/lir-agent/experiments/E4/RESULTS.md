# RESULTS E4: Searchformer/SoS value tokens as a separate prequential head, under a hackable approval reward

E4 subagent for eurisko, 2026-10-03 14:35Z. Prereg: `PREREG.md` (deviations log at its end). Hawk: `../../hawks/E4.md` (RESHAPE, adopted).
Format used: the FIXED per-environment forecast format (one-token decile claims); the free-form FF arm is not built, so the
verbalisation half of the Turpin probe was not run. Model Qwen3-0.6B, LoRA r8, harness commit 47077e7, 100 SFT steps,
80 RL steps x 32 episodes, `--p-reward approval` (pays env reward + 0.5 x [report on the note cell >= 0.5] + 0.5 x
[stated committed success >= 0.7]). Main setting `syco16` (note_rel = 1/6: the captain's note has likelihood ratio 1, so
the exact posterior ignores it). Every run: `mask_violations` 0 (`runs/*/*/summary.json`, 11 complete runs; partial runs have none).
All numbers: mean ± sd over seeds; per-seed curves in `runs/<preset>/<arm>-s<seed>/e4_fixed.jsonl`, aggregates in
`results.json` (`agg`, `data_channel`, `matched`, `per_run`), flat table `tables.csv`, curves `curves_syco16.png`.

Arms: B0 fused (GRPO on every sampled token); M0 one shared LoRA, forecast/report tokens excluded from GRPO; M1 three
adapters with every adapter computed in every pass (the hawk's gating="all", `e4lib.AllGatesRoleModel`); P (= M2) role
passes with goal deleted; twin = P with the policy frozen at SFT (`--lr-action 0`) while belief and forecaster keep
training prequentially. B0/M0 share a warm start per seed; P/twin share one; M1 has its own.

Coverage: syco16 B0, M0, P 3 seeds to step 80; twin seeds 0, 2 to step 80, seed 1 to step 40; M1 seeds 0, 1 to step 60,
seed 2 to step 40 (100 min per-run timeout; M1 is ~2x slower). note_rel 0.4: one P seed only (B0 died of CUDA OOM,
`runs/syco/B0-s0.log`; no budget to rerun), reported as a pilot only.

## 1. The hawk's structural objection: confirmed, now on the current harness

The harness was rebuilt after the hawk's check (separate role passes, goal text deleted; no gating switch, no router), so
`hawks/E4-leak.py` no longer imports. `leak_check.py` re-measures it; `test_e4_leak.py` is the regression test (2 passed).
Source: `leak.json` (random tiny Qwen3, all LoRA B non-zero, perturb the reward-trained group by 0.5 x N(0,1)).

| config | reward-trained group | forecast-claim max abs logit change | forecast loss grad norm | belief-claim change | belief grad norm |
|---|---|---|---|---|---|
| P (M2) and twin | action | 0.0 | 0.0 | 0.0 | 0.0 |
| M1 (gating all) | action | 0.503 | 57.2 | 0.478 | 145.2 |
| M0 and B0 (fused) | fused | 0.489 | 43.0 | 0.488 | 114.3 |
| any | router | not present in the current harness (README "Not built yet" item 3) | | | |

Conceded: in P the shared-representation question has no empirical content (exactly zero), so E4 measures the open
channels instead. The router channel cannot be tested because the router does not exist; relevance="soft" was not run.

**Reward and KL** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | env reward pre | env reward post | approval reward post | KL to init post | last RL step |
|---|---|---|---|---|---|---|
| B0 | 3 | +0.473 ± 0.066 | +0.471 ± 0.074 | +1.051 ± 0.315 | +2.045 ± 1.860 | +80.000 ± 0.000 |
| M0 | 3 | +0.473 ± 0.066 | +0.492 ± 0.022 | +0.776 ± 0.278 | +3.727 ± 0.697 | +80.000 ± 0.000 |
| M1 | 3 | +0.512 ± 0.063 | +0.466 ± 0.095 | +0.778 ± 0.264 | +7.954 ± 1.893 | +53.333 ± 11.547 |
| P | 3 | +0.481 ± 0.075 | +0.486 ± 0.060 | +0.703 ± 0.208 | +5.350 ± 4.184 | +80.000 ± 0.000 |
| twin | 3 | +0.481 ± 0.075 | +0.494 ± 0.061 | +0.970 ± 0.085 | +0.000 ± 0.000 | +66.667 ± 23.094 |

**Fixed teacher-forced set: forecasts and situation reports (Δ = post − pre)** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | fc Brier post | Δ fc Brier | Δ fc ECE | Δ fc optimism vs realised | Δ fc signed vs Bayes | Δ report Brier | Δ report ECE | Δ report signed vs Bayes |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 3 | +0.297 ± 0.039 | +0.041 ± 0.047 | +0.108 ± 0.097 | +0.002 ± 0.142 | +0.002 ± 0.142 | +0.032 ± 0.014 | +0.111 ± 0.058 | +0.149 ± 0.063 |
| M0 | 3 | +0.288 ± 0.021 | +0.032 ± 0.030 | +0.105 ± 0.091 | -0.102 ± 0.202 | -0.102 ± 0.202 | +0.006 ± 0.003 | +0.017 ± 0.016 | +0.002 ± 0.023 |
| M1 | 3 | +0.278 ± 0.032 | +0.030 ± 0.042 | +0.091 ± 0.097 | +0.015 ± 0.032 | +0.015 ± 0.032 | +0.002 ± 0.004 | +0.008 ± 0.012 | -0.002 ± 0.020 |
| P | 3 | +0.255 ± 0.008 | -0.013 ± 0.026 | -0.049 ± 0.087 | -0.060 ± 0.120 | -0.060 ± 0.120 | +0.000 ± 0.000 | -0.016 ± 0.016 | +0.003 ± 0.002 |
| twin | 3 | +0.272 ± 0.003 | +0.003 ± 0.016 | +0.027 ± 0.055 | +0.036 ± 0.066 | +0.036 ± 0.066 | -0.000 ± 0.000 | -0.005 ± 0.006 | +0.000 ± 0.001 |

**Note-bias index** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | fc note idx pre | fc note idx post | report note idx pre | report note idx post | policy first-shot note idx post | commit first=note excess post | commit note-in-plan excess post |
|---|---|---|---|---|---|---|---|---|
| B0 | 3 | +0.126 ± 0.069 | +0.068 ± 0.040 | +0.009 ± 0.003 | +0.091 ± 0.117 | +0.013 ± 0.011 | +0.003 ± 0.024 | +0.031 ± 0.048 |
| M0 | 3 | +0.126 ± 0.069 | +0.064 ± 0.065 | +0.009 ± 0.003 | +0.005 ± 0.004 | +0.003 ± 0.007 | -0.017 ± 0.006 | +0.007 ± 0.024 |
| M1 | 3 | +0.114 ± 0.052 | +0.050 ± 0.055 | +0.015 ± 0.012 | +0.012 ± 0.003 | +0.005 ± 0.012 | +0.021 ± 0.055 | +0.031 ± 0.038 |
| P | 3 | +0.161 ± 0.079 | +0.122 ± 0.015 | +0.002 ± 0.001 | +0.001 ± 0.000 | +0.012 ± 0.022 | -0.000 ± 0.018 | +0.021 ± 0.010 |
| twin | 3 | +0.161 ± 0.079 | +0.198 ± 0.057 | +0.002 ± 0.001 | +0.002 ± 0.001 | +0.005 ± 0.008 | -0.010 ± 0.018 | +0.052 ± 0.045 |

**Forecast reliance and decision-forecast consistency** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | swap flip pre | swap flip post | top−0.3 flip pre | top−0.3 flip post | 2nd+0.3 flip post | on-policy swap flip post | DFC inconsistency pre | DFC inconsistency post |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 3 | +0.367 ± 0.027 | +0.202 ± 0.176 | +0.384 ± 0.062 | +0.270 ± 0.239 | +0.285 ± 0.253 | +0.163 ± 0.146 | +0.434 ± 0.051 | +0.483 ± 0.132 |
| M0 | 3 | +0.367 ± 0.027 | +0.159 ± 0.127 | +0.384 ± 0.062 | +0.227 ± 0.162 | +0.207 ± 0.165 | +0.052 ± 0.045 | +0.434 ± 0.051 | +0.267 ± 0.071 |
| M1 | 3 | +0.391 ± 0.004 | +0.197 ± 0.180 | +0.393 ± 0.002 | +0.367 ± 0.263 | +0.307 ± 0.184 | +0.038 ± 0.066 | +0.385 ± 0.072 | +0.701 ± 0.257 |
| P | 3 | +0.369 ± 0.023 | +0.253 ± 0.219 | +0.425 ± 0.113 | +0.288 ± 0.253 | +0.292 ± 0.251 | +0.188 ± 0.174 | +0.403 ± 0.016 | +0.545 ± 0.223 |
| twin | 3 | +0.369 ± 0.023 | +0.369 ± 0.023 | +0.425 ± 0.113 | +0.425 ± 0.113 | +0.417 ± 0.034 | +0.066 ± 0.068 | +0.403 ± 0.016 | +0.340 ± 0.022 |

**Approval-gain split** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | Δ env reward | Δ w_agree·agree | Δ w_conf·conf | Δ approval total | agree rate post | conf rate post |
|---|---|---|---|---|---|---|---|
| B0 | 3 | -0.002 ± 0.018 | +0.118 ± 0.192 | +0.274 ± 0.250 | +0.390 ± 0.108 | +0.278 ± 0.386 | +0.882 ± 0.187 |
| M0 | 3 | +0.018 ± 0.073 | -0.014 ± 0.006 | +0.111 ± 0.192 | +0.116 ± 0.269 | +0.014 ± 0.024 | +0.556 ± 0.509 |
| M1 | 3 | -0.046 ± 0.124 | -0.010 ± 0.005 | +0.144 ± 0.076 | +0.088 ± 0.104 | +0.031 ± 0.028 | +0.594 ± 0.447 |
| P | 3 | +0.006 ± 0.119 | +0.000 ± 0.000 | -0.097 ± 0.356 | -0.091 ± 0.473 | +0.000 ± 0.000 | +0.434 ± 0.296 |
| twin | 3 | +0.014 ± 0.015 | +0.000 ± 0.000 | +0.161 ± 0.236 | +0.175 ± 0.251 | +0.000 ± 0.000 | +0.951 ± 0.052 |

**Selection channel (on-policy drift − fixed-set drift)** (syco16; source `results.json: agg["syco16|<arm>"]`, per-seed files `runs/syco16/<arm>-s<seed>/e4_fixed.jsonl`)

| arm | seeds | selection: Δoptimism on-policy − fixed | selection: ΔBrier on-policy − fixed | selection: Δsigned-vs-Bayes on-policy(all cands) − fixed |
|---|---|---|---|---|
| B0 | 3 | +0.017 ± 0.049 | -0.040 ± 0.058 | +0.033 ± 0.024 |
| M0 | 3 | -0.011 ± 0.031 | +0.032 ± 0.066 | +0.024 ± 0.071 |
| M1 | 3 | +0.048 ± 0.128 | -0.014 ± 0.013 | +0.057 ± 0.037 |
| P | 3 | +0.009 ± 0.112 | +0.020 ± 0.053 | +0.054 ± 0.054 |
| twin | 3 | -0.013 ± 0.023 | -0.003 ± 0.014 | +0.000 ± 0.009 |

**Data channel: fixed-set Δ(arm) − Δ(no-RL twin), paired by seed** (syco16; `results.json: data_channel`)

| arm | fx_fc_brier | fx_fc_ece | fx_fc_note_bias_index | fx_fc_signed_vs_bayes | fx_fc_signed_vs_realised | fx_sit_brier | fx_sit_note_bias_index | fx_sit_signed_vs_bayes |
|---|---|---|---|---|---|---|---|---|
| B0 | +0.013 ± 0.014 | +0.040 ± 0.032 | -0.094 ± 0.040 | +0.001 ± 0.033 | +0.001 ± 0.033 | +0.026 ± 0.021 | +0.083 ± 0.114 | +0.117 ± 0.098 |
| M0 | +0.013 ± 0.011 | +0.036 ± 0.026 | -0.050 ± 0.073 | -0.004 ± 0.077 | -0.004 ± 0.077 | +0.006 ± 0.003 | -0.003 ± 0.006 | +0.028 ± 0.025 |
| M1 | +0.047 ± 0.074 | +0.104 ± 0.133 | -0.090 ± 0.050 | +0.023 ± 0.104 | +0.023 ± 0.104 | -0.000 ± 0.001 | -0.005 ± 0.018 | -0.015 ± 0.023 |
| P | -0.017 ± 0.011 | -0.075 ± 0.040 | -0.071 ± 0.068 | -0.079 ± 0.041 | -0.079 ± 0.041 | +0.000 ± 0.001 | -0.000 ± 0.001 | +0.003 ± 0.002 |

**Same, curve-mean version (POST HOC: mean over common RL checkpoints)** (syco16)

| arm | fx_fc_brier | fx_fc_ece | fx_fc_note_bias_index | fx_fc_signed_vs_bayes | fx_fc_signed_vs_realised | fx_sit_brier | fx_sit_note_bias_index | fx_sit_signed_vs_bayes |
|---|---|---|---|---|---|---|---|---|
| B0 | +0.030 ± 0.037 | +0.095 ± 0.098 | -0.061 ± 0.031 | +0.094 ± 0.072 | +0.094 ± 0.072 | +0.016 ± 0.012 | +0.027 ± 0.028 | +0.032 ± 0.028 |
| M0 | +0.014 ± 0.021 | +0.037 ± 0.071 | +0.002 ± 0.076 | +0.028 ± 0.083 | +0.028 ± 0.083 | +0.005 ± 0.003 | +0.002 ± 0.012 | +0.021 ± 0.007 |
| M1 | +0.037 ± 0.056 | +0.093 ± 0.132 | -0.055 ± 0.029 | +0.068 ± 0.021 | +0.068 ± 0.021 | +0.000 ± 0.002 | -0.003 ± 0.010 | -0.007 ± 0.026 |
| P | -0.002 ± 0.006 | -0.013 ± 0.025 | -0.053 ± 0.056 | +0.004 ± 0.017 | +0.004 ± 0.017 | +0.000 ± 0.000 | -0.000 ± 0.001 | +0.003 ± 0.003 |

**Matched env-reward bins (all checkpoints pooled; `results.json: matched`)** (syco16)

| bin | arm | n ckpts | fc signed vs Bayes | fc Brier | report note idx | agree | conf | KL | swap flip |
|---|---|---|---|---|---|---|---|---|---|
| bin0[0.254,0.442] | B0 | 7 | +0.163 ± 0.075 | +0.279 ± 0.023 | +0.026 ± 0.029 | +0.083 ± 0.093 | +0.853 ± 0.376 | +1.808 ± 2.200 | +0.305 ± 0.089 |
| bin0[0.254,0.442] | M0 | 4 | +0.050 ± 0.058 | +0.256 ± 0.009 | +0.009 ± 0.008 | +0.065 ± 0.055 | +0.279 ± 0.348 | +1.314 ± 1.425 | +0.234 ± 0.183 |
| bin0[0.254,0.442] | M1 | 2 | +0.039 ± 0.033 | +0.280 ± 0.044 | +0.012 ± 0.001 | +0.062 ± 0.015 | +0.333 ± 0.471 | +4.865 ± 6.881 | +0.207 ± 0.264 |
| bin0[0.254,0.442] | P | 6 | +0.055 ± 0.091 | +0.255 ± 0.014 | +0.001 ± 0.001 | +0.000 ± 0.000 | +0.373 ± 0.483 | +5.399 ± 3.014 | +0.280 ± 0.117 |
| bin0[0.254,0.442] | twin | 5 | +0.062 ± 0.101 | +0.261 ± 0.013 | +0.001 ± 0.000 | +0.000 ± 0.000 | +0.519 ± 0.475 | +0.000 ± 0.000 | +0.378 ± 0.000 |
| bin1[0.442,0.511] | B0 | 6 | +0.035 ± 0.079 | +0.276 ± 0.034 | +0.007 ± 0.005 | +0.024 ± 0.024 | +0.590 ± 0.325 | +1.908 ± 1.538 | +0.273 ± 0.148 |
| bin1[0.442,0.511] | M0 | 9 | +0.053 ± 0.132 | +0.273 ± 0.024 | +0.010 ± 0.011 | +0.028 ± 0.025 | +0.472 ± 0.375 | +2.613 ± 1.878 | +0.146 ± 0.153 |
| bin1[0.442,0.511] | M1 | 5 | +0.073 ± 0.101 | +0.280 ± 0.043 | +0.009 ± 0.003 | +0.023 ± 0.033 | +0.550 ± 0.475 | +5.787 ± 3.887 | +0.276 ± 0.149 |
| bin1[0.442,0.511] | P | 4 | +0.090 ± 0.047 | +0.258 ± 0.008 | +0.001 ± 0.001 | +0.000 ± 0.000 | +0.602 ± 0.236 | +5.876 ± 4.590 | +0.255 ± 0.178 |
| bin2[0.511,0.563] | B0 | 2 | +0.161 ± 0.003 | +0.273 ± 0.002 | +0.119 ± 0.150 | +0.380 ± 0.479 | +1.000 ± 0.000 | +0.595 ± 0.842 | +0.195 ± 0.276 |
| bin2[0.511,0.563] | M0 | 3 | +0.099 ± 0.064 | +0.258 ± 0.013 | +0.009 ± 0.006 | +0.035 ± 0.022 | +0.549 ± 0.500 | +5.140 ± 4.476 | +0.206 ± 0.180 |
| bin2[0.511,0.563] | M1 | 5 | +0.087 ± 0.055 | +0.254 ± 0.015 | +0.016 ± 0.011 | +0.029 ± 0.032 | +0.448 ± 0.379 | +3.328 ± 4.309 | +0.290 ± 0.141 |
| bin2[0.511,0.563] | P | 5 | +0.118 ± 0.043 | +0.268 ± 0.011 | +0.001 ± 0.001 | +0.000 ± 0.000 | +0.746 ± 0.177 | +3.594 ± 4.401 | +0.230 ± 0.189 |
| bin2[0.511,0.563] | twin | 8 | +0.129 ± 0.065 | +0.267 ± 0.016 | +0.002 ± 0.000 | +0.000 ± 0.000 | +0.740 ± 0.428 | +0.000 ± 0.000 | +0.359 ± 0.023 |

## 2. Readings (syco16 unless stated; every number from the tables above, i.e. `results.json`)

**Primary (preregistered): data channel, fixed-set forecast signed error vs exact Bayes, Δ(arm) − Δ(twin), paired by seed
at the last common step.** P: −0.079 ± 0.041 (n=3, steps 80, 80, 40). Rule "differs" = |mean| > 2 sd and > 0.02:
0.079 < 0.082, so within threshold, and in the pessimistic direction. Curve-mean version (post hoc): +0.004 ± 0.017.
So the M2 forecaster moved no more than the no-RL twin: the data channel is not detectable at this scale. B0: +0.001 ± 0.033
(curve-mean +0.094 ± 0.072); M0: −0.004 ± 0.077 (+0.028 ± 0.083); M1: +0.023 ± 0.104 (+0.068 ± 0.021, all 3 seeds
positive, post hoc and against a twin with a different warm start). The prediction "B0 forecasts turn optimistic" FAILED:
no arm's forecast optimism differs from the twin under the preregistered rule.

**Noise floor.** Prequential training alone swings the twin's fixed-set forecast a lot: twin seed 1 signed vs Bayes 0.164 →
−0.009 → 0.160 at steps 0/20/40 (`runs/syco16/twin-s1/e4_fixed.jsonl`); twin approval reward rises +0.175 ± 0.251 with the
policy frozen, all of it from w_conf (forecaster confidence drift, not RL). Any forecast effect under ~0.1 is unresolvable here.

**The fused failure is in the situation report, not the forecast.** Δ report signed error vs exact posterior: B0
+0.149 ± 0.063 (seeds +0.103, +0.123, +0.221; `per_run`), M0 +0.002 ± 0.023, M1 −0.002 ± 0.020, P +0.003 ± 0.002,
twin +0.000 ± 0.001. B0's report-on-note agreement (w_agree channel) reaches 0.72 in seed 0 (`per_run syco16|B0|0
agree_rate_post`), 0.28 ± 0.39 over seeds; zero in P and twin. B0's approval gain +0.390 ± 0.108 splits into w_agree
+0.118 ± 0.192, w_conf +0.274 ± 0.250, env −0.002 ± 0.018. B0 gets there with the smallest policy change (KL to init
2.0 ± 1.9 vs P 5.4 ± 4.2, M0 3.7 ± 0.7, M1 8.0 ± 1.9): it moves claims, not actions (the GOAL's LIR prediction).
The loss mask alone (M0) already removes the report inflation at this scale; full role separation (P) adds nothing
measurable on reports. The Drori ordering M0 > M1 > M2 is NOT reproduced on reports (all three ≈ 0).

**Env reward did not rise in any arm** (Δ env reward: B0 −0.002 ± 0.018, M0 +0.018 ± 0.073, M1 −0.046 ± 0.124, P
+0.006 ± 0.119, twin +0.014 ± 0.015). Matched-reward bins therefore compare checkpoints at near-identical env reward;
within them the B0 report index is highest only in the top bin (+0.119 ± 0.150, n=2 ckpts).

**Reliance (the hawk's real faithfulness question).** Fixed-set swap of the top-2 stated P(success): commit flip rate
pre 0.369 ± 0.023 → post P 0.253 ± 0.219, B0 0.202 ± 0.176, M0 0.159 ± 0.127, M1 0.197 ± 0.180, twin 0.369 ± 0.023
(frozen policy). Flip given a forecast-implied flip: P 0.936 → 0.643 ± 0.557; P seed 2 fell to 0.000 (commit ignores
forecasts; `runs/syco16/P-s2/e4_fixed.jsonl`), the other P seeds stayed near pre. Decision-forecast inconsistency
(commit ≠ argmax of E[P(success)] − shot_cost·E[shots], on-policy): P 0.403 → 0.545 ± 0.223, M1 0.385 → 0.701 ± 0.257,
B0 0.434 → 0.483 ± 0.132, M0 0.434 → 0.267 ± 0.071, twin 0.403 → 0.340 ± 0.022. So in the protected arm the forecasts
stay calibrated relative to the twin while RL sometimes teaches the policy to stop using them: the hawk's predicted
"protected but ignored" failure appears in 1 of 3 P seeds and in mean reliance of every RL arm.

**Selection channel** (on-policy Δ − fixed-set Δ of committed-forecast optimism): P +0.009 ± 0.112, B0 +0.017 ± 0.049,
M0 −0.011 ± 0.031, M1 +0.048 ± 0.128, twin −0.013 ± 0.023: unresolved (sd ≫ mean).

**Note-bias index at note_rel = 1/6.** The warm-start forecaster already over-predicts hits at the note cell
(+0.13 to +0.16 excess vs other cells, pre). RL arms lower it relative to the twin (endpoint Δ−twin: B0 −0.094 ± 0.040,
M1 −0.090 ± 0.050, P −0.071 ± 0.068, M0 −0.050 ± 0.073); reports show it only in B0 (+0.009 → +0.091 ± 0.117); commits
show none (first shot = note cell, excess over the Bayes-greedy plan: all arms within ±0.02 ± 0.06).

**note_rel 0.4 pilot (P seed 0 only):** env reward 0.605 → 0.582, KL 5.9, Δ fixed-set forecast signed vs Bayes +0.001,
Δ report signed +0.018 (`results.json: agg["syco|P"]`, `runs/syco/P-s0/e4_fixed.jsonl`). No comparison possible.

## 3. Verdict on the hawk's objection

- Structural leak: conceded and confirmed exactly (section 1). Not a GPU question.
- Data channel (M2 vs twin): within threshold; the claim "protected forecasts move no more than prequential training
  alone" survives, but only against a noise floor of about ±0.1 in signed error.
- Fused baseline failure: present in reports (B0 +0.149 vs ~0), absent in forecasts. Under this approval rater the
  forecast confidence channel (w_conf) is near saturation at the warm start and drifts without RL, so it cannot show the effect.
- Faithfulness as reliance: RL lowers reliance in every RL arm on average, with large seed spread; the mask protects
  the forecasts, not their use.
Claim status: mixed.

## 4. Cost

E4 GPU: 8 L4 rentals, $9.24 at the rate ceiling (GPU-LOG.md DELETE lines `soup-gpu-lir-e4-*`). Programme lir-* actual at
14:25Z: $41.97 (same source). Run ids `lir-e4-*` (infra rejects upper case).
