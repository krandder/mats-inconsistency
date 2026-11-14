| rule | criterion | per seed | mean [95% CI] | threshold | pass |
|---|---|---|---|---|---|
| E1-D0 | 1-step battle accuracy (ply 0, Z materialised) | 1, 1, 1 | 1.0000 [1.0000, 1.0000] | >= 0.97 | True |
| E1-D0 | 5-known-slot NLL (nats) | 0.00016, 8e-05, 7e-05 | 0.0001 [-0.0000, 0.0002] | <= 0.03 | True |
| E1-D0 | 0-step belief KL to P*_full per hidden slot (nats) | 0.02306, 0.02536, 0.02325 | 0.0239 [0.0207, 0.0271] | >= 0.12 | False |
| E1-D2 (exploratory) | JSD reduction kc_dedup vs mix | 0.8955, 0.8412, 0.9345 | 0.8904 [0.7741, 1.0067] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction kc_dedup vs mix | -0.6697, -0.4158, -0.4969 | -0.5275 [-0.8496, -0.2053] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction kc_dedup vs ataraxos_mmd | 0.6091, 0.4247, 0.5893 | 0.5410 [0.2894, 0.7926] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction kc_dedup vs ataraxos_mmd | 0.1149, 0.1461, 0.1507 | 0.1372 [0.0889, 0.1856] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction attn_L3 vs mix | 0.6713, 0.7734, 0.7263 | 0.7237 [0.5968, 0.8506] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction attn_L3 vs mix | -0.00469, 0.2152, 0.1446 | 0.1184 [-0.1605, 0.3973] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction attn_L3 vs ataraxos_mmd | -0.2296, 0.179, -0.7147 | -0.2551 [-1.3664, 0.8562] | >= 0.5 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | value-MSE reduction attn_L3 vs ataraxos_mmd | 0.4674, 0.5267, 0.5147 | 0.5029 [0.4251, 0.5807] | >= 0.25 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | Spearman(trust) - Spearman(stiff), overall | 0.8279, 0.8872, 0.8282 | 0.8478 [0.7628, 0.9327] | >= 0.1 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | same, mean within 10 entropy strata | 0.2787, 0.3651, 0.2974 | 0.3138 [0.2008, 0.4267] | >= 0.15 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | delusional-rate cut trust vs stiff | 0.5, 0.5, 0.3333 | 0.4444 [0.2054, 0.6835] | >= 0.4 | n/a: exploratory, bar met |
| E1-D5 (exploratory) | Unified-LIR effective win vs P0 | 0.4662, 0.455, 0.5275 | 0.4829 [0.3860, 0.5798] | >= 0.55 | n/a: exploratory, bar not met |
| E1-D5 (exploratory) | Unified-LIR effective win vs Ataraxos-1Step | 0.4562, 0.485, 0.5387 | 0.4933 [0.3893, 0.5974] | >= 0.55 | n/a: exploratory, bar not met |
| E1-D5 (exploratory) | Unified-LIR effective win vs Tabular-MCTS | 0.4988, 0.4738, 0.5413 | 0.5046 [0.4198, 0.5894] | >= 0.55 | n/a: exploratory, bar not met |

### E1-D0 (gate)

| seed | battle acc ply 0 | battle acc all plies | 5-known NLL | 0-step KL vs P*_full | 0-step KL vs P*_c | n positions |
|---|---|---|---|---|---|---|
| 101 | 1.0000 | 0.9998 | 0.0002 | 0.0231 | 0.0755 | 2000 |
| 102 | 1.0000 | 0.9999 | 0.0001 | 0.0254 | 0.0818 | 2000 |
| 103 | 1.0000 | 0.9997 | 0.0001 | 0.0232 | 0.0762 | 2000 |
| mean [95% CI] | 1.0000 [1.0000, 1.0000] | | 0.0001 [-0.0000, 0.0002] | 0.0239 [0.0207, 0.0271] | 0.0778 [0.0692, 0.0864] | |

Gate: battle True, five-known True, belief gap False -> **FAIL**. KL(P*_full || P*_c) on these positions (data property): 0.0754 nats.

### E1-D2 (policy independence under flank skew, K = 24; reference K = 128) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | JSD mix | JSD ataraxos_mmd | JSD kc_dedup | JSD attn_L3 | MSE mix | MSE ataraxos_mmd | MSE kc_dedup | MSE attn_L3 | flank gap | key cos within / across |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 5.99e-03 | 1.60e-03 | 6.26e-04 | 1.97e-03 | 4.92e-03 | 9.28e-03 | 8.21e-03 | 4.94e-03 | 0.078 | 1.000 / 1.000 |
| 102 | 6.07e-03 | 1.67e-03 | 9.63e-04 | 1.37e-03 | 4.87e-03 | 8.07e-03 | 6.89e-03 | 3.82e-03 | 0.084 | 0.972 / 0.966 |
| 103 | 6.89e-03 | 1.10e-03 | 4.51e-04 | 1.88e-03 | 5.37e-03 | 9.47e-03 | 8.04e-03 | 4.60e-03 | 0.091 | 1.000 / 1.000 |
- kc_dedup vs mix: JSD reduction 0.890 [0.774, 1.007], MSE reduction -0.527 [-0.850, -0.205]
- kc_dedup vs ataraxos_mmd: JSD reduction 0.541 [0.289, 0.793], MSE reduction 0.137 [0.089, 0.186]
- attn_L3 vs mix: JSD reduction 0.724 [0.597, 0.851], MSE reduction 0.118 [-0.161, 0.397]
- attn_L3 vs ataraxos_mmd: JSD reduction -0.255 [-1.366, 0.856], MSE reduction 0.503 [0.425, 0.581]

pre-registered bar not met (EXPLORATORY: not a rule test, the D0 gate failed) (all 8 comparisons would have to meet it).

### E1-D3 (theta_trust vs output entropy) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | Spearman trust | Spearman stiff | strata trust | strata stiff | delusional trust | delusional stiff | delusional uniform | n rollouts |
|---|---|---|---|---|---|---|---|---|
| 101 | 0.402 | -0.425 | 0.251 | -0.028 | 0.0033 | 0.0067 | 0.0067 | 35184 |
| 102 | 0.457 | -0.430 | 0.335 | -0.030 | 0.0050 | 0.0100 | 0.0033 | 35184 |
| 103 | 0.393 | -0.435 | 0.276 | -0.022 | 0.0067 | 0.0100 | 0.0050 | 35184 |

Spearman gain overall 0.848 [0.763, 0.933], within strata 0.314 [0.201, 0.427]; stiff delusional choices over 3 seeds: 16. Part 1: pre-registered bar met (EXPLORATORY: not a rule test, the D0 gate failed); part 2: pre-registered bar met (EXPLORATORY: not a rule test, the D0 gate failed).

### E1-D5 (tournaments, 400 games per pair per seed, K = 16) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | vs P0 | vs Ataraxos-1Step | vs Tabular-MCTS |
|---|---|---|---|
| 101 | 0.466 (183/7/210) | 0.456 (180/5/215) | 0.499 (196/7/197) |
| 102 | 0.455 (178/8/214) | 0.485 (190/8/202) | 0.474 (184/11/205) |
| 103 | 0.527 (205/12/183) | 0.539 (208/15/177) | 0.541 (213/7/180) |
- vs P0: 0.483 [0.386, 0.580]
- vs Ataraxos-1Step: 0.493 [0.389, 0.597]
- vs Tabular-MCTS: 0.505 [0.420, 0.589]

pre-registered bar not met (EXPLORATORY: not a rule test, the D0 gate failed). Cells: effective win (wins/draws/losses).

