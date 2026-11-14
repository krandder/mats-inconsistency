| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |
|---|---|---|---|---|---|---|---|
| E2-D0 (1) | 1-step battle accuracy (ply-0 battles, Z materialised), test | 0.9999, 1.0000, 0.9999 | 0.9999 | [0.9998, 1.0001] | >= 0.97 | True |  |
| E2-D0 (2) | 7-known-slot NLL (nats), test | 0.0001, 0.0002, 0.0001 | 0.0001 | [0.0000, 0.0002] | <= 0.05 | True |  |
| E2-D0 (3a) | 0-step KL(P*_full || P_0) per hidden slot (nats), test | 0.0484, 0.0490, 0.0479 | 0.0484 | [0.0471, 0.0498] | >= 0.05 | False |  |
| E2-D0 (3a, reported) | same, total over hidden slots per position (nats) | 0.2730, 0.2748, 0.2691 | 0.2723 | [0.2650, 0.2796] | (>= 0.40 noted in spec) | n/a |  |
| E2-D0 (3b-i) | 16 unskewed rollouts: belief KL reduction vs P_0, calibration split | -10.5248, -10.2537, -10.8701 | -10.5495 | [-11.3170, -9.7820] | >= 0.10 (or 3b-ii) | False |  |
| E2-D0 (3b-ii) | 16 unskewed rollouts: action-value MSE reduction vs 0-step, calibration split | -1.3632, -1.0632, -2.0561 | -1.4942 | [-2.7592, -0.2292] | >= 0.10 (or 3b-i) | False |  |
| E2-D0 gate | capacity rule -> None; (1) and (2) and (3a) and (3b) |  |  |  | all | False |  |
| E2-D1 (1) (EXPLORATORY: gate failed) | key cosine within-flank minus across-flank | 0.0001, 0.0008, 0.0111 | 0.0040 | [-0.0113, 0.0193] | >= 0.25, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | JSD_mix / JSD_attn_L3, belief (CI on JSD_mix - JSD_attn) | 0.1494, 0.1629, 0.1720 | 0.1614 | [0.1332, 0.1897] | >= 2.5, CI(diff) > 0 | n/a | diff -0.018673 [-0.020397, -0.016949] |
| E2-D1 (2) (EXPLORATORY: gate failed) | JSD_mix / JSD_attn_L3, value (CI on JSD_mix - JSD_attn) | 3.5370, 3.5948, 3.5901 | 3.5739 | [3.4942, 3.6537] | >= 2.5, CI(diff) > 0 | n/a | diff 0.001315 [0.000946, 0.001683] |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: mix - attn_L3 | 0.0435, 0.0378, 0.0402 | 0.0405 | [0.0333, 0.0477] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: mix - attn_L3 | -0.0005, -0.0009, -0.0001 | -0.0005 | [-0.0015, 0.0006] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: ataraxos_mmd - attn_L3 | 0.1317, 0.1492, 0.1539 | 0.1449 | [0.1158, 0.1740] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: ataraxos_mmd - attn_L3 | 0.0010, 0.0013, 0.0008 | 0.0010 | [0.0004, 0.0017] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: kc_dedup - attn_L3 | 0.1329, 0.1497, 0.1627 | 0.1484 | [0.1114, 0.1855] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: kc_dedup - attn_L3 | 0.0015, 0.0016, 0.0013 | 0.0015 | [0.0010, 0.0019] | > 0, CI > 0 | n/a |  |
| E2-D1 (EXPLORATORY: gate failed) | (1) and all of (2) |  |  |  | all | n/a (would be False) |  |
| E2-D2 (EXPLORATORY: gate failed) | Spearman(beta_trust, -KL(Q*||q_r)) overall | 0.2210, 0.2504, 0.2113 | 0.2276 | [0.1771, 0.2781] | >= 0.25, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | Spearman(beta) - Spearman(-H) overall | 0.6050, 0.6108, 0.5545 | 0.5901 | [0.5133, 0.6670] | >= 0.20, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | same, mean within 10 entropy strata | 0.1844, 0.2269, 0.1495 | 0.1869 | [0.0906, 0.2832] | >= 0.15, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | all three |  |  |  | all | n/a (would be False) |  |
| E2-D3 (a) (EXPLORATORY: gate failed) | utility swap: shift rate toward higher true Miner utility (relevant boards) | 0.3684, 0.3108, 0.3333 | 0.3375 | [0.2654, 0.4096] | >= 0.75, CI lower > 0.5; hash identical; belief change 0.0000 | n/a | hash identical all seeds: True; max belief change 0; relevant boards per seed: 76, 74, 84 |
| E2-D3 (b) (EXPLORATORY: gate failed) | Unified-LIR effective win vs P0 (400 duplicate games/seed) | 0.4775, 0.5138, 0.5175 | 0.5029 | [0.4480, 0.5578] | > 0.52, CI lower > 0.5 | n/a |  |
| E2-D3 (EXPLORATORY: gate failed) | (a) and (b) |  |  |  | all | n/a (would be False) |  |

Capacity rule (calibration split, 3-seed means): primary meets (1),(2),(3a): False; fallback: False; selected: None.
Self-check: 69 headline numbers re-derived from the per-item CSVs; largest |difference| 8.03e-07 (report/selfcheck.csv).

| architecture | quantity | per seed | mean [95% CI] |
|---|---|---|---|
| primary | d0.battle_acc_ply0 | 0.9999, 1.0000, 0.9999 | 0.9999 [0.9998, 1.0001] |
| primary | d0.nll_7known | 0.0001, 0.0002, 0.0001 | 0.0001 [0.0000, 0.0002] |
| primary | d0.kl0_full_per_hidden_slot | 0.0484, 0.0490, 0.0479 | 0.0484 [0.0471, 0.0498] |
| primary | d0.kl0_constraint | 0.0884, 0.0903, 0.0939 | 0.0909 [0.0839, 0.0979] |
| primary | d0.kl_full_vs_constraint_oracle | 0.1180, 0.1180, 0.1180 | 0.1180 [0.1180, 0.1180] |
| primary | d0cal.battle_acc_ply0 | 0.9999, 0.9999, 0.9997 | 0.9999 [0.9995, 1.0002] |
| primary | d0cal.nll_7known | 0.0001, 0.0001, 0.0001 | 0.0001 [0.0001, 0.0001] |
| primary | d0cal.kl0_full_per_hidden_slot | 0.0470, 0.0475, 0.0455 | 0.0467 [0.0442, 0.0492] |
| primary | d0b.kl0 | 0.0470, 0.0475, 0.0455 | 0.0467 [0.0442, 0.0492] |
| primary | d0b.kl16 | 0.5416, 0.5344, 0.5406 | 0.5389 [0.5291, 0.5486] |
| primary | d0b.belief_kl_reduction | -10.5248, -10.2537, -10.8701 | -10.5495 [-11.3170, -9.7820] |
| primary | d0b.mse0 | 0.0163, 0.0189, 0.0104 | 0.0152 [0.0044, 0.0260] |
| primary | d0b.mse16 | 0.0385, 0.0391, 0.0319 | 0.0365 [0.0266, 0.0463] |
| primary | d0b.action_mse_reduction | -1.3632, -1.0632, -2.0561 | -1.4942 [-2.7592, -0.2292] |
| primary | d1.key_gap | 0.0001, 0.0008, 0.0111 | 0.0040 [-0.0113, 0.0193] |
| primary | d1.key_cos_within | 0.9983, 0.9894, 0.9713 | 0.9864 [0.9522, 1.0206] |
| primary | d1.key_cos_across | 0.9983, 0.9886, 0.9603 | 0.9824 [0.9333, 1.0314] |
| primary | d1.ref_cov_per_hidden_slot | 63.3454, 63.3454, 63.3454 | 63.3454 [63.3454, 63.3454] |
| primary | d1.ref_uncovered_share | 0.2197, 0.2194, 0.2194 | 0.2195 [0.2190, 0.2200] |
| primary | d2.spearman_trust | 0.2210, 0.2504, 0.2113 | 0.2276 [0.1771, 0.2781] |
| primary | d2.spearman_stiff | -0.3840, -0.3604, -0.3432 | -0.3625 [-0.4134, -0.3117] |
| primary | d2.strata_trust | 0.1539, 0.1678, 0.1249 | 0.1489 [0.0946, 0.2031] |
| primary | d2.strata_stiff | -0.0305, -0.0591, -0.0245 | -0.0380 [-0.0839, 0.0079] |
| primary | d2.kl_lowest_entropy_tenth | 1.0097, 0.9851, 1.0223 | 1.0057 [0.9587, 1.0527] |
| primary | d2.kl_other | 0.2185, 0.2473, 0.1960 | 0.2206 [0.1567, 0.2845] |
| primary | d2.terminal_share | 0.0937, 0.0944, 0.0947 | 0.0942 [0.0930, 0.0955] |
| primary | d3a.shift_rate_relevant | 0.3684, 0.3108, 0.3333 | 0.3375 [0.2654, 0.4096] |
| primary | d3a.n_relevant | 76.0000, 74.0000, 84.0000 | 78.0000 [64.8552, 91.1448] |
| primary | d3a.changed_action_rate | 0.2583, 0.3042, 0.2750 | 0.2792 [0.2215, 0.3368] |
| primary | d3a.shift_rate_all | 0.1500, 0.1167, 0.1500 | 0.1389 [0.0911, 0.1867] |
| primary | d3a.mean_miner_std | -0.0217, -0.0241, -0.0294 | -0.0250 [-0.0348, -0.0153] |
| primary | d3a.mean_miner_miner | -0.0026, -0.0145, -0.0138 | -0.0103 [-0.0268, 0.0063] |
| primary | d3a.belief_max_abs_change | 0.0000, 0.0000, 0.0000 | 0.0000 [0.0000, 0.0000] |
| primary | d3b.effective_win | 0.4775, 0.5138, 0.5175 | 0.5029 [0.4480, 0.5578] |
| primary | d3b.wins | 176.0000, 183.0000, 197.0000 | 185.3333 [158.7713, 211.8954] |
| primary | d3b.draws | 30.0000, 45.0000, 20.0000 | 31.6667 [0.4086, 62.9247] |
| primary | d3b.losses | 194.0000, 172.0000, 183.0000 | 183.0000 [155.6745, 210.3255] |
| primary | d3b.mean_len | 25.6500, 26.6475, 22.0375 | 24.7783 [18.7531, 30.8035] |
| primary | d1.mix.jsd_belief | 0.0034, 0.0037, 0.0037 | 0.0036 [0.0031, 0.0040] |
| primary | d1.mix.jsd_value | 0.0017, 0.0021, 0.0018 | 0.0018 [0.0013, 0.0023] |
| primary | d1.mix.kl_ref_belief | 0.2040, 0.2016, 0.1985 | 0.2014 [0.1945, 0.2082] |
| primary | d1.mix.mse_ref_value | 0.0015, 0.0017, 0.0017 | 0.0017 [0.0014, 0.0019] |
| primary | d1.ataraxos_mmd.jsd_belief | 0.0110, 0.0145, 0.0147 | 0.0134 [0.0081, 0.0187] |
| primary | d1.ataraxos_mmd.jsd_value | 0.0006, 0.0007, 0.0007 | 0.0007 [0.0005, 0.0009] |
| primary | d1.ataraxos_mmd.kl_ref_belief | 0.2922, 0.3130, 0.3122 | 0.3058 [0.2764, 0.3352] |
| primary | d1.ataraxos_mmd.mse_ref_value | 0.0030, 0.0040, 0.0026 | 0.0032 [0.0013, 0.0050] |
| primary | d1.kc_dedup.jsd_belief | 0.0127, 0.0161, 0.0176 | 0.0155 [0.0092, 0.0217] |
| primary | d1.kc_dedup.jsd_value | 0.0011, 0.0011, 0.0011 | 0.0011 [0.0010, 0.0012] |
| primary | d1.kc_dedup.kl_ref_belief | 0.2934, 0.3135, 0.3210 | 0.3093 [0.2738, 0.3448] |
| primary | d1.kc_dedup.mse_ref_value | 0.0035, 0.0043, 0.0030 | 0.0036 [0.0020, 0.0052] |
| primary | d1.attn_L3.jsd_belief | 0.0227, 0.0225, 0.0216 | 0.0223 [0.0209, 0.0236] |
| primary | d1.attn_L3.jsd_value | 0.0005, 0.0006, 0.0005 | 0.0005 [0.0004, 0.0006] |
| primary | d1.attn_L3.kl_ref_belief | 0.1605, 0.1638, 0.1583 | 0.1609 [0.1540, 0.1677] |
| primary | d1.attn_L3.mse_ref_value | 0.0020, 0.0027, 0.0017 | 0.0021 [0.0010, 0.0033] |
| primary | params | 1001644.0000, 1001644.0000, 1001644.0000 | 1001644.0000 [1001644.0000, 1001644.0000] |
| primary | train_sec | 2220.2834, 2232.1546, 2404.1468 | 2285.5283 [2029.9154, 2541.1411] |
| primary | lir_hash_identical | 1.0000, 1.0000, 1.0000 | 1.0000 [1.0000, 1.0000] |
| fallback | d0.battle_acc_ply0 | 0.9995, 1.0000, 1.0000 | 0.9998 [0.9992, 1.0005] |
| fallback | d0.nll_7known | 0.0001, 0.0002, 0.0002 | 0.0002 [0.0000, 0.0003] |
| fallback | d0.kl0_full_per_hidden_slot | 0.0524, 0.0506, 0.0512 | 0.0514 [0.0491, 0.0537] |
| fallback | d0.kl0_constraint | 0.0787, 0.0804, 0.0821 | 0.0804 [0.0762, 0.0845] |
| fallback | d0.kl_full_vs_constraint_oracle | 0.1180, 0.1180, 0.1180 | 0.1180 [0.1180, 0.1180] |
| fallback | d0cal.battle_acc_ply0 | 0.9999, 1.0000, 1.0000 | 1.0000 [0.9998, 1.0001] |
| fallback | d0cal.nll_7known | 0.0001, 0.0002, 0.0002 | 0.0002 [0.0000, 0.0003] |
| fallback | d0cal.kl0_full_per_hidden_slot | 0.0497, 0.0486, 0.0498 | 0.0493 [0.0476, 0.0511] |
| fallback | d0b.kl0 | 0.0497, 0.0486, 0.0498 | 0.0493 [0.0476, 0.0511] |
| fallback | d0b.kl16 | 0.5304, 0.5168, 0.5412 | 0.5295 [0.4991, 0.5598] |
| fallback | d0b.belief_kl_reduction | -9.6700, -9.6434, -9.8718 | -9.7284 [-10.0386, -9.4181] |
| fallback | d0b.mse0 | 0.0095, 0.0097, 0.0085 | 0.0092 [0.0076, 0.0109] |
| fallback | d0b.mse16 | 0.0307, 0.0294, 0.0292 | 0.0298 [0.0278, 0.0318] |
| fallback | d0b.action_mse_reduction | -2.2178, -2.0417, -2.4440 | -2.2345 [-2.7355, -1.7335] |
| fallback | d1.key_gap | 0.0001, 0.0006, 0.0272 | 0.0093 [-0.0292, 0.0479] |
| fallback | d1.key_cos_within | 0.9995, 0.9888, 0.9200 | 0.9694 [0.8623, 1.0766] |
| fallback | d1.key_cos_across | 0.9994, 0.9881, 0.8927 | 0.9601 [0.8145, 1.1056] |
| fallback | d1.ref_cov_per_hidden_slot | 63.3454, 63.3454, 63.3454 | 63.3454 [63.3454, 63.3454] |
| fallback | d1.ref_uncovered_share | 0.2197, 0.2194, 0.2194 | 0.2195 [0.2190, 0.2200] |
| fallback | d2.spearman_trust | 0.2521, 0.2312, 0.2185 | 0.2339 [0.1918, 0.2761] |
| fallback | d2.spearman_stiff | -0.3332, -0.2912, -0.3312 | -0.3185 [-0.3774, -0.2597] |
| fallback | d2.strata_trust | 0.1887, 0.1807, 0.1445 | 0.1713 [0.1129, 0.2298] |
| fallback | d2.strata_stiff | -0.0277, -0.0521, -0.0472 | -0.0423 [-0.0744, -0.0103] |
| fallback | d2.kl_lowest_entropy_tenth | 1.0368, 0.9631, 1.0009 | 1.0003 [0.9087, 1.0919] |
| fallback | d2.kl_other | 0.1789, 0.1822, 0.1733 | 0.1781 [0.1669, 0.1894] |
| fallback | d2.terminal_share | 0.0952, 0.0857, 0.0915 | 0.0908 [0.0790, 0.1027] |
| fallback | d3a.shift_rate_relevant | 0.4353, 0.2840, 0.3333 | 0.3509 [0.1591, 0.5426] |
| fallback | d3a.n_relevant | 85.0000, 81.0000, 81.0000 | 82.3333 [76.5965, 88.0702] |
| fallback | d3a.changed_action_rate | 0.3208, 0.3000, 0.2667 | 0.2958 [0.2280, 0.3637] |
| fallback | d3a.shift_rate_all | 0.1667, 0.1333, 0.1500 | 0.1500 [0.1086, 0.1914] |
| fallback | d3a.mean_miner_std | -0.0276, -0.0385, -0.0270 | -0.0310 [-0.0472, -0.0149] |
| fallback | d3a.mean_miner_miner | -0.0135, -0.0180, -0.0140 | -0.0151 [-0.0213, -0.0090] |
| fallback | d3a.belief_max_abs_change | 0.0000, 0.0000, 0.0000 | 0.0000 [0.0000, 0.0000] |
| fallback | d3b.effective_win | 0.5088, 0.5025, 0.5337 | 0.5150 [0.4739, 0.5561] |
| fallback | d3b.wins | 190.0000, 198.0000, 204.0000 | 197.3333 [179.8853, 214.7813] |
| fallback | d3b.draws | 27.0000, 6.0000, 19.0000 | 17.3333 [-8.9954, 43.6621] |
| fallback | d3b.losses | 183.0000, 196.0000, 177.0000 | 185.3333 [161.2061, 209.4606] |
| fallback | d3b.mean_len | 28.5025, 20.1050, 22.2400 | 23.6158 [12.7737, 34.4579] |
| fallback | d1.mix.jsd_belief | 0.0034, 0.0038, 0.0036 | 0.0036 [0.0031, 0.0041] |
| fallback | d1.mix.jsd_value | 0.0014, 0.0016, 0.0014 | 0.0015 [0.0011, 0.0019] |
| fallback | d1.mix.kl_ref_belief | 0.1962, 0.1922, 0.1986 | 0.1957 [0.1877, 0.2036] |
| fallback | d1.mix.mse_ref_value | 0.0014, 0.0017, 0.0014 | 0.0015 [0.0011, 0.0019] |
| fallback | d1.ataraxos_mmd.jsd_belief | 0.0138, 0.0138, 0.0131 | 0.0136 [0.0125, 0.0146] |
| fallback | d1.ataraxos_mmd.jsd_value | 0.0004, 0.0008, 0.0004 | 0.0005 [-0.0001, 0.0012] |
| fallback | d1.ataraxos_mmd.kl_ref_belief | 0.2968, 0.3249, 0.3066 | 0.3094 [0.2740, 0.3449] |
| fallback | d1.ataraxos_mmd.mse_ref_value | 0.0021, 0.0022, 0.0022 | 0.0021 [0.0020, 0.0023] |
| fallback | d1.kc_dedup.jsd_belief | 0.0156, 0.0157, 0.0161 | 0.0158 [0.0151, 0.0165] |
| fallback | d1.kc_dedup.jsd_value | 0.0007, 0.0013, 0.0008 | 0.0009 [0.0001, 0.0017] |
| fallback | d1.kc_dedup.kl_ref_belief | 0.2958, 0.3261, 0.3107 | 0.3108 [0.2733, 0.3484] |
| fallback | d1.kc_dedup.mse_ref_value | 0.0023, 0.0026, 0.0027 | 0.0025 [0.0019, 0.0031] |
| fallback | d1.attn_L3.jsd_belief | 0.0224, 0.0207, 0.0226 | 0.0219 [0.0193, 0.0245] |
| fallback | d1.attn_L3.jsd_value | 0.0003, 0.0002, 0.0003 | 0.0003 [0.0002, 0.0003] |
| fallback | d1.attn_L3.kl_ref_belief | 0.1513, 0.1616, 0.1637 | 0.1589 [0.1425, 0.1753] |
| fallback | d1.attn_L3.mse_ref_value | 0.0015, 0.0016, 0.0017 | 0.0016 [0.0015, 0.0017] |
| fallback | params | 492268.0000, 492268.0000, 492268.0000 | 492268.0000 [492268.0000, 492268.0000] |
| fallback | train_sec | 1775.8468, 1789.0688, 1795.9354 | 1786.9503 [1761.5861, 1812.3146] |
| fallback | lir_hash_identical | 1.0000, 1.0000, 1.0000 | 1.0000 [1.0000, 1.0000] |
