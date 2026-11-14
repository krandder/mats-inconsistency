
### E9.1 Group A (6 nets): share of the sharpening (w − Σw·q) carried by each part, mean [range]

| part | direct route | through MLP2 | both routes |
|---|---|---|---|
| R: query token's own residual (from layer 1) | −0.05 [−0.07, −0.03] | −0.05 [−0.07, −0.01] | −0.09 [−0.12, −0.04] |
| M: attention logits (readings moving the masses) | 0.05 [0.03, 0.13] | 0.10 [0.04, 0.22] | 0.15 [0.09, 0.25] |
| V: values | 0.48 [0.33, 0.58] | 0.47 [0.34, 0.55] | 0.94 [0.86, 1.01] |
| all paths | 0.48 [0.32, 0.64] | 0.52 [0.36, 0.68] | 1 |

Sharpness (slope of normalised weights on the Bayes posterior share): effective weights 1.02 [1.01, 1.02], QQ masses 0.67 [0.62, 0.77]. Mean total sensitivity by part: R_direct 0.070, R_mlp 0.065, M_direct -0.004, M_mlp -0.006, V_direct 0.397, V_mlp 0.341.

### E9.2 Group A: patches (flattening f: 1 = weights as flat as the masses; retained: 1 = shared pattern unchanged)

| patch | f [range] | mean error retained [95% CI] / on disagreeing third | mean MSD (all / disagreeing) | spread retained / disagreeing | log-sd MSD | KL ratio (all / disagreeing) |
|---|---|---|---|---|---|---|
| PVg value gain removed | 0.89 [0.80, 1.05] | 1.33 [1.04, 1.57] / 0.94 | 30.54 / 16.64 | 0.96 / 1.17 | 9.46 | 27.14 / 14.10 |
| PV values = linear in reading | 1.74 [1.36, 2.21] | 2.33 [1.79, 2.79] / 1.71 | 82.54 / 42.58 | 1.07 / 1.41 | 39.19 | 64.58 / 37.10 |
| PM MLP2 linearised | 0.18 [0.03, 0.38] | 1.40 [1.15, 1.67] / 1.16 | 5.88 / 3.93 | 1.37 / 1.18 | 2.60 | 4.89 / 3.52 |
| PVg + PM | 1.25 [1.02, 1.52] | 1.57 [1.24, 1.85] / 1.12 | 35.31 / 20.32 | 1.38 / 1.31 | 11.62 | 30.90 / 16.89 |
| PV + PM | 1.57 [0.92, 1.97] | 2.01 [1.59, 2.36] / 1.47 | 46.28 / 26.56 | 1.06 / 1.35 | 37.45 | 44.13 / 25.57 |

### E9.1 Group B (6 nets): share of the sharpening (w − Σw·q) carried by each part, mean [range]

| part | direct route | through MLP2 | both routes |
|---|---|---|---|
| R: query token's own residual (from layer 1) | −0.05 [−0.15, 0.02] | −0.01 [−0.03, 0.00] | −0.07 [−0.17, 0.02] |
| M: attention logits (readings moving the masses) | 0.07 [0.04, 0.09] | 0.10 [0.06, 0.18] | 0.17 [0.11, 0.27] |
| V: values | 0.49 [0.40, 0.60] | 0.40 [0.31, 0.50] | 0.90 [0.82, 0.94] |
| all paths | 0.51 [0.42, 0.60] | 0.49 [0.40, 0.58] | 1 |

Sharpness (slope of normalised weights on the Bayes posterior share): effective weights 0.99 [0.97, 1.00], QQ masses 0.58 [0.48, 0.77]. Mean total sensitivity by part: R_direct 0.252, R_mlp 0.055, M_direct 0.006, M_mlp 0.009, V_direct 0.332, V_mlp 0.205.

### E9.2 Group B: patches (flattening f: 1 = weights as flat as the masses; retained: 1 = shared pattern unchanged)

| patch | f [range] | mean error retained [95% CI] / on disagreeing third | mean MSD (all / disagreeing) | spread retained / disagreeing | log-sd MSD | KL ratio (all / disagreeing) |
|---|---|---|---|---|---|---|
| PVg value gain removed | 0.74 [0.51, 1.37] | 1.41 [1.28, 1.53] / 1.29 | 7.09 / 5.78 | 1.03 / 1.18 | 2.70 | 7.29 / 5.45 |
| PV values = linear in reading | 1.15 [0.80, 2.30] | 2.23 [1.96, 2.51] / 1.84 | 16.21 / 10.99 | 1.01 / 1.43 | 10.87 | 16.67 / 12.20 |
| PM MLP2 linearised | 0.22 [0.11, 0.42] | 1.39 [1.22, 1.56] / 1.16 | 3.11 / 2.47 | 1.24 / 1.20 | 1.89 | 2.93 / 2.48 |
| PVg + PM | 0.96 [0.67, 1.47] | 1.73 [1.53, 1.92] / 1.47 | 8.82 / 7.45 | 1.25 / 1.31 | 3.35 | 9.14 / 7.41 |
| PV + PM | 1.18 [0.81, 2.01] | 2.16 [1.90, 2.42] / 1.76 | 12.56 / 9.59 | 0.97 / 1.36 | 9.25 | 13.88 / 11.10 |

### E9.4 (post hoc) Value path split: each source's own QQ-token value vs all other tokens' values (share of the sharpening)

| group | own QQ value | other tokens' values | mean sensitivity own / cross |
|---|---|---|---|
| A (6 nets) | 0.94 [0.87, 1.02] | −0.00 [−0.01, 0.00] | 0.741 / -0.003 |
| B (6 nets) | 0.61 [0.55, 0.73] | 0.29 [0.18, 0.38] | 0.487 / 0.049 |

### E9.3 (post hoc) Reading gains of the QQ values: masses x gains vs the effective weights

| group | corr(gain, Bayes share p) | mean gain | slope of masses x gains on p | share of the sharpening reproduced | corr(masses x gains, effective weights) | corr(masses, effective weights) |
|---|---|---|---|---|---|---|
| A (6 nets) | 0.39 [0.35, 0.43] | 1.26 [1.20, 1.28] | 1.11 [0.89, 1.38] | 1.29 [0.60, 2.08] | 0.08 [0.02, 0.15] | 0.90 [0.85, 0.92] |
| B (6 nets) | 0.33 [0.30, 0.40] | 1.40 [1.30, 1.46] | 1.36 [0.22, 3.69] | 1.83 [−0.55, 7.05] | 0.03 [−0.01, 0.07] | 0.77 [0.69, 0.89] |

### Predictions

- P79: holds (value 0.94)
- P80: fails (value 0.52)
- P81: holds (PVg 0.89, PM 0.18)
- P82: holds (patches ['A:PVg', 'A:PV', 'A:PVgPM', 'A:PVPM', 'B:PVg', 'B:PV', 'B:PVgPM', 'B:PVPM'])
- P83: fails (patches ['A:PVg', 'A:PV', 'A:PVgPM', 'A:PVPM', 'B:PVg', 'B:PV', 'B:PVgPM', 'B:PVPM'])
- P85: holds (own 0.94)
- P84: fails (share 1.29, corr 0.39)
