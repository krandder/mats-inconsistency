### E13.1 Layer-1 encoding strength of the modified source: big miss minus spread (Δg), E12 fresh pairs

| range / nets | mean Δg [95% CI] | nets > 0 / < 0 | Σ Δg ÷ Σ g(spread) | mean corr(Δg, Δw) across pairs | matches |
|---|---|---|---|---|---|
| large, all 12 | +0.8944 [+0.8785, +0.9105] | 12 / 0 | +0.485 | +0.35 | yes |
| large, A | +0.6193 [+0.6017, +0.6372] | 6 / 0 | +0.459 | +0.48 | – |
| large, B | +1.1696 [+1.1478, +1.1910] | 6 / 0 | +0.511 | +0.21 | – |
| middle, A | -0.0784 [-0.0850, -0.0714] | 0 / 6 | -0.022 | +0.59 | yes |
| middle, B | +0.4887 [+0.4675, +0.5076] | 6 / 0 | +0.104 | +0.37 | yes |

### E13.2 Setting the big-miss encoding strength to the spread version's (values only): share of the weight difference removed

| range / nets | removed R [95% CI] | nets with R > 0 | per-net range | control (other source) R [95% CI] | MSD ratio (patched / unpatched) |
|---|---|---|---|---|---|
| large, all 12 (primary) | +0.81 [+0.79, +0.83] | 12/12 | +0.09 to +1.83 | +0.01 [-0.01, +0.02] | 0.823 |
| large, A | +1.24 [+1.20, +1.28] | 6/6 | +1.00 to +1.83 | +0.16 [+0.14, +0.18] | 0.775 |
| large, B | +0.53 [+0.51, +0.55] | 6/6 | +0.09 to +0.78 | -0.09 [-0.11, -0.08] | 0.870 |
| middle, A | +0.33 [+0.29, +0.36] | 5/6 | -0.03 to +0.62 | +0.02 [+0.02, +0.03] | 0.775 |
| middle, B | +0.43 [+0.41, +0.45] | 6/6 | +0.34 to +0.53 | -0.02 [-0.02, -0.01] | 0.870 |

E13.1 matches the weight pattern: True (large True, middle A True, middle B True). E13.2 verdict: carries.
