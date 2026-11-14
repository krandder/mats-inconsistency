
### E10.1 Group A: mean departure d̄_s (path-averaged weight − Bayes), by reliability rank and disagreement D_s tercile (group mean [range over nets]; nets with d̄ < 0)

| source | D low | D mid | D high |
|---|---|---|---|
| most reliable | -0.004 [-0.008, -0.000] (6/6) | -0.005 [-0.008, -0.002] (6/6) | -0.005 [-0.008, -0.002] (6/6) |
| middle | -0.007 [-0.010, -0.004] (6/6) | -0.003 [-0.006, -0.001] (6/6) | +0.001 [-0.001, +0.002] (1/6) |
| least reliable | +0.004 [+0.003, +0.006] (0/6) | +0.006 [+0.006, +0.007] (0/6) | +0.006 [+0.005, +0.007] (0/6) |

| source | local δ: D low / mid / high | gain units d̄/q: D low / mid / high |
|---|---|---|
| most reliable | -0.009 / -0.007 / -0.002 | -0.010 / -0.012 / -0.013 |
| middle | -0.006 / +0.002 / +0.005 | -0.024 / -0.011 / -0.003 |
| least reliable | +0.003 / +0.008 / -0.003 | +0.022 / +0.030 / +0.027 |

Disagreeing third (by reading sd): d̄ most / middle / least reliable = -0.007 (6/6 < 0) / -0.000 / +0.006. corr(d̄, p): all −0.22, disagreeing −0.22; mean |d̄|: all 0.018, disagreeing 0.019.

### E10.1 Group B: mean departure d̄_s (path-averaged weight − Bayes), by reliability rank and disagreement D_s tercile (group mean [range over nets]; nets with d̄ < 0)

| source | D low | D mid | D high |
|---|---|---|---|
| most reliable | -0.016 [-0.027, -0.007] (6/6) | -0.017 [-0.026, -0.011] (6/6) | -0.019 [-0.026, -0.015] (6/6) |
| middle | +0.003 [-0.004, +0.006] (1/6) | +0.003 [-0.002, +0.006] (1/6) | +0.004 [+0.001, +0.006] (0/6) |
| least reliable | +0.010 [+0.007, +0.012] (0/6) | +0.010 [+0.008, +0.012] (0/6) | +0.008 [+0.007, +0.009] (0/6) |

| source | local δ: D low / mid / high | gain units d̄/q: D low / mid / high |
|---|---|---|
| most reliable | -0.016 / -0.017 / -0.018 | -0.042 / -0.044 / -0.045 |
| middle | +0.004 / +0.004 / +0.005 | +0.000 / +0.001 / +0.006 |
| least reliable | +0.010 / +0.009 / -0.002 | +0.041 / +0.044 / +0.030 |

Disagreeing third (by reading sd): d̄ most / middle / least reliable = -0.018 (6/6 < 0) / +0.005 / +0.008. corr(d̄, p): all −0.40, disagreeing −0.37; mean |d̄|: all 0.026, disagreeing 0.026.

### E10.2 Does a model of the departure predict the shared mean error on held-out test episodes? (score: finite-ensemble scale; ≥ 0.5 = well)

| nets | departure model (HGB on d̄): score [95% CI] / disagreeing third | same form fitted directly (MLP): score [95% CI] / disagreeing | measured d̄, not held out: score / disagreeing | d̄ prediction R² (test) |
|---|---|---|---|---|
| all | 0.42 [0.38, 0.45] / 0.41 | 0.23 [0.19, 0.27] / 0.26 | 0.93 / 0.98 | 0.46 |
| A | 0.35 [0.32, 0.39] / 0.34 | 0.19 [0.14, 0.23] / 0.21 | 0.92 / 0.97 | 0.41 |
| B | 0.39 [0.36, 0.42] / 0.39 | 0.36 [0.31, 0.39] / 0.34 | 0.98 / 1.03 | 0.41 |

### Predictions

- P86: holds ({"A": {"n_neg": 6, "mean": -0.004966772397084647}, "B": {"n_neg": 6, "mean": -0.018651147175825205}})
- P87: fails ({"score": 0.3548787690220464})
