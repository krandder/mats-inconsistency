### Shared spread error retained by each intervention (8 nets; 1 = unchanged, 0 = removed)

| intervention | retained, mean over nets [95% CI] | group A / group B | pull slope on Bayes log-sd (baseline −0.06) | mean offset (baseline 0.03) |
|---|---|---|---|---|
| I1 derived readout (ridge decodes) | 0.94 [0.81, 1.07] | 0.95 / 0.93 | −0.16 | 0.04 |
| I1c derived readout (calibrated decodes) | 0.81 [0.70, 0.93] | 0.83 / 0.80 | −0.05 | 0.04 |
| I2 retrained head on r2 | 1.02 [1.01, 1.04] | 1.01 / 1.04 | −0.04 | 0.03 |
| I3 retrained MLP readout from a2 | 0.76 [0.71, 0.82] | 0.74 / 0.78 | −0.03 | 0.02 |
| I4a MLP2 write mean-ablated | 2.53 [2.27, 2.77] | 2.74 / 2.32 | −0.45 | 0.03 |
| I4b MLP2 skipped | 2.25 [2.08, 2.44] | 2.47 / 2.02 | −0.01 | 0.17 |
| I5 noise off everywhere | 0.95 [0.94, 0.96] | 0.94 / 0.96 | −0.04 | 0.02 |
| I6 noise off at layer 2 | 0.99 [0.99, 0.99] | 0.99 / 0.99 | −0.05 | 0.03 |
| I7 single-draw predictive (diagnostic) | 0.97 [0.97, 0.98] | 0.97 / 0.98 | −0.05 | 0.02 |
| term A: per-pass spread | 0.98 [0.97, 0.98] | 0.97 / 0.98 | −0.05 | 0.02 |
| term B: between-draw jitter | 0.02 [0.02, 0.03] | 0.03 / 0.02 | −0.01 | 0.01 |