World(s): tiny. Cells: wtiny. Mean over the matched runs (d = 0, 1; r = 0).

**Excess log loss against the exact posterior (nats; mean over K = [1, 4, 16])**

| arm @ weight | n = 0 (term alone) | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|---|
| real outcomes only | - | 0.056 | - | - | - | - |
| i @ 1 | - | 0.056 | - | - | - | - |
| iio @ 1 | - | 0.056 | - | - | - | - |
| iis @ 1 | 0.133 | 0.056 | - | - | - | - |
| iii @ 1 | - | 0.056 | - | - | - | - |
| iiisym @ 1 | - | 0.056 | - | - | - | - |
| synth @ 1 | - | 0.056 | - | - | - | - |
| one-pass net trained on the n real vectors | - | - | - | - | - | - |

References (no training on these real vectors): one pass of M size S 0.048; size M -; size L -; the score with exact weights on the same trajectories 0.352; plain average 0.290; the score T, exact 0.0264.

exu at K = 1: 0.631; data-only n = 20,480 at K = 1: -.
exu at K = 4: 0.301; data-only n = 20,480 at K = 4: -.
exu at K = 16: 0.123; data-only n = 20,480 at K = 16: -.
**Log loss on the held-out real outcomes (nats)**

| arm @ weight | n = 0 (term alone) | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|---|
| real outcomes only | - | 0.695 | - | - | - | - |
| i @ 1 | - | 0.695 | - | - | - | - |
| iio @ 1 | - | 0.695 | - | - | - | - |
| iis @ 1 | 0.786 | 0.695 | - | - | - | - |
| iii @ 1 | - | 0.695 | - | - | - | - |
| iiisym @ 1 | - | 0.695 | - | - | - | - |
| synth @ 1 | - | 0.695 | - | - | - | - |
| one-pass net trained on the n real vectors | - | - | - | - | - | - |

The exact posterior itself scores 0.627.

**Gain G = 1 - excess(arm) / excess(real outcomes only), and the paired difference of excess with its 95 % interval (arm minus data-only; mean over the two matched runs; negative = better)**

| arm @ weight | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|
| i @ 1 | +0% (+0.000*) | - | - | - | - |
| iio @ 1 | +0% (+0.000*) | - | - | - | - |
| iis @ 1 | +0% (+0.000*) | - | - | - | - |
| iii @ 1 | +0% (+0.000*) | - | - | - | - |
| iiisym @ 1 | +0% (+0.000*) | - | - | - | - |
| synth @ 1 | +0% (+0.000*) | - | - | - | - |

* = in both matched runs the 95 % interval of the paired difference excludes 0, same sign.

**Cross-order variance on held-out real vectors. Pair mismatch D (noise floor) and V_ord (noise floor), by K**

| predictor | D, K = 1 | D, K = 4 | D, K = 16 | V_ord, K = 1 | V_ord, K = 4 | V_ord, K = 16 |
|---|---|---|---|---|---|---|
| one pass of M (M itself) | 0.000 (0.000) | 0.000 (0.000) | 0.000 | 0.024 (0.000) | 0.024 (0.000) | 0.024 |
| the score, exact weights on K trajectories | 7.604 (4.703) | 2.752 (2.531) | 1.283 | 39.237 (56.523) | 18.164 (20.228) | 13.525 |
| plain average of M(x given h) | 7.604 (4.703) | 0.614 (0.511) | 0.099 | 39.237 (56.523) | 11.762 (7.981) | 1.535 |
| real outcomes only, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| real outcomes only, n = 256 | - | - | - | - | - | - |
| real outcomes only, n = 1,024 | - | - | - | - | - | - |
| real outcomes only, n = 4,096 | - | - | - | - | - | - |
| real outcomes only, n = 20,480 | - | - | - | - | - | - |
| i @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| iio @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| iis @ 1, n = 0 | 0.000 (0.000) | 0.000 (0.000) | 0.000 | 0.008 (0.003) | 0.018 (0.005) | 0.019 |
| iis @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| iii @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| iiisym @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |
| synth @ 1, n = 64 | 0.001 (0.001) | 0.001 (0.001) | 0.000 | 0.004 (0.004) | 0.007 (0.003) | 0.007 |

**Drag (H5), K = 16: placebo-adjusted change of the KL to the exact joint, for the order that was better / worse under data-only [95 % interval]. Positive = got worse.**

| arm @ weight | n | better order | worse order | data-only gap between the two orders |
|---|---|---|---|---|

**Excess by context size (0-9 / 10-19 / 20-29 checks revealed), K = 16**

| arm | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|
| data | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| i @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| iio @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| iis @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| iii @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| iiisym @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
| synth @ 1 | 0.005 ; 0.075 ; 0.241 | - ; - ; - | - ; - ; - | - ; - ; - | - ; - ; - |
