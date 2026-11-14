### Per-site sensitivity: search sup Λ_l vs rms over prior paths (||∇φ_l||)

**small net σ 0.5** (σ√ΣΛ² = 25.496, σ√Σrms² = 0.596, last-layer ||w|| = 0.49)

| site | rms (prior paths) | max sampled | max after ascent | Λ used | Λ / rms |
|---|---|---|---|---|---|
| 1 | 0.858 | 10.335 | 3.515 | 10.335 | 12.0 |
| 2 | 0.468 | 4.716 | 3.552 | 4.716 | 10.1 |
| 3 | 0.319 | 2.135 | 16.143 | 16.143 | 50.7 |
| 4 | 0.255 | 2.243 | 23.291 | 23.291 | 91.2 |
| 5 | 0.236 | 1.486 | 40.840 | 40.840 | 173.4 |

**small net σ 0.3** (σ√ΣΛ² = 29.273, σ√Σrms² = 0.458, last-layer ||w|| = 0.68)

| site | rms (prior paths) | max sampled | max after ascent | Λ used | Λ / rms |
|---|---|---|---|---|---|
| 1 | 1.079 | 6.516 | 15.661 | 15.661 | 14.5 |
| 2 | 0.558 | 4.818 | 28.399 | 28.399 | 50.9 |
| 3 | 0.400 | 1.745 | 41.682 | 41.682 | 104.3 |
| 4 | 0.337 | 1.557 | 44.768 | 44.768 | 132.9 |
| 5 | 0.354 | 1.143 | 68.759 | 68.759 | 194.3 |

**PFN 2L σ 0.05** (σ√ΣΛ² = 0.476, σ√Σrms² = 0.042)

| site | rms (prior paths) | max sampled | max after ascent | Λ used | Λ / rms |
|---|---|---|---|---|---|
| emb | 0.570 | 2.938 | 8.822 | 8.822 | 15.5 |
| a1 | 0.322 | 1.099 | 1.912 | 1.912 | 5.9 |
| m1 | 0.186 | 0.824 | 1.223 | 1.223 | 6.6 |
| r1 | 0.226 | 0.918 | 1.301 | 1.301 | 5.8 |
| a2 | 0.231 | 0.860 | 1.983 | 1.983 | 8.6 |
| m2 | 0.156 | 0.546 | 1.080 | 1.080 | 6.9 |
| r2 | 0.342 | 0.624 | 0.838 | 0.838 | 2.4 |

### small net σ 0.5: certified log2 P(A) vs truth (bits kept = certified / true rarity bits)

| true log2 p | t | certificate (bits kept) | O1: rms sensitivities (not a bound) | O2: input alone, log2 P(φ_0 > t) | plain sampling 95% bound at equal compute |
|---|---|---|---|---|---|
| -6.0 | 1.734 | -0.0 (0%) | -2.5 | -996.6 | -6.0 |
| -10.0 | 2.024 | -0.0 (0%) | -3.3 | -996.6 | -10.0 |
| -14.0 | 2.232 | -0.0 (0%) | -4.1 | -996.6 | -14.0 |
| -16.0 | 2.317 | -0.0 (0%) | -4.4 | -996.6 | -15.9 |
| -20.0 | 2.477 | -0.0 (0%) | -5.2 | -996.6 | -19.8 |
| -26.0 | 2.681 | -0.0 (0%) | -6.3 | -996.6 | -24.0 |
| -29.9 | 2.800 | -0.0 (0%) | -7.0 | -996.6 | -25.1 |
| -36.0 | 2.967 | -0.0 (0%) | -8.1 | -996.6 | -25.1 |
| -40.1 | 3.071 | -0.0 (0%) | -8.8 | -996.6 | -25.1 |

Compute: 231 s (= 1.06e+08 plain forward passes at the measured rate). IS: 65536 inputs, 55 shift centres.

### small net σ 0.3: certified log2 P(A) vs truth (bits kept = certified / true rarity bits)

| true log2 p | t | certificate (bits kept) | O1: rms sensitivities (not a bound) | O2: input alone, log2 P(φ_0 > t) | plain sampling 95% bound at equal compute |
|---|---|---|---|---|---|
| -6.0 | 2.087 | -0.0 (0%) | -3.3 | -996.6 | -6.0 |
| -10.0 | 2.354 | -0.0 (0%) | -4.3 | -996.6 | -10.0 |
| -14.0 | 2.543 | -0.0 (0%) | -5.2 | -996.6 | -13.9 |
| -16.0 | 2.618 | -0.0 (0%) | -5.6 | -996.6 | -15.9 |
| -20.0 | 2.754 | -0.0 (0%) | -6.5 | -996.6 | -19.7 |
| -26.0 | 2.927 | -0.0 (0%) | -7.8 | -996.6 | -23.7 |
| -30.0 | 3.031 | -0.0 (0%) | -8.6 | -996.6 | -24.4 |
| -36.0 | 3.171 | -0.0 (0%) | -9.9 | -996.6 | -24.4 |
| -40.0 | 3.255 | -0.0 (0%) | -10.7 | -996.6 | -24.4 |

Compute: 186 s (= 6.62e+07 plain forward passes at the measured rate). IS: 65536 inputs, 55 shift centres.

### PFN 2L σ 0.05: certified log2 P(A) vs truth (event: predictive mean of one noisy pass > t)

Output ceiling by search over reachable final query states: 3.2094 (without the noise ball 3.2079); max over 4,096 tail-tilted real paths 3.1881; max of φ_0 over the IS episodes 3.1742.

| true log2 p | t | ceiling − t | certificate (bits kept) | O1 (not a bound) | O2: log2 P(φ_0 > t) | plain sampling at equal compute |
|---|---|---|---|---|---|---|
| -6.1 | 1.9946 | 1.215 | -3.6 (59%) | -5.8 | -6.0 | -6.1 |
| -8.2 | 2.4782 | 0.731 | -5.0 (62%) | -7.8 | -8.1 | -8.1 |
| -9.9 | 2.8448 | 0.365 | -6.3 (64%) | -9.8 | -10.2 | -9.9 |
| -12.4 | 3.0584 | 0.151 | -7.2 (58%) | -11.5 | -12.2 | -12.3 |
| -14.4 | 3.1430 | 0.066 | -7.6 (53%) | -12.7 | -14.2 | -14.3 |
| -16.3 | 3.1672 | 0.042 | -7.7 (47%) | -13.2 | -18.0 | -16.1 |
| -17.7 | 3.1753 | 0.034 | -7.7 (43%) | -13.5 | -996.6 | -17.3 |
| -20.2 | 3.1833 | 0.026 | -7.7 (38%) | -13.7 | -996.6 | -19.4 |
| -22.6 | 3.1874 | 0.022 | -7.7 (34%) | -13.9 | -996.6 | -20.6 |
| -25.6 | 3.1914 | 0.018 | -7.8 (30%) | -14.0 | -996.6 | -21.7 |
| -30.6 | 3.1954 | 0.014 | -7.8 (25%) | -14.2 | -996.6 | -21.7 |
| -39.9 | 3.1994 | 0.010 | -7.8 (20%) | -14.4 | -996.6 | -21.7 |

Compute: 588 s (= 9.88e+06 forward passes). IS: 32768 episodes, φ_0 from 64 noise draws each.

### Predictions (amendment 8)

- small net σ 0.5: max Λ/rms 173.4; bits kept at 2^-40 0%, at 2^-10 0% (P43: ratio ≥ 2, 40–80% at 2^-40, ≥ 70% at 2^-10); beats sampling at all rungs ≤ 2^-30: False (P44); O1 within 3 bits down to 2^-20: False (P46)
- small net σ 0.3: max Λ/rms 194.3; bits kept at 2^-40 0%, at 2^-10 0% (P43: ratio ≥ 2, 40–80% at 2^-40, ≥ 70% at 2^-10); beats sampling at all rungs ≤ 2^-30: False (P44); O1 within 3 bits down to 2^-20: False (P46)
- PFN: ceiling − t(2^-40) = 0.010 (P45a: ≤ 0.06); bits kept at 2^-10 64% (≥ 70%), at 2^-40 20% (≤ 50%); O1 over-prediction beyond 2^-16: min 3.0 bits (P46: > 3)