### K = 64 noise floor and "network = Bayes"

| model | floor R² mean / log-sd / KL | network = Bayes R² mean / log-sd / KL | dev sd mean / log-sd / KL |
|---|---|---|---|
| 2L seed 4 | 0.802 / 0.908 / 0.873 | -0.01 / -0.21 / -0.29 | 0.047 / 0.066 / 0.022 |

### Single-site surrogates: R²(mean), closed form / unweighted / weighted (cost as % of the site)

| site : family | 2L seed 4 |
|---|---|
| emb : LR2 | < -10 / < -10 / < -10 (14%) |
| emb : LR4 | < -10 / < -10 / < -10 (26%) |
| emb : LR8 | < -10 / < -10 / < -10 (49%) |
| a1 : fp2 | < -10 / < -10 / < -10 (18%) |
| a1 : fp4 | < -10 / < -10 / < -10 (24%) |
| a1 : fpfull | < -10 / < -10 / < -10 (61%) |
| m1 : LR2 | < -10 / < -10 / < -10 (12%) |
| m1 : LR4 | -1.83 / -0.98 / 0.17 (21%) |
| m1 : LR8 | 0.79 / 0.80 / 0.79 (39%) |
| r1 : LR2 | < -10 / < -10 / < -10 (11%) |
| r1 : LR4 | < -10 / < -10 / < -10 (20%) |
| r1 : LR8 | 0.69 / 0.68 / 0.74 (38%) |
| a2 : fp2 | -8.78 / -6.21 / -5.50 (18%) |
| a2 : fp4 | -6.83 / -5.34 / -5.55 (24%) |
| a2 : fpfull | -6.87 / -5.33 / -5.48 (61%) |
| m2 : LR2 | < -10 / -8.58 / < -10 (12%) |
| m2 : LR4 | -4.13 / -0.99 / -1.23 (21%) |
| m2 : LR8 | 0.79 / 0.79 / 0.80 (39%) |
| r2 : LR2 | < -10 / < -10 / < -10 (11%) |
| r2 : LR4 | -5.61 / -5.79 / -4.72 (20%) |
| r2 : LR8 | -1.37 / -0.97 / -0.25 (38%) |

### Same, R²(log-sd) and R²(KL) for the weighted fits at the ≤ 25% level (rank 4; attention pattern + rank 2)

| site | 2L seed 4 |
|---|---|
| emb (LR4) | -5.11 / < -10 |
| a1 (fp2) | < -10 / < -10 |
| m1 (LR4) | 0.65 / -0.06 |
| r1 (LR4) | < -10 / < -10 |
| a2 (fp2) | -5.26 / < -10 |
| m2 (LR4) | 0.40 / -5.55 |
| r2 (LR4) | 0.24 / < -10 |

### Fit diagnostics (held-out 1,024 fit episodes): J-weighted error E||J(g−f)||² and site KL (nats), rank 4 / pattern + rank 2

| site | 2L seed 4 |
|---|---|
| emb | 2.1e+02 / 1.9e+02 / 16 ; KL 46058 / 45521 / 48727 |
| a1 | 53 / 31 / 26 ; KL 31971 / 24695 / 29054 |
| m1 | 5.4 / 3.3 / 2.1 ; KL 4569 / 3331 / 5995 |
| r1 | 91 / 72 / 21 ; KL 6915 / 6325 / 11822 |
| a2 | 56 / 38 / 33 ; KL 10583 / 7491 / 10924 |
| m2 | 10 / 7.3 / 3.5 ; KL 22365 / 17825 / 26901 |
| r2 | 8.4 / 11 / 1.9 ; KL 6629 / 6259 / 8643 |

### Whole chains (every site replaced): R² mean / log-sd / KL

| chain | cost | 2L seed 4 |
|---|---|---|
| ≤ 25% (emb r4, attn pattern + r2, MLP r4): closed | 3572 | < -10 / < -10 / < -10 |
| ≤ 25% (emb r4, attn pattern + r2, MLP r4): unweighted | 3572 | < -10 / < -10 / < -10 |
| ≤ 25% (emb r4, attn pattern + r2, MLP r4): weighted | 3572 | < -10 / < -10 / < -10 |
| ≤ 25% (emb r4, attn pattern + r2, MLP r4): e2e | 3572 | < -10 / < -10 / < -10 |
| rank 8 (attn pattern + r4): closed | 5892 | < -10 / < -10 / < -10 |
| rank 8 (attn pattern + r4): unweighted | 5892 | < -10 / < -10 / < -10 |
| rank 8 (attn pattern + r4): weighted | 5892 | < -10 / < -10 / < -10 |
| rank 8 (attn pattern + r4): e2e | 5892 | < -10 / < -10 / < -10 |

### Per-source DAG (restricted attention): real weights renormalised / refitted unweighted / refitted weighted, R² mean / log-sd / KL

| variant | 2L seed 4 |
|---|---|
| D1: real | 0.48 / 0.69 / 0.34 |
| D1: unweighted | 0.48 / 0.71 / 0.34 |
| D1: weighted | 0.38 / 0.68 / 0.23 |
| D1+lat: real | 0.48 / 0.69 / 0.34 |
| D1+lat: unweighted | 0.48 / 0.71 / 0.34 |
| D1+lat: weighted | 0.40 / 0.69 / 0.28 |

### Predictions (amendment 7)

- P39 (floor ≥ 0.95 mean, ≥ 0.98 log-sd and KL in every model): FAILS — 2L seed 4 0.802 / 0.908 / 0.873
- P40 (rank-4 weighted ≥ 0.5 wherever unweighted < 0; attention pattern + rank 2 in (0, 0.5)): emb/MLP 0 of 5; attention R² < -10, -5.50
- P41 (e2e ≤ 25% chain ≥ 0.5 and beats the weighted chain): 2L seed 4 e2e < -10 vs weighted < -10
- P42 (weighted D1 ≥ floor − 0.1 in own-source seeds; seed 2 D1+lat ≥ 0.5): 2L seed 4 D1 weighted 0.38 (floor 0.80), D1+lat weighted 0.40