### A. The decisive test: one extra row that is a copy vs an independent source with the same query reading

Pull toward the duplicated reading o_s0 when one row is added to a 6-source episode (600 episodes; 3 seeds). (i) true copy; (ii) same values on all 9 questions, new identity; (iv) new independent source (fresh labelled readings) with the same query reading. A reference with no within-episode comparison gives the same pull for all three: R1 0.076, R2 0.061. Bayes: copy 0, independent 0.176. In-context dedup = 1 − pull(i)/pull(iv): 0 = the model cannot tell a copy from an independent source.

| depth | keys | (i) copy | (ii) new identity, same values | (iv) independent source | in-context dedup 1 − (i)/(iv) |
|---|---|---|---|---|---|
| 1 | 0 | 0.049 | 0.050 | 0.054 | 0.09 [0.07..0.10] |
| 1 | 1 | 0.040 | 0.040 | 0.128 | 0.69 [0.63..0.73] |
| 2 | 0 | 0.048 | 0.048 | 0.087 | 0.44 [0.41..0.50] |
| 2 | 1 | 0.030 | 0.031 | 0.160 | 0.81 [0.81..0.82] |
| 4 | 0 | 0.047 | 0.048 | 0.093 | 0.49 [0.47..0.52] |
| 4 | 1 | 0.026 | 0.026 | 0.154 | 0.83 [0.81..0.86] |
| 8 | 0 | 0.046 | 0.047 | 0.090 | 0.49 [0.44..0.54] |
| 8 | 1 | 0.031 | 0.031 | 0.157 | 0.80 [0.78..0.82] |

### B. Preregistered split: share of each model's MSE improvement over count-all that is beyond R1 / R2

MSE of the predictive mean to the arm's Bayes mean, mean over 3 seeds. In-context share beyond R = (E_R − E_model)/(E_count-all − E_model); prior-level share = 1 − that. Shares only where the model improves on count-all by > 0.005 (on dup_singletons and singletons_nodup count-all's iid assumption is nearly exact, so 'improvement over count-all' is ill-posed there; see C).


keys = 0. References: dup_singletons: count-all 0.0191, R1 0.0445, R2 0.0370; shared_nodup: count-all 0.1134, R1 0.0467, R2 0.0298; singletons_nodup: count-all 0.0007, R1 0.0294, R2 0.0234

| depth | MSE shared_nodup | in-context beyond R1 | beyond R2 | MSE singletons_nodup (R1, R2 above) | MSE dup_singletons |
|---|---|---|---|---|---|
| 1 | 0.0289 | 0.21 | 0.01 | 0.0266 | 0.0350 |
| 2 | 0.0191 | 0.29 | 0.11 | 0.0073 | 0.0190 |
| 4 | 0.0166 | 0.31 | 0.14 | 0.0040 | 0.0146 |
| 8 | 0.0171 | 0.31 | 0.13 | 0.0049 | 0.0151 |

keys = 1. References: dup_singletons: count-all 0.0180, R1 0.0457, R2 0.0380; shared_nodup: count-all 0.1168, R1 0.0495, R2 0.0320; singletons_nodup: count-all 0.0000, R1 0.0302, R2 0.0244

| depth | MSE shared_nodup | in-context beyond R1 | beyond R2 | MSE singletons_nodup (R1, R2 above) | MSE dup_singletons |
|---|---|---|---|---|---|
| 1 | 0.0127 | 0.35 | 0.19 | 0.0077 | 0.0156 |
| 2 | 0.0047 | 0.40 | 0.24 | 0.0031 | 0.0065 |
| 4 | 0.0048 | 0.40 | 0.24 | 0.0029 | 0.0062 |
| 8 | 0.0039 | 0.40 | 0.25 | 0.0021 | 0.0068 |

### C. Duplicate excess (post-hoc, PREREG addendum): MSE(dup_singletons) − MSE(singletons_nodup)

References (no keys): count-all 0.0184, R1 0.0152, R2 0.0136, R2eq (drops exact equal values: leaks copy identity) 0.0017, Bayes 0. In-context share = (excess_R1 − excess_model)/(excess_count-all − excess_model).

| depth | keys | model excess | in-context share beyond R1 |
|---|---|---|---|
| 1 | 0 | 0.0084 | 0.68 |
| 1 | 1 | 0.0079 | 0.75 |
| 2 | 0 | 0.0117 | 0.52 |
| 2 | 1 | 0.0033 | 0.83 |
| 4 | 0 | 0.0106 | 0.59 |
| 4 | 1 | 0.0033 | 0.83 |
| 8 | 0 | 0.0103 | 0.60 |
| 8 | 1 | 0.0048 | 0.81 |
