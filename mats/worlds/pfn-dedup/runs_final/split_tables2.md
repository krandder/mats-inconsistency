### A. In-distribution test: how much of an exact copy does each model remove using within-episode evidence?

600 episodes with 6 singleton sources plus one exact copy of source s0 (seed 78000). rho = (d mu / d o* for s0, moving both its rows) / (d mu / d o* for another source s1 in the same episode). A reader that cannot tell which rows are copies has rho = 2; full deduplication has rho = 1. In-context dedup = 2 − rho. Mean over 3 seeds; 95% CI from a bootstrap over episodes (seeds fixed); seed range in brackets. conv = seeds meeting the preregistered convergence rule.

References: R1 (additive, no comparison) 2 − rho = 0.01; R2 (Bayes on query readings, no identities or labels) 0.11; Bayes 1.00.

| depth | keys | conv | in-context dedup 2 − rho | 95% CI | seed range | copy pull | untied independent pull (Bayes 0.121) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3/3 | 0.36 | [0.24, 0.47] | 0.30..0.40 | 0.059 | 0.105 |
| 1 | 1 | 2/3 | 0.49 | [0.45, 0.52] | 0.40..0.57 | 0.061 | 0.117 |
| 2 | 0 | 1/3 | 0.26 | [0.19, 0.32] | 0.23..0.29 | 0.080 | 0.117 |
| 2 | 1 | 2/3 | 0.66 | [0.64, 0.67] | 0.64..0.67 | 0.042 | 0.121 |
| 4 | 0 | 3/3 | 0.25 | [0.19, 0.31] | 0.23..0.29 | 0.086 | 0.119 |
| 4 | 1 | 2/3 | 0.70 | [0.68, 0.72] | 0.66..0.75 | 0.036 | 0.120 |
| 8 | 0 | 1/3 | 0.28 | [0.21, 0.34] | 0.26..0.29 | 0.081 | 0.118 |
| 8 | 1 | 1/3 | 0.62 | [0.60, 0.64] | 0.55..0.67 | 0.047 | 0.119 |

### B. Which cue does the deduplication use? Remove one cue from the copy at a time (out of distribution)

In-context dedup 2 − rho for an exact copy and for copies with one cue removed: the exact query tie (copy's query reading offset by delta ~ N(0, 0.5)); equality of the 8 labelled readings (fresh values); identity vector and group key (new ones). A large drop means that cue is used. These copies never occur in training.

| depth | keys | exact copy | no query tie | no labelled equality | new identity and key |
|---|---|---|---|---|---|
| R1 | | 0.01 | -0.00 | 0.01 | 0.01 |
| R2 | | 0.11 | -0.03 | 0.11 | 0.11 |
| 1 | 0 | 0.36 | 0.16 | 0.35 | 0.37 |
| 1 | 1 | 0.49 | 0.43 | 0.49 | 0.12 |
| 2 | 0 | 0.26 | 0.13 | 0.21 | 0.26 |
| 2 | 1 | 0.66 | 0.64 | 0.66 | 0.02 |
| 4 | 0 | 0.25 | 0.10 | 0.21 | 0.24 |
| 4 | 1 | 0.70 | 0.64 | 0.70 | 0.08 |
| 8 | 0 | 0.28 | 0.12 | 0.25 | 0.27 |
| 8 | 1 | 0.62 | 0.57 | 0.62 | 0.09 |

### C. Shared errors in three tiers (preregistered split, corrected reading)

MSE of the predictive mean to the arm's Bayes mean on shared-error episodes without duplicates (n = 3915), mean over 3 seeds. Tiers: prior (R1, no comparison), query-value comparison (R1 → R2), labelled/identity/key evidence (beyond R2). Two normalisations: the preregistered share of the model's gain over count-all, and the share of R's remaining error that the model removes. R1 and R2 are fitted per row count n (here n = 6 always).

| depth | keys | conv | MSE model | count-all | R1 | R2 | gain over count-all: prior / query-value / beyond R2 | R1's error removed | R2's error removed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3/3 | 0.0289 | 0.1134 | 0.0467 | 0.0298 | 0.79 / 0.20 / 0.01 | 0.38 | 0.03 |
| 1 | 1 | 2/3 | 0.0127 | 0.1168 | 0.0495 | 0.0320 | 0.65 / 0.17 / 0.19 | 0.74 | 0.60 |
| 2 | 0 | 1/3 | 0.0191 | 0.1134 | 0.0467 | 0.0298 | 0.71 / 0.18 / 0.11 | 0.59 | 0.36 |
| 2 | 1 | 2/3 | 0.0047 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.24 | 0.91 | 0.85 |
| 4 | 0 | 3/3 | 0.0166 | 0.1134 | 0.0467 | 0.0298 | 0.69 / 0.17 / 0.14 | 0.64 | 0.44 |
| 4 | 1 | 2/3 | 0.0048 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.24 | 0.90 | 0.85 |
| 8 | 0 | 1/3 | 0.0171 | 0.1134 | 0.0467 | 0.0298 | 0.69 / 0.17 / 0.13 | 0.63 | 0.43 |
| 8 | 1 | 1/3 | 0.0039 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.25 | 0.92 | 0.88 |
