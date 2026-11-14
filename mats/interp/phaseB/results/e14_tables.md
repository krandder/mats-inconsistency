### E14.1 Held-out R² of layer 1's encoding strength g from the source's four labelled errors (fit: 4,000 test episodes; held out: 8,000 fresh episodes; group mean [range])

| form | group A | group B |
|---|---|---|
| sum of squares (Bayes) | 0.89 [0.87, 0.90] | 0.51 [0.45, 0.71] |
| sum of absolute errors | 0.88 [0.86, 0.89] | 0.52 [0.46, 0.71] |
| largest error | 0.86 [0.83, 0.87] | 0.47 [0.42, 0.66] |
| count above a threshold tau | 0.81 [0.79, 0.82] | 0.48 [0.43, 0.64] |
| learned symmetric function (ceiling for error-only forms) | 0.90 [0.89, 0.91] | 0.53 [0.47, 0.71] |
| errors + other inputs (HGB; not error-only) | 0.93 [0.91, 0.93] | 0.67 [0.62, 0.82] |

Best named form: A SS (fits well: True), B SA (fits well: True). Fitted tau: A [0.8, 0.8, 0.8, 0.8, 0.8, 0.8], B [0.8, 0.8, 0.8, 0.8, 0.9, 0.8].

### E14.2 Predicted Δg on the E12 pairs (big miss − spread) vs E13's measured Δg (group means)

| form | A: large / middle predicted | A nets with large > 0 and middle < 0 | A: corr with measured | B: large / middle predicted | B: corr |
|---|---|---|---|---|---|
| sum of squares (Bayes) | -0.000 / -0.000 | 0/6 | nan | -0.000 / -0.000 | nan |
| sum of absolute errors | +0.355 / +0.690 | 0/6 | −0.65 | +0.395 / +0.653 | −0.08 |
| largest error | -1.000 / -1.791 | 0/6 | 0.62 | -1.032 / -1.504 | −0.03 |
| count above a threshold tau | +1.706 / -1.219 | 6/6 | 0.46 | +2.091 / -0.815 | 0.27 |
| learned symmetric function (ceiling for error-only forms) | +0.537 / -0.256 | 6/6 | 0.71 | +1.225 / +0.327 | 0.34 |
| measured (E13) | +0.619 / -0.078 | | | +1.170 / +0.489 | |

Same function in A and B (learned symmetric function fitted on a group's average g): A-fit R² within A 0.89, on B 0.04; B-fit within B 0.51, on A 0.19; same: False.
