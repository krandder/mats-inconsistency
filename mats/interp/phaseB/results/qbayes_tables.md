### E3a: predictive head = MLP(exact Bayes quantities); K = 64 R² mean / log-sd / KL on 2,000 test episodes

| width h (parameters) | 2L seed 4 | 2L seed 1 | 2L seed 2 | 3L seed 13 |
|---|---|---|---|---|
| 0 (300) | −0.08 / 0.18 / 0.04 (scen 5/9) | −0.01 / 0.14 / 0.01 (scen 5/9) | 0.00 / 0.14 / 0.01 (scen 5/9) | −0.06 / 0.12 / −0.10 (scen 6/9) |
| 8 (380) | 0.01 / 0.20 / 0.09 (scen 5/9) | 0.05 / 0.16 / 0.02 (scen 5/9) | 0.04 / 0.16 / 0.01 (scen 5/9) | −0.00 / 0.23 / −0.08 (scen 6/9) |
| 16 (876) | 0.01 / 0.23 / 0.10 (scen 5/9) | 0.04 / 0.17 / 0.02 (scen 5/9) | 0.02 / 0.18 / 0.01 (scen 5/9) | 0.01 / 0.23 / −0.09 (scen 6/9) |
| 32 (2252) | 0.02 / 0.29 / 0.09 (scen 5/9) | 0.04 / 0.21 / 0.03 (scen 5/9) | 0.03 / 0.20 / −0.00 (scen 5/9) | −0.00 / 0.27 / −0.10 (scen 6/9) |
| 64 (6540) | 0.03 / 0.34 / 0.10 (scen 5/9) | 0.05 / 0.26 / 0.03 (scen 5/9) | 0.03 / 0.23 / 0.01 (scen 5/9) | −0.01 / 0.30 / −0.13 (scen 6/9) |
| floor | 0.960 / 0.987 / 0.979 | 0.965 / 0.987 / 0.988 | 0.982 / 0.989 / 0.989 | 0.948 / 0.983 / 0.987 |

### Scenario outcomes (hit = the pre-registered pattern is present): network vs best-R² head

| model | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---|---|---|---|---|---|---|---|---|
| 2L seed 4 network | hit | – | hit | – | – | hit | – | hit | hit |
| 2L seed 4 head h=64 | – | – | – | – | hit | – | – | hit | hit |
| 2L seed 1 network | hit | – | hit | – | – | hit | – | hit | hit |
| 2L seed 1 head h=8 | – | – | – | – | hit | – | – | hit | hit |
| 2L seed 2 network | – | – | hit | – | – | hit | hit | hit | hit |
| 2L seed 2 head h=8 | – | – | – | – | hit | – | – | hit | hit |
| 3L seed 13 network | hit | – | hit | – | hit | hit | – | hit | hit |
| 3L seed 13 head h=16 | – | – | – | – | hit | – | – | hit | hit |