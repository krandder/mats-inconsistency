KL x 1e3 on TEST (mean over seeds, 95% t-interval half-width; seeds in brackets):

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| F-k0 | 7.116 +- 0.642 [5] | 4.222 +- 0.746 [5] | 1.455 +- 0.188 [5] | 0.746 +- 0.051 [5] | 0.409 +- 0.055 [5] |
| F-k0cell | 7.083 +- 0.595 [5] | 3.570 +- 0.835 [5] | 1.261 +- 0.121 [5] | 0.668 +- 0.057 [5] | 0.410 +- 0.098 [5] |
| F-E17-rbtd8 | 4.942 +- 0.929 [5] | 1.864 +- 0.430 [5] | 0.642 +- 0.145 [5] | 0.395 +- 0.103 [5] | 0.202 +- 0.027 [5] |

Paired difference vs F-k0 (KL x 1e3, arm minus base, same seeds = same training boards; 95% t-interval) and geometric-mean ratio:

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| F-k0cell | -0.032 +- 0.090 (x1.00) | -0.652 +- 0.952 (x0.84) | -0.194 +- 0.240 (x0.87) | -0.078 +- 0.078 (x0.89) | +0.001 +- 0.097 (x0.99) |
| F-E17-rbtd8 | -2.174 +- 0.970 (x0.69) | -2.358 +- 0.541 (x0.44) | -0.814 +- 0.174 (x0.44) | -0.351 +- 0.094 (x0.52) | -0.207 +- 0.066 (x0.49) |

KL x 1e3 per stage at N=1000 (test; stage = probes seen 0 / 2 / 4 / 6):

| arm | 0 | 2 | 4 | 6 | mean |
|---|---|---|---|---|---|
| F-k0 | 0.145 | 0.990 | 1.828 | 2.858 | 1.455 |
| F-k0cell | 0.168 | 0.815 | 1.552 | 2.510 | 1.261 |
| F-E17-rbtd8 | 0.082 | 0.426 | 0.809 | 1.249 | 0.642 |

Secondary metrics at N=1000 (test; seed means): outcome log-loss per cell, ECE, joint log-lik of the true board per board-stage (independent / net / autoregressive), training FLOPs per board to the best checkpoint, steps to the best checkpoint.

| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |
|---|---|---|---|---|---|---|
| F-k0 | 0.56359 | 0.0063 | -14.09 |  | 2.72e+10 | 1290 |
| F-k0cell | 0.56342 | 0.0061 | -14.09 |  | 2.88e+11 | 2730 |
| F-E17-rbtd8 | 0.56274 | 0.0048 | -14.07 | -9.38 | 6.08e+11 | 3600 |

Exact posterior joint log-lik of the true board: -8.84 per board-stage (the floor any joint model can reach).

Secondary metrics at N=10000 (test; seed means): outcome log-loss per cell, ECE, joint log-lik of the true board per board-stage (independent / net / autoregressive), training FLOPs per board to the best checkpoint, steps to the best checkpoint.

| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |
|---|---|---|---|---|---|---|
| F-k0 | 0.56247 | 0.0029 | -14.06 |  | 2.32e+10 | 11000 |
| F-k0cell | 0.56246 | 0.0026 | -14.06 |  | 2.24e+11 | 21240 |
| F-E17-rbtd8 | 0.56228 | 0.0033 | -14.06 | -8.91 | 3.32e+11 | 19640 |

Exact posterior joint log-lik of the true board: -8.84 per board-stage (the floor any joint model can reach).

N = 30000 (3 seeds; the 'eventually more accurate' check), TEST KL x 1e3:

| arm | KL | paired vs F-k0 |
|---|---|---|
| F-k0 | 0.268 +- 0.065 [3] |  |
| F-E17-rbtd8 | 0.158 +- 0.010 [3] | -0.110 +- 0.071 (x0.59) |
