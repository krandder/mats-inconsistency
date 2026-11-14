| seed | quantity | d0.json | independent CPU recomputation | abs diff |
|---|---|---|---|---|
| 101 | 0-step KL to P*_full | 0.023062 | 0.023062 | 2.9e-09 |
| 101 | 5-known NLL | 0.000161 | 0.000161 | 2.4e-09 |
| 101 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 101 | battles scored | 9744 | 9744 | |
| 102 | 0-step KL to P*_full | 0.025361 | 0.025361 | 7.8e-09 |
| 102 | 5-known NLL | 0.000082 | 0.000082 | 3.5e-09 |
| 102 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 102 | battles scored | 9744 | 9744 | |
| 103 | 0-step KL to P*_full | 0.023247 | 0.023247 | 4.5e-09 |
| 103 | 5-known NLL | 0.000071 | 0.000071 | 2.4e-09 |
| 103 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 103 | battles scored | 9744 | 9744 | |

Largest absolute difference: 7.8e-09 (GPU float32 evaluation vs CPU recomputation with float64 softmax).
