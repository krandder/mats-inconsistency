Task t2, split val. KL x1e3 = single-cell KL to the exact posterior (seed mean; reported, never used to select). dLL = seed-paired outcome log-loss difference vs B-rbtd8 x1e4 (negative = better than rbtd8). S_out = mean over N of dLL / (k0's log-loss gap to rbtd8) (selection score; 0 = rbtd8, -1 = twice rbtd8's gain over k0, +1 = k0).

| tag | eligible | N=100: KL x1e3 [dLL x1e4] | N=300: KL x1e3 [dLL x1e4] | N=1000: KL x1e3 [dLL x1e4] | N=3000: KL x1e3 [dLL x1e4] | N=10000: KL x1e3 [dLL x1e4] | S_out | KL ratio vs rbtd8 (geo, 5 N) | FLOPs/board @1k |
|---|---|---|---|---|---|---|---|---|---|
| B-rbtd8 | baseline | 10.82 [+0.00] (3s) | 4.60 [+0.00] (3s) | 2.03 [+0.00] (3s) | 1.08 [+0.00] (3s) | 0.55 [+0.00] (3s) | 0.000 | 1.00 | 8e+11 |
| WMs-D-only | yes | 13.99 [+31.73] (3s) | 6.80 [+19.76] (3s) | 1.85 [-1.46] (3s) | 1.05 [-0.29] (3s) | 0.97 [+3.60] (3s) | 0.334 | 1.21 | 3.5e+11 |
| B-k0 | baseline | 15.34 [+46.09] (3s) | 8.53 [+39.93] (3s) | 4.11 [+22.06] (3s) | 2.30 [+12.81] (3s) | 1.12 [+6.26] (3s) | 1.000 | 1.88 | 2.5e+10 |
| WMsb-D-only | not L0 | 16.84 [+59.46] (3s) | 15.12 [+103.78] (3s) | 14.89 [+127.86] (3s) | 14.64 [+135.18] (3s) | 14.58 [+139.18] (3s) | 8.497 | 6.74 | 1.9e+11 |

Gap k0 - rbtd8 in outcome log-loss x1e4 per N (the S_out scale): 100: 46.09, 300: 39.93, 1000: 22.06, 3000: 12.81, 10000: 6.26
* = partial sweep (not all 5 N with 3 paired seeds).

Champion by the pre-registered rule (lowest S_out among eligible): **WMs-D-only**
