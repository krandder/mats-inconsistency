# Scaling tables (scaling.py)

Cell: H(G) bracket [lower-upper] (lower includes the layer-1 Gaussian chain-rule bound) / E3 at 1% per-input change (nr = not reached) / E3 at mean-bias nMSE <= 1e-4 / D(1%) / n = noise-driven H(G|X) (plug-in).

## ARC He-ReLU, sigma = 0.01, depth 32

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 32 | 26-28 / nr / nr / D=32 / n=0.0 | 18-21 / nr / nr / D=32 / n=0.3 | 15-16 / nr / nr / D=32 / n=0.7 | 6.8 / 6.3 / 1.2 / D=32 / n=0.7 | 5.3 / 3.7 / 1.1 / D=32 / n=1.7 | 2.5 / 1.1 / 0.7 / D=32 / n=1.0 | 3.7 / -0.0 / 1.0 / D=32 / n=2.1 |
| 64 | 45-56 / nr / nr / D=64 / n=0.0 | 33-42 / nr / nr / D=64 / n=0.8 | 21-28 / nr / nr / D=64 / n=1.2 | 15-17 / nr / 8.9 / D=64 / n=1.6 | 12.3 / nr / 9.0 / D=64 / n=2.9 | 9.5 / nr / 4.4 / D=64 / n=3.6 | 8.8 / 7.3 / 5.6 / D=64 / n=4.2 |
| 128 | 87-110 / nr / nr / D=128 / n=0.0 | - | 19(H2>35*)-63 / nr / nr / D=128 / n=2.2 | 27-37 / nr / nr / D=128 / n=2.8 | 17-22 / nr / nr / D=128 / n=4.1 | 17-23 / nr / nr / D=128 / n=5.5 | 16-18 / nr / nr / D=128 / n=5.9 |
| 256 | 173-221 / nr / nr / D=256 / n=0.0 | - | 17(H2>31*)-129 / nr / nr / D=256 / n=4.1 | 17(H2>31*)-82 / nr / nr / D=256 / n=5.6 | 17(H2>31*)-52 / nr / nr / D=256 / n=6.9 | 30-43 / nr / nr / D=256 / n=7.5 | 28-40 / nr / nr / D=256 / n=7.9 |

## tanh chaotic (g = 3, LN, bias 0.5), sigma = 0.01, depth 32

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 32 | 37-43 / nr / -0.0 / D=64 / n=0.0 | 32-40 / nr / -0.0 / D=64 / n=0.7 | 25-37 / nr / -0.0 / D=64 / n=1.4 | 22-36 / nr / nr / D=64 / n=4.0 | 21-36 / nr / 7.9 / D=64 / n=7.3 | 20-34 / nr / nr / D=64 / n=7.9 | 20-35 / nr / nr / D=64 / n=8.0 |
| 64 | 71-86 / nr / -0.0 / D=128 / n=0.0 | 19(H2>35*)-80 / nr / -0.0 / D=128 / n=1.1 | 19(H2>35*)-76 / nr / -0.0 / D=128 / n=2.9 | 19(H2>35*)-75 / nr / -0.0 / D=128 / n=6.6 | 19(H2>35*)-74 / nr / 10.2 / D=128 / n=8.0 | 19(H2>35*)-74 / nr / nr / D=128 / n=8.0 | 19(H2>35*)-73 / nr / nr / D=128 / n=8.0 |
| 128 | 143-172 / nr / nr / D=256 / n=0.0 | - | 17(H2>31*)-154 / nr / nr / D=256 / n=5.3 | 17(H2>31*)-151 / nr / nr / D=256 / n=8.0 | 17(H2>31*)-152 / nr / nr / D=256 / n=8.0 | 17(H2>31*)-151 / nr / nr / D=256 / n=8.0 | 17(H2>31*)-150 / nr / nr / D=256 / n=8.0 |
| 256 | 285-367 / nr / nr / D=512 / n=0.0 | - | 17(H2>31*)-328 / nr / nr / D=512 / n=7.6 | 17(H2>31*)-320 / nr / nr / D=512 / n=8.0 | 17(H2>31*)-317 / nr / nr / D=512 / n=8.0 | 17(H2>31*)-318 / nr / nr / D=512 / n=8.0 | 17(H2>31*)-321 / nr / nr / D=512 / n=8.0 |

## tanh ordered (g = 0.25), sigma = 0.01

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | 11.8 / -0.0 / -0.0 / D=0 / n=0.0 | 8.9 / -0.0 / -0.0 / D=0 / n=0.2 | 1.0 / -0.0 / -0.0 / D=0 / n=0.1 | 0.9 / -0.0 / -0.0 / D=0 / n=0.8 | 0.0 / -0.0 / -0.0 / D=0 / n=0.0 | 0.8 / -0.0 / -0.0 / D=37 / n=0.8 | 0.8 / 0.8 / 0.8 / D=128 / n=0.8 |
| 256 | pending | pending | pending | pending | pending | pending | pending |

## sigma = 0.001: ARC (64, 256), tanh chaotic and ordered (64)

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | 45-56 / nr / nr / D=64 / n=0.0 | 33-42 / nr / nr / D=64 / n=0.1 | 21-28 / nr / nr / D=64 / n=0.1 | 15-17 / nr / nr / D=64 / n=0.2 | 12.1 / nr / 7.7 / D=64 / n=0.3 | 9.0 / 8.0 / 4.4 / D=64 / n=0.5 | 8.3 / 6.0 / 4.6 / D=64 / n=0.6 |
| 256 | pending | pending | pending | pending | pending | pending | pending |

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | 71-86 / nr / -0.0 / D=128 / n=0.0 | 19(H2>35*)-80 / nr / -0.0 / D=128 / n=0.1 | 19(H2>35*)-76 / nr / -0.0 / D=128 / n=0.3 | 19(H2>35*)-75 / nr / -0.0 / D=128 / n=1.3 | 19(H2>35*)-74 / nr / 4.8 / D=128 / n=6.4 | 19(H2>35*)-74 / nr / nr / D=128 / n=8.0 | 19(H2>35*)-73 / nr / nr / D=128 / n=8.0 |

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | pending | pending | pending | pending | pending | pending | pending |

## sigma = 0.1: ARC (64, 256), tanh chaotic and ordered (64)

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | 45-56 / nr / nr / D=64 / n=0.0 | 33-42 / nr / nr / D=64 / n=5.3 | 22-29 / nr / nr / D=64 / n=7.0 | 17-20 / nr / 5.5 / D=64 / n=7.5 | 17-20 / nr / nr / D=64 / n=8.0 | 18-22 / nr / nr / D=64 / n=8.0 | 16-18 / nr / nr / D=64 / n=8.0 |
| 256 | pending | pending | pending | pending | pending | pending | pending |

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | pending | pending | pending | pending | pending | pending | pending |

| width | layer 1 | layer 2 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|---|
| 64 | pending | pending | pending | pending | pending | pending | pending |

## RNNs (per step): H(G_t) / E3(1%) / D / noise-driven

T16: t1: 1.0 / 1.0 / 1.0 | t2: 4.1 / 2.4 / 2.4 | t3: 4.7 / 2.8 / 3.7 | t4: 5.3 / 3.6 / 4.7 | t5: 5.7 / 4.1 / 4.9 | t6: 6.0 / 4.0 / 5.8 | t7: 5.8 / 3.8 / 5.7 | t8: 5.3 / 3.5 / 4.9 | t9: 4.8 / 3.4 / 4.7 | t10: 4.4 / 2.3 / 3.6

W32s0: t1: 1.0 / 1.0 / 1.0 | t2: 6.8 / 2.4 / 2.4 | t3: 7.7 / 3.1 / 4.3 | t4: 8.5 / 4.1 / 6.6 | t5: 8.9 / 5.1 / nr | t6: 9.2 / 5.8 / nr | t7: 9.6 / 7.1 / nr | t8: 9.4 / 7.2 / nr | t9: 9.6 / nr / nr | t10: 9.5 / nr / nr

W64s0: t1: 1.0 / 1.0 / 1.0 | t2: 8.9 / 2.5 / 3.6 | t3: 9.8 / 4.9 / 8.0 | t4: 9.7 / 6.1 / nr | t5: 10.4 / 7.0 / nr | t6: 10.6 / 7.4 / nr | t7: 10.9-11.9 / nr / nr | t8: 10.9-12.2 / nr / nr | t9: 11.1-12.3 / nr / nr | t10: 11.4-12.8 / nr / nr

W32s1: t1: 1.0 / 1.0 / 1.0 | t2: 6.7 / 2.4 / 2.6 | t3: 7.7 / 4.5 / 4.5 | t4: 8.6 / 4.9 / 6.2 | t5: 9.2 / 5.5 / nr | t6: 9.6 / 6.1 / nr | t7: 10.1 / 7.3 / nr | t8: 10.5 / 9.4 / nr | t9: 10.8 / nr / nr | t10: 10.6 / nr / nr

W64s1: t1: 1.0 / 1.0 / 1.0 | t2: 9.3 / 2.7 / 3.4 | t3: 10.3 / 4.4 / 7.8 | t4: 10.5 / 5.6 / nr | t5: 10.9 / 7.0 / nr | t6: 11.2 / 7.0 / nr | t7: 11.4 / 8.1 / nr | t8: 11.8-12.9 / nr / nr | t9: 12.2-13.4 / nr / nr | t10: 12.5-13.9 / nr / nr


## Heuristic Fano lower bound on E3(1%) (assumes rho = min relevant single-flip rate holds for every substitution)

arc32_s0.01: l1: H_lo=25.6; min-f rho=0.49 -> 24.7; tree rho=0.93 -> 25.1; l2: H_lo=18.0; min-f rho=0.39 -> 17.0; tree rho=0.81 -> 17.5; l4: H_lo=14.7; min-f rho=0.31 -> 13.5; tree rho=0.44 -> 13.8; l8: H_lo=6.8; min-f rho=0.18 -> 4.8; tree rho=0.10 -> 3.0; l16: H_lo=5.3; min-f rho=0.03 -> -6.4; tree rho=0.02 -> -10.8; l24: H_lo=2.5; min-f rho=0.11 -> -1.0; l32: H_lo=3.7; min-f rho=0.03 -> -7.6; tree rho=0.01 -> -28.3

arc64_s0.01: l1: H_lo=44.6; min-f rho=0.69 -> 43.6; tree rho=1.00 -> 43.9; l2: H_lo=32.9; min-f rho=0.67 -> 31.8; tree rho=1.00 -> 32.2; l4: H_lo=20.7; min-f rho=0.58 -> 19.5; tree rho=0.96 -> 20.0; l8: H_lo=14.9; min-f rho=0.37 -> 13.0; l16: H_lo=12.3; min-f rho=0.32 -> 10.0; l24: H_lo=9.5; min-f rho=0.07 -> -0.8; l32: H_lo=8.8; min-f rho=0.06 -> -2.6

arc128_s0.01: l1: H_lo=86.8; min-f rho=0.68 -> 84.8; tree rho=1.00 -> 85.5; l4: H_lo=19.0; min-f rho=0.54 -> 16.5; tree rho=1.00 -> 17.6; l8: H_lo=27.0; min-f rho=0.44 -> 23.9; tree rho=0.99 -> 25.6; l16: H_lo=17.4; min-f rho=0.25 -> 12.0; l24: H_lo=17.0; min-f rho=0.07 -> -3.0; l32: H_lo=15.6; min-f rho=0.09 -> 1.6

arc256_s0.01: l1: H_lo=173.4; min-f rho=0.73 -> 169.8; tree rho=1.00 -> 170.7; l4: H_lo=17.0; min-f rho=0.58 -> 12.4; tree rho=1.00 -> 14.4; l8: H_lo=17.0; min-f rho=0.41 -> 10.6; tree rho=1.00 -> 14.4; l16: H_lo=17.0; min-f rho=0.21 -> 4.4; tree rho=1.00 -> 14.4; l24: H_lo=29.8; min-f rho=0.11 -> 6.8; tree rho=0.98 -> 27.1; l32: H_lo=27.7; min-f rho=0.07 -> -12.1; tree rho=0.92 -> 24.8

tanhC32_s0.01: l1: H_lo=37.0; min-f rho=1.00 -> 36.3; tree rho=1.00 -> 36.3; l2: H_lo=31.5; min-f rho=1.00 -> 30.8; tree rho=1.00 -> 30.8; l4: H_lo=25.4; min-f rho=1.00 -> 24.7; tree rho=1.00 -> 24.7; l8: H_lo=22.3; min-f rho=1.00 -> 21.6; tree rho=1.00 -> 21.6; l16: H_lo=21.2; min-f rho=0.99 -> 20.4; tree rho=1.00 -> 20.5; l24: H_lo=20.4; min-f rho=0.96 -> 19.7; tree rho=1.00 -> 19.7; l32: H_lo=19.7; min-f rho=0.78 -> 18.8; tree rho=0.99 -> 19.0

tanhC64_s0.01: l1: H_lo=70.8; min-f rho=1.00 -> 69.4; tree rho=1.00 -> 69.4; l2: H_lo=19.0; min-f rho=1.00 -> 17.6; tree rho=1.00 -> 17.6; l4: H_lo=19.0; min-f rho=1.00 -> 17.6; tree rho=1.00 -> 17.6; l8: H_lo=19.0; min-f rho=1.00 -> 17.6; tree rho=1.00 -> 17.6; l16: H_lo=19.0; min-f rho=1.00 -> 17.6; tree rho=1.00 -> 17.6; l24: H_lo=19.0; min-f rho=0.97 -> 17.6; tree rho=1.00 -> 17.6; l32: H_lo=19.0; min-f rho=0.73 -> 17.1; tree rho=1.00 -> 17.6

tanhC128_s0.01: l1: H_lo=142.9; min-f rho=1.00 -> 140.3; tree rho=1.00 -> 140.3; l4: H_lo=17.0; min-f rho=1.00 -> 14.4; tree rho=1.00 -> 14.4; l8: H_lo=17.0; min-f rho=1.00 -> 14.4; tree rho=1.00 -> 14.4; l16: H_lo=17.0; min-f rho=0.99 -> 14.3; tree rho=1.00 -> 14.4; l24: H_lo=17.0; min-f rho=0.97 -> 14.3; tree rho=1.00 -> 14.4; l32: H_lo=17.0; min-f rho=0.66 -> 13.0; tree rho=1.00 -> 14.4

tanhC256_s0.01: l1: H_lo=285.1; min-f rho=1.00 -> 279.9; tree rho=1.00 -> 279.9; l4: H_lo=17.0; min-f rho=1.00 -> 11.8; tree rho=1.00 -> 11.8; l8: H_lo=17.0; min-f rho=1.00 -> 11.8; tree rho=1.00 -> 11.8; l16: H_lo=17.0; min-f rho=0.99 -> 11.8; tree rho=1.00 -> 11.8; l24: H_lo=17.0; min-f rho=0.96 -> 11.6; tree rho=1.00 -> 11.8; l32: H_lo=17.0; min-f rho=0.56 -> 7.8; tree rho=1.00 -> 11.8

