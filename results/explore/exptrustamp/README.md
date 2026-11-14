# EXPLORATORY (`PROTOCOL.md` §10.1): Three-Parameter Separation — `params_real` (`theta_base + theta_trust`), `params_search`, and `params_amp` (`exptrustamp`)

Serves: auditable belief clearing where confiability ($\theta_{\text{trust}} \in \text{params}_{\text{real}}$) is trained from search into reality without self-gaming.

> **Honest `/experiments` Grade: `● No gain` (Diagnostic negative control confirms `M3` self-gaming blowup, but `M5` 1-pass distillation yields zero NLL improvement `0.282 -> 0.281` and `0%` Sudoku solve rate)** (EXPLORATORY `PROTOCOL.md` §10.1; spend `$0.26` ceiling; commit `d467287`; dev splits only, `test_csv_touched = false`).
> **Honest Verdict:** **No gain on amplification distillation; informative only as a negative-control demonstration.** Putting `theta_trust` inside `params_search` to minimize internal inconsistency (`M3_Search_SelfGamed_Trust_B16`) causes severe overconfidence (`ECE` `0.101 -> 0.446`, `PPL` `1.326 -> 1.803`), whereas training `theta_trust` against real labels (`M4`) avoids the blowup (`PPL` `1.285`). However, 1-pass distillation into `params_amp` (`M5`) leaves random-order NLL unchanged (`0.2821 -> 0.2812` on Sudoku) and slightly degrades N-Queens (`0.419 -> 0.451`), and `0%` of Sudoku boards are solved.

---

## 1. Sudoku-Extreme Development Split (`train.csv[5500:5628]`, 128 boards, 3-seed mean)

All numbers from [`explore_trust_and_amp_split.json`](./explore_trust_and_amp_split.json) under `summary.sudoku_extreme_dev_mean`:

| Condition | $B$ | `rand_order_nll_per_cell` | `rand_order_ppl` | `cross_order_nll_std` | `min_h_voi_ppl` | `autonomous_empty_acc` | `autonomous_unit_collisions` | `ece_calibration_error` | `brier_per_cell` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_Control_Random_Net` (Rule 9.1) | 16 | 2.24505 | 9.44093 | 1.00701 | 9.81803 | 0.10863 | 169.73438 | 0.04584 | 0.89596 |
| `C1_Control_PeerClue_Uniform` | 0 | 0.57133 | 1.77063 | 2.23693 | 1.40634 | 0.36803 | 35.99219 | 0.01089 | 0.72304 |
| `M1_ParamsReal_Base_B0` | 0 | 0.28211 | 1.32593 | 2.24690 | 1.24135 | 0.44823 | 5.56771 | 0.10062 | 0.62001 |
| `M2_Search_Unweighted_Clear_B16` | 16 | 0.36954 | 1.44708 | 2.42272 | 1.34725 | 0.45856 | 5.76823 | 0.06990 | 0.62373 |
| `M3_Search_SelfGamed_Trust_B16` | 16 | 0.58877 | 1.80273 | 9.32059 | 1.90220 | 0.47572 | 3.15885 | 0.44612 | 0.95069 |
| `M4_Search_RealityCalibrated_Trust_B16` | 16 | **0.25112** | **1.28546** | 2.59679 | 1.24005 | **0.49137** | 2.68229 | 0.14406 | 0.64094 |
| `M5_ParamsAmp_Distilled_B0` | 0 | 0.28123 | 1.32476 | **1.98885** | **1.22923** | 0.45743 | 5.54167 | **0.08028** | **0.60927** |
| `M6_ParamsAmp_Plus_Search_B16` | 16 | 0.25225 | 1.28693 | 2.52269 | 1.24120 | 0.48388 | **2.65104** | 0.14183 | 0.64145 |

### Sudoku Search Budget Scaling ($B \in \{4, 16, 32\}$ under `M4_Search_RealityCalibrated_Trust`)

All numbers from [`explore_trust_and_amp_split.json`](./explore_trust_and_amp_split.json) under `summary.sudoku_b_scaling_m4_mean`:

| Budget | `rand_order_nll_per_cell` | `rand_order_ppl` | `cross_order_nll_std` | `min_h_voi_ppl` | `autonomous_empty_acc` | `autonomous_unit_collisions` | `ece_calibration_error` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4` | 0.26390 | 1.30200 | 2.55254 | 1.25478 | 0.46907 | 2.96094 | 0.13635 |
| `B16` | 0.25112 | 1.28546 | 2.59679 | **1.24005** | **0.49137** | **2.68229** | 0.14406 |
| `B32` | **0.24598** | **1.27888** | 2.62222 | 1.24140 | 0.49081 | 2.72135 | 0.14780 |

---

## 2. N-Queens Development Splits ($10\times 10$ ID & $8\times 8$ OOD, 48 boards, 3-seed mean)

All numbers from [`explore_trust_and_amp_split.json`](./explore_trust_and_amp_split.json) under `summary.nqueens_dev_mean`:

| Split | Condition | $B$ | `rand_order_nll_per_cell` | `rand_order_ppl` | `cross_order_nll_std` | `min_h_voi_ppl` | `autonomous_empty_acc` | `autonomous_valid_board_rate` | `autonomous_collisions` | `ece_calibration_error` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `10x10_dev` | `C0_Control_Random_Net` (Rule 9.1) | 16 | 2.28274 | 9.81375 | 0.59329 | 9.79731 | 0.14917 | 0.00000 | 10.18750 | 0.02624 |
| `10x10_dev` | `C1_Control_PeerClue_Uniform` | 0 | 0.67302 | 1.96017 | 0.64269 | 1.57720 | 0.26796 | 0.20833 | 2.00000 | 0.01891 |
| `10x10_dev` | `M1_ParamsReal_Base_B0` | 0 | 0.41930 | 1.52114 | 1.53086 | 1.62880 | 0.51381 | **0.52778** | 0.59722 | 0.17244 |
| `10x10_dev` | `M2_Search_Unweighted_Clear_B16` | 16 | 0.60895 | 1.83893 | 1.26718 | 1.87659 | 0.47422 | 0.52083 | **0.55556** | 0.05257 |
| `10x10_dev` | `M3_Search_SelfGamed_Trust_B16` | 16 | 0.90170 | 2.46769 | 4.19274 | 3.60328 | 0.55341 | 0.51389 | 0.57639 | 0.32715 |
| `10x10_dev` | `M4_Search_RealityCalibrated_Trust_B16` | 16 | **0.38881** | **1.47645** | **1.08619** | **1.56621** | 0.57182 | 0.47917 | 0.78472 | **0.06692** |
| `10x10_dev` | `M5_ParamsAmp_Distilled_B0` | 0 | 0.45075 | 1.56980 | 1.73392 | 1.76763 | 0.48066 | 0.49306 | 0.56250 | 0.23799 |
| `10x10_dev` | `M6_ParamsAmp_Plus_Search_B16` | 16 | 0.41335 | 1.51371 | 1.18559 | 1.57846 | **0.59208** | 0.50694 | 0.67361 | 0.06962 |
| `8x8_dev` | `C0_Control_Random_Net` (Rule 9.1) | 16 | 2.07381 | 7.98329 | 0.45167 | 7.90883 | 0.17358 | 0.00000 | 8.09028 | 0.05108 |
| `8x8_dev` | `C1_Control_PeerClue_Uniform` | 0 | 0.48972 | 1.63187 | 0.54302 | 1.36522 | 0.40000 | 0.43750 | 1.18750 | 0.01616 |
| `8x8_dev` | `M1_ParamsReal_Base_B0` | 0 | **0.28639** | **1.33229** | 0.87448 | **1.40950** | 0.68553 | 0.68750 | 0.38194 | 0.13737 |
| `8x8_dev` | `M2_Search_Unweighted_Clear_B16` | 16 | 0.50502 | 1.65730 | 0.68455 | 1.59676 | 0.64528 | 0.65278 | 0.43056 | 0.10135 |
| `8x8_dev` | `M3_Search_SelfGamed_Trust_B16` | 16 | 0.59879 | 1.82488 | 2.43662 | 2.31509 | **0.72830** | **0.70833** | **0.34722** | 0.23259 |
| `8x8_dev` | `M4_Search_RealityCalibrated_Trust_B16` | 16 | 0.33268 | 1.39647 | 0.64293 | 1.42683 | 0.70314 | 0.65278 | 0.49306 | **0.09303** |
| `8x8_dev` | `M5_ParamsAmp_Distilled_B0` | 0 | 0.30820 | 1.36177 | 1.03687 | 1.44147 | 0.67421 | 0.68056 | 0.36111 | 0.16347 |
| `8x8_dev` | `M6_ParamsAmp_Plus_Search_B16` | 16 | 0.34283 | 1.41160 | **0.61060** | 1.41482 | 0.69686 | 0.58333 | 0.54167 | 0.10334 |

---

## 3. Compute & Cost Log (`GPU-LOG.md`)
- Attempt 1 (`soup-gpu-exptrustamp-1020182643-22012`, `us-central1-a` `g2-standard-8` SPOT L4): failed immediately (`rc=1`, missing venv `numpy`/`torch` bootstrap), `age_h=0.04`, `cost_usd=0.03`, auto-deleted at `2026-10-20T18:29Z`.
- Attempt 2 (`soup-gpu-exptrustamp-1020183040-5264`, `STANDARD` L4): `CREATE-FAILED` due to `ZONE_RESOURCE_POOL_EXHAUSTED` in `us-central1-a/b/c` ($0.00).
- Completed Run (`soup-gpu-exptrustamp-1020183155-15196`, `us-central1-a` `g2-standard-8` SPOT L4): `age_h=0.27`, `cost_usd=0.23` (list-price ceiling), `rc=0`, auto-deleted at `2026-10-20T18:49Z`.
- Total group spend (`explore-trustamp`): `$0.26` list-price ceiling (cumulative project spend `$435.52` in `GPU-LOG.md`).
