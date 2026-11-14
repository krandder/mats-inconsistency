# EXPLORATORY (`PROTOCOL.md` §10.1): Search-Augmented Any-Order Perplexity & Minimum-Entropy Rollout (`expaomppl`)

Serves: a general searcher that minimises inconsistency over an any-order world model (`params_real` + internal lookahead clearing).

> **Honest `/experiments` Grade: `● No gain`** (EXPLORATORY `PROTOCOL.md` §10.1; spend `$0.31` ceiling on spot L4 `soup-gpu-expaomppl-1020172141-21893`; commit `ace128b`; dev splits only, `test_csv_touched = false`).
> **Honest Verdict:** **No gain on Sudoku-Extreme (`0%` boards solved; label-free search NLL `0.607` is worse than the zero-parameter `Control_PeerClue_Uniform` heuristic `0.571`).** While increasing test-time search budget $B = 0 \to 32$ reduces the network's internal random-order NLL slightly (`0.720 -> 0.607` label-free; `0.285 -> 0.251` supervised) and lifts $8\times 8$ N-Queens valid board rate (`68.1% -> 86.1%`), our small 1-pass any-order network fails to learn multi-hop Sudoku propagation and remains far from exact constraint closure (`2.86`–`4.39` unit collisions per board).

---

## 1. Sudoku-Extreme Development Split (`train.csv[5500:5628]`, 128 boards, 3-seed mean)

All numbers from [`explore_aom_search_ppl.json`](./explore_aom_search_ppl.json) under `summary.sudoku_extreme_dev_mean`:

| Arm | Labels Used | $B$ | $\beta_{\text{clear}}$ | `rand_order_nll` | `rand_order_ppl` | `cross_order_nll_std` | `min_h_nll` | `min_h_ppl` | `min_h_auto_empty_acc` | `min_h_auto_collisions` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Control_Random_Network` (Rule 9.1) | 0 | 16 | 2.5 | 2.24322 | 9.42372 | 0.92740 | 2.28008 | 9.77751 | 0.10952 | 246.31510 |
| `Control_Untrained_Init` | 0 | 0 | 0.0 | 2.21113 | 9.12631 | 0.28302 | 2.21021 | 9.11789 | 0.10542 | 269.36198 |
| `Control_ClueCopyFloor` | 0 | 0 | 0.0 | 2.19722 | 9.00000 | 0.00000 | 2.19722 | 9.00000 | 0.10948 | 493.92188 |
| `Control_PeerClue_Uniform` | 0 | 0 | 0.0 | 0.57133 | 1.77063 | 2.23693 | 0.34099 | 1.40634 | 0.36803 | 35.99219 |
| `Base_AnyOrder_NoSearch_B0` (Label-Free) | 0 | 0 | 0.0 | 0.71972 | 2.05513 | 6.14049 | 0.82484 | 2.28304 | 0.37638 | 7.28125 |
| `Ablation_Unweighted_Rollout_Average_B16` (Label-Free) | 0 | 16 | 0.0 | 0.70379 | 2.02176 | 5.76082 | 0.81587 | 2.26182 | 0.38689 | 7.22917 |
| `Augmented_AnyOrder_InconsistencySearch_B4` (Label-Free) | 0 | 4 | 2.5 | 0.65302 | 1.92222 | 6.22626 | 0.79338 | 2.21197 | 0.38920 | 5.45833 |
| `Augmented_AnyOrder_InconsistencySearch_B16` (Label-Free) | 0 | 16 | 2.5 | 0.61703 | 1.85414 | 6.33958 | 0.75738 | 2.13531 | 0.39547 | 4.75260 |
| `Augmented_AnyOrder_InconsistencySearch_B32` (Label-Free) | 0 | 32 | 2.5 | **0.60656** | **1.83488** | 6.49564 | **0.75864** | **2.13805** | **0.39599** | **4.39062** |
| `MaskedTraj_AnyOrder_NoSearch_B0` (OurTrain 450 steps) | 43,200 | 0 | 0.0 | 0.28538 | 1.33027 | 2.24486 | 0.21667 | 1.24195 | 0.46101 | 5.72656 |
| `MaskedTraj_Unweighted_Rollout_Average_B16` | 43,200 | 16 | 0.0 | 0.37195 | 1.45058 | 2.37224 | 0.30808 | 1.36084 | 0.44352 | 5.75260 |
| `MaskedTraj_InconsistencySearch_B4` | 43,200 | 4 | 2.5 | 0.26920 | 1.30892 | 2.53172 | 0.23071 | 1.25954 | 0.45205 | 3.15104 |
| `MaskedTraj_InconsistencySearch_B16` | 43,200 | 16 | 2.5 | 0.25605 | 1.29182 | 2.53339 | 0.22125 | 1.24764 | 0.47449 | **2.83333** |
| `MaskedTraj_InconsistencySearch_B32` | 43,200 | 32 | 2.5 | **0.25094** | **1.28525** | 2.47402 | **0.21022** | **1.23398** | **0.48199** | 2.85677 |

---

## 2. N-Queens Development Splits ($8\times 8$ OOD & $10\times 10$ ID, 48 boards, 3-seed mean)

All numbers from [`explore_aom_search_ppl.json`](./explore_aom_search_ppl.json) under `summary.nqueens_dev_mean`:

| Split | Arm | Labels Used | $B$ | $\beta_{\text{clear}}$ | `rand_order_nll` | `rand_order_ppl` | `cross_order_nll_std` | `min_h_nll` | `min_h_ppl` | `min_h_auto_collisions` | `min_h_valid_board_rate` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8x8_dev` | `Control_Random_Network` (Rule 9.1) | 0 | 16 | 2.5 | 2.07660 | 8.00289 | 0.40849 | 2.07935 | 8.02232 | 9.28472 | 0.00000 |
| `8x8_dev` | `Control_Untrained_Init` | 0 | 0 | 0.0 | 2.05276 | 7.79129 | 0.04156 | 2.05545 | 7.81238 | 7.72222 | 0.00000 |
| `8x8_dev` | `Control_ClueCopyFloor` | 0 | 0 | 0.0 | 2.07944 | 8.00000 | 0.00000 | 2.07944 | 8.00000 | 15.68750 | 0.00000 |
| `8x8_dev` | `Control_PeerClue_Uniform` | 0 | 0 | 0.0 | 0.48972 | 1.63187 | 0.54302 | 0.31132 | 1.36522 | 1.18750 | 0.43750 |
| `8x8_dev` | `Base_AnyOrder_NoSearch_B0` (Label-Free) | 0 | 0 | 0.0 | 0.58346 | 1.79468 | 1.99069 | 0.72293 | 2.07377 | 0.31944 | 0.68056 |
| `8x8_dev` | `Ablation_Unweighted_Rollout_Average_B16` | 0 | 16 | 0.0 | 0.44890 | 1.56686 | 0.74945 | 0.41893 | 1.52161 | 0.29167 | 0.72222 |
| `8x8_dev` | `Augmented_AnyOrder_InconsistencySearch_B4` | 0 | 4 | 2.5 | 0.32542 | 1.38568 | 1.29859 | 0.41914 | 1.52067 | 0.20833 | 0.79167 |
| `8x8_dev` | `Augmented_AnyOrder_InconsistencySearch_B16` | 0 | 16 | 2.5 | 0.26897 | 1.30945 | 0.97871 | 0.29208 | 1.34022 | 0.17361 | 0.83333 |
| `8x8_dev` | `Augmented_AnyOrder_InconsistencySearch_B32` | 0 | 32 | 2.5 | **0.23142** | **1.26153** | **0.88543** | **0.25261** | **1.28769** | **0.14583** | **0.86111** |
| `8x8_dev` | `MaskedTraj_AnyOrder_NoSearch_B0` | 28,800 | 0 | 0.0 | 0.30206 | 1.35311 | 0.91425 | 0.33122 | 1.39589 | 0.61111 | 0.65278 |
| `8x8_dev` | `MaskedTraj_InconsistencySearch_B16` | 28,800 | 16 | 2.5 | **0.19133** | **1.21106** | **0.64793** | **0.19911** | **1.22034** | **0.19444** | **0.83333** |
| `10x10_dev` | `Control_Random_Network` (Rule 9.1) | 0 | 16 | 2.5 | 2.28231 | 9.81010 | 0.57518 | 2.29737 | 9.95739 | 12.10417 | 0.00000 |
| `10x10_dev` | `Control_PeerClue_Uniform` | 0 | 0 | 0.0 | 0.67302 | 1.96017 | 0.64269 | 0.45565 | 1.57720 | 2.00000 | 0.20833 |
| `10x10_dev` | `Base_AnyOrder_NoSearch_B0` (Label-Free) | 0 | 0 | 0.0 | 1.27037 | 3.56301 | 4.56873 | 1.70295 | 5.54326 | 0.76389 | 0.29167 |
| `10x10_dev` | `Augmented_AnyOrder_InconsistencySearch_B4` | 0 | 4 | 2.5 | 0.88420 | 2.42787 | 3.46520 | 1.11105 | 3.04108 | 0.54861 | 0.45139 |
| `10x10_dev` | `Augmented_AnyOrder_InconsistencySearch_B16` | 0 | 16 | 2.5 | 0.80371 | 2.23403 | 3.37701 | 0.93208 | 2.54656 | 0.50694 | 0.50694 |
| `10x10_dev` | `Augmented_AnyOrder_InconsistencySearch_B32` | 0 | 32 | 2.5 | **0.78797** | **2.20198** | **3.31738** | **0.83301** | **2.30841** | **0.47222** | **0.52778** |
| `10x10_dev` | `MaskedTraj_AnyOrder_NoSearch_B0` | 28,800 | 0 | 0.0 | 0.38896 | 1.47608 | 1.50973 | 0.39046 | 1.47842 | 0.51389 | 0.61806 |
| `10x10_dev` | `MaskedTraj_InconsistencySearch_B16` | 28,800 | 16 | 2.5 | **0.32677** | **1.38716** | **1.46340** | 0.39801 | 1.49386 | **0.31944** | **0.72222** |

## 3. Compute & Cost Log (`GPU-LOG.md`)
- Attempt 1 (`soup-gpu-expaomppl-1020165844-31305`, spot L4): spot-preempted at ~19m (`rc=255`).
- Completed Run (`soup-gpu-expaomppl-1020172141-21893`, `us-central1-b` `g2-standard-8` L4): `age_h=0.36`, `cost_usd=0.31` (list-price ceiling), `rc=0`, auto-deleted at `2026-10-20T17:45Z`.
