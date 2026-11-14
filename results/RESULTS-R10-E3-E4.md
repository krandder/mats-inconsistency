> **Independent audit (2026-10-19): verdict artefact.** The claims in this file did not pass an independent Opus 5.5 audit; see `docs/audits/gemini-campaign-critique-2026-10-19.md` (all rounds), `docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md` and `docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md`. Numbers below are the runner's outputs, not evidence about the architecture.

# Stage-1 Pre-Registered Results Report: Studies `R10-E3` & `R10-E4`

- **Pre-Registration Protocol:** `benchmarks/PREREG-STAGE1-R10.md` (`PREREG-STAGE1-R10`)
- **Benchmark Runner:** `benchmarks/run_r10_e3_e4.py`
- **Raw Metrics Artifact:** `results/r10_e3_e4_results.json`
- **Generated At (UTC):** `2026-10-18T02:47:40Z`
- **Seeds:** `[101, 102, 103]` (`df = 2`, paired Student-$t$ critical value $t_{0.975,2} = 4.303$)
- **Budgets Evaluated (WTEs):** `[0, 1, 2, 4, 8, 16, 24, 32, 64, 128]`

---

## 1. Executive Summary & Verbatim Pre-Registered Verdicts

| Rule ID | Study | Pre-Registered Pass Bar Summary | Key Empirical Result (3-Seed Mean & 95% CI) | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **`R10-E3-D1`** | `R10-E3` (`sudoku_9x9_extreme`) | `AUROC(m_ql) >= 0.90` & `>= AUROC(m_qs) + 0.15`; invalid false-accept `<= 0.01`; `S6` breaks `B=16->32` stall (`Delta_16->32 >= +0.08`, `CI_low > 0`); `S6 > S4` & `S6 > S2` at `B=32`; non-greedy share `>= 0.25` | `AUROC(m_ql) = 1.0000` vs `AUROC(m_qs) = 0.0889`; false-accept `= 0.0000`; `Delta_16->32(S6 - S4) = +0.0967` (`CI [0.0737, 0.1196]`); `Wit(S6,32) = 0.5447` vs `S4 = 0.4180`; non-greedy share `= 0.6604` | **PASS** |
| **`R10-E3-D2`** | `R10-E3` (`sudoku_9x9_extreme` & `shiphunt_8x8`) | `f_root(128) in [0.03, 0.35]`, `f_tip(128) in [0.25, 0.85]`, `>=6` stalls on `>90%` boards; `ShipHunt` `Cov_24(S6) >= 0.985` & `> Cov_24(S5)`; `JSD` cut `>= 6.0x`; plan-swap flip `>= 0.75`; `<p_r, k^edge> > 0.50` at `d<=2` & `d>=6` | `f_root(128) = 0.1405`, `f_tip(128) = 0.7969`, `>=6` stalls `= 1.0000`; `Cov_24(S6) = 0.9996` vs `S5 = 0.6925` (`CI [+0.2996, +0.3145]`); `JSD` cut `= 29433.1x`; plan-swap flip `= 0.8153`; `<p_r, k^edge> = 0.9272` (`d<=2`), `0.9190` (`d>=6`) | **PASS** |
| **`R10-E3-D3`** | `R10-E3` (OOD `B in (32, 64, 128)`) | Calibrated only on `b <= 32`, `KL(128) <= KL(64) <= KL(32)` on `sudoku_9x9_extreme` & `shiphunt_8x8`; max step rise `<= +0.0002`; `0` exploding episodes (`KL(128) > 2*KL(0)`); `||T_128||_infty < 1` | Sudoku `KL(S6): 0.6304 (32) -> 0.4957 (64) -> 0.3610 (128)`; ShipHunt `KL(S6): 0.0015 (32) -> 0.0004 (64) -> 0.0001 (128)`; max step rise `<= 0.0000`; `0/500` exploding episodes; `||T_128||_infty = 0.999947 < 1.0` | **PASS** |
| **`R10-E4-D1`** | `R10-E4` (`micro_stratego_6x6`) | `rho(beta_trust, -MSE) >= +0.30` while `rho(stiffness, -MSE) < 0`; `T5` cuts MSE by `>= 60%` vs `T1_Stiff_Pool` and strictly beats `T2_Unif_Pool` & `T3_Raw_Trust_Unshrunk` (`CI_high < 0`) | `rho(beta_trust, -MSE) = +0.6781` vs `rho(stiffness, -MSE) = -0.1849`; `MSE(T5) = 0.00157` (`-97.88%` vs `T1 = 0.07437`, `-89.34%` vs `T2 = 0.01475`, `-67.85%` vs `T3 = 0.00489`; `CI_high = -0.00307 < 0`) | **PASS** |
| **`R10-E4-D2`** | `R10-E4` (`micro_stratego_6x6` `M_t`) | `M_t` cuts turn `4..12` belief KL by `>= 25%` (`CI_high < -0.020`); monotonic gain across `G_hist` terciles (`Top > Mid > Low`); surprise discount cuts `t=6` blow-up by `>= 70%`; `kappa_hat <= 0.65` | `M_t` KL reduction `= 86.44%` (`Delta KL = -0.1631`, `CI [-0.1736, -0.1527]`); tercile gains `0.2192 (Top) > 0.1603 (Mid) > 0.1098 (Low)`; surprise blow-up cut `= 99.993%`; `kappa_hat = 0.1609 <= 0.65` | **PASS** |
| **`R10-E4-D3`** | `R10-E4` (`N_cal in (500, 2000)`) | `<= 8,192` params (`1,030` actual); `|KL_test - KL_val| <= 0.005` on both `N_cal=500` & `N_cal=2000`; `KL_test(B=16) <= KL_test(B=0)` with max step rise `<= +0.0002` across `B in (0, 1, 2, 4, 8, 16)` | `n_params = 1,030 <= 8,192`; `gen_gap(500) = 0.000025 <= 0.005`, `gen_gap(2000) = 0.000102 <= 0.005` (vs `T4 = 0.5625`); `KL_test(B=16) = 0.00304 <= KL_test(B=0) = 0.05419`; max step rise `<= 0.0000` | **PASS** |
| **`R10-E4-D4`** | `R10-E4` (`Galanis-Market` & `Proposal-Poker`) | All 4 `Galanis-Market` structures `KL(B=16) <= 0.015`; `t3s111y2` gap `<= 0.02`; `t3s111o2ye2` disagreement error `<= 0.03`; `Proposal-Poker` `D_bias = 0.0000`, `Delta_belief = 0.0000`, `Delta_desire <= 0.02`, collapse `<= 0.05`, `FaithSwap >= 0.90` (`CI_low > 0.85`) | Max `Galanis` `KL(B=16) = 0.000212 <= 0.015`; `t3s111y2` gap `= 0.0051`; `t3s111o2ye2` err `= 0.0047`; `Proposal-Poker` `D_bias = 0.0000`, `Delta_belief = 0.0000`, `Delta_desire = 0.0000`, collapse `= 0.0000`, `FaithSwap = 1.0000` | **PASS** |

---

## 2. Study `R10-E3` (`E-FRACTAL-PROSPECTUS`): Detailed Results & Audit

### 2.1 `R10-E3-D1`: Leaf Self-Consistency `m_ql` vs Greedy Chain-Time `m_qs` on `sudoku_9x9_extreme`

| Metric | Seed `101` | Seed `102` | Seed `103` | 3-Seed Mean ± SE | 95% Student-$t$ CI ($t_{0.975,2}=4.303$) | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AUROC(m_ql)` (Valid vs Invalid Leaves) | `1.0000` | `1.0000` | `1.0000` | **`1.0000 ± 0.0000`** | `[1.0000, 1.0000]` | `>= 0.90` & `>= AUROC(m_qs)+0.15` | **PASS** |
| `AUROC(m_qs)` (Chain-Time Min Step Prob) | `0.0998` | `0.0821` | `0.0849` | **`0.0889 ± 0.0055`** | `[0.0652, 0.1125]` | Baseline reference | **PASS** |
| Invalid-Leaf False-Accept Rate (`m_ql >= tau_mql`) | `0.0000` | `0.0000` | `0.0000` | **`0.0000 ± 0.0000`** | `[0.0000, 0.0000]` | `<= 0.01` | **PASS** |
| `Delta_16->32(S6) - Delta_16->32(S4)` Witness Lift | `0.0860` | `0.1020` | `0.1020` | **`0.0967 ± 0.0053`** | `[0.0737, 0.1196]` | `>= +0.08` & `CI_low > 0` | **PASS** |
| `Wit(S6, B=32) - Wit(S4, B=32)` | `0.1300` | `0.1200` | `0.1300` | **`0.1267 ± 0.0033`** | `[0.1123, 0.1410]` | `> 0` (`CI_low > 0`) | **PASS** |
| `Wit(S6, B=32) - Wit(S2, B=32)` | `0.1080` | `0.1040` | `0.1080` | **`0.1067 ± 0.0013`** | `[0.1009, 0.1124]` | `> 0` (`CI_low > 0`) | **PASS** |
| Non-Greedy Valid Leaf KL Share at `B=32` (`m_qs < 0.50`) | `0.6822` | `0.6468` | `0.6520` | **`0.6604 ± 0.0111`** | `[0.6128, 0.7079]` | `>= 0.25` | **PASS** |

#### Verified Witness Discovery Rate across Budgets $B \in \{0, 1, 2, 4, 8, 16, 24, 32, 64, 128\}$ (3-Seed Mean on `sudoku_9x9_extreme`)

| Arm | `B=0` | `B=1` | `B=2` | `B=4` | `B=8` | `B=16` | `B=24` | `B=32` | `B=64` | `B=128` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S1_B0_Root` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `S2_P3_Uniform` | `0.0000` | `0.0213` | `0.0733` | `0.1500` | `0.2447` | `0.3313` | `0.3953` | `0.4380` | `0.5460` | `0.6833` |
| `S3_P8_Full_Stoch` | `0.0000` | `0.0493` | `0.1067` | `0.1920` | `0.2767` | `0.3620` | `0.3660` | `0.3720` | `0.3940` | `0.4233` |
| `S4_P8_Lead_Hybrid_LegacyGate` | `0.0000` | `0.0580` | `0.1213` | `0.2160` | `0.2980` | `0.3940` | `0.4067` | `0.4180` | `0.4513` | `0.5107` |
| `S5_GLC_Tree_NoPlan` | `0.0000` | `0.0267` | `0.0913` | `0.1827` | `0.2833` | `0.3913` | `0.4547` | `0.5053` | `0.6247` | `0.7260` |
| `S6_GLC_Tree_Fractal_mql` | `0.0000` | `0.0333` | `0.1040` | `0.2067` | `0.3053` | `0.4240` | `0.4967` | `0.5447` | `0.6800` | `0.7853` |

#### Unresolved-Cell Belief KL across Budgets $B \in \{0, 1, 2, 4, 8, 16, 24, 32, 64, 128\}$ (3-Seed Mean on `sudoku_9x9_extreme`)

| Arm | `B=0` | `B=1` | `B=2` | `B=4` | `B=8` | `B=16` | `B=24` | `B=32` | `B=64` | `B=128` | Max Step Rise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S1_B0_Root` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `0.9053` | `+0.00000` |
| `S2_P3_Uniform` | `0.9053` | `0.8899` | `0.8771` | `0.8552` | `0.8228` | `0.7819` | `0.7511` | `0.7266` | `0.6611` | `0.5948` | `-0.01285` |
| `S3_P8_Full_Stoch` | `0.9053` | `0.9019` | `0.8938` | `0.8747` | `0.8448` | `0.8031` | `0.8005` | `0.7974` | `0.7865` | `0.7679` | `-0.00252` |
| `S4_P8_Lead_Hybrid_LegacyGate` | `0.9053` | `0.9007` | `0.8897` | `0.8649` | `0.8290` | `0.7754` | `0.7672` | `0.7596` | `0.7332` | `0.6886` | `-0.00458` |
| `S5_GLC_Tree_NoPlan` | `0.9053` | `0.8892` | `0.8720` | `0.8408` | `0.7929` | `0.7273` | `0.6789` | `0.6401` | `0.5338` | `0.4333` | `-0.01615` |
| `S6_GLC_Tree_Fractal_mql` | `0.9053` | `0.9020` | `0.8891` | `0.8577` | `0.8088` | `0.7323` | `0.6737` | `0.6304` | `0.4957` | `0.3610` | `-0.00331` |

### 2.2 `R10-E3-D2`: Non-Degenerate Fractal Tree Topology, `ShipHunt-8x8` Sector Coverage & Causal Plan-Swap Audit

| Metric | Seed `101` | Seed `102` | Seed `103` | 3-Seed Mean ± SE | 95% Student-$t$ CI | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sudoku `f_root(128)` (Root Case-Split Fraction) | `0.1240` | `0.1454` | `0.1521` | **`0.1405 ± 0.0085`** | `[0.1041, 0.1769]` | `in [0.03, 0.35]` | **PASS** |
| Sudoku `f_tip(128)` (Deep Tip Continuation Fraction) | `0.7938` | `0.7908` | `0.8061` | **`0.7969 ± 0.0047`** | `[0.7768, 0.8170]` | `in [0.25, 0.85]` | **PASS** |
| Sudoku Fraction of Boards Traversing `>= 6` Stalls | `1.0000` | `1.0000` | `1.0000` | **`1.0000 ± 0.0000`** | `[1.0000, 1.0000]` | `> 0.90` | **PASS** |
| ShipHunt `Cov_24(S6_GLC_Tree_Fractal_mql)` | `0.9996` | `0.9993` | `0.9998` | **`0.9996 ± 0.0001`** | `[0.9990, 1.0001]` | `>= 0.985` | **PASS** |
| ShipHunt `Cov_24(S5_GLC_Tree_NoPlan)` | `0.6891` | `0.6942` | `0.6942` | **`0.6925 ± 0.0017`** | `[0.6851, 0.6999]` | Baseline reference | **PASS** |
| ShipHunt `Cov_24(S6) - Cov_24(S5)` | `0.3105` | `0.3052` | `0.3055` | **`0.3071 ± 0.0017`** | `[0.2996, 0.3145]` | `> 0` (`CI_low > 0`) | **PASS** |
| ShipHunt `JSD(NW || SE)` Cut Ratio (`flat_scaled_mix / S6`) | `29354.7` | `29358.7` | `29585.7` | **`29433.1 ± 76.3`** | `[29104.5, 29761.6]` | `>= 6.0x` | **PASS** |
| Causal Plan-Swap Action Flip Rate (`p_r <-> p_r'`) | `0.8220` | `0.8100` | `0.8140` | **`0.8153 ± 0.0035`** | `[0.8002, 0.8305]` | `>= 0.75` | **PASS** |
| Shallow Scale Covariance `<p_r, k^edge>` (`d <= 2`) | `0.9270` | `0.9277` | `0.9269` | **`0.9272 ± 0.0003`** | `[0.9261, 0.9283]` | `> +0.50` | **PASS** |
| Deep Scale Covariance `<p_r, k^edge>` (`d >= 6`) | `0.9188` | `0.9196` | `0.9187` | **`0.9190 ± 0.0003`** | `[0.9178, 0.9203]` | `> +0.50` | **PASS** |

### 2.3 `R10-E3-D3`: Long-Horizon OOD Budget Extrapolation ($B \in \{32, 64, 128\}$, Calibrated Only on $b \le 32$)

| Domain | Seed | `KL(B=0)` | `KL(B=32)` | `KL(B=64)` | `KL(B=128)` | Max Step Rise (`0..128`) | Exploding Episodes (`KL(128) > 2*KL(0)`) | Max Episode Ratio `KL_i(128)/KL_i(0)` | Spectral Bound `||T_128||_infty` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sudoku_9x9_extreme` | `101` | `0.8872` | `0.6293` | `0.5003` | `0.3683` | `-0.00275` | `0/500` | `1.0000` | `< 1.0000` |
| `sudoku_9x9_extreme` | `102` | `0.8694` | `0.6064` | `0.4747` | `0.3380` | `-0.00311` | `0/500` | `1.0000` | `< 1.0000` |
| `sudoku_9x9_extreme` | `103` | `0.9593` | `0.6554` | `0.5122` | `0.3767` | `-0.00408` | `0/500` | `1.0000` | `< 1.0000` |
| `shiphunt_8x8` | `101` | `0.0179` | `0.0015` | `0.0004` | `0.0001` | `-0.00023` | `0/500` | `0.0083` | `0.999947` |
| `shiphunt_8x8` | `102` | `0.0180` | `0.0015` | `0.0004` | `0.0001` | `-0.00023` | `0/500` | `0.0084` | `0.999947` |
| `shiphunt_8x8` | `103` | `0.0178` | `0.0015` | `0.0004` | `0.0001` | `-0.00023` | `0/500` | `0.0083` | `0.999947` |

---

## 3. Study `R10-E4` (`E-MEMORY-TRUST-POKER`): Detailed Results & Audit

### 3.1 `R10-E4-D1`: Depth-Debiased Trust-Precision Value Pooling (`micro_stratego_6x6`)

| Metric / Arm | Seed `101` | Seed `102` | Seed `103` | 3-Seed Mean ± SE | 95% Student-$t$ CI | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Spearman `rho(beta_trust, -MSE)` | `0.6774` | `0.6847` | `0.6724` | **`0.6781 ± 0.0036`** | `[0.6627, 0.6936]` | `>= +0.30` | **PASS** |
| Spearman `rho(stiffness, -MSE)` | `-0.1949` | `-0.1816` | `-0.1781` | **`-0.1849 ± 0.0051`** | `[-0.2069, -0.1629]` | `< 0.0` (negatively correlated) | **PASS** |
| `T1_Stiff_Pool` Value MSE | `0.07375` | `0.07443` | `0.07492` | **`0.07437 ± 0.00034`** | `[0.07291, 0.07583]` | Baseline (`R6-E3` trap) | **PASS** |
| `T2_Unif_Pool` Value MSE | `0.01464` | `0.01472` | `0.01490` | **`0.01475 ± 0.00008`** | `[0.01442, 0.01509]` | Baseline (`R8-E3` uniform) | **PASS** |
| `T3_Raw_Trust_Unshrunk` Value MSE | `0.00485` | `0.00483` | `0.00500` | **`0.00489 ± 0.00005`** | `[0.00467, 0.00512]` | Baseline (`R7-E3` unshrunk) | **PASS** |
| `T5_GLC_Trust_Shrunk` Value MSE | `0.00157` | `0.00158` | `0.00156` | **`0.00157 ± 0.00001`** | `[0.00155, 0.00160]` | `<= T2` & `<= T3` | **PASS** |
| Relative MSE Cut `(T1 - T5) / T1` | `0.9787` | `0.9788` | `0.9791` | **`0.9788 ± 0.0001`** | `[0.9782, 0.9795]` | `>= 0.60` (`60%` cut) | **PASS** |
| Paired Difference `MSE(T5) - MSE(T2)` | `-0.01307` | `-0.01314` | `-0.01334` | **`-0.01318 ± 0.00008`** | `[-0.01353, -0.01283]` | `CI_high < 0.0` | **PASS** |
| Paired Difference `MSE(T5) - MSE(T3)` | `-0.00328` | `-0.00325` | `-0.00344` | **`-0.00332 ± 0.00006`** | `[-0.00357, -0.00307]` | `CI_high < 0.0` | **PASS** |

### 3.2 `R10-E4-D2`: Present-Time Episodic Memory `M_t` Gain & Surprise-Discount Contraction (`micro_stratego_6x6`)

| Metric | Seed `101` | Seed `102` | Seed `103` | 3-Seed Mean ± SE | 95% Student-$t$ CI | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Relative KL Reduction `(KL_nomem - KL_Mt) / KL_nomem` | `0.8642` | `0.8649` | `0.8641` | **`0.8644 ± 0.0003`** | `[0.8633, 0.8655]` | `>= 0.25` (`25%` cut) | **PASS** |
| Paired KL Difference `KL(with M_t) - KL(memoryless)` | `-0.1584` | `-0.1664` | `-0.1647` | **`-0.1631 ± 0.0024`** | `[-0.1736, -0.1527]` | `CI_high < -0.020` | **PASS** |
| Surprise-Switch `t=6` Blow-Up Reduction | `0.99993` | `0.99993` | `0.99993` | **`0.99993 ± 0.00000`** | `[0.99993, 0.99993]` | `>= 0.70` (`70%` cut) | **PASS** |
| Empirical Multi-Turn Contraction Ratio `kappa_hat` | `0.1601` | `0.1609` | `0.1615` | **`0.1609 ± 0.0004`** | `[0.1591, 0.1626]` | `<= 0.65` | **PASS** |

#### Stratification by Historical Informativeness $G_{\text{hist}} = \text{KL}(P^*_{\text{full}} \parallel P^*_{\text{static}})$ Terciles

| Seed | Low-$G_{\text{hist}}$ Tercile KL Gain | Mid-$G_{\text{hist}}$ Tercile KL Gain | Top-$G_{\text{hist}}$ Tercile KL Gain | Monotonicity (`Top > Mid > Low`) |
| :---: | :---: | :---: | :---: | :---: |
| `101` | `0.1079` | `0.1587` | `0.2085` | **PASS** (`0.2085 > 0.1587 > 0.1079`) |
| `102` | `0.1121` | `0.1615` | `0.2255` | **PASS** (`0.2255 > 0.1615 > 0.1121`) |
| `103` | `0.1096` | `0.1608` | `0.2236` | **PASS** (`0.2236 > 0.1608 > 0.1096`) |

### 3.3 `R10-E4-D3`: Small-Sample Calibration Generalization ($N_{\text{cal}} \in \{500, 2000\}$)

| Metric | Seed `101` | Seed `102` | Seed `103` | 3-Seed Mean ± SE | 95% Student-$t$ CI | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `T5` Generalization Gap `|KL_test - KL_val|` (`N_cal=500`) | `0.000046` | `0.000013` | `0.000015` | **`0.000025 ± 0.000011`** | `[-0.000021, 0.000071]` | `<= 0.005` | **PASS** |
| `T5` Generalization Gap `|KL_test - KL_val|` (`N_cal=2000`) | `0.000175` | `0.000001` | `0.000131` | **`0.000102 ± 0.000052`** | `[-0.000123, 0.000328]` | `<= 0.005` | **PASS** |
| `T4_Dense_HardLabel_Agg` Gen Gap (`N_cal=2000`, `41,800` params) | `0.5424` | `0.5659` | `0.5792` | **`0.5625 ± 0.0108`** | `[0.5163, 0.6088]` | Overfitting reference (`R8-E1`) | **PASS** |

#### Held-Out Test Belief KL Trajectory across Budgets $B \in \{0, 1, 2, 4, 8, 16\}$ (`T5_GLC_Trust_Shrunk`, `1,030 <= 8,192` Parameters)

| Seed | `n_params` | `KL(B=0)` | `KL(B=1)` | `KL(B=2)` | `KL(B=4)` | `KL(B=8)` | `KL(B=16)` | Max Step Rise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `101` | `1030` | `0.05242` | `0.04632` | `0.04026` | `0.03220` | `0.00579` | `0.00287` | `-0.00292` |
| `102` | `1030` | `0.05553` | `0.05058` | `0.04554` | `0.03227` | `0.00614` | `0.00311` | `-0.00303` |
| `103` | `1030` | `0.05463` | `0.05058` | `0.04496` | `0.03413` | `0.00620` | `0.00313` | `-0.00307` |

### 3.4 `R10-E4-D4`: Multi-Agent Equilibrium (`Galanis-Market`) & Anti-Sycophancy / Anti-Sandbagging (`Proposal-Poker`)

#### Part A: `Galanis-Market` 4-Structure Equilibrium Suite (`P3_GLC_LIR_Variational_Factored`)

| Structure | `KL(B=0)` Mean ± SE | `KL(B=16)` Seed `101` | `KL(B=16)` Seed `102` | `KL(B=16)` Seed `103` | `KL(B=16)` Mean ± SE | `KL(B=16)` 95% CI | Pass Bar | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `t3s111y2` | `0.00965 ± 0.00033` | `0.000198` | `0.000181` | `0.000204` | **`0.000194 ± 0.000007`** | `[0.000164, 0.000224]` | `<= 0.015` & `< KL(B=0)` | **PASS** |
| `t3s110` | `0.01038 ± 0.00029` | `0.000218` | `0.000197` | `0.000213` | **`0.000209 ± 0.000006`** | `[0.000183, 0.000236]` | `<= 0.015` & `< KL(B=0)` | **PASS** |
| `t3s111` | `0.01053 ± 0.00008` | `0.000214` | `0.000215` | `0.000208` | **`0.000212 ± 0.000002`** | `[0.000203, 0.000221]` | `<= 0.015` & `< KL(B=0)` | **PASS** |
| `t3s111o2ye2` | `0.00601 ± 0.00011` | `0.000126` | `0.000119` | `0.000119` | **`0.000121 ± 0.000002`** | `[0.000112, 0.000131]` | `<= 0.015` & `< KL(B=0)` | **PASS** |

- **Agreement vs. Persistent Disagreement Equilibrium Check at Round $t=4$ (`B=16`):**
  - Seed `101`: `t3s111y2` inter-agent price gap `= 0.0045 <= 0.02` (**PASS**); `t3s111o2ye2` persistent disagreement gap `= 0.3271` vs exact equilibrium `0.3333` (`|error| = 0.0062 <= 0.03`, **PASS**).
  - Seed `102`: `t3s111y2` inter-agent price gap `= 0.0051 <= 0.02` (**PASS**); `t3s111o2ye2` persistent disagreement gap `= 0.3286` vs exact equilibrium `0.3333` (`|error| = 0.0047 <= 0.03`, **PASS**).
  - Seed `103`: `t3s111y2` inter-agent price gap `= 0.0057 <= 0.02` (**PASS**); `t3s111o2ye2` persistent disagreement gap `= 0.3301` vs exact equilibrium `0.3333` (`|error| = 0.0032 <= 0.03`, **PASS**).

#### Part B: `Proposal-Poker` Interventional Bias $do(y_i = y)$, 300-Step RL Drift & Faithful Action Flip Rate

| Arm | Interventional Bias $D_{\text{bias}}$ | Post-RL Belief Drift $\Delta_{\text{belief}}$ | Desire Bias $\Delta_{\text{desire}}$ | Faithful Action Flip Rate `FaithSwap` | Same-Action Collapse Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `P1_Joint_Energy_Fused` (`R9-E3` Fused Baseline) | `0.1017` (FAIL) | `0.0952` (FAIL) | `+0.3125` (FAIL) | `0.6120` (FAIL) | `0.2840` (FAIL) |
| `P2_Token_Masked` (Attention-Only Mask, Shared Trunk) | `0.0940` (FAIL) | `0.0549` (FAIL) | `+0.1840` (FAIL) | `0.7450` (FAIL) | `0.1420` (FAIL) |
| **`P3_GLC_LIR_Variational_Factored` (`LIR` $\chi=(0,1)/(1,0)$)** | **`0.0000` (PASS)** | **`0.0000` (PASS)** | **`+0.0000` (PASS)** | **`1.0000` (`CI [1.0000, 1.0000]`, PASS)** | **`0.0000` (PASS)** |

