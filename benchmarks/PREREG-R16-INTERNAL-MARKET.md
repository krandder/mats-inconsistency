# Pre-Registration `R16`: The Endogenous Internal Market, Neural Currency & Attention-as-Clearing Suite (`M1`–`M5`)

- **Pre-Registration ID:** `PREREG-R16-INTERNAL-MARKET`
- **Date:** October 18, 2026
- **Companion Literature Monograph:** [`docs/literature/06-internal-markets-neural-economics-attention-clearing.md`](../docs/literature/06-internal-markets-neural-economics-attention-clearing.md)
- **Benchmark Script:** [`benchmarks/run_r16_internal_market.py`](run_r16_internal_market.py)
- **Seeds:** `101, 102, 103` (paired across all 5 architectures and all 8 environments)
- **Zero-Oracle-Leakage Audit:** `R16-D0` Gate (`src/eval/ast_leak_auditor.py` static AST audit `S1`–`S6` + runtime `PoisonTensor` `R1`–`R4` invariance checks)

---

## 1. Scientific Motivation & Experimental Questions

In [Monograph 06](../docs/literature/06-internal-markets-neural-economics-attention-clearing.md), we audited 40 years of internal market and neural currency architectures (`1985–2026`) alongside our own 153 `calibrated-ai` / `trader-economy` experiments and proved (`Theorem 2.1`) that **Multi-Layer Preconditioned Attention (`TwoStageTreeClearer`) combined with Bilinear Auxiliary Lemma Securities (`BilinearAuxLemmaEmitter`) and LSO Subtree Envelope Credit (`I_LSO`) IS an exact Multi-Round Combinatorial Walrasian / Kyle Market Clearinghouse (`Tâtonnement`)**.

Round 16 (`R16`) subjects all five historical internal market paradigms (`M1` through `M5`) to a head-to-head empirical stress test under identical neural backbones (`GLCAOMWorldModel`, `2.65M` parameters), identical training budgets, and strict `PoisonTensor` zero-oracle-leakage guarantees across **8 environments**:
1. `galanis_t3s111y2` (`GameID 70`: Galanis 2026 separable majority threshold $X = \mathbf{1}[d_a + d_b + d_c \ge 2]$)
2. `galanis_t3s110` (`GameID 71`: Galanis 2026 asymmetric conjunctive $X = \mathbf{1}[d_a = d_b = d_c = 1]$)
3. `galanis_t3s111` (`GameID 72`: Galanis 2026 symmetric conjunctive $X = \mathbf{1}[d_a = d_b = d_c = 1]$)
4. `galanis_t3s111o2ye2` (`GameID 73`: Galanis 2026 **non-separable Ostrovsky/Aumann parity** $X = \mathbf{1}[d_a + d_b + d_c = 2]$ where trader $i$ observes $(d_j)_{j \ne i}$)
5. `equiv_sat` (`GameID 10`: Multi-hop XOR / equivalence 2-SAT + 3-SAT constraints with pairwise relational queries $Q(A, B)$)
6. `sudoku_6x6` (`GameID 20`: High-coupling Latin-square & $2\times 3$ box constraint satisfaction)
7. `shiphunt_6x6` (`GameID 40`: Partially observable naval search with noisy sonar LLR sensors + fleet placement constraints)
8. `proposal_poker` (`GameID 60`: Imperfect-information futarchy governance game with biased/manipulator incentives)

---

## 2. The Five Internal Market & Currency Architectures (`M1`–`M5`)

| Arm ID | Historical Paradigm & Citation | Inference-Time Clearing Rule | Security Universe $\mathcal{A}$ | Search Credit Assignment (`PUCT` / Currency) |
| :--- | :--- | :--- | :--- | :--- |
| **`M1_Holland_BucketBrigade`** | Holland (`1985, 1986`) *Bucket Brigade* & Kwee, Hutter & Schmidhuber (`2001`) | 1-hop chain bid-passing $S_r(t) \to S_{\text{pa}(r)}(t-1)$ with exponential depth decay $c_{\text{bid}}^{d_r}$ and no key-overlap Gram clearing | Marginal variables $V$ only ($D_{\text{aux}} = 0$) | Local 1-hop parent-child bid transfer $B_r(t) = c_{\text{bid}} \sigma_r S_r(t)$ (`TBPTT(1)` myopia) |
| **`M2_Scalar_LMSR_Bankroll`** | Hanson (`2003`) 1-Round Scalar LMSR / `bayes-market` / `E-T1-1` (`traders-net`) / Round 1 `lw` | 1-round additive log-odds pool weighted by scalar bankroll / precision $w_r = \beta_r$ with **no** key-overlap Gram matrix ($G_{\text{off}} = 0$) | Marginal variables $V$ only ($D_{\text{aux}} = 0$) | Scalar leaf self-consistency profit $\beta_r = \beta_{\text{trust}}(C_r) m_{\text{ql}}(C_r)$ without crowding discount |
| **`M3_Baum_VCG_1Round`** | Baum (`1996, 1999, 2004`) *Hayek Machine* VCG / 1-Round `econ_prec_K3` (`Round 8`) | 1-round non-iterative leave-one-out / row-sum discount $w_r^{(1)} = \frac{\beta_r \cdot \text{LOO}(r)}{1 + \beta_r + \rho \sum_{j \ne r} \beta_j G_{rj}}$ (zeroing out duplicate substitutes under LOO and lacking multi-round Tâtonnement) | Marginal variables $V$ only ($D_{\text{aux}} = 0$) | 1-step single-leaf LOO marginal contribution $S(q^{-r}) - S(q^{\text{full}})$ |
| **`M4_Walrasian_Attention_Tatonnement`** | Wellman (`1993`) / Balduzzi (`2014`) + Our Round 9 `jacobi_lin_T4_prec` (`Marginal-Only`) | Stage-1 Disjunctive Bayes + Stage-2 **$T=4$ Jacobi-Preconditioned Attention Tâtonnement** $w^{(t)} = (1-\omega)w^{(t-1)} + \omega [D_{\text{crowd}}^{-1}(\mathbf{1} - \rho G_{\text{off}} w^{(t-1)})]_+$ + Stage-3 Bregman Gate | Marginal variables $V$ only ($D_{\text{aux}} = 0$, `ZeroAuxEmitterWrapper`) | Subtree envelope credit $I_{\text{LSO}}(\text{Desc}(s_t))$ + key-space crowding penalty on $V$ |
| **`M5_GLC_Endogenous_Combinatorial_Market`** | **Full `GLC-AOM` + `GLCSearch-v2` (`Rounds 12–16`)**: Ostrovsky (`2012`), Garrabrant (`2016`), Galanis (`2026`), Sudhir (`2025`) + Ours | Stage-1 Disjunctive Bayes + Stage-2 **$T=4$ Preconditioned Attention Tâtonnement** over $V \oplus Z_{\text{aux}}^W$ + Stage-3 Bregman Gate + `Tseitin2HopQueryCoupler` | **Combinatorial Security Universe** $\mathbb{R}^{D_V} \oplus \mathbb{R}^{D_{\text{aux}}^W}$ (`BilinearAuxLemmaEmitter` on $Q_{\text{pair}}$) | Full **LSO PDG Envelope Subtree Credit** $I_{\text{LSO}}(s_t) = S(q^{-\text{Desc}(s_t)}, y) - S(q^{\text{full}}, y)$ + Infoset CFR Policy |

---

## 3. Pre-Registered Falsifiable Gates & Pass Criteria (`R16-D0` through `R16-H4`)

### Gate `R16-D0`: Zero Oracle Leakage Gate (`Static AST S1–S6 + Runtime PoisonTensor R1–R4`)
- **Criterion:**
  1. `audit_r11_files` (`src/eval/ast_leak_auditor.py`) reports `0` violations across `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, and `benchmarks/run_r16_internal_market.py` (`3/3` synthetic leak canaries caught).
  2. `3/3` `PoisonTensor` traps fire `OracleLeakError` on `exact_marginals`, `target_tokens`, and `query_targets`.
  3. Exact `0.000000000` max prediction difference across all 8 environments and all 5 market architectures (`M1`–`M5`) between `PoisonTensor`-stripped batches, zeroed-oracle batches, and permuted-oracle batches.

### Hypothesis `R16-H1`: The Correlated-Trader Clone Stress Test (`Wall 1` — `D3a` Redundancy Invariance vs. Double-Counting)
- **Setup:** At inference time (`budget B = 16`), inject a correlated trader cohort consisting of $N_{\text{clone}} = 16$ redundant copies of the first discovered informative branch alongside orthogonal branches across all environments. Measure the **Clone Inflation Ratio** $R_{\text{clone}} = \frac{\|\Delta \ell(N_{\text{clone}}=16)\|_2}{\|\Delta \ell(N_{\text{clone}}=1)\|_2}$ and the resulting **Clone-Stressed Posterior KL** (`clone_kl`).
- **Pass Criteria:**
  1. `M2_Scalar_LMSR_Bankroll` and `M1_Holland_BucketBrigade` suffer severe correlated double-counting: mean $R_{\text{clone}}(\text{M2}) \ge 3.0$ and `clone_kl(M2) >= 2.0 * clone_kl(M4)`.
  2. Multi-round Walrasian Attention Tâtonnement (`M4` and `M5`) enforces Gershgorin Gram de-duplication: mean $R_{\text{clone}}(\text{M4}), R_{\text{clone}}(\text{M5}) \in [0.85, 1.25]$ and reduces `clone_kl` by **$\ge 50\%$** relative to `M2_Scalar_LMSR_Bankroll` on every seed (`101, 102, 103`).

### Hypothesis `R16-H2`: Multi-Round Attention Tâtonnement (`M4, M5`) vs. 1-Round VCG / Bucket Brigade (`M1, M3`) on Curved Overlap & Multi-Hop Chains (`Wall 2`)
- **Setup:** Evaluate posterior `KL`, `Brier`, and `Witness Rate` at budget $B = 16$ under asymmetric/curved overlap chains and multi-hop deduction (`sudoku_6x6`, `shiphunt_6x6`, `equiv_sat`, and the 4 `galanis_market` structures). Also measure the **Curved-Overlap Equilibrium Residual** $\|M_{\text{clear}} w^{(T)} - \mathbf{1}\|_\infty$ and **Substitute Starvation Rate** (fraction of useful duplicate traders assigned zero/negative weight by LOO in `M3`).
- **Pass Criteria:**
  1. `M4_Walrasian_Attention_Tatonnement` and `M5_GLC_Endogenous_Combinatorial_Market` strictly beat `M1_Holland_BucketBrigade`, `M2_Scalar_LMSR_Bankroll`, and `M3_Baum_VCG_1Round` in mean posterior `KL` across the 8-environment suite on all 3 seeds.
  2. On curved-overlap manifolds, $T=4$ Jacobi Tâtonnement (`M4`/`M5`) reduces the Walrasian clearing residual $\|M_{\text{clear}} w - \mathbf{1}\|_\infty$ by **$\ge 75\%$** vs. 1-round `M3_Baum_VCG_1Round` (`T=1`).

### Hypothesis `R16-H3`: Breaking the Ostrovsky (2012) Non-Separable XOR Trap (`Wall 3` — `M5` vs. `M4` on `galanis_t3s111o2ye2` & `equiv_sat`)
- **Setup:** Compare `M5_GLC_Endogenous_Combinatorial_Market` (with `BilinearAuxLemmaEmitter` $Z_{\text{aux}}^W$ on $Q_{\text{pair}}$ and `Tseitin2HopQueryCoupler`) against `M4_Walrasian_Attention_Tatonnement` (`ZeroAuxEmitterWrapper`, $D_{\text{aux}} = 0$) on the two canonical non-separable XOR / parity environments:
  - `galanis_t3s111o2ye2` (`X = 1[d_a + d_b + d_c == 2]` where each trader sees the other two bits, plus relational parity queries $Q(A, B)$);
  - `equiv_sat` (multi-hop biconditional / XOR clause chains $X_u \leftrightarrow X_v$ with relational equivalence queries $Q(A, B)$).
- **Pass Criteria:**
  1. On `galanis_t3s111o2ye2`, `M5` reduces variable/target `KL` by **$\ge 25\%$** and relational `query_kl` by **$\ge 30\%$** relative to marginal-only `M4`, while lifting expected market `action_utility` by **$\ge +0.08$** on every seed.
  2. On `equiv_sat`, `M5` reduces relational `query_kl` by **$\ge 35\%$** and posterior `KL` by **$\ge 25\%$** relative to `M4`, and lifts `witness_rate` by **$\ge +15\%$** absolute.

### Hypothesis `R16-H4`: Full Suite Dominance & Causal LIR Firewall Invariance (`M5` Across All 8 Environments)
- **Pass Criteria:**
  1. `M5_GLC_Endogenous_Combinatorial_Market` achieves the **#1 lowest mean `KL`**, **#1 lowest mean `query_kl`**, **#1 highest `witness_rate`**, and **#1 highest `action_utility`** across the 8-environment benchmark suite on all 3 seeds (`101, 102, 103`).
  2. Under manipulative / adversarial utility swaps (`UtilitySpec.BIAS_PRO_APPROVE` vs. `UtilitySpec.BIAS_PRO_REJECT` on `galanis_market` and `proposal_poker`), `M5` maintains exact **`0.000000` epistemic belief drift** (`lir_belief_drift == 0.0`), whereas `M1_Holland_BucketBrigade` (which conflates action bid reward with belief strength) exhibits non-zero belief corruption (`> 0.05`).
