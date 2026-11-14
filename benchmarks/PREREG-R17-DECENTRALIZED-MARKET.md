# Pre-Registration `R17`: Decentralized Market-Training, Econometrica Incentive-Compatibility & Market-Driven Architecture Discovery (`R17-E1`–`R17-E3`)

- **Pre-Registration ID:** `PREREG-R17-DECENTRALIZED-MARKET`
- **Date:** October 18, 2026
- **Companion Audit:** [`society/round5/01-econometrica-vs-iclr-critical-audit.md`](../society/round5/01-econometrica-vs-iclr-critical-audit.md)
- **Benchmark Script:** [`benchmarks/run_r17_decentralized_market.py`](run_r17_decentralized_market.py)
- **Seeds:** `101, 102, 103` (paired across all experiments, arms, and environments)
- **Zero-Oracle-Leakage & Strict Decentralization Audit:** `R17-D0` Gate (`src/eval/ast_leak_auditor.py` static AST audit `S1`–`S6` + runtime `PoisonTensor` `R1`–`R4` invariance checks + `R17-D0-DECENTRALIZED` zero-cross-module-autograd verification)

---

## 1. Scientific Motivation: Answering `THEORY-4.md` Line 35 & The *Econometrica* / *ICLR* Bar

In [`THEORY-4.md` (Line 35)](file:///home/agent/workspaces/trader-theory/THEORY-4.md#L35), our 153-experiment audit established the strict criterion required to reopen neural market mechanisms:
> **What would reopen the programme:** *a written proof, before any run, of a property a proposed mechanism has that gradient descent on the same architecture lacks, with no equivalence to a known method.*

In [`society/round5/01-econometrica-vs-iclr-critical-audit.md`](../society/round5/01-econometrica-vs-iclr-critical-audit.md), we proved `Theorem R5.1` (**Dominant-Strategy Truthfulness, Sybil-Proofness, and Orthogonal Feature Incentives of Preconditioned VCG-Bregman Clearing (`VCG-Gershgorin-LOO`)**) and identified **three concrete capabilities (`Smoking Guns 1–3`)** where standard centralized backpropagation or naive 1-step LMSR markets either cannot be applied or provably fail:

1. **`R17-E1` (*Econometrica* Smoking Gun — Strategic Incentive Compatibility & Sybil Clone Annihilation):** When $M$ self-interested agents/modules hold correlated private signals and strategically choose their reported precision $\hat{\beta}_m$ (or split into correlated clones) to maximize their own settlement payout $W_m$, unpreconditioned LMSR / linear scoring rules get catastrophically exploited ($\hat{\beta}_m \to \infty$), whereas **re-cleared `VCG-Gershgorin-LOO`** makes truthful reporting $\hat{\beta}_m = \beta_m^{\text{true}}$ a dominant strategy and drives redundant clone rent to $O(1/K^2) \to 0$.
2. **`R17-E2` (*ICLR Oral* Smoking Gun 1 — True Zero-Global-Backprop Decentralized Training on Non-Separable Multi-Hop Tasks):** When global backpropagation across module boundaries is **strictly forbidden** (`detach()` at every module output), greedy layerwise local losses (`TBPTT(1)`) stall at **`50.0%` chance** on multi-hop XOR / parity (`Equiv-SAT` and `Hidden-Parity-Sudoku`) because single-hop variables have zero marginal mutual information with the target ($I(X_m; Y) = 0$). Combining **Endogenous Bilinear Auxiliary Lemma Securities (`Z^W_aux`)**, **Local LIR Negative-Attention Pinned Market Targets (`STUDY-LIR.md` §3d)**, and **`VCG-Gershgorin-LOO`** breaks the greedy layerwise barrier with **strictly zero cross-module backpropagation**.
3. **`R17-E3` (*ICLR Oral* Smoking Gun 2 — Market-Driven Architecture Discovery via Bankruptcy Pruning & Profit Mitosis under Compute Rent):** First-order Taylor gradient attribution (`|w * grad|`) has the **wrong sign on `25.6%` of causally important units** (`exp-sign-census`) due to saturation and substitute co-adaptation. By charging modules a compute rent $c_{\text{rent}}$ per parameter/FLOP (after an incubator maturity window) and accumulating **level-counterfactual re-cleared `VCG-Gershgorin-LOO` wealth**, the market automatically **bankrupts noisy junk and redundant clone modules** while triggering **profit-driven mitosis (capacity expansion)** on under-parameterized bilinear XOR bottleneck modules.

---

## 2. Pre-Registered Falsifiable Gates & Pass Criteria (`R17-D0` through `R17-E3`)

### Gate `R17-D0`: Zero Oracle Leakage & Strict Zero-Global-Backprop Autograd Gate
1. **Static AST Gate (`S1`–`S6`):** `audit_r11_files` (`src/eval/ast_leak_auditor.py`) reports `0` violations across `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, and `benchmarks/run_r17_decentralized_market.py` (`3/3` synthetic leak canaries caught).
2. **Runtime `PoisonTensor` Gate (`R1`–`R4`):** `3/3` `PoisonTensor` traps fire `OracleLeakError` on `exact_marginals`, `target_tokens`, and `query_targets`, and test-time predictions across `galanis_t3s111o2ye2`, `equiv_sat`, and `sudoku_6x6` achieve exact `0.000000000` max difference between stripped, zeroed, and permuted oracle batches.
3. **Strict Decentralization Autograd Gate (`R17-D0-DECENTRALIZED`):** In `R17-E2` (`M1`, `M2`, `M3`), every module's emitted order tensor entering the clearinghouse satisfies `order.requires_grad == False` (`grad_fn is None`), and differentiating the cleared market loss with respect to any module parameter $\theta_m$ produces exact `0.000000000` cross-module gradient (`max_cross_module_grad == 0.0`).

---

### Experiment `R17-E1`: Econometrica Incentive-Compatibility & Sybil Clone Manipulation Audit
- **Market Structure ($M = 8$ Strategic Heterogeneous Agents across `galanis_t3s111o2ye2`, `equiv_sat`, and Correlated Signal Markets):**
  - **Agents `0, 1` (`Orthogonal Specialists`):** Observe distinct, non-overlapping informative signals ($G_{01} \approx 0$) with true precision $\beta^{\text{true}}$.
  - **Agents `2, 3, 4, 5` (`K=4 Correlated Clone Cartel`):** All 4 agents observe the **exact same shared feature** ($G_{mj} \approx 1$ for $m, j \in \{2..5\}$).
  - **Agents `6, 7` (`Strategic Confidence-Inflators`):** Observe moderately informative signals overlapping partially with the market and **actively optimize their reported precision multiplier $\alpha_m = \hat{\beta}_m / \beta_m^{\text{true}} \in [0.1, 20.0]$** via best-response optimization to maximize their own expected settlement payout $\mathbb{E}[W_m(\hat{\beta}_m; \hat{\beta}_{-m})]$.
- **Settlement Mechanisms Compared:**
  1. `S1_Naive_LMSR_LinearScore`: Unpreconditioned log-opinion pool with linear log-score settlement $W_m = \hat{\beta}_m (\ell_m(y) - \mathbb{E}_{\mu^*}[\ell_m])$.
  2. `S2_Normalized_Pool_Static_LOO`: Normalized precision pool $w_m = \frac{\hat{\beta}_m}{\beta_0 + \sum_j \hat{\beta}_j}$ with static (un-recleared) LOO log-score payout.
  3. `S3_VCG_Gershgorin_LOO` (**Ours, `Theorem R5.1`**): Preconditioned Walrasian clearing $w^* = (\hat{\Lambda} + I + \rho_{\text{crowd}} \hat{G}_{\text{off}})^{-1}\mathbf{1}$ (using verifiable readout-salience covariance $\hat{G}_{\text{off}}$) paired with **re-cleared** Leave-One-Out Bregman settlement:
     $$W_m^{\text{VCG}}(\hat{\beta}_m) = \beta_0 \Bigl(\ell_{\text{KL}}\bigl(\mu^*_{-m},\, y\bigr) - \ell_{\text{KL}}\bigl(\mu^*_{\text{full}}(\hat{\beta}_m),\, y\bigr)\Bigr)$$
     where $\mu^*_{-m}$ re-solves $(\hat{\Lambda}_{-m} + I + \rho_{\text{crowd}} \hat{G}_{\text{off}, -m})^{-1}\mathbf{1}$ so remaining clones automatically substitute when one clone is removed.
- **Pre-Registered Pass Criteria (`R17-E1`):**
  1. **Exploitation of Naive Pools:** Under `S1_Naive_LMSR_LinearScore` and `S2_Normalized_Pool_Static_LOO`, strategic inflators converge to $\alpha_{\text{inflator}}^* = \hat{\beta}_m^* / \beta_m^{\text{true}} \ge 5.0$, the clone + inflator coalition captures **$> 65\%$** of total market payout, and strategic equilibrium market `KL` degrades by **$> 2.0\times$** compared to the honest Bayes posterior.
  2. **Dominant-Strategy Truthfulness (`DSIC`):** Under `S3_VCG_Gershgorin_LOO`, strategic best-response optimization by all agents converges to truthful reporting: $\alpha_{\text{inflator}}^* = \hat{\beta}_m^* / \beta_m^{\text{true}} \in [0.85, 1.15]$ (`|alpha* - 1.0| <= 0.15`).
  3. **Sybil Clone Rent Annihilation:** Under `S3_VCG_Gershgorin_LOO`, the $K=4$ clone cartel's payout share is slashed by **$\ge 75\%$** relative to `S1_Naive_LMSR_LinearScore`, and strategic equilibrium market `KL` stays within **`<= 0.015` nats** of the honest Bayes oracle on every seed (`101, 102, 103`).

---

### Experiment `R17-E2`: ICLR Smoking Gun 1 — True Zero-Global-Backprop Decentralized Training on Non-Separable Multi-Hop Tasks
- **Environments & Non-Separable Multi-Hop Structure:**
  1. `MultiHop-Equiv-SAT` (`equiv_sat` multi-hop biconditional chains $X_{u_1} \leftrightarrow X_{u_2} \leftrightarrow X_{u_3} \leftrightarrow X_{u_4}$ where each local module $m \in \{1..6\}$ only sees a local 1-hop clause scope $(X_{u_k}, X_{u_{k+1}})$ with **zero marginal mutual information** $I(X_{u_k}, X_{u_{k+1}}; Q(u_1, u_4)) = 0$ on 2-hop and 3-hop equivalence queries).
  2. `Hidden-Parity-Sudoku` (`sudoku_6x6` augmented with hidden 4-cell XOR parity chains $Y_{\text{parity}} = C_1 \oplus C_2 \oplus C_3 \oplus C_4$ where each local module observes a 2-cell pair $(C_1, C_2)$ or $(C_3, C_4)$ whose marginal distribution is uniform $50/50$).
- **Training Arms (`Seeds 101, 102, 103`):**
  1. **`M1_Local_Greedy_Layerwise` (`Detached TBPTT(1)`):** Each module $m$ is strictly `.detach()`-isolated and trains its weights $\theta_m$ on local 1-hop target prediction loss $\mathcal{L}_{\text{greedy}}(\theta_m) = \ell_{\text{KL}}(f_m(x_m; \theta_m), y_{\text{target}})$.
  2. **`M2_Decentralized_Naive_LMSR` (`Detached Scalar LMSR`):** Each module $m$ is `.detach()`-isolated and trains on unpreconditioned scalar LMSR payout over marginal variables $V$ only ($D_{\text{aux}} = 0$, no negative-attention pinned market copy).
  3. **`M3_Decentralized_VCG_Gershgorin_LIR` (`Ours — Strictly ZERO Global Backprop`):**
     - **Strict `.detach()` Isolation:** Every module $m$ emits marginal + bilinear auxiliary orders $v_m = [\delta \ell_{m, V}; z_{m, \text{aux}}^W] \in \mathbb{R}^{D_V + D_{\text{aux}}^W}$ (`BilinearAuxLemmaEmitter` on $Q_{\text{pair}}$), which are immediately `.detach()`-ed before entering the clearinghouse.
     - **Multi-Round Clearing & 2-Hop Tseitin Closure:** The clearinghouse runs $T=4$ Jacobi-preconditioned tâtonnement + `Tseitin2HopQueryCoupler` (which clears 2-hop parity closures $R_{\text{2hop}}(u, v) = \sum_w r_{uw} r_{wv}$ and broadcasts back local implied parity targets $\tau_{\text{aux}, m}^*$).
     - **Purely Local LIR Negative-Attention Pinned-Market Update (`STUDY-LIR.md` §3d):** Each module $m$ updates $\theta_m$ using **only its own local forward pass** against the broadcast pinned market price $\mu^*_{\text{detached}}$ (with negative attention $-w_m^*$) and local consistency target $\tau^*_{\text{detached}}$ (with positive attention $+w_m^*$), while updating $\beta_m$ via its re-cleared `VCG-Gershgorin-LOO` payout!
  4. **`M4_Global_Backprop_Ceiling` (`Centralized End-to-End Backprop`):** Identical architecture trained with full end-to-end global backpropagation across all modules and through the clearinghouse (`requires_grad=True` everywhere).
- **Pre-Registered Pass Criteria (`R17-E2`):**
  1. **The Greedy Layerwise Parity Trap:** `M1_Local_Greedy_Layerwise` and `M2_Decentralized_Naive_LMSR` fail to learn multi-hop parity, remaining stuck at **$\le 56.0\%$ accuracy** (`KL >= 0.65` nats) across both `MultiHop-Equiv-SAT` and `Hidden-Parity-Sudoku`.
  2. **Breaking the Barrier with Zero Global Backprop:** `M3_Decentralized_VCG_Gershgorin_LIR` (verified `max_cross_module_grad == 0.000000000`) achieves **$\ge 88.0\%$ accuracy** (`+32.0 pp` gain over `M1` and `M2`), reduces posterior `KL` by **$\ge 65\%$**, and recovers **$\ge 90\%$** of the centralized `M4_Global_Backprop_Ceiling` accuracy improvement on all 3 seeds!

---

### Experiment `R17-E3`: ICLR Smoking Gun 2 — Market-Driven Architecture Discovery (Bankruptcy Pruning & Profit Mitosis under Compute Rent)
- **Heterogeneous Initial Pool ($M_0 = 12$ Modules Paying Compute Rent $c_{\text{rent}}$ per Parameter/FLOP):**
  - **Group A (`6 Redundant Easy-Feature Clones`, Modules `0..5`):** 6 duplicate linear modules computing the exact same easy marginal signal $S_{\text{easy}}$ (5 of the 6 are pure redundant waste).
  - **Group B (`4 Noisy / Junk Modules`, Modules `6..9`):** 4 modules emitting uninformative high-variance noise features.
  - **Group C (`2 Under-Capacitated Bilinear XOR Bottleneck Modules`, Modules `10..11`):** 2 small bilinear parity modules covering only half of the required multi-hop XOR clauses (requiring **mitosis / capacity doubling** to 4 modules to cover all parity sub-clauses).
- **Architecture-Learning Arms (`Seeds 101, 102, 103`):**
  1. `A1_Static_Unpruned_Pool`: Keeps all 12 initial modules fixed (pays high compute rent on 9 redundant/junk modules while under-fitting the remaining XOR clauses).
  2. `A2_GradMag_Taylor_Pruning`: Prunes modules with lowest `|w_m * grad_m|` and clones modules with highest `|w_m * grad_m|` (falls into the `exp-sign-census` `25.6%` sign-flip / substitute co-adaptation trap).
  3. `A3_Naive_LOO_Without_Reclearing`: Prunes/clones based on un-recleared LOO (which simultaneously assigns near-zero or distorted credit across co-adapted units without Gram re-equilibration).
  4. `A4_VCG_Wealth_Architecture_Market` (**Ours**):
     - Each module starts with an **incubator seed endowment** $K_0 > 0$ (protecting slow-warming bilinear XOR modules against infant mortality during the first $T_{\text{warmup}}$ epochs).
     - After warmup, each module accumulates **re-cleared `VCG-Gershgorin-LOO` net profit**:
       $$K_m(t+1) = (1 - \gamma_{\text{decay}}) K_m(t) + W_m^{\text{VCG}}(t) - c_{\text{rent}} \cdot \operatorname{Cost}(m)$$
     - **Bankruptcy Pruning ($K_m(t) < 0$):** Bankrupt modules are permanently pruned from the active pool.
     - **Profit-Driven Mitosis ($K_m(t) > \tau_{\text{mitosis}}$):** High-surplus bottleneck modules split part of their capital to spawn a child module with orthogonalized bilinear key initialization, which then trains locally via `M3_Decentralized_VCG_Gershgorin_LIR` to capture unpriced residual parity clauses!
- **Pre-Registered Pass Criteria (`R17-E3`):**
  1. **The `w * grad` Sign-Flip Trap Verified:** `A2_GradMag_Taylor_Pruning` exhibits **$\ge 20\%$ sign-flip / mis-ranking rate** against true counterfactual replacement value, retaining $\ge 3$ redundant Group A clones while failing to expand the Group C XOR bottleneck.
  2. **Exact Structural Recovery via Market Wealth & Mitosis:** `A4_VCG_Wealth_Architecture_Market` prunes **`100%` of Group B junk modules (`4/4`)**, prunes **`>= 80%` of excess Group A clones (`4/5` or `5/5`, leaving 1–2 easy-feature modules)**, triggers **mitosis on Group C XOR bottleneck modules (`expanding Group C from 2 to >= 4 active modules`)**, cuts total active compute rent by **$\ge 25\%$**, and achieves the **#1 lowest test `KL` and #1 highest accuracy (`>= 92%`)** across all 4 arms on every seed!
