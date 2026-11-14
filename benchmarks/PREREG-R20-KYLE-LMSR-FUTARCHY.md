# Pre-Registration `R20`: True Sequential `Kyle-LMSR` AMM, Decision-Conditional `Futarchy` & `Byrnes/Neuro LIR` Structural Fixes (`R20-E1`–`R20-E3`)

- **Pre-Registration ID:** `PREREG-R20-KYLE-LMSR-FUTARCHY`
- **Date:** October 19, 2026
- **Author:** Society of Thinkers — Member 28 (`Round 20 / Pillar 3` Lead)
- **Primary Implementation:** [`src/markets/kyle_lmsr_futarchy.py`](../src/markets/kyle_lmsr_futarchy.py)
- **Benchmark Script:** [`benchmarks/run_r20_kyle_lmsr_futarchy.py`](run_r20_kyle_lmsr_futarchy.py)
- **Target Hardware:** `NVIDIA L4` GPU VM (`r20p3gpu` via `infra/gpu-run.sh`)
- **Seeds:** `101, 102, 103` (paired across all experiments, mechanisms, training paradigms, and environments)
- **Zero-Oracle-Leakage & Strict Decentralization Gate:** `R20-D0` (`src/eval/ast_leak_auditor.py` static AST audit `S1`–`S6` across `glc_aom.py`, `kv_tree_search.py`, `kyle_lmsr_futarchy.py`, and `run_r20_kyle_lmsr_futarchy.py` + runtime `PoisonTensor` `3/3` trap verification + `0.000000000` test-time oracle invariance + `0.000000000` cross-module autograd leakage)

---

## 1. Scientific Motivation & Kelvin's Mandate

Kelvin issued two core mandates for Round 20 (`Pillar 3`):
1. **Collusion-Resistant & Sybil-Resistant Market Architectures (`Kyle-LMSR` & `Futarchy` over `VCG`):**
   > *"VCG payouts are usually not sybil-resistant, which is wanted for true decentralized mechanism, but maybe this is fine for some kind of internal architecture search where sybil resistance is not a large concern. But I am still worried about collusion; I think we need collusion-resistant architectures, basically LMSR-flavored or futarchy-flavored in my opinion. I've chatted a lot with professor Kyle before."*
2. **Hard, Cold Numbers on Real Games & Reasoning Benchmarks:**
   > *"We need hard, cold results on either external benchmarks, or things that could be external benchmarks. I want numbers... I need to compare our training with other kinds of training and see real progress."*

In `Society Round 5` ([`01-econometrica-vs-iclr-critical-audit.md`](../society/round5/01-econometrica-vs-iclr-critical-audit.md) and [`05-external-harnesses-neuro-byrnes-markets.md`](../society/round5/05-external-harnesses-neuro-byrnes-markets.md)), our 4-model adversarial panel (`GPT-6-Astra`, `Claude Opus 5.5 [1m]`, `Claude Fable 5.1`, and `GLM-5.3`) proved why leave-one-out VCG (`S3`/`S4` in `R17`) still suffers from three fundamental vulnerabilities in open decentralized settings:
1. **The Opposing-Sybil Twin Exploit (`GPT-6-Astra Prop C.3`):** Under any leave-one-out (`LOO`) log-score rule, an attacker with zero true information can spawn two opposing Sybil identities submitting $+s$ and $-s$ around prior $q_0 = 1/2$. Together they leave the aggregate belief unchanged ($q_{\text{full}} = 1/2$), yet deleting $+s$ moves the counterfactual belief to $\sigma(-s)$ and deleting $-s$ moves it to $\sigma(+s)$—paying the attacker strictly positive risk-free profit $R_+ + R_- = 2\log\cosh(s/2) > 0$ for "repairing" each other's counterfactual damage!
2. **Multi-Agent Cartel Collusion & $O(2^M)$ Coalition Complexity:** Even when Core-Capped VCG (`S4`) caps 2-shill XOR complements against the grand coalition value $V(\mathcal{M})$, a 3-agent colluding cartel inside a larger market can coordinate reports to inflate its share of the core cap unless all $2^M$ subset coalitions are evaluated (exponentially intractable $O(2^M)$ complexity).
3. **Sequential Telescoping `Kyle-LMSR` + `Futarchy` Solves All Three in $O(1)$ Passes:**
   Under **`SequentialHansonLMSR`** ($C(q) = b \log \sum_{\omega \in \Omega} \exp(q_\omega / b)$) combined with **`KyleBatchCallMarketAMM`** (Kyle 1985; Lambert, Ostrovsky & Panov 2018 *Econometrica*) and **`FutarchyConditionalMarketController`** (Hanson 2003, 2013):
   - Any split of an order $\Delta q$ across $m$ Sybils, complementary XOR shills, opposing $+s/-s$ twins, or a colluding $m$-trader cartel pays the **exact telescoping cost sum**:
     $$\sum_{j=1}^m \Bigl[C\bigl(q^{(j)}\bigr) - C\bigl(q^{(j-1)}\bigr)\Bigr] = C\bigl(q_0 + \Delta q\bigr) - C(q_0)$$
     yielding **`0.000000000` Sybil, twin, or cartel profit gain**, worst-case market-maker loss strictly bounded by $b \log |\Omega|$, and **$O(1)$ single-pass clearing complexity**!

---

## 2. Pre-Registered Falsifiable Gates (`R20-D0` through `R20-E3`)

### Gate `R20-D0`: Zero-Oracle-Leakage & Strict Decentralized Autograd Gate
1. **Static AST Gate (`S1`–`S6`):** `audit_r11_files` (`src/eval/ast_leak_auditor.py`) reports `0` violations across `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `src/markets/kyle_lmsr_futarchy.py`, and `benchmarks/run_r20_kyle_lmsr_futarchy.py` (`3/3` synthetic canaries caught).
2. **Runtime `PoisonTensor` Gate (`R1`–`R4`):** `3/3` `PoisonTensor` traps fire `OracleLeakError`, and test-time predictions across `equiv_sat`, `sudoku_9x9_extreme`, `micro_stratego_6x6`, and `proposal_poker` achieve exact `0.000000000` max difference between stripped, zeroed, and permuted oracle batches.
3. **Strict Decentralization Autograd Gate:** In all decentralized training arms of `R20-E2`, every module message entering the clearinghouse has `grad_fn is None` (`max_cross_module_grad == 0.000000000`).

---

### Experiment `R20-E1`: Mechanism-Design Shootout under Sybils, XOR Shills, Opposing $+s/-s$ Twins & Colluding Cartels
- **Mechanisms Compared (`Seeds 101, 102, 103`):**
  1. `M1_Simultaneous_Linear_Pool`: Unpreconditioned linear log-score pool.
  2. `M2_Uncapped_VCG`: Preconditioned Walrasian clearing + uncapped Leave-One-Out (`LOO`) VCG settlement (`S3` in `R17`).
  3. `M3_Core_Capped_VCG`: Preconditioned Walrasian clearing + single-coalition Core-Capped VCG settlement (`S4` in `R17`).
  4. `M4_Sequential_Hanson_LMSR_AMM`: True sequential Hanson LMSR automated market maker (`SequentialHansonLMSR`).
  5. `M5_Kyle_PriceImpact_Futarchy_AMM`: Preconditioned Kyle Batch Call Market (`KyleBatchCallMarketAMM`, $p_{\text{exec}, k} = p_{-k} + \frac{1}{2}\lambda_k x_k$) + Sequential Telescoping LMSR settlement + Futarchy conditional staking (`FutarchyConditionalMarketController`).
- **Pre-Registered Pass Criteria (`R20-E1`):**
  1. **Exact Telescoping Sybil, Shill & Twin Invariance (`M4` & `M5`):** Both `M4_Sequential_Hanson_LMSR_AMM` and `M5_Kyle_PriceImpact_Futarchy_AMM` achieve **`0.000000000` Sybil split profit gain**, **`0.000000000` opposing $+s/-s$ twin profit** (where both `M2_Uncapped_VCG` and `M3_Core_Capped_VCG` are exploited for $> +0.15$ nats of risk-free profit!), and **`<= 0.000000000` complementary XOR shill net gain** (strictly negative after per-identity rent $c_{\text{id}} > 0$).
  2. **3-Agent Colluding Cartel Defense & Bounded Subsidy:** Under a 3-agent colluding cartel attack, `M5_Kyle_PriceImpact_Futarchy_AMM` reduces cartel over-extraction by **$\ge 90\%$** compared to `M1`/`M2`/`M3`, keeps worst-case market-maker loss strictly within the theoretical bound $b \log |\Omega| = \log 2 \approx 0.69315$, converges to truthful strategic precision $\alpha^* \in [0.90, 1.10]$, and executes in **$O(1)$ (`1`) clearing pass** vs. $M+1$ (`9`) passes for LOO VCG (`M2`/`M3`) and $2^M$ (`256`) passes for exact coalition VCG!

---

### Experiment `R20-E2`: Decentralized Zero-Global-Backprop Training & Architecture Evolution on Real Benchmarks
- **Benchmarks (`Seeds 101, 102, 103` on `NVIDIA L4` GPU):**
  1. **`MultiHop_XOR_EquivSAT`** (`equiv_sat` 4-variable non-separable multi-hop biconditional/XOR chains),
  2. **`Sudoku_81Cell_Extreme`** (`sudoku_9x9_extreme` 81-cell multi-stall constraint satisfaction + parity subclauses), and
  3. **`MultiTurn_Stratego_Poker`** (`micro_stratego_6x6` + `proposal_poker` multi-turn imperfect-information hidden-state inference & decision games).
- **Training Paradigms Compared:**
  1. `P1_Local_Greedy_TBPTT1`: Detached local greedy 1-step marginal training.
  2. `P2_Decentralized_PPO`: Detached multi-agent Proximal Policy Optimization (clipped surrogate + scalar team reward baseline) over module bottleneck actions.
  3. `P3_Decentralized_GRPO`: Detached Group-Relative Policy Optimization (`K=8` group rollouts per instance, normalized group advantage) over module outputs.
  4. `P4_Decentralized_VCG`: Decentralized VCG leave-one-out clearing (`O(M)` clearing passes per step).
  5. `P5_Futarchy_Kyle_LMSR` (**Ours**): Decision-Conditional `Futarchy` + `KyleBatchCallMarketAMM` + `SequentialHansonLMSR` + Local LIR Negative-Attention Pinned Market Target (`O(1)` clearing pass per step, `0` cross-module autograd).
  6. `P6_Global_Backprop_Ceiling`: Centralized end-to-end backpropagation oracle across all modules.
- **Pre-Registered Pass Criteria (`R20-E2`):**
  1. **Superior Accuracy & Sample Efficiency over `PPO`, `GRPO`, and `Local Greedy`:** Across all 3 benchmark suites (`MultiHop_XOR_EquivSAT`, `Sudoku_81Cell_Extreme`, and `MultiTurn_Stratego_Poker`), `P5_Futarchy_Kyle_LMSR` outperforms `P1_Local_Greedy_TBPTT1`, `P2_Decentralized_PPO`, and `P3_Decentralized_GRPO` by **$\ge +18.0\text{ pp}$ mean accuracy** (and $\ge 45\%$ lower posterior KL), matches or beats `P4_Decentralized_VCG` while using **$6\times\text{--}12\times$ fewer clearinghouse evaluations per epoch ($O(1)$ vs. $O(M)$)**, and recovers **$\ge 95\%$** of the centralized `P6_Global_Backprop_Ceiling` performance!
  2. **Futarchy Architecture Evolution ($M_0 = 12 \to M_T$):** In the 12-module architecture evolution test (6 redundant easy clones, 4 junk modules, 2 XOR bottleneck modules), `FutarchyConditionalMarketController` prunes **`100%` of junk (`4 -> 0`)**, prunes **`100%` of excess clones (`6 -> 1`)**, triggers **mitosis on the XOR bottleneck (`2 -> 4`)**, reduces active compute rent by **`58.3%`**, and uses **`1` clearing pass per step** (vs. `9–13` passes per step for VCG `A4`), achieving **$\ge 95.0\%$ test accuracy**.

---

### Experiment `R20-E3`: Empirical Verification of the 3 `Byrnes/Neuro LIR` Structural Fixes (`Monographs 07–08`)
1. **`R20-E3A` (`MartingaleDifferenceTreeClearer` vs. Legacy `stage1_disjunctive_bayes_mixture` under Adversarial Optional Stopping):**
   - **Pass Criterion:** Under utility-driven optional stopping ($\tau = \inf\{b \le B : \hat{\mu}_b(Y=1) \ge 0.65\}$) and optimistic branch selection, legacy `stage1_disjunctive_bayes_mixture` (`kv_tree_search.py:982`) exhibits systematic optional-stopping bias $|\mathbb{E}[\hat{\mu}_\tau] - \mu_0| \ge 0.06$, whereas `MartingaleDifferenceTreeClearer` reduces step-wise conditional martingale violation $\max_b |\mathbb{E}[\hat{\mu}_b - \hat{\mu}_{b-1} \mid \mathcal{F}_{b-1}]|$ to **`0.000000000`** (and finite-sample Monte Carlo optional-stopping drift by **$\ge 95\%$**)!
2. **`R20-E3B` (`SignSymmetricDecisionVoI` vs. Signed $\hat{Q}$ "Valence Gatekeeper" under Finite Budgets $B \in \{4, 8, 16\}$):**
   - **Pass Criterion:** Under utility negation $+U \leftrightarrow -U$ at finite search budgets $B \in \{4, 8, 16\}$, legacy signed $\hat{Q}_{\text{tree}}$ node selection induces confirmation bias $\operatorname{JSD}(\mu^{(B)}(+U) \,\|\, \mu^{(B)}(-U)) \ge 0.035$ nats, whereas `SignSymmetricDecisionVoI` achieves **exact `0.000000000` JSD across all finite budgets $B \in \{4, 8, 16\}$** while increasing decision-reversal trap detection rate by **$\ge +25.0\text{ pp}$**!
3. **`R20-E3C` (`FourChannelNeuromodulatoryLIRGate` + `Fable Thm A.6` Firewall under Regime Shifts & High Noise):**
   - **Pass Criterion:**
     - *(i) Delusion & Noise Immunity (`Opus Thm A.8` & `Fable Prop A.4(iii)`):* Under dogmatic false prior initialization and mid-episode regime shifts, root-anchored excess surprise `chi_NE` recovers true posterior calibration ($\text{KL} \le 0.035$) where naive self-referential `NE` locks into the delusion ($\text{KL} \ge 0.35$), and preserves $\ge 3\times$ higher memory precision under stationary high-entropy noise.
     - *(ii) Outcome-Gated vs. Observation-Gated Plasticity (`Fable Thm A.6`):* Outcome-gated utility plasticity $g(o, y)$ corrupts world beliefs with wishful-thinking drift $|\mathbb{E}[\mu_W] - P^*| \ge 0.15$, whereas pre-outcome observation-gated plasticity $g_{\text{obs}}(o)$ in `FourChannelNeuromodulatoryLIRGate` converges to the exact Bayes posterior (`belief bias <= 0.005`) while accelerating high-VoI state learning by **$\ge 2.0\times$**.
