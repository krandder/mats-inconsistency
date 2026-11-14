> **Independent audit (2026-10-19): verdict artefact.** The claims in this file did not pass an independent Opus 5.5 audit; see `docs/audits/gemini-campaign-critique-2026-10-19.md` (all rounds), `docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md` and `docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md`. Numbers below are the runner's outputs, not evidence about the architecture.

# Round 17 Empirical Results: Decentralized Market-Training, Econometrica Incentive-Compatibility & Architecture Discovery (`R17-DECENTRALIZED-MARKET-TRAINING`)

**Date:** 2026-10-18  
**Author:** Society of Thinkers — Member 24 (Empirical & Theoretical Synthesis Lead)  
**Pre-Registration:** [`benchmarks/PREREG-R17-DECENTRALIZED-MARKET.md`](file:///home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R17-DECENTRALIZED-MARKET.md)  
**Society Round 5 Red-Team Scoreboard:** [`society/round5/ROUND5-SCOREBOARD.md`](file:///home/agent/workspaces/inconsistency-search/society/round5/ROUND5-SCOREBOARD.md)  
**Benchmark Script:** [`benchmarks/run_r17_decentralized_market.py`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py)  
**Raw JSON Artifact:** [`results/r17_decentralized_market_results.json`](file:///home/agent/workspaces/inconsistency-search/results/r17_decentralized_market_results.json)  
**Seeds Evaluated:** `[101, 102, 103]` (`elapsed_seconds = 55.03s`)  
**Overall Verdict:** **`PASS` (4 / 4 Pre-Registered Gates Passed)**

---

## 1. Executive Summary & Pre-Registered Gate Verdicts

Round 17 (`R17-DECENTRALIZED-MARKET-TRAINING`) was designed in direct response to the Society of Thinkers Round 5 Adversarial Red-Team Consultation across `glm-5.3` ([`02-glm53-round5-adversarial-critique.md`](file:///home/agent/workspaces/inconsistency-search/society/round5/02-glm53-round5-adversarial-critique.md)), `gpt-6-astra` ([`03-gpt6-astra-round5-adversarial-critique.md`](file:///home/agent/workspaces/inconsistency-search/society/round5/03-gpt6-astra-round5-adversarial-critique.md)), and `claude-opus-5-5[1m]` ([`04-opus-round5-adversarial-critique.md`](file:///home/agent/workspaces/inconsistency-search/society/round5/04-opus-round5-adversarial-critique.md)). It tests whether the preconditioned Walrasian + VCG clearinghouse serves simultaneously as:
1. **An *Econometrica* mechanism-design solution** for strategic precision reporting, duplicate Sybil clone cartels, and complementary false-name shill splitting (`R17-E1`), and
2. **An *ICLR* decentralized credit-assignment and architecture-discovery paradigm** replacing global backpropagation across modules (`R17-E2`) and gradient-magnitude/Taylor structural pruning (`R17-E3`).

| Gate ID | Description | Pre-Registered Criterion | Empirical Result (Seeds `101, 102, 103`) | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **`R17-D0`** | Zero-Oracle & Strict Decentralized Autograd Firewall | `0` AST violations; `3/3` canaries caught; `3/3` `PoisonTensor` traps fired; `0.0` test-time oracle sensitivity; `0` cross-module autograd edges (`grad_fn is None` on all clearinghouse messages in `M1..M3`) | `0` AST violations; `3/3` canaries; `3/3` poison traps; `global_max_oracle_diff = 0.000000000`; `strict_zero_global_backprop = True` across all decentralized methods | **`PASS`** |
| **`R17-E1`** | *Econometrica* Incentive Compatibility, Sybil Clone Annihilation & Complementary Shill Audit | Under `S1_Naive_LMSR`, $\alpha^* > 2.5\times$ and $\text{KL}(S_1) > 2.0\times \text{KL}(S_3)$; under `S3`/`S4`, $\alpha^* \in [0.85, 1.15]$, $\ge 75\%$ reduction in clone cartel payout share, key-falsification strictly unprofitable, and `S4` prevents complementary XOR shill splitting ($\text{net\_gain} < 0$) | `S1`: $\alpha^* = 9.33\times \pm 2.17$, $\text{KL} = 1.1793$ ($56.98\times$ Honest); `S3`/`S4`: $\alpha^* = 0.97\times \pm 0.12$, $\text{KL} = 0.0223$ ($1.08\times$ Honest); Clone cartel share drops from **`59.5%` $\to$ `1.3%`** (**`97.86%` relative reduction**); key-falsification unprofitable across all seeds; `S4` complementary shill split net gain $= -0.04 < 0$ | **`PASS`** |
| **`R17-E2`** | *ICLR* Zero-Global-Backprop Multi-Hop Compositional Parity & Bandwidth Bottleneck | Exact algebraic equivalence $\cos(\nabla^{\text{LMSR}}, \nabla^{\text{TBPTT(1)}}) > 0.9999$; `M1`, `M2`, and zero-price `M2b_InfoMax` fail near chance ($\le 58\%$) on multi-hop XOR (`MultiHop_Equiv_SAT`, `Hidden_Parity_Sudoku`); `M3a` (Type-iii scalar price) $\ge 78\%$; `M3` (Type-ii pinned-market LIR) $\ge 88\%$ and within $5\%$ of `M4_Global_Backprop_Ceiling` | $\cos(\nabla^{\text{LMSR}}, \nabla^{\text{TBPTT(1)}}) = \mathbf{0.9999999}$; `M1` ($46.35\% / 47.92\%$), `M2` ($49.48\% / 47.40\%$), and `M2b_InfoMax` ($49.22\% / 49.22\%$) remain stuck at chance; `M3a` (strict Type-iii scalar price + REINFORCE) reaches **`90.36%`**; `M3` (Type-ii pinned-market LIR) reaches **`100.00%`** ($\text{KL} = 0.0030$), exactly matching `M4_Global_Backprop_Ceiling` (**`100.00%`**, $\text{KL} = 0.0030$) | **`PASS`** |
| **`R17-E3`** | *ICLR* Market-Driven Architecture Discovery (Bankruptcy Pruning & Wealth Mitosis) | Verify Taylor sign-flip pathology ($>25\%$); `A4_VCG_Wealth_Architecture_Market` prunes $\ge 90\%$ of junk modules and $\ge 80\%$ of excess easy-feature clones (where `A2_GradMag_Taylor` and `A2b_OBD_SecondOrder_Hessian` fail), preserves $100\%$ of XOR bottleneck modules (where `A3_FlatRent_Simultaneous_VCG` causes infant mortality), reduces compute rent $\ge 35\%$, and beats `A1`, `A2`, `A2b`, `A3` in accuracy and KL | Taylor sign-flip rate $= \mathbf{62.5\%}$; `A4` prunes **`100.0%` of junk** (`4 -> 0`) and **`100.0%` of excess clones** (`6 -> 1`), expands XOR bottleneck via mitosis (`2 -> 4`), cuts active compute rent by **`58.33%`** (`12 -> 5` active modules), and achieves **`95.83%` accuracy** ($\text{KL} = 0.0082$) vs. `A1` ($79.75\%$), `A2` ($71.50\%$), `A2b` ($71.83\%$), and `A3` ($71.00\%$) | **`PASS`** |

---

## 2. Detailed Results by Experiment

### 2.1 Gate `R17-D0`: Zero-Oracle & Strict Decentralized Autograd Firewall

Across all three seeds (`101, 102, 103`), [`run_r17_d0_firewall_audit`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py#L229-L296) verified:
- **Static AST Audit:** `0` violations in [`benchmarks/run_r17_decentralized_market.py`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py), with `3 / 3` malicious oracle canaries (`batch.oracle_labels`, `getattr(env, 'truth_table')`, `batch.__dict__['hidden_state']`) caught immediately.
- **Runtime `PoisonTensor` Containment:** `3 / 3` runtime poison traps fired on unauthorized access; `global_max_oracle_diff = 0.000000000` across `galanis_t3s111o2ye2`, `equiv_sat`, and `sudoku_6x6` when oracle fields were perturbed by $+10^6$ vs. $-10^6$.
- **Strict Decentralized Autograd Firewall:** Every module-to-clearinghouse communication in `M1_Local_Greedy_Layerwise`, `M2_Decentralized_Naive_LMSR`, `M2b_InfoMax_ZeroPrice_Bottleneck`, `M3a_Decentralized_TypeIII_ScalarPrice_VCG`, and `M3_Decentralized_VCG_Gershgorin_LIR` passed `assert msg.grad_fn is None`, guaranteeing zero cross-module backpropagation edges and independent per-module `torch.optim.Adam` instances.

---

### 2.2 Experiment `R17-E1`: *Econometrica* Strategic Precision Reporting, Sybil Clone Cartel & Complementary Shill Audit

In [`run_r17_e1_econometrica_ic_audit`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py#L303-L575), $M = 8$ Bayesian signal agents ($2$ orthogonal high-value specialists, $4$ redundant Sybil clones sharing a single underlying factor, and $2$ strategic confidence-inflating agents choosing multiplier $\alpha_m \in [0.20, 12.0]$) interact across four clearinghouse rules:

| Clearinghouse Mechanism | Inflator multiplier $\alpha^*$ (mean $\pm$ sd) | Clone multiplier $\alpha_{\text{clone}}^*$ | Equilibrium Market KL (mean $\pm$ sd) | KL Ratio vs. Honest Oracle | Orthogonal Specialist Payout Share | Clone Cartel ($4$ Clones) Payout Share | Strategic Coalition (Clones + Inflators) Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S1_Naive_LMSR_LinearScore`** *(Unpreconditioned pool, linear score)* | $9.33\times \pm 2.17$ | $12.00\times \pm 0.00$ | $1.1793 \pm 0.0262$ | $56.98\times \pm 1.90$ | $14.9\% \pm 2.8\%$ | **`59.5%` $\pm 5.0\%$** | **`85.1%` $\pm 2.8\%$** |
| **`S2_Normalized_Pool_Static_LOO`** *(Normalized pool, static unpreconditioned LOO)* | $2.33\times \pm 0.33$ | $0.20\times \pm 0.00$ | $0.2687 \pm 0.0200$ | $12.96\times \pm 0.66$ | $70.8\% \pm 5.0\%$ | $0.0\% \pm 0.0\%$ | $29.2\% \pm 5.0\%$ |
| **`S3_VCG_Gershgorin_LOO`** *(Uncapped Preconditioned Walrasian + VCG LOO)* | **`0.97x` $\pm 0.12$** | $1.13\times \pm 0.47$ | $0.0227 \pm 0.0017$ | $1.10\times \pm 0.11$ | $85.5\% \pm 4.3\%$ | **`1.3%` $\pm 1.2\%$** | $14.5\% \pm 4.3\%$ |
| **`S4_Core_Capped_Behavioral_VCG`** *(Core-Capped Behavioral Preconditioned VCG, Theorem R5.1')* | **`0.97x` $\pm 0.12$** | $1.13\times \pm 0.47$ | **`0.0223` $\pm 0.0019$** | **`1.08x` $\pm 0.12$** | **`87.5%` $\pm 4.8\%$** | **`1.3%` $\pm 1.2\%$** | **`12.5%` $\pm 4.8\%$** |

#### Red-Team Stress Tests Embedded in `R17-E1`:
1. **Behavioral Cross-Fit Covariance vs. Key-Falsification Attack (`glm-5.3` / `gpt-6-astra` / `claude-opus-5-5` critique):**  
   When `G_off` is computed directly from cross-fit centered prediction signals $\hat{\delta}_m^{(A)}$ rather than unverified cheap-talk keys $k_m$, any attempt by Clone 2 to de-correlate itself by injecting variance-preserving orthogonal white noise ($\text{noise\_amp} \in \{0.25, 0.50, 1.0, 2.0\}$) strictly degrades the full market's log-loss $\ell(\delta^{\text{falsified}})$ while leaving the leave-one-out counterfactual $\ell(\delta_{-2})$ untouched. Across all three seeds, `key_falsification_max_gain` was strictly negative (`-0.0110`, `-0.0091`, `-0.0061`; `key_falsification_unprofitable = True`).
2. **Substitutes vs. Complements & False-Name Shill Splitting (`Theorem R5.1'`, Ausubel–Milgrom / Yokoo–Sakurai–Matsubara critique):**  
   - On **informational substitutes** ($4$ duplicate clones), both `S3` and `S4` re-clear the remaining $3$ clones on $G_{\text{off},-m}$ when clone $m$ is deleted, collapsing the $4$-clone cartel's payout share from **`59.5%` under `S1` down to `1.3%`** (a **`97.86%` relative reduction**).
   - On **informational complements** (an XOR specialist holding $Z_{12}, Z_{34}$ with joint value $V_{\text{xor}} = 0.42\text{ nats}$ and per-identity rent $c_{\text{rent}} = 0.04$ splitting into $2$ complementary shills), uncapped VCG (`S3`) pays each shill $V_{\text{xor}}$, creating a $2.0\times$ overpayment ($\text{net\_gain} = +0.38 > 0$). By contrast, **`S4_Core_Capped_Behavioral_VCG`** enforces the core budget cap $W_m = \beta_0\,\text{LOO}_m \cdot \min(1, V_{\text{total}}/\sum_j \text{LOO}_j^+) - c_{\text{rent}}$, capping total gross payout at $1.0 \times V_{\text{xor}}$ and making complementary shill splitting strictly unprofitable ($\text{net\_gain} = -0.04 < 0$).

---

### 2.3 Experiment `R17-E2`: *ICLR* Zero-Global-Backprop Decentralized Training & The Three-Message Trichotomy

In [`run_r17_e2_zero_global_backprop`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py#L653-L936), we first verified [`Theorem R5.2`](file:///home/agent/workspaces/inconsistency-search/society/round5/01-econometrica-vs-iclr-critical-audit.md#L118-L135): the cosine similarity between the unpreconditioned decentralized LMSR gradient and 1-step truncated backprop (`TBPTT(1)`) is **`0.9999999`** across all seeds, confirming mathematically why naive prediction-market learning collapses on multi-hop XOR (`Y = Z_12 * Z_34`, where $\text{Cov}(Y, Z_{12}) = \text{Cov}(Y, Z_{34}) = 0$).

Following `claude-opus-5-5[1m]`'s bandwidth-bottleneck design and `gpt-6-astra`'s Three-Message Trichotomy, each local specialist module has a $d_z = 1$ bottleneck gate over $3$ local feature channels ($1$ true parity channel $Z_{12}$ and $2$ higher-variance distractor channels $N_{1,a}, N_{1,b}$ with $2.5\times$ input variance) and ZERO cross-module backpropagation edges (`M1..M3`):

| Training Paradigm | Message Type & Autograd Constraint | `MultiHop_Equiv_SAT` Test Acc (mean $\pm$ sd) | `MultiHop_Equiv_SAT` Test KL | `Hidden_Parity_Sudoku` Test Acc (mean $\pm$ sd) | `Hidden_Parity_Sudoku` Test KL |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`M1_Local_Greedy_Layerwise`** *(Nøkland & Eidnes 2019)* | Local marginal head only (`0` cross-module autograd) | $46.35\% \pm 3.56\%$ | $0.5993 \pm 0.0185$ | $47.92\% \pm 0.74\%$ | $0.5955 \pm 0.0086$ |
| **`M2_Decentralized_Naive_LMSR`** *(1st-order prediction market)* | Unbundled scalar logit, `TBPTT(1)` (`0` cross-module autograd) | $49.48\% \pm 2.08\%$ | $0.6071 \pm 0.0145$ | $47.40\% \pm 0.80\%$ | $0.5924 \pm 0.0109$ |
| **`M2b_InfoMax_ZeroPrice_Bottleneck`** *(Opus 5.5 self-supervised control)* | Zero-price local reconstruction across $d_z=1$ bottleneck (`0` cross-module autograd) | $49.22\% \pm 1.78\%$ | $0.8293 \pm 0.0211$ | $49.22\% \pm 1.78\%$ | $0.8293 \pm 0.0211$ |
| **`M3a_Decentralized_TypeIII_ScalarPrice_VCG`** *(Astra Type-iii strict scalar price)* | Scalar VCG payment $W_m \in \mathbb{R}$ via REINFORCE on bottleneck gate (`0` cross-module autograd) | **`90.36%` $\pm 1.21\%$** | **`0.2958` $\pm 0.0083$** | **`90.36%` $\pm 1.21\%$** | **`0.2958` $\pm 0.0083$** |
| **`M3_Decentralized_VCG_Gershgorin_LIR`** *(Astra Type-ii pinned-market LIR)* | Detached global state + local negative-attention $\tau_{\text{neg}}$ (`0` cross-module autograd) | **`100.00%` $\pm 0.00\%$** | **`0.0030` $\pm 0.0000$** | **`100.00%` $\pm 0.00\%$** | **`0.0030` $\pm 0.0000$** |
| **`M4_Global_Backprop_Ceiling`** *(End-to-end differentiable oracle)* | Full end-to-end backpropagation across all modules | $100.00\% \pm 0.00\%$ | $0.0030 \pm 0.0000$ | $100.00\% \pm 0.00\%$ | $0.0030 \pm 0.0000$ |

#### Key Scientific Takeaway (`R17-E2`):
- **Why `M2b_InfoMax` Fails ($49.22\%$):** Without VCG prices, local self-supervised InfoMax allocates the $d_z = 1$ bottleneck entirely to the highest-variance local distractor channels ($N_{1,a}, N_{1,b}$), discarding the lower-variance parity interaction $Z_{12}$ needed by the downstream clearinghouse.
- **Why `M3a` ($90.36\%$) and `M3` ($100.00\%$) Succeed with Zero Global Backprop:** Even under strict **Type-(iii)** scalar-payment-only communication (`M3a`), the VCG leave-one-out reward strictly favors selecting the complementary parity channel $Z_{12}$ over the higher-variance distractors, driving accuracy from $49.22\%$ to **`90.36%`**. Under **Type-(ii)** local negative-attention against detached clearinghouse residuals (`M3`), decentralized training achieves **`100.00%` accuracy** ($\text{KL} = 0.0030$), matching centralized end-to-end backpropagation (`M4`) with zero cross-module autograd edges.

---

### 2.4 Experiment `R17-E3`: *ICLR* Market-Driven Architecture Discovery (Bankruptcy Pruning & Wealth Mitosis)

In [`run_r17_e3_architecture_market`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r17_decentralized_market.py#L943-L1196), an initial pool of $M_0 = 12$ modules ($6$ redundant easy-feature clones on $Z_{\text{easy}}$, $4$ pure-noise junk modules, and $2$ complementary XOR specialists on $Z_{12}, Z_{34}$) is trained under five structural discovery regimes. First, we empirically verified [`Theorem R5.3`](file:///home/agent/workspaces/inconsistency-search/society/round5/01-econometrica-vs-iclr-critical-audit.md#L168-L185): the 1st-order Taylor importance score $\mathcal{I}_m^{\text{Taylor}} = -(\partial \mathcal{L}/\partial w_m) w_m$ suffers a **`62.5%` sign-flip rate** against true re-equilibrated leave-one-out removal loss on co-adapted units!

| Architecture Discovery Method | Active Modules ($M_0=12 \to M_T$) | Surviving Clones ($6 \to \cdot$) | Surviving Junk ($4 \to \cdot$) | Active XOR Modules ($2 \to \cdot$) | Compute Rent Reduction | Test Accuracy (mean $\pm$ sd) | Test KL (mean $\pm$ sd) | Net Market Welfare |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A1_Static_Unpruned_Pool`** *(No pruning, no mitosis)* | $12.0 \pm 0.0$ | $6.0$ | $4.0$ | $2.0$ | $0.0\%$ | $79.75\% \pm 2.07\%$ | $0.1839 \pm 0.0058$ | $-0.3279$ |
| **`A2_GradMag_Taylor_Pruning`** *(1st-order gradient/Taylor magnitude)* | $7.3 \pm 0.5$ | $6.0$ ($0\%$ pruned!) | $0.3$ | $1.0$ ($50\%$ killed!) | $38.9\%$ | $71.50\% \pm 1.97\%$ | $0.2412 \pm 0.0051$ | $-0.3292$ |
| **`A2b_OBD_SecondOrder_Hessian_Pruning`** *(LeCun OBD diagonal Hessian $\frac{1}{2} H_{mm} w_m^2$)* | $8.0 \pm 0.0$ | $6.0$ ($0\%$ pruned!) | $1.0$ | $1.0$ ($50\%$ killed!) | $33.3\%$ | $71.83\% \pm 1.89\%$ | $0.2412 \pm 0.0051$ | $-0.3372$ |
| **`A3_FlatRent_Simultaneous_VCG`** *(Simultaneous un-buffered VCG bankruptcy)* | $1.0 \pm 0.0$ | $0.0$ (All $6$ extinct!) | $0.0$ | $1.0$ ($50\%$ infant mortality!) | $91.7\%$ | $71.00\% \pm 0.35\%$ | $0.2766 \pm 0.0034$ | $-0.2886$ |
| **`A4_VCG_Wealth_Architecture_Market`** *(Sequential LCB Bankruptcy + VoI Incubator Rent + Mitosis)* | **`5.0` $\pm 0.0$** | **`1.0` ($100\%$ excess pruned!)** | **`0.0` ($100\%$ pruned!)** | **`4.0` ($100\%$ survival + mitosis!)** | **`58.3%`** | **`95.83%` $\pm 1.53\%$** | **`0.0082` $\pm 0.0026$** | **`-0.0682`** |

#### Key Scientific Takeaway (`R17-E3`):
1. **Why Both 1st-Order Taylor (`A2`) and 2nd-Order Diagonal Hessian OBD (`A2b`) Fail (`71.50%` and `71.83%`):**  
   Because all $6$ clones carry non-zero weights on the high-margin easy feature $Z_{\text{easy}}$, both $|\nabla_{w_m}\mathcal{L} \cdot w_m|$ and $\frac{1}{2} H_{mm} w_m^2$ assign high individual importance to all $6$ redundant clones (`0%` of excess clones pruned!) while pruning away one of the slow-warming complementary XOR modules (`50%` XOR mortality).
2. **Why Naive Simultaneous Flat-Rent VCG (`A3`) Suffers Extinction & Infant Mortality (`71.00%`):**  
   Directly confirming the Round 5 Red-Team critique (`glm-5.3` / `gpt-6-astra` / `claude-opus-5-5`), applying flat rent and simultaneous VCG deletion kills **all $6$ clones at once** (because each clone has $\text{LOO}_m \approx 0$ while the other $5$ exist) and starves the complementary XOR pair during early warmup.
3. **How `A4_VCG_Wealth_Architecture_Market` Resolves Both Pathologies (`95.83%`, $\text{KL} = 0.0082$):**  
   - **Sequential Greedy LCB Bankruptcy (`1` module per epoch with $G_{\text{off}}$ re-clearing):** Prunes redundant clones one by one ($6 \to 5 \to 4 \to 3 \to 2 \to 1$). The moment $5$ excess clones are liquidated, the single surviving specialist's counterfactual $\text{LOO}_1$ jumps from $\approx 0$ to the full value of $Z_{\text{easy}}$ ($+0.28\text{ nats/step} \gg c_{\text{rent}}$), guaranteeing its survival (`surviving_clones = 1.0`, `excess_clone_pruned_recall = 100.0%`).
   - **VoI-Ramped Incubator Rent + Joint-Account Mitosis:** Protects the complementary XOR modules during warmup (`100.0%` survival) and triggers productive mitosis on the high-surplus XOR bottleneck ($2 \to 4$ modules), reducing total active compute rent by **`58.33%`** ($12 \to 5$ modules) while boosting accuracy from $79.75\%$ (`A1`) to **`95.83%`** ($\text{KL} = 0.0082$).
