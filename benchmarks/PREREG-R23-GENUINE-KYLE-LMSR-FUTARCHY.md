# Pre-Registration Protocol: Round 23 (`R23-GENUINE-KYLE-LMSR-FUTARCHY`) — 100% Real-`EnvBatch` Decentralized Neural `Kyle-LMSR AMM`, `Futarchy` Evolution & `Byrnes/Neuro` Search Benchmark on `NVIDIA L4` GPU

- **Document ID:** `PREREG-R23-GENUINE-KYLE-LMSR-FUTARCHY`
- **Status:** **LOCKED PRIOR TO ROUND-23 GPU EXECUTION (`A-PURE-NEURAL-SYMMETRIC`)**
- **Supersedes:** `PREREG-R20-KYLE-LMSR-FUTARCHY` (`WAVE-1`, where `R20-E2` was flagged by Round-6 Forensic Code Audit [`society/round6/01-round6-forensic-code-audit-of-r18-r21.md`](../society/round6/01-round6-forensic-code-audit-of-r18-r21.md) Finding 3 for using a synthetic 6-bit parity surrogate and a `sym_anchor_1` warmup anchor)
- **Date:** 2026-10-19
- **Author:** Society of Thinkers — Member 31 (Round-23 Genuine Decentralized Market & Neuro-LIR Runner)
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Static Enforcement Gate:** [`src/eval/ast_leak_auditor.py`](../src/eval/ast_leak_auditor.py) (`v2`, Rules `S1`–`S10`, `0` violations required, `7/7` canaries caught)
- **Primary Market & Neuro-LIR Library:** [`src/markets/kyle_lmsr_futarchy.py`](../src/markets/kyle_lmsr_futarchy.py)
- **Benchmark Script:** [`benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py`](run_r23_genuine_kyle_lmsr_futarchy.py)
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB` VRAM) via `infra/gpu-run.sh` (`--run-id r23gpu`)
- **Pre-Registered Seeds:** `101, 102, 103` ($N=3$ independent training & evaluation seeds)

---

## 0. Forensic Rectification Mandate (`R20-E2` Rejection $\to$ `R23` 100% Real-`EnvBatch` Execution)

In Round 6 ([`society/round6/01-round6-forensic-code-audit-of-r18-r21.md`](../society/round6/01-round6-forensic-code-audit-of-r18-r21.md), Finding 3), our line-by-line forensic code audit confirmed that while `R20-E1` and `R20-E3` were genuine mathematical/numerical verifications of `SequentialHansonLMSR`, `KyleBatchCallMarketAMM`, `MartingaleDifferenceTreeClearer`, `SignSymmetricDecisionVoI`, and `FourChannelNeuromodulatoryLIRGate`, **`R20-E2` (`build_real_benchmark_decentralized_dataset` in `run_r20_kyle_lmsr_futarchy.py` lines `656–754` & `978`)** was invalid as a real-benchmark neural training claim because:
1. It only extracted a scalar `env_sig = float(blind_b.obs_features.std())` from `sudoku_9x9_extreme` / `micro_stratego_6x6` and then generated a synthetic 6-bit parity problem (`bits = torch.randint(0, 2, (n_total, 6))`).
2. At line `978`, it used a warmup anchor (`sym_anchor_1 = cand_tr[:, 0, 0] + cand_tr[:, 1, 0]`) banned under `ast_leak_auditor.py v2` Rule `S8`.

**Round `R23` (`A-PURE-NEURAL-SYMMETRIC`) rectifies this completely under `src/eval/ast_leak_auditor.py v2` (`S1`–`S10`, `0` violations):**
- **Zero synthetic parity bits (`torch.randint(0, 2, (n_total, 6))` deleted):** Every module in `R23-E1` and `R23-E2` reads directly from real `EnvBatch` tensors (`blind_batch.obs_tokens`, `blind_batch.obs_mask`, `blind_batch.obs_features`, `blind_batch.memory_tokens`) sampled from actual registered environments (`equiv_sat`, `sudoku_6x6`, `galanis_market_t3s111o2ye2`, `micro_stratego_4x4`) and predicts the actual masked ground-truth game variables (`oracle_targets["target_tokens"]`).
- **Zero `sym_anchor_1` warmup cheats:** In `M4_Decentralized_Kyle_LMSR_Pinned_LIR` and `M5_Decentralized_Kyle_LMSR_Futarchy_Evolution`, each module $m$ optimizes its local proper-scoring / Negative-Attention Pinned-Market LIR loss directly against the **detached pre-entry sequential LMSR market state $q^{(m-1)}_{\text{detached}}$** (`grad_fn is None`) vs. the post-settlement target on masked game variables, which is strictly convex in module $m$'s log-odds order $\Delta \ell_m$ from epoch 0 without any warmup anchor!
- **Real Multi-Step Search Rollouts in `R23-E3`:** In `R23-E3`, `MartingaleDifferenceTreeClearer`, `SignSymmetricDecisionVoI`, and `FourChannelNeuromodulatoryLIRGate` are evaluated end-to-end on real multi-step search rollouts (`env.step_unmask` + `GLCAOMWorldModel` forward passes) across `proposal_poker`, `shiphunt_6x6`, and `sudoku_6x6`.

---

## 1. Pre-Registered Experimental Protocol (`R23-D0` through `R23-E3`)

### Gate `R23-D0`: Static AST (`S1`–`S10`) & Runtime `PoisonTensor` + Autograd Firewall Gate
1. **Static AST Audit (`v2`, Rules `S1`–`S10`):** `audit_r11_files` passes with **`0` violations** and **`7/7` synthetic canaries caught** across `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `src/markets/kyle_lmsr_futarchy.py`, and `benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py`.
2. **Runtime `PoisonTensor` Verification (`3/3` traps fired, `0.000000000` test-time oracle invariance):** All `EnvBatch` instances passed to any neural module or search controller are stripped via `strip_oracle_fields(batch)`, replacing `target_tokens`, `exact_marginals`, and `query_targets` with `PoisonTensor` guards.
3. **Strict Decentralized Autograd Firewall (`max_cross_module_grad == 0.000000000`):** In all decentralized training regimes (`M2`, `M3`, `M4`, `M5`), every tensor crossing a module boundary into the market clearinghouse satisfies `tensor.grad_fn is None and not tensor.requires_grad`, verified at runtime on every training step.

---

### Experiment `R23-E1`: `Kyle-LMSR-Futarchy` Sybil, Collusion & Information-Aggregation Shootout on Real Game Feature Views
- **Environments (`Seeds 101, 102, 103`):**
  1. `equiv_sat` (`N_vars = 16`, binary parity chains),
  2. `sudoku_6x6` (`N_vars = 36`, 6x6 constraint satisfaction grid),
  3. `galanis_market_t3s111o2ye2` (`N_vars = 4`, interactive Aumann partition prediction market).
- **Partitioned Real Feature Views:**
  - **4 Tier-1 Local Clue Modules (`m = 0..3`):** Trained on the `train` split of each environment reading disjoint spatial/variable subsets of single-variable clues (`blind.obs_tokens` and `blind.obs_mask`).
  - **4 Tier-2 Constraint Interaction Modules (`m = 4..7`):** Trained on the `train` split of each environment reading disjoint pairwise constraint / interaction channels from `blind.obs_features` (and `blind.memory_tokens` in `galanis_market_t3s111o2ye2`).
  - **Sybil Duplication ($k \in \{1, 2, 4, 8\}$) & 3-Module Colluding Coalition:** Tier-1 module `0` spawns $k \in \{1, 2, 4, 8\}$ Sybil clones splitting/duplicating its real-game log-odds signal, and a 3-module cartel executes a coordinated pivot-masking / opposing-twin attack (`+s / -s`).
- **7 Settlement Rules Evaluated:**
  1. `S1_Linear_Pool` (Simultaneous linear opinion pool)
  2. `S2_Simultaneous_Raw_VCG` (Uncapped leave-one-out log-score VCG)
  3. `S3_Bootstrap_Shapley` (Permutation-sampled coalition Shapley log-score)
  4. `S4_Core_Capped_VCG` (Grand-coalition core-capped leave-one-out VCG)
  5. `S5_Sequential_Hanson_LMSR` (`SequentialHansonLMSR` path-integral AMM)
  6. `S6_Kyle_Batch_Call_AMM` (`KyleBatchCallMarketAMM` with behavioral Gram covariance $G_{\text{off}}$ + Kyle price impact + sequential LMSR settlement)
  7. `S7_Futarchy_Conditional_Market` (`FutarchyConditionalMarketController` + `KyleBatchCallMarketAMM` with endogenous conditional reputation weighting)
- **Metrics Reported:** Exact Sybil invariance $\Delta_{\text{Sybil}}(k)$ across $k \in \{1, 2, 4, 8\}$, coalition collusion gain $\Delta_{\text{Collude}}$, opposing `+s/-s` twin exploit, held-out belief accuracy, Brier score, and KL on the real game targets.

---

### Experiment `R23-E2`: 100% Real-Game Decentralized Modular Neural Training (Zero Autograd Across Module Boundaries, Zero Synthetic Parity, Zero `sym_anchor_1`)
- **Environments (`Seeds 101, 102, 103`):**
  1. `sudoku_6x6` (predicting masked cell digits `1..6` on `~obs_mask`),
  2. `equiv_sat` (predicting masked variable bits `1..2` on `~obs_mask`),
  3. `micro_stratego_4x4` (predicting hidden opponent piece types `1..6` on unrevealed slots `~obs_mask`).
- **12-Module Neural Society (`M = 12` independent PyTorch `nn.Module`s, each with its own `AdamW` optimizer):**
  - `m = 0..3`: **4 Tier-1 Local Observation MLP/Attention Modules** reading `blind.obs_tokens` and `blind.obs_mask` (with quadrant/subset focus).
  - `m = 4..7`: **4 Tier-2 Bilinear/Constraint Interaction Modules** reading `blind.obs_features` (and `blind.memory_tokens`) interaction channels.
  - `m = 8..11`: **4 Redundant/Colluding Sybil Clone Modules** duplicating the Tier-1 `obs_tokens` views.
- **5 Training Regimes Compared:**
  1. `M1_Global_Backprop_Oracle`: End-to-end centralized autograd ceiling across all 12 modules.
  2. `M2_Decentralized_Raw_VCG_REINFORCE`: Strictly detached modules (`grad_fn is None`) trained via REINFORCE with simultaneous raw LOO VCG payouts under Sybil/collusion crowding.
  3. `M3_Decentralized_Sequential_LMSR_REINFORCE`: Strictly detached modules (`grad_fn is None`) trained via scalar REINFORCE using `SequentialHansonLMSR` / `KyleBatchCallMarketAMM` scalar net profits.
  4. `M4_Decentralized_Kyle_LMSR_Pinned_LIR`: Strictly detached modules (`grad_fn is None`) where each module $m$ optimizes its local proper-scoring / Negative-Attention Pinned-Market LIR loss against the detached pre-entry sequential LMSR market quote $q^{(m-1)}_{\text{detached}}$ (`p^{(m-1)}`) vs. post-settlement target on masked game variables, plus endogenous Kyle-LMSR wealth allocation $\beta_m^{(t+1)}$.
  5. `M5_Decentralized_Kyle_LMSR_Futarchy_Evolution`: `M4` + periodic Futarchy bankruptcy replacement of the lowest-wealth Sybil clones (`m in {8..11}`) by newly initialized Tier-2 constraint interaction specialists (with **zero `sym_anchor_1` warmup cheats**).

---

### Experiment `R23-E3`: End-to-End Integration of the 3 `Byrnes/Neuro` Mechanisms on Real Multi-Step Search Trees
Evaluated on real multi-step search rollouts (`env.step_unmask` + trained `GLCAOMWorldModel`) across **`proposal_poker`**, **`shiphunt_6x6`**, and **`sudoku_6x6`**:
1. **`R23-E3A` (`MartingaleDifferenceTreeClearer`):** Measures optional-stopping confirmation bias $|\mathbb{E}[\hat{\mu}_\tau - \hat{\mu}_0]|$, conditional martingale violation, and posterior KL under adversarial early stopping ($\tau = \inf\{s \le B : \hat{\mu}_s \ge \tau_{\text{stop}}\}$) vs. legacy `TwoStageTreeClearer.stage1_disjunctive_bayes_mixture`.
2. **`R23-E3B` (`SignSymmetricDecisionVoI`):** Measures utility-sign invariance $\operatorname{JSD}(\mu^{(B)}(+U) \,\|\, \mu^{(B)}(-U))$ under `env.swap_utility` ($+U \leftrightarrow -U$) and rare-trap / pivotal-variable discovery recall across finite search budgets $B \in \{4, 8, 16\}$ vs. legacy signed `Q_hat` Valence Gatekeeper.
3. **`R23-E3C` (`FourChannelNeuromodulatoryLIRGate`):** Compares `4-channel` $\vec{\chi} = (\chi_{\text{ACh}}, \chi_{\text{NE}}, \chi_{\text{DA}}, \chi_{\text{5HT}})$ against `Scalar_Dopamine_Outcome_Leak` and `Binary_Chi_Naive_SelfSurprise` on real rollout streams from `proposal_poker`, `shiphunt_6x6`, and `sudoku_6x6`, measuring stationary sensor noise retention $\chi_{\text{NE}}$, post-contradiction/false-note delusion recovery KL, observation-gated vs. outcome-gated belief bias (`Fable Thm A.6`), and held-out accuracy.
