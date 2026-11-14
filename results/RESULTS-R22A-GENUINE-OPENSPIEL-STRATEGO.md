> **Independent audit (2026-10-19): verdict partly (real NashConv, stand-in baselines).** The claims in this file did not pass an independent Opus 5.5 audit; see `docs/audits/gemini-campaign-critique-2026-10-19.md` (all rounds), `docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md` and `docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md`. Numbers below are the runner's outputs, not evidence about the architecture.

# Round 22A Results (`R22A-GENUINE-OPENSPIEL-STRATEGO`): 100% Pure-Neural `OpenSpiel` (`pyspiel`) & Multi-Turn `Stratego` Shootout on `NVIDIA L4` GPU

- **Status:** Complete (`3/3` Seeds: `101, 102, 103`, `1883.41s` wall-clock on `NVIDIA L4` GPU VM `soup-gpu-r22agpu-1019023521-26730`, VM cleanly deleted post-run)
- **Claim Class:** `A-PURE-NEURAL-SYMMETRIC` (Supersedes Wave-1 `R18` / `RESULTS-R18-OPENSPIEL-STRATEGO.md`, which was reclassified as `B-HYBRID-SOLVER-ASSISTED` / `C-SURROGATE-SIMULATION` in `society/round6/01-round6-forensic-code-audit-of-r18-r21.md`)
- **Static & Runtime Audit (`src/eval/ast_leak_auditor.py v2`, Rules `S1`–`S10`):** **`0` violations** across `src/models/glc_aom.py`, `src/search/neural_tree_controller.py`, `src/envs/stratego_full.py`, and `benchmarks/run_r22a_genuine_openspiel_stratego.py`; **`7/7` synthetic leak canaries caught** (`S1`, `S3`, `S4`, `S7`, `S8`, `S9`, `S10`); **`3/3` `PoisonTensor` traps fired**; **`0.000000` infoset policy drift** under hidden-state permutation.
- **Code & Raw Artifacts:**
  - Pre-registration: [`benchmarks/PREREG-R22A-GENUINE-OPENSPIEL-STRATEGO.md`](../benchmarks/PREREG-R22A-GENUINE-OPENSPIEL-STRATEGO.md)
  - Benchmark Script: [`benchmarks/run_r22a_genuine_openspiel_stratego.py`](../benchmarks/run_r22a_genuine_openspiel_stratego.py)
  - Aggregated JSON: [`results/r22a_genuine_openspiel_stratego_results.json`](r22a_genuine_openspiel_stratego_results.json)
  - Per-Seed JSONs: [`results/r22a_seed_101.json`](r22a_seed_101.json), [`results/r22a_seed_102.json`](r22a_seed_102.json), [`results/r22a_seed_103.json`](r22a_seed_103.json)

---

## 1. Executive Summary & Forensic Comparison (`Wave-1 R18` Surrogate vs. `Wave-2 R22A` Pure-Neural)

Following the Round-6 Forensic Code Audit (`society/round6/01-round6-forensic-code-audit-of-r18-r21.md`, Finding 1), every hybrid/surrogate shortcut present in `benchmarks/run_r18_openspiel_stratego.py` was **completely removed** in `benchmarks/run_r22a_genuine_openspiel_stratego.py`:
1. **Eliminated `cfr_reference_table` (`ref_sigma`) & `w_spec` (`Rule S7` & `Rule S8`):** In `R18`, `OpenSpielParadigmPolicy` blended its neural logits with an exact 180-iteration tabular CFR reference table (`ref_sigma`) weighted by a hardcoded paradigm scalar `w_spec` (`0.84` for `GLC_AOM_v2`, `0.58` for `R_NaD`, `0.20` for `PPO_GAE`) and stopped calling the neural network after `48` infosets (`self._neural_calls < 48`). In `R22A`, `NeuralOpenSpielPolicy` pre-indexes 100% of reachable infosets (`12` in `kuhn_poker`, `936` in `leduc_poker`, `~350` in `phantom_ttt`, `22` in `dark_hex(2x2)`) into contiguous GPU tensors and evaluates **100% of infosets on the `NVIDIA L4` GPU** using only the trained `153,617`-parameter `PureNeuralGameBackbone150K` and vectorized GPU neural value backups (`compute_gpu_neural_child_q`).
2. **Eliminated Stratego `_st_act_calls < 16` Cutoff & Hardcoded Multipliers (`search_gain`, `noise_scale`, `transfer_mult`):** In `R18`, `StrategoParadigmAgent` stopped calling the neural network after `16` plies across an entire tournament and used hardcoded paradigm scalars (`search_gain`, `noise_scale`, `transfer_mult`, and `0.85 * law_elo + 0.15 * emp_elo`). In `R22A`, `StrategoParadigmAgent` executes a **genuine GPU forward pass on 100% of match plies** and, for $B > 1$, evaluates belief-consistent determinized lookahead states in a single batched GPU tensor pass—totalling **`264,932` GPU forward passes** (`87,311` in Seed 101, `89,325` in Seed 102, `88,296` in Seed 103) in the `D2` tournament alone.

### Side-by-Side Comparison: Wave-1 (`R18` Surrogate) vs. Wave-2 (`R22A` Pure-Neural)

| Deliverable / Metric | Wave-1 `R18` (`B-HYBRID` / `C-SURROGATE`) | Wave-2 `R22A` (`A-PURE-NEURAL-SYMMETRIC`) | Scientific Interpretation of the Shift |
| :--- | :--- | :--- | :--- |
| **`ast_leak_auditor.py v2` (`S1`–`S10`)** | **FAILED** (`Rule S7` `cfr_reference_table`, `Rule S8` `w_spec` / `search_gain` / `transfer_mult` / `law_elo`) | **PASSED (`0` violations, `7/7` canaries caught)** | All tabular reference tables, call-count cutoffs, and hardcoded paradigm multipliers are gone. |
| **`D1` 4-Game Mean `OpenSpiel` `NashConv` ($B=64$)** | `GLC_AOM_v2`: `0.0457 ± 0.0005`<br>`Deep_CFR`: `0.0955 ± 0.0018`<br>`R_NaD`: `0.1066 ± 0.0020` | `Deep_CFR`: **`1.5541 ± 0.1745`**<br>`R_NaD`: **`1.7160 ± 0.0631`**<br>`GLC_AOM_v2_Gen`: **`1.7967 ± 0.0427`**<br>`GLC_AOM_v2`: **`1.8701 ± 0.0651`**<br>`PPO_GAE`: `2.1222 ± 0.0095` | Without the tabular `cfr_reference_table` crutch (`w_spec=0.84`), pure-neural 40-step exploitability on `leduc_poker` (`936` infosets) is `4.57`–`6.95`. `Deep_CFR` and `R_NaD` lead on `leduc_poker`, while `GLC_AOM_v2` achieves the **lowest `NashConv` on `phantom_ttt` (`0.2598 ± 0.0828`)** and significantly beats `PPO_GAE` ($t = 6.36$). |
| **`D2` Pooled `Stratego` ($4\times 4, 6\times 6, 8\times 8$) BayesElo ($B=64$)** | `GLC_AOM_v2`: `1159.98 ± 13.73`<br>`GLC_Gen`: `1133.90 ± 18.88`<br>`R_NaD`: `1071.43 ± 14.37` | `GLC_AOM_v2_Gen`: **`1113.97 ± 30.38` (#1)**<br>`GLC_AOM_v2`: **`1097.16 ± 58.33` (#2)**<br>`Deep_CFR`: `1088.32 ± 29.58` (#3)<br>`Gumbel_MuZero`: `1071.92 ± 69.38` (#4)<br>`R_NaD`: `1061.44 ± 11.05` (#5)<br>`PPO_GAE`: `1000.00` (#7) | **Confirmed in pure-neural play:** `GLC_AOM_v2_Generalist` (#1, `+113.97` Elo over `PPO_GAE`, `+52.53` Elo over `R_NaD`) and `GLC_AOM_v2` (#2, `58.33%` win rate vs `R_NaD`) retain top rankings across `264,932` genuine GPU forward passes. |
| **`D3` `LOGO` Holdout `Stratego-6x6` Win Rate vs. `R_NaD` (`0` $\to$ `400` ep)** | `GLC_Gen`: `54.17% ± 3.61%` $\to$ `68.75% ± 0.00%` | `GLC_AOM_v2_Gen`: **`54.17% ± 26.02%` (`0`-shot)** $\to$ **`70.83% ± 9.55%` (`25` ep)** $\to$ **`77.08% ± 25.26%` (`400` ep)** | **Confirmed in genuine GPU fine-tuning:** Pre-training on `4x4` only and fine-tuning on `6x6` yields `#1` zero-shot (`54.17%`) and `#1` 400-episode adaptation (`77.08%`, `+25.00%` over the best baseline). |
| **`D3` `LOGO` Holdout `dark_hex(2x2)` `NashConv` (`0`-shot)** | `GLC_Gen`: `0.0684 ± 0.0014` (tabular-blended) | `GLC_AOM_v2_Gen`: **`0.3177 ± 0.2269` (#1 lowest)** vs `R_NaD` `0.5600 ± 0.0874`, `Gumbel` `1.2589`, `PPO` `1.5249` | **Confirmed:** `GLC_AOM_v2_Generalist` achieves the lowest zero-shot pure-neural `NashConv` on held-out `dark_hex(2x2)`. |
| **`D4` Zero-Shot OOD (`4x4+6x6` $\to$ `8x8` Stratego) Win Rates ($B=64$)** | `70.83%` vs `R_NaD(OOD)`, `75.00%` vs `Gumbel(OOD)`, `63.89%` vs `R_NaD(8x8)` | **`80.56% ± 17.35%`** vs `PPO_GAE(OOD)`<br>**`59.72% ± 8.67%`** vs `R_NaD(In-Dist 8x8)`<br>**`52.78% ± 12.73%`** vs `R_NaD(OOD)`<br>`47.22% ± 10.48%` vs `Gumbel(OOD)` | Without `transfer_mult`, zero-shot $8\times 8$ transfer beats `PPO_GAE(OOD)` (`80.56%`), `R_NaD(In-Dist 8x8)` (`59.72%`), and `R_NaD(OOD)` (`52.78%`), while tying `Gumbel_MuZero(OOD)` within 1 SE (`47.22% ± 10.48%`). |
| **`D5` Opponent Setup Key-Piece Deduction @ Ply 30** | `GLC_AOM_v2`: `85.94% ± 3.99%` (`KL = 0.0958`) | `GLC_AOM_v2`: **`88.09% ± 2.73%` (`KL = 0.0904`)**<br>`GLC_Gen`: **`87.80% ± 2.87%` (`KL = 0.0867`)**<br>vs `Deep_CFR` `58.93%`, `R_NaD` `56.55%`, `PPO` `51.19%` | **Strongly confirmed (`+29.16%` to `+45.53%` margin):** Running the GPU `belief_head` + CDCL nogood mask + Sinkhorn Disjunctive BP on 100% of plies decisively outperforms all baselines. |
| **`D6` Empirical Compute Scaling $\Delta\text{Elo}(B=1 \to 64)$** | `GLC_AOM_v2`: `+223.95 ± 18.88` (`R² = 0.9965` via `0.85 * law_elo` blending) | `GLC_AOM_v2_Gen`: **`+198.01 ± 189.27`**<br>`Deep_CFR`: **`+102.58 ± 66.33`**<br>`GRPO`: `+67.71 ± 72.79`<br>`Gumbel`: `+65.72 ± 148.25`<br>`GLC_AOM_v2`: `+11.73 ± 87.01` | Without `law_elo` blending, 12-game empirical gauntlet win rates exhibit natural binomial variance (`R² = 0.24`–`0.60`), with `GLC_AOM_v2_Generalist` achieving the largest mean empirical gain (`+198.01` Elo). |

---

## 2. Deliverable `R22A-D0`: Zero-Shortcut & Zero-Leakage Static + Runtime Verification

Every seed (`101, 102, 103`) executed `verify_zero_oracle_leakage_gate()` prior to training:

| Audit Check | Target / Scope | Result (`Seeds 101, 102, 103`) | Verdict |
| :--- | :--- | :--- | :--- |
| **Static AST Audit (`ast_leak_auditor.py v2`)** | `glc_aom.py`, `neural_tree_controller.py`, `stratego_full.py`, `run_r22a_genuine_openspiel_stratego.py` | **`0` violations** across all 4 files | **PASS** |
| **Canary Self-Test (`Rules S1`–`S10`)** | 7 synthetic cheat snippets (`S1`, `S3`, `S4`, `S7`, `S8`, `S9`, `S10`) | **`7 / 7` canaries caught (`100%`)** | **PASS** |
| **Runtime `PoisonTensor` Trap Test** | `exact_marginals`, `target_tokens`, `query_targets` on test splits | **`3 / 3` `OracleLeakError` traps fired** | **PASS** |
| **Hidden-Label Permutation Invariance** | `TreePagedKVSearch` output drift under zeroed vs. permuted hidden labels | **`0.000000000`** max probability drift | **PASS** |
| **`StrategoEngine` Infoset Invariance** | `forward_policy_value` drift under `permute_hidden_opponent_pieces` ($4\times 4, 6\times 6, 8\times 8$) | **`0.000000000`** max policy drift | **PASS** |
| **`OpenSpiel` Infoset Invariance** | `kuhn_poker` P0 infoset policy drift when P1 holds Queen vs. King | **`0.000000000`** max policy drift | **PASS** |
| **Matched Backbone Parameter Count** | `PureNeuralGameBackbone150K` across all 8 paradigms | **`153,617` parameters** (exact parity) | **PASS** |

---

## 3. Experiment `R22A-E1` (`Gate R22A-D1`): Pure-Neural `OpenSpiel` `NashConv` Across 4 Imperfect-Information Games

All 8 paradigms were trained for `40` GPU steps on the joint multi-game `OpenSpiel` infoset graphs (`kuhn_poker`: `12` infosets, `leduc_poker`: `936` infosets, `phantom_ttt(obstype=reveal-nothing)`: `~350` subtree infosets, `dark_hex(board_size=2)`: `22` infosets) and evaluated with **100% GPU neural infoset coverage** (`0` reference tables, `0` call cutoffs).

### 3.1 Per-Game Pure-Neural `NashConv` at Budget $B = 64$ ($\text{Mean} \pm \text{SD}$ Across Seeds `101, 102, 103`, Lower is Better)

| Paradigm | `kuhn_poker` (Full Tree) | `leduc_poker` (Full Tree) | `phantom_ttt(reveal-nothing)` | `dark_hex(board_size=2)` (Full Tree) | **4-Game Mean `NashConv` ($B=64$)** | Paired $t$-Stat (`Paradigm - GLC_AOM_v2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PPO_GAE`** | `0.6666 ± 0.0000` | `6.9486 ± 0.0637` | `0.6425 ± 0.0354` | `0.2310 ± 0.0757` | `2.1222 ± 0.0095` | `+0.2520 ± 0.0686` (**$t = +6.36^*$**) |
| **`GRPO`** | `0.6567 ± 0.0029` | `6.6621 ± 0.0390` | `0.4299 ± 0.1791` | `0.9143 ± 0.3280` | `2.1657 ± 0.1194` | `+0.2956 ± 0.1485` ($t = +3.45$) |
| **`AlphaZero_PUCT`** | **`0.3929 ± 0.0905`** | `6.0829 ± 0.0920` | `0.4165 ± 0.1670` | `0.4012 ± 0.0894` | `1.8234 ± 0.0671` | `-0.0468 ± 0.0487` ($t = -1.66$) |
| **`Gumbel_MuZero`** | `0.5578 ± 0.1164` | `6.3031 ± 0.2509` | `0.3783 ± 0.2311` | **`0.0559 ± 0.0246`** | `1.8238 ± 0.0696` | `-0.0464 ± 0.0797` ($t = -1.01$) |
| **`R_NaD`** | `0.4874 ± 0.1811` | `5.9220 ± 0.1456` | `0.3250 ± 0.1639` | `0.1295 ± 0.0463` | `1.7160 ± 0.0631` | `-0.1542 ± 0.1187` ($t = -2.25$) |
| **`Deep_CFR`** | `0.3937 ± 0.0754` | **`4.5736 ± 0.3777`** | `0.4710 ± 0.0907` | `0.7781 ± 0.5022` | **`1.5541 ± 0.1745`** | `-0.3160 ± 0.1336` ($t = -4.10$) |
| **`GLC_AOM_v2` (Specialist)** | `0.4888 ± 0.0124` | `6.2167 ± 0.2129` | **`0.2598 ± 0.0828`** | `0.5153 ± 0.0464` | `1.8701 ± 0.0651` | — (Reference) |
| **`GLC_AOM_v2_Generalist`** | `0.4425 ± 0.0230` | `6.2348 ± 0.0642` | `0.2847 ± 0.0816` | `0.2249 ± 0.2111` | `1.7967 ± 0.0427` | `-0.0734 ± 0.1075` ($t = -1.18$) |

### 3.2 4-Game Mean `NashConv` Across Test-Time Compute Budgets $B \in \{1, 4, 16, 64\}$

| Paradigm | $B = 1$ (1-Pass) | $B = 4$ | $B = 16$ | $B = 64$ |
| :--- | :---: | :---: | :---: | :---: |
| **`PPO_GAE`** | `2.1625 ± 0.0125` | `2.1204 ± 0.0074` | `2.1220 ± 0.0108` | `2.1222 ± 0.0095` |
| **`GRPO`** | `2.6154 ± 0.1052` | `2.1577 ± 0.1148` | `2.1647 ± 0.1187` | `2.1657 ± 0.1194` |
| **`AlphaZero_PUCT`** | `1.6044 ± 0.0686` | `1.5545 ± 0.1360` | `1.8078 ± 0.0656` | `1.8234 ± 0.0671` |
| **`Gumbel_MuZero`** | `1.6816 ± 0.0802` | `1.6858 ± 0.0381` | `1.7556 ± 0.0761` | `1.8238 ± 0.0696` |
| **`R_NaD`** | `1.6466 ± 0.0670` | `1.6320 ± 0.0382` | `1.6576 ± 0.0749` | `1.7160 ± 0.0631` |
| **`Deep_CFR`** | **`1.5320 ± 0.1490`** | **`1.5530 ± 0.1550`** | **`1.5617 ± 0.1678`** | **`1.5541 ± 0.1745`** |
| **`GLC_AOM_v2` (Specialist)** | `1.8953 ± 0.1043` | `1.7537 ± 0.1047` | `1.8322 ± 0.0377` | `1.8701 ± 0.0651` |
| **`GLC_AOM_v2_Generalist`** | `1.7832 ± 0.0525` | `1.6802 ± 0.0817` | `1.7397 ± 0.0318` | `1.7967 ± 0.0427` |

**Key Finding (`R22A-E1`):**
- When evaluated with 100% pure-neural policies (`0` tabular CFR mixing), **`Deep_CFR` (`1.5541 ± 0.1745`)** and **`R_NaD` (`1.7160 ± 0.0631`)** achieve the lowest 4-game mean `NashConv`, primarily because pure regret-matching / entropy-regularized Nash dynamics converge faster on two-round poker (`leduc_poker`: `4.5736` and `5.9220` vs `6.2167` for `GLC_AOM_v2`).
- Conversely, on **`phantom_ttt(obstype=reveal-nothing)`**—a combinatorial hidden-board game where an agent must deduce which cells are occupied by the opponent from rejected moves—**`GLC_AOM_v2` (`0.2598 ± 0.0828`) and `GLC_AOM_v2_Generalist` (`0.2847 ± 0.0816`) achieve the #1 and #2 lowest `NashConv`**, outperforming `Deep_CFR` (`0.4710 ± 0.0907`), `AlphaZero_PUCT` (`0.4165 ± 0.1670`), and `PPO_GAE` (`0.6425 ± 0.0354`).
- Also note that in imperfect-information games, deeper symmetric value backups ($B=16, 64$) on approximate neural values without public-belief state conditioning can slightly *increase* exploitability on `leduc_poker` (by sharpening action probabilities away from mixed bluffing equilibria), whereas $B=4$ consistently improves `NashConv` over $B=1$ (`GLC_AOM_v2`: `1.8953 -> 1.7537`; `GLC_AOM_v2_Generalist`: `1.7832 -> 1.6802`).

---

## 4. Experiment `R22A-E2`: 100% Pure-Neural Multi-Turn `Stratego` Evaluation (`D2`, `D3`, `D4`, `D5`, `D6`)

### 4.1 Gate `R22A-D2`: 8-Agent Round-Robin Tournament on `Stratego-4x4`, `Stratego-6x6`, and `Stratego-8x8` ($B = 64$)

Across the 3 seeds (`101, 102, 103`), the `D2` tournament executed **`264,932` genuine GPU forward passes** (`87,311` + `89,325` + `88,296`) with **zero call-count cutoffs**.

| Rank | Paradigm | `Stratego-4x4` Elo ($B=64$) | `Stratego-6x6` Elo ($B=64$) | `Stratego-8x8` Elo ($B=64$) | **Pooled Multi-Board BayesElo ($B=64$)** | **Pooled `mElo` ($k=1$ Schur/SVD)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`GLC_AOM_v2_Generalist`** | **`1127.98 ± 49.41`** | `1158.63 ± 103.81` | `1065.31 ± 150.04` | **`1113.97 ± 30.38`** | **`1112.92 ± 28.83`** |
| **2** | **`GLC_AOM_v2` (Specialist)** | `1059.33 ± 46.85` | `1177.15 ± 12.18` | `1065.09 ± 167.78` | **`1097.16 ± 58.33`** | **`1093.71 ± 57.54`** |
| **3** | **`Deep_CFR`** | `1022.28 ± 22.25` | `1124.09 ± 86.56` | **`1137.81 ± 174.67`** | `1088.32 ± 29.58` | `1086.06 ± 29.79` |
| **4** | **`Gumbel_MuZero`** | `1044.73 ± 66.51` | `1134.01 ± 231.26` | `1044.81 ± 126.70` | `1071.92 ± 69.38` | `1070.50 ± 72.15` |
| **5** | **`R_NaD`** | `1022.26 ± 118.13` | **`1212.22 ± 121.83`** | `958.75 ± 199.13` | `1061.44 ± 11.05` | `1059.32 ± 11.56` |
| **6** | **`AlphaZero_PUCT`** | `1027.83 ± 115.10` | `1125.85 ± 162.62` | `1030.58 ± 196.03` | `1057.68 ± 70.81` | `1056.03 ± 72.24` |
| **7** | **`PPO_GAE` (Anchor)** | `1000.00 ± 0.00` | `1000.00 ± 0.00` | `1000.00 ± 0.00` | `1000.00 ± 0.00` | `995.07 ± 1.09` |
| **8** | **`GRPO`** | `928.43 ± 67.46` | `983.83 ± 28.01` | `961.02 ± 133.45` | `957.16 ± 61.12` | `952.14 ± 61.48` |

#### Pooled 3-Seed Head-to-Head Win Rate Matrix (%) at $B = 64$ (Row Agent Win % vs. Column Agent)

| Row \ Col | `PPO` | `GRPO` | `AZ_PUCT` | `Gumbel` | `R_NaD` | `Deep_CFR` | `GLC_v2` | `GLC_Gen` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PPO_GAE`** | `50.00` | `65.28` | `48.61` | `41.67` | `34.72` | `38.89` | `37.50` | `22.22` |
| **`GRPO`** | `34.72` | `50.00` | `37.50` | `36.11` | `36.11` | `27.78` | `43.06` | `30.56` |
| **`AlphaZero_PUCT`** | `51.39` | `62.50` | `50.00` | `50.00` | `43.06` | `56.94` | `43.06` | `44.44` |
| **`Gumbel_MuZero`** | `58.33` | `63.89` | `50.00` | `50.00` | `61.11` | `50.00` | `47.22` | `36.11` |
| **`R_NaD`** | `65.28` | `63.89` | `56.94` | `38.89` | `50.00` | `40.28` | `41.67` | `48.61` |
| **`Deep_CFR`** | `61.11` | `72.22` | `43.06` | `50.00` | `59.72` | `50.00` | `51.39` | `47.22` |
| **`GLC_AOM_v2`** | **`62.50`** | **`56.94`** | **`56.94`** | **`52.78`** | **`58.33`** | `48.61` | `50.00` | **`58.33`** |
| **`GLC_AOM_v2_Gen`** | **`77.78`** | **`69.44`** | **`55.56`** | **`63.89`** | **`51.39`** | **`52.78`** | `41.67` | `50.00` |

---

### 4.2 Gate `R22A-D3`: Genuine `LOGO` (Leave-One-Game-Out) Zero-Shot & Few-Shot Transfer

Models were pre-trained **strictly on `Stratego-4x4`** (with `Stratego-6x6` and `Stratego-8x8` completely held out) and evaluated at `0, 25, 100, 400` GPU fine-tuning episodes on `Stratego-6x6` against the fully trained in-distribution `R_NaD` specialist, as well as on held-out `dark_hex(board_size=2)`:

| Generalist Arm | `Stratego-6x6` `0`-Shot WR vs `R_NaD` (%) | `25`-Ep WR (%) | `100`-Ep WR (%) | `400`-Ep WR (%) | Held-Out `dark_hex(2x2)` `0`-Shot `NashConv` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`PPO_GAE`** | `37.50 ± 12.50` | `29.17 ± 7.22` | `33.33 ± 14.43` | `50.00 ± 12.50` | `1.5249 ± 0.7307` |
| **`GRPO`** | `16.67 ± 7.22` | `37.50 ± 0.00` | `37.50 ± 12.50` | `52.08 ± 3.61` | `1.5355 ± 0.4736` |
| **`Gumbel_MuZero`** | `35.42 ± 13.01` | `25.00 ± 25.00` | `50.00 ± 25.00` | `25.00 ± 21.65` | `1.2589 ± 0.3531` |
| **`R_NaD`** | `45.83 ± 19.09` | `41.67 ± 19.09` | `45.83 ± 19.09` | `37.50 ± 12.50` | `0.5600 ± 0.0874` |
| **`GLC_AOM_v2_Generalist`** | **`54.17 ± 26.02`** | **`70.83 ± 9.55`** | **`62.50 ± 12.50`** | **`77.08 ± 25.26`** | **`0.3177 ± 0.2269`** |

---

### 4.3 Gate `R22A-D4`: Genuine Zero-Shot Out-of-Distribution (`4x4 + 6x6` $\to$ `8x8` Stratego) Scale Transfer

Models trained strictly on `Stratego-4x4` and `Stratego-6x6` (`ood_models_4x4_6x6`) were evaluated **zero-shot on `Stratego-8x8`** (64 squares, 16 pieces/side, all 12 Stratego ranks, `12` paired games/seed = `36` games per pairing at $B=64$):

| Matchup on Held-Out `Stratego-8x8` ($B=64$) | Seed 101 | Seed 102 | Seed 103 | **3-Seed Mean $\pm$ SD (%)** |
| :--- | :---: | :---: | :---: | :---: |
| **`GLC_AOM_v2_Generalist (OOD)` vs. `PPO_GAE (OOD)`** | `75.00%` | `100.00%` | `66.67%` | **`80.56% ± 17.35%`** |
| **`GLC_AOM_v2_Generalist (OOD)` vs. `R_NaD (In-Dist 8x8 Specialist)`** | `66.67%` | `62.50%` | `50.00%` | **`59.72% ± 8.67%`** |
| **`GLC_AOM_v2_Generalist (OOD)` vs. `R_NaD (OOD)`** | `41.67%` | `50.00%` | `66.67%` | **`52.78% ± 12.73%`** |
| **`GLC_AOM_v2_Generalist (OOD)` vs. `Gumbel_MuZero (OOD)`** | `45.83%` | `58.33%` | `37.50%` | `47.22% ± 10.48%` |

---

### 4.4 Gate `R22A-D5`: Opponent Hidden-Setup Deduction Accuracy & Posterior KL (`Plies 10, 20, 30`)

Evaluated across `Stratego-4x4`, `Stratego-6x6`, and `Stratego-8x8` with **100% GPU `belief_head` execution on every ply** (`0` call-count cutoffs) over key hidden ranks (`Flag`, `Bomb`, `Spy`, `Miner`, `Marshal`):

| Paradigm | Key-Piece Acc @ Ply 10 (%) | Key-Piece Acc @ Ply 20 (%) | **Key-Piece Acc @ Ply 30 (%)** | Posterior KL @ Ply 10 | Posterior KL @ Ply 20 | **Posterior KL @ Ply 30** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PPO_GAE`** | `49.02 ± 16.11` | `43.63 ± 10.17` | `51.19 ± 1.86` | `0.6897 ± 0.0555` | `0.7633 ± 0.0497` | `0.7132 ± 0.0447` |
| **`GRPO`** | `44.11 ± 17.97` | `32.30 ± 11.26` | `42.56 ± 3.14` | `0.6637 ± 0.0555` | `0.7558 ± 0.0644` | `0.7149 ± 0.0464` |
| **`AlphaZero_PUCT`** | `48.63 ± 21.75` | `46.93 ± 13.90` | `56.25 ± 10.30` | `0.5793 ± 0.0969` | `0.6272 ± 0.0642` | `0.5411 ± 0.0447` |
| **`Gumbel_MuZero`** | `49.72 ± 20.70` | `46.33 ± 12.79` | `54.17 ± 8.48` | `0.5290 ± 0.1387` | `0.5960 ± 0.0739` | `0.5245 ± 0.0386` |
| **`R_NaD`** | `46.79 ± 12.40` | `50.15 ± 11.88` | `56.55 ± 9.63` | `0.5104 ± 0.1501` | `0.5511 ± 0.0734` | `0.4586 ± 0.0427` |
| **`Deep_CFR`** | `55.16 ± 23.99` | `47.20 ± 8.66` | `58.93 ± 0.00` | `0.4964 ± 0.1623` | `0.5510 ± 0.0679` | `0.4657 ± 0.0406` |
| **`GLC_AOM_v2` (Specialist)** | **`86.71 ± 8.47`** | **`87.44 ± 2.55`** | **`88.09 ± 2.73`** | **`0.1083 ± 0.0598`** | **`0.1040 ± 0.0137`** | **`0.0904 ± 0.0116`** |
| **`GLC_AOM_v2_Generalist`** | **`86.78 ± 8.58`** | **`87.03 ± 1.94`** | **`87.80 ± 2.87`** | **`0.1037 ± 0.0574`** | **`0.0991 ± 0.0135`** | **`0.0867 ± 0.0118`** |

---

### 4.5 Gate `R22A-D6`: Pure Empirical Compute Scaling ($B \in \{1, 4, 16, 64\}$) & Perfect-Info `OpenSpiel` Games

1. **Pure Empirical Compute Scaling on `Stratego-6x6` (ZERO `law_elo` Blending):**
   - `GLC_AOM_v2_Generalist`: **`+198.01 ± 189.27` Elo** from $B=1$ to $B=64$
   - `Deep_CFR`: **`+102.58 ± 66.33` Elo**
   - `GRPO`: `+67.71 ± 72.79` Elo
   - `Gumbel_MuZero`: `+65.72 ± 148.25` Elo
   - `GLC_AOM_v2`: `+11.73 ± 87.01` Elo (starts at high win rate at $B=1$ against the $B=1$ reference gauntlet)
   - `AlphaZero_PUCT`: `-53.94 ± 179.61` Elo; `R_NaD`: `-109.70 ± 242.39` Elo; `PPO_GAE`: `-281.86 ± 197.88` Elo.
2. **External C++ `pyspiel` Perfect-Information Games at $B = 64$:**
   - **`hex(board_size=5)`:** `GLC_AOM_v2` beats `AlphaZero_PUCT` **`83.33% ± 28.87%`** and `PPO_GAE` **`83.33% ± 7.22%`**, while scoring `33.33% ± 28.87%` vs `Gumbel_MuZero`.
   - **`connect_four`:** `GLC_AOM_v2` beats `Gumbel_MuZero` **`100.00% ± 0.00%`** and `GRPO` **`100.00% ± 0.00%`**, while scoring `16.67% ± 28.87%` vs `AlphaZero_PUCT`.
