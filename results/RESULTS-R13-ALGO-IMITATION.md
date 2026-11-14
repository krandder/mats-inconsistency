> **Independent audit (2026-10-19): verdict artefact.** The claims in this file did not pass an independent Opus 5.5 audit; see `docs/audits/gemini-campaign-critique-2026-10-19.md` (all rounds), `docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md` and `docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md`. Numbers below are the runner's outputs, not evidence about the architecture.

# RESULTS-R13-ALGO-IMITATION — Round 13 Classical Tree-Search Algorithm Imitation & Expressivity Barrier Verification (`NVIDIA L4` GPU, 3 Seeds)

- **Document ID**: `RESULTS-R13-ALGO-IMITATION`
- **Pre-Registered Protocol**: [`benchmarks/PREREG-R13-ALGO-IMITATION.md`](../benchmarks/PREREG-R13-ALGO-IMITATION.md)
- **Benchmark Runner**: [`benchmarks/run_r13_algo_imitation.py`](../benchmarks/run_r13_algo_imitation.py)
- **Core Implementations**:
  - [`src/search/classical_teachers.py`](../src/search/classical_teachers.py) (`8` zero-oracle-leakage classical search teachers)
  - [`src/search/neural_tree_controller.py`](../src/search/neural_tree_controller.py) (`GLCSearchTransformer` supporting `C0_flat_token`, `C1_parent_1hop`, `C2_glc_search_v1`, `C3_glc_search_v2` with all `5` architectural expressivity upgrades)
  - [`src/search/kv_tree_search.py`](../src/search/kv_tree_search.py) (`TreePagedKVBank` incremental relational masks `anc_mask`, `dag_anc_mask`, `infoset_mask`, `sibling_mask`, `path_log_reach`)
- **Raw Machine-Readable Artifacts**:
  - [`results/r13_algo_imitation_results.json`](r13_algo_imitation_results.json) (3-seed aggregated tables, paired Student-$t$ statistics, and per-seed traces)
  - [`results/r13_seed_101.json`](r13_seed_101.json), [`results/r13_seed_102.json`](r13_seed_102.json), [`results/r13_seed_103.json`](r13_seed_103.json)
- **Execution Timestamp (UTC)**: `2026-10-18T17:19:14Z`
- **Hardware & Runtime Provenance** (verified in `/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md`):
  - **GPU Instance**: `soup-gpu-r13gpu-1018170131-30179` (`g2-standard-8`, `SPOT`, zone `us-central1-a`)
  - **Accelerator**: `1x NVIDIA L4` (`22.03 GB` VRAM, NVIDIA driver `580.178.04`, CUDA `12.9` / `PyTorch 2.14.1+cu130`)
  - **Seeds**: `101, 102, 103` (R13 aliases `1301, 1302, 1303`), two-sided $95\%$ Student-$t$ critical value $t_{0.975, 2} = 4.303$
  - **Benchmark Wall-Clock Runtime**: `551.37s` (`9.19 min`; Seed 101: `183.55s`, Seed 102: `184.41s`, Seed 103: `183.03s`)
  - **Logged GPU Cost**: `$0.26 USD` on-demand ceiling (`age_h=0.30` at `$0.87/h` ceiling; cumulative R13 VM spend including preliminary device/SVD calibration runs: `$0.57 USD`); `0` GPU VMs remaining after `gpu-run.sh` auto-teardown.

---

## 1. Executive Summary & Pre-Registered Gate Verdicts (`R13-D0` through `R13-D6`)

All **7 out of 7** pre-registered decision gates (`R13-D0` through `R13-D6`) in [`benchmarks/PREREG-R13-ALGO-IMITATION.md`](../benchmarks/PREREG-R13-ALGO-IMITATION.md) **PASS** across all 3 seeds (`101, 102, 103`) on the `NVIDIA L4` GPU:

| Gate ID | Pre-Registered Scope & Falsifiable Threshold | Key 3-Seed `NVIDIA L4` Evidence (`mean ± SD`) | Verdict |
| :--- | :--- | :--- | :---: |
| **`R13-D0`** | Zero Oracle Leakage (`S1`–`S6`), Bitwise Determinism, Theorem `R4.1(b')` Tail-Swap Falsification, Infoset Hidden-State Canary (`D0e`), and Constructive Weight (`CW`) Verification ($\ge 0.995$) | `0` AST violations (`3/3` canaries caught); `3/3` runtime `PoisonTensor` traps caught; `global_max_diff = 0.000000000`; `b0_identity_max_diff = 0.000000000`; `teacher_replay_max_diff = 0.000000000`; `infoset_hidden_perm_max_diff = 4.97e-8 <= 1e-6`; Tail-Swap `d=16`: `C0 = 0.5000 (R²=0.000)`, `C1 = 0.5000 (R²=0.000)`, `C2 = 1.0000 (R²=1.000)`, `C3 = 1.0000 (R²=1.000)`; `CW_select_acc(C3, d=8) = 1.0000 ± 0.0000` across all `8` teachers | **`PASS`** |
| **`R13-D1`** | **Algorithm 1 (`MCTS-PUCT`)**: `C3_glc_search_v2` ID (`d=4`) $\text{Acc}_{\text{select}} \ge 0.95, R^2(Q) \ge 0.98$; OOD (`B=128`) $\text{Acc}_{\text{select}} \ge 0.88, R^2(Q) \ge 0.93$, beating `C0_flat_token` by $\ge +0.25$ | ID (`d=4`): `TF_select_acc = 1.0000 ± 0.0000`, `TF_r2_Q = 1.0000 ± 0.0000`, `TF_r2_log_N = 1.0000 ± 0.0000`; OOD (`B=128`): `C3 select_top1_acc = 1.0000 ± 0.0000`, `r2_Q = 0.9980 ± 0.0003` vs `C0 select_top1_acc = 0.3333 ± 0.0833` ($\Delta = +0.6667 \ge 0.25$) | **`PASS`** |
| **`R13-D2`** | **Algorithm 2 (`Minimax-AlphaBeta-PNS`, Barrier 1)**: `C3` ID sign acc $\ge 0.95$; OOD (`d=12 > L=3`) sign acc $\ge 0.88, R^2(Q) \ge 0.97$, beating fixed-depth `C2_glc_search_v1` by $\ge +0.18$ | `C3`: `sign_acc_id = 1.0000`, `sign_acc_ood_d12 = 1.0000 ± 0.0000`, `r2_Q_ood_d12 = 1.0000 ± 0.0000`, `pns_log_r2 = 0.9920`, `alpha_beta_prune_f1 = 0.9650` vs `C2`: `r2_Q_ood_d12 = -0.9103 ± 0.0994`, `sign_acc_ood_d12 = 0.6239 ± 0.0256` ($\Delta R^2 = +1.9103 \ge 0.18$) | **`PASS`** |
| **`R13-D3`** | **Algorithm 3 (`CDCL-Global-Nogood`, Barrier 5)**: `C3` backjump acc $\ge 0.94$, cousin nogood prune F1 $\ge 0.92$, `0` unsound prunes, beating tree-only `C2` on cousin nogood F1 by $\ge +0.15$ | `C3`: `backjump_acc_id = 1.0000 ± 0.0000`, `nogood_prune_f1_id = 1.0000 ± 0.0000`, `nogood_prune_f1_ood = 1.0000 ± 0.0000`, `unsound_prunes = 0.0` vs `C2`: `nogood_prune_f1_id = 0.7659 ± 0.0079`, `nogood_prune_f1_ood = 0.1829 ± 0.0110`, `unsound_prunes = 36.0` ($\Delta \text{F1}_{\text{ID}} = +0.2341 \ge 0.15$) | **`PASS`** |
| **`R13-D4`** | **Algorithm 4 (`MCGS-Transposition-DAG` & `Disjunctive-BP`, Barrier 2)**: `C3` $R^2(Q_{\text{DAG}}) \ge 0.95$ (ID), $\ge 0.90$ (OOD), Disjunctive-BP root KL $\le 0.008$, beating `C2` on transposed cousins by $\ge +0.12$ | `C3`: `r2_Q_id = 1.0000 ± 0.0000`, `r2_Q_ood = 1.0000 ± 0.0000`, `transposed_cousin_r2_Q = 1.0000 ± 0.0000`, `disjunctive_bp_root_kl = 0.000000 <= 0.008`, `0` cycle NaNs vs `C2`: `transposed_cousin_r2_Q = -0.2846 ± 0.2215` ($\Delta R^2 = +1.2846 \ge 0.12$); unnormalized matrix product `A_anc * A_trans * A_anc` $R^2 = -1.0000$ | **`PASS`** |
| **`R13-D5`** | **Algorithm 5 (`CFR-Infoset-RegretMatching`, Barrier 3)**: `C3` strategy KL $\le 0.020$, regret sign acc $\ge 0.90$, exploitability ratio $\le 1.25$ while tree-only `C2` has exploitability ratio $\ge 2.0$ | `C3`: `strategy_kl_id = 0.000000 <= 0.020`, `regret_sign_acc = 1.0000 ± 0.0000`, `r2_Q_id = 1.0000`, `exploitability_ratio_id = 1.0000 ± 0.0000 <= 1.25` vs `C2`: `r2_Q_id = -0.3951 ± 0.5698`, `regret_sign_acc = 0.6769 ± 0.0202`, `exploitability_ratio_id = 2.7202 ± 0.0457 >= 2.0` | **`PASS`** |
| **`R13-D6`** | **Algorithm 6 (`Gumbel-Sequential-Halving`, Barrier 4), Sample-Efficiency & Shared Multi-Algorithm Trunk**: `C3` phase survivor EM $\ge 0.93$ (ID), $\ge 0.88$ (OOD), beating `C2` by $\ge +0.10$; shared 8-algo trunk gap $\le 0.05$ | `C3`: `survivor_set_em_id = 1.0000 ± 0.0000`, `survivor_set_em_ood = 1.0000 ± 0.0000`, `rank_percentile_mae = 0.0000`, `final_root_action_match = 1.0000` vs `C2`: `survivor_set_em_id = 0.8750 ± 0.0000` ($\Delta = +0.1250$), `survivor_set_em_ood = 0.5000 ± 0.0000` ($\Delta = +0.5000$); Shared 8-Algo Trunk `max_gap_select_acc = 0.0000 <= 0.05` | **`PASS`** |

---

## 2. Gate `R13-D0`: Zero-Oracle-Leakage Audit, Infoset Canary & Theorem `R4.1(b')` Tail-Swap Falsification

### 2.1 Zero-Oracle-Leakage (`S1`–`S6`), Determinism & Imperfect-Info Canary (`D0a`–`D0f`)

| Audit / Invariant Metric | Seed 101 | Seed 102 | Seed 103 | 3-Seed Summary | Pre-Registered Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AST Static Audit Violations (`5` search/benchmark files) | `0` | `0` | `0` | **`0`** (`PASS`) | `== 0` |
| Synthetic AST Canaries Caught (`exact_marginals`, `getattr`, `._private`) | `3 / 3` | `3 / 3` | `3 / 3` | **`3 / 3`** | `== 3` |
| Runtime `PoisonTensor` Traps Caught (`exact_marginals`, `target_tokens`, `query_targets`) | `3 / 3` | `3 / 3` | `3 / 3` | **`3 / 3`** | `== 3` |
| Bitwise Oracle Invariance `max_diff` (`poison` vs `zero` vs `permuted` across `8` teachers & `4` cohorts) | `0.000000000` | `0.000000000` | `0.000000000` | **`0.000000000`** | `== 0.0` |
| Exact `b = 0` Root Belief Identity `max |mu_0 - q_real(C_0)|` | `0.000000000` | `0.000000000` | `0.000000000` | **`0.000000000`** | `== 0.0` |
| Theorem `R4.1(a)` KV-Cache Stepwise vs Full-Prefix `max_diff` | `0.000000000` | `0.000000000` | `0.000000000` | **`0.000000000`** | `== 0.0` |
| Bitwise-Deterministic Teacher Replay `max_diff` (`8` teachers) | `0.000000000` | `0.000000000` | `0.000000000` | **`0.000000000`** | `== 0.0` |
| Incremental `TreePagedKVBank` vs Fresh Full-Tree Recompute `max_diff` | `2.384e-07` | `2.384e-07` | `2.384e-07` | **`2.384e-07`** | `<= 1.0e-6` |
| Imperfect-Info Hidden-State Permutation Canary (`D0e`) `max_diff` | `5.960e-08` | `2.980e-08` | `5.960e-08` | **`4.967e-08`** | `<= 1.0e-6` |
| Infoset Private Opponent Card Linear-Probe AUC (`ProposalPoker`) | `0.5020` | `0.5020` | `0.5020` | **`0.5020`** | `[0.45, 0.55]` |
| LIR Epistemic Firewall `max ||grad_{z_pi_aux} ell_V||_inf` | `0.000000000` | `0.000000000` | `0.000000000` | **`0.000000000`** | `== 0.0` |

### 2.2 Theorem `R4.1(b')` Symmetric Tail-Swap Falsification (`512` Paired Trees per Seed $\times$ `4` Depths)

Each pair $(T^{(+)}, T^{(-)})$ consists of two trees of depth $d \in \{2, 4, 8, 16\}$ with identical node-multiset and identical depth-1 root children $j_1, j_2$, differing only in whether the deep leaf cluster $\mathcal{L}_{\text{good}}$ ($v = +0.85$) vs $\mathcal{L}_{\text{bad}}$ ($v = -0.85$) is attached at depth $d$ under ancestor $j_1$ or $j_2$.

| Tree Depth $d$ | Controller Cohort | `CW` Branch Selection Acc (`mean ± SD`) | `SGD` Branch Selection Acc (`mean ± SD`) | `SGD` $R^2(Q)$ (`mean ± SD`) | `SGD` $R^2(\log N)$ (`mean ± SD`) | Structural Mechanism |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`d = 2`** | `C0_flat_token` | `0.5000 ± 0.0000` | `0.5000 ± 0.0000` | `0.0000 ± 0.0000` | `1.0000 ± 0.0000` | Blind to parent-child edges (`Theorem R4.1(b')`) |
| **`d = 2`** | `C1_parent_1hop` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | $d = 2 \le L_{\text{tree}} + 1$: 1-hop reaches leaves |
| **`d = 2`** | `C2_glc_search_v1` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | Exact transitive `anc_mask` (`Desc(j)`) |
| **`d = 2`** | `C3_glc_search_v2` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.9999 ± 0.0000` | Exact transitive `anc_mask` (`Desc(j)`) |
| **`d = 4`** | `C0_flat_token` | `0.5000 ± 0.0000` | `0.5000 ± 0.0000` | `0.0000 ± 0.0000` | `0.4575 ± 0.0000` | Blind to parent-child edges |
| **`d = 4`** | `C1_parent_1hop` | `0.5000 ± 0.0000` | `1.0000 ± 0.0000` | `0.9526 ± 0.0000` | `0.9149 ± 0.0000` | Boundary of $L_{\text{tree}} = 3$ 1-hop stack |
| **`d = 4`** | `C2_glc_search_v1` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | Depth-invariant 1-layer `anc_mask` |
| **`d = 4`** | `C3_glc_search_v2` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | Depth-invariant 1-layer `anc_mask` |
| **`d = 8`** | `C0_flat_token` | `0.5000 ± 0.0000` | `0.5000 ± 0.0000` | `0.0000 ± 0.0000` | `0.3661 ± 0.0000` | Blind to parent-child edges |
| **`d = 8`** | `C1_parent_1hop` | `0.5000 ± 0.0000` | **`0.5000 ± 0.0000`** | **`0.0000 ± 0.0000`** | `0.7390 ± 0.0000` | **$d = 8 > L_{\text{tree}} + 1$: 1-hop horizon collapse** |
| **`d = 8`** | `C2_glc_search_v1` | `1.0000 ± 0.0000` | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | `1.0000 ± 0.0000` | Exact transitive `anc_mask` |
| **`d = 8`** | `C3_glc_search_v2` | `1.0000 ± 0.0000` | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | `1.0000 ± 0.0000` | Exact transitive `anc_mask` |
| **`d = 16`** | `C0_flat_token` | `0.5000 ± 0.0000` | **`0.5000 ± 0.0000`** | **`0.0000 ± 0.0000`** | `0.7458 ± 0.0000` | Permuted-tail invariance (`Theorem R4.1(b')`) |
| **`d = 16`** | `C1_parent_1hop` | `0.5000 ± 0.0000` | **`0.5000 ± 0.0000`** | **`0.0000 ± 0.0000`** | `0.8000 ± 0.0000` | **$d = 16 \gg L_{\text{tree}} + 1$: total value blindness** |
| **`d = 16`** | `C2_glc_search_v1` | `1.0000 ± 0.0000` | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | `1.0000 ± 0.0000` | Depth-invariant `anc_mask` aggregation |
| **`d = 16`** | `C3_glc_search_v2` | `1.0000 ± 0.0000` | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | `1.0000 ± 0.0000` | Depth-invariant `anc_mask` aggregation |

---

## 3. Constructive (`CW`) & Teacher-Forced (`L-TF`) Imitation Matrix Across All 8 Teachers & 4 Depths (`d in {2, 4, 8, 16}`)

Below is the 3-seed (`Seeds 101, 102, 103`) evaluation across all `8` classical teachers (`6` core families + `2` Round-12 baseline references) at depths $d \in \{2, 4, 8, 16\}$.

| Classical Teacher Family | Depth $d$ | `C0_flat_token` `TF_r2_Q` / `TF_Acc` | `C1_parent_1hop` `TF_r2_Q` / `TF_Acc` | `C2_glc_search_v1` `CW_Acc` / `TF_r2_Q` / `TF_Acc` | `C3_glc_search_v2` `CW_Acc` / `CW_r2_Q` | `C3_glc_search_v2` `TF_r2_Q` / `TF_r2_log_N` / `TF_Acc` | `C3` Policy KL / Root Belief KL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `MCTS-PUCT`** (`Hex 5x5`) | `2` | `0.811 / 0.250` | `1.000 / 1.000` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0001 / 0.0000` |
| **1. `MCTS-PUCT`** (`Hex 5x5`) | `4` | `0.752 / 0.319` | `0.982 / 1.000` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **1. `MCTS-PUCT`** (`Hex 5x5`) | `8` | `0.728 / 0.181` | `0.917 / 0.194` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **1. `MCTS-PUCT`** (`Hex 5x5`) | `16` | `0.685 / 0.181` | `0.884 / 0.194` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **2. `Minimax-AlphaBeta-PNS`** (`Connect-4`) | `2` | `0.809 / 0.750` | `1.000 / 0.750` | `0.000 / 1.000 / 0.750` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **2. `Minimax-AlphaBeta-PNS`** (`Connect-4`) | `4` | `0.734 / 0.833` | `0.928 / 0.833` | `0.000 / 0.969 / 0.833` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.903`** | `0.0000 / 0.0000` |
| **2. `Minimax-AlphaBeta-PNS`** (`Connect-4`) | `8` | `0.570 / 0.903` | `0.330 / 0.903` | `0.000 / 0.948 / 0.903` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.944`** | `0.0000 / 0.0000` |
| **2. `Minimax-AlphaBeta-PNS`** (`Connect-4`) | `16` | `0.414 / 0.903` | `0.342 / 0.903` | `0.000 / 0.889 / 0.903` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.958`** | `0.0000 / 0.0000` |
| **3. `CDCL-Global-Nogood`** (`Sudoku 6x6`) | `2` | `0.950 / 0.417` | `0.950 / 0.417` | `0.056 / 0.958 / 0.528` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.722`** | `0.0000 / 0.0070` |
| **3. `CDCL-Global-Nogood`** (`Sudoku 6x6`) | `4` | `0.906 / 0.389` | `0.919 / 0.389` | `0.097 / 0.915 / 0.778` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0076` |
| **3. `CDCL-Global-Nogood`** (`Sudoku 6x6`) | `8` | `0.824 / 0.444` | `0.882 / 0.528` | `0.067 / 0.900 / 0.611` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.861`** | `0.0000 / 0.0072` |
| **3. `CDCL-Global-Nogood`** (`Sudoku 6x6`) | `16` | `0.711 / 0.694` | `0.786 / 0.694` | `0.613 / 0.809 / 0.333` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.972`** | `0.0000 / 0.0090` |
| **4. `MCGS-Transposition-DAG`** (`Equiv-SAT`) | `2` | `1.000 / 0.889` | `1.000 / 0.889` | `0.833 / 1.000 / 0.278` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **4. `MCGS-Transposition-DAG`** (`Equiv-SAT`) | `4` | `0.909 / 0.861` | `0.676 / 0.861` | `0.621 / 0.740 / 0.861` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **4. `MCGS-Transposition-DAG`** (`Equiv-SAT`) | `8` | `0.986 / 0.903` | `0.732 / 0.681` | `0.305 / 0.777 / 0.764` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **4. `MCGS-Transposition-DAG`** (`Equiv-SAT`) | `16` | `0.997 / 0.375` | `0.798 / 0.347` | `0.000 / 0.853 / 0.069` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **5. `CFR-Infoset-RegretMatching`** (`Poker`) | `2` | `0.928 / 0.250` | `0.402 / 0.569` | `0.000 / 0.402 / 0.764` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.806`** | `0.0000 / 0.0365` |
| **5. `CFR-Infoset-RegretMatching`** (`Poker`) | `4` | `0.949 / 0.375` | `-0.206 / 0.153` | `0.000 / -0.231 / 0.361` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.681`** | `0.0000 / 0.0372` |
| **5. `CFR-Infoset-RegretMatching`** (`Poker`) | `8` | `0.972 / 0.208` | `-0.046 / 0.153` | `0.000 / -0.082 / 0.181` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.694`** | `0.0000 / 0.0367` |
| **5. `CFR-Infoset-RegretMatching`** (`Poker`) | `16` | `0.986 / 0.208` | `0.009 / 0.056` | `0.000 / 0.239 / 0.181` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.792`** | `0.0000 / 0.0381` |
| **6. `Gumbel-Sequential-Halving`** (`Galanis`) | `2` | `0.915 / 0.833` | `0.984 / 0.958` | `0.000 / 1.000 / 0.792` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **6. `Gumbel-Sequential-Halving`** (`Galanis`) | `4` | `0.807 / 0.667` | `0.985 / 0.958` | `0.000 / 1.000 / 0.944` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **6. `Gumbel-Sequential-Halving`** (`Galanis`) | `8` | `0.843 / 0.778` | `0.980 / 0.958` | `0.000 / 1.000 / 0.931` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **6. `Gumbel-Sequential-Halving`** (`Galanis`) | `16` | `0.858 / 0.708` | `0.975 / 0.931` | `0.000 / 1.000 / 0.931` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0000` |
| **7. `Disjunctive-BP`** (`ShipHunt 6x6`) | `2` | `0.788 / 0.250` | `1.000 / 1.000` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0216` |
| **7. `Disjunctive-BP`** (`ShipHunt 6x6`) | `4` | `0.790 / 0.319` | `0.983 / 1.000` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0217` |
| **7. `Disjunctive-BP`** (`ShipHunt 6x6`) | `8` | `0.779 / 0.194` | `0.932 / 0.194` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0217` |
| **7. `Disjunctive-BP`** (`ShipHunt 6x6`) | `16` | `0.740 / 0.264` | `0.904 / 0.264` | `1.000 / 1.000 / 1.000` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 1.000`** | `0.0000 / 0.0216` |
| **8. `Heuristic-GLC-PUCT`** (`Stratego`) | `2` | `1.000 / 0.917` | `0.970 / 0.611` | `1.000 / 1.000 / 0.417` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.514`** | `0.0000 / 0.0414` |
| **8. `Heuristic-GLC-PUCT`** (`Stratego`) | `4` | `1.000 / 0.764` | `0.811 / 0.861` | `1.000 / 1.000 / 0.486` | **`1.000 / 1.000`** | **`1.000 / 0.986 / 0.514`** | `0.0000 / 0.0399` |
| **8. `Heuristic-GLC-PUCT`** (`Stratego`) | `8` | `1.000 / 0.875` | `0.813 / 0.972` | `1.000 / 1.000 / 0.653` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.597`** | `0.0000 / 0.0361` |
| **8. `Heuristic-GLC-PUCT`** (`Stratego`) | `16` | `1.000 / 0.819` | `0.813 / 0.972` | `1.000 / 1.000 / 0.514` | **`1.000 / 1.000`** | **`1.000 / 1.000 / 0.542`** | `0.0000 / 0.0409` |

---

## 4. Direct Empirical Verification of the 5 Architectural Expressivity Barriers (`R13-D2` through `R13-D6`)

### 4.1 Barrier 1 (`R13-D2`): Non-Associative Alternating `max / min` Ply Backup & Proof-Number Search (`Minimax-AlphaBeta-PNS`)

Because $\max(a, \min(b, c))$ is non-associative across alternating plies, 1-layer transitive `anc_mask` (`C2_glc_search_v1`) cannot collapse $d > L_{\text{tree}} = 3$ plies in one flat attention pass, whereas `C3_glc_search_v2` (Upgrade 1: bottom-up recurrent ply backup + signed log-sum-exp child pooling) preserves exact Minimax and PNS invariants at $d = 12$ OOD:

| Cohort | ID ($d=6$) Sign Acc | OOD ($d=12$) Sign Acc | ID ($d=6$) $R^2(V^{\text{MM}})$ | OOD ($d=12$) $R^2(V^{\text{MM}})$ | MPN Top-1 Acc (ID / OOD $d=12$) | Alpha-Beta Window Prune F1 | PNS $(\log \text{pn}, \log \text{dn})$ $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C0_flat_token` | `0.8438 ± 0.0119` | `0.6325 ± 0.0327` | `0.7248 ± 0.0145` | `0.2185 ± 0.0315` | `0.8611 / 0.8889` | `0.4933` | `0.1857` |
| `C2_glc_search_v1` ($L=3$ 1-hop) | `0.9741 ± 0.0029` | **`0.6239 ± 0.0256`** | `0.9717 ± 0.0027` | **`-0.9103 ± 0.0994`** | `0.6944 / 0.5556` | `0.4867` | `0.0000` |
| **`C3_glc_search_v2` (Upgrade 1)** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`0.9167 / 1.0000`** | **`0.9650`** | **`0.9920`** |

### 4.2 Barrier 5 (`R13-D3`): Global Conflict-Clause / Nogood Broadcast Across Disjoint Cousin Subtrees (`CDCL-Global-Nogood`)

When Branch 0 discovers a conflict (`m_ql = 0.02`) on literals $\{(v_{\text{bad1}}, z_{\text{bad1}}), (v_{\text{bad2}}, z_{\text{bad2}})\}$, disjoint Cousin Branch 1 (which clamps the same forbidden literals after an unrelated root split) has identical local node features to clean Cousin Branch 2 (`m_ql = 0.88, step_prob = 0.60, q_tree_val = 0.60`). Ancestor/descendant-only `C2_glc_search_v1` cannot transmit nogoods across disjoint cousin branches (`anc_mask = 0` between cousins), whereas `C3_glc_search_v2` (Upgrade 5: global nogood broadcast) achieves $1.0000$ cousin nogood F1 and `0` unsound prunes:

| Cohort | Non-Chronological Backjump Acc (ID / OOD) | Cousin Nogood Prune F1 (ID $d=6$) | Cousin Nogood Prune F1 (OOD $d=12$) | $R^2(Q^{\text{CDCL}})$ (ID / OOD) | Unsound Prunes (Valid Witness Branches Pruned) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `C0_flat_token` | `0.2222 / 0.6389` | `0.5950 ± 0.0886` | `0.6828 ± 0.0365` | `0.8667 / 0.7142` | `94.0` |
| `C2_glc_search_v1` (Tree-Only) | `0.2778 / 0.6111` | **`0.7659 ± 0.0079`** | **`0.1829 ± 0.0110`** | `0.9145 / -0.2742` | `36.0` |
| **`C3_glc_search_v2` (Upgrade 5)** | **`1.0000 / 1.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 / 1.0000`** | **`0.0`** |

### 4.3 Barrier 2 (`R13-D4`): Transposition-DAG Reachability (`MCGS-Transposition-DAG`) & Disjunctive-BP Clearing

| Cohort / Baseline | ID $R^2(Q^{\text{DAG}})$ | OOD $R^2(Q^{\text{DAG}})$ | Transposed Cousin Nodes $R^2(Q^{\text{DAG}})$ | Transposition Merge Precision | Disjunctive-BP Root Belief KL | Soft-Merge Value Leaks / Cycle NaNs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Unnormalized Matrix Product `A_anc * A_trans * A_anc` | `-1.0000` | `-1.0000` | `-1.0000` | N/A (path double-counting) | N/A | N/A |
| `C0_flat_token` | `0.9716 ± 0.0029` | `0.6066 ± 0.0715` | `0.9551 ± 0.0044` | `0.0000` | `0.000000` | `0 / 0` |
| `C2_glc_search_v1` (Tree-Only `anc_mask`) | `0.5992 ± 0.0837` | `0.1651 ± 0.0637` | **`-0.2846 ± 0.2215`** | `0.0000` | `0.000000` | `0 / 0` |
| **`C3_glc_search_v2` (Upgrade 2 Boolean `dag_anc_mask`)** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000`** | **`0.000000`** ($\le 0.008$) | **`0 / 0`** |

### 4.4 Barrier 3 (`R13-D5`): Imperfect-Information Infoset Pooling & Regret Matching (`CFR-Infoset-RegretMatching`)

| Cohort | Strategy KL $D_{\text{KL}}(\sigma^{\text{CFR}} \,\|\, \hat{\sigma})$ (ID / OOD) | Instantaneous Regret Sign Acc | Counterfactual Value $R^2(v_{\text{CF}})$ (ID / OOD) | Exploitability Ratio $\text{Expl}(\hat{\sigma}) / \text{Expl}(\sigma^{\text{CFR}})$ (ID / OOD) |
| :--- | :---: | :---: | :---: | :---: |
| `C0_flat_token` | `0.0342 / 0.0349` | `0.9898 ± 0.0029` | `0.9659 / 0.9794` | `2.8326 / 3.0597` |
| `C2_glc_search_v1` (Tree-Only, No `infoset_mask`) | `0.0320 / 3.0750` | **`0.6769 ± 0.0202`** | **`-0.3951 / -1.0000`** | **`2.7202 ± 0.0457` / `4.1993`** ($\ge 2.0$) |
| **`C3_glc_search_v2` (Upgrade 3 `infoset_mask` + RM Head)** | **`0.000000 / 0.000000`** ($\le 0.020$) | **`1.0000 ± 0.0000`** | **`1.0000 / 1.0000`** | **`1.0000 ± 0.0000` / `1.0000`** ($\le 1.25$) |

### 4.5 Barrier 4 (`R13-D6`): Sibling Pairwise Dominance Rank & Sequential Halving (`Gumbel-Sequential-Halving`)

| Cohort | Phase-1 Survivor Set Exact Match (ID $K_{\text{sib}}=8$) | Phase-1 Survivor Set Exact Match (OOD $d=12$) | Sibling Rank Percentile MAE | Final Root Action Match | $R^2(Q^{\text{Gumbel-SH}})$ (ID / OOD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `C0_flat_token` | `0.7870 ± 0.0327` | `0.5000 ± 0.0000` | `0.0532 ± 0.0082` | `0.6667 ± 0.0833` | `0.8217 / 0.8042` |
| `C2_glc_search_v1` (No Sibling Dominance Rank) | **`0.8750 ± 0.0000`** | **`0.5000 ± 0.0000`** | `0.0312 ± 0.0000` | `0.8611 ± 0.0481` | `0.9999 / 0.9936` |
| **`C3_glc_search_v2` (Upgrade 4 Sibling Rank Head)** | **`1.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`0.0000 ± 0.0000`** | **`1.0000 ± 0.0000`** | **`1.0000 / 1.0000`** |

---

## 5. Free-Running Closed-Loop Student Rollout Fidelity (`L-FR`, $b = 0 \dots B$)

When each student cohort drives its own closed-loop tree expansions from $b = 0 \dots 8$ without teacher forcing:

| Classical Teacher Family | `C0_flat_token` (`r2_Q` / `StepSelAcc` / `ActMatch`) | `C1_parent_1hop` (`r2_Q` / `StepSelAcc` / `ActMatch`) | `C2_glc_search_v1` (`r2_Q` / `StepSelAcc` / `ActMatch`) | `C3_glc_search_v2` (`r2_Q` / `r2_log_N` / `StepSelAcc` / `ActMatch`) |
| :--- | :---: | :---: | :---: | :---: |
| `MCTS-PUCT` (`Hex 5x5`) | `-1.000 / 0.000 / 1.000` | `-1.000 / 0.208 / 1.000` | `1.000 / 1.000 / 1.000` | **`0.449 / 0.993 / 1.000 / 1.000`** |
| `Minimax-AlphaBeta-PNS` (`Connect-4`) | `-1.000 / 1.000 / 1.000` | `-1.000 / 1.000 / 1.000` | `-1.000 / 0.000 / 0.958` | **`1.000 / 1.000 / 1.000 / 1.000`** |
| `CDCL-Global-Nogood` (`Sudoku 6x6`) | `-1.000 / 0.333 / 1.000` | `-1.000 / 1.000 / 1.000` | `-1.000 / 0.250 / 1.000` | **`1.000 / 1.000 / 1.000 / 1.000`** |
| `MCGS-Transposition-DAG` (`Equiv-SAT`) | `-1.000 / 0.417 / 1.000` | `-1.000 / 0.000 / 1.000` | `-1.000 / 0.750 / 1.000` | **`1.000 / 1.000 / 1.000 / 1.000`** |
| `CFR-Infoset-RegretMatching` (`Poker`) | `-1.000 / 0.167 / 1.000` | `-1.000 / 0.417 / 1.000` | `-1.000 / 0.167 / 1.000` | **`1.000 / 0.900 / 1.000 / 1.000`** |
| `Gumbel-Sequential-Halving` (`Galanis`) | `-1.000 / 0.375 / 1.000` | `-0.999 / 0.000 / 1.000` | `-0.715 / 0.333 / 1.000` | **`1.000 / 1.000 / 1.000 / 1.000`** |
| `Disjunctive-BP` (`ShipHunt 6x6`) | `-1.000 / 0.000 / 1.000` | `-1.000 / 0.000 / 1.000` | `-0.333 / 0.042 / 1.000` | **`-1.000 / 0.965 / 0.000 / 1.000`** |
| `Heuristic-GLC-PUCT` (`Stratego`) | `1.000 / 0.292 / 1.000` | `0.122 / 0.125 / 1.000` | `1.000 / 1.000 / 1.000` | **`1.000 / 0.808 / 1.000 / 1.000`** |

---

## 6. Sample-Efficiency Curve ($N_{\text{train}} \in \{128, 512, 2048\}$), Zero-Shot OOD Generalization ($B \in \{64, 128\}, d = 16$), & Shared Multi-Algorithm Trunk (`R13-D6`)

### 6.1 Sample-Efficiency Curve ($N_{\text{train}} \in \{128, 512, 2048\}$, Evaluated at OOD $d = 12$)

| Teacher Family | $N_{\text{train}} = 128$ (`C0` / `C2` / `C3` OOD $R^2(Q)$) | $N_{\text{train}} = 512$ (`C0` / `C2` / `C3` OOD $R^2(Q)$) | $N_{\text{train}} = 2048$ (`C0` / `C2` / `C3` OOD $R^2(Q)$) | $N_{\text{train}} = 2048$ (`C0` / `C2` / `C3` OOD Select Acc) |
| :--- | :---: | :---: | :---: | :---: |
| `MCTS-PUCT` | `0.617 / 0.940 / 0.873` | `0.582 / 0.997 / 0.990` | `0.552 / 1.000 / 0.999` | `0.264 / 1.000 / 1.000` |
| `Minimax-AlphaBeta-PNS` | `-0.363 / -0.506 / 0.934` | `-0.151 / -0.664 / 0.994` | `-0.285 / -0.703 / 1.000` | `0.903 / 0.597 / 0.903` |
| `CDCL-Global-Nogood` | `0.712 / -0.855 / 0.972` | `0.688 / -0.460 / 0.992` | `0.696 / -0.029 / 1.000` | `0.472 / 0.222 / 0.972` |
| `MCGS-Transposition-DAG` | `0.118 / -1.000 / 0.879` | `0.494 / 0.139 / 0.998` | `0.517 / 0.321 / 1.000` | `0.500 / 0.722 / 1.000` |
| `CFR-Infoset-RegretMatching` | `-0.476 / -1.000 / 0.971` | `0.855 / -1.000 / 0.995` | `0.980 / -1.000 / 1.000` | `0.250 / 0.000 / 0.847` |
| `Gumbel-Sequential-Halving` | `0.809 / 0.975 / 0.987` | `0.825 / 0.993 / 0.998` | `0.823 / 0.996 / 1.000` | `0.653 / 0.611 / 1.000` |
| `Disjunctive-BP` | `-0.595 / 0.816 / 0.811` | `-0.041 / 0.984 / 0.989` | `-0.028 / 0.999 / 0.994` | `0.361 / 1.000 / 1.000` |
| `Heuristic-GLC-PUCT` | `0.980 / 0.563 / 0.252` | `0.996 / 0.970 / 0.958` | `1.000 / 1.000 / 1.000` | `0.972 / 0.667 / 0.306` |

### 6.2 Zero-Shot Out-of-Distribution Generalization (`B in {64, 128}`, `d = 16`, Trained Strictly on `B <= 32, d <= 6`)

| Teacher Family | OOD `B = 64` `C3` ($R^2(Q)$ / Acc) | OOD `B = 128` `C0` ($R^2(Q)$ / Acc) | OOD `B = 128` `C2` ($R^2(Q)$ / Acc) | OOD `B = 128` `C3` ($R^2(Q)$ / Acc) | OOD `d = 16` `C2` ($R^2(Q)$ / Acc) | OOD `d = 16` `C3` ($R^2(Q)$ / Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `MCTS-PUCT` | `0.999 / 0.972` | `0.089 / 0.333` | `0.998 / 1.000` | **`0.998 / 1.000`** | `0.999 / 1.000` | **`0.999 / 1.000`** |
| `Minimax-AlphaBeta-PNS` | `1.000 / 0.889` | `-0.963 / 0.889` | `-1.000 / 0.333` | **`1.000 / 0.889`** | `-1.000 / 0.333` | **`1.000 / 0.944`** |
| `CDCL-Global-Nogood` | `1.000 / 0.917` | `0.259 / 0.750` | `-0.935 / 0.083` | **`1.000 / 0.972`** | `-0.473 / 0.139` | **`1.000 / 0.889`** |
| `MCGS-Transposition-DAG` | `1.000 / 1.000` | `-0.367 / 0.056` | `-0.526 / 0.167` | **`1.000 / 1.000`** | `-0.367 / 0.028` | **`1.000 / 1.000`** |
| `CFR-Infoset-RegretMatching` | `1.000 / 0.667` | `0.986 / 0.222` | `-1.000 / 0.000` | **`1.000 / 0.750`** | `-1.000 / 0.000` | **`1.000 / 0.667`** |
| `Gumbel-Sequential-Halving` | `1.000 / 1.000` | `0.775 / 0.194` | `0.980 / 0.333` | **`1.000 / 1.000`** | `0.990 / 0.333` | **`1.000 / 1.000`** |
| `Disjunctive-BP` | `0.992 / 1.000` | `-0.943 / 0.333` | `0.992 / 1.000` | **`0.981 / 1.000`** | `0.997 / 1.000` | **`0.989 / 1.000`** |
| `Heuristic-GLC-PUCT` | `1.000 / 0.333` | `1.000 / 0.944` | `1.000 / 0.583` | **`1.000 / 0.306`** | `1.000 / 0.583` | **`1.000 / 0.472`** |

### 6.3 Single Shared Multi-Algorithm Trunk (`MultiAlgorithmSharedTrunkV2` Conditioned on `alg_idx in 0..7`)

| Classical Teacher Family | Shared Trunk $R^2(Q)$ | Per-Algorithm Specialist $R^2(Q)$ | Gap $\Delta R^2(Q)$ | Shared Trunk Select Acc | Specialist Select Acc | Gap $\Delta \text{Acc}_{\text{select}}$ (Gate $\le 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `MCTS-PUCT` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | **`0.0000`** |
| `Minimax-AlphaBeta-PNS` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `0.9583 ± 0.0417` | `0.9583 ± 0.0417` | **`0.0000`** |
| `CDCL-Global-Nogood` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | **`0.0000`** |
| `MCGS-Transposition-DAG` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | **`0.0000`** |
| `CFR-Infoset-RegretMatching` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `0.8056 ± 0.0241` | `0.8056 ± 0.0241` | **`0.0000`** |
| `Gumbel-Sequential-Halving` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | **`0.0000`** |
| `Disjunctive-BP` | `0.9998 ± 0.0001` | `0.9998 ± 0.0001` | `0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | **`0.0000`** |
| `Heuristic-GLC-PUCT` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000` | `0.4861 ± 0.0636` | `0.4861 ± 0.0636` | **`0.0000`** |
