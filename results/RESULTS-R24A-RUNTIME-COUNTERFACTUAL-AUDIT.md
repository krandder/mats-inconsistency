> **Independent audit (2026-10-19): verdict partly (self-audit).** The claims in this file did not pass an independent Opus 5.5 audit; see `docs/audits/gemini-campaign-critique-2026-10-19.md` (all rounds), `docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md` and `docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md`. Numbers below are the runner's outputs, not evidence about the architecture.

# Empirical Results Report: Round `R24A` — Stage-3 Runtime Name-Agnostic Counterfactual Audit (`Rules S11–S14`)

**Round ID:** `R24A` (`Stage-3 Runtime Name-Agnostic Counterfactual Audit Suite S11–S14`)  
**Author:** Member 33 (`Runtime Counterfactual Auditor S11–S14` Builder & Verifier, Society of Thinkers)  
**Pre-Registration Protocol:** [benchmarks/PREREG-R24A-RUNTIME-COUNTERFACTUAL-AUDIT.md](file:///home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R24A-RUNTIME-COUNTERFACTUAL-AUDIT.md)  
**Auditor Implementation:** [src/eval/runtime_counterfactual_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/runtime_counterfactual_auditor.py)  
**Benchmark Runner:** [benchmarks/run_r24a_runtime_counterfactual_audit.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r24a_runtime_counterfactual_audit.py)  
**Machine-Readable JSON Artifact:** [results/r24a_runtime_counterfactual_audit.json](file:///home/agent/workspaces/inconsistency-search/results/r24a_runtime_counterfactual_audit.json)  
**Checked At:** `2026-10-19T05:21:13Z` (Read-only verification check: `python3 benchmarks/run_r24a_runtime_counterfactual_audit.py` $\to$ exit code `0`, `"overall_pass": true`)

---

## 1. Executive Summary

Round `R24A` resolves the Society of Thinkers Round 6 critique `Opus-5` / `Fable-4` (*"Static AST Auditing Can Be Bypassed via Identifier Renaming"*, `society/round6/04-opus-round6-audit.md` §5 and `05-fable-round6-audit.md` §3) by pairing the Stage-1 static AST auditor ([src/eval/ast_leak_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/ast_leak_auditor.py), `Rules S1–S10`) with a name-agnostic **Stage-3 Runtime Behavioral Counterfactual Audit Suite** ([src/eval/runtime_counterfactual_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/runtime_counterfactual_auditor.py), `Rules S11–S14`).

| Verification Dimension | Pre-Registered Gate | Empirical Result (`R24A`) | Status |
| :--- | :--- | :--- | :--- |
| **Combined Canary Suite (`S1–S14`)** | `11 / 11` (`7/7` AST + `4/4` Runtime) | **`11 / 11` (`100.0%` caught)** | **PASS** |
| **Stage-1 Static AST Audit (`S1–S10`) on Wave-1 (`R18–R21`)** | Must flag violations ($> 0$) | **`73` violations flagged (`R18`: 39, `R19`: 5, `R20`: 6, `R21`: 23)** | **CAUGHT (`FAIL` as predicted)** |
| **Stage-1 Static AST Audit (`S1–S10`) on Wave-2 (`R22A, R22B, R23`) + `R24A`** | `0` violations across all 5 files | **`0` violations across all 5 files** | **PASS** |
| **Rule `S11` (`Graph Equivariance S11-E & 1-Hop Locality S11-L`)** | Wave-2 PASS ($\|\Delta_{\text{equiv}}\|_\infty < 10^{-5}$, $\Delta_{t=1} < 10^{-7}$, $\Delta_{t=L} > 10^{-5}$); Wave-1 FAIL | Wave-2: **PASS** ($\max \|\Delta_{\text{ptr}}\|_\infty = 1.79 \times 10^{-7}$, $\Delta_{t=1} = 0.0$, $\min \Delta_{t=L} = 2.44 \times 10^{-4}$)<br>Wave-1 (`R21-E1`): **CAUGHT** ($\|\Delta_{\text{mask}}\|_\infty = 1.20 \times 10^{-2}$, $\Delta_{t=1} = 6.81 \times 10^{-2}$) | **PASS** |
| **Rule `S12` (`Step-t=2 Search State Intervention S12-2 & Ratio Dispersion S12-4`)** | Wave-2 PASS ($\Delta_{t=1} \le 10^{-8}$, $\Delta_{t'>2} > 10^{-5}$, $\operatorname{Std}_i(r_i(B)) > 10^{-4}$); Wave-1 FAIL | Wave-2 (`R22B-E3`): **PASS** ($\Delta_{t=1} = 0.0$, $\Delta_{t'>2} = 5.88 \times 10^{-3}$, $\min \operatorname{Std}_i = 0.1862$)<br>Wave-1 (`R21-E2`): **CAUGHT** ($\Delta_{t'>2} = 0.0$, $\min \operatorname{Std}_i = 0.0$) | **PASS** |
| **Rule `S13` (`Track-0 Helper Baseline <= 5% & Symmetric Ledger`)** | Wave-2 PASS (Track-0 solve $\le 5.0\%$, symmetric helper calls); Wave-1 FAIL | Wave-2 (`R22B-E1`): **PASS** (Track-0 solve $= 0/48 = 0.00\%$, call spread $= 0$, node spread $= 0$)<br>Wave-1 (`R19`): **CAUGHT** (Track-0 solve $= 31/48 = 64.58\%$, call spread $= 3$, node spread $= 292$) | **PASS** |
| **Rule `S14` (`Counterfactual Broadcast Price Replay & Zero Peer Leak`)** | Wave-2 PASS (`grad_fn=None`, peer drift $\le 10^{-8}$, price shift $> 10^{-4}$); Wave-1 FAIL | Wave-2 (`R23-E2` `M4`/`M5`): **PASS** (`grad_fn=False`, peer drift $= 0.0$, price shift $= 0.2168$ / $0.2361$)<br>Wave-1 (`R20-E2` `P1`/`P5`): **CAUGHT** (`P1` price shift $= 0.0$; `P5` peer drift $= 1.3776$, price shift $= 0.0$) | **PASS** |

---

## 2. Canary Self-Test Verification (`11 / 11` Caught: `7 / 7` AST + `4 / 4` Runtime)

To prove that `Rules S11–S14` catch shortcuts even when every variable, function, and attribute is renamed to benign identifiers that pass static AST scanning (`0` violations under `S1–S10`), [src/eval/runtime_counterfactual_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/runtime_counterfactual_auditor.py) executes 4 synthetic runtime cheat canaries alongside the 7 AST canaries from [src/eval/ast_leak_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/ast_leak_auditor.py):

| Canary ID | Layer | Benignly Named Shortcut Mechanism | Empirical Detection Telemetry | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `AST-Canary-1` (`S1`) | Stage-1 AST | Reads `.exact_marginals` inside non-exempt function | Flagged at line 3 (`Rule S1`) | **CAUGHT (`1/7`)** |
| `AST-Canary-2` (`S3`) | Stage-1 AST | Calls `mechanistic_redundancy_delta_c` inside generator | Flagged at line 3 (`Rule S3`) | **CAUGHT (`2/7`)** |
| `AST-Canary-3` (`S4`) | Stage-1 AST | Branches on `if seed == 102 and b_budget == 2:` | Flagged at line 3 (`Rule S4`) | **CAUGHT (`3/7`)** |
| `AST-Canary-4` (`S7`) | Stage-1 AST | Reads `policy.cfr_reference_table[state]` | Flagged at line 3 (`Rule S7`) | **CAUGHT (`4/7`)** |
| `AST-Canary-5` (`S8`) | Stage-1 AST | Branches on `if paradigm == "GLC":` with `1.25 + 0.52 * log_b` | Flagged at line 3 (`Rule S8`) | **CAUGHT (`5/7`)** |
| `AST-Canary-6` (`S9`) | Stage-1 AST | Asymmetric `_complete_leaf_with_observable_search` call | Flagged at line 3 (`Rule S9`) | **CAUGHT (`6/7`)** |
| `AST-Canary-7` (`S10`) | Stage-1 AST | Writes `dist_vec` into `node_feats[b, :, 8]` | Flagged at line 3 (`Rule S10`) | **CAUGHT (`7/7`)** |
| `S11-Canary` (`S11`) | Stage-3 Runtime | `_S11CheatGraphProcessor` adds `canonical_slot = arange(N)/N` + `global_edge_summary = weights.mean()` (0 AST violations) | `max_ptr_err = 5.253e-01 > 1e-5`, `max_mask_err = 1.317e-01 > 1e-5`, `delta_t1 = 1.953e-02 >= 1e-7` | **CAUGHT (`8/11`)** |
| `S12-Canary` (`S12`) | Stage-3 Runtime | `_s12_canary_runner` ignores step-$t=2$ state perturbation and scales loss by `B ** (-0.40)` (0 AST violations) | `delta_final = 0.000e+00 <= 1e-5`, `Std_i(L_i(4)/L_i(1)) = 0.000e+00 <= 1e-4` | **CAUGHT (`9/11`)** |
| `S13-Canary` (`S13`) | Stage-3 Runtime | Track-0 helper solves `31/48` (`64.58%`) with `0` neural calls + asymmetric Track-B helper calls (0 AST violations) | `track0_solve_rate = 64.58% > 5.0%`, `track_b_call_spread = 36`, `track_b_node_spread = 320` | **CAUGHT (`10/11`)** |
| `S14-Canary` (`S14`) | Stage-3 Runtime | `_s14_canary_runner` ignores `price_offset` and leaks peer module noise + `grad_fn=True` (0 AST violations) | `broadcast_has_grad_fn = True`, `peer_drift_l2 = 1.837e-01 > 1e-8`, `price_grad_shift_l2 = 0.000e+00 <= 1e-4` | **CAUGHT (`11/11`)** |

---

## 3. Stage-1 Static AST Audit (`Rules S1–S10`): Wave-1 vs. Wave-2 + `R24A`

| Audited File | Wave / Cohort | AST Violations (`S1–S10`) | Status |
| :--- | :--- | :---: | :--- |
| [benchmarks/run_r18_openspiel_stratego.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r18_openspiel_stratego.py) | Wave-1 (Retracted) | `39` | **FAIL (Caught)** |
| [benchmarks/run_r19_public_reasoning_benchmarks.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r19_public_reasoning_benchmarks.py) | Wave-1 (Retracted) | `5` | **FAIL (Caught)** |
| [benchmarks/run_r20_kyle_lmsr_futarchy.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r20_kyle_lmsr_futarchy.py) | Wave-1 (Retracted) | `6` | **FAIL (Caught)** |
| [benchmarks/run_r21_clrs_ood_and_scaling.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r21_clrs_ood_and_scaling.py) | Wave-1 (Retracted) | `23` | **FAIL (Caught)** |
| [benchmarks/run_r22a_genuine_openspiel_stratego.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r22a_genuine_openspiel_stratego.py) | Wave-2 (Genuine) | **`0`** | **PASS** |
| [benchmarks/run_r22b_genuine_public_reasoning_clrs.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r22b_genuine_public_reasoning_clrs.py) | Wave-2 (Genuine) | **`0`** | **PASS** |
| [benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py) | Wave-2 (Genuine) | **`0`** | **PASS** |
| [src/eval/runtime_counterfactual_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/runtime_counterfactual_auditor.py) | Wave-2 Audit (`R24A`) | **`0`** | **PASS** |
| [benchmarks/run_r24a_runtime_counterfactual_audit.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r24a_runtime_counterfactual_audit.py) | Wave-2 Audit (`R24A`) | **`0`** | **PASS** |

---

## 4. Stage-3 Runtime Counterfactual Audit (`Rules S11–S14`): Detailed Results

### 4.1 Rule `S11`: Graph Permutation Equivariance (`S11-E`) & 1-Hop Message-Passing Locality (`S11-L`)

Tested across all 5 neural graph architectures in `R22B-E2` (`DeepSets`, `MPNN_Max`, `RelTransformer`, `Triplet_GMPNN`, `GLC_AOM_v2`) and all 6 `CLRS-30` tasks (`BFS`, `DFS`, `Dijkstra`, `Bellman_Ford`, `MST_Prim`, `Articulation_Points`), versus Wave-1 `R21-E1` (`CLRSNeuralProcessor` with canonical vertex index `idx_norm = arange(N)/N` and precomputed global shortest-path distances at $t=0$):

| Target Processor | Max Pointer Equivariance Error $\|p(\pi(G)) - P p(G) P^\top\|_\infty$ (`< 1e-5`) | Max Mask Equivariance Error $\|m(\pi(G)) - P m(G)\|_\infty$ (`< 1e-5`) | 1-Hop Locality at $t=1$ $\Delta_{t=1}(s)$ (`< 1e-7`) | Multi-Hop Sensitivity at $t=L=6$ $\Delta_{t=L}(s)$ (`> 1e-5`) | Rule `S11` Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Wave-2 `R22B-E2` `DeepSets`** | `2.980e-08` | `5.960e-08` | **`0.000e+00`** | `6.657e-04` | **PASS** |
| **Wave-2 `R22B-E2` `MPNN_Max`** | `1.192e-07` | `0.000e+00` | **`0.000e+00`** | `4.969e-04` | **PASS** |
| **Wave-2 `R22B-E2` `RelTransformer`** | `1.490e-07` | `5.960e-08` | **`0.000e+00`** | `1.706e-03` | **PASS** |
| **Wave-2 `R22B-E2` `Triplet_GMPNN`** | `1.192e-07` | `5.960e-08` | **`0.000e+00`** | `2.435e-04` | **PASS** |
| **Wave-2 `R22B-E2` `GLC_AOM_v2`** | `1.788e-07` | `5.960e-08` | **`0.000e+00`** | `2.651e-04` | **PASS** |
| **Wave-1 `R21-E1` `CLRSNeuralProcessor`** | `4.427e-03` (**FAIL**) | `1.199e-02` (**FAIL**) | `6.810e-02` (**FAIL**) | `6.810e-02` | **FAIL (Caught)** |

---

### 4.2 Rule `S12`: Step-Wise Search State Intervention (`S12-2`) & Loss-Ratio Dispersion (`S12-4`)

Tested by injecting an $\epsilon = 0.05$ perturbation into the internal search tree state (`child_logits` / `TreePagedKVBank`) at rollout step $t=2$ (`step_b == 1`) after step $t=1$ (`B=1`) has completed, and measuring cross-instance KL loss-ratio standard deviation $\operatorname{Std}_{i=1..M}(L_i(B)/L_i(1))$ across $B \in \{4, 16\}$:

| Target Search System | Step-$t=1$ Causality Drift $\Delta_{t=1}$ (`<= 1e-8`) | Downstream Rollout Shift $\Delta_{t' > 2}$ (`> 1e-5`) | Ratio Dispersion $\operatorname{Std}_i(L_i(4)/L_i(1))$ (`> 1e-4`) | Ratio Dispersion $\operatorname{Std}_i(L_i(16)/L_i(1))$ (`> 1e-4`) | Rule `S12` Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Wave-2 `R22B-E3` `ScaledSearchModel` (`Tiny_0.8M`)** | **`0.000e+00`** | **`5.881e-03`** | **`0.1862`** (mean `0.5422`) | **`0.1900`** (mean `0.7252`) | **PASS** |
| **Wave-1 `R21-E2` `ScalingRunner`** | `0.000e+00` | `0.000e+00` (**FAIL**) | `3.925e-17` (**FAIL**) | `0.000e+00` (**FAIL**) | **FAIL (Caught)** |

---

### 4.3 Rule `S13`: Track-0 Helper-Only Baseline (`<= 5%`) & Symmetric Evaluator Ledger Gate

Tested across all 4 public reasoning tasks (`Sudoku_Extreme_81`, `SATLIB_3SAT_n20`, `Sokoban`, `Countdown`, $12$ test instances per task $= 48$ total instances) using an uninformative uniform prior ($p_{b,v,k} = 1/K$, **$0$ neural forward calls**):

| Reasoning Evaluator | `Sudoku-81` Track-0 Solved | `SATLIB-3SAT` Track-0 Solved | `Sokoban` Track-0 Solved | `Countdown` Track-0 Solved | Total Track-0 Solve Rate (`<= 5.0%`) | Track-A Helper Calls | Track-B Call / Node Spread across 7 Paradigms | Rule `S13` Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Wave-2 `R22B-E1` (`apply_symmetric_one_pass_unit_prop`)** | `0 / 12` (`0.0%`) | `0 / 12` (`0.0%`) | `0 / 12` (`0.0%`) | `0 / 12` (`0.0%`) | **`0 / 48` (`0.00%`)** | `0` on all 7 paradigms | Call spread `0`, Node spread `0` | **PASS** |
| **Wave-1 `R19` (`_complete_leaf_with_observable_search`)** | `6 / 12` (`50.0%`) | `9 / 12` (`75.0%`) | `12 / 12` (`100.0%`) | `4 / 12` (`33.3%`) | **`31 / 48` (`64.58%`)** (**FAIL**) | Non-zero (`0..3`) (**FAIL**) | Call spread `3`, Node spread `292` (**FAIL**) | **FAIL (Caught)** |

---

### 4.4 Rule `S14`: Paired Counterfactual Broadcast Price Replay & Zero Cross-Module Leakage Gate

Tested on the 12-module decentralized society on real `equiv_sat` instances (`R23-E2` `M4_Decentralized_Kyle_LMSR_Pinned_LIR` and `M5_Decentralized_Kyle_LMSR_Futarchy_Evolution`) versus Wave-1 `R20-E2` (`P1_Local_Greedy` and `P5` at `epoch=0`):

| Decentralized Market Regime | Broadcast Price `grad_fn` (`False`) | Peer Module Perturbation Gradient Drift $\|g_m(\theta_{-m}) - g_m(\theta_{-m}+\Delta)\|_2$ (`<= 1e-8`) | Counterfactual Broadcast Price Shift ($\delta_P = 0.20$) $\|g_m(P_t) - g_m(P'_t)\|_2$ (`> 1e-4`) | Rule `S14` Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Wave-2 `R23-E2` `M4_Decentralized_Kyle_LMSR_Pinned_LIR`** | **`False`** | **`0.000000000`** | **`0.216789`** | **PASS** |
| **Wave-2 `R23-E2` `M5_Decentralized_Kyle_LMSR_Futarchy_Evolution`** | **`False`** | **`0.000000000`** | **`0.236145`** | **PASS** |
| **Wave-1 `R20-E2` `P1_Local_Greedy`** | `False` | `0.000000000` | `0.000000` (**FAIL: Epiphenomenal Price**) | **FAIL (Caught)** |
| **Wave-1 `R20-E2` `P5` (`epoch=0` Peer Anchor Leak)** | `True` (**FAIL**) | `1.377566` (**FAIL: Peer Leak**) | `0.000000` (**FAIL: Epiphenomenal Price**) | **FAIL (Caught)** |

---

## 5. Reproducibility & Self-Checking Commands

All results in this report and in [results/r24a_runtime_counterfactual_audit.json](file:///home/agent/workspaces/inconsistency-search/results/r24a_runtime_counterfactual_audit.json) can be reproduced and verified in $< 15$ seconds via:

```bash
# 1. Run standalone 11/11 canary self-test (7/7 AST + 4/4 Runtime Counterfactual canaries)
python3 src/eval/runtime_counterfactual_auditor.py --self-test

# 2. Run full R24A Wave-1 vs. Wave-2 AST + Runtime Counterfactual Audit (Rules S1-S14)
python3 benchmarks/run_r24a_runtime_counterfactual_audit.py
```
