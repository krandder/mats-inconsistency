# Pre-Registration: Round 12 (`R12-GPU-NEURAL-V2`) — Calibrated Combinatorial Queries, Iso-Update Header-FiLM Generalist & Ultra-Deep $B=256$ KV-Tree Search on NVIDIA L4 GPU

- **Document ID:** `PREREG-R12-GPU-V2`
- **Parent Baseline:** `RESULTS-R11-GPU-NEURAL.md` (`r11_gpu_neural_results.json`)
- **Seeds:** `101, 102, 103` ($n = 3$ independent seeds, two-sided 95% Student-$t$ CIs with $t_{0.975, 2} = 4.3027$)
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB` VRAM) via `gpu-run.sh`

---

## 1. Motivation & Targeted Hypotheses from Round 11

In Round 11 (`RESULTS-R11-GPU-NEURAL.md`), our real SGD-trained `GLCAOMWorldModel` + `TreePagedKVSearch` passed `R11-D0` (Zero Oracle Leakage: `0` AST violations, `PoisonTensor` `max_diff = 0.000000000`), `R11-D1`, `R11-D2`, `R11-D3`, and `R11-D5`, and achieved **partial passes** on `R11-D4` and `R11-D6`:

1. **`R11-D4` Diagnosis (High Query Accuracy `78.5%`, Uncalibrated `query_kl`):**
   - `A4_GLC_BilinearAux_QuerySteered` reduced variable KL on `Equiv-SAT` by **`209x`** (`0.0079` vs `1.6535` for `A1_NoAux`) and achieved the highest query accuracy (`78.5%` on `Sudoku-6x6`, `58.1%` on `Equiv-SAT`), but failed raw `query_kl <= 0.15` (`0.8735` / `1.0219`) due to:
     - (a) Edge-only `z_aux_edge` missing cross-depth path co-fluctuations $\Delta_{1,A}\Delta_{2,B}$ when variable $A$ is clamped at depth 1 and variable $B$ responds at depth 2.
     - (b) Unconstrained `AdamW` updates on `u_bilin, v_bilin` inflating the Parseval Johnson-Lindenstrauss basis norm by $\sim 4.5\times$ and over-saturating raw query log-odds at depth $B$.
     - (c) Absence of post-depth temperature/Platt calibration $(\tau_q(B), b_q(B))$ fitted on `split="cal"`.
2. **`R11-D6` Diagnosis (Update-Asymmetric Generalist vs. Specialist Elo Gap):**
   - `Unified-Generalist` (`2.04M` shared params, `15` steps/env) beat `Single-Game-Specialist` (`24.43M` total params, `60` steps/family) by **`-32.6%`** in single-agent KL (`0.1235` vs `0.1833`), **`4.06x`** in compute-scaling slope $\kappa$ (`24.48` vs `6.03`), and **`+1.199` nats** on OOD cross-game transfer, but trailed in raw in-distribution multiplayer Elo (`1123.6` vs `1438.0`) because each specialist received $4\times$–$12\times$ more per-game parameter updates and dedicated per-game heads.
3. **Ultra-Deep $B = 256$ Horizon Question:**
   - In Round 11, `GLC_Tree_Active_PUCT` was still actively improving at $B = 128$ (`KL = 0.3046`, `Mini-ECI = 1327.0`). Round 12 tests whether `TreePagedKVSearch` continues improving from $B = 128 \to 256$ across all 12 environments without spectral blowup or anytime KL regressions.

---

## 2. Pre-Registered Gates (`R12-D0` through `R12-D3`)

### Gate `R12-D0`: Zero Oracle Leakage Gate (`S1`–`S6` + `R1`–`R4`)
- **Static AST Audit (`S1`–`S6`):** `src/eval/ast_leak_auditor.py` audits `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `benchmarks/run_r11_gpu_neural.py`, and `benchmarks/run_r12_gpu_v2.py` (`0` violations; `3/3` synthetic canaries caught).
- **Runtime `PoisonTensor` & Canary Invariance (`R1`–`R4`):** Across all 12 environments and all 3 seeds, `PoisonTensor`-stripped `sub_blind`, all-zeros `sub_zeros`, and label-permuted `sub_perm_blind` produce bitwise-identical model and tree-search outputs (`global_max_diff == 0.000000000`).

### Gate `R12-D1`: Post-Depth `split="cal"` Platt/Temperature Calibration for `CombinatorialQueryHead`
- **Protocol (`R12-E1`):**
  - Keep `u_bilin.weight` and `v_bilin.weight` anchored on the Parseval Johnson-Lindenstrauss basis while training `cofluct_scale`, `neural_aux_head`, `prospectus_aux_proj`, `ctx_u`, `ctx_v`, `w_aux_to_v`, and `top_down_gain`.
  - Include both edge-local (`z_aux_edge`) and root-to-node path-cumulative (`z_aux_root`) bilinear co-fluctuations in `TreePagedKVBank.append_node`.
  - Fit a 2-parameter post-depth Platt / temperature scaling $(\tau_q(B), b_q(B))$ **strictly on held-out `split="cal"`** (minimizing Bernoulli KL on `cal_batches`, never touching `split="test"`!) and evaluate both uncalibrated (`raw_query_kl`) and `cal`-calibrated (`cal_query_kl`, `cal_query_brier`, `query_acc`) on blind `split="test"` for `A1_NoAux`, `A2_LinearAux`, `A3_BilinearAux_RandomSearch`, and `A4_GLC_BilinearAux_QuerySteered` across `B in {0, 2, 4, 8, 16, 32}` on `Equiv-SAT` and `Sudoku-6x6`.
- **Pass Criteria:**
  - `A4_GLC_BilinearAux_QuerySteered` at $B=16$ achieves `cal_query_kl <= 0.15` nats on both `Equiv-SAT` and `Sudoku-6x6`, beats `A1_NoAux`, `A2_LinearAux`, and `A3_BilinearAux_RandomSearch` in both `query_acc` and `cal_query_kl`, and maintains `Participation Ratio >= 0.25`.

### Gate `R12-D2`: Iso-Update + Header-FiLM `Unified-Generalist` vs. `Single-Game-Specialist` (`12x` Parameter Compression)
- **Protocol (`R12-E2`):**
  - Equip `Unified-Generalist-FiLM-IsoUpdate` with zero-initialized lightweight epistemic Header-FiLM modulation (`use_header_film=True`, `2.08M` total shared parameters vs. `24.43M` for the 12-env specialist ensemble — an **`11.8x` parameter reduction**) and train with matched per-environment gradient updates (`720` total steps = `60` steps/env, matching `60` steps for `Single-Game-Specialist`).
  - Compare `Unified-Generalist-R11 (15 steps/env, 2.04M)`, `Unified-Generalist-FiLM-IsoUpdate (60 steps/env, 2.08M)`, and `Single-Game-Specialist (60 steps/family, 24.43M)` across all 12 environments.
- **Pass Criteria:**
  - `Unified-Generalist-FiLM-IsoUpdate` (`2.08M` shared params) beats `Single-Game-Specialist` (`24.43M` total params) on single-agent KL (`<` specialist KL), matches or beats `Single-Game-Specialist` on multiplayer Elo at $B=128$ (and $B=256$) and overall `Mini-ECI`, achieves `>= +0.50` nats OOD cross-game KL advantage, and preserves exact `0.000000` LIR invariance (`d_bias_b0 == 0.000000`, `post_rl_belief_drift == 0.000000`).

### Gate `R12-D3`: Ultra-Deep KV-Tree Search up to $B = 256$
- **Protocol (`R12-E3`):**
  - Evaluate blind test-time KV-tree search across **`B in {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}`** (`9` transitions) across all 12 environments on `NVIDIA L4` GPU.
- **Pass Criteria:**
  - `GLC_Tree_Active_PUCT` (`Unified-Generalist-FiLM-IsoUpdate` and `GLC-AOM-KV-Tree`) continues improving from $B=128 \to 256$ (`KL(B=256) < KL(B=128)`, `Witness(B=256) >= Witness(B=128)`, `Mini-ECI(B=256) > Mini-ECI(B=128)`), maintains Gershgorin spectral bound `||T_b||_infty <= 1.05` at $B=256$, and achieves `9/9` non-increasing anytime KL transitions (`max_kl_rise <= 0.002`).
