# Society of Thinkers — Member 5: Unified Multi-Game Environment & Token Interface Specification

**Author:** Member 5 (Unified Multi-Game Environment & Token Interface Architect)  
**Module:** `src/envs/` (`futarchy-fi/inconsistency-search`)

---

## 1. Hierarchical 6-Token Header (`HEADER_LEN = 6`)

Every environment batch (`EnvBatch`) begins with an unambiguous 6-token integer header tensor `header` of shape `[B, 6]`:
$$\mathrm{Header} \;=\; \big[\,\texttt{GAME\_FAMILY},\; \texttt{GAME\_ID},\; \texttt{GRID\_H},\; \texttt{GRID\_W},\; \texttt{INVENTORY\_SPEC},\; \texttt{UTILITY\_SPEC}\,\big]$$

| Environment Registry Key | `GAME_FAMILY` | `GAME_ID` | `(GRID_H, GRID_W)` | `INVENTORY_SPEC` | Default `UTILITY_SPEC` | $\|V\|$ (`num_vars`) | Domain $K_{\text{dom}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `equiv_sat` | `1` (`CONSTRAINT_CSP`) | `10` | `(1, 16)` | `1208` | `0` (`EPISTEMIC_TRUTH`) | `16` | `2` (`{0, 1}`) |
| `sudoku_6x6` | `1` (`CONSTRAINT_CSP`) | `20` | `(6, 6)` | `23` (box $2\times 3$) | `0` (`EPISTEMIC_TRUTH`) | `36` | `6` (`1..6`) |
| `sudoku_9x9_extreme` | `1` (`CONSTRAINT_CSP`) | `21` | `(9, 9)` | `33` (box $3\times 3$) | `0` (`EPISTEMIC_TRUTH`) | `81` | `9` (`1..9`) |
| `shiphunt_6x6` | `2` (`EPISTEMIC_PROBE`) | `30` | `(6, 6)` | `432` (ships $4,3,2$) | `3` (`SECTOR_NW_PREF`) | `36` | `2` (water/ship) |
| `shiphunt_8x8` | `2` (`EPISTEMIC_PROBE`) | `31` | `(8, 8)` | `543` (ships $5,4,3$) | `3` (`SECTOR_NW_PREF`) | `64` | `2` (water/ship) |
| `minesweeper_6x6` | `2` (`EPISTEMIC_PROBE`) | `40` | `(6, 6)` | `6` ($6$ mines) | `5` (`SAFE_SURVIVAL`) | `36` | `2` (safe/mine) |
| `hex_5x5` | `3` (`TACTICAL_PERFECT_INFO`) | `50` | `(5, 5)` | `25` | `1` / `2` (active player) | `25` | `2` (P0/P1 stone) |
| `connect4_6x7` | `3` (`TACTICAL_PERFECT_INFO`) | `51` | `(6, 7)` | `42` | `1` / `2` (active player) | `42` | `2` (P0/P1 disc) |
| `micro_stratego_4x4` | `4` (`TACTICAL_IMPERFECT_INFO`) | `60` | `(4, 4)` | `6` ($6$ slots, $720$ setups) | `1` (`PLAYER_0_WIN`) | `6` | `6` (`F,B,S,M,G,X`) |
| `micro_stratego_6x6` | `4` (`TACTICAL_IMPERFECT_INFO`) | `61` | `(6, 6)` | `8` ($8$ slots, $10{,}080$ setups) | `1` (`PLAYER_0_WIN`) | `8` | `6` (`F,B,S,M,G,X`) |
| `galanis_t3s111y2` | `5` (`PREDICTION_MARKET`) | `70` | `(1, 4)` | `11123` | `6` (`MARKET_HONEST_PROFIT`) | `4` | `2` (`d_a,d_b,d_c,X`) |
| `galanis_t3s110` | `5` (`PREDICTION_MARKET`) | `71` | `(1, 4)` | `11033` | `6` (`MARKET_HONEST_PROFIT`) | `4` | `2` (`d_a,d_b,d_c,X`) |
| `galanis_t3s111` | `5` (`PREDICTION_MARKET`) | `72` | `(1, 4)` | `11133` | `6` (`MARKET_HONEST_PROFIT`) | `4` | `2` (`d_a,d_b,d_c,X`) |
| `galanis_t3s111o2ye2` | `5` (`PREDICTION_MARKET`) | `73` | `(1, 4)` | `22223` | `6` (`MARKET_HONEST_PROFIT`) | `4` | `2` (`d_a,d_b,d_c,X`) |
| `proposal_poker` | `5` (`PREDICTION_MARKET`) | `80` | `(1, 5)` | `75` ($\Pr(s_i=x)=0.75$) | `6` (`MARKET_HONEST_PROFIT`) | `5` | `2` (`x,s_0,s_1,s_2,app`) |

---

## 2. Unified `EnvBatch` Tensor Interface (`src/envs/base.py`)

Every environment's `sample_batch(batch_size, split, seed, device)` returns an `EnvBatch` supporting:
1. **State & Observation Tokens (`obs_tokens`, `obs_mask`, `obs_features`):**
   - `obs_tokens`: `[B, N_vars]` (`0` = unresolved/masked, `1 .. K_dom` = observed/clamped assignment).
   - `obs_mask`: `[B, N_vars]` bool (`True` where variable is observed at root $O$ or clamped during tree search).
   - `obs_features`: `[B, N_vars, D_feat]` float32 spatial/incidence/sensor features per variable.
   - `step_unmask(batch, var_idx, val_idx)`: applies `STEP(v, z)` on a cloned batch for tree-paged KV-cache search.
2. **Present-Time Episodic Memory Tokens (`M_t` — `memory_tokens`, `memory_mask`):**
   - `memory_tokens`: `[B, N_mem, D_mem]` float32 carrying prior-turn cleared consensus beliefs $h_{\text{cons}}^{(t-w)}$ and recent public window observations (`Micro-Stratego`, `Galanis-Market`, `Proposal-Poker`).
3. **Direct Combinatorial / Relational Query Tokens (`Q(A, B)` — `query_pairs`, `query_targets`):**
   - `query_pairs`: `[B, N_queries, 3]` `(var_a, var_b, query_type)` (`QueryType.EQUAL` for `P(X_A == X_B)`, `QueryType.GREATER` for rank comparison).
   - `query_targets`: `[B, N_queries]` exact ground-truth relational probability $P^*(Q(A, B) \mid x_O)$.
4. **Modular Utility & Zero-Retraining Swaps (`utility_vector`, `swap_utility`):**
   - `swap_utility(batch, new_utility_spec)` swaps sector preferences (`SECTOR_NW_PREF` $\leftrightarrow$ `SECTOR_SE_PREF` in `ShipHunt`) or governance biases (`BIAS_PRO_APPROVE` $+y_i \leftrightarrow$ `BIAS_PRO_REJECT` $-y_i$ in `Proposal-Poker` and `Galanis-Market`) while keeping `exact_marginals` ($P^*$) strictly invariant.
5. **Exact Bayesian Oracles & Fast Vectorized Witness Checking (`exact_marginals`, `check_witness`):**
   - Every environment computes exact (or game-theoretic rollout) posteriors `exact_marginals` `[B, N_vars, K_dom]` and provides a pure-tensor `check_witness(batch, candidate_tokens)` verifier on CPU/GPU that checks observable constraints without peeking at hidden ground truth.
