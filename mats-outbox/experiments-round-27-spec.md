# Round 27 Experiment Specification (`R27A`, `R27B`, `R27C`) — Reoriented Inconsistency Program

- **Date:** `2026-10-20`
- **Governing Orientation:**
  - `/home/agent/workspaces/mats/outbox/research-orientation-2026-10-19.md` (`19:48Z` & `20:50Z` updates by `opus-mats` & Kelvin)
  - `/home/agent/workspaces/inconsistency-search/docs/direction/research-direction-note-v7-2026-10-19.md` (`v7`, integrating `InfiLoop` `arXiv:2610.11570`)
  - `/home/agent/workspaces/mats/outbox/gemini-campaign-critique-2026-10-19.md`
  - `/home/agent/workspaces/inconsistency-search/docs/PROTOCOL.md`
- **Overnight Goal Bead:** `pool-30gvjz` (*"I have to go for today, but I'd like you to continue to run many more experiments until we exhaust the $300 budget."*)
- **Role Separation (`docs/PROTOCOL.md`):**
  - `gemini` (`agy`): Proposer & Coder (`PREREG-R27*.md`, `run_r27*.py`, `scripts/score_r27.py`, `external/infiloop/`).
  - `opus-mats`: Freezer & Coordinator (reviews/freezes `freeze-R27`, dispatches GPU runs on `eurisko` / `calibrated-ai`, runs `scripts/score_r27.py`, dispatches `board-opus` audit).

---

## 1. Three Pre-Registered Experiments in Round 27

### Pillar A — `R27A`: Label-Free Puzzle Inconsistency Minimisation (`TRM` / `InfiLoop` / `TRMP` Comparison Class)
- **Pre-Registration:** `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R27A-LABELFREE-INFILOOP-TRMP.md`
- **Runner:** `/home/agent/workspaces/inconsistency-search/benchmarks/run_r27a_labelfree_infiloop_trmp.py`
- **Vendored Upstream Code:** `/home/agent/workspaces/inconsistency-search/external/infiloop/sudoku_model.py` (`pixeli99/InfiLoop`, SHA256 `44341a493e010548f8bdc36943523e345935705adea8797844d4764696e27f63`)
- **Core Questions Tested:**
  1. **Zero-Label Sudoku Solving (`R27A-D1`):** Can the upstream `InfiLoop` architecture (`Config(hidden_size=256, num_heads=8, layers=2, h_cycles=3, l_cycles=4)`) solve 9x9 Sudoku boards when trained with **strictly zero ground-truth solution labels** (`y_train` stripped before training), minimizing only the attention world-model PDG inconsistency energy $\mathcal{E}_{\text{Sudoku}}(p_\theta; x)$?
  2. **Multi-Solution `N-Queens` & Zero-Shot Size Transfer $8\times 8 \to 12\times 12, 16\times 16$ (`R27A-D2`):** On under-constrained $8\times 8$ `N-Queens` clue boards with $|\mathcal{S}(x)| \ge 2$ valid completions, does single-label supervised cross-entropy (`Public_InfiLoop_Supervised_SingleLabel`) suffer mode-averaging collisions across symmetric completions, whereas label-free port-equivariant inconsistency minimization (`Inconsistency_InfiLoop_LabelFree`) solves multi-solution boards and transfers zero-shot to $12\times 12$ and $16\times 16$?
  3. **Eliminating Carry-Last Latent Overthinking (`R27A-D3`):** Does inconsistency-gated `(N_t, D_t)` streaming attention residual (`InfiLoop`) maintain monotonically non-decreasing solve accuracy across segments $s \in \{1, 4, 12, 36\}$ and achieve $\ge 50\%$ lower late-segment cell turnover than `Carry-Last` (`TRM`)?

### Pillar B — `R27B`: Joint Hidden-State & Uncertain-Rule Discovery $\mu_t(Z_{\text{hid}}, R_{\text{rule}})$, Strategic Active VoI Probing, and Zero-Drift Utility Swaps
- **Pre-Registration:** `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R27B-UNCERTAIN-RULES-ACTIVE-VOI.md`
- **Runner:** `/home/agent/workspaces/inconsistency-search/benchmarks/run_r27b_uncertain_rules_active_voi.py`
- **Core Questions Tested (Addressing `gemini-campaign-critique-2026-10-19.md` §5(a).1):**
  1. **Leak-Audited Raw 3-Channel Input (`R27B-D0`):** Strictly 3 raw observable input channels (`[revealed_mask, obs_hit_or_mine, obs_neighbor_clue]`), zero target leakage, and all 5 mandatory controls (`Control_Uniform`, `Control_TrainMean`, `Control_ZeroParam_LLR`, `Control_Untrained_Init`, `Ceiling_Exact_Joint_Bayes`).
  2. **Joint Hidden-State & Latent-Rule Belief Calibration (`R27B-D1`):** On `ShipHunt-6x6` and `Minesweeper-6x6` where each board is drawn from $R=4$ candidate rule variants $R_{\text{rule}} \in \{r_1, r_2, r_3, r_4\}$ unknown at Episode Start, does joint recursive inconsistency minimization over $\mu_t(Z_{\text{hid}}, R_{\text{rule}})$ beat all 4 controls and the rule-blind baseline (`TRM_RuleBlind_Baseline`) in both state KL and rule KL?
  3. **Strategic Rule-Disambiguating Active VoI Probing (`R27B-D2`):** Does active probing via $\text{VoI}_{\text{rule}}(v; R_{\text{rule}})$ reduce true posterior rule entropy $H(P^*(R_{\text{rule}} \mid \mathcal{O}_K))$ by $\ge 25\%$ relative to random probing and rule-blind probing?
  4. **Exact Zero-Drift Utility Swaps (`R27B-D3`):** Under utility swap $\text{do}(U_{\text{seek}} \to U_{\text{avoid}})$, does the LIR-separated architecture (`\chi_W = (1, 0)` vs `\chi_\pi = (0, 1)`) achieve exact `0.000000` belief drift ($\text{JSD} \le 10^{-7}$) while reversing action behavior (`seek >= 0.45`, `avoid >= 0.85`), whereas `Fused_RL` and `Token_Masked` suffer measurable wishful-thinking belief drift?

### Pillar C — `R27C`: Self-Referential Opponent Rationality + Proximity + Prior Inconsistency (`MMD`) & Policy Commitments ($\Pi_t$) on `OpenSpiel`
- **Pre-Registration:** `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R27C-OPPONENT-INCONSISTENCY-MMD.md`
- **Runner:** `/home/agent/workspaces/inconsistency-search/benchmarks/run_r27c_opponent_inconsistency_mmd.py`
- **Core Questions Tested (Extending `opus-mats`'s `/tmp/om/kuhn_qre_incons.py` & `research-orientation-2026-10-19.md` §§3–4):**
  1. **Upstream `OpenSpiel` Curve Reproduction & Exact Self-Referential Fixed Point (`R27C-D0`):** Reproduces upstream `CFRSolver`, `CFRPlusSolver`, `DeepCFRSolver`, and `RNaDSolver` curves on `kuhn_poker`, `leduc_poker`, and `liars_dice(numdice=1,dice_sides=4)`, and verifies that driving the 3-edge PDG inconsistency ($\mathcal{E}_{\text{rat}} + \mathcal{E}_{\text{prior}} + \mathcal{E}_{\text{prox}}$) to $0$ drives `OpenSpiel` `exploitability.nash_conv` to sequence-form QRE.
  2. **Necessity of the Proximity Edge $\beta_{\text{prox}} > 0$ (`R27C-D1`):** Shows that removing the proximity edge (`Simultaneous_SoftBR_NoProx`, $\beta_{\text{prox}} = 0$) causes limit-cycle oscillation with high `nash_conv`, whereas our neural commitment searcher (`Inconsistency_ProxPrior_Neural`) trained strictly via self-referential proximal + prior inconsistency minimization beats all controls and `Simultaneous_SoftBR_NoProx` across all 3 games (`5` seeds).
  3. **Policy-as-Commitment ($\Pi_t$) Under Opponent Readability $\alpha_{\text{read}} \in \{0.0, 0.5, 1.0\}$ (`R27C-D2`):** Tests Kelvin's `20:50Z` thesis: as opponent commitment readability shifts from $\alpha_{\text{read}} = 0$ (unreadable) to $\alpha_{\text{read}} = 1$ (mutually readable commitment), the unique zero-inconsistency fixed point $\Pi_{t+1}^* = \Pi_t^*$ transitions smoothly from pure exploitation to unexploitable minimax in zero-sum Kuhn Poker and cooperative commitment in general-sum trust games.

---

## 2. Deterministic Scoring & Budget Estimate

- **Scorer:** `PYTHONPATH=. python3 scripts/score_r27.py --results-dir results/r27 --out results/r27/score_summary.json`
- **Estimated Compute:** `~1.05 L4-hours` (`~$0.91`, hard cap `2.70 L4-hours` / `$2.35`, `< 1%` of the `$300` budget).
