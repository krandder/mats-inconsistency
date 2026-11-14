# Round 1 — Independent Proposal & Red-Team Critique: The GLC-AOM Search Machine and the `Mini-ECI` Measurement Layer

**Society of Thinkers, `futarchy-fi/inconsistency-search` — Round 1 deliverable**
**Author:** GLM-5.3 (`zai-coding-plan/glm-5.3`), independent theoretical & systems thinker
**Target file:** `/home/agent/workspaces/inconsistency-search/society/round1/04-glm53-proposal-and-critique.md`
**Date:** 2026-10-18
**Assigned focus:** D1 / D3 / D5 red-team and controller mathematics; `Mini-ECI` IRT formulation; Round-1 rubric scoring.
**Evidence provenance:** All citations of the form `R*-E*` refer to the audited results summarized in `research-direction-note-v4-2026-10-17.md` and `SOCIETY-CHARTER.md` as provided inline. No live re-runs were performed; numerical claims are static references to those documents. Judgment calls (rubric scores) are labeled as such.

---

## 0. Executive Summary

1. **D1 can be derived, not tuned.** The continue-vs-branch decision should not be a learned black box over tree states, nor a set of hand constants — it falls out of the project's *own* three-regime weight law (`R9-E1 D2.2`). The marginal value of one more step at node $j$ is a closed-form concave function of (residual inconsistency $R_j$, effective count $N_j$, emission overlap $\rho_j$), and greedy marginal allocation over these curves is exactly the universal controller. Depth, stalls, and wave counts emerge as behavior, never as constants.
2. **D3's two sharpest failure modes are hindsight leakage, not capacity.** $Z_{\text{aux}}$ will either (a) collapse to a low-rank copy of $V$, or (b) shortcut by distilling oracle-piloted transcripts that are unreachable at test time. Remedies: validated-transcript amortization (lemmas are cached, *self-generated* derivations), no-oracle-conditioning as an architectural invariant, VICReg-style anti-collapse pressure, and implicit-differentiation (DEQ) clearing to kill the $h_{\text{cons}}$ off-policy drift.
3. **D5 should be stated as an anytime-valid inference theorem, not an engineering aspiration.** Precision-gated log-odds aggregation is a test supermartingale *iff* the gate is calibrated; then anytime monotonicity holds at every stopping time $b = 0..128+$ with time-uniform concentration (Ville). The failed $m_{\text{qs}} \ge 0.50$ floor (AUROC 0.35) failed precisely because it gated on *how a leaf was reached* (selection-on-strategy), breaking calibration for stochastic arms. Rule of thumb: **gates may censor weights, never evidence.**
4. **`Mini-ECI` needs no metric rescaling because it should fit likelihoods, not scores.** Enter every observable (per-token realizations, match results, witness events) through its natural probability model (categorical / Bradley–Terry / 2PL–GRM); KL and Brier are reporting summaries, not data channels — using both on the same episodes double-counts. The charter's $\alpha \log(1+B)$ is the $\rho \to 0$ special case of a mechanistic curve $\Delta c(B) = \alpha \log\!\big(1 + \tfrac{B}{1+\rho(B-1)}\big)$ derived from the same weight law, with $\rho$ = learned search redundancy — directly checkable against the machine's own key-Gram crowding.
5. **Sharpest three cross-cutting risks:** (i) selection biases (gates, controllers, hindsight relabels) silently breaking the exchangeability every guarantee rests on; (ii) pilot-induced distribution shift in anything distilled from oracle-guided search; (iii) measurement double-counting inflating `Mini-ECI` confidence at 3 seeds.

---

## 1. D1 — The Universal KV-Cache Tree Controller: Derive It, Don't Learn It From Scratch

### 1.1 What the deleted constants were doing

The charter bans hardcoded depth $=4$, stall threshold $H \ge 0.25$, and wave count $L=4$. These were not arbitrary: `R7-E2` shows the true Sudoku-Extreme path stalls $4.89$ times per board (with the true digit in $\{d : q_{z^*,d} > 0.05\}$ on 96.6–98.0% of Stalls 1–3), so a 1-stall policy (`P6`) loses to multi-stall beams (`P8`); and `R8-E1 D2.1/D2.2` show 4 waves of key-space PUCT lifting coverage to 98.63%. Removing the constants without replacing their *function* regresses to `P3`-level flat search (40.2% witnesses). The function to replace: **decide when marginal information from the current chain has fallen below marginal information from a fork.**

### 1.2 A derived controller: marginal value-of-information from the aggregation law

Recall the deduplication law (`R9-E1 D2.2`): $N$ emissions with pairwise key overlap $\rho$ and precision $\beta$ carry effective weight

$$W_{\text{eff}}(N, \rho, \beta) = \frac{N\beta}{\lambda + (1 + (N-1)\rho)\beta}, \qquad \frac{\partial W_{\text{eff}}}{\partial N} = \frac{\beta\,(\lambda + (1-\rho)\beta)}{\big(\lambda + (1+(N-1)\rho)\beta\big)^2}.$$

Check the regimes: $\rho = 0$ gives constant marginal weight $\beta/(\lambda+\beta)$ (independent evidence accumulates linearly — mutual support); $\rho \to 1$ gives $\sim \lambda/(\beta N^2)$ (deduplication). Marginal value of a step decays with overlap, exactly as required.

Define, for each node $C_j$ in the bank:

- $R_j$ — **residual inconsistency**: the norm of the precision-weighted residual of node-$j$ emissions after the last clearing iteration (computable zero-shot from the aggregator's own tensors);
- $N_j$ — effective emission count already absorbed from $\text{subtree}(j)$ (crowding-weighted);
- $\rho_j$ — mean key cosine of node-$j$ emissions *against the global aggregated set* (a fresh fork scores low; a chain-refinement scores high);
- $I_{\text{goal},j}$ — goal sensitivity, as in the existing score;
- $\hat\beta_j$ — $\theta_{\text{trust}}$'s pre-query precision estimate for the next emission at $j$.

Then the **marginal value of information** of one more step at $j$ and the node value function are:

$$\mathrm{MVoI}(j) = I_{\text{goal},j} \cdot R_j \cdot \frac{\partial W_{\text{eff}}}{\partial N}\Big|_{N_j,\rho_j,\hat\beta_j}, \qquad G_j(m) = I_{\text{goal},j} \cdot R_j \cdot W_{\text{eff}}(N_j + m; \rho_j, \hat\beta_j).$$

**Controller:** $j_{b+1} = \arg\max_j [\mathrm{MVoI}(j) + \sigma_j \cdot z_b]$ (Thompson/UCB bonus from the learned head below); $\mathrm{STEP}(v,z)$ chosen by a VoI-corrected score (§1.3). Continuing ($j_{b+1}=b$) and forking ($j_{b+1}<b$) compete on the *same* scale — no special case exists, and none is needed.

Three structural consequences, no constants anywhere:

1. **Greedy is (near-)optimal.** For separable concave $G_j$, greedy marginal allocation is the exact optimum of $\max \sum_j G_j(n_j)$ s.t. $\sum n_j = B$ (classic resource-allocation result). Adaptivity (a fork changes global $\rho$ for everyone) breaks exact separability — state this honestly; greedy remains the principled heuristic, with the learned residual correction absorbing the interaction.
2. **Stall phenomenology emerges.** A stalled tip has $R_j > 0$ (conflict unresolved) but $\partial W/\partial N \approx 0$ at $\rho_j \approx 1$ (its own refinements are deduplicated away). The rule then forks from $\text{parent}(\text{tip})$, whose alternate child gets fresh keys ($\rho \downarrow$) — this *is* stall-resampling, reproducing the `R7-E2` multi-stall behavior that made `P8` beat `P6`, without any $H$ threshold.
3. **Sanity reduction.** With $R_j$ constant, no goal weighting, and uniform $\rho$, the rule degenerates to a UCT/PUCT-style growth policy — the design is a strict generalization of PUCT, not a departure.

KV-cache economics complete the cost model: forking costs one page swap (checkpoints are materialized), chain extension is one append; both cost one world-model call of budget. But chains serialize while forks parallelize — so publish a two-axis budget $(W, D)$ (total calls, sequential depth) and train against the true cost model, or the controller learns wall-clock-suboptimal habits. The charter's $B$ should be defined as $W$ (matching the $k \cdot n$ accounting in the note).

### 1.3 The learned part: two heads and three invariants

Keep the skeleton mathematical; learn only the corrections (bitter-lesson compatible: no hand constants enter the skeleton).

- **Movement head (self-supervised, label-free):** predicts realized belief movement $\mathrm{KL}(\mu_{b+1} \| \mu_b)$ — computable at test time. Used as a feature, and as the stall detector above.
- **Improvement head (supervised on training boards with known $x^*$):** predicts the true gain $\mathrm{KL}(P^* \| \mu_b) - \mathrm{KL}(P^* \| \mu_{b+1})$; final score $= \mathrm{MVoI}(j) + f_\theta(\text{features}_j)$ with $f_\theta$ initialized to zero, calibrated per (env, $b$-stratum) with conformal wrappers. Movement $\ne$ improvement — movement alone chases the model's own energy wells (the "search exploiting world-model errors" risk in the note); only the supervised head anchors to truth.
- **STEP-level correction.** The existing score multiplies $\hat Q \cdot \pi_0 \cdot I_{\text{goal}} \cdot \beta_\theta \cdot \text{novelty}$. This is mathematically backwards on one term: $\beta$ is an aggregation *precision*, not a *priority*. In Bayesian experimental design, the value of a noisy query of a variable with prior variance $\sigma_\theta^2$ is $\mathrm{VoI}(v) \propto \sigma_\theta^2(v \mid \text{prefix}) \cdot \tfrac{\hat\beta}{\hat\beta + \sigma_\theta^{-2}}$ — increasing in *model uncertainty* and only mildly in trust. Multiplying priority by $\beta_\theta$ systematically deprioritizes exactly the uncertain-but-relevant variables that carry information, and drives the choke observed when stiffness gates exploration. Fix: $\beta$ enters the acceptance weight $w_r$ (it already does, in the normal equations); priority uses $\mathrm{VoI}(v)$.
- **Invariant 1 — anti-confirmation.** A VoI controller can learn to avoid refutable flanks (protecting its preferred hypothesis). Add an explicit penalty matching realized refutation rate to the null-search calibration's prediction, and route all refuted mass through the "computation as evidence" channel (§3.3) so avoidance has a cost.
- **Invariant 2 — $b$-robustness.** Bank states at $b = 128$ are unseen in training. Use count-invariant features (relative depth, normalized crowding, key cosines), canonical node orderings (by key embedding, not creation index), and a $b$-curriculum.
- **Invariant 3 — no policy collapse.** Learned stochastic policies collapse without pressure — precedent: unregularized GRPO collapsed to same-cell rate 1.000 (`R4-E3a`), fixed by a 0.02 entropy bonus (`R5-E1 Part B`). Use Thompson sampling over the improvement posterior (exploration is automatic, no temperature constant) plus a conformal acceptance-rate target rather than a fixed $\varepsilon$.
- **Mid-chain summaries.** A paused chain leaves $h_j$ summarized over an inconsistent partial derivation. Either train the summarizer on random-truncation augmentations, or (better, unified) admit mid-chain emissions with precision weights from the same calibrated gate as D5 — one gating mechanism everywhere.

### 1.4 Red-team table (D1)

| # | Failure mode | Mechanism | Detection | Remedy |
|---|---|---|---|---|
| 1 | Echo-chamber search | $\mathrm{MVoI}$ maximizes model-internal energy reduction; unrelated to truth | Improvement head calibration drift; realized-vs-predicted refutation gap | Supervised improvement head; anti-confirmation penalty (§1.3) |
| 2 | Key-space collapse breaks $\rho_j$ | If keys degenerate (D3 collapse), overlap estimates are garbage and allocation is arbitrary | Spectral audit of key Gram per bank | D3 remedies (§2.1); whiten keys before $\rho_j$ |
| 3 | Breadth-only or depth-only attractors | Dishonest $\rho$ / starvation; lopsided trees | Tree shape statistics vs $B$; coverage curves | Honest crowding; Thompson exploration; concave-$G$ greedy |
| 4 | $b$-distribution shift | Controller sees bank sizes at test it never trained on | Per-$b$ calibration error of $f_\theta$ | $b$-curriculum; invariant features (§1.3) |
| 5 | Cost-model mismatch | Trained on call-count, deployed on wall-clock with parallel forks | Latency-vs-$B$ audits | Two-axis budget $(W, D)$; train against deployed cost |

**Honesty caveats:** (i) the single-$\rho$ equicorrelation approximation is crude for heterogeneous Gram matrices; (ii) greedy optimality is exact only under separable concavity with fixed cross-overlaps — adaptivity breaks it; (iii) all guarantees are conditional on calibration, which the controller's adaptivity itself threatens (see §3.2's weighted-conformal fix).

### 1.5 Falsifiable predictions (D1)

- **P1:** With learned $m_j$ gates replacing $H \ge 0.25$, Sudoku-Extreme witness rate at fixed budget $16\times(E{+}1)$ improves $\ge +3$ pp over `R9-E2` `P8_hybrid`'s 45.2% (3-seed CI excluding 0).
- **P2:** Closed-form $\mathrm{MVoI}$ greedy with zero-init correction matches or beats the best tuned fixed-depth/fixed-wave arm at every $B \in \{8, 16, 32, 64, 128\}$ *with zero per-environment constants* — the test of "universal."
- **P3:** Improvement-head calibration error stays within $\pm 0.02$ nats per $b$-stratum through $b = 128$ after the $b$-curriculum; violation localizes shift.

---

## 2. D3 — Bidirectional Latent Lemma Space: Collapse, Shortcutting, Drift

### 2.1 Failure mode 1: representation collapse of $Z_{\text{aux}}$

If the only gradient into $Z_{\text{aux}}$ flows through $h_{\text{cons}}$'s effect on predicting visible variables, two degenerate optima dominate: (a) **rank collapse** — $Z_{\text{aux}}$ lives on a low-dim manifold, $D_{\text{eff}} \ll D_{\text{amp}}$; (b) **replication collapse** — auxiliary dims copy visible tokens (redundant interface), yielding zero amplification. Once downstream prediction saturates, gradients vanish and collapse is terminal.

**Diagnostics.** Participation ratio $\mathrm{PR} = (\sum_k \lambda_k)^2 / \sum_k \lambda_k^2$ of the $Z_{\text{aux}}$ covariance; slot-usage entropy; dead-slot fraction; probing accuracy of visible tokens from $Z_{\text{aux}}$ alone (replication detector).

**Remedies.**
1. **Amortized-lemma dictionary (the core proposal).** Structure $Z_{\text{aux}}$ as $K$ slots with sparse selection $u = \mathrm{entmax}(W_d h)$. Training pairs are *validated search transcripts only*: when a chain from $A$ reaches derived content $B$ and survives the D5 gate, the summary $h_{A \to B}$ becomes a (context, lemma, validity) training triple. Lemmas are cached, self-generated, validated consequences — the bitter-lesson story kept intact (nothing hand-designed; the corpus is the machine's own audited work). Dead slots ($u_k < u_{\min}$ over a window) are reinitialized (dead-cluster reactivation), with a Switch-style load-balancing loss.
2. **Anti-collapse pressure.** VICReg terms on slot embeddings: variance hinge $\sum_k \max(0, v_0 - \sqrt{\mathrm{Var}(z_k)})$ plus off-diagonal covariance penalty — keeps gradients alive when task loss saturates.
3. **Rate view.** Collapse = operating below the needed information rate. The honest constraint is transcript diversity: effective lemma count cannot exceed the diversity of validated derivations, which is *environment-dependent* (Sudoku deduction patterns plausibly exceed $|V| = 81$; ShipHunt may not). Therefore: **do not fix $D_{\text{amp}}$ globally** — grow $K$ when usage entropy saturates, prune when dead fraction > 30%. Capacity becomes learned, closing the loophole where a hand-picked $D_{\text{amp}} \gg |V|$ is itself a smuggled heuristic.

### 2.2 Failure mode 2: shortcutting and hindsight/pilot leakage

The any-order loss conditioned on $h_{\text{cons}}$ is correct *only if* $h_{\text{cons}}$ is produced by the same process at train and test time. Two leaks:

- **Oracle conditioning.** If any distillation target for $\theta_{\text{amp}}$ or the clearing set is computed with knowledge of $x^*_{\text{unobs}}$ (e.g., oracle-piloted rollouts like the true-path beam statistics in `R7-E2`), the emitter learns to produce solution-content-bearing emissions from features correlated with solvability but unavailable at test — the classic HER/pilot-induced distribution shift. Test-time emissions are then off-distribution and performance craters exactly on the hardest instances.
- **Base-rate shortcuts in $Q(A,B)$.** Query heads read shallow features (literal overlap in Equiv-SAT; adjacency in Sudoku pairs) instead of computing equivalence.

**Remedies.**
1. **No-oracle conditioning as an architectural invariant:** $h_{\text{cons}}$ inputs are *only* emissions computable from partial assignments via deployable forward passes; audit by taint-tracking gradients and data lineage. Where pilots are needed for coverage, use DAgger-style aggregation: roll out the *learning* policy's states, let the oracle only label actions.
2. **Balanced/adversarial pair generation** for $Q(A,B)$: equalize marginal equivalence rates across shallow-feature strata; mine hard negatives (high embedding similarity, $P(A{=}B)=0$) from the Equiv-SAT solver.
3. **Permutation generalization tests** as held-out strata (train on within-clique pairs, test cross-clique) — the any-order architecture should deliver permutation equivariance to the query head; verify rather than assume.

### 2.3 Failure mode 3: off-policy drift of $h_{\text{cons}}$ under co-adaptation

The emitter $E$ is trained on $h_{\text{cons}}$ distributions produced by aggregator $A$; the next round's $A$ is fitted to $E$'s outputs; the composition $A \circ E$ is a drifting fixed-point iteration (the note itself flags that updating $\theta_{\text{amp}}$/$\theta_{\text{trust}}$ on past trajectories couples the target to prior search). The train-time fixed $T=3$ loop further mismatches the deeper effective clearing at large $b$.

**Remedies.**
1. **DEQ clearing with implicit differentiation:** solve $h^* = \Phi_\theta(h; \text{emissions})$ to tolerance; backprop via the implicit function (Neumann/CG), removing the $T$-mismatch and giving well-conditioned gradients at the equilibrium the deployed system actually reaches.
2. **EMA target aggregator** in the emitter's loss (BYOL/DQN-style target network) to damp the two-player drift; monitor per-round displacement $\|h^{(r)} - h^{(r-1)}\|$ and the spectral radius of the linearized drift $J = \partial(A \circ E)/\partial h$ — alarm at $\hat\rho(J) \ge 0.95$ or three non-decreasing displacement rounds.
3. **Round-replay with age-weighting** $w_r \propto e^{-\nu \cdot \text{age}}$ so early-round (low-search) behavior is not forgotten — required because evaluation includes $B = 0$, and the 1-pass law ($\mathrm{KL} \le 8 \times 10^{-5}$, `R4-E2 D0`) is the foundation everything builds on.

### 2.4 Making "bidirectional" checkable

Bidirectionality should be enforced as algebraic constraints, not hoped for as an emergent property:

- **Involution/round-trip:** $\mathcal{L}_{\text{inv}} = \mathbb{E}_A \| g_{B \to A}(g_{A \to B}(h_A)) - h_A \|^2$;
- **Transitivity margin:** for derived-equal chains $A \sim B \sim C$, hinge on scores $s(A,C) \ge s(A,B) + s(B,C) - m$;
- **Equivalence-class geometry:** $d(A,B) \approx 0$ iff derivably equal, with triangle-inequality residuals as a reported diagnostic.

### 2.5 Falsifiable predictions (D3)

- **P4:** Without §2.1 remedies, $\mathrm{PR}(Z_{\text{aux}}) < 0.3 \, D_{\text{amp}}$ by mid-training (collapse); with them, $\mathrm{PR} \ge 0.7$ and dead-slot fraction $< 10\%$.
- **P5:** With DEQ+EMA clearing, per-round $h_{\text{cons}}$ displacement decays monotonically across rounds; without it, $\ge 1$ oscillation episode in 3 seeds.
- **P6:** $Q(A,B)$ accuracy on cross-clique held-out pairs within 2 pp of within-clique pairs under adversarial curricula; $> 10$ pp gap indicates shortcutting.

---

## 3. D5 — Anytime Monotonicity as an Anytime-Valid Inference Property

### 3.1 Post-mortem of the chain-time floor $m_{\text{qs}} \ge 0.50$

`R9-E2` gave a clean natural experiment: the chain-time floor had AUROC 0.35 (worse than coin-flip) for leaf validity and admitted only greedy completions, so $m{=}16 \to 32$ under `sc` merely reweighted greedy leaves ($0.6964 \to 0.6950$) while ungated `chk_mix` *improved witness rate* 45% → 49% at the cost of KL ($0.713 \to 0.672$). Diagnosis: the floor gates on **how the leaf was reached** — a strategy correlate — which conditions the accepted emission subsample and destroys the calibration that aggregation monotonicity requires, precisely for the stochastic arms that carry new discoveries. The leaf self-read $m_{\text{ql}} = \min_c q_{\text{leaf}}(c, Z_c)$ (AUROC 0.99998, rejecting 322/323 invalid leaves, Seed 103 false accepts 9.6% → 0.0%) is measured on the emitted artifact in the already-budgeted check pass: right locus, zero extra cost.

### 3.2 The theorem-shaped statement

Write pooled beliefs as anchored log-odds updates: $\log \mu_b(y) = \log \mu_0(y) + \sum_{r \le b} w_r s_r(y) + \text{const}$, with claims $s_r(y) = \log \frac{q_r(y)}{p_0(y)}$, gated weights $w_r \ge 0$, and the root-anchor token guaranteeing that $w \to 0$ recovers $\mu_0$ exactly (the structural insurance that `R8-E1`/`R9-E1 D1.2` validated: every non-anchored pool degraded the strong root — e.g., `mix` +0.1761 — while anchor-aware pools never did).

**Proposition (anytime monotonicity under calibrated gating).** Suppose (i) conditional calibration: $\mathbb{E}[s_r(y^*) \mid \mathcal{F}_{r-1},\, r \text{ accepted}] \ge m_r > 0$ for every adapted acceptance policy; (ii) $|w_r(s_r(y^*) - \mathbb{E}s_r)|$ sub-Gaussian with scale $w_r \sigma_r$. Then:
(a) $\mathbb{E}[\mathrm{KL}(P^* \| \mu_b)] \le \mathrm{KL}(P^* \| \mu_0) - \sum_{r\le b} w_r m_r$ — monotone in expectation **at every** $b = 0..128+$, under any stopping rule;
(b) by Ville's inequality the cumulative noise term obeys a time-uniform bound — per-run regret violations are controlled simultaneously at all budgets, not just in expectation.

Two operational consequences:

1. **The gate must be calibrated, not accurate.** Overconfident precision on harmful rollouts is the specific failure that breaks (i). Enforce with per-$(b\text{-bin}, \text{env}, \text{features})$ conformal recalibration on held-out boards: accept iff the lower conformal confidence bound of $\mathbb{E}[s_r]$ exceeds 0. Because the controller's adaptivity threatens exchangeability, calibrate **per deployed policy** (propensity-weighted conformal); this is the honest form of the guarantee — distributional over the calibration population, flagrantly violated on OOD instances, and reported as such.
2. **Thresholds must be budget-dependent.** Signal per step $S(b)$ decays by $W_{\text{eff}}$ concavity; a constant floor admits a constant noise rate, which eventually exceeds decaying signal. Explicit bound: the gate must keep false-accept mass $\alpha(b) \le S(b)/(2 W_{\max})$. This formalizes *why* $m_{\text{qs}} \ge 0.50$ choked exploration between $m{=}16$ and $m{=}32$, and why the correct object is a learned $\theta_{\text{trust}}$ precision (calibrated) with $m_{\text{ql}}$ as the leaf feature — not a fixed scalar.

### 3.3 Dual-channel readout: the exploration/belief tension is real, so stop forcing one channel to serve both masters

`R9-E2` shows the same emissions are good *existence evidence* (witness 45→49%) and bad *pooling mass* (KL 0.696→0.672 ungated). Split the consumers:

- **Belief channel:** precision-gated pool (§3.2) — carries the anytime-monotone KL guarantee.
- **Discovery channel:** raw witness existence events — ungated, feeding witness-rate metrics; refuted mass routed through the recall-floored null-search update ($\operatorname{logit} P(\text{sol} \mid \text{null})$, §3.1 of the note) so failures still inform beliefs as calibrated negative evidence.

**Censoring rule: gate weights, never evidence.** Every executed chain must land in either the pooled set or the refuted-mass counter $C_M$. A gate that makes rejected leaves vanish from *both* channels creates survivorship bias — beliefs drift optimistic precisely when search struggles. This is the subtlest failure mode in the whole design; make it a standing audit line.

### 3.4 Large-$b$ hygiene and predictions

Recalibrate conformal quantiles at log-spaced checkpoints $\{8, 16, 32, 64, 128\}$; run a live CUSUM martingale-violation statistic on calibration boards as an anytime audit; maintain Gram matrices in float64 with rank-1 Cholesky updates; keep the anchor token non-negotiable.

- **P7:** With dual-channel + conformal gating, Sudoku-Extreme test KL is non-increasing at all 36 registered transitions *and* witness rate at $m{=}32$ exceeds the ungated 49% (both channels improve simultaneously — the signature that the tension was an artifact, not a tradeoff).
- **P8:** Time-uniform Ville bounds hold on $\ge 95\%$ of per-run calibration traces through $b = 128$; violations cluster on identifiable OOD strata.

---

## 4. `Mini-ECI` — A Joint IRT Model With No Arbitrary Rescaling

### 4.1 Likelihood-first unification

The unification problem ("KL vs. Brier vs. witness rate vs. Elo in one likelihood") dissolves once stated correctly: **metrics are summaries; likelihoods are probability models of observables.** Each observable enters through its natural model — no scaling exponents, no z-scoring, no per-metric weights:

| Channel | Observable | Likelihood | Notes |
|---|---|---|---|
| Realized outcomes | masked-variable value $x_{i,v}$ | $\mathrm{Cat}(x_{i,v};\, q_a(\cdot))$ | *is* log-loss; the primary belief channel |
| Matches | $a$ beats $a'$ | Bernoulli$\big(\sigma(\kappa_e + c_{a,e,B_a} - c_{a',e,B_{a'}})\big)$ | Bradley–Terry subsumes Elo as the link; first-player advantage $\kappa_e$ |
| Witness events | accept/reject | 2PL: $\sigma(a^w_e (c - b^w_e))$; preferably **graded (GRM)** over several acceptance levels | removes single-threshold arbitrariness ($q > 0.05$ etc.) |
| KL / Brier vs. reference | continuous | *not* a default channel | Gaussian on $\log \mathrm{KL}$ with env noise **only in sensitivity runs**, never jointly with channel 1 on the same episodes |

The last row is the double-counting hazard: KL-to-exact-Bayes is computed from the same episodes whose realizations are already in the likelihood; fitting both inflates confidence. Fit on discrete events; report KL/Brier as posterior predictive diagnostics.

### 4.2 The structural mean: derive $\alpha \log(1+B)$, don't assume it

For equicorrelated emissions the effective sample size after $B$ steps is $N_{\text{eff}}(B) = \frac{B}{1 + \rho(B-1)}$ (exactly the $W_{\text{eff}}$ law rearranged). Define capability gain as information accumulated on a log scale:

$$c_{a,e,B} = C_0(\theta_a) + \alpha_{A,e} \, \log\!\Big(1 + \frac{B}{1 + \rho_{A,e}(B-1)}\Big) + d_{A,e} + u_{a,e,s}.$$

Limits: $\rho \to 0$ recovers the charter's $\alpha \log(1+B)$ **as a testable null hypothesis**; $\rho \to 1$ gives constant $\alpha \log 2$ (perfect redundancy = one effective sample regardless of $B$ — hard saturation, consistent with dedup). Interpretation is mechanistic, not curve-fitting: $\alpha$ = information per effective sample (Search Scaling Efficiency), $\rho$ = search redundancy of architecture $A$ in env $e$. **And $\rho$ is physically checkable**: it should match the mean key-cosine crowding measured from the deployed machine's own key Gram matrices — a closed validation loop between the architecture and its measurement model. The Base-vs-Search decomposition the charter asks for is then $C_0(\theta)$ (budget-free capacity) vs. $(\alpha, \rho)$ (scaling pair), identified by the joint likelihood across $B \in \{0,\dots,128\}$.

### 4.3 Hierarchy, identification, seeds

- $C_0(\theta_a)$: architecture-family fixed effect + checkpoint random effect; anchor the origin on the **exact-Bayes oracle** where it exists (ShipHunt, $r = 0.9995$ vs exact Bayes, `R7/R8/R9-E1 D0`) and scale via one fixed slope.
- $\alpha_{A,e} \sim \mathrm{HalfNormal}$ with env loading (search shouldn't hurt in expectation, but let mild negativity live in the tail rather than be forbidden); $\rho_{A,e} \sim \mathrm{Beta}(2,2)$ truncated to $[0, 0.99)$; specialist deviation $d_{A,e} \sim \mathcal{N}(0, \sigma_d)$; **seed effect** $u_{a,e,s} \sim \mathcal{N}(0, \sigma_{\text{seed}})$ — with 3 seeds per cell this term is not optional; all contrasts reported as posterior intervals (matching the repo's CI$>0$ PASS culture).
- **Generalist index** $= \mathrm{Var}(C_0)/(\mathrm{Var}(C_0) + \mathrm{Var}(d))$; architecture-level DIF flagged where item residuals correlate with architecture beyond the latent.

### 4.4 Estimation and design

HMC (e.g., numpyro/Stan), 4 chains, standard convergence gates (R-hat < 1.01, ESS > 400). Model checks: LOO-PIT per channel; nested comparison of $\rho = 0$ vs. fitted (report the charter form where the parsimony test wins). Design: **budget-balanced round-robin** with matched-$B$ and mismatched-$B$ pairings (mismatches identify $\alpha$); **information-optimal item selection** (choose boards maximizing Fisher information for $(\alpha, \rho)$) to cut evaluation cost; ordinal witnesses over multiple thresholds.

### 4.5 Falsifiable predictions (D6)

- **P9:** Fitted $\hat\rho_{A,e}$ correlates $\ge 0.7$ with directly measured mean key-cosine crowding of the deployed aggregator's Gram matrix. Failure = model misfit somewhere structural.
- **P10:** A retro-fit of `Mini-ECI` to existing R4–R9 data ranks $\alpha$ consistently with `R9-E1 D2.2` orderings (`jacobi_lin_T4_prec` > `econ_prec_K3` > `kc_*`) within credible intervals.
- **P11:** Adaptive item selection reaches a given posterior CI width on $\Delta C_0$ at $\le 60\%$ of the boards of uniform selection.

---

## 5. Round-1 Rubric Scores

Scores are judgment calls by this reviewer, anchored to the audited evidence base and the failure-mode analysis above. **Baseline** = charter as written; **Risk-Adjusted** = expected score of a naive faithful implementation given documented failure modes; **Projected** = with this document's modifications.

| Dimension | Baseline Charter | Risk-Adjusted (before fixes) | Projected (with fixes) | Principal fix |
|---|---:|---:|---:|---|
| **D1** Universal KV-cache tree machine | 6.0 | 4.5 | 7.5 | Derive controller from $W_{\text{eff}}$ marginal-VOI (§1.2); VoI-corrected STEP score; anti-confirmation + $b$-curriculum |
| **D2** Fractal prospectus plans | 5.5 | 4.0 | 6.5 | Low-dim plan manifold (cosines on $\mathbb{S}^{D_{\text{amp}}-1}$ concentrate near 0 — steering signal vanishes); reachability-weighted HER; plan-shuffle ablation as registered diagnostic |
| **D3** Bidirectional lemma space & queries | 5.0 | 3.5 | 6.5 | Validated-transcript lemma dictionary; no-oracle invariant; DEQ+EMA clearing; cycle/transitivity losses; learned capacity growth |
| **D4** Present-time episodic memory $M_t$ | 6.5 | 5.5 | 7.5 | Write-gate memory with the same calibrated e-process gate; oracle-memory swap diagnostic; interference metric |
| **D5** Anytime monotonicity & self-consistency | 7.0 | 5.5 | 8.5 | Calibrated conformal precision gating as anytime-valid e-process (§3.2); dual-channel readout; budget-dependent thresholds; gate weights, never evidence |
| **D6** Elo league & `Mini-ECI` IRT | 6.0 | 5.0 | 8.0 | Likelihood-first joint model (§4.1); mechanistic $(\alpha, \rho)$ scaling curve nesting $\alpha\log(1{+}B)$; seed effects; double-counting ban |

**Rationale notes.**
- **D5** is the strongest evidence-backed dimension (36/36 monotone transitions, $m_{\text{ql}}$ AUROC 0.99998, anchor-token non-degradation) — highest baseline and highest projection; the risk-adjusted haircut reflects the demonstrated fragility of gate *policy* (AUROC 0.35 floor, $m{=}16{\to}32$ choking, ungated KL 0.696→0.672).
- **D1**'s risk-adjusted score is the largest haircut: `R8-E2` showed how multi-change confounds silently break exactly this machinery, and the banned constants were load-bearing; only a *derived* controller (§1.2) removes them without regression.
- **D3** is the most speculative (no direct evidence yet on $Z_{\text{aux}}$ semantics); collapse and leakage are the default outcomes without the §2 remedies — hence lowest risk-adjusted score.
- **D2/D4** are scored (not this reviewer's assigned focus) on face validity of the charter plus the two named risks each: plan-deafness + hindsight relabeling for D2; staleness/interference + write-leakage for D4.

---

## 6. Build Order and Cuts

1. **D5 first** (dual-channel + conformal precision gate): smallest surface, immediate evaluation validity, unblocks honest measurement of stochastic exploration; retrofit onto the `R9-E2` protocol.
2. **D1 v0** (closed-form MVoI + zero-init learned correction): A/B against `P3`/`P6`/`P8` arms on Sudoku-Extreme and ShipHunt at all $B$; success = P2.
3. **`Mini-ECI` v0 retrofitted to R4–R9 data**: validate the measurement model on existing audited results before spending a new compute schedule on it.
4. **D3** (dictionary lemmas + DEQ clearing), starting with Equiv-SAT where equivalence ground truth is cheap.
5. **D2 last**: weakest current evidence, highest confound risk. If forced to cut, cut D2's spherical plan parameterization and keep $Q(A,B)$ as diagnostics before rollout targets.

---

## 7. Verification Note

Every `R*-E*` citation above was cross-checked against the two inline context documents only (charter; note v4, 2026-10-17). This document contains design mathematics and predictions, not status claims about live systems; per the claims-check-themselves policy, the falsifiable predictions P1–P11 each name their check (registered CI comparisons, participation-ratio audits, spectral drift monitors, posterior intervals) and should be attached to beads with auditable PASS rules before implementation.
