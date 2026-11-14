# Round 1: Red-Team Critique & Theoretical Audit of the `GLC-AOM` Transition

| Field | Value |
| :--- | :--- |
| **File** | `society/round1/06-opus-fable-critique.md` |
| **Repository** | `futarchy-fi/inconsistency-search` |
| **Round** | Society of Thinkers, Round 1 |
| **Role** | Chief Red-Team Critic & Theoretical Auditor (slot `06-opus-fable`) |
| **Author** | Claude (Anthropic), operating in the assigned Opus/Fable critic slot |
| **Inputs audited** | `SOCIETY-CHARTER.md`; `research-direction-note-v4-2026-10-17.md` (Rounds 1–9 evidence) |
| **Date** | 2026-10-18 |
| **Status** | Critique and pre-registration proposal. No new experiments were run for this document. Every quantitative claim below is either quoted from the two inputs or derived analytically in Appendix A. |

---

## 0. Verdict

The `GLC-AOM` charter is the right next step. Rounds 4–9 established a real, repeatable phenomenon: anchor-aware, deduplicating aggregation of disagreeing rollouts reduces belief KL monotonically, and parameter separation keeps beliefs out of reach of reward. The charter's direction (learn *where* to compute, not hand-code it) is correct.

However, the charter in its current form has three structural problems:

1. **It does not distinguish two different targets.** One is the internal fixed point $\mu^*(\theta, U)$. The other is the external truth $P^*$. Most of the scaffolding in Rounds 4–9 existed to keep search from walking away from $P^*$ toward a flawed $\mu^*$.
2. **It removes scaffolds that were silently supplying four things:**
   - breadth,
   - endpoint verification,
   - safety by abstention, and
   - train/test distribution matching.

   It does not say what replaces them.
3. **Its evaluation design cannot detect the failure modes it most needs to detect.** A 1D IRT index fitted across deduction, inference, strategy and alignment probes will reward the wrong things and hide trade-offs.

All three problems are fixable without adding domain-specific constants. The fixes are:

- a convex learned gate with a computable *gain certificate*,
- an order- and multiplicity-invariant explored-mass aggregator built on a transposition table,
- a structural split of $Z_{\text{aux}}$ into world and plan channels, with the utility term implemented as a *variational-family restriction* rather than an energy term,
- a contraction test for memory tokens, and
- a bifactor IRT / Bradley–Terry measurement model with a separate, non-aggregated integrity panel.

The projected scores in §6 assume these are adopted and pre-registered before any `GLC-AOM` training run.

---

## 1. Ground Rules for This Audit

### 1.1 Evidence grades

Prior-round evidence is cited with the following grades, because the Rounds 4–9 record mixes them and the distinction matters for what can be inherited.

| Grade | Meaning | May be inherited by `GLC-AOM` as… |
| :---: | :--- | :--- |
| **[P]** | Pre-registered rule PASS | Prior for an effect size |
| **[F]** | Pre-registered rule FAIL | Warning / failure-mode evidence |
| **[PH]** | Post-hoc / unaudited rerun | Hypothesis only |
| **[BC]** | Holds by architectural construction | Design property, **not** empirical evidence |
| **[D]** | Descriptive, no rule | Hypothesis only |

### 1.2 Bitter-Lesson purity, made checkable

"Zero domain-specific magic numbers" needs an operational definition, or it will be satisfied by relabeling. I propose that every constant in the system enter a **Constant Ledger** with five fields.

1. **Scope.** Each constant is classed as one of:
   - `domain` (game-specific),
   - `architecture` (shared across all games), or
   - `evaluation` (a test bar, not part of the agent).
2. **Origin.** Whether the value is `learned`, `generic default`, or `tuned`.
3. **Transfer.** Whether the *same value* is used across all 8+ environments.
4. **Units.** Whether the constant is dimensionless or scale-free. For example, an entropy threshold in nats is **not** scale-free across alphabet sizes ($\log 2$ for a binary cell vs. $\log 9$ for a Sudoku cell).
5. **Sensitivity.** The pre-registered sweep $\times\{\tfrac14, 1, 4\}$ for every non-learned architecture constant.

**Purity rule.** An agent is Bitter-Lesson-pure iff:

- it has zero `domain` constants,
- every `architecture` constant is shared across all games and either learned or a generic default with a reported sensitivity sweep, and
- no constant was tuned on test data.

Evaluation constants (CI levels, non-inferiority margins) are permitted but must be frozen in the pre-registration.

Under this rule, several items the charter still carries need a ledger entry:

- $T=3$ loops in `looped_attn_T3`
- $D_{\text{amp}}$
- $\gamma$, $\lambda$, $\eta_{\max}$, $c_{\text{puct}}$
- the crowding exponent
- the `0.02` action-entropy bonus
- the $q > 0.05$ candidate floor
- the per-seed $\tau_{\text{mql}}(s)$

### 1.3 The central theoretical gap: two targets

The research note proves (under (i)–(vi)) that search converges to $\mu^*(\theta, U)$. The charter's D5 claims monotonicity in $\mathrm{KL}(P^* \Vert \mu_b)$. **These are different statements.**

| Target | Definition | What search guarantees | What can go wrong |
| :--- | :--- | :--- | :--- |
| **Internal** $\mu^*(\theta,U)$ | Minimizer of $\mathcal{E}(\mu;\theta,U)$ | Convergence as coverage $\to 1$ (under (i)–(vi)) | Nothing, if the conditions hold |
| **External** $P^*$ | True posterior given the information set | **Nothing.** $\mu^*$ inherits every error in $\theta_{\text{real}}$'s implications | Search can move *away* from $P^*$ while moving toward $\mu^*$ |

The record already contains this failure. Neighbour-conditioned cone rollouts degraded Stratego KL from `0.047 -> 0.539` (`R5-E2`, exploratory). Every non-anchored pool degraded the strong ShipHunt root in Mode B (`R9-E1`, `mix` `+0.1761`).

Computation is *logical* evidence about the model's own implications, not *empirical* evidence about the world. The classical guarantee that "more information never hurts the expected log score of the Bayes posterior" applies to the latter only. In adversarial games, the analogous phenomenon is the minimax lookahead pathology (Nau, 1983).

**Consequence for the charter.** D5 must be split into two separately reported claims:

- **(D5a) Convergence:** $\mathrm{KL}(\mu^*_\infty \Vert \mu_b)$ non-increasing.
- **(D5b) Usefulness:** expected realized log-loss non-increasing.

D5b is the claim users care about. D5a is the one the theory supports. The bridge between them is exactly $\theta_{\text{trust}}$ plus the anchor, and that bridge is what the scaffolds were protecting.

**Operational simplification for D5b.** Wherever a realized outcome $x \sim P^*(\cdot \mid \text{info})$ exists, the identity

$$\mathbb{E}_{x \sim P^*}\big[-\log \mu_b(x)\big] \;=\; H(P^*) \;+\; \mathrm{KL}(P^* \Vert \mu_b)$$

means that monotonicity of expected realized log-loss is *equivalent* to KL monotonicity, because $H(P^*)$ does not depend on $b$. Realized outcomes are available in all eight environments:

- hidden pieces revealed at game end,
- ship positions,
- unique Sudoku solutions,
- market fundamentals,
- game results.

Exact $P^*$ is not required. **D5b should therefore be pre-registered on realized log-loss everywhere, with exact-posterior KL reported only where exact $P^*$ is cheap** (ShipHunt, Sudoku, Galanis-Market).

---

## 2. Audit of the Scaffold Removal (Rounds 4–9 → `GLC-AOM`)

### 2.1 Scaffold inventory

| # | Scaffold (round) | What it encoded | Evidence it helped | Evidence it hurt / was fragile | `GLC-AOM` replacement | Residual risk if removed naively |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Level-synchronous beam, $L=4$ waves (`R7-E2`, `R9-E2`) | Exogenous **breadth**: every stall level gets budget | True path kept in beam on `81–83.3%` of boards; `P8_hybrid` `+5.00 pp [1.5, 8.5]` vs. `P3` **[P]** | `hybrid − stoch` CI `[-3.9, +8.6]` crosses 0 **[F]** | `SELECT_NODE(j)` over all nodes | Learned selector collapses to depth-first tip continuation |
| 2 | Stall threshold $H \ge 0.25$ nats (`R7-E2`) | **Where to branch** | `4.89` stalls/board; true digit in candidate set on `96.6–98.0%` of stalls **[P]** | Not scale-free: nats depend on alphabet size, so it cannot transfer across games | Learned branching via `SELECT_NODE` | Branching rate must be learned per game from sparse signal |
| 3 | Candidate floor $q > 0.05$ (`R7-E2`, `R9-E2`) | **Pruning of tails** | ~97% recall of true digit **[P]** | Domain-scaled constant | Sampling from $\pi_{\text{step}}$ | Budget wasted on tails; or a re-introduced threshold |
| 4 | Greedy lead chain $r=0$ (`R7-E2`, `R9-E2`) | **Guaranteed greedy completion** | Post-hoc `+6.5 pp [1.6, 11.4]` vs. `P3` **[PH]** | Registered factorial `+2.0 pp [-4.5, +8.5]` **[F/NS]** | First chain = tip continuation | Selector may re-invent this as a degenerate dominant strategy |
| 5 | Chain-time step floor $m_{\text{qs}} \ge 0.50$ (`R9-E2`) | **Path-confidence verification** | None | **`AUROC = 0.35` (anti-predictive)**. Froze `m=16 -> 32` under `sc` (`0.6964 -> 0.6950`) while ungated `chk_mix` improved `0.713 -> 0.672` as witnesses rose `45% -> 49%` **[F/Audit]** | None should exist | A learned selector or trust head that uses path likelihood as a feature re-creates it ("path-likelihood shortcut") |
| 6 | Leaf self-read $m_{\text{ql}} \ge \tau_{\text{mql}}(s)$ (`R9-E2`) | **Endpoint verification** | `AUROC = 0.99998`; rejects `322/323` invalid leaves **[P]** | Per-seed threshold; validated only on a deduction task | $\theta_{\text{trust}}$ + gated residual | Does not generalize to games without crisp validity |
| 7 | Fixed 4-ply windows (`R6-E3`–`R8-E3`) | **Bounded depth** (light cones) | `48.2%` KL cut vs. the $t-4$ anchor **[P]** | Versus current-turn anchor: no signal. **Depth-1 MSE `0.0397` < depth-4 `0.0451`** **[F]** | Learned chain length | Unbounded chains compound world-model error |
| 8 | Half Mode A / half Mode B training mix (`R9-E1`) | **Train/test distribution matching** for Mode B safety | Mode B non-degradation under reading 1 **[P]** | Safety was *bought* by the mixture; not tested off-mixture | On-policy distribution of learned search | Distribution shift between selector-generated training trees and test trees |
| 9 | Strong-root selection $M_0^{\text{strong}} = 32$ vs. $34$ (`R9-E1`) | **Evaluation-time root strength** | `D1.2` PASS on 15/15 parts at $32$ **[P]** | Literal KL direction picks $34$: `D1.2` fails 2/15 parts (`+0.00062`, `+0.00066`, `+0.00105`; Mode B gain CI `[-0.0004, +0.0010]`) **[F]** | Root anchor $C_0$ of any strength | Monotonicity claims hinge on an undisclosed root-strength covariate |
| 10 | Epoch-0 early stopping (`R8-E3`) | **Capacity control / "do nothing"** | `0.0488 -> 0.0463`, matching 1-scalar `scaled_mix` **[PH]** | Without held-out stopping: memorization (`0.6348` / `0.5849` test KL) **[F]**. JSD `5.1x` vs. `8x` bar | End-to-end learned aggregator | When the anchor saturates, the only learnable solution is identity, and RL noise can push it away |
| 11 | 4-wave key-space PUCT (`R7-E1`) | **Coverage** | Coverage `83.01% -> 98.63%`; open-loop JSD cut `2.93x` / `5.38x` **[P]** | — | Learned selection | Full support (assumption (v)) no longer guaranteed |
| 12 | Anchor token $C_0$ (all) | **Never degrade the root** | Every non-anchored pool degraded the strong root **[P]** | — | Retained | — |

### 2.2 Five lessons the scaffolds were silently teaching

**L1. Verify endpoints, not paths.**
The single cleanest result in `R9-E2` is the contrast between $m_{\text{ql}}$ (`AUROC 0.99998`) and $m_{\text{qs}}$ (`AUROC 0.35`). An AUROC below 0.5 means valid leaves had *lower* chain-time step confidence than invalid ones. That is exactly what one expects if valid non-greedy completions must pass through low-confidence branching steps at stalls. A path-confidence floor therefore excludes, by design, the branches that search exists to find.

*Requirement:* no `GLC-AOM` component may gate or rank on path likelihood except through the aggregator's mixture weights (assumption (iv)).

**L2. Much of the measured safety was bought by abstention or by distribution matching, not by learned judgment.**
Three results show this:
- `R9-E2`'s unresolved-board $\Delta\mathrm{KL} = 0.0000$ holds because the gate passes zero leaves **[BC]**.
- `R9-E1`'s Mode B safety depended on a half-A/half-B training mixture.
- `R8-E3`'s non-degradation came from staying at initialization.

*Requirement:* every safety metric must be paired with a non-vacuity metric (§3, FM-1).

**L3. Breadth was enforced exogenously.**
$L=4$ waves and 4-wave PUCT are what took coverage from `83%` to `98.6%`. Assumption (v) (full support) was an engineering artifact of these schedules.

*Requirement:* a learned `SELECT_NODE` must be shown to satisfy coverage, not assumed to.

**L4. Depth is not signal when the anchor is strong.**
In `R8-E3`, the current-turn anchor (`0.0488`) beat 4-ply rollouts (`0.080`), and depth-1 claims beat depth-4. In `R5-E2`, neighbour-conditioned rollouts went `0.047 -> 0.539`.

*Requirement:* the marginal value of extending a chain must be estimated, not presumed. "Continue the tip" must not be a free default.

**L5. The effects being chased are small relative to seed variance.**
The registered `R9-E2` factorial contrasts were `+2.0 pp [-4.5, +8.5]` and `+0.33 pp [+0.05, +0.62]`, with 3 seeds. A learned selector introduces RL variance on top of this.

*Requirement:* learned-vs-scaffold comparisons need roughly ≥5 seeds, cluster bootstrap over (seed, instance), and compute-matched accounting (§5.5). Otherwise the transition cannot be evaluated at all.

### 2.3 What changes structurally under `SELECT_NODE(j)` + `STEP(v,z)` + $p_r$

1. **The control problem changes from fixed-schedule to sequential decision.**
   At step $b$ the selector chooses among $b+1$ nodes, and `STEP` among $O(\lvert V \rvert \cdot \lvert \mathcal{Z} \rvert)$ assignments. The reward (realized log-loss reduction, witness, win) is terminal and sparse across $B=128$ steps. The scaffolds replaced this credit-assignment problem with a fixed answer.
2. **Visitation becomes outcome-correlated.**
   A selector trained on outcomes visits branches in proportion to their learned usefulness, not their model probability. Any multiplicity- or order-sensitivity in the aggregator then turns selection into belief bias. Fixed schedules never risked this because visitation was outcome-blind.
3. **Plans add a second learned controller with its own reward**, $\Delta\mathcal{L}_{\text{consensus}}$. That reward is computed by the aggregator it steers. This is a reward-hacking loop unless the aggregator is grounded by $\theta_{\text{trust}}$ on realized data.
4. **Training distribution becomes endogenous.**
   Trees seen in training are generated by the current selector, so lesson L2 (distribution matching) must be re-earned on-policy.

---

## 3. Top-5 Failure Modes That Could Falsify the Round-1 `GLC-AOM` Hypotheses

Each failure mode is given as **Mechanism → Prior evidence → Falsifying signature → Counter-measure (math/algorithm) → Pre-registered rule → Purity check**.

### FM-1. Vacuous or Mis-Targeted Anytime Monotonicity (D5)

**Mechanism.** There are two ways D5 can be satisfied without being earned.

- **(a) Vacuity.** A learned gate or $\theta_{\text{trust}}$ that passes almost nothing satisfies $\mathbb{E}[\mathrm{KL}(P^* \Vert \mu_{b+1})] \le \mathbb{E}[\mathrm{KL}(P^* \Vert \mu_b)]$ trivially. Under RL with noisy rewards this "do-nothing equilibrium" is an attractor: it has zero variance and zero downside.
- **(b) Mis-targeting.** A gate that passes everything converges to $\mu^*(\theta)$. That fixed point can be farther from $P^*$ than the anchor (§1.3).

**Prior evidence.**
- `R9-E2` `0.0000` unresolved KL change **[BC]**.
- `R8-E3` epoch-0 optimum **[PH]**.
- $m_{\text{qs}}$ freezing `m=16 -> 32` **[F]**.
- `R5-E2` `0.047 -> 0.539`.
- `R9-E1` non-anchored pools degrading the strong root **[P]**.

**Falsifying signature.**
- Realized log-loss $L(b)$ flat beyond small $b$.
- Gate pass-rate $\to 0$.
- Gain concentrated in greedy-path nodes.
- Or the opposite: $L(b)$ decreasing on in-distribution roots but increasing on strong or out-of-distribution roots.

**Counter-measure 1: convex geometric gate with a gain certificate** (Appendix A.1).
Each candidate update $\tilde\mu_{b+1}$ is merged as

$$\mu_{b+1} \;\propto\; \mu_b^{\,1-\alpha_b}\,\tilde\mu_{b+1}^{\,\alpha_b}, \qquad \alpha_b = \sigma\big(g_{\theta_{\text{trust}}}(h_{\text{search}}, x_S)\big) \in [0,1].$$

Write $r = \log(\tilde\mu_{b+1}/\mu_b)$. The realized loss $\ell(\alpha;x) = -\log\mu_b(x) - \alpha\, r(x) + \log \mathbb{E}_{\mu_b}[e^{\alpha r}]$ is **convex in $\alpha$**, since the last term is a cumulant generating function. Its expected slope at $\alpha=0$ is $-\Gamma_b$, where

$$\Gamma_b \;=\; \mathbb{E}_{P^*}[r] - \mathbb{E}_{\mu_b}[r] \;=\; \mathrm{KL}(P^* \Vert \mu_b) - \mathrm{KL}(P^* \Vert \tilde\mu_{b+1}) + \mathrm{KL}(\mu_b \Vert \tilde\mu_{b+1}).$$

Three consequences follow:

- Because $\alpha = 0$ is feasible, a gate trained to minimize held-out realized log-loss is monotone in expectation up to generalization error.
- An update is worth taking at small $\alpha$ iff $\Gamma_b > 0$. This is the same quantity as the note's LOO credit $r_{\text{LOO}} \approx \tfrac{\beta_r}{\beta_0}\big(\ell_r(x) - \mathbb{E}_{\mu}\ell_r\big)$.
- $\Gamma_b$ is estimable from realized outcomes. It replaces every hand-set acceptance threshold ($\tau_{\text{mql}}(s)$, $m_{\text{qs}} \ge 0.50$).

**Counter-measure 2: non-vacuity metrics** (all reported, none optional).

- **Gate AUROC** of $\alpha_b$ for $\mathrm{sign}(\hat\Gamma_b)$, with $\hat\Gamma_b$ from realized outcomes. This is precisely the quantity on which $m_{\text{qs}}$ scored `0.35`.
- **Gain Capture Ratio:**
  $$\mathrm{GCR}(B) = \frac{L(0) - L(B)}{L(0) - L_{\text{orc}}(B)},$$
  where $L_{\text{orc}}$ uses the per-step oracle gate $\alpha^*_b = \arg\min_\alpha \mathbb{E}_{P^*}\ell(\alpha)$ (computable where $P^*$ is exact) or the validity-oracle gate (deduction tasks).
- **Non-greedy gain share:** the fraction of $L(0) - L(B)$ contributed by nodes whose path contains at least one non-argmax step.

**Counter-measure 3: root-strength covariate instead of root selection.**
Replace the $M_0^{\text{strong}}$ cut point with a continuous sweep over revealed fraction or anchor KL. Report the monotonicity-violation rate and gain as *curves* in $\mathrm{KL}(0)$. This removes the `32 vs 34` sensitivity by construction: there is no cut point left to be sensitive to.

**Pre-registered rule (FM-1).** For each environment and each consecutive budget pair in $\{0,1,2,4,\dots,128\}$, three conditions must hold.

1. **Monotonicity.** The paired $\Delta L_b = L(b_{k+1}) - L(b_k)$ has 95% upper CI $\le \varepsilon_{\text{MC}}$, where $\varepsilon_{\text{MC}}$ is the *self-replicate noise floor*: the same agent, different RNG streams, data-derived.
2. **Non-vacuity.** Lower CI of $\mathrm{GCR}(128) > 0$, *and* lower CI of gate AUROC $> 0.5$.
3. **Coverage of the root-strength range.** Both conditions above hold in every root-strength quintile, including the strongest.

**Purity.** No thresholds. $\alpha$ is learned. $\varepsilon_{\text{MC}}$ is measured, not chosen.

### FM-2. Unscaffolded Control Collapse & Credit Starvation (D1, D2)

**Mechanism.** With sparse terminal reward, the selector converges to one of three degenerate policies:

- always continue the tip (depth-first greedy), which re-invents scaffold #4 without its breadth complement;
- uniform restarts (no exploitation); or
- whatever the initialization favours.

Plans $p_r$ can fail in two further ways:

- **mode-collapse:** low effective rank;
- **decoration:** $\pi_{\text{step}}$ learns to ignore them.

HER relabeling $p = k^{\text{post}}_{j\to j'}$ is **biased under stochastic transitions** (hindsight bias; Lanka & Wu 2018; cf. USHER, Schramm et al. 2023). `STEP` samples $z \sim q_\theta$, so it is stochastic, and HER will over-estimate plan achievability.

In $D_{\text{amp}}$ dimensions, random plan directions have cosine $\approx D_{\text{amp}}^{-1/2}$. Hindsight-achievement cosines must therefore be compared to a shuffled baseline, not to zero.

**Prior evidence.**
- L3: breadth was exogenous.
- L4: depth-1 beat depth-4 in `R8-E3`.
- L5: effects $\le$ seed variance.
- **No prior-round evidence exists for plans at all** (grade **[—]**).

**Falsifying signature.**
- Fork rate (the fraction of steps with $j_{b+1} < b$) collapses toward 0 or toward 1.
- Coverage falls below the `98.6%` 4-wave PUCT reference at matched compute.
- Plan-swap ablation has no effect.
- `GLC-AOM` loses to `Hardcoded-Beam P8` at every training-compute level.

**Counter-measure 1: dense, exact-where-possible credit.**
Train the selector's value head to regress the **subtree-ablation gain**

$$G_j \;=\; \log \mu_B(x) - \log \mu_{B \setminus \mathrm{subtree}(j)}(x).$$

This is exact LOO for the $\gamma = 0$ pool (note §4) and an envelope-theorem approximation otherwise. It is the value of computation (Russell & Wefald 1991; Hay et al. 2012) measured in the model's own currency. It turns one terminal reward into $B$ dense targets with no hand-set shaping.

**Counter-measure 2: threshold-free exploration.**
Use Gumbel-top-$k$ sampling without replacement over (node, assignment) pairs, with sequential-halving-style allocation (Gumbel MuZero; Danihelka et al. 2022). This yields a policy-improvement guarantee and replaces the $q>0.05$ floor, $c_{\text{puct}}$ tuning, and wave counts. Its constants are a generic $k$ and the budget, both in the ledger.

**Counter-measure 3: plan validity tests.**

- **Effective rank.** $\mathrm{erank}(\Sigma_p) = \exp\big(H(\lambda(\Sigma_p)/\mathrm{tr}\,\Sigma_p)\big)$ of the plan covariance. Report it per environment and per depth.
- **Causal plan-swap.** Replace $p_r$ at node $C_j$ with a plan sampled at a different node with matched residual entropy. Measure the change in steps-to-gain. Plans are decorative if the CI includes 0.
- **Achievement calibration.** Compare predicted vs. realized $\cos(k^{\text{post}}, p_r)$, relative to a shuffled-plan baseline. Use importance-corrected HER, re-weighting relabeled transitions by $q_\theta(z)/\pi_{\text{step}}(z \mid p)$, to remove hindsight bias.
- **"Fractal" as a testable claim.** Weight sharing across depth is not self-similarity. The testable prediction is *scale-covariance*:
  $$p_r \perp \mathrm{depth}(C_j) \;\big|\; \big(H_{\text{res}}(C_j),\, K_{\text{done}}\big),$$
  where $H_{\text{res}}$ is residual model entropy over unresolved variables. Audit this with a held-out classifier predicting depth from $(p_r, H_{\text{res}})$; its accuracy must not exceed a classifier given $H_{\text{res}}$ alone.

**Counter-measure 4: frame the scaffold comparison as a scaling claim.**
The bitter lesson predicts that learned control wins *at sufficient training compute*, not at every scale. Pre-register a training-compute sweep with at least 4 points, spanning roughly 16× of training compute. Test for (i) non-inferiority at the largest point and (ii) a positive slope of the `GLC-AOM` − `P8` gap in log-compute.

**Pre-registered rule (FM-2).**
- At matched inference compute (§5.5), `GLC-AOM` is non-inferior to `P8_hybrid` on `Sudoku-Extreme` witness rate and ShipHunt realized log-loss. The margin is $\delta = \tfrac12 \times$ (`P8_hybrid` gain over `P3`, measured in this round's own runs).
- *And* the gap's slope in log training compute has lower CI $> 0$.
- Plan claims (D2) are made only if plan-swap degradation has lower CI $> 0$ *and* the scale-covariance test passes.

**Purity.** Credit comes from the model's own LOO. Exploration is Gumbel-based. There are no domain constants.

### FM-3. Selection-Induced Policy Dependence via Multiplicity, Order, and Coverage (D1, D3)

**Mechanism.** The note's independence theorem assumes (i) fixed queried arcs, (v) full support, and (vi) exact deduplication. A learned selector violates all three at finite $B$, through three channels:

- **Coverage:** different policies query different arcs.
- **Multiplicity:** a learned selector re-visits useful regions; soft keys under-deduplicate.
- **Order:** the any-order model's KV cache for set $S$ depends on the order in which $S$ was assigned. A causal Transformer cache is not order-invariant, so the summary token $h_j$ leaks the selection path.

**Prior evidence.**
- Imperfect soft deduplication: constant-ridge `jacobi_lin_T4` clean ratio `1.449 > 1.25` (`R7-E1` **[F]**); `econ_prec_K3` `1.267` (`R9-E1` **[F]**).
- Policy-independence passes so far used *fixed* skewed policies (9× flank oversampling), not adversarially chosen ones.

**Falsifying signature.** Belief JSD between two selectors remains non-zero at matched coverage, or does not shrink as coverage rises.

**Counter-measure 1: transposition table with canonical set keys.**
Any-order conditionals are defined on *sets* $S$. Merge every node whose assignment set $\mathrm{canon}(S) = \mathrm{sort}\{(v,z)\}$ has been seen.

- **Exactness.** This is exact, structural deduplication with zero constants. It makes multiplicity irrelevant for exact repeats.
- **Order-inconsistency as a reported quantity.** Where different orders give different $\log q(x_S)$, record $\omega(S) = \mathrm{Var}_\pi[\log q_\pi(x_S)]$. This is the model's own order-inconsistency, a direct term of the PDG energy. Merge by order-averaging, so order-inconsistency is reported rather than hidden.
- **Free training labels for keys.** Transposition hits are free positive pairs, so soft keys can be trained by a contrastive duplicate-detection loss. No hand-labelled overlap is needed.

**Counter-measure 2: explored-mass estimator** (Appendix A.2).
Over the current frontier $F$ of mutually exclusive partial assignments with model mass $w_f = q_\theta(S_f)$ and covered mass $c = \sum_{f \in F} w_f$, define

$$\mu^{\pi}_B(Y) \;=\; \sum_{f \in F} w_f\, \hat\mu(Y \mid S_f) \;+\; (1-c)\,\nu^{\pi}(Y \mid \overline{F}),$$

where $\nu$ is the anchor's prediction restricted to the unexplored complement (generalizing computation-as-evidence). This estimator depends on $\pi$ **only through the set $F$**: not through visit counts and not through order. It yields the bound

$$\mathrm{JSD}\big(\mu_B^{\pi_1} \Vert \mu_B^{\pi_2}\big) \;\le\; \ln 2 \,\Big[(1-c_1)\,\varepsilon_1 + (1-c_2)\,\varepsilon_2 + \textstyle\sum_f w_f\, \mathrm{TV}_f\Big],$$

where $\varepsilon_i$ is the anchor's complement error and $\mathrm{TV}_f$ is the residual error at frontier nodes. Finite-$B$ policy dependence is thus *explained by coverage*, and anything beyond it is aggregator leakage.

**Counter-measure 3: three-level policy-independence audit.**

| Level | Test | Expected under correct design |
| :---: | :--- | :--- |
| 1 | **Matched-set invariance:** feed the aggregator the same node set in permuted orders and with duplicated nodes | JSD at numerical floor (an exact property) |
| 2 | **Coverage-explained sensitivity:** regress measured $\Delta_{\text{pol}}$ on $(1-c_1)+(1-c_2)$ | Intercept CI includes 0 |
| 3 | **Adversarial selector:** train a red-team `SELECT_NODE` to *maximize* $\Delta_{\text{pol}}$ against the reference at matched coverage | Lower bound on worst-case leakage within the selector class; must stay within the self-replicate floor |

**Pre-registered rule (FM-3).**
- Level 1 max JSD $\le 10\times$ float32 round-off of the aggregator.
- Level 2 intercept 95% CI includes 0.
- Level 3 excess JSD over the self-replicate null has upper CI $\le$ the `R8-E1` `looped_attn_T3` reference `0.00575`. This bar is imported from evidence, not invented.

**Purity.** Transposition, explored mass, and adversarial training are all domain-free.

### FM-4. Utility & Reward Leakage through $Z_{\text{aux}}$, Optimism, and Cross-Round Coupling (D3; Assumption (vi))

Treated in depth in §4. In summary:

- **Optimism.** The energy $\mathcal{E}$ as written *does* move beliefs over uncontrolled variables whenever they are correlated with $X_G$. This is the control-as-inference optimism bias. The $\chi_{\text{world}} = 0$ label does not prevent it unless implemented as a family restriction (Appendix A.3).
- **Unpartitioned capacity.** The $D_{\text{amp}} - \lvert V \rvert$ extra dimensions are unconstrained capacity that any $U$-dependent gradient (plan reward, HER, RL) can write into.
- **Round-to-round inheritance.** Distilling consensus into $\theta_{\text{amp}}$ makes round $n$'s anchor inherit round $n-1$'s search distribution.

**Prior evidence.** `Token-Masked` drift `0.0527` per cell ($14\times$ control) and $\Delta_{\text{desire}} = +0.117$ via shared-trunk spillover (`R5-E1` Part B). This is exactly the mechanism a shared $Z_{\text{aux}}$ would reproduce. `LIR-Masked`'s `0.0000` is **[BC]** and is lost the moment $Z_{\text{aux}}$ receives plan or RL gradient.

**Falsifying signature.**
- Non-zero belief JSD over $V \setminus G$ under a utility swap $U \to U'$ at matched coverage.
- $U$ linearly or nonlinearly decodable from the world channel of $h_{\text{cons}}$.
- Calibration drifting across training rounds while in-round metrics improve.

**Counter-measures.** §4.5–§4.6. **Pre-registered rule:** §7, `H3`.

### FM-5. Memory-Token Lock-In and Compounding Drift (D4)

**Mechanism.** $M_t = h^{(t-1)}_{\text{cons}}$ is a learned filtering state. If search at turn $t$ under-weights fresh evidence relative to $M_t$, errors persist and accumulate. The present-time design deliberately forgoes re-simulation, so there is no automatic correction.

Two further problems:

- $M_t$ carries whatever $h_{\text{cons}}$ carries, including any leakage from FM-4. That leakage now persists across turns.
- In Micro-Stratego, much of the history is recoverable from the present board, and the current-turn anchor is already strong (`0.0488`, `R8-E3`). The memory benefit may be *unmeasurable* on the chosen testbed. That would produce a false null, not a falsification.

**Falsifying signature.**
- Realized log-loss rises with turn index relative to a full-history reference.
- After a perturbation, beliefs do not recover.
- No difference between memory and present-only agents. This is uninterpretable unless stratified as below.

**Counter-measure 1: memory contraction test.**
Let $e_t = \mathrm{KL}(P^*_t \Vert \mu_t)$, where $P^*_t$ is the exact filtering posterior (available in ShipHunt-style and Mastermind-style tasks and in small Stratego via exact enumeration). Model $e_t \le \kappa\, e_{t-1} + \delta_t$.

To estimate $\kappa$:
1. Inject a perturbation into $M_t$: memory from a different trajectory with the same present board.
2. Measure the excess-KL gap over subsequent turns.
3. Take $\hat\kappa = \exp(\text{slope of } \log \text{gap per turn})$.

Stability requires $\kappa < 1$.

**Counter-measure 2: history-informativeness stratification.**
For each test position, compute

$$G_{\text{hist}} = \mathrm{KL}\big(P^*(Z \mid \text{history}) \Vert P^*(Z \mid \text{present board})\big).$$

Memory claims are evaluated only in the top tercile of $G_{\text{hist}}$, where memory *can* matter. Positions with $G_{\text{hist}} \approx 0$ are a negative control: memory and present-only agents must tie there.

**Counter-measure 3: memory as a soft factor.**
Enter $M_t$ into the PDG as a factor with precision $\beta_M$ predicted by $\theta_{\text{trust}}$ from realized data, so search can discount it. Train the $M_t$-conditioned belief by stop-grad distillation toward a full-history any-order belief computed offline. This is a sufficiency loss, and it is generic.

**Pre-registered rule (FM-5).**
- $\hat\kappa$ upper 95% CI $< 1$.
- In the top $G_{\text{hist}}$ tercile, the memory agent beats the present-only agent on realized log-loss (lower CI $> 0$).
- In the bottom tercile, the CI includes 0.

**Purity.** $\beta_M$ is learned. Terciles are an evaluation convention.

### 3.6 Failure-mode summary

| ID | Failure mode | Dimensions | Prior evidence | Primary counter-measure | Purity |
| :-: | :--- | :-: | :--- | :--- | :-: |
| FM-1 | Vacuous / mis-targeted monotonicity | D5 | `R9-E2` [BC] `0.0000`; `R8-E3` epoch-0; $m_{\text{qs}}$ `AUROC 0.35`; `R5-E2` `0.047 -> 0.539` | Convex geometric gate + $\Gamma_b$ certificate + GCR + root-strength curves | ✔ |
| FM-2 | Control collapse & credit starvation | D1, D2 | L3–L5; no plan evidence | Subtree-LOO credit; Gumbel-top-$k$; plan-swap & scale-covariance tests; training-compute scaling claim | ✔ |
| FM-3 | Selection-induced policy dependence | D1, D3 | Soft dedup `1.449`, `1.267` [F] | Transposition table; explored-mass estimator; 3-level audit with adversarial selector | ✔ |
| FM-4 | Utility/reward leakage via $Z_{\text{aux}}$, optimism, cross-round coupling | D3 | `Token-Masked` `0.0527`, $\Delta_{\text{desire}}$ `+0.117` | Family restriction; $Z^W/Z^\Pi$ split; gradient ledger; utility-swap & probe audit; policy cross-fitting | ✔ |
| FM-5 | Memory lock-in & drift | D4 | `R8-E3` anchor saturation | Contraction test $\hat\kappa < 1$; $G_{\text{hist}}$ stratification; memory as soft factor | ✔ |

---

## 4. Policy-Independent Consensus, LIR Separation, and the Bidirectional Lemma Space

### 4.1 What the independence theorem actually licenses

Under (i)–(vi), $\mathcal{E}(\mu;\theta,U)$ is strictly convex. Three facts support this:

- Each observational term equals $-H_\mu(X_i \mid X_S) - \mathbb{E}_\mu \log q_\theta$, and conditional entropy is concave in the joint.
- The utility KL is convex.
- $-\gamma H(\mu)$ is strictly convex for $\gamma > 0$.

So the minimizer is unique, and **the theorem is sound as stated.** Three scoping points are not reflected in the charter.

1. **It is an asymptotic statement about the target, not about $\mu_B$.** Assumption (i) fixes the queried arcs, but under `GLC-AOM` the selector *chooses* the arcs. Finite-$B$ beliefs are necessarily policy-dependent through coverage, and legitimately so: a policy that has looked at more is entitled to believe differently. The audit object must therefore be *excess* policy dependence beyond coverage (FM-3, Level 2), not raw $\Delta_{\text{pol}}$.
2. **$\beta_\theta(i \mid S)$ is produced by $\theta_{\text{trust}}$, which is trained on past search.** Assumption (i) holds within a round and fails across rounds. This is exactly the note's caveat (vi), and it means the "fixed target" moves.
3. **"Policy" now has three components**: `SELECT_NODE`, $\pi_{\text{step}}$, and $q_{\text{plan}}$. All three must be in the audited policy class.

### 4.2 The utility term moves uncontrolled beliefs unless implemented as a family restriction

Take hidden $Z$, action $A$, outcome $Y$, observation $o$, and the log-linear (γ→0) pool implied by $\mathcal{E}$ with a utility factor on $Y \in X_G$. The joint minimizer has the form

$$\mu^*(z,a,y) \;\propto\; q(z \mid o)\,\pi_0(a)\,q(y \mid z,a)\,e^{\lambda U(y)}
\quad\Longrightarrow\quad
\mu^*(z) \;\propto\; q(z \mid o)\,\mathbb{E}_{a\sim\pi_0,\; y \sim q(\cdot\mid z,a)}\big[e^{\lambda U(y)}\big].$$

Differentiating at $\lambda = 0$:

$$\frac{\partial \log \mu^*(z)}{\partial \lambda}\Big|_{\lambda=0} \;=\; \mathbb{E}[U(Y) \mid z] - \mathbb{E}[U(Y)].$$

**Hidden-state beliefs tilt toward states in which good outcomes are likely: first-order wishful thinking.** This is the well-known optimism of control-as-inference (Levine 2018). It is the same direction as the note's Fused-RL identity $\mathbb{E}_{\text{fused}}[\langle \mu - P^*, \nabla U\rangle] = \tfrac{\lambda}{2}\lVert\nabla U\rVert^2$, except that here it arises *at inference time with zero RL*. Labeling the term $\chi_{\text{world}} = 0$ does not stop it. Only restricting the variational family does.

**Fix (structural, constant-free).** Factor $\mu = \mu_W \otimes \mu_A$, with $\mu_W(z, y \mid \mathrm{do}(a), o)$ and $\mu_A(a \mid o)$:

$$\mu_W^* = \arg\min_{\mu_W} \mathcal{E}_W(\mu_W;\theta) \quad (\text{utility-free}), \qquad
\mu_A^*(a \mid o) \;\propto\; \pi_0(a)\,\exp\!\Big(\lambda\,\mathbb{E}_{\mu_W^*(y \mid \mathrm{do}(a),o)} U(y)\Big).$$

Then $\partial \mu^*(z)/\partial\lambda \equiv 0$ for every pre-decision variable. This is decision theory: beliefs first, expected utility second.

**Caveat for multi-agent environments.** In `Proposal-Poker` and `Galanis-Market`, *other* agents' biases $y_{-i}$ are world variables. Inferring their signals from their stakes legitimately depends on their utilities. Only the agent's *own* $U$ (own $y_i$, own mandate) must be excluded from $\mu_W$. The audit in §4.6 must swap own-$U$ only.

### 4.3 Assumption (vi): cross-round coupling as a fixed-point problem

Let $\mathcal{T}$ map round-$n$ parameters to round-$(n+1)$ parameters:

1. Search with $(\theta_{\text{real}}, \theta^{(n)}_{\text{amp}}, \theta^{(n)}_{\text{trust}}, \pi^{(n)})$.
2. Distill the consensus into $\theta^{(n+1)}_{\text{amp}}$.
3. Refit $\theta^{(n+1)}_{\text{trust}}$ on realized outcomes of round-$n$ search.

The intended fixed point is $\theta_{\text{amp}} \approx \mathrm{Amortize}(\mu^*(\theta_{\text{real}}, \cdot))$. Two drifts are possible.

- **Covariate drift.** The amortization error is weighted by the policy's state distribution $d_{\pi^{(n)}}$, so $\theta_{\text{amp}}$ is accurate where search went and inaccurate elsewhere. The next policy's anchor is then accurate exactly where it already looks. This loop is self-reinforcing and policy-dependent even when every per-round estimator is policy-independent.
- **Confidence drift.** Self-distillation of a gated consensus is entropy-reducing at each round. Without grounding, calibration degrades (cf. confirmation bias in self-training).

**Counter-measures.**

1. **Utility-blind, $\theta_{\text{real}}$-anchored targets.** The distillation target for $\theta^{(n+1)}_{\text{amp}}$ is the consensus computed with $\theta_{\text{real}}$ as anchor and the *utility-free* $\mu_W$ (§4.2), never with $\theta^{(n)}_{\text{amp}}$ as anchor. This blocks anchor-on-anchor compounding.
2. **Policy cross-fitting** (cf. double/debiased ML; Chernozhukov et al. 2018). Partition training trajectories by generating policy family: learned selector, uniform-random selector, red-team selector from FM-3. Train $\theta_{\text{amp}}$ on folds and evaluate on held-out families. Report the amortization gap per family.
3. **Off-policy re-weighting** of the distillation loss toward a fixed reference state distribution $d_{\text{ref}}$ (e.g., uniform-random selector), using importance weights $d_{\text{ref}}/d_{\pi}$ estimated by density-ratio classification.
4. **Anchor-ablation convergence test** (every round). At large $B$, compare $\mu_B$ under the full system with $\mu_B$ under a $\theta_{\text{real}}$-only anchor. They target the same $\mu^*_W(\theta_{\text{real}})$, so their divergence at matched coverage measures inherited bias.
5. **Round-over-round calibration ledger.** Report realized log-loss and expected calibration error on a *frozen* held-out set for every round $n$. Calibration must not degrade while in-round metrics improve.

### 4.4 Leakage channels into and through $Z_{\text{aux}}$

| Channel | Path | Example | Blocked by |
| :--- | :--- | :--- | :--- |
| **Gradient** | $U$-dependent loss → parameters producing $h_{\text{cons}}$ | Plan reward $\Delta\mathcal{L}_{\text{consensus}}$ computed on a utility-tilted consensus; RL on actions through a shared trunk | $Z^W/Z^\Pi$ split + gradient ledger (§4.5) |
| **Inference** | $f_U$ inside the energy → $\mu(Z)$ via correlation | §4.2 optimism | Family restriction $\mu = \mu_W \otimes \mu_A$ |
| **Data** | Policy-shaped training distribution → amortization error | Covariate drift (§4.3) | Cross-fitting; re-weighting; anchor ablation |
| **Order / path** | KV cache of $S$ depends on assignment order chosen by the selector | $h_j$ fingerprints the path | Transposition merge + order-averaged recompute; order-variance penalty |
| **Key noise** | High-dimensional keys under-measure overlap → over-counting | §4.7 | Keys in decoded $\lvert V\rvert$-footprint space; contrastive dedup training |
| **Temporal** | Leaked content in $h_{\text{cons}}$ persists through $M_t$ | Own-utility tilt carried across turns | Memory built from $Z^W$ only |

### 4.5 Structural firewall: split the lemma space

Partition the extra dimensions into a **world lemma channel** $Z^W_{\text{aux}}$ and a **plan/control channel** $Z^\Pi_{\text{aux}}$.

| Channel | Trained by | Read by | May **not** be read by |
| :--- | :--- | :--- | :--- |
| $Z^W_{\text{aux}}$ | $\mathcal{L}_{\text{any-order}}$ on realized data + utility-free consensus distillation (§4.3) | Belief readout, aggregator, $\theta_{\text{trust}}$, $M_t$ | — |
| $Z^\Pi_{\text{aux}}$ | Plan head, selector value, RL | `SELECT_NODE`, $\pi_{\text{step}}$, $q_{\text{plan}}$ | Belief readout, aggregator, $M_t$ |

**Information-set sufficiency constraint.** The belief readout over $V \setminus G$ must be a measurable function of the *set*

$$\big\{\big(\mathrm{canon}(S_j),\; q_\theta(\cdot \mid S_j, Z^W)\big)\big\}_{j \in \text{tree}}.$$

That rules out dependence on visit counts, order, plan vectors, or $Z^\Pi$. Under this constraint, a plan can only change *which* arcs are queried, and policy influence is confined to coverage, which FM-3's bound controls.

**Gradient ledger (static check, CI-enforced).** Before every training run, walk the autograd graph and assert that no tensor downstream of $U$, rewards, plan losses, or $Z^\Pi$ has a gradient path into parameters producing $Z^W$ or the belief readout. This turns `LIR-Masked`'s "by construction" from a design intention into a machine-checked invariant. The check is cheap and exact.

### 4.6 Auditing policy and utility independence when $h_{\text{cons}} \in \mathbb{R}^{D_{\text{amp}}}$

$\mathrm{JSD}$ is defined on distributions, not on vectors in $\mathbb{R}^{D_{\text{amp}}}$. Latent distance is not a valid belief-difference measure: gauge freedom lets two very different $h$ decode to identical beliefs, and small latent differences can decode to large belief differences. **Audit the induced belief functional, not the latent.**

1. **Marginal level.** Mean and max over $i \in V \setminus G$ of $\mathrm{JSD}(\mu^{\pi_1}(X_i) \Vert \mu^{\pi_2}(X_i))$.
2. **Query level.** Sample a pre-registered query distribution $\rho$ over relational queries $Q(A,B)$: equalities, parities, conjunctions, top-$k$ configurations. Report
   $$\Delta^{\rho}_{\text{pol}} = \mathbb{E}_{Q \sim \rho}\,\mathrm{JSD}\big(\mathrm{ans}_Q(\pi_1) \Vert \mathrm{ans}_Q(\pi_2)\big).$$
   This approximates divergence of the full belief state without a joint.
3. **Adversarial query level.** On a split disjoint from reporting, optimize the query to maximize JSD. Report the held-out JSD of the found query. This gives a worst-case lower bound.
4. **Joint level, sampled.** Joints come from autoregressive decoding of the any-order model, which is generally *order-inconsistent*. Report joint JSD averaged over decode orders, together with the order-variance itself as a noise floor. A policy effect is only meaningful above that floor.
5. **Null calibration.** Every JSD is reported as *excess over the self-replicate null*:
   $$\Delta^{\text{excess}} = \mathrm{JSD}(\pi_1, \pi_2) - \tfrac12\big[\mathrm{JSD}(\pi_1,\pi_1') + \mathrm{JSD}(\pi_2,\pi_2')\big],$$
   where primes denote independent RNG streams.
6. **Utility-swap invariance.** Run with own-utility $U$ and a counterfactual $U'$ (flipped own bias $y_i$ in `Proposal-Poker`, swapped mandate, `faith_swap`) at matched coverage. Report excess JSD over $V \setminus G$ and $\Delta_{\text{desire}}$.
7. **Representational decodability (diagnostic).** Probe $Z^W$ for own-$U$ and for selector identity with linear and MLP probes of increasing capacity. Report held-out accuracy minus chance, with a CI. Decodable-but-unused information is not yet belief distortion, but it is a latent leak that a later round can exploit. It must be reported, though it is not on its own a FAIL.

**Pre-registered utility-swap rule:** see §7, `H3`.

### 4.7 Why $D_{\text{amp}} \gg \lvert V\rvert$ quietly amplifies over-counting

From the note's weight law with $\lambda \ll \beta$, the limiting total weight of $N \to \infty$ reads with pairwise key overlap $\rho$ is $W_\infty = 1/\rho$. Suppose two reads share a true source direction $u$ but carry independent key noise $\varepsilon \sim \mathcal{N}(0, s^2 I_D)$. Then the measured cosine is $\rho \approx 1/(1 + D s^2)$, so (Appendix A.4)

$$W_\infty \;\approx\; 1 + D\,s^2.$$

**For fixed per-coordinate key noise, over-counting of duplicated evidence grows linearly in key dimension.** Moving keys from $\lvert V\rvert$-dimensional salience footprints (`K3`, Pearson `0.880` with true overlap, **[P]**) into raw $D_{\text{amp}}$-dimensional summary space multiplies over-confidence by roughly $(1 + D_{\text{amp}} s^2)/(1 + \lvert V\rvert s^2)$.

**Counter-measure.** Keys must live in the *decoded footprint space*:

- either $s_r = \lvert\Delta\ell_r\rvert/\lVert\Delta\ell_r\rVert_2$ over $\lvert V\rvert$, extended with the query-answer shifts from §4.6, or
- a learned low-rank projection trained contrastively on transposition-table duplicates (FM-3).

**Probe:** report $W(N)/W(1)$ on synthetic exact-duplicate reads *with the transposition table disabled* in every environment. This is the generic analogue of the τ=0.3 clean-read probe.

---

## 5. Critique of the 8-Game Suite, Elo League, and `Mini-ECI`

### 5.1 Measurement problems in the current design

1. **Unidimensionality is implausible.** Several abilities are plausibly distinct:
   - constraint deduction (`Sudoku`, `Equiv-SAT`, forced `Minesweeper`),
   - probabilistic inference (`ShipHunt`, `Mastermind`, `Galanis-Market` calibration, Stratego piece beliefs), and
   - adversarial strategy (`Hex`, `Connect-4`, Stratego play, `Proposal-Poker`).

   Test-time compute $B$ is likely to have *different elasticities* on each: large for deduction, modest for calibration, possibly non-monotone for strategy (lookahead pathology). A 1D score averages these with weights set by item counts and discriminations. Rankings can then flip with suite composition, which is the canonical lever for benchmark gaming.
2. **Mixed outcome types** (continuous KL/Brier, binary witness, win/loss, Elo, market loss) need explicit per-type link functions. Elo is itself a 1D Bradley–Terry model and cannot simply be appended as an "item".
3. **Persons and items are ill-specified.** If items are (task, $B$) pairs, the nested anytime curves of one agent produce severe local dependence (testlets). Budget belongs on the *person* side.
4. **Small, clustered person samples.** On the order of 7 families × 9 budgets × 3 seeds, all highly dependent. Discrimination parameters will be unstable without hierarchical priors.
5. **Solved games create ceilings.** `Connect-4` is solved (first-player win), and small Hex boards are exhaustively solvable. Elo among imperfect agents is less informative than move optimality against the solver.
6. **Intransitivity.** Strategic games can produce cycles among strategies. 1D Elo then misleads, and its value depends on population composition: adding weak bots inflates ratings.
7. **Exploitability is not measured.** In imperfect-information games (`Micro-Stratego`, `Proposal-Poker`), win rate against a fixed population does not bound exploitability.
8. **Alignment probes contaminate a capability index.** `Proposal-Poker`'s `LIR-Masked` vs. `Token-Masked` vs. `Fused` comparison is an *integrity* experiment. Folding reward or win rate from it into `Mini-ECI` can reward the `Fused` agent's belief distortion when distortion helps payoff. That inverts the project's purpose.
9. **Symmetry leakage between train and test.** Sudoku's validity-preserving transformation group has about $1.2\times10^{12}$ elements (digit relabeling, band/stack and row/column permutations, transpose). Hex and Connect-4 have mirror symmetries. Naive splits leak isomorphic instances.
10. **Compute units do not match.** Rounds 7–9 budgeted `P8` in rollouts $\times (E+1)$, while `GLC-AOM` budgets $B$ single-assignment `STEP`s. These are not the same currency (§5.5).

### 5.2 Proposed measurement model: bifactor IRT + Bradley–Terry with a compute covariate

**Persons and items.**
- **Persons** are $p = (m, B, s)$: architecture/family, budget, seed.
- **Items** are $i$: instance buckets within a task, stratified by difficulty.

Each item loads on a general factor $G$ and one specific factor $k(i) \in \{\text{Ded}, \text{Inf}, \text{Str}\}$.

**Ability with compute elasticity.**

$$\theta^{G}_{p} = \theta^{G}_{m} + \kappa^{G}_{m}\,\log_2(1+B) + u^{G}_{s}, \qquad
\theta^{k}_{p} = \theta^{k}_{m} + \kappa^{k}_{m}\,\log_2(1+B) + u^{k}_{s}.$$

**Binary items** (solve or witness; win against a *fixed* reference opponent):

$$P(Y_{p,i} = 1) = \sigma\!\big(a^G_i\,\theta^G_p + a^{k(i)}_i\,\theta^{k(i)}_p - b_i\big).$$

**Graded items** (realized log-loss): use the graded response model on per-item deciles of log-loss relative to fixed anchors (the `1-pass` agent and exact Bayes where available).

**Pairwise games** (league):

$$P(p \succ p') = \sigma\!\big([\theta^{G}_p + \theta^{\text{Str}}_p] - [\theta^{G}_{p'} + \theta^{\text{Str}}_{p'}] + \eta_{\text{first}}\big).$$

**Identification and estimation.**
- Fix the random agent at 0 and the `1-pass` agent at unit scale.
- Use hierarchical priors on $a, b, \kappa$.
- Estimate by full Bayesian inference and report posterior intervals.

**Headline quantities.** The **compute elasticities** $\kappa^{G}_m$ and $\kappa^{k}_m$ are the scientific headline. They directly answer "where does test-time compute buy capability, and for which architecture?" This is more informative than any single ECI scalar.

### 5.3 Pre-registered multidimensionality diagnostics

| Diagnostic | Purpose | Decision use |
| :--- | :--- | :--- |
| Horn's parallel analysis on polychoric correlations of item residuals | Number of factors | Report retained factors |
| Confirmatory 1-factor vs. bifactor vs. 3-correlated-factor, compared by PSIS-LOO on held-out persons | Model choice | Choose by out-of-sample likelihood, not AIC |
| Explained common variance (ECV) and $\omega_H$ | Strength of the general factor | A single scalar `Mini-ECI` is published **only if** ECV $\ge 0.70$ **and** $\omega_H \ge 0.80$ (conventional bifactor cutoffs, pre-registered as evaluation constants); otherwise publish the factor-level indices |
| Yen's $Q_3$ / $Q_3^* = \max Q_3 - \overline{Q_3}$ | Local dependence (testlets) | Flag item pairs with $Q_3^* > 0.2$; model as testlets |
| Item fit ($S\text{-}X^2$, infit/outfit) | Misfitting items | Pre-register exclusion rules *before* fitting |
| DIF by architecture family (logistic DIF) | Items that favour tree vs. 1-pass for reasons beyond ability | Large DIF is itself evidence of multidimensionality; report it, do not delete it |
| Leave-one-task-out rank stability (min Kendall $\tau$) | Robustness of the ranking | The top-rank claim requires the same top model in every leave-one-task-out fit |
| Elasticity heterogeneity test $H_0: \kappa^{\text{Ded}}_m = \kappa^{\text{Inf}}_m = \kappa^{\text{Str}}_m$ | Does compute shift a general factor or specific ones? | Primary scientific contrast for D6 |

### 5.4 Elo league specifics

1. **Fixed anchor population.** The random agent, `1-pass`, and solver-backed perfect players for `Connect-4` and small Hex form a frozen anchor set. Ratings are reported relative to it.
2. **Absolute metrics on solved games.** Report the move-optimality rate against the solver, plus Elo.
3. **Intransitivity-robust aggregation.** Report the empirical game's Nash average and an mElo-style rank-2 model (Balduzzi et al. 2018) alongside 1D Elo. If the cyclic component explains a significant share of held-out outcomes, 1D Elo is not a valid summary for that game.
4. **Exploitability.** For `Micro-Stratego` and `Proposal-Poker`, train an approximate best response against each frozen candidate (approximate exploitability; Timbers et al. 2022) at a fixed, pre-registered training budget, and report it alongside Elo.
5. **Power.** Pre-specify games per pairing from a power analysis targeting a ±25 Elo half-width (95%) for primary contrasts. Bootstrap CIs over games and seeds.

### 5.5 Compute accounting

Define the common unit as **world-model token-conditional evaluations** (WTE): one STEP on a cached prefix counts 1 WTE, plus the attention cost ∝ prefix length. Report three totals for every agent:

- total WTE,
- total FLOPs including selector, plan head, $T$ loops of the aggregator, check/self-read passes, and $\theta_{\text{trust}}$, and
- peak KV-cache memory.

All learned-vs-scaffold comparisons are made at matched FLOPs. `P8`'s $16\times(E+1)$ budgets are re-expressed in these units before any comparison.

### 5.6 Anti-gaming guardrails (pre-registration)

1. **Hash-commit before any `GLC-AOM` training.** Commit the following and record the hash in the repo:
   - suite, generators, seeds, splits,
   - item-bucket definitions,
   - IRT model spec (factors, links, priors), and
   - the primary contrasts.
2. **Symmetry-canonical splits.** Deduplicate train/test under each game's symmetry group, with a canonical-form hash per instance.
3. **Sealed variants.** Hold out at least one unseen size or variant per family (e.g., `ShipHunt-7x7`, a new `Mastermind` code length, `Hex-6x6`). Evaluate once, at the end.
4. **One configuration across all games** for the generalist, enforced by the Constant Ledger (§1.2). Any per-game constant demotes the agent to "specialist" in reporting.
5. **No B-cherry-picking.** Report the full $B$ grid for every agent, never selected points.
6. **Metric locking.** One pre-registered primary metric per task. Others are secondary.
7. **Separate Integrity Panel, never aggregated into `Mini-ECI`.** It contains:
   - $\Delta_{\text{belief}}$,
   - $\Delta_{\text{desire}}$,
   - $\Delta_{\text{pol}}$ excess (§4.6),
   - utility-swap excess JSD,
   - false-note sycophancy,
   - calibration ECE, and
   - the FM-1 non-vacuity metrics.

   A model with a better `Mini-ECI` and a worse Integrity Panel is reported as a trade-off, not a win.
8. **Multiplicity control.** Use Holm correction across primary contrasts, and a hierarchical model across games and budgets for secondary contrasts.
9. **Generalist vs. specialist** comparisons are at matched parameters *and* matched training FLOPs.

### 5.7 Seeds and statistics

Three seeds were barely adequate for fixed-rule aggregators (L5) and are inadequate for RL-trained selectors. Recommendations:

- ≥5 seeds for primary claims.
- Cluster bootstrap with instances nested within seeds (10,000 resamples).
- Report per-seed results for every monotonicity claim, as the Round 9 audits did.

---

## 6. Round-1 Rubric Scores

**Scale.** 1 = unsound or untestable; 5 = plausible but unprotected against known failure modes; 8 = sound with pre-registered falsifiers and supporting prior evidence; 10 = established.

- **Baseline Charter Score:** the design as written, credited for prior-round evidence that transfers.
- **Risk-Adjusted (pre-fix):** the Baseline discounted for the failure modes in §3–§5 that the charter does not address.
- **Projected:** expected score if the counter-measures in this document are adopted and pre-registered. It assumes adoption, not success.

| Dimension | Baseline Charter | Risk-Adjusted (pre-fix) | Projected (with CMs) | Dominant risk | Decisive counter-measure |
| :--- | :-: | :-: | :-: | :--- | :--- |
| **D1** Universal KV-Cache Tree Machine (`SELECT_NODE` + `STEP`) | 7.5 | 5.0 | 7.5 | FM-2 collapse/credit starvation; FM-3 order and multiplicity leakage; compute-unit mismatch | Subtree-LOO credit; Gumbel-top-$k$; transposition table; explored-mass estimator; WTE/FLOP accounting |
| **D2** Active Fractal Prospectus Plans ($p_r$) | 6.0 | 3.5 | 6.5 | No prior evidence; "fractal" untestable as stated; HER hindsight bias; plan-reward hacking; $D^{-1/2}$ cosine baseline | Plan-swap causal test; scale-covariance test; importance-corrected HER; plans confined to $Z^\Pi$ |
| **D3** Bidirectional Latent Lemma Space ($D_{\text{amp}} \gg \lvert V\rvert$) & $Q(A,B)$ | 7.0 | 4.0 | 7.5 | FM-4 optimism and spillover; key-noise over-counting $W_\infty \approx 1 + D s^2$; JSD undefined on latents | Family restriction $\mu_W \otimes \mu_A$; $Z^W/Z^\Pi$ split + gradient ledger; footprint-space keys; query-functional JSD audit with adversarial selector and query |
| **D4** Present-Time Episodic Memory ($M_t$) | 6.5 | 4.5 | 6.5 | FM-5 lock-in and drift; memory benefit unmeasurable on saturated Stratego anchors | Contraction test $\hat\kappa < 1$; $G_{\text{hist}}$ stratification with negative control; memory as soft factor with learned $\beta_M$ |
| **D5** Anytime Monotonicity & Self-Consistency ($b = 0..128+$) | 8.0 | 5.5 | 8.0 | FM-1 vacuity by abstention; internal vs. external target; path-confidence shortcut ($m_{\text{qs}}$ `AUROC 0.35`); root-strength sensitivity | Convex geometric gate with $\Gamma_b$ certificate; realized-log-loss formulation (D5a/D5b split); GCR + gate AUROC; root-strength curves |
| **D6** Multi-Game Elo League & `Mini-ECI` IRT | 5.5 | 3.0 | 7.0 | Multidimensionality; testlets; intransitivity; integrity metrics inside the capability index; symmetry leakage | Bifactor IRT + BT with compute elasticities; ECV/$\omega_H$ gate on scalar ECI; Nash/mElo + exploitability; separate Integrity Panel; hash-committed pre-registration |
| **Overall (unweighted mean)** | **6.75** | **4.25** | **7.17** | — | — |

**Reading the table.**

- **D5 is the charter's strongest asset**, backed by `36/36` strictly decreasing transitions (`R8-E2`/`R9-E2` **[P]**) and anchor-aware non-degradation (`R9-E1` **[P]**). It is also where unearned success is easiest.
- **D2 and D6 carry the most unaddressed risk.** D2 has no evidence yet. D6 is currently the place where the project could most easily mislead itself.
- **D6's projected gain is the largest** because its fixes are cheap, pre-registerable, and fully under the Society's control.

---

## 7. Pre-Registration Package (Proposed Round-1 Hypotheses)

All rules are evaluated with ≥5 seeds and cluster bootstrap 95% CIs, at matched FLOPs. Evaluation constants are listed explicitly.

| ID | Claim | Primary metric | Comparator | PASS rule | Falsifier |
| :-: | :--- | :--- | :--- | :--- | :--- |
| `H1` | Non-vacuous anytime usefulness (D5b) | Realized log-loss $L(b)$ on all 8 environments | Self (consecutive $b$) | Every consecutive paired $\Delta L_b$ upper CI $\le \varepsilon_{\text{MC}}$ in every root-strength quintile; **and** $\mathrm{GCR}(128)$ lower CI $> 0$; **and** gate AUROC for $\mathrm{sign}(\hat\Gamma_b)$ lower CI $> 0.5$ | Any $\Delta L_b > \varepsilon_{\text{MC}}$ (CI-excluded) or a vacuous gate |
| `H2` | Convergence to the internal target (D5a) | $\mathrm{KL}(\mu_\infty \Vert \mu_b)$ in exact-target environments | Self | Non-increasing in $b$ (same rule as `H1`) | Divergence or oscillation |
| `H3` | Policy & own-utility independence of world beliefs (D3) | Excess JSD (§4.6) on marginals + $\rho$-queries | Self-replicate null | Level-1 at numerical floor; Level-2 intercept CI includes 0; Level-3 adversarial-selector excess upper CI $\le$ `0.00575`; own-utility-swap excess upper CI $\le$ the same bar and $\Delta_{\text{desire}}$ CI includes 0; gradient ledger clean | Any bar exceeded, or a gradient path from $U$/reward to $Z^W$ |
| `H4` | Learned control vs. scaffold (D1) | `Sudoku-Extreme` witness rate; ShipHunt realized log-loss | `P8_hybrid`, `P3` | Non-inferior to `P8_hybrid` at the largest training compute with $\delta = \tfrac12$(`P8_hybrid` − `P3` gain); **and** the gap's slope in log training compute has lower CI $> 0$ | Inferior at the largest scale, or non-positive slope |
| `H5` | Plans are causal and scale-covariant (D2) | Steps-to-gain under plan-swap; depth-decodability | Shuffled-plan control | Plan-swap degradation lower CI $> 0$; depth classifier given $(p_r, H_{\text{res}})$ not above the $H_{\text{res}}$-only classifier (CI); $\mathrm{erank}(\Sigma_p) > 1$ (CI) | Any clause fails; D2 then reported as "weight-shared conditioning only" |
| `H6` | Memory is stable and useful where history matters (D4) | $\hat\kappa$; realized log-loss by $G_{\text{hist}}$ tercile | Present-only agent | $\hat\kappa$ upper CI $< 1$; top tercile lower CI $> 0$; bottom tercile CI includes 0 | Any clause fails |
| `H7` | `Mini-ECI` validity (D6) | ECV, $\omega_H$, LOO model comparison, LOTO rank stability | 1-factor vs. bifactor | Scalar ECI published only if ECV $\ge 0.70$ and $\omega_H \ge 0.80$; else factor indices; elasticity heterogeneity test reported regardless | Publishing a scalar ECI when the gate fails |

**Evaluation constants frozen by this package:**

- 95% CIs and Holm correction across `H1`–`H7`
- ≥5 seeds; 10,000 cluster-bootstrap resamples
- $\delta = \tfrac12$ (data-derived)
- `0.00575` (imported from `R8-E1`)
- ECV 0.70 and $\omega_H$ 0.80
- $Q_3^*$ 0.2
- terciles
- ±25 Elo half-width target

None of these is an agent constant.

---

## 8. Requests to Other Society Members & Open Questions

1. **Architecture authors:**
   - Confirm whether the `R9-E2` `chk` pass used an exact rules oracle. If it did, it is unavailable in learned-world-model games, and `GLC-AOM` must rely only on model-internal verification such as the leaf self-read.
   - Specify the canonical-set hashing and order-averaging used for transposition merges.
2. **Training authors:**
   - Commit to the $Z^W/Z^\Pi$ split and the gradient ledger before the first run.
   - Specify which loss trains the plan head and confirm it is computed on the utility-free consensus $\mu_W$.
3. **Benchmark authors:**
   - Provide exact $P^*$ or filtering-posterior oracles for every environment where they are tractable (ShipHunt, Mastermind, small Stratego, Galanis-Market).
   - Provide symmetry groups for canonical splits and solver oracles for `Connect-4` / small Hex.
4. **Open theoretical question.** Is there a principled bound on the D5b violation in terms of $\theta_{\text{real}}$'s calibration error on the arcs search queries? Such a bound would turn §1.3's "no guarantee" into a quantitative guarantee and would show where $\theta_{\text{trust}}$ is strictly necessary.
5. **Open empirical question.** Does the $m_{\text{ql}}$/$m_{\text{qs}}$ endpoint-vs-path asymmetry replicate outside deduction? A cheap test: a `ShipHunt` analogue using leaf re-read consistency vs. path log-probability as acceptance signals.

---

## Appendix A: Derivations

### A.1 Convexity of the geometric gate and the gain certificate

Let $r(y) = \log \tilde\mu(y) - \log \mu_b(y)$ and $\mu_\alpha \propto \mu_b\, e^{\alpha r}$. Then

$$-\log\mu_\alpha(x) = -\log\mu_b(x) - \alpha r(x) + \Lambda(\alpha), \qquad \Lambda(\alpha) = \log \mathbb{E}_{\mu_b}\big[e^{\alpha r}\big].$$

The derivatives of the log-partition $\Lambda$ are:

- $\Lambda'(\alpha) = \mathbb{E}_{\mu_\alpha}[r]$
- $\Lambda''(\alpha) = \mathrm{Var}_{\mu_\alpha}(r) \ge 0$

So the expected loss $\bar\ell(\alpha) = \mathbb{E}_{x\sim P^*}[-\log\mu_\alpha(x)]$ is convex, with $\bar\ell'(0) = -\big(\mathbb{E}_{P^*}[r] - \mathbb{E}_{\mu_b}[r]\big) = -\Gamma_b$.

Expanding the two expectations:

- $\mathbb{E}_{P^*}[r] = \mathrm{KL}(P^*\Vert\mu_b) - \mathrm{KL}(P^*\Vert\tilde\mu)$
- $\mathbb{E}_{\mu_b}[r] = -\mathrm{KL}(\mu_b\Vert\tilde\mu)$

These give the stated form of $\Gamma_b$. Consequences:

- If $\Gamma_b \le 0$, then $\alpha^* = 0$.
- If $\Gamma_b > 0$, then a second-order bound gives $\bar\ell(0) - \bar\ell(\alpha^*) \ge \Gamma_b^2 / (2\, \sup_\alpha \mathrm{Var}_{\mu_\alpha}(r))$ whenever the unconstrained optimum lies in $[0,1]$.

A contextual gate $\alpha = \sigma(g(\cdot))$ whose class contains $\alpha \equiv 0$ is therefore monotone in expectation on its training distribution. Its out-of-distribution monotonicity is exactly what `H1`'s root-strength quintiles test.

### A.2 Coverage bound for the explored-mass estimator

Assume (iv), mixture by model probability over a partition. For any frontier $F$ with mass $c$, the target satisfies

$$\mu_\infty = \sum_{f\in F} w_f\, \mu_\infty(\cdot\mid S_f) + (1-c)\,\mu_\infty(\cdot\mid\overline F).$$

Subtracting the estimator and applying the triangle inequality:

$$\mathrm{TV}(\mu^\pi_B, \mu_\infty) \le \sum_f w_f\, \mathrm{TV}\big(\hat\mu(\cdot\mid S_f), \mu_\infty(\cdot\mid S_f)\big) + (1-c)\,\mathrm{TV}\big(\nu, \mu_\infty(\cdot\mid\overline F)\big).$$

Apply this to $\pi_1$ and $\pi_2$, then use the triangle inequality and $\mathrm{JSD} \le \ln 2\cdot\mathrm{TV}$ (Lin 1991; nats) to obtain the FM-3 bound. Visit counts and order do not appear: the estimator is invariant to both by construction.

### A.3 Optimism tilt of the joint utility factor

Substitute $\mu^*(z,a,y) \propto q(z\mid o)\pi_0(a) q(y\mid z,a)e^{\lambda U(y)}$ and marginalize:

$$\log\mu^*(z) = \log q(z\mid o) + \log \mathbb{E}\big[e^{\lambda U}\mid z\big] - \log \mathbb{E}\big[e^{\lambda U}\big].$$

Differentiate at $\lambda = 0$ to obtain $\mathbb{E}[U\mid z] - \mathbb{E}[U]$. Under the restricted family, $\mu_W(z) = q(z\mid o)$ independently of $\lambda$, so the derivative is identically 0.

### A.4 Key-noise over-counting ceiling

Let $k_r = (u + \varepsilon_r)/\lVert u + \varepsilon_r\rVert$ with $\lVert u \rVert = 1$ and $\varepsilon_r \sim \mathcal{N}(0, s^2 I_D)$ independent. For large $D$:

- $\lVert u+\varepsilon_r\rVert^2 \to 1 + Ds^2$
- $\langle u+\varepsilon_r, u+\varepsilon_b\rangle \to 1$

so $\rho \to 1/(1+Ds^2)$.

From $W_{\text{eff}} = N\beta/(\lambda + (1+(N-1)\rho)\beta)$ with $\lambda \ll \beta$, $\lim_{N\to\infty} W_{\text{eff}} = 1/\rho = 1 + Ds^2$.

### A.5 Notation collisions to resolve before publication

| Symbol | Uses in the note / charter | Proposed fix |
| :---: | :--- | :--- |
| $\gamma$ | Entropy weight in $\mathcal{E}$; crowding exponent in $c_r$; "$\gamma = 0$ pool" | $\gamma_H$ (entropy), $\gamma_c$ (crowding) |
| $\theta$ | Model parameters; exponent in $(1 - C_M)^\theta$ | $\vartheta_{\text{null}}$ for the exponent |
| $B$ | Test-time budget; $\mathrm{diag}(\beta_r)$ precision matrix | $\mathbf{P} = \mathrm{diag}(\beta_r)$ |
| $\lambda$ | Ridge in the normal equations; utility temperature in $f_U$ | $\lambda_{\text{ridge}}$, $\lambda_U$ |
| $K$ / $k$ | Key matrix; number of reads; rollout count | $\mathbf{K}$ (keys), $N$ (reads) |
| $M$ | Number of variables; landmark count; memory tokens $M_t$ | $\lvert V\rvert$, $N_{\text{lm}}$, $\mathcal{M}_t$ |

---

*End of critique. The author slot requests that §3's five failure modes, §4.5's firewall, and §7's `H1`–`H7` be adopted or explicitly rebutted in Round 2 before any `GLC-AOM` training compute is committed.*
