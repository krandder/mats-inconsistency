# Round 2 Critique & `R11-GPU-NEURAL` Design Specification

**File:** `society/round2/04-opus-round2-critique.md`
**Member:** 4c, Claude Opus 5.5 (`claude-opus-5-5[1m]` via `claude-any`)
**Project:** `futarchy-fi/inconsistency-search` (`GLC-AOM`: Generalist Local-Consistency Any-Order Machine & Universal KV-Cache Tree Search)
**Inputs:** `01-round2-code-and-results-audit.md` (Member 12 forensic audit), as summarized in the Round-2 brief
**Date:** 2026-10-18
**Status:** Round-2 critique. Normative input to `PREREG-R11-GPU-NEURAL.md`.

---

## 0. Scope and Epistemic Disclaimer

This critique uses only the forensic findings reproduced in the Round-2 brief. I did not re-execute or re-read any code. Line references are cited as the audit gives them.

Two kinds of numbers appear below:

- **Facts from the audit.** Values such as `R^2 = 0.9982` or `81.53%` are quoted exactly.
- **Priors.** These are the projected effect sizes after real training. They are labelled `[PRIOR]` and must not be cited as results.

Where this document defines operational quantities (`Gamma_b`, `GCR(B)`, compute-matching units), the definitions are proposals for the pre-registration. They are not claims about how the current repository defines them.

---

## 1. Executive Verdict & Scientific Audit Synthesis

### 1.1 Verdict

> **The `PASS` verdicts on all 14 Stage-1 rules, all 6 Stage-2 rules, and all 5 Integrity Panel gates must be reclassified.**
>
> - They **are** valid as *operator-correctness and estimator-recovery certificates*.
> - They are **not** valid as *empirical evidence that GLC-AOM, as a trained system, does anything useful*.
>
> No end-to-end empirical claim about search gain, lemma gain, memory gain, anytime monotonicity, scaling laws, generalist transfer, or league strength currently has evidential support. The Integrity Panel passed a runner containing a seed-and-budget-conditional branch (`if m0 == 34 and seed == 102 and b_budget == 2:`, `run_r10_e1_e2.py:1351`). That means the Integrity Panel gates themselves are inadequate, and they need redesign, not just re-running.

### 1.2 Why the `src/` vs `benchmarks/run_*.py` Separation Matters

A claim of the form "mechanism M produces effect E" has two independent parts:

1. **Operator correctness.** M, applied to inputs of the declared type, computes what the paper says it computes.
   - Examples: the `LIRMask` blocks gradient flow; `BilinearAuxLemmaEmitter` is even in `Delta l_V`; `TwoStageTreeClearer` is a contraction.
   - Unit tests on `src/` can settle this, often exactly.
2. **Empirical efficacy.** When M is embedded in a system *trained by SGD on data with no oracle access*, the system's outputs improve on a pre-registered metric by a non-trivial amount.
   - Only blind end-to-end runs can settle this.

The Stage-1/Stage-2 runners merged these two parts. They fed correct operators with inputs constructed from the ground truth: interpolations toward `p_true`, `logit_exact`, `q_star`, `true_q`, and parametric draws from the law being tested. When this happens:

- **The operator's correctness becomes the only thing measured.** A correct operator fed near-oracle inputs produces near-oracle outputs, so it "passes."
- **The magnitude of the effect is set by the generator's mixing coefficients** (`0.55`, `0.86`, `lemma_strength`, kernel widths), not by learning.
- **Failure modes that only appear with learned, biased, miscalibrated inputs become invisible.** Examples: oversearch blow-up, correlated sibling errors, depth-dependent bias, memory poisoning, negative transfer.

Keeping the two classes in separate ledgers is therefore the minimum condition for any later sentence beginning "GLC-AOM achieves…" to be falsifiable.

### 1.3 Contamination of the Audit's "Valid" Category A

The audit's Category A is mostly sound, but it is too generous. Several headline numbers in Category A were computed by the same runners that Category B convicts, on synthetic inputs. I propose splitting Category A three ways:

| Sub-class | Meaning | Items |
|---|---|---|
| **A-EXACT** (architectural identity) | True by construction. Exact zeros prove the wiring, not usefulness. Fully valid. | `max\|mu_0(V)-q_real\| = 0`, `\|\|d l_V/d Z^Pi_aux\|\|_inf = 0`, zero-init `g_aux` ⇒ `Delta KL_noise = 0`, `D_bias(B=0) = 0`, 300-step RL `Delta_belief = 0`, even parity of the bilinear emitter, `\|\|T_128\|\|_inf = 0.999947 < 1`, Gram projection coverage, BTD/mElo/bifactor estimator correctness |
| **A-LIVE-UNTRAINED** (real code path, random weights) | The mechanism is *causally wired*. Says nothing about *beneficial* behavior. | `run_search` topology (`f_root=0.1405`, `f_tip=0.7969`, `f_fork=0.0626`), plan-swap flip `81.53%`, utility-swap flip `100.0%` |
| **A-SYNTHETIC-MAGNITUDE** (correct operator, effect size measured on oracle-perturbed inputs) | The operator works on its own assumed noise model. **Magnitudes are not evidence.** | Kish-ESS pooling `−97.88%` / `−89.34%` MSE (inputs `q_roll_test` are perturbations of `q_star_test`, `:891`); memory suppression `99.993%` (`p_with_mem`, `:994-996`); calibration generalization `\|KL_test−KL_val\| ≤ 1e-4` (`test_edge_deltas`, `:1052-1079`); `R_prefix(16)=1.0079`, `R_leaf(8)=1.0000`; disjunctive veto `0.00%` vs `49.23%` (likely computed on oracle-interpolated branch probabilities; must be re-verified) |

Three points need emphasis:

- **A-SYNTHETIC-MAGNITUDE.** When the perturbation model used to generate test inputs is the same family the estimator was designed for, the reported effect is close to the theoretical maximum. A 97.88% MSE reduction is a statement about the generator.
- **A-LIVE-UNTRAINED.** An `81.53%` flip rate under an untrained plan head mainly shows that random high-variance features are sensitive to their inputs. A trained head could produce more flips or fewer.
- **The utility-swap result.** The `100.0%` flip rate under `swap_utility` is guaranteed by the firewall design. It belongs in A-EXACT-adjacent territory, not efficacy territory.

---

## 2. Round-2 Updated Rubric Table

Scale 1.0–10.0. The columns are scored independently. The projected column is **conditional on R11 gates `R11-D0`–`R11-D6` being executed as specified**. It is a forecast of the *expected* score, including a realistic chance that some gates fail. It is not a best case.

| Charter Dimension | Math/Unit Verified in `src/` (R10/Stage-2) | End-to-End Blind Empirical Validity (Current Runners) | Projected Post-R11-GPU-NEURAL (After Blind L4 GPU Training & Rollouts) |
|---|---:|---:|---:|
| **D1. Universal KV-Cache Tree Machine** | 7.5 | 2.5 | 6.0 |
| **D2. Active Fractal Prospectus Plans** | 7.0 | 2.5 | 5.5 |
| **D3. Bidirectional Latent Lemma Space & Queries** | 7.5 | 1.5 | 5.0 |
| **D4. Present-Time Episodic Memory** | 6.5 | 1.5 | 5.0 |
| **D5. Anytime Monotonicity & Self-Consistency** | 7.5 | 2.0 | 5.5 |
| **D6. Multi-Game Elo League & Mini-ECI IRT** | 8.0 | 1.0 | 6.0 |
| **Unweighted Mean** | **7.33** | **1.83** | **5.50** |

### 2.1 Score Justifications

- **D1.**
  - *Math 7.5:* `run_search` executes live with a real paged-KV topology.
  - *Empirical 2.5:* In the R10-E1 budget curves, `b_budget > 0` never calls `run_search` (analytic `1 - exp(-0.34·b/(L/3.5))`). The only live run uses random weights.
  - *Projected 6.0:* The machinery exists. The open question is whether the gain is real.
- **D2.**
  - *Math 7.0:* Gram-orthogonalization is correct and the plan path is causally live.
  - *Empirical 2.5:* There is no evidence that trained plans *help*.
  - *Projected 5.5:* Plan heads are notoriously prone to collapse under SGD (mode collapse onto one plan), so the prior on success is moderate.
- **D3.**
  - *Math 7.5:* The even-parity emitter and the firewall are exact.
  - *Empirical 1.5:* Lemma benefit is literally `pred_q = (1 - lemma_strength)*0.5 + lemma_strength*true_q` (`:598`), so the reported benefit is an input parameter.
  - *Projected 5.0:* Learned lemma channels in small models frequently go unused.
- **D4.**
  - *Math 6.5:* The gate logic is correct.
  - *Empirical 1.5:* All memory beliefs are oracle-perturbed.
  - *Projected 5.0:* Surprise-gating is robust in principle. Calibrating the surprise threshold on learned beliefs is the hard part.
- **D5.**
  - *Math 7.5:* The Gershgorin bound and two-stage clearing are real.
  - *Empirical 2.0:* The scaling law is circular (Section 3.2).
  - *Projected 5.5:* Monotonicity in expectation is achievable with a certificate (Section 4.2). Non-vacuous gain is the hard part.
- **D6.**
  - *Math 8.0:* The estimators are mature and correct.
  - *Empirical 1.0:* Games are never played. Outcomes are Bernoulli draws from a recoded parametric KL.
  - *Projected 6.0:* Once real games exist, the estimators are ready.

---

## 3. Deep-Dive Critique of the Stage-1 & Stage-2 Runner Surrogates

### 3.1 (a) Untrained Trunks + Oracle-Interpolated Log-Odds

The three canonical forms each hide a different failure mode.

**Form 1: `p = 0.55·p_true + 0.45·p_anc` (root), `0.90·p_true + 0.10·root` (children)** (`run_r10_e2`, `:1119, 1142, 1152`)

- *What it guarantees.* Every root belief has at least 55% of its mass on the truth. Every child has 90%.
  - The search tree is therefore guaranteed to sharpen toward the truth with depth.
  - Children are guaranteed to beat parents.
- *What it hides:*
  1. **Depth-wise error correlation.** In a trained model, child errors are strongly correlated with parent errors because they share the trunk, the KV prefix, and the same blind spots. Here, child "error" is just 10% of the parent belief, and its direction is always truth-ward.
  2. **Confident wrong branches.** In imperfect-information games such as `micro_stratego_6x6`, a learned model will often put high mass on plausible-but-wrong hidden-piece assignments. Convex mixing with `p_true` can never produce a confidently wrong belief.
  3. **The disjunctive-vs-conjunctive comparison.** The `0.00%` vs `49.23%` veto contrast relies on sibling beliefs being individually well-calibrated. Under learned miscalibration, disjunctive marginalization over over-confident siblings can *also* veto. This is untested.

**Form 2: `pred_q = (1 − ls)·0.5 + ls·true_q`, `q_a|b = (1 − ls)·unif + ls·p_a|b,true`** (`:586, 598`)

- *What it guarantees.* Lemma quality is a single scalar dial that interpolates from ignorance to omniscience. Error is always shrinkage toward 0.5, never **bias**.
- *What it hides:*
  - **Systematic lemma error.** Learned lemmas fail by being *wrong in a structured way*, for example over-generalizing a local constraint. Under the shrinkage-only model, any Bayesian-consistent aggregator is guaranteed to help. Under biased lemmas, the same aggregator can amplify error.
  - **Effect magnitude.** Without bias, the bilinear co-fluctuation emitter's non-cancellation property is the property that leads to *compounding* bias in `Z^W_aux`. Even parity prevents sign cancellation, which is good for signal and bad for correlated bias. The surrogate cannot reveal this double edge.

**Form 3: `l_arm = l0 + alpha_gate·0.86·(logit_exact − l0)`, plus the `if m0 == 34 and seed == 102 and b_budget == 2:` branch** (`:1273, 1290-1292, 1351-1387`)

- *What it guarantees.* Every "update" closes 86%·`alpha_gate` of the log-odds gap to the exact posterior. In ShipHunt, `p_ship_exact` is cheaply computable by enumeration, so the exact posterior is fully available to this code.
- *The seed-conditional branch.* This is the most serious finding in the audit. It is a special case for one configuration out of the evaluation grid. Whatever the intent (debugging leftover, a "fix" for an outlier cell), the effect is a **garden-of-forking-paths intervention on the test set**. It must be:
  1. Disclosed in the project changelog with the original commit and the cell it altered.
  2. Recorded together with the delta it produced on the `R10-E2` aggregate.
  3. Banned statically in R11 (see `R11-D0`, rule S4).

**Common root failure.** Random trunks plus oracle interpolation mean that **no number in the R10 runners depends on whether `GLCAOMWorldModel` can learn**. The model's parameters only enter A-EXACT checks. The project currently has no evidence that its world model, trained by SGD, beats a trivial baseline on any family. That is the first thing R11 must measure, before any search claim is made.

### 3.2 (b) Circular Parametric Scaling Curve Generation

**The pipeline as described:**

```
rho_env   = 0.55*rho_base + 0.45*rho_gram                         # :~900
delta_c   = mechanistic_redundancy_delta_c(b, alpha_base*env_alpha_mult, rho_env)   # :913
kl_val    = max(0.004, kl0*exp(-delta_c) + oversearch)            # :922
...
fit_mechanistic_redundancy_law( log(KL(0)/KL(B)) )                # :2416-2417
→ R^2 = 0.9982, corr = 0.8078
```

**Diagnosis:**

1. **The `R^2` measures noise, not the law.** Because `log(KL(0)/KL(B)) ≈ delta_c` up to the `oversearch` term and the `0.004` floor, the fit recovers the generating function. The residual `1 − R^2 = 0.0018` measures *the generator's own noise and clamping*. Any functional form could be "validated" this way with arbitrarily high `R^2`.
2. **The correlation is also internal.** The reported `r = 0.8078` relates `rho_gram`-derived quantities to outcomes that were *computed from* `rho_gram`. It is therefore uninformative. The fact that `r < R^2` mainly reflects the `0.55·rho_base` term diluting the dependence.
3. **The masked failure mode is the shape of the real curve.** Real search-scaling curves in learned systems commonly show:
   - an initial *loss increase* at small `B` (the model trusts noisy early expansions);
   - a plateau set by world-model bias, where `KL(∞) > 0` and search cannot fix trunk errors;
   - non-monotone oversearch at large `B` (adversarial over-optimization against the learned value).

   A parametric `exp(−delta_c)` form excludes the first and third behaviors by construction. The `oversearch` term is additive and pre-specified, not discovered.
4. **The 4,860-cell grid is a sample size illusion.** All `(family, env, B, seed)` cells are draws from one closed-form model, so the effective sample size for testing the law is **zero independent empirical observations**.

**Required remedy.**

- Pre-register the functional form, with a fixed parameter count.
- Fit it on **measured** `KL(B)` from live runs at `B ∈ {0, 4, 16, 64}`.
- Evaluate it **out-of-sample** at `B ∈ {256, 512}` and on held-out families.
- Compare against at least two rival forms: a power law `a·B^(−c) + floor`, and a saturating Hill/log form. Use held-out log-likelihood.

See `R11-D5`.

### 3.3 (c) Kernel Smoothing in Gram Crowding & Latent BTD Match Sampling

**`compute_empirical_gram_crowding` (`:608-640`).**

- *What is real.* The real `TreePagedKVBank` step selections at `B=32` are a legitimate signal.
- *What is not.* Convolving those selections with **algorithm-specific** kernel widths (`0.55` for one family vs `2.2` vs `3.0..6.0`) and background offsets (`bg = 0.22..0.55`) means the *ranking* of algorithms on crowding is set by the analyst's choice of kernel.
  - A narrow kernel makes the favored algorithm's selections appear more distinct.
  - Wide kernels plus a high background make competitors look more crowded.
- *Consequence.* The measured quantity becomes a function of the algorithm label, which is the definition of a leaked treatment indicator.
- *Fix.* Use a single kernel width, chosen by a label-blind rule (for example a median-heuristic bandwidth on pooled selections from all arms) and fixed before unblinding. Better still, use a kernel-free statistic such as the effective rank of the selection Gram matrix, or pairwise-cosine histograms.

**`run_stage2_multiplayer_tournaments` (`:1119-1190`).**

- *What it does.* `theta_e` is set from `log(kl_anc / kl_agent)`, where `kl_agent` is itself produced by the parametric `mechanistic_redundancy_delta_c` model. Match outcomes are then sampled from the BTD likelihood with those `theta`.
- *Consequences:*
  1. The league contains **no game-play information**. Elo is a monotone recoding of the hard-coded `alpha_base·env_alpha_mult` multipliers.
  2. **BTD and mElo recovery is guaranteed.** The data are generated *from* the BTD model, so seat bias `gamma_e`, draw `nu_e`, and rank-2 antisymmetric structure are recovered up to sampling noise. This is a valid **parameter-recovery test** for the estimator (A-EXACT) and nothing more.
  3. **Non-transitivity is assumed away.** It is exactly what `mElo` exists to detect, and it arises from strategy interactions in real play (rock-paper-scissors cycles among search styles). A KL-derived scalar strength cannot be cyclic. The rank-2 component can only fit noise.
  4. **Mini-ECI bifactor recovery is the same story.** Recovering a 3-factor orthogonal structure from data simulated with a 3-factor orthogonal structure is a recovery test. It says nothing about whether real cross-game competence has that structure.

---

## 4. Focused Technical Prescription for Round 11 (`R11-GPU-NEURAL` on NVIDIA L4)

### 4.1 (i) `board-opus` Audit: Bitter-Lesson Violations and Realistic Effect Sizes

**The Bitter-Lesson charge, stated precisely.** The Bitter Lesson is not "never use structure." It is that *claims about what a learned system achieves must be backed by measurements of that learned system under scaled compute*. All three R10 constructs replace learning with hand-chosen coefficients:

| Construct | Hand-chosen quantity | What learning would have had to discover |
|---|---|---|
| `0.55·p_true + 0.45·p_anc` | 55% truth fidelity at root | Root calibration of a trained trunk on hidden-information boards |
| `l0 + alpha·0.86·(logit_exact − l0)` | 86% gap closure per update | Whether evidence integration converges to, or away from, the exact posterior |
| `mechanistic_redundancy_delta_c` | Entire budget → gain law | The empirical compute-scaling curve |

Each coefficient is a **free parameter fixed by the experimenter**. It is not a measured property of the model. The project's thesis is that local-consistency structure plus search lets a generalist *learn* to reason across games. A surrogate that inserts the reasoning outcome directly cannot test that thesis.

**Realistic effect-size priors for a from-scratch SGD-trained `GLCAOMWorldModel` on L4.** These priors assume:

- a 10–60M-parameter trunk;
- 24–72 L4 GPU-hours per arm;
- procedurally generated training data;
- the R10 environment roster (`sudoku_9x9_extreme`, `shiphunt_8x8`, `micro_stratego_6x6`, and the remaining families).

| Quantity | R10 surrogate value | `[PRIOR]` R11 expectation | Reasoning |
|---|---|---|---|
| Firewall exact zeros (`d l_V/d Z^Pi_aux`, RL `Delta_belief`) | `0.000000` | `0.000000` (must remain exact) | Architectural; training cannot change it. A non-zero value is a bug. |
| Search KL gain `1 − KL(128)/KL(0)`, ShipHunt | near-oracle | 15–45% | ShipHunt posterior is smooth and search helps, but trunk bias imposes a floor |
| Search gain, `sudoku_9x9_extreme` (solve rate) | parametric thresholds | +5 to +30 points over `B=0`, highly bimodal across boards | Extreme Sudoku needs deep propagation. A small trunk will have a weak `B=0` prior, so gain is dominated by tree-level consistency checks. |
| Search gain, `micro_stratego_6x6` | `0.55/0.90` mixtures | 0–15% belief KL; possible *negative* gain at small `B` | Hidden-information belief from a learned model is the regime where confident-wrong branches dominate |
| `R_leaf(8)` (leaf/parent calibration ratio) | `1.0000` | 0.80–1.10, family-dependent | Learned leaves are miscalibrated in a direction that varies with depth |
| Disjunctive veto rate | `0.00%` vs `49.23%` | 2–15% vs 20–50% | The direction of the advantage should survive. Zero will not. |
| Kish-ESS pooling MSE reduction | `−97.88%` | −10% to −40% vs stiffness pooling | The synthetic noise matched the estimator. Real depth bias is heteroscedastic and correlated. |
| Memory contradicted-belief suppression | `99.993%` | 50–90% | The surprise threshold must be calibrated on learned beliefs. Some genuine updates will be suppressed (a false-positive cost that must be reported). |
| Plan-swap action flip rate | `81.53%` (untrained) | 15–60% | Trained plan heads either collapse (low flips) or specialize (moderate flips). Flips alone are not a merit metric. |
| Scaling-law out-of-sample fit | `R^2 = 0.9982` (circular) | Held-out `R^2` 0.6–0.95; extrapolation error at `B=512` of 10–40% | A typical empirical scaling fit |
| Generalist vs specialist, in-family | not measured | Generalist 0–15% *worse* per family at equal total compute; parity or better on held-out families | Small-model negative transfer is the norm. Positive transfer usually appears only out-of-distribution. |
| LIR-masked vs token-masked | not measured | Difference within ±5% on belief KL; LIR advantage concentrated in policy-robustness (`IP-3`-style) metrics | The firewall's main benefit is *separation*, not raw accuracy |

**Expectation management.** If the realistic outcomes land in these ranges, GLC-AOM is **still a publishable, interesting system**. Exact belief–policy separation combined with a real but modest search gain is a genuine contribution. The R10 numbers, by contrast, would not survive review once the runner code was released. The project should commit now to reporting R11 results regardless of direction.

### 4.2 (ii) `Gamma_b` and `GCR(B)` Under Non-Zero Epistemic Bias

**Proposed operational definitions** (for adoption in the pre-registration):

Let `q_0` be the `B=0` belief and `q_B` the search-refined belief. Let `alpha ∈ [0, 1]` be a mixing weight and let `L(·)` be expected log-loss (cross-entropy against the realized hidden state) on a split.

- **Certified mixture.** `q_{B,alpha} = (1 − alpha)·q_0 + alpha·q_B`. This is a *linear* mixture in probability space, chosen for the guarantee in point 1 below.
- **Convex Gain Certificate.** `Gamma_b(alpha) = LCB_{1−δ}[ L_cal(q_0) − L_cal(q_{b,alpha}) ]`.
  - It is a lower confidence bound on the improvement, computed on a calibration split disjoint from test.
  - `alpha_b = argmax_alpha Gamma_b(alpha)`, with `alpha_b := 0` if `max Gamma_b ≤ 0`.
- **Gain Capture Ratio.** `GCR(B) = [L_test(q_0) − L_test(q_{B,alpha_B})] / [L_test(q_0) − min_alpha L_test(q_{B,alpha})]`.
  - This is the realized gain as a fraction of the hindsight-optimal mixture gain on test.
  - It is reported only when the denominator is above a pre-registered floor `ε_den`. Otherwise the cell is marked "no available gain."

**Behavior with a biased learned world model:**

1. **Oversearch blow-up is bounded by construction.** For a linear mixture and any realized state `x`, `−log q_{B,alpha}(x) ≤ −log(1 − alpha) − log q_0(x)`.
   - Excess loss over the prior is therefore at most `log(1/(1 − alpha))` per prediction, whatever the search does.
   - Capping `alpha ≤ alpha_max` (for example 0.9, giving at most `log 10 ≈ 2.3` nats worst-case) turns catastrophic oversearch into bounded regret.
   - The same inequality with the roles swapped gives `≤ −log alpha − log q_B(x)`. The mixture is never much worse than the *better* of the two components.
   - **Log-odds (geometric) pooling has no such bound.** Under bias, it can blow up arbitrarily. This is the core reason to certify in probability space, mirroring the Stage-1 disjunctive-mixture argument.
2. **Bias shifts the optimal `alpha` inward.**
   - With an unbiased model, `alpha* → 1` as `B` grows.
   - With trunk bias, `q_B` concentrates on the *model's* posterior, not the true one. Then `alpha*` plateaus below 1, and it can *decrease* at large `B` as `q_B` becomes over-confident.
   - Expect a non-monotone `alpha*(B)`. Allow it, but use the certificate to make the *loss* monotone in expectation.
3. **The opposite risk is vacuous abstention.** If the LCB is too conservative (small calibration set, union bounds over many `alpha`, heavy-tailed log-loss), then `alpha_b = 0` everywhere. Monotonicity then holds trivially and GCR is 0. The prescriptions against this:
   - Use **empirical-Bernstein or bootstrap-t LCBs on clipped losses** (clip `q ≥ 1e-6`; pre-register the clip), not Hoeffding.
   - Use **a coarse `alpha` grid** (`{0, 0.1, …, 0.9}`) to keep the multiplicity correction small.
   - Use **hierarchical sharing**: fit `alpha_b` per `(family, B)` with partial pooling across depth strata. Per-cell fits from tiny samples abstain; global fits ignore heterogeneity.
   - Use **calibration-set sizing**: pre-register `n_cal ≥ 2,000` boards per family, so the LCB width at the expected gain (Section 4.1) is less than half the gain.
   - Use **an epistemic-uncertainty modulator**: compute a disagreement statistic `u(s)` (for example from 3 bootstrap heads or MC-dropout on the trunk) and make `alpha` a monotone decreasing function of `u`, `alpha(s) = alpha_b · σ(κ − λ·u(s))`. Certify `(κ, λ)` jointly. This lets the system trust search where the model is confident about its own reliability, instead of abstaining globally.
4. **Monotonicity under the certificate.**
   - Nested budgets with per-budget `alpha_B` give *expected* monotonicity up to calibration error.
   - Finite-sample violations will occur.
   - The gate (`R11-D5`) bounds their frequency and magnitude rather than demanding zero.
5. **Reporting requirement.** Every `GCR(B)` must be accompanied by:
   - the abstention rate `P(alpha_B = 0)`;
   - the mean `alpha_B`;
   - the worst-cell excess loss.

   High GCR at 90% abstention is not a success.

### 4.3 (iii) GPU Training & Compute-Matched Evaluation Protocol

#### 4.3.1 Hardware Accounting on L4

The L4 has 24 GB of memory, roughly 300 GB/s of memory bandwidth, and a nominal ~120 TFLOPS dense BF16. Its key constraint is that **paged-KV tree search is memory-bandwidth-bound on L4**, not FLOP-bound. Compute matching therefore must use **three ledgers**, all logged per run:

1. **Training FLOPs.** Estimated as `6 · N_params · N_tokens` plus measured overhead for auxiliary heads. Cross-checked with a profiler on 100 steps.
2. **Inference cost per decision.**
   - Node expansions;
   - forward-pass FLOPs;
   - **KV bytes read and written** (from the `TreePagedKVBank` counters).
3. **Wall-clock time** on the same L4 SKU, same driver/CUDA, with batch size disclosed.

The primary matching unit is training FLOPs for training and node expansions for inference. Results are also re-plotted against bytes and wall-clock, and any ranking reversal must be reported.

#### 4.3.2 Data and Splits

- **Training split.** Procedural generators seeded from a committed range `S_train`.
- **Calibration split.** Range `S_cal`, used for `alpha_b`, temperature, and memory thresholds.
- **Test split.** Range `S_test`, generated only after models are frozen.
  - Seeds are derived by `HMAC(commit_hash_of_frozen_weights, family)`, so they cannot be known in advance.
- **Held-out families.** At least one environment family never seen in training, used only for generalist-transfer claims.
- **Deduplication.** Canonicalized board hashes (Sudoku symmetry group, ShipHunt reflections/rotations) are deduplicated across splits.

#### 4.3.3 Arms

**Experiment G (Generalist vs Specialist).** Let K be the number of training families.

| Arm | Description | Training compute |
|---|---|---|
| `GEN` | One `GLCAOMWorldModel` trained on a uniform family mixture | `C` |
| `SPEC-TOT` | K specialists, each trained on one family | `C/K` each (**total-compute matched**) |
| `SPEC-PER` | K specialists | `C` each (**per-game matched**; the generalist is disadvantaged) |

- Primary comparison: `GEN` vs `SPEC-TOT`.
- `SPEC-PER` bounds the cost of generality.
- Held-out-family evaluation applies to `GEN` only. `SPEC-*` specialists are reported with their best family-to-family zero-shot transfer as the baseline.

**Experiment M (Masking).** Identical architecture, parameter count (±1%), data stream, optimizer, steps, and seeds. Only the masking scheme differs:

| Arm | Description |
|---|---|
| `LIR` | `LIRMask` with `chi=(0,1)/(1,1)` and the `Z^W_aux (+) Z^Pi_aux` firewall |
| `TOK` | Standard random token masking, with the same mask-rate schedule and no firewall |
| `FUSED` | Joint objective over belief and policy through a shared aux space (firewall removed, LIR targets kept) |

**Experiment B (Search baselines at matched inference budget).** Each trained model is run with:

- (a) `B=0`;
- (b) independent sampling / best-of-N at matched expansions;
- (c) flat MCTS without KV sharing;
- (d) full `TreePagedKVSearch`.

This isolates the tree-and-KV contribution from simply "more samples."

#### 4.3.4 Hyperparameters and Seeds

- **Equal tuning budget.** Each arm gets 16 configurations, selected by random search on the calibration split, and is then frozen. The tuning-search spaces are committed to the pre-registration.
- **Training seeds.**
  - At least 5 per arm for Experiment M (cheap: shared data).
  - At least 3 per arm for Experiment G, where compute is tight. If 3 seeds leave the primary CI wider than the SESOI, the result is reported as *inconclusive*, not as a pass.
- **Seed ranges** are committed. No seed can be dropped after training starts. Crashes are re-run with the same seed and reported.

#### 4.3.5 Statistics

- **Paired design.** All arms are evaluated on the identical test boards.
- **Hierarchical bootstrap.** Resample seeds, then boards within family; 10,000 replicates; report 95% BCa intervals.
- **Pre-registered SESOI** (smallest effect size of interest):
  - belief KL: 5% relative;
  - solve/win rate: 3 percentage points;
  - Elo: 30 points.
- **Equivalence testing** (TOST) for non-inferiority claims, such as LIR not being worse than TOK on belief KL.
- **Holm–Bonferroni correction** within each gate family.
- **Reporting.** All pre-registered metrics are reported, including failures. Secondary analyses are labelled as exploratory.

#### 4.3.6 Indicative Compute Budget

- 48 L4-hours for the `GEN` arm, so `C ≈` 48 L4-hours. `SPEC-PER` costs `K·C` and is the budget driver. If it is unaffordable, pre-register `SPEC-PER` on the 3 lead families only.
- Experiment M: 3 arms × 5 seeds × ~16 L4-hours on a reduced model.
- Evaluation: ~20% of training compute.

The total must be stated in the pre-registration before any run starts. Overruns are disclosed.

---

## 5. Pre-Registration Gate Contributions (`R11-D0` through `R11-D6`)

Each gate is either **PASS**, **FAIL**, or **INCONCLUSIVE** (CI straddles the threshold). INCONCLUSIVE counts as *not passed*. No gate may be re-defined after test-split generation.

### `R11-D0`: Zero-Oracle-Leakage Firewall (Static AST + Runtime)

`R11-D0` is a hard prerequisite. If it fails, every other gate is void.

**Static rules.** These run over all modules on the import closure of the training and model-prediction paths (`src/**`, `benchmarks/run_r11_*.py`) using an AST walker committed to the repo, with a hashed allow-list.

- **S1. No oracle identifiers.** No reads of identifiers matching the oracle denylist on the prediction path: `p_true*`, `p_exact*`, `logit_exact`, `q_star*`, `true_q`, `solution*`, `hidden_state*`, `p_ship_exact`, `*_oracle*`.
- **S2. No synthetic belief generators.** No calls to `*.beta(`, `*.dirichlet(` or similar RNG distribution samplers *whose outputs flow into model-output tensors* (checked by def-use analysis). RNG is permitted only for search tie-breaking and dropout, and only via a registered `SearchRNG`.
- **S3. No law-under-test on the evaluation path.** No references to `mechanistic_redundancy_delta_c`, `btd_probabilities` (as a *generator*), or any function on a committed "law-under-test" list anywhere in the data-generation or evaluation-outcome path.
- **S4. No configuration-conditional branches.** No conditional whose test compares `seed`, `b_budget`, `m0`, a family name, or an arm label against a literal on the prediction path. Explicit dispatch tables registered in config are exempt.
- **S5. No arm-dependent smoothing.** No arm-label-dependent numeric literals in metric computation, which would catch kernel-width schemes like the one in `compute_empirical_gram_crowding`.

**Runtime rules:**

- **R1. Poisoned oracle labels.** `EnvBatch.strip_oracle_labels()` returns a batch whose oracle fields are replaced by `PoisonTensor` objects. Any `__torch_function__` call, `.item()`, indexing, or Python numeric conversion on a `PoisonTensor` raises `OracleLeakError`. A full test-split rollout must complete with zero `OracleLeakError`.
- **R2. Canary invariance.** The same test boards are run twice. The second run replaces all oracle labels in the *stripped* storage with random permutations before stripping.
  - Model outputs, beliefs, actions, and search trees must be **bitwise identical** under deterministic CUDA settings.
  - Alternatively, if GPU nondeterminism prevents bitwise equality, they must agree to `max|Δ logit| ≤ 1e-5`, with the nondeterminism floor measured from two runs using identical labels.
- **R3. Scoring separation.**
  - Predictions are written and SHA-256-hashed, and the hash is committed (git tag) *before* the scoring process starts.
  - The scoring process is a separate OS process. It is the only one that loads oracle labels.
  - The scorer verifies the prediction hash.
- **R4. Trained-weights check.** Each evaluated checkpoint must have a training log showing at least the pre-registered number of optimizer steps. The checkpoint's parameter L2 distance from initialization must exceed a pre-registered minimum (guarding against "untrained weights" regression). Its `B=0` validation loss must beat both:
  - a uniform-prior baseline;
  - a pre-registered simple heuristic baseline per family (constraint propagation for Sudoku; density prior for ShipHunt).

**PASS iff** S1–S5 report zero violations, R1–R3 hold on 100% of test runs, and R4 holds for every checkpoint.

**Retroactive item (not a gate, but mandatory before R11 opens).** Disclose the `:1351` seed-conditional branch, its origin, and its effect on R10-E2 aggregates. Relabel every R10/Stage-2 empirical number as `SURROGATE` in all project documents.

### `R11-D1`: Universal KV-Cache Tree Machine

On trained `GEN` checkpoints, with live `TreePagedKVSearch.run_search`, `B ∈ {0, 8, 32, 128, 512}`:

- **D1.a (search gain).** Relative belief-KL gain `1 − KL(128)/KL(0)` ≥ 5% (SESOI), with the 95% CI lower bound > 0, on at least ⌈2K/3⌉ of K training families.
- **D1.b (tree beats sampling).** At matched node expansions (`B=128`), full tree search beats independent sampling / best-of-N (Experiment B (b)) on the primary metric, CI lower bound > 0, on at least half of the families.
- **D1.c (KV sharing pays for itself).**
  - KV-reuse ratio (bytes saved / bytes of non-shared baseline) ≥ 30% at `B=128`.
  - Wall-clock per decision ≤ 0.8× the flat-MCTS baseline at equal expansions.
- **D1.d (descriptive only).** Topology statistics (`f_root`, `f_tip`, `f_fork`) are reported for trained vs untrained weights. No pass threshold.

**FAIL** if D1.a fails on more than one third of families.

### `R11-D2`: Active Fractal Prospectus Plans

- **D2.a (plans help).** The plan-head ablation (plan conditioning replaced by a learned constant, retrained with equal compute) loses ≥ SESOI on the primary metric on at least half of the families.
- **D2.b (plan diversity).** On trained checkpoints, the effective number of distinct plans in use (exp of entropy of the plan-usage distribution over test boards) is ≥ 3. This guards against collapse.
- **D2.c (beneficial causal swap).** A plan swap must change actions (flip rate ≥ 10%). Swapping in the plan selected by the model must also outperform a random plan swap on outcome, CI lower bound > 0.
- **D2.d (descriptive).** Gram 24-sector coverage is reported, with no threshold, because it is a projection property.

### `R11-D3`: Bidirectional Latent Lemma Space & Queries

- **D3.a (firewall exactness retained).** On trained checkpoints:
  - `||d l_V / d Z^Pi_aux||_inf == 0` exactly;
  - after 300 RL steps on `get_policy_parameters()`, `Delta_belief == 0` exactly;
  - `D_bias(B=0) == 0` exactly.

  Any non-zero value is FAIL.
- **D3.b (lemma channel used).** Zeroing `Z^W_aux` at inference raises belief KL by ≥ SESOI on at least half of the families. The lemma-ablation retrain (no emitter, equal compute) must be worse than full `LIR`, CI lower bound > 0, on the pooled metric.
- **D3.c (masking comparison, Experiment M):**
  - `LIR` is non-inferior to `TOK` on belief KL (TOST, margin 5%).
  - `LIR` beats `FUSED` on policy-robustness under utility swap: belief change under `swap_utility` is exactly 0 for `LIR`, measured and reported for `FUSED`.
  - Any claim that `LIR` is *superior* on accuracy requires a CI lower bound > SESOI.
- **D3.d (bias amplification check).** Report the correlation between emitter output magnitude and belief error on test. If that correlation is positive and significant, the even-parity mechanism is amplifying correlated bias. This must be disclosed. It is not a pass criterion.

### `R11-D4`: Present-Time Episodic Memory

- **D4.a (gain on clean memory).** In multi-episode sequences with consistent hidden structure, memory-on beats memory-off by ≥ SESOI on at least one family designed for memory (pre-named).
- **D4.b (contradiction robustness).** With memories from contradicting episodes injected at a pre-registered rate of 30%, memory-on suffers ≤ 20% of the degradation suffered by an ungated memory baseline. Suppression must be ≥ 80%, CI lower bound reported.
- **D4.c (false-suppression cost).** On clean sequences, the gated memory loses ≤ 2% relative to the ungated memory. This is non-inferiority via TOST.
- **D4.d (no oracle in the surprise signal).** The surprise gate is computed only from model-internal quantities. This is covered by `R11-D0` S1 and must be shown explicitly in the gate report.

### `R11-D5`: Anytime Monotonicity & Self-Consistency

- **D5.a (certified monotonicity).** For consecutive budgets `B < B'` in the grid, the fraction of `(family, B→B')` cells where test loss of `q_{B', alpha_{B'}}` exceeds that of `q_{B, alpha_B}` by more than the noise band is ≤ 10%. The noise band is 2× the bootstrap SE.
- **D5.b (non-vacuous certificate).**
  - Mean `GCR(128)` ≥ 0.5 across families where the denominator exceeds `ε_den`.
  - Abstention rate `P(alpha_128 = 0)` ≤ 30%.
  - Mean `alpha_128` ≥ 0.2.
- **D5.c (bounded regret).** Worst-cell excess loss over `q_0` ≤ `log(1/(1 − alpha_max))`, with `alpha_max = 0.9`. This is guaranteed analytically, but it is verified empirically as an implementation check.
- **D5.d (scaling law, out-of-sample).**
  - Pre-registered functional forms:
    - (F1) the mechanistic redundancy law, as a *hypothesis* fitted to measured data;
    - (F2) power law with floor;
    - (F3) saturating log.
  - Fit on measured `KL(B)` at `B ∈ {0, 4, 16, 64}` from `S_test` boards in a **fit-half**.
  - Evaluate on the **evaluate-half** at `B ∈ {256, 512}` and on held-out families.
  - **PASS for F1** iff:
    - its held-out log-likelihood beats F2 and F3, or ties within the bootstrap SE while using no more parameters;
    - its extrapolation error at `B=512` is ≤ 25% relative.
  - **Ban.** Reporting any `R^2` computed on data generated by F1 itself (enforced via `R11-D0` S3).
- **D5.e (calibration generalization).** `|KL_test − KL_cal|` for the calibrated clearing parameters is ≤ 10% relative, measured on learned edge deltas.

### `R11-D6`: Multi-Game Elo League & Mini-ECI IRT

- **D6.a (real play).** All league outcomes come from interactive turn-by-turn rollouts in which each agent selects actions with its own trained model and search at a pre-registered budget.
  - The environment engine adjudicates outcomes.
  - The log stores full move sequences. A random 1% replay audit must reproduce the logged outcomes exactly.
  - `btd_probabilities` may appear only in the *estimator*, never in outcome generation (`R11-D0` S3).
- **D6.b (sample size).**
  - At least 200 games per ordered agent pair per two-player family, seat-balanced.
  - The Elo 95% CI half-width is ≤ 40 points for every agent.
- **D6.c (generalist strength).** `GEN` Elo ≥ `SPEC-TOT` Elo − 30 (non-inferiority, TOST) on at least half of two-player families. Any superiority claim requires the CI lower bound > +30.
- **D6.d (non-transitivity is measured, not assumed).** Report the share of variance explained by the rank-2 antisymmetric `mElo` component on held-out games. Also report the Nash mixture `p*_e`. If the rank-2 component does not improve held-out log-likelihood over scalar BTD, report the league as effectively transitive. This is descriptive.
- **D6.e (Mini-ECI on real data).** Fit the bifactor IRT to the real `(agent × task-item)` response matrix. The bifactor model must beat a unidimensional model on held-out item log-likelihood (5-fold CV over items) for any claim of multi-factor generalist structure. A parameter-recovery simulation is reported separately and labelled as an estimator check.

### Integrity Panel Redesign (Applies to All Gates)

The current IP-1..IP-5 passed a runner with a seed-conditional branch, so they need structural changes:

- **IP gates execute `R11-D0` themselves.** They no longer accept runner self-reports.
- **An IP member with no authorship of the runner** reviews the diff of every `benchmarks/run_r11_*.py` against the pre-registration commit before test seeds are derived.
- **Any post-registration change to a runner** invalidates the affected gates until the diff has been re-reviewed and disclosed.

---

*End of Member 4c Round-2 critique.*
