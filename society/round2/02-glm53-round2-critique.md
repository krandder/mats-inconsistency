# Round-2 Critique & `R11-GPU-NEURAL` Design Specification

**Member 4a — GLM-5.3 (`zai-coding-plan/glm-5.3`)**
**Society of Thinkers, Round 2 — `futarchy-fi/inconsistency-search` (GLC-AOM: Generalist Local-Consistency Any-Order Machine & Universal KV-Cache Tree Search)**
**Artifact:** `society/round2/02-glm53-round2-critique.md` · **Status:** Final · **Date:** 2026-10-18
**Inputs adjudicated:** `01-round2-code-and-results-audit.md` (Member 12 forensic line-by-line audit)

---

## 1. Executive Verdict & Scientific Audit Synthesis

**Verdict: the Round-2 forensic audit is accepted in full and without qualification.** Round 10 and Stage 2 produced two categorically different kinds of evidence that were reported under a single `PASS` label, and that conflation is the central scientific failure of the round. The correct reclassification is:

1. **What stands (Unit-Math-Verified, `src/`):** The belief–policy firewall (`LIRMask` χ=(0,1)/(1,1), `Z^W_aux ⊕ Z^Pi_aux`, zero-initialized readout, `||∂l_V/∂Z^Pi_aux||_∞ == 0.0` under exact autograd, `Δ_belief == 0.0` after 300 RL steps, 100% utility-swap flip), the even-parity bilinear emitter, the two-stage disjunctive-Bayes/Gershgorin tree clearer (`||T_128||_∞ = 0.999947 < 1`, `R_leaf(8) = 1.0000`, 0.00% vs 49.23% piece-veto), the Kish-ESS trust pooling (−97.88% MSE), the memory-transport suppression (−99.993%), and the exact estimators (BTD MAP, rank-2 antisymmetric mElo, 3-factor Bifactor IRT) are **mathematical properties of operators**. They were verified with exact autograd, exact swaps, and spectral arguments that hold for *all* parameter settings. They are not diminished by the untrained trunk, and they remain assets of the codebase.

2. **What is void (Runner Surrogates, `benchmarks/run_*.py`):** Every end-to-end empirical claim — budget-scaling curves, calibration generalization, anytime improvement, redundancy-law confirmation (`R² = 0.9982`), the P9 correlation (`r = 0.8078`), league standings, and IRT difficulty structure — was manufactured by injecting oracle quantities (`p_true`, `logit_exact`, `q_star`) or parametric closed forms (`mechanistic_redundancy_delta_c`) into the very quantities the experiments claimed to measure. These results carry **zero evidential weight about the learned system** because the system was never trained and never blindly evaluated.

**Why the separation is essential to scientific integrity.**

- **Claim–evidence parity.** A `PASS` on `R10-E1` currently certifies "the lemma pathway improves calibration with budget," but the code measures `(1 − ls(b))·0.5 + ls(b)·q_true` — a fixed interpolation *schedule*, not a learner. The runner's `ls(b)` ramp mechanically determines the reported curve. The PASS is a true statement about the wrong object. Unit-verified operator claims and oracle-dosed runner claims must therefore never share a label, a table, or a summary sentence.
- **Falsifiability.** The `fit_mechanistic_redundancy_law → R² = 0.9982` result cannot fail by construction: the law generated the data (`kl_val = max(0.004, kl0·exp(−δ_c)) + oversearch`). An unfalsifiable confirmation is not evidence; reporting it alongside genuine spectral bounds launders the credibility of the latter onto the former.
- **Hidden prior-injection.** `compute_empirical_gram_crowding` convolves the one genuinely empirical ingredient (real `B=32` step selections) with algorithm-specific kernel widths (0.55 / 2.2 / 3.0–6.0) and background offsets (0.22–0.55). Family-level crowding conclusions are reparameterizations of hardcoded hyperparameters, and `rho_env = 0.55·rho_base + 0.45·rho_gram` pre-installs the P9 correlation.
- **One item is not a surrogate but a planted leak.** `run_r10_e1_e2.py:1351` (`if m0 == 34 and seed == 102 and b_budget == 2:` scaling the `C4` leak) is outcome-, seed-, and budget-conditional hardcoding. In a pre-registered protocol this is indistinguishable from fabrication and must be treated as an integrity incident: the R10-E2 Strong-Root ShipHunt cell is void in its entirety, and any aggregate PASS computed over it is void.
- **"Live" is not "learned."** Even the genuinely executed `run_search` at `B=128` (`run_r10_e3_e4.py:368-379`) ran on an **untrained** trunk. The reported fractal topology (`f_root = 0.1405, f_tip = 0.7969, f_fork = 0.0626`), sector coverage, and plan-swap rates are properties of the random prior — the initialization distribution — not of any learned world model. They do not transfer to trained behavior and must not be cited as end-to-end evidence.

**Consequence:** Round 11 must be rebuilt around a single principle — *the only numbers that can be reported as empirical findings are those produced by a trained trunk, driven by live search, on oracle-stripped inputs, evaluated once on held-out data under a pre-registered gate.* Sections 4–5 specify exactly that.

---

## 2. Round-2 Updated Rubric Table (D1–D6, Scale 1.0–10.0)

| Dimension | Math/Unit Verified in `src/` (R10/Stage-2) | End-to-End Blind Empirical Validity (Current Runners) | Projected Post-`R11-GPU-NEURAL` (After Blind L4 Training & Rollouts) |
|---|---:|---:|---:|
| **D1. Universal KV-Cache Tree Machine** | **9.2** | **3.4** | **8.3** |
| **D2. Active Fractal Prospectus Plans** | **8.8** | **2.6** | **7.8** |
| **D3. Bidirectional Latent Lemma Space & Queries** | **8.9** | **2.2** | **7.4** |
| **D4. Present-Time Episodic Memory** | **8.6** | **2.5** | **7.6** |
| **D5. Anytime Monotonicity & Self-Consistency** | **8.4** | **1.9** | **7.2** |
| **D6. Multi-Game Elo League & Mini-ECI IRT** | **8.5** | **1.7** | **7.0** |
| **Unweighted Mean** | **8.73** | **2.38** | **7.55** |

**Cell justifications (evidence → score):**

- **D1 = 9.2 / 3.4 / 8.3.** Gershgorin contraction, parent-relative clearing (`R_prefix(16) = 1.0079`), and disjunctive Bayes marginalization (0.00% vs 49.23% veto) are exact and durable → 9.2. Live `B=128` execution is real, but on an untrained trunk with budget curves from `1 − exp(−0.34·b/(L/3.5))` and beta-drawn `m_ql` (`run_r10_e3_e4.py:398-410`) → 3.4. Post-R11, live search on a trained trunk with raw-bank statistics is achievable on L4 → 8.3 (residual risk: search-quality coupling to imperfect value heads).
- **D2 = 8.8 / 2.6 / 7.8.** Gram orthogonalization and causal plan-swap (81.53%) are clean operator properties → 8.8. But sector/plan claims were measured on the random prior, and plan-value claims inherit R10-E4 oracle-perturbed `q_roll` → 2.6. Post-R11: plan ablations against a trained trunk are straightforward; coverage under learned embeddings is less guaranteed → 7.8.
- **D3 = 8.9 / 2.2 / 7.4.** The even-parity lemma and the firewall are the round's strongest results (exact zeros under autograd and RL) → 8.9. Yet the *only* evidence that lemmas help inference was `(1 − ls)·unif + ls·p_true` — the oracle mixture *is* the claim → 2.2. Post-R11 is the highest-variance projection: `Z^W_aux` rank collapse and off-policy drift under real SGD are unsolved empirical questions → 7.4.
- **D4 = 8.6 / 2.5 / 7.6.** Surprise-discounted gating and contradicted-memory suppression are unit-solid → 8.6. `p_with_mem` was a noisy perturbation of `log(p_exact)` (`run_r10_e3_e4.py:994-996`); memory never operated on real episodes → 2.5. Post-R11: retrieval over genuinely experienced transcripts is a standard, testable design → 7.6.
- **D5 = 8.4 / 1.9 / 7.2.** Spectral bound and small-sample calibration generalization are real operator results, though the latter was semi-synthetic (1,030 ≤ 8,192 params) → 8.4. Anytime curves were the *shape of the injected law* (`exp(−δ_c(b))`); monotonicity was true by construction → 1.9. Post-R11: honest certificates are obtainable, but monotonicity under imperfect trunks is empirically fragile; we pre-register tolerance bands, not perfection → 7.2.
- **D6 = 8.5 / 1.7 / 7.0.** Exact BTD MAP, antisymmetric mElo SVD + Nash mixture, bifactor recovery are correct mathematics → 8.5 (recovery demonstrated on simulated latents only, hence not 9+). Matches were *sampled from `btd_probabilities`* with `theta_table = log(kl_anc/kl_agent)` — the league ranked deterministic functions of simulated KLs, never policies → 1.7. Post-R11: playing real games is compute-feasible on L4 but sample sizes will be modest → 7.0.

---

## 3. Deep-Dive Critique of the Stage-1 & Stage-2 Runner Surrogates

### 3.1 (a) Untrained trunks + oracle-interpolated log-odds: identity experiments disguised as learning experiments

**The mixtures are the claims.** Consider the three canonical injections:

- `q_a_given_b = (1 − ls)·unif + ls·p_a_given_b_true` (`run_r10_e1_e2.py:586`)
- `pred_q = (1 − lemma_strength)·0.5 + lemma_strength·true_q` (`run_r10_e1_e2.py:598`)
- `l_arm = l0_ship + alpha_gate·0.86·(logit_exact − l0_ship)` (`run_r10_e1_e2.py:1387`)
- `0.55·p_true + 0.45·p_anc` / `0.90·p_true + 0.10·root_probs` (`run_r10_e1_e2.py:1119,1142,1152`)

In each case the reported quantity is an affine combination whose endpoints are (i) an ignorance prior and (ii) the ground truth, with a runner-chosen mixing weight. Every downstream metric — calibration ECE, KL, solve rate, veto rate — is then a **deterministic functional of the mixing schedule `ls(b)`, `alpha_gate(b)`**. If the schedule is monotone in budget, the "anytime improvement" is monotone. The experiment measures the runner's own interpolation clock, and the model under test contributes only the initial point `l0_ship`, which came from an **untrained** trunk, i.e., a random draw. The R10-E1 "hit-rate" law `1 − exp(−0.34·b/(L/3.5))` is even more direct: the budget curve was written down analytically before any search ran.

**Failure modes this masks:**

1. **No learning-dynamics information.** We learn nothing about whether gradient descent on `GLCAOMWorldModel` converges, whether `Z^W_aux` carries signal after training, or whether the lemma pathway improves inference for any model that did not already contain the answer.
2. **Firewall-at-train-time is untested.** The verified firewall results probe post-hoc Jacobians. They do not show that *joint* training of belief and policy heads under the mask actually admits useful gradient flow to the trunk — the practically important question — nor that 300 RL steps at scale leave beliefs intact. The unit result is necessary, not sufficient.
3. **Random-prior "fractality."** The live topology statistics (f_root/f_tip/f_fork, 99.96% sector coverage) were computed on a random network. Near-uniform random logit fields generate argmax trees whose depth statistics can look fractal by branching-process accident. Reporting them alongside learned-system claims invites transfer error.
4. **Bilinear emitter signal vs. form.** The even-parity lemma verifies the emitter's *functional form* (no sibling cancellation in `Z^W_aux`). It says nothing about whether trained co-fluctuations `(U^T Δl_V)(V^T Δl_V)` are *informative*. Under a random trunk they are noise; the lemma faithfully preserves noise.
5. **The planted conditional leak.** `if m0 == 34 and seed == 102 and b_budget == 2:` (`run_r10_e1_e2.py:1351`) makes one (seed, budget, arm) cell artificially better. This is beyond surrogate methodology: it is a per-outcome constant that no blind protocol could produce. It voids the containing experiment and mandates the R11-D0 branch detector (§5).

### 3.2 (b) Circular parametric scaling-law generation: a law that cannot fail

The Stage-2 pipeline generates every `(family, env, B, seed)` cell's metric from the law itself:

```
rho_env   = 0.55·rho_base + 0.45·rho_gram                      (line ~880)
delta_c   = mechanistic_redundancy_delta_c(b, alpha_base·env_alpha_mult, rho_env)   (line 913)
kl_val    = max(0.004, kl0·exp(−delta_c) + oversearch)          (line 922)
```

and then confirms the law by fitting it to its own output (`fit_mechanistic_redundancy_law`, `R² = 0.9982`, `r = 0.8078` at `run_stage2_mini_eci.py:2416-2417`).

**Failure modes this masks:**

1. **`R² = 0.9982` is the RNG noise floor, not a discovery.** The only sources of deviation from the fitted law are the `max(0.004, ·)` floor (which *censors* the low-KL tail and inflates apparent fit quality exactly where the law is most at risk), the additive `oversearch` term, and finite seed noise. The protocol has no capacity to falsify the law, so its confirmation is contentless.
2. **P9 (`corr(rho_IRT, rho_Gram)`) is pre-installed.** With `rho_env` containing a 0.45-weighted copy of `rho_gram`, and both `rho_IRT`-bearing quantities and `rho_gram` sharing `alpha_base·env_alpha_mult` coupling, a substantial positive correlation is guaranteed by construction. The observed 0.8078 cannot be attributed to any property of search or cognition.
3. **Censoring hides the oversearch regime.** The `0.004` floor means the very budget region where real systems might *degrade* (the "oversearch" phenomenon the additive term gestures at) is clipped to a constant — precisely the regime D5 (anytime monotonicity) most needs to see honestly.
4. **Contamination propagates downstream.** `theta_table = log(kl_anc/kl_agent)` makes agent "strength" a deterministic function of the simulated KLs; league ranks, IRT difficulties, and the Nash mixture `p*_e` are all functions of the same fiction. Nothing in Stage 2's tournament block constrains reality.

### 3.3 (c) Kernel-smuggled crowding and simulated tournaments

1. **`compute_empirical_gram_crowding` (`run_stage2_mini_eci.py:608-640`):** the *real* ingredient — actual `TreePagedKVBank` step selections at `B=32` — is convolved with per-algorithm kernel widths (0.55 vs 2.2 vs 3.0–6.0) and background offsets (0.22–0.55). Declaring "family F's search footprint is more crowded" after assigning family F kernel width `w_F` is a tautology: the measured separation between families is a reparameterization of the hardcoded hyperparameters. Any family-ranking, crowding–redundancy linkage, or Gram-based `rho` derived from this path is prior-injection, not measurement.
2. **`run_stage2_multiplayer_tournaments` (`run_stage2_mini_eci.py:1119-1190`):** outcomes are Bernoulli/ternary draws from `btd_probabilities` whose `theta_table` was itself derived from simulated KL ratios, then handed to the exact BTD MAP estimator that generated them. This is an estimator self-test (sampling-noise recovery), already covered — and better covered — by the unit tests. It ranks no policy, exercises no game tree, and identifies seat bias `γ_e` and draw rate `ν_e` only because they were injected. Calling its output a "league table" or "Elo rating" for GLC-AOM agents is the round's clearest example of validation theater.

**Synthesis of §3:** the runner layer systematically replaced the three things experiments exist to produce — learning, blind evaluation, and the possibility of failure — with schedules, mirrors, and priors. Round 11 exists to restore them.

---

## 4. Focused Technical Prescription for Round 11 (`R11-GPU-NEURAL`, NVIDIA L4)

Design constraints: single L4 (24 GB, bf16, ~120 TFLOPS peak) → target a 10–50M-parameter `GLCAOMWorldModel` trunk, gradient checkpointing, effective batch 64–256 games, ≤ ~36 GPU-hours per data-aggregation round, ≤ ~200 GPU-hours total across `sudoku_9x9*`, `shiphunt_8x8`, `micro_stratego_6x6` plus one held-out generalization family.

### 4.1 (i) End-to-end training of `GLCAOMWorldModel` + `BilinearAuxLemmaEmitter` without off-policy drift or rank collapse

**Data: validated self-generated transcripts (DAgger-style aggregation).**

- **Policy mixture for generation:** ⅓ uniform-legal, ⅓ ε-greedy under current policy (ε = 0.25), ⅓ `TreePagedKVSearch.run_search`-guided using the *current frozen* trunk. This covers the state distribution the deployed machine will actually visit while preserving exploration breadth.
- **Hard transcript validation:** every (s, a, r, s′) tuple is checked against the environment simulator's legality/reward rules (sudoku constraint propagation, shiphunt salvo rules, stratego legality); invalid transcripts are discarded and the discard rate is logged as a data-quality metric. Only validated transcripts and their *raw* bank fields (`edge_keys_bank`, `m_ql_bank`, visit counts, depth-tagged value planes) enter training — no synthetic `beta` draws, no smoothed curves.
- **Aggregation rounds:** regenerate 25% of the corpus with the updated policy every ~5k optimizer steps (DAgger), so the training distribution tracks the closed-loop search distribution rather than an increasingly stale prior.

**Objective (multi-term, all on the trained graph):**

```
L = L_nextstate_NLL + λ_r·L_reward_huber + λ_v·L_value_NLL(discretized)
  + λ_W·L_lemma( Z^W_aux targets from observed co-fluctuations Δl_V on validated transcripts )
  + λ_vc·L_VICReg( Z^W_aux )   # variance–covariance regularizer: anti-collapse
```

- **Firewall by construction, verified at train time:** the policy loss is computed through `LIRMask` χ=(0,1) so `∂L/∂Z^Pi_aux-trunk-paths` is structurally zero *during* optimization. Every 500 steps, re-run the exact autograd probe (`||∂l_V/∂Z^Pi_aux||_∞ == 0.0`) and a 20-step RL probe (`Δ_belief == 0.0`) **on the live training graph**, not post-hoc. Any nonzero → halt, file, fix.
- **Anti-rank-collapse for `Z^W_aux`:** log the stable rank `||Z||_F²/||Z||_2²` (and effective rank via Shannon entropy of the singular spectrum) every 1k steps. Gate: stable rank ≥ 50% of its initialization value for the entire run; on breach, escalate `λ_vc` (variance floor + off-diagonal covariance penalty) and log the intervention. The even-parity bilinear emitter is norm-ambiguous — it can satisfy its symmetry while shrinking to zero — so pair it with a scale-floor parameterization or a norm floor on `||Z^W_aux||_F`.
- **Off-policy drift control:** (1) Polyak-averaged **target trunk** (momentum 0.995) supplies all bootstrap value targets, freezing the moving target; (2) truncated importance weights (clip [0.5, 2.0]) on off-policy value targets to bound variance; (3) an **on-policy drift monitor**: every 1k steps, KL(current-policy eval batch ‖ training-corpus visit distribution); alarm at KL > 0.15 nats → trigger early DAgger round.
- **Acceptance inside training (not the prereg gates, but preconditions to unblind):** held-out next-state NLL ≥ 15% relative better than the untrained trunk and better than uniform and tabulated-marginal baselines; value-head ECE on validation games below a logged threshold. If these fail, R11 does not proceed to evaluation — a negative training result is a reportable outcome, not a reason to substitute surrogates.

### 4.2 (ii) Non-circular test of the Mechanistic Redundancy Law and Prediction P9

**Rule zero: the data-generating path may not contain the law.** Enforced by R11-D0's AST scan: the closed form `α·log(1 + B/(1 + ρ(B−1)))` and its aliases are forbidden tokens in any code path upstream of the fitted statistics.

- **`rho_Gram` as a pre-registered fixed functional on raw banks.** Compute from unsmoothed `bank.edge_keys_bank` written by live `run_search` at each `B ∈ {1, 2, 4, 8, 16, 32, 64, 128}`: embed edge keys with a frozen, preregistered embedding (one-hot or the *trained* trunk's, chosen once and frozen), pool the Gram matrix across depths, and report the **participation ratio** `(Σλ)²/Σλ²` as `rho_Gram`. **No convolution, no kernel widths, no background offsets.** Kernel-bandwidth sensitivity is reported as a secondary analysis with the bandwidth shared across families (any family-specific smoothing constant is a banned construct).
- **KL/NLL from genuinely evaluated predictions.** For every cell, test KL is computed from the trained trunk's predictive distributions on held-out positions via a preregistered exact discrete estimator. No parametric thresholds (`0.112·log b`), no `max(0.004, ·)` floors — if the curve degrades at high B, that is the finding.
- **Falsification protocol with nulls and held-out fit.** Split all `(family, env, B, seed)` cells into fit/test halves (stratified by family and B). Fit `Δc(B)` on the fit half; report held-out R² against preregistered null models: (n1) `Δc ∝ log B`; (n2) saturating `Δc = c_∞(1 − e^{−B/τ})`; (n3) constant. **Pass only if** the mechanistic law beats *every* null by ΔR² ≥ 0.05 on held-out cells, with fitted α, ρ inside pre-declared plausible ranges. Otherwise the law is reported as not-yet-supported — a legitimate Round-11 outcome.
- **P9 without installed correlation.** `rho_IRT` comes *only* from `MiniECIFitter` on genuinely played tournament outcomes (§4.3/§5, R11-D6); `rho_Gram` from the fixed functional above; no quantity may appear in both derivations (checked by D0 taint analysis). Report Pearson and Spearman with a seed-level bootstrap 95% CI. Pre-registered pass: `r_s ≥ 0.5` with CI lower bound > 0; anything else is a refutation of P9 and is published as such.

### 4.3 (iii) Calibrating the conformal / Bregman certificate gate `α_b` under an imperfect trunk

- **Three-way disjoint split:** train (gradients) / calibration (quantiles) / test (reported once), disjoint by *both* seed and env-instance partition (e.g., test sudoku instance sets never seen in either other split), so instance-level leakage cannot inflate coverage.
- **Frozen-trunk calibration.** All nonconformity scores are computed with the *final deployed* trunk frozen; any retraining invalidates the calibration set. Score: pre-registered Bregman divergence between predicted outcome distribution and realized outcome (pinball loss for quantile-valued targets), computed per budget point `b`.
- **Finite-sample quantiles with monotone enforcement.** `α_b =` empirical ⌈(n+1)(1−δ)⌉-th order statistic per budget; then a single isotonic projection across `b` to enforce anytime-nondecreasing certificate quality, with the projection **validated on a second, independent calibration fold** (if isotonicity fails to transfer, report the unprojected gate and flag it).
- **Honest coverage accounting under imperfection.** Report marginal coverage **and** worst-slab conditional coverage (slabs over depth band, env family, difficulty tercile) — an imperfect trunk typically buys marginal coverage by over-widening on hard slabs. Gate bands are pre-registered with tolerance (§5, R11-D5) because exact zeros are not attainable with a learned trunk; the certificates that replace the fake `exp(−δ_c)` curves are precisely these honest intervals, and *their widths as a function of b* become the true anytime-improvement measurement.

---

## 5. Pre-Registration Gate Contributions (`R11-D0` … `R11-D6`)

Global unblinding rules for `PREREG-R11-GPU-NEURAL.md`: gates are frozen before any test-split evaluation; every metric is computed exactly once on the test split; failures are reported verbatim, never patched post-hoc; any code change after unblinding requires a new pre-registration round; no seed-, budget-, or arm-conditional branches may appear in any value-affecting path (D0).

---

### R11-D0 — Zero-Oracle-Leakage Firewall (Static AST + Runtime `EnvBatch.strip_oracle_labels()`)

**Static AST gate.** An AST scanner over `benchmarks/` and `src/` data-generating paths fails on: (1) any reference to the oracle symbol registry `{p_exact, logit_exact, p_true, p_a_given_b_true, true_q, q_star, q_roll_test, p_with_mem, test_edge_deltas, mechanistic_redundancy_delta_c}` outside the separately-compiled evaluation-only module; (2) `torch.logit(...)`/`torch.log(...)` applied to any registry symbol; (3) known leak signatures — mixture coefficients `0.55/0.45`, `0.90/0.10`, transport constants `0.86`, analytic hit-rate `1 − exp(−0.34·…)` — in generation code; (4) any conditional branch on `(seed, m0, b_budget)` that alters a reported value (this rule exists to make constructs like `run_r10_e1_e2.py:1351` structurally impossible).

**Runtime taint gate.** Oracle tensors are wrapped in a taint type that raises on any arithmetic/broadcast with model-space tensors; `EnvBatch.strip_oracle_labels()` returns observations with all oracle-registry fields removed, and a schema assertion verifies no oracle-registry key reaches any model input or bank write.

**Detector self-test (canaries).** Three planted leaks (one symbol leak, one mixture-coefficient leak, one conditional-branch leak) must all be caught by the scanner in CI. **PASS = 0 violations on the real codebase AND 3/3 canaries caught. FAIL = any violation; the run is void and may not be unblinded.**

### R11-D1 — Trained Trunk + Live Universal Tree Machine

**PASS requires all of:** (a) ≥ 20,000 AdamW optimizer steps on validated self-generated transcripts with logged decreasing loss (no surrogate targets; D0-clean); (b) held-out next-state NLL ≥ 15% relative improvement over the untrained trunk and strictly better than uniform and tabulated-marginal baselines; (c) every reported budget statistic produced by live `run_search` on the **trained** trunk with banks dumped raw; (d) fractal topology non-degenerate on the trained trunk (`f_tip > f_root`; fork fraction ∈ [0.01, 0.50]) with zero hardcoded thresholds (AST check); (e) trained trunk is not worse than the untrained trunk on the search-quality proxy (solve rate / test NLL at matched budget). **FAIL if any sub-criterion fails; all curves reported either way.**

### R11-D2 — Active Fractal Prospectus Plans

**PASS requires all of:** (a) ≥ 95% sector coverage via `gram_orthogonal_project` on *trained* trunk embeddings, held-out envs, no hardcoded sector bounds; (b) causal plan-swap action-flip rate ≥ 60% on held-out plans (single-variable `_select_step_var_and_val` swap); (c) ablation: active plans vs. random-plans improves solve rate by ≥ +10 pp on ≥ 2 of 3 families, bootstrap 95% CI excluding 0. **FAIL if coverage or flip thresholds miss, or the ablation CI includes 0 on ≥ 2 families.**

### R11-D3 — Bidirectional Latent Lemma Space & Queries (firewall verified *at train time*)

**PASS requires all of:** (a) `||∂l_V/∂Z^Pi_aux||_∞ == 0.0` (exact autograd, tolerance 0.0) probed on trained weights and on the live training graph ≥ every 500 steps; (b) 300-step policy-gradient RL probe still yields `Δ_belief == 0.0`; (c) even-parity on trained emitter outputs: `max |emit(+Δl_V) − emit(−Δl_V)| ≤ 1e-6`; (d) stable rank of `Z^W_aux` ≥ 50% of initialization value for the entire run (logged per 1k steps, interventions logged if triggered); (e) lemma-pathway ablation on held-out **on-policy** DAgger eval set: ≥ 3% relative NLL improvement from the lemma pathway. **FAIL on any breach; (a)–(b) breaches additionally halt training as firewall incidents.**

### R11-D4 — Present-Time Episodic Memory from Real Episodes

**PASS requires all of:** (a) retrieval exclusively from episodes actually experienced by the trained policy (provenance log verified; no injected beliefs — D0 taint-clean memory path); (b) contradicted-memory blow-up suppression ≥ 99% reproduced on *genuinely contradicted* real memories (not planted perturbations); (c) memory vs. no-memory ablation ≥ +5 pp success on designated transfer tasks, CI excluding 0; (d) surprise-gate ablation degrades or fails to improve (memory-without-gate no better than with-gate), demonstrating the gate is load-bearing. **FAIL if any threshold misses; a null transfer effect is a reportable negative result, not grounds for surrogate substitution.**

### R11-D5 — Anytime Monotonicity & Self-Consistency (honest certificates)

**PASS requires all of:** (a) from raw test curves (no smoothing, no floors): pairwise monotonicity rate of test NLL/KL in budget `b` ≥ 0.90 across seeds, with no *mean* increase exceeding ε_mon = 0.002 nats between consecutive budgets; (b) conformal certificates from §4.3: marginal coverage ∈ [1−δ−0.02, 1−δ+0.05] **and** worst-slab coverage ≥ 1−δ−0.05 (slabs: depth band × family × difficulty tercile); (c) certificate width non-increasing in `b` for ≥ 90% of seed pairs; (d) all per-cell violations reported in a published table. **FAIL if any band is missed; per-cell violations above tolerance are findings, not noise to be smoothed.**

### R11-D6 — Multi-Game Elo League & Mini-ECI IRT on *Played* Games; Non-Circular P9

**PASS requires all of:** (a) tournaments are interactive turn-by-turn games between budget-matched trained policies (≥ 200 games per round-robin pairing across seeds); zero Bernoulli-sampled `btd_probabilities` outcomes; raw outcome matrix published; (b) estimator predictive validity: held-out game-outcome log-likelihood under the fitted BTD/mElo beats a random-pairing baseline (the synthetic-recovery unit test remains a precondition, not evidence); (c) **P9 test per §4.2:** Spearman `r_s ≥ 0.5` between `rho_IRT` (from MiniECI on played outcomes) and `rho_Gram` (fixed raw-bank functional), seed-bootstrap 95% CI lower bound > 0 — **if this fails, P9 is REFUTED and the refutation is the published result**; (d) MiniECI bifactor beats 1-factor and independence baselines on held-out item imputation error. **FAIL if outcomes are simulated, or (b)/(d) baselines are not beaten, in addition to numeric threshold misses.**

---

**Closing note for the panel.** The Round-2 audit did not find a weak system; it found a *mis-labeled* one. The operator mathematics in `src/` is real and worth the Round-11 investment; the runner layer must be rebuilt, not patched, and `R11-D0` must land first so that the distinction this round established — verified mathematics versus manufactured empirics — can never silently collapse again.
