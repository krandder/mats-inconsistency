• Model: Kimi (Kimi Code CLI, Moonshot family). Round-3 adversarial review, no files touched.

  **1. Cruxes**

  **C1 — Not decisive.** (a) "Bayes slope 0" is asserted, not derived: at N=4096 under a concentrated policy the Bayes posterior is dominated by the dominant group, so it *is* correlated with the dominant-group mean; Bayes slope is generally positive. The anchor must be simulated per configuration. (b) The N=64 clause is vacuous — at N=64 the test policy sits inside the training mixture, so contamination is small by construction; it carries no evidence about invariance. (c) The 0.5 threshold is absolute and unanchored. Corrected: simulate slope_Bayes and slope_prior; refuted if the model slope's 95% CI lies above slope_Bayes + 0.5·(slope_prior − slope_Bayes) over ≥1000 episodes. The informative signal is the N-dependence of the slope (grows with N ⇒ in-context contamination, not memorized policy response), plus an attention-mass-vs-multiplicity probe so "contamination" isn't defined solely by the regression.

  **C2 — Not decisive.** The linear-Gaussian chain posterior is a linear map from observations to query; a shallow net can represent it, so depth-2 matching Bayes refutes nothing about iterative reasoning — it proves this world is shallow-representable. It also confounds depth with rounds: looped 2×8 gets 16 mixing rounds vs fixed-6's 6; weight-tying is isolated only at matched rounds. Corrected: run on the discrete version (XOR/majority), where l-round message passing is provably necessary (transformer depth ↔ distributed-computation rounds). Decisive signature: error improves exactly one chain-rung per added layer, and looped-2×T matches fixed-T at equal T. Metric: KL(p_Bayes‖p_model), not raw nats; 8/10 seeds → mean with CI over ≥30 seeds.

  **C3 — Not decisive.** "k=32 slots" isn't measured in bits, so the impossibility bound cannot set the threshold — 0.03 nats is arbitrary. Worse, the criterion is gap-to-full-attention: if full attention itself fails to reach Bayes at K=512 (optimization), "within 0.03 of full attention" certifies two failures. Corrected: state capacity in bits/dim; report gap-to-Bayes for *both*; decisive test is the shape — bottleneck error vs K at fixed B should track the h2⁻¹(1−B/K) curve; refuted if the empirical curve stays flat where theory predicts growth.

  **C4 — Conditionally decisive.** Needs a pre-registered effect size: if the dominant source's bias is already pinned by thousands of context observations, 8 more readings move Bayes negligibly and "closer to" is noise. Require simulated |Δ_Bayes| ≥ ~0.3 nats in the test regime first. "Closer" needs a metric (KL to each reference's predictive), and the prior-level reference must be shown distinct from Bayes there (≥0.1 nats), else the classification is vacuous.

  **C5 — Direction missing.** As written it only states the supporting outcome; no refutation condition. "Prior-level plateau" is undefined until the prior-level reference loss is computed; "10× oracle steps" is seed-dependent; warm-start conflates initialization with landscape — the aux-loss arm is the only clean one. Corrected: two arms (NLL-only vs NLL+consistency), matched compute; plateau = smoothed-NLL slope < ε over a window, within CI of the precomputed prior-level loss; supported iff aux escapes and NLL-only doesn't; refuted if NLL-only escapes. Add a stall diagnostic so the disjunction is empirical: small gradients + saturated representations ⇒ optimization/landscape; small gradients + unsaturated ⇒ implicit bias; large gradients ⇒ training bug.

  **C6 — Make it analytic.** In the noisy/underdetermined regime, equal-residual optima differ by a nullspace vector of the incidence matrix; the Gaussian likelihood already penalizes residual, so only the prior picks between them. Exhibit via SVD: nullity ≥ 1 and prior odds ≥ e² between extreme points. "0.2 in query probability" is unnormalized — use posterior mass ratio. And check first that the regime isn't determined (point-mass Bayes makes the crux inapplicable).

  **2. Worlds**

  **W1 — Fatal:** |R|=2 gives 120 cells; the posterior map is a memorizable menu, so "in-context dedup" can be prior-learned lookup, and aggregation is absent (one measurement stream). Fix: episode-specific correlated item effects (per-episode GP-lengthscale b_x) so fresh measurements carry irreducible in-context information, plus relays through biased sources to force dedup-during-aggregation.

  **W2 — Fatal:** dense SPOT coverage makes the overlap graph dense; biases are identified by differencing and source-id→bias becomes a training-time lookup table. Nothing is in-context; C4 has no bite. Fix: source identities fresh per episode / held out at test, S growing with N — bias inference must happen in context.

  **W3 — Fatal:** linear-Gaussian is shallow-representable and cannot host the discrete-message motive. Fix: keep it only for the C6 nullspace demo; put C2 on the discrete chain.

  **W4 — Fatal:** training on recursion-generated data certifies imitation of the recursion, not discovery of transposition sharing; a model can match Bayes locally while never reusing subtree computation. Fix: contexts with exponentially many tree nodes but few distinct subtrees; success = matching DAG-Bayes at N where tree-Bayes is infeasible *plus* representational evidence of subtree-invariance (repeated-context probes), not just low NLL.

  **3. Missing**

  - **Dedup×aggregation world.** Markets double-count shared evidence. Crux: relay of an existing measurement through a biased source vs an independent fresh one — Bayes updates less for the relay (covariance-aware effective-sample-size division). This is the most market-like behavior in the goal and is only implicit today.
  - **Scaling crux transversal to all worlds.** Gap-to-Bayes vs N on log-spaced 64…4096, fitted as a power law. "Scales to more observations" holds iff the gap is flat/falling; refuted if it grows — rising contamination slope with N is the leading candidate failure.
  - **Early cheap reasoning probe.** The build order could burn everything on W1/W2 and never touch the reasoning/looped-net motive; W3-discrete at small l is cheap and decides it.

  **4. Ranking (12 h, ~$100 L4 ≈ 150–250 GPU-h; agents cheap)**

  1. **W1-fixed pilot** — nearly built, resolves the tau dispute and C1 at near-zero marginal cost, but only after the W1 fix, else it answers a memorization question nobody asked.
  2. **W2 minimal** (K=8, S=16, 3 noise levels, held-out sources) — runs C4 and the C3 state sweep in one training stack.
  3. **W3-discrete** — the depth↔rounds signature; decides the looped-architecture motive before any W4 spend.
  4. **W4** — stays gated on evidence of template-menu shortcuts.

  First three experiments: **E1** — both tau arms (with episode-correlated b_x), N-sweep 64→4096, slope vs simulated Bayes/prior anchors, attention-mass probe. **E2** — W2 minimal as above, pre-registered effect sizes, gap-to-Bayes curves for full vs 32-bit-state models across K. **E3** — depth × looped at matched rounds on XOR chains, l up to 8, checking the one-rung-per-layer signature.

