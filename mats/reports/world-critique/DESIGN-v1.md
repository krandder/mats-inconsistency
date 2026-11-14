# World portfolio and crux tests, v1 (after 3 critique rounds; opus-mats, 2026-10-28 ~02:20Z)

Critics: GPT-6 (Codex), Gemini (3.8 Flash in r1, 4 Argon in r2-r3), Claude Fable 5.1 (r1-r2), Kimi (r3).
Rounds: r1 independent, r2 cross-critique + portfolio + cruxes, r3 attack on SYNTHESIS-R2.md. All files here.

## Global rules (from r2-r3)
- Ignorable policies only: depend on visible keys, recorded observations, independent randomness. Unit-test that
  the oracle posterior is identical under different policies producing the same data.
- Metric: KL(p_Bayes || p_model) and excess log loss, with seed-level CIs (>= 10 seeds for crux claims; screens
  may use fewer). Report gap-to-Bayes of every reference, not only of the model.
- Anchors are simulated, never asserted (Kimi): e.g. the Bayes contamination slope is computed per configuration.
- Prefer paired, posterior-preserving or posterior-flipping interventions over regression slopes (Codex).
- Transversal scaling crux (Kimi): gap-to-Bayes vs N on log-spaced N = 64..4096 per world; "scales" iff flat or
  falling; refuted if it grows.

## W1 Similarity world (3-level hierarchical junta) — first
Items x in {0,1}^16; hidden R, |R|=2 (120 hypotheses); cell g ~ N(0,1); item v_x ~ N(g(x_R), tau^2); fresh
observation y ~ N(v_x, sigma^2). Copies only with observed provenance (event id) — hidden-ancestry relays dropped
(Bell-number oracle; Gemini/Codex). Exact Bayes O(120 N). tau>0: repeats of one item saturate at tau^2 (item dedup),
distinct items in a cell average tau^2 down (cell aggregation). Kimi's fatal flaw (120-hypothesis menu is
memorisable) is accepted as a limitation: W1 tests multiplicity/length behaviour, not open-ended similarity.
C1 (corrected, Gemini + Kimi): fixed clean balanced context N_clean = 48; append M in {0,16,64,512,4048} repeats of
one dominant item at Hamming distance 8 from the query and in a different cell; matched-length balanced control;
R-known oracle model as baseline. Simulated Bayes slope as anchor. Refuted (no policy-invariant extrapolation) if the
model's contamination exceeds Bayes by > 0.10 or grows > 0.03 per doubling of M at any M where the matched-length
control keeps KL < 0.05. Plus attention-mass vs multiplicity probe.

## W2 Market world (sources, biases, heterogeneous quotes, provenance) — second
K items v_k ~ N(0,1); S sources with continuous bias b_s ~ N(0,tau^2) and noise sigma_s; quote types SPOT (v_k),
DIFF (v_k - v_j), BASKET (mean of a subset); source identities fresh per episode (Kimi); unknown discrete noise level
for at most 8 sources, the rest known (Gemini/Codex: keeps exact Bayes tractable as S grows). Provenance/relay
quotes (Codex "missing" world, Kimi "dedup x aggregation"): a source may re-quote another source's measurement
(observed provenance) possibly through its own bias — Bayes uses the shared covariance. Exact Bayes: mixture over
<= 2^8 noise hypotheses of sparse linear-Gaussian solves.
C4 (corrected, Gemini + Codex): frequency-matched calibration — query item gets one quote from s_A and one from
s_B, both seen equally often elsewhere; elsewhere s_A is tight, s_B noisy/biased; pre-registered |Delta_Bayes| >= 0.3
nats in the test regime. Refuted (no in-context source inference) unless the model closes >= 50% of the gap between
a source-frequency-only reference and Bayes on >= 80% of episodes.
C7 (Gemini): diamond (one root quote reaching the query by two DIFF paths) vs two-anchor graph with matched path
lengths; Bayes discounts the shared root; naive message passing double-counts.
C3 (corrected, Codex + Kimi + Gemini): bounded state measured in bits/floats; quantised state; queries uniform over
items; compare gap-to-Bayes of both full attention and bottleneck; native bit-task first (shape should track
h2^-1(1 - B/K)); then W2 retroactive-calibration test (calibration evidence arrives after the item quotes).

## W3 Chain world (discrete messages) — third, cheap
Latent bits; unary noisy reports, XOR (and majority) relations; M >= 8 disjoint chains, random node ids and edge
orientation (Gemini); information matched across lengths (Codex). Linear-Gaussian chains are shallow-representable
(Kimi) and kept only for the C6 demo.
C2 (corrected, Kimi + Gemini + Codex): untied depth L in {2,4,6} vs weight-tied looped net at MATCHED rounds; train
l <= 4, test l <= 8; looped nets run extra test-time iterations. Signature of round-limited reasoning: error improves
one rung per added round and looped-T matches fixed-T at equal T. Refuted if depth-2 reaches KL < 0.05 at l = 6 with
the distractor chains present.

## W4 Tree / transposition — gated
Only if W1-W3 show template-menu shortcuts. One generative model first (tree oracle, or cutset conditioning for a
DAG; recursion Z = 1/2 L + 1/2 Z_l Z_r is wrong on DAGs).

## C5 Optimisation vs expressivity (Kimi + Gemini + Codex)
Two arms at matched compute: query-only NLL vs all-token next-observation NLL (no latent labels) [+ consistency
regulariser as a third arm]; plateau defined against the computed prior-level loss; stall diagnostics (gradient size,
representation saturation). Supported iff the denser/consistency arm escapes and query-only does not; refuted if
query-only escapes. (Overlaps astra's plateau study in the tiny world.)

## C6 Inconsistency minimisation vs Bayes (analytic first)
Gemini: point-estimate minimisation picks one sharp low-energy hypothesis while Bayes marginalises the volume of many
slightly worse ones that agree. Kimi: in underdetermined linear-Gaussian cases only the prior picks among equal-
residual optima (nullspace). Construct exact counterexamples in W1/W2; no training needed.

## Order for the next 12 h
E1 W1 pilot + C1 (in progress). E2 W2 build + C4/C7 (+ C3 later). E3 W3-discrete build + C2. C6 analytic note.
Then: mechanism study on the world where the model is closest to Bayes yet shows non-trivial in-context behaviour.
