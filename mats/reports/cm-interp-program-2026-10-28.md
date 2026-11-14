# Noisy-channel interpretability program (agreed with Kelvin, 2026-10-28 ~16:30Z)

## The bar (Kelvin, 2026-10-29 ~14:30Z)
"So let us make it very clear here: we're looking for a major advance. This is what we're looking for. We can find
minor advances as small steps, and get intermediate results that could be real papers, but only as we move towards a
major advance."
Major targets (a result counts if it reaches one; every episode states which one it moves toward and how):
- M1. ARC's open problems: a sample-free estimator that matches or beats sampling on deep random MLPs at depths that
  grow with width (e.g. 256 x 32), or mechanistically estimates AlgZoo M_{16,10}'s accuracy as well as sampling.
  Lanes: boundary-aware families (work/boundary-family), random-net phases, AlgZoo.
- M2. A general account of how transformers approximate Bayes in context and why they deviate: holds across sizes,
  tasks and pretrained LLMs, explains the inductive bias, predicts the departure from the architecture.
  Lane: phase B2 (big-miss departure as the lead).
- M3. Certified rare-event bounds on trained networks tighter than sampling's at equal compute. No active lane yet;
  candidate once boundary-exact families exist.
Minor (notes, not goals): improving one step of someone else's pipeline, narrow-regime wins, beating weak baselines.

Kelvin: "Yes, I now agree fully. Thank you. We now have a full program." Origin: Kelvin's conversation with Samuel Liew
and the discussion in session 52cc519b (15:00-16:30Z).

## Setup
1. **Noise makes every layer a Gaussian channel.** Train with i.i.d. Gaussian noise N(0, sigma^2) on every activation,
   with LayerNorm (power constraint). Each layer carries at most C_l = (d/2) log2(1 + P/sigma^2) bits;
   I(X;H_L) <= ... <= I(X;H_l) <= C_l. Densities are full-support and sigma-smooth; sigma is the natural precision.
2. **We describe the network by a chain of tractable distributions q_l, never by samples of p.**
   - q_0 covers the input distribution (in our worlds: the exact PFN prior).
   - Each layer is a Markov kernel K_l (weights + noise). K q_l(h) = E_{h'~q_l}[N(h; f(h'), sigma^2 I)] is computable
     from q_l and the weights (for sparse codes through piecewise-linear layers: k-dimensional integrals over linear
     regions).
   - q_{l+1} is a tractable family member chosen to cover K q_l.
   - Coverage of the real p is inherited: D_inf(p_{l+1} || q_{l+1}) <= D_inf(p_l || q_l) + D_inf(K q_l || q_{l+1})
     (data processing + additivity of max-divergence). A defensive wide component of weight delta bounds each step's
     coverage cost by log2(1/delta).
   - The real structure of p enters only as waste: a q that cannot represent some structure is wider; the description
     must still predict the next layer correctly on all of q's support (robust to out-of-distribution internal states).
3. **Scorecard (deductive, no sampling):**
   - coverage cost per step c_l = D_inf(K q_l || q_{l+1}) (kept small by construction); total epsilon = c_0 + sum c_l;
   - waste per step w_l = KL(q_{l+1} || K q_l) (bits by which the tractable description is broader than the exact
     pushforward); judge codes by total waste at the output, since waste can compound downstream;
   - compute per step (pushforward + refit) -- the axis that matters; model bits are secondary;
   - certified output statements: p_L(A) <= 2^epsilon q_L(A) for any event A; tightness set by total waste.
   - Win condition vs sampling: at equal compute, a tighter certified bound on rare-event probability or expected loss
     than sampling's (sampling cannot certify below ~1/compute).
4. **Families.** Gaussian (exact under the affine part); product grids over a few named directions (K^D cells);
   activation-pattern mixtures (wasteful: pattern entropy includes noise-destroyed distinctions); k-sparse codes
   h = D a + noise with bits ~ H(support) + sum_active 1/2 log2(1 + Var(a_i)/sigma^2), layer map by sparse transcoder;
   neuron code (pre-activations z = W1 LN(h): elementwise nonlinearity, interaction order 1) with gate groups found
   from the weights. Code-invariant hard object for piecewise-linear layers: the joint distribution of gates (entropy
   <= min(m, C)). Counting bound for needed dictionary size: F ~ (k! 2^I)^(1/k) / L with I = I(X;H_l).
5. **Limits to look for.** Waste growth with depth where layers expand q's support (chaotic regime); dense gate
   dependencies that no tractable family covers without large waste at polynomial compute. Random networks: waste
   small (near-Gaussian pre-activations, near-independent gates); waste is capped at C - I per layer by the
   "know-nothing" q (uniform on the LayerNorm sphere at noise precision).

## Execution
Phase A (workflow, running): toolkit, validation, round 1 on tiny PFN (sample-based change of measure).
Phase B (agent, running): noise-trained tiny PFNs, channel capacities and mutual information, families compared.
Phase B now re-targeted to the scorecard above: q_0 = exact PFN prior, deductive propagation per family, coverage
cost / waste / compute per layer, certified output bounds vs sampling at equal compute.

## Structure from the question (discussion 2026-10-28 late; proposed, untested)
Kelvin on the backward procedure: "This is again very beautiful and I am very happy that this seems to be possibe";
he then asked whether to go forward from the question's input distribution instead, and whether the two meet.
- A question = an input distribution D_Q plus an output event A (or function). Forward from D_Q drops what never
  happens; backward from A drops what does not matter; the explanation covers what both happens and matters.
- Meeting identity at every layer m: Pr[A] = E_{h~p_m}[v_m(h)], v_m(h) = Pr[A | h_m = h] (forward-backward on the
  layer chain). With q_m, psi_m: Pr[A] - <q_m,psi_m> = <p_m - q_m, v_m> + <q_m, v_m - psi_m>, so forward error counts
  only where v is large and backward error only where q has mass. The meeting object is the path distribution
  conditioned on A (Doob h-transform): Markov, transitions K_l v_l / v_{l-1}, marginals proportional to p_l v_l.
- Certificate at a cut m: Pr[A] <= 2^{eps_m} <q_m, psi_m>, forward coverage up to m, psi_m >= v_m pulled back from
  psi_L = 1_A via psi_{l-1} >= K_l psi_l. m = L is the current any-A certificate; m = 0 with q_0 = exact prior has no
  coverage factor.
- Each alone over-describes: forward-only = all input-driven variation (phase R width wall; phase B ~2^C cells);
  backward-only = psi right even on unreachable states.
- Evidence (phase R, provisional, splitting reference): psi_l from the forward Gaussian chain's own conditional of
  the output given h_l made twisted SMC 60-1,400x cheaper than splitting where it works (to 2^-30..2^-40 critical
  tanh, 2^-20 chaotic, 2^-10 ReLU); a black-box iterated-APF psi fails beyond 2^-15.
- Procedure: per-layer coordinates (sparse basis) -> node weights = expectation over D_Q of backward sensitivity
  (propagated-noise metric) -> build backward from the question (keep weighted nodes, smallest parent sets) -> local
  moves (drop to noise, prune edge, lateral merge, skip-merge) scored by change in question divergence vs description
  cost -> run forward, measure the question, add back where the residual is large. Expectations over D_Q also delete
  edges (constant parents, never-open gates: context-specific independence), so each question gets a sparser DAG.
  The tighter constraint leads; otherwise alternate passes (EP-like). The per-layer chain is a reference baseline only.
- First test: tiny PFN; the DAG should rediscover the three source branches joining at the query (plus the
  cross-source edges in those seeds) without being told; score = deviation-prediction R^2 and scenario hits.

## Execution update (2026-10-29)
Kelvin, ~01:30Z: "Alright. I'll leave it to you to decide the right experiments." (after phase B Update 9: single-site
surrogates at <= 25% of a site's parameters predict the network's deviation from Bayes worse than "network = Bayes";
those fits ignored the question.)
Phase B queue: E1 question-weighted site fits (J^T J metric) + end-to-end fit, 64 noise draws -> E2 whole-network
structure from the question (every site replaced; Pareto of cost vs deviation R^2 / scenario hits; does it rediscover
the per-source branches; attention via question-weighted fits and named-quantity patterns) -> E3 descriptions relative
to Bayes (exact Bayes quantities as free inputs; cost counted beyond them) -> E4 compute payoff on a distribution-level
question vs sampling the real net -> event-specific backward certificate.
Parallel lanes: ARC deep noisy MLPs (work/arc-noisy; sigma 0.001-0.1, low end primary; GPU cap $50), AlgZoo noisy RNNs
(work/algzoo-noisy; noisy loss must match the original; wider noisy RNN arm), phase R fixes after Codex review.
World FA parked (binding is the blocker).
