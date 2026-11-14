You are a co-prover and red team (probability, random dynamical systems, random matrices) working read-only.
Produce an independent attempt at the questions below. Be rigorous: precise statements, full proofs where you can,
and label every claim PROVEN, PROVEN UNDER <named hypothesis>, HEURISTIC, or CONJECTURE. Flag every literature
citation you are not certain of as "unverified". Prefer a short correct statement to a long shaky one. Where a
quantity can be computed in closed form (constants, exponents), compute it. Plain markdown, ASCII math. Do not edit
files. Do not read work/stationary/ (it is the other prover's work in progress); you may read the outbox reports
named below for context.

## Setting
Random critical He-ReLU layers, width n, no biases: W_l iid with iid N(0, 2/n) entries (n x n), relu(z) = max(z, 0)
coordinatewise. Two noise conventions.

- Model S (absolute noise, bi-infinite stack): h_l = relu(W_l h_{l-1}) + sigma xi_l for every l in Z, xi_l iid
  N(0, I_n), independent of the weights. "Fixed weights" means conditioning on the whole sequence (W_l). The
  sample measure mu_l^W is the law of h_l given the weights (over the noise), defined as a pullback limit if it
  exists.
- Model R (relative noise; the convention of earlier lanes): input x ~ N(0, I_n) at layer 0, layers l = 1..L,
  h_l = relu(W_l h_{l-1}) + s_l xi_l with s_l = sigma r_l, r_l = RMS over the noisy ensemble of the coordinates of
  relu(W_l h_{l-1}) (a deterministic number per layer given the weights, computed from a large pilot). sigma in
  [1e-3, 1e-1].
- Output and estimands: a_L = relu(W_L h_{L-1}) (no noise after the last layer). Targets: E[a_L | weights] (a vector)
  and a tail probability P(|a_L| > t | weights), at a deep layer L, scored per net by
  nMSE = |estimate - truth|^2 / tr Var(a_L | weights). Cost unit: one matrix-vector product, 2 n^2 flops.

## A heuristic model to test (the coordinator's; test it, do not assume it)
- Pullback view: looking backward from a deep layer, the state becomes a function of recent noise and recent
  weights only: a stationary sample measure, or random attractor, as in coupling from the past.
- Linearised picture: the state is a sum of past noise pushed through Jacobians. Direction k (Lyapunov gap about
  (k-1)/n) remembers noise for about n/(2(k-1)) layers. The radius, or log-gain, remembers longest (about
  1/(2|lambda_1|), roughly n layers); it does a random walk down to the noise floor. So the cloud around the
  attractor is near-Gaussian, with variance along k of about sigma^2/(2|lambda_k|).
- Bits: units whose pre-activation is within about sigma of zero have noise-decided gates. These bits are born at
  about n sqrt(2/pi) sigma/tau per layer (tau = typical pre-activation size) and die at the Lyapunov rates, giving a
  stationary live-bit budget, the non-Gaussian part. Prediction: estimation is easy when the bits are many and
  additive (CLT) or few (enumeration), and hard in between, e.g. when tau ~ sigma near dead layers.
- Input bits: below the memory time inputs are not yet merged (ARC's 256 x 32 sits there); the live bits then
  include the input's own unmerged gate history, with no birth term, decaying at the same rates.

## Questions
1. Existence and uniqueness of the stationary sample measure in Model S, for a.e. fixed weight sequence: state and
   prove (or cite precisely) a pullback convergence theorem, with a rate, and the class of initial laws for which
   uniqueness holds. Is there also a random point attractor when the noise is fixed too (synchronisation)? What
   changes in Model R? Is sigma a genuine parameter in either model?
2. The stationary scale in Model S: the law of |h_l|^2 / (n sigma^2) under the stationary measure (typical size as
   a function of n, its tail, its fluctuation over l and over weights). From it, the effective per-layer relative
   noise and the ratio sigma / tau.
3. Lyapunov spectrum of the tangent cocycle J_l = D_l W_l (D_l the gate mask of the trajectory) along stationary
   trajectories in Model S and along noiseless trajectories in Model R: exponents, gaps, finite-n corrections. The
   covariance of the stationary cloud: what does the Lyapunov equation predict, and when is it right?
4. Gate births: the expected number of units per layer whose gate is decided by the most recent layer's noise
   (define precisely), as a function of the noise-to-signal ratio of pre-activations; evaluate it in both models.
5. Age profile: how much does h_L depend on the noise injected j layers back, direction by direction? Give the
   profile in j and k, and the total effective number of noise variables that matter.
6. The live-bit budget, and where the near-Gaussian picture fails (which regime of n, sigma, depth).
7. Estimators of the targets at deep L as compute -> infinity at fixed n: give the MSE-vs-compute exponent and how
   the constant scales with n for: MC with burn-in; multilevel over burn-in depth; coupling from the past; nested
   samplers that share the deep past; RQMC on the noise; control variates from a small-noise linearisation; stratified
   or importance sampling of near-boundary units; deterministic small-noise expansions with exact near-boundary
   handling; hybrids. Which could beat MC's exponent, and which could have a constant that improves with n? Earlier
   result to build on (outbox/unrestricted-estimators-2026-10-31.md, section 3.7): for the noiseless net, a
   multilevel sampler that shares the late layers through one reference trajectory gains Omega(n/log^2 n) over MC at
   depth proportional to width. Does any analogue exist in the stationary regime?
8. Model R at depth L >> n: which parts of the state are stationary and which keep the input's memory (radius,
   log-gain)? Give the time scales.
9. Input bits: in Model S with an input of radius about sqrt(n) at layer 0, the depth at which the input is
   forgotten, as a function of n and sigma, and how input bits and noise bits combine below it.

Context you may read: outbox/unrestricted-estimators-2026-10-31.md, outbox/narrow-deep-2026-10-29.md,
outbox/narrow-deep-m1-2026-10-30.md, outbox/lyapunov-dimension-2026-10-29.md (all under
/home/agent/workspaces/mats/outbox/). End with a ledger: each claim, its label, and the cheapest numerical check.
