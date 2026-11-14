You are a co-prover (theoretical computer science and probability) working read-only. Produce an independent
attempt at the questions below. Be rigorous: give precise statements, full proofs where you can, and label every
claim PROVEN, PROVEN UNDER <named hypothesis>, HEURISTIC, or CONJECTURE. Flag every literature citation you are not
certain of as "unverified". Prefer a short correct statement to a long shaky one. Write plain text/markdown with
ASCII math. Do not edit files.

## Setting
Random critical He-ReLU networks: width n, depth L, W_l iid N(0, 2/n) (n x n), no biases, x ~ N(0, I_n),
h_0 = x, h_l = relu(W_l h_{l-1}) for l < L, output a_L = relu(W_L h_{L-1}) in R^n. (A noisy variant adds
sigma r_l xi_l after each hidden ReLU, xi_l ~ N(0, I) fresh, sigma ~ 1e-3; treat sigma = 0 unless stated.)
Estimand: mu(W) = E_x a_L in R^n. Metric (ARC's random-MLP instance of the "matching sampling principle"):
average-case over W, E_W |A(W) - mu(W)|^2 compared with E_W tr Var_x a_L / N, the error of N-sample Monte Carlo
(MC), at equal cost (arithmetic operations; one forward pass costs about 2 n^2 L). ARC's setting is n = 256,
L = 32; smaller nets (n = 8-32, L up to 64) are used for experiments. The interesting regime is n -> infinity with
L = alpha n, budgets polynomial in n.

Known (from earlier work in this project; do not redo, but you may use):
- Black box: no query algorithm beats MC by more than about 4x on explicit ReLU families; deterministic query
  algorithms lose by a factor of order the number of queries (worst case).
- Worst case white box: Boolean circuits reduce to Gaussian means of ReLU nets of the same size; so a
  subexponential deterministic estimator for all ReLU nets would give NEXP not in P/poly (IKW 2002), and under
  #NSETH deterministic estimators need time E^{2-o(1)} on E-edge nets (adapting Chen-Tell 2021).
- MC reaches inverse-polynomial precision in randomized polynomial time, so any hardness must be fine-grained or
  about a restricted class.
- Nonuniformly, determinism is free on average over W: for each size some hard-coded input set matches MC on
  average over W (averaging argument). If the joint law of the net is invariant under rotating the first layer,
  a fixed design has exactly the average error of its random rotation.
- Lower bounds already exist for restricted classes that pass a bounded summary across a cut (one-way linear
  summaries: the eps-rank of the cut operator), and for layer-local closures (several necessary conditions:
  the output scale is a product of independent per-layer factors with log-variance about 5L/n and never contracts;
  fixed-order expansions in the gain factors lose a constant fraction at L proportional to n).
- At infinite width (fixed depth) the net is a Gaussian process on the sphere with the iterated arc-cosine kernel;
  its spherical-harmonic (Gegenbauer) spectrum decays like l^{-(n+2)} for ReLU.

## The class this lane studies
Unrestricted deterministic algorithms with full access to all weights that compute jointly over the whole net
without a bounded interface: adaptive case splitting over gate patterns (branch-and-bound with value bounds from
the actual downstream weights), global region refinement, derandomisation (fixed low-discrepancy point sets,
PRG-based estimators), algebraic/analytic methods using weight structure, "heuristic explanations" that use the
whole net.

## Questions (answer each; number your answers)
1. Is "deterministic" a meaningful restriction on average over continuous random weights, uniformly (one explicit
   algorithm for all sizes)? Prove or refute: there is an explicit uniform deterministic algorithm whose average
   error over W matches MC with N samples at cost O(N forward passes) plus lower-order terms. If it holds, what
   restriction would make "mechanistic/deductive" meaningful and still contain fixed designs and moment closures?
2. Fixed point sets (deterministic QMC, spherical designs). Characterise the average-over-W error of a fixed
   (non-adaptive, linear) rule exactly. What bounds the best possible gain over MC, as a function of n, L and N?
   Does anything change for adaptive point-evaluation algorithms (information-based complexity, average case)?
3. Restricted-class bounds. For each of (a) estimators that are degree-D polynomials in W with instance-independent
   coefficients, (b) "stable" algorithms in the overlap-gap sense, (c) statistical-query-style models: state what
   can be proven for this problem and what cannot, and why. If a quantity can be measured that bounds class (a),
   give it (e.g. via noise stability Cov(mu(W), mu(rho W + sqrt(1-rho^2) W'))).
4. Connections to known average-case hard problems (spin-glass free energies at low temperature, random-circuit
   output probabilities, permanents, approximate counting/volume): give the precise correspondence, if any, to the
   mean problem with ARC's SD-normalised error, and to the tail problem P(a_L,j > t) for t with tiny probability.
5. Barriers. What would an unconditional lower bound "no algorithm beats MC by more than a constant at L = alpha n
   on average" imply in complexity terms? Do IKW-type (derandomisation <-> circuit lower bounds) or natural-proofs
   barriers apply here, and how exactly?
6. Branch-and-bound / partition methods that are exact on cells inside one linear region (for sigma = 0 the net is
   positively homogeneous and piecewise linear on cones): how should their cost to reach error eps scale with n and
   L? Give whatever rigorous statements you can (e.g. region counts met by the sphere for bias-free random nets;
   rates for adaptive cubature on piecewise-linear integrands) and heuristics otherwise.
7. Precise conjectures that would imply hardness (for the mean, and separately for tails), each with status,
   evidence, and the cheapest refutation test. Mark which one is the bottleneck.
8. "What it would take": the strongest necessary conditions you can prove for any whole-network estimator to beat
   MC by a growing factor at L = alpha n.

Context files you may read (read-only): /home/agent/workspaces/mats/outbox/hardness-mean-estimation-2026-10-29.md,
/home/agent/workspaces/mats/outbox/cut-rank-lower-bound-2026-10-30.md,
/home/agent/workspaces/mats/outbox/narrow-deep-m1-2026-10-30.md.
