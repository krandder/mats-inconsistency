# Can mean estimates on deep, wide networks provably beat random sampling? Hardness results (29 October 2026)

Answer to Kelvin's ask of 29 October: "Can we get a subagent do a deep math/TCS attempt at proving some hardness bounds
on making mean estimates on outputs on the limit (as deep/wide as needed) better than random sampling?" (bead
pool-z0ra9m). Work directory `work/hardness/`: proofs (`draft.md`), numerical checks (`numerics/`, logs in
`numerics/logs/`), verified sources (`notes/sources.md`, paper texts in `papers/`), adversarial reviews (`review/`:
Gemini 4 Argon on Sections 1-6, two GPT-6 audits at xhigh effort, one GPT-6 re-check of the fixes). Labels: PROVEN (proof
below), PROVEN UNDER X (X a named hypothesis), MEASURED, HEURISTIC, CONJECTURE.

## Summary for Kelvin

**Short answer.**
- If an algorithm may only run the network on inputs it chooses, sampling is optimal up to a constant factor of about 4
  (proven), even for networks with a handful of units.
- If it may read the weights, then in the worst case a *deterministic* method that matches sampling on every network
  does not exist under a standard fine-grained hypothesis (#NSETH): it must lose a factor of about the network size.
  And proving that even a slow (subexponential) deterministic method exists would prove a major open circuit lower
  bound. Both follow from a reduction I prove: any Boolean circuit becomes a Gaussian-input ReLU network of the same
  size whose mean is the circuit's acceptance probability (with or without biases, with small noise).
- For random He-ReLU networks with depth proportional to width (our M1 setting), no general lower bound is possible
  with current tools: on average over random weights, a hard-coded set of inputs is already a deterministic estimator
  that matches sampling, so an impossibility needs a formal definition of "mechanistic". What I proved instead are
  exact facts and restricted-class results that say what any layer-by-layer description must get right.
- New measurement: ARC's cheapest closure (K = 2) stays within a factor of about 3 of plain Monte Carlo, either way, at
  its own budget, along depth = width/8, /4 and /2 up to width 1024, but its advantage shrinks slowly with width (from
  2-8 at widths 32-128 to about 0.7-1.1 at width 1024). The next closure in ARC's hierarchy (factorized K = 3, 14-251 times the cost) falls behind plain Monte Carlo as width grows: gain 2.16 at width 32, 0.80 at width 64, 0.38 at width 128, 0.08 at width 256 (3 seeds) at depth = width/8, and a loss of 27-247x at depth = width/4.
- None of this refutes ARC's matching sampling principle (MSP). It shows where its content lies: in the informal word
  "mechanistic" and in the average case.

**Proven (unconditional).**
- Black box. On explicit ReLU families no query algorithm beats Monte Carlo by more than a factor about 4 (Theorem 2.1,
  Corollary 2.2), and a deterministic query algorithm loses by a factor of order the number of queries (Theorem 2.3).
- No coarse hardness. Sampling is polynomial-time at every inverse-polynomial precision, so every barrier must be
  fine-grained or about a restricted class of estimators (3.1). Pseudorandom functions have predictable means, and
  undetectable planting cannot change the distribution of the mean (3.2, 3.3). Beating the know-nothing estimate needs
  correlation across inputs (3.4).
- Determinism is free on average over random weights, nonuniformly (3.5): for each size some hard-coded input set
  matches sampling on average over the weights. (Whether explicit designs do is not settled here.)
- Circuits reduce to Gaussian means of ReLU networks of the same size, with or without biases, and with small absolute
  noise on every unit (4.1; checked numerically).
- Random-oracle networks: at a prescribed budget nothing beats sampling by more than 4x; over a budget window of size
  B the best possible advantage is Theta(log^2 B) for an explicit family, and every multiscale oracle family leaves some
  budget with advantage at least about log B / 4 (5.1). For advice to help, it must come from computation on the oracle
  that someone paid for (5.2).
- Random He-ReLU nets (each result is about a specific observable or estimator class; none transfers automatically to
  ARC's per-neuron means):
  - one input's output scale is a product of independent per-layer factors (log-variance about 5L/n), and the scale is
    shared across inputs (6.1-6.4, asymptotic);
  - an estimator that ignores the first half of the layers (knowing the rest exactly) is worth at most a constant
    number of Monte Carlo samples for the mean squared output norm (6.5; estimated upper bound 1.3-3 samples at widths
    64-128; ignoring the last half measured at 2-4);
  - the output error of any layer-by-layer description is exactly the signed sum of its local errors tested against
    the true backward functions (6.6); scale errors never shrink in bias-free ReLU nets (6.7); a mean error orthogonal
    to the state's span survives one layer with factor 1 - E[angle]/pi (6.8);
  - per-layer surrogates of fixed accuracy, and fixed-order expansions in the gain factors, lose a constant fraction of
    one input's output scale at depth proportional to width (6.10, 7.2); independent-neuron closures misjudge the next
    layer's variance by O(1) once the state has collapsed (6.11);
  - if backward functions contract, layer-local errors are depth-free (6.12); one noisy layer need not contract much
    when the states' images are far apart (6.13).
- The low-degree method cannot give hardness evidence for this problem (7.1), and "random nets are pseudorandom" cannot
  be the mechanism of any hardness (Section 7): pseudorandomness makes means easy.

**Proven under a named hypothesis.**
- Subexponential deterministic estimation for all ReLU networks would give NEXP not in P/poly
  (Impagliazzo-Kabanets-Wigderson 2002); 4.2.
- Under #NSETH, no deterministic estimator runs in time O(E^{2-eps}) on all ReLU networks with E edges at constant
  precision, while sampling is near-linear (adapting Chen and Tell, STOC 2021); 4.3.
- Explicit ReLU networks obey the random-oracle bounds under a stated joint summation-hardness assumption (Section 5);
  that assumption is not standard.

**Measured and heuristic (not proofs).**
- K = 2 against plain Monte Carlo along depth proportional to width (6.3). K = 2's gain over plain Monte Carlo at equal cost, for depth/width 1/8 to 1/2, falls from 2-8 at widths 32-128 to about 0.7-1.1 at width 1024 (fitted slopes -0.4 to -0.8 against Monte Carlo's -1); its error grows like depth^2 at shallow depth and much more slowly beyond depth 64 (tables in 6.3). Factorized K = 3 falls behind with width (above).
- Mean errors at criticality probably survive with weight about (layer/depth)^3, so constant local errors would add up
  linearly in depth; the angle asymptotics behind this are proven (6.9), the transport model is not (6.9').

**Changed after review.** Gemini found six fixable gaps (noise constants in 4.1c, a factor 2 in 3.2, variance-zero
cases); GPT-6 broke more: a claimed parameter-count ceiling on black-box hardness (false: a 4-unit family suffices),
"advice independent of the context carries no information" (false: it can tabulate all contexts), a per-scale
summation assumption used for a multiscale sum, a missing independence condition in 6.5, the accumulation corollary
stated as proven, a missing initial-error term in 6.12, and two overstated noise claims. All are fixed, narrowed or
retracted below (Section 10 lists them).

**Open.** Whether any polynomial-time estimator beats sampling by a factor that grows with width at depth
proportional to width. My conjecture (6.B): no layer-local description of polynomial size does. ARC's optimistic
alternative (a mechanistic estimator that *matches* sampling at all budgets) is not refuted and is consistent with it.

**What it means for M1** (sample-free estimator matching sampling on deep random MLPs, depth growing with width).
- No impossibility proof is possible without first formalizing "sample-free"; the worst-case barriers do not apply to
  random nets.
- For layer-local families the theorems say what must be got right, each under stated conditions: use the early
  layers (and, by measurement, the late ones); keep output-scale errors small at every layer, since they never shrink
  (about eps^2/L per layer if they are independent); keep the couplings once the state collapses; and keep the signed
  sum of local errors, tested against the true backward functions, below the sampling error at equal cost.
  Fixed-order expansions in the gain factors are not enough at L proportional to n. These are necessary conditions for
  specific observables, not an impossibility proof.
- Empirically the cheapest closure roughly matches plain Monte Carlo at its own low budget along L proportional to n
  (within about 3x either way up to width 1024), while losing ground slowly as width grows. But the next closure (factorized K = 3) falls further behind sampling as width grows, so matching sampling over a range of budgets, which is what M1 and ARC's MSP ask for, is not achieved by the cumulant hierarchy at depth proportional to width. So the M1 bar should be stated over a budget window (e.g. 1e9-3e11 FLOPs at 256 x 32, as in our
  arc-noisy runs), and against the best sampler, not one budget point against plain Monte Carlo.

**What it means for interpretable architectures.**
- Worst case: any architecture that can embed Boolean circuits as ReLU networks of near-linear size (with or without
  biases, with small absolute noise) inherits the barriers; "interpretable by design" helps only if it blocks such
  embeddings (e.g. bounded treewidth, probabilistic circuits with exact moments).
- Average case: if an architecture's layers make the backward functions contract, layer-local errors are depth-free
  (6.12); our measurements suggest this happens in the chaotic phase and with large noise, but it is not proven there.
  At criticality with homogeneous activations (ARC's family) the scale provably never contracts (6.7). A design target
  follows: layers that provably contract the statistics a description tracks. The tension (a reading, not a theorem):
  trainable deep plain nets sit near criticality, and residual streams preserve information by design, so the depth
  problem is likely generic for practical architectures unless descriptions are non-local.

## 1. The problem, formally

### 1.1 Networks, inputs, estimand
- A network is a feedforward ReLU network f_W: R^d -> R. The MLP case: W = (W_1, ..., W_L, v), W_1 in R^{n x d},
  W_l in R^{n x n}, h_0 = x, h_l = relu(W_l h_{l-1} + b_l), f_W(x) = v . h_L (or the vector h_L, scored coordinatewise).
  "He-ReLU" (ARC's family): d = n, b_l = 0, W_l iid N(0, 2/n) entries. "Width" n, "depth" L. "Size" E = number of
  nonzero weights; one forward pass costs c_fwd = Theta(E) arithmetic operations (2 n^2 L for a dense MLP). Where general
  DAG networks are needed (Section 4) they are ReLU units wired as any directed acyclic graph.
- Noisy networks: h_l = relu(W_l h_{l-1}) + sigma r_l xi_l, xi_l ~ N(0, I) fresh at every layer, r_l a fixed scale.
- Input law D: N(0, I_d) (ARC), uniform on {0,1}^d, or (in Section 2) uniform on [0,1]^d.
- Estimand: mu(W) = E[f_W(x)] over x ~ D (and the noise, for noisy nets). V(W) = Var f_W(x) over the same randomness
  (for noisy nets: over input and noise, so that one forward pass is one sample). Normalized comparisons are made only
  where V(W) > 0.

### 1.2 Estimators and access models
- (BQ) black-box query: the algorithm chooses inputs (adaptively, possibly at random) and sees f_W there; cost = number
  N of queries times c_fwd; other computation is free. (BS) black-box sample: it only sees iid pairs (x_i, f_W(x_i)).
- (WB) white-box: the algorithm receives W (rational entries) and may compute anything; cost = its running time,
  counted in arithmetic operations (Sections 2, 5, 6) or bit operations (Section 4). The two differ by the precision
  needed; Section 4 notes where that matters.
- (WB+A) white-box with advice: as WB, plus a string pi produced by an "explainer" (Section 1.5 states ARC's rules).
- Deterministic vs randomized: a deterministic estimator uses no random bits; a randomized one may. A *uniform*
  algorithm is one fixed program for all sizes; a *nonuniform* one may depend on the size (hard-coded designs).

### 1.3 Error
- Fixed network: MSE(A, W) = E_coins (A(W) - mu(W))^2; normalized nMSE = MSE / V(W) (ARC's metric).
- Random network W ~ P: E_W MSE(A, W), compared with E_W V(W) (ARC's MSP uses exactly this pair, see 1.5).
- Worst case over a class F: sup_{W in F} MSE(A, W) / V(W).

### 1.4 The sampling baseline and what "better than sampling" means
- Monte Carlo with N forward passes: MSE = V(W)/N exactly, at cost N c_fwd (plus drawing inputs). As a function of a
  budget T it is the work-normalized curve MSE_MC(T) = V(W) c_fwd / T (an idealization: a real run uses floor(T/c_fwd)
  complete passes).
- Advantage of an estimator family {A_T} at budget T: Adv(T) = MSE_MC(T) / MSE(A_T) (worst case: the infimum over the
  class of the per-network ratio; average case: E_W V c_fwd / T divided by E_W MSE).
- A family *matches sampling with constant C on a budget window [T_0, T_1]* if Adv(T) >= 1/C for every T in the window;
  it *beats sampling by G* if Adv(T) >= G on the window. Along a sequence of problem sizes (n -> infinity, L = L(n)),
  it *asymptotically beats sampling* if Adv(T(n)) -> infinity for budgets T(n) in a polynomial window [n^a, n^b].
- Why a window. For tiny tolerances exact computation can win for trivial reasons (for fixed width n and depth L the
  network is a fixed piecewise-linear map with finitely many regions, and its Gaussian mean can be computed to precision
  eps at a cost growing only polylogarithmically in 1/eps, Section 8), and the plain sampling baseline needs at least
  one forward pass. The interesting statements are about polynomial windows, uniformly in the window.
- Stronger samplers (RQMC, Kerdock/spherical designs) change MSE_MC by constant factors on ARC's nets (0.4/N at 256 x 32
  in our work/arc-noisy); Section 2 shows that no black-box algorithm does more than a constant factor in the worst case.

### 1.5 ARC's matching sampling principle (MSP), in these terms
From "Competing with sampling" (Neyman, Lecomte, Wu, Winer, Hilton, Robinson, ARC blog, 18 Nov 2025; I read it through
a fetch tool that returned quoted fragments, so the formulas below are a faithful paraphrase, not a verbatim copy):
- an architecture M maps (c, x) to R, with c a random "context" and x the input; parameters theta are worst case;
- an estimator G_M(theta, pi, c, eps), with advice pi depending on theta only, |pi| <= O(|theta|);
- for all eps: time O((1/eps^2) Time(M_theta)), error E_c[(G - E_x M_theta(c, x))^2] <= eps^2 E_c[Var_x M_theta(c, x)],
  and G is "mechanistic" (informal; working test: "avoids any random or pseudorandom sampling");
- train-and-explain variant: theta = T(s) for a trainer T with seed s, pi = E(s) with Time(E) <= O(Time(T)); the error
  is averaged over the seed s and the context c;
- the random-MLP instance: theta empty, c = W (iid Gaussian), x ~ N(0, I_n), depth a fixed constant, n -> infinity,
  1/eps = poly(n). So for random MLPs the MSP is an advice-free, average-case statement over W.
In the notation above, the MSP asks for Adv(T) >= 1/C at every budget T >= c_fwd, for a mechanistic estimator.
Kelvin's question ("better than random sampling, as deep and wide as needed") asks whether Adv can be made >= 1, or
-> infinity, as n and L grow. Wu et al. (2026, Sec. 7.3) list L -> infinity as open; Hilton's blog summary of that
paper names depth growing linearly in the width (quoted in our arc-deep-networks note); I specialize to L = alpha n.

### 1.6 Assumptions, and how each is checked
| assumption | role | check |
|---|---|---|
| A1 cost = arithmetic operations; FLOP discounts (ARC's symmetric-tensor counting) change constants only | Sections 2, 5, 6 | stated; constants never drive a claim |
| A2 weights are exact rationals of poly(size) bits; bit-operation cost (complexity statements) | Section 4 | the reduction uses weights in {0, +-1, +-2, +-K}, K a polynomially bounded integer |
| A3 estimators may be randomized unless stated | Sections 2-6 | each theorem says which class |
| A4 He-ReLU nets are bias-free with iid N(0, 2/n) weights and Gaussian input | Section 6 | as in ARC's paper and contest |
| A5 #NSETH (counting version of NSETH, Chen-Tell Assumption 6.1) | Corollary 4.3 only | named in the statement |
| A6 random oracle model; algorithms' initial information independent of the oracle unless charged | Section 5 only | named; any instantiation needs its own explicit assumption |

## 2. Black-box estimators: sampling is optimal up to a constant factor

**Theorem 2.1 (query lower bound). PROVEN.** Fix N >= 1 and m >= N. Let S_1, ..., S_m be disjoint measurable sets
(intervals in x_1 in the basic version) and g_1, ..., g_m functions with g_j = 0 outside S_j and finite second moments.
Put a_j = E g_j(x), b_j = E g_j(x)^2 under the input law D, and f_s = sum_j s_j g_j for s in {-1, +1}^m. Draw s
uniformly. Then every algorithm that makes at most N queries of f_s (adaptive, randomized, unbounded computation) has
risk, averaged over s and its coins,

  E (mu-hat - mu(f_s))^2 >= min_{|R| <= N} sum_{j not in R} a_j^2 >= (m - N) min_j a_j^2,

while Monte Carlo with N queries has E_s MSE = sum_j (b_j - a_j^2) / N.
*Proof.* Condition on the coins. A query x returns s_j g_j(x) for the unique j with x in S_j (or 0). Let R be the set
of indices j queried at a point where g_j != 0; |R| <= N. By induction on the queries the transcript depends on s only
through s_R, so given the transcript the s_j with j not in R are still iid uniform: E[mu | transcript] =
sum_{j in R} s_j a_j and Var(mu | transcript) = sum_{j not in R} a_j^2. Any estimator is a function of the transcript
and coins, so its conditional MSE is at least this conditional variance; average. Monte Carlo:
E_s Var_x f_s = sum_j b_j - E_s(sum_j s_j a_j)^2 = sum_j (b_j - a_j^2) (disjoint supports). QED

**Corollary 2.2 (ReLU realizations and constants).**
- Trapezoids g_j(t) = [relu(t - al) - relu(t - al - w) - relu(t - be + w) + relu(t - be)] / w on J_j = [al, be],
  0 < w <= (be - al)/2, four units each: f_s is a one-hidden-layer ReLU network with 4m units. With m = 2N, x_1 uniform on
  [0, 1], equal cells of length l = 1/(2N): a = l - w, b = l - 4w/3, and the ratio of Monte Carlo's risk to the Bayes
  risk N a^2 is 2(b - a^2)/(N a^2), which tends to 4(1 - 1/(2N)) as w -> 0 (it is larger for wide ramps, e.g.
  16/3 - 2/N for triangles). So no black-box algorithm beats Monte Carlo by more than a factor about 4 on this family.
- Gaussian input, cells of equal mass partitioning [-1, 1] (p = P(|Z| <= 1) = 0.6827), ramp fraction w' (ramp width /
  cell length): a_j >= (p/m)(1 - 2 e^{1/2} w'), b_j <= p/m, so the ratio is at most (4/p)/(1 - 3.3 w')^2 <= 5.9/(1 - 3.3 w')^2 (for 3.3 w' < 1).
- One trapezoid (four ReLU units) suffices per network in a variant: let f_{j,s} = s g_j with j uniform in [m] and
  s = +-1. With uniform input and equal cells (common moments a, b), the Bayes risk after N queries is (1 - N/m) a^2 and Monte Carlo's is
  (b - a^2)/N, ratio -> 4 - 2/N at m = 2N. So black-box hardness is not limited by the parameter count; what grows with
  N is the precision of the weights (the bump positions), not their number (from the GPT-6 review).
- Width-n, depth-L MLPs: carry x_1 and the running sum as positive/negative pairs (4 units) and spend the remaining
  n - 4 units per layer on hinge terms; m trapezoids fit in depth L <= 1 + ceil(4m/(n - 4)) (n >= 5).
- Bias-free (positively homogeneous) nets: f_s(x) = |x_2| sum_j s_j g_j(x_1/|x_2|) (one extra layer); the proof of 2.1
  covers it (disjoint supports in R^d, finite second moments), but x_1/|x_2| is Cauchy and coupled with |x_2|, so the
  constants must be recomputed for that version.

**Theorem 2.3 (deterministic black-box algorithms fail). PROVEN.** Add to every member of the family a fixed, known
function psi(x_2) of a second coordinate, independent of x_1, with mean 0 and variance 1/2 (e.g. a scaled clamp, two
ReLUs), so every member has variance at least 1/2. For any deterministic adaptive algorithm with N queries there are two
members, f_0 = psi and f' = psi + sum_{j not in R_0} g_j (R_0 = the intervals hit when it is run on f_0), on which it
gives the same answer although their means differ by sum_{j not in R_0} a_j >= N min_j a_j. For the uniform version
(m = 2N) this is >= 1/2 - o(1), so the algorithm's worst-case squared error is >= 1/16 - o(1), while Monte Carlo has
MSE <= 3/(4N) on both: deterministic black-box algorithms lose by a factor >= (N/12)(1 - o(1)) (as the ramps narrow) in
MSE and in nMSE.
*Proof.* Run it on f_0: it makes a fixed query sequence and outputs mu-hat_0. f' agrees with f_0 at every query, so the
run on f' is identical and also outputs mu-hat_0, while mu(f') - mu(f_0) = sum_{j not in R_0} a_j. QED

**Remarks.**
- The sample model (iid pairs, no choice of inputs) is weaker than the query model, so Theorem 2.1 holds there too.
- These are worst-case statements about explicit families. They say nothing about the average over He-ReLU weights,
  where structured samplers do gain constant factors (Kerdock about 2.4x at 256 x 32) and nothing here bounds them.
- Quantum amplitude estimation reaches error eps with O(1/eps) coherent oracle queries (Montanaro 2015,
  arXiv 1504.06987), given coherent access to the function and to the input state; the 1/eps^2 barrier is classical.

## 3. What cannot be proven, and what "mechanistic" has to mean

**Proposition 3.1 (no coarse hardness). PROVEN.** For networks with outputs in [a, b] and an input law that can be
sampled efficiently, Monte Carlo has MSE at most (b - a)^2 / (4N) at cost N c_fwd, and |mu-hat - mu| <= eps with
probability 1 - delta at cost O(c_fwd (b - a)^2 log(1/delta) / eps^2). For rational ReLU networks evaluated in floating
point with polynomially many bits, estimating mu(W) to inverse-polynomial *absolute* precision is therefore in
randomized polynomial time. Consequently no super-polynomial lower bound can hold for randomized estimators at that
precision, and every statement of the form "one cannot beat sampling" must be fine-grained (about exponents or constant
factors of polynomial costs) or about a restricted class of estimators (black-box, deterministic, or a formal model of
"mechanistic"). (Relative precision for tiny means, and exponentially small tolerances, are different problems.)

**Proposition 3.2 (indistinguishable network laws have nearly equal laws of the mean). PROVEN.** Let P_0, P_1 be laws
over networks of size E with outputs in [0, 1], and suppose no white-box distinguisher running in time t has advantage
more than adv. Let N = ceil(ln(2/eta) / (2 delta^2)) and suppose t >= N c_fwd + O(N d). Then the laws of mu(W) under
P_0 and P_1 are within Levy distance max(2 delta, eta + adv).
*Proof.* For a threshold tau, the test D(W) = 1[N-sample mean > tau] accepts with probability >= 1 - eta/2 when
mu(W) >= tau + delta and <= eta/2 when mu(W) <= tau - delta (Hoeffding). Hence
P_0(mu >= tau + delta) - eta/2 <= P_0(D = 1) <= P_1(D = 1) + adv <= P_1(mu > tau - delta) + eta/2 + adv, so with F_b the
distribution functions and x = tau + delta, F_1(x - 2 delta) <= F_0(x) + eta + adv; symmetrically with 0 and 1
exchanged. QED
*Consequence (narrow).* Planting structure that white-box distinguishers cannot detect (e.g. the backdoors of
Goldwasser, Kim, Vaikuntanathan and Zamir 2022, arXiv 2204.06974, white-box undetectable for random ReLU networks)
cannot change the *distribution* of the mean by more than this. It does not show that the instance-specific mean is
easy or that planting cannot change fine-grained difficulty.

**Corollary 3.3 (pseudorandom functions have predictable means). PROVEN, given PRF security.** Let {F_k} be a
pseudorandom function family {0,1}^m -> {0,1} (for example the Naor-Reingold functions, which are in TC^0) and
f_k(u) = F_k(u) on uniform input bits u. For every polynomial q, Pr_k(|mu(f_k) - 1/2| > 1/q(m)) is negligible, so the
constant estimate 1/2 beats Monte Carlo at every fixed polynomial budget for all but a negligible fraction of keys.
(For Gaussian input, binarize with the clipped ramps of 4.1 and add the transition-strip error m/(K sqrt(2 pi)); F_k
itself is computed exactly on Boolean inputs by threshold gates, each a difference of two ReLUs.)
*Proof.* For a truly random function, |mu - 1/2| <= 2^{-m/2} sqrt(ln(2/eta)/2) with probability 1 - eta. If the event
had non-negligible probability gamma for F_k, the O(q^2 log(1/eta))-sample test "is the empirical mean farther than
1/(2q) from 1/2" with eta much smaller than gamma would distinguish F_k from a random function. QED

**Proposition 3.4 (when the "know nothing" estimate wins). PROVEN.** For a random network W ~ P with finite second
moments, let m(x) = E_W f_W(x) and K(x, x') = Cov_W(f_W(x), f_W(x')), with x, x' independent draws from D. The annealed
estimate mu-bar = E_W mu(W) has E_W (mu-bar - mu)^2 = E_{x,x'} K(x, x'), while Monte Carlo with N samples has
E_W MSE = [E_x K(x, x) + Var_x m(x) - E_{x,x'} K(x, x')] / N. With rho = E_{x,x'} K / (E_x K(x,x) + Var_x m), the
annealed estimate beats N-sample Monte Carlo exactly when (N + 1) rho < 1.
*Proof.* Var_W(E_x f_W) = E_{x,x'} Cov_W(f_W(x), f_W(x')) by Fubini; E_W Var_x f_W = E_x E_W f^2 - E_W mu^2 =
E_x [K(x,x) + m(x)^2] - [Var_W mu + (E_x m)^2]. QED
*Consequences.* (a) Families whose outputs at independent inputs are nearly uncorrelated over the random weights
(pseudorandom functions; wide random nets with zero-Gaussian-mean activations, Dunbar and Aaronson, arXiv 2510.06527)
have predictable means up to budgets N of about 1/rho. (b) A family can be hard at budget N only if rho is at least about
1/N: hardness needs long-range structure across inputs, not randomness. (c) Which side He-ReLU nets fall on depends on
the observable and the depth: for one layer and the readout relu(w . x) the annealed estimate is worth order n samples
(GPT-6 review: Var_w mu ~ 1/(2 pi n) against E Var_x = 1 - 1/pi); for the output scale at depth proportional to width
the correlation is bounded below (Lemma 6.4) and the annealed estimate is worth O(1) samples.

**Proposition 3.5 (determinism is free on average, nonuniformly). PROVEN.** For any prior P over networks and any
randomized estimator A with E_{W, coins}(A - mu)^2 <= Delta there is a fixed coin string r* with
E_W (A_{r*}(W) - mu(W))^2 <= Delta. In particular, for each size, some fixed input set x_1..x_N, chosen without looking
at W, has E_W ((1/N) sum_i f_W(x_i) - mu(W))^2 <= E_W V(W)/N. If moreover the whole network law is invariant under
rotating the first layer, (W_1 Q, W_2, ..., v) =d (W_1, W_2, ..., v) for every orthogonal Q (true for He-ReLU, all
layers independent and the first isotropic), and D = N(0, I), then every fixed design X has exactly the average error
of its random rotation QX (an explicit design is exactly as good, on average over W, as the randomized design obtained
by rotating it; whether that randomized design matches Monte Carlo depends on the design and is not claimed here).
*Proof.* Averaging over coins. For the rotation claim: f_W(Q x) = f_{W'}(x) with W'_1 = W_1 Q and the other layers
unchanged; W' has the law of W, and mu(W') = mu(W) by rotation invariance of D. (The hypothesis must be on the joint
law; the GPT-6 review gives a counterexample in which only W_1's marginal is invariant.) QED
*Consequence.* In ARC's random-MLP instance of the MSP (average over W, no advice) the requirement that the estimator be
deterministic excludes nothing nonuniformly: for each size a hard-coded design matches Monte Carlo on average over W.
Whether explicit (uniform) designs do is a separate question; the Kerdock-type designs used in practice are randomized
by a Haar rotation, and Chen and Tell's conditional average-case derandomization suggests no barrier, but nothing here
proves one way or the other. "Mechanistic" must therefore be
formalized as something other than determinism (for example: never evaluates f_W at a point; or propagates a bounded
description layer by layer), and only for such restricted models can lower bounds be proven (Section 6). This fits Chen
and Tell (STOC 2021): under their assumptions, derandomization on average over samplable inputs costs only an n^eps
factor, while worst-case derandomization costs a factor n, optimal under #NSETH.

## 4. White-box, worst case: matching sampling deterministically is as hard as derandomization

**Theorem 4.1 (Boolean circuits reduce to Gaussian means of ReLU networks). PROVEN.** Let C be a Boolean circuit with
m inputs and s gates of fan-in <= 2 from {AND, OR, NOT}, depth d, with no gate reading the same wire twice (simplify
AND(a, a) = OR(a, a) = a first). Let K be a positive integer.
(a) There is a ReLU network f: R^m -> [0, 1], a DAG of at most 2m + 2s units and O(m + s) edges with weights in
{0, +-1, +-K} and biases in {0, +-1}, depth at most d + 1, such that
  | E_{x ~ N(0, I_m)} f(x) - Pr_{u ~ U{0,1}^m} [C(u) = 1] | <= m / (K sqrt(2 pi)).
If C is layered with width w counting carried wires, f is an MLP of width <= 2 max(w, m).
(b) (bias-free) There is a positively homogeneous network f°: R^{m+1} -> R with no biases, weights in {0, +-1, +-K},
O(m + s + d) units, depth at most d + 3, with 0 <= f°(x) <= |x_0| and |E f° / E|x_0| - Pr[C = 1]| <= m / (2K).
(c) (noisy, absolute noise) If every unit's output receives independent noise N(0, sigma^2), there is a network of
O(m + s) units and depth at most 2(d + 1) (a restore layer after the binarization layer and after every gate layer)
with |E f - Pr[C = 1]| <= m/(K sqrt(2 pi)) + 2U exp(-1/(1152 sigma^2)), U the number of units. So absolute noise
sigma <= 1/(24 sqrt(2 ln(2U/eta))) adds at most eta. (This does not cover noise proportional to each layer's RMS: the
binarization units relu(K x) have RMS of order K.)

*Proof of (a).* Binarize: beta_i(x) = relu(K x_i) - relu(K x_i - 1) = clamp(K x_i, 0, 1), which equals u_i = 1[x_i > 0]
unless 0 < x_i < 1/K. Gates on wires with values in [0, 1]: NOT a = relu(1 - a); AND(a, b) = relu(a + b - 1);
OR(a, b) = relu(a + b) - relu(a + b - 1) = min(1, a + b) (two units; consumers read their difference, an affine function
of unit outputs). Invariants: (I1) every wire value lies in [0, 1]; (I2) if every beta_i is in {0, 1}, every wire carries
the Boolean value of C at u (the gates are exact on {0, 1}). Hence f(x) = C(u(x)) outside B = union_i {0 < x_i < 1/K},
|f - C(u)| <= 1 everywhere, and P(B) <= m (1/K) phi(0) = m / (K sqrt(2 pi)). The signs of iid N(0,1) coordinates are
iid uniform bits. QED
*(b).* Compute r = |x_0| = relu(x_0) + relu(-x_0) and carry it (relu(r) = r). Replace every constant 1 by r:
beta_i = relu(K x_i) - relu(K x_i - r), NOT a = relu(r - a), AND = relu(a + b - r), OR = relu(a + b) - relu(a + b - r)
(x_i is available after the first layer as relu(x_i) - relu(-x_i)). Everything is positively homogeneous of degree 1,
every wire lies in [0, r], and outside B° = union_i {0 < x_i < r/K} every wire equals r times the Boolean value. Then
E f° = E[r] Pr[C = 1] + E[(f° - r C(u)) 1_{B°}] (r is independent of u), and the last term is at most
E[r 1_{B°}] <= sum_i E[r (r/K) phi(0)] = m E[r^2] / (K sqrt(2 pi)) = m / (K sqrt(2 pi)); divide by E r = sqrt(2/pi). QED
*(c).* Restore units R(v) = relu(2v - 1/2) - relu(2v - 3/2) = clamp(2v - 1/2, 0, 1) map [-1/4, 1/4] to 0 and
[3/4, 5/4] to 1. A restored wire is the difference of two noisy units, so if every noise variable has absolute value
<= tau = 1/24 it carries its Boolean value up to 2 tau. A gate fed by restored wires then produces a value within
6 tau = 1/4 of its Boolean value (OR is the worst case: clamp(a + b, 0, 1) is 1-Lipschitz, so 2 x 2 tau from its inputs
plus 2 tau from its own two units; AND 5 tau, NOT 3 tau; binarization pairs 2 tau outside B), and the next restore layer
returns the exact Boolean value up to its own 2 tau. Globally, the final output is a restored value in [0, 1] plus a
zero-mean noise difference: f = R_out + (xi_1 - xi_2) with R_out in [0, 1] for every realization, and R_out = C(u) on the
event that all |noise| <= tau and x is outside B. Taking unconditional expectations, |E f - Pr[C = 1]| <= P(B) +
P(some |noise| > 1/24) <= m/(K sqrt(2 pi)) + 2U exp(-1/(1152 sigma^2)). QED

Numerical check (numerics/capp_check.py, 6 random circuits with m = 10, s = 40, K = 1000, 4e5 Gaussian samples): the
network means equal the exact acceptance probabilities within Monte Carlo error in all three variants (for example
0.2500 exact; 0.2502 +- 0.0007 with biases; 0.2498 +- 0.0009 bias-free; 0.2501 +- 0.0007 with noise sigma = 0.02). This
checks the implementation, not the uniform statement; at sigma = 0.02 the union bound in (c) is already vacuous for
these sizes, yet the means were unaffected.

**Corollary 4.2 (the IKW barrier). PROVEN reduction + cited theorem.** Impagliazzo, Kabanets and Wigderson (JCSS 2002,
Section 5.1) call CAPP *nontrivially approximable* if for every eps > 0 there is a nondeterministic algorithm running in
time 2^{s^eps} with s^eps bits of advice which, on infinitely many sizes s and for every circuit of that size, has an
accepting computation and outputs on every accepting computation an approximation of the acceptance probability within
1/6. Suppose some algorithm (deterministic, or nondeterministic in the same sense, with E^{o(1)} advice) computes
E_{x~N(0,I)} f(x) within +-1/12 for every ReLU network with E edges and outputs in [0, 1] in time 2^{E^{o(1)}}. Apply
4.1(a) with K = 12 (m + s) (reduction error < 1/12; E = O(m + s); pad networks to common sizes): CAPP is nontrivially
approximable, so by their Theorem 32, NEXP is not contained in P/poly. A deterministic polynomial-time algorithm would
also give promise-BPP = promise-P (their Remark 33).
*Reading.* Such deterministic estimators are believed to exist (standard circuit lower-bound assumptions give them by
hardness-versus-randomness), but proving that they exist would prove a circuit lower bound far beyond current
techniques. So nobody should expect to *prove* an advice-free, worst-case version of "deterministic estimation can
match sampling for every network". The advice allowed in IKW is sublinear; ARC's advice of length O(|theta|) escapes
this (Proposition 4.4).

**Corollary 4.3 (deterministic estimators cannot match sampling in the worst case). PROVEN UNDER #NSETH.** Assume
#NSETH: for every eps > 0 there is a k such that no nondeterministic machine counts the satisfying assignments of
k-SAT formulas on v variables in time 2^{(1-eps) v} (unambiguously: every branch outputs the correct count or "reject",
and some branch outputs the count; Chen and Tell, Assumption 6.1 and Definition 3.3). Then for every eps > 0 there is no
deterministic algorithm that, for every ReLU DAG network f with E edges and outputs in [0, 1], computes
E_{x~N(0,I)} f(x) within +-1/24 in time O(E^{2 - eps}) (bit operations, with weights of O(log E) bits), even restricted
to networks whose output variance is at least 1/5 (so the statement holds equally for ARC's normalized error). Monte
Carlo does it in time O~(E log(1/delta)) with probability 1 - delta. So in the worst case no deterministic estimator
improves on quadratic time by any fixed polynomial factor, while sampling is near-linear: the loss is a factor
E^{1 - o(1)} at constant precision.
*Proof.* We follow Chen and Tell's proof of their Theorem 6.4 (case k = 1), which uses its derandomization hypothesis
only for one explicit family of randomized tests. Counting reduction: split the v variables of a k-CNF into two halves,
list the n = 2^{v/2} partial assignments of each half as 0/1 vectors indexed by clauses (1 = the clause is not
satisfied by this half), so that full assignments correspond to pairs and satisfying assignments to orthogonal pairs;
for fixed k a k-CNF has O(v^k) distinct clauses, so the dimension d is polylogarithmic in n and the count is preserved (this is the counting form of the k-SAT to OV reduction behind their Lemma 6.3, which is
stated for NSETH). The test: given a prime p in (n^4, 2n^4] (its primality checked deterministically in polylog n time),
a univariate polynomial H of degree O(n d) = n polylog n over F_p (the degree bound of Q in their construction)
and the 2-OV instance, accept iff H(x) = Q(x) at a random x in F_p, where Q(x) = sum_{j in [n]} X(P_1(j), P_2(x)) as in
their proof. P_2(x), H(x) and the sum take n polylog n field operations; with O(log n)-bit field elements and x obtained
by reducing a uniform b-bit integer mod p (b = O(log n), statistical error <= n^{-10}), the test is a Boolean circuit C
of size n polylog n, accepting with probability 1 if H = Q and at most deg/p + n^{-10} < 1/3 otherwise. Pad it with three
fresh input bits a, b, c: C'' = (a AND b) OR (a AND NOT b AND c) OR (NOT a AND b AND C), so Pr[C'' = 1] = 3/8 + Pr[C = 1]/4
lies in [3/8, 5/8] (Boolean variance >= 15/64; after the Gaussian smoothing of 4.1(a) with K = 48 |C''| the network's
variance stays above 1/5) and the two cases differ by >= 1/6 in mean. A deterministic O(E^{2-eps}) estimator with error
+-1/24 therefore distinguishes them in time n^{2 - eps'}. The nondeterministic counter then guesses p and H, verifies
H = Q with this test, and outputs sum_{i in [n]} H(i) mod p computed by fast multipoint evaluation in O~(n) time (the
true count is at most n^2 < p, so the residue is the count). This counts 2-OV solutions, hence k-SAT solutions,
unambiguously in time n^{2 - eps'} = 2^{(1 - eps'/2) v}, contradicting #NSETH. QED
*Reading.* This is the cleanest sense in which "a deduction-only method that is never worse than sampling, for every
network" is false: it is false for deterministic algorithms in the worst case, under a standard fine-grained
hypothesis, with a loss factor equal to the network size. Chen and Tell also show the matching upper bound (overhead
n T^eps) under one-way functions plus strong hardness, so the worst-case gap is about one factor of the size.
(The proof needs DAG networks with near-linear size; for layered MLPs the carried wires change the size accounting.)

**Proposition 4.4 (what the worst-case results say about ARC's MSP). PROVEN (logical relations).**
1. The MSP is quantified worst-case over theta *with advice pi(theta) of length O(|theta|)*; when a worst-case circuit
   sits in theta (and the context does not matter), pi can state the answer (ARC say so themselves). So Corollaries 4.2
   and 4.3 do not refute the MSP. (Advice of length O(|theta|) cannot in general store the answers for every context
   c; that is the point of ARC's random context.)
2. They do apply to the reading "for every network, a deterministic estimator without instance-specific advice matches
   sampling": proving it is at least as hard as NEXP not in P/poly, and its constant-factor version is false under
   #NSETH.
3. ARC's random-MLP instance has theta empty and c = W random; only average-case statements apply, and by
   Proposition 3.5 determinism is no restriction there nonuniformly (hard-coded designs, on average over W). Chen and Tell's
   conditional average-case derandomization (overhead n^eps over samplable input distributions) is relevant evidence
   that average-case statements meet no comparable barrier; it is not a theorem about the Gaussian MSP. For the
   train-and-explain variant the guarantee is averaged over the trainer's seed, so a trainer that outputs a random
   circuit also yields an average-case statement.
So the worst-case theory sharpens what the MSP must mean, but it cannot refute it: the open content is in "mechanistic"
and in the average case.

## 5. Cryptographic and idealized constructions: how much can anything beat sampling?

By Proposition 3.1, standard (polynomial-versus-superpolynomial) cryptography cannot make mean estimation hard at
inverse-polynomial precision; by 3.3 pseudorandom functions have predictable means, and by 3.2 undetectable planting
cannot change the distribution of the mean.
The right question is fine-grained: on hard instances, by how much can any estimator beat Monte Carlo? The setting where
this can be proven unconditionally is an oracle model.

**Model.** H: {0,1}* -> {-1, +1} is a uniformly random function (random oracle); queries (j, v) use an injective
encoding. Networks may contain oracle gates (an extension of the ReLU model). An estimator sees the whole network (it is
white-box for everything except the oracle), may compute anything, is charged one unit per oracle call (a hard budget),
and its initial information is independent of the oracle unless the oracle calls used to produce it are charged. A
forward pass that makes J oracle calls costs J.

**Theorem 5.1 (random-oracle networks). PROVEN in the random oracle model.** Adv below refers to the optimal (Bayes)
estimator at each budget, against the work-normalized Monte Carlo curve.
(a) Single budget. Let f(u) = H(u_1 ... u_m), u uniform input bits (or signs of a Gaussian input binarized as in 4.1).
Every estimator with at most Q oracle calls has E_H (mu-hat - mu)^2 >= (2^m - Q)^+ / 4^m, while Monte Carlo with N
calls has E_H MSE = (1 - 2^{-m}) / N. At Q = N = 2^{m-1} no estimator beats Monte Carlo by more than 4 (1 - 2^{-m}).
(b) A budget window. Let f(u) = sum_{j=1}^J H(j, u_1 ... u_j) (one forward pass = J calls). Every estimator with at most
Q calls has E (mu-hat - mu)^2 >= R(Q) := min { sum_j (2^j - q_j) 4^{-j} : sum_j q_j <= Q, 0 <= q_j <= 2^j }
>= 1/(Q + 2) - 2^{-J} (exactly R(Q) = 2^{-k} - 2^{-J} - (Q - S_k) 4^{-(k+1)} for S_k = 2^{k+1} - 2 <= Q < S_{k+1}),
while Monte Carlo has E MSE = J (J - 1 + 2^{-J}) / Q on the work-normalized curve. Hence for J <= Q <= 2^{J-1} - 2 no
estimator beats Monte Carlo by more than 4 J^2, and enumerating the coarse scales (all (j, v) with j <= k) beats it by
at least J (J - 1)/4. In this family the best possible advantage is Theta(J^2) = Theta(log^2 B) throughout the window
[J, B], B = 2^{J-1}.
(c) Polylogarithmic advantage is unavoidable for multiscale oracle constructions. Consider a network that, per forward
pass, touches scale j with probability pi_j in (0, 1] (one call each; inclusion decided by input bits independent of
the oracle), with cell_j(x) uniform on the 2^j cells of scale j given that j is touched, and outputs the unbiased
(Horvitz-Thompson) combination sum_{j touched} (w_j / pi_j) H(j, cell_j(x)), weights w_j >= 0. Then, on the
work-normalized curve, sum_{k=1}^{J} 1/Adv(2^k) <= 4 sum_j w_j^2 / (sum_j w_j)^2, so some budget 2^k has Adv(2^k) >= J/4,
and >= J^2/4 when the weights are equal.
*Proofs.* (a) Given the queried points, the unqueried values are iid uniform +-1, so
Var(mu | transcript) = (2^m - q)/4^m with q <= Q distinct points; Var_x f = 1 - mu^2 and E mu^2 = 2^{-m}. (b) Same per
scale: the posterior variance is sum_j (2^j - q_j) 4^{-j}. The minimizing allocation fills small j first (each query to
scale j removes 4^{-j}); with k maximal such that S_k <= Q, at most scales 1..k+1 are touched, so
R(Q) >= sum_{j >= k+2} 2^{-j} = 2^{-(k+1)} - 2^{-J} >= 1/(Q+2) - 2^{-J}. Monte Carlo: E_H Var_x f = J - sum_j 2^{-j}
(oracle values at different scales are independent), N = Q/J passes. When Q + 2 <= 2^{J-1} the bound is
>= 1/(2(Q+2)), giving advantage <= 2 J^2 (Q+2)/Q <= 4 J^2. Coarse enumeration leaves 2^{-k} - 2^{-J} <= 4/(Q+2).
(c) Completing scales j < k costs < 2^k calls, so R(2^k) <= sum_{j >= k} w_j^2 2^{-j}; summing,
sum_k 2^k R(2^k) <= 2 sum_j w_j^2. Monte Carlo: E Var_x f = sum_j w_j^2 (1/pi_j - 2^{-j}) >= sum_j w_j^2 / (2 pi_j), and
the expected cost per pass is sum_j pi_j; by Cauchy-Schwarz (sum_j w_j)^2 <= (sum_j w_j^2/pi_j)(sum_j pi_j). QED

**What this shows, and what it does not.** For these structureless families (networks built from random oracles, where
white-box access is worth nothing beyond the wiring) the optimal estimator beats sampling by polylogarithmic factors
uniformly over the window, and by at most a constant at a prescribed budget. It is not a statement about all oracle
models, nor about budgets beyond the window (at Q >= 2^m in (a), enumeration is exact). Evaluating a random function at
fixed points behaves exactly like sampling it, so in these families a fixed-subset average is a deterministic estimator
that matches Monte Carlo; whether that counts as "mechanistic" is a matter of definition.

**Instantiation (CONDITIONAL on an explicit, non-standard assumption).** Let F_k(j, v) in {-1, +1} be a keyed function
family computed exactly on Boolean inputs by a ReLU network of size t_F (e.g. a TC^0 pseudorandom function; threshold
gates on Boolean inputs are relu(S - theta + 1) - relu(S - theta) for integer S), and f_k(u) = sum_j F_k(j, u_1...u_j).
*Joint summation-hardness assumption JSH(F):* given k, every algorithm running in time t has
E_k (mu-hat - sum_j 2^{-j} sum_v F_k(j, v))^2 >= (1 - o(1)) R(t/t_F), with R as in 5.1(b), and the second moments of f_k
match those of the oracle family up to (1 + o(1)). Under JSH(F), the explicit ReLU networks f_k obey the bounds of 5.1(b)
with Q = t/t_F. JSH holds in the random oracle model. It is *not* implied by pseudorandomness (PRF security is about
adversaries without k), and it must be joint: a per-scale assumption does not give a lower bound for the sum (the GPT-6
review's objection to my first version). I know of no shortcut for standard constructions and did not search for one.

**Proposition 5.2 (advice, and ARC's "pi cannot depend on c"). PROVEN in the random oracle model.** Key the oracle by
the context: H_c(.) = H(c, .), c uniform over a context set.
1. If the advice is computed without oracle access (or the oracle calls used to compute it are charged to the budget),
   the bounds of 5.1 hold for G(theta, pi, c, eps) with the charged calls added to Q. Independence of pi from the
   *realized* c is not enough by itself: unbounded advice could tabulate mu(c) for every c in a finite context set (the
   GPT-6 review's counterexample). ARC's condition does its work together with the length bound |pi| <= O(|theta|) when
   the context set is large; I have not proven a quantitative version of that.
2. If pi may depend on the realized c, ceil(log2(J/eps)) + O(1) bits encode mu to within eps (mu lies in [-J, J]).
3. Train-and-explain: if the key sits in theta = T(s), the trainer makes at most t_T oracle calls and the explainer
   E(s) runs in time t_E, then theta and pi are functions of the seed and of at most t_T + t_E oracle values, so the
   bounds hold with Q replaced by Q + t_T + t_E. Advice from an explainer no stronger than the trainer helps only if the
   trainer itself spent a comparable amount of computation on the oracle.
So, in this model, advice helps only to the extent that someone (trainer or explainer) paid for computation on the
oracle. By analogy, hard instances for "beating sampling" would have to put their difficulty in computation nobody
performed, as random networks (c = W, never selected) do; this is an analogy, not a theorem about He-ReLU nets.

## 6. Average case: random He-ReLU networks at criticality, depth growing with width

Setting: x ~ N(0, I_n), h_0 = x, h_l = relu(W_l h_{l-1}), W_l iid N(0, 2/n) entries, no biases. For an input x let
u_l(x) = h_l / ||h_l|| and G_L(x) = ||h_L(x)||^2 / ||x||^2 (the output scale; G_L depends only on the direction of x).
alpha = L/n. c_l(x, x') = cosine between h_l(x) and h_l(x'); for l >= 1 both vectors are nonnegative, so c_l >= 0.

### 6.1 Exact facts

*Convention (extinction).* For each input run the auxiliary recursion z_l = W_l u_{l-1}, rho_l = ||relu(z_l)||^2,
u_l = relu(z_l)/sqrt(rho_l) if rho_l > 0 and u_l = e_1 otherwise, with u_0 = x/||x||, and set c_l = u_l . u'_l for two
inputs. These coincide with the physical quantities as long as h_l != 0; the physical output is 0 from the first
rho_l = 0 on (probability at most L 2^{-n}). Every u_l with l >= 1 is a nonnegative unit vector, so c_l >= 0 for l >= 1.

**Lemma 6.1 (single-input norm law). PROVEN (exact; known in this form, cf. Hanin 2018, arXiv 1801.03744; Hanin and
Nica 2020, arXiv 1812.05994, for the log-normal limit).** For fixed x != 0, G_L(x) = ||h_L(x)||^2/||x||^2 =
prod_{l=1}^L rho_l with rho_l iid, rho_l ~ (2/n) sum_{j=1}^n relu(g_j)^2, g_j iid N(0,1). E rho = 1, Var rho = 5/n, so
E G_L = 1 and E G_L^2 = (1 + 5/n)^L -> e^{5 alpha}.
*Proof.* Given layers < l, u_{l-1} is a past-measurable unit vector and z_l = W_l u_{l-1} has iid N(0, 2/n)
coordinates whatever it is; so rho_l has the stated law and is independent of the past. Var rho =
(4/n^2) n (E relu(g)^4 - (E relu(g)^2)^2) = (4/n)(3/2 - 1/4). QED

**Lemma 6.2 (two inputs). PROVEN (exact).** For nonzero x, x' and l >= 1,
E[rho_l(x) rho_l(x') | layers < l] = 1 + (4/n)(J(c_{l-1}) - 1/4), where for a standard bivariate normal pair with
correlation c, J(c) = E[relu(g)^2 relu(g')^2] = [3 sin t cos t + (pi - t)(1 + 2 cos^2 t)] / (2 pi), t = arccos c (the
degree-2 arc-cosine kernel of Cho and Saul, divided by 2). J(0) = 1/4, J(1) = 3/2, J(-1) = 0, and J'(c) = 4 mu(c) > 0
with mu(c) = E[relu(g) relu(g')].
*Proof.* (z_j, z'_j) are iid bivariate normal with correlation c_{l-1}; expand (2/n)^2 sum_{j,k} relu(z_j)^2
relu(z'_k)^2. Checked by Monte Carlo at 8 values of c (numerics/lemma_checks.py J). QED

**Lemma 6.3 (cosines stay positive). PROVEN.** Let c_* = 1/(4 pi). For l >= 2,
P(c_l < c_* | layers < l) <= exp(-n/(8 pi^2)) + 2 exp(-0.0646 n), and 1/(8 pi^2) = 0.01267.
*Proof.* Normalize variances to 1. Numerator sum_j X_j, X_j = relu(z_j) relu(z'_j) >= 0, E X_j = mu(c) >= 1/(2 pi)
for c >= 0, E X_j^2 = J(c). For independent nonnegative summands, P(S <= ES - s) <= exp(-s^2 / (2 sum E X_j^2)) (from
e^{-tX} <= 1 - tX + t^2 X^2/2 for t, X >= 0). With s = ES - n c_* this gives exp(-n R(c)), R(c) = (mu(c) - c_*)^2 /
(2 J(c)); R is increasing on [0, 1] (the GPT-6 review gives the analytic argument), so its minimum is R(0) =
1/(8 pi^2). Denominator: E exp(lambda relu(g)^2) = 1/2 + (1/2)(1 - 2 lambda)^{-1/2}; lambda = 0.215 gives
P(||relu(z)||^2 >= n) <= exp(-0.0646 n). On the complement of these events, c_l >= n c_*/n. QED

**Lemma 6.4 (the output scale is shared across inputs). PROVEN (asymptotic in n, L = O(n)).** Let
kappa_* = 4 (J(c_*) - 1/4) = 0.21552 and delta_* = kappa_*/n. For nonzero x, x' and L >= 2,
Cov_W(G_L(x), G_L(x')) >= (1 - 1/n)(1 + delta_*)^{L-2} - 1 - eps_{n,L},
eps_{n,L} <= sqrt(3) L (1 + delta_*)^L M_4(n)^{L/2} exp(-n/(16 pi^2)), M_4(n) = E rho^4 = 1 + 30/n + 251/n^2 + 558/n^3
(exact). eps tends to 0 when L = O(n), and for L = alpha n the bound tends to e^{kappa_* alpha} - 1 > 0. It is an
asymptotic statement: at the four table configurations of 6.5 the remainder bound is between 52 and 923, so the
inequality says nothing there.
*Proof.* Let Z_l = rho_l(x) rho_l(x') >= 0, m(c) = 1 + (4/n)(J(c) - 1/4), E_l = {c_k >= c_* for 2 <= k <= l},
Y_l = prod_{k<=l} Z_k 1_{E_l}. Base: E[Z_1] = m(c_0) >= 1 - 1/n (J >= 0) and E[Z_1 Z_2] >= E[Z_1] (c_1 >= 0, so
m(c_1) >= 1). For 3 <= l <= L: E[Y_{l-1} Z_l] = E[Y_{l-1} m(c_{l-1})] >= (1 + delta_*) E[Y_{l-1}] (on E_{l-1},
c_{l-1} >= c_*, and J is increasing), and Y_l >= Y_{l-1} Z_l - P_l 1(c_l < c_*) with P_l = prod_{k<=l} Z_k, the
subtracted term bounded by Cauchy-Schwarz with E[Z_k^2 | past] <= M_4(n) and Lemma 6.3. Iterate up to Y_{L-1} and
use E[G G'] = E[P_L] >= E[Y_{L-1} Z_L]. QED

**Theorem 6.5 (estimators that ignore the early layers are worth a bounded number of samples). PROVEN (asymptotic) +
MEASURED (finite n).** Let S(W) = E_x ||h_L(x)||^2 / n, the mean squared output norm (a quadratic observable, not ARC's
per-neuron means; E_W S = 1). Split W into A = (W_1, ..., W_{L-k}) and B = (W_{L-k+1}, ..., W_L), m = L - k >= 2. Any
estimator S-hat = F(B, xi) with A independent of the pair (B, xi) (xi may hold coins, the input law, mean-field
constants; any amount of computation on B) satisfies
  E (S-hat - S)^2 >= Var(E[S | A]) = Var_A( E_x ||h_m(x)||^2 / n ) >= (1 - 1/n)(1 + kappa_*/n)^{m-2} - 1 - eps_{n,m}.
Monte Carlo with N forward passes has MSE E_W Var_x(||h_L||^2/n) / N <= [(1 + 2/n)(1 + 5/n)^L - 1] / N. Hence the best
estimator of this kind is no better than Monte Carlo with N_* = E_W Var_x / Var(E[S|A]) samples, and for L = alpha n,
m = beta n the proven ceiling N_* <= (e^{5 alpha} - 1)/(e^{kappa_* beta} - 1) + o(1) is a constant (crude: about 91 at
alpha = 1/4, k = L/2).
(The independence hypothesis must be on the pair (B, xi): the GPT-6 review gave a counterexample in which xi alone is
independent of A but B and xi together encode A.)
*Proof.* For every fixed x, averaging the last k layers preserves ||h_m(x)||^2 (Lemma 6.1), so by Fubini
E[S | A] = E_x ||h_m(x)||^2 / n. With A independent of (B, xi), S - E[S|B, xi] = (S - E[S|A] - E[S|B,xi] + E S) +
(E[S|A] - E S), the two parts are orthogonal (the first has conditional mean 0 given A), so
E(S - F(B, xi))^2 >= Var(E[S|A]). Var(E[S|A]) = E_{x,x'} [(||x||^2/n)(||x'||^2/n) Cov(G_m(x), G_m(x'))] and Lemma 6.4
bounds the covariance uniformly over input pairs; E ||x||^2/n = 1. Monte Carlo: E_{W,x} f_L^2 = (1 + 2/n)(1 + 5/n)^L and
E_W Var_x <= E f^2 - (E_W S)^2. QED
*Numbers* (numerics/lemma_checks.py backward; 300 nets each, inner Monte Carlo variance subtracted; point estimates
without confidence intervals):

| n, L, m = L - k | Var_W(E[S|A]) measured | mean-field value | E_W Var_x(f_L) | N_* (upper bound on the samples such an estimator is worth) |
|---|---|---|---|---|
| 64, 16, 8 | 0.31 | 0.32 | 0.95 | 3.0 |
| 64, 32, 16 | 2.08 | 1.18 | 4.83 | 2.3 |
| 128, 32, 16 | 0.47 | 0.48 | 1.08 | 2.3 |
| 128, 64, 32 | 1.42 | 1.63 | 1.79 | 1.3 |

"Mean-field value" = prod_{l=1}^{m} [1 + (4/n)(J(c-bar_{l-1}) - 1/4)] - 1 with c-bar_0 = 0 and the mean-field cosine
recursion (Lemma 6.2 along typical cosines); it matches the measurement when 5m/n is small (0.32 vs 0.31, 0.48 vs
0.47). The estimated upper-bound ratio N_* is 1.3-3.0 at these sizes (point estimates; an upper bound on what the best
estimator of this kind is worth, not its exact worth), and the proven ceiling is a constant asymptotically. Whether
the same holds when the estimator also sees cheap statistics T(A) of the first layers (e.g. their Frobenius norms)
would need a bound on E Var(E[S|A] | T(A)); I have not proven one. This is the theorem behind
our finding that an input-independent backward stand-in fails at the critical point (random-net-phases, section 3),
and it is specific to homogeneous activations: in the chaotic tanh nets the scale is set by saturation in the last
layers and a backward stand-in works (k* = 24-40 there, measured). Estimators that ignore the *last* k layers fail in
the data too (MEASURED, `lemma_checks.py fwdtrunc`, nested sampling over 80 x 16 nets: Var_B(E[S|B]) = 0.47 at n = 64,
L = 16, k = 8 and 0.25 at n = 128, L = 32, k = 16, i.e. worth at most about 2 and 4 samples); I have no proof of that
mirror statement (for one layer, Var S_1 = 2/n^2, so the single-input gain variance does not transfer directly).

### 6.2 How errors propagate

**Lemma 6.6 (exact error identity). PROVEN.** Fix W. Let T_l be the true layer map on laws (pushforward through layer
l, including noise if any), P_0 = D, and G_l(h) = E[readout | h_l = h] the true backward function, so
G_{l-1} = T_l^* G_l. For any estimator that maintains finite (signed) measures Q_0, ..., Q_L, or linear functionals
whose domains contain the G_l and satisfy (T_l Q)(g) = Q(T_l^* g), with all pairings below finite, and outputs
mu-hat = Q_L(G_L),
  mu - mu-hat = (P_0 - Q_0)(G_0) + sum_{l=1}^{L} (T_l Q_{l-1} - Q_l)(G_l).
*Proof.* (T_l Q_{l-1})(G_l) = Q_{l-1}(G_{l-1}); telescope. QED
Each term is the *local closure error at layer l, tested against the true backward function G_l*. The output error is
the signed sum of these terms (they can cancel); bounding their absolute values gives a sufficient accuracy condition.
A cumulant-propagation algorithm fits this form only after a rule that turns its finite cumulants into a functional on
the (non-polynomial) G_l (for instance the Gaussian or Edgeworth law it implicitly assumes). Forgetting (contraction)
means that G_l becomes nearly constant on the typical support as L - l grows; criticality means it does not.

**Proposition 6.7 (scale errors never contract). PROVEN (exact, noiseless).** For a bias-free ReLU network (any
weights) the map from layer l to the output is positively homogeneous of degree 1, so G_l(c h) = c G_l(h) for c > 0. If
a description makes a relative scale error t > -1 at layer l (Q_l = law of (1 + t) h, h ~ T_l Q_{l-1}), the output
estimate is off by exactly the factor (1 + t) (for the quadratic observable of 6.5, (1 + t)^2), at every depth.
Independent centered multiplicative errors t_l with variances v_l give
Var[prod_l (1 + t_l) - 1] = prod_l (1 + v_l) - 1, which is about sum_l v_l while that sum is small and grows
exponentially in depth at fixed per-layer v. No distance-to-output discount exists for this mode. With additive
noise, homogeneity fails (by O(sigma) for units near their kink), and I claim no general rate for that case.

**Proposition 6.8 (one-step survival of mean errors). PROVEN (exact, to first order in delta).** Let Q be a law of the
state at layer l - 1 with finite first moment and Q({0}) = 0, and delta a nonzero vector orthogonal to every state in
the support of Q, both fixed before a fresh W with iid N(0, 2/n) entries. Translate every state by delta. The
first-order change of the next layer's mean is Delta = E_{h~Q}[D_W(h) W delta], D_W(h) = diag(1[W h > 0]), and
  E_W ||Delta||^2 = ||delta||^2 E_{h, h' ~ Q iid} [1 - theta(h, h')/pi]   (theta = angle between h and h').
*Proof.* Delta_j = p_j(w_j)(w_j . delta) with p_j(w) = P_{h~Q}(w . h > 0). The projections of a Gaussian row onto
span(supp Q) and its orthogonal complement are independent, so w_j . delta is independent of {w_j . h}, and
E Delta_j^2 = (2/n)||delta||^2 E_{h,h'} P(w . h > 0, w . h' > 0) with P(both > 0) = (pi - theta)/(2 pi). Sum over j. QED
*Scope.* An exactly orthogonal delta exists only if the support spans less than R^n; for the Gaussian-input law the
support spans R^n, so the exact hypothesis fails; I give no approximation bound for nearly orthogonal errors. The numerical check (n = 256, an empirical law of 128 states at depths 4, 16, 48: 0.741 +- 0.013 vs
0.740; 0.909 +- 0.020 vs 0.887; 0.957 +- 0.019 vs 0.964) tests the formula for that empirical law only.

**Proposition 6.9 (the angle recursion). PROVEN (deterministic mean-field map).** For the mean-field cosine map
kappa(c) = (sin t + (pi - t) cos t)/pi, t = arccos c, and any c_0 < 1, the iterates satisfy
1 - kappa(c) = (1 - c) - (2 sqrt 2/(3 pi))(1 - c)^{3/2} + O((1 - c)^2), hence 1 - c_l ~ 9 pi^2/(2 l^2) and
theta_l ~ 3 pi / l. The approach is slow: from c_0 = 0, (1 - c_l)/(9 pi^2/(2 l^2)) = 0.29, 0.68, 0.81, 0.95 at
l = 10, 50, 100, 500.
*Proof.* Expand kappa(cos t) = 1 - t^2/2 + t^3/(3 pi) + O(t^4); with u_l = 1 - c_l,
u_{l+1}^{-1/2} - u_l^{-1/2} -> sqrt 2/(3 pi). QED

**Heuristic 6.9' (accumulation at criticality). NOT PROVEN.** If an injected mean error kept the hypothesis of 6.8 at
every later layer and propagated contributions stayed uncorrelated, the survival from layer ell to L would be
prod_{l=ell}^{L} (1 - theta_l/pi), which with theta_l = 3 pi/l equals (ell-3)(ell-2)(ell-1)/(L(L-1)(L-2)) ~ (ell/L)^3
for large ell, and constant local errors e would accumulate (in mean square) to about e^2 L/4 (exactly (L - 3)/4 times
e^2 with the substituted product, summed from ell = 4): linear growth in depth. Both
hypotheses are unproven (after one layer the perturbation D_W(h) W delta depends on the state and is no longer a common
translation). Contest participants report the product law on ARC's nets within 1-6%, and the contrast with the chaotic
phase (where the corresponding factor tends to a constant below 1 and K = 2's error saturates) is measured
(random-net-phases), not derived here.

**Lemma 6.10 (multiplicative accumulation). PROVEN (exact).** Let rho_l and rho-hat_l be F_l-measurable with finite
second moments, and suppose E[rho_l^2 | F_{l-1}] = a + b, E[rho_l rho-hat_l | F_{l-1}] = a and
E[rho-hat_l^2 | F_{l-1}] = a for constants a, b. Then E(prod rho_l - prod rho-hat_l)^2 = (a + b)^L - a^L, so (for
a + b > 0) the MSE divided by E(prod rho_l)^2 is 1 - (a/(a+b))^L.
*Proof.* Successive conditioning on F_{L-1}, F_{L-2}, ... of the three products prod rho^2, prod rho rho-hat and
prod rho-hat^2 gives (a + b)^L, a^L and a^L. QED
*Example (per-layer polynomial surrogates).* Replace each layer's norm ratio rho_l by its degree-D Hermite projection in
the layer's weights, computed along the true normalized trajectory (a degree-D polynomial activation applied along the
true path). Rotation invariance makes the conditional moments constant: a = 1 + 5(1 - r_D)/n, b = 5 r_D/n, with
r_D = the fraction of Var(relu(g)^2) above degree D: r_1 = 1 - 8/(5 pi) = 0.491, r_2 = 3/5 - 8/(5 pi) = 0.091,
r_3 = r_4 = 0.0058, r_5 = 0.0016. At L = alpha n the normalized MSE tends to 1 - exp(-5 r_D alpha), e.g. 0.107 for D = 2
at alpha = 1/4, independent of n. Simulation agrees where the moments are resolved (n = 64, L = 16, D = 2:
0.314 +- 0.017 against 0.334; n = 128, L = 32: 0.332 +- 0.016 against 0.353; D = 1 at n = 256, L = 64: 1.55 +- 0.09
against 1.56); at 5L/n = 5-10 (n = 32 or 64 with L = 64) the heavy log-normal tails were not resolved by 4e4 runs and
those checks are uninformative. Limits: the surrogate needs the true trajectory, and this is a single-input statement.
It shows that local surrogates good to a fixed accuracy per layer lose a constant fraction at depth proportional to
width, even for a quantity a forward pass computes exactly.

**Proposition 6.11 (product-measure closures misjudge the next layer once the state collapses). PROVEN (exact second
moments).** Let h have covariance C over inputs (tr C > 0) and let a closure keep only D = diag(C) (independent
neurons). For a fresh neuron z = w . h, w ~ N(0, (2/n) I) independent of C, the predicted variance w^T D w differs from
the true w^T C w by w^T (C - D) w, with mean 0 and standard deviation sqrt(2)(2/n)||C - D||_F, against a mean true
variance (2/n) tr C. For C = lambda u u^T the ratio is sqrt(2 (1 - sum_i u_i^4)) -> sqrt 2 for a spread-out u. This
concerns the pre-activation variance, not directly the output-mean error. That the He-ReLU covariance is close to rank
one at depth is a measurement (participation ratio 2-4 at depths 32-64, random-net-phases), not a consequence of the
angle collapse (nearly aligned states can have isotropic covariance).

**Theorem 6.12 (contraction makes layer-local errors depth-free). PROVEN (generic).** Suppose for a seminorm ||.||_F
on test functions with dual distance d_F (finite on the measures involved), ||G_l||_F <= C eta^{L-l} with
0 <= eta < 1 and C uniform in depth, and every local error satisfies d_F(T_l Q_{l-1}, Q_l) <= e. Then
|mu - mu-hat| <= C eta^L d_F(P_0, Q_0) + C e (1 - eta^L)/(1 - eta).
*Proof.* Lemma 6.6 and |(P - Q)(G)| <= ||G||_F d_F(P, Q). (The first version omitted the initial term; the GPT-6
review's counterexample T_l(h) = eta relu(h), P_0 = N(0,1), Q_0 = delta_0 shows it is needed.) QED

**Proposition 6.13 (small noise need not give useful one-step contraction). PROVEN.** For a layer
h -> relu(W h) + tau xi, xi ~ N(0, I), tau > 0, the Dobrushin (total-variation) coefficient over a set S of states is
2 Phi(Delta_S/(2 tau)) - 1, Delta_S = sup_{h,h' in S} ||relu(W h) - relu(W h')|| (possibly infinite). It is below 1
when Delta_S is finite, but 1 - coefficient ~ (4 tau/(Delta_S sqrt(2 pi))) exp(-Delta_S^2/(8 tau^2)) when
Delta_S >> tau. So if the images of the relevant states are a distance of order sqrt(n) times the activation RMS apart
while tau = sigma times the RMS, one layer contracts total variation by at most a factor exponentially close to 1 in
n/sigma^2. This does not rule out contraction in other metrics or over several layers. That noise does give forgetting
on average, at depth about ln(1/sigma)/|lambda| in the ordered and chaotic phases and not by depth 64 at the critical
point for sigma <= 0.01, is our measurement (random-net-phases), not a consequence of this proposition.

### 6.3 Measured: ARC's cumulant closures against plain Monte Carlo along depth proportional to width (MEASURED)

`numerics/k2_scaling.py` (ARC's own code via work/arc-noisy/kprop_noisy.py, K = 2 "simple", average metric, sigma = 0;
truth by plain Monte Carlo with 2^15-2^17 paths, its 1/N floor subtracted; ARC's FLOP counter for K = 2, plain Monte
Carlo charged n(2n - 1) + n per layer per path). One net per seed of depth n/2, scored at every depth (truncation
convention). N_eq = number of Monte Carlo paths with the same cost as K = 2 (about 1.5 n); gain = (1/N_eq) / nMSE_K2,
summarized by its geometric mean over seeds with a bootstrap interval.

| alpha | n | depth | seeds | K=2 nMSE median | K=2 nMSE mean [95% CI] | MC 1/N_eq | gain = (1/N_eq)/nMSE: geo-mean [95% CI] | K=1 nMSE median |
|---|---|---|---|---|---|---|---|---|
| 1/8 | 32 | 4 | 16 | 2.02e-03 | 2.40e-03 [1.9e-03, 2.9e-03] | 1.72e-02 | 7.76 [6.40, 9.25] | 2.39e-02 |
| 1/8 | 64 | 8 | 16 | 2.60e-03 | 2.59e-03 [2.1e-03, 3.2e-03] | 9.47e-03 | 4.06 [3.25, 5.15] | 2.58e-02 |
| 1/8 | 128 | 16 | 16 | 1.87e-03 | 2.10e-03 [1.7e-03, 2.6e-03] | 4.97e-03 | 2.57 [2.11, 3.14] | 2.41e-02 |
| 1/8 | 256 | 32 | 16 | 1.36e-03 | 1.48e-03 [1.2e-03, 1.8e-03] | 2.54e-03 | 1.91 [1.53, 2.44] | 2.12e-02 |
| 1/8 | 512 | 64 | 16 | 9.51e-04 | 9.34e-04 [7.9e-04, 1.1e-03] | 1.29e-03 | 1.47 [1.24, 1.76] | 1.50e-02 |
| 1/8 | 1024 | 128 | 4 | 5.46e-04 | 6.42e-04 [4.2e-04, 9.6e-04] | 6.47e-04 | 1.09 [0.70, 1.55] | 9.32e-03 |
| 1/4 | 32 | 8 | 16 | 7.65e-03 | 8.15e-03 [6.8e-03, 9.6e-03] | 1.72e-02 | 2.25 [1.89, 2.66] | 5.96e-02 |
| 1/4 | 64 | 16 | 16 | 4.65e-03 | 6.81e-03 [4.6e-03, 9.1e-03] | 9.47e-03 | 1.78 [1.24, 2.59] | 4.59e-02 |
| 1/4 | 128 | 32 | 16 | 2.18e-03 | 2.92e-03 [2.1e-03, 4.1e-03] | 4.97e-03 | 2.01 [1.53, 2.55] | 4.09e-02 |
| 1/4 | 256 | 64 | 16 | 1.84e-03 | 2.21e-03 [1.6e-03, 2.9e-03] | 2.54e-03 | 1.34 [1.04, 1.73] | 2.10e-02 |
| 1/4 | 512 | 128 | 16 | 1.10e-03 | 1.22e-03 [8.8e-04, 1.6e-03] | 1.29e-03 | 1.27 [0.94, 1.76] | 1.25e-02 |
| 1/4 | 1024 | 256 | 4 | 1.24e-03 | 1.25e-03 [6.2e-04, 1.9e-03] | 6.47e-04 | 0.65 [0.35, 1.40] | 1.19e-02 |
| 1/2 | 32 | 16 | 16 | 1.87e-02 | 2.29e-02 [1.5e-02, 3.2e-02] | 1.72e-02 | 0.97 [0.69, 1.36] | 1.43e-01 |
| 1/2 | 64 | 32 | 16 | 8.68e-03 | 1.15e-02 [7.4e-03, 1.7e-02] | 9.47e-03 | 1.15 [0.76, 1.72] | 6.30e-02 |
| 1/2 | 128 | 64 | 16 | 2.41e-03 | 3.79e-03 [2.0e-03, 6.1e-03] | 4.97e-03 | 2.32 [1.35, 4.01] | 2.83e-02 |
| 1/2 | 256 | 128 | 16 | 2.08e-03 | 2.81e-03 [1.6e-03, 4.2e-03] | 2.54e-03 | 1.51 [0.87, 2.70] | 2.12e-02 |
| 1/2 | 512 | 256 | 16 | 1.62e-03 | 2.22e-03 [1.2e-03, 3.7e-03] | 1.29e-03 | 0.92 [0.58, 1.50] | 2.11e-02 |
| 1/2 | 1024 | 512 | 4 | 1.48e-03 | 1.48e-03 [5.9e-04, 2.4e-03] | 6.47e-04 | 0.68 [0.28, 2.31] | 9.37e-03 |

Fitted slopes of log nMSE_K2 against log n at fixed depth/width:
- alpha = 1/8: slope of log nMSE_K2 vs log n: median -0.41, mean -0.41 (plain MC at equal cost: -1; K = 2 keeps pace iff slope <= -1)
- alpha = 1/4: slope of log nMSE_K2 vs log n: median -0.56, mean -0.61 (plain MC at equal cost: -1; K = 2 keeps pace iff slope <= -1)
- alpha = 1/2: slope of log nMSE_K2 vs log n: median -0.74, mean -0.78 (plain MC at equal cost: -1; K = 2 keeps pace iff slope <= -1)

Read by depth instead of by depth/width ratio, the median over seeds of 1/gain = nMSE_K2 x N_eq (below 1: K = 2 wins):

| width n (seeds) | depth 4 | depth 8 | depth 16 | depth 32 | depth 64 | depth 128 | depth 256 | depth 512 |
|---|---|---|---|---|---|---|---|---|
| 32 (16) | 0.12 | 0.45 | 1.09 | - | - | - | - | - |
| 64 (16) | 0.06 | 0.27 | 0.49 | 0.92 | - | - | - | - |
| 128 (16) | 0.03 | 0.15 | 0.38 | 0.44 | 0.48 | - | - | - |
| 256 (16) | 0.02 | 0.08 | 0.29 | 0.53 | 0.72 | 0.82 | - | - |
| 512 (16) | 0.01 | 0.04 | 0.14 | 0.31 | 0.74 | 0.86 | 1.26 | - |
| 1024 (4) | 0.01 | 0.02 | 0.10 | 0.30 | 0.58 | 0.84 | 1.91 | 2.29 |

What this shows, with its limits:
- At fixed width, K = 2's loss ratio 1/gain grows about 4-5x per doubling of depth at shallow depth (error growing like
  depth^2, ARC's observation) and only about 1.2-2x per doubling beyond depth 64 (0.6-0.9 at depth 64-128, 1.3-2.3 at
  depth 256-512 for widths 512-1024). At fixed depth it improves with width, as ARC proved for fixed depth.
- Along L = alpha n (alpha = 1/8, 1/4, 1/2), K = 2 at its own (single, low) budget stays within a factor of about 3 of
  plain Monte Carlo, either way, up to n = 1024, but its advantage shrinks slowly with width: geometric means of 2-8 at
  widths 32-128 fall to 0.65-1.1 at width 1024 (4 seeds; intervals include 1). Fitted slopes of log nMSE against log n
  are -0.4 to -0.8, shallower than Monte Carlo's -1 at equal cost, so on these data K = 2 loses ground slowly.
- If ARC's conjectured rate c_2 (L/n)^2 were attained at depth proportional to width, K = 2 would lose to Monte Carlo
  by a factor growing like n. It does not, so that upper bound is far from tight there; the conjecture is about an upper
  bound, so this does not contradict it.
- Limits: plain Monte Carlo, not the best black-box sampler (Kerdock designs are about 2.4x better at 256 x 32); ARC's
  discounted FLOP count (raw counts are about 1.3x higher); 16 seeds per width up to 512 and few at 1024, with
  heavy-tailed per-net errors; sigma = 0; one budget point per method.

**The next closure up the hierarchy.** `numerics/k3f_scaling.py` runs ARC's factorized K = 3 (and K = 2 for reference) on fresh nets at depth n/8 and n/4, with exact FLOP counts for each depth:

| n | depth (alpha) | seeds | method | nMSE median | N_eq (MC samples at equal cost) | gain over MC: geo-mean [95% CI] |
|---|---|---|---|---|---|---|
| 32 | 4 (1/8) | 8 | k2 | 1.91e-03 | 58 | 8.28 [6.25, 10.59] |
| 32 | 4 (1/8) | 8 | k3f | 4.95e-04 | 826 | 2.16 [1.21, 3.40] |
| 32 | 8 (1/4) | 8 | k2 | 7.87e-03 | 58 | 2.44 [1.91, 3.16] |
| 32 | 8 (1/4) | 8 | k3f | 1.29e-02 | 1629 | 0.03 [0.01, 0.06] |
| 64 | 8 (1/8) | 8 | k2 | 3.06e-03 | 106 | 3.40 [2.49, 4.92] |
| 64 | 8 (1/8) | 8 | k3f | 4.23e-04 | 3114 | 0.80 [0.59, 1.09] |
| 64 | 16 (1/4) | 8 | k2 | 4.12e-03 | 106 | 2.06 [1.25, 3.43] |
| 64 | 16 (1/4) | 8 | k3f | 2.94e-03 | 6251 | 0.04 [0.02, 0.08] |
| 128 | 16 (1/8) | 8 | k2 | 1.71e-03 | 201 | 2.86 [2.00, 3.94] |
| 128 | 16 (1/8) | 8 | k3f | 2.35e-04 | 12297 | 0.38 [0.26, 0.65] |
| 128 | 32 (1/4) | 8 | k2 | 1.85e-03 | 201 | 2.57 [2.03, 3.21] |
| 128 | 32 (1/4) | 8 | k3f | 3.79e-03 | 24714 | 0.01 [0.01, 0.02] |
| 256 | 32 (1/8) | 3 | k2 | 1.01e-03 | 393 | 2.50 [1.12, 5.54] |
| 256 | 32 (1/8) | 3 | k3f | 3.17e-04 | 49097 | 0.08 [0.06, 0.11] |
| 256 | 64 (1/4) | 3 | k2 | 1.86e-03 | 393 | 1.55 [0.99, 2.74] |
| 256 | 64 (1/4) | 3 | k3f | 2.53e-03 | 98505 | 0.00 [0.00, 0.01] |

Factorized K = 3 costs 14-251 times more than K = 2 here. At depth = width/8 its gain over plain Monte Carlo falls with width (2.16 at width 32, 0.80 at width 64, 0.38 at width 128, 0.08 at width 256 (3 seeds)); at depth = width/4 it loses by a factor 27-247 (0.031 at width 32, 0.038 at width 64, 0.011 at width 128, 0.004 at width 256), and its error is not even below K = 2's. So the hierarchy's higher-budget points fall behind sampling at depth proportional to width, increasingly with width, which matches what ARC and the contest saw at 256 x 32 (factorized K = 3 about 11 times behind plain Monte Carlo there, our arc-noisy runs).

### 6.4 Conjectures

**Conjecture 6.A (bounded advantage of fixed-order closures at criticality).** For each fixed K and alpha > 0, ARC's
order-K cumulant propagation on He-ReLU nets with L = alpha n has advantage over plain Monte Carlo at equal cost that
stays bounded as n grows, and in fact tends to 0 (fast for factorized K = 3, slowly if at all for K = 2). Evidence: 6.3
(K = 2's advantage falls from 2-8 to about 0.7-1.1 between widths 32 and 1024; factorized K = 3 falls with width at
alpha = 1/8 and loses 27-247x at alpha = 1/4); the mechanisms of 6.6-6.11 and 7.2.
Status: CONJECTURE (two to six widths per cell, one budget per method).

**Conjecture 6.B (no layer-local estimator asymptotically beats sampling at criticality).** Any estimator that
propagates, layer by layer, a description of size n^{O(1)} computed by a polynomial-time map from the previous
description and the current layer's weights, and never evaluates f_W at a point, has advantage over sampling bounded
by a constant C(alpha) at L = alpha n, at every budget in a polynomial window. Status: CONJECTURE. A proof would need a
lower bound on the local errors in the non-contracting directions (6.6) for every polynomial-size description, which
is beyond current techniques (it would in particular separate this class from sampling in a fine-grained sense).

**ARC's alternative (for the record).** An estimator matching sampling at every budget exists for random MLPs at
L = alpha n. Nothing here refutes it; Conjecture 6.B allows it (matching is advantage of order 1), and 6.3 shows that
at the lowest budget the cheapest closure already roughly matches plain Monte Carlo. The open part is the higher
budgets, where the existing hierarchy falls behind (6.3) and a different kind of description would be needed.

## 7. The low-degree method and pseudorandomness (questions 3b and 3c)

**Proposition 7.1 (the low-degree method cannot give hardness evidence here). PROVEN (logical).** The low-degree method
(Hopkins; Kunisky, Wein and Bandeira 2019, arXiv 1907.11636) predicts that a planted-versus-null testing or estimation
problem needs time exp(Omega(D)) when degree-D polynomials of the observations fail. Here (i) the target mu(W) is a
deterministic function of the observation W, computable to inverse-polynomial precision in randomized polynomial time
(Proposition 3.1), so there is no polynomial-versus-superpolynomial threshold for the method to predict; and (ii) the
natural efficient algorithms are not low-degree polynomials of W: even one forward pass composes L layers. A failure of
degree-D polynomials would show only that a restricted class fails (as in Section 6), not that efficient algorithms do;
a success would not give an efficient algorithm either, since a degree-D polynomial in n^2 L variables has up to
(n^2 L)^D terms. So the answer to 3b as posed is that the low-degree test is the wrong instrument; its fine-grained
analogue (which polynomial-size computations reach which error) is exactly the open problem.

**Proposition 7.2 (how non-additive the output scale is across layers). PROVEN (exact).** With e_l = rho_l - 1
(Lemma 6.1), G_L = sum over subsets S of layers of prod_{l in S} e_l, and the terms are orthogonal (Lemma 6.10 with
rho-hat = 1) with E (prod_{l in S} e_l)^2 = (5/n)^{|S|}. The fraction of Var(G_L) carried by products of exactly s of
the gain factors is C(L, s)(5/n)^s / ((1 + 5/n)^L - 1), which for L = alpha n tends to the Poisson(5 alpha) law
conditioned on s >= 1. Truncating at first order (linear response around mean field) keeps the fraction
5 alpha/(e^{5 alpha} - 1) of a single input's output-scale fluctuation: 50% at alpha = 1/4, 3% at alpha = 1; truncating
at order s keeps the Poisson(5 alpha) mass up to s (orders 1-5 keep 99.7% at alpha = 1/4 and 61% at alpha = 1). So a fixed-order expansion of this kind loses a
constant fraction once L/n is of order 1 and the truncation order is below about 5 alpha. This concerns the expansion in
the gain factors (each of which depends on all earlier weights), not arbitrary descriptions that use few layers.

**Pseudorandomness (3c). PROVEN statements plus a reading.**
- Bias-free He-ReLU nets are not pseudorandom functions in any cryptographic sense: f_W(2x) = 2 f_W(x) is an efficient
  distinguisher. Their output scale at independent inputs is correlated over W, with correlation bounded below at depth
  proportional to width (Lemma 6.4, a statement about the quadratic observable; for wide nets, Dunbar and Aaronson's
  criterion says outputs are correlated because ReLU has nonzero Gaussian mean).
- If outputs were nearly independent across inputs, means would be easy (Proposition 3.4, Corollary 3.3). Provable
  pseudorandomness of random circuits exists (He and O'Donnell 2024, arXiv 2404.14648: random reversible circuits are
  almost k-wise independent), and it makes mean estimation trivial, not hard.
- So "deep critical nets act like PRFs" cannot be the mechanism of any hardness. Hardness, if it exists, needs
  structured long-range dependence that is not efficiently computable to the required precision. Candidates from
  Section 6: the shared, multiplicatively fluctuating output scale (6.4, 7.2), a collapsed state whose couplings
  dominate the next layer (6.11), and no contraction of the scale mode (6.7) with only slow decay of others (heuristic
  6.9'). None of them is pseudorandom; whether they can be computed to the precision sampling reaches is the open
  question.

## 8. What is not ruled out

| Route | Status after this note |
|---|---|
| ARC's MSP with advice (pi from theta, c random) | Not refuted by any result here. Worst-case barriers are sidestepped by advice (4.4); in the multiscale random-oracle families a fixed-subset average matches sampling and nothing beats it by more than polylog factors over the window (5.1); for random MLPs only a formal model of "mechanistic" could be refuted (3.5) |
| Mechanistic estimators for random He-ReLU at L proportional to n | Open. Proven to fail, each for a specific observable: estimators that ignore the first constant fraction of the layers (6.5, mean squared output norm); per-layer surrogates of fixed accuracy along the true trajectory (6.10, one input's output scale); fixed-order expansions in the gain factors (7.2, same). Shown to misjudge the next layer's variance by O(1): independent-neuron closures once the state has collapsed (6.11). Measured to fail: ignoring the last layers. No transfer of these single-observable results to ARC's per-neuron means is proven. Not ruled out: descriptions that keep the full coupling structure, track the scale exactly, and keep their accumulated local errors small; non-local (multi-layer or backward-informed) descriptions |
| Trained networks with structure | A trainer can embed circuits, so the worst-case barriers apply to adversarially trained DAG networks (4.1); train-and-explain gives the explainer the trainer's computation, which can carry the structure. Random-net results do not transfer (trained weights are not rotation invariant). Nothing here rules out efficient estimation for trained nets with structure |
| Noisy networks | Worst case: the circuit reduction survives small absolute noise (4.1c), so the barriers of 4.2-4.3 hold for noisy DAG networks with sigma = O(1/sqrt(log size)); noise proportional to each layer's RMS is not covered. Average case: one noisy layer need not contract total variation usefully when the states' images are far apart (6.13); in our measurements noise gives forgetting in the ordered and chaotic phases (depth about ln(1/sigma)/abs(lambda)) and at the critical point only at sigma around 0.1, where K = 2's error stopped growing. If such forgetting holds in the sense of Theorem 6.12's hypothesis, layer-local errors become depth-free; that hypothesis is not proven for these networks |
| Narrow networks, ARC's literal reading (n fixed, eps -> 0) | Easy for fixed depth: the net has finitely many linear regions (at most 2^{nL} activation patterns), each a polyhedron in fixed dimension n where the Gaussian integral of a linear function can be computed deterministically to precision eps in polylog(1/eps) operations (sketch: triangulate, then use high-order cubature on smooth integrands), so for small enough eps exact integration beats sampling. The cost is exponential in L, so the meaningful question is uniform in depth (n fixed, L -> infinity), which remains open |
| Quantum estimators | Beat sampling quadratically in the query model with coherent access (Montanaro 2015); outside the classical question |

## 9. What this means for M1 and for interpretable architectures

M1 (a sample-free estimator matching sampling on deep random MLPs at depths that grow with width):
1. *No impossibility proof is available, and none is possible without formalizing "sample-free".* On average over W a
   hard-coded input design (nonuniformly chosen) is deterministic and matches sampling (3.5), and the worst-case
   barriers (4.2, 4.3) do not apply to random nets. Any negative result for M1 must be about a formal class of
   descriptions.
2. *What the theorems say a layer-local description must get right, with the conditions under which each is proven.*
   It must use the early layers (6.5, for the mean squared output norm; the late layers too, by measurement). Its
   errors in the output scale do not shrink with distance to the output (6.7), so if they are independent and centered
   with relative variance v per layer they add up to about L v; matching sampling then needs v of order eps^2/L (6.7,
   6.10). Independent-neuron closures misjudge the next layer once the state has collapsed (6.11). The output error is
   the signed sum of local errors tested against the true backward functions (6.6); under the unproven transport
   heuristic 6.9' the mean-square error from constant local errors e grows like e^2 L/4. Fixed-order expansions in the
   gain factors lose a constant fraction at L proportional to n (7.2). None of these is a proof that a successful
   family cannot exist.
3. *Empirical status of the cheapest layer-local estimator (ARC's K = 2) along L proportional to n (6.3):* at its own
   budget it stays within a factor of about 3 of plain Monte Carlo, either way, at every depth and width tested, but
   its advantage shrinks slowly with width; the higher-budget point of the hierarchy (factorized K = 3) falls behind
   fast (6.3).

Interpretable architectures:
1. *Worst case.* Any architecture that can embed Boolean circuits as ReLU DAG networks of near-linear size (with or
   without biases, and with small absolute noise) inherits the barriers: no deterministic estimator matches sampling on
   all of its instances under #NSETH, and proving that a subexponential deterministic one exists would prove NEXP not
   in P/poly. For layered MLPs the carried wires change the size accounting, so the fine-grained version needs a
   separate argument there. "Interpretable by design" helps only if it restricts expressivity enough to block such
   embeddings, e.g. bounded treewidth or probabilistic circuits with exact moments (see the lit-tangledness note for
   what those cost).
2. *Average case.* The theory separates two kinds of depth. If the backward functions contract (Theorem 6.12's
   hypothesis), layer-local errors are depth-free; our measurements suggest something like this in the chaotic phase
   and with large noise, but it is not proven for those networks. Without contraction (homogeneous activations at
   criticality, exactly ARC's family) the scale mode never contracts (6.7) and, heuristically, other errors decay only
   polynomially (6.9'). A design target that follows: architectures whose layers provably contract the statistics a
   description tracks. The tension is that trainability of deep plain nets needs criticality (signal propagation
   theory), and residual streams preserve information by design; so the depth problem is likely generic for practical
   deep architectures unless descriptions are non-local. This last sentence is a reading (CONJECTURE), not a theorem.

## 10. Reviews and what changed

Three independent audits of the proofs, each asked to break every numbered statement (files in `review/`):
- Gemini 4 Argon (agy, Sections 1-6, no tools): no fatal error; six fixable gaps. Fixed: noise constant in 4.1(c)
  (a restored wire carries 2 tau, so tau = 1/24 and exp(-1/(1152 sigma^2))); the Levy distance in 3.2 (2 delta, not
  delta); the label of the accumulation corollary; a zero-variance member in 2.3 (now psi is added); variance padding in
  4.3; missing hypotheses in 5.1(c) (nonnegative weights, uniform cells); 5.2 counting the trainer's own oracle calls.
- GPT-6 (codex, xhigh) on Sections 1-5. Retracted or narrowed: the claim that black-box lower bounds must stop near
  the parameter count (false: a one-trapezoid, 4-unit family keeps the factor-4 bound for every N; precision, not
  parameter count, is what grows); "advice independent of the context carries no information" (false: unbounded advice
  can tabulate every context; now stated for oracle-independent or charged advice); the per-scale summation assumption
  (a direct sum does not follow; replaced by a joint assumption JSH); the rotation hypothesis in 3.5 (must be joint);
  "deterministic is free" now separates nonuniform existence from explicit designs; the relative-RMS noise extension of
  4.1(c) (removed); the 3/16 variance claim in 4.3 (new 3-bit padding, variance >= 1/5 after smoothing); the
  #NSETH counting reduction and fast multipoint evaluation are now written out; the interpretation of Section 5 is
  limited to the families proven. Constants confirmed: 4(1 - 1/(2N)), 5.9/(1 - 3.3 w'), R(Q), 4 J^2, J(J-1)/4, J/4.
- GPT-6 (codex, xhigh) on Section 6. Fixed: Theorem 6.5 needed A independent of the pair (B, xi) (counterexample with
  xi alone independent); its "1-3 samples" is now a measured upper bound, the proven ceiling is a crude constant, and S
  is identified as a quadratic observable; Lemma 6.4's remainder (exact M_4(n), sharp exponent) and indexing; the
  extinction convention for cosines; Lemma 6.6's "iff" (errors can cancel); Proposition 6.7's O(sigma^2) noise claim
  (false: O(sigma) near kinks) and its variance formula; Proposition 6.8's hypotheses (Q({0}) = 0, translation of every
  state, population versus empirical support); Corollary 6.9 split into a proven angle result and an unproven
  transport heuristic; Theorem 6.12's missing initial-error term (counterexample T(h) = eta relu(h)); Proposition 6.13's
  "never" (finite image diameter does give contraction, only exponentially weak). Confirmed: Lemmas 6.1-6.3, the
  covariance-growth mechanism, Lemma 6.10 (with a shorter proof), Proposition 6.11, the r_D values.
- GPT-6 (codex, high) re-check of the fixes: confirmed most repairs and found four remaining problems, all fixed: an explicit-design claim in 3.5 (a fixed design need not match Monte Carlo; only nonuniform existence and rotation equivalence remain), a degree bound in 4.3 (now O(nd)), overstatements in Sections 7-9 (each restricted-class result is now tied to its observable), and an unproven extension of 6.5 to estimators that also see cheap statistics of the early layers (removed).

## 11. Sources (verified)

Read in the paper text (local copies in `papers/`): Impagliazzo, Kabanets, Wigderson, "In search of an easy witness",
JCSS 65(4) 2002, Section 5.1 (Theorem 32, Remark 33, Corollary 34); Chen, Tell, "Simple and fast derandomization from
very hard functions", STOC 2021 / ECCC TR20-148, Sections 1.2 and 6 (Theorem 6.4, Assumption 6.1, Lemma 6.3); Wu,
Lecomte, Winer, Robinson, Hilton, Christiano, arXiv 2605.05179v2, Sections 5, 7.3 and Appendix D. ARC's "Competing
with sampling" (Neyman et al., 18 Nov 2025) read through a fetch tool. Abstracts or records checked via the arXiv API:
Hanin 2018 (arXiv 1801.03744); Hanin and Nica (arXiv 1812.05994); Goldwasser, Kim, Vaikuntanathan, Zamir
(arXiv 2204.06974); Dunbar and Aaronson (arXiv 2510.06527); He and O'Donnell (arXiv 2404.14648); Kunisky, Wein, Bandeira
(arXiv 1907.11636); Christiano, Neyman, Xu (arXiv 2211.06738); Rolnick and Kording (arXiv 1910.00744); Carlini,
Jagielski, Mironov (arXiv 2003.04884); Montanaro (arXiv 1504.06987); Poole et al. (arXiv 1606.05340); Schoenholz et al.
(arXiv 1611.01232); Polyanskiy and Wu (arXiv 1508.06025); Daniely and Vardi (arXiv 2006.03177); Chen, Gollakota,
Klivans, Meka (arXiv 2202.05258); Bartlett, Harvey, Liaw, Mehrabian (arXiv 1703.02930). Naor and Reingold, J. ACM 51(2)
2004 (PRFs in TC^0) via the publisher record. Not used because not verified: Williams 2010/2013 on CAPP speedups.
Related in-house notes: `outbox/arc-deep-networks-2026-10-29.md` (ARC's depth results), `random-net-phases-2026-10-29.md`
(phase measurements used in Section 6), `lit-tangledness-architectures-2026-10-29.md` (the parallel literature study:
worst-case NP/#P results, backdoors, by-design architectures; not repeated here).

## 12. Reproduction

All CPU, `nice -n 19`, Python at `/home/agent/.venvs/arc-noisy/bin/python`, from `work/hardness/numerics/`:
- `lemma_checks.py J | norm | backward n L m R | survival n depth | hermite n L D | fwdtrunc n L k` (logs in `logs/`).
- `capp_check.py 6` (reduction check, `logs/check_capp.txt`).
- `run_k2s.sh`, `run_k2s_b.sh` (K = 2 sweep; `results/k2s_*.json`), `analyze_k2s.py` (tables in 6.3).
- `run_k3f.sh` (factorized K = 3; `results/k3f_*.json`), `analyze_k3f.py`.
ARC's code is imported read-only from `work/arc-noisy/ext/mlp_cumulant_propagation` via `work/arc-noisy/kprop_noisy.py`.
