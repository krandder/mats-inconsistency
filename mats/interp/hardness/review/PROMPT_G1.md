You are an independent adversarial referee (theoretical computer science, probability, random matrix theory). Below is
a draft research note (Sections 1-6). Do NOT use tools; reason from the text. Try to BREAK every numbered statement
(Theorem/Lemma/Proposition/Corollary). For each, give a verdict: CORRECT, CORRECT WITH A FIXABLE GAP (say exactly what to
change), or WRONG (counterexample or failing step). Be concrete and check arithmetic and constants. Priorities:
(1) Theorem 4.1 (circuit -> ReLU reduction: gates, invariants, bias-free and noisy variants) and Corollaries 4.2-4.3
(correct use of Impagliazzo-Kabanets-Wigderson Thm 32 and of Chen-Tell's #NSETH Theorem 6.4 proof);
(2) Theorem 5.1 (random oracle bounds, especially (b) and (c)); (3) Theorem 2.1 and 2.3; (4) Propositions 3.2-3.5;
(5) Section 6: Lemma 6.4 (stopped product), Theorem 6.5 (ANOVA lower bound), Proposition 6.8, Corollary 6.9 asymptotics,
Lemma 6.10 (martingale orthogonality), Proposition 6.11. Also flag anything overclaimed or mislabelled
(PROVEN vs conditional). End with a list of required fixes ordered by severity.

===== DRAFT =====
# DRAFT (work/hardness) — formal sections

## 1. The problem, formally

### 1.1 Networks, inputs, estimand
- A network is a feedforward ReLU network f_W: R^d -> R. The MLP case: W = (W_1, ..., W_L, v), W_1 in R^{n x d},
  W_l in R^{n x n}, h_0 = x, h_l = relu(W_l h_{l-1} + b_l), f_W(x) = v . h_L (or the vector h_L, scored coordinatewise).
  "He-ReLU" (ARC's family): d = n, b_l = 0, W_l iid N(0, 2/n) entries. "Width" n, "depth" L. "Size" E = number of
  nonzero weights; one forward pass costs c_fwd = Theta(E) arithmetic operations (2 n^2 L for a dense MLP).
  Where general DAG networks are needed (Section 4) they are ReLU units wired as any directed acyclic graph.
- Noisy networks: h_l = relu(W_l h_{l-1}) + sigma r_l xi_l, xi_l ~ N(0, I) fresh at every layer, r_l a fixed scale.
- Input law D: N(0, I_d) (ARC), or uniform on {0,1}^d.
- Estimand: mu(W) = E_{x~D}[f_W(x)] (for noisy nets also over the noise). Per-input variance V(W) = Var_{x~D} f_W(x).

### 1.2 Estimators and access models
- (BQ) black-box query: the algorithm chooses inputs (adaptively, possibly at random) and sees f_W there; cost = number
  N of queries times c_fwd; other computation is free. (BS) black-box sample: it only sees iid pairs (x_i, f_W(x_i)).
- (WB) white-box: the algorithm receives W (rational entries) and may compute anything; cost = its running time
  (arithmetic operations for the average-case statements, bit operations for the complexity-theoretic ones).
- (WB+A) white-box with advice: as WB, plus a string pi produced by an "explainer" (Section 1.5 states ARC's rules).
- Deterministic vs randomized: a deterministic estimator uses no random bits; a randomized one may.

### 1.3 Error
- Fixed network: MSE(A, W) = E_coins (A(W) - mu(W))^2; normalized nMSE = MSE / V(W) (ARC's metric).
- Random network W ~ P: E_W MSE(A, W), compared with E_W V(W) (ARC's MSP uses exactly this pair, see 1.5).
- Worst case over a class F: sup_{W in F} MSE(A, W) / V(W).

### 1.4 The sampling baseline and what "better than sampling" means
- Monte Carlo with N forward passes: MSE = V(W)/N exactly, at cost N c_fwd (plus drawing inputs). At budget T it scores
  MSE_MC(T) = V(W) c_fwd / T.
- Advantage of an estimator family {A_T} at budget T: Adv(T) = MSE_MC(T) / MSE(A_T) (worst case: the infimum over the
  class of the per-network ratio; average case: E_W V c_fwd / T divided by E_W MSE).
- A family *matches sampling with constant C on a budget window [T_0, T_1]* if Adv(T) >= 1/C for every T in the window;
  it *beats sampling by G* if Adv(T) >= G on the window. Along a sequence of problem sizes (n -> infinity, L = L(n)),
  it *asymptotically beats sampling* if Adv(T(n)) -> infinity for budgets T(n) in a polynomial window [n^a, n^b].
- Why a window. For tiny tolerances exact computation can win for trivial reasons (e.g. for fixed width n and depth L
  the network is a fixed piecewise-linear map with finitely many regions, and its Gaussian mean can be computed to
  precision eps at a cost that grows only polylogarithmically in 1/eps, Section 8), and for budgets below one forward
  pass nothing is possible. The interesting statements are about polynomial windows, uniformly in the window.
- Stronger samplers (RQMC, Kerdock/spherical designs) change MSE_MC by constant factors on ARC's nets (0.4/N at 256 x 32
  in our work/arc-noisy); Section 2 shows that no black-box design can do more than a constant factor in the worst case.

### 1.5 ARC's matching sampling principle (MSP), in these terms
Quoted from "Competing with sampling" (Neyman, Lecomte, Wu, Winer, Hilton, Robinson, ARC, 18 Nov 2025):
- an architecture M maps (c, x) to R, with c a random "context" and x the input; parameters theta are worst case;
- an estimator G_M(theta, pi, c, eps), with advice pi depending on theta only, |pi| <= O(|theta|);
- for all eps: time O((1/eps^2) Time(M_theta)), error E_c[(G - E_x M_theta(c, x))^2] <= eps^2 E_c[Var_x M_theta(c, x)],
  and G is "mechanistic" (informal; working test: "avoids any random or pseudorandom sampling");
- train-and-explain variant: theta = T(s) for a trainer T with seed s, pi = E(s) with Time(E) <= O(Time(T));
- the random-MLP instance: theta empty, c = W (iid Gaussian), x ~ N(0, I_n), depth a fixed constant, n -> infinity,
  1/eps = poly(n). So for random MLPs the MSP is an advice-free, average-case statement over W.
In the notation above, the MSP asks for Adv(T) >= 1/C at every budget T >= c_fwd, for a mechanistic estimator.
Kelvin's question ("better than random sampling, as deep and wide as needed") asks whether Adv can be made >= 1, or
-> infinity, as n and L grow; ARC's open problem (Wu et al. 2026, Sec. 7.3) is the case L -> infinity, in particular
L proportional to n.

### 1.6 Assumptions, and how each is checked
| assumption | role | check |
|---|---|---|
| A1 cost = arithmetic operations; FLOP discounts (ARC's symmetric-tensor counting) change constants only | all comparisons | stated; constants never drive a claim |
| A2 weights are exact rationals of poly(n) bits (complexity statements) | Sec 4 | the reduction uses weights in {0, +-1, +-K} |
| A3 estimators may be randomized unless stated | Sec 2-6 | each theorem says which class |
| A4 He-ReLU nets are bias-free with iid N(0, 2/n) weights and Gaussian input | Sec 6 | as in ARC's paper and contest |
| A5 #NSETH (Carmosino et al.; counting version as in Chen-Tell Assumption 6.1) | Cor 4.3 only | named in the statement |
| A6 random oracle model | Sec 5 only | named; instantiation is a separate, explicit assumption |

## 2. Black-box estimators: sampling is optimal up to a constant factor

**Theorem 2.1 (query lower bound). PROVEN.** Fix N >= 1 and m >= N. Let J_1, ..., J_m be disjoint intervals and
g_1, ..., g_m functions with 0 <= g_j <= 1 and g_j = 0 outside J_j. Put a_j = E g_j(x_1), b_j = E g_j(x_1)^2 under the
input law D, and f_s(x) = sum_j s_j g_j(x_1) for s in {-1, +1}^m. Draw s uniformly. Then every algorithm that makes at
most N queries of f_s (adaptive, randomized, unbounded computation) outputs mu-hat with

  E (mu-hat - mu(f_s))^2 >= min_{|R| <= N} sum_{j not in R} a_j^2 >= (m - N) min_j a_j^2,

while Monte Carlo with N queries has E_s MSE = sum_j (b_j - a_j^2) / N.

*Proof.* A query x returns s_j g_j(x_1) for the unique j with x_1 in J_j (or 0). By induction on the queries, the
transcript (queries, answers, coins) depends on s only through s_R, where R is the random set of indices whose interval
was hit, |R| <= N. Given the transcript, the s_j with j not in R are therefore still iid uniform, so
E[mu | transcript] = sum_{j in R} s_j a_j and Var(mu | transcript) = sum_{j not in R} a_j^2. Any estimator is a function
of the transcript, so its conditional MSE is at least this conditional variance; average over transcripts. For Monte
Carlo, E_s Var_x f_s = sum_j b_j - E_s (sum_j s_j a_j)^2 = sum_j (b_j - a_j^2) (disjoint supports). QED

**Corollary 2.2 (ReLU realization; constants).** Take m = 2N, J_j equal-mass subintervals and g_j trapezoids
g_j(t) = [relu(t - al) - relu(t - al - w) - relu(t - be + w) + relu(t - be)] / w on J_j = [al, be] (4 units each). Each
f_s is a one-hidden-layer ReLU network with 8N units. For x_1 uniform on [0, 1] and w -> 0, the ratio of Monte Carlo's
MSE to the best possible MSE tends to 4 (1 - 1/(2N)): no black-box algorithm beats Monte Carlo by more than a factor
of 4 on this family at budget N. For Gaussian input with the J_j partitioning [-1, 1] (mass 0.683) the same computation
gives a factor <= 5.9 / (1 - 3.3 w')^2 (w' = ramp fraction). The family embeds in width-n, depth-L MLPs whenever
(n - 3) L >= 8 m roughly (bumps computed in a pipeline, an accumulator unit carries the running sum), and in bias-free
(positively homogeneous) nets with one extra layer, using f_s(x) = |x_2| sum_j s_j g_j(x_1 / |x_2|).

**Theorem 2.3 (deterministic black-box algorithms fail). PROVEN.** For any deterministic adaptive algorithm with N
queries there are two members of the family (coefficients in {0, 1}, m = 2N) on which its errors differ by at least
sum_{j not in R_0} a_j >= N min_j a_j, where R_0 is the set of intervals hit when it is run on f = 0. For the uniform
version this is >= 1/2 - o(1), so its worst-case squared error is >= 1/16 - o(1), while Monte Carlo has MSE <= 1/(2N)
on both: deterministic black-box algorithms lose by a factor >= N/8.
*Proof.* Run it on f_0 = 0: it makes a fixed query sequence and outputs mu-hat_0. f' = sum_{j not in R_0} g_j vanishes
on every query, so the run on f' is identical and also outputs mu-hat_0, while mu(f') - mu(f_0) = sum_{j not in R_0} a_j.
QED

**Remarks.**
- The sample model (iid pairs, no choice of inputs) is weaker than the query model, so Theorem 2.1 holds there too.
- Range of validity. The hard family needs about 8N units. Far above the parameter count, query algorithms can recover
  a generic network from its outputs (Rolnick and Kording 2020; Carlini, Jagielski and Mironov 2020) and then compute
  as white-box algorithms, so no black-box lower bound can hold much beyond the parameter count. In ARC's 256 x 32
  contest (2.1M weights, about 6.5e4 forward passes of budget) the budget is below the parameter count, but the explicit
  family above covers only N up to about (n - 3) L / 8, roughly 1,000 there; closing that gap needs a bit-extraction
  construction, not done here.
- This is a classical statement. Quantum amplitude estimation reaches error eps with O(1/eps) black-box queries
  (Montanaro 2015, arXiv 1504.06987).
- Structured samplers (QMC, spherical designs, Kerdock) are black-box algorithms: by 2.1 they gain at most constant
  factors in the worst case, which matches the 2.4-3x seen on ARC's nets.

## 3. What cannot be proven, and what "mechanistic" has to mean

**Proposition 3.1 (no coarse hardness). PROVEN.** For networks with outputs in [a, b], Monte Carlo reaches MSE
(b - a)^2 / (4N) at cost N c_fwd, and |mu-hat - mu| <= eps with probability 1 - delta at cost
O(c_fwd (b - a)^2 log(1/delta) / eps^2). So estimating mu(W) to inverse-polynomial precision is in randomized polynomial
time for every class of polynomial-size networks with polynomially bounded outputs. Consequently no super-polynomial
lower bound can hold for randomized estimators, and every statement of the form "one cannot beat sampling" must be
fine-grained (about exponents or constant factors of polynomial costs) or about a restricted class of estimators
(black-box, deterministic, or a formal model of "mechanistic").

**Proposition 3.2 (indistinguishable network laws have nearly equal mean laws). PROVEN.** Let P_0, P_1 be laws over
networks of size E with outputs in [0, 1], and suppose no white-box distinguisher running in time t has advantage more
than adv. Let N = ceil(ln(2/eta) / (2 delta^2)) and suppose t >= N c_fwd + O(N d). Then the laws of mu(W) under P_0
and P_1 are within Levy distance delta + eta + adv.
*Proof.* For a threshold tau, the test D(W) = 1[N-sample mean > tau] accepts with probability >= 1 - eta/2 when
mu(W) >= tau + delta and <= eta/2 when mu(W) <= tau - delta (Hoeffding). Hence
P_0(mu >= tau + delta)(1 - eta/2) <= P_0(D = 1) <= P_1(D = 1) + adv <= P_1(mu > tau - delta) + eta/2 + adv, and
symmetrically. QED
*Consequence.* Planting structure that white-box distinguishers cannot detect (for instance the backdoors of
Goldwasser, Kim, Vaikuntanathan and Zamir 2022, arXiv 2204.06974, which are white-box undetectable for random ReLU
networks) cannot move the law of the mean by more than inverse-polynomial amounts. Undetectable planting is not a
source of mean-estimation hardness.

**Corollary 3.3 (pseudorandom functions have predictable means). PROVEN, given PRF security.** Let {F_k} be a
pseudorandom function family {0,1}^m -> {0,1} (for example the Naor-Reingold functions, which are in TC^0 and so are
realized by constant-depth ReLU networks of polynomial size), and f_k(x) = F_k(sign bits of x). For every polynomial q,
Pr_k(|mu(f_k) - 1/2| > 1/q(m)) is negligible. So the constant estimate 1/2 beats Monte Carlo at every polynomial budget
for all but a negligible fraction of keys.
*Proof.* For a truly random function, |mu - 1/2| <= 2^{-m/2} sqrt(ln(2/eta)/2) with probability 1 - eta. If the
event had non-negligible probability gamma for F_k, the O(q^2 log(1/eta))-sample test "is the empirical mean farther
than 1/(2q) from 1/2" would distinguish F_k from a random function with advantage about gamma - 2 eta. QED

**Proposition 3.4 (when the "know nothing" estimate wins). PROVEN.** For a random network W ~ P, let
m(x) = E_W f_W(x) and K(x, x') = Cov_W(f_W(x), f_W(x')), with x, x' independent draws from D. The annealed estimate
mu-bar = E_W mu(W) has E_W (mu-bar - mu)^2 = E_{x,x'} K(x, x'), while Monte Carlo with N samples has
E_W MSE = [E_x K(x, x) + Var_x m(x) - E_{x,x'} K(x, x')] / N. So the annealed estimate beats N-sample Monte Carlo iff
(N + 1) E_{x,x'} K < E_x K(x, x) + Var_x m.
*Proof.* Var_W(E_x f_W) = E_{x,x'} Cov_W(f_W(x), f_W(x')) by Fubini; E_W Var_x f_W = E_x E_W f^2 - E_W mu^2 =
E_x [K(x,x) + m(x)^2] - [Var_W mu + (E_x m)^2]. QED
*Consequences.* (a) Families whose outputs at independent inputs are nearly uncorrelated over the random weights
(pseudorandom functions; wide random nets with zero-Gaussian-mean activations, Dunbar and Aaronson,
arXiv 2510.06527) have predictable means up to budgets N of about 1 / (average correlation). (b) A family can be hard to
estimate at budget N only if the average correlation of its outputs at independent inputs is at least about 1/N: hardness
needs long-range structure, not randomness. (c) Bias-free He-ReLU nets are the opposite extreme (Lemma 6.4 below: the
correlation of the output scale at two inputs stays bounded below); for them the annealed estimate is worth only a few
samples, and any difficulty lies in structured, weight-dependent parts of the mean.

**Proposition 3.5 (determinism is free on average). PROVEN.** For any prior P over networks and any randomized
estimator A with E_{W, coins} (A - mu)^2 <= Delta there is a fixed coin string r* with E_W (A_{r*}(W) - mu(W))^2 <= Delta.
In particular some fixed input set x_1..x_N, chosen without looking at W, has
E_W ((1/N) sum_i f_W(x_i) - mu(W))^2 <= E_W V(W) / N. If moreover W_1 Q has the same law as W_1 for every orthogonal Q
(iid Gaussian first layer) and D = N(0, I), every fixed design X has exactly the average error of its random rotation QX.
*Proof.* Averaging over coins. For the rotation claim: f_W(Q x) = f_{W'}(x) with W'_1 = W_1 Q, W' has the law of W,
and mu(W') = mu(W) by rotation invariance of D. QED
*Consequence.* In ARC's random-MLP instance of the MSP (average over W, no advice) the requirement that the estimator
be deterministic excludes nothing: a hard-coded design already matches Monte Carlo, and Kerdock-type designs beat it by
constant factors. "Mechanistic" must therefore be formalized as something other than determinism (for example: never
evaluates f_W at a point; or propagates a bounded description layer by layer), and only for such restricted models can
lower bounds be proven (Section 6). This is consistent with Chen and Tell (STOC 2021): under their assumptions,
derandomization on average over samplable inputs costs only an n^eps factor, while worst-case derandomization costs a
factor n (and that is optimal under #NSETH).

## 4. White-box, worst case: beating sampling deterministically is as hard as derandomization

**Theorem 4.1 (Boolean circuits reduce to Gaussian means of ReLU networks). PROVEN.** Let C be a Boolean circuit with
m inputs and s gates of fan-in <= 2 from {AND, OR, NOT}, depth d. Let K > 0.
(a) There is a ReLU network f: R^m -> [0, 1], a DAG of O(m + s) units and O(m + s) edges with weights in {0, +-1, +-K}
and biases in {0, +-1}, depth d + 1, such that
  | E_{x ~ N(0, I_m)} f(x) - Pr_{u ~ U{0,1}^m} [C(u) = 1] | <= m / (K sqrt(2 pi)).
If C is layered with width w (identity gates added on wires that skip levels), f is an MLP of width <= 2 max(w, m).
(b) (bias-free) There is a positively homogeneous network f°: R^{m+1} -> R with no biases, weights in {0, +-1, +-K},
O(m + s) units, depth d + 3, with 0 <= f°(x) <= |x_0| and |E f° / E|x_0| - Pr[C = 1]| <= m / (2K).
(c) (noisy) If every unit's output receives independent noise N(0, sigma^2), there is a network of O(m + s) units (a
restore layer after every gate layer) with |E f - Pr[C = 1]| <= m/(K sqrt(2 pi)) + 2U exp(-1/(512 sigma^2)), U the number
of units. So noise of size sigma <= 1/(16 sqrt(2 ln(2U/eta))) adds at most eta. (Relative noise sigma times the layer RMS
is no larger, since all wire values lie in [0, 1].)

*Construction and proof of (a).* Binarize: beta_i(x) = relu(K x_i) - relu(K x_i - 1) = clamp(K x_i, 0, 1), which equals
u_i = 1[x_i > 0] unless 0 < x_i < 1/K. Gates on wires with values in [0, 1]: NOT a = relu(1 - a); AND(a, b) =
relu(a + b - 1); OR(a, b) = relu(a + b) - relu(a + b - 1) = min(1, a + b) (two units; consumers read their difference,
an affine function of unit outputs). Invariants: (I1) every wire value lies in [0, 1] (each gate maps [0,1]^2 into
[0, 1]); (I2) if every beta_i is in {0, 1}, every wire carries the Boolean value of C at u (the gates are exact on
{0, 1}). Hence f(x) = C(u(x)) outside B = union_i {0 < x_i < 1/K}, |f - C(u)| <= 1 everywhere, and
P(B) <= m (1/K) phi(0) = m / (K sqrt(2 pi)). The signs u_i of iid N(0,1) coordinates are iid uniform bits. QED
*(b).* Compute r = |x_0| = relu(x_0) + relu(-x_0) and carry it (relu(r) = r). Replace every constant 1 by r:
beta_i = relu(K x_i) - relu(K x_i - r), NOT a = relu(r - a), AND = relu(a + b - r), OR = relu(a + b) - relu(a + b - r)
(x_i is available after the first layer as relu(x_i) - relu(-x_i)). Everything is positively homogeneous of degree 1,
every wire lies in [0, r], and outside B° = union_i {0 < x_i < r/K} every wire equals r times the Boolean value. Then
E f° = E[r] Pr[C = 1] + E[(f° - r C(u)) 1_{B°}] (r is independent of u), and the last term is at most
E[r 1_{B°}] <= sum_i E[r (r/K) phi(0)] = m E[r^2] / (K sqrt(2 pi)) = m / (K sqrt(2 pi)); divide by E r = sqrt(2/pi). QED
*(c).* After every gate layer insert restore units R(v) = relu(2v - 1/2) - relu(2v - 3/2) = clamp(2v - 1/2, 0, 1),
which map [-1/4, 1/4] to 0 and [3/4, 5/4] to 1. If every noise variable has absolute value <= tau = 1/16, a gate fed
by restored wires (each Boolean +- tau) produces a value within 4 tau = 1/4 of its Boolean value (OR, the worst case:
2 tau from its inputs, 2 tau from its two units), so restoration returns it exactly, up to the restore units' own noise
(<= 2 tau); binarization units likewise. Every restored value lies in [0, 1] before noise. So on the event that all
|noise| <= tau and x is outside B, f = C(u) + (zero-mean output noise), and |f - C(u) - noise_out| <= 1 elsewhere.
Gaussian tails give P(some |noise| > 1/16) <= 2U exp(-1/(512 sigma^2)). QED

Numerical check (numerics/capp_check.py, 6 random circuits with m = 10, s = 40, K = 1000, 4e5 Gaussian samples): the
network means equal the exact acceptance probabilities within Monte Carlo error in all three variants (for example
0.2500 exact; 0.2502 +- 0.0007 with biases; 0.2498 +- 0.0009 bias-free; 0.2501 +- 0.0007 with noise sigma = 0.02).

**Corollary 4.2 (the IKW barrier). PROVEN reduction + cited theorem.** Suppose some algorithm, deterministic (or even
nondeterministic, using n^eps advice, on infinitely many sizes), computes E_{x~N(0,I)} f(x) within +-1/12 for every
ReLU network with E edges and outputs in [0, 1] in time 2^{E^{o(1)}}. Then CAPP (estimate the acceptance probability of
a size-s Boolean circuit within 1/6) is "nontrivially approximable" in the sense of Impagliazzo, Kabanets and Wigderson
(JCSS 2002, Section 5.1): apply 4.1(a) with K = 12 s (reduction error < 1/12; E = O(s)). By their Theorem 32 this
implies NEXP is not contained in P/poly. A deterministic polynomial-time algorithm would also give
promise-BPP = promise-P (their Remark 33).
*Reading.* Such deterministic estimators are believed to exist (standard circuit lower-bound assumptions give them by
hardness-versus-randomness), but proving that they exist would prove a circuit lower bound far beyond current
techniques. So nobody should expect to *prove* an advice-free, worst-case version of "deterministic estimation can match
sampling for every network". Note the advice in IKW is sublinear; ARC's advice of length O(|theta|) escapes this
(Proposition 4.4).

**Corollary 4.3 (deterministic estimators cannot match sampling in the worst case). PROVEN UNDER #NSETH.** Assume
#NSETH (Chen and Tell, Assumption 6.1: for every eps > 0 there is a k such that no nondeterministic machine counts the
satisfying assignments of k-SAT formulas on v variables in time 2^{(1-eps) v}, in the unambiguous sense of their
Definition 3.3). Then for every eps > 0 there is no deterministic algorithm that, for every ReLU DAG network f with E
edges and outputs in [0, 1], computes E_{x~N(0,I)} f(x) within +-1/12 in time O(E^{2 - eps}). Monte Carlo does it in
time O(E log(1/delta)) with probability 1 - delta. So in the worst case a deterministic estimator loses to sampling by
a factor E^{1 - o(1)} at constant precision.
*Proof.* Chen and Tell's proof of their Theorem 6.4 (case k = 1) uses its derandomization hypothesis only for one
explicit family of randomized tests: given a prime p in (n^4, 2n^4], a univariate polynomial H of degree O(n log^2 n)
over F_p and a 2-OV instance Phi with dimension d = O(log n), accept iff H(x) = Q(x) at a uniformly random x in F_p,
where Q(x) = sum_{j in [n]} X(P_1(j), P_2(x)) is computable with O(n d) field operations after O~(n) preprocessing.
For each input (p, H, Phi) this test is a Boolean circuit C of size n polylog n (field arithmetic on O(log n)-bit
numbers; x obtained from O(log n) random bits with statistical error <= n^{-10}); it accepts with probability 1 if
H = Q and with probability <= deg/p + n^{-10} < 1/3 otherwise. Apply 4.1(a) with K = 12 |C|: a deterministic
O(E^{2-eps}) mean estimator distinguishes the two cases in time n^{2 - eps} polylog n. Plugging this test into their
nondeterministic counter (guess p and H, verify H = Q with the test, output sum_{i in [n]} H(i)) counts 2-OV solutions
nondeterministically in time n^{2 - eps'}, which by their Lemma 6.3 (the standard k-SAT to k-OV reduction) contradicts
#NSETH. QED
*Reading.* This is the cleanest sense in which "a deduction-only method that is never worse than sampling, for every
network" is false: it is false for deterministic algorithms in the worst case, under a standard fine-grained hypothesis,
with a loss factor equal to the network size. Chen and Tell also show the matching upper bound (overhead n times
T^eps) under one-way functions plus strong hardness, so the worst-case gap is about one factor of the size.

**Proposition 4.4 (what the worst-case results say about ARC's MSP). PROVEN (logical relations).**
1. The MSP is quantified worst-case over theta *with advice pi(theta) of length O(|theta|)*; for a worst-case circuit
   placed in theta, pi can state the answer (ARC say so themselves). So Corollaries 4.2 and 4.3 do not refute the MSP.
2. They do apply to the reading "for every network, a deterministic estimator without instance-specific advice matches
   sampling": proving it is at least as hard as NEXP not in P/poly, and its constant-factor version is false under
   #NSETH.
3. ARC's random-MLP instance has theta empty and c = W random; only average-case statements apply, and by
   Proposition 3.5 determinism is no restriction there. The train-and-explain variant with a trainer that outputs a
   random circuit is likewise an average-case statement over a samplable distribution, where Chen and Tell's average-
   case derandomization (overhead n^eps, under their assumptions) indicates no barrier.
So the worst-case theory sharpens what the MSP must mean, but it cannot refute it: the open content is in "mechanistic"
and in the average case.

## 5. Cryptographic and idealized constructions: how much can anything beat sampling?

By Proposition 3.1, standard (polynomial-versus-superpolynomial) cryptography cannot make mean estimation hard at
inverse-polynomial precision, and by 3.2-3.3 pseudorandomness and undetectable planting make means *easier*, not
harder. The right question is fine-grained: on hard instances, by how much can any estimator beat Monte Carlo? The one
setting where this can be proven unconditionally is an oracle model.

**Model.** H: {0,1}* -> {-1, +1} is a uniformly random function (random oracle); networks may contain oracle gates.
An estimator sees the whole network (it is white-box for everything except the oracle), may compute anything, and is
charged one unit per oracle call. A forward pass that makes J oracle calls costs J.

**Theorem 5.1 (random-oracle networks). PROVEN in the random oracle model.**
(a) Single budget. Let f(x) = H(u_1 ... u_m), u the input bits (or the signs of a Gaussian input, as in 4.1). Every
estimator with at most Q oracle calls has E_H (mu-hat - mu)^2 >= (2^m - Q)^+ / 4^m, while Monte Carlo with N calls has
E_H MSE = (1 - 2^{-m}) / N. At Q = N = 2^{m-1} no estimator beats Monte Carlo by more than a factor 4 (1 + o(1)).
(b) All budgets at once. Let f(x) = sum_{j=1}^J H(j, u_1 ... u_j) (one forward pass = J calls). Every estimator with at
most Q calls has E (mu-hat - mu)^2 >= R(Q) := min { sum_j (2^j - q_j) 4^{-j} : sum_j q_j <= Q, 0 <= q_j <= 2^j }
>= 1/(Q + 2) - 2^{-J}, while Monte Carlo with Q calls has E MSE = J (J - 1 + 2^{-J}) / Q. Hence for
2 <= Q <= 2^{J-1} - 2 no estimator beats Monte Carlo by more than 4 J^2, and enumerating the coarse scales (query all
(j, v) with j <= k for the largest k with 2^{k+1} - 2 <= Q) beats it by at least J (J - 1) / 4. In this family the best
possible advantage over sampling is Theta(log^2 B) uniformly over a budget window [1, B], B = 2^{J-1}.
(c) Polylogarithmic advantage is unavoidable for multiscale oracle constructions. Consider any network that, per
forward pass, touches scale j with probability pi_j (one call each) and outputs the unbiased (Horvitz-Thompson)
combination sum_{j touched} (w_j / pi_j) H(j, cell_j(x)) of independent oracle values on the 2^j cells of scale j. Then
sum_{k=1}^{J} 1/Adv(2^k) <= 4 sum_j w_j^2 / (sum_j w_j)^2, so some budget 2^k in the window has Adv(2^k) >= J/4, and
>= J^2/4 when the weights are equal.
*Proofs.* (a) Given the queried points, the unqueried values are iid uniform +-1, so
Var(mu | transcript) = (2^m - q)/4^m with q <= Q distinct points; Var_x f = 1 - mu^2 and E mu^2 = 2^{-m}. (b) Same
argument per scale: the posterior variance is sum_j (2^j - q_j) 4^{-j}. The minimizing allocation fills small j first
(each query to scale j removes 4^{-j}); with k maximal such that 2^{k+1} - 2 <= Q, at most scales 1..k+1 are touched,
so R(Q) >= sum_{j >= k+2} 2^{-j} = 2^{-(k+1)} - 2^{-J} >= 1/(Q+2) - 2^{-J}. Monte Carlo: E_H Var_x f = J - sum_j 2^{-j}
(oracle values at different scales are independent), and N = Q/J passes. When Q + 2 <= 2^{J-1} the bound is
>= 1/(2(Q+2)), giving advantage <= 2 J^2 (Q+2)/Q <= 4 J^2. Coarse enumeration leaves exactly 2^{-k} - 2^{-J} <= 4/(Q+2).
(c) Completing scales j < k costs < 2^k calls, so R(2^k) <= sum_{j >= k} w_j^2 2^{-j}; summing,
sum_k 2^k R(2^k) <= 2 sum_j w_j^2. Monte Carlo: E Var_x f >= sum_j w_j^2 (1/pi_j - 2^{-j}) >= sum_j w_j^2 / (2 pi_j), and
the expected cost per pass is sum_j pi_j; by Cauchy-Schwarz (sum_j w_j)^2 <= (sum_j w_j^2/pi_j)(sum_j pi_j). QED

**What this means.** Even when the network is literally built from random oracles (no structure at all, so white-box
access is worthless beyond the wiring), estimators beat sampling by polylogarithmic factors uniformly over budgets, and
by at most a constant at a prescribed budget. Conversely, the advantage of exhaustive deduction over sampling in such
worlds is only polylogarithmic. In the random oracle model "mechanistic" and "sampling" cannot be separated at all:
evaluating a random function at fixed points is the same as sampling it, and a fixed-subset average already satisfies
the MSP there.

**Instantiation (CONDITIONAL on an explicit assumption).** Replace H(j, v) by F_k(j, v) for a keyed function family
implemented by a ReLU network of size t_F (e.g. a TC^0 pseudorandom function; threshold gates on Boolean inputs are
exact as relu(S - theta + 1) - relu(S - theta) for integer S). *Summation-hardness assumption SH(F):* given the key k,
no algorithm running in time t estimates 2^{-j} sum_v F_k(j, v) with MSE below (1 - o(1)) (2^j - t/t_F)^+ / 4^j on
average over k. Under SH(F), the explicit ReLU networks f_k(x) = sum_j F_k(j, u_1 ... u_j) obey the bounds of 5.1(b)
with Q = t / t_F. SH is true in the random oracle model but is *not* implied by pseudorandomness (PRF security is about
adversaries without k); it is a fine-grained, non-standard assumption. I know of no shortcut for standard constructions
and did not search for one.

**Proposition 5.2 (advice, and ARC's "pi cannot depend on c"). PROVEN in the random oracle model.** Key the oracle by
the context: H_c(.) = H(c, .), c uniform.
1. Advice pi independent of c (of any length) carries no information about mu(c), so the bounds of 5.1 hold for
   G(theta, pi, c, eps). ARC's condition is exactly what makes them hold: if pi could depend on c,
   ceil(log2(1/eps)) + O(1) bits would encode mu to within eps.
2. Train-and-explain: if the key sits in theta = T(s) and the explainer E(s) runs in time t_E, its advice is a function
   of at most t_E oracle values, so the bounds hold with Q replaced by Q + t_E. Advice from an explainer no stronger
   than the trainer helps only if the trainer itself spent a comparable amount of computation on the oracle.
So hard instances for "beating sampling" must place their difficulty in computation that nobody performed: not the
trainer, not the explainer. Random networks (c = W, never selected) are of this kind; planted or backdoored networks
are not (their planter knows the structure, and Proposition 3.2 shows the planting cannot move means anyway).

## 6. Average case: random He-ReLU networks at criticality, depth growing with width

Setting: x ~ N(0, I_n), h_0 = x, h_l = relu(W_l h_{l-1}), W_l iid N(0, 2/n) entries, no biases. For an input x let
u_l(x) = h_l / ||h_l|| and G_L(x) = ||h_L(x)||^2 / ||x||^2 (the output scale; G_L depends only on the direction of x).
alpha = L/n. c_l(x, x') = cosine between h_l(x) and h_l(x'); for l >= 1 both vectors are nonnegative, so c_l >= 0.

### 6.1 Exact facts

**Lemma 6.1 (single-input norm law). PROVEN (exact; known in this form, cf. Hanin 2018, arXiv 1801.03744; Hanin and
Nica 2020, arXiv 1812.05994, for the log-normal limit).** For fixed x != 0, G_L(x) = prod_{l=1}^L rho_l with rho_l iid,
rho_l ~ (2/n) sum_{j=1}^n relu(g_j)^2, g_j iid N(0,1). E rho = 1, Var rho = 5/n, so E G_L = 1 and
E G_L^2 = (1 + 5/n)^L -> e^{5 alpha}.
*Proof.* Given layers < l, z = W_l h_{l-1} has iid N(0, (2/n)||h_{l-1}||^2) coordinates, whatever the direction of
h_{l-1}; so rho_l = ||relu(z)||^2 / ||h_{l-1}||^2 has the stated law and is independent of the past. (If some
h_{l-1} = 0, define the remaining factors with an arbitrary unit vector; the product is 0 either way.) Var: (4/n^2) n
(E relu(g)^4 - (E relu(g)^2)^2) = (4/n)(3/2 - 1/4). QED

**Lemma 6.2 (two inputs). PROVEN (exact).** For nonzero x, x' and l >= 1,
E[rho_l(x) rho_l(x') | layers < l] = 1 + (4/n)(J(c_{l-1}) - 1/4), where for a standard bivariate normal pair with
correlation c, J(c) = E[relu(g)^2 relu(g')^2] = [3 sin t cos t + (pi - t)(1 + 2 cos^2 t)] / (2 pi), t = arccos c (the
degree-2 arc-cosine kernel of Cho and Saul). J is increasing on [0, 1], J(0) = 1/4, J(1) = 3/2, J(-1) = 0.
*Proof.* (z_j, z'_j) are iid bivariate normal with correlation c_{l-1}; expand
(2/n)^2 sum_{j,k} relu(z_j)^2 relu(z'_k)^2. Monotonicity: J(c) = sum_k k! a_k^2 c^k with a_k the Hermite coefficients
of relu(g)^2. Checked by Monte Carlo at 8 values of c (numerics/lemma_checks.py J). QED

**Lemma 6.3 (cosines stay positive). PROVEN.** Let c_* = 1/(4 pi). For l >= 2,
P(c_l < c_* | layers < l) <= exp(-0.0126 n) + 2 exp(-0.064 n).
*Proof.* Normalize the variances to 1. Numerator sum_j X_j, X_j = relu(z_j) relu(z'_j) >= 0, with
E X_j = J_1(t)/(2 pi) >= 1/(2 pi) (J_1(t) = sin t + (pi - t) cos t >= 1 for c >= 0) and E X_j^2 = J(c). The one-sided
bound for nonnegative summands, P(S <= ES - s) <= exp(-s^2 / (2 sum E X_j^2)), with s = ES - n c_*, gives
exp(-n min_c (E X - c_*)^2 / (2 J(c))), and the minimum over c in [0, 1] is 0.01267 (at c = 0). Denominator: by a
Chernoff bound with E exp(lambda relu(g)^2) = 1/2 + (1/2)(1 - 2 lambda)^{-1/2}, lambda = 0.21,
P(||relu(z)||^2 >= n) <= exp(-0.0646 n). On the complement, c_l >= n c_* / n. QED

**Lemma 6.4 (the output scale is shared across inputs). PROVEN (asymptotic in n, L = O(n)).** Let
kappa_* = 4 (J(c_*) - 1/4) = 0.2155 and delta_* = kappa_*/n. For nonzero x, x' and L >= 2,
Cov_W(G_L(x), G_L(x')) >= (1 - 1/n)(1 + delta_*)^{L-2} - 1 - eps_n, eps_n <= sqrt(3) L (1 + delta_*)^L (1 + 31/n)^{L/2}
e^{-n/158}, which tends to 0 when L = O(n). For L = alpha n the right side tends to e^{kappa_* alpha} - 1 > 0.
*Proof.* Let Z_l = rho_l(x) rho_l(x') >= 0, m(c) = 1 + (4/n)(J(c) - 1/4), E_l = {c_k >= c_* for 2 <= k <= l},
Y_l = prod_{k<=l} Z_k 1_{E_l}. E[Z_1] = m(c_0) >= 1 - 1/n (J >= 0); E[Z_1 Z_2] >= E[Z_1] (c_1 >= 0 so m(c_1) >= 1).
For l >= 2, E[Y_{l-1} Z_l] = E[Y_{l-1} m(c_{l-1})] >= (1 + delta_*) E[Y_{l-1}] on E_{l-1} (J increasing), and
E[Y_l] >= E[Y_{l-1} Z_l] - E[prod_{k<=l} Z_k 1(c_l < c_*)], the subtracted term bounded by Cauchy-Schwarz with
E[Z_k^2 | past] <= E rho^4 = 1 + 30/n + O(n^-2) and Lemma 6.3. Iterating and using E[G G'] >= E[prod Z_k 1_{E_{L-1}}]
gives the claim. QED

**Theorem 6.5 (estimators that ignore the early layers fail). PROVEN.** Let S(W) = E_x ||h_L(x)||^2 / n (so E_W S = 1).
Split W into A = (W_1, ..., W_{L-k}) and B = (W_{L-k+1}, ..., W_L). Any estimator S-hat = F(B, xi) with xi independent
of A (xi may hold coins, the input law, mean-field constants, any amount of computation on B) satisfies
  E (S-hat - S)^2 >= Var(E[S | A]) = Var_A( E_x ||h_{L-k}(x)||^2 / n ) >= (1 - 1/n)(1 + kappa_*/n)^{L-k-2} - 1 - eps_n,
which for L - k = beta n tends to e^{kappa_* beta} - 1 > 0. Monte Carlo with N forward passes has MSE
E_W Var_x(||h_L||^2/n) / N <= [(1 + 2/n)(1 + 5/n)^L - 1] / N. So for L = alpha n and k <= L/2 such an estimator is no
better than Monte Carlo with a bounded number N_*(alpha) of samples, whatever its compute.
*Proof.* By Lemma 6.1 applied to the last k layers, E[S | A] = E_x ||h_{L-k}(x)||^2 / n. For independent A, B,
S - E[S|B] = (S - E[S|A] - E[S|B] + E S) + (E[S|A] - E S), and the two parts are orthogonal (the first has conditional
mean 0 given A, because E[E[S|B] | A] = E S). So E(S - F(B, xi))^2 >= E(S - E[S|B])^2 >= Var(E[S|A]).
Var(E[S|A]) = E_{x,x'} [(||x||^2/n)(||x'||^2/n) Cov(G_{L-k}(x), G_{L-k}(x'))], and Lemma 6.4 bounds the covariance
pointwise; E ||x||^2 / n = 1. The Monte Carlo bound uses E_W Var_x <= E f^2 - (E_W S)^2. QED
*Numbers* (numerics/lemma_checks.py backward; 300 nets each, inner Monte Carlo noise removed):

| n, L, L - k | Var_W(E[S|A]) | asymptotic bound | E_W Var_x(f_L) | N_* = MC samples worth |
|---|---|---|---|---|
| 64, 16, 8 | 0.31 | 0.004 | 0.95 | 3.0 |
| 64, 32, 16 | 2.08 | 0.03 | 4.83 | 2.3 |
| 128, 32, 16 | 0.47 | 0.016 | 1.08 | 2.3 |
| 128, 64, 32 | 1.42 | 0.04 | 1.79 | 1.3 |

The rigorous bound is loose by 1-2 orders (it uses c_* = 1/(4 pi) where the true cosines approach 1), but the
conclusion is quantitative in the data: knowing the second half of the network exactly, plus anything that does not
depend on the first half, is worth 1-3 Monte Carlo samples. This is the theorem behind our finding that an
input-independent backward stand-in fails at the critical point (random-net-phases, section 3), and it is specific to
homogeneous activations: in the chaotic tanh nets the scale is set by saturation in the last layers and a backward
stand-in works (k* = 24-40 there). By symmetry, estimators that ignore the last k layers also fail (the last layers'
gain is random with variance ~ 5k/n).

### 6.2 How errors propagate

**Lemma 6.6 (exact error identity). PROVEN.** Fix W. Let T_l be the true layer map on laws (pushforward through layer
l, including noise if any), P_0 = D, and G_l(h) = E[readout | h_l = h] the true backward function, so
G_{l-1} = T_l^* G_l. For any estimator that maintains laws (or signed measures, or linear functionals) Q_0, ..., Q_L and
outputs mu-hat = integral of G_L dQ_L (the readout),
  mu - mu-hat = integral G_0 d(P_0 - Q_0) + sum_{l=1}^{L} integral G_l d(T_l Q_{l-1} - Q_l).
*Proof.* integral G_l d(T_l Q_{l-1}) = integral G_{l-1} dQ_{l-1}; telescope. QED
Each term is the *local closure error at layer l, tested against the true backward function G_l*. An estimator is
accurate iff its local errors are small in the directions that G_l still sees. Forgetting (contraction) means that G_l
becomes nearly constant on the typical support as L - l grows; criticality means it does not.

**Proposition 6.7 (scale errors never contract). PROVEN (exact).** For a bias-free ReLU network (any weights), the map
from layer l to the output is positively homogeneous of degree 1, so G_l(c h) = c G_l(h) for c > 0. If a description
makes a relative scale error t at layer l (Q_l = law of (1 + t) h, h ~ T_l Q_{l-1}), the output estimate is off by
exactly the factor (1 + t), at every depth. Independent scale errors t_l at the L layers therefore produce a relative
output error with variance about sum_l Var(t_l): linear in L, with no discount for distance to the output. With
independent noise of fixed absolute size the identity becomes approximate (the noise breaks homogeneity by O(sigma^2)
per layer).

**Proposition 6.8 (one-step survival of mean errors). PROVEN (exact, to first order in delta).** Let Q be any law of
the state at layer l - 1 and delta a vector orthogonal to every state in the support of Q. Let
Delta = E_{h~Q}[D_W(h) W delta] (D_W(h) = diag(1[W h > 0])) be the first-order change of the next layer's mean when the
mean is shifted by delta. Over a fresh W with iid N(0, 2/n) entries,
  E_W ||Delta||^2 = ||delta||^2 E_{h, h' ~ Q iid} [1 - theta(h, h')/pi]   (theta = angle between h and h').
*Proof.* Delta_j = p_j(w_j)(w_j . delta) with p_j(w) = P_{h~Q}(w . h > 0). Because delta is orthogonal to the
support, w_j . delta is independent of {w_j . h}, so E Delta_j^2 = (2/n)||delta||^2 E_{h,h'} P(w . h > 0, w . h' > 0),
and P(both > 0) = (pi - theta)/(2 pi). Sum over n neurons. QED
Numerically (n = 256, state at depths 4, 16, 48, delta orthogonal to 128 sampled states): 0.741 +- 0.013 vs 0.740;
0.909 +- 0.020 vs 0.887; 0.957 +- 0.019 vs 0.964.

**Corollary 6.9 (accumulation at criticality). PROVEN UNDER the mean-field angle recursion.** If pairwise angles at
layer l follow the mean-field recursion c_{l} = kappa(c_{l-1}), kappa(c) = (sin t + (pi - t) cos t)/pi, then
1 - c_l ~ 9 pi^2/(2 l^2), theta_l ~ 3 pi / l, and the survival factor from layer ell to L is
prod_{l=ell}^{L} (1 - 3/l) ~ (ell/L)^3. Local errors of size e_ell (uncorrelated across layers, orthogonal to the state
span) then give output error variance sum_ell e_ell^2 (ell/L)^3, which is ~ e^2 L/4 for constant e: it grows linearly
with depth. In the chaotic phase the same factor tends to chi_c < 1 (decorrelated fixed point), errors survive only
O(1/ln(1/chi_c)) layers, and the accumulated error stays bounded as L grows; this is what we measured (K = 2 error flat
from depth 32 to 64 in chaotic tanh nets, growing at the critical point).
*Derivation.* Near c = 1, 1 - kappa(c) = (1 - c) - (2 sqrt 2 / (3 pi))(1 - c)^{3/2} + O((1-c)^2); the continuum version
d(1-c)/dl = -(2 sqrt 2/(3 pi))(1-c)^{3/2} gives 1 - c_l ~ 9 pi^2 / (2 l^2), so theta_l ~ 3 pi / l and
1 - theta_l/pi ~ 1 - 3/l. (The (ell/L)^3 law matches what contest participants measured on ARC's nets.)

**Lemma 6.10 (multiplicative accumulation). PROVEN (exact).** Let rho_l be F_l-measurable factors and rho-hat_l an
estimator's factors with e_l = rho_l - rho-hat_l, such that E[e_l rho-hat_l | F_{l-1}] = 0, E[rho-hat_l^2 | F_{l-1}] = a
and E[e_l^2 | F_{l-1}] = b are constants. Then E(prod rho_l - prod rho-hat_l)^2 = (a + b)^L - a^L, i.e. relative error
1 - (a/(a+b))^L.
*Proof.* Expand prod (rho-hat + e) over subsets S of layers carrying e. For S != S', at the largest index where they
differ one product has e_l and the other rho-hat_l; later factors coincide and their squares have constant conditional
expectations, earlier factors are F_{l-1}-measurable, so the cross term is a constant times E[... E[e_l rho-hat_l |
F_{l-1}]] = 0. Diagonal terms give b^{|S|} a^{L-|S|}. QED
*Example (per-layer polynomial surrogates).* Replace each layer's norm ratio rho_l (Lemma 6.1) by its Hermite
truncation of degree D in the layer's weights (conditionally on the past, i.e. a degree-D polynomial activation applied
along the true trajectory). Then a + b = 1 + 5/n, b = 5 r_D / n with r_D the fraction of Var(relu(g)^2) above degree D:
r_1 = 0.491, r_2 = 0.091, r_3 = r_4 = 0.0058, r_5 = 0.0016. At L = alpha n the relative error of the output scale of a
*single input* tends to 1 - exp(-5 r_D alpha): 0.11 for D = 2 at alpha = 1/4, independent of n. Checked by simulation
(n = 64, L = 16, D = 2: 0.314 +- 0.017 against 0.334; n = 128, L = 32: 0.332 +- 0.016 against 0.353; D = 1 at
n = 256, L = 64: 1.55 +- 0.09 against 1.56). A forward pass computes this quantity exactly; the point is that local
surrogates that are good to a fixed accuracy per layer lose a constant fraction at depth proportional to width.

**Proposition 6.11 (product-measure closures fail once the state collapses). PROVEN (exact second moments).** Let h
have covariance C (over inputs) and let a closure keep only the diagonal D = diag(C) (independent neurons). For a fresh
neuron z = w . h, w ~ N(0, (2/n) I), the predicted variance w^T D w differs from the true w^T C w by w^T (C - D) w,
which has mean 0 and standard deviation sqrt(2) (2/n) ||C - D||_F, against a mean variance (2/n) tr C. The relative
error is sqrt(2) ||C - D||_F / tr C per neuron per layer; for a collapsed state (C close to rank one with spread-out top
eigenvector, as at depth in He-ReLU nets, participation ratio 2-4 at depth 32-64) this is of order 1, independently of
width. This is the "coupling arithmetic" note made exact.

**Theorem 6.12 (contraction makes layer-local errors depth-free). PROVEN (generic).** Suppose for some seminorm ||.||_F
on test functions and the dual distance d_F, ||G_l||_F <= C eta^{L-l} with eta < 1 (backward forgetting), and each
local error satisfies d_F(T_l Q_{l-1}, Q_l) <= e. Then |mu - mu-hat| <= C e / (1 - eta) for every depth L.
*Proof.* Lemma 6.6 and |integral G d(P - Q)| <= ||G||_F d_F(P, Q). QED
**Proposition 6.13 (small noise gives no worst-case contraction). PROVEN.** For a layer h -> relu(W h) + tau xi,
xi ~ N(0, I), the Dobrushin coefficient over a set of states S is sup_{h,h' in S} TV = 2 Phi(Delta_S/(2 tau)) - 1 with
Delta_S the largest distance between relu(W h), relu(W h') on S; for typical states Delta_S ~ sqrt(n) times the
activation RMS while tau = sigma times the RMS, so the coefficient is 1 - O(exp(-n/(8 sigma^2))). Noise therefore helps
only through average-case mixing on typical states (the phase-dependent forgetting depth ln(1/sigma)/|lambda| we
measured), never through worst-case contraction, and at the critical point (lambda = 0) not even on average for
sigma <= 0.01 at depth 64.
