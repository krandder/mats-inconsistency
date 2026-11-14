PROVEN: Determinism alone does not create the proposed average-case barrier. There is a uniform way to obtain sampling randomness from first-layer rotational symmetry.
PROVEN: With unrestricted arithmetic cost, batching and fast matrix multiplication already give a growing advantage over the stated serial-MC benchmark.
The remaining hardness questions require a revised benchmark or a restricted computational class.

Write f_W(x) = a_L(x), and define

    V = E_W tr Var_x f_W(x),
    R(A) = E_W ||A(W) - mu(W)||^2,
    m_n = E chi_n.

All results below concern sigma = 0. Exact distribution transforms require an explicit computational convention, addressed in answer 1. No files were edited.

1. Determinism and a uniform construction

PROVEN: There is an explicit uniform deterministic estimator with average error at most V/N. Its implementation uses N network evaluations, O(n^2) preprocessing, and O(nN) scalar distribution transforms. Thus the requested arithmetic-cost statement holds under the named hypothesis that these transforms have unit cost. With polynomial-accuracy transforms, it gives asymptotic matching in the usual finite-precision implementation described below.

Here is the construction for n >= 6.

Perform three steps of a right-looking orthogonal reduction of W_1. Obtain an orthogonal Q and B_1 = W_1 Q such that the first three rows of B_1 have zeros after coordinates 1, 2, and 3, respectively. Leave the other layers unchanged, obtaining B.

The Gaussian decomposition has two properties:

    mu(B) = mu(W),

and the orthogonal directions removed during these three steps are independent of B. They supply

    (n-1) + (n-2) + (n-3) = 3n-6

independent continuous angular coordinates. Applying their conditional CDFs supplies that many independent uniforms, independent of B.

Proof of independence: a Gaussian row decomposes into an independent length and uniform direction. Rotate its direction to the first coordinate. Conditional on that rotation, every remaining independent Gaussian row remains Gaussian with its original law. Repeat in the remaining orthogonal complement. This proves the assertion inductively. The identity for mu follows from Gaussian input rotation invariance.

Use 2n of those uniforms to form independent U,V in [0,1)^n. For i = 1,...,N, set coordinatewise

    Z_i = (U + i V) mod 1,
    X_i,j = Phi^{-1}(Z_i,j).

For distinct integers i,j, the map

    (U,V) -> (U+iV, U+jV) mod 1

preserves Haar probability on the two-dimensional torus. Consequently X_i and X_j are independent N(0,I_n), although the entire collection need not be mutually independent.

Define

    A(W) = (1/N) sum_i f_B(X_i).

Conditional on B, pairwise independence gives

    E[A | B] = mu(B),
    E[||A-mu(B)||^2 | B] = tr Var_x f_B(x)/N.

Averaging proves R(A) = V/N exactly. This is a deterministic function of W; no external random bits occur.

PROVEN: The construction is uniform rather than advice-based. Three Householder transformations cost O(n^2), and the torus recurrence and coordinate transforms use O(nN) scalar operations/transforms. It does not search for a favorable design or precompute answers for individual sizes.

PROVEN: Small widths do not obstruct uniformity. For n >= 4, three reduction steps supply enough uniforms to generate pairwise-independent spherical directions, requiring 2(n-1) coordinates. Integrating the Gaussian radius analytically then gives error at most V/N. For n = 2,3, supplement the orthogonal coordinates with the first-layer Frobenius radius R. Positive homogeneity gives f_W = R f_B after normalization. Under squared-error weighting, R has its R^2-size-biased law; applying that law's CDF supplies one additional uniform independent of the normalized network. The same proof, under this weighted law, gives the spherical-MC risk. For n = 1,

    mu(W) = [f_W(1) + f_W(-1)] / sqrt(2 pi),

so two evaluations suffice.

PROVEN UNDER polynomial-accuracy distribution transforms: For L = alpha n and N polynomial in n, finite-precision versions attain

    R(A) <= (1+o(1)) V/N

with lower-order preprocessing and sampling work. The relevant approximation need not survive an exponentially large worst-case Lipschitz bound: for an independent He layer,

    E_W ||relu(Wa)-relu(Wb)||^2 <= ||a-b||^2.

To prove this, average the squared difference under W and -W and use

    (relu(s)-relu(t))^2
      + (relu(-s)-relu(-t))^2 <= (s-t)^2.

Thus downstream layers do not amplify input quantization error in mean square. The canonicalized first layer has the same operator norm as W_1, with bounded expected squared norm. Polynomially small transform errors therefore suffice. Exact equality, however, belongs to the ideal continuous-transform model, not to a finite-bit machine.

HEURISTIC, proposed definition: A useful “mechanistic” restriction would specify an auditable expression language. For example, permit fixed, instance-independent cubature nodes and specified moment operations, while forbidding the use of weight-derived quantities as seeds that select evaluation points. This contains fixed designs and ordinary moment closures and excludes the construction above.

PROVEN: Determinism, continuity, or numerical stability alone does not implement that restriction. Fixed-design network evaluation is itself continuous in the weights. Requiring invariance under every input rotation would also exclude ordinary fixed designs. There is no automatic semantic distinction between “deduction” and “sampling” in the final numerical answer.

2. Fixed point sets and adaptive evaluations

PROVEN: The average error of every fixed linear rule is exactly a kernel discrepancy.

On the unit sphere, define the finite-width, finite-depth kernel

    k(t) = E_W <f_W(u), f_W(v)>,     <u,v> = t.

Rotation invariance makes this well-defined, and critical initialization gives k(1) = 1. Put

    q = E_{U,V uniform on sphere} k(<U,V>).

Positive homogeneity gives

    E_W ||mu(W)||^2 = m_n^2 q,
    V = n - m_n^2 q.                                      (2.1)

For a rule A = sum_i w_i f_W(r_i u_i), write c_i = w_i r_i. Its exact risk is

    R(A)
      = sum_{i,j} c_i c_j k(<u_i,u_j>)
        - 2 m_n q sum_i c_i + m_n^2 q.                    (2.2)

This follows simply by expanding the square and integrating its three terms.

PROVEN: The harmonic version isolates exactly what a design must integrate. Let Y_{ell,m} be an orthonormal spherical-harmonic basis for probability surface measure, and write

    k(<u,v>) =
      sum_{ell >= 0} lambda_ell sum_m Y_{ell,m}(u)Y_{ell,m}(v).

Then lambda_ell >= 0, lambda_0 = q, and

    R(A)
      = q (sum_i c_i - m_n)^2
        + sum_{ell >= 1} lambda_ell
            sum_m (sum_i c_i Y_{ell,m}(u_i))^2.           (2.3)

In particular, a spherical t-design with c_i = m_n/N removes all degrees 1,...,t. Its remaining error depends on both the spectral tail and the discrepancies of the design at higher degrees.

PROVEN: For this ensemble there is an additional, useful power-series representation:

    k(t) = sum_{d >= 0} b_d t^d,
    b_d >= 0,
    sum_d b_d = 1.                                       (2.4)

Proof: condition on layers 2,...,L. The first-layer preactivations at u,v are Gaussian vectors with coordinatewise correlation t. Expand the downstream function of that Gaussian vector in multivariate Hermite polynomials. The identity

    E[H_a(Z) H_b(Z')] = 1_{a=b} t^{|a|}

gives (2.4), with b_d the expected squared norm of the degree-d Hermite coefficients.

This representation yields an explicit limitation on positive-weight rules. Define

    tau_{2s} = E <U,V>^{2s}
             = (2s-1)!! / [n(n+2)...(n+2s-2)].

For any positive spherical rule with N nodes and weights summing to one,

    R(A) >= (m_n^2 / (2N))
              sum_{s : tau_{2s} <= 1/(2N)} b_{2s}.        (2.5)

Proof: for the kernel t^d, the discrepancy equals

    ||sum_i w_i u_i^{tensor d} - E U^{tensor d}||^2,

so every degree contributes nonnegatively. For even d, all off-diagonal inner-product powers are nonnegative. Its discrepancy is therefore at least

    sum_i w_i^2 - tau_d >= 1/N - tau_d.

Sum the selected even degrees.

Consequently the gain of any such rule satisfies

    gain <=
      2V / [m_n^2 sum_{s : tau_{2s} <= 1/(2N)} b_{2s}].   (2.6)

For N <= n^p, fixed even degrees 2s with s > p eventually qualify. The unresolved quantity is their total coefficient mass at L = alpha n, relative to V/m_n^2.

PROVEN: Signed weights require a different bound. For a degree-2s term, flatten u^{tensor 2s} into the rank-one matrix

    u^{tensor s} (u^{tensor s})^T.

The quadrature approximation has rank at most N, even with signed weights. If

    M_s = E[U^{tensor s}(U^{tensor s})^T],

then its degree-2s discrepancy is at least the squared singular-value tail of M_s after rank N. Multiplication by m_n^2 b_{2s} gives a valid risk lower bound. This is a moment-matrix bound, not a running-time bound.

PROVEN: Radial integration is an immediate improvement that must be included in comparisons. Pairwise-independent spherical sampling has risk

    V_sph/N,     V_sph = m_n^2(1-q),

whereas

    V = (n-m_n^2) + V_sph.                               (2.7)

Here n-m_n^2 -> 1/2. If V_sph is much smaller than this radial term, radial integration gives a growing improvement over ordinary Gaussian-input MC.

PROVEN: A simple explicit design already matches spherical MC for N <= n. Take N orthogonal directions. Since k(0) = b_0 <= q,

    R = m_n^2 [1/N + (1-1/N)k(0) - q]
      <= m_n^2(1-q)/N.

PROVEN: For adaptive evaluation algorithms, the exact general characterization is instead Bayesian. If T is the complete evaluation transcript, then

    inf_{A measurable from T} E||A-mu||^2
      = E tr Cov(mu | T).                               (2.8)

The covariance kernel alone does not determine this quantity for a non-Gaussian prior.

PROVEN UNDER a Gaussian function prior: Adaptivity cannot improve optimal average squared error for estimating a linear functional from noiseless point evaluations. Gaussian conditioning makes posterior covariance depend only on the queried locations, not the observed values. Every realized adaptive sequence has posterior variance at least the minimum over fixed sequences; a minimizing fixed sequence attains that minimum.

That hypothesis is essential. The finite-width network prior is non-Gaussian; a nondegenerate final ReLU output is also non-Gaussian even when its preactivation has a Gaussian-process limit. Thus the Gaussian-prior theorem does not settle adaptive evaluation of the stated network.

CONJECTURE: A constant-gain bound for fixed or adaptive evaluations in the joint limit remains a separate assertion. The supplied fixed-depth spectral asymptotic does not establish it: its constants and crossover degrees must be controlled uniformly when L grows with n.

3. Polynomial estimators, stability, and statistical queries

PROVEN: Degree-D polynomials in the weights have an exact optimal-error formula.

Standardize all weights into G ~ N(0,I_M), M = n^2 L, and expand

    mu(G) = sum_a mu_a H_a(G),
    E_k = sum_{|a|=k} ||mu_a||^2.

Then

    inf_{degree(A) <= D} R(A) = sum_{k>D} E_k.            (3.1)

Proof: orthogonal projection in Gaussian L^2. Instance-independent coefficients are precisely the coefficients permitted in this projection problem.

This is a genuine restricted-class lower bound. It neither says that computing the optimal coefficients is cheap nor that efficient whole-network algorithms have low polynomial degree.

PROVEN: Weight-noise stability measures this spectrum. For independent G,G', set

    G_rho = rho G + sqrt(1-rho^2) G',
    S(rho) = E <mu(G)-E mu, mu(G_rho)-E mu>.

Then

    S(rho) = sum_{k>=1} rho^k E_k.                       (3.2)

Writing T = S(1), for 0 < rho < 1,

    max(0, T-rho^{-D}S(rho))
        <= sum_{k>D} E_k
        <= [T-S(rho)] / [1-rho^{D+1}].                  (3.3)

The lower bound follows from
S(rho) >= rho^D sum_{1<=k<=D} E_k.
The upper bound follows by retaining degrees above D in T-S(rho).

PROVEN: An unbiased measurement of the uncentered quantity in (3.2) uses independent inputs X,Y:

    E <f_G(X), f_{G_rho}(Y)>.

Using the same input at both networks estimates a different observable. Resolving a small high-degree tail from noisy stability measurements may itself be statistically expensive.

PROVEN: Stability gives a direct inequality without an overlap-gap hypothesis. Define

    d_mu(rho)^2 = E||mu(G)-mu(G_rho)||^2,
    d_A(rho)^2  = E||A(G)-A(G_rho)||^2.

The L^2 triangle inequality gives

    R(A) >= (max(0, d_mu(rho)-d_A(rho)))^2 / 4.          (3.4)

Thus an algorithm constrained to vary substantially less than the target under a particular perturbation must fail.

PROVEN: The usual overlap-gap argument is not supplied merely by calling an estimator “stable.” For each fixed W, acceptable estimates under squared loss form a Euclidean ball around mu(W), hence a connected convex set. An overlap-gap application would need an additional search representation, an overlap theorem for that representation, and a reduction from accurate mean estimation to that search problem. None follows from the present formulation.

PROVEN: An unrestricted statistical-query oracle can directly answer the question being studied. For a tail it can query

    phi(x) = 1{f_W,j(x)>t}.

For a nonnegative mean, query min(f_W,j(x),B)/B. With oracle tolerance tau, the resulting mean error is at most

    B tau + E_x f_W,j(x)^2 / B.

The second term follows from y 1{y>B} <= y^2/B.

Therefore an SQ lower bound needs restrictions on permitted queries, a tolerance cost, or a different observation model. Allowing arbitrary precise expectation queries makes mean estimation immediate; charging their implementation by sampling mostly builds sampling cost into the oracle. Standard SQ hardness statements cannot be transferred to a white-box algorithm without an explicit simulation theorem.

4. Connections to counting, volume, free energy, circuits, and permanents

PROVEN: There is an exact Gaussian-volume correspondence, but it is a representation rather than an average-case hardness reduction.

For each full gate pattern s, let C_s be its input cone and B_s the corresponding linear map. Up to null boundaries,

    f_W(x) = B_s x on C_s,

and therefore

    mu_j(W) = sum_s b_s,j^T E[X 1{X in C_s}],            (4.1)

    p_W,j(t) = sum_s gamma_n(
                     C_s intersect {b_s,j^T x > t}).     (4.2)

The cones are defined by the linear inequalities enforcing all gates along that pattern. Equation (4.1) is a sum of Gaussian first moments; (4.2) is a sum of Gaussian polyhedral volumes. There can be exponentially many terms, and their coefficients and inequalities are highly dependent functions of W.

PROVEN: Homogeneity also gives the exact radial formulas

    mu_j = m_n E_U f_W,j(U),

    p_W,j(t) =
      E_U Q_chi_n(t/f_W,j(U)),                           (4.3)

with the second integrand defined as zero when f_W,j(U)=0 and t>0. These formulas provide algorithms as well as structural correspondences.

PROVEN: The error metric prevents a direct transfer from relative counting or free-energy estimation. If a positive integral Z represents a partition function and

    F = log Z / (beta m),

then, locally,

    |delta F| approximately |delta Z| / (beta m Z).

ARC's allowed error eps sqrt(Var Y) therefore corresponds to free-energy error approximately

    eps * [sqrt(Var Y)/E Y] / (beta m).

An exponentially large coefficient of variation makes this a very weak free-energy guarantee. Conversely, constant additive error in F allows an exponentially large multiplicative error in Z.

PROVEN: A rare-event tail has an especially important mismatch. For a fixed W with probability p, the zero estimator has squared error p^2, while N-sample MC has error p(1-p)/N. Its gain over MC is

    (1-p)/(Np).

Thus p << 1/N makes zero a growing-factor improvement. Averaged over W, the exact ratio is

    E[p(1-p)] / (N E[p^2]).

A statement that p is “usually tiny” is insufficient if rare weights dominate E[p^2].

PROVEN: Mean estimation need not resolve tiny tails. For Y >= 0,

    E[Y 1{Y>t}] <= sqrt(E[Y^2] P(Y>t)).

This may put the entire tail contribution below the mean estimator's error allowance.

HEURISTIC, correspondence assessment: Spin-glass free energies, random-circuit output probabilities, and permanents suggest possible mechanisms, but none supplies a distribution-preserving, variance-calibrated reduction to iid He-ReLU weights here. Such a reduction would have to preserve both the random-instance law and the relevant error scale. Worst-case circuit encodings do neither automatically. Permanent and quantum-amplitude problems also have sign or complex-cancellation structures absent from the nonnegative network output.

No hardness consequence from those analogies is established in this attempt.

5. Complexity barriers and a counterexample to the proposed lower bound

PROVEN: Under the stated unrestricted arithmetic cost, “no algorithm beats serial MC by more than a constant” is false.

Let omega = log_2 7 < 3, using Strassen multiplication. Evaluate a batch of n inputs simultaneously. Each layer multiplies two n by n matrices and then applies n^2 ReLUs, so total cost is

    T = O(L n^omega).

The deterministic construction in answer 1 supplies n pairwise-independent inputs at lower-order overhead and achieves risk at most V/n.

At this same cost, the stated serial-MC benchmark buys only

    N_serial = Theta(n^{omega-2})

forward passes, with error Theta(V/n^{omega-2}). Hence the deterministic algorithm's gain is at least

    Omega(n^{3-omega}).

For L = alpha n, the budget remains polynomial. This improvement is computational batching; it does not require a better inference method.

PROVEN: The objection disappears if MC is charged using the same batched evaluation algorithm. Consequently a meaningful unrestricted lower bound must compare against an equally optimized sampling implementation, or explicitly restrict the arithmetic model. The distinction affects exponents, not just constants.

PROVEN: Even after that repair, a polynomial-factor lower bound between two polynomial running times does not by itself imply P != NP, NEXP not in P/poly, or a superpolynomial circuit lower bound. A uniform time lower bound also does not automatically become a nonuniform circuit lower bound.

PROVEN: The supplied IKW implication concerns sufficiently strong deterministic estimation on all encoded circuits. An estimator succeeding on average over iid Gaussian weights does not give that guarantee without a reduction taking arbitrary circuits to suitably distributed random weights. The symmetry construction in answer 1 illustrates why average-case determinism can be cheap while worst-case derandomization remains difficult.

The IKW 2002 and Chen-Tell 2021 results are used here only as supplied premises; their exact theorem formulations and bibliographic details are unverified in this attempt.

PROVEN: A natural-proofs barrier does not apply merely because a claim averages over random weights. Applying that framework would require an associated property of Boolean truth tables that is constructive, large, and useful against a specified circuit class. Gaussian measure on weight space is not automatically the required largeness condition on truth tables. No such transformation is provided here.

HEURISTIC: A corrected unrestricted arithmetic lower bound would nevertheless be a substantial complexity result. Kernel spectra, region counts, and low-degree failure provide restricted-class statements; they do not currently supply the missing bridge to arbitrary computations on all weights.

6. Branch-and-bound and region integration

PROVEN: The first layer has exactly 2^n nonempty full-dimensional gate cones almost surely. Gaussian W_1 is invertible almost surely, and inverse images of the 2^n open orthants give those cones. Every one intersects the sphere.

There are at most 2^{nL} full gate histories. This counts activation cells, not distinct maximal linear regions of the final function. They can merge: for example, if every entry of W_2 is negative, all later outputs vanish. That event has positive probability when L >= 2.

PROVEN: More strongly, polynomially many first-layer cells cover exponentially little input probability on average, even when selected after reading all weights.

Let p_s(W_1) be the Gaussian probability of first-layer sign pattern s, and put

    kappa_n = E_W sum_s p_s(W_1)^2.

For independent spherical U,V and T=<U,V>,

    kappa_n =
      E_T (1/2 + arcsin(T)/pi)^n.                        (6.1)

Indeed, one Gaussian row gives the same sign at U,V with the probability in parentheses, and rows are independent.

Splitting at T=1/2 gives

    kappa_n <= (2/3)^n + P(T>1/2)
             <= (2/3)^n
                + C sqrt(n) (3/4)^((n-3)/2).            (6.2)

The final inequality follows by integrating the spherical-coordinate density proportional to (1-t^2)^((n-3)/2).

For any weight-dependent selection S(W) of at most K patterns, Cauchy-Schwarz gives pointwise

    [sum_{s in S(W)} p_s]^2 <= K sum_s p_s^2.

Therefore

    E_W GaussianMass(selected cells) <= sqrt(K kappa_n). (6.3)

For polynomial K this is exponentially small.

PROVEN: This yields an actual error lower bound for methods that exactly integrate selected cells and discard everything else.

Let

    A_S(W) = E_X[f_W(X) 1{selected first-layer cells}].

Write Q(W)=E_X||f_W(X)||^2. Conditional Cauchy-Schwarz gives

    ||A_S(W)||^2 <= p_S(W) Q(W).

The critical fourth moment is

    E_W,X ||f_W(X)||^4
      = n(n+2)(1+5/n)^L.

This follows from the per-layer squared-norm multiplier
(2/n) sum_i (Z_i^+)^2, whose second moment is 1+5/n. Hence, for L=alpha n,

    E||A_S||^2 <= C_alpha n sqrt(K kappa_n).             (6.4)

On the other hand,

    E||mu(W)||^2
      >= n^2/[pi(n+2)] * (1+5/n)^(-(L-1))
      >= c_alpha n.                                    (6.5)

For completeness, if H=||h_{L-1}(X)||, then

    E mu_j = E H / sqrt(pi n),

and Holder gives

    E H >= (E H^2)^(3/2)/(E H^4)^(1/2).

Substituting the second and fourth moments proves (6.5).

The reverse L^2 triangle inequality applied to (6.4)-(6.5) now gives

    R(A_S) >= c'_alpha n

for polynomial K and sufficiently large n.

This theorem does not cover methods that bound, approximate, or analytically combine the unenumerated cells.

PROVEN UNDER a geometric mesh hypothesis: A useful upper rate is available for deterministic cubature exact on linear cells.

Partition S^{n-1} into M cells of diameter at most h, with

    M <= C h^{-(n-1)}.

Suppose their probabilities p_C and vector centroids v_C=E[U|C] are available. Use

    A = m_n sum_C p_C f_W(v_C).

A cell contained in one activation cone is integrated exactly. If f_W has Lipschitz constant K_W and its crease set has tubular measure at most B_W h, only boundary cells contribute error. Thus

    ||A-mu|| <= m_n K_W B_W h^2
              <= C_W M^{-2/(n-1)}.                     (6.6)

Proof: each boundary cell contributes at most its mass times m_n K_W h, and their total mass is at most B_W h.

Finite polyhedral crease sets satisfy such a tubular bound at fixed architecture. The constants can be enormous and have not been controlled uniformly in n,L.

PROVEN UNDER an adaptive mesh hypothesis: If a certified mesh needs only O(h^{-(n-2)}) cells while keeping every boundary cell's diameter at most h, the same proof gives error O_W(M^{-2/(n-2)}) for n>2. Constructing such a mesh, certifying gates, and computing cell moments must all be charged.

HEURISTIC: These rates explain a possible dimensional obstruction, but they are upper bounds, not lower bounds. Region counts alone cannot determine the cost of downstream-informed branch-and-bound. Exact evaluation on a region also does not make its Gaussian first moment or probability free.

7. Conjectures, evidence, and cheap refutation tests

PROVEN: No true conjecture can imply the unrestricted serial-MC lower bound as originally stated, because answer 5 gives a counterexample. Rare-tail hardness against polynomial-budget SD-normalized MC also needs repair because of the zero estimator.

Here are precise alternatives.

CONJECTURE M-design: Fix alpha>0 and p>0. With L=floor(alpha n), there are an integer s_0>p and c>0 such that

    sum_{s>=s_0} b_{2s} >= c V_sph/m_n^2

for all sufficiently large n.

PROVEN UNDER M-design: Every positive fixed spherical rule with N<=n^p has error at least a constant times V_sph/N, by (2.5). Thus it cannot obtain a growing gain over radially integrated MC at the same evaluation count.

HEURISTIC, evidence: The relevant object is the surviving angular kernel after its constant component is removed. Fixed-depth ReLU nonsmoothness supports nonzero high-degree mass, but does not establish its required magnitude at depth alpha n.

Cheapest refutation test: estimate the finite-width kernel and its even-degree mass across increasing n. An explicit sequence of positive designs with N R/V_sph -> 0 would directly refute the consequence and hence the conjecture. Small residual spectral masses require confidence intervals.

CONJECTURE M-whole, the bottleneck: After charging sampling and competing algorithms in the same arithmetic model, unrestricted whole-network estimators have at most a constant advantage over optimally implemented radial MC throughout a specified polynomial budget window.

More precisely, let C(n,L,m) denote the minimum cost, in the chosen model, of obtaining all outputs of m spherical sample inputs by full forward evaluation, including sample generation. Define

    B(T) = inf_{m : C(n,L,m)<=T} V_sph/m.

The conjecture is that, for fixed alpha and a specified polynomial budget window, some c>0 satisfies

    R(A_n) >= c B(T_n)

for every uniform estimator family of cost at most T_n in that window.

HEURISTIC, evidence: The supplied closure and partition experiments motivate testing this statement, but provide little evidence about unrestricted algorithms. M-design does not imply it. Neither a cut-rank bound nor a weight-degree bound supplies the needed computational bridge.

Cheapest refutation test: a uniform fixed design, analytic control variate, or global approximation whose charged advantage grows with n. Such a construction would settle the question more directly than further failure measurements of a fixed closure family.

CONJECTURE T-relative: Specify a computable threshold sequence t_n and constants 0<a<b for which

    P_W(e^{-bn} <= p_W,j(t_n) <= e^{-an}) >= 0.9.

On that rare-event ensemble, no polynomial-time algorithm estimates p_W,j(t_n) within relative error 1/2 with conditional success probability at least 2/3.

Both the rare-event window and the computational assertion are hypotheses; neither is proved here.

HEURISTIC, evidence: The union-of-polyhedra representation gives a possible counting mechanism, but no average-case reduction establishes hardness. Radial integration and importance sampling are serious competing explanations for easy tails.

Cheapest refutation test: first apply the exact radial formula (4.3), then test conditional or importance-sampling estimators for bounded relative variance and polynomial evaluation cost as n grows. The event threshold must be fixed by a reproducible rule, without using the final truth estimate to choose favorable instances.

PROVEN: T-relative concerns relative accuracy, not ARC's polynomial-budget SD-normalized criterion. For exponentially small p, ordinary MC needs exponentially many samples for constant relative error, while zero already performs well under the latter criterion.

8. Necessary conditions for any whole-network estimator

PROVEN: Every estimator must satisfy a quantitative correlation requirement. Let

    m = E_W mu(W),
    S = E_W ||mu(W)-m||^2,
    B = A-E A.

Define its scalar Hilbert-space correlation

    r = E<B,mu-m> / sqrt(E||B||^2 S).

Orthogonal projection onto the span of B gives

    R(A) >= S(1-r^2) + ||E A-m||^2.                     (8.1)

Consequently, achieving gain G over N-sample MC requires

    ||E A-m||^2 <= V/(NG),
    r^2 >= 1 - V/(NGS).                                 (8.2)

This concerns the actual mean vector, not the gain of one input trajectory or a proxy observable.

PROVEN: For L=alpha n with L>=2, S is of order n up to alpha-dependent constants.

Condition on all hidden-layer weights, and write m_h=E_x h_{L-1}(x). For final-layer row w_j,

    mu_j(w_j)
      = (1/2) w_j^T m_h
        + (1/2) E_x |w_j^T h_{L-1}(x)|.

The two terms are respectively odd and even in w_j and are orthogonal in Gaussian L^2. Hence

    S >= (1/2) E||m_h||^2
      >= n^2/[2 pi(n+2)] * (1+5/n)^(-(L-2))
      >= c_alpha n.                                    (8.3)

The upper bound S<=n follows from Jensen and critical second moments.

PROVEN: The same final-layer identity yields a stronger structural condition that applies to arbitrary estimators.

For each j, average only over the final row w_j, holding all other weights fixed, and define

    mhat_j = n E_{w_j}[w_j A_j(W)].

The target satisfies n E_{w_j}[w_j mu_j(W)] = m_h. Projecting A_j-mu_j onto the linear functions of w_j gives

    R(A) >= (1/(2n)) sum_j E||mhat_j-m_h||^2.            (8.4)

Therefore risk V/(NG) requires

    (1/n) sum_j E||mhat_j-m_h||^2 <= 2V/(NG).

Any successful whole-network estimator must encode the penultimate mean accurately in these final-row linear coefficients. This is a necessary condition, not an efficient reduction: computing those coefficients from an arbitrary estimator may itself be expensive.

PROVEN: For any chosen computational transcript or statistic T,

    R(A) >= E tr Cov(mu | T).

This becomes a lower bound only when T is genuinely restricted. Taking T=W makes the right side zero, so information loss cannot be assumed for an unrestricted white-box algorithm.

PROVEN: Analytic control variates give another exact cost-accuracy condition. If g_W has a computable integral and

    A = E_x g_W(x)
          + (1/M) sum_i [f_W(X_i)-g_W(X_i)],

then independent sampling gives

    R(A) = E_W tr Var_x(f_W-g_W)/M.

Thus a growing gain requires an appropriately shrinking residual variance after charging construction, integration, and evaluation of g_W. Approximation in an unrelated norm is insufficient.

PROVEN: Equations (8.1)-(8.4), the class-specific projection bounds, and accurate cost accounting are necessary conditions established here. They do not force an estimator to enumerate regions, propagate a bounded summary, approximate every intermediate law, or compute local gain factors separately. Those would be additional restrictions requiring their own justification.
tokens used
52,631
PROVEN: Determinism alone does not create the proposed average-case barrier. There is a uniform way to obtain sampling randomness from first-layer rotational symmetry.
PROVEN: With unrestricted arithmetic cost, batching and fast matrix multiplication already give a growing advantage over the stated serial-MC benchmark.
The remaining hardness questions require a revised benchmark or a restricted computational class.

Write f_W(x) = a_L(x), and define

    V = E_W tr Var_x f_W(x),
    R(A) = E_W ||A(W) - mu(W)||^2,
    m_n = E chi_n.

All results below concern sigma = 0. Exact distribution transforms require an explicit computational convention, addressed in answer 1. No files were edited.

1. Determinism and a uniform construction

PROVEN: There is an explicit uniform deterministic estimator with average error at most V/N. Its implementation uses N network evaluations, O(n^2) preprocessing, and O(nN) scalar distribution transforms. Thus the requested arithmetic-cost statement holds under the named hypothesis that these transforms have unit cost. With polynomial-accuracy transforms, it gives asymptotic matching in the usual finite-precision implementation described below.

Here is the construction for n >= 6.

Perform three steps of a right-looking orthogonal reduction of W_1. Obtain an orthogonal Q and B_1 = W_1 Q such that the first three rows of B_1 have zeros after coordinates 1, 2, and 3, respectively. Leave the other layers unchanged, obtaining B.

The Gaussian decomposition has two properties:

    mu(B) = mu(W),

and the orthogonal directions removed during these three steps are independent of B. They supply

    (n-1) + (n-2) + (n-3) = 3n-6

independent continuous angular coordinates. Applying their conditional CDFs supplies that many independent uniforms, independent of B.

Proof of independence: a Gaussian row decomposes into an independent length and uniform direction. Rotate its direction to the first coordinate. Conditional on that rotation, every remaining independent Gaussian row remains Gaussian with its original law. Repeat in the remaining orthogonal complement. This proves the assertion inductively. The identity for mu follows from Gaussian input rotation invariance.

Use 2n of those uniforms to form independent U,V in [0,1)^n. For i = 1,...,N, set coordinatewise

    Z_i = (U + i V) mod 1,
    X_i,j = Phi^{-1}(Z_i,j).

For distinct integers i,j, the map

    (U,V) -> (U+iV, U+jV) mod 1

preserves Haar probability on the two-dimensional torus. Consequently X_i and X_j are independent N(0,I_n), although the entire collection need not be mutually independent.

Define

    A(W) = (1/N) sum_i f_B(X_i).

Conditional on B, pairwise independence gives

    E[A | B] = mu(B),
    E[||A-mu(B)||^2 | B] = tr Var_x f_B(x)/N.

Averaging proves R(A) = V/N exactly. This is a deterministic function of W; no external random bits occur.

PROVEN: The construction is uniform rather than advice-based. Three Householder transformations cost O(n^2), and the torus recurrence and coordinate transforms use O(nN) scalar operations/transforms. It does not search for a favorable design or precompute answers for individual sizes.

PROVEN: Small widths do not obstruct uniformity. For n >= 4, three reduction steps supply enough uniforms to generate pairwise-independent spherical directions, requiring 2(n-1) coordinates. Integrating the Gaussian radius analytically then gives error at most V/N. For n = 2,3, supplement the orthogonal coordinates with the first-layer Frobenius radius R. Positive homogeneity gives f_W = R f_B after normalization. Under squared-error weighting, R has its R^2-size-biased law; applying that law's CDF supplies one additional uniform independent of the normalized network. The same proof, under this weighted law, gives the spherical-MC risk. For n = 1,

    mu(W) = [f_W(1) + f_W(-1)] / sqrt(2 pi),

so two evaluations suffice.

PROVEN UNDER polynomial-accuracy distribution transforms: For L = alpha n and N polynomial in n, finite-precision versions attain

    R(A) <= (1+o(1)) V/N

with lower-order preprocessing and sampling work. The relevant approximation need not survive an exponentially large worst-case Lipschitz bound: for an independent He layer,

    E_W ||relu(Wa)-relu(Wb)||^2 <= ||a-b||^2.

To prove this, average the squared difference under W and -W and use

    (relu(s)-relu(t))^2
      + (relu(-s)-relu(-t))^2 <= (s-t)^2.

Thus downstream layers do not amplify input quantization error in mean square. The canonicalized first layer has the same operator norm as W_1, with bounded expected squared norm. Polynomially small transform errors therefore suffice. Exact equality, however, belongs to the ideal continuous-transform model, not to a finite-bit machine.

HEURISTIC, proposed definition: A useful “mechanistic” restriction would specify an auditable expression language. For example, permit fixed, instance-independent cubature nodes and specified moment operations, while forbidding the use of weight-derived quantities as seeds that select evaluation points. This contains fixed designs and ordinary moment closures and excludes the construction above.

PROVEN: Determinism, continuity, or numerical stability alone does not implement that restriction. Fixed-design network evaluation is itself continuous in the weights. Requiring invariance under every input rotation would also exclude ordinary fixed designs. There is no automatic semantic distinction between “deduction” and “sampling” in the final numerical answer.

2. Fixed point sets and adaptive evaluations

PROVEN: The average error of every fixed linear rule is exactly a kernel discrepancy.

On the unit sphere, define the finite-width, finite-depth kernel

    k(t) = E_W <f_W(u), f_W(v)>,     <u,v> = t.

Rotation invariance makes this well-defined, and critical initialization gives k(1) = 1. Put

    q = E_{U,V uniform on sphere} k(<U,V>).

Positive homogeneity gives

    E_W ||mu(W)||^2 = m_n^2 q,
    V = n - m_n^2 q.                                      (2.1)

For a rule A = sum_i w_i f_W(r_i u_i), write c_i = w_i r_i. Its exact risk is

    R(A)
      = sum_{i,j} c_i c_j k(<u_i,u_j>)
        - 2 m_n q sum_i c_i + m_n^2 q.                    (2.2)

This follows simply by expanding the square and integrating its three terms.

PROVEN: The harmonic version isolates exactly what a design must integrate. Let Y_{ell,m} be an orthonormal spherical-harmonic basis for probability surface measure, and write

    k(<u,v>) =
      sum_{ell >= 0} lambda_ell sum_m Y_{ell,m}(u)Y_{ell,m}(v).

Then lambda_ell >= 0, lambda_0 = q, and

    R(A)
      = q (sum_i c_i - m_n)^2
        + sum_{ell >= 1} lambda_ell
            sum_m (sum_i c_i Y_{ell,m}(u_i))^2.           (2.3)

In particular, a spherical t-design with c_i = m_n/N removes all degrees 1,...,t. Its remaining error depends on both the spectral tail and the discrepancies of the design at higher degrees.

PROVEN: For this ensemble there is an additional, useful power-series representation:

    k(t) = sum_{d >= 0} b_d t^d,
    b_d >= 0,
    sum_d b_d = 1.                                       (2.4)

Proof: condition on layers 2,...,L. The first-layer preactivations at u,v are Gaussian vectors with coordinatewise correlation t. Expand the downstream function of that Gaussian vector in multivariate Hermite polynomials. The identity

    E[H_a(Z) H_b(Z')] = 1_{a=b} t^{|a|}

gives (2.4), with b_d the expected squared norm of the degree-d Hermite coefficients.

This representation yields an explicit limitation on positive-weight rules. Define

    tau_{2s} = E <U,V>^{2s}
             = (2s-1)!! / [n(n+2)...(n+2s-2)].

For any positive spherical rule with N nodes and weights summing to one,

    R(A) >= (m_n^2 / (2N))
              sum_{s : tau_{2s} <= 1/(2N)} b_{2s}.        (2.5)

Proof: for the kernel t^d, the discrepancy equals

    ||sum_i w_i u_i^{tensor d} - E U^{tensor d}||^2,

so every degree contributes nonnegatively. For even d, all off-diagonal inner-product powers are nonnegative. Its discrepancy is therefore at least

    sum_i w_i^2 - tau_d >= 1/N - tau_d.

Sum the selected even degrees.

Consequently the gain of any such rule satisfies

    gain <=
      2V / [m_n^2 sum_{s : tau_{2s} <= 1/(2N)} b_{2s}].   (2.6)

For N <= n^p, fixed even degrees 2s with s > p eventually qualify. The unresolved quantity is their total coefficient mass at L = alpha n, relative to V/m_n^2.

PROVEN: Signed weights require a different bound. For a degree-2s term, flatten u^{tensor 2s} into the rank-one matrix

    u^{tensor s} (u^{tensor s})^T.

The quadrature approximation has rank at most N, even with signed weights. If

    M_s = E[U^{tensor s}(U^{tensor s})^T],

then its degree-2s discrepancy is at least the squared singular-value tail of M_s after rank N. Multiplication by m_n^2 b_{2s} gives a valid risk lower bound. This is a moment-matrix bound, not a running-time bound.

PROVEN: Radial integration is an immediate improvement that must be included in comparisons. Pairwise-independent spherical sampling has risk

    V_sph/N,     V_sph = m_n^2(1-q),

whereas

    V = (n-m_n^2) + V_sph.                               (2.7)

Here n-m_n^2 -> 1/2. If V_sph is much smaller than this radial term, radial integration gives a growing improvement over ordinary Gaussian-input MC.

PROVEN: A simple explicit design already matches spherical MC for N <= n. Take N orthogonal directions. Since k(0) = b_0 <= q,

    R = m_n^2 [1/N + (1-1/N)k(0) - q]
      <= m_n^2(1-q)/N.

PROVEN: For adaptive evaluation algorithms, the exact general characterization is instead Bayesian. If T is the complete evaluation transcript, then

    inf_{A measurable from T} E||A-mu||^2
      = E tr Cov(mu | T).                               (2.8)

The covariance kernel alone does not determine this quantity for a non-Gaussian prior.

PROVEN UNDER a Gaussian function prior: Adaptivity cannot improve optimal average squared error for estimating a linear functional from noiseless point evaluations. Gaussian conditioning makes posterior covariance depend only on the queried locations, not the observed values. Every realized adaptive sequence has posterior variance at least the minimum over fixed sequences; a minimizing fixed sequence attains that minimum.

That hypothesis is essential. The finite-width network prior is non-Gaussian; a nondegenerate final ReLU output is also non-Gaussian even when its preactivation has a Gaussian-process limit. Thus the Gaussian-prior theorem does not settle adaptive evaluation of the stated network.

CONJECTURE: A constant-gain bound for fixed or adaptive evaluations in the joint limit remains a separate assertion. The supplied fixed-depth spectral asymptotic does not establish it: its constants and crossover degrees must be controlled uniformly when L grows with n.

3. Polynomial estimators, stability, and statistical queries

PROVEN: Degree-D polynomials in the weights have an exact optimal-error formula.

Standardize all weights into G ~ N(0,I_M), M = n^2 L, and expand

    mu(G) = sum_a mu_a H_a(G),
    E_k = sum_{|a|=k} ||mu_a||^2.

Then

    inf_{degree(A) <= D} R(A) = sum_{k>D} E_k.            (3.1)

Proof: orthogonal projection in Gaussian L^2. Instance-independent coefficients are precisely the coefficients permitted in this projection problem.

This is a genuine restricted-class lower bound. It neither says that computing the optimal coefficients is cheap nor that efficient whole-network algorithms have low polynomial degree.

PROVEN: Weight-noise stability measures this spectrum. For independent G,G', set

    G_rho = rho G + sqrt(1-rho^2) G',
    S(rho) = E <mu(G)-E mu, mu(G_rho)-E mu>.

Then

    S(rho) = sum_{k>=1} rho^k E_k.                       (3.2)

Writing T = S(1), for 0 < rho < 1,

    max(0, T-rho^{-D}S(rho))
        <= sum_{k>D} E_k
        <= [T-S(rho)] / [1-rho^{D+1}].                  (3.3)

The lower bound follows from
S(rho) >= rho^D sum_{1<=k<=D} E_k.
The upper bound follows by retaining degrees above D in T-S(rho).

PROVEN: An unbiased measurement of the uncentered quantity in (3.2) uses independent inputs X,Y:

    E <f_G(X), f_{G_rho}(Y)>.

Using the same input at both networks estimates a different observable. Resolving a small high-degree tail from noisy stability measurements may itself be statistically expensive.

PROVEN: Stability gives a direct inequality without an overlap-gap hypothesis. Define

    d_mu(rho)^2 = E||mu(G)-mu(G_rho)||^2,
    d_A(rho)^2  = E||A(G)-A(G_rho)||^2.

The L^2 triangle inequality gives

    R(A) >= (max(0, d_mu(rho)-d_A(rho)))^2 / 4.          (3.4)

Thus an algorithm constrained to vary substantially less than the target under a particular perturbation must fail.

PROVEN: The usual overlap-gap argument is not supplied merely by calling an estimator “stable.” For each fixed W, acceptable estimates under squared loss form a Euclidean ball around mu(W), hence a connected convex set. An overlap-gap application would need an additional search representation, an overlap theorem for that representation, and a reduction from accurate mean estimation to that search problem. None follows from the present formulation.

PROVEN: An unrestricted statistical-query oracle can directly answer the question being studied. For a tail it can query

    phi(x) = 1{f_W,j(x)>t}.

For a nonnegative mean, query min(f_W,j(x),B)/B. With oracle tolerance tau, the resulting mean error is at most

    B tau + E_x f_W,j(x)^2 / B.

The second term follows from y 1{y>B} <= y^2/B.

Therefore an SQ lower bound needs restrictions on permitted queries, a tolerance cost, or a different observation model. Allowing arbitrary precise expectation queries makes mean estimation immediate; charging their implementation by sampling mostly builds sampling cost into the oracle. Standard SQ hardness statements cannot be transferred to a white-box algorithm without an explicit simulation theorem.

4. Connections to counting, volume, free energy, circuits, and permanents

PROVEN: There is an exact Gaussian-volume correspondence, but it is a representation rather than an average-case hardness reduction.

For each full gate pattern s, let C_s be its input cone and B_s the corresponding linear map. Up to null boundaries,

    f_W(x) = B_s x on C_s,

and therefore

    mu_j(W) = sum_s b_s,j^T E[X 1{X in C_s}],            (4.1)

    p_W,j(t) = sum_s gamma_n(
                     C_s intersect {b_s,j^T x > t}).     (4.2)

The cones are defined by the linear inequalities enforcing all gates along that pattern. Equation (4.1) is a sum of Gaussian first moments; (4.2) is a sum of Gaussian polyhedral volumes. There can be exponentially many terms, and their coefficients and inequalities are highly dependent functions of W.

PROVEN: Homogeneity also gives the exact radial formulas

    mu_j = m_n E_U f_W,j(U),

    p_W,j(t) =
      E_U Q_chi_n(t/f_W,j(U)),                           (4.3)

with the second integrand defined as zero when f_W,j(U)=0 and t>0. These formulas provide algorithms as well as structural correspondences.

PROVEN: The error metric prevents a direct transfer from relative counting or free-energy estimation. If a positive integral Z represents a partition function and

    F = log Z / (beta m),

then, locally,

    |delta F| approximately |delta Z| / (beta m Z).

ARC's allowed error eps sqrt(Var Y) therefore corresponds to free-energy error approximately

    eps * [sqrt(Var Y)/E Y] / (beta m).

An exponentially large coefficient of variation makes this a very weak free-energy guarantee. Conversely, constant additive error in F allows an exponentially large multiplicative error in Z.

PROVEN: A rare-event tail has an especially important mismatch. For a fixed W with probability p, the zero estimator has squared error p^2, while N-sample MC has error p(1-p)/N. Its gain over MC is

    (1-p)/(Np).

Thus p << 1/N makes zero a growing-factor improvement. Averaged over W, the exact ratio is

    E[p(1-p)] / (N E[p^2]).

A statement that p is “usually tiny” is insufficient if rare weights dominate E[p^2].

PROVEN: Mean estimation need not resolve tiny tails. For Y >= 0,

    E[Y 1{Y>t}] <= sqrt(E[Y^2] P(Y>t)).

This may put the entire tail contribution below the mean estimator's error allowance.

HEURISTIC, correspondence assessment: Spin-glass free energies, random-circuit output probabilities, and permanents suggest possible mechanisms, but none supplies a distribution-preserving, variance-calibrated reduction to iid He-ReLU weights here. Such a reduction would have to preserve both the random-instance law and the relevant error scale. Worst-case circuit encodings do neither automatically. Permanent and quantum-amplitude problems also have sign or complex-cancellation structures absent from the nonnegative network output.

No hardness consequence from those analogies is established in this attempt.

5. Complexity barriers and a counterexample to the proposed lower bound

PROVEN: Under the stated unrestricted arithmetic cost, “no algorithm beats serial MC by more than a constant” is false.

Let omega = log_2 7 < 3, using Strassen multiplication. Evaluate a batch of n inputs simultaneously. Each layer multiplies two n by n matrices and then applies n^2 ReLUs, so total cost is

    T = O(L n^omega).

The deterministic construction in answer 1 supplies n pairwise-independent inputs at lower-order overhead and achieves risk at most V/n.

At this same cost, the stated serial-MC benchmark buys only

    N_serial = Theta(n^{omega-2})

forward passes, with error Theta(V/n^{omega-2}). Hence the deterministic algorithm's gain is at least

    Omega(n^{3-omega}).

For L = alpha n, the budget remains polynomial. This improvement is computational batching; it does not require a better inference method.

PROVEN: The objection disappears if MC is charged using the same batched evaluation algorithm. Consequently a meaningful unrestricted lower bound must compare against an equally optimized sampling implementation, or explicitly restrict the arithmetic model. The distinction affects exponents, not just constants.

PROVEN: Even after that repair, a polynomial-factor lower bound between two polynomial running times does not by itself imply P != NP, NEXP not in P/poly, or a superpolynomial circuit lower bound. A uniform time lower bound also does not automatically become a nonuniform circuit lower bound.

PROVEN: The supplied IKW implication concerns sufficiently strong deterministic estimation on all encoded circuits. An estimator succeeding on average over iid Gaussian weights does not give that guarantee without a reduction taking arbitrary circuits to suitably distributed random weights. The symmetry construction in answer 1 illustrates why average-case determinism can be cheap while worst-case derandomization remains difficult.

The IKW 2002 and Chen-Tell 2021 results are used here only as supplied premises; their exact theorem formulations and bibliographic details are unverified in this attempt.

PROVEN: A natural-proofs barrier does not apply merely because a claim averages over random weights. Applying that framework would require an associated property of Boolean truth tables that is constructive, large, and useful against a specified circuit class. Gaussian measure on weight space is not automatically the required largeness condition on truth tables. No such transformation is provided here.

HEURISTIC: A corrected unrestricted arithmetic lower bound would nevertheless be a substantial complexity result. Kernel spectra, region counts, and low-degree failure provide restricted-class statements; they do not currently supply the missing bridge to arbitrary computations on all weights.

6. Branch-and-bound and region integration

PROVEN: The first layer has exactly 2^n nonempty full-dimensional gate cones almost surely. Gaussian W_1 is invertible almost surely, and inverse images of the 2^n open orthants give those cones. Every one intersects the sphere.

There are at most 2^{nL} full gate histories. This counts activation cells, not distinct maximal linear regions of the final function. They can merge: for example, if every entry of W_2 is negative, all later outputs vanish. That event has positive probability when L >= 2.

PROVEN: More strongly, polynomially many first-layer cells cover exponentially little input probability on average, even when selected after reading all weights.

Let p_s(W_1) be the Gaussian probability of first-layer sign pattern s, and put

    kappa_n = E_W sum_s p_s(W_1)^2.

For independent spherical U,V and T=<U,V>,

    kappa_n =
      E_T (1/2 + arcsin(T)/pi)^n.                        (6.1)

Indeed, one Gaussian row gives the same sign at U,V with the probability in parentheses, and rows are independent.

Splitting at T=1/2 gives

    kappa_n <= (2/3)^n + P(T>1/2)
             <= (2/3)^n
                + C sqrt(n) (3/4)^((n-3)/2).            (6.2)

The final inequality follows by integrating the spherical-coordinate density proportional to (1-t^2)^((n-3)/2).

For any weight-dependent selection S(W) of at most K patterns, Cauchy-Schwarz gives pointwise

    [sum_{s in S(W)} p_s]^2 <= K sum_s p_s^2.

Therefore

    E_W GaussianMass(selected cells) <= sqrt(K kappa_n). (6.3)

For polynomial K this is exponentially small.

PROVEN: This yields an actual error lower bound for methods that exactly integrate selected cells and discard everything else.

Let

    A_S(W) = E_X[f_W(X) 1{selected first-layer cells}].

Write Q(W)=E_X||f_W(X)||^2. Conditional Cauchy-Schwarz gives

    ||A_S(W)||^2 <= p_S(W) Q(W).

The critical fourth moment is

    E_W,X ||f_W(X)||^4
      = n(n+2)(1+5/n)^L.

This follows from the per-layer squared-norm multiplier
(2/n) sum_i (Z_i^+)^2, whose second moment is 1+5/n. Hence, for L=alpha n,

    E||A_S||^2 <= C_alpha n sqrt(K kappa_n).             (6.4)

On the other hand,

    E||mu(W)||^2
      >= n^2/[pi(n+2)] * (1+5/n)^(-(L-1))
      >= c_alpha n.                                    (6.5)

For completeness, if H=||h_{L-1}(X)||, then

    E mu_j = E H / sqrt(pi n),

and Holder gives

    E H >= (E H^2)^(3/2)/(E H^4)^(1/2).

Substituting the second and fourth moments proves (6.5).

The reverse L^2 triangle inequality applied to (6.4)-(6.5) now gives

    R(A_S) >= c'_alpha n

for polynomial K and sufficiently large n.

This theorem does not cover methods that bound, approximate, or analytically combine the unenumerated cells.

PROVEN UNDER a geometric mesh hypothesis: A useful upper rate is available for deterministic cubature exact on linear cells.

Partition S^{n-1} into M cells of diameter at most h, with

    M <= C h^{-(n-1)}.

Suppose their probabilities p_C and vector centroids v_C=E[U|C] are available. Use

    A = m_n sum_C p_C f_W(v_C).

A cell contained in one activation cone is integrated exactly. If f_W has Lipschitz constant K_W and its crease set has tubular measure at most B_W h, only boundary cells contribute error. Thus

    ||A-mu|| <= m_n K_W B_W h^2
              <= C_W M^{-2/(n-1)}.                     (6.6)

Proof: each boundary cell contributes at most its mass times m_n K_W h, and their total mass is at most B_W h.

Finite polyhedral crease sets satisfy such a tubular bound at fixed architecture. The constants can be enormous and have not been controlled uniformly in n,L.

PROVEN UNDER an adaptive mesh hypothesis: If a certified mesh needs only O(h^{-(n-2)}) cells while keeping every boundary cell's diameter at most h, the same proof gives error O_W(M^{-2/(n-2)}) for n>2. Constructing such a mesh, certifying gates, and computing cell moments must all be charged.

HEURISTIC: These rates explain a possible dimensional obstruction, but they are upper bounds, not lower bounds. Region counts alone cannot determine the cost of downstream-informed branch-and-bound. Exact evaluation on a region also does not make its Gaussian first moment or probability free.

7. Conjectures, evidence, and cheap refutation tests

PROVEN: No true conjecture can imply the unrestricted serial-MC lower bound as originally stated, because answer 5 gives a counterexample. Rare-tail hardness against polynomial-budget SD-normalized MC also needs repair because of the zero estimator.

Here are precise alternatives.

CONJECTURE M-design: Fix alpha>0 and p>0. With L=floor(alpha n), there are an integer s_0>p and c>0 such that

    sum_{s>=s_0} b_{2s} >= c V_sph/m_n^2

for all sufficiently large n.

PROVEN UNDER M-design: Every positive fixed spherical rule with N<=n^p has error at least a constant times V_sph/N, by (2.5). Thus it cannot obtain a growing gain over radially integrated MC at the same evaluation count.

HEURISTIC, evidence: The relevant object is the surviving angular kernel after its constant component is removed. Fixed-depth ReLU nonsmoothness supports nonzero high-degree mass, but does not establish its required magnitude at depth alpha n.

Cheapest refutation test: estimate the finite-width kernel and its even-degree mass across increasing n. An explicit sequence of positive designs with N R/V_sph -> 0 would directly refute the consequence and hence the conjecture. Small residual spectral masses require confidence intervals.

CONJECTURE M-whole, the bottleneck: After charging sampling and competing algorithms in the same arithmetic model, unrestricted whole-network estimators have at most a constant advantage over optimally implemented radial MC throughout a specified polynomial budget window.

More precisely, let C(n,L,m) denote the minimum cost, in the chosen model, of obtaining all outputs of m spherical sample inputs by full forward evaluation, including sample generation. Define

    B(T) = inf_{m : C(n,L,m)<=T} V_sph/m.

The conjecture is that, for fixed alpha and a specified polynomial budget window, some c>0 satisfies

    R(A_n) >= c B(T_n)

for every uniform estimator family of cost at most T_n in that window.

HEURISTIC, evidence: The supplied closure and partition experiments motivate testing this statement, but provide little evidence about unrestricted algorithms. M-design does not imply it. Neither a cut-rank bound nor a weight-degree bound supplies the needed computational bridge.

Cheapest refutation test: a uniform fixed design, analytic control variate, or global approximation whose charged advantage grows with n. Such a construction would settle the question more directly than further failure measurements of a fixed closure family.

CONJECTURE T-relative: Specify a computable threshold sequence t_n and constants 0<a<b for which

    P_W(e^{-bn} <= p_W,j(t_n) <= e^{-an}) >= 0.9.

On that rare-event ensemble, no polynomial-time algorithm estimates p_W,j(t_n) within relative error 1/2 with conditional success probability at least 2/3.

Both the rare-event window and the computational assertion are hypotheses; neither is proved here.

HEURISTIC, evidence: The union-of-polyhedra representation gives a possible counting mechanism, but no average-case reduction establishes hardness. Radial integration and importance sampling are serious competing explanations for easy tails.

Cheapest refutation test: first apply the exact radial formula (4.3), then test conditional or importance-sampling estimators for bounded relative variance and polynomial evaluation cost as n grows. The event threshold must be fixed by a reproducible rule, without using the final truth estimate to choose favorable instances.

PROVEN: T-relative concerns relative accuracy, not ARC's polynomial-budget SD-normalized criterion. For exponentially small p, ordinary MC needs exponentially many samples for constant relative error, while zero already performs well under the latter criterion.

8. Necessary conditions for any whole-network estimator

PROVEN: Every estimator must satisfy a quantitative correlation requirement. Let

    m = E_W mu(W),
    S = E_W ||mu(W)-m||^2,
    B = A-E A.

Define its scalar Hilbert-space correlation

    r = E<B,mu-m> / sqrt(E||B||^2 S).

Orthogonal projection onto the span of B gives

    R(A) >= S(1-r^2) + ||E A-m||^2.                     (8.1)

Consequently, achieving gain G over N-sample MC requires

    ||E A-m||^2 <= V/(NG),
    r^2 >= 1 - V/(NGS).                                 (8.2)

This concerns the actual mean vector, not the gain of one input trajectory or a proxy observable.

PROVEN: For L=alpha n with L>=2, S is of order n up to alpha-dependent constants.

Condition on all hidden-layer weights, and write m_h=E_x h_{L-1}(x). For final-layer row w_j,

    mu_j(w_j)
      = (1/2) w_j^T m_h
        + (1/2) E_x |w_j^T h_{L-1}(x)|.

The two terms are respectively odd and even in w_j and are orthogonal in Gaussian L^2. Hence

    S >= (1/2) E||m_h||^2
      >= n^2/[2 pi(n+2)] * (1+5/n)^(-(L-2))
      >= c_alpha n.                                    (8.3)

The upper bound S<=n follows from Jensen and critical second moments.

PROVEN: The same final-layer identity yields a stronger structural condition that applies to arbitrary estimators.

For each j, average only over the final row w_j, holding all other weights fixed, and define

    mhat_j = n E_{w_j}[w_j A_j(W)].

The target satisfies n E_{w_j}[w_j mu_j(W)] = m_h. Projecting A_j-mu_j onto the linear functions of w_j gives

    R(A) >= (1/(2n)) sum_j E||mhat_j-m_h||^2.            (8.4)

Therefore risk V/(NG) requires

    (1/n) sum_j E||mhat_j-m_h||^2 <= 2V/(NG).

Any successful whole-network estimator must encode the penultimate mean accurately in these final-row linear coefficients. This is a necessary condition, not an efficient reduction: computing those coefficients from an arbitrary estimator may itself be expensive.

PROVEN: For any chosen computational transcript or statistic T,

    R(A) >= E tr Cov(mu | T).

This becomes a lower bound only when T is genuinely restricted. Taking T=W makes the right side zero, so information loss cannot be assumed for an unrestricted white-box algorithm.

PROVEN: Analytic control variates give another exact cost-accuracy condition. If g_W has a computable integral and

    A = E_x g_W(x)
          + (1/M) sum_i [f_W(X_i)-g_W(X_i)],

then independent sampling gives

    R(A) = E_W tr Var_x(f_W-g_W)/M.

Thus a growing gain requires an appropriately shrinking residual variance after charging construction, integration, and evaluation of g_W. Approximation in an unrelated norm is insufficient.

PROVEN: Equations (8.1)-(8.4), the class-specific projection bounds, and accurate cost accounting are necessary conditions established here. They do not force an estimator to enumerate regions, propagate a bounded summary, approximate every intermediate law, or compute local gain factors separately. Those would be additional restrictions requiring their own justification.
