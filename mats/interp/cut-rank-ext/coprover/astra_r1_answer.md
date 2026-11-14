codex
1. Target 1: verdicts and bounds

PROVEN: At fixed architecture, the linear precision exponent does not survive unrestricted continuous nonlinear readouts, bounded-precision communication, or unrestricted continuous finite-moment closures. It does survive a fixed bound on polynomial degree, or more generally on the readout’s separation dimension.

The distinction is between communication and computation. Sending the front weights communicates a finite-dimensional description; evaluating the resulting Gaussian integral can still be expensive.

Write p = cn², q = dn², δ = ε√Vbar, and

  ||v||_n = ||v||_2/√n,
  H = L²(back weights; R^n with norm ||·||_n),
  F(f) = M(f,·).

Assume n ≥ 2 and c,d ≥ 1 unless stated otherwise. All protocols below are chosen before drawing the weights. Their functions may depend on ε.

1.a. Arbitrary or continuous real messages

PROVEN: p real numbers suffice exactly, with a continuous encoder and a jointly continuous decoder:

  φ(f) = the cn² front weights,
  R(φ,b) = E_x h_L(x;φ,b).

This decoder is locally Lipschitz in all weights. Indeed, if every matrix on a line segment has operator norm at most R, the telescoping inequality gives

  ||M(f,b) − M(f′,b)||_n
  ≤ √c R^(L−1) ||f−f′||_2.

The bound follows from ReLU being 1-Lipschitz and E||x|| ≤ √n. Analogous bounds hold when the back changes.

Thus a single continuous protocol with s = p works for every ε, including ε = 0. Its computational cost is uncharged.

PROVEN: For the stated averaged criterion, even one bounded continuous real message suffices to achieve any positive error. Both encoder and decoder can be Lipschitz, with constants depending on the requested error.

Here is the construction and proof. Regard F as an H-valued square-integrable random variable. Approximate it in L² by a finite simple function taking values F_1,…,F_N, chosen from realizable front laws. By regularity of the front distribution, choose disjoint compact subsets K_i of the corresponding cells, losing arbitrarily little of the finite measure

  (1 + ||F(f)||_H²) dP_f.

Assign distinct scalar labels t_i ∈ [0,1]. Because finitely many disjoint compact sets have positive pairwise distances, the map φ = t_i on K_i has a finite Lipschitz extension to the whole front space, clipped to [0,1]. Define R(t,·) by linear interpolation between the F_i, extending constantly beyond the extreme labels.

On K_i, the approximation is F_i. Off their union,

  ||F − R(φ)||_H² ≤ 2||F||_H² + 2 max_i ||F_i||_H²,

so the discarded weighted measure controls the error.

If a Lipschitz constant uniform over b is required, first clip the back weights, as below. The finitely many interpolated functions then have a finite uniform Lipschitz constant in t.

The same construction works with φ a function of the cut law alone, using compact subsets of the space of laws with its W₁ metric. It also works for continuity in the weak topology: measurability and regularity of the induced law distribution suffice.

This scalar construction uses increasingly unstable encodings as ε decreases. It does not produce one fixed scalar continuous protocol that is exact.

The exact failure of the linear argument is K1: nonlinear decoding does not imply that the family {R(φ(f),·)} lies in an s-dimensional linear subspace of H. A curved family with a finite-dimensional parameterization can have an infinite-rank covariance operator.

On a ray this is especially explicit:

  h = R u  ⇒  M(f,b) = (E R) g_b(u) = g_b(E h).

The n entries of E h suffice exactly. The gate kinks responsible for K4 are evaluated by the decoder.

1.b. Lipschitz readouts and bounded encoders

“Lipschitz” needs a metric, a bound, and a quantifier over b. The following separates the relevant cases.

PROVEN: With weights as encoder input, a bounded encoder of fixed dimension p and a readout Lipschitz constant growing only polylogarithmically in 1/ε suffice.

Let F_l = ||W_l||_F and replace each matrix by

  W_l^R = min(1,R/F_l) W_l.

Positive homogeneity gives the exact identity

  M(W_1^R,…,W_L^R)
  = [∏_l min(1,R/F_l)] M(W_1,…,W_L).

Consequently,

  E ||M − M^R||_n² ≤ C_0 exp(−nR²/8),
  C_0 = L(2n)^L 2^(n²/2+1).

To verify the constant, use ||M||_n² ≤ ∏F_l², independence, and

  E F_l² = 2n,
  E[F_l² exp(nF_l²/8)] = 2n·2^(n²/2+1).

Choose

  R = max{1, [(8/n) log(4C_0/δ²)]^(1/2)}.

Send φ(f) = f^R/R. Then ||φ||_2 ≤ √c, and φ is 1/R-Lipschitz. The decoder clips each supplied front block to the unit Frobenius ball, rescales by R, clips its own back matrices, and evaluates the expectation. Its Lipschitz constant, for normalized vector output, is at most

  Λ_ε = √c R^L.

Its error is at most δ²/4. Componentwise Lipschitz constants are at most √n times this bound.

Thus bounded encoding and polylogarithmically growing stability constants do not preserve the power law.

PROVEN: A fixed Lipschitz bound and a fixed global Euclidean radius instead impose an error floor, regardless of s.

Suppose ||φ(f)|| ≤ A and

  ||R(z,b) − R(z′,b)||_n ≤ Λ||z−z′||

uniformly in b. For independent f,f′,

  Err ≥ (1/4) E[
    (||M(f,b)−M(f′,b)||_n − 2AΛ)_+²
  ].

This follows by the triangle inequality and (a+b)² ≤ 2a²+2b². The right side is positive for every finite AΛ: the target’s variation is unbounded under Gaussian weight scaling.

The same argument applies to an H-Lipschitz decoder. If instead both encoder and decoder have fixed global Lipschitz constants L_φ and Λ, replace 2AΛ by ΛL_φ||f−f′||. For a decoder bound uniform in b, back-weight scaling again gives a positive error floor.

This is infeasibility of arbitrarily high precision under fixed stability budgets, not an ε-dependent dimension lower bound.

PROVEN: Coordinatewise boundedness is different from a fixed Euclidean radius. With |φ_i| ≤ 1 and a fixed decoder Lipschitz constant Λ, polylogarithmically many coordinates suffice.

Repeat the normalized clipped front vector k times. The decoder averages the copies. Its Lipschitz constant becomes √c R^L/√k. Taking

  k = max{1, ceil(cR^(2L)/Λ²)}

gives s = pk = O_arch,Λ((log(1/ε))^L). The encoder’s global Lipschitz constant now grows.

PROVEN UNDER a compact-source small-ball hypothesis: the natural stable-width obstruction is an intrinsic-dimension bound.

Suppose, on an event of positive probability, a k-dimensional cube of parameters z has density bounded below and

  ||F_z − F_z′||_H ≥ a||z−z′||.

The quantization lemma in Section 3 gives a constant C > 0. If an encoder has radius A, the decoder is H-Lipschitz with constant Λ, and its RMS error is at most δ, covering the encoder ball gives

  s log(1 + 2AΛ/δ) ≥ (k/2) log(C/(4δ²)).

Indeed, rounding the encoding to a δ/Λ-net produces at most
(1+2AΛ/δ)^s reconstruction functions, with RMS error at most 2δ.

For fixed stability budgets on a compact family, this forces asymptotically s ≥ k. It does not force s to be a power of 1/ε.

1.c. Bit messages

PROVEN: For the genuine ensemble with c ≥ 3, arbitrary one-way bit protocols satisfy

  B ≥ n log₂(1/ε) + O_arch(1),

and there are protocols satisfying

  B ≤ p log₂(1/ε) + O_arch(log log(1/ε)).

The lower bound remains valid for arbitrary interaction and randomization; its proof is in Sections 2–3.

For the upper bound, clip all weights as above. Quantize the p front entries on a grid of spacing

  η = δ/[4√(cp) R^(L−1)],

then project each quantized matrix back to the radius-R Frobenius ball. The resulting front error is at most δ/4 in normalized output norm. Clipping costs at most δ/2 in RMS. A sufficient bit count is

  B ≤ p ceil log₂(2 + 8√(cp) R^L/δ).

Hence the number of codewords has power-law growth, but the communication cost—the logarithm of that number—does not.

The failure of the linear-rank argument is precise: B bits permit 2^B reconstruction functions, not B reconstruction functions.

1.d. Moments, cumulants, and ARC-style closures

PROVEN: For fixed n and c, some finite moment order determines every cut law exactly. With a sufficiently high finite order, there is an exact continuous final readout.

The moment order is architecture-dependent and non-effective in the proof below. It is independent of ε.

More generally, this holds for every distribution obtained by applying a bounded-complexity, homogeneous piecewise-linear map to a standard Gaussian. Section 3 proves it by Gaussian integration by parts on the finitely many cone faces, followed by the Hilbert basis theorem.

Thus, for some finite Q(n,c),

  φ(f) = (E h^α : 1 ≤ |α| ≤ Q)

admits an exact continuous decoder. Its dimension is

  binom(n+Q,Q) − 1,

independent of ε. Cumulants contain exactly the same information: the finite moment–cumulant transformations are polynomial inverses.

This is an existence theorem for an unrestricted continuous closure. It does not assert that Gaussian/Wick propagation implements that decoder.

There is also an exact layerwise interpretation. Choose an even moment order sufficiently high to identify all cone-linear laws arising through depth L. On realizable moment vectors:

  reconstruct the law;
  apply h ↦ ReLU(Wh);
  take its moments again.

This update is continuous on the realizable moment domain. If every intermediate update must extend continuously to the entire ambient Euclidean space, use a fixed descending sequence of even moment orders, each at least two larger than the next. The closed-set extension argument in Section 3 then applies at every step. The initial order remains finite and independent of ε.

PROVEN: For a specified order K, the exact information limit is

  e_K² = E ||F − E[F | Z_K]||_H²,

where Z_K is the vector of moments through order K. This is the infimum error over all measurable readouts, and also the infimum over continuous readouts. Continuous functions are dense in L² of the finite-dimensional joint distribution of (Z_K,b).

Therefore:

  e_K = 0: these moments suffice, although a particular closure may fail.
  e_K > 0: every closure using only these moments has an error floor.

The finite-determinacy theorem implies e_K = 0 for all sufficiently large K.

Two concrete cases:

PROVEN: For c = 1, moments through order two suffice. If Z ~ N(0,Σ), h = ReLU(Z), then

  E h_i = √Σ_ii/√(2π),

and, with ρ = Σ_ij/√(Σ_iiΣ_jj),

  E h_i h_j
  = √(Σ_iiΣ_jj)
    [√(1−ρ²) + (π−arccos ρ)ρ]/(2π).

The bracketed function is strictly increasing on [−1,1]. Thus these moments recover Σ and hence the entire cut law. The inverse is continuous, including zero-variance coordinates.

PROVEN: First moments alone have a positive error floor for n ≥ 2, c,d ≥ 1.

Condition on front matrices W_2,…,W_c being strictly positive and invertible, an event of positive probability. Then

  h_c = A ReLU(W_1x),
  E h_c = A s/√(2π),

where s contains the row norms of W_1. Conditional on A and s, the first moment is independent of the row directions.

Choose the first two row directions equal or opposite. A back row satisfying

  wᵀA = (1/s_1, −1/s_2, 0,…,0)

has expected ReLU output zero in the equal case and 1/√(2π) in the opposite case. Later back layers can transmit that difference. Continuity supplies neighborhoods of directions and back weights of positive probability. Hence the conditional variance given the first moment is positive.

1.e. Polynomial, analytic, and finite-separation readouts

PROVEN: Fixed polynomial degree preserves a power-law lower bound.

Suppose R(z,b) is a polynomial in z of total degree at most D, with arbitrary b-dependent coefficients. Then

  R(φ(f),b) = Σ_{|α|≤D} φ(f)^α ψ_α(b),

so its separation rank is at most binom(s+D,D).

Let a = 2(n−1)/3, and let C_* > 0 and ε_0 be constants for which K4 or K5 gives r(ε) ≥ C_*ε^(−a). For ε < ε_0,

  s ≥ (D! C_*)^(1/D) ε^(−a/D) − D.

This is the complete lower-bound proof: expand the polynomial and apply K1. For fixed D independent of n, its exponent is Θ(n).

PROVEN: If polynomial degree is unrestricted, s = p suffices for arbitrary positive accuracy. After clipping, the target is continuous on a compact set of front and back weights; polynomial approximation gives the decoder. Hence unrestricted analytic readouts also break the scaling.

More generally, if

  R(z,b) = Σ_{i=1}^N a_i(z) ψ_i(b),

then N ≥ r(ε), regardless of the encoder’s nonlinearity. The surviving restriction is finite separation dimension, not continuity or analyticity.


2. Target 2: bounded back-and-forth

2.a. Continuous real protocols

PROVEN: One-way communication of p reals already breaks the scaling.

PROVEN: Two messages, starting at the back, also give the exact protocol

  back → front: all q back weights;
  front → back: the n entries of M(f,b).

Total communication is q+n reals. If the front must speak first, add a constant dummy message. Every message map is continuous and locally Lipschitz.

The clipping construction gives uniformly Lipschitz approximate versions with polylogarithmically growing constants.

Thus bounded interaction cannot restore a lower bound already false for one-way communication.

2.b. Bit protocols

PROVEN: For any number of rounds, any protocol with at most B communicated bits has separation rank at most 2^B.

Fix all random coins. Every deterministic transcript leaf is a rectangle A_t × B_t. The output is

  Σ_t 1_{A_t}(f) 1_{B_t}(b) R_t(b),

with at most 2^B terms. Therefore K1 gives

  B ≥ log₂ r(ε)
    ≥ [2(n−1)/3] log₂(1/ε) + log₂ C_*.

For randomized protocols, conditioning on all coins produces deterministic protocols. The same lower bound applies to each seed’s achievable error and hence to their average.

PROVEN: A stronger bound for the genuine ensemble is

  Err ≥ C_bit 2^(−2B/n),

where C_bit > 0 depends only on the fixed architecture. Consequently,

  B ≥ n log₂(1/ε) + (n/2)log₂(C_bit/Vbar).

Here is a complete construction of the constant.

Use the K5 sign event E:

  W_{c−1}: first row positive, all other rows negative;
  v = W_c e_1: entrywise positive.

Its probability is p_E = 2^(−n²−n). On E,

  h_c(x) = a(x)v,
  M(f,b) = g_b(z),
  z = αv,  α = E_x a(x).

There exists a > 0 such that

  q_0 = P(α ∈ [a,2a] | E) > 0.

Restrict to this event, of probability p_0 = p_E q_0. Conditional on it, z has density at least

  ρ = (2a)^(−n)(n/π)^(n/2) exp(−n²)

on Q = [a,2a]^n.

Independently, restrict every back matrix to the Frobenius ball of radius 1/2 around D = I + 11ᵀ. Let the probability of this back event be p_B > 0. Every such matrix is strictly positive and has smallest singular value at least 1/2. Therefore, on the positive orthant,

  g_b(z) = A_b z,
  σ_min(A_b) ≥ 2^(−d).

For each fixed b and fixed random seed, a B-bit protocol has at most 2^B possible outputs. Apply the quantization lemma in Section 3 with k=n, cube side a, and separation constant 2^(−d)/√n. It gives

  C_bit =
    p_0 p_B ρ [2^(−2d)/n] a^(n+2)
    / [8(2v_n)^(2/n)],

where v_n is the volume of the Euclidean unit ball.

An entirely numerical lower bound on p_B is obtained by raising to the d-th power

  v_{n²} 2^(−n²) (n/(4π))^(n²/2)
  exp{−(n/4)[√(n²+3n)+1/2]²}.

These constants are tiny, but positive.

PROVEN: Two-way bit communication has the upper bound

  B ≤ (q+n) log₂(1/ε) + O_arch(log log(1/ε)).

The back sends quantized clipped back weights; the front evaluates the clipped network mean and returns a quantized n-vector. The same telescoping estimate used in Section 1 controls both errors.

For a compact fixed-amplitude ray family, replace n by its intrinsic dimension n−1 in the quantization lower bound. Its bit complexity is Θ((n−1)log(1/ε)), with architecture-dependent constants.

2.c. Polynomial restrictions throughout the protocol

PROVEN UNDER the following explicit restriction: there are t message turns; every transmitted scalar is polynomial of degree at most D in all earlier transmitted scalars, with arbitrary coefficients depending on the sender’s local input; the final output obeys the same restriction. No transcript-dependent branching is allowed.

Let C be the total number of transmitted scalars and

  H_C = binom(C+D,D),
  S = 1 + D + … + D^t.

Then the output’s separation rank is at most H_C^S.

Proof: a product of at most D functions of separation rank at most K has rank at most K^D. A polynomial has at most H_C monomials. Multiplication by a coefficient depending on only one party’s input does not increase rank. Thus each turn obeys K_new ≤ H_C K^D. Iterating, including the final output, gives the claim.

Consequently,

  C ≥ C_*^(1/(DS)) ε^(−a/(DS)) − D.

For fixed D and t, this retains a precision exponent Θ(n).

Restricting only the final output to be polynomial does not suffice: the back can send b, the front can return M(f,b), and the final decoder is the identity.

2.d. Moment-restricted interaction

PROVEN: If every front response depends on f only through one fixed Z_K, interaction cannot reduce the optimal error below e_K².

Conditioned on the random coins, the entire transcript and output are functions of (Z_K,b). Conversely, the optimal one-way readout already attains the conditional-expectation infimum.

This includes adaptive queries for polynomial moments of degree at most K: each answer is a linear combination of entries of Z_K.

If the front may answer arbitrary back-dependent integrals, the restriction has changed, and the q+n-real construction applies.

2.e. Frame naming and K7

PROVEN interpretation: K7 limits forward-measurable subspace prediction. It does not limit a subspace chosen after a backward message, and it does not identify frame recovery as necessary for estimating M.

Kelvin’s intuition is useful if a protocol must explicitly approximate a back-selected frame or many input-dependent frames. Such a task can require substantial precision. But the present communication model permits the back to send its q weights, which describe all those frames implicitly; the front can then compute with them.

Two missing implications would be needed for a frame-based lower bound:

  small error in M ⇒ accurate recovery of specified frames;
  accurate frame recovery ⇒ the proposed communication cost.

K7 supplies neither implication. The ray identity M = g_b(Eh) is direct evidence against a general necessity of recovering inputwise Jacobian frames.


3. Exact structure and supporting lemmas

3.a. A bi-Lipschitz feature map for rays

PROVEN: For any h,h′ ∈ R^n,

  2^(−d)||h−h′||²
  ≤ E_b ||g_b(h)−g_b(h′)||²
  ≤ ||h−h′||².

For scalars a,b, put p = ReLU(a)−ReLU(b) and
q = ReLU(−a)−ReLU(−b). Then p−q=a−b and

  (a−b)²/2 ≤ p²+q² ≤ (a−b)².

A centered Gaussian row is invariant under sign reversal, so E p² = E q². Since its entry variance is 2/n, summing over n rows gives

  (1/2)||h−h′||²
  ≤ E_W ||ReLU(Wh)−ReLU(Wh′)||²
  ≤ ||h−h′||².

Condition successively on each layer to obtain the result.

Therefore z ↦ g_b(z), viewed as an element of H, is bi-Lipschitz with lower constant 2^(−d/2)/√n. Its covariance can nevertheless have the K4 spectral tail. Finite intrinsic dimension and slow linear approximation are compatible.

3.b. Quantization lower bound

PROVEN: Let z range over a k-dimensional cube Q of side ℓ, with probability density at least ρ there. Suppose

  ||F_z−F_z′|| ≥ a||z−z′||.

For any N reconstruction points in the output normed space,

  E min_i ||F_z−y_i||²
  ≥ ρ a² ℓ^(k+2)
      / [8(2v_k)^(2/k)] · N^(−2/k).

Proof: the preimage of a radius-t ball around y_i has diameter at most 2t/a, hence volume at most v_k(2t/a)^k. Choose

  t = (aℓ/2)(2Nv_k)^(−1/k).

The N balls cover at most half the cube’s volume. On the remaining half, squared error is at least t². Multiplication by ρ gives the bound.

The same statement applies to an event of probability p after multiplying the bound by p.

3.c. Finite moment determinacy for cone-linear Gaussian laws

PROVEN, with a non-effective moment order.

Fix input dimension n, output dimension m, and integers S,H. Consider maps

  Y = A_i X when X ∈ C_i,   X ~ N(0,I_n),

where at most S polyhedral cones partition R^n up to Gaussian-null sets, and each cone has at most H defining homogeneous inequalities. No continuity across cone boundaries is required.

There exists K_0(n,m,S,H) such that equality of all output moments through order K_0 implies equality in distribution.

Proof:

For a face F of a cone, let E_F = span(F), P_F its orthogonal projector, and γ_F the standard Gaussian density on E_F. Define

  I_F(α) = ∫_F x^α γ_F(x) dx,
  ω_F = I_F(0).

For each facet G of F, let ν_FG be the outward unit conormal within E_F. Gaussian integration by parts gives, whenever α_i ≥ 1,

  I_F(α)
  = Σ_j (P_F)_ij (α_j−1_{j=i}) I_F(α−e_i−e_j)
    − (2π)^(−1/2) Σ_{G facet F}
        (ν_FG)_i I_G(α−e_i).

Terms with zero coefficient are omitted. The formula also covers faces with lineality; a full linear subspace simply has no boundary term. The zero-dimensional face has mass one and zero positive-degree moments.

Every term on the right has lower polynomial degree. Consequently every cone moment is a polynomial in the finitely many quantities

  entries of P_F;
  entries of ν_FG;
  Gaussian face masses ω_F.

Expanding (A_i x)^α shows that every output moment is a polynomial in these quantities and the entries of the A_i.

There are only finitely many possible face-incidence types for fixed S,H,n. For any pair of types τ,τ′, write their moment polynomials as P_τα(U) and P_τ′α(V). In the polynomial ring in the finitely many coordinates of U,V, form the ideal generated by

  P_τα(U) − P_τ′α(V),  for all α.

By the Hilbert basis theorem, finitely many of these differences generate the ideal. Let K_ττ′ be the largest degree |α| among a finite generating subset. Equality of moments through that order forces equality of every moment. Take the maximum over the finitely many type pairs.

Finally, every individual Y has Gaussian tails: ||Y|| ≤ max_i||A_i||·||X||. Thus its moment-generating function exists everywhere, and all moments determine its distribution. This proves the theorem.

For a depth-c width-n front, one may take

  S = 2^(nc),   H = nc,   m = n.

Each activation pattern supplies a linear map and a cone with at most nc inequalities.

3.d. Continuity of the finite-moment decoder

PROVEN: In the preceding theorem, take an even Q > max{K_0,2}. On the closure of the realizable moment vectors through order Q, those vectors determine a law continuously in W₁. Therefore every Lipschitz linear-growth observable has a continuous readout.

Here are the compactness details.

Consider a sequence of cone-linear laws with bounded second moments. After taking a subsequence, fix the numbers of cones and inequalities and normalize every nontrivial cone normal to unit length. The normals have convergent subsequences. Cone indicators then converge almost everywhere outside finitely many limiting hyperplanes.

For each limiting cone with positive Gaussian probability,

  B_i = E[XXᵀ 1_{C_i}]

is positive definite. The second-moment bound

  Σ_i tr(A_i B_i A_iᵀ) ≤ constant

therefore bounds A_i on every such piece. Extract convergent A_i there. Pieces whose limiting probability is zero disappear in probability. The limiting law is again in the same cone-linear family.

Now suppose moment vectors through Q converge. Their Q-th norm moments are bounded, because ||Y||^Q is a polynomial in the even coordinate monomials of degree Q. Consequently moments through K_0 are uniformly integrable. Every subsequential limiting law has the prescribed limiting low moments. Finite determinacy makes that law unique. The Q-th moment bound also gives W₁ convergence.

The same argument defines the law at every point in the closure of the moment image, even where a limiting highest moment contains mass lost in a vanishing-probability tail.

Since g_b is Lipschitz and grows at most linearly,

  R(z,b) = ∫ g_b(h) dP_z(h)

is jointly continuous on this closed moment-image set times the back-weight space. The Tietze extension theorem, componentwise, extends it continuously to the entire ambient Euclidean space.

This proves the global continuous final decoder claimed in Section 1.

For layerwise propagation with a common even order Q, continuity on realizable moment vectors follows more strongly in W_Q: convergence of the moment vector gives both weak convergence and convergence of E||Y||^Q. If global extensions are required at every intermediate stage, the descending-order construction ensures uniform integrability of every propagated moment.

3.e. Why topological widths do not recover K4

PROVEN: If a family contains a bi-Lipschitz image of a k-dimensional ball of radius r, any continuous encoding into R^s with s < k has uniform reconstruction error at least ar.

Restrict the encoder to the boundary sphere. Borsuk–Ulam supplies antipodal points with the same encoding. Their targets are at least 2ar apart, so one reconstruction error is at least ar.

This is a uniform-error theorem on a ball. The scalar construction in Section 1 uses averaged error and discards small transition regions.

Likewise, for a Hilbert ellipsoid with semiaxes σ_i, the continuous uniform width equals σ_{s+1}: projection gives the upper bound and Borsuk–Ulam on its first s+1 axes gives the lower bound. That is not the covariance-tail quantity Σ_{i>s}σ_i² from K1.

One must specify the approximated set and the error criterion before importing a nonlinear-width theorem.


4. Quantifiers, depth, and what remains outside

PROVEN quantifiers:

The exact weight-transmission protocol works for every finite weight realization. The finite-moment theorem also covers every realization, including degenerate activation patterns.

The one-scalar continuous approximation is distributional and ε-dependent. It does not establish exact scalar coding by one fixed continuous encoder/decoder.

The bit and bounded-degree lower bounds are ensemble statements for every protocol in the specified class, chosen independently of the sampled network. Randomization independent of the network does not remove them.

The K5-based constants use a rare front event. The stronger bit lower bound additionally uses a rare back event. Neither establishes a high-probability lower bound on typical networks.

An ensemble lower bound does not imply an almost-everywhere fixed-network lower bound. Conversely, the constructed ensemble upper bounds give high-probability guarantees by Markov’s inequality and almost-sure convergence along suitable sequences by Borel–Cantelli.

For a single preselected network, a protocol allowed to depend on that network can hard-code its answer and communicate nothing. A meaningful fixed-network claim must specify which protocol choices precede the network draw and what uniformity over queries remains.

Depth window:

All constructions and the finite-moment theorem hold at every fixed finite n,c,d. Their constants and the required moment order may grow extremely badly with architecture.

The K4 polynomial-degree consequences require the stated fixed finite back depth. The genuine-ensemble K5 consequences require c ≥ 3 and the stated finite-depth assumptions.

The stronger bit lower bound uses c ≥ 3 and d ≥ 1. The first-moment error-floor argument works for c,d ≥ 1.

There is no uniform assertion here when depth grows with 1/ε. For example, the Lipschitz construction uses Λ_ε = O_arch((log(1/ε))^(L/2)); treating L as variable changes that statement substantially.

For n=1, nonnegative cut states lie on one ray and the mean suffices; the angular exponent disappears.

What remains outside:

Unrestricted joint computation, or a decoder receiving all front weights, has no communication obstruction. Its difficulty is the cost of evaluating Gaussian expectations, handling activation regions, or implementing the finite-moment inverse.

A computational lower bound must specify allowed operations, constant precision, preprocessing, approximation criterion, and dependence on architecture. Neither separation rank nor K7 supplies such a bound.

Even an exponential dependence on n would not automatically imply an ε^(−Θ(n)) dependence at fixed n. Those are different asymptotics.


5. Conjecture ledger

No conjecture is used in the results above. In particular, finite moment determinacy and decoder continuity are proved rather than assumed.

5.a. Practical moment order

UNRESOLVED: the smallest useful K_0 and an effective reconstruction algorithm.

Weakest precise conjecture for a proposed order K_*:

  For the relevant cone types, equality of moments through K_*
  implies equality of all moments on their realizable parameter sets.

Conditional consequence: those moments determine the cut law; adding a sufficiently high even moment order supplies the continuity argument above. This gives an exact finite-order closure.

Status: unsure whether useful architecture-specific bounds are known. The Hilbert-basis proof establishes eventual finiteness, not a useful order.

Evidence for: order two suffices at c=1, and finite polynomial descriptions underlie all moments.

Evidence against a small bound: the number of cones and faces grows rapidly, and nearly degenerate cones can make inversion badly conditioned.

Cheapest refutation of a proposed K_*: optimize two small fronts to match moments through K_* while maximizing a held-out back mean difference. A numerical collision is evidence, not an exact proof; a certified collision would refute it.

5.b. A specified ARC/Wick update rule

UNRESOLVED because the statement does not specify that update rule.

A precise sufficient conjecture for a particular family of closures C_K is

  E ||F − C_K(Z_K,·)||_H² ≥ A K^(−2β)

for all K, with A > 0 and β bounded independently of n.

Conditional consequence:

  K ≥ (A/(ε²Vbar))^(1/(2β)),

so counting all moments through K gives a precision exponent approximately n/β.

Status: unsure; there is no defined C_K here to identify with a known open problem.

Evidence for: a prescribed closure can introduce bias despite possessing sufficient information.

Evidence against any information-only version: finite moments eventually determine the law exactly. Thus the conjecture must concern the actual update algorithm, not every continuous readout.

Cheapest refutation: on n=2 or 3, increase K and compare the specified closure against high-accuracy Gaussian integration, while measuring both its bias and numerical conditioning.

5.c. Typical-network lower bounds

UNRESOLVED for the surviving restricted classes.

A sufficient additional hypothesis for a positive-fraction failure theorem is a uniform small-ball statement for their error:

  P(loss ≥ θ E loss) ≥ p_*

with θ,p_* > 0 independent of ε and of the admissible protocol.

Together with the ensemble lower bound, this gives failure on at least a p_* fraction at the corresponding threshold.

Status: unsure. It is not a consequence of K4–K7.

Evidence against deriving it from the present proof: the spectral lower bound may be carried by rare sign configurations. A positive-fraction theorem still would not imply almost-sure eventual failure.

Cheapest refutation: measure the distribution of residual error, separately on the ray event and its complement, as ε decreases.

5.d. SINGLE BOTTLENECK: computational restrictions on the closure/readout

To recover the intended precision exponent beyond bounded-degree or finite-separation models, the unresolved issue is a computational restriction, not continuity.

A conditional computational claim would need an explicit model C and a statement such as:

  Every C-program attaining the ensemble criterion has cost
  at least A ε^(−α(n)).

That hypothesis would directly supply the desired lower bound. I cannot justify it from the statement, and I do not regard it as an established conjecture.

Status: unsure; without a specified model this is not yet a mathematical open problem.

Evidence against assuming the exponent: finite architecture gives finitely many cone integrals, and expensive architecture-dependent preprocessing might permit much better precision dependence.

Cheapest useful experiment: fix a tiny architecture and compare accuracy-versus-cost for direct cone integration, high-order moment inversion, and the actual ARC update. This can refute a proposed cost scaling, though it cannot prove a general computational lower bound.


6. Relevant literature

These references are from my existing knowledge; I did not inspect material beyond the supplied statement.

DeVore, Howard, and Micchelli, “Optimal nonlinear approximation” (1989): continuous nonlinear approximation and topological obstructions; relevant only after specifying the approximated set and whether error is uniform.

Cohen, DeVore, Petrova, and Wojtaszczyk, “Optimal stable nonlinear approximation”: stable manifold widths and their relationship to entropy; appropriate for explicitly bounded encoder/decoder stability.

Pinkus, “n-Widths in Approximation Theory” (1985): linear and Bernstein widths and their comparison; supplies the framework for ball and ellipsoid obstructions.

Carl, “Entropy numbers, s-numbers, and eigenvalue problems” (1981): inequalities connecting entropy and operator approximation quantities; these require the relevant operator/set hypotheses and do not turn K1’s covariance tail into a nonlinear-summary lower bound automatically.

Kolmogorov and Tikhomirov, “ε-entropy and ε-capacity of sets in functional spaces” (1959): covering-number foundations; bit cost is logarithmic in covering size.

Cover and Thomas, “Elements of Information Theory”: rate-distortion and quantization; the natural framework for bounded-precision messages under average squared error.

Kushilevitz and Nisan, “Communication Complexity” (1997): transcript rectangles and rank bounds for finite-bit protocols; directly yields rank ≤ 2^B here.

Yao, “Some complexity questions related to distributive computing” (1979): foundational communication-complexity formulation and randomized protocols.

Linial and Shraibman, “Lower bounds in communication complexity based on factorization norms”: stronger matrix-based communication methods; their finite-input/norm assumptions must be matched to this Gaussian average-error problem.

Traub, Wasilkowski, and Woźniakowski, “Information-Based Complexity” (1988): distinguishes allowed information from computational operations; essential once the weights themselves can be transmitted.

Basu, Pollack, and Roy, “Algorithms in Real Algebraic Geometry”: relevant to effective algebraic descriptions and certification behind the finite-moment argument; it does not by itself provide a useful moment-order bound or an efficient Gaussian cone-integral algorithm.
