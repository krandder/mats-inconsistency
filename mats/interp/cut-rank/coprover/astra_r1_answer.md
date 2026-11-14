codex
1. Reading of (T1)

I take σ = 0 and a_L = h_L. Write d = L′. As stated, (T1) is false: fixed-depth, fixed-accuracy regimes admit polynomial-rank protocols even when k* is proportional to n.

PROVEN. For every finite width n ≥ 3, every front depth, and every back depth, there is a protocol of rank at most

  S(n,m) = Σ_{ℓ=0}^m dim H_ℓ(S^{n−1}) ≤ binom(n+m,m)

whose error is at most

  (n−1)/[(m+1)(m+n−1)] ≤ 1/(m+1).

Here H_ℓ denotes the degree-ℓ spherical harmonics. Consequently,

  r(ε) ≤ binom(n+m,m),   m = ceil[1/(ε² Vbar)].

This is an exact finite-width bound. Its proof uses rotational invariance, homogeneity, and an averaged gradient identity; see Section 3.

For fixed c,d, define

  F(t) = [sqrt(1−t²) + (π−arccos t)t]/π,
  q_t = F composed t times, evaluated at 0.

The fixed-depth infinite-width limit gives

  Vbar → 1−q_{c+d} > 0.

Thus, for every fixed ε > 0 and fixed c,d,

  r(ε) ≤ n^{O_{c,d,ε}(1)}.

Whenever k* ≥ a_d n, as in the stated fixed-depth Fuss–Catalan regime, this contradicts exp(c₀k*).

Interpreting “1 ≪ c,d” as requiring divergent depths does not repair the quantifiers. Choose c=d=t, then choose n_t sufficiently large for the fixed-depth limits and for

  t² log n_t = o(n_t/t).

Since 1−q_{2t} is of order t^−2, the upper bound has log r = O_ε(t² log n_t), whereas the quoted k* scaling gives k* of order n_t/t. This diagonal sequence also contradicts (T1).

The replacement I would use is an accuracy-dependent dimension of a harmonic block that retains energy after front averaging. Precisely, Section 4 defines a block Gram matrix H_B and proves

  r(ε) ≥ [tr H_B − ε² Vbar]/||H_B||op.

Thus log(tr H_B/||H_B||op), subject to tr H_B exceeding the error budget, is a defensible effective dimension. Jacobian survival alone does not supply either requirement.

An exponential theorem is nevertheless real. Below I prove one for nonnegative ray-distributed fronts and one Gaussian He-ReLU back layer, at ε of order k^−3/4. The modified front law is essential; this is not a theorem for the original Gaussian-weight fronts.


2. A complete exponential lower bound in a simplified setting

PROVEN THEOREM. Let the cut dimension and output width both be k. Choose U uniformly on the positive orthant of S^{k−1}. The front indexed by U has distribution

  h = sqrt(k) R U,

where R ≥ 0 is independent, E R = 1, and Var R = 1; an exponential random variable is one choice. Let the back be

  g_W(h) = ReLU(W h),   W_ij independently N(0,2/k).

Define r_+(ε) using exactly the question’s shared-front-feature protocol and averaged output error. There is an absolute C > 0 such that

  ε² ≤ C k^−3/2
  ⇒ r_+(ε) ≥ exp[(log 2 − o(1))k].

In particular, any exponent strictly below log 2 works for sufficiently large k.

All steps below are PROVEN.

2a. Radial reduction and normalization.

Write a selected row of W as sqrt(2/k) G, with G standard Gaussian in R^k. Homogeneity gives

  M(U,G) = sqrt(2) (G·U)_+,
  V(U,G) = M(U,G)².

Therefore Vbar = 1. The other rows and the output index introduce no additional singular values: averaging over the selected row is exactly integration over G.

First allow U to be uniform on the entire sphere. We will recover nonnegative cut states in step 2e.

2b. Exact singular values on the full sphere.

Let σ be uniform probability measure on S^{k−1}, and let Y_{ℓ,a} be an orthonormal harmonic basis. Put

  D_{k,ℓ} = dim H_ℓ
          = (2ℓ+k−2)(ℓ+k−3)!/[ℓ!(k−2)!].

The Funk–Hecke eigenvalue of the kernel (u·v)_+ on degree ℓ is η_ℓ. Its values are

  η_0 = Γ(k/2)/[2 sqrt(π) Γ((k+1)/2)],
  η_1 = 1/(2k),

  η_{2m} = (−1)^{m−1}
           Γ(k/2) Γ(m−1/2)/[4π Γ(m+(k+1)/2)],   m ≥ 1,

  η_ℓ = 0 for odd ℓ ≥ 3.

To verify the formula, let P_ℓ be the normalized Gegenbauer polynomial, P_ℓ(1)=1. The coordinate density is

  w_k(t) = Γ(k/2)(1−t²)^{(k−3)/2}/[sqrt(π)Γ((k−1)/2)].

For f(t)=t_+, the spherical Laplacian satisfies, distributionally,

  (Δ_S + k−1)f = δ_0.

Taking its degree-ℓ coefficient gives

  [(k−1)−ℓ(ℓ+k−2)]η_ℓ = w_k(0)P_ℓ(0).

The standard Gegenbauer value at zero and the gamma duplication identity give the displayed even-degree formula. The odd part of t_+ is t/2, giving η_1 and the vanishing higher odd degrees.

Now write G=ρV, with V uniform on the sphere and Eρ²=k. Funk–Hecke gives

  M(U,G) = sqrt(2)ρ Σ_{ℓ,a} η_ℓ Y_{ℓ,a}(U)Y_{ℓ,a}(V).

The back coefficient functions are orthogonal. Hence the squared singular value in degree ℓ is

  λ_{k,ℓ} = 2k η_ℓ²,

with multiplicity D_{k,ℓ}.

2c. An exponentially large block with polynomial total energy.

Take the even degrees k ≤ ℓ ≤ 2k. Define

  H(α) = (1+α)log(1+α) − α log α.

Stirling’s formula, uniformly for α=ℓ/k in [1,2], yields

  log D_{k,ℓ} = kH(α) + O(log k),
  log λ_{k,ℓ} = −kH(α) + O(log k),
  D_{k,ℓ} λ_{k,ℓ} ≍ k^−5/2.

The last assertion includes the polynomial factors left after the exponential terms cancel.

There are order k even degrees in the block. Its total squared-singular-value mass therefore satisfies

  E_B ≍ k^−3/2.

Since H is increasing and H(1)=log 4,

  max_{ℓ∈B} λ_{k,ℓ}
    ≤ exp[−(log 4)k + O(log k)].

2d. Energy divided by largest eigenvalue gives the lower bound.

A rank-s approximation can capture at most s times the largest squared singular value from this block. Therefore

  Err_s ≥ E_B − s max_{ℓ∈B} λ_{k,ℓ}.

If ε² ≤ E_B/2, this implies

  r_full(ε) ≥ exp[(log 4 − o(1))k].

This is a statement about approximation energy, not merely the existence of infinitely many nonzero coefficients.

2e. Restriction to nonnegative cut states.

All 2^k orthants are equivalent under sign changes of U and corresponding column sign changes of W. The Gaussian back law is invariant under those transformations.

If the positive-orthant problem has a rank-s protocol, use its transformed copy on each orthant, with an orthant indicator in the front features. This constructs a full-sphere protocol of rank at most 2^k s and identical averaged error. Thus

  r_full(ε) ≤ 2^k r_+(ε),

which proves the theorem.

The back Jacobian in this example also has k* of order k. For a fixed nonzero input, approximately k/2 rows are active. On the input’s orthogonal complement, those rows form a rectangular Gaussian matrix; its nonzero singular values stay bounded away from zero, while the full matrix norm stays bounded. At δ=0.01, a linear fraction survive.

The theorem therefore exhibits exponential rank in a genuine Jacobian-survival scale, but at shrinking accuracy and with a specified front distribution.

One can also obtain fixed relative ε by taking Var R of order k^−3/2 instead. The mean and its spectrum stay unchanged, while Vbar shrinks. This illustrates why the variance normalization must be part of any proposed theorem.


3. Exact structure

3a. Rotational covariance and harmonic decomposition — PROVEN.

For the original back ensemble, define

  K_d(h,h′) = E_{b,j} g_{b,j}(h)g_{b,j}(h′).

Right-orthogonal invariance of its first Gaussian weight matrix, together with positive homogeneity, gives

  K_d(h,h′) = ||h|| ||h′|| κ_{n,d}(u·u′)/n,

where u=h/||h||. He scaling gives κ_{n,d}(1)=1.

Consequently,

  κ_{n,d}(t) = Σ_{ℓ≥0} a_ℓ P_ℓ(t),
  a_ℓ ≥ 0,   Σ_ℓ a_ℓ = 1.

Equivalently, for

  f_{b,j}(u)=g_{b,j}(sqrt(n)u)
           =Σ_{ℓ,a} ξ_{ℓ,a}(b,j)Y_{ℓ,a}(u),

the coefficient covariance is

  E ξ_{ℓ,a} ξ_{r,b}
    = λ_ℓ 1_{ℓ=r,a=b},
  λ_ℓ = a_ℓ/D_{n,ℓ}.

Proof: the covariance kernel is rotation invariant, so Funk–Hecke diagonalizes it into scalar operators on harmonic degrees. Its diagonal equals one, giving the trace identity.

This symmetry diagonalizes a second-moment operator. It does not make each individual back function radial.

3b. An exact averaged gradient identity — PROVEN.

For any nonzero h and any perturbation v,

  E_b ||Dg_b(h)v||² = ||v||².

For one layer, Gaussian sign symmetry gives

  E (w·v)² 1_{w·h>0} = ½ E(w·v)².

Summing rows with variance 2/n gives the identity. Conditioning successively on previous layers proves it at every depth. On an interior dead region both the activation and its derivative are zero, so the induction remains valid.

Averaging output coordinates and restricting to the unit sphere therefore gives

  E_{b,j} ∫ ||∇_S f_{b,j}(u)||² dσ(u) = n−1.

Since degree-ℓ harmonics have Laplacian eigenvalue ℓ(ℓ+n−2),

  Σ_ℓ ℓ(ℓ+n−2)a_ℓ = n−1.                         (A)

In particular,

  Σ_{ℓ>m} a_ℓ
    ≤ (n−1)/[(m+1)(m+n−1)].

This proves the spectral estimate used in Section 1.

3c. Front averaging is a separate moment operator — PROVEN.

Define the radial harmonic moments

  m_{ℓ,a}(f)
    = E_{h∼P_f} [||h||/sqrt(n)]Y_{ℓ,a}(h/||h||),

with the integrand defined as zero at h=0. Then

  M(f,b,j) = Σ_{ℓ,a} m_{ℓ,a}(f) ξ_{ℓ,a}(b,j).

Truncating at degree m is a valid rank-S(n,m) protocol. Jensen’s inequality and rotational invariance give

  Err ≤ E_{f,h,b,j}|g_b(h)−g_b^{≤m}(h)|²
      = (Σ_{ℓ>m}a_ℓ) E_{f,h}||h||²/n
      = Σ_{ℓ>m}a_ℓ.

The final equality is exact He second-moment preservation.

Thus the spectrum relevant to the mean is the back spectrum composed with the covariance of front moments. A back-only spectrum cannot determine it.

3d. One-layer parity obstruction — PROVEN.

For one back layer,

  E_h(w·h)_+
    = ½ w·E h + ½ E_h|w·h|.

All odd angular information beyond the first moment is invisible. Equivalently, all odd harmonic degrees ≥3 vanish exactly. Counting gate regions or surviving directions does not overcome this nullspace.

3e. Fixed-depth kernel limit — PROVEN.

For fixed depth, conditional Gaussianity and the law of large numbers give

  κ_{n,d}(t) → F^{∘d}(t).

For two independent Gaussian inputs propagated through the same complete network, their initial normalized overlap tends to zero. Therefore

  E_{f,b,j} M_j² → q_{c+d},
  Vbar → 1−q_{c+d}.

Uniform integrability follows from the fourth-moment calculation below. Expanding F near one gives

  F(1−s) = 1−s + [2sqrt(2)/(3π)]s^{3/2} + O(s^{5/2}),

and hence

  1−q_t ∼ 9π²/(2t²).

3f. Radial variation and finite-width intermittency — PROVEN.

Gaussian input can be written x=RU with R∼χ_n independent of U. Homogeneity implies

  Vbar ≥ 1−(Eχ_n)²/n ∼ 1/(2n).

Indeed, for each fixed network, apply Cauchy–Schwarz to its angular mean, then average its second moment.

There is also an exact norm-moment identity. If Q_ℓ=||h_ℓ||²/n, then conditional on h_{ℓ−1},

  E Q_ℓ = Q_{ℓ−1},
  E Q_ℓ² = (1+5/n)Q_{ℓ−1}².

Each squared coordinate has conditional mean Q_{ℓ−1} and second moment 6Q_{ℓ−1}², which proves the formula by summing independent rows. For Gaussian input,

  E Q_L² = (1+2/n)(1+5/n)^L.

Thus deep finite-width ensembles can have broad norm fluctuations even though their second moments remain critical. Typical angular synchronization alone does not control ensemble-weighted approximation error.


4. A precise crux lemma

Here is a sufficient statement that directly separates back expressivity from survival through front averaging.

PROVEN BLOCK LEMMA. Let B be any finite collection of harmonic coordinates with λ_i>0. Define

  C_B = E_f [m_B(f)m_B(f)^T],
  H_B = Λ_B^{1/2} C_B Λ_B^{1/2},

where Λ_B=diag(λ_i). Every rank-s protocol satisfies

  Err ≥ tr H_B − s||H_B||op.                     (B)

Proof. The back functions e_i=ξ_i/sqrt(λ_i) are orthonormal, and

  T e_i = sqrt(λ_i)m_i.

The restriction of T to their span has Gram matrix H_B. Restricting an arbitrary rank-s approximant still gives rank at most s. Eckart–Young on this restriction gives the sum of eigenvalues beyond s, which is at least the right side of (B).

For example, if B contains D coordinates and

  H_B ≽ (γ Vbar/D) I_D,

then

  r(ε) ≥ D(1−ε²/γ).

A weaker sufficient condition is

  tr H_B ≥ 2ε² Vbar.

Under that condition, define

  k_res(B) = log[tr H_B/||H_B||op].

Then

  r(ε) ≥ ½ exp[k_res(B)].

This is the formalization I would try to establish. It requires energy above the requested tolerance and sufficiently dispersed eigenvalues.

The proposed crux does not follow from k*. For example, take front laws that differ only by a positive scale:

  P_f = law(s_f H_0).

For every homogeneous back,

  M(f,b,j) = s_f E g_{b,j}(H_0),

so the cut rank is one, regardless of the back’s Jacobian spectrum. These are modified fronts, but they disprove any general implication from survival alone. Section 1 separately disproves the stated unrestricted-depth-window claim for the original fronts.

There is also no valid reduction from two-way protocols to the stated one-way class merely because directions were “backward-selected.” A projection depending on b cannot be computed from f alone. With unrestricted computation, the back can send its weights and the front can return the desired integral; that is a polynomial-size two-way description. Proving a one-way bound does not exclude it.


5. (T2): synchronization and collapse

Angular synchronization of a front does not by itself imply low cut rank. The fronts in Section 2 are supported exactly on rays, yet their cut operator has exponential rank at the stated precision. The ray varies across fronts.

Positive homogeneity also prevents literal merging of all input amplitudes: h_L(tx)=t h_L(x). The meaningful synchronization is angular.

Likewise, a back whose image lies on one ray may retain a complicated scalar amplitude function. For example,

  g_b(h)=v_b(w_b·h)_+

has Jacobian rank at most one but retains the nontrivial ReLU ridge spectrum. This is a counterexample to the implication “one output direction implies low cut rank,” not a claim about the original iid multilayer ensemble.

There is, however, a useful positive result for the original ensemble.

PROVEN, in an iterated infinite-width limit. For fixed c≥1 and d, use the rank-one protocol

  φ(f)=1,
  ψ(b,j)=E_f M_j(f,b).

Then, as n→∞,

  Err_1 → q_{c+d}−q_{d+1},
  Vbar → 1−q_{c+d}.                              (C)

Proof. The first second moment is q_{c+d}. For ψ², take two independent fronts and pass them through the same back. Their cut vectors have normalized overlap tending to 1/π=q_1: conditional on earlier layers, the two last front layers have independent Gaussian rows and positive ReLU means. Passing through d common back layers gives q_{d+1}. Subtract the second moments.

For fixed c and then d→∞,

  Err_1/Vbar ∼ 2(c−1)/d.

Thus, for each fixed ε and c, sufficiently large d followed by sufficiently large n gives r(ε)≤1. This collapse follows from an explicit protocol.

It does not establish the finite-width assertion at c or d ≫ n log(1/tol). That regime requires control of normalized spectral tails and the norm fluctuations in Section 3f. Front synchronization alone is insufficient; a sufficiently long back does yield collapse in the specified iterated limit.


6. (T3): ensemble bounds versus typical fixed networks

An ensemble lower bound gives only

  E Z ≥ lower bound,

where Z(f,b) is the protocol’s per-network squared error. It does not imply that Z exceeds ε²V on most networks. A small exceptional set can carry the expectation.

Even a bound

  E Z² ≤ C(E Z)²

only yields, by Paley–Zygmund,

  P[Z ≥ θ E Z] ≥ (1−θ)²/C.

That is a positive fraction, not probability tending to one.

A sufficient additional theorem, for a protocol chosen before the weights, would be

  E Z ≥ (1+η)ε² Vbar,
  Z/E Z → 1 in probability,
  V/Vbar → 1 in probability.

It would imply P[Z>ε²V]→1. Alternatively, one could prove a direct lower-tail bound for Z/V.

Such concentration is not automatic for arbitrary measurable protocols; their readouts can have arbitrarily heavy tails. Uniform statements require additional structure or a direct argument that handles the entire class.

The quantifiers also matter:

  For every predetermined protocol, most networks defeat it

is weaker than

  On most networks, every protocol fails.

The latter is impossible without restricting how the protocol is selected: after seeing one network, a rank-one constant answer can be chosen to equal that network’s mean exactly.

If “almost always” means probability one at finite n, dead-network events provide another obstruction. A useful target is probability 1−o(1), with its asymptotic regime stated explicitly.


7. Conjecture ledger

7a. Simplified exponential theorem.

No conjecture is needed. Sections 2a–2e are proved. The change of front ensemble is an assumption, not a claimed approximation to the original front law.

A cheap implementation check is to evaluate the gamma formulas and their multiplicities. For the shell k≤ℓ≤2k, its mass times k^{3/2} should remain bounded away from zero. Numerical SVD is unnecessary for this theorem.

7b. Original Gaussian-network fronts: SINGLE BOTTLENECK.

The missing step is spectral spread after front averaging.

A precise sufficient conjecture is that, in some specified regime and for an explicitly chosen harmonic block B,

  tr H_B ≥ A Vbar,
  ||H_B||op ≤ A Vbar exp(−a k*),

for constants A,a>0. Then ε²≤A/2 gives r(ε)≥½exp(a k*).

This is weaker than demanding that every direction in a whole harmonic degree be well conditioned. It asks only for retained energy and an upper bound on the largest eigenvalue.

A regime I would test, rather than assert, is

  c=d=floor(n^{1/3}),
  B = degrees between αd² and βd²,

with fixed 0<α<β. It avoids the fixed-depth obstruction. The scale d² is suggested by the critical correlation recursion and its derivative remaining one while its nonconstant energy decreases.

Status: CONJECTURE. I do not know a theorem proving this front-moment estimate, or a standard named open problem equivalent to it; unsure.

Cheapest refutation attempt: at small increasing widths, build independent-front/independent-back mean matrices and test low-rank predictors on held-out pairs. Use independent input-integration replicates to estimate and remove Monte Carlo noise; otherwise that noise manufactures a spectral tail. Direct harmonic-moment estimates can distinguish loss of block energy from concentration into a few modes.

7c. Finite-width synchronization.

For a proposed rank-one approximation S(f,b,j), the needed certificate is

  E(M−S)² = o(Vbar)

along a stated post-synchronization sequence. An angular-distance estimate without this normalization does not close the argument.

Status: CONJECTURE for the original finite-width post-synchronization regime. Relevant exact limits are proved above, but their uniform transfer to depths much larger than n is missing. Unsure whether an existing result supplies it.

Cheapest refutation attempt: measure normalized held-out rank-one error as d/n increases, alongside norm fourth moments and the contribution of unusually large networks.

7d. Typical fixed networks.

For a specified protocol family with a mean-error gap, relative variance bounds

  Var Z = o((E Z)²),   Var V = o(Vbar²)

would suffice. These are not plausible as uniform assumptions over all unrestricted measurable protocols. For that class, the needed replacement is a direct small-ball bound for Z/V.

Status: UNPROVEN; no confident reference establishes the required statement here.

Cheapest refutation attempt: pre-register a protocol, sample fresh networks, and inspect the distribution of Z/V and the fraction of the ensemble mean contributed by the largest errors.


8. Relevant literature

Eckart and Young, “The Approximation of One Matrix by Another of Lower Rank” (1936), together with the Hilbert-space Schmidt decomposition: the optimal average rank-s error is the squared-singular-value tail.

Allan Pinkus, n-Widths in Approximation Theory (1985): the linear-width framework; distinguish its worst-case widths from this problem’s average squared-error width.

I. J. Schoenberg, “Positive Definite Functions on Spheres” (1942): nonnegative spherical harmonic expansions of rotationally invariant positive-definite kernels.

Feng Dai and Yuan Xu, Approximation Theory and Harmonic Analysis on Spheres and Balls (2013): Funk–Hecke, Gegenbauer identities, harmonic multiplicities, and spherical Sobolev estimates.

Youngmin Cho and Lawrence Saul, “Kernel Methods for Deep Learning” (2009): arc-cosine kernels, including the exact one-layer ReLU covariance used here.

Jaehoon Lee et al., “Deep Neural Networks as Gaussian Processes” (2018), and Alexander Matthews et al., “Gaussian Process Behaviour in Wide Deep Neural Networks” (2018): fixed-depth wide-network kernel limits; they do not automatically justify growing-depth limits.

Francis Bach, “Breaking the Curse of Dimensionality with Convex Neural Networks” (2017): approximation spaces generated by ridge functions and relevant spherical spectral calculations.

Alberto Bietti and Francis Bach, “Deep Equals Shallow for ReLU Networks in Kernel Regimes” (2021): depth-dependent ReLU kernel spectra and endpoint regularity; relevant to the back spectrum, not by itself to front-averaged cut rank.

Francis Bach, “On the Equivalence between Kernel Quadrature Rules and Random Feature Expansions” (2017): spectral connections between integration, features, and approximation.

Traub, Wasilkowski, and Woźniakowski, Information-Based Complexity (1988), and Novak and Woźniakowski, Tractability of Multivariate Problems: integration complexity depends on the information and decoder model. These distinctions prevent turning a linear-summary lower bound into unrestricted sampling optimality.

Kunisky, Wein, and Bandeira, “Notes on Computational Hardness of Hypothesis Testing: Predictions Using the Low-Degree Likelihood Ratio” (2019): low-degree restrictions as an algorithmic model. Such results do not directly bound this operator’s unrestricted linear-summary rank.

David Gamarnik, “The Overlap Gap Property: A Topological Barrier to Optimizing over Random Structures” (2021): a source of algorithmic obstructions for specified algorithm classes. I see no direct overlap-gap implication for the present spectral question.
