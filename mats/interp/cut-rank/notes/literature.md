Model: Sonnet 5.5 (claude-sonnet-5-5)

# Literature map for the cut-rank question (critical bias-free ReLU nets, front/back cut)

Compiled 2026-10-30, 23:24-23:53 UTC, read-only research worker (single output file; scratch downloads in /tmp, removed at the end). Every URL listed was opened in that window
(arXiv abs pages by WebFetch ~23:25-23:45Z; PDFs fetched with curl and text-extracted with pypdf ~23:26-23:50Z;
ARC pages ~23:35-23:42Z). Nothing was cited from memory without an open attempt; failures are tagged.

Verification tags
- [FT]  full text read (PDF text extracted); theorem/section numbers are as printed in that text.
- [ABS] only abstract / landing page opened (WebFetch paraphrases abstracts; it does not reproduce them verbatim).
- [2ND] statement checked only as quoted by another source I opened (named in the entry).
- [UNV] UNVERIFIED: page blocked or not opened; metadata from memory or a search-tool snippet.

Memory-ID traps found while checking (do not reuse from memory): Novak survey is arXiv:1409.6714 (not 1409.6740);
Hinrichs-Novak-Ullrich-Wozniakowski is arXiv:1211.0871; Chattopadhyay-Mande-Sherif is ECCC TR18-176 (not TR18-17);
Schramm-Wein is arXiv:2008.02269; Alon's DOI is 10.1017/S0963548307008917.

======================================================================
## 0. Direct hits (ranked by usefulness for the cut-rank claim)

1. Bietti & Bach 2021, Thm 1 + Cor. 2 (arXiv:2009.14397): endpoint singularity (1-t)^nu at t=+-1 => Funk-Hecke eigenvalue
   mu_k ~ C k^{-d-2nu+1}; ReLU kappa_1 has nu=3/2 => k^{-(d+2)} on S^{d-1} (d = ambient dim = your n); deep (>=2 hidden ReLU
   layers) NNGP kernel keeps k^{-(d+2)}, constant grows linearly with depth. [FT]
2. Bach 2017 App. C.1/D.2 and Bietti & Mairal 2019 Lemma 17 / Prop. 5: for the single-hidden-layer ReLU kernel odd degrees
   k>=3 vanish exactly, even degrees ~ k^{-(d+2)}. [FT]
3. Hilton (ARC) note "Deductive estimation for random halfspace intersections", Thm 3.1 + Sec. 4 + Sec. 6: the same head/tail
   operator, eigenfunctions = spherical harmonics (Funk-Hecke), harmonic-count formula, algorithm = project onto degree<=D. [FT]
4. Wu, Lecomte, Winer, Robinson, Hilton, Christiano 2026 (arXiv:2605.05179): exactly your ensemble (bias-free MLP, N(0,2/n)
   weights, N(0,I) input, quenched mean); Thm 5.1/5.2 (polynomial activations); Sec. 7.3 lists "narrow networks, eps->0 at fixed
   n" as open. [FT]
5. Lee & Shraibman 2009 Thm 4.3 (+ proof) and Chattopadhyay-Mande-Sherif 2018/2020: a c-bit private-coin protocol with any number of
   rounds yields an acceptance-probability matrix of rank <= 2^c, so a rank-s lower bound forces >= log2(s) bits; approximate rank is
   only a lower bound (its log-conjecture as a characterization of randomized communication is refuted). [FT]
6. Cohen-DeVore-Petrova-Wojtaszczyk 2022 Thm 3.3/4.1/Cor. 4.2 (in Hilbert space, stable nonlinear widths have the same polynomial rate as
   entropy numbers) + Siegel 2024 (arXiv:2402.04407; Bernstein <= manifold <= Kolmogorov width, and Bernstein and manifold widths of L2
   Sobolev balls are both ~ n^{-s/d}): a Lipschitz/continuity requirement on nonlinear features gives no rate gain over linear
   projection in L2. [FT]
7. Barron 1993 Thm 6 and Siegel-Xu 2022 eq. (1.33)-(1.34): linear n-term spaces cannot beat n^{-1/d} on Barron's class; sharp
   n-width / entropy / Bernstein-width lower bounds for ReLU^k ridge classes. [FT]
8. Novak 2016 survey, Thm 1 and Thm 3 (+ Remark 3): Bakhvalov rates n^{-k/d} vs n^{-k/d-1/2}; the randomized rate is obtained by
   "approximate (deterministic) + Monte Carlo on the residual". [FT]
9. Bach 2017b (JMLR 18(21), arXiv:1502.06800) Prop. 3: the number of (stably bounded) linear features needed to approximate the unit
   RKHS ball is >= #{m : mu_m >= 144 lambda} / (4 log(...)), with a matching upper bound up to logs: eigenvalue counting, as in your rank claim. [FT]

======================================================================
## 1. Approximate rank and communication

Scope warning. The whole communication literature below is about entrywise (sup-norm) approximation of sign/Boolean matrices
at constant error. Your criterion is mean-square under the product weight ensemble (Hilbert-Schmidt). In HS norm the best
rank-s approximation is exactly the truncated SVD (Schmidt for integral operators; Eckart-Young; Mirsky):
||M - M_s||_F^2 = sigma_{s+1}^2 + sigma_{s+2}^2 + ... [ABS: https://en.wikipedia.org/wiki/Low-rank_approximation , opened 23:4xZ,
"Eckart-Young-Mirsky" section; standard fact, primary sources not opened]. So <phi(front), psi(back)> with arbitrary (even
nonlinear) phi, psi into R^s cannot beat the truncated SVD of the cut operator; the cited work supplies interpretation and
warnings, not the bound itself.

* Lee, Troy; Shraibman, Adi (2009). "Lower bounds in communication complexity." Foundations and Trends in Theoretical Computer
  Science 3(4) (2009) (venue from search metadata; page range not confirmed). URL opened: https://www.cs.toronto.edu/~toni/Courses/CommComplexity/Papers/adi-book.pdf [FT]
  - Def. 4.1 (approximate rank rank^alpha(A) = min rank of B with 1 <= A[i,j]B[i,j] <= alpha). Thm 4.3 (Krause 1996):
    R^pri_eps(A) >= log2 rank^alpha(A), alpha = 1/(1-2 eps). Proof: the acceptance-probability matrix is P[x,y] = sum_l U_l(x) V_l(y)
    over the <= 2^c leaves of the protocol tree, so rank P <= 2^c. The leaf decomposition is for an arbitrary protocol tree, so it
    holds for any number of rounds. Cor. 4.1: public-coin version via Newman, minus O(log n - log delta). Conj. 4.1 (log-approximate-rank
    conjecture). Thm 4.5 (Lee-Shraibman 2008): rank^alpha(A) = O(alpha^6/(alpha-1)^6 ln^3(4mn) gamma_2^alpha(A)^6). Thm 4.9 (discrepancy
    method): R_eps(A) >= log2((1-2 eps)/disc(A)), also round-agnostic.
  - For us: a front summary phi in R^s with inner-product readout is literally a rank-s factorization; Thm 4.3's proof is the
    protocol-to-factorization map. Rank bound = lower bound on every linear one-way (and every any-round bit) summary of the same dimension.

* Alon, Noga (2009). "Perturbed identity matrices have high rank: proof and applications." Combinatorics, Probability and Computing
  18(1-2):3-15, DOI 10.1017/S0963548307008917. URL opened: https://web.math.princeton.edu/~nalon/PDFS/identity1.pdf [FT]
  - Thm 1.1: B n x n real, |b_ii| >= 1/2, |b_ij| <= eps (i != j), 1/(2 sqrt n) <= eps < 1/4 => rank B >= c log n / (eps^2 log(1/eps)).
    Lemma 2.2 (the cheap one): symmetric A with unit diagonal and off-diagonal <= eps has rank >= n/(1+(n-1)eps^2) (Cauchy-Schwarz on
    eigenvalues: rank >= (tr A)^2/||A||_F^2). That trace/Frobenius argument is the entrywise analogue of a mean-square rank bound.
  - Lee-Shraibman (above) report rank_2(I_{2^n}) = Omega(n), hence R^pri_{1/4}(EQ) >= log n - O(1) [2ND].

* Chattopadhyay, Arkadev; Mande, Nikhil S.; Sherif, Suhail (2018; STOC 2019; J. ACM 67(4), Art. 23, 2020 [venue from search metadata, UNV]).
  "The log-approximate-rank conjecture is false." ECCC TR18-176. URLs opened: https://eccc.weizmann.ac.il/report/2018/176/ and /download/ [FT]
  - F = SINK o XOR on 2n variables has O(sqrt n) spectral norm, O(n^2) approximate rank, n^{O(log n)} approximate nonnegative rank, yet
    randomized bounded-error communication Omega(sqrt n): first exponential gap between log approx-rank and R. So approximate rank is a
    lower bound, not a characterization, for Boolean total functions at constant error. (Quantum follow-up Anshu-Boddu-Touchette,
    arXiv:1811.10525, seen only in a search result: UNV.)

* Gal, Anna; Syed, Ridwan (2019). "Upper bounds on communication in terms of approximate rank." ECCC TR19-006. URL opened: https://eccc.weizmann.ac.il/report/2019/006/ [ABS]
  - Approximate rank r => quantum protocols (no entanglement) with O(sqrt r log r) bits; deterministic O(r) if discrepancy delta is also
    controlled; private-coin randomized O((1/delta)^2 + log r). Converse direction: rank is a polynomially-related upper bound only with extra assumptions.

* One-way with real-valued vectors / product distributions.
  - Kremer, Ilan; Nisan, Noam; Ron, Dana (1999). "On randomized one-round communication complexity." Computational Complexity 8(1):21-49,
    DOI 10.1007/s000370050018. Original page NOT opened (Springer redirect; Semantic Scholar 403): [UNV]. Statement verified [2ND]
    through Sherstov, "Communication complexity under product and nonproduct distributions", Thm 3.1 there = KNR Thm 3.2:
    for sign matrices and product distributions, one-way distributional cost is O((1/eps) vc(A) log(1/eps)).
    Source opened: http://web.cs.ucla.edu/~sherstov/pdf/cc-prod-nonprod.pdf [FT].
    Search-tool summary of the KNR abstract says the paper also contains "a study of the problem of computing the inner product of two
    real valued vectors" in one round: this is the closest published match to "real-valued messages with linear decoder" but I could
    not read it. Worth fetching. Caution for you: your front and back are independent (a product measure); in the Boolean case one-way
    cost under product measures is VC-governed, not rank-governed. Your claim is only about linear (inner-product) summaries.
  - Chakrabarti, Amit; Regev, Oded (2010/2012). "An optimal lower bound on the communication complexity of Gap-Hamming-Distance."
    arXiv:1009.3460 (SIAM J. Comput. 2012 from memory: UNV). URL opened: https://arxiv.org/abs/1009.3460 [ABS]
    Omega(n) randomized lower bound at gap sqrt n, any number of rounds (earlier results covered deterministic or round-limited protocols).
    Translates to Omega(1/eps^2) bits to estimate an inner product to relative error eps: the 1/eps^2 sampling-like cost is
    communication-optimal for generic vectors (my reading, not a statement in the abstract).

* Bounded-round two-way protocols, real-valued messages. No reference found that proves rank/discrepancy-type lower bounds for
  real-valued multi-round messages. What I could verify: (i) LS Thm 4.3 and Thm 4.9 proofs do not use the number of rounds (bit
  messages); (ii) unrestricted real messages are vacuous (a space-filling curve encodes any compact set of a separable Banach space
  into one real number), so any real-message lower bound needs continuity / Lipschitz stability / finite precision: that is exactly the
  setting of Section 2 (Siegel 2024, intro, quoting DeVore-Howard-Micchelli 1989) [FT].

======================================================================
## 2. Widths (linear, Kolmogorov, Bernstein, manifold, stable manifold, entropy)

* Barron, Andrew R. (1993). "Universal approximation bounds for superpositions of a sigmoidal function." IEEE Trans. Inform. Theory 39(3):930-945.
  URL opened: http://www.stat.yale.edu/~arb4/publications_files/UniversalApproximationBoundsForSuperpositionsOfASigmoidalFunction.pdf
  (HTTPS certificate on that host had expired; fetched with curl -k, header/pagination match IEEE TIT vol. 39 no. 3 May 1993) [FT]
  - Sec. X "Lower bounds for approximation by linear subspaces", Thm 6: for every choice of n fixed basis functions the worst-case
    integrated squared error over Gamma_C (L2 on [0,1]^d, uniform) is >= kappa * (prefactor in C, d) * (1/n)^{2/d}, kappa a universal
    constant >= 1/(8 pi e^{pi-1}); equivalently the Kolmogorov n-width of Gamma_C is >~ n^{-1/d} (Eq. 60). Proof via Lemma 6 (an n-dim
    subspace cannot be within squared distance 1/2 of every vector of an orthonormal basis of a 2n-dim space). The exact prefactor in
    (C, d) is garbled in my text extraction; the exponents 2/d (squared) and 1/d are verified in the text and in the discussion ("no
    choice of n fixed basis functions ... smaller order than (1/n)^{2/d} uniformly for functions in Gamma_C").

* Siegel, Jonathan W.; Xu, Jinchao (2021/2022). "Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural
  networks." arXiv:2101.12365 (v10, revised 22 Feb 2024 per the arXiv listing; the PDF header date is just its compile date; published version DOI
  10.1007/s10208-022-09595-3 per the arXiv page, journal name Found. Comput. Math. inferred from the DOI prefix: UNV). URL opened: https://arxiv.org/pdf/2101.12365 [FT]
  - Eq. (1.31): d_n(B_1(P_k^d))_{L_p} <~ n^{-(pk+1)/(pd)}. Eq. (1.33): in L2, d_n >~ n^{-(2k+1)/(2d)}, eps_n >~ n^{-1/2-(2k+1)/(2d)},
    Bernstein b_n >~ n^{-1/2-(2k+1)/(2d)}; Eq. (1.34): eps_n(B_1(P_k^d))_{L2} ~ n^{-1/2-(2k+1)/(2d)} (sharp). Text after (1.33): linear
    methods are dramatically worse than shallow nets, and nonlinear methods cannot improve if stability [CDPW] or continuity [DHM89]
    is required. Sec. 1.1 defines entropy numbers, Kolmogorov, Gelfand, linear, Bernstein widths; "in a Hilbert space the linear
    n-widths and Kolmogorov n-widths coincide" (orthogonal projection).

* DeVore, Ronald A.; Howard, Ralph; Micchelli, Charles (1989). "Optimal nonlinear approximation." Manuscripta Math. 63(4):469-478.
  URL opened: http://eudml.org/doc/155392 (record only: title, authors, pages, keywords "n-widths; Sobolev spaces", MSC 41A46; no abstract) [ABS].
  Publisher page not opened (Springer redirect). Abstract text as returned by a search tool (UNV): defines nonlinear n-widths, determines them for the
  unit ball of W^r_p in L_q, spline manifolds with n free knots optimal. Results used elsewhere, verified [2ND]:
  - Thm 3.1: Bernstein width <= manifold width, b_n(K)_X <= delta_n(K)_X, via Borsuk-Ulam (as quoted in Siegel 2024, eq. (1.6), below).
  - Thm 4.2 (as stated in Yarotsky 2017, Thm 3): if eta: R^W -> C([0,1]^d) and a continuous M from the unit ball of W^{n,infty}([0,1]^d)
    to R^W satisfy ||f - eta(M(f))||_inf <= eps for all f, then W >= c eps^{-d/n} (n = smoothness there). Bach 2017 also cites it: "the order
    O(n^{-1/d}) cannot be improved (DeVore et al. 1989, Thm 4.2)" for ReLU approximation of Lipschitz functions with n neurons [FT].
  - Yarotsky, Dmitry (2017). "Error bounds for approximations with deep ReLU networks." arXiv:1610.01145 (Neural Networks 94, 2017, from memory: UNV).
    URL opened: https://arxiv.org/pdf/1610.01145 [FT] (Sec. 4.1, Thm 3).

* Cohen, Albert; DeVore, Ronald; Petrova, Guergana; Wojtaszczyk, Przemyslaw (2020/2022). "Optimal stable nonlinear approximation."
  arXiv:2009.09907; Found. Comput. Math. 22 (2022) 607-648 (volume/pages from search metadata; Springer page not opened: UNV).
  URL opened: https://arxiv.org/pdf/2009.09907 [FT]
  - Eq. (1.5): manifold width delta_n(K)_X (continuous encoder a: K->R^n, decoder M: R^n->X). Eq. (1.6)-(1.7): stable manifold width
    delta*_{n,gamma}(K)_X = inf over gamma-Lipschitz a, M (and norms on R^n). Thm 3.3 (Carl-type): eps_n(K)_X <= C (n+1)^{-r} sup_m (m+1)^r
    delta*_{m,gamma}(K)_X, so entropy numbers lower-bound stable widths. Thm 3.6: weaker-continuity version with a log loss (eps_{cn ln n}).
    Thm 4.1: K compact in a Hilbert space H, gamma = 2: delta*_{26 n}(K)_H <= 3 eps_n(K)_H; Cor. 4.2: delta*_{n,gamma}(K)_H = O(n^{-r}) iff
    eps_n(K)_H = O(n^{-r}). No Bernstein widths appear in this paper (they are in DHM89 and Siegel 2024).

* Siegel, Jonathan W. (2024). "Sharp lower bounds on the manifold widths of Sobolev and Besov spaces." arXiv:2402.04407 (v4).
  URLs opened: https://arxiv.org/abs/2402.04407 and https://arxiv.org/pdf/2402.04407 [FT]
  - Intro: Bernstein widths bound Gelfand, linear, Kolmogorov and manifold widths from below (Borsuk-Ulam). Eq. (1.3): Bernstein widths of
    Sobolev/Besov unit balls in L_p([0,1]^d) ~ n^{-s/d} for p >= q or 1 <= p <= q <= 2 (other ranges have extra factors). Eq. (1.4):
    manifold widths <= C n^{-s/d}; (1.5) lower bound by Bernstein widths in the easy range; new "sphere embedding widths" give
    n^{-s/d} for all 1 <= p,q <= infinity (Thm 2). Remarks: manifold (merely continuous) widths do NOT satisfy a Carl inequality, so
    entropy cannot lower-bound them (stable ones can, CDPW above); Bernstein widths can decay strictly faster than manifold widths.
    Cites Lorentz-von Golitschek-Makovoz (1996, Ch. 13-14), Pinkus (n-Widths in Approximation Theory, Ch. 7) and Tikhomirov for
    Kolmogorov/Gelfand/linear widths of Sobolev balls: those books were not opened [UNV].

* Spheres. Geller, Daryl; Pesenson, Isaac (2014). "Kolmogorov and linear widths of balls in Sobolev and Besov norms on compact manifolds."
  Math. Scand. 115(1):96-122; arXiv:1104.0632. URL opened: https://arxiv.org/abs/1104.0632 [ABS]. Upper asymptotic estimates for Kolmogorov and linear
  widths on compact Riemannian manifolds; "asymptotically exact" for compact homogeneous manifolds (spheres included) in the natural
  index ranges. The abstract gives no explicit rate. A J. Complexity 18 (2002) paper "Kolmogorov width of classes of smooth functions on the sphere
  S^{d-1}" (authors guessed by a search tool as Bordin/Kushpel/Levesley/Tozoni; reported d_N ~ N^{-r/(d-1)} for 2 <= p < q) could not be opened (403): UNV.
  (Derivation, not from a source: for an L2 Sobolev ball on S^{d-1} the widths are the singular values of the embedding, Laplace-Beltrami eigenvalue
  (k(k+d-2))^{-s/2} with multiplicity ~ k^{d-2}, giving d_n ~ n^{-s/(d-1)}.)

* DeVore, Ronald; Hanin, Boris; Petrova, Guergana (2021). "Neural network approximation." Acta Numerica 30 (venue from memory: UNV); arXiv:2012.14501.
  URL opened: https://arxiv.org/abs/2012.14501 [ABS]. ReLU nets form a parametrized nonlinear manifold with space-filling properties: better
  rate-distortion at the expense of numerical stability. Survey framing for the stability/continuity requirement.

Hilbert-space reading (mine, for your setting): the image of a unit ball under a compact operator is an ellipsoid; its Kolmogorov and Bernstein widths both
equal the (n+1)-th semi-axis, so by b_n <= delta_n <= d_n (Siegel 2024 intro) continuous nonlinear encodings give no worst-case gain over projection.
Your mean-square (HS) criterion is the average-case analogue, where the optimizer is again the SVD (see Section 1).

======================================================================
## 3. Dot-product kernels on spheres, Funk-Hecke, ReLU eigenvalues

Setup fact (Funk-Hecke, as stated in the papers below): for kappa on [-1,1] and x,y in S^{d-1} (tau = uniform probability),
the operator f -> int kappa(<x,y>) f(y) dtau(y) multiplies every degree-k spherical harmonic by
mu_k = (omega_{d-2}/omega_{d-1}) int_{-1}^{1} kappa(t) P_k(t) (1-t^2)^{(d-3)/2} dt, with P_k the degree-k Legendre/Gegenbauer polynomial in d dims
(P_k(1)=1) and multiplicity N(d,k) = ((2k+d-2)/k) binom(k+d-3, d-2) ~ 2 k^{d-2}/(d-2)!  [Bietti-Bach 2021, Sec. 2.2 Eq. (8), FT].
Bach 2017 uses the sphere S^d in R^{d+1} (his d = ambient dimension - 1) and calls it the Hecke-Funk formula (App. D.1, FT).
Textbook sources (not opened [UNV]): Dai-Xu, "Approximation Theory and Harmonic Analysis on Spheres and Balls" (Springer 2013), Thm 1.2.9 (cited as the
Funk-Hecke theorem in Hilton's note below, and Cor. 1.1.5 for harmonic dimension counts); Mueller (1966); Atkinson-Han (2012).

* Schoenberg, I. J. (1942). "Positive definite functions on spheres." Duke Math. J. 9(1):96-108, DOI 10.1215/S0012-7094-42-00908-6.
  Primary page NOT opened (Project Euclid 403/empty) [UNV]. Statements verified [2ND]: Daniely-Frostig-Singer Thm 8 (opened, see Sec. 4):
  "a continuous mu on [-1,1] is PSD iff for all d = 1,2,...,infinity the function kappa(x,x') = mu(<x,x'>) on S^{d-1} is a kernel" (power series with
  nonnegative coefficients). Fixed-d version (kappa(<x,y>) positive definite on S^{d-1} iff its Gegenbauer coefficients mu_k >= 0) is the
  content of the Funk-Hecke/Mercer decomposition in Bietti-Bach Eq. (8)-(9) [FT].

* Cho, Youngmin; Saul, Lawrence K. (2009). "Kernel methods for deep learning." Advances in Neural Information Processing Systems 22 (NIPS 2009), pp. 342-350.
  URL opened: http://papers.neurips.cc/paper/3628-kernel-methods-for-deep-learning.pdf [FT]
  - Eq. (3)-(7): k_n(x,y) = (1/pi)||x||^n ||y||^n J_n(theta); J_0 = pi - theta; J_1 = sin theta + (pi - theta) cos theta; k_1(x,x) = ||x||^2. This is exactly
    your |h||h'| kappa(cos angle) form for the first ReLU layer (infinite width).

* Bach, Francis (2017). "Breaking the curse of dimensionality with convex neural networks." JMLR 18(19):1-53 (venue confirmed at
  https://jmlr.org/papers/v18/14-546.html). Full text: https://arxiv.org/pdf/1412.8690 (arXiv:1412.8690) [FT]
  - App. C.1 (d=1, i.e. S^1): for sigma = (.)_+^alpha, alpha=1: lambda_k = -cos(k pi/2)/(pi (k^2-1)) (k != 1), lambda_1 = 1/4; zero for every other odd k, nonzero for
    all even k, decay 1/k^2. App. D.2 (S^d in R^{d+1}): for alpha in N, lambda_k != 0 for k <= alpha; lambda_k = 0 for k > alpha with k = alpha (mod 2); otherwise
    |lambda_k| ~ k^{-d/2-alpha-1/2}; the random-feature kernel eigenvalue is lambda_k^2 (so k^{-(d+1)-2alpha} in his d, i.e. k^{-D-2alpha} with D = ambient dimension).
    Also: the order n^{-1/d} for approximating Lipschitz functions with n ReLU neurons "cannot be improved (DeVore et al. 1989, Thm 4.2)".

* Bietti, Alberto; Mairal, Julien (2019). "On the inductive bias of neural tangent kernels." NeurIPS 2019; arXiv:1905.12173.
  URL opened: https://arxiv.org/pdf/1905.12173 [FT]
  - Lemma 17 (App. C.2, translating Bach): on S^{p-1}, sigma_alpha(u) = (u)_+^alpha: lambda_{alpha,k} != 0 for k <= alpha; = 0 for k > alpha with k = alpha mod 2;
    |lambda_{alpha,k}| ~ C k^{-p/2-alpha} otherwise; for alpha in {0,1}: mu_{alpha,k} ~ C k^{-p-2alpha} otherwise. So kappa_1 (ReLU): mu_k ~ k^{-p-2}, zero for odd k >= 3.
    Prop. 5 (2-layer ReLU NTK, no bias): mu_0, mu_1 > 0, mu_k = 0 for odd k >= 3, mu_k ~ C(p) k^{-p} otherwise.

* Bietti, Alberto; Bach, Francis (2021). "Deep equals shallow for ReLU networks in kernel regimes." ICLR 2021 (header of the PDF); arXiv:2009.14397.
  URL opened: https://arxiv.org/pdf/2009.14397 [FT]
  - Thm 1 (simplified; full version = Thm 7, App. B; uses Flajolet-Sedgewick Thm VI.8 and Watson's theorem, Lemma 6): if kappa is C^infty on (-1,1) with
    kappa(1-t) = p_1(t) + c_1 t^nu + o(t^nu), kappa(-1+t) = p_{-1}(t) + c_{-1} t^nu + o(t^nu), nu > 0 non-integer, then mu_k ~ (c_1 + c_{-1}) C(d,nu) k^{-d-2nu+1}
    for even k (if c_1 != -c_{-1}) and mu_k ~ (c_1 - c_{-1}) C(d,nu) k^{-d-2nu+1} for odd k (if c_1 != c_{-1}); when |c_1| = |c_{-1}| one parity is o(.).
    This answers the "(1-t)^beta endpoint singularity -> Gegenbauer coefficient asymptotics" question directly. Taylor coefficients b_k of kappa decay like k^{-nu-1} (App. B.1).
  - Eq. (12)-(13): kappa_0(1-t) = 1 - (sqrt2/pi) t^{1/2} + O(t^{3/2}); kappa_1(1-t) = 1 - t + (2 sqrt2/(3 pi)) t^{3/2} + O(t^{5/2}); kappa_1(-1+t) = (2 sqrt2/(3 pi)) t^{3/2} + O(t^{5/2})
    (printed O(t^{5/3}), a typo). Same coefficient at both ends (c_1 = c_{-1}), so odd degrees cancel: "decay of k^{-d-2} for even coefficients for kappa_1".
  - Cor. 2: deep RF/NNGP kernel of an L-layer ReLU net, L >= 3 (i.e. at least 2 hidden ReLU layers): mu_k ~ C(d,L) k^{-d-2}, C(d,L) differs with the parity of k and grows
    linearly with L; both parities nonzero. Cor. 3: deep NTK ~ k^{-d}, C grows quadratically with L. Sec. 3.3: Laplace kernel e^{-c||x-y||} gives k^{-d}.
  - Caveat: all of this is the infinite-width kernel recursion; the finite-width kernel E_B[g(h)g(h')] is a different dot-product function (1/n corrections; see Sec. 4).

* Geifman et al. (2020). "On the similarity between the Laplace and neural tangent kernels." NeurIPS 2020; arXiv:2007.01580. URL opened: https://arxiv.org/pdf/2007.01580 [FT]
  Thm 1 (with zero-initialized bias): NTK of FC nets has lambda_k > 0 for all k; C_1 k^{-d} <= lambda_k <= C_2 k^{-d} for L = 2, lambda_k >= C_3 k^{-d} for L >= 3.
  Chen, Lin; Xu, Sheng (2021). "Deep neural tangent kernel and Laplace kernel have the same RKHS." ICLR 2021; arXiv:2009.10683. URL opened: https://arxiv.org/abs/2009.10683 [ABS].
  Both concern the NTK (k^{-d}), not the NNGP/RF kernel (k^{-d-2}) that your back-operator uses; Bietti-Bach is the precise reference.

Table (infinite width, sphere S^{d-1}, d = ambient dim): ReLU RF/NNGP, one hidden layer: k^{-(d+2)}, even k only (plus k=1) | step-function kappa_0: k^{-d}, odd k |
ReLU RF/NNGP with >= 2 hidden layers: k^{-(d+2)}, both parities, constant ~ L | two-layer ReLU NTK, no bias: k^{-d}, odd k >= 3 zero | deep NTK: k^{-d}, constant ~ L^2.
Exact parity identity (my algebra, checked): kappa_1(t) - kappa_1(-t) = t, so the odd part of kappa_1 is exactly t/2.

======================================================================
## 4. High-dimensional random features and kernels; correlation maps; finite width

* Ghorbani, Behrooz; Mei, Song; Misiakiewicz, Theodor; Montanari, Andrea (2021). "Linearized two-layers neural networks in high dimension." Ann. Statist. 49(2):1029-1054
  (venue/pages from a search result, Project Euclid page not opened: UNV); arXiv:1904.12191. URL opened: https://arxiv.org/pdf/1904.12191 [FT]
  - Thm 1 (RF risk): (a) N <= d^{l+1-delta}: R_RF(f) ~ R_RF(P_{<=l} f) + ||P_{>l} f||^2_{L2}; (b) N = omega(d^l): R_RF(P_{<=l} f) <= eps||P_{<=l} f||^2. Thm 2 same for NT with degree l+1.
    Abstract: kernel methods with n between d^{l+delta} and d^{l+1-delta} samples fit at most a degree-l polynomial. This is the "degree-l needs ~d^l" polynomial barrier;
    d^l is the number of degree-l harmonics for fixed l (compare Section 8, count formula).

* Mei, Song; Misiakiewicz, Theodor; Montanari, Andrea (2022). "Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration."
  Appl. Comput. Harmon. Anal. 59:3-84 (venue per a third-party reference list: UNV); arXiv:2101.10588. URL opened: https://arxiv.org/abs/2101.10588 [ABS]
  KRR is well approximated by projection on the top l(n) kernel eigenfunctions; random features with N <= n^{1-delta} are approximation-limited, N >= n^{1+delta} match KRR.

* El Karoui, Noureddine (2010). "The spectrum of kernel random matrices." Ann. Statist. 38(1):1-50; arXiv:1001.0492. URL opened: https://arxiv.org/abs/1001.0492 [ABS]
  For kernel matrices f(X_i'X_j/p) with p comparable to n, "the problem becomes essentially linear" (abstract). My gloss of the known mechanism, not read in the text: in the proportional regime the kernel matrix behaves like a
  constant + linear (sample-covariance) term + a multiple of the identity, i.e. only low degrees of f matter, which is the n ~ d (degree <= 1) case of the Ghorbani et al. barrier.

* Daniely, Amit; Frostig, Roy; Singer, Yoram (2016). "Toward deeper understanding of neural networks: the power of initialization and a dual view on expressivity." NeurIPS 2016
  (pp. 2253-2261, as cited in Geifman et al.); arXiv:1602.05897. URL opened: https://arxiv.org/pdf/1602.05897 [FT]
  - Def. 4: dual activation sigma_hat(rho) = E_{(X,Y)~N_rho} sigma(X) sigma(Y); dual kernel kappa_sigma(x,y) = sigma_hat(<x,y>_H). Thm 8 = Schoenberg (above). Intro bullet: as depth grows
    the dual kernel of a FC net converges to a degenerate form for any nonlinear activation.

* Poole, Ben; Lahiri, Subhaneil; Raghu, Maithra; Sohl-Dickstein, Jascha; Ganguli, Surya (2016). "Exponential expressivity in deep neural networks through transient chaos." NeurIPS 2016;
  arXiv:1606.05340. URL opened: https://arxiv.org/abs/1606.05340 [ABS]. Mean-field theory of random deep nets: order-to-chaos transition; curvature grows exponentially with depth in the chaotic phase.
  Schoenholz, Samuel S.; Gilmer, Justin; Ganguli, Surya; Sohl-Dickstein, Jascha (2017). "Deep information propagation." ICLR 2017; arXiv:1611.01232. URL opened: https://arxiv.org/abs/1611.01232 [ABS].
  Depth scales for signal and correlation propagation; one diverges at the edge of chaos. (Correlation map c -> c' is the dual-activation map of Daniely et al.)

* Hayou, Soufiane; Doucet, Arnaud; Rousseau, Judith (2019). "On the impact of the activation function on deep neural networks training." ICML 2019 (PDF header); arXiv:1902.06853.
  URL opened: https://arxiv.org/pdf/1902.06853 [FT]
  - Lemma 3: for ReLU-like activations the (weak) edge of chaos is the single point (sigma_b^2, sigma_w^2) = (0, 2) = your ensemble. Prop. 1: on the EOC, 1 - c^l ~ gamma(1 - c~^l) ~ 9 pi^2/(2 l^2)
    (polynomial, vs exponential in the ordered phase); the network acts like a residual ReLU net for correlation propagation.
    Consistency check (mine): with 1-kappa_1(1-u) = u - a u^{3/2}, a = 2 sqrt2/(3 pi) (Bietti-Bach Eq. 13), iterating gives u_L ~ 4/(a^2 L^2) = 9 pi^2/(2 L^2): matches.

* Hanin, Boris; Nica, Mihai (2020). "Products of many large random matrices and gradients in deep neural networks." Commun. Math. Phys. 376:287-322 (venue from search result: UNV);
  arXiv:1812.05994. URL opened: https://arxiv.org/pdf/1812.05994 [FT]
  - Thm 1: (n_0/n_d)||M^{(d)} u||^2 ~ exp(N(-beta/2, beta)), beta = (3/p - 1) sum_i 1/n_i + (mu_4-3)/(p n_1)||u||_4^4. ReLU nets correspond to p = 1/2 (Prop. 2), so beta = 5 sum 1/n_i for Gaussian
    weights: relative fluctuation of the output norm^2 at depth L, width n is ~ 5L/n. Relevant to normalizing errors "relative to the per-network output variance".
  Hanin, Boris; Rolnick, David (2018). "How to start training: the effect of initialization and architecture." NeurIPS 2018; arXiv:1803.01719. URL opened: https://arxiv.org/pdf/1803.01719 [FT]
  (Cor. 1: variance 2/fan-in is the only stable scale; Cor. 2: variance of activation length grows like exp(c sum 1/n_k).)
  Hanin, Boris (2024). "Random fully connected neural networks as perturbatively solvable hierarchies." JMLR 25(267):1-58 (as cited in Wu et al.); arXiv:2204.01058. URL opened: https://arxiv.org/abs/2204.01058 [ABS].
  Cumulants of the output in powers of 1/n; "contributions of order 1/n^k often grow like L^k" (depth-to-width ratio L/n is the effective depth).
  Roberts, Daniel A.; Yaida, Sho; Hanin, Boris (2022). "The principles of deep learning theory." Cambridge Univ. Press; arXiv:2106.10165. URL opened: https://arxiv.org/abs/2106.10165 [ABS].
  Finite-width/depth corrections organised by L/n; source for the finite-width kernel recursion your exact kappa_n needs (the book chapters were not read).

======================================================================
## 5. Information-based complexity of integration

* Novak, Erich (2016). "Some results on the complexity of numerical integration." In Cools & Nuyens (eds.), Monte Carlo and Quasi-Monte Carlo Methods (MCQMC 2014),
  Springer Proc. Math. Stat. 163, pp. 161-183; arXiv:1409.6714. URLs opened: https://arxiv.org/abs/1409.6714 and https://arxiv.org/pdf/1409.6714 [FT]
  - Thm 0 (Bakhvalov/Smolyak): for convex balanced classes, nonadaptive linear algorithms are optimal. Thm 1 (Bakhvalov 1959): e(n, F^k_d) ~ n^{-k/d} for F^k_d = {f: [0,1]^d -> R, ||D^a f||_inf <= 1, |a| <= k}.
    Thm 3 (Bakhvalov 1959): e^ran(n, F^k_d) ~ n^{-k/d-1/2}. Remark 3: the randomized upper bound is "separation of the main part": with n = 2m, use m values to build f_m with ||f-f_m||_2 ~ m^{-k/d}, then the
    unbiased estimator S(f_m) + (1/m) sum (f-f_m)(X_i). Remark 1b: constants c_{d,k} decay with d. All known lower bounds use Bakhvalov's/Yao's average-case (Fubini) argument.
  - For us (my reading): "approximate + sampled residual" is the generic way to combine any s-dimensional summary with sampling; IBC lower bounds assume black-box function values, whereas
    your estimators read the weights, so Bakhvalov does not bound them, while the cut-rank bound does (for linear summaries).
  - Original: Bakhvalov, N. S. (1959), Vestnik MGU Ser. Mat. Mekh. 4:3-18 (cited as [5] there; Russian journal, not opened: UNV). Books (cited as [78],[89]-[92] there, not opened: UNV):
    Novak, "Deterministic and Stochastic Error Bounds in Numerical Analysis" (LNM 1349, 1988); Novak & Wozniakowski, "Tractability of Multivariate Problems" Vols. I-III (EMS 2008/2010/2012).

* Hinrichs, Aicke; Novak, Erich; Ullrich, Mario; Wozniakowski, Henryk (2014). "The curse of dimensionality for numerical integration of smooth functions." Math. Comp. 83(290):2853-2863;
  arXiv:1211.0871. URL opened: https://arxiv.org/pdf/1211.0871 [FT]
  - Abstract: n(eps, C^r_d) >= c_r (1+gamma)^d for eps <= eps_0. The "Main Theorem" in the arXiv v2 text: n(eps, C^r_d) >= c_r (1-eps) d^{d/(2r+3)}. Part II: J. Complexity 30 (2014) 117-143, arXiv:1304.3372 (search result only: UNV).
    Relevance: for fixed smoothness the deterministic evaluation-based cost is exponential in the dimension (here d ~ n); your width n is the dimension.

* Brauchart, Johann S.; Saff, Edward B.; Sloan, Ian H.; Womersley, Robert S. (2014). "QMC designs: optimal order Quasi Monte Carlo integration schemes on the sphere." Math. Comp. 83(290):2821-2851; arXiv:1208.3267.
  URL opened: https://arxiv.org/abs/1208.3267 [ABS]. Worst-case error O(N^{-s/d}) in H^s(S^d), s > d/2, for QMC designs; i.i.d. uniform random points are NOT QMC designs for any s > d/2 (their error is N^{-1/2}).
  Sphere-specific counterpart of Bakhvalov's deterministic rate (S^d in R^{d+1}).

* Bach, Francis (2017). "On the equivalence between kernel quadrature rules and random feature expansions." JMLR 18(21):1-38 (confirmed at https://jmlr.org/papers/v18/15-178.html); arXiv:1502.06800 (v2).
  URL opened: https://arxiv.org/pdf/1502.06800 [FT]
  - Closest published statement of "number of linear features = number of kernel eigenvalues above a threshold". Props. 1-2 (upper): approximating the unit ball of the RKHS in L2(rho) to O(lambda) needs about
    10 d(lambda) log(2 d(lambda)) features drawn from an optimized (non-uniform) distribution, with d(lambda) (degrees of freedom) within constants of m*(lambda) = max{m : mu_m >= lambda}. Prop. 3 (lower bound, any family psi_1..psi_n with
    average squared norm <= 2 tr(Sigma)/delta and coefficients ||beta||^2 <= 4/n, i.e. a stability-type constraint, approximating the unit ball within 4 lambda): n >= max{m : mu_m >= 144 lambda} / (4 log(10 tr(Sigma)/(lambda delta))).
    Upper and lower bounds match up to logs using only the eigenvalue decay; Sec. 4.4 specializes to quadrature and recovers Sobolev-space rates (Bach's reference list cites Hesse's lower bound for worst-case cubature error on
    spheres; not opened: UNV). Differences from your setting: it bounds features for the RKHS unit ball (worst case over functions), you need the HS tail under a random-front ensemble; the counting mechanism is the same.

* Griebel, Michael; Kuo, Frances Y.; Sloan, Ian H. (2013). "The smoothing effect of integration in R^d and the ANOVA decomposition." Math. Comp. 82:383-400 (venue from search results: UNV; a 2016 correction
  note exists: the main Thm 3.1 is correct but does not apply to Asian options). URL opened: https://ins.uni-bonn.de/legacy/wissrech.ins.uni-bonn.de/research/pub/griebel/inf-smooth-101215.pdf (INS Preprint 1007) [FT]
  - f = max(phi, 0) or |phi| with smooth phi on R^d: all ANOVA terms except the highest-order one can have unlimited smoothness (Thm 2 inheritance, Thm 4/5) when d(phi)/dx_j does not change sign. This is the "kinks get smoothed by integrating
    out variables" mechanism behind good QMC rates for kinked integrands; deep ReLU nets do not satisfy the sign condition in general (my reading).

* Average case / Bayesian quadrature. Ritter, Klaus (2000). "Average-case analysis of numerical problems." Lecture Notes in Math. 1733, Springer (Chapter VI, "Integration and approximation of multivariate functions", pp. 133-182 per a search-result chapter list).
  Book page not opened (Springer redirect): [UNV]; I could not confirm any isotropic-random-field-on-sphere theorem there.
  Briol, F.-X.; Oates, C. J.; Girolami, M.; Osborne, M. A.; Sejdinovic, D. (2015/2019). "Probabilistic integration: a role in statistical computation?" arXiv:1512.00933 (Statistical Science 2019 from memory: UNV).
  URL opened: https://arxiv.org/abs/1512.00933 [ABS]. Posterior contraction rates for probabilistic integrators; no sphere or deep-net specifics.

======================================================================
## 6. Average-case hardness tools: what each would need for "estimate the mean of a random deep net better than sampling"

* Low-degree method. Kunisky, Dmitriy; Wein, Alexander S.; Bandeira, Afonso S. (2019/2022). "Notes on computational hardness of hypothesis testing: predictions using the low-degree likelihood ratio." arXiv:1907.11636.
  URL opened: https://arxiv.org/abs/1907.11636 [ABS]. Schramm, Tselil; Wein, Alexander S. (2022). "Computational barriers to estimation from low-degree polynomials." Ann. Statist. 50(3):1833-1858; arXiv:2008.02269.
  URL opened: https://arxiv.org/abs/2008.02269 [ABS] (lower bound on the MSE of any degree-D polynomial estimator in signal-plus-noise models).
  Needs: a noisy observation / planted model and polynomials in the observed data. Here the data are the weights and the target M(W) is a deterministic function with no noise, so the natural object is the
  L2(W)-projection of M(W) onto degree <= D Wiener-chaos polynomials; a computational-hardness reading needs an added noise model. Caveat: Mao, Songtao (2026), "The polynomial-time low-degree conjecture is false", arXiv:2607.20318
  (v2 28 Oct 2026): https://arxiv.org/abs/2607.20318 [ABS] disproves the standard binary poly-time formulation for a permutation-invariant planted graph model; treat low-degree predictions as heuristic.
* Overlap gap property. Gamarnik, David (2021). "The overlap gap property: a topological barrier to optimizing over random structures." PNAS (DOI 10.1073/pnas.2108492118 on the arXiv page; volume 118(41) from memory: UNV; arXiv title: "...a geometric barrier..."); arXiv:2109.14409.
  URL opened: https://arxiv.org/abs/2109.14409 [ABS]. Needs: an optimization/search problem over a random landscape with a solution-overlap structure, and an algorithm class stable under input perturbation.
  It could address finding argmax of the random net, not computing a single expectation (no solution set).
* SQ lower bounds via moment matching. Diakonikolas, Ilias; Kane, Daniel M.; Stewart, Alistair (2017). "Statistical query lower bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures." FOCS 2017 (from memory: UNV);
  arXiv:1611.03473. URL opened: https://arxiv.org/pdf/1611.03473 [FT] (Thm 1.1 SQ lower bound for GMMs; Thm 1.2 robust mean). Needs: i.i.d.-sample access modeled as expectation queries with tolerance. Your estimators see the weights
  (white box), so SQ does not apply; the black-box analogue is the IBC bounds of Section 5. (Diakonikolas-Kane's 2023 book not opened: UNV.)
* Spin-glass free energy. Auffinger, Antonio; Ben Arous, Gerard; Cerny, Jiri (2013). "Random matrices and complexity of spin glasses." Comm. Pure Appl. Math. 66 (2013) (venue from memory: UNV); arXiv:1003.1129.
  URL opened: https://arxiv.org/abs/1003.1129 [ABS] (complexity of spherical p-spin models). Needs: a Gaussian field on the sphere plus a Gibbs temperature; yields laws over the disorder (e.g. the fluctuation of M), not instance-wise estimators.
  Link (my inference): at infinite width a random ReLU net restricted to the sphere is a Gaussian field with covariance kappa(x.y) = sum_p b_p (x.y)^p, b_p ~ p^{-5/2} (Bietti-Bach App. B.1 with nu = 3/2), i.e. a mixed spherical p-spin Hamiltonian;
  its sphere-average is exactly Gaussian with variance mu_0, so the interesting content is instance-wise (AMP/TAP-type) algorithms, which I did not verify.

======================================================================
## 7. Directly related: rank across a cut, meet-in-the-middle, mechanistic estimation

* Wu, Wilson; Lecomte, Victor; Winer, Michael; Robinson, George; Hilton, Jacob; Christiano, Paul (2026). "Estimating the expected output of wide random MLPs more efficiently than sampling."
  arXiv:2605.05179 (v1 6 May 2026, v2 14 May 2026; 68 pp.). URLs opened: https://arxiv.org/abs/2605.05179 and https://arxiv.org/pdf/2605.05179 [FT]
  - Setting (Sec. 2.1-2.2): MLP without biases (adapted in App. F), width n, weights i.i.d. N(0, 2/n), X ~ N(0, I_n); estimate E_X[M_theta(X)] from the weights, MSE O(eps^2) averaged over theta. Quenched, not annealed
    (Sec. 1, Sec. 8.3). Monte Carlo baseline (Sec. 2.3): N = 1/eps^2 samples, time Theta(n^2/eps^2).
  - Thm 5.1 (polynomial activations): E_theta[(G(theta,eps) - E_X M_theta(X)_i)^2] = O(eps(n)^2) in time O(n^2/eps^2). Thm 5.2: if eps(n)^2 = O(1/n^2), time O(n/eps^2), a factor n better than Monte Carlo.
    Sec. 5 conjecture: MSE <= c_K (L/n)^K, time c'_K L n^{K+1} (basic) or c''_K L^2 n^K (factorized, K >= 3); cumulant order K ~ log(1/eps^2)/log n. ReLU: empirical only (Fig. 1: depth 4, width 256), "with additional effort"
    extendable. Sec. 7.3 (open problems): "we do not know whether it is possible to match the performance of random sampling at large depth ... or for narrow networks (i.e., as the error tolerance eps -> 0 while the width n is held fixed)".
  - The estimator class is a linear one-way summary (cumulant tensors of the front, contracted with the back); App. S.2.2 (Thm S.2.4): harmonic decomposition of symmetric tensors coincides with the decomposition into spherical
    harmonics (cites Dai-Xu). Sec. 7.1 states the ARC conjecture that mechanistic estimation can match any estimation procedure given the training trajectory.
* Hilton, Jacob. "Deductive estimation for random halfspace intersections." ARC technical note (PDF dated May 2, 2025; hosted May 2026).
  URL opened: https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf [FT]
  - Sec. 2: deduction-projection estimators, "expectation machines". Thm 3.1 (Sketching theorem, informal): for a random function f on R^n and eps > 0 there is a sketch g~ of g with E_f[(<f|g~> - <f|g>)^2] < eps^2 E_f[<f|f>] <g|g>, specified by O(1/eps^2) bits;
    proof: project g onto the top-m eigenvectors of E_f[f f^T], error <= tr(E[f f^T]) <g|g>/(m+1), m = 1/eps^2. Sec. 4: for K(x,y) = E_f[f(x)f(y)] a function of the angle (here ((1/2)(1 - theta/pi))^{k-i}), eigenfunctions are harmonic polynomials p(x/|x|)
    with eigenvalue depending only on the degree (Funk-Hecke, Dai-Xu Thm 1.2.9 and Thm 1.1.3); projection formula proj_d g(x) = (1/|S^{n-1}|) int g(y) ((d+alpha)/alpha) C_d^{(alpha)}(x.y) dsigma(y), alpha = (n-2)/2. Sec. 6: number of projected
    eigenfunctions m = binom(n+D, n) - binom(n+D-2, n) (harmonic polynomials of degree <= D, Dai-Xu Cor. 1.1.5), log m ~ D log(n/D) for D = o(n); target runtime O(nk/eps^2) = O(nkm).
  - Same object as your cut operator (head = front density g, tail = back f). Normalization differs from yours: here the error budget is eps^2 E<f|f> <g|g> with <g|g> the L2 norm of the head; yours is relative to the per-network output variance.
* Hilton, Jacob; Robinson, George; Neyman, Eric; Christiano, Paul; Winer, Michael; Lecomte, Victor; Wu, Wilson; Wu, Gabriel (2026-05-15). "Mechanistic estimation for expectations of random products." ARC blog.
  URL opened: https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/ [ABS: read via WebFetch summary plus the raw HTML link list]. Deduction-projection with "mechanistic L2 sketching" (project the head onto the leading eigenfunctions of E[f f^T] of the random tail;
  error set by the largest unused eigenvalue); roughly matches or beats sampling up to log factors for halfspace intersections, random #3-SAT, {0,1}-permanents (permanents lag). No lower bounds stated. Linked notes (PDFs, same folder
  https://www.alignment.org/content/files/2026/05/): W. Wu, "Products of symmetric random functions" (rand_sym_dedproj.pdf, opened [FT]; Thm 20: for a compact Gelfand pair (G,H), projecting partial products onto the first D+1 irreps gives total MSE <= c_D (tr K)^k, sampling with m samples gives (tr K)^k/m; for S^{n-1}=SO(n)/SO(n-1) the irreps are the harmonic polynomials);
  perm_dedproj.pdf and products_fourier_simple_v5.pdf not opened [UNV].
* Neyman, Eric; Lecomte, Victor; Wu, Wilson; Winer, Michael; Hilton, Jacob; Robinson, George (2025-11-18). "Competing with sampling." ARC blog. URL opened: https://www.alignment.org/blog/competing-with-sampling/ [ABS: read via WebFetch summary].
  The matching sampling principle (a mechanistic estimator with an O(|theta|) explanation, time comparable to 1/eps^2 samples, no worse squared error than sampling averaged over a random context); examples: random half-spaces via Hermite basis, random MLPs via
  cumulant propagation; narrow MLPs, CNNs, transformers listed as open; remarks that in some eps ranges the best mechanistic estimate is only slightly better than sampling.
* Dunbar, John; Aaronson, Scott (2025/2026). "Wide neural networks as a baseline for the computational no-coincidence conjecture." arXiv:2510.06527 (v2 11 Jan 2026). URL opened: https://arxiv.org/abs/2510.06527 [ABS].
  Wide random nets have nearly independent outputs iff the activation is nonlinear with zero Gaussian mean (tanh yes; ReLU/GeLU only with an additive shift); plain ReLU does not. Relevance: with plain ReLU the outputs share a large mean/common component, consistent with kappa_1(0) = 1/pi > 0.
* Christiano, Paul; Neyman, Eric; Xu, Mark (2022). "Formalizing the presumption of independence." arXiv:2211.06738. URL opened: https://arxiv.org/abs/2211.06738 [ABS]. Background on heuristic estimators; origin of cumulant propagation (App. D per Wu et al.).
* Fischer, Kirsten; Rene, Alexandre; Keup, Christian; Layer, Moritz; Dahmen, David; Helias, Moritz (2022). "Decomposing neural networks as mappings of correlation functions." Phys. Rev. Research 4, 043143; arXiv:2202.04925.
  URL opened: https://arxiv.org/abs/2202.04925 [ABS]. Iterated transformation of distributions, correlation functions up to second order dominate in internal layers (cumulant-propagation precursor cited by Wu et al.).
* Separation rank across a partition (not a depth cut). Cohen, Nadav; Sharir, Or; Shashua, Amnon (2016). "On the expressive power of deep learning: a tensor analysis." COLT 2016 (PMLR 49:698-728); arXiv:1509.05009. URL opened: https://arxiv.org/abs/1509.05009 [ABS].
  Cohen, Nadav; Shashua, Amnon (2017). "Inductive bias of deep convolutional networks through pooling geometry." ICLR 2017; arXiv:1605.06743. URL opened: https://arxiv.org/abs/1605.06743 [ABS]. Separation rank = how far a function is from being
  separable across a partition of the INPUT variables (shallow nets: linear; deep nets: exponential for favored partitions). I found no paper on rank across a depth cut of a (random) network; the closest are the Hilton/Wu notes above.

======================================================================
## 8. Derived facts, checks, and gaps (labelled: mine, not literature)

D1. Exactness. For any fixed front/back ensemble the mean-square eps-approximate rank of the cut operator is the SVD tail (Section 1). Nonlinear phi, psi do not help inside this criterion; they only matter for nonlinear/adaptive
    estimators, where Section 2 (continuity/stability) applies. Caveat on the normalization: dividing each squared error by the per-network output variance makes the weight depend on (front, back) jointly (not a product),
    and weighted low-rank approximation with a non-separable weight has no SVD solution; Eckart-Young is exact for a constant or product-form weight. Since the per-network output scale concentrates for wide nets (Hanin-Nica Thm 1:
    log-variance beta = 5 sum 1/n_i for ReLU, i.e. ~5L/n), normalizing by the ensemble mean should change the criterion only by a 1 + O(sqrt(L/n))-type factor (my estimate, not proved); a rigorous write-up needs that step.
D2. Harmonic counts. Number of harmonics of degree <= D on S^{n-1}: binom(n+D, n) - binom(n+D-2, n) (Hilton Sec. 6) ~ n^D/D! for D << n (your large-n regime) and ~ 2 D^{n-1}/(n-1)! for fixed n, D -> infinity.
    Multiplicity of degree k: N(n,k) ~ 2 k^{n-2}/(n-2)! (Bietti-Bach Sec. 2.2). With mu_k ~ C k^{-(n+2)}, the kernel's own tail energy beyond degree D is sum_{k>D} N(n,k) mu_k ~ D^{-3}; for a back with exactly one hidden ReLU layer only
    even degrees (plus k=1) are nonzero, which halves the count. These give the "D(eps) set by (1-t)^{3/2}" scaling; whether the cut operator's spectrum actually follows it depends on the second moments of the fronts' harmonic components.
D3. Depth. The singularity coefficient of the L-fold ReLU recursion grows linearly in L (Bietti-Bach Cor. 2); my composition check: kappa^{(L)}(1-u) ~ 1 - u + L a u^{3/2} for u << L^{-2}; Hayou's 9 pi^2/(2 l^2) agrees (Section 4).
D4. Finite width. Bietti-Bach/Bach/Cho-Saul treat the infinite-width kernel. Your E_B[g(h)g(h')] at finite n is a different dot-product function (still homogeneous of degree (1,1)); I found no source proving it keeps the (1-t)^{3/2} endpoint
    singularity or the k^{-(n+2)} decay at finite n. This is the main literature gap for the claim as stated.
D5. Normalization trap (my reasoning; not stated in either source). Hilton's guarantee is E_f[(<f|g~> - <f|g>)^2] < eps_H^2 E<f|f> <g|g>, relative to the head's squared L2 norm <g|g> (head = density of the front's cut-layer law w.r.t. N(0,I));
    yours is relative to the per-network output variance, which after averaging over the back is about E_nu|h|^2 kappa(1) (a first-moment quantity of the front law). The ratio rho = V/(E<f|f><g|g>) is ~1/<g|g> when the front law is far from Gaussian,
    so m = 1/eps_H^2 = <g|g>/eps^2 in your normalization can be much larger than 1/eps^2. A rank lower bound far above 1/eps^2 is therefore not in conflict with Hilton's Thm 3.1, but check <g|g> for your fronts before comparing with ARC's
    upper bounds (concentrated angular laws, as the depth collapse c_l -> 1 of Section 4 suggests, make <g|g> large).
D6. Boolean one-way/product-measure caveat (Section 1): rank is only a lower bound for general protocols; your statement is restricted to linear inner-product summaries, which is exactly where rank is the right invariant.

======================================================================
## 9. Not verified / suggested next fetches
- Krause 1996 (TCS 156); Linial-Shraibman 2009 (RSA 34); Forster 2002; Kremer-Nisan-Ron page (inner-product-of-real-vectors section); Bakhvalov 1959; Novak-Wozniakowski books; Ritter 2000 book; Pinkus/Lorentz et al./Tikhomirov books;
  Bordin et al. 2002 (J. Complexity); Dai-Xu 2013; Schoenberg 1942 primary; DHM89 publisher page and its Thms 3.1, 4.2 (only second-hand here).
- Venue/page numbers taken from search metadata, not from publisher pages: CMS (JACM 67(4):23), CDPW (FoCM 22:607-648), Hanin-Nica (CMP 376:287-322), Ghorbani et al. (AoS 49(2):1029-1054), Mei et al. (ACHA 59:3-84), Griebel-Kuo-Sloan (Math. Comp. 82:383-400).
- The ARC links above were read at 2026-10-30 ~23:35-23:42Z; the dates quoted (May 2026) are the pages' own.
