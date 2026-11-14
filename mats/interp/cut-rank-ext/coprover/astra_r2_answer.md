codex
**Main findings:** most structural lemmas survive, but B1’s proof needs correction and the fixed regularity budgets create additional error floors. The Lipschitz bound strengthens to \(n(n+1)/2\); ARC’s exact-cumulant, one-step truncation has a provable bias floor for every finite \(K\), with a weak quantitative rate.

Checked the supplied files and Wu et al.’s actual equations (13)–(14), PDF pp. 47–48, on 2026-10-31 at 11:26 UTC. No files changed. Below, \(n\ge2\), \(a=2(n-1)/3\), and K1–K5 are assumed.

**1. Review of Claude’s lemmas**

Line references refer to [FORMAL-ext.md](/home/agent/workspaces/mats/work/cut-rank-ext/FORMAL-ext.md).

| Item | Verdict | Check or necessary correction |
|---|---|---|
| Theorem A, lines 28–40 | **CORRECT** | The rectangle factorization and Jensen argument work. Fix public coins before applying the rank bound; averaging over public coins need not preserve rank. The budget must bound transcript length on every run. |
| Corollary A1 | **CORRECT** | Superseded quantitatively by Astra A4’s \(B\ge n\log_2(1/\varepsilon)-O(1)\) for arbitrary interaction. Section 2 below further improves **one-way** communication to coefficient \(n(n+1)/2\). |
| Proposition A2 | **CORRECT AFTER FIX** | The one-way quantization and Gaussian-tail argument work. For a common transcript producing the entire output vector, the two-way count is \((p_b+n+o(1))\log_2(1/\varepsilon)\), not \(p_b+1\). The latter is valid for a separate scalar-coordinate query model; that model does not support A4’s vector-quantization argument unchanged. |
| Theorem B | **CORRECT** | Monomial lifting gives \(2^B\binom{s+p}{p}\). The inversion follows from \(\binom{s+p}{p}\le(s+p)^p/p!\). Include the constant monomial. |
| B’s multi-round remark | **CORRECT** | With nonadaptive backward bits, fix their complete string and concatenate the resulting front features. Adaptive backward thresholds invalidate this lifting: \(1[x\ge b]\), obtained from one forward real and one backward bit, already has infinite separation rank. Astra A8 supplies a different bound when **every** message map is polynomial without branching. |
| B1 | **CORRECT AFTER FIX** | The Cauchy coefficient estimate is correct. The displayed tail inequality and claimed asymptotic equivalents are not; details below. The stated lower-bound conclusion remains valid and can be strengthened. Its fixed output bound also makes arbitrarily small error impossible on the unbounded Gaussian ensemble. |
| B2 | **CORRECT** | At fixed \(s,k\), tensor Jackson approximation gives \(O_{s,k}(\Lambda p^{-k})\); hence \(s\ge ka\). Constants absorb the conversion between coordinate degree and total degree. However, this is only a necessary condition: under the literal fixed derivative budget, every fixed \(s\) actually has an error floor; see Section 2. |
| C1 | **CORRECT AFTER FIX** | The exact parameter summary and local Lipschitz estimate work. Replace “one real” by \(n\) reals for the vector answer. Raw parameters do not satisfy a bounded-encoder hypothesis. |
| C2 | **CORRECT** | Positive homogeneity proves \(M=g_b(Eh_c)\). This establishes an exact decoder for ray laws. It does **not** establish exactness of Gaussian/cumulant propagation fed the ray law’s true covariance and higher cumulants. |
| C3 | **CORRECT AFTER FIX** | Exact scalar coding works on the compact parameter ball, with a measurable right inverse. Choose a suitable Peano/Hilbert curve to obtain the stated Hölder exponent; an arbitrary continuous surjection need not have it. For all Gaussian weights, use an unbounded scalar domain and a continuous surjection \(\mathbb R\to\mathbb R^{p_f}\). A bounded scalar interval cannot cover the unbounded target exactly. |
| C4 | **CORRECT** | The covering argument and Markov step are valid, including the Hölder variant. The exceptional encoding mass \(\eta\) must remain in subsequent hypotheses. |
| C5 | **CORRECT AFTER FIX** | State the required bound on \(\eta\). Also, \(S\) can vanish on E because an earlier front may be dead; one cannot always choose the asserted lower quartile. Restrict to a positive-probability event \(\alpha=Ea(x)\in[u,2u]\). A9 then gives dimension \(n\), including scale. Section 2 strengthens this to \(n(n+1)/2\). |
| C6 | **CORRECT AFTER FIX** | The failure of “rank \(\le s\)” is correctly identified. Variable-scale rays form an \(n\)-dimensional family, not \(n-1\). Cauchy/Jackson approximation gives regularity-dependent rank constraints; it does not restore a power-law state bound for every finite regularity class. |
| Theorem D | **CORRECT** | Expanding each queried function in the common basis gives rank at most \(N\). Nonadaptivity and the linear final readout are essential here. |
| D1 | **CORRECT AFTER FIX** | Correct **within Theorem D’s linear-readout, nonadaptive class**. Naming projected moments cannot outperform sending all moments with the same allowed readout. With nonlinear or adaptive processing, the applicable obstruction is \(e_K\), not the linear rank tail. |
| D2 | **CORRECT** | Directly follows from B, with its message-order and nonadaptivity restrictions. Count the entire backward naming transcript. |
| D3 | **CORRECT AFTER FIX** | General vector output costs \(p_b+n\) reals. The one-back-layer construction already correctly states \(n^2+n\). |
| Section 5 | **CORRECT AFTER FIX** | The cut reduction works. Carry over bounded-state, uniform-regularity, and exceptional-mass hypotheses. C3 does not give an exact scalar streaming implementation with continuous updates, because its encoder is only measurable. For positive averaged error, scalar continuous streaming approximations can instead be built by finite quantization and smoothing, at fixed depth and without stability bounds. |
| Section 6 | **CORRECT AFTER FIX** | Ensemble versus fixed-network quantifiers are correctly distinguished. C2 only applies to ray fronts; bounded-domain C3 only applies there. Conditional small-ball bounds on the rare ray event do not establish failure on most networks. K3 controls the rank-derived bounds, not every independently imposed regularity obstruction. |
| Section 7 | **CORRECT AFTER FIX** | The computational limitation is correctly identified. Bakhvalov’s stated rate applies to a bounded-domain Lipschitz class; Gaussian inputs require an additional truncation/weighted argument. Stratified randomized cubature can beat ordinary i.i.d. Monte Carlo’s precision exponent, but is itself sampling. For interactive case splitting, A counts **all communicated bits**, not merely the final message. |

For **B1**, multiset splitting does give
\[
\binom{m_1+m_2+s-1}{s-1}
\le
\binom{m_1+s-1}{s-1}\binom{m_2+s-1}{s-1}.
\]
Applied with \(m_1=p+1\), its immediate tail bound has
\(\binom{p+s}{s-1}\), not \(\binom{p+s}{s}\). The displayed inequality fails, for example, at \(p=0,s=2\).

There is a better, dimension-independent argument. Write \(R=\sum_{m\ge0}P_m\) in homogeneous Taylor components. For each \(z\in[-\rho,\rho]^s\), apply one-variable Cauchy estimates to
\[
t\longmapsto R(tz/q,b),\qquad |t|\le1,\quad q=\rho/\rho'.
\]
Then
\[
|P_m(z,b)|\le A\sqrt{\bar V}\,q^m,\qquad
\left|R-\sum_{m=0}^pP_m\right|
\le \frac{A\sqrt{\bar V}\,q^{p+1}}{1-q}.
\]
This also validates B1’s stated sufficient condition, despite its incorrect intermediate inequality.

Put \(\ell=\log(1/\varepsilon)\), \(\beta=\log(1/q)\). Taking
\(p=\ell/\beta+O(1)\) gives
\[
\binom{s+p}{s}\ge C(2\varepsilon)^{-a}.
\]
Consequently,
\[
s=\Omega_{a,q}(\ell),
\]
stronger than the stated \(\ell/\log\ell\) bound. More precisely,
\[
\liminf_{\varepsilon\downarrow0}\frac{s}{\ell}\ge c(a,q),
\]
where \(c(a,q)>0\) solves
\[
(c+\beta^{-1})\log(c+\beta^{-1})
-c\log c-\beta^{-1}\log(\beta^{-1})=a.
\]
The claimed equivalents at lines 71–72 are false: when \(s=\sigma\ell/\log\ell\) and \(p=\Theta(\ell)\), the logarithm of the binomial coefficient is \(o(\ell)\).

C3 does not contradict my round-one scalar remark: my construction required a continuous encoder. Claude’s exact construction uses a measurable, generally discontinuous encoder.

I also rechecked **A5’s continuity step**. It survives: bounded even \(Q\)-moments give uniform integrability of moments of orders strictly below \(Q\); weak limits remain in the bounded-complexity cone-linear family; \(Q>K_0\) identifies the limit uniquely. Highest moments can lose mass, which is why the closure argument uses lower moments and why globally extended intermediate updates need descending orders. Tietze then applies to the closed moment-image domain times back-weight space. For \(K<Q\), continuous readouts attain the **infimum in the limit**, not necessarily the conditional-expectation minimizer itself.

**2. Strengthenings**

**2(a). Analytic and \(C^k\) upper bounds**

The proposed matching upper bounds are **false under the literal uniform budgets on the genuine Gaussian ensemble**.

For B1, every estimate satisfies \(|\widehat M_j|\le A\sqrt{\bar V}\), so
\[
\mathrm{Err}\ge
E_{f,b,j}\bigl(M_j-A\sqrt{\bar V}\bigr)_+^2>0.
\]
This holds for every \(s\). Thus neither \(s=\Theta(n\ell)\) nor any larger state removes that floor while \(A\) stays fixed.

For B2, even bounding only the derivatives of order exactly \(k\) creates a floor at every fixed \(s\). Taylor expansion about zero gives
\[
R(z,b)=P_{k-1,b}(z)+U(z,b),\qquad
|U|\le D_{s,k,\Lambda}\sqrt{\bar V}.
\]
The polynomial part has separation rank at most
\[
N=\binom{s+k-1}{k-1}.
\]
The target’s support contains arbitrarily large multiples of \(N+1\) linearly independent elements of \(H\): use the infinite-rank ray family and scale the front weights. Any \(N\)-dimensional subspace misses at least one of these targets by more than \(2D_{s,k,\Lambda}\sqrt{\bar V}\). Small neighborhoods have positive Gaussian probability. The uniformly bounded remainder cannot repair that error.

Therefore **fixed \(s,k,\Lambda\) cannot achieve \(\varepsilon\to0\)**. A fixed full \(C^k\) norm, as written in the coverage table, additionally includes the simpler output-amplitude obstruction.

Two useful corrected statements are:

- Allowing \(A_\varepsilon\) with \(\log A_\varepsilon=o(\ell)\) preserves the strengthened analytic lower bound \(s\ge(c(a,q)-o(1))\ell\). I do not have a matching upper bound under the fixed complex-radius ratio. Its coefficient should not be advertised as linear in \(n\): the displayed lower-bound coefficient itself grows roughly like \(q^{-a}\) for large \(a\), with \(q\) fixed.
- Allowing polylogarithmic derivative budgets gives a constructive
  \[
  s\le(k+1)p_f=O(kcn^2)
  \]
  upper bound for \(C^k\) decoders with measurable encoders.

For the latter, clip weights at \(T=O(\sqrt{\ell})\). Encode each normalized parameter using \(k+1\) interleaved Cantor-coordinate streams. Choosing the Cantor ratio sufficiently close to \(1/2\) makes the inverse coding Hölder of exponent \(>k\). Composing that inverse with the Lipschitz clipped target supplies Whitney \(C^k\) data with all positive-order jets zero. Whitney extension gives a decoder whose derivatives grow only polynomially in \(T\). Clipping controls the remaining error.

This proves an \(O(kcn^2)\) upper bound for the relaxed class, not \(O(kn)\). The next result rules out a width-uniform \(O(kn)\) bound when \(k\) is fixed.

**2(b). A genuine-ensemble chart of dimension \(n(n+1)/2\)**

**PROVEN, Astra round 2:**
\[
\boxed{\quad s_{\mathrm{Lip}}^*\ge\frac{n(n+1)}2\quad}
\]
under the bounded-encoding/high-probability encoding hypothesis and
\(\log(1+\rho\gamma)=o(\ell)\). Together with clipping,
\[
\frac{n(n+1)}2\le s_{\mathrm{Lip}}^*\le cn^2.
\]
This argument works for every \(c,d\ge1\).

Restrict \(W_2,\ldots,W_c\) to small neighborhoods of strictly positive invertible matrices. Then
\[
h_c=A\,\operatorname{ReLU}(Z),\qquad
Z\sim N(0,\Sigma),\qquad \Sigma=W_1W_1^\top .
\]
Condition on these later front matrices. The Wishart law of \(\Sigma\) has a positive density on every sufficiently small compact cube inside the positive-definite cone.

For fixed \(A\), consider the one-layer ridge means
\[
F_\Sigma(w)=E\,\operatorname{ReLU}
\bigl(w^\top A\operatorname{ReLU}(Z)\bigr).
\]
They recover every entry of \(\Sigma\) locally. Write \(\sigma_i=\sqrt{\Sigma_{ii}}\). At the following rows,
\[
F_\Sigma(A^{-\top}e_i)=\frac{\sigma_i}{\sqrt{2\pi}},
\]
and
\[
F_\Sigma(A^{-\top}(e_i-e_j))
=
\frac{\sigma_i-\sigma_j+
\sqrt{\Sigma_{ii}+\Sigma_{jj}-2\Sigma_{ij}}}
{2\sqrt{2\pi}}.
\]
The second identity follows from Gaussian sign symmetry and
\[
|u_+-v_+|+|(-u)_+-(-v)_+|=|u-v|.
\]

These \(n(n+1)/2\) quantities have an injective differential in \(\Sigma\). The map \(\Sigma\mapsto F_\Sigma\) is \(C^1\) into Gaussian \(L^2(w)\), by differentiating the Gaussian density. Its derivative functions are continuous in \(w\); therefore an \(L^2\)-zero derivative would vanish at all the displayed rows. The differential into \(L^2\) is injective.

On a sufficiently small covariance cube this gives
\[
c_0\|\Sigma-\Sigma'\|_F
\le \|F_\Sigma-F_{\Sigma'}\|_{L^2(w)}
\le C_0\|\Sigma-\Sigma'\|_F.
\]
The constants can be chosen uniformly for \(A\) in a compact neighborhood.

For \(d>1\), restrict the remaining back matrices to positive invertible neighborhoods. After the first back ReLU they act linearly, with uniformly positive smallest singular value. Hence the full \(H\)-distance dominates a constant times the one-layer distance.

Conditioning on \(A\) and integrating now gives the required small-ball estimate:
\[
P\{X(f)\in B_H(y,r),\ f\text{ in the selected event}\}
\le C r^{\,n(n+1)/2}.
\]
C4 proves the claimed lower bound. No claim of generic identifiability of the entire front is needed.

A further consequence is **PROVEN**:
\[
B_{\mathrm{one\text{-}way}}
\ge \frac{n(n+1)}2\log_2(1/\varepsilon)-O_{\mathrm{arch}}(1).
\]
A one-way \(B\)-bit code has at most \(2^B\) reconstruction functions in \(H\), so apply quantization to this chart. This improvement does not automatically extend to interaction.

**2(c). ARC’s exact-cumulant, one-back-layer closure**

**PROVEN, Astra round 2:** for the basic finite truncation in equations (13)–(14), fed exact cut cumulants, every fixed \(K\) has a positive ensemble bias floor on K5 ray fronts.

For odd \(K\), the paper also requires the additional \((K+1)\)-cumulant trace. Supply its exact value; “moments through \(K\)” alone does not completely specify that algorithm.

Here is a proof that avoids assuming convergence of an Edgeworth series.

The support of the K5 conditional front ensemble contains
\[
h_c=vA_t,\qquad
A_t=X_+ +t(-X)_+,\qquad X\sim N(0,1),\quad t>0,
\]
with \(v>0\). Construct the first two features as \(X_+,(-X)_+\), transmit them through intermediate layers, and use positive coefficients \(1,t\) at the bottleneck.

Its moments are
\[
E A_t^r=c_r(1+t^r),\qquad c_r=E X_+^r,
\]
and
\[
\mu_t=\frac{1+t}{\sqrt{2\pi}},\qquad
V_t=\frac{1+t^2}{2}-\frac{(1+t)^2}{2\pi}>0
\]
for **every real \(t\)**.

Fix a back row \(w\) with \(q=w^\top v>0\). Let \(T_K(t)\) be ARC’s truncated mean on these formal cumulants, and put
\[
D_K(t)=T_K(t)-q\mu_t.
\]
Every finite diagram expression is real analytic in \(t\) along the whole real axis: the cumulants are polynomial and \(V_t\) never vanishes.

For \(t>0\), \(q\mu_t\) is the true mean. If the closure were exact throughout any positive interval, the identity theorem would imply \(D_K\equiv0\). But at \(t=-1\), the formally continued variable is \(A_{-1}=X\). All cumulants above order two vanish, so, for \(K\ge2\),
\[
D_K(-1)=E(qX)_+=\frac q{\sqrt{2\pi}}>0.
\]
Contradiction. The negative-\(t\) continuation need not be a physical K5 front; only the finite closure formula is being continued.

Thus some physical \(t>0\) has nonzero bias. Continuity supplies neighborhoods of front and back weights with positive Gaussian probability. \(K=1\) follows directly from its positive Gaussian variance term.

For \(K=2\), the excess is explicit. On a scalar nonnegative ray amplitude with mean \(\mu>0\) and standard deviation \(\sigma>0\),
\[
\operatorname{bias}
=\sigma\varphi(\mu/\sigma)-\mu\Phi(-\mu/\sigma)>0.
\]

A weak quantitative rate is also available. Define \(b_K\) as the oracle RMS error, restricted to one fixed compact front/back neighborhood containing a positive interval of this family. Then
\[
\boxed{\quad b_K\ge c\exp[-C K\log(K+2)]\quad}
\]
with architecture-dependent positive constants.

The quantitative argument is:

1. Equations (13)–(14) use at most \(2K-1\) Hermite slots. On a common complex neighborhood of \([-1,2]\), avoid the two complex zeros of \(V_t\). Cumulants, Hermite coefficients and the number of diagrams give
   \[
   \sup |D_K|\le \exp[CK\log(K+2)].
   \]
   Factorial bounds suffice; no sharp diagram estimates are needed.
2. Apply the two-constants theorem on that neighborhood slit along a fixed positive interval \(J\). Its harmonic measure at \(-1\) is a fixed \(\theta>0\). Since \(D_K(-1)\) is a fixed positive constant,
   \[
   \sup_J|D_K|\ge c\exp[-CK\log(K+2)].
   \]
3. On the selected compact physical neighborhood, moment and closure Lipschitz constants have the same factorial bound, because all relevant variances stay bounded away from zero. A parameter ball of radius \(\exp[-CK\log(K+2)]\) therefore retains a comparable error. Gaussian density is bounded below there; integrating only changes \(C\).

Consequently, success of this **oracle one-step estimator** requires
\[
K\log(K+2)\gtrsim\log(1/\varepsilon),
\qquad
K\gtrsim\frac{\ell}{\log\ell}.
\]

Two limits matter:

- This does not prove convergence as \(K\to\infty\), an algebraic bias floor, or a lower bound for ARC’s full algorithm with previously **estimated** cumulants. Earlier errors can cancel final-step oracle bias.
- At fixed \(n\), the number of distinct symmetric cumulants is
  \[
  \sum_{r=1}^K\binom{n+r-1}{r}
  =\binom{n+K}{n}-1
  =\Theta_n(K^n).
  \]
  Thus the proved necessary symmetric state size is
  \[
  \Omega_{\mathrm{arch}}\!\left((\ell/\log\ell)^n\right).
  \]
  If an implementation literally stores dense \(n^K\) tensors, its corresponding bound is
  \(\exp[\Omega_{\mathrm{arch}}((\log n)\ell/\log\ell)]\).
  The paper’s fixed-\(K\), growing-\(n\) shorthand must not be used as the fixed-\(n\), growing-\(K\) asymptotic.

**3. Corrections to the coverage map**

For the **class table** in [coverage_map.md](/home/agent/workspaces/mats/work/cut-rank-ext/notes/coverage_map.md):

| Rows | Correction |
|---|---|
| L, P, P-multi | Correct under the stated degree, branching and cut hypotheses. Distinguish the communication variable \(C\) from the rank-bound constant also called \(C\). |
| An | Fixed amplitude implies an error floor for every \(s\). After relaxing amplitude subpolynomially, strengthen the necessary dimension to \(\Omega_{a,q}(\ell)\). |
| Ck | Distinguish a full \(C^k\) norm from B2’s top-derivative bound. Both preclude arbitrary precision at fixed \(s\) under their literal fixed budgets; retain \(s\ge ka\) as a necessary Jackson consequence. |
| Lip | Replace \(n\) by \(n(n+1)/2\). Keep the encoding-radius/exceptional-mass assumptions. Upper bound \(cn^2\) remains. |
| C0 | Exact scalar coding requires a compact source or an unbounded scalar domain, and permits a measurable encoder. |
| Bits | Keep \(n\) for arbitrary interaction; improve one-way to \(n(n+1)/2\). Upper bounds require the lower-order \(O_{\mathrm{arch}}(\log\log(1/\varepsilon))\) terms. |
| Mom | Write \(\mathrm{Err}\ge e_K^2\). Continuous readouts have the same infimum; attainment is separately established at sufficiently large finite order. |
| BF-real, BF-span | Correct with the stated vector count and nonadaptive linear-readout restrictions. |
| BF-bits | Correct for the complete, nonadaptive naming transcript. |
| Stream | Inherit all class assumptions. Replace the Lipschitz dimension by \(n(n+1)/2\); distinguish approximate continuous streaming from exact measurable scalar coding. |

For **every estimator row**, checked against [arc_estimators.md](/home/agent/workspaces/mats/work/cut-rank-ext/notes/arc_estimators.md):

| Estimator row | Corrected classification and consequence |
|---|---|
| Mean propagation | **Conflated algorithms.** CNX22 treats arithmetic circuits. Mean-only ReLU propagation from zero gives zero, but Wu’s \(K=1\) tracks a variance trace and produces nonzero means. C2 applies when the cut variance is deliberately set to zero, not when the actual \(K=1\) state is supplied. WhestBench implementation details were not verified in the fact sheet. |
| Covariance propagation | **Split the variants.** Wu’s leading-Hermite off-diagonal update differs from full Gaussian covariance propagation in the thesis. Wu’s \(K=2\) map extends continuously to valid PSD covariance states, although it is not uniformly Lipschitz in covariance coordinates. The zero-covariance log tests a different input law; it cannot exclude a ray-event bias proof with the true covariance. Section 2(c) supplies that oracle bias proof. |
| Basic/augmented cumulant propagation | Locally analytic where the required variances are positive; there is no uniform radius or derivative budget. Correctly outside the stated uniform P/An/Ck classes for ReLU. Add the basic oracle bias result, without transferring it automatically to full propagation or augmented variants. Polynomial-activation statements concern a different cut operator. |
| Factorized propagation | Same structural caveats; state depends on depth. Preserve the source’s distinction between the basic equivalence claim and the unresolved factorized-augmented code/paper discrepancy. Do not extrapolate fixed-\(K\) tensor counts to \(K\to\infty\). |
| Low-probability variant and baselines | Correct: different estimand. The mean theorem does not supply its error bound. |
| Harmonic deduction-projection | Correctly linear at each cut. Its own sketching analysis uses the **largest omitted eigenvalue** against an arbitrary head’s \(L^2\) norm; that is not identically our ensemble singular-value tail. A hypothetical ReLU implementation satisfying our linear-cut assumptions inherits K5. |
| Sketching theorem | Different normalization, so no conflict follows. Also retain “informal”: projecting onto \(m\) eigenfunctions immediately gives \(m\) real coefficients, not \(O(m)\) finite bits without amplitude/precision assumptions or side information. |
| Gelfand-pair/permanent deduction-projection | Correctly linear for their respective problems. Their own cut spectra are required; the ReLU exponent does not transfer. |
| Random-walk Fourier sketching | Instance-dependent sampled support is correctly identified. BF-bits applies only after all dependence on the tail is conveyed by a counted naming transcript. \(B\log(\#\text{characters})\) for one support is not automatically the complete communication cost. This is a different estimand. |
| GMHP | Fixed mixture cap, nonlinear updates and discontinuous pruning. For its actual RNN accuracy task, our mean bounds do not apply. “Only Bits” should mean a separately specified finite-precision communication implementation, not an automatic bound on its real state. |
| Region enumeration | Different estimand and growing case representation. Enumeration itself does not imply discontinuity of the resulting estimator. No direct application of the current ReLU-mean theorem. |
| Max-of-\(k\) bin propagation | Correctly a different trained-attention/discrete-input task; no present consequence. |
| ITGIS/MHIS, QLD, GLD | Different estimands and information access. Randomness alone does not put an estimator outside these theorems. Ordinary Monte Carlo can be represented by a nonlinear Lipschitz decoder of sampled cut activations; its growing state satisfies, rather than evades, the dimension bound. |
| MSP/explanations | Correctly outside a restricted cut-summary theorem. Replace “one real suffices in every communication model”: the answer-vector protocol or transmission of all weights works in the unrestricted model considered here. MSP’s advice must support its quantified query family; hard-coding one query’s answer is not equivalent. |
| Shin-tensor advice | Joint advice/computation proposal, without a direct current lower bound. Correct to mark unfinished. |

The map also omits substantive entries already in its fact sheet:

- E1’s Boolean independent-gate estimator, sparse covariance, and selected-term estimators.
- E2’s ablation, polynomial proof algorithm, and mechanistic-distillation application.
- E7’s exact Girard/region method.
- E9’s independent-feature/VAE activation modeling and three #SAT examples.
- E10’s pairing-structure estimators, regression merges, permanent sum estimates and activation-modeling proposal.
- E12c’s compression proposal and E14’s hybrid remarks.

These need separate rows or an explicit “other problems/proposals” grouping. E16 supplies no additional estimator. The fact sheet’s unopened sources prevent claiming an exhaustive list of all ARC proposals.

**4. Merged theorem list**

| Result | Status | Proposed by |
|---|---|---|
| Linear cut rank equals the singular-value-tail complexity; ray/genuine exponent \(a\) | **PROVEN**, assumed K1–K5 | Earlier cut-rank lane |
| Exact parameter summary; exact ray mean decoder; two-way \(p_b+n\) protocol | **PROVEN** | Both |
| One scalar, continuous encoder and decoder, arbitrary positive averaged accuracy with unrestricted stability | **PROVEN** | Astra R1 |
| Exact scalar coding with continuous decoder and measurable encoder, with the domain qualifications above | **PROVEN** | Claude C3 |
| Clipped weights: fixed dimension \(cn^2\), polylogarithmic Lipschitz constants | **PROVEN** | Astra R1 |
| Any-round bit rank bound; degree-bounded readout lifting | **PROVEN** | Both |
| Any-round bit lower bound with coefficient \(n\) | **PROVEN** | Astra R1 |
| Polynomial message protocols with fixed turns and no branching | **PROVEN UNDER** the all-message degree restriction | Astra R1 |
| Analytic/Jackson rank consequences | **PROVEN UNDER** their uniformity hypotheses; corrected and strengthened above | Claude; Astra R2 fixes |
| Literal fixed analytic amplitude and fixed-dimensional \(C^k\)-budget error floors | **PROVEN** | Astra R2 |
| Covering/quantization dimension obstruction; bi-Lipschitz ray map | **PROVEN** | Both; A9 by Astra |
| Genuine covariance chart, dimension \(n(n+1)/2\); stronger one-way bit bound | **PROVEN** | Astra R2 |
| Relaxed-budget \(C^k\) upper bound \(O(kcn^2)\) | **PROVEN** | Astra R2 |
| Fixed finite moments determine every fixed-architecture cut law; continuous exact decoder | **PROVEN**, non-effective order | Astra R1; Claude checked algebra, R2 rechecked continuity |
| \(c=1\): order two suffices; first moments alone have a floor | **PROVEN** | Astra R1 |
| Moment information limit \(e_K^2\); moment-restricted interaction cannot improve it | **PROVEN** | Astra R1 |
| Fixed feature-span interaction; frame naming and atlas-size consequences | **PROVEN UNDER** the specified readout/adaptivity restrictions | Claude |
| Streaming reduction | **PROVEN UNDER** inherited class hypotheses | Claude; R2 qualifications |
| ARC basic exact-cumulant one-step bias \(b_K\ge c e^{-CK\log(K+2)}\) | **PROVEN** for the paper’s finite truncation and correctly supplied oracle state | Astra R2 |
| Typical-network failure, sharp regularity widths, full-ARC precision rate, computational separation | **CONJECTURE / OPEN**, as specified next | Both |

**5. Merged conjecture ledger and bottleneck**

| Open step | Weakest useful statement and conditional consequence | Status, evidence, cheapest refutation |
|---|---|---|
| Full-front dimension beyond the covariance chart | Exhibit one additional positive-density chart of rank \(r>n(n+1)/2\). C4 then gives \(s_{\rm Lip}\ge r\). | **Open.** Claude I1 is false as an almost-everywhere statement: dead fronts have positive probability, and positive intermediate layers admit additional matrix-factorization symmetries. Test a specified active stratum’s Jacobian, then certify a nonzero minor. |
| Sharp \(C^k\) widths | After specifying subpolynomial derivative budgets, construct a decoder in dimension \(O(n^2+kn)\), or prove a larger lower bound. | **Open.** Literal I2 is false under fixed budgets; width-uniform \(\Theta(kn)\) also conflicts with the new \(n^2\) obstruction at fixed \(k\). Cheapest test: explicit smooth coding for a compact covariance family. |
| Analytic upper bound | Specify amplitude growth and fixed \(q\), then construct \(s\le C_{n,c,d,q}\ell\). | **Open.** Literal I3 has an amplitude floor. A coefficient uniform and linear in \(n\) is incompatible with the strengthened rank consequence for fixed \(q\). Cheapest refutation is already the binomial entropy calculation. |
| Effective finite moment order | Certify a proposed finite \(K_*\) separates all realizable cone-law types. This yields an effective information bound; conditioning/runtime still need separate analysis. | **Open**, eventual finiteness proved. Order two at \(c=1\) supports optimism; cone degeneracies oppose small, stable bounds. Cheapest refutation: a certified moment collision with different back means. |
| Strong ARC oracle rate | Prove \(b_K\ge cK^{-\beta}\), with \(\beta\) controlled independently of \(n\), or prove nonconvergence. The former gives symmetric state \(\Omega(\varepsilon^{-n/\beta})\). | **New conjecture; unproved.** We currently have only \(e^{-CK\log K}\). The paper itself cautions against fixed-width convergence. Cheapest test: high-precision scalar diagrams along \(A_t\), avoiding large tensors. |
| Transfer to the full ARC algorithm | Establish a quantitative full-estimator error floor directly, or a noncancellation comparison with the oracle error. | **Open.** Exact-cumulant input is not automatically the most favorable input for a biased closure. Cheapest refutation of a proposed comparison: compare full and oracle errors at small width while increasing \(K\). |
| Typical-network failure | A protocol-uniform error small-ball estimate with meaningful unconditional probability. | **Open.** Rare-event ensemble bounds do not supply it. Test residual distributions separately on and off the constructed events. |
| A separation from ARC’s general MSP | Specify allowed computation, precision, preprocessing/advice and query access; prove a cost lower bound in that model. | **Not yet a precise theorem target.** Parameter transmission and finite moment determinacy defeat an information-only argument. Tiny-architecture cone integration is a cheap test against proposed precision-cost exponents. |

**Single bottleneck:** connecting a restriction on the **actual closure computation** to its full estimation error. Information insufficiency is no longer available in general: finite moments eventually identify the cut law. The new oracle bias theorem makes that distinction concrete, but a stronger oracle rate and control of propagated-state cancellations are still needed to constrain ARC’s actual algorithm.
