**PROVEN:** The mean-field \(2/3\) noise exponent survives the audit; the claimed all-depth concavity fails at depth 9. The B&B’s analytic bounds are valid, but its numerical intervals are not rigorous certificates.  
Checked 2026-10-31, 11:43 UTC: the frozen FORMAL hash matches provenance; selected existing tests pass. No files were modified.  
Below, “mean-field” means hidden widths tend to infinity before the subsequent depth limit. This does not establish the corresponding result at width \(n\), depth \(L=\alpha n\).

**1. Audit of Claude’s statements**

**1.1 Proposition 1: CORRECT AFTER FIX — PROVEN for fixed \(n,L,N\), with an explicit precision parameter and suitable distribution-transform primitives.**

The torus argument and Riemann–Lebesgue argument are correct. The construction should be called \(D_{N,k}\), with risk converging as \(k\to\infty\); the proof does not specify a single precision choice with a uniform cost guarantee.

Uniform integrability needs an argument about the dependent variables actually constructed. Here one is available:
\[
V_{i,c}=\operatorname{frac}\!\left(2^k(w_c+iw_{n+c})\right).
\]
Its underlying variable is Gaussian, and its wrapped density is uniformly bounded over \(k\ge0\), for fixed \(n,N\). Therefore \(\Phi^{-1}(V_{i,c})\) has uniformly bounded moments of every fixed order. Hölder’s inequality, Gaussian weight moments, and
\[
\|f_W(x)\|\le \Bigl(\prod_\ell\|W_\ell\|_{\rm op}\Bigr)\|x\|
\]
give a uniform \(L^{1+\eta}\) bound on the squared error. Weak convergence then implies convergence of risks.

Necessary fixes:

- **PROVEN:** Handle \(n=1\) separately; there are not \(2n\) first-layer entries. The two-direction exact formula does this.
- **PROVEN:** Charge precision generation and inverse-CDF evaluation, or explicitly grant them as primitives. Ordinary arithmetic plus floor does not automatically make them unit-cost.
- **PROVEN:** The finite-bit extension needs a precision schedule. Taking fractional parts at the quantization scale can produce identically zero seeds.
- **PROVEN:** The argument gives fixed-size convergence, not an effective uniform rate in \(n,L,N\).

Consequence 1 is therefore **CORRECT AFTER FIX**: determinism does not prevent matching sampling in the stated idealized average-case model.

**Chebyshev consequence 2: WRONG as written — PROVEN corrections.**

Gaussian weights are not confined to \([-1,1]\). Even on that interval, \(T_{2^k}(w)\) converges in distribution to the **arcsine law**, not the uniform law. One can first map a weight smoothly into \((-1,1)\), obtain the arcsine limit by oscillation, then apply its CDF. But this requires additional transforms and \(O(k)\) arithmetic operations; it does not establish the displayed arithmetic-only sampling construction.

The useful conclusion is narrower: continuity or unrestricted algebraic degree alone does not prevent extracting rapidly oscillating seeds.

**Consequence 3: CORRECT AFTER FIX — PROVEN only with an additional derandomization simulation.**

Matching MC deterministically does not itself prove that every superior randomized estimator can be simulated deterministically at comparable cost. Nonuniform seed fixing follows by averaging; a uniform, cost-preserving simulation requires another argument. Proposition 1 alone does not establish the claimed equivalence.

**1.2 Proposition 2: CORRECT — PROVEN.**

The kernel-energy identity, addition-theorem positivity, design cancellation, and expected \(S_\ell=1/N\) for iid spherical points are correct.

Its consequences need three qualifications:

- **PROVEN:** Rotation invariance equates a fixed rule with its random rotation. It does not equate arbitrary deterministic QMC with scrambled QMC, because scrambling generally changes pairwise geometry.
- **PROVEN:** The asserted limit \(S_\ell\to\sum_i c_i^2\) requires \(n\ge3\) and no antipodal pairs, after merging duplicate nodes. Antipodes contribute \((-1)^\ell\); on \(S^1\), the harmonics oscillate without the asserted pointwise decay.
- **PROVEN UNDER fixed dimension, two-sided spectral bounds, and the appropriate cubature theorem:** Eigenvalues
  \[
  \lambda_\ell=a_\ell/N(n,\ell)\asymp \ell^{-(n+2)}
  \]
  give **squared** integration error \(N^{-1-3/(n-1)}\). Parity degeneracies and depth-dependent constants must be handled. This fixed-\(n,L\) statement is not uniform at \(L=\alpha n\). Bietti–Bach and Sobolev-cubature references: exact applicable hypotheses not independently checked here.

The IBC remark is **CORRECT AFTER FIX — PROVEN UNDER a genuinely Gaussian function prior**, fixed evaluation budget, and a linear target functional. Conditional covariance depends only on locations, so adaptive locations cannot beat the best fixed locations. A final-ReLU output is not Gaussian, even when its preactivation has a GP limit.

**1.3 Sections 3.1–3.4**

- **3.1: WRONG as a “PROVEN” impossibility claim.** **PROVEN:** A suitable sampling-optimality theorem would imply a substantial quantitative time lower bound once input precision, accuracy and computation model are fixed. **HEURISTIC:** Existing methods appear inadequate. “Not provable unconditionally” and “only restricted statements are possible” are not established mathematical conclusions.
- **3.2: CORRECT AFTER FIX.** **PROVEN:** Worst-case CAPP reductions do not transfer without a distribution-preserving, accuracy-preserving reduction. Also, \(\mu(W)\) is not polynomial; the one-layer norm example suffices. **HEURISTIC:** Familiar low-degree interpolation reductions therefore look unavailable. Probability zero of an exact circuit encoding proves little by itself, since every prescribed continuous instance has probability zero; robust neighborhoods and other reductions remain possible.
- **3.3: CORRECT AFTER FIX.** **PROVEN:** No natural-proofs barrier has been established here. A uniform RAM lower bound does not automatically imply a Boolean circuit lower bound, and averaging over Gaussian weights is not Razborov–Rudich “largeness.”
- **3.4: CORRECT AFTER FIX.** **PROVEN:** For \(N\ge n\), blocking gives
  \[
  O\!\left(L\lceil N/n\rceil n^\omega\right).
  \]
  The formula \(O(LNn^{\omega-1})\) is not a general rectangular-multiplication bound for \(N<n\). The baseline correction remains essential. Attribution: Astra round 1, as disclosed in PROVENANCE.

**1.4 Sections 4.1–4.3**

- **4.1 identity and displayed bound: CORRECT — PROVEN**, interpreting vector covariance as its trace and taking the positive part of the lower bound. The exact optimum is \(\sum_{k>D}w_k\).
- **4.1 fixed point: CORRECT AFTER FIX — PROVEN for the mean-field map.** The stated constant is correct. \(O(\delta^{-1/3})\) layers reach fixed relative accuracy; accuracy \(\eta\) requires a further logarithmic factor in \(1/\eta\).
- **4.1 transfer to the mean: PROVEN in the mean-field model**, by integrating the cross-input kernel; no “collapsed direction” assumption is needed. See 3(a).
- **4.1 coefficient law: CORRECT AFTER FIX.** **PROVEN:** The limiting cumulative Hermite tail has exponent \(2/3\). A pointwise statement \(w_k\asymp k^{-5/3}\) requires additional regularity; it does not follow merely from noise stability. \(L^3\) is a crossover scale, not a hard cutoff.
- **4.2: CORRECT — PROVEN.** The triangle inequality is valid. An OGP conclusion would require an additional search representation and an overlap theorem.
- **4.3: CORRECT AFTER FIX — PROVEN.** A vector mean needs coordinate queries or a vector-valued oracle; standard SQ queries are bounded, so unbounded outputs require truncation. The essential objection remains: unrestricted expectation queries already answer the problem.

**1.5 Section 5: CORRECT AFTER SUBSTANTIAL FIX**

**PROVEN:** For a scalar output, \((\log Z)'(0)=\mu\). Relative-precision counting hardness does not automatically imply hardness at sampling-scale additive error. The cited examples should not all be characterized as “exponentially small quantities”: partition functions can be large, and deterministic volume-oracle lower bounds concern a different information model.

**PROVEN:** For the normalized ReLU covariance,
\[
\kappa''(q)=\frac1{\pi\sqrt{1-q^2}},\qquad
h(q)=(\kappa''(q))^{-1/2}=\sqrt\pi(1-q^2)^{1/4},
\]
and
\[
h''(q)=-\frac{\sqrt\pi(2+q^2)}
                   {4(1-q^2)^{7/4}}<0.
\]

**PROVEN:** The actual final-ReLU field is not a Gaussian mixed spherical spin glass. A Gaussian surrogate with the same covariance is a different random field. For one actual layer, maximizing a coordinate of \(\operatorname{ReLU}(Wx)\) on the sphere is simply maximizing one linear functional and applying ReLU.

**PROVEN UNDER the relevant spherical-spin-glass theorem’s hypotheses:** A transfer would require:

1. A centered Gaussian Hamiltonian with covariance \(n\xi(\langle x,y\rangle/n)\).
2. The correct mixture, field and normalization.
3. The theorem’s regularity and summability assumptions.
4. The applicable Parisi/no-gap condition and algorithm class.

The linear covariance term is a random external field and cannot be discarded because it disappears from \(\xi''\). In the usual zero-temperature variational formulation, a field can introduce a lower support endpoint \(q_0\), with matching condition
\[
h_{\rm ext}^2+\xi_{\ge2}'(q_0)=q_0\xi_{\ge2}''(q_0).
\]
Moreover, ReLU has \(\xi''(1)=\infty\), outside common smooth-mixture assumptions. Approximation and limit arguments must be supplied.

Subag and Huang–Sellke are relevant references; their exact applicability to this nonsmooth mixture with field is **unverified**. Concavity alone does not establish the claimed optimization or tail theorem.

**1.6 Sections 6.1–6.2**

**6.1: CORRECT AFTER FIX — PROVEN UNDER both a region-count bound and an integration-cost hypothesis.**

Finite cone decomposition is exact. Numerical integration “to error \(\varepsilon\)” is not exact integration in finitely many arithmetic operations. Fixed-dimensional polylogarithmic accuracy dependence may be available, but triangulation, conditioning, coefficient precision and dimension dependence must be charged. A region-count assumption alone does not establish the stated cost.

Also:

- **PROVEN:** \(ER\le(CnL)^{n-1}\) gives an upper bound \(\exp(O(n\log n))\) at \(L=\alpha n\), not a matching \(\Theta\) statement.
- **PROVEN:** Gate-history cells and maximal linear regions of the output are different objects.
- **HEURISTIC:** The proposed \(\varepsilon\) crossover suppresses integration costs, logarithms and variance normalization.

**6.2: WRONG exponent — PROVEN correction under the stated geometric assumptions.**

Write \(m=n-1\), \(K\asymp r^{-m}\). A crossing cell has function error \(O(r)\), not \(O(r^2)\); its integrated error is \(O(r/K)\). There are \(O(Kr)\) crossing cells.

Thus:

\[
\begin{array}{ll}
\text{coherent deterministic errors:}
 & |\mathrm{error}|=O(r^2),\quad \mathrm{MSE}=O(K^{-4/m});\\[2mm]
\text{centered, uncorrelated cell errors:}
 & \mathrm{MSE}=O(K^{-1-3/m}).
\end{array}
\]

The second statement is **PROVEN UNDER the decorrelation hypothesis**, which is **HEURISTIC** for this algorithm. The claimed exponent advantage \(5/(n-1)\) should be \(3/(n-1)\) under that hypothesis. Certificate widths do not benefit from cancellation.

**1.7 Section 7**

**PROVEN:** C0 remains a conjecture after repairing the baseline and model; C1 needs the finite-width qualification; C2 needs a precise certification requirement; C3 does not by itself bound all design gains; C4’s all-depth concavity assertion is false. The merged ledger below gives the replacements.

---

**2. Audit of `lib/bnb.py`**

**PROVEN: the homogeneous extension is correct.** On face \((k,s)\),
\[
x=t\,v(y),\quad t=sx_k>0.
\]
A slice bound \(a^\top y+b\) extends to
\[
a^\top x_{\rm free}+bsx_k.
\]
Multiplying inequalities by \(t>0\) preserves them.

**PROVEN: the symbolic relaxations are correct in exact arithmetic.** Although the coefficients of `Hr` need not be nonnegative, its evaluated affine function is nonnegative on the cell. Therefore
\[
WH_m\pm |W|H_r
\]
are valid propagated bounds. The upper chord and lower choices \(0\) or \(Z_{\rm lo}\) are valid. Certification of every gate makes the propagated map exactly linear.

**PROVEN: the analytic moment formulas are correct.** Conditioning on \(t=|X_k|\) gives the displayed products of Gaussian interval probabilities and truncated first moments. There is no missing Jacobian: this calculation uses the original Gaussian coordinates, not a direct change to \((t,y)\).

**PROVEN: the implementation has the following defects or limitations.**

1. **Missing quadrature and tail enclosure.** `cell_moments` uses finite-order quadrature on \([0,T]\), and neither quadrature error nor the omitted tail enters `lb`/`ub`. This invalidates certification even before floating-point rounding.

   A concrete read-only test used \(W_1=I_2\), `q=4`, and refined until every cell was marked exact. It returned
   \[
   \mathrm{lb}=\mathrm{ub}\approx(0.39226701,0.39226701),
   \]
   whereas
   \[
   \mu=(1/\sqrt{2\pi},1/\sqrt{2\pi})
      \approx(0.39894228,0.39894228).
   \]
   Thus “exact on certified cells” means **the linear-region model is exact**, not that the computed integral is exact.

   A repair needs outward-rounded moment intervals, quadrature remainders and Gaussian tails. For example,
   \[
   |\text{free-moment tail}|\le\phi(T),\qquad
   \text{radial-moment tail}\le\phi(T)
   \]
   give simple coefficient-weighted tail bounds.

2. **The half-width certificate applies to `mid`.** It does not automatically certify `point`. For a general reported estimate \(a\), use coordinate bounds
   \[
   |\mu_j-a_j|\le\max(|a_j-\mathrm{lb}_j|,\,
                     |a_j-\mathrm{ub}_j|).
   \]

3. **Two FLOP undercounts.**
   - `np.matmul(absW, Hr)` evaluates the whole batch whenever any cell is inexact, but charges only the inexact cells.
   - `np.where(...)` eagerly evaluates four `ndtr` arrays, while `moments_flops` charges two CDF evaluations.

4. **Width one crashes:** an `argmax` is taken over zero free coordinates.

**PROVEN by the executed checks:** the existing partition-mass, quadrature-comparison and pointwise symbolic-bound tests pass. Those tests support the formulas and implementation accuracy on their cases; they do not supply rigorous numerical enclosures.

---

**3. Strengthened statements**

**3(a). Noise sensitivity of the mean**

**PROVEN in the infinite-width mean-field model.** Normalize one output so that its annealed second moment on a unit input is one. Define
\[
K_{\rho,0}(t)=t,\qquad
K_{\rho,\ell+1}(t)=\kappa(\rho K_{\rho,\ell}(t)),
\]
and let \(T=\langle U,V\rangle\) for independent uniform sphere points. Then
\[
S_L(\rho)=m_n^2\left(E_TK_{\rho,L}(T)-\frac1\pi\right),
\qquad m_n=E\chi_n.
\]
This follows by writing the product of means as a double input integral. In particular, it concerns the mean, not the same-input output correlation.

Put
\[
a=\frac{2\sqrt2}{3\pi},\qquad A=a^{-2/3}.
\]
The expansion
\[
1-\kappa(1-z)=z-az^{3/2}+O(z^{5/2})
\]
gives, for \(\rho=1-\delta\),
\[
1-c_\rho^*\sim A\delta^{2/3}.
\]
The derivative at the fixed point is \(1-\Theta(\delta^{1/3})\). Comparison of the deficit recurrence first brings it into the fixed-point scale in \(O(\delta^{-1/3})\) steps, then contracts its relative error.

At \(\rho=1\), for every fixed \(t<1\),
\[
1-K_{1,L}(t)\sim \frac{9\pi^2}{2L^2}.
\]
Consequently, whenever \(\delta\downarrow0\) and \(L\delta^{1/3}\to\infty\),
\[
\boxed{
1-\operatorname{Corr}(\mu(W),\mu(W_\rho))
\sim
\frac{(3\pi/(2\sqrt2))^{2/3}}{1-1/\pi}\,
\delta^{2/3}.
}
\]
The constant is approximately \(3.27263\).

**PROVEN:** At fixed depth, the small-\(\delta\) law is linear. In the large-depth mean-field limit its normalized derivative is asymptotic to
\[
\frac{L}{4(1-1/\pi)}.
\]
Thus the transition occurs at \(\delta\asymp L^{-3}\).

**PROVEN: a stronger, two-scale Hermite bound holds at every finite width and depth.** Let
\[
F(\rho)=S(1)-S(\rho),\qquad T_D=\sum_{k>D}w_k.
\]
Then
\[
\boxed{
T_D\ge
\left[
(1+\rho^{-D})F(\rho)-\rho^{-D}F(\rho^2)
\right]_+.
}
\]
Indeed, the coefficient of \(w_k\) on the right before taking the positive part is
\[
(1-\rho^k)(1-\rho^{k-D}),
\]
which is nonpositive for \(k\le D\) and at most one otherwise.

Taking \(\rho=e^{-c/D}\), with sufficiently small fixed \(c>0\), turns a uniform \(F(\rho)\asymp(1-\rho)^{2/3}\) law into
\[
T_D\asymp S(1)D^{-2/3}.
\]
The matching upper bound follows from
\[
T_D\le \frac{F(\rho)}{1-\rho^{D+1}}.
\]

**PROVEN in the mean-field scaling window:** this yields \(T_D=\Theta(S_L(1)D^{-2/3})\) for \(1\ll D\ll L^3\). For the limiting normalized chaos-energy distribution, the Tauberian theorem gives the sharper cumulative asymptotic
\[
\sum_{k>D}p_k\sim
\frac{C}{\Gamma(1/3)}D^{-2/3}.
\]
It does not, without another argument, give individual coefficients \(p_k\sim c k^{-5/3}\).

**PROVEN:** “Worth \(D^{2/3}\) samples” requires a variance prefactor. Against radius-exact MC, the ceiling is
\[
N_{\rm equivalent}
\lesssim \frac{V_{\rm sph}}{S(1)}D^{2/3}.
\]
In the deep mean-field model, \(V_{\rm sph}/S(1)=\Theta(L^{-2})\). Thus degrees \(D\ll L^3\) do not even match one radius-exact sample in that limit.

**CONJECTURE:** The corresponding \(2/3\) window persists for actual width-\(n\) nets at \(L=\alpha n\). Accumulated finite-width fluctuations prevent identifying this with the theorem above.

**3(b). Kernel values, \(b_2\), and cheap unbiased measurements**

**PROVEN:** With \(k(t)=\sum_d b_dt^d\), \(b_d\ge0\), \(k(1)=1\),
\[
\boxed{
\sum_{\substack{d\ge4\\d\ \mathrm{even}}}b_d
=\frac{1+k(-1)}2-k(0)-b_2,\qquad
b_2=\frac{k''(0)}2.
}
\]

**PROVEN:** Degree-2 spherical-harmonic energy does **not** identify \(b_2\). If \(a_2\) is its Gegenbauer coefficient, then
\[
a_2=
\sum_{s\ge1}b_{2s}
\frac{(n-1)(n+2)s}{n+2s}\,\tau_{2s},
\]
where
\[
\tau_{2s}=\frac{(2s-1)!!}{n(n+2)\cdots(n+2s-2)}.
\]
In particular,
\[
a_2=\frac{n-1}{n}b_2+
\frac{6(n-1)}{n(n+4)}b_4+\cdots.
\]
Higher even power degrees contaminate the spherical degree-2 measurement. It nevertheless gives the valid bound \(b_2\le n a_2/(n-1)\).

**PROVEN: inexpensive unbiased estimators use first-layer Gaussian Hermite scores.**

Condition on downstream weights \(B=(W_2,\ldots,W_L)\), and let \(F_B(z)\) be the output after injecting first preactivations \(\sqrt{2/n}\,z\). Draw independent \(Z,Z'\sim N(0,I_n)\), sharing \(B\). Write
\[
Q_1=Z^\top Z',\qquad
Q_2=\tfrac12\bigl[(Z^\top Z')^2-\|Z\|^2-\|Z'\|^2+n\bigr].
\]

| Target | Unbiased observation |
|---|---|
| \(k(-1)\) | \(\langle F_B(Z),F_B(-Z)\rangle\) |
| \(k(0)\) | \(\langle F_B(Z),F_B(Z')\rangle\) |
| \(b_1\) | \(Q_1\langle F_B(Z),F_B(Z')\rangle\) |
| \(b_2\) | \(Q_2\langle F_B(Z),F_B(Z')\rangle\) |

These require two downstream evaluations per target, or **three shared evaluations for all four**, and \(O(n)\) additional arithmetic. In ordinary net samples, take inputs \(e_1,e_2,-e_1\) and standardize the corresponding first-layer columns.

**PROVEN:** Antithetic odd/even parts of \(F\) can remove irrelevant parity contributions, at four shared evaluations. These are cheap unbiased estimators, not asserted minimum-variance estimators; score variance and cancellation can still be substantial.

**PROVEN:** Combining the even-tail identity with (2.5), for
\[
N\le \frac{n(n+2)}6,
\]
gives
\[
R(A)\ge \frac{m_n^2}{2N}
\left[\frac{1+k(-1)}2-k(0)-b_2\right]
\]
for every positive fixed spherical rule. Compare that bracket with \(1-q\), where \(q=E_Tk(T)\). Estimate \(1-q\) directly using
\[
\tfrac12\|f_W(U)-f_W(V)\|^2
\]
to avoid subtracting nearly equal kernel values.

This tests the required tail for budgets up to order \(n^2\). Larger polynomial budgets require subtracting more low even coefficients; \(b_2\) alone cannot test every instance of M-design.

**PROVEN in the mean-field model: M-design has a stronger positive result.** For every fixed \(s_0\), with \(L=\alpha n\),
\[
\frac{\sum_{s\ge s_0}b_{2s}}{1-q}\longrightarrow\frac12.
\]
Sketch: \(1-\kappa^{\circ L}(0)\sim9\pi^2/(2L^2)\), every fixed positive-order Taylor coefficient is \(O(L^{-3})\), and
\(\kappa^{\circ L}(-1)=\kappa^{\circ(L-1)}(0)\). Sphere concentration gives \(1-q\sim9\pi^2/(2L^2)\). Therefore the even tail carries asymptotically half the angular variance. This remains unproved for width equal to input dimension.

**3(c). Iterating the final-row identity**

**PROVEN: an exact depth-recursive identity is available.** Let \(\mu_r\) be the mean after layer \(r\), \(d=L-r\), and
\[
B_{r,L}=W_L\cdots W_{r+1}.
\]
Then
\[
\boxed{
E_{W_{r+1:L}}\!\left[B_{r,L}^{\top}\mu_L
       \mid W_{1:r}\right]=\mu_r.
}
\]
One step follows from odd/even decomposition:
\[
E_{W_\ell}[W_\ell^\top\mu_\ell\mid W_{<\ell}]
=\mu_{\ell-1}.
\]
Iterate conditional expectation.

For any estimator \(A\), define
\[
\widehat\mu_r
=E_{W_{r+1:L}}[B_{r,L}^{\top}A\mid W_{1:r}].
\]
Since \(E\|B_{r,L}v\|^2=2^d\|v\|^2\),
\[
\boxed{
R(A)\ge 2^{-d}E\|\widehat\mu_r-\mu_r\|^2.
}
\]

Equivalently, projecting onto first Hermite degree in every suffix layer gives
\[
P_{1,r+1}\cdots P_{1,L}\mu_L
=2^{-d}B_{r,L}\mu_r.
\]

**PROVEN:** Combining this with (8.2), a risk-\(\varepsilon^2\) estimator must induce earlier-mean estimators with error at most \(2^d\varepsilon^2\), and hence
\[
\operatorname{Corr}(\widehat\mu_r,\mu_r)^2
\ge1-\frac{2^d\varepsilon^2}{\operatorname{Var}(\mu_r)}.
\]

**PROVEN limitation:** The factor \(2^{-d}\) is sharp for this extraction operator. For polynomially small error, the informative recursion typically reaches only \(O(\log n)\) layers backwards. Evaluating the conditional expectations defining \(\widehat\mu_r\) need not be cheap. Therefore this is not yet a lower bound on algorithms that spend little per layer.

**3(d). What an unrestricted estimator must accomplish**

**PROVEN:** Fix a computation model and let \(m(T)\) be the largest radius-exact MC batch executable at cost \(T\), including the best allowed batching implementation. Set
\[
B(T)=V_{\rm sph}/m(T).
\]
An estimator with growing gain \(G\) must satisfy \(R(A)\le B(T)/G\).

For actual finite-width nets at \(L=\alpha n\), round 1 proved \(S=\operatorname{Var}_W\mu=\Theta_\alpha(n)\). Consequently it must satisfy:

\[
\begin{aligned}
\|EA-E\mu\|^2&\le B(T)/G,\\
\operatorname{Corr}(A,\mu)^2&\ge1-\frac{B(T)}{GS},\\
\|A-A_\rho\|_2&\ge
 \|\mu-\mu_\rho\|_2-2\sqrt{B(T)/G},\\
E\|\widehat\mu_r-\mu_r\|^2&\le 2^{L-r}B(T)/G.
\end{aligned}
\]

**PROVEN:** For any genuinely restricted transcript \(\mathcal T\),
\[
R(A)\ge E\,\operatorname{tr}\operatorname{Cov}(\mu\mid\mathcal T).
\]
For full white-box access, \(\mathcal T=W\) makes this bound zero.

**PROVEN:** A control-variate method using \(m\) residual samples has gain
\[
G=\frac{m}{m(T)}\frac{V_{\rm sph}}{V_{\rm residual}},
\]
after charging construction, integration and evaluation. At comparable sample counts it needs residual variance \(o(V_{\rm sph})\).

**PROVEN limitation:** These conditions impose high accuracy, correlation and sensitivity. They do not imply that a successful algorithm must enumerate gates, propagate a particular summary, or have high polynomial degree. The missing implication from computational cost to one of those restrictions is the unrestricted bottleneck.

---

**3(e). Kelvin’s two analogies**

**3(e)(i). Threshold-circuit derandomization**

**PROVEN: the representational connection is real, but the random ensembles differ substantially.**

A ReLU network’s gate pattern is determined by thresholds, but intermediate values are real amplitudes. Standard \(\mathrm{TC}^0\) concerns constant-depth Boolean threshold circuits of polynomial size, with a specified finite encoding. Here \(L=\alpha n\) is growing depth; real-weight precision and Gaussian inputs add further differences.

Even the basic random dynamics differ. For centered sign gates the correlation map is
\[
c\mapsto \frac2\pi\arcsin c,
\]
whereas normalized ReLU uses \(\kappa(c)\). For \(0<c<1\), the former decreases correlation and the latter increases it. Similar gate syntax does not establish similar average-case behavior.

**PROVEN:** “Intersections of a few halfspaces are depth 1” is convention-dependent: in threshold-circuit language they are \(\mathrm{AND}\circ\mathrm{LTF}\), ordinarily depth 2. Polynomial threshold degree is also distinct from circuit depth.

**PROVEN UNDER the cited fooling guarantees:** Positive theory exists for halfspaces, bounded-degree PTFs and specified intersections of halfspaces, but parameters and restrictions matter. Relevant references include Meka–Zuckerman, Kane’s Gaussian-PTF generators, and Harsha–Klivans–Meka on polytopes; exact versions and quantitative guarantees are **unverified here**. These results do not imply a general polynomial-time, inverse-polynomial-error counting algorithm for arbitrary threshold circuits.

The precise shallow transfer is:

- **PROVEN:** One ReLU layer has the exact Gaussian mean
  \[
  E(w^\top X+b)_+
  =\|w\|\phi(b/\|w\|)+b\Phi(b/\|w\|).
  \]
  A linear combination of such units is equally easy.
- **PROVEN:** A second nonlinear layer with \(k\) first-layer gates can be split into at most \(2^k\) regions. Each output-threshold event on one region is an intersection of at most \(k+1\) halfspaces.
- **PROVEN:** Fooling each intersection to error \(\eta\) therefore gives the generic union bound \(2^k\eta\). This transfers usefully for few gates; it does not give a polynomial-cost result at \(k=n\).
- **PROVEN:** Transferring acceptance-probability error to an unbounded mean also needs truncation and coefficient/tail control. ARC’s SD-normalized tolerance can be much smaller than a fixed absolute tolerance.

**PROVEN, checked against the local paper:** IKW, *In Search of an Easy Witness* (2002), Theorem 32 concerns nontrivial approximation of **general CAPP**, with particular infinitely-often, nondeterministic and advice allowances. Its lower-bound implication is not automatically an implication for arbitrary average-case Gaussian-weight estimators. Williams’s algorithms-to-lower-bounds framework is related; the exact theorem needed for a proposed restricted threshold class must be supplied, rather than inferred from the name.

**HEURISTIC, literature assessment:** Average-case derandomization and random neural/threshold circuits are studied, but I have not verified a theorem matching iid He weights, \(L=\Theta(n)\), and this normalized mean-loss criterion. Goldreich–Wigderson’s typically-correct derandomization is relevant background, not such a theorem.

**CONJECTURE assessment:** A worst-case encoding analogy is plausible and already partly realized. A proof of average-case similarity would need an efficient reduction preserving the random-instance distribution, precision, loss normalization and cost advantage. That is substantially stronger, and currently unsupported here.

**3(e)(ii). Convex-body volume**

**PROVEN: there is an exact geometric representation.** For activation cones \(C_s\) and their linear maps \(B_s\),
\[
\mu(W)=\sum_s B_s\,E[X1_{C_s}],
\]
and
\[
P(f_{W,j}(X)>t)
=\sum_s\gamma_n(C_s\cap\{b_{s,j}^{\top}X>t\}).
\]
Each summand has explicit linear inequalities once the gate history is supplied.

**PROVEN:** This gives explicit Gaussian polyhedral moments, but potentially exponentially many dependent cones. Their union describes a piecewise-linear integrand, not generally one convex body whose volume equals the mean.

**PROVEN, standard volume results:** Dyer–Frieze–Kannan give randomized polynomial-time relative volume approximation under the convex-body access and geometric assumptions of their theorem. Elekes and Bárány–Füredi give deterministic oracle-information obstructions. Exact oracle models and quantitative bounds in the latter references are **unverified here**.

**PROVEN:** An oracle lower bound does not transfer to an explicit H-polytope. The inequality list can reveal structure hidden from membership queries.

**HEURISTIC, current-literature assessment:** General deterministic polynomial-time relative Gaussian-volume approximation for explicit rational H-polytopes remains open; I have not verified a theorem resolving it. There is no general unconditional white-box superpolynomial lower bound supplied by the cited oracle results. Exact polytope-volume hardness, such as Dyer–Frieze’s complexity results, concerns a different precision problem and does not establish the desired approximation lower bound.

**PROVEN:** A similarity theorem would need to convert hard explicit polytopes into these network-generated cones, preserve the relevant distribution for an average-case claim, and translate volume accuracy into SD-normalized mean accuracy. Showing merely that both involve Gaussian integrals is insufficient.

**PROVEN:** The project’s “about \(4\times\)” black-box result has the same information-model limitation. It hides independent signs in disjoint bumps; queries reveal only visited signs. Reading network weights can expose them directly. The local note also assigns different constants to Gaussian and bias-free variants, so \(4\) is not universal.

**CONJECTURE assessment:** The volume analogy is useful for designing conditional hardness reductions and integration algorithms. It currently supplies neither an iid-weight hardness reduction nor a white-box sampling-optimality theorem.

---

**4. Merged theorem list**

| Statement | Status | Proposed by |
|---|---|---|
| Weight-derived seeds asymptotically match MC, with precision and transform costs specified | **PROVEN UNDER those computational conventions** | Claude |
| Rotational-symmetry extraction gives an exact deterministic sampling construction | **PROVEN UNDER exact transforms** | Astra r1 |
| Fixed-rule risk equals spherical kernel discrepancy | **PROVEN** | Both |
| Positive-rule even-power tail bound (2.5); signed-rule moment-rank bound | **PROVEN** | Astra r1 |
| Degree-\(D\) optimum equals the weight-Hermite tail | **PROVEN** | Both; provenance qualification applies |
| Two-noise-scale lower bound for that tail | **PROVEN** | Astra r2 |
| Mean-field mean correlation has the \(2/3\) law; cumulative chaos tail has exponent \(2/3\) | **PROVEN in mean-field** | Claude’s hypothesis; Astra r2 proof |
| M-design even-tail fraction tends to \(1/2\) | **PROVEN in mean-field** | Astra r2 |
| Final-row extraction and depth-recursive extraction | **PROVEN** | Astra r1/r2 |
| Polynomially many first-layer gate cells cover exponentially little mass; discarding the rest gives order-\(n\) risk | **PROVEN for that restricted method** | Astra r1 |
| Cone extension, symbolic bounds and analytic cell-moment formulas | **PROVEN in exact arithmetic** | Claude; audited Astra r2 |
| Current numerical B&B intervals are rigorous certificates | **Not established; counterexample given** | Claude claim, corrected Astra r2 |
| Coherent partition error \(O(r^2)\); decorrelated MSE \(O(K^{-1-3/(n-1)})\) | **PROVEN UNDER respective mesh/error hypotheses** | Astra r1/r2; correction to Claude |
| Serial MC is an inadequate unrestricted arithmetic benchmark | **PROVEN** | Astra r1 |
| All-depth concavity of \((\kappa_L'')^{-1/2}\) | **Refuted** | Claude conjecture; Astra r2 refutation |

**PROVEN, computer-assisted refutation of the last row:** For \(f=\kappa^{\circ9}\),
\[
\left((f'')^{-1/2}\right)''(2/5)
=\frac{3(f''')^2-2f''f''''}{4(f'')^{5/2}}
\in(0.12159613867739,\;0.12159613867740).
\]
I enclosed the analytic derivative recurrence using outward-rounded integer fixed-point arithmetic at 65 decimal places, square-root bounds from integer square roots, and 220-term alternating arctangent series with remainder bounds; \(\pi\) was enclosed using Machin’s formula. This refutes global concavity, not by itself an optimization theorem’s conclusion or the presence/absence of an OGP.

---

**5. Merged conjecture ledger and bottleneck**

| Statement | Status and evidence | Cheapest useful refutation |
|---|---|---|
| **C0 / M-whole:** every uniform estimator has bounded gain over optimally implemented radius-exact MC in a specified polynomial budget window | **CONJECTURE.** Small-width failures of B&B and closures provide limited evidence about unrestricted algorithms | A charged estimator family with gain diverging with \(n\) |
| **C1:** finite-width mean noise sensitivity has a \(2/3\) window at \(L=\alpha n\) | **CONJECTURE; PROVEN in mean-field.** Pointwise \(w_k\sim k^{-5/3}\) is stronger than the established cumulative law | Unbiased two-scale stability measurements showing a different scaling window |
| **C3 / M-design:** a fixed high-even-power tail retains a constant fraction of angular variance | **CONJECTURE at finite width; PROVEN in mean-field** | Estimate the kernel-value/Hermite-score residual relative to \(1-q\) |
| **C2:** certified partition methods require exponentially many cells | **PROVEN for covering fixed mass with fully gate-resolved cells; CONJECTURE for general downstream-informed certificates** | Polynomial cell count with genuinely enclosed integration error at fixed normalized precision |
| Expected bias-free region count \(\le(CnL)^{n-1}\) | **CONJECTURE/extra hypothesis** | Verified activation-cell counts violating the proposed scaling; sampled distinct patterns give only lower bounds |
| **C4:** global concavity holds at every depth | **PROVEN false at depth 9** | Already refuted |
| **T-relative:** appropriately specified rare tails resist polynomial-time relative approximation | **CONJECTURE.** No distribution-preserving reduction supplied | Radius integration plus importance sampling with polynomial cost and bounded relative variance |

**PROVEN: three preregistration interpretations need correction before using results as theorem evidence.**

- Common-scramble squared mean differences include integration-noise variance. Independent replicate cross-products can estimate the desired squared difference without that bias; minimizing noisy stability estimates also needs simultaneous uncertainty control.
- Selecting “healthy” nets changes the iid-weight ensemble. Conclusions for that conditional population need to be stated separately.
- Random-rotation equivalence does not justify P7’s comparison between unscrambled and scrambled Sobol geometries.

**CONJECTURE, single bottleneck:** M-whole. The missing theorem would connect the arithmetic cost of an arbitrary white-box computation to an unavoidable residual error in \(\mu(W)\). Kernel tails, Hermite degree, gate-cell counts and recursive mean extraction each constrain a specified representation or observable; none currently supplies that computational bridge.
