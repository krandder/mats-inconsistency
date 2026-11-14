| Claim | Label | Reason |
|---|---|---|
| P, P1, P2, R | **PROVEN UNDER** integrability and measurability | The contraction and scale-mixture reduction work, including singular fronts. |
| R1 inequality | **PROVEN** | Small positive scales require the stated inverse moment; the atom needs separate treatment. |
| Super-polynomial decay ⇒ polylog rank | **WRONG** | Stretched-exponential decay suffices; super-polynomial decay alone gives weaker rates. |
| Gate identities and softplus asymptotic | **PROVEN** | The constant is correct, along nonzero even coefficients. |
| Ray kernel, variance and harmonic formulas | **PROVEN** | The implementation follows the correct spectral identities. |
| “Exact” numerical ray ranks | **WRONG** | Uncertified quadrature and discarded tails invalidate some fine-precision entries. |
| A(iii), power class | **PROVEN UNDER** regular variation | The stated exponent and width threshold are correct. |
| A(iii), stretched exponential | **PROVEN UNDER** \(0<\gamma<1\) | The saddle argument can be justified; larger \(\gamma\) needs correction. |
| Depth-radius recurrence | **PROVEN UNDER** nondegenerate positive-series assumptions | The ordered-phase conclusion drawn from it is false. |
| GLU shift and bilinear finiteness | **PROVEN UNDER** stated independence/moments | Radial averaging changes the coefficients before shifting. |
| Monotone fronts have densities | **PROVEN UNDER** invertible square layers and \(\phi'>0\) | This does not establish the required \(L^2\) condition. |
| Folds imply failure of the common-density theorem | **WRONG** | They obstruct ordinary Lebesgue \(L^2\), not every possible reference measure. |
| Practical soft-gate prediction | **HEURISTIC** | Ordering and a smaller variance budget do not determine relative cut rank. |
| Fuzzy-ray transfer | **PROVEN UNDER** an explicit operator-error bound | A useful bound follows from softplus–ReLU uniform approximation. |
| Training/summary “window” | **CONJECTURE** | Neither the upper bounds nor the experiments establish it. |

**1. Summary lemmas and unbounded scales.**  
For P, assume \(f\mapsto P_f\) is a measurable probability kernel, \(g_{b,j}(h)\) is jointly measurable, fronts and backs are independent, and
\[
\mathbb E_{b,j}\int |g_{b,j}|^2\,d\bar P<\infty.
\]
Then \(Au(f)=\int u\,dP_f\) is well-defined **almost everywhere in \(f\)** on \(L^2(\bar P)\) equivalence classes, and Jensen gives \(\|A\|\le1\). Individual \(P_f\ll\bar P\) is unnecessary.

P1 follows by factoring \(T=AR\); \(RR^*\) is the uncentered back-kernel operator. Use “trace-class kernel eigenvalues,” since Mercer continuity assumptions are unnecessary. Both ranks must use the same budget \(\varepsilon^2\bar V\).

P2 needs the selected polynomials in \(L^2(\bar P)\), for example \(\mathbb E\|h\|^{2D}<\infty\). Their dimension is **at most** \(\binom{n+D}{n}\).

R is correct: use one common scalar polynomial \(p\), average its squared error, then take the infimum. Conditional on \(h\), \(w\cdot h\sim N(0,g^2\|h\|^2/n)\).

R1’s density-ratio inequality is correct. However:

- Accumulation near zero can make \(\mathbb E[S_{\max}/S;S>0]=\infty\).
- Gaussian \(L^2\) error alone does not control the error at zero.
- The atom can be handled exactly by using polynomial features restricted to \(h\ne0\), plus \(1_{\{h=0\}}\), costing one additional dimension.

**WRONG:** super-polynomial Hermite decay alone implies polylogarithmic rank. For example, decay like \(\exp[-(\log k)^2]\) is super-polynomial but not stretched exponential. Under the bounded-scale transfer assumptions, super-polynomial tails imply \(r(\varepsilon)=o(\varepsilon^{-\eta})\) for every \(\eta>0\), not necessarily polylogarithmic growth.

Here is a **PROVEN UNDER** repair for unbounded scales, requiring no inverse-scale moment. Assume polynomial growth of \(\phi\), \(0<\bar V<\infty\),
\[
\Pr(S>A)\le C e^{-aA^2},\qquad
H_D(2A):=\sum_{k>D}(k+1)c_k(\phi;2A)^2
 \le C(1+A)^p e^{-cD^\gamma/A^q}
\]
uniformly for \(A\ge1\).

Use polynomial features multiplied by \(1_{\{S(h)\le A\}}\). Gaussian weighted Sobolev embedding gives, for the Hermite residual at scale \(2A\),
\[
\sup_{0\le s\le A}\mathbb E|(\phi-p_D)(sZ)|^2\le C H_D(2A).
\]
Outside the cutoff, polynomial growth and the scale tail bound control the error without evaluating the polynomial there. Choosing
\[
A\asymp\sqrt{\log(1/\varepsilon)},\quad
D=O\!\left(\log(1/\varepsilon)^{(1+q/2)/\gamma}\right)
\]
proves
\[
r(\varepsilon)=O\!\left(\log(1/\varepsilon)^{n(1+q/2)/\gamma}\right).
\]
This establishes summary rank, using truncated polynomial features. It does not establish the original global-polynomial approximation claim.

Gaussian inputs alone do **not** give Gaussian-tailed \(S\) after averaging random front weights: even a product of two scalar Gaussians has exponential-scale tails.

**2. Gate formulas.**  
**PROVEN:** the gsrelu coefficients, GELU identity, logistic representation \(L\overset d=2KZ\), and softplus/SiLU/tanh mixture identities are correct. The implemented Gaussian bump and CDF coefficients also match the probabilists’ normalization.

The softplus asymptotic can be upgraded from heuristic to **PROVEN**, for fixed \(\beta,s>0\), along even \(k\):
\[
\log c_k^2=-\frac{2\pi}{\beta s}\sqrt{k}+O(\log k).
\]
Indeed, the mixing density near zero has exponential factor
\(\exp[-\pi^2/(2\sigma^2)]\). The coefficient integral has leading exponent
\[
\frac{\pi^2}{2\sigma^2}+\frac{k\sigma^2}{2\beta^2s^2},
\]
whose minimum is \(\pi\sqrt{k}/(\beta s)\). Squaring doubles it. All integrands have the same coefficient sign, preventing cancellation. Odd coefficients above degree one vanish.

Hille’s normalization agrees: a probabilists’ strip width \(\tau\) becomes physicists’ width \(\tau/\sqrt2\), yielding coefficient decay \(e^{-\tau\sqrt{k}}\). Hille supplies bounds under growth assumptions; the mixture argument supplies the matching asymptotic. Softplus has **logarithmic branch singularities**, while its derivative has poles.

The same argument establishes root-exponential classes for SiLU and tanh on their nonzero parity subsequences. The finite smoothstep fits do not establish an asymptotic rate.

The post-activation-noise statement needs qualification: conditional on a finite-width row \(w\), the next noise standard deviation is \(\sigma\|w\|\), not exactly \(g\sigma\). The quoted degree estimates assume concentration and a specified variance normalization.

**3. Ray spectra and computation.**  
**PROVEN:** with the stated independent radial law and finite second moments,
\[
k_T(t)=\sum_k(\mathbb E_Rc_k)^2t^k,\qquad
\bar V=\sum_k\{\mathbb E_Rc_k^2-(\mathbb E_Rc_k)^2\}.
\]
The Gegenbauer shares in the code are correct. So are the multiplicities and sorting by \(E_\ell/N(n,\ell)\), including partial occupation of a degree.

**WRONG:** calling the resulting table exact. For softplus16, recomputing just \(b_{6002},\ldots,b_{6020}\) gives a sum about \(9.85\times10^{-12}\), whereas the reported \(\varepsilon=10^{-6}\) budget is about \(9.98\times10^{-13}\). The log reports zero missing energy. Thus the truncation diagnostic fails.

`max(moment − coefficient_sum, 0)` is not a certified tail bound when the moment itself comes from quadrature. Even with exact moments, it bounds omitted cut energy rather than computing it exactly. Radial quadrature also needs convergence control.

**PROVEN UNDER** \(b_k\sim Bk^{-\alpha}\), \(\alpha>1\), on the contributing parity, fixed \(n\ge3\):
\[
E_\ell\asymp\ell^{1-2\alpha},\qquad
r(\varepsilon)\asymp\varepsilon^{-(n-1)/(\alpha-1)}.
\]
A proof follows from Stirling’s formula for the exact shares:
\[
e_{k,\ell}\asymp
\ell^{n-2}k^{-(n-1)/2}e^{-\ell^2/(2k)}
\]
on the dominant scale \(k\asymp\ell^2\). Rescale \(k=\ell^2u\); bounds outside that range justify dominated convergence. Summing the degree tail and multiplicities gives the rank exponent. The stated ReLU/ReLU² width thresholds follow.

For \(\log b_k\sim-ck^\gamma\), \(0<\gamma<1\), the same estimates rigorously give
\[
\log E_\ell\sim-c'\ell^{2\gamma/(1+\gamma)}.
\]
At \(\gamma=1\), exponential decay in \(\ell\) remains correct, but the quadratic saddle gives the wrong constant. For \(\gamma>1\), the proposed saddle lies below \(k=\ell\); instead \(\log E_\ell\sim-c\ell^\gamma\).

Also, unbounded radial averaging changes decay classes. **HEURISTIC:** exponential \(R\) changes Gaussian-smoothing coefficient decay to a \(k^{1/3}\) exponent and softplus decay to a \(k^{1/4}\) exponent.

**4. Depth.**  
The radius recurrence is **PROVEN UNDER** a nonconstant positive-coefficient series with \(1<R_1<\infty\), using a generalized inverse when the target is not reached.

**WRONG:** \(\chi_1<1\) implies \(R_d=R_1\). Counterexample:
\[
f(t)=0.8+\frac{0.2t}{2-t},\quad f'(1)=0.4,\quad
R_1=2,\quad R_2=12/7.
\]
Here \(R_d\downarrow1.6\). The correct ordered-phase statement is that the radii remain uniformly above one. Keeping \(R_1\) requires the additional boundary condition \(f(R_1^-)\le R_1\).

For finite \(R_1>1\), the chaotic geometric contraction and critical
\[
R_d-1\sim\frac{2}{f''(1)d}
\]
are **PROVEN UNDER** the stated derivative conditions and nonlinearity. Entire series remain entire under every finite composition.

**WRONG:** “smoothness survives depth only in the ordered phase.” Every finite composition preserves \(C^\infty\) regularity in every phase. Moreover, root-exponential coefficient classes already have radius one. The phase analysis concerns uniform control as depth grows.

Radius alone does not certify the displayed finite-error degrees; tail prefactors matter. At finite width, smoothness of each realized composition persists, but universal \(O(L/n)\) corrections to radii or high-degree tails are **CONJECTURE**, not established.

**5. Multiplication and front densities.**  
The GLU identity is **PROVEN**, with the covariance factor:
\[
K_{\rm GLU}(h,h')=\frac{g_2^2}{n}(h\cdot h')K_\phi(h,h').
\]
On rays the shifted coefficients are \((\mathbb E_R[sR\,c_k(sR)])^2\), as implemented.

Bilinear depth-\(d\) backs are degree-\(2^d\) polynomials: exact finite rank is **PROVEN UNDER** the required moments. Their variance map \(q\mapsto Aq^2\) has derivative two at its nonzero fixed point. Extending that instability to every GLU requires checking its variance map. Normalization generally removes polynomial finiteness; LayerNorm’s singular set includes constant vectors, not only zero.

The diffeomorphism/density claim is **PROVEN UNDER** square invertible layers and everywhere-positive derivatives.

**WRONG:** strictly monotone gates automatically pass the \(L^2\)-density condition. In one dimension,
\[
\int q(y)^2dy=\int\frac{p(z)^2}{\phi'(z)}\,dz.
\]
For \(\phi=\operatorname{erf}\) and \(Z\sim N(0,s^2)\), this is finite only when \(s^2<1\). Softplus and tanh pass this test at each fixed Gaussian scale, but ensemble integrability is separate; even random scalar linear maps have divergent expected Lebesgue density norm near zero weight.

Regular GELU/SiLU folds have the stated inverse-square-root singularities, including locally in higher dimensions when a fold is reached with positive input density. But this proves failure only against Lebesgue or a locally regular reference. A common reference carrying the same singularity can change the conclusion.

Furthermore, the cited theorem uses **radially weighted angular densities for homogeneous backs**. Nonhomogeneous smooth backs require a full-state factorization. **PROVEN UNDER** a common measure making both density and back-function factors Hilbert–Schmidt, the resulting trace-class operator gives \(r=o(\varepsilon^{-2})\).

**6. Practical prediction.**  
The computed normalized softplus value \(\chi_1=0.4608\) is correct. **WRONG** as a generalization to soft gates: the same calculation gives SiLU \(1.0683\), GELU \(1.0618\), and tanh \(1.3443\).

Ordering can reduce within-network input variation. It does not determine how the cut spectrum shrinks relative to \(\bar V\). Therefore “softplus1 raises practical rank” remains **HEURISTIC**.

Section 4 also substitutes \(\operatorname{Var}\phi(sZ)\) for the actual \(\bar V\). Its numbers are not general-front certificates. An upper bound exceeding sampling cannot rule out a useful window at \(n\ge16\).

**7. New results and usable statements.**

**PROVEN: genuine-front perturbation lower bound.** Couple fixed-architecture ReLU and softplus networks using the same Gaussian weights. Since
\[
0\le\operatorname{softplus}_\beta(z)-z_+\le\log2/\beta,
\]
Lipschitz propagation and finite Gaussian weight moments give
\(\|T_\beta-T_{\rm ReLU}\|_{\rm HS}\le C_{n,L}/\beta=:\Delta\).
Best rank-\(r\) approximation error is Lipschitz in this norm, hence
\[
r_\beta(\varepsilon)\ge
r_{\rm ReLU}\!\left(
\frac{\varepsilon\sqrt{\bar V_\beta}+\Delta}
{\sqrt{\bar V_{\rm ReLU}}}\right).
\]
This transfers the genuine-front lower bound where the tolerance exceeds the perturbation scale. It proves nothing asymptotic as \(\varepsilon\to0\) with fixed \(\beta\). Exponentially small fuzzy-ray errors require margins; without them the uniform guarantee is \(O(\beta^{-1})\).

**PROVEN UNDER a quantitative approximation certificate:** a gate beats sampling in summary dimension whenever
\[
\mathbb E_{b,j}\operatorname{dist}(g_{b,j},S_D)^2
\le\varepsilon^2\bar V,\qquad
\dim S_D<\varepsilon^{-2}.
\]
There is no universal criterion from \(C^\infty,n,\beta s,\) and depth alone. Front law, derivative bounds, gains and \(\bar V\) matter. Summary dimension also does not bound computation time.

**PROVEN approximation result; training remains CONJECTURE:** replace each sorting comparator’s ReLU by softplus. A sorting network of comparator depth \(d_{\rm sort}\) then has uniform coordinate error at most
\[
d_{\rm sort}\log2/\beta
\]
with unchanged connectivity and comparable parameter count. Thus smooth gates can approximate kinked sorting targets arbitrarily well, at increasing sharpness. Exact equality across a kink is impossible for a smooth network; successful training at matched resources is unproved.

Checked at **2026-10-31 02:03 UTC** against the requested sources. All 38 activation checks passed; the independent omitted-tail calculation exposes the ray-table certification failure. No files changed.
tokens used
81,572
| Claim | Label | Reason |
|---|---|---|
| P, P1, P2, R | **PROVEN UNDER** integrability and measurability | The contraction and scale-mixture reduction work, including singular fronts. |
| R1 inequality | **PROVEN** | Small positive scales require the stated inverse moment; the atom needs separate treatment. |
| Super-polynomial decay ⇒ polylog rank | **WRONG** | Stretched-exponential decay suffices; super-polynomial decay alone gives weaker rates. |
| Gate identities and softplus asymptotic | **PROVEN** | The constant is correct, along nonzero even coefficients. |
| Ray kernel, variance and harmonic formulas | **PROVEN** | The implementation follows the correct spectral identities. |
| “Exact” numerical ray ranks | **WRONG** | Uncertified quadrature and discarded tails invalidate some fine-precision entries. |
| A(iii), power class | **PROVEN UNDER** regular variation | The stated exponent and width threshold are correct. |
| A(iii), stretched exponential | **PROVEN UNDER** \(0<\gamma<1\) | The saddle argument can be justified; larger \(\gamma\) needs correction. |
| Depth-radius recurrence | **PROVEN UNDER** nondegenerate positive-series assumptions | The ordered-phase conclusion drawn from it is false. |
| GLU shift and bilinear finiteness | **PROVEN UNDER** stated independence/moments | Radial averaging changes the coefficients before shifting. |
| Monotone fronts have densities | **PROVEN UNDER** invertible square layers and \(\phi'>0\) | This does not establish the required \(L^2\) condition. |
| Folds imply failure of the common-density theorem | **WRONG** | They obstruct ordinary Lebesgue \(L^2\), not every possible reference measure. |
| Practical soft-gate prediction | **HEURISTIC** | Ordering and a smaller variance budget do not determine relative cut rank. |
| Fuzzy-ray transfer | **PROVEN UNDER** an explicit operator-error bound | A useful bound follows from softplus–ReLU uniform approximation. |
| Training/summary “window” | **CONJECTURE** | Neither the upper bounds nor the experiments establish it. |

**1. Summary lemmas and unbounded scales.**  
For P, assume \(f\mapsto P_f\) is a measurable probability kernel, \(g_{b,j}(h)\) is jointly measurable, fronts and backs are independent, and
\[
\mathbb E_{b,j}\int |g_{b,j}|^2\,d\bar P<\infty.
\]
Then \(Au(f)=\int u\,dP_f\) is well-defined **almost everywhere in \(f\)** on \(L^2(\bar P)\) equivalence classes, and Jensen gives \(\|A\|\le1\). Individual \(P_f\ll\bar P\) is unnecessary.

P1 follows by factoring \(T=AR\); \(RR^*\) is the uncentered back-kernel operator. Use “trace-class kernel eigenvalues,” since Mercer continuity assumptions are unnecessary. Both ranks must use the same budget \(\varepsilon^2\bar V\).

P2 needs the selected polynomials in \(L^2(\bar P)\), for example \(\mathbb E\|h\|^{2D}<\infty\). Their dimension is **at most** \(\binom{n+D}{n}\).

R is correct: use one common scalar polynomial \(p\), average its squared error, then take the infimum. Conditional on \(h\), \(w\cdot h\sim N(0,g^2\|h\|^2/n)\).

R1’s density-ratio inequality is correct. However:

- Accumulation near zero can make \(\mathbb E[S_{\max}/S;S>0]=\infty\).
- Gaussian \(L^2\) error alone does not control the error at zero.
- The atom can be handled exactly by using polynomial features restricted to \(h\ne0\), plus \(1_{\{h=0\}}\), costing one additional dimension.

**WRONG:** super-polynomial Hermite decay alone implies polylogarithmic rank. For example, decay like \(\exp[-(\log k)^2]\) is super-polynomial but not stretched exponential. Under the bounded-scale transfer assumptions, super-polynomial tails imply \(r(\varepsilon)=o(\varepsilon^{-\eta})\) for every \(\eta>0\), not necessarily polylogarithmic growth.

Here is a **PROVEN UNDER** repair for unbounded scales, requiring no inverse-scale moment. Assume polynomial growth of \(\phi\), \(0<\bar V<\infty\),
\[
\Pr(S>A)\le C e^{-aA^2},\qquad
H_D(2A):=\sum_{k>D}(k+1)c_k(\phi;2A)^2
 \le C(1+A)^p e^{-cD^\gamma/A^q}
\]
uniformly for \(A\ge1\).

Use polynomial features multiplied by \(1_{\{S(h)\le A\}}\). Gaussian weighted Sobolev embedding gives, for the Hermite residual at scale \(2A\),
\[
\sup_{0\le s\le A}\mathbb E|(\phi-p_D)(sZ)|^2\le C H_D(2A).
\]
Outside the cutoff, polynomial growth and the scale tail bound control the error without evaluating the polynomial there. Choosing
\[
A\asymp\sqrt{\log(1/\varepsilon)},\quad
D=O\!\left(\log(1/\varepsilon)^{(1+q/2)/\gamma}\right)
\]
proves
\[
r(\varepsilon)=O\!\left(\log(1/\varepsilon)^{n(1+q/2)/\gamma}\right).
\]
This establishes summary rank, using truncated polynomial features. It does not establish the original global-polynomial approximation claim.

Gaussian inputs alone do **not** give Gaussian-tailed \(S\) after averaging random front weights: even a product of two scalar Gaussians has exponential-scale tails.

**2. Gate formulas.**  
**PROVEN:** the gsrelu coefficients, GELU identity, logistic representation \(L\overset d=2KZ\), and softplus/SiLU/tanh mixture identities are correct. The implemented Gaussian bump and CDF coefficients also match the probabilists’ normalization.

The softplus asymptotic can be upgraded from heuristic to **PROVEN**, for fixed \(\beta,s>0\), along even \(k\):
\[
\log c_k^2=-\frac{2\pi}{\beta s}\sqrt{k}+O(\log k).
\]
Indeed, the mixing density near zero has exponential factor
\(\exp[-\pi^2/(2\sigma^2)]\). The coefficient integral has leading exponent
\[
\frac{\pi^2}{2\sigma^2}+\frac{k\sigma^2}{2\beta^2s^2},
\]
whose minimum is \(\pi\sqrt{k}/(\beta s)\). Squaring doubles it. All integrands have the same coefficient sign, preventing cancellation. Odd coefficients above degree one vanish.

Hille’s normalization agrees: a probabilists’ strip width \(\tau\) becomes physicists’ width \(\tau/\sqrt2\), yielding coefficient decay \(e^{-\tau\sqrt{k}}\). Hille supplies bounds under growth assumptions; the mixture argument supplies the matching asymptotic. Softplus has **logarithmic branch singularities**, while its derivative has poles.

The same argument establishes root-exponential classes for SiLU and tanh on their nonzero parity subsequences. The finite smoothstep fits do not establish an asymptotic rate.

The post-activation-noise statement needs qualification: conditional on a finite-width row \(w\), the next noise standard deviation is \(\sigma\|w\|\), not exactly \(g\sigma\). The quoted degree estimates assume concentration and a specified variance normalization.

**3. Ray spectra and computation.**  
**PROVEN:** with the stated independent radial law and finite second moments,
\[
k_T(t)=\sum_k(\mathbb E_Rc_k)^2t^k,\qquad
\bar V=\sum_k\{\mathbb E_Rc_k^2-(\mathbb E_Rc_k)^2\}.
\]
The Gegenbauer shares in the code are correct. So are the multiplicities and sorting by \(E_\ell/N(n,\ell)\), including partial occupation of a degree.

**WRONG:** calling the resulting table exact. For softplus16, recomputing just \(b_{6002},\ldots,b_{6020}\) gives a sum about \(9.85\times10^{-12}\), whereas the reported \(\varepsilon=10^{-6}\) budget is about \(9.98\times10^{-13}\). The log reports zero missing energy. Thus the truncation diagnostic fails.

`max(moment − coefficient_sum, 0)` is not a certified tail bound when the moment itself comes from quadrature. Even with exact moments, it bounds omitted cut energy rather than computing it exactly. Radial quadrature also needs convergence control.

**PROVEN UNDER** \(b_k\sim Bk^{-\alpha}\), \(\alpha>1\), on the contributing parity, fixed \(n\ge3\):
\[
E_\ell\asymp\ell^{1-2\alpha},\qquad
r(\varepsilon)\asymp\varepsilon^{-(n-1)/(\alpha-1)}.
\]
A proof follows from Stirling’s formula for the exact shares:
\[
e_{k,\ell}\asymp
\ell^{n-2}k^{-(n-1)/2}e^{-\ell^2/(2k)}
\]
on the dominant scale \(k\asymp\ell^2\). Rescale \(k=\ell^2u\); bounds outside that range justify dominated convergence. Summing the degree tail and multiplicities gives the rank exponent. The stated ReLU/ReLU² width thresholds follow.

For \(\log b_k\sim-ck^\gamma\), \(0<\gamma<1\), the same estimates rigorously give
\[
\log E_\ell\sim-c'\ell^{2\gamma/(1+\gamma)}.
\]
At \(\gamma=1\), exponential decay in \(\ell\) remains correct, but the quadratic saddle gives the wrong constant. For \(\gamma>1\), the proposed saddle lies below \(k=\ell\); instead \(\log E_\ell\sim-c\ell^\gamma\).

Also, unbounded radial averaging changes decay classes. **HEURISTIC:** exponential \(R\) changes Gaussian-smoothing coefficient decay to a \(k^{1/3}\) exponent and softplus decay to a \(k^{1/4}\) exponent.

**4. Depth.**  
The radius recurrence is **PROVEN UNDER** a nonconstant positive-coefficient series with \(1<R_1<\infty\), using a generalized inverse when the target is not reached.

**WRONG:** \(\chi_1<1\) implies \(R_d=R_1\). Counterexample:
\[
f(t)=0.8+\frac{0.2t}{2-t},\quad f'(1)=0.4,\quad
R_1=2,\quad R_2=12/7.
\]
Here \(R_d\downarrow1.6\). The correct ordered-phase statement is that the radii remain uniformly above one. Keeping \(R_1\) requires the additional boundary condition \(f(R_1^-)\le R_1\).

For finite \(R_1>1\), the chaotic geometric contraction and critical
\[
R_d-1\sim\frac{2}{f''(1)d}
\]
are **PROVEN UNDER** the stated derivative conditions and nonlinearity. Entire series remain entire under every finite composition.

**WRONG:** “smoothness survives depth only in the ordered phase.” Every finite composition preserves \(C^\infty\) regularity in every phase. Moreover, root-exponential coefficient classes already have radius one. The phase analysis concerns uniform control as depth grows.

Radius alone does not certify the displayed finite-error degrees; tail prefactors matter. At finite width, smoothness of each realized composition persists, but universal \(O(L/n)\) corrections to radii or high-degree tails are **CONJECTURE**, not established.

**5. Multiplication and front densities.**  
The GLU identity is **PROVEN**, with the covariance factor:
\[
K_{\rm GLU}(h,h')=\frac{g_2^2}{n}(h\cdot h')K_\phi(h,h').
\]
On rays the shifted coefficients are \((\mathbb E_R[sR\,c_k(sR)])^2\), as implemented.

Bilinear depth-\(d\) backs are degree-\(2^d\) polynomials: exact finite rank is **PROVEN UNDER** the required moments. Their variance map \(q\mapsto Aq^2\) has derivative two at its nonzero fixed point. Extending that instability to every GLU requires checking its variance map. Normalization generally removes polynomial finiteness; LayerNorm’s singular set includes constant vectors, not only zero.

The diffeomorphism/density claim is **PROVEN UNDER** square invertible layers and everywhere-positive derivatives.

**WRONG:** strictly monotone gates automatically pass the \(L^2\)-density condition. In one dimension,
\[
\int q(y)^2dy=\int\frac{p(z)^2}{\phi'(z)}\,dz.
\]
For \(\phi=\operatorname{erf}\) and \(Z\sim N(0,s^2)\), this is finite only when \(s^2<1\). Softplus and tanh pass this test at each fixed Gaussian scale, but ensemble integrability is separate; even random scalar linear maps have divergent expected Lebesgue density norm near zero weight.

Regular GELU/SiLU folds have the stated inverse-square-root singularities, including locally in higher dimensions when a fold is reached with positive input density. But this proves failure only against Lebesgue or a locally regular reference. A common reference carrying the same singularity can change the conclusion.

Furthermore, the cited theorem uses **radially weighted angular densities for homogeneous backs**. Nonhomogeneous smooth backs require a full-state factorization. **PROVEN UNDER** a common measure making both density and back-function factors Hilbert–Schmidt, the resulting trace-class operator gives \(r=o(\varepsilon^{-2})\).

**6. Practical prediction.**  
The computed normalized softplus value \(\chi_1=0.4608\) is correct. **WRONG** as a generalization to soft gates: the same calculation gives SiLU \(1.0683\), GELU \(1.0618\), and tanh \(1.3443\).

Ordering can reduce within-network input variation. It does not determine how the cut spectrum shrinks relative to \(\bar V\). Therefore “softplus1 raises practical rank” remains **HEURISTIC**.

Section 4 also substitutes \(\operatorname{Var}\phi(sZ)\) for the actual \(\bar V\). Its numbers are not general-front certificates. An upper bound exceeding sampling cannot rule out a useful window at \(n\ge16\).

**7. New results and usable statements.**

**PROVEN: genuine-front perturbation lower bound.** Couple fixed-architecture ReLU and softplus networks using the same Gaussian weights. Since
\[
0\le\operatorname{softplus}_\beta(z)-z_+\le\log2/\beta,
\]
Lipschitz propagation and finite Gaussian weight moments give
\(\|T_\beta-T_{\rm ReLU}\|_{\rm HS}\le C_{n,L}/\beta=:\Delta\).
Best rank-\(r\) approximation error is Lipschitz in this norm, hence
\[
r_\beta(\varepsilon)\ge
r_{\rm ReLU}\!\left(
\frac{\varepsilon\sqrt{\bar V_\beta}+\Delta}
{\sqrt{\bar V_{\rm ReLU}}}\right).
\]
This transfers the genuine-front lower bound where the tolerance exceeds the perturbation scale. It proves nothing asymptotic as \(\varepsilon\to0\) with fixed \(\beta\). Exponentially small fuzzy-ray errors require margins; without them the uniform guarantee is \(O(\beta^{-1})\).

**PROVEN UNDER a quantitative approximation certificate:** a gate beats sampling in summary dimension whenever
\[
\mathbb E_{b,j}\operatorname{dist}(g_{b,j},S_D)^2
\le\varepsilon^2\bar V,\qquad
\dim S_D<\varepsilon^{-2}.
\]
There is no universal criterion from \(C^\infty,n,\beta s,\) and depth alone. Front law, derivative bounds, gains and \(\bar V\) matter. Summary dimension also does not bound computation time.

**PROVEN approximation result; training remains CONJECTURE:** replace each sorting comparator’s ReLU by softplus. A sorting network of comparator depth \(d_{\rm sort}\) then has uniform coordinate error at most
\[
d_{\rm sort}\log2/\beta
\]
with unchanged connectivity and comparable parameter count. Thus smooth gates can approximate kinked sorting targets arbitrarily well, at increasing sharpness. Exact equality across a kink is impossible for a smooth network; successful training at matched resources is unproved.

Checked at **2026-10-31 02:03 UTC** against the requested sources. All 38 activation checks passed; the independent omitted-tail calculation exposes the ray-table certification failure. No files changed.
