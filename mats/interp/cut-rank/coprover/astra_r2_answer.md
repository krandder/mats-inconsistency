codex
1. **Lemma C: the spectrum is correct; the rank and cutoff claims need correction.**

**PROVEN.** Use normalized spherical measure. Write \(b_\ell\) for the cap multiplier, reserving \(\lambda_\ell\) for back eigenvalues. For the toy’s Gaussian-ridge normalization,
\[
\kappa(1)=\tfrac12,\qquad
E_\ell=N(k,\ell)\mu_\ell,\qquad
b_\ell=\frac{I_{\ell+\nu}(s^{-2})}{I_\nu(s^{-2})},
\quad \nu=k/2-1.
\]
The normalized He kernel is twice this kernel.

If the front is the probability law \(Q_U\), with uniform centre \(U\), then
\[
\int Y_{\ell a}(v)\,Q_U(dv)=b_\ell Y_{\ell a}(U).
\]
Orthogonality under uniform \(U\) therefore gives
\[
A_{\ell a,mb}=b_\ell^2\delta_{\ell m}\delta_{ab}.
\]
Consequently, the eigenvalues and variance in Lemma C are exactly
\[
e_\ell=\mu_\ell b_\ell^2
\quad\text{with multiplicity }N(k,\ell),\qquad
\bar V=\frac12-\sum_\ell E_\ell b_\ell^2.
\]
This assumes unit radius and a normalized cap law. Direction-dependent radial weights require the modification in Section 2.

The exact one-layer formula avoids the quadrature instability:
\[
\mu_{2j}
=k\left[
\frac{\Gamma(k/2)\Gamma(j-\tfrac12)}
 {4\pi\Gamma(j+(k+1)/2)}
\right]^2,\quad j\ge1,
\]
with \(\mu_1=1/(4k)\), and \(\mu_\ell=0\) for odd \(\ell\ge3\).

**Numerical check.** Recomputing from these formulas gives the following \(\varepsilon=0.1\) ranks:

| \(s\backslash k\) | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| .03 | 22 | 217 | 1,622 | 46,479 | **713,593** | **46,983,832** |
| .1 | 11 | 39 | 112 | 462 | 1,527 | 7,114 |

The two corrected entries were also checked with high-precision positive-series Bessel arithmetic. The discrepancies come from quadrature; using \(\bar V=\tfrac12-\sum E_\ell b_\ell^2\) also avoids estimating the unsmoothed infinite tail by truncation.

There are two errors in the stated rank inequality: it counts null odd degrees, and it counts the entire boundary degree although only part may be needed. Put
\[
T_{\ge D}=\sum_{\ell\ge D}E_\ell b_\ell^2,\qquad
\delta=\varepsilon^2\bar V.
\]
For the even boundary degree \(D\ge2\) satisfying \(T_{>D}\le\delta<T_{\ge D}\),
\[
r(\varepsilon)
=N_{<D}+
\left\lceil\frac{T_{\ge D}-\delta}{e_D}\right\rceil,
\qquad
N_{<D}=k+\binom{k+D-3}{D-2}.
\]
Thus the exponential counting survives, but the displayed lower bound in Lemma C is not correct as written.

For \(D/k\to\alpha>0\), the corrected count gives
\[
\log r=kH(\alpha)+O(\log k),\qquad
H(\alpha)=(1+\alpha)\log(1+\alpha)-\alpha\log\alpha.
\]
Your \(k\log(1+D/k)\) captures a lower exponential rate; the full entropy includes the additional term \(D\log(1+k/D)\).

**HEURISTIC, with a controlled heat-kernel regime.** The appropriate cap approximation is
\[
b_\ell\approx
\exp\!\left[-\frac{s^2}{2}\ell(\ell+k-2)\right].
\]
Here \(s\) is a *per-tangent-direction* spread; total angular radius is approximately \(\sqrt{k-1}\,s\) when the cap is narrow.

The proposed \(D\sim\sqrt{2\log(1/\varepsilon)}/s\) is only a leading-log approximation in a restricted regime. It neglects both the algebraic back tail and the dependence of \(\bar V\) on \(s\). For fixed \(k\), as \(s\to0\),
\[
\bar V\sim \frac{k-1}{2}s^2,\qquad
E_\ell\sim C_k\ell^{-4}\quad(\ell\text{ even}),
\quad
C_k=\frac{k2^{k-1}\Gamma(k/2)^2}{\pi^2\Gamma(k-1)}.
\]
For \(D\gg k\), putting \(x=sD\), the heat approximation gives
\[
T_{\ge D}\approx\frac{C_k}{2}\int_D^\infty t^{-4}e^{-s^2t^2}\,dt.
\]
Hence
\[
D\asymp_k(\varepsilon^2s^2)^{-1/3}
\quad\text{when }s\ll\varepsilon^2,
\]
whereas in the strongly damped regime \(x\gg1\),
\[
x^2+5\log x
\approx
\log\frac{C_ks}{2(k-1)\varepsilon^2}.
\]
Only when the other logarithms are negligible does your proposed expression follow. At fixed \(s\), sufficiently extreme precision eventually leaves the Gaussian Bessel regime altogether.

**PROVEN connection to round 1.** For normalized He covariance, cap fronts with independent radius \(R\), \(\mathbb ER=1\), satisfy
\[
\bar V=\operatorname{Var}R+\sum_\ell a_\ell(1-b_\ell^2).
\]
If \(sk\) stays bounded, the shell \(k\le\ell\le2k\) retains \(\asymp k^{-3/2}\) energy. Therefore
\[
\varepsilon^2\bar V\le c\,k^{-3/2}
\quad\Longrightarrow\quad
r_{\rm full}\ge e^{(\log4-o(1))k}.
\]
Restricting the *centres* to an orthant gives the conservative rate \(e^{(\log2-o(1))k}\) by sign symmetrization. A vMF front still has support outside that orthant, so this is not yet a nonnegative-ReLU-front construction.

For \(\operatorname{Var}R=1\), the accuracy is the round-1 scale \(\varepsilon^2\lesssim k^{-3/2}\). For unit radius and \(s\asymp1/k\), \(\bar V\asymp1/k\), giving \(\varepsilon^2\lesssim k^{-1/2}\). These normalizations cannot be interchanged.

There is no exponential-in-\(k\) theorem at fixed \(s,\varepsilon\) as \(k\to\infty\): fixed-concentration vMF laws approach uniformity, and eventually rank at most one suffices.

2. **Synchronized fronts: a finite-width lower bound with explicit hypotheses.**

**PROVEN.** Normalize the back by
\[
F_b(u)=g_{b,j}(\sqrt n\,u),\qquad
\mathbb E_{b,j}F_b(u)F_b(v)=\kappa_{n,d}(u\cdot v),
\quad \kappa_{n,d}(1)=1.
\]
Let
\[
\kappa_{n,d}=\sum_\ell a_\ell P_\ell,\qquad
\lambda_\ell=\frac{a_\ell}{N(n,\ell)}.
\]

Assume exact synchronization:
\[
h_f(x)=\sqrt n\,R_f(x)U_f,\qquad R_f(x)\ge0.
\]
Set
\[
A_f=\mathbb E_xR_f,\quad
Q_f=\mathbb E_xR_f^2,\quad
\beta=\mathbb E_fA_f^2>0.
\]
Then, without any independence between \(A_f\) and \(U_f\),
\[
M(f,b,j)=A_fF_b(U_f),\qquad
\bar V=\mathbb E_fQ_f-\beta.
\]

The relevant directional law is the **squared-radial-mean-weighted law**
\[
\nu(B)=\frac{\mathbb E_f[A_f^2\mathbf1_{\{U_f\in B\}}]}{\beta}.
\]
Assume
\[
\frac{d\nu}{d\sigma_+}\le\rho,
\]
where \(\sigma_+\) is uniform probability measure on the positive orthant.

For a union \(B\) of complete harmonic degrees, define
\[
T_B=\sum_{\ell\in B}a_\ell,\qquad
\lambda_B^*=\sup_{\ell\in B}\lambda_\ell.
\]
Then
\[
\boxed{\displaystyle
r(\varepsilon)\ge
\frac{[\beta T_B-\varepsilon^2\bar V]_+}
     {\beta\,\rho\,2^n\,\lambda_B^*}.}
\]
Taking \(B=\{\ell\ge\ell_0\}\) gives the requested tail bound.

**Proof.** The block Gram matrix is
\[
H_B=\beta\Lambda_B^{1/2}
 \left(\int Y_B(u)Y_B(u)^T\,d\nu(u)\right)
 \Lambda_B^{1/2}.
\]
The addition theorem gives
\[
\operatorname{tr}H_B=\beta T_B
\]
for every directional law. For any coefficient vector \(z\),
\[
\int\left|\sum_{i\in B}\sqrt{\lambda_i}z_iY_i\right|^2d\nu
\le \rho\,2^n
\int_{S^{n-1}}\left|\sum_{i\in B}\sqrt{\lambda_i}z_iY_i\right|^2d\sigma
\le\rho\,2^n\lambda_B^*\|z\|^2.
\]
Apply the block lemma \(\mathrm{Err}_r\ge\operatorname{tr}H_B-r\|H_B\|\). Infinite tails follow by finite-block approximation.

Thus the orthant loss is \(2^n\) in the denominator. Equivalently, sign-transforming the first back matrix extends an orthant protocol to all orthants with at most \(2^n\) times its rank. Full-sphere density domination has no such factor.

An unweighted directional density bound suffices if
\(\mathbb E[A_f^2\mid U_f]=\beta\), in particular under independence. Otherwise it must be replaced by the weighted bound above. No lower density bound is needed.

**Important qualification for actual ReLU fronts.** Their direction law generally has mass on coordinate faces, so it has **no finite density relative to full-dimensional \(\sigma_+\)**. In the exactly synchronized model followed by one fresh Gaussian front layer:

- conditional on the active set \(S\), \(U_f\) is uniform on that face’s positive sphere;
- its direction is independent of the new radial norm;
- \(|S|\sim\mathrm{Binomial}(n,\tfrac12)\), before radial weighting.

Apply the theorem face by face, using harmonic dimension \(|S|\), and average the conditional error bounds. The back width remains \(n\). This avoids paying for the exponentially unlikely all-positive face.

For approximate synchronization, let \(\widetilde M\) denote the synchronized surrogate and
\(\eta=\|M-\widetilde M\|_{L^2}\). Replace the error budget by
\[
(\varepsilon\sqrt{\bar V}+\eta)^2.
\]
A sufficient certificate is
\[
\eta^2\le
\frac1n\mathbb E_{f,x}\|h_f(x)-\widetilde h_f(x)\|^2.
\]
This follows from Jensen and the back’s mean-square contraction. Typical angular convergence alone does not provide this energy-weighted certificate.

3. **Finite-width high-degree energy: an exact lower bound and a gate-jump identity.**

**PROVEN.** At every finite width and depth,
\[
\boxed{\displaystyle
a_\ell^{(d)}\ge2^{-(d-1)}a_\ell^{(1)}
\quad\text{for every }\ell.}
\]

For the final layer, write \(\operatorname{ReLU}(z)=(z+|z|)/2\). The mixed covariance terms vanish under \(W\mapsto-W\). Consequently,
\[
\kappa_{n,d}=\tfrac12\kappa_{n,d-1}+R_{n,d},
\]
where \(R_{n,d}\) is a rotation-invariant positive-semidefinite covariance kernel. Its harmonic coefficients are nonnegative, proving the claim inductively.

For fixed \(n\), the normalized one-layer energies satisfy
\[
a_\ell^{(1)}
\sim C_n^{\rm He}\ell^{-4}
\quad(\ell\text{ even}),\qquad
C_n^{\rm He}
=\frac{n2^n\Gamma(n/2)^2}{\pi^2\Gamma(n-1)}.
\]
Therefore
\[
\boxed{\displaystyle
\liminf_{m\to\infty}
m^3\sum_{\ell>m}a_\ell^{(d)}
\ge
\frac{2^{-(d-1)}C_n^{\rm He}}6.}
\]
This proves a polynomial high-frequency tail at every fixed finite width and depth. A coefficientwise claim covering all degrees must respect the one-layer odd-degree zeros.

There is also a stronger local fact.

**PROVEN gate-jump identity.** Fix a generic facet of first-layer gate \(i\), with normal \(w_i\), and let \(v\) be the first-layer activation on that facet. When \(v\ne0\),
\[
[J_g]=Dg_{\rm rest}(v)e_iw_i^T.
\]
The averaged gradient identity gives
\[
\boxed{\displaystyle
\mathbb E_{\rm rest}\|[J_g]\|_F^2=\|w_i\|^2.}
\]
Thus the expected squared first-layer Jacobian jump does not decay with downstream depth. When \(v=0\), the same result follows from homogeneity and
\(\mathbb E\|g_{\rm rest}(e_i)\|^2=1\).

This identity establishes persistent kinks. Extracting a degree-uniform lower bound from it additionally requires control of how different facets contribute and cancel in harmonic projections.

Combining the elementary spectral lower bound with the gradient upper bound already gives a useful theorem. Let \(k\) be the harmonic dimension, including a coordinate-face dimension. Then
\[
\sum_{\ell\ge k}a_\ell^{(d)}
\ge c\,2^{-(d-1)}k^{-3/2},
\]
while
\[
\sup_{\ell\ge k}\lambda_\ell^{(d)}
\le
\sup_{\ell\ge k}
\frac{k-1}{\ell(\ell+k-2)N(k,\ell)}
\le e^{-(\log4)k+O(\log k)}.
\]
Under Section 2’s weighted-density hypothesis,
\[
\varepsilon^2\bar V
\le c'\beta\,2^{-(d-1)}k^{-3/2}
\]
therefore implies
\[
\boxed{\displaystyle
r(\varepsilon)\ge
\exp\!\left[
(\log2)k-(d-1)\log2-\log\rho-O(\log k)
\right].}
\]
This is a general multilayer finite-width result, although its guaranteed accuracy deteriorates exponentially in depth.

For the exact synchronized-front model followed by a fresh Gaussian front layer, averaging faces with
\(|S|=n/2+O(\sqrt n)\) gives
\[
\log r\ge
\tfrac12(\log2)n-(d-1)\log2-O(\sqrt n+\log n)
\]
at an error budget of order \(\beta2^{-(d-1)}n^{-3/2}\).

**CONJECTURE needed for stronger accuracy.** The missing statement is a lower bound on energy in a specified degree block, uniform over the desired growing-depth regime and large enough relative to \(\bar V/\beta\). Persistent kinks alone do not establish that scale.

Depth-independent energy at fixed resolution cannot be inferred from criticality: in the kernel limit \(F^{\circ d}\to1\), so nonconstant energy tends to zero, even though the gradient identity remains valid at every finite depth.

4. **The attenuation objection depends on whether radius means total radius or per-direction spread.**

**HEURISTIC.** I agree with your conclusion if the stipulated multiplier really is
\(\exp(-\ell^2r_f^2/2)\). At \(\ell\asymp d^2\), \(r_f\asymp1/d\), it is exponentially small in \(d^2\).

But the kernel-limit estimate \(r_f\asymp1/c\) describes a **total angular radius**. For an isotropic narrow cap on \(S^{n-1}\), this corresponds to per-direction variance \(r_f^2/(n-1)\), giving
\[
b_\ell\approx
\exp\!\left[
-\frac{\ell(\ell+n-2)r_f^2}{2(n-1)}
\right].
\]
In my proposed regime
\[
c=d=n^{1/3},\qquad \ell\asymp d^2\ll n,
\]
the exponent is \(\asymp\ell/c^2\asymp1\). The cap model therefore predicts a constant attenuation, not disappearance of the block.

Actual front laws need not be isotropic caps. This calculation neither proves conjecture 7b nor refutes it. It does show that the stated radius argument does not invalidate that regime.

Front synchronization is a cleaner sufficient setting because it removes within-front angular smoothing. It is not the only possible setting.

**Single bottleneck, CONJECTURE.** For a specified regime of the original Gaussian fronts, establish a harmonic block \(B\) with
\[
\operatorname{tr}H_B\ge2\varepsilon^2\bar V,
\qquad
\frac{\operatorname{tr}H_B}{\|H_B\|}
\ge e^{a k^*}.
\]
Then \(r(\varepsilon)\ge\tfrac12e^{a k^*}\).

For synchronized fronts, Section 2 turns this into explicit weighted-direction and back-spectrum conditions. At the shrinking accuracy proved in Section 3, transferring the result to actual deep fronts requires sufficiently accurate synchronization in the energy-weighted \(L^2\) sense.

5. **Corrections and qualifications to round 1.**

- **The proved spectral, gradient, truncation, ray-front, and iterated-limit statements stand.** I found no contradiction with FORMAL.md.
- **Conjecture 7b remains unproved.** Its proposed degree block survives the isotropic-cap test when \(1/c\) means total angular radius. Neither growing-depth kernel convergence nor the actual front moment matrix was controlled.
- **Conjecture 7c needs a narrower scope.** Front synchronization alone supplies no collapse theorem. A long-back finite-width collapse claim requires a normalized approximation-error estimate; the iterated-limit result does not supply it.
- **The \(n^{D(\varepsilon)}\) picture is proved only as an upper bound.** Neither my argument nor Heuristic H establishes a matching lower bound or the proposed nugget interpretation.
- **Uniform orthant rays are a modified ensemble.** Applying that theorem to actual synchronized ReLU fronts requires radial weighting, coordinate-face conditioning, and a quantitative approximation bound.
- The measured \(n=8,c=d=4\) results support the reported finite-sample ranks. Their growing small-\(\varepsilon\) ranks do not establish saturation or an asymptotic exponential law.
