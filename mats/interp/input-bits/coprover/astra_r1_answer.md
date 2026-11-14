codex
HEURISTIC — numerical conclusion: at 256 × 32, the variance budget is approximately 3.8% input radius, 59.4% input-dependent amplitude, and 36.8% output direction. Radius-exact sampling therefore buys only about 1.04×. The hypothesis that the original input radius carries most of the variance fails.

I did not read files in the prohibited directory. Computations below used single-threaded NumPy/SciPy: exact two-replica chains and eight fresh networks. Numerical observations are labelled HEURISTIC, not PROVEN. Checks completed on 2026-10-31 at approximately 18:20 UTC.

Q1. Variance budget

PROVEN — exact decomposition for each fixed noiseless network.

Write
\[
R=\|X\|,\qquad U=X/R,\qquad g(U)=A(U)d(U),
\]
with \(R\) independent of \(U\), and define
\[
m_n=E R=\sqrt2\,\frac{\Gamma((n+1)/2)}{\Gamma(n/2)},\qquad
v_n=n-m_n^2.
\]
Choose \(d\) arbitrarily when \(A=0\). All expectations in the following display are over \(U\), conditional on \(W\):
\[
\boxed{
V:=\operatorname{tr}\operatorname{Var}_X a
=
\underbrace{v_n E A^2}_{V_R}
+
\underbrace{m_n^2\operatorname{Var}A}_{V_A}
+
\underbrace{m_n^2\big((EA)^2-\|E[Ad]\|^2\big)}_{V_D}.
}
\]
All three terms are nonnegative. Equivalently, with independent \(U,U'\),
\[
V_A=\frac{m_n^2}{2}E(A-A')^2,\qquad
V_D=m_n^2E[AA'(1-d\cdot d')].
\]
No independence between amplitude and direction is assumed. Radius-exact MC removes exactly \(V_R\).

PROVEN — fixed-depth, large-width expansion.

Define
\[
\kappa(c)=\frac{\sqrt{1-c^2}+(\pi-\arccos c)c}{\pi},
\qquad c_j=\kappa^{\circ j}(0),\quad c_0=0.
\]
For correlated standard normals \(Z,Z'\), define the covariance of their squared ReLU gains:
\[
B(c)=4E[Z_+^2Z_+'^2]-1
=\frac2\pi\left[(1+2c^2)(\pi-\arccos c)
+3c\sqrt{1-c^2}\right]-1.
\]
Thus \(B(0)=0,\ B(1)=5\). Put
\[
C_L=\frac14\sum_{j=0}^{L-1}[5-B(c_j)].
\]
For fixed \(L\), as \(n\to\infty\),
\[
\boxed{
E_WV_R=v_n=\frac12-\frac1{8n}+O(n^{-2}),\quad
E_WV_A=C_L+O(n^{-1}),\quad
E_WV_D=n(1-c_L)+O(1).
}
\]
The amplitude formula comes from the covariance of the two per-layer norm gains. It is not obtained by treating the gains of different inputs as independent.

Numerical values of the exact iterated-kernel expressions are:

| \(L\) | \(C_L\), amplitude | \(256(1-c_L)\), direction |
|---:|---:|---:|
| 8 | 5.470 | 42.389 |
| 16 | 7.191 | 18.059 |
| 32 | 8.554 | 6.472 |
| 64 | 9.469 | 2.031 |
| 128 | 10.016 | 0.582 |
| 256 | 10.320 | 0.157 |

PROVEN — a useful uniform bound, including growing depth.

Let
\[
D_\ell=E_{W,U,U'}[A_\ell A_\ell'(1-\cos\theta_\ell)].
\]
Then
\[
D_\ell\le 1-c_\ell.
\]
To see why, write \(M_n(c)\) for the expected product of the two unit-input norm gains. Cauchy–Schwarz gives \(M_n(c)\le1\), and the exact cross-output expectation is \(\kappa(c)\). Jensen’s inequality, applied with the amplitude weights, gives the iterated bound.

Moreover, Gaussian Hermite expansion of
\[
z\mapsto \sqrt{2/n}\,\|z_+\|
\]
gives
\[
0\le1-M_n(c)\le \frac{2(1-c)}n.
\]
Consequently,
\[
0\le E_WV_A(\ell+1)-E_WV_A(\ell)
\le \frac{2m_n^2}{n}D_\ell.
\]
Since \(\sum_\ell(1-c_\ell)<\infty\), the amplitude increments have a uniformly summable tail.

Combining this bound with the fixed-depth expansion proves
\[
\boxed{
E_WV_A\longrightarrow C_\infty\approx10.6505
}
\]
whenever \(n\to\infty\) and \(L(n)\to\infty\). Also,
\[
E_WV_D\le m_n^2(1-c_L)=O(n/L^2).
\]
Thus, for \(L/\sqrt n\to\infty\),
\[
E_W(V_R,V_A,V_D)\longrightarrow(1/2,\;10.6505,\;0).
\]
These are annealed absolute variances, not averages of per-net fractions.

HEURISTIC — crossover depth.

Using
\[
1-c_L\sim\frac{9\pi^2}{2L^2}
\]
gives
\[
L_{\rm cross}\sim
\sqrt{\frac{9\pi^2}{2C_\infty}}\,\sqrt n
\approx2.042\,\sqrt n.
\]
The exact iterated-kernel equation \(n(1-c_L)=C_L\) predicts the crossing at \(L=28\) for \(n=256\). Finite-width corrections move it earlier.

The exact Gram-chain computation at \(n=256\) gave:

| Depth | \(E_WV_A\) | \(E_WV_D\) |
|---:|---:|---:|
| 24 | \(7.61\pm0.11\) | \(8.12\pm0.04\) |
| 25 | \(7.62\pm0.11\) | \(7.55\pm0.04\) |
| 26 | \(7.64\pm0.11\) | \(7.04\pm0.03\) |

Here and below, ± denotes one Monte Carlo standard error. The finite-width crossing is approximately depth 25. Refutation of the asymptotic crossover prediction: \(L_{\rm cross}/\sqrt n\) fails to approach 2.042 on a growing-width grid.

HEURISTIC — ARC measurement.

With 32,768 independent two-replica chains:
\[
E_WV_R=0.49951,\quad
E_WV_A=7.820\pm0.084,\quad
E_WV_D=4.837\pm0.019.
\]
Their sum is approximately \(13.156\). The resulting variance-weighted fractions are \(0.038,\ 0.594,\ 0.368\).

Eight independently generated networks, each evaluated on 16,384 inputs, gave mean terms \(0.483,\ 8.114,\ 4.947\). Amplitude dominated in seven of eight networks; direction dominated in the remaining one. This supports the ensemble prediction without making it a per-net guarantee.

PROVEN — the two requested limits differ.

- Fixed \(L\), growing \(n\): direction variance is \(\Theta(n)\); amplitude and radius remain \(O(1)\).
- Fixed \(L/n=\alpha>0\), growing \(n\): amplitude tends to 10.6505, radius to 0.5, and direction is \(O(n^{-1})\).
- In the latter limit, radius-exact MC has variance-weighted gain
  \[
  \frac{10.6505+0.5}{10.6505}\approx1.0469.
  \]

The exact coefficient of the \(n^{-1}\) direction term at \(L=\alpha n\) requires the finite-width chain. Substituting the infinite-width coefficient there is not justified by the fixed-depth expansion.

Q2. Input gate bits

PROVEN — exact disagreement identity.

Conditional on two nonzero states entering layer \(\ell\), with angle \(\theta_{\ell-1}\),
\[
E_{W_\ell}[\#\text{different gates}]
=\frac n\pi\theta_{\ell-1}.
\]
Therefore the exact annealed answer is
\[
\frac n\pi E\theta_{\ell-1},
\]
computed using the finite-width Gram chain. If exactly one state is zero, its contribution is \(n/2\); if both are zero, it is zero.

PROVEN — fixed-depth leading formula.
\[
E[\#\text{different gates at }\ell]
\sim \frac n\pi\arccos(c_{\ell-1}).
\]
In the kernel regime \(1\ll\ell\ll n\),
\[
\arccos(c_{\ell-1})\sim\frac{3\pi}{\ell},
\qquad
E[\#\text{different gates}]\sim\frac{3n}{\ell}.
\]

PROVEN — marginal entropy requires more than two replicas.

For a fixed network, let
\[
p_i=P_U(\text{gate }i\text{ is on}),\qquad H_\ell=\sum_i h_2(p_i).
\]
Two-input disagreement determines \(\sum_i2p_i(1-p_i)\), not \(\sum_i h_2(p_i)\).

For example,
\[
H_\ell\ge2\,E_{U,U'}[\#\text{different gates}\mid W].
\]
Different distributions of the \(p_i\) can have the same disagreement count and different entropy. The entropy sum is also only an upper bound on joint gate entropy.

HEURISTIC — Gaussian cloud closure for entropy.

Approximate a row’s preactivation over inputs by a Gaussian common mean plus Gaussian input fluctuation. If the replica correlation entering the layer is \(c\), this predicts
\[
H_\ell\approx n\,\mathcal H(c),\qquad
\mathcal H(c)=E_Zh_2\!\left(
\Phi\!\left(\sqrt{\frac c{1-c}}Z\right)\right).
\]
For small angle \(\theta=\arccos c\),
\[
\mathcal H(c)\sim0.73516\,\theta.
\]
Thus the kernel regime gives approximately
\[
H_\ell\sim6.929\,\frac n\ell.
\]

At layer 31, \(n=256\):

- Exact iterated-kernel disagreement prediction: 19.33 gates.
- Finite-width Gram-chain estimate: \(18.50\pm0.03\) gates.
- Kernel Gaussian-cloud entropy prediction: 44.48 bits.
- Eight fresh fixed nets: mean 39.47 bits, range 28.58–57.02 bits.

The reported 38 bits is entirely plausible. Refutation of the closure: the distribution of fixed-row activation probabilities, or its entropy integral, systematically disagrees across many nets after supplying the measured cloud geometry.

PROVEN — the per-layer amplitude increment is a gain-covariance quantity:
\[
\Delta C_\ell=\frac{5-B(c_{\ell-1})}{4}.
\]
At small angle,
\[
\Delta C_\ell\sim2(1-c_{\ell-1})
\sim\theta_{\ell-1}^2
\sim\frac{9\pi^2}{\ell^2}.
\]
Hence gate disagreements decay approximately as \(n/\ell\), while new amplitude variance decays as \(\ell^{-2}\).

This is not a “variance per gate bit” law. Correlated norm gains fluctuate even when the two paths share their gate pattern. Earlier nonlinear processing can remain encoded in a scalar amplitude after most later gates stop varying.

HEURISTIC — algebraic versus exponential.

Finite-angle contraction governs independent inputs through the kernel regime. At depths comparable to or exceeding \(n\), typical infinitesimal angular differences can instead follow finite-width Lyapunov gaps. Annealed gate counts cannot be assigned the typical Lyapunov exponent: averaging multiplicative fluctuations changes decay rates.

Refutation: compare median log-angle decay, mean angle decay, and the exact Gram-chain prediction separately. Agreement of only one does not establish the others.

Q3. Interaction structure over Gaussian input coordinates

PROVEN — Q1 does not give three canonical ANOVA components.

The three terms in Q1 form a nonnegative variance identity. They are not three orthogonal functions of the Cartesian coordinates \(X_1,\ldots,X_n\). Consequently, an exact, unique “Sobol index of the direction term” does not follow from that identity.

A useful diagnostic is to compare the ANOVA of \(a(X)\) with that of \(\|a(X)\|\). Their total variances differ by exactly \(V_D\), but differences between individual ANOVA energies can be signed.

PROVEN — exact finite-width replica formulas.

Let \(X^{[s]}\) share exactly \(s\) coordinates with \(X\), with all remaining coordinates independent, and define
\[
F_s=E_{W,X,X^{[s]}}[a(X)\cdot a(X^{[s]})].
\]
Then
\[
\bar V=F_n-F_0,
\]
\[
\boxed{
S_1^{\rm ann}=\frac{n(F_1-F_0)}{\bar V},\qquad
\bar d^{\rm ann}=\frac{n(F_n-F_{n-1})}{\bar V}.
}
\]
These are variance-weighted ensemble quantities:
\[
S_1^{\rm ann}=\frac{E_W[V_WS_{1,W}]}{E_WV_W},
\]
and similarly for mean dimension. They are not \(E_WS_{1,W}\).

The two-replica Gram chain evaluates these expressions exactly in distribution: initialize it with the Gram matrix of the appropriate partially shared Gaussian inputs. Replacing the terminal dot product by the product of terminal norms gives the corresponding scalar-norm quantities.

PROVEN — fixed-depth, large-width limits.

Let \(K_L(t)=\kappa^{\circ L}(t)\). For fixed \(L\),
\[
\boxed{
S_1^{\rm ann}\longrightarrow
\frac{K_L'(0)}{1-K_L(0)},\qquad
\bar d^{\rm ann}\longrightarrow
\frac{1}{1-K_L(0)}.
}
\]
Here
\[
K_L'(0)=\prod_{j=0}^{L-1}
\left(1-\frac{\arccos c_j}{\pi}\right).
\]

Examples:

| Fixed depth | Limiting \(S_1\) | Limiting mean dimension |
|---:|---:|---:|
| 8 | 0.3021 | 6.04 |
| 16 | 0.1888 | 14.18 |
| 32 | 0.1098 | 39.56 |
| 64 | 0.0604 | 126.04 |

These limits are not accurate substitutes for \(n=256,L=32\), where amplitude variance matters.

PROVEN — Hermite degree and ANOVA dimension differ.

Let \(H_k(W)\) be the variance in total Gaussian Hermite degree \(k\). Then
\[
\sum_{k\ge1}kH_k(W)=E_X\|J_W(X)\|_F^2.
\]
Critical Gaussian weights preserve the expected squared tangent norm at every layer, so
\[
\boxed{E_W\sum_k kH_k(W)=n.}
\]
Thus the annealed mean Hermite degree is exactly \(n/\bar V\). ANOVA dimension counts distinct coordinates, not total polynomial degree, and can be much smaller.

At ARC, the variance estimate gives mean Hermite degree approximately 19.46.

HEURISTIC — amplitude spectrum and the finite-width linear component.

Extend the amplitude coefficient to arbitrary initial correlation:
\[
C_L(t)=\frac14\sum_{j=0}^{L-1}
[5-B(K_j(t))].
\]
The leading amplitude covariance kernel is \(C_L(0)-C_L(t)\). Its linear coefficient is
\[
a_{1,L}=-C_L'(0)
=2\sum_{j=0}^{L-1}c_{j+1}K_j'(0).
\]
Numerically,
\[
a_{1,32}=2.9817,\qquad a_{1,\infty}=3.09504.
\]
The quadratic coefficient tends to approximately 2.21579.

This predicts an amplitude first-order share approaching
\[
3.09504/10.6505\approx0.2906
\]
in ordinary, unrotated coordinates.

The linear component has a concrete origin. For the first-layer scalar norm, its leading linear coefficient vector is
\[
b\approx\frac1{\sqrt{2\pi n}}\sum_i\|w_i\|w_i,
\]
whose expected squared norm tends to \(2/\pi\). A fixed random network has a preferred input direction even though the ensemble is rotationally invariant. Subsequent layers accumulate further amplitude sensitivity.

Refutation: measure the linear Hermite energy of the scalar norm across widths and depths; failure of convergence to the series above rejects this amplitude-spectrum approximation.

HEURISTIC — component-level dimension predictions.

For the original radius alone,
\[
R-\sqrt n=
\frac1{2\sqrt n}\sum_i(X_i^2-1)+o_{L^2}(1),
\]
so its variance is asymptotically additive: \(S_1\to1,\ \bar d\to1\).

For amplitude, the spectral approximation predicts
\[
\bar d_A\approx\frac{2L}{C_L}
\quad\text{when }L\ll\sqrt n.
\]
For \(L\gg\sqrt n\), the near-diagonal amplitude kernel satisfies
\[
C_\infty(\cos\theta)\sim3\pi\theta.
\]
Refreshing one Gaussian coordinate creates
\(\theta\approx |Z-Z'|/\sqrt n\), leading to
\[
\bar d_A\sim
\frac{6\sqrt\pi}{C_\infty}\sqrt n
\approx0.999\,\sqrt n.
\]

At \(L=\alpha n\), this predicts for the complete Gaussian-input map
\[
\boxed{
S_1\to\frac{3.09504+0.5}{10.6505+0.5}
\approx0.3224,\qquad
\bar d\sim0.954\,\sqrt n.
}
\]
Radius-exact sampling removes the 0.5 contribution and predicts \(S_1\to0.2906\).

Thus mean interaction order can diverge while a substantial additive fraction survives. These growing-depth ANOVA limits remain heuristic; the uniform variance bound in Q1 does not itself prove them.

Refutation: exact pick-freeze estimates at \(L/n\) fixed fail to approach these constants or the \(\sqrt n\) scaling.

HEURISTIC — ARC checks.

The exact two-replica experiment gave
\[
n(F_n-F_{n-1})=129.67\pm0.85,
\]
hence
\[
\bar d^{\rm ann}\approx9.86.
\]
The scalar norm had mean dimension approximately 4.75.

For eight fresh networks, independent-half estimates of additive Hermite energy through degree four gave:

- Mean per-net share: 0.274.
- Range: 0.175–0.337.
- Ratio of summed estimated additive energy to summed variance: 0.287.

Degrees three and four contributed approximately zero within sampling error. Higher univariate degrees were not measured, so this is a truncated estimate of \(S_1\).

PROVEN — rotations.

A data-independent orthogonal rotation cannot improve ensemble-average performance through network alignment: the weight distribution is rotationally invariant.

A \(W\)-dependent rotation can change ANOVA additivity. However, linear Hermite energy is already additive in every orthogonal coordinate system; aligning its coefficient changes truncation dimension, not its total \(S_1\).

For a scalar quadratic component \(x^\top Hx-\operatorname{tr}H\), diagonalizing \(H\) makes the entire component additive. Multiple output quadratic forms generally cannot be simultaneously diagonalized.

HEURISTIC — useful rotation.

When deep outputs nearly share one direction, estimate the quadratic component of their scalar amplitude and diagonalize it. This could move much of the amplitude’s degree-two energy into \(S_1\), giving a constant-factor improvement.

Refutation: after charging the pilot and rotation, the held-out additive share or estimator variance fails to improve. I see no width-growing gain from a fixed-degree rotation alone.

Q4. Age and Lyapunov directions

PROVEN — exact infinitesimal statement.

Along a reference input,
\[
J_\ell=D_\ell W_\ell,\qquad
P_{\ell:1}=J_\ell\cdots J_1.
\]
For an isotropic infinitesimal perturbation with covariance \(\varepsilon^2I\),
\[
C_\ell^{\rm tangent}
=\varepsilon^2P_{\ell:1}P_{\ell:1}^\top.
\]
Its energy in a backward QR direction \(q_k(\ell)\) is exactly
\[
q_k^\top C_\ell^{\rm tangent}q_k.
\]
QR directions are not generally covariance eigenvectors. Substituting \(e^{2\lambda_k\ell}\) for these energies requires an additional asymptotic approximation.

PROVEN UNDER survival of the required QR rank — noiseless radial-frame exponents.

Anchor the first QR vector to the reference state’s direction. If \(M\) gates are active in a layer, then
\[
R_{kk}^2\stackrel d=\frac2n\chi^2_{M-k+1},
\qquad M\ge k.
\]
Consequently,
\[
\lambda_k=
\frac12\left[
\log(4/n)+E\psi((M-k+1)/2)
\right]
\]
under the corresponding rank conditioning. For fixed \(k\), large \(n\),
\[
\lambda_k=-\frac{k+1/4}{n}+O(n^{-2}),
\qquad
\lambda_k-\lambda_1=-\frac{k-1}{n}+O(n^{-2}).
\]

At finite width, an all-off layer eventually kills a trajectory. These are finite-window or survival-conditioned exponents, not literal nonzero infinite-time exponents.

HEURISTIC — tangent age profile.

After age \(\ell\), relative tangent energy in direction \(k\) scales roughly as
\[
\exp[-2(k-1)\ell/n].
\]
Thus approximately \(n/\ell\) transverse directions can remain important. Initial projections, QR mixing, and finite-window fluctuations affect the constants.

Refutation: compare this prediction against directly propagated tangent covariance in the measured QR basis, with at least two QR initialization horizons.

PROVEN — finite differences have an additional forcing term.

For two finite trajectories,
\[
\Delta_\ell=J_\ell\Delta_{\ell-1}+r_\ell,
\]
where
\[
r_\ell=
\operatorname{relu}(z_\ell+W_\ell\Delta_{\ell-1})
-\operatorname{relu}(z_\ell)
-D_\ell W_\ell\Delta_{\ell-1}.
\]
Hence
\[
\Delta_\ell=P_{\ell:1}\Delta_0+
\sum_{j=1}^{\ell}P_{\ell:j+1}r_j.
\]
The \(r_j\) are generated at gate crossings. They are correlated, deterministic functions of the original inputs and weights; they are not independent noise births.

HEURISTIC — independent inputs.

Finite-angle differences continually generate these terms, so their QR energy profile need not match a tangent perturbation injected only at layer zero. They can populate directions that an old infinitesimal perturbation has nearly lost.

Most importantly, the late radial mode contains both \(R\) and the accumulated \(A(U)\). “The difference is radial at layer \(L\)” does not imply “the original Gaussian radius carries it.”

Refutation: record the exact \(r_j\), propagate them through the reference tangent products, and measure their QR-resolved contributions and cross terms. An independent-injection approximation fails if those cross terms are substantial.

Q5. What sets RQMC, antithetic, and multilevel gains?

PROVEN — \(S_1\) alone does not determine RQMC variance.

For an appropriate scrambled digital net, variance has a resolution-dependent decomposition of the form
\[
N\operatorname{Var}(\hat\mu)
=\sum_{u,\kappa}\Gamma_{u,\kappa}(N)\,\sigma_{u,\kappa}^2.
\]
The terms distinguish coordinate subsets and their frequency/resolution content. Ordinary Sobol indices aggregate over \(\kappa\), losing information needed to predict the gain.

PROVEN UNDER a component-efficiency model —
\[
G\approx\frac1{1-S_1}
\]
holds if additive components are almost completely integrated and every remaining component behaves like MC.

A more useful model is
\[
G^{-1}\approx
r_1S_1+\sum_{i<j}r_{ij}S_{ij}+r_{\ge3}S_{\ge3}.
\]
The pair efficiencies \(r_{ij}\) depend on the specific two-dimensional projections and the functions occupying them. At \(N=2^m\), their digital-net quality and resolution relative to \(m\) matter. Pair count alone is insufficient.

HEURISTIC — application to the input regime.

At ARC, \(S_1\approx0.27\) predicts only about \(1.37\times\) under the additive-only model. A measured gain near \(2.1\times\) therefore requires substantial suppression of interactions, a different coordinate mapping, or a materially larger unmeasured additive tail.

Refutation: directly measure full scalar-coordinate \(S_1\) and the estimator variance on the same nets and coordinate map. The additive-only explanation fails if their confidence intervals exclude \(G^{-1}=1-S_1\).

PROVEN — hybrid sampling has a specific MC floor.

Let \(U\) be the recent Sobol coordinates and \(Z\) older noise sampled independently for every trajectory. Define \(g(U)=E_Zf(U,Z)\). Then
\[
\operatorname{Var}(\hat\mu)
=
\operatorname{Var}_{\rm RQMC}\!\left(\frac1N\sum_i g(U_i)\right)
+\frac1N E_U\operatorname{Var}_Zf(U,Z).
\]
This older-noise term includes interactions with recent noise. It is not merely the variance explained by older noise alone.

PROVEN UNDER fixed finite dimension and full nested uniform scrambling — square-integrable integrands have variance \(o(N^{-1})\).

Consequently, a plateau over \(2^8\)–\(2^{16}\) does not establish an asymptotic \(1/N\) law for fully scrambled finite-dimensional integration. A hybrid MC floor can establish such a law for the hybrid estimator. Stronger \(N^{-3}\)-type claims require smoothness conditions that Gaussian inverse-CDF ReLU integrands do not automatically satisfy.

PROVEN — antithetics.

Let \(s_{\rm even}\) be the fraction of centered variance invariant under the chosen sign reversal. At equal numbers of network evaluations,
\[
\boxed{G_{\rm anti}=\frac1{2s_{\rm even}}.}
\]
It is total parity, not ANOVA order, that controls this gain.

HEURISTIC — input parity.

The radius is entirely even. The amplitude model gives
\[
s_{{\rm even},A}
\approx1-\frac{C_L(-1)}{2C_L(0)}.
\]
Since
\[
C_L(-1)=1.5+C_{L-1}(0),
\]
the deep amplitude even fraction tends to approximately 0.4296. Including the radius gives a deep raw-input antithetic gain near 1.099; radius-exact input antithetics approach approximately 1.164.

The eight ARC nets gave even fractions 0.442–0.476, corresponding to gains 1.05–1.13. Refutation of the asymptotic prediction: parity estimates at growing width and \(L/n\) fixed converge elsewhere.

HEURISTIC — stationary-noise parity.

Suppose the normalized noise-stability difference satisfies
\[
\frac{F(1)-F(t)}{F(1)-F(0)}
\approx(1-t)^{2/3},
\]
where corresponding noise coordinates in the replicas have correlation \(t\). Then
\[
s_{\rm even}\approx1-2^{-1/3}=0.20630,
\]
and
\[
\boxed{G_{\rm anti}\approx2.42366.}
\]
This agrees strikingly with the reported large-width noise gains. It remains a closure for the full output covariance, not a proof supplied by the agreement.

Refutation: measure the stability curve at several \(t\), including \(-1\), rather than fitting only the antithetic point.

PROVEN — multilevel gains depend on variance times cost.

For independent level estimates with variances \(V_j\), costs \(C_j\), and sampling budget \(B\),
\[
\operatorname{MSE}_{\rm opt}
=\frac{(\sum_j\sqrt{V_jC_j})^2}{B}.
\]
Its advantage comes from estimating substantial variance on cheap prefixes while leaving little variance on expensive levels.

There is no single Sobol share that determines this gain. For the shared-trajectory construction, the important quantities are amplitude increments and the direction residual requiring a full pass. If the latter remains large, cheap treatment of amplitude cannot produce a large total gain.

CONJECTURE — a preregisterable RQMC predictor.

On held-out networks, estimate:

1. The hybrid floor \(\eta=E_U\operatorname{Var}_Zf/V\).
2. First-order and pair ANOVA shares of \(g(U)\).
3. Pair projection efficiencies in predetermined quality bins.

Predict
\[
G^{-1}_{\rm pred}
=\eta+(1-\eta)
\left[r_1S_1+\sum_b r_{2,b}S_{2,b}+S_{\ge3}\right],
\]
with the \(r\)'s fixed from separate calibration runs.

Reject it if held-out variance ratios differ by more than 20% with confidence intervals excluding the prediction. Likely failure mechanisms are useful higher-order integration, unsuitable calibration spectra, or poor one-dimensional integration.

Growing mean interaction order alone is not an explanation: it neither determines \(S_1\) nor the efficiencies of the surviving components.

Q6. Stationary-noise first-order share

PROVEN — it is not a function of width alone.

The answer also depends on \(d\), the coordinate transformation, the treatment of older noise, and the fixed weight sequence. The measured mean interaction order over layer blocks does not determine first-order share over scalar Sobol coordinates.

For Gaussian coordinates, the total degree-one energy is exactly
\[
H_1=\sum_j\|E[f(\Xi)\Xi_j]\|^2
=\sum_j\|E[\partial_j f(\Xi)]\|^2.
\]
The scalar-coordinate first-order ANOVA share includes additional univariate Hermite degrees.

HEURISTIC — explicit frozen-radius prediction.

Use the stationary radius proxy \(Q=0.577078\,n\). For noise correlation \(t\), solve
\[
(Q+1)c(t)=Q\kappa(c(t))+t.
\]
This freezes the fluctuating radius and is therefore an adiabatic model.

For the ReLU output, its predicted linear-chaos fraction is
\[
s_{\rm lin}^{a}(Q)=
\frac{\kappa'(c(0))}
{[Q+1-Q\kappa'(c(0))]\,[1-\kappa(c(0))]}.
\]

| \(n\) | Predicted output linear share |
|---:|---:|
| 8 | 0.674 |
| 16 | 0.663 |
| 32 | 0.656 |
| 64 | 0.652 |
| 128 | 0.651 |
| 256 | 0.651 |
| \(n\to\infty\) | \(2/3\) |

If individual coordinates have diffuse influence, higher univariate degrees should become small, making this a candidate approximation for full scalar-coordinate \(S_1\).

Refutation: scalar-coordinate pick-freeze estimates, with sufficient recent-noise coverage, remain far from this table or do not approach \(2/3\). Radius fluctuations and radial covariance are particularly plausible causes of failure.

HEURISTIC — comparison with the measured gains.

This prediction does not explain 7.6× at \(n=8\) using \(1/(1-S_1)\): it predicts about 3.1×. At \(n=128\), it predicts about 2.86× before accounting for older-noise variance and imperfect projections, reasonably close to 2.1–2.6×.

An additive-only account of the reported gains would require
\[
S_1\approx0.868\quad(n=8),
\qquad
S_1\approx0.524\text{–}0.615\quad(n=128).
\]
Those values should be measured, not inferred from the gains.

My prediction is that small-width RQMC also integrates substantial nonadditive structure, while high-width performance loses some of that benefit and may encounter an older-noise floor.

If the implementation caps Sobol dimension near 21,000, the recent window is also changing materially: \(d\le164\) at \(n=128\), and \(d\le82\) at \(n=256\). That decreases \(d/n\). Refutation: repeat at fixed \(d/n\), or separately measure the exact hybrid floor; unchanged gains would weaken this explanation.

Q7. Structured samplers and full costs

PROVEN — radius-exact MC is the correct cheap baseline.

Sample \(U\) uniformly and return
\[
Y(U)=m_n g(U).
\]
Each sample costs \(L\) mv, with only \(O(n)\) additional work. At budget \(B\) mv,
\[
\operatorname{MSE}=\frac{V_{\rm sph}L}{B},
\qquad V_{\rm sph}=V_A+V_D.
\]
There is no pilot. Its gain is approximately 1.04 at ARC and does not grow with width in the deep-input limit.

PROVEN UNDER the shared-trajectory level bounds — multilevel MC has a width-growing advantage.

For dyadic prefixes, use the bound
\[
E\|Y-G_r\|^2=O(1/r+n/L^2).
\]
Then, at \(L=\alpha n\),
\[
\operatorname{MSE}=O(\log^2 n/B)
\]
in mv units, versus \(\Theta(n/B)\) for radius-exact MC. This gives the stated \(\Omega(n/\log^2 n)\) gain.

The charged budget is
\[
B=L+B_{\rm pilot}+\sum_jN_jC_j.
\]
The reference costs \(L\) mv; a prefix sample costs \(r_j\) mv; the terminal correction costs \(L\) mv. A pilot of \(P_j\) samples per level costs \(\sum_jP_jC_j\). Pilot samples used to choose an allocation should not silently be reused in a way that biases the production estimator.

The rate supplies no practical constant at \(n=256\). The observed near-break-even result at ARC is compatible with the theorem.

PROVEN — an exact-mean first-layer control.

Let
\[
Q_1(U)=\|\operatorname{relu}(W_1U)\|^2,\qquad
q_1=E_UQ_1(U)=\frac{\|W_1\|_F^2}{2n}.
\]
Replace the first prefix surrogate
\[
G_1=m_ns_1\sqrt{Q_1}\,q_L
\]
by the analytically integrable control
\[
C_1(U)=m_ns_1q_L
\left[\sqrt{q_1}+\frac{Q_1(U)-q_1}{2\sqrt{q_1}}\right].
\]
Its mean is exactly
\[
E C_1=m_ns_1\sqrt{q_1}\,q_L.
\]
Estimate \(G_1-C_1\) and retain the remaining multilevel corrections. The estimator remains unbiased.

Cost: one full reference pass; \(O(n^2)\) work to compute \(q_1\); one mv per first-level residual sample; no fitted coefficient or pilot for this control.

PROVEN UNDER a fourth-moment bound — its first-level residual variance is \(O(n^{-1})\).

The exact identity
\[
\sqrt Q-\sqrt q-\frac{Q-q}{2\sqrt q}
=-\frac{(Q-q)^2}{2\sqrt q(\sqrt Q+\sqrt q)^2}
\]
shows this if \(q_1\) is bounded away from zero and
\[
E_U(Q_1-q_1)^4=O(n^{-2}).
\]
A Gaussian fluctuation approximation with \(\operatorname{Var}Q_1\approx5/n\) predicts residual variance approximately
\[
0.78125\,s_1^2/n
\]
after centering, versus an uncontrolled first-layer amplitude variance near \(1.25s_1^2\).

HEURISTIC — practical value of that control.

This is my first sampler experiment. It should remove most of the first-level sampling cost without a learning phase. It does not by itself improve the full multilevel asymptotic rate: later levels can still dominate.

Refutation: \(n\) times the measured first-level residual variance grows substantially with width, or the fully charged estimator fails to improve on the existing multilevel sampler.

HEURISTIC — rotated-coordinate RQMC and quadratic controls.

For a scalar amplitude approximation, diagonalize a fitted quadratic Hermite component and put influential coordinates first. Alternatively, subtract the quadratic control directly; its Gaussian mean is known exactly.

Costs:

- \(P_0\) full pilot samples: \(P_0L\) mv.
- Dense eigendecomposition: \(O(n)\) mv-equivalents.
- Folding the rotation into \(W_1\): \(O(n)\) mv-equivalents, after which forward samples still cost \(L\) mv.
- An explicit dense rotation per sample instead costs approximately one extra mv.
- General vector-valued quadratic controls can cost \(O(n)\) mv per sample; a scalar-amplitude or low-rank restriction avoids that expense.

At \(L/n\) fixed, the amplitude model places about 49.9% of spherical variance in degrees one and two. Removing them would give roughly a 2× gain if the residual is otherwise MC-like. At fixed \(L\), these controls generally remove a constant fraction, not a fraction tending to one.

Refutation: held-out residual variance does not match the captured spectral energy, or pilot/setup cost consumes the gain. A rotation chosen from the same samples used to score it is not an honest validation.

PROVEN UNDER component RQMC rates — RQMC can be used on multilevel corrections.

If level \(j\) has variance \(v_jN_j^{-p}\), with common \(p>1\), the optimal allocation scales as
\[
N_j\propto(v_j/C_j)^{1/(p+1)}.
\]
For fixed dimension, fully nested scrambled sampling can give \(o(N_j^{-1})\) without a specified power.

HEURISTIC — no uniform power improvement is established here.

The dimension, gate geometry, Gaussian tails, and level functions all vary with \(n\). A measured \(p>1\) on a short range does not establish a width-uniform rate.

Refutation: preregister a common exponent and width-independent rescaling, then test both on unseen widths and larger budgets.

CONJECTURE — several references may reduce the expensive direction residual.

Use \(K\) references, paying \(KL\) mv setup, and choose or combine their suffix predictions from a prefix state. The required test is the decrease in terminal residual variance versus the added setup, selection, and prefix costs.

I do not have a theorem that polynomially many references cover the relevant nonlinear image well enough for a width-growing gain. Fixed late-layer gates or low gate entropy do not establish such a covering property.

Failure condition: residual variance decreases too slowly to compensate for \(K\), or choosing references requires nearly a full forward pass.

PROVEN — the noisy transition requires a correction.

For the actual \(\sigma>0\) estimand, radius-exact noiseless sampling alone is biased. An unbiased decomposition is
\[
\mu_\sigma=\mu_0+
E[f_\sigma(X,\Xi)-f_0(X)].
\]
The coupled correction costs about \(2L\) mv per sample. Add its variance-times-cost contribution to the allocation formula.

HEURISTIC — this correction should be cheap sufficiently below memory loss if its coupled variance is small. Near \(T_{\rm mem}\), that advantage can disappear. The supplied input-variance fraction is not a substitute for measuring the coupled correction variance and mean shift.

Q8. What I would change or test first

PROVEN — three distinctions should be explicit in the framing.

1. Late-layer radial information includes \(A(U)\); it is not just \(\|X\|\).
2. Layer-block interaction order, scalar-coordinate ANOVA order, and Hermite degree are different quantities.
3. Ratios of annealed variances are not average per-net normalized errors or gains.

PROVEN — fixed \(\sigma\), \(n\to\infty\), and \(L/n\) fixed do not automatically remain in the input regime. The factor \(\sigma\sqrt n\) changes. A clean asymptotic input-regime statement must specify the joint scaling of \(\sigma,n,L\).

CONJECTURE — the most informative next experiment is a matched-net decomposition, not another broad sampler sweep:

- Measure exact radial removal, scalar-norm variance, direction residual, scalar-coordinate \(S_1\), and parity on the same nets.
- Measure the older-noise hybrid floor separately from older-noise first-order importance.
- Compare actual RQMC variance with the preregistered additive-plus-pairs predictor.
- Test the first-layer squared-norm control with setup and pilot charged.

The central falsifiable prediction is: amplitude dominates once depth passes order \(\sqrt n\), while a nonzero additive fraction survives even as mean interaction order grows. That permits constant RQMC gains and width-growing multilevel gains simultaneously.

The research results are contained in this response. Saving an artifact and filing the required bead were blocked by the read-only sandbox; recall also failed because SSH rejected its configuration permissions.
