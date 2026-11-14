Section 6 needs revision before circulation: **2 statements are correct, 6 have fixable gaps, and 5 are wrong as stated.** The norm law, Gaussian kernel, covariance-growth mechanism, and multiplicative-error identity survive scrutiny. The main problems are the information restriction in Theorem 6.5, its “1–3 samples” interpretation, the unsupported error-transport step in Corollary 6.9, and the noise claims. Theorem 6.12 also omits initial error. I checked Sections 1 and 6, the numerical code, and all three supplied logs; independent computations used one process at a time under the requested limits. The full report was printed, but the requested file could not be created.

## Lemma 6.1 — CORRECT

For a fixed nonzero input, conditional on preceding layers, the standardized projections of the fresh Gaussian rows are independent standard normals. Their conditional distribution does not depend on the history. This proves **independence across layers**, not merely identical marginal distributions.

Writing \(X=2(g_+)^2\), we have
\[
\mathbb EX=1,\qquad \mathbb EX^2=6.
\]
Consequently,
\[
\operatorname{Var}\rho=\frac5n,\qquad
\mathbb EG_L=1,\qquad
\mathbb EG_L^2=(1+5/n)^L.
\]
The limit is \(e^{5\alpha}\) when \(L/n\to\alpha\).

The extinction convention works. A live trajectory dies at its next layer with probability \(2^{-n}\); its product already contains a zero factor. Subsequent factors can use fresh weights and any **past-measurable** unit direction, preserving the iid law. They are then auxiliary factors rather than actual ratios of nonzero norms.

Specify “past-measurable” in the convention. Actual cosines after extinction still need a definition in the following lemmas.

## Lemma 6.2 — CORRECT WITH A FIXABLE GAP

On histories with nonzero incoming states,
\[
\mathbb E[\rho\rho'\mid\text{past}]
=\frac4{n^2}\left[nJ(c)+\frac{n(n-1)}4\right]
=1+\frac4n\left(J(c)-\frac14\right).
\]

The displayed closed form for \(J\), including its denominator \(2\pi\), is correct. A common arc-cosine-kernel convention uses twice this Gaussian expectation; there is no missing factor here. The endpoint values are also correct.

The Hermite-series monotonicity argument works on \([0,1]\). Alternatively, with \(\mu(c)=\mathbb E[g_+g'_+]\),
\[
J'(c)=4\mu(c)>0.
\]

**Required fix:** define the cosine after extinction, or restrict the conditional identity to live incoming states. A consistent global convention is
\[
z_l=W_lu_{l-1},\quad \rho_l=\|\operatorname{relu}(z_l)\|^2,\quad
u_l=
\begin{cases}
\operatorname{relu}(z_l)/\sqrt{\rho_l},&\rho_l>0,\\
e_1,&\rho_l=0.
\end{cases}
\]
Continue this auxiliary recursion after the actual trajectory dies and define \(c_l=u_l^\top u'_l\). It agrees with the physical cosine whenever both trajectories survive.

The numerical values support the formula but do not replace its analytic justification.

## Lemma 6.3 — CORRECT WITH A FIXABLE GAP

The stated lower-tail inequality is valid for independent nonnegative summands with finite second moments:
\[
\Pr(S\le \mathbb ES-s)
\le \exp\!\left[-\frac{s^2}{2\sum_j\mathbb EX_j^2}\right].
\]
Indeed, for \(t\ge0\),
\[
e^{-tX}\le1-tX+\frac{t^2X^2}{2},
\]
which yields the required exponential-moment bound. The denominator is the sum of **second moments**, not variances.

The minimum can be established analytically. Put \(a=1/(4\pi)\) and
\[
R(c)=\frac{(\mu(c)-a)^2}{2J(c)}.
\]
Using
\[
\mu'(c)=\frac{\pi-\arccos c}{2\pi},\qquad J'=4\mu,
\]
the sign of \(R'\) is the sign of
\[
H=\mu'J-2\mu(\mu-a).
\]
Now
\[
H(0)=\frac1{16}-\frac1{4\pi^2}>0,\qquad
H'=\mu''J+2a\mu'>0.
\]
Thus the minimum occurs at zero and equals
\[
\frac1{8\pi^2}=0.0126651479552922.
\]
The lemma’s exponent \(0.0126\) is safe.

There is a small Chernoff-constant error:

- \(\lambda=0.21\) gives rate **0.06457398594**, not a rate at least \(0.0646\).
- Changing to \(\lambda=0.215\) gives **0.06462829850**, making \(0.0646\) valid.
- The final stated exponent \(0.064\) is valid either way.

The denominator argument is correct: if both squared norms are at most \(n\), their product’s square root is at most \(n\).

**Required fix:** use the auxiliary cosine convention above, or condition on both incoming states being nonzero. Setting physical cosines to zero after extinction makes the conditional bound false on an already-extinct history: its left side becomes one.

## Lemma 6.4 — CORRECT WITH A FIXABLE GAP

The asymptotic covariance lower bound survives, but its proof needs these repairs.

**The recurrence starts at \(l=3\).** The displayed inequality
\[
\mathbb E[Y_{l-1}Z_l]\ge(1+\delta_*)\mathbb EY_{l-1}
\]
is not valid at \(l=2\), because \(E_1\) imposes no condition on \(c_1\). For \(x'=-x\), surviving first-layer states have cosine zero, so the next conditional multiplier is exactly one.

The separate base calculation
\[
\mathbb E[Z_1Z_2]\ge1-\frac1n
\]
is correct. Start there, stop on bad cosines through layer \(L-1\), and apply the final multiplier without imposing a condition on \(c_L\). This gives the stated exponent \(L-2\).

**The subtraction step is valid.** For \(P_l=\prod_{j\le l}Z_j\),
\[
Y_l
=Y_{l-1}Z_l-P_l1_{E_{l-1}}1_{\{c_l<c_*\}}
\ge Y_{l-1}Z_l-P_l1_{\{c_l<c_*\}}.
\]
Under a live-history formulation, previously extinct products vanish.

**The exact fourth moment is**
\[
M_4(n)=\mathbb E\rho^4
=1+\frac{30}{n}+\frac{251}{n^2}+\frac{558}{n^3}.
\]
Thus the expansion \(1+30/n+O(n^{-2})\) is correct, but
\[
M_4(n)\le1+\frac{31}{n}
\]
holds only for integer \(n\ge254\).

Conditional Cauchy–Schwarz gives
\[
\mathbb E[Z_l^2\mid\text{past}]\le M_4(n).
\]
A valid all-width remainder is therefore
\[
\epsilon_{n,L}\le
\sqrt3\,L(1+\delta_*)^L M_4(n)^{L/2}
e^{-n/(16\pi^2)}.
\]
For \(n\ge254\), this implies the draft’s remainder because \(16\pi^2<158\).

Retain the **sharp** concentration exponent in this derivation: the rounded \(0.0126\) alone does not imply \(e^{-n/158}\) after taking a square root.

The remainder tends to zero for \(L=O(n)\). Also,
\[
\kappa_*=0.215521457681215,
\]
so the claimed positive limiting covariance bound holds when \(L/n\to\alpha>0\).

## Theorem 6.5 and its table — WRONG AS STATED

The hypothesis “\(\xi\) independent of \(A\)” permits an information leak. It does not imply that the joint pair \((B,\xi)\) is independent of \(A\).

Encode the Gaussian block \(A\) by a uniform variable \(U\) from which \(A\) can be recovered. Let \(V=V(B)\) be uniform and set
\[
\xi=(U+V)\bmod1.
\]
Then \(\xi\) is independent of \(A\), but \(B\) and \(\xi\) together recover \(A\). An unrestricted estimator can compute \(S\) exactly.

A concrete scalar example has two layers, \(A=a\), \(B=b\), with independent \(N(0,2)\) weights. Then
\[
S=b_+^2a^2/2,\qquad
\operatorname{Var}(\mathbb E[S\mid A])=2.
\]
Taking \(U=\Phi(a/\sqrt2)\), \(V=\Phi(b/\sqrt2)\) supplies the counterexample above.

**Required fix:** assume
\[
A\ \perp\!\!\!\perp\ (B,\xi).
\]
External coins independent of all weights satisfy this.

With that repair, the mathematical argument works:

- For every fixed \(x\), averaging the last \(k\) layers preserves its squared norm. Conditional Fubini/Tonelli therefore gives
  \[
  \mathbb E[S\mid A]=\mathbb E_x\|h_{L-k}(x)\|^2/n.
  \]
  Independence across inputs is unnecessary.
- The ANOVA decomposition is orthogonal and gives the claimed risk inequality.
- Lemma 6.4 is uniform over deterministic input pairs, so its bound can be integrated against the nonnegative norm weights. Their expectation is one.
- Use \(m=L-k\ge2\) and explicitly write \(\epsilon_{n,m}\).
- The Monte Carlo bound is correct:
  \[
  \mathbb E_{W,x}f_L(x)^2=(1+2/n)(1+5/n)^L.
  \]

The table accurately rounds the supplied log. The code’s subtraction of the estimated inner Monte Carlo variance is an unbiased correction. **It does not remove sampling uncertainty.**

More seriously, its displayed “bound” omits the finite-width remainder:

| \(n,L,m\) | Displayed leading expression | Valid remainder upper bound |
|---|---:|---:|
| 64, 16, 8 | 0.004433 | 52.30 |
| 64, 32, 16 | 0.031813 | 592.18 |
| 128, 32, 16 | 0.015834 | 75.43 |
| 128, 64, 32 | 0.043549 | 923.44 |

These are loose upper bounds on the remainder, not estimates of its actual size. Nevertheless, **the proved finite-width lower bound is vacuous at every table size**. Even using the draft’s unjustified moment substitution gives remainders approximately \(46.09,459.70,71.79,836.09\).

The ratio
\[
\frac{\mathbb E_W\operatorname{Var}_x(f_L)}
{\operatorname{Var}(\mathbb E[S\mid A])}
\]
provides an upper bound on sample equivalence using exact population quantities. The code estimates that ratio; it does not estimate the optimal \(B\)-only risk, which also contains interaction variance. It supplies no uncertainty intervals.

The logged ratios are \(3.04,2.32,2.31,1.26\). Even treating them as exact, sufficient integer sample counts are **4, 3, 3, and 2**. “Is worth 1–3 samples” overstates both precision and what was measured.

The theorem’s loose asymptotic bounds, with \(k=L/2\), instead give limiting ratios approximately:

- \(91.20\) at \(\alpha=1/4\);
- \(202.002\) at \(\alpha=1/2\).

Thus it proves a constant sample ceiling, not a ceiling of three. The code’s printed \(e^{5L/n}-1\) is also an asymptotic expression rather than the general finite-\(n\) bound.

Finally:

- \(S\) is a **quadratic energy observable**, not Section 1’s linear-readout mean. No transfer theorem is supplied.
- The “by symmetry” assertion about ignoring late layers needs a separate proof and a depth condition. Single-input gain variance cannot simply be assigned to an input-averaged observable: for one layer,
  \[
  S_1=\|W_1\|_F^2/(2n),\qquad \operatorname{Var}S_1=2/n^2,
  \]
  rather than \(5/n\).

## Lemma 6.6 — CORRECT WITH A FIXABLE GAP

The telescoping identity is exact for integrable measures and for linear functionals whose domains contain the required backward functions and are compatible with pullback:
\[
(T_lQ)(g)=Q(T_l^*g).
\]

Specify these domains and finite pairings. A finite moment or cumulant representation does not automatically define \(Q(G_l)\), since \(G_l\) is generally nonpolynomial. Cumulants themselves are not linear functionals. A reconstruction or evaluation rule is needed before the identity applies to a cumulant-propagation algorithm.

The following sentence, “accurate iff its local errors are small,” is false. Large local terms can cancel. For identity layers, initial law \(\delta_0\), approximations \(Q_1=\delta_R\), \(Q_2=\delta_0\), and readout \(h\), the local terms are \(-R,+R\), with zero final error.

Replace it with: **The output error is the signed sum of the tested local errors; bounding their absolute values gives a sufficient accuracy condition.**

## Proposition 6.7 — WRONG AS STATED

The noiseless degree-one homogeneity assertion is correct for positive scale factors. A single perturbation multiplies the downstream output by \(1+t\); purely multiplicative errors at several layers multiply it by \(\prod_l(1+t_l)\).

Specify positive factors and a nonzero reference output when discussing relative error. For Theorem 6.5’s quadratic observable, the factor is \((1+t)^2\).

The variance assertion requires centering and a small-error regime. For independent centered errors with variances \(v_l\),
\[
\operatorname{Var}\!\left[\prod_l(1+t_l)-1\right]
=\prod_l(1+v_l)-1.
\]
This is approximately \(\sum_l v_l\) only when that total is small. At fixed positive per-layer variance, growth is exponential in depth.

The uniform \(O(\sigma^2)\) noise claim is false. Consider two scalar downstream ReLUs with unit weights and additive noise of standard deviation \(\tau\) between them. For \(h\ge0\),
\[
G(h)=\mathbb E(h+\tau\xi)_+
=h\Phi(h/\tau)+\tau\phi(h/\tau).
\]
Therefore
\[
G(0)-cG(0)=\frac{(1-c)\tau}{\sqrt{2\pi}},
\]
an \(O(\tau)\) homogeneity defect. The same order occurs for positive \(h=O(\tau)\).

Keep the noiseless identity. State the exact variance formula. Remove the general \(O(\sigma^2)\) claim unless boundary-mass, smoothness, and amplification assumptions are supplied.

## Proposition 6.8 — CORRECT WITH A FIXABLE GAP

The Gaussian independence argument is correct. For
\[
H=\operatorname{span}(\operatorname{supp}Q),
\]
the projections of a Gaussian row onto \(H\) and \(H^\perp\) are independent. Thus \(w^\top\delta\) is independent of the entire collection of state projections when \(\delta\in H^\perp\).

For nonzero states, the gate-overlap probability and resulting factor are correct:
\[
\Pr(w^\top h>0,w^\top h'>0)=\frac{\pi-\theta}{2\pi}.
\]

Add these hypotheses:

- \(Q,\delta\) are determined before the fresh \(W\);
- \(Q\) has finite first moment;
- \(Q(\{0\})=0\);
- “shift the mean” means **translate every state by the same vector**.

The zero-atom exclusion matters. For \(Q=\delta_0\), the displayed definition gives \(\Delta=0\), whereas the actual one-sided derivative is \(\operatorname{relu}(W\delta)\), whose expected squared norm is \(\|\delta\|^2\). The angle is undefined.

The numerical experiment correctly tests the formal one-step derivative for an **empirical law on 128 states in \(\mathbb R^{256}\)**. It does not test orthogonality to population support, finite displacements, or multilayer survival.

For example, with invertible first-layer weights and full-support Gaussian inputs, the state support spans \(\mathbb R^n\), leaving no nonzero exactly orthogonal perturbation. Small participation ratio does not establish an exact orthogonal complement.

## Corollary 6.9 — WRONG AS A PROVEN COROLLARY

The deterministic angle asymptotics are correct for fixed \(c_0<1\):
\[
1-\kappa(c)
=(1-c)-\frac{2\sqrt2}{3\pi}(1-c)^{3/2}
+O((1-c)^2).
\]
Writing \(u_l=1-c_l\),
\[
u_{l+1}^{-1/2}-u_l^{-1/2}\longrightarrow\frac{\sqrt2}{3\pi},
\]
which establishes
\[
1-c_l\sim\frac{9\pi^2}{2l^2},\qquad
\theta_l\sim\frac{3\pi}{l}.
\]
Exclude \(c_0=1\), for which every angle remains zero.

The product needs an asymptotic qualification. For integers \(\ell\ge4\),
\[
\prod_{l=\ell}^L(1-3/l)
=\frac{(\ell-3)(\ell-2)(\ell-1)}{L(L-1)(L-2)}.
\]
It is equivalent to \((\ell/L)^3\) when \(\ell\to\infty\), not for fixed \(\ell\). Including \(l=3\) makes the approximation zero. Retain actual gate factors during the initial layers.

The main problem is the **multilayer error transport**. Proposition 6.8 does not iterate automatically. After one layer, the perturbation is
\[
D_W(h)W\delta,
\]
which depends on the state. It is no longer a common translation of the new law, and its mean need not be orthogonal to the new state span. Pairwise angles alone do not determine its subsequent transport.

Likewise, uncorrelated injected errors need not remain uncorrelated after random downstream transformations. A conditional-orthogonality or martingale argument is missing.

Under an additional scalar transport model and orthogonality of propagated contributions, the arithmetic is correct:
\[
e^2\sum_{\ell=1}^L(\ell/L)^3
=e^2\left(\frac L4+\frac12+\frac1{4L}\right).
\]
But that does not prove the network claim.

Separate the proved angle asymptotics from the heuristic accumulation model. The chaotic-tanh extension needs its own derivative calculation and error-mode assumptions. Deterministic infinite-width recursion also requires separate justification at \(L=\Theta(n)\).

## Lemma 6.10 and its example — CORRECT WITH A FIXABLE GAP

Explicitly require both \(\rho_l\) and \(\widehat\rho_l\) to be \(\mathcal F_l\)-measurable, with finite stated conditional moments. Under adaptedness, the orthogonality proof is correct.

A shorter proof uses
\[
\mathbb E[\rho_l^2\mid\mathcal F_{l-1}]=a+b,\quad
\mathbb E[\rho_l\widehat\rho_l\mid\mathcal F_{l-1}]=a,\quad
\mathbb E[\widehat\rho_l^2\mid\mathcal F_{l-1}]=a.
\]
Successive conditioning gives the claimed product MSE. Neither layerwise independence nor conditional centering of \(e_l\) is needed.

“Relative error” here means MSE divided by \(\mathbb E G^2\), not variance-normalized error or expected squared relative error \( \mathbb E[(G-\widehat G)^2/G^2]\).

The Hermite example satisfies the hypotheses when it uses exact projection along the **true normalized trajectory**. Rotational invariance makes the conditional moments constant, and Hermite orthogonality gives the required cross-moment zero:
\[
a=1+\frac{5(1-r_D)}n,\qquad b=\frac{5r_D}n.
\]

The exact fractions are:

| Degree | \(r_D\) |
|---|---:|
| 1 | \(1-8/(5\pi)=0.4907041821\) |
| 2 | \(3/5-8/(5\pi)=0.0907041821\) |
| 3, 4 | \(3/5-28/(15\pi)=0.0058215458\) |
| 5 | \(3/5-47/(25\pi)=0.0015774140\) |

The draft’s rounding is correct. At \(D=2,\alpha=1/4\), the limiting normalized MSE is **0.1071888759**.

The favorable logged predictions match the exact calculation. However, the code uses quadrature coefficients, making its projection numerically rather than algebraically exact.

The log also contains unresolved heavy-tail checks:

- \(n=32,L=64,D=2\): MSE \(93.79\) versus \(5921.20\); second moment \(206.02\) versus \(10847.05\).
- \(n=64,L=64,D=1\): MSE \(42.67\) versus \(111.11\).

These do not refute the identity; they show that those runs failed to resolve the moments. They should be disclosed rather than omitted from the validation account.

Retain two limitations: the surrogate already needs the true trajectory, and single-input error does not automatically imply an input-averaged lower bound.

## Proposition 6.11 — CORRECT

For symmetric \(M\) independent of \(w\sim N(0,s^2I)\),
\[
\mathbb E[w^\top Mw]=s^2\operatorname{tr}M,\qquad
\operatorname{Var}(w^\top Mw)=2s^4\operatorname{tr}(M^2).
\]
Taking \(s^2=2/n\) and \(M=C-\operatorname{diag}C\) gives exactly the stated mean and standard deviation.

The ratio
\[
\frac{\sqrt2\|C-D\|_F}{\operatorname{tr}C}
\]
is RMS prediction error divided by **mean true variance over \(w\)**. It is not error divided by each neuron’s realized variance. Assume \(\operatorname{tr}C>0\).

For \(C=\lambda uu^\top\), the ratio is
\[
\sqrt{2\left(1-\sum_i u_i^4\right)},
\]
which tends to \(\sqrt2\) for a spread-out eigenvector.

The conclusion concerns preactivation variance, not output-mean error. Angular alignment does not itself establish covariance collapse: \(h=a\mathbf1+\varepsilon z\), with independent symmetric sign coordinates and \(a\gg\varepsilon\), has nearly aligned nonnegative states but covariance \(\varepsilon^2I\).

The participation-ratio numbers are empirical and are not established by the supplied checks.

## Theorem 6.12 — WRONG AS STATED

The initial discrepancy from Lemma 6.6 is missing.

Take scalar dynamics
\[
T_l(h)=\eta\operatorname{relu}(h),\qquad 0<\eta<1,
\]
with \(P_0=N(0,1)\), \(Q_0=\delta_0\), and exact propagation of \(Q_l\). Every local error is zero. For the Lipschitz seminorm,
\[
\|G_l\|_{\mathrm{Lip}}=\eta^{L-l},
\]
so the stated contraction hypothesis holds with \(C=1\). Nevertheless,
\[
\mu=\frac{\eta^L}{\sqrt{2\pi}},\qquad \widehat\mu=0,
\]
contradicting the asserted zero error.

The correct bound is
\[
|\mu-\widehat\mu|
\le C\eta^L d_F(P_0,Q_0)
+\frac{Ce(1-\eta^L)}{1-\eta}.
\]
Alternatively, explicitly assume \(Q_0=P_0\).

Require \(0\le\eta<1\), finite pairings, and constants uniform in depth. A seminorm annihilating constants also requires compatible masses for finite dual distance.

## Proposition 6.13 — WRONG AS STATED

The Gaussian TV formula is correct for \(\tau>0\):
\[
\delta(K;S)=2\Phi\!\left(\frac{\Delta_S}{2\tau}\right)-1,
\]
using TV in \([0,1]\). Define \(\Delta_S\) as a supremum, possibly infinite.

Its consequences contradict the blanket “never” statement:

- Infinite image diameter gives coefficient one.
- Finite image diameter gives coefficient **strictly below one**.
- Constant image gives coefficient zero; \(W=0,\tau>0\) is an immediate counterexample.

For large \(\Delta/\tau\),
\[
1-\delta(K;S)
\sim\frac{4\tau}{\Delta\sqrt{2\pi}}
e^{-\Delta^2/(8\tau^2)}.
\]
The precise exponent \(n/(8\sigma^2)\) requires the corresponding precise diameter scaling. An order-of-magnitude relation alone does not justify its constant.

“Typical states” must name a set and establish its image diameter. Activation RMS alone does not determine separations.

A defensible conclusion is that small noise **need not provide useful dimension-uniform TV contraction** on sets with large image diameter. This does not rule out contraction in other metrics or over multiple steps.

The average-mixing-depth and depth-64 noise assertions are empirical, are not consequences of the proposition, and are not tested by the supplied logs.

## Required fixes, ordered by severity

1. **Remove the proved-network-accumulation claim from 6.9** until error transport and propagated-error orthogonality are established. Preserve the deterministic angle result separately.
2. **Repair 6.5’s independence assumption and scope.** Replace the “1–3 samples” conclusion with a qualified empirical upper-bound estimate and uncertainty. Identify \(S\) as a quadratic observable.
3. **Restore initial error in 6.12.** Remove the uniform \(O(\sigma^2)\) claim in 6.7 and universal noncontraction language in 6.13.
4. **Repair 6.4’s indexing and remainder.** Supply the exact fourth moment and sharp concentration rate. The table’s positive leading expressions are not certified finite-width bounds.
5. **Handle extinction and zero atoms explicitly.** Correct the Chernoff constant; state the population-versus-empirical-support distinction in 6.8.
6. **Specify functional domains, adaptedness, and error normalization.** Replace the false “iff” after 6.6 and qualify the linear-accumulation approximation in 6.7.
7. **Separate proofs, conditional models, and experiments.** Disclose the under-resolved Hermite runs and source the participation-ratio, chaotic-tanh, and noise observations separately.

The corrected norm moments and depth/width dependence are consistent with the Hanin and Hanin–Nica framework. The deterministic correlation asymptotics are consistent with the Poole and Schoenholz mean-field framework. Those results do not supply the missing arbitrary-error transport theorem or a uniform finite-width justification at depth proportional to width. Stability of a correlation fixed point also must not be confused with stability of infinitesimal trajectory perturbations in a chaotic regime.

Evidence checked at **2026-10-29 18:42:43 UTC**. Reviewed draft SHA-256: `d2c20187c2712f4e048c1e5ea7102cf36f8a0bb82aa07d9ce2f507adfb9a4180`. Independent numerical arithmetic used substantially less than one CPU-second; expensive Monte Carlo routines were inspected, not rerun.

Automatic approval review **dcg rejected the save/print command** for unresolved filesystem-command syntax. The session also has read-only filesystem permissions, so `/tmp/hardness_review_r2/review.md` was **not created**.
