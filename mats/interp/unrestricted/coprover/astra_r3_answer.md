1. **PROVEN UNDER layerwise batching costs and \(m(cT)=O_c(m(T))\): C0 is false.** A multilevel estimator achieves a growing gain of order \(n/\log^2 n\) at \(L=\alpha n\). It uses the realized common direction and multiplicative scale, while sampling input-dependent amplitudes through progressively longer prefixes.

Write \(Y(U)=m_n h_L(U)\), \(U\) uniform on \(S^{n-1}\). Generate one independent reference direction \(Q\), and propagate its **normalized** trajectory:
\[
q_0=Q,\qquad
\nu_\ell=\|\operatorname{ReLU}(W_\ell q_{\ell-1})\|,\qquad
q_\ell=\operatorname{ReLU}(W_\ell q_{\ell-1})/\nu_\ell.
\]
When \(\nu_\ell=0\), set \(q_\ell=e_1\); keep the zero in subsequent products. Define
\[
s_r=\prod_{\ell=r+1}^{L}\nu_\ell,\qquad
G_r(U)=m_n s_r\|h_r(U)\|q_L.
\]
Thus \(G_0=m_n f_W(Q)\) is constant in \(U\), with a known integral. Evaluating \(G_r\) costs an \(r\)-layer prefix; all reference quantities cost one full pass.

**PROVEN: the following residual bound holds at finite width**, without assuming mean-field convergence. Put
\[
\delta_\ell=1-\kappa^{\circ\ell}(0),\qquad
\kappa(c)=\frac{\sqrt{1-c^2}+(\pi-\arccos c)c}{\pi}.
\]
Then
\[
\boxed{
D_r:=E\|Y(U)-G_r(U)\|^2
\le
\frac{4m_n^2}{n}\sum_{\ell=r}^{L-1}\delta_\ell
+2m_n^2\delta_L
=O\!\left(\frac1{r+1}+\frac{n}{(L+1)^2}\right).
}
\tag{1}
\]

**PROVEN: two one-layer inequalities establish (1).** For unit vectors of correlation \(c\), let \(S_x,S_y\) be their squared output norms and \(C\) their output cosine, taking \(C=0\) when a norm vanishes. Exchangeability and AM–GM give
\[
E[S_xC]
=E\!\left[C\frac{S_x+S_y}{2}\right]
\ge E[C\sqrt{S_xS_y}]
=\kappa(c).
\tag{2}
\]
Also, for \(a(z)=\sqrt{2/n}\,\|z_+\|\), Gaussian noise stability gives
\[
E(\sqrt{S_x}-\sqrt{S_y})^2
\le 2(1-c)E\|\nabla a\|^2
\le\frac{4(1-c)}n.
\tag{3}
\]
The same bound holds for negative \(c\), by the Hermite expansion. Applying Jensen to (2) bounds the norm-weighted angular deficit at layer \(\ell\) by \(\delta_\ell\). Starting two suffix trajectories with equal norms, (3) accumulates their squared norm difference; their remaining angular difference contributes at most \(2\delta_L\). This proves (1). Reference resets can only improve the inequality in (2).

**PROVEN: finite-width amplitude variation changes the relevant variance scale.**
\[
\boxed{V_{\rm sph}=\Theta_\alpha(1),\qquad V=\Theta_\alpha(1)
\quad\text{at }L=\alpha n.}
\tag{4}
\]
For the upper bound, \(D_0=2V_{\rm sph}\), \(\sum_{\ell\ge0}\delta_\ell<\infty\), and \(\delta_L=O(L^{-2})\). For the lower bound, define
\[
A_r=m_n^2E_W\operatorname{Var}_U\|h_r(U)\|.
\]
Conditional Cauchy–Schwarz shows \(A_{r+1}\ge A_r\), and \(V_{\rm sph}\ge A_L\). At the first layer, if \(S=\|h_1(U)\|^2\),
\[
E_W\operatorname{Var}_U S=\frac5n-\frac2{n^2}.
\]
Expanding \(\sqrt S\), with an \(L^2\) remainder \(O(n^{-1})\), gives
\[
A_1=\frac54+o(1).
\]
Consequently the deep mean-field estimate \(V_{\rm sph}\asymp n/L^2\) misses a persistent finite-width amplitude term.

**PROVEN: an unbiased multilevel construction has the required error.** For dyadic depths \(0=r_0<r_1<\cdots<r_J=L\),
\[
\mu
=G_0+\sum_{j=1}^{J}E_U(G_{r_j}-G_{r_{j-1}})
       +E_U(Y-G_L).
\tag{5}
\]
Use independent sample batches for the expectations, conditional on \(W,Q\). Equation (1) implies
\[
E\operatorname{Var}_U(G_{r_j}-G_{r_{j-1}})
=O_\alpha(r_j^{-1}),\qquad
E\operatorname{Var}_U(Y-G_L)=O_\alpha(L^{-1}).
\]
With
\[
M_j=N\left\lceil\frac{L}{Jr_j}\right\rceil
\]
samples at level \(j\), and \(N\) final residual samples,
\[
\boxed{R=O_\alpha\!\left(\frac{\log^2 L}{LN}\right).}
\tag{6}
\]

**PROVEN UNDER the stated cost hypotheses: this comparison includes optimized MC batching.** Let \(C_L(N)\) be the best allowed layerwise cost for \(N\) full evaluations. Implement each level as repeated batches of size \(N\), using that same implementation:
\[
T\le C_L(1)+
\sum_j\left\lceil\frac{L}{Jr_j}\right\rceil
\frac{r_j}{L}C_L(N)+C_L(N)
=O(C_L(N)).
\]
Norms, products, and sample aggregation add only comparable or smaller work. If \(m(cT)=O_c(m(T))\) in the specified polynomial window, equations (4)–(6) give gain
\[
\boxed{\Omega_\alpha(n/\log^2 n).}
\]
Schoolbook costs satisfy these hypotheses, as do regular batching regimes. The explicit cost and error formulas remain valid if a proposed cost model lacks the doubling property; that model requires its own comparison. The earlier symmetry construction can supply deterministic sampling, under its stated computational conventions.

The requested competing families compare as follows.

| Family | Status, charged cost, and error assessment |
|---|---|
| Collapse onto a common direction | **PROVEN:** Collapse alone leaves amplitude variation. The computable control above has residual \(D_r=O(1/r+n/L^2)\), with setup \(O(Ln^2)\) and prefix cost \(O(rn^2)\) per sample. A two-level version with \(r\asymp\sqrt L\) already gives \(\Omega_\alpha(\sqrt n)\) gain under the same cost hypotheses. |
| Shared multiplicative scale | **PROVEN:** Recording the realized \(\nu_\ell\) costs one reference pass and captures the shared scale exactly. Using only that pass is one MC sample, with error \(V_{\rm sph}\). The growing gain comes from cheaply estimating the remaining amplitude corrections, as in (5). |
| Final-row identity and depth recursion | **PROVEN:** The linear part is exactly \(\tfrac12W_L\mu_{L-1}\), but its integral requires the penultimate mean. The extraction inequality loses \(2^{-(L-r)}\). **HEURISTIC:** These identities alone offer no growing gain; they neither exclude nor replace the amplitude-based estimator above. |
| Low rank or fixed low degree in early layers | **PROVEN UNDER the Gaussian singular-value limit:** Rank \(o(n)\) captures \(o(1)\) of a first-layer matrix’s Frobenius energy. **HEURISTIC:** Generic low rank is therefore unpromising. Fixed-degree input controls have computable moments, but vector coefficient storage already scales as \(O(n^{p+1})\) at degree \(p\); growing gain requires a vanishing residual fraction, which is not established for fixed \(p\). |
| Multilevel controls with computable integral | **PROVEN:** Equation (5) supplies such a scheme. Only \(G_0\) needs an analytic integral; every other term is an unbiased sampled correction. Its error and charged cost are (6) and the preceding cost bound. |
| Designs in input directions | **PROVEN UNDER C3:** Positive fixed designs have bounded gain for polynomial sample counts, using the existing even-power-tail bound. **CONJECTURE:** C3 remains unresolved at finite width. Signed or weight-adaptive designs require separate analysis; fixed-design kernel bounds do not cover the estimator above. |

**HEURISTIC, numerical check:** I ran a read-only pilot with \(L=n/4\), 96 unconditioned networks per size, and 32 directions per network. At \(n=64,128,256,512\), estimated ideal multilevel variance-times-cost ratios relative to schoolbook MC were \(3.05,2.59,1.94,1.49\), excluding setup. Thus this pilot demonstrates no practical win yet. The cheapest next test is the same paired level-increment measurement at larger widths, followed by a fully charged estimator run; no accurate reference mean is needed to estimate the increment variances.

2. **PROVEN: bounded aspect ratio, kink area, and mean jump alone do not imply a positive universal certificate lower bound.** For \(f(u)=(a^\top u)_+\), two hemispherical cells give exact affine bounds despite a nonzero kink area and jump. An axis-aligned hinge similarly admits exact gnomonic box partitions. Closely spaced opposite jumps also defeat an area-only argument. Adaptive alignment and cancellation must be excluded explicitly.

**PROVEN: there is an exact finite-width calculation of the expected regular kink geometry.** Normalize hypersurface area by \(|S^{n-1}|\), put \(d=n-1\), and define
\[
a_d=\frac{|S^{d-1}|}{|S^d|},\quad
p_n=1-(n+1)2^{-n},\quad s_n=1-2^{-n}.
\]
A *birth facet* is a regular preactivation zero inside an earlier activation region, with nonzero preceding activation. Zero-activation interiors are excluded. Distinct birth facets overlap only on area-null sets almost surely.

For the final vector output’s regular kink set \(\Sigma\),
\[
\boxed{
E\mathcal A(\Sigma)
=n a_d\sum_{\ell=1}^{L}p_n^{\ell-1}s_n^{L-\ell}
=\Theta_\alpha(n^{3/2}L).
}
\tag{7}
\]
The factor \(p_n\) follows by conditioning on the number \(K\sim\mathrm{Bin}(n,1/2)\) of active units: for \(K\ge2\), normalized tangent length has mean multiplier
\[
E\chi_{K-1}\,E\chi_K^{-1}=1;
\]
for \(K\le1\), it is zero. Coarea then gives the birth area. A nonzero vector derivative survives each suffix layer with probability \(s_n\).

**PROVEN: the area-weighted mean gradient jump is of constant order in Euclidean norm.** Write the Jacobian jump as \(b\otimes\nu\), with unit normal \(\nu\), and let \(J_2=\|b\|_2\), \(J_1=\|b\|_1\). Then, at \(L=\alpha n\),
\[
\boxed{
E\int_\Sigma J_2\,da=\Theta_\alpha(n^{3/2}L),\qquad
E\int_\Sigma J_1\,da=\Theta_\alpha(n^2L).
}
\tag{8}
\]
Here is a quantitative derivation. With
\[
t_n=\sqrt{2/n}\,E[\chi_K1_{\{K\ge2\}}]
=1-\frac5{8n}+o(n^{-1}),
\]
the expected sum of birth-gradient magnitudes at layer \(\ell\) is
\[
d\sqrt{n/\pi}\,t_n^{\ell-1}.
\]
For propagation through \(s\) suffix layers, the derivative of a unit perturbation satisfies
\[
E\|b_s\|_2^2=1,\qquad
E\|b_s\|_2^4=(1+5/n)^s,
\]
hence
\[
(1+5/n)^{-s/2}\le E\|b_s\|_2\le1.
\]
One further layer gives \(E\|b_s\|_1=\sqrt{n/\pi}\,E\|b_{s-1}\|_2\). These establish (8). The resulting Palm mean jump is \(\Theta_\alpha(1)\) in \(\ell_2\), or \(\Theta_\alpha(\sqrt n)\) in \(\ell_1\); this is not a median or high-probability statement.

**PROVEN UNDER the following nonalignment and isolation hypotheses: an adaptive partition lower bound follows.** Use the code’s summed coordinate width
\[
\Gamma=\sum_i\int_{C_i}\sum_j(U_{ij}-L_{ij}).
\]
Include the Gaussian radius factor in this definition. Assume \(n>2\), and that each relevant cell has a selected kink patch of normalized area \(a_i\), scale \(h_i\), and jump at least \(j_i\), satisfying:

- The patch has a two-sided neighborhood inside the cell on which the network is one affine map plus one hinge, with jump comparable to \(j_i\).
- The cut is quantitatively balanced; chart and integration-density distortion are controlled.
- Uniformly,
  \[
  a_i\le C_d h_i^{d-1},\qquad
  \Gamma_i\ge c_d m_n j_i a_i h_i^2.
  \tag{9}
  \]
- The selected patches capture the stated amount of kink geometry. Aspect ratio alone does not guarantee this.

**PROVEN:** The local inequality in (9) holds for an isolated hinge through the center of a ball: affine upper and lower bounds have center gap at least \(j_i h_i/2\). Integrating their affine gap over the ball proves (9). For an undistorted ball the dimensional factor is of order \(d^{-1/2}\); poorer cuts or chart distortion must be charged explicitly.

**PROVEN UNDER (9):** Put \(q=(d-1)/(d+1)\) and \(M_q=\sum_i j_i^q a_i\). Eliminating \(h_i\), then applying the power-mean inequality, gives
\[
\boxed{
\Gamma\ge
c_d C_d^{-2/(d-1)}m_n
M_q^{(d+1)/(d-1)}K^{-2/(d-1)}.
}
\tag{10}
\]
Indeed, each cell contributes at least a constant times
\(m_nj_i a_i^{(d+1)/(d-1)}\). For a quasiuniform mesh, the stronger geometric restriction instead gives
\[
\boxed{
\Gamma\gtrsim c_d m_n
\left(\sum_i j_i a_i\right)K^{-2/d}.
}
\tag{11}
\]
Thus adaptive concentration near a hypersurface changes the denominator from \(d\) to \(d-1\); it cannot simply be treated as a quasiuniform mesh.

**PROVEN UNDER uniform versions of these hypotheses:** For a fixed cell budget, Jensen permits replacing \(M_q^{1/q}\) in the expected-width bound by \((EM_q)^{1/q}\). A certificate of expected width at most \(\varepsilon\sqrt V\) therefore requires
\[
\boxed{
K\ge
\left[
\frac{c_d C_d^{-2/(d-1)}m_n(EM_q)^{1/q}}
{\varepsilon\sqrt V}
\right]^{(d-1)/2}.
}
\tag{12}
\]
A high-probability conclusion additionally needs lower-tail control of the captured geometry.

**PROVEN UNDER additional dimension-uniform capture assumptions:** If the selected patches retain a fixed fraction of (7), have \(J_1\gtrsim_\alpha\sqrt n\), and satisfy \(c_d\gtrsim n^{-1/2}\), \(C_d=O(1)\), then (4) and (12) give, up to \(\alpha\)-dependent constants,
\[
K\gtrsim
\left(\frac{n^3}{\varepsilon}\right)^{(n-2)/2}.
\]
For Euclidean rather than summed-coordinate certificate width, the corresponding sufficient lower-bound expression replaces \(n^3\) by \(n^{5/2}\).

**CONJECTURE:** Those capture and isolation hypotheses hold at the resolutions relevant to adaptive B&B. They are especially unproved for coarse cells containing many interacting kinks. Therefore the conditional exponential cell bound does **not** establish C2 for `lib/bnb.py`; its numerical moment-enclosure limitation from round 2 also remains separate.
