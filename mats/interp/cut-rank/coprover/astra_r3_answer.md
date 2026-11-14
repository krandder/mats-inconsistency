codex
E is PROVEN after correcting the harmonic count. E′ is stronger than stated: for ray fronts, every fixed finite depth has the same precision exponent 2(n−1)/3. The extension to Gaussian fronts remains open.

1. Verdict on E and E′

Theorem E: PROVEN, with these qualifications.

Assume n ≥ 2, 0 < vR < ∞, and w ~ N(0,I_n). The covariance is the ensemble covariance; a single realized finite-width back does not have the exact arc-cosine kernel.

Your squared singular values nη_l² and Vbar = vR/2 are correct. For even l,

  N(n,l)nη_l² ~ C_n l^−4,
  C_n = n 2^(n−1) Γ(n/2)² / [π² Γ(n−1)].

Consequently,

  tail beyond D ~ (C_n/6)D^−3.

The nonzero harmonic count through an even degree D is

  n + binom(n+D−1,n−1) ~ D^(n−1)/(n−1)!,

not 2D^(n−1)/(n−1)!. Odd degrees above one vanish. A partially included boundary degree changes the count by only O_n(D^(n−2)). Thus the sharper statement is

  r(ε) ~ [1/(n−1)!] [C_n/(3ε²vR)]^((n−1)/3).

Your exponent and the threshold n ≥ 5 are correct. The numerical log agrees; its largest ranks use floating-point counts, not exact integer arithmetic.

Theorem E′:

(i) PROVEN. Write q = 2^−(d−1). B2 gives T_d(D) ≥ qT_1(D), hence ≥ c_n qD^−3 for sufficiently large D at fixed n.

(ii) PROVEN for l ≥ 1, using the normalized kernel κ(1)=1:

  λ_l ≤ (n−1)/[l(l+n−2)N(n,l)] = O_n(l^−n).

(iii) The block lemma is PROVEN, but the displayed substitution drops a factor q. With δ = ε²Vbar, that calculation gives

  r ≥ C_n q [qβ/δ]^((n−3)/3).

Your claimed weaker bound nevertheless follows from a stronger argument:

  K_d − qK_1 is positive semidefinite.

Averaging both arguments against front laws preserves this ordering. Therefore the front covariance operators satisfy G_d ≥ qG_1. Min–max gives, for every rank s,

  Err_d,opt(s) ≥ q Err_1,opt(s).

This applies to arbitrary front laws, with absolute error budgets. For uniform ray fronts it yields

  r_d(ε) ≥ c_n [qβ/(ε²Vbar)]^((n−1)/3).

For these rays β/Vbar = 1/vR under consistent normalization. Thus n ≥ 5 already suffices at every fixed finite depth.

Your upper bound is valid but loose. The gradient identity permits the smallest m satisfying

  β(n−1)/[(m+1)(m+n−1)] ≤ δ.

At fixed n, m = O_n(sqrt(β/δ)), giving precision exponent at most n−1. For general, unsynchronized fronts, replace β by Q = E||h||²/n; Q=1 for the original He ensemble.

There is also a matching upper bound for fixed finite architectures:

  T_d(D) = O_(n,d)(D^−3).

Proof: each realized homogeneous ReLU network is linear on finitely many polyhedral cones. Let H bound the number of supporting hyperplanes of their boundaries, and L_b its Lipschitz constant on the sphere. A uniformly rotated geodesic segment of length 2θ crosses a boundary with probability at most O(Hθ). Its central second difference is O(L_bθ) on crossings and O(L_bθ²) otherwise. Hence its averaged squared second difference is O(H E[L_b²] θ³), with E[L_b²] finite.

By isotropy that same quantity equals

  Σ_l a_l [6 − 8P_l(cos θ) + 2P_l(cos 2θ)].

For n ≥ 3, the standard normalized Gegenbauer bound makes the bracket bounded below uniformly for l ≥ D when θ=A_n/D with A_n sufficiently large. This proves the tail bound. For n=2, average θ over a comparable interval and use Fourier modes.

Harmonic truncation now gives the matching upper rank bound. Therefore:

  r_d(ε) = Θ_(n,d)((ε²vR)^−(n−1)/3)

for uniform ray fronts and every fixed finite depth. The constants need not be uniform in depth.

(iv) PROVEN UNDER THE PREVIOUS FACE HYPOTHESES. On a k-dimensional coordinate face, replace the harmonic dimension by k; the hidden width remains n. For a squared-radial-mean-weighted direction density bounded by ρ relative to uniform orthant measure, the lower bound loses a factor ρ2^k.

When conditioning on faces, include their probability and radial energy in the global error budget. After radial weighting, the face-size law is not literally Binomial(n,1/2), although its mass remains near n/2 in the fresh-Gaussian-layer construction.

Two qualifications matter:

  • This proves the exact synchronized model followed by a fresh Gaussian layer, not arbitrary approximately synchronized fronts.
  • At fixed n and ε→0, rare larger faces can dominate. “k≈n/2” is a useful large-n restriction, not necessarily the ultimate precision exponent.

These are bounds on linear-summary dimension. They do not establish runtime lower bounds for unrestricted estimators.

2. Uniformity in depth

NEEDS A UNIFORM HIGH-FREQUENCY ESTIMATE.

B3 preserves expected local jump energy. It does not control cancellations between facets at a specified harmonic resolution, or how quickly the asymptotic regime begins. The piecewise-linear upper proof above also has architecture-dependent constants.

A depth-independent lower bound at every fixed degree cutoff is untenable: the infinite-width kernels approach the constant kernel as depth increases, so their nonconstant energy vanishes.

The weakest additional hypothesis needed for your original E′ argument is only at its chosen cutoff D_ε:

  T_(n,d)(D_ε) ≥ c T_(n,1)(D_ε),

with c bounded away from zero uniformly along the stated width/depth/accuracy sequence. Combined with the gradient bound, this removes q from the lower bound with exponent 2(n−3)/3.

To remove q from the sharper exponent 2(n−1)/3, a sufficient stronger conjecture is comparison of the optimally ordered spectral tails:

  Err_d,opt(s) ≥ c Err_1,opt(s)

at the relevant ranks. A degree-tail comparison alone does not imply this stronger statement.

Bietti–Bach (2021), Corollary 2, supports this direction for infinite-width kernels at fixed depth. It does not supply the finite-width, growing-depth interchange. Their near-diagonal expansion suggests resolving angular scales much smaller than 1/d; a natural harmonic condition is

  D(D+n−2) / [(n−1)d²] ≫ 1.

Neither that onset condition nor the required finite-width comparison is proved here. “For sufficiently large D depending arbitrarily on d” would give an eventual statement, not a useful uniform precision window.

3. Data and spread fronts

Checked against scratch/tabulate.py and the JSON files at 2026-10-31 00:00 UTC.

The width and fixed-ratio ranks you quote at F′=256 are correct. The depth series 117/89/60 uses F=512; at F′=256 it is 86/67/48. For the quoted configurations, the full-F r_in secant exponents between 0.1 and 0.03 are approximately 0.46–0.94.

None contradicts the theory:

  • These are Gaussian-network fronts, not the ray ensemble.
  • Fixed-depth asymptotics give no guarantee that ε=0.03 is sufficiently small.
  • Population rank is capped empirically by both the sampled fronts and sampled back/output columns.
  • Several held-out ranks are unresolved: 129 or 257 means the tested rank range was exhausted, not a measured rank of that value.

The cross-split Gram can have negative eigenvalues; its optimized tail is biased downward. Both identical-front controls have negative tail_in(1), so a literal signed “three times control tail” is not a valid positive noise threshold. Independent repeats should estimate its uncertainty. Those controls test estimation noise; they do not test an ensemble of distinct synchronized fronts.

For spread fronts, there is no universal lower exponent. For example, with one back layer and isotropic front multipliers b_l ≍ l^−s,

  r(ε) ≍ ε^−[2(n−1)/(3+2s)].

Heat smoothing instead gives growth of order (log(1/ε))^((n−1)/2). These are specified smoothing models, not conclusions about Gaussian fronts.

A useful general sufficient condition is PROVEN: if the radially weighted front laws have densities q_f relative to a common angular probability measure μ, with

  E_f ||q_f||²_L²(μ) < ∞,

then the cut operator factors through two Hilbert–Schmidt operators. Its singular values are summable, implying

  r(ε) = o(ε^−2)

for fixed positive Vbar.

This condition is not automatic for Gaussian ReLU fronts. One-active-neuron input cones have positive Gaussian mass and can produce ray atoms; after further front layers, those rays can have random directions. Smooth-cap intuition can therefore miss the eventual tail.

Finally, the proved polynomial upper bounds rule out growth faster than every fixed power of 1/ε at fixed finite n,c,d. “Exponential in precision bits” remains possible, but sampling itself is exponential in those bits.

4. Conjecture ledger

Each claim must specify either a fixed (n,c,d) precision limit or a joint sequence n,c,d,ε. Finite experiments can refute specified quantitative bounds; they cannot refute an asymptotic claim whose constants and onset remain unspecified.

4a. Front averaging preserves sufficiently dispersed energy.
Formulation: ensemble one-way linear. NEW CONJECTURE; SINGLE BOTTLENECK.

For the original Gaussian fronts, let H_B be the block Gram operator from round 1. The sufficient certificate is

  tr H_B ≥ 2ε²Vbar,
  tr H_B / ||H_B|| ≥ 2g(ε),
  ε²g(ε) → ∞.

Then r(ε) ≥ g(ε), which beats sampling’s summary count. To claim a dimension-dependent power, specify g(ε)=Cε^−an and its regime. No lower bound on every individual harmonic direction is necessary.

Evidence for: the ray theorem, persistent gates, and growing measured ranks.
Evidence against: front integration smooths; scale-only fronts have rank one; the common-L²-density condition above gives a sampling-compatible regime.
Cheapest test: increase both independent fronts and backs, control input-integration noise, and estimate retained block energy and effective rank separately.
Status: no established theorem or named equivalent open problem is identified here.

4b. Approximate synchronization transfers the ray theorem.
Formulation: ensemble one-way linear; optional route. NEW CONJECTURE.

Construct a ray surrogate with the required weighted face-direction laws and prove

  ||M−M_tilde||₂ = o(ε sqrt(Vbar)).

The proved triangle inequality then transfers a ray lower bound with a vanishing relative change in tolerance.

Evidence for: angular synchronization.
Evidence against: angular diagnostics omit radial energy and exceptional fronts; a fixed nonzero surrogate error cannot justify ε→0 at fixed front depth.
Cheapest test: measure the paired, energy-weighted surrogate residual against ε²Vbar.
This route is unnecessary if 4a is proved directly.

4c. Remove depth-dependent precision losses.
Formulation: ensemble one-way linear. NEW FINITE-WIDTH CONJECTURE, related to the known infinite-width result.

Use the cutoff or ordered-tail comparison in Section 2.
Evidence for: B3 and Bietti–Bach.
Evidence against: collapse at fixed resolution and uncontrolled onset of the high-frequency regime.
Cheapest test: measure T_d(D)/T_1(D) across depth and resolution, with a predeclared onset and lower constant.
This is unnecessary for the fixed-depth exponent theorem.

4d. A typical fixed back remains hard with rich readouts.
Formulation: fixed back with rich readouts. SEPARATE NEW CONJECTURE.

Specify a probability law μ_b over an infinite readout family q, and define

  T_b(f,q) = E_x q(g_b(h_f(x))).

Require, with probability 1−o(1) over b, a block certificate analogous to 4a for T_b, using its own variance normalization.

Evidence for: nontrivial measured R3 ranks.
Evidence against: an ensemble bound can distribute complexity across backs; finitely many readouts impose a finite rank cap. Threshold readouts can also introduce hardness themselves.
Cheapest test: freeze independent backs, increase both fronts and readouts, and include an affine-back control.
There is no proved ensemble-to-fixed-back implication.

4e. Surviving Jacobian directions imply nonlinear local complexity.
Formulation: local slice. SEPARATE NEW CONJECTURE.

For a specified slice distribution and radius, require the same retained-energy/effective-rank certificate for the local evaluation operator.

Evidence for: gate crossings can introduce nonlinear components.
Evidence against: a slice contained in one activation cell is affine and has rank at most k+1, regardless of Jacobian survival.
Cheapest test: vary radius and measure the residual after affine projection together with its spectral spread.

A back-dependent slice does not automatically yield a front-only summary lower bound. That missing reduction cannot be supplied by calling the directions “backward-selected.” No comparison with k* is needed for the proved ray precision theorem.

4f. Ensemble error implies failure on almost every fixed net.
Formulation: a predetermined one-way protocol, evaluated on random networks. SEPARATE NEW CONJECTURE.

Let E_s be the optimal population error at rank s. A sufficient small-ball condition, weaker than concentration, is

  P[ Z_P/V ≤ t E_s/Vbar ] ≤ h(t)+o(1),
  h(t)→0 as t→0,

for the specified protocol family, where Z_P is its per-network squared error. If E_s/(ε²Vbar)→∞, this implies failure probability tending to one. Normalize by E_s, not E Z_P: arbitrary protocols can inflate their own mean error on rare events.

Evidence for: the measured failure fractions.
Evidence against: rare networks can carry ensemble energy; norm intermittency and localized spectral modes obstruct transfer.
Cheapest test: evaluate genuinely held-out protocols and inspect the lower tail of Z_P/V across independent networks.

Probability one at fixed finite n is obstructed by positive-probability dead-network events. The viable target is probability 1−o(1) in a stated regime, for each predetermined protocol. “On most networks every retrospectively selected protocol fails” remains false.
