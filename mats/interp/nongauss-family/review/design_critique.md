# GPT-6 design critique (gpt-6-astra via codex-any exec, high effort), 29 Oct 2026 ~17:20Z

Prompt: review/prompt_design.md. Raw transcript: review/design_critique.raw.

The radial reduction is useful, but the design needs correction before preregistration. The exploration measures the wrong next-layer gates, the noisy gnomonic law is not globally well defined, and the proposed composition experiment has a noise floor above its target.

1. **[critical] The exploration omits a ReLU.** In [explore_shape.py:71](/home/agent/workspaces/mats/work/nongauss-family/scratch/explore_shape.py:71), `T = (u + D) @ Wn.T` should use `ReLU(u + D)` to describe the next pre-activation without noise. The comparison variable `Zn` correctly includes that ReLU. Consequently, the reported comparison changes both scale and network map; it does not isolate scale mixing. A read-only check on 8,192 paths of net 112 changed near-boundary counts from **36 to 44 at layer 32**, and **15 to 17 at layer 48**. Recompute the gate census, gate-direction coverage, and associated skew/kurtosis comparisons.

2. **[major] The homogeneity identity is correct under its stated positivity assumption; the noisy extension needs a separate test.** For \(m=E\rho>0\), \(d\nu=\rho\,dP/m\), and positively homogeneous \(f\),
   \[
   E f(z)=mE_\nu f(u+\delta).
   \]
   Independent \(\rho'\) works whenever \(E\rho'=m\); its particular distribution is irrelevant **for noiseless homogeneous observables**, though it changes variances and other observables. Choosing \(u=Ez/\|Ez\|\) gives \(E_\nu\delta=0\), provided the chart exists.

   Here the noise scales are fixed population/pilot RMS values, not pathwise RMS values. Thus \(F_s(\rho v)=\rho F_{s/\rho}(v)\), where every suffix noise scale is divided by \(\rho\). The noisy mean generally requires the joint radial-angular law. Independent radial resampling loses that information.

   At one scalar gate, Gaussian smoothing changes the mean by at most \(s/\sqrt{2\pi}\), or \(3.99\times10^{-4}r\) for \(s=0.001r\). With a smooth pre-activation density, the leading correction is instead \(s^2p(0)/2\). Neither supplies a suffix nMSE bound. **The supplied evidence does not establish the error at \(\sigma=0.001\).** Add an exact-angular empirical control with independent radius, versus the original joint samples, using paired suffix noise and a noiseless counterpart.

3. **[major] The gnomonic chart is not globally valid, even if every observed deep path has positive \(\rho\).** With nonzero Gaussian noise and full-rank weights, the pre-activation distribution has full support. Therefore \(P(u^\top z\leq0)>0\). Moreover, restricting to \(\rho>0\) does not fix its population covariance: generically,
   \[
   E_{\nu_+}\|\delta\|^2\ \propto\
   E\!\left[\frac{\|z_\perp\|^2}{\rho}\mathbf1_{\rho>0}\right]=\infty
   \]
   because the density is positive near \(\rho=0\). This may be numerically invisible but invalidates an exact finite-covariance family definition.

   The clamp at line 45 hides the problem and breaks orthogonality on affected paths. Use a norm-weighted sphere, multiple bounded charts, or an explicitly truncated chart with a separately represented remainder. A post-ReLU simplex chart, \(\rho=\sum_j a_j\), also avoids negative radii; handle any zero-state mass separately. The mean direction is a convenient centre, not a positivity guarantee or an established optimal centre.

4. **[critical] The polar push requires a change of measure that the sketch omits.** Noiselessly define
   \[
   g(\delta)=W\operatorname{ReLU}(u+\delta),\quad
   q(\delta)=u'^\top g(\delta),\quad
   \delta'=g(\delta)/q(\delta)-u'.
   \]
   On valid positive charts,
   \[
   m'=mE_\nu q,\qquad
   E_{\nu'}\phi(\delta')=
   \frac{E_\nu[q\,\phi(\delta')]}{E_\nu q}.
   \]
   In particular,
   \[
   C_{\nu'}=\frac{E_\nu[gg^\top/q]}{E_\nu q}-u'u'^\top.
   \]
   Ordinary means and bivariate covariances of rectified coordinates do not determine this ratio expectation. Even a linear network map produces a fractional angular map and reweights the law. This is an additional closure problem beyond the acknowledged difficulty of pushing component marginals.

5. **[major] PI and PR are valid non-Gaussian oracle families, but their advertised properties need qualification.** Independent empirical PC marginals satisfy “no Gaussian component”; they remain sample-derived representations with \(O(N_{\rm fit}n)\) stored marginal values unless compressed. PR also qualifies, but is not “shape-free”: it imposes symmetric components with excess kurtosis \(-2\). Every linear projection has zero skewness and excess kurtosis
   \[
   -2\frac{\sum_k b_k^4}{(\sum_k b_k^2)^2}.
   \]
   PR therefore tests a specific bounded, symmetric shape.

   Arbitrary linear mixtures of independent sources are closed under linear maps. **PCA-defined independent components are not:** re-diagonalizing after a map generally rotates independent non-Gaussian sources into dependent ones. Preserve this distinction when interpreting repeated projection. Also, matching \(C_\nu\) does not imply matching the original \(z\)-covariance after independent radial resampling.

6. **[major] PC3I/PC3R do not yet specify reproducible families or retain an “exact joint core.”** Sixty-four cell probabilities preserve a coarse joint histogram, not the within-cell joint law. Specify cell boundaries, probabilities, centring, local versus global PCA, treatment of empty cells, and whether the top-three coordinates are jointly resampled or independently reconstructed. Conditional independence in global PCs can discard within-cell covariance; conditional PCA avoids that particular loss but still discards higher dependence.

   Untruncated PC3G and locally rotated PC3R can generate points outside their assigned cells. That is permissible for a mixture, but then “exact core up to bins” needs qualification. All Gaussian controls should use the same fit data and comparable conditioning rules. PC3R matching PC3I would support one coarse approximation, not establish candidate (b)’s exact remembered subspace.

7. **[major] ATTk confounds shape transfer with arbitrary PCA coordinates.** Eigenvalue ordering does not identify physical modes across layers. Nearly degenerate eigenspaces rotate; signs based on third moments become unstable near zero skewness. Whitening also amplifies poorly estimated small eigenvalues. Stable marginal cumulants do not establish stable joint shape.

   Specify \(k\), retained rank, eigenvalue regularization, and alignment before testing. A stronger attractor test transports source tangent directions through the intervening network, aligns well-conditioned subspaces using a separate paired fit set, and evaluates held-out gate stop-loss functions and suffix means. Add transfer from another weight seed after the same alignment. Successful same-net transfer alone can reflect remembered trajectory-specific structure rather than a reusable attractor.

8. **[major] The proposed nMSE subtraction is wrong for general replacement laws.** Conditional on a fitted family \(F\), independent evaluation contributes
   \[
   \frac{\operatorname{tr}\operatorname{Cov}_F(a_L)}
        {N_{\rm samples}\operatorname{tr}\operatorname{Cov}_{P}(a_L)},
   \]
   which equals \(1/N_{\rm samples}\) only when total output variances match. Estimate that variance term. The truth contribution is approximately \(2^{-22}=2.38\times10^{-7}\).

   The nominal propagation contribution is \(2^{-20}=9.54\times10^{-7}\); the empirical-fit scale is \(2^{-19}=1.91\times10^{-6}\). The latter is **not a universal floor for every fitted family**: smoothing, parameter estimation, and nonlinear fitting change its bias and variance. Fit directly with radial weights instead of first generating an additional size-biased bootstrap sample. Use independent fit replicates and path-level uncertainty estimates; correlated neurons are not independent replicates.

9. **[critical] E2 cannot test a \(10^{-5}\) bound at its current particle count without separating bootstrap error.** With approximately conserved downstream variance, 32 multinomial resampling steps contribute roughly
   \[
   32/2^{20}=3.05\times10^{-5}
   \]
   before initial fitting error. EMP measures one genealogical noise process; subtracting EMP’s scalar nMSE from another family does not remove that family’s sampling error or cross terms.

   Carry unresampled paths as the plain-Monte-Carlo control. For projection chains, use independent replicates and several particle counts to distinguish finite-particle effects from limiting projection bias. Simple bootstrap noise below \(2\times10^{-6}\) would require roughly \(1.6\times10^7\) particles under the same no-contraction approximation. Averaging replicate chains reduces variance, but finite-particle fitting bias still needs checking.

10. **[major] The depth criterion is useful as a screen, but neither sufficient nor necessary for M1.** Fixed-cut suffix growth diagnoses whether one perturbation survives. Composition diagnoses repeated approximation. Starting from oracle truth at layer 32 does not establish that a deductive method can reach that state from the input.

    Conversely, some depth growth can be compatible with M1 if width-dependent improvement compensates for it. Make at least one fixed-\(L/n\) comparison mandatory for any scaling claim, for example \(128\times32\), \(256\times64\), then \(512\times128\). Report unnormalized squared error and output variance alongside nMSE. If all candidates fail, reject these fitted families and projection rules; do not infer that fully non-Gaussian families generally fail.

11. **[major] Use numerical gates with an explicit inconclusive region.** A defensible preregistered screening rule is:

    - **Reduction control:** upper 95% confidence bound on the exact-angular independent-radius error at \(\sigma=0.001\) below \(10^{-6}\).
    - **Single cut:** evaluation-noise-corrected nMSE, retaining fitting error, below \(10^{-5}\) at primary endpoints \((32,40),(32,48),(32,64),(48,56),(48,64)\). Upper 95% bounds on \(R(32;64)-R(32;40)\) and \(R(48;64)-R(48;56)\) below \(3\times10^{-6}\).
    - **Composition:** after resolving particle-count effects, upper 95% bound below \(10^{-5}\) through layer 64; growth from output depth 48 to 64 below \(3\times10^{-6}\).
    - **Resolution:** interval half-width at most \(2\times10^{-6}\) for decisive comparisons. Pass when the upper bound clears the threshold; fail when the lower bound violates it; otherwise report inconclusive.

    Apply simultaneous uncertainty control to the declared endpoints and show all four held-out nets. These are screening thresholds, not a claim to match Kerdock. That claim requires a same-net, equal-cost benchmark near its actual \(6.7\times10^{-6}\) level.

12. **[major] Two-cut error minus the sum of single-cut errors does not isolate coherence.** Let \(e_{32}\), \(e_{48}\), and \(e_{32,48}\) be output-mean error vectors against the same population target. Write
    \[
    e_{32,48}=e_{32}+e_{48}+d.
    \]
    The residual \(d\) captures the second projection acting on an already altered distribution. Report both the noise-corrected inner product \(e_{32}^\top e_{48}\) and \(\|d\|^2\), normalized by total truth variance. Otherwise nonlinear interaction can masquerade as coherent addition. Also separate error parallel to the truth mean from orthogonal error; persistent scale bias is particularly relevant here.

13. **[major] The strongest failure mechanism is loss of gate-conditioned dependence, not insufficient marginal non-Gaussianity.** Small correlations of squared scores among five PCs constrain very little of the relevant joint law. Weak pairwise statistics can coexist with substantial conditional dependence near intersecting gate boundaries. Critical dynamics can retain the resulting signed mean bias.

    Before the full experiment, compare the exact weighted angular joint sample with independently shuffled PC scores, then with a jointly preserved top-\(k\) block for \(k=2,4,8,16\). Measure paired one-step means, rectified cross-moments, and short suffix means. This cheaply tests the exact independence assumption PI makes. Use the corrected gates and measure omitted contributions such as \(E[(-t)_+]\); \(|Et|/\operatorname{sd}(t)>1\) does not make a gate safely linear.

14. **[major] A more informative missing family is a bounded joint core with conditional non-Gaussian residuals.** One concrete candidate is
    \[
    \delta=b(y)+B(y)\eta,
    \]
    where \(y\) has a positive piecewise-polynomial joint density in a small core, and \(\eta\) has centred bounded beta components. Beta shapes accommodate skewness and platykurtosis; \(B(y)\) retains full conditional residual covariance. Choose the core using collapse directions plus poorly represented gate normals, rather than variance alone. Retain a few conditional radial bins if the noise-reduction control fails.

    Test \(k=4,8,16\) and require coverage of every materially contributing gate, not median coverage. This extends the conditional-cell idea by retaining within-cell dependence and explicit bounded shapes. It is a candidate to test, not an established improvement; residual gate switching and required core dimension may still kill it.

15. **[critical] No listed candidate currently has a demonstrated scalable deductive push.** Characteristic-function products give a scalar projection law for independent components, but finite-grid inversion is numerical, empirical characteristic functions have a construction cost, and the stated next-layer projection again omits the intervening ReLU. Updating the size-biased angular law remains harder than computing those scalar laws. Third/fourth cumulants do not specify a unique valid distribution; exponentiating a truncated higher-order cumulant series generally does not yield a valid characteristic function.

    The stated pair-integration cost is already problematic: with \(n=256\), \(n_{\rm rel}=30\), \(G=128\), it is about **\(3.2\times10^{10}\) grid-factor operations per layer**, or \(10^{12}\) over 32 layers, before cells and other work. ATT has no Gaussian-style bivariate formulas merely because its shape is fixed; integrating an empirical joint shape still uses its stored points.

    A plausible sample-free route requires a small joint latent dimension, deterministic integration over gate-split regions, and a controlled residual approximation. Demonstrate one complete push, including reweighting and shape update, with error and counted cost before commissioning the large oracle sweep. “Polynomial in width” is not evidence of feasibility at 10M+ parameters.

Checked on 2026-10-29 against the supplied documents, source, stored exploration output, and the read-only 8,192-path calculation. No files were modified. Durable recall was unavailable because its SSH invocation failed.
