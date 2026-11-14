# Coverage map: ARC's estimators against the cut-rank-ext bounds (revised after co-prover round 2, 11:46Z, and after the
# final GPT-6 review findings 7-8, 12:16Z)

Facts and sources: `notes/arc_estimators.md` (entries E1-E16, opened 10:49-11:01Z, tags there). Code-level facts: ARC's
released repository `mlp_cumulant_propagation` as vendored at `work/arc-noisy/ext/mlp_cumulant_propagation`
(src/mlp_kprop: kprop_harmonic.py, wick.py). Corrections from gpt-6-astra round 2 are applied (PROVENANCE R2.7).

## Classes (genuine ensemble at fixed n, c >= 3, d; eps -> 0; l = log(1/eps); a = 2(n-1)/3; p = n^2 c; q = n^2 d)

Budgets: a class with an eps-INDEPENDENT bound on the decoder's amplitude, Lipschitz constant or derivatives has an
error floor for every s (the target is unbounded under Gaussian weights; astra A3, R2.1). The classes below therefore let
the budget grow subpolynomially in 1/eps (polynomially where stated).

| class | lower bound | upper bound | status, source |
|---|---|---|---|
| L: s front numbers read linearly | C eps^{-a} | Theta | PROVEN (cut-rank K4/K5) |
| P: readout polynomial of degree <= D in s reals (B backward bits allowed, non-adaptive) | 2^B C(s+D, D) >= C eps^{-a}; B = 0: s >= (D! C)^{1/D} eps^{-a/D} - D | | PROVEN (Thm B; astra 1.e) |
| P-multi: t turns, every message AND the final output polynomial of degree <= D in earlier messages, no branching | C_msg >= C^{1/(DS)} eps^{-a/(DS)} - D, S = 1 + D + ... + D^t | | PROVEN (astra A8) |
| An: readout holomorphic on a fixed polydisc (radius ratio rho), amplitude <= eps^{-kappa} | (c(a, rho, kappa) - o(1)) l | (p + o(1)) l / ln 2 at rho = 1/2, front and back clipped, kappa >= p log2 3 (multilinear decoding of quantised weights) | lower PROVEN (Claude B1, corrected and strengthened by astra R2.2); upper PROVEN (Claude Addendum 2, checked by the final review) |
| Bits: any rounds, any decoder | n log2(1/eps) - O(1); one-way: (n(n+1)/2) log2(1/eps) - O(1) | (p + o(1)) log2(1/eps) one-way; (q + n + o(1)) log2(1/eps) two-way | PROVEN (astra A4, R2.4; Claude A, A2) |
| Ck: readout C^k, derivative budget polylog | max(k a, n(n+1)/2)(1 - o(1)) | (k+1) p | PROVEN (Claude B2; astra R2.3; Claude Addendum 1) |
| Lip: decoder Lipschitz, log(radius x constant) = o(l) | (n(n+1)/2)(1 - o(1)) | p (clipped weights) | PROVEN (astra R2.4 + C4; astra A2) |
| C0: continuous decoder, measurable encoder | none | 1, exact (space-filling curve; compact source or unbounded scalar domain) | PROVEN (Claude C3) |
| Mom: moments through order K, any continuous closure | Err >= e_K^2 (information limit) | e_K = 0 for K >= Q(n, c), non-effective | PROVEN (astra A5, A7) |
| ARC's basic closure (Wu et al. eqs. (13)-(14)) fed EXACT cut cumulants, one back layer | bias floor b_K >= c exp(-C K log(K+2)) > 0 for every K, so K log K >~ l and a symmetric state of size Omega((l / log l)^n) | | PROVEN (astra R2.5), oracle one-step only |
| BF-real: back-and-forth with real messages | none | 2 rounds, q + n reals, exact | PROVEN (both) |
| BF-span: front answers E_{P_f} e, e in a fixed N-dim space, back non-adaptive, linear readout | N >= C eps^{-a} | | PROVEN (Claude Thm D) |
| BF-mom: every front answer a function of the order-K moments | Err >= e_K^2 | | PROVEN (astra A7) |
| Stream: layer-by-layer state of size s | the class of the composed readout from the cut state (hypotheses on the actual cut state and composed map): polynomial of degree D: s >= c eps^{-a/D}; C^k: as Ck; Lipschitz: n(n+1)/2 | | PROVEN UNDER the inherited hypotheses (Claude Sec. 5) |

## ARC's estimators for expectations of random networks (and closest relatives)

| ARC estimator (source) | structure | class | what the bounds give | what keeps it out |
|---|---|---|---|---|
| Mean propagation, arithmetic circuits (CNX22 App. D.2) | one mean per wire, sum/product gates | not a ReLU net | n/a | different model |
| K = 1 cumulant propagation (Wu et al. 2026, mean plus one variance trace) | layer-by-layer, state n + 1, Hermite step | Stream, continuous on positive variance | n + 1 < n(n+1)/2 for n >= 3: IF its readout were Lipschitz with subpolynomial constants, the Lipschitz bound would rule out arbitrary accuracy; it is not uniformly Lipschitz, so this is conditional | nonlinear Hermite step |
| Mean-only ReLU propagation with the cut variance set to zero | m <- relu(W m) | Lip | exact on the K5 ray event (C2); ARC's own K = 2 code fed (E h_c, 0) reproduces it to 2e-5 (`logs/arc_zero_cov_check.log`) | the rays behind K5 are easy for it |
| Covariance propagation for ReLU nets, full Gaussian moment matching (Xu 2024 Method 1; G. Wu thesis 2.1); CNX22 D.3 is the arithmetic-circuit version (product gates), a different model | (mu, Sigma) per layer, exact Gaussian ReLU moments | Stream, continuous on PSD states | dimension bounds only conditionally (no uniform regularity); Bits only for a finite-precision implementation | nonlinear closure |
| Wu et al. K = 2 (Alg. 2: off-diagonal Sigma_ij c_i c_j, leading Hermite term); WhestBench covariance baseline (implementation not verified) | (mu, Sigma) per layer | Stream, continuous on PSD states, not uniformly Lipschitz in Sigma | same; plus the oracle bias floor R2.5 at fixed K | as above |
| Basic and augmented K-th order cumulant propagation (Wu et al. Sec. 4.1-4.2, S.4.1) | cumulants through K (~ n^K/K! entries); Hermite diagram sums with per-unit Gaussian centring; pdf/cdf coefficients with sigma^{-(k-1)} factors | Stream, locally analytic where variances are positive; no uniform radius or budget | for ReLU: dimension bounds only conditionally (no uniform regularity), Bits only for a finite-precision implementation; the oracle one-step bias floor R2.5 (basic variant) | non-uniform regularity; a full-algorithm bound would have to control propagated errors (open) |
| Factorised cumulant propagation (Wu et al. Sec. 4.3) | top tensor in O(n) factors, list grows with depth | Stream with growing state | as above | as above; the paper/code discrepancy for factorised-augmented is unresolved (repo TODO.md) |
| Polynomial-activation kprop (Wu et al. Thm 5.1/5.2; S.2 polynomial algorithm) | polynomial updates | P-multi | vacuous: for polynomial activations the cut operator has finite rank | no conflict with ARC's theorems |
| Ablated kprop, mechanistic distillation, low-probability variant (Wu et al. App. E, H, G) | diagnostic; training objective; tail probability | other estimands | n/a | n/a |
| Deduction-projection with harmonic projection (Hilton halfspace note) | carried function projected onto a FIXED, instance-independent subspace; linear deductions; linear output | L at every cut | covered; for ReLU MLPs the same pattern would need >= C eps^{-2(n-1)/3} numbers at fixed n, more than 1/eps^2 for n >= 5 | covered; proposed only for halfspaces, products, permanents. Its own analysis uses the largest omitted eigenvalue against the head's L2 norm, a different normalisation from our ensemble tail |
| Sketching theorem (Hilton note Thm 3.1, informal) | projection onto m eigenfunctions | L (m reals; "O(1/eps^2) bits" needs precision assumptions) | consistent with our bounds (other normalisation) | n/a |
| Gelfand-pair and permanent deduction-projection (W. Wu notes) | linear projection onto fixed irreps; least squares | L, for their problems | covered for their own cut spectra (the ReLU exponent does not transfer) | n/a |
| Random-walk Fourier sketching (Winer note) | backward random walks choose an instance-dependent support; forward linear propagation | back-and-forth, sampled, adaptive; among the sources inspected the only ARC estimator with a backward component | BF-bits only once the whole tail-dependent naming transcript is counted | adaptive nonlinear support selection; different estimand |
| Independent-gate, sparse-covariance, selected-term estimators (CNX22 Sec. 6.2, D.4, D.7); arithmetic-circuit cumulant propagation (CNX22 D.5-D.6, Algorithm 1) | Boolean circuits, arithmetic circuits, linear circuits, permanents | other problems | n/a | n/a |
| #SAT estimators, Methods 1-3 (Xu 2024 appendix) | independence, conditioning on the number of true variables, regression merge | other problems | n/a | n/a |
| Pairing structures, permanent sums, regression merges (LIE 2024 Sec. 4-5) | other problems | n/a | n/a | n/a |
| Activation modelling with Gaussian or independent-feature (VAE) classes (Xu 2024; LIE Sec. 7.3; Christiano-Matolcsi-Robinson) | layer-by-layer fitted models | Stream (fitted, not continuous in general) | only Bits | proposals |
| GMHP (G. Wu thesis Sec. 6.4) | <= C Gaussian components, split per ReLU, pruned by weight | Stream with discontinuous pruning | Bits only for a specified finite-precision implementation | discontinuous selection; RNN accuracy, a different estimand |
| Region enumeration and Girard volumes (AlgZoo 2026; Hilton 2024-12; thesis 6.3) | exact per region, case counts grow with depth | adaptive case splitting (enumeration itself need not make the estimator discontinuous) | none | different estimand; unbounded case counts |
| Bin propagation, max-of-k (G. Wu 2024) | trained attention model | n/a | n/a | n/a |
| ITGIS, MHIS, QLD, GLD (G. Wu and Hilton 2024/25) | sampling, adaptive proposals | randomised, sampling | sampling is a Lipschitz readout of sampled cut states; its growing state meets the dimension bounds rather than evading them | different estimand and access |
| Matching sampling principle with explanations pi, |pi| = O(|theta|), train-and-explain (Competing with sampling 2025) | advice computed from all weights | unrestricted joint computation | none: sending all weights (or the answer vector) already works in the unrestricted model | outside every size bound; needs a running-time bound in an explicit model |
| Shin-tensor advice for trained two-layer MLPs; compression approach (Competing with sampling, proposed) | advice / joint computation | joint computation | none | unfinished proposals |
| Hybrids that randomise what to analyse (Winer 2026) | statements, no algorithm | n/a | n/a | n/a |

The fact sheet's unopened sources (listed there) mean this list is not exhaustive.

## What stays uncovered, and its distance from ARC's proposals

- Unbounded joint computation (ARC's MSP formalism with explanations): no communication or summary-size bound can
  constrain it. This is the regime ARC states its principle in.
- ARC's actual ReLU cumulant propagation (fixed-size layer-by-layer state, nonlinear updates without uniform
  regularity): the dimension bounds apply only conditionally on regularity it lacks, and meeting a state-count
  threshold would not by itself give arbitrary accuracy. What is proven against its specific closure
  is the oracle one-step bias floor (R2.5): every fixed K is biased on the ray event, so K must grow at least like
  l/log l. A precision-power lower bound for it would need a polynomial rate b_K >= c K^{-beta} plus control of
  propagated-state errors (open).
- Adaptive case splitting with unbounded case counts (region enumeration, mixtures): outside every regularity class;
  ARC uses it only for tiny trained models.
- Sampling and hybrids: different estimands or access in ARC's uses; plain sampling is a Lipschitz readout of sampled
  cut states and meets the dimension bounds rather than evading them.
- Covered: linear deduction-projection (ARC's halfspace, product and permanent notes) and anything that reads a front
  summary linearly or with bounded polynomial degree. ARC has not proposed such an estimator for ReLU MLPs.
