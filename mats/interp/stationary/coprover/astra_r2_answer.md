codex
PROVEN (file audit): I audited the frozen v1, whose archived copy matches hash `b88f4579...`, and the unchanged Part A preregistration. FORMAL.md became v2 during this review. No files were edited. Checks completed 2026-10-31, approximately 13:50Z.

PROVEN (numerical scoring): PA3's two numerical checks and PA10(b) should pass under the registered median-over-nets rule. The radius-history model still misses angular covariance by factors up to 11.57.

HEURISTIC: a finite-angle closure predicts radial CV of order `n^(-1/3)` and explains the observed constants reasonably well. It also suggests a different mechanism from a stationary, diagonal Lyapunov covariance.

1. Audit of v1

PROVEN: Lemma 1, the annealed scalar radius transition, and positive-sigma pathwise scale invariance survive the audit. The following statements need correction or qualification.

| Location | Label | Correction |
|---|---|---|
| Setting: Model R uses a clean pilot | PROVEN | `lib/st.py` uses a noisy pilot, recomputed for each sigma. The evaluated noise schedule therefore changes with sigma beyond multiplication by sigma. |
| Lemma 2: PROVEN with compactification left unchecked | PROVEN | Replace the sketch by P2's sign-symmetrisation proof. At `p=1`, the explicit contraction is `rho = 1 - 5/(8n) + O(n^-2)`. Smaller positive powers follow by Jensen. |
| Theorem 1(iv): uniqueness only among “tempered” families | PROVEN | P1 gives uniqueness among all probability-valued entrance laws. Its global reset needs no moment assumption. The displayed uniform annealed moment condition is also stronger than the usual meaning of temperedness. |
| Theorem 1(v): rate `-5/(4n)` and “far pairs contract at least as fast in L^s” | PROVEN | This conflates typical, conditional-survival growth with moment contraction. The proven first-moment contraction rate is approximately `5/(8n)`. Fixed-width infinite-time tangent exponents are all `-infinity`. |
| Model R: arbitrary deterministic schedules | PROVEN | Pullback existence does hold for any finite nonnegative schedule, using global resets. The fractional-moment summability argument alone needs control of the schedule's growth. |
| Model R: two inputs retain their radius ratio | PROVEN | With one shared pilot schedule and shared noises, global resets eventually make the two trajectories exactly equal. Rescaling the input **and its pilot schedule** is a different operation and preserves global scale by homogeneity. A stationary directional law was not established. |
| Item 2: every estimator's efficiency is sigma-independent | PROVEN UNDER scale-equivariant estimator, initialization, target and loss | State distributions scale exactly. Fixed thresholds, fixed absolute tolerances, or non-scaling algorithms need not have invariant efficiency. |
| Item 3: definition of `e_l` | PROVEN | Supply an arbitrary measurable unit vector when `g_l=0`; this event has probability `2^(-n)`. The scalar transition remains valid. |
| Item 3(b): bounded remainder independent of incoming radius | PROVEN | The remainder `D = sqrt((sqrt(g)R+eta)^2+C)-sqrt(g)R` depends on `R`; only its bound is independent of `R`. Use P4's exact affine perpetuity instead. |
| Item 3(b): Goldie application | PROVEN UNDER scalar implicit-renewal theorem | The affine representation verifies the needed hypotheses and gives `P(X>x) ~ C_n/x`, `C_n=1/E[A log A] = 0.4n+13/75+O(n^-1)`. Goldie theorem number/bibliographic citation: **unverified**. |
| Item 3(c): finite mean and variance of `log g` | PROVEN | Unconditionally `E log g=-infinity` because `P(g=0)>0`; its variance is not finite. The expansions concern conditioning on `g>0`, or finite windows without death. |
| Item 3(c): inverse-gamma bulk only HEURISTIC | PROVEN | P5 supplies a stationary-limit proof: `X/n => 2/(5Z)`, `Z~Exp(1)`. This is a large-width limit, not an exact finite-width distribution. Dufresne citation: **unverified**; P5 also derives the density directly. |
| Item 3(d): near-floor probability approximately `exp(-0.4n)` | CONJECTURE | This extrapolates the bulk limit into a moving far-left tail. It needs a separate large-deviation argument. |
| Item 4: finite Lyapunov exponents along noisy trajectories | PROVEN | P7 applies. Under P8's rank-conditioned law, the finite exponents and their corrections are explicit even along noisy trajectories; the one-vector mean/variance identity itself is correct. |
| Item 5: stationary noiseless attractor direction | PROVEN | At fixed finite width, the noiseless state eventually dies. The useful direction is a finite-window, survival-conditioned reference. |
| Item 5: radial covariance about `0.4 n^2 sigma^2/Z` | PROVEN | That is a squared-radius scale, not a centered radial variance. Approximately, `Var(R) = CV(R)^2 E[R]^2`; the CV factor matters. |
| Item 5: Frobenius mean implies typical leading singular amplitude | PROVEN | `E||J_j||_F^2=n` does not establish `s_1(J_j)^2 ~ n exp(2S_j)` typically. Rare amplification and rank loss prevent this inference from the mean. |
| Item 5: frozen-gate covariance and thinness estimate | HEURISTIC | Noise-dependent gates do not give a Gaussian linear covariance after conditioning on them. The proposed `log(n)/n` angular fraction is unsupported; a competing closure below gives `n^(-2/3)`. |
| Item 5: radius as an unavoidable quenched non-Gaussian failure | PROVEN | The annealed heavy tail does not imply a quenched heavy tail. P3 gives quenched squared-radius exponential tails; the data permit increasingly Gaussian central fluctuations. |
| Item 6: births | PROVEN | Resampling and deletion are different observables. The Gaussian-unit resampling formula is correct for that model, but is not exact for a fixed finite-width network. |
| Item 6: deaths and live-bit budget | HEURISTIC | Neither a universal radial lifetime nor independent gate contributions was established. Multiplying births by one lifetime does not account for the measured interactions. |
| Item 7: transition time | HEURISTIC | This is a drift-based crossover estimate, applicable when the initial signal exceeds the stationary scale. It is not a burn-in guarantee. At `n=256, sigma=0.001`, the displayed formula gives about `941` layers. |

PROVEN: the exact fixed-row resampling probability, conditional on the clean activation `a`, is

```text
t = (w.a)/(s |w|),
p_resample = 2 Phi(t) Phi(-t).
```

PROVEN UNDER the independent Gaussian-unit model: averaging gives `acos(r)/pi`, with small-noise coefficient `sqrt(2)/pi = 0.450158`. Deletion instead has coefficient `1/pi = 0.318310`; the boundary-band coefficient is `sqrt(2/pi) = 0.797885`.

PROVEN (scoring audit): the following conclusions follow from `PREREG.md`, `scratch/analyze_A.py`, and the saved JSON values.

| Prediction | Correct reading |
|---|---|
| PA1 | The explicit 20% refutation criterion at `n<=64` is avoided; the stronger 10% prediction at every width fails. At `n=256`, the 90th-percentile ratio is `1.190`. Calling this an unconditional “hit” would be generous. |
| PA2 | Median and Hill checks pass; monotonic approach fails. The JSON separates these correctly. |
| PA3 | **Both numerical checks pass after registered median aggregation.** Median `n lambda_1` values are `-1.423,-1.163,-1.061,-1.254,-1.284`; median gaps for `n>=32` are `1.931,1.231,1.063,1.056`. The scorer instead requires every net to pass. |
| PA4 | Numerical pass, but the implementation fits relative distance `|hA-hB|/|hA|`, not absolute distance. |
| PA5 | Still fails after median aggregation: at `n=32`, covariance slope is `-0.612`, and frozen-attractor slope is `-1.171`. The every-net scoring is stricter than registered but does not change the overall failure. |
| PA6 | Pass: Spearman `0.942857`; required large-width ratios exceed 2. |
| PA7 | Pass. The scorer silently permits decreases up to `0.02`, which was not registered; actual medians increase, so the slack changes nothing here. |
| PA8 | Fail under the explicitly stated every-bin requirement. Evaluating the formula at a bin's median radius instead of averaging predictions also introduces approximation error, but does not excuse the registered failure. |
| PA9 | Pass under median aggregation; max/min across width medians is `1.339`. Individual-net failures do not overturn that registered aggregation. |
| PA10 | `(a)` pass, **`(b)` pass under median aggregation**, `(c)` fail, `(d)` pass, `(e)` fail. For `(b)`, the largest median discrepancy factor is `1.914`, below 2. |
| PA11 | Fail: at `n=256`, neither net meets the specified radial non-Gaussianity criterion. |
| PA12 | Pass: maximum relative discrepancy `5.11e-15`. |
| PR1, PR2 | PR1 passes. Both PR2 conditions fail overall, including after median aggregation. |

PROVEN (implementation audit): three additional limitations affect interpretation.

- The Lyapunov warm-up is empty: its default start is `-12n+1`, but its requested endpoint is `-30n`. QR therefore starts from zero, with only the subsequent `2n` discarded. QR completion and clipping also produce finite reported numbers after rank loss. At `n=32`, approximately `3.14` layers with fewer than nine active gates are expected in `28n` layers, so the nine-exponent gap statistic cannot represent nine surviving tangent directions.
- Age, Lyapunov and coupling runs use only two seeds even at smaller widths, despite the four-net specification. Large-width radius runs use two paths for 40,000 layers instead of four paths for 20,000 layers.
- The age sums are the registered trapezoidal approximations through `4n`. They are not exact discrete sums over all ages. Radial share in PA7 is an **uncentered** second-moment share, not a covariance share.

2. Angular covariance and quenched radial variance

PROVEN UNDER an externally prescribed homogeneous reference trajectory and its linear response: let `r_l u_l` be the reference, `P_l=I-u_l u_l^T`, and `J_l=D_l W_l`. Define

```text
B_l = (r_(l-1)/r_l) P_l J_l P_(l-1),
G_j = B_0 ... B_(-j+1).

C_ang,0 =
  r_0^2 sum_(j>=0) [sigma^2/r_(-j)^2] G_j P_(-j) G_j^T.
```

PROVEN UNDER the additional diagonal-mode approximation `G_j e_k = exp(-gap_k j)e_k`: this becomes Claude's formula. At constant radius,

```text
variance_k = sigma^2/[1-exp(-2 gap_k)]
           ~ sigma^2 n/[2(k-1)].
```

HEURISTIC: applying this to the actual cloud needs assumptions that remain unverified: a common transported mode frame, negligible transient amplification, negligible gate changes, and coefficients sufficiently independent of the injected noises. Substituting an ensemble RMS radius does not establish those assumptions.

PROVEN UNDER the same exponential-reference approximation: if instead `r_0/r_(-j) ~ exp(gamma_1 j)`, the radius factor cancels the gap correction:

```text
exp(2 gamma_1 j) exp(-2(gamma_1-gamma_k)j)
  = exp(2 gamma_k j).
```

Thus changing absolute exponents to gaps is not by itself a correction to absolute covariance.

PROVEN (saved post-hoc results): the measured/model median ratios are:

| n | Seed 1000 | Seed 1001 |
|---:|---:|---:|
| 32 | 1.88 | 2.49 |
| 64 | 6.22 | 3.05 |
| 128 | 2.67 | 4.34 |
| 256 | 2.44 | 11.57 |

HEURISTIC: radius history contributes, but this particular model does not explain the cloud. In the large-radius `n=256` net it performs worse than its constant-radius version, whose ratio is `9.03`.

PROVEN: an exact finite-window alternative is the Gaussian covariance identity. For all driving noises collected into `Z`, independent copy `Z'`, and `Z_t=tZ+sqrt(1-t^2)Z'`,

```text
Cov_xi(h_0 | W)
  = sigma^2 sum_j integral_0^1
      E_xi[J_j(Z) J_j(Z_t)^T | W] dt.
```

Here `J_j` is the tangent product from noise age `j` to the observation. This expression involves **two correlated trajectories**, including their different gates. A single frozen trajectory does not supply it. Stationary covariance follows by taking the covariance limit; exchanging that limit with the derivative sum needs justification.

PROVEN: there is also an exact quenched radial moment identity. Writing `b=relu(W_l h_(l-1))`,

```text
Var_xi(|h_l|^2 | W)
  = Var_xi(|b|^2 | W)
    + 4 sigma^2 E_xi[|b|^2 | W]
    + 2n sigma^4.
```

This does not close in `n`. A quenched variance cannot be a universal function of width alone.

HEURISTIC: the following finite-angle closure gives a useful width prediction. Let two independent noise replicas at the same weights have approximately equal radius, and put

```text
X = R^2/(n sigma^2),
delta = E_xi[1-u.u' | W].
```

PROVEN: the Gaussian ReLU overlap kernel has the expansion

```text
kappa(1-delta)
  = 1-delta + [2 sqrt(2)/(3 pi)] delta^(3/2)
    + O(delta^(5/2)).
```

HEURISTIC: norm concentration and slowly varying radius then give the balance

```text
delta_(l+1)-delta_l
  ~ 1/X_l - [2 sqrt(2)/(3 pi)] delta_l^(3/2),

delta ~ c X^(-2/3),
c = [3 pi/(2 sqrt(2))]^(2/3) = 2.230920.
```

This describes finite angular separation. Its relaxation scale is `O(X^(1/3))`, rather than the infinitesimal projective scale `O(n)`.

PROVEN: for two unit directions separated by angle `theta`, their one-layer squared gains satisfy

```text
Cov_W(g(u),g(v))
  = [5 - 4 theta^2 + O(theta^4)]/n.
```

HEURISTIC: consequently, direction-dependent gain differences inject approximately `4 delta/n` variance per layer into the pair's log-radius difference. Additive noise provides radial restoration approximately `1/X`. For `s_R^2=Var_xi(log R | W)`, this gives

```text
s_R^2(next)-s_R^2
  ~ -2 s_R^2/X + 2 delta/n + 1/(nX).

Frozen-radius equilibrium:
CV(R)^2 ~ s_R^2 ~ c X^(1/3)/n + 1/(2n).
```

HEURISTIC: substituting the benchmark radius `X=[2/(5 log 2)]n` gives

```text
CV(R) ~ 1.36285 n^(-1/3),

Var_xi(R | W) ~ 1.07 sigma^2 n^(4/3),
tr C_ang       ~ 1.86 sigma^2 n^(4/3).
```

| n | HEURISTIC CV, including `1/(2n)` | PROVEN saved median CV |
|---:|---:|---:|
| 32 | 0.447 | 0.371 |
| 64 | 0.352 | 0.324 |
| 128 | 0.278 | 0.252 |
| 256 | 0.219 | 0.208 |

HEURISTIC: these constants are encouraging, not a fitted or proved quenched law. Using each layer's current radius overpredicts CV particularly on large-radius layers. The corresponding history-dependent closure is

```text
s_R,0^2 ~ sum_j [2 delta_(-j)/n + 1/(n X_(-j))]
                   exp[-2 sum_(r=-j+1)^0 1/X_r].
```

CONJECTURE: typical quenched radial CV scales as `n^(-1/3)`, with a random history-dependent prefactor. The saved width medians fit exponent approximately `-0.297`. Neither this closure nor the Lyapunov gaps determine the covariance eigenvalue spectrum; a `1/k` spectrum remains a separate empirical observation.

3. Interactions, shared pasts and live bits

PROVEN: for independent noise-age blocks and vector-valued Hoeffding components with variances `V_S`,

```text
T_j = sum_(S containing j) V_S/V,
sum_j T_j = sum_S |S| V_S/V.
```

Thus the exact sum is the variance-weighted mean interaction order. It is not a count of independent contributions.

HEURISTIC: the gain cascade is a natural source of the excess. Older noise changes direction; direction changes subsequent gains and gates; those changes multiply later and earlier contributions. A sum of independent, linearly propagated injections omits these products.

CONJECTURE: a concrete asymptotic candidate is

```text
sum_j T_j = O(n^(1/6))
```

for typical environments, with substantial finite-width and radius-history effects.

HEURISTIC argument: replacing one complete noise layer initially separates directions by `theta_0=O(n^(-1/2))`. The small-angle kernel gives approximately

```text
theta_j ~ theta_0/[1+theta_0 j/(3 pi)].
```

Its integrated angular response is therefore `O(sigma^2 n^(3/2))`. Transfer into the slower radial component preserves this order when `X=O(n)`. Dividing by the proposed cloud variance `O(sigma^2 n^(4/3))` gives `n^(1/6)`.

HEURISTIC: the present data do not identify this exponent. A log fit, approximately `0.459 log2(n)`, also describes the width medians; a power fit gives approximately `n^0.233`. Calibrating the `n^(1/6)` candidate at the current `n=256` median predicts about `4.42` at 512 and `4.96` at 1024.

PROVEN: a cheaper interaction check than resampling every age separately is to refresh each age independently with probability `epsilon`. Then

```text
E|F-F_refreshed|^2/(2V)
  = sum_S [1-(1-epsilon)^|S|] V_S/V.
```

Its derivative at zero is the total-index sum. Paired full-window simulations estimate this curve without the cost of a separate continuation for each age.

PROVEN: the old-age percentage is not the variance fraction caused by the old past. Interactions are counted repeatedly. For a deep-past block `D`, distinguish

```text
S_D = Var(E[F|D])/V,
T_D = E Var(F|all blocks outside D)/V.

S_D <= T_D <= sum_(j in D) T_j.
```

PROVEN: sharing one deep past across `N` independent recent branches gives

```text
Var(sample mean) = V [S_D + (1-S_D)/N].
```

With `M` independent pasts and `K` branches per past, it becomes

```text
V [S_D/M + (1-S_D)/(MK)].
```

The cheapest relevant check is therefore a paired estimate of `S_D`, plus a whole-block resampling estimate of `T_D`. The reported old-age total-index sum alone cannot determine the sharing floor.

HEURISTIC: a live-bit budget must specify an observable, accuracy, and interaction convention. The `n^(3/2)` births-times-lifetime estimate remains a possible event count, but cannot be interpreted as that many independent variance contributions. P12's recent-noise truncation bound supplies a rigorous alternative upper bound, with order `n^2` coordinates times a logarithmic accuracy factor and weight-dependent constants.

4. Model R

PROVEN UNDER a surviving clean finite-depth trajectory and a pilot schedule continuous at `sigma=0`: ordinary small-noise linear response still exists:

```text
h_L(sigma)
  = h_L(0)
    + sigma sum_(j=1)^L r_j(0) J_(L:j+1) xi_j
    + o_L2(sigma).
```

Consequently, covariance and angular mean-square deviation are proportional to `sigma^2` asymptotically at fixed `n,L`. The observed ratios `13-133` show that the tested sigmas are not uniformly in this regime; they do not disprove that asymptotic statement.

HEURISTIC: three effects explain its poor uniformity:

- The noisy pilot changes with sigma.
- Relative to the decaying clean reference, accumulated radial perturbations have a neutral leading direction rather than strong restoration.
- Gate crossings and transient longitudinal amplification accumulate with depth. Their onset depends on the particular input's radius relative to the pilot.

PROVEN UNDER the linear-response approximation: the leading log-radius variance is

```text
sigma^2 sum_j [r_j(0)/R_L(0)]^2
                 |u_L^T J_(L:j+1)|^2.
```

This includes angular-to-radial transfer and need not equal a naive direct-radial-noise calculation.

HEURISTIC: the steep small-sigma spectrum is consistent with this near-frozen linear regime. Larger sigma mixes gate histories and changes the spectrum. Births can still follow their local Gaussian-unit approximation because they probe one fresh injection, rather than an accumulated response.

PROVEN (scope of evidence): growth from `4n` to `16n` establishes depth dependence over those windows. It does not establish unlimited growth or exclude eventual stabilization in pilot-normalized units.

5. Part B design

HEURISTIC: the strongest candidates are a cheap surrogate that retains radius-history interactions, full-history Gaussian coordinates ordered for RQMC, and exact final-noise integration. Radius-only modeling is insufficient to guarantee a width-growing gain: the closure above assigns both radial and total angular variance order `n^(4/3)`.

PROVEN: final-noise Rao-Blackwellisation of the mean is immediate. Conditional on `b=relu(W_0 h_-1)`, write

```text
mu_i = w_i.b,
s_i  = sigma |w_i|.

E[a_i | b] = mu_i Phi(mu_i/s_i) + s_i phi(mu_i/s_i).
```

It needs the same output matrix-vector product, removes the final `n` Gaussian coordinates, and cannot increase variance.

PROVEN: Gaussian Poincare bounds its removed trace variance by `sigma^2 ||W_1||_F^2`, typically order `n sigma^2`.

HEURISTIC: if total variance is order `sigma^2 n^(4/3)`, final-noise integration alone removes only an `O(n^(-1/3))` fraction asymptotically. Its additional value is smoothing the output for other estimators.

PROVEN: for the norm tail, coordinatewise Gaussian formulas cannot be multiplied because output preactivations are correlated. There is nevertheless an exact one-dimensional integration. Choose a fixed unit direction `v`, decompose final noise as `xi=zv+xi_perp`, and condition on everything except `z`. Then

```text
|a(z)|^2 = sum_i max(c_i+d_i z,0)^2.
```

This is convex, so its sublevel set at `t^2` is an interval `[z_-,z_+]`, possibly empty or unbounded. Hence

```text
P(|a|>t | remaining randomness)
  = Phi(z_-) + 1-Phi(z_+).
```

The endpoints can be found from the piecewise quadratics. A pilot-derived fixed radial direction allows `W_1 v` to be precomputed.

HEURISTIC: the most useful surrogate upgrade is to retain a scalar radius and a few angular factors, including their multiplicative interaction, while making each surrogate step much cheaper than an `n x n` multiplication. Couple it to the true path through the same projected noises. Estimate its expectation analytically where possible, or with many cheap independent surrogate paths.

PROVEN UNDER independent level estimators: for surrogate `G`, the optimal two-level variance-cost constant is

```text
K = [sqrt(Var(F-G) C_fine)
       + sqrt(Var(G) C_surrogate)]^2.
```

The surrogate need not be correct to yield an unbiased finite-window estimator. This avoids treating the K2 approximation as ground truth.

CONJECTURE: capturing both the radius cascade and leading angular fluctuation could leave a residual fraction decreasing with width. Under an accurate leading-state approximation and regular gate-margin density, the ReLU crossing remainder suggests a residual fraction of order angular RMS, potentially `n^(-1/3)`. Accuracy of that state approximation is the missing hypothesis.

HEURISTIC: the cheapest test is independent-pilot, held-out comparison at `n=32,64,128` of:

```text
residual variance fraction,
surrogate cost / true-path cost,
setup cost,
variance explained in R itself.
```

Marginal Gaussianity of `R` does not imply that a linear projection of all driving noises predicts it well; test that directly.

PROVEN: an orthogonal transformation of the entire finite Gaussian noise vector preserves its law. Thus global radius-sensitive coordinates, learned on an independent pilot, can be placed first in RQMC while the remaining coordinates retain their correct Gaussian distribution.

HEURISTIC: use coordinates spanning the full relevant history, with additional angular coordinates, rather than ordering solely by recency. The current Sobol dimension cap would cover only about `0.32n` recent layers at `n=256`; the interaction results make that truncation potentially costly.

PROVEN: radial importance sampling must use a known likelihood ratio. A fitted Gaussian marginal for the actual radius is insufficient. A valid alternative tilts Gaussian driving coordinates:

```text
Z ~ N(mu,I),
likelihood ratio = exp(-mu.Z + |mu|^2/2).
```

A direction chosen from a pilot radius or tail gradient can target radial excursions while retaining exact weighting.

PROVEN UNDER the ideal scalar event `Z>b`: shifting by `mu=b` gives variance-reduction factors approximately `37.1` for probability `0.01`, and `286.6` for probability `0.001`. These are reference constants, not predictions for the network.

HEURISTIC: first test whether a few full-history Gaussian coordinates predict the tail. Combine successful tilts with final-noise integration; report likelihood-weight variability and independent-replicate error, since a few missing high-weight samples can make a correct importance sampler look biased.

PROVEN UNDER exponential quenched burn-in error and summable multilevel variance-cost terms: burn-in MLMC removes the accuracy-dependent logarithmic overhead of plain burn-in MC. It does not establish a better power than cost exponent one.

CONJECTURE: radius-aware RQMC may improve the exponent or constant, but neither a width-uniform gain nor a better power follows from the current cloud spectrum and total-index sums.

The claim ledger follows.

| Claim | Label | Cheapest check |
|---|---|---|
| Frozen v1 and Part A provenance preserved | PROVEN | Recompute archived v1 hash and trimmed Part A prefix hash. |
| PA3 numerical checks and PA10(b) pass after registered aggregation | PROVEN | Take width-wise medians of saved rows. |
| Finite-width infinite-time tangent exponents are `-infinity` | PROVEN | All-off-layer argument; separately log rank loss in QR. |
| Radius-history diagonal model explains angular covariance quantitatively | CONJECTURE; contradicted by current tests | Inspect saved measured/model ratios, especially `11.57` at `n=256`. |
| Angular dispersion is order `n^(-2/3)` | HEURISTIC | Measure normalized two-replica overlap against radius history. |
| Quenched radial CV is order `n^(-1/3)` | CONJECTURE | Compare `n^(1/3) CV` across more independent weight windows; test the history equation. |
| Total-index sum grows like `n^(1/6)` | CONJECTURE | Sparse whole-layer-refresh curve before expensive per-age scans. |
| Old-age total-index share determines a shared-past floor | PROVEN false | Estimate the closed block index `S_D` directly. |
| Model R recovers sigma-squared response at fixed finite depth | PROVEN UNDER surviving clean path and regular pilot limit | Hold the pilot fixed and halve sigma repeatedly. |
| Final-noise mean and scalar tail integration reduce variance | PROVEN | Paired estimates at identical path cost. |
| Cheap radius-angular surrogate gains improve with width | CONJECTURE | Held-out residual variance times full amortized cost. |
| Gaussian-coordinate tail tilting is unbiased | PROVEN UNDER exact likelihood weighting | Independent replicates and direct likelihood-ratio checks. |
