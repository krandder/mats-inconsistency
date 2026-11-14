# The bits that matter at layer l: what they are, what they cost, and where the hash argument holds (bits-bounds, 1 November 2026)

Lane: bits-bounds (theory on ARC's agenda, target M1). Topic bead pool-ttdzrk; lane bead pool-ttdzrk.12 (21 leaves).
Status: final for 1 November (23:47Z). Reviewed twice by gpt-6-astra in fresh sessions (16 and 14 findings, all
applied or answered; section 10 lists the changes); sections 3.5-3.6 (the bits formula, added later) were audited by
co-prover round 4 (12 findings, applied) rather than by a fresh reviewer.

Work directory `work/bits-bounds/`:
- `FORMAL.md`: theory v2.3 (v1, frozen at 21:18Z before the co-prover's answer, is `coprover/FORMAL_v1_as_written.md`;
  hashes in `notes/formal_hashes.txt`).
- `PREREG.md` (registered 21:32Z; Amendments 1-5 at 21:45Z, 22:04Z, 22:20Z, 22:29Z and 23:17Z; hashes and times in
  `prereg_hashes.txt`, each before its runs).
- `coprover/`: lemma-free statement, prompts, raw logs, four gpt-6-astra answers, `PROVENANCE.md` (with the checks).
- `lib/bb.py` (nets, frozen-gate futures, merges), `lib/meanfield.py`, `lib/chain3.py` (exact finite-width
  three-replica chain), `lib/closure.py` (exact single-Gaussian recursion), `tests/test_bb.py` (7 pass,
  `logs/test_bb.log`), `tests/test_closure.py` (3 pass, `logs/test_closure.log`).
- `scratch/run_eval.py`, `scratch/run_ml.py`, `scratch/run_gain.py`, `scratch/run_closure.py`, `scratch/run_bits5.py`
  (runs); `scratch/score_prereg.py`, `scratch/score_ml.py`, `scratch/score_gain.py`, `scratch/score_closure.py`,
  `scratch/score_closure4.py`, `scratch/score_bits5.py` (scorers); `scratch/tables.py`, `scratch/posthoc.py`,
  `scratch/variance_made.py` (descriptive and post-hoc tables); `scratch/check_r3.py`, `scratch/check_quadrature.py`
  (numerical checks); `runs/`, `runs5/`, `results/`, `logs/`, `review/`, `notes/`.

Co-prover: gpt-6-astra, four rounds at xhigh effort (round 1 independent from a lemma-free statement; round 2 a
claim-by-claim audit of v1 plus new derivations; round 3 a lower bound for the Gaussian closure; round 4 an
independent derivation and audit of the bits formula). Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL,
MEASURED, HEURISTIC, CONJECTURE. Provenance: [C] Claude, [A] gpt-6-astra, [C+A] both independently, [R] the
relevance lane, [IB] the input-bits lane.

## Summary for Kelvin

**Short answer.** There is now a theory of the bits that matter at layer l, with a past half (how many directions the
cloud of states still has) and a future half (the scale below which later layers act linearly on it, and how much of
the output's variance they still make from the layer-l state). For cell methods it comes down to one heuristic
formula, N_l(eps) ~ (d_l/p) log2(D_l(1)/eps^2): the cloud's Lyapunov dimension d_l ~ n/l times the log of the
variance share the future still makes from layer l (item 5). It passed a registered test on 12 new nets (6 of 6
predictions), and it matches the bit-accounting lane's learned codes in order of magnitude. But its strength is in
separating the cuts that need many bits from those that need none; it gets the intermediate counts only within a
factor ~2, and the co-prover showed that its lower-bound version rests on hypotheses that fail globally for real
nets. The theory is proven in its laws and identities, it gives lower bounds for restricted classes
(forward-only cell methods, point queries to an unknown table) and conditional ones for methods that touch the
future at points, and it explains where your hash picture is right: at ARC's instance and wherever the variance is
made, the future is rough on a high-dimensional cloud.
It does not prove that nothing beats the best multilevel sampler. The open classes are methods that average
analytically and adaptive point methods on actual nets (covered only under a Gaussian surrogate). The cleanest
averaging method, an exact Gaussian closure, gained at most about 3x over sampling up to width 256 at depth equal to
width; under a one-layer hypothesis its annealed gain is bounded by a constant at large width (item 10).

1. **Model** (PROVEN placements [C+A]). Estimators that sweep the layers with p passes and M bits of state, paying for
   all work. ARC-style cumulant propagation (with a stated reconstruction), Gaussian closures with gate splits,
   branch-and-bound and the multilevel shared-trajectory sampler (one pass) all fit.
2. **Information bounds cannot see your bits** (PROVEN [C+A]). Any information lower bound on an unrestricted state is
   at most the weights' description length, n^2 L log(1/eps) bits, and the information a layer-l state carries about
   the output is the same at every layer. A count of input bits can only bound a restricted class, or time.
3. **The past's half: about n/l directions** (MEASURED on 64 nets; the fit is POST HOC). The angular cloud at layer l
   has covariance participation ratio PR = 1.04 (n/l) + 1.38 (R^2 = 0.96, 256 (net, layer) points, nine
   configurations), and angular size about 3 pi/(l + 6) for l << n (smaller in the deep half at finite width). The n/l
   term is the rank-conditioned Lyapunov count: these are your "Lyapunov covariances". They do not single out
   directions for the future: given the past, the future's frame is uniformly random relative to the cloud's axes
   [R, A].
4. **The future's half: a linearity scale of about 2.45/L' radians** (PROVEN: exact for one layer at every width; mean
   field for L' layers in closed form [C, A]). Freezing the future's gates at a reference state c leaves residual
   (2|h|^2/pi)(phi - sin phi cos phi) ~ (4/(3 pi)) |h|^2 phi^3 for one layer (phi = angle between h and c) and
   (4 L'/(3 pi)) phi^3 for L' layers when L' phi is small; freezing beats merging radially while
   L' phi < 3 pi (2^{1/3} - 1) = 2.45. Beyond that scale the future has merged the directions and states differ through
   gains, a rough field with variance about (6 pi/n) x angle [A, proven in a limit; measured within 2-3x on nets]. The
   residual comes only from the future gates on which the two trajectories disagree (exact identity).
5. **The bits at layer l, as one formula** (section 3.5). rho_l = cloud size / linearity scale ~ 3.8 L'/l sets the
   regime: below 1 (the last ~20% of the layers) the future is smooth on the cloud, above 1 it is rough. The output's
   variance is made early: the log-radius difference of two inputs, frozen at death, is a martingale over layers whose
   increments depend only on the current angle between them (PROVEN, every width [C, repaired by A]), and along the
   contracting angles about half of it is made in the first 8 layers and 94% in the first 128, at any width (mean
   field; the input-bits lane's profile [IB]; the share left after layer l is ~8.3/(l + 9)). So one merge at a rough
   cut l leaves a width-free share D_l(1) ~ (3-9)/l of the variance. For forward-only cell methods the bits that
   matter at layer l are N_l(eps) ~ (d_l/p) log2(D_l(1)/eps^2) (HEURISTIC), with d_l an effective dimension of the
   cloud (fitted ~1.4 PR_l ~ 1.4 n/l over 4-256 cells: the past) and p the distortion's exponent in the cell size
   (1 in the rough window, 2-3 when smooth). Two caveats from the co-prover [A]: for the exponential Lyapunov spectrum
   the effective dimension grows with the precision, so the bits grow like the square of log(D_l(1)/eps^2) (close
   to the formula at D_l(1)/eps^2 ~ 10); and the lower-bound version (PROVEN for an idealised source [C]) needs a
   linear minorant of the future's distortion and a small-ball condition, both false globally for nets (the
   distortion is quadratic at small angles; the cloud has atoms), so it is not yet a lower bound for nets.
   Registered on 12 new nets (Amendment 5, 6 of 6 hit): the variance profile within [0.5, 1.5] of mean field on 48 of
   48 cells; the bits formula agrees on 82 of 90 pairs, made of 63 pairs that need more than 8 bits, 14 that need none,
   and only 5 of 13 intermediate counts within 2 bits (it over-predicts those by a median factor 1.8). At 512 x 64 it
   puts 50-600 bits at the measured cuts, a table exponential in d_l; at 192 x 192, l = 96, about 6 bits at
   eps^2 = 0.01 (measured 2.9). Cross-lane
   (POST HOC, one net; section 3.6): with d_l = PR_l it reproduces the bit-accounting lane's learned local-code bits
   at ARC's rough layers (224, 112, 45 against their 230, 104, 34-52 at l = 4, 8, 16), and a Gaussian water-filling on
   the cloud's measured spectrum, with no choice of d, lands within 1.6x of them at every layer. At the smooth layers
   their codes settle at the output's own ~19 bits, a floor my cell count lacks because its merges keep the state.
6. **Bounds tied to it.**
   - Forward-only cell methods need log2 M >= (d/3) log2(E R^2/D) + O(1) cells, while cells aligned with the actual
     future's regions are exact; averaged over futures the separation is unbounded at high precision (PROVEN in the
     class, under angular-source hypotheses [C, corrected by A]).
   - Merges with M past-chosen references and exact cell statistics: their future-averaged error is an exact sum of
     per-cell terms plus cross terms between cells; with equal-mass cells and negligible cross terms it is at most
     D/M, the error of a stratified sampler with one draw per cell (PROVEN identity and conditional upper bound [C]).
     Measured, their per-net bias is about D(M)/M in the rough regime (median ratio 0.82 over 453 cells).
   - For past-chosen linear point rules (merges whose references, weights and merge types are fixed by the past; not
     our future-selected "best" merge) the future-averaged error is exactly a kernel discrepancy under the
     finite-width kernel, whose cubic singularity is proven; the resulting rate, error^2 >= c M^{-1-3/d}, is a
     CONJECTURE until a kernel-quadrature theorem is applied with checked hypotheses. For adaptive point methods the
     rate M^{-1-2H/d} holds under a Gaussian surrogate of the future, which real finite-width nets are not; adaptive
     point methods on actual nets are not covered.
   - Point queries to a table of 2^N values with an unknown common bias need Theta(min(2^N, eps^{-2})) queries
     (PROVEN [A]).
7. **Where the bits are, measured** (our k-means construction, four cuts per net). At ARC's 256 x 32 the cloud has
   13-33 effective directions at the measured cuts, and 256 cells do not bring the merge residual to eps^2 = 0.1 at
   l = 8, 16, 24. At L = n the middle layer needs more than 8, more than 8, 6.1 and 0.8 bits at eps^2 = 0.01 for
   n = 32, 64, 128, 256: those bits fall with width, because the residual left after one merge at layer l is about
   (3-9)/l of the variance at any width. The variance is made in the first few dozen layers, where the cloud has ~n/l
   directions and the remaining future is long and rough.
8. **Samplers built on the bits** (MEASURED, registered; ideal propagation-only comparisons). The one-reference
   multilevel sampler's gain over MC grows with width at L = n (1.21, 1.41, 1.58, 2.43 at n = 32-256), is 2.2-3.9 at
   L = 4n and 0.99 at 256 x 32. Several reference trajectories add 1.4-1.55x at L = 4n, nothing measurable at L = n
   for n >= 128, and lose at 256 x 32.
9. **Your hash argument.** What survives: beyond the future's linearity scale, the future is rough on a cloud with
   ~n/l directions; merges with exact cell statistics behave in measured size like stratified sampling, and under the
   Gaussian surrogate no point method beats sampling by more than M^{2H/d}; ARC's 256 x 32 sits in that regime at
   every cut we measured except the last (l = 28 of 32). What fails: the count is not exp(k*) (it is d_l ~ n/l and
   rho_l; k* does not enter); the final layers are the easiest, not the hardest (at most 1.7 bits at 7L/8 for L >= n
   at eps^2 = 0.01); a random hash with a known ensemble mean is easy to average (PROVEN [C+A]; the gain over MC is at
   least 4(K-1)/K at every budget, from the ensemble prior alone, not from any structure); and plain sampling is beaten
   by the multilevel sampler.
10. **The open step: averaging methods, and how far the cleanest one gets.** Methods that average analytically
    (Gaussian closures, cumulant propagation) integrate kinks without resolving them and evade every bound above. I
    tested the cleanest, the exact single-Gaussian recursion (exact Gaussian ReLU moments, matched at every layer), at
    its matrix-product cost of 2n passes (registered Amendments 3-4; generous, since it ignores the pairwise moment
    evaluations). Its median gain over MC at L = n is 0.38, 0.28, 0.63, 3.26 for n = 16-128, where its error falls
    like n^{-2}, and 0.87 and 1.02 at n = 192 and 256 (4 nets each). The registered comparisons at n = 192-256
    (B3, B4) came out on the side of a peak, but the evidence is moderate: post hoc, the n = 192 and 256 nets sit 1.9
    and 1.7 standard errors above the extrapolated n^{-2} trend, and per-net gains scatter over two decades. At
    L = n/2 the gain does not grow (2.28 at n = 16, then 0.4-0.9). The co-prover proved why the gain cannot grow
    without bound, under one hypothesis. Gaussianisation discards the third and fourth cumulants and raises the
    cloud's mean radius by 9/(16 sqrt n) at the first layer, and that defect survives any later suffix of depth
    alpha n (PROVEN for one Gaussianisation step followed by the exact suffix [A]; constants checked numerically [C]).
    If Gaussianisation never lowers the mean radius on average over a fresh random layer (H_R, CONJECTURE; its
    pointwise version is false), then at fixed alpha and all large enough n the recursion's annealed error
    E_W|err|^2 is at least about 1e-3 x E_W V/n at L = n, so its annealed gain at its own cost is bounded by a
    constant, about 470 (PROVEN UNDER (H_R) [A]). That constant sits two to three decades below the measured errors,
    so it bounds the gain without saying where growth stops. The lead is therefore weakened, not closed: the
    measured gain is at most about 3 up to n = 256, and unbounded growth would need (H_R) to fail. Corrected closures
    (higher order, mixtures, gate splits) remain the open class.
11. **If you want one parameter.** N_l(eps) of item 5. Its pieces are the cloud's dimension d_l and the merge
    residual's decay exponent gamma_l = p/d_l (measured for our construction: gamma_R x PR = 0.72 post hoc, 0.74
    registered, in the rough regime; 0.01-0.14 at ARC's 256 x 32, 0.3-0.4 at L/2 of L = n nets), and the variance
    share D_l(1). The adopted linear-feature count N prices representation (forward-only: the cloud's ~n/l
    directions; two-sided: N), but by itself gives no lower bound for the mean (a linear future has a large N and an
    exact mean) [A].

Pre-registered scorecard (section 8): of 11 main predictions 3 hit and 8 missed (P8 counts as a hit under its
registered wording; the hashed scorer used a stricter reading); five misses are narrow, one comes from the comparison
target (P2), and one was a wrong expectation: I expected the bits at the middle layer of L = n nets to stay above 8 as
width grows, and they fall (P11). Amendments: the one-reference multilevel bands hit, two predictions missed narrowly,
the rough-gain law is within the band at every width; at L = n the closure's error fell faster than registered over
n = 16-128 (B1 and B2 missed there, hit at L = n/2), and the registered comparisons at n = 192-256 then came out on
the side of a peak (B3, B4 hit; moderate evidence, item 10). Amendment 5 (the bits formula on 12 new nets; bands
calibrated post hoc on the old nets, disclosed) hit all six predictions, the bit counts mainly through their many-bit
and zero-bit cases (5 of 13 intermediate counts within 2 bits).

Corrections made today: from the co-prover's audit, my finite-width "enhancement" of the cubic law was a normalisation
artifact, the forward-only/two-sided separation holds averaged over futures (not for a fixed net), and the hash
proposition's smallest gain over budgets is 4(K-1)/K, just under 4, not "at least 4"; from the fresh review,
"references are samples" is an upper bound
(not a lower bound), the point-method rate is conjectural without the surrogate, the closure cost figure describes one
implementation, the "best" merge is not forward-only, and several claims were narrowed to the tested constructions;
from round 3, the hardness lane's K = 2 slopes, which I had cited as evidence on closures, describe ARC's truncated
recursion rather than the exact Gaussian closure.

## 1. The question, made precise
(see FORMAL.md sections 1-2)
- Nets: width n, depth L, W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), h_l = relu(W_l h_{l-1}), output a = h_L; mean
  mu = E_x a; criterion |est - mu|^2 <= eps^2 V (per net or averaged). Cut at layer l: cloud P_l (law of h_l),
  future g_l (layers l+1..L), L' = L - l. g_l is positively homogeneous and piecewise linear, g_l(h) = J_l(h) h.
- Criteria are kept apart (FORMAL section 1): the per-net nMSE |est - mu|^2/V(W), its median over nets, the annealed
  ratio E_W|est - mu|^2 / E_W V, and absolute distortions. E1-E3 use unit inputs (the radius is exact by homogeneity)
  and report residuals in units of V_ang = tr Var_U h_L(U); V = m_n^2 V_ang + (n - m_n^2) E|h_L(U)|^2 with
  m_n = E|x|. The closure test uses Gaussian inputs and V itself.
- Model (Def 2.1) [C+A]: p passes, M bits of persistent memory (everything kept counts), public randomness, all work
  charged. Placements (Prop 2.2): MC and designs, the multilevel sampler (one pass: the reference is one more particle),
  cumulant propagation, Gaussian closures with splits (two-sided when the split choice reads the future), branch and
  bound (breadth first, one pass per refinement round), cone enumeration.

## 2. What information bounds can and cannot say (PROVEN)
- A p-pass sweep is a (2p-1)-round protocol at every cut (Thm 3.1), so M >= CC_{2p-1}/(2p-1); with the cut-rank-ext
  bounds: one forward pass needs (n(n+1)/2) log2(1/eps) - O(1) bits, any p passes (n log2(1/eps) - O(1))/(2p-1)
  (fixed architecture, eps -> 0).
- The cap (Prop 3.3) [C+A]: n^2 l log(...) bits of weights always suffice information-theoretically.
- Exact forward-only characterisation (Thm 3.4) [A]: error >= the 2^b-point quantisation error of the mean-query
  function T(A) in L^2 over futures, attained with unrestricted computation.
- A worst-case family with Omega(d) forward-only bits against O(log d) after one future-reading pass (Prop 3.5) [A].
- Trajectory information is layer-free for noiseless nets (Prop 3.6) [C+A]; with a linear future it is large while the
  mean costs nothing (E Y = A m) [A].
Reading: the input's bits cannot appear in an information bound on an unrestricted state. They appear only for a class
whose state represents the cloud (measure-propagation sweeps) or whose access to the future is by points, or in time.

## 3. The bits that matter at layer l
Figure `work/bits-bounds/figs/fig1_bits.png`: (a) the angular cloud's participation ratio against n/l, all 64 nets and
four layers each, with the fit; (b) frozen-gate and radial merge residuals divided by phi^2 against t phi: mean-field
lines (Thm 5.2) and registered net data (bin medians of per-bin means; the nets are normalised by their own output norm,
which inflates both at small t phi, but the crossover sits where the theory puts it); (c) the best merge residual
against the number of reference cells at three layers (ARC's 256 x 32 at l = 16; 128 x 128 at l = 64 and 96).
### 3.1 Exact structure (PROVEN)
- Measure-propagation sweeps (Def 4.1: the state is a finite signed combination of Diracs, Gaussians and
  cone-restricted Gaussians) contain particle methods, the multilevel sampler, Gaussian closures and mixtures,
  branch-and-bound and cone enumeration; cumulant propagation only through a stated measure-valued reconstruction.
  Their error is a telescoping sum of closure defects tested against the exact future (Thm 4.2) [C], under an
  integrability condition; a forward-only defect, averaged over the future, is a kernel
  discrepancy under k(h, h') = |h||h'| kappa_{n,L'}(cos angle) (Prop 4.4) [C+A], and a Minkowski sum of such
  discrepancies bounds a forward-only closure's error [A r2].
- Euler residual (Lemma 4.3) [C; the relevance lane found it independently]: g(h) - [g(c) + J(c)(h - c)] =
  (J(h) - J(c)) h, a sum over future layers of frozen-propagated flip vectors on the disagreeing gates.
### 3.2 The future's half
- One layer, every width (Thm 5.1) [C]: frozen residual (2|h|^2/pi)(phi - sin phi cos phi); radial residual
  2|h|^2(1 - kappa(cos phi)); expected disagreeing gates n phi/pi. Monte Carlo check within 4 SE.
- L' layers, mean field (Thm 5.2) [C; closed form A r2, checked to 7 digits]: the frozen cosine is
  q = s r_t + (1 - s^2) P_t; frozen residual (4t/(3 pi)) phi^3 - (t(t-1)/pi^2) phi^4 + ...; in the scaling limit
  x = t phi/(3 pi): frozen phi^2 [1 + (1+x)^{-2} - 2(1+x)^{-3}] against radial phi^2/(1+x)^2, crossing at
  t phi = 3 pi (2^{1/3} - 1) = 2.4497.
- Finite width (Prop 5.3) [C chain; coefficient A r2]: an exact three-replica Gram chain; the raw cubic coefficient is
  (4/(3 pi)) sum_{j<t} (1 - 2^{1-n})^j, so no t/n correction. (My earlier "1.7-3.3x enhancement" came from dividing by
  the output norm.)
- Rough regime [A r2, PROVEN under log regularisation and iterated limits]: 5 - B(cos theta) = 4 theta^2 - ...; the
  log-gain difference of two states at angle phi has variance (6 pi/n) phi x/(1+x); the radial residual is about
  (3 pi/t)^2 + (6 pi/n) phi in the window 1/phi << t << n.
### 3.3 The past's half (MEASURED, POST HOC)
- PR of the angular cloud = 1.044 (n/l) + 1.38, R^2 = 0.961 (256 (net, layer) points, 64 nets, nine configurations;
  `results/posthoc.md`). HEURISTIC reading: a variance profile exp(-2(k-1) l/n), the rank-conditioned Lyapunov
  contraction, gives PR = coth(l/n) ~ n/l; the eigenvalue log-slopes match -2 l/n for l/n <~ 0.2 and flatten deeper;
  the offset ~1.4 is not explained.
### 3.4 Measured profiles (registered E1; medians over nets; residuals in units of the net's V_ang)
| n x L | l | rho = theta L'/2 | PR | D_radial(1) | D_best(16) | D_best(64) | D_best(256) | bits at eps^2 = 0.1 / 0.03 / 0.01 |
|---|---|---|---|---|---|---|---|---|
| 256 x 32 (ARC) | 8 | 7.2 | 33 | 0.96 | 1.02 | 1.01 | 1.05 | >8 / >8 / >8 |
| 256 x 32 | 16 | 3.0 | 18 | 0.71 | 0.68 | 0.66 | 0.62 | >8 / >8 / >8 |
| 256 x 32 | 24 | 1.1 | 15 | 0.62 | 0.35 | 0.31 | 0.25 | >8 / >8 / >8 |
| 256 x 32 | 28 | 0.46 | 13 | 0.59 | 0.12 | 0.11 | 0.09 | 6.8 / >8 / >8 |
| 128 x 128 | 32 | 11 | 4.8 | 0.27 | 0.14 | 0.12 | 0.10 | >8 / >8 / >8 |
| 128 x 128 | 64 | 3.4 | 3.5 | 0.074 | 0.017 | 0.010 | 0.0065 | 0 / 1.4 / 6.1 |
| 128 x 128 | 96 | 0.82 | 2.5 | 0.025 | 0.0019 | 0.00095 | 0.00053 | 0 / 0 / 0 |
| 256 x 256 | 128 | 2.8 | 3.4 | 0.014 | 0.0058 | 0.0029 | 0.0020 | 0 / 0 / 0.8 |
| 32 x 128 | 64 | 0.80 | 1.2 | 0.047 | 0.00013 | 2.2e-5 | 8.3e-6 | 0 / 0.3 / 0.9 |
Full tables: `results/tables.md` (all nine configurations, four layers, forward-only and two-sided references) and
`results/posthoc.md` (decay exponents, bits, bias ratios). The decay exponent gamma of D_best(M) is 0.01-0.2 in the
rough regime and 0.3-1.5 in the smooth one; in the smooth regime 3/gamma is about twice PR. Disclosures: D_best
chooses frozen or radial per cell on the fit split by looking at the future's outputs there, so it is not a
forward-only method (forward-only statements use the frozen, radial and rule merges); the bits are log2 of the least M
on the grid {1, 4, 16, 64, 256} with D_best(M) <= eps^2, interpolated log-linearly between grid points (POST HOC), and
M = 1 uses the test split.

### 3.5 The bits at layer l as one formula (FORMAL 6.6; registered test Amendment 5)
- Where the variance is made (PROVEN identity [C, repaired by A r4]; profile from the input-bits lane [IB]). For two
  unit inputs, the log-radius difference log|h_l(u)| - log|h_l(u')|, frozen at its value before either state dies,
  is a martingale over layers whose conditional increment variance v_n(theta_j) depends only on the current angle
  between the two states (exact at every width; the frozen process differs from the raw one with probability at most
  2 L 2^{-n}); v_n ~ 2 theta^2/n at small angles. Along the mean-field angles (theta_j ~ 3 pi/(j + 6)) the increments
  are summable: n E_W Var_u log|h_inf| = 10.65 ([IB]'s C_inf), and the share made after layer l is 0.49, 0.20, 0.060
  at l = 8, 32, 128, about 8.3/(l + 9), whatever the width (fixed-depth mean field; extending it to L ~ n needs
  finite-width control). That is why one merge at a rough cut l leaves a width-free ~(3-9)/l of the variance
  (l D_R(1) ranges 0.2-8.6 over all cuts at L >= n, so the law is for the rough cuts only).
- The formula (HEURISTIC). For forward-only cell methods, N_l(eps) ~ max(0, (d_l/p) log2(D_l(1)/eps^2)): d_l an
  effective dimension of the layer-l cloud (fitted ~1.4 PR_l over 4-256 cells, PR_l ~ n/l: the past, the Lyapunov
  count); D_l(1) the normalised one-reference distortion, which in the rough window is about twice the future's share
  of the pair variance (the future); p the exponent of the distortion in the cell size (1 for radial merges in the
  rough window, 2 for radial merges at small angles, 3 for frozen merges). Two refinements from the co-prover [A r4].
  First, for a cloud with the exponential Lyapunov spectrum lambda_k ~ exp(-2 (k - 1) l/n), the dimension that a
  quantiser sees grows with its resolution, d(r) ~ (n/l) log(sigma/r), so the bits grow like
  (n/(2l)) ln^2(D_l(1)/eps^2)/ln 2: quadratically in the log of the precision (close to the formula when
  D_l(1)/eps^2 ~ 10). Second, the lower-bound version (my rough-regime analogue of Cor 7.2: log2 M >=
  d log2(a (E|h|^2/V)(d/(d+1)) theta/eps^2) - log2 K) is PROVEN for an idealised source under a linear minorant
  rho(phi) >= a min(phi, phi_0) of the future-averaged radial distortion and a small-ball condition, but both
  hypotheses fail globally for nets: rho(phi) <= phi^2 at small angles, and the layer-l angular law has atoms
  (single-neuron input cones map onto coordinate rays). A lower bound for nets would need a minorant on a window of
  cell sizes and a small-ball bound for the bulk of the cloud.
- MEASURED. Post hoc on the 64 E1 nets: V_ang/E|A|^2 from mean field within [2/3, 1.5] on 94% of nets; the merge
  exponent gamma_R x PR has median 0.72 in 117 cells with theta_median L'/2 > 2. Registered (Amendment 5, 12 new nets:
  192 x 192, 128 x 256, 384 x 96, 512 x 64; all six predictions hit, `results/bits5_scores.json`): the log-radius
  profile within [0.5, 1.5] of mean field on 48 of 48 (net, layer) cells (median ratio 1.01); V_ang/E|A|^2 within
  [2/3, 1.5] on 10 of 12 nets (configuration medians 0.88-1.15); D_R(1) against its future-half prediction,
  configuration medians 0.49-1.96; gamma_R x PR median 0.74 over 30 rough cells; the zero-input prediction of
  D_R(256) within 3x on 73%; the bits formula agrees on 82 of 90 (rough cell, eps) pairs. That count needs
  unpacking: 63 of the pairs need more than 8 bits (both agree), 14 need none (both agree), and of the 13 resolved
  nonzero counts only 5 are within 2 bits, the formula over-predicting by a median factor 1.8. Post hoc on the old
  nets the same split is 249/251, 36/36 and 23/64. So the formula separates the cuts that need many bits from those
  that need none, and is good to about a factor 2 in between.
- What it says. At 512 x 64 the formula puts 50-600 bits at the cuts l = 8-48 (eps^2 = 0.1-0.01): a table of
  ~2^N cells, exponential in the cloud's Lyapunov dimension, which is the hash picture made quantitative with
  d_l ~ n/l in place of k*. At 192 x 192, l = 96, it puts 0 bits at eps^2 = 0.1 and about 6 at 0.01 (measured 2.9).
  As an idealised comparison, full enumeration to a pointwise distortion eps^2 is cheaper than eps^{-2} samples only
  where d_l log2(D_l(1)/eps^2) < log2(1/eps^2): where d_l is O(1) or D_l(1) is near eps^2, the late layers that make
  little of the variance. This is not a necessary condition for beating sampling (a mean estimator can cancel,
  stratify and sample the residual) [A r4].

### 3.6 Cross-lane check: the bit-accounting lane's learned local codes (POST HOC; one net)
The bit-accounting lane trained local codes on ARC's 256 x 32 (`st.W_layer`, seed 4000): an MLP encoder of h_l alone
into b bits, a decoder to a layer-l state, then the true future, to nMSE 0.1 of V for Gaussian inputs. Their counts,
as relayed by main: more than 256 bits at l = 0 and 2, 230 at l = 4, 104 at l = 8, 34-38 at l = 16, 19-21 at
l >= 24, with the output's own count about 19. On disk (`work/bit-accounting/runs/A1_arc4000_*.json`): at l = 16,
34-45 (channel capacity) and 45-52 (finite scalar quantiser, real bits); at l = 31, 19-23 and 26; at the output,
19-23. I computed the formula's inputs on the same net (`scratch/compare_bitacc.py`, `results/compare_bitacc.json`;
unit inputs with the radius kept exactly; converting to their V changes the residuals by 3%).

| l | PR | rho | D_R(1) | formula, d = 1.4 PR (k-means exponent) | formula, d = PR | learned local code |
|---|---|---|---|---|---|---|
| 1 | 161 | 19 | 1.55 | 883 | 629 | more than 256 (at l = 0) |
| 2 | 105 | 16 | 2.26 | 657 | 469 | more than 256 |
| 4 | 60 | 12 | 1.39 | 315 | 224 | 230 |
| 8 | 33 | 6.8 | 1.05 | 157 | 112 | 104 |
| 16 | 17 | 2.6 | 0.65 | 63 | 45 | 34-38 (channel), 45-52 (quantiser) |
| 24 | 13 | 0.98 | 0.50 | 42 | 30 | 19-21 |
| 28 | 12 | 0.49 | 0.47 | 36 (frozen merges, smooth regime: 4) | 26 | 19-21 |
| 31 | 11 | 0.12 | 0.44 | 33 (frozen merges: 0) | 24 | 19-23 (channel), 26 (quantiser) |

- In the rough layers it is the same quantity, up to the radius. A radial merge is a code: the cell index plus the
  state's radius (one real number, a few bits at nMSE 0.1). So there my cells are a forward-only code of h_l, a
  subclass of the lane's local codes (whose decoders are trained through the true future), so the optimum of their
  class cannot exceed my count plus the radius bits. Their learned counts are consistent with that: 230 against 315,
  104 against 157, 34-52 against 63. With
  d = PR, the exponent an efficient quantiser of a PR-dimensional source would reach (k-means reaches only
  1/(1.4 PR)), the formula gives 224, 112 and 45: within 3-8% of the learned codes at l = 4 and 8, and between their
  channel and quantiser counts at l = 16. This is POST HOC on one net, and I chose d = PR after seeing their numbers.
  A check that needs no choice of d (`scratch/compare_bitacc_wf.py`, `results/compare_bitacc_wf.json`): Gaussian
  reverse water-filling on the measured angular spectrum of the layer-l cloud, at the angular error that brings the
  radial residual to 0.1, gives 276, 157, 56 bits at l = 4, 8, 16 (and 31, 25, 22 at l = 24, 28, 31) if the residual
  falls like the angle squared (the smooth law, which holds at the fine angles these targets need), and 732, 507,
  237 if it stays linear in the angle down to the target. The learned codes sit within 1.6x of the first at every
  layer, so the agreement is robust in order of magnitude; its 3-8% precision with d = PR should not be over-read.
- In the smooth layers (l >= 24) they are different quantities. There the right merge keeps the whole state and
  quantises only the future's gate pattern (frozen-gate merges, 0-15 bits): that is not a code of the state. Any code
  of the state must carry the output's own information, about 19 bits at nMSE 0.1 (the lane's theorem: every layer
  needs at least the output's bits), and that is where their counts settle. My formula has no such floor: it counts
  only what the future still makes from the layer-l direction beyond what the merge carries for free.
- So the cross-lane answer: at ARC's rough layers the learned local-code bits are PR_l log2(D_l(1)/eps^2) (within
  about 10% at l = 4 and 8, and inside the spread of their two code types at l = 16): the cloud's Lyapunov dimension
  times the log of the variance share that the future still makes from layer l. At the smooth layers they are the
  output's own count. A registered test on more nets and widths (both lanes' codes on the same nets) would settle
  whether d = PR holds generally.

## 4. Lower bounds (FORMAL section 7)
- Thm 7.1 [C] (PROVEN): forward-only cell sweeps need log2 M >= R_rho(eps^2 V), the rate-distortion function of the
  cloud under the future-averaged merge distortion (randomised codebooks: condition on the public randomness [A r2]).
- Cor 7.2 [C, corrected by A r2] (PROVEN UNDER a regular d-dimensional angular source and a cubic minorant): the
  distortion depends on the radius and the angle only, so the problem is angular quantisation under the
  radius-weighted law, and log2 M >= (d/3) log2(E R^2/D) + O(1) with d an actual source dimension.
- Thm 7.3 [C, corrected by A r2] (PROVEN UNDER the same hypotheses): averaged over futures, every future-blind policy's
  cell count diverges as D -> 0 while cells aligned with the future's at most 2^{n L'} gate regions are exact. This is
  a separation in representation; it is not a mean-estimation separation (cancellations; exact conditional means
  make every partition mean-exact) [A r2].
- Thm 7.4 [C] (PROVEN identity; conditional upper bound): with M past-chosen cells, references and merge rule and
  exact cell statistics, E|e|^2 = sum_k P_k^2 E|Rbar_k|^2 + cross terms, E|Rbar_k|^2 <= D_k (Jensen), the cross
  terms being mixed second differences of the back kernel between cells. With equal-mass cells and negligible cross
  terms, E|e|^2 <= D/M, the error of a stratified sampler with one draw per cell; the cross terms are not controlled
  in general. It is not a lower bound (residual means can cancel). MEASURED (POST HOC): bias/(D/M) median 0.82
  [10-90%: 0.18, 3.8] rough (453 cells), 1.54 [0.27, 8.2] smooth (315).
- Thm 7.5 [C]. (a) Linear point rules (values and derivatives at M past-chosen points, past-chosen weights; every
  merge whose references and merge type are fixed by the past, which excludes our future-selected "best" merge): the
  future-averaged error is the kernel discrepancy under the exact finite-width back kernel (PROVEN), whose
  angular part has the cubic singularity 1 - phi^2/2 + (S_{n,t}/(3 pi)) phi^3 [A r2]. The rate error^2 >= c A
  M^{-1-3/d} for a regular d-dimensional source is a CONJECTURE until a kernel-quadrature theorem with checked
  spectral hypotheses is applied. (b) Adaptive or nonlinear point methods: M^{-1-2H/d}, PROVEN UNDER (G), a Gaussian
  surrogate of the future with an isotropic singular part A |h - h'|^{2H}; (G) does not describe adaptive queries to
  actual finite-width nets. Bits under (G): log2 M >= log2(A/(eps^2 V))/(1 + 2H/d) - O(1).
- Prop 7.6 [A] (PROVEN): query tables with unknown common bias need Theta(min(K, eps^{-2})) queries.
- Prop 7.7 [C] (PROVEN, assuming q << mu, matching marginals on the modelled features and integrability): a hybrid's
  error (estimate minus truth) is +E[r (dq/dmu - 1)], so the relevant features suppress the forward model's error by
  sqrt(residual); there is no general mean lower bound from the feature count N alone [A r2]: a linear future has a
  large N and an exact mean.
- Prop 7.8 [C] (cost accounting for a specified dense implementation; not a lower bound for the closure class): that
  implementation of an order-K closure costs ~n^{K-1} passes, so at fixed eps it is dearer than MC once
  n^{K-1} > eps^{-2}; by the same arithmetic, at fixed eps only sweeps with O(n^2 polylog) work per layer can win by a
  growing factor.
- Exact single-Gaussian recursion [A r3]: at L = alpha n, for fixed alpha and all large enough n, its annealed error
  E_W|m_L - mu|^2 is at least c(alpha) E_W V/n (c = 1.1e-3 at alpha = 1), PROVEN UNDER the one-layer radius
  hypothesis (H_R); a 1/n floor is unconditional for one Gaussianisation step followed by the exact suffix. See
  section 9.

## 5. Upper bounds and the comparison with the multilevel sampler
All sampler numbers are ideal propagation-only ratios: setup, prefixes, clustering and assignment are not charged.
- Two-level merge samplers (registered E3; ideal variance x cost relative to radius-exact MC, below 1 = gain): the
  one-reference sampler of the unrestricted lane, as the configuration median at the best common prefix depth r, is
  at 1.06 at ARC's 256 x 32 (no gain), 0.99, 0.80, 0.81, 0.46 at L = n (n = 32, 64, 128, 256) and 0.64 at L = 4n; as
  the median over nets of each net's best r (the statistic registered in P9) it is 1.06, 0.94, 0.73, 0.74, 0.43 and
  0.54. With 256 forward-only references the residual falls, but level 1 then carries the full variance: 0.73, 0.69,
  0.66, 0.42 at L = n, 0.32 at L = 4n, and 2.0-2.2 (worse than MC) at 256 x 32 (configuration medians at the best r).
- Full multilevel (registered Amendment 1; best nonempty hierarchy of dyadic levels, chosen on the test split, so
  optimistic): one reference trajectory gains
  1.21, 1.41, 1.58, 2.43 at L = n (n = 32, 64, 128, 256), 2.16 and 3.92 at L = 4n (n = 32, 64), 0.99 at 256 x 32.
  M shared reference trajectories (best of M = 4-256): 1.43, 1.46, 1.58, 2.48 at L = n, 3.36 and 5.50 at L = 4n,
  0.49-0.65 at 256 x 32.
- Deterministic merged estimates: with exact cell statistics their error is about D(M)/M (e.g. 6e-5 at 128 x 128,
  l = 64, M = 256), but the cell statistics are a depth-l problem; with a computable baseline the idealised gain over
  MC is eps^{-6/(d+3)} (smooth) or eps^{-2/(d+1)} (rough) [A r2]: a power of the precision, not of the width.
- Regimes (HEURISTIC; multilevel bounds PROVEN UNDER H_ML [A]): L < sqrt(n): closures' regime, no growing multilevel
  gain; illustrative arithmetic under ARC's conjectured (L/n)^K law (an upper bound, not known to be tight): an order-K
  closure costs ~n^{K-1} passes and reaches nMSE (L/n)^K, where MC needs (n/L)^K passes, so it wins polynomially only
  for L < n^{1/K}. sqrt(n) < L < n: multilevel gains L/(log^2 L + n/L). L = alpha n: n/log^2 n for the multilevel
  sampler; ARC's K = 2 closure (kind 'simple', which keeps only the linear term of the off-diagonal covariance [A r3])
  loses ground slowly (nMSE ~ n^{-0.4..-0.74} at ~1.5 n passes; hardness lane); the exact single-Gaussian recursion
  gained at most about 3x up to n = 256, and under (H_R) its annealed gain is bounded by a constant for large n
  (section 9). Within the classes our bounds cover (forward-only cells, proven; past-chosen linear point rules,
  conjectural rate; adaptive point methods under (G)) nothing gains more than constants; adaptive point methods on
  actual nets and corrected closures are not covered. L >> n: the cloud becomes nearly radial and every method
  profits, but the variance is still made in the first layers, which the multilevel sampler already samples cheaply.
  A polynomial gain over the multilevel sampler at L = alpha n would need a method outside those classes, for example
  an averaging method that carries the first layers' variance through the long rough future (conjecture C4).

## 6. Kelvin's hash argument, revisited (FORMAL section 9)
- Proposition 9.1 (PROVEN [C+A], corrected by A r2). Model: K = 2^N equally likely states, table values iid across
  states over the random weights with a known ensemble mean; ARC's averaged criterion. Partial enumeration (observed
  values plus the known mean for the rest) has risk (K - T) s^2/K^2 against (K - 1) s^2/(K T) for T-sample MC: a gain
  K(K-1)/(T(K-T)), at least 4(K-1)/K at every budget (the minimum, at T = K/2; it is K at T = 1, and enumeration is
  exact at T = K). The gain comes from the ensemble prior alone: the values are iid with a known mean (a uniformly
  random function with a known alphabet mean is one example), and no piecewise-linear structure, kink or non-uniform
  input law is used. For a fixed unknown table, sampling without replacement gains only 1/(1 - T/K). With an
  unknown common bias and point queries it needs Theta(min(2^N, eps^{-2})) queries (Prop 7.6) [A]. So "hash" must
  say which: a random hash with a known ensemble mean self-averages; an unknown-bias hash under point access is as
  hard as sampling.
- True (S1, PROVEN UNDER (G); MEASURED for our construction): for point-evaluation methods the future beyond its
  linearity scale is a rough field on a cloud with ~n/l effective directions; merges with exact cell statistics do no
  worse than stratified sampling when the cells have equal mass and the cross terms are negligible (Thm 7.4, an upper
  bound) and measure about that size (bias ~ D/M); under (G) the deterministic advantage is at most M^{2H/d}
  (Thm 7.5b). ARC's 256 x 32 is in this regime at every cut we measured but the last (l = 28).
- True (S2, PROVEN in the cell class, averaged over futures): forward computations cannot see the future's frame
  [A 5.4, R, K7], and for cells this costs an unbounded number of bits at high precision (Thm 7.3).
- False (F1): the count is not exp(k*): the past gives d_l ~ n/l directions, the future the ratio rho_l and the
  exponent H; k* does not enter.
- False (F2): the final layers are the easiest: the future is smooth on the collapsed cloud for l >~ 0.8 L and the
  measured cell bits at 7L/8 are at most 1.7 at eps^2 = 0.01 for L >= n (our construction).
- False (F3): "hash implies hard to average" (Prop 9.1); averaging methods integrate kinks analytically and evade the
  bits. Their obstacle is different: for the exact Gaussian closure it is the radius defect of Gaussianisation, which
  the future does not forget (round 3), and for a specified dense order-K implementation the cost (Prop 7.8).
- False (F4): "nothing beats sampling": the multilevel sampler beats plain MC; whether anything beats the best
  multilevel sampler by a growing factor is open.

## 7. What it means for ARC's agenda
- The bits that matter live where the variance is made: the first few dozen layers, absolute, at any width, where the
  cloud has ~n/l directions and the remaining future is long and rough. Late layers have few bits but little variance
  left. Measured directly: one radial merge at absolute layer l leaves (3-9)/l of the variance (e.g. 0.047 at l = 64
  of 256 x 256, 0.074 at l = 64 of 128 x 128, 0.32 at l = 16 of 64 x 64), so at L = n the middle layer's residual falls
  roughly like 1/n.
- At ARC's 256 x 32 the whole net is in that early regime (l/n <= 1/8): PR 13-33 at the measured cuts, none of our
  merge constructions helps, the one-reference multilevel gains nothing (0.99), and the best measured deterministic
  method (K = 2 closure, about 2x at its own cost, hardness lane) is a constant-factor method. The theory explains why
  rather than proving it.
- For ARC's own route (cumulant propagation and Gaussian closures) at depth proportional to width: the exact
  single-Gaussian recursion gained at most about 3x up to n = 256 (whether its gain has peaked is not settled), and
  under (H_R) its annealed gain is bounded by a constant for large n at fixed alpha. The proven part of
  its error is what Gaussianisation throws away (the third and fourth cumulants, which the future sees through the
  radius); the measured error is two to three decades larger than that floor, and which layers' defects make it has
  not been measured. HEURISTIC: a closure that beats sampling by a growing factor there would have to carry those
  cumulants, or their effect on the radius, through about alpha n rough layers at O(n^2 polylog) work per layer;
  that is the concrete form of C4.

## 8. Measurements: pre-registered scorecard
64 nets in nine configurations (L = n: n = 32, 64, 128, 256; L = n/2: 64, 128, 256; ARC's 256 x 32; L = 4n: 32 x 128),
seeds 100-107 (100-103 at n = 256), no dead nets; Amendment 1 on 44 of them plus 8 nets at 64 x 256; Amendment 2 on
12; Amendments 3-4 on 76 nets in eleven closure configurations from 16 x 8 to 256 x 256, with Gaussian inputs (48 of
them are the E1-E3 weight instances); Amendment 5 on 12 new nets (192 x 192, 128 x 256 seeds 100-103; 384 x 96,
512 x 64 seeds 100-101).
Scored by the hashed scorers (`results/prereg_scores.json`, `results/ml_scores.json`, `results/gain_scores.json`,
`results/closure_scores.json`, `results/bits5_scores.json`; Amendment 4 by its registered hand rule, reproduced by
`scratch/score_closure4.py` in `results/closure4_scores.json`).
| prediction | outcome |
|---|---|
| P1 cubic exponent of the frozen residual in [2.4, 3.6] on >= 80% of cells | missed narrowly: 79% of 169; median 2.91 (10-90%: 2.19-3.43) |
| P2 nets' cubic coefficient within [1/3, 3] of the exact chain's | missed: 47%. The chain's output-normalised mean is inflated at small width (the inverse-norm effect found in round 2): net/chain 0.06, 0.35, 0.52, 0.77 at n = 32-256, while the nets' own coefficients 0.84, 0.68, 0.51, 0.46 approach the mean-field 4/(3 pi) = 0.42. The chain's small-angle jitter floor (~1.6e-12) and Monte Carlo error also enter, so the whole discrepancy cannot be put on the inverse-norm effect |
| P3 frozen better for L' phi <= 0.5, radial better for L' phi >= 5 | hit: 84% of 105 |
| P4 one reference: frozen worse at L/4 (>= 90%); with 16 cells frozen better at 7L/8 (>= 80%) | missed narrowly on the first clause (89%); second 100% |
| P5 decay exponents of the merge residual in the registered bands | missed: 17 groups pass, 10 fail; at ARC's 256 x 32 and at L/4 of L = n/2 nets the residual barely falls (gamma 0.00-0.07), at L = 4n it falls faster than predicted (0.5-1.6) |
| P6 two-sided cells <= 0.8 x forward-only at M = 64 on >= 60% | missed: 56%; median ratio 0.77 (10-90%: 0.54-0.99) |
| P7 bias of merged estimates <= D/10 for M >= 16 (where resolved) | hit: 95% of 730 |
| P8 mean-propagation error decreases from L/4 to 7L/8 (>= 80%); <= 0.05 at 7L/8 (>= 75%) | hit as worded (endpoint decrease on 64 of 64 nets; second clause 100%). The hashed scorer used a stricter reading (non-increasing over all four cuts), under which it misses (61%; the deep values sit at the 4096-sample noise floor). Counted as a hit in the main score |
| P9 one-reference two-level sampler in [0.5, 1.2]; multi-reference below it on >= 70%; at most 1.7x better on >= 80%; no growth from n = 64 to 256 | missed on "below it" (50%: multi-reference loses at L <= n/2); the other clauses hit (91%, 97%, 0.91x) |
| P10 eigenvalue log-slope in [-0.7, -0.15] (>= 90%); PR falls with l (>= 80%) | missed narrowly on the first clause (85%); second 95% |
| P11 at L = n: more than 8 bits at L/2 for eps^2 = 0.01 (>= 75%); at most 6 bits at 7L/8 (>= 75%) | missed on the first clause (43%): the bits at L/2 fall with width; second 100% |
| A1 one-reference multilevel gains in the registered bands | hit in all seven configurations |
| A2 that gain grows 1.0-2.0x from n = 32 to 256 at L = n | missed narrowly: 2.01x |
| A3 several reference trajectories do not beat one (>= 75% of nets) | missed: 65%; they beat it by more than 1.25x on 35% of nets (L >= 4n, small n); full dyadic versions are worse on 75% |
| G1 rough-gain law within [1/3, 3] at each width and stable within 2x | missed narrowly on stability (2.04x); within the band at every width (1.02, 1.77, 0.50) |
| B1 closure nMSE falls no faster than 1/n (log-log slope in [-1, 0]), n = 16-128 | hit at L = n/2 (-0.72); missed at L = n (-2.05) |
| B2 closure gain at its matrix-product cost <= 10, and grows at most 2x from n = 16 to 128 | hit at L = n/2 (0.39x); missed at L = n (0.38 -> 3.26, 8.6x) |
| B3 (Amendment 4, after B1-B2 were seen) median gain at 256 x 256 <= that at 128 x 128 (3.26) | hit: 1.02 (192 x 192: 0.87) |
| B4 median gain at 256 x 128 <= 2x that at 128 x 64 (0.89) | hit: 0.40 |
| Q1 (Amendment 5) per-net log-radius variance at L/8, L/4, L/2, L within [0.5, 1.5] of mean field on >= 75% | hit: 48 of 48 cells; median ratio 1.01 (configuration medians 0.91-1.15) |
| Q2 V_ang/E\|A\|^2 within [2/3, 1.5] of mean field on >= 75% of nets, configuration medians in [0.8, 1.25] | hit: 10 of 12 nets; medians 0.88 (128 x 256), 0.89 (192 x 192), 1.15 (384 x 96), 0.95 (512 x 64) |
| Q3 D_R(1) against its future-half prediction, configuration medians in [1/3, 3] | hit: 0.49 (128 x 256), 0.58 (192 x 192), 1.96 (384 x 96), 1.28 (512 x 64) |
| Q4 rough cells: median gamma_R x PR in [0.45, 1.1] | hit: 0.74 over 30 cells |
| Q5 zero-input D_R(256) within 3x on >= 60% of rough cells | hit: 73%; median ratio 0.60 |
| Q6 bits formula agrees with the measured cell bits on >= 70% of (rough cell, eps) pairs | hit: 82 of 90, made of 63 pairs where both exceed 8 bits, 14 zero-bit pairs and only 5 of 13 resolved nonzero counts within 2 bits (median over-prediction 1.8x) |
Reading: the qualitative claims held (the cubic exponent on average, the crossover, merged estimates of about the size
of stratified sampling, merges helping only in the smooth regime, constant-factor multi-reference gains, the rough-gain
law's size and shape). The misses are bands set too narrow, the P2 normalisation problem, and two wrong expectations:
P11, and the closure's fast decline over n = 16-128 at L = n (B1-B2). B3-B4 then hit, but with 4 nets at n = 192
and 256 and per-net gains spread over two decades the evidence for a peak is moderate. POST HOC
(`results/posthoc_closure.md`): the per-net log-log slope is -1.97 +- 0.35 over n = 16-128 and +0.5 +- 1.3 over
128-256, and the n = 192 and 256 means sit 1.9 and 1.7 SE above the n^{-2} trend extrapolated from n = 16-128.
H_peak is favoured, not established. Amendment 5 tested the bits formula of section 3.5 out of sample: its bands were
calibrated post hoc on the old nets (disclosed in PREREG), and all six held on configurations never run before,
including two wider (384, 512) and one deeper (L = 2n) than any calibration net; the bit-count clause held through
its many-bit and zero-bit cases, not the intermediate counts.

## 9. Conjectures and the bottleneck (FORMAL section 11)
- C4 (BOTTLENECK; = C0* restricted to sweeps; astra's C_comp): at L = alpha n, every polynomial-memory constant-pass
  sweep costs at least n^{-o(1)} times the best shared-trajectory multilevel sampler. The bounds above cover
  forward-only cells (proven), past-chosen linear point rules (conjectural rate) and adaptive point methods under the
  surrogate (G); closures, and adaptive point methods on actual nets, are the gap. A K-only floor cannot hold for
  classes containing polynomial Gaussian mixtures (they contain MC) [A r2]. Sufficient target for a specified closure
  class: cost at precision eps = n^{-q/2} at least
  n^{max(1, q) - o(1)} [A r2]. Evidence: ARC's K = 2 along L = alpha n has nMSE slopes -0.41/-0.56/-0.74 in n at
  alpha = 1/8, 1/4, 1/2 against MC's -1 at equal cost, and its gain falls from 2-8x at widths 32-128 to about 1 at 1024
  (hardness lane).
- C4 for the exact single-Gaussian recursion (`lib/closure.py`; round 3, `coprover/astra_r3_answer.md`; checks
  `logs/check_r3.log`).
  - PROVEN [A r3]: the error is exactly a sum of Gaussianisation defects tested against the exact future's kernel, with
    cross terms of no fixed sign; the kernel has a radius component (a_n^{2(t-1)}/pi)|h||h'|, a_n = 1 - 5/(8n) +
    O(n^-2), which survives a suffix of depth alpha n with the factor e^{-5 alpha/4}; the first Gaussianisation raises
    the mean radius by 9/(16 sqrt n) + o(n^-1/2), because it discards the third and fourth cumulants of relu(W x).
    Hence one Gaussianisation step after layer 1, followed by the exact suffix, has
    liminf n E|error|^2 >= (81/(256 pi)) e^{-5 alpha/4} (0.029 at alpha = 1), unconditionally.
  - PROVEN UNDER (H_R) [A r3]: (H_R) says that Gaussianisation does not lower the mean radius on average over a fresh
    He layer, for every Gaussian input law (CONJECTURE; the pointwise version, for a fixed layer, is false). Under it,
    at fixed alpha and for all large enough n, E_W|m_L - mu|^2 >= c(alpha) E_W V/n with c(alpha) =
    81 e^{-5 alpha/4}/(512 pi C_V), C_V = 13.481, so c = 1.1e-3 at alpha = 1: at its own cost of 2n passes the
    recursion's annealed gain over MC (E_W of the squared error against E_W V, not a median or per-net gain) is at
    most 1/(2c), about 470. If (H_R) fails, a faster decline needs later radius defects that cancel the first to
    leading order.
  - CHECKED [C]: n(1 - a_n) -> 5/8 (exact sum); sqrt(n) E rho_1 = 0.536, 0.556, 0.561 at n = 16, 64, 256 (-> 0.5625);
    (E|H|^4 - E|G|^4)/n -> 9/2; the counterexample; the constants (C_V by a stable recurrence for 1 - c_j, whose tail
    constant reaches 9 pi^2/2 as it should; the reviewer's correction). ARC's K = 2 'simple' kind keeps only the
    linear off-diagonal covariance term (0.125 against the exact 0.1453 at correlation 1/2), so the hardness lane's
    slopes describe a different recursion [A r3, checked in the code's diagram budget].
  - MEASURED (registered Amendments 3-4): median gain at the matrix-product cost at L = n of 0.38, 0.28, 0.63, 3.26,
    0.87, 1.02 for n = 16, 32, 64, 128, 192, 256; at L = n/2 2.28, 0.59, 0.80, 0.89, 0.40 for n = 16-256. At the
    estimated implementation cost (an operation count for Phi2 by 48-node quadrature, about 2005 + 2n passes) the
    configuration medians are below 0.4 everywhere (single nets reach 5.3: 128 x 128, seed 103). The squared error
    is estimated by the cross-product (a - mu1).(a - mu2), an unbiased numerator: it removes the upward bias of
    reference sampling, not its noise (relative SE 0.36 and 0.14 on the two nets with the smallest errors, replayed
    by the reviewer; smaller on the others and below the between-net spread). The measured annealed ratio
    n E[err^2]/E[V] is 0.21-1.25 (0.2-1.0 at L = n), two to three decades above c(alpha), so the floor bounds the gain
    without predicting where growth stops; these finite-width values neither test the asymptotic theorem nor (H_R).
    Numerics: Phi2 with 48 and 128 quadrature nodes agree to 1e-14 (relative) on those two nets
    (`results/quadrature_check.jsonl`), and the reviewer found correlations up to 0.999906 and negative covariance
    eigenvalues of at most 1.2e-16 (relative) at 256 x 256, seed 103: no material instability, though these checks do
    not exclude every numerical cause.
- C2 (decay with references) PROVEN UNDER a regular source [A r2]; which d applies to the nets is open.
- C3 (rough gains) PROVEN in the iterated limit (c = 6 pi) [A r2]; t = alpha n needs more estimates.
- C5 (two-sided region count) open.

## 10. Process, review, compute, deviations
- Co-prover (gpt-6-astra, xhigh, read-only; model line checked in every raw log; `coprover/PROVENANCE.md`).
  - Round 1 (21:06-21:25Z), independent, from the lemma-free `coprover/STATEMENT_r1.md`. My v1 was frozen at 21:18Z
    before its answer. Found by both: the model, the cap on information bounds, layer-free trajectory information,
    the nonlinear-part-only point for the mean, the kernel-discrepancy form of forward-only defects, the easy random
    hash, the multilevel allocation, the computational bottleneck. New from astra: the exact forward-only
    characterisation, the worst-case INDEX separation, the unknown-bias table lower bound, the Gaussian water-filling
    rate, the multilevel regime map, annealed tangent-energy preservation and the future-blind subspace theorem, the
    infinite stationary second moment in the noise regime. New from me: measure-propagation sweeps and the telescoping
    identity, the Euler residual, the one-layer law, the mean-field frozen/radial recursion and crossover, the exact
    three-replica chain, the cell-class rate-distortion bound, the separation, and the measurements.
  - Round 2 (21:27-21:52Z), a claim-by-claim audit of v1 plus new derivations: the closed form and next term of the
    cubic law, the crossover constant 3 pi (2^{1/3} - 1), the raw finite-width coefficient (no width correction), the
    rough-gain constant 6 pi, the angular quantisation bound, the optimised gains, the closure-class caveats and the
    sufficient target for C4. Two of my v1 claims were wrong (the fixed-net separation; "at least 4x" in the hash
    proposition) and one interpretation was an artifact (the finite-width enhancement); all corrected in FORMAL v2.
  - Round 3 (22:17:59-22:32Z, model line checked): a lower bound for the exact single-Gaussian recursion. It was
    launched before Amendment 3's first evaluation run (22:21:02Z); its prompt cites the hardness lane's K = 2 results
    and my dev checks. Results in section 9; my numerical checks of its constants and expansions agree
    (`logs/check_r3.log`); the one-defect theorem's proof was read, not re-derived.
  - Round 4 (23:24-23:41Z, model line checked; FORMAL 6.6 v2.2 frozen at 23:21Z before it): an independent derivation
    of the variance profile, the cell lower bound and the effective dimension, then an audit of 6.6. Findings applied
    (FORMAL 6.6 v2.3, `coprover/PROVENANCE.md` round 4): the martingale needs the frozen-at-death convention (my
    survival-conditioned version was wrong); my profile constants came from a truncated sum (10.63, 0.058) and a
    wrong approximation (5.5/(l + 5.5)); a positive linear minorant is impossible at small angles (rho <= phi^2) and
    the cloud has angular atoms, so the lower bound is a theorem for an idealised source only; the effective dimension
    of an exponential-spectrum cloud grows with resolution (bits quadratic in log(1/eps)); the fitted 1.4 PR is an
    empirical exponent, not a source dimension; the 88% agreement mostly counts censored and zero-bit cases (23 of 64
    resolved counts); the enumeration comparison is not a necessary condition for beating sampling. My checks: the
    profile constants (B_inf = 21.301) and the censoring split reproduce.
- Review 1: gpt-6-astra in a fresh session (`review/review_gpt6.md`, prompt and hashes in `review/`), verdict "major
  revision", 16 findings, all answered (FORMAL section 13 lists them with the change). Fourteen are applied in the
  text; finding 11 (k-means) has a code fix and a disclosure but the affected results were not regenerated (the
  reviewer's replay of one net changes a two-sided D_best by 3%); the scorer-timing point is a disclosure only. In
  short: (1) Thm 7.4 is an
  upper bound, not "references are samples" as a lower bound; (2) Thm 7.5's rate is a conjecture without a checked
  kernel-quadrature theorem, and (G) does not cover adaptive queries to real nets; (3) Prop 7.8 is the cost of one
  implementation, not a class bound; (4) D_best reads the future (merge type) and the two-sided assignment uses test
  outputs: relabelled; (5) universal claims narrowed to the tested construction, cuts and sizes; (6) sampler ratios
  labelled ideal propagation-only; (7) P8 scored under both readings (hit as worded); (8) the chain's jitter floor
  softens P2's reading; (9) criteria and the radius conversion stated; (10) cumulant truncations are not measures:
  Def 4.1 restricted; (11) k-means could return labels from before the last centre update: fixed in `lib/bb.py`
  after the registered runs (the registered results used the hashed code; the reviewer's replay changes one
  two-sided D_best from 0.0422 to 0.0436); (12) the F1 "best subset" excluded plain MC: now "best nonempty
  hierarchy"; (13) the hash proposition's model and comparator made exact; (14) cell bits disclosed as interpolated,
  future-chosen merge type, test split at M = 1; (15) several summary numbers corrected (F1 44 + 8 nets, P5 17/10,
  bias counts 453/315, E3 medians); (16) hypotheses added to Thm 4.2, Lemma 4.3, Thm 5.1, Prop 5.3, Prop 7.7 and
  Cor 7.2.
- Review 2: gpt-6-astra in a fresh session on the closure test, round 3 and the application of review 1
  (`review/prompt_review2.md`, `review/review2_gpt6.raw`, model line checked; 22:56-23:03Z), verdict "major
  revision", 14 findings; it reproduced every registered score and quoted statistic. Changes: (1) "the error stops
  falling, the gain peaks, the lead is closed" softened to moderate evidence for a peak (B3-B4 hit; post hoc
  1.7-1.9 SE above the extrapolated trend) and "weakened, not closed"; (2) the qualifiers "at fixed alpha, large n",
  "annealed" and "under (H_R)" carried into every statement of the closure floor; (3) the cross-product described
  as an unbiased numerator, with its reference noise; (4) the numerical claim narrowed to the specific convergence
  checks; (5) "actual cost" became "estimated implementation cost", with the per-net exception; (6) C_V recomputed
  with a stable recurrence (13.481; the old tail constant was wrong, the value survives); (7) the closure population
  (76 nets, 48 shared with E1-E3) and round 3's chronology corrected; (8) Thm 7.4's equal-mass and cross-term
  conditions restored wherever it is used; (9) the uncovered classes named (adaptive point methods on actual nets,
  corrected closures), future-selected merges excluded from the linear-rule identity, the l = 28 exception kept;
  (10) the jitter caveat placed beside P2; (11) the hash gain stated as at least 4(K-1)/K (K at T = 1) with the iid
  model; (12) Prop 7.7's sign corrected to +E[r(dq/dmu - 1)] with its assumptions; (13) both E3 aggregation orders
  reported; (14) this section now separates applied fixes from disclosure-only answers. The PREREG text and the
  hashed `scratch/run_closure.py` keep the phrase "without the samplers' noise"; it should read "without the
  samplers' upward bias".
- Pre-registration: PREREG.md hashed at 21:32Z before any evaluation run (the header time was mistyped 21:50Z and
  rehashed at 21:32:27Z with no data in between); Amendment 1 hashed at 21:45Z before any F1 run (disclosed: E1-E3
  results for all configurations except 256 x 256 had been seen); Amendment 2 at 22:04Z, Amendment 3 at 22:20Z,
  Amendment 4 at 22:29Z and Amendment 5 at 23:17Z (with `scratch/run_bits5.py`, `scratch/score_bits5.py`; the first
  run started at 23:17:25Z), each before its runs (Amendment 4 after Amendment 3's results, Amendment 5 after all
  earlier results, both disclosed in their headers; header times fixed and rehashed with no run in between).
  The main scorer was hashed at 21:34:30Z, after the first evaluation outputs existed (the queue started at
  21:32:31Z) and before I first read any output (21:38Z, via `scratch/tables.py`); timestamps alone cannot verify the
  latter. Dev seeds 9001-9005 only calibrated; evaluation seeds 100-107.
- Deviations: (1) the F1 queue was stopped by PID at 21:51Z and restarted with one worker so that the scorer could run
  without exceeding two heavy processes; finished jobs were kept (outputs are deterministic) and two jobs that had
  started finished normally. (2) Post-hoc analyses (the PR law, decay exponents, cell bits, bias ratios, the layer
  residual law, the closure's geometric means and slopes) are labelled POST HOC. (3) The k-means fix above, after the
  registered runs, without regenerating the affected results. (4) The quadrature check and the round-3 checks are
  unregistered numerical checks, not measurements.
- Compute: CPU only, nice 19, idle IO, single-threaded BLAS, at most two heavy processes; no GPU. The machine was
  heavily shared (load 25-35 on 16 vCPUs).

## Handoff
- Theory: `FORMAL.md` (v2 with the round-3 and closure updates), `coprover/astra_r1_answer.md`,
  `coprover/astra_r2_answer.md`, `coprover/astra_r3_answer.md`, `coprover/PROVENANCE.md`.
- Code: `lib/bb.py` (nets, gate paths, frozen futures, merges, k-means on directions, angular spectra),
  `lib/meanfield.py` (mean-field recursion), `lib/chain3.py` (exact three-replica chain; note it reports
  output-normalised residuals, see Prop 5.3), `lib/closure.py` (exact single-Gaussian recursion); tests in
  `tests/test_bb.py`, `tests/test_closure.py`.
- Runs: `scratch/run_eval.py N L SEED` (E1-E3), `scratch/run_ml.py N L SEED` (F1), `scratch/run_gain.py` (Amendment 2),
  `scratch/run_closure.py N L SEED` (Amendments 3-4), `scratch/run_bits5.py N L SEED` (Amendment 5); queues
  `scratch/run_queue*.sh`; scoring `scratch/score_*.py`; tables `scratch/tables.py`, `scratch/posthoc.py`,
  `scratch/variance_made.py`; outputs `runs/`, `runs5/`, `results/`.
- Next, if wanted: (1) prove the two hypotheses behind the bits lower bound for actual nets: the linear minorant of
  the future-averaged radial distortion at finite width (an exact two-replica chain gives rho(phi) exactly), and a
  small-ball condition for the layer-l cloud with the measured exponential spectrum; (2) test (H_R) where it matters:
  the signed radius defects of the closure along its own trajectories (the co-prover's diagnostic, with a coupling
  that makes it cheap), and which layers' defects make the measured error; (3) the same for one corrected closure
  (Gaussian mixture or gate split) to see whether it carries the first layers' cumulants; (4) the noise regime
  (Model S) with the corrected angle law theta ~ Q^{-1/3}.
