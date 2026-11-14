# How many random bits does an estimator of a random net's mean need? Structured sampling on ARC's nets (2 November 2026)

Status (final, ~16:20Z): registered runs complete; co-prover rounds 1-2 and the gpt-6-astra review applied (section 8).

Lane: randomness (target M1), bead pool-ttdzrk.16 under pool-ttdzrk (request bead pool-ov3ikt). Work directory
`work/randomness/`: `FORMAL.md` (v1 frozen 13:32:08Z before the co-prover's answer, sha 67d8e227; additions E1, E2),
`PREREG.md` (part 1 hashed 13:45:49Z before any evaluation run; Amendment 1, the exponent analysis, hashed 13:50:47Z;
Amendment 2, the 512 x 64 test, hashed 15:00:43Z; `prereg_hashes.txt`, `frozen/`), `review/` (the final review),
`coprover/` (gpt-6-astra rounds 1-2, `PROVENANCE.md`), `notes/` (QUESTIONS.md, READING.md, LITERATURE.md with
LITERATURE_raw.md), `lib/rnd.py` with `tests/test_rnd.py`, `scratch/`, `runs/`, `results/` (scorecard.md, rates.md,
tables.md, ladder_table.json, tradeoff.json), `logs/` (deviations.txt). Labels: PROVEN, PROVEN UNDER <hypothesis>,
NUMERICAL, MEASURED, HEURISTIC, CONJECTURE. Provenance: [C] this lane, [A] gpt-6-astra, [prior: lane] an earlier
lane's report. Nets: ARC's He-ReLU family (W_l iid N(0, 2/n), no bias, x ~ N(0, I_n)); 256 x 32 nets 200-204
(2^26-path truths) and the ladder L = n/8, n = 32, 64, 128, 8 nets each (2^24-path truths). Gains are over plain Monte
Carlo at equal forward passes (MC: nMSE x N = 1).

## Summary for Kelvin

**What this is:** your question "how many random bits does an estimator really need?", made precise and then
attacked from the sampling side, as you asked at 13:05Z: how much structure can a sample set carry, and does any of it
change the error-vs-compute exponent? Theory with gpt-6-astra (two rounds), verified literature, 32 pre-registered
predictions on ARC's 256 x 32 nets 200-204 and the L = n/8 ladder, and a gpt-6-astra review (applied, section 8).

1. **The asymptote (exponents first).** Fitted exponents at 256 x 32 (risk ~ N^-p, MC p = 1; N = 2^8..2^18, five
   nets, median [95% interval over nets]): scrambled Sobol 1.045 [1.03, 1.08], a small positive finite-range excess over
   MC; the same in the first layer's polar frame (white-box) 1.058 [1.01, 1.08], with no detected difference from the
   standard frame (-0.002 [-0.04, 0.03]); fixed Sobol and lattice point sets 1.06-1.13 with wide, overlapping
   intervals. No structured sampler was shown to beat RQMC's exponent. Proven [A]: the best positive point set chosen
   without the weights has exponent between 1 + 2/(n-1) and 1 + 3/(n-1) (1.008-1.012 at n = 256); unions of
   independently rotated copies of a fixed design have exponent exactly 1.
2. **The constants at ARC's budget (2^16 passes), medians over nets:** antipodal Kerdock 5-design 2.9x over MC
   (2.2-3.9; the best median); scrambled Sobol 2.1x; Sobol in the polar frame 2.2x; pairwise-independent MC with
   16 kbit 1.5x. None of these randomised samplers reaches 4x on any net. Fixed point sets with zero random bits: 2.1x
   (centred Sobol) and 2.7x (polar frame) in median, but erratic per net (0.8x to 5.5x).
3. **Runtime randomness does not separate these estimators on ARC's ensemble.** Proven (existentially): some fixed
   point set matches radius-exact MC's ensemble risk at every budget; it gives no efficient construction. Proven under
   exact-transform primitives: seeds read off the weights simulate any randomised estimator (finite-precision weights
   do not supply unlimited bits). Measured: across nine orders of magnitude of (idealised) random bits the error at
   fixed compute does not track the bits: plain MC (5.4e8 bits) 1x, Kerdock (1e6) 2.9x, scrambled Sobol (1.4e5) 2.1x,
   pairwise MC (16,384) 1.5x, fixed sets (0) 2.1-2.7x. What bits buy is conditional unbiasedness and a replicate
   variance on the given net, not accuracy. "Noise sensitivity forces long seeds" has the direction wrong: MC-length
   seeds always suffice; at budgets polynomial in n the spectrum only forbids much shorter ones.
4. **Why the gains are constants (new theorem, astra round 2; I checked its structure and its agreement with the data,
   not every step).** At finite width the variance a point set fights is the accumulated fluctuation of activation
   norms in the early layers (E V -> 11.15 as n -> infinity with L/sqrt n -> infinity), which the infinite-width picture
   drops. The fixed-degree spectrum converges, so the gains of the simplex, cross-polytope and Kerdock designs converge
   to 2.09, 2.36, 3.15 (ensemble limits); measured medians at 256 x 32: 1.93, 2.39, 3.04. No positive weight-independent
   rule has a gain growing with width at polynomial budgets. A registered test of astra's finite-size bands at
   512 x 64 failed narrowly on all four (simplex 1.71, low-degree share 0.41; mean V = 10.6 sits near the limit): the
   limit is proven, the speed of approach is not. Plugging the measured spectra into the proven black-box bound gives
   an estimated ceiling of ~10-19x at ARC's budget (an estimate, not a certified bound; section 4.3); a 7-design would
   need 87x ARC's budget.
5. **Boundary mass is real but does not decide it.** The gate partition has expected perimeter ~nL/sqrt(2 pi) (exact
   [A]) and some gate flips with probability ~0.96 under noise 1e-6; the output is Lipschitz and decorrelates linearly
   (per-net slope median 18.6 at 256 x 32). What point sets can cancel lives at harmonic degrees <= 5 and is set by
   norm fluctuations, not gate counts.
6. **White-box structure buys little at L = n/8.** Gate-aligned Sobol: +31% at 2^14, gone at 2^16 (PC1 fails), worse
   than standard Sobol at 512 x 64 (1.7x vs 2.4x at 2^12): erratic, not a reliable constant. Aligned designs with flat
   coordinates are badly biased (Kerdock in the polar frame: nMSE x N = 53-204). First-layer controls nearly repeat
   what a 2-design does (one dev net, HEURISTIC). No common linear tail mask exists in the last two layers (sampled).
   The multilevel sampler is 0.83 -> 0.99 from n = 32 to 256 and 1.00-1.07 at 512 x 64: its gain needs depth in the
   hundreds (L / (2 log^2 L) >> 1, HEURISTIC).
7. **The tested bounded-independence designs do not fool these nets beyond MC**: the polynomial-hash k-wise designs are
   MC-like at their own size where resolved (worse than MC at 256 x 32: the pairwise design 0.5x); high-k cells are
   mostly below the truth floor. astra exhibits a worst-case 127-wise independent law that a width-256 ReLU net
   misreads by 2x.
8. **Verdict.** On random weights runtime randomness is not what separates estimators, so the framing does not make
   ARC's goal a derandomisation problem. It does give a checkable cap on weight-independent structure (best exponent
   1.008-1.012 at n = 256; gains that stay bounded as width grows; ~2.9x measured, an estimated ~10-19x ceiling), a
   mechanism for it (norm fluctuations), and a precise target for the sampling direction: a cheap weight-dependent
   predictor of the per-input norm profile. It also implies ARC's "avoids random or pseudorandom sampling" test has to
   be about evaluating the network at points, not about random bits.

Registered outcome: 32 predictions, 25 hold and 7 fail (the white-box frame at 2^16, two per-net bands, and astra's
four finite-size bands at 512 x 64; section 5). Decision for you: none needed; the next useful step is the
norm-profile control (item 8), registered at 256 x 32 and at L = n.

## 1. The question, sharpened, and where the conjecture was wrong

The conjecture as put to you: "deduction uses 0 random bits and sampling uses many; the question is the length of the
shortest pseudorandom seed (deterministic point set) that fools the net's output to precision eps; at depth
proportional to width the output's noise sensitivity forces long seeds, so ARC's goal is a derandomisation problem,
and the obstacle is boundary mass, not bits." Checked against definitions:

1. **"Sampling uses many random bits": no.** ARC's criterion is a mean squared error, for which Chebyshev needs only
   pairwise independence between samples: N samples need about 2 x (bits per input) random bits, whatever N is. Here,
   pairwise-independent MC with 16,384 random bits at N = 2^16 has MC's mean error (PROVEN; measured in E2, with a
   caveat about its skew, section 4.7).
2. **"The real step is from few bits to zero": on ARC's ensemble that step costs no precision.** PROVEN [C, A
   independently], existentially: for every N, some fixed point set chosen without looking at the weights has ensemble
   risk at most radius-exact MC's (average the risk over iid point sets; some realisation is below the mean); the
   argument gives no efficient construction or short description. PROVEN UNDER exact-transform primitives [prior:
   unrestricted 3.1; A]: the polar factor Q of W_1 = H Q is Haar and independent of the rest of the net, so seeds read
   off the weights simulate any randomised estimator at the same average risk (finite-precision weights do not supply
   unlimited bits). Runtime randomness alone does not separate estimators under the ensemble-risk criterion; support
   size, construction cost and error estimation remain distinct questions. What an estimator's own randomness buys is
   conditional unbiasedness on the given net and a replicate variance (an error bar), not accuracy [A].
3. **"Noise sensitivity forces long seeds": wrong direction.** On average, seeds of MC length (log2 of MC's sample
   count) always suffice. A heavy high-degree spectrum only forbids seeds much SHORTER than MC's: no weight-independent
   point set beats MC by more than the inverse of the even spectral share it cannot cancel (section 2). Long seeds
   (superpolynomial in 1/eps or exponential in n) arise only for worst-case classes, which is what the PRG literature
   is about and ARC's random nets are not.
4. **"The obstacle is boundary mass": the boundary mass is real but it is not the obstacle.** The gate partition is
   maximally noise-sensitive: expected perimeter (1 - 2^-n)^{l-1}/sqrt(2 pi) per neuron, PROVEN [A]; measured total
   0.77-1.07 x nL/sqrt(2 pi) at 256 x 32; some gate changes with probability 0.93-0.98 already at correlation
   1 - 1e-6. The output is Lipschitz and decorrelates linearly in the noise, with slope = mean Hermite degree ~ n/V
   (~19 at 256 x 32). Every non-linear harmonic of the output is made by kinks, but a point set acts on degrees <= 5,
   and how kink strength lands there is a large-scale property of many weak kinks, not a count of boundaries.
5. **Black-box vs white-box (the coordinator's point (b)): right, and now exact.** A point set chosen without the
   weights has ensemble risk exactly sum_d a_d B_d(Q): the net's spherical-harmonic energies a_d against the point
   set's aliasing B_d (FORMAL T1-T2; Wozniakowski's average-case identity specialised). White-box rules escape this
   identity; section 4.6 measures what that buys.
6. **Caratheodory (point (c)): right.** For one net, n + 1 weighted points reproduce all n output means; finding them is
   the integration problem itself. With randomness free, the trade-off is compute vs precision, and "structure" means
   whatever lowers the error per forward pass.

So the question that survives is your 13:05Z one: starting from sampling, how much structure can a sample set carry,
and does any structure change the error-vs-compute exponent (the asymptote) or only the constant?

## 2. Theory (FORMAL.md; [C] this lane, [A] gpt-6-astra)

- **Exact pricing of point sets** (PROVEN [C][A]). For a rule Q = sum_i w_i delta_{u_i} on the sphere (radius exact),
  ensemble risk = sum_{d>=1} a_d B_d(Q), with a_d the expected energy of the degree-d spherical-harmonic part of m_n
  g_W and B_d(Q) = sum_ij w_i w_j P_d(u_i . u_j) >= 0 (P_d the Gegenbauer polynomial, P_d(1) = 1); for one net and a
  Haar-rotated rule, the same with that net's own energies E_d(W). Consequences: antipodal pairs cancel every odd
  degree and pay every even degree twice (gain A / (2 A_even)); a t-design cancels degrees 1..t; R independently
  rotated copies of a fixed rule have risk exactly risk/R, so **every union of independently rotated copies of a fixed
  design has error-vs-compute exponent exactly 1: such designs buy constants, never rates.** The same identity makes
  the rotation variances of designs with known aliasing a truth-free measurement of the spectrum (used in section
  4.3).
- **The black-box ceiling** (PROVEN [A] in power form; PROVEN [C] in harmonic form, T4). For positive weights summing
  to one, risk >= (m_n^2/(2N)) sum_{s: tau_2s <= 1/(2N)} b_2s, where k(t) = E_W <g(u), g(v)> = sum_r b_r t^r with
  b_r >= 0 and tau_2s = E (U . V)^{2s}; in harmonic form, risk >= (1/(2N)) sum over even d with |min P_d| <= 1/(2N) of
  the degree-d energy. At n = 256 every even degree from 6 on is paid in full for N up to 8.6e4 (harmonic form; 5.7e5
  in power form), which covers ARC's budget and the Kerdock set. So no rule chosen without the weights (fixed designs,
  Haar-randomised designs, RQMC with weight-independent scrambling) can beat radius-exact MC by more than
  2 A / (energy at even degrees >= 6) at ARC's budget (the numbers in section 4.3 use this harmonic form); the
  cancellable degree grows only like 2 log N / log n while log of the degree is small against log n.
- **The black-box exponent** (PROVEN [A], round 2). At fixed width the even power tail of a ReLU net decays like
  D^{-3/2} (kink tail) and tau_d like d^{-(n-1)/2}, so every positive weight-independent rule has risk
  >= c N^{-1-3/(n-1)} for large N, while stratifying the sphere into N equal cells attains N^{-1-2/(n-1)}. The exponent
  of the best black-box rule therefore lies in [1 + 2/(n-1), 1 + 3/(n-1)] = [1.008, 1.012] at n = 256: MC's 1, in any
  measurable sense. In the joint wide/deep limit at polynomial budgets the optimal black-box risk is Theta(1/N): no
  weight-independent point set has a gain that grows with n. (These are asymptotic statements; a slope fitted over
  2^8..2^18 is not constrained by them, which is why section 4.1 measures it.)
- **Design sizes** (PROVEN, Delsarte-Goethals-Seidel via Bannai-Bannai; the weighted converse [A]). A spherical
  5-design in R^256 has >= 65,792 points; the antipodal Kerdock set (129 mutually unbiased bases, 66,048 points) is
  within 0.4% of that; a 7-design needs >= 5.66e6 points, 87x ARC's budget. Exactness through degree 2s needs
  >= C(n+s-1, s) + C(n+s-2, s-1) points.
- **Noise sensitivity bounds the tails** (PROVEN [C]; Hermite version and a two-scale bound [A]; the lower bound can
  be zero, so it bounds rather than pins). For spherical pairs at
  inner product rho, NS(rho) = sum_d e_d (1 - P_d(rho)) exactly (e_d the energy shares), so the share tau_t a t-design
  cannot cancel satisfies (NS - max_{d<=t} a_d)/(max_{d>t} a_d - max_{d<=t} a_d) <= tau_t <= NS / min_{d>t} a_d,
  a_d = 1 - P_d(rho). Gaussian (Ornstein-Uhlenbeck) noise is a Hermite statement and bounds spherical tails one way only
  [A]. This is the precise, average-case form of "noise sensitivity governs the seed", with the direction of point 3.
- **Gate perimeter vs output roughness** (PROVEN [A]). Gate-set perimeter (1 - 2^-n)^{l-1}/sqrt(2 pi) per neuron in
  expectation (dead-region interfaces included), ~ nL/sqrt(2 pi) in total; P(gate history changes) ~ (sqrt 2 n /
  pi)(sum_j beta_n^j) sqrt(delta), beta_n = 1 - (n + 1) 2^-n; E sum_k k H_k = n exactly, so the annealed output
  decorrelation is (n / E V) delta (1 + o(1)); for one net V NS(delta) = D delta - (2 J_kink/(3 sqrt pi)) delta^{3/2}
  + o(delta^{3/2}), and the Hermite tail beyond t is ~ (J_kink / (3 pi)) t^{-3/2}, J_kink the surface integral of
  squared Jacobian jumps over the kink set. Gate counts do not price output effects (an unused gate has no jump).
- **Bounded independence** (PROVEN [A]). Exact k-wise independent Gaussian coordinates match every polynomial of degree
  <= k; but a 127-wise independent law on R^256 and a width-256 positively homogeneous ReLU net exist whose mean the
  law gets wrong by a factor 2. Coordinate independence is not sample independence, and for homogeneous f the natural
  matching object is the radius-weighted direction measure, not Gaussian moments.
- **White-box frames** (PROVEN identity [C][A]; effect HEURISTIC). With W_1 = H Q (polar), Q is Haar and independent of
  (H, W_2, ..., W_L). A weight-independent rule sees a Haar-rotated net (the identity above); a rule placed in the polar
  frame sees F(H s), whose ensemble kernel is only permutation invariant. In that frame each first-layer pre-activation
  has own-coordinate share (8/(3 pi))^2 ~ 0.72 (quarter-circle law), so first-layer gates are nearly coordinate-wise
  and Sobol's best-stratified projections stratify them. Unbiasedness needs every point to be exactly Gaussian
  conditional on W (jointly, not coordinate by coordinate): scrambling or a cube shift does that [A]; signed
  permutations of a fixed design do not, and Kerdock's flat +-1/16 vectors make the aligned design biased (measured;
  for f = relu the aligned rule estimates 1/2 against the true 1/sqrt(2 pi) [A]).
- **Multilevel** (PROVEN UNDER a residual and cost hypothesis [A, prior: unrestricted]). Gain Omega(L / log^2 L) over
  radius-exact MC when E|F - G_r|^2 <= C (1/r + n/L^2) with prefix cost r/L. At L = n/8 the prefix-norm control's
  residual at the best prefix depth stays ~0.9 of the variance through n = 256, so the gain is ~1 there (section 4.6).
- **Where the finite-width variance comes from** (PROVEN [A], round 2; the main new theorem of this lane; I checked
  its structure and its numerical agreement with the measurements, not every step of the limit argument). For two
  inputs, 1 - k_j(t) = [1 - E r_j r'_j] + [direction mismatch]: a norm term (the two inputs' activation norms differ)
  plus a direction term bounded by 9 pi^2/(2 j^2). Each layer adds a norm-mismatch term of order 1/n; with width taken to
  infinity first these vanish, which is why the mean-field picture (degrees 1-2 carry O(1/L)) is wrong at finite width.
  Along L = alpha n with L/sqrt(n) -> infinity, E V -> S(0) + 1/2 = 11.15 (S an explicit convergent series over layers),
  the fixed-degree shares converge (degree 1: 0.29, degree 2: 0.21, odd 0.57, even from degree 4: 0.22, from degree 6:
  0.17), and the design gains over MC converge to constants: simplex 2.09, cross-polytope 2.36, Kerdock 3.15. Measured
  at 256 x 32 (medians of 5 nets): 1.93, 2.39, 3.04, with shares 0.47 (degrees <= 2), 0.56 (odd), 0.22, 0.17. So the
  spectrum point sets fight is the accumulated activation-norm fluctuation of the first ~10-30 layers, and its
  black-box value is a constant ~2-3, not a quantity that grows or vanishes with width. The degree-1 part is exactly the
  per-net mean Jacobian: E_1(W) = (m_n^2/n) ||E_X J_f(X)||_F^2 [A].

## 3. Literature: what is known and what it implies here (notes/LITERATURE.md; quotes and URLs in LITERATURE_raw.md)

- **Worst-case PRGs reach ReLU-sized networks only at depth 1.** Halfspaces: Meka-Zuckerman O(log n + log^2(1/eps)),
  Gopalan-Kane-
  Meka O(log(n/eps)(log log(n/eps))^2). Gaussian degree-d PTFs: Kane (CCC'14) O_{c,d}(log n eps^-c); O'Donnell-Servedio-
  Tan (STOC'20) d^{O(log d)} log n; Kelley-Meka (CCC'22) d^{O(1)} log(n/eps) log(1/eps)/eps^2. Intersections and
  functions of k halfspaces: Harsha-Klivans-Meka O(log n poly(log k, 1/eps)); O'Donnell-Servedio-Tan (polytopes)
  poly(log m, 1/eps) log n; Chattopadhyay-De-Servedio O(log n + poly(k, 1/eps)). Threshold circuits: Hatami-Hoza-Tal-
  Tell, depth d with n^{1+delta} wires, delta = 2^{-O(d)}, seed O(n^{1-delta}). A width-n ReLU layer has n^2 wires;
  fooling all constant-depth ReLU nets would separate TC0 from NC1 (Chen-Li-Li-Meka 2022). No PRG, bounded-independence
  or deterministic-counting result for ReLU nets was found (named searches). For a ReLU net these results reach only the
  first layer: one first-layer cell is an intersection of n origin halfspaces (seed O(log n + poly(log n, 1/eps)),
  Chattopadhyay-De-Servedio), the whole first-layer gate pattern a function of n of them (poly(n, 1/eps) bits). A valid
  worst-case PRG for the full net would also give an average-case estimator, but none was found in the searches, and
  the barrier above says why one would be hard to establish [A].
- **Bounded independence**: O~(1/eps^2)-wise for halfspaces (Diakonikolas-Gopalan-Jaiswal-Servedio-Viola), poly(1/eps)
  for degree 2 (Diakonikolas-Kane-Nelson), O_d(eps^{-4d 7^d}) for Gaussian degree d (Kane). Uniform over all k-wise
  laws, with supports of n^{Theta(k)} points: as estimators of one function far costlier than MC.
- **Noise sensitivity and surface area**: Kane (PTFs: GSA <= d/sqrt(2 pi)); Klivans-O'Donnell-Servedio (NS_delta <=
  sqrt(pi delta) Gamma); Nazarov (k halfspaces: Gamma <= sqrt(2 ln k) + 2); Hanin-Rolnick (region boundaries ~ number of
  neurons, with biases); Jonasson-Steif-Zetterqvist (bias-free deep SIGN nets are noise sensitive); De Palma-Kiani-
  Lloyd (random deep ReLU classifiers are biased to simple functions). Our nets: the gate pattern is on the sensitive
  side, the ReLU output on the stable side.
- **Few random bits**: pairwise-independent sampling (Chor-Goldreich: 2n or 3n - 1 coins), expander walks (Gillman),
  extractor samplers (Zuckerman); the Canetti-Even-Goldreich lower bound (~input length) is worst-case over all
  functions with a confidence parameter, not ARC's average-case MSE.
- **QMC**: scrambled (0,m,s)-nets never exceed e x MC variance and are o(1/N) for every square-integrable f (Owen);
  for certain discontinuous f in d dimensions the provable RMSE-rate gain is ~1/(4d - 2) (He-Wang), i.e. ~1/(2d - 1)
  in this report's MSE convention, ~0.002 at d = 256. Preintegration smooths kinks under monotonicity
  (Griewank-Kuo-Leovey-Sloan; Gilbert-Kuo-Sloan): the sibling preint lane's subject. Average-case integration
  (Wozniakowski; Novak-Wozniakowski Vol. II, Cor. 13.1): the average error of a linear rule is the worst-case error on
  the covariance kernel's RKHS; ARC's problem, restricted to point sets, is this problem for the kernel k(t) above.

## 4. Measurements (registered unless marked descriptive)

### 4.1 Exponents first (Amendment 1)
Kelvin's question is the asymptote, so this comes first. Fitted exponent p (risk ~ N^-p; plain MC has p = 1 exactly),
per net by weighted least squares of log risk on log N (randomised series: replicate variances at N = 2^8, 2^10,
2^12, 2^14, 2^16, 2^18 with 32, 16, 16, 16, 16, 8 replicates; deterministic series: error against the truth at
N = 2^8..2^16), median over the five 256 x 32 nets with a 95% bootstrap interval of the median (scratch/rate_fit.py,
frozen before the E5 runs; results/rates.md):

| series | nets | median p | 95% CI (bootstrap over nets) | p per net |
|---|---|---|---|---|
| rqmc_std | 5 | 1.045 | [1.031, 1.076] | [1.053, 1.031, 1.032, 1.076, 1.045] |
| rqmc_polar | 5 | 1.058 | [1.008, 1.075] | [1.058, 1.023, 1.064, 1.075, 1.008] |
| sobol_rb | 5 | 1.085 | [0.835, 1.139] | [1.072, 1.137, 1.139, 1.085, 0.835] |
| sobol_rb_polar | 5 | 1.072 | [1.014, 1.189] | [1.189, 1.152, 1.053, 1.072, 1.014] |
| sobol_plain | 5 | 1.120 | [1.024, 1.475] | [1.475, 1.284, 1.115, 1.12, 1.024] |
| lattice_rb | 5 | 1.130 | [1.009, 1.223] | [1.223, 1.009, 1.145, 1.13, 1.099] |
| lattice_baker_rb | 5 | 1.063 | [0.994, 1.181] | [0.994, 1.045, 1.063, 1.181, 1.093] |
| polar_minus_std | 5 | -0.002 | [-0.036, 0.032] | [0.005, -0.008, 0.032, -0.002, -0.036] |

Gains over MC at the measured sizes (medians over nets, results/tradeoff.json): scrambled Sobol 1.39, 1.84, 1.94, 1.87,
2.13, 2.44 at 2^8, 2^10, 2^12, 2^14, 2^16, 2^18; the same in the polar frame 1.61, 2.11, 2.23, 2.50, 2.22, 2.05.

- **RQMC shows a small positive finite-range excess over MC; no structured sampler was shown to beat it.** Scrambled
  Sobol: p = 1.045 [1.031, 1.076] over N = 2^8..2^18 (PA1 holds; the interval excludes 1); in the white-box polar
  frame p = 1.058 [1.008, 1.075] (PA2 holds); polar minus standard -0.002 [-0.036, 0.032]: no detected difference (not
  an equivalence test). The zero-bit point sets fit 1.06-1.13 (centred Sobol 1.085, in the polar frame 1.072, Korobov
  lattice 1.130, baker 1.063), with intervals that contain RQMC's: their single-realisation errors swing by up to a
  few times between neighbouring N, so their slopes are poorly determined, and their higher point estimates do not
  establish an ordering. The intervals are bootstrap intervals of the median over five per-net fits. No series exceeds
  1.5 (PA3 holds). Unions of rotated designs (Kerdock blocks, simplices, cross-polytopes) have p = 1 exactly
  (theorem), plain and pairwise MC 1.
- **The small excess over 1 is what the black-box bound itself allows over this range.** Between 2^8 and 2^18 the
  cancellable degree of the bound steps up twice (even degrees from 4 to 6 at ~5.5e3 points, to 8 at ~8.6e4), so its
  constant drops by at least the factor A_{even >= 6}/A_{even >= 4} ~ 0.77 measured here: an effective slope of at
  least ~1.04 over these ten doublings, the size of RQMC's fitted excess. The asymptotic black-box exponent is
  1.008-1.012 (section 2); He-Wang's provable RQMC excess, in MSE terms ~1/(2d - 1) ~ 0.002 at d = 256, concerns
  certain discontinuous integrands.
- **Along the ladder** (descriptive; E1 sizes 2^10-2^14, 16 replicates, results/ladder_rates.json): scrambled Sobol
  p = 1.05, 1.08, 1.02, 1.02 at n = 32, 64, 128, 256 and the polar frame 1.06, 1.00, 1.03, 1.04; the only interval
  clearly above 1.1 is the SVD frame at n = 32 (1.16 [1.13, 1.20]), the shallow net where the first layer's
  structure still dominates.

So the answer to "does any design beat RQMC's exponent at 256 x 32": none was shown to; the fitted exponents are
1.0-1.13 with overlapping intervals, RQMC's own excess over MC is ~0.05 over this range, and the white-box frame shows
no detected change in exponent and none in constant at ARC's budget. That no weight-dependent structure of this kind
changes the asymptotic exponent is a CONJECTURE (astra's, round 2).

### 4.2 Constants at ARC's budget, 2^16 passes (E2)
**Table C. ARC's budget, 256 x 32 (E2): N = 2^16 points (Kerdock 66,048), radius exact, nMSE against the truth; gain = 1 / (nMSE x N), plain MC = 1**

| sampler | random bits | net 200 | 201 | 202 | 203 | 204 | median gain |
|---|---|---|---|---|---|---|---|
| pairwise-independent MC | 16384 | 1.70 | 1.53 | 1.61 | 0.40 | 0.95 | 1.53 |
| scrambled Sobol (black-box) | 135168 | 2.77 | 1.97 | 2.13 | 2.15 | 1.95 | 2.13 |
| scrambled Sobol, polar frame of W_1 (white-box) | 135168 | 2.64 | 1.83 | 2.22 | 2.47 | 1.87 | 2.22 |
| Kerdock 5-design, Haar rotation | 1044480 | 3.93 | 3.15 | 2.79 | 2.89 | 2.20 | 2.89 |
| centred Sobol, fixed | 0 | 1.84 | 1.78 | 5.55 | 3.06 | 2.07 | 2.07 |
| centred Sobol, polar frame, fixed | 0 | 0.83 | 3.27 | 2.41 | 4.35 | 2.68 | 2.68 |

Bit counts are idealised accounting at 32 bits per random number; the implementation differs (scipy's scrambling
uses 30-bit digits by default, the pairwise sampler draws 53-bit floats), and exact unbiasedness and pairwise
independence hold for the continuous constructions, not literally for their finite grids [A]. Setup costs (a Haar
rotation or a polar decomposition, O(n^3), a few passes) are not charged.

- **The antipodal Kerdock 5-design is the best sampler at ARC's budget: 2.89x over MC (2.20-3.93; PC2, PC3 hold).**
  None of the randomised structured samplers reaches 4x on any net; the estimated black-box ceiling at this size is
  ~10-19x (section 4.3) and astra's ensemble limit for Kerdock 3.15.
- Scrambled Sobol gains 2.13x (1.95-2.77); in the polar frame 2.22x (1.83-2.64): at 2^16 the frame's advantage is gone
  (ratios 0.93-1.15; PC1 FAILS, it needed >= 1.2x on 4 of 5 nets).
- Zero random bits: centred Sobol 2.07x (1.78-5.55), the same in the polar frame 2.68x (0.83-4.35; PC5 holds); the
  fixed sets exceed 4x on single nets (5.5x on net 202, 4.4x on 203) and fall below 1x on another. On average over nets
  a fixed point set has the risk of its Haar-rotated version (T2; this does not cover Sobol scrambling or a
  weight-dependent frame), but on a given net it is a lottery ticket with no error bar.
- Pairwise-independent MC with 16,384 random bits: median 1.53x (0.40-1.70); its mean risk is exactly MC's (PC4 holds,
  section 4.7 for the skew).

### 4.3 The spectrum, and the black-box ceiling (E1)
Method (FORMAL T1): Haar-randomised designs with known aliasing, their rotation variance measured on each net (64
rotations; 12 for Kerdock): MC (all degrees), antipodal pairs (even degrees only), orthonormal basis (kills degree 2),
regular simplex (2-design), cross-polytope (antipodal 3-design), Kerdock (antipodal 5-design). Differences give the
shares of the radius-exact variance A by degree, with no truth and no high-variance U-statistics. Shares are noisy
per net (the degree-2 share of net 202 comes out -0.02 +- ~0.05); the registered statements are medians.

**Table A. 256 x 32, per net: harmonic shares of the radius-exact variance and Haar-randomised design gains over plain MC (E1)**

| net | V | V_rb/V | odd | even | deg 1 | deg 2 | even >= 4 | even >= 6 | ONB | simplex | cross | Kerdock | T4 ceiling (vs MC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 21.2 | 0.940 | 0.585 | 0.417 | 0.44 | 0.18 | 0.174 | 0.113 | 1.30 | 2.79 | 3.06 | 4.69 | 18.8 |
| 201 | 30.7 | 0.972 | 0.556 | 0.443 | 0.32 | 0.15 | 0.241 | 0.169 | 1.21 | 1.93 | 2.13 | 3.04 | 12.1 |
| 202 | 11.9 | 0.960 | 0.572 | 0.432 | 0.49 | -0.02 | 0.218 | 0.161 | 1.02 | 1.97 | 2.39 | 3.24 | 13.0 |
| 203 | 5.4 | 0.981 | 0.563 | 0.432 | 0.25 | 0.18 | 0.201 | 0.186 | 1.25 | 1.81 | 2.54 | 2.74 | 11.0 |
| 204 | 4.8 | 0.981 | 0.531 | 0.459 | 0.05 | 0.21 | 0.308 | 0.212 | 1.29 | 1.38 | 1.66 | 2.40 | 9.6 |

**Table B. Along L = n/8: medians over nets [min, max] (8 nets at n = 32, 64, 128; 5 at 256). Gains are over plain MC at equal points unless stated (E1)**

| quantity | n = 32, L = 4 | n = 64, L = 8 | n = 128, L = 16 | n = 256, L = 32 |
|---|---|---|---|---|
| V_rb / V | 0.96 [0.94, 0.97] | 0.96 [0.95, 0.97] | 0.96 [0.95, 0.97] | 0.97 [0.94, 0.98] |
| odd share | 0.63 [0.61, 0.64] | 0.59 [0.57, 0.64] | 0.58 [0.54, 0.59] | 0.56 [0.53, 0.58] |
| degree-1 share | 0.47 [0.40, 0.58] | 0.38 [0.31, 0.42] | 0.29 [0.13, 0.37] | 0.32 [0.05, 0.49] |
| degree-2 share | 0.28 [0.22, 0.36] | 0.27 [0.23, 0.34] | 0.24 [0.08, 0.31] | 0.18 [-0.02, 0.21] |
| even >= 4 share | 0.09 [0.07, 0.11] | 0.15 [0.11, 0.18] | 0.20 [0.16, 0.31] | 0.22 [0.17, 0.31] |
| gain: orthonormal basis | 1.46 [1.33, 1.62] | 1.43 [1.35, 1.58] | 1.37 [1.12, 1.52] | 1.25 [1.02, 1.30] |
| gain: simplex (2-design) | 4.49 [3.58, 5.61] | 2.82 [2.54, 4.05] | 2.10 [1.52, 2.86] | 1.93 [1.38, 2.79] |
| gain: cross-polytope (antipodal 3-design) | 5.52 [4.74, 7.27] | 3.50 [2.91, 4.98] | 2.58 [1.67, 3.20] | 2.39 [1.66, 3.06] |
| gain: Kerdock (antipodal 5-design) | - | - | - | 3.04 [2.40, 4.69] |
| gain: scrambled Sobol, 2^14 | 5.29 [4.42, 6.89] | 3.26 [2.43, 4.90] | 2.37 [1.61, 3.04] | 1.87 [1.42, 2.42] |
| gain: scrambled Sobol, polar frame, 2^14 | 6.53 [5.93, 8.57] | 3.43 [2.27, 5.69] | 2.80 [1.80, 4.35] | 2.50 [1.54, 3.32] |
| gain: scrambled Sobol, SVD frame, 2^14 | 9.23 [7.30, 11.88] | 3.70 [3.03, 5.35] | 2.34 [1.84, 3.41] | 1.84 [1.58, 3.22] |
| gain: antipodal Sobol, 2^14 | 4.84 [3.84, 6.89] | 3.10 [2.29, 4.27] | 2.32 [1.67, 3.36] | 2.06 [1.78, 3.80] |
| multilevel prefix control, gain vs RB-MC | 0.83 [0.79, 0.86] | 0.86 [0.76, 0.93] | 0.94 [0.90, 0.97] | 0.99 [0.96, 1.03] |

What it shows (MEASURED; PS1-PS4, PB1-PB2 hold):
- **At 256 x 32 the odd harmonics carry 0.56 of A** (0.53-0.59), degrees 1-2 about 0.47 (0.26-0.62), the even part
  from degree 4 on 0.22 and from degree 6 on 0.17 (0.11-0.21). Antipodal pairs alone therefore gain only 1.09-1.20 over
  radius-exact MC (they kill the odd 0.56 but pay the even 0.44 twice).
- **The antipodal Kerdock 5-design is the best point set at ARC's budget: 3.0x over MC (2.4-4.7; PB1 holds)**.
  Astra's round-1 prediction, made before any data (not one of my registered predictions), was N E err^2 / V in
  [0.33, 0.50], i.e. 2-3x: inside on 3 nets, better on 2. It cancels every degree <= 5 and pays the even tail from
  degree 6 twice.
- **An estimated black-box ceiling at <= 8.6e4 points: 2 A / A_{even >= 6} = 9.6-18.8x over MC (median 12.1).** The
  bound is PROVEN; the numbers are a plug-in: the even >= 6 energy is estimated as half the Kerdock rotation variance,
  so each ceiling is 4x that net's measured Kerdock gain by construction, with the 12-rotation estimates' ~8-18%
  relative error and a ~3% aliasing correction (N B_6 / 2 = 1.031 [A]) ignored; it is not a certified bound. No
  weight-independent positive point set on the sphere of that size (a fixed design, a Haar-randomised one,
  radius-exact scrambled Sobol, a radius-normalised k-wise family) can do better on these nets on average over
  rotations (rules with varying radii have a radius-weighted defect and are not covered as stated [A]). The measured
  best is 4x below the ceiling: a factor 2 is the antipodal doubling, a factor 2 the bound's own slack; a
  non-antipodal 5-design of affordable size (none known) would sit between.
- **Along L = n/8 every black-box constant falls with width**: the simplex 4.5x -> 1.9x, the cross-polytope 5.5x ->
  2.4x, scrambled Sobol at 2^14 5.3x -> 1.9x from n = 32 to 256, while the even share from degree 4 grows 0.09 -> 0.22:
  the spectrum moves up in degree as depth grows with width. In the limit they do not fall to 1: astra's round-2
  theorem (section 2) gives constant limits 2.09 (simplex), 2.36 (cross-polytope), 3.15 (Kerdock) as n -> infinity with
  L/sqrt(n) -> infinity, and 256 x 32 sits at 1.93, 2.39, 3.04.
- **The registered test of astra's finite-width prediction at 512 x 64 failed on all four bands, narrowly (Amendment 2,
  three fresh nets, 16 rotations per design).** Measured: degree <= 2 share 0.41 (band [0.45, 0.60]), odd share 0.54
  ([0.55, 0.59]), simplex gain 1.71 ([1.8, 2.6]; the cross-polytope's 2.11 was inside [2.0, 2.8]), n / mean V = 48.4
  ([35, 45]). The variance is where the theorem puts it (mean V = 10.6 against the limit 11.15), but the low-degree share
  and the simplex gain moved further from their limits between 256 x 32 and 512 x 64 (0.47 -> 0.41, 1.93 -> 1.71),
  so the approach to the limit is not visible at these rungs (L/sqrt n = 2.0 and 2.8); the 16-rotation share estimates
  carry roughly +-0.08. The limit theorem stands as proven; how fast finite nets approach it is open. Scrambled Sobol at
  2^12 gains 2.36x (median) at 512 x 64, and the polar frame 1.68x (worse); the multilevel prefix control 1.00-1.07.

### 4.4 Gate perimeter and noise sensitivity (E3)
Method: 4,096 Gaussian pairs at correlation 1 - delta, delta = 1e-6 ... 1, per layer: gate flips, whether any gate of
layers <= l changed, and the state's normalised decorrelation; plus spherical pairs at exact inner product rho for the
output (FORMAL T6). Perimeter = flips / (2 sqrt(delta/pi)), pooled over delta = 1e-6, 1e-5, 1e-4.

**Table D. Gate perimeter and noise sensitivity (E3): medians over nets**

| n x L | layer-1 GSA / (n/sqrt(2 pi)) | total / (nL/sqrt(2 pi)) [min, max] | P(some gate changes), delta = 1e-6 | output NS slope (mean Hermite degree) | n / V |
|---|---|---|---|---|---|
| 32 x 4 | 1.008 | 1.010 [0.92, 1.11] | 0.054 | 2.9 | 3.0 |
| 64 x 8 | 0.988 | 0.943 [0.89, 1.02] | 0.191 | 5.1 | 4.7 |
| 128 x 16 | 0.992 | 0.910 [0.80, 1.20] | 0.566 | 8.6 | 10.8 |
| 256 x 32 | 1.008 | 1.006 [0.77, 1.07] | 0.964 | 18.6 | 21.6 |

- **The perimeter law holds** (PN1, PN3 hold): layer 1 at 0.94-1.07 x n/sqrt(2 pi) on all 29 nets (a theorem, so this
  validates the estimator), the whole partition at 0.77-1.07 x nL/sqrt(2 pi) at 256 x 32, matching astra's exact
  annealed (1 - 2^-n)^{l-1}/sqrt(2 pi) per neuron. Per layer, per net, it ranges 0.33-1.31 x n/sqrt(2 pi): PN2 (every
  layer within [0.5, 1.3]) FAILS, on net 200's last layer (0.33, synchronised gates) and net 202's layer 19 (1.31).
- **The gate pattern is maximally noise-sensitive** (PN5 holds): at delta = 1e-6 some gate changes with probability
  0.93-0.98 at 256 x 32 (0.05, 0.19, 0.57 at n = 32, 64, 128: it grows like the total perimeter, ~n^2/8 along L = n/8).
- **The output is not**: its decorrelation is linear in delta with slope (mean Hermite degree) 18.6 at 256 x 32
  (median; 2.9, 5.1, 8.6 at n = 32, 64, 128), ~0.86 x n/V per net. PN4 (slope within [0.4, 1.5] x n/V on every net)
  FAILS on one net (201: 2.09): per-net Jacobian energy varies ~5x around its annealed value n.
- **Noise sensitivity and the design spectrum agree** (PN6 holds): on every 256 net the design-measured tail beyond
  degree 2 (0.38-0.74) exceeds the lower bound that FORMAL T6 derives from the spherical NS curve (0.19-0.44). The T6
  bounds are valid but loose for low degrees; the design variances measure the shares directly and better.
- The layer-l state's slope grows with depth at every width (1.5 at layer 1, the ReLU's own mean degree 1.47, to 18.5 at
  layer 29 of 256 x 32; descriptive). My first reading, "slope ~ 1 + l/2", is not the law: the annealed slope is exactly
  n / E V [A], and E V -> 11.15 along L = alpha n (section 2), so the asymptotic slope is ~n/11.15, ~0.72 L at L = n/8.
  Measured V is already O(1) along the ladder: mean 10.6, 15.5, 14.6, 14.8 and median 10.7, 13.5, 11.9, 11.9 at
  n = 32, 64, 128, 256 (per net 4.8-30.7 at 256 x 32), so n / mean V = 3.0, 4.1, 8.8, 17.3 grows like n, as the
  theorem says.

### 4.5 Zero-random-bit point sets and bounded independence (E4)
Fixed point sets use no random bits; their error on a given net is one number, scored against the truth. Bounded
independence: the full family of degree-(k-1) polynomials over GF(q) (q the least prime > n), each coordinate decoded to
one of q symmetric levels whose even moments match N(0, 1) to degree min(k, 6), so the design matches every Gaussian
moment of degree <= k exactly (tests/test_rnd.py); N = q^k points.

**Table E. Zero-random-bit point sets against the truth (E4): median nMSE x N over nets (plain MC = 1); k-wise rows give N = q^k**

| n x L | centred Sobol 2^10 / 2^14 / 2^16 | Sobol, polar frame 2^10 / 2^14 / 2^16 | Korobov lattice 2^14 | k-wise k = 2 | k = 3 | k = 4 | Kerdock fixed / polar-aligned |
|---|---|---|---|---|---|---|---|
| 32 x 4 | 0.24 (8/8) / 0.17 (8/8) / 0.11 (8/8) | 0.18 (8/8) / 0.14 (8/8) / 0.08 (8/8) | 0.36 (8/8) | 0.31 (8/8) | 0.41 (8/8) | 0.51 (2/8) | - / - |
| 64 x 8 | 0.33 (8/8) / 0.27 (8/8) / 0.24 (8/8) | 0.22 (8/8) / 0.37 (8/8) / 0.19 (8/8) | 0.56 (8/8) | 0.62 (8/8) | 0.99 (8/8) | 4.90 (1/8) | - / - |
| 128 x 16 | 0.32 (8/8) / 0.51 (8/8) / 0.33 (8/8) | 0.68 (8/8) / 0.42 (8/8) / 0.38 (8/8) | 0.88 (8/8) | 1.86 (8/8) | 1.49 (8/8) | - | - / - |
| 256 x 32 | 0.45 (5/5) / 0.34 (5/5) / 0.48 (5/5) | 0.64 (5/5) / 0.33 (5/5) / 0.37 (5/5) | 0.53 (5/5) | 1.90 (5/5) | - | - | 0.56 (5/5) / 95.94 (5/5) |

(Each cell: median over the nets resolved above 3x the truth floor, with (resolved / run) nets; - = not run. k-wise
cells with few resolved nets are not evidence about k.)

- **Zero-bit Sobol is as good as scrambled Sobol on average**: centred unscrambled Sobol, radius exact, nMSE x N ~0.3-0.5
  at 256 x 32 (a 2-3x gain), with large per-net and per-budget swings (0.18-1.07 on dev nets; the same erratic
  behaviour the nd-exact lane saw). Its fitted exponent over N = 2^8..2^16 has median 1.09 (0.84-1.14 over the five
  nets) (PQ1 holds).
- **Korobov lattices** (centred or baker-transformed, generator searched by the weighted P_2 criterion) are MC-like to
  2x at 256 x 32 (PQ4 holds): lattice rules need a periodic, smooth integrand, and an inverse-normal-mapped kinked net is
  neither.
- **The tested bounded-independence designs do not fool these nets beyond MC, and get worse with width.** (These are
  particular finite-level polynomial-hash constructions; their radius-normalised variants no longer keep coordinate
  independence; high-k cells are mostly unresolved, Table E counts.) At n = 32 the k-wise designs
  sit at nMSE x N 0.1-0.8 (k = 2, 3, 4; resolved points), at n = 64 0.3-2.4, at n = 128 0.8-4.7, and at 256 x 32 the
  pairwise design (66,049 points) has nMSE x N 0.55-4.0 (median 1.9: worse than MC). PQ3 (every resolved point in
  [0.05, 5]) holds: 114 resolved (net, k, variant) points, plain and radius-normalised counted separately, nMSE x N
  from 0.13 to 4.9. Matching moments to degree k does not
  cancel what point sets need to cancel here (harmonic
  degrees of the homogeneous output), and the rigid algebraic structure aliases higher degrees systematically. With
  astra's counterexample (a 127-wise independent law misreading a width-256 ReLU net by 2x), the answer to "how fast do
  k-wise designs fool the nets as k grows", for these constructions at the resolved budgets: they do not start to;
  their error is MC-like at their own size N = q^k,
  so the "degree needed for eps" is k ~ 2 log(1/eps)/log n, MC's sample count in disguise.
- **The unrotated Kerdock design** (zero bits) has nMSE x N 0.11-0.80 at 256 x 32 (median 0.56); on average over nets a
  fixed design has its rotated version's risk (FORMAL T2), per net it can be lucky or unlucky.

### 4.6 White-box structure: frames, aligned designs, controls, multilevel, exact tails
The coordinator's question: which white-box structure (gate patterns, shared trajectories, exact tails) lets a sample
set use fewer random bits and less compute for the same precision?
- **Gate patterns, via the first layer's frame.** Scrambled Sobol placed in the polar frame of W_1 (first-layer gates
  nearly coordinate-wise; unbiased, same random bits as standard RQMC, O(n^3) setup ~ 4 passes) beat standard RQMC at
  2^14 by a median 1.31x at 256 x 32 (1.37, 1.15, 1.76, 1.31, 0.82; PW1 holds) and by 1.26, 0.99, 1.17 at n = 32, 64,
  128 (PW2 holds). At 2^16 the advantage is gone: polar/standard ratios 0.95, 0.93, 1.04, 1.15, 0.96 on nets 200-204
  (E2; PC1 FAILS). With 16 replicates per size each gain carries roughly +-35% sampling error, and at 512 x 64
  (Amendment 2 nets, 2^12, 8 replicates) the polar frame was worse than the standard one (1.68x vs 2.36x): the frame
  moves the constant by a few tens of percent either way, and section 4.1 detects no change in the exponent. The SVD
  frame (directions ordered by first-layer gain) is large at n = 32 (9.2x vs 5.3x) and gone by n = 128: first-layer
  structure matters less as depth grows.
- **Gate-adapted designs can be badly biased.** The Kerdock design aligned to the polar frame (randomised only by
  signed permutations, which keep the alignment) has nMSE x m = 54-195 (PW3 holds); the fixed aligned design 53-204
  (E4). Its vectors have flat +-1/16 coordinates, so in the canonical frame they are atypical inputs to the first
  layer. Alignment keeps unbiasedness only for point sets that are exactly Gaussian conditional on W (scrambled nets,
  cube shifts) [A].
- **First-layer controls** (dev, descriptive, one net; HEURISTIC). Regressing the output on first-layer features with
  closed-form means (z_i, |z_i|, relu(z_i), z_i^2, z_i|z_i|, |h_1|^2) explains 0.39 of the variance out of sample on dev
  net 100, nearly all of it harmonic degree 1 (the linear part, 0.49 of the odd variance) and degree 2 (0.25 of the
  even variance; |z_i| adds 0.02): a 2-design already cancels most of it. The features do carry higher harmonics, so
  the redundancy is approximate, not exact [A].
- **Shared trajectories (multilevel prefix-norm controls).** Best two-level gain over radius-exact MC 0.96-1.03 at 256
  x 32 (PW4 holds) and 0.83, 0.86, 0.94, 0.99 along the ladder: at L = n/8 the median gain approaches 1 from below as
  width grows (one net, 202, reaches 1.03). The control's residual at the best prefix depth is ~0.9 of the variance;
  even with a 31-layer prefix it is 0.27-0.56. Its proven Omega(L / log^2 L) growth needs the residual to fall like
  1/r + n/L^2, which shows only when depth reaches about the width (1.2 -> 2.4 at L = n for n = 32 -> 256, [prior:
  bits-bounds]).
- **Exact tails** (descriptive, runs/tail_check_n256_L32.json). On 4,096 inputs per net, 2-6% of the last layer's gates
  differ from the per-neuron majority pattern; 16 of 4,096 inputs on net 200 (none on the others) match it in the last
  layer and none in the last two. The common-tail-mask structure behind nd-exact's RQMCJ (3-15x over RQMC at 8 x 256)
  is absent at 256 x 32; this rules out one common mask, not every algebraic simplification of the tail.
- **Answer.** At 256 x 32 none of the three white-box structures beats the best black-box one (the Kerdock 5-design,
  3.0x); the gate-aligned frame adds at most a few tens of percent at some budgets, multilevel and exact tails nothing.
  Random bits can be cut to zero at no loss in ensemble accuracy (a suitable fixed point set exists; the proof is
  existential), or kept at 16 kbit (pairwise),
  at the price of per-net unbiasedness and error bars (section 4.7). Less compute per unit precision beyond a constant
  would need structure that lowers the effective dimension; none is present at L = n/8.

### 4.7 Random bits against precision at fixed compute
At ARC's budget (2^16 passes, 256 x 32; Table C), the random bits each sampler draws span nine orders of magnitude,
from plain MC's N n b = 5.4e8 (b = 32 bits per number) through a Haar rotation's ~1.0e6 (32,640 angles), scrambling's
135,168 (LMS matrices and a digital shift) and pairwise independence's 2 n b = 16,384 (X_i = Phi^-1((U + iV) mod 1)) to
zero (fixed point sets). Precision does not follow: the five randomised samplers and the two fixed ones all land
between 1.5x and 2.9x over MC (medians), with the fixed polar Sobol set (zero bits) at 2.68x and pairwise MC (16 kbit)
at 1.53x. Randomness is not the currency; structure is.
What the bits do buy, on the given net:
- **Unbiasedness and an error bar.** Randomised rules are unbiased conditionally on W and give an unbiased replicate
  variance (an unbiased variance estimate, not automatically a confidence interval with reliable coverage [A]);
  fixed rules are right only on average over nets and report nothing about their error on this net (per net the two
  fixed Sobol sets range 0.8x-5.5x at 2^16, Table C).
- **A caveat on few bits.** Pairwise independence fixes the MEAN risk at exactly MC's, not its distribution over
  seeds. In a toy check (d = 3, N = 64, 200,000 seeds) the Kronecker-form pairwise sampler had mean risk 0.994 x MC
  but median 0.067 x MC, and the worst 1% of seeds carried 49% of the mean. On the nets, three of five 16-replicate
  averages came out at 0.59-0.65 x plain MC, one at 1.06, and one drew a bad seed (net 203: 2.52 x MC). PC4 is scored
  on medians and holds (median 0.67 x MC; the band was [0.6, 1.5]).
- **Zero bits for free, on average.** Weight-derived seeds [prior: unrestricted 3.1; A] turn any of these randomised
  rules into a deterministic one with the same ensemble risk, at O(n^3) extra cost, under exact-transform primitives
  (with finitely many bits per weight only approximately).

## 5. Scorecard
Generated from the hashed scorers' outputs (scratch/score.py, rate_fit.py, score_a2.py via
scratch/insert_scorecard.py; results/scorecard_full.md). 32 registered predictions: 25 hold, 7 fail.

| id | prediction (PREREG wording abbreviated) | measured | result |
|---|---|---|---|
| PS1 | median odd share in [0.45, 0.70], every net in [0.40, 0.75] (256 x 32) | [0.585, 0.556, 0.572, 0.563, 0.531] | HOLDS |
| PS2 | median share of degrees <= 2 in [0.30, 0.65] (256 x 32) | [0.619, 0.467, 0.472, 0.436, 0.263] | HOLDS |
| PS3 | median even >= 6 share in [0.08, 0.30] (256 x 32) | [0.113, 0.169, 0.161, 0.186, 0.212] | HOLDS |
| PB1 | Kerdock (Haar) gain over MC: median in [2.0, 5.0], >= 1.5 on 5 of 5 | [4.69, 3.04, 3.24, 2.74, 2.4] | HOLDS |
| PB2 | no Haar-randomised design gains more than 6 over MC (256 x 32) | 4.69 | HOLDS |
| PW1 | polar RQMC beats standard RQMC at 2^14 on >= 4 of 5 nets, median ratio >= 1.2 (256 x 32) | [1.37, 1.15, 1.76, 1.31, 0.82] | HOLDS |
| PW3 | polar-aligned Kerdock (signed permutations) biased: nMSE x m >= 5 on >= 4 of 5 | [94.6, 195.1, 54.2, 94.1, 93.2] | HOLDS |
| PW4 | multilevel two-level best gain over RB-MC <= 1.1 on 5 of 5 (256 x 32) | [1.0, 0.986, 1.03, 0.988, 0.961] | HOLDS |
| PS4 | simplex gain and E_1 share fall from n = 32 to 256 (medians) | {'simplex_gain': {'32': 4.49, '64': 2.82, '128': 2.1, '256': 1.93}, 'E1': {'32': 0.466, '64': 0.38, '128': 0.288, '256': 0.319}} | HOLDS |
| PW2 | polar/standard RQMC ratio at 2^14, median per width in [0.8, 1.3] for n = 32, 64, 128 | {'32': 1.26, '64': 0.99, '128': 1.17} | HOLDS |
| PC1 | rqmc_polar gain >= 2.0 on >= 4 of 5 and >= 1.2 x rqmc_std on >= 4 of 5 (N = 2^16) | {'polar': [2.64, 1.83, 2.22, 2.47, 1.87], 'std': [2.77, 1.97, 2.13, 2.15, 1.95]} | FAILS |
| PC2 | Kerdock gain >= 1.8 on >= 4 of 5 (66,048 points, vs truth) | [3.93, 3.15, 2.79, 2.89, 2.2] | HOLDS |
| PC3 | best structured sampler gains in [2, 8] over MC on every net (none reaches 10x) | [3.93, 3.15, 2.79, 2.89, 2.2] | HOLDS |
| PC4 | pairwise-independent MC: median nMSE x N / (V_rb/V) in [0.6, 1.5] | [0.63, 0.67, 0.65, 2.57, 1.08] | HOLDS |
| PC5 | zero-bit polar Sobol: median nMSE x N <= 1 | [1.2, 0.31, 0.41, 0.23, 0.37] | HOLDS |
| PN1 | layer-1 GSA (pooled delta = 1e-6..1e-4) within +-10% of n/sqrt(2 pi), every net, all widths | [0.94, 1.071, 29] | HOLDS |
| PN2 | per-layer GSA (l >= 2) in [0.5, 1.3] x n/sqrt(2 pi), every layer, every 256 net | [0.329, 1.313] | FAILS |
| PN3 | total partition GSA in [0.6, 1.2] x n L / sqrt(2 pi) (256 x 32) | [0.767, 0.853, 1.034, 1.006, 1.071] | HOLDS |
| PN4 | NS_out(1e-5)/1e-5 in [0.4, 1.5] x n/V on every 256 net | [0.873, 2.089, 0.861, 0.435, 0.551] | FAILS |
| PN5 | P(gate pattern changes) at delta = 1e-6 >= 0.8 on every 256 net | [0.928, 0.944, 0.968, 0.964, 0.976] | HOLDS |
| PN6 | validity: design tail tau_2 >= the T6 lower bound from NS_sph (tolerance 0.05), every net | [[0.381, 0.185, True], [0.533, 0.329, True], [0.528, 0.287, True], [0.564, 0.258, True], [0.737, 0.441, True]] | HOLDS |
| PQ1 | centred Sobol (rb) fitted rate p over N = 2^8..2^16: median in [0.6, 1.4] (256 x 32) | [1.07, 1.14, 1.14, 1.09, 0.84] | HOLDS |
| PQ4 | Korobov lattice (rb) median nMSE x N at 2^14 >= 0.5 (256 x 32) | [0.46, 1.0, 0.53, 0.77, 0.41] | HOLDS |
| PQ2 | centred Sobol (rb) median nMSE x N at 2^14 higher at n = 256 than at n = 32 | {'32': 0.166, '64': 0.266, '128': 0.506, '256': 0.337} | HOLDS |
| PQ3 | k-wise designs, every resolved (net, k): nMSE x N in [0.05, 5] | {'count': 114, 'min': 0.132, 'max': 4.896} | HOLDS |
| PA1 | rqmc_std median p over 2^8..2^18 in [0.85, 1.25] | 1.045 | HOLDS |
| PA2 | rqmc_polar median p in [0.85, 1.35] | 1.058 | HOLDS |
| PA3 | no series has median p > 1.5 at 256 x 32 | 1.13 | HOLDS |
| PA5 | 512 x 64: median degree <= 2 share in [0.45, 0.60] | 0.405 | FAILS |
| PA6 | 512 x 64: median odd share in [0.55, 0.59] | 0.542 | FAILS |
| PA7 | 512 x 64: median simplex gain in [1.8, 2.6] and cross-polytope gain in [2.0, 2.8] | [1.71, 2.11] | FAILS |
| PA8 | 512 x 64: n / (mean V over the nets) in [35, 45] | 48.4 | FAILS |

Reading the failures: PN2 and PN4 put per-net, per-layer bands on quantities whose ensemble values hold (one layer of
one net at 0.33 x n/sqrt(2 pi); one net's Jacobian energy at 2.1 x its annealed value). PC1 is the white-box frame's
advantage, real at 2^14 and absent at 2^16. PA5-PA8 are astra's finite-size bands for 512 x 64, all narrowly missed;
astra's limit theorem is not tested by them. No prediction was changed after its data were seen. Part 1 was hashed
before any evaluation run; Amendment 1 was hashed before every E5 run but after I had seen the first E1 progress lines
of net 200 (svd-frame RQMC gains), so its sentence "before any evaluation result was read" is wrong (disclosed in
logs/deviations.txt); Amendment 2 was hashed before its runs. logs/deviations.txt lists every procedural change.

## 6. Verdict: what this says about ARC's goal
Does the randomness framing say something new and checkable about ARC's goal, beyond the arc-update benchmark? Yes,
three things, and it removes one reading of the goal.

1. **Runtime randomness alone does not separate estimators here (PROVEN existentially; PROVEN UNDER exact primitives
   for weight-derived seeds).** On ARC's ensemble some fixed, weight-independent point set (no random bits, no
   sampling in the literal sense) matches MC's average error at every budget, though the proof gives no efficient
   construction, and weight-derived seeds simulate any randomised estimator under exact-transform primitives. So ARC's
   working test for "mechanistic", "avoids any random or pseudorandom sampling", cannot be about random bits: a
   deterministic point set passes it and is still sampling. If the test is meant to exclude point evaluation, it has
   to say so about the representation (does the method evaluate the network at inputs?), and then derandomisation
   theory, PRGs included, has nothing to add. Your question "how many random bits does an estimator need" has the
   answer: none are needed for ensemble precision; bits buy per-net unbiasedness and an error estimate.
2. **It puts a measurable cap on structured sampling that does not look at the weights (PROVEN + MEASURED).** The best
   positive such point set has exponent between 1 + 2/(n-1) and 1 + 3/(n-1) (1.008-1.012 at n = 256), and its gain is
   bounded by the even harmonic share it cannot cancel: an estimated ~10-19x over MC at ARC's budget on these nets,
   against 2.9x achieved by the best one known (the Kerdock 5-design, essentially the strongest design that fits the
   budget). No positive weight-independent rule has a gain growing with width at polynomial budgets (PROVEN in the joint
   limit), and Kerdock's ensemble gain tends to 3.15. The cause is now known: the variance a
   point set fights at finite width is the accumulated activation-norm fluctuation of the early layers, a quantity the
   infinite-width picture drops (astra, round 2). This is checkable by anyone: measure five design variances.
3. **It quantifies the boundary-mass intuition and shows why it does not decide anything (PROVEN + MEASURED).** The gate
   partition has expected perimeter ~nL/sqrt(2 pi) and changes almost surely under noise of size 1e-6; the output is
   Lipschitz, decorrelates linearly (slope n/E V ~ n/11), and its harmonic content at the degrees point sets act on is
   a finite-width norm effect, not a count of boundaries. Any "hash" argument for hardness has to get from the
   scrambled gates to the smooth output, and this is the gap it has to close.
4. **What would change the asymptote.** Only structure that depends on the weights and lowers the effective dimension:
   an exact linear tail (exists when L >> n, not at 256 x 32), shared trajectories that track the norm profile (the
   multilevel sampler; it pays when L / (2 log^2 L) >> 1, i.e. L in the hundreds, HEURISTIC), or exact pre-integration
   along lines (the preint lane). The white-box structures tested at 256 x 32 (gate-aligned frames, first-layer
   controls, multilevel, exact tails) change the constant by at most a few tens of percent at some budgets and none at
   ARC's budget; every fitted exponent at 256 x 32 is 1.0-1.13, RQMC's excess over MC is ~0.05, and no change between
   frames was detected (section 4.1). That none of these moves the asymptotic exponent is a CONJECTURE. The open
   question astra names is the sharp version of your sampling direction: can a cheap weight-dependent construction
   predict the per-input norm profile well enough that the residual variance goes to zero, with its cost counted?

Net: the framing does not turn ARC's goal into a derandomisation problem (on random weights derandomisation is free);
it turns it into "how much error per forward pass can structure buy", answers that for weight-independent structure
(a bounded constant: 2.9x measured at 256 x 32, an estimated ~10-19x ceiling, a limit that does not grow with width),
and leaves the exponent question to weight-dependent structure, where nothing tested here moved it.

## 7. What is missing
- An unconditional statement for white-box point sets: the black-box identity and ceiling do not cover rules that
  depend on the weights, and the polar frame shows the dependence can matter at some budgets. No exponent bound for
  weight-dependent point sets is proven here.
- Widths beyond 256 along L = n/8: every trend in Table B is a slope over a factor 8 in width. The 3-net 512 x 64 test
  (Amendment 2) missed astra's finite-size bands; more nets and widths are needed to see how fast finite nets approach
  the proven limits.
- A non-antipodal 5-design of affordable size (it would avoid paying the even tail twice), and designs of strength 7
  (>= 5.7e6 points at n = 256), to measure the spectrum beyond degree 6 at full width; at small widths a U-statistic
  measurement of the degree-3..8 energies would do it.
- Error bars on the exponents beyond the 5 nets (E5 has 8 replicates at 2^18), and nets beyond 200-204 at 256 x 32.
- The k = 3 bounded-independence design at 256 x 32 (1.7e7 points) was started on one net as a descriptive extra and
  stopped unfinished to free the CPU slot for Amendment 2.
- Deductive (non-point) estimators are outside this lane; the sibling preint lane covers exact pre-integration along
  lines, the one structure here whose exponent could differ from 1.

## 8. Review (gpt-6-astra, fresh session, read-only; "model: gpt-6-astra" checked; review/review_answer.md)
Started 15:52Z on the report as hashed in review/review_hashes.txt; it recomputed the scorers, rates, tables and
trade-off results from runs/ and found them matching, and it confirmed that the frozen hashes match the code. Verdict:
"needs revision before Kelvin; no critical numerical corruption", with 7 major and 3 minor findings, all accepted and
applied:
1. The exponent window is for the best positive weight-independent rule, not every rule; Kerdock's 3.15 is its own
   limit, not a bound for all rules (summary items 1, 4, 8 and section 6 reworded).
2. The 9.6-18.8x "ceiling" is a plug-in (4x the measured Kerdock gain by construction, 8-18% relative error, a 3%
   aliasing correction ignored), not a certified bound (now labelled an estimate).
3. RQMC's exponent interval excludes 1: a small finite-range excess; "no structured sampler beats RQMC" became "none was
   shown to"; the frame difference is "not detected", and the no-improvement claim a CONJECTURE.
4. Zero-randomness statements carry their qualifications (existential; exact primitives; finite-precision weights;
   a replicate variance is not a confidence interval; T2 covers Haar rotation, not scrambling or aligned frames).
5. First-layer control redundancy is HEURISTIC and approximate; the tail check rules out one common mask (net 200 has
   16 of 4,096 inputs matching it in the last layer), not every tail simplification.
6. The bounded-independence conclusion is limited to the tested constructions; Table E now shows resolved/run counts
   (k = 4 cells rest on 2 of 8 and 1 of 8 nets).
7. "Nothing reaches 4x on any net" was false for the fixed point sets (5.5x, 4.4x on single nets); corrected.
8. Amendment 1's "before any evaluation result was read" is wrong (limited E1 exposure, disclosed); section 5 says so.
9. Numbers corrected: fixed Kerdock 0.11-0.80; aligned fixed Kerdock 53-204; pairwise MC three of five nets at
   0.59-0.65x plain MC; multilevel reaches 1.03 on one net; slope wording; 1.94 and 1.61 gains; polar ladder exponent
   1.03 at n = 128; median V 10.7 at n = 32; section 7 updated (512 x 64 done; the k = 3 run stopped).
10. Bit counts labelled idealised (30-bit scipy scrambling, 53-bit floats in code); He-Wang's RMSE exponent converted to
    MSE (~0.002 at d = 256); "no full-net PRG exists" became "none was found".
