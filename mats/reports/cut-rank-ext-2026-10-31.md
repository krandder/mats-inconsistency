# Nonlinear readouts and back-and-forth: where the (1/eps)^n law survives (cut-rank extension, 31 October 2026)

Lane: cut-rank-ext (theory, target M1). Bead: pool-4fm6vg.

Kelvin: "Of course, you should definitely seek the lane to prove that" (nonlinear readouts of a summary; bounded
back-and-forth methods). Earlier: "I am very happy with proving only the asymptotics."

Work directory `work/cut-rank-ext/`:
- `FORMAL-ext.md`: my statement and lemmas, hashed at 11:02Z before I read the co-prover, plus two addenda
  (`notes/formal_hashes.txt`).
- `coprover/`: the lemma-free statement, prompts, both answers, `PROVENANCE.md`.
- `notes/arc_estimators.md` (ARC's estimators, sourced), `notes/lit_widths_comm.md` (widths, real-valued communication),
  `notes/coverage_map.md` (the full coverage map).
- `PREREG.md` (hashes in `prereg_hashes.txt`, two amendments), `scratch/`, `runs/`, `logs/`, `review/`.

Co-prover: gpt-6-astra, two rounds at xhigh effort. Final review: GPT-6 in a fresh session (section 11). Labels: PROVEN,
PROVEN UNDER X, CONJECTURE, MEASURED.

## Summary for Kelvin

**Short answer: no, as stated.** The (1/eps)^Theta(n) law survives only as a count of distinguishable summaries (bits:
Theta(log 1/eps); readouts analytic on a fixed complex neighbourhood: at least c log(1/eps)), or for readouts of bounded
polynomial degree. A readout of any finite smoothness needs a number of reals that does not grow as eps -> 0. All bounds are
asymptotic in eps at a fixed architecture, with astronomically small constants.

1. **It breaks for nonlinear readouts (PROVEN, by construction).**
   - The front's own weights, n^2 c numbers, read by a locally Lipschitz decoder, are exact at every precision. After
     clipping, a decoder with a polylogarithmic Lipschitz constant reaches any eps.
   - The genuine-ensemble theorem of the last lane rests on a rare event on which the cut state is a ray. There the cut
     mean (n numbers) pushed through the back is exact. That is mean propagation. ARC's own K = 2 code, fed the mean with
     zero covariance, reproduces it to 2e-5.
   - So the precision exponent belonged to linear readouts. The kink functions u -> g_b(u) form an n-dimensional curved
     family. Its linear span is infinite-dimensional; its dimension is n.
2. **It breaks for back-and-forth with real numbers (PROVEN).** Two rounds suffice: the back sends its n^2 d weights and
   the front returns the n answers.
3. **Where it survives (PROVEN).**
   - Readouts of polynomial degree at most D need s >= c eps^{-2(n-1)/(3D)} numbers. With t rounds in which every message
     and the final output are polynomials of earlier messages, with no branching, the exponent is divided by
     D(1 + D + ... + D^t) (gpt-6-astra).
   - Bits, with any number of rounds and any decoder: at least n log2(1/eps) - O(1) bits, so on the order of (1/eps)^n
     distinct transcripts (gpt-6-astra). One-way the bound is (n(n+1)/2) log2(1/eps) - O(1); (n^2 c + o(1)) log2(1/eps)
     suffice.
   - Readouts analytic on a fixed complex neighbourhood, with amplitude allowed to grow polynomially in 1/eps, need at
     least c log(1/eps) numbers. A matching O(log 1/eps) is shown for radius ratio 1/2.
4. **Finite smoothness gives a count that does not depend on eps (PROVEN).**
   - Lipschitz decoders need between n(n+1)/2 and n^2 c numbers. This needs the message to stay in a ball of radius R
     outside a small exceptional mass, with log(R x Lipschitz constant) = o(log 1/eps).
   - C^k decoders (all derivatives up to order k budgeted) need between max(2k(n-1)/3, n(n+1)/2) and (k+1) n^2 c.
   - With budgets fixed in advance, every count leaves an error floor, because the target is unbounded. The budgets may
     grow polylogarithmically in 1/eps.
5. **Moments (gpt-6-astra, PROVEN).** At a fixed architecture some finite moment order determines every cut law. So
   moments plus an unrestricted continuous closure are exact. The order is not explicit. The mean alone is not enough
   in general.
6. **ARC's actual closure (gpt-6-astra, PROVEN in a narrow version).** Take Wu et al.'s basic truncation for one back
   layer, fed the exact cut cumulants. It is biased on the ray event at every fixed order K. So that one-step estimator
   needs K to grow at least like log(1/eps)/log log(1/eps). That forces only a polylogarithmic state, nowhere near a
   loss to sampling. ARC's own code shows the bias numerically (1-4% at K = 4 on a ray family).
7. **Your frame argument.**
   - Naming frames adds nothing to a moment summary that is read linearly, provided the naming and the readout do not
     react to the front's replies (PROVEN).
   - Under the same condition, with B bits of naming and s numbers read with degree p, 2^B C(s+p, p) >= r(eps). The atlas
     must then hold at least r(eps)/C(s+p, p) frames.
   - That is the precise sense in which "even that message is long". In real numbers it is short: the back's weights.
     With adaptive naming only the information limit of the moments remains.
8. **For ARC (coverage map, section 7).**
   - Covered: estimators that read a front summary linearly. These are ARC's deduction-projection estimators for
     halfspaces, products and permanents, and the same pattern applied to narrow ReLU nets (which would need more numbers
     than 1/eps^2 samples once n >= 5).
   - Not covered: ARC's ReLU cumulant propagation, mean and covariance propagation, mixtures, region splitting, sampling
     hybrids and advice-based estimators.
   - For these, unrestricted communication defeats any precision-power separation. What applies is the dimension and bit
     counts, conditional on regularity they do not have uniformly, and the narrow bias floor of item 6. What is left is
     running time.
9. **Measured (pre-registered; post hoc parts marked in section 6).** Widths 8 and 16; cut moments estimated from 8,192
   input directions per front, radius exact.
   - Errors are in units of Vbar under the registered normalised loss. Merged fronts (n, c, d) = (8, 64, 8): the
     in-sample estimate of the best linear protocol with 9 numbers (optimistic) leaves 0.09. Mean propagation with 8
     numbers leaves 0.0002 (Monte Carlo floor 0.00006). ARC's K = 2 closure fed the estimated mean and covariance leaves
     0.71.
   - Inside the window mean propagation still beats that linear estimate with one more number, by 1.1-2.8x. ARC's K = 2
     closure is worse than mean propagation at width 8 and level with it at width 16.
   - A bit protocol on the mean (quantised mean, decoded by mean propagation): 48 bits reach 0.01 on merged fronts, and
     the error falls like 2^{-2j} with j bits per coordinate. This uses a global norm code, after the review found an
     uncounted scale. It is an upper-bound demonstration, unrelated to the hypotheses of the bit lower bound.
   - Identifiability is inconclusive. The Jacobian spectra fall smoothly, and at our Monte Carlo precision only 3-16 of
     22-81 directions are resolved.
10. **Bottleneck.** A lower bound in an explicit model of computation. For ARC's actual algorithm, the missing piece is a
    polynomial-rate bias bound for its closure that survives errors carried in from earlier layers.

## 1. The question, made precise

**Setting** (as in the cut-rank lane).
- Width n, depth L = c + d, He-ReLU, no bias, noiseless, x ~ N(0, I).
- Front = layers 1..c, back = layers c+1..L, drawn independently. P_f is the law of the cut state h_c; g_b is the back
  map; M(f, b) = E g_b(h_c).
- Criterion: ARC's averaged matching-sampling one, Err = E_{f,b} (1/n)|M - est|^2 <= eps^2 Vbar.
- Notation: a = 2(n-1)/3, p = n^2 c, q = n^2 d, l = log(1/eps).

**Known before this lane.**
- A one-way linear summary needs r(eps) numbers, the eps-rank of the cut operator.
- For the genuine ensemble at fixed (n, c >= 3, d), r(eps) = Theta(eps^{-a}).
- The lower bound comes from a rare event E (probability about 2^{-n^2}) on which h_c(x) = alpha(x) v, a ray.

**A protocol.** The front sends a message computed from its weights alone; the back outputs an estimate from the message
and its own weights. Back-and-forth protocols exchange several messages.

**Why a regularity class is needed.** With arbitrary decoders one real number carries everything: interleave binary
expansions (Luo-Tsitsiklis 1993, p. 1021), or use a space-filling curve (DeVore-Howard-Micchelli 1989, p. 469; Siegel
2024). So each result below names its class: what the decoder may do with the numbers it receives.

**Budgets.** If a class fixes, independently of eps and uniformly over backs, the decoder's amplitude, or its Lipschitz
constant together with a bound on the message radius, or its derivative bounds on a bounded message domain, then every
message size leaves an error floor. The target is unbounded under Gaussian weights, and such a decoder cannot follow it
(gpt-6-astra, round 1 A3 and round 2 R2.1). The classes below therefore let budgets grow subpolynomially in
1/eps, or polynomially where stated. The constants hidden in every lower bound come from the rare event E, so they are
astronomically small.

## 2. Nonlinear readouts: the full ladder (target 1)

Minimal message size s* for the genuine ensemble at fixed (n, c >= 3, d), as eps -> 0. "Claude" = this lane's FORMAL-ext;
"astra" = gpt-6-astra (round, item); details and proofs are in `FORMAL-ext.md` and `coprover/astra_r*_answer.md`.

| decoder class | lower bound on s | upper bound on s | who |
|---|---|---|---|
| linear in s reals | C eps^{-a} | Theta | cut-rank lane |
| polynomial of degree <= D | (D! C)^{1/D} eps^{-a/D} - D | | Claude B; astra 1.e |
| t rounds of polynomial messages, no branching | C^{1/(DS)} eps^{-a/(DS)} - D, S = 1 + D + ... + D^t | | astra A8 |
| analytic on a fixed polydisc (radius ratio rho < 1), amplitude <= eps^{-kappa} | c(a, rho, kappa) l | (p + o(1)) l / ln 2 at rho = 1/2, front and back clipped, kappa >= p log2 3 | Claude B1, fixed by astra R2.2; upper bound Claude Addendum 2, checked by the review |
| bits, any rounds, any decoder | n log2(1/eps) - O(1) | (p + o(1)) log2(1/eps) one-way | astra A4; Claude A2 |
| bits, one-way | (n(n+1)/2) log2(1/eps) - O(1) | (p + o(1)) log2(1/eps) | astra R2.4 |
| C^k on a bounded message cube, polylog budget on all derivatives up to order k, uniform over backs | max(k a, n(n+1)/2)(1 - o(1)) | (k+1) p | lower: Claude B2 and the Lipschitz row; upper: astra R2.3 (my Addendum 1 gives only C^{k-1,1}) |
| Lipschitz: decoder gamma-Lipschitz from R^s into the mean-square space over backs, message in a ball of radius R except on a mass below a fixed fraction of the chart event's probability, log(R gamma) = o(l), n >= 2 | (n(n+1)/2)(1 - o(1)) | p (clipped weights) | astra R2.4 with Claude C4; astra A2 |
| continuous decoder, measurable encoder | none | 1, exact (continuous surjection R -> R^p) | Claude C3 |
| moments through order K, any continuous closure | error >= e_K^2 | exact for K >= Q(n, c) (not explicit) | astra A5, A7 |

All entries are PROVEN. The analytic upper bound (Addendum 2) was checked by the final review, not by the co-prover. The
n(n+1)/2 rows need the genuine ensemble: on ray fronts alone the family has dimension n.

**Why the linear argument fails (PROVEN).** The linear bound says: estimates <phi(f), psi(b)> form the operators of rank
<= s, and Eckart-Young bounds their error by the singular-value tail. A nonlinear readout R(phi(f), b) instead ranges over
the image of an s-dimensional set. On ray fronts the exact answer M = g_b(E h_c) is such an image, of dimension n, with an
infinite-rank covariance. So the step "rank <= s" fails, and only dimension survives. Each surviving bound converts the
readout's regularity back into rank:
- *Degree D.* Expand the polynomial into monomials of the s numbers: rank C(s+D, D).
- *Analytic.* Cauchy estimates on a fixed polydisc bound the homogeneous Taylor parts by A rho^m. Truncate at degree
  m ~ l/log(1/rho) and compare C(s+m, s) with eps^{-a}. The entropy of that binomial forces s >= c(a, rho) l. My first
  version claimed only l/log l, with a wrong asymptotic (astra round 2).
- *Analytic, upper bound.* Quantise the clipped front weights to (p + o(1)) log2(1/eps) bits, send them as reals in
  {-1, 1}, and decode with the multilinear extension of the target (back weights clipped too). It is entire, with
  amplitude at most 3^B times the target on the radius-2 polydisc, so radius ratio 1/2 (the review's sharper bound
  sqrt(5)^B absorbs the polylog factors). For other ratios the amplitude threshold changes.
- *C^k.* Jackson's theorem in s variables gives a polynomial of degree (Lambda/eps)^{1/k}; the rank count forces s >= k a.
- *C^k, upper bound.* Send k + 1 interleaved Cantor-coordinate streams of the clipped weights (astra): the inverse of
  this code is Holder of order strictly above k. The clipped target, Lipschitz in the weights, is then a Whitney jet with
  zero derivatives on the code's image, and Whitney's extension gives a C^k decoder whose budget grows only
  polylogarithmically. My 1/k-snowflake version (Assouad) has an inverse of order exactly k and gives only C^{k-1,1}.
- *Lipschitz.* A gamma-Lipschitz decoder maps the encoder's ball of radius R_enc in R^s onto a set covered by
  (1 + 2 R_enc gamma/t)^s balls of radius t. So s is at least the covering dimension of the law of M(f, .), up to the
  exceptional mass. Astra found a chart of dimension n(n+1)/2 inside the
  genuine ensemble: when W_2..W_c are near positive invertible matrices, h_c = A relu(Z) with Z ~ N(0, W_1 W_1^T). The
  one-layer ridge means E relu(w^T A relu(Z)) at w = A^{-T} e_i and A^{-T}(e_i - e_j) equal sigma_i/sqrt(2 pi) and
  (sigma_i - sigma_j + sd(Z_i - Z_j))/(2 sqrt(2 pi)); these recover the covariance bi-Lipschitzly. I re-derived both
  identities.
- *Bits.* Any protocol tree with B bits has at most 2^B leaves, each a rectangle. The output is then a rank-2^B linear
  protocol, whatever the decoder, so 2^B >= r(eps). For each fixed back the output takes at most 2^B values. So a
  quantisation (rate-distortion) bound on any bi-Lipschitz cube of fronts gives the stronger n log2(1/eps): astra used
  the ray event times a back event on which g_b is linear.
- *Continuous decoder.* A continuous surjection from R onto R^p (space-filling curves on growing balls), with a
  measurable selector as the encoder, gives s = 1, exact for every weight vector.

**The mean alone (astra A6).** First moments have an error floor in general (n >= 2), but are exact on rays. For c = 1,
moments through order 2 suffice.

## 3. Bounded back-and-forth (target 2)

- **Real messages (PROVEN).** Two rounds: the back sends its q weights, the front returns the n entries of M. That is
  exact, and the final readout is the identity. Bounded interaction cannot restore a bound that already fails one way.
  Adaptivity makes real-number counting collapse further: about log2 m + 1 adaptive continuous measurements recover a
  vector in R^m to any precision (Krieg-Novak-Ullrich, arXiv:2412.06468).
- **Exact smooth protocols (literature).** Abelson (1978): a C^2 protocol with any number of rounds needs at least the
  rank of the mixed Hessian of M in (front, back) weights. That rank is at most min(p, q), so even this bound never
  exceeds the parameter count. Abelson's own example shows two-way protocols can beat the one-way span dimension by any
  factor.
- **Bits (PROVEN).** As in section 2: B >= n log2(1/eps) - O(1) for any number of rounds.
- **Fixed feature span (PROVEN, Claude Theorem D).** Suppose every number the front sends is E_{P_f} e(h), with e in a
  fixed N-dimensional space of functions, the back's choices do not depend on the front's answers, and the back reads the
  answers linearly. Then the protocol is linear of rank <= N, so N >= r(eps), whatever the number of rounds.
- **Frame naming (PROVEN, Claude D1 and D2; both need the naming and the output coefficients to ignore the front's
  replies).**
  - If the back names any frame V and the front returns moments of degree <= K of the projected law, every feature is a
    polynomial of degree <= K in h. Read linearly, naming frames then never beats sending all moments of degree <= K one
    way.
  - With B bits of naming, s forward reals and a degree-p readout: 2^B C(s+p, p) >= r(eps). An adaptive threshold on a
    real reply already has infinite separation rank, so this needs non-adaptive naming.
  - With nonlinear processing of moments, the obstruction is the information limit e_K: interaction restricted to
    moment answers cannot beat it (astra A7).
- **Breaking it with features.** If the back may name features from a curved family, two rounds suffice. For one back
  layer the ridge features relu(w.h) at the back's own rows are exact with n^2 + n reals.
- **The forward-measurability result (kstar-frames Proposition 2)** shows that a front alone cannot compute the back's
  frame. It does not constrain two-way protocols, because the back can send the frame. No bound above needs it.

## 4. Layer-by-layer fixed-size states

A layer-by-layer estimator keeps a state S_l in R^s with S_l = U_l(S_{l-1}, W_l). At any cut its state is a front
message, and the rest of the computation is a decoder. So it inherits the one-way class of that composed decoder
(PROVEN). The regularity and radius hypotheses must hold for the actual cut state and the composed readout, not only for
one update:
- updates of polynomial degree <= D with a linear output: s >= (D/e) r(eps)^{1/D} - D at the last cut (needs L - 1 >= 3,
  or rays);
- C^k or Lipschitz updates: the eps-independent counts of section 2;
- merely continuous updates: nothing.

## 5. ARC's actual closure

**Information is not the obstruction (astra A5).**
- For a fixed architecture the cut law is a push-forward of a Gaussian through finitely many cone-linear pieces.
  Gaussian integration by parts on cone faces makes every moment a polynomial in finitely many quantities.
- The Hilbert basis theorem then gives a finite order Q(n, c) beyond which equal moments force equal laws. A continuous
  decoder of the moments through Q is exact (compactness, then Tietze).
- So "a fixed continuous map of a moment vector" is exact once the order is high enough. Q is not explicit.

**ARC's specific truncation is not (astra R2.5, PROVEN for exact inputs).**
- Take Wu et al.'s basic finite truncation (their eqs. (13)-(14)), fed the exact cut cumulants, with one back layer, on
  the ray event.
- Along the realisable family alpha = X_+ + t(-X)_+ (t > 0), the truncated estimate is real-analytic in t on the whole
  real line, because the variance never vanishes.
- Fix a back row w with w.v > 0. The physical target (w.v)(1 + t)/sqrt(2 pi) is analytic in t too. At t = -1 the formal
  continuation is a standard Gaussian, for which the K >= 2 closure is exact and returns (w.v)/sqrt(2 pi), while the
  continued target is 0. So closure minus target is a nonzero analytic function of t and cannot vanish on any interval
  of t > 0. For odd K the extra trace of cumulant K + 1 must be supplied exactly.
- Quantitatively, the RMS error b_K over a fixed positive-probability neighbourhood of fronts and backs is at least
  c exp(-C K log(K+2)). The exact-input one-step estimator therefore needs K log K >~ log(1/eps). Its symmetric state
  then has Omega((l/log l)^n) entries at fixed n.
- Numerically (post hoc, `scratch/arc_bias_alpha_t.py`, `logs/arc_bias_alpha_t.log`): ARC's own code fed the exact
  cumulants of alpha_t v (n = 4, one back layer, t from 1/4 to 4, three back matrices) has relative bias 3-49% at K = 1,
  3-19% at K = 2, 1-59% at K = 3 and 1-4% at K = 4.
- I checked the moments E alpha_t^r = c_r(1 + t^r), the positivity of the variance on R, realisability for c >= 3, and
  exactness of the K >= 2 closure at the Gaussian point.

**What this does not show.**
- It concerns ARC's closure fed exact cumulants. The full algorithm propagates estimated cumulants, and earlier errors
  could cancel the final step's bias.
- A polylogarithmic state is not a loss to sampling.
- A precision-power lower bound would need a polynomial rate, bias >= c K^{-beta}. That would force a state of
  eps^{-n/beta} entries (CONJECTURE, section 10).

## 6. Measurements (pre-registered; MEASURED)

**Design** (`PREREG.md`, hashed 11:04:53Z before any evaluation run).
- Pilot: n = 8, c = d = 2, 16 fronts. It was run once through the scorer to test the code
  (`runs/pilot_n8_c2_d2_scores.json`); nothing was changed after it and it is not reported.
- Amendment 1 (11:29Z, after the first two configurations were seen). ARC's released code floors variances at 1e-10 in
  absolute units, and deep fronts have cut scales down to 1e-7. So its runs were corrupted on low-scale fronts. I reran
  it on unit-scaled cumulants, which is the same estimator by homogeneity, and report both. K2X, rerun the same way as a
  control, did not change.
- Nets: the cut-rank lane's ensemble and seeds. 512 fronts (256 train, 256 test) x 32 backs x n outputs.
- Three independent splits of 8,192 input directions per front, radius integrated exactly. The front numbers (mean,
  covariance, third cumulant) are estimates from one split, not exact moments.
- Amendment 2 (12:10Z, after the final review): NC2 rerun with a global norm code, ARC3 rerun with the fourth cumulant,
  and a moment-sample convergence check. All three are post hoc corrections.
- Primary criterion: the cut-rank lane's scale normalisation (each front row and back column divided by its RMS output);
  raw in brackets.

**NC1: linear against nonlinear readouts of the same front numbers.** Err/Vbar on test fronts.

| (n, c, d) | MC floor | MF: g_b(mean), n numbers | best linear, n+1 numbers (in-sample SVD estimate) | LIN1 (held out) | K2X: Gaussian closure, exact ReLU moments | ARC2 (unit-scaled) | LIN2 (held out) | best linear, 1+n+n(n+1)/2 numbers (estimate) |
|---|---|---|---|---|---|---|---|---|
| (8,4,4) | 5e-5 | 0.046 [0.019] | 0.051 [0.033] | 0.074 | 0.078 | 0.081 (median 0.035) | 0.023 | 0.0031 |
| (8,8,8) | 6e-5 | 0.031 [0.0036] | 0.087 [0.0097] | 0.15 | 0.82 | 0.95 (median 0.12) | 0.073 | 0.0016 |
| (16,8,8) | 5e-5 | 0.059 [0.023] | 0.12 [0.064] | 0.20 | 0.057 | 0.064 (median 0.033) | 0.21 | 0.0009 |
| (8,64,8) merged (211 of 256 test fronts alive) | 6e-5 | 0.00018 [4e-5] | 0.091 [1e-4] | 0.17 | 0.70 | 0.71 (median 0.11) | 0.11 | 0.0019 |

- On merged fronts mean propagation sits near the Monte Carlo floor. The in-sample estimate for the best linear protocol
  with one more number leaves 500 times as much.
- ARC's K = 2 closure fed the estimated mean and covariance is far worse on merged fronts. Fed the same mean with zero
  covariance, the same code is mean propagation (`logs/arc_zero_cov_check.log`).
- Before Amendment 1 the merged-front ARC2 error was 1.5e9 of Vbar (the variance-floor artifact); after unit scaling it
  is 0.71. Elsewhere the two versions agree.
- Convergence (Amendment 2, `logs/conv_check.log`): cut moments from 65,536 instead of 8,192 directions, on 64 test
  fronts of (8,8,8) and the 53 surviving ones of the first 64 of (8,64,8). The two sets of estimates differ by a mean
  square of 0.0003-0.0015 Vbar, and the errors change by 0.01-0.015 (ARC2: 0.91 to 0.92 and 0.76 to 0.77). So the large
  K = 2 closure errors are stable across these sample counts. That does not exclude every moment-estimation effect.
- ARC3: the registered runs omitted the trace of the fourth cumulant, which Wu et al.'s basic K = 3 needs, so they are an
  incomplete-input variant. Rerun with it (Amendment 2), ARC3 is unstable on these fronts: median per-pair errors of
  9.5 to 7.7e6 Vbar, and 32-88% of (front, back) pairs above 100 Vbar. Whether the closure or amplified
  moment-estimation noise causes this is not separated.
- The SVD columns are optimistic in-sample estimates (eigen-tails of a noisy cross-Gram) of the best linear protocol of
  that size under the registered normalised loss. They are not certified bounds, and Err/Vbar is not a per-network
  guarantee.
- LIN2 at width 16 fits 153 features on 256 training fronts and overfits (held out 0.21 against an in-sample optimum of
  0.0009); its P3 comparison is therefore generous to ARC2.

**NC2: bits on the mean** (j bits per direction coordinate plus 16 for the norm, decoded by g_b). The registered code
decoded the norm with a front-dependent scale that was never transmitted, so it was not a true bit protocol (review
finding 9). With a global norm code (Amendment 2; no overflows) the numbers are unchanged: on (8,64,8) the error falls
like 2^{-2.1 j} and 48 bits reach 0.01 Vbar. For comparison, the rank-based bound evaluated at the empirical in-sample
rank, log2 r_in(0.1), is 4.8 bits.

**NC3: identifiability** (Jacobian of front weights -> back-observed means; 64 backs, 65,536 inputs, two input sets).
- The exact symmetries behave as predicted: hidden-unit rescalings give singular values of 1e-15 relative, and input
  rotations stay at or below the Monte Carlo noise.
- The rest of the spectrum falls smoothly, spanning 3.4 to 8.2 decades, with no gap. Only 7-8 of 22 directions (n = 4,
  c = 2), 3-6 of 34 (4, 3), 11-16 of 51 (6, 2) and 8-12 of 81 (6, 3) are resolved at 10x the noise.
- Two fronts have numerical zeros (1e-14 relative), consistent with, but not certified as, the positive-probability
  extra symmetries astra pointed out (always-on units).
- Fronts rejected for dead units: 2, 6, 0 and 2. NC3 is inconclusive about the unresolved directions.

**Scorecard.**

| prediction | outcome |
|---|---|
| P1 merged: Err_MF <= 0.1 x best linear (n+1) | held (0.00018 against 0.0091) |
| P2 window: Err_MF >= 0.1 | failed in all three (0.031-0.059): mean propagation is better than I expected inside the window |
| P3 window: ARC2 < LIN2 in at least 2 of 3 | failed: 1 of 3 (only (16,8,8)); at width 8 ARC's K = 2 from sample-estimated cut moments is worse than a held-out linear readout of the same numbers |
| P4 LIN1, LIN2 >= 0.8 x the SVD floors | held in all four |
| P5 bits slope in [-3, -1] per coordinate bit | held (-2.1); the registered code used an uncounted front scale, and the corrected global code (Amendment 2) gives the same slope |
| P6 bits: 48 bits within [log2 r_in(0.1) - 1, 12n + 16] | held only after the correction: with the registered code it was not a 48-bit protocol; with the global norm code (no overflows) 48 bits again reach 1% |
| P7 every non-symmetry direction resolved at 10 x noise | failed in all four configurations |
| P8 rescalings exactly zero, rotations <= 3 x noise | held in all four |

## 7. Coverage map: ARC's estimators (sources in `notes/arc_estimators.md`; full table in `notes/coverage_map.md`)

| ARC estimator (source) | structure | what our bounds give |
|---|---|---|
| K-th order cumulant propagation, basic, augmented, factorised (Wu, Lecomte, Winer, Robinson, Hilton, Christiano 2026, arXiv:2605.05179, Sec. 4, S.4; code `mlp_cumulant_propagation`) | layer-by-layer, ~n^K/K! cumulants; Hermite diagram sums with per-unit Gaussian centring; coefficients with sigma^{-(k-1)} factors | nonlinear without uniform regularity, so the dimension bounds apply only conditionally and the bit bound only to a finite-precision implementation. Meeting a state-count threshold would not by itself give arbitrary accuracy. The exact-input bias floor of section 5 applies to the basic truncation |
| ReLU covariance propagation: Wu et al. K = 2 (Alg. 2, leading-Hermite off-diagonal update); full Gaussian moment matching (Xu 2024 Method 1; G. Wu thesis 2025 Sec. 2.1); WhestBench baseline (implementation not verified) | mean and covariance per layer | as above |
| K = 1 (mean plus a variance trace, Wu et al.) and mean-only propagation | state n + 1 or n | n + 1 is below n(n+1)/2 for n >= 3, so IF its readout were Lipschitz with subpolynomial constants it could not reach arbitrary accuracy; its Hermite step is not uniformly Lipschitz, so this is conditional. Mean-only from the exact cut mean is exact on the ray event |
| Polynomial-activation results (Wu et al. Thm 5.1, 5.2) | polynomial updates | covered by the degree bound, but vacuous there: the cut operator of a polynomial net has finite rank. No conflict with ARC's theorems |
| Deduction-projection with harmonic projection (Hilton, halfspace note, 2025) | projection onto a fixed, instance-independent subspace; linear steps | covered at every cut. For ReLU nets the same pattern would need >= C eps^{-2(n-1)/3} numbers. ARC proposed it for halfspaces only |
| Gelfand-pair and permanent deduction-projection (W. Wu notes, 2025-26) | linear projections | covered, for their own problems |
| Sketching theorem (Hilton note, Thm 3.1, informal) | m eigenfunction coefficients | another normalisation; no conflict |
| Random-walk Fourier sketching (Winer note, 2026) | backward random walks choose an instance-dependent support | among the sources inspected, the only ARC estimator with a backward component; covered only by counting its whole naming transcript in bits; a different estimand |
| Gaussian-mixture pruning (G. Wu thesis Sec. 6.4); region enumeration (AlgZoo 2026; Hilton 2024) | adaptive case splitting: pruning is discontinuous, region counts grow with depth | outside the regularity classes; different estimands (accuracy of tiny trained RNNs) |
| Importance sampling and logit extrapolation (G. Wu and Hilton, ICLR 2025) | sampling | a different estimand and access; sampling itself is a Lipschitz readout of sampled cut states, so it meets the dimension bounds rather than evading them |
| Matching sampling principle with explanations, train-and-explain (Neyman et al., "Competing with sampling", 2025) | advice computed from all weights | outside every size bound: sending the weights already works. Needs a running-time bound |

The coverage map also lists, as other problems: Christiano-Neyman-Xu's arithmetic-circuit mean, covariance and cumulant
propagation (2022, App. D.2-D.6), Boolean-circuit and selected-term estimators, Xu's three #SAT estimators (2024),
pairing and permanent estimators, activation modelling and other proposals (`notes/coverage_map.md`). The list is not
exhaustive: the fact sheet names the sources it could not open.

**The four classes asked about.**
- *One-way continuous readout.* Every ARC estimator that reads a cut summary is in it, but the class carries no precision
  bound: a continuous decoder needs one real, a Lipschitz one between n(n+1)/2 and n^2 c. Only linear or bounded-degree
  readouts keep eps^{-a}, and ARC's ReLU closures are neither.
- *Bit-bounded messages.* Every finite-precision implementation is in it. The bound, n log2(1/eps) - O(1) bits with any
  number of rounds, is far below the state of any ARC estimator (n^K floats).
- *Bounded back-and-forth.* Among the sources inspected only Winer's random-walk sketching has a backward component.
  With real messages there is no bound; with bits, only the logarithmic one.
- *Layer-by-layer fixed-size state with continuous updates.* This is exactly ARC's cumulant propagation and its mean and
  covariance special cases. The class inherits only the dimension bounds of its composed readout, and only under uniform
  regularity that ARC's updates lack. With polynomial updates of degree D it would inherit eps^{-a/D}, but ARC's ReLU
  updates are not polynomial (Gaussian pdf and cdf of each unit's mean and variance). What is proven against its specific
  update is the exact-input bias floor of section 5.

**What stays uncovered, and how far it is from ARC.**
- Unbounded joint computation: this is the regime of ARC's matching sampling principle, so the gap there is total.
- ARC's actual ReLU cumulant propagation: only the exact-input bias floor touches it, plus dimension bounds that are
  conditional on regularity it lacks.
- Adaptive case splitting with unbounded case counts: ARC uses it only for tiny trained models.

## 8. Quantifiers and depth window (target 3)

- **Lower bounds are ensemble statements per protocol.** For each protocol in the class, the ensemble error exceeds the
  bound. They hold at fixed (n, c >= 3, d) as eps -> 0. The bounds derived from r(eps) also hold for ray fronts at fixed
  (n, d); the n(n+1)/2 bounds (Lipschitz, one-way bits) need the genuine ensemble, because ray fronts alone have
  dimension n (n - 1 at fixed amplitude).
- **They are carried by rare events.** The front ray event has probability about 2^{-n^2}. The bit bound also uses a
  rare back event, and the n(n+1)/2 chart an event with positive later front layers.
- **What does not follow.** "Most nets defeat each protocol" would need a small-ball bound, which is not proven. "On most
  nets every protocol fails" is false: a constant fitted to one net is exact.
- **Not uniform.** Nothing here is uniform in n or depth. At fixed eps as n grows, r(eps) is polynomial in n, and every
  rank-derived bound becomes polynomial or logarithmic.
- **Upper bounds differ in scope.**
  - Exact for every net and every eps: the raw-weight protocol (locally Lipschitz decoder), the two-round protocol and
    the continuous-surjection code.
  - Exact on ray laws only: the ray mean.
  - Ensemble approximation to any eps, not exact: the clipped Lipschitz, Whitney and multilinear constructions.

## 9. What stays outside

- **Unrestricted joint computation.** Communication is cheap: p or q + n reals, O(min(p, q) log(1/eps)) bits. So a
  separation from sampling must be a running-time statement in an explicit model of computation.
- **Black-box cubature.** Randomised cubature for Lipschitz integrands in m dimensions has error N^{-1/2-1/m} (Bakhvalov;
  Novak's survey). After truncating the Gaussian input, this already beats plain sampling's exponent at fixed n. It is
  itself a sampling method. So "nothing beats sampling" is false as a black-box time statement at fixed n. For white-box
  methods a time lower bound needs a complexity assumption.
- **Adaptive case splitting with unbounded case counts.** Its updates are typically discontinuous (pruning, thresholds)
  and its state grows, so the regularity classes need not apply. The bit bound applies to all bits it communicates and
  is weak.

## 10. Conjecture ledger

Each item says what would close it and the cheapest test.

**L1 (the bottleneck): a computational lower bound.**
- Statement: in an explicit model (allowed operations, precision, preprocessing, query access), every program meeting
  the criterion costs >= A eps^{-alpha(n)} with alpha(n) > 2.
- Status: not yet a precise target. Information arguments cannot give it: weights and finite moments already suffice.
- Test: on a tiny architecture, measure accuracy against cost for direct cone integration, high-order moment inversion and
  ARC's update.

**L2: a polynomial rate for ARC's closure.**
- Statement: bias b_K >= c K^{-beta}, with beta independent of n. With exact inputs this forces a symmetric state of
  eps^{-n/beta} entries.
- Status: new. Only exp(-C K log K) is proven. Wu et al. themselves do not expect convergence at fixed width (S.6).
- Test: high-precision scalar diagrams along alpha_t, increasing K.

**L3: from exact inputs to the full ARC algorithm.**
- Statement: a lower bound on the full estimator's error, or a comparison showing propagated errors do not cancel the
  final-step bias.
- Status: open.
- Test: full against exact-input ARC errors at small width as K grows.

**L4: a chart of dimension above n(n+1)/2.**
- Statement: a positive-density stratum of fronts with back-observed differential rank r > n(n+1)/2, giving
  s*_Lip >= r.
- Status: open. My conjecture I1 (generic rank p - n(n-1)/2 - n(c-1)) is false almost everywhere (astra R2.6), and NC3
  cannot resolve it.
- Test: certify a nonzero minor of the restricted Jacobian on a specified stratum with exact integration.

**L5: sharp widths.**
- Statement: C^k decoders in O(n^2 + kn) numbers, or a larger lower bound; an analytic upper bound with eps-independent
  amplitude.
- Status: open.

**L6: typical-network failure.**
- Statement: a small-ball bound for every protocol in a class, so that most nets defeat it.
- Status: open (the cut-rank lane's C4). The rare-event proofs cannot supply it.
- Test: error distributions on and off the constructed events.

**L7: an effective moment order.**
- Statement: an explicit Q(n, c) and a stable reconstruction.
- Status: open. Order 2 suffices at c = 1.
- Test: a certified moment collision with different back means.

## 11. Process and review

- **Co-prover** (`coprover/PROVENANCE.md`).
  - Round 1 was independent: astra saw only the lemma-free statement. We found the same caps, the bit-rank and degree
    bounds, and the ray mean independently.
  - New from astra: the n log2(1/eps) bit bound, finite moment determinacy, multi-round polynomial protocols, the
    bi-Lipschitz ray map, the fixed-budget floors, the n(n+1)/2 chart, the C^k Cantor construction and the
    exact-input ARC bias floor. I re-derived each, numerically where possible.
  - New from me: the analytic and C^k ladders (B1, B2), the exact space-filling protocol, the feature-span, frame and
    atlas lemmas, the streaming reduction, the snowflake construction and the analytic upper bound.
  - Astra corrected my B1 (the rate is log(1/eps), not log/loglog), my count for two-way protocols (n answers, not one),
    my C5 (the dimension n includes scale) and my conjecture I1 (false almost everywhere).
- **Final review** (GPT-6 = gpt-6-astra in a fresh session, high effort, 12:00-12:09Z; `review/review_gpt6_final.md`).
  - Verdict: major revision, 15 findings (11 major, 4 minor). It recomputed all four NC1 configurations, the NC2 slope
    and the NC3 counts from the raw files, verified both PREREG hashes, and found the central
    communication-against-computation conclusion intact.
  - All findings were applied, as follows:
    - (1) theorem domains and budgets restated; (2) the analytic upper bound restricted to radius ratio 1/2; (3) the C^k
      upper bound from astra's Cantor code (mine gives only C^{k-1,1});
    - (4) exactness and quantifiers made result-specific; (5) non-adaptivity added to the frame and atlas bounds;
      (6) section 5 reworded;
    - (7, 8) coverage map corrected: split variants, conditional dimension bounds, missing rows, "among the sources
      inspected";
    - (9) NC2 rerun with a global norm code (same result); (10) ARC3 rerun with the fourth cumulant (still unstable);
      (11) "exact moments" corrected and a convergence check added;
    - (12) SVD numbers relabelled as optimistic estimates; (13) pilot scoring and both ARC versions disclosed; (14) NC3
      descriptions corrected; (15) the summary's scope restored.
  - A focused re-check (fresh session, 12:27-12:30Z; `review/review_gpt6_r2.md`) returned minor revision: nine
    findings addressed and six partly (convergence wording, hypotheses in the Lipschitz and C^k rows, the coverage
    notes, case splitting, empirical bounds, summary units). Those six were then completed; no further rerun was needed
    and no third review was run.
- **Compute:** CPU only, nice 19 and idle I/O priority. The width-16 run used up to six BLAS threads for about ten
  minutes before I capped later runs at one or two threads. No GPU.

## Handoff

- Theory: `FORMAL-ext.md`, `coprover/astra_r1_answer.md`, `coprover/astra_r2_answer.md`, `coprover/PROVENANCE.md`.
- Code:
  - `scratch/nl_readout.py` (NC1/NC2 simulation), `scratch/nl_analyze.py` (scoring),
    `scratch/arc_unit_rerun.py` (Amendment 1);
  - `scratch/ident_jac.py` (NC3), `tests/test_ident.py` (finite-difference check of the Jacobian, passes at 6e-10);
  - `scratch/arc_zero_cov_check.py`, `scratch/arc3_diag.py` (post hoc).
- Data: `runs/nl_*.npz` with `_scores.json` and `_arcunit.npz`; `runs/nc3_*.json`.
- Next if wanted: L2 (the rate of ARC's closure bias on alpha_t), because it is the one route from these results to a
  precision power against ARC's actual method.
