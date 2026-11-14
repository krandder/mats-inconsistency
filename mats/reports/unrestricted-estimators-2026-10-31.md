# Can an unrestricted deterministic estimator beat sampling on random critical ReLU nets? (31 October 2026)

Lane: unrestricted estimators (theory and experiment), target M1. Bead pool-uwmcvo. Status: final, after two GPT-6
reviews (both "major revision"; every finding applied).

Work directory `work/unrestricted/`:
- `FORMAL.md`: theory v2.1. v1, written before the co-prover's answer, is frozen in `coprover/FORMAL_v1_as_written.md`.
- `PREREG.md` and Amendment 1; hashes in `prereg_hashes.txt` (11:28:40Z, before any evaluation result).
- `coprover/`: three gpt-6-astra rounds (prompts, raw logs, extracted answers) and `PROVENANCE.md` (who found what,
  what I checked, what I took back).
- `lib/` (nets, the branch-and-bound, the multilevel sampler), `tests/` (all pass on the registered code: `logs/test_bnb_final.log`,
  `logs/test_mlmc.log`), `scratch/` (runs and analysis), `runs/`, `results/`, `logs/`, `notes/` (questions,
  literature, analogies).

Labels: PROVEN, PROVEN UNDER X, COMPUTED (numerics of a stated model), MEASURED, HEURISTIC, CONJECTURE.
Provenance tags: [C] me (Claude), [A] gpt-6-astra, [C+A] both independently.

## Summary for Kelvin

**Short answer.** The adaptive whole-network estimator I built (a certified branch-and-bound) fails as a deductive
method: its bounds-only estimate never beats RQMC, and its centre-point estimate, which evaluates the net's gates at
chosen points and so is structured sampling, catches up with RQMC only at width 3. Its cost grows exponentially with
width. The difficulties prong then found two things. First, "deterministic" is no restriction on random weights, so
the question is whether sampling is optimal among all algorithms, and no one can currently prove that. Second, the
natural conjecture that plain Monte Carlo is optimal is false: astra's red-team of my conjecture found a multilevel
sampler that shares the late layers through one reference trajectory and beats radius-exact MC by a factor that grows
with depth (proven asymptotically under stated cost hypotheses; estimated ideal gain 15x at width 8 and depth 256;
about break-even at ARC's 256 x 32). So, in the idealised arithmetic model of 3.1, deterministic estimators can beat
plain sampling at depth, by better sampling rather than by deduction, and ARC's question at depth becomes matching
that sampler without sampling.

**Progress prong (MEASURED, pre-registered, 72 nets of widths 3-32 and depths 4-64).**
- The estimator is a branch-and-bound over input cones. It is exact on every cell where all units are certified
  stable, bounds every other cell with symbolic linear bounds computed from the actual downstream weights, returns a
  certificate, and refines where the certified gap (or a backward-sensitivity proxy) is largest.
- The deductive estimate (the midpoint of the certified bounds) never beats RQMC: its best ratio over all 126
  net-arm runs and budgets is 0.18 (width 3).
- The centre-point estimate against radius-exact scrambled-Sobol RQMC at equal FLOPs: at width 3 it reaches parity at
  1e6-4e6 passes (both near nMSE 1e-12) and wins 4-18x in 4 of 24 runs, all at depth 16-32. Those wins mostly shrink
  below 3x if its cost were doubled, and they concern only the noiseless mean (noise at sigma = 1e-3 moves these
  means by nMSE up to 1e-4). At width 4 it loses 33x-2e5x; at widths 6-8, 10^4-10^6x. At widths 16-32 it is still at
  the accuracy of 2-30 Monte Carlo samples after 4e6 passes, and 10^5-10^6x behind RQMC at 2.6e5-1e6 passes, the
  largest budgets where RQMC was run there.
- The certificate is tight only at width 3 (nMSE 5e-13 to 6e-8) and is vacuous from width 6 on, growing to 1e34 at
  depth 64.
- Cost: the passes it needs for nMSE 1e-4 grow about 10x per unit of width (1e4-4e4 at width 3, 2e6-8e6 at width 6,
  out of reach at 8 and above), against 130-8,200 for RQMC, and a cell costs 46-170 forward passes. Extrapolating
  that growth (HEURISTIC; no large model was run), this family cannot reach 10M+ parameters.
- Centred unscrambled Sobol points did 2.7-4.5x better than scrambled Sobol at widths 3-8 (geometric means). They are
  sampling at fixed points, and the gain of any such design over radius-exact MC is capped by the low-degree content
  of the net's angular function: the heuristic 2-design ceiling at L = n/4 falls from about 7 (width 16) to about 2
  (widths 64-256). Astra's proven bound caps positive-weight designs at about 4x in the infinite-width limit, and a
  plug-in estimate of the same bound at ARC's 256 x 32 is also about 4x (post hoc, 3 nets, noisy; not itself a
  bound).

**Difficulties prong.**
1. In an idealised arithmetic model, determinism is no restriction on random weights (PROVEN; one exact construction
   from the first layer's rotation symmetry, one from the weights' digits). So the question is whether sampling,
   implemented as well as possible, is optimal among all algorithms. "As well as possible" matters twice: batching
   inputs through fast matrix multiplication beats serial MC by n^{0.19}, and the multilevel sampler beats
   radius-exact MC by Omega(n / log^2 n) at depth proportional to width (PROVEN UNDER astra's layerwise cost
   hypotheses, which schoolbook costs satisfy). Its estimated ideal two-level gains, setup and pilot excluded, are
   1.5x at L = n = 128, 1.7-3x at L = 4n, 7x at L = 16n and 15x at width 8 and depth 256, and about break-even at
   L <= n/4.
2. No one can currently prove that any estimator is optimal. It would be a super-linear time lower bound for a natural
   explicit problem, and I know of no technique for those. The worst-case barriers (IKW, #NSETH) do not reach random
   weights, the interpolation route to average-case hardness needs a polynomial estimand, and the known average-case
   hard problems (spin-glass partition functions, random-circuit amplitudes, permanents) concern exact or relative
   precision. ARC's mean is the additive, infinite-temperature end, which sampling already reaches.
3. Restricted classes that provably fail, with sizes: positive-weight designs (gain at most about 4 in the
   infinite-width limit); polynomials in the weights (the mean's Hermite degree grows with depth, with
   finite-difference proxies of about 2, 7 and 17 at depths 4, 16 and 32, and plug-in estimates put degree-8
   polynomials at 6-17 MC samples at depths 16-32); first-layer cell selection (exponentially little mass); stable
   estimators (must be as rough as the mean). Any estimator must also meet three proven necessary conditions
   (correlation, final row, depth recursion).
4. Your analogies (section 4). Threshold-circuit derandomisation: close in the worst case (ReLU nets contain TC0), but
   on random weights its barrier disappears; no transfer to average-case hardness is established. Convex-body volume:
   the oracle separation has a proven ReLU analogue (no black-box algorithm beats MC by more than about 4x); no
   white-box or average-case reduction is established. Spin glasses give the only precise, testable observation: the
   infinite-width kernel fails the full-RSB concavity test at depth 9 and above (depths 3, 5 and 7 unchecked), which
   suggests, for a Gaussian surrogate only, an overlap gap for finding near-maximal outputs. It says nothing about
   means.
5. Bottleneck (section 5): the revised conjecture C0*, that nothing beats the best multilevel sampler by a
   polynomially growing factor at depth proportional to width. It has little evidence. Its cheapest refutation is a
   better control for the first layers' amplitude variation, which carries the variance the reference trajectory
   cannot remove.

**Which prong wins.** For deductive estimators the difficulties prong wins: our deductive estimate never beat RQMC,
none of the analysed classes matches sampling at scale, and no impossibility proof is in reach. The one positive
result of the lane came from red-teaming the difficulties prong, and it is about sampling: plain MC is the wrong
baseline at depth.

## 1. Setting and the two error measures

- Nets: width n, depth L, W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), h_l = relu(W_l h_{l-1}), output a_L in R^n
  (ARC's critical He-ReLU family; the noisy variant adds sigma r_l xi_l after each hidden ReLU).
- Notation (as in `FORMAL.md`): mu(W) = E_x a_L; V(W) = tr Var_x a_L and V = E_W V(W); m_n = E|x|; g_W is the output
  on unit inputs, so f(x) = |x| g(x/|x|) without noise; V_sph = m_n^2 E_W tr Var_u g_W is the per-sample error of
  radius-exact MC ("RB-MC"); S = E_W |mu - E mu|^2.
- Theory uses the average risk R(A) = E_W |A(W) - mu(W)|^2, compared with V/N for N-sample MC or V_sph/N for RB-MC.
  Experiments use the per-net nMSE = |est - truth|^2 / V(W), reported per net and as medians over nets; plain MC
  scores 1/N in expectation. The two are different averages, so each claim says which one it uses. Dead nets
  (V(W) = 0) are excluded by the census (section 2.2).
- Cost unit: one forward pass of one input with schoolbook products (2 n^2 L FLOPs, flopscope rates).
- Deterministic = no random bits. Sample-free (ARC's working test) = the estimate is not an average of f over a point
  set. Certified = outputs an interval that contains mu.

## 2. Progress prong: a certified branch-and-bound over the whole net

### 2.1 The method (`lib/bnb.py`)
- **Cells.** R^n splits into the 2n cones over the faces of the cube. A cell on face (k, s) is the cone over a box of
  gnomonic coordinates: {t v(y) : t > 0, y in [a, b]}, with v(y)_k = s and v(y)_i = y_i otherwise. Because the net is
  positively homogeneous, f is linear on each cone that lies inside one activation region.
- **Exact cell moments.** Conditioning on t = |x_k|, a cell's Gaussian mass and first moment are one-dimensional
  integrals of products of normal-cdf differences (`cell_moments`). They are computed with 32-96 Gauss-Legendre
  nodes in t; tested relative error <= 1e-9 against 400 nodes and against MC (`tests/test_bnb.py`).
- **Bounds from the actual weights.** Forward symbolic linear bounds in the cell's slice coordinates
  (ReluVal / Neurify style), propagated through every layer with the real downstream weights: Z_hi = W H_mid + |W| H_rad,
  Z_lo = W H_mid - |W| H_rad, chord upper relaxation, 0-or-identity lower relaxation. If every unit of every layer is
  certified stable on the cell, the bounds are the exact linear map. By homogeneity a slice bound c . (y, 1) becomes
  the x-linear bound c_free . x_free + c_const s x_k on the whole cone, so its Gaussian integral is exact.
- **Estimates.** On certified cells, the exact integral. Elsewhere: (a) "point": the local linear map at the cell
  centre applied to the cell's first moment (exact if the cell lies in the centre's region; costs about two forward
  passes); (b) "mid": the midpoint of the integrated bounds (uses only the bounds). The certificate is the sum of the
  integrated bounds; its half-width bounds the error of "mid" (not of "point" [A]).
- **Refinement.** Split the cell with the largest certified gap (primary arm; the secondary arm uses mass x summed
  unstable-unit range fractions, a backward-sensitivity proxy) at the midpoint of the coordinate that explains most of
  its unstable units' ranges.
- **What the certificate is.** Valid in exact arithmetic (checked by astra, round 2); in this implementation it does
  not enclose the moment quadrature error (relative <= 1e-9 tested) or float rounding, so it is "certified up to
  quadrature and rounding". A rigorous version needs outward-rounded moment intervals and tail bounds [A].
- **Is it deductive?** The bounds and "mid" never evaluate f at a point: deductive in ARC's sense. "point" evaluates the
  net's gates at cell centres: a deterministic adaptive quadrature, i.e. structured sampling at chosen points.
- **Cost charged** (flopscope rates): bounds about 2-4 n^3 L per cell (2 n^3 L once a cell is exact so far), cell
  moments about 300 q (n - 1) (q = 32-96 nodes, cdf 96 and pdf 54 FLOPs), the centre pass about 4 n^2 L. In
  forward-pass units (2 n^2 L) the bounds cost n to 2n passes and the moments 150 q / (n L) passes, so a cell costs
  46-54 passes at depth 32 and 128-170 at depth 8 for widths 3-8 (dev runs; the moments dominate at small width and
  depth, the bounds at large width). Two undercounts found by astra (|W| H_rad computed for all cells in a batch but
  charged only for inexact ones; four ndtr arrays evaluated, two charged) cost at most a factor 2; section 2.2 checks
  the wins against it.

### 2.2 Results against MC, RB-MC and RQMC at equal FLOPs (MEASURED, pre-registered)

Setup: 72 census nets (3 healthy seeds per configuration, seeds from 5000; healthy rates in section 6). Truth: 8
scrambled-Sobol radius-exact replicates at 2^23 points (2^21 for widths 16-32); truth errors 1e-16 to 1e-9 in nMSE.
RQMC: scrambled Sobol, radius exact, 8-32 replicates per N up to 2^22 (2^20 for widths 16-32). Every comparison uses
the B&B record's own cost. Medians over the 3 nets, primary arm (gap priority, midpoint splits); full tables for both
arms in `results/eval_tables.md`:

| n | L | B&B point-estimate nMSE at 2^16 / 2^20 / 2^22 passes | RQMC : B&B at 2^16 / 2^20 / 2^22 | RB-MC : B&B at 2^22 | exact mass | certified nMSE (final) |
|---|---|---|---|---|---|---|
| 3 | 4 | 6.6e-06 / 2.1e-10 / 1.5e-12* | 2.0e-04 / 0.019 / 2.9* | 1.1e+05* | 0.93 | 9.0e-11 |
| 3 | 8 | 1.1e-06 / 6.4e-11 / 3.5e-13* | 4.9e-04 / 0.29 / 2.6* | 5.3e+05* | 0.97 | 4.8e-11 |
| 3 | 16 | 6.6e-07 / 3.3e-10 / 9.3e-13 | 3.0e-03 / 0.068 / 1.2 | 1.6e+05 | 0.97 | 3.3e-10 |
| 3 | 32 | 2.3e-08 / 9.6e-13 / 6.1e-15* | 0.095 / 2.7 / 13* | 2.6e+07* | 0.99 | 8.9e-09 |
| 4 | 4 | 9.6e-05 / 4.5e-07 / 2.7e-08 | 1.2e-04 / 7.6e-04 / 8.5e-04 | 7.3 | 0.40 | 7.4e-05 |
| 4 | 8 | 8.2e-04 / 1.6e-06 / 2.4e-07 | 1.1e-04 / 5.3e-04 / 6.7e-03 | 0.83 | 0.36 | 4.6e-04 |
| 4 | 16 | 1.0e-04 / 1.1e-06 / 1.7e-07 | 2.5e-04 / 1.6e-04 / 1.3e-04 | 1.1 | 0.52 | 9.6e+03 |
| 4 | 32 | 3.5e-05 / 8.8e-07 / 7.3e-08 | 1.4e-03 / 7.7e-04 / 6.5e-04 | 2.7 | 0.39 | 3.5e+06 |
| 6 | 4 | 8.2e-03 / 8.2e-04 / 3.1e-04 | 1.7e-05 / 4.3e-06 / 1.5e-06 | 6.5e-04 | 0.00 | 0.87 |
| 6 | 8 | 8.6e-03 / 9.2e-04 / 3.0e-04 | 7.8e-06 / 1.6e-06 / 8.8e-07 | 6.3e-04 | 0.00 | 8 |
| 6 | 16 | 4.0e-03 / 4.4e-04 / 1.1e-04 | 1.7e-04 / 2.9e-05 / 2.8e-05 | 1.9e-03 | 0.00 | 7.9e+05 |
| 6 | 32 | 1.5e-03 / 1.3e-04 / 4.0e-05 | 1.4e-04 / 3.9e-05 / 2.9e-05 | 5.1e-03 | 0.00 | 7.5e+10 |
| 6 | 64 | 3.9e-03 / 5.6e-04 / 1.4e-04 | 5.4e-05 / 7.1e-06 / 4.0e-06 | 1.4e-03 | 0.00 | 1.9e+24 |
| 8 | 4 | 0.025 / 7.4e-03 / 3.5e-03 | 7.9e-06 / 1.2e-06 / - | 6.0e-05 | 0.00 | 5.5 |
| 8 | 8 | 0.023 / 6.2e-03 / 2.6e-03 | 5.6e-06 / 1.6e-06 / 5.5e-07 | 7.9e-05 | 0.00 | 4.5e+02 |
| 8 | 16 | 0.023 / 3.9e-03 / 1.4e-03 | 3.2e-05 / 2.1e-06 / 2.8e-06 | 1.5e-04 | 0.00 | 4.5e+06 |
| 8 | 32 | 0.013 / 4.7e-03 / 2.0e-03 | 6.6e-05 / 1.1e-05 / 6.6e-06 | 1.0e-04 | 0.00 | 1.3e+16 |
| 8 | 64 | 0.017 / 4.5e-03 / 2.3e-03 | 1.5e-05 / 3.4e-06 / 1.2e-06 | 8.6e-05 | 0.00 | 2.6e+34 |
| 16 | 8 | 0.24 / 0.13 / 0.088 | 5.6e-06 / - / - | 2.4e-06 | 0.00 | 6.4e+05 |
| 16 | 16 | 0.1 / 0.045 / 0.032 | 2.9e-05 / 2.8e-06 / - | 7.0e-06 | 0.00 | 4.4e+12 |
| 16 | 32 | 0.22 / 0.12 / 0.091 | 4.3e-06 / 4.1e-07 / - | 2.4e-06 | 0.00 | 2.9e+24 |
| 32 | 8 | 0.67 / 0.5 / 0.43 | 3.3e-06 / - / - | 5.3e-07 | 0.00 | 6.4e+08 |
| 32 | 16 | 0.58 / 0.45 / 0.39 | 8.7e-06 / 5.9e-07 / - | 5.8e-07 | 0.00 | 1.1e+17 |
| 32 | 32 | 0.82 / 0.66 / 0.56 | 5.3e-06 / - / - | 4.0e-07 | 0.00 | 1.9e+34 |

- **Widths 16 and 32** (P4: hit under a substituted rule, 18 of 18 nets; see section 6). At 4e6 passes the B&B's nMSE
  is 0.03-0.09 at width 16 and 0.39-0.56 at width 32, about what 2-30 Monte Carlo samples give. RQMC was run only to
  2^20 at these widths; at the largest B&B checkpoints with an RQMC value (2.6e5-1e6 passes) RQMC is 10^5-10^6 times
  better. No cell is ever certified, and the certificate's nMSE is 6e5 to 2e34.
- **The deductive estimate.** The table shows the centre-point estimate. The midpoint of the certified bounds, the
  part that never evaluates the net at a point, never beats RQMC: its best resolved RQMC ratio over all runs and
  budgets is 0.18 (width 3, depth 8). Its median nMSE at 4e6 passes is 8e-12 to 3e-8 at width 3, 2e-5 to 2e7 at width
  4, and 0.5 to 4e34 at widths 6-32 (the midpoint of vacuous bounds). Entries marked * include a net whose B&B error is
  below ten times the truth's own error (registered resolution rule): those ratios are not claimed.

Reading the table:
- **Width 3.** The B&B certifies 93-99% of the Gaussian mass as exactly integrated, and its final certificate has
  nMSE 2e-11 to 5e-8 at depth 4-8 and 5e-13 to 6e-8 at depth 16-32 (P5: hit on 6 of 6 nets at depth <= 8). Against
  RQMC it reaches parity only at 1e6-4e6 passes, where
  both are near nMSE 1e-12 and most of the B&B's errors fall below ten times the truth's own error, so they are not
  resolved. Resolved wins of 3x or more: 4 of 24 runs (both arms), all at depth 16-32 (3.8x, 5x, 11x and 12-18x at
  2.7e5-4.2e6 passes). At depth 4-8, none (P1, registered for depth <= 8: refuted). Deep width-3 nets are easier because
  they collapse onto few gate patterns (14-32 per net at every depth, section 3.5). All of this is for the noiseless
  mean. At sigma = 1e-3 the noise moves the mean of these width-3 nets by nMSE 9e-10 to 1e-4 (post hoc, paired MC),
  far more than the 1e-12 where the B&B catches up with RQMC, so in ARC's noisy setting the width-3 results say
  nothing about the target there.
- **Width 4.** 33x to 2e5x worse than RQMC at every scored budget (P2: hit, 12 of 12 nets); at 4e6 passes it is
  between slightly worse than and 7x better than radius-exact MC.
- **Widths 6-8.** 10^4-10^6 times worse than RQMC and 10^3-10^4 times worse than radius-exact MC at every budget
  (P3 and P8: hit, 30 of 30 nets). No cell is ever certified exact (exact mass below 1%), and the certificate is
  vacuous: its nMSE grows from about 1 at depth 4 to 1e16-1e34 at depth 32-64, because symbolic bounds through
  unstable units compound with depth.
- **Monotone in width** (P6: hit): the median nMSE at 2^20 passes is 6e-11, 2e-6, 9e-4, 6e-3 at widths 3, 4, 6, 8
  (depth 8) and 3e-10, 1e-6, 4e-4, 4e-3 (depth 16).
- **The secondary arm** (backward-sensitivity priority) is within a factor of a few of the primary everywhere and
  better at width 3 and depth 16-32 (above).
- **Sensitivity to the FLOP undercounts astra found** (post hoc): if every B&B cost were doubled (the most those
  undercounts can amount to), RQMC gets twice the budget, and at width 3 it improves fast with budget: of the five
  resolved wins of 3x or more, one stays above 3x (12x becomes 5.6x), three fall to 0.4-2.3x, and one cannot be
  checked (RQMC was not run beyond 2^22). The width-3 wins are therefore fragile; the losses elsewhere only grow.
- **Deterministic Sobol points** (P7, an empirical prediction whose rationale was wrong [A]): the centred,
  unscrambled Sobol set had 0.30, 0.22 and 0.37 times the nMSE of scrambled Sobol (geometric means over the 54 nets of
  widths 3-8, at N = 2^10, 2^14, 2^18): it was 2.7-4.5x better, outside the registered band [1/3, 3]. So a fixed
  deterministic point set did better than its randomised version here; it is still sampling at fixed points.
### 2.3 Cost against width and depth, and what could scale

Forward passes the primary B&B needed to reach a given nMSE, against RQMC (medians over the nets that reached it
within 8.4e6 passes; "-" = none did; post hoc table, `results/eval_tables.md`):

| n | L | B&B to 1e-3 | B&B to 1e-4 | B&B to 1e-6 | RQMC to 1e-3 | RQMC to 1e-4 | RQMC to 1e-6 |
|---|---|---|---|---|---|---|---|
| 3 | 4 | 5.3e+03 (3/3) | 3.6e+04 (3/3) | 2e+05 (3/3) | 64 (3/3) | 1.3e+02 (3/3) | 2e+03 (3/3) |
| 3 | 8 | 6.2e+03 (3/3) | 1.3e+04 (3/3) | 9.7e+04 (3/3) | 64 (3/3) | 2.6e+02 (3/3) | 2e+03 (3/3) |
| 3 | 16 | 7e+03 (3/3) | 1.4e+04 (3/3) | 8.4e+04 (3/3) | 64 (3/3) | 2.6e+02 (3/3) | 4.1e+03 (3/3) |
| 3 | 32 | 2.1e+03 (3/3) | 1.2e+04 (3/3) | 4.1e+04 (3/3) | 64 (3/3) | 1.3e+02 (3/3) | 2e+03 (3/3) |
| 4 | 4 | 1.4e+04 (3/3) | 1.3e+05 (3/3) | 1.1e+06 (3/3) | 1.3e+02 (3/3) | 5.1e+02 (3/3) | 8.2e+03 (3/3) |
| 4 | 8 | 3.4e+04 (3/3) | 2.8e+05 (3/3) | 2.1e+06 (3/3) | 1.3e+02 (3/3) | 5.1e+02 (3/3) | 8.2e+03 (3/3) |
| 4 | 16 | 1.9e+04 (3/3) | 1.4e+05 (3/3) | 2.1e+06 (3/3) | 1.3e+02 (3/3) | 5.1e+02 (3/3) | 8.2e+03 (3/3) |
| 4 | 32 | 2.3e+04 (3/3) | 3.5e+04 (3/3) | 1.1e+06 (3/3) | 1.3e+02 (3/3) | 5.1e+02 (3/3) | 1.6e+04 (3/3) |
| 6 | 4 | 1.1e+06 (3/3) | - (0/3) | - (0/3) | 2.6e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 6 | 8 | 1.1e+06 (3/3) | 4.2e+06 (1/3) | - (0/3) | 2.6e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 6 | 16 | 2.8e+05 (3/3) | 8.4e+06 (3/3) | - (0/3) | 2.6e+02 (3/3) | 2e+03 (3/3) | 6.6e+04 (3/3) |
| 6 | 32 | 2.7e+05 (3/3) | 2.1e+06 (2/3) | - (0/3) | 2.6e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 6 | 64 | 1.1e+06 (3/3) | 8.4e+06 (3/3) | - (0/3) | 1.3e+02 (3/3) | 1e+03 (3/3) | 1.6e+04 (3/3) |
| 8 | 4 | - (0/3) | - (0/3) | - (0/3) | 2.6e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 8 | 8 | 4.2e+06 (1/3) | - (0/3) | - (0/3) | 1.3e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 8 | 16 | 8.4e+06 (2/3) | - (0/3) | - (0/3) | 1.3e+02 (3/3) | 1e+03 (3/3) | 6.6e+04 (3/3) |
| 8 | 32 | - (0/3) | - (0/3) | - (0/3) | 2.6e+02 (3/3) | 2e+03 (3/3) | 6.6e+04 (3/3) |
| 8 | 64 | - (0/3) | - (0/3) | - (0/3) | 1.3e+02 (3/3) | 1e+03 (3/3) | 3.3e+04 (3/3) |
| 16 | 8 | - (0/3) | - (0/3) | - (0/3) | 2.6e+02 (3/3) | 2e+03 (3/3) | 6.6e+04 (3/3) |
| 16 | 16 | - (0/3) | - (0/3) | - (0/3) | 5.1e+02 (3/3) | 4.1e+03 (3/3) | 2.6e+05 (3/3) |
| 16 | 32 | - (0/3) | - (0/3) | - (0/3) | 5.1e+02 (3/3) | 4.1e+03 (3/3) | 1.3e+05 (3/3) |
| 32 | 8 | - (0/3) | - (0/3) | - (0/3) | 5.1e+02 (3/3) | 4.1e+03 (3/3) | 2.6e+05 (3/3) |
| 32 | 16 | - (0/3) | - (0/3) | - (0/3) | 5.1e+02 (3/3) | 4.1e+03 (3/3) | 2.6e+05 (3/3) |
| 32 | 32 | - (0/3) | - (0/3) | - (0/3) | 1e+03 (3/3) | 8.2e+03 (3/3) | 2.6e+05 (3/3) |

- The B&B's cost to nMSE 1e-4 grows by roughly 10x per unit of width from 3 to 6 (1e4-4e4 passes at width 3,
  4e4-3e5 at width 4, 2e6-8e6 at width 6) and is out of reach (> 8.4e6) at width 8 and above, while RQMC needs
  130-8,200 passes throughout. Depth matters much less than width at these sizes. This is the curse of dimensionality the
  partition analysis predicts (3.5, 3.7): cells must resolve the kink geometry in n - 1 dimensions, and a cell costs
  46-170 forward passes.
- The certificate degrades exponentially with depth wherever units stay unstable on a cell (its nMSE reaches 1e16-1e34
  at widths 6-8 and depth 32-64), so certification is practical only where cells can be made exactly linear, which in
  our runs means width 3.

How each deterministic family's cost grows ("gain" is against the best sampler at equal cost):

| Family | Deductive in ARC's sense? | Cost growth | Gain measured or proven | 10M+ parameters? |
|---|---|---|---|---|
| Certified B&B over input cones (this lane) | bounds and "mid": yes; "point": no | a cell costs 46-170 passes (n^3 L FLOPs for bounds); passes to nMSE 1e-4 grow about 10x per unit of width (table above) | centre-point estimate only at width 3, 4 of 24 runs, fragile under a 2x cost change; the deductive estimate never | no (extrapolated from widths 3-32; untested at scale) |
| Exact enumeration of linear cones (3.5) | yes | number of cones: conjecturally at most (C n L)^{n-1}, exp(O(n log n)) at L = alpha n; measured 14-32 gate patterns at width 3, flat in depth | exact, so it wins as eps -> 0 at fixed n | no (conditional on the count; untested at scale) |
| Multilevel reference-trajectory sampler (astra, 3.7) | no (structured sampling; deterministic via 3.1) | passes, as MC; prefix levels cost r/L of a pass | grows with depth (estimated ideal gains): about break-even at L <= n/4, 1.5x at L = n = 128, 15x at width 8 and depth 256; proven Omega(n / log^2 n) at L = alpha n under stated cost hypotheses | cost scales like MC; the estimated gain needs depth about n or more at the widths tested, or very large width (untested at scale) |
| Fixed designs: unscrambled Sobol, spherical designs, Kerdock sets | no (evaluates f at chosen points) | one forward pass per point; constructions exist at any width | bounded: 2-design ceiling against RB-MC at L = n/4 falls from 7 (width 16) to about 2 (widths 64-256), 1.7 at 256 x 32 (heuristic ceiling, measured inputs); about 4x for positive designs in the mean-field limit [A]; Kerdock 2.4x over plain MC at 256 x 32 (arc-noisy lane) | yes in cost, with a constant gain only (extrapolation) |
| Cumulant closures, K = 2 and factorised K = 3 (ARC; hardness lane 6.3) | yes | n^{K+1} per layer | K = 2: 2-8x over plain MC at widths 32-128, about 1 at 1024 (depth n/8 to n/2); K = 3 falls behind as width grows | no growing gain |
| Gate-pattern mixtures (narrow-deep-m1 lane) | yes | pieces grow exponentially with width | error floor about 1e-3 at width 8 | no |
| Polynomials of degree D in the weights (3.5) | yes | M^D coefficients in a dense representation | degree 4-8 worth a few MC samples at depth 16-32 (plug-in estimates) | no (extrapolation) |

## 3. Difficulties prong

### 3.1 On random weights, "deterministic" is no restriction (in an idealised arithmetic model)
- PROVEN UNDER exact-transform primitives [A] (real arithmetic with unit-cost inverse cdfs). QR-reduce three rows of
  W_1, B_1 = W_1 Q. Then mu(B) = mu(W), and the 3n - 6 angular coordinates removed are uniform and independent of B
  (Bartlett decomposition of Gaussian rows). Their conditional cdfs give uniforms U, V in [0, 1)^n, and
  X_i = Phi^{-1}((U + i V) mod 1) are pairwise independent N(0, I_n), because (u, v) -> (u + i v, u + j v) mod 1
  preserves Haar measure on the torus for i != j. The average of f_B over X_1..X_N has risk exactly V/N. Cost: N
  forward passes plus O(n^2 + n N). This needs n >= 6 (3n - 6 >= 2n); smaller widths use radius-exact spherical
  sampling (n >= 4) or the first layer's Frobenius radius as an extra seed (n = 2, 3) [A].
- PROVEN for Gaussian weights at each fixed size, with floor and inverse-cdf primitives [C, integrability step A].
  Seeds frac(2^k w) from a few weight entries become uniform and independent of W as k grows (Riemann-Lebesgue), and
  the wrapped density of the Gaussian combinations used is bounded uniformly in k, so the risk tends to V/N. The same
  works for the noise variables of noisy nets. For a general weight density the weak convergence holds but the risk
  convergence needs an integrability assumption: the review gave a density for which it fails.
- PROVEN UNDER digit-splitting and exact-transform primitives [C, after an objection from A]: any randomized
  estimator can be simulated deterministically at the same average risk. From the free directions draw a Haar
  rotation and, from other digits, the estimator's random bits; rotating the canonical first layer gives an instance
  with the law of W and the same mean, independent of those bits. Extra cost O(n^3) plus the randomness access. With
  finitely many bits per weight an exact simulation is impossible (it would need H(W', R) = H(W) + H(R) > H(W)); only
  approximate versions, like the digit seeds, remain.
- Threshold circuits (only partly). For iid symmetric nonzero weights on uniform +-1 inputs, flipping an input and
  its weight column preserves the acceptance probability, so canonicalising the signs of one row frees n fair bits
  independent of the canonical circuit (PROVEN [C]). A pairwise-independent sample of n-bit inputs needs 2n bits, and
  a free permutation does not supply fair bits exactly, so a deterministic match of MC for random threshold circuits
  is not established here.
- Consequences [C+A]. In the idealised model, determinism excludes nothing in ARC's random-MLP instance, uniformly and
  explicitly; this is typically-correct derandomisation (Goldreich-Wigderson 2002) made exact by the ensemble's
  symmetry. Neither continuity in W (a fixed design is continuous in W [A]) nor algebraic operations only (iterated
  Chebyshev polynomials give rapidly oscillating seeds [C, corrected by A]) separates deduction from sampling. A useful
  restriction must be representational, for example an auditable language with no weight-derived seeds [A], or a
  bounded interface (the sibling lane). The brief's question therefore becomes: is MC, implemented as well as
  possible, optimal among all algorithms on average over random critical nets?

### 3.2 The baseline must be the best sampler (PROVEN) [A]
A batch of N >= n inputs through an n x n layer costs O(ceil(N/n) n^omega) with fast matrix multiplication instead of
2 N n^2, so batched MC beats serial MC by n^{3 - log2 7} = n^{0.19} (Strassen) at equal arithmetic cost. "Nothing
beats serial MC by more than a constant" is false. A conjecture must compare with the best implementation of
sampling, or fix a cost model in which a matrix-vector product costs n^2. Every experiment here uses the latter.

### 3.3 Why no unrestricted impossibility proof is in reach
- PROVEN (logical) [C+A]. Fix the precision of the weights, an accuracy schedule and a RAM cost model. Then "every
  algorithm of cost T has risk at least c times that of the best sampler at cost T", for budgets of M^{1+delta}
  operations (M = n^2 L weights), is a super-linear time lower bound for an explicit distributional problem. It would
  not by itself imply P != NP or a circuit lower bound [A].
- HEURISTIC [C, narrowed by the review]. I know of no technique that proves a lower bound of this kind for a natural
  problem like this one (time-hierarchy arguments give super-linear bounds only for diagonal problems). This is a
  statement about available techniques, not a theorem.
- PROVEN (logical). Worst-case barriers do not transfer. IKW (deterministic estimation within +-1/12 for all nets in
  time 2^{E^{o(1)}} would give NEXP not in P/poly) and the #NSETH bound (deterministic needs time E^{2 - o(1)}) are
  worst-case results (hardness lane, Cor. 4.2 and 4.3). On random weights derandomisation is free in the idealised
  model (3.1).
- PROVEN (logical) [C; narrowed by A and the review]. The polynomial-interpolation route to worst-to-average
  reductions (Lipton; Ball-Rosen-Sabin-Vasudevan; Gamarnik-Kizildag for the SK partition function) needs an estimand
  that is a low-degree polynomial in the instance. mu is not one: it is positively 1-homogeneous in each layer, and
  already mu = |w| / sqrt(2 pi) at L = 1. Other random self-reductions exist (discrete logarithm is one), so this
  excludes one route, not all.
- PROVEN (logical) [A]. Natural proofs would need a large, constructive property of truth tables; Gaussian measure on
  weights is not that largeness condition, so the barrier does not apply as such.

### 3.4 Where the known average-case hard problems sit
- PROVEN [C+A]. With Z(beta) = E exp(beta f), the mean is (log Z)'(0): the infinite-temperature end.
- The average-case hardness results I know concern exact computation or relative precision. Exact computation of the
  SK partition function is hard on average unless P = #P (Gamarnik-Kizildag 2021, verified abstract). Exact output
  probabilities of random quantum circuits and exact permanents of Gaussian matrices are #P-hard on average; their
  approximate versions are conjectures (from memory). For convex-body volume, randomised algorithms succeed
  (Dyer-Frieze-Kannan) and deterministic oracle algorithms fail, a worst-case information bound (from memory). None of
  these is a hardness result for additive precision at the standard-deviation scale, which MC reaches in polynomial
  time (hardness lane Prop. 3.1), so none applies to ARC's mean as stated.
- PROVEN [A, corrected by the review]. Moving between free-energy error and partition-function error loses a factor
  proportional to the coefficient of variation, exponential only when that is exponential.
- PROVEN [A]. Tails: under ARC's SD-normalised metric, zero beats N-sample MC on average over nets exactly when
  E[p^2] < E[p (1 - p)] / N. Tail hardness has to be stated for relative error.

### 3.5 Restricted classes: what provably fails, and by how much
- **Fixed point sets** (deterministic QMC, designs).
  - PROVEN [C+A]. The risk of any fixed linear rule is the kernel discrepancy
    sum_{ij} c_i c_j k(u_i . u_j) - 2 m_n q sum_i c_i + m_n^2 q, with k(t) = E_W <g(u), g(v)> and q = E k(U . V). A fixed rule
    and its random rotation have the same risk. This does not equate a Sobol set with scrambled Sobol, since
    scrambling changes the geometry [A].
  - PROVEN [A]. k(t) = sum_d b_d t^d with b_d >= 0 (Mehler expansion through the first layer). For positive weights
    summing to one, the gain over RB-MC is at most 2 (1 - q) / sum_{s : tau_{2s} <= 1/(2N)} b_{2s}, where
    tau_{2s} = E (U . V)^{2s} = (2s - 1)!! / (n (n + 2) ... (n + 2s - 2)). For N <= n^p only even degrees above about 2p
    count. Designs gain at most the inverse of the high even-degree mass of the kernel.
  - PROVEN in the infinite-width mean-field limit (infinite width first, then depth) [A]: at L = alpha n that even tail
    carries half the angular variance, so the bound tends to about 4x.
  - MEASURED (registered M2, `scratch/harmonic_profile.py`, 2^18 directions per net, 3 nets per width, L = n/4). The
    fraction f1 + f2 of the angular variance in harmonic degrees 1 and 2, and the heuristic ceiling 1/(1 - f1 - f2) of
    a 2-design that is MC-like above degree 2 (median over nets): width 16: 0.76-0.93, ceiling 7.0; width 32:
    0.62-0.86, 2.8; width 64: 0.45-0.61, 2.0; width 128: 0.34-0.53, 1.9; width 256: 0.50-0.58, 2.2. At ARC's 256 x 32:
    0.30-0.43, ceiling 1.7 against radius-exact MC (consistent with the 2.4x a Kerdock design gave there against plain
    MC). The low-degree share, and with it any design's room, falls as width and depth grow together.
  - MEASURED, post hoc (`scratch/kernel_even.py`, 3 nets per size, ratio of averages): kernel values k(-1), k(0) and
    the degree-2 energy give the even power-series mass above degree 2 as a fraction of the angular variance: 0.03,
    0.11, 0.20, 0.13, 0.24 at (16, 4), (32, 8), (64, 16), (128, 32), (256, 64), and 0.48 at ARC's 256 x 32. Through
    astra's bound this caps the gain of any positive-weight design with at most n(n+2)/6 points (about 11,000 at width
    256) over RB-MC at 64, 19, 10, 15, 8.4 and 4.1. These are plug-in estimates, not bounds: they rest on 3 nets,
    sampled outputs and noisy integrals, and at ARC's 256 x 32 the estimated degree-2 energy a_2 came out negative
    (-0.0023), so the step b_2 <= n a_2/(n-1) [A] cannot be applied to it as a bound. The fraction drifts toward the
    mean-field 1/2, so the plug-in cap near ARC's setting is near the limiting value of about 4.
  - PROVEN UNDER the Gaussian-process limit at fixed n and L, two-sided spectral bounds (with the parity
    degeneracies handled) and an applicable cubature theorem [C; hypotheses not checked]: the best N-point rule has
    squared error of order N^{-1 - 3/(n-1)}, a gain of order N^{3/(n-1)} (N^{0.43} at n = 8, N^{0.012} at n = 256).
- **Polynomials of degree D in the weights** (instance-independent coefficients).
  - PROVEN. The best risk is the Hermite tail T_D = sum_{k > D} E_k of mu ([C] planned before astra's run; [A]
    independently). Noise stability S(rho) = Cov(mu(W), mu(rho W + sqrt(1 - rho^2) W')) = sum_k rho^k E_k ([C] and [A])
    bounds it from below and above; the explicit bounds are astra's (I wrote my one-sided version after seeing its
    progress line): T_D >= [(1 + rho^{-D}) F(rho) - rho^{-D} F(rho^2)]_+ with F = S(1) - S, and T_D <= F(rho)/(1 - rho^{D+1}).
  - PROVEN in the mean-field limit [A, proving my heuristic]. S_L(rho) = m_n^2 (E_T K_{rho,L}(T) - 1/pi), with
    K_{rho,0}(t) = t and K_{rho,l+1} = kappa(rho K_{rho,l}). At fixed depth the decorrelation is linear in delta = 1 - rho,
    with slope of order L; for L delta^{1/3} -> infinity, 1 - Corr ~ 3.27 delta^{2/3}, so T_D = Theta(S(1) D^{-2/3}) for
    1 << D << L^3.
  - MEASURED (registered M1: 48 net pairs per configuration, 2^14 Sobol points per net). At n = 32 the measured
    decorrelation matches the mean-field formula within about 5% at L = 4 for every delta from 1e-5 to 0.3. At L = 16
    and 32, after dividing by the measured value at delta = 1 (post hoc; it should be 1 and is 1.6), it matches within
    -21% to +29% for delta <= 1e-2 (`logs/nstab_mf_compare.log`). Finite-difference proxies of the mean Hermite
    degree, D(1e-5)/1e-5: 2.2 at L = 4, 7 at L = 16, 17 at L = 32 (27 before the division).
  - Plug-in estimates of the class ceiling (registered estimator, against plain MC, relative standard errors of the
    inputs 20-60%): at n = 32, degree-4 polynomials are worth at most about 2.3 (L = 16) and 1.2 (L = 32) MC samples,
    degree-8 about 14.5 and 5.7. The post-hoc normalised estimator gives 6.5 and 3.9 for degree 4 and 17 for degree 8
    at L = 32. At n = 16 and L >= 16 rare high-energy nets dominate the second moments and the estimates are
    unresolved. These are estimates, not established bounds.
- **Stable estimators.** PROVEN [C+A]. 2 |A - mu| >= |mu - mu_rho| - |A - A_rho| in L^2, so an accurate estimator must be
  at least as rough in W as mu, which the previous item quantifies. Overlap-gap arguments do not apply as such: the
  acceptable estimates for one net form a ball, a connected convex set [A].
- **Statistical queries.** PROVEN [C+A]. Bounded scalar queries need one query per output coordinate and a truncation
  of the unbounded ReLU outputs, but with arbitrary expectation queries the mean, or a tail, is answered directly. SQ
  lower bounds count queries, so they say nothing unless queries are restricted or charged.
- **Cell selection.** PROVEN [A]. Any K first-layer gate cells, chosen after reading W, cover expected mass at most
  sqrt(K kappa_n), with kappa_n = E_T (1/2 + arcsin(T)/pi)^n exponentially small. A method that integrates selected
  cells exactly and drops the rest has risk of order n at L = alpha n for polynomial K.
- **Partition cubature exact on linear cells.** PROVEN UNDER a mesh hypothesis [A]; these are upper bounds, not
  limits on all partition methods. M cells give error C_W M^{-2/(n-1)} with coherent cell errors; with centred,
  uncorrelated cell errors (unproved for our algorithm) the squared error is M^{-1 - 3/(n-1)} [A; my v1 exponent was
  wrong]. The exponent's advantage over MC, 3/(n-1), vanishes with width; the constants grow with depth.
- **Exact enumeration of linear cones.** PROVEN UNDER a region-count bound and an integration-cost hypothesis [C]:
  if E R <= (C n L)^{n-1} and each fixed-dimensional cone integral costs polylog(1/eps), exact integration at fixed width
  costs a polynomial in L. HEURISTIC: it then beats MC once eps is below about (C n L)^{-(n-1)/2}. The bound on R is
  an extra hypothesis: Hanin and Rolnick prove a count of order (number of neurons)^(input dimension) for nets with
  random biases (abstract verified), not this bias-free spherical version. At L = alpha n it is exp(O(n log n)).
  MEASURED (registered M4, `scratch/regions.py`): healthy width-3 nets show only 14-32 distinct gate patterns on 2^22
  Fibonacci-sphere directions, with medians 22, 23, 18, 15 at depth 4, 8, 16, 32: no growth with depth (P13, slope
  in [1, 3], refuted). Narrow critical nets that stay alive collapse onto few gate histories, so the count saturates
  far below the hypothesised bound; these are lower bounds and describe the healthy subpopulation (3-60% of seeds).

### 3.6 What any estimator would need (PROVEN necessary conditions) [A]
- Correlation. R(A) >= S (1 - r^2) + |E A - E mu|^2, where S is of order n at L = alpha n and r is the correlation of A
  with mu across nets. A gain G over N-sample MC needs r^2 >= 1 - V / (N G S).
- Final row. relu(z) = (z + |z|)/2 makes the odd part of mu_j in the last row w_j exactly (1/2) w_j . m_h, with
  m_h = E_x h_{L-1}. So R(A) >= (1/(2n)) sum_j E |mhat_j - m_h|^2 with mhat_j = n E_{w_j}[w_j A_j]: an accurate
  estimator carries the whole penultimate mean vector in its linear response to the last layer.
- Depth recursion. With B = W_L ... W_{r+1}, E[B^T mu_L | W_1..W_r] = mu_r. Any A therefore induces an estimator of the
  depth-r mean with risk at most 2^{L-r} R(A). This is informative only a few layers back, and the induced estimator
  need not be cheap.
- Control variates. A = E g + mean(f - g) has risk E tr Var(f - g) / M; a growing gain needs a residual variance that
  shrinks after charging the construction and integration of g.
- Limitation [A]. None of these forces an estimator to enumerate regions, propagate a summary or have high degree.
  The missing link from computational cost to such a restriction is the bottleneck (section 5).

### 3.7 Red-team result: plain MC is not optimal at depth proportional to width
- **Construction** [A, round 3]. Run one reference direction Q through the net, keeping its normalised path q_l and
  the per-layer norms nu_l; let s_r be the product of nu_l over the layers after r. For an input direction U, the
  control G_r(U) = m_n s_r |h_r(U)| q_L runs U through the first r layers only and then treats it as merged with the
  reference. The telescoping sum mu = G_0 + sum_j E_U(G_{r_j} - G_{r_{j-1}}) + E_U(Y - G_L), Y(U) = m_n h_L(U), with
  independent samples at each level, is unbiased.
- **Why it works** (PROVEN UNDER layerwise batching costs and m(cT) = O(m(T)), which schoolbook costs satisfy [A]).
  Input directions merge with depth (the cosine deficit 1 - kappa^{(l)}(0) decays like 9 pi^2 / (2 l^2)), while the
  ensemble-average angular variance V_sph stays of order 1 at L = alpha n because of amplitude variation set in the
  first layers. Astra
  proves E |Y - G_r|^2 <= (4 m_n^2 / n) sum_{l >= r} delta_l + 2 m_n^2 delta_L = O(1/r + n/L^2), so dyadic levels give
  risk O(log^2 L / (L N)) at the cost of N full passes: a gain of order n / log^2 n over radius-exact MC at
  L = alpha n. I checked the telescoping identity and unbiasedness numerically (`tests/test_mlmc.py`) and the decay of
  the residual qualitatively (below), not astra's two one-layer inequalities behind the bound.
- **Estimated ideal gains, not end-to-end speedups** (post hoc, `scratch/mlmc_check.py`, `logs/mlmc_check.log`, run
  12:40-12:50Z; 4 fresh nets per cell, seeds 9100-9103,
  32 seeds at width 8 and depth 256; nets with zero angular variance skipped; 4,096 directions per net; ideal
  variance x cost relative to RB-MC, so below 1 is a gain). Two-level version; the prefix depth is chosen as the best
  dyadic one on the same 4,096 directions that estimate the variances, so the figures are optimistic by that
  selection, and they exclude the setup pass, the pilot, integer allocation and lower-order work:

  | width n | L = n/8 | L = n/4 | L = n | L = 4n | L = 16n | L = 32n |
  |---|---|---|---|---|---|---|
  | 8 | - | - | - | 0.37 | - | 0.065 (12 live nets of 32 seeds) |
  | 16 | - | 1.29 | 1.04 | 0.33 | 0.14 | - |
  | 32 | - | 1.11 | 0.94 | 0.60 | - | - |
  | 64 | - | 1.07 | 0.89 | 0.34 | - | - |
  | 128 | - | 0.96 | 0.66 | - | - | - |
  | 256 | 1.01 (ARC's 256 x 32) | 0.91 | - | - | - | - |

  The full dyadic multilevel version is worse than two levels at these sizes (1.1-2.2 at L <= n). The residual
  fraction D_r / V_sph falls with the prefix depth as the bound says (1.17, 0.73, 0.33, 0.06, 0.02 at r = 0, 4, 16, 64,
  128 for n = L = 128). Astra's own pilot at L = n/4 found no gain up to n = 512 either.
- **Consequences.** C0 as I stated it is false, asymptotically, under astra's layerwise cost hypotheses and against
  radius-exact MC with the best allowed batching. With the symmetric derandomisation of 3.1 (idealised arithmetic)
  the estimator becomes deterministic, so an unrestricted deterministic estimator does beat plain sampling by a factor
  that grows at depth proportional to width; it does so by better sampling, not by deduction. At the sizes tested
  the estimated gain is about break-even at L <= n/4 (0.91-1.29, including ARC's 256 x 32 at 1.01), up to 1.5x at
  L = n, and larger at L >= 4n. Nothing here shows it beats RQMC, and the finite-precision version of the
  derandomisation is not analysed. The baseline for ARC's question at depth should be this structured sampler, and
  matching it without sampling is the open question. At width 8 and depth 256, the narrow-deep-m1 setting, the
  estimated two-level gain over radius-exact MC is about 15x (noiseless; median over the 12 of 32 seeds whose nets had
  positive empirical angular variance). That lane's sampled-prefix arm beat RQMC 4.6-6x at small budgets but carried a
  bias, which multilevel corrections remove; the two lanes' baselines and nets differ, so no ranking between this
  sampler and RQMC follows. All of this is for sigma = 0.
- **Reuse.** The sampler is implemented, documented (inputs, cost model, rate) and tested in `lib/mlmc.py`
  (`tests/test_mlmc.py`); see section 8.
- **Partition certificates** [A, round 3]. Bounded aspect ratio alone gives no lower bound (a hinge can be aligned
  with the cells). Under explicit nonalignment, isolation and capture hypotheses, holding uniformly in dimension, a
  certificate of expected summed-coordinate width eps sqrt(V) needs K >~ (n^3 / eps)^{(n-2)/2} cells, using the exact
  expected kink area Theta(n^{3/2} L) of the regular kink set; for Euclidean width (the metric of our certificates)
  n^3 becomes n^{5/2}. Not checked numerically; the hypotheses are unproved for coarse cells.

## 4. Which known random object is this most like? (Kelvin's question)

For each analogy: how close it is, what is known, what a proof of similarity would need, and my verdict. My
assessment is in `notes/analogies.md`; astra's (round 2, section 3(e)) reaches the same verdicts in weaker words
("no reduction supplied"), which I adopt.

### 4.1 Derandomising threshold circuits (CAPP for TC0)
- **Close in the worst case (PROVEN, hardness lane Thm 4.1).** On Boolean inputs, a threshold gate with integer
  weights and threshold is relu(S - theta + 1) - relu(S - theta), and steep ramps binarise Gaussian inputs. So
  constant-depth threshold circuits of size s become ReLU nets of constant depth (with constant overhead) and size
  O(s), whose Gaussian mean is the acceptance probability up to O(m/K). Deterministic estimation for constant-depth
  ReLU nets would therefore give deterministic CAPP for TC0. The mean is also an integral of counting problems:
  E relu(z) = int_0^inf P(z > t) dt.
- **What is known** (titles verified; theorem details from memory). Unconditional pseudorandom generators and
  deterministic approximate counting exist for halfspaces (Gopalan-Kane-Meka 2015, seed nearly logarithmic), low-degree
  PTFs (Meka-Zuckerman; Kane) and intersections of few halfspaces (Harsha-Klivans-Meka; O'Donnell-Servedio-Tan). For
  polynomial-size depth-2 threshold circuits nothing comparable is known; even lower bounds are about n^{1.5} gates and
  n^{2.5} wires (Kane-Williams, STOC 2016), and Williams' algorithms-to-lower-bounds framework ties non-trivial CAPP
  algorithms for a class to lower bounds against it.
- **Where the known depth-1 theory runs out for ReLU nets (PROVEN for the bound; the conclusion is about known
  techniques).** One ReLU layer has a closed-form Gaussian mean. A second layer on k first-layer units splits input
  space into at most 2^k regions, and each output-threshold event there is an intersection of at most k + 1
  halfspaces, so fooling each to error eta gives 2^k eta [A]. That is useful for few units and useless at k = n. I know
  of no other route for width-n nets at depth 2.
- **Two calibrations.** Derandomisation asks for deterministic polynomial time at constant eps; ARC asks to match
  sampling at every eps. The known PRG-based counting algorithms for one halfspace cost (n/eps)^{O((log log(n/eps))^2)}
  evaluations, more than sampling's O(1/eps^2) as eps -> 0 (an upper bound on what they cost, not a lower bound for
  deterministic methods). And by Beck's discrepancy theorem no N-point set has error below about N^{-1/2 - 1/(2(n-1))}
  on every spherical cap, so against a rich class a fixed point set can at best match sampling. The random ensembles
  also behave differently [A]: in the infinite-width limit, for 0 < c < 1, sign gates contract correlations
  ((2/pi) arcsin c < c) while ReLU expands them (kappa(c) > c).
- **On random weights the analogy's core question disappears** (3.1, in the idealised model): determinism is free
  there, so the barrier that makes TC0 derandomisation hard cannot appear; what remains is efficiency, which the TC0
  literature does not address. I found no study of average-case CAPP for random threshold circuits under an
  SD-normalised error (astra: none verified).
- **Proof of similarity would need** a reduction from worst-case depth-2 TC0 CAPP to random-weight mean estimation that
  preserves the instance law, the precision, the loss normalisation and the cost advantage. The embedding needs integer
  weights with steep ramps, far from He weights; and the hardness lane's planting result (Prop. 3.2) only bounds how
  undetectable planting can change the distribution of the mean, so it neither supplies nor excludes such a
  reduction. **Verdict: no transfer is established and I see no route to one.** The analogy is accurate worst-case
  context (deterministic estimation of depth >= 2 ReLU nets sits at the TC0 frontier) and a useful calibration.

### 4.2 Convex-body volume
- **What is known** (from memory unless marked). Randomised polynomial-time relative approximation for membership
  oracles (Dyer-Frieze-Kannan 1991). Deterministic oracle algorithms cannot approximate within exponential factors
  (Elekes 1986; Barany-Furedi 1987). Exact volume is #P-hard for explicit H- and V-polytopes. For explicit H-polytopes,
  deterministic polynomial-time approximation is open in general; the known deterministic FPTAS/QPTAS for structured
  polytopes use correlation decay or zero-freeness (Gamarnik-Smedira, arXiv 2312.03906; Bencs-Regts 2024; arXiv
  2409.07283; abstracts verified). promise-BPP = promise-P would derandomise explicit finite-input problems such as
  explicit polytope volume, not oracle access.
- **The ReLU analogue exists in both models.** White box (PROVEN [A]): mu(W) = sum_s B_s E[X 1{X in C_s}], a sum of
  Gaussian first moments of explicit polyhedral cones (each cut out by at most n L inequalities; up to 2^{nL} of
  them), and tail probabilities are sums of Gaussian volumes. It is a piecewise-linear integrand, not one convex body.
  Black box (PROVEN, hardness lane Thms 2.1 and 2.3, worst case over explicit families): no query algorithm beats MC by
  more than a small constant (about 4 on the uniform-input family; the constants differ for other variants [A]), and a
  deterministic query algorithm loses a factor of order its number of queries, because mass can be hidden where it
  does not look. That is the ReLU form of Elekes and Barany-Furedi, with the same limitation: reading the weights
  exposes what queries cannot see [A].
- **A link (HEURISTIC).** The known deterministic volume algorithms for structured polytopes use correlation decay. The
  ReLU counterpart is contraction of backward value functions across depth, which makes layer-local errors
  depth-free (hardness lane Thm 6.12). At criticality the scale mode provably never contracts (hardness lane
  Prop. 6.7). That does not exclude methods that handle the scale analytically.
- **Proof of similarity would need** a white-box lower bound for deterministic approximation of Gaussian moments of
  explicit cones, converted to SD-normalised mean accuracy. At polynomial scale that would contradict promise-BPP =
  promise-P for an explicit problem; the #NSETH result of the hardness lane concerns worst-case nets, not the volume of
  one explicit cone; and on random weights determinism is free in the idealised model, which does not rule out
  hardness shared by randomised and deterministic estimators. **Verdict: the similarity is proven in the oracle model;
  for white-box or average-case versions no reduction is established.**

### 4.3 Spin glasses (for comparison)
- The mean is the infinite-temperature end (3.4). Tail probabilities at large thresholds are governed by large
  deviations, i.e. by log Z at large beta, which is where the low-temperature results live.
- COMPUTED. For the infinite-width ReLU kernel kappa_L of L layers, (kappa_L'')^{-1/2} shows no concavity violation on
  my grid of (0, 1) at the depths I checked up to 8 (1, 2, 4, 6, 8), and is not concave at L = 9: astra proved that at
  q = 0.4 by outward-rounded interval arithmetic; my scans show violations at every depth tested from 10 to 64. With
  the external field h^2 = kappa_L'(0), the relevant interval is [qhat, 1) with h^2 + xi'(qhat) = qhat xi''(qhat)
  (formula recalled, unverified). My post-hoc scan found violations there from L = 10, and the review's interval
  computation found one at L = 9, q = 0.5, inside the interval (qhat = 0.4926). So the condition fails at depth 9 in
  both forms and at every depth tested above it; depths 3, 5 and 7 were not checked. My registered P12 (concave at
  every depth) is refuted.
- HEURISTIC reading, a testable statement about a Gaussian surrogate only. In spherical spin glasses, concavity of
  (xi'')^{-1/2} is the condition under which Lipschitz algorithms reach the maximum (Subag; Huang-Sellke; recalled,
  unverified beyond the abstract). Its failure
  suggests an overlap gap for maximising the surrogate field at depth 9 and above. It is not a result about the actual net:
  the ReLU output field is not Gaussian; the kernel's second derivative is infinite at q = 1, outside the smoothness
  assumptions; Huang-Sellke's hardness theorem is stated for even mixtures (abstract verified) while this mixture has
  odd terms, including the field; and failure of a sufficient condition does not itself prove a gap. Whether ARC's
  1e-12 tails (7-13 SD at n = 256) sit below the algorithmic threshold would need that threshold evaluated for this
  kernel, which I have not done.
- **Verdict:** spin glasses give the only precise, testable observation found here, about a Gaussian surrogate of deep
  ReLU nets and near-maximal outputs. It says nothing about means.

## 5. Conjectures and the bottleneck (merged ledger with astra)

**C0, the conjecture I set out to make, is false** [A's M-whole; refuted by astra's own red-team, 3.7].
- Statement as made precise after review A. Cost model: arithmetic on real numbers with unit-cost normal cdf and
  inverse cdf (or b = O(log n)-bit floats), expected cost over W, all preprocessing charged. m(T) = the largest number
  of radius-exact spherical samples whose generation and full forward evaluation, with the best batching the model
  allows, cost at most T. Claim: for fixed alpha > 0 and 0 < a < b there is c = c(alpha, a, b) > 0 such that every
  uniform estimator family A_n of expected cost T_n in [n^a T_fwd, n^b T_fwd] has R(A_n) >= c V_sph / m(T_n) at
  L = alpha n.
- Refutation: the multilevel reference-trajectory sampler of 3.7 has gain Omega(n / log^2 n) under schoolbook-type
  costs (PROVEN UNDER the stated cost hypotheses [A]); estimated ideal two-level gains up to 1.5x at L = n and 7x at
  L = 16n (post hoc), about break-even at L = n/4 up to width 256.

**C0\*, THE BOTTLENECK (revised).** Same model and window; the baseline is now the best multilevel
reference-trajectory sampler B\*(T) (any prefix depths and sample allocations). Claim: no uniform estimator family has
R(A_n) <= B\*(T_n) / G_n with G_n growing polynomially in n at L = alpha n.
- Status: CONJECTURE with little evidence: nothing tested here or in earlier lanes beats the plain samplers, let alone
  this one, but the multilevel sampler itself shows how unsafe such conjectures are. No implication from a standard
  hypothesis is established, and a proof would face the barrier of 3.3.
- Cheapest refutation: an estimator with a polynomial gain over B\*: for example a better control for the first
  layers' amplitude variation (which carries the Theta(1) variance the reference trajectory cannot remove), or several
  reference trajectories.
- Why it is the bottleneck: any honest "cannot beat sampling" statement for unrestricted algorithms needs a theorem
  connecting the arithmetic cost of an arbitrary white-box computation to an unavoidable residual error in mu(W), and
  the baseline must already exploit the net's depth collapse. Kernel tails, Hermite degree, gate-cell counts and the
  recursive mean extraction each constrain one representation or one observable; none supplies that bridge [C+A].
- ARC's own question (a sample-free estimator matching sampling) is untouched by the refutation; at depth it now
  means matching B\*, not plain MC.
- Relation to the sibling lane (cut-rank-ext, finished today): once a bounded interface may carry real numbers or
  nonlinear readouts, its precision-power separations disappear and only running time is left. C0\* is that
  running-time question for whole-network estimators.

| Conjecture | Status and evidence | Cheapest refutation |
|---|---|---|
| **C1 (weight-noise sensitivity at finite width).** For L = alpha n and delta in [C L^{-3}, delta_0], 1 - Corr(mu(W), mu(W_rho)) = Theta(delta^{2/3}) uniformly in n. | PROVEN in the mean-field limit [A]. MEASURED agreement with mean field at n = 32, L <= 32, only in the linear regime (delta <= 1e-2 lies below this window at these depths), so the 2/3 law itself is untested at finite width. | Unbiased two-scale stability measurements at larger L showing another exponent. |
| **C3, M-design.** For each p, sum_{s > p} b_{2s} >= c_p (1 - q) at L = alpha n, so positive designs with N <= n^p have bounded gains (3.5). | PROVEN in mean field (fraction 1/2, gain bound about 4) [A]; CONJECTURE at finite width. MEASURED (post hoc, 3 nets per size): the fraction is 0.03-0.24 at widths 16-256 with L = n/4 and 0.48 at 256 x 32, drifting toward 1/2. | Kernel values k(-1), k(0), b_2 across n showing the even tail vanishing. |
| **C2 (partition methods).** Distinguish three classes: methods that discard unselected cells (PROVEN to fail, 3.5); methods that cover a fixed mass with fully gate-resolved cells (PROVEN to need exponentially many cells [A]); methods with downstream-informed certificates on coarser cells (CONJECTURE: exp(c n) cells at fixed normalised precision; PROVEN UNDER explicit nonalignment and capture hypotheses, 3.7). | MEASURED for our B&B: passes to nMSE 1e-4 grow about 10x per unit of width from 3 to 6 and exceed 8.4e6 from width 8 (section 2.3). | A polynomial cell count with enclosed integration error at fixed normalised precision. |
| **Region count.** E R <= (C n L)^{n-1} for bias-free nets. | Extra hypothesis (sampled pattern counts are lower bounds only). MEASURED at n = 3: 14-32 patterns, flat in depth 4-32, far below the bound (healthy nets only). | Verified counts above that scaling. |
| **T-relative (tails)** [A r1]. A computable threshold sequence t_n with P_W(e^{-bn} <= p_W(t_n) <= e^{-an}) >= 0.9 on which no polynomial-time algorithm gets relative error 1/2 with conditional success probability >= 2/3. | CONJECTURE; no distribution-preserving reduction; the spin-glass observation (4.3) concerns near-maximal outputs of a surrogate. | Radius integration plus importance sampling with polynomial cost and bounded relative variance. |

## 6. Pre-registered scorecard and deviations

Scored by `scratch/analyze_eval.py` and `scratch/analyze_theory.py` (outputs in `results/`).

| Prediction (PREREG) | Outcome |
|---|---|
| P1. n = 3, L in {4, 8}: B&B beats RQMC by >= 3x at some budget in [2^16, 2^22] on >= 4 of 6 nets | **Refuted**: 0 of 6 (parity at best at 1e6 passes; the larger apparent wins at 4e6 passes are below the truth's resolution). Resolved wins appear only at depth 16-32 (4 of 24 runs, 3.8-18x), outside the registered range. |
| P2. n = 4: >= 10x worse than RQMC at every budget in [2^12, 2^22] on >= 9 of 12 nets | **Hit**: 12 of 12 (10^3-10^4 worse). |
| P3. n in {6, 8}: >= 100x worse than RQMC at every budget in [2^14, 2^22] on all 30 nets | **Hit on the available comparisons**: 30 of 30 (10^4-10^6 worse); the 2^22 comparison is missing for the three 8 x 4 nets (their 2^22 record overshoots RQMC's largest N by more than 1%). |
| P4. n in {16, 32}: >= 1000x worse than RQMC at the largest budget reached, certificate nMSE >= 1 throughout, all 18 nets | **Hit under a substituted rule**: all 18 nets reached 2^22 passes, but RQMC was run only to 2^20 there, so the comparison uses the largest checkpoint with an RQMC value (2^18-2^20): 10^5-10^6 worse; certificate nMSE 6e5 to 2e34 throughout. Under the registered rule it cannot be scored. |
| P5. certificate nMSE <= 1e-6 on >= 4 of 6 nets at n = 3, L <= 8; on none at n >= 6 | **Hit**: 6 of 6 at width 3 (final certified nMSE 2e-11 to 5e-8), none of 30 at widths 6-8. |
| P6. median B&B nMSE at 2^20 increases with n (3, 4, 6, 8) at L = 8 and 16, by >= 100x | **Hit**: 6e-11, 2e-6, 9e-4, 6e-3 (L = 8); 3e-10, 1e-6, 4e-4, 4e-3 (L = 16). |
| P7. geometric mean of QMC-det / RQMC nMSE in [1/3, 3] at N = 2^10, 2^14, 2^18 (its stated rationale was wrong [A]; scored as an empirical prediction) | **Refuted**: 0.30, 0.22, 0.37; the centred deterministic Sobol set was better than scrambled Sobol by 2.7-4.5x. |
| P8. n in {6, 8}: B&B loses to RB-MC at every budget on >= 2/3 of nets | **Hit**: 30 of 30. |
| P9. secondary arm against primary (descriptive) | Within a factor of a few everywhere; better at width 3 and depth 16-32. |
| P10a. noise-stability slope over delta in [1e-4, 1e-2] in [0.5, 0.8] at L = 16, 32 | **Refuted**: 0.84-1.00 on all four. The mean-field formula (found afterwards) predicts 0.76-0.94 in that window: the 2/3 regime starts later than I guessed. |
| P10b. slope over [1e-5, 1e-3] >= 0.8 at L = 4 | **Hit**: 1.00 and 1.00 (n = 16, 32). |
| P10c. D(0.1) >= 0.25 and degree-4 tail bound >= 0.1 at L = 16, 32 | **Hit as registered** (D(0.1) = 0.44-1.82, tail 0.14-10.3), but the n = 16, L = 16 values exceed 1, which is impossible for the population quantity: that cell is dominated by rare nets and is not valid evidence. |
| P11. f1 + f2 in [0.3, 0.95] on every net, and the median 2-design ceiling changes by < 2x from n = 16 to n = 256 (L = n/4) | **Refuted** on the second clause: the first holds (0.30-0.93), but the median ceiling fell from 7.0 to 2.2, a factor 3.2. The measured quantity is f1 + f2 over the first 16 output coordinates (the registered code computes degree 2 on at most 16 outputs), which is all outputs only at width 16. |
| P12. (kappa_L'')^{-1/2} concave on (0, 1) for every L | **Refuted** from L = 9 (registered check: concave at L = 1, 2, 4, 8; not at 16, 32, 64). |
| P13. region-count log-log slope in L at n = 3 in [1, 3] | **Refuted**: the counts do not grow (medians 22, 23, 18, 15 at depth 4-32); healthy narrow nets collapse. |
| M5. noise effect at sigma = 1e-3 (descriptive) | nMSE of (noisy mean - noiseless mean), paired MC with 2^20 inputs: widths 6-8: 6e-12 to 7e-7 at depth 4-8, 7e-8 to 1e-4 at depth 16-64 (one net with tiny variance: 3e-3); widths 16-32: below 4e-8 at every depth. So at widths 6-8 and depth >= 16 any noiseless deterministic estimate is biased at the level RQMC reaches with 10^4-10^6 passes; at widths 16-32 and depth <= 32 the noise does not matter at any budget we ran. Post hoc at widths 3-4: 6e-10 to 1e-4 (up to 0.4 on one width-4 net whose output variance is tiny). |

**Deviations and post-hoc analyses.**
- Amendment 1 (11:29Z, before any evaluation result): secondary B&B arm at n <= 8 only; three jobs at a time. Its
  header time was first mistyped as 11:36Z and rehashed with the right time (`prereg_hashes.txt`).
- 11:39Z: the queue was restarted with the theory jobs interleaved; three in-flight truth jobs were killed and rerun
  from scratch. No result was seen or discarded.
- Analysis choices fixed before any B&B evaluation result: comparisons use each B&B record's own cost (slightly above
  2^k because of batch granularity) for every baseline, and RQMC interpolation clamps a 1% overshoot of its largest N.
  Comparisons beyond RQMC's largest N are dropped rather than scored; this affects P3 (the 8 x 4 nets at 2^22) and P4
  (scored at 2^18-2^20), as stated in the table.
- The registered resolution rule (an nMSE counts only if at least ten times the truth's own error) is applied in
  scoring P1 and marked with * in the results table; P7 includes a few unresolved deterministic-QMC cells at 2^18
  (for example two of the 6 x 64 nets) and stays refuted on its resolved 2^10 and 2^14 values.
- Post hoc: the external-field spin-glass scan; the mean-field comparison of M1 and its normalisation by D(1); the
  kernel test of M-design (`scratch/kernel_even.py`); the noise effect at widths 3-4 (`scratch/noise_effect_posthoc.py`);
  the cost-to-precision table; the factor-2 FLOP sensitivity of the width-3 wins; the dev-only kink-split arm; the
  midpoint-estimate statistics of 2.2; the multilevel-sampler measurements of 3.7 (`scratch/mlmc_check.py`, declared
  in PREREG.md at 12:40Z; run 12:40-12:47Z on 13 configurations with seeds 9100-9103, then at 12:48-12:50Z on 256 x 32,
  8 x 32 and 8 x 256, the last with seeds 9100-9131 because most width-8 nets are dead at depth 256).
- The census conditions the ensemble on healthy nets (as registered). Healthy rates were 60% at width 3 and depth 4-8,
  11% and 3% at depth 16 and 32; 12-100% at width 4; 43-100% at widths 6-8; 100% at widths 16-32. Results at small width
  and large depth describe that conditional population [review].

## 7. Process

**Co-prover (gpt-6-astra, xhigh effort, read-only; `coprover/PROVENANCE.md`).**
- Round 1 (11:01-11:22Z): an independent attempt from the questions alone. New from astra: the exact symmetric
  derandomisation, the fast-matrix-multiplication counterexample to a serial-MC conjecture, the kernel power series and
  the positive-design bound, the cell-selection bound, the necessary conditions (correlation, final row), and
  conjecture M-whole. My v1 (frozen before its answer arrived) had the digit-seed derandomisation, the kernel-energy
  identity, the noise-stability identity and degree heuristic, the spin-glass concavity test, the conditional exact
  cost and the barrier statements. Disclosure: I read astra's last progress lines about 4 minutes before finishing
  v1; its fast-matmul point entered v1 from there.
- Round 2 (11:32-11:52Z): a line-by-line audit of my v1 and of `lib/bnb.py`, plus the analogies. Nine corrections
  accepted (listed in PROVENANCE.md): among them my Chebyshev remark (wrong), the partition-rate exponent (wrong), the
  simulation gap in "deterministic = all algorithms" (repaired by me), the barrier wording (relabelled HEURISTIC), the
  spin-glass hypotheses, and the certificate's quadrature caveat. New from astra: the mean-field 2/3 law for the
  mean (which I then checked against our measurements), the two-scale Hermite bound, the even-tail identity and
  estimators, the depth-recursive extraction.

**Reviews.**
- GPT-6 review A (fresh session, high effort, 12:00-12:10Z; `review/review_gpt6_A.md`), on sections 1 and 3-5:
  verdict "major revision", 16 findings, all applied (`FORMAL.md` section 9 lists them). The main ones: separate the
  theory's average risk from the per-net nMSE; label the deterministic simulation PROVEN UNDER its primitives (and
  note it is impossible with finitely many bits); restrict the digit-seed result to Gaussian weights (the review gave
  a counterexample density); withdraw the threshold-circuit match (n free bits are too few); correct my mean-field
  agreement figures (5% only at L = 4; -21% to +29% deeper) and call the degree-D numbers plug-in estimates; the
  spin-glass condition fails at depth 9 in both forms, and the statement is about a Gaussian surrogate only; narrow
  the barrier wording (time hierarchies; discrete log; IKW's 2^{E^{o(1)}}); fix "sampling fails" for volume (randomised
  algorithms succeed); soften the analogy verdicts to "no transfer established"; fix the provenance of the degree
  bound; make C0 and the other conjectures precise.
- GPT-6 review B (fresh session, high effort, 14:15-14:21Z; `review/review_gpt6_B.md`), on the whole report:
  verdict "major revision", 11 findings, all applied. The main ones: the tables showed the centre-point estimate,
  which samples, while the summary called the method deductive; the deductive midpoint estimate never beats RQMC, and
  the summary and verdict now say so. The 4.1x design cap at 256 x 32 was a plug-in estimate (its a_2 estimate was
  negative), not a bound. P3 and P4 were scored with dropped or substituted budgets, now declared. Unresolved table
  entries are marked. The multilevel figures are estimated ideal gains with prefix selection on the same data, and
  the prose now matches the table. The partition bound's norm is stated. Several ranges were corrected (width-4
  losses 33x-2e5x; RQMC to 1e-4 up to 8,200 passes). P9 and P11's measured quantity are stated. The reusable
  allocator could exceed its budget; fixed and tested (`tests/test_mlmc.py`). The spin-glass and tail wording was
  narrowed. The reviewer reproduced seven headline numbers from the raw files and the factor-2 sensitivity.

**Sources and how they were checked.** Verified this session from abstracts or bibliographic records:
Goldreich-Wigderson 2002 and Hoza 2019 (typically-correct derandomisation), Hanin-Rolnick 2019 (abstract only),
Gamarnik-Kizildag 2021, Beck 1984 and Brauchart-Dick 2013 (discrepancy, from search snippets), Ritter 2000 (table of
contents), Gopalan-Kane-Meka 2015 and Kane-Williams 2016 (records), Gamarnik-Smedira and the 2024 structured-volume
papers (abstracts), Huang-Sellke 2021 (abstract: hardness for even mixtures). Read in full by the hardness lane:
IKW 2002, Chen-Tell 2021, Wu et al. 2026. From memory, with theorem hypotheses not checked: the PRG and fooling
results for halfspaces, PTFs and polytopes, Dyer-Frieze-Kannan, Elekes, Barany-Furedi, the random-circuit and
permanent results, Subag's full-RSB criterion and the Huang-Sellke field formula, Bietti-Bach's spectrum, and the
Sobolev cubature rates. The co-prover's citations are marked unverified in its answers.

**Compute.** CPU only, nice 19 and idle IO, at most 3 heavy processes, single-threaded BLAS; no GPU. One slip: a
post-hoc job runner started early because its wait condition matched nothing (11:52Z) and then matched my own log
line (11:54Z); both times I killed its jobs by PID within a minute and trashed their partial outputs (`logs/queue.log`).

**Deviations.** Listed with the scorecard in section 6.

## 8. Handoff

**Reusable multilevel shared-trajectory sampler** (for the stationary-regime lane and others): `lib/mlmc.py` in
`work/unrestricted/`. Its module docstring names the inputs (weights W of a bias-free ReLU net, a reference
direction or an rng, the prefix depths `levels`, the sample counts per level), the cost model (setup 1 pass; a level
sample costs r_j / L passes, the last level 1 pass; optimal allocation N_j ~ sqrt(V_j / C_j)), the proven rate and the
measured gains. Functions: `ReferencePath`, `level_samples`, `pilot_level_stats`, `allocate`, `mlmc_estimate`. Tests:
`python3 tests/test_mlmc.py` (telescoping identity exact; unbiased against an RQMC truth, max |z| = 2.5 over 16
outputs). The variance-times-cost measurements are `scratch/mlmc_check.py` (`logs/mlmc_check.log`).

**Other code.** `lib/bnb.py` (certified branch-and-bound) with `scratch/run_bnb.py`; `lib/nets.py` (nets, samplers,
FLOP conventions); `scratch/truth_baselines.py` (truth, RQMC, deterministic Sobol); `scratch/noise_stability.py`,
`scratch/harmonic_profile.py`, `scratch/kernel_even.py`, `scratch/spin_glass*.py`, `scratch/regions.py`,
`scratch/noise_effect*.py`; scoring in `scratch/analyze_eval.py` and `scratch/analyze_theory.py`. Tests:
`python3 tests/test_bnb.py`, `python3 tests/test_mlmc.py`. Queue: `scratch/run_queue.sh` with `scratch/jobs_eval2.txt`.

**Next steps, if wanted.**
1. An asymptotic-rate improvement to the multilevel sampler (HEURISTIC). With prefix levels, the first level costs
   1/L of a pass per sample and carries the Theta(1) amplitude variance of the first layers, so this family's gain over
   RB-MC is at most of order L. That variance is a smooth function of the input direction: |h_1(U)|^2 is a quadratic-type
   function whose angular variance sits mostly in harmonic degrees 1-3. Integrating the first level with a spherical
   2-design, or using the exactly computable mean of |h_1(U)|^2 as a control variate, would remove that floor; the gain
   would then be limited by the angular-mismatch terms. Test: the level-1 variance after the control, against L.
2. Several reference trajectories (or per-cluster references) to shrink the final direction-mismatch term n / L^2,
   which is what keeps the gain near 1 at L <= n/4, ARC's regime.
3. RQMC within each level (a constant factor, likely 2-10x at small width), and the noisy case sigma = 1e-3 with the
   noise shared between the two terms of a level.
4. A fully charged run of the sampler (pilot and allocation included) on the registered nets against RQMC.
5. For the theory: the finite-width 2/3 window (C1) at larger depth, and the M-design kernel test (C3) at width 256.
