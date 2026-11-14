# Do the public counter-claims escape the ladder? The "scale mode" law (forum 18298) and "covariance propagation beats sampling in Phase 2" (18198), tested (2 November 2026)

Status (13:24Z): final for this lane's run; revised after a GPT-6 fresh-session review (major revision, 9 findings) and a follow-up check (minor revision, 6 leftovers), all applied (section 9).

Lane: scale-mode (target M1), bead pool-ttdzrk.15 under pool-ttdzrk. Work directory `work/scale-mode/`: `notes/SOURCES.md`
(the claims verbatim, the contest's specification, the public state of the art), `notes/AUDIT.md` (the third-party code,
read before anything ran), `PREREG.md` + `prereg_hashes.txt` (predictions, decision rule and frozen code, hashed
11:22:22Z before any evaluation run), `scripts/` (adapters, queues, the hashed scorer), `runs/` (evaluation),
`runs_dev/` (development), `results/` (score.json from the hashed scorer; tables.md, report_tables.md; explore_fs.txt),
`figs/` (fig1_256x32.png, fig2_gain_vs_width.png, fig3_1024x16.png), `logs/` (deviations.txt), `GPU-LOG.md`, `review/`.
Labels: MEASURED (registered unless marked exploratory), NUMERICAL, HEURISTIC, CONJECTURE.
Companion: arc-update (outbox/arc-update-2026-11-02.md; work/arc-update/).

## Summary for Kelvin

**No major positive. Neither public claim met the escape criteria registered in advance, so within what was tested
the negative stands, and is stronger for it. The "scale mode" estimator (forum 18298) is real and is the best
order-2-cost deductive estimator we have measured on the deep shapes (depth = width/8 and /4), but it buys a constant factor in a narrow budget window. Its
two-level ladder ends up as steep as ARC's at 256 x 32, and on its own depth = width/4 diagonal it beats RQMC at only
some widths. "Covariance propagation beats sampling in Phase 2" (18198) is ARC's own K2: true under the contest's
compute-adjusted score and at equal compute, not against full-budget sampling. Higher orders and larger widths remain
untested. No decision needed.**

1. **Tested as its author ran it.** We ran the author's code unchanged on CPU and on two short GPU runs. It reproduces
   his 64 x 16 headline (nMSE 1.517e-3 against his 1.52e-3) and, to his printed precision, his 1024 x 256 net 0. We
   registered 45 prediction rows (40 hold) and a decision rule for "escapes the ladder" before any evaluation run.
2. **Our nets, depth = width/8 (Phase 1's 256 x 32 and arc-update's ladder).** At 256 x 32 the estimator errs 3.6e-4 at
   688 passes. That is 2.5x better than RQMC at the same cost (95% CI 2.0-4.7 over nets and scrambles) and 2.1x better
   than the best ARC variant at or below its cost. RQMC reaches the same error at ~2.3x that cost, and at the Phase 1
   budget RQMC is 33-65x more accurate.
   Its order-3 level ("SL3", ARC's K3 with the scale removed) beats ARC's K3 on every net. It still loses to RQMC by
   4-20x at its own cost, and it costs more than the whole Phase 1 budget. The error-cost exponent between the two
   levels is 1.0, 0.72, 0.52, 0.30 at n = 32 to 256. That is steeper than ARC's at small widths (0.54, 0.37, 0.28) but
   the same at 256 x 32 (0.31), and below sampling's 1 from n = 64 on.
3. **The claim's own diagonal, depth = width/4 (8 fresh nets per width, 64 x 16 to 1024 x 256).** It beats naive Monte
   Carlo in 4 of 5 cells, as claimed (0.89 at 512 x 128, CI 0.57-1.63). Against RQMC, which is ~2x Monte Carlo on these
   nets, its median gain is 0.79, 0.73, 1.2, 0.43, 2.0. Only at 64 x 16 (a loss) and 1024 x 256 (a win) does an exploratory
   95% interval exclude 1; the cells between are undecided, and so is the trend with width (+0.19; -0.20 without the
   1024 cell). Its order-3 level
   barely improves on it (exponent ~0.1 at 128 and 256) and loses to RQMC by 10-100x. Error varies ~10x between nets of
   one shape.
4. **Phase 2's 1024 x 16 (12 public MLPs, 1e9-sample truths).** The baseline is 6x better than RQMC at its own cost.
   The contest's score multiplies error by the fraction of budget used (floored at 10%), so a cheap method wins it
   while being 6x worse than full-budget RQMC in raw error. Registered test N2 holds: full-budget RQMC beats every
   deductive method we ran within the budget, including ARC's best order-2 variant (by 1.06-1.9x). Beyond the budget the
   ladder is steep here (exponent 0.83): ARC's unengineered order 3 is 19x more accurate than full-budget RQMC at 1.5x
   the budget. The public leaderboard (external evidence) shows engineered order-3 chains doing ~100x better than MC
   within 12% of the budget, so in practice deduction wins at this wide, shallow shape. The negative concerns depth
   growing with width.
5. **For arc-update.** The claim is a deductive estimator related to arc-update's strongest lead: a Gaussian closure
   with a corrected amplitude law, here a Gamma scale fitted by a fourth-moment rule rather than the exact radius law.
   Its cost and error fall inside the budget window that arc-update's break-even arithmetic allowed (~2x over RQMC at
   1.75 order-2 steps per layer). A partial re-pricing of its special functions at the contest meter's rates
   (exploratory) doubles the cost and shrinks the window to 1.16x at 256 x 32.

## 1. The claims, exactly (notes/SOURCES.md has the verbatim text, dates and links)

**Claim (a), forum topic 18298** (pranav_viswanath; opened 28 Oct 23:57Z, two minutes before the Phase 2 write-up
deadline; 8 versions, the last 30 Oct 16:51Z): "when depth grows with width, a collective scale mode builds up that
poisons the order-by-order expansion. Separating it out exactly via ReLU homogeneity and carrying it as a one-parameter
law restores convergence and beats sampling at every width tested on across networks of various sizes (64x16 through
1024x256 ...)". What the repository (github.com/PranavViswanath/scale-separation, HEAD bfeab38) actually measures:
- the diagonal is L = n/4 (64x16 ... 1024x256 are width x depth), 10 nets per cell;
- "sampling" is naive Monte Carlo at equal FLOPs (ARC's counter plus 1 flop per ndtr), with the sample charged 2 n^2 (L+1)
  (an output layer the estimand does not use); antithetic and radial-exact MC are compared at 64x16 only, and no
  quasi-Monte Carlo design anywhere;
- the estimator ("LVTV2") is one level, at about ARC's order-2 cost: margin over naive MC 2.31, 1.57, 1.53, 1.42, 1.21 at
  64, 128, 256, 512, 1024; the higher levels of the "ladder with the scale law" converge but "sit above the sampling line
  at their FLOPs";
- since 30 Oct the repository itself says the margin narrows with width (exponent -0.20, 95% CI [-0.38, -0.04]), that
  the author's pre-registered width test failed, that at 1024x256 the margin's interval is [0.89, 1.83], and that "a
  single level of this cost would fall behind sampling at some larger width". The PDF attached to the forum post predates
  these sentences and is titled "Solving ...".
- at the contest's own 1024x16, the K2-cost estimator is 1.09-1.76x worse than ARC's AUGMENT K2 (its own C1-Q1); the
  1.25x and 1.44x gains are the scale hook inside ARC's factored K3, and the 1.73x leaderboard gain is inside a fitted
  engine.

**Claim (b), forum topic 18198** (keenanpepper, 24 Aug): six entries tied at "3.93e-7 adj / 3.93e-6 MSE are probably
all the plain & simple covariance propagation baseline. Which decisively beats sampling in Phase 2." The baseline is the
starter kit's example 03; "adj" multiplies the raw final-layer MSE by max(0.1, C/B), and the baseline uses 2.35% of the
budget, so its adjusted score is 0.1 x its raw MSE.

**Topic 18147** (the Phase 1 numbers in the brief: diagonal Gaussian closure 7.18e-4, full-covariance 9.61e-5,
"budget-matched sampling estimator ~2.5e-7" at 256 x 32) is a Phase 1 write-up whose own conclusion is that Gaussian
closures lose to a budget-matched sampler by ~400x at depth 32: it supports the negative and needs no separate test
(arc-update measured the exact Gaussian recursion at 256 x 32: nMSE 1.0-2.0e-3, i.e. raw MSE ~2e-5 to 2.4e-4).

## 2. The contest, its state of the art, and which phase our nets match (notes/SOURCES.md sections 2-4)
- Phase 1 = 256 x 32 (L/n = 1/8), 2.72e11 FLOPs = 64,850 forward passes; Phase 2 = 1024 x 16 (L/n = 1/64), 2^41 FLOPs =
  65,536 passes. Both: He weights N(0, 2/width) in every layer, no biases, no output layer, x ~ N(0, I), estimand the
  per-neuron means, 1e9-sample truths, score = final-layer MSE x max(0.1, C/B), FLOPs metered by flopscope (Phase 1:
  plus priced residual wall time; Phase 2: flopscope only).
- State of the art (public boards, not the prize ranking, which is due ~15 November): Phase 2 final-layer MSE 1.10e-8 at
  12% of the budget (adjusted 1.3e-9), from engineered deductive K3 chains; full-budget MC on these nets would score
  ~1.2e-6, so the leaders are ~100x below it. Phase 1: 3.63e-8 at 51% of the budget, and the organisers say Phase 1's best
  methods "tended to be based on variations of sampling". ARC's own published method is unchanged (arXiv:2605.05179 v2).
- Our five nets 200-204 are Phase 1's setting (shape, weight law, input law, estimand); we score nMSE = |est - Y|^2/tr Var
  with ARC's FLOP counter instead of raw MSE x max(0.1, C/B) with flopscope. arc-update's ladder (L = n/8) extends Phase 1's
  depth ratio. Phase 2 differs (L/n = 1/64), so it is measured separately here on its own public MLPs 0-11.

## 3. What the scale-mode estimator is, and the checks that the code runs as its author ran it
- The estimator carries, per layer, the mean, the covariance and one scalar V: the pre-activation is modelled as
  z = sqrt(S)(m + g) with g Gaussian and S ~ Gamma(mean 1, variance V). ReLU's homogeneity makes every moment a Gaussian
  moment times a closed-form E S^{k/2}; V is updated by matching the weight-averaged fourth moment of the norm. It is related
  to the lead arc-update called its strongest (a Gaussian closure with a corrected amplitude law), which arc-update
  measured only as a finite-sample recursion with the exact radius law; this estimator instead fits a Gamma scale by
  moments, so it is not the same closure. arc-update estimated that a deductive closure of that kind would beat MC at
  that lead's error only if it cost under ~5 order-2 steps per layer, and RQMC under ~2-3.5 (FORMAL E13). This one costs
  1.75 order-2 steps per layer at 256 x 32 by the claim's accounting (688 vs ARC k2's 393 passes; 3.4 when its ndtr and
  exp calls are re-priced at flopscope's rates, a partial re-pricing, not full flopscope metering).
- Untrusted code: read in full before running (notes/AUDIT.md); only `lvlib.py` and the de-scaling hook ever ran, from my
  own scripts, `python -I`, `env -i` (no secrets), nice 19, ionice -c3, CPU.
- Reproduction on the claim's own networks (rebuilt from its seeds with ARC's MLP class), my truths: its 64 x 16 headline
  (10 nets: nMSE 1.517e-3, margin over naive MC 2.32 in its accounting; the claim reports 1.52e-3 and 2.31) and its
  1024 x 256 net 0 (V_L = 0.0284 and 1.199e12 FLOPs; its log prints 0.0284 for that net, to four decimals, and 1.20e12 as
  the cell mean). The SL3 adapter with the hook off
  reproduces ARC's factored K3 to 6.7e-16 with identical FLOPs. So the numbers below are the claim's estimator, not a
  re-implementation.

## 4. Why separating the scale should not change the ladder's exponent (CONJECTURE, with its assumptions, and the measurement that tests it)
The claim's own theory (extensions/THEOREM.md, Theorem 3) describes the scale mode: for the scale law with zero mean, the
shared scale's standardized cumulants are (r-1)!(2r-1)!! V^{r-1}, the same for every readout and every width; for a vector
with independent coordinates read by a balanced readout, each coordinate's own non-Gaussianity is diluted by width as
n^{1-r}. If those two pictures carried over to the de-scaled states of a fixed-weight network (zero mean, independence
and balanced readouts are not established there, and neither result bounds the remaining error from below), removing
the scale would leave the diluted per-neuron part, the object ARC's wide-network expansion was built for, with
truncation error that ARC conjectures to be at most c_K (L/n)^K at ~n^{K-1} passes per order. If, further, that law were
tight as a lower bound, uniformly over the orders used (the hypothesis arc-update's section 2.5 needs and does not
establish), the best affordable error could not fall faster than C^-gamma with gamma = log(n/L)/log n: 0.375 at
256 x 32, 0.25 at 256 x 64, 0.20 at 1024 x 256, tending to 0 at L = alpha n. Under those assumptions the separation would
change the constants c_K (and remove the inversion that undiluted scale cumulants cause at L/n = 1/4), not gamma. None
of this is proven; it is why we registered two measurable escape routes (PREREG section 6): a two-level exponent
gamma_SL near 1 at n = 128 and 256 with SL3 beating RQMC (D1), or a fixed-cost advantage over RQMC above 1 at every
diagonal width and growing with width (D2). Failing them does not rule out escapes at higher orders, larger widths or by
other methods.

## 5. Our nets, L = n/8 (E1 = Phase 1's 256 x 32, nets 200-204; E2 = arc-update's ladder, 8 nets per width)
All numbers are medians over nets from the hashed scorer (results/score.json, results/tables.md T1); costs in forward
passes with ARC's counter (+1 per ndtr for the claim's arms, its own accounting); G = the competitor's error at the
method's own cost divided by the method's error (> 1 = the method wins); X_RQ = the cost multiple at which RQMC reaches
the method's error.

| n x L | LVTV2 nMSE @ cost | G_MC | G_RQ (95% CI, nets x scrambles; wins vs MC, RQMC) | X_RQ | SL3 nMSE @ cost | SL3 G_RQ | gamma_SL | gamma_ARC | law log(n/L)/log n |
|---|---|---|---|---|---|---|---|---|---|
| 32 x 4 | 7.2e-4 @ 200 | 7.0 | 1.8* (1.1-2.6; 8/8, 7/8) | 1.8* | 8.3e-5 @ 1.6e3 | 1.5 | 1.0 | 0.54 | 0.60 |
| 64 x 8 | 6.7e-4 @ 287 | 5.2 | 2.0 (1.7-2.7; 8/8, 8/8) | 2.0 | 7.2e-5 @ 6.1e3 | 0.73 | 0.72 | 0.37 | 0.50 |
| 128 x 16 | 6.0e-4 @ 426 | 4.0 | 2.4 (1.7-2.9; 8/8, 8/8) | 2.1 | 9.5e-5 @ 2.1e4 | 0.19* | 0.52 | 0.28 | 0.43 |
| 256 x 32 | 3.6e-4 @ 688 | 4.0 | 2.5 (2.0-4.7; 5/5, 5/5) | 2.3 | 7.2e-5 @ 7.5e4 | 0.067* | 0.30 | 0.31 | 0.375 |

(gamma_ARC is recomputed here per net on the same nets, k2a -> best order-3 variant; arc-update's medians-based values
are 0.52, 0.39, 0.23, 0.30. At 32 x 4 the method's cost, 200 passes, is below RQMC's smallest measured size, 256 points,
so G_RQ and X_RQ there use RQMC's curve extrapolated with slope -1; likewise SL3 at 128 x 16 and 256 x 32 costs
1.3x and 1.15x more than RQMC's largest measured size. Entries marked * use such an extrapolation. The CIs are an
exploratory bootstrap over nets and the 8 RQMC scrambles (results/explore_uncertainty.txt), whose mean error near these
costs has a relative SE of 7-46% per net.)
- **The order-2-cost level is the best deductive point we have measured at these shapes, and it is a window.** At 256 x 32
  LVTV2 errs 3.6e-4 (2.3x below ARC k2a's 8.4e-4, 1.35x below arc-update's finite-sample radial closure, 4.9e-4) at
  688 passes, beats RQMC at its own cost by 2.3-5.8x on 5 of 5 nets, and RQMC reaches its error at 2.1-3.4x its cost
  (median 2.3x, ~1,600 passes, 2.5% of the Phase 1 budget). At the budget (64,850 passes) RQMC is 33-65x more accurate
  than every scale-law method that fits in it (P2).
- **At fixed level, its gain over MC falls with width and its gain over RQMC rises only slowly.** Along L = n/8 the gain
  over MC falls (exponent -0.28, 95% CI [-0.49, -0.06]); the gain over RQMC rises slightly (+0.14, CI [+0.001, +0.55],
  four widths), because RQMC's own gain over MC falls with width here (~4-5x at n = 32, ~2x at 256), not because the
  estimator improves: its error falls only like n^-0.31 while its cost grows like n^0.59, and the precision it reaches is
  a floor at each width (~4-7e-4). With the contest meter's prices for its ndtr and exp calls (flopscope: 96 and 16
  flops per element; a partial, exploratory re-pricing of those calls only, scripts/explore_fs.py) its cost roughly doubles at 256 x 32 (688 -> 1,325 passes) and
  its median gain over RQMC falls from 2.5 to 1.16; at n = 32-128 it then loses to RQMC (0.47-0.86).
- **The scale-law ladder converges (SL3 below LVTV2 on every net, below ARC k3f on every net); its two-level exponent
  is steeper than ARC's at small widths and the same at 256 x 32.** SL3's error is flat in width (7-10e-5 at every n) while its cost grows like n^1.8 (1.6e3 to 7.5e4 passes; above
  the Phase 1 budget at 256 x 32); its gain over RQMC falls from 1.5 to 0.067. The exponent between the two levels falls
  with width, 1.0, 0.72, 0.52, 0.30, against ARC's 0.54, 0.37, 0.28, 0.31 on the same nets: steeper at n <= 128, equal at
  256 x 32 and below the uniform law's 0.375 there. These are slopes between two implemented levels, not all-order
  exponents; the order-4 level (SL4) was not run on our nets.

## 6. The claim's own diagonal, L = n/4 (E3: fresh nets of our weight stream, 8 per width)
1024 x 256, the 128 x 32 nets 611-617, the 256 x 64 cell and four of the 512 x 128 nets ran on a rented L4 (PREREG
Amendments 1-3; parity gates against CPU runs passed), the rest on CPU. Medians over nets; "of mean" is the claim's own
statistic (MC at the mean cost over the mean nMSE); the claim's G_MC is converted to our L-layer pass. Brackets after
G_RQ are the range over nets; the 95% CI (exploratory bootstrap over nets and RQMC scrambles) is in the second bullet.
Entries marked * (SL3 at 128 x 32 and 256 x 64) extrapolate RQMC with slope -1, 2x and 7.6x beyond its largest
measured size (2^14 points). X_RQ values are point estimates.

| n x L | nets | LVTV2 nMSE @ cost | G_MC (median; of mean) | claim's G_MC | G_RQ [range] (wins vs RQMC) | X_RQ | RQMC gain over MC near that cost | ARC best at <= cost / LVTV2 | SL3 nMSE @ cost, G_RQ | gamma_SL | law |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 x 16 | 8 | 2.0e-03 @ 302 | 1.7; 1.7 | 2.17 | 0.79 [0.59, 1.2] (1/8) | 0.81 | 2.2 | 3.5 | 3.8e-04 @ 1e+04, 0.091 | 0.45 | 0.333 |
| 128 x 32 | 8 | 1.8e-03 @ 436 | 1.3; 1.3 | 1.52 | 0.73 [0.3, 2] (2/8) | 0.7 | 1.8 | 3.3 | 1.2e-03 @ 3.3e+04, 0.0096* | 0.11 | 0.286 |
| 256 x 64 | 8 | 6.8e-04 @ 695 | 2.1; 2.1 | 1.51 | 1.2 [0.7, 1.5] (7/8) | 1.1 | 1.9 | 3.5 | 3.7e-04 @ 1.2e+05, 0.01* | 0.11 | 0.250 |
| 512 x 128 | 8 | 1.0e-03 @ 1208 | 0.81; 0.89 | 1.41 | 0.43 [0.22, 2.5] (3/8) | 0.45 | 1.8 | 1.1 | - | - | 0.222 |
| 1024 x 256 | 8 | 1.1e-04 @ 2233 | 4.1; 3.5 | 1.21 | 2 [0.97, 11] (7/8) | 1.9 | 1.7 | 3.8 | - | - | 0.200 |

- **The claim's margin over naive MC replicates only roughly.** By its own statistic (MC at the mean cost over the mean
  nMSE) the estimator beats naive MC in 4 of 5 cells (1.7, 1.3, 2.1, 0.89, 3.5 against its 2.17, 1.52, 1.51, 1.41, 1.21),
  so our registered transfer C1 fails at 512 x 128, as a point estimate: the bootstrap interval there is 0.57-1.63. Its error varies ~10x between nets of one shape (512 x 128: 1.4e-4 to
  2.4e-3; 1024 x 256: 2.3e-5 to 3.0e-4), so 8- or 10-net cell means move by factors of 2: our 1024 x 256 nets are ~3x
  easier for it than the claim's (and ~2x easier for ARC k2), our 512 x 128 nets ~1.6x harder (for both). The code is not the cause
  (its net-0 V_L reproduces to the printed precision, the estimator is equivariant under the first-layer scaling that
  distinguishes the claim's nets from ours, to 1e-9, and a CPU rerun of one 1024 x 256 net matches the GPU to 4e-13).
  Truth noise matters little for the medians: scoring against either truth half instead of the cross-half product moves
  a cell median by <= 11% and a single net by <= 31%; on single 1024 x 256 nets the two halves differ by up to 52% of the
  cross-half error.
- **Against RQMC it is mixed on this diagonal.** RQMC is 1.7-2.2x better than MC near these costs (its 8-scramble mean
  error has a relative SE of 18-70% per net), and the estimator's median gain over it is 0.79 (exploratory pointwise 95% interval 0.58-0.99), 0.73 (0.51-1.13),
  1.16 (0.83-1.45), 0.43 (0.26-1.95), 1.96 (1.09-7.4). The intervals resample nets and RQMC scrambles; they are
  conditional on the saved truths and on the log-log interpolation. They exclude 1 only at 64 x 16 (a loss) and
  1024 x 256 (a win). Where it wins, RQMC catches up within ~1.1-1.9x its cost (point estimates; no interval). The width
  trend is undecided: exponent +0.19 (95% CI over nets only, -0.01 to +0.74) for the gain over RQMC, +0.20 (-0.05 to
  +0.65) over MC, and
  leaving out the 1024 cell turns it to -0.20 (leaving out 512: +0.37). The claim's narrowing margin (our transfer C6,
  on medians) does not reproduce, and neither does a clear widening.
- **The scale-law ladder is nearly flat on the diagonal.** SL3 improves on LVTV2 by a median factor of only 5 (64 x 16),
  1.6 (128 x 32) and 1.8 (256 x 64) for 34x, 76x and 179x the cost: gamma_SL = 0.45, 0.11, 0.11 against the uniform law's
  0.33, 0.29, 0.25 and sampling's 1, and SL3 does not improve on LVTV2 on every net (6 of 8 at 128 x 32, so our per-net transfer C2 fails there,
  although the mean error improves, 1.81e-3 to 1.50e-3, as the claim's cell-mean statement says). At its own cost SL3 loses to RQMC by ~10x (64 x 16) and ~100x (128 x 32, 256 x 64; RQMC extrapolated). The claim's own numbers say
  the same at 64 x 16: from its figure data, its ladder's exponents are 0.20 (level 2 to 3) and 0.28 (3 to 4).
- The estimator does beat ARC's best variant at or below its cost in every cell (1.1-3.8x; C4 holds), because ARC's
  order-2 variants are poor here (AUGMENT diverges at most widths), as the claim says.

## 7. Phase 2's shape, 1024 x 16 (E4: the public 'mini' MLPs 0-11 with their 1e9-sample truths; GPU run 2)

Mean avg_variance 0.080; full-budget (65,536-pass) MC raw MSE 1.2e-06; full-budget RQMC raw MSE 7.5e-07.

| method | nets | mean raw final-layer MSE | cost (passes, ARC counter) | G_MC | G_RQ | full-budget RQMC raw / method raw (median) | adjusted score (contest formula, ARC-counter cost) |
|---|---|---|---|---|---|---|---|
| COV | 12 | 4.5e-06 | 2049 | 8.4 | 6.3 | 0.17 | 4.5e-07 |
| ARC-k2 | 12 | 4.5e-06 | 1545 | 11 | 8.7 | 0.17 | 4.5e-07 |
| ARC-k2a | 12 | 1.1e-06 | 1603 | 48 | 36 | 0.72 | 1.1e-07 |
| LVTV2 | 12 | 1.4e-06 | 2162 | 26 | 20 | 0.53 | 1.4e-07 |
| LVT | 12 | 1.9e-06 | 2157 | 19 | 14 | 0.4 | 1.9e-07 |
| LVTV2f1 | 12 | 1.4e-06 | 2162 | 26 | 19 | 0.53 | 1.4e-07 |
| LVT0 | 12 | 4.2e-06 | 2156 | 8.7 | 6.5 | 0.18 | 4.2e-07 |
| ARC-k3f | 6 | 4.0e-08 | 96968 | 22 | 13* | 19 | 5.8e-08 |
| SL3 | 6 | 3.2e-08 | 212483 | 12 | 7.5* | 24 | 1.0e-07 |

gamma_ARC (k2a -> k3f) median 0.83 [range 0.76, 0.88]; gamma_SL median 0.83; law log 64 / log 1024 = 0.600.
* RQMC extrapolated with slope -1 beyond its largest measured size (2^16 points on MLPs 0-5): 1.5x for ARC-k3f, 3.2x for
SL3. The full-budget RQMC column is measured at 2^16 on MLPs 0-5 and extrapolated with slope -1 from 2^14 on MLPs 6-11.

- **"Plain covariance propagation" is ARC's own K2.** The starter kit's baseline and ARC's K2 'simple' give the same
  estimate on every MLP (P8a). Its raw final-layer MSE is 4.5e-6 (mean over the 12 MLPs; the leaderboard's 3.93e-6 is the
  same method on the hidden suite).
- **It beats sampling under the contest's score and at equal compute, not at the full budget.** At its own cost it is
  6.3x better than RQMC and 8.4x better than MC (medians; P8c). The contest multiplies the raw error by the fraction of
  the budget used, floored at 10%: the multiplier rewards cheap methods, and the floor limits that reward (without it
  COV would score ~1.40e-7 at its own cost). With the floor, COV's adjusted score is 4.5e-7 against full-budget MC's 1.2e-6 on 12 of 12
  MLPs (P8e). So claim (b) is true of the leaderboard's metric. In raw accuracy, full-budget RQMC (7.5e-7 mean; measured
  at 2^16 points on MLPs 0-5, extrapolated with slope -1 from 2^14 on 6-11) beats COV ~6x on every MLP and beats every
  deductive method we ran within the budget (P8d; registered verdict N2 holds, 12 of 12). That includes ARC k2a
  (1.1e-6, by 1.06-1.9x; on MLP 7, extrapolated, the margin 1.06 is within noise, CI 0.98-1.14) and the scale-mode
  estimator (1.4e-6, worse than k2a on 12 of 12 MLPs, as its author says).
- **Beyond the budget the ladder is steep at this shape, and order 3 wins.** ARC's factored K3 reaches 4.0e-8 at 96,968
  passes (1.48x the budget in ARC's counter; flopscope bills it at 2.24x, per the graded entry's notes), 18.5-20.6x
  below full-budget RQMC (P8f); the scale hook on top (SL3) gains 1.25x more at 2.2x the cost (their gains over RQMC at
  their own costs extrapolate RQMC beyond 2^16). Between orders 2 and 3, gamma_ARC = 0.83 (0.76-0.88 over MLPs 0-5)
  against the uniform law's log 64 / log 1024 = 0.6 (at 256 x 32 it was 0.30).
- **What this says about the negative at Phase 2.** For the implementations tested under ARC's counter, it holds (N2):
  full-budget RQMC beats every deductive method that fits the budget. That does not establish a general Phase 2
  negative. External evidence on the public board points the other way: participants' engineered order-3 chains run in
  ~12% of the budget and reach 1.1e-8 on the hidden suite, ~100x below full-budget MC. Our unengineered order 3 is ~20x
  more accurate than full-budget RQMC at 1.5x the budget, which is consistent with that. Whether this exact code keeps
  its error once engineered into the budget was not tested. Phase 2 is the wide, shallow regime (L/n = 1/64) where ARC's
  theorems and law favour deduction; the negative concerns depth growing with width.

## 8. Verdict, by the registered rule (PREREG section 6), and what it changes
- **D1 fails**: the two-level exponent of the scale-law ladder is 0.52 at n = 128 and 0.30 at n = 256 along L = n/8
  (needed >= 0.9), and its order-3 level beats RQMC on 0 of 5 nets at 256 x 32. **D2 fails**: on the diagonal the
  estimator's median gain over RQMC is below 1 at three of five widths (the exploratory interval excludes 1 only at
  64 x 16), and its width trend is undecided. **Neither registered escape criterion was met; the tested levels keep a limited winning window** (D:
  FAILS). That strengthens the negative for the methods and sizes tested. It does not settle higher orders (SL4 and
  beyond were not run on our nets), larger widths, or other methods.
- What claim (a) adds is real and specific: a deductive, fit-free Gaussian scale mixture that is the most accurate
  order-2-cost estimator measured on the deep shapes (at Phase 2's shape ARC k2a is better) (2.5x better than RQMC at its own cost at Phase 1's 256 x 32, ~2x along
  L = n/8), and a repair of the inversion of ARC's ladder at L/n = 1/4 (its own evidence; we did not run SL4). It is
  related to arc-update's radius-corrected lead (a Gamma amplitude law fitted by moments, not the exact radius law, so not
  the same closure), and its measured cost and error fall inside the window arc-update's break-even arithmetic allowed:
  at 1.75 order-2 steps per layer (3.4 when its special functions are re-priced at flopscope's rates) it beats RQMC in a
  window that closes at ~2x its cost. Its two-level exponent is steeper than ARC's at small widths, equal at 256 x 32,
  and ~0.1 at widths 128 and 256 on the L = n/4 diagonal (0.45 at 64).
- Claim (b) is true of the leaderboard's compute-adjusted metric and at equal compute, false about full-budget raw
  accuracy at order 2 (N2), and does not bear on the deep regime: at Phase 2's shape order 3 is far more accurate than
  full-budget sampling, and engineered order-3 entries beat it within the budget (public board, external evidence).
- For arc-update's report: its section 6 item 8 ("untested routes ... participant proposals such as the scale-mode
  correction of forum topic 18298") can now say "tested at orders 2 and 3: a constant factor in a window; two-level
  exponent equal to ARC's at 256 x 32 and ~0.1 at widths 128 and 256 on the L = n/4 diagonal". Its break-even estimate for a deductive
  radius-corrected closure (beat MC below ~5, RQMC below ~2-3.5 order-2 steps per layer) is consistent with this related
  method.
- What remains open is what arc-update listed (implicit representations of the early structure, future-aware
  projections, non-layer-wise mechanisms) plus the scale-law ladder above order 3. Nothing here tests those.

## 9. Process, deviations, compute, review
- Pre-registration (prereg_hashes.txt; frozen copies in frozen/): Part 1 hashed 11:22:22Z before any evaluation run
  (development: dev net 9500 at L = n/8, the claim's own cell B nets, our dev net 699 at 128 x 32 and 256 x 64, Phase 2
  MLP 12, the claim's 1024 x 256 net 0; section 1 of PREREG says what each showed). Amendment 1 (11:40:28Z, GPU cell at
  1024 x 256, 8 nets, plus 512 x 128 nets 634-637) and Amendments 2-3 (12:05Z, 12:06Z: the 256 x 64 cell, all Phase 2 jobs
  and the 128 x 32 nets 611-617 to a second GPU run) were each hashed before any result they cover existed; none changed
  a prediction, a threshold or the scorer's statistics. The hashes establish the content of each registration; the
  chronology rests on the logged hash times and queue timestamps, which the reviewer found consistent. Results
  (results/tables.md T3): of 45 registered prediction rows (C1-C6 are our operationalized transfers of the claim's
  statements, not literal reproductions of them; P1-P8 and N2 are ours), 40 hold and 5 fail: C1 at L = n/4 (margin over naive MC 0.89 at 512 x 128), C2 at 128 x 32 (SL3 below
  LVTV2 on 6 of 8 nets), C6 (the claim's narrowing margin does not reproduce), P7a and P7b (at 1024 x 256 the estimator
  did better than predicted: G_MC 4.1, G_RQ 2.0). The decision rows D1, D2 and D fail: no escape.
- Deviations (logs/deviations.txt, all disclosed): queue reshuffles to move work to the GPU (queue scripts stopped by PID,
  jobs never run twice); the scorer's E3 net lists brought in line with Amendment 1 after an interim score showed 4 nets
  at 1024 x 256 (counts 4 -> 8, re-hashed; no other change); amendment headers typed a few minutes ahead of the clock
  (hash times are authoritative); a watcher started with a wrong PID from pgrep and was restarted (no effect).
- GPU (GPU-LOG.md): two sequential spot L4 runs in us-west1-c, 0.75 h and 0.50 h, $0.65 + $0.43 = $1.08 at the list-price ceiling ($0.87/h; cap $3), each VM deleted
  by gpu-run.sh at the end; only gpu_src*/ (copies of the needed code, the two audited third-party files, and the Phase 2
  weights) went to the VMs. Parity gates against CPU runs passed (deductive estimates to <= 2.7e-10 relative with identical FLOP
  counts, RQMC estimates to <= 8.8e-8 with the same Sobol points; truths within Monte Carlo noise; weight checksums
  equal); an extra check ran LVTV2 on a 1024 x 256 net on the CPU too: 3.9e-13.
- CPU: nice 19, ionice -c3, at most two heavy processes, python -I under env -i (no secrets), the machine's load 19-37.
- Review: a fresh GPT-6 session (gpt-6-astra via codex-any; "model: gpt-6-astra" checked in review/review_gpt6.raw,
  13:02-13:13Z) recomputed the headline numbers from the raw files, re-ran the scorer's arithmetic (identical
  results/score.json), verified all 27 manifest hashes and the 237 GPU output copies, and asked for a major revision with
  9 findings, all applied: the scale-separation argument (section 4) restated as a conjecture with its assumptions; no
  promotion of failed escape criteria to an asymptotic negative, and the two-level exponent called steeper than ARC's
  at small widths (it is); the Phase 2 verdict states N2 and labels the leaderboard as external evidence; exploratory
  uncertainty added (bootstrap over nets and RQMC scrambles, truth halves, leave-one-width-out slopes;
  results/explore_uncertainty.txt, scripts/explore_uncertainty.py); C1-C6 called operationalized transfers, with the
  claim's own mean statistics; extrapolated entries marked; the 10% floor's direction corrected (it limits the reward
  for cheap methods); the link to arc-update's radial closure narrowed; "exactly" replaced by "to the printed precision"
  and brackets labelled as ranges. A short follow-up check (same model, review/check2_gpt6.raw, 13:17-13:20Z) found
  six leftovers (an unfloored-score example using the wrong cost, an exponent rounding, the truth-half statistic's
  wording, 'resolved' for pointwise exploratory intervals, unflagged extrapolated table entries, two scope phrases);
  all fixed. The final text was not re-reviewed.
- Not done: the dense de-scaled order-4 ladder (SL4) on our nets (exploratory at best; the claim's own 64 x 16 numbers
  give its exponents, section 6); the claim's graded contest entry (a fitted engine, not the claim's estimator).
