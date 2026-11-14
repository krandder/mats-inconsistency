# PREREG-TAILS-p3b — does the part 3 readout extend ARC's tail regime at depth 32?

Written 2026-10-29 12:13Z, before any depth-32 run of ARC's augmented description finished and before any scoring
here. Brief (opus-mats, under Kelvin's delegation): existing part 2 truths only, no new splitting; ARC's augmented
factorized K = 3 at depth 32 on the evaluation nets, CPU only (at most 4 processes, nice); pre-register the bar; short
write-up as part 3b by about 15:30Z with a quick GPT-6 check; if kprop at depth 32 is too slow, report how far it got.
No depth-16 truths exist, so depth 16 is not tested.

## What is known before this file
- Part 2, depth 32, sigma = 0.001 (20 evaluation nets): ARC's tail method (K3f, heaviside) never beats zero and is
  within 50% for at most 1% of events at every level; the order-4 saddlepoint on the K3f moments is within 50% for
  3-12% of events and needs fallbacks for 53-61% of them; the chi-mixture variant on the K3f moments returns no value for
  most events (no admissible saddlepoint). Monte Carlo moments + saddlepoint reach 0.56 within 50% at 1e-4 only.
- Part 3: the chi-mixture saddlepoint on ARC's augmented description (P3A) beats ARC's readout of that description at
  depths 4 and 8 on fresh nets. Its behaviour at depth 32 has not been looked at.
- K3fa at depth 8 took 4-9 minutes per net on this loaded machine; at depth 32 it is expected to be several times slower.

## Data and runs
- Part 2 evaluation nets 0-19, depth 32, outputs 0-7, the five part 2 thresholds per output, the part 2 truths
  (32 frozen splitting runs of 2,000 particles; provisional flags as in part 2). sigma = 0.001 primary.
- ARC's factorized K = 3 AUGMENT kind (run_lpe settings, our weights and noise) at depth 32, run with
  run_kprop_tail.py (heaviside readout for the five threshold vectors and the four moment readouts), nets in the order
  0, 1, 2, ... at sigma = 0.001, at most 4 processes, until 15:00Z. sigma = 0.003 only if all 20 nets at 0.001 finish.
- Scoring uses every net completed by then; a claim needs at least 8 nets, otherwise the result is reported as "too slow:
  n nets" with descriptive numbers only.

## Methods (code unchanged from part 3)
- P3A: deduce.radial(order 4) on the K3fa moments, fallbacks SP4, order 3, Gaussian (p3_score.p3a).
- Competitors: ARC's heaviside readout of the same K3fa description (C-K3fa) and of K3f (C-K3f, part 2 runs).
- Secondary: P3A on the K3f moments (no extra runs).

## Bars
- Primary 1, regime: P3A beats zero at depth 32, 1e-4 (median relative error over the events <= 0.9, with the upper end
  of the net-bootstrap 95% CI below 1), all events. Reported: the deepest level at which it beats zero, all events and
  validated events only; the same for C-K3fa and C-K3f.
- Primary 2, gain: at 1e-4 the mean over nets of the per-net difference (P3A minus C-K3fa) in the fraction of the
  net's 8 events within 50% is above 0 with its bootstrap 95% CI above 0.
- Secondary: the same difference at 1e-6 ... 1e-12 (validated events only reported alongside, since half the 1e-12
  truths are provisional at depth 32); differences against C-K3f; fallback counts; costs (ARC's counter).

## Predictions
Uncertain. I expect P3A to beat zero at 1e-4 and perhaps 1e-6, with a positive gain at 1e-4, and to fail deeper, since
depth-32 tails are far from what four moments describe (some outputs reach 1e-12 at 13 SD).
