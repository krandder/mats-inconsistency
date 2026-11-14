# E-SUD addendum 1b (written 2026-10-15 ~02:08Z, after seed 0 of round 1 and before any round-1b model exists)

## Why

Round 1 seed 0 (30k steps) gave KL(exact || P0) = 0.55 nats on all test cells but also 0.47 nats on one-step cells
(naked singles), so the second half of the D0 gate (KL <= 0.02 on one-step cells) is not met: the network has not
learned the local law. Its own Solvable check then accepts 40-74% of invalid completions, so the witness arms fail for
a reason that is about training, not about search. The round-1 sweep still runs to completion and is reported as
pre-registered (D4 is scored on it as written). Validation outcome loss was still falling at step 30k.

## What changes (post hoc, labelled as such everywhere)

- Training length 30,000 -> 150,000 steps (cosine schedule over the new length), everything else identical: data,
  model (d 128, 4 layers), lr, batch, best-by-validation-outcome-loss checkpoint, seeds 0, 1, 2, inference arms, fits,
  metrics and the D4 rule with the same thresholds and statistics.
- The step count was chosen from the validation curve only; no round-1b test number has been seen.
- Each 1b seed runs on its own VM (esud-d/e/f) in budget group overnight-sud.

## Reporting

Round 1 (pre-registered) and round 1b (post hoc) are reported side by side. A pass in 1b is labelled "post hoc";
the pre-registered verdict is round 1's. D0 is recorded for both.
