# PREREG, S0 worker 3 (LI families: D-delay, D-prove, D-expensive), written 2026-09-27 before any run

Scope: substrate code (question families + delayed settlement), not an experiment. Runs are
validation checks and a difficulty baseline so later workstreams know what the families cost.

Arms
1. Label checks (exact): tautologies A|~A and (A&B)->A are 1 at every t, A&~A is 0; each
   exclusive-exhaustive triple has exactly one true member at every t; a twin equals its base;
   an expensive question at t equals its generator k steps ahead; AND/OR equal their operands.
2. No-leak check: for every (q, t) the window at t contains only labels whose reveal time <= t;
   train positions never overlap eval positions under the stream's split.
3. Determinism: same seed gives identical arrays, different seeds differ.
4. DelayBook: every booked trade settles exactly once, at t + delay[q], at the booked price.
5. Difficulty baseline: per-family prequential BCE (nats) of an online per-question logistic
   model on the revealed window (test then train at reveal, lr 0.05, W=32), T=20000, and the
   coherence violations of that baseline before resolution (it has no cross-question structure,
   so it should violate D-prove constraints; this is the floor M5/M6 must beat).

Metric: arms 1-4 pass/fail; arm 5 BCE per family and mean abs violation per kind.
Seeds: 0, 1, 2; arm 5 reported as mean +- sd over seeds.
Stop rule: all checks green (or report what still fails after 3 fix rounds). CPU only, 1 job.
Expected (arm 5): delay0 < delay64 <= delay512 BCE on learnable generators; twins cost the same
as solos for a per-question baseline (it cannot read the twin's base); exp BCE near ln2 for
k >= 256 on CA110/LFSR for the logistic baseline.
