# PREREG (S0 worker 1): prequential single pass vs multi-epoch stress

Written 2026-09-27 before the first run. Code: `substrate/prequential.py`, driver `w1/run_w1.py`.

**Question.** How far do wealth-net-style in-sample numbers (multi-epoch on a fixed train split)
sit from the prequential single-pass numbers every later arm will use, per family?

**Stream.** `substrate/streams.py` (verbatim wealth-net), Q=64, W=32, T=64032 (n_pos=64000;
below the LFSR-16 / prime period 65535, so no question repeats itself). Stream seed = model seed.
Model: `substrate/model.py` Net d=64 L=2 H=4, Adam lr 1e-3; logistic baseline Adam lr 1e-2.
1 CPU thread per run.

**Arms.**
- `P4`: prequential single pass, B=4 positions per step (16000 steps).
- `P16`: prequential single pass, B=16 (4000 steps).
- `ME`: multi-epoch stress: train on positions [0, 51200) for 60 shuffled epochs, B=128
  (24000 steps); held out = [51200, 64000), never trained.

**Metric.** Per family BCE (nats) and Brier on the last 20% of positions [51200, 64000):
for P arms the trade-time (prequential) prediction; for ME the held-out prediction after the final
epoch (4096 evenly spaced held-out positions) and the in-sample value on 4096 train positions.
Derived: ME memorisation gap = held-out BCE − in-sample BCE; P−ME difference on held-out.
Also: content-overlap audit (fraction of held-out 49-bit contexts+targets seen anywhere in the
train region of any question) per family per seed.

**Seeds.** 1, 2, 3. Report mean ± sd (sample sd, n=3).

**Expected.** Gap > 0.05 nats on the unlearnable families (pi, CA30) for ME; P arms at ≈ ln 2
there. Learnable families (periodic, and/or compounds of periodic) similar across arms.

**Stop rule.** Fixed budget: all 9 runs complete, no early stopping, no re-runs except crashes.
A run exceeding 90 min wall time is stopped and reported partial. At most 4 of my jobs at once.

**Amendment 1 (2026-09-27 22:27Z, before any result existed).** Measured 0.26 s per B=128 step on
this loaded box (load ~40 on 16 CPUs), so 60 epochs (24000 steps) would pass the 90-min cap. ME is
changed to 15 epochs (6000 steps); everything else unchanged. The three ME runs started at 22:12Z
were killed with no output written.
