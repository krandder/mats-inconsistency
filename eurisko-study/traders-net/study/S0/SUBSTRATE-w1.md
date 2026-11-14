# SUBSTRATE, worker 1 part: base copy, prequential single pass, multi-epoch stress

2026-09-27, S0 worker 1. Prereg: `PREREG-w1.md` (amendment 1: ME 60 -> 15 epochs for wall time,
made before any result). All numbers below: `w1/SUMMARY-w1.csv` (built by `w1/summarize.py` from
`w1/results/<arm>-s<seed>.json`); curves in `w1/results/<arm>-s<seed>-curve.csv`; trade-time
logits [n_pos,Q] for the P arms in `/tmp/traders-net/S0/w1/P*-s*.npz` (bulky, root fs, not synced).

## Files
- `substrate/streams.py`, `substrate/model.py`: verbatim copies of `../wealth-net/` (do not edit).
- `substrate/prequential.py`:
  - `ArrayStream(bits, fam, W, delay=None, live=None)`: the Stream interface
    (`window(idx)`, `label(idx)`, `label_mask(idx)`, `delay[Q]`, `fam[Q]`, `n_pos`). A delayed
    question's window shows only revealed bits (shifted by D). Worker 2 subclasses it
    (`streams_ext.to_array_stream`), worker 3's `LIStream` implements the same interface.
  - `run_prequential(model, stream, opt, B, baseline=, opt_b=, on_step=)`: step s predicts block
    [sB,(s+1)B) at the current weights (trade-time logits kept), then takes one step on every
    (i,q) with i+delay[q] in that block. Asserts every trained label was predicted earlier and is
    trained once. Returns z, zb, y, mask, pred_step, train_step.
  - `run_multiepoch(..., epochs, B, frac=0.8)`: train [0, split), held out [split, n_pos);
    `assert_disjoint` enforces that every train target index precedes every eval target.
  - `overlap_audit(bits, fam, split, W, L=48)`: content-level leak check (held-out 49-grams that
    occur in any question's train region).
- `substrate/tests/test_prequential.py`: seeded determinism, predict-before-train-once, delay
  windows/training lag, disjointness assert raises, known per-family BCE/Brier on a toy.
  Whole suite: `cd substrate; PYTHONPATH=../../../../wealth-net/.pylib:. python3 -m pytest -q tests`
  -> 39 passed (23:00Z).

## Result: prequential vs multi-epoch (Q=64, W=32, n_pos=64000, 3 seeds, BCE nats on positions 51200-64000)

| family | P4 prequential | P16 prequential | ME held-out | ME in-sample | ME gap (ho-in) | overlap49 |
|---|---|---|---|---|---|---|
| periodic | 0.0002 ± 0.0000 | 0.0003 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 1.00 |
| lfsr16 | 0.46 ± 0.40 (0.694/0.002/0.694) | 0.10 ± 0.17 (0.004/0.299/0.005) | 0.0003 ± 0.0002 | 0.0002 | 0.0001 | 1.00 |
| ca30 | 0.6945 ± 0.0003 | 0.6964 ± 0.0007 | 0.683 ± 0.013 | 0.679 | 0.004 ± 0.004 | 0.00 |
| ca110 | 0.015 ± 0.007 | 0.016 ± 0.010 | 0.007 ± 0.005 | 0.012 | -0.006 ± 0.006 | 0.97 |
| prime | 0.190 ± 0.001 | 0.189 ± 0.002 | 0.201 ± 0.024 | 0.162 | 0.039 ± 0.023 | 1.00 |
| pi | 0.6933 ± 0.0004 | 0.6957 ± 0.0004 | **0.830 ± 0.021** | **0.562 ± 0.022** | **0.267 ± 0.043** | 0.06 |
| and | 0.29 ± 0.11 | 0.29 ± 0.11 | 0.25 ± 0.10 | 0.24 | 0.014 ± 0.007 | 0.57 |
| or | 0.48 ± 0.08 | 0.47 ± 0.07 | 0.36 ± 0.06 | 0.35 | 0.014 ± 0.015 | 0.05 |

Steps: P4 16000, P16 4000, ME 6000 (15 epochs x 400, B=128). Wall (1 thread, load 30-40):
P4 246 ± 11 s, P16 158 ± 8 s, ME 1916 ± 32 s.

Reading:
1. **Multi-epoch memorises pi; prequential cannot.** ME pushes pi in-sample to 0.56 nats and
   held-out to 0.83 (worse than ln 2 = 0.693): this is the in-sample memorisation that
   wealth-net's in-sample books pay for. Prequential pi sits at ln 2 (0.6933 ± 0.0004 for P4).
   Prereg expectation "gap > 0.05 on pi and CA30": held for pi, **failed for CA30** (gap
   0.004 ± 0.004; the net does not memorise CA30 in 15 epochs).
2. **LFSR is a phase transition, bimodal across seeds** under prequential training (learnt in 1/3
   seeds at P4, 2/3 at P16, per-seed values in the table; P16-s2 drops from 0.695 to 0.383 in its
   last 8k positions, `w1/results/P16-s2-curve.csv`). Do not report LFSR as mean ± sd alone;
   report learnt/not-learnt counts.
3. **Leak flag for long streams:** at T ≈ 64k every LFSR-16 and prime question covers nearly
   its whole 65535-cycle, so held-out LFSR/prime contexts occur in other questions' training
   regions (overlap49 = 1.00 in 3/3 seeds). LFSR is also locally learnable (linear recurrence
   of order 16 fits in W=32), so ME's 0.0003 cannot be read as generalisation. Worker 2's
   PRNG-parity family (overlap 0%) is the clean unlearnable control; keep T < 65535 or treat
   LFSR/prime held-out numbers as in-sample.
4. The logistic baseline (Adam lr 1e-2) under prequential training sits above ln 2 on pure-noise
   families (pi 0.714 ± 0.0001 at P4): learner noise, a floor any market layer must beat
   (worker 2 saw the same).
5. More compute helps ME on the compounds (and/or 0.25/0.36 vs 0.29/0.47 prequential), so
   compare arms at matched steps x batch, not matched positions.

## For later workstreams
- Use `run_prequential` for every "prequential" number (M1 books v2, M3, M5-M9); use
  `run_multiepoch` only as the stress mode. Keep `torch.set_num_threads(1)`.
- B=16 is 1.6x cheaper than B=4 and gives the same noise-family numbers; LFSR learning depends
  on B and seed.
