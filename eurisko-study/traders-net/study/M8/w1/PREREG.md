# M8 worker 1 PREREG: the sign test (reduced M8)

Written 2026-09-28 ~09:45Z, before any run. Code: `sign_gate.py`; aggregation: `aggregate.py`.

**Why reduced.** Gate B not passed: M3 found no currency beating recent |a·g| on generic neurons
(`../../M3/CURRENCY.md` "Gate B fails on M3") and M2 found no coalition win (`../../M2/BAKEOFF.md`
"Gate B coalition win: NO"). The 00:25Z RETARGET cut M8 to "the sign test only". No exit price,
no rebirth, no η/α/c grid. Item 2 (|a·g|-gated arm with reinit controls) is worker 2's.

**Question.** When MLP neurons are gated by their gradient income, does the *sign* of that income
carry usable information? The sign-flip diagnosis (`../../M1/CURRENCY.lock`) predicts it does not.

**Substrate.** S0 shift stream `streams_ext.make_streams(Q=64, T=40000, seed, heldout=8,
regime_every=10000, regime_frac=0.25)` (Qt=72 tokens), `model.Net(d=64, H=4, L=2, mlp=4)` (2x256
MLP neurons), Adam 1e-3, prequential test-then-train, B=4 positions per step, one pass. The last
512 positions are never trained: they are the held-out oracle segment for the census.

**Income.** Per step, on the training batch (every label already scored, prequential):
c_i = −Σ h_i·∂L/∂h_i over batch and questions, h_i the gated activation. EMA with half-life 1000
steps (M3's middle value). Recorded in every arm, including ungated.

**Gate.** Per layer, z = standardised EMA income; w = N·[(1−α)·softmax(s·η·z) + α/N], α = 1e-3
(M3's fixed-share recommendation), mean gate 1, detached, updated every step after 500 warm-up
steps (gate 1 before).

**Arms (6), seeds 1, 2, 3 (18 runs).**
- `none`: ungated.
- `pos0.1`, `pos1`: s = +1 (pay positive income), η = 0.1, 1.
- `neg0.1`, `neg1`: s = −1 (sign flipped), η = 0.1, 1.
- `shuf1`: the `pos1` gate values randomly permuted across neurons, permutation redrawn every
  1000 steps (same gate distribution, no link to income).

**Metrics.** Primary: prequential BCE (nats) over all live (position, question) pairs.
Secondary: BCE over the 2000 stream time steps after each shift (10k/20k/30k), all questions and
switched questions only; BCE in the 2000 before each shift. Census (end of run, held-out segment):
single-neuron zero ablation ΔBCE; share of the top 10% neurons by ΔBCE whose EMA income is
negative, vs the share in all neurons; Spearman(income, ΔBCE); Spearman(|a·g| EMA, ΔBCE) only as a
reference column. Mean ± sd (ddof 1) over 3 seeds.

**Decision rules (fixed now).**
1. Sign informative: `pos1` beats `neg1` by ≥ 0.005 nats primary BCE in 3/3 seeds AND beats
   `shuf1` by ≥ 0.005 in 3/3 seeds. Otherwise: the sign is not a usable gate signal.
2. Plan stop rule analog ("no gain over random reinit in 3 seeds: stop"): if `pos1` does not beat
   `shuf1` in 3/3 seeds, M8 stops here; no further arms.
3. Census: the sign is "anti-informative on important units" if the top-10% negative share exceeds
   the all-neuron negative share in 3/3 seeds in the ungated arm.

**Predictions.** (1) fails; `neg1` within noise of or better than `pos1`; census (3) holds or ties.
Budget: CPU only, at most 4 concurrent jobs, `torch.set_num_threads(1)`.

Amendment 09:52Z, before any full run (only a timing smoke test at T=6000, seed 9, not reported): held-out census segment cut from 2048 to 512 positions (512 ablation passes were too slow at box load 35; smoke tests at T=6000 and T=4000, seed 9, not reported). The census ablates on every 4th of those 512 positions (128 positions x 72 questions) for the same reason.

Amendment 10:20Z, before any full run finished (first wave killed at ~step 4000, nothing read): after
worker 2 reported that LFSR16 learn events dominate whole-run BCE differences (`../STATUS.md` 10:12Z),
the runs also record per-family BCE, the primary BCE with LFSR16 excluded, and the number of
LFSR16 questions learnt (mean BCE < 0.3 over the last 5000 time steps). Decision rule 1 is applied
to both the primary BCE and the LFSR16-excluded BCE; the sign is called informative only if it
passes on both. Rules 2-3 unchanged.
