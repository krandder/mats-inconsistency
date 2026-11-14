# Item 3 pre-registration: P8's pricing rules against a gain-over-base truth at N = 4096

Frozen before any run that produces test numbers. The sha256 of this file (and of the analysis and run code) is in
`PREREG.sha256`, written before launch. Changes after the freeze go in RESULTS.md under "Deviations".

## Question

P8 (N = 1024) found that charging learnability hurts: against the deployed truth, the one-code price R4 ranked the
32 maze intermediates at Kendall tau-b 0.29 against `gain_vs_base`, while importance alone (R1) got 0.74. One reading
is that at N = 1024 the pipeline cannot yet use hard-to-learn intermediates, so a learnability charge should matter
more with more data. At N = 4096, where the baselines are strong (C2: B2 0.952, B3 0.993 held-out exact-maze
accuracy), **do R4 (bits saved minus bits to learn Z) or R8 (importance − λ·learnability, λ fixed in advance) rank
the 32 items better than R1 (importance only), against a gain-over-base deployed truth?**

## Design (everything as in P8 unless stated)

- Task: P8's maze task (11x11, density 0.3, Y = 1[BFS distance ≤ 8]); inputs X = wall/start/inside/decoy.
- Items: the 32 P8 maze items, 17 pool variables (g1-g7, m2/m4/m6/m8, nz05, nz20, d2/d4/d6, reach) in context
  S = ∅ and 15 in S = {g2, g4}. Sanity items copyY, random, null, open, startZ (S = ∅).
- Seeds: 0 .. n−1 with n = **5** (power calculation below). Seed 99 was used only for the timing probe and is
  excluded from every analysis.
- N = 4096 for both the price stream and the truth-train bank.
  - Price stream: `seed_int("maze-price", s)`, as in P8, now 4096 mazes long.
  - Truth-train bank: `seed_int("maze-truth4k", s)`, the draw of C2's N = 4096 point (P8 used "maze-truth" at 1024).
  - Held set: P8's held set, 1024 mazes from `seed_int("maze-held", s)`, plus the 15x15 set (512 mazes, reported only).
- Code: `code/` = P8's `jobs.py`, `p8lib.py`, `maze_gen.py`, `dtnet_vendored.py` with three patches: N from `P8_N`,
  the truth-train tag from `P8_TRUTH_TAG`, and R7a switched off (no hats in `price_rep`). New: `jobs3.py` (job split
  per replicate), `driver.py` (queue + automatic gates), `analyze3.py` (all numbers below), `power3.py`.

### Prices (rules)

Each price is the mean over KREP = **8** paired prequential replicates (P8 `price_rep`: blocks 8, 16, ...,
4096; own init and own data order per replicate; lr 3e-3, batch 32, max 2000 steps, patience 100, 2-layer GConv h16).

- R1 = L(Y|X,S) − L(Y|X,S,Z) (importance only).
- R4 = R1 − L(Z|X,S) (bits saved minus bits to learn Z; P8's rule).
- R8 = R1 − λ·L(Z|X,S) with **λ = 0.0063** (P8's parity fit, 10^−2.2 on P8's grid). Fixed now.
- Secondary: R8o = R1 − 0.0631·L(Z|X,S) (P8's in-sample maze oracle λ at N = 1024, fixed now); R2 = −L(Z|X,S) among
  items with R1 > 0 (P8 convention); R3 final-block importance; R5 last-block saving; R7b wrapper validation (80/20
  split, test-loss difference × N, mean over **4** replicates instead of P8's 12, for cost).
- R7a is not run (it needs the hats pipeline; P8 found it partly circular).

### Truth (defined exactly)

For seed s, replicate r ∈ {0, 1} and item i = (context c, variable z):

- Run P8's `job_truth` (maze) on the N = 4096 truth-train bank, with every training seed replaced by
  ts = s + 1000·r (cross-fitted Ẑ stages A/B/C, Y nets 2-layer GConv h16 with 2 inits, lr 3e-3, batch 32, max 6000
  steps, patience 600). Banks depend on s only; r changes all training randomness.
- gain_vs_base(s, r, i) = mean over the 1024 held mazes of [bits of the base Y net (inputs X + Ŝ_c, per-example
  bits averaged over the 2 inits) − bits of the pipeline Y net (inputs X + Ŝ_c + Ẑ_i, averaged over the 2 inits)],
  in bits per maze.
- **Truth T(s, i) = mean over r ∈ {0, 1} of gain_vs_base(s, r, i).**
- P8's control truth (free-latent GStack − pipeline) is also computed and reported as a secondary column only.

### Primary statistic

- τ_R(s) = Kendall tau-b between rule R's prices and T(s, ·) over the 32 items.
- d_R(s) = τ_R(s) − τ_R1(s), for R ∈ {R4, R8}.
- Primary: mean over seeds of d_R with a two-sided **97.5%** t-CI (Bonferroni over the two contrasts), for R4 and R8.
- Also reported: τ per rule with 95% t-CIs over seeds; P8's paired permutation test (swap the two rules' within-seed
  ranks per item, 4000 draws) in both directions.

### Bars (decided now)

- **Gain** ("learnability should count at N = 4096"): for R4 or R8, mean d ≥ 0.05 and the 97.5% lower bound > 0.
- **No gain** ("learnability should not count"): for both R4 and R8, the 97.5% upper bound < 0.05.
- **Informative**: anything else (the data cannot separate the two at this precision).
- **Problem**: any gate fails, or fewer than 3 complete seeds.
- Also stated: "R4 hurts" if R4's 97.5% upper bound < 0.

### Power calculation (sets n)

`code/power3.py` (noncentral t; 97.5% two-sided t-CI; power = P(lower bound > 0)):

- Assumed SD of d over seeds: σ = 0.041, P8's largest per-seed SD of a paired difference vs R1 under `gain_vs_base`
  (R4 − R1). P8's R8 − R1 SD was 0.011, so R8 is over-powered at any n ≥ 3.
- Assumed effect: Δ = 0.10 (a clear improvement in rank agreement).
- Power at σ = 0.041, Δ = 0.10: n = 3: 0.37, 4: 0.72, **5: 0.91**, 6: 0.98. n = 5 is the smallest n with power ≥ 0.8.
- At the 0.05 margin itself power is low (n = 5: 0.37, n = 8: 0.72); an effect that small will likely read
  Informative. Expected 97.5% half-width at σ = 0.041: n = 5: 0.064, n = 6: 0.053, n = 8: 0.041.
- **Chosen: n = 5 seeds (0-4), KREP = 8 price replicates, 4 R7b replicates.** n = 5 is the power-calculation
  minimum; more seeds do not fit the budget. KREP = 8 instead of P8's 12 is for cost: P8's G3 noise ratio at 12
  replicates was 0.04-0.06, which scales to about 0.05-0.075 at 8, under the 0.10 bar (G3 checks it per seed).
  Fewer replicates add price noise, so σ = 0.041 may be slightly optimistic.

## Gates (in `driver.py`, automatic; a failed gate writes GATE_FAILED.json, kills every running job and exits)

Each gate is checked per seed as soon as its inputs exist, so a failure stops the run early.

- **G1 baselines strong at N = 4096:** C2's selected B2 (8-layer conv, h64, lr 1e-3, max 8000 steps) trained on the
  seed's truth-train bank reaches held-out exact-maze accuracy ≥ 0.90. B3 is not rerun (cost); C2 reported 0.993.
- **G2 truth reliability (A/A):** Spearman between gain_vs_base of replicate 0 and replicate 1 over the 32 items
  ≥ 0.80, with all values finite, in every seed.
- **G3 price validity, per seed:** all R1, R4, L_Z finite; noise ratio 3·sd(R4 of S=∅:g2 over replicates)/√KREP /
  R4(S=∅:g2) ≤ 0.10 (P8's Phase 0 standard); random Z: R4 within 5% of −N·121; null Z: |R1| ≤ 0.01. copyY's R4 is
  reported, not gated (its P8 band [−2F, 0] depends on N).
- **G3a completeness:** any job that fails twice stops the run.
- **Deadline (contingency):** at **2026-09-30 02:20:00Z** (epoch 1790734800) the driver kills every running job and exits. The analysis then uses
  the seeds that are complete (prices, both truth replicates and B2, all passing their gates), taken in seed order;
  R7b is reported on the seeds where it finished. With fewer than 3 complete seeds the status is Problem.
- Checked on P8's N = 1024 files (seen before this freeze): noise ratio 0.04-0.06, random within 1.3%, null exactly 0,
  copyY −969.

## Pre-registered check: the "noisy copy of Y" (nz05 vs g2, S = ∅)

P8's verifier noted that P8's truth rewarded a noisy copy of Y (nz05) over g2. At N = 1024 under `gain_vs_base`
(P8 files, seen): nz05 18.7 / 16.2 / 18.5 vs g2 15.7 / 16.6 / 15.9 bits/maze. R1 prices nz05 far above g2
(63k vs 15k bits at seed 0), R4 far below (−23k vs 13k).

- Statistic: per held maze, bits(nz05 pipeline) − bits(g2 pipeline), averaged over inits and the 2 replicates; mean
  over mazes per seed, then a 95% t-CI over seeds.
- Verdict: "g2 > nz05" if the CI is above 0, "nz05 > g2" if below, else "tie". Also reported: for each rule, the share
  of seeds where it orders the pair the same way as the truth.
- Reading: "g2 > nz05" is a concrete case where importance alone (R1) misorders and the learnability charge helps.
  "nz05 > g2" means the new truth still rewards the noisy copy; the primary verdict then carries that caveat.

## Secondary and exploratory (no verdicts)

- Per-context taus (S = ∅, S = {g2, g4}).
- The in-sample λ curve (grid 0 and 10^−3 .. 10^1, 41 points): exploratory only, never used to pick λ.
- Taus against P8's control truth at N = 4096.
- Reuse: P8's N = 1024 price files (seeds 0-2, `P8/results/price_maze_s*.json`, `r7b_maze_s*.json`) scored against
  the new N = 4096 truth, to see whether cheaper pricing ranks the same.

## Disclosures

- Earlier results seen before this freeze: P8's RESULTS.md and result files at N = 1024 (taus, per-seed SDs, prices,
  truth), and C2's N = 4096 B2/B3 numbers. They set σ, λ and the gate bars. No N = 4096 price or truth number has been
  seen.
- Timing probe (seed 99, excluded): one price replicate, one truth replicate, one B2 gate run and one R7b replicate.
  B2 took 449 s and R7b 849 s (4 jobs on 32 vCPUs). A spot preemption at 23:10Z killed the price and truth probes
  after more than 23 minutes each; the VM was recreated automatically. n and KREP were set from these times and P8's
  N = 1024 times (price 406 s per replicate, truth 355 s, R7b 124 s), assuming about 2,500 s per price or truth
  replicate on an idle core and about 1.6x slower with 31 jobs on 32 vCPUs; the deadline bounds the cost if this is
  wrong. Only CPU times were read, plus the probe's B2 held accuracy (a gate input, 0.965), to avoid paying for a run
  that G1 would stop.
- Order of work in the driver: B2 for all seeds, then both truth replicates for all seeds, then price replicates
  seed by seed (so seeds complete in order), then R7b.
- Budget: $5 at the ceiling ($1.20/h, c2d-highcpu-32 spot, VM created 22:41:50Z on 2026-09-29).
- Jobs run up to about 25-40 minutes without an in-job checkpoint; the driver resumes at job granularity.
