# M6 stress s2 PREREG: 50% injected noise traders

Written 2026-09-28 09:45Z, before any run. Claim: `../STATUS.md` (s1's split: s2 = stress item 3 only).
Code: `noise.py` (imports `../w2/pool.py` unchanged; job copy in `job/`).

- **Harness.** w2's T=60000 harness exactly (S0 `make_streams(Q=64, T=60000, heldout=8, release=T/2,
  regime_every=0)`, single-pass prequential, B=4, trunk d=64 L=2, 16 trader heads per question,
  market l0 = per-question logistic, detached). Runs on one G1 VM (8 vCPU, 1 thread per run); clean
  runs are redone on the VM so that every comparison is within one machine.
- **Noise traders.** 8 of the 16 heads (heads 8-15) are replaced by untrainable noise traders with
  scale sigma = 1.0 logit. Two kinds:
  - `iid`: d_i ~ N(0, 1) fresh per sample, question and pass (pure noise, cannot be cancelled);
  - `fixed`: d_i = frozen random linear readout of the question's own input window,
    (2·win − 1)·R_i / sqrt(W), R_i ~ N(0,1), frozen (confident, consistent, wrong; the trainable
    heads can in principle learn to cancel it).
  Condition `clean` = no injection (w2's harness).
- **Arms** (all surviving base arms that live in w2's harness, plus the kill controls):
  `a` plain logit sum; `b` l0 + Σd; `cg` convex pool, k global learned; `cq` convex pool, k per
  question learned; `fb01`, `fb05` bounded bets l0 + Σ b·tanh(d/b), b = 0.1, 0.5; kill controls
  `wd` (arm a, AdamW weight decay 0.1) and `boost` (l0 + 0.1·Σd). In every arm the noise heads enter
  exactly where a trainable head would (the sum, the pool, the tanh bound, the 0.1 shrinkage).
  Not run: ES (no epochs in single pass; s1 owns multi-epoch), fixed point / 1-pass / 2-pass
  (w4 harness), d0/dfs (dropped at base).
- **Seeds.** 1, 2, 3 (stream, init and noise generator paired across arms and conditions).
  8 arms × 3 conditions × 3 seeds = 72 runs. Mean ± sample sd.
- **Metrics.** Primary: learnable-family BCE (mean over periodic, lfsr16, ca110, prime, and, or,
  periodic_ho; all positions), and its degradation Δ = noisy − clean, paired per seed.
  Secondary: the same without lfsr16 (bimodal); noise-family excess max over {ca30, prng} of
  (BCE − ln 2); for cg/cq the pool mass on noise heads at the end (k_noise); rel. to l0 alone.
- **Hypotheses / decision rule (per noise kind).**
  - H1 "the pool defunds noise traders": cg or cq has k_noise < 0.1 at the end in 3/3 seeds.
  - H2 "protection beyond kill controls": an arm's noisy learnable BCE < min(wd, boost) noisy
    learnable BCE, paired, in 3/3 seeds, and the same holds without lfsr16. An arm with H2 is a
    stress win on noise traders; with no arm passing H2, report that shrinkage (boost) already
    covers noise traders.
  - H3 "harm is bounded": Δ ≤ +1% of the arm's clean BCE in 3/3 seeds.
- **Stop rule.** One pass of the 72 runs; no tuning of sigma, b or ν after seeing results. If the VM is
  preempted twice, finish the missing runs on CPU here (≤ 4 jobs at a time, after checking load).
