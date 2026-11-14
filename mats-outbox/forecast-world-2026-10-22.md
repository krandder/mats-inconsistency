# ForecastWorld: forecasts plus source keys, aggregated by attention (2026-10-22, revised after gate 2026-10-23)

Stage 2 of the Deger plan, without LLMs. All numbers come from the result JSONs on `inconsistency-search` main (f890815),
`results/explore/forecast-world-20261022/`. The README there is generated and lists every harness commit, 1f3f3a5 included.
Plot data: `forecast-world-2026-10-22-figure-data.json`. Checked 2026-10-23 ~02:30Z. Metric: KL per question from (A), nats.

**World.** 50 hidden factors drive 20 yes/no questions. 40 news-like sources each give 8 noisy readings of a few factors.
Source popularity follows a power law with exponent alpha (the overlap knob). Share of forecasters reading a top-3 source:
0.21, 0.50, 0.82, 0.97, 1.00 for alpha 0 to 2. There are N = 10 forecasters, and each reads k = 3 sources.

**Ceilings.** (A) is Bayes given every source read. (B) is Bayes given the forecasts plus the exact source lists (exact
least squares). An independent check reproduced both. Here (B) is about (A) (0.0008 at alpha 1). That is a counting property
of this world: 200 forecast numbers overdetermine about 132 distinct readings. It is not a general law.
So within this world, any gap to (B) belongs to the learner.

**Part 1: forecasters report exact posteriors** (3 seeds, alpha 1).
- Mean log-odds scores 0.118 and extremized 0.056. Attention reaches 0.036 on forecasts only, 0.027 with an exact key and 0.025 with exact source lists.
- With 4x training and a full 20x20 readout (1 seed), the key arm gets to 0.015. That closes 74% of the extremize-to-(B) gap, and the run was still improving.
- Overlap curve, measured as the share of the prior's KL left:
  - Extremizing gets worse only in this share (23% to 34% from alpha 0 to 2). In nats it improves (0.057 to 0.047).
  - Attention with exact keys gets better (strong runs: 9.0% to 4.1%).
  - The key's gain over no key grows with overlap: 15% to 39% in the short 6000-step runs, 49% to 63% in the strong runs.
- Noisy keys: the earlier result "sigma 3 is worse than no key" was an optimization artefact. With the key standardized,
  given its own input projection and trained 4x longer, sigma 3 scores 0.0297 against 0.0317 for no key at the same budget.

**Part 2: learned forecasters.** One shared network reads raw readings mixed with nuisance features, plus a source
description (which factors, how reliable). It sends forecasts plus a learned key, or a free 64-number message. Everything is
trained together on outcomes only. The learned forecasters' own forecasts are near exact (KL 0.003).
- Share of prior KL left, alpha 0 to 2: free message 18% to 12%; forecasts + key 20% to 18%. Extremized exact forecasts: 23% to 34%.
- Fair bar: one transformer over all 30 (forecaster, source) readings, with duplicates kept and the same descriptions.
  It scores 0.023 / 0.0175 / 0.011 at alpha 0 / 1 / 2. The free message is 1.5-1.9x worse.
- Overlap does not explain that gap. With all reading sets made disjoint (no overlap at all), the ratio stays: free
  message 0.0249 vs fair bar 0.0151 (1.64x). Forecasts + key is 2.6x. The gap sits in the per-forecaster message and how
  it is learned, not in combining overlapping messages.
- Longer training (30000 steps, 1 seed, alpha 1): free message 0.024, forecasts + key with the free message's readout 0.027,
  fair bar 0.0097. The fair bar improves fastest, and nothing has converged. The old "bar" was over-privileged: it was
  handed each distinct source once and beats the fair bar even on disjoint sets (0.0106).
- Held-out sources: the descriptions do not carry the loadings, so even exact Bayes cannot read a new source.
  Fair comparators ignore new sources: Bayes 0.016, extremizing 0.065. The learned arms score 0.08-0.13, worse than both.
  This is a robustness bug (they do not discount unreadable sources), not a test of transfer.

Spend: about $17.6 for the first batch plus $3.5 for this gate batch.
Not done: (b) descriptions carrying the full loadings. That only becomes a real transfer test once loadings are redrawn every
episode, so forecasters must read the descriptions. Proposed as the next build.
