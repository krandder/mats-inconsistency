# Tiny-PFN plateau: historical audit for the paired study

Checked at **2026-10-28 10:54 UTC** by reading the 135 retained `meta.json`
validation curves, `runs/plateau.md`, `PREREG.md`, `plateau.py`, `pfn.py`,
`world.py`, and the proposed paired-study preregistration and source. I did not
load a checkpoint, train a model, contact a provider, or modify the original
`work/tiny-pfn/` tree. Durable recall returned no tiny-PFN-specific prior
decision.

The 21-case machine-readable audit is
`tiny-pfn-plateau-historical-audit-2026-10-28.json`, SHA256
`1a1d6fa794b139b179fdd19af81628ca396d7e48d5c21aa8a502f2d0522b75f2`.

## Decision-relevant result

`21/95` is a pooled plateau-incidence count across 19 different depth-2
configurations. It is not an escape probability for one model. The pool mixes
small models whose final capacity limit lies in the plateau band, Gaussian and
mixture heads, R^4 and R^16 identities, fresh and warm starts, and 50k, 90k,
and 170k training horizons.

The historical escape rule also overcounts. Five of the 21 entrants eventually
had one validation gap below 0.14. Four stopped near 0.13 and remained far from
the Bayes-like regime. Only `w16_h2_L2` seed 2 acquired the in-context solution.
It first had three consecutive gaps below 0.12 at 44k, below 0.10 at
49k, and below 0.05 at 87k. The third confirming observations arrived at 46k,
51k, and 89k respectively.

This history supports a 60k minimum horizon for measuring provisional escape.
A 40k study can measure whether a seed acquires early or enters a plateau, but
it cannot estimate historical escape from the plateau. The revised paired-study
preregistration now prioritizes 80k and 60k horizons and uses three consecutive
gaps below 0.10 as its primary endpoint.

## What the 95-run denominator contains

| Family | Runs | Horizon | Plateau entrants | One-point `<0.14` | Sustained `<0.10` |
|---|---:|---|---:|---:|---:|
| Main R^16 Gaussian grid, width 16/32 | 25 | 40k or 80k constant phase, then 10k cooldown | 3 | 1 | 1 |
| R^4 Gaussian controls, width 16/32 | 10 | same adaptive 50k/90k total | 0 | 0 | 0 |
| Width 4/8 depth-2 micro models | 20 | same adaptive 50k/90k total | 14 | 0 | 0 |
| Cold-start K=4 mixture models | 15 | same adaptive 50k/90k total | 4 | 4 | 0 |
| Fixed-160k Gaussian reruns | 15 | 160k constant phase plus 10k cooldown | 0 | 0 | 0 |
| Mixture models warm-started from the long Gaussian runs | 10 | 40k constant phase plus 10k cooldown | 0 | 0 | 0 |
| **Total** | **95** | **50k, 90k, or 170k** | **21** | **5** | **1** |

The original training settings were batch 256, AdamW with PyTorch's default
epsilon `1e-8`, learning rate `1e-3`, weight decay `0.01`, 1k warmup, and
per-model gradient clipping at 1.0. Validation used the same 4,096 fixed
episodes every 1,000 updates. These settings are specified in
`../work/tiny-pfn/PREREG.md:49-59` and implemented in
`../work/tiny-pfn/pfn.py:18-22,202-265`.

The 20 micro runs are the main denominator problem. Fourteen entered the band,
and none escaped, but their sibling seeds also ended near the query-only level.
These runs are capacity-confounded: their size may impose a final capacity
limit, or it may make optimization harder. The retained runs do not distinguish
those explanations. The seven entrants from configurations whose sibling seeds
reached the Bayes-like regime are less capacity-confounded historical
optimization cases: three main Gaussian runs and four cold mixture runs. One
of those seven genuinely escaped by its observed horizon. That `1/7` is
descriptive evidence from heterogeneous settings, not a rate estimate for the
new Gaussian study.

The complete retained table is
`../work/tiny-pfn/runs/plateau.md:15-46`. The preregistered diagnostic used ten
consecutive observations in `[0.14, 0.18]`; at 1k evaluation spacing these ten
points span 9k updates. `plateau.py` then called the first single point below
0.14 after the last qualifying stretch an escape
(`../work/tiny-pfn/plateau.py:11-29`).

## What the early NLL logs say

These records provide no validated predictor of later acquisition from
validation gap through 40k. The sole genuine escape overlaps the two trapped
Gaussian seeds:

| `w16_h2_L2` seed | 10k | 20k | 30k | 40k | 50k | terminal |
|---:|---:|---:|---:|---:|---:|---:|
| 2, genuine escape | 0.180 | 0.157 | 0.159 | 0.156 | 0.083 | 0.044 at 90k |
| 3, trapped | 0.169 | 0.164 | 0.164 | 0.158 | 0.151 | 0.151 at 50k |
| 4, trapped | 0.167 | 0.158 | 0.162 | 0.163 | 0.152 | 0.152 at 50k |

Seed 2's first clear loss break came immediately after the old 40k decision:
0.156 at 40k, 0.139 at 41k, 0.119 at 42k, and 0.083 at 50k. Its old
convergence statistic was 0.003818, just above the 0.003 extension threshold,
so it continued to 80k. The trapped seeds' statistics were -0.001045 and
0.001570, so they stopped. This is one positive case. It does not validate the
convergence statistic as a predictor, and one false weak escape also exceeded
the extension threshold at 40k.

Across all 21 historical plateau entrants, three consecutive gaps below 0.12,
0.10, or 0.05 each select only seed 2. The first-event times differ as described
above. The 0.10 rule leaves about 0.03 nats of separation from the four false
escapes at 0.130-0.133 and is the best operational endpoint among those checked.

The NLL curve can identify acquisition once the transition starts. These
records do not validate an early predictor before the transition. Attention,
functional binding, and optimizer-state measurements are therefore needed
rather than a new cutoff fitted to this single positive trajectory.

## The plateau models use some nonquery context

The name “query-only plateau” describes the level and attractor. It should not
be read as a literal claim that these models predict from the three query
readings alone. The retained per-episode test NLLs allow a direct check against
exact query-only Bayes on the same 16,000 held-out episodes.

Exact query-only Bayes has mean NLL 0.485549. Exact Bayes using both labels and
query readings has mean NLL 0.300567, so the query-only gap is 0.184982 with a
paired finite-test-set SE of 0.004321. For each plateau run I computed
`model NLL - query-only Bayes NLL` episode by episode:

| Plateau cases | Paired mean difference | Per-run SE | 95% interval entirely below zero |
|---|---:|---:|---:|
| Four cold-start mixture cases | -0.03861 to -0.04109 | 0.00215 to 0.00220 | 4/4 |
| Two trapped `w16_h2_L2` Gaussian cases | -0.02319, -0.02288 | 0.00283, 0.00286 | 2/2 |
| Genuine `w16_h2_L2` escape | -0.14200 | 0.00418 | 1/1 |
| Fourteen micro cases | -0.00394 to -0.02246 | 0.00270 to 0.00290 | 13/14 |
| **All plateau cases** | 21 negative means | paired within each run | **20/21** |

The four mixture cases illustrate the correction behind the reported 0.144
gap: their model NLL is 0.4445-0.4469, roughly 0.040 below query-only Bayes,
while both remain about 0.144 above full Bayes. The two trapped Gaussian cases
also beat query-only Bayes by about 0.023 NLL.

Under the proper log score, the exact query-only posterior predictive minimizes
expected NLL among predictors that use only the query readings. A significant
negative paired difference therefore supports use of some nonquery context. It
does not identify which contextual field matters. Contextual readings alone can
inform reliability; other possibilities include observed labels, a pooled
residual statistic, or partial source binding. Each can beat query-only Bayes
while remaining far from full Bayes. The binding-disruption and source-resolved
attention metrics are aimed at the narrower source-binding distinction.

Writing `Q` for the three query readings and `C` for the remaining context, the
proper-score decomposition is
`E KL(p(Y|C,Q) || q(Y|Q)) = I(Y;C|Q) + E_Q KL(p(Y|Q) || q(Y|Q))`.
The empirical negative NLL difference is consistent with the conditional
information term being used, but it does not attribute that information to the
observed `y` fields.

These intervals describe finite held-out-episode uncertainty for each retained
checkpoint. The 21 models share the same test episodes, so the episode count is
not treated as 21 independent seed replications. One micro case,
`w4_h1_L2_attn_ds4` seed 0, has difference -0.00394 with SE 0.00273 and an
interval spanning zero; it is the only inconclusive individual comparison.

## Recommended endpoint and reporting contract

Use four mutually exclusive paths for each seed, based on the order of the
first qualifying events:

1. **Acquisition before plateau:** three consecutive validation gaps below
   0.10 before any qualifying plateau. If a qualifying plateau occurs later,
   retain and report that chronology.
2. **Plateau then escape:** ten consecutive evaluations in `[0.14, 0.18]`,
   followed later by three consecutive gaps below 0.10. Record the first point
   of each qualifying sequence and the third point when it became observable.
3. **Right-censored plateau:** a qualifying plateau without primary escape by
   the run's last evaluation. State the censoring horizon. Do not call this a
   permanent failure.
4. **Neither event observed:** no qualifying plateau and no acquisition by the
   run's last evaluation. State the censoring horizon.

Report the old single-point `<0.14` rule only for historical comparability.
Report three-point `<0.12` and `<0.05` sensitivity results. The fresh terminal
validation set can confirm the terminal gap, but it cannot reconstruct a
three-point event.

Counts need separate denominators by width and intervention. For each
width-condition cell, report plateau incidence among all seeds, acquisition
among all seeds, and escape among plateau entrants. Wilson intervals are
appropriate descriptive intervals. Pair conditions by initialization seed and
report discordant seed counts at a common update horizon. For the batch-1024
arm, also compare at common episode exposure. A changed batch size otherwise
changes both gradient noise and the number of episodes seen per update.

At 50 seeds, a single proportion still has a worst-case 95% Wilson interval
roughly 27 percentage points wide. The paired discordance is more informative
for intervention effects than comparing two unpaired percentages.

## Instrumentation

The fixed 256-episode early probe should retain these measurements at steps 0,
1k, 3k, 5k, and 10k:

- layer-1 attention from each source's query-reading token to the four labelled
  tokens of the same source, separately by source and head;
- layer-2 query-token attention to each of the three source query readings,
  separately by head and source;
- validation NLL change after cycling labelled-token source identities while
  leaving readings, labels, and query-reading identities fixed;
- distance or projection of the predicted mean between query-only Bayes and
  exact Bayes on the same probe;
- unclipped gradient norm, clip fraction, update-to-parameter norm, and Adam
  bias-corrected denominator quantiles for first-layer Q/K and source-ID input
  weights.

The source and head axes matter. Summing the three layer-2 query-reading weights
and averaging heads can hide a specialized head or source-selective binding.
Store scalar summaries for quick reporting and the small per-head/source arrays
for the predictor audit. Log cumulative episode exposure as well as updates.

For early prediction at landmark `k`, exclude seeds already acquired by `k`.
Define the outcome as later acquisition by the observed horizon. Report AUC only
when at least five eligible seeds are in each outcome class, separately by
width and condition. Pooled results should condition on width and intervention.
With only one historical escape, no early quantity currently has validated
predictive value.

## Review of the proposed paired implementation

Static inspection supports the vectorized training core. Each stacked model has
its own parameter slice and elementwise Adam state. The summed loss produces one
independent mean-batch gradient per slice, and `clip()` computes and applies a
separate norm per model (`study.py:21-47,150-190`). Resetting the explicit GPU
generator for each condition and drawing a full batch of 1,024 before taking the
batch-256 prefix pairs the data streams as specified (`study.py:50-69,150-157`).

The revised preregistration fixed the historical horizon conflict before cloud
results: its priority menu starts at 80k and 60k, and the primary endpoint is
three points below 0.10 (`PREREG.md:36-66`). The analysis script at
`../work/astra-tiny-pfn-plateau-20261028/analyze.py` implements that endpoint,
the weak historical rule, right-censoring, Wilson intervals, paired update and
episode-exposure discordance, threshold sensitivities, and landmark-restricted
AUC. Its endpoint functions reproduce the retained counts exactly: 95 runs, 21
entrants, five weak crossings, and one sustained escape.

At 11:05 UTC, `cpu-conformance.json` reported `passed: true`. Root isolated the
earlier parameter discrepancy to the null key-projection bias, a
softmax-invariant gauge recorded separately by the preregistered conformance
test. Root had launched the width-32 STANDARD VM; width-16 launch attempts were
still retrying after stockouts. The conditions execute in priority order
against one absolute deadline, so unexpected runtime can censor or omit later
arms more often. The pre-run timing decision and external hard stop must be
preserved, and paired comparisons must use only observed pairs at their common
horizon.

Historical single-process CPU medians imply about 19.4 minutes for a 60k
`w16_h2_L2` run and 27.5 minutes for `w32_h2_L2`. Fifty seeds across three arms
and both widths would be about 117 historical CPU-hours. Those old shared-host
times do not predict a vectorized L4. The preregistered 50-model L4 benchmark,
measured before outcomes, is the appropriate admission check for the four-hour
and $10 caps.
