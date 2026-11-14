# Start here: MATS research in this repo

This page is meant to sit at `mats/START-HERE.md` (not committed as of the check below); repo links are relative to
that path and resolve only once it is there. The repo and the site are private; the site needs the access link Kelvin
shares. Items marked *(local only)* are in the MATS outbox on the research machine; repo sync 18, which is pending,
would bring them in. Items marked *(outside the repo)* live in other workspaces, and no sync for them is planned.

Checked on 3 November 2026 against `main` at `7a10d6a1` (PR #27). The answers below summarise each group's reports. They
are not new checks of the numbers.

**Research question.** How do learned systems combine evidence into coherent beliefs? How close do they get to exact
Bayes, and can we explain or estimate what a network computes without brute-force sampling?

**Terms used below.**
- *ARC*: the Alignment Research Center. Its contest asks for a way to estimate the average output of random deep
  networks more accurately than sampling, for the same compute (the *contest budget*).
- *Log loss* measures how much probability a forecast gave to what happened; a *nat* is its unit. Smaller is better.
- A *seed* is one training run from a different random start.
- *Pre-registration* (or a *frozen protocol*): the prediction and the pass rule are written down before the run.
- *M1* and *M2* are the two targets of the late-October program: M1 is ARC's estimation problem, M2 is explaining how
  small transformers approximate Bayes.

## Where things are

| What | Where | Notes |
|---|---|---|
| Conversations | [removed] | Not part of the public copy |
| Research log | planned, not written yet | Dated log of what was run and decided. Only drafts exist so far |
| Concepts | [insights page](https://mats.ekelvin.com/insights/) | Five concepts on search and aggregation, no results |
| Direction | [direction page](https://mats.ekelvin.com/direction/) | The direction note (v7) behind group 3 |
| Reports | [`reports/`](reports/) | One note per experiment or study, 26 Oct to 31 Oct. Reviews sit next to the report they cover |
| Interpretability lanes | [`interp/`](interp/README.md) | One folder per lane: preregistration, code, results, reviews |
| Worlds | [`worlds/`](worlds/README.md) | Synthetic worlds with exact Bayes, training code and runs for the small transformers |
| Pages | [`pages/`](pages/) | Source for the interpretability page, the World FA page and the scale-free explainer |
| Folder guide and status table | [`README.md`](README.md) | Per-workstream status and the reviews behind it |
| Off-repo files | [`ARTIFACTS.md`](ARTIFACTS.md) | Checkpoints and arrays that stayed on the research machine, with sizes and hashes |
| Earlier search and aggregation work | [`../docs/`](../docs/), [`../results/`](../results/), [`../aggbench/`](../aggbench/README.md) | Direction notes, specs, papers, audits, the search rounds and the aggregation harness |
| Site | [mats.ekelvin.com](https://mats.ekelvin.com/), [experiments index](https://mats.ekelvin.com/experiments/) | 232 entries in 6 streams, including "Readings and side experiments" (14). Covers work up to 20 October; groups 5-8 have no results page |

## The 8 groups

Status words: **positive**, **mixed**, **negative**, **parked**. Most work is exploratory. From about 20 October most
lanes registered their predictions before running and had a model review before making a claim.

### 1. Coherence of LLM forecasts (Aug to 10 Oct)
- **Question:** does making an LLM's forecasts consistent, at answer time or through training, make them more accurate?
- **Answer so far (mixed):** projecting answers onto a consistent set cut log loss on synthetic question sets, but gave
  no significant gain on 347 real forecasting questions. Fine-tuning toward consistency gave a small edge.
- **Links:** [site stream: Coherence of forecasts](https://mats.ekelvin.com/experiments/stream-coherence.html),
  [site stream: Training and running forecasters](https://mats.ekelvin.com/experiments/stream-forecasters.html),
  [forecasting bot](https://mats.ekelvin.com/forecast/). The reports are in a separate workspace *(outside the repo)*.

### 2. Markets inside networks (September)
- **Question:** can a network's parts be paid like traders, and does that help it learn or think longer?
- **Answer so far (negative, parked 30 Sep):** across 19 preregistered experiments, training this way never beat
  backprop. An early depth result turned out to come from the random training order.
- **Links:** [site stream: Markets inside networks](https://mats.ekelvin.com/experiments/stream-markets.html)
  (archived), [stop note](https://mats.ekelvin.com/experiments/trader-theory-4.html). The catalogue is
  *(outside the repo)*.

### 3. Learned search over a world model, "inconsistency search" (5 Oct to 2 Nov)
- **Question:** can a network learn a world model from outcomes and then improve its forecasts by searching for, and
  removing, inconsistencies in its own beliefs?
- **Answer so far (mixed, mostly negative):** early on, ShipHunt training through chains of reveals reached the same
  test loss with about 0.28x the data *(outside the repo)*, and the rounds 4-9 aggregators and a belief-drift result
  under GRPO were positive. The rounds 10-25 campaign produced no capability claim that survived audit; the first
  frozen rounds (R27) were negative except a weak N-Queens pass. A small follow-up found that a recurrent graph network
  trained only on a hand-written consistency energy, with no labels, builds valid breadth-first search trees in 2 of 3
  seeds. That is a toy result.
  The Löb/PrudentBot side track is parked.
- **Links:** [direction note v7](../docs/direction/research-direction-note-v7-2026-10-19.md)
  ([site](https://mats.ekelvin.com/direction/)), [critique of the rounds 10-25 campaign](../docs/audits/gemini-campaign-critique-2026-10-19.md),
  [protocol](../docs/PROTOCOL.md) and [R27A audit](../docs/audits/R27A-audit.md),
  `toy-search-followup-2026-11-02.md` *(local only)*.

### 4. Learned aggregation against exact Bayes (20 to 29 Oct)
- **Question:** can a learned aggregator over agents' messages, or a small transformer reading all the evidence, match
  exact Bayes on worlds where Bayes is computable?
- **Answer so far (mixed; parts parked):** in separate small set-ups, learned role keys and rich agents came within a
  few thousandths of a nat of the best achievable loss. The main bar was that short, blind messages match one
  transformer reading everything; the best attempt still missed it by 0.02-0.05 nats.
  Small transformers remove part of a duplicated piece of evidence, never all of it. World FA and the W3 chains world
  are parked.
- **Links:** [aggregation harness](../aggbench/README.md), [worlds](worlds/README.md),
  [deduplication report](reports/pfn-dedup-depth-2026-10-26.md),
  [World FA page](https://mats.ekelvin.com/fa/). All 55 stage-1 reports and the gate reviews are *(local only)*.

### 5. M2: how small transformers approximate Bayes in context (27 Oct to 2 Nov)
- **Question:** how do small transformers approximate Bayes in context, and can the architecture predict where they
  fall short?
- **Answer so far (positive, toy worlds only):** in nets with a mixture head, about 29% of the gap to Bayes comes from
  how they treat the pattern of a source's past errors, mostly the signs. Which of two solution types a net ends up
  with is set by its initialisation and tracks the number of heads in its first block. Nothing has been tested across tasks or on LLMs yet.
- **Links:** [program and bar](reports/cm-interp-program-2026-10-28.md),
  [phase B report](reports/cm-interp-phaseB-2026-10-28.md) (repo copy goes up to episode E76; E77-E86 are *(local only)*),
  [phase B lane](interp/phaseB/), [interpretability page](https://mats.ekelvin.com/interp/) (a proposal, not results).

### 6. M1: ARC mean estimation, measured (26 Oct to 2 Nov)
- **Question:** can any method beat sampling at estimating the average output of ARC's deep random networks?
- **Answer so far (negative for the methods tested):** on ARC's 256-wide, 32-deep Phase 1 networks, at the contest
  budget, randomised quasi-Monte Carlo sampling (a better-spread form of sampling) beat every method tried that fits
  the budget, and hybrids built on it only matched it. On the shallower Phase 2 shape (1024 wide, 16 deep), ARC's own
  order-3 method is about 19x more accurate than it. The one confirmed win is on rare-event tails at depth
  8. A reviewer ruled that this falls short of showing the problem is hopeless.
- **Links:** [ARC noisy deep networks](reports/arc-noisy-deep-2026-10-29.md),
  [unrestricted estimators](reports/unrestricted-estimators-2026-10-31.md), [lane](interp/arc-noisy/),
  `synthesis-draft-2026-11-02.md` *(local only; the best single summary, still a draft)*.

### 7. M1: ARC mean estimation, theory (29 Oct to 2 Nov)
- **Question:** why is beating sampling hard on these networks, and what information at a given layer does the output
  depend on?
- **Answer so far (mixed: it explains, it does not estimate):** proven that, with only runs of the network, sampling is
  optimal up to about a factor of 4, and that in the worst case no deterministic method matches it under a standard
  complexity assumption. One working premise was refuted: that summarising the front half of the network needs
  exponentially many numbers. Layer-to-layer maps look almost linear, but a deliberately scrambling control looks the
  same, so this is not yet evidence either way on whether information gets scrambled.
- **Links:** [hardness](reports/hardness-mean-estimation-2026-10-29.md),
  [cut-rank lower bound](reports/cut-rank-lower-bound-2026-10-30.md), [theory summary](reports/summary-2026-10-31.md),
  `bit-accounting-2026-11-02.md` *(local only)*.

### 8. Scale-free information combination (29 to 30 Oct)
- **Question:** does combining evidence need a densely connected network, or can a hierarchy that combines sources
  only through sums do the job? How does this relate to Richard Ngo's scale-free agency?
- **Answer so far (positive, toy scale):** when told which tokens belong to which source, a sum tree with one broadcast
  round matches the dense transformer at equal data. Dense attention is needed only to route tokens and to find
  hidden shared errors. 5 of 7 registered predictions held.
- **Links:** [experiments](reports/scale-free-exp-2026-10-29.md), [theory](reports/scale-free-theory-2026-10-29.md),
  [comparison with Ngo](reports/scale-free-agency-ngo-2026-10-29.md),
  [explainer source](pages/scale-free-explainer/) (not on the site).

## Known gaps

- Work from 1 and 2 November (groups 3, 5, 6 and 7, including phase B episodes E77-E86) is not in the repo yet.
- All 55 stage-1 aggregation reports (group 4) are local only.
- Groups 1 and 2 have no reports in the repo at all; they live in other workspaces, as does the ShipHunt result.
- The research log is not written yet.
- The site stops at 20 October. Groups 5-8 have no results page.
- The repo copies of the phase B report and the running overnight report are older than the local copies.
