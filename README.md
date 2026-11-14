# mats-inconsistency

Research record from Kelvin Azevedo Santos's MATS autumn 2026 project (mentor: Deger Turan, Metaculus).
The work started from the logical consistency of LLM forecasts and moved through learned search over
any-order world models, learned information aggregation measured against exact Bayes, the mechanism
of in-context Bayesian inference in small PFN-style transformers, and mean estimation for deep random
MLPs (ARC's problem).

This is a working record, not a paper. Read it with that in mind:

- It includes negative results, parked lines, drafts and plans alongside the results that held up.
- Most experiments were designed, run and written up by AI agents. Many are exploratory and
  unreviewed. Where a pre-registration, an audit or a review exists, the file says so; a result
  without one should be treated as unverified.
- The audit of rounds 10-25 (`docs/audits/gemini-campaign-critique-2026-10-19.md`) found that no
  capability claim from that campaign passes a pre-registered rule. The original README of that
  campaign is kept as `README-rounds-10-25.md` with the audit note at its top.

## Where to start

- `mats/START-HERE.md`: map of the MATS research by group, with what each group found.
- `mats/log/RESEARCH-LOG.md`: week-by-week research log (August to early November 2026).
- `mats/reports/research-inventory-2026-11-03.md`: inventory of the experiment groups.
- `docs/PROTOCOL.md`: the research protocol used from 19 October 2026 (freeze, run, score, audit).

## Directories

| Path | Contents |
|---|---|
| `mats/` | Work from 26 October on: `reports/` (results notes, literature notes, plans, reviews), `interp/` (interpretability and mean-estimation lanes, with code, logs and result files), `worlds/` (synthetic worlds with exact Bayes posteriors), `pages/` (HTML explainers), `log/` (research log), `tools/` and `infra/` (helper scripts) |
| `aggbench/` | Aggregation benchmark: forecasters with rich messages and aggregators scored against exact Bayes ceilings |
| `benchmarks/` | Pre-registrations and runners for rounds 11-29 |
| `results/` | Result tables and JSON for rounds 10-27 and the exploratory runs (`results/explore/`) |
| `rounds/r4-r9/` | Earlier experiments: logical-induction networks, LIR agents, internal-market puzzles, micro-Stratego, linear aggregators |
| `src/` | Environments, models, search and evaluation code |
| `docs/` | Protocol, specs, direction notes, literature notes, write-ups and their audits |
| `society/` | Multi-model critique rounds of the rounds 10-25 campaign |
| `outbox/` | Pre-registration and freeze-review notes for rounds 26-29 |
| `site/` | Sources for the results site pages |
| `scripts/`, `tests/`, `explore/`, `external/` | Runners, tests, exploratory code and patches for third-party code |
| `consistency-ft/` | The earlier consistency work: benchmarks of LLM forecast consistency, answer-time projection, training pilots (copy of a separate repository; see its `PROVENANCE.md`) |
| `consistency-ft-runs/` | Analysis outputs of consistency-ft training runs kept outside that repository (8-arm summary of 23 September 2026) |
| `minibench-backtest/` | Pre-registration, results and aggregate analysis of the 347-question MiniBench backtest |
| `trader-theory/` | Theory notes on markets inside networks; `THEORY-4.md` is the closing review |
| `trader-economy/` | "Trader economy" experiments on payment rules inside small networks, and the Market ResNet runs |
| `eurisko-study/` | wealth-net (wealth accounting in a transformer), traders-net and logical-soup (evolved traders in a prediction market) |
| `mats-outbox/` | Further research reports, designs, gate reviews and literature notes from October and November 2026 |
| `deep-study/` | Explicit mesa-optimizer toy studies of 10 November 2026, each with an independent rerun under `check/`, and the agenda draft |

Each directory added from outside the original repository has a `PROVENANCE.md` saying where it came from and
what was left out.

The ARC mean-estimation work has its own repository, https://github.com/krandder/arc-mean-estimation; the
copies of those lanes under `mats/interp/` and `mats/reports/` here are the working record.

## What was left out of this copy

This repository is a filtered copy of private working repositories and working folders, published with fresh history.

- Conversation records (call transcripts and notes, agent session transcripts, message excerpts)
  are not included. Lines in the remaining files that summarised them are marked `[removed]`.
- Model checkpoints, pickled training states, arrays, files over 5 MB, paper PDFs and text copies
  of papers, and third-party code vendored without a licence (`external/infiloop/`, which
  `benchmarks/run_r27a_labelfree_infiloop_trmp.py` imports) are not included. Files that refer to
  them (for example `mats/ARTIFACTS.md`) describe material that is not here.
- Machine addresses, cloud project names and some local paths were replaced by placeholders
  (`[ip-removed]`, `GCP_PROJECT`, and similar), so scripts with hard-coded paths will not run
  unchanged, and SHA-256 values recorded for files that contained such strings no longer match.
