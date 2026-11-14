# Jev exploratory evidence — 2026-09-17

Two separate experiments: 26 synthetic fleet-triage calls (`report.md`, `results.json`) and 73 architecture/coherence diagnostic calls (`api-experiments.md`, the three probe/control result files). All successful responses resolved to jev-1.13.0. Synthetic inputs only; no fleet transcripts or credentials are included.

The later original/cohort benchmark is a separate run; see `../../data/benchmark-runs/typesafe-20260917-original.receipts.jsonl` and the project's benchmark methodology. Do not merge these diagnostic observations into benchmark scores.

Scripts were copied from the experiment directory, with any hardcoded credential-file read replaced by the `TYPESAFE_API_KEY` environment variable for reproducibility. Running collectors makes new billable API calls and overwrites their result files; run copies in a fresh directory if intentionally repeating. No new calls are needed to inspect the recorded evidence. `analyze_probes.py` computes local summaries from the archived diagnostic results.

Research hypotheses are explicitly separated from public disclosures and measured behavior in `research.md` and `consistency-relevance.md`. Prices are rate-based estimates, not invoice totals.
