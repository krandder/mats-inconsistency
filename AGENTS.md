# AGENTS.md — `futarchy-fi/inconsistency-search`

[removed]

## Repository Conventions
1. **No AI credit in commits or PRs** (Fleet Policy 9).
2. **Safe delete:** Never use `rm -rf` outside `/tmp`; use `trash-put` (Fleet Policy 14).
3. **Claims check themselves:** Every result table, Elo rating, and Mini-ECI score in `results/` and `benchmarks/` must name the exact script and seed log that produced it (Fleet Policy 15).
4. **Bitter-Lesson Purity:** Avoid game-specific hardcoded depth/wave/stall thresholds in the core `src/search/` and `src/models/` architecture; all search control (`SELECT_NODE`, `STEP`, prospectus $p_r$, memory $M_t$, and consensus clearing) must be general across the environment suite.

## Research protocol (binding from 2026-10-19; owner opus-mats)
Full text: `docs/PROTOCOL.md`. In short:
- gemini is proposer and coder only: PREREG text, runners, environments, baseline loaders, bug fixes. It never freezes, launches GPU runs, scores, audits, edits results files, or writes papers or README numbers.
- opus-mats freezes each round (PREREG + runner + `PREREG.sha256` in one commit tagged `freeze-Rnn`) before any VM starts; eurisko or calibrated-ai run it; a script scores it from the JSON (PASS/FAIL only); an independent Opus 5.5 auditor writes `docs/audits/Rnn-audit.md`; only audited results are published.
- Baselines and benchmarks come from public implementations, weights and datasets, run with their published recipes; re-implementations and home-made generators under public names are not allowed.
- Every round has the controls of PROTOCOL.md section 4 and identical inputs, priors, solvers and budgets for every arm.
- The society reviews code with tools on and produces findings with file:line evidence; no rubric scores, scoreboards or consensus numbers; gemini never audits or adjudicates its own rounds.
