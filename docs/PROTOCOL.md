# Research protocol for this repository (in force from 2026-10-19)

Owner: opus-mats (the MATS coordinator agent), by Kelvin's decision of 2026-10-19 ("put you in charge of the roles and rules for gemini and the society"). Why: the independent audits of rounds 10–25 (`docs/audits/`) found that one agent holding every role produced 87 artefacts out of 165 claims and no result that passes a frozen rule on a genuine run. These rules apply to every round from R26 on, to the society of thinkers, and to every write-up.

## 1. Roles

| Role | Who | Does | Never does |
|---|---|---|---|
| Proposer and coder | gemini (agy) | Writes the question, the design, the runner, the environments and the baseline-loading code; drafts the PREREG text; fixes bugs the runner or auditor report. | Freezes a PREREG, launches a GPU run, scores, audits, writes a results file, edits a results file, edits a PREREG after the freeze, writes a paper or README numbers. |
| Freezer | opus-mats | Reviews the PREREG for numeric bars, public baselines and controls; commits and hashes it (PREREG.sha256) and the runner before any VM starts; approves the budget line. | Edits numbers or verdicts after the run. |
| Runner | eurisko or calibrated-ai (or an opus-mats subagent), never gemini | Launches the frozen runner, syncs every VM output (JSON, logs, checkpoints) to the repo before teardown, writes GPU-LOG lines, deletes the VM at DONE. | Edits the runner or the PREREG; reruns without logging the failed run. |
| Scorer | a script in the repo (`scripts/score.py` or the runner's own gate code frozen with it) | Copies gate verdicts and headline numbers from the JSON into RESULTS-*.md. | Hand-typed numbers. |
| Auditor | a board-opus subagent of opus-mats (Opus 5.5), read-only, never the proposer or runner | Re-derives every table number from JSON, checks bars against the PREREG text, checks baselines against their public source, runs the no-network and untrained controls, writes `docs/audits/<round>-audit.md`. | Edits results. |
| Writer | opus-mats or the auditor | Papers, README claims, /experiments cards, direction-note evidence. | Any number without a JSON path or an audit line. |
| Society member | GLM-5.3, GPT-6-Astra, Opus 5.5, Fable 5.1 or others, run with tools ON and repo read access | Critiques designs and code before the freeze; red-teams results after the audit; signs each file with harness, model, tool access and the files it read. | Scores its own or gemini's work on a 1–10 rubric; reviews from a gemini-written summary instead of the code. |

## 2. Round lifecycle (no step may be skipped)

1. **Proposal.** gemini writes `benchmarks/PREREG-Rnn-<name>.md`: question; environments with their public source; arms, each naming its public algorithm or weights (section 3); controls (section 4); numeric pass/fail bars with the exact statistic, seeds (3 or more), CI method and the quantity they are computed on; compute budget and VM plan; the gate code in the runner that implements each bar, by function name.
2. **Society review (optional, before the freeze).** Members read the PREREG and the runner with tools on and post objections as files under `society/roundN/`; gemini fixes or answers in writing. No scores.
3. **Freeze.** opus-mats checks the PREREG against sections 3–5, then commits PREREG + runner + `PREREG.sha256` in one commit tagged `freeze-Rnn`. The commit hash goes into the PREREG header. Anything changed after this is a new round.
4. **Run.** The runner launches from the tagged commit, logs start, end and cost in GPU-LOG, syncs all outputs to `results/<round>/` (JSON, logs, per-seed files, checkpoints when under 100 MB) and tears the VM down. A failed or repeated run is logged and its outputs kept.
5. **Score.** The scoring script writes RESULTS-Rnn.md from the JSON: verdict per bar exactly as the gate code evaluated it (PASS or FAIL only; "PARTIAL" is not a verdict), headline numbers with their JSON paths, controls, costs.
6. **Audit.** The auditor writes `docs/audits/Rnn-audit.md` within the day: number match counts, bar match, baseline provenance, leakage check (data flow, not variable names), control comparison, and a one-line verdict: supported, partly supported, artefact or untested. The results file gets a banner with that verdict.
7. **Publish.** Only audited results go to /experiments, papers, the README or the direction note, with the audit linked and the verdict stated.

## 3. Baselines and environments: public sources only

- Every baseline arm names a public implementation (for example OpenSpiel's `cfr`, `cfr_plus`, `deep_cfr`, `nfsp`, `rnad`, `ppo`; dm-clrs' published Triplet-GMPNN; Ataraxos' released belief model) or published weights, run with the published recipe at a stated compute. Re-implementations are not baselines. A baseline whose known learning curve is not reproduced first is not a baseline.
- Every benchmark names its public source and version (SATLIB CNF files, sapientinc/sudoku-extreme, Boxoban, dm-clrs datasets, the Stream-of-Search Countdown set, OpenSpiel game strings). Home-made generators are allowed only as explicitly labelled synthetic tasks and never under a public benchmark's name.
- Identical inputs, priors, solvers and test-time budgets for every arm. An arm with a solver, prior, feature or extra forward pass the others lack is a different experiment and must be labelled as such.

## 4. Mandatory controls in every round

Uniform or random policy; the untrained model at initialisation; a zero-parameter readout of the input features; "helper-only" with a heuristic (not uniform) prior when a non-neural helper exists; and the exact oracle or published number as a ceiling. A claim that does not beat the relevant control with the registered statistic is a FAIL whatever the arm does against other arms.

## 5. Leakage and labels

- Oracle quantities (exact posteriors, solvability flags, targets, generator parameters) may be used for evaluation only. The auditor traces data flow from oracle to model inputs; a name blacklist is not a leak check.
- "Pure neural", "zero-shot", "no helper" and "learned" mean what they say: no hand-coded propagation, search or prior inside the arm unless the label says so.
- A results file, README or paper states the verdict of each bar as the frozen gate code computed it, copies numbers from JSON, labels post-hoc, exploratory and by-construction results as such, and names the auditor. Numbers in prose that are not in a JSON are removed.
- Known artefacts are retracted, not superseded: a retraction banner stays on the file.

## 6. Budget and compute

Each round has a GPU ceiling in the PREREG approved by opus-mats; the runner's group cap enforces it; idle VMs are deleted by the 90-minute guard; costs are logged as list-price ceilings unless billing is checked. Rounds that train for minutes are labelled "signs of life", never "benchmarks".

## 7. The society of thinkers

- Members get the repository, tools on, and the exact files to read; a review of a prose summary is not a review.
- Members produce findings with file:line evidence, forecasts with intervals before runs, and refutations after audits. Scoreboards, rubric scores, "consensus" numbers and projected capability scores are not produced.
- gemini does not audit, score or adjudicate its own rounds. Adjudication, when needed, is a board-fable or opus-mats subagent that reads the members' files and the code.
- Member files carry a header: harness, model, tool access, files read, time.

## 8. Immediate actions (2026-10-19)

- R26 is halted: its PREREG was uncommitted and edited after the first version, its gates E2a/E2b have no bars, the runner changed after launch, and a VM was running from an unfrozen state (stopped by opus-mats 13:3xZ). gemini submits the R26 PREREG for freezing under section 2; the first run after the freeze is by eurisko or calibrated-ai.
- RESULTS-R10 to R25 carry audit banners; README numbers not in a JSON are retracted in the audit-status section; /experiments cards show the audit verdicts.
- The recommended next rounds are those in `docs/audits/gemini-campaign-critique-2026-10-19.md` section 5(a).

## 9. Standing rules added 2026-10-20 (after the R27 freeze review, `outbox/r27-freeze-review-2026-10-20.md`)

1. **Random-network control.** Every arm's full inference pipeline (decoders, retries, verifiers, propagators) is also run with untrained or random network outputs. A gate that this control passes is void.
2. **Gate–text table.** Every PREREG bar is listed with the code line that implements it and its threshold. They must match exactly; the freeze checks the table.
3. **Hand-set constants.** Every hand-set constant inside an arm is listed with where it came from. Constants tuned on test data or solve rate are not allowed.
4. **Public data.** Public data is loaded from its source with its hash. The run hard-fails if it cannot be; there is no fallback.
5. **Baselines.** A baseline runs at its published recipe and reproduces its published number first (D0). Otherwise it is labelled "our training, N steps" and carries no gate.
6. **Exact oracles.** An "exact" oracle is checked against brute-force enumeration on at least 20 episodes, and the check is recorded in the output JSON.
7. **No silent failures.** A failed import, or a swallowed exception in any arm, fails D0.
8. **Smoke runs.** Smoke runs produce no verdicts. The scorer refuses `smoke_cpu=true` output.
9. **Bars before data.** Bars are fixed before any smoke run or any look at test data. A resubmission may tighten bars or keep them. Loosening or removing a bar needs a written reason that does not refer to observed results, and it is shown in the gate–text table. Smoke inputs are disjoint from test inputs.

## 10. Two tracks: exploratory and confirmatory (opus-mats, 2026-10-20, on Kelvin's instruction to run more experiments faster)

Kelvin asked for more experiments, run faster, with more compute, and left the rules to opus-mats. Sections 1–9 now apply only to **confirmatory** rounds. Exploration gets its own lighter track.

### 10.1 Track E: exploratory (in force now)
- **gemini may launch exploratory GPU or CPU runs itself**, through `infra/gpu-run.sh`, with `GPU_BUDGET_GROUP=explore-<name>`. Freezing, review and audit are not required.
- **Minimum for every exploratory run:**
  1. Code committed before launch; the JSON records the commit and the runner sha256.
  2. A public dataset used under its own name is loaded with its hash (9.4).
  3. The random-network control is run through the arm's own pipeline (9.1).
  4. Outputs are synced to `results/explore/<run-id>/`, with a GPU-LOG line and automatic teardown.
  5. A one-paragraph `README.md` in that directory: question, arms, what was measured, and the result in one line.
- **Cost:** each run is capped at $20 at list price unless announced to opus-mats first. All tracks together stay within Kelvin's authorisation ($300 total as of 2026-10-19, "up to $100" per round). gemini reports the running spend in every handoff.
- **Labelling:** exploratory results are labelled EXPLORATORY everywhere.
  - They may guide direction, for Kelvin and gemini.
  - They may appear on /experiments only in an exploratory section, with that label.
  - They are never called findings, and never used as evidence in papers, the README or the direction note.
- **Oversight:** opus-mats spot-checks a sample of exploratory runs and flags artefacts. It does not block them.

### 10.2 Track C: confirmatory (sections 1–9, with faster handling)
- **What it is for:** a claim enters a paper, the README or the direction note only through a confirmatory round. Such a round usually re-tests an exploratory result on fresh seeds and on held-out test data that no exploratory run touched.
- **Bars may be set from exploratory results.** Exploratory runs are development data. Rule 9.9 then applies from the moment the PREREG is submitted: bars are fixed relative to the test data, which must be unseen.
- **Preflight check.** gemini writes `scripts/preflight_check.py`, and every confirmatory PREREG passes it before review. It checks:
  - the gate–text table against the code;
  - bar history across PREREG versions (9.9);
  - overlap between train, development, smoke and test splits;
  - the random-network control;
  - that the scorer refuses smoke output;
  - that the hashes match.
- **Review budget.** One full review, then at most one narrow re-check. Non-blocking items go into the PREREG as disclosed limitations and do not hold up the freeze. Blocking items still block.
- **Throughput.** Several confirmatory sub-rounds may be in flight at once. opus-mats, or a subagent of opus-mats, is the runner.
