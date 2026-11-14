# Does consistency across linked questions help a strong model that searches the web itself? (2026-10-01)

Status: **pre-registered** 2026-10-01, before any forecast call (bead pool-4obqp4). The Results section is
filled in after the run; everything above it stays as written (any change is listed under "Deviations" with
its reason).

## Question

Projection onto the 12 Boolean forms of a pair helps weak models a lot and Opus 5.5 with no tools a little
(all-12 gain −0.005 [−0.008, −0.003], `docs/strength-scaling-20260930.md`). The MiniBench backtest gave its
consistency arm about nothing. Kelvin's question: does consistency **across genuinely linked questions** improve
a strong model's forecasts when the model **searches the web itself** at forecast time? His decisions: 100
pairs; the model queries Exa at runtime and writes its own queries in every call; every query is saved.

## PREREG

### Questions and pairs (built before this commit; outcome-blind)

Script: `scripts/linked_pairs_build.py` (`questions`, `generate`, `candidates`, `judge`, `select`); outputs in
`/home/agent/data/linked-pairs-20261001/`. Nothing in it reads an outcome: the Metaculus list API returns
`resolution = null` for every post, Manifold resolutions enter only as "YES or NO (not N/A)" in SQL, and the
MiniBench `truth/` files are not opened.

- **Sources.** Metaculus binary questions and binary subquestions of question groups (read-only API, resolved
  status; 700 binary posts plus the question groups); MiniBench rounds (the same API, tournament slug contains
  "minibench", plus the bt-scale snapshots); Manifold binary markets and multiple-choice answers from the as-of
  database (`/home/agent/data/manifold-asof-20260923/asof.sqlite`), with at least 15 distinct bettors before
  t and none of the personal/joke/random groups.
- **Resolution criteria.** The Metaculus API returns empty `resolution_criteria`, `fine_print` and
  `description` for every resolved post (checked on the list and detail endpoints and on the 2026-09-20 API
  dump). Metaculus questions are therefore shown with their **title only** ("Resolution criteria: not
  available"). MiniBench questions use the criteria of the bt-scale snapshots (taken while open; 174 of 177).
  Manifold questions use the title, description and answer text **as of t** (`market_versions`,
  `answer_versions`).
- **End and t.** end = the earlier of the actual and the scheduled resolution (Manifold: close or resolution).
  For a pair, **t = 00:00 UTC of (earlier end − 7 days)**, or the first 00:00 UTC after both questions opened
  if that is later. (MiniBench questions close for forecasting about 3 h after opening, so their forecasting
  close cannot be used.) A pair is feasible if t ≥ 2026-07-01 (after Opus 5.5's June 2026 knowledge cutoff),
  t is at least 1 day before the earlier end, neither question has resolved by t, and every Manifold market is
  still open for trading at t (its close date as of t is after t).
- **What the model and the judges see as the date.** Never the actual end (an early end reveals an early
  resolution; 185 of 700 Metaculus binaries resolved before their scheduled day). They see the **scheduled**
  date: Metaculus/MiniBench `scheduled_resolve_time` (unchanged by resolution: 179 of 179 MiniBench snapshots
  agree with the API), Manifold close date as of t.
- **Candidates** (each with a feasible t; identical texts dropped):
  - structural: up to 3 random sibling pairs per Metaculus group or Manifold multiple-choice market;
  - lexical: TF-IDF title cosine ≥ 0.20 (Manifold) or ≥ 0.10 (Metaculus, MiniBench, whose titles are
    topically diverse), the 5 nearest per question, within a source;
  - generated (MiniBench and Metaculus only): Opus 5.5 (effort medium) lists linked pairs from the titles and
    scheduled dates of one MiniBench round or one window of 60 Metaculus questions (titles only, no outcomes).
- **Link judge** (Opus 5.5, effort medium, outcome-blind; batches of 20; at most 450 random candidates per
  source): link 0–3 (3 logical constraint, 2 dependence of about 10+ percentage points, 1 same topic, 0
  unrelated), near_duplicate, forecastable. Kept: forecastable, not a near-duplicate, link ≥ 2.
- **Selection** (seed 20261001): targets Manifold 34, Metaculus 33, MiniBench 33; a source short of its target
  passes the shortfall to Metaculus, then Manifold, then MiniBench. Each question in at most 2 pairs; at most
  3 pairs per structural cluster. A seeded run order. One leftover eligible pair is the smoke pair.

**Counts at each step** (`selection-counts.json`):

| step | Manifold | Metaculus | MiniBench |
|---|---|---|---|
| questions with end ≥ 2026-07-08 | 527 | 235 | 177 |
| candidates with a feasible t | 920 (60 structural, 860 lexical) | 257 (5 structural, 41 generated, 211 lexical) | 91 (58 generated, 33 lexical) |
| judged | 450 (random cap) | 257 | 91 |
| dropped: link < 2 | 223 | 215 | 57 |
| dropped: near-duplicate | 14 | 5 | 0 |
| dropped: not forecastable | 4 | 0 | 0 |
| dropped: a market closed at t | 5 | 0 | 0 |
| eligible | 204 | 37 | 34 |
| **selected** | **35** | **35** | **30** |

MiniBench fell 3 short of 33 under the 2-uses rule; the shortfall went to Metaculus (+2) and Manifold (+1).
Selected links: 33 at link 3, 67 at link 2. 154 distinct questions, 46 of them in 2 pairs. t falls in July
(51), August (42) and September (7). The pair list is committed with this doc:
`docs/linked-pairs-search-20261001-pairs.json`. Smoke pair (not in the sample): `mf:153707|mf:18105`.

### Calls

Script: `scripts/linked_pairs_search.py` (`smoke`, `run`, `truth`, `analyze`, `spend`).

- **Model.** `claude -p` pinned to **claude-opus-5-5**, `--effort medium` (as MiniBench), through Prime
  (`127.0.0.1:7800`, tag `token-proxy-managed:calibrated-ai`), `CLAUDE_CODE_NO_MODEL_FALLBACK=1`, no session.
  The result's `modelUsage` must be exactly `["claude-opus-5-5"]`; anything else, or a model-not-found error,
  **stops the run** (no other model is ever used).
- **One tool.** No built-in tools (`--tools ""`, so no web search or fetch); one MCP tool, `exa_search(query,
  days_back?)` (`scripts/exa_mcp_server.py`), allowed by name. The cutoff = t is enforced **in the tool**, by
  the MiniBench strict backtest path (`mb/research.py`): Exa `endPublishedDate` = t − 36 h, then
  `leak_filter(strict=True)` and `verify_dates`; only results that pass both reach the model (up to 6 results,
  1,000-character highlights each). **At most 4 searches per call**; a 5th is refused with "answer now".
- **Queries saved.** Every query goes to `/home/agent/data/linked-pairs-20261001/queries.jsonl`: call id,
  cutoff, query string, `days_back`, the full Exa request filters, every returned result (URL, title, published
  date, author, highlights, kept or dropped and why), counts, the sha256 of the text shown to the model,
  Exa cost. Raw replies (assistant text, thinking, tool calls) per call in `raw/<call_id>.json`.
- **Prompt.** v7b_explicit (`training/verbalized_baseline.py`: `BASE_SYSTEM`, `question_block_v7`,
  `explicit_statement(..., negation_note=True)`), as in the strength-scaling run, with three changes: the
  question block's date line reads "Scheduled close date: YYYY-MM-DD." (the scheduled date above); the
  research instruction "You have one tool, exa_search, which searches news and web pages published before
  today. Before you answer, research the question(s) with it: write your own search queries (at most 4
  searches)."; and the answer instruction "When you are done, reply with exactly one line of this form and
  nothing else: Probability: 0.xx". No " /no_think". The frame "Assume today is t. Use only information
  available up to that date; do not use knowledge of later events." comes first. Singles show only their own
  question; compounds show both.
- **Per pair 16 calls:** the 12 forms once each (sample 0) plus samples 1 and 2 of P and of Q (independent
  calls, identical prompts). 1,600 calls, about 10 in parallel, in the seeded run order (a budget stop leaves
  a random prefix of pairs).
- **Parsing.** `parse_probability`; an unparseable reply gets the protocol's one retry with `RETRY_REMINDER`
  appended to the prompt (a fresh call, its own searches, logged as `<call_id>-retry`). Still unparseable =
  missing.
- **Context every call shares (cannot be removed; the same for all arms and as in MiniBench).** This host's
  managed SessionStart hooks inject the fleet `AGENTS.md` (about 9k characters) into every Claude Code session,
  and Claude Code's environment block states the real date (2026-10-01); `disableAllHooks` and a separate config
  dir do not remove either. The "Assume today is t" frame and the tool's cutoff carry the as-of date.

### Arms (per question side X ∈ {P, Q} of a pair)

- **B** = median of the 3 plain answers for X (sample 0 and samples 1, 2); median of those present; 0.5 if none.
- **C_self** = the P/NOT P projection: (s_X + 1 − s_NOT X) / 2 from the sample-0 answers (each clipped to
  [0.01, 0.99]); one present → that one (or 1 − s_NOT X); none → 0.5.
- **C_pair** = the full 12-form lattice: the kl coherence projection (`scripts/coherence_projection.fit`, four
  atoms) fitted to the sample-0 answers present (missing forms left out of the fit), X's marginal read back.
- A pair is **complete** when all 16 of its calls have run; an answer still unparseable after the retry (or
  dropped by the leak audit) is missing, and the fallbacks above apply.

### Analysis (fixed now)

- **Truth** is fetched only after the main run (`truth`): MiniBench from the bt-scale `truth/` files (else the
  Metaculus API), Metaculus from the post detail (GET only), Manifold from the as-of database. Annulled,
  ambiguous or N/A questions leave their pair out (counted).
- **Primary: singles log loss, C_pair − C_self.** Units = the 2 sides of every complete, scored pair;
  probabilities clipped to [0.01, 0.99]. 95% CI from a **paired bootstrap clustered by question** (a unit's
  cluster is its question; 2,000 resamples of the questions with replacement, seed 20261001; ratio of sums).
- **Secondaries** (reported, not used for the bar): C_pair − B and C_self − B (singles log loss, same
  bootstrap); **joints**: mean log loss over the four AND forms of the C_pair atoms against the product of the
  B singles (and the stated AND answers against that product), bootstrap over pairs; **spread vs gain**:
  Spearman correlation of a side's spread (max − min of its plain samples) with its C_pair − C_self and C_pair −
  B, and the gain for spread = 0 vs spread > 0; **Brier** for every contrast; the primary per source; search
  behaviour (queries per call, refusals, kept results per query).
- **Bars** (primary, 95% CI):
  - **No gain**: CI lower bound ≥ −0.01 (checked first).
  - **Gain**: otherwise, CI upper bound < 0.
  - **Inconclusive**: otherwise.
  - **Problem** (overrides): P1 the model was unavailable or substituted and the run stopped; P2 more than 5% of
    calls missing after the retry; P3 fewer than 90 complete, scored pairs (e.g. a budget stop); P4 the
    independent verifier's recomputation differs from a headline number by more than 0.001; P5 the leak audit
    finds post-t information in 2 or more of its 20 calls.
- **Size.** 100 pairs = 200 singles in ≤ 154 question clusters. With a per-pair SD of the difference of about
  0.04 nats, n = 100 detects about 0.011 nats (80% power, two-sided 5%). If the true gain is 0, the "No gain"
  bar is met about 70% of the time at this n.
- **Leak audit.** An independent agent reads the kept results (dates, titles, highlights as shown) of **20
  random calls** (seed 20261001, among calls with at least one search) and the call's t, and marks each call
  "post-t information: yes/no" with evidence. Calls marked yes are dropped (counted as missing) and reported;
  2 or more is Problem P5.
- **Verifier.** An independent agent recomputes the headline numbers (arm log losses, primary with CI, C_pair
  − B, joints, missing counts) from `calls.jsonl`, `truth.jsonl` and `pairs.jsonl` without this script's
  analysis code; a difference above 0.001 is Problem P4.

### Budget

Kelvin's cap: **$250 at list price, LLM plus Exa; Exa cash tracked separately.** LLM cost = Claude Code's
`total_cost_usd`; Exa = `costDollars` per search. The run reserves $0.12 per call in flight and stops before a
call that could pass the cap (`costs.jsonl` + `exa_cost.jsonl`, plumbing, generator and judge included). Ledger:
`calibrated-ai/BUDGET-20261001.md` (rows with a running total). Spent before this commit: about $3.6 (plumbing
tests, generator, judge). Expected: $0.05–0.10 per call plus about $0.025 of Exa, so about $120–200 for 1,600
calls.

### Smoke test

The smoke pair's 16 calls first: served model, tool use, query logging, parsing, cost per call. A change it
forces is listed under Deviations before the main run.

### Outputs

`docs/linked-pairs-search-20261001.json` (all numbers), this doc's Results. Data (not committed):
`/home/agent/data/linked-pairs-20261001/` (`calls.jsonl`, `raw/`, `queries.jsonl`, `truth.jsonl`,
`analysis.json`, `units.jsonl`, cost logs).

## Results

Main run 2026-10-01 about 04:33-05:29Z; numbers from `analysis.json` (copied to `docs/linked-pairs-search-20261001.json`).

**Bar: Inconclusive.** Primary (singles log loss, C_pair − C_self) = **+0.0047 nats, 95% CI [−0.012, +0.020]**
(192 units, 140 question clusters). The CI does not clear the No-gain bound (−0.01) and does not exclude 0. No
Problem condition fired (P1–P5 all clear). Point estimate: the 12-form lattice is slightly *worse* than the
P/NOT P projection.

- **Pairs.** 100 selected (Manifold 35, Metaculus 35, MiniBench 30), 100 complete, **96 scored** (Manifold 35,
  Metaculus 31, MiniBench 30); 4 Metaculus pairs out for annulled questions.
- **Calls.** 1,600 of 1,600 parsed (0 missing, 0 retries left unparsed), every call served by claude-opus-5-5.
- **Singles log loss (Brier).** raw 0.385 (0.130), B 0.378 (0.128), C_self 0.382 (0.127), C_pair 0.387 (0.128).
- **Secondaries.** C_pair − B +0.0089 [−0.007, +0.024]; C_self − B +0.0042 [−0.011, +0.017]; B − raw −0.0071
  [−0.022, +0.008]. Brier: C_pair − C_self +0.0004 [−0.006, +0.007], C_pair − B −0.0002 [−0.007, +0.006].
  **Joints** (4 AND forms, per pair): C_pair atoms 0.311 vs product of B singles 0.325, difference −0.014
  [−0.032, +0.004]; stated AND answers 0.317 (−0.008 [−0.035, +0.018]).
- **Spread vs gain.** Mean spread of the 3 plain samples 0.091; 8% of sides have spread 0. Spearman(spread,
  C_pair − C_self) −0.08, (spread, C_pair − B) +0.04. C_pair − C_self is +0.015 [+0.006, +0.029] when spread = 0
  (15 units) and +0.004 [−0.014, +0.020] when spread > 0.
- **By source (primary).** Manifold +0.015 [−0.002, +0.035], Metaculus +0.007 [−0.038, +0.046], MiniBench
  −0.010 [−0.028, +0.006].
- **Search.** 5,471 queries (in 1,574 of 1,600 calls; 3.4 per call), 534 refused 5th attempts, 7 tool errors,
  3.1 kept results per query. All 16,980 kept results carry a date before t − 36 h. Every query:
  `/home/agent/data/linked-pairs-20261001/queries.jsonl`.
- **Leak audit.** An independent agent read 20 random calls (`leak-audit/packet.json`, `verdicts.json`):
  **0 of 20** with post-t information, so nothing dropped. Borderline cases marked no: a live price page
  (cryptopricelist zcash) whose excerpt was crawled later than its Exa date but still before t, and a USA Today
  page updated after its Exa date with pre-t content. Weakness noted: live or updated pages keep their first-crawl
  date while the excerpt can come from a later crawl. (A second, full-reading pass of the same 20 calls was
  started as an extra check and did not finish; it is not part of the result.)
- **Verifier.** An independent agent recomputed counts, arm log losses, primary, C_pair − B and joints with its
  own code (`verifier.json`): **MATCH**, largest point difference 3e-9; its bootstrap CI [−0.011, +0.020].
- **Spend.** $129.29 at list price of the $250 cap: LLM $90.51, Exa $38.77 (Exa cash). The audit and verifier
  agents ran on the session's own budget, not this ledger.

Reading: with a strong model that researches each question itself, projecting onto the full lattice of linked
forms does not improve singles over the simple P/NOT P projection or over the median of 3 samples; the only
hint of value is in joint probabilities (−0.014 nats vs the independence product of B, CI touching 0).

## Deviations

- Before this commit, the candidate generator was run three times: first with each question's actual end
  date in the list (replaced by the scheduled date, because an early end reveals an early resolution), then
  twice with scheduled dates (a reply-parsing failure gave 0 pairs in 2 groups; the parser was fixed and every
  reply is now saved). Only the last run's pairs are candidates; the judge's earlier labels of pairs that are no
  longer candidates are ignored.
- Smoke test (2026-10-01 04:22-04:23Z, 16 calls on the smoke pair): 16 of 16 parsed on the first try, served
  model claude-opus-5-5 on every call, 4 searches in every call (13 calls asked for a 5th and were refused), 4.4
  results kept per search, none dated after t − 36 h; $0.073 LLM + $0.028 Exa per call. No change to the design.
- **Truth for Metaculus questions** (found during the main run, before any outcome of a sampled question was read):
  the post-detail API returns `resolution: null` for resolved questions on both of our read keys (Metaculus's
  2026-03-09 API change shows a closed question's resolution only to accounts that forecasted it; checked on 3
  non-sample questions). `truth` therefore reads the public question page instead, read-only, in one owned tab
  of the fleet browser on deploy-host (curl and the headless browser get Cloudflare's challenge). Primary source: the
  question data embedded in the page (per question id: status, resolution, actual resolve time; each id matched
  exactly once; saved in `metaculus-resolutions.jsonl`, 45 questions). Cross-check: the page header text
  ("RESOLVED" then "Yes"/"No", `metaculus-pages.jsonl`, 42 posts); `truth` stops on any disagreement. It agrees
  on all 38 questions where the header shows a verdict (the other 7 are subquestions of group posts, whose header
  has none; 3 of them were checked against the values shown on the page). Result: 19 Yes, 22 No, 4 annulled
  (mx:21258, mx:21259, mx:21260, mx:39337), so 4 pairs leave the analysis as the prereg's annulled rule says.
  Code change after the prereg commit: `cmd_truth` gained this page fallback (`page_resolution`); MiniBench
  (bt-scale `truth/`) and Manifold (as-of database) are unchanged. Caveat: Metaculus's terms ask for permission
  before their data is used to evaluate AI models.
- **Descriptive counts corrected** (text only; the committed pair list was always the sample and the run used it
  unchanged): the 100 pairs have 145 distinct questions, 55 of them in 2 pairs (not 154 and 46), and t falls in
  July (51), August (40) and September (9) (not 42 and 7); so "Size" means ≤ 145 question clusters. Where the wrong
  numbers came from was not traced.
