# Market game runner

A frozen frontier model (GPT-6 Astra or GPT-5.6 Luna through the Codex CLI)
plays a combinatorial prediction market over the 1,265 binary questions of
Turtel et al. 2025's released test set. Each session is one Codex invocation
that sees N questions with their current market prices and Turtel's frozen
evidence and answers with a block of trades against the bayes-market factored
LMSR engine. P passes cover every question; the market is scored against the
known resolutions after each pass. Nothing is trained; nothing is spent until a
real provider is selected.

Status 2026-09-22: built and exercised offline against the mock provider only.
No Codex call has been made. `--dry-run` prints the exact command.

## Files

| path | role |
|---|---|
| `harness/market_game.py` | library: dataset, engine wrapper, grouping, prompt, parser, providers, arena |
| `scripts/run_market_game.py` | CLI runner (mock / astra / luna, resume, dry run, smoke limit) |
| `scripts/score_market_game.py` | scores a run directory: `scores.json` + `scores.md` |
| `test_market_game.py` | 27 unittest cases (`python3 -m unittest test_market_game`) |
| `data/market-game/table1-verification.json` | Turtel Table 1 recomputed from the released file |

## Inputs

**Test set.** `LightningRodLabs/outcome-rl-test-dataset`, file
`data/train-00000-of-00001.parquet` (1,265 rows), mirrored into
`/home/agent/data/forecasting-benchmarks/LightningRodLabs--outcome-rl-test-dataset/`
with the archive's conventions (`source-metadata.json` with URL, fetch time,
revision `f200d538760aa94848f2bc803d5307f5552fa73f`, per-file sha256; the
parquet's sha256 `4a9ff156...259cf` matches its LFS oid). `--download` or a
missing file triggers the same mirror code (`market_game.ensure_dataset`);
existing files are never overwritten.

Table 1 reproduces from the file: Brier polymarket 0.1509 (paper 0.151), o1
0.2017 (0.202), ReMax 0.1907 (0.191); base rate 29.6% yes. The runner refuses
to start on the full set if this check fails.

Reading the parquet needs pyarrow, which only
`/home/agent/workspaces/venv-cft/bin/python` has (installed with uv on
2026-09-22). The first load writes `runs/market-game/questions-cache.jsonl`
keyed by the parquet's sha256; afterwards the stdlib `python3` runs everything.

**Engine.** `/home/agent/workspaces/bayes-engine/inference/`
(`FactoredMarket.from_nodes`, junction-tree LMSR). 1,265 independent binaries
at 0.5, liquidity b = 100, max_width 8. A marginal trade reweights one
variable; a conditional trade `P(A | B=yes)` reweights only the context slice
(I-projection: `P(A | B=no)` and `P(B)` are unchanged, `P(A)` reprices as the
mixture) and forces A and B into one cluster; trades that would push the
treewidth past the budget raise `JointMarketError` and are logged as
`engine_error` rejections with the engine untouched.

**Codex command.** `market_game.codex_command` is a verbatim copy of
`harness/run_native_checks.py native_command()` for astra/luna; the test suite
asserts equality with the original. Prompt on stdin, developer rule through
`-c developer_instructions=...`, final message through `-o`, the JSON event
stream on stdout (`thread.started` gives the thread id, the single
`item.completed/agent_message` the text, `turn.completed` the usage). Each
session gets its own empty `-C` directory under `<out>/codex/<session>/attempt-N/`
where stdout, stderr and the final file are kept.

## Running

```
# offline smoke, no provider calls
python3 scripts/run_market_game.py --provider mock --limit 20 --passes 3 --out runs/market-game/smoke20
python3 scripts/score_market_game.py runs/market-game/smoke20

# full mock (15 s wall on this box)
python3 scripts/run_market_game.py --provider mock --n-per-session 8 --passes 3 --workers 6 --out runs/market-game/mock-full

# exact command and first prompt, nothing executed
python3 scripts/run_market_game.py --provider astra --effort high --dry-run --out runs/market-game/dry

# real run (Kelvin's go required), resumable, optional usage watch
python3 scripts/run_market_game.py --provider astra --effort high --passes 3 --workers 6 \
    --out runs/market-game/astra-20260922 --quota-check-cmd '<cmd printing PAUSE when quota is tight>' --resume
```

Flags: `--provider mock|astra|luna`, `--effort` (Codex `model_reasoning_effort`,
default medium), `--n-per-session` (8), `--passes` (3), `--workers` (6),
`--seed` (0), `--limit K` (seeded random subset for smoke), `--resume`,
`--dry-run`, `--budget-factor` (0.3), `--cap-first-pass`, `--liquidity`,
`--max-width`, `--snapshot-every` (50), `--quota-check-cmd`, `--timeout`
(1800 s per invocation), `--data`, `--download`, `--table1-json`, and the
`--mock-*` knobs (flip rate 0.2, step 0.5, invalid 0.1, conditional 0.15,
big-jump 0.05, garbage 0.02, latency 0).

The dry-run command line for astra at effort high (paths under `--out`):

```
/home/agent/.local/bin/codex-any exec --ephemeral --json --skip-git-repo-check --sandbox read-only \
  -C <out>/codex/dry-run/empty --model gpt-6-astra -c 'model_reasoning_effort="high"' \
  -c 'service_tier="default"' -c project_doc_max_bytes=0 -c 'web_search="disabled"' \
  -c features.multi_agent_v2=false -c features.shell_tool=false -c features.exec_tool=false \
  -c 'developer_instructions="<developer rule below>"' -o <out>/codex/dry-run/final.txt -
```
with env `FLEET_ACTOR=FUTARCHY_AGENT_NAME=AI_AGENT_NAME=consistency-market-game-20260922`,
`TMPDIR=<session dir>`, and `ANTHROPIC_*/OPENAI_*/XAI_*/ZAI_*/ZHIPU_*/CODEX_API_*`
and `TMUX` stripped (same as the native checks).

## Session design

- **Grouping.** Questions are partitioned into groups of 8 (152 groups of 8
  and 7 of 7 for 1,265). TF-IDF cosine similarity over `question_text` (pure
  Python) gives each question its 20 nearest neighbours; sibling sets are
  detected by Polymarket event slug when one is present in the id or text
  (none in this file), else by structural event keys: `versus` (`... resolve
  to "X"? A vs. B`), `award` (quoted nominee + "win"), `numeric` (template
  with numbers replaced, dates kept; a set with a `between x-y` member is
  `bins`, otherwise `series`), plus a `similar` rule (cosine >= 0.8, same
  dates and numbers, not keyed to different structural events). 84 sets on
  the real data (53 bins, 20 award, 4 versus, 2 series, 5 similar; 394
  questions), 77 marked exclusive. Greedy clustering per pass: seeds in
  seeded random order, each gathering its sibling set and then its nearest
  unassigned neighbours (sibling sets travel as bundles, cores of >= 2 kept),
  leftovers dealt at random. The seed changes per pass, so a question keeps
  its siblings but sees different other neighbours (31% neighbour overlap
  between passes 1 and 2; mean within-group cosine 0.32 vs 0.03 random).
  Group order within a pass is shuffled with the seed.
- **Budget.** Pass 1 uncapped; later passes `eps_total = 0.3 * N` nats
  (2.4 for N = 8). KL per trade is `KL(target || pre)` of the traded slice
  (the marginal, or the conditional for a conditional trade). Once a trade
  would exceed the remaining budget, it and every later trade in the block
  are rejected with `kl_budget`.
- **Execution.** Parse the last fenced json block (fallback: the first
  `{"trades"` object); trailing commas and other malformed JSON are rejected
  with the decoder message. Validate each trade (id in the session, exact or a
  unique >= 12-character prefix; target within [0.01, 0.99]; context ids in
  the session, not the traded variable, no duplicates, outcomes yes/no; at
  most 2 trades per question). Execute valid trades in order under the engine
  lock; engine refusals are recorded per trade. A response with no valid
  trade is retried once with the error list appended to the prompt
  (`PREVIOUS ATTEMPT REJECTED`), then marked `failed`. Questions without a
  trade are logged as `missing_questions` (not retried).
- **Concurrency.** `ThreadPoolExecutor(--workers)`; the engine is
  single-threaded state, so prompt views, trade execution, log appends and
  snapshots are serialised under one lock while provider calls run in
  parallel. Sessions therefore see slightly stale prices; each record carries
  `snapshot_version` (executed-trade count when its prompt was built).
- **Snapshots and resume.** `snapshots/pass-P.json` after every pass and
  `snapshots/latest.json` every 50 sessions, each with `meta.log_lines`.
  `--resume` reads `sessions.jsonl`, skips sessions with a final status,
  loads the snapshot with the highest `log_lines` and replays the executed
  fills logged after it (pre-trade prices must match to 1e-6). The
  configuration identity (provider, effort, N, seed, limit, liquidity, width,
  budget, mock knobs) must match; `--passes` may be increased.
- **Usage watch.** `--quota-check-cmd` runs before each session (at most once
  per 20 s); if it prints `PAUSE` the pool blocks and re-checks every 30 s.

## What is logged

`<out>/sessions.jsonl`, one line per attempt (a retried session has two):
`session_id` (`p<pass>-g<group>`), `attempt`, `pass`, `group`, `question_ids`,
`n`, `prompt_sha256`, `prompt_chars`, `snapshot_version`, `provider`, `model`,
`effort`, `thread_id`, `provider_ok`, `provider_error`, `exit_code`, `usage`
(Codex `turn.completed` usage; token estimate for the mock), `raw_response`,
`parse_errors`, `parsed_trades`, `proposed`, `invalid` (validation
rejections), `missing_questions`, `kl_budget`, `kl_used`, `prices_at_prompt`,
`feedback` (retry only), `status` (`completed`/`retry`/`failed`), `executed`
(per fill: trade, `pre`, `post`, `pre_marginal`, `post_marginal`, `cost`,
`shares`, `kl`, `kl_cumulative`, `version`), `rejected` (trade, `reason`,
`detail`; reasons `kl_budget`, `engine_error`, `context_zero_probability` plus
the validation codes), `errors`, `timings` (`prompt_build`, `provider`,
`execute`, `total`), `started_at`, `finished_at`, `version_after`.

Also in the run directory: `config.json` (config, identity, provider
description, developer rule and its sha256), `questions.jsonl` (full records
plus `sibling_set`), `sibling-sets.json`, `groups.json` (groups per pass),
`snapshots/`, `table1-verification.json`, `summary.json`, and after scoring
`scores.json` / `scores.md`. Codex receipts live under `<out>/codex/`.

## Scoring

`scripts/score_market_game.py <out>` reports per pass: Brier and log loss of
the market marginals, improvement versus the previous pass, extremization
rate (share of marginals in [0, 0.1] or [0.9, 1]), mean price, and the
sibling coherence report (sum of marginals over exclusive sets with distinct
texts, share of sets summing above 1.05, and mean |sum - 1| over sets that
are complete partitions, i.e. exactly one member resolved yes); the Turtel
baselines from the file plus constant 0.5 and base rate; herding as the
Pearson autocorrelation of per-question price increments between consecutive
passes (and the same-direction share); trade validity (executed / proposed),
the rejection histogram, missing-question rate, KL-budget hits, conditional
trades executed and called off; realized LMSR payoff per session
`sum b (log p_new(w) - log p_old(w))` with conditional trades called off when
the context resolved the other way; and the telescoping check: per variable
the sum of its executed marginal-trade payoffs must equal
`b (log p_final(w) - log 0.5)`; the residual is reported and must be ~0 for
variables never linked by a conditional trade (linked variables reprice
through their neighbours, and their residual is reported separately).

Mock results, 2026-09-22 (seed 0, flip rate 0.2, 6 workers):

| run | sessions | wall | Brier p1 / p2 / p3 | validity | KL-budget hits |
|---|---|---|---|---|---|
| full 1,265, N=8, 3 passes (`runs/market-game/mock-full`) | 477 (8 retries, 0 failed) | 8.7 s (15 s cold) | 0.1716 / 0.1496 / 0.1183 | 0.940 (4163/4428) | 71 sessions |
| `--limit 20`, 3 passes (`runs/market-game/smoke20-final`) | 9 (2 retries) | 0.02 s | 0.0738 / 0.1242 / 0.1434 | 0.892 (66/74) | 2 sessions |

Full-run details: log loss 0.538 / 0.488 / 0.390, extremization 0.06 / 0.32
/ 0.48, herding autocorrelation -0.24 (the mock's noise is independent per
pass), 553 conditional trades executed (278 called off at resolution),
rejections kl_budget 215, context_self 17, target_range 14, unknown_variable
10, context_duplicate 9, realized payoff 37,357 (b = 100), telescoping
residual for the 510 unlinked variables 6e-14. Baselines on the full set:
polymarket 0.1509, o1 0.2017, ReMax 0.1907, constant 0.5 0.2500, base rate
0.2082. The mock is not a forecaster (it moves half-way toward a 20%-flipped
resolution); these numbers only show that the loop, the budget, the
rejections and the scoring behave. With 6 workers the run is not
bit-reproducible (sessions see stale prices depending on interleaving);
`--workers 1` is.

## The exact prompt text

Developer instructions (`developer_instructions`, sha256
`da908216ad21c8cab63b34b4334eedc6240d2a0137a42b8e666e34c0f75fc0e8`):

```
You are a careful forecaster trading in a combinatorial prediction market. Follow the date rule, the market rules and the output format in the user message exactly. Do not use tools, browse, read files, consult memory, or delegate. End your reply with exactly one fenced JSON block containing your trades.
```

User prompt (`market_game.build_prompt`; `{...}` are the filled fields; the
question block repeats for each of the N questions; rule 5 has an uncapped
variant on pass 1; the `PREVIOUS ATTEMPT REJECTED` section appears only on
the retry):

```
COMBINATORIAL PREDICTION MARKET SESSION {session_id} (pass {pass}, {N} questions)

DATE RULE
Answer as of each question's stated date (the "Today's date" line inside its evidence). Do not use any knowledge from after that date, even if you have it; pretend you do not. Reason only from the evidence shown and from general knowledge available before that date.

MARKET RULES
1. Every question is a binary variable with outcomes "yes" and "no". The price shown is the current market probability of "yes".
2. A trade sets a new market probability ("target") for one variable. The market maker is a logarithmic market scoring rule (LMSR) with liquidity b = 100.
3. A trade may carry a context: one or more other variables from this session fixed to "yes" or "no". Such a conditional trade sets P(A | B=yes) (or P(A | B=no)) instead of P(A); it creates a dependency between A and B in the market, and P(B) itself does not move.
4. Payoff at resolution: b * (log p_new(w) - log p_old(w)), where w is the realised outcome of the traded variable and p_old, p_new are its probabilities before and after your trade (conditional probabilities for a conditional trade). You gain when you move the price towards the truth and lose when you move it away. A conditional trade is called off (payoff 0) if its context does not resolve as stated.
5. KL budget: the total KL divergence you move in this session, summed over your trades (KL(new || old) of the traded probability, in nats), must stay within eps_total = {budget:.3f} nats. Trades are executed in the order given; once the budget is exhausted the remaining trades are rejected.
   [pass 1 variant: 5. KL budget: this pass is uncapped. There is no limit on the total KL divergence you move (KL(new || old) of the traded probability, summed over your trades, in nats); later passes cap it. Trades are executed in the order given.]
6. Trades execute against the live market; prices may have moved slightly since this prompt was built.

QUESTIONS
--- Question {k} of {N} ---
id: {question_id}
question: {question_text}
close date: {question_close_date}
current market probability of "yes": {price:.3f}
current conditionals involving this question: none
   [or one line per linked question:
  - with {other_id} ("{other text}"): P(this=yes | other=yes) = 0.xxx, P(this=yes | other=no) = 0.xxx; P(other=yes | this=yes) = 0.xxx, P(other=yes | this=no) = 0.xxx]
evidence (frozen retrieval; the "Today's date" line is the date you must answer as of):
{Turtel's prompt from "Question Background:" through "</News>", verbatim, including the "Today's date:" and "Question close date:" lines}

REQUIRED OUTPUT
First, for each question, a short rationale: the evidence you rely on (cite article numbers) and the key considerations. Then, at the very end of your reply, exactly one fenced JSON block of the form:
```json
{"trades": [{"variable": "<id>", "target": 0.xx, "context": [{"variable": "<id>", "outcome": "yes"}]}]}
```
Constraints:
- "variable" is one of the question ids above (copy it exactly).
- "target" is a number within [0.01, 0.99] (the new probability of "yes").
- "context" is optional; when present it lists one or more other question ids from this session, each with an "outcome" of "yes" or "no". Do not repeat a variable inside a context and do not put the traded variable in its own context.
- At least one trade per question. Express "no change" by targeting the current price. At most 2 trades per question (for example one marginal trade and one conditional trade).
- Use conditional trades only where you believe one question's outcome should change the probability of another.
- Trades that violate these constraints are rejected individually; a response without a valid JSON block is rejected as a whole.

PREVIOUS ATTEMPT REJECTED
Your previous response for this session was rejected for the reasons below. Respond again, following the output format exactly.
- {error}
```

A rendered first-session prompt (8 real questions) is 41k characters; print
one with `--dry-run`.

## Smoke checklist before a real run

1. `python3 -m unittest test_market_game` passes (27 tests; run it with the
   venv python too so the Table 1 test reads the parquet directly).
2. `--provider mock --limit 20 --passes 3` completes, `score_market_game.py`
   prints the table, `telescoping unlinked_ok` is true.
3. `--provider astra --effort <e> --dry-run` shows the expected model, effort,
   developer rule sha256 and the prompt; check the `-C`/`-o` paths are
   absolute and under the run directory.
4. `--provider astra --limit 8 --passes 1 --workers 1` on Kelvin's go: one real
   session. Inspect `sessions.jsonl` (`provider_ok`, `thread_id`, `usage`,
   `parsed_trades`, `executed`, `rejected`) and `<out>/codex/p1-g000/attempt-1/`.
   Confirm no tool use (`provider_error` empty) and that the block parsed.
5. `--limit 40 --passes 2 --workers 3` to exercise the budget on pass 2 and
   stale prices under concurrency; then the full run with `--resume` and a
   `--quota-check-cmd`.
6. After each pass, `score_market_game.py <out>` and compare with the
   baselines table; watch the extremization rate and the exclusive-set sums.

## Unverified

- No Codex invocation has been executed; the command is asserted equal to the
  native checks' command, and the event decoding follows
  `run_native_probes.py`, but the real stream has not been observed here.
- Model behaviour on 41k-character prompts (rationale length, id copying,
  conditional trade usage) is unknown; the parser tolerates prose, a second
  json block and unique id prefixes, and rejects malformed JSON.
- The exclusive sibling sets are heuristic (structural keys plus a strict
  similarity rule); the coherence report is descriptive only.
