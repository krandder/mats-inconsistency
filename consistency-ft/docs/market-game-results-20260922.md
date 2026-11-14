# Frontier market game, 22 September 2026: results

Stage 0 of the self-trade goal. Frozen frontier models traded the bayes-market
factored LMSR over Turtel et al. 2025's released test set (1,265 Polymarket
yes/no questions, frozen Exa prompts, resolutions known). No training. Live
page: https://consistency.futarchy.ai/market-game.html. Runner and scorer:
`harness/market_game.py`, `scripts/run_market_game.py`,
`scripts/score_market_game.py`; campaign: `scripts/overnight_market_game.sh`.

## Setup

- Market: 1,265 binary variables at 50%, independent, liquidity 100, treewidth
  cap 8. Trades are `trade_to_probability` with optional context (conditional
  bets that create structure). Verified before the run: LMSR fills match the
  closed forms, payoffs telescope to 1e-14, conditional trades touch only their
  slice, over-width trades are rejected with state untouched.
- Session: one Codex invocation, no tools, reasoning effort high, N questions
  grouped by TF-IDF similarity, each with evidence, close date, current price and
  conditionals; rationale per question then one JSON block of trades with short
  aliases (q1..qN). Pass 1 uncapped; later passes budgeted at 0.3 nats per
  question. One retry on an invalid block. 18 concurrent sessions via the proxy.
- Three configurations, three passes each, reshuffled groupings per pass:
  Astra N=8, Luna N=8, Astra N=16. 26.4M tokens total.
- Scoring replays each pass's market from the session log by fill version
  (resumed runs rewrite the snapshot files). Joint log loss is −log P(ω)/n
  under the whole junction tree by the chain rule; verified order-invariant and
  against a hand-computed three-variable joint.

## Results

Brier on all 1,265 questions after each pass; joint log loss in nats per
question; herding is the autocorrelation of per-question price increments
between consecutive passes.

| Run | Pass 1 | Pass 2 | Pass 3 | Log loss p3 | Joint p3 | Structure gain p3 | Extremization p1→p3 | Herding | Valid trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Astra N=8 | 0.1746 | 0.1696 | 0.1699 | 0.518 | 0.491 | 0.026 | 23% → 30% | 0.01 | 4,096 / 4,100 |
| Luna N=8 | 0.1874 | 0.1887 | 0.1950 | 0.609 | 0.599 | 0.010 | 24% → 36% | 0.14 | 4,066 / 4,068 |
| Astra N=16 | 0.1761 | 0.1684 | 0.1662 | 0.508 | 0.477 | 0.031 | 23% → 30% | −0.01 | 4,009 / 4,011 |

Turtel Table 1 on the same questions (verified from the released file): base
DeepSeek-R1-Distill-14B one-shot 0.215, o1 0.202, ReMax 0.191, Polymarket
price 0.151, constant 50% 0.250.

## Findings

1. **Valid combinatorial trading works.** 12,171 of 12,179 proposed trades
   executed (99.9%); the eight rejections were out-of-range targets. Conditional
   trades: 721 (Astra 8), 286 (Luna), 590 (Astra 16). The market reached
   treewidth 6 with components up to the session size; no width rejections.
2. **The market beats every one-shot number Turtel reports**, including his
   RL-trained model, but this is contaminated: the questions closed in February
   and March 2025, inside both models' training windows. The prompt forbade
   later knowledge; compliance cannot be verified. Within-run comparisons are
   the clean results.
3. **Repeated passes help Astra and hurt Luna.** Astra improved from pass 1 to
   pass 2 (Brier −0.005 at N=8, −0.008 at N=16) with zero herding and kept
   improving the joint score through pass 3. Luna got worse every pass, with
   positive herding (0.14, 70% of moves in the direction of the previous pass)
   and extremization rising to 36%. Same market, prompt and budget: the game
   amplifies the trader's dynamics, so calibration under feedback decides
   whether more passes help.
4. **The joint score keeps improving after the marginals flatten.** For Astra
   the structure gain (marginal minus joint log loss) grew every pass, to 0.026
   at N=8 and 0.031 at N=16, meaning the conditional dependencies the model
   built are informative when neighbours resolve. This is the number the LMSR
   settles on and the one that credits coherence.
5. **Coherence tightened for every model**, including Luna: mean deviation of
   mutually exclusive sibling sets from summing to one fell from 0.27–0.37 to
   0.23–0.30.
6. **Bigger scopes helped Astra**: N=16 gave the best market (0.166, joint
   0.477) at fewer tokens, at the cost of a higher untraded rate (10%) because a
   refusal drops 16 questions at once.
7. **Extremization is the creep**: it rose about seven points per run for
   Astra and cost log loss on Astra N=8's pass 3. Turtel's guardrails or a
   tighter late-pass budget are the lever.
8. **Refusals**: Astra declined political-figure questions in 20 sessions at
   N=8 and 16 at N=16 ("I can't assign probability scores to a public
   official's statements"); Luna in 3. Retries within the pass recovered some.

## Operational lessons

- The Codex wrapper resolves `codex` from PATH; the global npm copy on General
  is broken (missing platform package); `~/.local/bin` holds the working
  standalone 0.155.1.
- Long hex question ids were mis-copied by the model once in eight; short
  per-session aliases removed the problem.
- Resume-and-retry after later passes trades on the wrong market and rewrites
  `pass-N.json`. Retries now happen inside a pass, and scoring replays from the
  log's fill versions, which are exact.
- The engine is not the bottleneck: 1.6 ms per trade including re-triangulation,
  10 ms for the full joint score, against one to three minutes per session.

## Data produced

- `runs/market-game/{astra-n8,luna-n8,astra-n16}/`: per-session logs with
  prompts (hashed), raw responses, parsed and executed trades, fills, usage,
  thread ids; snapshots; scores. Raw responses are the distillation traces.
- Stage D in parallel: `/home/agent/data/polymarket-train-20260922/` (18,176
  resolved questions, families, splits, price history for half) and
  `/home/agent/data/polymarket-evidence-exa-20260922/` (10,000 questions,
  93,056 dated snippets, $140 of Exa; 6.5% of returned results violated the
  date bound and are quarantined; 90,779 leak-screen rows prepared for Jev).

## Next

1. Jev leak screen on the evidence pool, calibrated against a few hundred
   Astra-labelled snippets.
2. A second Astra burn on training questions to harvest distillation traces.
3. Stage 1: the 14B replication on the rebuilt training set, scored on the same
   test set.
