# Luna link graph over the consistency-ft events (2026-09-27)

Kelvin's 27 Sep design: an external model says which events are linked, separately for train, validation and
test. The graph only decides which joint questions we later ask Qwen3-32B. It carries no values: no
"A and not B = 0" rules, no probabilities, no directions, no outcomes.

Code: `scripts/luna_linkgraph.py` (`nodes`, `run`, `edges`, `cands2`, `run --pass2`, `edges --pass2`,
`connectivity`, `stats`, `show`).
Data: `/home/agent/data/luna-linkgraph-20260927/{train,validation,test}/`.

## Events (nodes)

- The events are every binary market and every multiple-choice option of the parents in each split of
  `jev-disjoint-v1-filtered-20260919` (train 1,886 parents, validation 120, test 612; the parent sets
  do not overlap). This covers more than the pair manifests: all options of each parent are included, not
  only the options that were paired. `in_pair_manifest` marks the ones that were paired.
- Kept: resolved YES or NO. Dropped: N/A (CANCEL) and partial (MKT) resolutions.
- Times come from the as-of store (`manifold-asof-20260923/asof.sqlite`):
  - `open_time`: when the market was created, or the answer was added if that was later.
  - `close_time`: the market close in force just before resolution.
  - `resolution_time`: when the market or option resolved.
  - `end_time = min(close, resolution)`: the last moment the event was open and unresolved.
  
  Question and criteria text are the versions in force just before `end_time`. A later step can ask each
  week only about events with `open_time <= week < end_time`.
- Also dropped: events open for less than one day (`end_time - open_time < 1 day`). They could not share a
  week with anything, so they cannot be linked. Most are same-day sports prop options. This removed 737
  events in train, 91 in validation and 159 in test.

| split | events | binary | MC options | YES rate |
|---|---|---|---|---|
| train | 5,413 | 1,427 | 3,986 | 0.405 |
| validation | 1,213 | 73 | 1,140 | 0.440 |
| test | 2,684 | 453 | 2,231 | 0.336 |

## Candidates

For each event, up to 40 candidates from the same split, drawn from events whose open windows overlap
the event's window by at least 1 day. They are ranked by BGE-small (bge-small-en-v1.5, CLS pooling)
cosine on the question text plus the option text, with a +0.06 bonus for sharing a Manifold group. Groups
that hold more than 10% of the split's events do not count. At most 15 candidates can be options of the
event's own parent, which leaves room for other markets. Links never cross splits, because candidates
never do.

## Linker

- Model: GPT-6 Luna (`gpt-6-luna`) via `codex-any exec`, reasoning effort medium. Web search, shell and
  multi-agent are off, and there are no MCP tools. The model is never substituted: a model rejection stops
  the run.
- Each session covers 10 focus events. Batches are grown greedily from each seed's candidates, so they share
  rows in one question table (about 120 rows, about 16k input tokens per session).
- Up to 4 sessions run in parallel. A 429 or 529 from any session pauses all of them, and so do an active
  `proxy` claim in fleet-now and a `PAUSE` file in the output root. Runs are resumable: the batch plan is
  frozen in `plan-full.json`, and each session writes `raw/<session>.json`.
- The prompt asks only which candidates are logically or causally connected. For each focus event, Luna
  returns up to 5 candidate ids and one relation type per id: same-quantity, same-event-different-date,
  alternatives, cause/effect or related-topic. It is told to give no probabilities, outcomes, directions
  or explanations. The parser keeps only [candidate id, type] pairs where the id is in that event's list
  and the type is one of the five. Everything else is dropped and counted in `parse`. The raw text is
  kept in `raw/`.
- Edges (`edges.jsonl`) are undirected. Each edge has `u`, `v`, `type`, `types`, `proposed_by` (focus
  event id mapped to the type it gave), `mutual` (true when both ends proposed it), `model` and
  `sessions`. When the two ends give different types, `type` takes the first in the order listed above.

## Pilot (test, first 5 sessions = 50 events)

- All 5 sessions parsed. One needed a retry after invalid JSON. One session named 5 ids that were not in the
  focus event's list, and they were dropped.
- About 12 s per session. Input was 16.1k to 17.6k tokens and output 410 to 480 tokens per session.
- The pilot gave 183 edges, 48 of them proposed by both ends.

Three example neighbourhoods. "Proposed by" says which end named the link.

```
* Ukraine recognizes Russian sovereignty over Ukrainian territory in 2025? [Polymarket] [056d28qz8y]
    - cause/effect: Russia x Ukraine ceasefire in 2025? [both]
    - cause/effect: Will the war between Russia and Ukraine end in 2025? [both]
    - cause/effect: Will the Russia - Ukraine war end before 2026? [both]
    - same-quantity: Will Ukraine recognize Russian sovereignty over Ukrainian territory in 2025? [both]
    - related-topic: Will EU leaders use frozen Russian assets to finance Ukraine ... by 31 December 2025? [both]
* New wave of mobilization in Russia by end of 2025 :: An official order on a new mobilization wave is published
    - alternatives: ... :: No new mobilization wave occurs by Dec 31, 2025 [both]
    - cause/effect: ... :: Exit or travel restrictions for mobilization-eligible citizens [both]
    - related-topic: ... :: An official target number for new call-ups is publicly announced [this]
    - related-topic: ... :: Regional authorities announce local mobilization measures [this]
    - related-topic: ... :: President or Defence Minister say a new mobilization wave ... [this]
* Who will Trump pardon in 2025? [Polymarket] :: Changpeng Zhao
    - alternatives: same market :: Do Kwon [both]; Himself, Elon Musk, Sam Bankman-Fried, Roger Ver [this];
      Diddy [neighbour]
    - alternatives: Will Donald Trump pardon Sean 'Diddy' Combs before January 1, 2026? [neighbour]
    - related-topic: Trump intervenes in P. Diddy legal process in 2025? [neighbour]
```

Observations:
- The links look sensible. Duplicate markets are caught (the two Polymarket copies are linked as
  same-quantity), and options in the same market are grouped.
- The relation labels are loose. "alternatives" is also used for options that are not mutually exclusive
  (the pardon options), and for the binary "pardon Diddy" market. Treat the type as a hint for which
  question forms to ask, not as a constraint.

## Full run (pass 1)

- 933 sessions: test 269 (the 5 pilot sessions included), validation 122, train 542. They ran 04:50-05:49Z on
  27 Sep with 4 workers, about 12 s and 16-22k input tokens per session. There were no 429/529 errors and no
  pauses. A few invalid-JSON answers were fixed by the automatic retry. Every event got an answer.
- Pass-1 edges (`edges.jsonl`, `proposer: luna`): train 16,684, validation 4,282, test 8,858.

## Pass 2: wider candidates for events outside the main component

After pass 1, many markets were isolated or sat in small components, and several were plain misses. For
example, on test, "UEFA Women's Euro 2025 Final stage matches" and "What will happen at the Women's Euro
Final?" were not linked, and neither were the EPL prop-bet markets or the chess prop-bet markets.

An earlier idea was to add embedding "bridge" edges until the graph was connected. That was dropped (Kelvin:
"I dont want to force relations that are not there"). Pass 2 asks Luna again instead:

- Targets: every event whose market lies outside the largest market-level component of the pass-1 graph
  (train 379 markets / 1,569 events, validation 40 / 802, test 153 / 1,850).
- Candidates (`candidates-pass2.jsonl`): up to 80 events of other markets in the same split, with open windows
  overlapping by at least 1 day. They are the union of up to 25 sharing a non-generic Manifold group, up to
  25 sharing a rare capitalised name in the question or option, and the nearest by BGE cosine for the rest.
- Linker: the same prompt and rules as pass 1 (only which candidates are connected, up to 5, one relation
  type), 10 focus events per session. The raw output is in `raw-pass2/`. The new edges go to
  `edges-pass2.jsonl` with `proposer: luna-pass2`, leaving out pairs already in `edges.jsonl`.
- 423 sessions (test 185, validation 81, train 157), 06:02-06:31Z, 4 workers, with no errors or pauses.
- New edges: train 1,672, validation 666, test 2,170.

Components before and after (`connectivity.json`). A market-level component treats each market as one node
and uses only links between different markets:

| split | events: components (largest; singletons) | markets: components (largest; isolated) |
|---|---|---|
| train | 249 (3,662; 111) -> 146 (4,531; 62) | 216 (1,338; 153) -> 121 (1,483; 79) |
| validation | 45 (411; 6) -> 34 (433; 4) | 16 (69; 9) -> 13 (72; 6) |
| test | 96 (834; 45) -> 49 (1,155; 24) | 66 (455; 53) -> 30 (474; 24) |

What happened to the cases named on test:

- Now linked:
  - Women's Euro final-stage matches and the Final market.
  - The EPL prop-bet and matchweek markets, now in an EPL/sports component of 89 markets.
  - The chess prop bets, now in a chess component of 15 markets.
  - "Iran strikes Israel before 2026?", now in the main component.
- Still isolated:
  - The YouTube monthly markets for June, August and September. The Bundesliga matchdays are linked only
    partly: 5 of the 13 on test (1, 2, 5, 7 and 11) joined the 89-market component, and 3, 4, 6, 8, 9, 10, 12
    and 15 are still isolated. EPL 2024-25 Matchweek 36 is also still isolated.
  - Why they stay isolated: these markets are open in back-to-back periods that do not overlap. The June
    market closes on 1 July, and July's opens on 1 July. Each Bundesliga matchday is open for 2-3 days. Under
    the rule that linked events must share an open week, they cannot be joint questions, so they stay
    unlinked.
- The main component is still not the whole graph. The EPL/sports component (89 markets) and the chess
  component (15) are separate from the politics and economics component. Luna did not link them, and
  nothing was forced.

10 random new links (seeded):

```
[test] related-topic: NFL Week 9 Mega-Market :: Colts beat Steelers <-> NFL Week 10 Mega-Market :: Steelers beat Chargers
[test] related-topic: EPL [Matchday 13] top 3 or bottom :: Chelsea <-> EPL Matchweek 13 fixtures :: Aston Villa beat Wolves
[test] related-topic: EPL [Matchday 1] top of the table :: Arsenal <-> EPL Weekend Prop Bets (Matchday 1) :: own goal scored
[test] related-topic: EPL Matchweek 12 :: Liverpool beat Nottingham Forest <-> EPL Prop Bets (Matchday 12) :: 3+ draws
[test] cause/effect: EPL [Matchday 9] top 3 or bottom :: Liverpool <-> EPL Matchweek 9 :: Everton beat Tottenham
[validation] same-event-different-date: NFL Week 9 Prop Bets :: QB 3+ turnovers <-> NFL Week 6 Prop Bets :: QB 3+ turnovers
[validation] related-topic: NFL Week 11 Prop Bets :: 3 missed FGs/XPs <-> NFL Week 4 Prop Bets :: a missed extra point
[train] related-topic: Soccer Signings Summer 2025 :: Müller to Vancouver <-> Arsenal summer signings :: Eze
[train] related-topic: Cincinnati Masters semifinalists :: Taylor Fritz <-> US Open mixed doubles semis :: Rybakina + Fritz
[train] related-topic: Joe Burrow snap after Week 2? <-> NFL Survivor Pool Week 4 :: Titans @ Texans
```

They are sensible, and most are loose related-topic links. Only 7-26% of pass-2 edges were proposed by both
ends, against 31-38% in pass 1.

## Final graph (pass 1 + pass 2)

`graph-stats.json` per split. The treewidth figures are networkx upper bounds; degeneracy is a lower bound.

| split | events | edges (both ends) | mean / median / max degree | isolated | triangles | max clique | treewidth <= (min-fill) | degeneracy |
|---|---|---|---|---|---|---|---|---|
| train | 5,413 | 18,356 (6,545) | 6.78 / 6 / 80 | 62 | 20,036 | 9 | 73 | 9 |
| validation | 1,213 | 4,948 (1,524) | 8.16 / 7 / 81 | 4 | 8,539 | 10 | 18 | 9 |
| test | 2,684 | 11,028 (2,927) | 8.22 / 7 / 51 | 24 | 12,576 | 8 | 45 | 9 |

Most edges are related-topic, for example 7,359 of the 11,028 on test. About half of the edges join options of
the same parent market.
