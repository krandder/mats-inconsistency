# Dark room and white noise: is there a framing with no fitted parameters?

For Kelvin, 2026-10-03. Synthesis of the seven notes in `math/darkroom/`. Your sharpened version (`value-of-information`) was the one without a skeptic pass, so I tested it in a new toy (`math/darkroom/synthesis/`). Numbers cite script outputs run on 2026-10-03.

## 1. The answer

1. No objective built from the agent's own sensations dissolves both rooms. Six framings each fail one room or need a parameter that decides it (six cases, not a proof).
2. The reason is shared: they value information about the sensory stream itself. To predict, a blank stream is best (dark room); to learn about, a stream with many parameters is best (noisy TV).
3. Your sharpening works: information is worth what it changes about a goal, a question about the world. Dark and static both score exactly 0 because neither depends on the goal, for any noise law, and the goal's strength drops out.
4. The irreducible choice is which variable is the goal: an input, not a fitted number. Without one, every room scores 0 and cost picks dark; a goal about your own senses brings back the dark room or the noisy TV.
5. If it must learn which room bears on the goal, it also needs posterior sampling and a prior hypothesis "this stream is irrelevant to my goal". With both, the useful room won in every run and prior tried.

## 2. The seven framings

| Framing | Dark room | White noise | What decides it |
|---|---|---|---|
| Expected free energy (Friston) | with a parameter | not solved | Preferences must rank dark low; their weight flips the choice. Novelty prefers noise. |
| Learning / compression progress (Schmidhuber) | not solved | with a parameter | Progress decays to 0, then cost picks dark. The model class decides noise. |
| Persistence, steady state (Friston 2013) | not solved | with a parameter | A constant stream is the best fuel. The sensor's energy gap decides noise. |
| Empowerment (Klyubin, Salge) | not solved | not solved | Passive structure scores 0 like dark. A TV that changes channel when you act scores up to 1 bit. |
| Logical-induction profit (Occam) | not solved | with a parameter | Profit is bounded everywhere; puzzles posed in the dark pay as well as the world. The frequency grid decides noise. |
| PDG inconsistency (raw / LIR reducible part) | not solved / with a parameter | max-entropy noise only / not solved | Raw inconsistency is surprise, so dark wins. The reducible part keeps ~20% attention on generic noise. |
| Goal-relative information value | solved at equal cost while a goal is open | solved | The goal variable; stakes vs cost when costs differ. |

## 3. The framing that works: price news by what it moves on your question

**Plain version.** The agent is a trader with a position in one question, "is the open door on the left?" News is worth what it is expected to move that price. A blank wall moves it by nothing; so does static, however many bits per second it shows. A muffled voice from behind the door is worth something, noisy as it is. Curiosity holds a position in every sentence about its own sensor readings, where static is the busiest market going (it moves those prices 0.5 per step forever, `value-of-information/output.txt` part D). Surprise minimisation is paid for low volatility in its own senses, and a dark room is the cheapest way to collect.

**Formal version.** For goal variable Z and a candidate look x, the value is I(Z; x | all seen so far), the expected log-score gain on Z (Lindley 1956; Bernardo 1979). In a PDG, a hard preference edge on "goal met" (W = 1) adds inconsistency −log P(W = 1) (Richardson 2022, Prop. 2), so x is worth E log P(W = 1 | x) − log P(W = 1), and the edge's confidence drops out. In a market it is E|ΔP| on the query sentences. All three give the same zeros. Active learning made the same fix: EPIG targets predictions instead of parameters, because BALD "is prone to prioritise data of limited relevance" (Bickford Smith et al. 2023).

**How this differs from Friston's answer.** His preferences must name the dark room's outcomes and rank them low, and their size decides. This goal never mentions light or noise, and its size drops out. The dark room loses because it cannot move the question, which the agent works out from its own model.

**Toy A, relevance known** (`value-of-information/pdg-output.txt`). Of seven rooms, every one unrelated to the goal scores 0.0000 at every look count and preference confidence tested; noisy labels of the goal score 0.632 nats at confidence 0.25 and at 100. Curiosity pays the dark room 1 bit and unknown-bias noise (1/2) log2 n bits, uncapped.

**Toy B, relevance learned** (new: `synthesis/relevance.py`, `relevance-output.txt`). Each episode has a fresh goal bit Z; the agent takes 3 equal-cost looks, guesses Z, then is told it. Rooms: dark, fair noise, a 64-symbol TV with a random law, and S, which shows Z with 25% of bits flipped. The agent is not told which room is about Z. 2000 episodes, 16 seeds:

| Policy | S share, last quarter | Success (best 0.84) | TV looks of 6000 | Runs not ending in S |
|---|---|---|---|---|
| Goal value, posterior sampling, "irrelevant" hypothesis | 0.996 | 0.84 | 113 | 0/16 |
| Goal value, posterior sampling, no such hypothesis | 0.994 | 0.84 | 501 | 0/16 |
| Goal value, greedy | 0.375 | 0.62 | 2609 | 10/16 |
| Curiosity (information about every latent) | 0.990 | 0.85 | 1427, still rising | 0/16 |
| Surprise minimisation | 0.188 (dark 0.81) | 0.56 | 0 | 13/16 |

- Greedy never tries a room whose relevance it cannot yet see (its expected value is exactly 0); after a forced dark start it stays on the TV in 16 of 16 runs. Posterior (Thompson) sampling has nothing to set and does try S.
- Across 12 prior settings (Dirichlet a0 0.1-4, irrelevant-mass 0.1-0.9, forced dark start or not), posterior sampling keeps S's share at 0.991 or more; the prior only sizes the TV transient (44-670 looks).
- Noisier TV (`synthesis/tvsize-output.txt`, 4 to 256 symbols): TV looks rise 40 → 156 with the irrelevant hypothesis, 87 → 1297 without it, and 74 → 2808 (94% of looks) for curiosity.
- Curiosity ends in S only because the goal is redrawn each episode; with one fixed goal (toy A) it pays noise without limit.

**Still a choice.** (1) The goal variables. Weights among world goals leave both rooms at 0; mixing in a goal about your own senses flips to dark once its weight passes 0.842. (2) Stakes vs cost when costs differ: the agent leaves the dark room when stakes exceed about 20 times the cost per look, which is problem data if the stakes are real. (3) Once the goal is settled the agent rests, which I count as correct; a lasting drive needs new goals or a drifting world.

## 4. What it means for the reasoning network

A block starts with three kinds of claim:
1. Axioms as hard edges. Their confidence makes no difference (PDG note, check A1).
2. A house view that includes, for each sentence, "this has nothing to do with the queries". Often logic settles it; where it does not, the market learns it, and toy B says the hypothesis keeps the noisy-TV cost small.
3. Query edges: a hard preference "query q gets settled" for each question we ask, the only source of value. No preference on its own inputs or outputs, no curiosity bonus.

"Attend to what you can fix" (LIR 2026) then means: run the computation with the largest expected move in the query prices, picking lemmas to try by posterior sampling. Settled sentences get no attention. Pseudorandom ones are priced at their frequency and then ignored (logical induction Thm 4.2.7), with no noise detector. Puzzles that bear on no query get none; that closes the trap where thinking in the dark paid as well as the world. And it stops when no computation is expected to move a query price.

Without the query edge the rule is curiosity again: noise frequencies and self-made busywork count as fixable, as the PDG and logical-induction skeptics found. With no open query the net has no reason to think; queries have to come from us.

## 5. The one experiment

Give an LI-net block a few queries and four kinds of candidate computation: settled sentences (dark), pseudorandom sentences from the XOR-with-random-constants timeliness family (noise), lemmas on the path to the queries, and unrelated hard sentences (busywork). At equal compute compare (a) expected query-price move with posterior sampling, (b) the same, greedy, (c) expected price move over all sentences (curiosity), (d) lowest inconsistency. Grow the pseudorandom pool from 10 to 1000.

Prediction: (a) puts over 90% of compute on lemmas at every size and its pseudorandom spend barely grows; (c)'s grows with the pool; (d) sits on settled sentences; (b) sometimes never finds the lemmas. If (a)'s pseudorandom spend grows in proportion to the pool, learned prices are inventing relevance and the block needs the irrelevant hypothesis built in.

## Sources

Each note in `math/darkroom/` lists its fetched sources. Added: Bickford Smith et al., "Prediction-Oriented Bayesian Active Learning", AISTATS 2023, https://arxiv.org/abs/2304.08151 (abstract fetched 2026-10-03). Re-run: `cd math/darkroom/synthesis && python3 relevance.py && python3 tvsize.py` (about 11 and 9 min on 16 cores).
