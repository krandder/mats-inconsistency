# MATS research log, Kelvin Azevedo Santos

Made from agent transcripts, reports and repo history; each week was checked against its sources on 3 Nov 2026. Times are UTC.

## Overview

- Before the program (August to September), Kelvin studied Metaculus Market Pulse. Public data had no forecast histories, so the work moved to measuring how logically consistent LLM forecasts are.
- In mid-September, after a call with Deger, the main question became whether training for consistency improves accuracy. The first small training runs were void or showed no clear gain; making a model's answers consistent at answer time, with no training, did help.
- Week 1 (28 Sep - 4 Oct): forcing answers to agree helped weak models but added little for strong models or on real short-term questions. "Markets inside networks" gave no gains, and the focus moved to logical induction, PDGs and keeping beliefs separate from reward.
- Week 2 (5 - 11 Oct): from planning an LLM coherence-training run to small-network tests on ShipHunt. Reveal-chain TD matched plain training with about 0.28 of the training boards (audited).
- Week 3 (12 - 18 Oct): the base model no longer has to be consistent by itself. The new direction was an any-order world model, a learned search policy and a learned aggregator of disagreeing rollouts. Rounds 1-9 of small experiments were mostly negative or mixed.
- Week 4 (19 - 25 Oct): an audit found no capability claim from the learned-search campaign that survives a pre-registered rule. After the 20 Oct call with Deger, the main line became information aggregation measured against exact Bayes, and by 25 Oct an energy for inconsistent any-order models.
- Week 5 (26 Oct - 3 Nov): tiny PFN-style transformers trained against exact Bayes gave a first located mechanism. From 29 Oct most work went to ARC's problem of estimating the mean output of deep random MLPs; at 256 x 32 no method beat good sampling by more than a constant factor.

## Weeks

| Week | Dates | Entries |
|---|---|---|
| [Before the program](#before-the-program-10-august-to-27-september-2026) | 10 Aug - 27 Sep | 7 weekly entries |
| [Week 1](#week-of-28-september---4-october-2026) | 28 Sep - 4 Oct | 7 days |
| [Week 2](#week-of-5-october-2026) | 5 - 11 Oct | 7 days |
| [Week 3](#week-of-12-october-2026-mon-12---sun-18-oct-utc) | 12 - 18 Oct | 7 days |
| [Week 4](#week-of-19-october-2026) | 19 - 25 Oct | 7 days |
| [Week 5](#week-of-26-october---3-november-2026) | 26 Oct - 3 Nov | 9 days (3 Nov to ~02:00Z) |

## Before the program: 10 August to 27 September 2026

The program started on Monday 28 September. For these seven weeks there is one entry per week. Each entry lists the days with recorded activity; other days have no recorded activity. Times are UTC.

Status words: positive, negative, mixed, void, running, parked.

Short source names used below:
- OUT = `/home/agent/workspaces/mats/outbox/`
- CFT = `/home/agent/workspaces/consistency-ft-next-run-20260919/docs/` (consistency-ft repo; commit dates in UTC)
- CAI = `/home/agent/workspaces/calibrated-ai/experiments_page.py` (dated experiment list)
- [removed]
- [removed]

### Week of 10 August

Kelvin learned what Metaculus Market Pulse is and what data it exposes. The agent drafted a first research proposal for the Deger Turan stream.

#### Mon 10 – Sun 16 August

- **Active days:** Wed 12, Fri 14, Sun 16 August.
- **Focus:** What is Market Pulse, how many questions do bots actually forecast, and could its forecasts be tested for decision value?
- **Ran / finished:**
  - Market Pulse public research (12 Aug): Kelvin asked for an independent study. The "11 markets" shown for round 26Q3 are 11 question groups; the launch post lists 12 groups and 65 subquestions. Status: mixed.
  - Public question sweep (16 Aug): 60 Metaculus pages fetched, 34 question groups, 11 groups found for round 26Q3. Status: mixed (public pages only, no forecast histories).
  - Evaluation spec, harness tests on synthetic data and a literature table on whether forecasts lead to decisions (31 sources), all 16 Aug. Status: positive (method only).
  - Draft proposal "From forecast accuracy to decision value" (16 Aug), written by the agent. It is a discussion draft and there is no record that it was sent.
  - Public market-data baseline (16 Aug, started 22:32): 11 of 14 sources accepted and 96 rolling configurations run with no future-data leaks, all by 22:35. Its overnight bootstrap check kept running. This checks inputs only, not Metaculus forecasts. Status: running.
- **Decided:**
  - 12 Aug: Kelvin asked for a deep study of Market Pulse.
  - [removed]
- **Meetings:** [removed]
- **Sources:** OUT `market-pulse-corpus-report.md`, `market-pulse-evaluation-spec.md`, `market-pulse-simulation-soak.md`, `forecast-to-decision-evidence.md`, `mats-research-proposal-v1.md`, `STATUS.md` (16 Aug 22:31–22:37 lines).; [removed]

### Week of 17 August

The Market Pulse plan ran into missing data: public sources do not include forecast histories. Separately, the consistency-ft project started. It measures how logically consistent LLM forecasts are.

#### Mon 17 – Sun 23 August

- **Active days:** Mon 17, Tue 18, Wed 19, Thu 20 August.
- **Focus:** Can public data support a Market Pulse study, and how consistent are current models' probability answers?
- **Ran / finished:**
  - Market Pulse public baseline run finished 17 Aug, 03:11, when its bootstrap check completed (26,880 of 26,880 chunks). 11 of 14 sources were accepted, 96 rolling configurations completed and no future-data leaks were found. This checks inputs only, not Metaculus forecasts. Status: positive (checks only).
  - Consistency benchmark of 15 open-weight models (19 Aug). More capable models are more consistent, but none reach full consistency: for example, mean violation 0.082 for kimi-k3 against 0.159 for gpt-oss-20b. Status: mixed.
  - The consistency research plan passed its first review gate with Jessica Taylor (20 Aug), who added a training-cutoff audit.
  - Metaculus API access check (20 Aug). Neither documented API tier covers 12 of the 13 critical data needs, chiefly historical forecast snapshots. The report's advice: no API token yet; first get written permission and a description of the data. Status: negative.
  - Repeat sweep for new 26Q3 questions (20 Aug): nothing new verified. Status: negative.
  - Contamination probe set built (20 Aug): 129 dated probes over 9 quarters, to check model training cutoffs. Status: running (built, not yet run).
- **Decided:** no decision by Kelvin is recorded.
- **Meetings:** [removed]
- **Sources:** OUT `market-pulse-public-baseline-report.md`, `metaculus-access-gap-report.md`, `market-pulse-corpus-delta.md`; consistency-ft git log, commits of 19–20 Aug (202b055 "Multi-model benchmark", 38be840 "Gate 0 cleared", 46eed97 probe sets).

### Week of 24 August

No research or MATS work is recorded this week.

#### Mon 24 – Sun 30 August

- No recorded activity on any day.
- **Sources checked:** CODEX first thread (no research exchanges); consistency-ft git log (no commits 21 Aug – 11 Sep); OUT `STATUS.md`.

### Week of 31 August

[removed] Reading: Radiant, ARIA Safeguarded AI and Open Agency.

#### Mon 31 August – Sun 6 September

- **Active days:** Tue 1, Thu 3, Fri 4 September.
- [removed]
- **Ran / finished:**
  - [removed]
- **Decided:**
  - [removed]
- **Meetings:** [removed]
- **Sources:** [removed]

### Week of 7 September

The consistency benchmark was rerun on frontier models and reviewed. Collecting answers from a small open model (Qwen) for later training hit a sampler bug.

#### Mon 7 – Sun 13 September

- **Active days:** Sat 12, Sun 13 September.
- **Focus:** How often do frontier models break basic probability rules, and does more reasoning help?
- **Ran / finished:**
  - Consistency benchmark on 274 reviewed questions (13 Sep). Fewest violations: Grok 4.6 (0.064), Luna at max reasoning (0.066) and GLM-5.3 (0.067). Most: Astra (0.116) and Luna with no reasoning (0.122). One run each, no confidence intervals. Status: mixed.
  - Reasoning effort (13 Sep): Luna's violations fall from 0.122 with no reasoning to 0.063 at max (one run each). Status: mixed.
  - Factual recall probes (13 Sep): 836 of 1,161 answers were supported, and no model's training cutoff could be verified. Status: mixed.
  - Qwen own-answer collection (13 Sep): stopped at 22 of 2,926 because of a CPU sampler bug. The bug was fixed the same day and collection restarted with the fixed sampler. Status: void (first attempt).
- **Decided:** no decision by Kelvin is recorded. This was agent-run work in the consistency-ft repo.
- **Meetings:** [removed]
- **Sources:** CFT `native-evaluation.md`, `reasoning-sensitivity.md`, `contamination-audit.md`, `qwen-cpu-sampler-defect-20260913.md`; CAI (13 Sep rows); consistency-ft git log, 12–13 Sep.

### Week of 14 September

Kelvin returned to MATS work. After a call with Deger, the consistency research became the core of his MATS project, with the question "does training for consistency improve accuracy?" The five-week plan was written and shared, and the first small training runs gave no clear accuracy gain.

#### Mon 14 – Sun 20 September

- **Active days:** every day, Mon 14 to Sun 20 September.
- **Focus:** Does training a model to be logically consistent improve its forecasts, or does it trade off against accuracy?
- **Ran / finished:**
  - Reward explainers for consistency checks and a "belief training" design page (15 Sep). Design only.
  - Jev (a forecasting model reached through an API) on the consistency benchmark (17 Sep): violation 0.087, rank 10 of 24 settings. Status: mixed.
  - First Qwen3-0.6B pilot (17 Sep): 5 epochs, 240 validation queries. Log loss was 0.653 for accuracy-only and 0.646 with the consistency term, and inconsistency was 0.042 against 0.020. Accuracy was 0.650 against 0.629. Status: mixed.
  - Archive of Manifold market histories, comments and bets for training data (17 Sep). Data only.
  - First GPU training run, Qwen3-0.6B with arms A1–A5 (planned 19 Sep, run 20 Sep). On 20 Sep Kelvin judged the gap he saw (0.64 against 0.663) too small to matter. On 23 Sep the run was found void: the prompt left out the answer option of multiple-choice markets, which defines about 79% of the statements. Status: void.
- **Decided:**
  - [removed]
  - 15 Sep (after the Deger call): write a weekly update and a week-by-week plan. Point the consistency work at AIs as better economic agents and use Manifold data. The central question is whether consistency training improves accuracy. Goals: better forecasts, better economic decisions, and other prediction-based tasks if time allows.
  - 15 Sep: models should state probabilities rather than be trained through sampling. Kelvin added two ideas: models that output explicit structures (e.g. Bayes nets), and models that predict how much more reasoning would move their answer. Some training-data leakage is acceptable if remembered and new questions are mixed.
  - 17 Sep: the 10-week plan. Week 1 tests consistency against accuracy-only training. Week 2 sets up the next stages. Weeks 3–4 use GRPO for explicit joint distributions. Weeks 5–7 combine new information over time, using Radiant and Bayes Market as the environment. Weeks 8–10 cover decisions and the write-up. Kelvin argued that training each step on the next step's prices, anchored on final outcomes, is sound by backward induction. He kept it as a hypothesis to test.
  - [removed]
  - 19–20 Sep: plan the next run with the fixes and stop early when epochs stop helping. He asked for an adversarial review of the method and cost estimates for replicating published forecasting baselines (Halawi et al.; Turtel et al., arXiv 2505.17989).
- **Meetings:** [removed]
- **Sources:** OUT `mats-five-week-research-plan.html`; CFT `jev-benchmark-20260917.md`, `supervised-semantic-replication-20260919.md`, `next-run-plan-20260919.md`; pilot numbers from `/home/agent/workspaces/consistency-ft-followups-20260913/data/research/pilot375-training-5epochs-v1/report.json`.; [removed]

### Week of 21 September

The work moved from training a tiny model to using structure at answer time. Making a model's own answers logically consistent clearly improved them with no training, but most attempts to find links between events added nothing. On 27 Sep, studies of traders and markets inside networks started with Jessica Taylor.

#### Mon 21 – Sun 27 September

- **Active days:** Mon 21, Tue 22, Wed 23, Thu 24, Sun 27 September. Fri 25 and Sat 26: no recorded activity.
- **Focus:** Where does the gain from consistent answers come from: real logical structure, or just averaging out noise? And can links between events be found and used?
- **Ran / finished:**
  - Frontier market game (22 Sep): models trade a combinatorial market over 1,265 linked questions. Astra's market reached Brier 0.166 after 3 passes, better than one-shot models (o1 0.202) but not the Polymarket price (0.151). Luna got worse with each pass (0.187 to 0.195). The questions may be in the models' training data. Status: mixed.
  - Qwen3-32B zero-shot (23 Sep): test log loss 0.611, worse than a 12-number base-rate table (0.570). Status: negative.
  - Qwen3-0.6B with a consistency loss, 8 arms, rerun after the fix (23 Sep): beats the 32B but not the table, and consistency mostly pulls answers toward the prior. Status: negative.
  - Qwen3-235B (pre-cutoff) and GPT-6 Luna on the same test (23 Sep): both beat the table; Luna probably saw these questions in training. Status: mixed.
  - Making answers consistent with no training (24 Sep): 32B 0.611 to 0.541, 235B 0.553 to 0.516. Status: positive.
  - Structure against averaging, 1,066 test pairs (24 Sep): projection cut log loss by 0.081, averaging 5 samples by 0.017. With wrong relations the projection does 0.05 to 0.06 worse than with the true ones. Status: positive.
  - Choosing the most informative questions (24 Sep): finds the strong links in 57 questions, against 114 for random pairs. Status: positive.
  - Global world model at one date, 400 events (24 Sep): almost no real links between events. Status: negative.
  - Models against market prices on ordinary pairs (24 Sep): no model beats market plus independence. On pairs with built-in links, adding the model's link beats the market. Status: mixed.
  - Link graph (27 Sep): GPT-6 Luna marked related events in the train, validation and test sets (11,028 links on the test set), as the outside model that assigns links. Data only.
  - Weighting answers by confidence (27 Sep): no signal beats equal weights. Status: negative.
  - Eliciting weights (27 Sep): a thought-through answer helps a little (−0.030 nats), and the model cannot predict its own change of mind. Status: mixed.
  - Weekly world model, 35 test weeks (27 Sep): pair and triple terms add nothing, and multiplying single answers is best. Status: negative.
  - Balanced 16-question design around random triples (27 Sep): beats independence on joint questions (0.393 against 0.480) and repairs single answers. Status: positive.
  - EURISKO literature study and a "logical soup" of evolving traders (27 Sep). More traders per question helped: Brier on periodic sequences 0.204 at 4 per question, 0.084 at 64 (3 seeds each). More compute, hard sequences, an arbitrageur and ablations did not help. Status: mixed.
  - Wealth accounting on a small network (27 Sep): removing the poorest 1% of neurons hurts (loss 1.05 to 1.30, random 1% gives 1.07), and no recent-income measure tracks neuron value. Status: negative.
- **Decided:**
  - 21 Sep: with Jessica Taylor, an idea drawing on logical inductors and Hanson's combinatorial market maker. Kelvin asked for a full run proposal that combines Turtel-style reinforcement learning with Bayes Market consistency.
  - 22 Sep: answered the pre-run questions. R1-Distill-Qwen-14B is fine as the model. Data comes through date-limited search. Leak screening uses many simple questions answered by JEV.
  - 23 Sep: do not train on the frontier traces after the leak check; keep them aside. Run a 32B zero-shot baseline before any training. Rerun the 0.6B with all arms, including the consistency-weight sweep. Forecasts are made 7 days before close, and only pairs where both questions are open are kept. Models should see full market context (prices, comments, other markets). Consistency is internal to the model, not agreement with market prices.
  - 23 Sep: chose mats.ekelvin.com as the results site for Deger.
  - 24 Sep: main interest is the nature of noise and decorrelation, and how much structure a systematic pass can reveal, including by choosing which questions to ask. Not yet training models on Bayes Market output.
  - 27 Sep: train the 0.6B on pairs with built-in links, with and without consistency. An external model assigns links for all train and test events. Run the 32B on all 35 test weeks. Use a balanced question design to remove AND/OR bias.
  - 27 Sep: after the EURISKO report, study self-modifying agents (DGM) before any EURISKO-style experiment. Make the evolving-program "soup" a market of traders on prediction tasks, like the traders in logical induction. With Jessica Taylor, study traders inside neural networks (wealth accounting, traders made of smaller traders).
- **Meetings:** [removed]
- **Sources:** CFT `market-game-results-20260922.md`, `qwen32b-verbalized-baseline-20260923.md`, `coherence-projection-20260924.md`, `averaging-vs-structure-20260924.md`, `active-structure-20260924.md`, `worldmodel-global-20260924.md`, `structured-pairs-20260924.md`, `confidence-weighting-20260927.md`, `eliciting-weights-20260927.md`, `weekly-worldmodel-20260927.md`, `balanced-triangles-20260927.md`, `luna-linkgraph-20260927.md`; CAI (21–27 Sep rows); `/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md`, `/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md`; OUT `weekly-update-2026-09-22.html`; [removed]

## Week of 28 September - 4 October 2026

Consistency work found that forcing a model's answers to agree helps weak models but adds little or nothing for strong models or on real short-term questions. The "markets inside networks" line ended with no gains, and the focus moved to logical induction, PDGs and keeping beliefs separate from reward.
First MATS week. A live MiniBench forecasting bot and a backtest loop were approved.

### Monday 28 September

- **Focus:** Do answers about pairs and triples of questions tell us anything new about a single question?
- **Ran / finished:**
  - Order matrix ("does B inform A?"). The gain on single questions comes from a question's own answers, mostly from A agreeing with not-A (64%). Answers about other questions add 0. **negative**
  - Paying the parts of a network (wealth-net and traders-net studies). Exact leave-one-out payment tracked a unit's true value (+0.42, against +0.04 for gradient-based wealth). But a block-local market added little beyond random skipping of units (held-out loss 0.299 against 0.335; wealth scaling alone 0.306), and gating units by wealth hit its pre-registered stop rule. **mixed**
- **Decided:** Kelvin agreed to change the network-markets study's target. Gradient credit and market payment disagree in sign on the units that matter. The dense-network bookkeeping work becomes a negative-result write-up.
- **Meetings:** [removed]
- **Sources:** CFT/order-matrix-20260928.md; experiments_page.py rows dated 28 Sep; eurisko-study/wealth-net/RESULTS.md; [removed]

### Tuesday 29 September

- **Focus:** Can a small model learn the consistent answers directly, and does consistency help a model learn faster over time?
- **Ran / finished:**
  - Clearer "NOT" wording in prompts (Qwen3-32B). The correlation between p(A) and p(not A) went from -0.18 to -0.61 (ideal -1). Log loss over all 12 question forms went from 0.622 to 0.594. **positive**
  - Consistency distillation v1 (Qwen3-0.6B student). A student trained on the projected (consistent) answers scored 0.555. One trained on raw answers scored 0.568. That beats the raw teacher (0.594) and a base-rate table (0.570), but it covers only about a third of the distance from the base rate to the projected teacher (0.526). **positive** (small)
  - Prequential run: questions in time order, each scored before the model trains on it. Consistent targets gave a shorter code than raw targets by 1,012 bits. The gap was constant over time, so consistent targets were not faster to learn from. All students stayed close to the base rate. **mixed**
  - First MiniBench backtest on 30 questions. The consistency arm was level with the plain arm (+0.2 points, CI [-2.0, +2.6]). **negative**
  - Trader-economy, Market ResNet and puzzle toys (about 50 small runs). Shapley-style payment ranks units well by true value (0.92 for weights, 0.965 for hidden units). Training on that payment never beat backprop (for example, log loss 0.808 against 0.743). The Market ResNet reached 0.93 at 100x its training length against 0.12 for backprop through depth, but an audit the same day showed its payment equals detached one-step backprop. A batch of 12 puzzle experiments gave no gains. **mixed**
- **Decided:** Kelvin approved rewriting the negation prompts. He made the main question for the consistency training work whether it helps a model learn faster on time-ordered data, scored prequentially. He asked for a larger distillation run (1.7B student, 235B teacher) on spot GPUs with checkpointing. A theory track was started on his question of whether internal markets can work at the right learned representation. He asked for a weekly summary for MATS, which was drafted.
- **Meetings:** [removed]
- **Sources:** CFT/prompt-negation-fix-20260929.md, consistency-distill-20260929.md, prequential-20260929.md; CAI/trader-economy/CATALOGUE.md; CAI/mats-weekly-summary-20260929.md; experiments_page.py rows dated 29 Sep; [removed]

### Wednesday 30 September

- **Focus:** Does the consistency gain survive with stronger models and on real questions?
- **Ran / finished:**
  - Strength scaling on the same 300 pairs (pre-registered). Gain from projection: Qwen3-32B -0.074, Qwen3-235B -0.036, GPT-6 Luna -0.071, Opus 5.5 -0.005 (below the 0.01 bar). The gain follows how inconsistent the model is, not how strong it is. Opus partly recalled the 2025 outcomes, but it still gained only -0.005 on pairs where it was unsure. **mixed**
  - MiniBench backtest on 347 resolved questions with Opus 5.5 (pre-registered). Consistency minus plain was -0.30 points (CI [-1.92, +1.41]), so no gain. With Luna, the consistency arm first scored worse (-6.36), but that came from a bug in how the arm blended answers; after the fix neither model showed a significant difference (exploratory). **negative**
  - Weekly world model: a prompt warning that "A and B" answers run too high. Joint log loss beat multiplying single answers by -0.046 on 6,000 joints and by -0.062 on held-out joints. Thinking mode improved single answers by 0.11 nats. **positive**
  - Same warning prompt on the Manifold pair set (pre-registered). It did not transfer: +0.016 (CI [+0.001, +0.032]) against the product of singles. **negative**
  - Distillation v2. With the 235B teacher, a 1.7B student on consistent targets beat one on raw targets by -0.006. The edge did not grow with student size. The best single-call student scored 0.536, still 0.023 behind the projected teacher. **mixed**
  - Market ResNet follow-ups. The 100x depth result came from random training order, not the market. A published baseline (DT-Recall) beat the economy, 589/600 against 557/600. **negative**
  - Exact consistency engine checked against Metaculus's map_forecasting code. It passed their 14 tests. An 8-variable chain took 0.14 s, against 1,777 s for theirs. **positive**
  - "Markets inside networks" theory review, round 4. It counted 19 pre-registered experiments with zero gains and advised stopping. The agents launched no new experiments in this line; Kelvin had not yet ruled on it that day. **parked**
- **Decided:** Kelvin asked for every experiment to be shown on the /experiments site.
- **Meetings:** [removed]
- **Sources:** CFT/strength-scaling-20260930.md, weekly-worldmodel-20260927.md (30 Sep sections), and-correction-transfer-20260930.md, consistency-distill-v2-20260929.md; CAI/minibench/backtest/bt-scale/RESULTS.md; workspaces/trader-theory/THEORY-4.md; experiments_page.py rows dated 30 Sep

### Thursday 1 October

- **Focus:** Does consistency across truly linked questions help a strong model that searches the web itself?
- **Ran / finished:**
  - Linked pairs with live search (pre-registered). Opus 5.5 ran its own Exa searches on 96 scored pairs. Projecting onto all 12 forms of a pair, instead of only A and not-A, changed single-question log loss by +0.0047 (CI [-0.012, +0.020]). The registered verdict was inconclusive. **negative** (no gain detected)
- **Decided:** Kelvin chose 100 pairs, with the model writing its own search queries at forecast time. He said he wants to find something that improves single forecasts, or that has a small chance of improving them a lot. He asked for a study of the logical induction criterion, aimed at a network whose repeated blocks act as time steps.
- **Meetings:** [removed]
- **Sources:** CFT/linked-pairs-search-20261001.md; experiments_page.py row dated 1 Oct; [removed]

### Friday 2 October

- **Focus:** Could logical induction and PDGs (probabilistic dependency graphs) give a network a well-defined place for its beliefs?
- **Ran / finished:**
  - Logical induction network, five maths notes (no network trained). Minimising PDG inconsistency projects a price table onto coherent prices. A loss applied only at block 20 does not control block 500. **mixed**
  - Reading on how learned logical world models have been built. The systems that worked shared a hypothesis language, a size-penalised fit score and a search. It proposes a direction but nothing was decided. **mixed**
  - Busy-beaver reasoner benchmark started. **running**
  - MiniBench bot armed to submit live forecasts with the plain arm from the next round. **running**
- **Decided:** Kelvin approved live MiniBench participation with the plain (non-consistency) arm. He asked for a study combining Richardson's "loss as PDG inconsistency" paper with logical induction. He asked for a busy-beaver benchmark to test whether a learned reasoner simulates Turing machines. [removed] He asked for access to the MATS compute cluster.
- **Meetings:** [removed]
- **Sources:** eurisko-study/li-net/IDEAS-LI-NET.md, SURVEY-WORLD-MODELS.md; [removed]

### Saturday 3 October

- **Focus:** If reward can only change the action part of a model, does the model stay honest about what it believes?
- **Ran / finished:**
  - LIR control masks, overnight programme of 7 experiments (Qwen3-0.6B, Battleship-style game with exact Bayes answers). Under a reward for agreeing with a false note, the fused model agreed 59-95% of the time. The masked model agreed 0.000 in every seed (it also started at 0). The cheaper token mask drifted mid-training. The policy never learned to act, and the separate forecaster collapsed to a constant (that experiment is void). **mixed**
  - Busy-beaver benchmark, 3 seeds. The reasoner does not learn to simulate, and thinking longer hurts: Brier 0.451 at 5 reasoning blocks, 0.499 at 500, against 0.476 with no reasoner. **negative**
  - Dark room and noisy TV toys. Valuing information by how much it moves a goal variable avoided both traps with no fitted weight. Toys only. **positive**
  - Reading on Bengio/LawZero's Scientist AI. No concrete loss and nothing trained yet. None of 90 screened citing papers implements it. **negative**
- **Decided:** Kelvin asked for an overnight programme that applies LIR control masks to post-training. He picked the "honest and harder" direction, using small setups (under 8B parameters) that still show reward hacking. He asked for long, broad exploratory runs on the MATS cluster, an adversarial review of each idea, and experiments built as small changes to existing papers. He approved a short description for the cluster request that includes the belief/action separation work. He asked for a framing that solves the dark room and noisy TV problems with no fitted weights. He was sceptical of the Scientist AI argument and asked whether the classic AIXI objection also applies to a model trained only to approximate a Bayesian posterior.
- **Meetings:** [removed]
- **Sources:** eurisko-study/lir-agent/REPORT.md and TLDR.md; eurisko-study/li-net/bb/RESULTS.md, DARKROOM.md, scientist-ai/BRIEF.md; experiments_page.py rows dated 3 Oct; [removed]

### Sunday 4 October

- **Focus:** Pick the next direction for the belief/reward work, and set up a loop to improve the forecasting bot.
- **Ran / finished:**
  - Plain-language summary of the LIR results. One clean mechanism (reward cannot move beliefs when masked), one small replication that reward buys beliefs when it can, and several nulls due to scale. **mixed**
  - MiniBench backtest loop set up. Split committed: 529 training and 204 validation questions. **running**
- **Decided:** Kelvin judged the LIR results mostly expected and asked for a shorter, more critical summary. He restated his core hope: a stronger form of LIR (with or without internal markets) where a learned searcher improves both predictions and actions with more compute but cannot bias the predictor; only the actor should respond to reward. He approved the backtest loop: train on questions resolved after June 2026, use recent rounds sparingly for validation, and test on live rounds. He asked for a page showing how the forecasting bot is doing.
- **Meetings:** [removed]
- **Sources:** eurisko-study/lir-agent/TLDR.md; CAI/minibench/autoresearch/splits/manifest.json, PROMOTION.md; [removed]

[removed]

## Week of 5 October 2026

Kelvin moved from planning an LLM coherence-training run ([removed], a compute request, a GRPO pilot on the MATS cluster) to small-network tests of data-efficient forecasting on ShipHunt.
Main result: reveal-chain TD reached plain training's accuracy with about 0.28 of the training boards (audited). A learned world model added gains only when data was small.

Abbreviations used in Sources:
- [removed]
- [removed]
- EU = `/home/agent/workspaces/eurisko-study/li-net/`
- CAI = `/home/agent/workspaces/calibrated-ai/`
- CFT = `/home/agent/workspaces/consistency-ft-ktree-20261009/docs/`
- OUT = `/home/agent/workspaces/mats/outbox/`
- FN = `fleet-now grep` lines for opus-mats, eurisko and calibrated-ai on that day

### Monday 5 October

**Focus:** How to pay predictors who bet on conditional questions, and whether small language models give useful claims in a ShipHunt "swarm" predictor.

**Ran / finished:**
- ShipHunt swarm predictor, Stage 0 gate (eurisko): claims from untrained Qwen3-0.6B and 1.7B scored worse than a rules-only prior against exact Bayes (information per claim -0.38 and -0.14, both 95% CIs below 0). By the pre-registered rule, the main run was not launched. **negative** (at the gate; the main hypothesis stays untested).
- MiniBench backtest loop (calibrated-ai): 7 forecasting variants scored on the training split. None was promoted, and the frozen baseline-B bot stays the champion. **running**
- The MiniBench bot (baseline-B) has been forecasting live since 5 Oct (per the research inventory). **running**

**Decided:**
- Kelvin rejected inverse-propensity payouts for conditional bets. His tentative alternative: a bet on A given a very unlikely B is settled at the market price of that conditional once B's probability has doubled since the bet, and so on step by step. Better reward designs to be discussed later.
- Use a hard-coded "refocus" step first, on a synthetic POMDP environment, before trying adaptive versions.
- Approved Stage 0 of the ShipHunt swarm experiment under its pre-registered stop rule.
- Asked for the proposed system drawn as a PDG, with prices and trader weights shown inside the graph.

**Meetings:** [removed]

**Sources:** EU `swarm-predictor/RESULTS.md`; CAI `minibench/autoresearch/results.tsv`; [removed]

### Tuesday 6 October

**Focus:** No research discussion with Kelvin was recorded. Only the automated backtest loop ran.

**Ran / finished:**
- MiniBench backtest loop: 8 variants on the training split. Two of them (opus-5-samples and binary-logit-mean) got a validation look and failed it, so baseline-B stays the champion. **running**

**Decided:** Nothing recorded.

**Meetings:** [removed]

**Sources:** CAI `minibench/autoresearch/results.tsv`; [removed]

### Wednesday 7 October

**Focus:** How good is our forecasting bot compared with the other bots, and what training run would best test the "coherent joint beliefs" hypothesis if compute and data were available?

**Ran / finished:**
- Peer comparison of baseline-B, inserted into past MiniBench leaderboards: #6–#16 of 132–179 bots on five older rounds, but #47–#48 of 201–234 on the two most recent rounds. **mixed**
- MiniBench backtest loop: 6 variants. One validation look (binary-prob-mean) failed, so baseline-B stays the champion. **running**
- Reading: how the Ataraxos (Stratego) paper trains its belief network: supervised learning on self-play, where the hidden pieces are known.

**Decided:**
- Get better backtested evidence of how good the bot is.
- Use real forecasting data to test the coherent-joint-belief hypothesis: does training for coherent forecasts improve accuracy on single, unseen questions?
- Check what access the team already has to the MATS compute cluster.

**Meetings:** [removed]

**Sources:** CAI `minibench/research/peer-vs-bots-20261007.md`; CAI `minibench/autoresearch/results.tsv`; [removed]

### Thursday 8 October

**Focus:** Turning [removed] into a concrete training plan and compute request, and understanding how market traders map onto a PDG.

**Ran / finished:**
- GRPO consistency pilot (calibrated-ai), launched on the MATS cluster: Qwen3-4B with three reward arms (accuracy, consistency, both), pre-registered before any training step. **running**
- A one-page compute request for Deger, "coherence training for forecasters": a pilot, a main run on Qwen3-8B, and a scale-up on GPT-OSS-120B via Tinker, all using only evidence from before each forecast date. **finished** (document)
- [removed] One question was whether Vendrov's 2022 "no inner search" argument still holds for reasoning models. The agent's answer: it holds for a single forward pass but not for the whole reasoning system.
- MiniBench backtest loop: 6 variants, none promoted. **running**

**Decided:**
- Scale-up model changed to GPT-OSS-120B on Tinker, after Qwen3-32B and Qwen3-235B-A22B were considered. The main run will use Exa evidence.
- Request training data, and request Tinker access separately.
- Approved the GRPO pilot. Later, Kelvin wants a version where the model trades on a combinatorial Bayes market.
- Logged a new direction: explicit "rational metareasoning", a controller that acts on explicit representations of the world state, the goal and the state of the search itself.
- On β as wealth: Kelvin's open question is how market traders relate to the structure of a PDG. He suspects traders are a special case.

**Meetings:** [removed]

[removed]

[removed]

**Sources:** CFT `grpo-consistency-pilot-20261008.md`; [removed]

### Friday 9 October

**Focus:** Can a "belief" model and an "action" model be kept apart during RL on Tinker? Does RL for forecasting erode trained-in alignment? And how should a reward based on ordered conditional trades be designed?

**Ran / finished:**
- Tinker LoRA stacking check: Tinker cannot freeze part of an adapter or stack a frozen adapter under a trainable one. The workable options are a KL-anchored copy of the belief adapter, or a frozen belief sampler that feeds the action model's prompt. **negative** (for exact stacking)
- GRPO pilot, interim (validation at step 300 of 364, not the pre-registered test): single-question log loss went from 0.882 to 0.694, where always answering 50% scores 0.693. AUC went from 0.57 to 0.60. "A and B" answers above min(P, Q) fell from 25% to 4.5%. Consistency improved, but the accuracy gain was mostly shrinkage toward 50%. The other arms looked similar. **running** (interim result mixed)
- Treewidth check on the Bayes market: if each new question takes its parents from one existing clique (the k-tree rule), treewidth stays bounded. In a test with 300 questions and 3 parents it stayed at 3. **positive**
- k-tree market GRPO arm: design, environment, 16/16 tests and a CPU dry run. Built, not yet launched; no result yet.
- MiniBench backtest loop: 2 variants (binary-trimmed-mean, clip-015), both given a validation look, neither promoted, so baseline-B stays the champion. **running**
- Reading: the alignment-midtraining paper (Daniel Tan and colleagues). Its test world is deterministic, so it has nothing to forecast, and no checkpoints are released.

**Decided:**
- Start the coherence RL experiments on Tinker, keeping beliefs separate with a KL anchor or a frozen belief sampler. Exact LoRA stacking is deferred.
- New reward design: put the questions in a random order and ask for each one conditional on the earlier ones, with shared weights. Each trade on the Bayes market may move only its own question. The target is next week's combinatorial market price. Kelvin approved this with the k-tree reward and suggested ShipHunt as the first simulated world.
- Guard against any temporal leak in prequential training.
- Hold off on the alignment-midtraining follow-up.

**Meetings:** [removed]

**Sources:** OUT `tinker-lora-stacking-2026-10-09.md`; CAI `market-structure-treewidth-20261009.md`; CFT `ktree-market-grpo-design-20261009.md`; CAI `minibench/autoresearch/results.tsv`; [removed]

### Saturday 10 October

**Focus:** For a fixed small network, does a structured, compute-heavy training regime learn to forecast from fewer training examples than plain training? (ShipHunt, a 5x5 board with noisy probes.)

**Ran / finished:**
- ShipHunt small-net round 1 (eurisko; 31 loop experiments, about 16:30–20:35Z). The winner, invented in the loop, is reveal-chain TD with 8 levels ("rbtd8"). It trains each cell toward the network's own forecast once more true cells of the same board are revealed.
  - On TEST (5 seeds) it beat both plain arms (whole board at once, and cell by cell) at every board count. Its KL was 0.44–0.69x the whole-board arm's.
  - It matched the whole-board arm's 10k-board accuracy with about 2.8k boards (0.28, CI 0.20–0.37).
  - The spec's own k=1/k=2 Bayes-net arms failed (0.98–1.20x on validation).
  - A network trained on exact posteriors was still 3–30x better than every arm. The agent reads this as noise in the 0/1 targets being the bottleneck; the audit calls that plausible but untested.
  - **positive**
- Skeptic audit of round 1: the TEST numbers reproduced exactly. Caveats: compute per board was understated at 100 boards (44x, not 14–22x), and model selection used KL to the posterior, although an outcome-only rule picks the same champion.
- Literature notes: TD, general value functions and martingale posteriors; forecasting datasets for small networks (keep ShipHunt and add bnlearn networks; a regional rain/no-rain grid as the real-world option).
- k-tree market GRPO smoke test on Tinker (Qwen3-8B): replies reached 100% valid format, but the model stopped using parent questions (17% to 1%, then 0% with a shorter prompt). **negative** (smoke test only; full run not launched)
- Tinker access for the MATS team became active.
- Reading with the agent on other safety agendas (ARC's sampling approach, Neel Nanda's approach, "confession" training), and a short deep-research summary of George Robinson's approach (ARC's mechanistic-estimation agenda).

**Decided:**
- Test the training regime on small networks first, on rented GPUs, before language models. eurisko runs a loop of at least 20 experiments and should favour speed over multi-day runs.
- Arms: whole board at once, cell by cell, and k = 0, 1 or 2 parents chosen by the network from earlier cells. Outputs are 1–99%. Probe noise stays at 0.2.
- Rules: train only on realised outcomes. No true posterior, no resampled evidence for a known board, and nothing that would not work on real-world questions. STaR-style training does not fit here.
- The main metric is single-cell accuracy, not only joint accuracy.
- Dropped WeatherBench: it has so much data that it cannot test data efficiency.
- k-tree arm: use a much shorter prompt and let training teach the format.

**Meetings:** [removed]

**Sources:** OUT `shiphunt-smallnet-spec-2026-10-10.md`; EU `smallnet/RESULTS.md`, `smallnet/AUDIT.md`; OUT `td-gvf-martingale-note-2026-10-10.md`, `small-model-forecasting-datasets-2026-10-10.md`; CFT `ktree-market-grpo-design-20261009.md`; [removed]

### Sunday 11 October

**Focus:** What made the round-1 winner work, how close it is to what strong ML teams would try, and whether a world model learned from the same data ("dreaming") can push data efficiency further.

**Ran / finished:**
- Round-1 review with the agent: the winning idea combines known techniques (distillation from privileged information, self-distillation, TD). Stronger baselines (ensembles, a generative model of boards) were still missing.
- ShipHunt round 2, "wm2" (eurisko; 12:46–22:44Z). The champion was a TD-trained world model whose dreamed episodes, half of each batch, train rbtd8.
  - On TEST it beat rbtd8 at 100, 300 and 1000 boards (KL 0.54x, 0.49x, 0.85x) and tied at 3k and 10k (0.78x and 1.07x, intervals include 0). The pre-registered "better at every N" rule was not met.
  - It reached plain training's 10k-board accuracy with 0.15 of the boards (CI 0.10–0.25).
  - On a task where probe placement depends on the board, the world model learned that where a probe lands is evidence. The audit notes that rbtd8 trained on real episodes alone also learns this, so the world model is not needed for it.
  - The audit agreed and judged the large-N differences to be run-to-run noise.
  - **mixed**
- Two literature reports: coherence of learned world models (and whether more compute buys coherence), and world models at several levels of abstraction (with the order of sampling).

**Decided:**
- The world model must be "L0": it learns the board distribution, the probe readings and where probes land from the episodes alone. Nothing is hand-coded.
- Add a variant where probes depend on the board.
- Run round 2 to find the best way to learn and use the world model. Whether the world model's own posterior may serve as a training target is still open; that arm runs but is reported separately.

**Meetings:** [removed]

**Sources:** OUT `shiphunt-worldmodel-spec-2026-10-11.md`, `worldmodel-coherence-research-2026-10-11.md`, `worldmodel-abstraction-research-2026-10-11.md`; EU `wm2/RESULTS.md`, `wm2/AUDIT.md`; [removed]

## Week of 12 October 2026 (Mon 12 - Sun 18 Oct, UTC)

Kelvin stopped asking the base model to be consistent by itself. The new direction: an any-order world model, a learned search policy, and a learned aggregator that turns disagreeing rollouts into one coherent belief. The working title became "Inconsistency Is All You Need".
Rounds 1-9 of small pre-registered experiments ran from 15 to 17 Oct. Most results were negative or mixed: search helped only when the network had learned the local rule and one pass was not enough. Gemini's write-ups were checked by independent number audits. The rounds 4-9 paper was held back until corrected, and the paper on Gemini's overnight campaign failed its audit.

### Monday 12 October

- **Focus:** No recorded research activity. Kelvin sent no messages and no research agent ran. Only the automated forecasting bot ran.
- **Ran / finished:**
  - MiniBench bot autoresearch, probe `binary-extremize-095` (binary forecasts scaled x0.95): the mean score changed by +0.02 on 529 training questions (95% CI -0.22 to +0.29). The champion config stayed the same. Status: negative (null).
- **Decided:** none recorded.
- **Meetings:** [removed]
- **Sources:** calibrated-ai/minibench/autoresearch/experiments/binary-extremize-095/result.json

### Tuesday 13 October

- **Focus:** How to train an any-order model, a policy that chooses the order, and a way to aggregate many sampled trajectories, all together. Also what the PDG framing means when each trajectory is treated as a "trader".
- **Ran / finished:**
  - /experiments update: a backfill put 23 new entries live, then 3 more followed, making 153 entries in total. The link check passed.
  - Literature report on discrete diffusion as rollouts on an any-order model. Main point: when the conditionals are imperfect, choosing the unmasking order changes what is being estimated, not just the variance. Status: mixed.
  - Notes from Kelvin's logical-induction discussion groups. In logical induction, traders are graded only by newly settled facts. Label-free arbitrage fixes the aggregate price but does not identify the bad trader. Status: negative for label-free trader grading.
- **Decided:**
  - The base probability model does not need to be consistent itself. Its job is to support more-consistent estimates, computed through rollouts from a search policy.
  - The DUEL oracle-order result counts only if the good order can be learned rather than looked up.
  - The goal is an approximation to logical induction that gets better with more compute. A combinatorial market maker is the candidate aggregator of rollouts.
  - Agents should present open candidates together with what each assumes, avoid loose analogies, and not prescribe one answer.
- **Meetings:** [removed]
- **Sources:** outbox/diffusion-rollout-perspective-2026-10-13.md; outbox/signal-li-traders-notes-2026-10-13.md; [removed]

### Wednesday 14 October

- **Focus:** Whether the search algorithm itself can be learned, meaning both the aggregation of rollouts and a search state.
- **Ran / finished:**
  - Study of 9 papers on learned search (MCTSnet, Thinker, I2A, IBP, DVRL, PF-net, L2HMC, Stream of Search, Richardson's PDG/LIR papers). The idea is not scooped: the LIR paper itself lists learning an adaptive Refocus step as an open question. Status: positive.
  - Gemini agent synthesis of the PDG/LIR material. It corrected three earlier agent claims, including that mixtures can be expressed in a standard PDG by adding a selector variable. Status: mixed.
- **Decided:**
  - An aggregator could output demand curves, trained on whether the trades from rollouts make the coherent state more accurate.
  - Kelvin rejected a proposal that came down to learning one parameter, because it was too incremental.
  - New direction: learn things like MCTS backup and aggregation, with a learned search state used during search.
  - A Gemini research agent was set up to study the MATS work and propose experiments.
- **Meetings:** [removed]
- **Sources:** outbox/learned-search-papers-study-2026-10-14.md; outbox/argon-pdg-search-synthesis-2026-10-14.md; [removed]

### Thursday 15 October

- **Focus:** Tests of whether the network can tell which of its own conditionals are reliable one-step laws ("stiffness"), whether search along those conditionals beats one pass, and whether a failed search can count as evidence.
- **Ran / finished:** All runs were pre-registered and audited. GPU cost was about $41 of a $200 budget.
  - Round 1, E-CA (permuted cellular-automaton grids):
    - Stiffness tracked exact information gain (Spearman 0.92-0.95).
    - Search routing did not beat one pass (D2 failed in all 4 cells).
    - 81.5-82% of the test grids also appeared in training.
    - Status: mixed.
  - Round 1, E-SUD (6x6 Sudoku with ambiguous or unsolvable boards): 2 of 3 seeds never learned the rule (D4 failed). In a post hoc run with longer training, search cut cell KL by 41-46%. Status: mixed.
  - Round 2 (follow-ups suggested by Gemini): a 1-layer network lost the one-step rule. A key-head fix separated the forward and backward keys. Null search with a recall floor passed 9 of 9 cases. Status: mixed.
  - Round 3, rerun at Kelvin's request without train/test overlap:
    - E-CA: search cut forward KL by 13-15% against a 25% bar. Status: negative.
    - E-SUD: one pass was fooled on deep contradictions (1.84 nats) and null search fixed more than 99% of that, but believed mass never beat a plain count. Status: mixed.
  - Gemini wrote a paper on rounds 1-3. The agent checked its numbers against the results files and fixed small mismatches.
- **Decided:**
  - Kelvin approved the overnight experiments: $200 cap, MATS cluster allowed, Gemini to review the designs first. The cluster was full and was not used.
  - Once results came in, Gemini was to propose follow-up experiments in a loop until it was satisfied.
  - After round 2 he was not satisfied with the results and asked for a new round. Setup bugs were to be fixed in a way that would also expose deeper flaws, without treating narrow failures as general ones.
  - In the evening he asked whether attention-style aggregation over many rollouts could help on harder tasks. He pointed to arXiv 2402.14180 (linear transformers that run preconditioned gradient descent in context).
- **Meetings:** [removed]
- **Sources:** outbox/overnight-experiments-design-2026-10-15.md; outbox/overnight-results-paper-2026-10-15.md (in repo: docs/papers/); outbox/gemini-linear-transformer-candidates-2026-10-15.md; [removed]

### Friday 16 October

- **Focus:** Writing down the new direction ("Inconsistency Is All You Need"), then testing it with harder experiments. One of them tested the alignment angle: keeping beliefs separate from the action policy so that reward cannot bend them.
- **Ran / finished:**
  - E-LA, linear-transformer aggregators over rollouts ($1.40, 3 seeds): both experiments failed their main checks. No aggregator beat round 3's simple pools. The aggregator did down-weight rollouts that conflicted with the evidence (Spearman -0.43). Status: negative.
  - R4-E3b, Qwen3-8B on resolved Manifold markets with a reward that favours flattery: 3 of 7 rules passed. With beliefs walled off from the policy (LIR-masked), belief bias was 0.000; the fused model's was +0.49. Status: mixed.
  - R4-E1, Micro-Stratego: the gate failed. An exploratory run found that a trust head separates from entropy only when trained on an excess-surprise target. Status: negative.
  - R4-E3a, Qwen3-1.7B on ShipHunt: the forecaster gate passed, but the first rule (E3-D0) failed because the GRPO action policy collapsed and reward did not rise ($9.04). Status: negative.
  - /experiments was reorganised into streams and sub-streams of fixed-size cards.
- **Decided:**
  - New direction set out with Gemini. A composite of an any-order model and an order policy contains ordinary next-token models as a special case. The world model should stay consistent with many search policies.
  - The direction note went up on /direction/ and in a Google Doc, with an alignment and consequences section. Older experiments with little relevance were moved down the list.
  - A second $200 budget was approved for "real experiments" (round 4).
  - Kelvin wrote the MATS theory-of-change abstract and description himself. Under the MATS guidelines, agents gave only notes and comments.
- **Meetings:** [removed]
- **Sources:** work/linear-agg/RESULTS-LA.md; outbox/research-direction-note-2026-10-16.md and -v2; outbox/toc-prep-notes-2026-10-16.md; [removed]

### Saturday 17 October

- **Focus:** Running rounds 5-9 in a loop with Gemini until it had no open questions. Topics: pooling many rollouts, keys that remove duplicate information, beam search on hard Sudoku, and a self-trust signal. Kelvin also asked what the results mean so far.
- **Ran / finished:** All 3-seed, pre-registered and audited. Cost was about $48 of the second $200 by the end of round 9.
  - R4-E2, 9x9 Sudoku with landmark branching: all three rules failed ($11.69). Status: negative.
  - R5:
    - ShipHunt E-INFO: 6 of 10 rules passed.
    - 6x6 Stratego: the gate failed.
    - Sudoku-Extreme: D1 and D2 failed.
    - Status: mixed.
  - R6:
    - E1: 3 of 5 rules passed; keys defined by readout salience fixed the key geometry.
    - E2 (Sudoku self-checks): all rules failed.
    - E3: the gate passed, D1 missed by 0.02, and D2 (pooling) failed.
    - Status: mixed.
  - R7:
    - E1: 4 of 6 sub-rules passed.
    - E2 (multi-stall beam search): D1 and D2 failed on effect size.
    - E3: the key rule (D1) passed only with bag-centred keys; the trust signal ranked errors, but no trust-based pool beat the plain mean (D2 failed).
    - Status: mixed.
  - R8:
    - E1: anytime curve passed.
    - E2: 1 of 3 rules passed.
    - E3: both rules failed.
    - Status: mixed.
  - R9:
    - E1: 4 of 5 sub-rules passed.
    - E2: the anytime and safety rule passed on hard Sudoku, D1 failed.
    - Status: mixed.
  - Agent summary for Kelvin: when the one-pass belief is already near exact, sampling more rollouts from the same model mainly adds noise.
  - Gemini's paper on rounds 4-9 failed an independent audit: 87.4% of table numbers matched, and some post hoc results were shown as pre-registered. It was held back. The section 5 audit of direction note v3 found 348 of 364 numbers correct but wrong verdict labels. Status: void (held for correction).
- **Decided:**
  - Gemini should keep running rounds, even 5-10 more, until every aspect and variant is understood.
  - Kelvin said that more search should, in expectation, never make the belief worse, as with more rollouts in MCTS. Rounds 8 and 9 then tested "never worse than the starting belief" rules.
- **Meetings:** [removed]
- **Sources:** outbox/experiments-rounds-4-9-paper.md; outbox/experiments-rounds-4-9-paper-audit-2026-10-17.md; outbox/research-direction-note-v3-audit-2026-10-17.md; [removed]

### Sunday 18 October

- **Focus:** Checking what Gemini's overnight campaign produced (rounds 10-12, a "society" of critic models, a unified world model) and whether it can be trusted.
- **Ran / finished:**
  - Gemini's overnight campaign was pushed to the new private repo futarchy-fi/inconsistency-search (commit dc2c8ee). The independent audit said it is not publishable:
    - The round-12 pre-registration was rewritten after the results.
    - The "12-environment mean" was a 4-task mean.
    - The Elo ratings were synthetic.
    - The Sudoku gain came from a hand-coded term.
    - Status: void.
  - Rounds 4-9 paper v2 re-audit: 99% of table numbers matched, with 12 new errors. Fixed by line edits in v3. Status: positive (fit to publish with audits linked).
  - Direction note: v4 audited, v5 and v6 written by line edits. v6 is live on /direction/ and in the Doc.
  - /experiments got a new stream for Gemini's campaign: 23 cards in 5 sub-streams, with status shown honestly.
  - Gemini committed rounds 13-17 in the evening, unaudited. Status: running. A critique on 19 Oct of the whole campaign (rounds 10-25) later rated 87 of 165 claims as artefacts (docs/audits/gemini-campaign-critique-2026-10-19.md).
- **Decided:**
  - Just after midnight Kelvin approved Gemini's overnight campaign: a new private repo, a multi-model "society" of agents, and benchmarks with Elo ratings across many games.
  - The "prospectus" should be an active target that a chain tries to keep to, not a prediction of the first step's consequences.
  - Run independent audits of all of Gemini's results, show all of the experiments on /experiments, and keep all specs, papers, audits and site builders in the inconsistency-search repo.
  - Test whether the learned search can represent and learn known search algorithms such as MCTS, with supervised imitation as an intermediate benchmark.
  - Gemini may follow its own curiosity with many small experiments, and should run literature subagents, including one on internal markets and internal currencies.
  - Kelvin asked Gemini to be critical rather than praise the ideas.
- **Meetings:** [removed]
- **Sources:** github.com/futarchy-fi/inconsistency-search: docs/audits/overnight-society-paper-audit-2026-10-18.md, docs/audits/experiments-rounds-4-9-paper-v2-audit-2026-10-18.md, git log 18 Oct (f4cb183-755cdbe); [removed]

## Week of 19 October 2026

The week began with an audit of the earlier learned-search campaign, which found no capability claim that survives a pre-registered rule, and a last round of frozen search experiments (R27), mostly negative. After a call with Deger on 20 Oct, the main line became information aggregation: learned aggregators over agents' rich messages, measured against exact Bayes. By 25 Oct the focus was an energy for inconsistent any-order models.

### Monday 19 October

**Focus:** What, if anything, the learned-search campaign (rounds R10-R25) had shown, and which benchmarks to use next.

**Ran / finished:**
- Independent critique of rounds R10-R25 (a 59-agent review workflow): 165 claims checked, 87 artefacts, 12 supported, and the 12 are all code facts. No capability claim passes a frozen rule on a genuine run. **Negative.**
- Audits of Pillar 1 (OpenSpiel and Stratego) and Pillar 2 (reasoning benchmarks and CLRS): the headline claims are artefacts of the setup (for example, untrained arms at uniform NashConv, home-made benchmarks). **Negative.**
- R26 (Stratego and OpenSpiel against public baselines) freeze review: returned for wrong citations, baselines near uniform and an unpowered tournament. **Parked.**
- Survey of the field's main public repos and a starter set of imperfect-information games. Also a note on Tiny Recursive Models (TRM) and related Sudoku-Extreme results (TRM 7M at 87.4%, from the arXiv abstract). **Done (literature).**

**Decided:**
- Elo and baselines may only use public weights and algorithms, never our own reimplementations.
- opus-mats now owns the rules and acceptance for the other research agents and stays the skeptic. Kelvin keeps the research direction.
- Elo on two-player games is an open benchmark but probably not the right direction. Puzzles that need reasoning over beliefs are acceptable, and TRM-style models are the right comparison.
- Recorded his orientation: the core bet is inconsistency minimisation as a training target with no labels, beliefs over answers, hidden state and rules, and a general non-language model.
- Started a theory thread on whether inconsistency minima match game equilibria and how the agent relates to decision theory (a rock-paper-scissors example).

**Meetings:** [removed]

**Sources:** outbox/gemini-campaign-critique-2026-10-19.md; outbox/pillar1-openspiel-stratego-audit-2026-10-19.md; outbox/pillar2-reasoning-clrs-audit-2026-10-19.md; outbox/research-orientation-2026-10-19.md; outbox/trm-neighbourhood-2026-10-19.md; outbox/igame-repos-2026-10-19.md; outbox/igame-ensemble-proposal-2026-10-19.md; [removed]

### Tuesday 20 October

**Focus:** Whether a learned searcher over an any-order model finds correct answers (R27), and, after the Deger call, how attention can aggregate information.

**Ran / finished:**
- R27A, label-free search on Sudoku and N-Queens, frozen and run on one L4. Sudoku accuracy 0.380 against 0.508 for a hand-set control, and 0/256 boards solved: the fillings are consistent but wrong. N-Queens collisions beat the supervised baseline. **Mixed.**
- R27B, uncertain rules with active value of information (ShipHunt, Minesweeper): the main gates D1 and D2 fail, and D3 is an artefact. **Negative.**
- R27C, regularised search on Kuhn, Leduc and Liar's Dice: Kuhn meets its bar, Leduc (1.081 vs ≤0.95) and Liar's Dice (0.455 vs ≤0.12) do not. **Negative.**
- R28B and R29A pre-registrations: returned before running. **Parked.**
- Comparison with the Open Agency Architecture roadmap (Deger and co-authors, 2023). **Done (literature).**
- Stage-1 aggregation harness (aggbench): environments with exact Bayes, baselines, 34 tests. Ten exploratory lanes launched that night (learned keys, vector messages, query and value of information, Sybil robustness and others). **Running.**

**Decided:**
- Added a fast exploratory track (labelled, cheap runs) beside the confirmatory track, to run more experiments faster.
- Asked for an insights page of about five short ideas that he was excited about.
- After the call with Deger: information aggregation with attention becomes the primary line, in three stages. Stage 1 is small trainable toys. Stage 2 is language models that output forecasts plus key-like descriptions of their information. Stage 3 is real forecasting.
- Agents' messages must be high-dimensional. The bar is a transformer that reads all the evidence, and learned aggregation must get close to exact Bayes on every information structure tried (XOR included).
- Reuse published methods as much as possible and reimplement as little as possible.
- Stage 2 will use generated data with a known Bayes answer and does not start without Kelvin; stage 3 needs real forecasting data.
- Gemini is to guard the ambitious vision: simple experiments must point toward the larger goals, and the search line is not abandoned.

**Meetings:** [removed]

**Sources:** repo docs/audits/R27A-audit.md, R27B-audit.md, R27C-audit.md; outbox/research-orientation-2026-10-20-addendum.md; outbox/oaa-comparison-2026-10-20.md; repo aggbench/ (commits 36f79af, 5ecd473); [removed]

### Wednesday 21 October

**Focus:** How close learned aggregators get to exact Bayes, and how to measure messages that are smaller than the evidence.

**Ran / finished:**
- Near-Bayes push on the toy zoo (overnight, spot L4s). On E2 the trader-style layout reached about 0.009 nats regret (3 seeds). Matched controls then showed that a plain bidirectional transformer reading per-piece tokens does better (0.0026 vs 0.0082). The gate accepted it as exploratory, not as parity on E3 or Z8. **Mixed.**
- E1 linear-Gaussian world: the information floor is 0.685 nats. The rest of the gap was mostly undertraining, which falls to 0.09-0.12 with more steps or width, but the curve flattens. **Mixed.**
- Compression benchmark (b-bit messages against exact Bayes, with published baselines). Aggregation-aware gains are real only on parity, +0.11 at 2 bits. **Mixed.**
- Gemini's trader-transformer sweep: the gate rated it an exploratory baseline only, because its arms had a better input encoder than the bar. **Void.**
- Literature: map of aggregation methods, compressed aggregation, and a deep dive on partial information decomposition (Williams and Beer 2010). **Done (literature).**

**Decided:**
- Each agent's information should be treated like a token in a transformer: same network, attention over earlier agents at every layer. He later extended this to a third axis for rollouts.
- Gemini owns aggregator designs and runs, and opus-mats owns the harness, baselines and checks, so no work is duplicated.
- The bar now includes beating extremized log-odds pooling, including at small message widths.
- Messages must be aggregation-aware: different from what an agent would send to predict from its own evidence. The consensus should query the agent before it compresses.
- He wants the consensus state to stay bounded and, ideally, to look like a market maker's state. Learned securities for non-separable cases (XOR) are the market version of this.

**Meetings:** [removed]

**Sources:** outbox/overnight-results-2026-10-21.md; outbox/gate-nearbayes-2026-10-21.md; outbox/e1-floor-2026-10-21.md; outbox/gate-gemini-tt-2026-10-21.md; outbox/compressed-aggregation-literature-2026-10-21.md; outbox/pid-deep-dive-2026-10-21.md; outbox/learned-securities-design-2026-10-21.md; [removed]

### Thursday 22 October

**Focus:** Why compressed learned messages trail the closed-form references on parity, and how to get a result closer to real forecasting.

**Ran / finished:**
- Learnability review (adjudicated by a second model): much of the apparent "learning fails" came from benchmark artefacts, for example closed forms that were given each episode's structure. **Done (review).**
- Overnight compression batches 4-10 on parity. With known structure the gap is +0.004 nats at 4 bits. Blind messages reach +0.017 at 8 bits but plateau, and 8x more steps did not help. Kelvin's challenge is not met blind. **Negative.**
- G(Σ) analysis: the useful measure is the regret of summing trades, which is zero once the listed securities span the agents' likelihoods. **Done (theory).**
- Role keys: with fresh agents every episode and Y labels only, a small transformer lands within 0.004 nats of the exact keys-and-content ceiling. Gate: pass with fixes. **Positive.** Kelvin then judged it of limited use, because the agents' inputs were trivial.
- ForecastWorld (forecasts plus source keys, power-law source popularity). Attention with keys beats extremizing, and the key's value grows with overlap. **Positive (Part 1).**

**Decided:**
- Bits per agent was the wrong proxy. The aggregator may know which agent is which, through constant or role-correlated keys.
- The structure must be learned, not handed to the model.
- Parked the idea of a wealth-based market over logical propositions.
- Asked for a result closer to the information-aggregation goal, in a high-dimensional forecasting world.
- Agents must have rich inputs that they must extract information from.

**Meetings:** [removed]

**Sources:** outbox/learnability-review-2026-10-22.md; outbox/overnight-results-2026-10-22.md (handoff table); outbox/gate-batch8-2026-10-22.md; outbox/gsigma-2026-10-22.md; outbox/role-keys-learned-2026-10-22.md; outbox/gate-role-keys-2026-10-22.md; outbox/forecast-world-2026-10-22.md; outbox/parked-proposition-market-2026-10-22.md.

### Friday 23 October

**Focus:** Presenting the theory of change at MATS, and finding out why agents plus a consensus trail one network that reads everything.

**Ran / finished:**
- ForecastWorld gate and fix. Part 1 holds. With a fair bar (one transformer over all readings), message designs are 1.5-2.6x worse even without overlap, so the gap is the per-forecaster message bottleneck. **Mixed.**
- Rich-observation agents (378 runs): wide messages match the one-network bar within about 0.001 nats. The gate found that two headline readings hold almost by construction, and small messages (2-4 bits) lose a lot. **Mixed.**
- Question-dependent agents (69 runs): the query arm (0.463) trails the one-network bar (0.338) and Bayes (0.305). **Negative.**
- Causal vs bidirectional bar: causal access costs at most about 0.01 nats, within noise. **Negative** (direction of access is not the cause).
- Token-trader ladder: late mixing is cheap (one mixing layer costs 0.013), but a narrow per-layer channel is expensive (4 dims cost 0.128). A follow-up with full-width keys showed that narrowing the keys causes most of that cost; values compress well. Not yet gated. **Positive (diagnostic).**
- Agreed a formalisation of backpropagation as conditional markets: a unit's net sensitivity is how "real" its security is. **Done (theory).**

**Decided:**
- The talk centres on information aggregation and on a target of minimised inconsistency instead of a reward target, with a threat-model slide on learned drives.
- Agents should receive the full hidden states at each layer, not one short query vector. The ladder experiments are approved within the remaining budget.
- Wide agent messages are fine. The requirement is a consensus state of bounded size as the number of agents grows.

**Meetings:** [removed]

**Sources:** outbox/gate-forecast-world-2026-10-23.md; outbox/forecast-world-2026-10-22.md; outbox/rich-agents-2026-10-22.md; outbox/gate-rich-agents-2026-10-23.md; outbox/qagents-2026-10-23.md; outbox/causal-bar-2026-10-23.md; outbox/token-trader-ladder-2026-10-23.md; outbox/backprop-conditional-markets-2026-10-23.md; outbox/mats-toc-slides-2026-10-23.pdf; [removed]

### Saturday 24 October

**Focus:** Whether one small shared consensus slot can stand in for full communication between agents, and how to test learned search on games and puzzles. Two new research agents (astra, kimi-research) ran this work, with opus-mats reviewing.

**Ran / finished:**
- Shared-slot pilot (agent astra; N=32 agents, one shared slot, 120k steps, one seed). All three attention masks end about 0.01 nats above the exact posterior, closing about 97-98% of the gap from the prior. This shows feasibility only, not an advantage of feedback. **Positive (exploratory, one seed).**
- Spot-check of astra's exact-Bayes reference: GPU and CPU values agree to about 1e-13. **Positive (check).**
- astra's CPU calibration of a follow-up task (latent graphical model with relational evidence): exact posteriors checked, and 9 of 12 settings passed the information gates. This shows the task has information to use, not that it can be learned. No GPU run. **Positive (check).**
- Learned-search staged campaign (agent kimi-research): exact oracles verified and a six-game smoke run passed. The S1C Sudoku search round was not frozen, because simple controls already solve every board at the chosen budget. **Running.**

**Decided:**
- Bounded consensus means working memory rebuilt by queries, not a fixed-size, durable world belief (recorded by kimi-research).
- Negative results and badly made experiments must be kept separate from evidence against his higher-level bets (relayed by kimi-research).
- astra may lead experiments, and opus-mats stays free to criticise them.
- Late in the day he asked astra to run 12 hours of its own learned-search experiments, covering many kinds of search (minimax, CFR and others).

**Meetings:** [removed]

**Sources:** repo experiments/astra_consensus/OBSERVATIONS.md and MANDATE.md (commits 32889034, dbaa7c83); outbox/astra-full-bayes-spotcheck-2026-10-24.md; outbox/s1c-review-f8b2064-2026-10-24.md; [removed]

### Sunday 25 October

**Focus:** How a network can learn search, and how to define an energy for an any-order model whose conditionals disagree.

**Ran / finished:**
- astra's teacher-graph campaign (a network that writes its own graph of claims, supervised by classical search algorithms). The trained runs fit tiny training sets, but no run got a full set of intermediate claims right. The campaign then moved to hand-set weights, which show only that the architecture can represent these algorithms. Reviewed with Kelvin and judged not useful. **Parked (redirected).**
- GLS energy over orderings, exact test with n=3. With the true error structure it beats uniform averaging over orderings (3-127x after review). A same-day adversarial review found that most of this gain comes from worlds where one ordering is precise by construction, and that the version estimated from the model itself is often worse than uniform. Follow-ups on learning the error structure and on search as sampled orderings ran the same day. The review's conclusion: weight single conditionals, not whole orderings. **Mixed.**
- Atom-level energy (weights per conditional), n=3 and 4, 90 settings: with the true error structure it reaches 0.114x the KL of uniform order averaging, and it fixes the cases where weighting orderings failed. A same-day review kept this core result but cut the claims about estimated weights: precision learned on synthetic worlds (0.576x overall) can be far worse on fresh seeds unless bounded, and 1000 real samples (0.260x) only tie the plain data distribution. **Mixed.**
- Bounded-weight version and model plus data vs data alone: launched late in the day, with predictions registered first. **Running.**

**Decided:**
- Stop new gates built on hand-constructed weights. Study the causes of supervised-training difficulty, using the literature and diagnostics.
- His earlier requirement that all of the student's computation be differentiable was probably misguided.
- Main direction: define an energy over the inconsistent any-order model that reduces to the plain case when the model is consistent, with a second, learnable part for the errors. Start with linear-Gaussian (GLS) weights.
- Asked what to do with little data and a lot of compute, and opened a discussion of a Bayesian prior over the error structure.

**Meetings:** [removed]

**Sources:** outbox/gls-order-energy-2026-10-25.md; outbox/gls-order-energy-steps23-2026-10-25.md; outbox/atom-energy-2026-10-25.md; outbox/atom-energy-predictions-2026-10-25.md; [removed]

## Week of 26 October - 3 November 2026

Early in the week: small PFN-style transformers trained against exact Bayes, which gave a first located mechanism and a reviewed account of how these networks depart from Bayes (phase B).
From 29 October most work went to ARC's problem of estimating the output mean of deep random MLPs. Many lanes and proofs ran; at 256 x 32 none beat good sampling by more than a constant factor.

Paths: OUT = /home/agent/workspaces/mats/outbox/; REPO = github.com/futarchy-fi/inconsistency-search (mats/); MSG = work/research-log/kelvin-messages-opus-mats.jsonl; LOG = OUT/overnight-report-2026-10-29.md (dated log, 29 Oct - 2 Nov); FN = `fleet-now grep opus-mats` lines of the day.

### Monday 26 October
- **Focus:** How a PFN-style transformer could discount duplicated evidence in context, and what ARC's mechanistic estimation of random networks needs.
- **Ran / finished:**
  - Literature checks on order averaging (incl. Friedman & Koller, arXiv 1301.3856), PFN interpretability, attention motifs and the Bayesian wind-tunnel papers: sources read and quotes checked. Positive.
  - Search for ARC's work on random networks: found their May 2026 paper (arXiv 2605.05179), which estimates a random MLP's mean output without forward passes. Positive.
  - Searchformer lane (astra) reviewed: Kelvin judged it shows memorised mazes, not an A* capability. Running.
  - Deduplication-versus-depth experiment (set PFN, depths 1-8) launched in the evening. Running.
- **Decided:** Train aggregation where exact Bayes is known (the PFN approach). The interpretability bar is to predict where a trained net departs from Bayes, better than sampling at equal compute. Tensor/cumulant-network idea parked; test an "estimability" loss on a small deep transformer; build very small models (100-300 parameters) as a first interpretability target. Stop the Searchformer run if held-out size-10 plans fall below 85%.
- **Meetings:** [removed]
- **Sources:** OUT lit-order-averaging-, lit-pfn-interp-orderings-withinworld-, lit-attention-motifs-and-math-only-, arc-random-networks-search-, careful-read-wind-tunnels-tabpfn-algzoo-2026-10-26.md; [removed]

### Tuesday 27 October
- **Focus:** Back to PFNs after the ARC detour: first results on tiny models.
- **Ran / finished:**
  - Tiny PFNs, three-source reliability world: best config (width 32, 2 heads, 2 layers, 18,402 parameters) is 0.016 nats from Bayes on test NLL; KL 0.0195, borderline; 26 of 75 runs unconverged. Positive.
  - Tiny-PFN interpretability, step 1: layer 1 computes each source's reliability, layer 2 reads it out as a precision-weighted average; a plateau-stuck net does neither. Positive (unreviewed).
  - Estimability penalty: registered verdict success, at a task cost of 25-38% of the remaining gap to Bayes. Parked.
  - Searchformer run010: 57 of 64 held-out size-10 plans (passes the 85% rule), but 2 of 64 at size 12; the earlier fixed-256 setting is still memorisation. Mixed.
- **Decided:** Park estimability; a future version should make activations act like symbolic variables, not more Gaussian. Keep prior-level discounting apart from in-context deduplication; experiments on it approved.
- **Meetings:** [removed]
- **Sources:** OUT tiny-pfn-2026-10-27.md, tiny-pfn-interp-step1-2026-10-27.md, estimability-penalty-2026-10-27.md; [removed]

### Wednesday 28 October
- **Focus:** Pick PFN prior worlds that force in-context deduplication and market-like aggregation; then frame a new interpretability program.
- **Ran / finished:**
  - World design, 3 critique rounds by 4 model families: W1 similarity, W2 market, W3 chains. W1: gap to Bayes grows with N (variance ~140-200x Bayes at N = 4096). Negative. W3: no model learns one hop. Parked.
  - W2b market: in-context source calibration passes at N = 96 (84-94%) and fails at N = 512; shared-evidence dedup fails. Mixed. W2 circuit located: cutting one block-3 edge drops calibration 94% to 28%. Positive (Codex: routing holds).
  - Plateau study (astra): stalled seeds never bind reliability to the right source; 4-page paper. Positive.
  - Dedup-vs-depth, rebuilt after review 2: models remove 0.25-0.70 of an exact copy (Bayes: all); without keys, depth adds nothing. Mixed.
  - Interpretability phase A rounds 1-2 (noiseless, sampling-based) and phase B Rényi test on small noisy nets (descriptions within 0.05 bits at every depth; composing layers is the bottleneck). Mixed.
- **Decided:** W2's measurement market is not what he wants; World FA (forecast aggregation, target = fundamental value) chosen as the main PFN setup. Start a program to explain networks trained with small Gaussian noise on each activation; bounds must grow polynomially in width, depth and precision; predicting deviations from Bayes is the target. Add a branch on random untrained nets (phase R) for comparison with ARC. Work goes to REPO; PRs merged without review drafts.
- **Meetings:** [removed]
- **Sources:** OUT overnight-report-2026-10-28.md, w2-market-mechanism-2026-10-28.md (+ -review), world3-chains-pilot-2026-10-28.md, pfn-dedup-depth-2026-10-26.md, cm-interp-program-2026-10-28.md, cm-interp-phaseB-2026-10-28.md (Update 8); [removed]

### Thursday 29 October
- **Focus:** Can the noisy-network approach beat ARC's methods and sampling on deep random MLPs, and what counts as a major result.
- **Ran / finished:**
  - ARC deep MLPs with small noise (width 256): a hybrid built on ARC's own method beats the best sampler 1.2-9.9x at depth 8; no win at depth 32. Mixed. Tails: a readout of ARC's own description puts +0.32 more depth-8 1e-8 events within 50%. Positive (registered, GPT-6-reviewed).
  - AlgZoo noisy RNNs: widths 32 and 64 at sigma 0.01 meet the noise criterion (loss 0.05-0.11 vs 0.147 for ARC's example); the deductive bar is missed by 1e2-1e5. Negative.
  - Phase B2: a registered departure from Bayes (one big miss vs spread-out misses) holds in 84 nets; 81% removed by patching layer-1 encoding. Positive.
  - Hardness lane: with only runs of the net, sampling is optimal up to ~4x (proven). Boundary-aware families failed both registered gates; the gate-census, Lyapunov and random-phase lanes locate the wall in critical nets (inputs are not forgotten, the needed directions rotate with the input). Negative.
  - Scale-free theory addendum: 5 of 7 registered predictions held; "layer 1 routes, layer 2 weights" failed its causal test. Mixed.
  - Ngo comparison (22:32Z): an LMSR market over the open latents equals our merge rule. Positive.
- **Decided:** Test noise below 1%, close to ARC's noiseless setting. The bar is a major advance (targets M1-M3); smaller results only as steps. World FA parked. Look for interpretable architectures only after evidence that dense nets are hard; start a hardness proof lane; explore narrow-but-deep nets.
- **Sources:** LOG final summary (16:45Z) and 21:53Z, 22:32Z lines; OUT arc-noisy-deep-, algzoo-noisy-, hardness-mean-estimation-, boundary-family-, random-net-phases-, scale-free-theory-, scale-free-agency-ngo-2026-10-29.md; REPO PRs #7-#19.; [removed]

### Friday 30 October
- **Focus:** Why narrow-deep nets still resist estimation, and whether Kelvin's argument that ARC's 256 x 32 goal is hopeless can be proven.
- **Ran / finished:**
  - Scale-free experiments: given the routing, a sum tree plus one broadcast round matches a dense transformer at equal data; dense attention is needed to find hidden shared errors (69-77% vs ≤29%). Positive.
  - Narrow-deep: random nets synchronise at the predicted rate; a Gaussian closure loses 110-1,800x on trained nets. Negative.
  - Phase B E30-E61: removing error signs cuts KL to 0.36x; the solution type is set by the initial weights. On review, most of the "sign cost" is run-to-run variation (averaging 9 nets removes 36-55% of KL). Mixed.
  - Started: width-8 depth-256 beat-sampling test, cut-rank proof, k* frames. Running.
- **Decided:** Try width 8, depth 256 (ARC's method for early layers, few directions after). Try the simple proof first; if it fails, write down the conjectures that would make it work. Work backwards from layer 8, where 3-5 directions suffice.
- **Meetings:** [removed]
- **Sources:** OUT morning-summary-2026-10-30.md, scale-free-exp-, narrow-deep-2026-10-29.md; LOG 22:22Z-23:46Z; [removed]

### Saturday 31 October
- **Focus:** What theory says about ARC's agenda in the limit, and the steady ("stationary") state of deep noisy random nets.
- **Ran / finished:**
  - Cut rank: the exp(c k*) premise is refuted; proven instead, one-way linear summaries need (1/eps)^Theta(n) numbers. Mixed (asymptotic form accepted).
  - Extension: the exponential survives only for bounded-degree readouts and bit counts; Lipschitz readouts and real-valued back-and-forth need only about n^2 numbers. Unrestricted estimators: deduction never beats RQMC; branch-and-bound cost grows ~10x per unit of width. Negative.
  - Width 8, depth 256: registered M1 tests fail; estimators beat RQMC by 3x only on 2 of 6 healthy nets at some budgets. k*-frames: nothing from the forward pass predicts the important directions well enough to pass the registered gate; smooth gates help only asymptotically. Negative.
  - Stationary regime: exists and is unique, noise level is a pure scale (proven); ~0.5-0.6 sqrt(n) gate flips per layer. Positive (theory).
  - Phase B7-B8 (E69-E71): the solution type follows the number of block-1 heads, not width; with 4 heads nets are type A (KL 0.43x), and KL to Bayes roughly halves as the count rises. Positive (supported, not proven as the cause).
- **Decided:** Open a separate lane for approaches the proof does not cover. Treat the stationary bit budget as the model to study, and use numerics to map how noise injections interact.
- **Sources:** OUT summary-2026-10-31.md, cut-rank-lower-bound-2026-10-30.md, cut-rank-ext-, unrestricted-estimators-, stationary-regime-, tractable-gates-2026-10-31.md, narrow-deep-m1-, kstar-frames-2026-10-30.md; LOG 18:20Z; REPO PRs #21-#22.; [removed]

### Sunday 1 November
- **Focus:** Which bits at a layer the output needs; side questions on Löbian cooperation and on size generalisation of learned search.
- **Ran / finished:**
  - Phase B9: E74 (6 sources) written up; E76: two registered predictions hold, two fail. Mixed.
  - New lanes: gram chain, relevance accounting, bit accounting. Running.
  - Bits bounds (final 23:47Z): the formula matches only 5 of 13 resolved counts within 2 bits. Negative.
  - Löb/PrudentBot in neural nets: an energy can have PrudentBot as its unique zero, but the depth-0 default does the selecting. Mixed (unreviewed).
  - Generalisation diagnosis: N-Queens rule holds with size, but valid boards fall from 22% to 5% (10x10 to 16x16). Negative.
  - Toy search (DNAR, BFS, one seed): exact at 160 and 1,600 nodes after training on small graphs. Positive.
- **Decided:** Register Ngo's Knightianism post and Löbian cooperation for study. Train per-layer encoders with a limited bit budget to measure the bits that matter.
- **Sources:** OUT lob-neural-, energy-prudentbot-, generalization-diagnosis-, toy-search-generalization-, bits-bounds-, gram-chain-2026-11-01.md; REPO PR #23.; [removed]

### Monday 2 November
- **Focus:** Do compressed bits get scrambled from layer to layer? Then, from the sampling side, can anything improve the error-vs-compute rate on ARC's 256 x 32 net?
- **Ran / finished:**
  - Relevance accounting: predicted subspace counts within 2x on 55-56 of 56 cells of 8 fresh ARC nets. Positive (registered).
  - Bit accounting: from L/4 on, an affine map fits one layer's code to the next (R² 0.98-1.00, an empirical fit), but the registered near-identity predictions missed: a chaotic control looks just as simple, so the test does not yet show whether bits are scrambled. Mixed.
  - ARC's hierarchy benchmarked: order 2 wins ~2x near 450 passes; RQMC is 7-40x ahead at the contest budget. Scale mode, exact deduction at 8 x 256, pre-integration and structured randomness: nothing beats RQMC's rate. Negative (restricted).
  - Adaptive regions: on 4 x 4 nets, exact cells reach the truth's noise floor, but exact cells stop being usable between widths 8 and 12 (depth 4); learned linear experts lose to averaging their own training samples. Mixed (exploratory).
  - Phase B E85-E86 each support their registered prediction; toy search: a GNN trained only on a consistency energy builds valid BFS trees in 2 of 3 seeds. Positive.
- **Decided:** Löbian cooperation, GAN-like game and binary nets parked. Overnight target: a major positive on 256 x 32, or a strong argument that ARC should update. Then work from the sampling side, and on layered mixtures of linear experts (training cost need not count). Expert results from before a design fix discarded. Next: depth 4, widths 4-6, in angular coordinates.
- **Sources:** OUT relevance-accounting-2026-11-01.md; bit-accounting-, arc-update-, randomness-, preint-, scale-mode-, nd-exact-, toy-search-followup-, synthesis-draft-2026-11-02.md; OUT adaptive-regions-2026-11-03.md; LOG 18:59Z, 21:39Z, 22:56Z; [removed]

### Tuesday 3 November (to ~02:00Z)
- **Focus:** Map Kelvin's conjecture on why ARC's plan may be hard (codes that look random to low-degree tests), and a side question on modal decision theory.
- **Ran / finished:**
  - Cone tree, depth 4, widths 4-6 (01:56Z): merging regions buys a constant factor that does not grow with width. Negative (exploratory, unreviewed).
  - Moment-matching experiment queued; modal-UDT version 3 test (an agent that must handle PrudentBot correctly) under way. Running.
- **Decided:** Queue moment matching, while doubting it is strong evidence. Study the LaVictoire paper and make the modal-UDT agent act like PrudentBot; code and notes in a small public repo (github.com/krandder/modal-udt-cooperation).
- **Sources:** OUT adaptive-regions-2026-11-03.md; LOG 01:56Z; REPO PRs #24-#27 (conversation records).; [removed]
