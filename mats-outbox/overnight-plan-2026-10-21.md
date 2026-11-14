# Overnight plan, stage 1 (opus-mats, 2026-10-21 ~01:30Z until Kelvin is back)

Kelvin (verbatim): "Alright, can I trust you're going to be trying all this overnight, and using all the budget and continuing to do experiments until then?" and "Also, always, with the adversarial check on the vision/guardian?"

## Goal
FIRST PRIORITY (Kelvin, verbatim): "Let us get any toy examples aggregated to near-exact-bayes-optimal first, even on XOR-like things." Before any ablation, get the trader-transformer (and the bidirectional bar) to regret below 0.01 nats, and below 5% of the prior-to-Bayes gap, on E2, Z1 and the XOR zoo env, with unaligned inputs. Training speed does not matter: scale data (the simulators are free), steps, width and depth, and compare outcome loss with prefix-Bayes distillation (labelled). The ablations (A)/(B)/(C)/grid come after. If no setting gets there, diagnose why before going on.

Run Kelvin's trader-transformer design (research-orientation-2026-10-20-addendum.md, last sections) through the empirical questions:
- (A) how much cross-trader attention per layer is needed;
- (B) whether a loss on the final position only is enough, against a loss at every prefix;
- (C) the residual in the token direction: kv, recurrent (the CEO summary) or feedback;
- the 3-D grid with per-head axial masks (Kelvin's default) against the full cone and alternating axial layers.

Every run is scored by regret to exact Bayes and to the bidirectional full_context_transformer bar, on E2, Z1, Z5, Z6 and the XOR zoo env, with unaligned inputs, keys none/noisy/full, random order and 3 seeds.

## Budget
- Kelvin authorised $300 total for stage-1/research GPU (PROTOCOL §10).
- At 01:30Z the list-price cost of all VMs deleted since 10-19 was $12.30, with about 9 VMs still alive. Check: awk over GPU-LOG DELETE cost_usd, groups since 10-19.
- Overnight cap: stop new launches when the list-price total reaches $270; about $30 stays in reserve.
- Each run is capped at $20 unless announced first. Use spot L4s through work/run_on_l4.sh. General (16 cores) is not for training, at 2 threads per job.
- Tinker is not used overnight (stage 2 needs Kelvin).

## Loop, every ~30 min
1. gemini sees Agent Mail only at its next prompt. If its pane is idle and mail is unread, ring the urgent-mail doorbell (hub wiki agent-messaging.md, sections 2-3). Check the builders (trader_transformer, the aggzoo rerun, the aggext table), gemini's Agent Mail and its results, VMs alive and spend, and General's load.
2. Keep the L4s busy with the next cells of the sweep. gemini runs its lane; opus-mats fills gaps that nobody owns and does not duplicate gemini's cells (check results/explore first).
3. **Adversarial gate on every result before it is reported or put on /experiments:**
   - (i) A design-fit reviewer, given Kelvin's verbatim words from the addendum. It checks (a) what a token is, (b) what a layer is, (c) what crosses between traders and at what width, (d) regret to Bayes and to the bar on XOR-type structures. It also checks rich messages vs scalars, aligned vs unaligned, and leaks.
   - (ii) gemini, as guardian of the vision, asked whether the run tests Kelvin's idea or waters it down.
   - A result that fails either is labelled a baseline or an artefact, never a finding.
4. Every ~2 h, post a fleet-now change line with the results so far, the spend and what is running.
5. Once a step is done, start the next one: the (A)/(B)/(C) cells, then the grid layouts, then scaling in the number of traders, then data scaling. Stop only for a named blocker.

## Morning handoff
outbox/overnight-results-2026-10-21.md: the table per question, what passed the adversarial gate, the spend, and what is next.
