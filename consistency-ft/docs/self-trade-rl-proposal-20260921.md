# Self-trade RL on a combinatorial market: proposed run

Draft 2026-09-21 for Kelvin and Jessica. Combines outcome-based RL in the style
of Turtel et al. 2025 (arXiv 2505.17989) and Mantic (Thinking Machines post,
March 2026) with the bayes-market factored LMSR as the world model. Nothing here
is funded or scheduled; costs are estimates at September 2026 rental prices.

## 1. Claims the run tests

Each is falsifiable on held-out weeks and has a control arm.

1. **Search improves beliefs.** The market after K trades by the policy, W_K,
   scores better on settlement log loss than the policy's one-shot output on
   the same evidence.
2. **Amortization works.** Distilling W_K into the one-shot output improves next
   round's one-shot. (1) and (2) together are the policy-improvement operator
   that makes this AlphaZero-like rather than an ensemble.
3. **Coherence transfers.** One-shot outputs become more coherent on formula
   queries they were never trained on (Paleka checks, cross-session overlap),
   without a penalty term, because the market maker enforced coherence during
   search.
4. **Beats plain outcome RL at equal compute.** Settlement log loss on held-out
   weeks beats a Turtel-style single-question GRPO arm with the same model,
   data, tokens and reward, and beats the per-form base-rate table that no
   supervised arm cleared on 20 September.
5. **No memorization.** Accuracy on retired (settled, no-evidence) questions
   does not exceed accuracy on fresh held-out questions by more than the
   base-rate gap.

## 2. Environment

### 2a. Data: two domains

**Domain A, Polymarket, for comparability with Turtel.** Only Turtel's test
set is public: 1,265 yes/no questions closing 13 February to 3 March 2025,
with frozen Exa prompts and the market price at prediction time
(`LightningRodLabs/outcome-rl-test-dataset`; his Table 1 reproduces from it).
It is the held-out set for every arm, giving a direct comparison with ReMax
0.190, o1 0.202 and market 0.151. The training set is rebuilt to his recipe:
about 10k resolved Polymarket binaries from the public Gamma and CLOB APIs,
resolved before February 2025, chronologically ordered, with date-bounded Exa
retrieval and a leak judge in the same prompt format as the test set. His 100k
synthetic questions cannot be rebuilt; the 10k-only ReMax result (0.199) is
the replication target. Multi-outcome events give sum-to-one partitions and
by-date series give monotone chains, so the market game has structure to trade
here too, with fewer derived forms than Manifold.

The test window fixes the base model: Qwen3 was released after it and may have
seen those outcomes, so the clean choices are his DeepSeek-R1-Distill-Qwen-14B
(faithful) or gpt-oss-120b (June 2024 cutoff). The 14B is assumed below.

**Domain B, Manifold, for consistency depth.** The local archive with its
explicit multiple-choice structure and 14k labeled propositions, used for the
stage 0 gate and for the coherence claims, where derived Boolean forms are far
richer. Qwen3-8B, questions from May 2025. Details follow.

**Variables (domain B).** Manifold propositions from the local archive: ordinary binaries
plus multiple-choice options as separate propositions, with the derived Boolean
forms (negation, conjunction, disjunction, by-date chains) as additional
variables. Cohort start is the base model's release week. A question enters the
market at creation week and leaves at resolution or close. Families (same
parent, near-duplicate text, BGE cosine at or above 0.95) are seeded as scopes
so by-date chains are represented as one monotone structure rather than as
independent markets.

**World model.** The bayes-market `FactoredMarket`: junction tree, LMSR with
liquidity b, minimum-KL reweighting on every trade, conditional trades create
structure, treewidth cap 8, resolutions condition the joint. Initial state is
50% for every variable, all independent. Market prices from Manifold are never
in the initial state; they enter only as an evaluation ceiling and, in one
optional arm, as dated evidence.

**Time.** Strict chronological rounds, one calendar week each, single pass, no
replay. Round T: agents trade using evidence dated at or before T. At the end
of the round W_T is snapshotted. At the start of round T+1, questions that
resolved during week T+1 are applied as resolutions to the joint and pay out
every position that references them, including called-off conditional bets;
the questions are then retired from the trading set permanently.

**Evidence.** Two sources, in order of leakage safety.

- The Manifold history archive already on General: comments, bets and
  description edit history, all timestamped. This is dated evidence with no
  retrieval leak and it is free. It is also weak; it carries crowd opinion,
  which is why price-at-T is kept as a separate arm.
- Exa date-bounded search with a leak judge (Turtel's method), roughly $0.02
  per question for search and $0.02 to $0.07 for judge and summaries with
  Claude Haiku 4.5. Precomputed once per (question, week) and stored as static
  prompts, as Mantic does; retrieval is not in the RL loop.

## 3. A turn

A turn is one policy trajectory and one unit of credit.

**State shown to the agent.** A scope: a variable v, its current marginal, the
conditionals and neighbours in its Markov blanket, candidate neighbours from
the same family and from BGE nearest neighbours not yet linked, the dated
evidence for v and its neighbours, the current week, the turn's KL budget, and
the remaining treewidth on that component.

**Action.** Reasoning of up to 1,500 tokens, then a structured block of at most
m trades, each either `trade_to_probability(variable, target, context?)` or a
share trade on a formula. The engine executes them in order. A rejected trade
(over width, malformed, exceeds the KL budget) is returned as feedback and
costs nothing; the agent may retry once within the turn. Adding a link is a
conditional trade; removing one is trading the conditional back to the
marginal.

**Trust region.** The total KL moved per turn is capped at ε. This is the same
constraint PPO puts on policy updates, applied to belief updates. It also caps
extremization, which was Turtel's main failure mode (39% of forecasts in the
extreme bins under plain GRPO), and it is the leakage budget of section 6.

**Coverage.** K turns per round so that every active variable is visited at
least r times, with scopes sampled in proportion to evidence volume and time to
close.

## 4. Reward

Per turn, the sum over executed trades of the LMSR payoff, which is a log-score
difference against the pre-trade market:

- If the trade's target and context variables have all settled by evaluation
  time: the realized payoff b·(log p'(ω) − log p(ω)) on the slice; called off
  if a context variable did not resolve.
- Otherwise: mark-to-market against a reference distribution r at T+1,
  b·[KL(r‖p) − KL(r‖p')] on the slice. Mark-to-market increments telescope, so
  this has the same expectation as holding to settlement under a calibrated
  market; it is the TD bootstrap.

**Reference r.** Two arms. Live: r is W_{T+1} produced by the current policy's
own trades, which is the cheapest and carries the herding hazard (clones
rewarded for moving toward where later clones follow). Frozen: r is the
end-of-round market produced at T+1 by the previous checkpoint trading on T+1
evidence in shadow, a league member, which removes the self-fulfilling channel
at the cost of staleness. Settlement dominates in both arms because it is the
only non-zero-sum profit source.

**No coherence penalty.** The market maker prices every formula by inference,
so there is no internal Dutch book to exploit. Coherence reaches the one-shot
model through distillation (section 5), not through a loss term.

**Guardrails.** Advantage without division by the group standard deviation
(Turtel's modified GRPO, Mantic's choice); a small format penalty for an
unparseable action block, not Turtel's strict 1.0 Brier; rejected trades score
zero.

## 5. Learning

**Policy.** Qwen3-8B with LoRA, thinking enabled, verbalized actions. A 4B
model for the pilot. The 0.6B run showed a model that cannot read criteria text
learns only base rates, so 8B is the smallest size worth the full pass.

**Algorithm.** Group-relative policy gradient, G rollouts of the same turn
state executed on forked copies of the market (the engine's snapshot and
restore), each scored against the reference or settlement. One rollout is
committed to the live market: sampled at random in v1 (on-policy), chosen by a
value head in v2, which is where best-of-G search starts improving the market
beyond the policy. KL anchor β to the previous checkpoint. A league of past
checkpoints plays a fraction of turns for diversity and provides the frozen
reference.

**Distillation.** After each round, two supervised heads on the same model:

- One-shot belief head: given a scope and its evidence, emit the slice tables
  directly; target is the end-of-round W_T slice; loss is KL(W_T slice ‖
  output). This is the AlphaZero policy target and it is what gives an explicit
  belief structure at inference time without running the market.
- Predicted-movement head: predict E[KL(W_{T+1} slice ‖ W_T slice)]; scored
  against the realized movement. This is the value-of-more-compute estimate.

**Schedule.** One pass over weeks. Rounds are the only unit of replay; settled
questions never return.

## 6. Leakage and memorization

1. **Retire on settle, single pass.** A settlement is used once, as the scalar
   reward of past turns, and the question leaves the trading set. Turtel found
   multi-epoch training "learns the future"; here there is nothing to learn it
   from.
2. **Labels never enter as tokens.** They shape the policy only through a
   KL-bounded update, so the information a settlement can write into the
   weights is bounded in nats per update by β and ε. This is a quantity that
   can be reported.
3. **Most reward is the martingale term**, market movement between rounds,
   which is not a label. Resolved questions are a small fraction of the active
   set each week.
4. **Hold out time, not questions.** Train on weeks 1..N, freeze, evaluate on
   weeks N+1..N+8. Families are one scope, so a by-date resolution fixing its
   siblings is inference the model is supposed to do, not leakage.
5. **Measure it.** After training, query retired questions with no evidence;
   report recall accuracy against fresh held-out accuracy as two numbers.
6. **Evidence leakage** is the remaining channel: the Manifold archive is safe
   by construction; Exa needs the judge and its residual is reported per
   Turtel.

## 7. Evaluation

Held-out weeks, market game run in evaluation mode (no updates), plus one-shot
mode without the market. Report:

- Settlement log loss and Brier on questions resolving in held-out weeks, for
  each arm, against constant 50%, the per-form base-rate table, Manifold price
  at T (ceiling), the supervised A3 arm, and the Turtel-style GRPO control.
- Coherence of one-shot outputs on formula queries never trained on, and
  cross-session overlap disagreement.
- Recall versus forecast accuracy on retired questions.
- Extremization rate (share of forecasts in the 0 to 10% and 90 to 100% bins)
  and herding (autocorrelation of market increments across rounds).
- Predicted-movement calibration.

Arms, same model, data and token budget: (1) Turtel-style single-question GRPO
with Brier reward and no market, the control; (2) market game, live reference;
(3) market game, frozen reference plus league; (4) arm 3 plus distillation.

## 8. Synthetic data

Turtel's gains came from 100k synthetic questions generated from news with
later news as the label, and that set is where their reproducibility died. We
have three cheaper synthetic sources, in this order.

- **Derived Boolean forms** are synthetic but exact: negations, conjunctions,
  disjunctions and by-date chains of real questions, free, already built for
  14k propositions and extensible to the 119k embedded ones.
- **The market's own targets**: W_K is a synthetic supervision signal the run
  manufactures each round.
- **Generated questions from dated news**, Turtel and Bosse style, with Claude
  Haiku 4.5 at roughly $0.05 to $0.20 per question; 10k for about $1k. Only
  once the dated-evidence pipeline exists, since these need the same leak
  screening.

Phase 1 uses only real Manifold questions and derived forms.

## 9. Stages and cost

H100 at $2.25 to $3.49 per GPU-hour; rollouts dominate. Assumptions: prompt
2.5k tokens with evidence, 1.5k generated, G = 4.

| Stage | Domain | What it settles | Compute | Cost | Weeks |
|---|---|---|---|---|---|
| 0 | Manifold | Claim 1 with a frozen model: 4 weeks of data, frozen 8B as trader, K turns per round, W_K versus one-shot on settlements. No RL. | 1×H100, ~8 h | ~$50 | 1 (engine fork API, turn loop, evidence from archive) |
| D | Polymarket | Rebuild the 10k training set with Exa and judge; verify the test-set scoring against Table 1 | API only | ~$500 to $800 | 1 to 2 |
| 1 | Polymarket | Turtel replication: ReMax and modified GRPO at 14B on the 10k set, scored on his test set; target 0.199. This is the control arm and a result on its own. | 4×H100, ~40 h | ~$400 to $600 | 2 (TRL or verl loop, reward parser) |
| 2 | Polymarket | Market-game arms 2 to 4 at 14B on the same 10k set and test set; claims 2, 4 and 5 | 4×H100, ~60 h | ~$700 to $1k | 1 to 2 (environment wrapper, league, distillation heads) |
| 3 | Manifold | Coherence claims 3 and 5 at depth: full pass at 8B over 70 weeks on the winning arm | 4×H100, ~5 days | ~$1.5k to $2.5k | 1 |

Stage 0 is the gate: if W_K does not beat one-shot with a frozen model, the
market is not a policy-improvement operator and stages 2 and 3 should not
run. Stage 1 stands on its own as the Turtel replication. Total through
stage 2 is about $2k and five to six weeks of engineering, most of it the RL
loop and the retrieval pipeline we would need for any outcome-RL work.

## 10. Decisions

Status 2026-09-22. Decided: phased budget with stage 0 and stage D funded now
(about $600) and stages 1 and 2 decided on their results; base model
DeepSeek-R1-Distill-Qwen-14B for the Polymarket domain; the training-set
rebuild is funded, with questions from Halawi's raw Polymarket dump (14,246
questions, 2020 to mid-2024, on General) plus the Gamma API to January 2025,
and the evidence source (Exa, GDELT, Common Crawl News or Perplexity) still to
choose; default commit rule and constants; page published.

Open, pending discussion with Jessica: the stage-0 gate rule, evaluation
pre-registration, and whether to keep the live-reference arm.

Original list:

1. Base model for the Polymarket domain: DeepSeek-R1-Distill-Qwen-14B
   (faithful to Turtel, dense, well-trodden RL) or gpt-oss-120b (June 2024
   cutoff, Mantic's choice, MoE engineering). Qwen3 is excluded by the test
   window. Manifold domain stays on Qwen3-8B.
2. Whether to run domain B at all before the Polymarket result is in, or use
   stage 0 only as the gate and defer stage 3.
3. Commit rule in the market game: random on-policy rollout (v1) or
   value-guided best-of-G (v2), which is the real AlphaZero step and adds a
   value head to train.
4. Liquidity b, KL budget ε and turns per round K; proposed starting points
   b such that the whole market's subsidy is about 10× one week's settlement
   payout, ε = 0.05 nats per turn, K = 2 visits per active variable.

## Known risks

- Bootstrapping on the policy's own market is the deadly-triad setting;
  the frozen reference and the KL anchor are the stabilizers, and arm 2
  exists to measure how bad the live version is.
- Treewidth rejections make the action space state-dependent; the policy must
  see rejections as feedback, never as losses.
- Re-triangulation latency at 14k variables is unmeasured; the engine claims
  milliseconds per trade at width 8 for thousands of markets.
- A 4B pilot may not read evidence well enough to show claim 1; if stage 0
  fails at 4B, repeat it once at 8B before concluding.
