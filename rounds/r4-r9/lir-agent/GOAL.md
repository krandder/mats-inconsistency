# GOAL: LIR control masks for transformer post-training (overnight programme)

Kelvin + eurisko, 2026-10-03 ~06:45Z. Bead: filed under pool-dursw (eurisko research lane).
Owner: eurisko (scientific lead). Builders: Opus 5.5 subagents and workflows; gpt-6-astra where
useful. Compute: this box (16 CPU), spot L4s through `logical-soup/infra` (auto-delete, watchdog,
cost audit), Tinker (LoRA-only; role adapters fit it) if calibrated-ai's account allows; the MATS
cluster when opus-mats secures access. GPU cap for the night: $60. No real-world actions, no
production systems, no new secrets. Disk: everything under this directory, under 5 GB on /srv;
bulky intermediates on /tmp or the VMs.

## Goal line

    /goal Execute /home/agent/workspaces/eurisko-study/lir-agent/GOAL.md: overnight programme on LIR-style control masks for transformer post-training (belief / forecast / action adapters; policy-initiated forecast blocks trained only on realised outcomes; goal-blind content attention) on a text POMDP; adversarial hawks before every experiment; each experiment a small modification of a named paper; GPU cap $60; by morning a report with calibration-under-RL curves, optimism gap vs search, gradient-localisation metric, and the hawks' verdicts.

## The claim under test (one paragraph)

GRPO and RLHF are one Local Inconsistency Resolution step with attention on the reward edge and a
control mask of all ones, so the inconsistency between "I want reward" and "this is what I believe
and predict" is resolved partly by moving beliefs and forecasts (sycophancy, confident false
success, drift of the internalised preference), not only by moving actions (Richardson 2024 §6.8,
§7.6; LIR 2026 §4.7). If reward may control only an action adapter, while beliefs (situation
report, self-model) and forecasts are controlled only by what later turned out to be true, and
belief content cannot attend to goal/utility tokens, then calibration and introspection should
survive RL while reward rises, and the fused baseline should show the known failures.

## Settled design (Kelvin, 2026-10-03)

- One potential (PDG inconsistency), different control masks per phase; no separate losses unless
  the shared potential leaks (then separate losses as fallback, recorded as a result).
- Variables: O evidence; S situation (hidden, varies per episode) incl. a self-model; A plans and
  the committed action; O' realised consequences; U utility (fixed reward model / environment
  reward); T the Faith edge b(T|U) ∝ exp(u), never under the agent's control.
- Policy-initiated deliberation: the action adapter writes candidate plans and opens
  `<forecast>` blocks; the forecaster adapter fills them; the policy reads them, proposes more or
  commits. A forecast of the committed plan is mandatory before any real action.
- Two attentions: relevance attention (goal-conditioned: what to attend to) and content attention
  (goal-blind: belief and forecast tokens cannot attend to goal, utility or reward tokens).
- Honest, harder variant: the belief and forecast adapters keep learning during RL, on realised
  truth only. Frozen-belief is a control arm, not the main arm.
- Domain: text POMDP first (Mastermind / Battleship style: hidden state inferred from evidence,
  actions with consequences, truth known; counterfactual outcomes available by branching the
  simulator). Models under 8B.
- Hawks: one adversarial reviewer per idea and per experiment, run BEFORE the experiment; a hawk
  may kill or reshape an experiment (eurisko decides on its verdict). Every experiment is a small
  modification of a named paper, never a new system from scratch.

## How the forecaster is trained (the part Kelvin flagged as most important)

Format: a `<forecast>` block is a fixed set of scorable claims about O' for the plan under
evaluation: P(success), distribution over steps/cost, P(each environment-defined outcome fact),
plus optional free text. Fixed format makes it a cpd τ(O'|S,A) with the forecaster's confidence.
Signals, all prequential, none from reward:
1. Committed plan: log score / Brier of its forecast against the realised outcome (every episode).
2. Non-committed candidates: (a) in simulation, branch the environment from the same state and
   execute them: counterfactual truth (arm with, arm without, to learn whether (b)-(d) suffice
   where branching is impossible); (b) coherence across the deliberation's forecasts (exclusive
   outcomes sum ≤ 1, prefix plans dominate, situation-report consistency), the calibrated-ai
   coherence losses applied to forecasts; (c) cross-episode generalisation: the forecaster is
   trained on all (situation, plan, outcome) triples, so plans not taken here are taken elsewhere;
   (d) on-policy weighting: forecasts of plans the policy actually selects are the ones corrected
   first, so phrasing tricks that fool the forecaster are corrected by their own outcomes.
3. Multi-step: forecasts of intermediate states scored against realised intermediates; a
   martingale check along the trajectory (forecast revisions about the same outcome must have zero
   mean; systematic early optimism is squirming), Kelvin's "predicted movement" measure.
4. Optimism-gap model: trained on (forecast at commit, number of candidates, spread,
   forecaster uncertainty) → realised minus forecast; corrects the value the policy acts on
   (Smith-Winkler shrinkage learned from data); gives the stopping bound: expected gain from more
   candidates ≤ expected exploitation of forecaster error.
Control masks: forecaster steps update only the forecaster adapter; the policy's GRPO objective
excludes forecast tokens and its gradient is routed to the action adapter only; sense-making
tokens are scored against revealed hidden state and update only the belief adapter; the
self-model is trained by predicting one's own next action (goal-blind).
Inputs to the forecaster: evidence, situation report, the plan, the self-model; never the goal,
utility or reward tokens (content mask). Plans are complete action sequences in the first
experiments (no self-model needed for forecasting); partial plans + self-model come later.

## Experiments (each: paper → modification → one claim → hawk → run → verdict)

E1 Phasic Policy Gradient (Cobbe 2021) + prequential belief phase: calibration under RL.
E2 Gradient routing (Cloud et al. 2024) for a reward-only action adapter: gradient-localisation
   fraction and its correlation with sycophancy probes.
E3 Reasoning-via-Planning / Tree of Thoughts with a separately trained forecaster adapter vs the
   same LLM as world model: optimism gap vs number of forecasts (Gao et al. 2022 curve).
E4 Stream of Search (Gandhi 2024) with prequential forecast tokens inside the trace: CoT
   faithfulness (Turpin 2023 probes) before/after.
E5 Goal-blind content attention vs goal-conditioned relevance attention: does goal-blindness
   cost perception on the POMDP?
E6 Grounded vs latent anchoring of the utility under continued belief learning: ranking drift.
E7 Self-prediction (Binder 2024) under RL, fused vs protected: does introspection break?
Theory T1: metareasoning stopping bound (Russell-Wefald; optimizer's curse) with a refuter.
Theory T2: self-model consistency via LI finite-precision self-knowledge, with a refuter.
Baseline B0: fused GRPO on everything, same model, same environment, same compute.

## Metrics (every run, mean ± sd over ≥3 seeds)

Reward; calibration (ECE, Brier) and coherence of situation reports and forecasts before/after
RL; sycophancy probes; optimism gap vs candidates considered; gradient-localisation fraction;
forecast-revision martingale drift; Faith-swap test (change the reward edge, measure action change
without belief retraining); CoT faithfulness; introspection accuracy.

## Stop rules and cadence

Hawk kills are final for the night (record why). A run over 2 h on one L4 stops and is reported.
GPU over $50 stops new launches. Check-ins every 2 h via `fleet-now change lir-agent`. Morning
deliverable: REPORT.md + HTML at /home/agent/memo-serve/lir-agent/ (and the public site
if Kelvin wants), hub PR draft, and a 3-line handoff.

## Addendum (Kelvin + eurisko, 07:05Z): free-form forecasts scored by a judge

The fixed per-environment forecast format is a control, not the design. Round-two format: a
forecast is a free-form list of atomic claims, each with a probability, chosen by the policy.
Scoring: a judge model reads the realised outcome and marks each claim true / false / unresolved
ONLY (it never grades forecast quality); the score is computed by formula (log score / Brier) over
resolved claims; a forecast with too few resolvable claims counts as abstention (so unfalsifiable
claims earn nothing). The judge is the "deductive process" of logical induction: frozen or trained
only on human adjudications, goal-blind (never sees what the agent wanted), a different model
family from the forecaster, never updated on agreement with the forecaster. Validate the judge in
the simulator against programmatic truth (resolution error, phrasing bias) before relying on it.
If the harness builder can add this path cheaply, add it as arm FF alongside the fixed format;
otherwise it is the first item of the next round. Report agents: state which format each result
used.
