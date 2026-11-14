# Environments for measuring & training Bayesian beliefs (research sweep 2026-08-19)

Workflow: 6 search angles → 70 candidates → 16 deep-verified (repo, license,
Fireworks-pluggability, reward signal, single-GPU feasibility). Full agent
verdicts in the session journal; this file is the synthesis.

## The meta-finding: the thing we want does not exist off the shelf

Every verified candidate falls in exactly one of three buckets:
(a) **explicit probabilities + proper-scoring reward, but static/single-turn**
(Paleka tuples, BLInD, ForecastBench, OpenForesight); (b) **multi-turn hidden
state + verifiable reward, but beliefs only implicit in actions** (TextArena,
Battleship, SPIRAL/poker, Avalon); (c) **training infra with no belief content
at all** (verifiers, Environments Hub, reward-kit). Nobody has combined them.
The recipe that falls out: take (c) as the vehicle, inject (a)'s elicitation +
proper-scoring rubric into (b)'s environments — and prefer environments where
the harness already knows the exact posterior, so "being Bayesian" gets a
dense per-turn verifiable reward. That combination is an open contribution
("the belief gym"), same flavor of gap as our KL-anchor lane.

## Tier 1 — build on these now

| What | Why | Status |
|---|---|---|
| **verifiers** (PrimeIntellect-ai/verifiers, MIT) | The vehicle: rewards are arbitrary async Python → log/Brier scoring rules are ~10 lines; eval plugs Fireworks today (`base_url` in EvalClientConfig); GRPO trainer for local LoRA if we self-host. 4.5k★, pushed daily. Build on **v1 API** (v0 tutorials are stale). | USE-NOW |
| **Collaborative Battleship** (gabegrand/battleship, MIT, ICLR'26) | Sequential evidence + active information gathering; the harness computes a ~exact Monte-Carlo posterior over boards → reward = log score of the model's *stated* P(hit)/P(ship at X) against harness posterior + EIG for question choice. OpenAI-SDK client, 2-line Fireworks switch. The explicit posterior is the scaffold's, not the LLM's — our addition is the elicitation layer. | ADAPT (top pick) |
| **BayesBench multi-turn** (Ankur-Samanta/BayesBench, MIT code, arXiv:2606.30850) | Bookbag-and-poker-chips for LLMs: 2 of 4 envs have closed-form true posteriors (Beta-Bernoulli coin flips; MovieLens mixture). Per-turn distance-to-posterior = exactly the "within-market updating" skill the futarchy null exposed. Reuse envs+metrics, swap their vLLM-logprob probes for verbalized probabilities. | ADAPT |
| **BLInD** (HLR/BLInD, ⚠ no license, AAAI'25) | Verbalized Bayes nets with pgmpy-exact ground-truth posteriors + a seedable generator → infinite, perfectly verifiable conditional-probability rewards. Static/single-turn: curriculum stage 1, not the destination. Ask authors for a license grant. | ADAPT |

## Tier 2 — the forecasting lane (overlaps what we already have)

- **OpenForesight/OpenForecaster** (MIT, VeRL pipeline, 52.7k questions):
  ⚠ strategic flag — they already showed accuracy-RL cuts Paleka arbitrage
  violations **43.5%**. Our positioning: they need resolved outcomes; the
  KL-anchor lane is label-free and works on unresolved questions. Reuse their
  dataset + offline-news retrieval + reward shapes; skip their trainer.
- **ForecastBench** (MIT/CC BY-SA): canonical leaderboard; resolved-pairs repo
  = offline reward data; submit our tuned model bi-weekly as the public OOD eval.
- **Metaculus AIB + forecasting-tools** (MIT): 5-minutes-to-live bot (litellm →
  Fireworks works); proper peer scores; slow reward → deployment eval, not RL.
- **Manifold API**: replay-mode training from historical dumps (bots + academic
  training explicitly permitted); pair P&L with Brier (play-money noise).
- **Paleka repo** (⚠ no license): already vendored as our harness; ~20-line
  patch to point at Fireworks.
- SKIP: TruthTensor (no code, crypto framing), Prophet Arena (platform closed;
  pm_rank scoring lib unlicensed), PolyBench (no license, dormant).

## Tier 3 — implicit-belief games (thesis evidence, weaker training fit)

- **SPIRAL** (MIT): self-play RL on Kuhn poker transfers to math reasoning —
  the strongest published evidence for the game-RL→belief-machinery thesis.
  But: no probability elicitation, full-param training on ~8×H100, can't take
  an API-served policy. WATCH; cite, don't build on.
- **TextArena** (MIT, very active): 100+ text games incl. belief-games; a
  Fireworks agent is literally `OpenAIAgent(base_url=fireworks)`. Terminal
  win/loss only → wrap our elicitation + scoring on top (same recipe).
- **UnstableBaselines** (MIT, young): LoRA-first self-play trainer for
  TextArena — only relevant if we train on local weights (e.g. the DGX Spark);
  requires local vLLM-served adapters, not Fireworks.
- Sweep-level (not deep-verified): AvalonBench/Werewolf-Plus/MindGames (social
  deduction, heavier), Hanabi (cooperative beliefs, non-LLM tooling), PokerBench
  (solver-labeled SFT), Decrypto (theory-of-mind), reasoning-gym (procedural
  probability tasks, GRPO-ready), Rewarding Doubt arXiv:2503.02623 (PPO on a
  log-score betting game — closest published training recipe to "wager your
  credence"), ConfTuner (differentiable tokenized-Brier loss), KellyBench
  (bankroll sizing), ABIDES/TwinMarket/AucArena/Concordia (econ sims: beliefs
  load-bearing but tangled with strategy).

## Proposed curriculum (the "belief gym" shape)

1. **BLInD** — static exact posteriors (is the model able to compute?)
2. **BayesBench envs** — sequential evidence, closed-form posteriors (does it update?)
3. **Battleship** — active evidence-gathering under hidden state (does it choose informative evidence and update on it?)
4. **futarchy-fi/simulations** — multi-agent strategic (does it update on *others'* behavior? — the failed skill from the market experiments)
5. **ForecastBench/Metaculus/Manifold** — open-world OOD eval (does any of it generalize to reality?)

All five stages share one elicitation format and one proper-scoring rubric,
implemented once as verifiers tasks → same GRPO plumbing as the RLEF pilot.
