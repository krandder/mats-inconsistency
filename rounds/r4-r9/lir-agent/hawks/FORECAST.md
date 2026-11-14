# Hawk: FORECAST (forecaster training scheme in GOAL.md)

Hawk: adversarial reviewer, 2026-10-03. Verdict: **RESHAPE**. Item reviewed: GOAL.md, section "How the forecaster is trained".
Objection given to me: "without branching, non-committed candidates are unsupervised and the policy exploits them; coherence is a weak substitute."
Evidence: a CPU toy, about 2 min on one process.
- `hawks/FORECAST-sim.py` -> `.out`/`.json` (arms: committed / IPS / branch)
- `hawks/FORECAST-sim2.py` -> `.out`/`.json` (IPS step-size sweep)
- `hawks/FORECAST-sim3.py` -> `.out`/`.json` (label-matched branch)

All paths are relative to `/home/agent/workspaces/eurisko-study/lir-agent/`.

## 1. The objection points at the wrong failure

**Optimistic exploitation on committed plans corrects itself.** If the policy inflates a forecast (through phrasing or a lucky error) and commits to the plan, the realised outcome scores that forecast. This is the on-policy mechanism (d) in GOAL.md, and it works.
- In the toy, the forecaster starts fooled: it gives a "puffed phrasing" feature a weight of +1.0 logit.
- That weight decays to about 0 under committed-only supervision: -0.004 ± 0.002 (tabular), -0.003 ± 0.001 (linear), 5 seeds.
- Puffed commits fall to 1.1% (tabular). Source: `hawks/FORECAST-sim.out`.

**Pessimistic errors on never-chosen plans are what is never corrected.** When a good candidate is forecast too low, it is never committed, so it is never scored. This is a known impossibility result, not just an empirical risk:
- Othman & Sandholm 2010: no scoring rule is quasi-strictly proper when paired with the deterministic "max" decision rule.
- Chen, Kash, Ruberry & Shnayder 2011, Theorem 2: a strictly proper decision scoring rule exists **iff** the decision rule has full support.
- Their Theorem 1 gives the construction: score the chosen action a with S = s(P_a)/d_a, where d_a is the probability of choosing a. That is inverse-propensity weighting.

I checked both theorems in the PDF text (`/tmp/lir-agent/hawk-forecast/chen.txt`, lines 273-310).

**Toy numbers.** "Pessimism lock-in" means the share of contexts where the truly best plan is forecast more than 0.1 below its true value and is not chosen. Source: `hawks/FORECAST-sim.out`, `hawks/FORECAST-sim3.out`.

| Arm | Pessimism lock-in (tabular) | Pessimism lock-in (linear) | Brier vs exact truth, all candidates (tabular) | Regret per episode (tabular) |
|---|---|---|---|---|
| committed only | 0.352 ± 0.057 | 0.144 ± 0.032 | 0.061 | 0.116 ± 0.017 |
| full branching | 0.036 ± 0.023 | 0.024 ± 0.020 | 0.0033 | 0.021 ± 0.002 |

So the experiment as written will mostly measure **under-exploration**, not "exploitation". The mechanism and its fix are already known.

## 2. Hidden assumption that makes the planned comparison uninformative

**The comparison will show "no branching ≈ branching", for the wrong reason.** The domain is Mastermind/Battleship, plans are complete action sequences, and the belief adapter is "scored against revealed hidden state". With deterministic dynamics, the outcome of every candidate is O' = f(S, plan). Once S is revealed, every counterfactual is free, with or without branching. It can also leak in through the situation-report input, which an S-supervised belief adapter writes.

As specified, the "arm without branching" therefore still gets counterfactual supervision through the back door. A null difference between the arms would say nothing about the real-world case, where S is never revealed. That case is the only reason the arm exists.

**Second confound: label count.** Full branching gives K labels per episode against 1. A label-matched arm that branches only **one** uniformly random non-committed candidate recovers most of the effect (`hawks/FORECAST-sim3.out`):

| Arm | Brier vs truth, all (tabular) | Pessimism lock-in (tabular) | Regret (tabular) | Regret (linear) |
|---|---|---|---|---|
| committed only | 0.061 | 0.352 | 0.116 | 0.031 |
| one random branch | 0.0048 ± 0.0004 | 0.096 ± 0.046 | 0.043 | 0.0135 |
| full branching | 0.0033 | 0.036 | 0.021 | 0.0072 |

What matters is any off-argmax label drawn with known propensity, not counterfactual truth for every candidate.

## 3. Coherence is weaker than "weak"

**Coherence carries no information about the evidence.** An evidence-ignoring forecaster that always outputs the prior predictive (one distribution over S, pushed through each plan) is perfectly coherent: exclusive outcomes sum to at most 1, prefix dominance holds, and it agrees with itself across candidates. In Mastermind, joint coherence across candidate guesses means only that some common distribution over secrets exists, and the prior is one.

**So coherence only moves non-committed forecasts through structure they share with committed ones.** It cannot fix a candidate whose error is independent of the committed plans.

**Paleka et al. 2024 does not license coherence as a training signal.** They show that consistency metrics correlate with forecasters' ground-truth Brier scores. That is a correlation between models used for evaluation. It does not show that optimising coherence improves accuracy, and a forecaster can satisfy coherence by collapsing to the prior.

## 4. Metric confounds

- **Calibration on committed plans only** is confounded by selection and by the plan mix. As RL concentrates the policy on late-game, easy-to-forecast plans (for example, one consistent secret left, so P(success) = 1), Brier/ECE improves with no change in the forecaster. Oesterheld & Conitzer 2020 make the incentive version of this point: scoring anything beyond the mean pushes recommendations toward easy-to-predict actions.
- **The martingale / "early optimism" check is biased by selection** if revisions *before* commit are measured on committed plans. In the toy, a **perfect Bayesian** forecaster shows a mean pre-commit revision of +0.394 on committed plans, which looks like squirming or under-reaction. Its post-commit gap is 0.0004 ± 0.0009 (`hawks/FORECAST-sim.out`, "martingale artefact"). Revisions are only zero-mean when conditioned on information at the start of the window. Commit depends on the later forecasts, so the conditioning must use stopping-time-measurable events.
- **The learned optimism-gap model has an off-the-shelf baseline it must beat:** the double estimator (van Hasselt 2010), where one forecaster head or seed selects and an independent one evaluates. Smith & Winkler 2006 is the shrinkage reference already in GOAL.md.

## 5. Reshape: the exact change

1. **Supervision arms for non-committed candidates.** Replace "(a) with / without, test whether (b)-(d) suffice" with five arms. (c) and (d) are properties of every arm, not interventions.
   - **A0:** committed only.
   - **A1:** A0 plus the coherence loss.
   - **A2:** A0 plus a full-support commit rule: with ε = 0.1, commit a uniform random candidate; score with s/d_a, IPS-weighted, optionally doubly robust (Dudik, Langford, Li 2011). This is the real-world substitute for branching. Report the reward it costs.
   - **A3:** A0 plus one uniformly random branched candidate per episode (label-matched).
   - **A4:** full branching.
2. **Close the back door.** Add exogenous outcome noise that S does not determine (for example, noisy Mastermind feedback or a Battleship sensor that misreports with p = 0.1, with the noise seed never revealed). In A0-A2, the forecaster never trains on targets derived from S. Its situation-report input comes from a belief adapter that is frozen or trained only on committed O'. Make "belief adapter S-supervised: yes/no" an explicit factor.
3. **Oracle evaluation.** Score every candidate, plus a fixed probe set of (evidence, plan) pairs sampled independently of the policy, against the exact posterior predictive. Mastermind 4x6 is enumerable: at most 1296 codes. Report:
   - signed error, split chosen / non-chosen;
   - the pessimism lock-in fraction;
   - regret against the Bayes-optimal choice among the candidates;
   - the plan-mix-adjusted Brier: committed-plan Brier reweighted to the fixed probe distribution.

   Never report committed-only calibration as the headline.
4. **Exploitation channel as factor B.** Compare canonical plans (parsed and re-rendered; no free text reaches the forecaster) with a free-text rationale that does reach it. With complete canonical plans (E1-E3), the only channel is selection, so "the policy exploits the forecaster" is untestable without factor B.
5. **Martingale check.** Use only revisions after the commit decision, or revisions of a probe plan fixed before deliberation starts.
6. **Optimism-gap model.** It must beat the double-estimator baseline on oracle targets, as a function of the number of candidates.
7. **Pre-registered decision.**
   - "Coherence suffices" means A1 is within 25% of A3 on non-committed oracle Brier **and** on lock-in fraction, with ≥ 5 seeds. IDEAS-NET X1: 3 seeds understate the sd by a median 1.7x.
   - Otherwise the real-world recipe is A2, and its exploration cost is a reported result.
   - A prior-collapse forecaster is shown next to A1: its coherence violations are 0 and its Brier is at prior level.
8. **Cost.** Run A0-A4 x B on the CPU toy/POMDP simulator first. Take only A0, A2 and A3 with canonical plans to the L4 LLM runs.

## Addendum (second pass, 07:06Z): the back door exists in the actual harness

Checked `harness/env/shiphunt.py` (ShipHunt, default `EnvConfig`) against section 2. The leak is real and larger than assumed:
- `Outcome.hit_facts` is the "true occupancy of every planned cell (fired or not)". The committed plan's outcome therefore already labels shots that were never fired. That is free partial branching inside the "no branching" arm.
- `shot_noise` defaults to 0.0, and `truth()` returns the full `occ`. With the belief adapter scored against `truth()`, every candidate's outcome is a deterministic function of revealed S.
- The default world has 17 placements on 12 cells (`hawks/FORECAST-sim4.out`). Outcome facts are per-cell occupancy, shared across plans. A forecaster that learns P(cell occupied | evidence) answers every candidate at once. Here coherence (same cell, same probability across plans) plus cross-episode generalisation is close to full supervision by construction.
- Measured on 4000 episodes with random 3-cell plans (`hawks/FORECAST-sim4.json`):
  - mean share of a candidate's cells that are also in the committed plan: 0.247;
  - share of candidates whose success is fully determined by the committed plan's `hit_facts`: 0.287 (0.256 from fired shots alone);
  - share determined once `truth()` is used: 1.0.

Consequence: on default ShipHunt, "A1 (coherence) ≈ A4 (branching)" is the expected result whether or not coherence works in general. Extra items for the reshape:
- **2a.** In arms A0-A2, the forecaster trains on fired shots only: drop unfired `hit_facts` from its targets.
- **2b.** Set `shot_noise ≥ 0.1`.
- **2c.** Make the belief adapter's S-supervision a declared factor.
- **2d.** Use a world large enough that per-cell memorisation is not the whole task (for example 6x5 with ships (3,2)), or a plan-level outcome that does not factor over cells (`need_hits ≥ 2` with an order-dependent `steps`).

If the builders keep the default config, the FORECAST comparison should be reported as "not informative about settings where S is never revealed".

## Caveats on my own toy

It is a bandit with 50 contexts x 8 plans and 20k episodes, not an LLM.
- **IPS (A2) is label-starved here:** about 5 random labels per context-plan cell. It improves Brier vs truth: 0.037 ± 0.004 against 0.061 for committed-only (tabular, lr 0.2). Its regret (0.164) is worse, because regret includes the cost of exploration and the high-variance weights cause a winner's curse on chosen plans (+0.097).
- **The step-size sweep did not rescue IPS** within 20k episodes (`hawks/FORECAST-sim2.out`).

In the LLM runs, A2's sample cost is itself a result to report, not a reason to drop the arm.

## Sources (fetched 2026-10-03 unless marked)

- Othman & Sandholm, "Decision rules and decision markets", AAMAS 2010, pp. 625-632. I did not fetch the paper; its result (Theorems 1 and 4) is verified secondhand, as cited in Oesterheld & Conitzer 2020 (lines 105-115 and 290-300 of `/tmp/lir-agent/hawk-forecast/oest.txt`) and in Chen et al. 2011. No direct URL verified.
- Chen, Kash, Ruberry & Shnayder, "Decision Markets With Good Incentives", WINE 2011. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wine11-decision-markets.pdf (Theorems 1-2 read). Journal version: Chen, Kash, Ruberry & Shnayder, "Eliciting Predictions and Recommendations for Decision Making", ACM TEAC 2014, https://dl.acm.org/doi/abs/10.1145/2556271
- Oesterheld & Conitzer, "Decision Scoring Rules", WINE 2020. https://www.cs.cmu.edu/~conitzer/decisionWINE20.pdf (abstract and Section 3 read).
- Dudik, Langford & Li, "Doubly Robust Policy Evaluation and Learning", ICML 2011. https://arxiv.org/abs/1103.4601
- Smith & Winkler, "The Optimizer's Curse", Management Science 52(3), 2006. https://doi.org/10.1287/mnsc.1050.0451 (the publisher page returned 403; metadata verified via search, JSTOR https://www.jstor.org/stable/20110511).
- van Hasselt, "Double Q-learning", NIPS 2010. https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
- Paleka et al., "Consistency Checks for Language Model Forecasters", 2024. https://arxiv.org/abs/2412.18544
- Perdomo, Zrnic, Mendler-Dünner & Hardt, "Performative Prediction", ICML 2020. https://arxiv.org/abs/2002.06673 (forecasts that steer which plan is executed are performative; retraining converges to a stable point, not an optimal one).
- Augenblick & Rabin, "Belief Movement, Uncertainty Reduction, and Rational Updating", QJE 136(2) 2021. https://www.semanticscholar.org/paper/96789387c91de3ae31a23b28b242482e3697c850 (the "predicted movement" test; verified via search metadata).
- Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization", 2022. https://arxiv.org/abs/2210.10760
- Stroebl, Kapoor & Narayanan, "The Limits of Inference Scaling Through Resampling" (imperfect verifiers, false positives cap best-of-n), https://arxiv.org/abs/2411.17501
- Schrittwieser et al., "MuZero", https://arxiv.org/abs/1911.08265. The model is trained only along taken trajectories yet planning works; that is a precedent for (c)/(d) in deterministic games. Its off-trajectory failure modes are a recollection (unverified).
- Ye et al., "EfficientZero", https://arxiv.org/abs/2111.00210. That it adds a self-supervised consistency loss is a recollection (unverified; the abstract page does not say).
