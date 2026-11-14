# Kelvin's research orientation (his answers, 2026-10-19 ~19:00Z; recorded by opus-mats)

- Core bet (most disappointed if refuted): inconsistency minimisation is a better training target.
- Audience: himself first.
- Uncertainty in scope now: logical (puzzles) and hidden state; not other agents, not real-world forecasting (yet).
- Training signal to test: no labels at all (no answers, no reward); inconsistency with the rules/world model only.
- Beliefs: over answers, hidden state and the rules themselves, so the agent can act to find out which rules hold.
- Success (his words): "a kind of general model that can do in-context learning, solve puzzles, play games, including with uncertain rules ... with exact rules encoded on its world-model, it can achieve 99%+ accuracy on puzzles with enough 'reasoning/rollout' compute, and we can see it evolve its confidence in an answer over time. It should be capable of acting strategically to get information, including e.g. about what the rules of this world are ... possible to change the utility of the agent and it then optimizes for something else." Need not beat specialised architectures on parameter count.
- Decomposition is the point: learning world models vs searching/planning under whatever world model is available (certain or uncertain); world-model learning possibly later.
- Scale: start small, expand; long-run "AGI-like model without language", trainable for < $100k.
- Undecided: the form in which the world model is given to the searcher.

## Clarifications (Kelvin, 2026-10-19 ~19:20Z)
- "AGI" means general in the sense that frontier LLMs are general for language, classifiers for classification, LLM-based reasoners for many RL environments.
- World model to searcher: (1) simulator, "definitely not": the searcher must eventually search over a learned world model; hard-coded world models are fine for testing and training only if they could in principle be learned. (2) PDG factors are enticing, but he wants to leverage transformers and attention: the world model "defined" in a transformer/attention-like procedure, or at least an interface that would later allow an any-order transformer with attention; "maybe the interface is more important than the actual format of the world-model". (3) In-context rules: later upgrade to a transformer world model that reads a rule description or few-shot examples that the rest of the world-model sampling attends to.

## Design discussion (2026-10-19 ~20:20Z): opponents in the world model; equilibria as inconsistency minima
- Kelvin: uncertainty over the world includes uncertainty over opponents; the same algorithm should recognise self-play worlds (copies of itself with other objectives) and reuse its own circuits to model opponents who search like it; include self-play or strong opponents in the training environment; ideally learn which search algorithm works best.
- Candidate construction (opus-mats): per decision node of player i, a self-referential edge a ~ softmax(Q_i^mu(I,.)/tau), Q computed by the searcher under mu with utility U_i from player i's information; inconsistency = sum of reach-weighted KL(mu(a|I) || that edge). Normal form: tau * sum KL equals the entropy-regularised NashConv exactly (checked numerically, /tmp/om/qre_check.py: 0.91134 = 0.91134); zeros = logit QRE; tau -> 0 gives minimax in 2p zero-sum. Extensive form (dilated entropy, reach-weighted) not yet verified.
- Minimising it: naive simultaneous softmax best responses cycle (inconsistency 1.1-2.3 forever in the check); adding a proximity edge to the previous iterate plus a magnet edge = MMD, which reached 0 (6e-17) after 3,000 steps. Ataraxos' search is one tabular MMD step.
- Kelvin (~20:25Z): "inconsistency minimisation with proximity and prior edges" as the known-good search procedure a learned searcher is measured against: "Yes, this is what we want."
- Extensive-form check on Kuhn (/tmp/om/kuhn_qre_incons.py; OpenSpiel 2.0.2 nash_conv): reach-weighted per-infoset KL to softmax(counterfactual soft Q / tau), minimised by behavioural MMD (uniform magnet). Inconsistency reaches ~1e-16 at every tau; NashConv 0.528 (tau 0.5), 0.319 (0.2), 0.093 (0.05), 0.0098 (0.01); uniform 0.917. At tau 0.01 P1 bets J 0.189, Q 0.002, K 0.600 (K/J 3.18; Kuhn family: J in [0,1/3], K = 3J).

## Decision theory (Kelvin, 2026-10-19 ~20:50Z)
- Self-modification left out initially. Embedded-agency self-reference: "we need to understand this more deeply first".
- No claim that prior/KL edges keep optimisation mild (v6 note does not claim it; checked by grep). Preferences over own policy may be fine; to discuss.
- Regularisation as part of the target: "more like a hack; this can be part of the learned policy".
- Direction he likes: pick policies, not actions, or policies alongside actions with actions tied to policies by consistency ("policies are just like more constraints on the actions"); aim for a PDG-style decision theory where copies or predictors may read only part of the agent's mechanism, or all of it.
