# Choosing a prior world for learning in-context similarity (critique round brief)

You are one of several independent critics (different model families) helping choose the synthetic "prior world"
for the next experiments in Kelvin's MATS research. Be critical and concrete. Do not flatter the candidates.

## The research goal (Kelvin's words)
- "I'd like to understand potential *scalable* in-context aggregation mechanisms, that could potentially scale to N
  observations, with no clear limit on N, and be more likely to 'converge to truth' as long as the observations
  'cover' as much as possible, without depending much, if possible, on the 'policy' that determines what sources are
  observed more frequently etc."
- "I suspect that a large transformer would be able to learn how to do this in a way with a PFN environment."
- The motivating idea (10-15): "the default UCT has no 'similarity measure'; you can test one 'move', then after
  expensive multiple rollouts decide this 'line' is not promising, but then you can sample another move that may turn
  out to be semantically similar in some relevant sense, but you can't easily learn that this other move is really
  just very similar to the other that wasn't good. In some ways transformer LLM + CoT solved this..." and "the
  learning of redundancy ... depends on the search-generation policy (more copies of 'similar' edges -> have to learn
  smaller weights)". He wants keys/fingerprints learned so redundancy is handled without depending on the policy.
- He cares about in-context aggregation and deduplication (using within-episode evidence), NOT prior-level
  discounting (a fixed shrinkage learned from the training distribution).

## Method constraints
- PFN-style training: small transformers (1e3 to 1e6 params) trained from scratch on fresh episodes sampled from the
  prior world; set-style tokens; one forward pass; output a predictive distribution.
- Exact (or numerically exact) Bayes posterior must be computable per episode, cheaply, so deviations from Bayes can
  be measured and later explained mechanistically (interpretability is a goal: we want to find HOW the network does it).
- The world must allow: varying N (train small, test much larger), varying observation policy (uniform, concentrated,
  adaptive), references that separate (i) pooling repeats of the same item, (ii) sharing evidence across similar items
  when similarity is known, (iii) inferring the similarity itself in context.
- Compute: a few hundred dollars at most; CPU/L4 scale.

## What we already know (from our runs)
- Set-style PFN on a linear-Gaussian source world: attention does all the dedup there is; with explicit group keys,
  depth 2+ dedups 80%; without keys, about 45% via matching residual values; 1-layer models are prior-level only.
- Tiny models (3 sources, unknown reliability): 1 layer = prior-level ("query-only"); 2 layers ~ Bayes; ~20% of seeds
  get stuck at the prior-level solution.

## Candidate worlds (attack them; propose better ones)
A. Junta world: 256 moves = 8 surface bits; hidden relevant subset R (|R| in {1,2,3}) per episode; value = g(bits in R),
   g a random N(0,1) table; observations y = value + noise at moves chosen by a varying policy; target = value of a
   query move (often unobserved), or the best move's value. Exact Bayes = 92-way mixture over R with conjugate per-cell
   posteriors (count + sum). "Similar" = agree on R.
B. GP with per-episode relevant dimensions/lengthscales (ARD) over continuous keys; exact GP posteriors mixed over a
   discretised hyperparameter grid.
C. Low-dimensional hidden projection f(x) = g(Ux) (multi-index model); approximate Bayes only.
D. Tree/MCTS-shaped world: moves lead to sub-moves; values propagate (expectation or max); similarity across branches.


## Related literature you must account for (Kelvin asked: "Isn't this similar to the contextual bandit literature?")
Yes: candidate A is a structured (sparse/junta) contextual- or multi-armed bandit / Bayesian-optimisation reward model,
with the action policy made exogenous. Known work (verify, extend): sparse linear bandits; GP-UCB / kernelised bandits;
Thompson sampling; KR-UCT (Yee, Lisy, Bowling 2016, MCTS sharing evidence across similar actions by kernel
regression); PFNs4BO (Mueller et al. 2023, PFN trained on GP priors for Bayesian optimisation); Decision-Pretrained
Transformer (Lee et al. 2023, transformers pretrained on bandit tasks behave like posterior sampling); Algorithm
Distillation (Laskin et al. 2022); in-context linear regression (Garg et al. 2022). Say what, given this literature,
would still be NEW in our study (we think: the aggregation mechanism itself, its invariance to an exogenous policy,
scaling to N far beyond training, and in-context dedup vs prior-level discounting) or whether it is already done.

## What to deliver (plain text, ≤ 900 words)
1. For each candidate: the strongest objection (does it actually test the goal? can a model pass by a shortcut?
   is Bayes really policy-independent here? does "similarity" mean what Kelvin means? does it scale in N?).
2. Up to two better or modified worlds you would propose instead, each with: generative process, exact-Bayes method,
   the references that separate (i)/(ii)/(iii), the expected mechanism a transformer would need, and the main risk.
3. Your ranking and the one experiment you would run first.
