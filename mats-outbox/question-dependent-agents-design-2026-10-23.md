# Question-dependent agents: design v2 (opus-mats, 2026-10-23)

Why v2: the gate on the rich-observation toy (outbox/gate-rich-agents-2026-10-23.md) found the hard part was missing:
- extraction was one fixed linear projection;
- wide messages lost nothing;
- the "query" read only earlier messages, never the aggregator's state;
- the bar was itself a per-agent encoder.

## The ingredient that makes the hard part real: the QUESTION varies per episode
- **World.** K = 16 latent bits b_1..b_16 (uniform). Each episode draws a QUESTION q: a random parity or AND over a small random subset S_q of the bits (|S_q| in {2, 3}), plus its distractors. Y = f_q(b_{S_q}).
- **Agents.** N = 8; each observes x_i ∈ R^128 = Σ_{j∈R_i} a·U_j(2b_j−1) + V g + ε. Here R_i is the agent's reading set: 4 of the 16 bits, Zipf-popular, so overlap follows a power law. Each agent therefore holds evidence on several bits, and only some of them matter for this episode's question.
- **Bottleneck.** The message is b bits per agent (FSQ; b in {2, 4, 8}), far less than the agent's evidence (4 bits of signal + nuisance). It must send what matters FOR THIS QUESTION, given what others already said.
- **Who knows what the question is.** The consensus (aggregator state) knows q. Agents know it only through a query from the consensus:
  - **one-shot arm:** agents never see q;
  - **query arm:** each agent reads the consensus state, including q and the earlier agents' messages, before writing. That is Kelvin's "consensus queries the agent before the compression".
- **Keys.** Content-derived: key_i = W·onehot(R_i) + σ·ε (σ = 0.5).

## Yardsticks
- (1) exact Bayes given all x (enumerate the bits in S_q; integrate the others and g analytically; tractable because |S_q| ≤ 3);
- (2) the transformer bar: ONE transformer over all agents' raw x tokens plus q, mixing across agents from layer 1;
- (3) a closed-form query-aware b-bit protocol: each agent sends its quantised local log-likelihood ratios for the bits in S_q, which it can do only if it knows q. This is an achievable reference for the query arm;
- (4) the same protocol without q (the agent must guess which bits matter): the one-shot reference.
The gap (4) − (3) is the information value of the query; the learned arms are measured against both.

## What it tests (Kelvin's points)
| Point | Covered? |
|---|---|
| 1. Large private evidence; agents must find what matters for this question | yes |
| 2. Aggregation-aware messages after a query | yes |
| 3. Aggregator learns who knows what, against Bayes and a fair bar | yes |
| 4. Market and local learning | not yet |

Expected, if the thesis holds: query arm >> one-shot at small b; one-shot is near (4); query approaches (3).

## Rules before compute
- A design-fit reviewer checks this against Kelvin's words, and the closed-form references are validated.
- Unconditional means are reported; stalls are counted, not filtered.
