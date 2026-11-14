# Parked idea: a wealth-based market over logical propositions (2026-10-22)

Status: PARKED by Kelvin ("Let us store this as something to save/discuss in the future"). Not built.

## The idea, in Kelvin's terms
- **Variables.** A fixed population of base bits y. Only the y's are ever settled.
- **Propositions.** A second, evolving population z with RULE-BASED (logical) relations to the y's, in both directions:
  - z = f(y-subset), e.g. "cells A and B coupled" in Sudoku;
  - y = f(z-subset), e.g. y = z1 XOR z2 with z latent.
- **Fuzzy connections** are mediated by predictors or rollouts (read predictions of z2, trade z1), never by learned weights.
- **Settlement.** Only y's are revealed. The market's joint distribution is conditioned on them, so the z's settle at the posterior: mass contradicted by evidence is removed.
- **Wealth.** Each proposition has wealth (liquidity). It grows when trading on the proposition improves accuracy on the settled y's and decays when unused. Broke propositions are replaced.
- **Open problems Kelvin named:**
  - (a) a proximity space so that wealthy propositions can mutate or divide;
  - (b) key/query representations so predictors generalise as the securities change;
  - complementary payment (z1 pays only if z2 is traded well) can be a problem: the market version of the parity saddle.

## What we found (literature, all references verified)
- outbox/proposition-space-literature-2026-10-22.md:
  - truth-table/semantic distance for (a);
  - Fourier-coefficient keys with moment queries for (b);
  - XCS don't-care mutation;
  - Hayek-style conserved-money division (z -> z∧yk, z∧¬yk);
  - risks: pay for beating the price, 2^n growth, exact keys make the toy too easy.
- outbox/learned-securities-literature-2026-10-21.md and compressed-aggregation-literature-2026-10-21.md (background).
- Fleet prior work: wiki trader-economy-soup-design, trader-soup-neural-mixture-design, trader-economy-depth-literature:
  - coherence arbitrage never evolved in the soup;
  - gradient wealth fails, exact log-score settlement works;
  - one-step market credit is weak.
- Kelvin's junction-tree LMSR: /home/agent/workspaces/bayes-engine/inference/factored_market.py (the market substrate).

## A 7-step toy design exists (proposed, not built)
In the literature note: 6 bits, random 2-3 literal propositions, exact pricing, wealth dynamics, held-out-proposition generalisation test.
