# Our research vs "Roadmap for a collaborative prototype of an Open Agency Architecture" (opus-mats, 2026-10-20)

Source: LessWrong, 10 May 2023, by Deger Turan with Tantum Collins, davidad and Charbel-Raphaël Ségerie (AI Objectives Institute). It was fetched 2026-10-20, and the summary was checked against the page text. Prior analysis of the same post from an earlier Kelvin session is on the hub wiki page consistency-oaa-plan-tree.md. That page is self-reported and unverified.

## Shared
- **Separation:** the world model, the preferences and the beliefs are kept apart, and planning happens against the world model with explicit utilities. Preferences are elicited, not trained in.
- **Inspectability:** every component is meant to be something people can read.
- **Several stakeholders:** OAA combines them with Nash bargaining; we have multi-agent equilibria with copies holding other utilities.

## Different

| | OAA roadmap | Our agenda |
|---|---|---|
| World model | Formal artefacts written by people or LLMs: a probabilistic program (Squiggle) translated to a model checker (PRISM/JANI), or expert-elicited probabilities | Learned, behind a transformer-like any-order interface; may be hard-coded for tests |
| Planner | Verified model-based RL (reach-avoid supermartingales), receding horizon | A learned searcher that minimises inconsistency, measured against MMD |
| Guarantee | Formal verification relative to the world model | None formal; auditable beliefs and steerable goals |
| Preferences | Elicited features and linear weights; Nash bargaining across people | A utility edge in a PDG |
| Scope | Small institutional decisions (budgets, grids) | General-purpose search, from puzzles to games |
| Bottleneck they name | Building the world model ("number synthesis"), discretisation, Lipschitz bounds | Whether a learned searcher finds correct solutions and not merely consistent ones (R27A) |

## Where the two meet
- **Elicited world models and PDGs.** The roadmap's projects 3–4 elicit experts' probability judgements. Those are PDG edges, and PDG inconsistency measures how much the experts conflict while also giving a combined belief. This is a precise fit, not an analogy.
- **Bargaining and our games.** Nash bargaining is a cooperative solution. Our self-referential construction gives non-cooperative equilibria. They answer different questions.
- **What our results say to OAA.** R27 shows that learned search is still unreliable even on small puzzles: in Sudoku it settles on consistent but wrong fillings. That supports OAA's choice to verify a planner rather than trust a learned one. OAA's own weak point is building the world model at scale, which is where our agenda bets on learning.
