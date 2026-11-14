# Learned securities: a first experiment (opus-mats, 2026-10-21)

Kelvin (verbatim): "the problem is the non-separable information... The good example is the XOR example, when we actually only care about a single bit which is the parity. So we need to find a way to "learn" the securities we need. This is what I'd most love to see."
Reuse: Kelvin's factored junction-tree LMSR, /home/agent/workspaces/bayes-engine/inference/factored_market.py. Trades are minimum-KL multiplicative reweightings of the joint; resolution is conditioning; tree width is a hard budget.

## Market
- **State.** The state S is an LMSR joint over the target y and m latent discrete variables z_1..z_m, stored as a junction tree with width at most w.
- **Securities.** The structure (which z share a clique with which, and with y) and the link potentials are what the market designer learns (theta). Together they define the securities: the tables traders can trade on.
- **Traders.**
  1. Trader i reads the current prices (the marginals of S, unbounded and read-only).
  2. From its evidence x_i it produces a trade: a log-potential table on one clique scope, chosen from the scopes the market lists, of b bits (quantised).
  3. The market applies the trade multiplicatively (LMSR), so the next trader sees new prices.

## Loss
- **Per trader.** Each trader's loss is its LMSR profit at settlement, log p_after(y) - log p_before(y), with y's marginal computed by sum-product over z.
- **Summed.** Summed over traders, this telescopes to the log loss of the final state, so the local losses train the global objective.
- **Learning.** The market's theta is learned on the same summed loss.

## Prediction to test (Ostrovsky 2012: aggregation for all information structures iff the security is separable)
- **Arm A, y only.** Market on y alone (m = 0), on Z9 k-bit parity. Traders reading one noisy bit cannot move the price of y, so regret stays near the prior's.
- **Arm B, latent securities.** m = k latent bits with a learned chain of partial parities (width 2), y linked at the end. Traders trade on "their" z_j, and the regret should approach Bayes.
- **Arm C, oracle structure.** Hand-set securities z_j = b_j with the true parity chain. This is the reference for what learning should find.
- **Bars.** Exact Bayes, the appended trader-transformer at the same message size b, and extremizing.
- **Then.** E2 / E3-sybil (duplicates: does the price-reading trader trade nothing on repeated evidence?) and Z5 (several questions).

## Open design questions for Kelvin
- Do traders choose which listed scope to trade on (learned routing), or is each trader assigned one?
- Grow the structure during an episode (add a latent when trades want a scope the tree lacks), or learn a fixed structure across episodes first?
