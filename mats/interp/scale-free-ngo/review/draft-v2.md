# Ngo's scale-free agency and our open-latent definition (29 October 2026)

For Kelvin (theory lane); status tags in brackets.

## Summary
- Ngo (March 2025) proposes one theory of agency for every scale: agents are coalitions of subagents, recursively, with something like prediction markets for beliefs, auctions for actions and voting for values, chosen for incentive compatibility. He calls it "vague and tentative"; no follow-up I found proves his central hypothesis.
- Our definition formalises a restricted piece, honest probabilistic belief combination, where the link is exact: in a log market scoring rule (LMSR) market over the open latents, a myopic Bayesian trader buys its log-likelihood and the market state sums the trades.
- Mismatches: his subagents are strategic, have goals and may overlap, and his formal follow-up studies welfare under log pooling, not evidence fusion. Ours offers him a communication argument for posted consensus and a cost for grouping subagents; his gives us a caveat on our first round's honesty. No decision needed.

## 1. What Ngo claims
"Towards a scale-free theory of intelligent agency" (LessWrong, 21 March 2025) wants a theory that holds when you "zoom in" or "zoom out". Expected-utility maximisation separates goals from beliefs; active inference unifies them, but its agents are "cooperative within themselves", privileging one level. Coalitional agency: subagents, each a "roughly-internally-consistent cluster of beliefs and goals" (a matter of degree), bargain over incentive-compatible decision procedures rather than outcomes. Inside an active-inference agent he proposes replacing precision-weighted averaging with a prediction market (modelled by Garrabrant induction), something like Demski et al.'s auctions for actions, and random-ballot voting for values, "or similar mechanisms". Hypothesis: idealised bargaining "in many cases" converges to such a procedure. In replies he says "clustering is efficient" and that subagents are not "totally separate entities".

Later posts, to October 2026: well-foundedness; "scale-free approaches under which inconsistencies mostly get resolved locally"; belief webs, "typically locally consistent ... but not necessarily globally consistent"; Garrabrant-induction wealth as degree of truth; Knightian uncertainty; social theory. The formal follow-up, Lee, Kondor and Ngo (arXiv:2509.06701, ICML 2026), treats agents as outcome distributions that also encode goals, with log-score welfare; a coalition is their log pool with weights summing to one; strict unanimous gain is impossible under linear pooling or with two outcomes, possible with three or more.

## 2. Correspondences and mismatches
| Ours | Ngo | Status |
|---|---|---|
| subtree with few open latents | subagent; an agent as "a densely-connected region" of a belief web | interpretation; ours are disjoint and sized |
| message: likelihood of the open latents | a trade. In LMSR over outcomes k of the open latents, a myopic risk-neutral trader taking the price p as prior, with full-support evidence x conditionally independent of earlier traders' given k, buys b log L(x\|k) + c shares and nets b log[L(x\|k*)/Σ_k p_k L(x\|k)] at settlement | exact (Hanson 2007); Garrabrant induction is a different market |
| merge: add log-likelihoods, sum out closing latents | the market maker adds shares; junction-tree market makers marginalise (Laskey et al. 2018; your bayes-markets code); markets can run message passing (Storkey 2011) | cited |
| round-2 message, Gaussian source of known precision tau: score and minus curvature at the broadcast mu | predictive coding's precision-weighted prediction error tau(x - mu), and tau | exact |
| routing; rounds | coalition formation (iterated vote-trading "produces hierarchical coalitions"); trading at a posted price, predictive-coding iterations | interpretation |

Your 22 October design lines fit [interpretation]: the bounded consensus is a market maker over few securities (a fixed size for all N needs exponential families, under regularity, T3); "query before compression" is the posted price; "forecast ... plus own contribution" is a trade on existing securities plus a proposed new one, a latent open at the agent's cut.

Mismatches:
1. **Incentives.** Our components are honest by assumption, removing his strategic problem rather than solving it; our trained combining layers do weight sources by inferred precision (median r = 0.87, correlational) [measured].
2. **Pooling.** Lee et al. ask when a log pool with weights summing to one benefits every member: compromise, not evidence fusion. With a common prior pi and private evidence conditionally independent given the target, Bayes is pi^(1-N) Π P_i over posteriors, or pi Π L_i over likelihoods (our prior-as-sink, entered once). Each member's weight is one, so for N >= 2 no member weights summing to one match Bayes for all evidence [exact]. Multiplicative pooling is the only "individualwise Bayesian" rule (Dietrich and List 2016, Thm 3).
3. **Mixture and product.** Kelly-bettor markets price the wealth-weighted average belief, wealth updated by Bayes' rule (Beygelzimer, Langford, Pennock 2012): a mixture, like random ballot. Our cross-source merge is a product; tiny-PFN Bayes needs both [exact].
4. **Scope.** No actions, goals, self-reference, overlap or Knightian uncertainty; we use exact Bayes in a known model, while he looks "beyond the Bayesian expected utility maximization paradigm".
5. **What is scale-free.** For him the whole theory, incentives included; for us the merge rule, while sizes may grow (and must, like log N, for bounded noisy messages, T11d).

## 3. What each side says about the other
Ours to Ngo:
- **Posted consensus**, a reason for markets and predictive coding independent of incentives. Without a broadcast, reports must cover everywhere the posterior might land (48-64 numbers per source for KL <= 1e-3; linear rank 1,646 at RMS 1e-3 with three open latents); after one, 8 numbers give window-conditional KL 8.6e-8 at N = 1024 [measured; fixed codes]. Trained two-pass trees match a dense transformer up to N = 64, one-pass trees do not [measured, draft].
- **Which securities.** Garrabrant induction prices every sentence; a bounded coalition must choose. Securities on the open latents suffice, and are minimal for their joint posterior given rich outside data (T2, T2m). A target-only market can fail on parity: in Ostrovsky's (2012) Example 1, an XNOR of two traders' signals, truthful announcements "will never learn the true value", while a 1-bit message across the cut suffices. This is your learned-securities question.
- **Grouping.** Forming his clusters is, in inference, a routing cost: log N label bits if membership is visible (T5); about min(T, 1/rho^2) bits per member if it shows only in correlated behaviour (T7, proven for a promise problem). Trained dense nets stall there: KL 0.194 without the routing mask, 0.011 with it, at N = 12 [measured, draft].
- **Well-foundedness**, a structural analogue of one example only (his concept concerns conflict, ours dependence). Fractistan, "fractally dividing the country in two at every level of organization", against Holistan is our straddling groups: intermingled groups put every split group's latent in each node's message, aligned ones pass only y* between groups [exact, our model].
- **Consistency.** Junction trees make local consistency global; with cycles it need not be (Wainwright and Jordan 2008). Conjecture: only-locally-consistent belief webs are what bounded agents get on tangled tasks.

Ngo to ours:
- **Honest first round.** Under market losses each source trades twice. For a binary security over finitely many rounds, truthful betting is a perfect Bayesian equilibrium when signals are conditionally independent given the state, but not when they are unconditionally independent (Chen et al. 2010). Conditional independence given y* means only the target is open (W0); with shared errors (W1, W2), first-round reticence becomes possible.

## 4. Experiments on our small worlds
**D1. Local market losses against end-to-end.** W0 and W1, N = 8 and 64. Per-source networks read their history and the posted market state, then in each of two passes trade a few numbers toward their current posterior (nothing counts twice). Train (a) end-to-end or (b) each agent on its own LMSR profit. Prediction: (b) matches (a) in W0; in W1 (b) is worse, its first-pass trades carrying less evidence as straddling groups grow, until securities on group errors are added. Evidence: KL to exact Bayes; probes of each agent's sufficient statistic in its first-pass trade. Refuted if (b) matches (a) in W1 or fails in W0. Moves design point 4.

**D2. Securities as open latents.** W1 with regions aligned with groups (Holistan) or splitting them (Fractistan), and W2 (hidden pairs); regions write k fixed-width slots after the root's broadcast. Conjecture: the smallest k for KL <= 0.01 grows with straddling groups, not N; nonlinear probes recover the open latents; in W2, k stays large until a pilot re-routes the pairs. Refuted if k grows with N. Moves M2 and the bounded consensus.

**D3. Predictive coding in trained transformers.** Dense set nets on W0, N = 64 and 256. Prediction: layer 1 forms a pilot; layer-2 per-source features carry each source's score and curvature at it. Evidence: probes (R^2 >= 0.5); patching in another episode's pilot moves them as the score predicts. Refuted if they ignore the pilot while KL matches the two-pass tree. Moves M2.

## Checks
Read in full: the post, its 55 comments, seven later posts, Lee et al., Ostrovsky's Example 1, Dietrich and List's Theorem 3; other citations by abstract (identifiers in work/scale-free-ngo/CITATIONS.md). Prior work: wiki ngo-garrabrant-agency-research; your 29 June Codex session. GPT-6 (gpt-6-astra) review applied: 18 findings, four misreadings of Ngo (work/scale-free-ngo/review/).
