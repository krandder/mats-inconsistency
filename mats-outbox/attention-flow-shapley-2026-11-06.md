# Attention flow, Shapley values and our market ideas: which connections survive (6 November 2026)

For Kelvin, from a subagent of opus-mats. Read-only: I sent nothing and trained nothing. The only computation was an
exact proof check of a few lines' algebra (LPs on 4-token graphs and Shapley sums over at most 4 players, a few CPU
seconds): `work/attention-flow-shapley/checks.py`, with its output in `checks.out` next to it. Labels: PROVEN (proof
here, checked numerically where marked), CITED, MEASURED (ours: earlier lanes or the check script), HEURISTIC,
CONJECTURE. Same test as `work/not-circuits/connect.md`: (1) exact mapping, (2) something it buys that we lack,
(3) where it breaks.

## TL;DR

1. No new connection survives. The real link is that Shapley credit equals market-scoring-rule payments averaged over trading orders. It is exact, but it is Conitzer 2009 (arXiv:1205.2654) and has been in our trader-economy note since 29 Sep (G2, parked after 19 nulls).
2. E&J's theorem is trivial. They fix one max flow and set v(S) = the sum of fixed outflows, an additive game in which any numbers are "Shapley values" (Metzger et al. 2022 call it trivial). I also show that v(N) = 1 always, the max flow is not unique, and rollout is one of the max flows.
3. Flow and rollout see only attention weights. Our departures from Bayes and our deduplication happen in value processing: the LayerNorm cap, error signs, and pool-then-MLP at depth 1. At a softmax layer, a copy with an identical key can only gain credit.
4. Shapley splits credit between copies, but splitting credit is not deduplication. The copies' total stays put only in additive (log-odds) games. In a 3-source information game it rose from 0.171 to 0.178 nats; in a complements game it fell from 0.5 to 0.333.
5. Strength: weak (re-descriptions of things we have). No tests proposed, no decision needed. One question for you is at the end.

## 1. The papers in precise terms

### Abnar & Zuidema 2020, "Quantifying Attention Flow in Transformers" (arXiv:2005.00928, ACL 2020)
- Graph (CITED): a DAG with one node per (layer, position). Edges run from each node to every node of the layer below, weighted by
  attention averaged over heads, with the residual added as A = 0.5·W_att + 0.5·I. Single heads are covered in Appendix A.1.
- Rollout (CITED): Ã(l_i) = A(l_i)·Ã(l_{i-1}), a product of the residual-adjusted attention matrices. A path's weight is
  the product of its edges, summed over paths. Cost O(d·n²).
- Flow (CITED): the attention weights are the capacities of a flow network, and the score of input token v for node u is the
  max-flow value from u to v. In the authors' code (`compute_flows` in `attention_graph_util.py`, read 6 Nov) this is a
  separate `maximum_flow_value(G, u, v)` for each input v, then normalised to sum to 1 over the inputs. A path's weight is its
  minimum edge. Cost O(d²·n⁴).
- Evidence (CITED): a 6-layer, 8-head, 128-wide encoder on subject-verb agreement (accuracy 0.96). Spearman with blank-out at
  layer 6: raw 0.29, rollout 0.71, flow 0.70. With input gradients at layer 6: 0.46, 0.54, 0.61. On DistilBERT/SST-2
  every method is 0.08-0.14.
- Stated limits (CITED): "many simplifying assumptions"; only attention weights are used; encoders only (decoders need a
  receptive-field normalisation); they suggest effective attention or gradient adjustments as follow-ups.

### Ethayarajh & Jurafsky 2021, "Attention Flows are Shapley Value Explanations" (arXiv:2105.14652, ACL 2021 short)
- Game (CITED): players are possible subjects of an explanation. Absent players are replaced by "a non-subject that cannot affect the
  outcome". v: 2^N → R with v(∅) = 0.
- Axioms (CITED): null player, symmetry and additivity, which with efficiency characterise the Shapley value.
- Prop 1 (CITED): outside the degenerate case, attention weights are not the Shapley values of any game. In the proof, removing a
  player redistributes its attention, so the weights are recomputed for each coalition.
- Prop 2 (CITED): players all come from one layer, f is a max flow on the attention graph, and v(S) = |f(S)|, "the value of the
  flow when only permitting flow through players in S". The proof says the flow "is only calculated once — with the
  entire graph — and not for each possible subgraph". So v(S∪{i}) = v(S) + |f_o(i)| for every S, and the outflow is the
  Shapley value. For players in different layers efficiency fails. The level is per layer, never the whole network.
- Rollout (CITED): they say it "cannot be Shapley Values", by "a trivial extension" of Prop 1. Prop 3: leave-one-out is not
  Shapley unless degenerate.
- Stated limits (CITED): the equivalence holds only for the payoff "total flow through a layer", which they describe as
  "reflective of model confidence but not of the prediction correctness". Exact Shapley is intractable in general. The paper has no experiments.

### What the papers' own constructions imply (my checks)
- **The E&J game is additive, so the theorem says nothing about the network** (PROVEN). With f fixed, v(S) = Σ_{i∈S} w_i.
  The Shapley value of an additive game is w, whatever w is. The same construction therefore makes any per-player numbers
  "Shapley values": raw attention received, rollout, gradient×input. Metzger et al. (arXiv:2205.15389) state this
  ("this theoretical correspondence to a Shapley value is trivial") (CITED). The difference between Prop 1 and Prop 2 is
  a choice, not a theorem: Prop 1 recomputes the weights for each coalition and Prop 2 forbids recomputing (PROVEN, from their
  proofs). If the flow is recomputed for each coalition, the game becomes a network-flow game (Kalai & Zemel 1982), in
  which players from one layer compete for downstream capacity. Then outflows are not Shapley values, which is Metzger et al.'s
  counter-example (CITED).
- **The payoff is the constant 1** (PROVEN for A&Z's single-source case). Every node's outgoing capacities sum to 1. Send
  each node's inflow onward in proportion to its capacities. Each layer then carries total flow 1, so every node receives at most 1 and
  every edge stays within capacity. That flow has value 1 = the source's out-capacity, so it is a maximum. Its inflow at input j is
  the rollout entry R[u, j]. So **rollout is one of the max flows**, which contradicts E&J's sentence on rollout. v(N) does not depend on the
  input at all, so the "model confidence" reading fails. Checked on 5 random residual graphs (n = 4, L = 3): max flow
  1.000000 every time, and rollout feasible (MEASURED, check script).
- **The "attention flow" Shapley value is not determined by the attention graph** (MEASURED, check script). Over all
  max flows, every one of the 20 input shares in those 5 graphs ranged from 0.000 up to 0.33-1.00. So E&J's
  application, "zero flow at layer k means the token's label information was already extracted", depends on which max-flow
  solver was used.
- **E&J's object is not A&Z's published score** (PROVEN from the code). A&Z run one max flow per input token and then
  normalise. The per-pair values summed to 2.56-3.16 before normalising (MEASURED, check script), so they are not the node
  flows of one flow. In all 5 graphs the normalised vector happened to fall inside the max-flow ranges. I have not proved that it always does.

### Follow-ups and critiques (each CITED; titles and ids checked against the arXiv API on 6 Nov)
- Attention as explanation: Jain & Wallace, arXiv:1902.10186; Wiegreffe & Pinter, arXiv:1908.04626; Serrano &
  Smith, arXiv:1906.03731; Bastings & Filippova, arXiv:2010.05607 (use saliency instead); Brunner et al.,
  arXiv:1908.04211 (identifiability).
- The weights leave out the values: Kobayashi et al., arXiv:2004.10102 (vector norms ‖αf(x)‖) and arXiv:2109.07152
  (residual and LayerNorm). Liu et al., arXiv:2201.12114: attention-based explanations, raw attention most of all, cannot
  give the polarity of a feature's effect (a "faithfulness violation test").
- Rollout kept, layer matrices improved: ALTI (Ferrando et al., arXiv:2203.04212) builds per-layer contribution matrices
  from the whole block and combines them across layers by the same product, R^l = C^l···C^1 (their Eq. 11). It beats rollout
  on comprehensiveness and sufficiency. GlobEnc (arXiv:2205.03286) and DecompX (arXiv:2306.02873) are in the same line, and Value
  Zeroing (Mohebbi et al., arXiv:2301.12971) measures context mixing per layer by zeroing values. Conservative relevance
  propagation: Chefer et al., arXiv:2012.09838; Ali et al., arXiv:2202.07304. Gradients on attention: Hao et al.,
  arXiv:2004.11207.
- The Shapley claim: Metzger et al., arXiv:2205.15389 (trivial; may break symmetry if read with recomputation). No
  paper defending the claim against this was found (one web search, 6 Nov). Shapley in general: Sundararajan &
  Najmi, arXiv:1908.08474 ("the many Shapley values": uniqueness only within a chosen game); Kumar et al.,
  arXiv:2002.11097. Survey: Lyu et al., arXiv:2209.11326.
- Markets: Conitzer 2009, arXiv:1205.2654, Def. 5 and Prop. 4: the "Shapley value information mechanism" is the average
  over orders of sequential proper-scoring-rule payments, and it is ex-interim incentive compatible when the model is consistent. Agarwal
  et al., arXiv:1805.08125: Shapley is not robust to replication (copies). Chen & Waggoner, arXiv:1703.08636: informational
  substitutes aggregate at once, complements late.

## 2. What we already have (the agenda side)

- G2, paying the parts (CATALOGUE, the trader-economy note of 29-30 Sep):
  - MEASURED (ours): Shapley pay, MSR over random orders, LOO, IG and Owen splits, on agents, weights and hidden units. Line 28 of the note already states "Averaging those MSR payments over all orders gives exactly the Shapley value".
  - MEASURED (ours): Shapley values units well (Spearman 0.965 with true unit value, E-MLP-3) but trains worse than backprop (0.609 against 0.435, E-MLP-1). Copies of a unit get 0.52 and 0.51 of its pay (E-MLP-5). Layer-wise Owen gives inputs 19% where flat Shapley gives 61% (E-SPLIT-2).
- Attention as a market (hub wiki trader-soup-neural-mixture-design, read through `recall`, 6 Nov):
  - Already proposed, 19-21 Oct: "softmax = LMSR" settlement across attention (values paid at the uniform price, keys paid (a_j − 1/n)·u_j).
  - MEASURED (ours, trader-soup-validation-study, M1): first-order books leak 45-95% of revenue across live attention. The three-rule settlement "conserves exactly but gives no ranking or ablation-faithfulness gain" over the simpler rule.
- Backprop as conditional markets (outbox/backprop-conditional-markets-2026-10-23.md): realness = |dl/da_j|. Open question:
  can agents find J by bidding, without the Jacobian?
- G4/G5: dedup is partial (0.25-0.70 of a copy removed). Without keys the cue is the exact tie of query readings, and depth 1 is
  "pool then nonlinearity" (outbox/pfn-dedup-depth-2026-10-26.md). Phase B's departure involves the block-1 LayerNorm cap and
  error signs (outbox/cm-interp-phaseB-2026-10-28.md, E20, E55, E57).
- G8: "a myopic Bayesian trader buys its log-likelihood" (outbox/scale-free-agency-ngo-2026-10-29.md).
- 21 Oct (outbox/pid-deep-dive-2026-10-21.md): two-source Shapley = I(Y;M_i) + (S − R)/2; "Shapley is not
  replication-robust". Your words that day on a Shapley payout: "Yeah this is not totally right ... We can come back to the
  market perpective." (outbox/research-orientation-2026-10-20-addendum.md).

## 3. Connections table

| # | Candidate | (1) Mapping | (2) What it buys | (3) Where it breaks | Verdict |
|---|---|---|---|---|---|
| C1 | E&J's flow-Shapley as a cheap exact pay rule for network parts (G2) | None. E&J's game is additive with no counterfactual: v(S) never removes anything. G2's games remove the coalition and rescore (zero-out ablation). | Nothing. In an additive game every score is "Shapley". | The payoff is the constant 1, the max flow is not unique, and rollout also qualifies (section 1, PROVEN/MEASURED). | **Fails** |
| C2 | LMSR payouts and Shapley (your market ideas, G2, G8) | Exact. With v(S) = s(P(E \| info of S)) − s(prior), a fixed-order MSR pays each trader its marginal v-step in that order, and the average over orders is Shapley (CITED, Conitzer Def. 5). The premise "scoring rules pay for information gain, not Shapley marginal contribution" is half wrong: both pay realised score improvement and differ only in order. In the check, the copy got 0 when second and 0.151 when it came after b and c, while Shapley gave each copy 0.089 (PROVEN numerically). | Nothing new. It is in our 29 Sep note, already measured (MSR-16 is within 18% of exact Shapley, E-AUD-2) and tied to PID on 21 Oct. Under G8's conditionally independent sources the shares bought (log L_i) do not depend on order, so all the rules coincide. | Order matters only with interactions (shared errors, complements), where G8 already needs dense attention. Neither paper is involved. | **Exact but already ours**: fails (2) |
| C3 | Rollout/flow as conservative credit through attention (royalties, trader-soup settlement, realness J) | Partial. Rollout is exactly the Jacobian ∂h_L[q]/∂h_0[i] of a surrogate that freezes the attention, makes values identity and reweights the residual to 0.5/0.5 (PROVEN). As a pay rule it is a royalty proportional to attention weight, blind to what a value delivers. | Nothing. Our three-rule settlement already conserves across attention with value- and routing-aware shares and gained nothing over the simpler rule (MEASURED, ours). ALTI and DecompX are the faithful versions of the same product (CITED). The real J is cheap to compute in our tiny nets. | It drops the routing term (credit for q·k), which is the "attention leak" we measured. Polarity is lost (Liu et al.). In G2, value-blind royalties (equal split) paid noise agents almost as much as useful ones (0.89 against 1.04, E-T2-1); an attention-proportional split is also value-blind (HEURISTIC that it fails the same way, not tested). | **Analogy only** |
| C4 | Attention flow as credit for evidence sources in PFN aggregation and dedup (G4, G5) | None to the quantity we measure. Our dedup and departure metrics are behavioural (pull of a copy, weight against Bayes's). Flow uses attention weights only. | Nothing. It predicts no dedup fraction or departure. | (a) At one softmax layer, an exact copy with an identical key raises its group's attention from e^s/(Z+e^s) to 2e^s/(Z+2e^s) (PROVEN), so attention credit for a duplicated source can only rise there, while the depth-1 nets remove 0.36 of a copy after pooling, in the MLP. (b) Phase B's departure runs through the LayerNorm cap and error signs, which attention weights cannot express. Whether later layers lower a copy's attention: not checked. | **Fails** |
| C5 | Shapley symmetry and additivity vs double counting | Exact definitions. Bayes-correct dedup is a statement about v(N): copies add nothing to the posterior. Shapley then shares v(N) (efficiency) and gives copies equal shares (symmetry). | Nothing for dedup: the credit rule says nothing about whether the aggregate double counts, which is what G4 measures. | Shapley is not merge-invariant. A single source's credit (a) equals the copies' total only if the game is additive between the copy group and the rest (log-odds of conditionally independent evidence: 1.3 → 0.65 + 0.65). In the expected-information game (binary Y, flip rates 0.2/0.3/0.35) it rose from 0.1714 to 0.1783 nats, taken from the substitutes b and c. In a complements game it fell from 0.5 to 0.333 (PROVEN numerically). G2's 0.52 + 0.51 = 1.03 (E-MLP-5) is consistent with a small gain but not distinguishable from none. A market also dedups only if the copy trader knows its signal is already in the price; a myopic trader who takes the price as prior buys log L again (PROVEN, the conditional-independence caveat in the Ngo report). | **Fails** as a dedup mechanism. Credit-side facts known (Agarwal et al.) |
| C6 | Max-flow/min-cut as a bound on information from a source (your 21 Oct "paths can be superposed") | No. Attention weights are mixing proportions, not capacities. Without noise, scaling a value by a is invertible, so a carries no information limit. | With per-node noise, a cut-set bound exists, but edge capacity would be about ½log(1 + a²·SNR), not a, and nodes are multiple-access sums. A&Z's flow is not that bound (HEURISTIC). | Nothing in either paper; our G7 bit accounting measures bits per layer directly. | **Analogy only** |
| C7 | E&J's null-player guarantee ("information extracted before layer k") | The game's payoff is total flow, not the prediction. | Nothing. | v(N) is constant and zero shares depend on the solver (section 1). | **Fails** |

## 4. Tests

None proposed: no candidate passes (2). What would revive C3/C4 (CONJECTURE, not proposed for running): a
Jacobian-free, value-aware credit computed in one forward pass that matches exact token-Shapley of the net's log
score on our PFNs. ALTI and DecompX are the nearest published forms, and neither paper here supplies one.

## Question for Kelvin

Which market idea did you have in mind? (a) Paying network parts (G2): the Shapley-equals-ordered-MSR identity is
there already, and these papers add nothing. (b) Pricing evidence sources in an aggregator (G4/G8): Shapley is order-averaged
LMSR pay, and it neither performs nor measures deduplication. (c) Something about flows through the
(depth, token, rollout) grid: then the open issue is C6's, that attention weights are not capacities.

## Sources

- Papers read in full from arXiv PDFs on 6 Nov: 2005.00928 (v2), 2105.14652 (v1), 2205.15389 (v1, sections 1-3),
  2203.04212 (method and conclusion), 1205.2654 (sections 2-4). A&Z code:
  github.com/samiraabnar/attention_flow, `attention_graph_util.py`, `compute_flows`. Other arXiv ids: titles and
  abstracts via the arXiv API, 6 Nov. Kalai & Zemel 1982 (Math. Oper. Res. 7(3)) is cited from memory, not re-read.
- Ours: /home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md;
  calibrated-ai/trader-economy-note-20260929.md (lines 3, 27-30, 65-79); trader-theory/THEORY-4.md;
  mats/outbox/{backprop-conditional-markets-2026-10-23, scale-free-agency-ngo-2026-10-29, pfn-dedup-depth-2026-10-26,
  cm-interp-phaseB-2026-10-28, pid-deep-dive-2026-10-21, research-orientation-2026-10-20-addendum,
  research-inventory-2026-11-03}.md; mats/work/bubbles/CONJECTURES.md; hub wiki trader-soup-neural-mixture-design.md
  and trader-soup-validation-study.md (via `recall --read-wiki`, 6 Nov).
- Check script: mats/work/attention-flow-shapley/checks.py and checks.out (scipy LP, exact Shapley enumeration).
