# Scope lens: what this study should show Kelvin

Scoping note for the traders-net study. Lens: Kelvin's taste. 2026-09-27, a subagent of the scoping workflow.
Sources read: `../CONTEXT.md`; `../../logical-soup/REPORT-SOUP.md` sections 1-3; `../../wealth-net/README.md` and
`../../wealth-net/results/summary.json` (`../wealth-net/RESULTS.md` does not exist yet); hub wiki `consistency-ft.md`
(the core-idea, Garrabrant-fork and review sections), `bayesian-rl-environments.md` and `futarchy-research.md`; the
fleet memories on "surprises" and HTML pages (Kelvin, 09-27) and Grafana over HTML (09-24).

## 1. TL;DR

1. **Kelvin's taste, in one line:** a general mechanism with a scaling axis (bitter lesson), coherent and calibrated
   by construction, shown as a live demo. He also wants one surprise per result, with its punchline in a sentence.
   A proof that two things are formally the same bores him unless a plot follows it.
2. **wealth-net already has two surprises.** Both are partial, from `summary.json` at step ~8.8k of 60k, read at
   about 21:55Z. (a) The net prices LFSR-16 (eval BCE 0.000-0.001) and CA rule 110 (0.006-0.022), which 64k evolved
   soup traders never priced. (b) The net is badly overconfident on pi parity: eval BCE is 3.6-3.7, against 0.69 for
   a coin. Yet the books pay pi the most (the out-layer wealth on pi is 466-537, the largest of any family).
   **Wealth kept on training data rewards memorisation.** This is Kelvin's "traders shouldn't think they are smarter
   than the market", failing in front of us.
3. **Three Kelvin-shaped experiments:**
   - (E1) Humble traders: prequential books plus market-anchored bounded bets.
   - (E2) The books as a futarchy forecast: at every scale, wealth predicts the effect of removing a unit, and we
     score that as a forecast.
   - (E3) Priced MoE: thousands of tiny experts, with capacity priced by tâtonnement, wealth-gated routing, and
     birth and death.
   E1 and E2 run on existing code for about $20; E3 takes days and about $50-150.
4. **Risks:**
   - The inner books are first-order attribution (activation × gradient). Molchanov et al.'s Taylor pruning criterion
     is the same quantity, so the novelty has to come from wealth *dynamics*, not the ledger.
   - Conservation leaks 45-64% at the first inner cuts once attention is live.
   - DeepSeek's loss-free balancing is already a capacity price.

## 2. A model of Kelvin's taste (evidence)

| trait | evidence | what it implies for this study |
|---|---|---|
| Bitter lesson | Section 5 of the soup report frames the result as "the frontier of patterns priced correctly expands with compute"; "nothing hand-coded". | Every result needs a scaling axis: units, experts, K/Q or FLOPs. An anecdote about one small net is not a result. |
| Coherence and calibration | Consistency-FT (wiki `consistency-ft.md`): negation, conjunction and entailment bounds; RLE read as amortised logical induction; Brier and log-score everywhere. | Report calibration (ECE, reliability plot) and the A∧B / A∨B coherence violations for every arm, and compare them with the soup's 0.01-0.09. |
| Markets as the mechanism | Futarchy research (wiki `futarchy-research.md`): LMSR, batch clearing, manipulation. The LLM-trader arms were "calibrated but not coherent: the market underperforms a poll". | Frame units as traders only where a price, a budget and settlement are real. The test of the framing is whether the market outperforms a poll of its units. |
| "Traders know they only improve here-and-there" | His verbatim line in CONTEXT.md. | The key design axis: anchor to the market price, bound the bets, settle out of sample. |
| Fractal, non-hierarchical ownership | His verbatim line in CONTEXT.md. | Conservation across scales is itself a coherence property. Show it as one zoomable picture, and show where it fails. |
| Compute = wealth; shared computation | His verbatim line in CONTEXT.md. | Charge FLOPs. Compare the soup and the net at equal compute. |
| Surprises, demos, HTML | Memory `kelvin-class-a-and-surprises` (Kelvin, 09-27): "I like htmls, demos". Memory `grafana-over-html-pages` (09-24): recurring metrics go to Grafana. | One HTML demo page per finding, plus Grafana panels for the over-time wealth curves. |
| Anti-ceremony, anti-overfitting | Memories `no-overfitting-to-metrics` and `root-causes-not-instances`. | Hold-out families; no tuning to the headline metric; each failure treated as a class. |
| Deferred mechanisms | "we don't want any Harberger tax now". | Do not build tax or self-assessment machinery. Observe wealth first, then gate by it. |

## 3. What he would like, and what would bore him

**He would like:**
- **Soup vs net on the same streams, on one page.** Show per-family price trajectories, and show where shared
  computation beats 64k independent traders (LFSR, CA110). Show where it is worse: pi overconfidence. The soup was at
  least humble there, at 0.25 Brier.
- **A fractal wealth treemap.** Model → layer → head/MLP → neuron, sized by wealth, coloured by the family that paid.
  Toggle between in-sample and prequential books to watch the pi units deflate.
- **The books graded as a forecaster.** For each unit or coalition, the books predict "removing it changes held-out
  loss by X". Score that forecast with a calibration plot, as if the internal market were a futarchy decision market
  over pruning decisions.
- **"Load balancing is a price."** A short derivation plus a live plot: an MoE router's bias, updated by excess
  demand, is a Walrasian price. The auxiliary loss is a penalty-method stand-in for it, and routing collapse is a
  market with no price.
- **A universal.** The soup has a wealth Gini of 0.75-0.82. The net's head Gini is already 0.78-0.88 (positive part).
  If the same number shows up across substrates and scales, that is the kind of surprise he likes. It also needs a
  null model to rule out a statistical artefact.

**He would be bored by:**
- Pages re-deriving softmax = LMSR or cross-entropy gradient = settlement. Chen and Vaughan, Storkey, and Balduzzi
  already have these. One paragraph with citations, then data.
- "Negative-wealth neurons hurt the model." Without a twist this is Molchanov's Taylor pruning, with parasites
  renamed. It needs the dynamics (cumulative, prequential, multi-scale, gating) to be new.
- Hyperparameter sweep tables with no one-sentence claim. Single-neuron interpretability vignettes.
- Harberger taxes, and mechanism ornaments on a model that is not yet observed.
- Long reports. He wants a three-line TL;DR, a demo link and a Grafana panel.

## 4. Candidate headline insights to test (not yet claims)

1. **Settlement must be prequential.** A market settles on outcomes the traders had not seen. Books over re-used
   training data pay memorisers; the pi evidence supports this. Keep the books on each example's first visit, or on
   a held-out stream, and wealth becomes out-of-sample information. That is the logical-induction reading.
2. **Shared computation vs selection.** The net found LFSR and CA110. The soup did not, despite 64k traders and 64x
   the fuel. Is the lever shared computation (gradient descent through a shared trunk), or just that selection is a
   weak optimiser? Compare them at equal FLOPs.
3. **Humility has a mechanism.** The two mechanisms are anchoring to the market price (the prior log-odds ℓ0 in the
   pooling formula) and bets bounded by wealth. Together they should give chance-level prices on unpredictable
   families (pi, CA30) without losing LFSR.
4. **The fractal accounting is exact only for degree-1-homogeneous blocks.** With live attention, the first cuts leak
   45-64% of R (`cons_resid_cum` -0.64, -0.45, -0.45). An attribution with the completeness axiom (Integrated
   Gradients, DeepLIFT or Shapley paths) restores exact sums, at more compute. Kelvin's "big trader = sum of small
   traders" needs one of these.
5. **MoE balancing is already a market.** DeepSeek's bias is dual ascent on a capacity constraint. What wealth adds is
   memory: a long-run budget, Kelly-style priors on routing, and bankruptcy and birth.

## 5. The three most Kelvin-shaped experiments

### E1. Humble traders: prequential books plus market-anchored bounded bets
- **Hypothesis.** A net whose units settle out of sample and bet relative to a market price stays at chance on
  unpredictable families, keeps LFSR and CA110, and is coherent on A∨B.
- **Arms**, on the wealth-net streams (8 families, 3 seeds):
  - (a) the current net;
  - (b) books settled prequentially: first visit of each example, or a disjoint held-out stream;
  - (c) output = confidence-weighted pooling ℓ* = (ℓ0 + Σ k_i ℓ_i)/(1 + Σ k_i). Here ℓ0 is a running base-rate
    "market" and k_i comes from prequential wealth;
  - (d) logit clipping with a budget proportional to wealth;
  - (e) the logistic baseline and the soup numbers, as references.
- **Measures:**
  - per-family eval BCE and Brier, and ECE;
  - A∧B / A∨B coherence violations;
  - rank correlation between in-sample and prequential wealth;
  - FLOPs to reach each Brier, compared with soup virtual instructions.
- **Demo.** "The overconfident trader": price trajectories per family, soup vs net vs humble net, with the pi units
  shrinking as the books switch to prequential.
- **What would excite him.** The humble arm holds pi and CA30 at 0.69 BCE, keeps LFSR under 0.01, and cuts
  A∨B violations close to zero. In-sample wealth ranks pi first and prequential wealth ranks it last.
- **Kill result.** The pi overconfidence goes away with plain early stopping or weight decay, at no cost. Then it was
  ordinary overfitting and the market framing added nothing: report that plainly.
- **Cost.** 1-2 days, on CPU or L4 hours, under $20.

### E2. The books as a futarchy forecast of interventions, at every scale
- **Hypothesis.** Cumulative prequential wealth forecasts the held-out loss change from removing a unit or coalition
  better than instantaneous Taylor importance, gradient norm or a random choice. Its failures are structured: Hydra
  self-repair, and synergy in coalitions.
- **Setup.** Use the wealth-net checkpoints and the ablation test already planned in CONTEXT item 7.
  - Units: neurons, heads, layers, random coalitions of each, and coalitions that cross layers.
  - Books: first-order books (current) vs completeness-satisfying books (Integrated Gradients along the ablation
    path).
  - Treat each unit's forecast as a market price on "the loss rises by more than δ if it is removed", resolved by
    the actual ablation.
- **Measures:**
  - calibration slope and Brier of the forecast, per scale;
  - superadditivity, defined as ablation(A∪B) − ablation(A) − ablation(B): the size of the "synergy" the sum rule
    misses;
  - the fraction of each ablation that downstream units repair.
- **Demo.** The zoomable fractal treemap. Next to it, a reliability plot per scale and a toggle between
  "books predicted" and "ablation measured".
- **What would excite him.** A calibration slope near 1 at head and layer scale. Prequential cumulative wealth beating
  Molchanov's instantaneous criterion. Coalition wealth adding up within a few percent once the books are complete.
  That last result is the quantitative form of "big trader = sum of small traders".
- **Kill result.** Cumulative wealth is no better than instantaneous |a·g|. Then the books are a pruning heuristic and
  nothing more.
- **Cost.** Hours to 2 days, under $10, and it reuses what is already running.

### E3. Priced MoE: thousands of tiny experts, priced capacity, wealth-gated routing, birth and death
- **Hypothesis.** The smallest modification of MoE that makes wealth real:
  - the routing bias is the capacity price, updated by excess demand (as in DeepSeek's loss-free balancing);
  - each expert's routing prior is log-wealth from prequential books (Kelly reading: wealth-weighted price = BMA);
  - an expert whose wealth falls below a floor is reinitialised as a split of a rich expert;
  - each token routed to an expert is charged a flat compute cost.

  This should match or beat auxiliary-loss and loss-free routing without an auxiliary coefficient, have no dead
  experts, and show a wealth distribution that scales.
- **Arms.** Switch auxiliary loss, DeepSeek loss-free bias, expert-choice, and market. Sweep the number of experts
  64 → 16k (PEER-style tiny experts) at fixed active compute, on the soup streams plus one small text corpus.
- **Measures:**
  - loss at equal FLOPs;
  - dead and collapsed experts;
  - price convergence;
  - wealth Gini and births per step vs expert count;
  - the soup's K/Q analogue: tokens per expert per question cluster.
- **Demo.** A live "expert economy" page: the price and wealth of each expert over training, births and bankruptcies
  as events, and a Grafana panel of Gini and price variance over time.
- **What would excite him:**
  - the market arm ties or beats loss-free balancing and does better as the expert count grows;
  - the soup's K/Q law reappears as tokens-per-expert;
  - a Gini near 0.8 shows up again.
- **Kill result.** The market arm equals loss-free. That still says something: "DeepSeek already built the price;
  wealth memory adds nothing at this scale".
- **Cost.** 3-6 days, one L4 or A100, about $50-150.

## 6. Ideas in CONTEXT.md that look wrong or overstated
- **Idea 3, "backprop already pays each contributor demand × (outcome − price)".** This is true at the output layer
  only. Inside the net, `a · dR/da` is gradient × input attribution. Nobody took a position at a price and no budget
  binds. The ledger is Molchanov-style Taylor importance accumulated over time (https://arxiv.org/abs/1611.06440).
  The claim is fine as an accounting identity but should not be sold as a market.
- **Idea 6, "partial ownership automatic; big trader wealth = sum".** Conservation fails where blocks are not
  degree-1 homogeneous. The current run leaks 45-64% of R at the first cuts, and LayerNorm needs a convention that
  changes the numbers. Without a complete attribution (IG, DeepLIFT or Shapley), wealth at different scales is not
  coherent.
- **Wealth as a quality signal.** The books settle on training examples, so wealth pays memorisation (pi). Any gating
  by wealth needs prequential settlement first, or it amplifies overfitting.
- **Idea 2, "wealth = marginal information (Shapley-like)".** First-order attribution is not Shapley. It ignores
  interactions, which are exactly the synergy E2 measures.
- **Idea 1, "Kelly traders = BMA".** This holds for Kelly bettors with fixed beliefs, trading exclusive outcomes, at
  the wealth-weighted equilibrium price (Beygelzimer, Langford and Pennock). Gradient-trained units are not Kelly
  bettors, so the equivalence is a design target, not a description of the current net.
- **Idea 5, "the fixed-point pooling is new".** Wealth-weighted log-odds pooling is a logarithmic opinion pool or
  product of experts. Storkey showed that markets implement PoE or MoE depending on agent utility
  (https://arxiv.org/abs/1106.4509; isoelastic follow-up https://arxiv.org/abs/1206.6443). The new part would be the
  explicit market-price prior and P&L-set weights, not the pool itself.
- **Idea 3, "the residual stream is a sequential market".** Intermediate logit-lens readouts are uncalibrated
  (recollection of the tuned-lens paper, unverified). Early layers "read the running price" only if that price is
  calibrated. Do not claim it without a calibration check per layer.
- **"Load-balancing losses are unprincipled".** Half true. The auxiliary loss is a penalty method; loss-free
  balancing is already the dual (price) method (https://arxiv.org/abs/2408.15664). The open question is what
  long-run wealth adds on top.

## 7. Prior art to check (novelty and framing)
- Garrabrant et al., Logical Induction: https://arxiv.org/abs/1609.03543 (URL from memory, not re-fetched).
- Chen and Vaughan 2010, cost-function markets = FTRL / no-regret: https://arxiv.org/abs/1003.0034
- Beygelzimer, Langford and Pennock 2012, Kelly bettors, market price = wealth-weighted belief:
  https://arxiv.org/abs/1201.6655
- Storkey 2011, Machine Learning Markets (markets implement PoE and MoE): https://arxiv.org/abs/1106.4509. Storkey,
  Millin and Geras 2012, isoelastic agents and wealth updates: https://arxiv.org/abs/1206.6443. Hu and Storkey 2014,
  multi-period markets: https://arxiv.org/abs/1403.0648
- Lay and Barbu, artificial prediction markets for classification: https://arxiv.org/abs/1102.1465
- Balduzzi 2014, Cortical prediction markets: neurons as proper-scoring-rule agents that backpropagate incentives.
  This is the nearest prior art to idea 3 and idea 6. https://arxiv.org/abs/1401.1465. Also Kickback:
  https://arxiv.org/abs/1411.6191
- Baum's Hayek machine (economy of program agents, bucket-brigade-like credit): Baum 1999
  https://link.springer.com/article/10.1023/A:1007593124513; Baum and Durdanovic 2000
  https://pubmed.ncbi.nlm.nih.gov/11112253/. Kwee, Hutter and Schmidhuber 2001, market-based RL:
  https://arxiv.org/abs/cs/0105025
- Economy of Minds (June 2026, generalises the Hayek machine to LLM agents): https://arxiv.org/abs/2606.02859. This is
  fresh competition for the "traders as agents" framing.
- Molchanov et al., Taylor pruning criterion |a·g|: https://arxiv.org/abs/1611.06440. This is the baseline E2 must
  beat.
- McGrath et al., the Hydra effect (self-repair defeats first-order ablation estimates):
  https://arxiv.org/abs/2307.15771
- Integrated Gradients (completeness axiom): https://arxiv.org/abs/1703.01365 (URL from memory, not re-fetched).
  DeepLIFT: https://arxiv.org/abs/1704.02685 (URL from memory, not re-fetched).
- DeepSeek loss-free balancing: https://arxiv.org/abs/2408.15664. Switch Transformer:
  https://arxiv.org/abs/2101.03961. BASE layers (assignment-auction routing): https://arxiv.org/abs/2103.16716.
  Expert-choice routing: https://arxiv.org/abs/2202.09368. Mixture of a Million Experts (PEER):
  https://arxiv.org/abs/2407.04153. The last four URLs are from memory, except PEER and loss-free, which were
  checked.
- Holland's classifier systems (bucket brigade, 1985-86), the classic wealth-passing credit assignment. Recollection,
  unverified, no URL checked.

## 8. How to present results (format)
- For each experiment: a three-line TL;DR, one HTML demo page (served like the existing
  `http://[ip-removed]:8765/wealth-net/`), and Grafana panels for curves over training (wealth Gini, price, births).
  Every "current" number shows when it was checked and which file proves it.
- The lead visual for Kelvin is soup vs net on one axis of compute (FLOPs or virtual instructions), with Brier per
  family. This is the bitter-lesson curve he already knows from the soup.
