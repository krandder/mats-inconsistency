# NOVELTY (L1): what is new in "traders inside a neural network"

L1 lead (synthesizer), 2026-09-27 ~23:20Z. Sources: the three cluster reports `A/CLUSTER-A.md`,
`B/CLUSTER-B.md`, `C/CLUSTER-C.md`, plus a lead gap search (`lead/verify-lead.json`,
`lead/quotes-lead.txt`). Every arXiv ID cited here was fetched and its title, authors and date were checked
against its abs page: `A/verified-ids.csv`, `B/verify-B.json`, `C/verify-C.json`, `lead/verify-lead.json`.
`python3 check_novelty.py` re-checks that every cited ID is in one of those files and that every quoted paper
number appears verbatim in a quotes file (result: `lead/check_novelty.json`). Literature only; no runs, so no
PREREG and no seeds. "Ideas 1-7" are the numbered ideas in `../../CONTEXT.md`.

## The one-line claim for Jessica

> Markets of *fixed* traders are Bayes/Hedge (Beygelzimer-Langford-Pennock, Storkey), "backprop is a market" is
> an old analogy (Balduzzi, Wentworth, Sudhir), per-unit credit is movement/Taylor/LCA, and MoE load balancing is
> already a price (DeepSeek); what we did not find is **a logical-induction-style market whose traders are the
> SGD-learned, computation-sharing units of one network, whose wealth is settled prequentially from a named prior
> through attribution (so traders need not be standalone predictors), and which is then measured for
> exploitability by a fixed trader class before resolution.**

Status: **unverified novelty** (no counter-example found in the sources listed above; not a
systematic search). The parts that survive are the *combination* and the *LI test*, not any single
mechanism. The two nearest misses, both found by the lead's gap search and not in the plan:
- Hedge Backpropagation (Sahoo et al. 2017, https://arxiv.org/abs/1711.03705) already keeps multiplicative,
  prequential Hedge weights, with a floor, over 19 classifiers that share one SGD-trained trunk
  (`lead/quotes-lead.txt`, "1711.03705" blocks). The delta: its traders are whole-depth predictors with their own loss,
  so no attribution is needed. We price partial contributors (neurons, heads) that only have wealth through the books.
- Continual backprop (Dohare et al., https://arxiv.org/abs/2306.13812) keeps an EMA "contribution utility"
  |h|·Σ|w_out| per unit and reinitialises the lowest-utility units (`lead/quotes-lead.txt`, "2306.13812" block).
  That is M8's wealth-gated rebirth with an unsigned, unsettled currency. The delta is the signed, settled currency.

## Claim → nearest prior art → delta

| # | our claim (CONTEXT idea / plan Q) | nearest prior art (verified) | delta that survives | verdict |
|---|---|---|---|---|
| 1 | Kelly traders with P&L wealth = Bayesian model averaging; LMSR = no-regret (idea 1) | BLP 2012 https://arxiv.org/abs/1201.6655 (price = wealth-weighted belief, Bayes update, log regret); Chen-Vaughan 2010 https://arxiv.org/abs/1003.0034 (cost-function market = FTRL, LMSR = Hedge); Storkey 2011/2012 https://arxiv.org/abs/1106.4509 https://arxiv.org/abs/1206.6443 | none | **known**; cite, do not claim |
| 2 | Traders emit a confidence-sized correction to the market's log-odds; "only improve here and there" (idea 2, Q4) | fractional Kelly = belief mixed toward the market (BLP §6-7, C/CLUSTER-C.md); GLN https://arxiv.org/abs/1910.01526 (every unit corrects a base prediction by geometric mixing, clipped, local log loss); Abernethy-Frongillo https://arxiv.org/abs/1111.2664 (paid by improvement on a published hypothesis; abs only) | wealth-weighted rather than OGD-weighted corrections, plus fixed-point clearing inside a trained layer | **mostly known**; the humble layer is fractional Kelly + GLN |
| 3 | softmax = LMSR over exclusive outcomes; CE gradient p−y = settlement; residual stream = sequential market (idea 3) | Chen-Vaughan 2010 (LMSR = softmax of shares); Balduzzi 2014 https://arxiv.org/abs/1401.1465 (neurons as scoring-rule traders); Sudhir-Tran-Thanh 2025 https://arxiv.org/abs/2503.05828 Thm 3.1 (a ReLU MLP forward pass is a deep market); Wentworth 2018 (LessWrong, A/CLUSTER-A.md); Mouzouni 2026 https://arxiv.org/abs/2604.04230 (softmax routing = congestion equilibrium); Moakhar et al. 2023 https://arxiv.org/abs/2310.09952 (self-interested neurons whose optimal strategy matches backprop); Venkatasubramanian 2024 https://arxiv.org/abs/2405.10955 (neurons compete for utility; "arbitrage equilibrium" in trained nets; abs only) | none as framing | **known framing** |
| 4 | unit = anything with an additively separable logit contribution (idea 4) | conductance https://arxiv.org/abs/1805.12233 (layer completeness); AttnLRP https://arxiv.org/abs/2402.05602; PEER https://arxiv.org/abs/2407.04153 (about a million single-neuron experts) | none | **known** |
| 5 | confidence-weighted pool with a fixed point (idea 5, M6 g) | LI MarketMaker (Brouwer fixed point) https://arxiv.org/abs/1609.03543; Barbu-Lay https://arxiv.org/abs/1102.1465 (equilibrium price of price-dependent betting functions, solved numerically); Storkey pools; DEQ https://arxiv.org/abs/1909.01377 (abs only) | the fixed point with SGD-learned traders *inside* a network, trained through implicit gradients | **open** (combination) |
| 6 | MoE load balancing as a market; wealth replaces the aux loss (idea 5, Q5, M7) | DeepSeek ALF-LB https://arxiv.org/abs/2408.15664 (per-expert load price by tatonnement); Han-Zhong https://arxiv.org/abs/2512.03915 (the bias is a dual price; needs a step-size grid); Routing-Free MoE https://arxiv.org/abs/2604.00801 (adaptive global λ); MoB https://arxiv.org/abs/2512.10969 (VCG auction, 4 experts, label-using bids, no wealth); Chang 2020 https://arxiv.org/abs/2007.02382 (cloned Vickrey society) | pricing *quality* (P&L) rather than load; no paper here routes by accumulated expert wealth | **open**, and the bar is high: beat DeepSeek at its best step size |
| 7 | fractal books: backprop is the accounting of a firm; any cut conserves; big trader = sum (idea 6) | Kickback https://arxiv.org/abs/1411.6191 (backprop = local no-regret learners); conductance and AttnLRP (cut conservation); LCA https://arxiv.org/abs/1909.01440 (per-parameter loss change over training, conserved, summable); movement pruning https://arxiv.org/abs/2005.07683 (S = −Σ_t g·W: cumulative in-sample books); Taylor https://arxiv.org/abs/1906.10771; Hayek/Kwee https://arxiv.org/abs/cs/0105025 (parent owns a 0.25 copyright share of a child) | cut conservation *and* time accumulation per unit, settled prequentially, validated against exact single and coalition ablations (M1/M2) | **partly new** (measurement, not mechanism) |
| 8 | exact LMSR settlement from a named prior = log-score gain (T1 lemma 2) | Integrated Gradients https://arxiv.org/abs/1703.01365 (completeness from a baseline, Aumann-Shapley); Chen-Vaughan 2010 | none; T1 already tags it "known (2.1)" (`../T1/lead/HEADER.md`) | **known** |
| 9 | wealth ≈ marginal information (Shapley-like); removing negative-wealth units repairs the net (ideas 2, 7) | Neuron Shapley https://arxiv.org/abs/2002.09815 (removing negative-Shapley filters repairs bias); Hydra effect https://arxiv.org/abs/2307.15771 (self-repair: direct ≠ total effect) | a cheap, online, cumulative estimate vs an expensive, static, held-out one | **partly new** (cost-accuracy trade-off only) |
| 10 | gate execution by wealth, with a floor, exit price and rebirth (idea 6, Q6, M8) | soft movement pruning (2005.07683); continual backprop https://arxiv.org/abs/2306.13812 and its precursor https://arxiv.org/abs/2108.06325 (EMA utility, reinit lowest); RigL https://arxiv.org/abs/1911.11134 (drop by magnitude, regrow by gradient); HBP 1711.03705 (Hedge weights with floor s); Hayek, Kwee, Economy of Minds https://arxiv.org/abs/2606.02859 (rent, bankruptcy, births) | a signed, prequentially settled currency; units that are partial contributors | **open, narrow** |
| 11 | a net of traders is (un)exploitable by a fixed class; coherent before resolution on delayed and deductive questions (Q3, M5) | LI criterion https://arxiv.org/abs/1609.03543; online multivalid learning https://arxiv.org/abs/2101.01739 (calibration conditional on every group, polynomial time); hub `consistency-ft.md` Exploit LP (static single-check arbitrage) | E_C(t) of a finite computable class against an SGD-learned market with delayed and deductive settlement | **new as a measurement**; the finite, computable shadow of the LI criterion |
| 12 | thousands of Kelly traders coexist or collapse (Q8, K1) | Blume-Easley 2006 https://doi.org/10.1111/j.1468-0262.2006.00691.x (complete markets select the most accurate; full text not read); BLP 2012; Kwee 2001 ("making Hayek stable still seems to remain an art", `lead/quotes-lead.txt`) | only incompleteness (niches) with a compute price and learning traders | **known for fixed beliefs**; K1 is a demo unless niches or learning change the outcome |
| 13 | fund of funds with fixed fractions = flat mixture; only rebalancing adds power (T1 lemma 6) | Cover universal portfolios (constant rebalancing; recollection, unverified); T1 tags 6.1-6.3 theorem (`../T1/lead/HEADER.md`) | none beyond T1's own proofs | **known in spirit** |
| 14 | K/Q collapse transfers from the soup to a net (Q7, M9) | lottery ticket https://arxiv.org/abs/1803.03635 (magnitude, reset to θ0) | no prior art found for a K/Q law in gradient markets; not searched beyond cluster B | **open** (unsearched) |
| 15 | censored feedback for unexecuted units (M3/M8/M7) | SparseMixer https://arxiv.org/abs/2310.00811 (router-gradient estimator); T1 lemma 7 (Exp3 shadow credit) | none; use it as a control | **known problem** |

## Numbers quoted from papers (each has a path)

- Balduzzi 2014: turning feedback plasticity off takes the tracker from 98% to 95% and the foveator from 95% to
  93% (`lead/quotes-lead.txt`, "1401.1465").
- Kwee 2001: birth when wealth exceeds 1000; copyright share 0.25 (`lead/quotes-lead.txt`, "cs/0105025").
- Barbu-Lay 2012: CT lymph-node detection from 79.6% to 81.2% at 3 false positives per volume
  (`lead/quotes-lead.txt`, "1102.1465 abs meta").
- Storkey 2012: 20 tree agents; wealth adapted "on the complete training set with 1 training epoch"
  (in-sample settlement) (`lead/quotes-lead.txt`, "1206.6443").
- Economy of Minds 2026: mathematical reasoning from 15.9% to 57.0% and financial research from 45.0% to 60.0%;
  6 founders in the initial population (`lead/quotes-lead.txt`, "2606.02859"). **Correction to
  `A/CLUSTER-A.md`:** it wrote "MATH 45.0% -> 60.0% vs ReAct"; 45.0→60.0 is Finance-Agent-Bench, and MATH is
  15.9→57.0.
- Hedge Backpropagation 2017: 19 classifiers on 19 hidden layers, β = 0.99, floor s = 0.2
  (`lead/quotes-lead.txt`, "1711.03705").
- Continual backprop: ImageNet binary accuracy from 89% on an early task to 77% on the 2000th task without it
  (`lead/quotes-lead.txt`, "2306.13812 abs meta").

## What this means for the study (for D1 and X1)

1. **Say "known" early.** Ideas 1, 3, 4 and 8 are prior art and should be cited in `IDEAS-NET.md`, not
   presented as insights. Idea 2's humble trader is fractional Kelly plus GLN.
2. **The claim rests on Q1 and Q3.** If M2 ties prequential books with Taylor/movement, the "wealth" part reduces to
   movement pruning with a new settlement rule (row 7). The LI measurement (row 11) then carries the novelty alone.
3. **Premise check.** "A convex pool cannot add independent evidence" is consistent with Storkey 2011 (the
   exponential-utility pool is a *normalised* geometric mean); GLN's unnormalised geometric mixing is the known
   fix (A/CLUSTER-A.md, item 3 and item 10).
4. **Arms that prior art already ran** are listed in `NOVELTY-FLAGS.md`, with the action each owner should take.

## Unverified items (policy 15)

- Wentworth 2018 (LessWrong) was found by worker 1 and not re-read by the lead.
- Blume-Easley 2006: metadata via Crossref; full text not read. The Massari 2013 comment's content is from a search summary.
- Hayek (Baum 1999, Baum-Durdanovic 2000): metadata via Crossref; full texts not read.
- Cover universal portfolios, Herbster-Warmuth fixed share (the M3/M8 floor α), OBD/skeletonization: recollections, unverified.
- 1111.2664, 1403.0648, 2108.06325, 1911.11134, 2405.10955, 2310.09952, 1909.01377: abs page only.
- Scope: 45 verified sources across three clusters and one lead gap search (`lead/check_novelty.json`, field verified_sources). This was not a
  systematic review. A negative ("not found") is weak evidence.
