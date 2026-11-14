# Traders inside a neural network: what the study found

Kelvin and Jessica Taylor, traders-net study, 2026-09-28. Numbers come only from `study/X1/CLAIMS.md` (rendered 2026-09-28T13:51:07Z, 209 rows: 153 verified, 19 refuted, 37 unverified; check: `python3 study/X1/check_claims.py`). Paths are relative to `traders-net/study/` unless they start with `wealth-net/`. Anything not verified there is marked. D1 checks (2026-09-28T14:07Z): every number below is in CLAIMS.md or, for the few wealth-net numbers marked not X1-verified, in `wealth-net/RESULTS.md`, and every path exists (`python3 study/D1/check_ideas.py`, output `D1/check_ideas.json`); X1's checker re-run on a scratch copy passes (`D1/x1-recheck-check_claims.json`).

## TL;DR

1. **Attribution flow is not settlement:** a unit's gradient credit can have the opposite sign of what it earns as a settled trader (proved on a toy; on a real net 26% vs 3% of the most important neurons).
2. **Exact leave-one-out settlement is the only currency that ranks units well** (log score with vs without the unit; Spearman with the ablation oracle 0.417 vs 0.168 for gradient norm), but no wealth idea won a gate: no coalition win, no MoE router better than DeepSeek, and the humble layer's surviving arms beat boosting by only about 0.001 nats and never beat early stopping.
3. **Next:** traders that commit by design (block-local markets settled at their boundary) and a coherence test with disjoint-view traders; lock-in is already solved by known fixed share.

## For Jessica: attribution flow is not settlement (the sign evidence)

**One-line claim (novelty unverified):** in a network, the credit that backprop assigns to a unit (−w·∂L/∂w, the "books") is a flow of attribution, not a settled bet; settle the same unit as a bet (its exact leave-one-out log score on held-out data) and the sign flips on the units that matter most, so books cannot be wealth; only committed traders can be settled.

The evidence, all verified by X1 on fresh seeds or static re-reads:

- **Toy, closed form (M1-a1, verified).** Take a readout weight w whose true coefficient is 1. Its gradient credit C = −w·L'(w) and its ablation value V = L(0) − L(w) have opposite signs on all of (1, 2.1786): C < 0 < V, at 23 of 23 grid points. V ≥ C everywhere by convexity, and V(1) = I(X;Y) = 0.1109 nats. An over-weighted but useful unit is told to shrink, and books read that as the unit losing money. Source: `M1/lead/results/toy_summary.json`, `M1/lead/results/toy_closed_form.csv`.
- **Toy, simulated (M1-a, verified).** Redundant useful units get negative credit: −75.66 ± 1.9 (Adam) and −50.52 ± 0.1 (SGD). LOO settlement pays them +467.8 and +487.5. A pure-noise unit gets the *same* credit as the useful ones (−78.5 vs −74.5), so credit carries no usefulness signal; settlement separates them (−39.5 vs +466). Source: `M1/lead/results/toy_summary.json`; `X1/w2/results/R1_toy_fresh.json`.
- **Real net, sign census (M1-b, verified).** Wealth-net, 3 seeds, final step: among the top 5% of neurons by oracle importance (26 of 512), 0.256 ± 0.059 have negative all-time books; only 0.026 ± 0.022 have negative held-out LOO settlement (0.0 on the training stream). Source: `M1/lead/results/sign_census_summary.csv`.
- **Correction to our own lock file (M1-lock-why-b, refuted as worded).** `M1/CURRENCY.lock` says important neurons are negative "near or above the base rate" under books. Under all-time books they are negative *less* often than average (0.256 vs 0.384 for all neurons), so the book sign is weakly informative. It is not a usable signal, though: a quarter of the most important neurons still look bankrupt. The decision stands.
- **Theory (T1, verified).** The exact LMSR settlement equals the tangent (first-order) term plus a KL term, to 1.1e-14 (T1-L2.1). In-sample books pay 100% of pure-noise features; prequential books charge 97.5% of them (T1-L1.4a, T1-L1.4b). Books vs true income: R² 0.32 (SGD), 0.14 (Adam) (T1-L1.2b, T1-L1.2c). Source: `T1/w1/out/L1b.json`, `T1/w1/out/L1c.json`, `T1/w1/out/L2.json`. The planned T1 "Lemma 8" (general sign conditions) was not written; the toy is the proof we have.

FIG:toy

FIG:signstrip

FIG:census

**How it relates to prior art.** The mechanism behind the toy is old: at a trained minimum the first-order term vanishes, which is why Optimal Brain Damage prunes with second-order saliency (LeCun, Denker, Solla 1989, https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html), and Taylor pruning scores units by the *squared* first-order term (https://arxiv.org/abs/1906.10771). Loss change allocation (https://arxiv.org/abs/1909.01440) already reports that many parameters raise the loss at any given step. What we add is the reading for markets: signed gradient credit cannot serve as a trader's wealth, a settled bet can, and on a real net a quarter of the units that matter most look bankrupt under the books. L1's search ran before this finding (23:13Z, before the 00:25Z retarget) and did not look for sign-gap prior art specifically. The nearest misses in L1's search are Hedge Backprop (https://arxiv.org/abs/1711.03705), which weights layers by hedging, and continual backprop (https://arxiv.org/abs/2306.13812), which scores unit utility; neither settles units as traders. Balduzzi's cortical prediction markets (https://arxiv.org/abs/1401.1465) and Storkey's machine-learning markets (https://arxiv.org/abs/1106.4509, https://arxiv.org/abs/1206.6443) are markets of learners, not settlements of units inside one backprop net. L1 lists 45 verified sources, 40 cited IDs and 0 failed checks (`L1-1`). L1's pre-retarget one-liner is: "an LI-style market whose traders are the SGD-learned, computation-sharing units of one network, settled prequentially from a named prior through attribution, and measured for exploitability before resolution" (`L1/NOVELTY.md`). Novelty is a search result, not a proof: **unverified**.

## Answers to Q1-Q8

| Q | Answer | Key verified numbers | Evidence |
|---|---|---|---|
| Q1 Currency | Yes, but only for exact leave-one-out settlement (a settled bet), not for any book, and not enough to win a gate. | Spearman vs oracle: all-time books 0.043 ± 0.058; gradient norm 0.168 ± 0.07; LOO settlement on training positions 0.417 ± 0.034 (held-out LOO 0.576 in M3's table, an upper bound because the oracle is also held-out; not a separate X1 row). M2 neurons: settlement 0.464 ± 0.041 vs abs Taylor 0.103 ± 0.084 vs books 0.199 ± 0.075. Readouts: books 0.303 ± 0.022 lose to movement 0.814 ± 0.027. Coalitions: no win (Gate B2). Book currencies do not beat gradient norm (M3-A-stop, gate B1 false); read literally, LOO settlement beats it by 0.245 ± 0.097 on a fresh segment (3/3). | `M3/results/currency_summary.csv`; `M2/w3/results/spearman_summary.csv`; `M2/lead/gate_b.json`; `X1/lead/gates.json` |
| Q2 Novelty | The market reading of the sign gap (gradient credit is not settlement, so books cannot be wealth), plus measuring an internal market's exploitability before resolution. L1 found nothing doing the second; the first rests on known facts (OBD, Taylor, LCA; see section 2) and was not searched for. | L1-1: 45 sources, 40 cited IDs, 0 fails. Novelty itself unverified. | `L1/NOVELTY.md`, `L1/NOVELTY-FLAGS.md` |
| Q3 Induction | Exploitable, and not coherent before resolution. | Pre-resolution violation rates 0.477 / 0.545 / 0.536 vs 0.5 uninformed (M5-6). A calibration leak invades at 8.26e-4 (P4) / 1.42e-3 (P16) nats (M5-3). Frozen checkpoints get *more* exploitable with training (M5-7). Pattern traders in Kelly currency do not exploit it (3/3 negative, M5-5). | `M5/w2/results/summary.csv`; `M5/w1/SUMMARY-w1.csv`; `M5/results/headline.csv`; `M5/w3/results/summary.json` |
| Q4 Humility | No. Humble traders do not beat early stopping, weight decay or boosting beyond noise. | Survivors pass by about 0.001 nats (cg − BOOST −0.0014 ± 0.00025, M6-12). Excluding LFSR16, cg ties BOOST (M6-6). Committed heads: −0.3 / −1.3 / −0.9% BCE vs U, but clones (corr 0.9934) with wealth unrelated to marginal value (Spearman −0.0016 ± 0.0221, M6-10). | `M6/lead/M6-verdict.json`, `M6/lead/M6-stress-verdict.json`; `X1/w3/results/skeptic.json` (M6-12); `M6/w5/results/summary.csv` (M6-10) |
| Q5 MoE | A per-family wealth prior helps a little; wealth cannot replace load balancing. DeepSeek's bias already is the price. | dsfw vs DeepSeek −1.30% (E8) and −1.87% (E64) BCE, 3/3 (M7-2); worse after a shift, +0.041 to +0.092 (M7-4). Exp3 collapses (max load 0.775, M7-5); a global prior collapses fully (max load 1.00, M7-6). No balancing at all beats DeepSeek by 0.40-0.71% (M7-8). FLOP-matched dense wins at E8: 0.3083 vs 0.3314 (M7-9). | `M7/merge/arms.csv`, `M7/merge/gate.json`, `M7/w1/results/controls.csv` (M7-9) |
| Q6 Lock-in | Kelly locks in; fixed share (Herbster-Warmuth) fixes it. Gating a dense net by income hurts. | After a shift: fixed share 1e-3 0.047 ± 0.002 vs Kelly 0.69 ± 0.05 nats (post-shift learnable BCE, M3-B-fs, M3-B-kelly); lag 120 vs about 9007 steps (censored); stationary cost 0.001 (M3-B). Ungated beats income-gated 3/3 (M8-none-better); the sign of income is no gate signal (M8-sign-shuf); the whole-run lead is one family, LFSR16 (M8-onefamily). | `M3/w2/results/summary.csv`, `M3/w2/results/decision.json`; `M8/M8.md` |
| Q7 K/Q | Mostly search. The soup is a lottery over K/Q draws; the net's heads do not collapse, but K/Q shows no trend. | Lottery law R² 0.79-0.94, p=0.001 (M9a-lottery); CA110 needs K/Q ≥ 1358 for half its gain (M9a-predict); no K/Q trend in {1, 4, 16, 64} (M9c-notrend); 46.5 effective traders per question (M9c-eff). FLOPs-at-equal-Brier (M9b) is unverified. | `M9/w1/lottery_fit.csv`; `M9/w3/results/VERDICT.json`; `M9/w3/results/SUMMARY.csv` (M9c-eff) |
| Q8 Economy | They coexist only with niches; without niches one species takes everything. | Niches on (10k traders, Q64): all slots alive, Brier 0.1195 vs oracle 0.1191. Niches off: one species ≥ 99.7% of wealth. Gini 0.44 vs null 0.74. Rebalancing funds lose (−6.18 ± 1.19). Compute price c only destroys. Money conserved to 4.1e-12. The post-shift recovery-by-births claim is refuted. Live view: the Grafana `traders-net` check passes (V1). | `K1/w1/cells.csv`; `K1/w3/out/summary_N10000.csv` (K1-fof); `K1/K1.md`; `D1/w2/grafana-check/latest.json` (V1 check re-run) |

FIG:spearman

FIG:postshift

FIG:maxload

FIG:lottery

FIG:flops

FIG:k1

**Plan gates (flags as set by X1).**

| gate | flag | what it means | evidence |
|---|---|---|---|
| A | currency_locked = true | M1 locked "no bookkeeping currency for generic neurons; committed-trader settlement only" (its why(b) wording is refuted; the decision stands) | `M1/CURRENCY.lock`; `X1/lead/gates.json` |
| B1 | beats_baseline = false | no book currency beats gradient norm by 0.05 in 3/3; LOO settlement would, read literally | `M3/results/stop_rule.json`; `X1/w2/results/R2_rederive_fresh.json` |
| B2 | coalition_win = false | books tie abs Taylor and movement on random coalitions | `M2/lead/gate_b.json`; `X1/w2/results/R2_gateb_fresh.json` |
| B3 | wealth_beats_deepseek = false | a judgement: dsfw's edge replicates but is best-of-3, costs after a shift and needs family labels | `M7/merge/gate.json`; `X1/w3/results/skeptic.json` |
| B4 | base_arms_survive = true | 10 arms survive the base rule; the stress pass keeps b, cq, fb05 | `M6/lead/M6-verdict.json`; `M6/lead/M6-stress-verdict.json` |

## Insights, marked expected or surprising

- **Surprising:** books and settlement disagree in *sign*, not just in scale. The toy makes it a theorem; the census makes it a quarter of the important neurons.
- **Surprising:** books pay pure noise as much as signal. The noise unit's credit matches the useful units' credit (M1-a-noise); in-sample books pay 100% of noise features (T1-L1.4a).
- **Surprising:** training makes a net *more* exploitable, not less (M5-7). Coherence before resolution is at chance (M5-6).
- **Surprising:** removing load balancing entirely beat DeepSeek at this scale (M7-8), and a FLOP-matched dense net beat every MoE arm at E8 (M7-9).
- **Surprising:** skilled Kelly traders are *less* unequal than a beliefs-shuffled null (Gini 0.44 vs 0.74). The "rich get richer" signature we planned to show is inverted.
- **Expected:** Kelly wealth locks in after a regime shift, and fixed share fixes it at almost no stationary cost (M3-B). Textbook Herbster-Warmuth.
- **Expected:** without niches, Kelly selection yields one survivor (K1-niches-off; Blume-Easley, unverified recollection).
- **Expected:** exact LOO settlement ranks units far better than any book (Q1). It is an ablation scored in log loss, so it inherits ablation's quality; it is not cheap (one extra forward pass per unit).
- **Expected, and a warning:** one family (LFSR16) is a lottery that decides many BCE comparisons (M6-6, M8-onefamily, X1 cross-cutting finding 4). Three seeds understate variance (X1 finding: median 1.7x).

## Ranked suggestions and experiments

1. **Committed traders by construction (block-local markets).** Split the net into blocks that each post a committed log-odds correction, settle each block at its boundary, and compare with plain backprop under the controls C-random and C-wealth-only. This study did not run it (`M6/M6.md`, "Not done in this pass"), but the wealth-net lane ran a first version (3 seeds, **not X1-verified**, `wealth-net/RESULTS.md` "Block-local settlement"): with blocks trained on their own logit lens, LOO settlement agrees with the ablation oracle at Spearman +0.857 against +0.395 end-to-end; the market arm's robustness comes from skipping and rebirth, which a random-skip control matches, not from choosing units by settlement; and the negative price of anarchy (−0.183) measures regularisation against memorised unpredictable families, not a gain on learnable ones. Next: 5+ seeds, streams free of the pi-offset leak, a benchmark without unpredictable families, and early stopping for the end-to-end arm.
2. **A logical-induction coherence test with traders that see disjoint views.** M5 found no coherence before resolution; but traders that share one trunk cannot disagree much. Give traders disjoint inputs, clear with a fixed point (M6's Newton converged in about 2 iterations, M6-9 unverified), and measure A∨¬A and (A∧B)→A violations against the 0.5 baseline.
3. **The sign lemma as a short note for Jessica.** Write the general condition (T1's missing Lemma 8): when a unit's signal is shared by others, first-order credit is negative while leave-one-out value is positive. Add the census as the empirical half. This is the cleanest publishable piece.
4. **MoE at scale with a per-family wealth prior plus fixed share.** dsfw wins 1.3-1.9% in-distribution but loses after a shift; fixed share is the known fix for exactly that. One arm, 3 → 5 seeds, E64+.
5. **Robustness hygiene before any new claim.** 5 seeds by default (X1: 3-seed sd understates by a median 1.7x), report results with and without LFSR16, and use leak-free streams.

## What is not verified or was refuted

- Unverified: novelty itself (L1 is a search); M1-c (three rules conserve, ranks = cp); M5-1 probe-exploitability magnitudes; M6-9 (fixed point not kept, Newton 2.05 iterations); M8-stop; M9b-flops (Figure 7); T1 Lemmas 8-9 (not written); everything from `wealth-net/RESULTS.md` block-local section (not in CLAIMS).
- Refuted: CURRENCY.lock why(b) wording; M2 neuron stop-rule verdict "win" (books); M6-3; M9c-thin magnitude; K1-shift (recovery by births); S0 delay-ordering numbers and "pi ≈ ln 2 for any seed" (pi-offset leak at seed 4).
- Recollections (unverified): Hayek's price system, Blume-Easley market selection, Cover's universal portfolio and Herbster-Warmuth fixed share are cited from memory or metadata only.
- Cost: GPU spend for the whole study $6.07 of $45 (`G1/COST.md`; ledger equals GCE receipts, X1 row G1.2). D1 used no GPU.
- Page demos not built: the wealth treemap (the sign strip in section 2 is the in-sample/prequential toggle instead) and humble-layer price trajectories (M6 committed no per-step price data). Figure 7 (M9b, soup vs net on one compute axis) is shown but unverified.

## Sources

- Optimal Brain Damage (LeCun, Denker, Solla 1989): https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
- Taylor importance for pruning (Molchanov et al. 2019): https://arxiv.org/abs/1906.10771
- Hedge Backpropagation: https://arxiv.org/abs/1711.03705
- Continual backprop / loss of plasticity: https://arxiv.org/abs/2306.13812
- Balduzzi, cortical prediction markets: https://arxiv.org/abs/1401.1465
- Storkey, machine learning markets: https://arxiv.org/abs/1106.4509
- Storkey et al., isoelastic agents and wealth updates: https://arxiv.org/abs/1206.6443
- Logical induction: https://arxiv.org/abs/1609.03543
- Chen and Vaughan, market makers and no-regret learning: https://arxiv.org/abs/1003.0034
- Abernethy, Chen, Vaughan, efficient market making via convex optimization: https://arxiv.org/abs/1201.6655
- Gated linear networks: https://arxiv.org/abs/1910.01526
- DeepSeek auxiliary-loss-free load balancing: https://arxiv.org/abs/2408.15664
- Movement pruning: https://arxiv.org/abs/2005.07683
- Loss change allocation: https://arxiv.org/abs/1909.01440
- Economy of Minds: https://arxiv.org/abs/2606.02859
- MoB: https://arxiv.org/abs/2512.10969
