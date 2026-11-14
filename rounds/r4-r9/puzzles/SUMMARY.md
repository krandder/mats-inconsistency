# Trader-economy puzzles: what we learned

For Kelvin. Sources, read 2026-09-29:
- `CATALOGUE-PUZZLES.md` entries P0-P7 and P12. P0-P6 were each checked by an independent verifier. P7 and P12 are as their runners reported them and were not re-verified in this batch.
- The P0-P6 verifier reports.

Spend: P0-P6 cost about $20.72. GPU budget left for new runs: none, as verified during this batch's review.

## Bottom line

Seven experiments (P0-P6), zero gains. No market mechanism beat the plain baseline on the baseline's own terms.

| Outcome | Experiments |
|---|---|
| Informative | P0 (the ruler) and P7 (local latent settlement) |
| No gain | P1, P3, P4 |
| Failed their own validity checks | P2, P5, P6 |

The strongest positive result in the whole programme came from outside the market. In P12, learned local costs plus an exact solver solved 27% of wide mazes exactly, where DT solved about 3%.

## Pricing computation (when to stop thinking, who gets credit)

The baseline already does this.
- DT-Recall with progressive loss solves prefix sums at 0.9998.
- One-step credit does just as well at that ceiling (0.9948, p = 1.0).
- Simple stopping rules land within 1-2% of the oracle: the KL-change rule, or a one-step gate.

So a compute market can earn almost nothing by deciding when to stop. P0 measured 1-5% headroom. In P1, multi-horizon value heads saved 1.1-1.6% of steps (upper CI 2.6%) against a pre-registered bar of 20%.

The depth axis also has a structural problem. Per-step market pay with a detached prior has exactly the gradient of deep supervision, so "a market over depth" is a relabel of what the baseline already does.

Two open ends remain, and neither is a finding:
- The market-resnet pilot gain (arm B) was neither explained away nor confirmed, because the P0 control used 3x the compute.
- Maze credit horizon is untested, because every maze arm was under-trained. Mazes also need width 128, not 64.

## Joint structure

P2 is the only test, and it failed as an experiment before it failed as an idea:
- Its toy check missed the bar before the GPU run started.
- The exactly priced quadtree market then lost to hard-EM labels by 18 points [â35, â2].
- Its latent carried about a third of the route information.

Theory predicted this:
- A joint LMSR is #P-hard unless it is structured.
- XOR-type structure is invisible to prices (Feigenbaum et al.).

Exactly priced joint markets are a real gap in the literature, but we have no evidence that they train.

## Latent variables

Pay rules do not discover factors.
- **P3.** Shapley-type pay found the natural factors 3/20 times, the same as additive pay, and collapsed to constant codes. Locatello et al. predict this (no neutral mechanism finds factors), and Wolpert-Tumer predict it too (Shapley pay is not aligned with the goal).
- **P4.** Aumann-Shapley is a decent but not the best way to rank latents. Sampled Shapley at the same cost ranks better (0.981 vs 0.924), and a market on the top latent did not help.
- **P7, the one positive.** Likelihood-message settlement trains depth-4 latent chains without backprop. However, only the exact forward-backward form learns the hard target. That makes it a local implementation of exact inference, not a new learner.

## Memory inside a network

P5 never reached the market. The backbone could not learn maze path parity beyond depth 1 (0.53 against a 0.90 bar), partly because the curriculum made paths longer instead of shorter. We learned nothing about memory markets. We learned that our task and curriculum were not ready.

## Entry and exit

In P6, bankruptcy plus cloning did not speed learning on the one metric measured. It took 125 more iterations to reach the cap, 95% [â50, 375]. The run is unfinished: there are no ladder evals and no RESULTS.md.

## Process, harshly

- **Three runs spent GPU money despite a failed gate.** P1's capacity gate failed; P2's toy check failed before the GPU run; P5's backbone gate failed. Gates must stop runs automatically.
- **Runners overclaimed, and the verifiers caught it.** The cases:
  - a threshold picked on test (P0);
  - a loss-artefact claim built on unequal compute (P0);
  - an undisclosed failed gate (P1);
  - a stale number (P2);
  - an omitted seed (P6).

  Verification paid for itself; keep it mandatory.
- **Reference arms were broken twice.** DT-L scored 0/2 in P0 and exact 0.0 in P6. A broken baseline makes any "win" worthless.

## What next

**Most promising moonshot: a breadth-by-depth compute market.**
- **The case for it.**
  - Every depth-axis test says the stopping decision is nearly solved.
  - Breadth has documented headroom: EqR takes Sudoku-Extreme from 93.0% to 99.8% with 128 restarts.
  - Nobody prices the split between more restarts and more depth.
  - Candidates are where prices come naturally. Each restart's final residual or validity can be checked, which is P12's lesson: pay for local, checkable signals and clear them with an exact rule.
- **The test.** On Sudoku-Extreme, then mazes, traders buy restarts or extra depth at equal compute (NFE).
- **The kill rule.** Stop if the market does not beat EqR's smallest-residual selection and majority vote at equal NFE.
- **Honest odds: low.** Those selection rules are strong and cheap.

**Cheaper first:**
- Finish P6 (about $4.5).
- Rerun the maze credit test at width 128 with full training.

**Decision needed from Kelvin:** a GPU budget. None is left, so nothing here can run without one.

## Verified literature

### Value of computation and adaptive depth
- Russell & Wefald 1991, doi:10.1016/0004-3702(91)90015-C
- Hansen & Zilberstein 2001, doi:10.1016/S0004-3702(00)00068-0
- Callaway et al. 2018, arXiv:1711.06892
- Graves 2016, ACT, arXiv:1603.08983
- Banino et al. 2021, PonderNet, arXiv:2107.05407
- Bai, Kolter & Koltun 2019, DEQ, arXiv:1909.01377
- Schwarzschild et al. 2021, arXiv:2106.04537
- Bansal et al. 2022, arXiv:2202.05826
- Anil et al. 2022, arXiv:2211.09961
- Raposo et al. 2024, MoD, arXiv:2404.02258
- Schuster et al. 2022, CALM, arXiv:2207.07061
- Geiping et al. 2025, Huginn, arXiv:2502.05171
- Wang et al. 2025, HRM, arXiv:2506.21734
- Jolicoeur-Martineau 2025, TRM, arXiv:2510.04871
- ARC Prize 2025, HRM analysis, https://arcprize.org/blog/hrm-analysis

### Prediction markets and scoring rules
- Feigenbaum, Fortnow, Pennock & Sami 2003/2005, https://lance.fortnow.com/papers/files/infomarkets.pdf
- Chen, Ruberry & Wortman Vaughan 2012, arXiv:1210.4837
- Chen et al. 2010, Algorithmica 58, https://hanson.gmu.edu/bluffing.pdf
- Hanson 2002/2007, https://mason.gmu.edu/~rhanson/mktscore.pdf
- Abernethy, Chen & Wortman Vaughan 2013, ACM TEAC 1(2)
- Chen, Fortnow, Lambert, Pennock & Wortman 2008, arXiv:0802.1362
- Abernethy, Kutty, Lahaie & Sami 2014, arXiv:1402.5458
- Beygelzimer, Langford & Pennock 2012, arXiv:1201.6655
- Garrabrant et al. 2016, arXiv:1609.03543
- Srinivasan, Karger & Chen 2023, arXiv:2306.04305
- Abernethy & Frongillo 2011, NIPS 24
- Chen, Kash, Ruberry & Shnayder 2011, WINE
- Lambert, Pennock & Shoham 2008, EC
- Augenblick & Rabin 2021, QJE 136(2):933-985

### Latent variables and factors
- Saxe et al. 2018, ICLR
- Xu et al. 2020, arXiv:2002.10689
- Hewitt et al. 2021, arXiv:2109.09234
- Voita & Titov 2020, EMNLP, aclanthology 2020.emnlp-main.14
- Shalizi & Crutchfield 2001, arXiv:cond-mat/9907176
- Subramanian et al. 2022, JMLR 23(12)
- Ni et al. 2024, arXiv:2401.08898
- Locatello et al. 2019, arXiv:1811.12359
- Geiger et al. 2022, PMLR 162:7324-7338
- Makelov et al. 2023, arXiv:2311.17030
- Pennock & Xia 2011, arXiv:1202.3756
- van den Oord et al. 2017, VQ-VAE, NeurIPS
- Knutson et al. 2024, arXiv:2410.03020
- Taufeeque et al. 2024, arXiv:2407.15421

### Algorithmic reasoning benchmarks
- Schwarzschild et al. 2021, datasets, arXiv:2108.06011
- Ivanitskiy et al. 2023, maze-dataset, arXiv:2309.10498
- Palm et al. 2018, RRN, arXiv:1711.08028
- Dehghani et al. 2019, Universal Transformers, arXiv:1807.03819
- Fan et al. 2025, arXiv:2409.15647
- VeliÄkoviÄ et al. 2022, CLRS, arXiv:2205.15659
- Rodionov & Prokhorenkova 2023, arXiv:2306.13411
- Huang, Geng & Kolter 2026, EqR, arXiv:2605.21488

### Economic credit assignment
- Wilson 1995, XCS, doi:10.1162/evco.1995.3.2.149
- Schmidhuber 1989, neural bucket brigade
- Baum 1999, Machine Learning 35:155-185
- Kwee, Hutter & Schmidhuber 2001, arXiv:cs/0105025
- Chang et al. 2020, arXiv:2007.02382
- Pallavi Sudhir & Tran-Thanh 2025, arXiv:2503.05828
- Balduzzi 2014, arXiv:1401.1465
- Wolpert & Tumer 2001, doi:10.1142/S0219525901000188
- Foerster et al. 2018, COMA, arXiv:1705.08926
- Wang et al. 2021, InfoPro, arXiv:2101.10832
- Nøkland & Eidnes 2019, arXiv:1901.06656
- Huang et al. 2018, BoostResNet, PMLR 80:2058-2067
