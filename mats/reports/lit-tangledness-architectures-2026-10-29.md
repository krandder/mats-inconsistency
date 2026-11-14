# Can dense networks be interpreted, and is their tangledness needed? Literature check (29 October 2026)

For Kelvin (brief from main). **P** proven, **M** measured, **C** conjecture or argument. Papers verified on arXiv or
their sites; I read every abstract, plus any § or Table named after a link. GPT-6 fact-check applied
(`work/lit-tangledness/review-gpt6-2026-10-29.md`); nothing rerun.

## Answer summary

**1. General arguments.** Worst case, proven by adversarial or cryptographic constructions: finding a violating input
of a ReLU net is NP-complete, counting such inputs #P-complete, many circuit-discovery queries NP-hard and
inapproximable, and some backdoors escape every efficient test even given the weights. None concerns trained nets on
natural data. For random nets the sharp form of your argument is dense *non-Gaussian* dependence, not treewidth:
Gaussian dependence is cheap at any density (means, covariances, conditioning), and finite-width theory ties the rest
to depth/width L/n. ARC conjecture their estimator's mean squared error scales as (L/n)^K; our census found no
compression at 256 x 32. Expected region counts are exponential in input dimension, not depth. On trained nets the
best dictionaries and attribution graphs leave much unexplained. Nothing proves that trained nets on structured data
lack compact descriptions.

**2. Needed, or easier?** Never shown necessary; repeatedly shown to buy capacity and ease learning. For some tasks
superposition provably computes m features with about sqrt(m) neurons. In practice good sparse subnetworks are found
through dense training, and tractable circuits plateau as they grow until a transformer supervises their latents.
Weight sparsity is not untangling (sparse random graphs have linear treewidth). Compositional sparsity, a polynomial
bounded-fan-in DAG, is what every efficient computation has, so it bounds neither treewidth nor estimation cost. On
synthetic tree-generated data, dense nets that learn features learn efficiently and approximate exact tree inference.

**3. By-design architectures.** Costs run from none (SoLU to about 50B parameters; mixture-of-experts) to 100-1000x
compute (weight-sparse). Their interpretability is per neuron, expert or weight; none gives M1/M3's object, exact or
certified expectations and tail probabilities under an input law. Tractable probabilistic models give exact marginals
and moments, not arbitrary tails, and lag transformers. Codebook features (chosen activations are sums of k of C
codes) come nearest our gate census.

**4. Scale-free, formally.** Fix a hierarchy T over the inputs. Each node covering s inputs sends the rest C(s) bits
(finite under activation noise); the task's one-way communication complexity across that cut, at tolerance epsilon,
bounds C(s) below. Untangled at every scale: C(s) grows sublinearly in s; C(s) proportional to s, a volume law,
compresses nothing. Exact estimation also needs low treewidth (the only graph-only route for arbitrary potentials;
Gaussian parts excepted), messages closed under the layer maps, and an input law factoring along T, perhaps given a
small latent. Evidence is thin: 1-bit fusion trees often keep the optimal error exponent; fixed-state sequence models
lose mainly on recall; hierarchies dense within levels (Hourglass, H-Net) match flat transformers. I found no strictly
low-treewidth model at transformer loss on a natural task, perhaps because natural structure is input-dependent (parse
trees; which token belongs to which source) and dense attention largely pays for routing, as H-Net's learned chunking
and our World FA binding failure suggest.

**5. Proposal** (last section). On the PFN prior Bayes adds per-source log-likelihoods of the answer; pit a tree net
with messages sized by that structure against the dense transformer, then estimate its deviation from Bayes
deductively.

## Key papers

**A. Arguments against interpretability in general (Q1)**

| Paper | Claim (status first) | Relevance |
|---|---|---|
| [Katz+ 2017](https://arxiv.org/abs/1702.01135) (appendix); [Marzari+ 2023](https://arxiv.org/abs/2301.07068) | **P.** Finding a violating input is NP-complete; counting them (discretized) #P-complete | worst case |
| [Adolfi+ 2024](https://arxiv.org/abs/2410.08025); [Barceló+ 2020](https://arxiv.org/abs/2010.12265) | **P.** Many circuit-discovery queries NP-hard, inapproximable; explanations harder for MLPs than trees | worst case |
| [Goldwasser+ 2022](https://arxiv.org/abs/2204.06974) | **P.** Backdoors no efficient test detects, even with weights and data (random ReLU nets; lattice assumption) | no audit |
| [Hanin & Nica 2019](https://arxiv.org/abs/1909.05989); [Roberts, Yaida, Hanin 2021](https://arxiv.org/abs/2106.10165); [Hanin 2022](https://arxiv.org/abs/2204.01058) | **P** (book: physics rigor). Over random weights, L/n sets non-Gaussianity; 1/n^k terms often grow like L^k | ensembles, not one net |
| [Wu+ (ARC) 2026](https://arxiv.org/abs/2605.05179) (§5, §7.1, App. D) | **P, M; C; C.** Cumulant propagation beats sampling on wide nets; MSE at most c_K (L/n)^K; given the training process, mechanistic estimation can match any estimator | M1 reference |
| [Montúfar+ 2014](https://arxiv.org/abs/1402.1869); [Hanin & Rolnick 2019](https://arxiv.org/abs/1906.00904) | **P; P, M.** Regions: up to exponential in depth; expected count at initialization at most about (neurons)^(input dimension), tight in training | the exponential |
| [Gao+ 2024](https://arxiv.org/abs/2406.04093) (§4.1); [Lindsey+ 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html); [Engels+ 2024](https://arxiv.org/abs/2410.14670) | **M.** 16M-latent SAE spliced into GPT-4: loss of a model with 10% of its compute; attribution graphs satisfy on about 1/4 of prompts; larger SAEs fail on the same contexts | state of the art |
| Ours (gate-entropy, boundary-family, random-net-phases, arc-noisy-deep, cm-interp-phaseB) | **M.** Random 256 x 32: all gates relevant, entropy proportional to width, no merging found, factored families fail. PFN: named features explain 6-16% of per-episode deviation, maps-based descriptions 85-92% | our evidence |

**B. Is tangledness needed or helpful? (Q2)**

| Paper | Claim (status first) | Relevance |
|---|---|---|
| [Elhage+ 2022](https://arxiv.org/abs/2209.10652); [Adler & Shavit 2024](https://arxiv.org/abs/2409.15318) | **M; P.** Superposition of sparse features (toy); for pairwise logic, m outputs need and can use about sqrt(m) neurons, up to logs | tangling buys capacity |
| [Safran & Shamir 2018](https://arxiv.org/abs/1712.08968) | **P; M.** Spurious minima common in small ReLU regressions (6-20 units); mild over-parameterization sharply reduces them | width helps |
| [Chizat+ 2019](https://arxiv.org/abs/1812.07956); [Yang & Hu 2021](https://arxiv.org/abs/2011.14522) | **P, M.** Lazy (kernel) training does worse on vision; standard parametrizations lose feature learning at infinite width | features, not density |
| [Frankle & Carbin 2019](https://arxiv.org/abs/1803.03635); [Frankle+ 2020](https://arxiv.org/abs/1912.05671); [Frankle+ 2021](https://arxiv.org/abs/2009.08576); [Malach+ 2020](https://arxiv.org/abs/2002.00585) | **P; M.** Good sparse subnetworks exist (provably, in wide random nets); in practice found by dense training; pruning at initialization no better than random per layer | existence, not findability |
| [Evci+ 2019](https://arxiv.org/abs/1906.10732); [Evci+ 2022](https://arxiv.org/abs/2010.03533); [RigL](https://arxiv.org/abs/1911.11134) | **M.** Sparse training from scratch does worse; connecting paths found only through dense space; topology updates close the gap | density eases search |
| [SparseGPT](https://arxiv.org/abs/2301.00774); [Frantar+ 2023](https://arxiv.org/abs/2309.08520); [Gao 2009](https://arxiv.org/abs/0907.5481) | **M; M; P.** 60% of OPT-175B prunable in one shot; optimal sparsity rises with data; Erdős–Rényi graphs above 1.073 edges per vertex have linear treewidth | sparse ≠ untangled |
| [Poggio+ 2017](https://arxiv.org/abs/1611.00740); [Poggio 2025](https://arxiv.org/abs/2510.11942) | **P.** Deep nets avoid the curse on compositional functions; efficient computability gives polynomial bounded-fan-in DAGs | fan-in, not treewidth |
| [Lin, Tegmark, Rolnick 2017](https://arxiv.org/abs/1608.08225); [Mossel 2016](https://arxiv.org/abs/1612.09057) | **C; P.** Hierarchical generative processes favour depth; deep algorithms needed for one tree family | data side |
| [Cagnetta+ 2024](https://arxiv.org/abs/2307.02129) (§II); [Garnier-Brun+ 2024](https://arxiv.org/abs/2408.15138); [Allen-Zhu & Li 2023](https://arxiv.org/abs/2305.13673) | **M.** Deep nets learn a random hierarchy model from polynomially many samples, lazy and shallow nets do not; transformers approximate exact tree inference scale by scale, and CFG dynamic programming | dense nets find trees |
| [Filan+ 2021](https://arxiv.org/abs/2103.03386); [Csordás+ 2021](https://arxiv.org/abs/2010.02066); [Mittal+ 2022](https://arxiv.org/abs/2206.02713) | **M.** Trained nets more clusterable than random but do not reuse modules; learned modular routing is suboptimal | partial modularity |

**C. Interpretable by design (Q3).** Exact? = tractable exact expectations or tail probabilities under an input law.

| Architecture | Capability cost | Scale | Exact? |
|---|---|---|---|
| [Weight-sparse](https://arxiv.org/abs/2511.13653) (§1, §5) | Sparsest 1 in 1,000 weights; 100-1000x compute for equal capability | tens of M nonzero | no |
| [BIMT](https://arxiv.org/abs/2305.08746) | Toy tasks, MNIST | tiny | no |
| [KAN](https://arxiv.org/abs/2404.19756); [fair comparison](https://arxiv.org/abs/2407.16674) | Ahead only on symbolic fitting at equal parameters and FLOPs | small | no |
| [Backpack](https://arxiv.org/abs/2305.16765) | 170M matches GPT-2's 124M | 170M | no |
| [Codebook features](https://arxiv.org/abs/2310.17230) (Table 2) | Pythia-410M, attention codes, k = 8: loss 2.74 vs 2.41, the 160M model's level | 410M | no; discrete codes |
| [CRATE](https://arxiv.org/abs/2306.01129); [CRATE-α](https://arxiv.org/abs/2405.20299) (§1); [CRATE LM](https://arxiv.org/abs/2410.16443) (§5.2) | CRATE-B 70.8% vs ViT-S 72.4%, same size (ImageNet); α-L 85.1% with a modified block and ImageNet-21K; LM near GPT-2 | ViT-L; 81M LM | no |
| [Monet](https://arxiv.org/abs/2412.04139); [MoE-X](https://arxiv.org/abs/2503.07639) (Table 2 each) | Monet 4.1B: 0-shot mean 0.511 vs 0.520 for LLaMA 3.8B; MoE-X about equals a same-size Switch | 4.1B | no |
| [Bilinear MLP](https://arxiv.org/abs/2410.08417); [χ-nets](https://arxiv.org/abs/2504.02667) | Log-perplexity 1.960 vs ReLU 1.997, SwiGLU 1.944 ([Shazeer](https://arxiv.org/abs/2002.05202), 65k steps); χ-nets 1.5-2.2 points below ReLU (SVHN) | small | no; exact per-layer algebra |
| [SoLU](https://transformer-circuits.pub/2022/solu/index.html) (§1, §5, §6.2) | Same loss; interpretable neurons 35% to 60% up to 40 layers, 33% to 42% at 64; needs an extra LayerNorm that may hide features | about 50B | no |
| [Probabilistic circuits](http://starai.cs.ucla.edu/papers/ProbCirc20.pdf) (§4); [LVD](https://arxiv.org/abs/2210.04398); [GeLaTo](https://arxiv.org/abs/2304.07438) | Plateau as they grow; distilled latents bring them near VAEs and flows | ImageNet32 | marginals, moments: yes, linear in size |

**D. Scale-free structure (Q4)**

| Paper | Claim (status first) | Relevance |
|---|---|---|
| [Cohen, Sharir, Shashua 2016](https://arxiv.org/abs/1509.05009); [Cohen & Shashua 2017](https://arxiv.org/abs/1605.06743); [Levine+ 2018](https://arxiv.org/abs/1704.01552); [Levine+ 2019](https://arxiv.org/abs/1803.09780) | **P.** In arithmetic-circuit nets, depth gives exponentially higher separation rank for some cuts; pooling and channel min-cuts choose which; reuse gives volume-law entanglement | tangledness, formally |
| [Vidal 2008](https://arxiv.org/abs/quant-ph/0610099); [Stoudenmire 2018](https://arxiv.org/abs/1801.00315); [Loconte+ 2024](https://arxiv.org/abs/2409.07953) | **P; M; P.** MERA: exact local expectations; tree tensor networks train linearly in input size and data (MNIST); tensor factorizations are circuits | exact by construction |
| [Chandrasekaran+ 2008](https://arxiv.org/abs/1206.3240) | **P.** For arbitrary potentials, low treewidth is the only graph property giving tractable inference (grid-minor hypothesis) | your treewidth point |
| [Mehta & Schwab 2014](https://arxiv.org/abs/1410.3831); [Koch-Janusz & Ringel 2018](https://arxiv.org/abs/1704.06279) | **P; M.** Exact map from variational RG to stacked RBMs; mutual-information RG recovers the Ising exponent | RG analogy |
| [Lin & Tegmark 2017](https://arxiv.org/abs/1606.06737) (Fig. 1) | **P; M.** Mutual information decays exponentially under regular grammars, can decay as a power law under context-free ones; text shows power laws | trees, not chains |
| [Jelassi+ 2024](https://arxiv.org/abs/2402.01032); [Zoology](https://arxiv.org/abs/2312.04927); [Based](https://arxiv.org/abs/2402.18668); [Sanford+ 2023](https://arxiv.org/abs/2306.02896) | **P; M.** Fixed-state models cannot copy long strings; recall explains 82% of one gap to attention; state size trades against recall; communication-complexity bounds | recall needs state |
| [Hourglass](https://arxiv.org/abs/2110.13711); [H-Net](https://arxiv.org/abs/2507.07955) (§2); [Gu 2025](https://goombalab.github.io/blog/2025/tradeoffs/) | **M; C.** Downsampling hierarchies match or beat flat transformers; H-Net: SSM encoder-decoder, Transformer core, width growing with stage; SSMs as compressors | hierarchy helps; core dense |
| [Tay, Tsitsiklis, Win 2008](https://arxiv.org/abs/0803.2337); [Wagstaff+ 2019](https://arxiv.org/abs/1901.09006) | **P.** Bounded-height fusion trees with 1-bit messages often keep the optimal error exponent; continuous sum-pooling of arbitrary set functions needs dimension at least N | free or volume law |

## Proposal: a tree with Bayes-sized messages against the dense transformer (PFN prior)

**Why here.** In `work/tiny-pfn/world.py`, given the answer y* and the shown labels, sources' readings and residuals
are independent across sources, and exact Bayes adds per-source log-likelihoods of y*, each fixed by the reading and
the sum of squared residuals. A sum-pooling tree can thus represent Bayes with r-dimensional messages, r being the
epsilon-rank of that family of functions of y* at the tolerance the posterior mean needs; r is computable for each N
before training. Phase B supplies dense baselines; AlgZoo's outcome is predictable and World FA is parked.

**Design** (CPU; 3 seeds; activation noise σ = 0.05 as in phase B; matched parameters and FLOPs; Bayes recomputed
per world extension):
1. Tree: per-error MLP summed within the source, joined with its reading into a k-dim embedding, summed over
   sources, readout; k from r/2 to 4r. Dense: the phase B transformer; one token per source (no binding confound).
2. Knobs: N = 3 to 24 sources; G shared-error groups of unobserved random membership, coupling ρ = 0 or 0.5, so
   Bayes must compare sources: bits across a cut should grow with G if comparisons summarize, else with N.
3. Deduction (proposed; certifying it is part of the work): given y* (at ρ > 0 also shared errors and memberships),
   per-source terms are independent, so the joint law of the tree's root state and Bayes's rank-r state is a
   convolution in k + r dimensions. Push it through both readouts for the mean deviation from Bayes and
   P(|mean − Bayes| > t), with interval bounds, against sampling at equal FLOPs.

**What counts** (preset tolerance on KL to Bayes):
- For: the tree matches the dense net at k ≤ 2r for every N at ρ = 0, and at ρ > 0 with k growing in G, not N;
  deduction beats sampling's cost and tail bound (about 3/M at 95% after M hitless draws).
- Against: (a) at ρ = 0 the tree needs k far above r: learning, not representation, wants width (compare distilling
  from the dense net and from Bayes); (b) at ρ > 0 needed k grows with N: sum-pooling cannot summarize the
  comparisons, so attention's tangling earns its cost; (c) deduction is no cheaper than sampling even here.
