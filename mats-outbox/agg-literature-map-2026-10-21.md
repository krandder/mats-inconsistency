# Literature map: attention-style aggregation toward the Bayes posterior

**Verification.** Every work listed was confirmed to exist through the arXiv API, Crossref or a fetched DOI or URL. I re-checked four 2025–26 ids against their arxiv.org/abs titles: 2609.01337, 2609.33257, 2512.22625 and 2511.07678. Content claims come from the abstracts unless marked **(recall)**.

## 1. Ten must-know works

1. **Satopaa, Jensen, Pemantle, Ungar (2015/17), Partial Information Framework, arXiv:1505.06472.** Each forecast is treated as a conditional expectation given a subset of a shared information pool. The Gaussian oracle is Σ_YX Σ_XX⁻¹ X **(recall)**. This is the closest existing model of our "keys" for overlap and duplication.
2. **Winkler (1981), doi:10.1287/mnsc.27.4.479, and Clemen & Winkler (1985), doi:10.1287/opre.33.2.427.** GLS pooling w ∝ Σ⁻¹1. Under equicorrelation, n experts are worth n/(1+(n−1)ρ) independent ones, which tends to 1/ρ **(recall)**. Our Gaussian clearing layer is a keyed, low-rank version of this.
3. **Müller et al. (2022), Transformers Can Do Bayesian Inference (PFN), arXiv:2112.10510.** Train on samples from a prior over tasks and the model approximates the posterior predictive. This is our training recipe almost exactly.
4. **Lee et al. (2019), Set Transformer, arXiv:1810.00825.** The order-free form of the "plain self-attention over all agents" bar, built from SAB+PMA blocks (block names from recall).
5. **Garcia Satorras & Welling (2021), Neural Enhanced BP, arXiv:2003.01998.** An exact BP backbone with a learned GNN correction to its messages at each iteration. This is the template for our inference layer.
6. **Julier & Uhlmann (1997), Covariance Intersection, doi:10.1109/ACC.1997.609105; Grime & Durrant-Whyte (1994), channel filter, doi:10.1016/0967-0661(94)90349-2.** CI is the conservative answer when correlation is unknown. The channel filter subtracts common information exactly, but only when provenance is tracked (channel-filter content from recall).
7. **Arieli, Babichenko, Smorodinsky (2018), doi:10.1073/pnas.1813934115; Neyman & Roughgarden (2021), arXiv:2111.03153.** Impossibility results for robust aggregation that sees forecasts only, plus averaging and extremizing under substitutes.
8. **Palley & Soll (2019), doi:10.1287/mnsc.2018.3047; Prelec, Seung, McCoy (2017), doi:10.1038/nature21054.** Meta-predictions (each source's estimate of the crowd) as a second-order signal that is not provenance. The latter is the "surprisingly popular" rule (content from recall).
9. **Das et al. (2019), TarMAC, arXiv:1810.11187.** Learned signature, value and query messages trained from reward only. This is the closest prior form of our key/value design, and it was never scored against Bayes (mechanism from recall).
10. **Hahn (2020), arXiv:1906.06755; Velickovic et al. (2024), Softmax is not Enough, arXiv:2410.01104.** Fixed-depth attention cannot handle PARITY-like functions as length grows, and softmax selection disperses as n grows.

Also worth reading: LEAP (arXiv:2609.01337), which elicits likelihoods per evidence item and combines them with a fixed Bayes model; Kim et al. (arXiv:2506.07962), which finds correlated errors across more than 350 LLMs; and Xu et al., GIN (arXiv:1810.00826), on mean vs sum aggregation (that result from recall).

## 2. Baselines for aggbench

**Closed-form ladder:**
- linear pool
- mean logit
- extremized mean logit with fitted a (Satopaa et al. 2014, doi:10.1016/j.ijforecast.2013.09.009)
- Winkler GLS with fitted Σ
- Gaussian PIF oracle
- CI (convex pool in information form)
- ICI (Noack et al. 2017, doi:10.1016/j.automatica.2017.01.019)
- naive sum of LLRs
- channel filter with oracle provenance
- myopic-trader LMSR (Hanson 2003, doi:10.1023/A:1022058209073)
- surprisingly popular / pivoting, run when the meta-prediction field is present

**Learned ladder:**
- Deep Sets sum (1703.06114)
- CommNet mean (1605.07736)
- TarMAC with 1 round, then multiple rounds
- unrolled BP with learned per-edge weights (Nachmani et al., 1607.04793)
- Set Transformer SAB+PMA
- a sequential PFN-style transformer averaged over agent permutations, which is the bar
- learned linear pool of LLM forecasters (2609.33257)

**Metrics:** report KL to the exact posterior as well as log loss and Brier. Report results on held-out information structures and at train-n vs 2n–4n test agents.

**Real data:** ForecastBench (2409.19839) and the Metaculus AI Benchmark (bot, Pro and community forecasts on the same questions). Model family, retrieval source and prompt can serve as provenance keys.

## 3. Architectures to borrow, by open problem

**Learning who knows what from content (hidden provenance)**
- Predict the structure (B, or the shared information between each pair of sources), not the fusion weights. Precedents are ICI-style bounded subtraction and DIFNet (2508.18854).
- Amortized structure learning across questions: AVICI (2205.12934) and CSIvA (2204.04875). Supervise the relation-classification head with the simulator's true relation labels.
- Add a meta-prediction message field. Its Bayes-optimal value can be computed in the simulator.

**XOR and complementary information**
- Pairwise q·k attention cannot express a factor over 3 or more sources in one step. Add factor or hyperedge tokens (FGNN, 1906.00554).
- Keep XOR arity k fixed and sweep n and k separately.

**Beating the full-context transformer bar**
- Hybrid layer: exact (KᵀBK+Λ)w = KᵀB·dl solve plus a learned residual (NEBP; Satorras, Akata & Welling, 1906.02547).
- Use the BPNN-D construction (2007.00295) to keep the exact solver's fixed points for any weights. That gives a guaranteed floor.
- Add an explicit sum/count path (GIN, PNA 2004.05718), because softmax gives each source O(1/n) influence (recall).
- Ablation: unrolled linear-attention gradient-descent steps (von Oswald et al., 2212.07677) vs the exact solve.
- Algorithmic alignment (1905.13211) predicts better sample efficiency for this design. Measure it.

**Label-free training**
- The simulator already gives labels for free in the PFN setting.
- CE on outcomes and distillation from Bayes have the same population minimizer. So the finding that only distillation makes vector messages use the extra bandwidth is probably a variance effect. Test it by scaling outcome data or using soft labels before drawing conclusions.
- For real data, the PIF forecast covariance encodes overlap with no outcomes needed (recall). Nachmani's codeword independence shows how symmetry can cut the labels a model needs.
- Measure I(message; evidence) with an information-bottleneck penalty (IMAC, 1911.06992).

## 4. Theory predicting where attention aggregators fail

- **Normalization:** a softmax readout is a convex pool, so it falls in the linear-pool or log-pool family. With n independent sources, Bayes needs Σ logit pᵢ − (n−1)·logit(prior). A readout whose weights sum to 1 cannot express this (Genest & Zidek 1986; Tsai et al., 1908.11775).
- **Duplicates:** in the Hopfield/kernel view, softmax weights a cluster of near-identical keys by its size (Ramsauer et al., 2008.02217). Mean pooling cannot tell k agreeing independent sources from one source copied k times. This explains why a self-attention layer before the readout beats a single q·k readout.
- **Identifiability:** forecasts alone cannot reach Bayes across arbitrary structures (Arieli et al.; Neyman & Roughgarden). With no provenance, the best available is a conservative bound such as CI. So "keys need provenance" is the expected result, not a failure.
- **Sequential aggregation:** a market reaches the pooled posterior only for separable securities, and XOR-like ones can fail (Ostrovsky 2012, doi:10.3982/ECTA8479).
- **Sensitivity and size:** sensitive functions sit at sharp points of parameter space (Hahn & Rofin, 2402.09963). Count-dependent fixes generalize poorly to unseen lengths (Chiang & Cholak, 2202.12172).
- **Representation:** sum-decomposition needs latent dimension of at least n to be universal (Wagstaff et al., 1901.09006).
- **Loopy Gaussian BP:** when it converges, the means are correct but the variances are generally wrong (Weiss & Freeman 2001, doi:10.1162/089976601750541769, recall). Solve directly and score calibration.
- **Prior dependence:** below a task-diversity threshold, a transformer tracks Bayes under the training mixture, not the true prior (Raventos et al., 2306.15063).

## 5. What is new and what is not

**Not new:**
- Transformers trained to approximate a posterior (PFN)
- Key/value messages learned end to end (TarMAC)
- Gaussian clearing (GLS, information filter, PIF)
- BP plus a learned correction (NEBP)
- Extremizing, and meta-prediction aggregation

**New, as far as these facets found:**
- Scoring learned aggregators against the *exact* Bayes posterior across a controlled taxonomy of information structures (duplicates, correlated noise, collusion, XOR, hypothesis vs evidence sources). None of the MARL communication papers or LLM ensembles checked do this.
- Attention that classifies relations between sources and feeds an exact clearing layer.
- The provenance/identifiability result, framed as which second-order signals (provenance, meta-predictions, track records) restore Bayes.
- Real LLM crowds as the correlated-source regime (2506.07962, 2512.22625), which links the synthetic findings to Metaculus data.