# Literature: order-averaging for inconsistent any-order models (2026-10-26)

Scope: papers that already link an any-order model, dependency network or set of conditionals to a single joint by averaging or reconciling over orderings, or that propose doing so in future work.
Method: Exa search plus WebSearch. I fetched each paper (arXiv/PMLR/JMLR/PMC PDF or HTML) and grepped its full text. Every quote below appears in the fetched text; I only normalised ligatures and line-break hyphens. Venues come from the paper header, the PMLR/JMLR page or Crossref. Checked 2026-10-26.
Not included: anything I could not open (a Stanford CS229 project on order space plus EM imputation; a 403'd OpenReview PDF). One Exa result had wrong author metadata (arXiv 2606.00997 is by Lin Yao, not "Asano et al."), so I took authors only from the paper text.

## 1. Inconsistent conditionals in masked / any-order models, and ensembling fixes

**Young, Chen & You (2022-24). "On the Inconsistencies of Conditionals Learned by Masked Language Models" (v1) / "Inconsistencies in Masked Language Models" (v3). arXiv preprint. https://arxiv.org/abs/2301.00068**
- What it does: shows that T5, UL2-20B and RoBERTa give conditionals that no joint can produce (bigram cycle test on RoBERTa; disagreement across mask patterns on MMLU/Lambada/BigBench). It then max-pools over mask patterns ("Ensemble of Conditionals").
- Abstract: "distributions corresponding to different masking patterns can demonstrate considerable inconsistencies, i.e., they cannot be derived from a coherent joint distribution when considered together."
- Sec. 4.3: "To ensemble different conditionals for a final prediction, we use the max-pooling approach".
- Sec. 6 (future work): "it can only be seen as a limited patch-up method that only unite a certain number of selected conditionals. We believe that for long-term research, this problem should be ideally addressed as part of the expensive pretraining stage".
- What we can reuse: the bigram 4-cycle test (their Eq. 4, from Arnold & Press 1989) is a cheap per-cycle inconsistency measure on real MLMs. Their combiner is max-pooling, not a joint, so ours is not anticipated.

**Uria, Côté, Gregor, Murray & Larochelle (2016). "Neural Autoregressive Distribution Estimation." JMLR 17. https://arxiv.org/abs/1605.02226** (ensembles first appeared in Uria, Murray & Larochelle, ICML 2014, https://arxiv.org/abs/1310.1757)
- What it does: trains one order-agnostic network (DeepNADE), then averages the probabilities of K random orderings (EoNADE).
- Sec. 4: "The models for different orderings will not be consistent with each other: they will assign different probabilities to a given test vector. However, we can use the models' differences to our advantage by creating ensembles of NADE models".
- Sec. 4.1: "This disagreement is undesirable if we require consistent inferences for different inference problems". The estimator is to "use the average probability 1/K ∑ p(x|θ,o(k)) as an estimator."
- What we can reuse: this is the arithmetic, uniform-weight baseline of our Bayesian mixture over orderings (the uniform prior with no evidence weighting). We should report EoNADE as a named baseline.

**Germain, Gregor, Murray & Larochelle (2015). "MADE: Masked Autoencoder for Distribution Estimation." ICML 2015. https://arxiv.org/abs/1502.03509**
- What it does: order- and connectivity-agnostic training, with test-time ensembles over orderings and masks.
- Sec. 4.2: "an ensemble of autoregressive models can be constructed on the fly, by exploiting the fact that the conditionals for two different orderings are not guaranteed to be exactly consistent".
- What we can reuse: the same uniform arithmetic ensemble, and a cheap architecture for exact small-n tests.

**Shih, Sadigh & Ermon (2022). "Training and Inference on Any-Order Autoregressive Models the Right Way." NeurIPS 2022. https://arxiv.org/abs/2205.13554**
- What it does: argues that AO-ARMs define the joint redundantly through N! paths. MAC trains on fewer, reweighted conditionals.
- Sec. 1: "By scaling up to all possible orderings, AO-ARMs inadvertently model the same distribution with a huge amount of redundancy."
- Sec. 5.2 (Limitations): "Since AO-ARMs evaluate marginal and conditional likelihoods using only compatible orderings, they are implicitly discarding away all the models in the N! ensemble with incompatible orderings. This can give biased estimates for marginal and conditional likelihoods."
- What we can reuse: the closest statement of our problem in the any-order literature. They name the "N! ensemble" and say that dropping members biases the estimates, but they do not propose weighting it. Their path-graph view (nodes = subsets, edges = univariate conditionals) matches our atom graph.

**Torroba Hennigen & Kim (2023). "Deriving Language Models from Masked Language Models." ACL 2023 (short). https://arxiv.org/abs/2305.15501**
- What it does: treats MLM unary conditionals as a dependency network and compares four ways to get an explicit joint over 2 tokens: MRF, Hammersley–Clifford–Besag and Arnold–Gokhale (AG, the nearest-compatible joint under KL to all conditionals). AG wins.
- Sec. 1: "If the conditionals are incompatible, then we can adapt approaches from statistics for deriving near-compatible joint distributions from incompatible conditionals (AG; Arnold and Gokhale, 1998)."
- Sec. 6 (future work): "It would be interesting to explore whether explicitly regularizing the conditionals to be compatible during MLM training would lead to better modeling of the distribution of language."
- Sec. 7 (limitations): "We focused on models over pairwise tokens ... this limits the generality of our approach (e.g., we cannot score full sentences)."
- What we can reuse: AG is an atom-level projection, the same move as our "precision-weighted projection of all atoms" in gls-order-energy. It has already been run on BERT for pairs, with the result that AG's unary conditionals sometimes beat the MLM's own. AG is unweighted, so our version would add per-atom precision weights and go beyond pairs.

**Majid, Pukdee, Agrawal, Varıcı & Ravikumar (2025). "On the Consistent Recovery of Joint Distributions from Conditionals." AISTATS 2025. https://proceedings.mlr.press/v258/majid25a.html**
- What it does: gives necessary conditions (path consistency) and necessary-and-sufficient conditions (autoregressive + swap consistency) for masked conditionals to come from one joint, and proposes consistency measures as proxies.
- Abstract: "we propose the concepts of autoregressive and swap consistency, which we show are necessary and sufficient conditions for a consistent joint."
- Sec. 6: "We pose an open question on how can we parameterize models of conditional distributions to enable learning of rich joint distributions?"
- What we can reuse: their swap-consistency residuals are exactly the "ordering contrasts" our gls_est fits variances from. Their measures could replace our moment estimator of C.

**Goyal, Dyer & Berg-Kirkpatrick (2022). "Exposing the Implicit Energy Networks behind Masked Language Models via Metropolis–Hastings." ICLR 2022. https://arxiv.org/abs/2106.02736**
- What it does: defines a joint energy from MLM full-conditional scores (raw or locally normalised) and samples it with MH, using the MLM conditionals as proposals.
- Abstract: "samples are proposed from the same masked conditionals used for training the masked language models, and they are accepted or rejected based on their energy values according to the target distribution", and "the masked conditionals on their own do not yield a Markov chain whose stationary distribution is that of our target distribution".
- What we can reuse: (a) their E_norm = −∑_t log p_mlm(x_t|x_\t) is the equal-weight geometric pool of the full-conditional atoms only, a special case of our atom energy. (b) It is the template for "the proposal or search policy changes efficiency, not the target" in exactly our model class.

**Lin Yao (2026). "Decoding in Order-Agnostic Language Models: Chain-Rule Deviation and Uniform Spreading." arXiv preprint (single LLaDA-2.1-mini checkpoint). https://arxiv.org/abs/2606.00997**
- Abstract: "the learned conditionals are not exact factorizations of a coherent joint distribution: changing only the reveal order shifts target log-likelihood by up to 0.49 nats/token".
- Sec. 5: "restoring chain-rule coherence is a training-side problem outside our scope."
- What we can reuse: an order-dependence measurement on a current dLLM, as motivation and a sanity check on scale. Weak evidence (one checkpoint).

## 2. Dependency networks and pseudo-Gibbs with incompatible conditionals

**Heckerman, Chickering, Meek, Rounthwaite & Kadie (2000). "Dependency Networks for Inference, Collaborative Filtering, and Data Visualization." JMLR 1. https://www.jmlr.org/papers/volume1/heckerman00a/html/node6.html**
- General Dependency Networks › Definition and Basic Properties: "Another disturbing observation is that the joint distribution obtained will depend on the order in which the pseudo-Gibbs sampler visits the variables." It continues: "Due to the near consistency of the local distributions, however, the joint distributions obtained from different orderings will be close."
- What we can reuse: the original statement of scan-order dependence, with a 2-variable example in which one scan gives dependence and the other independence. Their defence assumes near-consistency, which our "fuzzy atoms" regime breaks.

**Lowd (2012). "Closed-Form Learning of Markov Networks from Dependency Networks." UAI 2012. https://arxiv.org/abs/1210.4896**
- What it does: converts a DN into a log-linear MN using an ordering o and a base instance x′. For inconsistent DNs it averages over base instances and orderings.
- Sec. 5: "no single choice of x′ and o leads to the true, uniform distribution. To obtain a uniform distribution, we must average over several conversions."
- Sec. 5.1: "convert the DN to an MN k times and take the geometric mean of the resulting distributions."
- Sec. 5.2: "Since the conversion depends on the ordering, we would like to average over all possible orderings as well." Also: "Note that these orderings should not be weighted equally." (The unequal weights there are combinatorial counts, not learned precisions.)
- What we can reuse: the closest prior art to our geometric pool over orderings. It covers a closed-form log-linear construction, the exact feature-weight bookkeeping for averaging over orderings, and averaging base instances by product-of-marginals rather than uniformly. What it lacks is data- or precision-dependent weights; that is our GLS addition.

**Chen, Ip & Wang (2011). "Gibbs Ensembles for Nearly Compatible and Incompatible Conditional Models." Computational Statistics & Data Analysis 55(4):1760–1769. https://pmc.ncbi.nlm.nih.gov/articles/PMC3030131/**
- What it does: computes the stationary joint of each of the m! fixed-scan pseudo-Gibbs samplers and takes a weighted average, with weights set by each joint's divergence from the given conditionals.
- Sec. 1: "Then a weighted average of all the distributions in the ensemble is taken as a solution. The combination weights are inversely proportional to the divergences of the members in the ensemble."
- Sec. 5 (future work): "By varying scan patterns to generate ensembles of Gibbs distributions, we believe that the power of GE can be further expanded. Instead of using pre-determined probabilities to perform random scan, one may be able to further refine the search in the space of a Gibbs distribution for a better solution".
- What we can reuse: this is our "weighted mixture over orderings" with a self-consistency weight (divergence to the model's own conditionals, no data), which is the same information structure as gls_est. They report that "the use of different kinds of weights seems to play a relatively minor role" (Sec. 3), which we should test against. Follow-ups: WIREs Comput Stat 2013 (doi 10.1002/wics.1273) and Stat. Interface 2014 (doi 10.4310/sii.2014.v7.n2.a10). I found both but did not read them.

**Chen & Ip (2015). "Behavior of the Gibbs Sampler When Conditional Distributions Are Potentially Incompatible." J. Stat. Comput. Simul. 85(16):3266–3275. https://pmc.ncbi.nlm.nih.gov/articles/PMC4572746/**
- Abstract: "we demonstrate that what a Gibbs sampler converges to is a function of the order of the sampling scheme."
- Sec. 3: "the surmise [11] that the stationary distribution for a random-scan Gibbs sampler is a mixture of the stationary distributions for all systematic scan Gibbs samplers is not true in general. It only holds for d = 2."
- What we can reuse: a counterexample showing that random-scan pseudo-Gibbs is not an average over orderings for d ≥ 3. It supports our finding that pseudo-Gibbs is a different target from either order-average.

**Kuo & Wang (2019). "Pseudo-Gibbs sampler for discrete conditional distributions." Ann. Inst. Stat. Math. 71:93–105. https://www.ism.ac.jp/editsec/aism/pdf/s10463-017-0625-x.pdf**
- Abstract: "we prove that systematic pseudo-Gibbs projections converge for every scan order, and the stationary distributions share marginal distributions in a circularly fashion."
- Sec. 4: "Pseudo-Gibbs sampler translates the incompatibility among conditional models into an ordered diversity."
- What we can reuse: each Gibbs step is an I-projection, and circularly related scans share univariate marginals. This structure can be used to parameterise or reduce the d! scan joints before pooling them.

**Muré (2019). "Optimal compromise between incompatible conditional probability distributions, with application to Objective Bayesian Kriging." ESAIM: Probability and Statistics (doi 10.1051/ps/2018023). https://arxiv.org/abs/1703.07233**
- Abstract: the random-scan pseudo-Gibbs stationary law "minimizes a mean squared misfit between them and its own conditional distributions."
- Sec. 2: "If the Markov kernels (πi) are compatible, then the scanning order influences only the rate of convergence of the algorithm." Also: "If the Markov kernels are not compatible, then every scan order may produce a different target distribution."
- What we can reuse: (a) a published definition of an order-free target (the "optimal compromise") for full conditionals only, which we can compare our atom energy against; (b) the exact sentence for our principle that the search policy should only change efficiency.

**Jiang, Lim, Yao & Ye (2011). "Statistical ranking and combinatorial Hodge theory." Mathematical Programming (doi 10.1007/s10107-010-0419-x). https://arxiv.org/abs/0811.1067** (a tool, not a connection)
- Abstract: an edge flow splits into "a gradient flow that represents the L2-optimal global ranking and a divergence-free flow (cyclic) that measures the validity of the global ranking", and "discrete Hodge decomposition may be computed via a linear least squares regression."
- What we can reuse: log-odds atoms are edge flows on the hypercube. The gradient part is the least-squares consistent joint and the curl/harmonic part is the inconsistency. **Gap: I found no paper that applies Hodge decomposition to compatibility of conditionals.** The nearest are Muré (mean squared misfit) and Arnold–Gokhale via Torroba Hennigen & Kim (KL).

Also seen, lower priority: Wu, Zhou, Zhou & Lyu (2026-10-22), "Finite-Query Coherent Projection of Black-Box Conditional Predictions", arXiv 2610.30708. It gives minimax rates for projecting black-box conditional tables onto a pairwise joint class ("Local conditional predictions from a black box need not admit a common joint distribution."). It is 4 days old and unreviewed.

## 3. Bayesian averaging over orderings

**Friedman & Koller (2000). "Being Bayesian about Network Structure." UAI 2000 (journal version Machine Learning 2003). https://arxiv.org/abs/1301.3856**
- What it does: runs MCMC over orderings, with closed-form sums over DAGs consistent with each ordering.
- Sec. 4.1 (order-prior bias): "graphs that are consistent with more orderings are more likely; for example, a Naive Bayes graph is consistent with (n−1)! orderings, whereas any chain-structured graph is consistent with only one."
- What we can reuse: the bias appears exactly when the object we care about is shared by many orderings. For a full any-order model, each ordering is a complete DAG and the bias does not arise. If we prune conditioning sets to Markov blankets, it comes back.

**Koivisto & Sood (2004). "Exact Bayesian Structure Discovery in Bayesian Networks." JMLR 5. https://www.jmlr.org/papers/volume5/koivisto04a/koivisto04a.pdf**
- Abstract: computes "the exact posterior probability of a subnetwork" in "time O(n2^n + n^{k+1}C(m))".
- Sec. 3: "averaging over orders renders the marginal distribution of the cardinality |Gi| slightly biased from uniform, since small cardinalities are favored as they are consistent with more orders."
- What we can reuse: the subset dynamic program sums over all n! orderings in O(n2^n) when the order score factorises over (variable, predecessor set). Our ordering score ∑_i log q(x_i | x_{pred(i)}) factorises exactly that way. So for n ≲ 25 the exact Bayesian mixture over all orderings of an any-order model is computable with no search.

**Niinimäki, Parviainen & Koivisto (2016). "Structure Discovery in Bayesian Networks by Sampling Partial Orders." JMLR 17. https://www.jmlr.org/papers/volume17/15-140/15-140.pdf**
- Abstract: "arbitrary priors can be handled by dividing the importance weight of each sampled DAG by the number of topological sorts it has".
- What we can reuse: partial-order sampling (smoother space) plus an exact correction of the order-prior bias.

**Kuipers & Moffa (2017). "Partition MCMC for Inference on Acyclic Digraphs." JASA 112(517). https://arxiv.org/abs/1504.05006**
- Abstract: "Current examples of algorithms taking advantage of grouping are the biased order MCMC", while partition MCMC keeps "the property of producing an unbiased sample."
- What we can reuse: the standard unbiased alternative, if we ever need DAG-level rather than ordering-level posteriors.

Neural autoregressive / any-order models with orders as latent variables:
- **Wang, Shi, Heess, Gretton & Titsias (2025). "Learning-Order Autoregressive Models with Application to Molecular Graph Generation." ICML 2025. https://arxiv.org/abs/2503.05979.** Sec. 2.1 reads the AO-ARM objective as a bound on a mixture over orderings: "log pθ(x) = log ∑σ p(σ)p(x|σ) ≥ Ep(σ)[log p(x|σ)]". Sec. 3: "We replace the uniform prior distribution p(σ) in AO-ARMs by with a learnable distribution over orderings." Reuse: the formal statement that an AO-ARM's implied joint is a mixture over orderings. In their model the order policy is part of the model, so it changes the target; that is the opposite of our design.
- **Chen, Han, Hu, Ruiz & Liu (2021). "Order Matters: Probabilistic Modeling of Node Sequence for Graph Generation." ICML 2021. https://arxiv.org/abs/2106.06189.** Abstract: "we approximately marginalize out the node orderings and compute a lower bound on the log-likelihood using variational inference." Reuse: a variational posterior q(π|x) over orderings for a neural AR model, i.e. an amortised version of our evidence-weighted mixture.
- **Li, Trabucco, Park, Luo, Shen, Darrell & Gao (2021). "Discovering Non-monotonic Autoregressive Orderings with Variational Inference." ICLR 2021. https://arxiv.org/abs/2110.15797.** Abstract: an encoder and decoder "perform variational inference with autoregressive orders (represented as permutation matrices) as latent variables."
- **Toth, Knoll, Pernkopf & Peharz (2025). "Effective Bayesian Causal Inference via Structural Marginalisation and Autoregressive Orders." AISTATS 2025. https://arxiv.org/abs/2402.14781.** Abstract: "decomposing structure marginalisation into the marginalisation over (i) causal orders and (ii) directed acyclic graphs (DAGs) given an order", with "a novel auto-regressive distribution over causal orders (ARCO)". This is the closest neural order-averaging paper (see item 4).
- **Gap:** I found no paper that takes a trained, inconsistent any-order model and weights its orderings by evidence or prequential score to define one joint, and none that proposes this in future work. The nearest are Shih et al.'s "N! ensemble" limitation (unweighted) and EoNADE's uniform average.

## 4. Learned proposals / amortised samplers over orderings or DAGs with the target held fixed

**Deleu, Góis, Emezue, Rankawat, Lacoste-Julien, Bauer & Bengio (2022). "Bayesian Structure Learning with Generative Flow Networks." UAI 2022. https://arxiv.org/abs/2202.13903**
- Sec. 3.1: the flow-matching system "is in general underdetermined, and therefore it may admit many solutions Fθ(s→s′) that all induce the same distribution ∝R(s)."
- Sec. 5.2: "This distribution π(G→G′) can be arbitrary; ... as long as it assigns non-zero probability to any next state G′."
- Sec. 7 (future work): "learning the structure of models with latent variables."
- What we can reuse: the explicit claim that the target is fixed by the reward, while the exploration and training distribution only affects learning. It maps directly onto a GFlowNet over orderings whose reward is our order score.

**Toth et al. (2025), AISTATS (above). https://arxiv.org/abs/2402.14781**
- Sec. 4.2: p(L|θ) "can be interpreted as proposal distribution with importance weights as defined in Equation (4), where the optimal proposal is the true posterior over causal orders."
- What we can reuse: a neural autoregressive proposal over orderings, corrected by importance weights so the posterior does not depend on the proposal. This is the clearest published instance of "the policy affects efficiency, not the target" for orderings.

**Goyal et al. (2022), ICLR (item 1).** MLM conditionals serve as MH proposals and the energy fixes the target, in our exact model class.

**Levy, Hoffman & Sohl-Dickstein (2018). "Generalizing Hamiltonian Monte Carlo with Neural Networks." ICLR 2018. https://arxiv.org/abs/1711.09268**
- Abstract: neural MCMC kernels "that converge and mix quickly to their target distribution", "trained to maximize expected squared jumped distance, a proxy for mixing speed." Sec. 4.1.3: each step "satisfies detailed balance with respect to p".

**Song, Zhao & Ermon (2017). "A-NICE-MC: Adversarial Training for MCMC." NeurIPS 2017. https://arxiv.org/abs/1706.07561**
- Sec. 4: "We want to train a Markov chain such that the stationary distribution πθ is exactly pd", which they obtain by MH acceptance.
- What we can reuse (both): the standard pattern of training the proposal for mixing while MH or detailed balance keeps the target exact.

Contrasts, where the order policy does change the result:
- **Wang, Du, Zhu, Ke, Chen, Hao & Wang (2021). "Ordering-Based Causal Discovery with Reinforcement Learning." IJCAI 2021. https://arxiv.org/abs/2105.06631.** Abstract: "use RL to optimize the proposed model based on the reward mechanisms designed for each ordering". This is an RL search for one best-scoring ordering, not averaging.
- **Kim, Shah, Kontonis, Kakade & Chen (2025). "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions." ICML 2025. https://arxiv.org/abs/2502.06768.** Abstract: "a suitable strategy for adaptively choosing the token decoding order significantly enhances the capabilities of MDMs" (Sudoku <7% to ≈90%). In MDM decoding the order policy is part of the output distribution, the opposite of our design and the reason the distinction matters.

## 5. Evidence, Kelly wealth, BMA vs stacking, forecast combination

**Beygelzimer, Langford & Pennock (2012). "Learning Performance of Prediction Markets with Kelly Bettors." AAMAS 2012. https://arxiv.org/abs/1201.6655**
- Abstract: "the market prediction is a wealth-weighted average of the individual participants' beliefs. Second, the market learns at the optimal rate, the market price reacts exactly as if updating according to Bayes' Law, and the market prediction has low worst-case log regret to the best individual participant."
- What we can reuse: a citable "evidence = Kelly wealth" link. Weighting orderings by prequential likelihood is a Kelly market among orderings. It also gives fractional Kelly as a tempered-posterior variant (Sec. 6).

**Minka (2000; PDF dated 2002). "Bayesian model averaging is not model combination." MSR technical note. https://tminka.github.io/papers/minka-bma-isnt-mc.pdf**
- "As more data arrives, the hypotheses become more distinguishable and BMA will always focus its weight on the most probable hypothesis". The fix: "This is BMA applied to a new hypotheses space of “stacked” models."
- What we can reuse: the core objection to an evidence-weighted mixture over orderings. With data it collapses to one ordering, while the atom/GLS pool is a combination. That is our two candidate targets in one paragraph.

**Yao, Vehtari, Simpson & Gelman (2018). "Using Stacking to Average Bayesian Predictive Distributions." Bayesian Analysis 13(3) (doi 10.1214/17-BA1091). https://arxiv.org/abs/1704.02030**
- Sec. 1: "In the M-open and M-complete case, BMA will asymptotically select the one single model in the list that is closest in Kullback-Leibler (KL) divergence."
- What we can reuse: log-score stacking of orderings (with PSIS-LOO) as the M-open alternative to evidence weights. Every ordering of an inconsistent model is misspecified, so this is the M-open case.

**Claeskens, Magnus, Vasnev & Wang (2016). "The forecast combination puzzle: A simple theoretical explanation." International Journal of Forecasting 32:754–762. https://www.janmagnus.nl/papers/JRM113a.pdf**
- Abstract: "If the fact that the weights are random rather than fixed is taken into account during the optimality derivation, then the forecast combination will be biased (even when the original forecasts are unbiased), and its variance will be larger than in the fixed-weight case. In particular, there is no guarantee that the ‘optimal’ forecast combination will be better than the equal-weight case".
- What we can reuse: a direct explanation of our gls_est result (estimated GLS weights lose to uniform at small precision ratios), and the bias/variance terms for estimated weights.

## 6. Inconsistency as loss / energy (PDGs)

**Richardson & Halpern (2021). "Probabilistic Dependency Graphs." AAAI 2021. https://arxiv.org/abs/2012.10800**
- Abstract: "PDGs can capture inconsistent beliefs in a natural way". It provides "three semantics for PDGs, each of which can be derived from a scoring function ... that can be viewed as representing a distribution's incompatibility with the PDG."
- Discussion (future work): "We hope to report further on the deep connection between inference, updating, and the resolution of inconsistency in PDGs in future work."
- What we can reuse: an any-order model with one weighted edge per atom is a PDG. Our atom energy is its scoring function, and the β weights are our precisions.

**Richardson (2022). "Loss as the Inconsistency of a Probabilistic Dependency Graph: Choose Your Model, Not Your Loss Function." AISTATS 2022. https://arxiv.org/abs/2202.11862**
- Abstract: "many standard loss functions arise as the inconsistency of a natural PDG".
- Sec. 7, Prop. 14: "For all weighted factor graphs Ψ, we have that ⟨⟨MΨ⟩⟩1 = −log ZΨ."
- What we can reuse: for a geometric pool of atoms (a weighted factor graph), the PDG inconsistency is −log Z of the pool. That gives a scalar "how inconsistent are these orderings or atoms" for free.

**Richardson, Halpern & De Sa (2023). "Inference for Probabilistic Dependency Graphs." UAI 2023. https://arxiv.org/abs/2311.05580**
- Abstract: "We present the first tractable inference algorithm for PDGs with discrete variables", via exponential-cone convex programs for bounded treewidth.
- What we can reuse: an exact solver for the atom-level projection on small or bounded-treewidth instances, a reference to test our GLS approximation against.

**Richardson, Samiei, Shakerinava, Viviano, El Kabid, Parviz & Bengio (2026). "Local Inconsistency Resolution: The Interplay between Attention and Control in Probabilistic Models." AISTATS 2026. https://arxiv.org/abs/2604.17140**
- Abstract: "iteratively focus on a subset of the model and resolve inconsistencies using the parameters under control".
- Prop. 4: with non-deterministic Refocus, "the possible runs of LIR(...) are precisely those of BP for different message schedules."
- Sec. 6 (future work): "Under what assumptions does LIR converge? Can we learn adaptive Refocus mechanisms (drawing inspiration from the transformer architecture)?"
- What we can reuse: Refocus plays the role of our search policy over orderings and atoms. Their open question (when does the focus schedule leave the fixed point unchanged?) is our "policy affects only efficiency" claim in PDG terms. We could answer it for the atom energy, since its optimum is global and convex.

## 7. Hidden / missing variables jointly with order-MCMC

**Friedman & Koller (2000), UAI. https://arxiv.org/abs/1301.3856**
- Sec. 6 (future work): "We believe that this approach can be extended to deal with data sets where some of the data is missing, by extending the MCMC over orderings with MCMC over missing values, allowing us to average over both. If successful, we can use this combined MCMC algorithm for doing full Bayesian model averaging for prediction tasks as well."
- Koivisto & Sood (2004, Sec. 6.2) repeat it: "as noted by Friedman and Koller, MCMC can be naturally extended to deal with data sets where some of the data is missing."

Who did something close. All of these are in DAG space, not ordering space:
- **Antal, Millinghoffer, Hullám, Timmerman, Moreau, Gevaert & De Moor (2009). "Structure learning of Bayesian networks with MCMC: extension to incomplete data and decision trees as local models." White paper / KU Leuven ESAT report 09-170. https://ftp.esat.kuleuven.be/pub/stadius/ida/reports/09-170.pdf.** "a novel Metropolis-in-Gibbs algorithm for structure learning, which performs a random walk in the joint space of DAGs and missing values." They also discuss ordering MCMC, but the missing-data sampler is over DAGs.
- **Grzegorczyk (2023). "Being Bayesian about learning Gaussian Bayesian networks from incomplete data." Int. J. Approx. Reasoning (doi 10.1016/j.ijar.2023.108954). https://research.rug.nl/en/publications/being-bayesian-about-learning-gaussian-bayesian-networks-from-inc/.** Abstract: "We present a Markov Chain Monte Carlo sampling algorithm that allows for simultaneously sampling directed acyclic graphs (DAGs) as well as the values of the unobserved data points." DAG MCMC with BGe; abstract read only.
- **Myers, Laskey & Levitt (1999). "Learning Bayesian Networks from Incomplete Data with Stochastic Search Algorithms." UAI 1999. https://arxiv.org/abs/1301.6726.** Abstract: "Our approach evolves structure and the missing data."
- **Pirola, Stella & Grzegorczyk (2025). "LUME-DBN: Full Bayesian Learning of DBNs from Incomplete data in Intensive Care." arXiv 2511.04333.** Gibbs imputation combined with DBN structure MCMC (abstract read only).
- Neural side: DAG-GFlowNet names latent-variable structure learning as future work (item 4). LO-ARM, Order Matters and Li et al. treat the ordering itself as the latent variable, but none adds hidden data variables.
- **Gap:** I found no paper that implements Friedman & Koller's proposal as stated (MCMC over orderings jointly with MCMC over missing values), and none for neural any-order models. Searches: Exa ×8 query variants and one WebSearch, terms "order MCMC" + missing/incomplete/hidden/latent/imputation/data augmentation.

## Read first (ranked)

1. **Lowd 2012 (UAI)**: an existing geometric mean over orderings (and base instances) of an inconsistent conditional model, with exact log-linear bookkeeping and the remark that orderings "should not be weighted equally". Our GLS pool is this plus learned weights, so we must position against it.
2. **Torroba Hennigen & Kim 2023 (ACL)**: atom-level nearest-compatible projection (Arnold–Gokhale) applied to BERT, beating MRF and HCB. This is our proposed "aggregate at the atom level" already done for pairs, with no weights. It gives us a baseline and a first neural experiment to replicate.
3. **Chen, Ip & Wang 2011 (CSDA)**, read with Chen & Ip 2015 and Kuo & Wang 2019: divergence-weighted mixtures over all scan orders, the statistics literature's version of our weighted mixture over orderings using self-consistency weights only. It includes the d ≥ 3 counterexample showing random scan is not an order mixture.
4. **Goyal, Dyer & Berg-Kirkpatrick 2022 (ICLR)**: in our exact model class (MLMs), an energy built from conditionals defines the target and the MLM is only the MH proposal. It is the closest published template for "the search policy affects efficiency, not the target". Add **Toth et al. 2025** for a learned neural order proposal with importance weights.
5. **Muré 2019 (ESAIM P&S)**: a formal "optimal compromise" target for incompatible conditionals, with the exact sentence that scan order affects only the convergence rate when kernels are compatible and changes the target when they are not. It is the target definition our atom energy has to be compared against.

Also needed for the weighting question: Minka (BMA collapses to one ordering), Yao et al. 2018 (stacking), Claeskens et al. 2016 (why estimated weights lose to uniform). For exact small-n order sums: Koivisto & Sood 2004.
