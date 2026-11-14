# Careful read: Bayesian wind tunnels (I, II, III), TabPFN mechanism papers, ARC AlgZoo

2026-10-26. For the MATS study of how a small set-style PFN transformer deduplicates correlated evidence and aggregates information.

**How this was checked.** I downloaded all five arXiv PDFs on 2026-10-26 (2512.22471 v5, 2512.22473 v5, 2607.19379 v1, 2605.21288 v1, 2506.08982 v2), extracted the text with pypdf and read each one in full, including the appendices. Quotes are copied from the extracted text. I only restored spaces that pypdf drops between words. For AlgZoo I read ARC's blog post (alignment.org, 2026-01-26), the GitHub README, the model-grid figures (axis labels and accuracies taken from the SVG source) and the repo's issues via the GitHub API. Two of my own numerical checks are marked **[check]**. My own inferences and proposals are marked **[ours]**. Where a paper is silent or ambiguous, I say so rather than guess.

---

## A. Agarwal, Dalal & Misra, "The Bayesian Geometry of Transformer Attention" (Paper I), arXiv 2512.22471 v5

### Setup
- **Tasks ("wind tunnels").**
  - Bijection elimination: V=20, context of input→output pairs, "[x1, y1, SEP, …, x19, SEP]", with the loss at every y position.
  - HMM filtering: S=5 hidden states, V=5 symbols, Dirichlet(1) rows, a "10-token header encoding flattened T and E", trained at K=20 and evaluated at K=20/30/50.
  - Associative recall: N=16 pairs, vocabulary 256+256.
  - Bayesian linear regression: d=3, k=6 points, 41 output bins.
- **Models.**
  - Bijection transformer: 2.67M parameters, 6 layers, 6 heads, d_model=192, d_ffn=768.
  - HMM transformer: 2.68M parameters, 9 layers, 8 heads, d=256.
  - Regression transformer: 857k parameters, 4 layers.
  - Baselines: Mamba (3.77M), LSTM (4.77M) and "capacity-matched" MLPs (18–20 layers, width 384–400, fed the whole flattened context).
- **Positional encoding.** "learned token embeddings, learned absolute positional embeddings, pre-norm residual blocks" (§3.5). These are causal sequence models, not set models.
- **Training.** AdamW, batch 64, fresh data every batch. Bijection: lr 1e-3, 150k steps. HMM: 3e-4, cosine schedule, 100k steps. Regression: 5e-4, 50k steps.
- **Metric.** Mean absolute error between model entropy and analytic entropy, in bits. KL and TVD are reported only in Table 1.

### Three most important findings (as claimed)
1. **The model's entropy matches the exact posterior's.** "small transformers reproduce exact Bayesian posteriors for filtering and hypothesis elimination with 10−3–10−4 bit accuracy, while capacity-matched MLPs fail by orders of magnitude" (Abstract). Table 1 reports KL(model‖Bayes) of 8.2×10⁻⁴ nats for new bijection keys and 1.1×10⁻⁴ for the HMM. The HMM error grows smoothly past the training length: "MAE(K=30)=1.25×10−2, MAE(K=50)=2.88×10−2" (§4.2).
2. **Architectures differ in which "primitives" they realize.** Table 2 reports the following.

   | Model | Bijection (bits) | HMM (bits) | Recall |
   |---|---|---|---|
   | Transformer | 0.007 | 0.049 | 100% |
   | Mamba | 0.010 | 0.024 | 97.8% |
   | LSTM | 0.009 | 0.411 | 0.5% |
   | MLP | 1.85 | 0.40 | — |

   "Transformers realize all three primitives; Mamba realizes accumulation and transport but struggles with random-access binding" (Abstract).
3. **Internal geometry: "orthogonal key bases, low-dimensional value manifolds parameterized by posterior entropy"** (Abstract), plus "a single Layer 0 hypothesis-frame head whose removal is uniquely destructive" (§4.1). Fig. 15 caption: FFN layers "implement most of the numerical Bayesian update, while attention primarily provides routing".

### What "orthogonal key bases" and "value manifolds parameterized by entropy" mean operationally
- **Key orthogonality (§5.1, Fig. 16).** The paper uses the Layer-0 key vectors "for all input tokens in the bijection model at 150k steps", which means the K projection of each of the 20 input-token types. The statistic is the mean |off-diagonal cosine similarity| between those keys: "0.052 ± 0.004 versus 0.082 ± 0.003 for random vectors in d=192 dimensions—a 37% reduction (p<0.001, paired t-test)", and for the HMM model "0.061±0.006 vs. 0.079±0.002". The paper does not say whether the positional embedding is included or whether keys are per-head (d_head=32) or concatenated (192). How the random baseline was generated is not described.
- **Value manifold (§5.4, Fig. 18).** This is a "PCA projection of attention outputs in the bijection model, colored by analytic posterior entropy". At 100k steps the low-entropy states cluster. At 150k "they lie along a smooth one-dimensional curve parameterized by entropy". The caption contains an unedited author note: "Each point is an attention output (head output or block attention output – whichever you used)". So even the paper does not say which vectors were plotted. The layer is described only as "final layers". The figure is a qualitative picture. Paper I gives no variance-explained number and no fit statistic for the transformer.
- **Fig. 15 (FFN vs attention).** For each block, a linear probe predicting analytic entropy is trained on the residual stream *before* a sublayer and then evaluated, unchanged, *after* it. The plotted quantity is ΔMSE. The caption contradicts itself on sign: "negative values mean the block improves an entropy-linear probe. Positive values indicate that the block reduces probe error." The task is not stated, and the main text gives no numbers.

### Critical assessment
- **The key-orthogonality baseline looks wrong, and with a correct baseline the claim disappears.** [check] For isotropic random unit vectors, E|cos| ≈ √(2/(πd)). That is 0.058 at d=192 (my simulation with 20 vectors gives 0.0585) and 0.050 at d=256 (0.047–0.052). The paper's "random" 0.082 corresponds to d≈95, and 0.079 to d≈100. Against the correct baseline, bijection keys (0.052) are only about 10% more orthogonal than random, and HMM keys (0.061) are *less* orthogonal than random in 256 dimensions. Twenty vectors in 192 dimensions are near-orthogonal at initialization anyway. The finding needs comparisons against the untrained network and a correctly specified random baseline, plus a causal test such as forcing keys to correlate and measuring the damage. None of these is given.
- **In the bijection task, "entropy" is a function of position, so the value manifold is confounded.** With non-repeating inputs, H_Bayes(k)=log2(V−k+1) is a function of position k only. With learned absolute positional embeddings, a 1-D curve "parameterized by entropy" cannot be told apart from a curve parameterized by position. There is also an internal contradiction: §3.1 says "inputs never repeat", but Table 1 and Fig. 4 analyze "repeated keys" and entropy that "collapses to near zero when an input repeats". Either way, entropy is a trivial function of (position, repeat flag). The real evidence of elimination is the KL on new keys (8.2×10⁻⁴ nats), not the manifold picture.
- **Fig. 15 cannot support "attention only routes".**
  - A frozen probe measures whether a sublayer writes along the probe's existing direction. It does not measure whether the sublayer computes. Writes in new directions, or norm changes from LayerNorm interplay, move the score without any "computation".
  - No retrained probes and no causal ablation of the FFNs vs attention on calibration are shown.
  - The routing/computation split is ill-defined: an attention output is a weighted average of values, and averaging is a numerical computation (counting, pooling).
  - The paper's own Fig. 11 shows late attention is needed for long-horizon calibration. Removing it raises MAE to 0.555 at K=30 and 1.79 at K=50.
- **The baselines are weak, so the "primitive" taxonomy is unsupported.**
  - The MLP gets the whole context yet reaches 1.85 bits on bijection, where tracking the set of seen outputs is trivial. That points to a hard-to-optimize 18–20-layer MLP over a flattened one-hot sequence, not a principled impossibility. §4.4 also says MLPs "process each position independently", which contradicts §3.5.
  - HMM filtering is a fixed-dimensional recursion (a 5-dim belief), which is the textbook case for an RNN. An LSTM at 0.411 bits (≈ the MLP's 0.40) suggests failure to learn the header-conditioned update under this protocol, not a lack of "transport". The authors concede: "Our primitives conclusions characterize what each architecture achieved under standard training, not absolute capability limits" (§7.4).
  - An LSTM at 0.5% on 16-pair recall is likewise a training outcome, not an architectural fact.
- **Internal inconsistencies.**
  - Bijection MAE: 3×10⁻³ (§4.1), 0.007 (Table 2), "10⁻³–10⁻⁴" (abstract).
  - HMM MAE: 7.5×10⁻⁵ for the "2.68M" model vs 0.049 for a "2.7M parameter-matched" model, a 650× difference between nearly identical sizes.
  - Mamba HMM MAE: 0.024 (Table 2), 0.027 (§7.4), 0.031 (Paper II §9.4).
  - LSTM HMM MAE: 0.411 vs 0.416.
  - Recall pairs: 16 (§3.3) vs "64 cue–target pairs" (§4.6).
  - HMM training target: §3.2 says supervised on p(s_t|o_1:t), but §2.4 evaluates next-observation entropy.
  - Training data: "training uses only 10^5 samples" vs 150k steps × 64 fresh sequences ≈ 9.6M.
  - "This error is smaller than single-precision numerical noise" (§4.1) is false for 3×10⁻³ bits.
- **Overclaim.** "the first empirical proof that neural sequence models can realize exact Bayesian posteriors" ignores PFNs (Müller et al. 2022) and work the paper itself cites (Panwar et al., Reuter et al.). What is supported: small transformers fit these posteriors to within 10⁻²–10⁻⁴ bits, including some length extrapolation, and a few heads and blocks are causally important.

### Implications for the set-PFN dedup study [ours]
- Report key geometry against the isotropic expectation √(2/(πd)) *in the same dimension* and against the network at initialization. Near-orthogonality of fewer items than dimensions is free.
- In a set model the analogue of the position confound is context size n: posterior entropy falls with n. Any "manifold parameterized by posterior entropy/precision" must be shown after regressing out n, and ideally on worlds where entropy and n are decoupled (duplicates add items without adding information).
- Use retrained probes plus causal interventions (mean-ablation, attention forcing). Do not use frozen-probe deltas. In a set model, aggregation by attention averaging *is* the numerical update, so a "routing vs computation" split should not be assumed.

---

## B. Agarwal, Dalal & Misra, "Gradient Dynamics of Attention: How Cross-Entropy Sculpts Bayesian Manifolds" (Paper II), arXiv 2512.22473 v5

### Setup
- **Theory.** First-order gradients for a single softmax attention head with linear readout and cross-entropy (§2–3). No residual connections, no LayerNorm, no Adam.
- **Toy simulation 1.** T=5, d_x=3, d_k=2, d_v=2, C=3, "Targets y_i are random in {0,1,2}", 100 manual steps.
- **Toy simulation 2 ("sticky Markov chain").**
  - 8 symbols, self-transition 0.3, T=2000.
  - Input x_t = μ_{y_{t−1}} + N(0,I) in R²⁰.
  - One single-layer, single-head attention block (d_k=10, d_v=15), causal, 1000 steps.
  - Compared: SGD at lr 0.01 for all parameters vs an "EM-like" schedule with lr_V=0.1 and lr 0.01 for W_Q, W_K, W_O. 5 seeds.

### Three most important findings (as claimed)
1. **Advantage-style score gradient.** "∂L/∂s_ij = α_ij (b_ij − E_αi[b]), b_ij := u_i⊤ v_j" (Abstract, Eq. 23). Values update as "Δv_j = −η Σ_i α_ij u_i" (Eq. 33).
2. **EM interpretation.** "attention weights implement an E-step (soft responsibilities), while values implement an M-step" (Abstract). The authors qualify it: "this is an interpretive framework: we do not derive a surrogate objective with guaranteed monotonic improvement" (§5).
3. **The EM-like schedule converges faster.** Table 1 reports final loss 1.970±0.034 vs 2.058±0.007 and accuracy 0.244 vs 0.200, and "EM reaches SGD's final loss level in only 430 ± 143 steps—a 2.3× convergence speedup". The "theoretical Bayesian minimum (1.829 bits)" is reached by neither run.

### Critical assessment
- **The equations are correct, but they are just the softmax chain rule.** The score gradient is the standard softmax Jacobian applied to ∂L/∂α = u⊤v, so it is not a discovery. The "EM" mapping is an analogy with no objective, no fixed-point analysis and no prediction that a non-EM reading would not also make.
- **The "EM" experiment is a learning-rate comparison with no tuned baseline.**
  - The "EM-like" condition is SGD with a 10× larger learning rate on W_V. Beating untuned uniform SGD after 1000 steps says nothing about EM structure.
  - The missing control is SGD with a tuned learning rate, or the same 10× boost given to another parameter group.
  - Both runs end ≥0.17 nats/bits from the stated minimum. Loss and entropy units are mixed: the loss is presumably in nats, while the minimum is "1.829 bits".
  - The task does not need attention at all. x_t already encodes y_{t−1}, and a first-order chain only needs the previous symbol, so attending to self suffices.
  - In the first toy, random targets on 5 tokens can only be memorized, so "attention sharpens" there just means overfitting a fixed batch.
- **The links back to Paper I cite numbers that are not in Paper I.** §6.1 says that in the wind tunnels "PC1 explains 84–90% of variance, strongly correlated with posterior entropy (|r|>0.9)". These numbers do not appear in Paper I v5. I searched the extracted text: the only PC1 numbers are for Mamba (21.9%) and LSTM (92.3%). §6.2 repeats the key-orthogonality comparison against "0.08 for random vectors", which inherits the baseline problem above.
- **The "frame–precision dissociation" is asserted, not measured.** Paper I says attention maps "change little across checkpoints" but gives no metric. Paper II gives none either.
- **The dual-entropy section is unfalsifiable as written.** It says the advantage "is large precisely when H_I is high and H_P is low" (§6.4), but this is argued verbally, not derived or measured.
- **What it supports.** It gives useful bookkeeping identities, such as the compatibility matrix b_ij = u_i⊤v_j and the advantage matrix, that one could log during training. It is not evidence that training "sculpts Bayesian manifolds".

### Implications [ours]
- Cheap and possibly informative: during our PFN training, log attention-pattern drift (for example, mean JS divergence between checkpoints on a fixed probe set) alongside value/readout drift. This directly tests whether routing stabilizes before precision in our setting. Paper II's evidence does not settle it.
- A structural point for dedup that follows from these equations plus permutation equivariance [ours]. In a set transformer with no positional encoding, m exact copies of an item have identical representations at every layer, so they get identical scores, identical attention weights and identical gradients. The model can never treat "the first copy" differently from "the others". Its only way to dedup is to lower the score of *every* copy by about log m (or to rescale their values). That requires the copies' keys/queries to carry a similarity count, which an earlier attention layer must compute (attending to near-identical items). This is a concrete circuit to look for: an early layer whose output on an item encodes its multiplicity, and a later layer whose score for that item drops by about log m.

---

## C. Dalal, Misra & Parekh, "Bayesian Wind Tunnels for Model Selection" (Paper III), arXiv 2607.19379 v1

Note: this is not the "Paper III" cited inside Papers I/II (that one is "Geometric Scaling of Bayesian Inference in LLMs", 2512.23752, which I did not read).

### Setup
- **Tasks.**
  - Model selection between a structured class and all bijections on Z_n, with prior π=0.5.
  - Rotations x→x+c (n=16).
  - Fixed-point-free involutions (n=16). The input order interleaves "after every 2 fresh inputs … a previous output y_j as input".
  - Involutions vs 3-cycles, a non-nested comparison (n=12).
  - Scalar multiplication cx mod 17.
  - Polynomials of degree 1–5 mod 17.
- **Encodings.** "Integer tokens" vs "opaque" tokens, which are relabeled at random every episode. For rotations, opaque episodes include a header that gives the cyclic order.
- **Model.** The same 2.8M architecture as Paper I's bijection model: "6 layers, 6 heads, d_model = 192, d_ff = 768". Scaling runs at 25M (8L/8H/512), 152M and 316M.
- **Positional encoding.** Not stated in this paper. It is presumably Paper I's learned absolute embeddings, but that is unverified.
- **Training.** AdamW, lr 3e-4, cosine schedule, batch 64, 150K steps (300K for involutions). One curriculum "annealing π from 0.95 to 0.5 over 30K warmup steps".
- **Readout.** The class posterior is inferred from the model's mass on the structured prediction, P̂(R|D) = (p(y*) − 1/(n−k)) / (1 − 1/(n−k)) (Eq. 16). For involutions it is read only at reciprocal positions.

### Three most important findings (as claimed)
1. **Exact nested model selection, including with opaque symbols.** "a 2.8M-parameter transformer achieves 0.01-bit entropy agreement with the Bayesian optimum (3 seeds), with both integer tokens and opaque symbols whose meanings change every episode" (Abstract). Involutions reach 0.011 bits with integer tokens and 0.009 with opaque ones. Class-posterior MAE is 0.004 / 0.003.
2. **A "perceptual access condition".** Arithmetic classes succeed with integer tokens but "fails completely with opaque symbols, and this boundary persists under 112× scaling" (Abstract): rotations reach 0.12 bits with integers vs 1.28 opaque, and 1.20/1.19/1.17 at 25M/152M/316M. A header ablation leaves the opaque model unchanged at 1.55 bits, and a model trained on the composed header-plus-arithmetic subtask "reaches only 18.6% accuracy" (§4).
3. **A "polynomial barrier".** "No degree shows any learning" (§7.2). Affine maps reach 1.11±0.02 bits at 2.8M and 1.12±0.02 at 25M (8 seeds), against 0.12 for rotations and 0.03 for scalar multiplication.

### Critical assessment
- **The opaque-involution result follows from symmetry, so it is not evidence.** [ours, but follows from the paper's own definitions]
  - Under a relabeling σ, a pair (x, f(x)) becomes (σx, σfσ⁻¹(σx)).
  - Uniform-over-involutions, uniform-over-bijections and uniform-over-3-cycles are all invariant under conjugation.
  - Inputs drawn uniformly without replacement stay uniform, and the reciprocal-test schedule depends only on relational structure.
  - So per-episode relabeling leaves the distribution of training sequences *exactly unchanged*. For a model with randomly initialized learned embeddings, "integer" and "opaque" involutions are the same experiment, and 0.011 vs 0.009 (and 0.077 vs 0.076 in the non-nested case) is seed noise.
  - The paper half-acknowledges this for the non-nested case ("This is expected—both involutions and 3-cycles are label-invariant", §6.3), yet calls the opaque involution result "the central result of the paper" (§5.6).
  - The same argument applies to the "stationarity control". A *fixed* relabeling of token IDs is an exact symmetry of a learned embedding table. It should therefore equal the integer condition up to seed noise. Instead the paper reports 0.009 (3 seeds) vs 0.12 for integer rotations. A 13× gap between isomorphic conditions means either large run-to-run variance in the rotation runs or an unstated configuration difference (curriculum? steps?). Either way it weakens every single-run comparison, including "80K steps of plateau … characterized here from a single representative run".
- **More internal inconsistencies.** The 25M integer rotation model is reported at 0.008 bits (§4 scaling). The same section's extended 25M run is still at "0.097 at step 436,000".
- **"Non-nested selection beyond simplicity bias" is overclaimed.** By the paper's own §6.2, one reciprocal observation settles the class ("one-shot discrimination"), so the class posterior is exactly 0 or 1 at nearly every reported position. A class-posterior MAE under 0.001 is then close to guaranteed once the model can test f(y_i)=x_i, which is an induction-head lookup. Meanwhile the *entropy* MAE is 0.077 bits, 7× worse than the nested case. That is not "near-exact" (the abstract does not quote it).
- **"Arithmetic depth" is a doubtful explanation of the barrier.**
  - Scalar multiplication succeeds (0.03 bits) even though recovering c from one pair needs modular division (c = y·x⁻¹). Meanwhile affine maps are said to fail because "Modular inversion is a qualitatively harder circuit" (§7.2).
  - [ours] With learned token embeddings, a per-pair invariant g(x,y) (y−x, or y·x⁻¹) is just a 17×17 lookup. Rotation and scalar-multiplication membership reduces to "do all observed pairs share the same g?". That is pairwise and attention-friendly.
  - Affine and higher-degree classes have *no* per-observation invariant. Checking them needs a relation over ≥3 observations.
  - A sharper hypothesis is arity (does a per-item invariant exist?), not "depth". It is testable: affine maps with a fixed known slope have a per-pair invariant and should be learnable.
  - The authors themselves note the curriculum used to rescue rotations "were not applied to the affine case" (§7.2), so "learnability boundary" is premature.
- **The LLM comparison is not a calibration test.** The prompt says "f is a bijection" and never tells the LLM about the 50% involution prior, yet it is scored against the Bayes posterior under π=0.5. The probes use top-5 logprobs with the remaining mass spread uniformly, or 20 samples per position, over 50–100 sequences. The "55× gap" mixes prior mismatch with readout error. The authors partly concede this ("not an apples-to-apples comparison", §6.4).
- **What is solid.**
  - The closed-form posteriors (Props. 1–5).
  - Opaque *rotations* fail with a correct header present: the header ablation changes nothing (1.55 bits in all three conditions), and attention to the header in layers 2–7 sits at the uniform baseline (0.358 vs 0.360).
  - The composed header-plus-arithmetic subtask stalls at 18.6% while parsing alone reaches 100%.
  - Affine detection fails for both 2.8M and 25M models under this protocol (8 seeds).

### Implications [ours]
- **Exact-duplicate dedup is "relational" in their sense**: it is an identity match between items, invariant to relabeling. On their evidence attention should learn it smoothly, like the involution reciprocal test.
- **Correlated-but-not-identical evidence is the risky case.** For example, sources sharing a latent cause that shows only in joint statistics of ≥3 reports. That is the analogue of their affine/polynomial failures. A good world ladder is: exact duplicates → near-duplicates (pairwise similarity suffices) → redundancy detectable only through ≥3-way structure. Then measure where learning breaks, with ≥3 seeds per rung.
- **Run a symmetry audit before claiming an encoding result.** If the world generator is invariant under a relabeling, do not count "opaque vs plain" as two results.

---

## D. Biloš, Wilson, Schneider & Nevmyvaka, "A Mechanistic Study of Tabular Foundation Models", arXiv 2605.21288 v1 (Morgan Stanley)

### Setup
- **Models (all released checkpoints, inference only, except a TabICLv1 retrain).**
  - TabPFNv2 (release 7.0): 7.24M parameters, 12 blocks, d=192, "6 feature + 6 item" heads. Per-cell tokens with alternating row (across samples) and column (across features) attention. A learned per-feature positional matrix W∈R^{192×48} plus bias b. Features grouped in pairs. No row positional encoding.
  - TabICLv2: 27.55M parameters. A column-embedding SetTransformer compresses each row to one token, followed by a 12-block ICL transformer. RoPE along the feature axis.
  - Mitra (AutoGluon): 75.67M parameters. A shared cell embedding, a separate label slot, and "no positional encoding on either axis".
- **Data.** 49 classification datasets (4 synthetic, 3 sklearn, 42 OpenML) and 10 regression datasets. 80/20 context/query splits, 5–20 seeds.
- **Methods.** Per-layer probes, block and head knockouts, uniform-attention forcing, activation patching, "faithful surrogate" readout rules, invariance edits and 8 targeted attacks.

### Three most important findings (as claimed)
1. **TabPFNv2 predicts by an attention-weighted label vote at layer 9.**
   - "At layer 9 of TabPFNv2, row-attention sharpens abruptly: L9 is the sharpest layer on all 49 datasets … (mean maximum weight 0.925 …)". The vote "tracks the model's predicted probabilities at mean Pearson r=0.89" (§3.1).
   - "Replacing the L9 attention pattern with a uniform one collapses accuracy from 0.87 to 0.49 (the majority-class baseline)".
   - The vote alone reaches 0.804 accuracy vs 0.854 native (Table 1), with ECE 0.201 vs 0.043 (Table 85).
   - The similarity is learned, not raw distance: L9 attention is explained with R² 0.218 by input ℓ2 distance, 0.477 by L8 representation distance and 0.600 by the full basis (Table 80).
2. **Different backbones read out differently.** TabICLv2 predicts by "a nearest-prototype readout in the final representation": 0.854 vs 0.864 native. Swapping readouts costs 33–40 pp: TabPFN with a prototype readout drops to 0.523, and TabICL with a vote readout drops to 0.470 (Table 1). Mitra's vote reaches 0.815 vs 0.860, which the authors say "does not fully meet the faithful-surrogate bar".
3. **The models largely ignore their symmetry-breaking parameters.**
   - Setting TabPFNv2's W=0 leaves accuracy at 0.854→0.854 and gives exact within-group column invariance. Removing TabICLv2's RoPE costs −0.1 pp.
   - Hand-built one-layer models prove a 0.75 "multiset bound" for column-invariant row representations (Lemma 1).
   - Collapse only shows when both within-row defenses are removed: TabICLv2 then falls to the majority baseline on a d=12 stress set (−50 pp, Table 79).

### Critical assessment
- **The vote is a necessity result plus a correlational surrogate. It is not a mechanism.**
  - Uniform-L9 forcing shows L9 attention is *necessary*. But the paper also reports that "MLP knockout at any block also collapses accuracy to the same vicinity (§E.5): L9 attention is one part of a working circuit" (Table 44: 0.874→0.415).
  - This conflicts with Table 13, where single MLP knockouts at blocks 1–4 cost ≤1.6 pp on a retrained probe. Probably Table 44 knocks out *all* MLPs, but it is ambiguous.
  - The surrogate loses 5 pp of accuracy and 5× in ECE.
  - What computes the L8 similarity, which is the interesting part, is not explained.
  - Heads disagree. Per-head vote correlations include H1 = −0.432 on credit-g, H4 = −0.624 on ilpd and −0.400 on diabetes (Table 18). Averaging over heads hides heads doing something other than voting.
  - Pearson r on probabilities is inflated on easy binary tasks, where both probabilities sit near 0 and 1.
- **The activation patching is tautological.** Every TabPFN block patch has "recovery = 1.000" (Table 20). Patching a full block's output from the clean run makes everything downstream identical, as the authors say themselves for regression ("once the corrupted state is overwritten … the score follows", §E.3). This localizes nothing.
- **The probe numbers carry a red flag.** Mean probe accuracy at L0–L8 is 0.29–0.47 (Table 12), *below* the mean majority-class baseline of 0.499 (Table 82). A logistic probe below majority implies a train/eval shift. The protocol is also inconsistent: §2 trains on context rows and evaluates on *query* rows, while §A.2 evaluates on held-out *context* rows.
  - [ours] If probes train on label-bearing context rows and test on query rows, then the "L8→L9 jump" (0.467→0.780) partly measures when label information is written into query rows. That is the vote itself, not an independent finding.
- **The attack evidence for mechanism is weak.**
  - Hub poison costs TabPFN −3.9 pp, XGBoost −4.8 and MLP −2.8 (Table 65), with overlapping CIs. XGBoost is hit *harder*, so this does not isolate the vote readout.
  - Rank warp costs XGBoost −7.6 pp, although trees are invariant to any consistent monotone transform. Cube warp costs XGBoost exactly 0.0 [0.0, 0.0]. [ours] So the rank warp was evidently not applied as one consistent map across context and query (for example, ranks computed separately). It is a distribution-shift attack, not a "monotone warp" probe of the readout.
  - SVD-hide is defined two incompatible ways: "Signal injected … into test queries but not context" (§H.1) vs "hitting context and query consistently" (§H.10).
- **What is solid.**
  - The invariance edits: W=0 is free, RoPE removal is free in v2 but costs −15.4 pp in v1, and the One-vs-All wrapper is exact.
  - The multiset-bound lemma and the M0–M3 constructions.
  - The redundant-defense finding (Table 79).
  - The cross-backbone transplant asymmetry: a linear head nearly recovers TabICL (0.859) but not TabPFN (0.617).
  - The uniform-attention necessity at TabPFN L9.

### Implications [ours]
- **A vote readout predicts no deduplication, and the shape of the failure is specific.** Suppose one context item is copied m times with identical keys. Softmax then gives the copy group mass m·e^s/(m·e^s+Z): it grows with m and saturates. Neither Bayes target has this form.
  - If the copies are uninformative repeats, the target mass is constant in m.
  - If they are independent draws, the target log-odds are linear in m. A normalized average cannot produce unbounded log-odds without an extra mechanism, such as a sink or BOS token whose share encodes 1/n.
  - So a model that dedups, or that accumulates properly, must do work beyond a plain vote. Plotting group mass and output log-odds against m is a sharp, cheap test. It can be run on our model and on TabPFNv2 L9 directly.
- **Reuse their "faithful surrogate" bar,** which requires all four of:
  1. r≥0.85 with the model's probabilities;
  2. accuracy within 3 pp;
  3. κ≥0.8;
  4. survival under an intervention that keeps only the named components.
- Avoid whole-block patching, and use retrained-vs-frozen probes as they do in §B.1.3.
- **For a set model, Mitra is the right reference:** no positional encoding on either axis, row attention as the load-bearing path, exact invariance. TabPFNv2's late, sharp, single-layer retrieval is the alternative mechanism to look for.

---

## E. Rubachev, Kotelnikov, Kartashev & Babenko, "On Finetuning Tabular Foundation Models", arXiv 2506.08982 v2 (Yandex/HSE)

### Setup
- **Model.** TabPFNv2 (~7M parameters; the paper gives no architecture details or positional-encoding description).
- **Finetuning compared.** Full finetuning, LoRA, last layers, LayerNorm+head+embeddings, and added numerical embeddings.
- **Protocol.** The learning rate is tuned over 10 values in [5e-6, 5e-4]. Each step uses 1024 objects for the loss, with the rest as the prompt. Early stopping checks every 10 steps, with patience 16.
- **Data.** 21 academic datasets (Grinsztajn et al.; Gorishniy et al.), mean ≈15K rows, plus 8 subsampled TabReD datasets.
- **Analysis.** "last layer" attention from test objects to training objects.

### Three most important findings (as claimed)
1. **Full finetuning is the practical default.** "Our findings establish full finetuning as the most practical solution for TabPFNv2 in terms of time-efficiency and effectiveness" (Abstract). The gap to the PEFT variants is "minimal" (§3).
2. **Finetuning sharpens the similarity used for retrieval.** "after gradient-based adaptation, the dot products of the query-representations of test objects and the key-representations of in-context training objects more accurately reflect their target similarity" (Abstract).
   - The evidence is Table 3: an attention-weighted average of training targets using last-layer weights improves after finetuning.
   - CA 0.473→0.407, HO 3.404→3.313, DI 0.183→0.155, CH 0.856→0.866, AD 0.856→0.872, pol 6.845→3.165.
   - "The test scores are calculated on a single seed."
3. **Gains concentrate where attention sharpens, and real-world data is harder.** "the majority of finetuning performance gains come from samples where inter-sample attention becomes more sharply concentrated after finetuning" (§1; Figs 3–4). On temporally shifted TabReD subsamples, TabPFNv2 "is less stable and prior methods remain better" (Abstract).

### Critical assessment
- **The headline mechanistic claim is never measured directly.** No correlation between q·k (or attention weight) and target similarity |y_i − y_q| is reported before and after finetuning. The evidence is that a readout *of the model's own attention* improves when the model improves. Any finetuning that improves the model is likely to improve that readout too.
- **Controls are missing.**
  - Attention-kNN from a model finetuned on a different dataset.
  - Attention-kNN from L9 vs the last layer: Biloš et al. find the vote is sharpest at L9.
  - Which column's or head's attention is used. TabPFNv2 attends between items per feature cell, and the paper does not say whether the target column or a head-average is used.
  - Multiple seeds.
- **Figs 3–4 are correlational.** They sort test samples by entropy change and smooth the error change with a Gaussian filter, with no statistics. On House and Pol, entropy *increases* for 70–80% of samples. On Churn the pattern is absent, and the authors keep it "for transparency".
- **The "TabPFN as retriever" support is weak.** It rests on a dataset-level correlation of 0.89 between TabPFNv2's and ModernNCA's gains over an MLP, vs 0.53 for MLP-PLR.
- **What is solid.** The benchmarking:
  - Full finetuning beats in-context use on most academic datasets.
  - The ensemble of 5 finetunes ranks best (mean rank 1.05).
  - Finetuning helps more on larger datasets (Fig. 2).
  - Instability on TabReD, for example Sberbank, where FULL degrades in every split.

### Implications [ours]
- This supports, weakly, the picture that TabPFN's in-context prediction is similarity-weighted retrieval whose kernel lives in late QK. For dedup, the question becomes whether q·k (or the attention mass per item) changes with an item's *redundancy*, not only its similarity.
- A direct test in our model is the per-item attention mass on a duplicate group vs m, as in D. Report the q·k–target-similarity correlation explicitly rather than inferring it from readout accuracy.

---

## F. ARC (Alignment Research Center), "AlgZoo: uninterpreted models with fewer than 1,500 parameters"

**Sources.** Blog by Jacob Hilton, 2026-01-26: https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/ . Repo: https://github.com/alignment-research-center/alg-zoo . Alignment Forum cross-post: https://www.alignmentforum.org/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters

### Definition of a successful explanation (exact quotes, blog)
- "given a model that has been optimized for a particular loss, an "explanation" of the model amounts to a mechanistic estimate of the model's loss. We evaluate mechanistic estimates in one of two ways. We use surprise accounting to determine whether we have achieved a full understanding; but for practical purposes, we simply look at mean squared error as a function of compute, which allows us to compare the estimate with sampling."
- "By "mechanistic", we mean that the estimate reasons deductively based on the structure of the model, in contrast to a sampling-based estimate, which makes inductive inferences about the overall performance from individual examples."
- **Matching sampling:** "Our matching sampling principle is roughly the following conjecture: there is a mechanistic estimation procedure that (given suitable advice) performs at least as well as random sampling in mean squared error for any given computational budget."
- **Surprise accounting:** it "asks: how surprising is the model's actual accuracy, now that we have access to the mechanistic estimate?" Full understanding is "a mechanistic estimate with as few bits of total surprise as the number of bits of optimization used to select the model."
- **The challenge, verbatim:** "Challenge: Design a method for mechanistically estimating the accuracy of the 432-parameter model M_{16,10} that matches the performance of random sampling in terms of mean squared error versus compute. A cheap way to measure mean squared error is to add noise to the model's weights (enough to significantly alter the model's accuracy) and check the squared error of the method on average over the choice of noisy model."
- Footnotes add that "the method should match sampling on average over random seeds", that "a better but more expensive way … is to instead average over random seeds used to train the model", and that a good method should also work on `zoo_2nd_argmax(16, 10, seed=0..4)`.
- **The bar:** "For most of the larger models from AlgZoo (including the 432-parameter model M_{16,10} …), we would consider it a major research breakthrough if we were able to produce a mechanistic estimate that matched the performance of random sampling".

### What is in AlgZoo (README and figures)
- **Overall.** "tiny RNNs and transformers trained to perform simple algorithmic tasks", ranging from 8 to 1,408 parameters. Training used 2^30 sequences, with checkpoints at 0, 2^20, …, 2^29. Each zoo model is "the best of 5 random seeds".
- **2nd argmax.** 1-layer ReLU RNNs with no biases: h_{t+1}=ReLU(W^hh h_t + W^hi x_t), logits = W^oh h_n, on standard Gaussian inputs. Hidden sizes {2,3,4,5,6,8,16,32} × sequence lengths {2,3,4,5,6,8,10}. Examples: M_{2,2} (10 parameters, error ≈1/13,000), M_{4,3} (32 parameters, 98.5%), zoo M_{16,10} 94.8%, M_{32,10} 99.4%. The challenge model `example_2nd_argmax()` (432 parameters, 95.3%) is "not officially part of the zoo".
- **Argmedian (RNN).** Hidden {2,4,8,16,32} × length {3,5,7,11}.
- **Median (RNN, RMSE).** Hidden {2,4,8,16,32} × length {2,3,5,10}.
- **Longest cycle of f:{0..n−1}→{0..n−1} (transformers).** Hidden {2,3,4,6,8} × n {3,4,5,6}.
- Handcrafted 2nd-argmax models are included: lengths 2 and 3, and a 22-neuron length-10 model with 99% accuracy.

### ARC's own status
"The largest model that we believe we more-or-less fully understand has 32 parameters; the next largest model that we have put substantial effort into, but have failed to fully understand, has 432 parameters." For M_{4,3}: "we are able to roughly match the computational efficiency of sampling". For M_{16,10} they identified "leave-one-out-maximum" feature subcircuits (neurons 1, 2, 4, 6, 7), but "we not have not turned this understanding into an adequate mechanistic estimate".

### Results or solutions
I found no ARC announcement or accepted solution. Checked 2026-10-26: the repo has 60 stars, 6 forks and one issue (#1, 2026-10-19, "How was example_2nd_argmax() trained?"). Third-party attempts exist, all unreviewed and none accepted by ARC:
- **github.com/panaali/beyond-sampling.**
  - Its RESULTS.md states "ARC's open challenge … is **not met**. After about 30 attempts, every deductive method loses to sampling (8x to >300x)".
  - Its later STATUS.md (2026-10-19) claims a "line walks" estimator gives "about half [sampling's] error from about a million runs per model on a sealed … panel", while noting "it stays a quadrature-type method: no bound from the weights on its averaging error". It also claims about 50× lower squared error than sampling on M_{4,3}.
  - These are self-reported. The repo looks agent-run (it contains AGENTS.md and CLAUDE.md).
- A gist by mrsirrisrm, "(Almost) mechanistic model" of M_{16,10}, and github.com/mogar/algzoo_interp, which targets M_{32,10}. I did not verify either.

### Implications [ours]
- ARC's bar is an *estimate of the model's loss from weights that beats sampling at equal compute*, averaged over noisy-weight or seed variants. Surrogate readouts that reproduce 89% of variance, as in A–E, are nowhere near this bar.
- ARC has not met the bar at 432 parameters, so it is unrealistic for a whole ~10⁵–10⁶-parameter PFN. A defensible version for us would apply it to (a) a deliberately tiny dedup model, 1 layer and 1–2 heads, or (b) one quantity, such as the model's mean KL to Bayes on a family of duplicate-count inputs. That quantity would be predicted from the attention/value weights and compared with Monte-Carlo estimation at matched FLOPs, over noise-perturbed weights.

---

## Bottom line for the planned small-PFN dedup / aggregation study [ours]
1. **The most testable prediction from this literature.** Plain softmax-attention voting, which is what TabPFNv2 L9 appears to do, counts duplicates with saturating weight m·e^s/(m·e^s+Z). That matches neither "copies are uninformative" (constant) nor "independent evidence" (log-odds linear in m). So any correct dedup or accumulation needs an identifiable extra mechanism: copy-marking keys, a count or sink channel, or MLP correction. Plot attention group mass and output log-odds against m, against the exact Bayes curves under both priors.
2. **Design the world ladder to separate pairwise from higher-order redundancy.** Paper III's real (if mislabeled) finding is that per-item and pairwise relations are learned easily while relations needing ≥3 items are not. Exact and near duplicates should be learnable. Redundancy visible only in ≥3-way structure is where a learnability failure, and the more interesting mechanism, is likely.
3. **Methodological guards learned from these papers.**
   - Use the correct random baseline in the same dimension and compare with the network at initialization.
   - Control for context size n before calling anything a "posterior manifold".
   - Use retrained probes plus interventions, not frozen-probe deltas or whole-block patching.
   - Use ≥3 seeds per comparison.
   - Run a symmetry audit before claiming encoding-invariance results.
   - Use Biloš et al.'s four-part faithful-surrogate bar, and reserve "mechanistic explanation" for ARC's estimate-beats-sampling standard.
