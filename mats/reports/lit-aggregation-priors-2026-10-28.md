# Literature sweep: in-context aggregation, dedup and prior worlds (2026-10-28)

For: Kelvin (MATS, PFN aggregation/dedup program).
Inputs: research and adversarial-verifier passes on five topics: PFNs and amortised Bayes; ICL vs in-weights learning and plateaus; counting and length generalisation; mechanisms of in-context regression; structured bandits and similarity. The workflow announced 8 topics, but the text that reached this step was cut off partway through topic 5. For topic 5 the findings arrived without the verifier block, and topics 6-8 did not arrive at all. Anything from topic 5 is therefore marked U. I did not re-read any source myself.

Status tags:
- **V**: the verifier confirmed it (abstract or full text).
- **V\***: verified, with the verifier's nuance applied.
- **C**: the researcher overstated it and the verifier corrected it; the corrected version is what is written here.
- **U**: not verified. This covers all of topic 5, plus items the verifier listed as "missing work" that nobody checked.
- **I**: our own inference, not a published result.

Items the verifier marked not-found were dropped. That includes the "independent review" of the wind-tunnels paper and the "Wu et al." length-transfer claim.

---

## 1. What is already established

| Result | Source | Status |
|---|---|---|
| Softmax attention without positions or a sink depends only on token proportions: inputs "10" and "1100" get identical representations, so absolute N and uniform duplication are invisible. | Barbero et al. 2024, arXiv:2406.04267, Prop. 6.1/B.9 (proof ignores MLP and norm) | V |
| Uniform copies cancel in softmax attention. | Liu et al. 2026, arXiv:2610.16801, App. A.5 | V |
| In TabPFN v1, duplicating both classes uniformly has no visible effect. Duplicating one class shifts predictions non-locally. | McCarter, arXiv:2502.08978 | V |
| An encoder-only PFN gives a flat prediction on all-heads coin contexts: 1 head and 10 heads look the same. The authors propose add_zero_attn, or TabPFN v2 noise features. Shown for one best-loss model picked from a grid search, not across the whole grid. The paper's own small martingale check is roughly satisfied. | Mueller et al. 2025, arXiv:2505.23947 | C |
| Production tabular PFNs over-count repeated measurements of the same site. Under an IID prior that couples p(X) and p(Y\|X), odds are inflated by Lambda^(r-1). In TabPFN-v3, part of the harm runs through a row-count input (QASS). Resetting that count to the number of distinct sites removes low-noise harm but also removes high-noise gains. Attention transplant fixes only 18 of 40 harmed cells. Per-site means work best. Error is 1.6-5.3x at 256 readings, on the quadratic task for 5 of 6 models. | Liu et al. 2026, arXiv:2610.16801 | V\* |
| TabPFN v2 adds a per-row N(0,1) "unique sample identifier" so that duplicate rows can be told apart. It trains on ≤2,048 rows and is evaluated to 10k, with an explicit caveat about going further. For 2.5 the pretraining row count is not reported, so we cannot say how far it extrapolates. | Hollmann et al. 2025 (Nature); arXiv:2511.08667 | C |
| PFN amortisation error is small but concentrated on inputs that are rare under the prior. The N distribution used in pretraining decides where the error falls: uniform N puts it at small n, log-uniform at large n. Independently trained PFNs disagree on 4-9% of predictions. That disagreement flattens beyond about 1e4 pretraining datasets and shrinks only slowly with parameters. | Huelskamp et al. 2026, arXiv:2610.13764 | V |
| PFN transformers guarantee that variance vanishes with n. Bias vanishes only if the predictor is localised. | Nagler 2023, arXiv:2305.11097 | V (abstract) |
| If context actions depend only on history plus exogenous randomness ("compliance"), DPT equals posterior sampling. Compliant pretraining distributions with the same support give the same model (Prop 6.4). Expert-biased, non-compliant pretraining gives LCB-like behaviour. The regret constant is a ratio of test to pretraining task distributions, not context distributions. | Lee et al. 2023, arXiv:2306.14892 | C |
| ICRL models (DPT, DPT', GDPT) fail sufficient-statistic invariance: Sort (reorder) and Replace (resample rewards, keep means). DPT' and GDPT also fail martingale tests in most settings, even when their regret looks TS-like. Making the model permutation-invariant fixes only Sort. p-values do not shrink monotonically with n. | Chen et al. 2026, arXiv:2610.28977 | C |
| Small transformers trained on fresh-latent tasks with exact posteriors (bijection elimination, HMM filtering) match the posterior to 1e-3 to 1e-4 bits. Capacity-matched MLPs fail. Mamba lacks "binding"; LSTMs only accumulate. | Agarwal, Dalal, Misra 2025, arXiv:2512.22471 | V |
| Task-diversity threshold. Below it, the model acts as Bayes over the finite set of training tasks (memorises). Above it, the model acts like ridge regression. | Raventos et al. 2023, arXiv:2306.15063 | V |
| PFNs near-match GP posteriors. PFNs4BO handles irrelevant dimensions by injecting them into the prior (30% in HEBO+), with ARD hyperpriors integrated out. | Mueller et al. 2022, arXiv:2112.10510 (V); Mueller et al. 2023, arXiv:2305.17535 | V / U for the 30% detail |

**Not found anywhere.** None of the three sweeps that looked found a paper on in-context deduplication of copies vs fresh measurements under an exact-Bayes target. None found a circuit-level account of how a PFN counts or discounts copies; 2505.23947 lists counting as an open problem. Before claiming novelty in print, check Yan et al. 2023, *Understanding ICL from Repetitions* (arXiv:2310.00297), which the verifier flagged as missing and nobody checked (U).

## 2. Priors others used, and what they showed

| Prior | Used by | What it showed | Status |
|---|---|---|---|
| Coin, p uniform on {0.01..0.99}, 1-100 flips | Mueller et al. 2025 | A set-PFN cannot count identical observations | C |
| r readings per site, Y = f(X) + noise, GP-like or quadratic f | Liu et al. 2026 | IID-row priors over-count repetition; per-site means work best | V |
| SCM tables (about 130M datasets, ≤2,048 rows) | TabPFN v2 | Random row IDs; extrapolation to about 5x rows (accuracy, not Bayes-correctness) | V |
| BNN label prior on TabICL covariates, MCMC reference | Huelskamp et al. 2026 | Error concentrates on rare inputs; the N schedule controls where it falls | V |
| GP with hyperpriors; HEBO-like GP with injected irrelevant dims | PFN 2022, PFNs4BO 2023 | Near-GP posteriors; ARD-like relevance inferred in context | V / U |
| 5-arm bandit, mu ~ U[0,1], sigma 0.3, Dirichlet-mixture behaviour policies | DPT 2023 | Policy invariance under compliance; non-compliant pretraining changes the estimator | V |
| Random bijections; HMMs with fresh (T, E) each sequence | Wind tunnels 2025 | Exact-posterior match; attention routes, FFN updates | V |
| Class selection over opaque per-episode relabelled symbols | Dalal et al. 2026, arXiv:2607.19379 | Relational/equality classes work. Arithmetic on opaque symbols fails across a 112x scale-up. Fixed relabelling fixes it | V |
| Finite pool of M linear tasks vs Gaussian prior | Raventos 2023; Carroll 2025; Wurgaft 2025 | Memorise vs generalise threshold; ridge appears transiently, then the model specialises | V |
| Mixed per-sequence noise levels in linear regression | Bai et al. 2023; Vladymyrov et al. 2024 | In-context algorithm selection; noise-adaptive step sizes. 4 layers beat adaptive ridge only at high noise | V\* |
| Tree-structured non-Gaussian latents (linear regression, generalised linear attention) | Raj & Reddy 2026, arXiv:2606.16694 | One layer is optimal for Gaussian priors; depth helps for tree-like priors | C (narrow setting) |
| Mixture of transition distributions with hidden per-token weights | D'Angelo & Flammarion 2026, arXiv:2604.10848 | Mirror descent in context; 3 layers per step | V |
| Bursty Omniglot; Zipf class frequencies | Chan 2022; Singh 2023/2025; Reddy 2024 | Data properties drive ICL vs IWL; transience; coopetition | V |
| Gaussian items, random labels, K classes | Nguyen & Reddy 2025, arXiv:2412.00104 | Exponential init sensitivity; bimodal seeds near threshold | V |
| Permuted-label MNIST | Kirsch et al. 2022, arXiv:2212.04458 | Plateaus; three regimes; cheap fixes | V |
| Query-count / histogram over vocabulary m, length n | Yehudai et al.; Behrens et al. | Counting constructions and limits | V / C |
| Bernoulli regions split by delimiters | Golkar et al. 2024, arXiv:2406.02585 | Models adopt any fixed-count token as the softmax bias | V |
| Fresh Boolean function per episode (conjunctions, DNF, majority, parities) | Bhattamishra et al. 2024, arXiv:2310.03016 | Correlated classes easy. Parities stay at chance, with no training progress. Teaching sequences fix it | U |
| Random k-juntas with noisy features | Peters et al. 2026, arXiv:2602.08695 | Models get trapped at a lower-sensitivity noise-optimal predictor | U (abstract) |

## 3. Known mechanisms; depth and width

**Primitives**
- Linear attention can implement one GD step on an in-context least-squares loss, and deeper trained models track GD++ (von Oswald 2023, arXiv:2212.07677; V).
- Softmax GPT-2 layer predictions match iterative Newton, about 3 steps per middle layer (Fu et al., arXiv:2310.17086; V).
- Transformers can select between algorithms in context (Bai et al., arXiv:2306.04637; V).
- Softmax attention learns a nearest-neighbour window that narrows as n grows and widens with noise, and it projects onto low-rank task subspaces. Linear attention cannot do either (Collins et al., arXiv:2402.11639; V).
- GP regression in GPT-2-style models aligns across depth with kernel ridge via a preconditioned Richardson iteration (Yan, Li et al. 2026, arXiv:2605.08475; U).

**Duplicates under these primitives (I).** GD sums, NW kernels and mirror descent all weight a duplicate by its count. Dedup therefore needs an extra stage that counts and then reweights. One contrast with the Richardson result: the K^-1 solve in GP regression splits weight among collinear copies automatically. That would make dedup an iterative-solve quantity that improves with depth (I).

**Counting**
- A histogram over orthogonal embeddings counts exactly when d ≥ vocabulary m.
- CountAttend counts any vocabulary in 1 layer and 1 head when d > log m. The query attends to equal tokens and an MLP inverts 1/c, at a cost of MLP width Omega(n) and log n logits.
- So exact counting is not bounded by d_model, and this paper does not establish a 2-layer minimum for dedup.
- Source: Yehudai et al., arXiv:2407.15160 v3. Status C: the researcher's "groups > d_model" limit was corrected.

**The sink as a count channel**
- With a BOS token, the attention mass on BOS becomes a scalar proxy for count. Without BOS, softmax falls back to inventory counting (Behrens et al., ICML 2025, arXiv:2407.11542; C: d ≥ T is sufficient, not necessary).
- Models pick any token whose count is fixed as their softmax bias, and that choice determines which OOD axis they fail on (Golkar et al.; V).
- Sinks get stronger with training context length (Barbero et al., COLM 2025, arXiv:2504.02732; V).
- Hypothesis (I): with equal logits, sink weight e^b/(e^b+n) has the same form as conjugate shrinkage n/(n+kappa). A sink could therefore carry prior pseudo-count shrinkage. Untested.

**Depth**
- One layer without an FFN is optimal for Gaussian latents. Tree-like non-Gaussian latents need depth, which works by coarse-to-fine routing through function vectors (Raj & Reddy 2026). Status C: shown only for linear regression in generalised linear attention.
- Depth lower bounds of log kappa (unrestricted) or sqrt(kappa) (restricted). Deep non-looped models are fragile under O(e^-L) shifts; looped models are robust (Gatmiry et al., arXiv:2410.21698; V).
- Looped transformers match standard ones with under 10% of the parameters, and win mainly on sparse linear regression and trees. They need specific training to reach a stable fixed point (Yang et al., arXiv:2311.12424; C).
- Wind tunnels: one layer-0 "hypothesis-frame" head is critical, many later heads are partly redundant, and the frame forms before precision does (V).

**State**
- ICL quality tracks the accessible state size more than parameter count. LSTM reached 25% vs 74% for the transformer on MNIST (Kirsch 2022; V).

**Reading models out**
- In LLMs the count is linearly decodable even when the output is wrong: a late MLP overwrites it (Venkatesh 2026, arXiv:2605.09239; Ait Hou et al. 2026, arXiv:2609.22230; V).
- Matching posterior means does not identify the algorithm. Different posteriors can share the same means, and "OLS-like" models break on shifts that OLS ignores (Hill et al., arXiv:2507.09440; Aswadi et al., arXiv:2607.17060 PMC; V).
- Fit each checkpoint as a mixture of candidate exact strategies (Park et al., arXiv:2412.01003; Wurgaft et al., arXiv:2506.17859; V). Hidden-state collapse and attention entropy diagnose plateaus (Gopalani & Hu, arXiv:2506.13688; V).

## 4. Failure modes, with the sharpest prediction for our worlds

Predictions are ours (I) unless they restate the source.

| Failure | Source (status) | Prediction / test for us |
|---|---|---|
| Multiplicity is invisible to sink-free softmax | Barbero 2024 (V); 2610.16801 A.5 (V) | Under uniform duplication, a keyless set-PFN with no sink, BOS or count feature gives exactly the same output with and without the copies. Under non-uniform duplication it weights items by multiplicity. Adding a sink is the first ablation to run, crossed with 1 vs 2+ layers. |
| A count input causes over-counting | 2610.16801 (V) | A count or N token can make dedup worse if the prior does not draw each item once. The exact-Bayes target has to come from the generative story: draw items, then copies and fresh measurements. |
| Bias-token shortcut | Golkar 2024 (V) | Any structural count held fixed in training becomes the sink and breaks extrapolation on exactly that axis. This covers N, number of groups, copies per item, fresh measurements per item and separators. Randomise each one independently and check where the sink mass goes. |
| Sharp heads disperse as n grows (finite vocabulary or bounded inputs) | Velickovic et al., arXiv:2410.01104 (V\*) | A head that detects copies loses sharpness as N grows, while a diffuse averaging head does not. At 10x N_train, keyed dedup degrades first, and its attention entropy rises. Ablate with log-N or SSMax (arXiv:2501.19399; V) and ASEntmax (arXiv:2506.16640; V). None of these was tested on mean-like Bayesian aggregation. |
| Shrinkage tuned to training N | Zhang, Frei, Bartlett, arXiv:2306.09927, O(1/M + 1/N_train^2) upper bound (V) | With a narrow N range in training, shrinkage is miscalibrated at large N even when the mechanism is right in form. Train with log-uniform N and report calibration against Bayes as a function of N. |
| Positional encodings break length generalisation | Panwar et al., arXiv:2306.04891 App. A.2 (V) | Not a risk for set-style models without PE. Any large-N failure we see is then algorithmic or numerical, which makes it a cleaner mechanism probe. |
| TabPFN extrapolation does not transfer to our targets | Nagler (V); Chunked TabPFN, arXiv:2509.00326 (C: exact tiled attention; regression plateaus around 20k) | TabPFN's large-N wins are on targets that depend mainly on the empirical measure. Targets that depend on absolute n or non-uniform multiplicity (posterior variance, n/(n+kappa), dedup under a skewed policy) have no PFN extrapolation evidence. |
| Non-compliant policies | DPT (V) | If the policy sees hidden values (for example value-informed MCTS), sampling frequency is evidence, and an invariant estimator is not Bayes-optimal. Policy invariance is a valid target only for compliant policies, with training support covering the test policies. |
| Sufficient-statistic invariance fails even with set architecture | 2610.28977 (C) | Collapsing k known copies into one item, or reordering, should leave the output unchanged. Expect violations even in permutation-invariant models. |
| Plateau with exponential sensitivity to init; bimodal seeds | Nguyen & Reddy (V); Kirsch (V) | The 20-25% stuck rate may be the long tail of escape times, not an attractor. Run 50+ seeds, longer training, and escape-time histograms. Exactly balanced counts make failure more likely, so do not balance group sizes. |
| Removing the intermediate strategy can increase stuck seeds | Reddy 2024, arXiv:2312.03002 (C: with L=N, 4 of 6 seeds hit a sub-optimal minimum); Singh 2025, arXiv:2503.05631 (C) | "Devalue fixed shrinkage to unstick seeds" is not a known fix and could backfire. The simpler strategy may scaffold the full circuit. Test both directions. |
| Simplicity / low-sensitivity bias | Edelman 2024 (V); Hahn & Rofin, arXiv:2402.09963 (U); Peters 2026 (U) | Fixed shrinkage is the low-sensitivity solution. Before training, compute how sensitive the Bayes predictor is to single items in each world; higher sensitivity should mean a higher stuck rate. |
| Parity-like Boolean targets: plateau with no progress | Bhattamishra 2024 (U); Abedsoltan 2025 (U) | A junta world with XOR interactions will plateau a single-pass set-PFN. Use correlated (monotone, AND/OR-like) junta functions, or add an auxiliary "relevant bits" target. |
| Transience; L2 conflict | Singh 2023 (V: L2 helps persistence); Nguyen & Reddy (V: heavier weight decay on attention than MLP causes transience) | Measure dedup along the whole trajectory, not only at the end. Weight decay on attention vs MLP is a factor to sweep. |
| Fixes for plateaus | Kirsch: biased batch fraction, smaller Adam eps (more than halves the plateau), larger batch (V). Hoffmann: Heat Treatment / NormSoftmax (V). Kim et al.: task mixtures (V). Zucchet: within-context repetition helps; plateau ~ d^1.29 T^0.80 B^-0.80 for a 2-layer transformer (C) | Cheap things to try against the stuck rate before redesigning a prior: smaller eps, larger batch, a fraction of keyed or exact-copy episodes, and a mixture of worlds. Plateau length grows roughly linearly with context length, so a short-N curriculum is in tension with large-N generalisation. |
| Opaque per-episode symbols plus arithmetic | Dalal et al. 2026 (V) | Use random per-episode group keys only for equality matching. If a key has to enter numeric reweighting, expect a ceiling that scale does not fix. It is unclear whether our 80% with-keys result reflects this (I). |
| Seed disagreement floor | 2610.13764 (V) | Seed-to-seed disagreement is normal and measurable. A seed ensemble gives a diagnostic that needs no reference; break disagreement down by region of the prior (large N, many copies). |

## 5. Bounded-state architectures and evidence

| Architecture | Evidence | Status |
|---|---|---|
| Non-causal linear attention (TabFlex) | Scales to millions of rows. Causal and Mamba variants degrade with N. | V |
| Recurrent mixers in PFNs (LA, GLA, DeltaNet, Gated DeltaNet, Mamba-2) | Degrade beyond 2-4x pretraining context; non-causal softmax and non-causal linear attention stay stable. An oracle state at 128k rows beats softmax, so the limit is state drift, not capacity. Fix: write rate eta_t = c/(t+c) plus a final-state readout. A permutation-invariant ridge state also generalises. State passing helps GLA and hurts DeltaNet. (Schnurr et al. 2026, arXiv:2609.36337) | C |
| Bidirectional SSM (Hydra TabPFN) | Competitive, less order-dependent (arXiv:2510.14573) | V |
| Constant-memory attentive NP (CMAB) | Condition, query and update in constant memory (arXiv:2305.14567) | V |
| Longhorn; Mamba-1 | Longhorn extrapolates about 16x; Mamba-1 blows up past its training length | V |
| State passing / noise-state init | 2k to 128k for LMs (Buitrago Ruiz & Gu, arXiv:2507.02782); did not transfer cleanly to tabular PFNs | V |
| RNNs/LSTMs generally | No ICL under Chan 2022 data; LSTM at 25% vs transformer at 74% (Kirsch). Wind tunnels: LSTM only accumulates, Mamba lacks binding | V |
| Hyena, LSTM, DSS on Boolean ICL | Match transformers except on nearest-neighbour retrieval | U |

Sharpest points:
- The fix that worked, a 1/t write rate, is the running-mean or conjugate update rate (I: a reading, not the paper's claim). This suggests bounded-state models extrapolate when their update is a count-aware sufficient-statistic update.
- With explicit keys, dedup reduces to accumulating into per-key slots. With inferred similarity it needs binding, which is where Mamba and LSTM fail in the wind tunnels.
- Sum-decomposable set models need latent dimension at least the set size to represent arbitrary set functions (Wagstaff et al. 2019, arXiv:1901.09006; U, listed as missing). The state has to scale with the number of distinct groups, not with N.
- No study found tests a bounded-state model on per-group statistics when the number of groups grows with N.

## 6. What would be new, stated narrowly

Each item is marked as not found in the material reviewed. The three missing topics (6-8) could change this.

1. **Exact-Bayes in-context dedup.** A world whose Bayes target comes from an explicit story (items drawn once; copies vs fresh measurements vs related items), with a multiplicity-control pair: one item copied k times vs k fresh agreeing measurements. 2610.16801 studies repetition in production PFNs trained under the wrong (IID) prior. Its proposed fix, pretraining under a prior that draws each site once and adds count and variance inputs, is untested. Our program would be that test. Nearest unchecked prior: Yan et al. 2023 (U).
2. **A circuit account of counting and copy discounting in a set-PFN.** Is it a sink, CountAttend or an inventory? Is the count represented but mis-weighted? Listed as open in 2505.23947. Includes the untested question of whether zero-attn or a BOS sink fixes the coin failure and also improves copy discounting.
3. **Policy invariance as a measured property.** Hold the sufficient statistics fixed and vary a compliant exogenous policy systematically. Theory exists (DPT Prop 6.4) and sufficient-statistic tests exist (2610.28977), but a systematic policy-mismatch test does not.
4. **Bayes-correctness, not accuracy, at N far beyond training, for targets that depend on absolute counts or multiplicity.** TabPFN reports only accuracy, and on measure-like targets.
5. **Bounded-state aggregation with inferred similarity,** where the number of groups grows with N.
6. **Stuck-at-prior seeds in PFN training,** characterised as plateau vs attractor with escape-time statistics. The ICL-plateau literature is the analogue, but it uses classification and regression, not aggregation.

Not new; cite instead of rediscovering:
- softmax cannot see multiplicity;
- IID-prior PFNs over-count repeats;
- invariance under compliance;
- PFNs fit GP/ARD posteriors;
- seed bimodality and plateau fixes in ICL;
- the difficulty ordering of in-context Boolean classes;
- removing PE helps length generalisation.

## 7. Implications for choosing prior worlds

| Candidate world | Overlap with prior work | Verdict |
|---|---|---|
| **GP with ARD kernels** | Heavily covered: PFN 2022, PFNs4BO, Collins (projection), Richardson iteration (U). | Redundant as a main world. Keep it as a calibration control. It becomes novel only if copies or a skewed policy are added; the K^-1 solve explains duplicates away, which gives a natural dedup target. The DVA ~10-dim limit (arXiv:2509.20950) is about trainability at fixed dimension, not dimension extrapolation (verifier correction). |
| **Hierarchical item effects with copies and fresh measurements** | Closest to 2610.16801, but there the models were not trained under the correct prior. If Gaussian, Raj & Reddy (C) predict shallow depth suffices once copies are handled. | Primary world. Most direct test of item 1 in §6. It also cleanly separates "depth for counting" from "depth for non-Gaussian structure". |
| **Hidden junta (bit-vectors)** | In-context Boolean learning is covered (Bhattamishra, U); noisy juntas trap (Peters, U); parity plateaus (U). | Redundant if the question is only "can it learn a junta in context". Novel if the junta defines similarity for dedup and pooling. Use correlated or monotone junta functions, size C(n,k) above the memorisation threshold (Raventos; Lu et al., arXiv:2405.11751, sizing heuristic), hold out subsets, and resample relevance every episode. Similarity that stays fixed across episodes gets compiled into the QK weights (Collins). |
| **Context-tree weighting (hidden tree cuts)** | No PFN or ICL work found on CTW in the material received. Tree-structured non-Gaussian latents are where depth should matter (Raj & Reddy, C). | Novel, and a good depth test. Prediction (I): per-node KT estimates have the sink form and should be attention-native, while mixing over cuts needs absolute-count sensitivity and is the part that should fail to extrapolate in N. |
| **MCTS transpositions** | Classical engineering literature documents the over- and under-counting (Childs 2008; UCD 2012; Czech et al. 2021; KataGo graph-search notes, all U). The search's Q is policy-weighted by definition. | Hardest stress test (absolute counts plus dedup), so not a first world. The target must be fixed explicitly: the minimax/Bayes value (policy-invariant) or the visit-weighted Q (policy-dependent). A value-informed policy breaks compliance. Natural non-Bayes baselines are KR-UCT (count-weighted, not invariant), RAVE's beta and Sampled MuZero's beta-hat/beta correction (U). |

Tests that are novel or cheap and should be standard in every world:
- Multiplicity-control pairs: k copies vs k agreeing fresh measurements.
- Sink/zero-attn/BOS ablation × depth (1 vs 2+).
- Sufficient-statistic invariance (collapse known copies, reorder) plus the martingale check.
- Policy shift at matched sufficient statistics.
- N at 5-50x train with log-uniform training N, reporting bias (copy mis-weighting) separately from variance.
- Probes for n_g and 1/n_g.
- Strategy-mixture fits over checkpoints: prior shrinkage / naive pooling / copy-dedup Bayes / full Bayes.
- Escape-time histograms over 50+ seeds.

Design rules supported by the literature:
- Randomise N, group count, copies and fresh measurements independently (Golkar).
- Use random continuous per-episode keys, for equality only (Yehudai; Dalal).
- Keep policies compliant, with broad support (DPT).
- Randomise noise scale per episode (PreDeToR follow-up, arXiv:2510.00347, U).
- Do not balance group sizes exactly (Nguyen & Reddy).
- State in each world whether item identity or covariates are coupled to the value latent (2610.16801).

### Implications from topics 5-8

Sources and statuses for these bullets are in the addendum below.
- **CTW is no longer an unstudied world.** Zhou, Tian, Diggavi (arXiv:2410.05493, AISTATS 2026; V, full text) train transformers on a CTW prior with an exact Bayes baseline. 1 layer gives roughly unigram performance, 2+ layers track CTW and gains saturate at about 4, and they report no evaluation beyond the training window. Reuse their setup (ternary alphabet, D ≤ 5, stop probability 0.15, Dirichlet(0.5) leaves). What is still new: N well beyond the window, a set-style or no-PE variant, and copies added to the tree world. This supersedes "No PFN or ICL work found on CTW" in the table above.
- **Add the Gaussian partial-information world as the reference dedup world** (Ernst, Pemantle, Satopää, Ungar 2016; V, numbers recomputed). Its Bayes answer is closed-form, it runs continuously from copy (ρ=1) to fresh (ρ=0), and with ρ latent it tests inferred overlap directly. Report the effective number of independent sources against the Bayes value.
- **Separable vs non-separable junta twins.** Use an additive or threshold value over the junta bits (separable) and an XOR with the bits spread across items (non-separable; Ostrovsky 2012 Example 1, V). Scalar per-item summaries provably cannot aggregate the second. In-context parities show a total plateau (Bhattamishra, now V\*), so the non-separable twin is a negative control unless it gets auxiliary relevance supervision. Do not cite Peters et al. 2026 as an in-context result: it trains on one fixed junta with supervised learning (C).
- **Resample the similarity structure every episode,** with a fixed-structure control. Similarity that stays fixed is compiled into the QK weights (Collins, V) or into a frozen kernel (HyperBO, V).
- **Make keyless copies identifiable.** A copy should repeat the exact noisy value (shared error), and a fresh measurement should draw independent noise. This is the classical data-fusion principle (Dong et al. 2009; U). The exact posterior over provenance partitions is then the Bayes target.
- **Add an arrival-order knob and an open-groups knob** (contiguous vs interleaved copies; number of distinct groups). Exact streaming distinct counting needs linear state, but approximate counting needs only log state (verifier correction). Bounded state therefore rules out exact dedup with many interleaved groups, not approximate dedup.
- **Baselines with known mechanisms.** Place each trained model among these:
  - KR-UCT / Nadaraya-Watson: count-weighted, copies count as fresh, policy-dependent (V).
  - MC-RAVE β: assumes independence, so it double-counts (V).
  - GP K^-1: explains away collinear copies.
  - Volpp Bayesian Aggregation: precision weights that ignore other points, so it cannot discount copies (V).
  - Mesa/ridge state: exact only with γ=1 and converged CG (C).
  - NCP-style pooling: sum within a group, then across groups, which is policy-invariant by construction (V).
- **Mismatch protocol** (Garnier-Brun et al., V). Train at one copy rate or relatedness level and test at others. If the model matches the Bayes predictor of its training prior even where that is suboptimal, it implements prior-specific Bayes.
- **Score the low-evidence regime.** BMA, aggregate-then-select and fixed shrinkage differ mainly at finite N (Selective Induction Heads, C).
- **Log the implied pooling geometry and extremizing factor** against overlap and N. The Bayes factor depends on overlap and grows with combined information (Satopää, abstract-level). A curve that stays flat means a prior-level solution, and a factor fitted only on the training N range will mis-extrapolate (I).
- **Keep the gap between fixed and in-context discounting large.** Weak per-episode similarity evidence makes fixed discounting nearly Bayes-optimal (the forecast-combination-puzzle regime; I, explanation contested in the literature, C). That hides the mechanism and may feed the stuck-seed basin.
- **Fix the MCTS target before building the world.** KataGo's Q is by definition the expected utility of the search's posterior policy, so it is policy-dependent (V). The policy-invariant target is the minimax/Bayes value.
- **Test policy invariance in two separate parts:** change counts at a fixed shuffled order, and permute order at fixed counts. Causal and recurrent mixers fail the second as well (TabFlex/Koch, C/V).
- **Before claiming an architecture gap on dedup,** sweep learning rate for each architecture and run 5-10+ seeds. Recall gaps are partly optimisation, and lock-in depends on the seed (Okpekpe & Orvieto V; Boesch & Wee V, toy-scale). Curricula on copy distance or group count are the cheap test for stuck seeds.
- **Recurrent or bounded-state variants:** use a Robbins-Monro write rate β_t·c/(t+c) and heavy-tailed N (Schnurr et al., V). Expect degradation beyond 2-4x the training length without them.

---

## Addendum: topics 5-8

Inputs: research and adversarial-verifier passes on topic 5 (structured bandits, BO and MCTS with shared evidence), 6 (market and forecast aggregation), 7 (bounded-state architectures) and 8 (hierarchical, tree and mixture priors for ICL). This supersedes the header's note that topic 5 lacked a verifier and that topics 6-8 did not arrive. Topic 8's verifier block was itself cut off after Cluster-PFN/NCP/Chelly. Everything in topic 8 after that point is therefore U: Teh/Jabbour/Polyanskiy and Cannella et al. (empirical Bayes), Dong et al. (data fusion), Gao et al. and GullibleBench (copy weight), Zhu/Oermann/Cho and Shi et al. (multi-task), Saanum et al. and Shai et al. (probing). I did not re-read any source myself. The tags are the same as above (V, V\*, C, U, I).

Dropped as citation-wrong: the four-author "Anunrojwong, Chen, Waggoner, Xu" paper on universal complements. The result is from Anunrojwong's 2018 senior thesis, and Chen-Waggoner Cor. 5.1.1 already gives XOR as an example. Corrected citations used: Kalman Linear Attention is Shaj et al., arXiv:2602.10743 (not 2605.11007). The ICML version of CMANP is arXiv:2305.14567. The ICLR 2026 causal-structure paper is by Wei, Chen, He, Yang. The Economics Letters 2026 shared-information paper is by Peker.

### A.0 Status changes to sections 1-7 (topic 5 now verified)

| Item in sections 1-7 | Old | New | Note |
|---|---|---|---|
| Bhattamishra et al. 2024, Boolean ICL (§2, §4, §7) | U | V\* | Table 1 numbers match, e.g. Parity-(10,2) 53.7% at 140 points, while an FFN trained by GD on the same points gets 100%. Pairwise correlation, not VC dimension, tracks difficulty (assumes uniform inputs). Teaching sequences and the 50/50 two-algorithm switch are confirmed. |
| "Hyena, LSTM, DSS match transformers except NN" (§5) | U | C | Only Hyena matches broadly (NN 85.5 vs 100). LSTM and DSS lag on linear regression; DSS and RetNet lag on 0-1 threshold (about 72 vs 85). |
| Peters et al. 2026, noisy juntas (§2, §4) | U | C | Supervised training on one fixed random junta per run, bit-flip feature noise, clean labels. Not in-context. The main-text regulariser is -λI[f], so it penalises LOW sensitivity; the abstract's wording is inconsistent. Transfers to a PFN only by analogy. |
| Hahn & Rofin 2024 (§4) | U | V (abstract) | Applying it to stuck seeds is our hypothesis (I). |
| Abedsoltan et al. 2025, parity without CoT (§4) | U | V | ICML 2025. |
| Wen et al. 2025, sparse parity (§4 context) | U | C | Supervised learning of a single fixed parity, not ICL. |
| Richardson iteration, Yan, Li et al. 2026 (§3, §7) | U | V (abstract) | Dedup improving with depth remains our inference (I). |
| PFNs4BO 30% irrelevant dims (§1, §2) | U | V | Full text. Lengthscale prior Gamma(1.2107, 1.5212); stated limits about 1000 points and 100s-1000s of irrelevant features. TabPFN v1 App. B.3.2 degradation confirmed. |
| Transposition literature: Childs 2008, UCD 2012, MCGS/Czech 2021, KataGo notes (§7) | U | V | "UCT3 is TD(0)-like" is from secondary sources, unchecked. |
| KR-UCT, RAVE β, Sampled MuZero (§7) | U | V | Full text. |
| Wagstaff et al. 2019 (§5) | U | V | Latent dim ≥ M for universal continuous sum-decomposition with scalar elements. Max is proven in the JMLR 2022 extension (arXiv:2107.01959); median is only an empirical illustration. Extending it to "count distinct" is I. |
| PreDeToR follow-up, noise shift, arXiv:2510.00347 (§7 rules) | U | V | A preprint with self-described preliminary results. |
| DPT horizon | n/a | C | 500 steps for basic bandits, 200 for linear bandits; no extrapolation is tested. |

### A.1 What is already established

| Result | Source | Status |
|---|---|---|
| KR-UCT shares evidence through a fixed execution-noise kernel. Kernel density W(a)=Σ_b K(a,b)n_b replaces n_a, each sample counts as its own data point, and a Dirac kernel gives UCT. | Yee, Lisý, Bowling, IJCAI 2016 | V |
| MC-RAVE min-MSE β = ñ/(n+ñ+4nñb̃²). b̃ is tuned empirically. The authors admit the independence assumption is false (the same simulations feed both estimates). | Gelly & Silver, AIJ 175 (2011) | V |
| Transpositions cause staleness, over-counting and information leaks. MCGS stores Q and N on edges and nodes and corrects edge Q toward the node value. | Childs et al. 2008; Saffidine et al. 2012; Czech et al., arXiv:2012.11045 | V |
| KataGo: Q(n) = (U(n)+Σ_a N(n,a)Q(child_a))/N(n), "the expected utility of the posterior policy". Crediting every parent over-counts (D: 0.55→0.35 with best child 0.56). | Wu, KataGo docs/GraphSearch.md | V |
| Sampled MuZero corrects for which actions were sampled with (β̂/β)·π, which converges as K→∞. | Hubert et al., arXiv:2104.06303 | V |
| Softmax attention learns an NN window that widens with smoother or noisier tasks and projects onto shared subspaces; linear attention cannot. Prompt length scales with intrinsic r. Scores resemble anisotropic oracle kernels. The claimed degradation under test-Lipschitzness mismatch was not checked. | Collins et al., arXiv:2402.11639; Oko et al., arXiv:2411.02544; Shen et al., arXiv:2610.24441 | V |
| Sparse linear bandits: Θ(n^{2/3}) dimension-free in the data-poor regime (explore-then-commit), √n under minimum signal; feature feedback gives k√T. | Hao, Lattimore, Wang, arXiv:2011.04020; Oswal et al., arXiv:1903.03705 | V |
| HyperBO meta-learns a GP prior with a KL loss, then freezes it. NN encoder + Matérn not confirmed. | Wang et al., JMLR 25 (2024), arXiv:2109.08215 | V (abstract) |
| PreDeToR predicts reward per arm and beats AD and DPT-greedy (not oracle DPT) in most experiments. It fails at low data diversity. | Mukherjee et al., arXiv:2406.05064 | C |
| Ostrovsky Thm 1: separable securities aggregate in every equilibrium (any strictly proper MSR). Non-separable ones have a prior with a non-aggregating equilibrium. Example 1 is the Geanakoplos-Polemarchakis XOR, where truthful posterior announcements never reveal the value. The "scalar summaries cannot beat the prior" gloss holds for this uniform-prior XOR case. | Ostrovsky, Econometrica 80(6) 2012 | V |
| Informational substitutes give immediate aggregation; complements give delayed, worst-case aggregation. XOR is a universal complement. | Chen & Waggoner, FOCS 2016, arXiv:1703.08636 | V |
| Cost-function market makers correspond to exponential families (shares as natural parameters, prices as means). An additive LLR increment per trader holds only if the trader uses the market price as its prior. | Abernethy et al., arXiv:1402.5458 | V (abstract); increment I |
| Combinatorial LMSR pricing is #P-hard (permutation, Boolean). The tractable Bayesian-network pricing belongs to Pennock & Xia 2011, not this paper. | Chen et al., arXiv:0802.1362 | C |
| ML markets: utility choice gives PoE vs MoE; isoelastic agents give α-mixtures; Kelly prices are wealth-weighted and update like Bayes. Which utility yields which pool is not confirmed. | Storkey 2011/2012; Beygelzimer et al. 2012; Barbu & Lay 2012 | C |
| With independent private information and a common prior, Bayes group belief is multiplicative: p ∝ p0^{1-n}Πp_i. Overlap gives a more complex form. | Dietrich 2010; Genest et al. 1986 | V |
| Partial-information Gaussian model, p=0.6, q=0.8: probit average 0.708, ρ=1/2 gives 0.814, ρ unknown 0.833, log-odds sum 0.857. Averages are underconfident (convex hull); the log-odds sum is overconfident. | Ernst et al., arXiv:1608.04717 | V |
| Under projective substitutes with a known prior, averaging then extremizing linearly away from the prior is optimal, with d→√3 (k=2: d≈1.29). Stated in the paper itself, for real-valued forecasts, not log-odds. | Neyman & Roughgarden, arXiv:2111.03153 | C |
| Robust aggregation: guarantees exist only for TWO experts (Blackwell-ordered or cond. i.i.d.). With many cond. i.i.d. experts nothing asymptotically beats 0.5. Richer queries restore optimality. A calibration-only LP benchmark sits strictly below Bayes. Deterministic calibrated refinement is NP-hard. | Arieli et al., PNAS 2018 (C); Frongillo et al., arXiv:2512.05271 (V); Guo, Li, Mu, arXiv:2606.31020 (V); Tang & Zhang, arXiv:2607.08744 (V) | C / V |
| Equicorrelated experts are worth k/(1+(k-1)ρ) → 1/ρ independent ones. The forecast-combination puzzle has contested explanations: weight-estimation error (Smith-Wallis, Claeskens) vs low-power, size-distorted tests (Frazier et al., arXiv:2308.05263). | Clemen & Winkler 1985 and others | C |
| Shared information biases crowd averages. Pivoting on meta-predictions (Palley-Soll; Rilling, arXiv:2404.17737) and peer prediction (Peker, Econ. Lett. 2026) recover independent signals. | as listed | V |
| Correlation neglect (copies treated as independent) shrinks sharply when structure is made explicit. Persuasion bias makes influence proportional to exposure. | Enke & Zimmermann 2019; DeMarzo et al. 2003 | V (bibliographic) |
| Bayesian Aggregation in NPs: a precision-weighted posterior whose weights do not look at other points, so it cannot discount copies. Mean aggregation is a bottleneck (ANP). The GP-vs-NP KL splits into label contamination O(1), bottleneck and amortisation; second-order pooling is recommended. | Volpp et al., ICLR 2021; Kim et al. 2019; Young, arXiv:2605.21798 | V |
| On 15 LLMs × 254 questions, a learned MLP aggregator is no better than logistic regression; both beat classical pools. | Douven, arXiv:2607.18269 (venue unverified) | V\* |
| One-layer linear attention reaches the global optimum on in-context regression and is brittle under covariate shift. "Cannot invert XᵀX" is an interpretation. | Zhang, Frei, Bartlett, arXiv:2306.09927 | V |
| MesaNet solves a gated, discounted ridge with 30 truncated CG steps and stores associations in one shot (DeltaNet needs repeated presentations). It is exact ridge only with γ=1 and converged CG. | von Oswald et al., arXiv:2506.05233 | C |
| Test-time regression unifies linear attention, SSMs, DeltaNet, Mesa and softmax as weighted regression. | Wang, Shi, Fox, JMLR 27 (2026), arXiv:2501.12352 | V |
| Recall gap: 82% of the perplexity gap is recall (Zoology). State-recall Pareto frontier (Based). Fixed-state models cannot copy beyond their state (Jelassi). RNNs fail retrieval even with CoT (Wen, Dang, Lyu). | arXiv:2312.04927, 2402.18668, 2402.01032, 2402.18510 | V |
| Gather-and-Aggregate heads carry retrieval in transformers and SSMs. Swapping one SSM layer for attention raises Llamba-8B MMLU from 33% to 50%. Mamba does induction through its short convolution. | Bick, Xing, Gu, arXiv:2504.18574; Arora et al., arXiv:2505.15105 | V |
| Mamba matches transformers on ICL regression via online-GD-like dynamics, beats them on sparse parity and single-index targets, and fails vector-valued MQAR. | Park et al., arXiv:2402.04248; Jiang et al., arXiv:2509.23779; Oh et al., arXiv:2510.12026 | V |
| Transformers represent and match Kalman filters in context (Nadaraya-Watson construction; linear-Gaussian match, near EKF/PF otherwise). | Goel & Bartlett, arXiv:2312.06937; Akram & Vikalo, arXiv:2410.16546 | V |
| Transformers on a CTW prior track CTW. 1 layer is about unigram; attention-only does much worse; softmax gives frequencies, so counts are rebuilt from forward/backward statistics and position. Construction with D+2 layers exact, 2 layers approximate. No evaluation beyond the window. The count-removal ablation degrades only modestly. | Zhou, Tian, Diggavi, arXiv:2410.05493 | V |
| Learning Universal Predictors: transformers reach CTW-level in distribution but fail on longer sequences. The failure is in the original paper (Fig. 2: train 256, test 1024, sin/cos PE); the PE attribution (Young & Witbrock, arXiv:2408.12065) is hedged. | Grau-Moya et al., arXiv:2401.14953 | V\* |
| On a tree-broadcast model with exact BP, transformers reach BP-optimal performance, learn levels bottom-up (staircase), and probes at layer m decode ancestors to level m. Trained at k_train and tested at k_test, they exactly match BP_{k_train}. The grammar is fixed (in weights). | Garnier-Brun et al., arXiv:2408.15138 | V |
| A 2-layer model with relative PE implements BMA over causal parent graphs resampled per context; trained models approximate it. For continuous LDS the bilinear score cannot represent BMA (a representational limit). | Wei, Chen, He, Yang, ICLR 2026 | V |
| Selective induction heads: a 3-layer attention-only circuit with soft weights that resembles BMA. It converges to ML only asymptotically in some cases, and exact BMA also picks the likeliest lag as T grows. The real difference is the finite-T aggregation rule. | D'Angelo, Croce, Flammarion, arXiv:2509.08184 | C |
| Induction heads blend partial context matches (Jelinek-Mercer-like), and attention to BOS adds Dirichlet pseudo-counts. | D'Angelo et al., ICML 2026, arXiv:2607.02800 | V (abstract) |
| Cluster-PFN outputs posteriors over K and over assignments and beats AIC/BIC/VI on K. The DP/unbounded case is stated as open. NCP pools within then across clusters for arbitrary sizes but labels sequentially (order-dependent; fixed by Chelly et al., AISTATS 2025, arXiv:2502.19337). | Bhaskaran & Viering, arXiv:2510.24318; Pakman et al., arXiv:1901.00409 | V |
| Permutation-invariant, no-PE transformers do Poisson empirical Bayes in context and beat NPMLE and Robbins. | Teh, Jabbour, Polyanskiy, arXiv:2502.09844; Cannella et al. 2026 | U |
| LLM aggregators count restated evidence: copy weight w_m = 0.06-0.42, largely all-or-nothing per response. Exact-Bayes matched pairs (GullibleBench). | Gao et al., arXiv:2610.06192; GullibleBench | U |
| Transformers linearly represent Bayesian belief-state geometry for HMMs in the residual stream. | Shai et al., arXiv:2405.15943 | U |

### A.2 Priors others used, and what they showed

| Prior | Used by | What it showed | Status |
|---|---|---|---|
| CTW branching prior: ternary, D ≤ 5, stop 0.15, Dirichlet(0.5) leaves, window 512-1536 | Zhou et al. 2026 | Depth ≥ 2 tracks CTW; 1 layer about unigram | V |
| Variable-order Markov sources and UTM program outputs | Grau-Moya et al. 2024 | CTW-level in distribution; length failure | V\* |
| Tree-broadcast grammar with filtering level k | Garnier-Brun et al. | BP-optimal, bottom-up staircase, prior lock-in | V |
| Per-context random parent graphs, several sequences per context | Wei et al., ICLR 2026 | BMA over structures | V |
| Interleaved Markov chains with unknown lag | D'Angelo et al. 2025 | Aggregate-then-select circuit | C |
| Finite GMM, Dir(0.1) weights, NIW, k ~ U(1,K), missingness | Cluster-PFN | K posterior beats AIC/BIC/VI | V |
| Gaussian partial information (sub-intervals of white noise, covariance = overlap) | Satopää, Pemantle, Ungar; Ernst et al. | Closed-form Bayes between copy and fresh | V |
| Partition information structure with XOR security | Geanakoplos-Polemarchakis; Ostrovsky | Non-aggregation | V |
| Common plus private signals | Palley-Soll; Peker 2026 | Meta-predictions recover private signals | V |
| GP with Matérn, Gamma lengthscale hyperprior, 30% irrelevant dims | PFNs4BO | In-context ARD | V |
| Continuous-action MCTS with Gaussian execution noise (curling) | KR-UCT | Kernel sharing beats UCT/RAVE/PW | V |
| 5-arm and linear bandits, TS-collected contexts | DPT | Compliance → posterior sampling | V (C on horizon) |
| Linear, bilinear, latent, non-linear multi-task bandits | PreDeToR | Per-arm reward head beats DPT-greedy | C |
| TabPFN v1 prior at n ≤ 1000, 12.5M-parameter mixers | Schnurr et al. 2026 | DeltaNet > other linear mixers, Mamba-2 last; recurrent models degrade beyond 2-4x | V |
| Haystack associative recall with distractors and a distance curriculum | Boesch & Wee 2026 (~29k params) | Lock-in lottery, 1/10 vs 7/10 seeds | V (toy) |
| Poisson empirical Bayes with prior-of-priors | Teh et al. 2025 | Set transformer beats NPMLE | U |
| Binary state, sensors of reliability q, restated readings | Gao et al. 2026 | Copy-weight metric | U |

### A.3 Mechanisms; depth, width and state

- **Gather then aggregate.** Retrieval runs through a few gather heads followed by aggregate heads (Bick et al., V). For dedup the analogous step is: match same-group items, then pool (I). This is consistent with our 2+-layer requirement but does not explain it. Probe for a sharp head attending to same-group items.
- **Counts and BOS pseudo-counts.** In Markov ICL, attention to BOS adds Dirichlet pseudo-counts (D'Angelo et al. 2026, V abstract). This is published support, in a different setting, for the sink-as-prior-shrinkage hypothesis in §3 (still I for our worlds). In CTW, softmax gives only frequencies, so counts are rebuilt from position and backward statistics (Zhou, V).
- **Depth as iterative inference steps.** One mirror-descent step on mixture weights takes 3 layers (D'Angelo & Flammarion, V). Richardson iteration for GP aligns with depth (V abstract). Prediction (I): if dedup is an iterative reweighting or solve, its quality rises layer by layer. Test with a logit lens on contexts with many copies.
- **Structure from mutual information.** First-layer attention gradients track chi-squared MI between positions (Nichani, V; Wei et al., V). Prediction (I): weakly informative similarity (subtly related items) is learned last or not at all.
- **Update rules and copies (I, untested).**
  - Additive linear attention counts each copy.
  - Ridge/Mesa weights each copy as a fresh observation.
  - With β=1, unit keys and no interfering writes, the delta rule writes an exact copy only once. It then overwrites fresh measurements under the same key instead of averaging them.
  - The verifier notes that this fails under interfering writes or β<1. So copies vs fresh could tell the update rules apart.
- **Pooling geometry.** For independent evidence Bayes is multiplicative with a prior correction; for exact copies it is "take one"; for model uncertainty it is linear (Dietrich, Genest, V). A "fixed shrinkage" solution corresponds to a fixed exponent between 1/n and 1 (I).
- **Market reading of the residual stream (I, untested).** The stream as accumulated natural parameters, with layers as rounds of trading. Treat this as a hypothesis for a probe, not as an analogy to rely on.

### A.4 Failure modes, with predictions for our worlds

| Failure | Source (status) | Prediction / test for us |
|---|---|---|
| Non-aggregation under non-separable/XOR structure | Ostrovsky (V); Bhattamishra (V\*) | XOR across items in a junta world gives a total plateau for a single-pass set-PFN and is unreachable for scalar-summary pooling. Use it as a negative control. |
| Kernel smoothers are count-weighted | KR-UCT (V); Collins (V) | A 1-layer model behaves like a learned-bandwidth NW smoother: copies count as fresh, and the estimate is policy-dependent (I). |
| Double counting of shared evidence | RAVE (V); KataGo (V); Volpp BA (V) | Precision-weighted or additive pooling over-counts copies by construction. A model that pools by observation stream rather than by distinct source fails in a predictable direction. |
| Averages underconfident, log-odds sum overconfident | Ernst et al. (V) | Our partial dedup should fall between the two; report where along ρ. |
| Fixed extremizing does not track N | Satopää (abstract); Neyman-Roughgarden (C) | A prior-level discount is minimax-sensible but wrong for each episode. Beyond the training N it miscalibrates in a known direction (I). |
| Prior lock-in | Garnier-Brun (V) | Models implement the Bayes predictor of their training copy rate. Test at mismatched copy rates. |
| Plateau from a simpler competing solution | Edelman (V); Park (V); Gibson, Cui, Reddy (V) | Stuck seeds are consistent with a unigram-like competitor. Levers to test: task mixture, context length, data diversity. No paper reports per-seed stuck fractions, so there is no direct comparator for our 20-25%. |
| Lock-in lottery under sparse supervision | Boesch & Wee (V, toy) | A curriculum on copy distance or group count may move the stuck fraction (1/10 → 7/10 seeds in their task). |
| Recurrent length failure | Schnurr (V); Buitrago-Gu (V) | Beyond 2-4x N, unless write rates decay like 1/t. State passing helps GLA/Mamba and hurts DeltaNet. |
| Order dependence of causal mixers | TabFlex (C: shown mainly for causal attention); Koch (V) | Confounds policy-invariance tests; separate counts from order. |
| Learning-rate sensitivity inflates architecture gaps | Okpekpe & Orvieto (V) | Sweep LR per architecture before comparing dedup. |
| Noise-level shift | DPT follow-up (V, preliminary); TabPFN v1 irrelevant features (V) | Randomise noise scale and irrelevant fraction per episode. |
| Data incest in LLM aggregators | Gao et al. (U) | If confirmed, the per-response all-or-nothing pattern suggests checking whether our 45% dedup is a per-episode mixture of full and no dedup (bimodal) or uniform partial discounting. |

### A.5 Bounded-state architectures (additions to §5)

| Architecture | Evidence | Status |
|---|---|---|
| MesaNet (ridge state, CG) | Gated ridge with truncated CG; exact only with γ=1 and converged CG | C |
| Preconditioned DeltaNet | Linear attention and delta rule coincide under exact preconditioning | V |
| MambaFormer | Mamba replaces MLPs, initial Mamba block, no PE; gets both ICL regression and retrieval | V |
| Latent bottlenecks (LBANP, CMANP, ISAB in TabICL) | Query cost independent of N; TabICL 60K → 500K rows | V |
| Single-query softmax readout (Aaren) | Streams in O(1) | V |
| ARMT | Trained at 16k; 79.9% at 50M tokens; recall beyond 10x | V |
| Streaming Recurrent Memory | Count and Presence tasks; no transformer baseline, no length test | V |
| Kalman Linear Attention | Information-form bounded-state filter | U |

Points that matter:
- Query-to-context readout is never the bottleneck. The cost lies in item-to-item matching.
- Prediction (I): m inducing latents dedup well up to about m distinct groups and degrade beyond that.
- Exact Bayes in a conjugate world needs no growing state. Such a world therefore tests capacity only once copies or related items break sufficiency.

### A.6 Effect on "What would be new" (§6)

- **Item 1 (exact-Bayes in-context dedup) still stands.** Topics 6, 7 and 8 each searched separately and found no trained model, PFN or bounded-state model tested on copies vs fresh against exact Bayes. Nearest work to check before claiming novelty:
  - Gao et al. 2026: pretrained LLM aggregators, not trained PFNs (U).
  - Dong et al. 2009: non-neural copy detection (U).
  - Liu et al. 2026: already in §1.
- **Item 3 (policy invariance) still stands.** Topic 6 found no neural literature. The formal condition is ignorability or compliance (Rubin 1976, from memory, U; DPT, V).
- **Item 5 (bounded state with inferred similarity) still stands.** Topic 7 found no bounded-state dedup test. The open-groups and arrival-order knobs are the cleanest way to separate architecture classes.
- **CTW as a world is no longer new** (Zhou et al., V). What is still new: N beyond the window, a no-PE set variant, and copies inside the tree world.
- **New candidates for "new":**
  - Separable vs non-separable aggregation twins in a trained PFN.
  - Implied-extremizing-factor curves against N and overlap.
  - An in-context (per-episode) version of the tree-filtering mismatch protocol. The Garnier-Brun grammar is fixed.

Add to "Not new; cite instead of rediscovering":
- in-context Boolean difficulty ordering (Bhattamishra);
- CTW ICL with depth ≥ 2 (Zhou);
- scalar-report aggregation fails under XOR (Ostrovsky);
- averages are underconfident and log-odds sums overconfident (Ernst);
- BA cannot discount copies (Volpp);
- recurrent recall gaps and 1/t write-rate fixes (Zoology, Schnurr).

### A.7 Missing work the verifiers flagged (unchecked, U)

- Topic 5: Garg et al. 2022 (arXiv:2208.01066; length claim still unverified); Barak et al. 2022 and Edelman et al. 2022 on hidden progress in sparse parity; SAASBO, REMBO, Add-GP-UCB; cRAVE and Continuous MCGS (arXiv:2210.01426); Mossel-O'Donnell-Servedio junta learning; Gumbel MuZero.
- Topic 6: Ranjan & Gneiting 2010 (linear pools of calibrated forecasts are uncalibrated); Baron et al. 2014; Prelec et al. 2017, "surprisingly popular"; Frongillo, Chen, Kash 2015; Morris/Winkler correlated-expert Bayes; Pennock & Xia 2011; FFORMA.
- Topic 7: Akyurek et al. 2022 (arXiv:2211.15661); Ahn et al. 2023 (arXiv:2306.00297); Schlag et al. 2021 (arXiv:2102.11174); Grazzi et al. 2024 (arXiv:2402.03170); "Just read twice" (arXiv:2407.05483, order sensitivity on set-like recall); Mamba-2/SSD; TTT and Titans; Transformer Neural Processes; Equilibrium Aggregation (arXiv:2202.12795); HyperLogLog-style approximate distinct counting.
- Topic 8: the verifier block was cut off, so missing work for this topic was not listed.
