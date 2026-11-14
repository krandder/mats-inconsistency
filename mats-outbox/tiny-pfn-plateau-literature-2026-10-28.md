# Tiny-PFN plateaus: verified literature and predictions

Checked 2026-10-28 10:54 UTC against the seven primary PDFs linked below, including visual inspection of Kirsch Fig. 21. Local setup checked against `work/tiny-pfn/{world.py,pfn.py,pfn_amend2.py,PREREG.md}`. Existing experimental results below are from the October 27 reports, not independently rerun. No training or paid compute was performed.

**Recommendation:** test Adam epsilon first, measuring its actual effect on the missing source-routing circuit. Larger batches are a useful second intervention, but the closest published study found faster escape in updates and *slower* escape in examples. Neither result establishes an escape-time law for our PFN. Do not remove the query-dominated strategy, unbalance source counts, or increase weight decay based only on this literature.

## What the primary papers establish

| Primary source and locator | Published result | What transfers to our case |
|---|---|---|
| **Kirsch et al., General-Purpose In-Context Learning by Meta-Learning Transformers**, [2212.04458v2](https://arxiv.org/pdf/2212.04458v2), §4.3, Figs. 6–8; App. A.6, Figs. 17–21 | Long meta-learning plateaus with tiny gradients in most tensors, while the final layer has larger gradients. Fig. 21 compares Adam epsilon **1e-8, 1e-16, 1e-24**; main text reports smaller epsilon more than halves plateau length. Task batches **32–4096** shorten plateaus in optimizer steps but increase examples processed before escape (Fig. 20). Giving a fraction of sequences a fixed label permutation can remove the plateau (Fig. 8: 0%, 10%, 90%, 100% permuted-label conditions). Main plateau demonstrations use augmented FashionMNIST, not exclusively MNIST. | Direct motivation for measuring per-tensor Adam moments and updates. A smaller epsilon is a conditional intervention, not a promised 2× speedup. Biased task sampling was a particular curriculum; the paper did not test adding duplicate measurements to Bayesian regression. |
| **Nguyen & Reddy, Differential Learning Kinetics Govern the Transition from Memorization to Generalization during In-Context Learning**, [2412.00104v2](https://arxiv.org/pdf/2412.00104v2), §§3.4–3.7, Eqs. 7–11, Fig. 6 and supplementary Fig. A.4 | In a reduced classification model, acquisition time depends exponentially on an initial attention parameter: for the stated small-initialization, infinite-task limit, `t_ICL ≈ N sqrt(2π) exp(-β₀)`, up to a time conversion constant. Full-network tests find long-tailed escape times and bimodal outcomes near the finite-task diversity threshold. Forcing exactly equal positive/negative label counts removes a useful majority-label gradient and makes acquisition less reliable. Stronger regularization of attention than MLP can yield transient ICL. Experiments use SGD, not Adam. | Measure individual escape times and circuit trajectories; finite-horizon failure is not proof of a permanent attractor. Equal **label counts in this classification task** are not equivalent to equal numbers of readings from our three sources. Their exponential equation does not supply our Adam escape time. |
| **Reddy, The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task**, [2312.03002v1](https://arxiv.org/pdf/2312.03002v1), Fig. 6c and accompanying text; supplementary Fig. A.5 | A cheap intermediate strategy improves accuracy from `1/L` to `1/N` before induction-head formation. Setting **L=N=8, B=1** removes that advantage: **4/6 seeds** converge to a suboptimal solution in the reduced-model experiment. Full-model checks also find reliable acquisition for `L>N`, but not `L=N`. | An easy strategy can scaffold a later circuit. “Destroy the query-only shortcut” is not an established fix. Whether our shortcut scaffolds or suppresses source routing requires intervention. |
| **Chan et al., Data Distributional Properties Drive Emergent In-Context Learning in Transformers**, [2205.05055v6](https://arxiv.org/pdf/2205.05055v6), §2.1, §3, Figs. 2–6 | In Omniglot sequences, burstiness, many rare classes, and changing class-to-label mappings promote ICL. Their bursty sequence has the query class three times in context. Increasing class diversity from 100 through 1600 and then 12,800 improves ICL in their setup. A Zipf exponent around 1 permits both frequent-class IWL and ICL. | Shows that data statistics alter which strategy is learned. It does not show that our finite variance prior should become Zipfian or that duplicate readings help infer reliability. Our source IDs are freshly sampled, so they cannot carry a reusable source-to-variance lookup across episodes. |
| **Raventós et al., Pretraining Task Diversity and the Emergence of Non-Bayesian In-Context Learning for Regression**, [2306.15063v2](https://arxiv.org/pdf/2306.15063v2), §§2–3, Fig. 2; App. D, Fig. 9 | Below a diversity threshold, linear-regression predictions track the exact discrete-prior Bayes estimator over a fixed training task pool; above it, they become ridge-like on fresh Gaussian tasks. At equal total sequences, **B=128/256/512 with 1M/500k/250k steps** have matching measured final behavior in the tested setup. | Fixed-pool task memorization and a query-dominated inference strategy are different hypotheses. Tiny-PFN has 27 possible variance assignments, but fresh targets, observations, and IDs; using a finite variance prior correctly still requires contextual inference. The batch control also cautions against treating more examples as a batch-specific improvement. |
| **Singh et al., The Transient Nature of Emergent In-Context Learning in Transformers**, [2311.08360v3](https://arxiv.org/pdf/2311.08360v3), §§5–6, Fig. 9; App. A and D, Fig. 16 | ICL can appear then fade during training. L2 coefficients **1e-5, 5e-5, 1e-4** can mitigate fading over the tested horizon; coefficients above 1e-4 can over-regularize. Penalizing MLP/embedding layers helps persistence; penalizing only attention does not. **L2 is added to the loss under Adam**, excluding batch-norm parameters. Two seeds per condition; up to 50M steps. | Track context use throughout training. This is primarily evidence about *loss of an acquired circuit*, not initial plateau escape. The coefficients are not portable to our AdamW decay=0.01. Our MLP also computes reliability features; it cannot simply be equated with their memorization pathway. |
| **Zucchet et al., The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition**, [2505.17863v2](https://arxiv.org/pdf/2505.17863v2), §2.3–2.4, Fig. 3, App. D.1 | For a two-layer transformer, a fitted relation is `T_plateau = 0.76 d^1.29 T^0.80 B^-0.80`. Here **B is the number of within-context copies of the relevant token**, not minibatch size. Exponents change with task, architecture and optimizer; this transformer removes layer normalization. | Useful qualitative connection to acquiring selective attention. The `B^-0.80` exponent cannot predict our minibatch effect. Repeated relevant tokens and independent noisy measurements also have different statistical meanings. |

## Quantitative predictions for this experiment

These are algebraic predictions or explicit extrapolations, not results already observed in tiny-PFN.

**Epsilon.** Let `q = sqrt(v_hat)` be Adam's bias-corrected second-moment scale for a coordinate. Holding parameters and optimizer state fixed, changing epsilon from `e_old` to `e_new` multiplies the adaptive update by

`A(q) = (q + e_old) / (q + e_new)`.

This concerns the adaptive gradient term; AdamW's decoupled weight-decay term is separate. For **1e-8 → 1e-12**:

| q | Update multiplier |
|---:|---:|
| 1e-5 | 1.001 |
| 1e-6 | 1.010 |
| 1e-7 | 1.100 |
| 1e-8 | 2.000 |
| 1e-9 | 10.989 |
| 1e-10 | 100.0 |

Thus an immediate Kirsch-style explanation predicts materially larger updates specifically where routing needs to grow. If the relevant coordinates all have `q ≥ 1e-6`, the immediate amplification is at most 1%; that weakens epsilon-limited updates as the proximal explanation. Very small `q` does not itself show useful signal: also measure first moments, update direction and the functional routing response. A large change in small noisy updates can destabilize training.

Start with the existing **B=256, LR=1e-3** and compare epsilon **1e-8 vs 1e-12**, adding **1e-16** if needed to test the published range. Keep initialization, data stream, loss normalization, clipping, precision, LR schedule and decay fixed. Record effective epsilon explicitly instead of relying on the optimizer default. A 2× update multiplier is **not** a prediction of 2× faster escape because the trajectory and moments change thereafter.

**Batch.** For IID episodes and mean-reduced gradients at fixed parameters, `Cov(g_batch)=Σ/B`. Relative to B=256, the gradient-noise standard deviations are **0.707 at 512**, **0.500 at 1024**, and **0.354 at 2048**. If weak but consistent source-routing gradients are masked by sampling noise, larger batches should improve the direction estimate; whether that accelerates escape is unproven. At a noise-dominated coordinate, the second moment also shrinks with batch, so epsilon and batch need not act independently.

At **B=1024**, a 4× increase over baseline:

- Equal updates consume 4× as many episodes.
- Equal episodes permit one quarter as many updates. If escape steps scale as `B^-α`, example efficiency improves only when **α>1**. Kirsch's tested regime has the opposite example-efficiency result.
- Equal wall time depends on measured throughput; gradient accumulation alone does not supply a parallelism benefit.
- Equal episodes also change Adam's moment-memory horizon in examples and cumulative AdamW shrinkage. At constant LR=1e-3 and decay=0.01, 80k versus 20k updates give pure-decay factors about **0.449 versus 0.819**. This calculation excludes warmup/cooldown and gradient updates; log `sum(LR)` so the difference is visible.

Report escape in **updates, episodes and seconds**, with censored runs retained. Do not claim a universal batch power law from a small sweep.

## Mechanistic readout and interpretation limits

The existing plateau checkpoint has local squared residuals after MLP1 (`R²≈0.89`) but weak source pooling: own-source/other-source layer-1 attention ratios **1.1 and 1.2**, versus some successful heads at **94–299**. Its output barely responds to changed labelled residuals (full-swap slope **0.08**, `R²=0.04`). This makes routing a sharper readout than total loss alone. These are findings in the October 27 interpretability report, not a new audit.

Measure the layer-1 Q/K and input source-ID blocks separately from values, MLPs and the output head. Log moment-scale quantiles, adaptive update norms, decay norms, clipping frequency, source-attention ratios, and the paired labelled-residual intervention response. The hypothesis predicts acquisition of effective source routing together with increased use of labelled evidence; attention sharpness without the intervention response is insufficient.

The old report's plateau gap is about **0.144**, while its exact query-only reference gap is **0.185**. Therefore “query-only” should remain shorthand for **query-dominated, weak labelled-context use**, not a demonstrated identity with that estimator. Different strategies can have similar output means; the intervention evidence is more informative.

Do not merge these outcomes:

1. No escape by a fixed horizon: censored optimization outcome.
2. Routing appears but predictions remain poor: downstream computation/readout failure.
3. Routing and context use appear, then disappear: transience.
4. Fresh-episode success fails while a finite fixed set fits: memorization/generalization gap.

Existing training code saves model weights, not Adam moments. An old-checkpoint restart with fresh Adam is an **optimizer-reset intervention**, not a clean continuation. For paired epsilon continuation tests, first save and clone full optimizer state, RNG state and schedule; otherwise use matched runs from initialization and label any warm restart accurately.

The literature gives credible mechanisms to test, but no paper here establishes a tiny-PFN capacity limit, permanent attractor, or guaranteed remedy. More seeds estimate escape probability; they do not repair the mechanism by themselves.
