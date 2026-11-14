# Do wider noise-trained networks have sparser circuits? Literature check

29 October 2026. Our result is in `outbox/algzoo-noisy-2026-10-29.md`, §5.2.

## Answer

I found neither a published version of this claim nor a refutation; the nearest is Gao et al., in weight-sparse transformers. Adjacent results agree on the share: wider or larger networks have more removable or redundant units (Casper et al.), and a smaller share of their weights (Li et al.; Rosenfeld et al.) or circuit edges (Bayat Makou et al.) suffices. Absolute size grows in dense models (Tigges et al.; ours: 2,106 kept edges at width 16, 16,000-22,000 at 64). "More localized" is open: dropout and noise breed redundant copies (Morcos et al.; Marshall & Kirchner), and removing one copy need not flip answers. Width and noise are confounded in our data; Qian et al. and the AlgZoo row hint that width drives it.

## Closest papers

| Paper | What it shows | Closeness to ours | Verdict | Read |
|---|---|---|---|---|
| [Casper et al. 2021](https://arxiv.org/abs/1912.04783), "Frivolous units" | CNNs and MLPs, 1/4x-4x width: prunable units (random ablation changes few labels) and redundant ones (linear combinations of others) more than double per width doubling; the rest grow slower; dropout changes this little. | Closest: width sweep, label-agreement test; but units and random ablation. | Supports: share falls, absolute count rises; spare width goes to both kinds. | Abstract, §3-6 |
| [mogar/algzoo_interp](https://github.com/mogar/algzoo_interp) (GitHub 2026, unreviewed) | ARC's noiseless 32x10 models (5 seeds) have dedicated neurons (new-max detector, running max, vetoes, clock); 7-10 neurons carry 90% of the max/2nd identity. Both 16x10 models spread it over 15 of 16. | Same task and architecture, no noise; qualitative. | Supports; suggests width alone localizes. | README, HYPOTHESES.md §F-G; not rerun |
| [Gao et al. 2025](https://arxiv.org/abs/2511.13653), weight-sparse transformers | Pruning mean-ablated nodes to a target task loss: with nonzero weights fixed, a larger hidden size gives lower loss and smaller circuits. | Same kind of search; weight-sparse transformers. | Supports, at a fixed weight budget. | Abstract, §2.2, §3.1 |
| [Bayat Makou et al. 2026](https://arxiv.org/abs/2606.06267); [Tigges et al. 2024](https://arxiv.org/abs/2407.10827) | ACDC keeps 30.9% to 2.8% of edges from Pythia-70M to 1.4B on one task (shallower 1B: 9.1%); repeat runs disagree on 27-79% of edges. Tigges: node counts rise with size. | Depth varies too; per-model thresholds (circuit KL 0.24 at 70M, 0.50 at 1.4B). | Supports, uncontrolled. | Abstracts; §3.3, §4.2; Tigges §2.2, size results |
| [Li et al. 2020](https://arxiv.org/abs/2002.11794); [Rosenfeld et al. 2021](https://arxiv.org/abs/2006.10621) | Larger transformers survive more pruning; under IMP (lottery-ticket pruning), ResNet error at fixed depth tracks w^ψ·d (width w, density d): wider nets reach equal error at lower density. | Weights, with retraining. | Supports; known for weights. | Abstracts; Li §4.3; Rosenfeld §4 |
| [Elhage et al. 2022](https://arxiv.org/abs/2209.10652); [Jermyn et al. 2022](https://arxiv.org/abs/2211.09169); [Lecomte et al. 2023](https://arxiv.org/abs/2312.03096) | Sparse features share neurons when they outnumber them; more neurons give more single-feature neurons; with spare neurons, about n²/4m stay polysemantic by chance under L1 (n features, m neurons). | Toy autoencoders. [Bereska et al. 2025](https://arxiv.org/abs/2512.13568) find little superposition in two algorithmic tasks, attributing it to dense inputs. | Supports a mechanism that may not apply to 2nd-argmax. | Abstracts; Elhage neuron sections; Jermyn §4.1.4; Lecomte §2-3; Bereska §6.1-6.2 |
| [Qian, Liang & Fiete 2025](https://arxiv.org/abs/2512.13707) | On a modular task, additive activation noise gives non-modular wiring in nonlinear nets; activity-scaled noise gives modular wiring. | Our noise is additive; modularity, not circuit size. | Neutral on width; against noise as the cause. | Abstract, §II |
| [Morcos et al. 2018](https://arxiv.org/abs/1803.06959); [Marshall & Kirchner 2024](https://arxiv.org/abs/2401.17975); [Poole et al. 2014](https://arxiv.org/abs/1406.1831) | Dropout nets resist ablation only up to the dropout rate, consistent with copies; dropout makes codes redundant; injected noise spreads information across units. | No width sweep or circuits. | Points the other way (redundancy); our W64 (σ = 0.03) spreads 80% of decision noise over 26 neurons, T16 over 5. | Abstracts; Morcos §3.3; Marshall & Kirchner and Poole results |
| [Chizat et al. 2019](https://arxiv.org/abs/1812.07956); [Woodworth et al. 2020](https://arxiv.org/abs/2002.09277); [Luo et al. 2021](https://arxiv.org/abs/2007.07497) | Width alone does not set the regime: 1/√m output scaling makes wide nets lazy; small initialization gives sparse (L1-like) or condensed solutions. | Two-layer or linear theory. | Neutral; condensation would look like redundancy. | Abstracts; Chizat §3; Woodworth §4; Luo §1 |

## What our result would add

If it holds: a controlled width sweep of a dense recurrent net on one algorithmic task, counting answer-preserving (neuron, step) edges, also relative to the task's own variables, which none of the width studies above does. Were it localization rather than redundancy, it would extend Casper et al.'s prunable units to single edges and time steps.

## Checks to make it robust

1. Separate width from noise, with more seeds: d = 16, 24, 32, 48, 64 at σ = 0 and 0.01, three seeds each, plus ARC's noiseless M_{32,10} (five trained seeds). Report kept share and absolute kept edges, also at matched loss (wide checkpoints at T16's loss), since better models may tolerate more ablation.
2. Control the ablation and the search: rerun with resample and optimal ablation ([Li & Janson 2024](https://arxiv.org/abs/2409.09951): optimal ablation gives lower loss than mean at equal size; [Miller et al. 2024](https://arxiv.org/abs/2407.08734): circuit scores depend on such choices). Per width, add a random-edge baseline (flips against share ablated, as Casper et al. did for units) and three repeat searches with edge overlap.
3. Test redundancy against localization: forbid the kept edges and search again. If a disjoint set meets the 1% target at width 64 but not at 16, the extra sparsity is redundancy.

Read status: last column; Li & Janson, Miller et al.: abstract, §2-3. ARC's AlgZoo post and Snider's LessWrong post (26 October, read via a summarizer) do not discuss width. Nothing rerun.
