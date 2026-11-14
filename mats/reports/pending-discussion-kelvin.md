# Held for later discussion (Kelvin asked: do not answer yet)
- 2026-10-26: "What can we learn (if anything) from this paper: Ye, Han-Jia; Liu, Si-Yang; Chao, Wei-Lun (2025). 'A Closer Look at TabPFN v2: Strength, Limitation, and Extension'. arXiv:2502.17361v1 [cs.LG]."
- 2026-10-26: first-world choice for the PFN plan (GP many-samples-near-a-point vs forecasting world with shared sources), still open.

## Parked 2026-10-26 23:10Z (Kelvin: "Let us park this tensor exploration.")
Riggs, "Tensor-Transformer Variants are Surprisingly Performant" (LW hp9bvkiN3RzHgP9cq, 2026-01-12). My read: trained models are not polynomial (RMSNorm per block, QK-norm, final norm, tanh logit cap); scale-invariance claim wrong (residual mixes degrees); 1 run per arm, softmax-tuned hparams, +10% params. Unstarted options: polynomial-attention arm in the estimability test; bilinear-attention arm for dedup (no key normalisation, duplicates count linearly).

## Decided 2026-10-26 ~23:15Z
Mechinterp target = the network's deviations from exact Bayes (Kelvin: "We need a mechanistic understanding of the errors from bayes."). "Network = Bayes" is a flat baseline on the MSE-vs-compute curve; a mechanistic estimate must stay below sampling past where sampling crosses it, over seeds and held-out conditions.

## Parked 2026-10-27 ~15:00Z: estimability track
Kelvin: "I don't think a loss that makes activations more gaussian is the way to go. Instead, making its activations
behave more like "symbolic activations" behaving according to known rules, according to some
expectation-maximization analogy, is probably more useful." Result on file: outbox/estimability-penalty-2026-10-27.md.
Searchformer/astra lane: paused (no scale011 work or spend) pending Kelvin's decision.

## Registered 2026-10-28 ~20:40Z: Rényi divergence across agendas
Kelvin: "I am curious about the Renyi divergence showing up again. It shows up prominently in the LIR paper. Let us register
this for the future, to investigate deep connections between these research agendas."
Context: in the noisy-channel interpretability program, per-event upper bounds from one model q come from Hölder:
log2 p(A) <= (1 - 1/alpha) [log2 q(A) + D_alpha(p||q)] (alpha -> inf gives the worst-case D_inf certificate that failed in
phase B; finite alpha uses average-case Rényi divergences, which phase B found small). LIR (Richardson et al.) uses
Rényi/alpha-divergences in its inconsistency measure. To investigate: whether the inconsistency-minimisation and
interpretability agendas share one divergence family and composition rules (weak triangle inequality across steps).

## Registered 2026-10-28 ~23:55Z: treewidth / lateral composition and combinatorial AMMs
Kelvin: "This is very interesting, and connections to my interests regarding the tractability of combinatorial AMM. Again for
future discussion."
Context: lateral composition of interpretability structures = a sparse surrogate Bayesian network (DAG) over unit groups;
KL chain rule adds over nodes (depth and width); computing expectations/event probabilities in the surrogate is exact
inference whose cost is governed by treewidth. Combinatorial market makers (LMSR over joint outcome spaces) face the
same tractability boundary (pricing is #P-hard in general, tractable for bounded-treewidth structures). To investigate:
one shared theory of when aggregation/description over combinatorial structure is tractable.
Addendum (same thread, ~00:05Z): question-driven structure search. Kelvin: "One can dream of asking questions and then our
divergence for the question automatically finds the structure that explains things." Formal handle: a question family F
defines the metric sup_{f in F}|E_p f - E_q f| (integral probability metric); structure search minimises it at given
description cost. The simplest model answering F exactly is the max-entropy exponential family with sufficient statistics
F -- the same object an LMSR market maker holds when pricing the securities F. Candidate unification: explanations of
question families <-> market makers pricing security families.

## Surprise accounting under noise (registered 2026-10-29 ~02:10Z)
Kelvin: "So they do have an info-theoretical approach as well, in a sense." ARC's surprise accounting (AlgZoo post) is a
description-length bar: bits of the facts about the weights an explanation uses + surprise of the actual accuracy given
the estimate, compared with the bits of optimization that selected the model. ARC price a fact by the precision the
calculation needed (e.g. a near-equality to 1 part in 2^11 = 22 bits in M_{2,2}); noiseless nets have no natural
precision scale. Open candidate: with activation noise sigma, price each fact only to the precision at which it can
change the noisy model's behaviour (about log2(1/sigma) bits per relation?), and use our KL/waste bits as the
"surprise given the explanation" term. Not yet worked out; check against the AlgZoo noisy lane's sigma = 0.001 model.

## Observation: relevance-weighted rotation, E[g g^T] (registered 2026-10-29 ~15:30Z, Kelvin: "This seems very deep... not to forget")
At each layer, choose coordinates that both vary across real inputs and move the question's answer: the top
eigenvectors of E_x[g g^T], where g = d(question)/d(state at that layer), averaged under the forward (input)
distribution. Forward (what varies) and backward (what matters) meet in one matrix; plain PCA of the activations
ignores the question. Related objects to study:
- the question's active subspace (Constantine 2015, "Active Subspaces");
- the outer product of gradients (OPG) and sufficient dimension reduction in statistics (Li 1991 SIR; Xia et al. 2002);
- the average gradient outer product and the Neural Feature Ansatz (Radhakrishnan, Beaglehole, Pandit, Belkin,
  Science 2024): trained weights' Gram matrix aligns with this kind of matrix, which suggests training itself builds
  these coordinates;
- multi-index models (SGD learns the relevant subspace: Damian-Lee-Soltanolkotabi 2022; Abbe et al. 2022-23; Ben
  Arous-Gheissari-Jagannath 2021);
- Fisher information (for log-loss questions), and the information bottleneck's "relevance".
Open: does training's own gradient outer product equal the right coordinates for describing the trained net? If
so, explanations could be read off training.
Kelvin, same message: "we need the families that can line up with the gate hyperplanes and allow
similar/functionally-correlated gates to be compressed".

## Coupling arithmetic for factored families (2026-10-29, after Kelvin: "Why can't you just know this already?")
For a random next layer z'_k = sum_j W_kj h_j (W fresh, entries ~N(0, 2/n)):
- means of z' are exact under any family with exact marginals (linearity);
- Var(z'_k) = sum_j W_kj^2 v_j + sum_{j!=l} W_kj W_kl c_jl. The off-diagonal part has mean 0 and std ~ 2*sqrt(2) c_rms against
  a diagonal of ~2v. So dropping couplings costs relative error ~rho_rms per neuron per layer, independent of width.
- Low-rank couplings (the size and collapse modes) make W'^T C W' fluctuate at O(1).
Hence independent products (gates, neurons or features) fail on random nets. Families must carry the full covariance (cheap:
n^2), and the open question is the higher-order joint structure (what killed the Gaussian copula at depth 32).

## Higher-order structure that composes: latent hub + Gaussian residual (2026-10-29, Kelvin's ask)
- Family: p(h_l) = integral of N(h; m_l(xi), S_l) p_l(xi) d(xi). The non-Gaussian, higher-order structure lives in a latent xi:
  either dense and low-dimensional (collapse modes, random nets) or sparse and high-dimensional (dictionary features,
  trained nets). Given xi, the state is Gaussian.
- Composes: given xi, each next layer is a wide sum of the residual, so near-Gaussian by the CLT; only the latent's law is
  propagated exactly (a k-dim Markov chain, or sparse features via transcoders).
- Cumulant tensors: low-rank Tucker through xi + per-neuron diagonal from rectification given xi.
- Treewidth: dense dependence is confined to the Gaussian part (always tractable); the latent is a size-k separator, so
  the DAG-over-neurons treewidth problem goes away.
- Tested by: the boundary-family lane (dense latent) and the sparse-code lane (sparse latent). Kill tests: whether the
  needed k grows with depth; whether the residual is conditionally Gaussian at near-boundary neurons.

## To study (info-aggregation track): H-Net, dynamic chunking (registered 2026-10-29 ~17:00Z)
Hwang, Wang, Gu 2025, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", https://arxiv.org/abs/2507.07955.
- A routing module predicts boundaries between adjacent elements from a similarity score.
- A smoothing module interpolates representations to soften uncertain boundaries.
- An auxiliary loss targets a downsampling ratio, so the network learns compression end to end.
- The encoder and decoder work better as SSMs ("an inductive bias for compression", Gu 2025).
Kelvin: "curious whether this is interesting for us to learn from on the info aggregation track." Possible links:
- similarity-based merging of adjacent elements as learned in-context dedup;
- a learned, bounded-size compressed state (the consensus-state bound Kelvin wants);
- SSM compression bias vs attention for aggregation.
Not yet read.

## Bidirectional importance of directions: Lyapunov / CLV view (2026-10-29 ~18:10Z)
Kelvin: "a 'bidirectional measure' of importance of directions (say model it for an infinite-layer random network), that
helps preserve important long-depth structures as much as possible."
- Rigorous version: covariant Lyapunov vectors (Ginelli et al., PRL 2007). Forward and backward iterations split each
  layer into expanding, neutral and contracting directions.
- Random-net prediction: for Gaussian matrix products, Lyapunov exponents are spaced about 1/(2n) apart (Newman 1986).
  So about 2n/L' directions survive a remaining depth L'; that is O(1) when depth is proportional to width. This fits the
  observed collapse to 2-5 effective dimensions at 256x32.
- Data-assimilation analogue with theorems: track only the unstable-neutral subspace; errors elsewhere decay (Trevisan
  and Uboldi 2004; Bocquet, Carrassi et al. 2017).
- Design: be exact (non-Gaussian) in the neutral/expanding subspace and loose elsewhere, so the description
  self-corrects. Caveat: ReLU gate flips are nonlinear. Being tested by the nongauss-family lane (Amendment 1 and after).

## ResNets with capped per-layer change as a theory testbed (registered 2026-10-29)
Kelvin: "beyond the ARC setup, we can also study resnets, where we can cap each layer to cause a small change on the resnet
only if needed or useful for theory". h_{l+1} = h_l + eps f_l(h_l). As eps -> 0 with L*eps^2 fixed, the net becomes an SDE,
and the activation law evolves by a Fokker-Planck flow, where each layer's closure error is small and controlled. This
separates "number of layers" from "total nonlinearity" (L*eps or L*eps^2) as the source of description difficulty. Papers in
kelvin-study-list.md (Li-Nica-Roy 2021 log-Gaussian; Hayou-Yang 2023; Tensor Programs VI; Bordelon et al. 2023).
