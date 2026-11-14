## 1. Literature: what is known about very narrow, very deep networks

Every reference below was checked against its arXiv abstract, proceedings page or full text on 29 October
(`work/narrow-deep/notes/refs.md`: 77 entries, 65 verified as recalled, 4 corrected, 3 bibliographic only, 5 with an
unverified detail such as a venue).

**What narrow nets can represent.**
- A ReLU net whose hidden layers are no wider than its input cannot approximate general functions, however deep:
  - width <= d_in fails (Lu et al. 2017; Hanin and Sellke 2017; Johnson 2019, for any width-<=d_in layered net);
  - every decision region of such a net is unbounded (Beise, Dias Da Cruz and Schroeder 2021);
  - disconnected regions need a layer wider than the input (Nguyen, Mukkamala and Hein 2018).
- One extra unit is enough. Width d_in + 1 approximates any continuous scalar function on a compact set (Hanin and
  Sellke 2017). Width d_in + 4 does so in L1 on all of R^n (Lu et al. 2017). For vector outputs the exact minimum is
  max(d_x + 1, d_y) in L^p (Park, Yun, Lee and Shin 2021) and max(d_x, d_y, 2) on compact domains (Kim, Min and
  Park 2024; Cai 2023 for leaky ReLU). General activations need d_in + d_out + 2 (Kidger and Lyons 2020).
- Depth can replace width cheaply. Any ReLU net is approximated by one of width O(d), with only a linear-factor
  increase in parameters (Vardi, Yehudai and Shamir 2022). A ResNet whose blocks each have a single hidden ReLU is
  universal (Lin and Jegelka 2018).
- Our random nets have width = input dimension, so they sit just below the universality threshold. A residual
  stream removes the obstruction.

**What depth separations do and do not buy.**
- Narrow deep nets can compute functions that shallow nets cannot: Telgarsky's sawtooth at width ~2-3 (Telgarsky
  2016), and Eldan and Shamir (2016).
- Gradient descent does not learn those functions when shallow nets cannot approximate them weakly (Malach, Yehudai,
  Shalev-Shwartz and Shamir 2021; Malach and Shalev-Shwartz 2019).

**Trainability at initialisation: the parameter is depth/width.**
- In a fully connected ReLU net, the variance of the squared activation length and of Jacobian entries grows like
  exp(c sum_l 1/n_l), i.e. exp(c L/n) at constant width (Hanin and Rolnick 2018; Hanin 2018).
- One input's squared output norm is log-normal with log-variance 5 sum 1/n_l (Hanin and Nica 2020, Comm. Math.
  Phys.). The NTK stays random and moves during training when L/n is not small (Hanin and Nica, ICLR 2020). The
  aspect ratio L/n controls the deviations from the Gaussian limit (Roberts, Yaida and Hanin 2022).
- Collapse:
  - at fixed width, the probability that a ReLU net is "born dead" (constant) tends to 1 with depth; with symmetric
    init P <= 1 - prod_l (1 - 2^-n_l) (Lu, Shin, Su and Karniadakis 2020);
  - the angle between two inputs shrinks exponentially with an e-fold depth of about n at finite width, against ~1/l
    at infinite width (Jakub and Nica 2024, JMLR). Their expansion is for n >> 1 and is checked numerically only at
    n = 256 and 1024;
  - unshaped ReLU covariances degenerate as depth/width grows (Li, Nica and Roy 2022);
  - representations lose rank with depth without normalisation (Daneshmand et al. 2020; Feng et al. 2022).
- Narrow linear nets are slow to train. Gradient descent on a width-1 depth-k linear net needs time exponential in k
  (Shamir 2019). Plain deep linear nets need width growing with depth (Du and Hu 2019); residual ones need only
  d_in + d_out (Zou, Long and Gu 2020).

**Residual and normalised narrow nets; infinite-depth limits.**
- Branches scaled by 1/sqrt(L) give a stable, non-trivial infinite-depth limit (Hayou et al. 2021, "Stable ResNet").
  It is the only non-trivial scaling for iid initialisations, and its limit is a neural SDE, not an ODE (Marion,
  Fermanian, Biau and Vert).
- The output is log-Gaussian, with parameters set by d/n (Li, Nica and Roy 2021). Width and depth limits commute
  under that scaling (Hayou and Yang 2023). Infinite-depth diffusion limits: Peluchetti and Favaro (2020). Neural ODEs
  (Chen et al. 2018), and when trained ResNets actually discretise them (Sander, Ablin and Peyre 2022; Cohen et al.
  2021).
- Feature-learning depth limits: Depth-muP (Yang, Yu, Zhu and Hayou 2024); depthwise hyperparameter transfer
  (Bordelon, Noci, Li, Hanin and Pehlevan 2024).
- Initialisations that make thin residual nets trainable without normalisation: Fixup (16/32/64-channel ResNets to
  10,000 layers on CIFAR-10), SkipInit (De and Smith 2020), ReZero (10,000 fully connected layers fitted to the
  CIFAR-10 training set; width not stated).
- Batch norm without skips makes gradients explode with depth (Yang et al. 2019).

**RNNs and SSMs: narrow state, deep in time, shared weights.**
- AlgZoo (ARC, January 2026): bias-free ReLU RNNs, 2nd argmax, hidden size 2-32, lengths 2-10.
  - Accuracy falls with length unless width grows: hidden 16 reaches 94.8% at length 10 (95.3% for the blog's
    example M_{16,10}); hidden 8 reaches 42.7%.
  - No zoo model is longer than 11.
- Vanilla RNNs have vanishing and exploding gradients (Pascanu, Mikolov and Bengio 2013). An identity-initialised
  ReLU RNN with 100 units does the adding problem at T = 150-400 and pixel-by-pixel MNIST, 784 steps (Le, Jaitly and
  Hinton 2015). Gated cells widen the trainable region (Chen, Pennington and Schoenholz 2018).
- An LSTM with two memory cells bridges a 1,000-step lag among 1,000 distractors (Hochreiter and Schmidhuber 1997).
- SSMs keep a small state per channel but many channels: S4 uses state 64; Mamba's default per-channel state is 16,
  and its Table 10 shows perplexity still improving from N = 1 to 16 (Gu and Dao). LRU: Orvieto et al. (2023).

**Which interesting tasks narrow-deep nets have been trained on.**
- With skips or gates, yes, routinely:
  - highway MLPs of 50 units x 100 layers on MNIST, matching 10-layer nets in training loss (Srivastava, Greff and
    Schmidhuber 2015); plain 71-unit nets degrade beyond 20 layers;
  - ResNet-1202 with 16-64 filters: 7.93% CIFAR-10 test error;
  - Fixup to 10,000 layers;
  - a width-1-per-block ResNet learns a 2-D disk that width-2 MLPs cannot (Lin and Jegelka 2018);
  - two-cell LSTMs on 1,000-step lags; AlgZoo's tiny RNNs.
- Without skips, only short nets:
  - widths 10-15 x 10-25 layers reach ~95% MNIST in one epoch, and 30 x 15 drops to 82% (Jakub and Nica 2024,
    Tables 7-8);
  - widths 15-20 train on MNIST at a speed that tracks sum 1/n_l (Hanin and Rolnick 2018).
- Not found:
  - any plain MLP of width <= 16 and depth >= 64 trained to non-trivial accuracy on a real task;
  - any learned (not hand-built) narrow-deep sorting network;
  - gradient-descent learning of Telgarsky-type targets.
- Our trained nets in section 4 add a width-16 RNN on 2nd argmax at T = 64 and a width-16 residual MLP of depth 64
  that sorts 16 numbers.
