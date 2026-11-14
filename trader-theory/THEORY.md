# Internal markets at the right representation

trader-theory, 29 Sep 2026. Advisor note for Kelvin's "economy inside a network". Every reference was resolved on 29 Sep 2026 (arXiv abstract page or Crossref record); numbers about our own runs come from `calibrated-ai/trader-economy/CATALOGUE.md`.

## 1. What is known (and what we should stop re-deriving)

**K1. A market on a latent, settled against the network's own later value of that latent, is predictive coding, and with linear settlement it approximates backprop.** Rao and Ballard's predictive coding is exactly "each layer bets on the next layer's activity and is paid by the error"; Millidge, Tschantz and Buckley show these local errors approximate backprop on arbitrary graphs. Synthetic gradients (Jaderberg et al.) are a learned *price* of a latent, settled against the true gradient; target propagation (Lee et al.) is a learned *settlement target* for a latent. All three exist, all work somewhat worse than backprop at fixed depth, and all are depth-local, which is the property we want. Stream 3 markets settled against later scratch states buy nothing beyond this.

**K2. "Markets on the future mean belief add nothing" is right about the expected payment and wrong about the estimator.** Settling an early trade against a later price instead of the outcome is temporal-difference learning (Sutton 1988). It leaves expected pay unchanged (martingale) but changes variance and the timing of credit, which is the whole reason TD exists. calibrated-ai's `q_targets` in `market-resnet/econ.py` is this and should be called TD, not a new market. Holland's bucket brigade is the 1985 version.

**K3. Uncertainty and variance markets must be run jointly with the mean.** Variance is not elicitable on its own; the pair (mean, variance) is (Osband's principle, Fissler and Ziegel 2016; elicitable properties in general: Lambert, Pennock and Shoham 2008). Kelvin's belief-training page already does this with the identity "expected squared movement equals expected Brier improvement". That identity says a variance market *is* a value-of-computation market (Russell and Wefald 1991; Hay et al. 2012). Once its price steers halting or routing it becomes a decision market (Hanson 2013; Othman and Sandholm 2010), and Chen, Kash, Ruberry and Shnayder 2014 prove those are only incentive-compatible when every action keeps positive probability. Arm C's "halt when predicted gain is at or below cost" breaks this: the gain head is trained only on steps that were run, so it never learns what an early stop cost. This is a concrete bug, fixed by continuing with probability epsilon and importance-weighting settlement.

**K4. Which question is worth pricing is value of information, and the market already prices it.** Under the log score with liquidity b, a trader who learns a self-checkable event E and moves the price from p(Y) to p(Y|E) expects to earn b times the KL divergence, so the expected subsidy for listing E is b times I(E; Y | questions already listed). Kelvin's "latent that, once known, flips most outputs" is the event with the largest expected price move, that is the largest conditional mutual information (Lindley 1956; Howard 1966). Greedy listing by conditional mutual information is CMIM feature selection (Fleuret 2004) and decision-tree splitting. Listing a question when b times its conditional information exceeds its compute cost is the information-bottleneck Lagrangian (Tishby, Pereira and Bialek): the bottleneck trade-off parameter is the ratio of market liquidity to the price of compute. So the selection principle is known; what is open is running it inside a network with learned questions and a compute cost.

**K5. Latents are not identifiable, so coordinates cannot be priced.** Without side information a latent is defined only up to an invertible transform (Locatello et al. 2019); with auxiliary conditioning, up to permutation and scaling (Hyvärinen, Sasaki and Turner 2019; Khemakhem et al. 2020). Anything priced must be a function of examples, not of coordinates: an event, a partition, a prediction. Our own rescaling result (E-RS-1 to E-RS-3) is the one-unit version of this.

**K6. The security space is the representation.** Any bounded-loss market maker is a convex cost function whose prices live in the convex hull of the securities' payoff vectors (Abernethy, Chen and Wortman Vaughan 2013); market scoring rules are follow-the-regularised-leader (Chen and Wortman Vaughan 2010); combinatorial markets are priced by inference over that security space (Hanson 2003), tractably by constraint generation on a restricted space (Dudík, Lahaie and Pennock 2012). Abernethy and Frongillo 2011 already ran a market whose securities are *hypotheses*, paid by their loss. Choosing which internal questions to price is choosing securities; the tools for pricing them coherently exist.

**K7. Logical induction settles claims by a later computation** (Garrabrant et al. 2016): markets over sentences whose truth is decided by a deductive process, inexploitable by any efficient trader. The MATS soup tried to evolve the consistency arbitrageur and it never appeared (`exp-soup-arbitrageur`, No gain). Lesson: install the arbitrageur, do not evolve it.

**K8. Depth extrapolation without markets is the bar to beat.** Recall of the input plus a progressive loss lets a small recurrent net extrapolate to far more iterations without "overthinking" (Schwarzschild et al. 2021; Bansal et al. 2022). HRM (27M parameters, 1000 examples, Sudoku, mazes, ARC) trains with deep supervision on the refined output and, per the paper, a one-step gradient rather than backprop through all steps; the ARC Prize ablation found the outer refinement loop is the driver and the hierarchy adds little. TRM does it with 7M parameters; Geiping et al. scale recurrent depth in an LLM. Our arms B and Bds are this ingredient (per-step log-score pay on the belief stream). The economy's honest claim is then not "better than backprop at depth 1x" but "the same ingredient with bounded-trust updates, a value-of-computation signal and an open question set at 100x to 10000x depth and O(1) memory in depth".

**K9. Markets as routers are mixtures of experts** (Jacobs et al. 1991; Shazeer et al. 2017): a self-selection auction where the highest bidder computes is a gate; fixed-share redistribution is load balancing; Baum's Hayek machine and Kwee, Hutter and Schmidhuber 2001 are the bidding-for-control precedents. Block-local losses (Nøkland and Eidnes 2019; Belilovsky et al. 2019; Hinton's forward-forward) are the layerwise version, and our block-local payment result (MATS, +0.86 rank correlation) is that.

**K10. Private markets.** A private *question* has no anchor and collapses to a constant, the failure BYOL and I-JEPA avoid only by asymmetry (Grill et al. 2020; Assran et al. 2023). Private *positions* in public markets are fine and are a memory. "Information that matters early but not later" is already paid: a market scoring rule pays whoever moves the price first (Hanson 2007; Kyle 1985 for the finance analogue), so nothing special is needed for early credit, only the ability to hold and close positions.

## 2. Best principled answer: price events, not coordinates

An internal question is a pair (E, C). E is a predicate the network evaluates on its own state for the current example, so "did E happen" needs no ground truth: it is self-certifying. C is a claim about something that is eventually settled: the outputs, or a later public price. The regulator opens Hanson conditional markets on Y given E and Y given not-E. The value of the question is b times I(E; Y | questions already listed), which the market pays out by itself. Non-identifiability disappears because E is a set of examples: any reparametrisation of the scratch state that decides E the same way is the same question, and two traders proposing the same partition in different coordinates hold the same asset.

This is not "outputs and Boolean combinations of outputs". E ranges over the *computation*: "the wavefront has reached row 10", "the two candidate paths share cell c", "cell (3,4) has one candidate digit left". Such questions are priced by their information about the outputs, and the conditional prices p(Y | E) are a new, learned feature basis chosen by information value per unit of compute. Questions useful only at step t (a wavefront position) are paid at t by the conditional log-score move and then close; that is how transient information is handled.

There are four honest ways to settle an internal claim, and no fifth:

1. Terminal, against y: claims in output space, including pair securities y_i y_j (Kelvin's covariance markets), which are combinatorial markets whose coherent pricing is inference in a pairwise model (K6).
2. Chained, against a later public price: TD (K2). Adds variance reduction and early credit, never information. Needs the terminal anchor; an unanchored chain collapses.
3. Self-certifying: the event itself, computed from the state now. Free to settle; only its conditional claim needs 1 or 2.
4. Decision settlement: claims conditional on an action the network takes ("if trader j runs next, the belief moves by at least delta"), settled only when the action happens, incentive-compatible only with an epsilon-random action rule (K3).

A market on "coordinate 17 of h at step 30 will be 0.3" fits none of these and should be refused.

Which questions get listed: a question auction. Any trader may list (E, horizon) by paying a listing fee in nats. The lister earns a royalty equal to the Aumann–Shapley (integrated-gradients) share of the log-score gain that passes through the conditional structure, measured on examples the lister did not trade on (held-out settlement blocks Goodhart on a noisy batch estimate of information). Questions whose royalty falls below the fee de-list and the slot respawns. This makes the growth of the representation itself economic and open-ended, which is the one thing gradient descent over a fixed architecture does not do, and it is the only place where I expect the economy to beat backprop rather than match it.

Kelvin's private-market wants, made settleable: memory becomes private *positions* in public markets; early credit is already in the market scoring rule; settlement is inherited from the public markets. Private *questions* are dropped.

Falsifier for the whole programme: if event markets never beat deep supervision on the belief stream plus input recall (arm Bds) at 1000x depth on mazes, then the economy is a valuation and control tool (pruning, who lives, when to stop) and not a learning method. Our results so far say that (E-MLP-1, E0-2).

## 3. Moonshots, ranked

Testbeds: prefix parity (running), mazes (maze-dataset, Ivanitskiy et al. 2023, and the easy-to-hard data of Schwarzschild et al.), Sudoku-Extreme (1000 examples, from HRM). Nets under 1M parameters; report accuracy against the ratio of test depth to training depth (10, 100, 1000, 10000) and peak memory against depth; 5 seeds, arm Bds and Bansal recall as baselines.

1. **Question auction (event markets).** Build: K question slots per position on the econ market ResNet; a slot is a hard binary predicate from a trader head (straight-through), a conditional belief adjustment given E, a listing fee, an integrated-gradients royalty settled on held-out examples, de-listing and respawn; local training only. Success: surviving questions carry conditional mutual information well above dead ones, and depth extrapolation (train 32, test 3200) at least matches Bds and beats econ with no questions. Falsified if surviving questions carry no conditional information, or extrapolation is no better than K = 0 over 5 seeds.
2. **Wealth cap as a depth-robust trust region.** The worst-case-loss cap on each trade in `econ.py` bounds the belief move per step. Hypothesis: it is what stops overthinking at 100x to 10000x depth without a progressive loss. Three arms: cap on, cap off, Bansal's progressive loss. Success: cap-on accuracy does not fall with depth ratio. Falsified if cap-off is as stable or the progressive loss dominates. Cheapest experiment on the list.
3. **Value-of-computation decision market, epsilon-continued (fix arm C).** Per-position market on squared belief movement over the next k steps, run jointly with the belief (K3), settled on realised movement; halting keeps probability epsilon of continuing, with importance-weighted settlement. Success: steps used grow with maze path length and puzzle hardness at equal accuracy per unit cost; realised gain on epsilon-continued steps matches the prediction. Falsified if epsilon-continued steps show large gains the head priced at or below cost (self-fulfilling stops) or the step histogram is flat across difficulty.
4. **Flow-conservation arbitrageur on mazes.** Securities per edge "the path uses e"; an installed, zero-risk arbitrage trader enforces flow conservation and triangle inequalities by trading; learned traders only propose. No-arbitrage prices of this family are the shortest-path dual, so consistency enforcement is the reasoning primitive, without gradients (K7). Success: exact paths on mazes 10x to 100x larger than training, with the arbitrageur's profit falling to zero over depth. Falsified if it adds nothing over chained TD on the same securities.
5. **Bottleneck markets (sufficient statistics for sale).** A trader sells a k-bit message per cell per step (discrete, hence an event partition and identifiable); downstream pays integrated-gradients royalties; a channel fee per bit. Information bottleneck with a price instead of beta (K4). Success: 2-bit messages reach the accuracy of the 128-wide scratch and bits concentrate on the wavefront or the most constrained cells. Falsified if no accuracy at 8 bits or bits spread uniformly.
6. **Distance ladder with monotonicity.** Nested securities "distance to goal at most d", monotone in d (as with by-date monotonicity in Kelvin's external markets), settled by chained TD against neighbours plus one, anchored on the path label; Value Iteration Networks (Tamar et al. 2016) as the gradient baseline. Success: prices track true BFS distances (rank correlation above 0.95) with no distance supervision. Falsified if prices do not converge within BFS depth.
7. **Private positions as memory.** Traders hold open conditional orders in public markets that only they read; settlement inherited. Test on Sudoku, where backtracking needs "I tried digit 5 in cell c". Success: closing profit is positive and accuracy beats scratch-only memory. Falsified if no gain.
8. **Question selection by expected movement against the MRV heuristic.** On Sudoku, candidate questions "cell c is digit d"; price expected movement; compare the learned order with most-constrained-variable. Success: the orders agree and more puzzles solve at fixed depth. Falsified if the learned order is unrelated to constraint counts.
9. **Chained settlement as variance reduction.** Quantify K2: a TD-lambda ladder on the belief stream; measure gradient variance and sample efficiency against lambda at equal final loss. Bounded, likely "informative". If no difference, drop mean markets on stream 3 for good.

## References

- Abernethy, J., Chen, Y. & Wortman Vaughan, J. (2013). Efficient market making via convex optimization, and a connection to online learning. ACM TEAC 1(2). https://doi.org/10.1145/2465769.2465777
- Abernethy, J. & Frongillo, R. (2011). A collaborative mechanism for crowdsourcing prediction problems. NeurIPS. https://arxiv.org/abs/1111.2664
- Assran, M. et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. CVPR. https://arxiv.org/abs/2301.08243
- Bansal, A. et al. (2022). End-to-end algorithm synthesis with recurrent networks: logical extrapolation without overthinking. NeurIPS. https://arxiv.org/abs/2202.05826
- Belilovsky, E., Eickenberg, M. & Oyallon, E. (2019). Greedy layerwise learning can scale to ImageNet. ICML. https://arxiv.org/abs/1812.11446
- Chen, Y., Kash, I., Ruberry, M. & Shnayder, V. (2014). Eliciting predictions and recommendations for decision making. ACM TEAC 2(2). https://doi.org/10.1145/2556271
- Chen, Y. & Wortman Vaughan, J. (2010). A new understanding of prediction markets via no-regret learning. ACM EC. https://doi.org/10.1145/1807342.1807372
- Dudík, M., Lahaie, S. & Pennock, D. (2012). A tractable combinatorial market maker using constraint generation. ACM EC. https://doi.org/10.1145/2229012.2229047
- Fissler, T. & Ziegel, J. (2016). Higher order elicitability and Osband's principle. Annals of Statistics 44(4). https://arxiv.org/abs/1503.08123
- Fleuret, F. (2004). Fast binary feature selection with conditional mutual information. JMLR 5. https://www.jmlr.org/papers/v5/fleuret04a.html
- Garrabrant, S. et al. (2016). Logical induction. https://arxiv.org/abs/1609.03543
- Geiping, J. et al. (2025). Scaling up test-time compute with latent reasoning: a recurrent depth approach. https://arxiv.org/abs/2502.05171
- Grill, J.-B. et al. (2020). Bootstrap your own latent. NeurIPS. https://arxiv.org/abs/2006.07733
- Hanson, R. (2003). Combinatorial information market design. Information Systems Frontiers 5. https://doi.org/10.1023/A:1022058209073
- Hanson, R. (2007). Logarithmic market scoring rules for modular combinatorial information aggregation. J. Prediction Markets 1(1). https://doi.org/10.5750/jpm.v1i1.417
- Hanson, R. (2013). Shall we vote on values, but bet on beliefs? J. Political Philosophy 21(2). https://doi.org/10.1111/jopp.12008
- Hay, N., Russell, S., Tolpin, D. & Shimony, S. (2012). Selecting computations: theory and applications. UAI. https://arxiv.org/abs/1207.5879
- Hinton, G. (2022). The forward-forward algorithm. https://arxiv.org/abs/2212.13345
- Howard, R. (1966). Information value theory. IEEE Trans. Systems Science and Cybernetics 2(1). https://doi.org/10.1109/TSSC.1966.300074
- Hyvärinen, A., Sasaki, H. & Turner, R. (2019). Nonlinear ICA using auxiliary variables and generalized contrastive learning. AISTATS. https://arxiv.org/abs/1805.08651
- Ivanitskiy, M. et al. (2023). A configurable library for generating and manipulating maze datasets. https://arxiv.org/abs/2309.10498
- Jacobs, R., Jordan, M., Nowlan, S. & Hinton, G. (1991). Adaptive mixtures of local experts. Neural Computation 3(1). https://doi.org/10.1162/neco.1991.3.1.79
- Jaderberg, M. et al. (2017). Decoupled neural interfaces using synthetic gradients. ICML. https://arxiv.org/abs/1608.05343
- Jolicoeur-Martineau, A. (2025). Less is more: recursive reasoning with tiny networks. https://arxiv.org/abs/2510.04871
- Khemakhem, I. et al. (2020). Variational autoencoders and nonlinear ICA: a unifying framework. AISTATS. https://arxiv.org/abs/1907.04809
- Kwee, I., Hutter, M. & Schmidhuber, J. (2001). Market-based reinforcement learning in partially observable worlds. ICANN. https://arxiv.org/abs/cs/0105025
- Kyle, A. (1985). Continuous auctions and insider trading. Econometrica 53(6). https://doi.org/10.2307/1913210
- Lambert, N., Pennock, D. & Shoham, Y. (2008). Eliciting properties of probability distributions. ACM EC. https://doi.org/10.1145/1386790.1386813
- Lee, D.-H., Zhang, S., Fischer, A. & Bengio, Y. (2015). Difference target propagation. ECML. https://arxiv.org/abs/1412.7525
- Lindley, D. (1956). On a measure of the information provided by an experiment. Ann. Math. Statist. 27(4). https://doi.org/10.1214/aoms/1177728069
- Locatello, F. et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. ICML. https://arxiv.org/abs/1811.12359
- Millidge, B., Tschantz, A. & Buckley, C. (2020). Predictive coding approximates backprop along arbitrary computation graphs. https://arxiv.org/abs/2006.04182
- Nøkland, A. & Eidnes, L. (2019). Training neural networks with local error signals. ICML. https://arxiv.org/abs/1901.06656
- Othman, A. & Sandholm, T. (2010). Decision rules and decision markets. AAMAS. https://doi.org/10.65109/eddc7137
- Rao, R. & Ballard, D. (1999). Predictive coding in the visual cortex. Nature Neuroscience 2. https://doi.org/10.1038/4580
- Russell, S. & Wefald, E. (1991). Principles of metareasoning. Artificial Intelligence 49. https://doi.org/10.1016/0004-3702(91)90015-C
- Schwarzschild, A. et al. (2021). Can you learn an algorithm? Generalizing from easy to hard problems with recurrent networks. NeurIPS. https://arxiv.org/abs/2106.04537
- Shazeer, N. et al. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. ICLR. https://arxiv.org/abs/1701.06538
- Sutton, R. (1988). Learning to predict by the methods of temporal differences. Machine Learning 3. https://doi.org/10.1007/BF00115009
- Tamar, A. et al. (2016). Value iteration networks. NeurIPS. https://arxiv.org/abs/1602.02867
- Tishby, N., Pereira, F. & Bialek, W. (2000). The information bottleneck method. https://arxiv.org/abs/physics/0004057
- Wang, G. et al. (2025). Hierarchical reasoning model. https://arxiv.org/abs/2506.21734
- ARC Prize (2025). The hidden drivers of HRM's performance on ARC-AGI. https://arcprize.org/blog/hrm-analysis
- Earlier fleet results cited by id: `calibrated-ai/trader-economy/CATALOGUE.md`; MATS side tracks: https://mats.ekelvin.com/experiments/ (login-protected; source /home/agent/memo-serve/mats-experiments/)

Addendum, 29 Sep 2026 (from calibrated-ai's mail #2554): Kelvin's hindsight-posterior settlement of latent markets (settle p(z|x) against q(z|x,y) ∝ p(z|x)P(y|z), chained backwards) is assessed in `THEORY-2.md`, section 0: it is generalised EM in market form, principled as a proper score against the posterior, and its falsifiers are experiment P7's arms.
