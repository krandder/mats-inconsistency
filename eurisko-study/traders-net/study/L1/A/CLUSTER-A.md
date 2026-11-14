# L1 cluster A: markets in ML (worker 1)

Checked 2026-09-27 ~22:15Z. Each arXiv ID was fetched from arxiv.org/abs (title, authors, date
match the plan's attribution) and the full PDF was read (text extracted with pypdf into
/tmp/traders-net/L1/A/*.txt, not kept). ID table: `study/L1/A/verified-ids.csv`.
"Ideas 1-6" are the numbered ideas in `traders-net/CONTEXT.md`. Nothing here is a run; no PREREG
needed (literature only).

**All 10 plan IDs verified; none misattributed.** Hayek machine has no arXiv ID; located by DOI
via Crossref (abstract read for the 2000 paper; 1999 paper metadata only).

## Per-paper notes (5 lines each: built / overlap / not done)

**1. Balduzzi, "Cortical prediction markets", AAMAS 2014. https://arxiv.org/abs/1401.1465**
- Built: discretised spiking neurons whose STDP-like objective is shown to be a *proper scoring
  rule* (Thm 5); synapses are traders inside a neuron, neurons are traders in a "cortical market".
- Defines a neuron's "usefulness" as the sum of its active downstream weights and makes neurons
  maximise it via feedback spikes ("backpropagate incentives", Cor. 7).
- Experiments: two toy spiking nets (tracker, foveator; 20x20 grid, ~100-neuron layers); feedback
  plasticity raises accuracy 95->98% and 93->95%.
- Overlap: ideas 3 and 6 as framing (backprop as incentive flow, units as scoring-rule agents).
- Not done: no wealth/budget state, no ledger of payments, no bankruptcy/birth, no prices read by
  units, no attribution test against ablation, no modern backprop nets.

**2. Balduzzi, Vanchinathan, Buhmann, "Kickback", AAAI 2015. https://arxiv.org/abs/1411.6191**
- Built: Backprop decomposed into local learners ("agents that minimise their losses", Thm 1) with
  a regret bound for any scalar feedback (Thm 2); error signal factorises into global error x
  path "influence" (Thm 3).
- Kickback keeps only global error x one-layer influence tau_j; follows the gradient if the net is
  "coherent" (tau_j>0 for all nodes, Thm 4). Matches Backprop on SARCOS/Barrett regression.
- Overlap: idea 6 (backprop = accounting among local agents; influence = attribution share).
- Not done: no money, wealth or markets beyond the framing; single-output regression only.
- Useful for T1 lemma 1/7: an existing statement that backprop is a set of no-regret local
  learners glued by feedback.

**3. Storkey, "Machine Learning Markets", AISTATS 2011. https://arxiv.org/abs/1106.4509**
- Built (theory only; no experiments found in the text): utility-based agents trading
  Arrow-Debreu goods at market equilibrium.
- Log utility -> equilibrium price = wealth-weighted mean of beliefs (eq. 16: mixture / linear
  pool); exponential utility -> geometric mean, PoE with wealth irrelevant (eq. 21); risk-neutral
  -> weighted median. Local potentials and message passing as markets.
- Overlap: ideas 1, 5 (pool form is chosen by the utility), M6 arm (e) log vs exponential utility.
- Not done: no learning of agent beliefs, no inside-a-network traders, no delayed settlement,
  no attribution.
- Note for the premise "a convex pool cannot add independent evidence": Storkey's exponential
  utility gives a *normalised* geometric mean (weights 1/N_A), which also averages rather than
  adds log-odds.

**4. Storkey, Millin, Geras, "Isoelastic agents and wealth updates", ICML 2012. https://arxiv.org/abs/1206.6443**
- Built: markets of isoelastic-utility agents; equilibrium price = alpha-mixture with wealth-
  dependent components; online wealth updates = Bayesian model averaging (log agents);
  batch updates = one EM step of mixture weights.
- Experiments: 20 trees of a random forest (or NNs) as agents, 7 UCI sets, 30 repeats, test log
  likelihood; isoelastic market of trees beats RF, NN and DT (Table 2).
- Overlap: ideas 1, 5; M6 arms (c)/(d)/(e) (convex pool with weights from wealth; log vs other
  utilities). Very close to "wealth = realised P&L weights a pool".
- Not done: agents are *fixed pretrained* classifiers; only wealth adapts. Wealth is updated on
  the *same training set the trees were fit on* ("market wealths were adapted on the complete
  training set with 1 training epoch") - in-sample settlement, the memorisation problem our
  premise names. No prequential books, no shared computation, no births.

**5. Barbu and Lay, "Artificial prediction markets for classification", JMLR 13 (2012). https://arxiv.org/abs/1102.1465**
- Built: participants = (budget, betting function) with equilibrium price solved numerically;
  budget update after each example conserves total budget; update shown to be (approximate)
  maximum likelihood.
- Constant betting = linear aggregation (a random forest is a constant-betting market with equal
  budgets); linear/aggressive/logistic betting functions; specialised participants that bet only
  on their region (leaves of random trees).
- Experiments: thousands of synthetic sets, UCI, and CT lymph-node detection (AdaBoost 79.6% ->
  81.2% detection at 3 FP/vol).
- Overlap: ideas 1, 2 (price-dependent demand; aggressive betting ~ demand vs price), 5; M6 (d)/(f)
  (budget-bounded bets, pool weights from P&L); M7 "specialists only trade where they know".
- Not done: budgets trained on the same training data as the trees (in-sample again); fixed
  participants; no inside-network units, no multi-scale/fractal ownership, no delay.

**6. Kwee, Hutter, Schmidhuber, "Market-based RL in partially observable worlds", ICANN 2001. https://arxiv.org/abs/cs/0105025**
- Built: reimplementation of Baum-Durdanovic Hayek4 (Post-production-rule agents, auctions for the
  right to act, winner pays previous owner, execution tax proportional to compute time, birth
  when wealth > 1000 with parents receiving a 0.25 "copyright share" of children's profit,
  removal after 100 idle instances), plus a memory bit register for a toy POMDP.
- Result: Hayek solves aliased toy POMDP with the register; authors report reproduction was hard
  ("making Hayek stable still seems to remain an art"), larger tasks unstable.
- Overlap: ideas 6 (parent owns share of child = partial ownership), compute = cost (tax per
  compute time), bankruptcy/birth (M7, M8, K1).
- Not done: no probabilistic prices, no gradient learning, tiny populations; no scaling result.
- **Hayek machine itself:** Baum, "Toward a Model of Intelligence as an Economy of Agents",
  Machine Learning 35:155-185 (1999), https://doi.org/10.1023/A:1007593124513 (Crossref lists
  Baum only; Kwee cites Baum and Durdanovic); Baum and Durdanovic, "Evolution of Cooperative
  Problem Solving in an Artificial Economy", Neural Computation 12:2743-2775 (2000),
  https://doi.org/10.1162/089976600300014700 - abstract: random-code economy learns Blocks World
  with 200-block goal stacks (vs <=8 for competitors), half a Rubik's cube. Full texts not read.

**7. Chang, Kaushik, Weinberg, Griffiths, Levine, "Decentralized RL via local economic transactions", ICML 2020. https://arxiv.org/abs/2007.02382**
- Built: "cloned Vickrey society": primitives bid in a second-price auction for the right to
  transform the state; winner is paid by the next winner; each primitive has a clone so the top
  two valuations coincide, making truthful bidding Bellman-optimal (Thm 5.5); credit-conserving
  variant (CCV) is best.
- Experiments: bandits, Chain, Duality MDPs, Two Rooms (selecting pretrained options; adapts to a
  reward swap faster than PPO baselines, fewer weights shift) and Mental Rotation (composing
  affine transforms in front of a fixed MNIST classifier).
- Overlap: M7 (router as auction/broker; redundancy to fix incentives; credit conservation =
  our conservation residual); idea 6 (payments flow backward along the trajectory).
- Not done: no wealth (agents bid values, wealth not a budget), no probability pricing, few
  (<=~12) primitives, no shared trunk, no dense-net units.

**8. Sudhir, Tran-Thanh, "Market-based architectures in RL and beyond", AAMAS 2025 (5 pp). https://arxiv.org/abs/2503.05828**
- Built (position paper, no experiments found): "deep market" algorithm (bids capped by wealth,
  reward added to wealth) and a goods-factored market where agents own state axes.
- Thm 3.1: any ReLU MLP forward pass is a deep market; prices at equilibrium computed by
  backpropagating consumer bids (eq. 1, grad = Jacobian . price), generalising Wentworth 2018.
- Overlap: ideas 3, 4, 6 as framing ("markets generalise neural networks"; "price = derivative").
- Not done: no implementation, no training by market, no attribution, no measurements; it lists
  "integrated models that backprop by default" as future work - the space traders-net occupies.
- Extra prior art it cites, verified by search: Wentworth, "Competitive Markets as Distributed
  Backprop", LessWrong 2018, https://www.lesswrong.com/posts/brhWPoNsBN7za3xjs/competitive-markets-as-distributed-backprop
  (firms at equilibrium backpropagate output prices to input prices).

**9. Qi et al., "Economy of Minds", arXiv June 2026. https://arxiv.org/abs/2606.02859**
- Built: frozen-LLM agents = (wake-up prompt, action prompt, fixed bid, wealth); first-price
  auction for the right to act; bucket-brigade transfer W_t -= b_t + r_t, W_{t-1} += b_t; rent rho;
  wealthy agents mutate prompts (exploitation), negative wealth -> bankruptcy and replacement
  (exploration). Generalises Hayek (their footnote 2).
- Experiments: 5 agentic tasks (MATH, financial and scientific research, accelerator design,
  distributed-system optimisation); e.g. MATH 45.0% -> 60.0% vs ReAct; population starts from
  6 founders, 30 episodes; ablations of auctions, exploitation, rent.
- Theory (App. C): surviving bid frontier converges to best specialist's value (Thm 1); outcome
  reward suffices when auctions pick near-best agents (Thm 2); regret vs oracle coordinator (Thm 3).
- Overlap: Hayek-style wealth, rent, bankruptcy and births (M7/M8 rebirth, K1 split-and-die).
- Not done: adaptation only in prompt space with a frozen backbone (their stated limitation);
  no gradient training, no probabilistic pricing, no shared computation, tens of agents not
  thousands; no comparison to a plain credit-assignment baseline on the same units.

**10. Veness et al., "Gated Linear Networks", AAAI 2021 (arXiv 2019). https://arxiv.org/abs/1910.01526**
- Built: backprop-free nets where *every neuron predicts the target*: geometric mixing
  sigma(w . logit(p)) of the previous layer's probabilities, weights chosen by half-space
  context gating, each neuron trained by online gradient descent on its own log loss (convex,
  regret bound); base predictions and a bias input; probabilities and weights clipped.
- Results: universality in the limit, capacity comparable to deep ReLU nets, strong resistance to
  catastrophic forgetting (comparable to MLP+dropout+EWC).
- Overlap: closest *mechanism* to ideas 2 and 5 and M6: each unit emits a log-odds correction
  on the previous layer's prices with an explicit base prediction (market prior), local
  settlement by log loss, bounded bets via clipping (M6 f), layer-by-layer sequential market
  (idea 3). Geometric mixing with unconstrained w *can* add independent evidence (w not
  normalised), so it is the known fix for the convex-pool premise.
- Not done: no wealth or multiplicative budgets (weights are OGD parameters), no bankruptcy/birth,
  no fixed-point within a layer, no attribution/books; no shared learned trunk (features fixed).

## Cluster-level findings for the synthesizer

1. **"Backprop is a market / money flows backward like gradients" (ideas 3, 6) is framing prior
   art**: Balduzzi 2014 (incentives backpropagated, usefulness = downstream weights), Kickback
   2015 (backprop = local no-regret learners), Wentworth 2018 (price = derivative), Sudhir 2025
   Thm 3.1 (MLP forward pass is a deep market). None keeps a ledger on a trained modern net or
   tests the ledger against held-out ablation. Delta for traders-net: measured books + ablation
   validation (M1/M2), not the analogy.
2. **Wealth-weighted pools of classifiers (ideas 1, 5; M6 c-e) were run**: Barbu-Lay 2012 and
   Storkey 2012 on UCI with fixed pretrained participants. Both settle wealth **in-sample** (same
   training data the participants were fit on). Prequential settlement inside a network is not
   in this cluster.
3. **Log vs exponential utility pool (M6 e)** is Storkey 2011 theory (mixture vs normalised
   geometric mean) and Storkey 2012 experiment (isoelastic). Treat M6(e) as a replication inside
   a net, not a new question.
4. **Local log-loss units correcting a base prediction with clipped bets (M6 b, f)** is GLN
   (2019). M6 should cite GLN and ideally include a GLN-style geometric-mixing arm or say why not.
5. **Bankruptcy, birth, rent, parent's share of child profit (M7/M8 rebirth, K1, idea 6 partial
   ownership)**: Hayek (1999/2000), Kwee 2001 (copyright share 0.25, compute tax), Economy of
   Minds 2026 (rent, mutation, bankruptcy, 6 founders). All are action-auction economies with
   small populations and no gradient learning; reports of instability (Kwee) are relevant to K1.
6. **Auction routers with credit conservation (M7)**: Chang 2020 cloned Vickrey society; clones
   remove the incentive gap. Relevant to "router-as-broker" (T1 lemma 7) and to M7's leak metric.
7. Not found in this cluster: LI-style coherence before resolution, exploitability by a trader
   class, delayed/deductive settlement, shared-trunk cost sharing, thousands of traders sharing
   computation. These are the candidate novelty areas (to be checked against clusters B and C).

Candidate one-line delta from cluster A alone (for the synthesizer to reconcile): *prior market-in-ML
work either pools fixed pretrained classifiers with in-sample wealth (Storkey, Barbu-Lay), runs
small action-auction economies without gradients (Hayek, Chang, Economy of Minds), or states
backprop-as-market as an analogy (Balduzzi, Wentworth, Sudhir); traders-net would be the first to
keep prequentially settled books on the units of a gradient-trained network and test them against
exact ablations.* (Claim status: unverified until clusters B/C are checked.)

Adjacent, verified by abs page only (not read in full):
- Abernethy and Frongillo 2011, crowdsourced learning mechanism (participants wager on updates
  to a published hypothesis, paid by test-set improvement): https://arxiv.org/abs/1111.2664 -
  close to idea 2 (corrections to the market hypothesis paid by improvement).
- Hu and Storkey 2014, multi-period trading markets with a market maker optimise a global
  objective: https://arxiv.org/abs/1403.0648.
