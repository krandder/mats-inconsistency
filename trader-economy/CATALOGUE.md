# Trader economy: experiment catalogue

One small experiment per question. Status: Gain, Informative, No gain, Problem, Running. Numbers are means with 95% t-intervals over seeds. Each entry also exists as `catalogue/<id>.json`. Design note: `../trader-economy-note-20260929.md`. Built by `catalogue.py`.

| id | question | status |
|---|---|---|
| E-T1-1 | Does an unregulated Kelly market of trader agents keep many agents alive? | Problem |
| E-T1-2 | With a regulator paying by counterfactuals, does pay follow each agent's true value? | Informative |
| E-T1-3 | How do read fees and compute cost change the economy? | Informative |
| E-AUD-1 | How do the pay rules treat substitutes and complements? | Problem |
| E-AUD-2 | How many random orders does sampled Shapley need? | Informative |
| E-T2-1 | With two layers and royalties, does money reach the useful first-layer agents? | Gain |
| E0-1 | Is a hidden unit needed, or can an agent be just some of the weights? | Gain |
| E0-2 | Does training each agent on its own Shapley pay reach the backprop solution? | Problem |
| E0-3 | When agents are weights, does their pay follow their true value? | Informative |
| E0-4 | Do useless weights go to zero or go bankrupt? | Informative |
| E0-5 | Is there a pay rule between leave-one-out and Shapley that both trains well and pays fairly? | Informative |
| E-IG-1 | Can Aumann-Shapley pricing (integrated gradients) replace Shapley pay at scale? | Gain |
| E-IG-2 | Where do IG and Shapley pay disagree? | Informative |
| E-IG-3 | Can pay be recomputed only every N steps on a sample of examples? | Gain |
| E-MLP-1 | Does training each hidden unit on its own Shapley pay match backprop? | Problem |
| E-MLP-2 | Do the other pay rules train hidden units as well as backprop? | Informative |
| E-MLP-3 | Does a hidden unit's pay track its true value? | Gain |
| E-MLP-4 | Do useless hidden units go bankrupt? | Gain |
| E-MLP-5 | Do two copies of the same hidden unit share its pay? | Gain |
| E-MLP-6 | Does unit-level Shapley pay work with two hidden layers? | Informative |
| E-RS-1 | Is counterfactual pay unchanged when one unit is rescaled without changing the function? | Informative |
| E-RS-2 | Can pay or cost based on activation size be gamed by rescaling? | Problem |
| E-RS-3 | Does normalising activations stop the rescaling game? | Gain |
| E-SPLIT-1 | Does Shapley pay an input and a processor by scarcity or by effort? | Informative |
| E-SPLIT-2 | How much of the pay reaches the inputs, and does a layer-by-layer (Owen) split differ from flat Shapley? | Informative |
| E-SPLIT-3 | Does training each unit on its own flat-Shapley or Owen pay work in a two-layer network? | No gain |
| E-MR-1 | Does paying every step of a recurrent residual net its own log-score change make it extrapolate to longer inputs? | Gain |
| E-MR-10 | If traders must fund the covariance markets themselves by voluntary contributions, does the economy still learn? | Problem |
| E-MR-11 | If each trader is charged for the covariance markets in proportion to its Aumann-Shapley credit (Lindahl pricing), does the economy learn as well as with house funding? | Gain |
| E-MR-12 | How do the GPU memory and time of one training step grow with depth for the local economic update and for backprop through depth? | Informative |
| E-MR-13 | Does the trader economy show the predicted risks: bubbles, fixed-point stall, TD divergence? | Informative |
| E-MR-2 | At equal compute, does priced halting (halt when predicted gain <= cost) beat PonderNet and random halting, with and without a real cost in training? | Gain |
| E-MR-3 | When training halts examples, how must the halted steps be settled (the decision-market bug)? | Problem |
| E-MR-4 | Does Aumann-Shapley credit split each step's pay sensibly across the x, b and h streams? | Problem |
| E-MR-5 | Without any backprop across depth, do traders paid only by covariance markets extrapolate in depth and length further than the same traders trained end to end? | Informative |
| E-MR-6 | In the trader economy, does halting when the VOC market's price falls below the compute cost beat random and fixed-depth halting at matched compute, and does training with the cost help? | Informative |
| E-MR-7 | Can the traders' own profit bids choose who acts (an auction for each step) and when to halt? | Problem |
| E-MR-8 | Does a stream of private slots, whose owners pay writers their Aumann-Shapley share and pay rent, carry useful state in the trader economy? | Problem |
| E-MR-9 | In the trader economy, do the wealth cap and the entry of new traders change what is learned? | Informative |
| E-P1-1 | Multi-horizon anytime-value heads (T2m mechanism): does halting on heads that predict future belief movement at horizons {1,2,4,8,16,32,64,cap} beat one-step, {1,cap}, myopic+entropy, Q-ACT, probe and simple rules at matched compute when the plateau length varies within a size and half the instances never resolve? | No gain |
| E-P1-2 | Multi-horizon anytime-value heads on mazes: does the label-free test-time-refit head (R9) beat the simple-rule envelope, the refit consistency probe and Q-ACT at matched pixel-steps? | No gain |
| E-P1-3 | Multi-horizon anytime-value heads (label-free KL settlement, Q3): after a label-free test-time refit on the unlabelled maze pool, are KL-settled multi-horizon heads (R9) level-calibrated in every bin at maze sizes 59 and 99, and when movement stops tracking information, does sufficiency (realized gain G vs KL K) or the martingale condition (entropy drop vs K) fail? | No gain |
| E-P2-0 | Stage-0 toy: can a tree of latent codes hold a long-range parity choice that a factorised belief cannot? | Problem |
| E-P2-1 | Are the P2 maze sets, decoders and training recipe valid (unit tests, decoder trap, U-canon gate, floor rule)? | Problem |
| E-P2-2 | On mazes with tied shortest paths, does an exactly priced quadtree latent-parent market (M-tree) find a valid shortest path more often than commitment baselines (canonical labels, hard-EM)? | Problem |
| E-P2-3 | Does the M-tree code carry the route choice, and do independent per-tile mixtures (M-tile) suffice? | Informative |
| E-P2-4 | Does tie VSP keep rising with more iterations and hold on larger mazes, and does the canonical rule extrapolate? | Informative |
| E-P4-0 | On exact program tables, does flip-Shapley capture latent worth that reach, LOO and mutual information miss? | Informative |
| E-P4-1 | On nets with known circuits, does Aumann-Shapley pay (AS/IG) rank latents like the net's exact Shapley? | Informative |
| E-P4-2 | Is AS cheaper than sampled Shapley for the same rank quality at n>=16 latents? | Informative |
| E-P4-3 | With noisy latents, does AS-PAY track usefulness u and AS-SOFTFLIP track reach r? | Informative |
| E-P4-4 | On naturally trained nets (no SIIT), does AS rank aligned codes better than LOO, above a random-init null? | Informative |
| E-P4-5 | Do markets on the latents AS picks improve market-resnet depth extrapolation over plain fine-tuning and random picks? | No gain |
| E-P5-1 | Can an online market for private per-cell memory (owners post TD-learned prices, writers paid at write time) learn maze path-parity answered only at T? | Problem |
| P0-1 | Does the P0 harness reproduce DT-Recall and give trustworthy strict-fp32 numbers? | Informative |
| P0-2 | Does a width-64 DT-Recall retrain reach the maze bar (>= 0.90 on 33x33 at 300 iterations)? | Informative |
| P0-3 | Does DT-Recall 1D reach the prefix-sum bar on much longer strings, and does a DT-L baseline? | Informative |
| P0-4 | Does shortening the credit horizon to k=1 (chunked backprop) hurt DT-Recall's extrapolation? | Informative |
| P0-5 | How does the maze bar model hold up under distribution shift? | Informative |
| P0-6 | How much room do these puzzles leave for smarter halting (value-of-computation markets)? | Informative |
| P0-7 | Did market-resnet's pilot gain come from per-step pay, or from lacking the progressive loss? | Informative |
| P3 | Does Shapley-type pay make latent traders pick the output-natural factors rather than a rotation, beyond a mechanism-sparsity baseline? | No gain |
| E-MR-14 | What carries the trader economy's 100x extrapolation on prefix sums: the market, random training order, the detached per-step loss, the VOC targets or the per-trader optimizers? | Informative |
| E-MR-15 | Does the economy's 0.93 against 0.12 gap at 100x survive the strongest fair backprop baseline, Bansal et al. 2022 DT-recall with progressive loss? | No gain |
| E-MR-16 | Which regularizer stops the belief-logit runaway past the training length and keeps the 100x accuracy: an epsilon floor, an L2 fee on the belief logits, a logit cap (bounded liquidity) or a trade-size cap? | Informative |
| E-MR-17 | Does belief liquidity that grows with depth (b_t = t or sqrt(t), positions carried over, each step settled at its own liquidity) stop the runaway and keep the 100x accuracy? | No gain |
| E-MR-18 | Does the belief-logit runaway itself cost accuracy at 100x? | Informative |
| E-MR-19 | At a single length of 64, does VOC-priced halting beat an entropy threshold on the belief and fixed depth at matched mean steps? | Informative |
| E-MR-20 | At a fixed length of 32, how does training-step memory grow with depth for full, truncated and checkpointed BPTT and for the batched and online economy? | Informative |

## E-T1-1: Does an unregulated Kelly market of trader agents keep many agents alive?

Status: Problem

Setup: 10 agents, each reads 1-3 of 12 synthetic features, trains an online logistic model on its own log loss and bets half its wealth in an LMSR (b=0.5). Fees per feature read and compute cost; bankrupt agents are replaced.

Result: No. Wealth concentrates on one agent: 1.8±1.0 alive at the end, effective number of agents 1.2±0.6. Output log loss 0.543±0.018 vs 0.459 for logistic regression on all features (Bayes 0.436). Pay vs true value: Spearman 0.12±0.14.

Files: `econ.py`, `results/summary.json`

## E-T1-2: With a regulator paying by counterfactuals, does pay follow each agent's true value?

Status: Informative

Setup: Same 10 agents report truthful logits; the regulator fits a combiner and pays each epoch's pot by one rule. True value = Shapley of the information in the union of the agents' features.

Result: Spearman(pay, true value): LOO 0.81±0.10, MSR one order 0.54±0.14, MSR over 16 orders 0.68±0.05, exact Shapley 0.76±0.04, Kelly 0.12±0.14. Budget: LOO pays 0.58±0.13 of the pot; the MSR rules pay 1.02-1.03. All 10 agents alive under MSR-16 and Shapley; 5.2±2.8 under LOO. Output log loss 0.52-0.54, still about 0.06 nats worse than logistic regression.

Files: `econ.py`, `results/summary.json`, `charts/toy1_pay_vs_value.json`, `charts/toy1_logloss.json`

## E-T1-3: How do read fees and compute cost change the economy?

Status: Informative

Setup: MSR-16 and Kelly, fee per feature read in {0, 0.004, 0.012, 0.03} x compute per step in {0, 0.004, 0.012, 0.03}, 3 seeds. Also fees burned instead of paid to feature owners.

Result: At fee 0.03 or compute 0.03 no agent survives and the output falls to the prior (0.693). Best log loss 0.520±0.069 at fee=compute=0.004. Burning the fees leaves 3.4±1.4 agents alive (10 when owners are paid) and pay tracks value worse (0.30±0.36 vs 0.68±0.05).

Files: `econ.py`, `results/summary.json`

## E-AUD-1: How do the pay rules treat substitutes and complements?

Status: Problem

Setup: 10 designed agents, one feature each: x0 and its noisy copy x4, x1 and its copy x5, a complementary pair x6/x7 (useless alone, strong together), weak x2/x3, noise x8/x9. Pay in nats per epoch.

Result: Substitutes: Shapley pays x0 16.1 and its copy 9.2 (true values 19.0 and 11.8). LOO pays both copies about 0 or less (x4: -1.9). MSR in one order gives the credit to whichever comes first (x4: -1.4 forward, 21.5 reversed). Complements: every rule pays x6 and x7 below zero (Shapley -0.6 and -1.3) while their true values are +7.1 and +7.3. An agent that reports only its own belief carries almost none of the pair's information.

Files: `econ.py`, `results/summary.json`, `charts/audit_categories.json`

## E-AUD-2: How many random orders does sampled Shapley need?

Status: Informative

Setup: Same designed agents; relative L1 error of MSR averaged over K orders vs exact Shapley.

Result: Error 0.83±0.07 at K=1, 0.36±0.05 at K=4, 0.18±0.01 at K=16, 0.09±0.01 at K=64. Rank agreement with true value: LOO 0.23±0.17, MSR one order 0.28±0.21, MSR-16 0.66±0.15, exact Shapley 0.69±0.14.

Files: `econ.py`, `results/summary.json`, `charts/msrK_error.json`

## E-T2-1: With two layers and royalties, does money reach the useful first-layer agents?

Status: Gain

Setup: 8 first-layer agents; 4 readers each read 3-4 of them and pass half of their positive earnings upstream, split by a rule.

Result: Equal-split royalties pay noise agents nearly as much as useful ones (0.89±0.48 vs 1.04±0.21); Spearman with true value 0.20±0.26; log loss 0.657±0.071. Splitting by Shapley pays noise agents -0.008±0.011 and useful ones 1.55±0.17; Spearman 0.71±0.16; log loss 0.579±0.025. Kelly readers collapse to 1.4±0.7 alive.

Files: `econ.py`, `results/summary.json`, `charts/toy2_pay_vs_value.json`, `charts/toy2_useless_died.json`

## E0-1: Is a hidden unit needed, or can an agent be just some of the weights?

Status: Gain

Setup: Softmax regression, 8 inputs, 3 classes. Agents own one weight (24 agents), one input's weights (8), or one class's weights (3). Trade = additive logit contribution; ledger, compute cost per active weight, bankruptcy and respawn. 800 rounds of 250 fresh examples.

Result: Weights work as agents at every granularity; the ledger balances in every run. With leave-one-out pay, each agent's own-pay gradient is exactly the global gradient: at zero cost the weights match backprop to 0.000 relative difference.

Files: `exp0.py`, `results/exp0_summary.json`

## E0-2: Does training each agent on its own Shapley pay reach the backprop solution?

Status: Problem

Setup: Same model; each agent follows the gradient of its own sampled-Shapley pay (a dropout-like objective). Cost 0.01 per active weight.

Result: No, except at the coarsest grain. Input-column agents: test log loss 0.808±0.003 vs 0.743±0.001 for backprop (Bayes 0.738); weights 68% apart. The copy of x0 gets 0.73 of x0's weight (backprop: 0.04), and the complementary pair is underweighted (0.37 vs 1.70). Single-weight agents: 0.808±0.006. Class-row agents (3): 0.743±0.001, same as backprop. MSR in one order: 0.815±0.019.

Files: `exp0.py`, `results/exp0_summary.json`, `charts/exp0_logloss.json`, `charts/exp0_weights.json`

## E0-3: When agents are weights, does their pay follow their true value?

Status: Informative

Setup: True value = Shapley of the refit game (best log score using only the coalition's weights). Pay averaged over the second half of training.

Result: Shapley-trained column agents: Spearman 0.92±0.02. Shapley pay evaluated at the backprop solution: 0.80±0.05; IG pay there: 0.82±0.03. LOO pays 2.1 times the pot and gives each complement 0.52, the value of the whole pair; for class-row agents it pays only 0.85.

Files: `exp0.py`, `results/exp0_summary.json`, `charts/exp0_pay_vs_truth.json`

## E0-4: Do useless weights go to zero or go bankrupt?

Status: Informative

Setup: Noise inputs x6, x7. Cost per active weight 0, 0.01 or 0.05 per round (a useful column earns about 0.03-0.25 per round).

Result: At cost 0.01 noise agents go bankrupt repeatedly (6.8±0.6 deaths per run vs 1.2±1.0 for all useful agents together, Shapley arm), but their weights do not go to zero: the two noise inputs keep a total |w| of 0.10±0.04, the same as backprop with the same L1. At 0.05 the cost also kills useful agents (68±1 deaths per run) and log loss rises to 0.823.

Files: `exp0.py`, `results/exp0_summary.json`

## E0-5: Is there a pay rule between leave-one-out and Shapley that both trains well and pays fairly?

Status: Informative

Setup: Semivalue: every other agent is present independently with probability pi (pi near 1 is LOO, whose gradient is backprop; pi = 0.5 is the Banzhaf value, close to Shapley here). Payments rescaled to sum to the pot. Column and single-weight agents, cost 0.01.

Result: A clean dial. Column agents, pi = 0.5 / 0.8 / 0.95: test log loss 0.808±0.004 / 0.777±0.001 / 0.750±0.001 (backprop 0.743); Spearman(pay, true value) 0.91±0.02 / 0.84±0.01 / 0.71±0.02. Weight on the copy of x0: 0.74 / 0.53 / 0.17 of x0's. Single-weight agents give the same picture (0.807 / 0.767 / 0.748).

Files: `exp0.py`, `results/exp0_semi_summary.json`

## E-IG-1: Can Aumann-Shapley pricing (integrated gradients) replace Shapley pay at scale?

Status: Gain

Setup: Designed agents with additive logits (informative, substitutes, complements, noise, harmful), binary and 3-class, n = 10, 100, 1000; 3 seeds. IG with m midpoint steps vs exact Shapley (n=10) or 400-1000 sampled orders.

Result: IG with m=20 is off from Shapley by 0.21±0.08 (n=10), 0.25±0.04 (n=100), 0.14±0.02 (n=1000) relative L1; Spearman 0.94-0.98. Sampled Shapley needs K=64 orders to match: 704 / 6,464 / 64,064 passes against IG's 20. Budget gap of IG: 0.15-1.1% at m=20, at most 0.2% at m=50.

Files: `exp_ig.py`, `results/ig_summary.json`, `charts/ig_error_vs_shapley.json`

## E-IG-2: Where do IG and Shapley pay disagree?

Status: Informative

Setup: Pay by category of designed agent, n = 10, exact Shapley.

Result: IG pays complements about twice what Shapley pays (37±17 vs 18±12 nats per agent, binary), substitutes about 30% less (23 vs 34) and informative agents about 13% less; noise and harmful agents agree within about 2%.

Files: `exp_ig.py`, `results/ig_summary.json`, `charts/ig_categories.json`

## E-IG-3: Can pay be recomputed only every N steps on a sample of examples?

Status: Gain

Setup: Online softmax regression, 10 input agents, 1000 steps of 512 examples; IG pay every N steps on s examples, scaled up.

Result: Cumulative pay error vs full recomputation: 5.5±3.0% (binary) and 3.8±1.1% (3-class) at N=10, s=256, which costs 5%; 7.4±6.9% at N=50, s=512 (2% of the cost); 22±12% at N=50, s=32.

Files: `exp_ig.py`, `results/ig_summary.json`, `charts/ig_online_subsample.json`

## E-MLP-1: Does training each hidden unit on its own Shapley pay match backprop?

Status: Problem

Setup: One hidden layer of 32 tanh units, 4 of them wired to noise only; a unit owns its incoming weights, bias and outgoing weight. Pay = Shapley (16 orders, zero-out ablation) of the batch log-score gain; each unit follows the gradient of its own pay (8 predecessor sets). Synthetic stream, 1200 rounds, cost 0.0005 per connection; wdbc 70/30 split plus 4 noise columns.

Result: No. Synthetic test log loss 0.609±0.012 vs backprop 0.435±0.004 (logistic regression 0.433, true model 0.4355). The function is right but about 2.8 times too confident: one fitted output scale (0.36±0.03) brings it to 0.435, and its logit gap to backprop then (0.13±0.04) equals the gap between two backprop runs (0.13±0.03). This is dropout without the test-time rescale. wdbc: 0.55±0.31 vs 0.104±0.043; the rescale does not rescue it (0.32±0.37), it also overfits.

Files: `exp_mlp.py`, `results/mlp_summary.json`, `charts/mlp_logloss_synth.json`, `charts/mlp_logloss_wdbc.json`

## E-MLP-2: Do the other pay rules train hidden units as well as backprop?

Status: Informative

Setup: Same network; units follow the gradient of their own LOO pay or their fixed-order MSR pay.

Result: LOO gives exactly the backprop function at cost 0 (logit gap 0.000) and 0.435 at cost 0.0005, but it pays out only 21% of the pot and useful units go bankrupt 142±24 times per run. Fixed-order MSR reaches 0.439±0.005 with 1-2 surviving units (29±6 connections vs 386 for backprop): the first unit in the order takes all the value.

Files: `exp_mlp.py`, `results/mlp_summary.json`, `charts/mlp_logloss_synth.json`

## E-MLP-3: Does a hidden unit's pay track its true value?

Status: Gain

Setup: Backprop-trained network. True value = unit Shapley with 200 orders on held-out data (Neuron Shapley style); pay from 16 orders on the training batch.

Result: Spearman(pay, true value): Shapley-16 0.965±0.027, LOO 0.84±0.10, fixed-order MSR 0.83±0.13 (synthetic); Shapley-16 0.969±0.016 on wdbc. Zero-out and mean-ablation truths agree (1.00 synthetic, 0.96 wdbc). Inside the over-confident Shapley-trained network LOO pay ranks units backwards (-0.85±0.10).

Files: `exp_mlp.py`, `results/mlp_summary.json`, `charts/mlp_pay_vs_value.json`

## E-MLP-4: Do useless hidden units go bankrupt?

Status: Gain

Setup: 4 unit slots wired only to noise; bankrupt units respawn with fresh weights.

Result: Yes under Shapley pay: at any cost above zero every noise slot went bankrupt and stayed dead after respawn (4.8-5 bankruptcies per run over the 4 slots) and ended with zero outgoing weight. Backprop only shrinks them (|outgoing weight| 0.016 vs 0.24 for useful units).

Files: `exp_mlp.py`, `results/mlp_summary.json`

## E-MLP-5: Do two copies of the same hidden unit share its pay?

Status: Gain

Setup: Copy the highest-paid unit and halve the outgoing weight of both copies, so the function does not change.

Result: Shapley pays each copy 0.52±0.03 and 0.51±0.04 of the original's pay (sum 1.03). LOO pays each 0.27 (sum 0.54): it under-pays redundancy. Fixed-order MSR splits 0.58/0.42 depending on which copy comes first.

Files: `exp_mlp.py`, `results/mlp_summary.json`

## E-MLP-6: Does unit-level Shapley pay work with two hidden layers?

Status: Informative

Setup: 16 + 16 tanh units, 800 rounds, synthetic; all 32 units are players in one game on the network's log score.

Result: The first layer gets 0.52±0.01 of the Shapley pay against a true share of 0.54±0.02; Spearman(pay, true value) 0.97±0.03. Training on own Shapley pay shows the same over-confidence: 0.593, or 0.449 after one output scale.

Files: `exp_mlp.py`, `results/mlp_summary.json`

## E-RS-1: Is counterfactual pay unchanged when one unit is rescaled without changing the function?

Status: Informative

Setup: ReLU network, 16 units, backprop-trained. Each unit is split into an in-agent (incoming weights) and an out-agent (outgoing weight). Multiply the incoming weights by c and divide the outgoing weight by c, c in {0.1, 10}; top 6 units, 5 seeds.

Result: Yes. The function is identical (largest logit change 2e-15) and Shapley, mean-ablation Shapley and LOO pay are unchanged (ratio 1.000).

Files: `exp_rescale.py`, `results/rescale_summary.json`, `charts/mlp_rescale_ratio.json`

## E-RS-2: Can pay or cost based on activation size be gamed by rescaling?

Status: Problem

Setup: Same rescaling. Rules that use magnitude: pay split by share of activation, readers paying per unit of activation read, L2 cost on weights.

Result: Yes. At c = 10 the in-agent's activation-share pay rises 6.13±0.18 times and the other in-agents lose 39%, for the same function. Per-activation payments move 10 times. L2 cost moves 100 times onto the in-agent at c = 10, or onto the reader at c = 0.1.

Files: `exp_rescale.py`, `results/rescale_summary.json`, `charts/mlp_rescale_ratio.json`

## E-RS-3: Does normalising activations stop the rescaling game?

Status: Gain

Setup: Standardise each activation to unit standard deviation, with the statistics recomputed after the rescale (and, as a check, frozen before it).

Result: Mostly. Activation-share pay, per-activation payments and the reader's L2 cost become exactly invariant. Two gaps remain: L2 on the incoming weights still moves 100 times (the in-agent can escape it by shrinking, at no cost to the function), and with statistics frozen before the rescale the game is back (6.19 times).

Files: `exp_rescale.py`, `results/rescale_summary.json`, `charts/mlp_rescale_ratio.json`

## E-SPLIT-1: Does Shapley pay an input and a processor by scarcity or by effort?

Status: Informative

Setup: Logit = a·x times input gates times processor gates. (a) 1 input read by k duplicate processors, (b) 1 processor reading k duplicate inputs, (c) one of each. Duplicates either substitute (any one suffices) or add up (each carries 1/k). Exact Shapley and IG (m = 20, 50); 5 seeds.

Result: By scarcity. In (a) the input's Shapley share is 0.50, 0.667 and 0.833 for k = 1, 2, 5 when any one processor suffices, and 0.50, 0.607±0.003, 0.672±0.005 when they add up; (b) is the mirror image (input 0.333 and 0.167 at k = 2, 5). (c) splits 0.50/0.50, which is the symmetry axiom. IG agrees in the substitute case (0.59 and 0.78) but gives exactly 0.50 in every additive case: the logit grows as t squared along the straight path, so IG cannot see replaceability there.

Files: `exp_split.py`, `results/split_summary.json`, `charts/split_scarcity.json`

## E-SPLIT-2: How much of the pay reaches the inputs, and does a layer-by-layer (Owen) split differ from flat Shapley?

Status: Informative

Setup: ReLU network 12 inputs -> 8 -> 4 -> 1, trained by backprop; two first-layer units read only noise features. Players: inputs and units. Flat Shapley and flat IG over all players; nested Owen value (each unit's union holds what it reads; an input read by several units is split into one player per path, i.e. Winter's levels value on the unrolled tree). Output bias fixed at the prior; 5 seeds.

Result: Share of the total going to inputs / layer 1 / layer 2: flat Shapley 0.61±0.06 / 0.46±0.59 / -0.07±0.59; IG 0.31±0.02 / 0.63±0.31 / 0.06±0.32; nested Owen 0.19±0.07 / 0.10±0.34 / 0.71±0.34. In the nested split each unit keeps about 0.7 of its union's value, since it can veto the union, so the inputs' share shrinks at every layer. Input pay ranks inputs like their drop-feature value (Spearman: flat 0.89±0.10, Owen 0.85±0.10, IG 0.78±0.10) but only 0.53-0.61 like their information Shapley; the substitute pair gets about 0 where information Shapley gives it 0.067. Noise-only units get -0.002±0.003.

Files: `exp_split.py`, `results/split_summary.json`, `charts/split_layers.json`

## E-SPLIT-3: Does training each unit on its own flat-Shapley or Owen pay work in a two-layer network?

Status: No gain

Setup: 1000 rounds, ledger checked in code, bankruptcy and respawn. Arms: backprop, own flat-Shapley pay (dropout-like), own Owen pay (exact gradient for one sampled nested order, pay every 10 rounds).

Result: No. Test log loss at cost 0: backprop 0.440±0.001, Shapley 0.499±0.013, Owen 0.495±0.023 (Bayes 0.4355). At cost 0.002 per connection both economy arms collapse towards the prior (0.671 and 0.639; about 11.5 bankruptcies per 6 useful first-layer slots) while backprop stays at 0.439.

Files: `exp_split.py`, `results/split_summary.json`, `charts/split_training.json`

## E-MR-1: Does paying every step of a recurrent residual net its own log-score change make it extrapolate to longer inputs?

Status: Gain

Setup: Market ResNet phase 1: one recurrent residual block (width 128) with streams x (input, re-injected), b (belief logits, each step paid its LMSR/log-score change) and h (scratch); prefix-sum parity, trained on lengths 8-32 with T = 24 steps, 10000 iterations, batch 128, 3 seeds per arm, L4 spot GPUs. Arms: A final loss only, B per-step market scoring rule, Bds deep supervision, C B plus a gain head for priced halting, Ctt train-time halting, D PonderNet, P Bansal et al. 2022 progressive loss (the bar). Tested at L64 and L128 to a cap of twice the length.

Result: Yes. Exact accuracy L64 / L128: B, Bds and C 1.000 / 1.000 in every seed; Ctt 1.000 / 0.999; A 0.63 +- 0.80 / 0.11 +- 0.39; P 0.99 +- 0.03 / 0.64 +- 0.82 (seeds 0.78, 0.87, 0.26); PonderNet 0.007 / 0.000. C meets the bar (extrapolates at least as well as A) and beats Bansal at L128. Deep supervision does as well, so the gain comes from scoring every step, not from the market form of the score. Monotone improvement is not guaranteed: at L128 only 0.50 (B) to 0.75 (Bds) of examples never lose log score at any step.

Files: `market-resnet/RESULTS.md`, `market-resnet/mresnet.py`, `market-resnet/train.py`, `market-resnet/evaluate.py`, `market-resnet/analyze_mr.py`, `market-resnet/results/summary.json`, `market-resnet/charts/p1_fixed_steps_L64.json`, `market-resnet/charts/p1_fixed_steps_L128.json`, `GPU-LEDGER.md`

## E-MR-10: If traders must fund the covariance markets themselves by voluntary contributions, does the economy still learn?

Status: Problem

Setup: Phase 2b: the cov markets' LMSR payout is charged to traders; each trader decides per episode whether to fund (Bernoulli, REINFORCE on its own profit), unfunded episodes settle no cov market, funders split the payout; 2 seeds.

Result: No: free riding (Samuelson 1954). Funders pay 0.125 per episode (Lindahl: 2.23), most cov markets go unsettled, and round-robin accuracy falls to 0.98 at L32, 0.21 at L320 and 0.00 at L3200 (house-funded: 1.00, 1.00, 0.93); training loss spikes above 0.1 nats per bit 3-4 times after iteration 1000.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/run_p2b_vm6.sh`, `market-resnet/results/summary.json`

## E-MR-11: If each trader is charged for the covariance markets in proportion to its Aumann-Shapley credit (Lindahl pricing), does the economy learn as well as with house funding?

Status: Gain

Setup: Phase 2b: cov markets always open; their LMSR payout is charged to traders in proportion to each trader's Aumann-Shapley credit of stream 3 in its own belief earnings (Foley 1970); 2 seeds.

Result: Yes. Traders pay 2.23 per episode and end with wealth 930-1,510 (house-funded: about 2,850), and accuracy matches or exceeds the house-funded economy: L3200 1.00 from 1600 steps (house-funded 0.88 at 1600, 0.93 at 3200); halting at L64 c = 0.003: 7.8 steps, 0.986.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/run_p2b_vm6.sh`, `market-resnet/results/summary.json`

## E-MR-12: How do the GPU memory and time of one training step grow with depth for the local economic update and for backprop through depth?

Status: Informative

Setup: memdepth.py: one training step at depth D = L in 32, 320, 1000, 3200, 10000, batch 16, one L4 (24 GB): the economy's online update (streaming, O(1) records in D), its batched update (stores the rollout, no graph) and BPTT through the same traders.

Result: Informative (audit 2026-09-29): a known property of truncated or detached updates, measured on a configuration with no accuracy result. Peak memory of one step at D = L = 32 / 320 / 1000 / 3200 / 10000: online economy 0.006 / 0.09 / 0.30 / 1.16 / 4.20 GB (10000: 1948 s); batched economy 0.07 / 7.4 GB, out of memory at 1000; BPTT 0.09 / 15.9 GB, out of memory at 1000. The 4.2 GB row is train_step_online with voc=0, never trained to completion or evaluated for accuracy; the batched voc=1 economy behind the 0.93 runs out of memory at D = 1000 like BPTT. memdepth.py set L = D, mixing depth and length. Fixed L = 32 with truncated (k=1) and checkpointed BPTT: RESULTS.md section 17.

Files: `market-resnet/RESULTS.md`, `market-resnet/memdepth.py`, `market-resnet/econ2.py`, `market-resnet/results/vm/trader-mresnet-4/p2/memdepth.json`, `market-resnet-AUDIT.md`

## E-MR-13: Does the trader economy show the predicted risks: bubbles, fixed-point stall, TD divergence?

Status: Informative

Setup: All phase-2 runs of E-MR-5 to E-MR-11: wealth shares, bid and VOC calibration, belief log loss vs depth past the training length, training-loss spikes (> 0.1 nats per bit) after iteration 1000.

Result: Runaway confidence, yes: past the training length belief logits grow without check (mean log loss 501 nats per bit at L3200 and 3200 steps with 93% of examples exactly right; 8,429 at L32000), and auction bids are 23 times too high; wealth does not concentrate (largest share <= 0.32). Stall only without stream 3: the b-only economy stays at 0.25-0.31 nats per bit with 0 accuracy and the slots economy near chance; the cov economy keeps improving with depth. No TD divergence: the markets settle on realized outcomes, not later prices, and none of the 19 house- or Lindahl-funded cov runs logged a loss spike after iteration 1000 (bptt_cov_prog: 4 and 7; voluntary bounty: 3 and 4).

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/charts/p2_wealth.json`, `market-resnet/results/summary.json`

## E-MR-2: At equal compute, does priced halting (halt when predicted gain <= cost) beat PonderNet and random halting, with and without a real cost in training?

Status: Gain

Setup: Phase-1 models of E-MR-1. Operating point = cheapest cost within 0.005 of the best accuracy; every other policy read at the same mean steps (PonderNet from its accuracy-vs-steps curve over all thresholds q; random halting with the same step histogram, E_perm, or geometric, E_geom; fixed depth). Cost sweep: Ctt_eps trained and halted at c = 0.01, 0.05, 0.2 (epsilon-continuation, 1/eps settlement), 3 seeds each, against C priced after training at the same c.

Result: Gain out of distribution, none in distribution. mix_8_64: C reaches 1.000 at 17.7 steps where random halting gets 0.54, geometric 0.43, PonderNet 0.75 (prior 1/12) and 0.96 (prior 0.2). id_8_32: PonderNet ties (10.2 vs 9.9 steps). L64 (one length): random with C's histogram is as good (0.99). With a training cost, Ctt_eps on mix_8_64 takes 26.9 / 21.0 / 17.8 steps at acc 1.000 for c = 0.01 / 0.05 / 0.2, against random 0.58 / 0.57 / 0.54 and PonderNet 0.77 / 0.77 / 0.75 (prior 0.2: 0.97 / 0.98 / 0.96) at the same steps; training with the cost adds nothing over pricing a cost-0 model after training (34.7 / 20.4 / 17.9 steps, 1.000). L128 still extrapolates (0.97 to 1.000).

Files: `market-resnet/RESULTS.md`, `market-resnet/mresnet.py`, `market-resnet/evaluate.py`, `market-resnet/run_grid2b.sh`, `market-resnet/results/summary.json`, `market-resnet/charts/p1_halting_mix_8_64.json`, `market-resnet/charts/p1_halting_L64.json`, `market-resnet/charts/p1_cost_sweep_mix_8_64.json`, `market-resnet/charts/p1_cost_sweep_L64.json`

## E-MR-3: When training halts examples, how must the halted steps be settled (the decision-market bug)?

Status: Problem

Setup: Phase 1: Ctt trained with halting at cost 0.05 after 20% of training, settled four ways: counterfactual pay, epsilon-continuation (eps = 0.1, weights 1/eps; Chen et al. 2014), realized pay (0 after a halt, self-fulfilling) and executed steps only; 3 seeds; tested at L64. Phase 2: the trader economy trained with VOC halting at 0.002, settled on realized value (haltreal) or with epsilon-continuation (halteps), 2 seeds.

Result: Settling only executed steps is a Problem: at L64 it halts too early (c = 0.05: 0.970 +- 0.055 accuracy, post-halt gain 0.040 nats; c = 0.2: 0.948, 0.090), because the gain head never sees what happens after a halt. Counterfactual and epsilon settlement fix it (1.000, post-halt gain 0.000). The self-fulfilling realized-pay version showed no harm in either phase (phase 1 1.000; phase 2 haltreal 7.8 steps at 1.000 against halteps 14.1 steps at 0.992 at c = 0.001). Also found and fixed: a double-counted compute cost in econ.py/econ2.py halting (no reported result affected: all earlier runs used cost 0) and cost charged on halted steps in econ2.py (fixed before use).

Files: `market-resnet/RESULTS.md`, `market-resnet/mresnet.py`, `market-resnet/econ2.py`, `market-resnet/charts/p1_post_halt_gain.json`, `market-resnet/results/summary.json`

## E-MR-4: Does Aumann-Shapley credit split each step's pay sensibly across the x, b and h streams?

Status: Problem

Setup: Integrated-gradients (Aumann-Shapley) credit of each stream's writes to later per-step payments, 64 integration points, phase-1 arms A, B and C, 3 seeds; completeness error = |sum of credits - payment| at the worst step.

Result: No. The h stream gets 2-4% of the credit and a step's own b write 2.7 to 3.1 times the total, so other parts carry large negative credit. Completeness fails for the market arms: 0.46 +- 1.25 (C) and 0.75 +- 0.82 (B) nats at the worst step against 0.05 for A, so the path integral is not accurate enough to use these credits.

Files: `market-resnet/RESULTS.md`, `market-resnet/evaluate.py`, `market-resnet/charts/p1_credit_by_lag.json`, `market-resnet/results/summary.json`

## E-MR-5: Without any backprop across depth, do traders paid only by covariance markets extrapolate in depth and length further than the same traders trained end to end?

Status: Informative

Setup: Phase 2 (econ2.py): 4 traders with own weights and Adam, shared across levels j (offsets 0, +-2^j), reading detached state and trained only on their own profit (LMSR earnings minus compute cost); wealth ledger with W0 = 20, trade size capped at half the worst-case loss, respawn at 0. Stream 3 = covariance markets 'is y_i = y_(i+2^j)?' plus VOC markets (k = 5, 50, 500). BPTT control: the same traders, trained end to end on the final loss (bptt_cov, 3 seeds), with Bansal's progressive loss (bptt_cov_prog, 2), on every step's loss (bptt_cov_ds, 3), or on every step's loss plus the covariance targets on c (bptt_cov_dscov, 3; in the other BPTT arms c is a free latent). Controls without stream 3: econ_none, bptt_none. 3000 iterations, batch 64, D = 24, lengths 8-32; test at L = 32, 320, 3200, 32000 in round-robin order. Equal compute: 657 s vs 694 to 772 s training, 0.28 vs 0.29 GB.

Result: Informative (audit 2026-09-29): the extrapolation is real, the economics is not what causes it. Exact-string accuracy (every output bit right) at L32 (24 steps) / L320 (320) / L3200 (3200): economy 1.00 / 1.00 / 0.93 +- 0.14 (3 seeds, 40 strings); bptt_cov_dscov (full BPTT through the 24 training steps, same per-step and covariance losses, trained in fixed round-robin order) 1.00 / 0.98 +- 0.05 / 0.12 +- 0.35 (seeds 0.00, 0.08, 0.28); bptt_cov 0.97 / 0.00 / 0.00; bptt_cov_prog 0.86 / 0.00 / 0.00; bptt_cov_ds 0.99 / 0.00 / 0.00. The market payoff is exactly detached one-step backprop with a loss at every step (truncated BPTT k=1 plus deep supervision): per-trader gradient cosine 1.0000000 against a BPTT replica, with and without the VOC and bid heads (eqv4.py, results/eqv4.txt). The wealth cap never binds (nocap 0.92). Minimal honest claim: on prefix sums, detached per-step-loss training in random module order extrapolates to 100x the training length (0.93 vs 0.12 for full BPTT); the market payoff is a relabelling of that objective. Confounds (random vs fixed training order, VOC targets only for the economy, per-trader optimizers) are tested in RESULTS.md section 17. Both arms become extremely overconfident past the training length (log loss 501 and 5,900 to 11,000 nats per bit at L3200).

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/run_p2.sh`, `market-resnet/run_p2post.sh`, `market-resnet/results/summary.json`, `market-resnet/charts/p2_depth_L32.json`, `market-resnet/charts/p2_depth_L320.json`, `market-resnet/charts/p2_depth_L3200.json`, `market-resnet/charts/p2_depth_L32000.json`, `market-resnet/run_p2cov.sh`, `market-resnet/eqv4.py`, `market-resnet/results/eqv4.txt`, `market-resnet-AUDIT.md`

## E-MR-6: In the trader economy, does halting when the VOC market's price falls below the compute cost beat random and fixed-depth halting at matched compute, and does training with the cost help?

Status: Informative

Setup: E-MR-5 economy; halting test on mixed lengths 4 to 64 (written as L64 before the 2026-09-29 audit), 500 examples, cap 96 steps; halt when VOC <= c for c = 0.001, 0.003, 0.01. Cost-0 run (3 seeds) halted post hoc at c, and runs trained with cost c and halting at c, epsilon-settled (2 seeds each). Random halting with the same step histogram and fixed depth at the same mean. No PonderNet/ACT arm: both need backprop across depth.

Result: Informative (audit 2026-09-29): confidence halting works; no market-specific value shown. Cost-0 run at c = 0.001 / 0.003 / 0.01: 9.6 / 7.6 / 6.6 steps, accuracy 0.994 / 0.980 / 0.859, against random halting 0.836 / 0.743 / 0.634 and fixed depth 0.985 / 0.876 +- 0.169 / 0.753 +- 0.129 (within noise). The test set was mixed lengths 4 to 64, not L64. By construction VOC = sigmoid(v0) x belief entropy / 5, and a plain entropy threshold on the belief, with no market, matches VOC within 0.02 at every cost (audit CPU check). Trained at c: 14.0 / 8.1 / 6.9 steps, 0.994 / 0.975 / 0.859, so training with the cost adds nothing, as in phase 1. VOC prices are calibrated (corr 0.82). Rerun at L64 only against an entropy threshold: RESULTS.md section 17.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/run_p2b_vm6.sh`, `market-resnet/charts/p2_priced_halting.json`, `market-resnet/results/summary.json`, `market-resnet-AUDIT.md`

## E-MR-7: Can the traders' own profit bids choose who acts (an auction for each step) and when to halt?

Status: Problem

Setup: E-MR-5 economy tested in auction order (the highest predicted-gain bid acts) against round-robin order, at L32 and L320; bid halting when the best bid <= c at L64.

Result: No. Bids are 23 times too high (predicted 0.126 vs realized 0.0055, corr 0.20), likely a winner's curse from picking the highest of four noisy bids (not tested). So bid halting almost never fires (85-90 of 96 steps at every cost) and gains nothing over random, and auction order is erratic at depth: L320 at 320 steps 0.99 / 0.02 / 0.66 by seed in the base economy and 0.00 in the halteps, Lindahl-bounty and all six priced runs, while round-robin order holds 0.99 or better in every cov run.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/results/summary.json`, `market-resnet/charts/p2_depth_L320.json`

## E-MR-8: Does a stream of private slots, whose owners pay writers their Aumann-Shapley share and pay rent, carry useful state in the trader economy?

Status: Problem

Setup: E-MR-5 economy with stream 3 = private banks: only the owner reads its bank, any trader may write, the owner pays writers their Aumann-Shapley (path-integrated-gradient) share of its earnings, rent per slot per step; 2 seeds.

Result: No. Training loss ends at 0.641 nats per bit (chance 0.693), exact accuracy 0 at every length and depth, 71 respawns per run, final wealth 1-15 (start 20): rent and payments bankrupt traders faster than they learn.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/results/summary.json`

## E-MR-9: In the trader economy, do the wealth cap and the entry of new traders change what is learned?

Status: Informative

Setup: E-MR-5 economy with the wealth cap (trade worst-case loss <= half of wealth, respawn at 0), without it (nocap), and with a new trader every 500 iterations (grow, 5 entrants); 2 seeds each.

Result: With house-funded markets all traders end rich and equal (2,770-2,950 from 20; largest share 0.25), so the cap never binds and nocap is the same (L3200 0.92 vs 0.93). Later entrants earn less (2,009 for an original trader down to 79 for the last entrant; largest share 0.18) and accuracy is unchanged (L3200 0.99 +- 0.16). No respawns in any cov run except 2 in one Lindahl-bounty seed.

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/charts/p2_wealth.json`, `market-resnet/results/summary.json`

## E-P1-1: Multi-horizon anytime-value heads (T2m mechanism): does halting on heads that predict future belief movement at horizons {1,2,4,8,16,32,64,cap} beat one-step, {1,cap}, myopic+entropy, Q-ACT, probe and simple rules at matched compute when the plateau length varies within a size and half the instances never resolve?

Status: No gain

Setup: Frozen MarketResNet w=64 (recipe v2 curriculum, 8000 it; both backbones pass the validity gate), 2 backbone seeds x 3 head seeds; pools L64 and L128 (500 per backbone seed), L256 descriptive. Cost = mean steps to reach a* = 0.98 x best fixed-T log-score gain. Arms R0/R2x4/R3x3 (envelope), R4 one-step gate, R4c same head {1,cap}, R4e, R5 PonderNet, R11, R12, R13 Q-ACT, R7 multi-H label, R8 multi-H KL. 2000 bootstrap resamples (unit, then examples).

Result: R7 halts at 14.2 [13.0, 15.5] steps at L64 and 29.1 [26.7, 31.6] at L128, within 1% of the oracle (14.1, 29.0) and 33-35% below the simple-rule envelope (21.3, 44.7); R8 (label-free) matches it (+0.1%). But R4c, the same head using only H=1 and cap, is within 1.1% [0.0, 2.1] and 1.6% [0.0, 2.6]: S2 fails (saving vs best competitor 0.3% and 0.9%) and the F2 non-myopia kill fires. R13 Q-ACT 18.8 / 36.1, one-step R4 46.9 / 101, confidence R3 about 46 / 92. The cap (remaining-value) horizon does the work; intermediate horizons add nothing. Resolvable exact at a*: R7 0.79 vs R4 0.997 (clause fails). Head capacity gate fails as specified (R7 Spearman 0.46-0.63 at H in {4,16}), so formally not a clean falsification; post hoc, blocked instances (pure-noise gain) cap the pooled Spearman, resolvable-only 0.67-0.81.

Files: `puzzles/P1/RESULTS.md`, `puzzles/P1/PREREG.md`, `puzzles/P1/analyze.py`, `puzzles/P1/eval.py`, `puzzles/P1/train_heads.py`, `puzzles/P1/capdiag.py`, `puzzles/P1/results/p1_results.json`, `puzzles/P1/results/p1_chart.json`, `puzzles/P1/results/capdiag_t2m.json`

## E-P1-2: Multi-horizon anytime-value heads on mazes: does the label-free test-time-refit head (R9) beat the simple-rule envelope, the refit consistency probe and Q-ACT at matched pixel-steps?

Status: No gain

Setup: Bansal et al. DT-Recall checkpoint (w=128; reproduction gate passes: 13x13 1.00, 59x59 1.00 vs paper 99.88% / 97.30%). Pre-registered headroom gate on dev: 1 - oracle/envelope in steps at a* = 0.98 x best fixed-T accuracy must reach 0.30 at 33 or 59.

Result: No gain (no headroom): Headroom gate failed: 0.059 at 33 (envelope R2:2:1 104.6 steps vs earliest-exact oracle 98.5, N=200) and 0.053 at 59 (280.3 vs 265.4, N=100). DT-Recall solves every dev maze by T_cap and a KL lag-2 threshold is already within 6% of the oracle, so no halting head can win; S1 cannot pass. Maze runs cut to Q3 (59 and 99).

Files: `puzzles/P1/RESULTS.md`, `puzzles/P1/PREREG.md`, `puzzles/P1/eval.py`, `puzzles/P1/results/vm/trader-cat-p1-7/maze/headroom_gate.json`, `puzzles/P1/results/vm/trader-cat-p1-7/maze/repro.json`

## E-P1-3: Multi-horizon anytime-value heads (label-free KL settlement, Q3): after a label-free test-time refit on the unlabelled maze pool, are KL-settled multi-horizon heads (R9) level-calibrated in every bin at maze sizes 59 and 99, and when movement stops tracking information, does sufficiency (realized gain G vs KL K) or the martingale condition (entropy drop vs K) fail?

Status: No gain

Setup: Bansal et al. DT-Recall checkpoint (w=128), heads trained on 9x9 trajectories at horizons {2,4,8,16,32,64,cap}, refit label-free on each pool (2 epochs), 3 head seeds. Eval N=500 at 59 (T_cap 708) and N=200 at 99 (T_cap 1188), decisions every 4 steps. S3: Spearman >= 0.8, every quintile pred/real in [0.8,1.25], <20% of K mass on steps with G<0, within 2pp of the labelled refit R7TT at matched compute (0.5x/1x/2x oracle pixel-steps). 2000 bootstrap resamples. At 99, H=64 and cap overflowed float16 storage; read at H<=32 plus a 65504-bound sensitivity rerun (PREREG Addendum I).

Result: S3 fails at both sizes and F3 fires at both. R9 Spearman 0.79-0.81 at 59 (lower CI 0.77-0.79), 0.87-0.90 at 99; top two quintiles level (0.79-1.15 at 59, 0.80-1.32 at 99) but low quintiles off 3-50x (0.02-0.33) and the middle quintile up to 12x at 59; harmful K share 13.8% (59) and 15.9% (99) at H=2, about 2% at H>=4 (passes). R9 - R7TT exact at 0.5x oracle budget: -0.10 [-0.18,-0.02] at 59, -0.09 [-0.15,-0.01] at 99. Sufficiency holds where mazes are still being solved (G/K 0.89-1.21) but the martingale condition fails everywhere (entropy drop 1.2-4.4% of K); in settled mazes movement is 2-cycle oscillation without information (G << K, 487/500 and 182/200 mazes end in a 2-cycle). The refit consistency probe R12TT beats R9 (+16pp at 59, +6pp at 99 at 0.5x budget), so F4 does not fire only because the probe is better; no VOC arm beats the KL lag-2 envelope (R9 537 vs 267 steps at a* at 59, 1001 vs 679 at 99). Float16 sensitivity at 99 leaves the verdict unchanged.

Files: `puzzles/P1/RESULTS.md`, `puzzles/P1/PREREG.md`, `puzzles/P1/analyze.py`, `puzzles/P1/eval.py`, `puzzles/P1/sens_f16.py`, `puzzles/P1/results/p1_results.json`, `puzzles/P1/results/p1_chart.json`, `puzzles/P1/results/sensitivity_f16_n99.json`

## E-P2-0: Stage-0 toy: can a tree of latent codes hold a long-range parity choice that a factorised belief cannot?

Status: Problem

Setup: 1D broadcast parity (two equally likely answers per string), depth-recurrent 1D trunk, progressive market loss; U (factorised), M-global (K=2), M-tile (8-position leaves), M-tree (binary tree, K=2, <= 2 levels in training, up to 4 at test); 3 seeds, 3000 steps, CPU. Readout: exact match while U has not resolved (U never resolves here).

Result: Exact match in U's unresolved window: n=32: M-tree 51%, M-global 48%, M-tile 27%; n=64: M-tree 51%, M-global 50%, M-tile 6%; n=128: M-tree 51%, M-global 49%, M-tile 2%. Pre-registered pass (M-tree >= 40% and M-tile <= 5% at n=64 and 128): False.

Files: `puzzles/P2/toy.py`, `puzzles/P2/results/toy/summary.json`

## E-P2-1: Are the P2 maze sets, decoders and training recipe valid (unit tests, decoder trap, U-canon gate, floor rule)?

Status: Problem

Setup: Frozen test/dev/ext sets (maze-dataset lattices, percolation p in {0, 0.05, 0.1}), n=9 training pool with no hash overlap; exact DAG and tree dynamic programs checked against enumeration; every decoder on constant beliefs and random nets must stay <= 1% VSP; U-canon s0 gate >= 95% at n=9 and >= 80% at n=19 (p=0, T <= 300); floor rule: U-canon dev no-tie VSP >= 50% at T=300.

Result: Unit tests pass: True; trap test incomplete (spot preemption): 26/30 cells, max VSP 1.0% in those, missing n33_p0.05_de1, n33_p0.05_de0, n33_p0.1_de1, n33_p0.1_de0; counted as not passed; gate {'n9': 1.0, 'n19': 0.97} pass True; floor sizes [13] ({'13': 0.805, '19': 0.439, '25': 0.243, '33': 0.14}); GPU utilisation in the pilot 93.09803921568627%.

Files: `puzzles/P2/tests.py`, `puzzles/P2/gen.py`, `puzzles/P2/results/p2_results.json`

## E-P2-2: On mazes with tied shortest paths, does an exactly priced quadtree latent-parent market (M-tree) find a valid shortest path more often than commitment baselines (canonical labels, hard-EM)?

Status: Problem

Setup: Depth-recurrent conv net (about 107k params, trained on n=9 with the progressive market loss, 5000 steps, 5 paired seeds), tested on n=13-33 at T=300. M-tree: 8x8 leaves, K=4 codes per quadtree node, exact upward-pass joint log score on uniformly sampled tie labels, decoded by exact tree MAP. Baselines: U-canon (canonical move-order labels), U-min (hard-EM Viterbi labels), U (sampled labels), U-any (exact P(some shortest path)); decoders get at most 5 local cleanup sweeps, no global search.

Result: Tie VSP at T=300 (primary set, 300 mazes, n in [13]): U 33.4 [22.0, 44.9], U-canon 50.0 [44.7, 55.3], U-min 64.4 [61.1, 67.7], U-any 62.1 [45.2, 79.0], M-tree 46.1 [29.9, 62.2], M-tile 43.3 [20.3, 66.4]. M-tree - U-canon: -3.9 [-23.8, 15.9] points (bootstrap [-16.4, 9.1]), DiD -13.9 [-23.3, -4.5]. M-tree - U-min: -18.3 [-34.8, -1.9] points (bootstrap [-28.9, -7.7]), DiD -26.2 [-37.5, -14.9]. p=0 cost (U-canon - M-tree): 1.0 [-0.2, 2.2]. GAIN (a) False, (b) False, (c) False. Falsified: 1_upper_M_minus_maxbase_lt_3, 2_DiD_le_0. Per n, tie VSP M-tree - U-min (all 4 sizes, descriptive): n13 -18.3 [-34.8, -1.9], n19 -11.5 [-31.9, 8.9], n25 0.0 [-10.0, 10.0], n33 1.9 [-6.6, 10.3].

Files: `puzzles/P2/jm.py`, `puzzles/P2/train.py`, `puzzles/P2/eval.py`, `puzzles/P2/report.py`, `puzzles/P2/RESULTS.md`, `puzzles/P2/results/p2_results.json`

## E-P2-3: Does the M-tree code carry the route choice, and do independent per-tile mixtures (M-tile) suffice?

Status: Informative

Setup: 256 ancestral samples of (codes, pixels) per primary tie maze; branch = which tied route the sample's best path takes; I(code at the lowest node covering the contested region; branch) / H(branch). M-tile: same model without cross-tile transitions.

Result: M-tree I/H 0.32 [0.22, 0.42] (pooled 0.26 [0.18, 0.34]; pass >= 0.8: False). M-tile I/H 0.73 [0.64, 0.82] on the 23% of tie mazes whose contested region fits one tile. Contested > 32 px: M-tree - M-tile 5.0 [-25.2, 35.2] points (Holm p 1). Share of tie mazes with the contested region inside one tile: 23%.

Files: `puzzles/P2/eval.py`, `puzzles/P2/results/p2_results.json`

## E-P2-4: Does tie VSP keep rising with more iterations and hold on larger mazes, and does the canonical rule extrapolate?

Status: Informative

Setup: Decisive arms (M-tree, U-canon, U-min): T = 300 -> 1000 on 100 tie mazes per cell (n <= 33), T = 3000 on 50 per p at n = 33, n = 59 at T = 300 and 1000; slope of tie VSP against n per seed.

Result: M-tree: n33 T300 0.0, T3000 0.0, n59 T300 0.0, T1000 0.0; U-canon: n33 T300 1.5 [0.6, 2.4], T3000 1.8 [1.0, 2.5], n59 T300 0.1 [-0.3, 0.5], T1000 0.2 [-0.2, 0.7]; U-min: n33 T300 5.0 [-20.4, 30.4], T3000 4.5 [-14.6, 23.6], n59 T300 0.2 [-2.9, 3.4], T1000 0.5 [0.5, 0.5]. Deep-eval seeds: M-tree 1, U-canon 4, U-min 2. Slope per n (VSP per unit n): n/a (needs >= 2 floor sizes).

Files: `puzzles/P2/eval.py`, `puzzles/P2/results/p2_results.json`

## E-P4-0: On exact program tables, does flip-Shapley capture latent worth that reach, LOO and mutual information miss?

Status: Informative

Setup: 30 generated boolean circuits (R1 family plus H1 parity-with-copies and H3), exact tables over all inputs; flip, soft and pay games; theorem checks (copies get LOO 0, an epistatic node gets flip-LOO and flip-Shapley 0, H1 I(p;Y)=0, Shapley efficiency) and within-level rank agreement between reach r, marginal, Shapley, MI-Shapley and pay-Shapley on gates.

Result: All theorem checks pass (efficiency gap 9e-16). On gates, reach agrees with flip-Shapley at within-level Spearman 0.90 (top-1 0.88), with MI-Shapley at only 0.33. H1 level 3: each copy of p has reach 0 and I(p;Y)=0 but flip-Shapley 0.106, so Shapley-type pay is the only rule that sees it.

Files: `puzzles/P4/stage0.py`, `puzzles/P4/gen.py`, `puzzles/P4/results/stage0.json`, `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P4-1: On nets with known circuits, does Aumann-Shapley pay (AS/IG) rank latents like the net's exact Shapley?

Status: Informative

Setup: 41 SIIT-trained nets on 30 circuits (all accepted, IIA>=0.99, strict IIA>=0.997; pilot 5/5); 123 net-levels, circuit-clustered bootstrap (32 clusters). AS-SOFTFLIP and AS-PAY midpoint rule (m=5-50, cost 3m) vs the net's exact flip-Shapley; baselines sampled FLIP-SHAP at matched cost (K=16), FLIP-LOO, GRADxACT, MI-1BIT, VINFO, RANDOM. S1 = rho>=0.85 (lo>=0.80), top-1>=80%, TOST equivalence to matched Shapley (margin 0.05), H1 p first.

Result: Partly. AS-SOFT-m50 rho 0.924 [0.908, 0.938], top-1 95%; AS-PAY 0.923, 94%. But sampled FLIP-SHAP at matched cost gets 0.981 [0.976, 0.986], top-1 100%, and even one order (cost ~11) gets 0.964; AS is 0.058 [0.045, 0.071] below, so the TOST clause fails and S1 fails. AS plateaus by m=5-10 (it converges to the diagonal value, not Shapley). FLIP-LOO 0.929 (top-1 89%), GRADxACT 0.37, MI-1BIT 0.13, RANDOM 0.21. H1: p ranks first under AS, exact Shapley and the joint intervention in 3/3 seeds; FLIP-LOO misses it in 2/3. Completeness at m=50: soft 0.03%, pay 0.6%. Falsifier 1 not triggered.

Files: `puzzles/P4/nets.py`, `puzzles/P4/crit.py`, `puzzles/P4/run.py`, `puzzles/P4/results/rows_r1.json`, `puzzles/P4/charts/p4_r1_rho.json`, `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P4-2: Is AS cheaper than sampled Shapley for the same rank quality at n>=16 latents?

Status: Informative

Setup: S2: AS reaches within 0.02 of its asymptotic rho using <=1/3 of the forward-equivalents sampled FLIP-SHAP needs for the same rho. R2 codes (10 natural nets) and market-resnet groups (16 groups x 8 h channels, bases pilot_A and pilot_Bds, t=16 and t=8, P=512, 448-order Shapley reference, L4 GPU).

Result: Mixed. R2: 10/10 nets pass, AS plateaus at m=5 (15 forward-equivalents), cost ratio 0.04-0.31 (median 0.16). Market-resnet: 1/4 pass; ratios A t16 0.47, Bds t16 0.24, A t8 4.9, Bds t8 0.47; at t=8 AS completeness is still 21% off at m=200. AS vs the Shapley reference at t=16: rho 0.93 (A), 0.91 (Bds).

Files: `puzzles/P4/ftmarket.py`, `puzzles/P4/results/stage3/vm/select`, `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P4-3: With noisy latents, does AS-PAY track usefulness u and AS-SOFTFLIP track reach r?

Status: Informative

Setup: 6 SIIT nets on circuits with calibrated noisy nodes (5 accepted, 3 calibrated); Spearman of each pay rule with u and with r, clustered by circuit.

Result: Half. AS-PAY tracks u more than r: 0.79 vs 0.68, difference +0.11 [0.08, 0.15]. AS-SOFTFLIP does not track r more than u: -0.003 [-0.013, 0.013]. Only 3 calibrated circuits, so the intervals are wide.

Files: `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P4-4: On naturally trained nets (no SIIT), does AS rank aligned codes better than LOO, above a random-init null?

Status: Informative

Setup: R2: 10 natural nets with 24 sign units vs 10 random-init null nets; codes aligned to circuit gates by kappa; four-curve AUC (deletion/insertion x flip/pay) per pay rule.

Result: Yes, AS beats LOO. Alignment is above the null but weak (best code kappa 0.42 vs 0.21, 1.3% of codes aligned, 2.6% of gates recovered). Four-curve AUC: AS-SOFT-m50 0.953 vs FLIP-LOO 0.832, difference +0.12 [0.045, 0.198], so the expected +0.05 holds and falsifier 5 is not triggered. AS agrees with a flip-Shapley reference at 0.983 (LOO 0.83); sampled Shapley K=8 gets 0.989.

Files: `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P4-5: Do markets on the latents AS picks improve market-resnet depth extrapolation over plain fine-tuning and random picks?

Status: No gain

Setup: Bases pilot_A and pilot_Bds (L64 repro 0.866/0.917 vs 0.865/0.916); 16 groups x 8 h channels, AS top-2 picks (A [8,7], Bds [11,7], both equal to Shapley's top-2); arms F0, F-AS-mkt, F-AS-aux, F-LOO-mkt, F-rand-mkt; 5 seeds (F0 L128 SD>3 points), 1000 iterations, L4 spot GPU. Primary = mean of L64 and L128 accuracy; paired over (base, seed), n=10.

Result: No, and the run is invalid. F-AS-mkt minus F0: -0.34 [-0.68, -0.00]; minus F-rand-mkt: -0.24 [-0.53, +0.05]; S3 fails. Falsifier 3 (all markets <= F0+2) and 4 (F-AS-aux >= F-AS-mkt, +0.16 [-0.18, 0.50]) trigger. F-LOO-mkt beats F-AS-mkt by +0.14 [0.02, 0.26], but LOO's picks are tie-breaks among 15-16 zero scores. 7 of 40 runs collapsed a chosen group (3/10 F-AS-mkt). Invalid: F0's seed SD is 11 (A) and 32 (Bds) points on primary, since fine-tuning flips extrapolation between near-0 and near-1 by seed (Bds F0: 0.96, 0.24, 1.00, 0.89, 0.58). GPU 1.64 h, est. $1.16, VMs deleted.

Files: `puzzles/P4/ftmarket.py`, `puzzles/P4/vmplan.sh`, `puzzles/P4/gpu.sh`, `puzzles/P4/results/rows_ft.json`, `puzzles/P4/charts/p4_ft_primary.json`, `GPU-LEDGER.md`, `puzzles/P4/RESULTS.md`, `puzzles/P4/analyze.py`, `puzzles/P4/results/summary.json`

## E-P5-1: Can an online market for private per-cell memory (owners post TD-learned prices, writers paid at write time) learn maze path-parity answered only at T?

Status: Problem

Setup: P5 as specified: perfect DFS 9x9 mazes; y = XOR of Bernoulli tokens on the S->T path, one belief market at T. Four conv traders with private banks, D = 64, Adam, batch 64. Pre-registered gate G0: tuned A1 (DT-Recall) or A2 (same traders, full BPTT) must reach 90% on 9x9 validation, falling back to a curriculum (S-T distance uniform over 1..L_train) and 12k iterations. The dense check requires A1 to reach 95% exact-maze accuracy on MP-dense at iteration 128. Phase 0 tests all pass: the adjoint equivalence of A5 and A6 with A2t holds to 2e-16; the neighbourhood logistic is 0.503; caps were made label-free after a leak was found.

Result: No P5 verdict: gate G0 failed. val9 (seed 0, 1000 mazes, Wilson 95% CI): A1 0.528 (0.50-0.56) and A2 0.484 (0.45-0.52) at 4000 iterations; with the curriculum and 12k iterations, A1 0.517 and A2 0.533. Only d=1 (T's own token) is learned; every path of two or more cells is at chance. All 12 tuning configs are at chance. The dense validity bar also fails: A1 exact-maze 0.000 at 4000 iterations (per-cell 0.62), 0.047 at 12k (diagnostic). Diagnostic outside the spec: an incremental length curriculum at 12k iterations gives A1w 0.603, which is 0.92 at d 4-5 and chance beyond d 10. Task finding: parity over a 2D path supervised only at the end is too hard for the baselines at this scale. The market arms (A5, A6, A7) were not run. GPU: 2.33 VM-h, est. $0.93.

Files: `puzzles/P5/RESULTS.md`, `puzzles/P5/results/summary.json`, `puzzles/P5/results/chart.json`, `puzzles/P5/results/gate_diag.json`, `puzzles/P5/results/tests.json`, `puzzles/P5/p5.py`, `puzzles/P5/mazes.py`, `puzzles/P5/driver.py`, `puzzles/P5/gpu.sh`

## P0-1: Does the P0 harness reproduce DT-Recall and give trustworthy strict-fp32 numbers?

Status: Informative

Setup: Unit tests against the published DTNet code, kill-and-resume from .partial, pretrained M-ref (w=128) on 59x59 and on the n9 shift cell, lattice checker vs exact match, market-resnet pilot reload, bitwise repeat, TF32 vs fp32.

Result: Yes, with one data caveat. DTNet max abs diff 0.0; resume gives identical weights and logs; M-ref 59x59@3000 = 1.0; shift n9 de0 = 0.72 (target 0.72±0.08); checker 0 disagreements in 80 evals; pilot reload max diff 0.002; bitwise repeat identical; TF32 matches fp32 (1.0 both, 115 s vs 181 s). The easy-to-hard and maze-dataset 33x33 generators differ in maze depth (curves up to 0.19 apart at 75-100 iterations) but agree per dead-end round (0.604 vs 0.599), so compare arms only on the same frozen sets.

Files: `puzzles/P0/tests.py`, `puzzles/P0/evalrun.py`, `puzzles/P0/p0eval.py`, `puzzles/P0/check.py`, `puzzles/P0/frozen/manifest.json`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-2: Does a width-64 DT-Recall retrain reach the maze bar (>= 0.90 on 33x33 at 300 iterations)?

Status: Informative

Setup: M1: w=64, bansal progressive loss alpha 0.01, 15k steps, 3 seeds; M1-long: 30k steps, 1 seed; M-ref: published w=128 checkpoint. Strict-fp32 eval on frozen 33x33 (500) and 59x59 (100) sets.

Result: No. M1 scores 0.0/0.0/0.0 on 33x33@300 (best val 9x9 <= 0.042; one seed collapses to loss ln 2). M1-long learns 9x9 after 15k steps (val 0.996, 13x13 0.883) but scores 0.086 on 33x33@300 and 0.06 on 59x59@3000. M-ref scores 1.0 on 33x33@300 and on 59x59@3000. Falsify branch (b): the maze bar becomes M-ref and maze economy experiments should run at w=128 (caveat: M-ref's schedule is longer than ours, so width and training length are not separated).

Files: `puzzles/P0/train.py`, `puzzles/P0/dtnet.py`, `puzzles/P0/vm/plan.py`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-3: Does DT-Recall 1D reach the prefix-sum bar on much longer strings, and does a DT-L baseline?

Status: Informative

Setup: P1: DT-Recall 1D, bansal alpha 1, 8 seeds; P-L: our DT-L reimplementation (w=89, 157,261 params), 2 seeds (cut from 3). Exact accuracy on fL128/fL256/fL512, strict fp32.

Result: P1 passes: 8/8 seeds, fL512@300 0.9998±0.0006, unchanged at 3000 iterations, no overthinking. P-L scores 0.0 on every set: our DT-L reimplementation is broken and must not be used as a baseline until fixed.

Files: `puzzles/P0/dtl.py`, `puzzles/P0/train.py`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-4: Does shortening the credit horizon to k=1 (chunked backprop) hurt DT-Recall's extrapolation?

Status: Informative

Setup: Matched ladder: prefix P1 (full progressive) vs P4 (k=1) 8 seeds each, P4-cm (k=1, 22.4k steps compute-matched) 3 seeds, P5 (k=8) 2 seeds; mazes M-prog (bansal alpha 1) vs M4 (k=1), 3 seed pairs, 15k steps.

Result: Prefix: no. P4 8/8 at 0.9948±0.0068 vs P1 0.9998±0.0006 (Fisher p=1.0, median gap 0 pp, paired per-seed gaps <= 2.2 pp); P4-cm 0.966±0.130 (3/3); F1 refuted on prefix. Mazes: M4 minus M-prog on 33x33@300 = +0.0, -31.6, +0.2 pp with every arm under-trained: inconclusive. Side finding: k=1 models stay correct without converging (P4 fixed points on fL512 per seed 0-500 of 500, rest 2-cycles or wandering, still 1.0 after 30k iterations), while every P1 seed ends at a fixed point.

Files: `puzzles/P0/train.py`, `puzzles/P0/vm/plan.py`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-5: How does the maze bar model hold up under distribution shift?

Status: Informative

Setup: M-ref (w=128) at 300 iterations on a 3x3x2 grid: size 9/19/33, extra-opening probability p 0/0.01/0.05, dead-end start 0/1; 100 mazes per cell (one cell could not be generated).

Result: On its own distribution (p 0, de 1) 1.0/1.0/0.99 at 9/19/33; without the dead-end start 0.72/0.74/0.77. Loops break it: p 0.01 gives 0.63-0.84 at 9x9 but 0.07 at 33x33; p 0.05 gives 0.33 at 9x9 and 0.0-0.04 at 19-33. Exact, checker and Knutson scoring agree in every cell.

Files: `puzzles/P0/build_frozen.py`, `puzzles/P0/frozen/manifest.json`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-6: How much room do these puzzles leave for smarter halting (value-of-computation markets)?

Status: Informative

Setup: Stably-correct oracle vs fixed T, KL between successive predictions, max confidence, a logistic stop probe and a size-aware T, on M-ref 33x33 (and the 100 hardest) and P1 fL512 (8 seeds). Room = accuracy gap at matched mean iterations and iterations needed to come within 1 pp.

Result: Almost none. The best rule is 7.8 pp (maze) and 45 pp (prefix median) behind at exactly matched iterations, but that is a cliff: within 1 pp at 1.013x (maze) and 1.011x (prefix). One fixed KL threshold (3e-4) reaches within 1 pp at 1.013-1.018x on all 10 subjects. Spec falsifier 4 fires: about 1-2% compute headroom.

Files: `puzzles/P0/analyze.py`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P0-7: Did market-resnet's pilot gain come from per-step pay, or from lacking the progressive loss?

Status: Informative

Setup: MR-P: market-resnet backbone (w=128) trained with P1's progressive loss and random starts (T=24, 10k steps, 3 seeds), vs market-resnet's single-seed pilot checkpoints A, B, Bds re-evaluated in the P0 harness. Spec test: T=24 on fL256 and fL512. The final-grid comparison (P-mr) was not run: no GRID_DONE in market-resnet.

Result: Spec test at floor: MR-P and pilots A/B/Bds all 0.0 on fL256 and fL512 at T=24 (falsifier 5 true in letter: True). At T=100 still near floor (fL256 MR-P 0.019 vs B 0.006; fL512 MR-P 0.0 vs B 0.0). Off floor on legacy sets MR-P matches pilot B (L64@100 0.999 vs 1.0; mix@24 0.74 vs 0.75; L128@256 0.552 vs 0.41; L128@100 0.244 vs 0.334) and beats pilot A everywhere: the pilot's B-over-A gain looks like a loss/schedule artefact, not per-step pay (single-seed B). The backbone itself is far behind DT-Recall on extrapolation: fL512@300 0.0007 vs P1 0.9998.

Files: `puzzles/P0/mr_adapter.py`, `puzzles/P0/results/vm/general/eval`, `puzzles/P0/RESULTS.md`, `puzzles/P0/results/p0_results.json`, `puzzles/P0/results/chart_p0.json`

## P3: Does Shapley-type pay make latent traders pick the output-natural factors rather than a rotation, beyond a mechanism-sparsity baseline?

Status: No gain

Setup: T1 toy: hidden z* = (b1, b2) drives 32 outputs in 4 groups (b1, b2, b1 AND b2, b1 OR b2, each XOR a seen side bit, 5% noise); traders see z* as noisy tetrahedron points (variant S); 3 latent traders with K=2 straight-through codes are paid by LOO/backprop, exact Shapley, Banzhaf, per-output AS-IG or joint AS-IG (negative control); baselines are an additive reader and Moebius-L1 interaction penalties. Variant S, 20 seeds per arm (2 processes x 10 replicates), 5k steps; Bn/Bm bit variants 10 seeds. ALIGNED = b1 and b2 each found (NMI >= 0.8) and no live trader tracks b1 XOR b2. Holm-corrected one-sided Fisher tests, Wilson/Newcombe 95% CIs. M0: CPU maze oracle (DEF, BFS_S, BFS_G cellular automata, bounded 3x3 reader) for the credit-horizon prediction.

Result: No. AS-IG aligned in 3/20 seeds (CI 0.05-0.36), the same as the additive reader (3/20), exact Shapley (3/20), Banzhaf (3/20) and LOO (2/20); Holm p = 1.0 for every comparison, so pre-registered falsifier (a) holds. AS-IG mostly collapsed to constant codes (Bayes gap 6.3 nats/example). The oracle's predicted 'natural pair plus duplicate' never appeared (0/40). Diagnostic (not pre-registered): the tabular maker cannot split codes; letting it price unvisited buckets (explore 0.05) lifts Shapley to 10/20 on seeds 0-1, but the pre-stated seeds 2-3 replication fails against the additive reader (6/20 vs 5/20, Holm p = 0.50; vs LOO 0/20, Holm p = 0.020). M0 settles prediction (b) at $0: before the backtrack, BFS earns -21% to -22% of its value at c = 6, and DEF earns in every live chunk. So Stage 4 (GPU) was not run. Cost: $1.14 spot CPU VM (ceiling rate), $0 GPU.

Files: `puzzles/P3/RESULTS.md`, `puzzles/P3/t1.py`, `puzzles/P3/m0.py`, `puzzles/P3/analyze.py`, `puzzles/P3/results/summary.json`, `puzzles/P3/results/charts.json`, `puzzles/P3/results/m0.json`

## P12: Does in-loop flow BP (a stop-gradient arbitrageur that corrects a DT maze solver's edge prices) give more exact shortest paths than DT, DT with feedback and post-hoc decoders on weighted mazes 3-7x wider than training?

Status: No gain

Setup:
- **Task.** Weighted mazes: train on 100k 9x9 mazes; val at n = 25; test at n = 9, 15, 25, 33, 49, 65 and two shift sets (extra open walls, p = 0.1 / 0.5, n = 33).
- **DTxARB.** A DTNet with feedback. After each step, max-sum BP over 3-state edge variables (51-configuration cell factors) turns the proposer's logits u into prices b = u + sg(5 tanh(delta / 5)).
- **Baselines.**
  - DT, DT+fb.
  - Post-hoc on DT: DT+dec (min-cost path on -log sigma(u) + c) and DT+ARB(test) (flow BP at test time). Both tuned per seed on val.
  - ARB-kappa (flow BP with one cost on every edge).
  - CostNet+BB (learned cell costs plus an exact shortest-path solver).
- **Training.** w = 64 (the 90% 9x9 gate failed: 60%), 12k steps, 4 seeds for the main arms.
- **Primary cell.** n in {33, 49, 65}, L > 21.
- **Compute.** Spot L4 VMs, $4.88 estimated at the rate ceiling.

Result:
- **Falsified on both counts.**
  - Primary exact rate: DTxARB 2.5% [-0.5, 5.5] against 3.1% for the per-seed best baseline (DT+dec or DT+ARB(test)). Paired difference -0.6 points [-3.5, 2.3]; hierarchical bootstrap [-2.6, 1.1].
  - DTxARB outputs a valid path at n = 49 in only 1.5% of mazes (the bar was 95%).
- **In distribution the arbitrageur helps DT.** At n = 9, DTxARB is exact in 65.8% against 56.8% for DT and valid in 94% against 66%. But post-hoc decoding of plain DT does more: 73.4% exact, with no training change.
- **The arbitrageur's prices know when they are wrong.** Summed semantic loss detects wrong mazes in the primary cell with AUROC 0.92 (DT+fb: 0.59). But the prices cannot repair a proposer whose logits carry no path information past n = 33.
- **What does extrapolate: learned local costs plus an exact solver.** This is the CostNet+BB reference arm (1 seed, evaluated from its step-9000 checkpoint). It is exact in 27.0% of the primary cell and in 40.5% / 18.0% / 9.0% of mazes at n = 33 / 49 / 65, where DTxARB gets 4.6% / 0.5% / 0.2%. Given the true costs, flow BP is exact in 99% (A0-flow). For the trader question: pay for local, size-invariant signals and aggregate them with an exact clearing rule, rather than asking one recurrent trader to carry the global state.
- **Bottleneck.** The proposer's size extrapolation, not the consistency layer. Ablations (noFB, bp, DT+pen) were not completed: two replacement spot VMs were preempted within 7 minutes.

Files: `puzzles/P12/RESULTS.md`, `puzzles/P12/results/summary.json`, `puzzles/P12/charts/p12_charts.json`, `puzzles/P12/{flow,nets,train,evaluate,analyze}.py`, `GPU-LEDGER.md` (P12 section)

## E-MR-14: What carries the trader economy's 100x extrapolation on prefix sums: the market, random training order, the detached per-step loss, the VOC targets or the per-trader optimizers?

Status: Informative

Setup: Phase 3 (after the 2026-09-29 audit). Gradient identity first: each trader's market-pay gradient equals BPTT truncated to 1 step with a loss at every step plus covariance and VOC targets, in random order (cosine 1.0000000, eqv4.py). Ablation arms (econ2.py): the economy in random and round-robin order, without VOC markets, online (voc = 0); full BPTT (bptt_cov_dscov) in round-robin and random order, with and without v on the VOC targets; BPTT truncated to 1 step in both orders, with and without VOC targets, and with per-trader Adam and gradient clipping (tb1_voc_rep_rand, the economy's exact twin). Exact-string accuracy (every output bit right) at L320 with 320 steps (10x) and L3200 with 3200 steps (100x), 3 seeds, t-intervals, on the phase-2 sets (40 strings per seed at L3200); 'fresh' = 200 new strings per seed at L3200 (seed 777 + L), k/n pooled with a 95% Wilson interval. Every arm: 3000 iterations, batch 64, D = 24, lengths 8 to 32. Audit decision rule: if truncated BPTT in random order reaches the economy's fresh accuracy, retire the economics attribution; else name and test the remaining difference.

Result: Random training order is the main ingredient; the market adds nothing measurable. At 100x: economy 0.93 +- 0.14 (557/600 [0.90, 0.95] fresh); same economy in round-robin order 0.40 +- 0.27 (235/600 [0.35, 0.43] fresh); full BPTT round-robin 0.12 +- 0.35 (56/600 [0.07, 0.12] fresh), random order 0.79 +- 0.47 (449/600 [0.71, 0.78] fresh); truncated BPTT(1) round-robin 0.07 +- 0.32 (71/600 [0.09, 0.15] fresh), random order 0.80 +- 0.11 (482/600 [0.77, 0.83] fresh), with VOC targets 0.79 +- 0.53 (497/600 [0.80, 0.86] fresh); the exact twin with per-trader Adam and clipping 0.89 +- 0.04 (529/600 [0.85, 0.91] fresh), overlapping the economy (per seed 0.87, 0.87, 0.91 against 0.99, 0.92, 0.88). Economy without VOC markets 0.87 +- 0.19 (506/600 [0.81, 0.87] fresh); online economy (the 4.2 GB configuration, first trained here) 0.88 +- 0.20 (502/600 [0.80, 0.86] fresh). Decision: truncated BPTT in random order falls short on the pooled fresh strings; the named difference (per-trader optimizers and clipping) closes it to overlapping intervals, and since the twin's gradients equal the economy's, what is left is run-to-run variation. The economics attribution is retired: detached per-step training in random order extrapolates; per-trader optimizers add a little.

Files: `market-resnet/RESULTS.md`, `market-resnet/eqv4.py`, `market-resnet/results/eqv4.txt`, `market-resnet/econ2.py`, `market-resnet/reeval.py`, `market-resnet/mkjobs3.sh`, `market-resnet/mkjobs4.sh`, `market-resnet/run_p3.sh`, `market-resnet/analyze_p3.py`, `market-resnet/results/p3_summary.json`, `market-resnet/charts/p3_depth_L3200_ablation.json`, `market-resnet-AUDIT.md`

## E-MR-15: Does the economy's 0.93 against 0.12 gap at 100x survive the strongest fair backprop baseline, Bansal et al. 2022 DT-recall with progressive loss?

Status: No gain

Setup: dtrecall.py: Bansal et al. 2022's recurrent block with input recall, width-3 convolutions, progressive loss alpha 1, up to 30 iterations in training; three budgets: published width 400 and schedule (12,000 iterations, dtr_full), width 400 for the economy's training time (700 s, dtr_eqtime), width 96 for the economy's training FLOPs (3,090 iterations, dtr_eqflop). Also progressive loss on our own block (bptt_cov_progcov, alpha 0, 0.5, 1). Same task and tests. Exact-string accuracy (every output bit right) at L320 with 320 steps (10x) and L3200 with 3200 steps (100x), 3 seeds, t-intervals, on the phase-2 sets (40 strings per seed at L3200); 'fresh' = 200 new strings per seed at L3200 (seed 777 + L), k/n pooled with a 95% Wilson interval. Every arm: 3000 iterations, batch 64, D = 24, lengths 8 to 32.

Result: No: DT-recall beats the economy at 100x. dtr_full 0.98 +- 0.04 (589/600 [0.97, 0.99] fresh), dtr_eqtime 0.98 +- 0.04 (574/600 [0.94, 0.97] fresh), dtr_eqflop 1.00 (600/600 [0.99, 1.00] fresh), against the economy 0.93 +- 0.14 (557/600 [0.90, 0.95] fresh). Progressive loss on our own block: alpha 0 0.14 +- 0.32 (104/600 [0.15, 0.21] fresh), 0.5 0.80 +- 0.65 (487/600 [0.78, 0.84] fresh), 1 0.80 +- 0.11 (513/600 [0.82, 0.88] fresh), the level of truncated BPTT in random order (phase 1's 'Bansal bar 0.64' was this kind of arm). The economy's remaining edge is reach, from a hand-built dyadic layout (offsets +-2^j): at L3200 with 320 steps it scores 0.78 where DT-recall, which moves about 5 positions per iteration, scores 0.00 (0.98 at 1600 steps). DT-recall's log loss is small at 100x (0.000 to 0.061 nats per bit) but not immune to blow-ups (dtr_full 2.3 at 10x; seed 1 overflowed on 3 fresh strings).

Files: `market-resnet/RESULTS.md`, `market-resnet/dtrecall.py`, `market-resnet/econ2.py`, `market-resnet/reeval.py`, `market-resnet/mkjobs4.sh`, `market-resnet/results/p3_summary.json`

## E-MR-16: Which regularizer stops the belief-logit runaway past the training length and keeps the 100x accuracy: an epsilon floor, an L2 fee on the belief logits, a logit cap (bounded liquidity) or a trade-size cap?

Status: Informative

Setup: The economy (econ_cov_voc) retrained with: an epsilon floor 0.01 on the forecast (fixed share, p' = 0.99 p + 0.005; floor), an L2 fee 1e-3 on belief logits (l2b), belief logits capped at +-10 (bcap), trades squashed by tanh (dbcap); the same cap +-10 without a market on the economy's BPTT(1) twin (tb1_voc_bcap_rand) and on full BPTT in round-robin order (bptt_cov_dscov_bcap). Log loss = mean over bits of the model's own forecast, nats per bit. Exact-string accuracy (every output bit right) at L320 with 320 steps (10x) and L3200 with 3200 steps (100x), 3 seeds, t-intervals, on the phase-2 sets (40 strings per seed at L3200); 'fresh' = 200 new strings per seed at L3200 (seed 777 + L), k/n pooled with a 95% Wilson interval. Every arm: 3000 iterations, batch 64, D = 24, lengths 8 to 32. Extreme depths: L32 to 24,000 steps, L320 to 32,000 steps.

Result: 10x / 100x accuracy and log loss: unregularized 1.00 / 0.93 +- 0.14, log loss 0.000 / 501 nats per bit; fresh 557/600 [0.90, 0.95], log loss 341. Cap +-10: 1.00 / 1.00, log loss 0.000 / 0.000 nats per bit; fresh 600/600 [0.99, 1.00], log loss 0.000: keeps the accuracy and makes log loss sane. It is not economic: the same cap on the no-market twin gives 1.00 / 0.99 +- 0.04, log loss 0.000 / 0.000 nats per bit; fresh 598/600 [0.99, 1.00], log loss 0.000; on full BPTT in round-robin order it gives only 0.30 +- 1.29 at 100x (152/600 [0.22, 0.29] fresh), so it does not replace random order. Floor: 1.00 +- 0.01 / 0.64 +- 0.29, log loss 0.007 / 0.707 nats per bit; fresh 347/600 [0.54, 0.62], log loss 0.849 (raw logits still run away, raw log loss 4,825). L2 fee: 1.00 +- 0.01 / 0.56 +- 1.15, log loss 0.019 / 0.104 nats per bit; fresh 328/600 [0.51, 0.59], log loss 0.100. Trade-size cap: 0.97 +- 0.08 / 0.42 +- 0.90, log loss 0.162 / 221 nats per bit; fresh 277/600 [0.42, 0.50], log loss 166. The floor, the L2 fee and the cap hold 1.00 at extreme depths (L32 to 24,000 steps, L320 to 32,000 steps); the trade-size cap degrades there (one seed of three falls to 0.00 from 2,400 steps at L32 and from 3,200 at L320; mean 0.67).

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/mkregjobs3.sh`, `market-resnet/reeval.py`, `market-resnet/results/p3_summary.json`, `market-resnet/charts/p3_depth_L3200_reg.json`, `market-resnet/charts/p3_depth_L32_reg.json`

## E-MR-17: Does belief liquidity that grows with depth (b_t = t or sqrt(t), positions carried over, each step settled at its own liquidity) stop the runaway and keep the 100x accuracy?

Status: No gain

Setup: Kelvin's arms: at step t the belief LMSR has liquidity b_t = t (liqt) or sqrt(t) (liqsqrt), b_1 = 1; positions q carry over, so the opening price is softmax(q_(t-1) / b_t) and relaxes toward the middle without trades; the logit is the running average of trades (step size 1/t); settlement is each step's log-score change at its own b. Otherwise the economy (econ_cov_voc). Exact-string accuracy (every output bit right) at L320 with 320 steps (10x) and L3200 with 3200 steps (100x), 3 seeds, t-intervals, on the phase-2 sets (40 strings per seed at L3200); 'fresh' = 200 new strings per seed at L3200 (seed 777 + L), k/n pooled with a 95% Wilson interval. Every arm: 3000 iterations, batch 64, D = 24, lengths 8 to 32. Early steps: accuracy against depth at L32, L320, L3200.

Result: The logits stay bounded as designed, but accuracy falls at long lengths. b_t = t: 0.89 +- 0.19 / 0.34 +- 0.26, log loss 0.608 / 12.6 nats per bit; fresh 214/600 [0.32, 0.40], log loss 12.1; sqrt(t): 1.00 +- 0.01 / 0.71 +- 0.18, log loss 0.001 / 91.9 nats per bit; fresh 454/600 [0.72, 0.79], log loss 64.2; against unregularized 1.00 / 0.93 +- 0.14, log loss 0.000 / 501 nats per bit; fresh 557/600 [0.90, 0.95], log loss 341. Median |logit| of wrong bits 41.5 against 17,112. Early information is not diluted in the training range (L32: 0.58 at 6 steps against 0.53) but is at long lengths (L320: 0.55 against 0.79 at 16 steps; L3200: 0.25 against 0.78 at 320 steps). More steps make up for it at L320 (0.84 at 320, 0.97 at 3,200, 1.00 at 32,000 steps). Neither variant beats the unregularized accuracy or the logit cap's log loss (E-MR-16).

Files: `market-resnet/RESULTS.md`, `market-resnet/econ2.py`, `market-resnet/mkregjobs3.sh`, `market-resnet/reeval.py`, `market-resnet/results/p3_summary.json`, `market-resnet/charts/p3_depth_L3200_reg.json`, `market-resnet/charts/p3_depth_L32_reg.json`

## E-MR-18: Does the belief-logit runaway itself cost accuracy at 100x?

Status: Informative

Setup: reeval.py --clamp 20: the fresh L3200 strings (200 per seed, 3 seeds) rerun with b, c and v clamped to +-20 after every step, at evaluation only, no retraining; for the economy and for full BPTT in round-robin order.

Result: Yes. Economy 557/600 [0.90, 0.95] -> 586/600 [0.96, 0.99], log loss 341 -> 0.095 nats per bit; full BPTT round-robin 56/600 [0.07, 0.12] -> 288/600 [0.44, 0.52], log loss 7,439 -> 0.193 (per seed 0.00, 0.08, 0.20 -> 1.00, 0.08, 0.37). About 5 points for the economy; the ranking does not change. Training with the cap (E-MR-16) removes the problem.

Files: `market-resnet/RESULTS.md`, `market-resnet/reeval.py`, `market-resnet/results/p3_summary.json`

## E-MR-19: At a single length of 64, does VOC-priced halting beat an entropy threshold on the belief and fixed depth at matched mean steps?

Status: Informative

Setup: halt64.py: 500 strings of length 64 per seed, 3 seeds of econ_cov_voc, step cap 96; costs 0.0001 to 0.03; entropy threshold, fixed depth (linear interpolation at exactly the matched mean steps) and random halting with the same histogram, each read at the VOC rule's mean steps. Also section 10's mixed lengths 4 to 64.

Result: VOC halting equals an entropy threshold (0.981 +- 0.067 against 0.995 +- 0.019 at 9.0 steps; 0.871 against 0.866 at 8.4); both beat fixed depth by 0.12 to 0.17 at costs up to 0.003 (fixed 0.856 +- 0.164 at 9.0 steps). On the mixed set they also beat fixed depth at matched steps (0.859 and 0.858 against 0.652 +- 0.047 at cost 0.01); section 10's fixed-depth column rounded the mean steps up, giving it extra compute. Confidence halting helps; the market adds nothing over the belief's entropy.

Files: `market-resnet/RESULTS.md`, `market-resnet/halt64.py`, `market-resnet/results/p3_summary.json`, `market-resnet-AUDIT.md`

## E-MR-20: At a fixed length of 32, how does training-step memory grow with depth for full, truncated and checkpointed BPTT and for the batched and online economy?

Status: Informative

Setup: memdepth4.py: one training step, L = 32, batch 16, one L4 alone, depths 24, 240, 2,400 and 24,000; peak GPU memory and time. Modes: full BPTT, checkpointed BPTT, BPTT(1) with one backward at the end, BPTT(1) streaming (a backward at every step), the batched economy (the 0.93 configuration) and the online economy.

Result: Flat memory belongs to any streaming one-step update, not to markets: BPTT(1) streaming 0.007 GB and the online economy 0.006 GB at every depth (the online economy scores 0.88 +- 0.20 at 100x). Full BPTT, BPTT(1) with one backward and the batched economy grow linearly (6.7, 6.7 and 3.0 GB at 2,400 steps) and run out of memory at 24,000 (about 21-22 GB). Checkpointed BPTT: 1.0 GB at 24,000 steps, 1.5 times full BPTT's time at 2,400 (26.9 s against 17.9 s).

Files: `market-resnet/RESULTS.md`, `market-resnet/memdepth4.py`, `market-resnet/results/p3_summary.json`, `market-resnet-AUDIT.md`
