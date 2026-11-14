# TD, GVFs and martingale posteriors for ShipHunt-hard data efficiency

*2026-10-10. Read-only research note. Sources were read in full text (links below). Anything marked "our hypothesis" is not a claim made by the papers.*

Setup in one line: the forecasts f_t (after t = 0, 2, 4, 6 probes) form a sequence that ends at the true board. Every current arm (k = 0, 1, 2) trains each stage directly on the true board. In Sutton's terms that is Monte Carlo, or TD(1).

## 1. Sutton 1988: Learning to predict by the methods of temporal differences
Link: http://incompleteideas.net/papers/sutton-88-with-erratum.pdf (Machine Learning 3:9-44, https://doi.org/10.1007/BF00115009)

**Key idea.** TD trains each prediction toward the next prediction instead of toward the final outcome. TD(lambda) mixes the two targets, and lambda = 1 is plain supervised learning. On the bounded random walk with 10 training sequences, every lambda < 1 beat lambda = 1. Under repeated presentation lambda = 0 was best; with a single presentation, about 0.3 was best. The paper's explanation is that supervised learning minimizes error on the training set, while batch linear TD(0) converges to the maximum-likelihood estimate of the Markov process. In his words, the next prediction is "uncorrupted by random factors that subsequently influence the final outcome."

**How it maps.** Train stage t on a target that mixes lambda x (true board) with (1 - lambda) x f_{t+1}. Here f_{t+1} is computed on the realized next two probes, with stop-gradient, and stage 6 is anchored to the true board. This follows the rule, because the targets are only model outputs and realized boards. For k = 1/2, apply TD to the per-cell marginals (the scored quantity), or score stage t's log-likelihood on boards sampled from the stage-(t+1) model. Sutton suggests fitting backward (stage 6 first) when data is offline. TD also gives a way to trade compute for data: draw many probe sequences per training board. Each sequence gives a different TD target, while the MC target is always the same board.

**Prediction.** The gain should be largest when training boards are few and at early stages. With infinite data both methods converge to the same answer (Sutton says so explicitly).

**Pitfalls.**
- Bootstrapping bias: errors in f_{t+1} propagate back. A small fixed network, or k = 0, cannot represent the true f_{t+1}, which acts like Sutton's failure case (the "bad state" whose value depends on how it was reached).
- Exact evidence sets rarely recur, so any gain depends on the network generalizing across them.
- The theory covers only linear TD on absorbing Markov chains.
- Lambda = 0 propagates the end-of-sequence signal slowly; with only 3 hops this matters little.

## 2. Sutton et al. 2011: Horde (AAMAS, pp. 761-768)
Link: http://incompleteideas.net/papers/horde-aamas-11.pdf

**Key idea.** Knowledge is stored as many general value functions (GVFs). Each GVF is a question defined by a target policy, a cumulant r, a termination gamma and a terminal value z, and each is learned by its own "demon." All demons learn in parallel from one stream of experience, off-policy, with GQ(lambda), a gradient-TD method with linear function approximation and constant cost per step. The evidence is from a robot: time-to-obstacle and stopping-time predictions, and 8 control demons learned in parallel in "only four times longer" than 1 demon learned on-policy.

**How it maps.** The 25 cells x 4 stages are already GVFs. More demons would be extra heads on the shared trunk:
- the next probe's noisy reading;
- pairwise co-occupancy for candidate parent pairs, which is exactly the joint statistic the k = 1/2 tables need;
- row and column ship counts;
- "cell j's forecast after 2 more probes."

**Off-policy.** Probes are random and do not depend on the board. So the occupancy-given-evidence questions do not depend on the probe policy and need no importance weights (our derivation). Only questions about future evidence under a different probe policy (for example adaptive hunting) need the ratios pi/b, and those ratios add variance.

**Pitfalls.**
- Horde shows that parallel learning is feasible. It does not show that extra predictions improve a main forecast's data efficiency.
- In a small fixed network, extra heads can compete for capacity (our hypothesis).
- Off-policy TD with function approximation can be unstable; that is why Horde uses GQ(lambda), whose stability guarantee is for the linear case.

## 3. Fong, Holmes & Walker 2023: Martingale posterior distributions (JRSS-B 85(5):1357-1391)
Links: https://arxiv.org/abs/2103.15671, https://doi.org/10.1093/jrsssb/qkad005

**Key idea.** Uncertainty comes from data you have not yet seen. You specify one-step predictives, impute future data by sampling from your own predictive and updating ("predictive resampling"), and read uncertainty off the limit. The forecasts must be coherent: a martingale, E[P_i | y_1:i-1] = P_{i-1} (the c.i.d. condition). That condition guarantees the forecasts converge and stay unbiased, E[P_inf | y_1:n] = P_n. Exact Bayes is the special case covered by Doob's theorem.

**How it maps.** The exact ShipHunt posterior marginals are a martingale across stages, as long as probe choice ignores the hidden board. Learned forecasts need not be. The regularizer:
1. From an evidence set E_t, sample the next probes' readings from the model's own stage-t predictive, including the 0.2 flip.
2. Compute f_{t+1} on each sample.
3. Penalize the gap between f_t and the mean of f_{t+1}.

The penalty needs no labels, so it can run on extra simulated evidence sequences. That spends compute rather than boards. The difference from TD: TD uses the next probe as it actually came out on the true board, while the martingale term uses probes the model imagines.

**Pitfalls.**
- Coherence is not accuracy: a forecast that ignores evidence is a perfect martingale. Keep the term as a weighted regularizer (beta) next to the outcome loss.
- The paper's theory is for i.i.d. data, and it makes no data-efficiency claims for neural forecasters.
- c.i.d. predictives depend on data order, and the paper averages over M = 10 permutations. The analog here is averaging over the random cell orders of k = 1/2.

## Suggested arms (each on top of k = 0/1/2)
1. **TD(lambda) across stages:** lambda in {0, 0.3, 1}, where 1 is the current baseline.
2. **TD plus probe resampling:** R probe sequences per training board. Compare MC with TD at the same R, to price the compute-for-data trade.
3. **GVF auxiliary heads:** next-probe reading, parent-pair co-occupancy and row/column counts, trained from outcomes only.
4. **Martingale regularizer:** sweep beta and add label-free simulated evidence. Report each arm's martingale gap as a diagnostic.

Metric for all four: per-cell log-loss against the true board, plotted against the number of training boards, at each stage, with compute per board reported alongside.
