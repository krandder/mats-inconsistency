# What discrete diffusion teaches the "rollouts on an any-order support" perspective

*2026-10-13. Read-only literature research for Kelvin (MATS). Earlier reports this builds on: `worldmodel-coherence-research-2026-10-11.md` (D1–D7) and `worldmodel-abstraction-research-2026-10-11.md`. I avoided repeating them.*

**How this was checked.** I confirmed every paper cited below through web search or by fetching its arXiv/venue page on 2026-10-13. Claims attributed to a paper come from its abstract, or from its HTML full text where I fetched it (marked **[full text]**). Tags:
- **[ours]**: my own derivation or hypothesis. The cited paper does not make this claim.
- **[memory]**: a detail recalled but not re-fetched today.
- **[prior report]**: verified in the 10-11 reports, not re-fetched today.

The session's web-search budget ran out near the end. A few late checks were therefore done by fetching pages directly, and the remaining gaps are tagged.

**Vocabulary mapping used throughout.** In diffusion terms, a rollout is an *unmasking order* over k variables. The four roles map as follows:
- **Clamped evidence**: the variables that are unmasked from the start (the prompt or inpainting context).
- **Scored evidence**: a *potential* or *twist* in Sequential Monte Carlo (SMC) language. In Bayes-net language it is *likelihood weighting*.
- **Sampled latents**: the tokens that get unmasked.
- **Untouched**: the tokens that are still masked when the query is read.

The search policy corresponds to the *planner / unmasking policy*. Repair corresponds to *remasking / corrector* steps.

---

## A. Unique insights

**A1. Under incoherent conditionals, the order policy changes the estimand as well as the variance. Choosing an order is choosing a model.**
- **Claim.** If the denoiser is perfect, every unmasking order gives the same distribution, so order learning helps only when the conditionals are imperfect.
  - Xu et al. 2026, "Scheduling Thoughts" (SAS), Remark 3.4 **[full text]**.
  - Kim et al. 2025 make the same chain-rule argument **[full text]**.
- **Claim.** With real models, the exact likelihood of a sequence depends on the deterministic unmasking rule. An oracle search over orders lowered MDM perplexity on AG News to 36.47, against 52.11 for an autoregressive model (Turok, De Sa & Kuleshov 2026, DUEL).
- **Implication [ours].** The procedure's target is the policy-induced path law:
  - P_policy(x) = Σ_σ v(σ | ·) · q_σ(x),
  - conditioned on the scored evidence.
  - It is not "the model's posterior", because no such object exists once the conditionals are incoherent.
- **Two consequences.**
  1. The policy's log-loss training does model selection or model averaging over the n! factorizations, not just variance reduction.
  2. DUEL's oracle number is a warning. Choosing orders with knowledge of the very values being scored is optimistic selection. Any order-selection rule that looks at the evidence values is the same kind of selection, so it must be validated on held-out outcomes, never on the in-sample weights.

**A2. Clamping evidence asks the model for the hard inverse conditionals. Scoring evidence asks for the easy forward ones. This is a learnability argument for the "scored evidence" role.**
- **Claim.** Kim et al. 2025 ("Train for the worst, plan for the best", ICML Outstanding Paper) study "latents-and-observations" data: latents drawn from a prior, observations as efficiently learnable functions of them.
  - Masked training contains subproblems (predicting masked latents from observations) that are computationally intractable under standard hardness assumptions. Ordered training (latents, then observations) is easy.
  - Adaptive inference recovers performance: on L&O-NAE-SAT, accuracy goes from 78→63% with vanilla decoding to 94→89% with adaptive decoding, as the instance size grows **[full text]**.
- **Implication [ours].** Each choice uses a different conditional:
  - Placing a probe reading *before* the board cells (clamped) uses q(board cell | readings), the inverse problem.
  - Placing it *after* (scored) uses q(reading | board), the local noise model the network learns easily.
  - So in ShipHunt and similar data, scoring should beat clamping when data are limited, at the cost of weight variance.
- **What would falsify it.** Clamping wins at all N. That would mean the inverse conditionals are learnable here and the weight variance dominates.

**A3. In an any-order model the optimal twist is one forward pass. Clamped and scored evidence are the two ends of one twisted-SMC family, and the gap between them is a coherence diagnostic you can compute without labels.**
- **Claim, continuous diffusion.** Conditional sampling needs p(y | x_t), which is intractable. DPS approximates it with a Laplace approximation (Chung et al. 2023), which is biased. The Twisted Diffusion Sampler (TDS; Wu et al. 2023) wraps such heuristics in SMC to make them asymptotically exact, with gains "with as few as two particles".
- **Claim, worst case.** Gupta et al. 2024 prove that diffusion posterior sampling is computationally intractable in the worst case, even when unconditional sampling is fast.
- **Claim, LLMs.** Zhao et al. 2024 have to *learn* twists, because an autoregressive LM cannot evaluate p(future constraint | prefix).
- **Implication [ours].** An any-order model can evaluate ψ_t = q(e | latents sampled so far) directly. That gives three estimators in one family:
  - **Twisted SMC with model-exact twists.** Sample the next latent from q(· | prefix), multiply the weight by ψ_{t+1}/ψ_t, and resample when the effective sample size (ESS) drops.
  - **Clamping.** Sample the next latent from q(· | prefix, e). This is the fully adapted (locally optimal) proposal.
  - **Pure scoring.** The weight is q(e | full prefix) at the end. This is plain likelihood weighting: no twist and no resampling.
- **The coherence check [ours].** Under a coherent q, the intermediate twists form a martingale: E_{x_{t+1} ∼ q(·|prefix)} [ψ_{t+1}] = ψ_t. In words, the mean incremental weight should be 1. A mean incremental weight different from 1 measures Bayes-rule incoherence at that exact step, without labels. It is the stepwise form of the martingale check in D5 of the 10-11 report.

**A4. Weight only by the evidence likelihood, never by the likelihood of the sampled latents. The latter is a temperature tilt that sharpens forecasts.**
- **Claim.** Self-Rewarding SMC for masked diffusion LMs (Luo et al. 2026) weights particles by the product of confidences of every accepted token. It resamples when ESS < N/2 and returns the argmax particle.
  - It improves LLaDA-1.5 from 49.3% to 52.1% average accuracy with N=4 particles **[full text]**.
  - The gains are not monotone in N: on GSM8K, N=2 is best.
- **Implication [ours].**
  - Drawing x ∼ q and weighting by q(x) targets something proportional to q(x)², i.e. q at temperature ½. That helps "pick the best answer" tasks.
  - For forecasting it makes the estimate overconfident, so it is wrong for calibrated probabilities.
  - Kelvin's weights q(e_j | prefix) are proper importance weights for conditioning on e. Self-confidence terms must stay out of them.
- **Related trap.** Zheng et al. 2024 found that float32 categorical sampling in MDMs *lowers the effective temperature*. That silently biases Monte Carlo marginalization in the same direction. Mechanism: the Gumbel-max trick with truncated float32 uniform draws **[memory]**. Sample latents in float64, or by inverse-CDF sampling in float64.

**A5. ESS / weight variance is a valid label-free proxy only between policies that target the same distribution with the same roles. Across role assignments it is trivially gamed.**
- **Claim.** Ou, Pani & Li 2025 find that the proposal is the main lever in discrete-diffusion SMC. They train an amortized proposal to minimize the log-variance of the weights, whose unique optimum is the locally optimal proposal. Suboptimal proposals cause "particle degeneracy" **[full text]**.
- **Claim.** Chanchu, Abdulsamad & Naesseth 2026 name weight degeneracy as the failure of bootstrap SMC for discrete-diffusion steering, and overoptimism as the failure of best-of-n.
- **Implication [ours].**
  - Clamping every piece of evidence gives uniform weights, so ESS = N, whatever the accuracy.
  - ESS therefore rewards "clamp everything" and cannot arbitrate the clamp-vs-score decision in A2.
  - Use ESS / log-variance only *within* a fixed role assignment, for example to pick orders or twists. Decide roles by outcome log loss.

**A6. Generation-oriented order heuristics ("most confident first") target a different objective from marginal estimation. For a marginal, the sampled latents should be the high-entropy, high-influence variables.**
- **Claim, Sudoku.** Kim et al. 2025 report solve rates of 6.88% (random order), 18.51% (top probability) and 89.49% (top-probability *margin*). On a harder split, margin decoding scores 49.88% against 32.57% for an ordered autoregressive model with 7× the parameters **[full text]**.
- **Claim, Sudoku.** SAS (Xu et al. 2026) on a 1B MDM **[full text]**:
  - Random 62.7%, confidence 82.0%, margin 80.0%, human order 76.6%.
  - Learned order 91.8%, and 97.5% after fine-tuning along the learned order.
- **Claim, real MDLMs.** Zhong et al. 2026 find that on tasks needing backward information, such as Sudoku, real MDLMs fill the easier blanks first.
- **Implication [ours].**
  - These numbers measure *one sample being correct*.
  - Kelvin's estimator Rao-Blackwellizes the query and marginalizes untouched variables implicitly. Monte Carlo variance comes only from the latents that are actually sampled.
  - Low-entropy ("confident") variables add almost no information when sampled. They should usually stay untouched or be sampled late.
  - The variables worth sampling are those whose value most changes q(query | ·): the "mode-committing" variables. Sclocchi et al. 2025 show these are decided at a sharp transition, while details change smoothly.
  - A policy imported from the dLLM decoding literature may therefore be close to the *worst* choice for marginal estimation. This needs testing (experiment C3).

**A7. The policy can be trained label-free against the model, or outcome-only against data. SAS shows the outcome-only version and its exact objective.**
- **Claim.** SAS trains a small order policy on a frozen MDM with GRPO. The reward is the frozen model's teacher-forced pathwise log-likelihood of the *true* sequence along the chosen order: R = Σ_t log q(x_{i_t} | revealed so far) **[full text]**.
  - This upper-bounds KL(data ‖ policy-mixture model) (their Theorem 3.2).
  - The bound is tight under joint realizability.
- **Claim.** UPO (Hong et al. 2025, ICLR 2026) treats unmasking as a KL-regularized Markov decision process with max-confidence as the reference policy, and keeps the base model frozen **[full text]**.
  - Reward: the fraction of correct cells.
  - Sudoku with LLaDA-8B: random 0.616, max-confidence 0.705, UPO 0.817.
  - The guarantee that it moves samples closer to the data distribution is proved only for a binary reward.
- **Claim.** Jazbec et al. 2025 (ICML 2026) use a one-layer policy whose only inputs are the dLLM's token confidences. It transfers to new dLLMs and longer sequences.
- **Implication.** SAS's reward is exactly Kelvin's "log loss against real outcomes" applied to a whole-sequence order. It needs only realized outcomes, so it fits the rules.
  - For forecasting [ours], replace the whole-sequence log-likelihood with log q(query_true | rollout prefix).
  - Policy features should be the model's own confidences, entropies and weights: domain-agnostic, so nothing is hand-coded.
  - A KL-to-reference term (UPO) is the cheap safeguard against policy collapse. SAS has no entropy or KL term.

**A8. Repair by pseudo-Gibbs on incoherent conditionals converges to a scan-order-dependent distribution. Remasking repair must target a named joint, or it silently changes the estimand.**
- **Claim.** Heckerman et al. 2000: ordered pseudo-Gibbs on inconsistent conditionals has a unique stationary distribution that depends on the visiting order.
- **Claim.** Sana et al. 2026: across five MLMs, a "rectangle test" finds incompatibility of 0.64–0.80, against 0.000 for GPT-2. Glauber dynamics mixes fast at high temperature but is metastable, with exponentially slow escape from basins, at low temperature.
- **Claim.** Campbell et al. 2022: their predictor-corrector has the right stationary distribution only for the *true* reverse rate. In practice they plug in the learned rate.
- **Claim.** GIDD (von Rütte et al. 2025): self-correction "without explicit training" works for models trained with some uniform noise and gives very little for mask-only models.
- **Claim.** On Sudoku-Extreme, inference-only ReMDM reaches 51.3% against 95.2% for a corrector trained on the model's own outputs (Kornhardt et al. 2026, Self-Aware Markov Models). The first number comes from the paper's table as quoted by a search summary; the abstract confirms only the 95%.
- **Implication [ours].**
  - Implement repair as Metropolis–Hastings: propose with q(x_i | x_−i) and accept against a fixed joint such as q_σ(x) for one reference order σ. This is the construction Goyal et al. 2022 use to expose MLMs' implicit energy. The stationary distribution is then known.
  - Report repair's effect on outcome log loss separately from its effect on the consistency checks. Repair can increase apparent self-consistency by moving into a metastable basin.
  - A mask-only base model will not self-correct well. Add some uniform noise to the base training (GIDD), or train a dedicated corrector head.

**A9. Steering ("faith" exp(U)) must enter as an SMC potential, annealed in late. Changing the logits does not sample the tilted target.**
- **Claim.** Chidambaram et al. 2024 prove that guidance does not sample the intended tilted distribution. It pushes samples toward the boundary of the support, and leaves the support entirely under any score error with large guidance.
- **Claim.** Rojas et al. 2025: in masked diffusion, strong guidance early, while most tokens are masked, harms quality; late guidance helps.
- **Claim.** Ou, Pani & Li 2025 compare weights for two kinds of tilt **[full text]**:
  - For a reward tilt q(x)·exp(r(x)), the SMC weights are exact.
  - For product or classifier-free-guidance targets, they are exact only with a perfect reverse model.
  - They anneal the tilt from 0 to 1 across the trajectory for stability.
- **Claim.** Feynman–Kac Correctors (Skreta et al. 2025) and FK steering (Singhal et al. 2025, which covers text diffusion) are the principled versions.
- **Implication.** Treat exp(U) as one more scored potential, multiplied into the weights, with incremental twists and an annealing schedule. Never add it to logits.
  - Because U tilts the *target*, it must be reported separately from accuracy. Otherwise the policy learns to raise U rather than log loss [ours].

**A10. The masked-diffusion training loss fixes which conditionals are good: roughly uniform over context *sizes*. Concentrating training where the policy queries is a large, cheap win, but it narrows the support.**
- **Claim.** The time variable is redundant. The masked-diffusion loss equals the any-order autoregressive objective: an average over uniformly random orders of exact autoregressive log-losses, i.e. a weighted integral of cross-entropies (Ou et al. 2024; Zheng et al. 2024; Shi et al. 2024; Sahoo et al. 2024).
- **Claim.** Under a single scalar schedule the loss is invariant to the schedule. Per-token (multivariate) schedules break this invariance and let training *learn* an order (Garg et al. 2025).
- **Claim.** MAC (Shih et al. 2022) upweights the conditionals that inference uses, and argues that covering all N! orders wastes capacity.
- **Claim.** PUMA (Kim et al. 2026) aligns training masks with inference masks and speeds pretraining by about 2.5×.
- **Claim.** Cai & Shi 2026 prove "mode blindness": schedules that always keep large contexts visible can get mode weights wrong at exponentially small cost. Positive mass on the full mask characterizes when excess risk controls joint KL.
- **Implication [ours].**
  - The low-visibility conditionals that scored-evidence rollouts use to sample latents (A2) are exactly where mode blindness bites.
  - The base-model schedule must keep the full mask and low-visibility masks.
  - After a policy is learned, fine-tune the base model on the policy's own mask distribution. SAS's 91.8→97.5% is the evidence.
  - Hold out random-order loss as a guard, because this narrows the "support" the policy can later explore.

---

## B. Where the procedure as stated is likely wrong or incomplete

1. **"Merge across orders by MIS or BMA" is under-specified, and the cheap version is biased.**
   - True multiple importance sampling needs each sample's density under *every* proposal order: k forward passes per order per sample.
   - Bayesian model averaging that weights orders by q_σ(e) selects the order that fits *this* evidence best, the DUEL oracle effect (A1). With many orders, that posterior concentrates and becomes overconfident [ours].
   - Uria et al. 2014 found that a *uniform* ensemble over orderings beats any single order on test log-likelihood, and that individual orders disagree.
   - MAC notes that single-order estimates are unbiased for the joint but biased for marginals and conditionals.
   - **Fix:** a fixed or prior-weighted mixture over a small order set. Calibrate any evidence-based order weighting on held-out outcomes.
2. **Parallel unmasking is not sampling from q.**
   - When several latents are sampled in one forward pass, the proposal is a product of marginals (Liu et al. 2024, Discrete Copula Diffusion).
   - The rollout's target then differs from the sequential one. The weights stay valid only for that factorized path law.
   - Sample latents one at a time when correctness matters. Use parallel blocks only as a speed knob whose effect on log loss is measured.
3. **Adaptive (confidence-chosen) orders can make weights intractable.**
   - Ou, Pani & Li note that for MaskGIT-style low-confidence sampling the ratio p/q is "not explicitly tractable". They fell back to ReMDM to keep weights computable **[full text]**.
   - The evidence-only weight q(e | prefix) stays valid for the policy-induced target (A1). But any scheme that also needs the proposal density of the chosen *order* must include the policy's probability of the order.
4. **ESS as the label-free proxy is gameable** by role choice (A5).
5. **Self-consistency repair without a named target** changes the estimand and can lock into metastable basins (A8).
6. **"Faith" via logit modification** does not sample the tilted target (A9).
7. **Missing roles.** The diffusion literature suggests two more:
   - **Soft or partially noised evidence.** Per-variable noise levels, as in Diffusion Forcing (Chen et al. 2024 **[prior report]**) and GIDD's uniform noise, which the model can still revise.
   - **Revisable commitments.** Remask or insert. Jiang, Haghtalab & Chen 2025 prove a strict expressivity gap: diffusion LMs with remasking or revision can do things those without it cannot.
8. **For real-world LLM forecasting, "variables = tokens" breaks.**
   - Kim et al. 2026 ("From Interface to Inference") show that an any-order *interface* does not give any-order *inference*, because of positional uncertainty: the model knows *what* belongs but not *where*.
   - Their fixes are insertion-based and latent-segment MDMs, which search over orders of semantic units.
   - Forecast variables (questions, facts) should be units of that kind, not token positions.

---

## C. Cheap experiments (ShipHunt / Sudoku, within the rules)

All of these use one frozen any-order base model trained on outcomes only, with the full mask in the schedule. Scoring uses log loss on realized outcomes. The exact ShipHunt posterior is used for reporting only.

**C1. Clamp vs score vs twisted-score (tests A2, A3; highest priority).**
- **Setup.** On ShipHunt at N ∈ {300, 3k}, estimate each held-out cell's marginal in three ways:
  - (a) clamp all probes and read q(cell | probes);
  - (b) sample board cells in random order, then weight by Π_j q(probe_j | board prefix);
  - (c) as (b), but with model-exact twists ψ_t = q(probes | prefix) and resampling at ESS < K/2.
- **Budget.** K = 64 particles.
- **Log.** Outcome log loss, ESS, and the stepwise martingale ratio E[ψ_{t+1}]/ψ_t.
- **Prediction.** (b) and (c) beat (a) at small N. (c) beats (b) as the number of probes grows. The disagreement between (a) and (c) predicts per-query error.
- **Cost.** Inference only, minutes.

**C2. Does ESS pick the right roles? (tests A5).**
- For 20 random role assignments (which probes are clamped and which are scored), record ESS and outcome log loss.
- **Prediction.** ESS correlates with log loss *within* an assignment, across orders. Across assignments it correlates *negatively*, because clamping more raises ESS.

**C3. Which latents to sample for a marginal (tests A6).**
- On Sudoku (Kim et al. data or SAS data), compare four orders by:
  - (i) the single-rollout solve rate;
  - (ii) the log loss of the K = 16 Rao-Blackwellized per-cell marginals against the true solution.
- The four orders: confidence-first, margin-first, entropy-first (most uncertain first), and random.
- **Prediction.** Margin/confidence wins (i). Entropy-first matches or beats it on (ii) at equal compute.
- Then train a tiny GRPO policy on (ii)'s outcome log loss (SAS-style reward, adapted to the query) and check that it transfers to harder puzzles.

**C4. Repair with and without a named target (tests A8).**
- On ShipHunt, run three repairs for the same number of steps:
  - (a) random-scan pseudo-Gibbs;
  - (b) fixed-scan pseudo-Gibbs in two scan orders;
  - (c) Metropolis–Hastings against q_σ(x) for one fixed σ.
- **Measure.**
  - The scan-order dependence of the resulting cell marginals, as the difference between the two scans in (b).
  - Outcome log loss.
  - Sana's rectangle incompatibility on the base model.
- **Prediction.** (b) shows order dependence that grows as N shrinks. Gibbs repair improves the self-consistency metrics more than it improves log loss.

**C5. Weighting by the latents' own likelihood (tests A4).**
- On ShipHunt, compare evidence-only weights with evidence × Π q(sampled latent | prefix) (SR-SMC-style).
- **Measure.** Log loss and a calibration curve.
- **Prediction.** Sharper and worse calibrated, with lower log loss only where the model is underconfident.

---

## D. Open problems (no answer found)

1. **No theory of which distribution an adaptive policy targets** when the conditionals are incoherent. There is also no account of whether outcome-trained policy selection converges to the best mixture of orders or overfits order choice. SAS's bound is for whole-sequence teacher forcing, not for conditional queries.
2. **Model-exact twists vs learned twists.** In any-order models the twist is free (A3) but inherits the model's incoherence. I found no paper that compares these with learned twists (Zhao et al. 2024) when the model is wrong.
3. **Cheap multiple importance sampling across orders.** Densities under other orders cost k passes. No found work gives a cheap unbiased estimator or a principled small "order basis".
4. **Whether fine-tuning the base model on policy-induced masks** (PUMA, SAS stage 2, Mohamud et al. 2026 co-training) improves coherence or only accuracy along the policy's orders, and what it does to orders the policy later needs.
5. **When remasking converges to something better than the model's joint.** Trained correctors (Self-Aware Markov Models, PRISM) beat ReMDM by a wide margin on hard Sudoku. Theory says remasking adds expressivity (Jiang et al.). Nothing found ties a corrector's fixed point to a coherent joint.
6. **Mode-committing variables that are not addressable tokens.** Coarse-to-fine theory (Sclocchi et al.) says commit high-level features first. ShipHunt's ship placement and a forecast's "world state" are spread over many tokens. Latent MDMs (Kim et al. 2026) are the only direction found.
7. **Calibration under steering.** SMC steering papers report reward, diversity and perplexity, not the calibration of marginal estimates. Nobody measures whether exp(U) tilts preserve calibration on the untilted variables.

---

## Sources

**Masked diffusion = any-order AR, training loss, samplers**
- Ou et al. 2024, *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data* (RADD), ICLR 2025. https://arxiv.org/abs/2406.03736
- Sahoo et al. 2024, MDLM, NeurIPS 2024. https://arxiv.org/abs/2406.07524
- Shi et al. 2024, MD4, NeurIPS 2024. https://arxiv.org/abs/2406.04329
- Lou, Meng & Ermon 2024, SEDD, ICML 2024. https://arxiv.org/abs/2310.16834
- Zheng et al. 2024, *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling*, ICLR 2025. https://arxiv.org/abs/2409.02908 (the Gumbel mechanism is [memory])
- Garg, Kohli & Sarawagi 2025, *Masked Diffusion Models are Secretly Learned-Order Autoregressive Models*, EurIPS 2025 workshop. https://arxiv.org/abs/2511.19152
- Turok, De Sa & Kuleshov 2026, DUEL, preprint. https://arxiv.org/abs/2603.01367
- Uria, Murray & Larochelle 2014, *A Deep and Tractable Density Estimator*, ICML. https://arxiv.org/abs/1310.1757
- Shih, Sadigh & Ermon 2022, MAC, NeurIPS. https://arxiv.org/abs/2205.13554
- Cai & Shi 2026, *An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules*, preprint. https://arxiv.org/abs/2608.01383
- Kim, Geuter, Alvarez-Melis, Kakade & Chen 2026, PUMA, preprint. https://arxiv.org/abs/2602.10314
- Liu, Broadrick, Niepert & Van den Broeck 2024, *Discrete Copula Diffusion*, ICLR 2025. https://arxiv.org/abs/2410.01949
- Chen et al. 2025, *Masked Diffusion Models as Energy Minimization*, NeurIPS 2025. https://arxiv.org/abs/2509.13866
- Cai & Li 2026, *Confidence-Based Decoding is Provably Efficient for Diffusion Language Models*, preprint. https://arxiv.org/abs/2603.22248

**Unmasking order as policy**
- Kim, Shah, Kontonis, Kakade & Chen 2025, *Train for the Worst, Plan for the Best*, ICML 2025. https://arxiv.org/abs/2502.06768 [full text]
- Peng et al. 2025, *Path Planning for Masked Diffusion Model Sampling* (P2). https://arxiv.org/abs/2502.03540
- Liu et al. 2024, DDPD (planner + denoiser), ICLR 2025. https://arxiv.org/abs/2410.06264
- Hong, An, Kim & Ye 2025, UPO, ICLR 2026. https://arxiv.org/abs/2510.05725 [full text]
- Jazbec et al. 2025, *Learning Unmasking Policies for Diffusion Language Models*, ICML 2026. https://arxiv.org/abs/2512.09106
- Mohamud, Hasan, Ravanelli & Bengio 2026, *Adaptive Order Policies for Masked Diffusion*, preprint. https://arxiv.org/abs/2606.00295
- Xu et al. 2026, *Scheduling Thoughts* (SAS), preprint. https://arxiv.org/abs/2606.23567 [full text]
- Lee et al. 2025, LookUM, preprint. https://arxiv.org/abs/2511.05563
- Zhong et al. 2026, *Parallelism and Generation Order in MDLMs*, preprint. https://arxiv.org/abs/2601.15593
- Kim et al. 2026, *From Interface to Inference*, preprint. https://arxiv.org/abs/2607.26504

**Remasking, correctors, Gibbs**
- Wang, Schiff, Sahoo & Kuleshov 2025, ReMDM, NeurIPS 2025. https://arxiv.org/abs/2503.00307
- Campbell et al. 2022, *A Continuous Time Framework for Discrete Denoising Models*, NeurIPS. https://arxiv.org/abs/2205.14987
- Lezama et al. 2023, *Discrete Predictor-Corrector Diffusion Models for Image Synthesis*, ICLR. https://openreview.net/forum?id=VM8batVBWvg
- Zhao et al. 2024, *Informed Correctors for Discrete Diffusion Models*. https://arxiv.org/abs/2407.21243
- von Rütte et al. 2025, GIDD, ICML 2025. https://arxiv.org/abs/2503.04482
- Kim et al. 2025, PRISM (*Fine-Tuning Masked Diffusion for Provable Self-Correction*), preprint. https://arxiv.org/abs/2510.01384
- Kornhardt et al. 2026, *Self-Aware Markov Models for Discrete Reasoning*, preprint. https://arxiv.org/abs/2603.16661
- Jiang, Haghtalab & Chen 2025, *Diffusion Language Models are Provably Optimal Parallel Samplers*, preprint. https://arxiv.org/abs/2512.25014
- Heckerman et al. 2000, *Dependency Networks*, JMLR 1. https://www.jmlr.org/papers/volume1/heckerman00a/html/node6.html
- Sana et al. 2026, *Mixing Times of Glauber Dynamics on Masked Language Models*, preprint. https://arxiv.org/abs/2605.16378
- Goyal, Dyer & Berg-Kirkpatrick 2022, MH on MLM energies, ICLR. https://arxiv.org/abs/2106.02736

**SMC, twisting, steering, guidance**
- Wu, Trippe, Naesseth, Blei & Cunningham 2023, TDS, NeurIPS. https://arxiv.org/abs/2306.17775
- Trippe et al. 2022, SMCDiff, ICLR 2023. https://arxiv.org/abs/2206.04119
- Dou & Song 2024, FPS, ICLR. https://openreview.net/forum?id=tplXNcHZs1
- Skreta et al. 2025, Feynman–Kac Correctors, ICML 2025. https://arxiv.org/abs/2503.02819
- Singhal et al. 2025, FK steering. https://arxiv.org/abs/2501.06848
- Zhao, Brekelmans, Makhzani & Grosse 2024, twisted SMC for LMs, ICML 2024. https://arxiv.org/abs/2404.17546
- Lew et al. 2023, SMC steering of LLMs. https://arxiv.org/abs/2306.03081
- Ou, Pani & Li 2025, *Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design*. https://arxiv.org/abs/2505.22524 [full text] (a notes site lists it as ICLR 2026 [unverified])
- Luo et al. 2026, Self-Rewarding SMC for MDLMs, preprint. https://arxiv.org/abs/2602.01849 [full text]
- Chanchu, Abdulsamad & Naesseth 2026, nested SMC for discrete diffusion, preprint. https://arxiv.org/abs/2608.20123
- Chung et al. 2023, DPS, ICLR. https://arxiv.org/abs/2209.14687
- Gupta et al. 2024, *Diffusion Posterior Sampling is Computationally Intractable*, ICML. https://arxiv.org/abs/2402.12727
- Chidambaram et al. 2024, *What does guidance do?*, NeurIPS. https://arxiv.org/abs/2409.13074
- Rojas et al. 2025, CFG in masked diffusion, preprint. https://arxiv.org/abs/2507.08965
- Nisonoff et al. 2024, *Unlocking Guidance for Discrete State-Space Diffusion and Flow Models* (venue ICLR 2025 per citing papers [unverified]). https://arxiv.org/abs/2406.01572

**Hierarchy, reasoning and Sudoku**
- Sclocchi, Favero & Wyart 2025, PNAS 122(1):e2408799121. https://arxiv.org/abs/2402.16991
- Nie et al. 2025, LLaDA, NeurIPS 2025. https://arxiv.org/abs/2502.09992
  - Its "low-confidence remasking" only chooses which *new* predictions to keep, i.e. an order policy. Unlike ReMDM, it never revises committed tokens.
- Ye et al. 2025, Dream 7B. https://arxiv.org/abs/2508.15487
- Zhao, Gupta, Zheng & Grover 2025, d1 / diffu-GRPO (RL on the denoiser; 4×4 Sudoku 11.7→22.1% per a summary). https://arxiv.org/abs/2504.12216
- Ye et al. 2024, *Beyond Autoregression* (MGDM; Sudoku 100% vs AR 20.7%), ICLR 2025. https://arxiv.org/abs/2410.14157
- Ye et al. 2024, *Diffusion of Thought*, NeurIPS 2024. https://arxiv.org/abs/2402.07754
- Du et al. 2024, IRED, ICML (harder Sudoku 62.1% vs RRN 28.6%; more inference steps help). https://arxiv.org/abs/2406.11179
- Liu et al. 2026, DiBS (a diffusion model orders branches for a complete symbolic Sudoku solver), preprint. https://arxiv.org/abs/2606.06518
- Chen et al. 2024, Diffusion Forcing. https://arxiv.org/abs/2407.01392 [prior report]
