# ARC work on mechanistically estimating random/untrained networks vs sampling

Search date: 2026-10-26. I fetched and read every page quoted below (alignment.org blog pages 1-4, the arXiv abstracts, and the full 68-page PDF of arXiv 2605.05179v2). I also searched arXiv by author for Hilton, Neyman, Lecomte, W. Wu, Christiano, Winer and Xu. Every quote is copied from the fetched text; LaTeX is simplified to plain text.

## Bottom line

**Yes, it exists.** ARC's May 2026 paper (Wu, Lecomte, Winer, Robinson, Hilton, Christiano) estimates the expected output of a *specific* randomly initialised ReLU MLP over Gaussian inputs. It never runs a forward pass. It beats Monte Carlo in MSE versus FLOPs, both in theory (a factor-n speed-up at fixed depth) and in experiments (width 256, at least 2-4 hidden layers). The win **fails at depth**: with 8 hidden layers at width 256 it falls behind sampling once K=4, and the error grows like L^K with depth. Matching sampling at large depth or small width is stated as **open**. The other random instances (halfspaces, #3-SAT, permanents) are not neural networks. Random RNNs, CNNs and transformers are only open questions, with no results.

## Core items (random networks, compared with sampling)

### 1. Estimating the expected output of wide random MLPs more efficiently than sampling (paper)
- **Authors / date / URL:** Wilson Wu, Victor Lecomte, Michael Winer, George Robinson, Jacob Hilton, Paul Christiano. arXiv:2605.05179, v1 6 May 2026, v2 14 May 2026. https://arxiv.org/abs/2605.05179. Code: `mlp_cumulant_propagation` (linked from the blog post).
- **Networks:** fully connected MLPs R^n -> R^n with no biases. ReLU by default; GELU and tanh in Appendix F. Weights are i.i.d. N(0, c/n) (He init for ReLU: N(0, 2/n)). In the experiments width n runs from 4 to 256 and hidden layers L from 2 to 12.
  - "We vary the width n from 4 to 256 and the number of hidden layers L from 2 to 12"
- **Quantity estimated:** E_{x~N(0,I)}[M_theta(x)] for one particular network, which is a quenched average.
  - "our algorithms take as input the weights of a particular network and estimate the expected output over random inputs, i.e., a quenched average ... This is not to be confused with the annealed average over the choice of weights"
- **Estimator:** cumulant propagation, with four variants: basic, augmented, factorized and factorized augmented. It propagates cumulants up to order K layer by layer through Hermite expansions. It uses "power cumulants" and other new details.
  - "we track cumulants of order at most K"
  - "In cumulant propagation, we form an approximate representation of each activation distribution using its low-order cumulants, which can be thought of as a series expansion about a Gaussian distribution."
- **Comparison with sampling:** variance-normalised MSE against counted FLOPs, averaged over output neurons and 5 seeds. Ground truth uses 2^34 Monte Carlo samples.
  - "This choice of units means that Monte Carlo sampling with N samples has a variance-normalized MSE of exactly 1/N."
- **Main results:**
  - **Theory.** "for networks with a fixed, constant depth, our best-performing algorithms achieve a mean squared error of O(ε²) in time O(n/ε²). By comparison, Monte Carlo sampling achieves a mean squared error of Θ(ε²) in time Θ(n²/ε²), a factor of n slower."
    - Footnote: "Our theoretical results assume polynomial activation functions and conditions on the growth rate of ε(n)."
  - **Experiments.** "for random ReLU networks with 4 hidden layers and width 256. We are able to outperform Monte Carlo sampling at computational budgets spanning 7 orders of magnitude, in some cases by a factor of over 100."
  - **By depth.** "The factorized and factorized augmented versions of our algorithm outperform Monte Carlo sampling by a factor of around 1,000 for 2 hidden layers, and around 10–100 for 4 hidden layers. They start to underperform sampling for 8 hidden layers once the maximum cumulant order K reaches 4".
  - **Depth scaling (Appendix D).** Conjectured MSE is at most c_K (L/n)^K.
    - "This depth scaling is worse than Monte Carlo sampling, whose error does not increase with depth."
  - **Small width.** "Our algorithms perform poorly at low width, especially when the number of hidden layers L is large, and can even get worse with the maximum cumulant order K in this setting." (Appendix C)
  - **Low probabilities (width 256, 4 layers, final threshold 1{z>3}).** "using a similar number of FLOPs to Monte Carlo sampling with N = 5,695 samples, our algorithm achieves a relative error of under 30% for probabilities 100 times lower than 1/N".
  - **Wall-clock.** The blog footnote says: "Our algorithms often underperform Monte Carlo sampling in wall-clock time, since we make no serious attempt to optimize their performance on hardware."
  - **Open problems.** "Even for random MLPs, we do not know whether it is possible to match the performance of random sampling at large depth ... or for narrow networks".
  - **Mechanistic distillation.** A random width-16 teacher distilled into a width-8 student: "Mechanistic distillation works well, but is slightly less efficient than distillation using stochastic gradient descent."

### 2. Mechanistic estimation for wide random MLPs (blog post for item 1)
- **Author / date / URL:** posted by Jacob Hilton on 2026-05-07, covering joint work with Wu, Robinson, Winer, Lecomte and Christiano. https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/ (also cross-posted to LessWrong and the Alignment Forum).
- **Framing:**
  - "we see this result as (part of) the "base case": handling networks at initialization."
  - "Our paper achieves our goal of producing mechanistic estimates that outperform random sampling in the case of wide, randomly initialized MLPs."
- **Why it may not carry over to trained networks:** "our methods rely on the networks being randomly initialized ... In a trained network, specific higher-order deviations can become much more important".
- **Novelty:** "the high-level algorithm in our paper is not new: it is a form of cumulant propagation, an algorithm we introduced in 2022 in Formalizing the presumption of indepdence (Appendix D)" [sic], although "a number of specific details about our algorithm are new, and these are essential for achieving low mean squared error."

### 3. Competing with sampling (blog post; introduces the Matching Sampling Principle)
- **Authors / date / URL:** Eric Neyman, posting joint work with Victor Lecomte, Wilson Wu, Michael Winer, Jacob Hilton and George Robinson; 2025-11-18. The paper cites it as "Neyman, Lecomte, Wu, Winer, Hilton, Robinson 2025". https://www.alignment.org/blog/competing-with-sampling/
- **What it adds:** the MSP. It is a mechanistic estimator whose squared error, averaged over a random "context" c (for example random weights), is at most ε² E_c[Var_x], running in time O(Time(M)/ε²).
- **Random MLP instance:** the weights are the random context c. "The expected squared error of G (over independent, normally distributed weights) is less than the expected squared error of the naive sampling-based algorithm".
- **Status in Nov 2025:** "We believe we have a mechanistic algorithm that competes with sampling for estimating the expected output of random MLPs on Gaussian inputs. (We have an empirical demonstration of competitiveness with sampling, and a proof sketch that we are working on expanding into a full proof.)" Item 1 superseded this.
- **Random RNNs (open, no result):** "We would be really interested in finding a way to mechanistically estimate the average output of random recurrent neural networks (RNNs). We believe that this will be much more difficult than the MLP setting, because of the weight sharing."
- **Other architectures:** "Can we solve the problem in the case of narrow MLPs ...? What about random CNNs? Random transformers?" These are open questions only.
- **Non-network random instance:** intersection of random halfspaces, solved with a degree-d Hermite polynomial approximation where n^d ≤ 1/ε². "We have a mechanistic algorithm that competes with sampling (in theory and in practice) for estimating the size of an intersection of randomly chosen halfspaces."
- **Trained (not random) case, partial progress only:** a 2-layer MLP whose second layer is trained, using "ש-tensors" as advice. "while we haven't fully solved it, we have made substantial progress."

### 4. Mechanistic estimation for expectations of random products (blog post with internal notes)
- **Author / date / URL:** posted by Jacob Hilton, 2026-05-15. https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/. The notes are by Hilton, Wilson Wu and Mike Winer.
- **Instances:** random halfspace intersections, random #3-SAT and random {0,1}-permanents. These are random but not neural networks. One link to networks: the halfspace instance "can also be thought of as the probability that all k output neurons of a 1-layer ReLU MLP are active."
- **Estimator:** deduction-projection with "mechanistic L² sketching", which projects onto the top eigenfunctions of E_f[f fᵀ].
- **Result:** "In each case, we roughly match or outperform random sampling up to logarithmic factors, except for the third case, where we are further behind." Here the third case is the {0,1}-permanent.
- **Framing:** these random instances are "similar to randomly-initialized networks rather than trained networks".

### 5. Announcing the ARC White-Box Estimation Challenge (blog post, with an update on LessWrong)
- **Author / date / URL:** Jacob Hilton, 2026-06-02. https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/. The LessWrong version carries the Phase 1 update: https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c
- **Setup:** the same as the paper. Warm-up used width 256 with L=8. "Algorithms will be evaluated on MLPs with randomly-sampled Gaussian weights. The goal is to achieve as low mean squared error as possible, subject to certain computational constraints." Sampling is allowed.
- **ARC's own admission:** "we produced white-box methods that outperform black-box methods for MLPs with large width, but they break down as the depth grows".
- **Phase 1 update (LessWrong):**
  - "For Phase 1, we have increased the depth of the network from 8 to 32 hidden layers. Our existing algorithms scale poorly with depth".
  - "The best-performing algorithms in the warm-up round appear to be variants on the factorized 3rd cumulant propagation algorithm we introduced in our paper, combined with learned networks that consume the cumulant estimates as features."
- **Phase 2 (AIcrowd forum, 2026-08-23):** "1,024 wide, 16 deep", with a FLOP budget of 2^41. Submissions closed on 17 Oct 2026. A participant's forum post says winners are announced on 15 Nov 2026.
- **Not verified:** I found no official ARC results post. Participant write-ups on the AIcrowd forum, which are third-party and which I did not verify, report that sampling and hybrid methods did well at depth 32.

### 6. A Mike's-Eye View of ARC's Research (blog post)
- **Author / date / URL:** Michael Winer, 2026-06-09. https://www.alignment.org/blog/a-mikes-eye-view-of-arcs-research/
- **Relevance:** it is an agenda overview. It calls the random-MLP estimator "one of the crown jewels". "The (average case) MSP states that for any architecture and degree of precision, there is a mechanistic estimator that at least matches the performance of sampling over random instances of that architecture." It reports no new random-network results.

## Background items (checked; do not compare random networks against sampling)

| Item | Authors, date, URL | Random networks? | Comparison with sampling? |
|---|---|---|---|
| Formalizing the presumption of independence | Christiano, Neyman, Xu; arXiv 2211.06738, 12 Nov 2022 | No. Introduces cumulant propagation (App. D) for arithmetic circuits; the 2026 paper says it "introduced cumulant propagation in the context of arithmetic circuits" | No |
| Towards a Law of Iterated Expectations for Heuristic Estimators | Christiano, Hilton, Lincoln, Neyman, Xu; arXiv 2410.01290, 2 Oct 2024 (blog by Neyman, 2024-10-07) | No. Its test problems are "estimating the expected product of jointly normal random variables" and "the permanent of a matrix" | No, accuracy properties only |
| Estimating the Probabilities of Rare Outputs in Language Models (low probability estimation) | Gabriel Wu, Jacob Hilton; arXiv 2410.13211, Oct 2024 (blog by G. Wu, 2024-10-18) | No. "small transformer language models" (1-, 2- and 4-layer) | Yes, with a budget of about 2^16 forward passes: "importance sampling outperforms activation extrapolation, but both outperform naive sampling". These are not random networks |
| Estimating Tail Risk in Neural Networks | Mark Xu, 2024-09-13, alignment.org | No, conceptual | "in experiments so far we have not surpassed baselines based on importance sampling" |
| Backdoor defense, learnability and obfuscation | Christiano, Hilton, Lecomte, Xu; arXiv 2409.03077, Sep 2024 | Not random networks; a formal backdoor game | No |
| A computational no-coincidence principle | Eric Neyman, 2025-02-14, alignment.org | Random *reversible circuits*, not neural networks | No |
| AlgZoo: uninterpreted models with fewer than 1,500 parameters | Jacob Hilton, 2026-01-26, alignment.org | No, these are trained tiny models | Uses MSE versus compute. On small trained models a mechanistic estimate beat or matched sampling. For the larger ones (e.g. 432 parameters), "we would consider it a major research breakthrough if we were able to produce a mechanistic estimate that matched the performance of random sampling" |
| A bird's-eye view of ARC's research (Hilton, 2024-10-23); Obstacles in ARC's research agenda (Hilton post pointing to David Matolcsi's sequence, 2025-05-03) | alignment.org | No | No |

**Adjacent, not by ARC:** Dunbar & Aaronson, "Wide Neural Networks as a Baseline for the Computational No-Coincidence Conjecture", arXiv 2510.06527 (Oct 2025). It is about wide random networks, with no estimator-versus-sampling comparison. It proposes zero-mean-activation random networks "as a promising candidate for the Alignment Research Center's computational no-coincidence conjecture".

## What does not exist (as of 2026-10-26)
- No ARC result on random RNNs, CNNs or transformers. These appear only as open questions in "Competing with sampling".
- No ARC estimator that matches sampling for deep random MLPs (depth growing with width) or for narrow ones. The paper lists both as open.
- The trained-network case is only partial: the 2-layer, trained-second-layer setting.
