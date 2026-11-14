# ARC on deeper networks, and whether we can improve on it

Written 28 October 2026, 23:50Z, for Kelvin's question: "What is ARC's recent results on deeper networks? ... and whether we'll be able to improve on it." Section 6 gives the read status of every source. Numbers from AIcrowd challenge participants are third-party, mostly AI-assisted and not reproduced; they are marked "participant". Derivations marked "mine" have not been tested.

## Summary

ARC's random-MLP result (Wu et al., arXiv 2605.05179, May 2026) is about width. At fixed depth, cumulant propagation beats sampling by a factor that grows with width. At fixed width it loses ground as depth grows. At width 256 the factorized versions beat Monte Carlo by about 1,000× with 2 hidden layers and 10–100× with 4, and "start to underperform sampling for 8 hidden layers once the maximum cumulant order K reaches 4". ARC conjecture that the error is at most c_K (L/n)^K and measure MSE ∝ L^K, while sampling's error does not grow with depth. Combining that conjecture with ARC's runtime bounds (my arithmetic), the depth at which cumulant propagation stops beating sampling grows only like √n (for K = 2 and for factorized K = 3) or more slowly (for higher K). So depth proportional to width, which ARC name as an open problem, is out of reach at every fixed K. ARC's contest found the same thing: at width 256 and 32 layers, "the best-performing methods in Phase 1 tended to be based on variations of sampling", so the organizers made Phase 2 wider and shallower (1,024 × 16) to give cumulant propagation a chance. I found no newer ARC work on deep networks, residual networks, transformers or trained networks.

Nothing in our current approach improves on this for ARC's problem. For E[output] our forward chain is ARC's K = 2. The control-variate estimator built on it brings no reliable gain on ReLU nets, and participants measured the same loss of coupling with depth on ARC's own nets. The KL chain rule, the Donsker–Varadhan and Hölder bounds and layer-wise twisted SMC all need noise in every layer, so on ARC's noiseless nets they are undefined or vacuous. What our approach changes is the object. If noise is kept at a fixed fraction of the activation scale (as in our nets, where LayerNorm fixes the scale), errors should fade geometrically with depth instead of polynomially (mine, untested), and that would remove the depth problem for every forward method, ARC's included. Our phase R data point the same way: ARC's K = 2 on our ReLU + LN nets at width 256 has error growing about as L² without noise and about as L^1.3 with σ = 0.3 (L = 2 to 8). So a good depth result on noisy nets would tell us about noise. It would not show that our descriptions are better than ARC's.

Section 3 gives an experiment on ARC's exact family that separates the method effect from the object effect. My estimates, explained in section 4: about 5% that anything we have beats ARC's sample-free estimator and RQMC on ARC's nets at depths 16–32. About 60% that we can measure cleanly how much of the depth problem noise removes. About 50% that unbiased rare-event samplers beat ARC's deterministic tail estimates below about 1e-10, at budgets far above ARC's. That last result would extend ARC's regime without touching its depth problem.

## 1. What ARC has shown for deep networks (question a)

### 1.1 The paper

Wu, Lecomte, Winer, Robinson, Hilton, Christiano, "Estimating the expected output of wide random MLPs more efficiently than sampling", arXiv 2605.05179 (v1 6 May 2026, v2 14 May 2026, 68 pages). No venue is listed on arXiv or on ARC's blog.

| Item | Setting |
|---|---|
| Network | M_θ: R^n → R^n with L hidden layers of width n, no biases, ReLU by default; W^(l) iid N(0, 2/n) (He). Each depth L reuses the same network, truncated after L layers. App. F adds GELU (He init) and tanh (critical init at q* = 0.85, σ_w² = 2.025, σ_b² = 0.111, with biases) |
| Input and estimand | x ~ N(0, I_n); E_x[M_θ(x)] per output neuron for one fixed θ (a quenched average) |
| Metric | MSE over output neurons and 5 seeds, divided by the variance of one forward pass, so Monte Carlo with N samples scores exactly 1/N. Truth from 2^34 samples. Cost in FLOPs, with symmetric-tensor work discounted |
| Grid | n from 4 to 256 (figures show 16 to 256), L from 2 to 12, cumulant order K from 1 to 4, four variants (basic, augmented, factorized, factorized augmented) |
| Cost, leading terms (Table 1) | K = 2: 7n³L. Factorized K = 3: 30n³L². Factorized K = 4: 24n⁴L². One Monte Carlo sample costs about 2n²(L+1) |

| Claim | Where | What it says |
|---|---|---|
| Theory | Thms 5.1, 5.2 | For polynomial activations at fixed depth, the basic algorithm reaches MSE O(ε²) in time O(n²/ε²), the same as Monte Carlo. The factorized one does it in O(n/ε²) when ε² = O(1/n²), "a factor of n" faster. "the dependence of our algorithms on depth is worse" |
| Width 256, by depth | §6.2, Fig. 2 | Factorized versions beat Monte Carlo "by a factor of around 1,000 for 2 hidden layers, and around 10–100 for 4 hidden layers. They start to underperform sampling for 8 hidden layers once the maximum cumulant order K reaches 4", and would win again at larger width. Read literally, the cheaper K = 1 to 3 points still win at 8 layers at their own budgets, and what is lost is the high-accuracy end |
| Depth scaling | §5, App. D, Fig. 8 | Conjecture: "mean squared error at most c_K(L/n)^K, although the dependence on L is difficult to analyze precisely". At n = 256 and L = 2, 4, 8 the error "roughly follows a curve proportional to L^K". "This depth scaling is worse than Monte Carlo sampling, whose error does not increase with depth. We believe that a sample-free algorithm can be developed whose error also does not increase with depth either" |
| Narrow nets | App. C, Fig. 7 | "Our algorithms perform poorly at low width, especially when the number of hidden layers L is large, and can even get worse with the maximum cumulant order K in this setting." At high depth the factorized versions cost more FLOPs than the basic ones |
| Other activations | App. F | GELU and tanh at n = 256, L = 2 to 12: "qualitatively similar" |
| Low probability | §6.6, App. G | n = 256 and 4 hidden layers (three ReLU layers, then the threshold 1{z > 3}). Factorized K = 3 at the FLOPs of 5,695 Monte Carlo samples (about 3e9 FLOPs by my count) has relative error under 30% for probabilities 100× below 1/N and beats every baseline down to nearly 10,000× below; "At sufficiently small probabilities, however, none of the methods significantly outperform zero". The baselines were Monte Carlo, Monte Carlo with Gaussian or cumulant extrapolation, and zero; there was no importance sampling or splitting. Only depth 4 was tested |
| Wall clock | App. I | "In most cases, our algorithms underperform sampling" (the code is unoptimized) |
| Open problems | §7.3 | "we do not know whether it is possible to match the performance of random sampling at large depth (i.e., asymptotically as the number of hidden layers L tends to infinity), or for narrow networks". Also listed: non-Gaussian or dependent weights, learned and adversarial MLPs, empirical input distributions |

ARC's public code (github.com/alignment-research-center/mlp_cumulant_propagation) changed in July and August 2026. Its TODO.md says the earlier implementation summed more Edgeworth terms than the paper describes, and that "All cached kprop outputs, data files, and FLOP counts (and any downstream figures) for k_max >= 2 are stale". The figure-level numbers above may move; I have not checked by how much.

### 1.2 What ARC's scalings imply for depth (mine)

Take ARC's conjectured error c_K (L/n)^K, their runtime bounds (basic at most c′_K L n^{K+1}, factorized at most c″_K L² n^K for K ≥ 3), and Monte Carlo at about 2Ln² FLOPs per sample. Run each algorithm at its own cost. The ratio of its MSE to Monte Carlo's at the same FLOPs is then:

- for the basic algorithm of order K, (c_K c′_K / 2) · L^K / n, so it loses beyond a depth L* ∝ n^{1/K};
- for the factorized algorithm with K ≥ 3, (c_K c″_K / 2) · L^{K+1} / n², so L* ∝ n^{2/(K+1)}.

For K = 2, and for factorized K = 3, the crossover depth grows like √n. A higher K helps only while L/n is small, and its crossover depth grows more slowly. No fixed K reaches depth proportional to width, which is the regime ARC call open. A participant's grid of uncorrected K = 2 against FLOP-matched Monte Carlo (topic 18177, medians over 48 nets in the deep cells) fits L* ≈ 2√n:

| Width n | 64 | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|
| Measured crossover depth (participant) | 16–24 | about 20 | about 32 (ratio 1.04, 95% CI 0.75–1.38) | 32–48 | above 48 |
| 2√n | 16 | 23 | 32 | 45 | 64 |

### 1.3 ARC's posts and related work

"Mechanistic estimation for wide random MLPs" (Hilton, 7 May 2026) summarizes the paper and adds in a footnote: "It remains an open problem to match the performance of random sampling for random MLPs in other regimes, for example, with the depth growing linearly in the width." Random networks are called the "base case". For trained networks, the estimator would need "advice" saying which higher-order deviations to track, updated with each training step (the "inductive step").

"Competing with sampling" (Neyman, Lecomte, Wu, Winer, Hilton, Robinson, 18 November 2025) states the matching sampling principle. Its random-MLP instance has a fixed depth: "there are k hidden layers (k is some fixed constant)". Its working test for "mechanistic" is "whether it avoids any random or pseudorandom sampling". It lists random RNNs ("much more difficult than the MLP setting, because of the weight sharing"), narrow MLPs, random CNNs and random transformers as open. For the trained two-layer case it reports "substantial progress" and names "deeper MLPs" as one direction to generalize.

"Mechanistic estimation for expectations of random products" (Hilton et al., 15 May 2026) describes deduction–projection estimators with "mechanistic L² sketching". The forward state (the "head" of a product) keeps only its projection onto the leading eigenfunctions of the kernel of the random remainder (the "tail"). For halfspace intersections, #3-SAT and permanents these "roughly match or outperform random sampling up to logarithmic factors", except for {0,1}-permanents. There is no neural depth here. It matters for us because it is ARC's own way of using backward (tail) information to choose the forward description.

AlgZoo (Hilton, 26 January 2026) works on tiny trained RNNs. The brute-force analysis that settles the 10-parameter model would not scale, because "the number of regions would grow exponentially with depth". Matching sampling on the 432-parameter M_{16,10} would be "a major research breakthrough". Hilton wrote on 17 August 2026: "we are not much closer to succeeding at the challenge".

G. Wu and Hilton (ICLR 2025, arXiv 2410.13211) estimated rare outputs of trained 1-, 2- and 4-layer transformers (width 512), with probabilities from 1e-9 to 1e-5 and a budget of 2^16 model calls. Importance sampling beat activation extrapolation. "ITGIS does better on smaller models, while MHIS does better on larger models", which the authors attribute to larger models being "less easily approximated as linear functions".

Winer's agenda post (9 June 2026) adds no depth results; I did not watch Christiano's IPAM talk (31 August 2026). ARC's blog index ends at 9 June 2026. Hilton's LessWrong activity up to 27 October has no new research post, and arXiv has nothing newer on this from the ARC authors.

### 1.4 The White-Box Estimation Challenge

The contest, run by ARC with AIcrowd, uses the paper's family (ReLU, He init, no bias, Gaussian input). It scores the final-layer post-ReLU means as MSE × max(0.1, used/budget). Any method is allowed, but ARC "strongly expect the best possible algorithms for this problem to be 'mechanistic' (i.e., to avoid black-box sampling entirely)". ARC's stated reason for the contest: "we produced white-box methods that outperform black-box methods for MLPs with large width, but they break down as the depth grows".

| Round | Width × hidden layers | FLOP budget per net | Outcome |
|---|---|---|---|
| Warm-up (June) | 256 × 8 | 6.8e10 | The leaders were "variants on the factorized 3rd cumulant propagation algorithm ... combined with learned networks that consume the cumulant estimates as features" (Hilton) |
| Phase 1 (18 June to August) | 256 × 32 | 2.72e11 | Hilton: "Our existing algorithms scale poorly with depth". AIcrowd organizer, 23 August: "The best-performing methods in Phase 1 tended to be based on variations of sampling." Official Phase 1 results are not yet published |
| Phase 2 (23 August to 21 October) | 1,024 × 16 | 2^41 ≈ 2.2e12 | "We have made the network wider and shallower for Phase 2, to make cumulant-propagation-style methods more competitive. Nevertheless, 16 layers is deep enough that we expect handling depth-dependence well to be a challenge." The private re-evaluation is pending; a participant quoting the rules gives 15 November for the winners |

At 256 × 32 the leading-term cost of factorized K = 3, about 30 · 256³ · 32² ≈ 5e11 FLOPs, is roughly twice the Phase 1 budget.

Participant measurements that bear on depth (forum text read, attached PDFs not read):

| Finding | Source topic |
|---|---|
| Raw final-layer MSE at 256 × 32: best published sample-free entries about 2.7e-6 to 1.5e-5; a moment chain fed the true means and covariances at every layer 8.1e-7; plain Monte Carlo at the full budget about 6.5e-7; best published structured sampler about 1.5e-7. The sample-free entries use far fewer FLOPs, but the 0.1 floor on the compute multiplier stops that from paying | 18177, 18157 |
| Depth exponents of the layer-32 error at width 256 (30 nets): 2.69 for factorized K = 3 and 4.13 for factorized K = 4, close to ARC's L^K. A sample-free variant that regenerates the fourth cumulant from at most four covariance-response modes lowers the exponent to 2.21 and the error by 77% against K = 3, and is still 7.9× worse than a structured QMC sampler on the same nets | 18157 (submission 327725) |
| An error made at layer ℓ keeps ∏_k (1 − arccos(c_k)/π) of its energy at the output, where c_k is the mean-field replica cosine. Two teams confirm it within 1–6%. Their summary: a constant per-layer error accumulates to "about 6 layers' worth at depth 16, 10 at depth 32 and 19 at depth 64" | 18282, 18292, 18267 |
| The nets collapse toward a common mean with depth: the covariance's participation ratio falls from 128 to 5.2 over 32 layers, and the mean carries 94.6% of the activation energy at layer 32 | 18097, 18193 |
| Coupled surrogates lose correlation with depth: the per-neuron coupling correlation of the Gaussian-optimal affine surrogate is 0.86, 0.55, 0.41, 0.25 at 1, 4, 8, 32 layers | 18085 |
| At 1,024 × 16 the stock covariance-propagation baseline beats a Sobol sampler 3.4× on the score. Factorized K = 3 chains with fitted corrections reach about 2e-9 to 4e-9 adjusted, reported as 84× below full-budget Monte Carlo at 14% of the budget | 18219, 18282, 18251 |
| Two entries at the top of the Phase 1 public board, with raw MSE near 1e-9 and no published method, were removed in an 18 August rescore without explanation | 18157 |

For the survival law, the arc-cosine recursion of He-ReLU nets gives 1 − arccos(c_l)/π ≈ 1 − 3/l at large depth (mine). An error made at layer ℓ therefore keeps about (ℓ/L)³ of its energy at layer L, which matches the participants' numbers.

### 1.5 Where ARC's methods break

1. Error accumulation. Each layer's truncation error is carried forward. In noiseless He-ReLU nets it fades only as (ℓ/L)³, and the MSE grows like L^K at fixed width (ARC's measurement, and the participants' exponents of 2.7 and 4.1 for K = 3 and 4).
2. Cost. Factorized cost grows like L² n^K, so at depth factorization stops paying.
3. Crossover depth. It grows like √n or more slowly (§1.2), so no fixed K reaches depth proportional to width.
4. Bias floor. A deterministic estimate does not improve as the budget grows between K steps, so at large budgets sampling catches up. One participant's hybrid (4.5e-7 adjusted) attributes at least 58% of its error to closure bias (18085).
5. Degeneracy. As the nets collapse to a common mean, the input-dependent part that sampling averages becomes small, while the chain has to get the mean right to that same small accuracy.
6. Narrow nets. Raising K can make the estimate worse.

## 2. Our approach against these difficulties (question b)

On our side I read the program (outbox/cm-interp-program-2026-10-28.md), the public page, phase B (outbox/cm-interp-phaseB-2026-10-28.md, including Update 8), astra's review of phase B (/tmp/pbrev/review.md), and phase R (outbox/cm-interp-random-2026-10-28.md, with work/cm-interp-R/results).

### 2.1 Difficulty by difficulty

| ARC's difficulty | Does our approach avoid it? | Our evidence and its limits |
|---|---|---|
| Error compounding along depth | Only in the accounting. The KL chain rule, KL(P‖Q) = Σ_l E_P ‖f_l(h) − g_l(h)‖² / 2σ², is exactly additive and scores each surrogate layer on the real path. D_∞ coverage also adds over layers | Phase R: per-layer description costs stay flat over 16 layers at d = 32 and 8 layers at d = 64, and end-to-end KL at depth 4 is 30–70% of the per-layer sum. But each term grows linearly with width and as 1/σ². K = 2's implied linear map costs 68, 451 and 547 bits per layer at d = 256 (critical, chaotic, ReLU), and Donsker–Varadhan intervals built on it are 37 to 26,500× wider than sampling's confidence interval. End-to-end depth was measured only to 4 layers; astra's review: "Local KL stability does not prove global error cannot accumulate." None of this is defined at σ = 0 |
| Composition of certificates | No; our certificates have their own depth problem | Phase B Update 8: composed Rényi bounds grow by 0.4–10.6 bits per layer, because each step back doubles the order, while the directly measured D_2 stays at or below 0.05 bits |
| Width | No; ours is worse | ARC's error falls as n^{-K}. Our coverage and KL costs grow linearly in n for every family tried (Gaussian, product, copula). Every rigorously computable certificate in phase R was vacuous, and the lower bound rules out useful certificates on these chains beyond width about 64. Phase B's non-vacuous Hölder bounds at width 8 rest on estimated divergences, not proofs |
| Bias floor at large budgets | Partly | Our hybrids (the K = 2 mean plus a coupled correction, twisted SMC, splitting) are unbiased and improve as 1/N. The control variate gains 70–1,000× over Monte Carlo in the ordered regime (nearly linear nets), 2–5× at criticality, and 0.75–1.4× for ReLU, less at 8 layers than at 2; its residual variance for ReLU is 60–67% of the plain variance already at L = 4. Participants measured the same decline on ARC's nets (coupling 0.25 at 32 layers). And these estimators sample, so they do not address ARC's mechanistic goal |
| Rare events below the deterministic floor | Partly | Splitting reaches 2^-60 with error bars where Gaussian tails fail. But at ARC's rare-event shape (width 256, 4 layers, ReLU), our chain-guided twisted SMC worked only down to 2^-20, with relative error 0.47 at 2.8e9 FLOPs for one event. ARC report under 30% at about 2e-6 for all 256 outputs at about 3e9 FLOPs. (The nets differ: ours have LN and noise) |
| Sample-free estimation | No | Our only sample-free mean estimate is K = 2 |
| Deterministic networks | No | Layer-wise SMC, the KL chain rule and the DV and Hölder bounds all need per-layer noise. At σ = 0 only input-space samplers remain |

### 2.2 What comes from changing the object

Noise changes how fast errors fade. Under the participants' law, a mean error passes a He-ReLU layer with energy factor 1 − arccos(c)/π, where c is the replica cosine of the pre-activations. Suppose independent noise is kept at a fixed fraction σ of the activation scale. Then c is at most 1/(1 + σ²), and an error shrinks by a factor e within at most about π/arccos(1/(1 + σ²)) layers: about 8 at σ = 0.3, 22 at σ = 0.1 and 74 at σ = 0.03. This is mine (Gaussian pre-activations, ReLU, no LN) and untested. Without normalization the accumulated noise inflates the activation scale, the decorrelation per layer shrinks with depth, and forgetting is slower. If errors do level off at this forgetting depth, the arithmetic of §1.2 makes the advantage of ARC's basic K-propagation over Monte Carlo independent of depth. A depth-flat result on noisy nets would then hold for ARC's method as well.

Our data lean this way. On our ReLU + LN nets at d = 256, ARC's K = 2 has normalized MSE 1.1e-5, 4.0e-5 and 1.6e-4 at L = 2, 4, 8 without noise (about L², as ARC conjecture for K = 2). With σ = 0.3 it has 1.2e-5, 2.9e-5 and 7.3e-5 (about L^1.3). That is one seed without noise and two with; at d = 16 and 64 the pattern is not clean. How LN is approximated also matters at small width: at d = 16, L = 8, K = 2's error differs by a factor of 10 between the two LN treatments in phase R.

Four more effects come from the object:

- LayerNorm caps the output: the projection of the last layer onto w is at most √d times the norm of w's centered part. Phase R's finding that Gaussian (K = 2) output tails are "up to 47 bits too high", and that handling the last layer exactly fixes them, follows from the cap. In phase R's no-LN control at d = 16, the K = 2 tail is too low by up to 21 bits and both last-layer-exact chains are off by 21 to 30 bits at 2^-60.
- LayerNorm removes each sample's overall scale, which in bias-free ReLU nets is set by the input norm. Participants report that errors along the scale direction never fade in ARC's nets (18248, 18282); I have not checked that their scale direction is exactly this one.
- Noise adds its own Gaussian tail to every rare event, and that part is known in closed form. ARC's rare events are input tails of a deterministic map.
- Precision: phase R's description costs vanish at 0.5 bits per dimension (σ = 1) and grow roughly as 1/σ² (waste of 81 bits per layer at σ = 0.1, d = 16). Every claim has to state σ, and σ → 0 is ARC's problem.

These count as object effects only for ARC's question. The noisy-channel program is about noisy networks by design, and for it they are real properties that make width, rather than depth, its binding constraint.

### 2.3 What phase R's comparisons with ARC do and do not show

- "the chain is ARC's K = 2 and reproduces ARC's result, 13-810x better than sampling": this is K = 2 only, on LN nets, for L ≤ 8. It reproduces ARC's pattern, not ARC's best method.
- "K = 2 output tails are up to 47 bits too high": this is against K = 2 on LN nets. ARC's rare-event method is factorized K = 3 on nets without LN, and we never ran it.
- The control variate's "2-1,000x": the 1,000× is in the ordered regime, where K = 2 alone already has normalized MSE 5.5e-9 at d = 256, L = 8. For ReLU, ARC's default, it brings no reliable gain.

## 3. The cleanest experiment (question c)

The claim under test is that some part of our approach estimates E[output] of ARC's random MLPs at depths 4–32 with lower MSE than ARC's cumulant propagation and the best simple sampler at equal FLOPs. The test can fail on ARC's own nets. Its noise arms then show whether any gain seen elsewhere comes from the object.

Objects, all on the same weights:

- O0, the primary object: ARC's family exactly, with ReLU, W iid N(0, 2/n), no bias, no LN, no noise, and x ~ N(0, I_n).
- Oσ: O0 with N(0, (σ r_l)²) noise added after each ReLU, where r_l is the RMS of layer l's ReLU output in that noisy net, computed once from the weights (for example by K = 2 propagation), so the noise stays a fixed fraction σ of the activation scale; σ ∈ {0.03, 0.1, 0.3}, with forgetting depths of about 74, 22 and 8 by §2.2.
- O_LN, tertiary: phase R's LN + noise nets, where ARC's method needs an LN approximation.

Estimand and grid: per-neuron final-layer post-ReLU means. This is the challenge form; ARC's footnote says that dropping the final linear layer "makes essentially no difference". Depths L ∈ {4, 8, 16, 32}, obtained by truncating one 32-layer net, as ARC do. Width 256 is primary; widths 64 and 1,024 (the latter to L = 16) are secondary. Use at least 20 nets per cell and report median and mean normalized MSE with bootstrap intervals, because per-net errors are heavy-tailed. Truth comes from our own 2^26 samples per net and object, about 3e14 FLOPs per 256 × 32 net. That gives a normalized floor of 1.5e-8, far below the errors at issue. AIcrowd's public 1e9-sample truths serve as a check where they exist.

All methods use one FLOP counter (flopscope, fp32), and all setup is counted:

- Baselines: Monte Carlo; RQMC (scrambled Sobol or a randomly shifted lattice), because participants found structured samplers at 256 × 32 about 4× better than plain Monte Carlo; and ARC's mlp_kprop at the current main branch, K = 1 to 4, basic and factorized. On Oσ, kprop handles the noise exactly: independent Gaussian noise adds (σ r_l)² to the diagonal of the second cumulant and changes no other cumulant.
- U1, the control variate: the K = 2 (and K = 3) mean plus the correction y − y_surrogate on coupled paths (same x, and the same ε on Oσ).
- U2, the only way our noisy object can bear on O0's estimand: kprop's estimate on Oσ plus the mean of M_0(x) − M_σ(x, ε) over coupled samples. Its bias is kprop's error on Oσ and its relative variance per sample is of order Lσ² (my estimate, assuming order-one layer-to-output gains). It can therefore beat Monte Carlo only if Lσ² < 1 and kprop is much more accurate on Oσ than on O0.
- U3, tails, at ARC's §6.6 shape (width 256, three ReLU layers, threshold at 3): splitting and importance sampling over x, with the chain's ψ as the twist. Layer-wise twisted SMC is available only on Oσ.

Endpoints:

- Primary, the method effect on ARC's problem: on O0 at width 256 and L ∈ {16, 32}, the advantage A = min(MSE_MC, MSE_RQMC) / MSE at equal FLOPs, over budgets from 1e9 to 3e11 FLOPs (Phase 1's budget is 2.72e11). An improvement on ARC means A_ours > max(1, A_kprop) at every budget from 1e10 to 3e11, with intervals excluding 1.
- Secondary, the object effect: A_kprop(Oσ) / A_kprop(O0) by L and σ.
- Scaling: the crossover depth L*(n) for n = 64, 128, 256 and 512. An exponent above 1/2 would be new, since ARC's conjectured scalings give at most 1/2 (§1.2).
- Tails: FLOPs per event to reach 50% relative error at p ∈ {1e-6, 1e-8, 1e-10}, reported per event and amortized over the 256 outputs, against kprop K = 3 at about 3e9 FLOPs.

Fairness:

1. Claims about ARC's problem are made on O0 only. Results on Oσ and O_LN are reported as object effects.
2. ARC's method runs on every object, with the noise handled exactly, so noise cannot favor our estimators.
3. Each object's MSE is normalized by its own single-sample variance, and Monte Carlo on noisy objects integrates the last layer's noise analytically, so the noise does not handicap sampling.
4. The sampling baselines include RQMC as well as plain Monte Carlo.
5. All setup FLOPs are counted (pilot runs, fits, SMC). Sample-free and sampling estimators are reported separately, since only the sample-free ones bear on ARC's open problem.
6. Anything fitted (corrections, ψ) is fitted on nets not used for evaluation.

Run two controls first: ARC's Fig. 2 trend at width 256 and L = 2, 4, 8 (within 2×), and K = 2's parity with Monte Carlo at 256 × 32.

Predictions, written before any run:

- P1: on O0 at width 256, kprop K = 2's normalized MSE grows roughly as L^1.5 to L² up to L = 32 (the participant grid gives ratios to FLOP-matched Monte Carlo of 0.11, 0.37 and 1.04 at L = 8, 16, 32, then 1.07 at 48, where the nets have collapsed).
- P2: on Oσ at L = 32, kprop's error is at least 10× below O0's at σ = 0.3, and within 2× of it at σ = 0.03.
- P3: U1's coupling correlation on O0 is 0.6 or less for L ≥ 8, and U1 gains at most 1.5× over RQMC at L ≥ 16.
- P4: U2 gains at most 2× over RQMC at L ≥ 16 for budgets of 1e10 or more.
- P5: at about 3e9 FLOPs, kprop K = 3 beats our per-event samplers for p ≥ 1e-7. At p ≤ 1e-10, splitting reaches 50% relative error or better with at least 1e11 FLOPs per event, where kprop does no better than zero.

The kill tests take under a CPU-day. First, U1's coupling correlation on O0 at L = 16 and 32: U1 is dead if it is 0.6 or less. Second, kprop on Oσ against O0 at L = 16 and 32: U2 is dead unless the gain at σ ≤ 1/√L is at least 10×. Third, the per-event cost of splitting at 1e-8 on the §6.6 net.

Cost: factorized kprop with K = 3 at 256 × 32 needs about 5e11 FLOPs per net, which takes minutes on CPU. Factorized K = 4 needs about 1e14 per net, so beyond L = 16 it needs a GPU or is dropped. The truths need about 3e14 FLOPs per net and object, roughly 15 CPU-hours for 20 nets of one object.

## 4. Can we improve on ARC's deep-network results? (question d)

| Outcome | What would count | Estimate | Why |
|---|---|---|---|
| Beat ARC on ARC's problem | A sample-free estimator on O0 at width 256, L = 16–32, with A > max(1, A_kprop) at budgets from 1e10 to 3e11 on at least 20 fresh nets; or a crossover depth L*(n) growing faster than √n | about 5% | Our only sample-free mean estimator is K = 2. None of the contest's published write-ups (dozens, mostly AI-assisted) describes a sample-free method that beats sampling at 256 × 32; the best one is 7.9× behind a structured QMC sampler, and a moment chain fed the true means and covariances is still 5× behind the best published sampler in raw MSE |
| Beat the best sampler with a hybrid | At least 2× over RQMC on O0 at L ≥ 16 | 10–15% | U1 and U2 run into the measured loss of coupling and the bias floor, and the contest's structured samplers are already about 4× better than plain Monte Carlo |
| Extend ARC's tail regime | On O0 at the §6.6 shape, relative error of 50% or better at p ≤ 1e-10, where ARC's estimate is no better than zero, at stated FLOPs per event | about 50% | Splitting is strong at extreme rarity. But it is a generic sampler, it works one event at a time, and at ARC's budget ARC wins |
| Diagnose the depth problem | With ARC's own method, an object effect of at least 10× at L = 32 for σ between 0.1 and 0.3, matching the forgetting-depth prediction | about 60% | Our σ = 0 versus σ = 0.3 data already lean this way. This would be a finding about noise, not an improvement on ARC |
| A certificate ARC does not have | A non-vacuous two-sided bound within 10× of sampling's confidence interval at width 64 or more | about 10% | Every family we tried is vacuous at width 64 and above; event-specific backward certificates are untested |

Things that would not count: results on noisy or LN nets alone; comparisons against K = 2 rather than ARC's factorized K = 3 and 4; per-event tail results compared with ARC's all-outputs estimate without amortizing; FLOP counts that leave out setup; nets used both to fit and to evaluate.

## 5. Questions for Kelvin

1. Which claim should this branch aim at? One option is an improvement on ARC's noiseless problem, for which we have no sample-free idea yet. The other is a statement about noisy networks: at b bits per dimension depth is cheap and width is the constraint. The experiment serves both, but the primary endpoint and the write-up differ.
2. Is U2 worth running? It is the only way our noisy object touches ARC's estimand, and its bias floor makes a large win unlikely.

## 6. Sources and read status

| Source | Status |
|---|---|
| Wu, Lecomte, Winer, Robinson, Hilton, Christiano, arXiv 2605.05179v2, https://arxiv.org/abs/2605.05179 | Read: main text, App. A–J, supplement S.1 and S.4.3; proofs skimmed. Figure values were not extracted, so only the text claims are used. No venue listed; I found no NeurIPS 2026 listing, so "NeurIPS 2026" is unverified |
| ARC code: https://github.com/alignment-research-center/mlp_cumulant_propagation (commits to 4 August 2026, TODO.md) | Read |
| Hilton, "Mechanistic estimation for wide random MLPs", 7 May 2026, https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/ | Read in full |
| Neyman et al., "Competing with sampling", 18 November 2025, https://www.alignment.org/blog/competing-with-sampling/ | Read in full |
| Hilton et al., "Mechanistic estimation for expectations of random products", 15 May 2026, https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/ | Post read in full; the linked internal notes not read |
| Hilton, Christiano, Wu, "Announcing the ARC White-Box Estimation Challenge", 2 June 2026, https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/, and its LessWrong version with the 23 June Phase 1 update and comments, https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c | Read |
| AIcrowd organizer posts: Phase 2 launch, 23 August 2026 (https://discourse.aicrowd.com/t/18197), Phase 2 update, 17 October 2026 (https://discourse.aicrowd.com/t/18251) | Read |
| AIcrowd participant posts, topics 18063, 18085, 18097, 18147, 18157, 18177, 18193, 18198, 18219, 18251 (replies), 18267, 18282 (https://discourse.aicrowd.com/t/NNNNN) | Forum text read; attached PDFs not read; numbers not reproduced; most declare AI assistance |
| Hilton, "AlgZoo", 26 January 2026, https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/, and his comment of 17 August 2026 | Read |
| Winer, "A Mike's-Eye View of ARC's Research", 9 June 2026, https://www.alignment.org/blog/a-mikes-eye-view-of-arcs-research/ | Read for depth content; none found |
| G. Wu and Hilton, "Estimating the Probabilities of Rare Outputs in Language Models", ICLR 2025, https://arxiv.org/abs/2410.13211 | Read: abstract, setup, results, conclusion; appendices skimmed |
| Christiano, "Mechanistic Estimation", IPAM, 31 August 2026, https://www.youtube.com/watch?v=_BPsuomAcoI | Not watched |
| Winer and Hanin, "Deep Neural Nets as Hamiltonians", arXiv 2503.23982; Dunbar and Aaronson, arXiv 2510.06527 | Abstracts only |
| Search coverage: ARC blog index (7 posts, newest 9 June 2026); Hilton's LessWrong activity to 27 October 2026 (GreaterWrong); arXiv API by author (Hilton, Lecomte, Winer, Neyman, Christiano, W. Wu) and by phrase ("mechanistic estimation", "cumulant propagation"); ARC's GitHub organization; the AIcrowd forum's topic list to 28 October | Checked; no newer ARC work on depth, residual nets, transformers or trained nets |
| Ours: outbox/cm-interp-program-2026-10-28.md; work/interp-page/index.html; outbox/cm-interp-phaseB-2026-10-28.md (TL;DR, setup, Update 8); /tmp/pbrev/review.md; outbox/cm-interp-random-2026-10-28.md (final, 23:23Z); work/cm-interp-R/results/tables.md and eout_*.json; outbox/arc-random-networks-search-2026-10-26.md | Read |
