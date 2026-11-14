# PREREG: sparse codes on random MLPs (work/sparse-codes), 29 October 2026

Written 16:35Z, before any main run. Target: M1 (a sample-free estimator for deep random MLPs at depths that grow with
width). The question is whether the program's leading family, k-sparse codes h = D a + noise with transcoders for the
layer maps, can carry what the Gaussian closure misses, compactly.

## Brief and retargets
- Brief (main, relaying Kelvin): test the key risk on random networks first. The risk is that features are too
  dependent, with dense couplings. Dictionaries in bits; dependence; propagation; three kill tests.
- Retarget 1 (main, ~16:15Z), adopted. An independent-feature product cannot work, by a short calculation. With fresh
  next-layer weights, dropping the off-diagonal covariance changes Var(z'_k) by a relative O(rho_rms), and this does
  not shrink with width. So every family below carries the full covariance. The question becomes the higher-order
  (3rd/4th) structure beyond the covariance, measured against neuron-level K=2 and ARC's cumulants. Propagation runs
  through transcoders that carry the covariance.
- Retarget 2 (main, ~16:40Z), adopted. Frame the sparse code as a latent-conditional Gaussian: exact sparse latent,
  plus a Gaussian residual with full covariance given the latent. This is the analogue of the boundary-family lane's
  dense collapse-mode latent. Report:
  - the higher-order structure carried, against a dense k-dim latent at matched size, and against K=2 and K3f;
  - whether the residual is conditionally Gaussian given the features (per-neuron excess kurtosis).
- The independent-product kill test of the brief ("couplings dense, so a factored model misses the next layer by more
  than a Gaussian closure") is therefore replaced by KT2 and KT3 below. The phi and magnitude correlations are not
  measured.

## Seen before this file (dev seed 64 only, plus timings)
- `scratch/explore1.py`, width 64, sigma = 0.01, seed 64, 2^16 paths (logs/explore1_*_64.log).
  - ARC He-ReLU: PR falls from 41 (layer 1) to 1.2 (layer 32). Eigenvalues of Cov(a_l) above s^2: 64, 64, 62, 60, 55,
    41 and 33 (layers 1, 2, 4, 8, 16, 24, 32). PCA components needed for a per-coordinate error below the noise: 64, 64,
    59, 52, 47, 29 and 22. Active neurons: about 32 of 64 (24 at layer 32).
  - ARC, noise-driven Var(a | x): 1, 2.6, 12, 33, 30 and 60 s^2 per coordinate at layers 2-32 (0-3.4% of the
    variance).
  - Chaotic tanh: about 9,000 s^2 per coordinate and PR 25-39 at every layer; 63 of 63 eigenvalues above s^2.
  - Ordered tanh: the variance falls to 0.2-0.3 s^2 per coordinate from layer 8 on (the mean alone is below the noise).
- Smoke runs of `gpu/run_main.py` at tiny sample sizes (2^12-2^13), dev seed 64, layers 4 and 8, 1-2 SAE epochs. Code
  paths only; their numbers are dominated by sampling noise and undertraining.
- ARC kprop K=2 and K=3f ran on eval seed 65 (widths 64 and 128). FLOPs and run times were printed; no errors were
  computed (no truth existed).

## Nets (exact)
- **arc:** ARC's He-ReLU family as in work/arc-noisy (`ext/anet_copy.py`, unedited):
  - x ~ N(0, I_n), z_l = W_l h_{l-1}, W iid N(0, 2/n), a_l = ReLU(z_l), h_l = a_l + s_l eps;
  - s_l = sigma r_l, with r_l from ARC's seeded 2^12-path pilot (`ext/rtable_sc.json`);
  - no bias, no LayerNorm.
- **tord / tcha:** phase R's tanh + LN nets (`ext/rnet_copy.py`, unedited): z = W h + b, W iid N(0, g^2/n),
  b iid N(0, 0.25), a = LN(tanh(z)), h = a + sigma eps.
  - g = 0.25 (ordered) and g = 3 (chaotic), the gate-entropy census's settings.
  - Caveat from the random-phases lane: LN tanh "ordered" is ordered here because of the bias, not the gain alone.
- Depth 32. Widths n = 64, 128 and 256.
- sigma = 0.01 (primary) for all families; sigma = 0.001 (secondary) for arc.
- Seeds: 64 = development (exploration above only); 65 = evaluation (every number in the report); 66 = replicate (arc,
  sigma = 0.01, n = 64 and 256, if time allows).
- Layers: arc {1, 2, 4, 8, 16, 24, 31}; tord {1, 2, 4, 8, 31}; tcha {1, 8, 31}. Layer 31 stands for depth 32 because
  E2 needs a next layer.

## Definitions
- a_l is the noiseless post-activation, the thing the next layer's noise is added to; s = s_l is that noise's sd.
- **Error in bits relative to the noise:** bits = E||a - a_hat||^2 / (2 s^2) / ln 2, the path KL of replacing a by
  a_hat. **Noise units** = E||a - a_hat||^2 / (n s^2); below 1 means "below the noise".
- **SAEs:** TopK (`lib/sae.py`).
  - Sizes F = n, 2n, 4n, 8n, 16n; k = n/16, n/8, n/4, n/2.
  - Training: 2^18 fresh paths, 30 epochs, batch 2048, Adam (lr 2e-3 / sqrt(max(1, F/4n)), linear decay over the last
    40%), AuxK for dead latents, unit-norm decoder.
  - Scored on 2^15 held-out paths, both raw and with a least-squares refit of the coefficients on the TopK support.
  - Training-length control: F = 4n is also trained with 2x epochs at arc layers 8 and 24 (all widths).
- **Baselines (E1):**
  - mean only (k = 0);
  - PCA-k (top-k components);
  - neuron top-k around the mean;
  - for ReLU, neuron top-k from zero (the exact code is the active set, L0 = #active).
- **Latent-conditional Gaussians (E2, E3).** All carry the exact mean and full covariance of a_l (up to sampling).
  - G: N(mu, Sigma). One step of this is K=2.
  - DENSE(k): the top-k PCA coordinates are exact (taken from true samples); the other n - k coordinates are
    independent Gaussians with their variances.
  - DENSEQ(k), k <= 16: as DENSE(k), with a residual mean quadratic in the latent (regressors [y, y_i y_j]) and a
    Gaussian residual with the Schur-complement covariance.
  - SPARSE(F, k): the SAE reconstruction a_hat = D TopK(E a) is exact (from true samples). The residual is
    r | a_hat ~ N(m_r + B (a_hat - E a_hat), Sigma_e), with B the linear regression of r on a_hat and Sigma_e its
    Schur complement. This matches the full covariance exactly.
  - Matched size, two pairings: (i) equal latent dimension per sample, DENSE(k) against SPARSE(F, k) with L0 = k;
    (ii) equal map parameters, n k for DENSE against n F for SPARSE, so DENSE(k = F) when F <= n/2. Both are
    reported.
- **E2 one-step metrics.** 2^18 fresh true states a_l, pushed one layer with common noise (the models share the
  latent and the noise with the truth):
  - cap3 = 1 - sum_k (kappa3_model(z'_k) - kappa3_true(z'_k))^2 / sum_k kappa3_true(z'_k)^2, over the next layer's
    pre-activations z'_k. cap4 is the same for kappa4.
  - Mean error: nMSE of E[act(z')], i.e. mean_k err_k^2 / mean_k Var(act(z'_k)). act = ReLU, or tanh before LN.
  - References: G (analytic), EW3/EW4 (Edgeworth readouts with the true per-neuron kappa3, kappa4 of z': what a
    per-neuron cumulant closure could do at one step), TRUE_indep (an independent fresh truth; the sampling floor
    for independent samples).
- **Residual Gaussianity (E2).** Per-neuron excess kurtosis of the model's residual e given the latent:
  - marginal;
  - after dividing by a conditional sd regressed on the latent (half to fit, half to score; sparse: the 512 most
    active features' magnitudes and indicators; dense: [y, y_i y_j]);
  - both in neuron space and projected by W_{l+1} (what the next layer sees).
  - Medians and 90th percentiles over neurons.
- **E3 representation at a cut (arc).**
  - Cuts m = 4 (to depth 8), 8 (to 16 and 32), 16 (to 32), 24 (to 32).
  - 2^20 states drawn from each model at the cut (true states for the latent, common random numbers across models),
    pushed through the real noisy layers m+1..L.
  - nMSE of E[a_L] against a 2^24-path truth, minus both Monte Carlo floors. TRUE (no replacement) is the pipeline
    control.
  - Compared with ARC's kprop K=2 and K=3f run end to end (`scratch/run_kprop.py`, ARC's own counter for FLOPs).
- **E4 transcoder chain (arc).** One transcoder per layer l = 1..32, h_{l-1} -> a_l:
  - TopK during training (F = n and 4n, k = n/8 and n/4); at inference a JumpReLU with per-feature thresholds taken
    from training;
  - replacement chain: h_l = a_hat_l + s_l eps;
  - path KL in bits = sum_l E_real ||a_l - T_l(h_{l-1})||^2 / (2 s_l^2 ln 2);
  - the chain's own E[a_L] by sampling, against the truth;
  - a deductive feature-level Gaussian closure through the chain, carrying the mean and full covariance of the
    features (per-feature closed forms and Mehler/Hermite off-diagonals), read out at depths 8, 16 and 32 against the
    truth, neuron K=2 and K3f;
  - code check: the identity transcoder (F = n, E = W, D = I, threshold 0) must reproduce neuron K=2.

## Kill tests (fixed now)
- **KT1 (dictionaries; brief's test 1).** For a family x width x layer: dead at noise precision if no SAE with F <= 16n
  and L0 <= n/4 reaches noise units < 1 (raw or refit) on held-out data. The n/2 column is reported, not used: the
  neuron code is exact at L0 = #active (about n/2).
- **KT2 (sparse vs dense latent; retargets 1-2).** arc, sigma = 0.01, layers 8, 16, 24, 31, widths 64/128/256. Fires
  if at matched latent dimension k in {n/16, n/8}, the best SPARSE(F <= 16n, k) has one-step mean nMSE no better than
  the better of DENSE(k) and DENSEQ(k) (for k <= 16), in at least half of the (width, layer, k) cells. Meaning: on random
  nets the sparse latent carries no more higher-order structure than the dense collapse-mode latent.
- **KT3 (one step vs the Gaussian closure; brief's test 3).** arc, sigma = 0.01, the same cells. Fires if the best
  SPARSE(F <= 16n, k <= n/4) has one-step mean nMSE >= G's in at least half of the cells. Meaning: one-step
  propagation is no better than the Gaussian closure.
- **KT4 (representation vs K3f at depth 32; M1).** arc, sigma = 0.01, n = 256. Fires if no SPARSE(F <= 16n, k <= n/4)
  at any cut m in {8, 16, 24} has a representation error at L = 32 below ARC's K3f end-to-end nMSE at 32. Then no
  propagation rule can make this family beat K3f there: a family cannot be propagated more accurately than it
  represents the state.
- **KT5 (transcoder chain; brief's test 3, multi-step).** Fires if the feature-level Gaussian closure through the
  transcoders is worse than neuron K=2 at depth 8 (n = 64 and 256, sigma = 0.01), for every transcoder setting.

## Predictions (written before any main result)
- P1 (E1).
  - arc: KT1 fires at layers 1-16 at every width; at layers 24-31 the best SAE at L0 = n/4 is within 3x of PCA-(n/4),
    and is below the noise only where PCA-(n/4) is.
  - Error falls slowly with F at fixed k: error(16n) / error(n) > 0.5 at L0 = n/8.
  - tcha: KT1 fires everywhere (>= 100 noise units at L0 = n/4).
  - tord: below the noise with the mean alone from layer 8 on; KT1 fires at layers 1-2.
- P2 (E2, arc).
  - One-step mean nMSE of G between 1e-4 and 1e-3 at layers 8-31.
  - The latent-conditional models with k = n/8 capture >= 80% of kappa3 at layers >= 8.
  - KT2 fires: sparse is no better than dense at matched k.
  - KT3 does not fire: SPARSE at k >= n/8 beats G by >= 2x.
- P3 (E3, n = 256).
  - G at cut 24 gives 2-3e-4 at depth 32. DENSEQ(16) and SPARSE(k = 32) at cut 24 give about 5e-5, comparable to each
    other.
  - KT4 does not fire. Sparse is not better than dense.
- P4 (residual).
  - Median marginal excess kurtosis of the residual > 0.5.
  - After conditional-variance normalization < 0.3.
  - Projected by W_{l+1}: median < 0.1.
- P5 (E4).
  - Path KL > 100 bits per layer at L0 <= n/4 for layers <= 16.
  - KT5 fires: the transcoder closure is worse than neuron K=2 at depth 8.

## Compute and rules
- CPU jobs run under `nice -n 19 ionice -c3`, at most 3 processes.
- E1-E3 run on one on-demand L4 (`launch_gpu.sh`, cap $10, GPU-LOG.md). General is at load ~50 and trains a width-64
  F = 4n SAE at about 1-2 GFLOP/s per process.
- A GPT-6 critique of this design (codex-any exec) runs alongside the first GPU job. Changes it causes are dated
  amendments below, with reruns where needed.
- A GPT-6 review comes before any claim.
- Amendments are appended with times and hashes in `prereg_hashes.txt`.

## Amendment 1 (16:42Z, before any E4 run and before any E1-E3 result)
- **Leak in E4 as first written.** The transcoders are fitted on real paths of the standard input law, so their output
  biases carry the real per-layer means. A badly fitting chain can still return E[a_L] of that law to about 1/N_train
  (smoke run: a one-epoch chain, thousands of noise units per layer, matched the std-law mean to sampling noise).
  On the training law, "chain against truth" therefore measures memorisation, not description.
- **Change.** E4 also runs everything from two input laws the transcoders never saw (`lib/nets.input_law`):
  - shift: N(0.5 g, I), with g ~ N(0, I) fixed per net;
  - aniso: N(0, Q diag(lam) Q^T), lam_i proportional to 1/i, trace n, Q random.
  - The truths are 2^24-path Monte Carlo per law.
  - Neuron K=2 (our closure with identity transcoders, checked against ARC's K=2) and ARC's K=2 and K3f
    (`ext/kprop_noisy_law.py`, a copy of arc-noisy's wrapper with an input-law argument) are run from the same input
    moments.
  - Reference for memorisation: the std-law truth used as the estimate for the other laws.
  - Also reported: the closure against its own chain's sampled mean, to separate closure error from replacement error.
- **KT5 restated.** Fires if, for every transcoder setting, the feature-level closure through the chain is worse than
  neuron K=2 at depth 8 under the shift or the aniso law (n = 64 and 256, sigma = 0.01). The std-law comparison is
  reported but not used.
- **P5 unchanged.** Added: on the shift and aniso laws the chain is no better than the std-law mean used as an estimate
  (memorisation) wherever its path KL exceeds 1,000 bits.

## Amendment 2 (16:51Z, after the GPT-6 design critique)
Disclosure: while monitoring the v1 runs, a few v1 log lines on eval seed 65 were seen before this amendment.
- tord n = 64: E1 at layers 1-2, E2 at layer 1.
- arc n = 64: E1 at layers 1-4, E2 at layers 1-2.
- These lines include SAE noise units of 5-1,500 and one-step nMSEs.
None of the rules below were set with those numbers in mind beyond what is stated.
The critique is in `review/design_critique.md` (18 findings, 3 critical). The code is `gpu/run_main2.py` (v2) with
`lib/sae.py` (refit fix and new options). v1 (`gpu/run_main_v1.py`) keeps running on arc sigma = 0.01 (run sc-runa2)
as a consistency check of E1 only. Run sc-runb (v1, tanh) is being stopped (16:48-16:53Z).

Scope (critique 1-2, 13):
- E1-E3 are screening diagnostics with oracle access: true latents, and moments fitted on true states.
- KT4 is a screening threshold for fitted oracle representations. It is not a necessary condition for every
  propagation rule.

Changes:
1. KT1 asks for compression beyond the neuron code (critique 3, 4).
   - The exact neuron code has L0 = #active, about n/2. Its support distribution is reported.
   - Zero-origin TopK SAEs (b_dec fixed at 0, the first n atoms initialised at the neuron basis; F = n and 4n) are part
     of the arc grid. They enter E2 and E3 as SPARSEZ. KT1 uses the best of both origins.
2. Convergence (critique 5).
   - Held-out learning curves at 10 checkpoints for every SAE.
   - Controls at arc layers 8 and 24, for both origins at F = 4n: same seed with 2x epochs, and a different-seed
     restart.
   - A KT1 cell whose best error is 1-3 noise units is "near threshold". It is settled by the controls, not by one run.
3. SPARSE is named for what it is (critique 6): an affine-mean, homoscedastic Gaussian given the reconstruction. The
   held-out covariance match is reported.
4. Matched size (critique 7-8).
   - The equal-parameter pairing is removed: no dictionary has F <= n/2.
   - KT2 compares at equal per-sample active dimension only, and is read narrowly as "TopK-SAE latent vs PCA latent".
   - Support entropy of each sparse latent is reported: the sum of per-feature binary entropies, and log2 C(F, k).
5. Sample splits (critique 9).
   - E2 fits everything (PCA, regressions, residual covariances) on FIT (2^18), selects on SEL (2^17) and decides on
     CONF (2^17).
   - E1's PCA is fitted on the SAE training set and scored on held-out data.
6. E2 readouts and errors (critique 10-11).
   - Given the latent, a model's next pre-activation is Gaussian, so E[act(z')] is integrated per state.
   - kappa3 and kappa4 of z' come from the latent part. The norms of their denominators and the absolute errors are
     reported.
   - nMSE is debiased with 8 blocks and given a jackknife SE.
   - TRUE_indep and the Edgeworth oracles are dropped from v2. v1 has them, as diagnostics only.
7. Residual diagnostics, all held out (critique 12): signed and absolute skewness and excess kurtosis, marginal and
   after a conditional-variance regression fitted on FIT, with that regression's held-out R^2. DENSE and DENSEQ get
   the same diagnostics.
8. E3 (critique 14): common random numbers across models for the residual draw and for every downstream noise. Each
   model is debiased with its own output variance.
9. tanh (critique 17): the post-LN next-state mean is added, sampled with common random numbers.
10. Refit and units (critique 16).
    - The least-squares refit uses only atoms with a positive TopK activation (signed coefficients).
    - "Bits" are the Gaussian-channel KL under the coupling a -> a_hat, not a code length.

Kill rules made explicit (critique 15):
- **KT1, per (family, width, layer).** Fires if no SAE (either origin, F <= 16n, L0 <= n/4) reaches noise units < 1,
  raw or refit, on held-out data.
- **KT2.** arc, sigma = 0.01, seed 65; 24 cells (3 widths x layers 8, 16, 24, 31 x k in {n/16, n/8}).
  - Per cell, "sparse" is the best SPARSE/SPARSEZ over F, and "dense" the better of DENSE(k) and DENSEQ(k); both are
    selected on SEL.
  - Sparse wins a cell if, on CONF, nMSE_sparse + 2 SE_sparse < nMSE_dense.
  - Fires if sparse wins fewer than 12 cells. A missing cell counts as not won.
- **KT3.** The same nets; 12 cells (width x layer). "Sparse" is the best SPARSE/SPARSEZ over F and k <= n/4, selected
  on SEL. It wins a cell if, on CONF, nMSE + 2 SE < G's nMSE. Fires if it wins fewer than 6 cells.
- **KT4.** n = 256, seed 65. Fires if the minimum E3 error at depth 32, over cuts {8, 16, 24} and sparse models with
  k <= n/4 (both origins), is >= ARC's K3f end-to-end nMSE on the same net against the same truth.
- **KT5.** As in amendment 1. Fires if at both n = 64 and n = 256, no transcoder setting beats our neuron K=2 at
  depth 8 under both the shift and aniso laws.
- **Replication.** arc, sigma = 0.01, seed 66, n = 64 (E1-E3). Statements about the family are made only where seeds
  65 and 66 agree.

Predictions: unchanged. P2's "KT2 fires, KT3 does not" now refers to the rules above.

## Amendment 3 (16:56Z): E4 rerun
- Run sc-runc's feature-level closure returned NaN. Thresholds of never-firing features (1e9) overflowed the
  Hermite recursion. Fix: t is clamped to +-38, where phi underflows to 0 and Q is 0 or 1. The coefficients were
  checked against Monte Carlo.
- Seen before this amendment (sc-runc, n = 64, before it was stopped):
  - path KL of the chain: 2.7e5 bits (k = n/8) and 9.0e4 bits (k = n/4), F = n;
  - chain-sampled nMSE at depths 8, 16, 32: 0.03-0.6 under all three laws;
  - identity closure (neuron K=2): 3.6e-3, 5.3e-3, 7.9e-3 (std).
- Changes for the rerun (sc-c2):
  - a k = n/2 setting is added;
  - transcoder weights are saved;
  - widths n = 64 and 256 only, for budget (KT5 is registered at these two).
- KT5 is unchanged and applies to every setting, including k = n/2.

## Amendment 4 (written before any E5 run except a CPU smoke test at tiny sizes on dev seed 64): E5, the dependence
## test in the covariance-carrying frame
Why: E2 and E3 give the sparse latent its exact joint law, taken from true states. That leaves Kelvin's question open:
do the features' couplings beyond the covariance matter? E5 (`gpu/run_e5.py`) replaces the exact latent law with
feature families that keep exact marginals and the full covariance and nothing more.
- Nets: arc, sigma = 0.01, seed 65, n = 64, 128 and 256; seed 66 at n = 64 for replication.
- Cuts 8, 16, 24. SAEs are trained in the script: F = n and 4n; origins mean and zero; k = n/16, n/8, n/4.
- **Models.** Every model except the _raw ones is recoloured to the exact mean and covariance of the state.
  - SPARSE: exact latent + v2 residual.
  - SPARSECOP: features from a Gaussian copula with exact zero-inflated marginals and the normal-scores correlation,
    + residual.
  - SPARSEIND: independent features with exact marginals, + residual.
  - NEUCOP: neurons with exact marginals + Gaussian copula.
  - G: Gaussian.
  - DENSE(k): as before.
  - _raw versions are not recoloured. SPARSEIND_raw is the independent-feature product (the analytic case of
    retarget 1, quoted only as a check).
- **Scores.**
  - One step: Rao-Blackwellised E[ReLU(z')], debiased with 8 blocks.
  - Representation at the cut, pushed through the real layers to depth 32 (and 16 from cut 8).
- **KT6 (dependence beyond covariance; fixed now).** For each (width, cut) cell at depth 32 (seed 65: 3 widths x
  3 cuts = 9 cells), compare SPARSECOP (best over F, origin and k <= n/4) with the exact-latent SPARSE (same
  selection). Each of these two is chosen by its depth-32 representation error. KT6 fires (the couplings beyond the
  covariance carry what matters) if, in at least 5 of the 9 cells:
  - SPARSECOP's error is more than 3x the exact latent's, and
  - SPARSECOP's error is not below G's by 2x.
- **Prediction.** KT6 fires. The TopK support constraint (exactly k features active) and the joint support pattern are
  not representable by a Gaussian copula over many sparse features. Further:
  - SPARSECOP lands within 2x of G, and NEUCOP within 2x of G, at depth 32.
  - SPARSEIND_raw is much worse than G (it drops the covariance).
- Addendum to amendment 4 (17:20Z, before any E5 result):
  - The first E5 attempt ran as extra processes on the sc-a3big VM. At n = 256 it hit CUDA out-of-memory (three
    processes on one GPU) and was stopped before any result.
  - E5 now runs on its own VM (sc-e5b), as two parallel processes.
  - At n = 256 the copula and residual fits use 2^16 FIT states instead of 2^17 (memory). The rules are unchanged.
- Addendum 2 to amendment 4 (written before any E5 result):
  - The capped E5 chain on sc-b2 exceeded its 4 GB cap and was stopped before any result.
  - The fits now use 2^16 FIT states at every width, codes are computed one k at a time, and quantile tables are
    built in blocks of 64 features. There is no logic change.

## Amendment 5 (18:10Z): E3 dense-latent bug
- In `gpu/run_main2.py` v2, the DENSE and DENSEQ closures read the variable `mu` when called, not when defined.
  E3 calls them after the layer loop, so every cut used layer 31's mean.
  - Seen at n = 256: DENSE errors of 0.3-1.9 at every cut, against 7.6e-5 to 2.0e-4 for G.
  - E1, E2 and E3's TRUE, G and SPARSE rows are not affected: E2 calls the closures inside their own layer, and the
    sparse closures bind their parameters as defaults.
- The dense rows of E3 are rerun with `gpu/run_e3dense.py`, which uses the same data streams, fits and common random
  numbers. Its TRUE and G rows must reproduce run_main2's, which checks the rerun.
- The analysis replaces E3's DENSE/DENSEQ rows with the rerun's. KT4 (sparse vs K3f) does not involve the dense rows.
- The fix (`mu` bound as a default argument) is in run_main2.py, so the n = 128 run is not affected.

## Amendment 6 (18:33Z): scope trims for time and budget, before their results
- Dropped: ordered tanh at n = 256 (it is skipped when it starts) and ARC sigma = 0.001 at n = 256 (stopped once the
  other jobs on the VM finish). Both are secondary.
- ARC sigma = 0.001 at n = 64 is kept (E1, E2).
- Ordered tanh at n = 64 failed earlier (GPU out of memory while sharing the VM) and is not rerun. The ordered-tanh
  statements rest on n = 128.
- Addendum to amendment 6 (18:46Z), written before these results:
  - ARC n = 128 is rerun as E1 + E2 only, without E3 and without controls. Two capped attempts ran out of memory.
    KT2 and KT3 need E2 only.
  - E5 at n = 256 completed cut 8 before running out of memory. Cuts 16 and 24 are rerun in a separate process.
    The analysis merges the cuts.
- Addendum 2 to amendment 6 (18:56Z): the n = 128 rerun hit its memory cap again. It runs last on the VM, without the
  16n dictionary (F <= 8n), as E1 + E2. KT2 and KT3 are already decided without n = 128 (14 and 8 wins against the 12
  and 6 needed).
