# Round-1 estimators: adversarial review (2026-10-28)

Reviewed: METHOD.md, VALIDATION.md, ROUND1.md, `cmlib/`, `tests/`, and the code that produced ROUND1 (`r1run.py`, `lib/r1est.py`, `lib/tp.py`, `r1fragility.py`). Reruns are in `review_estimators/` (CPU, at most 2 processes, about 10 minutes of compute). Checked at 2026-10-28 19:00–19:15Z.

**TL;DR.** The weight-free direct numbers stand: Δ and Δ_rms for q0 reproduce on fresh episodes within about 1.5 SE, and the G quadrature, the conditional sampler, the labels and the fit/A/B splits pass their checks. Three readings in ROUND1 fail when re-run: "half the gap is off-manifold noise" (a residual resampled from real residuals costs the same as the Gaussian one, so that half is missing structure), "4 G-relevant directions carry the gap" (the same 4 directions carry 97–99.8% of R_in's gap too, so this describes G's readout, not q0), and the L(α) shape (it moves with the classifier's training size). No decision is needed; §4 lists proposed edits, none made.

## 1. Which claims can the design answer?

| ROUND1 claim | Can the design answer it? | Rerun |
|---|---|---|
| q0 loses 0.08–0.39 nats at read sites (19–81× KL_model) | Yes: paired and weight-free | Reproduces (§3) |
| About half of Δ(q0) is off-manifold noise rather than missing structure; the Gaussian base is the wrong family where the noise share is ≥ 0.5 | No: the noise control cannot separate the residual's shape from its independence of x | Fails (§2.1) |
| 4 G-relevant directions per token carry 92–98% of q0's gap | No: there is no readout control | True of any q, so it does not locate what q0 misses (§2.2) |
| The bridge is accurate at AUC ≥ 0.9; most of the L(α) drop happens by α = 0.2 | Endpoints only, and they reuse the direct draws; the interior is not identified | Endpoints fine, interior fails (§2.3) |
| Block 1's description pushed through the real block 2 beats q0 fitted at h2 | Partly: the comparison mixes rungs | Holds in 4/5 seeds with matched rungs (§2.4) |

## 2. Problems, most severe first

### 2.1 The fragility control does not show off-manifold extrapolation

Affects ROUND1's TL;DR bullet 2, §6, §10.1 and round-2 plan (a), and METHOD §4.2.

ROUND1's own fragility table fits a second-order (locally quadratic) response. Δ(q0 with noise sd halved) equals Δ(mean only) + ¼·Δ(real + noise) within ±5.4% in all 12 rows, and Δ(q0) equals Δ(mean only) + Δ(real + noise) within −3% to +15%. Under a second-order response, a zero-mean residual drawn without regard to x costs ½·tr(HΣ) whatever its distribution (H is the Hessian of G in the token state); only its covariance Σ enters. A Gaussian residual and a residual resampled from real residuals with the same Σ should then cost the same.

They do. `e124_real.py` keeps q0's mean and draws each token's residual from the fit split's own residuals (independently per token, as q0 does). It also compares real states plus Gaussian noise with real states plus the difference of two real residuals divided by √2 (same covariance). Fresh episodes, seed 63500; SEs of the two noise columns are 0.001–0.004.

| net, site | noise share | Δ q0 (Gaussian residual) | Δ real-residual base | Δ real + Gaussian noise | Δ real + residual difference | Δ mean only |
|---|---|---|---|---|---|---|
| 2L_s0 h1-ALL | 0.34 | 0.408 ± 0.009 | 0.417 ± 0.009 | 0.140 | 0.138 | 0.318 |
| 2L_s0 h2-QT | 0.47 | 0.184 ± 0.007 | 0.190 ± 0.007 | 0.087 | 0.087 | 0.086 |
| 3L_s11 h2-ALL | 0.71 | 0.376 ± 0.006 | 0.382 ± 0.007 | 0.265 | 0.256 | 0.157 |
| 3L_s11 h3-QT | 0.51 | 0.092 ± 0.006 | 0.098 ± 0.006 | 0.047 | 0.046 | 0.041 |

The real-residual base is no better than the Gaussian at any site, including 3L_s11 h2-ALL, the site with noise share 0.71 where ROUND1 calls the Gaussian base the wrong family. The noise cost comes from how much variance q0 leaves unexplained around its linear mean (Σ includes the mean misfit) and draws independently of x; reducing it needs a better conditional mean, which is more structure. A share near 0.5 is what one gets when the actual residual costs about as much as a random draw with the same covariance, so METHOD's 0.5 threshold marks the generic case and detects nothing. The fragility numbers themselves reproduce (noise share 0.34 and 0.71 against ROUND1's 0.34 and 0.70).

For round 2: q^R (resampling real states within feature bins) can only help through its bins, which make the state depend on φ nonparametrically. Run it with the real-residual base above as the control and read any gain as structure.

### 2.2 The 4-d subspace result measures where G reads, not what q0 misses

Affects ROUND1's TL;DR bullet "Where the gap sits" and the §4 subspace table.

Uout comes from the gradient of the output-space log ŵ, which depends on h only through the 12 head outputs, so it lies in directions the downstream reads; ROUND1 has no control for that. `e5_subspace.py` (2L_s0, 3,000 fresh cells, seed 64500) uses U_G, the top-4 eigenvectors of E_p[∇G ∇Gᵀ] on real draws, built with no classifier and no q. G's sensitivity is concentrated: the top 4 eigenvalues hold 81% of the trace at h1-QQ and 99% at h2-QT, the first one alone 85%.

Fraction of each rung's own Δ left when a 4-d subspace is kept real (knock-in, as in `r1est.subspace_knock`):

| site | q0, U_G | q0, Uout (ROUND1, eval A) | q0, random 4-d (mine / ROUND1) | R_in, U_G |
|---|---|---|---|---|
| 2L_s0 h1-QQ | 0.032 | 0.048 | 0.24 / 0.24 | 0.028 (Δ(R_in) = 0.47) |
| 2L_s0 h2-QT | 0.036 | 0.126 | 0.55 / 0.56 | 0.002 (Δ(R_in) = 4.93) |

The gradient of G alone does at least as well as Uout. Keeping the same 4 directions real also removes 97–99.8% of the gap of R_in, a rung with no Bayes latents. So "4 directions carry 97–98% of q0's gap" would hold for any q and says nothing about where q0's missing structure is. The naming loop still worked as a heuristic (its picks passed acceptance on fresh B), and U_G would have served as well. The component result (the MLP output carries 98–100%), the other half of "two localisations agree", has no readout control either; I did not test it.

### 2.3 The interior of L(α) is not identified; T1, V5 and "the bridge is accurate" check little

Affects ROUND1 §3 (the L(α) table and "The L(α) shape") and the TL;DR bullet on Kelvin's estimators.

At AUC ≥ 0.9 the per-cell bridge reproduces the synthetic and real means at α = 0 and 1 for any classifier that ranks the classes well, because the weight 1/(1 + ŵ) is near 1 on synthetic draws and near 0 on real ones. bridge/direct then follows the AUC (0.94 at AUC 0.90, 0.99 at 0.98) and adds nothing to the direct Δ computed from the same draws. V5 adds nothing statistical either: dL/dα = Cov_ω(G, log ŵ) is the exact derivative of the self-normalised curve, so ∫dL/dα differs from L̂(1) − L̂(0) only by trapezoid error, and its 46/64 pass rate measures the α grid.

The interior depends on log ŵ values the data do not pin down. In the rerun, out-of-fold log ŵ on synthetic draws reaches −36 to −47 at the 1% quantile and −83 to −101 at the minimum (the three site-nets with AUC ≥ 0.92). The logistic loss has gradient below e^−10 there, so these values are extrapolated, not fitted; yet at α = 0.1 a synthetic draw at −47 gets 67 times less weight than one at −5. In ROUND1, E_p[1/ŵ] at 2L_s0 is 2.7e4 to 2.6e12 at every site (target 1); T2 fails at every high-AUC site, and ROUND1 does not say by how much.

Two reruns show the interior moving while the endpoints pass:
- Known answer (`e3_bridge.py`): p = N(0, I₈), q = p shifted by 2.8 along one axis, G = h₁², AUC 0.976, per-cell bridge with 8 + 8 draws as in `r1est._bridge_cells`. With the true log w the curve is right (interior error ≤ 0.12 on a range of 7.8). With 2·log w or sign(log w)·|log w|^1.5 (same ranking, same AUC) the gap stays at 102–104% of the direct gap and T1 passes, but L(0.2) is 5.1 or 4.4 against an exact 6.0, and the share of the drop done by α = 0.2 rises from 36% to 47–56%.
- Real nets (`e124_real.py`): same nets, sites and q0 family, classifier trained on 3,000 (2L) or 2,000 (3L) cells instead of 6,000. The share of the drop done by α = 0.2 is 0.53 instead of ROUND1's 0.79 (2L_s0 h1-ALL), 0.73 vs 0.82 (2L_s0 h2-QT), 0.58 vs 0.88 (3L_s11 h2-ALL) and 0.43 vs 0.65 (3L_s11 h3-QT). Multiplying log ŵ by 0.5 or 2 moves bridge/direct between 0.66–0.84 and 0.87–1.02, and the drop share by 0.05–0.10.

So "most of the drop happens by α = 0.2, which fits a gap concentrated in a few directions" is a property of the classifier (the few-directions part is true for the reason in §2.2). With the true ratio, per-cell self-normalisation bias is small (≤ 1.5% of the range at M = 8), and the bridge's per-cell ESS stays ≥ 5.2 of 16 at every round-1 site, so weight degeneracy is not the problem; the ratio's scale is.

### 2.4 Tier ii mixes rungs

Δ(F#q_l) at h2 equals Δ(q_l) at h1 exactly (tier i, max |ΔG| = 0), and q_l was q1 while q^fit was q0. "0.02–0.10 vs 0.16–0.52" therefore compares q1 at h1 with q0 at h2. With q0 at both cuts (eval A, 3L, seeds 10–14): h1-ALL 0.093, 0.389, 0.090, 0.082, 0.103 against h2-ALL 0.168, 0.372, 0.525, 0.284, 0.441. The ordering holds in 4/5 seeds and s11 reverses. Quote matched rungs.

### 2.5 Confidence intervals

These matter for marginal claims, not for the headline gaps.
- Per-seed Hill indices are 1.1–1.8 at many sites (3L h1-QQ, h1-ALL, a3-QT, h3-QT; 2L h1-QT, h2-QT), so the per-cell differences have infinite variance there. METHOD §3.1 says to double n when Hill < 3; round 1 did not. VALIDATION found 0.90 coverage for a lighter tail. Use the seed-level t₄ interval, as ROUND1 §11 itself suggests, and drop within-seed ± values at those sites.
- Ratio-based ± values combine the ratio-fixed cell bootstrap (VALIDATION: 42–58% coverage) with an SD over 2 refits that differ only in fold split and seed (1 degree of freedom, no data resampling). VALIDATION's half-sampling fix was not used. Consistency Γ SEs come from 3 refits (2 df), so "within 2 SE in 4/5" is not reliable.
- VALIDATION's coverage figures come from 12–40 replicates (binomial SE 0.035–0.08); 11/12 covered has a 95% interval of 0.62–1.00.
- No test or known-answer run covers the estimators round 1 used. VALIDATION checks `cmlib.estimators` with a few discrete groups and thousands of draws per group, and with estimated ratios only at AUC ≤ 0.87; round 1 used `lib/r1est` (per-cell bridge over continuous cells with 8 + 8 draws, cross-fit by cell), mostly at AUC 0.90–0.99. `e3_bridge.py` now covers the per-cell bridge with a known ratio.

### 2.6 Classifier split leakage and the sign of T5

`fit_ratio` splits draws, not cells, into train/cal/test, so its calibration and test splits share cells (and real/synthetic pairs) with training. In the rerun the within-fold test BCE is 0.006–0.026 nats below the out-of-fold BCE. ROUND1's AUC, BCE and KL are out-of-fold by cell and are fine. The Platt slope is fitted on the leaky split; refitted out of fold, the slope is 0.85–0.93, so log ŵ is about 10% overconfident on new cells. `train_minus_test_bce` is train minus test, which is negative under overfitting, so METHOD's T5 rule "< 0.01" passes any overfit classifier; the observed −0.014 to −0.055 would fail with the sign fixed. Round 1 does not use T5, so this only needs fixing before something does.

### 2.7 Multiple comparisons and post-hoc choices

The cross_o/cross_p and meansq sub-splits (Amendment 2) were chosen after seeing results and scored on B, the acceptance split. The motif claim rests on 3 one-pool against 2 two-pool seeds; the smallest exact permutation p-value for a 3/2 split is 0.1, and these are the 5 nets B2 used, so it is agreement between two methods rather than replication. The within-seed effects are large (cross_o: 0.48 → 0.34, 0.59 → 0.35, 0.48 → 0.34 in one-pool seeds; 0.76 → 0.74, 0.90 → 0.90 in two-pool seeds), so I expect them to hold. P5 ("ratio < 1 at 6/6 sites") has probability 1/64 under symmetric noise, and ROUND1 §3 places the attenuation at AUC 0.6–0.75 while P5 is defined below 0.6. The h-space AUC of 0.999 quoted in the TL;DR is a density chasm (METHOD T6) and supports no particular reading.

## 3. What survives, and the check behind it

- Direct Δ and Δ_rms of every rung. On fresh episodes (seeds 63500 and 64500): 2L_s0 h1-ALL 0.408 ± 0.009 (ROUND1 eval A 0.403), h1-QQ 0.325 ± 0.007 (0.323), h2-QT 0.184 ± 0.007 (0.198), 3L_s11 h2-ALL 0.376 ± 0.006 (0.372), h3-QT 0.092 ± 0.006 (0.095). The effects are 10–100 times their SE, so the heavy tails do not threaten them.
- G quadrature on synthetic outputs, which METHOD §2.4 required and round 1 skipped: 201 and 2,001 grid points over the same Bayes ±9 sd range agree to ≤ 2.2e-12 nats at 4 site-nets (6,400 draws each). No synthetic output sits at the lower log-sd clip; real outputs agree to 6e-16.
- The conditional sampler is exact for this world. With y ~ N(0,1) and e ~ N(0, v_s) iid (world.py), the labelled y, the residual directions and the IDs are independent of v given (Q, S). No feature reads v or y* (v is only carried along; nll_bayes is stored but unused). G's target p_B(·|c) stays valid because every draw keeps c. The halves j < 4 and j ≥ 4 are independent given c, so the Δ_rms² estimator is unbiased.
- The F/A/B separation of fitting, selection and acceptance, so q1's acceptance (16/16 at full cuts) stands.
- q0 is far from closing; the Bayes latents cut Δ from about 5 nats to 0.1–0.4; q0noID ≈ q0.
- Kelvin's q-side identity and q-side tempering are unusable here (ESS 0.01–1% of draws), as ROUND1 says.
- Region shares by decile of log ŵ_out and by episode quintile: selecting and estimating on the same draws or on disjoint halves differ by at most 0.011 (≤ 1.6 SE) at 4 site-nets, so there is no material selection bias.
- The fragility measurements reproduce; only their reading fails (§2.1).

## 4. Proposed edits (not made)

1. ROUND1 TL;DR bullet 2, §6 and §10.1: call the noise term the cost of residual variance drawn without regard to x, which is missing structure. Reframe round-2 (a) as a nonparametric conditional mean with the real-residual base as its control. Drop the 0.5 rule in METHOD §4.2.
2. Add U_G (from ∇G) as the readout baseline to every subspace or component localisation, and claim only what beats it and what a weaker rung such as R_in does not also show.
3. Drop the L(α) interior and shape claims, V5, and "the bridge is accurate". If L(α) is wanted, show that it is stable as the classifier's training set grows and under logit rescaling, or telescope as METHOD §3.3 says for AUC near 1.
4. Fix the T5 sign, split `fit_ratio` by cell, and report the out-of-fold calibration slope and the E_p[1/ŵ] values next to T2.
5. Use seed-level t₄ intervals; at Hill < 2 sites apply METHOD's rule or add a trimmed-mean cross-check.
6. Add known-answer tests for `lib/r1est` at AUC 0.95–0.99.
7. Quote tier ii with matched rungs.

## Reproduce

```
cd /home/agent/workspaces/mats/work/cm-interp/review_estimators
PY=/home/agent/.venvs/aggext/bin/python; export OMP_NUM_THREADS=1
$PY e3_bridge.py                     # §2.3 known answer; seconds; prints (saved copy: e3_bridge.out.txt)
$PY e124_real.py 2L_s0 3000          # §2.1, §2.3, §2.6, G-grid check, region shares; about 4 min
$PY e124_real.py 3L_s11 2000         # same for 3L_s11; about 3 min
$PY e5_subspace.py 2L_s0 3000        # §2.2 with q0; about 1 min
$PY e5b_subspace.py 2L_s0 3000 Rin   # §2.2 with R_in (in its output, key "q0" is the rung named on the command line)
```

Outputs: `e124_*.json`, `e5_*.json`, `e3_bridge.out.txt` and `log_*.txt` in the same folder.
