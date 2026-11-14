# Adversarial review of the round-1 interpretation (2026-10-28)

Scope: does ROUND1.md mean what it says? I read METHOD.md, PREREG.md, VALIDATION.md, ROUND1.md, the round-1 code
(`r1run.py`, `lib/tp.py`, `lib/r1est.py`, `r1fragility.py`, `r1posthoc.py`, `r1summ.py`) and the tiny-PFN reports in
`outbox/` (tiny-pfn-2026-10-27, tiny-pfn-interp-step1 with B1/B2/heads/L3, tiny-pfn-plateau and its historical audit).
Every number below was re-read from `r1/*.json` (checked 2026-10-28 ~19:20Z), except those marked **spot check**: two
small reviewer runs on the round-1 fit split and eval B. Their scripts and outputs are in `review-interp/`. No
round-1 file was changed.

**TL;DR**
- **Survives:**
  - q0 and q1 close no read-heavy site. This rests on the validated direct estimator, fresh data and every seed.
  - In one-pool seeds the query-question (QQ) tokens carry other sources' readings.
  - Kelvin's q-side identity cannot be used at these separations.
  - In a nested test the reviewer ran, the plateau net's QQ states gain almost nothing from source-bound features once
    unordered content is present (one net, linear-Gaussian family).
- **Does not survive as stated:**
  - Most of "structure beyond q" is q's functional form in variables q already reads.
  - The localisations measure redundancy under Gaussian imputation, not location.
  - "The computation beats a direct description" restates the tier-i identity.
  - The report's own q0U-vs-q0 evidence does not isolate binding.
  - Shrinking q's noise counts as progress under both the acceptance rule and the description-length ledger.
- **Most informative next experiment:** an information-matched, on-manifold oracle. Replace a site's tokens by the
  network's real states on a fresh episode drawn from the world given exactly what q may read. It splits each gap into
  form, missing variables and binding, with no fitted family and no off-manifold states.

## 0. Can the design answer the claim?

The claim is that L(0) − L(1) = Δ(q) measures structure in layer l beyond q. To read it that way, a rung must change
what q knows without changing how q may use it. Every round-1 rung does both: new features enter a linear-Gaussian
map. Every q sample also lies off the network's manifold (h-space AUC ≈ 0.999). A large Δ can therefore come from four
sources:
- (a) variables q lacks;
- (b) nonlinear dependence on variables q has;
- (c) the wrong conditional shape;
- (d) how the network responds off its manifold.

Only (a) is new structure in the mechanistic sense, and round 1 has no condition that isolates it. Problems P1–P3
follow from this.

## 1. Ranked problems

### P1. "Structure beyond q" is mostly q's form, not missing variables (high; the central claim)
ROUND1 names two structures and accepts q1 at every full cut (TL;DR, §5).
- **q0's linear mean drives the closure metric.** With no noise, q0's mean alone keeps 90–100% of q0's Δ_rms at 2L
  h1-ALL and h2-QT in every seed (s0 h1-ALL 0.626 vs 0.626; s1 h2-QT 0.490 vs 0.531; `fragility_*.json`).
- **Structure (ii) adds no information.** It says the query token depends roughly quadratically on the predictive mean
  and the latents. But `quadlat` and `meansq` at QT are products of q0's own QT features (`lib/tp.py` 199–201 and
  218–225, against 172–174).
- **Structure (ii) is also not specific.** On fresh B at the last QT cut of the near-Bayes nets, quadlat cuts Δ_rms by
  74–88%, betaQ by 69–85% and meansq by 63–82% (`posthoc_*.json`). Any nonlinear expansion of the same latents works.
- **Spot check: LayerNorm gauge.** At the last cut the head reads LN(h), so the state's scale is a gauge there. Round 1
  still fits centred but unscaled states (`needs_norm` scales only context tokens at the second-to-last cut). Refitting
  the same q0 in the LN gauge lowers the gaps as follows:

  | net (cut) | q0 as in round 1: Δ / Δ_rms | q0 in LN gauge: Δ / Δ_rms | drop | q0 + quadlat in LN gauge: Δ_rms |
  |---|---|---|---|---|
  | 2L s0 (h2-QT) | 0.185 / 0.484 | 0.099 / 0.365 | 46% / 25% | 0.111 |
  | 2L s1 (h2-QT) | 0.244 / 0.534 | 0.107 / 0.397 | 56% / 26% | 0.080 |
  | 3L s10 (h3-QT) | 0.101 / 0.326 | 0.060 / 0.268 | 41% / 18% | 0.051 |
  | 3L s13 (h3-QT) | 0.068 / 0.256 | 0.056 / 0.241 | 17% / 6% | 0.067 |
  | plateau (h2-QT) | 0.197 / 0.390 | 0.153 / 0.335 | 22% / 14% | 0.188 |

  So 17–56% of the signed final-cut gap is a choice of coordinates. The nonlinearity itself is not a gauge effect:
  quadlat still removes most of what is left. It is a real nonlinear function of variables q0 already has. It is not a
  new variable.
- **Structure (i) is new information, and its motif split is what makes it mechanism.** Other sources' readings at QQ
  (`cross_o`) are the only named group that adds variables. They cut Δ_rms by 30%, 42% and 30% in the one-pool seeds
  s0, s2 and s3, against 2% and −1% in the two-pool seeds s1 and s4. That split separates (i) from curve-fitting. (The
  report's "35–48%" is the full `cross` group; `cross_o` alone gives 30–42%.)
- **Consequence.** After one round, 100–470× ε is left. Nothing in round 1 says whether that remainder is missing
  variables or form, and adding more named features to a linear-Gaussian cannot say either.

### P2. Localisation measures redundancy under Gaussian imputation, not where the gap lives (high)
ROUND1 claims that "everything q0 misses is in the MLP's write", that "4 output-relevant directions carry 97–98%" and
that "two localisations agree" (TL;DR, §4).
- **Only knock-in was reported.** The knock-out runs are in the same JSON (`localisation_A`):
  - **Subspace.** Drawing the 4-d Uout subspace from q given the real complement leaves 0–23% of Δ (mostly ≤ 10%).
    Random 4-d subspaces and Uh leave ≤ 7% (mostly 0). Four real read directions suffice for linear imputation, but
    they are not necessary.
  - **Heads.** At the first-block cuts, knocking out either attention head given the rest costs ≤ 0.004 nats. The
    "carried" shares add up to about 1.9 (s0: 0.20 + 0.48 + 0.19 + 0.99).
  - **MLP.** Drawing the MLP output from q, given its real inputs, leaves very different shares by net:

    | net | `cross_o` at h1-QQ | Δ left by MLP knock-out |
    |---|---|---|
    | 2L s0, s2, s3 (one pool) | −30% to −42% | 11–20% |
    | 3L s11 | −73% | 19% |
    | plateau | −35% | 6% |
    | 2L s1, s4 (two pools) | ≈ 0 | 60–64% |
    | 3L s10, s12, s13, s14 | ≈ 0 | 58–96% |
- **The split is clean in 11 of 11 nets.** It is the reading the data support (found post hoc by the reviewer):
  - where layer-1 attention brings other sources' content into QQ, q0's gap enters through attention;
  - where it does not, the gap arises in the MLP.

  "Everything is in the MLP's write" holds only in the second group.
- **"4 directions" largely reflects the readout.** They are directions weighted by the Jacobian of a 12-output head. No
  control used the top-4 read directions without the ratio.
- **The 2L h1 component split is distorted by scaling.** There every component is divided by the whole state's sd
  (`r1run.py: comp_q`), so even the embedding stops being linear in φ.
  - In that scaling the embedding "carries" 0.20–0.36 at 2L h1.
  - Unscaled, at 3L h1, it carries 0.00–0.02.
  - V1 shows the embedding is exact (residual 1e-6).

### P3. Tier ii does not test computation, and "computation beats a direct description" is the tier-i identity (high)
ROUND1's claim (TL;DR, §7): in 3L, the block-1 description pushed through the real block 2 (Δ 0.02–0.10) beats q0
fitted at block 2 (0.16–0.52), so "the family at l+1 is the bottleneck, not the computation".
- **The two numbers are equal by construction.** Δ(F#q_l) matches Δ(q1 at layer l, ALL, eval B) to 3 decimals in all
  16 pairs (3L s10 0.023 vs 0.023; s11 0.104 vs 0.105). The quoted contrast is therefore q1 at h1 against q0 at h2: two
  different rungs at two different cuts. It shows only that the 3L h1 cut is easier to describe. That fits the L3
  report, which finds the per-source code completed in block 2.
- **The other 11 pairs go the opposite way, and the report does not mention them.** The description fitted directly at
  l+1 beats the pushforward:
  - 2L h1→h2: 0.03–0.05 direct against 0.16–0.23 pushed;
  - 3L h2→h3: 0.01–0.02 against 0.05–0.31;
  - plateau: 0.08 against 0.19.
- **The real block carries no hypothesis.** Tier ii only measures how far q_l is from p_l, seen through block l+1 and
  the family at l+1. P7 predicted failure because q_l is not closed, so its "hold" carries no information. Only tier
  iii (a short surrogate F̃ in place of the block) tests computation.
- **The 3L h1→h2 rule passes are weak.** There Γ_fit lies within 2 SE in 4/5 seeds while AUC is 0.71–0.98.
  VALIDATION (iii) showed that the one-sided Γ rule passes 20/20 known-inconsistent cases in exactly this
  configuration.

### P4. The stuck-vs-escaped evidence does not isolate binding; a nested test does, and it supports the claim (high)
ROUND1 claims that the binding-free q0U describes the plateau net better and that this "matches astra's weak-binding
result from another angle" (TL;DR, §8).
- **q0U is not "q0 minus binding".** It drops the own-source posteriors and β_s. It adds the sorted log SS of all three
  sources, mean Q and the blind reference's predictive moments (`tp.py` 175–178), which are cross-source and nonlinear
  content.
- **The same sign shows up in nets that bind well.** These nets prefer q0U at QQ too:
  - 2L one-pool s0, s2, s3: Δ_rms ratio 0.93, 0.83, 0.93;
  - 3L h2-QQ: 0.60–0.77 in 4/5 seeds.

  astra's identity-cycle test gives these nets 82–99% of the Bayes response.
- **The plateau gains from bound cross-source features too.** Bound `cross_o` cuts the plateau's QQ Δ_rms by 35%,
  as much as in the one-pool seeds.
- **At QT the comparison is circular.** Comparing q0 with q0U there pits the Bayes mean against the blind reference's
  mean as regressors. It measures which reference the output is closer to, which astra already measured at the output.
- **Spot check: nested feature sets at h1-QQ** (eval B, Δ_rms, same LinearGaussian family):

  | net | q0 | q0U | q0 ∪ q0U | bound own-source latents added to q0U | q0U's extras added to q0 |
  |---|---|---|---|---|---|
  | 2L s0 / s2 / s3 (one pool) | 0.48 / 0.60 / 0.48 | 0.45 / 0.48 / 0.44 | 0.27 / 0.27 / 0.26 | −39% / −44% / −41% | −43% / −55% / −46% |
  | 2L s1 / s4 (two pools) | 0.76 / 0.90 | 0.98 / 1.17 | 0.73 / 0.86 | −25% / −27% | −3% / −4% |
  | plateau | 0.411 | 0.186 | 0.175 | **−6%** | −57% |

  - **The report's evidence came from q0U's extra content.** That content helps the plateau and the one-pool seeds
    alike.
  - **The nested increment is the binding test.** Once the unordered and query-average content is present, the plateau
    is the only net whose QQ states gain almost nothing from source-bound reliability.
  - The plateau claim therefore survives in this narrower form, but it rests on one stuck net. That net was trained
    differently (cold-start mixture, not warm start), and its KL is 15× larger, so ε and Δ/KL cannot be compared
    across the contrast.

### P5. The objective and the description-length ledger can be won without describing anything (medium-high)
- **Shrinking q's noise counts as progress.**
  - Halving q0's noise sd lowers Δ_rms at 3L h2-ALL by 7%, 26%, 15%, 25% and 52% (s10–s14), with no feature or
    parameter added. Under the PREREG acceptance rule's ≥ 10% threshold, that passes in 4/5 seeds. (This diagnostic did
    not record SEs.)
  - q0's mean with no noise is not a density at all. Yet it has a lower signed Δ than q0 at all 44 fragility
    site-nets, and its Δ_rms is no worse than q0's by more than 0.003 at 43/44.
  - A frontier on (DL, Δ_rms) rewards a concentrated q.
- **"Off-manifold noise, not missing structure" (TL;DR, §6) is a false choice.** A Gaussian with the right covariance,
  placed off the manifold, is exactly q missing the manifold's shape.
  - The control adds that noise on top of the real states, which roughly doubles the residual variance. It measures the
    network's sensitivity, not a part of Δ(q0).
  - The "noise share" is in signed Δ. In Δ_rms, the closure metric, the mean dominates at 2L (P1).
- **The network anchor is inflated.** METHOD §8 quantises the block by the same rule as q; round 1 counts it at fp16
  (137k bits). At 8 bits per parameter the block is 68k bits. q0 at a full cut is then 47–81% of the block and q1 is
  45–99%: no compression, at 20–470× ε.
- **The DL axis says nothing about the network yet.**
  - BIC for q0 is 40,901 bits in all 11 nets.
  - The quantised DL is mostly noise: b* ranges from 2 to 14 bits because the tolerance, 0.02·KL, is about 1/1000 of
    Δ.
- **Library features are free computation.** β_s comes from the 27-hypothesis posterior. At QT, the Bayes predictive
  mean and log variance (the network's target) cost about 5.6 bits each. METHOD requires the compute cost C(q) next to
  DL; round 1 reports none.

### P6. "Linear ID terms do nothing" is false as stated, and the crosstalk inference is below resolution (medium)
- **The claim fails in 3L.**
  - In 3L s12 at h1, dropping the linear ID terms multiplies Δ by 6.8 (LAB), 5.8 (QQ) and 3.0 (ALL).
  - In 3L s11 h1-LAB it multiplies Δ by 1.65.
  - ROUND1's own seed-mean table shows 3L h1-QQ 0.136 against 0.183, and h1-ALL 0.151 against 0.187.
- **The 2L test is too coarse for the inference.** At 2L h1-ALL the B2 crosstalk effect to be detected is 0.22·KL =
  0.0014–0.0030 nats. The SE of each Δ is 0.006–0.008, and "within 2%" means ±0.008. So "the 22% crosstalk must enter
  nonlinearly" does not follow.
- **One untested reading fits both.** Linear IDs matter at a cut only if source binding happens after that cut. In
  s12, the QQ state after block 1 is still almost linear in the raw inputs (R_in 0.057 against q0 0.048).

### P7. The bridge agrees with the direct estimate only where it reduces to it (medium)
- **The agreement is expected, not a validation.** At AUC ≥ 0.97 the classifier simply sorts the pooled 8 + 8 draws.
  L(0) and L(1) are then the two sample means of the same draws the direct estimate uses (2L h1-ALL: L(0) 0.3959 vs
  direct L(q) 0.3988; L(1) 0.0105 vs L(p) 0.0101), so "98.5% recovery" adds nothing. Where the ratio could add
  information (AUC 0.6–0.8), it recovers only 34–69%.
- **P5 holds vacuously.** It "holds" on 6 sites whose direct Δ is ≤ 2e-4 nats and whose classifier AUC is 0.500.
- **The L(α) shapes say nothing about dimensionality.** Two claims are made:
  - "most of the drop comes by α = 0.2, which fits a gap in a few directions";
  - "the plateau curve is linear".

  Both describe the classifier's log-ratio distribution and the endpoint ratio (L(1)/L(0) = 0.026 vs 0.37).
- **The q-side identity really fails** (ESS 0.01–0.5%, k̂ 0.6–1.6). The SNIS "stall" at α = 1 also reflects only 8
  draws per cell; the M ∈ {8, 16, 32} check was cut.

### P8. Smaller overclaims (low)
- **The predictions that held were the low-risk ones.**
  - P3 holds only in absolute nats; relative to KL it reverses (81 vs 41 at the full cut).
  - P5 is vacuous.
  - P7 predicted that a non-closed q would fail.
  - P8: one linear round cannot close a 100× gap.

  The risky predictions (P2, P4, P6) failed, so "P3, P5, P7 and P8 hold" overstates confirmation.
- **"The latents are the right variables" lacks a control.** At QT the comparator R_in holds only the query-question
  ID, while q0 holds the Bayes predictive mean and variance. No generic feature set of equal size was tried (for
  example raw (Q, log S), or random features of c). For the plateau net, q0 barely beats R_in at QQ (0.246 vs 0.271).
- **One regions finding is circular.** "The top decile of log ŵ_out holds 28–42% of the gap" restates that the gap
  sits where q is easiest to tell apart from p.
- **Some post-hoc results were never confirmed on fresh data.** The Amendment 2 sub-splits were scored on the same
  eval B that was used for acceptance.

## 2. What survives
1. **q0 and q1 do not close any read-heavy site.** The linear-Gaussian, per-token family with own-source Bayes
   latents leaves Δ of 0.05–0.55 nats per net at every QQ, full-cut and final-QT site, at least 10× ε, in every seed. The evidence uses fresh data and the direct
   estimator, which VALIDATION found unbiased. As a statement about this family, the result is solid.
2. **One-pool QQ tokens carry other sources' readings.** `cross_o` cuts Δ_rms by 30–42% there and by about 0 in
   two-pool seeds. The mixer head's knock-in carries 0.46–0.60, against about 0 in two-pool seeds. P2's knock-outs place
   q0's gap in attention in exactly these nets. This is information outside q0, tied to a known head, and it replicates
   B2's MLP1 replacement result with a different estimator.
3. **The plateau net's QQ states use little source binding** (nested test, P4) and hold little own-source reliability
   (q0 ≈ R_in), consistent with the step-1 probes and astra's intervention. This rests on n = 1 and a fitted family.
4. **The q-side estimators fail here.** Kelvin's q-side identity and q-side tempering cannot be used at these
   separations, and the bridge adds nothing to the direct Δ. Direct Δ / Δ_rms is the only closure number to trust.
5. **Tier i is exact (plumbing).** The 3L h1 cut is much easier to describe than h2.

## 3. Most informative next experiment: an information-matched, on-manifold oracle

**Design.** Replace a site's tokens by the network's real states, computed on a fresh episode x' ~ p(x | ψ(x)).
- All ID vectors are held fixed. At partial sites the rest of the episode stays real.
- No family is fitted, nothing is Gaussian, and no state is off the manifold.
- Every round-1 feature is a function of world variables, so exact conditioning can replace METHOD's binned q^R, with
  no bin-width confound.
- The sampler is `tp.cond_draws` plus two small exact extensions. In this world y ~ N(0, 1) and v_s is iid uniform on
  {0.1, 0.5, 2}.

**Conditioning sets ψ:**
- **ψ_own(s)** holds source s's raw inputs and all IDs fixed. Draw v_s from p(v_s | Q_s, S_s), then y* from
  p(y* | Q_s, v_s). The other sources' variances, labelled residuals and query readings come from the prior given y*
  and the fixed labels.
- **ψ_own+c** is ψ_own(s) plus the other sources' (Q, S). It contains all the information q0 and q1 read at QQ.
- **ψ_c** is c = (Q, S): `cond_draws` with the IDs kept.
- **ψ_U** is (Q, unordered S): reassign the three SS values to sources according to the exact posterior over the 6
  assignments given (Q, U). The blind reference already computes these weights.
- **At full cuts**, run two versions:
  - per-source independent draws, which tests the factorisation q0 assumes;
  - one joint draw, a V6 control that must give 0.

**Readouts** (Δ_rms on the round-1 eval B cells):
- **Form:** Δ_rms(q0) − Δ_rms(p̃[ψ_own+c]), what q0 loses through its family alone.
- **Cross-source information:** Δ_rms(p̃[ψ_own]) − Δ_rms(p̃[ψ_own+c]).
- **Binding:** Δ_rms(p̃[ψ_U]) − Δ_rms(p̃[ψ_c]).
- **Factorisation and nuisance:** Δ_rms(p̃[ψ_c]) with per-source draws.

**Decision rules (pre-register before running):**
- **D1.** If Δ_rms(p̃[ψ_own+c]) ≤ 0.1·Δ_rms(q0) at a site, at least 90% of q0's gap there is form. Stop naming
  features at that site and model the function instead (tier iii, a short surrogate block).
- **D2.** If the cross-source information at h1-QQ is ≥ 0.1 nats in s0, s2 and s3 and ≤ 0.02 in s1 and s4,
  structure (i) holds against the strongest control.
- **D3.** If the binding term at h1-QQ is ≥ 0.1 nats in the near-Bayes nets and ≤ 0.02 in the plateau net, the stuck
  net's binding failure holds without any fitted family.

**Why this experiment.**
- It answers P1 and P4 directly and removes the off-manifold confound of P5 by construction.
- It decides the direction of round 2: add variables, or model the function.
- It is cheap: forward passes and the G quadrature only, about 1 min per (net, site, ψ) at 48k draws. All 11 nets fit
  in roughly 1 h on 4 CPU processes.
- It is broader than ROUND1's own round-2 item (a), which proposes binned resampling only at sites with noise share
  ≥ 0.5.

## 4. Spot checks (reviewer runs, not part of round 1)
- **Code.** `review-interp/ln_check.py` and `review-interp/binding_check.py`, with outputs in `review-interp/*.json`
  and `*.log`. The scripts write to `/tmp/review-interp/`; the copies here are the record.
- **Data.** Fit split seed 60000 (20k episodes); eval B seed 63000 (6000 cells × 8 exact conditional draws), as in
  round 1.
- **Reproduction.** The q0 baselines match round 1's eval-B numbers (2L s0 h2-QT Δ_rms 0.484 vs 0.482; h1-QQ 0.480 vs
  0.484).
- **Results.** Tables are in P1 (LN gauge) and P4 (nested sets). Rows not shown, for 2L s0 at h1-QQ:
  - R_in + mean Q: 0.58;
  - R_in + mean Q + sorted SS: 0.59, so sorted SS adds nothing linearly.

  The plateau shows the same pattern: 0.28 and 0.29.
