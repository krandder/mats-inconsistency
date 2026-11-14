# PREREG (nongauss-family lane), 29 October 2026, written 17:50Z, before any run on nets 113-116

Question (Kelvin, 29 Oct): is there a family with no Gaussian part whose re-projection error does not grow with depth
on ARC's critical He-ReLU nets? This file fixes the tests, the statistics, the pass / fail / inconclusive rules and my
predictions. Design: DESIGN.md (v1 + section 6, v2 after the GPT-6 critique). Code: lib/ng.py, scratch/run.py,
scratch/e0.py, hashed with this file in prereg_hashes.txt.

Seen before this file (exploration only, not used by any test): net 112 at 256 x 64 (E0 descriptives, both the buggy
and the corrected version); toy runs of every phase on the width-64 versions of nets 112-113 at 64 x 16 with 2^15-2^16
paths (code checks; their numbers looked like: G grows with depth, PC3I flat, PI between). Nets 113-116 at widths 128,
256 and 512 have never been run by this lane. Noise scales r_l were computed for all of them (lib/rtable_ng.json).

## Objects

ARC's family (lib/anet.py): x ~ N(0, I_n), W iid N(0, 2/n), no bias, noise sigma r_l after each ReLU, sigma = 0.001.
Nets: seeds 113, 114, 115, 116 at three sizes with L/n = 1/4: 128 x 32, 256 x 64 (main), 512 x 128. Seed 112 at
256 x 64 is the exploration net and the ATTX reference only.
Estimand: E[a_L] at every output layer L. Metric (ng.scores): nMSE(L) = mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j)
minus the truth floor 1/N_truth and the estimate's own sampling floor tr Cov_F(a_L) / (N tr Var(a_L)).
Truth: plain Monte Carlo, 2^24 paths (128 and 256), 2^23 (512).

## Families (lib/ng.py v2; "fully non-Gaussian" = no Gaussian in any component)
EMP (fit-set resample, floor); PEMP (sphere-chart reduction control: w = z / |z| resampled with weights |z|, |z| drawn
independently); controls with a Gaussian part: G, PCA3, PG, PC3G; fully non-Gaussian candidates: I, PI, PR, PB2, PB4,
PB8, PB16, PB32, PC3I, PC3R, ATT8, ATT16, ATTX. Definitions are the docstrings of lib/ng.py.

## Runs (one L4 GPU job, scratch/gpu_job.sh)
- R1, main single cuts, 256 x 64: cuts 16, 32, 48; fit set 2^20 true paths per replicate; 2^21 family samples; two
  independent replicates (fit set and sampling seeds) for EMP, PEMP, G, PG, PI, PB16, PC3I; one replicate for PCA3,
  I, PR, PB2, PB4, PB8, PB32, PC3G, PC3R, ATT8, ATT16, ATTX.
- R2, composition, 256 x 64, nets 113 and 114: fresh true paths at layer 32, NC = 2^20 and 2^21; projection at every
  layer 32..63 for MC (no projection), EMP (bootstrap), G, PG, PI, PB16, PC3I; projection at 32 and 48 only for G, PI,
  PB16.
- R3, fixed L/n: 128 x 32 (cuts 8, 16, 24; two replicates; 2^20 / 2^21) and 512 x 128 (cuts 32, 64, 96; one replicate;
  fit 2^19, samples 2^20) for EMP, PEMP, G, PG, PI, PB16, PB32, PC3I.
- R4, linearised composition, 256 x 64, nets 113 and 114: single cuts at every layer 32..63 for EMP, G, PI, PB16
  (fit 2^20, samples 2^22), estimates at L = 64.
- E0 (CPU, descriptive): scratch/e0.py on nets 113-116 at 256 x 64 (2^17 paths).

## Statistics
Per net, the replicate mean. Across the four nets: mean m and a 95% interval m +- 3.18 s / 2 (t with 3 degrees of
freedom; s the standard deviation over nets). A rule PASSES if the interval's upper end meets it, FAILS if its lower end
violates it, and is INCONCLUSIVE otherwise. Per-net values are always shown.

## Tests and rules

T0 (reduction control; must pass for the polar families to be read as "sphere-chart" families): at 256 x 64,
D0 = R_PEMP - R_EMP at (cut 32, L = 64) and (cut 48, L = 64). Pass if both intervals lie below 2e-6.

T1 (single cut, "scales on depth by representation"), at 256 x 64, for every fully non-Gaussian family F:
- level: R_F(32; 64) <= 1e-5 and R_F(48; 64) <= 1e-5;
- growth: R_F(32; 64) - R_F(32; 40) <= 3e-6 and R_F(48; 64) - R_F(48; 56) <= 3e-6.
F passes T1 only if all four pass; F fails T1 if any one fails; otherwise inconclusive. G, PG, PC3G and PCA3 are
scored the same way for comparison.

T2 (fixed L/n), for EMP, PEMP, G, PG, PI, PB16, PB32, PC3I: the ratio rho_F = R_F(L/2; L) at 512 x 128 over
R_F(L/2; L) at 128 x 32 (per net, then the geometric mean over nets) and the same for 3L/4. A family is "width-stable"
if both ratios are <= 1, "width-growing" if both are >= 2. Descriptive otherwise (one replicate at 512).

T3 (composition), nets 113 and 114, L = 64 and L = 48, projection at every layer from 32:
- composed bias estimate B_F(L) = 2 C_F(L; 2^21) - C_F(L; 2^20) (C = nMSE of the chain; assumes C = B + V / NC), and
  the same for EMP (B_EMP, which should be about 0);
- F's composition is "bounded and useful" if, on both nets, B_F(64) <= 1e-5 and B_F(64) - B_F(48) <= 3e-6;
  "unbounded" if on both nets B_F(64) >= 3 B_F(48) or B_F(64) >= 1e-4. Otherwise inconclusive.
- R4 cross-check: S_F = sum over cuts l = 32..63 of (e_F(l) - e_EMP(l)) at L = 64 (error vectors), reported as nMSE.

T4 (coherence, 256 x 64): cosine between the output error vectors at L = 64 of cuts 32 and 48 (replicate 0), per
family; and from R2's two-cut chains, the interaction residual d = e_{32,48} - e_32 - e_48 (as nMSE).

## Predictions (written before any result on nets 113-116)
- P0: T0 passes (PEMP within 2e-6 of EMP).
- P1: G fails T1: R_G(32; 64) about 1e-4 to 4e-4, growing from L = 40 to 64; R_G(48; 64) above 3e-5.
- P2: PG is no better than G (within a factor 2), as in the width-64 toy runs.
- P3: PI fails T1 on level at cut 32 (2-5x better than G, not 1e-5); PR is worse than PI; I is worse than PI.
- P4: the PB ladder improves monotonically with k; PB32 may pass the level at cut 48 but not at cut 32.
- P5: PC3I is the best fully non-Gaussian family; it passes the level at cut 48 on most nets and fails at cut 32; its
  growth term is the smallest of the candidates. Overall no family passes all of T1.
- P6: ATT8, ATT16 and ATTX are no better than PG.
- P7: T2: G and PI are width-growing or flat; PB16 width-growing; PC3I unclear.
- P8: T3: G, PG and PI compositions are unbounded (B(64) >= 1e-4); PB16 and PC3I lower but above 1e-5.
- P9: T4: G's cut-32 and cut-48 error vectors have cosine above 0.3 (coherent).

## What each outcome would mean
- A family passing T1 and T3 would be the first evidence of a fully non-Gaussian family that scales on depth here
  (as an oracle; the deductive push remains to be built and costed).
- Failing T1 while beating G would say: non-Gaussian shape helps but does not stop the accumulation.
- PB_k reaching EMP only at large k would quantify how many jointly carried directions the representation needs, i.e.
  candidate (b)'s remembered-space dimension, and T2 says whether that number grows with width.

## Amendment 1 (written 17:51Z, after seeing R1 replicate 0 of net 113 in the streamed GPU log; before any run of
## the families below on nets 113-116)

What was seen: on net 113 every approximate family's error grows after the cut by a similar factor (cut 48 to L = 64:
G 1.1e-5 -> 1.6e-4, PB16 6e-7 -> 2.5e-5, PB32 about 0 -> 7.5e-6, PC3I 2e-7 -> 1.2e-5), and the PB ladder is
monotone in k. ATT8 / ATT16 / ATTX are at G's level.

Question: is the growth caused by unrepresented joint structure along the directions that the downstream network
amplifies, rather than along the state's own top principal components? And how many jointly carried directions are
needed to reach the fit-set floor?

Families (lib/ng2.py, runner scratch/run2.py; same fit paths as R1 replicate 0, because the walk seeds are the same):
- PB64, PB128 (lib/ng.py's PB family with larger blocks);
- PBA8, PBA16: joint block on span(top-k weighted PCs of the direction law, top-k eigenvectors of the downstream
  sensitivity M = E[g g^T], g = d(r . a_64)/d z_l, r ~ N(0, I), 2^13 paths), the rest regressed linearly on the block
  and carried by independent exact marginals;
- PBS16: the block on the top-16 sensitivity eigenvectors only.
Runs: 256 x 64, nets 113-116, one replicate, cuts 32 and 48, fit 2^20, samples 2^21, truth 2^24 (recomputed with the
same seeds on a second L4).

Rules (descriptive comparisons, per net and mean over nets, at L = 64):
- H1 (sensitivity directions matter): PBA8 (at most 16 block directions) beats PB16 by a factor >= 2 at both cuts on
  at least 3 of 4 nets -> supported; PBA8 within a factor 1.5 of PB16 on at least 3 nets -> not supported.
- H2 (block size): the smallest k in {16, 32, 64, 128} with R_PBk(c; 64) <= 3 x R_EMP(c; 64) + 2e-6, per cut.
Predictions: H1 not supported (PBA8 within 1.5x of PB16); PB64 about 3-5e-6 and PB128 near the floor at cut 48;
at cut 32 PB128 above 5e-6 (still growing).

## Amendment 2 (written 18:09Z; Kelvin's questions relayed by main at about 18:00Z and 18:10Z; before any run below)

Kelvin: "How can you tell it's not just that we haven't found the right structure that has low-error per layer, or
that can even self-correct?" and "a 'bidirectional measure' of importance of directions ... that helps preserve
important long-depth structures". Main's operationalisation: (1) count the directions the downstream net amplifies
against width at fixed L/n; Lyapunov spectra of the downstream Jacobian products (Gaussian-product theory: about 2n/L'
directions survive a horizon L'); (2) a family exact in the bidirectionally important subspace and loose elsewhere;
(3) a family-agnostic learned compressor. Seen before this amendment: R1 on nets 113-114 (streamed log), and toy runs
of every new script at 64 x 16 (net 113) as code checks.

T5a (scratch/amp.py; spread-weighted downstream sensitivity). S = R M R with M = E[J^T J] (J = d a_Lo / d z_l,
2^16 paths, one random output projection per path), R = (P C P)^(1/2) (C = Cov z_l from 2^18 paths, P removes the
mean direction, which homogeneity makes exact). Sizes 128 x 32 (cuts 16, 24), 256 x 64 (32, 48), 512 x 128 (64, 96);
Lo in {l + 4, l + 8, l + 16, L}; nets 113-116. Reported: k90 and k99 of S (directions holding 90% / 99% of its
trace), PR, and the same for P M P. Rule at Lo = L, for cut 3L/4 and cut L/2 separately (mean over nets): "few" if k99
at 512 is at most 1.5 x k99 at 128; "about linear in n" if it is at least 3 x; otherwise "sublinear".
Prediction: sublinear (k99 grows 1.5-3x from 128 to 512); k90 at 256 x 64 between 5 and 30.

T5b (scratch/lyap.py; finite-time Lyapunov spectra of single paths). Cut l = L/4; horizons L' = L/8, L/4, L/2
(fixed L'/n across sizes); nets 113, 114; 128 paths each. Reported: N_surv(e^-1), N_surv(e^-2) (singular values within
e^-1, e^-2 of the top), the per-layer gaps, against the Gaussian-product value 2n/L'. Rule: "O(1) at fixed L'/n" if
N_surv(e^-1) at 512 is within 1.5x of its value at 128 for every L'/n.
Prediction: N_surv(e^-1) is 3-8 (well below 2n/L', because of the ReLU masks) and roughly constant across widths at
fixed L'/n (rule met).

T5c (lib/ng2.py PBW{k}; the family exact in the bidirectional subspace). Joint block on U = orth(R V_k), V_k the
top-k eigenvectors of S (built at the cut from the fit paths, sensitivity to L = 64), the rest regressed on the block
and carried by independent exact marginals. 256 x 64, nets 113-116, cuts 32 and 48, k = 8, 16, 32, with the R1 fit
paths (replicate 0). Rules: T1's growth criterion (R(c; 64) - R(c; c + 8) <= 3e-6, t-interval over nets) and level
criterion (R(c; 64) <= 1e-5) for each k; compared with PB_k at equal k.
Prediction: PBW_k beats PB_k at equal k by less than 2x, and its error still grows after the cut (fails growth at cut
32 for every k).

T6 (scratch/comp.py; learned compressor). Cut 3L/4 at all three sizes (net 113), plus cut L/2 at 256 x 64; code size
k = 1, 4, 16, 64; models 'mlp' (trained end to end through the true layers on the pathwise output error, 3000 Adam
steps, batch 4096), 'lin' (trained linear bottleneck), 'pca' (untrained projection). Reported on 2^17 held-out paths:
pathwise nMSE, the channel floor (the output variance that the activation noise after the cut creates by itself),
the bias of E[a_L] through the deterministic bottleneck, and the bias with a shuffled residual (code of path i plus
the reconstruction residual of an independent path). Rules: k_path = smallest k with pathwise nMSE <= 0.01; k_bias =
smallest k with shuffled-residual bias <= 1e-5. "Compact structure" if k_path or k_bias at 512 is at most its value at
128 (and both are reached); "growing with width" if the value at 512 is larger, or not reached while reached at 128.
Prediction: no k <= 64 reaches pathwise 0.01 at any width (the per-path downstream information is high-dimensional);
the channel floor is about 1e-5 to 1e-4; k_bias grows with width or is not reached at 512.

## Amendment 3 (written 18:26Z, after seeing R1 on all four nets and Amendment 1 on nets 113-115 in the streamed logs;
## before any run below)

Seen: at 256 x 64 the PB ladder reaches the fit-set floor only at k = 64 (cut 32: PB64 about 1.7e-6 against EMP about
5e-7; PB32 1.6e-5) and is at the floor at k = 128. Question (Kelvin's O(1) against O(n)): how does the needed number
of jointly carried directions scale with width at fixed L/n?
Runs (scratch/run2.py, third L4): PB ladder with EMP at 128 x 32 (k = 8, 16, 32, 64; cuts 16, 24) and 512 x 128
(k = 32, 64, 128, 256; cuts 64, 96), nets 113-116, one replicate (walk seeds as R1 replicate 0), fit 2^20 (128) and
2^19 (512), samples 2^21 / 2^20, truths 2^24 / 2^23 with R3's seeds.
H3: k*(size, cut) = the smallest k in the ladder with mean R_PBk(c; L) <= 3 x mean R_EMP(c; L) + 2e-6 (L = output).
With 256 x 64 from R1 / Amendment 1 (k in {16, 32, 64, 128}). Rule: "O(1)" if k*(512) <= 1.5 x k*(128) at both
relative cuts; "about linear in n" if k*(512) >= 3 x k*(128) at both; otherwise "sublinear"; "not reached" counts as
above the largest k tried.
Prediction: about linear in n (k* about n/4: 32, 64, 128).

## Amendment 4 (written 18:43Z, after seeing the streamed logs of R2 (composition, nets 113-114), T5a at 128 and
## 256, and Amendment 3 at 128 (all nets) and 512 (net 113); before any run below)

Seen: every family's composition from layer 32 grows far above its single-cut error (net 114 at L = 64: G 2.3e-3, PI
2.4e-3, PB16 7.7e-4, PC3I 3.9e-4; net 113: 2.5-5e-4), while the single-cut PB ladder reaches the floor at k = 64 at
both 128 x 32 and 256 x 64. Question: does a family that is at the floor for single cuts (PB64, PB128) also stay
bounded under re-projection at every layer?
Runs (scratch/run.py e2, fifth L4): composition from layer 32 at 256 x 64, nets 113 and 114, NC = 2^20 and 2^21,
projection at every layer for MC, EMP, PB64, PB128, and at the cuts 32 and 48 only for PB64; and from layer 16 at
128 x 32 (nets 113, 114, NC = 2^20 and 2^21) for MC, EMP, PB32, PB64.
Rule: T3's rule ("bounded and useful" if on both nets B(L) <= 1e-5 and B(L) - B(3L/4) <= 3e-6 with B = 2 C(2^21) -
C(2^20); "unbounded" if on both nets B(L) >= 3 B(3L/4) or B(L) >= 1e-4; otherwise inconclusive).
Prediction: PB128 bounded and useful at 256 x 64; PB64 inconclusive (composed error 5e-6 to 3e-5); at 128 x 32 PB64
bounded (k = n/2) and PB32 unbounded.

## Deviations (recorded as they happen)

- 19:05Z: T5b at 512 x 128 was stopped by hand after 24 minutes. cusolver SVDs of the 512 x 512 float64 Jacobian
  products failed to converge and fell back to a much slower method. T5b is therefore reported for 128 x 32 and
  256 x 64 only, and its width rule (512 against 128) cannot be scored. T6 started in its place on the same VM.
- 18:30Z: T5a, T5b and T6 failed without running on the second VM (launch_gpu.sh did not stage amp.py, lyap.py and
  comp.py). They were rerun unchanged on a fourth VM (hashes identical to Amendment 2).
- 19:05Z: T5b at 512 x 128 rerun unchanged on the CPU (General, nice 19; torch CPU SVD via LAPACK), outputs in
  runs/cpu_a2/.
- 19:48Z: R4 (linearised composition) was stopped after net 113 to stay within the $5 GPU cap (estimated total
  then about $4.6); net 114 was not run. R4 is reported for net 113 only.
- 20:05Z (after the claims review): the first-draft analysis of H3 used max(mean EMP, 0) in the threshold instead of
  the pre-registered 3 x mean EMP + 2e-6; corrected in scratch/analyze2.py (literal verdict: about linear in n). The
  T2 geometric ratios in scratch/analyze.py exclude nets with nonpositive corrected errors (affects only the EMP and
  PEMP control ratios, which are reported as unresolved).
