# Fully non-Gaussian families for deep critical He-ReLU nets: design (nongauss-family lane, 29 October 2026)

Status: design v1, 17:25Z, before the GPT-6 critique. Nothing here is a claim. Numbers marked "exploration" come from
development net 112 (width 256, depth 64, sigma = 0.001), which is reserved for exploration and is not used by any
pre-registered test.

## 0. Brief and target

Kelvin (29 Oct): "we should just stop assuming any gaussian part whatsoever then. Let us seek a family that is fully
non-gaussian, even if its bad at the input, and hope that it at least scales on depth."
Context: on ARC's He-ReLU MLPs (x ~ N(0, I), W iid N(0, 2/n), no bias, noise sigma r_l after each ReLU), every family
with a Gaussian part (one Gaussian, K = 2, ARC's cumulants, cells with Gaussians inside, the latent-conditional
Gaussian) loses information at each re-projection, and at the critical point nothing damps those errors. The program
target is M1: a sample-free estimator that matches sampling at depths that grow with width. This lane asks a narrower
question first: is there a family with no Gaussian component whose re-projection error does not accumulate with depth?

The criterion (from the handoff, to be pre-registered with numbers in PREREG.md):
- R_F(l; L): replace the true state at layer l by the member of family F fitted to it (an oracle fit to Monte Carlo
  paths), run the true remaining layers (with noise), and record the nMSE of E[a_L] against an independent truth.
- F "scales on depth" if (i) R_F(l; L) does not grow with L for deep cuts l, and (ii) re-projecting onto F at every
  layer (composition) keeps the total bounded, at a level that is useful (near the sampler's error at the contest
  budget, about 1e-5 at 256 x 32).
- Caveats carried over: errors can add coherently (then the per-layer error must fall faster than 1/l^2), and
  distributional errors are not damped like mean shifts (Gaussianizing at layer 24 grows 4.5x by layer 32). Both are
  handled by measuring R_F(l; L) and the composition directly rather than predicting them from a survival law.

## 1. Exact structure used: positive homogeneity removes the scale

Noiselessly every layer is positively homogeneous: z_l(c x) = c z_l(x) for c > 0. Write the pre-activation at a cut as
    z = rho (u + delta),   u = E[z] / |E[z]|,   rho = u . z > 0,   delta = z / rho - u  (orthogonal to u)
(the gnomonic chart around the mean direction; rho > 0 on every path from layer 2 on, exploration). The gates of
every later layer depend only on delta. For any degree-1 homogeneous f (e.g. a_L as a function of z_l),
    E[f(z)] = E[rho] E_nu[f(u + delta)],   nu = the rho-weighted ("size-biased") law of delta.
So for the mean estimand a family only has to carry one scalar E[rho] and one law nu on R^{n-1}; the scale law enters
only through E[rho] and the size-biasing, and the scale mixture that Gaussians misread is exact by construction. With
noise this holds up to O(sigma) effects (the noise does not scale with rho); our oracle runs always use the noisy net.
By construction E_nu[delta] = 0.

Exploration (net 112): the chart is a large simplification at depth.
| layer | RMS of delta (nu) | x l | PR of Cov_nu(delta) | top-3 shares | sd log rho | gate-relevant neurons* | median abs skew / ex. kurt of t at those | same for z |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.435 | 3.5 | 29 | .09 .08 .06 | 0.151 | 84 | 0.08 / 0.06 | 0.06 / 0.14 |
| 16 | 0.238 | 3.8 | 18 | .14 .10 .08 | 0.175 | 42 | 0.13 / 0.09 | 0.12 / 0.24 |
| 24 | 0.181 | 4.4 | 14 | .17 .14 .09 | 0.183 | 38 | 0.15 / 0.15 | 0.14 / 0.23 |
| 32 | 0.192 | 6.1 | 9.5 | .24 .14 .09 | 0.187 | 36 | 0.18 / 0.06 | 0.25 / 0.32 |
| 40 | 0.116 | 4.7 | 9.0 | .25 .14 .10 | 0.192 | 24 | 0.15 / 0.00 | 0.20 / 0.45 |
| 48 | 0.103 | 4.9 | 8.6 | .23 .16 .13 | 0.195 | 15 | 0.34 / 0.12 | 0.14 / 0.46 |
| 56 | 0.084 | 4.7 | 7.8 | .26 .14 .12 | 0.194 | 17 | 0.23 / 0.00 | 0.15 / 0.39 |
(*neurons of the next layer whose normalized pre-activation t_i = w_i . (u + delta) has |mean| / sd < 1 under nu;
"z" = the same statistics for the raw next pre-activation, with the scale.) Read: the log-size spread saturates at
about 0.19 from layer 28 (the remembered scale is a fixed random variable set early); the angular spread falls about
like 1/l from layer 40; the number of gate-relevant neurons falls (36, 24, 15, 17); the whitened top principal
components of nu are platykurtic (excess kurtosis -0.1 to -0.65) and skewed (|skew| up to 0.44); the fourth-order
dependence among the top five components is weak (median |corr| of squared scores 0.03-0.05, max 0.12). Much of the
per-neuron non-Gaussianity of z at depth is the scale mixture (all-neuron |skew| 0.43-0.49 for z against 0.16-0.25
for t).

## 2. Families

All are fitted at the cut as oracles to 2^19 true paths (a fit set independent of the truth), then sampled (2^20
draws) and run through the true remaining layers. "Fully non-Gaussian" = no Gaussian distribution appears in any
component. Gaussian versions are kept only as controls, to attribute effects.

Charts. z-chart: the family describes z directly. Polar chart: the family describes nu (fitted on a size-biased
resample of the fit set, indices drawn with probability proportional to rho); samples are z = rho' (u + delta) with
rho' drawn from the empirical (unweighted) law of rho independently of delta, so E[rho' f(u + delta)] =
E[rho] E_nu[f] (the exact identity of section 1).

| name | chart | description | Gaussian part? | role |
|---|---|---|---|---|
| EMP | z | resample of the fit set | no | floor (weighted points: structured sampling, reference only) |
| G | z | one Gaussian, true mean and covariance | yes | control (prior work) |
| PCA3 | z | 64 quantile cells on the top-3 PCs, a Gaussian per cell (boundary-family) | yes | control (prior best) |
| I | z | z = mean + V s, s_k independent with the exact marginals of the PC scores of z | no | candidate (z-chart LNG) |
| PG | polar | delta ~ Gaussian with Cov_nu | yes | control (polar effect alone) |
| PI | polar | delta = V s, s_k independent with the exact nu-marginals of its PC scores | no | candidate (b-like, "LNG") |
| PR | polar | delta = V Lambda^1/2 e, e_k = +-1 (two-point, shape-free) | no | candidate (second order only) |
| PC3I | polar | 64 cells on the top-3 nu-PCs (exact joint law up to bins) x within-cell independent components with exact marginals | no | candidate (d, no Gaussian copula) |
| PC3G | polar | the same cells, Gaussian within | yes | control |
| PC3R | polar | the same cells, two-point within | no | candidate ((b): does the remainder's shape matter?) |
| ATTk | polar | delta = V_l Lambda_l^1/2 zeta, zeta = the joint whitened PC scores of nu at layer l - k (same net), signs fixed by third moment | no | candidate (a): attractor transfer |

Why each might scale on depth:
- PI / PR: closed under linear maps (V -> W V), and in the deep regime the map is linear except at the
  gate-relevant neurons, whose number falls with depth (above). Second order is exact by construction (the
  coupling arithmetic says it must be kept); PI also keeps every principal component's marginal shape. What it drops is
  dependence beyond second order between components; exploration says the fourth-order part is weak among the top
  components. PR asks whether any shape beyond second order is needed once the scale is exact.
- PC3I / PC3R: the low-dimensional core (top-3 components of nu, 33-52% of its variance at depth) is carried with its
  exact joint law; the remainder is carried either with exact marginal shapes (PC3I) or with second moments only
  (PC3R). If PC3R matches PC3I, the brief's candidate (b) is consistent: "exact in the remembered directions, moments
  for the rest".
- ATTk: if the whitened law of nu is (nearly) stationary in depth (an attractor), then using the shape from k layers
  earlier costs little, and a deductive family could fix the shape once and carry only (u, Cov_nu, E rho), at the cost
  of a Gaussian closure. If ATT fails while the shape statistics are stable in distribution, the PC matching (not the
  attractor) may be at fault; that is reported, not hidden.

Deductive push (stage 2, only for a family that passes the oracle tests): polar charts make the scale a scalar
recursion (E rho_{l+1} = E rho_l x E_nu[u_{l+1} . h(u + delta)]). For PI, the next pre-activations
t_i = w_i . (u + V s) are sums of independent components, so every per-neuron law (hence E[a]) is exact by 1-D
characteristic functions (O(n^2 G) for G grid points); the new covariance needs 2-D integrals for pairs with a
gate-relevant neuron (O(n_rel n^2 G^2)); the new components' marginals are the open part (they need the law of a
projection of dependent rectified variables; candidates: cumulants of order 3-4 of the projections, or a second
independence assumption). For ATT, only the covariance and the mean direction are pushed under a fixed shape (O(n^3)
per layer, like a Gaussian closure). Costs scale as n^3 per layer per cell, so all candidates are polynomial in width.

## 3. Measurements

Nets: width 256, depth 64, sigma = 0.001; exploration net 112; pre-registered nets 113-116. Truths: 2^22 Monte Carlo
paths per net (all 64 layers, float32 matmuls, float64 sums), seeds disjoint from the fit and sampling seeds. Fit set
2^19 paths of z_l; family samples 2^20; nMSE minus 1/N_truth minus 1/N_samples. EMP gives the fit-set floor (about
1/2^19 = 1.9e-6). Optional width axis: width 128, depth 64 (nets 112-114 at n = 128), if time permits.

- E0 (descriptive, attractor): the table of section 1 on nets 113-116 at layers 8-63: spread x l, PR, sd log rho,
  gate-relevant count, whitened top-5 PC skew/kurtosis, fourth-order dependence, rho-delta dependence, and the share of
  the gate-relevant neurons' t-variance inside the top-k nu components (k = 2, 4, 8, 16; the consistency condition of
  candidate (b)).
- E1 (single cut): R_F(l; L) for every family above, cuts l = 16, 32, 48, all L from l + 1 to 64.
- E2 (composition): re-project onto F at every layer from l0 = 32 to 63 (fit, draw 2^20 fresh samples, one exact noisy
  layer, repeat); error at every L. Families: G, PG, PI, PR, PC3I and EMP (re-sampling the paths themselves, which is
  plain Monte Carlo with bootstrap noise: the floor for composition). Coherence: project at the two cuts 32 and 48 only
  and compare with R(32; 64) + R(48; 64).
- E3 (optional): E0 and E1 for the best two candidates at width 128, depth 64 (l / n up to 1/2).

## 4. What would kill the idea (to be fixed with numbers in PREREG.md)

- Every fully non-Gaussian family has R_F(l; 64) growing with 64 - l at deep cuts as fast as G's, or no better than G
  in level: no family in this class scales on depth by representation.
- A family passes E1 but its composition grows like G's (coherent accumulation): representation is not enough.
- PR = PI = PG (shape irrelevant once the scale is exact): then the gain is the chart, not the non-Gaussianity, and
  "fully non-Gaussian" is the wrong axis.

## 5. Risks

- Fit-set floor: 2^19 paths give a floor of about 2e-6; families near it cannot be ranked. Ranking at the floor needs
  bigger fit sets (memory: 2^20 x 256 x 8 bytes = 2 GB per cut).
- Composition injects bootstrap noise each layer (it is plain Monte Carlo with re-fitting); the EMP composition
  measures it.
- One width, five nets, one noise level: an oracle study, not an estimator; costs of the deductive pushes are sketched,
  not measured.

## 6. v2 after the GPT-6 critique (17:45Z; `review/design_critique.md`, 15 findings, 4 critical)

What changed, by finding:
- (1, critical) The exploration's next-layer gate variable omitted the ReLU (t = W (u + delta) instead of
  W ReLU(u + delta)). Fixed in `scratch/e0.py`; the corrected net-112 table is `runs/explore/e0_256_64_112.json`. The
  table in section 1 is superseded: with the ReLU, gate-relevant counts are 68 / 39 / 43 / 24 / 17 / 13 at layers
  8 / 16 / 32 / 40 / 48 / 63, nearly the same with or without the scale (t vs z), and the top-16 nu-components explain
  (linear R^2, median over gate-relevant neurons) 0.76 at layer 16 and 0.84-0.93 at layers 32-63; top-4: 0.39-0.75.
- (2, 3) Chart: the gnomonic chart is replaced by the sphere chart w = z / |z|, rho = |z| (defined for every z != 0,
  bounded, finite covariance). The noise breaks homogeneity (the noise scales are fixed, so independent radius
  resampling is not exact for the noisy net): measured directly by PEMP (exact weighted angular law, independent
  radius) against EMP, a pre-registered control.
- (4, 15, critical) No candidate has a demonstrated deductive push; a polar push needs a change of measure (gain
  reweighting, ratio expectations). Accepted. This round answers only Kelvin's representation question (does some
  fully non-Gaussian family scale on depth when fitted to the truth); it makes no estimator claim and no cost claim.
  If a family passes, the push (including the reweighting) is the next step and must be demonstrated with counted cost.
- (5) PR is a specific shape (two-point, excess kurtosis -2), not shape-free; PCA-defined independent components are
  not closed under linear maps. Wording fixed; interpretation kept to "this fitted family".
- (6) Cells specified exactly (lib/ng.py `Cells`): weighted quantile bins (4 per axis) on the global top-3 weighted PCs;
  per cell its own weighted mean, covariance and eigenbasis; within-cell shape G / I / R; cell probability = cell
  weight; cells with fewer than 2n members dropped (not expected at 2^20 fit paths); samples may leave their cell.
- (7) ATT keeps its PCA matching (rank: eigenvalues above 1e-10 of the top; signs by weighted third moment) and adds
  ATTX (same layer of exploration net 112: universality across weights). A failure of ATT is reported as a failure of
  this transfer, not of every attractor; the critique's tangent-transport alignment is not done.
- (8) Sampling floors: each estimate's floor is the family's own tr Cov_F(a_L) / (N tr Cov_P(a_L)) plus 1/N_truth
  (`ng.scores`); weighted fits replace the size-biased bootstrap; two independent fit replicates per net.
- (9, critical) Composition noise: composition runs at two particle counts (2^20, 2^21) on fresh true paths, with plain
  Monte Carlo on the same paths (MC) and the bootstrap chain (EMP) as controls; a separate linearised estimate sums the
  single-cut error vectors of every layer 32..63 (minus EMP's, which removes the shared fit-set noise to first order)
  at 2^22 samples per cut.
- (10) A fixed-L/n series is mandatory for any scaling statement: 128 x 32, 256 x 64, 512 x 128 (L/n = 1/4), cuts at
  L/4, L/2, 3L/4. Unnormalized MSE and output variance are stored next to nMSE.
- (11) Numerical gates with an inconclusive region (PREREG).
- (12) Coherence: cosine of the output error vectors from cuts 32 and 48, and the two-cut interaction residual.
- (13) The independence assumption is tested directly by the PBk ladder: the top-k weighted PC scores drawn jointly
  (one fit-set row), the rest independent with exact marginals: k = 0 (PI), 2, 4, 8, 16, 32; k = n is PEMP.
- (14) The suggested bounded joint core with conditional beta residuals is not built; PC3I (cells x conditional
  independent residuals with exact marginals) is the nearest family tested.

Compute: one on-demand L4 (cap $5), estimated 1.5 h; E0 descriptives on the CPU (nice 19, ionice idle).
