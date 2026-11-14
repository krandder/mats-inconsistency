# Boundary-aware families for deep critical He-ReLU nets: design (boundary-family lane, 29 October 2026)

Status: design v2 (15:40Z). v1 (15:20Z) was critiqued by GPT-6 (`review/design_critique.md`, 16 findings, 3 critical);
v2 adds section 6 (changes after the critique, with Kelvin's design inputs of 15:25-15:35Z: relevance axes, hyperplane
cells, size per cell, gate compression). Exploration on development nets only (width 64 nets 500-501, width 256 ARC
dev net 100). Nothing below is a claim; every number is a dev-net exploration number, and the one-step numbers of F3/F4
are paired (cells fitted on the same paths as the reference), so they measure closure residuals, not estimator error.

## 0. Target and bar

M1 (program, Kelvin 29 Oct): a sample-free estimator that matches or beats sampling at equal compute on deep random
MLPs at depths that grow with width, here 256 x 32 (ARC's He-ReLU nets, sigma = 0.001). The bars on 256 x 32:
- per-neuron means E[a_32]: ARC's factorized K=3 2.2e-4 at 2.06e11 FLOPs (median, 24 eval nets); the Kerdock sampler
  6.7e-6 at 2.72e11 (0.40/N), 3.6e-5 at 6.8e10, and parity with sample-free entries at about 2e9;
- output moments of z_32 (orders 1-4): good enough that the chi-mixture saddlepoint beats zero at 1e-4 and 1e-6 (it
  does with 2^20-2^22-path Monte Carlo moments; none of the tested deductive descriptions do);
- tails P(z_32,i > t) at 1e-4 and 1e-6 directly (ARC's readouts never beat zero at depth 32).
A result counts toward M1 only if its error-vs-compute curve keeps falling with its compute parameter at least as fast
as the sampler's 1/N up to the contest budget. Beating ARC's readout, or winning at depth 8 only, is a note.

## 1. What a family has to carry (measured on dev nets, exploration)

Notation: z_l = W_l h_{l-1} (pre-activation), a_l = ReLU(z_l), h_l = a_l + s_l eps. Noiselessly z_l(x) = |x| zeta_l(x/|x|)
(positive homogeneity): the gates depend only on the direction of the state; a scale passes every ReLU exactly.

F1. **The Gaussian closure's error at depth is partly a scale error, from misreading a scale mixture.** Exact Gaussian
closure (bivariate-normal ReLU moments), error of E[a_L] split into the component along the true mean and the rest:
width 64, depth 16: nMSE 3.5e-3, 81% along the mean, a +2.2% overestimate of its norm; width 256, depth 32 (net 100):
7.6e-4, 50% along the mean, +0.47%. Mechanism: at depth the top mode of z is radial (66% of the variance at width 64
layer 15, 58% at width 256 layer 31); if z = N d with a random scale N, then for a neuron at its boundary the
moment-matched Gaussian gives E[ReLU] proportional to sqrt(E N^2), the truth to E N, an overestimate of about
Var(N)/2 per layer for near-boundary neurons, which compounds multiplicatively.
F2. **Scale can be carried exactly** (log-radial Gaussian, LRG: z = exp(rho)(u + d), (rho, d) jointly Gaussian, readout by
Gaussian tilting, push by Stein's lemma and the delta method; `lib/lrg.py`). Width 64 depth 16: 1.4e-3 (2.5x better
than the Gaussian closure, scale error +0.26%). Width 256 depth 32 (net 100): 9.0e-4, no better than the Gaussian
closure (it overcorrects the scale, -0.36%), and the orthogonal error is unchanged. ARC's K3f on that net: 3.2e-4.
F3. **One step: cells in the top collapse modes hold the missing information.** With the true distribution of z_l
(2^18-2^19 Monte Carlo paths), the one-step error of the moments of a_l = ReLU(z_l):
| | Gaussian (true mean/cov) | 16 cells, top mode | 16 cells, top 2 | 64 cells, top 3 |
|---|---|---|---|---|
| width 64, layer 8 / 12 / 15 (mean of a, nMSE) | 4.4e-4 / 3.1e-4 / 2.4e-4 | 4.6e-5 / 7.2e-5 / 4.3e-5 | 2.5e-5 / 3.5e-5 / 2.4e-5 | 1.6e-5 / 1.4e-5 / 9.3e-6 |
| width 256, layer 16 / 24 / 31 | 2.9e-5 / 5.8e-5 / 2.3e-5 | 1.8e-5 / 2.9e-5 / 1.4e-5 | 1.1e-5 / 1.0e-5 / 8.5e-6 | 4.8e-6 / 6.4e-6 / 6.6e-6 |
(cells: quantile bins of the top-k PCA modes of z_l; Gaussian per cell with the true within-cell moments; the width-256
row is near its sampling floor of about 4e-6.) Per-neuron non-Gaussianity is large at depth (width 64 layer 15: median
|skew| 0.83, excess kurtosis 1.3; width 256 layer 31: 0.47 and 0.47), and the direction d is much more Gaussian than z
(width 64: |skew| 0.34, exkurt 0.35).
F4. **One push keeps most of it.** The same oracle cells at layer l pushed one layer by a per-cell Gaussian closure: width
64, read out at l+1: 7.2e-5 (l = 8), 2.4e-5 (12), 2.9e-5 (14), against 1.6e-3, 4.7e-4, 4.5e-4 for the single Gaussian
pushed the same way (15-23x).
F5. **Deductively generated cells do not reproduce F3/F4.** Splitting a Gaussian cell (moment-preserving 3-way splits
along nonlinearity-weighted or PCA directions, exact boundary splits with the gate forced on each side, Ward merging),
then per-cell pushes: width 64 depth 16, best 2.5e-3 with 27 cells (Gaussian closure 3.5e-3); exact boundary splits of
single neurons, 8 cells: 3.4e-3. A split of a Gaussian represents that Gaussian; it does not recover non-Gaussian
structure lost earlier, and the closure inside each still-wide cell keeps losing it. (Width 64 is dominated by early
layers: the Gaussian closure is already at 1.1e-3 by layer 4 there, against 3.4e-5 at width 256.)
F6. ARC's K3f at width 256 (net 100) is excellent while the state is high-dimensional and degrades in the collapsed
regime: 3.1e-6 (layer 8), 1.8e-5 (16), 8.8e-5 (24), 3.2e-4 (32). At width 64 every kprop variant is worse than K2
(depth 16: K2 3.5e-3, K3f 7.5e-3, K3f-augmented 2.2e-2).

Reading: the missing information is (i) the scale mixture (radial mode), (ii) low-dimensional non-Gaussian structure in
the top collapse modes, both strong only at depth, and (iii) weak, high-dimensional per-neuron non-Gaussianity at the
shallow end, which cumulants handle well and cells do not. Cells capture (i)+(ii) when they are fitted to the truth;
they must therefore be created from an accurate description, and kept narrow in the modes where each later layer acts
nonlinearly.

## 2. Candidate A: homogeneous boundary-aware cells (HBC)

### State
At layer l (pre-activation), a mixture over cells c with weights pi_c:
    z_l | c = exp(rho) (u_l + d),   d orthogonal to u_l,   (rho, d) | c ~ N(mean_c, Cov_c)   (LRG per cell)
u_l is the unit reference direction (the mean direction; u_l . z > 0 almost surely once the state has collapsed, from
layer 2 on in these nets). Cells partition the direction space d along the decision-relevant collapse modes; the scale
is exact by homogeneity in every cell. The family contains the plain Gaussian mixture as the limit Var(rho) -> 0.

### Readout (exact under the family)
E[a_i | c] = E[exp(rho) ReLU(u_i + d_i)] = exp(E rho + V rho / 2) * E[ReLU(u_i + d_i + c_i)], c_i = Cov(d_i, rho)
(Gaussian tilting), a rectified-Gaussian mean. Second moments: exp(2 E rho + 2 V rho) E[ReLU(.. + 2c_i) ReLU(.. + 2c_j)].
Tails of an output z_L,i: P(exp(rho)(u_i + d_i) > t) per cell, a 1-D quadrature over d_i with the Gaussian conditional
of rho (no saddlepoint, no moment truncation; the lognormal radial part gives heavy upper tails by construction).

### Where the decision boundaries enter
The gate of neuron i is sign(u_i + d_i), independent of rho. In cell c it is on with probability
Phi(t_ci), t_ci = (u_i + E[d_i|c]) / sd(d_i|c): the mass on each side of the boundary is computed exactly for the cell's
Gaussian; the residual (within-cell) spread smooths the gate into this probit. Gates with |t_ci| > 5 are decided in that
cell (linear neuron). Splitting a cell along a collapse mode moves its boundaries out of the residual into the cell
structure: a boundary that cut through the parent passes between children, and each child sees that gate as nearly
decided. The cells needed are the cells of the hyperplane arrangement restricted to the k-dimensional collapse
subspace: O(D^k / k!) for D near-boundary gates, not 2^D.

### Step l -> l+1 (per cell, from the weights only)
1. y = W_{l+1} ReLU(u + d) with d | c Gaussian: exact mean (rectified means), exact covariance (bivariate-normal formula
   for pairs with an uncertain neuron, closed form for decided pairs), Cov(y, rho) = W (c o Phi(t)) by Stein's lemma.
2. z' = exp(rho) y + noise; u' = E y / |E y|; rho' = rho + log(u' . y), d' = P' y / (u' . y) by the delta method (first
   order in the covariances, second order in the means). [F2 shows this step overcorrects the scale at width 256; to be
   replaced by a radial grid (discrete scale nodes, each pushed as one scaled copy, since ReLU commutes with scale), or
   by matching the first two moments of z' exactly and the log-radial variance from the gain recursion.]
3. Split: before the push, every cell whose nonlinearity energy E_c = sum_i Var(z_i | c) g(t_ci) (g = the part of
   Var ReLU(t + Z) not explained by its best linear fit) exceeds a threshold is split 3-way (moment-preserving, child
   variance 1/2 along the mode) along its top direction mode, until the budget C_max is reached; the mode choice is the
   top eigenvector of Cov(d | c) weighted by the uncertain gates (the "decision-relevant" rotation).
4. Merge after the push (Ward cost in the shared whitened metric; merged moments exact), back to C_max.

### Initialization (the lesson of F5)
Cells must be created from an accurate description, before the collapse makes the state strongly non-Gaussian:
- shallow regime (PR of z large; non-Gaussianity weak and high-dimensional): a cumulant description (ARC's K3f, run up
  to a switch layer m_s, or our own order-3 per-neuron correction of the Gaussian closure);
- at m_s: convert to cells along the top k modes with nodes that match the first three or four cumulants of each mode
  (skewed 3-node rules), then HBC to depth L.
The switch is chosen by the participation ratio of z (deductive: from the description's own covariance).

### Cost model (flopscope rates, n = 256, L = 32)
Per cell per layer: rectified moments n_unc n pairs x 2.5e3 FLOPs (n_unc = 50-130: 3e7-8e7) + 4 n^3 for W Cov W^T
(6.7e7) + O(n^2): about 1.5e8. C cells over the last L - m_s layers: C (L - m_s) 1.5e8, e.g. 64 cells x 16 layers = 1.5e11,
plus the K3f prefix to m_s = 16 (about 5e10, ARC's counter): about 2e11, at K3f's depth-32 cost and below the contest
budget 2.72e11. A homoscedastic variant (cells share the residual covariance; per cell only the mean and per-neuron
variances, n^2 + n_unc n x 50 with a cheap pair formula) costs about 1e6 per cell per layer, so C ~ 10^4 fits the same
budget; its closure error is the price (to be measured).

## 3. Candidate B: backward from the question

Meeting identity at a cut m: E[f] = E_{p_m}[v_m(h)], v_m(h) = E[f | h_m = h], f = a_L,i (mean) or 1{z_L,i > t} (tail).
- Backward functions over the last j layers are boundary-aware in the same sense: on the forward cells' support, gates
  with |t| > 5 are linear, so v_m restricted to a cell is a j-layer net whose only nonlinear units are that cell's
  uncertain gates (few at depth). For j = 1 the meeting is exact per cell (rectified Gaussian for the mean; Gaussian
  tail of w . h for the event). For j >= 2 the cell expectation of v_m is again a forward push, so B for the mean is
  "A plus goal-oriented refinement".
- Goal-oriented refinement (dual-weighted residuals): the output error is approximately sum_l <closure residual_l,
  adjoint_l>; split the cells where |residual x adjoint| is largest instead of where the residual is largest. For a tail
  event the adjoint is concentrated on the cells that reach the event; splitting there is the deterministic analogue of
  splitting/importance sampling ("spend precision only where the backward function is large").
- Tails at depth 32: the event is driven by the scale's upper tail and the direction's tail along w_i; HBC's per-cell
  lognormal radial part gives P(z_L,i > t) directly, so the readout needs no moment truncation.

## 4. Depth growing with width: why it could scale, and what kills it early

Why it could scale (hypotheses):
- H-scale: the depth-growing non-Gaussianity is mostly the radial (norm) mode: Var(log |h_L|) grows about linearly in
  L/n (F1; measured log-scale sd 0.30-0.35 at width 64 layers 4-15). A family that carries the scale exactly has no
  error growth from it at any L/n.
- H-angle: the angular deviation contracts with depth (He-ReLU correlation map, 1 - c_l ~ 1/l^2), so fewer neurons are
  uncertain per layer (about n/l) and the ReLU is closer to linear on each cell; per-layer direction closure error should
  fall with l, so the total saturates instead of growing with L.
- H-cells: the decision-relevant subspace at depth has a few dimensions (PR of z 2.8-7.4 at width 256 layers 16-31), so
  the cells needed (3^k, k <= 4) do not grow with n at fixed L/n.
Early kill measurements (cheap, dev nets):
- K1. Propagated oracle cells: true cells at layer m (16, 24), per-cell pushes to 32. If the error at 32 is not at least
  10x below the Gaussian closure's and below 3e-5 at width 256, no initialization can rescue HBC (no deductive cell
  family beats a perfect start). [running: scratch/diag4.py]
- K2. Error vs cells: if the propagated error does not fall at least like 1/C over C = 1..64 (at fixed m_s), HBC cannot
  track the sampler's 1/N.
- K3. Depth trend at fixed L/n: one-step direction closure error per layer vs l (width 64 and 256); if it does not
  decrease with l, H-angle fails and the total grows with L.
- K4. Width trend at fixed L/n (64 x 8, 128 x 16, 256 x 32): number of cells needed for a fixed one-step error; growth
  with n kills H-cells.

## 5. Tests (to be pre-registered in PREREG.md before running)

Development: width 64 (dev nets 500-509) and ARC dev nets 100-111 at width 256. Evaluation: fresh width-256 nets
>= 64 for any claim (new 2^24-2^26-path truths, CPU), or ARC's eval nets 0-23 for comparison with its tables
(reused, not fresh). Bars, all at equal or lower cost than the comparator:
- T1 (means, 256 x 32): HBC beats ARC's K3f (median nMSE ratio > 1 with CI above 1) at <= 2.06e11 FLOPs; M1-relevant only
  if it also beats the Kerdock sampler at its own cost.
- T2 (moments): the HBC output moments read out by the chi-mixture saddlepoint beat zero at 1e-4 and 1e-6 at depth 32.
- T3 (tails): HBC's direct tail readout beats zero at 1e-4 and 1e-6 at depth 32 (ARC part 2 truths, nets 0-19).
Kill rules: K1-K2 on dev nets before any evaluation run.

## 6. Changes after the GPT-6 critique and Kelvin's inputs (v2, 15:40Z)

Critique findings taken (numbering as in `review/design_critique.md`):
- (1, 2, 12) LRG: `from_gauss` mixed a lognormal radial variance with linearized cross-covariances and can give a
  non-PSD joint covariance; the radial chart needs u . z > 0, which noise does not guarantee. Fix: first-order chart
  (the joint covariance of (rho, d) is the linear image [u, P]^T S [u, P] / |m|^2, PSD by construction) with a PSD and
  reconstructed-moment check; keep the excluded mass. "Scale exact" is withdrawn: the scale passes ReLU exactly, its
  Gaussian closure in log space does not.
- (3) Forced-gate bugs in `gmprop.py` (readout ignored forced gates; on/off order dependence): fixed 15:30Z. The
  boundary-split runs of F5 are invalid until rerun.
- (4, 7) An untruncated Gaussian cell is not a boundary cell: it overlaps every hyperplane. Boundary cells are
  represented below as truncated Gaussians (constraints carried with the cell), as the critique's alternative (16)
  proposes; within a cell the map is affine for the constrained gates.
- (5) The one-step oracles are paired; from now on cells are fitted on one Monte Carlo set and scored against the
  independent 2^25-path truth at the output (diag4, diag5).
- (6) Section 1's mechanism for F1 is a hypothesis, not established. Contrary evidence is kept in view: the direction
  part is not very low-dimensional (width 64 layer 15: PR of d 6.75, top four modes 68%), and the prior lane's
  top-8-mode cumulants recovered only 1.9-4.1x against 19-120x for per-neuron cumulants.
- (9, 10, 11) The decisive measurement is representation error vs push error at a cut: the same cells (fitted once)
  propagated (a) through the exact remaining network by sampling (representation) and (b) by the per-cell closure
  (push). If (a) is small and (b) is not, the propagation rule is the problem; if (a) is not small, the family is.
  Gates for M1 are separate from gates for beating K3f, and all are judged on error against total FLOPs.
- (13) Candidate B is an objective until a refinement score is defined; it is deferred until a forward family passes.
- (14, 15) Costs will count every operation (charts, splits, merges, readouts) with the same weights for all methods;
  the noise scales r_l are part of the net definition (as in the prior lane), not charged.

Kelvin's inputs (via main), and how they enter:
1. **Relevance axes.** Cell axes are the top eigenvectors of M_l = E[g g^T], g = gradient of the question with respect
   to z_l under the forward distribution (for the all-outputs mean: g = J^T r with r ~ N(0, I), so E[g g^T] = E[J^T J]).
   PCA of the deviation ignores the question. Both are reported, with their overlap (principal cosines), the variance
   the relevance axes carry and the gradient energy the PCA axes carry (diag5).
2. **Hyperplane cells.** Cells are cut along the hyperplanes of the most relevant near-boundary gates (relevance = gate
   entropy x sqrt(M_ii)); each cell is a truncated Gaussian with its sign pattern as constraints, so the mass on each side
   of each retained gate is exact for the cell's law; cells with the same downstream effect are merged. The number of
   cells carrying mass is recorded against the gate-entropy census (outbox/gate-entropy-2026-10-29.md, due ~17:30Z).
3. **Size per cell.** Each cell carries its own scale (the size gain depends on which gates are on): per-cell
   log-radial coordinate (rho_c) with the per-cell chart of section 2.

### Gate compression (Kelvin: "families that can line up with the gate hyperplanes and allow similar/functionally-
### correlated gates to be compressed")
Raw near-boundary gates at layer l: D_raw = #{i : gate entropy h(P(z_i > 0)) > 0.1 bit}. They are grouped by three
criteria, each giving an effective number of boundaries D_eff:
- (geometry) nearly parallel and nearly coincident hyperplanes inside the relevance subspace U (k = 2-3): gate i is the
  hyperplane {y : mu_i + U_i . y = 0} in y = U^T (z - mu) coordinates; two gates are grouped when their unit normals
  U_i / |U_i| have cosine > c and their offsets mu_i / |U_i| differ by less than delta (in sd units of y);
- (co-switching) the phi coefficient of the two gate indicators on the forward distribution exceeds c;
- (downstream effect) the cosine of the gates' downstream effect vectors (the columns d a_L / d a_l,i, averaged on the
  boundary) exceeds c.
A group is represented by one boundary: the hyperplane of its highest-entropy member (or the group's mean normal and
offset), plus a within-group correction: members' gates are probits given the representative's side (their conditional
on-probabilities inside each cell), which is what the per-cell Gaussian already computes for any gate not cut.
Reported per layer: D_raw, D_eff(c) for c in {0.8, 0.9, 0.95} for each criterion, the cells with mass > 1e-4, and the
error cost: representation and push error of cells cut along the D_eff representatives against cells cut along the
same number of raw gates and against the uncompressed set (diag6).

### Revised early kills (dev nets 100-102, width 256, cuts m = 16 and 24, before any evaluation run)
- K1' (representation): the best family's representation error at depth 32 must be <= 1/10 of the single Gaussian's
  and <= 2e-5 with <= 64 cells, else the family is dead (no propagation rule can beat its own representation).
- K1'' (push): the per-cell closure must keep the representation error within 3x, else the propagation rule is dead.
- K2' (compute): error against total FLOPs at C = 1, 4, 16, 64 must fall at least as fast as 1/FLOPs between 4 and 64
  cells, else it cannot track the sampler.
Only if K1'-K2' pass: the cumulant-to-cell conversion at the switch layer (projected joint cumulants, positive-weight
PSD components, checked on held-out boundary readouts), then PREREG and evaluation.

## 7. The discrete part must be factored (v3, 16:10Z; main's request after the gate-entropy census)

The census (outbox/gate-entropy-2026-10-29.md) finds, for ARC's He-ReLU nets: every gate relevant; H(G) proportional to
width at every depth (layer 32 at width 256: 28-40 bits); no merging at width >= 128, not even for the mean; a
conditional Fano illustration of about 2^25 classes at layer 32. A family that lists gate classes is therefore out at
256 x 32. This lane's measurement D (PREREG, `scratch/diag7.py`) agrees and adds the pairwise picture: among the
near-boundary gates (entropy > 0.1 bit) of width-256 nets 101-102 at layers 16-31 (43-136 gates, 29-83 bits of summed
marginal entropy), the median |phi| between two gates is 0.05-0.08 and the largest is 0.45-0.67: no co-switching groups
at |phi| >= 0.5 (D_eff = D_raw), no downstream-effect groups at cosine >= 0.8, and the top-3 (top-8) PCA latent
predicts the sign of only 26-55% (37-68%) of the near-boundary gates in >= 90% of paths. Same picture at 64 x 8 and
128 x 16 (L/n = 1/8).

So the discrete part, if there is one, has to be a factored model: near-boundary gates conditionally independent given
a low-dimensional latent (the collapse modes), or grouped into nearly independent blocks. The central family is then
    z_l = mu + U y + r,   y in R^k with an explicit (cell) density,   r | y with independent coordinates (or blocks),
so that the gates are conditionally independent given y and the next pre-activation z_{l+1,j} | y = sum_i W_ji a_i is a
sum of independent terms (all its cumulants add: exact per-neuron marginals at O(n^2 K) per cell). Its weak point is the
off-diagonal residual covariance: W (off-diagonal part) W^T has a random-matrix spectrum, not a low rank, so a diagonal
residual may get the next layer's per-neuron variances wrong. Test K-F (PREREG amendment 1) measures exactly this: the
per-neuron variance and third cumulant of z_{m+1} (pairwise and triple co-switching along W) and the next layer's mean
under F3, F8, F3C against the full Gaussian and unfactored cells, at width 256.

## 8. The unifying family made explicit: latent-conditional Gaussian (LCG) (v4, 16:25Z; main's proposal)

p(z_l) = integral of N(z; m_l(xi), S_l(xi)) p_l(xi) d xi, with xi a k-dimensional latent (collapse / PCA modes, k = 3-8)
carried with an exact non-Gaussian law (a B^k grid of bins), and the residual Gaussian given xi; near-boundary gates
are probits given xi. Why it would compose: given xi, the next pre-activations are wide sums of the residual (CLT), so
the per-node Gaussian closure is nearly exact, and only the k-dimensional law needs exact propagation (a k-dimensional
Markov chain over layers, axes re-chosen every layer). In cumulant terms: higher-order tensors become low rank through
xi (Tucker core k^3, k^4) plus the per-neuron diagonal from rectification given xi. Dense dependence is confined to the
Gaussian part; the latent is a separator of size k.
Implementation (`lib/lcg.py`): cells = bins of the latent; each layer, (1) re-bin the mixture along the top-k axes of
its total covariance (exact 1-D Gaussian truncations axis by axis, slices in the same bin merged by moment matching),
(2) read out, (3) push every cell with its exact Gaussian closure. The PCA3 oracle cells of sections 6-7 are this family
fitted to the truth at one cut.
Kill tests (main): (1) does the needed k grow with depth; (2) is the residual conditionally Gaussian at near-boundary
neurons (per-neuron excess kurtosis given xi); (3) does the error scale like 1/n_eff of the residual.
Evidence so far (dev nets): on (2), no. Within the 64 PCA3 cells (k = 3, fitted to 2^18 true paths, paired),
the median per-neuron |skew| / excess kurtosis of z given the cell is 0.07 / 0.05 at layer 8, 0.13 / 0.11 at 16,
0.17 / 0.22 at 24, 0.22 / 0.27 at 31 (width 256, net 100), against 0.14-0.47 / 0.15-0.47 unconditionally; at width 64
0.07-0.40 / 0.10-0.56. The residual given xi is not Gaussian and becomes less so with depth. The deductive prototype
at 64 x 16 (net 500): k = 1, 2, 3 with B = 4 give nMSE 2.8e-3, 2.0e-3, 1.65e-3 (Gaussian closure 3.5e-3, K2 3.5e-3,
K3f 7.5e-3, LRG 1.4e-3), at 4-64 cells.

## 9. Status after the pre-registered tests (16:45Z)
- K1' failed by its written rule: passed on net 101 (PCA3 cells, representation 4.8e-6 against 6.8e-5 for one
  Gaussian), failed on net 102 (best 8.1e-5 against 2.0e-4). Candidate A at <= 64 cells is dead at 256 x 32.
- K-F failed on both nets as predicted: conditional independence given <= 8 PCA dims mis-states the next layer's
  variances by 1.6-10% and its one-step mean error is 0.85-8.2x the Gaussian's.
- The deductive LCG (16 re-binned cells) at 256 x 32 is at K=2's level (6.7e-4, 1.0e-3 on nets 100, 101), 2-7x worse
  than K3f; K3f is far better through layer 24.
- Candidate B was not built (deferred by the critique until a forward family passes; none did).
- Untested next step (section 4 of the report): cells built from a cumulant description at a switch layer, first as an
  oracle from the true cumulants at layer 24.
