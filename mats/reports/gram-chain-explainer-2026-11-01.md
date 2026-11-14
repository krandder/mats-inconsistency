# The replica Gram chain: what it is, what it computes, what it cannot (1 November 2026)

For Kelvin. Lane gram-chain (bead pool-qxszj1.23). Model S, h_l = relu(W_l h_{l-1}) + xi_l, W_l iid N(0, 2/n), sigma = 1
(sigma is a pure scale). Labels: PROVEN, NUMERICAL (exploratory exact-chain runs; error bars come in the lane report),
HEURISTIC (mean field). Version 2, after a gpt-6-astra check (section 7).

## 1. The object
Take R replicas of one net: they share every weight matrix and differ only in their noise (or, at sigma = 0, in their
input). S_ab is the correlation of replicas a and b's noise at a layer (1 = same vector, 0 = independent). The replica
Gram matrix is G_l[a,b] = h_l^(a) . h_l^(b) / n; Q_a = G_aa (squared radius per unit), rho_ab = G_ab/sqrt(G_aa G_bb),
delta_ab = 1 - rho_ab (one minus the cosine), ell_ab = log|h^(a)| - log|h^(b)|.

## 2. Why it is an exact Markov chain (PROVEN; numerics lane P-G1)
Condition on the past. The rows w_i of W_l are iid N(0, (2/n) I) and independent of it, so the pre-activations across
replicas, z_i = (w_i . h^(a))_a, are iid over units with law N(0, 2 G_{l-1}); the noise vectors are iid over units with
covariance S. Hence

    G_l = (1/n) sum_{i=1..n} v_i v_i^T,   v_i = relu(z_i) + xi_i,   z_i ~ N(0, 2 G_{l-1}),   xi_i ~ N(0, S),

a sum of n iid rank-one matrices whose law depends on the past only through G_{l-1}. The weights are integrated out
exactly, layer by layer, at any n. Cost O(n R^2 + R^3) per layer, against O(n^2 R) for the net. The conditional mean is
the kernel map plus the noise, E[G_l[a,b] | G_{l-1}] = sqrt(G_aa G_bb) kappa(rho_ab) + S_ab with
kappa(rho) = (sqrt(1 - rho^2) + (pi - arccos rho) rho)/pi, and the conditional covariance is (1/n) x the four-point
source: the chain's 1/n expansion over a fixed number of layers is the Roberts-Yaida kernel + vertex recursion with a
noise term (not uniform in depth). Every rotation-invariant function of the replicas' path (radii, overlaps; after
integrating one more layer, e.g. the expected number of gates where a and b disagree, n arccos(rho_ab)/pi) has the same
law under the chain as under the net with random weights. The same argument makes the joint Gram of states and tangent
vectors J u an exact chain (PROVEN).

## 3. What has a stationary law
One input has one in Model S (correcting how this is often put). The radius is Q = (chi2_n/n) r with r an exact
perpetuity (numerics lane); its median is 0.5770780 n to leading order and P(Q > x) ~ C_n/x, C_n = 1/E[A log A] =
2n/5 + 13/75 + O(1/n). What it lacks is a mean: E[Q_l | Q_{l-1}] = Q_{l-1} + 1 exactly, so the stationary mean is
infinite and the kernel recursion (an equation for E Q) and its vertex correction have no fixed point (PROVEN). At
sigma = 0 the only stationary law is Q = 0: an all-off layer (probability 2^-n per layer) kills the state for good.

Replicas add bounded or slowly varying coordinates, the angles and the log-radius ratios. They do not form a Markov
chain by themselves (their transition depends on Q), so their stationary law is a mixture over the radius. The
mean-field closure (normalise the conditional-mean Gram entries; equal radii Q, noise correlation c) is
rho' = (Q kappa(rho) + c)/(Q + 1): a layer's unshared noise opens the angle by (1 - c)/(Q + 1), and the kernel closes a
finite angle, 1 - kappa(1 - delta) = delta - c_k delta^{3/2} - delta^{5/2}/(15 sqrt2 pi) + ..., c_k = 2 sqrt2/(3 pi).
Its fixed point, for c < 1 and Q -> infinity, is

    delta*(Q) = eps [1 - (2/(3(1 - c)) + 1/30) eps + O(eps^2)],   eps = ((1 - c)/(c_k Q))^{2/3} = 2.231 ((1 - c)/Q)^{2/3}

(exact expansion of the closure, checked numerically; within 0.3% of its fixed point at Q >= 148, c = 0). Independent
noise: 2.231 Q^{-2/3}, 3.22 n^{-2/3} at the median radius. Antithetic noise (c = -1): 2^{2/3} times that, to leading
order. Shared noise every layer: replicas merge (from distinct states too, almost surely; stationary lane).

## 4. The stationary law for 2 and 3 replicas
Two replicas, independent noise; three coordinates on three time scales (mean field, HEURISTIC).
- Q: each replica's marginal is the one-input law; it evolves on the scale of n layers.
- delta: fast, relaxing in about 1.49 Q^{1/3} layers (8 at n = 256) toward delta*(Q). Exploratory exact-chain runs
  (1,000-4,000 chains, 6n layers; Q = mean of the two radii): median of delta/delta*(Q) 1.04, 0.99, 1.00 at n = 16, 64,
  256 (with the geometric-mean radius of the lane report: 0.97, 0.96, 0.97, rising to 0.99 at n = 1024); SD 0.76, 0.47, 0.28,
  i.e. about 1.85 n^{-1/3}. The numerics lane's mean ratio falls from 1.12 (n = 8) to 1.00 (n = 1024).
- ell: slow (restored at rate about 1/(Q + 1) per layer), of size n^{-1/3}: SD 1.5-1.8 n^{-1/3} at n = 16-256, and it
  does not shrink at large Q (E[ell^2] n^{2/3} rises from 1.6 to 5.3 as Q/n goes from below 0.3 to above 10, n = 64).
So, given the radius, the angle is pinned near delta*(Q) ~ Q^{-2/3} with relative fluctuations of order n^{-1/3}, and the
two radii differ by order n^{-1/3}.

Three replicas, independent noise: three draws from one net's cloud, so three angles near delta*(Q) that differ by about
sqrt(2/PR) relative (PR the cloud's participation ratio; Gaussian-cloud HEURISTIC): mean (max - min)/mean 0.75, 0.56,
0.39 at n = 16, 64, 256 in the chain against 0.70, 0.50, 0.34 predicted. Update (22:00Z, lane report): PR is not
(log n)^2 as a 1/k spectrum would give; a four-point balance gives PR = n theta*/pi ~ 0.67 n Q^{-1/3} ~ n^{2/3}
(gpt-6-astra, re-derived), and the exact chain follows its radius dependence (PR falls 3.5x from q = 0.2 to 10 at
n = 256, as predicted). So the triangle becomes equilateral like n^{-1/3}.

Mixed sharing: replicas 1, 2 share all noise except layer j, replica 3 is independent. Then E|u1 - u2|^2 / E|u1 - u3|^2
(u = h/|h|) is layer j's total effect on the direction relative to its variance. In mean field one layer's noise opens
1/(Q + 1), and with shared noise delta' = (Q/(Q + 1))(1 - kappa(1 - delta)): algebraic decay
delta_t ~ delta_0/(1 + (c_k/2) sqrt(delta_0) t)^2 at intermediate times, exponential at rate 1/(Q + 1) later; summed over
layers the single-layer effects give 2.99 Q^{1/6} times the variance (numerics lane's interaction-order mechanism).

## 5. What the chain computes exactly
The weight average E_W X(W) of any fixed-weight quantity X(W) = E_noise[phi(Gram path of R replicas)], for a fixed
sharing pattern and any phi that is integrable (polynomials are not required). Examples:
- variance and Sobol quantities of a scale-free observable f (direction u, log|h|, the output direction): E_W Var(f | W)
  from 2 independent replicas; total effect of a block B of layers from replicas sharing all noise except B; closed
  effect E<f, f^(share B)> - E<f, f^(share none)>; mean interaction order sum_B E T_B / E Var for a partition into blocks;
  tr Cov^2 from 4 replicas, so participation ratios as ratios of annealed moments;
- across-net scatter: a quantity built from r replicas needs 2r for its across-net second moment (4 for a variance or a
  total effect), so Var_W of it, and whether it self-averages, are chain quantities;
- noise-stability curves: with noise correlation t (Hermite degrees) or Bernoulli sharing with probability t (ANOVA
  orders), C(t) = 1 - E|f - f_t|^2 / E|f - f_0|^2. Only E S_ab enters the mean-field recursion, so for both the closure
  ([A], input-bits round 1) predicts 1 - (1 - t)^{2/3}: 0.175, 0.370, 0.603 at t = 0.25, 0.5, 0.75, first coefficient 2/3.
  The input-bits lane measured, for scalar-coordinate sharing of the output, 0.16-0.18, 0.36-0.38, 0.60-0.64 at
  n = 8-128, flat in width (a HEURISTIC match, not noted before; the exact chain separates the two kinds of sharing);
- gate statistics: next-layer disagreements and births in closed form given G; cascades by propagation;
- at sigma = 0 with replicas = inputs, the exact multi-input Gram recursion (amplitude/direction split; the 3n/l decay
  of differing gates is its mean-field asymptotic);
- tangents: E_W |J u|^2 = |u|^2 exactly at every depth (PROVEN; criticality), while at large width the typical value
  decays like exp(-2.5 t/n) before an all-off layer kills it: annealed linear-response means are carried by rare nets.
Integrability matters: E Q is infinite and the radius ratio does not shrink at large Q, so E_W tr Cov(h | W) for the raw
state is infinite (n = 64: the top 1% of chains by radius carry 82% of the sample mean, 94% of it radial). Annealed
statements need scale-free observables. For conditionally iid replicas, all finite Gram-array laws together fix the
across-net law of the fixed-weight noise law up to rotation (Gram-de Finetti, Dovbysh-Sudakov).

## 6. What it cannot compute
- Per-net ratios (a fixed-weight Sobol index T_B(W)/V(W), a per-net participation ratio, a median over the noise) are not
  weight averages of a fixed number of replicas. Growing numbers of replicas approach them; moments alone need not. The
  chain gives the means of numerator and denominator and their joint moments, and ratios of annealed means weight
  large-radius nets. Conditioning on a replica's radius does not keep replicas iid given the weights, so it is at most
  an approximation (to be tested).
- Which directions. The future's response to a perturbation of layer l depends on the perturbation only through its
  norm and overlaps with the replicas (W_{l+1} is rotation invariant on its input side given the past). So the chain
  can say how much of a layer-l difference survives and how many directions carry it (spectral moments such as
  tr C^2), never which; that needs the specific weights (a backward pass). This is the kstar-frames lane's Proposition 2
  seen from the future. (Output coordinates are not rotation invariant: E[h' | h] is along the all-ones vector.)
- It averages over the weights: it describes typical and average nets, not a fixed net's mean.

## 7. Check
gpt-6-astra (high effort, 20:51-20:56Z; `work/gram-chain/coprover/explainer_check_answer.md`) reproduced every number
and judged v1 to need major revision (27 findings, 13 errors: overreach on stationary constants, replica observables,
ANOVA and symmetry). All are applied here; the main ones: sigma = 0 law, closure labels, ratios of expected
discrepancies, Hermite vs ANOVA sharing (my layer-block tail claim withdrawn), 2r replicas for across-net moments, the
direction claim limited to layer-l perturbations, integrability (confirmed: infinite raw cloud trace).
