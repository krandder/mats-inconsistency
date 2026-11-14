# Input-bits lane: theory v2 (31 Oct 2026; v1 frozen 18:07:44Z is `coprover/FORMAL_v1_as_written.md`)

v2 merges gpt-6-astra's round 1 (`coprover/astra_r1_answer.md`, 18:26Z) into v1. Labels: PROVEN, PROVEN UNDER <hyp>,
HEURISTIC, CONJECTURE. Provenance: [C] Claude, [A] astra, [C+A] both independently (`coprover/PROVENANCE.md`).
Measured numbers live in the report and `results/`; this file is theory only.

## 0. Setting and notation
Width n, depth L, W_l iid N(0, 2/n), no biases, x ~ N(0, I_n), h_l = relu(W_l h_{l-1}) + sigma xi_l (l = 1..L-1),
a = relu(W_L h_{L-1}). sigma = 0 unless stated: a(x) = |x| g(u), u = x/|x|, A(u) = |g(u)|, d(u) = g/|g|, m_n = E|x|,
v_n = n - m_n^2 = 1/2 - 1/(8n) + O(n^-2). kappa(c) = (sqrt(1-c^2) + (pi - arccos c) c)/pi, K_l = kappa^(l), c_l = K_l(0),
theta_l = arccos c_l. B(c) = 4 E[Z_+^2 Z'_+^2] - 1 = (2/pi)[(1 + 2c^2)(pi - arccos c) + 3c sqrt(1-c^2)] - 1 (B(0) = 0, B(1) = 5).
C_L = (1/4) sum_{j=0}^{L-1} [5 - B(c_j)] (= Lambda(L)/4 in v1's notation): 1.25, 3.70, 5.47, 7.19, 8.55, 9.47, 10.02, 10.32
at L = 1, 4, 8, 16, 32, 64, 128, 256; C_inf = 10.65.

## 1. Exact identities (PROVEN)
1.1 [C+A] Three-term split, per fixed net: tr Var_x a = V_rad + V_amp + V_dir with V_rad = v_n E A^2, V_amp = m_n^2 Var A =
(m_n^2/2) E(A - A')^2, V_dir = m_n^2 [(E A)^2 - |E A d|^2] = m_n^2 E[A A'(1 - d.d')], all >= 0. Radius-exact MC removes
exactly V_rad. The three terms are a variance identity, not three orthogonal ANOVA components [A].
1.2 [C+A] Gate disagreements: E_{W_l} #{units whose gate differs} = n theta/pi given the layer-(l-1) angle theta (a dead
state contributes n/2 against a live one). With sigma = 0 every gate is a function of x: no births.
1.3 [A] Replica formulas (annealed, weights integrated out; the two-replica Gram chain computes them exactly): with
F_s = E_{W,X,X^[s]}[a(X).a(X^[s])] for inputs sharing exactly s coordinates, V-bar = F_n - F_0,
S_1^ann = n (F_1 - F_0)/V-bar and d^ann = n (F_n - F_{n-1})/V-bar (variance-weighted ensemble first-order share and mean
dimension; not E_W of per-net values).
1.4 [A] Mean Hermite degree: sum_k k H_k(W) = E_x |J_W(x)|_F^2 and E_W of it equals n (criticality), so the annealed mean
Hermite degree of the input-to-output map is exactly n / V-bar. (ANOVA dimension counts distinct coordinates and is smaller.)
1.5 [C+A] Antithetic identity: gain 1/(2 s_even) at equal evaluations; parity, not ANOVA order, sets it.
1.6 [C+A] Stein: degree-1 share W_1 = sum_c |E[a_c x]|^2/V, degree-2 share (1/2) sum_c |E[a_c (x x^T - I)]|_F^2/V; first-order
Sobol share S_1 >= W_1 + diagonal degree-2 part; mean dimension D = sum_j T_j = sum_S |S| V_S/V.
1.7 [C+A] RQMC under Owen-type scrambling: N Var = sum over ANOVA components (and resolutions) of gain coefficients times
component variances [A: resolution-dependent; Sobol shares alone do not determine G]. For RQMC on recent coordinates U with
older noise Z sampled by MC: Var = Var_RQMC(mean g(U_i)) + (1/N) E_U Var_Z f, g = E_Z f [A] (a hybrid floor eta).
For finite dimension and full nested scrambling, square-integrable integrands have variance o(1/N) [A], so a plateau over
2^8-2^16 does not prove a 1/N law for the fully scrambled part.

## 2. The input variance budget
2.1 [C+A] Fixed L, n -> inf (PROVEN [A]): E_W V_rad = v_n, E_W V_amp = C_L + O(1/n), E_W V_dir = n(1 - c_L) + O(1).
2.2 [A] PROVEN: 0 <= E V_amp(l+1) - E V_amp(l) <= (2 m_n^2/n) D_l with D_l = E[A_l A'_l (1 - cos theta_l)] <= 1 - c_l, so
E_W V_amp -> C_inf = 10.65 whenever n, L -> inf, and E_W V_dir <= m_n^2 (1 - c_L) = O(n/L^2). For L/sqrt(n) -> inf:
(E V_rad, E V_amp, E V_dir) -> (1/2, 10.65, 0).
2.3 [C+A] HEURISTIC crossover V_amp = V_dir at L ~ 2.04 sqrt(n) [A] (kernel equation: 28 at n = 256 [A], my 24 [C]; finite-width
chain about 25 [A]). Fixed L, n -> inf: the direction (high order) dominates; fixed L/n: amplitude and radius carry all.
At ARC's 256 x 32 the annealed shares are about 0.04 / 0.59 / 0.37 [A, chain] (my dev nets 0.02-0.04 / 0.55-0.59 / 0.38-0.43).
So the "radius" that carries the input at depth is the amplitude A(u) (input-dependent gains), not |x|.
2.4 [C+A] Per-layer amplitude increments: Delta C_l = (5 - B(c_{l-1}))/4 ~ theta_{l-1}^2 ~ 9 pi^2/l^2: the amplitude keeps collecting
input variance over O(100) layers whatever n is (tail C_inf - C_r ~ 89/r). Not a variance-per-bit law [A]: correlated norm
gains fluctuate even when two paths share their gates; v1's "pi^2 Ham^2/n^3" is only the small-angle bookkeeping.

## 3. Input gate bits (sigma = 0)
3.1 [C+A] Hamming_l ~ n arccos(c_{l-1})/pi ~ 3n/l for 1 << l << n (algebraic decay; finite-angle kernel contraction).
Lyapunov gaps act on infinitesimal differences only; annealed gate counts cannot be assigned the typical Lyapunov
exponent [A].
3.2 [C+A] HEURISTIC Gaussian-cloud closure: H_marg ~ n E_Z h_2(Phi(sqrt(c/(1-c)) Z)) ~ 0.735 theta n = 2.31 x Hamming
(my Gaussian-cap integral ratio 2.606/1.128 [C] is the same constant). Prediction at layer 31 of 256 x 32: 44.5 bits [A];
astra's 8 nets 28.6-57.0 (mean 39.5). Two-replica statistics fix sum 2 p(1-p), not the entropy [A].
3.3 [C] With noise, births ~ n arccos(Q/(Q+1))/pi per layer; negligible below T_mem/2 for a typical radius. (The input regime
is net-dependent: a net whose radius decays faster reaches the noise earlier.)

## 4. Interaction structure over input coordinates
4.1 [A] Fixed L, n -> inf: S_1^ann -> K_L'(0)/(1 - K_L(0)) and d^ann -> 1/(1 - K_L(0)) (the GP limit; 39.6 at L = 32):
not accurate at 256 x 32, where the amplitude matters.
4.2 [A] HEURISTIC amplitude spectrum: the amplitude covariance kernel is C_L(0) - C_L(t) with
C_L(t) = (1/4) sum_j [5 - B(K_j(t))]; its linear coefficient a_1 = -C_L'(0) = 2 sum_j c_{j+1} K_j'(0) = 2.98 (L = 32), 3.095 (inf).
Hence a finite-width degree-1 (additive) part: amplitude S_1 -> 0.29, full map S_1 -> (3.095 + 0.5)/(10.65 + 0.5) = 0.32 at
L = alpha n. Origin: a fixed net has a preferred input direction b ~ (2 pi n)^(-1/2) sum_i |w_i| w_i (first-layer norm),
and later layers add to it [A]. (Replaces v1's CONJECTURE about the low-rank deep Jacobian.)
4.3 [C+A] HEURISTIC mean dimension: direction part ~ Gamma_L / delta_L [C] (= 1/(1 - K_L(0)) at infinite width [A]);
amplitude part ~ 2L/C_L for L << sqrt(n) and ~ 0.999 sqrt(n) for L >> sqrt(n) [A]; so D grows like sqrt(n) at fixed L/n
while a nonzero additive fraction (~0.32) survives [A]. Mean Hermite degree n/V-bar grows like n at fixed L/n (1.4).
4.4 [C+A] Rotations: data-independent rotations cannot help on average; a W-dependent rotation diagonalising the
amplitude's degree-2 matrix moves its off-diagonal degree-2 energy into the additive part; the linear part is additive in
every basis [A].

## 5. Synthesis: what sets a structured method's gain (HEURISTIC; tested in Part 2)
5.1 RQMC: 1/G = sum_k rho_k(N, d) s_k (component efficiencies). Calibration [C]: order-1 removed; order-2 efficiency
rho_2 depends on N/d (1.5, 0.47, 0.10 at N = d, 4d, 16d for d = 256), so at fixed N a growing dimension alone shrinks the
gain. Pair efficiencies depend on the specific projections [A]; a more refined model bins pairs by projection quality and
includes the hybrid floor eta [A].
5.2 Noise regime [A]: frozen-radius closure for the linear share of the output, s_lin(Q) = kappa'(c0)/([Q + 1 - Q kappa'(c0)]
[1 - kappa(c0)]) with (Q+1) c0 = Q kappa(c0): 0.674 (n = 8) ... 0.651 (n = 128) -> 2/3. Flat in width. Antithetic:
s_even ~ 1 - 2^(-1/3) = 0.206 from a (1 - t)^(2/3) noise-stability curve, gain 2.42 (closure).
5.3 Input regime: s_even (x -> -x) -> about 0.43 for the deep amplitude [A] (gain ~1.1).
5.4 Multilevel shared trajectory: MSE = (sum_j sqrt(V_j C_j))^2 / budget; the relevant structure is depth locality of the
amplitude (C_r profile) and the size of the direction residual that needs a full pass, not an ANOVA share [C+A].

## 6. Samplers (HEURISTIC unless stated; tested in Part 3)
6.1 Radius-exact MC is the cheap baseline (gain 1/(1 - s_rad) ~ 1.04-1.05) [C+A].
6.2 [A] PROVEN UNDER layerwise cost hypotheses (unrestricted lane): dyadic shared-trajectory multilevel has MSE
O(log^2 n / budget) at L = alpha n, a width-growing gain Omega(n/log^2 n) over radius-exact MC; no practical constant.
6.3 [A] Exact-mean first-layer control C_1(U) = m_n s_1 q_L [sqrt(q_1) + (Q_1(U) - q_1)/(2 sqrt q_1)], Q_1 = |relu(W_1 U)|^2,
q_1 = |W_1|_F^2/(2n): first-level residual variance O(1/n) under a fourth-moment bound (PROVEN UNDER); no pilot.
6.4 [C+A] Quadratic Hermite controls (exact Gaussian mean) and rotated RQMC target the ~50% of spherical variance in Hermite
degrees 1-2 at L/n fixed [A]: a constant factor (about 2x), not a width-growing gain.
6.5 [A] RQMC on multilevel corrections: with level variances v_j N_j^-p, allocation N_j ~ (v_j/C_j)^(1/(p+1)); no width-uniform
power is established.
