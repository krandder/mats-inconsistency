## 2. Random narrow-deep nets (n = 8 and 16, L = 64-256)

Nets (`lib/nd.py`):
- **Plain**: ARC's critical He-ReLU form: z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps_l, W_l iid
  N(0, 2/n), no bias.
- **Residual**: u_l = W_l h_{l-1}, h_l = h_{l-1} + V_l (ReLU(u_l) + s_l eps_l) / sqrt(L), with V_l iid N(0, 1/n).
- Noise s_l = sigma x (RMS of layer l's clean activation); sigma = 1e-3 unless stated.
- 64 nets per width for the census, 4 per configuration otherwise. n = 8 per-net analyses use nets that are alive at
  L (Amendment 1).

### 2.1 Collapse at initialisation: inputs synchronise, then narrow nets die whole

| plain, sigma | median angle between two inputs at depth 4 / 16 / 64 / 128 / 256 (rad) | gate disagreement between two inputs, same depths | nets (of 64) with every input dead by L = 64 / 128 / 256 |
|---|---|---|---|
| n = 8, 0 | 0.62 / 0.015 / 0 / 0 / 0 | 25% / 4% / 0 / 0 / 0 | 10 / 25 / 39 |
| n = 8, 1e-3 | 0.62 / 0.026 / 0.009 / 0.009 / 0.008 | 25% / 4.8% / 1.4% / 2.3% / 1.9% | 10 / 21 / 37 (> 95% dead) |
| n = 16, 0 | 0.72 / 0.15 / 3.1e-3 / 1.4e-5 / 0 | 27% / 8.3% / 0.67% / 0.08% / 0 | 0 / 0 / 0 |
| n = 16, 1e-3 | 0.72 / 0.15 / 0.013 / 0.012 / 0.008 | 27% / 8.3% / 1.2% / 1.1% / 1.0% | 0 / 0 / 0 |
| residual n = 16, L = 256 | 1.6 / 1.6 / 1.5 / 1.5 / 1.3 | 50% / 50% / 48% / 46% / 43% | (no path death in a residual net) |

Read: in the plain nets every input is carried onto one trajectory set by the weights. For comparison, wide nets
(256 x 64) still disagree on 6-10% of gates per layer at depth 32 (lyapunov lane).
- **Rate (measured, matches a proven single-path exponent).** Beyond the first few layers the pair angle shrinks
  exponentially. Fitted log-slope of the median angle, layers 32-128, over 64 nets: n = 16, -0.0867 per layer
  (median -0.0862).
  - The half-width Lyapunov gap between the radial and the first angular exponent of one path is 0.0864
    (formula below). This is Theorem C / Corollary C1 of the lyapunov lane applied to an infinitesimal separation:
    gap_1 = (1/2) E[psi(m/2) - psi((m-1)/2)], m ~ Bin(n, 1/2), about 1/n for large n.
  - Jakub and Nica's large-n rate 1/n gives 0.0625 at n = 16.
  - n = 8: -0.26 per layer over layers 8-40, against gap_1 = 0.214. The extra speed comes from exact
    synchronisation: when a layer leaves one active unit, every input on that unit leaves it with the same
    direction. In 26 of 61 nets the median pair was exactly synchronised by depth 32.
- **Collective death (exact mechanism, measured rate).** After synchronisation all inputs share each layer's gates.
  When the shared trajectory meets a layer with no active unit, the whole net outputs 0. A single path dies at each
  layer with probability 2^-n, independently of the past (Theorem C's proof). The mean over 64 nets of the fraction
  of dead inputs is:
  - 0.19 / 0.41 / 0.63 at depth 64 / 128 / 256 (n = 8);
  - against 1 - (1 - 2^-8)^L = 0.22 / 0.39 / 0.63.
  - Per net the fraction is bimodal: at L = 256, 20 nets have < 5% dead, 39 have > 95%, 5 are in between. This is
    Lu et al.'s "born dead" collapse, here without biases.
  - n = 16 loses 0.4% of paths by 256.
  - With sigma = 1e-3 a net that dies whole is partly revived by the noise of the layers where a few paths survived.
    Its output is then mostly noise: in such nets 53-89% of the output variance is noise, against < 1% in the others.
- **Residual: no collapse.** Gates disagree on 43-50% of units per layer at every depth, and pair angles stay at
  1.3-1.6 rad.

### 2.2 Per-path surviving directions against remaining depth

N_surv(J; delta) counts the singular values of the path Jacobian J_x = d a_L / d z_c within delta of the largest
(1,024 paths per net and cut). Plain nets, sigma = 1e-3, L = 256, mean over 4 nets:

| L' (layers to go) | 4 | 8 | 16 | 32 | 64 | 128 | 192 |
|---|---|---|---|---|---|---|---|
| n = 16: N_surv(0.01) | 5.0 | 3.9 | 2.5 | 1.7 | 1.3 | 1.1 | 1.0 |
| n = 8: N_surv(0.01) | 2.4 | 1.8 | 1.4 | 1.0 | 1.0 | 1.0 | 1.0 |
| n = 8: paths whose Jacobian has rank 1 | 0.01 | 0.26 | 0.33 | 0.80 | 0.92 | 1.0 | 1.0 |

- Only the radial direction survives (Euler: J a = output). s_2/s_1 decays at 0.089-0.098 per layer at L' >= 64 for
  n = 16, against the predicted 0.086.
- rank(J_x) <= min_l m_l(x) exactly, where m_l is the number of active units. At n = 8 almost every path passes a
  layer with at most one active unit by L' = 64.
- Residual nets keep every direction: N_surv(0.01) = n at every L'. The spread grows slowly: N_surv(e^-1) falls from
  n to about 0.35 n by L' = 250.

### 2.3 The union a fixed block needs

M = E_x J_x^T J_x is the sensitivity that one block for all inputs must cover (Theorem B/D of the lyapunov lane).
k*_F(M; 1e-2) is the number of directions that leaves at most 1% of tr M outside the block. S = Sigma^(1/2) M
Sigma^(1/2) weights M by the state's spread. Plain nets, sigma = 1e-3, L = 256:

| cut c (layer) | 4 | 8 | 64 | 128 | 224 | 248 |
|---|---|---|---|---|---|---|
| n = 16: k*_F(M; 1e-2) / k*_F(S; 1e-2) | 12.2 / 7.0 | 10.5 / 4.2 | 3.0 / 1.2 | 2.5 / 1.0 | 2.8 / 1.2 | 4.0 / 1.2 |
| n = 8 | 5.0 / 3.5 | 3.8 / 2.2 | 2.5 / 1.2 | 2.5 / 1.2 | 3.0 / 1.0 | 2.2 / 1.0 |

- After synchronisation a fixed block needs 1-3 directions; at the input cut it needs most of the space. This is the
  opposite of the wide nets, where 185 of 195 directions were needed at depth 32.
- Residual nets need all n at every cut: k*_F(M; 1e-2) = n, and PR(M) is 9-16 at n = 16.

### 2.4 Kelvin's hope: a bound from the weights alone on which directions can matter

Candidates at a cut, k = 1 direction. The table gives the share of the sensitivity tr M that each candidate captures;
the optimum is the top eigenvector of M.
- **lin**: the unmasked product W_L ... W_{c+1};
- **rand**: random masks, E_D J_D^T J_D with D_l iid Bernoulli(1/2);
- **worst case over masks**: PR of the top directions over 1,024 random mask sequences, and the gain bound
  prod ||W_l||;
- **avg**: the product with each layer's marginal gate probabilities;
- **att**: the Jacobian along one reference input's path (one forward pass);
- **pcs**: the state's top principal component, for reference.

| plain, sigma = 1e-3 | optimum | lin | rand | avg | att | pcs | PR of top directions, random masks | log gain: actual median / mask-free bound |
|---|---|---|---|---|---|---|---|---|
| n = 16, L = 256, c = 4 (before sync) | 0.52 | 0.12 | 0.08 | 0.19 | 0.12 | 0.05 | 13.0 of 16 | -17.8 / +242 |
| n = 16, L = 256, c = 64 | 0.93 | 0.04 | 0.08 | 0.93 | 0.91 | 0.11 | 12.8 | -11.8 / +184 |
| n = 16, L = 256, c = 128 | 0.82 | 0.08 | 0.19 | 0.80 | 0.79 | 0.07 | 13.0 | -7.4 / +123 |
| n = 16, L = 64, c = 32 | 0.76 | 0.13 | 0.10 | 0.70 | 0.60 | 0.03 | 12.9 | -1.1 / +31 |
| n = 8, L = 256, c = 128 | 0.96 | 0.36 | 0.26 | 0.87 | 0.65 | 0.05 | 3.8 of 8 | -20.8 / +113 |
| n = 8, L = 64, c = 32 | 0.83 | 0.15 | 0.05 | 0.66 | 0.50 | 0.12 | 6.5 | -6.3 / +28 |

**What fails.**
- No gate-free object predicts the relevant directions. The unmasked product and random masks capture 4-36% of the
  sensitivity, near chance (1/n) at n = 16. The top directions under random masks spread over 4-13 of n dimensions,
  so no small subspace contains them.
- The mask-free gain bound exceeds the actual gain by e^30 to e^240.
- Reason: a path's masks are not free. They are tied to its own state (ReLU(Wh) = D W h, with D chosen so that the
  result is nonnegative). That tie makes the state's own direction the radial, dominant one, and a mask-free object
  has no access to it.

**What holds.**
- The masks of a single trajectory are enough once the inputs have synchronised. Its Jacobian (att) or the marginal
  gate probabilities (avg) capture the optimum's share to within 0.01-0.1 at n = 16 for c >= 64.
  - At n = 8, avg works (0.87 vs 0.96) but att varies by net: 0.07 on one of four nets at c = 128, because that
    reference path was not on the shared trajectory.
  - Both are computable from the weights plus one forward pass, so the bound is "weights plus one trajectory's
    gates".
  - It holds only after the synchronisation depth, about n ln(theta_0/theta) / (n gap_1) layers, roughly n per e-fold
    of angle. Before it (cut 4), nothing short of each input's own gates works: the best candidate captures 0.19
    against an optimum of 0.52.
- In residual nets nothing is concentrated. The optimum itself captures 8-40% at k = 1, and every candidate is within
  a few points of it. There is no small subspace to find, but the whole state is only n-dimensional.

**The prereg's own score.** The pre-registered share of tr S makes every candidate look perfect after
synchronisation: tail 0.000 for lin, rand, avg and att at c >= 64 (n = 16). The state is one-dimensional there, and
any block not orthogonal to it captures it. By that rule a "gate-independent bound exists" at deep cuts, but only
because the state has collapsed, not because the weights identify sensitive directions. We report the sensitivity
share above as the informative score (Amendment 1).
