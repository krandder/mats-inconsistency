codex
PROVEN: Model S has a unique pullback sample measure and exact synchronisation. Its stationary radius has an annealed power tail, although every fixed-weight sample measure has exponentially decaying squared-radius tails. At fixed finite width, all infinite-time tangent exponents are `-infinity`; finite exponents require a survival-conditioned interpretation.

The useful finite-depth picture has memory scales proportional to `n`, but it does not justify a Gaussian stationary cloud or a width-uniform estimator gain. Below, I prove the statements I can and isolate the remaining hypotheses.

I use `|.|` for Euclidean norm, `W` for the entire weight sequence, and `E_xi` for expectation conditional on `W`. “Annealed” averages over weights and noises. Assume `sigma > 0` unless stated otherwise. No files were edited, and `work/stationary/` was not read.

**[P1] PROVEN. Model S has a globally attracting pullback point, hence a unique sample measure, without moment restrictions on initial laws.**

Write

```text
F_l(x) = relu(W_l x) + sigma xi_l.
```

Consider disjoint two-layer blocks ending at even indices. If

```text
every entry of W_{2j} is negative,
xi_{2j-1} is coordinatewise nonnegative,
```

then, for every `x in R^n`,

```text
F_{2j}(F_{2j-1}(x)) = sigma xi_{2j}.
```

Indeed, `F_{2j-1}(x)` is nonnegative, so the next preactivation is nonpositive. This block is a constant map on all of `R^n`.

These events are independent across blocks and have probability

```text
p_reset = 2^(-n^2-n).
```

Almost surely, there is such a block at a finite distance before every observation time. Consequently, all sufficiently long pullbacks, from every initial state simultaneously, are exactly equal. Their common value `H_l(W,xi)` satisfies

```text
H_l = F_l(H_{l-1}).
```

It is a stationary, causal function of the driving sequence. Conditioning on weights defines

```text
mu_l^W = Law_xi(H_l(W,xi)).
```

Here is a quenched rate that holds uniformly over all initial probability laws. Let `N_T(W)` count all-negative matrices at the ends of the eligible disjoint blocks in the last `T` layers. Coupling all initial laws with the same noises gives

```text
sup_{nu,nu'} TV(nu K_{-T:0}^W, nu' K_{-T:0}^W)
    <= (1 - 2^(-n))^(N_T(W)).
```

For almost every weight sequence,

```text
N_T(W)/T -> 2^(-n^2)/2.
```

Thus the upper bound decreases exponentially. Annealed, it is at most

```text
(1 - 2^(-n^2-n))^floor(T/2).
```

This proves uniqueness among all probability-valued entrance laws consistent with the conditional transition kernels, including laws without finite moments.

It also proves a random point attractor when noises are fixed: the attraction is exact after a finite random time, uniformly over all initial states. This sufficient reset mechanism has an impractically large mean waiting time, approximately `2^(n^2+n+1)` layers. It is an existence proof, not the relevant practical memory estimate.

For `sigma = 0`, an all-negative matrix following a ReLU layer kills every state. The unique pullback measure is then `delta_0`.

---

**[P2] PROVEN. A fractional-moment contraction gives a practical, order-`n` memory bound.**

Let

```text
M ~ Binomial(n, 1/2),
A = (2/n) chi_square_M,
```

with `chi_square_0 = 0`. For `1 <= p < 2`, define

```text
rho_{n,p} = E[A^(p/2)]
          = (4/n)^(p/2)
            sum_{m=1}^n P(M=m) Gamma((m+p)/2)/Gamma(m/2).
```

Then

```text
E_W |relu(Wx) - relu(Wy)|^p
    <= rho_{n,p} |x-y|^p,
rho_{n,p} < 1.
```

A proof of the uniform inequality is useful. Represent each symmetric Gaussian row as `epsilon_i v_i`, with independent fair signs. Set

```text
d_i = v_i . (x-y),
c_i = |v_i . x| - |v_i . y|.
```

The difference of ReLU outputs in coordinate `i` is

```text
(epsilon_i d_i + c_i)/2,
```

and `|c_i| <= |d_i|`. Conditional on the unsigned rows, the expectation of the `p`th power of the vector norm is convex and separately even in each `c_i`. It is therefore maximised at `|c_i|=|d_i|`. At those endpoints, each squared coordinate is independently either zero or `d_i^2`, with equal probability. Averaging the Gaussian magnitudes gives the stated bound.

Finally, `EA=1`, and strict concavity of `a^(p/2)` gives `rho_{n,p}<1`.

Under synchronous coupling, additive noises cancel, so

```text
E |h_T^x - h_T^y|^p
    <= rho_{n,p}^T E|x-y|^p.
```

The initial states may be random and depend on past weights, provided they are independent of future weights and have the indicated moment.

For fixed `p` and large `n`,

```text
rho_{n,p}
    = 1 - 5p(2-p)/(8n) + O(n^(-2)).
```

In particular,

```text
rho_{n,1} = 1 - 5/(8n) - 23/(128n^2) + O(n^(-3)),
-log rho_{n,1} = 5/(8n) + 3/(8n^2) + O(n^(-3)).
```

This also proves `E|H|^p < infinity` for every `p<2`: construct the pullback from zero and sum the `L^p` norms of successive added-noise contributions.

The same contraction applies to Model R when two trajectories use the same pilot-derived noise schedule. The schedule may depend on weights; its additive contribution still cancels.

---

**[P3] PROVEN. Every quenched stationary law has all moments, and zero-start pullbacks converge exponentially in quenched mean square.**

Fix almost every weight sequence, and find an all-negative matrix `W_b` before the observation time. For any incoming state,

```text
0 <= relu(W_b h_{b-1})
   <= relu(sigma W_b xi_{b-1})
```

coordinatewise, because `h_{b-1}=relu(...) + sigma xi_{b-1}`. Hence

```text
|h_b| <= sigma |W_b|_op |xi_{b-1}| + sigma |xi_b|.
```

Propagating through the finitely many subsequent matrices bounds the observation by a finite weighted sum of Gaussian norms, independently of the state before `b-1`.

Therefore, for some positive, weight-dependent constants,

```text
E_xi exp(c_W |H_0|^2/sigma^2) < infinity,
P_xi(|H_0|^2/(n sigma^2) > x) <= C_W exp(-c'_W x).
```

The same domination holds uniformly for sufficiently long zero-start pullbacks.

Let `H_0^(T)` denote a pullback started at zero `T` layers ago. By [P2],

```text
E_W E_xi |H_0^(T)-H_0| <= rho_{n,1}^T E|H_0|.
```

Markov's inequality and Borel-Cantelli imply, for almost every `W`, an exponential bound on the conditional first moment with any rate below `-log rho_{n,1}`. Interpolating this bound with the uniform conditional moments of arbitrarily high order proves:

```text
For every beta < -log rho_{n,1},

E_xi |H_0^(T)-H_0|^2 <= C_beta(W) sigma^2 exp(-beta T).
```

This is a useful quenched convergence theorem, with an asymptotic admissible rate approaching `5/(8n)`.

The constants are not controlled uniformly over weights. In fact, [P4] shows that the corresponding annealed mean-square truncation error is infinite at every finite `T`.

---

**[P4] PROVEN. The stationary squared radius has an exact scalar perpetuity law and a computable power-tail constant.**

Put

```text
Q_l = |H_l|^2/(n sigma^2).
```

Its annealed transition law is exactly

```text
Q_l = A_l Q_{l-1} + B_l,

M_l ~ Binomial(n,1/2),
U_l | M_l ~ chi_square_(M_l),
V_l | M_l ~ chi_square_(n-M_l),

A_l = 2U_l/n,
B_l = (U_l+V_l)/n.
```

The pairs `(A_l,B_l)` are iid; `A_l` and `B_l` within a pair are dependent.

To prove the transition, condition on the previous radius. Each new coordinate divided by `sigma` has law

```text
sqrt(2q) Z_+ + xi.
```

When `Z<0`, its square is `chi_square_1`. When `Z>0`, symmetry of `xi` shows that its square has the same distribution as

```text
(sqrt(2q) Z + xi)^2 =_law (1+2q) chi_square_1.
```

Summing the independent coordinates gives the recursion.

Useful exact identities are

```text
EA = EB = 1,
EA^2 = 1+5/n,
EAB = EB^2 = 1+2/n,

E[Q_l | Q_{l-1}=q] = q+1,
Var(Q_l | Q_{l-1}=q) = (5q^2+4q+2)/n.
```

The stationary solution is the perpetuity

```text
Q = B_0 + A_0 B_1 + A_0 A_1 B_2 + ...
```

with the pairs indexed into the past. This sum actually terminates at the first zero `A`; nevertheless,

```text
EQ = infinity.
```

Its tail is

```text
P(Q>x) ~ C_n/x,

C_n = 1/m_n,

m_n = E[A log A]
    = sum_{m=1}^n P(M=m) (2m/n)
      [log(4/n) + psi(m/2+1)].
```

Here `0 log 0` means zero.

This uses the scalar implicit-renewal theorem: for a nonnegative perpetuity with `EA=1`, nonarithmetic positive-part `log A`, `0<E[A log A]<infinity`, and sufficiently integrable `B`, the tail constant at exponent one is

```text
E[(AQ+B)-AQ] / E[A log A] = EB / E[A log A].
```

All hypotheses hold here: the positive part of `A` has a continuous density, all positive moments of `A,B` exist, and strict convexity gives `E[A log A]>0`. The numerator is evaluated before taking expectations, avoiding an invalid subtraction of two infinite means.

Literature citation: C. M. Goldie, “Implicit Renewal Theory and Tails of Solutions of Random Equations,” *Annals of Applied Probability* 1 (1991), 126-166; bibliographic details **unverified** in this session.

The constant has expansion

```text
m_n = 5/(2n) - 13/(12n^2) + O(n^(-3)),
C_n = 2n/5 + 13/75 + O(n^(-1)).
```

Direct evaluation gives:

| n | C_n |
|---:|---:|
| 8 | 3.408421 |
| 32 | 12.981018 |
| 256 | 102.574257 |

Thus

```text
E|H|^p < infinity exactly when 0 <= p < 2.
```

For a zero-start pullback,

```text
E|H_0^(T)|^2 = n sigma^2 T.
```

Since the stationary second moment is infinite,

```text
E|H_0-H_0^(T)|^2 = infinity
```

for every finite `T`. This is a substantial obstruction to transferring annealed mean-square results from finite noiseless networks.

---

**[P5] PROVEN. The large-width stationary scale is inverse-gamma, with explicit constants.**

As `n -> infinity`,

```text
Q/n => X,

P(X <= x) = exp(-2/(5x)),   x>0,
density(X) = (2/5) x^(-2) exp(-2/(5x)).
```

In particular,

```text
median Q ~ [2/(5 log 2)] n = 0.577078... n,

median |H| ~ sqrt(2/(5 log 2)) sigma n
           = 0.759657... sigma n.
```

For every fixed `0<p<2`, moments also converge:

```text
E[(|H|/(sigma n))^p]
    -> (2/5)^(p/2) Gamma(1-p/2).

E|H| ~ sqrt(2pi/5) sigma n
     = 1.120998... sigma n.
```

A proof proceeds directly from the perpetuity.

For a fixed time interval, the logarithms of products of `A` satisfy a triangular-array functional central limit theorem:

```text
sum_{i < nt} log A_i
    => sqrt(5) B_t - (5/2)t.
```

Zeros occur with probability at most `nt 2^(-n)` on this interval. Conditional on positivity, the requisite expansions are

```text
E log A = -5/(2n)+O(n^(-2)),
Var(log A) = 5/n+O(n^(-2)),
```

and the Gaussian/chi-square mixture gives the Lindeberg condition. Also, replacing `B_i` by one in the truncated, divided-by-`n` perpetuity has expected absolute error at most

```text
T E|B-1| <= T sqrt(2/n).
```

Thus the truncated perpetuity converges to

```text
integral_0^T exp(sqrt(5) B_t - (5/2)t) dt.
```

To justify removing the truncation, choose `0<s<1` and group the perpetuity into blocks of `n` terms. A fresh divided-by-`n` block has expectation one, hence `s`th moment at most one. Independence of its preceding product gives

```text
E[(tail after Kn terms / n)^s]
    <= sum_{k>=K} (EA^s)^(kn).
```

Since

```text
EA^s = 1 - (5/2)s(1-s)/n + O(n^(-2)),
```

this bound decreases geometrically in `K`, uniformly for large `n`. It also proves uniform integrability for any smaller positive moment.

The limiting exponential functional is the stationary law of

```text
dX_t = dt + sqrt(5) X_t dB_t.
```

The zero-flux stationary equation,

```text
p = (5/2)(x^2 p)',
```

gives the displayed inverse-gamma density after normalisation.

This proof also identifies `n` layers as the scale on which order-one stationary radius fluctuations occur in the large-width limit.

---

**[P6] PROVEN. Stationarity does not mean a constant scale across layers or weight sequences. Sigma only rescales Model S.**

For `sigma>0`, pathwise homogeneity gives

```text
H_l^(sigma) = sigma H_l^(1).
```

Therefore `sigma` is not a shape parameter of the stationary Model S distribution. Small absolute noise does not make its standardised stationary cloud more Gaussian.

For a fresh row, conditional on `H_{l-1}`,

```text
preactivation standard deviation tau
    = sqrt(2/n) |H_{l-1}|
    = sigma sqrt(2Q_{l-1}).

sigma/tau = 1/sqrt(2Q_{l-1}).
```

Thus its typical order is `n^(-1/2)`, independently of `sigma`.

The stationary process is an ergodic function of iid driving variables. Its empirical radius distribution over layers agrees with the annealed stationary distribution. Moreover, writing

```text
qbar_l(W) = E_xi Q_l,
```

we have

```text
qbar_l(W) < infinity a.s.,
E_W qbar_l(W) = infinity.
```

Ergodicity and truncation imply that time averages of both `Q_l` and `qbar_l(W)` diverge to infinity almost surely. Quantiles are therefore more informative than average squared radius.

For the vector mean, the stated nMSE is invariant under rescaling `sigma`. For a tail probability, one must also rescale `t`; additionally, dividing scalar probability error by `tr Var(a_L)` introduces a factor `sigma^(-2)`. That tail score is not invariant under a change of output units.

---

**[P7] PROVEN. At fixed finite width, all true infinite-time tangent exponents are `-infinity`.**

Along a nonzero adapted trajectory, every fresh layer has

```text
P(D_l=0 | past) = 2^(-n).
```

These all-off events are iid Bernoulli events along the trajectory. At their first occurrence,

```text
J_l = D_l W_l = 0,
```

so every tangent product starting earlier is zero thereafter.

In Model S the state is subsequently revived by noise, but the derivative with respect to the earlier state remains zero. In a noiseless network the state itself stays zero.

Thus finite values such as `-5/(4n)` describe a survival-conditioned or finite-depth regime. They are not the infinite-time Lyapunov exponents of the stated finite-width model.

---

**[P8] PROVEN UNDER rank-conditioned layer law H_r. The finite exponents, gaps, and finite-width corrections are explicit, including along noisy trajectories.**

Define `H_r` as the modified process that, at each layer, conditions its fresh weights on at least `r` active gates. For the first `r` tangent exponents,

```text
gamma_k
  = (1/2) [log(4/n)
           + E psi((M-k+1)/2 | M>=r)],
  1 <= k <= r.
```

For fixed `k,r` and large `n`,

```text
gamma_k
    = -(k+1/4)/n
      -(k^2+k+1/24)/n^2
      + O(n^(-3)),

gamma_1 - gamma_k
    = (k-1)/n
      +(k-1)(k+2)/n^2
      + O(n^(-3)).
```

The proof does not require a noiseless trajectory. For a Gaussian row `w`, the outer product `w w^T` is unchanged by `w -> -w`, whereas the gate sign reverses. Consequently, conditional on the past, the gate is a fair coin independent of that row outer product.

For any adapted orthonormal `k`-frame `V`, conditional on `M=m`,

```text
V^T J^T J V =_law (2/n) G^T G,
```

where `G` is an `m x k` standard Gaussian matrix. Gaussian Gram-Schmidt gives independent squared volume factors with degrees of freedom

```text
m, m-1, ..., m-k+1.
```

The expected logarithms are `log 2 + psi(degrees/2)`. Summing iid logarithmic volume increments proves the formula. Applying the argument to a finite basis of each exterior power identifies the ordered singular-value exponents.

The conditioning matters quantitatively: over `T` physical layers,

```text
P(some M_l<r) <= T P(Binomial(n,1/2)<r).
```

For `n=256`, conditioning on `M>=4` gives:

| k | gamma_k | `1/(2|gamma_k|)` |
|---:|---:|---:|
| 1 | -0.004914330 | 101.743 |
| 2 | -0.008882837 | 56.288 |
| 3 | -0.012883225 | 38.810 |
| 4 | -0.016916015 | 29.558 |

Absolute perturbations decay at `gamma_k`. Perturbations relative to the leading direction decay at `gamma_k-gamma_1`. These are different memory questions.

---

**[P9] PROVEN UNDER externally specified linear coefficients. The Lyapunov covariance formula has a narrower scope than the proposed heuristic.**

For a genuinely linear system

```text
Y_l = A_l Y_{l-1} + sigma xi_l,
```

with coefficients fixed independently of the noises and summable products, the conditional stationary covariance is exactly

```text
C_l = sigma^2 sum_{j>=0} P_{l,j} P_{l,j}^T,
P_{l,0}=I,
P_{l,j}=A_l ... A_{l-j+1}.
```

For a constant normal matrix with eigenvalue magnitudes `exp(lambda_k)<1`,

```text
variance_k = sigma^2/(1-exp(2lambda_k))
           ~ sigma^2/(2|lambda_k|).
```

Both the prescribed coefficients and the normal/eigenvector structure are essential. Lyapunov exponents alone do not determine a covariance; products can have substantial transient amplification and rotating singular directions.

In the ReLU system, gates depend on noises. Conditioning on gate histories truncates the Gaussian noise law, so it does not produce the Gaussian linear model above.

There is also an exact obstruction to an annealed covariance calculation:

```text
E[J_l^T J_l | past] = I,
E|J_{l+j} ... J_{l+1} v|^2 = |v|^2.
```

Hence the expected squared tangent influence does not decay with age. Negative typical growth and decay of mean squared growth cannot be interchanged.

---

**[P10] PROVEN. A precise gate-birth definition gives an exact formula.**

Define a “new-noise gate flip” by comparing the actual next gate with the gate obtained after deleting only the most recent injected noise.

Write the previous state as

```text
h = a + s xi,    a=relu(previous preactivation).
```

For a fixed next row `w`, the flip probability is exactly

```text
Phi(-|w.a|/(s |w|)).
```

If `a=0`, take the clean gate to be off; the flip probability is then `1/2`.

Averaging over a fresh Gaussian row, put

```text
R=|a|,
epsilon=s sqrt(n)/R.
```

For `R>0`,

```text
expected flip count = n F_n(epsilon),

F_n(epsilon)
    = E_U Phi(-sqrt(n)|U_1|/epsilon),
```

where `U` is uniform on the unit sphere.

For `n>=2` and small `epsilon`,

```text
F_n(epsilon)
  ~ [sqrt(2) Gamma(n/2)
      /(pi sqrt(n) Gamma((n-1)/2))] epsilon.
```

The bracket tends to `1/pi`.

Under the separate independent-Gaussian signal/noise model,

```text
S~N(0,tau^2), N~N(0,nu^2), epsilon=nu/tau,

P(sign S != sign(S+N)) = arctan(epsilon)/pi.
```

A boundary-band count is different:

```text
P(|S| <= nu)
    = 2Phi(epsilon)-1
    ~ sqrt(2/pi) epsilon.
```

Thus the coordinator's coefficient `sqrt(2/pi)` counts membership in a one-noise-standard-deviation band. It is not the actual gate-flip coefficient.

Also, noise is added before the next matrix multiplication. Its preactivation standard deviation is `s|w|`, approximately `sqrt(2)s`, rather than `s`.

---

**[H11] HEURISTIC, with exact underlying formulas. Typical gate births are order `sqrt(n)` in stationary S and order `n sigma` in a healthy R ensemble.**

For Model S, [P5] and concentration of one-layer norm gains give

```text
epsilon ~ 1/sqrt(nX),
expected flips conditional on the signal radius
    ~ sqrt(n)/(pi sqrt(X)).
```

The corresponding weak distributional statement follows from [P5]. Averaging its limit suggests

```text
E[number of flips] ~ sqrt(5/(8pi)) sqrt(n)
                   = 0.446031... sqrt(n).
```

This averaged asymptotic additionally requires control of very small signal radii; I have not supplied that uniform-integrability proof. The analogous boundary-band constant is

```text
sqrt(5)/2 = 1.118034...
```

multiplying `sqrt(n)`.

In Model R, define the sample's relative radius

```text
U = |a|/(sqrt(n) r).
```

Then exactly

```text
epsilon = sigma/U.
```

For samples with `U` close to one, the Gaussian approximation predicts

```text
births ~ n arctan(sigma)/pi.
```

At `n=256`, this is approximately `0.08149`, `0.81485`, and `8.12173` for `sigma=0.001`, `0.01`, and `0.1`.

Small `U` is an explicit failure condition for this approximation. If the signal is zero, the count is `n/2` regardless of how small the nominal relative-noise parameter is.

---

**[P12] PROVEN. Noise-age derivatives are explicit, but they do not give a summable annealed influence budget.**

Away from gate boundaries, for noise injected `j` layers before `h_L`,

```text
d h_L / d xi_{L-j}
    = sigma J_L ... J_{L-j+1},
```

with the product equal to `I` at `j=0`.

For every finite age,

```text
E |d h_L / d xi_{L-j}|_F^2 = n sigma^2.
```

The product is nevertheless zero if an all-off gate occurs in that interval. Along a realised stationary trajectory, the backward age of the latest all-off layer has a geometric distribution with mean `2^n`.

These facts coexist because rare surviving products have large squared magnitudes.

An all-off layer removes local derivatives of earlier noise. It need not remove finite counterfactual effects: changing old noise enough can change the gate that was all-off. Exact dependence on the remote past is erased by a global coalescence event such as [P1].

There is a rigorous alternative to counting derivatives. At fixed weights, define effective dimension by how many recent Gaussian coordinates suffice to approximate the output in mean square. From [P3], a sufficient number is

```text
n T,

T >= beta^(-1)
     log(C_beta(W) sigma^2/(epsilon V_W)),
```

where `V_W=tr Var(a_L|W)` and the desired error is `epsilon V_W`.

Since admissible `beta` approaches `5/(8n)`, this gives a coefficient of order `n^2` multiplying the logarithmic accuracy requirement. It is an upper bound using whole recent noise blocks, not a minimal intrinsic dimension.

---

**[H13] HEURISTIC. The directional age profile and live-bit budget need additional assumptions.**

If one substitutes the finite exponents into a fixed, approximately normal linear model, the directional energy profile is

```text
energy(k,j) proportional to exp(2 gamma_k j),
memory_k approximately 1/(2|gamma_k|)
                 approximately n/[2(k+1/4)].
```

The age participation ratio of one such geometric profile is

```text
( sum_j exp(2gamma_k j) )^2
 / sum_j exp(4gamma_k j)
    = coth(|gamma_k|)
    approximately 1/|gamma_k|.
```

Summing these modewise effective counts gives order `n log n`. This calculation assumes a mode decomposition that can be used across ages; the nonlinear random cocycle does not supply it automatically.

A live-bit count additionally needs a rule assigning gate events to persistent modes. For example:

```text
birth rate b, all events assigned lifetime O(n):
    budget O(b n);

birth rate b, events spread uniformly over O(n) modes:
    average lifetime O(log n),
    budget O(b log n).
```

These give different predictions, such as `n^(3/2)` versus `sqrt(n) log n` in Model S. Neither assignment has been established.

A large number of gates is insufficient for a CLT: their effects share the same state, radial gain, and future matrices. The stationary inverse-gamma radius is an explicit non-Gaussian collective fluctuation that survives large width.

For a rigorous variance decomposition, use the Gaussian input/noise coordinates and their Hoeffding-ANOVA decomposition. Input-only terms, noise-only terms, and their interactions are orthogonal. An additive “input bits plus noise bits” budget is justified only if the interaction terms are negligible.

---

**[P14] PROVEN. Model R has an exact scale/shape decomposition; global input scale and individual-input memory must be distinguished.**

Let

```text
q_l = E_xi,x |h_l|^2/n,
r_l^2 = E_xi,x |relu(W_l h_{l-1})|^2/n.
```

Then exactly

```text
q_l = (1+sigma^2) r_l^2.
```

Set `Z_l=h_l/sqrt(q_l)` and

```text
g_l^2 = E|relu(W_l Z_{l-1})|^2/n.
```

The recursion becomes

```text
q_l = (1+sigma^2) g_l^2 q_{l-1},

Z_l = relu(W_l Z_{l-1})/[sqrt(1+sigma^2) g_l]
      + sigma xi_l/sqrt(1+sigma^2).
```

Thus the normalised ensemble has fixed injected-noise scale, while

```text
log sqrt(q_L)
  = log sqrt(q_0)
    + sum_l [log g_l + (1/2)log(1+sigma^2)].
```

Rescaling the entire input ensemble by a positive constant rescales every `r_l` and every state by that same constant. Global input scale therefore persists exactly. This rules out a unique nonzero absolute-scale attractor across all initial scales.

Moreover,

```text
E[q_l | past] = (1+sigma^2) q_{l-1},
q_l/(1+sigma^2)^l is a nonnegative martingale.
```

No stationary absolute-scale law with a finite, positive annealed second moment can satisfy this recursion.

For an individual sampled input, the situation differs. Once the pilot schedule is fixed, changing that input does not change the noise amplitudes. Synchronous coupling then obeys [P2], and the reset mechanism of [P1] eventually erases its effect exactly. Positive noise does not preserve each input's radius by homogeneity.

Here `sigma` is a genuine parameter: rescaling states cannot change its ratio to ensemble signal.

---

**[C15] CONJECTURE / conditional results. Stationarity of Model R's normalised shape and its precise memory times remain unresolved here.**

A natural conjecture is uniqueness and attraction of a stationary normalised ensemble process for each fixed `n` and `sigma>0`. The displayed normalised recursion alone does not prove it.

PROVEN UNDER a stationary ergodic normalised-shape process and an additive-functional CLT: if

```text
zeta_l = log g_l + (1/2)log(1+sigma^2),
E zeta_l = gamma_R,
```

then

```text
log sqrt(q_L)
    = log sqrt(q_0) + gamma_R L + O_probability(sqrt(L)).
```

A positive CLT variance gives continuing log-gain fluctuations. There is no restoring term in absolute log scale.

For `sigma=0`, before death, the single-path log-radius increments are iid with

```text
mean = (1/2)[log(4/n)+E psi(M/2 | M>0)]
     = -5/(4n)-49/(24n^2)+O(n^(-3)),

variance = (1/4)[E psi_1(M/2 | M>0)
                 + Var(psi(M/2) | M>0)]
         = 5/(4n)+O(n^(-2)).
```

The survival probability through `L` layers is `(1-2^(-n))^L`. Local relative tangent modes have times approximately `n/(k-1)` under [P8].

For noisy R, a normalised perturbation's decay rate depends on the difference between its tangent growth rate and the ensemble scale growth rate. Neither a universal `n` nor a universal `1/sigma^2` radius-memory time follows from the supplied heuristic. Establishing `gamma_R` and the normalised-shape dynamics is the missing step.

---

**[P16] PROVEN, with a separate typical-path heuristic. Model S forgets an initial radius `sqrt(n)` after an order-`n` logarithmic transient.**

Couple an input-started process, `x~N(0,I_n)`, with the stationary process using identical future weights and noises. Then

```text
E|h_L^x-H_L|
    <= rho_{n,1}^L [sqrt(n)+E|H_0|].
```

Using [P5],

```text
P(|h_L^x-H_L| > epsilon sigma n)
 <= rho_{n,1}^L
    [1/(sigma sqrt(n)) + sqrt(2pi/5)+o(1)]/epsilon.
```

Thus a sufficient depth for failure probability at most `delta` is

```text
L >=
 log( [1/(sigma sqrt(n))+sqrt(2pi/5)+o(1)]
      /(epsilon delta) )
 / (-log rho_{n,1}).
```

Its order is

```text
n [1 + log_+(1/(sigma sqrt(n)))
     + log(1/(epsilon delta))].
```

HEURISTIC: matching a typical noiseless decaying radius to the stationary radius gives, when `sigma sqrt(n)<<1`,

```text
sqrt(n) exp(-5L/(4n)) approximately sigma n,

L approximately (4n/5) log(1/(sigma sqrt(n))),
```

followed by an order-`n` equilibration interval. This is a typical-path estimate, whereas the preceding inequality controls a specified probability.

Below this depth, input and noise effects interact through gates. There is no proved rule that independently decaying input bits can simply be added to a stationary noise-bit count.

---

**[P17] PROVEN. The fixed-weight MC baseline and its normalisation are simple.**

For `N` independent exact samples at the same observation layer,

```text
E[nMSE of vector sample mean | W] = 1/N.
```

For the tail probability `p_W`,

```text
MSE = p_W(1-p_W)/N.
```

Under the requested normalisation, divide this by `V_W=tr Var(a_L|W)`.

Sampling different layers of one stationary trajectory generally estimates an annealed/time-averaged target, not the particular `mu_L^W`. Samples for a quenched target must retain the fixed weight sequence.

All costs below include other arithmetic in equivalent `2n^2`-flop units.

---

**[P18] PROVEN UNDER the stated variance-decay bounds. Burn-in multilevel sampling removes the logarithmic precision overhead, but no growing width gain follows yet.**

Let `F_T` be the output from a zero-start pullback of length `T`. Couple approximations by sharing their recent noises. From [P3], for vector outputs and almost every fixed `W`,

```text
E_xi |F_T-F_infinity|^2 <= K_W exp(-beta T).
```

For an indicator at a positive threshold, conditional Gaussian smoothing gives local anti-concentration. Combining

```text
P(indicators differ)
 <= P(|F_T-F_infinity|>d)
    + P(||F_infinity|-t|<=d)
```

with the contraction gives an exponential bound too, generally with a smaller exponent. Constants depend on the final weights and threshold.

For ordinary independent burn-in MC, choosing `T` proportional to `log C/beta` gives

```text
nMSE = O_W(beta^(-1) log(C)/C).
```

For multilevel sampling, take dyadic depths `T_j`, increments

```text
Delta_j = F_(T_j)-F_(T_(j-1)),
```

and costs `c_j=O(T_j)`. If `v_j` denotes their normalised variance, optimal allocation gives asymptotic cost-error constant

```text
K_ML(W) = (sum_j sqrt(v_j c_j))^2.
```

Exponential decay makes this sum finite, yielding

```text
nMSE = O_W(1/C).
```

An unbiased random-level version also follows directly: sample level `j` with probabilities proportional to the square root of its second moment divided by its cost, and return the appropriately weighted telescoping increment. The same summability guarantees finite variance and finite expected cost.

If the normalised decay constants were uniformly bounded and `beta` were order `1/n`, this calculation would give `K_ML=O(n)`. Those uniform bounds have not been proved here.

Exact CFTP also has exponent one, with constant proportional to its expected sampling cost. The explicit global reset construction gives an annealed expected cost of order `2^(n^2+n)`, so it is principally a correctness device.

---

**[P19] PROVEN. Sharing a deep past has an exact variance tradeoff and can create a bias-like variance floor.**

Let `Z` be the shared past state and `F(Z,U)` the final output. Define

```text
v_between = tr Var(E[F|Z]),
v_within  = E tr Var(F|Z).
```

With `M` independent pasts and `K` conditionally independent continuations per past,

```text
MSE = v_between/M + v_within/(MK).
```

If past generation costs `c_p` and each continuation costs `c_f`, the cost-error product is

```text
(c_p + K c_f)(v_between + v_within/K).
```

Ignoring integer constraints, the optimum is

```text
K = sqrt(c_p v_within/(c_f v_between)).
```

Using one shared past forever leaves error `v_between`. Multilevel corrections or a retreating cut are needed to remove it.

---

**[P20-H23] The remaining estimator families have the following qualified conclusions.**

| Method | Label and MSE-vs-compute conclusion | Width-dependent constant |
|---|---|---|
| RQMC on a fixed finite noise vector | PROVEN UNDER standard scrambled-net hypotheses: `o(C^(-1))` for square-integrable integrands; no universal additional power follows | Requires dimension/regularity control |
| RQMC with smooth mixed derivatives | PROVEN UNDER the applicable smoothness theorem: rates approaching `C^(-3)` up to logarithms are possible | Those hypotheses are not established for these Gaussian-transformed ReLU paths |
| Increasing stratification, fixed finite depth | PROVEN as detailed below: mean `C^(-1-2/d)` and regular indicator `C^(-1-1/d)`, up to logarithms | `d` grows with the number of input/noise coordinates |
| Fixed control variate | PROVEN: exponent one, constant equal to residual variance times cost | Improves with width only if the charged residual variance does |
| Fixed near-boundary stratification or importance sampling | PROVEN with finite importance-weight second moment: exponent one | Can improve a rare-event constant; no automatic width gain |
| Fixed-order deterministic small-noise expansion | PROVEN: a nonzero truncation bias produces an error floor as `C -> infinity` | Small `sigma` does not cure this at fixed positive `sigma` |
| Increasing-order deterministic quadrature with exact boundary partition | PROVEN UNDER a fixed-dimensional smoothness/error bound of order `r`: squared error `O(C^(-2r/d))` | Partition count and constants can grow severely with width/depth |
| Increasing-quality controls or hybrids | PROVEN UNDER a learnable-control hypothesis: if cost `C_0` gives exactly integrable control residual variance `O(C_0^(-alpha))`, with bounded evaluation cost, then allocating comparable setup and sampling budgets gives `O(C^(-1-alpha))` | The approximation-cost hypothesis is the substantive requirement |

Citation for the scrambled-net statements: Owen's scrambled-net variance/convergence results; exact theorem references **unverified** here. I have not verified their stronger smoothness hypotheses for this problem.

For the fixed-depth stratification claim, truncate the Gaussian coordinates to a cube of radius `O(sqrt(log C))`, controlling the omitted bias. Partition into `N` equal spatial cells and sample conditionally within each cell using its exact Gaussian probability.

For a Lipschitz vector integrand in dimension `d`, cell diameter is `O(N^(-1/d))`, each cell probability is `O(N^(-1))`, and therefore

```text
variance <= sum_cells probability^2 * O(diameter^2)
         = O(N^(-1-2/d)),
```

up to truncation logarithms. For an indicator with a finite union of regular hypersurface boundaries, only `O(N^((d-1)/d))` cells intersect the boundary, giving `O(N^(-1-1/d))`.

This proves that exponent one can be beaten at fixed finite depth. It does not establish the same exponent for stationary S: there, the truncation dimension grows with requested accuracy.

For any finite stratification with stratum probabilities `p_s`, variances `V_s`, and sample costs `c_s`, optimal allocation gives the constant

```text
(sum_s p_s sqrt(V_s c_s))^2.
```

That formula makes the necessary variance reduction explicit.

---

**[P24] PROVEN UNDER a smooth boundary-density hypothesis. Small-noise linearisation has a computable single-hinge residual.**

Let `Z` have a density continuous at zero, independently perturb it by `nu G`, and define

```text
R = relu(Z+nu G) - relu(Z) - nu G 1[Z>0].
```

With sufficient local domination for passage to the limit,

```text
ER ~ (1/2) f_Z(0) nu^2,

ER^2 ~ [2sqrt(2)/(3sqrt(pi))] f_Z(0) nu^3.
```

For a fixed perturbation `u`, the residual is supported on the crossing interval of length `|u|`, and direct integration gives

```text
integral R(z,u) dz = u^2/2,
integral R(z,u)^2 dz = |u|^3/3.
```

Averaging over the Gaussian perturbation proves the constants.

Propagating this estimate through a network requires control of downstream amplification, multiple crossings, and atoms at dead states. An atom at zero can change the orders to `nu` and `nu^2`.

For stationary S, scaling `sigma` scales the nominal state and boundary distances as well. Consequently, this calculation does not provide a vanishing standardised control residual as `sigma -> 0`.

One immediately available exact control is final-noise integration. Conditional on the preceding clean state, each output mean is

```text
E relu(m+nu G) = nu phi(m/nu) + m Phi(m/nu),
```

where `m=w.a` and `nu=s|w|`. This is a valid Rao-Blackwellisation for both models and removes one noise block at modest cost. It does not require independent output coordinates.

---

**[C25] CONJECTURE / unresolved. A stationary analogue of the `Omega(n/log^2 n)` gain is not established by these arguments.**

The permitted report's section 3.7 uses noiseless positive homogeneity, directional merging, and a reference continuation shared across inputs.

Stationary S introduces fresh randomness at every layer, has an annealed infinite second moment, and has no annealed mean-square finite-past approximation. Those facts invalidate a direct transfer of that argument.

There is a rigorous quenched analogue in the weaker sense of [P18]: multilevel sampling over past depth attains finite variance-cost product and removes the burn-in logarithm. A gain growing with `n` would additionally require a width-uniform bound on its normalised increment variances, or an effective shared continuation/control that integrates substantial recent randomness cheaply. I have not proved either.

The present results therefore support testing structured sampling, but do not establish an optimal exponent or a width-growing advantage for stationary networks.

The following checks were performed in memory on 2026-10-31: direct evaluation of the gamma/digamma constants; comparison of the exact radial transition with direct Gaussian-ReLU simulation; and radial-chain quantiles at widths 16, 64, and 256. For example, at `n=256`, simulated quartiles of `Q/n` were approximately `0.2875, 0.5717, 1.4009`, against limiting values `0.2885, 0.5771, 1.3904`. These are checks of formulas, not substitutes for the proofs.

| Claim | Label | Cheapest numerical check |
|---|---|---|
| P1: pullback uniqueness and exact synchronisation | PROVEN | Force one all-negative matrix and nonnegative preceding noise; compare arbitrary initial states |
| P2: fractional contraction and its constants | PROVEN | Sample one-layer coupled pairs, including collinear and opposite inputs |
| P3: quenched moments and mean-square convergence | PROVEN | Fix one weight stack; couple increasing burn-in depths |
| P4: radial perpetuity and tail constant | PROVEN | Simulate scalar `(A,B)` recursion; inspect `x P(Q>x)` |
| P5: inverse-gamma width limit | PROVEN | Compare quantiles of `Q/n` across widths |
| P6: scale invariance and varying quenched scale | PROVEN | Reuse identical driving variables at two `sigma` values |
| P7: true exponents are `-infinity` | PROVEN | Track Jacobian products until the first all-off mask |
| P8: finite spectrum and corrections | PROVEN UNDER H_r | QR iteration while rejecting masks with fewer than `r` active units |
| P9: covariance formula | PROVEN UNDER externally specified linear coefficients | Compare a frozen linear covariance recursion with empirical covariance |
| P10: exact gate-flip formula | PROVEN | Delete/resample only the latest noise and count changed gates |
| H11: averaged stationary birth constant | HEURISTIC | Scalar-radius sampling plus spherical gate-flip integration |
| P12: derivatives and finite-block approximation | PROVEN | Automatic differentiation by noise age; paired truncation errors |
| H13: directional/live-bit budgets | HEURISTIC | Measure variance explained by recent blocks and interaction terms |
| P14: Model R scale/shape decomposition | PROVEN | Rescale the entire input ensemble and pilot consistently |
| C15: normalised R stationarity | CONJECTURE | Couple two normalised ensemble recursions using common weights |
| C15: log-gain CLT conclusion | PROVEN UNDER stationary-shape CLT | Estimate drift and variance of log-scale increments |
| P16: input-forgetting bound | PROVEN; typical coefficient HEURISTIC | Couple input-started and stationary paths over `n,sigma` |
| P17: MC risk | PROVEN | Repeat independent fixed-weight sample means |
| P18: multilevel exponent and constant | PROVEN UNDER stated decay bounds; bounds proved quenched here | Estimate each level's `variance * cost` and their square-root sum |
| P19: nested sharing tradeoff | PROVEN | Estimate between-past and within-past variances |
| P20-H23: RQMC, stratification, deterministic and hybrid rates | PROVEN UNDER stated hypotheses; stationary extensions unresolved | Fix dimension first; plot risk against fully charged cost |
| P24: hinge residual constants | PROVEN UNDER boundary-density hypothesis | One-dimensional Gaussian hinge integration |
| C25: growing-width stationary estimator gain | CONJECTURE / unresolved | Compare normalised multilevel variance-cost constants across widths |
