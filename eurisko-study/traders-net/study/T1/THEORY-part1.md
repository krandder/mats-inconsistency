# THEORY, part 1 (T1 worker 1): lemmas 1, 2 and 3

Checked 2026-09-27 ~22:29Z. Scripts and outputs: `study/T1/w1/` (`PREREG.md`, `check_L1.py`,
`check_L2.py`, `check_L3.py`, the importable `costshare.py`; results in `out/L1a.json`, `out/L1b.json`,
`out/L1c.json`, `out/L1d.json`, `out/L2.json`, `out/L3.json`, `out/flops_M7.json`, `out/flops_K1.json`).
Toys: a float64 copy of wealth-net's `Net` (Q=16, W=8, d=16, H=2, L=2) on the wealth-net streams
(`w1/streams.py`, byte-identical to `study/S0/substrate/streams.py`); a one-hidden-layer ReLU MLP; a
logistic regression with a known posterior; and closed-form LMSR and cost games. Seeds 1-3. Numbers
are mean ± sd over the 3 seeds, or, for exact identities, the maximum error over seeds. CPU only, GPU $0.
Rerun: `cd study/T1/w1; export PYTHONPATH=../../../../wealth-net/.pylib; python3 check_L1.py a; python3 check_L1.py b; python3 check_L1.py c; python3 check_L1.py d; python3 check_L2.py; python3 check_L3.py`
(check_L1 b took 11 min under load 40; the rest take under 3 min each). S0 landed during the run; its streams.py is byte-identical to the copy used, so no S0 rerun is needed.

Notation. A batch of B examples by Q questions; logit z, price p = sigma(z), outcome y, loss
L = mean BCE. Books (wealth-net `books.py`): R = mean(z * s) with s = y - p detached, and a unit's
income is a * dR/da for its activation a. A "unit" is a readout row, an MLP neuron, or an attention
head. theta_u is the unit's out-side parameter group (the weights leaving it), or its in-side group
(the weights and bias entering it).

---

## Lemma 1: books income is -theta . grad L; under SGD it is norm growth, under weight decay it is rent, and at calibration it has mean zero

**1.1 Euler identity. [theorem; known]** dR/dz = -dL/dz (both are (y - p)/(BQ) per entry, the first
with p detached). So a * dR/da = -a * dL/da. If the network output depends on theta_u only through
a homogeneous map of degree 1 (a readout row with bias; the out-side weights of a neuron or head,
where the output is linear in w_out; the in-side weights and bias of a ReLU neuron; the V rows of a
head), then Euler gives theta_u . dL/dtheta_u = a * dL/da, and

    income_u = -theta_u . grad_{theta_u} L        (in-side income = out-side income for a ReLU unit).

LayerNorm is homogeneous of degree 0 in its input, so it breaks this identity for every unit
upstream of an LN. wealth-net restores it by detaching the LN denominators in the books graph
(`freeze_ln`). The books are then -theta . grad of a *surrogate* loss, not of the loss being trained.
Prior art: this is the first-order Taylor importance, and summed over steps it is the
movement-pruning score (Sanh, Wolf, Rush 2020, "Movement Pruning", https://arxiv.org/abs/2005.07683;
that the score is exactly the accumulated -W * dL/dW is a recollection, unverified). In-side = out-side
is the per-step form of the balancedness law of Du, Hu, Lee 2018,
https://arxiv.org/abs/1806.00900 (differences of squared layer norms are invariant under gradient
flow for homogeneous nets; checked on the abstract page).

**1.2 Norm growth under SGD, and the Adam residual. [theorem]** Plain SGD with step eta and no
weight decay, with books taken on the update batch before the step (as in wealth-net `train.py`):
theta' = theta - eta g gives, exactly,

    -theta_u . g_u = (||theta_u'||^2 - ||theta_u||^2) / (2 eta) - (eta/2) ||g_u||^2,

so cumulative true income = Delta||theta_u||^2 / (2 eta) - (eta/2) sum_t ||g_u,t||^2. Under gradient flow
the second term vanishes: wealth *is* half the growth of the squared norm (Kunin et al. 2021,
"Neural Mechanics", https://arxiv.org/abs/2012.04728, derive the same conservation and breaking laws
for scale and rescale symmetries at finite eta, with weight decay and momentum). With heavy-ball
momentum, Delta||theta||^2/(2 eta) = -theta_t . m_t + (eta/2)||m_t||^2 with m_t = sum_s beta^(t-s) g_s: the norm
tracks an exponentially lagged income evaluated at the current theta. **Adam residual:** with
D_t = diag(1/(sqrt(v_hat)+eps)), Delta||theta||^2/(2 eta) = -theta . D_t m_hat_t + (eta/2)||D_t m_hat_t||^2. The
norm tracks a *preconditioned* income sum_i theta_i m_i/sqrt(v_i); the residual against the books is
sum_i theta_i g_i (1 - c/sqrt(v_i)), which is zero only when v is constant within a unit. wealth-net
trains with Adam, so its "wealth = norm growth" reading fails for hidden units (L1b: R^2 0.02 for neurons).

**1.3 Rent under weight decay. [theorem]** SGD with coupled L2 (g~ = g + lambda theta):

    income_u = Delta||theta_u||^2/(2 eta) + lambda ||theta_u||^2 - (eta/2)||g~_u||^2
             =        drift           +      rent          -   step term.

At stationarity the drift is zero, so every surviving unit's time-average income equals its rent
lambda ||theta_u||^2, less a step term of order eta. Books income then ranks units by magnitude: under weight
decay, wealth is magnitude pruning. (Decoupled decay, AdamW, gives the same rent with the Adam
preconditioner of 1.2 on the income side; not checked.)

**1.4 Zero-mean tangent and the in-sample corollary. [theorem]** If p(x) = P(y = 1 | x), then for any
unit contribution c_u(x), E[c_u(x)(y - p(x))] = E[c_u(x) E[y - p | x]] = 0, for every unit and for the
total. More generally, tangent income has mean zero for every unit iff the predictor is calibrated
against each unit's contribution, a multicalibration condition with the units as the test class
(Hebert-Johnson et al. 2018, https://arxiv.org/abs/1711.08513, recollection, unverified). In-sample
corollary: at a stationary point of L + (lambda/2)||theta||^2 on the training set, theta_u . (grad L + lambda theta_u) = 0,
so in-sample income = lambda ||theta_u||^2 >= 0 for every group, noise features included. In-sample books
cannot separate signal from memorisation. Books on data not yet trained on (prequential) can.

**Checks.**

| check | result | pass |
|---|---|---|
| 1.1 Euler, frozen-LN graph, all 5 unit kinds, Net after 200 Adam steps (`out/L1a.json`) | worst relative error 8.8e-16 over units and seeds | yes (< 1e-9) |
| 1.1 books vs -theta.grad of the *live*-LN loss (`out/L1a.json`) | readouts: 0 gap; neurons: median per-unit gap 0.28-0.38, Pearson 0.87-0.95; heads: median gap 0.21-1.27, Pearson 0.71-1.0 | LN leak confirmed |
| 1.2 per-step SGD identity, neuron out-side (`out/L1b.json`, `sgd.per_step_identity_max_rel_resid`) | max relative residual 1.8e-13 ± 3.9e-14 (every 100th step) | yes (< 1e-9) |
| 1.2 cumulative SGD identity, all kinds (`sgd_exact_resid_rel`) | max relative residual 3.6e-15 to 1.6e-14; step term = 1.2% (readouts) to 21% (heads) of |Delta norm|/2eta | yes |
| 1.2 R^2 across units, books (frozen LN) vs Delta||theta||^2/2eta, 2000 steps (`R2_Wbooks_vs_dnorm`) | SGD: readouts 1.00, neurons 0.35 ± 0.04 (out) / 0.33 ± 0.05 (in), heads 0.62 ± 0.21 / 0.57 ± 0.20; momentum: 1.00, 0.43 ± 0.06, 0.72 ± 0.21; Adam: 0.91 ± 0.06, 0.022 ± 0.008, 0.36 ± 0.26 | see text |
| 1.2 R^2 across units, true -theta.g vs Delta||theta||^2/2eta (`R2_Wtrue_vs_dnorm`) | SGD: 0.989-1.000 for every kind; momentum: 0.987-1.000; Adam: readouts 0.91 ± 0.06, neurons 0.57 ± 0.11 (out) / 0.67 ± 0.14 (in), heads 0.89 ± 0.13 / 0.87 ± 0.07 | see text |
| 1.3 identity income = drift + rent - step term, MLP, 30k SGD steps, second half (`out/L1c.json`) | max abs residual 2.8e-17 for lambda in {0, 1e-3, 1e-2} | yes |
| 1.3 rent / income (median over units) | lambda = 1e-3: 0.70 ± 0.03; lambda = 1e-2: 1.008 ± 0.004 | yes |
| 1.3 Spearman(income, ||theta||^2) across units | lambda = 0: 0.40 ± 0.09; 1e-3: 0.924 ± 0.015; 1e-2: 0.9993 ± 0.0006 | yes |
| 1.3 fraction of live units with positive time-mean income | 100% at every lambda | yes |
| 1.4 calibrated predictor, n = 2e5, 5 coordinates + bias (`out/L1d.json`) | max abs z of unit means 2.48, 1.95, 1.35; total mean -0.0007, -0.0003, +0.0024 (se 0.0011) | yes (< 3 se) |
| 1.4 overfit logistic regression, 5 signal + 95 noise features, n_train = 200, lambda = 1e-3 | in-sample income - rent: max 1.8e-8; in-sample income > 0 for 100% of noise features | yes |
| 1.4 same, prequential (1e5 held-out) | noise income -0.024 ± 0.003, negative for 97.5 ± 1.2% of noise features; signal income +0.068 ± 0.024; test BCE 2.07 ± 0.20 | yes |

L1b in words (`out/L1b.json`, Net Q=16, W=8, d=16, H=2, L=2, 2000 steps; SGD lr 0.1, momentum 0.9 at
lr 0.01, Adam lr 1e-3). Three separate facts:
(i) *True* income -theta . grad L tracks norm growth under SGD and momentum (R^2 >= 0.987 for every unit
kind); Adam lowers it to 0.57-0.91: this is the Adam residual of 1.2.
(ii) wealth-net's books (frozen LN) track the *true* cumulative income poorly: R^2(books, true) is 1.00 for
readouts but 0.32 ± 0.05 for neurons and 0.58 ± 0.20 for heads under SGD, and 0.14 ± 0.03 (neurons) under
Adam. The per-step correlation is higher (Pearson 0.87-0.95, L1a), but the LN leak is systematic and
accumulates.
(iii) So books vs norm growth, the premise's claim, holds for readouts only. Under Adam (wealth-net's
optimizer) neuron books explain 2% of the variance of neuron norm growth.
The cumulative identity residual under momentum and Adam (8-1000 relative) is the expected failure of
the SGD formula and is not a check.

**What this means for the study.**
- The premise "a . dR/da is signed movement importance, i.e. norm growth under SGD" is exact only
  for plain SGD on a graph without LN (1.1, 1.2). wealth-net has both LN and Adam, so its books are
  neither the true -theta . grad L nor the norm growth; they are -theta . grad of a frozen-LN surrogate
  (cumulative R^2 against the true income: 0.32 for neurons under SGD, 0.14 under Adam). M1 must say
  which of the three it books: surrogate, true -theta . grad L, or exact settlement (lemma 2).
- With weight decay, cumulative books income ranks units like ||theta||^2 (Spearman 0.9993 at lambda = 1e-2). Any
  M-arm that uses weight decay must beat magnitude pruning, not just random pruning.
- In-sample books pay memorisation (100% of noise features earn). Prequential books flip the sign for
  97.5% of them. M1's `CURRENCY.lock` should require prequential settlement.

---

## Lemma 2: exact LMSR settlement = log-score gain = integrated gradients from a named prior; its mean is information gain

**Setup.** Binary LMSR with liquidity 1 in logit units: quantities (q1, q0), cost
C = log(e^q1 + e^q0), price sigma(l) with l = q1 - q0. Moving the market from l0 to l1 buys
Delta = l1 - l0 YES shares for C(after) - C(before) = sp(l1) - sp(l0), where sp = softplus.

**2.1 Settlement identity. [known]** With y in {0, 1}, log p_l(y) = y l - sp(l), so

    exact payoff = Delta y - sp(l1) + sp(l0) = log p1(y) - log p0(y)
                 = Delta (y - sigma(l1))  +  KL(Bern sigma(l1) || Bern sigma(l0)).
                   \___ tangent ___/      \___ Bregman divergence of sp, >= 0 ___/

The market-scoring-rule form is Hanson 2007, "Logarithmic market scoring rules for modular
combinatorial information aggregation", J. Prediction Markets 1(1):3-15 (working paper
http://hanson.gmu.edu/mktscore.pdf, checked). Tangent books (settle every share at the final
marginal price, as wealth-net does) undercount the exact payoff by KL(p1 || p0), a term that grows
with the size of the move.

**2.2 The IG split and its closed form. [theorem]** Let l1 = l0 + sum_i c_i (unit contributions to the
readout logit) and l0 the prior's logit. Integrated gradients of the log-score gain along the straight
path from the prior give

    share_i = c_i * integral_0^1 (y - sigma(l0 + t Delta)) dt = c_i (y - p_bar),
    p_bar = (sp(l1) - sp(l0)) / (l1 - l0)   (the average execution price),

and sum_i share_i = the exact payoff (completeness; Sundararajan, Taly, Yan 2017, "Axiomatic
Attribution for Deep Networks", https://arxiv.org/abs/1703.01365, title checked, axiom recalled).
Reading: every unit buys its part of the joint move at the average price. The closed form needs no
quadrature. If a quadrature is used (for paths through hidden units, where the logit is not linear in
the contributions), the n-step midpoint error of the total is at most |Delta|^3 max|sigma''| / (24 n^2), with
max|sigma''| = 1/(6 sqrt 3). The left-Riemann rule ("settle each step at the pre-step price") has O(1/n)
error.

**2.3 Expected settlement from the base-rate prior is mutual information. [theorem; standard]** If l1
is the calibrated posterior logit given X and l0 = logit P(Y = 1), then
E[log p1(Y) - log p0(Y)] = H(Y) - H(Y | X) = I(X; Y) >= 0. By 1.4 the tangent part has mean 0, so
E[KL(p1 || p0)] = I(X; Y): at calibration, *all* expected income lives in the term that tangent books drop.
Exact settlement from a named prior turns wealth into an information measure. Tangent settlement
turns it into a calibration residual with mean zero.

**2.4 Coalitions and arrival order. [theorem for the bound; convention for the choice]** IG shares
add over coalitions: share(S) = c_S (y - p_bar) with c_S = sum over S. The exact ablation value of S,
v(N) - v(N \ S), is c_S (y - p_bar_S), with p_bar_S the average price over [l1 - c_S, l1]. So
|ablation - share| <= |c_S| (sigma(hi) - sigma(lo)) over the hull of {l0, l1, l1 - c_S}. IG is the
Aumann-Shapley value of the continuous game (lemma 3.2). It differs from the discrete Shapley value
and from the P&L of a sequential market, which depends on arrival order.

**Checks** (`out/L2.json`).

| check | result | pass |
|---|---|---|
| 2.1 payoff = log-score gain = tangent + KL, 1e6 draws, l in [-20, 20] | max abs error 1.1e-14; payoff - tangent >= -2.9e-15 | yes (< 1e-12) |
| 2.2 closed form vs exact payoff, 2e5 draws per range | max residual 3e-15 | yes |
| 2.2 16-step midpoint, max abs error of the total (nats) vs bound, |Delta| <= 5 / 10 / 20 | 0.00099 / 0.0041 / 0.0167 vs 0.0020 / 0.0157 / 0.125 | yes |
| 2.2 p99 relative error at 16 steps, |Delta| <= 5 / 10 / 20: midpoint | 0.3% / 1.3% / 5.0% | |
| 2.2 same, trapezoid | 0.64% / 2.5% / 10% | |
| 2.2 same, left-Riemann | 15% / 31% / 67% (64 steps: 3.6% / 7.3% / 15%) | rule rejected |
| 2.3 E[exact from base rate] vs I(X;Y) (200-node Gauss-Hermite), n = 4e5 | seed 1: 0.2645 vs 0.2656 (z = -1.30); seed 2: 0.1975 vs 0.1984 (z = -1.15); seed 3: 0.1230 vs 0.1220 (z = +1.52) | yes (< 3 se) |
| 2.3 mean tangent income | within 1.3 se of 0 in every seed | yes |
| 2.4 coalition gap, K = 8, 300 instances x 10 subsets per seed | median 0.116 ± 0.005 nats, p90 0.75 ± 0.08; bound holds in 100% | yes |
| 2.4 exact Shapley vs IG, relative L1 (median) | 0.19 ± 0.03 | |
| 2.4 tangent shares vs IG, relative L1 (median) | 0.72 ± 0.02 | |
| 2.4 sd of unit 0's P&L over 20 random arrival orders / its IG share (median) | 0.95 ± 0.06 | |

**What this means for the study.**
- M1: settle exactly from a named prior (base rate, or the previous market price) with the closed form
  `c_i (y - p_bar)`. Where a path integral is unavoidable, use midpoint (16 steps: under 1.3% p99 error
  for |Delta| <= 10), never left-Riemann.
- Tangent shares differ from IG shares by 72% relative L1 on this toy. The choice of currency is not a
  detail.
- A sequential market's per-unit P&L is dominated by arrival order (sd about equal to the IG share).
  Books that emulate a sequential market need the order-free IG/Aumann-Shapley rule, or many orders.

---

## Lemma 3: cost sharing for a shared trunk; FLOP budgets for M7 and K1

**Setup.** Traders i = 1..K with usage x_i (routed tokens, trades), total FLOP cost C(x). A cost
rule assigns charges phi_i(x). Budget balance: sum_i phi_i = C(x) - C(0).

**3.1 Euler for costs. [theorem]** If C is homogeneous of degree r, sum_i x_i dC/dx_i = r C. So
marginal-cost pricing collects r C: it leaves a concave (economies-of-scale) shared trunk underfunded
and a convex one overfunded. This is the same identity as 1.1: book conservation and budget balance
are one theorem.

**3.2 Aumann-Shapley. [known]** phi_i^AS = x_i integral_0^1 dC/dx_i(t x) dt. It is budget balanced for any
smooth C with C(0) = 0, and it is the unique rule satisfying the Billera-Heath price axioms (Billera &
Heath 1982, "Allocation of shared costs: a set of axioms yielding a unique procedure", Math. of OR
7(1):32-39, https://doi.org/10.1287/moor.7.1.32, checked; Mirman & Tauman 1982, seen in search, not
read). It is the same integral as IG in 2.2: IG of a settlement and AS of a cost are one rule. For
C = F(sum x) + sum c_i x_i (a shared trunk plus separable own costs), AS = own cost + trunk cost split in
proportion to usage (average-cost pricing; serial and proportional rules in Friedman & Moulin 1999,
JET 87(2):275-312, https://ideas.repec.org/a/eee/jetheo/v87y1999i2p275-312.html, checked).

**3.3 Per-token trunk as a public good. [theorem; known]** On one token, the trunk costs F if any
consumer runs and 0 otherwise (a union game). Its Shapley value is an equal split among the
consumers of that token (Littlechild & Owen 1973, the airport game, Management Science 20(3):370-372,
https://doi.org/10.1287/mnsc.20.3.370, checked). The per-token games add up over tokens, and with
unit usage per consumer AS gives the same split. The rule in `costshare.py`:

- **own FLOPs** (expert MLP, trader head and update): paid in full by the trader;
- **shared FLOPs** (trunk, window features): split equally among the consumers on that
  token/question. In an MoE with L layers and top-k, each routed activation pays trunk/(L k);
- **rent**: FLOPs that run whether or not the trader is used (its router column, 2d per token per
  layer) are charged on every token seen.

Conventions for M7. (A) *Consumers pay the trunk* (the rule above; the trunk is infrastructure).
(B) *The maker pays the trunk*: the trunk is a trader with its own books, and experts pay only their own
FLOPs plus rent. (B) equals marginal-cost pricing for the experts. Under (A) a dead expert pays only
rent, which drains it: an exit mechanism with no tuned coefficient.

**3.4 Equal split subsidises crowding. [conjecture]** Under the equal split, a trader that reads
the same shared features as n others pays shared/n. Crowding onto popular features is therefore
subsidised relative to full-cost or marginal charging, and a Kelly economy under equal split
should show less feature diversity and more herding. The subsidy is bounded by
shared/(shared + own) per trade: 0.18 for logistic-32, 0.22 for pattern-2 (4/18) and 0.53 for
pattern-8 (16/30). Test as a K1 arm: {equal split, full cost, marginal (own only)} x seeds 1-3;
metrics are the entropy of shared-key usage and the wealth Gini.

**Checks** (`out/L3.json`, `out/flops_M7.json`, `out/flops_K1.json`).

| check | result | pass |
|---|---|---|
| 3.1 marginal-cost revenue / C for r = 0.5 / 1 / 1.5, 100 instances x 3 seeds | 0.5 / 1.0 / 1.5 (sd < 3e-16) | yes |
| 3.2 AS revenue / C, same instances | 1.0 (worst 0.99999993) | yes |
| 3.2 concave trunk (F (sum x)^0.7) + own costs, K = 8: AS budget residual; AS vs closed form | 3e-7; 1.2e-7 | yes |
| 3.2 exact Shapley (2^8 coalitions) budget residual | 1e-13 | yes |
| 3.2 marginal revenue / C (trunk = 97-98% of cost) | 0.707 ± 0.001 | underfunds |
| 3.2 L1(AS - Shapley) / C; L1(marginal - AS) / C | 0.045 ± 0.014; 0.29 | |
| 3.3 union game: equal-split closed form vs exact Shapley | max difference 1.6e-14 | yes |
| 3.3 M7 budget balance, skewed routing (Dirichlet 0.3), 4096 tokens, E = 8 / 64 | residual 0.0 | yes |
| 3.3 marginal revenue / FLOPs spent, E8 / E64 (fine-grained experts; full-width experts) | 0.555 / 0.580; 0.712 / 0.723 | underfunds |

**M7 FLOP table** (forward, 1 multiply-add = 2 FLOPs; `out/flops_M7.json`; wealth-net dims d = 64, L = 2,
H = 4, W = 32, T = 64 questions attended, top-2; training is about 3x forward):

| config | trunk / token | expert / routed token | charge / routed token (A) | rent / token / expert / layer | active / token | trunk share |
|---|---:|---:|---:|---:|---:|---:|
| E8, fine-grained (h_e = 128) | 107,072 | 32,896 | 59,664 | 128 | 240,704 | 0.445 |
| E8, full-width (h_e = 256) | 107,072 | 65,792 | 92,560 | 128 | 372,288 | 0.288 |
| E64, fine-grained | 107,072 | 32,896 | 59,664 | 128 | 255,040 | 0.420 |
| E64, full-width | 107,072 | 65,792 | 92,560 | 128 | 386,624 | 0.277 |

**K1 FLOP table** (`out/flops_K1.json`, per question traded). Pattern-k: shared 2k (read k bits),
own 14. Logistic-W: shared W (window encoding), own 4W + 16. For 800 pattern traders (k in {2, 4, 8}) and
200 logistic-32 traders on Q = 64 (one niche question each, plus 2% generalist trades): budget residual
0.0 in every seed; mean charge per trade 15.0 (pattern) and 148.2-148.6 (logistic, against a full cost
of 176). Break-even FLOP price at 0.1 nats per trade: 5.7e-4 nats/FLOP for a lone logistic-32 trader,
6.9e-4 with 100 sharers, 4.5e-3 for a lone pattern-4 trader.

**For M7 and K1.** `from costshare import moe_token_flops, moe_charges, k1_trader_flops, k1_charges`
(path `study/T1/w1/costshare.py`, numpy only). `moe_charges(counts[L, E], n_tokens, fl, rule)` returns
charges per (layer, expert) and the FLOPs spent; the `"shapley"` rule is convention (A) and the
`"marginal"` rule is convention (B) for the experts.

---

## Prior art (checked = the page was opened in this session)

- Hanson 2007, LMSR, http://hanson.gmu.edu/mktscore.pdf (checked).
- Sundararajan, Taly, Yan 2017, Integrated Gradients, https://arxiv.org/abs/1703.01365 (title checked; completeness axiom recalled).
- Sanh, Wolf, Rush 2020, Movement Pruning, https://arxiv.org/abs/2005.07683 (title checked; score formula recalled).
- Du, Hu, Lee 2018, layers automatically balanced, https://arxiv.org/abs/1806.00900 (checked).
- Kunin et al. 2021, Neural Mechanics, https://arxiv.org/abs/2012.04728 (checked).
- SGD noise equilibrium and norm balance, https://arxiv.org/abs/2402.07193 (seen in search, not read).
- Hebert-Johnson, Kim, Reingold, Rothblum 2018, Multicalibration, https://arxiv.org/abs/1711.08513 (unverified).
- Billera & Heath 1982, https://doi.org/10.1287/moor.7.1.32 (checked); Mirman & Tauman 1982 (seen in search).
- Friedman & Moulin 1999, https://ideas.repec.org/a/eee/jetheo/v87y1999i2p275-312.html (checked).
- Littlechild & Owen 1973, https://doi.org/10.1287/mnsc.20.3.370 (checked).
