# THEORY (T1): lemmas 1-7 for traders-net

Assembled 2026-09-27 from three parts written in parallel, then merged and checked by the T1 lead.
Each lemma is tagged theorem, known (cited) or conjecture, and each has a numerical check on an S0 toy.
- Part 1 (lemmas 1-3: books identity, exact LMSR = IG, trunk cost sharing and FLOP budgets): `THEORY-part1.md`, `w1/`
- Part 2 (lemmas 4-5: delayed settlement, finite-population LI): `THEORY-part2.md`, `w2/`
- Part 3 (lemmas 6-7: fund of funds, top-k router as broker): `THEORY-part3.md`, `w3/`

The part files are the sources. Below the lead summary, this file is their concatenation. To regenerate it:
`cd study/T1 && cat lead/HEADER.md THEORY-part1.md THEORY-part2.md THEORY-part3.md > THEORY.md`.
All paths are relative to `study/T1/`. No GPU was used (CPU only, torch 2.14 cpu), so the GPU cost was $0.

## Lead summary (2026-09-27 ~22:46Z)

**How to check this page.**
- `python3 lead/check_claims.py --out lead/claims_check.json` re-reads the output files and asserts the 57
  headline numbers below and in the parts. The last run passed 57 of 57 (`lead/claims_check.json`).
- `python3 lead/compare_rerun.py /tmp/traders-net/T1/rerun --out lead/rerun_compare.json` compares every numeric
  leaf of the committed outputs with a clean rerun of all scripts (see "Reproduction" below).

**The seven lemmas.** Tags use the parts' own labels; seeds are 3 unless stated, as mean ± sd.

| lemma | tag | what it says | key check (file) | verdict |
|---|---|---|---|---|
| 1 books identity | theorem (1.1 known Euler) | books income a·dR/da = −θ·∇L. Under SGD it is norm growth, under weight decay rent; at calibration its mean is 0 | Euler identity rel err 8.8e-16 (`w1/out/L1a.json`). SGD per-step identity 1.8e-13 (`w1/out/L1b.json`). Books vs true income R² 0.32 (SGD) and 0.14 (Adam) on neurons (`w1/out/L1b.json`). With λ=1e-2, Spearman(income, ‖θ‖²) 0.9993 (`w1/out/L1c.json`). In-sample books pay 100% of noise features; prequential books charge 97.5% of them (`w1/out/L1d.json`) | holds; the books are **not** the true income under Adam or LN |
| 2 exact LMSR = IG | known (2.1) + theorem (2.2-2.4) | exact settlement = log-score gain = tangent + KL; its IG split has a closed form; the mean from the base rate is I(X;Y) | payoff − tangent − KL max err 1.1e-14. Seed 1: E[exact] 0.2645 vs I 0.2656 nats. Tangent vs IG shares: relative L1 0.72; arrival-order sd/share 0.95 (`w1/out/L2.json`) | holds |
| 3 trunk cost sharing | theorem + known (Aumann-Shapley); 3.4 conjecture | Euler cost shares balance the budget; the equal split of the trunk subsidises crowding | concave-trunk marginal revenue covers 0.707 of cost (`w1/out/L3.json`). E8 fine-grained: 240,704 active FLOPs/token and 59,664 per routed token (`w1/out/flops_M7.json`). K1 break-even 5.7e-4 nats/FLOP (`w1/out/flops_K1.json`) | holds; 3.4 is untested |
| 4 delayed settlement | theorem (4.1-4.5); known (4.7 LI); 4.6 statement only | book at the trade price, mark in between, settle at reveal. Re-pricing at reveal books hindsight | telescoping 3.3e-14 (`w2/results/summary.json`). Hindsight gap on periodic 0.0051 (D=64) and 0.046 (D=512) at lr 0.05 (`w2/results/summary_delay.csv`). Stale/‖W_trade‖ −0.747 and −1.197 (`w2/results/summary.json`). Self-resolution pays KL: prng D=0 income 0.056 vs KL 0.055 (`w2/results/summary_delay.csv`) | holds. Prereg 4b fails; it passes only in the exploratory arm |
| 5 finite-population LI | theorem (5.1-5.5, some known); 5.6 conjecture | members are bounded by ln(1/ε); price readers need fixed-point clearing; delay costs (D+1); births bound E_C at about φ·n/ρ | member E 6.90739 ≤ 6.907755 (`w2/results/summary.json`). Reader at one-pass clearing 48.4 (seed 1). a-D64: 447.81 ≤ 449.00. E_C periodic 1239 → 472 with births, diag 4535 → 1065 (`w2/results/summary_li.csv`) | bounds hold; plateau fails on periodic (ratio 0.29) and passes on diag (0.18) |
| 6 fund of funds | theorem (6.1-6.3); nested Fixed Share conjecture | buy-and-hold hierarchy = flat Bayes mixture; constant rebalancing = flat CRP; only rebalancing adds power | DAG vs flat: 8.9e-16; CRP DAG vs flat CRP: 9.1e-13. Toy CRP +0.087 vs leaf −0.424 nats/step. On ca110, EG beats the best leaf by 0.0132 (`w3/out/summary.json`, `w3/out/L6_growth.csv`) | holds |
| 7 router as broker | theorem (7.1-7.6); 7.7 known | router gradient g_i(r_i − R) is zero-sum; top-1 and unnormalised gates lock in under censoring, top-2 renormalised explores; Exp3 shadow credit is unbiased; router = commission-free broker | closed form vs autograd 8.6e-16. Top-1 censored finds the best 6.7% of the time with regret 0.378; with Exp3, 52% and 0.023. Censored top-2 gradient rel err 0.43. Broker residual 2.6e-18 (`w3/out/summary.json`) | holds; the Exp3 arm misses its prereg prediction "near 1" |

**Deviations from preregistration and failures (all disclosed in the parts; collected here).**
- w2 4b failed as preregistered: the lr 0.05 learner is unstable for D ≥ 64. It passes only in the exploratory
  scaled-step arm (addendum in `w2/PREREG.md`, dated 22:17:59Z, before that arm was run).
- w2 5.4 plateau: fails on periodic (slope ratio 0.29 against < 0.2) and passes on diag (0.18) (`w2/results/summary_li.csv`).
- w2 P5.4-d on periodic holds only vacuously. Pattern-4's E is negative (slopes −0.114 → −0.116), so
  there is nothing to bound.
- w2 used 60 bisection iterations, not the preregistered 50 (stated in part 2).
- w3 C7e predicted Exp3 would find the best expert "near 1" of the time; it did so 52 ± 4% of the time. Its low
  regret (0.023 ± 0.002) shows the misses are near-ties, which the regret alone supports.
- w3 C7b: sigmoid gates are not zero-sum (|Σu| 3.3e-4 to 7.9e-4, on a scale of 4.4e-4 to 8.1e-4). The criterion was refined before the run.
- w3 C7d used 1000 × 256 draws, against the preregistered 2e5.
- w1 started from a copy of the wealth-net files before S0 existed. `model.py` and `streams.py` are byte-identical
  to `study/S0/substrate/` (sha 14128c2a… for streams.py), so no rerun was needed.
- Lead corrections (22:45Z) to part 2, section 4.3:
  - The lr 0.05 hindsight gap had been quoted with the scaled-step values; it is now 0.0051 ± 0.00006 and
    0.046 ± 0.0008.
  - "|gap| ≤ 1.5e-4 on lfsr16/prng" had to be widened to ≤ 4.9e-4 (lfsr16 at D=64).
  - The part 2 check time is ~22:37Z, not ~22:45Z.

**What M1, M5, M7 and K1 should take from this.**
- **M1 (currency).**
  - Settle exactly from a named prior, using the IG closed form with the 16-step midpoint rule.
  - Require prequential settlement in `CURRENCY.lock`: in-sample books pay memorisation (lemma 1.4).
  - Do not call books "true income" under Adam or LayerNorm (lemma 1.2).
  - Cache books at trade time under delay (4.3); self-resolved labels pay KL, not accuracy (4.5).
- **M5 (LI metrics).**
  - Report E_C both at the clearing price and as traded.
  - Use fixed-point clearing whenever price readers are present.
  - Under delay, use D+1 interleaved books.
- **M7 (MoE).**
  - Use cost convention (A), consumers pay, with (B) as the sensitivity arm.
  - The FLOP tables in `w1/out/flops_M7.json` come from `w1/costshare.py`.
  - The router is a broker (7.6).
  - Expect lock-in only for top-1 or non-renormalised gates.
  - Normalise expert wealth by routed-token count.
- **K1.**
  - Break-even prices are in `w1/out/flops_K1.json`.
  - Births must be targeted: uniform births from a big pool wait n/ρ steps.
  - Use the diag family as the cheap detector of one-pass clearing.

**Reproduction.**
- A clean copy of w1/w2/w3 was rerun under `/tmp/traders-net/T1/rerun` (single-thread CPU, same seeds).
- Every script was rerun except `check_L1 b` (11 min; its numbers are covered by `lead/check_claims.py`).
- `lead/rerun_compare.json` lists the per-file results. Result at 22:46Z: 23 output files and 33185 numeric values were compared; all are bit-identical (max relative difference 0, nothing missing).

**Not verified.**
- Items marked "(unverified)" in the parts' prior-art lists (recollections of papers not opened here).
- The "only if" direction of conjecture 5.6.
- Conjecture 3.4 (crowding) and the nested Fixed Share conjecture (part 3).
- 4.6 (LM next-token and RL), which is argument only and was not run.
- The LI toy learner is per-question logistic, not wealth-net.

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
# THEORY, part 2 (T1 worker 2): lemmas 4 and 5

Checked 2026-09-27 ~22:37Z. Scripts and outputs are in `study/T1/w2/`: `PREREG.md`, `check_delay.py`,
`check_li.py` and `summarize.py`. Results are `results/delay.json`, `results/delay_stable.json`,
`results/li.json`, the tables `results/summary_delay.csv`, `results/summary_delay_stable.csv`,
`results/summary_li.csv` and `results/li_curves.csv`, and the aggregate `results/summary.json`.
Substrate: `study/S0/substrate/streams_ext.make_questions` (8 families, Q=64, seeded PRNG-parity); its
sha256 prefix is recorded in every results file. Seeds 1-3. Numbers are mean ± sd over the 3 seeds, or, for
exact identities, the maximum error over seeds. CPU only, one process, GPU $0.
To rerun: `cd study/T1/w2; export OMP_NUM_THREADS=1; python3 check_delay.py; python3 check_delay.py --eta-scale --out results/delay_stable.json; python3 check_li.py; python3 summarize.py`
(about 2 min, 2 min and 20 min under load 40).

Notation. A contract on question q is opened at trade time t at logit ℓ_t (price p_t = σ(ℓ_t)) against
a prior logit ℓ0_t. Its outcome y is revealed at t+D. Books: tangent ℓ_t(y − p_t), and exact
LMSR log p_t(y) − log p0_t(y) (lemma 2). A mark m ∈ [0,1] is a probability for y=1 used to value an
open contract before reveal.

---

## Lemma 4: delayed settlement. Book at the trade price, mark to market in between, settle at reveal

Toy (`check_delay.py`, PREREG check 4): one online logistic model per question on the last W=16
revealed bits, with plain SGD, lr 0.05 and T=8000. With delay D ∈ {0, 64, 512} the forecast at t sees only
bits revealed before t, and the trade-time gradient is cached and applied at t+D (stale-gradient
SGD). The exact-book prior is the Laplace running base rate. Every contract caches x, θ, ℓ, ℓ0 and a
self-sampled outcome y′ in a ring buffer. Open contracts are re-marked every 16 steps and at D/2.

**4.1 Telescoping. [theorem, elementary]** A contract's value at mark m is linear in m:
tangent V(m) = ℓ_t(m − p_t), and exact V(m) = m·[log p_t(1) − log p0_t(1)] + (1−m)·[log p_t(0) − log p0_t(0)].
For any mark sequence p_t = m_0, m_1, …, m_n = y, the mark-to-market increments sum to V(y) − V(m_0).
- Tangent: this is exactly the settlement.
- Exact: V(y) is the settlement, and V(p_t) = KL(p_t‖p0_t) is the "day-one" value a trader books if it marks at its own trade price.

Check: the max residual over contracts, families, D and seeds is 3.3e-14 for the tangent books (`results/delay.json`, `tel_tan_max`). For exact books the check only confirms V(y) = settlement (0.0, by construction).

**4.2 Martingale marks, and marks from the trader's own model. [theorem; known]** If the marks are the
conditional probabilities m_k = P(y=1 | F_k) for some filtration, V(m_k) is a martingale, because V is linear. So
E[settlement | F_k] = V(m_k), and income can be paid at every mark without bias. Real marks come from
a model, and a mark taken from the trader's *own* current model is biased toward its own trade: at m = p_t
it books KL(p_t‖p0_t) ≥ 0 whatever the truth (see 4.5). Measured at D/2, with bias = final settlement minus the mid-life mark:
- **Mid-life marks were optimistic.** The bias was −11.2 to −0.011 at lr 0.05 (`results/summary_delay.csv`, `mtm_bias_ex`) and −0.059 to +0.002 with the delay-scaled step (`results/summary_delay_stable.csv`).
- **The marks tracked settlement poorly.** corr(mid mark, settlement) was ≤ 0.18 at lr 0.05 and ≤ 0.48 with the scaled step.

Rule: mark open contracts with a price that the trader does not control (the market's clearing price or a
held-out model), never with the trader's own model.

**4.3 Hindsight gap: cache at trade, settle at reveal. [theorem: identity; measured: partial]**
Suppose a contract is booked at reveal with the reveal-time model p′ on the cached input, instead of at the trade price. The
credit then changes by gap = log p′(y) − log p_t(y). This is learning between t and t+D on that input (p′ has
been updated on the labels of the contracts opened in the D steps before t, though not on y itself). It is a prequential leak, not trade income. At D=0 the gap is
exactly 0 (checked: 0.0 in every D=0 cell). Rule: store ℓ_t, ℓ0_t and the per-unit shares at trade time, and apply the settlement to
those cached books at reveal. Never re-price at reveal.
- *Preregistered 4b fails as stated* (`results/summary_delay.csv`). At lr 0.05 the gap on periodic is +0.0051 ± 0.00006
  (D=64) and +0.046 ± 0.0008 (D=512). But it is negative on ca110 at D=64 and far from 0 on shuffled labels
  (periodic D=512 shuffled −0.57 ± 0.16). The learner is unstable for D ≥ 64 at this step: exact income on
  prng falls from −0.061 (D=0) to −1.78 (D=512).
- *Exploratory, logged in PREREG addendum* (η_D = 0.05·16/(16+D), `results/summary_delay_stable.csv`).
  All four preregistered learnable families are positive and increase with D. From D=64 to D=512:
  - periodic: 0.0053 → 0.0474
  - prime: 0.0021 → 0.0296
  - and: 0.0028 → 0.0307
  - or: 0.0013 → 0.0120

  On lfsr16 and prng, |gap| ≤ 4.9e-4 (the largest is lfsr16 at D=64, −0.00049 ± 0.00021; ≤ 1.5e-4
  at D=512 and for prng; lead correction 22:45Z). The shuffled controls stay positive at D=512 (periodic 0.017, prime
  0.027, and 0.028): shuffling keeps each question's base rate, and the model still learns it. So the gap measures
  *all* learning, and the "shuffled ≈ 0" clause was a wrong prediction, not a bug.
- **Size.** At D=512 (scaled step), hindsight booking adds 0.030 on prime and 0.031 on "and", where the trade-price
  income is −0.004 and −0.012. It flips the sign of those families' income.

**4.4 Stale-gradient norm identity (the delay residual to lemma 1.2). [theorem]** Plain SGD applies at
step k the gradient g_τ computed at θ_τ (k = τ + D). Then
Δ‖θ‖²/2η = W_trade − S_stale + (η/2)Σ‖g_τ‖², with W_trade = −Σ θ_τ·g_τ (the trade-time books of lemma 1)
and S_stale = Σ (θ_{τ+D} − θ_τ)·g_τ. Check: the max relative error is 2.7e-13 (`results/summary.json`,
`delay.norm_rel_err_max`). The residual is as large as the income:
- At lr 0.05, S_stale/|W_trade| is −0.747 ± 0.004 at D=64 and −1.197 ± 0.066 at D=512.
- With the scaled step, the ratio is −0.97 ± 0.03 at D=64. At D=512, W_trade = −1890 ± 1544 crosses 0, so the ratio is meaningless there; S_stale = −19153 ± 1842.

So under delay (delayed labels, asynchronous data parallelism, pipeline staleness) "wealth = norm growth" does not
hold even with SGD. The books must be the cached W_trade.

**4.5 Self-resolution pays confidence, not accuracy. [theorem]** Settle on y′ ~ Bernoulli(p_t), the model's own
sample. Then E[ℓ_t(y′ − p_t)] = 0 and E[log p_t(y′) − log p0_t(y′)] = KL(p_t‖p0_t) ≥ 0.
Check over all 48 cells of both runs:
- |mean tangent|/SE ≤ 1.63.
- Exact mean vs KL: relative difference ≤ 3.7% at lr 0.05 and ≤ 14% in the scaled run, where KL is as small as 0.0017 (`results/summary_delay*.csv`).

Example, prng at D=0: true-label exact income −0.061, self-resolved +0.056, KL 0.055.

**4.6 What changes for LM next-token and RL. [statements; not run]**
- *LM next-token pretraining:* the label is the next data token, so D=0 per token. Books settle at once
  and lemmas 1-2 apply unchanged. There are two exceptions:
  - With stale updates (async data parallelism, pipeline or accumulation delay), 4.4's S_stale term appears.
  - With labels from the model's own samples (self-training, distillation on its own generations), 4.5 holds: books pay KL(p‖p0), not accuracy. They need an external resolver (a verifier, or later real data).
- *Sequence-level resolution (RL, verifier reward at the end):* open a per-token contract at s_t with the critic's mark
  V(s_t) and settle it at the return G_t. With λ=1 the TD errors telescope: Σ_k γ^k δ_{t+k} = G_t − V(s_t)
  (4.1; `mrp` in `results/delay.json`, max residual 5.8e-15, on a 20-state reward chain with H=50, γ=0.9 and a wrong critic).
  One-step books (λ=0) mark at the critic's next value. They differ from books marked at the true value by exactly
  γ·e(s_{t+1}), where e = V − v_true (identity residual 1.6e-15; mean bias 0.28-0.76 per seed for e ~ N(0.5, 1)). So a
  biased critic taxes or subsidizes every step of a trajectory. Known: TD(λ), Sutton 1988, and Holland's
  bucket brigade 1985 (both unverified). Marking with the actor's own critic is 4.2's own-model mark bias; mark with
  the population's price.

**4.7 Deductive settlement (ties to M5's D-prove). [known: LI]** In Logical Induction
(https://arxiv.org/abs/1609.03543), holdings in φ are worth 1 or 0 once the deductive process proves φ or ¬φ,
and are valued at market prices before that. Settling at proof time is 4.1 with the reveal time set to the proof time.
A bundle that is incoherent at the market prices (A and ¬A bought for less than 1) has a guaranteed value as soon as A∨¬A is
proven, before A is ever decided. For M5's D-prove family (never sampled), an arbitrageur's E_C should
therefore be booked at proof time; waiting for A's label books nothing, because that label never comes.

---

## Lemma 5: finite-population logical induction. E_C(t), invasion fitness, and what makes E_C sublinear

Market (`check_li.py`, PREREG check 5): one Kelly market per question, with price p = Σ w_i q_i / Σ w_i
(Beygelzimer, Langford and Pennock, https://arxiv.org/abs/1201.6655, from the scope doc) and update
w_i ← w_i q_i(y)/p(y), which conserves Σw. The traders:
- residents: pattern-0/1/2 (KT counts);
- class C: pattern-4, pattern-8 and 20 smooth price-reading calibration traders, q = σ(logit p ± bump_b(p)) over 10 bins;
- 9 constant "junk" traders, used only in the birth pool.

The family "diag" has 8 extra questions with y_t = 1[p_t < 1/2] on the market's own price. E_T(t) = Σ_s log q_T(y_s)/p_s(y_s) at
the clearing price, and E_C = max over T ∈ C. The invasion fitness φ_T is the slope of E_T for a shadow T (wealth → 0).
ε = 1e-3 and T = 8000. Conservation error is ≤ 1.6e-11 in every arm (`results/summary.json`, `li.*.max_conservation_err`).

**5.1 Members are bounded. [theorem; known: Bayes mixture / BLP]** If T trades in the population
from wealth share ε, then w_T(t) = ε·exp(E_T(t)) ≤ 1, so E_T(t) ≤ ln(1/ε) = 6.9078 for all t.
Check (P5.1 pass): the max member E is
- arm b: 6.90739;
- arm a: 6.907755278977;
- arm d: 6.907755278982234, against a bound of 6.907755278982137 (a 1e-13 float overshoot).

(`results/summary.json`, `li.{a,b,d}.max_member_traded_E`)

**5.2 Price readers need fixed-point clearing. [theorem]** A trader whose belief q_T(p) reads the price, cleared in
one pass, trades on the *pre*-price. Its traded E (belief at the pre-price, scored at p_clear) obeys 5.1. The LI-style
test, belief at p_clear scored at p_clear, is unbounded. With fixed-point clearing p* = Σ w_i q_i(p*)/Σw, a
root exists by the intermediate value theorem for continuous q_i and is found by bisection (60 iterations; PREREG said 50). Then the two
coincide and 5.1 covers the reader. This is the finite version of LI's Brouwer step and of the continuity
requirement in smooth calibration (Foster and Hart, https://arxiv.org/abs/2210.07152).
Check (P5.2 pass): arm a readers at clearing reach 48.4, 36.4 and 26.9 by seed (> 6.91), while arm b readers reach ≤ 6.907
(`li.a.max_reader_clearing_E_per_seed`, `li.b...`).

**5.3 Delay costs a factor (D+1). [theorem; known technique]** With labels D steps late, run D+1 interleaved
wealth vectors, where sub-market j trades the steps t ≡ j (mod D+1) and so only ever trades on settled wealth. 5.1 per
sub-market gives E_T ≤ (D+1)·ln(1/ε) (Weinberger and Ordentlich 2002, unverified; Joulani, György and Szepesvári,
https://arxiv.org/abs/1306.0686). Check (P5.3 pass): arm a-D64 max 447.81 against a bound of 449.00, which is 99.7% of the bound, so the bound
is nearly tight (`li.a-D64`).

**5.4 Births bound E_C by the wait until birth. [theorem for the bound; measured for the size]** A trader born
at τ with wealth ε′ has E_T(t) − E_T(τ) ≤ ln(1/ε′) from then on (5.1). With births drawn uniformly from a pool of n
at rate ρ per step, τ is geometric with mean n/ρ, so E_T ≲ φ_T·min(t, n/ρ) + ln(1/ε′). Repeated births restart a
trader that went broke before a shift, which is the fixed-share mechanism (Herbster and Warmuth 1998, unverified).

Here n = 31, so n/ρ = 3100 steps at ρ = 1e-2 and 31000 > T at ρ = 1e-3 (`results/summary_li.csv`):
- periodic, ρ = 1e-2: E_C = 472 ± 236, against a prediction of 0.157·3100 + 7 ≈ 494; arm N reached 1239 ± 188.
- diag, ρ = 1e-2: E_C = 1065 ± 444, against arm N 4535. The prediction is 897-1786, depending on whether one or both of pattern-4 and pattern-8 must be born.
- ρ = 1e-3: little change (periodic 1173, diag 3367), as predicted.

The preregistered plateau test (second-half slope < 0.2 × first-half slope at ρ = 1e-2) passes on diag (0.228 → 0.040, ratio 0.18)
and fails on periodic (0.092 → 0.027 ± 0.034, ratio 0.29): T is only about 2.6 birth waits. Fixed-point births (cfp arms) equal the c arms
to 4 digits on periodic, because pattern traders do the exploiting, not price readers.

**5.5 Covering: membership is not necessary. [theorem]** If a member R satisfies
log q_R(y_s) ≥ log q_T(y_s) − c_s on the realized path with Σ_{s≤t} c_s = o(t), then E_T ≤ E_R + Σ c ≤ ln(1/ε_R) + o(t).
Check (arm d: residents plus pattern-8, with pattern-4 as a shadow):
- periodic: E_pat4 = −907 ± 16, with slopes −0.114 → −0.116. It never exploits, which meets P5.4-d.
- diag: E_pat4 = 141 with slopes 0.0286 → 0.0065 (ratio 0.23). This is sublinear but positive, because pattern-8 only approximately covers pattern-4 there (256 contexts to fill against 16).

**5.6 Refined conjecture. [conjecture]** For a finite population of Kelly traders under BLP clearing,
with C made of price-continuous traders, E_C(t) = o(t) on every stream if and only if every T ∈ C with φ_T > 0 is
eventually *covered* at the price actually used. Covering here means one of: membership (5.1), a covering member (5.5), or a birth
process that samples T or a cover at a positive rate (5.4). A price reader counts as covered only under fixed-point
clearing (5.2); under delay, bounds scale by (D+1) (5.3).
- The "if" direction follows from 5.1-5.5.
- The "only if" direction is the conjecture. The evidence is arm N, where all three routes fail and E_C grows linearly: periodic slope 0.156 → 0.157, positive on 3/3 seeds; diag 0.544 → 0.603 with market log-loss 0.694; prng stays near 0 (E_C 0.9), since there is nothing to learn.
- Known risk: "covered" is close to a restatement. The testable content is the *rates*: φ·n/ρ for births, (D+1)·ln(1/ε) for delay, and the size of the reader gap without a fixed point.

**Implications.**
- **M5.** Report E_C both at the clearing price and as traded. Use fixed-point clearing whenever C contains price readers (calibration bins). Book D-delay with D+1 interleaved books, or report the (D+1) scale. Book D-prove at proof time (4.7).
- **K1.** Births from a pool bound E_C at about φ·n/ρ + ln(1/ε). With a 100k-trader pool, uniform births wait n/ρ steps, so births must target (for example, mutations of wealthy traders) or the plateau never arrives within a run. Put the diag family in K1 as the cheapest detector of one-pass clearing.
- **S0 and M1.** Books under any delay must be cached at trade time (4.3); norm growth is not wealth under stale updates (4.4); self-generated labels pay KL (4.5).

**What this does not show.**
- There are no LM or RL runs; 4.6 is argument plus the reward-chain identity.
- The toy learner is a per-question logistic model, not wealth-net.
- 4b passes only in an exploratory arm.
- The plateau test in 5.4 fails on periodic at T = 8000.
- 5.6's "only if" direction is unproven.

## Prior art (checked = the page was opened in this session)
- Garrabrant, Benson-Tilsen, Critch, Soares and Taylor, Logical Induction: https://arxiv.org/abs/1609.03543 (checked)
- Joulani, György and Szepesvári 2013, online learning under delayed feedback: https://arxiv.org/abs/1306.0686 (checked)
- Foster and Hart, smooth calibration and fixed points (GEB 109:271-293): https://arxiv.org/abs/2210.07152 (checked)
- Beygelzimer, Langford and Pennock 2012, learning performance of prediction markets with Kelly bettors: https://arxiv.org/abs/1201.6655 (from the scope doc; not opened here)
- Chen and Vaughan 2010, cost-function markets and no-regret learning: https://arxiv.org/abs/1003.0034 (from the scope doc; not opened here)
- Weinberger and Ordentlich 2002, delayed-feedback prediction via interleaving (unverified)
- Sutton 1988, TD(λ); Holland 1985, bucket brigade (unverified)
- Herbster and Warmuth 1998, tracking the best expert with fixed share (unverified)
- Blume and Easley 2006, market selection (from the scope doc; unverified)
# THEORY, part 3 (T1 worker 3): lemmas 6 and 7

Checked 2026-09-27 ~22:35Z. Scripts and outputs: `study/T1/w3/` (`PREREG.md`, `check_L6.py`,
`check_L7.py`, `aggregate.py`; results in `out/L6-s{1,2,3}.json`, `out/L7-s{1,2,3}.json`,
aggregated in `out/summary.json` and `out/L6_growth.csv`). Stream: S0's verbatim copy of the
wealth-net streams (`study/S0/substrate/streams.py`), seeds 1-3. Numbers: mean ± sd over the 3 seeds,
or, for exact identities, the maximum error over seeds.
Rerun: `cd study/T1/w3; for s in 1 2 3; do python3 check_L6.py --seed $s; PYTHONPATH=../../../../wealth-net/.pylib python3 check_L7.py --seed $s; done; python3 aggregate.py`

---

## Lemma 6: a fund of funds with fixed fractions is a flat mixture; only rebalancing adds power

**Setup.** Outcomes y_t; a price pi_t over outcomes (Arrow securities, exogenous or endogenous).
A leaf trader l with belief p_l,t bets full Kelly: its wealth relative is x_l,t = p_l,t(y_t)/pi_t(y_t).
A fund f holds claims on children c (leaves or funds) with fractions A[f,c] of its capital.

**6.1 Buy-and-hold is a flat Bayes mixture. [theorem]** If every fund buys at t = 0 and never
trades, then for every node f and every t

    W_f,t / W_f,0 = sum_l lambda_fl * W_l,t / W_l,0,   lambda_f = row f of (I - A)^-1, restricted to leaves,

that is, the sum over paths of the products of edge fractions. This also holds for cyclic
cross-holdings when the spectral radius of the fund-to-fund block is below 1 (the Leontief
inverse). So f is one Kelly trader whose belief is the Bayesian posterior mixture
p_f,t = sum_l w_l,t p_l,t, with w_l,t proportional to lambda_l prod_{s<t} p_l,s(y_s).
Consequences: log-regret to leaf l is at most ln(1/lambda_l) (Beygelzimer, Langford, Pennock 2012,
https://arxiv.org/abs/1201.6655), and the asymptotic growth rate is the best leaf's. With an
endogenous wealth-weighted price, a market of buy-and-hold funds has exactly the price path of the
flat market of their leaves: consolidation does not move prices.
*Proof:* the value of held shares is linear in the children's values; induct over the DAG, or solve
V = A V for the cyclic case.

**6.2 Constant rebalancing is a flat CRP. [theorem]** If every fund rebalances to fixed fractions of
its current value at every step, then x_f,t = sum_l lambda_fl x_l,t with the same path-product
weights. The root is one constant-rebalanced portfolio over the leaves. Its belief is the fixed
linear opinion pool sum_l lambda_l p_l,t. Hierarchy again adds nothing beyond the flat weights.

**6.3 Where power comes from. [theorem for the toy; known for CRPs]** A fixed-fraction hierarchy
never adds power; only the rebalancing rule changes growth. A CRP can strictly beat every leaf and
every buy-and-hold fund. Toy (truth 0.5, price 0.3, leaves 0.9 and 0.1): each leaf and the
buy-and-hold fund grow at -0.424 nats/step, while the 50/50 CRP grows at +0.087. The best CRP in
hindsight is at least as good as the best leaf, and causal universal or EG portfolios track it
(Cover 1991, "Universal portfolios", unverified; Helmbold, Schapire, Singer, Warmuth 1998, EG
portfolio, unverified). Hierarchy matters only when an inner node's allocation is a *nonlinear*
function of its own children's history (adaptive rebalancing, for example a nested Fixed Share,
Herbster and Warmuth 1998, unverified). Even then, the recursive-Bayes case is a flat mixture under
a structured prior (Context Tree Weighting, Willems, Shtarkov, Tjalkens 1995, unverified).
**[conjecture]** A nested Fixed Share over a tree equals a flat switching mixture whose transition
prior is shaped by the tree.

**Checks** (`out/summary.json`, `out/L6_growth.csv`; 10 leaves: KT context counters k = 0,1,2,3,4,6,8,
an online logistic on a 16-step window, and constants 0.9 and 0.1; price = order-0 KT base rate; T = 6000, Q = 64):

| check | result | pass |
|---|---|---|
| C6a random 3-level DAG, buy-and-hold vs flat mixture | max \|Δ log W\| 8.9e-16 | yes |
| C6a root as ONE Kelly trader with the Bayes-mixture belief vs flat | 1.8e-11 (over 6000 steps, log W up to ~2600 nats) | yes |
| C6a cyclic cross-holdings (ρ = 0.24) vs Leontief weights | 1.8e-15 | yes |
| C6b constant-rebalanced DAG vs flat CRP | 9.1e-13 | yes |
| C6c endogenous price: buy-and-hold funds vs flat leaves | max \|Δπ\| 6.7e-16 | yes |
| C6c the same with rebalancing funds | max \|Δπ\| 0.73 ± 0.08 (prices differ, as predicted) | yes |
| C6d buy-and-hold uniform minus best leaf, per question | min -3.84e-4, max -2.7e-4 ± 0.3e-4 nats/step; bound -ln(10)/T = -3.84e-4 | yes |

C6d, growth per step against the market (nats/step, mean ± sd over seeds; the full table is
`out/L6_growth.csv`):

| family | best leaf (hindsight) | buy-and-hold | online EG rebalancing | EG − best leaf | best CRP (hindsight) |
|---|---|---|---|---|---|
| ca110 | 0.134 ± 0.020 | 0.134 ± 0.020 | **0.147 ± 0.021** | **+0.0132 ± 0.0012** (96% of questions) | 0.143 ± 0.020 |
| periodic | 0.436 ± 0.012 | 0.436 ± 0.012 | 0.432 ± 0.012 | -0.0043 ± 0.0001 | 0.436 |
| prime | 0.044 ± 0.009 | 0.044 ± 0.009 | 0.040 ± 0.009 | -0.0039 ± 0.0001 | 0.044 |
| and / or | 0.017 / 0.017 | 0.016 / 0.016 | 0.015 / 0.016 | -0.0011 / -0.0003 | 0.018 / 0.019 |
| ca30 | 0.00006 | -0.0003 | -0.0025 | -0.0026 ± 0.0005 | 0.0025 (in-sample only) |
| lfsr16, pi | 0.0000 | -0.0004 | -0.003 | -0.0036 / -0.0027 | 0.0004-0.0005 (in-sample only) |

Reading: buy-and-hold is always within ln(N)/T of the best leaf, as 6.1 says. Causal rebalancing
adds real power on CA110 only, where different context lengths win on different stretches, and it
pays an O(√T)-type cost elsewhere. On CA30, LFSR and pi, the best CRP in hindsight beats every leaf
on 100% of CA30 questions, but that is in-sample memorisation: the causal EG loses there. This is the
settled premise "in-sample books pay memorisation" in portfolio form.

**What this means for the study.**
- Kelvin's "a big trader's wealth is the sum of its constituents", with fixed hierarchical or partial
  ownership, is case 6.1. Consolidation is accounting: it changes no prediction, no price and no
  selection. It is a view, not a mechanism. Additive a·dR/da books consolidate the same way.
- **K1**, funds of funds: fixed fractions must reproduce the flat economy exactly. Use
  `w3/check_L6.py:bh_fund_values` and `flat_weights` as an assertion (tolerance 1e-9). Only rebalancing
  funds can change survivors or Brier; report EG or universal rebalancing, not only uniform
  fractions, which cost growth (the uniform CRP gets 0.20 vs 0.44 on periodic).
- **M8** gating and any per-step wealth-driven allocation are rebalancing rules, so this is where any
  power has to come from. A wealth-proportional *linear* pool in probability space is exactly
  buy-and-hold (Bayes), with nothing beyond Bayes. **M6**(d) pools in log-odds (a geometric pool) and is
  outside this lemma. A neural logit sum is a product of experts, not a mixture.

---

## Lemma 7: the top-k router gradient is gate-weighted relative revenue under censored feedback; the router is a broker

**Setup.** h = sum_{i in S} g_i E_i(x), S = top-k(z) (piecewise constant, held fixed a.e.).
r_i = <E_i(x), -dL/dh> is the revenue expert i *would* earn at unit gate; it is defined for every
expert, but observed only for i in S. R = sum_S g_i r_i = <h, -dL/dh> is the revenue paid to the block.

**7.1 Router gradient. [theorem; two-line softmax product rule]** u_i := -dL/dz_i equals:

| convention | u_i |
|---|---|
| (i) softmax renormalised over S | g_i (r_i − R) for i in S; 0 outside |
| (ii) softmax over all N, gates of S not renormalised | g^N_i (r_i·1[i∈S] − R): outsiders get −g^N_i R, their revenue imputed as 0 (censored) |
| (iii) sigmoid, normalised over S (the DeepSeek-V3 form, unverified detail of https://arxiv.org/abs/2412.19437) | (1 − σ(z_i)) g_i (r_i − R) for i in S |
| (iv) top-1 renormalised | 0 (the gate is identically 1; the router never learns) |
| (v) top-1 not renormalised (Switch, https://arxiv.org/abs/2101.03961) | selected: g(1−g) r_sel; outsiders: −g^N_i g_sel r_sel |
| dense softmax (full information) | g_i (r_i − R): Hedge / replicator step on revenue |

Check C7a: the closed form matches autograd to at most 8.6e-16 relative error for all six conventions,
3 seeds (`out/summary.json`, keys `C7a_*`).

**7.2 Zero-sum and mass transfer. [theorem]** For softmax gates, sum_i u_i = 0 (shift invariance),
so router updates are pure reallocation (C7b: ≤ 1.0e-18, 3 seeds). In (ii) the routed experts jointly gain
R(1 − sum_S g) and the outsiders jointly lose the same (checked ≤ 4.3e-19). When the routed
portfolio earns (R > 0), logit mass flows from *every* unrouted expert to the routed ones, whatever
the outsiders' own r_i. Rich-get-richer is an identity, not a tendency. Sigmoid gates are not
shift-invariant and are not zero-sum (|sum| 3.3e-4 to 7.9e-4 against a gradient scale of 4.4e-4 to 8.1e-4, 3 seeds). This refinement
was written into `check_L7.py` before the run; PREREG's C7b criterion applies to softmax
conventions only.

**7.3 The router bias is a censored Hedge log-wealth. [theorem under SGD]** With z = W_r x + b and
plain SGD at rate η, b_i(T) − b_i(0) = η Σ_t Σ_tokens u_i: cumulative gate-weighted relative revenue,
with unobserved revenue imputed as zero. Under Adam it is sign-normalised and not a wealth. The
DeepSeek loss-free bias (https://arxiv.org/abs/2408.15664) enters only the top-k selection and is
updated by load, not revenue (recollection, unverified). It is a capacity price, orthogonal to this ledger.

**7.4 Censoring and lock-in. [theorem for a bias-only router with top-1 not renormalised; the rest is
empirical]** Under (v), if the incumbent's revenue stays positive, its logit rises and every
outsider's falls, so the router keeps the *first* expert with positive revenue forever. Toy C7e
(16 experts, μ_i ~ U(0.1, 1), noise 0.3, η = 0.1, 5k steps, 100 instances per seed; `out/summary.json` key `C7e`):

| arm | final argmax = best expert | best routed, last 1k steps | regret, last 1k |
|---|---|---|---|
| top-1 not renormalised, censored | 6.7 ± 2.5% (chance 6.25%) | 6.7 ± 2.5% | 0.378 ± 0.014 |
| top-1 + Exp3 shadow credit | 52 ± 4% (the rest are near-ties) | 52 ± 4% | **0.023 ± 0.002** |
| top-2 renormalised, censored | 88 ± 3% | 98 ± 2% | 0.089 ± 0.002 (includes the 2nd slot; not comparable) |

So collapse from censoring is sharp for top-1 and for non-renormalised gates while R > 0. From
(ii), two incumbents can both rise while their r_i > R, so the same lock-in holds (derived, not run).
It is not sharp for renormalised top-k ≥ 2 with a stationary bias-only router: relative revenue inside
S evicts the weaker incumbent (**exploration by eviction**). Context-dependent routers and experts
that keep learning are M7's question, not this toy's.

**7.5 Exp3 shadow credit is unbiased for the full-information gradient. [theorem; standard
importance weighting]** Sample one unrouted expert J with probability q_J and run it at zero gate
(+1/k FLOPs, no effect on h). Set r̂ = r on S, r̂_J = r_J/q_J, and 0 elsewhere. Then
û_i = g^N_i (r̂_i − Σ_j g^N_j r̂_j) has expectation equal to the dense-softmax gradient. C7d: max |z|
over 8 experts = 1.17, 1.47, 1.82 (seeds 1-3; 1000 draws × 256 tokens). The censored top-2 gradient
is 43 ± 23% away (relative L2; 35%, 68%, 24% by seed) from the full-information one. Exp3: Auer, Cesa-Bianchi, Freund,
Schapire 2002 (unverified).

**7.6 Gate convention: the router is a commission-free broker. [theorem (Euler) + convention]** At
each gate, the tangent books sum to Σ_i g_i ∂R/∂g_i + Σ_i ⟨E_i, ∂R/∂E_i⟩ = 2R, because the product is
homogeneous of degree 2 in (g, E). C7c: ratio in [2 − 1.6e-13, 2 + 7e-14] for every convention; the
SwiGLU (gate·up) factor books are 2.0 × the product books. **Convention:**
- **(a) Broker, adopted.** Expert i is paid g_i r_i. This conserves: the sum is R (residual ≤ 2.6e-18),
  and it keeps conserving into the expert's neurons plus output bias (residual ≤ 2.6e-18). The router
  earns nothing from the product. Its ledger is the zero-sum reallocation u_i of 7.1-7.2: routing
  preference, not wealth.
- **(b) Euler half split, sensitivity only.** It conserves (≤ 2.2e-18) but pays the router g_i r_i/2
  of *level* revenue for the expert's work.
- **Router skill score, not wealth.** The router's tangent logit book z·(−∂L/∂z) = Cov_g(z, r) over S
  (identity checked ≤ 2.6e-18). It is positive exactly when higher logits went to higher-revenue
  experts, and it lies outside the conserved flow.
- **Scope.** The same doubling hits GLU FFNs and QK attention. Wealth-net's ReLU MLP avoids it.

**7.7 Routed-expert wealth is sleeping-expert wealth. [known]** Expert books accrue only on routed
tokens. This is the specialists / sleeping-experts setting (Freund, Schapire, Singer, Warmuth 1997,
unverified). Cross-expert wealth comparisons (M7 bankruptcy, K1) must normalise by routed-token count
or use shadow credit.

**For M7.** Stage 0 is done here: `w3/check_L7.py` (functions `gates`, `closed_form`, `check_exp3`)
can be imported. Use convention (a) with (b) as sensitivity. Expect lock-in in no-balancing arms only
for top-1 or non-renormalised gates; top-2 renormalised explores by eviction. The Exp3 arm estimates
the dense-softmax gradient without bias.

**Prior art found (search, 2026-09-27; summaries not verified by reading the papers).**
- "When Are Experts Misrouted?" (https://arxiv.org/abs/2605.07260) frames the missing counterfactual
  losses of unrouted experts, the closest match to "censored feedback".
- ProbMoE (https://arxiv.org/abs/2606.01509) notes that gradients reach only the selected softmax
  weights.
- Router-noise exploration: https://arxiv.org/abs/2609.13058.
- Limited-observation experts regret (Kale 2014, via https://arxiv.org/abs/2606.14929).
- Noisy top-k gating (Shazeer et al. 2017, https://arxiv.org/abs/1701.06538, unverified id).
- SparseMixer (https://arxiv.org/abs/2310.00811, unverified id).

No source found states 7.2's exact mass-transfer identity, 7.6's broker convention or the
eviction/lock-in split (search only; L1 should confirm).
