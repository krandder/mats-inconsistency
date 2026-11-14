# PREREG-FA: World FA as built — 2026-10-28

Forecast aggregation with a hidden, correlated information structure. The target is the exact Bayes predictive for
the fundamental value v; the difficulty is only the information structure. Spec: `SPEC.md` (v0, 13:20Z). Critique:
`r1-{codex,gemini,kimi}.md`; the coordinator's rulings on it are applied in section 9. This file is written before
any training. Later changes go to the Amendments section at the end, not into the text above it.

Code (all in `work/world-fa/`; nothing in `work/worlds/` was edited):
`fa_core.py` generator, tokens, exact Bayes, references, beta* (torch, CPU or GPU, float64 maths) ·
`world_fa.py` harness world · `fa_brute.py` independent brute-force / Monte Carlo references (tests only) ·
`test_fa.py` unit tests · `audit_fa.py`, `sweep_fa.py`, `gate_scan.py`, `pilot_gap.py` audits · `audit/` outputs.

    cd work/worlds
    PYTHONPATH=../world-fa python harness.py train   --world world_fa --world-cfg '<CFG>' --nmin 4 --nmax 4 ...
    PYTHONPATH=../world-fa python harness.py evalset --world world_fa --world-cfg '<CFG>' --Ns 4 \
        --policies T0,T8,T32,T128,shufT32 --out runs/es
    PYTHONPATH=../world-fa python harness.py eval    --run runs/X --evalset runs/es

## 1. Generative process (every prior)

N traders, T resolved history rounds, one query round. Per round v_t ~ N(0, 1); report r_{t,i} = v_t + e_{t,i}, with
e_t ~ N(0, Sigma) iid over rounds. History rounds show every report and v_t; the query round shows the N reports.
Target: v of the query round. Sigma is drawn once per episode. Trader ids and round ids are random unit vectors in
R^32, fresh per episode.

Three priors over Sigma (world cfg `prior`). Each prior's data are generated from that prior, so each Bayes is exact
for its own world:

**(f) Enumerated finite family (main for v1, after the critique).** A list of templates; each template is a set of
blocks [size, rho, n_negative] (the other traders independent), one marginal variance `marg` for every trader, an
optional common source `common` (added to every entry), a weight `w`, and a `train` flag. The family is every
distinct relabelling of every template (a block's global sign flip is the same Sigma); a template's weight is split
uniformly over its relabellings. M = family size is the explicit difficulty knob against T. Within a block,
Sigma_ij = marg rho s_i s_j (s = +-1). Default: GPT-6's isolating case (N = 4, one pair, rho = 0.8: M = 6).

**(a) Source-graph DP family (large perm-closed family).** A random partition into blocks of size <= 4 with
p(partition) proportional to prod_B alpha_|B|. Each block draws a marginal level s2_B from A_M and, if |B| >= 2, a
shared fraction f_B from A_F = {.25, .5, .75, .9, .97} (prior pi_F) and per-member signs (P(-1) = a_psign);
Sigma_B = s2_B [(1 - f_B) I + f_B s s^T]. Equal marginals within a block; near-copies at f = .97, no exact ties.
Chosen regime (section 6): A_M = {1, 3, 8} (Var v = 1), alpha = (1, 1, 2, 4), pi_F prop. to (1, 1, 1, 3, 8),
a_psign = 0.25. 67-82% of traders sit in a shared block for N = 3-8 (86% at N = 12).

**(b) Inverse Wishart (separate world).** h ~ finite list (delta, sig2, rho); nu = N + 1 + delta (delta > 0 is
asserted, so nu > N + 1), Psi = delta M, M = sig2 [(1 - rho) I + rho 11^T], Sigma ~ IW(nu, Psi). E Sigma = M and every
k x k sub-block's law is independent of N. Chosen regime: delta = 2, sig2 = 4, rho in {0, 0.6} with weights 1/2
("is there a common source?"). IW does not represent sparse source structure; it is the scalable N-extrapolation
world, not the structured target.

Optional held-out structures (memorisation check): (f) templates with `train: false`; (a) `a_train_alpha` (block
sizes with weight 0 never appear in training). Training episodes exclude them; Bayes always uses the full prior; the
reference `bayes_train` is Bayes under the training part.

## 2. Tokens

Main (`tok_mode: report`), one token per (trader, round) report, plus one query token, no positions:

    [ trader id (32) | round id (32) | r | v | e = r - v | hist | query-report | query | log N | log(1 + T) ]   F = 72

History token (t, i): uid_i, rid_t, r_{t,i}, v_t, e_{t,i} (explicit error, critique item 4), hist = 1. Query-round
token i: uid_i, rid of the query round, r_i, query-report = 1. Query token (last): query flag. log N and log(1 + T) on
every token. Length N (T + 1) + 1. Binding of history to query is only through the shared trader id.

Comparator (`tok_mode: column`, Gemini): 2N + 1 tokens. History token i: [uid_i | h_i | hist | log N | log(1 + T)]
with h_i = T^-1/2 sum_t e_{t,i} u_t, u_t = sqrt(32) x round id t (shared random per-round features, so
<h_i, h_j> / 32 ~ S_ij / T); query token i as in the main mode. `tok_err: false` removes the error channel.

## 3. Exact Bayes

**(f)** p(h | history, r) prop. to p(h) N(history errors; 0, Sigma_h) N(r; 0, Sigma_h + 11^T): the query reports'
marginal is part of the weights (critique item 1; a test shows our weights equal an independent enumeration with
that factor and differ from history-only weights). p(v | h, r) = N(m_h, V_h), V_h = 1 / (1 + 1' Sigma_h^-1 1),
m_h = V_h 1' Sigma_h^-1 r. Batched over episodes x hypotheses with cached factorizations (cost E M N^2); output the
M-component mixture (a grid mixture above 512 hypotheses). Also beta* = E[Sigma^-1 1 | data] per trader (dense
auxiliary target, critique item 5).

**(a)** p(v | data) prop. to N(v; 0, 1) sum_P prod_B alpha_|B| g_B(v), g_B(v) = sum over the block's levels of
prior x N(history errors of B) x N(r_B - v 1) (Woodbury, history through the scatter matrix): the query factor is
inside g_B. The sum over partitions is a DP over subsets on an adaptive v grid (384 points, first-order kernel
deconvolution). Exponential in N: exact up to N ~ 12-14 (6 ms/episode at N = 4, 60 ms at N = 7 on one CPU thread,
more with signs).

**(b)** Sigma | history ~ IW(nu + T, Psi + S). Query errors are multivariate t; the quadratic form reduces to the
scalars a = 1' P 1, b = 1' P r, c = r' P r (P = Psi'^-1), and p(v | data) prop. to N(v; 0, 1) (1 + a v^2 - 2 b v + c)^-(nu'+1)/2
is integrated in 1-D: a Gamma scale mixture in lambda, trapezoid rule in log lambda (96 nodes after a range search):
an exact Gaussian mixture, O(N^3 + Q). Hyperparameter weights = matrix-t history evidence x query evidence.

## 4. References

| name | definition |
|---|---|
| `bayes` | exact Bayes (section 3) |
| `bayes_T0` | exact Bayes ignoring history: the history-blind Bayes p(v given r, N, prior); it can still infer structure from the query reports |
| `diag_bayes` | exact Bayes under the same prior with every correlation removed (strongest correlation-blind rule) |
| `prec` | precision weights ignoring correlations, per-trader variance from history (pseudo-count 2) |
| `equal` | equal-weight average, variance of the average error from history |
| `gls_plugin` | GLS with ((N + 2) Sigma_bar + S) / (N + 2 + T) |
| `oracle` | GLS with the true Sigma (cheating bound) |
| `prior` | N(0, 1) |
| `bayes_true` | shuffled-history cells: Bayes on the real episode |
| `bayes_train` | held-out structures configured: Bayes under the training part of the prior |

## 5. Evaluation cells (harness policies)

`T<t>` fixed history length; `mix` training mixture; `shufT<t>` misbinding control (history columns permuted among
traders; `bayes` = Bayes on the shuffled input, `bayes_true` on the real one; read the model against both);
`freshT<t>` held-out structures; `dedup<k>T<t>` one-hop dedup, graded f = 0 .. .97 with common random numbers and
equal marginals (coordinator lesson from W2c); `gate<k>T<t>` history gate (identical query reports, histories from
f = .97 .. 0; read KL shifts, not NLL). Relabelling invariance (same episode, permuted id assignment) is measured by
`diag_fa.py` on a trained model (model KL between the two labelings; Bayes's is 0 by the invariance test).

## 6. Rewiring audit (before training)

Metric (Gemini's): miss = 1 - R^2 of the best affine fit of Bayes's predictive mean on a structure-blind rule's mean,
on the training prior. Against `prec` (spec), `diag_bayes` (stricter) and jointly (`diag_bayes`, `bayes_T0`) (the
in-context structure share). Also NLL gaps (nats), which the critique asks for (GPT-6: measure excess log loss, not
only mean variance).

Findings.
1. Plain precision weighting conflates structure with nonlinear heavy-tail shrinkage (under IW it misses up to 40%
   at T = 0, where no structure can be known); `diag_bayes` removes that confound.
2. (b): pure IW (rho = 0) reaches miss >= 0.15 only at sig2 = 8 (45 regimes, `audit/sweep_b.jsonl`); a common
   source drives it.
3. (a) without signs tops out at miss 0.14 (noise scale, bigger blocks, more near-copies; `sweep_a`, `sweep_a2`);
   making shared blocks individually accurate lowers it (`sweep_a3`). Dedup changes Bayes's confidence much more
   than its mean (NLL gaps 0.10-0.25 nats at miss 0.04-0.12). Signed loadings (errors cancel inside a block) lift it
   to 0.22 (`sweep_a4`).

Chosen regimes, final audits (`audit/a_chosen.json`, E = 1500, N 3-7, T 0-32; `audit/b_d2s4_mix.json`, E = 8000,
N 3-16, T 0-32):

| | miss vs diag_bayes | miss vs prec | joint miss (diag, T0) | NLL gap diag | NLL gap bayes_T0 (history value) | NLL gap oracle |
|---|---|---|---|---|---|---|
| (a) chosen | **0.208** | 0.199 | 0.175 | 0.415 | 0.345 | -0.169 |
| (b) chosen | **0.225** | 0.235 | 0.202 | 0.632 | 0.193 | -0.246 |

By T (miss vs diag_bayes): (a) T0 0.12, T1-4 0.20, T5-16 0.23, T17+ 0.22; (b) 0.11, 0.21, 0.24, 0.25. Gate passes
for both (>= 0.15).

History gate (KL between Bayes predictives, identical query reports, f = .97 vs 0 histories): (b) N = 8, T = 16,
k = 3: 0.34 nats (k = 4: 0.54); (a) N = 6, T = 16, k = 3: 0.13 nats — in (a), near-copies are already visible in
the query round, so history adds less (`audit/gate_scan_*.txt`). One-hop dedup, Bayes's log predictive-variance
ratio f = .97 vs f = 0: (b) 0.45, (a) 0.19.

**Pilot families (coordinator item 6; `audit/pilot_gap.txt`, E = 3000 per cell).** History value = NLL(bayes_T0) -
NLL(bayes), nats:

| family | M | T = 0 | T = 8 | T = 32 | T = 128 | P(KL >= 0.3) at T = 32 | oracle gain over bayes_T0 at T = 0 |
|---|---|---|---|---|---|---|---|
| GPT-6: N = 4, one pair rho = .8, marg 1 | 6 | 0 | 0.018 | 0.021 | 0.021 | 0.01 | 0.026 |
| same, rho = .95 | 6 | 0 | 0.012 | 0.017 | 0.017 | 0.01 | 0.016 |
| same, marg .25 | 6 | 0 | 0.024 | 0.025 | 0.025 | 0.01 | 0.030 |
| N = 4, triple rho = .8 | 4 | 0 | 0.015 | 0.017 | 0.013 | 0.00 | 0.015 |
| **N = 4, anti-correlated pair (rho = .8, one sign flipped)** | 6 | 0 | **0.202** | **0.239** | **0.227** | 0.24 | 0.221 |
| Gemini: N = 8, 5-clique rho = .8, marg 1 | 56 | 0 | 0.052 | 0.045 | 0.044 | 0.02 | 0.043 |
| N = 4, pair rho = .8 or anti-correlated pair (M = 6 + 6) | 12 | 0 | 0.147 | 0.159 | 0.175 | 0.14 | 0.170 |

**Blocker for the pilot as specified:** GPT-6's case has almost nothing to learn: a positively correlated pair among
four equal-variance traders moves the GLS weights only from 1 to 1/(1 + rho), and the query reports already reveal
part of it, so the full-vs-history-blind gap is 0.02 nats (even the oracle gains only 0.026 nats). The
anti-correlated pair keeps every isolating property (6 relabellings, identical marginals, identical spectra
{1 +- 0.8}) and has a 0.20-0.24 nat history value with 24% of episodes above 0.3 nats. Pilot plan in section 10 uses
it as the main arm and keeps the GPT-6 case as a null arm. Gemini's N = 8 5-clique has the same problem in milder
form: the correlation-blind rule loses 0.6-0.67 nats, but the history-blind Bayes finds the tight 5-cluster in the
query reports, so history adds only 0.045 nats (its signed and low-noise variants: Amendment A1).

## 7. Tests (`OMP_NUM_THREADS=1 python -m pytest -q test_fa.py`): 54 passed, 2 skipped (by design)

- (a) Bayes = brute-force enumeration of every partition x level x sign hypothesis (independent scipy code, query
  marginal included): mean 5e-5, variance 1e-4 relative, log evidence 1e-6, log density <= 2e-3 nats within +-3 sd
  (N = 1-4, T = 0-7, five configurations incl. signed, unequal alpha, unequal level priors, the chosen regime).
- (b) Bayes = direct v-grid quadrature of N(v) x scipy multivariate-t (mean, variance 1e-6; log density 1e-5) and =
  importance sampling over IW draws (two-hyperparameter mixture; within MC error; log evidence 0.03).
- (f) Bayes = independent enumeration incl. N(r; 0, Sigma_h + 11^T) (weights 1e-9, evidence 1e-8); history-only
  weighting differs by > 0.05 in weights (query-marginal test); family sizes (6, 6, 56, ...).
- Generator <-> Bayes calibration (a, b, f): E(v - m)^2 = E var within 4 SE and PIT uniform (KS p > 1e-3); Bayes has
  the lowest NLL among non-oracle references.
- Reductions: T = 0 equals history-free Bayes and depends on the query reports only as a multiset; IW with
  delta -> infinity = GLS with M; IW with T = 20000 -> GLS with true Sigma; (a) one-hypothesis family = GLS;
  (a) singleton family with long history = precision weighting with true variances; N = 1: diag_bayes = bayes.
- Consistent relabelling of traders leaves Bayes and every reference unchanged and permutes the token multiset;
  permuting history columns relative to the query changes Bayes (binding).
- Tokens: report layout incl. e, log N, log T; column mode h_i formula; dedup/gate sets (common random numbers,
  equal marginals, identical query reports); shuffled control; held-out structures never in training batches;
  bayes_train worse than bayes on fresh cells; beta* = Sigma^-1 1 once identified; T curriculum.

## 8. Known limits

1. Discrete hidden structure with random ids must be summed over relabellings: (a) is exponential (subset DP), (f) is
   exact only while M is enumerable. N-scaling to 64+ runs on (b) only.
2. No common source in (a) (it couples every block in every round); (f) has it.
3. Near-copies only (f <= .97 in (a); rho < 1 in (f)); no exact ties.
4. Gate and dedup cells are diagnostic: read shifts, not NLL.
5. Report tokens: N = 64, T = 128 is 8257 tokens; evalsets need `--eps-fixed`.
6. beta* is exact for (f), history-only for (b), absent for (a). The beta* arm needs a per-trader head (not in the
   harness); the target is ready, the arm is not built.

## 9. Critique rulings applied (coordinator, 2026-10-28)

1. Query marginal in mixture weights: already the case in (a), (b) and (f); explicit test added.
2. IW: nu > N + 1 asserted; scalar reduction (a, b, c) with a 1-D log-space integral was already the method; IW kept
   as a separate world generated from IW; the structured family is the main Bayes target.
3. Structured family: (f) enumerated, GPU-batched, optional common source, grid variances (marg per template),
   M as the knob, held-out templates for fresh-structure (memorisation) cells; (a) keeps the large DP family.
4. Tokens: explicit error channel; Gemini's 2N comparator (`tok_mode: column`); log N and log(1 + T) inputs; N and T
   vary per episode in training (`--nmode batch` varies N per batch).
5. Head: harness mixture head with Gaussian NLL; beta* target available; T curriculum large -> small (`T_curr`).
6. Pilot: section 6 shows GPT-6's case has a 0.02-nat history value; section 10.

## 10. GPU pilot plan (pre-registered; run on the coordinator's go, cap $12, on-demand L4, VM deletion verified)

World (f), N = 4. Arms: family {anti-correlated pair (main), GPT-6 positive pair (null)} x tokens {report, column}
x seed {0, 1} = 8 runs; depth 2, width 64, heads 4, mixture 4, 20000 steps, bs 256, T ~ 10% T = 0 else log-uniform
1-128 after a curriculum (steps 0-6000: T in [32, 128], no T = 0). Eval cells T0, T8, T32, T128, shufT32 (2000
episodes each). Read-outs per arm:
- fraction of the history value recovered at T = 8, 32, 128: (NLL_bayes_T0 - NLL_model) / (NLL_bayes_T0 - NLL_bayes);
- relabelling invariance: mean KL(model(ids) || model(permuted ids)) on T = 32 episodes (Bayes: 0);
- misbinding: on shufT32, model NLL against bayes_T0 and bayes_true (a binding model loses its gain);
- null arm: the same numbers where the history value is 0.02 nats (sanity: no spurious gain, no damage).
Expected outcome is not pre-judged; success = fraction >= 0.5 at T = 32 in the main arm with relabelling KL < 0.01
and the gain gone under shufT32. Then Gemini's N = 8 family in the variant with a substantial history value (Amendment A1).

## Amendments

**A1 (2026-10-28 17:20Z, before any Gemini-family training).** Gemini's N = 8 family, history value
NLL(bayes_T0) - NLL(bayes) in nats (`audit/pilot_gap.txt`, E = 1500):

| variant | M | T = 8 | T = 32 | T = 128 | P(KL >= 0.3) at T = 32 | diag gap at T = 32 |
|---|---|---|---|---|---|---|
| 5-clique rho = .8, marg 1 | 56 | 0.052 | 0.045 | 0.044 | 0.02 | 0.672 |
| 5-clique rho = .8, marg .25 | 56 | 0.061 | 0.053 | 0.052 | 0.03 | 0.723 |
| 5-clique, 2 of 5 anti-loaded, marg 1 | 560 | 0.209 | 0.236 | 0.231 | 0.25 | 0.374 |

The follow-up after the N = 4 pilot uses the signed 5-clique (M = 560, Bayes as a 256-point grid mixture since
M > 512), with the unsigned 5-clique as its null arm, same read-outs as section 10.

**A2 (2026-10-28 18:45Z): N = 4 pilot result (`runs/fa-pilot-1028b/`, L4 us-west4-c, 2.63 h, $2.29 ceiling, VM
deleted 18:37Z: GPU-LOG DELETE line and an empty `gcloud compute instances list`).** 8 arms, 20000 steps, depth 2,
width 64 (105k parameters). `diag_fa.py`, E = 2000 per T, NLL gaps in nats:

| arm | T = 0 gap to Bayes | history value (T = 8 / 32 / 128) | model gap to Bayes (T = 8 / 32 / 128) | fraction recovered at T = 32 | relabel KL | shufT32 gain over bayes_T0 |
|---|---|---|---|---|---|---|
| anti, report, s0 / s1 | 0.010 / 0.011 | 0.208 / 0.218 / 0.248 | 0.218 / 0.224 / 0.262 (both seeds within 0.002) | -0.03 / -0.03 | <= 4e-4 | -0.006 |
| anti, column, s0 / s1 | 0.003 / 0.005 | same | 0.215 / 0.219-0.224 / 0.253 | -0.01 / -0.03 | 1e-4 | -0.001 / -0.006 |
| null, report / column | 0.004-0.007 | 0.014 / 0.020 / 0.019 | 0.017-0.018 / 0.022-0.025 / 0.023-0.029 | -0.12 .. -0.27 | 1e-4 | -0.002 .. -0.005 |

Reading: every arm learned the history-blind Bayes almost exactly (T = 0 gap 0.003-0.011 nats; at T > 0 its gap
equals bayes_T0's within 0.01) and recovered none of the history value; shuffling the history changes nothing,
consistent with the history being ignored; relabelling invariance holds. Validation gap plateaus from step ~8000 at
the history-blind level in both token designs. Pre-registered success (fraction >= 0.5 at T = 32) not met.

**A3 (2026-10-28 18:55Z, before any of these runs): binding arms (coordinator's go; cap $6, on-demand, verify
deletion).** Reading of A2: the 0% history value with exact history-blind Bayes is the source-binding plateau
astra found in the tiny world (`outbox/tiny-pfn-plateau-2026-10-28.md`, `outbox/binding-plateau-paper-2026-10-28/`):
stalled models do not bind history columns to query traders.

Fixed in every arm: world (f), anti-correlated pair, N = 4, M = 6, report tokens with the explicit error channel;
SetPFN depth 4, width 64, heads 4, mixture 4 (`harness.SetPFN`); AdamW lr 5e-4, weight decay 0.01, warmup
steps/60, cosine to 0.1, gradient clip 1; TF32 matmuls; trainer `train_fa.py`. One history length per batch, drawn
from the curriculum stage (no padding): first 30% of steps T log-uniform in [32, 64] ("long histories first"), then
T = 0 w.p. 0.1 else log-uniform in [1, 64]. T_max is 64 (the A2 pilot used 128) to fit the budget; the history value
at T = 32 / 64 is ~0.22 nats, and T = 128 is evaluated as extrapolation. Dense target: beta* = E[Sigma^-1 1 | history,
query] (exact for (f); values 1-5 here), predicted from each query-report token by its own LayerNorm + Linear head,
MSE on beta* / 3, loss weight 1, for the whole run.

| arm | dense target | trader ids | batch | steps | seed |
|---|---|---|---|---|---|
| dense_s0, dense_s1 | yes | random unit | 256 | 60000 | 0, 1 |
| dense_ortho_s0 (binding aid a) | yes | orthonormal within episode | 256 | 60000 | 0 |
| dense_bs1024_s0 (binding aid b) | yes | random unit | 1024 | 15000 (equal examples) | 0 |
| nll_s0, nll_s1 (control: no dense target) | no | random unit | 256 | 60000 | 0, 1 |
| nll_ortho_s0 (aid a without dense target) | no | orthonormal | 256 | 60000 | 0 |

Same seed = same initial weights and the same batch sequence (values; ids differ under the orthonormal option).

Read-outs per arm (`diag_fa.py` on best.pt, chosen by val NLL at T = 0 plus T = 32; 2000 fresh episodes per T):
- **primary: fraction of the history value recovered at T = 32**, (NLL_bayes_T0 - NLL_model) / (NLL_bayes_T0 -
  NLL_bayes); also T = 8, 64 and 128 (extrapolation), and T = 0 gap to Bayes;
- binding attention (astra's diagnostic): per-token attention from trader i's query-report token to its own history
  tokens over other traders' history tokens, ratio of means pooled over heads, traders and episodes, for every layer
  (layer 1 = astra's primary) and every head; 1 = no binding (astra: stalled 1.0-1.24, acquired 14.6-233);
- functional binding (astra's ID-intervention analogue): history columns permuted among traders; model vs Bayes
  mean-change RMS ratio, slope and correlation, and the distribution-change ratio mean J(model) / mean J(Bayes);
- relabelling invariance KL; learning curves every 2000 steps (500 for the batch arm): T = 0 / T = 32 gap,
  fraction at T = 32, attention ratio by layer, dense-target MSE.
"Acquired" = fraction >= 0.5 at T = 32. With one or two seeds per arm these are run descriptions, not acquisition
rates (astra's width-16 baseline acquired in 39/50 seeds).

Budget: one on-demand L4, 7 arms in parallel, per-arm wall budget 3.2 h (an arm that hits it stops, saves and is
reported with its step count), GPU_RUN_TIMEOUT 4.5 h, VM self-deletion at 6 h: worst case 6 h x $0.87 = $5.22.

**A3 deviation (19:35Z).** In run fa-bind-1028d, arm dense_s0 died with a CUDA out-of-memory error during stage 1
(seven processes reserved the L4's 22 GB; the batch-1024 arm holds 6.2 GB). The other six arms continue unchanged.
dense_s0 is rerun alone on a second on-demand L4 (run fa-bind-1028e; identical code except the arm filter, same seed,
hence the same initial weights and batch sequence; wall budget 1.6 h, GPU_RUN_TIMEOUT 2 h). Worst case for both VMs
together stays under the $6 cap (4.5 h + 2 h ceilings: $5.66).

**A4 (2026-10-28 22:05Z): binding-arm results** (`runs/fa-bind-1028d/`, `runs/fa-bind-1028e/` (dense_s0),
`runs/bind-all/SUMMARY.md`). Both VMs deleted (GPU-LOG DELETE lines 21:57Z and 20:12Z; `gcloud compute instances
list --filter='name~soup-gpu-fa-'` returned `[]` at 21:57Z). Cost ceiling $2.55 + $0.57 = $3.12. All seven arms
reached their requested steps (none hit the wall budget). `diag_fa.py` on best.pt, 2000 episodes per T:

| arm | fraction of history value recovered T = 8 / 32 / 64 / 128 | T = 0 gap | attention own/other, layers 1-4 (T = 32); head range | shuffle response RMS ratio / J ratio | dense MSE at end |
|---|---|---|---|---|---|
| dense_s0 | -0.01 / -0.00 / -0.01 / -0.01 | 0.002 | 1.00 / 1.00 / 1.00 / 1.00; 0.993-1.007 | 4e-4 / 2e-7 | 0.274 |
| dense_s1 | -0.01 / -0.01 / -0.02 / -0.01 | 0.002 | 1.00 x 4; 0.999-1.004 | 2e-4 / 3e-8 | 0.272 |
| dense_ortho_s0 | -0.01 / -0.00 / -0.01 / -0.01 | 0.002 | 1.00 x 4; 0.998-1.006 | 3e-4 / 1e-7 | 0.274 |
| dense_bs1024_s0 | -0.03 / -0.01 / -0.02 / -0.03 | 0.001 | 1.00 x 4; 0.994-1.003 | 6e-4 / 4e-7 | 0.279 |
| nll_s0 | -0.02 / -0.00 / -0.02 / -0.02 | 0.002 | 1.00 x 4; 0.999-1.001 | 1e-4 / 1e-8 | - |
| nll_s1 | -0.02 / -0.01 / -0.01 / -0.03 | 0.002 | 1.00 x 4; 0.998-1.000 | 1e-4 / 1e-8 | - |
| nll_ortho_s0 | -0.02 / -0.00 / -0.01 / -0.02 | 0.002 | 1.00 x 4; 0.999-1.001 | 1e-4 / 2e-8 | - |

History value at T = 32: 0.218 nats; SE of each fraction about 0.06. Over all 30 learning-curve evaluations of every
arm the layer attention ratios stayed within 0.996-1.006 and the fraction at T = 32 never exceeded 0.008.
Reading: no arm acquired (pre-registered threshold 0.5). Every arm learned the history-blind Bayes almost exactly
(T = 0 gap 0.001-0.002 nats; at T > 0 its NLL equals bayes_T0's within 0.01) and is the stalled phenotype in astra's
terms, more extreme than hers: query-report tokens put 18-97% of their attention on history tokens (per layer, T =
32; 18-38% in layer 1), spread evenly over own and other traders, and permuting history columns moves the predictive mean by <0.1% of
Bayes's move. The dense per-trader target did not help: its MSE settles at 0.27, the level of the history-blind
beta* (computed: 0.0 at T = 0, 0.38-0.42 at T = 8-64, about 0.3 under the stage-2 mix). Neither orthonormal ids nor
batch 1024 at equal examples changed anything in these single-seed runs. Same-seed reruns are not bit-identical
(GPU nondeterminism; dense_s0 ran alone on a second VM).

**A5 (2026-10-28 22:10Z, before any of these runs): decomposing the conjunction (coordinator's go; cap $5,
on-demand, verify deletion).** Using history needs four steps: (a) group each trader's history tokens; (b) estimate
cross-trader error covariances (pair rounds across traders, aggregate over rounds); (c) turn them into weights
(here: find the anti-correlated pair, weight it up); (d) route the result to the trader's query-report token.
A4 shows the conjunction is not learned. Each arm below gives the model some steps and asks whether the rest is
learnable. Everything not listed is as in A3 (anti-correlated pair, N = 4, depth 4, width 64, heads 4, mixture 4,
AdamW 5e-4 cosine, 60000 steps, batch 256, one T per batch, T curriculum: first 30% of steps T in [32, 64], then
T = 0 w.p. 0.1 else log-uniform [1, 64]); the flat-token baseline is A4's nll_s0 / nll_s1.

| arm | tokens | given | left to learn |
|---|---|---|---|
| 1 binding given (`qcol`) | N query-report tokens + query token, NO history tokens; query token i also carries its own trader's history error column e_{1..T, i} / sqrt(T), zero-padded to 64 rounds (column position = round) | (a), (d) | (b) covariances from aligned columns (an inner product), (c) weighting |
| 2 binding + covariance given (`qcov`) | as arm 1, plus the trader's empirical covariance row attached by trader key: own variance S_ii / T and c_i = sum_{j != i} (S_ij / T) uid_j, with orthonormal trader ids so <c_i, uid_j> = S_ij / T exactly | (a), (b), (d) | (c) weighting |
| 3 flat + pair-correlation target (`pairaux`) | A3's report tokens | dense supervision of (b): every ordered pair's empirical error correlation S_ij / sqrt(S_ii S_jj), predicted from the two query-report tokens' final states by a bilinear head ((A h_i) . (B h_j) / 4, A, B: 64 -> 16), MSE weight 1, T >= 2 | all, with a dense signal for (b)+(d) |
| 4 flat + trader curriculum (`ncurr`) | A3's report tokens | none; N = 2 for the first 20% of steps (family: the pair is anti-correlated or independent, 50/50, M = 2), N = 3 for the next 20% (anti pair, M = 3), then N = 4 | all, starting from 2 traders |

Two seeds per arm (arms 1-2 are 5 tokens per episode, cheap; arms 3-4 like A3's arms). History value of the
curriculum families (NLL bayes_T0 - bayes, T = 8 / 32 / 64): N = 2: 0.079 / 0.115 / 0.097; N = 3: 0.166 / 0.187 /
0.174; N = 4: 0.21 / 0.237 / 0.232.

Read-outs (`diag_fa.py`, best.pt, 2000 episodes per T, N = 4): **fraction of the history value recovered** at T = 8,
32 (primary), 64 (and 128 for flat arms; arms 1-2 hold at most 64 rounds); T = 0 gap; binding diagnostic: own/other
per-token attention ratio from query-report tokens to history tokens (flat arms; given by construction in arms 1-2);
partner ratio (new, all arms): per-token attention from an anti-pair member's query-report token to its partner's
tokens (history tokens in flat arms, the partner's query-report token in arms 1-2) over the non-partner traders'
tokens; functional response to permuting history columns among traders (RMS and J ratios); arm 3: pair-correlation
MSE against the MSE of predicting 0.
Success / partial / fail at T = 32: fraction >= 0.5 / 0.1-0.5 / < 0.1.

Interpretation (pre-registered):

| arm 2 (c left) | arm 1 (b + c left) | arms 3 / 4 (flat) | reading |
|---|---|---|---|
| fail | any | any | weighting from given covariances is not learned: (c) is the blocker; the decomposition stops there |
| success | fail | any | (c) is learnable; covariance estimation from aligned columns (b) is not, even with binding given |
| success | success | both fail | (b) + (c) are learnable once binding is given: binding (a)/(d) in flat tokens is the blocker |
| success | success | arm 3 success | dense supervision of pair correlations rescues flat tokens: the blocker is credit assignment through (b)/(d), not representability |
| success | success | arm 4 success | starting from two traders makes binding learnable: the blocker is the difficulty of first acquiring binding at N = 4 |
| success | success | arm 3 aux MSE falls but its fraction stays < 0.1 | the model can represent pair correlations at the query tokens but does not use them for weighting in flat tokens |

Budget: one on-demand L4, 8 runs in parallel, per-arm wall budget 2.5 h, GPU_RUN_TIMEOUT 4 h: expected cost with
working deletion <= 4.2 h x $0.87 = $3.65 (the 6 h self-deletion is only the backstop if deletion fails).
A5 implementation notes (22:18Z, before launch): arm 2 uses orthonormal trader ids (needed for the exact key read-out);
arms 1-2 are evaluated at T <= 64 (the column holds 64 rounds); the partner ratio is computed for anti-pair members
only; the N = 2 curriculum stage uses f_templates_byN. Run fa-decomp-1028f, on-demand L4 us-east4-a, 8 runs.

**A6 (2026-10-29 00:15Z): decomposition results** (`runs/fa-decomp-1028f/`; on-demand L4 us-east4-a, 1.91 h, $1.66
ceiling; VM deleted 00:13Z, GPU-LOG DELETE line, `gcloud compute instances list --filter='name~soup-gpu-fa-'`
returned `[]` at 00:13Z). All 8 runs reached 60000 steps. `diag_fa.py` on best.pt, 2000 episodes per T:

| arm | fraction recovered T = 8 / 32 / 64 / 128 | T = 0 gap | first eval with fraction >= 0.5 | attention, layers 1-4 (T = 32) | history-shuffle response RMS / J ratio |
|---|---|---|---|---|---|
| 1 binding given, s0 | 0.95 / 0.95 / 0.93 / - | 0.003 | step 8000 | partner 1.8 / 4.7 / 7.1 / 5.5 | 0.97 / 0.87 |
| 1 binding given, s1 | 0.98 / 0.95 / 0.96 / - | 0.005 | step 6000 | partner 2.2 / 2.6 / 2.5 / 4.7 | 0.94 / 0.81 |
| 2 binding + covariance given, s0 | 0.85 / 0.99 / 0.99 / - | 0.004 | step 4000 | partner 3.5 / 36 / 1.3 / 1.8 | 0.97 / 0.83 |
| 2 binding + covariance given, s1 | 0.87 / 0.99 / 0.99 / - | 0.003 | step 4000 | partner 3.8 / 48 / 2.0 / 1.0 | 0.97 / 0.90 |
| 3 flat + pair-correlation target, s0 | -0.02 / -0.01 / -0.02 / -0.02 | 0.001 | never | own 1.00 x 4, partner 1.00 x 4 | 0.000 / 0.000 |
| 3 flat + pair-correlation target, s1 | -0.01 / -0.00 / -0.01 / -0.02 | 0.002 | never | own 1.00 x 4, partner 1.00 x 4 | 0.000 / 0.000 |
| 4 flat + trader curriculum, s0 | -0.02 / -0.00 / -0.02 / -0.03 | 0.005 | never | own 1.00 x 4, partner 1.00 x 4 | 0.001 / 0.000 |
| 4 flat + trader curriculum, s1 | -0.02 / -0.00 / -0.01 / -0.04 | 0.003 | never | own 1.00 x 4, partner 1.00 x 4 | 0.000 / 0.000 |

Arm 3's pair-correlation head ends at MSE 0.213 / 0.218. References on the same stage-2 distribution: predicting 0
gives 0.234, the history-blind expectation E[rho | query reports] 0.212, the true correlation 0.135 (stage 1: 0.125,
0.101, 0.019; the heads sat at 0.102-0.103 there). So the head learned only what the query reports show, nothing from
the history.

Reading, from the pre-registered table: arms 1 and 2 succeed, arms 3 and 4 fail, so (b) covariance estimation and
(c) weighting are learnable once binding is given (both within 4000-8000 steps, both seeds); the blocker is binding,
(a)/(d), in flat report tokens. Neither dense supervision of the pair statistics at the query tokens nor starting from
two traders produced any binding (own/other and partner ratios 1.00 throughout). In arms 1-2 the model attends
preferentially to the anti-correlated partner (partner ratio up to 7 in arm 1 and 36-48 in arm 2's layer 2) and
responds to permuted history columns about as Bayes does (RMS 0.94-0.97 of Bayes's mean change).

**A7 (2026-10-29 00:20Z, before any of these runs): binding remedies on flat tokens, then the SPEC's aggregation
tests on the binding-given design (Kelvin authorised continuing World FA; this step: $5 GPU, about 4 h, on-demand,
verify deletion).** Both parts run on one VM at the same time (part 2 does not depend on part 1's runs; only the
choice of main setup does).

Part 1, flat report tokens (A3 data: anti-correlated pair, N = 4, one T per batch, same T curriculum; read-outs as
A6: fraction of the history value recovered at T = 8 / 32 / 64 / 128, own/other and partner attention ratios by
layer, history-shuffle response). The remedies from the tiny-PFN study (`outbox/tiny-pfn-plateau-2026-10-28.md`):
astra's acquiring configuration (depth 2, width 32, heads 2, lr 1e-3, warmup 1000, 80000 steps at batch 256), the
same with batch 1024 for 20000 steps (equal exposure), and learned trader-ID embeddings shared across rounds and
episodes (`id_mode: slot`: trader i's id is a one-hot slot vector, so the input layer gives it a learned embedding;
slots carry no structural information because the family is closed under relabelling). Two seeds each:

| arm | config | ids |
|---|---|---|
| w32_s0, w32_s1 | depth 2, width 32, heads 2, mixture 4, lr 1e-3, batch 256, 80000 steps | random unit, fresh per episode |
| w32b_s0, w32b_s1 | same, batch 1024, 20000 steps | random unit |
| slot_s0, slot_s1 | same as w32 | learned slot embeddings |

Decision rule: if no flat arm reaches fraction >= 0.5 at T = 32, the arm-1 design (each query token carries its
trader's history error column) becomes the World FA main setup.

Part 2, aggregation tests on the arm-1 design (runs qmain_s0, qmain_s1; depth 4, width 64, heads 4, lr 5e-4, 60000
steps, batch 256, A3's T curriculum). Training family (N traders, every trader's error variance 1, each template
equally likely): one anti-correlated pair (rho 0.8, one loading flipped), one shared source among 3 traders (rho 0.8),
or all independent. One N per batch, log-uniform in [3, 12]. Error columns hold 128 rounds; the T rounds of an
episode go to a random set of column positions shared by all its traders (rounds are exchangeable, so every position
is trained and T = 128 is a full column never seen in training). Tests (`fa_tests.py`, best.pt):
1. scaling: gap to Bayes and fraction of the history value recovered vs T in {0, 8, 32, 64, 128} at N = 8
   (128 beyond training) and vs N in {4, 8, 16, 32, 64} at T = 32 (16-64 beyond training);
2. in-context structure inference: (a) history gate: identical query reports, histories from the shared-source or
   anti-pair structure vs from independent traders; tracking = mean KL(model_A || model_B) / mean KL(Bayes_A ||
   Bayes_B) and the slope of the model's mean shift on Bayes's; (b) fresh structures never trained on (a shared source
   among 4; two anti-correlated pairs), against Bayes under the training family and Bayes that knows the fresh
   structure, plus plug-in GLS;
3. discounting shared information (one-hop dedup): paired episodes, 3 traders on one shared source (rho 0.8) vs
   3 independent traders with the same marginal accuracy, common random numbers, T in {0, 8, 32}: the model's log
   predictive-variance ratio against Bayes's, and its mean response.
Budget: 8 runs in parallel, per-arm wall budget 2.4 h, GPU_RUN_TIMEOUT 3.6 h: expected cost <= 3.8 h x $0.87 = $3.3.
A7 parked 2026-10-29 00:35Z before any run (Kelvin: park World FA); no VM was started.
