# M6 worker 4 pre-registration: arm (g), the fixed-point market layer, vs 1-pass and 2-pass

Written 2026-09-28 ~06:40Z, before any run. Base pass only (no stress: no multi-epoch, no noise
traders, no independent-evidence family). Owner of arm (g); (a) and the kill controls are w1's,
rerun here only as in-harness references on identical streams and trunks.

## Layer (shared by 1-pass, 2-pass, FP; they differ only in the solver)
Trunk: S0 `model.Net` (base: d=64, L=2; LI: d=32, L=1 as S0 w3), residual h_q = lnf(x_q).
Market maker: S0 `Logistic` trained by its own BCE (Adam 1e-2), price l0 detached into the layer.
K=16 traders per question q. With p = sigmoid(l) and price features x_q = [2p_q - 1, m_q],
m = M (2p - 1) (M a learned QxQ mix, init 0, so traders can read other questions' prices):
  b_iq = wb_iq.h_q + cb_iq + ub_iq.x_q         (belief logit)
  k_iq = softplus(wk_iq.h_q + ck_iq + uk_iq.x_q) (confidence, init about 0.13)
  F_q(l) = (l0_q + sum_i k_iq b_iq) / (1 + sum_i k_iq)   (market clears linear edge demands
          d_i = k_i (b_i - l) against the maker's d_0 = l0 - l)
- 1-pass: l = F(l0).  2-pass: l = F(F(l0)) (backprop through both).
- FP (g): l* with F(l*) = l*, joint over the Q questions of one position; Newton on
  (I - J) dl = F(l) - l from l = F(l0), at most 8 iterations, step clipped to +-4, tol 1e-5;
  implicit gradient dl*/dtheta = (I - J)^-1 dF/dtheta (DEQ, https://arxiv.org/abs/1909.01377).
  Logged: iterations used, final residual, fraction not converged.
References (not owned): (a) plain logit sum z = sum_i b_i(h, x=0); (a)+wd (AdamW wd 0.1);
boost: z = l0 + 0.3 * sum_i b_i(h, 0) (market as offset, shrinkage 0.3).
All arms Adam 1e-3, prequential single pass (S0 `run_prequential`), B=4, torch 1 thread.

## Streams, seeds
- base: `streams_ext.make_streams(Q=64, T=24032, seed, heldout=8, release=T//2, regime_every=0)`
  (72 questions, static, held-out periodic family released mid-run). Seeds 1, 2, 3.
- LI: `li_families.make_li_stream(seed, T=20000)` (68 questions). Seeds 1, 2, 3.

## Metrics (prequential trade-time logits)
- M1 learnable-family BCE: mean over families {periodic, lfsr16, ca110, prime, and, or,
  periodic_ho} of the per-family BCE on positions >= n_pos/2 (base stream).
- M2 noise excess: max(BCE - ln 2) over {prng, ca30}, same positions (base stream).
- M3 D coherence: mean of the `pre:` violations (li_families.violations, tol 0.05) over
  {taut_or, contra, impl, ex3} on the LI stream, all live steps; also and/or live violations
  on both streams and LI BCE by family.
- Diagnostics: FP iterations, residual, non-converged fraction; wall time.

## Stop rules (STUDY-PLAN section 3, applied to my arm)
- Drop (g) if its M1 >= (a)'s M1 * 1.01 in 3/3 seeds (paired by seed). Same test reported for
  1-pass and 2-pass.
- Keep the fixed point (vs 2-pass) only if it cuts M3 by >= 50% against 2-pass (mean over seeds,
  and in >= 2/3 seeds). Otherwise report "fixed point not kept", even if it survives the drop rule.
- base_arms_survive (my part) = (g) not dropped.
Expected (planner's table): no coherence gain before resolution (nothing trains the D-prove
questions before reveal), FP ~ 2-pass on BCE.
Time cap: 3 h wall for this pass; at most 4 concurrent CPU jobs from me.
