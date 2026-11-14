# M6 worker 2 PREREG: convex pools (c) k learned, (d) k from prequential wealth

Written 2026-09-28 06:40Z before any full run (only a 1200-step smoke in /tmp). Code: `pool.py`.

- **Substrate.** S0 `streams_ext.make_streams(Q=64, T=60000, seed=s, heldout=8, release=T//2,
  regime_every=0)` (static, no shift), single-pass prequential (`prequential.run_prequential`, B=4),
  torch 1 thread. Market price l0 = S0 per-question `Logistic`, Adam 1e-2, trained as the run's
  baseline, fed DETACHED into the layer. Trunk = S0 `Net` d=64 L=2 H=4; 16 trader heads per
  question read the final-LN residual and l0. Adam 1e-3 (pool weights theta: 1e-2).
- **Arms.** `a` plain logit sum (reference only; w1 owns (a)); `cg` convex pool, k=softmax(theta),
  one k per trader; `cq` same with k per (question, trader); `d0` k = softmax(prequential log-wealth)
  per question, wealth paid by the locked currency `M1/currency.committed_settlement(p_pool, p_i, y)`
  from the trade-time TEST pass (Bayes/Kelly mixture, alpha=0); `dfs` same + fixed share alpha=0.01.
  Trader belief p_i = sigma(l0 + d_i); pool p = sum_i k_i p_i; loss BCE(p, y).
- **Seeds.** 1, 2, 3 (stream and init paired across arms). Report mean ± sample sd.
- **Metrics.** Primary: learnable-family BCE = mean of per-family prequential BCE over
  {periodic, lfsr16, ca110, prime, and, or, periodic_ho}, all positions. Secondary: noise excess
  max over {ca30, prng} of (BCE − ln 2); late-half (positions ≥ n/2) versions; market l0 alone;
  pool concentration (effective number of traders exp(H(k)), wealth Gini, top share).
  LFSR16 is bimodal (S0): per-seed values reported.
- **Stop rule (plan §3).** An arm fails if its learnable BCE ≥ 1.01 × arm `a`'s in 3/3 seeds.
  If every pool arm across M6 (c)-(g) fails, drop the pool (decided at the M6 merge; my part
  reports (c),(d)). base_arms_survive (my part) = at least one of cg/cq/d0/dfs does not fail.
  Also reported, not a stop rule: beats l0 alone and beats `a` by ≥1% in 3/3 seeds.
- **Not in this pass.** Stress (multi-epoch, independent-evidence family, noise traders, D
  families) waits for Gate B.
