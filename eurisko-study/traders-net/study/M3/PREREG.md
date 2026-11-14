# M3 PREREG (reduced, worker 1, 2026-09-28)

Scope (RETARGET 00:25Z): one short note on currency and lock-in, built on wealth-net's decay and
settlement results. No training, no GPU. Read-only on ../../../wealth-net/results.

- Arms (currencies, MLP neurons, 512/seed): all-time gradient wealth; decayed wealth half-lives
  100/300/1k/3k/10k; income over last 100/500/2k steps (all from wealth-net decay.py);
  exact output settlement (LOO, held-out and training; wealth-net settle.py);
  NEW here: windowed gradient-norm income (gradnorm_cum difference over last 100/500/2000 steps,
  and all-time) from the same snapshots; disjoint-window income rank stability.
- Metric: Spearman vs single-neuron zero-ablation oracle at the final checkpoint
  (wealth-net results/decay_oracle_s*.npy); pruning dBCE from decay_ablation_s*.csv.
- Seeds: 1, 2, 3 (wealth-net train-s1..s3); report mean +- sd (ddof=1).
- Control: shuffled oracle (1000 permutations) -> null sd of Spearman.
- Stop rule (plan section 3, M3): if no decayed/windowed currency beats the gradient-norm control
  by >= 0.05 in 3/3 seeds, gate by gradient magnitude and treat wealth as accounting only.
  (Recent |a.g| itself is not in wealth-net snapshots; windowed gradient norm is the proxy.)
- Not run: regime-shift lag (no shift run exists on wealth-net; M1 did not produce one).
