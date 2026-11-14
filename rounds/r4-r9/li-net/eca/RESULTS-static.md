## 2. What ran

- Question (Kelvin, 10-14, via the gemini chat): can one any-order network over homogeneous cells learn which of its own
  conditionals are stiff one-step laws (stiffness head beta_int), and can a search policy that routes through stiff arcs
  improve beliefs about long-range queries with compute, where the one-pass conditional fails? Secondary: do key
  fingerprints deduplicate redundant arcs?
- Data: noisy elementary CA, rules 110 and 90, eps = 0.015, W = 8, T = 6 (48 cells, row 0 iid), periodic. One fixed random
  permutation of the 48 cell ids; the network sees permuted token ids only, with learned position embeddings and no
  topology bias. Splits: train 50,000 grids, cal 5,000 (held out for the stiffness and key heads), val 3,000, test 3,000.
- Exact posteriors: forward-backward over rows (256 states), checked against brute force over all 2^16 grids at W = T = 4
  (max difference 3.8e-15). Used only for scoring, the D0 gate and the D1 score.
- Network: smallnet's transformer block unchanged (d 128, 4 layers, 4 heads); heads: Bernoulli logit per cell, stiffness
  (predicted held-out surprise, with the v2 floor), 16-d key (attention-weighted source fingerprint + h_i).
- Training arms, 3 seeds each: T1 plain masked loss; T2 reveal-chain TD with 8 levels (smallnet's rbtd, lam = 0). Reveal
  count k ~ U{1..47}. Early stopping on validation outcome log loss.
- Search: K = 16 chains per query; each chain materialises M cells one at a time (S_t = S_{t-1} + z_t, z_t sampled from
  q(z_t | S_{t-1})); selection P2-Stiff (sample by exp(beta_int)), P2-VoI (top-6 stiff cells, max expected squared change of
  q(Y)), P3-Conf (most peaked), P4-Rand (uniform). Claims q(Y | S_M) pooled by the prior-corrected learned-w pool (lw, scale
  fitted on validation outcomes), with Gram dedup (P5, dd), and by the P6 aggregators.
- Query families: F (row 0 known, query rows 3-5), B1 (rows 0 and 5 known, query rows 2-3), B2 (half of row 0, row 5,
  columns 0-1 known, query the interior), R (random reveal), S1 (one-step check: query's 3 parents known).
- Runs: 2 rules x 2 training arms x 3 seeds = 12 primary runs (the `eca-b` batch, 6 L4 VMs with 2 runs each, launched
  02:33Z after the float32 fix, Sec. 10), the same 12 as the original 01:35Z batch (now the replication, Sec. 9), 6
  rule-90 runs at T = 8 (secondary hedge, Sec. 9), and one priced 4-job smoke. Every training run stopped on patience
  (10 validation evaluations without improvement) after 17-76 minutes of training (primary runs); the 140-minute cap
  never bound.
- Test sets: P0 on 3,000 instances per family; chains, pools and D2 on the first 1,000 instances per family (same
  instances for every seed, selector and M, so seed-paired comparisons are on identical queries); D1 on 2,000 pairs.
- Compute: GCP on-demand L4s (g2-standard-8), budget group overnight-ca (cap $110). MATS cluster not used (full).
