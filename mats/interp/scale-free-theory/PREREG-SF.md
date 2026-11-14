# PREREG-SF: addendum measurements M3, M4, M5 (scale-free-theory, 2026-10-29)

Approved by the coordinator (about 2-3 h CPU, at most 3 processes, nice/ionice). Written before any M3/M4/M5 outcome
was computed. Seen beforehand: N5-N7 results (outbox/scale-free-theory-2026-10-29.md, 6.4) and plumbing checks only:
the re-implemented forward (`sf_net.py`) equals the model exactly; permuting the 15 context tokens changes outputs by
at most 1.4e-6 (the nets have no positional encoding, so routing can only use ID and content features); substituting a
net's own attention reproduces it exactly; a full-rank bottleneck reproduces it to 3e-6; the noisy forward used for M5
reproduces nnoise's forward exactly.

**Relation to phase B's registered interventions (PREREG-B.md, read 2026-10-29 ~20:30Z).** Not repeated: E8.2's
layer-2 query-row patches on the 12 noisy nets (Pbayes, PbayesQQ, Ptrue, Puniform, Pmean); E22's uniform within-source
attention at block 1; E24/E25's matched-pair transplants of a query-question token's attention row, values or terms.
M3 replaces whole-layer attention on the clean nets by a label-plus-weight fit or by an unrelated episode's pattern
(aligned or misaligned); M4 is a rank bottleneck on the residual stream; M5 is observational.

**Nets** (copied to `nets/`, hashes in `nets/MANIFEST.sha256`). Clean 2-layer tiny-PFN nets w32_h2_L2 s0-s4 (Gaussian
head) and w32_h2_L2_mix4 s0-s4 (mixture head). The learned set is these 10 minus w32_h2_L2_mix4_s3 (stuck, KL to Bayes
0.144; reported separately): 9 nets. M5 also uses w32_h2_L1 s0-s4 and the 12 noisy phase-B nets.

**Episodes.** Fit set: 2,048 episodes, seed 501. Test set: 4,000 episodes, seed 502. M3 partners: a fixed derangement
of the test set (seed 503); misalignment: one random permutation of the 15 context positions per episode (seed 504).
M4 training: fresh 512-episode batches (seeds 10,000 + step), 600 Adam steps, lr 3e-3. KL in nats on a 1,201-point grid
over [-6, 6], averaged over test episodes.

**M3** (`m3_subst.py`). For layer l in {1, 2}, the layer's attention (all rows, both heads) is replaced by:
fit (R_h(pair class) fitted per head on fit episodes, plus the test episode's own per-key effect w_j);
aligned (the partner episode's attention; same layout, so same labels); misaligned (the partner's attention with its
context positions permuted). K_V = mean KL(net || net with substitution V).
Routing cost rho_l = K_misaligned - K_aligned; weighting cost omega_l = K_aligned - K_fit.
- P-M3a (ladder): K_fit < K_aligned < K_misaligned for both layers in >= 7 of the 9 learned nets.
- P-M3b (layer roles, from N7): rho_1 > rho_2 in >= 7 of 9, and omega_2 > omega_1 in >= 7 of 9.
- P-M3c (the decomposition suffices): K_fit <= 0.25 x K_aligned for both layers in >= 7 of 9.
Refuted as stated if the counts fail.

**M4** (`m4_bottleneck.py`). The context tokens' residual stream after block 1 is replaced by a shared rank-k map
(k = 0 keeps only a learned constant), trained by distillation to the net. B_k = mean KL(net || bottlenecked).
- P-M4a: the smallest k with B_k <= 0.01 is <= 4 in >= 7 of 9 learned nets.
- P-M4b (sanity): B_0 >= 0.05 in all 9 learned nets.
Reported also: KL(Bayes || bottlenecked) against KL(Bayes || net).

**M5** (`m5_sink.py`). In the combining layer (block 2; block 1 for one-layer nets), per head, correlation r of
log a_self (query row's attention on the query token; noisy nets: mean over 8 noise draws) with log of Bayes's
posterior variance on test episodes, and the slope (identity: slope 1 if the query token acts as the prior sink).
- P-M5a: the best head has r >= 0.5 in >= 5 of the 9 learned clean 2-layer nets.
- P-M5b: the best layer-2 head has r >= 0.5 in >= 6 of the 12 noisy phase-B nets.
Exploratory: slopes, correlation with the net's own predictive variance, one-layer nets.

**GPT-6 check** of results and the addendum text before any claim is reported.
