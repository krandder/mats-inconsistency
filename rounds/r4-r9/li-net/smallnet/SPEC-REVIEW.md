# SPEC-REVIEW: ShipHunt small-network data-efficiency experiment

Reviewer: Opus 5.5 subagent for eurisko, 2026-10-10. Spec reviewed: `/home/agent/workspaces/mats/outbox/shiphunt-smallnet-spec-2026-10-10.md` (opus-mats, agreed with Kelvin 2026-10-10).
Superseding instructions: Kelvin (relayed by opus-mats, mail 3343 and the brief): run an autoresearch loop of at least 20 experiments, cap $200, finish in about a day, nothing that would not carry over to real questions, priority = data-efficient training (high compute per sample is welcome).

Verdict: the spec is sound and runnable. Nothing blocks it. Fifteen points below; the ones that change the experiment are marked **[changes design]**. Each has the fix adopted in the code (`smallnet.py`, `bninfer.py`, `netdef.py`, `data.py`).

## A. Correctness of the measurements

1. **[changes design] Early stopping must not use the exact posterior.** "Early stopping on a validation set" with the primary metric (KL to the exact posterior) would put the oracle into training decisions. That would not work on real questions. Fix: every run early-stops on the validation boards' per-cell log-loss against their TRUE cells (outcome-only). In expectation that equals KL to the posterior plus a constant, so it selects the same checkpoint up to noise. The KL is computed and logged but never used inside a run. The loop's choice between arms uses validation KL, because that is the research question; TEST stays untouched until the final champion.
2. **Exact inference for k = 2: "brute force at this size is fine" holds once, not at this volume.** Brute force is 2^25 = 33.5M states per (board, stage, order): about 45 ms on an L4, about 1 s on CPU. The sweep needs millions of these (validation every few hundred steps, several orders, 3000 + 5000 boards, hundreds of runs). Fix: exact variable elimination along the topological order, carrying the joint of the "frontier" cells that later cells still need. In simulation with nearest or random earlier parents the frontier peaks at 12 cells (median) and 16 (max), so it costs about 2 ms per net on one CPU core. Checked against brute force: identical to 1e-10 on 30 random 12-cell k = 2 nets and 30 k = 1 forests (`python3 bninfer.py`). k = 1 is a forest: marginals are exact by forward propagation, O(25).
3. **k = 2 Bayes-net marginals are biased even with perfect tables; measured, the bias is negligible.** A cell's marginal under the net needs the joint of its two parents, and a k = 2 net does not keep pairwise joints of cells that are not linked. So with exact local conditionals, k = 1 marginals are exact (forest) but k = 2 marginals are not. Measured on 150 validation boards x 4 stages with oracle tables (exact conditionals from the enumerated posterior): k = 1 bias 0 (as theory says), k = 2 bias 1e-5 nats per cell (max 3e-4) with nearest-earlier parents and 1e-5 with max-covariance parents (`/tmp/smallnet/diag_oracle.py`). Compare the prior-only KL of 0.017 / 0.033 / 0.050 at 2 / 4 / 6 probes. So the readout is fine as specified; noted, not changed.
4. **The single-cell readout of a Bayes-net arm depends on the random cell order.** One order gives one net; different orders give different marginals. Fix: average the net marginals over M = 4 fixed random orders (the same orders for every arm), harden parents to the top score per order, and also report M = 1 and the net's own single-cell head (`val_bn_M1`, `val_marghead` in every result).
5. **Stage 0 carries no evidence**, so every board has the same input there and every arm only has to learn the prior marginals. Averaging over stages dilutes the stages where evidence matters. Fix: report KL per stage (0, 2, 4, 6 probes) as well as the mean. The pre-registered rule uses the mean over the 4 stages, as the spec says.
6. **Decision rule needs exact definitions** (interpolation and CI). Fix in PREREG.md: log-linear interpolation of KL against log N between sweep points, a paired bootstrap over seeds, and 5 seeds (not 3) for the TEST comparison of the champion against k = 0 and k0cell.
7. **Validation boards are data too.** With N = 100 training boards, a 3000-board validation set is more data than the training set. Fix: the validation set is fixed and identical for every arm and every N, used only for early stopping (one scalar per checkpoint) and arm selection. The data-efficiency curves count training boards only, and this caveat is stated with the results.

## B. The rules (real-world validity)

8. **[changes design] The martingale penalty in the TD note uses the noise model** ("sample the next probes' readings ... including the 0.2 flip"). The noise model is evaluation-only. Fix: a label-free drift penalty computed only from the observed trajectory. For each stage and each forecast bin, the batch-mean change q(t+1) - q(t) should be 0, because exact posteriors have no predictable drift given the current forecast. Gradient flows to both stages; it is a regulariser added to the per-cell log score, never a target.
9. **"TD plus probe resampling" (TD note, item 2) breaks the no-resampling rule.** Not run. The allowed substitute is to reorder the probes the episode already observed. Probes are uniformly random cells, independent of the board, so any prefix of a reordering of the 6 observed probes is a valid evidence set: no new evidence, no simulator. That arm is tagged "real-world-valid: yes, if evidence items are exchangeable". This holds here and for independent reports, but not for time-ordered news. The strictly valid variant uses all 7 prefixes (0 to 6 probes) of the observed order.
10. **The reference net "trained directly on exact posteriors"** is tagged real-world-valid: no. It is evaluation-only, a capacity floor, and never a candidate champion.

## C. Arms and model

11. **Kelvin's two plain arms are "full map" and "cell by cell"** (his words to opus-mats, 2026-10-10 00:57Z and 01:00Z). The spec's k = 0 is the full map: 25 outputs in one pass. Added `k0cell`: one question per pass (a query flag on the asked cell, read only that cell), 25 passes per board-stage, the same network. The champion must beat both.
12. **Masked-cell and full-chain arms feed true cells into the network as inputs.** To keep "the network fixed across arms", every arm has the same backbone with a reveal channel per cell (hidden / revealed empty / revealed occupied) and a query flag. k = 0 never sets them. The k = 1 / 2 heads are factorized pair and triple heads on the shared cell representations: one backbone pass per board-stage, all candidate parents scored at once.
13. **Soft parent choice: an expected loss, not a mixture likelihood.** A mixture likelihood can learn to rely on blends of several parent sets, which hardening at evaluation would then break. The expected loss sum_S w_S * CE_S has its optimum at one-hot weights, which matches "harden to the top choice at eval". A small term beta * mean_S CE_S (beta = 0.1) keeps the tables of unchosen candidates trained. At position 1 only one earlier cell exists, so k = 2 uses the k = 1 table there.
14. **Candidate counts.** "24 singles / 276 pairs" is the maximum. In a random order, the cell at position t has t singles and C(t, 2) pairs among earlier cells.

## D. Budget and infrastructure

15. **The project-wide GPU cap is lower than the smallnet group cap.** `logical-soup/infra/lib.sh` has `BUDGET_USD=200` for the whole project, and $95.47 was already logged (GPU-LOG.md, `spend_usd`, checked 2026-10-10 16:00Z). gpu-up.sh refuses a VM if project spend plus reservation exceeds $200, so smallnet can spend at most about $104 before launches stop, even with `GPU_GROUP_BUDGET_USD=200`. The expected spend is far below that (see PREREG pricing), so this is a flag only; raising it would be eurisko's call.
   Also fixed: `smallnet/guard-smallnet.sh` added up audited cost only for DELETE lines dated 2026-10-0x, so from 10-10 on it would have counted $0 of finished smallnet VMs. The regex now matches any date. The guard was restarted at 15:52Z; the original is kept at `/tmp/smallnet/guard-smallnet.sh.orig`.

Minor: the spec's "$50 cap, no launch until Kelvin OKs" is superseded by Kelvin's later instruction ($200, run the loop). The spec's compute line ("spot L4 or CPU") holds, but the 16-core box is shared (load average about 20), so the sweep runs on spot L4s and CPU is used for smoke and tests only.

## Reference numbers measured during review (validation boards, `/tmp/smallnet/diag_oracle.py`, `smoke/`)

| quantity | value |
|---|---|
| placements (distinct occupied-cell sets, 7 cells each) | 11088 |
| prior-only single-cell KL at 0 / 2 / 4 / 6 probes | 0 / 0.0168 / 0.0332 / 0.0497 nats per cell |
| exact joint log-lik of the true board (mean over stages) | -8.89 nats per board-stage |
| best independent-cell joint log-lik (product of exact marginals) | -14.07 |
| k = 2 net with oracle tables: single-cell KL bias | 1e-5 nats per cell |
| exact elimination, k = 2, one CPU core | 2.3 ms per net |
