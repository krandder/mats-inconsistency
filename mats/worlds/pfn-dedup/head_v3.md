# Does attention deduplicate? Set-style PFN, depth 1–8 (opus-mats worker, 2026-10-27)

Kelvin (10-26): "I somehow expected attention to be responsible for much of the deduplication; this will show."

## Answer (v3: all 8 arms trained 40k steps; revised after the adversarial review, which is at the end)

- **Attention does all of the deduplication there is, and the architecture forces that.** MLPs act per token and
  there is no positional encoding, so only attention can see that two rows are the same. The testable question is
  which attention. The answer depends on whether the partition is visible (keys).
- **At depth 1, readout attention alone gets a large part of the discount.** With one copy (m = 2, the only copy
  count seen in training), the pull toward the duplicated value is 0.049 without keys and 0.040 with keys. For
  reference: true duplicates 0, count-all 0.121, an extra independent reading 0.176. At depth 1 a copy's attention
  score equals the original's, so the discount comes from softmax normalisation. The trained query puts more mass on
  the duplicated source's whole row, so copies take mass from each other: copy-mass ratio 1.50, against 1.72
  untrained.
- **Without keys, depth adds nothing on duplicates, and deeper readouts never lower the copies' scores.**
  - The pull at m = 2 is 0.046–0.048 at depths 2, 4 and 8, against 0.049 at depth 1. The duplicate MSE excess is
    0.56–0.64 of count-all's at depths 2–8, against 0.45 at depth 1.
  - The readout mass on the copies is what identical scores give, within 0–6% at depths 2–4 and within 10% at depth 8.
  - Deeper models still use context-mixed state. Patching after layer 0 at m = 2 raises the pull from 0.046–0.048 to
    0.06–0.07, whether the copies' state or the *rest* of the context is replaced. Replacing both gives 0.08–0.09; a
    random patch of the same norm does nothing.
  - That state is spread over the whole context, not held in the copies, and its net effect leaves the pull no lower
    than at depth 1.
- **With keys (group identity visible), depth helps and the readout scores do target copies.**
  - The pull at m = 2 falls from 0.040 at depth 1 to 0.026–0.031 at depths 2–8. The duplicate excess falls from 0.44
    to 0.18–0.27 of count-all's.
  - At layers ≥ 1 the scores remove 20–77% of the copies' extra mass: depth 2 0.41, depth 4 0.35–0.58, depth 8 up to
    0.75–0.77 in layers 2–3.
  - The copies' own state matters more than the rest of the context: at m = 2, patching the copies gives a pull of
    0.067–0.076, the rest 0.043–0.056.
  - On shared-error episodes, per-source readout attention tracks the GLS weights: corr 0.86 at depth 2 (last layer)
    and 0.78 at layer 2 of depth 4 (0.59 at its last layer), against ≈ 0 untrained.
- **No model deduplicates fully.** Every trained model keeps a pull of 0.026–0.049 at m = 2. That is 21–40% of
  count-all's pull, or about 25–50% of the reviewer's no-dedup baseline measured on the models themselves (≈ 0.10).
  The prereg duplicate level (L1, ≥ 75% of the NLL gap closed) is reached by no arm; the best is 0.72, at depths 4 and
  8 with keys.
- **Shared errors are handled at every depth, depth 1 included.** Depth 1 closes 88% (no keys) and 95% (keys) of the
  gap between iid-distinct and Bayes on shared-error episodes, and the MSE to the Bayes mean agrees. Predictions 1
  and 3 failed.

## Status, caveats (read first)
- All 8 arms (depth 1/2/4/8 × keys 0/1, 3 seeds each) are trained to 40k steps; 15/24 converged by the prereg rule.
  - Depths 1 and 2 come from pfndedup-1026c.
  - Depth 8 without keys comes from pfndedup-1027c.
  - Depth 4 (both arms) and depth 8 with keys come from the coordinator's on-demand relaunch, pfndedup-1027e/f/g. It
    replaces the 1026c depth-4 models, which had timed out at 10k–35k steps. Those old models stay in runs_1026c/ and
    are not reported.
- The prereg L1 test (NLL, duplicates vs count-all) is weak: count-all's own gap there is only 0.074 nats, and the
  model NLL mixes deduplication with variance calibration. The mean-based duplicate excess and the copy sweep below
  measure deduplication directly.
- The copy sweep goes to m = 8, but training never had more than 2 rows per source (k ≤ 3 *distinct* sources
  copied once). m ≥ 3 is out of distribution; m = 2 is the in-distribution number.
- Copy-sweep, attention and patching results are 3 seeds per arm. Probes and ablation use 3 seeds at depths 1–4;
  at depth 8 the ablation is seed 0 only, and depth 8 is missing from the probe table. Every probe is retrained per
  layer (ridge, episode-level split, inner-CV λ), with untrained (same init seed) and shuffled-target baselines.
- **Cost: about $10.9, against a $6 cap.**
  - My runs 1026/1026b/1026c: $3.71.
  - 1027a–d: 3.6 VM-hours of spot, logged at the on-demand ceiling of $0.87/h, so $3.1. Real spot billing is lower;
    I could not check it because the Billing API is disabled.
  - The coordinator's 1027e/f/g: $4.04.
  - All VMs are deleted: `gcloud compute instances list --filter=name~pfndedup` returned nothing at 06:00Z.
- The adversarial reviewer saw only the timed-out depth-4 models and the 6k-step depth-8 models. The converged models
  agree with the revised headlines without keys. With keys they strengthen the "readout scores target copies" claim
  and change "depth does not help duplicates" to "depth helps with keys only".
