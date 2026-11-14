# P4: does Aumann-Shapley pay pick the latent worth a market?

Status: **No gain**. Aumann-Shapley (AS) tracks exact Shapley well on known circuits, but not as well as a cheap sampled Shapley. Markets on the latents that AS picked made extrapolation worse. The decision test is formally **invalid** because fine-tune seed noise was too large.

Numbers: means with 95% intervals. R1 intervals are circuit-clustered bootstraps (32 clusters, 123 net-levels); stage 3 intervals are paired t-intervals over (base, seed), n = 10. Everything is recomputed by `python analyze.py`, which writes `results/summary.json`, `results/rows_r1.json` (per net, level and rule), `results/rows_ft.json` (per base, arm and seed) and the two charts in `charts/`.

## Stage 0: exact program tables
- All theorem checks pass:
  - LOO gives copies 0.
  - An epistatic node has flip-LOO and flip-Shapley equal to 0.
  - H1's I(p;Y) = 0.
  - Shapley efficiency gap is 9e-16.
- On the 30 program tables (gates only), reach (r) agrees with Shapley at within-level rho 0.90 (top-1 0.88), but with MI-Shapley at only 0.33.
- H1 level 3: every p copy has reach 0 and MI 0, and flip-Shapley 0.106 each.

## Stage 1 (R1): fidelity on 41 SIIT nets
- **Nets:** all 41 accepted (IIA ≥ 0.99, strict IIA ≥ 0.997). The pilot passed 5 of 5.
- **Rank agreement with each net's exact flip-Shapley:**

| rule (cost in forward-equivalents) | rho vs exact flip-Shapley | top-1 |
|---|---|---|
| AS-SOFT m=50 (150) | 0.924 [0.908, 0.938] | 95% |
| AS-PAY m=50 (150) | 0.923 [0.909, 0.938] | 94% |
| AS-SOFT m=5 (15) | 0.918 [0.900, 0.935] | 94% |
| sampled FLIP-SHAP, matched cost (K=16) | 0.981 [0.976, 0.986] | 100% |
| sampled FLIP-SHAP, one order (K=1, ~11) | 0.964 [0.954, 0.973] | 100% |
| FLIP-LOO | 0.929 [0.900, 0.952] | 89% |
| GRADxACT | 0.37 [0.23, 0.48] | 50% |
| MI-1BIT / RANDOM | 0.13 / 0.21 | 45% / 22% |

- **S1 fails, on the equivalence clause only.** Mean rho (≥ 0.85, lower bound ≥ 0.80) passes, top-1 (≥ 80%) passes, and H1 passes. TOST vs matched-cost FLIP-SHAP fails: AS-SOFT is 0.058 [0.045, 0.071] lower, outside the ±0.05 margin.
- **Falsifier 1 is not triggered:** the gap is under 0.10 and top-1 is 95%.
- **Why AS falls short:** AS reaches its plateau by m=5–10 and more midpoints do not raise it. It converges to the diagonal (Aumann-Shapley) value, which differs from Shapley, not to Shapley itself.
- **Seed noise floor:** two SIIT seeds of the same circuit agree at rho 0.948 [0.934, 0.961], so AS sits just below this floor.
- **H1 level 3:** p ranks first under AS-SOFT, AS-PAY and exact flip-Shapley, with copies summed. It also ranks first under the joint intervention in all 3 seeds. FLIP-LOO misses in 2 of 3 seeds.
- **Completeness at m=50:** soft 0.03%, pay 0.6% (max 1.9%). Under the 2% bar.

## Noise check
- 5 of 6 nets were accepted and 3 were calibrated; the gate passes.
- AS-PAY tracks u more than r: +0.11 [0.08, 0.15], as expected.
- AS-SOFT does not track r more than u: −0.003 [−0.013, 0.013], which was not expected.

## Stage 2 (R2): 10 natural nets with 24 sign units vs 10 random-init nets
- **Alignment is above the null, but weak:**
  - best code kappa 0.42 vs 0.21 (difference [0.14, 0.27]);
  - only 1.3% of codes aligned and 2.6% of gates recovered.
- **Four-curve AUC:** AS-SOFT 0.953 vs FLIP-LOO 0.832. The difference is +0.12 [0.045, 0.198], so the expected AS ≥ LOO + 0.05 holds and falsifier 5 is not triggered.
- **Agreement with the flip-Shapley reference:** AS 0.983, LOO 0.83.
- **S2 cost holds here:** on all 10 nets, AS reaches its plateau at m=5 (15 forward-equivalents), which is 0.04–0.31 of what sampled Shapley needs for the same rho (median 0.16).

## Stage 3: market-resnet selection and fine-tunes
- **Base checkpoints reproduce:**
  - L64: A 0.866 (target 0.865), Bds 0.917 (target 0.916), on both CPU and GPU;
  - L128: 0.054 and 0.070.
- **Selection at t=16 (16 groups of 8 h channels, P=512, 448-order Shapley reference):**
  - AS vs the reference: rho 0.93 (A) and 0.91 (Bds).
  - FLIP-LOO scores 15 or 16 of the 16 groups at exactly 0 (redundant groups), so its "picks" are tie-breaks.
  - GRADxACT is saturated at about 0.
- **Picks used for the fine-tunes** (CPU selection, P=256):
  - A: [8, 7], the same as the Shapley reference.
  - Bds: [11, 7], the same as the reference top-2. GPU AS at P=512 would have picked [11, 14].
- **S2 on market-resnet fails in 3 of 4 cases.** Cost ratios: A t16 0.47, Bds t16 0.24, A t8 4.9, Bds t8 0.47. At t=8, AS completeness is still 21% off at m=200.
- **Fine-tunes:** 5 seeds per arm (triggered by F0 L128 SD > 3 points), 1000 iterations, 50 runs. Primary = mean of L64 and L128 accuracy. The table gives the mean with a t-interval over 5 seeds, clipped to [0, 1].

| arm | A | Bds |
|---|---|---|
| base checkpoint (no fine-tune) | 0.46 | 0.49 |
| F0 | 0.43 [0.29, 0.57] | 0.74 [0.34, 1.00] |
| F-AS-mkt | 0.40 [0.20, 0.60] | 0.08 [0.00, 0.30] |
| F-AS-aux | 0.28 [0.07, 0.50] | 0.52 [0.04, 1.00] |
| F-LOO-mkt | 0.62 [0.42, 0.81] | 0.15 [0.00, 0.35] |
| F-rand-mkt | 0.50 [0.00, 1.00] | 0.46 [0.12, 0.80] |

- **Paired differences (n = 10):**
  - F-AS-mkt − F0: −0.34 [−0.68, −0.00].
  - F-AS-mkt − F-rand-mkt: −0.24 [−0.53, +0.05].
  - **S3 fails.**
- **Triggered:**
  - Falsifier 3: every market arm is ≤ F0 + 2 points.
  - Falsifier 4 / read-out: F-AS-aux ≥ F-AS-mkt (+0.16 [−0.18, +0.50]), so the market framing adds nothing.
- **LOO read-out:** F-LOO-mkt ≥ F-AS-mkt − 2 points (+0.14 [0.02, 0.26]). But LOO's picks are tie-breaks among zero scores, so this reads as "AS picks were not better than arbitrary ones", not as "use LOO".
- **Collapse:** 7 of the 40 runs with a chosen group pushed its variance below 0.5× base: 3 of 10 F-AS-mkt, 4 of 10 F-AS-aux and 1 of 10 F-rand-mkt.
- **Invalid by the preregistered rule:** F0's seed SD at 5 seeds is 11 points (A) and 32 points (Bds) on primary, and 8 and 45 points on L128. Continued training at lr 3e-4 flips extrapolation between near-perfect and near-zero depending on the seed (Bds F0 per seed: 0.96, 0.24, 1.00, 0.89, 0.58). Every arm inherits this, so the S3 comparison cannot resolve 5 points.

## Deviations and simplifications
- **H1 SIIT:** trained with a per-node probe loss (0.2). Plain SIIT plateaued at IIA 0.68 on the 8-bit parity.
- **Base-checkpoint repro:** the VM was started before the Bds CPU repro finished. That repro later matched: 0.917.
- **Patching:** the market-resnet selection game patches h and keeps b.
- **Extra AS settings:** m=100 and 200 were added on the GPU.
- **Spot preemption:** trader-cat-p4-1 was preempted at 14:02Z. Its 9 unsynced seed-4 runs were redone on trader-cat-p4-2.

GPU: 1.64 VM-hours, estimated $1.16 at the on-demand ceiling (cap $8), recorded in `../../GPU-LEDGER.md`. At 14:50Z, `gcloud compute instances list` and `disks list` showed no trader-cat-p4 resource.

Files:
- code: `gen.py`, `stage0.py`, `crit.py`, `nets.py`, `run.py`, `ftmarket.py`, `vmplan.sh`, `gpu.sh`, `analyze.py`;
- results: `results/summary.json`;
- charts: `charts/p4_r1_rho.json`, `charts/p4_ft_primary.json`.
