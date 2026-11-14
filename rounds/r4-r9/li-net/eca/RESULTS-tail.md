## 10. Deviations from the design (full log with times: PREREG.md Sec. 9-12)

- **Design v2 applied in full before any training run** (00:59-01:08Z): sequential materialisation chains, eps 0.015, reveal
  count k ~ U{1..47}, families F/B1/B2/R, attention-source key head, stiffness floor, rule 30 dropped, decision rules D0-D3.
- **Key-head training bags without topology.** v2 asks for "same-blanket" redundant views; knowing the blanket would hand
  the CA topology to training, so redundant views share a random core set instead. Topology enters only the D3a evaluation.
- **cal split.** 5,000 extra training-distribution grids, held out from the main loss, fit the stiffness and key heads, so
  the stiffness head learns held-out surprise.
- **P1 folded into P4-Rand + mix.** v2 replaced one-cell contraction with materialising chains; random-order chains averaged
  by the plain mixture are the P1 arm (SNIS weights are 1 because proposal and target are the same network chain rule).
- **K = 16 chains**, fixed at 01:15Z from a first-principles Monte Carlo noise argument, before any trained model existed.
- **Rule-90 T = 8 hedge** (secondary, Sec. 9 here), launched 01:36Z because the smoke's rule-90 models were already close to
  exact at T = 6.
- **Float32 evaluation bug and the eca-b rerun** (PREREG Sec. 12). The 01:35Z code pooled claims in float32, so confident
  pooled probabilities rounded to exactly 1.0 (infinite KL, NaN log loss). Fixed at 02:31Z (float64 everywhere, training
  untouched); all 12 main runs were rerun with the identical configuration as eca-b (02:33Z). The primary-source rule was
  declared at 02:35Z, after reading one finished run (r90-T1-s2) and before any other result. All 12 eca-b runs came home,
  so every primary number below is from eca-b; the original runs are the replication (Sec. 9).
- **MATS cluster not used** (opus-mats, 01:0xZ: 8/8 L40s allocated until 11-19). Zero jobs submitted.
- **Infrastructure only:** on-demand L4s were stocked out in all three us-central1 zones at 01:36-01:39Z; launches retried
  through an eca-local copy of gpu-up/gpu-run with more zones (`gpu/gpu-up-zones.sh`, `gpu/gpu-run-zones.sh`). Every VM
  still landed in us-central1. Seven original-batch and T = 8 rentals lost the `END` line in their launch logs because
  `gpu/launch.sh` was edited while they ran; their VMs were deleted normally (GPU-LOG.md DELETE lines).
- **Training length.** Every run stopped on patience (10 evaluations without a better validation outcome log loss); the
  140-minute cap never bound.
- **D1 pair construction.** S is drawn with k ~ U{1..47} and the target is then removed, so about half the pairs have
  k - 1 reveals (one pair has an empty S). Validation ran every 1,000 steps (PREREG Sec. 10), not every 500 (Sec. 3).
- **Brute-force check.** The auditor's rerun gives max |difference| 5.3e-14 (this report first quoted 3.8e-15 from the
  01:00Z run); both are float64 round-off.
- **Stretch goal not run.** Candidate 2 (modular-arithmetic constraint nets) needs a categorical model and new exact
  inference; not attempted tonight.
