# PREREG-R5-E2 amendments

Changes after `PREREG-R5-E2.md` was hashed (sha256 732514b5..., 2026-10-17T03:25:20Z), with UTC time and
reason. None may change a decision rule, threshold, scoring set or hyperparameter after a test number has been seen.

- (none: no rule, threshold, scoring set or hyperparameter was changed.)

## Operational log (UTC; no rule change)

- 2026-10-17T03:26Z GPU timing smoke r5e2-s999-smoke (seed 999, 300 steps, smoke-sized stages; parts d0, d0cal, d3b
  because the late oracle files did not exist yet): both architectures ran end to end, outputs discarded (moved to
  /tmp/r5e2/smoke-s999). Training 101 ms/step (primary) and 63 ms/step (fallback) with both on one L4. VM deleted at
  03:34Z, 0.12 h, $0.10.
- 2026-10-17T03:36Z seeds 101, 102, 103 launched (spot L4, GPU_RUN_TIMEOUT 120m) while the CPU generator was still
  writing the Q* calibration set, the D1 positions and the brute-force check (training needs only train/cal records).
  Each run's eval step waits for /root/soup/data/READY_ALL.
- Found at 03:50Z: the driver's launch ssh stays attached to the remote pipeline (sudo keeps the session), so drive.sh's
  in-loop push could not run mid-run; added gpu/push_late.sh to push the late files instead.
- 2026-10-17T04:00Z data generation finished: brute-force replay on 30 test positions, max abs error 6.7e-16 (P*_full),
  0 (P*_c); 500 D1 positions from 60,543 two-flank candidates.
- 2026-10-17T04:02:44Z seed 103's spot VM (soup-gpu-r5e2-s103-1017033621-9474) was preempted by GCE during training
  (0.42 h, $0.37, COST-AUDIT line written). Relaunched at 04:11Z as run r5e2-s103-b on STANDARD (stop rule iii, first
  relaunch). Nothing from the lost run is used.
- 2026-10-17T04:07-04:16Z CPU smoke of the remaining eval parts (d0b, d1, d2, d3a) on a 40-step CPU model (seed 998,
  /tmp/r5e2/cpusmoke, numbers meaningless): no errors. No code change resulted.
- 2026-10-17T04:22Z full data tree pushed to r5e2-s101, r5e2-s102 and r5e2-s103-b, READY_ALL set.

## Corrections from the independent audit (2026-10-17, AUDIT-R5-E2.md; no rule change)

- Data timing. PREREG section 3 says the data and oracle sets were generated before the hash. That holds for the train,
  calibration and test records and the two D0 oracle sets (03:15-03:17Z). The Q* test set (03:38Z), the Q* calibration
  set and the D1 positions (03:58Z) and the brute-force check (04:00Z) were finished after the hash (03:25:20Z), by
  the same generator run, started at about 03:04Z from code unchanged since 03:02Z. (The operational log above listed
  the late files but omitted the Q* test set.)
- Cone distance. PREREG section 4 defines the cone as the 3 hidden opponent slots nearest the root move's target square,
  at any distance; the spec's "within tactical range (distance <= 2)" limit was not applied and this was not listed in
  section 6. 65% of cone slots are more than 2 squares from the target. Exploratory D1 numbers only; the gate fails
  without any rollout ((3a)).
- D2 strata: "10 equal-count strata" were built from quantile cut points, which with tied entropies (seed 102) gave
  unequal strata; exactly equal-count strata give +0.180 [0.108, 0.252] instead of +0.187 [0.091, 0.283]. Exploratory;
  no outcome changes.
