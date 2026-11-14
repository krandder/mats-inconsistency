# Guarded genome hashing on one L4

Q4096, fuel256, seed1, eight host threads, ten epochs per process.
K16: baseline4.115M vs guarded4.433M virtual instructions/s (1.0773x).
K18: pooled baseline3.038M vs guarded3.239M/s (three alternating pairs,
1.0659x; individual speedups1.0670/1.0865/1.0452x).

`summary.json` has raw process/CUDA timing and the separate phase profiles.
`comparison.json` pools only the main trials. `validation.json` independently
checks all ten CSVs and ledgers. `js-validation.json` records local exact JS
parity; `make-check.log` records the complete green isolated check.

The one VM and boot disk are deleted, verified13:09:38Z on2026-09-27:
849.478s lifetime, $0.20529 at the unverified $0.87/h ceiling. Operations,
empty-list receipts and cost-audit.json are retained. Full interpretation,
limits and device migration candidates: ../../docs/gpu-utilisation.md.
