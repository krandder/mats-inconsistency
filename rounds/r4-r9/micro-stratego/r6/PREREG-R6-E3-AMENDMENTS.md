# PREREG-R6-E3 amendments

Changes after `PREREG-R6-E3.md` was hashed (sha256 5b45ec96..., 2026-10-17T06:34:20Z), with UTC time and reason.
None may change a rule, threshold, scoring set or hyperparameter after a test number has been seen.

- (none: no rule, threshold, scoring set or hyperparameter was changed.)

## Operational log (UTC; no rule change)

- 2026-10-17T06:35:51Z launched run r6e3-main (one on-demand L4, all three seeds, GPU_RUN_TIMEOUT 75m). No
  outbox/experiments-round-6-amendments*.md existed (checked 06:34Z and 06:35Z).
- 06:40:42Z the driver's launch command did not start the pipeline: its output redirect into /root ran as the
  unprivileged ssh user (permission denied). Found at 06:43Z (GPU 0%, no logs); started by hand at 06:44:14Z with the
  redirect inside sudo. The corrected driver is `r6/gpu/drive_v2.sh`. `r6/gpu/drive.sh` was edited in place at
  06:44:50Z while the driver process was still reading it, then restored by reversing the one-line edit; no copy of the
  launched bytes exists, so byte identity cannot be verified (audit); the driver finished cleanly. No code, data or setting of the experiment changed.
- Audit (AUDIT-R6-E3.md) disclosures added after the run: the pre-hash smoke also showed a key gap of 0.054 and
  trust_pool 29% worse than stiff_pool on 12 positions (no code changed after it); report_e3.py was written after the
  hash and one table label changed at 06:37:17Z; the D1 key gap is averaged per position over the two policies (the
  prereg text's order gives 0.2274 instead of 0.2276, same verdict).
