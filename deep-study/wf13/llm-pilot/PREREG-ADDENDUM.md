# Addendum to PREREG.md: a K-extension (registered before any test result was inspected)

Written 2026-11-10 ~17:42Z, while job 57877 was still generating the test rollouts (88 of 300 problems done). At this
time the only look at test-set output was a pipeline check on the first 48 records (did `analyze.py` run; the size of
the rounding difference for W1; one record printed). No accuracy, distance or movement number had been read.

Why: PREREG's H4 asks whether the importance-weighted aggregation returns to the target "and at what compute", but its
grid stops at K = 64. The sizing check of the tuned writers (`out/pilot_lora.json`, pilot seed) showed that the
helpful writer W4 writes almost one demonstration per problem (1.1 to 1.25 distinct in 32), so K = 64 may be too small
to say anything about the cost.

What: the first 32 test problems of each family (same generator sequence as the test set, so the same problems),
new independent rollouts (sampling seeds offset by 7,777,777): W1, W3, W3s, W3p, W4 at 256 rollouts each, and a
reference of 1,024 further rollouts of M (its first 256 serve as the second M pool for the noise floor).
K grid 1 ... 256, blocks as before. Script: `run_main.py config.json main ext`, `analyze.py OUT ext`.

Registered expectations (no outcome bin; the bins of PREREG.md are judged on the K = 64 run only):
- E1. K* := the smallest K on the grid with D(w,snis,K) <= D(W1,vote,16). Expected: K* <= 256 for W3 and W3s in both
  families; for W4 on `vars`, K* is NOT reached at 256 (it almost never writes M's own mistakes).
- E2. D(w,dedup,256) for every writer, with the found mass and the bound 1 - found mass / valid mass.
- E3. The plain vote of the biased writers does not approach the target as K grows (its distance stays flat).

Code changes for this (the analysis of the K = 64 run is unchanged): `run_main.py` takes an optional writer subset
and seed offset per set; `analyze.py` takes the set name and extends the K grid for `ext`; `config.json` gains the
`ext` set. New hashes are in `PREREG.sha256`.
