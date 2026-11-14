# X1 lead PREREG: merge rule and seed-5 tie-breaks

Written 2026-09-28T13:26:13Z (mtime) (queue start 13:26:25Z, /tmp/traders-net/X1lead/queue.log), before any seed-5 lead run was launched and before any seed-5 lead
result was read. Worker PREREGs: `w1/PREREG.md` .. `w5/PREREG.md`. Plan: `../../STUDY-PLAN.md` section 6.

## 1. One status scheme for `CLAIMS.md`
Workers used different words. The lead register maps them to three statuses plus a `basis` column.
- **verified**: the number is in its file (static re-read) AND, where a fresh run exists, the fresh value lies
  within claimed mean +- 2 sd (sample sd of the committed 3 seeds) and the qualitative claim holds.
  Rows with no fresh run can be verified only with basis `static` or `deterministic-rerun` (byte-identical).
- **refuted**: the number is not in its file / is wrong, or the fresh seed(s) fall outside 2 sd on the same side
  (two fresh seeds when a tie-break ran) AND the qualitative claim fails, or two fresh seeds outside on the same
  side of a quantitative claim (w1's rule).
- **unverified**: not rerun, mixed fresh results, or outside 2 sd with the qualitative claim intact
  (w3 "partly verified" maps here, with the note "qualitative holds").
- Mapping of worker words: w3 "partly verified" -> unverified (unless the seed-5 tie-break below resolves it);
  w5 "verified from receipts" -> verified (basis static); w5 "plausible" -> unverified; w5 "partly refuted" ->
  refuted for the specific sub-claim; "refuted, now superseded" -> refuted with note.
- `basis` in {fresh-seed, fresh-segment, deterministic-rerun, static, analysis}.

## 2. Seed-5 tie-breaks (w1's pre-registered tie-break, applied to w3/w4 rows)
w1 ran seed 5 whenever seed 4 fell outside 2 sd; w3 and w4 did not. The lead runs seed 5 for every w3/w4
fresh-seed row that fell outside 2 sd:

| row | job (seed 5) | metric |
|---|---|---|
| R-M5b, R-M5c.P16, R-M5d.P16 | M5 w4 `probes.py --source preq --ckpt P16 --seed 5` on S0 w1 P16 seed 5 (X1 w1 bulk npz) | probe L phi, price-only phi, E_C b |
| R-M5c.P4 | `probes.py ... --ckpt P4 --seed 5` | price-only phi |
| R-M6b.A, R-M6b.B | M6 w1 `run_w1.py ME A 5`, `ME B 5` | (a) noise excess; (b)-(a) learn BCE |
| R-M6c.paired | M6 w2 `pool.py --arm cg --seed 5 --T 60000`; lead `pool_kc.py --arm boost --seed 5 --T 60000` | cg - BOOST noise excess |
| R-M6d | M6 w4 `fp_layer.py li fp 5`, `li pass2 5` | fp vs 2-pass D-coherence rel |
| M8-wholerun, M8-spearman | M8 w1 `sign_gate.py none 5` | prequential BCE; spearman income |
| M8-stop, M8-postshift-table | M8 w2 `gated.py absag 5`, `gated.py random 5` | post-shift switched BCE |
| M9c-thin | M9 w3 `run_kq.py A 1 128 5`, `A 64 2 5` | K64-K1 learn BCE ex-LFSR |
| K1-shift | K1 w2 `w2run.py S-on10k 5` | recovery steps; newborn share |

Code: the w3 clean copy (restored from trash into `/tmp/traders-net/X1lead/clean/`) and the w4 clean copy
(`/tmp/traders-net/X1lead/w4/study/`), with only /tmp output paths changed (X1w3/X1w4 -> X1lead). The seed-5
S0 prequential checkpoints are X1 w1's (`/tmp/traders-net/X1w1/bulk/P{4,16}-s5.npz`, same run_w1.py).
Derived metrics use the workers' own formulas (w3 `compare_w3.py`, w4 `check_w4.py`) with seed 4 -> 5.

**Resolution rule (fixed now):** seed 5 inside 2 sd -> the row stays unverified (mixed: one out, one in)
unless the seed-4 miss is explained by a named artefact (e.g. the pi-offset leak, seed 4 has a close pi pair
and seed 5 does not), in which case verified with that note. Seed 5 outside on the same side as seed 4 ->
refuted for the number if the claim is quantitative; if the qualitative claim still holds on both fresh seeds
the row is "unverified, qualitative holds, sd understated". A 3-seed sd is a poor estimate; we also report
the 5-seed mean +- sd.

## 3. Limits and stop rule
CPU only, nice 10, 1 thread, at most 3 concurrent (fleet cap 12), launch only while 1-min load <= 40.
Any job over 3x its original wall is stopped and its row stays unverified. Wall cap 2 h for the batch.
Bulk under `/tmp/traders-net/X1lead/`; only JSON/CSV copied to `X1/lead/results/`. No GPU.

## 4. Checks
`X1/check_claims.py` re-runs the worker checks (w1 make_claims, w2 check_claims, w3 check_w3, w4 check_w4,
w5 recompute_summary) and verifies that every evidence path in `CLAIMS.md` exists.

## 5. Clarification (appended 2026-09-28T13:29Z, before any seed-5 result was read)
Row kinds, fixed before reading seed 5, decide the "both fresh seeds outside, same side" case:
- quantitative (the claim is the quoted number +- sd) -> **refuted** (as w1): M5-1, M5-4, M6-3, M6-4, M8-wholerun,
  M8-postshift-table, M9c-thin, K1-shift.
- qualitative (the claim is a sign, tie or decision) -> **unverified, qualitative holds, sd understated** when
  the sign/decision holds on both fresh seeds, else **refuted**: M6-2 (tie), M6-9 (fixed point not kept),
  M8-spearman (income ranks weakly, |a.g| better), M8-stop (stop fires).
A row with several quantities takes the worst quantity. Implementation: `X1/lead/tiebreak.py`.
