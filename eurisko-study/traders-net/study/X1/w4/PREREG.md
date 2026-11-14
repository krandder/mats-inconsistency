# X1 worker 4 PREREG: skeptic + fresh-seed reproduction of M8, M9, K1

Written 2026-09-28 ~12:10Z, before any rerun. Claim: `../STATUS.md` (worker4: M8, M9, K1).

## Method
1. **File check.** Rerun each workstream's own check/analysis script in a clean copy under
   `/tmp/traders-net/X1w4/` (no writes to M8/M9/K1), and re-read each quoted headline number from its
   source file (`check_w4.py`, this dir).
2. **Fresh-seed rerun.** Seed 4 (never used by M8/M9/K1), committed scripts copied unchanged into
   `/tmp/traders-net/X1w4/study/...` (only output paths redirected; `sed` diffs recorded in `rerun.sh`).
   CPU only, at most 3 concurrent single-thread jobs, none launched while 1-min load > 40. GPU: none.
3. **Criterion (fixed now).** For a headline quantity with per-seed values x1..x3 (mean m, sample sd s),
   the seed-4 value x4 is *within 2 sd* iff |x4 − m| ≤ 2s. Paired per-seed differences are tested the same
   way on the difference. Where s < 1e-4 nats the test is also reported against a floor of s = 1e-3 and both
   are shown (a 3-seed sd can be accidentally tiny).
4. **Status per claim.**
   - `verified`: file check passes AND (fresh seed within 2 sd, or no fresh seed applicable and the
     analysis reproduces byte-identically).
   - `refuted`: a quoted number does not match its file, or the fresh seed lands outside 2 sd in the
     direction that reverses the claim's stated conclusion.
   - `unverified`: no check possible here, or the fresh seed is outside 2 sd without reversing the
     conclusion (reported with the value).
   Stop-rule claims are also checked against the workstream's own PREREG rule text.

## Arms (seed 4 each)
- **M8** (T=40000): `w1/sign_gate.py` none, pos1, neg1, shuf1; `w2/gated.py` absag, random.
  Headlines: nolfsr shuf1−pos1, none−pos1, neg1−pos1; census neg_share_top10 vs neg_share_all (none);
  Spearman income / |a·g|; w2 absag−random post-shift switched BCE and bce_all_nolfsr.
- **M9(c)** (`w3/run_kq.py`, 6000 steps): B K1 hw128, B K64 hw2, B K1 hw8, B K64 hw8, A K1 hw128, A K64 hw2.
  Headlines: learn_bce_ex_lfsr K64−K1 (B-FT flat; A-FT +0.0068), humility (noise excess) up at K=64.
- **M9(a)**: `w1/fit_lottery.py` rerun on the committed soup table (no fresh soup runs: out of scope, plan §5).
- **M9(b)**: file check only (wealth-net read-only); prequential FLOPs-to-Brier uses S0 P4 runs, whose
  seed-4 rerun is X1 worker 1's (S0) item, not redone here.
- **K1** (econ.py, seed 4): w1 10k Q64 on / off (T=10000); w2 S-on10k, S-off10k, C0-10k, C0.1-10k;
  w3 crp-within and bh-within at N=10000.
  Headlines: Q64-on Brier vs oracle, Gini vs null, alive; Q64-off top species share ≥ 0.997; shift
  recovery steps and newborn share; best regret at c=0; fund premium crp-within vs bh.

## Stop rule
Stop when every listed arm has run once (no retries except a crash, which is logged), or at 4 h wall.
