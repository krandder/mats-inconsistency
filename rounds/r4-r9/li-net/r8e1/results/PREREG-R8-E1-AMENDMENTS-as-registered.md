# PREREG-R8-E1-AMENDMENTS: gemini's pre-spend amendments to readings 3, 6 and 7, folded in before launch

- Source: `/home/agent/workspaces/mats/outbox/experiments-round-8-amendments.md`. Its R8-E1 part (timestamp 15:58Z; the file's
  first 45 lines, sha256 d7b193a7e43f3519be7bbc47630278af153a11b08c065850eee014ade2405a4d) is quoted verbatim in
  `PREREG-R8-E1.md` section 2b. Relayed by opus-mats #3741 (15:58:46Z, urgent); eurisko acked in #3743. The file's later
  Part B (16:02Z) concerns R8-E2 only.
- Folded in and implemented before registration and before any spend; this file and `PREREG-R8-E1.md` were registered
  together at 2026-10-17T16:07:15Z (sha256 of this file in `PREREG-R8-E1-AMENDMENTS.sha256`).
- Approved as written: readings 1, 2, 4, 5, 8, 9 and 10 of my mail #3738.

## Implementation of each amended item (code: `li-net/r8e1/e1v5.py`)

1. **Reading 7, interleaved P-FullSupport stream** (`fs_stream`, used in `test8`'s E1-D1.1 block).
   - Per board, a random order of the 6 partition tiles (`tl6`), shared by U_NW and U_SE; tile slots = `tl6` then its first
     2 tiles again (`tiles8`). The amendment says "2 additional partition tiles" without naming them; I re-read the first 2
     of the board's order.
   - Per preference, 24 VOI-drawn 90/10 skewed clean reads (`sk`).
   - Slot 4b = tile slot b; slots 4b+1..4b+3 = `sk[:, 3b:3b+3]`. The first 24 reads are the 6 tiles plus `sk[:, :18]`, R6/R7's
     P-FullSupport up to order.
   - Test: `test_fs_stream_interleaves_one_tile_per_block`.
2. **Reading 3, anytime pool training** (`fit8`):
   1. Target: ref(M_0) = sigmoid(union call with root M_0 on the same 128 reference patches as the standard reference); M_0 = 2
      uses the standard reference. Loss: BCE to ref(M_0) (`REFS`), not occupancy.
   2. K support: `KS_TRAIN = (0, 1, 2, 4, 8, 16, 24, 32)`, `pool_any = 32` reads per half per board per root strength, bags of
      up to 32 reads (`K_tr = 32`).
   3. Precision-weighted crowding with the anchor token: `crowding_prec`, used by `AttnL3P` and `LoopedAttnT3P` (the two
      anytime pools): c_r = sum_{b=0..K} v_b (beta_b / beta_clean) relu(<k_r,k_b>)^2. The anchor's relative weight is
      beta_0 / beta_clean = M_0 / 6 (3 at M_0 = 18). Test: `test_precision_weighted_crowding_and_anchor_identity`.
   4. opus-mats' summary also asks to "pass per-cell root certainty |l0_c| (or the anchor weight) to the pool". I implement
      the anchor-weight branch: beta_0 enters each read's crowding and the anchor token's own log-beta input. Per-cell |l0_c|
      is not added; it is not in gemini's file.
3. **Reading 6, strong-root read noise** (`test8`'s strong-root block):
   - Primary, and the one E1-D1.2 is decided on: tau ~ `RO.draw_tau`, 50/50 over 0.3 / 1.5 (R7-E1's mixed search
     distribution), at M_0 = 6, 12, 18 (tags `sr6`, `sr12`, `sr18`).
   - Reported, not decisive: the three-tier version (tau uniform over 0.3 / 1.5 / 2.5) at M_0 = 18 (tag `sr18t3`). The
     amendment allows either one with the anchor-precision crowding.
   - The anytime pools are trained across all three tiers (the calibration pools draw tau uniform over the 3 tiers).

## Cost and ETA

Unchanged in substance: 32 instead of 24 reads per half adds about 96k calibration reads per seed (about 1 minute). The run
budget (remote 28 min, timeout 38) and the expected cost (about $0.30-0.35) stay as mailed; the ceiling stays $0.80.
