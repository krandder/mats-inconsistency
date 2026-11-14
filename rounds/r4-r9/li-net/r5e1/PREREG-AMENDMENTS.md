# PREREG-AMENDMENTS R5-E1: amendment A1, bounded-context patch rollouts (Part A v2)

- Written 2026-10-17T03:3xZ (exact time = the `Registered:` line at the end; sha256 in `PREREG-AMENDMENTS.sha256`), before any
  launch of the amended variant (run id `r5e1-v2`) and therefore before any amended score exists.
- Source: gemini's amendment, `/home/agent/workspaces/mats/outbox/experiments-round-5-amendments.md` (written 2026-10-17T03:15Z,
  **sha256 d04269d4b0be4bb889e5daef6ddf0093ce4451afa572800d41f05ce1941f355f**), relayed by opus-mats (mail #3693, 03:15:43Z;
  #3696, 03:27:44Z) and by eurisko (03:2xZ). It approves every operationalisation in `PREREG.md` and asks for one structural
  change to Part A: each rollout becomes a bounded-context call that conditions theta_real only on the sensor readings inside
  its patch, while the 0-step root pass sees only the commander's note (or 2 anchor probes). opus-mats: run it as the
  **primary** pre-registered Part A variant; keep the original Part A and the gap ladder as secondary/descriptive; Part B
  unchanged; report both. A rollout in v2 is a local evidence read, not a hypothetical sample from the model.
- Timing: the amendment mail reached me at 03:27Z, after the registered-design main run had launched at 03:25Z (VM
  `soup-gpu-r5e1-main-1017032529-20808`). That run continues unchanged (`PREREG.md`).

## 1. What is primary after this amendment

| rule | source run | design |
|---|---|---|
| E1-D0.1 clause 1 (world model) | `r5e1-v2` | theta_real v2's one-pass belief on the standard root format (6 readings + note) vs exact Bayes on the same evidence; the Part-B model's value from `r5e1-main` is reported next to it |
| E1-D0.1 clause 2, E1-D1.1, E1-D1.2, E1-D2.1, E1-D2.2 | `r5e1-v2` | Part A v2 (section 2) |
| E1-D0.2, E1-D3.1 to E1-D3.4 | `r5e1-main` | Part B exactly as registered |
| (secondary, descriptive) | `r5e1-main` | the registered Part A (hypothetical-sample rollouts) and its gap ladder |

The rule text is unchanged (spec 1.3 verbatim, `PREREG.md` section 8). Gates: E1-D0.1 (v2) gates E1-D1/D2 (v2); E1-D0.2 gates E1-D3.

## 2. Part A v2 (`partA2.py`; theta_real, networks, aggregators and bag protocol otherwise as `PREREG.md`)

- **Board evidence.** The same ship prior (49,504 patterns) and commander's note (0.30 / 0.70). Added: a dense sonar field, one
  reading per cell, s_c = occ_c XOR Bern(0.15). E_board = all 36 readings + the note. Per board, 2 anchor cells (uniform,
  distinct) and, for clause 1, 6 standard cells (uniform, distinct). New fixed streams: calibration 21, test 22, validation 23,
  monitor 24 (500 test boards, 4,000 calibration boards).
- **theta_real v2.** Same WorldModel (669,633 parameters), optimiser, schedule, batch 512 and **40,000 steps**, trained only on
  real boards: the evidence shown is a random subset of the board's readings (20% 2 random readings; 25% 2 + one patch; 15%
  2 + the union of 2-8 patches; 10% all 36; 15% a uniform subset of size U{0..36}; 15% 6 random readings), plus the registered
  any-order occupancy reveals on 30% of examples. It never sees a corrupted reading. Final checkpoint, no selection.
- **Root (0-step) belief** l0 = theta_real(note + the 2 anchor readings).
- **Rollout r = read of patch A_r**: l_r = theta_real(note + anchor readings + the readings of A_r's 6 cells), Delta l_r = l_r - l0.
  tau_r = 0.3 (w.p. 1/2): clean read, the stored readings (repeated reads of a patch are identical). tau_r = 1.5 (w.p. 1/2):
  corrupted read, each patch reading re-flipped w.p. 0.127, fresh per read (effective flip 0.239, the flip rate whose
  log-likelihood ratio is the sensor's divided by 1.5); anchor readings are never corrupted. theta_real is not told the kind of
  read; theta_trust gets log tau as before.
- **Keys** from stop_grad final hidden states on the 6 read cells, attention-pooled (last layer, head mean, 37 query tokens).
- **theta_trust features** [hbar_r, mean entropy H(q_r), log tau_r, reading-level leave-one-out swing = mean over the 6 patch cells
  of |l_r(c) - l_r(c) with c's own reading removed|]; target = real excess surprise on the patch cells vs the realised board.
  7 theta_real passes per read.
- **K = 128 reference** (aggregator target, never an oracle) = theta_real(note + anchors + the clean readings on the union of the
  patches of 128 unskewed reads, 64 NW-half + 64 SE-half), i.e. the model's own belief from all the evidence an unskewed K = 128
  search inspects (coverage logged).
- **Search policy, bags, aggregators, controlled bags**: exactly as `PREREG.md` sections 3-4 (VOI from the root belief). In the
  controlled bags the 16 tau = 0.3 cluster reads are identical clean reads (dedup regime) and the 16 tau = 1.5 reads carry
  independent corruption (mutual-support regime).
- **Evaluation target** P* = exact Bayes P*(Z | E_board) (all 36 readings + note), test split only.

## 3. Computation changes (only these; everything else as `PREREG.md` section 8)

- **E1-D0.1** PASS iff mean clause-1 Pearson (theta_real v2 one-pass belief on 6 standard readings + note vs exact Bayes on the
  same evidence, 500 x 36 test cells) >= 0.65 AND mean 1 - KL(P* || mix of the unskewed K = 24 read bag) / KL(P* || mu0) >= 0.15,
  mu0 = the root belief. (The root belief's correlation with full-board Bayes is reported, not used: it measures missing
  evidence, not the world model.)
- E1-D1.1, E1-D1.2, E1-D2.1, E1-D2.2: as registered, on the v2 bags, with P* = P*(Z | E_board).
- Added descriptive comparators (no rule): the "union call" (theta_real given the clean readings on the union of a bag's patches,
  one call); theta_real with all 36 readings in one call; an unskewed bag of clean reads only.

## 4. Runs, cost, code

- One L4 (`r5e1-v2`, spot first; STANDARD after a preemption), seeds 101, 102, 103 concurrently, through gpu-run.sh with
  GPU_BUDGET_GROUP=r5 GPU_GROUP_BUDGET_USD=22, remote 80 min (timeout 90 min). Expected about $0.9; R5-E1 total about $2.5 of
  its $6 ceiling (`ledger.py --check-launch` before launch).
- Code at registration of this amendment: partA2.py 474c7bbd4cc8eee16ac6fede65639f375bcc17701c2cea0ddd76af18bf2845d0;
  remote.sh 91513560212dc94af535eb2920ed776df877d9e044819c574968daf5ce5af0d0 (only change: tags starting `v2` run partA2.py);
  test_r5e1.py 8818b15c8f15acac18a0c50d409e1a68ffcf0fb47c22bd60e6c3959a6f6a7852 (two tests appended, the registered 8 unchanged;
  10 passed at 03:35Z); run.py 0a6a897a... (unchanged since the main launch), env6.py, wm.py, rollouts.py unchanged since
  registration.
- Analysis: `analyze.py` gets a v2 mode with the clause-1 change above, written before any v2 result exists.

Registered: 2026-10-17T03:36:58Z
