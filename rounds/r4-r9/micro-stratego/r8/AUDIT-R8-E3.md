# AUDIT-R8-E3: independent audit of R8-E3 (E-MS-SAFE-ANCHOR, 6x6 Stratego)

Auditor: a read-only board investigator subagent running on Opus 5.5 (claude-opus-5-5[1m]), separate from the executor.
Audit run 2026-10-17 16:39Z-17:14Z UTC. RESULTS line numbers below refer to the audited version (16:39:03Z). This is the path the executor waits for; RESULTS-R8-E3.md and PREREG section 7
point to `r8/AUDIT-R8-E3.md`, so the executor should copy this file there. Paths are relative to
`/home/agent/workspaces/mats/work/micro-stratego/` unless absolute.

Files audited (sha256 prefix and mtime, read at 17:00:41Z): `r8/RESULTS-R8-E3.md` 30780cb41fce1eb8 (16:39:03Z);
`r8/PREREG-R8-E3.md` dc5e5681134009a2 (15:58:20Z); `r8/PREREG-R8-E3-AMENDMENTS.md` d6a4b2d8c930b55e (16:25:57Z);
`r8/report/verdicts.csv` b6b206891ba56168 (16:12:26Z); `r8/report/expl_es.json` 6c01eb1d9703143e (16:37:22Z);
`r8/GPU-LOG-R8-E3.md` 4b7b77507d6d04af (16:36:59Z); spec `/home/agent/workspaces/mats/outbox/experiments-round-8-spec.md`
b19a33e098a3309e (the hash the prereg cites); amendments `/home/agent/workspaces/mats/outbox/experiments-round-8-amendments.md`
e4b0ab7ceb4e9764 (16:01:43Z). Nothing in r5/-r8/, /tmp/r5e2 or /tmp/r8e3 was edited; no git, no GPU, no gcloud
create/delete, no messages, 0 GitHub calls.

## Verdict

**Supported with corrections.** With my own code I re-derived every pre-registered statistic from the per-item CSVs and
got the same 26 criterion outcomes and the same verdicts as `r8/report/verdicts.csv` and RESULTS-R8-E3.md: E3-D1 FAILS
(part (1) passes, part (2) fails for both attention aggregators) and E3-D2 FAILS (parts (1), (2a) and (2b) each fail).
CSV rounding moves no rule statistic by more than 6e-5, no criterion is near its bar, and dropping the test positions
the pre-hash smoke saw changes no outcome. An independent CPU recomputation from the seed-101 checkpoint, with my own
forward pass, tapes, window likelihood and rollout code, reproduces the K = 0 anchor KL (0.048975 vs 0.048975 on all
500 D1 positions), LL_window for given cone types (to 6e-6), and all 15 tier statistics of the clean and noisy rollouts
within sampling error on all 2,361 test pairs; with a shared random stream it reproduces the executor's rollouts draw
for draw. No oracle quantity enters any fit or selection, and I agree that fitting the D1 aggregators to the true
hidden types of calibration positions is not oracle leakage under the spec's protocol. The hash, the timeline and the
verbatim rules check out, and nothing in the design changed after the disclosed smoke. The post hoc early-stopping
rerun is labelled, uses only calibration data, reuses the registered test bags exactly and changes no verdict. The
corrections are one wrong provenance claim (the amendments file was rewritten at 16:01:43Z and gained R8-E2 amendments,
not "R8-E1 only, last rewritten 15:58:46Z"), prose that claims more than the numbers show (window consistency, "any
small step is safe", two untested mechanisms), a cost figure that leaves out deletion time ($0.46 vs $0.49) and some
clarity fixes. None changes a verdict.

## Checks A-I

| check | result | evidence (check run, UTC) |
|---|---|---|
| A. Pre-registration | PASS (correction C1) | `sha256sum r8/PREREG-R8-E3.md` = dc5e5681...29a1b02 = `PREREG-R8-E3.sha256` (17:00:57Z). Hash 15:58:30Z < `r8/gpu/launches.log` first line 15:58:44Z < first PENDING-CREATE (GPU-LOG.md, minute stamp 15:58Z; VM name stamp 1017155844 = 15:58:44Z; gcloud insert operation started 15:58:53.5Z). Prereg lines 20-30 = spec lines 123-133 byte for byte (`cmp`, 2,812 bytes, 17:00:57Z). Smoke disclosed; no design change after it (details A). Amendments log misstates the outbox file's last rewrite (C1); nothing in that file is for R8-E3. |
| B. No oracle in fitting or selection | PASS | Oracle files are read only at `r8/run_r8e3.py:141-144` (part_d1) and `:280-286` (part_d2); r8lib, r7lib, e3lib and expl_d1_es read no file (grep, 16:48Z). Fits and selection use the calibration split only: true hidden types (D1 target), realised results (trust head, hyperparameters). I agree the true-type target is not leakage (reasons under B). Nothing else oracle-derived enters. |
| C. Headline numbers from per-item CSVs | PASS | `aux/aud_cd.py` (own code, 16:43:46Z): 288 D1 KL/JSD means, max abs diff 3.5e-7; 13 pools x 2 bag types rebuilt from d2_rollouts.csv: per-pair Q_hat vs d2_actions.csv within 4.3e-6, MSE within 1e-7 relative, Bias^2 within 3.6e-9; Spearmans within 1.8e-5 (rule statistics), 4.5e-5 (descriptive R5-E2 head). Rounding moves nothing that matters. |
| D. Verdicts recomputed | PASS | Own seed-t (t = 4.303, df = 2) on all 26 criterion rows: per-seed values within 6e-5 of verdicts.csv, every pass flag equal, interval ends within 1e-4 (t 4.303 vs the report's 4.30265). E3-D1 FAIL, E3-D2 FAIL, as in verdicts.csv and the RESULTS verdict table. |
| E. Independent CPU recomputation (seed 101) | PASS | (i) anchor KL on all 500 D1 positions 0.0489748 (mine) vs 0.0489753 (run), per-position max abs diff 2.3e-4, r = 0.9999999 (16:52:57Z). (ii) LL_window for given cone types within 6e-6 of the executor's function; my clean / myopic / tau-3.5 rollouts on all 2,361 pairs match all 15 tier statistics within sampling error (largest abs z 1.20); draw for draw identical with a shared random stream (16:57:41Z). |
| F. Code vs prereg spot checks | PASS (minor undeclared details) | 15 items checked at file:line (table under F). Undeclared but harmless: LL token feature clamped to [-30, 0] before /5, residuals clamped to [-8, 8] in the attention input, strata of 3,778/3,777 (37,776 is not divisible by 10). |
| G. Compute | PASS (correction C7) | Ledger $0.18 + $0.13 + $0.15 = $0.46 at $0.87/h (0.53 h). gcloud operations log (16:47:49Z): insert start to delete done 800 / 592 / 650 s = $0.193 + $0.143 + $0.157 = $0.49. Registered VM gone 106 s after its pipeline finished; es2 65 s; es (aborted) 58 s after its driver gave up. Instances and disks lists (GCP_PROJECT, name~soup-gpu-r8e3-) return [] (16:47:26-30Z and 17:00:57-17:01:01Z). |
| H. RESULTS-R8-E3.md | PASS with corrections C2-C6, C8-C10 | Every rule has a verdict with per-seed values or means; two rows lack the rule's own interval (C8). Failures stated as failures; post hoc labelled. 64 numbers checked against the files, all match (list under H). Overreach: C2, C3, C9; precision: C4-C6. |
| I. Exploratory early-stopping rerun | PASS (wording C5, C6) | Labelled post hoc in RESULTS, the amendments log, both scripts and the page; not inside the JSON outputs. verdicts.csv (16:12:26Z) predates it. In every seed all 36 naive-baseline columns, all 16 K = 0 columns, nL_L, nL_R, anchor_t4 and routeb are identical per position in runs/ and runs_es/. Early stopping uses calibration roots only. Best step = initialisation (-1 in fit_es.json) for both attention nets in all seeds; their outputs are identical per position; KL numbers match. |

## Details

### A. Timeline and pre-registration (file mtimes and log stamps; executor-controlled except the gcloud stamps)

- 15:42:28Z spec written (sha256 b19a33e0..., equals the prereg's citation).
- 15:51:33Z `r8/r8lib.py`, 15:53:35Z `r8/run_r8e3.py` last modified.
- 15:54:07-15:55:17Z pre-hash CPU smoke (`/tmp/r8e3/smoke/`; seed 101; first 32 D1 and first 8 D2 (100 pairs) TEST
  positions). Disclosed in prereg section 6 item 11. Every disclosed number is in `/tmp/r8e3/smoke/eval/*.json`
  (KL(0) 0.0423; K = 16: raw_count 0.675, ataraxos_mmd 3.885, union_t 0.058, scaled_mix 0.138, looped 0.045; D2 MSE
  0.0185 / 0.0186 / 0.0448; tier KL 0.374 / 0.236 / 0.440; mixed 0.0184 / 0.0217 / 0.0194), except attn_L3-K3c at
  K = 16, which is 0.0625 in the file and 0.063 in the disclosure (a rounding slip, no effect).
- 15:54:50Z and 15:55:42Z prereg drafts (`/tmp/r8e3/prereg_head.md`, `prereg_tail.md`). A diff against the final file
  shows sections 2-6 unchanged; the later additions are the amendments sentence, the verbatim spec section, the note
  on the R8-E1 amendment (section 6 item 1), the smoke disclosure (item 11) and section 7. No rule, threshold, grid or
  definition changed after the smoke.
- 15:57:12Z `r8/report_r8e3.py` last modified (after the smoke, before the hash); 15:57:15Z it was run on copies of
  the smoke output (`/tmp/r8e3/fakeruns/`). It is the scoring script; my own implementation of section 2 (check D)
  reproduces all 26 of its criterion outcomes, so it encodes the registered rules. 15:57:34-15:57:50Z gpu scripts.
- 15:58:20Z prereg last modified; 15:58:30Z hashed (sha file mtime 15:58:30.73Z; the prereg's ctime 15:58:30.73Z is
  probably the chmod to read-only). 15:58:44Z launch (launches.log); 15:58:53.5Z gcloud insert started. GPU-LOG.md has 20
  lines naming soup-gpu-r8e3-, all from 15:58Z on: no earlier r8e3 VM.
- Code that ran: the staged copies in `/tmp/r8e3/src` (r5 env6/msgame6/model6/train6/eval6/agents6, r6 e3lib/run_e3,
  r7 r7lib, r8 r8lib/run_r8e3/expl_d1_es, gpu run_vm/run_vm_es) equal the current sources byte for byte (sha256,
  16:46Z). r5/r6/r7 sources predate round 8 (latest r7lib.py 09:32:49Z). `r8/gpu/stage.sh` was rewritten at
  16:15:16Z for the exploratory run (it now also copies expl_d1_es.py and run_vm_es.sh).
- Amendments. Log entries: the outbox file (15:57:11Z) is R8-E1 only; rewritten 15:58:46Z, 2 s after launch, still
  R8-E1 only (15:59:01Z); at 16:12:22Z "last rewritten 15:58:46Z, R8-E1 only". Present state (stat and grep, 16:41Z
  and 17:00:41Z): the only round-8 amendments file has mtime 16:01:43Z and holds Part A (R8-E1, "Timestamp
  15:58Z") and Part B (R8-E2 readings 5, 8, 9, 11, 13, "Timestamp 16:02Z"). A grep for R8-E3, Stratego, trust,
  cons_pool and attn_L3 finds only R8-E1 code names. So no R8-E3 amendment was missed; the 16:12:22Z entry and
  RESULTS lines 6-7 and 181 misstate the last rewrite and the file's scope (C1). The 15:57:11Z and 15:58:46Z
  versions were overwritten, so I could not read them. Stop rule (iv) would have applied an R8-E3 amendment that
  arrived after launch but before scoring; none arrived.
- Recomputing every criterion without the smoke-seen test positions (D1 positions 32-499, D2 positions 8-239;
  `aux/aud_smoke_excl.py`, 16:59:54Z) gives the same 26 outcomes (for example attn KL(16) - KL(0) 0.588
  [0.447, 0.730]; Spearman overall 0.331 [0.198, 0.463]).

### B. Oracle use

- Reads of oracle data (grep for np.load, oracle, pstar, qstar, open; 16:48Z): `r8/run_r8e3.py:141` and `:144`
  (oracle_d1_test.npz, pstar_full; part_d1) and `:280`, `:286` (oracle_q_test.npz, qstar; part_d2). `r8/r8lib.py`,
  `r7/r7lib.py`, `r6/e3lib.py` and `r8/expl_d1_es.py` read no file. `r8lib.bias2`'s argument named qstar is called only
  from part_d2.
- fit_d1 (`r8/run_r8e3.py:57-126`): roots from `train6.two_flank_roots(Rc, ...)`, which selects on legal moves and t
  only (`r5/train6.py:263-267`); target `Rc["slot_true"]` (true hidden types of calibration positions) by
  cross-entropy (`r8/r8lib.py:305-309`). fit_trust_select (`r8/run_r8e3.py:199-261`): calibration roots only; target
  excess surprise against `Rc["result"]`; hyperparameters by log loss of the pooled effective win against the realised
  result on 1,000 further calibration roots. expl_d1_es.py: training and validation both from `Rc`.
- My view on the true-type target: it is not oracle leakage. The spec's protocol (item 1) makes the exact posteriors
  and Q* evaluation-only; the true types of calibration positions are realised data, the same labels theta_real's
  belief head was trained on, and the spec's own R8-E1 design trains its attention pools "on cal boards using only
  realized board labels Y". No test position's true types are read anywhere. The cost of this target is variance: one
  realised type per slot is a noisy target, and with no held-out stopping the attention nets overfit it. That is a
  fitting failure (which RESULTS states), not leakage.
- Other oracle-derived inputs: none. Test positions are indexed by (game, t) from the oracle files (identical to
  `d1_pos.npz`), and part_d2 asserts the legal-move order against oracle_q_test's `canon` field; neither feeds a
  model, a fit or a selection.

### C and D. Own re-derivation and verdicts (`aux/aud_cd.py`; output `aux/aud_cd.out`)

Rule rows (3-seed mean [95% seed-t interval]; my values from the CSVs; every pass flag equals verdicts.csv):

| criterion | per seed (101, 102, 103) | mean [95% CI] | bar | pass |
|---|---|---|---|---|
| D1(1) raw_count KL(16) - KL(0) | 0.58895, 0.59761, 0.58606 | 0.5909 [0.5759, 0.6058] | > 0.010, CI > 0 | yes |
| D1(1) ataraxos_mmd (0.25) KL(16) - KL(0) | 4.6622, 4.3518, 5.1703 | 4.7281 [3.7016, 5.7546] | > 0.010, CI > 0 | yes |
| D1(2a) attn max_K KL(K) - KL(0) | 0.54612, 0.63171, 0.58022 | 0.5860 | <= 0.0005 each seed | no |
| D1(2b) attn KL(16) - KL(0) | same | 0.5860 [0.4790, 0.6931] | CI < 0 | no |
| D1(2a) looped max_K | 0.50701, 0.55561, 0.54562 | 0.5361 | <= 0.0005 each seed | no |
| D1(2b) looped KL(16) - KL(0) | same | 0.5361 [0.4723, 0.5998] | CI < 0 | no |
| D1(2c) JSD scaled_mix / attn | 0.0141, 0.0176, 0.0123 | 0.0147; diff -0.0184 [-0.0198, -0.0171] | >= 8, diff CI > 0 | no |
| D1(2c) JSD ataraxos_mmd / attn | 3.167, 3.270, 3.311 | 3.249; diff 0.0421 [0.0368, 0.0473] | >= 8, diff CI > 0 | no |
| D1(2d) KL scaled_mix - attn | -0.5485, -0.6342, -0.5829 | -0.5885 [-0.6957, -0.4814] | CI > 0 | no |
| D1(2d) KL ataraxos_mmd - attn | 4.1161, 3.7201, 4.5900 | 4.1421 [3.0600, 5.2242] | CI > 0 | yes |
| D1(2d) KL kc_dedup - attn | -0.5471, -0.6317, -0.5793 | -0.5860 [-0.6921, -0.4800] | CI > 0 | no |
| D2(1) MSE unif - cons_pool | 0.00020, -0.00083, 0.00011 | -0.00017 [-0.00159, 0.00125] | CI > 0 | no |
| D2(1) 1 - cons/stiff | 0.6313, 0.5855, 0.7156 | 0.6441; diff 0.0257 [0.0171, 0.0343] | >= 0.60, diff CI > 0 | yes |
| D2(1) MSE trust - cons | 0.00113, -0.00080, 0.00052 | 0.00029 | mean > 0 | yes |
| D2(1) Bias^2 unif / stiff / trust - cons | | 0.00040 / 0.02285 / 0.00045 | means > 0 | yes |
| D2(2a) Spearman(beta, -KL) overall | 0.2913, 0.3915, 0.3086 | 0.3305 [0.1974, 0.4636] | >= 0.40, CI > 0 | no |
| D2(2a) within pairs | 0.3346, 0.3688, 0.3594 | 0.3543 [0.3103, 0.3982] | >= 0.40, CI > 0 | no |
| D2(2a) beta minus -H, overall | 0.6879, 0.7795, 0.6710 | 0.7128 [0.5678, 0.8578] | >= 0.60, CI > 0 | yes |
| D2(2a) same within 10 H strata | 0.1777, 0.2921, 0.2118 | 0.2272 [0.0814, 0.3730] | >= 0.20, CI > 0 | yes |
| D2(2b) trust_gate 1 - X/unif | -0.0596, -0.0285, -0.1235 | -0.0705; diff -0.00095 [-0.0023, 0.0004] | >= 0.20, diff CI > 0 | no |
| D2(2b) trust_gate 1 - X/stiff | 0.6160, 0.5795, 0.6559 | 0.6171; diff 0.0247 [0.0191, 0.0303] | >= 0.60, diff CI > 0 | yes |
| D2(2b) cons_trust 1 - X/unif | -0.0974, 0.0068, -0.0293 | -0.0400; diff -0.00057 [-0.0026, 0.0014] | >= 0.20, diff CI > 0 | no |
| D2(2b) cons_trust 1 - X/stiff | 0.6023, 0.5939, 0.6847 | 0.6270; diff 0.0251 [0.0190, 0.0312] | >= 0.60, diff CI > 0 | yes |
| D2(2b) clean cons_trust - unif, every seed | 0.00126, -0.00008, 0.00058 | | <= 0 each seed | no |

Rebuild details: Q*_eff rebuilt from `oracle_q_test.npz` in (position, legal index) order equals d2_actions.csv
(pos, act, qstar_eff to 5e-7). Pools rebuilt from d2_rollouts.csv (slots 0-15 clean; mixed = slots 0-9 and 16-21)
with trust.json's selections (tau_cons 8 / 1 / 8; alpha 0.5; M 2 / 2 / 4; (tau, alpha) for cons_trust (2, 0.5),
(8, 0.5), (2, 0.5); gamma 0): unif, stiff2 = (1 - H/log 3)^2, stiff_fit, trust_pool = beta^alpha, cons_pool =
exp((LL - max LL)/tau), cons_trust_pool, cons_tau1, trust_gate (drop the M lowest beta, stable sort so ties drop the
earlier index), clean10. Within-pair Spearman: no constant bag (2,361 of 2,361 used). Rounding: the CSVs carry 6
significant digits; the largest effect on any rule statistic is 1.8e-5 (within-pair Spearman of -H, seed 103), on a
D1 mean 3.5e-7, on an MSE 1e-7 (relative). The criterion nearest its bar (D2(2a) strata, 0.227 vs 0.20, lower bound
0.081) is far from any rounding effect. The report's t = 4.30265 vs my 4.303 moves interval ends by at most 1e-4 and flips nothing.

### E. Independent recomputation (seed-101 checkpoint, CPU, 4 threads)

My code: `aux/aud_net.py` (forward pass written from the R5 model description and the checkpoint tensors, tapes from
`test.npz`, cone choice, stiff_voi any-order sampling at t0 with the slot-trust order, piece corruption, LL_window,
cone_lp at t, value rollouts inside theta_real), driven by `aux/aud_e.py`. It imports nothing from r5-r8. I wrote it
after reading the executor's code, so it shares the executor's reading of the prereg; it shows that the numbers are
reproducible and the code does what the prereg says (F), not that this reading is the only one.

(i) Anchor (`aud_e.py anchor`, 16:52:57Z): KL(P*_full || ell_t) per hidden slot on all 500 D1 positions: mean 0.0489748
(mine) vs 0.0489753 (`r8/runs/s101/eval/d1_positions.csv`, kl_*_K0); first 200 positions 0.0439229 vs 0.0439227;
per-position mean abs diff 1.9e-5, max 2.3e-4 (GPU TF32 vs my FP32), correlation 0.9999999; same positions, order
and hidden-slot counts.

(ii) LL_window for given cone assignments (`aux/aud_e_ref.py`, 16:57:41Z): on 96 window tapes of D2 test positions,
with the same cone types fed to both, my LL vs `r8lib.ll_window` differs by at most 6.0e-6 (sampled types) and 5.2e-6
(corrupted types). My window tapes, opponent-move masks, move indices, root tapes and cone choice (all 2,361 pairs)
are identical to the executor's; forward-pass logits agree within 1.9e-5.

Rollouts, distribution level (`aud_e.py rollouts 3 0|1|2`, own random streams; combined by `aux/aud_e_combine.py`):
16 clean + 3 myopic noisy + 3 tau-3.5 noisy rollouts on each of the 2,361 test pairs, seed 101, compared pair by
pair with the executor's rollouts for the same pairs in d2_rollouts.csv (difference of pair means; SE by bootstrap over
pairs) and with d2.json:

| tier | statistic | mine, all 2,361 pairs | run (d2.json) | mine - run [SE] | z |
|---|---|---|---|---|---|
| clean | KL(Q* \|\| q_r) | 0.3597 | 0.3581 | +0.0016 [0.0032] | +0.50 |
| clean | squared error | 0.04598 | 0.04577 | +0.00021 [0.00047] | +0.44 |
| clean | share H < 0.15 | 0.1343 | 0.1344 | -0.00003 [0.0021] | -0.01 |
| clean | LL_window | -6.176 | -6.177 | +0.001 [0.014] | +0.09 |
| clean | cone_lp | -2.071 | -2.064 | -0.007 [0.010] | -0.65 |
| myopic noisy | KL | 0.2634 | 0.2676 | -0.0042 [0.0048] | -0.88 |
| myopic noisy | squared error | 0.03955 | 0.04046 | -0.00091 [0.00088] | -1.03 |
| myopic noisy | share H < 0.15 | 0.0243 | 0.0254 | -0.0011 [0.0022] | -0.52 |
| myopic noisy | LL_window | -6.304 | -6.269 | -0.035 [0.031] | -1.15 |
| myopic noisy | cone_lp | -6.191 | -6.143 | -0.048 [0.040] | -1.20 |
| tau-3.5 noisy | KL | 0.4161 | 0.4251 | -0.0090 [0.0106] | -0.84 |
| tau-3.5 noisy | squared error | 0.05827 | 0.05819 | +0.00008 [0.00164] | +0.05 |
| tau-3.5 noisy | share H < 0.15 | 0.0959 | 0.1015 | -0.0057 [0.0049] | -1.16 |
| tau-3.5 noisy | LL_window | -6.346 | -6.356 | +0.011 [0.032] | +0.34 |
| tau-3.5 noisy | cone_lp | -6.167 | -6.194 | +0.027 [0.040] | +0.66 |

51,942 own rollouts in three CPU jobs of every third position (offsets 0, 1, 2; 220, 231 and 247 s; 16:53-17:09Z). The
run's values for the same pairs, rebuilt from d2_rollouts.csv, equal d2.json. Largest |z| 1.20 on the combined set; in
the 45 per-part comparisons one |z| exceeds 2 (myopic LL in part 2, z = -2.45), about what chance gives.

Rollouts, draw for draw (`aux/aud_e_ref.py`): with the same generator seed and the same order of random calls as
`r8lib.det_rollouts`, on 120 pairs x 2 rollouts per tier, my e and n_ply equal the executor's in 240 of 240 rollouts in
each tier; LL within 9.5e-6, cone_lp within 4.8e-6, q_leaf within 5.7e-7. The run wrote nothing into r5-r8 (it ran
with PYTHONDONTWRITEBYTECODE=1; a find for files newer than the brief at 16:58Z shows only the executor's
`r8/site-entry.json` from 16:39:45Z).

My rollouts also reproduce the RESULTS finding that the corrupted myopic tier is more accurate than the clean tier
(squared error 0.0396 vs 0.0460) and the tau-3.5 tier is worse (0.0583).

### F. Code vs prereg (file:line)

| item | code | matches prereg |
|---|---|---|
| anchor = current-turn belief | `r6/e3lib.py:90` p0b = make_roots' belief at t; `r8/run_r8e3.py:159-160` K = 0 returns it | yes |
| residual = lmu - ell_t on cone slots | `r8/r8lib.py:184` | yes |
| determinization at t0 = t - 4 | `r8/r8lib.py:180` (D1), `:123` (D2), sampled on the t0 tape | yes |
| LL over the opponent's window moves only | `r8/r8lib.py:57`, `:80` | yes |
| raw_count 1% floor | `r8/r8lib.py:229` | yes |
| fixed gamma 0.25 | `r8/run_r8e3.py:34`, `:131` (times beta_r, as declared) | yes |
| attention inputs: anchor token W_a ell_t + learned vector; k_r, W_v dl_r, log beta_r, log c_r, LL_r/5 | `r8/r8lib.py:285-289` | yes; LL clamped to [-30, 0] and dl to [-8, 8] (undeclared, harmless) |
| softplus weights initialised at 0.02 | `r8/r8lib.py:278-279` (zero output weights) | yes |
| looped = one block x 3 with step embedding | `r8/r8lib.py:271`, `:291-294` | yes |
| K-prefix bags, 85/15 skew, own pools per policy | `r8/r8lib.py:194-217`; `r8/run_r8e3.py:154-157` | yes |
| fitting data and target, steps, learning rates | `r8/run_r8e3.py:62-119` | yes |
| theta_trust inputs (no tier label), 3,000 roots, 48,000 rollouts | `r8/r8lib.py:142`; `r8/run_r8e3.py:205-221` | yes |
| selection roots disjoint from training roots | `r8/run_r8e3.py:207-209` | yes |
| noisy tier: every cone type changed; 3 depth-1, 3 depth-4 at 3.5 | `r8/r8lib.py:89-93`; `r8/run_r8e3.py:194-195` | yes |
| Bias^2 = MSE - mean(var_within x sum w~^2) | `r8/r8lib.py:329-335` | yes |
| H strata | `r8/run_r8e3.py:340-345` | yes; sizes 3,778 / 3,777, not "exactly equal" (37,776 / 10 is not whole) |

Note: the r5 trust features include n_ply/6, and ply_lp is 0 when no ply is sampled, so the depth-1 tier is visible
to theta_trust through its features, and cone_lp separates corrupted cones (-6.1 vs -2.1 per slot). The spec asks for
exactly these features; "no tier label" holds literally. Here it did not help, because the noisy rollouts were not
worse.

### G. Compute

| VM | ledger (GPU-LOG.md DELETE line) | gcloud operations (insert start -> delete done) | after the run finished |
|---|---|---|---|
| soup-gpu-r8e3-main-1017155844-4062 | 0.21 h, $0.18 | 15:58:53.5Z -> 16:12:13.6Z, 800 s, $0.193 | all_done 16:10:27Z; delete call 16:11:22Z (55 s; the driver polls every 60 s); gone 106 s after |
| soup-gpu-r8e3-es-1017161527-19250 (aborted) | 0.15 h, $0.13 | 16:15:35.7Z -> 16:25:27.4Z, 592 s, $0.143 | no result; gone 58 s after the driver gave up (16:24:29Z) |
| soup-gpu-r8e3-es2-1017162557-241 | 0.17 h, $0.15 | 16:26:04.7Z -> 16:36:54.9Z, 650 s, $0.157 | all_done 16:35:50Z; delete call 16:36:00.1Z; gone 65 s after |
| total | 0.53 h, $0.46 | 0.567 h, $0.49 | |

The ledger's age runs from creation to the delete call (`infra/gpu-down.sh:65-69`, `infra/lib.sh` age_hours) and
leaves out about 50 s of deletion per VM. Both totals are under the $0.50 ceiling; the rate is the unverified
list-price ceiling. Each VM was deleted within about a minute of its driver returning (58-59 s), as RESULTS says; the
spec's "immediately" was met within the driver's 60-s polling interval. Present state: instances list and disks list
(project GCP_PROJECT, filter name~soup-gpu-r8e3-) both return [] at 16:47:26-30Z and again at 17:00:57-17:01:01Z;
the same commands with name~soup-gpu- return an r8e1 VM and its disk, so the filter works.

### H. Numbers in RESULTS-R8-E3.md checked against the files (version 16:39:03Z, sha256 30780cb4...)

All match (file in brackets; 3-seed means unless stated):
1. anchor KL(0) 0.0488 / 0.049 (d1.json K0: 0.04898, 0.04886, 0.04860). 2. raw_count KL(16) 0.640 (0.63969).
3. ataraxos_mmd (0.25) KL(16) 4.78 (4.7769). 4. attn KL(16) 0.635 (0.63483). 5. looped KL(16) 0.585 (0.58489).
6. attn KL(1) 0.247 (0.24729). 7. cal CE 1.117 -> 0.976 (fit.json; attn). 8. thirteenfold (13.0). 9. scaled_mix
0.0488 -> 0.0463, never above K = 0 (max per seed -0.00039, -0.00044, -0.00041). 10. alpha 0.039 (0.03898).
11. union_t K16 0.0736. 12. ataraxos_mmd fitted step 0.0035 (0.00352), KL 0.0481. 13. JSD K16 0.00027 / 0.00056 /
0.0187 / 0.0608. 14. 68x (0.018705 / 0.000274). 15. t-4 anchor 0.179, every slot 0.080 (d1.json). 16-21. clean MSE
0.01394 / 0.03982 / 0.01440 / 0.01412 / 0.01453 / 0.01501. 22-27. clean Bias^2 0.01186 / 0.03432 / 0.01191 /
0.01146 / 0.01146 / 0.01143. 28-34. mixed MSE 0.01432 / 0.03996 / 0.01527 / 0.01489 / 0.01434 / 0.01482 / 0.01529.
35-37. squared error 0.0451 / 0.0397 / 0.0592. 38-40. KL to Q* 0.359 / 0.272 / 0.434. 41-43. share H < 0.15
0.135 / 0.024 / 0.100. 44-46. LL -6.18 / -6.30 / -6.31. 47-48. cone_lp -2.07 / -6.14 to -6.17. 49-50. ply_lp
-2.22 / -9.52. 51-53. beta 4.02 / 3.96 / 3.37. 54-57. Spearman 0.330 / -0.382 / -0.042 / 0.193; within LL -0.006.
58. dropped-noisy share 0.399. 59. delusional about 0.005 for stiff_pool only (0.0042 clean, 0.0056 mixed).
60. effective sample 14 of 16 (14.07); noise 0.0027 vs 0.0021. 61. 56% off-support moves (0.556). 62. selected
hyperparameters (trust.json). 63. 108,616 / 41,800 parameters; 473 s per seed (done.json). 64. self-check 318
numbers, max 4.09e-7 (selfcheck.csv). Exploratory table and text: KL curves, -0.0002 per seed, -0.0025 [-0.0028,
-0.0023], best steps 1,200 / 600 / 500 (scaled_mix) and 1,200 (kc_dedup), held-out CE 1.0864 / 1.1116 / 1.1380 and
1.0844 / 1.1089 / 1.1348, JSD ratio 5.1, KL difference +0.0004 [-0.0004, +0.0011] (expl_es.json, fit_es.json).

Overreach and precision issues: C2-C6, C8-C10. Every rule has a verdict; failures are failures; the exploratory rerun is
labelled and kept out of the verdict table.

### I. Exploratory early-stopping rerun

- Labels: RESULTS (TL;DR "Post hoc rerun ... exploratory; no verdict changes", section header "Exploratory ...
  (post hoc ...)", deviations log), PREREG-R8-E3-AMENDMENTS.md (16:15:26Z "POST HOC, EXPLORATORY (no verdict
  changes)"), `r8/expl_d1_es.py` and `r8/expl_es_summary.py` docstrings, the page
  `/home/agent/memo-serve/lir-agent/r8-e3.html` (7 "post hoc", 10 "exploratory"; read 16:46Z). The JSON
  outputs (`r8/runs_es/s*/eval/*.json`, `r8/report/expl_es.json`) carry no label of their own; the folder name and
  scripts do (optional fix: add a "label" key).
- No verdict changes: verdicts.csv (16:12:26Z) predates the rerun's launch (16:15:27Z); the RESULTS verdict table equals
  verdicts.csv.
- Same test bags: per position, in every seed, the 36 naive-baseline columns (raw_count, union_t and ataraxos_mmd at
  gamma 0.25; KL and JSD at every K), all 16 K = 0 columns, nL_L, nL_R, kl_anchor_t4, kl_routeb_all, game, t and
  n_hidden are identical between `r8/runs/` and `r8/runs_es/` d1_positions.csv; only the five fitted aggregators
  differ (16:45Z).
- Calibration only: `r8/expl_d1_es.py:46-49` draws 2,000 calibration roots with the registered generator seed (the same
  roots as the registered fit), trains on the first 1,500 and validates on the last 500 against their true types.
  The aborted first attempt synced nothing (its driver exited before the sync loop), so runs_es comes from es2 only.
- Claims: best step is -1 (before the first step, the initialisation) for both attention nets in all seeds
  (fit_es.json); RESULTS calls it "step 0". Held-out CE was already worse at the first check (seed 101: 1.0851 at
  initialisation, 1.1088 at step 50, 1.1272 at step 100); checks ran every 50 steps. The initial output is exactly
  0.02 per rollout and slot because the output layer starts with zero weights; attn and looped outputs are identical
  per position at every K (max abs diff 0). The "5.1x" JSD ratio and the +0.0004 KL difference are against the
  scaled_mix refitted in the rerun; against the registered scaled_mix they are 4.3 [2.8, 5.9] and +0.00004 [-0.0003,
  +0.0004] (C6). Same conclusion either way.

## Corrections (exact suggested wording)

C1 (provenance; RESULTS lines 6-7 and 181, PREREG-R8-E3-AMENDMENTS.md 16:12:22Z entry). Replace "The round-8 amendments
file covers R8-E1 only; it was read before hashing, again after it was rewritten at 15:58:46Z, and before scoring" with:
"The round-8 amendments file has nothing for R8-E3. It was read before hashing (R8-E1 only) and after its rewrite at
15:58:46Z (still R8-E1 only). It was rewritten again at 16:01:43Z, before scoring, to add Part B (R8-E2 readings); my
pre-scoring check logged the wrong last-rewrite time and missed Part B. Neither part concerns R8-E3, so nothing
changes." Append to the amendments log: "<UTC>: correction after audit: the 16:12:22Z entry gave the amendments
file's last rewrite as 15:58:46Z. It was rewritten at 16:01:43Z, adding Part B (R8-E2 readings 5, 8, 9, 11, 13).
Nothing in it is for R8-E3; no change."

C2 (overreach; TL;DR line 29 and "What this means" item 4). TL;DR: replace "window-consistency weighting lowers bias a
little but adds more noise" with "window-consistency weighting did not lower MSE. Its 3-seed mean Bias^2 is slightly
lower than the plain mean's, but the difference, 0.0004 [-0.0003, 0.0011], includes 0, and its weights add noise".
Item 4: replace "Window consistency is real but weak evidence here. cons_pool's importance weights exp(LL / tau) move
the mean slightly toward Q* (Bias^2 0.01146 vs 0.01186), as the spec predicted." with "Window consistency gave no
measurable gain here. cons_pool's 3-seed mean Bias^2 is lower (0.01146 vs 0.01186), as the spec predicted, but the
difference, 0.0004 [-0.0003, 0.0011], includes 0, and LL does not rank rollouts by accuracy (Spearman with -KL -0.042
overall, -0.006 within pairs)."

C3 (overreach; "What this means" item 1). Replace "Any rule whose step is small enough is safe." with "The rules with
a small fitted step, scaled_mix and ataraxos_mmd with a fitted step (and, post hoc, the early-stopped attention), stayed
below the anchor at every K in every seed; kc_dedup(K3c) rose +0.0009 above it at its worst K in seed 103, over the
0.0005 bar." Replace "the rollouts' window evidence is already in the one-pass belief, so adding it again double
counts" with "a likely reason, not tested here, is that the window evidence is already in the one-pass belief, so
adding it again double counts". Replace "The exact posterior is closer than the anchor, but not by much, and nothing
here found a way to close more than 5% of the gap." with "The anchor is already close to the exact posterior (KL
0.049); the best rule here closed 5% of that gap."

C4 (precision; TL;DR lines 19-20). Replace "Their calibration cross-entropy fell from 1.12 to 0.98 while test KL rose
thirteenfold." with "attn_L3-K3c's calibration cross-entropy fell from 1.12 to 0.98 (looped_attn_T3-K3c: 0.99) while its
test KL rose thirteenfold (looped_attn_T3-K3c: twelvefold)."

C5 (precision; TL;DR line 25, item 3, exploratory section). Replace "no training step ever improved held-out
cross-entropy" with "no checkpoint, checked every 50 steps, improved held-out cross-entropy; it was already worse at
step 50 in every seed". After "the best held-out step was step 0, the initialisation" add "(recorded as -1 in
fit_es.json)". Replace "It changes one thing:" with "It changes only the fitting:" (the training set also shrank from
2,000 to 1,500 positions).

C6 (clarity; TL;DR line 27 and item 3). After "5.1x below scaled_mix, bar 8x" and after "difference +0.0004 [-0.0004,
+0.0011]" add: "(against the scaled_mix refitted with early stopping in the same rerun; against the registered
scaled_mix: ratio 4.3 [2.8, 5.9], KL difference +0.00004 [-0.0003, +0.0004])".

C7 (cost; TL;DR line 38 and Compute section line 175). Line 38: replace "Cost: **$0.46** at the $0.87/h list-price
ceiling" with "Cost: **$0.46** at the $0.87/h list-price ceiling by the ledger (VM creation to delete call), $0.49
counting each VM until its deletion completed (gcloud operations log: 800, 592 and 650 s); both under the $0.50
ceiling". Line 175: replace "Total **$0.46** of the $0.50 R8-E3 ceiling" with "Total **$0.46** of the $0.50 R8-E3
ceiling by the ledger ($0.49 counting each VM until its deletion completed)".

C8 (completeness; verdict table rows for the JSD clause). scaled_mix row: replace "0.0147 (attn's JSD 0.0187 is 68x
higher)" with "0.0141, 0.0176, 0.0123; 0.0147; difference -0.0184 [-0.0198, -0.0171] (attn's JSD 0.0187 is 68x
higher)". ataraxos_mmd row: replace "3.25 [3.06, 3.43]" with "3.17, 3.27, 3.31; 3.25; difference 0.0421 [0.0368,
0.0473] (the interval passes; the ratio misses the 8x bar)".

C9 (untested mechanisms; items 2 and 5). Item 5: replace "the value head right after the root move is a good
estimator, and the corrupted pieces matter only if the root move attacks them" with "the value head right after the root
move is a good estimator here; why corrupted cones barely hurt it was not tested (the value head reads the corrupted
slot types too, so an attack is not the only route)". Item 2: replace "A 100k-parameter network with an anchor token
that identifies each position learned those positions' answers" with "A 100k-parameter network fitted on the same
2,000 positions for 1,200 full-batch steps learned those positions' answers (held-out loss rose from the first check in
the post hoc rerun; the route, for example the anchor token acting as a position identifier, was not tested)".

C10 (pointer; RESULTS line 193). Replace "Independent audit: `r8/AUDIT-R8-E3.md` (pending at the time of writing)."
with "Independent audit: `r8/AUDIT-R8-E3.md` (copied from
/home/agent/workspaces/fable-strategy/scratch/audit-r8e3/AUDIT-R8-E3.md): supported with corrections; every
verdict reproduced."

## Post-audit status (read-only checks 17:10:49Z-17:13Z)

- At 17:07:54Z the executor copied an UNFINISHED draft of this file to `r8/AUDIT-R8-E3.md` (sha256 bb3342c06fc80680,
  35,825 bytes; it still holds four unfilled placeholders, for the largest z, the pair count, the rollout table and three
  squared errors in check E, and two superseded sentences). Re-copy this final file over it. The draft's corrections C1-C10 already had their final wording,
  except that C7 did not yet name RESULTS line 175; the executor changed that line anyway.
- At the same time the executor applied C1-C10 to `r8/RESULTS-R8-E3.md` (now sha256 38520a1106557649, mtime 17:07:54Z)
  and logged them in `r8/PREREG-R8-E3-AMENDMENTS.md` (two entries stamped 17:07:54Z). Read at 17:11Z: all ten
  corrections are present with the suggested wording; the verdict table, verdicts and numbers are unchanged.
- One leftover from the edit, in "What this means" item 4: the old sentence that follows the new text now repeats the
  Spearman numbers and opens with "But". Suggested: replace "But two opponent moves carry little evidence about three
  cone pieces: LL barely separates good determinizations from bad ones (Spearman with -KL -0.042, within pairs -0.006),
  so the weights mostly add noise" with "Two opponent moves carry little evidence about three cone pieces, so the
  weights mostly add noise".
- At 17:10:50-55Z gcloud instances list and disks list (GCP_PROJECT, name~soup-gpu-r8e3-) again return [].

## What I could not check

- The 15:57:11Z and 15:58:46Z versions of the outbox amendments file (overwritten; only the 16:01:43Z version exists).
- That the VMs ran exactly the staged files: the staging folder was rebuilt at 16:15Z and 16:25Z; its current copies
  equal the current sources, whose mtimes predate the registered launch, but no VM-side copy of the code was kept.
- An independent recomputation of the D1 rollout pools, the naive baselines at K >= 1 and the fitted D1 aggregators
  (the pools are not stored, and refitting needs a GPU-scale run); I checked these through the CSVs, the JSONs and the
  identity of the naive columns between the registered run and the rerun.
- The theta_trust fit and the hyperparameter search (checked by code reading and the selected values only); my
  rollout check covers seed 101 only.
- True spend (the Cloud Billing API is disabled per the ledger); only the $0.87/h list-price ceiling.
- The agent mail to opus-mats (thread 3734) that the log says preceded the exploratory launch.
- I used no automatic text classification, so no hand-labelled sample was needed.

## Commands I ran (all read-only outside my scratch folder; UTC)

- 16:39:22Z `date`; `ls -la /tmp/r8e3`; read `/tmp/r8e3/audit_brief.md`.
- 16:39-16:41Z `ls -la --time-style=full-iso` and `stat` on r8/, r8/gpu, r8/report, r8/runs*, /tmp/r8e3 (find with
  mtimes); `cat` of PREREG, its .sha256, the amendments log, launches.log, GPU-LOG-R8-E3.md; `sha256sum
  r8/PREREG-R8-E3.md`; `sed -n 123,133p` spec vs `sed -n 20,30p` prereg with `cmp` (for this I wrote two 2.8 kB text
  files in /tmp at 16:40Z and removed them a minute later; every later scratch file is in my folder).
- 16:41Z `stat` and `grep` of `/home/agent/workspaces/mats/outbox/experiments-round-8-amendments.md`.
- 16:41-16:50Z read code with `cat -n`/`sed -n`: r8/r8lib.py, r8/run_r8e3.py, r8/report_r8e3.py, r8/expl_d1_es.py,
  r8/expl_es_summary.py, r8/gpu/*.sh, r6/e3lib.py, r5/model6.py, r7/r7lib.py, r5/train6.py, r5/eval6.py,
  logical-soup infra gpu-down.sh and lib.sh; grep for oracle reads.
- 16:43:46Z `python3 aux/aud_cd.py > aux/aud_cd.out` (checks C, D); 16:58Z row-by-row comparison of its verdict rows
  with verdicts.csv (26 rows, 0 mismatches).
- 16:44-16:46Z python one-liners on fit.json, trust.json, d1.json, d2.json, done.json, expl_es.json, fit_es.json;
  per-position comparison of runs/ vs runs_es/ d1_positions.csv; sha256 of staged vs current code; grep of the page.
- 16:45Z `grep r8e3` in `/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md`; `cat` of
  /tmp/r8e3/launch-r8e3-*.log, mail drafts and watch_es2.sh.
- 16:47:26-16:47:52Z `gcloud compute instances list` and `gcloud compute disks list` (project GCP_PROJECT,
  `--filter='name~soup-gpu-r8e3-'`, and `name~soup-gpu-` as a filter sanity check); `gcloud compute operations list
  --filter='targetLink~soup-gpu-r8e3-'`.
- 16:52:57Z `python3 aux/aud_e.py anchor`; 16:53:05Z `aud_e.py rollouts 60` (timing); 16:53-16:57:24Z
  `aud_e.py rollouts 3` (offset 0), 17:01:31-17:09:34Z `aud_e.py rollouts 3 1` and `... 3 2`; 17:09:43Z `python3
  aux/aud_e_combine.py aux/aud_e_s3.out aux/aud_e_s3_o1.out aux/aud_e_s3_o2.out > aux/aud_e_combined.out`.
- 16:57:41Z `R5E2_DATA=/tmp/r5e2/data PYTHONDONTWRITEBYTECODE=1 python3 aux/aud_e_ref.py`; before and after, `find
  r5 r6 r7 r8 -newer /tmp/r8e3/audit_brief.md` (only the executor's r8/site-entry.json, 16:39:45Z).
- 16:59:54Z `python3 aux/aud_smoke_excl.py`.
- 17:00:41Z and 17:00:57-17:01:01Z final hashes of the audited files, prereg hash, `cmp` of section 1, launches.log,
  GPU-LOG.md, and both gcloud lists again.
- 17:10:49-17:10:55Z sha256 and stat of RESULTS and the amendments file, `find r5 r6 r7 r8 /tmp/r5e2 -newer
  /tmp/r8e3/audit_brief.md` (showed the executor's 17:07:54Z edits), both gcloud lists; 17:11Z `cmp`/`diff` of
  r8/AUDIT-R8-E3.md against this file and `sed -n` of the corrected RESULTS.

All python ran with OMP_NUM_THREADS=4 (torch.set_num_threads(4)), PYTHONPATH=/tmp/r5e2/pylib, R5E2_DATA=/tmp/r5e2/data,
`nice -n 10`, one process at a time, each under 4 minutes. Scripts and outputs (re-runnable):
`/home/agent/workspaces/fable-strategy/scratch/audit-r8e3/aux/` (aud_cd.py, aud_net.py, aud_e.py,
aud_e_ref.py, aud_e_combine.py, aud_smoke_excl.py and their .out files).
