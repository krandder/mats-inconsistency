# AUDIT-R6-E3: independent audit of R6-E3 (E-MS-MARGINAL)

Auditor: an independent read-only investigator (Claude Opus 5.5, 1M context, model id claude-opus-5-5[1m]), separate from
the executor. Audit run 2026-10-17 06:58Z to 07:24Z (UTC throughout). Requested location
`/home/agent/workspaces/mats/work/micro-stratego/r6/AUDIT-R6-E3.md`; the auditor may write only in its own scratch, so this
file is `/home/agent/workspaces/fable-strategy/scratch/audit-r6e3/AUDIT-R6-E3.md`. All auditor scripts and outputs
are in the same folder. Paths below without a leading `/` are relative to `/home/agent/workspaces/mats/work/micro-stratego/`.
Audited files (sha256 prefix, mtime; unchanged during the audit, checked 07:21Z): `r6/RESULTS-R6-E3.md` 15349a3b (06:57:26Z), `r6/report/verdicts.csv` 0ceb34d9 (06:54:19Z), `r6/report/expl_d2.json` 367e924a (06:55:44Z), `r6/PREREG-R6-E3-AMENDMENTS.md` 38787050 (06:44:58Z). Nothing in the experiment folders was edited. CPU only, one process with at most 4 torch threads at a time (the oracle
reproduction used 4 worker processes), PYTHONPATH=/tmp/r5e2/pylib, R5E2_DATA=/tmp/r5e2/data. No GPU, no git, no messages.

## Verdict

**Supported with corrections.** The pre-registration is intact: its sha256 matches, it was hashed at 06:34:20Z, before the
only launch (06:35:51Z) and the first PENDING-CREATE (06:36Z), and its section 1 equals spec section 3.3 byte for byte. No
oracle file or oracle quantity reaches any fitted quantity. Every headline number re-derives from the per-item CSVs with my
own code (153 like-for-like comparisons; largest gap 2.0e-5, from CSV rounding), and every pre-registered verdict reproduces with my own
seed-t intervals (df = 2, t = 4.303): gate E3-D0 PASS (16 rollouts cut KL by 48.2% [46.9, 49.5]), E3-D1 FAIL (key gap
0.228 [0.211, 0.244], bar 0.25; all five part-(2) comparisons pass), E3-D2 FAIL (trust_pool MSE 26.8% worse than
stiff_pool; delusional rate 0 for every pool, so the comparison fails at floor as pre-registered). My own CPU
recomputation from the checkpoints (own tape, window, likelihood and cone code) reproduces the anchor KL and the
marginal window posterior on all 1,000 calibration and 1,000 test gate positions for all three seeds (per-position
difference at most 8.6e-4 nats, means within 4e-6), and my own 16-rollout draw gives a 48.1% [47.0, 49.3] cut. Compute:
one VM, $0.26 at the $0.87/h ceiling, no VM or disk left (07:17Z). The corrections concern wording, not verdicts: the
self-audit sentence on rounding overstates the tie effect by about four orders of magnitude; "5-12x" should be about
5-10x; the post hoc claim that no reweighting removes the shared bias is not shown by `r6/report/expl_d2.json`, and
about 0.0022 of that "bias" is the sampling noise of Q* itself; the scaled_mix gate number promised by the
pre-registration is missing; the D1 text should say that the calibration reference is itself the union call; and the
statement that `r6/gpu/drive.sh` stayed byte-identical cannot be verified (the file was rewritten in place at 06:44:50Z).

## Checks A-H

| check | result | evidence (read-only check, UTC time) |
|---|---|---|
| A. Pre-registration | **pass** (notes) | `sha256sum r6/PREREG-R6-E3.md` at about 07:00Z = 5b45ec96...83be11 = `r6/PREREG-R6-E3.sha256`. Hash time 06:34:20Z (file mtime 06:34:20.07Z, file read-only) < first `r6/gpu/launches.log` line 06:35:51Z < first `PENDING-CREATE vm=soup-gpu-r6e3-*` 06:36Z (logical-soup `GPU-LOG.md` line 2069). Section 1 vs spec section 3.3 (the last section of the spec): 2,088 identical bytes, sha256 be41b05e... for both, after dropping the one blank line that separates section 1 from "## 2."; also identical to the executor's `/tmp/r6e3/spec-3.3-verbatim.md`. The spec's sha256 is still 8be631f1..., as cited. Smoke disclosure: intro and section 6 item 9 disclose a pre-hash CPU smoke and say its numbers were seen (example: gate (2) 0.39; the smoke file says 0.3907). Sizes confirmed in `/tmp/r6e3/smoke_s101/eval/` (48 / 48 / 48 positions, 12 D2 positions = 132 pairs, 32 roots, 20 steps). Not listed but also visible then: key gap 0.054, JSD ratios 1.76 and 2.48, D2 Spearman(beta) 0.362, trust_pool MSE 29% worse than stiff_pool, gamma 0. No design change after the smoke: `r6/e3lib.py` mtime 06:22:57Z, and its `.pyc` (written 06:24:58Z by the smoke) records the same source mtime and size (10,759 bytes); `r6/run_e3.py` mtime 06:24:34Z, before the smoke started (06:24:58Z); the smoke ended 06:32:24Z. The VM copies staged at 06:35:46Z (`/tmp/r6e3/src/`) are byte-identical to the local `r6/` and `r5/` files (`cmp`). The `r5/` modules used were last changed at 03:22Z. Post-hash edit not disclosed: `r6/report_e3.py` changed at 06:37:17Z (after the hash and after a dry run on the smoke output at 06:37:07Z, before any real output at 06:51:55Z). The `.pyc` shows the earlier version was 13,529 bytes; the current one is 13,530. The dry-run output has "\|\|" where the final has "‖" (1 extra UTF-8 byte), so the edit was a label change. My own verdicts (D) agree with it. |
| B. No oracle in fitting or selection | **pass** | `grep -n -i 'oracle\|pstar\|qstar\|np.load\|slot_true'` over `r6/e3lib.py`, `r6/run_e3.py`, `r5/train6.py`, `r5/model6.py`: oracle files are loaded only in `run_e3.py` lines 161 and 170 (part_d0), 222 (part_d1) and 285 (part_d2). `e3lib.py` reads no file. `fit_aggregators` and `fit_gamma` receive only the calibration records (`load_split("cal")`). Their targets: the union call over base cover + 64 Left + 64 Right reference rollouts, the mean leaf value of those 128 rollouts, and for gamma the realised result (0 win, 1 draw, 2 loss for the player to move, `r5/msgame6.py` `result_for`). The records give the tape (`slot_true` only where publicly revealed, `r5/model6.py` `tape_from`), legal moves, plies and result; they hold no P* or Q*. alpha (unskewed bags), eta, u, the K4 keys (with L_key) and attn_L3 are all fitted inside `fit_aggregators`. The reused slot and rollout trust heads were trained in R5-E2 on realised hidden types, not on the oracle. |
| C. Headline numbers from per-item CSVs | **pass** | Own code `audit_cd.py` (no import of `report_e3.py`, `run_e3.py` or `e3lib.py`). 153 like-for-like comparisons with the JSON summaries: D0 (63, both splits) max difference 9.4e-8; D1 (54) max 5.3e-8; D2 (36) max 2.0e-5 (within-strata Spearman(beta), seed 103, from CSV rounding of beta and KL; overall Spearman max 2.6e-6). 3-seed means: battle accuracy 0.99994 (11,034 battles; 1, 0, 1 errors), 7-known NLL 1.2e-4 (12,159 slots), calibration anchor KL 0.1852, 16-rollout KL 0.0959, cut 48.21%; D1 JSD union / mix / scaled_mix / ataraxos_mmd / kc_dedup / attn_L3 = 1.9e-6 / 0.00167 / 0.00239 / 0.00219 / 0.00109 / 0.00023, KL = 0.1004 / 0.1193 / 0.1087 / 0.1222 / 0.1187 / 0.1016, key gap 0.2276; D2 Spearman(beta, -KL) 0.2469, Spearman(-H, -KL) -0.4167, within-strata difference 0.1965, MSE unif / stiff / trust 0.01340 / 0.01340 / 0.01705, delusional 0 / 0 / 0. Repooling the 16 rollouts per pair from `d2_rollouts.csv` reproduces `d2_actions.csv` to 1e-6; stiff_pool equals unif_pool exactly (gamma 0 in every seed). The key gap in the order the pre-registration text gives (mean over positions per policy, then the two policies) is 0.2274 [0.2107, 0.2441] instead of 0.2276; the run averages the two policies per position first. Verdict unchanged. **Rounding of H:** 4,330 / 4,406 / 4,437 rollouts per seed share one exact entropy (0.0800525, the terminal claim 0.98 + 0.02/3), and a further 2,879-4,408 rollouts share a rounded value in 1,365-1,989 smaller tie groups. With average ranks (as in the run), the CSV recomputation moves Spearman(-H, -KL) by 2.7e-8, 2.1e-7 and 3.5e-7. Even the worst ordering of the ties created by rounding moves it by less than 3e-6 (`audit_ties.json`). Strata rebuilt from the rounded H match the run's strata column for all 37,776 rollouts in every seed. Only ordinal ranks for the exact terminal ties would move it by about 0.003. |
| D. Pre-registered verdicts recomputed | **pass** | Own seed-t intervals (t = 4.303) from my CSV-derived per-seed values (`audit_cd.json`): D0 (0) 0.1852 [0.1828, 0.1876] pass; D0 (1) 0.9999 [0.9998, 1.0001] and 0.0001 [0.0000, 0.0002] pass; D0 (2) 0.4821 [0.4690, 0.4952] pass; **gate PASS**. D1 (1) 0.2276 [0.2109, 0.2443] fail. D1 (2): JSD ratio ataraxos_mmd 9.71 (difference 0.00196 [0.00123, 0.00270]) pass; scaled_mix 10.51 (0.00215 [0.00170, 0.00260]) pass; KL differences scaled_mix 0.0071 [0.0060, 0.0082], ataraxos_mmd 0.0206 [0.0142, 0.0269], kc_dedup 0.0171 [0.0100, 0.0242], all pass. **D1 FAIL.** D2 (1): 0.2469 [0.1749, 0.3188], difference 0.6636 [0.5667, 0.7605], strata 0.1965 [0.1055, 0.2875], all pass. D2 (2): relative MSE reduction -0.2680 [-0.3902, -0.1459], MSE difference stiff - trust -0.00365 [-0.00641, -0.00090], fail; delusional 0 vs 0, fail at floor. **D2 FAIL.** All 18 rows agree with `r6/report/verdicts.csv` and with the verdict table in `r6/RESULTS-R6-E3.md`. The only numeric difference is the lower bound of the ataraxos_mmd KL interval: 0.014250 rounds to 0.0142 with t = 4.303 and to 0.0143 with the executor's t = 4.30265. |
| E. Independent CPU recomputation from checkpoints | **pass** | Own code `audit_e.py`; it uses only the network class from `r5/model6.py` (`make_net("primary")`). It builds tapes from the npz records, takes the window root as the record of the same game at t0 = max(t-4, t mod 2), keeps its first t - t0 plies, scores only plies with actor == 1 with the policy head at the pre-ply token, and sets every other hidden slot to MASK. All 1,000 calibration and 1,000 test gate positions, all three seeds, 6,093 hidden slots per split (runs 07:08-07:12Z). Against `d0_gate_cal.csv`, per position: kl_anchor max \|diff\| 5.5e-4, 4.0e-4, 4.5e-4 (mean \|diff\| 2.4e-5 to 2.9e-5); kl_routeb_all max 8.6e-4, 7.8e-4, 6.1e-4 (mean \|diff\| 5.5e-5 to 5.9e-5); means agree within 4e-6; correlation 1.0000; no position differs by more than 1e-3. The test split is the same (max 6.9e-4). The +-8 message clamp in `e3lib.py` touches 2 to 7 of 6,093 slots and changes no mean at 6 decimals. The one-pass belief at t (kl_p0) also matches (max 3.9e-4). My own 16-rollout replication (fresh numpy draws, my own cone code) gives cuts of 0.4798, 0.4778, 0.4864, mean 0.4813 [0.4701, 0.4926], against the run's 0.4805, 0.4779, 0.4880; covered share 0.786 vs 0.787. Test: 0.4456 [0.4445, 0.4467] vs 0.4451. The residual differences fit TF32 on the GPU vs FP32 on CPU and 6-digit CSV rounding. |
| F. Code vs pre-registration | **pass** (3 notes) | Marginal messages: other hidden slots MASK, window t0 = max(t-4, t mod 2) (`r5/train6.py` `load_split` asserts that records are stored game by game), opponent's real plies only (`e3lib.py` lines 49, 79); confirmed by E. Cone: 3 hidden slots nearest (Chebyshev) to the target square, ties by distance to the from-square, then slot index (`model6.select_cone`); my own cone code reproduces the coverage. The spec's "strictly within tactical range or nearest" is pre-registered as "nearest 3" with no range limit, so the undeclared-deviation note of AUDIT-R5-E2 does not recur here. Base cover: greedy, shared by both bags with the same samples (`run_e3.py` line 235, then passed to both `skewed_bag` calls). My own greedy cover on the 500 D1 positions covers 0.79262 of hidden slots; the maximum coverable is 0.79291 and is reached in 499 of 500 positions. Skewed bags: n_L ~ Binomial(20, 0.85 / 0.15), drawn from separate pools of 24 + 24 for each policy. Observed means 16.98-17.14 and 2.94-3.12 (expected 17 and 3), variances 2.44-2.75 (expected 2.55). K4: LayerNorm(fingerprint - bag mean) concatenated with the 8-slot salience, linear to 16 dimensions, normalised; L_key on normalised salience; fitted with kc_dedup, then frozen for attn_L3. scaled_mix fitted on unskewed bags (p_L = 0.5). D2: 16 rollouts per legal action (2,361 pairs = sum of n_legal; 37,776 rollouts), depth 4, `real_100.pt`, stiff_voi with the 100% slot-trust head, legal order asserted against the oracle. Strata: equal count (3,777 / 3,778), ties by index. Delusional: argmax of Q_hat, Q_hat > 0.80 and Q*_eff < 0.30. `oracle_d1_test.npz` reproduces bit for bit with the R5 code path (max difference 0, `audit_oracle_d1.json`). Notes: (1) the +-8 clamp on messages is not in the pre-registration; its effect is negligible (E). (2) Key-gap aggregation order (C). (3) The calibration belief loss of the union call is 1.2e-10, 5.6e-7 and 1.0e-6 (`fit.json`), so the model-only fitting reference equals the union call of every training bag. attn_L3 was therefore trained toward an exactly policy-independent target. This was pre-registered, but `RESULTS-R6-E3.md` does not say it (correction 3). |
| G. Compute | **pass** | Own parse of logical-soup `GPU-LOG.md` (about 07:15Z): exactly one VM, soup-gpu-r6e3-main-1017063551-4510, us-central1-a, STANDARD, PENDING-CREATE 06:36Z, CREATE 06:37Z, DELETE 06:54Z, age_h 0.30, so 0.30 h x $0.87/h = **$0.26** (minute bounds 17-19 min give $0.25-$0.28). This equals `r6/GPU-LOG-R6-E3.md` and RESULTS. `gcloud compute instances list` and `gcloud compute disks list --project GCP_PROJECT --filter="name~soup-gpu-r6e3-"` at 07:03:49Z and again at 07:17:21Z: **no instance, no disk** (exit 0, empty). A control query (`name~soup-gpu-`) at the same times listed the live soup-gpu-r6e1-main2 VM and disk, so the query works. |
| H. RESULTS-R6-E3.md | **pass with corrections** | Every rule has a verdict, per-seed numbers and a seed-t interval. For D2 (2) the interval shown is that of the relative change; the pre-registered interval of the MSE difference is only in `verdicts.csv` (correction 6). Numbers carry file paths in the tables and most text. Failures are stated as failures. The post hoc section is labelled. I checked 78 numbers by hand-picked mapping against my own recomputation (`audit_h_numbers.json`): 76 match at the stated precision; 1 is the t-rounding artifact above; 1 is imprecise ("5-12x", correction 2). I reviewed all 78 mappings by hand; my labels agree with the automatic rounding comparison on 77 of 78 (the exception is the t artifact, which I label a match). Post hoc: every number in that section reproduces from my own code (max difference 8e-9 vs `expl_d2.json`). Two claims go beyond `expl_d2.json`: "no weighting of the same rollouts removes that", and the attribution of all non-rollout-noise error to shared bias (correction 4). Other text checks: the self-audit sentence on ties (correction 1); the scaled_mix gate number promised in the pre-registration (section 6 item 1) is missing (correction 5); the driver-file statement (correction 7); "the CPU smoke about 9 min" (smoke files span 06:24:58-06:32:24Z, `done.json` 445 s; correction 9). |

## Corrections (exact suggested wording)

1. **Rounding and ties** (RESULTS lines 187-189). Replace "The CSVs round to 6 significant digits; ties among
   near-terminal entropies can move a recomputed Spearman(-H) in the third decimal." with: "The CSVs round to 6
   significant digits. Recomputed from the CSVs with average ranks, Spearman(-H, -KL) moves by at most 3.5e-7 and
   Spearman(beta, -KL) by at most 2.6e-6. Any ordering of the ties that rounding creates moves Spearman(-H) by less than
   3e-6. The 4,330-4,437 terminal claims per seed share one exact entropy (0.0800525); only ordinal ranks for those
   ties would move Spearman(-H) in the third decimal (about 0.003)."
2. **Policy sensitivity range** (line 69). Replace "The baselines that add or average the duplicates are 5-12x more
   policy-sensitive (JSD 0.0011-0.0024)" with: "The baselines are about 5-10x more policy-sensitive (3-seed mean JSD ratio
   to attn_L3: kc_dedup 4.8x, mix 7.4x, ataraxos_mmd 9.7x, scaled_mix 10.5x; 3.9x to 12.7x across seeds; JSD
   0.0011-0.0024)".
3. **What D1 (2) shows** (after line 68, "...nearly reproduces it (JSD 0.0002, KL 0.102)."). Add: "That was its
   training target. The calibration reference is the union call over the base cover plus 128 reference rollouts. The
   4 greedy base-cover rollouts already reach every hidden slot that any cone can reach (499 of 500 D1 test positions),
   so that reference equals the union call of every training bag (calibration belief loss of the union call 1e-10 to
   1e-6, `r6/runs/s10x/eval/fit.json`). D1 (2) shows that attn_L3 learns to imitate exact deduplication when trained
   toward it, not that it finds deduplication on its own."
4. **Shared bias and reweighting** (TL;DR lines 31-32, post hoc line 154, line 77-78). Replace "The uniform pool's error
   is mostly shared bias (noise is about 0.0022 of its 0.0134 MSE), and no weighting of the same rollouts removes that."
   with: "Of the uniform pool's 0.0134 MSE, about 0.0022 is rollout noise and about 0.0022 is noise in Q* itself (64
   playouts per action); about 0.009 is bias of the depth-4 rollouts against Q*. Trust and entropy reweighting did not
   remove it; no other reweighting was tested." In line 154 replace "Most of the uniform pool's 0.0134 is shared bias of
   the rollouts against Q*." with "Of the uniform pool's 0.0134, about 0.0022 is the sampling noise of Q* (64 playouts)
   and about 0.009 is bias of the rollouts against Q*." In lines 77-78 replace "Reweighting the same rollouts cannot
   remove it" with "Trust and entropy reweighting did not remove it (Q* lies inside the range of the 16 claims for 89% of
   pairs, so the data alone do not show that no weighting could)".
5. **Missing scaled_mix gate number** (pre-registration section 6 item 1 says it is reported). Add a row to the
   secondary table: "gate, scaled_mix with 16 rollouts (reported, not a rule): KL 0.110, cut 40.7% [40.2, 41.3]
   on calibration; 37.6% [37.1, 38.2] on test (`r6/runs/s10x/eval/d0.json`, reduction_scaled16_vs_anchor)".
6. **D2 (2) pre-registered interval** (verdict table, line 53). Append: "; MSE difference stiff - trust -0.0037
   [-0.0064, -0.0009]".
7. **Driver file** (RESULTS lines 180-182 and `r6/PREREG-R6-E3-AMENDMENTS.md`). Replace "`r6/gpu/drive.sh` kept
   byte-identical while its process was running" with: "`r6/gpu/drive.sh` keeps its original inode but was rewritten in
   place at 06:44:50Z (file time), 15 ms after `/tmp/r6e3/drive_v2.sh` was written. The executor states that its content
   is the original. No copy of the launched bytes exists. The driver's output after its polling loop
   (`/tmp/r6e3/launch-r6e3-main.log`) is clean, which is consistent with unchanged bytes."
8. **Window evidence vs the current belief** (lines 63-64). Replace "the window evidence brings no improvement" with
   "the stale anchor plus window evidence stays worse than that belief (0.078 at best vs 0.047); whether the window adds
   anything to the current belief was not tested".
9. **Minor.** Line 172: "the CPU smoke about 9 min" becomes "the CPU smoke about 7.5 min (06:24:58-06:32:24Z)". D1 (1)
   row: add "per position the mean of the two policies' gaps, then over positions; the pre-registration's order gives
   0.2274 [0.2107, 0.2441]; 3, 5 and 3 policy bags per seed with rollouts on one flank only are skipped". Deviations
   log: add "the smoke also showed a key gap of 0.054 and trust_pool 29% worse than stiff_pool on 12 positions; no code
   changed after it (e3lib.py 06:22:57Z, run_e3.py 06:24:34Z, smoke 06:24:58-06:32:24Z)" and "report_e3.py: label
   change at 06:37:17Z after a dry run on the smoke output (`||` to `‖`); no rule logic changed".

## Auditor observations (post hoc, after the verdicts; not rule tests)

From `audit_h_posthoc.py` on the synced CSVs and `oracle_q_test.npz`, 3-seed means [seed-t 95%]:

- Q* is a 64-playout estimate (all Q*_eff values are multiples of 1/128). Its plug-in sampling variance is 0.0022 per
  pair, the same size as the uniform pool's rollout-noise term. That is an MSE floor shared by every pool.
- Q*_eff lies within the range of its 16 rollout claims for 0.893 [0.859, 0.927] of pairs. A convex weighting that
  knows Q* would reach MSE 0.0008. This is not a feasible pool; it only shows that the data do not prove "no weighting
  removes the bias".
- beta also ranks rollouts by the error of the pooled quantity: Spearman(beta, -abs(e_r - Q*_eff)) is +0.18 [0.16, 0.20]
  overall and +0.18 [0.08, 0.29] within pairs (for -H: -0.19 overall). So the gap between good ranking and worse pooling
  is not only KL vs value error. The ESS explanation in RESULTS covers about half of it.

## What I could not check

- GPU-side quantities beyond the gate: the fitted K4 keys and attn_L3 weights were not synced (CSV/JSON only), and the
  rollouts depend on the run's CUDA random streams. D1 and D2 are re-derived from the synced CSVs, not recomputed from
  the checkpoints.
- The exact random root moves of the gate (CUDA generator). I replicated the gate with my own draws instead (E).
- The launched bytes of `r6/gpu/drive.sh` (see correction 7).
- The executor's checks for `outbox/experiments-round-6-amendments*.md` at 06:34Z, 06:35Z and 06:54Z. I can only confirm
  that none exists at 07:13Z (`ls /home/agent/workspaces/mats/outbox/`).
- Real billing. $0.87/h is the unverified on-demand list-price ceiling (Billing API disabled, per GPU-LOG); I saw no
  invoice.
- Whether smoke attempts ran before the final code. Only the last smoke output survives.
- Oracle correctness beyond reproducibility. `oracle_d1_test.npz` reproduces exactly with the R5 code path. I did not write
  an independent exact-Bayes oracle, and I did not recompute Q* or the R5-E2 D0 oracles (R5-E2 records a brute-force
  check of 6.7e-16 in `meta.json`).
- That the VM ran exactly `/tmp/r6e3/src`. I checked the staged copy against the local code; the transfer is from
  `gpu-run.sh` (rsync of GPU_SOURCE_DIR) and the launch log; the VM is deleted.

## Commands run (all read-only outside the auditor scratch)

- `sha256sum r6/PREREG-R6-E3.md outbox/experiments-round-6-spec.md`; `cat r6/PREREG-R6-E3.sha256`; `stat` and
  `ls --time-style=full-iso` on `r6/`, `r6/gpu/`, `r6/runs/*/eval/`, `r6/report/`, `r5/`, `/tmp/r6e3/`, `/tmp/r5e2/`.
- Section-1 comparison: `awk` extracts of spec section 3.3 and prereg section 1 into `spec_33.txt` / `prereg_s1.txt`,
  then `diff` and a Python byte comparison; `cmp /tmp/r6e3/spec-3.3-verbatim.md spec_33.txt`.
- `grep -n 'r6e3' .../logical-soup/GPU-LOG.md`; own parse of the same file for the ledger.
- `.pyc` header read (Python `struct`) for `r6/__pycache__/e3lib` and `report_e3`; `cmp` of `/tmp/r6e3/src/{r5,r6,gpu}`
  against the local files; `ls -li r6/gpu/`.
- `grep -n -i 'oracle\|pstar\|qstar\|np.load\|slot_true\|own_true'` over `r6/e3lib.py r6/run_e3.py r5/train6.py
  r5/model6.py`; code reading of `r6/*.py`, `r6/gpu/*.sh`, `r5/model6.py`, `r5/train6.py`, `r5/eval6.py`,
  `r5/generate6.py`, `r5/msgame6.py`.
- `python3 audit_cd.py` (checks C and D) -> `audit_cd.json`; `python3 audit_ties.py` -> `audit_ties.json`.
- `python3 audit_e.py SEED 1000 {cal,test}` for seeds 101-103 (check E) -> `audit_e_{cal,test}_s10x_n1000.json`.
- `python3 audit_f.py` -> `audit_f.json`; `python3 audit_oracle_d1.py` (4 workers, in memory) -> `audit_oracle_d1.json`.
- `python3 audit_h_posthoc.py` -> `audit_h_posthoc.json`; `python3 audit_h_numbers.py` -> `audit_h_numbers.json`.
- `gcloud compute instances list` and `gcloud compute disks list` with `--project GCP_PROJECT
  --filter="name~soup-gpu-r6e3-"` at 07:03:49Z and 07:17:21Z, plus control queries with `name~soup-gpu-`.
- Auditor scratch: `/home/agent/workspaces/fable-strategy/scratch/audit-r6e3/` (scripts, JSON outputs, logs).
