# AUDIT R9-E1 (E-INFO-STRONG-ROOT)

Independent audit by a board-investigator subagent (Claude Opus 5.5, model id claude-opus-5-5[1m]); I did not run R9-E1. Read-only
on li-net, /tmp/r9e1 and the run's logs; CPU only, at most 2 threads, no GPU. My role may write only under
`fable-strategy/scratch/`, so this file is `/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/AUDIT-R9-E1.md` and
my scripts and outputs are in `/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/audit/` (to be installed as
`li-net/r9e1/AUDIT-R9-E1.md` and `li-net/r9e1/audit/`). Audit window 2026-10-17T20:04Z-20:34Z. Audited write-up:
`RESULTS-R9-E1.md` sha256 12008db5...ceb6 (completeness marker on line 351, seen 20:14:36Z; file unchanged at 20:33:12Z).

## Overall verdict

**CONFIRMED, with changes to RESULTS.** The run is clean and every rule verdict is right. RESULTS needs 8 small edits; one adds a
material disclosure.

- Integrity: registration (19:16:38Z) precedes the first launch-log line (19:16:46Z) and the first `soup-gpu-r9e1-` GPU-LOG line
  (19:40Z). All registered and staged hashes match. The only change after registration is deviation 1, which is logged
  (`chmod +x`, contents unchanged). The registered `analyze9.py` and `numbers9.py` reproduce `verdicts.md`, `numbers.txt`, the
  CSVs and `results.json` exactly.
- No oracle in training or selection. No world-model training. The amended items and the approved readings are implemented as
  the PREREG states.
- Numbers re-derived from `raw.json` with my own code: the largest gap to `summary.json` is 2.7e-4 (on a number no rule uses).
  On the brief's headline numbers it is 8.7e-5. All 348 per-seed KLs lie inside the interval that 5-decimal rounding allows.
- Rules re-decided from my numbers give the same verdicts as `verdicts.md` and RESULTS: E1-D0 PASS, E1-D1.1 PASS, E1-D1.2 PASS
  (E1-D1 PASS), E1-D2.1 FAIL on two clauses, E1-D2.2 PASS (E1-D2 FAIL).
- Sensitivity that RESULTS does not state: the E1-D1.2 PASS rests on approved reading 1. Under the spec's literal selection
  order, M_0_strong would be 34 in every seed, and E1-D1.2 fails at M_0 = 34.
- Cost: $0.31 per GPU-LOG; at most $0.33 from the GCE operation times; cap $0.60. VM deletion began 7 s after the run ended and
  finished 57 s after it. No r9e1 VM was alive at 20:09:08Z, 20:28:41Z or 20:33:14Z.
- RESULTS: all 312 table cells match. One prose number is double-rounded (jacobi_lin_T4_prec a, seed 103: 2.20 printed, 2.19
  in the file). A few sentences claim more than the numbers show, or need a scope.

## Check 1 - timing and integrity: PASS

Evidence: `audit/check1.py` -> `check1.txt` (run 20:28:22Z), `audit/anarepro.sh` -> `anarepro.txt` (20:27:23Z).

- **Registration time.** Both `.sha256` files say "registered 2026-10-17T19:16:38Z" (mtimes 19:16:38.552Z and 19:16:38.554Z).
  - The first launch-log line is `2026-10-17T19:16:46Z LAUNCH r9e1-main try 1` (`/tmp/r9e1/logs/launch-r9e1-main.log`). That try
    stopped locally with no VM.
  - The first `soup-gpu-r9e1-` line in `logical-soup/GPU-LOG.md` is line 2170, `2026-10-17T19:40Z PENDING-CREATE`.
  - Registration precedes both.
- **Registered hashes.** `sha256(results/PREREG-R9-E1-as-registered.md)` = 3a1db6d1...de11, equal to `PREREG-R9-E1.sha256`.
  `PREREG-R9-E1-AMENDMENTS.md` (current and as registered) = 232df930...9f50, equal to its `.sha256`.
- **PREREG differs only in section 5.** Lines 1-248 (through the section-5 heading) and the final `Registered:` line are
  identical. The registered body of section 5 was the placeholder "(none yet)"; it is now deviation 1 (9 lines). This is a
  replacement of the placeholder rather than a strict byte-append, but nothing outside section 5 changed.
- **Staged hashes.** `results/stage-manifests.txt` holds 4 identical blocks: 19:16:23Z (registration staging), then the launch
  tries at 19:16:46Z, 19:19:14Z and 19:40:18Z.
  - All 23 files (code, imports, `smallnet/netdef.py`, 3 checkpoints) equal the registered hashes: full hashes for 4 files,
    8-hex prefixes for 19.
  - Current li-net files equal the staged hashes, and `/tmp/r9e1/src` still passes `sha256sum -c MANIFEST.sha256`.
  - `gpu-run.sh` rsyncs `${GPU_SOURCE_DIR}` = `/tmp/r9e1/src` (its line 78). Its log line prints the logical-soup path (line
    75), which is a cosmetic quirk of the infrastructure.
- **Files changed after registration** (mtime or ctime after 19:16:38Z; `results/` excluded):
  - `PREREG-R9-E1.md` (19:19:10Z): deviation 1.
  - chmod of `driver9.sh`, `launch9.sh`, `remote9.sh` and `stage9.sh`: ctime 19:18:57.8Z; deviation 1 gives 19:19:05Z. Their
    contents are unchanged: the hashes match in all 4 manifest blocks.
  - `GPU-LOG.md` (20:03:47Z): ledger output.
  - `RESULTS-R9-E1.md` and `site-entry.json` (20:09Z): the write-up.
  - No code, rule or analysis file changed.
- **Spec and amendments.**
  - Spec sha256 28778d60... (mtime 18:31Z) and amendments sha256 87a336c0... (19:00:19Z) equal the PREREG's values.
  - Spec lines 36-41 (b9896435...) and 73-98 (20f8a0ae...) and amendments Part B lines 43-53 (49db9e10...) all match, and all
    appear verbatim in the registered PREREG. Part B also appears verbatim in the AMENDMENTS file.
  - Order of events: folded in at 19:07:09Z, registration at 19:16:38Z, launch at 19:40:18Z. `e1v6.py` and `test_r9e1.py` were
    last modified at 19:06:06Z.
  - No other round-9 amendments file exists (outbox listing, 20:28Z).
- **Analysis outputs are the unedited output of the registered scripts.** I re-ran hash-checked copies of `analyze9.py` and
  `numbers9.py` on copies of the three `summary.json` files. The result is byte-identical `verdicts.md`, `numbers.txt`,
  `anytime_curves.csv`, `d3_weights.csv` and `d3_mse.csv`, and a JSON-equal `results.json`.

## Check 2 - no oracle in training or selection: PASS

I read all 719 lines of `e1v6.py` and the functions it calls in `e1v5.py`, `e1v4.py`, `e1v3.py`, `rollouts.py`, `partA2.py` and
`wm.py`.

- **Exact Bayes.** `env6.bayes_marginals` is called only in `test9` (`e1v6.py` lines 549-550, `Pfull` and `Pstd6`), for test
  KL/MSE and the clause-1 Pearson.
  - The other callers in the imported modules are their own test functions and world-model training monitors. `e1v6` calls
    none of them.
  - The registered unit test runs a full tiny `fit9` with `bayes_marginals` replaced by a function that raises. On a
    hash-checked copy it gave 10 passed in 12.5 s (CPU, 20:15Z).
- **Strong-root selection.** It uses the 4,000 calibration boards (stream 21).
  - ref(36) is theta_real on all 36 sonar readings plus the note (line 174), not the occupancy.
  - The selection KL is KL(ref36 || root) (line 178). `select_strong` (line 70) takes the smallest of 30/32/34 with KL <= 0.035,
    else 34.
  - The selection is made in `fit9`, before `test9` exists.
- **Fit targets are the model's own references.**
  - Anytime pools: BCE to ref(M_0), the union call from the M_0 root (lines 213 and 284).
  - R8's pools: BCE to ref (line 325).
  - Scalars (eta, u, econ, alpha): BCE to ref (line 357).
  - econ_kc_K3 (D3 fit) and econ_prec_K3: BCE to `ref[b]` on the D3 calibration probes (lines 409 and 157).
  - jacobi_lin_T4_prec: standard-bag BCE to ref, P-FullSupport BCE to `ref[:2000]` (line 143) and D3 MSE to ref (lines 444-451).
  - theta_trust: MSE to `RO.es_target`, the real excess surprise against calibration-board occupancy (lines 223 and 241).
- **No world-model training.** `E5.load_checked` checks the npz sha256 against the json (line 697). The model's parameters are
  frozen (`requires_grad_(False)`, lines 699-700), and no optimizer holds them. Reads and union calls run under `torch.no_grad`.
- **Amended items.**
  - `econ_prec_lambda` returns `F.softplus(a - b * beta)` (line 81). b = softplus(b_raw), u = exp(.), with init u = 1,
    a = log(e - 1) and b = 0.1 (lines 146-152). The fit is 1,500 Adam steps at lr 0.02, BCE on the D3 probes (first 80% of 2,000
    boards).
  - jacobi_lin_T4_prec's loss is `0.5 x std BCE + 0.5 x PFS BCE + 0.25 x D3 MSE` (line 451): 6,000 steps, lr 1e-3, a batch of
    256 of each kind, gradient clip 1.0.
  - The three seeds' configs are identical: std_w 0.5, pfs_w 0.5, d3_w 0.25, fit_steps 1500, agg_steps 6000, agg_batch 256,
    n_cal 4000, n_any 2000, n_d3 2000, n_pfs 2000, n_test 500 (`audit/check3_extra.txt`).
- **Approved readings.**
  - Mode A reads from EV[M] with `draw_tau3` (lines 614-615).
  - Mode B reads from EV[2] and L0[2] with `RO.draw_tau` (tau 0.3/1.5), pooled on l0 = L0[M] (lines 618-621).
  - Anytime training (`any_bag9`, lines 127-134): Mode B only at M_0 >= 6 with p 0.5, anchor l0(M_0),
    beta_0 = M_0 beta_clean / 6, target ref(M_0).
- **Disclosed researcher choice, not an oracle.** Reading 1 (reference-first KL) was chosen after a calibration-only CPU check
  showed it selects 32, while the spec's literal order selects 34. No test number existed then, and gemini approved the reading
  before launch. The choice decides E1-D1.2 (check 5).

## Check 3 - anchor token and Mode B: PASS

Evidence: `audit/rederive.py` -> `rederive.txt` and `rederive_out.json`; `audit/check3_extra.txt`.

- **K = 0 identity.**
  - `summary.json` `identity_K0` is 0.0 (KL difference and logit difference) for both anytime pools at every M_0, including 2
    and 32, in all seeds.
  - From raw, the K = 0 beliefs of looped_attn and attn_K3 (and of mmd, mix and scaled_mix) equal the root beliefs exactly (max
    difference 0.0). This holds on the sparse stream (M_0 = 2) and in both modes at M_0 = 32; the KL difference is 0.0.
- **beta_0.** `Pools8.beta0` returns M beta_clean / 6 (`e1v5.py` lines 218-219), and training uses
  `M0T[Mi] * beta_clean / 6` (`e1v6.py` line 134). beta_clean is the mean theta_trust beta of tau-0.3 standard reads
  (3.496 / 3.724 / 3.340).
- **Mode B formula.**
  - In code, `bag_reads3` sets dl = lr - l0 with l0 = L0[2]. `P.all(l0 = L0[32])` then returns l0(32) plus the weighted shifts
    for every pool, and mix = `agg_mix(lr)`, the mean of the reads' beliefs (`e1v5.py` line 236). All pools clamp Delta l to
    +-8, as in R6-R8.
  - Numerical check from raw at K = 1 and 2:
    - Mode B scaled_mix equals sigmoid(l0(32) + alpha * sum clamp(l_r - l0(2))) to within 1.4e-5 to 4.2e-5 (5-decimal rounding).
      With the token l_r - l0(32) it would be off by 0.19-0.37.
    - Mode A matches the token l_r - l0(32) to within 2.4e-5; the other token would be off by 0.16-0.31.
    - The implied per-cell weights of looped_attn, attn_K3 and mmd at K = 1 are never negative (share 0.0) in either mode.
    - mmd's implied weight is one scalar per board (median within-board CV <= 0.0011), with implied beta_1 2.60-2.90 in Mode B
      and 2.87-3.27 in Mode A.
- **Same patches in both modes.** The Mode A and Mode B patch arrays are identical in every seed and half.
  - Tau shares: Mode A 0.33 / 0.33 / 0.34 over 0.3 / 1.5 / 2.5; Mode B 0.49-0.51 over 0.3 / 1.5 only.
  - Stream structure: the sparse stream has the 6 partition tiles in slots 0, 4, ..., 20, then 2 repeats.
  - Skew: about 87% of the skewed reads are NW-only in the U_NW bag. The D3 background patches never overlap A*.

## Check 4 - headline numbers re-derived from raw.json: PASS (largest gap 2.7e-4, from rounding)

Evidence: `audit/rederive.py` (json and numpy only; KL with q clamped to [1e-6, 1 - 1e-6], as in the registered metric), run
20:28Z, giving `rederive.txt` and `rederive_out.json`; `audit/headline_diffs.py` and `audit/rounding_bounds.py`.

- **Scope.** I compared 1,125 per-seed numbers with `summary.json` (678 of them on the brief's headline list) and 331 three-seed
  means with `analysis/results.json`.
- **Largest |mine - summary.json|.**
  - Overall: 2.74e-4, s103 sparse-root ataraxos_mmd KL at K = 32 (0.163325 vs 0.163051; no rule uses it).
  - Headline list: 8.69e-5, s101 Mode B scaled_mix KL at K = 24 (0.135889 vs 0.135802).
  - By category: root KL 2.2e-7, sparse looped_attn KL 2.2e-7, JSD(24) 6.8e-9, the five K = 24 KLs 1.5e-5, Mode A curves 1.3e-6,
    Mode B curves 8.7e-5, W(N) 4.8e-7, D3 MSE 2.8e-8.
  - Three-seed means against `results.json`: 2.16e-4 overall, 6.1e-5 on the headline list.
- **All gaps are rounding.** All 348 per-seed KL numbers lie inside the interval that 5-decimal rounding of the pool beliefs and
  exact Bayes allows (`rounding_bounds.txt`). The largest gaps come from saturated mmd and scaled_mix beliefs (rounded to 0 or 1)
  at large K.
- **My three-seed means.**
  - Root KL by M_0: 2: 0.27966; 6: 0.24520; 12: 0.19299; 18: 0.14444; 30: 0.04618; 32: 0.03219; 34: 0.01687.
  - Sparse-root looped_attn KL at K = 0 / 1 / 2 / 4 / 8 / 16 / 24 / 32: 0.27966 / 0.23087 / 0.19181 / 0.15309 / 0.10655 /
    0.05833 / 0.03261 / 0.03231. JSD(24) 0.00699 [0.00636, 0.00761].
  - KL at K = 24: jacobi_prec 0.03490, looped_lin 0.06642, kc_K1 0.09502, kc_K3 0.06468, econ_prec 0.04405.
  - M_0 = 32 curves: Mode A looped_attn goes from 0.03219 (K = 0) to 0.01328 (K = 24). Mode B at K = 24: looped_attn 0.02884,
    attn_K3 0.02889, mmd 0.12611, mix 0.20763, scaled_mix 0.13635.
  - W(16)/W(1) at tau 0.3: econ_prec 1.2668, jacobi_prec 1.1085.
  - D3 MSE over all tiers: econ_prec 0.038178, econ_D3 0.038269, jacobi_prec 0.038339, kc_K3 0.039815.

## Check 5 - every rule re-decided (PREREG section 3, my numbers): same verdicts

| rule | mine | `verdicts.md` | RESULTS | decisive numbers (mine; 3-seed mean [seed-t 95% CI]) |
|---|---|---|---|---|
| E1-D0 | PASS | PASS | PASS | each npz sha256 (li-net copy and staged copy) equals its `.json` and `summary.json`; Pearson 0.99946 (from `summary.json`; cannot be re-derived from raw); K = 0 difference 0.0 |
| E1-D1.1 | PASS | PASS | PASS | largest step +0.000155 (s101, K 24 -> 32); KL(32) 0.0323 [0.0298, 0.0348]; JSD(24) 0.00699 |
| E1-D1.2 | PASS | PASS | PASS | KL(0) 0.0322; Mode B KL(16) - KL(0): mix +0.1761 [0.1755, 0.1767], mmd +0.0470 [0.0386, 0.0554]; looped_attn largest rise per seed, A -0.00325 / -0.00294 / -0.00298, B -0.00048 / +0.00003 / -0.00041; KL(0) - KL(16): A 0.0170 [0.0159, 0.0181], B 0.0030 [0.0027, 0.0033]; all 8 "beats" CIs > 0 (smallest lower bound 0.0027) |
| E1-D2.1 | FAIL | FAIL | FAIL | W(16)/W(1) at tau 0.3 1.2668 [1.2366, 1.2970], above 1.25 in every seed (1.2546 / 1.2667 / 1.2789); MSE minus econ_kc_K3 (D3 fit) 0.000091 [-0.000090, 0.000273]; passing parts: CI upper bound 1.297, monotone in every seed, 2.659 at tau 2.5, MSE minus kc_K3 0.001637 [0.001357, 0.001916] |
| E1-D2.2 | PASS | PASS | PASS | W(16)/W(1) 1.1085 and 2.515 [2.335, 2.694]; monotone in every seed; P-FullSupport KL 0.0349 [0.0343, 0.0355]; 4 paired CIs > 0 (smallest lower bound 0.0080) |
| E1-D1 / E1-D2 | PASS / FAIL | same | same | |

Using t = 4.303 (registered) or 4.302653 (`analyze9.py`) changes no part.

**I would decide no rule differently.** Sensitivities, none of which overrides the registered readings:

1. **E1-D1.2 under the spec's literal selection order.** KL(root || ref36) at M_0 = 32 is 0.0354 / 0.0353 / 0.0355, above 0.035,
   so M_0_strong would be 34 in every seed. At M_0 = 34, E1-D1.2 FAILS on two parts (`audit/check8_extra.txt`):
   - Mode B never-degrade: looped_attn_T3 rises +0.00062 in seed 102 (bar +0.0005).
   - Mode B KL(0) - KL(16) = 0.00029 [-0.00045, 0.00103]; the CI is not above 0.
   At M_0 = 30 it would pass.
2. **E1-D1.1 read per seed.** If KL(32) <= 0.033 had to hold in every seed, seed 101 (0.0332) would fail. The registered rule
   uses the mean.
3. **E1-D2.1 does not depend on the comparator.** Clause 4 also fails against the other 2-scalar fit, econ_kc_K3 R7-style
   (0.00012 [-0.00006, 0.00030]). Clause 1 fails in every seed. The FAIL is robust.

## Check 6 - parameter claims and econ_prec_K3's closed form: PASS

Evidence: `audit/check6_params.py` -> `check6_params.txt`; `rederive.txt`.

- **Parameter counts.** Hand counts from the class definitions equal the counts of the instantiated registered classes and
  `summary.json`:

| pool | parameters | of which K-key head |
|---|---|---|
| looped_attn_T3 | 39,428 | 592 (K3) |
| attn_L3-K3 | 106,180 | 592 (K3) |
| attn_L3-K1 | 107,652 | K1 head |
| looped_lin_T4 | 2,789 | K4 head |
| jacobi_lin_T4 | 812 | 592 (K3) |
| jacobi_lin_T4_prec | 814 (222 + 592) | 592 (K3) |
| econ_prec_K3 | 3 (u, a, b) | none counted |

  econ_prec_K3 also uses attn_L3-K3's 592-parameter K3 key head, which is not counted, as for econ_kc_K3.
- **Closed form on identical reads.** At tau 0.3 the N reads of A* are identical, so
  W(N) = N u beta / (lambda(beta) + N beta + C_bg), and N / W(N) must be linear in N with slope 1/u on each board.
  - Over 500 test boards x 3 seeds, the median slope is 0.898276 / 0.881570 / 0.880841, against 1/u = 0.898279 / 0.881573 /
    0.880829.
  - The largest per-board slope deviation is 0.00093; the largest residual is 4.8e-5. The implemented weights follow the
    registered formula exactly.
  - My implementation and `e1v6.econ_prec_weights` agree on a synthetic bag of 16 identical reads (W(16) = 1.10466 / 1.12292 /
    1.12238).
- **Auditor's calculation (descriptive).** Without background reads, the fitted scalars at the calibration tier-mean beta give
  W(16)/W(1) = 1.116 / 1.151 / 1.171 at tau 0.3, under the 1.25 bar. On identical tau-2.5 reads they give 1.741 / 1.787 / 1.723.
  - The per-board intercept (lambda + C_bg)/(u beta) has median 0.243 / 0.246 / 0.263, against lambda/(u beta) =
    0.112 / 0.143 / 0.162 without background.
  - So the 8 background reads' crowding of A*'s reads is what lifts the measured clean ratio to 1.255-1.279. The test reads'
    betas are not saved, so I used the calibration means.

## Check 7 - cost, teardown and live VMs: PASS

- **GPU-LOG.** `logical-soup/GPU-LOG.md` lines 2170-2176 hold one VM, `soup-gpu-r9e1-main-1017194018-32101` (spot g2-standard-8 +
  L4): CREATE 19:40Z, RUN-END 20:02Z rc=0, DELETE 20:03Z, age_h 0.36, cost_usd 0.31 (the $0.87/h list-price ceiling x age). There
  are no other r9e1 lines; the two earlier tries created no VM.
- **GCE operations** (`gcloud compute operations list --filter='targetLink~soup-gpu-r9e1'`, 20:09:40Z): insert
  19:40:28.4Z-19:40:36.7Z; delete 20:02:22.1Z-20:03:12.6Z.
  - Billed life is at most 22 min 44 s (0.379 h), or $0.33 at the ceiling rate. The GPU-LOG's 0.36 h gives $0.31.
  - Both are under the $0.60 cap and the PREREG's $0.49 worst case. Group r9: R9-E2 $0.36 + R9-E1 $0.31 = $0.67 of $1.35.
- **Teardown speed.** The last seed's `rc.json` was written at 20:02:12.5Z, and `driver.txt` logs remote rc=0 at 20:02:15Z.
  Deletion started 7 s later and finished at 20:03:12.6Z, 57 s after the run ended.
- **Live VMs.** `gcloud --project=GCP_PROJECT compute instances list --filter='name~r9e1' --format='value(name,status)'` returned
  nothing at 20:09:08Z, 20:28:41Z and 20:33:14Z. There was no soup-gpu VM at all at 20:09Z.

## Check 8 - RESULTS-R9-E1.md read critically: mostly sound; 8 requested changes

Evidence: `audit/check8_tables.py`, `check8_extra.py`, `check8_prose_tokens.py`, `check8_fullprec.py` and their `.txt` outputs.

- **Tables.** All 312 cells match `analysis/results.json` at the printed precision: section 4 (96), section 5 (132) and section 6
  (84, CIs included).
- **Prose numbers.**
  - Automatic screen: 374 decimal tokens in the prose, looked up as strings in the files RESULTS cites. 363 were found verbatim
    and 11 were not.
  - I checked all 11 by hand against `summary.json` and `results.json`. All are correct at their printed precision; they are
    single roundings of JSON values.
  - I hand-labelled a seeded sample of 30 found tokens: 30 of 30 are correct for the quantity they name.
  - The screen has a known false positive: line 266's "2.20" is found verbatim (as the 2.20 bar) but is wrong as a value.
  - A double-rounding screen raised 54 candidates. I checked every measured one at full precision (`check8_fullprec.txt`).
  - **One mismatch:** jacobi_lin_T4_prec a for seed 103 is printed as 2.20, but `summary.json` has 2.19467. `numbers.txt` prints
    2.195, which RESULTS rounded a second time.
- **Failures are reported as failures.** The TL;DR leads with the E1-D2 FAIL and E1-D2.1's two failing clauses. Seed 102's
  +0.00003 (Mode B, K = 1) and the M_0 = 34 exceedance (+0.00062) are both reported.
- **Labels and claims.**
  - Interpretations are labelled ("interpretation, not tested", "descriptive", "reported").
  - The caveat that the anchor pools were trained on both read modes is stated, and so is "No ablation separates the anchor token
    from the Mode B training bags".
  - The exceptions are changes 1 and 3-8 below.
- **Timeline and cost.** These match the launch log, `rc.json`, `driver.txt` and the GCE operations. Section 9 names the gcloud
  check and its time.
  - Seconds-level nits, no change requested: deviation 1's 19:19:05Z against the chmod ctime of 19:18:57.8Z.
  - "20:03:28Z scoring" against `results.json` written at 20:03:26.3Z (plots 20:03:28.6Z, `numbers.txt` 20:03:40.6Z).

## What I could not check

- **Real billing.** The Billing API is disabled, so costs are list-price ceilings; the spot price is lower.
- **Pre-registration CPU work.** The 18:38-18:40Z calibration check and the 18:51:53-18:52:55Z real-checkpoint smoke left no logs
  in `/tmp/r9e1`. The untrained smokes are there: checkpoints 88f7d1b7 / 85e548fd / 6718304e, marked "UNTRAINED random init".
  I did not read the agent-mail messages (#3753, #3758, #3762, #3765).
- **Code executed on the VM.** No VM-side hashes were recorded. The evidence is the staged manifest, gpu-run.sh's rsync of
  `GPU_SOURCE_DIR`, and the run's own log lines for the checkpoint sha256.
- **Clause-1 Pearson and the logit-level K = 0 identity.** Neither is in `raw.json`; I took them from `summary.json`.
- **Trained pools.** Per-read betas, keys and weights and the trained pool parameters were not saved (the run synced JSON only),
  so I could not re-evaluate pools on CPU. I verified the Mode A and Mode B structure through the K = 1-2 arithmetic only. I did
  not reproduce the GPU numerics (TF32).

## Requested changes (RESULTS-R9-E1.md; line numbers at sha256 12008db5...)

1. **Disclose that E1-D1.2 depends on reading 1** (section 3, line 112).
   - Quote: "M_0_strong = 32 in every seed. The other order, KL(root || ref(36)), is 0.0354 / 0.0353 / 0.0355 at M_0 = 32
     (reported)."
   - Fix: append "Under that order, the spec's literal KL_cal(root || ref(36)), every seed is above 0.035 at M_0 = 32, so
     M_0_strong would be 34. At M_0 = 34, E1-D1.2 fails two parts in Mode B: looped_attn_T3 rises +0.00062 in seed 102 (bar
     +0.0005), and KL(0) - KL(16) = 0.0003 [-0.0004, 0.0010]. The PASS therefore rests on reading 1 (reference first), chosen on
     calibration boards only and approved before launch."
   - Also add a pointer in the TL;DR's E1-D1.2 row.
2. **Fix the double rounding** (section 7, line 266).
   - Quote: "a = 2.70 / 2.55 / 2.20"
   - Fix: "a = 2.70 / 2.55 / 2.19" (`fit.103.jacobi_prec.a` = 2.19467; 2.20 is `numbers.txt`'s 2.195 rounded a second time).
3. **Scope the Mode A comparison** (section 1, lines 44-45).
   - Quote: "The anchor pools gained slightly more than the other learned pools (-0.0170 at K = 16)."
   - Fix: "At K = 24 the anchor pools end below every other learned pool (by 0.0004-0.0018 against econ_prec_K3, both econ_kc_K3
     fits, kc_dedup K3 and the two Jacobi pools; paired CIs > 0). At K = 16 their margin over econ_prec_K3 and the econ_kc_K3
     fits is 0.0002-0.0003, and 5 of those 6 paired CIs include 0 (descriptive, `summary.json`)."
4. **Correct an untested causal claim** (section 2, lines 64-65).
   - Quote: "R8-E1's 18-anchor root sat at KL 0.144, too weak for naive pools to degrade it."
   - Fix: "R8-E1's 18-anchor root sat at KL 0.144, and with R8's cumulative-context reads even ataraxos_mmd improved it."
   - Optionally add: "In this run at M_0 = 18, bounded Mode B reads let mix degrade that root at once (+0.0952 at K = 1), and mmd
     ends above it (+0.0031 at K = 24); descriptive."
5. **Correct the Mode B description** (section 2, line 83).
   - Quote: "Every pool uses its standard-root fit; mix averages the reads' beliefs."
   - Fix: "Every pool except the two anchor-aware pools (trained over the 7 root strengths with half Mode A and half Mode B bags)
     uses its standard-root fit; mix averages the reads' beliefs."
6. **Scope the ridge claim** (section 6, line 248).
   - Quote: "The amendment removed the 2.0 ceiling, but the fit stayed below it."
   - Fix: "The amendment removed the 2.0 ceiling. At each tier's mean beta the fitted ridge stays below 2. It exceeds 2 only for
     reads with beta below 0.82 (seed 101) or 0.78 (seed 102), where lambda(0) = 2.68 and 2.48; how many test reads have such a
     low beta is not saved."
7. **Add the evidence and a label to the ridge comparison** (section 6, lines 250-251).
   - Quote: "The precision-dependent ridge takes clean deduplication from 1.432 to 1.267 and keeps the noisy rise (2.659)."
   - Fix: append "(same run, same keys and D3 probes. Paired difference in W(16)/W(1) at tau 0.3: 0.166 [0.155, 0.176]. At tau 2.5,
     econ_prec_K3 minus econ_kc_K3 = +0.073 [-0.152, 0.297]. Descriptive, not a registered test.)"
   - In section 11, line 327 ("**Mechanisms.** A precision-dependent ridge brings the 1-step econ pool most of the way to clean
     deduplication ..."), add "(descriptive: a same-run comparison with econ_kc_K3, not an ablation)".
8. **Scope the Mode B safety summary** (section 11, lines 322-323).
   - Quote: "The anchor-aware attention pools, trained on such bags, stayed just below the root."
   - Fix: "... stayed just below the root in the 3-seed means (seed 102: looped_attn_T3 +0.00003 above it at K = 1)."

Optional:

9. **Note the background-read effect** (section 11, line 334).
   - Quote: "econ_prec_K3 fitted with a clean-read bag term, to see whether its 0.017 gap to the 1.25 bar is a fit issue."
   - Add: "Auditor's calculation: on identical clean reads with no background reads, the fitted scalars give W(16)/W(1) =
     1.116 / 1.151 / 1.171. The D3 bags' 8 background reads raise each board's intercept (lambda + C_bg)/(u beta) 1.6-2.2-fold,
     which is what lifts the measured ratio to 1.255-1.279. The gap may come from background crowding rather than the fit."
10. **Flag the per-seed KL(32)** (section 4, line 140).
    - Quote: "KL(32) = 0.0323 [0.0298, 0.0348] (per seed 0.0332 / 0.0325 / 0.0312; bar 0.033 on the mean)."
    - Add: "Seed 101 alone is above 0.033; the PASS uses the registered mean reading."

## Files

- This audit: `/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/AUDIT-R9-E1.md`.
- Scripts and outputs, all in `/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/audit/`:

| purpose | script | output |
|---|---|---|
| check 1 | `check1.py` | `check1.txt` |
| analysis reproduction | `anarepro.sh` | `anarepro.txt` |
| check 4 re-derivation | `rederive.py` | `rederive.txt`, `rederive_out.json` |
| check 4 headline gaps | `headline_diffs.py` | `headline_diffs.txt` |
| check 4 rounding intervals | `rounding_bounds.py` | `rounding_bounds.txt`, `rounding_bounds_out.json` |
| checks 2-3 | `check3_extra.py` | `check3_extra.txt` |
| check 6 | `check6_params.py` | `check6_params.txt` |
| check 8 tables | `check8_tables.py` | `check8_tables.txt` |
| checks 5 and 8 | `check8_extra.py` | `check8_extra.txt` |
| check 8 prose | `check8_prose_tokens.py` | `check8_prose_tokens.txt` |
| check 8 full precision | `check8_fullprec.py` | `check8_fullprec.txt` |

- `check3_extra.py` and `check6_params.py` import from `audit/tmpsrc`, a hash-checked copy of `/tmp/r9e1/src/*.py` and
  `smallnet/netdef.py` (verify with `sha256sum -c` against `MANIFEST.sha256`). The temporary copies `audit/tmpsrc` and
  `audit/anarepro` were moved to the trash with `trash-put` (restorable for 7 days).
