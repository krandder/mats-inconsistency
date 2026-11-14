# AUDIT-R5-E2: independent audit of R5-E2 (6x6 Barrage micro-Stratego)

Auditor: an independent read-only Board investigator (Claude Opus 5.5, 1M context, model id claude-opus-5-5[1m]),
2026-10-17, 05:15-05:45Z. Audited: `RESULTS-R5-E2.md` sha256 `70e69f12...` (read 05:16Z, unchanged at 05:44Z),
`PREREG-R5-E2.md` sha256 `732514b5...`, the code in `r5/`, the synced files in `r5/runs/` and `r5/report/`, the data in
`/tmp/r5e2/data/` and the checkpoints in `/tmp/r5e2/ckpt/`. Paths are relative to
`/home/agent/workspaces/mats/work/micro-stratego/` unless absolute. The auditor's scripts and outputs are in
`/home/agent/workspaces/fable-strategy/scratch/r5e2-audit/` (called `AUD/` below). Nothing in the experiment
directory was edited. This file was written to `AUD/` because the auditor may write only there; copy it to the
experiment directory if wanted.

## Verdict

**Supported with corrections.** The pre-registered result stands: the E2-D0 gate fails, so D1-D3 are not rule tests,
and every exploratory D1-D3 rule would also fail. I re-derived every headline number from the per-item CSVs with my own
code (300 numbers, largest difference 6.3e-6). I recomputed D0 (1), (2), (3a) and the capacity rule on CPU from the
saved checkpoints with my own tape code, for all 3 seeds, both nets and both splits (largest difference 2.2e-6). Every verdict
comes out the same. The pre-registration hash and timing pass, and section 1 is a byte-exact copy of the spec. No oracle
reaches training, fitting, selection or the agents. The pre-registered capacity rule was applied as written, on the
calibration split: neither net reached 0.05 there, so no net was selected. The corrections are small and none changes a
verdict. (1) In seed 102 the D2 "10 equal-count entropy strata" are not equal-count, because of tied entropies. (2) The
cone ignores the spec's "distance <= 2" limit: 65% of cone slots are farther away, and this is not declared as a
deviation. (3) The prereg says all data were generated before hashing, but three scoring files and the oracle's brute-force
check were finished 13-35 minutes after the hash, by the same unchanged generator run. (4) A few numbers and phrases in RESULTS need small fixes.

## Checks

| check | result | evidence (command or file; numbers; UTC time of the check) |
|---|---|---|
| A. Pre-registration integrity | **pass** | `sha256sum PREREG-R5-E2.md` = `732514b572d45ea8...` = `PREREG-R5-E2.sha256` (05:15Z, again 05:44Z). Hash time 03:25:20Z is before the first `r5/gpu/launches.log` line (03:26:12Z, smoke) and before the first `PENDING-CREATE vm=soup-gpu-r5e2-*` (03:26Z, `logical-soup/GPU-LOG.md` line 2005; no earlier r5e2 line in that file). Seed runs started 03:36:21Z. Section 1 of the prereg and spec section 2.3 are byte-identical: 2,243 bytes each after trimming blank lines at the ends, both sha256 `4ce6d8be...` (`AUD/spec23.txt`, `AUD/prereg_s1.txt`, `cmp`). The spec file's sha256 `7958f340...` equals the value recorded in the prereg. Extra evidence: the Python code staged for the VMs at 03:25:56Z (smoke) and 03:36:05Z (seed runs) (trashed copies `/.Trash-1954572927/files/src_1`, `src_2`) and at 04:22:07Z (`/tmp/r5e2/src`) is byte-identical to the current `r5/*.py`. Only `gpu/run_vm.sh` changed between the smoke and the seed runs (it adds the logged READY_ALL wait). Data timing: see correction 3. |
| B. No oracle in training or selection | **pass** | grep (05:30Z): `train6.py` opens only `{train,cal}.npz` (line 54); `model6.py` opens no file; `agents6.py` uses `msgame6` only for `encode_root`, `encode_ply`, `legal_canon` and `LMAX`, and zeroes every unrevealed true type before a tape is built (lines 62-63). None of the three names an oracle quantity. `eval6.py` opens oracle files only in its scoring functions: `part_d0` (line 155), `part_d0b` (180, 203), `part_d2` (312) and `part_d3a` (390, where `miner` is read after both choices are made). The capacity rule in `report6.py` (lines 71-79) reads only `d0cal` summaries. Aggregator targets are model-only: `coverage_reference` of the net's own 64 + 64 rollouts (`train6.py` 324-326). theta_trust targets realised types and results on CAL; theta_amp targets the net's own rollouts on TRAIN. The one oracle-dependent choice is the capacity rule (P*_full on the calibration split). It was applied as pre-registered: primary 0.0467 [0.0442, 0.0492], fallback 0.0493 [0.0476, 0.0511], both < 0.05, so no net was selected and the gate is reported for the primary. Note: `audit6.py`'s own B check misses the f-string read in `part_d0` and flags the harmless `msgame6` import. The grep above covers both. |
| C. Re-derivation from per-item CSVs | **pass** (one definitional discrepancy) | `AUD/my_rederive.py` (05:20Z), my own code, no executor import: 300 numbers against the JSON summaries; largest difference 6.3e-6 (D2 strata, 6-digit CSV rounding); `AUD/compare.csv`. Primary, 3-seed means: D0 test KL 0.0484 per hidden slot, total 0.272, 7-known NLL 0.0001, battle accuracy 0.9999 (11,034 battles per seed); calibration KL 0.0467. D0-3b: belief reduction -10.55, action-value reduction -1.49. D1: key gap 0.0040; JSD ratio mix/attn 0.161 (belief) and 3.574 (value); KL to reference: mix 0.201, ataraxos_mmd 0.306, kc_dedup 0.309, attn_L3 0.161; value MSE: 0.0017, 0.0032, 0.0036, 0.0021. D2: Spearman trust +0.2276, -H -0.3625, within-strata difference +0.187. D3a: shift rate 0.3375 (76, 74, 84 relevant boards). D3b: 0.5029. `audit6.py`, run on a byte-identical copy in `AUD/copy/` (05:19Z; output `AUD/audit6_output.txt`, `AUD/copy/r5/report/audit_rederived.csv`): A precedes=True; C 186 numbers, largest difference 8.5e-5; D verdicts agree; E on-VM differences <= 1.0e-6. **Largest discrepancy:** D2 within-strata Spearman of -H, seed 102 primary. `d2.json` gives -0.0591; exactly equal-count strata give -0.0341 (-0.0336 to -0.0339 with random tie-breaking). The cause: 5,349 rollouts that end in a predicted terminal share one entropy value (0.0800525), so the quantile cut put 331 and 11,002 rollouts in the two lowest strata. The 3-seed within-strata difference becomes +0.180 [0.108, 0.252] instead of +0.187 [0.091, 0.283]. No verdict changes. |
| D. Verdicts recomputed (t = 4.303, prereg section 2) | **pass: all agree** | `AUD/my_verdicts.py` -> `AUD/my_verdicts.csv` (05:21Z). Capacity rule: no net selected. Gate: FAIL ((3a) 0.0484 < 0.05; (3b-i) -10.55 and (3b-ii) -1.49, both < 0.10). D1 would fail: (1) 0.0040 [-0.0113, 0.0193]; belief ratio 0.161; value MSE mix - attn_L3 -0.0005 [-0.0015, 0.0006]. D2 would fail on (1): +0.2276 [0.1770, 0.2781] < 0.25. Parts (2) +0.590 and (3) pass under both strata choices. D3 would fail: (a) 0.3375 [0.2654, 0.4096]; (b) 0.5029 [0.4480, 0.5578]. Every per-seed value, mean and pass flag equals `r5/report/verdicts.csv` and the RESULTS verdict table. Intervals differ by at most 1e-4 (t = 4.303 vs 4.30265). The fallback net, secondary and not a rule outcome, also fails D0 (3b) and would fail D1-D3. Its test (3a) is 0.0514 [0.0491, 0.0537], as RESULTS says. |
| E. Independent CPU recomputation from checkpoints | **pass** | `AUD/my_cpu_d0.py` (05:22-05:27Z): my own tapes built from `test.npz`/`cal.npz` records matched by (game, t), float64 softmax, only `RealNet` from `model6.py`. Run for all 3 seeds x 2 nets x test (2,000 positions) and calibration (1,000 positions); `AUD/cpu_d0_all.jsonl`. Against `eval/d0.json` / `d0cal.json`: KL difference <= 2.2e-6, 7-known NLL <= 2.4e-7, battle accuracy 0 (same 11,034 and 18,555 ply-0 battles, same misses). Against `eval/audit_vm.json`: KL <= 3e-9. Per-position differences <= 4.7e-4 (the VM ran TF32). Seed 101 primary, first 500 positions: 0.047159 vs the CSV subset mean 0.047158. Checkpoint capacity rule: primary calibration 0.04667, fallback 0.04935, so no selection is confirmed. Oracle sanity: the true type has positive P*_full mass on every hidden slot (minimum 0.0044 test, 0.0109 calibration), and the P*_full support lies inside the P*_c support. Oracle reproduction: the executor's brute-force replay (plain-loop policy + `env6.step`) on 64 more stored positions (44 test, 20 calibration, t 6-36), against the stored `pstar_full`/`pstar_c`: max error 6.7e-16 and 0 (`AUD/bf_*.json`, `AUD/bfb_*.jsonl`; the recorded check covered only t 6-16). |
| F. Code vs pre-registered definitions | **pass**, with one undeclared interpretation and two small deviations | Cone: my plain-loop version of "3 nearest hidden slots by Chebyshev distance to the target; ties by from-distance, then slot index" agrees with `model6.select_cone` on 5,000 of 5,000 random (test record, legal move) pairs (`AUD/my_cone_check.py`). But 64.9% of cone slots are more than 2 from the target square: the spec's "within tactical range (distance <= 2)" is not applied, and `in_range` is computed but unused (correction 2). Windows: t0 = max(t-4, t mod 2) (`train6.py` 59, `agents6.py` 44). Same observer parity for all 98,816 test records; 4-ply windows for t >= 4; 7,356 of 7,356 sampled window plies equal the real plies. The Z_s-masked reference term is constant in k and cancels in the softmax (`model6.py` 474-499). Coverage reference (`model6.py` 287-296) and mix (226-233) match the prereg. D1: two pools of 24 + 24, U_Left n_L ~ Binomial(24, 0.8) from pool 1, U_Right Binomial(24, 0.2) from pool 2, reference 64 + 64 (`eval6.py` 262-273); observed mean n_L 19.1-19.3 and 4.7-4.8. The 500 D1 positions come from 455 games (the prereg does not ask for one per game here). The key-gap averaging order differs from the prereg text (per position over policies, then over positions, instead of per policy over positions, then over policies); effect <= 1e-4. D2: 2,361 (position, action) pairs x 24 rollouts (6 depths x 2 checkpoints x 2 repeats) = 56,664 per seed. Random cone order. `qstar_eff` equals the oracle file (max 5e-7). 120 + 120 positions, one per game; Miner-contact filter holds on 120 of 120 (replayed with `env6`). Strata are not equal-count in seed 102 (see C). D3a: relevance and shift recomputed from `oracle_q_test.npz` `miner` and the chosen action indices are identical (76/74/84; 0.368/0.311/0.333). Hash identical, belief change 0. D3b: every pair index is played with both colours (200 x 2 in every seed and net); the pairs re-drawn with `rng(seed + 500)` are test setups; results are in {0, 0.5, 1}. Legal lists in the records equal the oracle `canon` lists exactly (Q* test and calibration). Split disjoint; each split's games use only its own setups. `AUD/my_spotchecks.py` -> `AUD/spotchecks.json` (05:29Z). |
| G. Compute and clean-up | **pass** | My own parse of the 35 r5e2 lines in `logical-soup/GPU-LOG.md` (about 05:33Z): 5 VMs, 3.13 VM-hours at the $0.87/h list-price ceiling. Smoke 0.12 h $0.10; s101 0.88 h $0.77; s102 0.88 h $0.77; s103 spot, preempted, 0.42 h $0.37 (COST-AUDIT line); s103-b on-demand 0.83 h $0.72. **Total $2.73** (sum of logged per-VM costs; $2.72 unrounded), within the $9 cap. Read-only `gcloud compute instances list` and `disks list`, project GCP_PROJECT, filter `name~soup-gpu-r5e2-`: no rows at 05:34:30Z or at 05:44:24Z (rc 0). A control listing in the same project returned other instances, so the empty result is not an auth failure. Nothing from the preempted run is in `r5/runs/s103/`: every file there is from 04:34Z or later, and its driver log shows no sync before the preemption. Minor: the ledger row for the preempted VM shows "deleted 04:11Z" (the COST-AUDIT line time; the preemption was at 04:02:44Z) and the basis "DELETE line". The cost is unaffected. `gpu/ledger.py` was not run, because it rewrites `r5/GPU-LOG-R5-E2.md`; its parse and the gcloud calls were reproduced instead. |
| H. RESULTS-R5-E2.md | **pass, with small corrections** | Every rule has a verdict with per-seed numbers, means and seed-t intervals. D1-D3 are marked "not a rule test (gate failed)" and "would fail". The rule table equals `r5/report/tables.md` (`diff`: only the characters "‖" and "\|\|" differ). Failures are stated as failures, and exploratory and post hoc items are labelled. I checked 42 claims against the files (list below; several carry more than one number): 40 match at the stated precision, 2 do not (corrections 4 and 5). One verdict-table phrase is incomplete (correction 6). Some numbers have no source path (correction 7). The post hoc mechanism claim (zero claims come from the one-of-a-kind types) checks out: with my own random draws, 21.5% of covered claims give the true type < 1e-3, and for 98.2% of those the true type is already used up by the other sampled cone slots or by revealed slots (`AUD/check_claim_zeros.py`, seed 101, 05:38Z). No automatic text classifier was used; H is a manual read. |

Numbers checked in RESULTS (line: claim -> file value). OK unless marked.
L14 0.0484 [0.0471, 0.0498] -> verdicts.csv. L15 0.0467, 0.0493 -> tables.md and checkpoints. L13 battle 0.9999, NLL
0.0001 "both" -> primary yes, fallback 0.9998, 0.0002 (**correction 5**). L17 KL 0.047 -> 0.539 -> 0.0467 -> 0.5389.
L17/L44 MSE 0.015 -> 0.037 -> 0.0152 -> 0.0365 (0.036493) (**correction 4**). L19 0.99 / 0.98, gap 0.004 -> 0.9864 /
0.9824 / 0.0040. L20 3.6x -> 3.574. L21 6.2x -> 1 / 0.1614. L22 +0.23 vs -0.36 -> 0.2276 / -0.3625. L23 34% -> 0.3375.
L24 0.503 -> 0.5029. L27 21% -> 21.6 / 21.1 / 21.4%. L28 0.54 / 0.17 / 0.053 -> 0.542 / 0.172 / 0.0533. L29 0.076 vs
0.048 -> 0.0764 / 0.0479. L31 0.40 vs 0.30 -> 0.401 / 0.299. L32 $2.73. L39 11,034 battles. L41 0.272 -> 0.2723.
L43 -10.52, -10.25, -10.87; -10.55. L44 -1.36, -1.06, -2.06; -1.49. L52 0.0514 [0.0491, 0.0537]. L52 -9.73 and -2.23 ->
-9.7284 / -2.2345. L58 0.118 -> 0.117962. L61 0.019 and 0.098 -> 0.01856 / 0.09814. L62 0.051 vs 0.048. L67 78% -> 0.781.
L68 about 60% worse -> 1.59x. L74 u_b 0.09 -> 0.0882. L77 1.01 vs 0.22 -> 1.0057 / 0.2206. L98 1,979,305 / 164,321 /
98,816 positions; 39.4 / 39.6 / 20.9% -> meta.json. L102 6.7e-16 and 0 -> meta.json. L106 1,001,644 / 492,268 ->
train_meta.json. L172 +0.228 [0.177, 0.278], +0.590 [0.513, 0.667], +0.149 vs -0.038, +0.187 [0.091, 0.283] -> d2.json
(strata caveat, correction 1). L174 169,992 rollouts, 9.4% terminal. L176-178 76 / 74 / 84; 34% [27%, 41%]; 28%; -0.025
-> -0.010. L180 0.478, 0.514, 0.518; 0.503 [0.448, 0.558]. L181-182 fallback 0.009, 0.16, 5.6, +0.234 vs -0.319,
+0.171 vs -0.042, 0.35, 0.515 [0.474, 0.556]. L199-203 0.178, 0.076 [0.073, 0.080], 0.048 [0.045, 0.051]; 0.193 /
0.083 / 0.048; 0.181 -> 0.081 vs 0.051. L205-207 fingerprints 0.35/0.30, 0.40/0.30, 0.46/0.30, keys and Jaccard ->
expl_keys_gibbs.json. L232-234 27.7 / 17.7 min training (1,658-1,677 s and 1,062-1,073 s), generation 56 min (3,367 s).
L256 69 numbers, 8.0e-7 -> selfcheck.csv. L259 1.0e-6, 2.4e-7, 0 -> audit_vm.json.

## Corrections (exact suggested wording)

1. **D2 strata** (RESULTS line 172-173; same caveat for the D2 row of the rule table). After "difference +0.187 [0.091,
   0.283] (bar 0.15)." add: "Seed 102's strata are not equal-count: 5,349 rollouts that end in a predicted terminal
   share one entropy value, so the quantile cut put 331 and 11,002 rollouts in the two lowest strata. With exactly
   equal-count strata the seed-102 difference is 0.207 instead of 0.227, and the 3-seed difference is +0.180 [0.108,
   0.252] (AUDIT-R5-E2.md). The rule outcome does not change."
2. **Cone limit not declared** (add to the deviations log, after the line on declared deviations): "Cone definition
   (PREREG section 4, not listed in section 6): the cone is the 3 hidden opponent slots nearest the root move's target
   square, at any distance. The spec asks for slots 'within tactical range (distance <= 2)'. That limit was not applied:
   `model6.select_cone` computes `in_range`, but nothing uses it. In 5,000 random (test record, legal move) pairs, 65% of
   cone slots were more than 2 squares (Chebyshev) from the target square (AUDIT-R5-E2.md). This affects the exploratory
   D1 key and cone-overlap numbers. It cannot change the gate verdict, because (3a) fails without any rollout."
3. **Data timing** (deviations log, the 03:36Z item; the same fact belongs in `PREREG-R5-E2-AMENDMENTS.md`): "PREREG
   section 3 says the data and oracle sets were generated before the hash. That holds for the train, calibration and
   test records and the two D0 oracle sets (03:15-03:17Z). The Q* test set (03:38Z), the Q* calibration set and the D1
   positions (03:58Z) and the brute-force check (04:00Z) were finished after the hash (03:25:20Z). They came from the
   same generator run, started at about 03:04Z from code unchanged since 03:02Z (`/tmp/r5e2/gen.log`, file times). The
   amendments log's list of late files omits the Q* test set."
4. **Rounding** (RESULTS lines 17 and 44): "(MSE 0.015 -> 0.037)" -> "(MSE 0.015 -> 0.036)".
5. **Fallback one-step numbers** (RESULTS lines 12-13): "battle accuracy 0.9999, 7-known-slot NLL 0.0001 nats" ->
   "battle accuracy 0.9999 and 0.9998, 7-known-slot NLL 0.0001 and 0.0002 nats (primary, fallback)".
6. **D1 verdict-table phrase** (RESULTS line 46): "would fail ((1) and the belief half of (2))" -> "would fail ((1), the
   belief half of (2), and the value-MSE comparison with mix)".
7. **Sources** (RESULTS lines 59, 79 and 103): "against 0.075 on R4-E1's 4x4 board" -> "against 0.075 on R4-E1's 4x4
   board (`RESULTS-MS.md`)"; "(+0.23 vs +0.42)" -> "(+0.23 vs +0.42 on 4x4, `RESULTS-MS.md`)"; "(max error 4.4e-16)" ->
   "(max error 4.4e-16; no output file was kept; the audit reproduced it on 418 pairs)". Also note that
   `r5/report/expl_routeb_summary.json|csv` comes from no script in `r5/`. It matches the per-seed files to 7e-6
   (auditor's check).

Optional wording (interpretations stated as facts; not required): line 62-63 "so capacity is not what limits it" -> "so
halving the parameters changes it little (0.051 vs 0.048)". Line 170, start with "A possible reason, not tested:" before
"The reference pools the same sharp claims".

## What I could not check

- I did not write an independent implementation of the behaviour-policy likelihood, P*_full or the Q* simulator. The
  oracle evidence is the executor's brute-force replay (a separate plain-loop code path), which I re-ran on 64 stored
  positions, and the sanity checks in E. The Q* playouts were not re-simulated.
- The rollout-based numbers (D0-3b, D1, D2, D3) were re-derived from the per-item CSVs, not re-run from the
  checkpoints. GPU sampling cannot be replayed bit for bit on CPU.
- The $0.87/h rate is the list-price ceiling. The real bill is unverifiable, because the Billing API is disabled.
- The order of events relies on file times, self-recorded hash times and log lines. They agree with each other, but
  they are not independent timestamps.
- The behaviour-policy constants and the spec's 2-ply reply term were checked only against the prereg's description,
  not for game quality.
- R5-E1's share of the shared r5 group budget ($22 cap) was not checked; only R5-E2's own ledger was.

## Commands run (all read-only outside `AUD/`; CPU, at most 4 processes, torch at most 4 threads)

```
# in the experiment directory:
sha256sum PREREG-R5-E2.md; cat PREREG-R5-E2.sha256 r5/gpu/launches.log; grep -n r5e2 .../logical-soup/GPU-LOG.md
# in AUD/, with PYTHONPATH=/tmp/r5e2/pylib R5E2_DATA=/tmp/r5e2/data:
awk (spec section 2.3 and prereg section 1) > spec23.txt prereg_s1.txt; cmp; sha256
rsync -a r5 + PREREG files -> copy/ (hash-verified identical); cd copy/r5 && python3 audit6.py | tee ../../audit6_output.txt
python3 my_rederive.py; python3 my_verdicts.py
python3 my_cpu_d0.py SEED ARCH SPLIT [N]   # 101 primary test 500, then all 3 seeds x 2 nets x test/cal -> cpu_d0_all.jsonl
python3 my_spotchecks.py; python3 my_cone_check.py; python3 repro_policy_check.py; python3 check_claim_zeros.py 101
python3 repro_bruteforce_late.py {184,431,530,1678}; python3 repro_bruteforce_batch.py {0..3} 4
grep -n (oracle reads, msgame6/env6 uses, slot_true, in_range) r5/*.py; cmp trashed and staged src trees with r5/*.py
gcloud compute instances list --project GCP_PROJECT --filter="name~soup-gpu-r5e2-"; gcloud compute disks list (same filter)
```
