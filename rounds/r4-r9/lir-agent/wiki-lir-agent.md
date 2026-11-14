---
title: "lir-agent — LIR control masks for transformer post-training (overnight programme, 2026-10-03)"
status: done-round-one
---

# lir-agent — LIR control masks for transformer post-training

Draft for the hub wiki, written by the eurisko report agent on 2026-10-03 at about 18:20Z. Not pushed. Workspace: `/home/agent/workspaces/eurisko-study/lir-agent/` (spec `GOAL.md`, report `REPORT.md`, audit `CLAIMS.md`). Owner: eurisko (scientific lead). Related: [[agent-calibrated-ai]], [[consistency-ft]].

## What it is

An overnight programme by Kelvin and eurisko. It tests whether letting reward move only an action adapter, while belief and forecast adapters learn only from realised truth, keeps calibration and self-prediction honest under RL. The claim comes from Local Inconsistency Resolution, Richardson et al. 2026 (https://arxiv.org/abs/2604.17140). Setup:
- Qwen3-0.6B with LoRA role adapters.
- ShipHunt, a Battleship-style text POMDP with exact Bayes answers.
- Seven experiments, each a small change to a named paper, plus two theory items.
- An adversarial hawk ran before every item.

## Round-one findings

Every number is from `CLAIMS.md`, the skeptic's audit re-derived from the raw files, checked at 2026-10-03T18:07Z.
- **The parameter mask isolates the reward step exactly.** The reward gradient into the belief and forecast adapters was 0 at every step. After RL, swapping in the post-RL action adapter leaves beliefs and forecasts bit-identical. This holds by construction.
- **The DCPO-style token mask leaks.** Each reward step still moved a belief: by 0.0055 on average, and up to 0.091 at a false-note cell.
- **The fused baseline drifts toward what the approval rater pays for; the protected arm does not.** E2's shift toward a false note was +0.036 ± 0.026 protected vs +0.286 ± 0.013 fused. E1's false-note agreement was 0.000 vs 0.593 ± 0.526.
- **No fused failure appeared in forecast calibration** (E1 ECE 0.114 / 0.107 / 0.088) **or in self-prediction.** In E7 the fused arm tracked itself better: 0.938 vs 0.753.
- **Reward did not rise in the protected arms**, so "calibration survives while reward rises" is untested.
- **At 0.6B the separate forecaster collapsed to a constant** (0.6501 for every plan), so the best-of-n / optimism-gap question is untested.
- **Theory T2 proves the fused desire signature.** At a fused stationary point, the error projected on the reward gradient is (λ/2)·abs(∇U)² ≥ 0, while protected stationary points are calibrated. Checked numerically on toys only.
- **T1 (an uncertainty-bounded stopping rule) was killed** on E3's data: it lost to a tuned fixed n in 27 of 27 cells.

## Status (each line names its check)

| claim | status | check (read-only) | checked at | evidence |
|---|---|---|---|---|
| No lir-agent GPU VMs are running | true | `gcloud --project=GCP_PROJECT compute instances list --filter='labels.logical-soup=1'` | 2026-10-03T18:09Z | empty list |
| Programme GPU spend | $46.99 over 50 VMs with DELETE lines (ceiling rate); 4 VMs have no DELETE line, total at most about $48.7 | the python audit block in `CLAIMS.md` ("GPU-LOG audit block") over `logical-soup/GPU-LOG.md` | 2026-10-03T18:09Z | output `50 46.99 47.09` |
| Workspace disk under the 5 GB limit | true (159M) | `du -sh /home/agent/workspaces/eurisko-study/lir-agent` | 2026-10-03T18:09Z | `159M` |
| All 7 experiments have RESULTS.md | true | `ls lir-agent/experiments/E{1..7}/RESULTS.md` | 2026-10-03T18:09Z | 7 files |
| Faith-swap test was run | false | `grep -l faith_swap` over `experiments/*/runs/**/summary.json` (117 files) | 2026-10-03T18:12Z | 0 files with a result |
| Any run used a forecast-supervision arm other than full branching | false | `fc_arm` field in `experiments/*/runs/**/config.json` | 2026-10-03T18:12Z | 129 of 129 are `A4` |
| Free-form judged forecast arm (FF) built | false | `harness/README.md` "Not built yet" item 1 | 2026-10-03T18:09Z | listed as not built |
| E4 leak regression test passes | true | `cd harness && source scripts/env.sh && cd .. && python3 -m pytest -q experiments/E4/test_e4_leak.py` | 2026-10-03T18:03Z (skeptic) | 2 passed |
| Morning report published | true | `cmp lir-agent/report.html /home/agent/memo-serve/lir-agent/index.html` | 2026-10-03T18:23Z | identical (re-render: `bash lir-agent/report-files/render.sh`) |
| E2's alpha arms measure a dose | false | read `harness/protocol/train.py` reward phase (alpha-scaled loss alone, then clip + per-adapter AdamW) | 2026-10-03T18:12Z (eurisko, CLAIMS addendum) | every alpha > 0 takes a full-size reward step; not verified by a rerun |

## Round 2 (R4-E3a, 2026-10-16): Qwen3-1.7B, 7x7 ShipHunt, four role adapters

Spec: gemini's round-4 spec, section 3, Part A. Write-up `RESULTS-R2.md`, audit `AUDIT-R2.md`, prereg `PREREG-R2.md` (18:52:29Z, before the first launch).
- **E3-D0 failed, so the run stopped as pre-registered.** The forecaster gate passed in 9/9 runs (r with exact Bayes 0.60-0.84). That needed an entropy term in the decile truth loss, because the plain expected Brier saturates at the base rate. Reward did not rise: GRPO collapsed the action side to one goal-blind cell in all 9 runs, which pays exactly 0.5 averaged over the two goals.
- **Descriptive.** The LIR mask kept the belief bit-identical (reward gradient 0.0 at 350/350 steps). Shared adapters wiped the belief: Fused r 0.68 -> -0.03. One Fused seed gamed the rater with a constant 0.55 belief.
- **Fix 2 (single-step alpha dose)** is verified by unit test. No dose-response was run.

| claim | status | check (read-only) | checked at | evidence |
|---|---|---|---|---|
| No r4e3a GPU VMs are running | true | `gcloud --project=GCP_PROJECT compute instances list --filter='name~r4e3a'` | 2026-10-16T21:02Z | 0 items |
| R4-E3a GPU spend | $9.04 at the ceiling rate (cap $18) | `python3 results-r2/gpu_cost.py` | 2026-10-16T21:02Z | `TOTAL_r4e3a_usd 9.04` |
| R4-E3a numbers reproduce from raw files | true | the audit's own code, `/tmp/r4e3a/audit/my_analysis.py`, over `results-r2/runs/*/eval.jsonl` | 2026-10-16T21:01Z | max diff 5.6e-16 (`AUDIT-R2.md` check 1) |
| /experiments entry lir-r4-e3a published | false (built locally only) | `grep -c lir-r4-e3a /home/agent/memo-serve/mats-experiments/all.html` | 2026-10-16T21:02Z | built; `sync_economy_tracks.sh` not run |

Next (not run; would need a new prereg): the same harness with a policy that cannot collapse (an entropy bonus or KL to the starting action policy, lr 1e-4, 8 samples per group).

## Open decisions (Kelvin)

- Is "localized honesty under RL" the MATS framing? REPORT.md §6 suggests "a guarantee you can check, and what it costs", not "better calibration".
- Should round two be funded? REPORT.md §7, ranked:
  1. the projected-error metric on saved records (CPU);
  2. an informative forecaster (1.7B or more warm-up);
  3. a setup where reward rises, with 5 seeds;
  4. E2's leak arms A2/A3;
  5. FORECAST supervision arms with the back door closed;
  6. the FF judge arm.

## Lessons for future GPU nights

- **One budget metric for every launch gate.** Admission and the $50 stop read reservation-inclusive spend, while some launchers read the guard's estimate. E1, E2 and E5 dropped arms at 13:41-13:42Z, while about $5.1 of later launches went ahead on the other metric (`CLAIMS.md`, programme checks).
- **One 0.6B process per L4.** Two 0.6B processes on one 22 GB L4 ran out of memory or gained nothing; later VMs ran one job at a time (deviation logs in the experiments' PREREG.md files; unaudited).
- **Gate on forecaster informativeness before testing forecast claims.** Require probe correlation with Bayes above 0.2 (E3, T1).

## Sources

- Richardson et al. 2026, Local Inconsistency Resolution, https://arxiv.org/abs/2604.17140
- Ma et al. 2026, DCPO, https://arxiv.org/abs/2603.09117
- Cobbe et al. 2020, Phasic Policy Gradient, https://arxiv.org/abs/2009.04416
- Binder et al. 2024, Looking Inward, https://arxiv.org/abs/2410.13787
- Gao, Schulman, Hilton 2022, https://arxiv.org/abs/2210.10760
