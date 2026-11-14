# E7 RESULTS: self-prediction (Binder et al. 2024) under GRPO, fused vs protected

E7 subagent for eurisko, 2026-10-03 17:15Z. Prereg: `PREREG.md` (deviations log at its end).
Hawk: `../../hawks/E7.md` (verdict reshape; all nine reshape items adopted, item 2's frozen control done post hoc).
Paper modified: Binder et al. 2024, Looking Inward, https://arxiv.org/abs/2410.13787 (abstract re-fetched 2026-10-03).

**Label: "self-prediction tracking", not introspection.** Gate 0 found no privileged self-access
(below), so every accuracy here measures how well a self-model tracks the policy. None of it shows
introspection.

All numbers are mean ± sd over seeds, held-out states (256 test states, seeds 80M+; never used in
SFT, RL or eval). Source of every table: `results/results.json` (built by `aggregate.py` from
`an/*.json` and `runs/*/eval.jsonl`); flat copy `results/tables.csv`; plot `results/insitu.png`.
Gate 0 alone, computed before any RL launch: `gate0/results.json`.
Seeds: P-all 5, B0 5, P-role 3 (the 5-seed P-role VM was lost, so it was rerun at reduced scale; see Deviations).
Model Qwen3-0.6B (bf16 base, LoRA r8), ShipHunt `easy`, goal fixed to "hit". Each run: SFT 80 steps,
then GRPO 60 steps × 32 episodes. `summary.json: mask_violations` = 0 in all 13 RL runs (`runs/*/summary.json`).

## Arms

| arm | what reward can move | self-model computation | runs |
|---|---|---|---|
| P-all | action adapter only | belief adapter + the live action adapter stacked at the self-query positions (privileged route AND reward leak; hawk H1) | `runs/P-all-s{0..4}` |
| P-role | action adapter only | belief adapter on the frozen base (Binder M2 by construction, no leak) | `runs/P-role-s{0..2}` |
| B0 | one fused adapter (reward on every sampled claim token) | the same fused adapter | `runs/B0-s{0..4}` |

All arms get the same self-prediction loss: NLL of their own committed first shot, every step, lr 2e-4
(`runs/*/summary.json: self_loss`). No own past actions appear in any self-model context. Each
episode has one decision, and the role passes are separate sequences (`summary.json: past_action_typing`).

## 1. Gate 0 (before RL): Binder M1 vs M2 on the SFT policy (P-all warm starts, 5 policies × 5 predictor seeds)

| quantity | value | file |
|---|---|---|
| acc S (own action adapter stacked) | 0.803 ± 0.089 | gate0/results.json fresh."P-all\|pre_rl" |
| acc X1 (twin policy stacked) | 0.841 ± 0.053 | same |
| acc X0 (fresh adapter, frozen base) | 0.779 ± 0.066 | same |
| mode baseline | 0.327 ± 0.098 | same |
| **self-advantage** S − max(X0, X1, mode) | **−0.038 ± 0.072**; hierarchical bootstrap 95% CI **[−0.104, +0.006]** | gate0/results.json gate0_and_selfadv_bootstrap |
| entropy-matched / collapsed-fits-dropped / idiosyncratic subset | −0.038 ± 0.074 / −0.043 ± 0.068 / −0.025 ± 0.049 | same (self_adv_matched, _nc, _idio) |

The CI includes 0, and the point estimate is negative: the twin predicts P's modal shot at least as well as
P's own weights do. This matches Song, Hu, Mahowald 2025 (https://arxiv.org/abs/2503.07513, re-fetched
2026-10-03: no privileged self-access in 21 open models). E7 is therefore relabelled "self-prediction tracking".

## 2. Reward and policy entropy per checkpoint (eval 128 episodes; H(pi) = first-shot entropy on held-out states)

| arm | reward pre → rl30 → post | H(pi) pre → rl30 → post | commit entropy post |
|---|---|---|---|
| P-all | 0.658 ± 0.039 → 0.529 ± 0.097 → 0.510 ± 0.153 | 0.490 ± 0.040 → 0.187 ± 0.097 → 0.086 ± 0.061 | 0.500 ± 0.286 |
| P-role | 0.659 ± 0.023 → 0.620 ± 0.058 → 0.610 ± 0.067 | 0.469 ± 0.069 → 0.142 ± 0.031 → 0.122 ± 0.030 | 0.504 ± 0.196 |
| B0 | 0.654 ± 0.034 → 0.623 ± 0.009 → 0.533 ± 0.235 | 0.672 ± 0.184 → 0.382 ± 0.101 → 0.303 ± 0.242 | 0.673 ± 0.250 |

(results.json insitu."<arm>|<tag>".eval_reward / H_pi / eval_commit_entropy.) **Reward did not rise in
any arm.** GRPO from this SFT start, at lr 2e-4 for 60 steps, lowered reward. One B0 seed collapsed
(s0 post reward 0.119, commit entropy 1.07: `runs/B0-s0/eval.jsonl`), and P-all s4 fell to 0.277.
First-shot entropy collapsed in every arm, as Cui et al. 2025 predicts (https://arxiv.org/abs/2505.22617).
So the claim's precondition ("reward rises") was not met, and the comparison below is at roughly matched (falling) reward.

## 3. In-situ self-model (the one trained during RL), held-out states

| arm, tag | acc (raw) | acc entropy-matched | mode baseline | TV(q, pi) | acc idiosyncratic | frac idiosyncratic |
|---|---|---|---|---|---|---|
| P-all pre | 0.767 ± 0.129 | 0.830 ± 0.112 | 0.327 ± 0.098 | 0.227 ± 0.085 | 0.668 ± 0.198 | 0.212 ± 0.097 |
| P-all post | 0.854 ± 0.117 | 0.753 ± 0.085 | 0.415 ± 0.297 | 0.211 ± 0.131 | 0.433 ± 0.372 | 0.244 ± 0.352 |
| P-role pre | 0.852 ± 0.024 | 0.898 ± 0.016 | 0.307 ± 0.073 | 0.236 ± 0.046 | 0.747 ± 0.103 | 0.305 ± 0.119 |
| P-role post | 0.645 ± 0.153 | 0.578 ± 0.087 | 0.255 ± 0.068 | 0.383 ± 0.162 | 0.492 ± 0.364 | 0.311 ± 0.117 |
| B0 pre | 0.888 ± 0.037 | 0.955 ± 0.018 | 0.413 ± 0.121 | 0.104 ± 0.030 | 0.830 ± 0.106 | 0.254 ± 0.126 |
| B0 post | 0.955 ± 0.063 | 0.938 ± 0.063 | 0.431 ± 0.323 | 0.042 ± 0.031 | 0.830 ± 0.219 | 0.208 ± 0.258 |

rl30 rows: results.json. Change from pre to post in entropy-matched accuracy (per-seed differences; results.json delta_pre_to_post):
P-all −0.077 ± 0.139, B0 −0.017 ± 0.074, P-role −0.321 ± 0.101 (t-CI [−0.570, −0.071], n=3).
Raw accuracy rises in P-all and B0 while entropy-matched accuracy falls. This is the hawk's entropy confound, measured.

## 4. Valenced (signed) vs neutral properties. PRIMARY METRIC

Valenced property = Bayes P(ship) of the predicted shot minus that of the policy's shot, as E_q[b] − E_pi[b].
A positive value means optimism. The neutral properties are the shot's row and column.

| arm, tag | valenced signed error | neutral row acc | neutral col acc | neutral abs col error |
|---|---|---|---|---|
| P-all pre | −0.060 ± 0.057 | 0.859 ± 0.090 | 0.813 ± 0.104 | 0.215 ± 0.062 |
| P-all post | −0.013 ± 0.029 | 0.895 ± 0.118 | 0.863 ± 0.121 | 0.249 ± 0.156 |
| P-role pre | −0.040 ± 0.030 | 0.910 ± 0.010 | 0.875 ± 0.021 | 0.257 ± 0.116 |
| P-role post | −0.000 ± 0.043 | 0.771 ± 0.079 | 0.690 ± 0.128 | 0.479 ± 0.170 |
| B0 pre | −0.016 ± 0.013 | 0.927 ± 0.025 | 0.911 ± 0.038 | 0.128 ± 0.082 |
| B0 post | −0.003 ± 0.004 | 0.980 ± 0.025 | 0.966 ± 0.054 | 0.041 ± 0.031 |

**Primary (prereg): B0 − P-all valenced signed error at post_rl = +0.010, Welch 95% CI [−0.026, +0.046]**
(results.json primary_welch."B0-minus-P-all|post_rl|val_signed_exp"). B0's own value, −0.003 ± 0.004,
is about zero, so B0 shows no optimism. The CI is wide only because P-all varies across seeds.
Companions: B0 − P-all entropy-matched accuracy at post = +0.185 [+0.073, +0.296]; neutral |col error|
−0.208 [−0.406, −0.010]; change in entropy-matched accuracy +0.059 [−0.113, +0.232].
Matched-reward OLS over all 39 units: arm B0 − P-all on entropy-matched accuracy is +0.18 and on valenced
error +0.018. The reward coefficient is +0.09 and −0.11 respectively (results.json matched_reward_ols;
descriptive, no CI).

## 5. Fresh predictors after RL (Binder M1 vs M2), 2 predictor seeds, twin = seed (i+1) mod 5

| arm, tag | S | X1 | X0 | mode | self-adv | bootstrap 95% CI | self-adv idiosyncratic |
|---|---|---|---|---|---|---|---|
| P-all pre (Gate 0) | 0.803 ± 0.089 | 0.841 ± 0.053 | 0.779 ± 0.066 | 0.327 ± 0.098 | −0.038 ± 0.072 | [−0.104, +0.006] | −0.025 ± 0.049 |
| P-all post | 0.923 ± 0.039 | 0.930 ± 0.033 | 0.919 ± 0.058 | 0.415 ± 0.297 | −0.013 ± 0.010 | [−0.027, −0.002] | +0.003 ± 0.075 |
| B0 post | 0.957 ± 0.027 | 0.946 ± 0.032 | 0.952 ± 0.030 | 0.431 ± 0.323 | +0.005 ± 0.004 | [−0.005, +0.013] | −0.013 ± 0.082 |

(results.json fresh / gate0_and_selfadv_bootstrap.) Neither arm has a self-advantage after RL. X0, a
fresh adapter on the frozen base, gains as much as S does (P-all pre→post: X0 +0.140 ± 0.030, S +0.120 ± 0.108).
That is the "competence/low entropy = predictability" effect. Signed valenced error of the fresh predictors
at post is −0.014/−0.014/−0.012 (P-all S/X1/X0) and −0.005/−0.001/−0.004 (B0).
Not run: B0 pre-RL fresh predictors (lost with the w2b VM), and P-role fresh predictors (S == X0 by construction).

## 6. Staleness control (hawk item 2), P-all

| tag | trained self-model | frozen belief adapter + live action adapter stacked | fully stale warm self-model |
|---|---|---|---|
| rl30 | 0.760 ± 0.084 | 0.532 ± 0.238 | 0.448 ± 0.237 |
| post | 0.854 ± 0.117 | 0.502 ± 0.207 | 0.405 ± 0.236 |

(results.json insitu."P-all|<tag>".acc_insitu / acc_frozen_live_action / acc_stale_warm_selfmodel.)
Stale B0: 0.384 ± 0.180; stale P-role: 0.504 ± 0.082 at post. This reproduces Binder's staleness gap
(trained 35.4% vs stale 21.7% there). In P-all, the live action adapter stacked under a frozen self-model
recovers +0.098 ± 0.086 of accuracy over fully stale (per-seed differences 0.070, 0.055, 0.008, 0.234, 0.121).
That is the gating-all route carrying behaviour change into self-prediction without any self-prediction
training. It is the reward-leak path of hawk H1, and Gate 0 shows it gives no advantage over a twin.

## 7. Self-fulfilment (planted statement "My first shot will be: X." in the proposal context)

| arm, tag | follows own prediction | baseline agreement | follows swapped X | argmax changes when swapped | TV shift |
|---|---|---|---|---|---|
| P-all pre / post | 0.944 / 0.798 | 0.767 / 0.854 | 0.206 ± 0.101 / 0.063 ± 0.116 | 0.280 / 0.100 | 0.262 / 0.109 |
| P-role pre / post | 0.923 / 0.771 | 0.852 / 0.645 | 0.174 ± 0.121 / 0.095 ± 0.084 | 0.228 / 0.145 | 0.213 / 0.142 |
| B0 pre / post | 0.966 / 0.988 | 0.888 / 0.955 | 0.130 ± 0.109 / 0.097 ± 0.122 | 0.171 / 0.115 | 0.164 / 0.106 |

(results.json insitu fulfil_*.) The action does read a planted self-claim: after SFT, 13–21% of argmaxes move to a
swapped cell. After RL this falls to 6–10%. In this protocol the self claim never enters the action's context
(R2), so in-trajectory self-fulfilment is structurally zero. The probe only measures susceptibility.

## 8. Direction of resolution (TV over held-out states; D > 0 = the prediction moved toward the behaviour)

| arm, step | TV(pred_t, act_t+1) | TV(pred_t+1, act_t) | D | TV(act_t, act_t+1) |
|---|---|---|---|---|
| P-all pre→rl30 / rl30→post | 0.575 / 0.378 | 0.559 / 0.338 | +0.016 ± 0.120 / +0.040 ± 0.118 | 0.569 / 0.336 |
| P-role pre→rl30 / rl30→post | 0.519 / 0.372 | 0.469 / 0.346 | +0.049 ± 0.100 / +0.026 ± 0.103 | 0.495 / 0.374 |
| B0 pre→rl30 / rl30→post | 0.460 / 0.380 | 0.433 / 0.359 | +0.027 ± 0.022 / +0.021 ± 0.034 | 0.435 / 0.361 |

(results.json direction.) D is small and positive in every arm and every interval. Only B0 pre→rl30 has an sd
small enough to read as a sign. The cross distances are about as large as the behaviour change itself
(TV(act_t, act_t+1)). Neither side "waits" for the other: both move by about the same amount within each
30-step interval, so the measure cannot separate the directions at this checkpoint spacing.

## Verdict on the claim

Prereg classification: **mixed**, under the rules written before the runs. The primary CI [−0.026, +0.046] neither
excludes 0 nor lies inside ±0.02. On substance, the predicted fused failure did not appear. B0's
self-predictions were not optimistic on the valenced property (−0.003 ± 0.004). Its neutral properties got
better, not worse. Its entropy-matched tracking accuracy stayed higher than the protected arm's
(+0.185 [+0.073, +0.296]). The protected role-gated arm (P-role) tracked worst (Δ −0.321 ± 0.101, n=3).
Two caveats limit the reach of this result:
(a) reward fell instead of rising, so "introspection survives while reward rises" was never tested;
(b) in B0 the self-claim token is not in the REINFORCE term (harness `protocol/losses.py: reward_losses`
covers the cell and forecast tokens only), so reward reaches the fused self-model only through shared
parameters.

## The hawk's objection, answered or conceded with numbers

- "Protected wins by construction / trained vs stale": **answered by design.** Every arm got the same
  self-prediction loss schedule. The protected arm did *not* win: B0 tracks better (§3–4). Staleness
  reproduced: trained 0.854 vs stale 0.405 in P-all (§6).
- "At <8B there is no introspection to break": **conceded.** Gate 0 self-advantage −0.038, CI [−0.104, +0.006].
  Post-RL: P-all −0.013 [−0.027, −0.002], B0 +0.005 [−0.005, +0.013] (§1, §5). Relabelled tracking.
- "Privileged access and the leak are one switch (gating all vs role)": **confirmed and measured.** Gating
  all beats gating role on post-RL tracking by +0.175 entropy-matched (P-role − P-all −0.175 [−0.350, −0.001]),
  and the stacked live action adapter alone recovers +0.098 over stale (§6). Gate 0 shows this route gives
  no advantage over a twin's adapter.
- "Entropy collapse makes raw accuracy rise": **confirmed.** H(pi) fell by 0.35–0.40 in every arm. Raw
  accuracy rose (P-all +0.087, B0 +0.067) while entropy-matched accuracy fell (P-all −0.077, B0 −0.017) (§2–3).
- "Competence = predictability": **confirmed.** The fresh outside predictor X0 gained +0.140 ± 0.030 after RL,
  as much as S (§5). The idiosyncratic subset is reported (§3, §5): P-all post 0.433 ± 0.372, B0 0.830 ± 0.219.
- "Self-fulfilment": the channel is absent in this protocol. The planted-statement susceptibility is 6–21% (§7).
- "Valence": no optimism in any arm at post (|signed| ≤ 0.013 on average). After SFT every arm is slightly
  *pessimistic* (P-all −0.060 ± 0.057) (§4).

## Deviations (full log with timestamps in PREREG.md)

RL 80 → 60 steps (no throughput gain from two processes per L4). Gate 0 finished on a second VM by hand
after a `wait`/tee bug. The first w2c B0 run hit OOM with two B0 processes on one L4 and was rerun with one lane.
The first w2a P-all results were lost to my in-place edit of a running driver.sh, and w2a was rerun.
The w2b P-role VM became unreachable during analysis (probably spot preemption; nothing synced), so P-role
was rerun with 3 seeds and no fresh predictors. B0 pre-RL fresh predictors were not run. Run ids
lir-e7-w1a/w1b/w2a/w2c/w2c2/w2r3 (not "lir-E7": upper case is invalid). A robustness variant `self_adv_nc` was added after 2 Gate-0 seeds.

## GPU

Seven deleted VMs in GPU-LOG.md (lir-e7-*): $1.55 + 1.55 + 0.43 + 1.47 + 1.39 + 0.77 + 1.49 = $8.65 at the rate
ceiling. The lost w2b VM has no DELETE line; it ran about 1.5 h, about $1.30 more. **E7 total ≈ $9.95 at the
ceiling** (spot is lower). Programme estimate at 17:02Z: GUARD.log total_est $47.05, which excludes w2b's ~$1.30.

## Files

PREREG.md, RESULTS.md, run_arm.py, analyze.py, aggregate.py, summarize.py, remote.sh, driver.sh, launch.sh,
launch2.sh, pipeline*.sh. Data: an/*.json (per-state analysis), runs/*/{config,summary}.json + eval.jsonl +
steps.jsonl, gate0/, results/. Checkpoints and eval pickles: /tmp/lir-agent/e7/{ckpt,evalrecs} (not on /srv).
