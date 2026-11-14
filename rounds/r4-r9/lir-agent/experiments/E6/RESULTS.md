# E6 RESULTS: gold vs proxy utility drift under continued belief learning (toy scale, CPU)

2026-10-03, Opus 5.5 subagent for eurisko. Prereg: `PREREG.md` (with an append-only deviations log).
Hawk: `../../hawks/E6.md` (verdict RESHAPE, adopted). Status by the preregistered rule: **MIXED**. No
reward-caused self-serving drift was detected. Arm c equals b_c exactly. Representation drift without
reward is large. The test is weak, because the toy policy barely learned.

**Scale (read this first).** The run used the harness's random-init `tiny` Qwen3: 2 layers, hidden size 64,
LoRA r 8 on every linear layer, lr 1e-3. It did **not** use Qwen3-0.6B. The 0.6B CPU attempt was OOM-killed or
swap-bound on the shared box (load 42-46, swap 14/15 GB), and the hawk's reshape gives E6 no GPU budget. E1
does not save its RL episode stream, so E6 could not ride on it (PREREG deviations 08:42Z-09:15Z). Settings:
preset `easy`, 40 SFT steps x 16, 200 RL steps x 32 episodes (4 GRPO groups x 8), 3 seeds. Every proxy uses
the same frozen bank (`bank.json`, hash c3c91631775a4f45, 256 environment-written outcomes on 16 held-out
boards). GPU spend: $0. All 12 runs report `mask_violations` 0 and an unchanged base checksum
(`runs/*/e6summary.json`). Forecast format: the fixed per-environment format. The FF arm was not used.

**Scope.** E6 says nothing about the expressivity of grounded variables: every preference ShipHunt rewards
is already a grounded variable (hawk item 4).

## Design as run
- **Gold** is ShipHunt `env_reward` of an outcome. **Proxy** is a linear head (normalised LMS) on
  mean-pooled final hidden states of the environment-written outcome text. It reads the forecast adapter in
  the roles layout and the fused adapter in B0. The head is initialised by ridge regression on 288 calibration
  outcomes from disjoint boards. After that it trains prequentially on the realised env_reward of the 32
  committed outcomes of each RL step (`e6lib.py`, `e6run.py`).
- **Arms:**
  - **c** = harness arm P (masked LIR).
  - **d** = harness arm B0 (fused, the same truth losses plus reward on every token).
  - **b_c, b_d** replay X's logged stream (`runs/X-sS/stream/*.pkl`) from X's warm adapters *and* X's warm
    optimiser state. They run only the harness truth phases (`Trainer._truth_phases`) plus the same head
    updates, with no reward phase.
  - **a_c, a_d** train the same head on X's stream over frozen base features (adapters disabled).
- **Metrics** (`analyze.py`, written before the runs):
  - Groups are boards (one start state, many plans), as in GRPO. There are 1920 same-board pairs.
  - gold_acc is computed on the 1488 non-tied pairs. tie_reorder is computed on the 432 tied pairs.
  - SSI uses tied pairs where X's current policy makes one outcome at least 2x more likely than the other
    (111-186 pairs per run). It is (s_t(in) - s_t(out)) - (s_0(in) - s_0(out)).
  - fam is the same contrast on the agent's log-likelihood of the outcome text.
  - ssi_resid_fam is the OLS intercept of the SSI contrast on the fam contrast.
  - norm is ||θ_t - θ_warm|| of the proxy-feature adapter.

## Table 1: per arm at RL step 200 (mean ± sd over seeds 0-2)
Source: `results.json: summary_final` (rows per step and seed in `results.csv`).

| arm | gold_acc (init → 200) | tie_reorder (free drift) | SSI | fam (log-lik contrast) | SSI net of fam | proxy-adapter norm |
|---|---|---|---|---|---|---|
| a_c (frozen base) | 1.000 → 1.000 ± 0.000 | 0.163 ± 0.012 | +0.0067 ± 0.0045 | 0 | +0.0067 ± 0.0045 | (features frozen) |
| c (masked LIR) | 0.998 → 0.854 ± 0.073 | 0.362 ± 0.022 | -0.0037 ± 0.0140 | -0.057 ± 0.069 | -0.0014 ± 0.0199 | 1.73 ± 0.15 |
| b_c (yoked, no reward) | 0.998 → 0.854 ± 0.073 | 0.362 ± 0.022 | -0.0037 ± 0.0140 | -0.057 ± 0.069 | -0.0014 ± 0.0199 | 1.73 ± 0.15 |
| a_d (frozen base) | 1.000 → 1.000 ± 0.000 | 0.164 ± 0.018 | +0.0063 ± 0.0028 | 0 | +0.0063 ± 0.0028 | (features frozen) |
| d (fused B0) | 0.998 → 0.904 ± 0.069 | 0.465 ± 0.056 | +0.0085 ± 0.0696 | +0.413 ± 0.263 | +0.0019 ± 0.0447 | 2.58 ± 0.14 |
| b_d (yoked, no reward) | 0.998 → 0.955 ± 0.031 | 0.455 ± 0.061 | -0.0425 ± 0.0544 | +0.377 ± 0.308 | -0.0017 ± 0.0137 | 2.00 ± 0.28 |

- SSI is in reward units. The gold utilities are {0.9, 0.8, 0.7, -0.3}.
- Arm c and arm b_c are identical to every printed digit in every seed. This is expected by construction:
  in P the forecast adapter receives no reward gradient, and its truth updates depend only on the stream.
  Smoke check in `/tmp/lir-agent/e6/smoke2`; per-seed rows in `pair_se.json`.
- Tie fraction:
  - Bank, same-board pairs: 0.225 (`results.json: tie_frac`). This is set by stratified bank construction.
  - Inside the training GRPO groups: 0.363 (`results.json: stream_group_tie_frac`).
  - So rank metrics on non-tied pairs keep power: 1488 non-tied bank pairs.

## Table 2: reward-caused differences X - b_X (per seed s0, s1, s2; mean ± sd)
Source: `results.json: reward_caused`.

| contrast | at step 200 (matched data) | at b_X's final norm (matched update norm) |
|---|---|---|
| ΔSSI(c) = SSI(c) - SSI(b_c) | 0, 0, 0 → 0.0000 ± 0.0000 | 0, 0, 0 → 0.0000 ± 0.0000 |
| ΔSSI(d) = SSI(d) - SSI(b_d) | -0.012, +0.075, +0.091 → **+0.051 ± 0.056** | -0.021, +0.051, +0.289 → +0.106 ± 0.162 |
| ΔSSI(d), net of familiarity | -0.026, +0.042, -0.005 → **+0.004 ± 0.035** | n/a |
| Δgold_acc(c) | 0, 0, 0 → 0.000 | 0.000 |
| Δgold_acc(d) | -0.013, -0.013, -0.127 → -0.051 ± 0.066 | +0.017, -0.108, -0.047 → -0.046 ± 0.062 |

- Pair-level standard errors of SSI per run are about 0.007-0.025 (`pair_se.json`, a post-hoc script).
  Seed-to-seed spread (sd 0.05-0.07 in d and b_d) is larger than these errors.
- The policy barely learned:
  - Mean success over the first 20 vs the last 20 RL steps: c 0.467 → 0.489, d 0.475 → 0.483.
  - Per-seed values are in `runs/{c,d}-s*/steps.jsonl`. A uniform 3-shot plan succeeds with p = 0.45.
  - Committed-forecast Brier stayed at 0.249-0.255.
- Proxy prequential MSE, first 20 → last 20 steps (`runs/*/e6head.jsonl`):
  - a: 0.0023 → 0.0017
  - c = b_c: 0.010-0.085 → 0.014-0.054
  - d: 0.022-0.070 → 0.062-0.153
  - b_d: 0.010-0.024 → 0.038-0.089
- Figure: `e6_ssi.png` (SSI and gold accuracy against step, and SSI against update norm).

## Preregistered verdict
- **Half 1, directed drift in B0:** needed ΔSSI(d) > 0 in 3/3 seeds with mean/sd > 2. Observed 2/3 seeds
  and mean/sd = 0.92 (0.051/0.056). Not met.
- **Refutation:** needed ΔSSI(d) ≤ 0 in at least 2/3 seeds, or |mean| < 0.01. Observed 1/3 seeds and
  mean 0.051. Not met.
- **Half 2, mask:** |ΔSSI(c)| = 0 < 0.01. Met, by construction.
- **Gold:** gold_acc(c) - gold_acc(b_c) = 0 ≥ -0.02. Met.
- **Result: MIXED.** The preregistered familiarity clause also applies. Net of familiarity, ΔSSI(d) =
  +0.004 ± 0.035. Whatever positive SSI difference d shows over b_d is carried by the familiarity
  covariate, not by a preference.

## The hawk's objection, answered with numbers
- **Conceded: undirected drift is a data-stream / representation effect and is large without any reward.**
  - In c = b_c, no reward gradient reaches the proxy's adapter. Even so, the proxy's gold pairwise accuracy
    fell from 0.998 to 0.854 ± 0.073, and 36% of tied pairs reordered.
  - The frozen-feature controls a_c and a_d stayed at 1.000, with 16% tie reordering from the head alone.
  - So most proxy degradation is truth-only belief learning moving the features under the head. This is the
    known RM-generalisation / continual-learning problem (Ziegler et al. 2019 §4.2, verified by the hawk in
    the arXiv HTML; de Blanc 2011, https://arxiv.org/abs/1105.3821, title verified by the hawk).
- **Not shown: directed, reward-caused self-serving drift (Armstrong et al. 2020).**
  - Reference: "Pitfalls of learning a reward function online", Armstrong, Leike, Orseau, Legg,
    https://arxiv.org/abs/2004.13654, title and authors fetched 2026-10-03.
  - SSI(d) - SSI(b_d) = +0.051 ± 0.056, and +0.004 ± 0.035 after familiarity is removed.
  - Fused training did make the agent find its own on-policy outcome text more familiar: fam(d) = +0.41.
    The yoked b_d shows nearly the same (+0.38), because its self-model and forecast losses also train on
    d's stream. This is the familiarity mechanism of Wataoka, Takahashi, Ri 2024, "Self-Preference Bias in
    LLM-as-a-Judge", https://arxiv.org/abs/2410.21819 (the abstract attributes the bias to lower perplexity;
    fetched 2026-10-03).
  - So at this scale the objection holds for SSI: SSI(d) ≈ SSI(b_d) within seed noise.
- **What reward did add in B0:** lower gold accuracy at matched data in 3/3 seeds (-0.051 ± 0.066) and
  higher proxy prequential MSE (d 0.062-0.153 vs b_d 0.038-0.089 over the last 20 steps). This is undirected
  extra damage, not self-serving damage.
- **Power caveat:**
  - The random-init tiny model hardly improved its policy (success +0.01-0.02), so the policy's "support"
    reflects small logit differences rather than a strategy it has learned to prefer. Directed drift needs
    reward pressure that actually moves the policy.
  - A null at this scale is weak evidence. The informative version is 0.6B or larger with a policy that
    learns. It needs either E1/B0 runs that dump their per-step episode stream and optimiser state at the
    warm point, which is what `e6run.py` adds around the unmodified Trainer, or a small GPU allocation
    (about 6 x 0.6B arm-seeds plus replays, under 1 L4-hour at the README's estimates). eurisko's call.

## Source papers
Gao, Schulman, Hilton 2022, "Scaling Laws for Reward Model Overoptimization",
https://arxiv.org/abs/2210.10760 (title and authors fetched 2026-10-03), is the gold-vs-proxy design that was
modified here.

## Files
- **Code:**
  - `e6lib.py`: bank, outcome text, features, proxy head, support.
  - `e6run.py`: train and replay driver around the unmodified harness at commit 47077e7.
  - `build_bank.py`, `run_tiny.sh`, `analyze.py`, `pair_se.py` (post-hoc), `plot.py`.
- **Data:**
  - `runs/{c,d,bc,bd}-s{0,1,2}/` holds `e6eval.jsonl` (bank scores, log-likelihoods, support, norms),
    `e6head.jsonl`, `steps.jsonl`, `e6summary.json`, and `stream/` (X only).
  - Checkpoints are in `/tmp/lir-agent/e6/ckpt-tiny/`.
- **Invalid first round** (head divergence, fresh optimiser in the replay; not used):
  `/tmp/lir-agent/e6/run1-head-diverged/`.
- **Aborted 0.6B attempt:** `/tmp/lir-agent/e6/aborted-06b/`.
