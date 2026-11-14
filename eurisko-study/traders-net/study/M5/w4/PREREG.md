# M5 worker 4 PREREG: online MLP probes as class-C shadow traders

Written 2026-09-28 ~02:05Z before any run.

## Class (item 4 of M5)
Online MLP probe traders. Each probe reads (features, price) for one (question, time) and emits a
log-odds correction d; belief q = sigmoid(logit p + d). Last layer zero-initialised, so a probe starts
as a price-follower (humble by construction). One probe is shared across all 64 questions (family
one-hot is an input). Two hidden ReLU layers of width h in {8, 32, 128} (sizes S, M, L).
Update: prequential, predict all 64 questions at time t, score, then one Adam step (lr 1e-3) on
the log loss of q at time t. Feature sets:
- `win`: the question's 32-bit window (+-1), the same per-question input the net sees;
- `hid` (wealth-net source only): the net's final LayerNorm output for that question (d=64), detached.
Both also get logit p (clipped to +-8) and the family one-hot.

## Markets (price sources)
1. `logistic`: per-question online logistic model (wealth-net's baseline: Adam 1e-2, W=32), run
   prequentially over the whole wealth-net stream (Q=64, T=24000); probes trade from t=0.
2. `wn@<step>`: frozen wealth-net checkpoints (train-s{1,2,3}, steps 8000, 16000, 24000, final
   31100), prices on the held-out region (positions >= 0.8*n_pos, ~4.8k steps, in time order).
   Note: wealth-net trained multi-epoch on positions < split, so this is a held-out test.
3. `wn-random`: untrained init (random-s*.pt), positive control (an exploitable market).

## Metrics
- E_T(t) = sum_{s<=t, q} log q_T(y)/p(y) (nats, summed over questions), per probe; E_C = max_T E_T.
  Shadow traders (wealth -> 0) do not move the price, so clearing = as traded for the shadow runs.
- Slopes via S0 `panel.ec_slopes` (loglog exponent, linear slope over the last half, final).
- Invasion fitness phi_T = last-half slope of E_T per question-step (nats/question/step).
- Per-family E_T at the end (the pi family is the in-stream null: expect phi <= 0).
- Member run at eps=1e-3 (one 2-trader Kelly market per question: net/logistic belief with wealth
  1-eps, probe L with eps), fixed-point clearing p* = (1-eps) p0 + eps q(p*) by bisection (40 iters).
  Report member E (bound ln(1/eps)=6.908 per question-market), final probe wealth share, and
  market log-loss change vs p0. E reported both at p* (clearing) and belief-at-p0 scored at p* (as traded).
- Control: shuffled-label probe (probe trained and scored against labels permuted within each
  question over time, and price scored on the same permuted labels): expected phi <= 0.

## Seeds
3 seeds; seed s pairs stream/checkpoint seed s with probe init seed s. Mean +- sd (sample sd).

## Hypotheses (expected, from T1 lemma 5 and S0 validation)
- H1: E_C slope > 0 on logistic (it cannot see across questions; AND/OR compounds and CA are exploitable).
- H2: E_C on wn@final smaller than on logistic per question-step; E_C decreases with checkpoint step.
- H3: probe size: larger probes exploit more on logistic (more capacity) but lose more on pi early.
- H4: member E <= ln(1/eps) always (theorem 5.1 check).

## Stop rule
Harness, no kill rule (STUDY-PLAN). Wall cap 4 h CPU; at most 3 concurrent jobs, 1 torch thread each,
launched only when `uptime` 1-min load < 40. If the wealth-net held-out region is too short for
the probes to leave the warm-up (all phi within 1 sd of 0 on wn-random, the positive control),
report the harness as underpowered there rather than extend.

## Amendment A1 (before the first run, 02:10Z)
The shuffled-label control is replaced: with a fixed net price, permuted labels make the price
miscalibrated against y' and the probe earns by learning base rates, so it is not a null. Controls now:
(i) the pi family inside every run (true noise, expect phi <= 0); (ii) a `price-only` probe of size M
(window/hidden features zeroed, keeps logit p and family one-hot): this is a smooth calibration
trader, so probe minus price-only = the part of the edge that needs features.

## Amendment A2 (02:15Z, before these arms run)
Add source `preq-P4` / `preq-P16`: S0 worker 1's prequential nets (single pass, test-then-train, B=4
and B=16; trade-time logits `z` in /tmp/traders-net/S0/w1/P{4,16}-s{1,2,3}.npz, 64000 positions x 64
questions, wealth-net streams). Features `win` rebuilt from the label history (y[t-32..t-1]; zeros before
t=32). This is the net as a live market (prices before training on the label), the setting LI is about.

## Amendment A3 (02:11Z, before any full run)
The load gate (<40) never opened: 1-min load stayed 47-69 for 25 min, driven by other lanes. Gate dropped;
cap kept at 3 concurrent single-thread jobs at nice 10 (run_all.sh now includes the preq arms). Timing
(seconds) is therefore not comparable across arms.

## Amendment A4 (02:32Z, before any preq arm starts)
Load 90+; a 64000-step preq run would take over an hour each. preq arms use the first 32000 positions
(P4: net trained on 8000 batches by then; P16: 2000). Prices are clipped to |logit| <= 8 in every arm
(CLIP in probes.py), so pi's memorised held-out logits enter the market at most at 8 nats.
