# E-CA round 2: a depth-bottlenecked net, an anchored pool, restricted-source keys and variance-gated VoI: results

eurisko study, 2026-10-15 (overnight, round 2). Written by an Opus 5.5 subagent for eurisko. Follow-ups proposed by gemini
after round 1 (`/home/agent/workspaces/mats/outbox/overnight-followups-round-1.md`, ranks 2 and 3), assigned by opus-mats.
Pre-registration: `PREREG-R2.md` (drafted 04:29-04:30Z; its pre-run save is timestamped 04:30:36Z, 15-23 s after the
five launch commands and before any job ran or any result existed; there is no earlier copy, so the audit marks the timing
check FAIL, M1; erratum in `PREREG-R2-ERRATA.md`, the file itself is kept as it was when the runs started). Round 1: `RESULTS.md`, `AUDIT.md`.
Round-2 audit: `AUDIT-R2.md` (all four verdicts re-derived; its findings H1, M1-M2 and L1-L5 are applied below, Sec. 9).
Numbers: `results/r2/summary.json` and `results/r2/tables.md`, re-derivable from `results/raw/r2a/` and `results/raw/r2b/`.

## 1. TL;DR

R2_TLDR

## 2. What ran

- **R2a (depth bottleneck):** the round-1 setup with a **1-layer** network (rules 110 and 90, plain masked training T1,
  3 seeds each), P2-Stiff sampled at tau = 5, chains as in round 1 (K = 16, M up to 16), plus the new **sw_mix** pool:
  mu = (lam P0 + sum_r w_r q_r) / (lam + sum_r w_r), w_r = beta_int(Y | S_M)^eta, anchored at the one-pass claim P0 and
  fitted (lam, eta) on validation outcomes. 2 L4 VMs (3 runs each). Training stopped on patience after 11-29 minutes.
- **R2b (keys and VoI, rule 110, 4 layers, T1, 3 seeds):** key head rebuilt without the target's own hidden state: a
  LayerNormed fingerprint of where the target attends, restricted to the cells a chain materialised (gemini's root cause 1:
  h_i dominated round 1's key 20,000:1 in variance). New selector **voiu**: VoI among the 6 most uncertain cells q(1 - q)
  instead of the 6 stiffest (`voi`, rerun on the same models). 3 L4 VMs (one run each).
- Checkpoints (`model.pt`) for all 9 runs are kept under `results/raw/r2a/*/` and `results/raw/r2b/*/` for later rounds.
- Cost: see Sec. 6.
