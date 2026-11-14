---
title: "Traders inside a neural network (traders-net study, 2026-09-27/28)"
status: reference
freshness: checked 2026-09-28 (re-run the check below; each status line names its evidence)
---

# Traders inside a neural network

Kelvin and Jessica Taylor asked whether "thousands of traders" can live inside one neural network:
units with wealth, settled like a logical-induction market, sharing computation (MoE-like). The
eurisko study (56 agents planned, 2026-09-27/28) built a small test bed on the logical-soup question
streams and ran the plan in `traders-net/STUDY-PLAN.md`. This page is the plain-language result.
Workspace: `/home/agent/workspaces/eurisko-study/traders-net/` (on General).

**Draft, not pushed.** Written for a hub PR that adds `wiki/traders-net.md`.

## The main finding: attribution flow is not settlement

Backprop credit (gradient x activation, summed over training) cannot serve as a trader's wealth.
It disagrees in sign with market settlement on exactly the units that matter: a unit whose signal
the network already relies on gets zero or negative gradient credit, while removing it hurts.

- **Toy, provable (M1a).** For a unit that enters the logit linearly, gradient credit is
  C = -w L'(w) and its value is V = L(0) - L(w). Convexity gives V >= C; at the optimum C = 0 while
  V = I(X;Y) = 0.1109 nats, and C < 0 < V for weights in (1, 2.1786) (23/23 grid points). A pure-noise
  unit gets the same gradient credit as useful units (-78.5 vs -74.5 on fresh seeds), while exact
  settlement separates them (-39.5 vs +466). Evidence: `study/M1/lead/results/toy_summary.json`,
  `study/X1/w2/results/R1_toy_fresh.json` (claims M1-a1, M1-a-noise).
- **In the trained net (M1b).** Among the 5% of neurons that matter most (by held-out ablation),
  25.6% +- 5.9% have negative all-time book credit; with exact leave-one-out settlement only
  2.6% +- 2.2% are negative. Evidence: `study/M1/lead/results/sign_census_summary.csv` (M1-b-books,
  M1-b-settle).
- **What does work.** Exact leave-one-out log-score settlement ranks neurons against the ablation
  oracle at Spearman 0.417 +- 0.034 (fresh held-out segment 0.441 +- 0.022); the cumulative book
  gets 0.043 +- 0.058 and gradient norm 0.168 +- 0.07. Evidence: `study/M3/results/currency_summary.csv`,
  `study/X1/w2/results/R2_rederive_fresh.json` (M3-A-settle, M3-A-cum, M3-A-gradnorm).

Prior art for the mechanism: at a trained minimum the first-order term vanishes, which is why
Optimal Brain Damage prunes with second-order saliency
(https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) and Taylor
pruning squares the first-order term (https://arxiv.org/abs/1906.10771). What is new here is the market
reading (books cannot be wealth; a settled bet can) and the size of the effect on a trained net. The
literature search (`study/L1/NOVELTY.md`) ran before this finding and did not look for it specifically.

Consequence, locked for the study (`study/M1/CURRENCY.lock`, 2026-09-28T01:40Z): no bookkeeping
currency for generic neurons. A trader must be a designed object that commits to a price or a
log-odds correction on a defined question, with a bounded stake, and is paid the exact LMSR log-score
gain against the price at trade time when the label arrives.

## Answers to the eight questions (verified rows only; ids point into `study/X1/CLAIMS.md`)

1. **Currency.** Settled books do not beat |Taylor| or movement pruning on random coalitions
   (coalition_win = false, M2-gateB); no book currency beats gradient norm (M3-A-stop). Exact
   leave-one-out settlement does beat both (above), so "wealth" is useful only as settlement, not as
   bookkeeping.
2. **Novelty.** Fixed-trader markets are Bayes/Hedge, "backprop is a market" is an old analogy, and
   MoE load balancing is already a price (DeepSeek). The open part is the combination: a
   logical-induction-style market whose traders are the SGD-learned, computation-sharing units of one
   network, measured for exploitability before resolution (`study/L1/NOVELTY.md`; novelty itself is
   unverified: no counter-example found, not a systematic search). Nearest misses: Hedge
   Backpropagation https://arxiv.org/abs/1711.03705 and continual backprop https://arxiv.org/abs/2306.13812.
3. **Induction.** The net is not coherent before resolution: violation rates 0.477 / 0.545 / 0.536
   against 0.5 for an uninformed market (M5-6). It is exploitable by an online probe on every fresh
   seed, but the quoted magnitudes did not reproduce (M5-1, unverified).
4. **Humility.** Ten market-layer arms survive the base stop rule (base_arms_survive = true, M6-1);
   the stress pass keeps b, cq and fb05. The surviving pools win only by about 0.001 nats of noise
   protection (M6-12). Committed trader heads (each head commits a correction and is settled)
   slightly beat the uncommitted layer (BCE -0.3% to -1.3%, 3/3 seeds), but their wealth does not
   track marginal value (Spearman -0.0016 +- 0.0221, M6-10).
5. **MoE.** A per-family wealth prior on top of DeepSeek's bias lowers loss by 1.30% (8 experts) and
   1.87% (64 experts), 3/3 seeds (M7-2), but is worse after a regime shift, by 0.041-0.092 (M7-4), and
   needs family labels. Wealth alone collapses routing: Exp3 max load 0.775 vs DeepSeek 0.368 (M7-5);
   a global wealth prior sends everything to one expert (M7-6). Verdict: DeepSeek already built the
   price (wealth_beats_deepseek = false, M7-11).
6. **Lock-in.** Fixed share 1e-3 is the currency that recovers after a shift; it beats Kelly-
   multiplicative wealth by >= 0.05 in 3/3 seeds, at a stationary cost of 0.001 (M3-B-decision). Gating neurons by income hurts against no gate
   (M8-none-better); the |a.g| gate with rebirth did not beat random reinitialisation (stop).
7. **K/Q law.** The soup's law is a lottery over traders per question: Brier = 0.25 -
   D(1-(1-pi)^{K/Q}), R^2 0.79-0.94 (M9a-lottery). In the net, K/Q from 1 to 64 shows no trend
   (M9c-notrend): the law is about search, not markets.
8. **Economy.** Without niches one species holds >= 99.7% of the wealth (Blume-Easley selection,
   K1-niches-off); with niches every slot stays alive and the market reaches oracle Brier
   (K1-niches-on), with a Gini of 0.44 against a shuffled-belief null of 0.74 (K1-gini). Money is
   conserved to 4.1e-12 (K1-conservation). A compute price only destroys (K1-price).

## What did not hold up (from the reproduction pass)

- The pi family leaks: 62.9% of seeds put two pi questions within the model's 32-digit window, so
  "pi stays at ln 2" is true only for lucky seeds (S0-X-pi refuted; `study/X1/w1/results/pi_offsets.json`).
- Three-seed standard deviations understate seed variance (5-seed sd a median 1.7x larger; up to 18x).
  Signs and orderings survive far more often than magnitudes.
- "Delay 64 is best" (S0-3a/b/c) and the K1 shift-recovery numbers are refuted as quoted.
- Whether a run learns LFSR16 decides many BCE comparisons; claims with LFSR16 removed reproduce.

## Not done (open)

- The block-local architecture test (end-to-end vs block-local vs block-local with leave-one-out
  wealth) from the 01:00Z addendum was not run inside this study (`study/M6/M6.md`, "Not done in this
  pass"). The wealth-net lane ran a first version, 3 seeds, not checked by the reproduction pass
  (`eurisko-study/wealth-net/RESULTS.md`, "Block-local settlement"): block-local settlement agrees
  with the ablation oracle (Spearman +0.857 vs +0.395 end-to-end), and the market arm's robustness comes
  from skipping and rebirth, which a random-skip control matches.
- Heads inside the trunk or MoE experts as committed sub-market traders; T1 lemmas 8 (sign
  conditions) and 9 (price of anarchy) are not in `study/T1/THEORY.md` (the M1a toy covers the sign
  case).

## Status (checkable)

Run `python3 study/D1/w2/check_wiki_claims.py` and `python3 study/D1/check_ideas.py` from the workspace (read-only; writes
`study/D1/w2/wiki-check.json`). Checked at 2026-09-28T14:07Z, all rows hold (`study/D1/w2/wiki-check.json`); X1's own `check_claims.py` re-run by D1 on a scratch copy at 14:02Z passes (`study/D1/x1-recheck-check_claims.json`):

| claim | check | result at check time |
|---|---|---|
| Claims register: 209 rows, 153 verified, 19 refuted, 37 unverified | `study/X1/CLAIMS.md` header + `study/X1/lead/results/check_claims.json` ok | holds |
| Gate flags: currency_locked true, beats_baseline false, coalition_win false, wealth_beats_deepseek false, base_arms_survive true | `study/X1/lead/gates.json`, `study/M1/CURRENCY.lock`, `study/M6/lead/M6-verdict.json`, `study/M7/merge/gate.json` | holds |
| GPU spend $6.07 of the $45 cap | `study/G1/COST.md` total line (ledger equals GCE receipts, X1 row G1.2) | holds |
| No traders-net GPU VM, disk or running job left | `gcloud compute instances/disks list` in GCP_PROJECT, filter tn-gpu-* or label traders-net; `study/G1/queue/running/` empty | holds (0/0/0) |
| Grafana dashboard `traders-net` loads, all 15 panels PASS (panel 2 is a table, skipped) over 165 runs | copy of `study/V1/check.py`, output `study/D1/w2/grafana-check/latest.json` | holds |
| Page http://[ip-removed]:8765/traders-net/ is served | HTTP 200, title "Traders Inside a Net" | holds |

Links: dashboard [removed] memo page
http://[ip-removed]:8765/traders-net/; wealth-net page http://[ip-removed]:8765/wealth-net/.
Related: [[consistency-ft]] (Exploit LP), the logical soup (`eurisko-study/logical-soup/REPORT-SOUP.md`).
