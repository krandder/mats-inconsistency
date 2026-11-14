# M5 lead merge: pre-registration (2026-09-28 ~04:20Z)

Worker preregs: `w1/PREREG.md` .. `w4/PREREG.md` (arms, seeds, stop rules of each class item). This file covers only
the merge, which runs no new trader experiment.

- **Arms.** None new. The merged class C = w1 calibration bins + w2 coherence arbitrageurs + w3 pattern-2/4/8 +
  w4 online MLP probes. Markets: prequential net P4/P16 and S0 logistic (wealth-net stream, S0 w1 npz), per-question
  logistic on the wealth-net stream (w4 only), frozen wealth-net checkpoints on the held-out tail, the S0 LI stream (w2),
  soup `sh_*` columns.
- **Metric.** The harmonised headline is invasion fitness phi in nats per question-step (best member of each item,
  late-window slope; w1's is at wealth eps = 1e-3), read from the workers' CSV/JSON, never retyped. w2's arbitrageurs
  are per constraint trader (2-3 legs) on the LI stream and stay in their own row. Coherence before resolution = w2's
  `pre_viol`.
- **Seeds.** The workers' 3 seeds (mean ± sd), unchanged.
- **One diagnostic run** (`exploit.py clipcheck`): held-out BCE of the final wealth-net checkpoints under each worker's
  price clip, 3 seeds, to reconcile four different "held-out BCE" values (0.69-1.07) reported for the same market.
  (A first pass on seed 1-2 ran at 04:15Z before this file was written; it only motivated the check.)
- **Stop rule.** None (harness, plan section 3). Done when `python3 exploit.py check` passes: every worker check or
  test green, EXPLOIT.md re-rendered from `results/headline.csv` with no drift, and every `runs/*/metrics.csv` has
  `ec_*` columns for V1.
