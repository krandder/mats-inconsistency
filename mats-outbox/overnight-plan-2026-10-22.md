# Overnight plan 2, stage 1 learnability (opus-mats, 2026-10-22 from ~02:00Z until Kelvin is back)

Kelvin (verbatim): "Alright, go ahead and fix all the problems, in overnight runs and keep trying the ideas needed."
Challenge (Kelvin): a compressed consensus must learn the aggregation structure as well as an uncompressed transformer that reads everything, on the same data ("more like (1)"; "our challenge is well put; we just need to work harder on it").

## Source of truth
- The review: outbox/learnability-review-2026-10-22.md (bugs B1-B13; plan steps 0-8).
- Results log: outbox/overnight-results-2026-10-22.md.

## Owners
- **Compression builder (opus-mats subagent):**
  - step 0/B1/B9 (reframed table: blind references, provenance tags, symbol entropy);
  - step 1 (decoder-side keys);
  - B4 (fresh anchor, best-val);
  - step 3 (B6, B7, B10, B12, B8);
  - step 4 (count-leak-free Z9 knob);
  - step 5 (biased-bit knob and curriculum).
- **gemini:**
  - B5 (FCT relabel), B11 (unmatched comparisons), B13 (readout token);
  - step 2 (decoder-only on frozen raw_code);
  - step 6 (FSQ/VQ at b=2).
- **G(Sigma)/span subagent:** objective (1) per review section 4: Y-span residual with additive pooling, d_omega, bits including support pattern, SVD span, principal angles.
- **opus-mats:**
  - step 7 (securities as message) once steps 1-5 report;
  - step 8 (architectures) last;
  - gates.

## Rules
- **Gate.** Every result gets an adversarial gate before it is reported as more than "ungated": a design-fit reviewer with Kelvin's words, plus gemini as guardian.
- **Data.** Fresh episodes or a best-val checkpoint, and symbol entropy logged with every arm. Score against the fresh uncompressed anchor and the fair (blind) references.
- **Budget.** Kelvin's $300 authorisation. $171.30 had been spent (list price of deleted VMs) at 02:00Z, so the overnight cap is $260 total, keeping a reserve of about $40. Each run is capped at $20.
- **Compute.** Spot L4s via work/run_on_l4.sh, on-demand after preemptions. General is for CPU at 2 threads per job.
- **Check-in.** A fleet-now check-in about every 2 h.
- **Keep going.** When a step finishes, start the next. When the plan runs out, try the next idea that the results point to, and write down why. Stop only for a named blocker.

## Morning handoff
outbox/overnight-results-2026-10-22.md: per step, what was fixed, what each result means (in the review's step-5 terms), the current gap(b) against the anchor, the spend, and what to try next.
