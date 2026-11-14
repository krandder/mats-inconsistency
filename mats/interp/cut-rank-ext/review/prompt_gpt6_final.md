You are GPT-6 acting as a fresh, adversarial reviewer (not a co-author) of a theory report and its evidence. Read-only
sandbox; do not modify files. The report is /home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md.
Its work directory is this directory (cut-rank-ext): FORMAL-ext.md (the author's lemmas with two addenda),
coprover/ (statement, prompts, gpt-6-astra's two answers, PROVENANCE.md), notes/ (arc_estimators.md: sourced fact sheet
of ARC's estimators; lit_widths_comm.md: literature; coverage_map.md), PREREG.md and prereg_hashes.txt, scratch/ (code),
runs/ (data: nl_*.npz, *_scores.json, *_arcunit.npz, nc3_*.json), logs/, tests/. Earlier context, if needed:
../cut-rank/ (the previous lane: FORMAL.md, coprover/, review/) and the outbox report
/home/agent/workspaces/mats/outbox/cut-rank-lower-bound-2026-10-30.md.

Check, in this order, and report numbered findings (severity critical / major / minor; location; problem; fix):
1. Every mathematical claim in the report's sections 2-5 and 8-9: is the statement correct as written, are its
   hypotheses (budgets, encoder/decoder regularity, adaptivity, quantifiers) stated, and is the label right? Give extra
   attention to claims NOT checked by the co-prover: FORMAL-ext Addendum 2 (analytic upper bound by multilinear
   decoding of quantised weights; amplitude 3^B), C3 (exact space-filling protocol), Theorem D, D1, D2, Section 5
   (streaming reduction), and the summary's one-line versions of each result. Check also the co-prover's results the
   report relies on: the n log2(1/eps) bit bound, the n(n+1)/2 chart, finite moment determinacy, the fixed-budget floors,
   the C^k (k+1)p construction, and the exact-input bias floor for ARC's closure (including whether the family
   alpha_t = X_+ + t(-X)_+ is realisable on the K5 event and whether the closure is analytic in t).
2. The coverage map (report section 7 and notes/coverage_map.md) against notes/arc_estimators.md: wrong
   classifications, overstated coverage, missing ARC estimators, misattributed sources.
3. The numbers: recompute a sample of the NC1/NC2/NC3 entries in the report from runs/ (the scores JSON and the npz
   files; scratch/nl_analyze.py is the scorer), and check the scorecard against PREREG.md as hashed (and its
   Amendment 1). Flag any unreported deviation, any post hoc analysis presented as registered, and any number that does
   not match.
4. Whether the summary for Kelvin is faithful to the body: overclaims, missing caveats, wrong attributions.
5. A verdict (accept / minor revision / major revision) with the three most important fixes.
Be concrete and concise. Write everything to stdout.
