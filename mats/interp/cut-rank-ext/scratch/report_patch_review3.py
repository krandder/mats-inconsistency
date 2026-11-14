"""Third batch of final-review fixes (findings 10, 11, 13): ARC3 relabelling, convergence check, both ARC versions,
review paragraph."""
p = '/home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md'
s = open(p).read()
rep = [
    ("- ARC3 (K = 3 from exact cumulants, width 8) is unstable: mean errors of 1e16 to 1e31 of Vbar, and 16-61% of\n  (front, back) pairs above 100 Vbar even after unit scaling (post hoc diagnostics in `scratch/nl_analyze.py`).",
     "- Before Amendment 1 the merged-front ARC2 error was 1.5e9 of Vbar (the variance-floor artifact); after unit scaling it\n  is 0.71. Elsewhere the two versions agree.\n- Convergence (Amendment 2, `logs/conv_check.log`): with cut moments from 65,536 instead of 8,192 directions, the ARC2\n  and K2X errors on 64 test fronts move by at most 0.002 of Vbar ((8,8,8): ARC2 0.91 against 0.92; (8,64,8): 0.76\n  against 0.77). So the K = 2 closure errors are not moment-estimation noise.\n- ARC3: the registered runs omitted the trace of the fourth cumulant, which Wu et al.'s basic K = 3 needs, so they are an\n  incomplete-input variant. Rerun with it (Amendment 2), ARC3 is unstable on these fronts: median per-pair errors of\n  9.5 to 7.7e6 Vbar, and 32-88% of (front, back) pairs above 100 Vbar. Whether the closure or amplified\n  moment-estimation noise causes this is not separated."),
    ("- **Final review:** REVIEWPLACEHOLDER",
     "- **Final review** (GPT-6 = gpt-6-astra in a fresh session, high effort, 12:00-12:09Z; `review/review_gpt6_final.md`).\n  - Verdict: major revision, 15 findings (11 major, 4 minor). It recomputed all four NC1 configurations, the NC2 slope\n    and the NC3 counts from the raw files, verified both PREREG hashes, and found the central\n    communication-against-computation conclusion intact.\n  - All findings were applied:\n    - (1) theorem domains and budgets restated; (2) the analytic upper bound restricted to radius ratio 1/2; (3) the C^k\n      upper bound from astra's Cantor code (mine gives only C^{k-1,1});\n    - (4) exactness and quantifiers made result-specific; (5) non-adaptivity added to the frame and atlas bounds;\n      (6) section 5 reworded;\n    - (7, 8) coverage map corrected: split variants, conditional dimension bounds, missing rows, \"among the sources\n      inspected\";\n    - (9) NC2 rerun with a global norm code (same result); (10) ARC3 rerun with the fourth cumulant (still unstable);\n      (11) \"exact moments\" corrected and a convergence check added;\n    - (12) SVD numbers relabelled as optimistic estimates; (13) pilot scoring and both ARC versions disclosed; (14) NC3\n      descriptions corrected; (15) the summary's scope restored.\n  - REVIEW2PLACEHOLDER"),
]
for a, b in rep:
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('patched', len(rep))
