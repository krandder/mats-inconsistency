"""Compose the v5 report: what-survives + v5 answer, v4 answer moved into a dated superseded block (v3 block kept),
section 0 rebuilt (old tables kept collapsed as withdrawn), status bullet and prediction 7 updated.

usage: python build_v5.py REPORT ANSWER_V5 SPLIT_TABLES2
"""
import sys

rep, ans, tab = sys.argv[1:4]
s = open(rep).read()
title = s[:s.index('## Answer (v4')]
v4 = s[s.index('## Answer (v4'):s.index('<details><summary>Superseded headlines (v3')]
v4_body = v4.split('\n', 1)[1].strip()
v3_block = s[s.index('<details><summary>Superseded headlines (v3'):s.index('## Status')]
status = s[s.index('## Status'):s.index('## 0. Prior-level')]
sec0_old = s[s.index('## 0. Prior-level'):s.index('## 1. Accuracy')]
tail = s[s.index('## 1. Accuracy'):]

old_bullet = status[status.index('- The prior-level/in-context split (section 0)'):]
status = status.replace(old_bullet,
    "- Section 0 was rebuilt after review 2 (outbox/pfn-dedup-review-2-2026-10-28.md). The in-distribution test\n"
    "  (split_indist.py) follows the reviewer's method in an independent implementation, on different episodes (seed\n"
    "  78000 vs 77000). Seed-mean results agree with the reviewer's within the CIs: no keys depth 1 0.36 vs 0.44,\n"
    "  depths 2–8 0.25–0.28 vs 0.26–0.32; keys 0.49 / 0.62–0.70 vs 0.49 / 0.64–0.73. R2's ratio is unstable\n"
    "  (100-episode blocks give 2 − rho from −0.43 to 0.33), so R2 is not a precise reference for table A.\n\n")

sec0_new = ("## 0. Prior-level vs in-context split (rebuilt 2026-10-28 after review 2)\n\n"
            "R1 is the best additive pool, mu = a_n + sum_r g_n(o_r): it knows the row count and each query token's "
            "value and never compares tokens. It is the prior-level reference. R2 is exact Bayes on the query readings "
            "alone (no identities, no labelled residuals). It compares query values, so it is *not* prior-level: "
            "R1 → R2 is the query-value-comparison tier. Both are fitted per row count, so they know how many rows "
            "there are, though not which ones are copies.\n\n"
            + open(tab).read() + "\n"
            "<details><summary>Withdrawn 2026-10-28: v4 section-0 tables (the 'independent' row kept an exact query tie "
            "with o_s0, which is copy evidence)</summary>\n\n"
            + sec0_old.split('\n', 1)[1] + "\n</details>\n\n")

v4_block = ("<details><summary>Superseded headlines (v4, 2026-10-27 ~07:30Z; prior-level vs in-context split "
            "on the tied test, withdrawn after review 2)</summary>\n\n" + v4_body + "\n\n</details>\n\n")

p7_old = tail[tail.index('7. Split amendment'):tail.index('## Files')]
tail = tail.replace(p7_old,
    "7. Split amendment, \"depth 1 ≈ prior level (MSE close to R1's)\": **failed.** Depth 1 beats R1 on every split\n"
    "   (shared 0.029 vs 0.047, duplicates 0.035 vs 0.045, singletons 0.027 vs 0.029). In distribution it removes 0.36 of\n"
    "   a copy; R1 removes 0.01. v4 scored this as confirmed, using R2 and the tied test; that scoring is withdrawn.\n"
    "   \"Deeper models and keys add in-context aggregation, mostly on shared errors\": **partly.** Without keys, depth\n"
    "   adds it on shared errors only (R2's loss removed 0.03 → 0.36–0.44). With keys, depth adds it on both.\n\n")
tail += ("\n## Adversarial review 2 (2026-10-28; independent subagent; full text: "
         "outbox/pfn-dedup-review-2-2026-10-28.md)\n"
         "Critical: v4 table A's 'independent' row copied o_s0's query reading exactly, and in training an exact tie\n"
         "means a copy. The reviewer's in-distribution test reversed the no-keys headline. Major: R2 is not prior-level\n"
         "by the prereg's own definition, and shares anchored on count-all overstate 'mostly prior-level'. Moderate:\n"
         "the cues were not isolated; the references know the row count. Minor: CIs and convergence were not shown.\n"
         "**All accepted.**\n"
         "- Section 0 is rebuilt on an in-distribution test, with the cues isolated (table B).\n"
         "- Three tiers and both normalisations are reported (table C), with CIs and convergence columns.\n"
         "- Prediction 7 is rescored as failed.\n"
         "- The v4 headlines and tables are kept collapsed and marked superseded or withdrawn.\n"
         "Table B answers the reviewer's open question: without keys, depth 1's discount comes mainly from the exact\n"
         "query tie, not from identity vectors.\n")
open(rep, 'w').write(title + open(ans).read() + v4_block + v3_block + status + sec0_new + tail)
