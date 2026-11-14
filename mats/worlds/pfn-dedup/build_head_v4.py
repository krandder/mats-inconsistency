"""Compose the v4 report head: new answer, v3 answer as a superseded block, v3 status, split section.

usage: python build_head_v4.py REPORT
"""
import sys

rep = sys.argv[1]
s = open(rep).read()
title = s[:s.index('## Answer')]
v3 = s[s.index('## Answer'):s.index('## Status')]
v3_body = v3.split('\n', 1)[1].strip()
rest = s[s.index('## Status'):]
status, tail = rest[:rest.index('## 1. Accuracy')], rest[rest.index('## 1. Accuracy'):]
split = ("## 0. Prior-level vs in-context split (Kelvin's distinction; references preregistered before computing)\n\n"
         "R1 is the best additive pool, mu = a_n + sum_r g_n(o_r): it knows the row count and each query token's own "
         "value, never compares tokens, and is fitted on 300k fresh training episodes. R2 is exact Bayes given only "
         "the query readings, with no identities and no labelled residuals, under a duplicate-blind model in which "
         "all rows are distinct sources. A tie is scored like any two readings 0 apart, so R2 cannot use exact value "
         "equality as copy identity, though it does compare values. R2eq drops exact duplicate values first, so it "
         "leaks copy identity; it is shown only as a bracket. Code: split.py, split_copy.py, split_tables.py.\n\n"
         + open(sys.argv[2]).read() + "\n")
new = (title + open(sys.argv[3]).read() + v3_body + "\n\n</details>\n\n" + status + split + tail)
open(rep, 'w').write(new)
