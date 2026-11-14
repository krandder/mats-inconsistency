"""Replace the report's NLL tables (from '## keys = 0' up to the ablation heading) with those in SUMMARY_MD.

usage: python splice_nll.py REPORT SUMMARY_MD
"""
import sys

rep, summ = sys.argv[1], sys.argv[2]
s = open(rep).read(); t = open(summ).read()
new = t[t.index('## keys = 0'):t.index('## Mechanism')]
a = s.index('## keys = 0'); b = s.index('### Context-to-context ablation')
open(rep, 'w').write(s[:a] + new + s[b:])
