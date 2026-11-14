"""Replace the generated-tables block of the report with a fresh tables.md.

usage: python splice_tables.py REPORT TABLES_MD
"""
import sys

rep, tab = sys.argv[1], sys.argv[2]
s = open(rep).read()
a = s.index('### Predictive mean vs the Bayes mean')
b = s.index('How to read the attention tables')
open(rep, 'w').write(s[:a] + open(tab).read() + '\n' + s[b:])
