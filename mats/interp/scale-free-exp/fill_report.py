"""Replace {{TABLE_*}} placeholders in the report with the matching sections of results/tables.md (mktables output).
usage: python fill_report.py REPORT TABLES [KEY ...]  (KEYs: W0 W1 W2 E2 M4)"""
import re, sys

rep, tab = sys.argv[1], open(sys.argv[2]).read()
keys = sys.argv[3:] or ['W0', 'W1', 'W2', 'E2', 'M4']
heads = {'W0': '### W0 test KL', 'W1': '### W1 test KL', 'W2': '### W2 test KL', 'E2': '### E2 cut cost',
         'M4': '### M4 '}


def section(h):
    i = tab.index(h)
    j = tab.find('\n### ', i + 1)
    body = tab[i:j if j > 0 else None].split('\n', 1)[1].strip('\n')
    return body


s = open(rep).read()
for k in keys:
    ph = '{{TABLE_' + k + '}}'
    if ph in s:
        s = s.replace(ph, section(heads[k]))
        print('filled', k)
open(rep, 'w').write(s)
