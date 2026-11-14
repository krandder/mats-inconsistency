"""Replace the report's head (title through the Status section) with HEAD_MD.

usage: python splice_head.py REPORT HEAD_MD
"""
import sys

rep, head = sys.argv[1], sys.argv[2]
s = open(rep).read()
b = s.index('## 1. Accuracy by depth')
open(rep, 'w').write(open(head).read() + '\n' + s[b:])
