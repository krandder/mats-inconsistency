"""Assemble the deliverable: head (summary) + formal sections from draft.md (6.3/6.4 moved into Section 6) + tail.
usage: assemble.py HEAD DRAFT TAIL OUT"""
import re, sys
head, draft, tail, out = sys.argv[1:5]
d = open(draft).read()
d = re.sub(r'^# DRAFT.*\n', '', d)
# move the trailing "### 6.3 ..." and "### 6.4 ..." blocks to just before "## 7."
m = re.search(r'\n### 6\.3 .*', d, re.S)
if m and d.find('\n## 7.') < m.start():
    block = d[m.start():]
    d = d[:m.start()]
    i = d.find('\n## 7.')
    d = d[:i] + block.rstrip('\n') + '\n' + d[i:]
open(out, 'w').write(open(head).read().rstrip('\n') + '\n\n' + d.strip('\n') + '\n\n' + open(tail).read())
print('wrote', out, len(open(out).read()), 'chars')
