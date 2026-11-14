"""Replace the text between two markers of the outbox report with the content of a file.
usage: python splice.py START_MARKER END_MARKER NEWFILE"""
import sys
P = '/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md'
start, end, newf = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(P).read()
a = s.index(start); b = s.index(end)
s = s[:a] + open(newf).read() + s[b:]
open(P, 'w').write(s)
print('spliced', a, b)
