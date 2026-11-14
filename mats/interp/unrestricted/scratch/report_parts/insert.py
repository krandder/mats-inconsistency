"""Insert or replace report sections from text files (keeps edits out of shell heredocs).
Usage: insert.py before <marker> <file>        insert the file's text just before the marker line
       insert.py replace <start> <end> <file>  replace the text from start marker up to (not incl.) end marker"""
import sys
REPORT = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(REPORT).read()
mode = sys.argv[1]
if mode == 'before':
    marker, fn = sys.argv[2], sys.argv[3]
    i = s.index(marker)
    s = s[:i] + open(fn).read() + s[i:]
elif mode == 'replace':
    start, end, fn = sys.argv[2], sys.argv[3], sys.argv[4]
    i = s.index(start); j = s.index(end)
    s = s[:i] + open(fn).read() + s[j:]
open(REPORT, 'w').write(s)
print('ok', len(s))
