"""Replace section 10 (from its heading to the end of the file) of the outbox report with notes/sec10.txt."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = '/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md'
NEW = os.path.join(HERE, '..', 'notes', 'sec10.txt')
s = open(REPORT).read()
marker = '\n## 10. Follow-up'
i = s.find(marker)
assert i > 0 and s.count(marker) == 1, 'section 10 marker not found exactly once'
new = open(NEW).read()
assert new.lstrip('\n').startswith('## 10. Follow-up')
out = s[:i].rstrip('\n') + '\n\n' + new.lstrip('\n')
open(REPORT, 'w').write(out)
print('replaced; report now', out.count('\n'), 'lines')
