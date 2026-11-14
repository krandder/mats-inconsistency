"""Replace the Section 6.3 block of draft.md (from '### 6.3 Measured' to '### 6.4 Conjectures') with notes/sec63_v2.txt."""
base = '/home/agent/workspaces/mats/work/hardness/'
p = base + 'draft.md'
d = open(p).read()
new = open(base + 'notes/sec63_v2.txt').read()
a = d.index('### 6.3 Measured')
b = d.index('### 6.4 Conjectures')
d = d[:a] + new + d[b:]
open(p, 'w').write(d)
print('ok')
