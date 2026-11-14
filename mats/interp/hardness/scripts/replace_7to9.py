"""Replace Sections 7-9 of draft.md with notes/sec7to9_v2.txt and fix Lemma 6.6's pairing condition."""
import sys
base = '/home/agent/workspaces/mats/work/hardness/'
p = base + 'draft.md'
d = open(p).read()
new = open(base + 'notes/sec7to9_v2.txt').read()
new = new.replace('[POISSON_NUMBERS]', 'orders 1-5 keep 99.7% at alpha = 1/4 and 61% at alpha = 1')
a = d.index('## 7. The low-degree method')
b = d.index('### 6.3 Measured')
d = d[:a] + new + d[b:]
old = "whose domains contain the G_l and satisfy (T_l Q)(g) = Q(T_l^* g), and outputs mu-hat = Q_L(G_L),"
if old in d:
    d = d.replace(old, "whose domains contain the G_l and satisfy (T_l Q)(g) = Q(T_l^* g), with all pairings below finite, and outputs\nmu-hat = Q_L(G_L),")
else:
    print('lemma 6.6 string not found', file=sys.stderr)
open(p, 'w').write(d)
print('ok')
