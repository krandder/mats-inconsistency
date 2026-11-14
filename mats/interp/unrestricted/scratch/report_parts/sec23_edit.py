"""Fill section 2.3: cost-to-precision table and the families table rows (run after the queue)."""
import re
p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
T = open('/home/agent/workspaces/mats/work/unrestricted/results/eval_tables.md').read()
i = T.index('## Cost to precision'); j = T.index('## Scores')
ctp = '\n'.join(l for l in T[i:j].splitlines() if l.startswith('|'))
top = open('/home/agent/workspaces/mats/work/unrestricted/scratch/report_parts/sec23_top.txt').read().replace('TABLE_CTP', ctp)
a = s.index('### 2.3 Cost against width and depth, and what could scale')
b = s.index('| Family | Deductive in ARC')
s = s[:a] + top + '\n' + s[b:]
old_row = s[s.index('| Certified B&B over input cones (this lane) |'):]
old_row = old_row[:old_row.index('\n')]
new_row = ('| Certified B&B over input cones (this lane) | bounds and "mid": yes; "point": no | a cell costs 46-170 passes '
           '(n^3 L FLOPs for bounds); passes to nMSE 1e-4 grow about 10x per unit of width (table above) | only at width 3, '
           '4 of 24 runs, fragile under a 2x cost change | no |')
s = s.replace(old_row, new_row)
ml_row = ('| Multilevel reference-trajectory sampler (astra, 3.7) | no (structured sampling; deterministic via 3.1) | '
          'passes, as MC; prefix levels cost r/L of a pass | grows with depth: break-even at L <= n/4, 1.5x at L = n = 128, '
          '15x at width 8 and depth 256; proven Omega(n / log^2 n) at L = alpha n | yes in cost; the gain needs depth >> width '
          'or very large width |')
anchor = s[s.index('| Exact enumeration of linear cones (3.5) |'):]
anchor = anchor[:anchor.index('\n')]
s = s.replace(anchor, anchor + '\n' + ml_row)
open(p, 'w').write(s)
print('ok')
