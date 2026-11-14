"""check_firewall.py -- the arms never import the exact-answer modules. -> out/check_firewall.out"""
import ast, sys
HERE = '/home/agent/workspaces/mats/work/deep-study/wf13/loop/'
ARM = ['pw.py', 'ptree.py', 'ppol.py', 'ploop.py', 'pdata.py', 'prun.py', 'sk.py', 'spol.py', 'srun.py', 'mk_tables.py', 'pscale2.py']
SCORE = ['pscore.py', 'sscore.py', 'pscale.py', 'pilot1.py', 'pilot2.py', 'spilot.py', 'sdebug.py', 'srescore.py', 'sdiag.py']
BAD = {'pexact', 'sexact', 'exact', 'oracle', 'evalib'}
ok = True
lines = []
for f in ARM + SCORE:
    try:
        tree = ast.parse(open(HERE + f).read())
    except FileNotFoundError:
        continue
    imp = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imp |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imp.add(node.module.split('.')[0])
    hit = sorted(imp & BAD)
    if f in ARM:
        # prun.py and srun.py hand over to pscore / sscore only after training and search have finished
        ok &= not hit
        lines.append('%-14s arm      imports exact modules: %s' % (f, hit or 'none'))
    else:
        lines.append('%-14s scoring  imports exact modules: %s' % (f, hit or 'none'))
lines.append('FIREWALL %s' % ('HOLDS' if ok else 'BROKEN'))
open(HERE + 'out/check_firewall.out', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
