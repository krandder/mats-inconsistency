"""check_firewall.py : no arm-side file imports exact inference or reads an exact reference."""
import re
arm = ['cgen.py', 'cmodel.py', 'ctrain.py', 'cdirect.py', 'mkjobs.py', 'ctrain_ph.py', 'cgen_c1b.py', 'cref.py']
bad = []
for f in arm:
    s = open(f).read()
    code = '\n'.join(ln.split('#')[0] for ln in s.split('\n'))
    code = re.sub(r'""".*?"""', '', code, flags=re.S)
    for pat in (r'^\s*(from|import)\s+exact\b', r'exact[^\n]*\.npz', r'U_part', r'pstar', r'cexact', r'summary\.json'):
        if re.search(pat, code, flags=re.M):
            bad.append((f, pat))
print('arm-side files checked:', ', '.join(arm))
print('violations:', bad if bad else 'none')
