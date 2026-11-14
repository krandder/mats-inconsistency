"""Write the pre-registered job lists (PREREG.md sections 4-5): jobs1.txt (E1, E3 from scratch), jobs2.txt (E2
fine-tunes; @OUT@ is replaced by the VM's run directory). One job per line: NAME ARGS."""
J1, J2 = [], []
W0 = '--world indep'
W1 = '--world groups --rho 0.5 --gsize 4'
W2 = '--world pairs --rho 0.9'


def add(name, args):
    J1.append(f"{name} sftrain.py {args}")


# heavy first: token nets at large N
for N in (24, 12, 6, 3):
    for s in (0, 1, 2):
        add(f'e1w0_tok_N{N}_s{s}', f'--arch tok {W0} --N {N} --seed {s}')
for N in (24, 16, 8, 4):
    for s in (0, 1):
        add(f'e1w1_tokg_N{N}_s{s}', f'--arch tok --gid {W1} --N {N} --seed {s}')
for N in (8, 6, 4):
    for s in (0, 1):
        add(f'e1w2_tok_N{N}_s{s}', f'--arch tok {W2} --N {N} --seed {s}')
# E3 routing variants (3 layers W0; 4 layers W1)
E3 = dict(A='dense:all,dense:all,dense:all', B='dense:self,uniform:src,uniform:all',
          C='dense:self,dense:src,dense:all', D='idroute:all,idroute:all,idroute:all',
          Bp='dense:self,uniform:src,dense:all')
for N in (12, 3):
    for s in (0, 1):
        for k, spec in E3.items():
            add(f'e3w0_{k}_N{N}_s{s}', f'--arch tok --L 3 --attn {spec} {W0} --N {N} --seed {s}')
E3G = dict(A4='dense:all,dense:all,dense:all,dense:all', G4='dense:self,uniform:src,dense:grp,uniform:all',
           G4u='dense:self,uniform:src,uniform:grp,uniform:all')
for N in (16, 8):
    for s in (0, 1):
        for k, spec in E3G.items():
            add(f'e3w1_{k}_N{N}_s{s}', f'--arch tok --gid --L 4 --attn {spec} {W1} --N {N} --seed {s}')
# set nets
for N in (24, 12, 6, 3):
    for s in (0, 1, 2):
        add(f'e1w0_set_N{N}_s{s}', f'--arch set {W0} --N {N} --seed {s}')
        add(f'e1w0_setnc_N{N}_s{s}', f'--arch set --nocross {W0} --N {N} --seed {s}')
for N in (24, 16, 8, 4):
    for s in (0, 1):
        add(f'e1w1_setg_N{N}_s{s}', f'--arch set --gid {W1} --N {N} --seed {s}')
for N in (8, 6, 4):
    for s in (0, 1):
        add(f'e1w2_set_N{N}_s{s}', f'--arch set {W2} --N {N} --seed {s}')
# hierarchical and tree nets
for N in (24, 16, 8, 4):
    for s in (0, 1):
        add(f'e1w1_tree32_N{N}_s{s}', f'--arch tree --m 32 {W1} --N {N} --seed {s}')
        for m2 in (4, 16):
            add(f'e1w1_hiersum_m{m2}_N{N}_s{s}', f'--arch hier --inner sum --m1 32 --m {m2} {W1} --N {N} --seed {s}')
            add(f'e1w1_hierattn_m{m2}_N{N}_s{s}', f'--arch hier --inner attn --m1 16 --m {m2} {W1} --N {N} --seed {s}')
for N in (8, 6, 4):
    for s in (0, 1):
        for m in (8, 32, 128):
            add(f'e1w2_tree{m}_N{N}_s{s}', f'--arch tree --m {m} {W2} --N {N} --seed {s}')
for N in (24, 12, 6, 3):
    for s in (0, 1, 2):
        for m in (1, 2, 4, 8, 16, 32):
            add(f'e1w0_tree{m}_N{N}_s{s}', f'--arch tree --m {m} {W0} --N {N} --seed {s}')

# E2 fine-tunes
FT = '--steps 10000 --cool 3000 --lr 3e-4 --warm 0'
for N in (24, 12, 3):
    for s in (0, 1):
        base = f'e1w0_tok_N{N}_s{s}'
        J2.append(f'e2w0_ctl_N{N}_s{s} --arch tok {W0} --N {N} --seed {s} {FT} --init @OUT@/{base}')
        for cut, ks in (('xsrc', (0, 1, 2, 4)), ('toq', (1, 2, 4, 8)), ('labout', (0, 1, 2, 4))):
            for k in ks:
                J2.append(f'e2w0_{cut}{k}_N{N}_s{s} --arch tok {W0} --N {N} --seed {s} {FT} --cut {cut} --cut_k {k} '
                          f'--init @OUT@/{base}')
for s in (0, 1):
    base = f'e1w1_tokg_N8_s{s}'
    J2.append(f'e2w1_ctl_N8_s{s} --arch tok --gid {W1} --N 8 --seed {s} {FT} --init @OUT@/{base}')
    for cut in ('xsrc', 'xgrp'):
        for k in (0, 2):
            J2.append(f'e2w1_{cut}{k}_N8_s{s} --arch tok --gid {W1} --N 8 --seed {s} {FT} --cut {cut} --cut_k {k} '
                      f'--init @OUT@/{base}')

# every phase-2 line runs sftrain.py
J2 = [l.split(' ', 1)[0] + ' sftrain.py ' + l.split(' ', 1)[1] for l in J2]
# E4 (amendment 1): World FA routing arms, heaviest, first
E4 = dict(flat='dense:all,dense:all,dense:all,dense:all', flat1='dense:self,dense:all,dense:all,dense:all',
          told='dense:self,dense:src,dense:all,dense:all', toldu='dense:self,uniform:src,dense:all,dense:all',
          learned='dense:self,idroute:all,dense:all,dense:all')
J1 = [f'e4_{k}_s{s} e4fa.py --attn {v} --seed {s}' for s in (0, 1) for k, v in E4.items()] + J1
open('jobs1.txt', 'w').write('\n'.join(J1) + '\n')
open('jobs2.txt', 'w').write('\n'.join(J2) + '\n')
print(len(J1), 'phase-1 jobs;', len(J2), 'phase-2 jobs')
