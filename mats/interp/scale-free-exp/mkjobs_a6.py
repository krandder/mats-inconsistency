"""Amendment 6 jobs appended to sfx-b4's phase 2: M4 (internal bottleneck on every edge out of a source's tokens,
rank-k keys and values, base frozen) and one-pass trees matched to tree2's capacity (h 96, m = 3 + m2)."""
W0 = '--world indep'
J = []
for N in (24, 12, 3):
    for s in (0, 1):
        for k in (1, 2, 4, 8):
            J.append(f'a6_m4both{k}_N{N}_s{s} sftrain.py --arch tok {W0} --N {N} --seed {s} --steps 2500 --cool 750 '
                     f'--lr 3e-3 --warm 0 --cut both --cut_k {k} --freeze_base --init @OUT@/e1w0_tok_N{N}_s{s}')
for N in (256, 64, 24, 3):
    for s in (0, 1):
        for m in (7, 11):
            J.append(f'a6_tree{m}h96_N{N}_s{s} sftrain.py --arch tree --m {m} --h 96 {W0} --N {N} --seed {s}')
base = open('/tmp/sfx_stage/sfx-b4/jobs2.txt').read().rstrip('\n').split('\n')
open('jobs_b4_2_a6.txt', 'w').write('\n'.join(base + J) + '\n')
print(len(base), '+', len(J), 'jobs')
