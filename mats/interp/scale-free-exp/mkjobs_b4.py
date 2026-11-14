"""Amendment 3 job list (rounds arm, N 64 / 256): jobs_b4_1.txt; jobs_b4_2.txt is empty."""
J = []
W0 = '--world indep'
for N in (256, 64):
    for s in (0, 1):
        J.append(f'e1w0_set_N{N}_s{s} sftrain.py --arch set {W0} --N {N} --seed {s}')
for N in (256, 64, 24, 3):
    for s in (0, 1):
        for m2 in (2, 4, 8):
            J.append(f'a3_tree2m{m2}_N{N}_s{s} sftrain.py --arch tree2 --m1 3 --m {m2} {W0} --N {N} --seed {s}')
for N in (256, 64):
    for s in (0, 1):
        for m in (4, 8, 16, 32):
            J.append(f'e1w0_tree{m}_N{N}_s{s} sftrain.py --arch tree --m {m} {W0} --N {N} --seed {s}')
open('jobs_b4_1.txt', 'w').write('\n'.join(J) + '\n')
open('jobs_b4_2.txt', 'w').write('')
print(len(J), 'jobs')
