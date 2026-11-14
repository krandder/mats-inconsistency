"""Reorder the evaluation job list: interleave the theory measurements early, skip jobs whose output exists."""
import os, re
J = open('scratch/jobs_eval.txt').read().strip().split('\n')
def out_of(j):
    m = re.search(r'truth_baselines.py (\d+) (\d+) (\d+)', j)
    if m: return f'runs/truth_n{m[1]}_L{m[2]}_seed{m[3]}.json'
    m = re.search(r'run_bnb.py (\d+) (\d+) (\d+) .*--priority (\w+) --split (\w+)', j)
    if m: return f'runs/bnb_{m[4]}-{m[5]}_n{m[1]}_L{m[2]}_seed{m[3]}.json'
    m = re.search(r'noise_stability.py (\d+) (\d+) (\d+)', j)
    if m: return f'runs/nstab_n{m[1]}_L{m[2]}_K{m[3]}.json'
    m = re.search(r'harmonic_profile.py (\d+) (\d+) (\d+)', j)
    if m: return f'runs/harm_n{m[1]}_L{m[2]}_s{m[3]}.json'
    if 'spin_glass' in j: return 'runs/spin_glass.json'
    return None
theory = [j for j in J if 'noise_stability' in j or 'harmonic_profile' in j]
rest = [j for j in J if j not in theory]
order, ti = [], 0
for i, j in enumerate(rest):
    if i % 4 == 0 and ti < len(theory):
        order.append(theory[ti]); ti += 1
    order.append(j)
order += theory[ti:]
keep = [j for j in order if not (out_of(j) and os.path.exists(out_of(j)))]
open('scratch/jobs_eval2.txt', 'w').write('\n'.join(keep) + '\n')
print(len(J), 'jobs,', len(keep), 'to run')
