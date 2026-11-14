"""E26 net list (PREREG-B amendment 34) -> results/e26_nets.json. Sets:
  M    the 15 mixture-head nets among E22's 48 acquired depth-2 ds-16 clean nets (includes the 12 bases of the noisy nets);
  G    the other 33 nets of E22's 48 (Gaussian heads);
  L1   the 20 depth-1 ds-16 clean nets of E15 (none acquired the task);
  NZ   the 12 noisy nets (group A: L2_s0.05, L2seed1, L2mx0, L2mx1, L2mx2, L2nb6; group B: L2seed2, L2ws0, L2ws3, L2mx4, L2nb5, L2nb7);
  HD   the heads cohort's base and L3 arms, Gaussian and mixture heads (E17; 20 nets);
  DY5, DY6  checkpoints of E16's replays of base_mix4_s5 (B lineage) and base_mix4_s6 (A lineage): every checkpoint up to
            10,000 steps, then every 5,000, plus each lineage's registered final (s5: 89,000; s6: 90,000);
  RND  one random-init net (plumbing only).
"""
import glob, json, os

CB = '/home/agent/workspaces/mats/work/cm-interp-B'
A = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05']
B = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']


def main():
    rows = []
    for r in json.load(open(os.path.join(CB, 'results/e22_nets.json'))):
        rows.append(dict(set='M' if 'mix4' in r['tag'] else 'G', tag=r['tag'], kind='clean', run=r['run'], cfg=r['cfg']))
    for r in json.load(open(os.path.join(CB, 'results/e15_nets.json'))):
        if r['depth'] == 1 and r['ds'] == 16:
            rows.append(dict(set='L1', tag=r['tag'], kind='clean', run=r['run'], cfg=r['cfg']))
    for name in A + B:
        rows.append(dict(set='NZ', tag=name, kind='noisy', run=os.path.join(CB, 'runs', name), group='A' if name in A else 'B'))
    for r in json.load(open(os.path.join(CB, 'results/e17_nets.json'))):
        if r['kind'] == 'clean' and r['arm'] in ('base_g', 'base_ws', 'L3_g', 'L3_ws'):
            rows.append(dict(set='HD', tag=r['tag'], kind='heads', run=r['run'], arm=r['arm'], depth=r['depth']))
    for s in (5, 6):
        for ck in sorted(glob.glob(os.path.join(CB, f'runs/dyn_base_s{s}/ckpt/step_*.pt'))):
            step = int(os.path.basename(ck)[5:11])
            if not (step <= 10_000 or step % 5000 == 0 or step == {5: 89_000, 6: 90_000}[s]):
                continue
            rows.append(dict(set=f'DY{s}', tag=f's{s}_step{step:06d}', kind='ckpt', run=os.path.join(CB, f'runs/base_mix4_s{s}'),
                             ckpt=ck, step=step))
    rows.append(dict(set='RND', tag='random_init', kind='random', run=''))
    json.dump(rows, open(os.path.join(CB, 'results/e26_nets.json'), 'w'), indent=0)
    from collections import Counter
    print(Counter(r['set'] for r in rows))


if __name__ == '__main__':
    main()
