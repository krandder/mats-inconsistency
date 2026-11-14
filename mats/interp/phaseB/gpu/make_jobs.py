"""Job files for E37-E39 (PREREG-B amendments 45-47). One line per training: NAME, then train_sw.py arguments.
  E37 (format switch; dense checkpoints), seeds 0-11, arms:
    ctrl, mag                 pure arms (90,000 steps);
    c2m{2,3,6,10,30}k         ctrl to mag after K steps;
    m2c{3,10,30}k             mag to ctrl after K steps;
    mag60                     mag, constant lr to 50,000 then cooldown to 60,000 (the budget of c2m30k after its switch);
  E39 sgp: sgnmag paired at initialisation (sparse checkpoints), seeds 0-11;
  E38 (size; sparse checkpoints), seeds 0-9: ctrl and mag at width 64, width 128 (depth 2) and depth 3 (width 32).
VM a: E37 + E39 seeds 0-5; VM b: E37 + E39 seeds 6-11; VM c: E38. Every file is ordered by seed, so a deadline cut leaves whole seeds.
usage: python gpu/make_jobs.py -> gpu/jobs_e37a.txt, gpu/jobs_e37b.txt, gpu/jobs_e38.txt
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E37 = [('ctrl', '--f0 ctrl --f1 ctrl'), ('mag', '--f0 mag --f1 mag')]
E37 += [('c2m%dk' % k, '--f0 ctrl --f1 mag --switch %d' % (k * 1000)) for k in (2, 3, 6, 10, 30)]
E37 += [('m2c%dk' % k, '--f0 mag --f1 ctrl --switch %d' % (k * 1000)) for k in (3, 10, 30)]
E37 += [('mag60', '--f0 mag --f1 mag --const 50000 --cooldown 10000')]
E38 = [('%s%s' % (c, f), '--f0 %s --f1 %s %s --ckpt sparse' % (f, f, a)) for c, a in (('w128', '--width 128'), ('d3', '--depth 3'),
                                                                                   ('w64', '--width 64')) for f in ('ctrl', 'mag')]


def lines_e37(seeds):
    out = []
    for s in seeds:
        out += ['e37_%s_s%d %s --seed %d --ckpt dense' % (n, s, a, s) for n, a in E37]
        out.append('e39_sgp_s%d --f0 sgnmag --f1 sgnmag --pair_init --seed %d --ckpt sparse' % (s, s))
    return out


def main():
    files = {'jobs_e37a.txt': lines_e37(range(0, 6)), 'jobs_e37b.txt': lines_e37(range(6, 12)),
             'jobs_e38.txt': ['e38_%s_s%d %s --seed %d' % (n, s, a, s) for s in range(10) for n, a in E38]}
    for fn, ls in files.items():
        open(os.path.join(HERE, fn), 'w').write('\n'.join(ls) + '\n')
        print(fn, len(ls))


if __name__ == '__main__':
    main()
