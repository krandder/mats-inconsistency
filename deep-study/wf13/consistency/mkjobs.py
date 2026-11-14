"""mkjobs.py : the registered job list (one line "ARM N D R" per run), most important block first (PREREG sections 4, 8)."""
NS = [64, 256, 1024, 4096, 20480]
ARMS = ['i@10', 'i@1', 'iio@10', 'iio@1', 'iis@1', 'iis@0.1', 'iis@10', 'iii@1', 'synth@1', 'iiisym@10']
out = ['fixed:one 0 0 0', 'fixed:exu 0 0 0', 'fixed:meanu 0 0 0', 'fixed:exuv 0 0 0']
for n in NS:                                        # block 1: subset d = 0, one run per cell
    out += ['data %d 0 0' % n] + ['%s %d 0 0' % (a, n) for a in ARMS]
out += ['%s 0 0 %d' % (a, r) for a, rs in (('iis@1', (0, 1)), ('synth@1', (0, 1)), ('i@1', (0,)), ('iio@1', (0,)), ('iii@1', (0,))) for r in rs]
out += ['data %d 0 1' % n for n in NS]              # second initialisation: the placebo of the drag analysis
for n in NS:                                        # block 2: subset d = 1
    arms = ARMS if n <= 1024 else ['i@10', 'iio@10', 'iis@1', 'synth@1']
    out += ['data %d 1 0' % n] + ['%s %d 1 0' % (a, n) for a in arms]
print('\n'.join(out))
