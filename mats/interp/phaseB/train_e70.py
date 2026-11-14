"""E70 training dispatcher (PREREG-B amendment 78): --arm b1h4 runs train_b1h4.py (4 heads in block 1, 2 in block 2; ctrl's initial values
and episodes); --arm ownmag runs train_own.py with --variant own (E60's restricted learned pool) in the format given by --f0/--f1 (mag).
usage: python train_e70.py --arm b1h4|ownmag --f0 F --f1 F --seed S --out DIR [train_sw.py options]
"""
import sys

i = sys.argv.index('--arm'); arm = sys.argv[i + 1]; del sys.argv[i:i + 2]
if arm == 'b1h4':
    import train_b1h4 as M
elif arm == 'ownmag':
    sys.argv += ['--variant', 'own']
    import train_own as M
else:
    raise SystemExit('unknown arm %s' % arm)

if __name__ == '__main__':
    M.TS.main()
