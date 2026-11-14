"""E48 plumbing (disclosed in amendment 56): synthetic per-net results with planted values, written under /tmp/e48_synth/results/,
to check that e48_summ.py recovers them (run e48_summ.py with that directory as the working directory).
Planted: world 1 K ctrl 0.012, mag 0.005, sgp 0.006, sq 0.003, sgnsq 0.0036 (sq/mag 0.6, sgnsq/sq 1.2); rom_large mag 0.07, sq
0.02; c_sign ctrl 0.20, sgnsq 0.03; seed 5's sq net stalled (gap 0.14). World 2: D = 1 ctrl 0.003, canon 0.003, prod 0.0021,
sgnprod 0.0021 (prod/canon 0.7); D = 2 ctrl 0.018, canon 0.009, prod 0.0045, sgnprod 0.006 (prod/canon 0.5, sgnprod/prod 1.33);
c_sign ctrl 0.34, sgnprod 0.04. Multiplicative per-seed noise of +-1% keeps the ratios.
"""
import os
import numpy as np

R = '/tmp/e48_synth/results'
rng = np.random.default_rng(0)


def put(sub, name, **kw):
    os.makedirs(os.path.join(R, sub), exist_ok=True)
    np.savez(os.path.join(R, sub, name + '.npz'), **kw)


def main():
    for s in range(12):
        e = 1 + 0.01 * rng.uniform(-1, 1)
        w1 = dict(ctrl=(0.012, 0.13, 0.05, 0.20), mag=(0.005, 0.07, -0.02, None), sgp=(0.006, 0.10, -0.01, 0.01),
                  sq=(0.003, 0.02, -0.03, None), sgnsq=(0.0036, 0.03, -0.02, 0.03))
        for a, (k, rl, rm, cs) in w1.items():
            gap = 0.14 if (a == 'sq' and s == 5) else k * e
            kw = dict(gap=gap, rom_large=rl * e, rom_middle=rm, lab_last=0.2, xq=0.01, arm=a, seed=s)
            if cs is not None:
                kw.update(sign_pc=cs * 0.01, sign_kl=0.01)
            sub, nm = {'ctrl': ('e37', 'e37_ctrl_s%d'), 'mag': ('e37', 'e37_mag_s%d'), 'sgp': ('e39', 'e39_sgp_s%d'),
                       'sq': ('e48w1', 'e48_w1sq_s%d'), 'sgnsq': ('e48w1', 'e48_w1sgnsq_s%d')}[a]
            put(sub, nm % s, **kw)
        for D, vals in ((1, dict(ctrl=0.003, canon=0.003, prod=0.0021, sgnprod=0.0021)), (2, dict(ctrl=0.018, canon=0.009, prod=0.0045, sgnprod=0.006))):
            for a, k in vals.items():
                kw = dict(gap=k * e, blind_gap=0.17, gap_lev0=k, gap_lev1=k, gap_lev2=k, fmt=a, seed=s, D=D)
                if a == 'ctrl':
                    kw.update(sign_pc=0.0034, sign_kl=0.01)
                if a == 'sgnprod':
                    kw.update(sign_pc=0.0004, sign_kl=0.01)
                sub = 'e40' if a in ('ctrl', 'canon') else 'e48t2'
                put(sub, '%s_d%d%s_s%d' % ('e40' if sub == 'e40' else 'e48', D, a, s), **kw)


if __name__ == '__main__':
    main()
