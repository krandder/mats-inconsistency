"""E51 plumbing (disclosed in amendment 59; no registered quantity): e51_summ.pair_corr and the shared-share statistic on synthetic
shifts with a planted cross-net correlation; the small-sample run outputs (60 episodes, written by e51_plumb.sh to
/tmp/e51_plumb/results/e51/) are summarised separately by e51_summ.py run in /tmp/e51_plumb.
usage: python e51_plumb.py
"""
import numpy as np
import e51_summ as S


def main():
    r = np.random.default_rng(0)
    for rho in (0.0, 0.3, 0.7):
        common = r.standard_normal(20000)
        H = np.stack([np.sqrt(rho) * common + np.sqrt(1 - rho) * r.standard_normal(20000) for _ in range(12)])
        sys_h = H.mean(0).var() / H.var(1).mean()
        print('planted rho %.1f: pair_corr %.3f, shared share %.3f (expected about rho + (1 - rho)/12 = %.3f)' % (
            rho, S.pair_corr(H), sys_h, rho + (1 - rho) / 12))
    print('pair_corr with a repeated index list equals the distinct set:', abs(S.pair_corr(H, [0, 0, 1, 2]) - S.pair_corr(H, [0, 1, 2])) < 1e-12)


if __name__ == '__main__':
    main()
