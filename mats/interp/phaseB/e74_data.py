"""E74 data (PREREG-B amendment 82, planned): a test set and E12-style matched pairs for world 1 with S sources (world_n.World(S)).
  test   N_TEST = 16,000 episodes (numpy seed 74_000 + S): tokens (ctrl format), exact Bayes NLL of y* (world.bayes, NV^S components);
  pairs  for every (episode, source) row of N_PAIR = 9,000 // S episodes (seed 74_100 + S) whose labelled sum of squares SS is in E12's middle
         [0.679992, 3.182394) or large [3.182394, inf) range: two versions of that source's four labelled errors with the same SS,
         spread (kappa 0.25: four equal sizes) and big miss (kappa 0.85: one error sqrt(0.85 SS), three sqrt(0.05 SS)), random signs and
         position (as e12_data.py); every other input unchanged; tokens of both versions; exact Bayes effective weights of every
         source, d mean / d o*_s by central differences (h = 1e-3) of the exact mixture mean; a check that the Bayes mean, variance and
         weights do not depend on the version.
usage: python e74_data.py S -> results/e74_data_S<S>.npz
"""
import os, sys
import numpy as np

import world_n as WN

LO, HI = 0.679992, 3.182394
KAPPAS = [0.25, 0.85]
N_TEST = 16000
PAIR_EPISODES = 9000                                    # N_PAIR = PAIR_EPISODES // S: about 6,000 pair rows for every S
QL, Q = WN.QL, WN.Q


def bayes_mean(world, d):
    M, V, W = world.bayes(d)
    return (W * M).sum(1), (W * (V[None] + M ** 2)).sum(1) - (W * M).sum(1) ** 2


def bayes_weights(world, d, h=1e-3):
    out = []
    for s in range(world.S):
        dp = dict(d); dm = dict(d)
        dp['o'] = d['o'].copy(); dm['o'] = d['o'].copy()
        dp['o'][:, s, QL] += h; dm['o'][:, s, QL] -= h
        out.append((bayes_mean(world, dp)[0] - bayes_mean(world, dm)[0]) / (2 * h))
    return np.stack(out, 1)


def main():
    S = int(sys.argv[1]); world = WN.World(S); p = 'results/e74_data_S%d.npz' % S
    assert not os.path.exists(p)
    rng = np.random.default_rng(74_000 + S); d = world.sample_np(N_TEST, rng)
    semb = rng.standard_normal((N_TEST, S, 16)) / 4; qemb = rng.standard_normal((N_TEST, Q, 16)) / 4
    X, valid = world.assemble(semb, qemb, d['o'], d['y'])
    M, V, W = world.bayes(d); ystar = d['y'][:, QL]
    out = dict(test_X=X, test_valid=valid, test_o=d['o'], test_y=d['y'], test_semb=semb, test_qemb=qemb, test_ystar=ystar,
               test_nll_bayes=WN.mix_nll(M, V, W, ystar))
    N_PAIR = PAIR_EPISODES // S
    rng = np.random.default_rng(74_100 + S); d = world.sample_np(N_PAIR, rng)
    semb = rng.standard_normal((N_PAIR, S, 16)) / 4; qemb = rng.standard_normal((N_PAIR, Q, 16)) / 4
    signs = rng.choice([-1.0, 1.0], size=(N_PAIR, S, QL)); pos = rng.integers(0, QL, size=(N_PAIR, S))
    res = d['o'][:, :, :QL] - d['y'][:, None, :QL]; SS = (res ** 2).sum(2)
    sel = np.argwhere(SS >= LO); b, s = sel[:, 0], sel[:, 1]; m = len(b)
    out.update(src=s, base=b, range=np.where(SS[b, s] >= HI, 'large', 'middle'), SS=SS[b, s], kappas=np.array(KAPPAS))
    means = []
    for ki, kap in enumerate(KAPPAS):
        o = d['o'][b].copy(); y = d['y'][b].copy(); ss = SS[b, s]
        r = np.repeat(np.sqrt((1 - kap) / 3 * ss)[:, None], QL, 1); r[np.arange(m), pos[b, s]] = np.sqrt(kap * ss); r = r * signs[b, s]
        o[np.arange(m), s, :QL] = y[:, :QL] + r
        dd = dict(o=o, y=y, v=d['v'][b].copy(), vidx=d['vidx'][b].copy())
        Xk, vk = world.assemble(semb[b], qemb[b], o, y)
        out['X%d' % ki] = Xk; out['valid%d' % ki] = vk; out['d%d_o' % ki] = o; out['d%d_y' % ki] = y
        out['WB%d' % ki] = bayes_weights(world, dd); means.append(bayes_mean(world, dd))
    out['semb'] = semb[b]; out['qemb'] = qemb[b]
    dm = float(np.abs(means[1][0] - means[0][0]).max()); dv = float(np.abs(means[1][1] - means[0][1]).max())
    dw = float(np.abs(out['WB1'] - out['WB0']).max())
    print('S = %d: test %d episodes; pairs %d rows (large %d, middle %d); max |change| across kappa: Bayes mean %.1e, variance %.1e, weights %.1e'
          % (S, N_TEST, m, int((out['range'] == 'large').sum()), int((out['range'] == 'middle').sum()), dm, dv, dw), flush=True)
    assert dm < 1e-8 and dv < 1e-8 and dw < 1e-6, 'Bayes not invariant'
    np.savez(p, **out)


if __name__ == '__main__':
    main()
