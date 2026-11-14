"""E48 plumbing checks (disclosed in amendment 56; no registered quantity). Compares checkpoints written by short CPU runs made
beforehand into /tmp/e48_plumb/ (see the amendment) with each other and with existing runs:
  w1: train_lin.py ctrl == train_sw.py ctrl after 500 steps (bit for bit); sq's step 0 == E37 ctrl s0's step 0; sgnsq's (pair
      init) step 0 == E39 sgp s0's step 0;
  w2: train_t2lin.py ctrl == train_t2.py ctrl after 500 steps; prod's and sgnprod's step 0 identical; every shared parameter ==
      E40 d2ctrl s0's step 0; the new input columns within +-1/sqrt(D+2);
  formats: token builders on a few episodes (sq, sgnsq, prod, sgnprod) against direct formulas; flip invariance of prod tokens.
usage: python e48_plumb.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e48_fmt as F  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as L  # noqa: E402
import world as W  # noqa: E402

PL = '/tmp/e48_plumb'


def ld(p):
    return torch.load(p, map_location='cpu')


def same(a, b, keys=None):
    keys = keys or a.keys()
    return all(torch.equal(a[k], b[k]) for k in keys)


def main():
    # world 1
    a = ld(f'{PL}/w1_sw_ctrl/model_final.pt'); b = ld(f'{PL}/w1_lin_ctrl/model_final.pt')
    print('w1 train_lin ctrl == train_sw ctrl at 500 steps:', same(a, b))
    s0 = ld('runs/e37/e37_ctrl_s0/ckpt/step_000000.pt')
    print('w1 sq step 0 == E37 ctrl s0 step 0:', same(ld(f'{PL}/w1_sq/ckpt/step_000000.pt'), s0))
    g0 = ld('runs/e39/e39_sgp_s0/ckpt/step_000000.pt')
    print('w1 sgnsq (pair init) step 0 == E39 sgp s0 step 0:', same(ld(f'{PL}/w1_sgnsq/ckpt/step_000000.pt'), g0))
    # world 2
    a = ld(f'{PL}/w2_t2_ctrl/model_final.pt'); b = ld(f'{PL}/w2_lin_ctrl/model_final.pt')
    print('w2 train_t2lin ctrl == train_t2 ctrl at 500 steps:', same(a, b))
    c0 = ld('runs/e40/e40_d2ctrl_s0/ckpt/step_000000.pt')
    p0 = ld(f'{PL}/w2_prod/ckpt/step_000000.pt'); q0 = ld(f'{PL}/w2_sgnprod/ckpt/step_000000.pt')
    print('w2 prod step 0 == sgnprod step 0:', same(p0, q0))
    shared = [k for k in c0 if k != 'inp.weight']
    print('w2 prod shared params == E40 d2ctrl s0 step 0:', same(p0, c0, shared),
          '| inp ctrl columns equal:', torch.equal(p0['inp.weight'][:, :4], c0['inp.weight']),
          '| new columns max abs %.4f (bound %.4f)' % (p0['inp.weight'][:, 4:].abs().max(), 1 / np.sqrt(4)))
    # formats
    rng = np.random.default_rng(5)
    d = W.sample_np(3, rng); semb = rng.standard_normal((3, 3, 16)) / 4; qemb = rng.standard_normal((3, 5, 16)) / 4
    r2 = (d['o'][:, :, :4] - d['y'][:, None, :4]) ** 2
    Xs, _ = F.make_X(d['o'], d['y'], semb, qemb, 'sq'); Xc, _ = F.make_X(d['o'], d['y'], semb, qemb, 'ctrl')
    lab = [5 * s + q for s in range(3) for q in range(4)]
    ok = np.allclose(Xs[:, lab, 32] - Xs[:, lab, 33], r2.reshape(3, 12), atol=1e-5)
    ok &= np.array_equal(np.delete(Xs, 32, 2), np.delete(Xc, 32, 2)) and np.array_equal(Xs[:, [4, 9, 14, 15], 32], Xc[:, [4, 9, 14, 15], 32])
    print('w1 sq tokens: labelled reading - label = r^2, everything else = ctrl:', bool(ok))
    Xg, _ = F.make_X(d['o'], d['y'], semb, qemb, 'sgnsq')
    ok = np.array_equal(Xg[:, :, :36], Xc) and np.allclose(Xg[:, lab, 36], r2.reshape(3, 12), atol=1e-5)
    ok &= np.all(Xg[:, [4, 9, 14, 15], 36] == 0)
    print('w1 sgnsq tokens: ctrl + r^2 on labelled tokens only:', bool(ok))
    for D in (1, 2):
        e = T.sample(4, np.random.default_rng(7), D, 8)
        Xp, _ = L.tokens(e, 'prod'); Xq, _ = L.tokens(e, 'sgnprod'); X0, _ = T.tokens(e, 'ctrl')
        x, y = e['x'], e['y']
        G = np.einsum('nkd,nke->nde', x, x); b = np.einsum('nkd,nk->nd', x, y); yy = (y * y).sum(1)
        P = Xp[:, :8, D + 2:].sum(1)
        iu = [(i, j) for i in range(D) for j in range(i, D)]
        ok = np.allclose(P[:, :len(iu)], np.stack([G[:, i, j] for i, j in iu], 1), atol=1e-4)
        ok &= np.allclose(P[:, len(iu):len(iu) + D], b, atol=1e-4) and np.allclose(P[:, -1], yy, atol=1e-4)
        ok &= np.array_equal(Xq[:, :, :D + 2], X0) and np.array_equal(Xq[:, :, D + 2:], Xp[:, :, D + 2:])
        ok &= np.all(Xp[:, :8, :D + 2] == 0) and np.array_equal(Xp[:, 8, :D + 2], X0[:, 8])
        s = np.where(np.random.default_rng(8).random((4, 8)) < 0.5, -1.0, 1.0)
        f = dict(e); f['x'] = e['x'] * s[:, :, None]; f['y'] = e['y'] * s
        ok &= np.array_equal(L.tokens(f, 'prod')[0], Xp)
        print(f'w2 D={D} prod/sgnprod tokens: sums = X^T X, X^T y, y.y; sgnprod = ctrl + products; prod flip-invariant:', bool(ok))


if __name__ == '__main__':
    main()
