"""cexact.py WORLD CKPT SIZE DATADIR OUT.npz : EXACT REFERENCES BY ENUMERATION over 2^20. EVALUATION CODE ONLY.

No arm reads this file's output (cgen.py, cmodel.py, ctrain.py, cdirect.py do not import exact.py or open exact*.npz;
check_firewall.py tests it). For every context of the test sets t1, t2, t3 and every check c:
  pstar  the true world's P(x_c = 1 | x_S)
  pTu    the same under the score T: M's empty-context numbers for the assumptions times M(x_c | h) for the checks
and for t3's terminal vectors the exact log P*(x) and log T(x). Also stores the test sets' x, mask and index arrays.
"""
import os
import sys

import numpy as np
import torch

PW = os.environ.get('PROBWORLD', '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.path.insert(0, PW)
sys.dont_write_bytecode = True
from world import World, N_H, N_X, N_VAR  # noqa: E402
import model as MM  # noqa: E402
from model import MASK  # noqa: E402
from exact import Exact  # noqa: E402
import cmodel as CM  # noqa: E402  (only load_set, to join shards)

if __name__ == '__main__':
    ws, ckpt, size, ddir, outp = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('NTHREADS', 2)))
    W = World(ws)
    E = Exact(W.checks, W.eps, dev)
    par = E.par.double()
    la, le = np.log(1 - W.eps), np.log(W.eps)
    lp_true = E.log_prior(W.prior)
    loga = torch.load(ddir + '/score.pt')['loga'].to(dev).double()
    lp_M = E.bitsf @ (loga[:, 1] - loga[:, 0]) + loga[:, 0].sum()
    ck = torch.load(ckpt, map_location=dev)
    m = MM.build(size, N_VAR).to(dev)
    m.load_state_dict(ck['state'])
    m.eval()
    ucache = sorted(__import__('glob').glob(ddir + '/U_part_*.npy'), key=lambda f: int(f[:-4].split('_')[-1]))
    if ucache:                                                        # M(x_c | h) for all 2^20 h, computed by cU.py
        U = torch.cat([torch.tensor(np.load(f)) for f in ucache], 0).to(dev).double()
        assert U.shape[0] == 2 ** N_H
    else:
        U = []
        with torch.no_grad():
            for i in range(0, 2 ** N_H, 16384):
                tok = torch.cat([E.bits[i:i + 16384].long(), torch.full((min(16384, 2 ** N_H - i), N_X), MASK, dtype=torch.long, device=dev)], 1)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                    lg = m(tok)
                U.append(torch.sigmoid(lg.float())[:, N_H:].clamp(1e-6, 1 - 1e-6).double())
        U = torch.cat(U, 0)
    U1, U0 = U.log(), (1 - U).log()
    out = {}
    bs = 64 if dev.type == 'cuda' else 32
    for name in ('t1', 't2', 't3'):
        d = CM.load_set('%s/%s.pt' % (ddir, name))
        x, mask = d['x'].to(dev).double(), d['mask'].to(dev).double()
        ps, pt, lzs, lzt = [], [], [], []
        for i in range(0, x.shape[0], bs):
            a1, a0 = (mask[i:i + bs] * x[i:i + bs]).T, (mask[i:i + bs] * (1 - x[i:i + bs])).T
            lw = lp_true[:, None] + par @ (a1 * la + a0 * le) + (1 - par) @ (a1 * le + a0 * la)
            lzs.append(torch.logsumexp(lw, 0))
            ppar = torch.softmax(lw, 0).T @ par
            ps.append(ppar * (1 - W.eps) + (1 - ppar) * W.eps)
            lw = lp_M[:, None] + U1 @ a1 + U0 @ a0
            lzt.append(torch.logsumexp(lw, 0))
            pt.append(torch.softmax(lw, 0).T @ U)
        out[name + '_pstar'] = torch.cat(ps).cpu().numpy()
        out[name + '_pTu'] = torch.cat(pt).cpu().numpy()
        out[name + '_logPctx'] = torch.cat(lzs).cpu().numpy()       # log P*(x_S): the probability of the context itself
        out[name + '_logTctx'] = torch.cat(lzt).cpu().numpy()
        out[name + '_x'], out[name + '_mask'] = d['x'].numpy(), d['mask'].numpy()
        out[name + '_p_one'] = d['p_one'].numpy()
        for k in ('X', 'Y', 'xreal', 'perm', 'xvec', 'logRhat', 'logRhat_old', 'p_one_S', 'p_one_M', 'p_one_L'):
            if k in d:
                out['%s_%s' % (name, k)] = d[k].numpy()
        if name == 't3':                                             # exact log probability of each terminal vector
            xv = d['xvec'].to(dev).double()
            a1, a0 = xv.T, (1 - xv).T
            out['t3_logPstar'] = torch.cat([torch.logsumexp(lp_true[:, None] + par @ (a1[:, i:i + bs] * la + a0[:, i:i + bs] * le)
                                                            + (1 - par) @ (a1[:, i:i + bs] * le + a0[:, i:i + bs] * la), 0)
                                            for i in range(0, xv.shape[0], bs)]).cpu().numpy()
            out['t3_logT'] = torch.cat([torch.logsumexp(lp_M[:, None] + U1 @ a1[:, i:i + bs] + U0 @ a0[:, i:i + bs], 0)
                                        for i in range(0, xv.shape[0], bs)]).cpu().numpy()
    np.savez_compressed(outp, **out)
    ps, pt, mk = out['t1_pstar'], out['t1_pTu'].clip(1e-6, 1 - 1e-6), ~out['t1_mask']
    kl = (ps * np.log(ps / pt) + (1 - ps) * np.log((1 - ps) / (1 - pt)))[mk].mean()
    print('exact refs written: t1 excess of the score T %.4f nats over %d queries' % (kl, mk.sum()))
