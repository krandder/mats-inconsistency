import numpy as np
import torch
import mech as Mx


def setup(NET):
    ep, d = Mx.c4_set()
    eps = ep['eps']
    X = torch.as_tensor(ep['X']); M = torch.as_tensor(ep['mask'])
    E, T, _ = X.shape
    mem = d['meta_member'].astype(int); pair = d['meta_pair'].astype(int)
    i0 = np.flatnonzero(mem == 0); i1 = np.flatnonzero(mem == 1)
    pos = {k: np.zeros(E, int) for k in ('Aq', 'Bq')}
    AEV = np.zeros((E, T), bool); BEV = np.zeros((E, T), bool)
    for i, e in enumerate(eps):
        onq = e.ev_items[e.q_ev][:, 0] == e.query
        isA = e.q_src == e.meta['A']; isB = e.q_src == e.meta['B']
        pos['Aq'][i] = np.flatnonzero(isA & onq)[0]; pos['Bq'][i] = np.flatnonzero(isB & onq)[0]
        AEV[i, :e.n] = isA & ~onq; BEV[i, :e.n] = isB & ~onq
    pb = Mx.Hn._unragged(d)
    g = (np.ones((E, 1)), d['mean_freq_only'][:, None], d['var_freq_only'][:, None])
    klf = Mx.Hn.kl_bayes_model(pb, g)
    model = Mx.load_net(NET)

    def run(fac):
        ms, preds = [], []
        for a in range(0, E, 100):
            sl = np.arange(a, min(E, a + 100))
            (m, ls, (w, mu, v)), _, _ = Mx.forward(model, X[sl], M[sl], keep_sites=False,
                                                   attn_edit=fac(sl) if fac else None)
            ms.append(m.numpy()); preds.append((w.numpy(), mu.numpy(), v.numpy()))
        return np.concatenate(ms), tuple(np.concatenate([p[k] for p in preds]).astype(np.float64) for k in range(3))
    bm, _ = run(None)
    dm = bm[i0] - bm[i1]; keep = np.abs(dm) >= 0.05

    def score(m, p):
        kl = Mx.Hn.kl_bayes_model(pb, p)
        sf = (m[i0] - m[i1])[keep] / dm[keep]
        return dict(shift_frac_median=float(np.median(sf)), shift_frac_q25=float(np.percentile(sf, 25)),
                    shift_frac_q75=float(np.percentile(sf, 75)), c4_closure_frac=float((1 - kl / klf >= 0.5).mean()))
    return dict(eps=eps, E=E, T=T, pair=pair, pos=pos, AEV=AEV, BEV=BEV, run=run, score=score, model=model)


def equalise(C, blocks):
    T = C['T']; QI = T - 1

    def fac(sl):
        ar = torch.arange(len(sl)); pa = torch.as_tensor(C['pos']['Aq'][sl]); pq = torch.as_tensor(C['pos']['Bq'][sl])

        def ed(li, a):
            if li not in blocks:
                return a
            a = a.clone()
            mA = a[ar, :, QI, pa]; mB = a[ar, :, QI, pq]
            a[ar, :, QI, pa] = 0.5 * (mA + mB); a[ar, :, QI, pq] = 0.5 * (mA + mB)
            return a
        return ed
    return fac
