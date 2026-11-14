"""E72 exploration, part 4 (eighth agent; DEV NETS ONLY; descriptive, not a registered test): which feature of the real pool makes
the departure that the idealised reduced model (M1) misses? On real simulated episodes (the net's own tokens), the pooled statistic
T_h = (sum w g, sum w r g) per pooling head with
  V_real  w = the real attention renormalised over own keys, g = A_j / n_j (= M2);
  V_coef  w = softmax_j((L_h + K_h r_j) g_j) with the head's global mean L_h, K_h, and the real g_j (rho_j, A_j);
  V_src   as V_coef with the source's own mean L and K (over its four keys) in place of the global means;
  V_A     g_j = (1 + r_j^2 / c_j^2)^(-1/2) with the token's real c_j = A_j / U (rho = 0), global mean L_h, K_h;
  V_rho   g_j = (1 + 2 r_j rho_j / c + r_j^2 / c^2)^(-1/2) with the token's real rho_j and the median c, global mean L_h, K_h;
  V_id    g_j = (1 + r_j^2 / c^2)^(-1/2), median c, global mean L_h, K_h (= M1 on real r).
Each read by the CB classifier as in e72_run.py (100,000 simulated episodes; the pairs' real tokens).
usage: python e72x_ladder2.py OUTJSON TAG=RUNDIR:EVALFILE [...]
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import e63_run as E63  # noqa: E402
import e72_run as R  # noqa: E402

S, Q, QL, T = 3, 5, 4, 16
OC, YC = 32, 33
NS = 100_000


def parts(model, X, pool):
    """Per (episode, source): real w (B, S, P, 4); r; A_j, rho_j (B, S, 4); U; L_j, K_j (B, S, P, 4)."""
    A = R.attn1(model, X); B = X.shape[0]
    w = np.stack([A[:, :, R.SUMM[s], R.LABS[s]] for s in range(S)], 1)[:, :, pool]; w = w / w.sum(-1, keepdims=True)
    Wm = model.inp.weight.detach().numpy().astype(np.float64); bm = model.inp.bias.detach().numpy().astype(np.float64)
    lab = [j for s in range(S) for j in R.LABS[s]]
    Xl = X[:, lab, :].astype(np.float64); X0 = Xl.copy(); X0[..., OC] = X0[..., YC]
    a0 = X0 @ Wm.T + bm; a0 = a0 - a0.mean(-1, keepdims=True); u = Wm[:, OC] - Wm[:, OC].mean()
    Aj = np.linalg.norm(a0, axis=-1); U = float(np.linalg.norm(u)); rho = (a0 @ u) / (Aj * U)
    r = (Xl[..., OC] - Xl[..., YC]).reshape(B, S, QL)
    _, qh, blk = E63.block1_parts(model, X)
    L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    Lj = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1)[:, :, pool]; Kj = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)[:, :, pool]
    return w, r, Aj.reshape(B, S, QL), rho.reshape(B, S, QL), U, Lj, Kj


def stat(kind, w, r, Aj, rho, U, Lj, Kj, Lg, Kg, c):
    if kind in ('V_real', 'V_coef', 'V_src'):
        g = 1.0 / np.sqrt(1.0 + 2 * r * rho * U / Aj + r ** 2 * U ** 2 / Aj ** 2)
    elif kind == 'V_A':
        g = 1.0 / np.sqrt(1.0 + r ** 2 * U ** 2 / Aj ** 2)
    elif kind == 'V_rho':
        g = 1.0 / np.sqrt(1.0 + 2 * r * rho / c + r ** 2 / c ** 2)
    else:
        g = 1.0 / np.sqrt(1.0 + r ** 2 / c ** 2)
    out = []
    for p in range(w.shape[2]):
        if kind == 'V_real':
            ww = w[:, :, p]
        else:
            if kind == 'V_src':
                Lh, Kh = Lj[:, :, p].mean(-1, keepdims=True), Kj[:, :, p].mean(-1, keepdims=True)
            else:
                Lh, Kh = Lg[p], Kg[p]
            zz = (Lh + Kh * r) * g; zz = zz - zz.max(-1, keepdims=True); ww = np.exp(zz); ww /= ww.sum(-1, keepdims=True)
        out += [(ww * g).sum(-1), (ww * r * g).sum(-1)]
    return np.stack(out, -1)


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        hd = R.heads_of(model, arm, meta); pool = hd['pool']; c = hd['c']
        Lg = [hd['L'][h] for h in pool]; Kg = [hd['K'][h] for h in pool]
        rng = np.random.default_rng(7203); d = W.sample_np(NS, rng)
        se = rng.standard_normal((NS, S, 16)) / 4; qe = rng.standard_normal((NS, Q, 16)) / 4
        Xs, _ = E63.F.make_X(d['o'], d['y'], se, qe, arm)
        PX = []
        for k in (0, 1):
            ps, pq = E63.F.ids_from_X(z['X%d' % k])
            PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
        chunks = lambda X: [parts(model, X[i:i + 4000], pool) for i in range(0, len(X), 4000)]
        cs, c0, c1 = chunks(Xs), chunks(PX[0]), chunks(PX[1])
        o = dict(net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']), pool=pool, c=c)
        for kind in ('V_real', 'V_coef', 'V_src', 'V_A', 'V_rho', 'V_id'):
            f = lambda ch: np.concatenate([stat(kind, *pp, Lg, Kg, c) for pp in ch])
            Fs, F0, F1 = f(cs), f(c0), f(c1)
            o[kind] = R.cb_rom_feats(R.fit(Fs.reshape(NS * S, -1), d['vidx'].reshape(-1)), F0, F1, z)
        o['seconds'] = time.time() - t0; res[tag] = o
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps(o), flush=True)


if __name__ == '__main__':
    main()
