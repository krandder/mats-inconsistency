"""Hölder per-event bounds with Rényi divergences on the trained small noisy nets of dec.py (PREREG-B amendment 3).

For one general model (the fixed-size description chain q_1..q_L, S Gaussian cells + delta defensive):
  local      d_l(b) = D_b(K_l q_{l-1} || q_l)   (Rényi, average case), b on an order grid, by multiple importance sampling
             with pooled samples of K q (node mixture) and q; KL; D_inf by search.
  composed   B(l, a) >= D_a(p_l || q_l) by the weak triangle inequality + data processing, optimised split per step:
             B(l, a) = min_p [((a - 1/p)/(a - 1)) B(l-1, p a) + d_l(q (a - 1/p))], q = p/(p-1); B(1, a) = d_1(a).
  direct     D_a(p_l || q_l) with p_l's density from a bank of 2^16 true parents (validation; uses samples of p).
Events per depth l: A = {w_l.h_l + b_l > t}, (w_l, b_l) the least-squares task readout at layer l (trained readout at L).
Bounds (bits): out-level  (1 - 1/a)[log2 q_l(A) + B(l, a)];  through-kernel (1 - 1/a)[log2 E_{q_{l-1}} g^(a/(a-1)) + B(l-1, a)],
g(h) = P(A | h_{l-1} = h) exact; 'oracle' versions use the direct D_a instead of B (composition-free).

usage: python holder.py SIGMA S  -> results/holder_s{SIGMA}_S{S}.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm, beta as betad
from scipy.special import logsumexp
import dedu as Dd
import e2 as E2
import dec as Dc

torch.set_num_threads(1)
L, DIN, D8 = E2.L, E2.DIN, 8
LN2 = math.log(2)
BETAS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
ALPHAS = [2.0, 4.0, 8.0]
AGRID = [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192]
PGRID = [1.01, 1.03, 1.06, 1.1, 1.2, 1.35, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 16, 32, 64, 1e9]


def gm_sample(q, n, rng):
    c = rng.choice(len(q.w), n, p=q.w)
    return q.mu[c] + np.einsum('nde,ne->nd', q.L[c], rng.standard_normal((n, q.D)))


def node_sample(K, n, rng, sigma):
    j = rng.choice(len(K.w), n, p=K.w)
    return K.mu[j] + sigma * rng.standard_normal((n, K.D))


def renyi_mis(lP_P, lQ_P, lP_Q, lQ_Q, betas):
    """D_b(P||Q) in bits for each b, from n samples of P and n of Q (pooled = samples of M = (P+Q)/2)."""
    lP = np.r_[lP_P, lP_Q]; lQ = np.r_[lQ_P, lQ_Q]; lM = np.logaddexp(lP, lQ) - math.log(2)
    out = {}
    for b in betas:
        if b == 1.0:
            out[b] = float(np.mean(lP_P - lQ_P) / LN2)
        else:
            v = b * lP + (1 - b) * lQ - lM
            out[b] = float((logsumexp(v) - math.log(len(v))) / (b - 1) / LN2)
    return out


class Curve:
    """Monotone (non-decreasing) Rényi curve d(b), b >= 1, piecewise linear in log b; d(b) = dinf beyond the grid."""

    def __init__(self, est, dinf):
        bs = np.array(sorted(est)); vs = np.maximum.accumulate(np.array([est[b] for b in bs]))
        self.bs, self.vs = bs, vs; self.dinf = max(float(dinf), float(vs[-1]))

    def __call__(self, b):
        if b >= self.bs[-1] or not np.isfinite(b):
            return self.dinf
        return float(np.interp(math.log(b), np.log(self.bs), self.vs))


def compose(curves):
    """B[l](a) for l = 1..L (1-based list index l-1), a on AGRID, plus Binf[l] = sum of D_inf."""
    Bs, Binf = [], []
    for l, d in enumerate(curves):
        if l == 0:
            B = {a: d(a) for a in AGRID}; Bi = d.dinf
        else:
            prevB, prevI = Bs[-1], Binf[-1]
            def prev(o):
                if o >= AGRID[-1]: return prevI
                xs = np.log(AGRID); ys = np.maximum.accumulate(np.array([prevB[a] for a in AGRID]))
                return float(np.interp(math.log(o), xs, ys))
            B = {}
            for a in AGRID:
                best = prev(a) + d.dinf                                      # p -> 1 limit
                for p in PGRID:
                    q = p / (p - 1) if p < 1e8 else 1.0
                    c = (a - 1 / p) / (a - 1)
                    val = c * prev(p * a) + d(q * (a - 1 / p))
                    best = min(best, val)
                B[a] = best
            Bi = prevI + d.dinf
        Bs.append(B); Binf.append(Bi)
    return Bs, Binf


def Bval(B, a):
    return float(np.interp(math.log(a), np.log(AGRID), np.maximum.accumulate(np.array([B[x] for x in AGRID]))))


def true_paths(net, sigma, n, rng):
    x = rng.standard_normal((n, DIN)); hs = []; h = x
    for l in range(L):
        h = net.f(l, h) + sigma * rng.standard_normal((n, net.d)); hs.append(h)
    return x, hs


def run(sigma, S):
    t_all = time.time()
    net, w_ro, b_ro = Dc.train_net_ro(D8, sigma); s2 = sigma ** 2; d = net.d
    rng = np.random.default_rng(1)
    U = np.random.default_rng(0).standard_normal((4, DIN))
    # ---- readouts per depth (least squares on true paths; trained readout at L)
    x, hs = true_paths(net, sigma, 2 ** 16, rng); ytask = np.sin(x @ U.T).sum(1)
    ro = []
    for l in range(L):
        if l == L - 1:
            ro.append((w_ro, b_ro))
        else:
            A = np.c_[hs[l], np.ones(len(x))]; c = np.linalg.lstsq(A, ytask, rcond=None)[0]; ro.append((c[:-1], float(c[-1])))
    # ---- chain (deductive) and local divergences
    t0 = time.time()
    q = Dd.GM([1.0], [np.zeros(DIN)], [np.eye(DIN)]); qs, Ks, curves, loc = [], [], [], []
    for l in range(L):
        n_per = max(128, 2 ** 15 // len(q.w)); pts, wq, _ = q.nodes(n_per, seed=10 + l)
        C = net.f(l, pts); Kq = Dd.GM(wq, C, iso=s2)
        qt, qn = E2.fit_mix(C, wq, s2, S, seed=l)
        nP = 2 ** 14
        hP = node_sample(Kq, nP, rng, sigma); hQ = gm_sample(qn, nP, rng)
        lP_P, lQ_P, lP_Q, lQ_Q = Kq.logpdf(hP), qn.logpdf(hP), Kq.logpdf(hQ), qn.logpdf(hQ)
        est = renyi_mis(lP_P, lQ_P, lP_Q, lQ_Q, BETAS)
        cand = np.r_[hP[:2500], C[rng.choice(len(C), 2500, p=wq / wq.sum())] + sigma * rng.standard_normal((2500, d))]
        dinf, _ = Dd.sup_log_ratio(Kq, qn, cand, steps=30, top=16)
        curves.append(Curve(est, dinf)); qs.append(qn); Ks.append((pts, wq))
        loc.append(dict(layer=l + 1, est={str(k): v for k, v in est.items()}, dinf_search=float(dinf), dinf_used=curves[-1].dinf))
        print(f'S {S} layer {l + 1}: KL {est[1.0]:.2f} D2 {est[2.0]:.2f} D4 {est[4.0]:.2f} D8 {est[8.0]:.2f} D64 {est[64.0]:.2f} Dinf {dinf:.2f}', flush=True)
        q = qn
    t_chain = time.time() - t0
    Bs, Binf = compose(curves)
    comp = [dict(layer=l + 1, **{f'B{int(a)}': Bval(Bs[l], a) for a in ALPHAS}, Binf=Binf[l],
                 naive_sum={str(int(a)): float(sum(curves[j](a) for j in range(l + 1))) for a in ALPHAS}) for l in range(L)]
    for c in comp: print('composed', c, flush=True)
    # ---- direct D_a(p_l || q_l): p_l density from a bank of true parents
    direct = []
    xb, hb = true_paths(net, sigma, 2 ** 16, np.random.default_rng(5)); xf, hf = true_paths(net, sigma, 2 ** 13, np.random.default_rng(6))
    for l in range(L):
        par = xb if l == 0 else hb[l - 1]
        P = Dd.GM(np.ones(len(par)), net.f(l, par), iso=s2)
        hP = hf[l]; hQ = gm_sample(qs[l], len(hP), rng)
        est = renyi_mis(P.logpdf(hP), qs[l].logpdf(hP), P.logpdf(hQ), qs[l].logpdf(hQ), [1.0] + ALPHAS)
        direct.append(dict(layer=l + 1, **{str(k): v for k, v in est.items()}))
        print('direct', direct[-1], flush=True)
    # ---- truth per depth: plain MC (2^20) + subset simulation (3 runs)
    t1 = time.time(); xm, hm = true_paths(net, sigma, 2 ** 20, np.random.default_rng(7)); rate = 2 ** 20 / (time.time() - t1)
    Dfull = DIN + L * d
    pcn = lambda cur, rho, g: (rho * cur[0] + math.sqrt(1 - rho ** 2) * g.standard_normal(cur[0].shape),)
    events = []
    for l in range(L):
        wl, bl = ro[l]; ym = hm[l] @ wl + bl
        def score(st, l=l, wl=wl, bl=bl):
            u = st[0]; h = u[:, :DIN]; nz = u[:, DIN:].reshape(len(u), L, d)
            for j in range(l + 1): h = net.f(j, h) + sigma * nz[:, j]
            return h @ wl + bl
        curves_t = [Dc.sus(score, lambda n, g: (g.standard_normal((n, Dfull)),), pcn, np.random.default_rng(100 + 10 * l + r))[0] for r in range(3)]
        targets = [0.5, 0.25, 0.1, 0.05] + [2.0 ** -k for k in range(6, 42, 2)]
        tmax = min(max(t for t, p in c if p > 0) for c in curves_t)
        tg = np.linspace(np.quantile(ym, 0.4), tmax, 600)
        Psus = np.exp(np.mean([np.log(np.maximum(Dc.interp_p(c, tg), 1e-300)) for c in curves_t], 0))
        Pmc = np.array([(ym > t).mean() for t in tg]); Pg = np.where(Pmc * len(ym) >= 64, Pmc, Psus)
        rows = []
        for pt in targets:
            if Pg.min() > pt: continue
            i = int(np.argmin(np.abs(np.log2(np.maximum(Pg, 1e-300)) - math.log2(pt)))); t = float(tg[i]); p = float(Pg[i])
            r = dict(t=t, p=p, log2p=math.log2(p))
            r['q_l'] = qs[l].prob_halfspace(wl, t - bl)
            if l >= 1:
                pts, wq, _ = qs[l - 1].nodes(max(256, 2 ** 15 // len(qs[l - 1].w)), seed=99)
                g = norm.sf((t - bl - net.f(l, pts) @ wl) / (sigma * np.linalg.norm(wl)))
            else:
                xx = np.random.default_rng(3).standard_normal((2 ** 16, DIN)); wq = np.ones(len(xx))
                g = norm.sf((t - bl - net.f(0, xx) @ wl) / (sigma * np.linalg.norm(wl)))
            r['Kq_prev'] = float((wq @ g) / wq.sum())
            for a in ALPHAS:
                ap = a / (a - 1); Eg = float((wq @ g ** ap) / wq.sum())
                Bl = Bval(Bs[l], a); Bp = Bval(Bs[l - 1], a) if l >= 1 else 0.0
                Dl = direct[l][str(a)]; Dp = direct[l - 1][str(a)] if l >= 1 else 0.0
                r[f'out_a{int(a)}'] = (1 - 1 / a) * (math.log2(max(r['q_l'], 1e-300)) + Bl)
                r[f'tk_a{int(a)}'] = (1 - 1 / a) * (math.log2(max(Eg, 1e-300)) + Bp)
                r[f'out_oracle_a{int(a)}'] = (1 - 1 / a) * (math.log2(max(r['q_l'], 1e-300)) + Dl)
                r[f'tk_oracle_a{int(a)}'] = (1 - 1 / a) * (math.log2(max(Eg, 1e-300)) + Dp)
            r['tk_dinf'] = math.log2(max(r['Kq_prev'], 1e-300)) + (Binf[l - 1] if l >= 1 else 0.0)
            rows.append(r)
        events.append(dict(layer=l + 1, rows=rows))
        print('events layer', l + 1, [(round(r['log2p'], 1), round(min(0, min(r[f'tk_a{int(a)}'] for a in ALPHAS)), 1)) for r in rows], flush=True)
    t_total_bound = t_chain                                   # compute charged to the deductive bound (chain + local divergences)
    n_eq = int(rate * t_total_bound)
    for ev in events:
        for r in ev['rows']:
            k = int(round(r['p'] * n_eq)); r['samp_log2'] = math.log2(float(betad.ppf(0.95, k + 1, n_eq - k)))
    out = dict(sigma=sigma, S=S, local=loc, composed=comp, direct=direct, events=events, rate=rate, t_chain=t_chain, n_eq=n_eq,
               seconds=time.time() - t_all,
               inequality='D_a(P||R) <= ((a-1/p)/(a-1)) D_{pa}(P||Q) + D_{q(a-1/p)}(Q||R), 1/p+1/q=1 (Mironov 2017 Prop. 11); '
                          'with data processing D_b(KP||KQ) <= D_b(P||Q), P=p_l, Q=K_l q_{l-1}, R=q_l')
    json.dump(out, open(os.path.join(E2.OUT, f'holder_s{sigma}_S{S}.json'), 'w'), indent=1)
    print('done', time.time() - t_all, flush=True)


if __name__ == '__main__':
    run(float(sys.argv[1]), int(sys.argv[2]))
