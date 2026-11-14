"""Part A, interaction structure (Kelvin's ask via the coordinator, 14:05Z): which noise injections interact, and why.
Model S stationary (sigma = 1); window of J = 4n layers before layer 0, deep past = the state at -J-1 of each sample.
Noise in the window is parametrised causally: xi_l = eta_l e_l + (I - e_l e_l^T) zeta_l, e_l = direction of the
pre-noise activation a_l of THIS path, eta_l ~ N(0,1), zeta_l ~ N(0, I_n) iid; so xi_l ~ N(0, I) exactly on every path,
and "radial" (eta) and "angular" (zeta) coordinates are well defined blocks.
Observables at layer 0: h (h_0), logR (log|h_0|), u (h_0/|h_0|), a (relu(W_1 h_0)), g (output gates 1{W_1 h_0 > 0}).
usage: partI.py MODE n seed [--P 512] [--model S|clamp] [--out file]
MODE curves: refresh each layer's (eta, zeta) with probability eps (eps in 0.01, 0.03, 0.1, 0.3, 1): G_f(eps) =
  E|f - f_eps|^2 / (2 V_f), V_f = E Var(f | deep past) estimated by eps = 1; mean dimension ~ G(0.01)/0.01; also
  refresh ALL radial coordinates only, ALL angular only, and radial coordinates with probability eps.
MODE pairs: superset importance Y2_jk = E[(f - f^(j) - f^(k) + f^(jk))^2]/4 / V_f for ages j < k on a grid.
MODE trunc: closed index of the most recent d layers C(d) = 1 - E|f - f^(older than d from another sample)|^2/(2V)
  and the sharing floor S_D(d) = 1 - E|f - f^(recent d refreshed)|^2/(2V), V the full stationary variance.
model clamp: after every layer h <- Rbar h/|h|, Rbar = sqrt(2/(5 ln 2)) n (the median stationary radius): removes the
  radius feedback (mechanism a) and keeps gate switching (mechanism b).
MODE kesten (n only): the annealed scalar chain Q' = A Q + B; eps-curves of Q, sqrt Q, log Q over per-layer blocks.
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=512); ap.add_argument('--model', default='S'); ap.add_argument('--out')
a = ap.parse_args(); n, seed, P = a.n, a.seed, a.P
t0 = time.time()
EPS = [0.01, 0.03, 0.1, 0.3, 1.0]
J = 4 * n
RBAR = math.sqrt(2 / (5 * math.log(2))) * n
res = dict(mode=a.mode, n=n, seed=seed, P=P, model=a.model, J=J)


def blk(kind, l, v, shape):
    return np.random.default_rng([8484, seed, n, l + st.OFF, v, kind]).standard_normal(shape)


if a.mode == 'kesten':
    rng = np.random.default_rng([8585, n])
    M = 20000; Jk = 4 * n

    def draw(shape, r):
        Mb = r.binomial(n, 0.5, shape)
        U = r.chisquare(np.maximum(Mb, 1e-9), shape) * (Mb > 0)
        V = r.chisquare(np.maximum(n - Mb, 1e-9), shape) * (Mb < n)
        return 2 * U / n, (U + V) / n
    # deep past: run each chain 30n steps from 0
    Q0 = np.zeros(M)
    for _ in range(30 * n):
        A_, B_ = draw(M, rng); Q0 = A_ * Q0 + B_
    A0, B0 = draw((Jk, M), rng); A1, B1 = draw((Jk, M), rng)

    def run(mask):
        Q = Q0.copy()
        for t in range(Jk):          # t = 0 is the oldest layer in the window
            A_ = np.where(mask[t], A1[t], A0[t]); B_ = np.where(mask[t], B1[t], B0[t])
            Q = A_ * Q + B_
        return Q
    base = run(np.zeros((Jk, M), bool))
    obs = lambda Q: dict(Q=Q, sqrtQ=np.sqrt(Q), logQ=np.log(Q))
    fb = obs(base)
    curves = {k: {} for k in fb}
    for eps in EPS:
        mask = rng.random((Jk, M)) < eps
        fe = obs(run(mask))
        for k in fb:
            curves[k][str(eps)] = float(np.mean((fb[k] - fe[k]) ** 2) / 2)
    out = {}
    for k in fb:
        V = curves[k]['1.0']
        out[k] = {e: v / V for e, v in curves[k].items()}
        out[k]['mean_dim'] = out[k]['0.01'] / 0.01
    res['curves'] = out
else:
    Wf = st.WCache(seed, n, max_bytes=500e6)
    clamp = a.model == 'clamp'

    def step(h, l, eta, zeta):
        apre = np.maximum(h @ Wf(l).T, 0.0)
        nr = np.linalg.norm(apre, axis=1, keepdims=True)
        e = apre / np.where(nr > 0, nr, 1.0)
        zp = zeta - np.sum(zeta * e, 1, keepdims=True) * e
        hn = apre + eta[:, None] * e + zp
        if clamp:
            hn = hn * (RBAR / np.maximum(np.linalg.norm(hn, axis=1, keepdims=True), 1e-300))
        return hn

    def observe(h):
        R = np.linalg.norm(h, axis=1)
        z1 = h @ Wf(1).T
        return dict(h=h, logR=np.log(R), u=h / R[:, None], a=np.maximum(z1, 0.0), g=(z1 > 0).astype(float))

    # deep past
    h = np.zeros((P, n)); rng0 = np.random.default_rng([8686, seed, n])
    for l in range(-J - 12 * n, -J):
        apre = np.maximum(h @ Wf(l).T, 0.0)
        h = apre + rng0.standard_normal((P, n))
        if clamp:
            h = h * (RBAR / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300))
    hdeep = h

    def run_window(h_start, l_start, choose):
        """choose(l) -> (eta, zeta) for layer l; runs l_start..0."""
        h = h_start
        for l in range(l_start, 1):
            eta, zeta = choose(l)
            h = step(h, l, eta, zeta)
        return h

    base_choice = lambda l: (blk(0, l, 0, (P,)), blk(1, l, 0, (P, n)))
    states = {}
    h = hdeep
    for l in range(-J, 1):
        eta, zeta = base_choice(l)
        h = step(h, l, eta, zeta)
        states[l] = h
    f0 = observe(states[0])
    keys = list(f0.keys())
    sqd = lambda x, y: float(np.sum((x - y) ** 2)) / x.shape[0] if x.ndim > 1 else float(np.sum((x - y) ** 2)) / x.shape[0]

    if a.mode == 'curves':
        out = {k: {} for k in keys}
        rngm = np.random.default_rng([8787, seed, n])
        for eps in EPS:
            masks = {l: rngm.random(P) < eps for l in range(-J, 1)}

            def ch(l, masks=masks):
                e0, z0 = base_choice(l)
                e1, z1 = blk(0, l, 1, (P,)), blk(1, l, 1, (P, n))
                m = masks[l]
                return np.where(m, e1, e0), np.where(m[:, None], z1, z0)
            fe = observe(run_window(hdeep, -J, ch))
            for k in keys:
                out[k][str(eps)] = sqd(f0[k], fe[k]) / 2
        for label, which in (('radial_all', 'eta'), ('angular_all', 'zeta')):
            def ch(l, which=which):
                e0, z0 = base_choice(l)
                if which == 'eta':
                    return blk(0, l, 1, (P,)), z0
                return e0, blk(1, l, 1, (P, n))
            fe = observe(run_window(hdeep, -J, ch))
            for k in keys:
                out[k][label] = sqd(f0[k], fe[k]) / 2
        for eps in (0.01, 0.1):
            masks = {l: rngm.random(P) < eps for l in range(-J, 1)}

            def ch(l, masks=masks):
                e0, z0 = base_choice(l)
                return np.where(masks[l], blk(0, l, 1, (P,)), e0), z0
            fe = observe(run_window(hdeep, -J, ch))
            for k in keys:
                out[k][f'radial_eps{eps}'] = sqd(f0[k], fe[k]) / 2
        Vtot = {k: float(np.sum(np.var(f0[k], 0))) if f0[k].ndim > 1 else float(np.var(f0[k])) for k in keys}
        norm = {}
        for k in keys:
            V = out[k]['1.0']
            d = {kk: vv / V for kk, vv in out[k].items()}
            d['mean_dim'] = d['0.01'] / 0.01
            d['mean_dim_eps0.03'] = d['0.03'] / 0.03
            d['radial_mean_dim'] = (d['radial_eps0.01'] / 0.01) / max(d['radial_all'], 1e-300)
            d['V_window_over_V_total'] = V / max(Vtot[k], 1e-300)
            # one-parameter fit: |S| - 1 ~ Poisson(mu): G(eps) = 1 - (1 - eps) exp(-mu eps)
            mus = np.linspace(0, 30, 3001)
            errs = [sum((d[str(e)] - (1 - (1 - e) * math.exp(-mu * e))) ** 2 for e in EPS[:-1]) for mu in mus]
            mu = float(mus[int(np.argmin(errs))])
            d['poisson_mu'] = mu; d['poisson_fit_rms'] = float(math.sqrt(min(errs) / 4))
            d['p1_additive_fit'] = math.exp(-mu)
            from scipy.stats import poisson
            d['superposition_dim_99_fit'] = int(1 + poisson.ppf(0.99, mu)) if mu > 0 else 1
            norm[k] = d
        res['curves'] = norm

    elif a.mode == 'pairs':
        ages = [j for j in sorted(set([0, 2, 8, 32, n // 2, n, 2 * n])) if j < J]
        single = {}

        def ref(lset):
            def ch(l):
                if l in lset:
                    return blk(0, l, 1, (P,)), blk(1, l, 1, (P, n))
                return base_choice(l)
            lo = min(lset)
            return observe(run_window(states[lo - 1] if lo - 1 >= -J else hdeep, lo, ch))
        for j in ages:
            single[j] = ref({-j})
        Vw = {}
        # window variance by refreshing everything in the window
        allr = observe(run_window(hdeep, -J, lambda l: (blk(0, l, 1, (P,)), blk(1, l, 1, (P, n)))))
        for k in keys:
            Vw[k] = sqd(f0[k], allr[k]) / 2
        tab = []
        for ii, j in enumerate(ages):
            for k2 in ages[ii + 1:]:
                fjk = ref({-j, -k2})
                row = dict(j=j, k=k2)
                for k in keys:
                    D = f0[k] - single[j][k] - single[k2][k] + fjk[k]
                    row[k] = float(np.sum(D * D) / D.shape[0] / 4) / max(Vw[k], 1e-300)
                tab.append(row)
        tot = []
        for j in ages:
            row = dict(j=j)
            for k in keys:
                row[k] = sqd(f0[k], single[j][k]) / 2 / max(Vw[k], 1e-300)
            tot.append(row)
        res.update(ages=ages, pairs=tab, totals=tot)

    elif a.mode == 'trunc':
        Vtot = {k: float(np.sum(np.var(f0[k], 0))) if f0[k].ndim > 1 else float(np.var(f0[k])) for k in keys}
        ds = [d for d in sorted(set([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, n // 2, n, 2 * n, 4 * n])) if d <= J]
        rows = []
        for d in ds:
            lo = -d + 1
            start_own = states[lo - 1] if lo - 1 >= -J else hdeep
            start_other = np.roll(start_own, 1, axis=0)
            f_old = observe(run_window(start_other, lo, base_choice))
            f_rec = observe(run_window(start_own, lo, lambda l: (blk(0, l, 1, (P,)), blk(1, l, 1, (P, n)))))
            row = dict(d=d)
            for k in keys:
                row[f'C_{k}'] = 1 - sqd(f0[k], f_old[k]) / 2 / max(Vtot[k], 1e-300)
                row[f'SD_{k}'] = 1 - sqd(f0[k], f_rec[k]) / 2 / max(Vtot[k], 1e-300)
            rows.append(row)
        res.update(rows=rows)

res['sec'] = time.time() - t0
s = json.dumps(res, default=float)
if a.out:
    open(a.out, 'w').write(s)
print(s[:3000], flush=True)
