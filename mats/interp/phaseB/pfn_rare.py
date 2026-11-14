"""(1) Rare outputs of the noisy tiny PFN: ground truth by subset simulation over ALL randomness (episode + every noise
site), validated against plain Monte Carlo; plain-sampling certified bound at equal compute.

Event A_t = {|mu_model - mu_Bayes| / sd_Bayes > t}, mu_model = mean of the mixture head for ONE noisy forward pass.
All randomness is a standard-normal vector u (labels y, reading errors e (scaled by sqrt(v)), source / question IDs,
noise at every site) plus the discrete reliabilities v. Subset simulation: levels at the p0-quantile of the score,
pCN moves (rho) on u and Gibbs re-draws of one v_s, each move accepted iff the score stays above the level.

usage: python pfn_rare.py RUN_DIR  -> RUN_DIR/rare.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W

torch.set_num_threads(1)


class Sim:
    def __init__(self, model, meta):
        self.m, self.sg, self.ds = model, meta['sigma'], meta['ds']
        self.sites = model.site_names(); T, d = 16, model.d
        self.shapes = {s: (T, 2 * d if s.startswith('m') else d) for s in self.sites}
        self.nw = W.Q + W.S * W.Q + W.S * self.ds + W.Q * self.ds
        self.dim = self.nw + sum(a * b for a, b in self.shapes.values())

    def split(self, u):
        n = len(u); i = 0
        def take(k):
            nonlocal i; out = u[:, i:i + k]; i += k; return out
        y = take(W.Q); e = take(W.S * W.Q).reshape(n, W.S, W.Q)
        semb = take(W.S * self.ds).reshape(n, W.S, self.ds) / 4; qemb = take(W.Q * self.ds).reshape(n, W.Q, self.ds) / 4
        Z = {s: take(a * b).reshape(n, a, b) for s, (a, b) in self.shapes.items()}
        return y, e, semb, qemb, Z

    def score(self, u, vidx, want_parts=False):
        y, e, semb, qemb, Z = self.split(u)
        v = W.VLEV[vidx]; o = y[:, None, :] + e * np.sqrt(v)[:, :, None]
        X, valid = W.assemble(semb, qemb, o, y)
        d = dict(vidx=vidx, v=v, y=y, o=o)
        _, _, mom, _ = W.references(d)
        mb, vb = mom['bayes']
        with torch.no_grad():
            Xt = torch.from_numpy(X); allow = self.m.allow_mask(torch.from_numpy(valid)); sg = self.sg
            zt = {s: torch.as_tensor(Z[s], dtype=torch.float32) for s in self.sites}
            x = self.m.f_emb(Xt) + sg * zt['emb']
            for l in range(1, self.m.depth + 1):
                xa = self.m.f_a(l, x, allow) + sg * zt[f'a{l}']
                mm = self.m.f_m(l, xa) + sg * zt[f'm{l}']
                x = self.m.f_r(l, xa, mm) + sg * zt[f'r{l}']
            mu, ls, lg = self.m.head(x)
            w = torch.softmax(lg, -1); mean = (w * mu).sum(-1).numpy().astype(np.float64)
        s = np.abs(mean - mb) / np.sqrt(vb)
        return (s, mean, mb, vb) if want_parts else s

    def score_batched(self, u, vidx, bs=4096):
        return np.concatenate([self.score(u[i:i + bs], vidx[i:i + bs]) for i in range(0, len(u), bs)])


def prior(sim, n, rng):
    return rng.standard_normal((n, sim.dim)).astype(np.float32), rng.integers(0, W.NV, (n, W.S))


def subset_sim(sim, rng, N=2000, p0=0.1, rho=0.8, tmax=1e9, max_levels=13):
    """Returns thresholds and tail probabilities (piecewise curve) and forward-pass count."""
    u, v = prior(sim, N, rng); s = sim.score_batched(u, v); passes = N
    curve = []; pk = 1.0; levels = []; last_acc = 0.3
    for k in range(max_levels):
        order = np.argsort(-s); nk = int(p0 * N)
        gamma = s[order[nk - 1]]
        # record the tail curve within this level from the current particles
        for t in np.quantile(s, np.linspace(0, 1 - p0, 10)):
            curve.append((float(t), pk * float((s > t).mean())))
        if gamma >= tmax:
            for t in np.linspace(gamma, s.max(), 5):
                curve.append((float(t), pk * float((s > t).mean())))
            break
        pk *= p0; levels.append((float(gamma), pk))
        seeds_u, seeds_v, seeds_s = u[order[:nk]], v[order[:nk]], s[order[:nk]]
        reps = N // nk
        cu, cv, cs = np.repeat(seeds_u, reps, 0), np.repeat(seeds_v, reps, 0), np.repeat(seeds_s, reps, 0)
        acc = 0
        rho = min(0.98, rho + 0.05) if k > 0 and last_acc < 0.2 else (max(0.5, rho - 0.05) if k > 0 and last_acc > 0.5 else rho)
        for step in range(8):
            pu = rho * cu + math.sqrt(1 - rho ** 2) * rng.standard_normal(cu.shape).astype(np.float32)
            pv = cv.copy(); j = rng.integers(0, W.S, len(cv)); pv[np.arange(len(cv)), j] = rng.integers(0, W.NV, len(cv))
            ps = sim.score_batched(pu, pv); passes += len(pu)
            ok = ps >= gamma; acc += ok.mean()
            cu[ok], cv[ok], cs[ok] = pu[ok], pv[ok], ps[ok]
        u, v, s = cu, cv, cs; last_acc = acc / 8
        print(f'rho {rho:.2f} level {k}: gamma {gamma:.3f}  p {pk:.2e}  accept {acc / 8:.2f}  passes {passes}', flush=True)
    return sorted(curve), levels, passes


def run(run_dir):
    model, meta = N.load_run(run_dir); sim = Sim(model, meta)
    rng = np.random.default_rng(123)
    t0 = time.time()
    # plain MC for validation of the upper part of the curve
    nmc = 2 ** 19; ss = []
    for i in range(0, nmc, 8192):
        u, v = prior(sim, 8192, rng); ss.append(sim.score_batched(u, v))
    ss = np.concatenate(ss); t_mc = time.time() - t0
    ts = np.linspace(0, 12, 121)
    mc = [(float(t), float((ss > t).mean())) for t in ts]
    print(f'MC {nmc} passes in {t_mc:.0f}s; p(s>1) {np.mean(ss > 1):.2e} p(s>2) {np.mean(ss > 2):.2e} p(s>3) {np.mean(ss > 3):.2e}', flush=True)
    runs = []
    for rep in range(3):
        t1 = time.time()
        curve, levels, passes = subset_sim(sim, np.random.default_rng(1000 + rep))
        runs.append(dict(curve=curve, levels=levels, passes=passes, seconds=time.time() - t1))
        json.dump(dict(sigma=meta['sigma'], depth=meta['depth'], mc=mc, nmc=nmc, t_mc=t_mc, rate=nmc / t_mc, sus=runs),
                  open(os.path.join(run_dir, 'rare.json'), 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
