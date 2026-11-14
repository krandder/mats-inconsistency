"""Reviewer check (in-distribution copy test + untied independent source). Small, CPU, 1 thread."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/pfn-dedup')
import world as W, split, mech2
from pfn import PFN
torch.set_num_threads(1)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
H = 0.05
rng = np.random.default_rng(77000)
d = W.sample_np(N * 6, rng)
sing = np.array([len(set(l)) == W.S for l in d['lab']])
keep = np.where((d['k'] == 0) & sing)[0][:N]
d = {k: v[keep] for k, v in d.items()}
n = len(keep); ar = np.arange(n)
semb = rng.standard_normal((n, W.S, W.DS)) / 4; qemb = rng.standard_normal((n, W.Q, W.DS)) / 4
gemb = rng.standard_normal((n, W.S, W.DS)) / 4
s0 = rng.integers(0, W.S, n); s1 = (s0 + rng.integers(1, W.S, n)) % W.S
src6 = np.tile(np.arange(W.S), (n, 1)); src7 = np.concatenate([src6, s0[:, None]], 1)
# untied independent 7th source (fresh on all 9 questions, new identity, new singleton group)
fresh = d['y'][:, None, :] + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SG2) + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SE2)
o7 = np.concatenate([d['o'], fresh], 1); semb7 = np.concatenate([semb, rng.standard_normal((n, 1, W.DS)) / 4], 1)
gemb7 = np.concatenate([gemb, rng.standard_normal((n, 1, W.DS)) / 4], 1); lab7 = np.concatenate([d['lab'], np.full((n, 1), 6)], 1)
onew = fresh[:, 0, W.QL]

def shift(o, s, h):
    o = o.copy(); o[ar, s, W.QL] += h; return o

def mu_of(model, keys, o, src, se=semb, ge=gemb, lab=None):
    lab = d['lab'] if lab is None else lab
    X, vm = W.assemble(se, qemb, ge, src, np.ones_like(src, bool), o, d['y'], lab, keys)
    with torch.no_grad():
        return mech2.run(model, torch.tensor(X), torch.tensor(vm))[0].numpy().astype(np.float64)

def weights(f, o, src):
    """dmu/do for source s0 (all its rows) and s1, central differences."""
    w0 = (f(shift(o, s0, H), src) - f(shift(o, s0, -H), src)) / (2 * H)
    w1 = (f(shift(o, s1, H), src) - f(shift(o, s1, -H), src)) / (2 * H)
    return w0, w1

def boot(fn, B=400):
    r = np.random.default_rng(1); v = [fn(r.integers(0, n, n)) for _ in range(B)]
    return float(np.std(v))

out = {'n': n, 'refs': {}, 'models': []}
# references: R1 (additive per-n), R2 (duplicate-blind Bayes on query readings)
coef = split.fit_r1()
def r1f(o, src):
    Q = np.take_along_axis(o[:, :, W.QL], src, 1); c = coef[Q.shape[1]]
    return split.basis(Q).sum(1) @ c[:-1] + c[-1]
def r2f(o, src):
    return split.r2_mean(np.take_along_axis(o[:, :, W.QL], src, 1))
for name, f in (('R1', r1f), ('R2', r2f)):
    w0, w1 = weights(f, d['o'], src7); w60, w61 = weights(f, d['o'], src6)
    out['refs'][name] = {'rho7': float(w0.mean() / w1.mean()), 'w6': float(w60.mean())}
    print(name, out['refs'][name], flush=True)
for f in sorted(glob.glob(sys.argv[1] + '/L*_k*_s*/meta.json')):
    meta = json.load(open(f)); keys = meta['keys']
    torch.manual_seed(meta['seed']); model = PFN(meta['fin'], meta['depth'])
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(f), 'model.pt'), map_location='cpu')); model.eval()
    g = lambda o, src: mu_of(model, keys, o, src)
    w0, w1 = weights(g, d['o'], src7)
    w60, w61 = weights(g, d['o'], src6)
    rho = float(w0.mean() / w1.mean())
    rho_se = boot(lambda i: w0[i].mean() / w1[i].mean())
    mu6 = g(d['o'], src6); mu7c = g(d['o'], src7)
    odup = d['o'][ar, s0, W.QL]
    pull_copy = mech2.slope(mu7c - mu6, odup - mu6)
    mu7i = mu_of(model, keys, o7, src7 * 0 + np.concatenate([src6, np.full((n, 1), 6)], 1), semb7, gemb7, lab7)
    pull_indep_untied = mech2.slope(mu7i - mu6, onew - mu6)
    se_c = boot(lambda i: mech2.slope((mu7c - mu6)[i], (odup - mu6)[i]))
    se_i = boot(lambda i: mech2.slope((mu7i - mu6)[i], (onew - mu6)[i]))
    r = dict(depth=meta['depth'], keys=keys, seed=meta['seed'], conv=meta['converged'], rho7=rho, rho7_se=rho_se,
             w6_s0=float(w60.mean()), w7_s0=float(w0.mean()), w7_s1=float(w1.mean()),
             pull_copy=pull_copy, pull_copy_se=se_c, pull_indep_untied=pull_indep_untied, pull_indep_untied_se=se_i)
    out['models'].append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open('/tmp/rev2/indist.json', 'w'), indent=1)
