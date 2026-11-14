# E3 hawk sanity sim: optimism gap vs n for four forecaster types. CPU, numpy, ~10 s.
# gap(n) = E[f(argmax f) - v(argmax f)], v = Bayes success prob of the candidate.
import numpy as np, json, sys
rng = np.random.default_rng(0)
S, N = 4000, 64                      # states, candidates per state
ns = [1, 2, 4, 8, 16, 32, 64]
lg = lambda p: np.log(p/(1-p)); sg = lambda z: 1/(1+np.exp(-z))
def draw_p(shape): return np.clip(rng.beta(1, 3, shape), 1e-4, 1-1e-4)
s = 1.0                              # forecaster signal noise (logit units)
# fit a calibrated map x -> E[p|x] on a training set from the SAME proposal distribution
pt = draw_p(400000); xt = lg(pt) + s*rng.standard_normal(pt.shape)
bins = np.quantile(xt, np.linspace(0, 1, 201)); idx = np.clip(np.digitize(xt, bins)-1, 0, 199)
cal = np.array([pt[idx == k].mean() for k in range(200)])
def calibrated(x):
    return cal[np.clip(np.digitize(x, bins)-1, 0, 199)]
p = draw_p((S, N)); x = lg(p) + s*rng.standard_normal((S, N))
# shifted proposer: fraction q of candidates carry a bias b in the forecaster's signal it never trained on
q, b = 0.1, 1.5
mask = rng.random((S, N)) < q; xs = x + b*mask
arms = {
  'oracle f=v': p,
  'calibrated E[v|x], in-distribution': calibrated(x),
  'unbiased-noise f=sg(lg v+eps) (LLM-like)': sg(x),
  'calibrated E[v|x], shifted proposer (10% exploit +1.5 logit)': calibrated(xs),
}
y = (rng.random((S, N)) < p).astype(float)   # realised 0/1 outcomes
out = {}
for name, f in arms.items():
    row = {}
    for n in ns:
        fsub, psub, ysub = f[:, :n], p[:, :n], y[:, :n]
        j = fsub.argmax(1); r = np.arange(S)
        gb = fsub[r, j] - psub[r, j]; gy = fsub[r, j] - ysub[r, j]
        row[n] = dict(gap_bayes=round(gb.mean(), 4), se_bayes=round(gb.std()/np.sqrt(S), 4),
                      gap_realised=round(gy.mean(), 4), se_realised=round(gy.std()/np.sqrt(S), 4),
                      gold=round(psub[r, j].mean(), 4))
    out[name] = row
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for name, row in out.items():
    print(name)
    for n in ns: print('  n=%2d gap_bayes %.4f±%.4f  gap_realised %.4f±%.4f  gold %.4f' % (n, row[n]['gap_bayes'], row[n]['se_bayes'], row[n]['gap_realised'], row[n]['se_realised'], row[n]['gold']))
