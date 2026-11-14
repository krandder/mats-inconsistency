"""Diagnostic for Amendment 7 (not an estimator): how far the TOP3 quadrature is from the exact integral of its own
per-node model. The exact integral is estimated by averaging the per-input prediction over sampled inputs (the
prediction at each input's own top-3 values and positions). usage: python top3_conv.py MODEL.npz ESTIM.json OUT.json LOG2N BUDGETS"""
import sys, json, time
import numpy as np
import az, top3
t0 = time.time()
path, epath, out, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
buds = [tuple(int(v) for v in b.split('x')) for b in sys.argv[5].split(',')]
W0, r0, sigma = az.load_npz(path)
m = az.model_from(W0, r=r0, sigma=sigma); m.calibrate_r(2 ** 18, iters=2)        # model 0 of the estim family
T3 = top3.Top3(W0, m.r.double().numpy(), sigma)
rng = np.random.default_rng(2468)
x = rng.standard_normal((2 ** L, T3.n)); pr = T3.predict_inputs(x)
res = dict(model=path, truth=json.load(open(epath))['truths'][0], mc_integral=float(pr.mean()),
           mc_integral_se=float(pr.std() / np.sqrt(len(pr))), n=len(pr), quad=[])
print('MC integral of the node model %.5f +- %.5f (truth %.5f)' % (res['mc_integral'], res['mc_integral_se'], res['truth']), flush=True)
for (KS, KG) in buds:
    t1 = time.time(); p, fl, _, _ = T3.estimate(KS, KG)
    res['quad'].append(dict(KS=KS, KG=KG, p=p, flops=fl, seconds=time.time() - t1)); print((KS, KG), '%.5f flops %.3g' % (p, fl), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0; json.dump(res, open(out, 'w'), indent=1)
