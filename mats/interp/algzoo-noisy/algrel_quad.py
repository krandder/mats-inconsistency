"""Quadrature check for algrel.py (review 2, finding 1): the deductive average of the algorithm-indexed description with
the default grid, a denser grid, and a Monte Carlo integral of the same eps_hat over 2^17 inputs.
usage: python algrel_quad.py MODEL.npz OUT.json"""
import sys, json
src = open('algrel.py').read()
head = src[:src.index("flc = [0.0]")]
model, out = sys.argv[1], sys.argv[2]
sys.argv = ['algrel.py', model, 'unused.json', 'unused.json', 'skipdag']
g = {'__name__': 'algrel_quad'}
exec(compile(head, 'algrel.py', 'exec'), g)
import numpy as np
res = {}
res['quad_default'] = g['desc_average'](g['W'], g['r'], g['CL'][g['n']])
os2 = g['os2_nodes']
g['os2_nodes'] = lambda tn, KS=24, KM=8, KL=6: os2(tn, KS=96, KM=32, KL=24)
res['quad_dense'] = g['desc_average'](g['W'], g['r'], g['CL'][g['n']])
rng = np.random.default_rng(99)
vals = []
for i in range(8):
    x = rng.standard_normal((2 ** 14, g['n'])); A = g['alg'].alg_states(x)
    vals.append(g['eps_hat'](x, A, g['CL'][g['n']], g['W_oh']))
v = np.concatenate(vals)
res['mc_integral'] = float(v.mean()); res['mc_se'] = float(v.std() / np.sqrt(len(v)))
json.dump(res, open(out, 'w'), indent=1); print(res)
