"""Cut matrix for one (activation, configuration): the cut-rank lane's design with a general activation.

usage: python3 scratch/run_cut_act.py NAME P ARCH n c Lp F B Np [--S 8] [--control] [--name RUN]
  NAME/P: activation (P = 'none' if no parameter), ARCH: plain | glu. Heavy: run under nice -n 19 ionice -c3.
Fronts: layers 1..c of independent nets (lib/gnet.py, pre-activation second moment q0 = 2 at every layer); backs:
layers c+1..c+Lp of independent nets. Inputs x ~ N(0, I_n) (no radius integration: the activation is not
homogeneous). M[f, s, b, j] = mean over Np inputs of split s of output unit j; V[f, b, j] = Var_x of that output
(all S*Np inputs, unbiased). --control: identical-front control (front 0 repeated F times, independent inputs per
row; true rank 1). Saves runs/RUN.npz and appends a JSON summary to logs/runs.jsonl.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import gnet as G

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('p'); ap.add_argument('arch')
for k in ('n', 'c', 'Lp', 'F', 'B', 'Np'):
    ap.add_argument(k, type=int)
ap.add_argument('--S', type=int, default=8)
ap.add_argument('--control', action='store_true')
ap.add_argument('--run', default='')
ap.add_argument('--bias_frac', type=float, default=0.0)
a = ap.parse_args()
p = None if a.p == 'none' else float(a.p)
spec = G.Spec(a.name, p, a.arch, bias_frac=a.bias_frac)
n, c, Lp, F, B, Np, S = a.n, a.c, a.Lp, a.F, a.B, a.Np, a.S
run = a.run or f'cut_{spec.label}_n{n}_c{c}_L{Lp}' + ('_ctrl' if a.control else '')
tag = G.tag_of('tgcut', spec.label, n, c, Lp)
fid = (lambda f: 0) if a.control else (lambda f: f)
t0 = time.time()

# ---- fronts: cut states, columns ordered (front, split, sample) ------------------------------------------------
H = np.empty((n, F * S * Np), dtype=np.float32)
for f in range(F):
    Wf = G.weights(spec, n, c + Lp, tag, 0, fid(f))[:c]          # a front = first c layers of its own net
    bf = G.biases(spec, n, c + Lp, tag, 0, fid(f))
    bf = None if bf is None else bf[:c]
    for s in range(S):
        X = G.rng(tag, 2, f, s).standard_normal((n, Np)).astype(np.float32)
        H[:, (f * S + s) * Np:(f * S + s + 1) * Np] = G.fwd_cols(spec, X, Wf, bf)
t_front = time.time() - t0

# ---- backs ---------------------------------------------------------------------------------------------------------
M = np.empty((F, S, B, n), dtype=np.float64)
E2 = np.empty((F, B, n), dtype=np.float64)
chunk = max(1, int((160 << 20) // (S * Np * n * 4)))
for b in range(B):
    Wb = G.weights(spec, n, c + Lp, tag, 1, b)[c:]               # a back = last Lp layers of its own net
    bb = G.biases(spec, n, c + Lp, tag, 1, b)
    bb = None if bb is None else bb[c:]
    for f0 in range(0, F, chunk):
        f1 = min(F, f0 + chunk)
        Aout = G.fwd_cols(spec, H[:, f0 * S * Np:f1 * S * Np].copy(), Wb, bb).reshape(n, f1 - f0, S, Np)
        M[f0:f1, :, b, :] = Aout.mean(3, dtype=np.float64).transpose(1, 2, 0)
        E2[f0:f1, b, :] = np.mean(np.square(Aout, dtype=np.float64), axis=(2, 3)).T
t_back = time.time() - t0 - t_front
Ntot = S * Np
Mbar = M.mean(1)
V = (E2 - Mbar ** 2) * Ntot / (Ntot - 1)

# ---- diagnostics: realised pre-activation scale per layer (2 nets), energy concentration ---------------------
s_pre = [G.preact_std(spec, G.weights(spec, n, c + Lp, tag, 9, i), n, N=8192, seed=i, bs=G.biases(spec, n, c + Lp, tag, 9, i)) for i in range(2)]
e_front = (V + Mbar ** 2).mean(axis=(1, 2)); e_back = (V + Mbar ** 2).mean(axis=(0, 2))


def share_top(e, q=0.8):   # fraction of units carrying share q of the energy
    es = np.sort(e)[::-1]; cs = np.cumsum(es) / max(es.sum(), 1e-300)
    return float((np.searchsorted(cs, q) + 1) / len(e))


np.savez_compressed(os.path.join(HERE, '..', 'runs', run + '.npz'), M=M, V=V)
summ = dict(run=run, act=spec.label, arch=a.arch, n=n, c=c, Lp=Lp, F=F, B=B, Np=Np, S=S, control=a.control,
            g2=spec.g2, chi=spec.chi, s_pre_mean=float(np.mean(s_pre)), s_pre_first=float(s_pre[0][0]),
            s_pre_last=float(np.mean([sp_[-1] for sp_ in s_pre])), Vbar=float(V.mean()),
            EM2=float(np.mean(Mbar ** 2)), front_share80=share_top(e_front), back_share80=share_top(e_back),
            t_front=round(t_front, 1), t_back=round(t_back, 1), t_total=round(time.time() - t0, 1))
print(json.dumps(summ), flush=True)
with open(os.path.join(HERE, '..', 'logs', 'runs.jsonl'), 'a') as fh:
    fh.write(json.dumps(summ) + '\n')
