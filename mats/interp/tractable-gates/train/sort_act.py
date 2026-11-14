"""Grid T, sorting MLP (PREREG): narrow-deep's residual bias-free sorting recipe with a general activation; several seeds batched.

Task: x ~ N(0, I_16), target sort(x). Residual: h_0 = x; h_l = h_{l-1} + V_l g(W_l h_{l-1}); out = h_L.
  single-gate arms: g(u) = phi(u), W_l (n x n) ~ N(0, 2/n), V_l (n x n) ~ N(0, 1/(n L));
  GLU arms (swiglu: silu(W1 h) * (W2 h); bilinear: (W1 h) * (W2 h)): W1, W2 (m x n) ~ N(0, 2/n), V (n x m) ~ N(0, 1/(m L)),
  m = 11 (3*16*11 = 528 parameters per layer against 512).
Adam lr 2e-3, warm-up steps/32, cosine to 0.01x, batch 1024, grad-norm clip 1.0, no noise. Copied structure from
work/narrow-deep/train/mlp_train.py (not edited there).
Saves per seed: OUT_s{seed}.npz (weights) and OUT_s{seed}.json (final nMSE on 2^18 fresh inputs, log, per-layer RMS
pre-activation at init and at the end, per-unit pre-activation mean/std at the end).
usage: python train/sort_act.py NAME P L SEEDS STEPS OUTPREFIX [MODE]   (P = 'none'; SEEDS like 0,1,2; MODE pin | pn)
NAME tanhb: tanh with trainable per-unit biases (init 0), PREREG Amendment 1 (a bias-free tanh net is odd; sort is not).
"""
import sys, os, time, math, json
import numpy as np
import torch

NAME = sys.argv[1]; P = None if sys.argv[2] == 'none' else float(sys.argv[2]); L = int(sys.argv[3])
seeds = [int(s) for s in sys.argv[4].split(',')]; STEPS = int(sys.argv[5]); OUT = sys.argv[6]
n = 16; M = len(seeds); B = 1024; LR = 2e-3; warm = max(STEPS // 32, 1)
torch.set_num_threads(int(os.environ.get('NTHREADS', '1')))
GLU = NAME in ('swiglu', 'bilinear'); m = 11 if GLU else n


def smoothstep(u):
    a = (u + 1).clamp_min(0); b = (1 - u).clamp_min(0)
    fa = torch.where(a > 0, torch.exp(-1 / torch.where(a > 0, a, torch.ones_like(a))), torch.zeros_like(a))
    fb = torch.where(b > 0, torch.exp(-1 / torch.where(b > 0, b, torch.ones_like(b))), torch.zeros_like(b))
    return torch.where(u >= 1, torch.ones_like(u), torch.where(u <= -1, torch.zeros_like(u), fa / (fa + fb + 1e-300)))


def phi(u):
    if NAME == 'relu':
        return torch.relu(u)
    if NAME == 'softplus':
        return torch.nn.functional.softplus(u, beta=P, threshold=20.0)
    if NAME == 'gelu':
        return torch.nn.functional.gelu(u)
    if NAME in ('silu', 'swiglu'):
        return torch.nn.functional.silu(u)
    if NAME in ('tanh', 'tanhb'):
        return torch.tanh(u)
    if NAME == 'smoothstep':
        return u * smoothstep(u / P)
    if NAME == 'gsrelu':            # post hoc arm: ReLU with integrated Gaussian pre-activation noise of std P
        a = u / P
        return P * (a * 0.5 * torch.erfc(-a / math.sqrt(2.0)) + torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi))
    if NAME == 'bilinear':
        return u
    raise ValueError(NAME)


Ws, W2s, Vs = [], [], []
for s in seeds:
    g = torch.Generator().manual_seed(5000 + s)
    Ws.append(torch.randn(L, m, n, generator=g) * math.sqrt(2.0 / n))
    if GLU:
        W2s.append(torch.randn(L, m, n, generator=g) * math.sqrt(2.0 / n))
    Vs.append(torch.randn(L, n, m, generator=g) * math.sqrt(1.0 / (m * L)))
W = torch.nn.Parameter(torch.stack(Ws)); V = torch.nn.Parameter(torch.stack(Vs)); params = [W, V]
if GLU:
    W2 = torch.nn.Parameter(torch.stack(W2s)); params.append(W2)
BIAS = NAME == 'tanhb'          # PREREG Amendment 1: trainable biases (init 0) for the tanh arm only
MODE = sys.argv[7] if len(sys.argv) > 7 else ''           # optional mode: pin | pn (overrides the PIN env variable)
PIN = (MODE == 'pin') or (MODE == '' and os.environ.get('PIN', '0') == '1')   # Amendment 2: per-unit RMS-pinned gate inputs
PRENORM = MODE == 'pn'          # PREREG Amendment 3: branch input RMSNorm(h) = sqrt(n) h / |h| (no gain)
CLIP_PER_SEED = os.environ.get('CLIP_PER_SEED', '0') == '1'
run_rms = torch.full((M, L, 1, m), math.sqrt(2.0))
if BIAS:
    Bb = torch.nn.Parameter(torch.zeros(M, L, 1, m)); params.append(Bb)
opt = torch.optim.Adam(params, lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1)))))
gen = torch.Generator().manual_seed(777 + sum(seeds) + 100 * L)


def forward(x, record=None, train=False):
    h = x
    for l in range(L):
        hin = h * (math.sqrt(n) / h.pow(2).sum(-1, keepdim=True).add(1e-12).sqrt()) if PRENORM else h
        u = torch.bmm(hin, W[:, l].transpose(1, 2))
        if PIN:
            if train:
                rr = u.pow(2).mean(1, keepdim=True).add(1e-12).sqrt()
                with torch.no_grad():
                    run_rms[:, l] = 0.99 * run_rms[:, l] + 0.01 * rr.detach()
            else:
                rr = run_rms[:, l]
            u = u * (math.sqrt(2.0) / rr)
        if BIAS:
            u = u + Bb[:, l]
        if record is not None:
            record.append(u.detach())
        a = phi(u) * torch.bmm(hin, W2[:, l].transpose(1, 2)) if GLU else phi(u)
        h = h + torch.bmm(a, V[:, l].transpose(1, 2))
    return h


def preact_stats(N=16384, seed=31):
    g2 = torch.Generator().manual_seed(seed)
    rec = []
    with torch.no_grad():
        forward(torch.randn(M, N, n, generator=g2), rec)
    rms = torch.stack([u.pow(2).mean((1, 2)).sqrt() for u in rec], 1)        # (M, L)
    mean = torch.stack([u.mean(1) for u in rec], 1); std = torch.stack([u.std(1) for u in rec], 1)   # (M, L, m)
    return rms, mean, std


VAR_Y = float(np.mean(np.var(np.sort(np.random.RandomState(0).standard_normal((1 << 20, n)), 1), 0)))
if PIN:                                    # initial running statistics from 200 pilot train-mode passes (no optimiser steps)
    with torch.no_grad():
        for _ in range(200):
            forward(torch.randn(M, B, n, generator=gen), train=True)
rms0, _, _ = preact_stats()
t0 = time.time(); log = []; acc = torch.zeros(M); cnt = 0
for step in range(STEPS):
    x = torch.randn(M, B, n, generator=gen)
    y = x.sort(-1).values
    mse = (forward(x, train=True) - y).pow(2).mean((1, 2))
    opt.zero_grad(); mse.sum().backward()
    if CLIP_PER_SEED:                 # post hoc (final review finding 9): clip each seed's gradient by its own norm
        with torch.no_grad():
            nrm = torch.sqrt(sum(p_.grad.reshape(M, -1).pow(2).sum(1) for p_ in params))
            fac = torch.clamp(1.0 / (nrm + 1e-6), max=1.0)
            for p_ in params:
                p_.grad.mul_(fac.view(M, *([1] * (p_.grad.dim() - 1))))
    else:
        torch.nn.utils.clip_grad_norm_(params, 1.0)   # registered recipe (as narrow-deep's): one norm over all seeds
    opt.step(); sched.step()
    acc += mse.detach(); cnt += 1
    if (step + 1) % 1024 == 0 or step + 1 == STEPS:
        row = dict(step=step + 1, sec=round(time.time() - t0, 1), nmse=[float('%.4g' % v) for v in (acc / cnt / VAR_Y).tolist()])
        log.append(row); print(json.dumps(row), flush=True); acc.zero_(); cnt = 0
rms1, mean1, std1 = preact_stats()
g3 = torch.Generator().manual_seed(4242); se = torch.zeros(M, dtype=torch.float64); Ne = 0
with torch.no_grad():
    for i in range(16):
        x = torch.randn(M, 16384, n, generator=g3)
        se += (forward(x) - x.sort(-1).values).double().pow(2).mean(2).sum(1); Ne += 16384
for k, s in enumerate(seeds):
    Wk = W[k].detach().double()
    if PIN:                                # fold the frozen per-unit factor into W: a standard residual net
        Wk = Wk * (math.sqrt(2.0) / run_rms[k, :, 0, :].double())[:, :, None]
    d = dict(W=Wk.numpy(), V=V[k].detach().double().numpy())
    if GLU:
        d['W2'] = W2[k].detach().double().numpy()
    if BIAS:
        d['b'] = Bb[k, :, 0].detach().double().numpy()
    np.savez(f'{OUT}_s{s}.npz', **d)
    meta = dict(act=NAME, p=P, L=L, n=n, m=m, seed=s, steps=STEPS, pin=PIN, prenorm=PRENORM, clip_per_seed=CLIP_PER_SEED, batch=B, lr=LR, var_y=VAR_Y,
                params=int(sum(p_[k].numel() for p_ in params)), nmse=float(se[k] / Ne / VAR_Y), n_eval=Ne,
                rms_init=rms0[k].tolist(), rms_final=rms1[k].tolist(), unit_mean=mean1[k].tolist(), unit_std=std1[k].tolist(),
                log=[dict(step=r['step'], nmse=r['nmse'][k]) for r in log], seconds=time.time() - t0)
    json.dump(meta, open(f'{OUT}_s{s}.json', 'w'))
    print('FINAL', NAME, P, L, s, f'nmse {meta["nmse"]:.4f}', flush=True)
print('ALL-DONE', round(time.time() - t0, 1), flush=True)
