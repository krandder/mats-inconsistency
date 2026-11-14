"""Train narrow-deep bias-free ReLU MLPs (residual or plain) to sort a Gaussian vector; several models batched.

Task: x ~ N(0, I_n) (n = 16), target y = sort(x) ascending (a positively homogeneous map, like the nets).
Residual ("res"):  h_0 = x;  u_l = W_l h_{l-1};  b_l = ReLU(u_l) + sigma r_l eps_l;  h_l = h_{l-1} + V_l b_l;  out = h_L.
  Init W_l ~ N(0, 2/n), V_l ~ N(0, 1/(n L)) (branch scale 1/sqrt(L)); all trainable; no biases.
Plain ("plain"):   h_0 = x;  a_l = ReLU(W_l h_{l-1}) + sigma r_l eps_l;  out = W_out h_L (needed: ReLU states are >= 0).
  Init W_l ~ N(0, 2/n) (He, critical), W_out ~ N(0, 1/n).
Noise: r_l = EMA of the batch RMS of the clean activation (detached), frozen by calibration at the end.
Loss: mean squared error over the n outputs; reported as nMSE = MSE / mean_i Var(y_i).
Optimizer: Adam, lr LR (default 2e-3), warmup steps/32, cosine decay to 0.01x; batch 1024; grad-norm clip 1.0.

usage: python mlp_train.py ARCH n L "SEED:SIGMA,..." STEPS OUTPREFIX [DEVICE]
"""
import sys, os, time, math, json
import numpy as np
import torch

ARCH = sys.argv[1]; n = int(sys.argv[2]); L = int(sys.argv[3])
specs = [(int(a), float(b)) for a, b in (p.split(':') for p in sys.argv[4].split(','))]
STEPS = int(sys.argv[5]); OUT = sys.argv[6]
dev = torch.device(sys.argv[7] if len(sys.argv) > 7 else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_num_threads(int(os.environ.get('NTHREADS', '1')))
M = len(specs); B = int(os.environ.get('BATCH', 1024)); LR = float(os.environ.get('LR', 2e-3))
LOG_EVERY = int(os.environ.get('LOG_EVERY', 1024)); warm = max(STEPS // 32, 1)
Ws, Vs, Wo = [], [], []
for (seed, sg) in specs:
    g = torch.Generator().manual_seed(1000 + seed)
    Ws.append(torch.randn(L, n, n, generator=g) * math.sqrt(2.0 / n))
    if ARCH == 'res':
        Vs.append(torch.randn(L, n, n, generator=g) * math.sqrt(1.0 / (n * L)))
    else:
        Wo.append(torch.randn(n, n, generator=g) * math.sqrt(1.0 / n))
W = torch.nn.Parameter(torch.stack(Ws).to(dev))                                  # (M, L, n, n)
params = [W]
if ARCH == 'res':
    V = torch.nn.Parameter(torch.stack(Vs).to(dev)); params.append(V)
else:
    WO = torch.nn.Parameter(torch.stack(Wo).to(dev)); params.append(WO)
sig = torch.tensor([s for _, s in specs], device=dev)[:, None, None]
r = torch.ones(M, L, n, device=dev)
opt = torch.optim.Adam(params, lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1)))))
gen = torch.Generator(device=dev).manual_seed(777 + sum(s for s, _ in specs) + 100 * L)


def forward(x, noise=True, update=False):
    h = x
    for l in range(L):
        u = torch.bmm(h, W[:, l].transpose(1, 2))
        a = torch.relu(u)
        if update:
            with torch.no_grad():
                r[:, l] = 0.99 * r[:, l] + 0.01 * a.pow(2).mean(1).sqrt()
        if noise:
            a = a + sig * r[:, l][:, None, :] * torch.randn(a.shape, generator=gen, device=dev)
        h = h + torch.bmm(a, V[:, l].transpose(1, 2)) if ARCH == 'res' else a
    return h if ARCH == 'res' else torch.bmm(h, WO.transpose(1, 2))


with torch.no_grad():
    for _ in range(3):
        x = torch.randn(M, 8192, n, generator=gen, device=dev); h = x; rr = []
        for l in range(L):
            a = torch.relu(torch.bmm(h, W[:, l].transpose(1, 2))); rr.append(a.pow(2).mean(1).sqrt().clamp_min(1e-30))
            a = a + sig * r[:, l][:, None, :] * torch.randn(a.shape, generator=gen, device=dev)
            h = h + torch.bmm(a, V[:, l].transpose(1, 2)) if ARCH == 'res' else a
        r.copy_(torch.stack(rr, 1))

VAR_Y = float(np.mean(np.var(np.sort(np.random.RandomState(0).standard_normal((1 << 20, n)), 1), 0)))


def save(tag):
    for m, (seed, sg) in enumerate(specs):
        d = dict(W=W[m].detach().double().cpu().numpy(), r=r[m].double().cpu().numpy(), sigma=sg, arch=ARCH)
        if ARCH == 'res':
            d['V'] = V[m].detach().double().cpu().numpy()
        else:
            d['W_out'] = WO[m].detach().double().cpu().numpy()
        np.savez(f'{OUT}_{ARCH}_L{L}_s{seed}_sig{sg}{tag}.npz', **d)


t0 = time.time(); log = []; acc_l = torch.zeros(M, device=dev); cnt = 0
for step in range(STEPS):
    x = torch.randn(M, B, n, generator=gen, device=dev)
    y = x.sort(-1).values
    out = forward(x, True, True)
    mse = (out - y).pow(2).mean((1, 2))
    opt.zero_grad(); mse.sum().backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    with torch.no_grad():
        acc_l += mse.detach(); cnt += 1
    if (step + 1) % LOG_EVERY == 0 or step + 1 == STEPS:
        row = dict(step=step + 1, sec=round(time.time() - t0, 1), lr=opt.param_groups[0]['lr'],
                   nmse=[float('%.4g' % v) for v in (acc_l / cnt / VAR_Y).tolist()])
        log.append(row); print(json.dumps(row), flush=True); acc_l.zero_(); cnt = 0
    if (step + 1) % (STEPS // 4) == 0 and step + 1 < STEPS:
        save(f'_ck{step + 1}')
# final: fixed-point r, evaluation with noise on/off on 2^18 fresh vectors
with torch.no_grad():
    g2 = torch.Generator(device=dev).manual_seed(999)
    for it in range(3):
        acc = torch.zeros(M, L, n, device=dev, dtype=torch.float64); tot = 0
        for i in range(16):
            x = torch.randn(M, 16384, n, generator=g2, device=dev); h = x
            for l in range(L):
                a = torch.relu(torch.bmm(h, W[:, l].transpose(1, 2))); acc[:, l] += a.double().pow(2).sum(1)
                a = a + sig * r[:, l][:, None, :] * torch.randn(a.shape, generator=g2, device=dev)
                h = h + torch.bmm(a, V[:, l].transpose(1, 2)) if ARCH == 'res' else a
            tot += 16384
        r.copy_((acc / tot).sqrt().clamp_min(1e-30).float())
    g3 = torch.Generator(device=dev).manual_seed(4242)
    res = {m: dict(mse_on=0.0, mse_off=0.0, maxerr_gt_0p1_on=0.0) for m in range(M)}; n_ = 0
    for i in range(16):
        x = torch.randn(M, 16384, n, generator=g3, device=dev); y = x.sort(-1).values
        for noise, tag in ((True, 'on'), (False, 'off')):
            out = forward(x, noise, False)
            e = (out - y)
            for m in range(M):
                res[m]['mse_' + tag] += e[m].pow(2).mean(1).sum().item()
                if noise:
                    res[m]['maxerr_gt_0p1_on'] += (e[m].abs().amax(1) > 0.1).double().sum().item()
        n_ += 16384
save('')
for m, (seed, sg) in enumerate(specs):
    ev = {k: v / n_ for k, v in res[m].items()}; ev['nmse_on'] = ev['mse_on'] / VAR_Y; ev['nmse_off'] = ev['mse_off'] / VAR_Y; ev['n'] = n_
    meta = dict(arch=ARCH, n=n, L=L, seed=seed, sigma=sg, steps=STEPS, batch=B, lr=LR, var_y=VAR_Y, eval=ev,
                log=[dict(step=l['step'], nmse=l['nmse'][m]) for l in log], seconds=time.time() - t0, device=str(dev))
    json.dump(meta, open(f'{OUT}_{ARCH}_L{L}_s{seed}_sig{sg}.json', 'w'), indent=1)
    print('FINAL', ARCH, L, seed, sg, json.dumps(ev), flush=True)
print('ALL-DONE', round(time.time() - t0, 1), flush=True)
