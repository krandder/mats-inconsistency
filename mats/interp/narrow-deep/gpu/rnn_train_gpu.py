"""Train narrow noisy 2nd-argmax RNNs (AlgZoo architecture) on long sequences, several models batched in one process.

Adapted from a copy of work/algzoo-noisy/train_batched.py (that lane's file is not edited).
Model (as az.py / ARC's DistRNN, bias=False):
  h_0 = 0;  a_t = ReLU(W_hh h_{t-1} + W_hi x_t);  h_t = a_t + sigma * r_t * eps_t  (noise only if sigma > 0)
  logits = W_oh h_T;  target = 2nd argmax of x (positions 0..T-1).  r_t = EMA of batch RMS of a_t during training.
Recipe: ARC's training.py defaults (AdamW, lr 0.04/sqrt(D), weight decay 0.001, batch 1024, exponential decay to
0.01x at half the run), warmup = min(2^14, steps/16) because our runs are shorter than ARC's 2^20 steps.

usage: python rnn_train.py D T "SEED:SIGMA,..." STEPS OUTPREFIX [DEVICE]
Writes OUTPREFIX_s{seed}_sig{sigma}.npz (W_hi, W_hh, W_oh, r, sigma) and .json (log, final eval).
"""
import sys, os, time, math, json
import numpy as np
import torch

D = int(sys.argv[1]); T = int(sys.argv[2])
specs = [(int(a), float(b)) for a, b in (p.split(':') for p in sys.argv[3].split(','))]
STEPS = int(sys.argv[4]); OUT = sys.argv[5]
dev = torch.device(sys.argv[6] if len(sys.argv) > 6 else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_num_threads(int(os.environ.get('NTHREADS', '1')))
M = len(specs); B = int(os.environ.get('BATCH', 1024))
warm = int(os.environ.get('WARM_STEPS', min(2 ** 14, max(STEPS // 16, 1)))); dmin = 0.01; lr0 = float(os.environ.get("LR_BASE", 0.04)) * math.sqrt(B / 1024) / math.sqrt(D)
LOG_EVERY = int(os.environ.get('LOG_EVERY', 1024))
k = 1 / math.sqrt(D)
Whi, Whh, Woh = [], [], []
for (seed, sg) in specs:
    g = torch.Generator().manual_seed(seed)
    Whi.append((torch.rand(D, generator=g) * 2 - 1) * k); Whh.append((torch.rand(D, D, generator=g) * 2 - 1) * k)
    Woh.append((torch.rand(T, D, generator=g) * 2 - 1) * k)
W_hi = torch.nn.Parameter(torch.stack(Whi).to(dev)); W_hh = torch.nn.Parameter(torch.stack(Whh).to(dev))
W_oh = torch.nn.Parameter(torch.stack(Woh).to(dev))
sig = torch.tensor([s for _, s in specs], device=dev)[:, None, None]
r = torch.ones(M, T, D, device=dev)
opt = torch.optim.AdamW([W_hi, W_hh, W_oh], lr=lr0, weight_decay=0.001)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warm, 1) * max(dmin ** (2 * s / STEPS), dmin))
gen = torch.Generator(device=dev).manual_seed(12345 + sum(s for s, _ in specs) + 1000 * T)


def forward(x, noise=True, update=False):
    h = torch.zeros(M, x.shape[1], D, device=dev)
    for t in range(T):
        a = torch.relu(torch.bmm(h, W_hh.transpose(1, 2)) + x[:, :, t:t + 1] * W_hi[:, None, :])
        if update:
            with torch.no_grad():
                r[:, t] = 0.99 * r[:, t] + 0.01 * a.pow(2).mean(1).sqrt()
        h = a + sig * r[:, t][:, None, :] * torch.randn(a.shape, generator=gen, device=dev) if noise else a
    return torch.bmm(h, W_oh.transpose(1, 2))


with torch.no_grad():
    for _ in range(3):
        x = torch.randn(M, 8192, T, generator=gen, device=dev); h = torch.zeros(M, 8192, D, device=dev); rr = []
        for t in range(T):
            a = torch.relu(torch.bmm(h, W_hh.transpose(1, 2)) + x[:, :, t:t + 1] * W_hi[:, None, :]); rr.append(a.pow(2).mean(1).sqrt())
            h = a + sig * r[:, t][:, None, :] * torch.randn(a.shape, generator=gen, device=dev)
        r.copy_(torch.stack(rr, 1))


def save(tag):
    for m, (seed, sg) in enumerate(specs):
        np.savez(f'{OUT}_s{seed}_sig{sg}{tag}.npz', W_hi=W_hi[m].detach().double().cpu().numpy(),
                 W_hh=W_hh[m].detach().double().cpu().numpy(), W_oh=W_oh[m].detach().double().cpu().numpy(),
                 r=r[m].double().cpu().numpy(), sigma=sg)


t0 = time.time(); log = []; acc_l = torch.zeros(M, device=dev); acc_a = torch.zeros(M, device=dev); cnt = 0
for step in range(STEPS):
    x = torch.randn(M, B, T, generator=gen, device=dev)
    y = x.argsort(-1)[..., -2]
    lo = forward(x, True, True)
    ce = torch.nn.functional.cross_entropy(lo.reshape(-1, T), y.reshape(-1), reduction='none').reshape(M, B).mean(1)
    opt.zero_grad(); ce.sum().backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc_l += ce.detach(); acc_a += (lo.argmax(-1) == y).float().mean(1); cnt += 1
    if (step + 1) % LOG_EVERY == 0 or step + 1 == STEPS:
        row = dict(step=step + 1, sec=round(time.time() - t0, 1), lr=opt.param_groups[0]['lr'],
                   loss=[round(v, 4) for v in (acc_l / cnt).tolist()], acc=[round(v, 4) for v in (acc_a / cnt).tolist()])
        log.append(row); print(json.dumps(row), flush=True); acc_l.zero_(); acc_a.zero_(); cnt = 0
    if (step + 1) % (STEPS // 4) == 0 and step + 1 < STEPS:
        save(f'_ck{step + 1}')
save('')
# final evaluation: fixed-point r (3 passes, 2^18 sequences), accuracy/CE with noise on and off on 2^18 fresh sequences
with torch.no_grad():
    g2 = torch.Generator(device=dev).manual_seed(999)
    for it in range(3):
        acc = torch.zeros(M, T, D, device=dev, dtype=torch.float64); tot = 0
        for i in range(16):
            x = torch.randn(M, 16384, T, generator=g2, device=dev); h = torch.zeros(M, 16384, D, device=dev)
            for t in range(T):
                a = torch.relu(torch.bmm(h, W_hh.transpose(1, 2)) + x[:, :, t:t + 1] * W_hi[:, None, :])
                acc[:, t] += a.double().pow(2).sum(1)
                h = a + sig * r[:, t][:, None, :] * torch.randn(a.shape, generator=g2, device=dev)
            tot += 16384
        r.copy_((acc / tot).sqrt().float())
    res = {m: dict(ce_on=0.0, acc_on=0.0, ce_off=0.0, acc_off=0.0) for m in range(M)}; n_ = 0
    g3 = torch.Generator(device=dev).manual_seed(4242)
    for i in range(16):
        x = torch.randn(M, 16384, T, generator=g3, device=dev); y = x.argsort(-1)[..., -2]
        for noise, tag in ((True, 'on'), (False, 'off')):
            lo = forward(x, noise, False)
            ce = torch.nn.functional.cross_entropy(lo.reshape(-1, T), y.reshape(-1), reduction='none').reshape(M, -1).sum(1)
            ac = (lo.argmax(-1) == y).double().sum(1)
            for m in range(M):
                res[m]['ce_' + tag] += ce[m].item(); res[m]['acc_' + tag] += ac[m].item()
        n_ += 16384
save('')
for m, (seed, sg) in enumerate(specs):
    ev = {kk: v / n_ for kk, v in res[m].items()}; ev['n'] = n_
    meta = dict(d=D, T=T, seed=seed, sigma=sg, steps=STEPS, batch=B, warmup=warm, peak_lr=lr0, eval=ev,
                log=[dict(step=l['step'], loss=l['loss'][m], acc=l['acc'][m]) for l in log], seconds=time.time() - t0, device=str(dev))
    json.dump(meta, open(f'{OUT}_s{seed}_sig{sg}.json', 'w'), indent=1)
    print('FINAL', seed, sg, json.dumps(ev), flush=True)
print('ALL-DONE', round(time.time() - t0, 1), flush=True)
