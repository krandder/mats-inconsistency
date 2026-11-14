"""Train several noisy 2nd-argmax RNNs of the same width at once (from scratch, ARC's recipe), on CPU or GPU.

usage: python train_batched.py D N "SEED:SIGMA,SEED:SIGMA,..." NTRAIN OUTPREFIX [DEVICE] [REFS]
  REFS: comma-separated reference originals for the paired evaluation, e.g. "16:10:ex,16:10:2" (default).
Recipe (alg_zoo/training.py defaults): AdamW lr 0.04/sqrt(D), weight decay 0.001, batch 1024, warmup 2^14 steps,
exponential decay to 0.01x at half the run, NTRAIN sequences. Each model has its own data stream, init and noise.
Noise model as in az.py: h <- ReLU(z) + sigma * r * eps, r = EMA of the batch RMS of ReLU(z) (detached) during
training; frozen by a fixed-point calibration (2^20 sequences) at the end. Checkpoints at 2^17..2^20 steps.
Writes OUTPREFIX_s{seed}_sig{sigma}.npz/.json (az.save_model format, eval against each reference).
"""
import sys, os, time, math, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import az

D = int(sys.argv[1]); N = int(sys.argv[2])
specs = [(int(a), float(b)) for a, b in (p.split(':') for p in sys.argv[3].split(','))]
NTRAIN = int(sys.argv[4]); OUT = sys.argv[5]
dev = torch.device(sys.argv[6] if len(sys.argv) > 6 else ('cuda' if torch.cuda.is_available() else 'cpu'))
REFS = (sys.argv[7] if len(sys.argv) > 7 else '16:10:ex,16:10:2').split(',')
torch.set_num_threads(1)
M = len(specs); B = 1024
steps = NTRAIN // B; warm = int(os.environ.get("WARM_STEPS", 2 ** 14)); dmin = 0.01; lr0 = 0.04 / math.sqrt(D)   # WARM_STEPS: Amendment 8 scales the warmup with the run length
k = 1 / math.sqrt(D)
Whi, Whh, Woh = [], [], []
for (seed, sg) in specs:
    g = torch.Generator().manual_seed(seed)
    Whi.append((torch.rand(D, generator=g) * 2 - 1) * k); Whh.append((torch.rand(D, D, generator=g) * 2 - 1) * k)
    Woh.append((torch.rand(N, D, generator=g) * 2 - 1) * k)
W_hi = torch.nn.Parameter(torch.stack(Whi).to(dev)); W_hh = torch.nn.Parameter(torch.stack(Whh).to(dev))
W_oh = torch.nn.Parameter(torch.stack(Woh).to(dev))
sig = torch.tensor([s for _, s in specs], device=dev)[:, None, None]
r = torch.ones(M, N, D, device=dev)
opt = torch.optim.AdamW([W_hi, W_hh, W_oh], lr=lr0, weight_decay=0.001)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warm, 1) * max(dmin ** (2 * s / steps), dmin))
gen = torch.Generator(device=dev).manual_seed(12345 + sum(s for s, _ in specs))


def forward(x, noise=True, update=False):
    h = torch.zeros(M, x.shape[1], D, device=dev)
    for t in range(N):
        a = torch.relu(torch.bmm(h, W_hh.transpose(1, 2)) + x[:, :, t:t + 1] * W_hi[:, None, :])
        if update:
            with torch.no_grad():
                r[:, t] = 0.99 * r[:, t] + 0.01 * a.pow(2).mean(1).sqrt()
        h = a + sig * r[:, t][:, None, :] * torch.randn(a.shape, generator=gen, device=dev) if noise else a
    return torch.bmm(h, W_oh.transpose(1, 2))


with torch.no_grad():                                   # initial r: a few EMA-free calibration passes
    for _ in range(3):
        x = torch.randn(M, 8192, N, generator=gen, device=dev); h = torch.zeros(M, 8192, D, device=dev); rr = []
        for t in range(N):
            a = torch.relu(torch.bmm(h, W_hh.transpose(1, 2)) + x[:, :, t:t + 1] * W_hi[:, None, :]); rr.append(a.pow(2).mean(1).sqrt())
            h = a + sig * r[:, t][:, None, :] * torch.randn(a.shape, generator=gen, device=dev)
        r.copy_(torch.stack(rr, 1))
t0 = time.time(); log = []; acc_l = torch.zeros(M, device=dev); acc_a = torch.zeros(M, device=dev); cnt = 0


def save(tag):
    for m, (seed, sg) in enumerate(specs):
        np.savez(f'{OUT}_s{seed}_sig{sg}{tag}.npz', W_hi=W_hi[m].detach().double().cpu().numpy(), W_hh=W_hh[m].detach().double().cpu().numpy(),
                 W_oh=W_oh[m].detach().double().cpu().numpy(), r=r[m].double().cpu().numpy(), sigma=sg)


for step in range(steps):
    x = torch.randn(M, B, N, generator=gen, device=dev)
    y = x.argsort(-1)[..., -2]
    lo = forward(x, True, True)
    ce = torch.nn.functional.cross_entropy(lo.reshape(-1, N), y.reshape(-1), reduction='none').reshape(M, B).mean(1)
    opt.zero_grad(); ce.sum().backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc_l += ce.detach(); acc_a += (lo.argmax(-1) == y).float().mean(1); cnt += 1
    if (step + 1) % 16384 == 0 or step + 1 == steps:
        row = dict(step=step + 1, sec=time.time() - t0, lr=opt.param_groups[0]['lr'], loss=(acc_l / cnt).tolist(), acc=(acc_a / cnt).tolist())
        log.append(row); print(json.dumps(row), flush=True); acc_l.zero_(); acc_a.zero_(); cnt = 0
    if (step + 1) in (2 ** 17, 2 ** 18, 2 ** 19):
        save(f'_ck{step + 1}')
save('')
# final: per model, fixed-point r, paired eval against each reference, marginal CE
res_all = []
for m, (seed, sg) in enumerate(specs):
    W = (W_hi[m].detach().double().cpu().numpy(), W_hh[m].detach().double().cpu().numpy(), W_oh[m].detach().double().cpu().numpy())
    mod = az.model_from(W, r=r[m].cpu().numpy(), sigma=sg).to(dev)
    mod.calibrate_r_dev = None
    with torch.no_grad():                               # fixed-point calibration on device
        g2 = torch.Generator(device=dev).manual_seed(999 + m)
        for _ in range(3):
            acc = torch.zeros(N, D, device=dev, dtype=torch.float64); tot = 0
            for i in range(16):
                x = torch.randn(65536, N, generator=g2, device=dev); h = torch.zeros(65536, D, device=dev)
                for t in range(N):
                    a = torch.relu(h @ mod.W_hh.T + x[:, t:t + 1] * mod.W_hi); acc[t] += a.double().pow(2).sum(0)
                    h = a + sg * mod.r[t] * torch.randn(a.shape, generator=g2, device=dev)
                tot += 65536
            mod.r.copy_((acc / tot).sqrt().float())
    evs = {}
    for ref in REFS:
        rd, rn, rs = ref.split(':')
        Wr = az.load_arc(int(rd), int(rn), seed=None if rs == 'ex' else int(rs), example=(rs == 'ex'))
        orig = az.model_from(Wr, sigma=0.0).to(dev)
        g3 = torch.Generator(device=dev).manual_seed(4242)
        s = dict(cv=0.0, co=0.0, dc=0.0, dc2=0.0, av=0.0, ao=0.0, da2=0.0, n=0)
        with torch.no_grad():
            for i in range(64):
                x = torch.randn(65536, N, generator=g3, device=dev); y = x.argsort(-1)[:, -2]
                lv = mod(x, noise=True, gen=g3).double(); lo_ = orig(x, noise=False).double()
                cv = torch.nn.functional.cross_entropy(lv, y, reduction='none'); co = torch.nn.functional.cross_entropy(lo_, y, reduction='none')
                av = (lv.argmax(-1) == y).double(); ao = (lo_.argmax(-1) == y).double()
                s['cv'] += cv.sum().item(); s['co'] += co.sum().item(); dd = cv - co; s['dc'] += dd.sum().item(); s['dc2'] += (dd * dd).sum().item()
                s['av'] += av.sum().item(); s['ao'] += ao.sum().item(); da = av - ao; s['da2'] += (da * da).sum().item(); s['n'] += 65536
        n_ = s['n']; Lv, Lo, Av, Ao = s['cv'] / n_, s['co'] / n_, s['av'] / n_, s['ao'] / n_
        dl = s['dc'] / n_; dls = math.sqrt(max(s['dc2'] / n_ - dl ** 2, 0) / n_); da = Av - Ao; das = math.sqrt(max(s['da2'] / n_ - da ** 2, 0) / n_)
        evs[ref] = dict(n=n_, L_var=Lv, L_orig=Lo, dL=dl, dL_se=dls, dL_ci95=[dl - 1.96 * dls, dl + 1.96 * dls], rel_gap=Lv / Lo - 1,
                        A_var=Av, A_orig=Ao, dA=da, dA_se=das, meets_loss=bool(Lv <= 1.02 * Lo), meets_acc=bool(Av >= Ao - 2 * das),
                        meets=bool(Lv <= 1.02 * Lo and Av >= Ao - 2 * das))
    with torch.no_grad():                               # marginal CE, 16 draws, 2^18 sequences
        g4 = torch.Generator(device=dev).manual_seed(77); tl = 0.0; ta = 0.0; nn_ = 0
        for i in range(16):
            x = torch.randn(16384, N, generator=g4, device=dev); y = x.argsort(-1)[:, -2]
            P = sum(torch.softmax(mod(x, noise=True, gen=g4).double(), -1) for _ in range(16)) / 16
            tl += -torch.log(P.gather(1, y[:, None]).clamp_min(1e-300)).sum().item(); ta += (P.argmax(-1) == y).sum().item(); nn_ += 16384
    meta = dict(mode='fs_batched', d=D, n=N, seed=seed, sigma=sg, length=NTRAIN, n_steps=steps, peak_lr=lr0, eval_refs=evs,
                eval=evs[REFS[0]], marginal=dict(L_marg=tl / nn_, A_marg=ta / nn_, K=16, n=nn_), log=[dict(step=l['step'], loss=l['loss'][m], acc=l['acc'][m]) for l in log],
                seconds=time.time() - t0, device=str(dev))
    mod = mod.cpu()
    az.save_model(mod, f'{OUT}_s{seed}_sig{sg}.npz', meta)
    res_all.append(meta); print('FINAL', seed, sg, json.dumps({k: (v['L_var'], v['L_orig'], v['A_var'], v['A_orig'], v['meets']) for k, v in evs.items()}), flush=True)
print('ALL-DONE', time.time() - t0, flush=True)
