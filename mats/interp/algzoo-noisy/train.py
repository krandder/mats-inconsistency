"""Train noisy 2nd-argmax RNNs: fine-tune (FT) from ARC's weights, or from scratch (FS) with ARC's recipe.

usage:
  python train.py ft  D N SEED|ex SIGMA STEPS OUT     (FT: peak lr 0.1*0.04/sqrt(d), warmup 2^10, decay to 0.1x at half)
  python train.py fs  D N SEED SIGMA NTRAIN OUT       (FS: ARC's training.py defaults, n_train sequences)
Writes OUT.npz (weights, frozen r, sigma) and OUT.json (training log + final evaluation, paired against the original).
"""
import sys, time, math, json, functools
import numpy as np
import torch
import az

torch.set_num_threads(1)


def lr_mult(step, n_steps, warmup, decay_min):
    return min(step / warmup, 1) * max(decay_min ** (2 * step / n_steps), decay_min)


@torch.no_grad()
def paired_eval(model, W_orig, n_samples=2 ** 22, seed=4242, bs=2 ** 16):
    """Variant (noise on, one draw) vs original (noiseless) on the same inputs. Returns means, paired SEs."""
    orig = az.model_from(W_orig, sigma=0.0)
    g = torch.Generator().manual_seed(seed)
    s = dict(cv=0.0, co=0.0, dc=0.0, dc2=0.0, av=0.0, ao=0.0, da2=0.0, n=0)
    for i in range(0, n_samples, bs):
        x = torch.randn(bs, model.n, generator=g)
        y = az.target(x)
        lv = model(x, noise=True, gen=g).double(); lo = orig(x, noise=False).double()
        cv = torch.nn.functional.cross_entropy(lv, y, reduction='none'); co = torch.nn.functional.cross_entropy(lo, y, reduction='none')
        av = (lv.argmax(-1) == y).double(); ao = (lo.argmax(-1) == y).double()
        s['cv'] += cv.sum().item(); s['co'] += co.sum().item(); d = cv - co; s['dc'] += d.sum().item(); s['dc2'] += (d * d).sum().item()
        s['av'] += av.sum().item(); s['ao'] += ao.sum().item(); da = av - ao; s['da2'] += (da * da).sum().item(); s['n'] += bs
    n = s['n']
    L_var, L_orig, A_var, A_orig = s['cv'] / n, s['co'] / n, s['av'] / n, s['ao'] / n
    dl = s['dc'] / n; dl_se = math.sqrt(max(s['dc2'] / n - dl ** 2, 0) / n)
    da = A_var - A_orig; da_se = math.sqrt(max(s['da2'] / n - da ** 2, 0) / n)
    ok_loss = L_var <= 1.02 * L_orig
    ok_acc = A_var >= A_orig - 2 * da_se
    return dict(n=n, L_var=L_var, L_orig=L_orig, dL=dl, dL_se=dl_se, dL_ci95=[dl - 1.96 * dl_se, dl + 1.96 * dl_se],
                rel_gap=L_var / L_orig - 1, A_var=A_var, A_orig=A_orig, dA=da, dA_se=da_se,
                meets_loss=bool(ok_loss), meets_acc=bool(ok_acc), meets=bool(ok_loss and ok_acc))


@torch.no_grad()
def marginal_ce(model, n_samples=2 ** 18, K=16, seed=999, bs=2 ** 14):
    g = torch.Generator().manual_seed(seed); tot = 0.0; acc = 0.0; n = 0
    for i in range(0, n_samples, bs):
        x = torch.randn(bs, model.n, generator=g); y = az.target(x)
        P = torch.zeros(bs, model.n, dtype=torch.float64)
        for k in range(K):
            P += torch.softmax(model(x, noise=True, gen=g).double(), -1)
        P /= K
        tot += -torch.log(P.gather(1, y[:, None]).clamp_min(1e-300)).sum().item(); acc += (P.argmax(-1) == y).sum().item(); n += bs
    return dict(L_marg=tot / n, A_marg=acc / n, K=K, n=n)


def main():
    mode = sys.argv[1]; d = int(sys.argv[2]); n = int(sys.argv[3]); sd = sys.argv[4]; sigma = float(sys.argv[5])
    length = int(sys.argv[6]); out = sys.argv[7]
    t0 = time.time()
    example = (sd == 'ex'); seed = 0 if example else int(sd)
    W_orig = az.load_arc(d, n, seed=None if example else seed, example=example) if (mode == 'ft' or d * n > 0) else None
    bs = 1024
    if mode == 'ft':
        model = az.model_from(W_orig, sigma=sigma)
        n_steps = length; warmup = 2 ** 10; decay_min = 0.1; lr = 0.1 * 0.04 / math.sqrt(d); ramp = warmup
        if sigma > 0: model.calibrate_r(2 ** 18, iters=2)
    else:
        torch.manual_seed(seed)
        model = az.RNN(d, n, sigma=sigma, seed=seed)
        n_steps = length // bs; warmup = 2 ** 14; decay_min = 0.01; lr = 0.04 / math.sqrt(d); ramp = 0
        if sigma > 0: model.calibrate_r(2 ** 16, iters=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, functools.partial(lr_mult, n_steps=n_steps, warmup=warmup, decay_min=decay_min))
    g = torch.Generator().manual_seed(10_000 + seed * 7 + int(sigma * 1e4))
    gv = torch.Generator().manual_seed(55)
    xv = torch.randn(2 ** 16, n, generator=gv); yv = az.target(xv)
    log = []; run_loss = 0.0; run_acc = 0.0; k = 0
    target_sigma = sigma
    for step in range(n_steps):
        if ramp: model.sigma = target_sigma * min(1.0, (step + 1) / ramp)
        x = torch.randn(bs, n, generator=g); y = az.target(x)
        lo = model(x, noise=True, gen=g, update_r=True)
        loss = torch.nn.functional.cross_entropy(lo, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run_loss += loss.item(); run_acc += (lo.argmax(-1) == y).float().mean().item(); k += 1
        if (step + 1) % 4096 == 0 or step + 1 == n_steps:
            with torch.no_grad():
                lv = model(xv, noise=True, gen=torch.Generator().manual_seed(56))
                vl = torch.nn.functional.cross_entropy(lv, yv).item(); va = (lv.argmax(-1) == yv).float().mean().item()
            log.append(dict(step=step + 1, train_loss=run_loss / k, train_acc=run_acc / k, val_loss=vl, val_acc=va,
                            lr=opt.param_groups[0]['lr'], sec=time.time() - t0))
            print(json.dumps(log[-1]), flush=True); run_loss = run_acc = 0.0; k = 0
    model.sigma = target_sigma
    if target_sigma > 0: model.calibrate_r(2 ** 20, iters=3)
    ev = paired_eval(model, W_orig) if W_orig is not None else None
    mc = marginal_ce(model) if target_sigma > 0 else None
    meta = dict(mode=mode, d=d, n=n, seed=sd, sigma=target_sigma, length=length, n_steps=n_steps, peak_lr=lr,
                eval=ev, marginal=mc, log=log, seconds=time.time() - t0)
    az.save_model(model, out + '.npz', meta)
    print('FINAL', json.dumps(ev), json.dumps(mc), flush=True)


if __name__ == '__main__':
    main()
