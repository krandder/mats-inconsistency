"""Task 2 training (t2_world.py): pfn2.py's set transformer (model, mixture loss, AdamW, clipping, lr warmup; no positional
encoding, so the context examples are exchangeable) on in-context linear regression with an unknown noise level.
  - format schedule as train_sw.py: F0 for steps 1..K, F1 after (a pure arm has F0 = F1);
  - fixed length: constant lr to CONST steps, then a linear cooldown over COOLDOWN steps (default 80,000 + 10,000);
  - validation on 4,096 fixed episodes (seed 10,002) in both formats every 1,000 steps; model.pt = the best validation NLL in F1
    among evaluations after the switch; model_final.pt = the last step;
  - checkpoints at train_sw.py's dense or sparse schedule;
  - initialisation torch.manual_seed(seed); episode stream np.random.default_rng([stream_seed, 2, 7]); the sampler's draws do not
    depend on the format, so arms with the same seed see the same episodes.
usage: python train_t2.py --f0 ctrl --f1 canon --switch 30000 --seed S --out DIR [--D 1 --N 8 --width 32 --heads 2 --depth 2
       --mixk 4 --const 80000 --cooldown 10000 --ckpt sparse]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402

DENSE = sorted(set([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000, 25000,
                    30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000]))
SPARSE = [2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]
AFTER = (250, 500, 1000, 2000, 5000)
VAL_SEED, N_VAL = 10_002, 4096


def ckpt_steps(kind, K, total):
    if kind == 'none':
        return set()
    s = set(x for x in (DENSE if kind == 'dense' else SPARSE) if x <= total)
    if kind == 'dense' and 0 < K < total:
        s |= {K} | {K + d for d in AFTER if K + d <= total}
    return s


def build(meta):
    return P2.PFN(meta['D'] + 2, meta['depth'], meta['width'], meta['heads'], mixk=meta['mixk'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f0', required=True, choices=T.FMTS); ap.add_argument('--f1', required=True, choices=T.FMTS)
    ap.add_argument('--switch', type=int, default=0); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stream_seed', type=int, default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--D', type=int, default=1); ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--mixk', type=int, default=4)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ckpt', default='sparse', choices=['dense', 'sparse', 'none'])
    a = ap.parse_args()
    K = 0 if a.f0 == a.f1 else a.switch
    TOTAL = a.const + a.cooldown
    stream_seed = a.seed if a.stream_seed is None else a.stream_seed
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    meta = dict(task='t2', D=a.D, N=a.N, width=a.width, heads=a.heads, depth=a.depth, mixk=a.mixk, seed=a.seed,
                stream_seed=stream_seed, fmt=a.f1, fmt0=a.f0, fmt1=a.f1, switch=K, const=a.const, cooldown=a.cooldown,
                device=P2.DEV, recipe='train_t2')
    torch.manual_seed(a.seed)
    model = build(meta); meta['params'] = P2.n_params(model)
    rng = np.random.default_rng([stream_seed, 2, 7])
    model.to(P2.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=P2.LR, weight_decay=P2.WD)
    vd = T.sample(N_VAL, np.random.default_rng(VAL_SEED), a.D, a.N)
    VX = {f: T.tokens(vd, f) for f in {a.f0, a.f1}}
    vy = vd['yq']
    vref, _ = T.references(vd); vb = float(vref['bayes'].mean())
    CK = ckpt_steps(a.ckpt, K, TOTAL)

    def save(name):
        torch.save(P2.cpu_sd(model), os.path.join(a.out, name))

    if CK:
        save('ckpt/step_000000.pt')

    def lr_at(s):
        if s <= a.const:
            return P2.LR * min(1.0, s / P2.WARMUP)
        return P2.LR * max(0.0, (TOTAL - s) / a.cooldown)

    curve, best, best_step, t0, step = [], 1e9, -1, time.time(), 0
    while step < TOTAL:
        step += 1
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        d = T.sample(P2.BATCH, rng, a.D, a.N)
        f = a.f0 if step <= K else a.f1
        X, m = T.tokens(d, f)
        head = model(torch.from_numpy(X).to(P2.DEV), torch.from_numpy(m).to(P2.DEV))
        loss = P2.head_nll_t(head, torch.from_numpy(d['yq']).float().to(P2.DEV)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in CK:
            save('ckpt/step_%06d.pt' % step)
        if step % P2.EVAL_EVERY == 0:
            v = {fm: float(P2.head_nll_np(P2.predict(model, *VX[fm]), vy).mean()) for fm in VX}
            curve.append((step, v[a.f0], v[a.f1]))
            if step > K and v[a.f1] < best:
                best, best_step = v[a.f1], step
                save('model.pt')
            print(f"[t2 D{a.D} N{a.N} w{a.width} L{a.depth} {a.f0}->{a.f1}@{K} s{a.seed}] step {step} lr {lr_at(step):.2e} "
                  f"val[{f}] {v[f]:.4f} gap {v[f] - vb:+.4f} best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, val_blind=float(vref['blind'].mean()),
                seconds=time.time() - t0, curve=curve, ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
