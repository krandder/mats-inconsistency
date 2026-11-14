"""E48 world-2 training (PREREG-B amendment 56): train_t2.py's recipe unchanged (pfn2's set transformer, mixture loss, AdamW,
clipping, lr warmup; fixed 80,000 + 10,000-step schedule; validation on 4,096 fixed episodes, seed 10,002, every 1,000 steps;
model.pt = the best validation NLL; sparse or dense checkpoints), with t2lin.py's formats (ctrl, canon, prod, sgnprod) and its
paired initialisation (prod and sgnprod start from the ctrl initialisation of the same seed plus the new input columns). The
episode stream is np.random.default_rng([stream_seed, 2, 7]) and the sampler's draws do not depend on the format, so every
format of a seed sees E40's episodes. Pure formats only (no switches).
usage: python train_t2lin.py --fmt prod --D 2 --seed S --out DIR [--N 8 --width 32 --heads 2 --depth 2 --mixk 4
       --const 80000 --cooldown 10000 --ckpt sparse]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as L  # noqa: E402
import train_t2 as TT  # noqa: E402

VAL_SEED, N_VAL = TT.VAL_SEED, TT.N_VAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmt', required=True, choices=L.FMTS); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stream_seed', type=int, default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--D', type=int, default=1); ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--mixk', type=int, default=4)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ckpt', default='sparse', choices=['dense', 'sparse', 'none'])
    a = ap.parse_args()
    TOTAL = a.const + a.cooldown
    stream_seed = a.seed if a.stream_seed is None else a.stream_seed
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    meta = dict(task='t2', D=a.D, N=a.N, width=a.width, heads=a.heads, depth=a.depth, mixk=a.mixk, seed=a.seed,
                stream_seed=stream_seed, fmt=a.fmt, fmt0=a.fmt, fmt1=a.fmt, switch=0, const=a.const, cooldown=a.cooldown,
                fin=L.fin(a.fmt, a.D), pair_init=a.fmt not in T.FMTS, device=P2.DEV, recipe='train_t2lin')
    model = L.build(meta, P2); meta['params'] = P2.n_params(model)
    rng = np.random.default_rng([stream_seed, 2, 7])
    model.to(P2.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=P2.LR, weight_decay=P2.WD)
    vd = T.sample(N_VAL, np.random.default_rng(VAL_SEED), a.D, a.N)
    VX = L.tokens(vd, a.fmt)
    vy = vd['yq']
    vref, _ = T.references(vd); vb = float(vref['bayes'].mean())
    CK = TT.ckpt_steps(a.ckpt, 0, TOTAL)

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
        X, m = L.tokens(d, a.fmt)
        head = model(torch.from_numpy(X).to(P2.DEV), torch.from_numpy(m).to(P2.DEV))
        loss = P2.head_nll_t(head, torch.from_numpy(d['yq']).float().to(P2.DEV)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in CK:
            save('ckpt/step_%06d.pt' % step)
        if step % P2.EVAL_EVERY == 0:
            v = float(P2.head_nll_np(P2.predict(model, *VX), vy).mean())
            curve.append((step, v))
            if v < best:
                best, best_step = v, step
                save('model.pt')
            print(f"[t2lin D{a.D} N{a.N} w{a.width} L{a.depth} {a.fmt} s{a.seed}] step {step} lr {lr_at(step):.2e} "
                  f"val {v:.4f} gap {v - vb:+.4f} best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, val_blind=float(vref['blind'].mean()),
                seconds=time.time() - t0, curve=curve, ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
