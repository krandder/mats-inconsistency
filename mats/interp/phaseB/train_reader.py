"""E76 training (PREREG-B amendment, planned): a trained net's input embedding and block 1 frozen, its reader (blocks 2.., the final
LayerNorm and the output head) retrained. train_sw.py's loss, optimiser (AdamW, lr 1e-3, weight decay 0.01 on the trainable parameters),
clipping, validation set and episode sampler, in the net's own format; episodes from a new stream, np.random.default_rng([seed + 7600,
cid, 7]), cid from the source net's config name.
  --mode fresh      the reader re-initialised (PyTorch defaults, torch.manual_seed(seed + 7600)), then train_sw's schedule: 1,000-step warmup,
                    constant lr to --const, linear cooldown over --cooldown (defaults 80,000 + 10,000);
  --mode continue   the trained reader kept, then a constant lr of --lr (default 1e-4) to --const and a linear cooldown over --cooldown.
Saves model.pt (best validation NLL), model_final.pt, meta.json (the source's meta plus reader fields) and sparse checkpoints.
usage: python train_reader.py --src RUNDIR --mode fresh|continue --seed S --out DIR [--const 80000 --cooldown 10000 --lr 1e-4]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e30_fmt as F0  # noqa: E402
import e48_fmt as F  # noqa: E402
import e30_eval as E30  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
SPARSE = [2000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--mode', required=True, choices=['fresh', 'continue'])
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    model, arm, meta = E30.load(a.src); model.requires_grad_(True); model.train()
    assert not meta.get('variant') and not meta.get('l2_heads'), 'standard nets only'
    for p in list(model.inp.parameters()) + list(model.blocks[0].parameters()):
        p.requires_grad_(False)
    if a.mode == 'fresh':
        torch.manual_seed(a.seed + 7600)
        for mod in list(model.blocks[1:]) + [model.ln, model.out]:
            for m in mod.modules():                                          # Linear and LayerNorm submodules (the module itself included)
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
    train = [p for p in model.parameters() if p.requires_grad]
    TOTAL = a.const + a.cooldown
    base_lr = P2.LR if a.mode == 'fresh' else a.lr
    opt = torch.optim.AdamW(train, lr=base_lr, weight_decay=P2.WD)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(meta['cfg']))
    rng = np.random.default_rng([a.seed + 7600, cid, 7])
    rv = np.random.default_rng(P2.VAL_SEED); vd = W.sample_np(P2.N_VAL, rv)
    vs = rv.standard_normal((P2.N_VAL, W.S, 16)) / 4; vq = rv.standard_normal((P2.N_VAL, W.Q, 16)) / 4
    VX = F.make_X(vd['o'], vd['y'], vs, vq, arm); vy = vd['y'][:, W.QL]
    _, vnll, _, _ = W.references(vd); vb = float(vnll['bayes'].mean())
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    save = lambda nm: torch.save(P2.cpu_sd(model), os.path.join(a.out, nm))
    save('ckpt/step_000000.pt')

    def lr_at(s):
        if a.mode == 'fresh' and s <= P2.WARMUP:
            return base_lr * s / P2.WARMUP
        return base_lr if s <= a.const else base_lr * max(0.0, (TOTAL - s) / a.cooldown)

    v0 = float(P2.head_nll_np(P2.predict(model, *VX), vy).mean())
    curve, best, best_step, t0 = [(0, v0)], v0, 0, time.time()
    if a.mode == 'continue':
        save('model.pt')
    for step in range(1, TOTAL + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        d = W.sample_np(P2.BATCH, rng)
        semb = rng.standard_normal((P2.BATCH, W.S, 16)) / 4; qemb = rng.standard_normal((P2.BATCH, W.Q, 16)) / 4
        X, m = F.make_X(d['o'], d['y'], semb, qemb, arm)
        head = model(torch.from_numpy(X), torch.from_numpy(m))
        loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float()).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(train, 1.0); opt.step()
        if step in SPARSE:
            save('ckpt/step_%06d.pt' % step)
        if step % P2.EVAL_EVERY == 0:
            v = float(P2.head_nll_np(P2.predict(model, *VX), vy).mean()); curve.append((step, v))
            if v < best:
                best, best_step = v, step; save('model.pt')
            print('[reader %s %s s%d] step %d lr %.2e val %.4f gap %+.4f best %.4f@%d (%.0fs)' % (a.mode, os.path.basename(a.src), a.seed, step, lr_at(step),
                                                                                             v, v - vb, best, best_step, time.time() - t0), flush=True)
    save('model_final.pt')
    meta = dict(meta, reader_mode=a.mode, reader_src=a.src, reader_seed=a.seed, reader_lr=base_lr, reader_const=a.const, reader_cooldown=a.cooldown,
                total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, val_start=v0, seconds=time.time() - t0, curve=curve, recipe='train_reader')
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
