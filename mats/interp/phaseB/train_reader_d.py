"""E77 training (PREREG-B amendment, planned; ninth agent): a deeper fresh reader on a trained net's frozen input embedding and block 1.
The net is rebuilt with depth 1 + R (pfn2.PFN, the source's width, heads and output head); the embedding (inp.*) and block 1 (blocks.0.*)
are copied from the source's model.pt and frozen; the R reader blocks, the final LayerNorm and the output head are initialised with
PyTorch's defaults after torch.manual_seed(seed + 7600) (in module order), then trained with train_reader.py's fresh schedule (train_sw's:
1,000-step warmup, lr 1e-3 to --const, linear cooldown over --cooldown; AdamW, weight decay 0.01 on the trainable parameters, clipping
1.0) on train_reader.py's episode stream np.random.default_rng([seed + 7600, cid, 7]) with cid from the source's config name, so the
episodes equal E76's fresh run of the same source. With R = 1 this is E76's fresh arm except for the order of the initial draws.
Saves model.pt (best validation NLL), model_final.pt, meta.json (the source's meta with depth = 1 + R and reader fields) and sparse
checkpoints (step 0 included).
usage: python train_reader_d.py --src RUNDIR --rblocks R --seed S --out DIR [--const 80000 --cooldown 10000]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e48_fmt as F  # noqa: E402
import e30_eval as E30  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
SPARSE = [2000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--rblocks', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    a = ap.parse_args()
    src, arm, meta = E30.load(a.src)
    assert not meta.get('variant') and not meta.get('l2_heads') and not meta.get('b1_heads') and meta['depth'] == 2, 'standard depth-2 nets only'
    meta = dict(meta, depth=1 + a.rblocks)
    model = P2.build(meta)
    sd, ss = model.state_dict(), src.state_dict()
    frozen = [k for k in sd if k.startswith('inp.') or k.startswith('blocks.0.')]
    for k in frozen:
        assert sd[k].shape == ss[k].shape, k
        sd[k] = ss[k].clone()
    model.load_state_dict(sd)
    torch.manual_seed(a.seed + 7600)
    for mod in list(model.blocks[1:]) + [model.ln, model.out]:
        for m in mod.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    model.requires_grad_(True); model.train()
    for p in list(model.inp.parameters()) + list(model.blocks[0].parameters()):
        p.requires_grad_(False)
    train = [p for p in model.parameters() if p.requires_grad]
    TOTAL = a.const + a.cooldown
    opt = torch.optim.AdamW(train, lr=P2.LR, weight_decay=P2.WD)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(meta['cfg']))
    rng = np.random.default_rng([a.seed + 7600, cid, 7])
    rv = np.random.default_rng(P2.VAL_SEED); vd = W.sample_np(P2.N_VAL, rv)
    vs = rv.standard_normal((P2.N_VAL, W.S, 16)) / 4; vq = rv.standard_normal((P2.N_VAL, W.Q, 16)) / 4
    VX = F.make_X(vd['o'], vd['y'], vs, vq, arm); vy = vd['y'][:, W.QL]
    _, vnll, _, _ = W.references(vd); vb = float(vnll['bayes'].mean())
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    save = lambda nm: torch.save(P2.cpu_sd(model), os.path.join(a.out, nm))
    save('ckpt/step_000000.pt')
    lr_at = lambda s: P2.LR * s / P2.WARMUP if s <= P2.WARMUP else (P2.LR if s <= a.const else P2.LR * max(0.0, (TOTAL - s) / a.cooldown))
    v0 = float(P2.head_nll_np(P2.predict(model, *VX), vy).mean())
    curve, best, best_step, t0 = [(0, v0)], v0, 0, time.time()
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
            print('[reader_d R%d %s s%d] step %d lr %.2e val %.4f gap %+.4f best %.4f@%d (%.0fs)' % (a.rblocks, os.path.basename(a.src), a.seed, step,
                                                                                               lr_at(step), v, v - vb, best, best_step, time.time() - t0), flush=True)
    save('model_final.pt')
    meta = dict(meta, reader_mode='fresh_d', reader_blocks=a.rblocks, reader_src=a.src, reader_seed=a.seed, reader_lr=P2.LR, reader_const=a.const,
                reader_cooldown=a.cooldown, total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, val_start=v0,
                seconds=time.time() - t0, curve=curve, recipe='train_reader_d')
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
