"""Hybrid initialisations (E44, PREREG-B amendment 52): train_sw.py's recipe and fixed schedule in one format, starting from the
recipient seed's initialisation with one parameter group copied from the donor seed's initialisation; the recipient seed's episode
stream. Groups (the LayerNorm parameters are identical at initialisation in every seed and stay the recipient's):
  inp     input embedding (inp.*);
  b1att   block 1 attention (blocks.0.qkv.*, blocks.0.o.*);   b1mlp  block 1 MLP (blocks.0.mlp.*);
  b2att   block 2 attention (blocks.1.qkv.*, blocks.1.o.*);   b2mlp  block 2 MLP (blocks.1.mlp.*);
  head    output head (out.*).
Validation, model selection (best validation NLL), checkpoints and outputs are train_sw.py's.
usage: python train_hy.py --seed RECIPIENT --donor DONOR --group G --out DIR [--fmt ctrl --ckpt sparse]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e30_fmt as F  # noqa: E402
import train_sw as SW  # noqa: E402

GROUPS = {'inp': ('inp.',), 'b1att': ('blocks.0.qkv.', 'blocks.0.o.'), 'b1mlp': ('blocks.0.mlp.',),
          'b2att': ('blocks.1.qkv.', 'blocks.1.o.'), 'b2mlp': ('blocks.1.mlp.',), 'head': ('out.',)}


def hybrid(a):
    r = argparse.Namespace(width=a.width, heads=a.heads, depth=a.depth, ds=a.ds, mixk=a.mixk, seed=a.seed, pair_init=False)
    model, meta = SW.build_model(r, a.fmt)
    d = argparse.Namespace(**vars(r)); d.seed = a.donor
    donor, _ = SW.build_model(d, a.fmt)
    sd, dsd = model.state_dict(), donor.state_dict()
    moved = []
    for k in sd:
        if k.startswith(GROUPS[a.group]):
            sd[k] = dsd[k].clone(); moved.append(k)
    assert moved, a.group
    model.load_state_dict(sd)
    return model, meta, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmt', default='ctrl', choices=F.ARMS); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--donor', type=int, required=True); ap.add_argument('--group', required=True, choices=sorted(GROUPS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--mixk', type=int, default=4)
    ap.add_argument('--ds', type=int, default=W.DS)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ckpt', default='sparse', choices=['dense', 'sparse', 'none'])
    a = ap.parse_args()
    TOTAL = a.const + a.cooldown
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    model, meta, moved = hybrid(a)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(meta['cfg']))
    rng = np.random.default_rng([a.seed, cid, 7])                       # the recipient's stream
    meta.update(params=P2.n_params(model), fmt=a.fmt, fmt0=a.fmt, fmt1=a.fmt, switch=0, const=a.const, cooldown=a.cooldown,
                pair_init=False, stream_seed=a.seed, donor=a.donor, group=a.group, moved=moved, recipe='train_hy')
    model.to(P2.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=P2.LR, weight_decay=P2.WD)
    rv = np.random.default_rng(P2.VAL_SEED)
    vd = W.sample_np(P2.N_VAL, rv)
    vs = rv.standard_normal((P2.N_VAL, W.S, a.ds)) / 4
    vq = rv.standard_normal((P2.N_VAL, W.Q, a.ds)) / 4
    vX = F.make_X(vd['o'], vd['y'], vs, vq, a.fmt, a.ds)
    vy = vd['y'][:, W.QL]
    _, vnll, _, _ = W.references(vd)
    vb = float(vnll['bayes'].mean())
    CK = SW.ckpt_steps(a.ckpt, 0, TOTAL)

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
        d = W.sample_np(P2.BATCH, rng)
        semb = rng.standard_normal((P2.BATCH, W.S, a.ds)) / 4
        qemb = rng.standard_normal((P2.BATCH, W.Q, a.ds)) / 4
        X, m = F.make_X(d['o'], d['y'], semb, qemb, a.fmt, a.ds)
        head = model(torch.from_numpy(X).to(P2.DEV), torch.from_numpy(m).to(P2.DEV))
        loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float().to(P2.DEV)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in CK:
            save('ckpt/step_%06d.pt' % step)
        if step % P2.EVAL_EVERY == 0:
            v = float(P2.head_nll_np(P2.predict(model, *vX), vy).mean())
            curve.append((step, v, v))
            if v < best:
                best, best_step = v, step
                save('model.pt')
            print(f"[{meta['cfg']} hybrid s{a.seed}<-{a.donor}:{a.group}] step {step} val {v:.4f} gap {v - vb:+.4f} "
                  f"best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, seconds=time.time() - t0, curve=curve,
                ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
