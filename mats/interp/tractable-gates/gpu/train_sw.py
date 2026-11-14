"""E37-E39 (PREREG-B amendments 45-47): the tiny-pfn recipe (pfn2.py model, loss, optimiser, clipping and data, unchanged) with
  - an input-format schedule: format F0 for training steps 1..K, format F1 for steps K+1.. (E37's switch arms). A pure arm has
    F0 = F1 (K is then irrelevant);
  - a fixed length: constant lr (after pfn2's 1,000-step warmup) to CONST steps, then a linear cooldown to 0 over COOLDOWN steps.
    Default 80,000 + 10,000 = 90,000, pfn2's extended length; the convergence rule is not used, so every arm trains equally long.
    For a 90,000-step pfn2 run the lr schedule is identical;
  - validation on pfn2's 4,096 validation episodes in both formats every 1,000 steps. The selected model (model.pt) is the best
    validation NLL in the final format F1 among evaluations after the switch (all evaluations for a pure arm, as in pfn2);
    model_final.pt is the last step;
  - checkpoints (state dicts) at the steps in ckpt_steps(K) (ckpt/step_XXXXXX.pt; step 0 = the initial weights);
  - --pair_init (E39): for a format with an extra input feature (sgnmag), every shared parameter is copied from the ctrl
    initialisation of the same seed, and the extra input column is drawn U(-1/6, 1/6) (the ctrl columns' bound, 1/sqrt(36)) from
    torch.Generator seeded 31337 + seed;
  - --stream_seed (default = seed): the seed of the training-episode stream, np.random.default_rng([stream_seed, cid, 7]) as in
    pfn2, with cid from the config name (format-independent). Identity draws are consumed in every format, so arms with the same
    stream seed see the same episodes; the initialisation is torch.manual_seed(seed) as in pfn2.
usage: python train_sw.py --f0 ctrl --f1 mag --switch 30000 --seed S --out DIR [--width 32 --heads 2 --depth 2 --mixk 4]
       [--const 80000 --cooldown 10000] [--pair_init] [--stream_seed S2] [--ckpt dense|sparse|none]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e30_fmt as F  # noqa: E402

DENSE = sorted(set([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000, 25000,
                    30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000]))
SPARSE = [2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]
AFTER = (250, 500, 1000, 2000, 5000)


def ckpt_steps(kind, K, total):
    if kind == 'none':
        return set()
    base = DENSE if kind == 'dense' else SPARSE
    s = set(x for x in base if x <= total)
    if kind == 'dense' and 0 < K < total:
        s |= {K} | {K + d for d in AFTER if K + d <= total}
    return s


def cfg_args(a):
    return argparse.Namespace(width=a.width, heads=a.heads, depth=a.depth, attn_only=False, ds=a.ds, mixk=a.mixk, phase1_fixed=0,
                              l2_heads=0, l2_dh=0, init_from='', seed=a.seed)


def build_model(a, fmt):
    meta = dict(width=a.width, heads=a.heads, depth=a.depth, attn_only=False, ds=a.ds, seed=a.seed, cfg=P2.cfg_name(cfg_args(a)),
                mixk=a.mixk, phase1_fixed=0, l2_heads=0, l2_dh=0, device=P2.DEV)
    torch.manual_seed(a.seed)
    if not a.pair_init:
        P2.fin_of = lambda ds: F.fin(fmt, ds)
        return P2.build(meta), meta
    assert F.fin(fmt, a.ds) == F.fin('ctrl', a.ds) + 1, 'pair_init is for formats with one extra input feature'
    P2.fin_of = lambda ds: F.fin('ctrl', ds)
    ref = P2.build(meta)                                   # the ctrl initialisation of this seed (same draws as the ctrl arm)
    P2.fin_of = lambda ds: F.fin(fmt, ds)
    model = P2.build(meta)
    sd = ref.state_dict(); new = model.state_dict()
    g = torch.Generator().manual_seed(31337 + a.seed)
    with torch.no_grad():
        for k, v in new.items():
            if k == 'inp.weight':
                fin0 = sd[k].shape[1]
                v[:, :fin0] = sd[k]
                v[:, fin0:] = (torch.rand(v.shape[0], v.shape[1] - fin0, generator=g) * 2 - 1) / math.sqrt(fin0)
            else:
                assert v.shape == sd[k].shape, k
                v.copy_(sd[k])
    model.load_state_dict(new)
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f0', required=True, choices=F.ARMS); ap.add_argument('--f1', required=True, choices=F.ARMS)
    ap.add_argument('--switch', type=int, default=0); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stream_seed', type=int, default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--mixk', type=int, default=4)
    ap.add_argument('--ds', type=int, default=W.DS)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--pair_init', action='store_true'); ap.add_argument('--ckpt', default='dense', choices=['dense', 'sparse', 'none'])
    a = ap.parse_args()
    assert F.fin(a.f0, a.ds) == F.fin(a.f1, a.ds), 'formats with different input widths cannot be switched'
    pure = a.f0 == a.f1
    K = 0 if pure else a.switch
    TOTAL = a.const + a.cooldown
    stream_seed = a.seed if a.stream_seed is None else a.stream_seed
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    model, meta = build_model(a, a.f1)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(meta['cfg']))
    rng = np.random.default_rng([stream_seed, cid, 7])                          # pfn2's training stream
    meta.update(params=P2.n_params(model), fmt=a.f1, fmt0=a.f0, fmt1=a.f1, switch=K, const=a.const, cooldown=a.cooldown,
                pair_init=a.pair_init, stream_seed=stream_seed, recipe='train_sw')
    model.to(P2.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=P2.LR, weight_decay=P2.WD)
    # validation episodes: pfn2.fixed_set(VAL_SEED, N_VAL) with train_fmt's token builder, in both formats
    rv = np.random.default_rng(P2.VAL_SEED)
    vd = W.sample_np(P2.N_VAL, rv)
    vs = rv.standard_normal((P2.N_VAL, W.S, a.ds)) / 4
    vq = rv.standard_normal((P2.N_VAL, W.Q, a.ds)) / 4
    VX = {f: F.make_X(vd['o'], vd['y'], vs, vq, f, a.ds) for f in {a.f0, a.f1}}
    vy = vd['y'][:, W.QL]
    _, vnll, _, _ = W.references(vd)
    vb = float(vnll['bayes'].mean())
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
        d = W.sample_np(P2.BATCH, rng)
        semb = rng.standard_normal((P2.BATCH, W.S, a.ds)) / 4                   # the same draws as world.tokens_np
        qemb = rng.standard_normal((P2.BATCH, W.Q, a.ds)) / 4
        f = a.f0 if step <= K else a.f1
        X, m = F.make_X(d['o'], d['y'], semb, qemb, f, a.ds)
        head = model(torch.from_numpy(X).to(P2.DEV), torch.from_numpy(m).to(P2.DEV))
        loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float().to(P2.DEV)).mean()
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
            print(f"[{meta['cfg']} {a.f0}->{a.f1}@{K} s{a.seed}] step {step} lr {lr_at(step):.2e} val[{f}] {v[f]:.4f} "
                  f"gap {v[f] - vb:+.4f} best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, seconds=time.time() - t0, curve=curve,
                ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
