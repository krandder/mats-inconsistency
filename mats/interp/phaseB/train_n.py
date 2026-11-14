"""World 1 with S sources (seventh agent; a tool for a later registered episode): train_sw.py's recipe (pfn2's model, loss, optimiser,
clipping, 1,000-step warmup, constant lr to CONST, linear cooldown over COOLDOWN; validation every 1,000 steps on 4,096 fixed episodes;
model.pt = best validation NLL, model_final.pt = last step; sparse checkpoints) on world_n.World(S), with pfn_n's variants and the
ctrl or mag format. With S = 3, ctrl format and no variant, the initialisation, the validation set and the episode stream equal
train_sw.py's ctrl run of the seed (the config name is pfn2's, with '_S<S>' appended for S != 3).
usage: python train_n.py --S 6 [--variant own|uni|nocons|split|splitnc] [--fmt ctrl|mag] --seed S --out DIR
       [--width 32 --heads 2 --depth 2 --mixk 4 --const 80000 --cooldown 10000 --ckpt sparse|none]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_n as PN  # noqa: E402
import world_n as WN  # noqa: E402

SPARSE = [2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]
DS = 16


def make_X(world, d, semb, qemb, fmt):
    o = d['o']
    if fmt == 'mag':
        o = o.copy(); o[:, :, :WN.QL] = d['y'][:, None, :WN.QL] + np.abs(o[:, :, :WN.QL] - d['y'][:, None, :WN.QL])
    return world.assemble(semb, qemb, o, d['y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--S', type=int, required=True); ap.add_argument('--variant', default=None); ap.add_argument('--fmt', default='ctrl', choices=['ctrl', 'mag'])
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2); ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--mixk', type=int, default=4); ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ckpt', default='sparse', choices=['sparse', 'none'])
    a = ap.parse_args()
    world = WN.World(a.S); TOTAL = a.const + a.cooldown
    cfg = argparse.Namespace(width=a.width, heads=a.heads, depth=a.depth, attn_only=False, ds=DS, mixk=a.mixk, phase1_fixed=0, l2_heads=0, l2_dh=0, init_from='', seed=a.seed)
    name = P2.cfg_name(cfg) + ('' if a.S == 3 else '_S%d' % a.S)
    meta = dict(width=a.width, heads=a.heads, depth=a.depth, attn_only=False, ds=DS, seed=a.seed, cfg=name, mixk=a.mixk, phase1_fixed=0, l2_heads=0,
                l2_dh=0, device=P2.DEV, S=a.S, variant=a.variant, fmt=a.fmt, const=a.const, cooldown=a.cooldown, recipe='train_n')
    torch.manual_seed(a.seed)
    model = PN.build_any(meta, P2, 2 * DS + 4)
    meta['params'] = P2.n_params(model)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    rng = np.random.default_rng([a.seed, cid, 7])
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    model.to(P2.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=P2.LR, weight_decay=P2.WD)
    rv = np.random.default_rng(P2.VAL_SEED)
    vd = world.sample_np(P2.N_VAL, rv)
    vs = rv.standard_normal((P2.N_VAL, a.S, DS)) / 4; vq = rv.standard_normal((P2.N_VAL, WN.Q, DS)) / 4
    VX = make_X(world, vd, vs, vq, a.fmt); vy = vd['y'][:, WN.QL]
    M, V, Wb = world.bayes(vd); vb = float(WN.mix_nll(M, V, Wb, vy).mean())
    CK = set(SPARSE) if a.ckpt == 'sparse' else set()
    save = lambda nm: torch.save(P2.cpu_sd(model), os.path.join(a.out, nm))
    if CK:
        save('ckpt/step_000000.pt')
    lr_at = lambda s: P2.LR * min(1.0, s / P2.WARMUP) if s <= a.const else P2.LR * max(0.0, (TOTAL - s) / a.cooldown)
    curve, best, best_step, t0, step = [], 1e9, -1, time.time(), 0
    while step < TOTAL:
        step += 1
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        d = world.sample_np(P2.BATCH, rng)
        semb = rng.standard_normal((P2.BATCH, a.S, DS)) / 4; qemb = rng.standard_normal((P2.BATCH, WN.Q, DS)) / 4
        X, m = make_X(world, d, semb, qemb, a.fmt)
        head = model(torch.from_numpy(X).to(P2.DEV), torch.from_numpy(m).to(P2.DEV))
        loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, WN.QL]).float().to(P2.DEV)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in CK:
            save('ckpt/step_%06d.pt' % step)
        if step % P2.EVAL_EVERY == 0:
            v = float(P2.head_nll_np(P2.predict(model, *VX), vy).mean()); curve.append((step, v))
            if v < best:
                best, best_step = v, step; save('model.pt')
            print('[%s %s %s s%d] step %d lr %.2e val %.4f gap %+.4f best %.4f@%d (%.0fs)' % (name, a.variant, a.fmt, a.seed, step, lr_at(step), v, v - vb,
                                                                                           best, best_step, time.time() - t0), flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, seconds=time.time() - t0, curve=curve, ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
