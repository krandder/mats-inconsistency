"""Branching runs (to be registered as its own episode): train_sw.py's recipe and fixed schedule, one format, with the episode
stream switched at step K. Steps 1..K draw from np.random.default_rng([stream_seed, cid, 7]) (the parent run's stream), steps
K+1.. from np.random.default_rng([stream2, cid, 7]) (a new stream). K = 0 trains on the new stream from the start, i.e. the
parent's initialisation with a new data order. --init_seed sets the torch seed of the initialisation (default: --seed), so a run
with a new initialisation and the parent's stream is --init_seed S' --stream2 <unused> --branch 90000.
Validation, model selection (best validation NLL), checkpoints and outputs are train_sw.py's.
usage: python train_br.py --fmt ctrl --seed S --branch K --stream2 S2 [--init_seed S3] --out DIR [--width 32 --depth 2 ...]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmt', default='ctrl', choices=F.ARMS); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--branch', type=int, required=True); ap.add_argument('--stream2', type=int, required=True)
    ap.add_argument('--init_seed', type=int, default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--mixk', type=int, default=4)
    ap.add_argument('--ds', type=int, default=W.DS)
    ap.add_argument('--const', type=int, default=80_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ckpt', default='sparse', choices=['dense', 'sparse', 'none'])
    a = ap.parse_args()
    init_seed = a.seed if a.init_seed is None else a.init_seed
    K, TOTAL = a.branch, a.const + a.cooldown
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    b = argparse.Namespace(**vars(a)); b.seed = init_seed; b.pair_init = False
    model, meta = SW.build_model(b, a.fmt)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(meta['cfg']))
    rngA = np.random.default_rng([a.seed, cid, 7]); rngB = np.random.default_rng([a.stream2, cid, 7])
    meta.update(params=P2.n_params(model), fmt=a.fmt, fmt0=a.fmt, fmt1=a.fmt, switch=0, const=a.const, cooldown=a.cooldown,
                pair_init=False, stream_seed=a.seed, stream2=a.stream2, branch=K, init_seed=init_seed, seed=a.seed,
                recipe='train_br')
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
    CK = SW.ckpt_steps(a.ckpt, 0, TOTAL) | ({K} if 0 < K < TOTAL and a.ckpt != 'none' else set())

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
        rng = rngA if step <= K else rngB
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
            print(f"[{meta['cfg']} br@{K} s{a.seed}->{a.stream2} init{init_seed}] step {step} val {v:.4f} gap {v - vb:+.4f} "
                  f"best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)", flush=True)
    save('model_final.pt')
    meta.update(total_steps=TOTAL, best_val=best, best_step=best_step, val_bayes=vb, seconds=time.time() - t0, curve=curve,
                ckpt_steps=sorted(CK))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
