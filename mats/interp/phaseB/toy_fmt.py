"""E41 (PREREG-B amendment 49): E20's minimal 'ln' model (minimal.py: tokens -> linear -> LayerNorm -> single-query attention
pool -> readout MLP -> 3 noise levels) trained with an input-format schedule, to test whether the sign-format amplification and
lock-in (E30, E37) arise in single-source pooling alone.
Formats of the four labelled readings: ctrl o = y + r (minimal.py's); mag o = y + |r| (the error's sign removed; Bayes, which
depends on SS = sum r^2, is unchanged). Episodes, identity vectors and initialisation are minimal.py's for the same seed (the
sampler's draws do not depend on the format).
Schedule: AdamW (lr 1e-3, weight decay 0.01), batch 512, clipping 1; constant lr to CONST steps, then linear cooldown over COOL
steps; format F0 for steps 1..K, F1 after. Arms: ctrl, mag (24,000 + 6,000 steps); c2m2k, c2m12k (ctrl then mag after
K steps); mag18 (mag, 12,000 + 6,000 steps: c2m12k's budget after its switch); m2c12k (mag then ctrl after 12,000 steps).
Seeds 0-11.
Evaluation on 20,000 fresh episodes (seed 4141) in the final format, against the exact posterior over the three levels:
  kl         mean KL(Bayes || net);
  sign_pc    the KL reduction of the normalised geometric mean of the net's class probabilities over the base and 8 random
             sign patterns of the four errors (Bayes unchanged; ctrl-format nets only), and sign_kl the family's mean KL;
  neg_pc     the same over the base and the negation of all four errors.
usage: python toy_fmt.py ARM SEED -> results/e41/ARM_sSEED.json (+ .pt)
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

import minimal as M

torch.set_num_threads(1)
ARMS = {'ctrl': ('ctrl', 'ctrl', 0, 24000, 6000), 'mag': ('mag', 'mag', 0, 24000, 6000),
        'c2m2k': ('ctrl', 'mag', 2000, 24000, 6000), 'c2m12k': ('ctrl', 'mag', 12000, 24000, 6000),
        'mag18': ('mag', 'mag', 0, 12000, 6000), 'm2c12k': ('mag', 'ctrl', 12000, 24000, 6000)}


def tokens(y, r, sid, qid, fmt):
    return M.tokens(y, np.abs(r) if fmt == 'mag' else r, sid, qid)


def logp(net, y, r, sid, qid, fmt, bs=5000):
    out = []
    with torch.no_grad():
        for i in range(0, len(y), bs):
            sl = slice(i, i + bs)
            out.append(torch.log_softmax(net(tokens(y[sl], r[sl], sid[sl], qid[sl], fmt)), -1).numpy().astype(np.float64))
    return np.concatenate(out)


def evaluate(net, fmt):
    net.eval()
    k, y, r, sid, qid = M.sample(20000, np.random.default_rng(4141))
    lb = np.log(M.bayes_post(r) + 1e-300); pb = np.exp(lb)
    kl = lambda lq: (pb * (lb - lq)).sum(1)
    base = logp(net, y, r, sid, qid, fmt)
    out = dict(kl=float(kl(base).mean()))
    if fmt == 'ctrl':
        rr = np.random.default_rng(4142)
        fam = [base] + [logp(net, y, r * np.where(rr.random(r.shape) < 0.5, -1.0, 1.0), sid, qid, fmt) for _ in range(8)]
        for name, F in (('sign', fam), ('neg', [base, logp(net, y, -r, sid, qid, fmt)])):
            L = np.stack(F, 0); g = L.mean(0); lz = np.log(np.exp(g).sum(1))
            out[name + '_pc'] = float(np.mean(-lz)); out[name + '_kl'] = float(np.mean([kl(x).mean() for x in F]))
    return out


def main():
    arm, seed = sys.argv[1], int(sys.argv[2])
    f0, f1, K, CONST, COOL = ARMS[arm]; TOTAL = CONST + COOL
    torch.manual_seed(seed); rng = np.random.default_rng([seed, 2027]); net = M.Net('ln')
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    os.makedirs('results/e41', exist_ok=True); t0 = time.time(); curve = []
    for st in range(1, TOTAL + 1):
        lr = 1e-3 if st <= CONST else 1e-3 * max(0.0, (TOTAL - st) / COOL)
        for g in opt.param_groups:
            g['lr'] = lr
        k, y, r, s, q = M.sample(512, rng)
        fmt = f0 if (f0 != f1 and st <= K) else f1
        loss = nn.functional.cross_entropy(net(tokens(y, r, s, q, fmt)), torch.as_tensor(k))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if st in (K, 1000, 2000, 3000, 5000, 8000, 12000, 18000, 24000, 30000) and st > 0:
            curve.append((st, fmt, evaluate(net, fmt)['kl'])); net.train()
    out = dict(arm=arm, seed=seed, fmt0=f0, fmt1=f1, switch=K, const=CONST, cool=COOL, curve=curve, seconds=time.time() - t0)
    out.update(evaluate(net, f1))
    torch.save(net.state_dict(), 'results/e41/%s_s%d.pt' % (arm, seed))
    json.dump(out, open('results/e41/%s_s%d.json' % (arm, seed), 'w'))
    print(arm, seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if k != 'curve'}, flush=True)


if __name__ == '__main__':
    main()
