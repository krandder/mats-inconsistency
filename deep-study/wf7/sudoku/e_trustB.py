"""e_trustB.py -- the trust policy for the puzzle+solution model M_B: paired routes from the clue state of REAL
training puzzles to their real solutions. Selection on real validation puzzles.
Usage: python -u e_trustB.py [MODEL=mB6]"""
import sys, time, json
import numpy as np, torch
import core, data

name = sys.argv[1] if len(sys.argv) > 1 else "mB6"
torch.set_num_threads(1); torch.manual_seed(0); K, N = 6, 36; T0 = time.time()
model, meta = core.load_model(f"ckpt/{name}.pt"); M = core.Net(model, K, name)
tr = np.load("data/k6_train_min.npz"); va = np.load("data/k6_val_min.npz"); syms = np.load("data/k6_syms.npy")
stream = data.PuzzleStream(tr["puz"], tr["sol"], K, 77, syms)


def log(*a):
    print(*a, flush=True)


def sample_real(n):
    x, clue = stream.sample(n)
    return x, torch.where(clue, x, torch.full_like(x, -1)), clue


VP = torch.from_numpy(va["puz"][:512]); VS = torch.from_numpy(va["sol"][:512]); VC = VP >= 0
log(f"== trust policy for {name} ({M.params} parameters)")
rules = {"lowest entropy": core.pol_entropy, "top probability": core.pol_maxprob, "top-1 minus top-2": core.pol_margin}
base = {}
for lab, p in rules.items():
    out = core.rollout(M, p, VP, X=VS, flag=VC); g = core.rollout(M, p, VP, flag=VC, greedy=True)
    base[lab] = dict(surprise=float(-out["logp"].mean()), acc=float((g["s"] == VS).all(1).float().mean()))
    log(f"   own confidence, {lab:18s}: validation route surprise of the real solution {base[lab]['surprise']:.4f}; one-pass accuracy {base[lab]['acc']:.4f}")
policies, hist = core.fit_trust(M, K, sample_real, core.pol_margin, iters=4, boards_per_iter=1024, n_dev=6, seed=4, log=log)
sel, acc = [], []
for p in policies:
    out = core.rollout(M, p, VP, X=VS, flag=VC); g = core.rollout(M, p, VP, flag=VC, greedy=True)
    sel.append(float(-out["logp"].mean())); acc.append(float((g["s"] == VS).all(1).float().mean()))
best = int(np.argmin(sel))
log(f"   validation route surprise of the iterates {[round(v, 4) for v in sel]}; one-pass accuracy {[round(a, 4) for a in acc]} -> pi_trust^B = iterate {best+1}")
torch.save(dict(cfg=policies[best].m.cfg, state=policies[best].m.state_dict()), f"ckpt/trust_{name}.pt")
conf_star = min(base, key=lambda l: base[l]["surprise"])
json.dump(dict(base=base, sel=sel, acc=acc, best=best, hist=hist, conf_star_by_surprise=conf_star,
               conf_star_by_acc=max(base, key=lambda l: base[l]["acc"])), open(f"out/e_trustB_{name}.json", "w"), indent=1)
log(f"done ({time.time()-T0:.0f}s)")
