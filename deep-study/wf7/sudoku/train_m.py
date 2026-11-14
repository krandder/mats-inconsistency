"""train_m.py -- train an any-order network on REAL data only (masked any-order loss; no consistency term).
  variant A: solved grids only.             variant B: (puzzle, solution) pairs, clue cells flagged.
Usage: python -u train_m.py NAME VARIANT K D L STEPS BS SEED DATA [LABELS]
  DATA for A: 'k6:train' (uniform grids of the train orbits) | 'k4:orbit1' | 'k4:all'
  DATA for B: path of an npz with puz/sol (real solutions), or with LABELS = path of an npz whose 'answer' array
              replaces the solutions (used for the distilled proposal: labels come from the search, not from reality).
"""
import sys, time
import numpy as np, torch
import core, data

name, variant, K, D, L, steps, bs, seed, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), \
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
labels = sys.argv[10] if len(sys.argv) > 10 else None
import os, json
if os.path.exists(f"out/override_{name}.json"):          # lets a queued job take a changed step count (see addenda)
    steps = json.load(open(f"out/override_{name}.json"))["steps"]
torch.set_num_threads(1); torch.manual_seed(seed)
N = K * K; T0 = time.time()


def log(*a):
    print(*a, flush=True)


if variant == "A":
    if src == "k6:train":
        d = np.load("data/k6_base.npz"); stream = data.GridStream(d["B"][d["base_split"] == 0], 6, seed)
        val = data.GridStream(d["B"][d["base_split"] == 1], 6, seed + 1)
    else:
        d = np.load("data/k4.npz"); G = d["G"]; orb = d["orbit"]
        tr = G[orb == 1] if src == "k4:orbit1" else G
        stream = data.GridStream(tr, 4, seed); stream.sample = (lambda n, s=stream: torch.from_numpy(
            s.B[s.rng.integers(len(s.B), size=n)]))            # all 24 relabellings are already in G
        val = None
    sampler = lambda n: (stream.sample(n), torch.zeros(n, N, dtype=torch.bool))
else:
    d = np.load(src); puz, sol = d["puz"], d["sol"]
    if labels:
        lab = np.load(labels); keep = lab["solved"]; puz, sol = puz[keep], lab["answer"][keep]
        log(f"labels from the search: {keep.sum()} of {len(keep)} puzzles have an accepted answer")
    syms = np.load(f"data/k{K}_syms.npy")
    stream = data.PuzzleStream(puz, sol, K, seed, syms)
    sampler = stream.sample
    val = None

model = core.RelFormer(K, d=D, L=L)
log(f"{name}: variant {variant}, K {K}, d {D}, L {L}, params {core.n_params(model)}, steps {steps}, bs {bs}, data {src}")
core.train_anyorder(model, K, sampler, steps, bs=bs, seed=seed, ckpt=f"ckpt/{name}.pt", log=log)
core.save_model(model, f"ckpt/{name}.pt", name=name, variant=variant, K=K, steps=steps, bs=bs, seed=seed, data=src,
                labels=labels)
if val is not None:                                  # held-out masked-cell loss on real validation grids
    gen = torch.Generator().manual_seed(123); mask = core.relmask(K); tot = 0.0
    with torch.no_grad():
        for _ in range(8):
            x = val.sample(512); s, masked = core.anyorder_batch(x, torch.zeros_like(x, dtype=torch.bool), gen)
            lg = model(torch.where(s < 0, torch.full_like(s, K), s), torch.zeros_like(s), mask)
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, K), x.reshape(-1), reduction="none").view(x.shape)
            tot += ((ce * masked).sum(1) / masked.sum(1)).mean().item()
    log(f"validation (held-out orbits) masked-cell NLL {tot/8:.4f}; x {N} = {tot/8*N:.3f} nats "
        f"(any-order bound on the route surprise; reality's entropy is ln 28200960 = 17.155)")
log(f"done ({time.time()-T0:.0f}s)")
