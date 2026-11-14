"""train_gen.py -- one size-generic any-order network, trained on real 4x4 and 6x6 grids only, to be applied to 9x9
with no further training. Symbols live in 9 slots; every board draws a random injective assignment of its digits
to slots, and the network's output is restricted to the slots in play (the alphabet is part of the interface, the
rules are not). Usage: python -u train_gen.py NAME STEPS SEED"""
import sys, time
import numpy as np, torch
import core, data

name, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(1); torch.manual_seed(seed); KM = 9; T0 = time.time()
d6 = np.load("data/k6_base.npz"); s6 = data.GridStream(d6["B"][d6["base_split"] == 0], 6, seed)
G4 = torch.from_numpy(np.load("data/k4.npz")["G"].astype(np.int64))
gen = torch.Generator().manual_seed(seed)


def to_slots(x, K):
    """Random injective map digits -> 9 slots, per board. Returns slot-valued boards and the alphabet mask."""
    n = len(x); perm = torch.rand(n, KM, generator=gen).argsort(1)[:, :K]          # K distinct slots per board
    alpha = torch.zeros(n, KM, dtype=torch.bool); alpha.scatter_(1, perm, True)
    return torch.gather(perm, 1, x), alpha


def sampler(bs):
    if torch.rand(1, generator=gen).item() < 0.5:
        x = G4[torch.randint(0, len(G4), (bs,), generator=gen)]; K = 4
    else:
        x = s6.sample(bs); K = 6
    xs, alpha = to_slots(x, K)
    return xs, torch.zeros_like(xs, dtype=torch.bool), alpha


model = core.RelFormer(KM, d=64, L=4)
print(f"{name}: size-generic, 9 slots, params {core.n_params(model)}, steps {steps}", flush=True)
core.train_anyorder(model, 0, sampler, steps, bs=256, seed=seed, ckpt=f"ckpt/{name}.pt", log=lambda *a: print(*a, flush=True))
core.save_model(model, f"ckpt/{name}.pt", name=name, steps=steps, seed=seed)
print(f"done ({time.time()-T0:.0f}s)", flush=True)
