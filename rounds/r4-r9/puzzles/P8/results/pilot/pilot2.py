"""P8 pilot: prequential code-length saving of a hidden variable Z on mazes.
saving(Z) = L(Y|X) - [L(Z|X) + L(Y|X,Z)], bits per test-stream, 2-layer 3x3 conv (receptive radius 2).
Reuses the P7 maze generator read-only."""
import sys, time, json, math
sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P7")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from maze import gen_mazes

torch.set_num_threads(3)
N_, H = 11, 8
BOUNDS = [16, 64, 256, 1024]


def data(num, seed):
    rng = np.random.default_rng(seed)
    w, s, d, _ = gen_mazes(num, N_, 0.3, rng)
    x = np.stack([w, s], 1).astype(np.float32)
    g = {k: (d <= k).astype(np.float32) for k in range(1, H + 1)}
    return x, g, rng


class Net(nn.Module):
    def __init__(self, cin, hid=16, layers=2):
        super().__init__()
        ls, c = [], cin
        for _ in range(layers):
            ls += [nn.Conv2d(c, hid, 3, padding=1), nn.ReLU()]
            c = hid
        ls += [nn.Conv2d(c, 1, 1)]
        self.f = nn.Sequential(*ls)

    def forward(self, x):
        return self.f(x).squeeze(1)


def pad_walls(x):
    # outside the maze counts as wall (so the edge is visible)
    return x


def preq(inp, tgt, seed, layers=2):
    """prequential codelen (bits) of tgt (B,n,n binary) given inp (B,C,n,n)."""
    torch.manual_seed(seed)
    inp = torch.tensor(inp); tgt = torch.tensor(tgt)
    net = Net(inp.shape[1], layers=layers)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    bits = BOUNDS[0] * tgt[0].numel() * 1.0  # first block uniform
    for i in range(len(BOUNDS) - 1):
        a, b = BOUNDS[i], BOUNDS[i + 1]
        # train on [0,a)
        steps = 250
        for _ in range(steps):
            idx = torch.randint(0, a, (32,))
            loss = F.binary_cross_entropy_with_logits(net(inp[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lg = net(inp[a:b])
            bits += float(F.binary_cross_entropy_with_logits(lg, tgt[a:b], reduction="sum")) / math.log(2)
    return bits


def main(seed=0):
    x, g, rng = data(BOUNDS[-1], seed)
    Y = g[H]
    rnd = (rng.random(Y.shape) < 0.5).astype(np.float32)
    openc = 1 - x[:, 0]
    t0 = time.time()
    out = {"LY_X": preq(x, Y, seed)}
    print("LY_X", round(out["LY_X"]), "uniform", BOUNDS[-1]*N_*N_, round(time.time()-t0), flush=True)
    cands = {"g2": g[2], "copyY": Y, "random": rnd, "open": openc, "g6": g[6]}
    for name, z in cands.items():
        LZ = preq(x, z, seed)
        LYZ = preq(np.concatenate([x, z[:, None]], 1), Y, seed)
        out[name] = {"LZ_X": LZ, "LY_XZ": LYZ, "saving": out["LY_X"] - LZ - LYZ}
        print(name, {k: round(v) for k, v in out[name].items()}, round(time.time() - t0), flush=True)
    # two-rung chain g2 -> g4 -> g6 -> Y (teacher-forced), each rung from X + previous rung
    chain = preq(x, g[2], seed)
    prev = g[2]
    for k in (4, 6):
        chain += preq(np.concatenate([x, prev[:, None]], 1), g[k], seed)
        prev = g[k]
    chain += preq(np.concatenate([x, prev[:, None]], 1), Y, seed)
    out["ladder_total"] = chain
    print("ladder", round(chain), round(time.time()-t0), flush=True)
    out["ladder_saving"] = out["LY_X"] - chain
    # end-to-end deeper net (8 conv layers, receptive radius 8) on Y alone
    out["LY_X_deep8"] = preq(x, Y, seed, layers=8)
    out["secs"] = round(time.time() - t0)
    print(json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in out.items()}, default=str))
    json.dump(out, open(f"/tmp/p8pilot/pilot_s{seed}.json", "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
