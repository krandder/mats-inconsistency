"""Matched-depth reference codelens for P8 (critic fix #1). Same data/seed/code as pilot2."""
import sys, json, time, math
job = sys.argv[1]
import numpy as np, torch, torch.nn.functional as F
import pilot2 as P
torch.set_num_threads(1)
layers, steps = {"deep4": (4, 250), "deep8": (8, 250), "deep8_1k": (8, 1000), "shallow_1k": (2, 1000)}[job]
x, g, rng = P.data(P.BOUNDS[-1], 0)
Y = g[8]
def preq(inp, tgt, seed, layers, steps):
    torch.manual_seed(seed)
    inp = torch.tensor(inp); tgt = torch.tensor(tgt)
    net = P.Net(inp.shape[1], layers=layers)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    bits = P.BOUNDS[0] * tgt[0].numel() * 1.0
    for i in range(len(P.BOUNDS) - 1):
        a, b = P.BOUNDS[i], P.BOUNDS[i + 1]
        for _ in range(steps):
            idx = torch.randint(0, a, (32,))
            loss = F.binary_cross_entropy_with_logits(net(inp[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            bits += float(F.binary_cross_entropy_with_logits(net(inp[a:b]), tgt[a:b], reduction="sum")) / math.log(2)
    return bits
t0 = time.time()
b = preq(x, Y, 0, layers, steps)
json.dump({"job": job, "bits": b, "secs": round(time.time() - t0)}, open(f"/tmp/p8pilot/ref_{job}.json", "w"))
print(job, round(b), round(time.time() - t0), flush=True)
