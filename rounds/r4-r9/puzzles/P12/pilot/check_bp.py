import itertools, numpy as np, torch
from pilot import gen_maze, batch, bp_step, beliefs, local_incoherence
torch.set_num_threads(1)
def brute(oE, oS, S, G, un):
    n = oE.shape[0]; edges = [((i,j),(i,j+1)) for i in range(n) for j in range(n-1) if oE[i,j]] + [((i,j),(i+1,j)) for i in range(n-1) for j in range(n) if oS[i,j]]
    Z = 0.0; marg = np.zeros(len(edges))
    for bits in itertools.product([0,1], repeat=len(edges)):
        deg = {}
        for b, (u, v) in zip(bits, edges):
            if b: deg[u] = deg.get(u,0)+1; deg[v] = deg.get(v,0)+1
        ok = all((deg.get((i,j),0) == 1) if (i,j) in (S,G) else deg.get((i,j),0) in (0,2) for i in range(n) for j in range(n))
        if not ok: continue
        w = np.exp(sum(b*un for b in bits)); Z += w; marg += w*np.array(bits)
    return edges, marg/Z
rng = np.random.default_rng(3)
for p, un in [(0.0, 0.0), (0.5, 0.0), (0.5, -1.0)]:
    errs = []
    for _ in range(5):
        mz = gen_maze(4, p, rng); oE, oS, S, G = mz[:4]
        if oE.sum()+oS.sum() > 18: continue
        edges, mb = brute(oE, oS, S, G, un)
        op, en, tgt = batch([mz]); u = torch.full(op.shape, un); m = torch.zeros(op.shape)
        for _ in range(200): m = bp_step(m, u, op, en, damp=0.5)
        b = torch.sigmoid(beliefs(m, u, op))[0]
        dmap = {(0,1):0,(1,0):1}
        est = [b[a[0],a[1],dmap[(c[0]-a[0],c[1]-a[1])]].item() for a, c in edges]
        errs.append(np.abs(np.array(est)-mb).max())
    print("p", p, "unary", un, "max|BP-exact| per maze", np.round(errs, 4))
# incoherence formula vs brute force at one cell
p = torch.rand(1,1,1,4); o = torch.ones(1,1,1,4,dtype=torch.bool)
for end in (False, True):
    e = torch.tensor([[[end]]]); f = local_incoherence(p, o, e).item()
    tot = 0.0
    for bits in itertools.product([0,1], repeat=4):
        d = sum(bits); ok = (d == 1) if end else d in (0, 2)
        if ok: tot += np.prod([p[0,0,0,k].item() if b else 1-p[0,0,0,k].item() for k, b in enumerate(bits)])
    print("endpoint", end, "formula", round(f, 6), "brute", round(-np.log(tot), 6))
