import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import bansal as b, flow, nets
torch.set_num_threads(4)
x, y = b.load_raw('val'); B=16
xb = b.blocks(torch.from_numpy(x[:B])); ys = torch.from_numpy(y[:B, 3:-3:2, 3:-3:2].copy()) > 0.5
n = xb.shape[-1]; o = b.grid_open(B, n); S, G = b.ends(xb); tgt = b.node_to_edges(ys, o).float()
lam=1280.
# node costs, plain SGD
th = torch.ones(B, n, n, requires_grad=True)
for s in range(40):
    c = th.clamp_min(0.01); ec = flow.edge_cost_tensor(c, o); yh = nets.BBShortestPath.apply(ec, o, S, G, lam)
    loss = (yh * (1 - tgt) + (1 - yh) * tgt)[..., :2].sum() / B
    th.grad=None; loss.backward()
    g = th.grad
    onpath = ys; wrong = (b.edges_to_nodes(yh.bool(), S, G) & ~ys)
    if s%5==0: print(s, round(loss.item(),2), 'exact', (yh.bool()==tgt.bool()).flatten(1).all(1).float().mean().item(),
          'mean grad on wrong nodes', g[wrong].mean().item() if wrong.any() else None, 'on tgt-only nodes', g[ys & ~b.edges_to_nodes(yh.bool(),S,G)].mean().item() if (ys & ~b.edges_to_nodes(yh.bool(),S,G)).any() else None, flush=True)
    with torch.no_grad(): th -= 0.5*lam*g
