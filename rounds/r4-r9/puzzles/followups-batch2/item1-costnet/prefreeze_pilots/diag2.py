import sys, torch
sys.path.insert(0, '.')
import bansal as b, flow, nets
x, y = b.load_raw('val'); B=64
xb = b.blocks(torch.from_numpy(x[:B])); ys = torch.from_numpy(y[:B, 3:-3:2, 3:-3:2].copy()) > 0.5
n = xb.shape[-1]; o = b.grid_open(B, n); S, G = b.ends(xb); tgt = b.node_to_edges(ys, o).float()
for lo in (0.0, 1e-6, 1e-3, 0.01, 0.1):
    w = torch.where(o, torch.where(tgt>0, torch.full_like(tgt, lo), torch.full_like(tgt, 80.)), torch.zeros_like(tgt))
    on, ok = nets.shortest_path(w, o, S, G)
    print(lo, 'yp==tgt', (on.float()==tgt).flatten(1).all(1).float().mean().item(), 'ok', ok.float().mean().item())
