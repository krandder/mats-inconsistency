import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import bansal as b, flow, nets
torch.set_num_threads(1); torch.manual_seed(0)
x, y = b.load_raw('val'); xb = b.blocks(torch.from_numpy(x[:64])); ys = torch.from_numpy(y[:64, 3:-3:2, 3:-3:2].copy()) > 0.5
n = xb.shape[-1]; o = b.grid_open(64, n); S, G = b.ends(xb); tgt = b.node_to_edges(ys, o).float()
wall = xb.max(1).values < 0.5
for lam in (float(a) for a in sys.argv[1:]):
    th = torch.zeros(64, n, n, requires_grad=True); opt = torch.optim.Adam([th], lr=0.05); t0 = time.time()
    for s in range(300):
        c = F.softplus(th) + 0.01; ec = flow.edge_cost_tensor(c, o); yh = b.BBPos.apply(ec, o, S, G, lam)
        loss = (yh * (1 - tgt) + (1 - yh) * tgt)[..., :2].sum() / 64; opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 49:
            print(lam, s + 1, round(loss.item(), 2), 'exact', (yh.bool() == tgt.bool()).flatten(1).all(1).float().mean().item(),
                  'ratio', round((c[wall].median() / c[~wall].median()).item(), 2), 'gradnz', round((th.grad != 0).float().mean().item(), 3), round(time.time() - t0, 1), flush=True)
