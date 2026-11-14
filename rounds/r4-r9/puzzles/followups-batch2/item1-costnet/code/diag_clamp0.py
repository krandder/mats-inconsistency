# How often does the BB backward solve (perturbed costs, clamp_min 0) fail to return a terminated path in P12's setup?
import sys, torch
sys.path.insert(0, '.')
import train as T, nets, flow
torch.set_num_threads(4)
pool = T.load_pool("W9", "cpu"); g = torch.Generator().manual_seed(7)
stats = {"calls": 0, "notok": 0, "zero_edges": 0}
orig = nets.shortest_path
def wrapped(w, o, S, G, max_iter=None):
    on, ok = orig(w, o, S, G, max_iter)
    return on, ok
class BBc(nets.BBShortestPath):
    @staticmethod
    def backward(ctx, gy):
        ecost, open_, S, G, y = ctx.saved_tensors; lam = ctx.lam
        gsym = gy + flow.nb(gy)
        wp = (ecost.detach() + lam * gsym).clamp_min(0.0)
        yp, ok = orig(wp, open_, S, G)
        stats["calls"] += len(ok); stats["notok"] += int((~ok).sum())
        stats["zero_edges"] += int(((wp == 0) & open_).flatten(1).any(1).sum())
        return -(y - yp.float()) / lam, None, None, None, None
nets.BBShortestPath = BBc
for name in sys.argv[1:]:
    ck = torch.load(name, map_location="cpu", weights_only=False)
    m = nets.CostNet(32); m.load_state_dict(ck["state"])
    stats.update(calls=0, notok=0, zero_edges=0)
    for _ in range(10):
        op, pa, code, S, G = T.batch(pool, g, 64, "cpu")
        loss, y = nets.costnet_loss(m, op, S, G, code, pa, lam=20.0); loss.backward()
    print(name.split('/')[-3:], 'step', ck.get("step"), stats, flush=True)
m = nets.CostNet(32); torch.manual_seed(0); m = nets.CostNet(32)
stats.update(calls=0, notok=0, zero_edges=0)
for _ in range(10):
    op, pa, code, S, G = T.batch(pool, g, 64, "cpu")
    loss, y = nets.costnet_loss(m, op, S, G, code, pa, lam=20.0); loss.backward()
print('init', stats)
