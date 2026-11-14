"""P12 CPU phase (nice'd, 4 threads): BP variant / damping choice for the in-loop arbitrageur,
ARB-kappa on the training pool. Writes results/cpu_phase.json.

BP variant proxy (chosen before any net is trained, stated in RESULTS.md): a synthetic proposer
u = 3(2y - 1) + N(0, sigma^2) (y = label edges, sigma in {2, 3}), held fixed; run the arbitrageur with
persistent messages; score = exact rate of decode(theta + delta) at t = 30 (n = 9, pool mazes) and at
t = 5n (n = 25, val mazes). Variants: unary theta in {u (log-odds), log sigma(u) - 0.03, log sigma(u) - 0.3}
x {max-sum, sum-product tau = 1, tau = 0.3} x damping {0, 0.5, 0.8}.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze, flow
from evaluate import ARBK_GRID, run_flow_bp

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tens(d, sl=slice(None)):
    return (torch.tensor(d["open"][sl], device=DEV), torch.tensor(d["path"][sl], device=DEV),
            torch.tensor(d["S"][sl].astype(np.int64), device=DEV), torch.tensor(d["G"][sl].astype(np.int64), device=DEV),
            torch.tensor(d["cost"][sl], device=DEV), d["opt"][sl])


def proxy(d, T, sigma, tau, damp, mode, c, seed=0):
    o, pa, S, G, cost, _ = d
    g = torch.Generator().manual_seed(seed)
    B, n = o.shape[:2]
    y = pa[..., :2].float()
    uE = 3 * (2 * y[..., 0] - 1) + sigma * torch.randn(B, n, n, generator=g).to(DEV)
    uS = 3 * (2 * y[..., 1] - 1) + sigma * torch.randn(B, n, n, generator=g).to(DEV)
    u = flow.es_to_slots(uE, uS)
    endm = flow.endpoints(S, G, B, n, DEV)
    bp = flow.FlowBP(o, S, G, tau=tau, damp=damp); st = bp.init_state()
    th = flow.unary(u, mode, c)
    for _ in range(T):
        st = bp.step(st, th)
    b = th + bp.delta(st).clamp(-30, 30)
    on = flow.decode(b, o); onu = flow.decode(u, o)
    return {"exact": float(flow.exact(on, pa).float().mean()), "valid": float(flow.valid_path(on, endm).float().mean()),
            "u_exact": float(flow.exact(onu, pa).float().mean()), "finite": bool(torch.isfinite(b).all())}


def main():
    out = {}
    pool = maze.load(os.path.join(HERE, "data", "pool_W9.npz"))
    val = maze.load(os.path.join(HERE, "data", "val_W25.npz"))
    p9 = tens(pool, slice(0, 500)); v25 = tens(val)
    t0 = time.time(); rows = []
    for mode, c in (("logodds", 0.0), ("logprob", 0.03), ("logprob", 0.3)):
        for tau in (0.0, 1.0, 0.3):
            for damp in (0.0, 0.5, 0.8):
                r = {"theta": mode, "c": c, "tau": tau, "damp": damp}
                for sigma in (2.0, 3.0):
                    a = proxy(p9, 30, sigma, tau, damp, mode, c); b = proxy(v25, 125, sigma, tau, damp, mode, c)
                    r[f"n9_s{sigma}"] = a; r[f"n25_s{sigma}"] = b
                r["score"] = float(np.mean([r[k]["exact"] for k in r if k.startswith("n")]))
                rows.append(r); print(json.dumps(r), flush=True)
    best = max(rows, key=lambda r: r["score"])
    out["bp_variant"] = {"rows": rows, "chosen": {k: best[k] for k in ("theta", "c", "tau", "damp")},
                         "seconds": round(time.time() - t0, 1)}
    # ARB-kappa on the training pool (500 mazes) and val
    t0 = time.time(); ak = {}
    for name, d in (("pool500_W9", p9), ("val_W25", v25)):
        o, pa, S, G, cost, opt = d
        for tau, kap in ARBK_GRID:
            th = torch.where(o, torch.full(o.shape, -kap, device=DEV), torch.zeros(o.shape, device=DEV))
            on, it = run_flow_bp(th, o, S, G, tau, 0.5, 20 * o.shape[1], o.shape[1])
            endm = flow.endpoints(S, G, o.shape[0], o.shape[1], DEV)
            ak[f"{name}/{tau}_{kap}"] = {"exact": float(flow.exact(on, pa).float().mean()),
                                         "valid": float(flow.valid_path(on, endm).float().mean()), "iters": it}
            print(name, tau, kap, ak[f"{name}/{tau}_{kap}"], flush=True)
    sel = max(ARBK_GRID, key=lambda p: (ak[f"val_W25/{p[0]}_{p[1]}"]["exact"], ak[f"pool500_W9/{p[0]}_{p[1]}"]["exact"]))
    out["arb_kappa"] = {"grid": ak, "selected_on_val": list(sel), "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(HERE, "results", "cpu_phase.json"), "w"), indent=1)
    print("CPUPHASE chosen", out["bp_variant"]["chosen"], "arbk", sel)


if __name__ == "__main__":
    main()
