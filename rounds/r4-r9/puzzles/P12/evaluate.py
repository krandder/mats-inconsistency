"""Evaluate one checkpoint (or a solver arm) on the P12 val/test sets.

python evaluate.py --ckpt runs/DTxARB_s0.pt [--sets val_W25,test_W9,...] [--device cuda]
python evaluate.py --solver arbk|a0flow|a0deg --sets ...          (non-learned arms; tau/kappa grids inside)

Loop arms run to 20n with snapshots at k*n (k in 2,5,10,20; selectable on val) plus 30, n, 3n, 7n, 15n.
Every decode is flow.decode (logit > 0 and open) of the price b_t (for arbitrageur arms b = u + sg(delta);
else b = u); for arbitrageur arms the proposer u_t is decoded too. Post-hoc arms on the DT checkpoint:
DT+dec (min-cost path on -log sigma(u) + c) and DT+ARB(test) (flow BP on unary(u) = u - c or
log sigma(u) - c at temperature tau,
cap 20n, early stop once the decode has not changed for max(50, n) iterations in the whole chunk) are
tuned per seed on val_W25 over (k, grid) and then run on the test sets only at the selected setting.
Output: results/eval/<name>.npz (per-maze arrays) + .json (summary, histograms, timings).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze, flow, nets
from train import ARMS

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (2, 5, 10, 20)
DEC_C = (0.01, 0.1, 0.3, 1.0, 3.0)
# post-hoc DT+ARB(test): (tau, theta mode, c): theta = u - c (logodds) or log sigma(u) - c (logprob)
ARBT_GRID = ((0.0, "logodds", 0.0), (0.0, "logodds", 2.0), (0.0, "logodds", 4.0), (1.0, "logodds", 0.0),
             (0.0, "logprob", 0.01), (0.0, "logprob", 0.1), (0.0, "logprob", 0.3), (0.0, "logprob", 1.0),
             (0.3, "logprob", 0.1), (1.0, "logprob", 0.1))
ARBK_GRID = ((0.0, 1.0),) + tuple((t, k) for t in (0.3, 1.0) for k in (0.1, 0.3, 1.0, 3.0))       # (tau, kappa)
HBINS = np.linspace(-9, 5, 141)   # log10 score bins for per-cell AUROC histograms
LIMIT = 0   # --limit N: first N mazes per set (smoke only)
DEFAULT_SETS = "val_W25,test_W9,test_W15,test_W25,test_W33,test_W49,test_W65,shift_p0.1_W33,shift_p0.5_W33"


def snaps(n):
    return sorted({k * n for k in KS} | {30, n, 3 * n, 7 * n, 15 * n})


def load_set(name, dev):
    d = maze.load(os.path.join(HERE, "data", name + ".npz"))
    t = {"open": torch.tensor(d["open"], device=dev), "path": torch.tensor(d["path"], device=dev),
         "code": torch.tensor(d["code"].astype(np.int64), device=dev), "S": torch.tensor(d["S"].astype(np.int64), device=dev),
         "G": torch.tensor(d["G"].astype(np.int64), device=dev), "cost": torch.tensor(d["cost"], device=dev)}
    if LIMIT:
        t = {k: v[:LIMIT] for k, v in t.items()}
        return t, d["L"][:LIMIT], d["opt"][:LIMIT]
    return t, d["L"], d["opt"]


def chunks(N, n, arb):
    cap = 200_000 if arb else 400_000
    b = max(1, min(N, cap // (n * n)))
    return [(i, min(N, i + b)) for i in range(0, N, b)]


def auroc_from_hist(hp, hn):
    """AUROC of score (higher = error) from positive/negative histograms over the same bins."""
    P, Nn = hp.sum(), hn.sum()
    if P == 0 or Nn == 0:
        return None
    cn = np.cumsum(hn) - hn   # negatives strictly below each bin
    return float(((hp * (cn + 0.5 * hn)).sum()) / (P * Nn))


def maze_metrics(on, path, endm, ec, opt):
    ex = flow.exact(on, path); vp = flow.valid_path(on, endm); sy, okc = flow.syndrome_ok(on, endm)
    f1 = flow.edge_f1(on, path)
    pc = flow.path_cost(on, ec).double()
    cr = torch.where(vp, pc / opt, torch.full_like(pc, float("nan")))
    return ex, vp, sy, f1, cr


class Acc:
    """Collects per-maze arrays per (set, key)."""

    def __init__(self):
        self.d = {}

    def put(self, key, lo, hi, N, arr, dtype=np.float32):
        if key not in self.d:
            self.d[key] = np.full(N, np.nan if np.issubdtype(dtype, np.floating) else -1, dtype)
        self.d[key][lo:hi] = arr


def run_loop_set(model, cfg, tset, L, opt, n, acc, pre, dev, bp_tau, bp_damp, hist, post=None, fw50=False,
                 bp_theta="logodds", bp_c=0.0, bp_kappa=5.0):
    """Run the recurrent arm on one set; fill acc with per-maze arrays under prefix pre."""
    N = len(L); arb = cfg.get("arb", "none") != "none"
    S_ = snaps(n); Tmax = max(S_); stab_w = 20
    for lo, hi in chunks(N, n, arb):
        o = tset["open"][lo:hi]; pa = tset["path"][lo:hi]; S = tset["S"][lo:hi]; G = tset["G"][lo:hi]
        code = tset["code"][lo:hi]; B = hi - lo
        ec = flow.edge_cost_tensor(tset["cost"][lo:hi], o)
        optt = torch.tensor(opt[lo:hi], device=dev)
        lp = nets.Loop(model, cfg, o, S, G, code, bp_tau=bp_tau, bp_damp=bp_damp, bp_theta=bp_theta, bp_c=bp_c, bp_kappa=bp_kappa)
        endm = lp.endm
        first = torch.full((B,), -1, device=dev); prev = None; same = torch.zeros(B, dtype=torch.long, device=dev)
        stab_t = torch.full((B,), -1, device=dev); stab_ex = torch.zeros(B, dtype=torch.bool, device=dev)
        syn_t = torch.full((B,), -1, device=dev); syn_ex = torch.zeros(B, dtype=torch.bool, device=dev)
        soft_t = torch.full((B,), -1, device=dev); soft_ex = torch.zeros(B, dtype=torch.bool, device=dev)
        st = lp.init()
        with torch.no_grad():
            for t in range(1, Tmax + 1):
                st, u, b = lp.step(st)
                on = flow.decode(b, o)
                ex = flow.exact(on, pa); vp = flow.valid_path(on, endm)
                sem = flow.semantic_loss(torch.sigmoid(b), o, endm)
                first = torch.where((first < 0) & ex, torch.full_like(first, t), first)
                if prev is not None:
                    unchanged = (on == prev).all(-1).all(-1).all(-1)
                    same = torch.where(unchanged, same + 1, torch.zeros_like(same))
                    hit = (stab_t < 0) & (same >= stab_w)
                    stab_t = torch.where(hit, torch.full_like(stab_t, t), stab_t); stab_ex = torch.where(hit, ex, stab_ex)
                prev = on
                hit = (syn_t < 0) & vp
                syn_t = torch.where(hit, torch.full_like(syn_t, t), syn_t); syn_ex = torch.where(hit, ex, syn_ex)
                hit = (soft_t < 0) & (sem.amax((1, 2)) < 0.1)
                soft_t = torch.where(hit, torch.full_like(soft_t, t), soft_t); soft_ex = torch.where(hit, ex, soft_ex)
                if t in S_:
                    exb, vpb, syb, f1b, crb = maze_metrics(on, pa, endm, ec, optt)
                    acc.put(f"{pre}/exact@{t}", lo, hi, N, exb.cpu().numpy(), np.int8)
                    acc.put(f"{pre}/valid@{t}", lo, hi, N, vpb.cpu().numpy(), np.int8)
                    acc.put(f"{pre}/synd@{t}", lo, hi, N, syb.cpu().numpy(), np.int8)
                    acc.put(f"{pre}/f1@{t}", lo, hi, N, f1b.cpu().numpy())
                    acc.put(f"{pre}/costratio@{t}", lo, hi, N, crb.cpu().numpy())
                    acc.put(f"{pre}/sem_mean@{t}", lo, hi, N, sem.mean((1, 2)).cpu().numpy())
                    acc.put(f"{pre}/sem_sum@{t}", lo, hi, N, sem.sum((1, 2)).cpu().numpy())
                    acc.put(f"{pre}/sem_max@{t}", lo, hi, N, sem.amax((1, 2)).cpu().numpy())
                    if arb:
                        onu = flow.decode(u, o)
                        acc.put(f"{pre}/u_exact@{t}", lo, hi, N, flow.exact(onu, pa).cpu().numpy(), np.int8)
                        acc.put(f"{pre}/u_valid@{t}", lo, hi, N, flow.valid_path(onu, endm).cpu().numpy(), np.int8)
                        acc.put(f"{pre}/u_sem_mean@{t}", lo, hi, N,
                                flow.semantic_loss(torch.sigmoid(u), o, endm).mean((1, 2)).cpu().numpy())
                    if t in [k * n for k in KS]:
                        # per-cell error-detection histograms split by maze valid / invalid
                        cellerr = (on != pa).any(-1)
                        sc = torch.log10(sem.clamp_min(1e-12)).cpu().numpy()
                        ce = cellerr.cpu().numpy(); vv = vpb.cpu().numpy()
                        for tag, mm in (("valid", vv), ("invalid", ~vv)):
                            key = f"{pre}/k{t // n}/{tag}"
                            h = hist.setdefault(key, {"pos": np.zeros(len(HBINS) + 1), "neg": np.zeros(len(HBINS) + 1)})
                            s_ = sc[mm]; e_ = ce[mm]
                            h["pos"] += np.bincount(np.digitize(s_[e_], HBINS), minlength=len(HBINS) + 1)
                            h["neg"] += np.bincount(np.digitize(s_[~e_], HBINS), minlength=len(HBINS) + 1)
                        if fw50 and lo < 50:
                            m = min(hi, 50) - lo
                            arbv = flow.true_arbitrage(torch.sigmoid(b[:m]), o[:m], endm[:m], iters=300)
                            acc.put(f"{pre}/fw_arb_mean@{t}", lo, lo + m, 50, arbv.mean((1, 2)).cpu().numpy())
                            acc.put(f"{pre}/fw_sem_mean@{t}", lo, lo + m, 50, sem[:m].mean((1, 2)).cpu().numpy())
                    if post is not None and t in post["ts"]:
                        post["fn"](t, u, o, pa, S, G, endm, ec, optt, lo, hi, N)
        for k, v in (("first_exact", first), ("stab_t", stab_t), ("stab_exact", stab_ex), ("synhalt_t", syn_t),
                     ("synhalt_exact", syn_ex), ("softhalt_t", soft_t), ("softhalt_exact", soft_ex)):
            acc.put(f"{pre}/{k}", lo, hi, N, v.cpu().numpy().astype(np.int32), np.int32)


def run_flow_bp(theta, o, S, G, tau, damp, cap, n):
    """Flow BP to cap iterations; early stop when the decode is unchanged for max(50, n) iterations."""
    bp = flow.FlowBP(o, S, G, tau=tau, damp=damp); st = bp.init_state(); prev = None; same = 0; w = max(50, n)
    for t in range(1, cap + 1):
        st = bp.step(st, theta)
        on = flow.decode(theta + bp.delta(st), o)
        if prev is not None and torch.equal(on, prev):
            same += 1
            if same >= w:
                break
        else:
            same = 0
        prev = on
    return on, t


def make_post(acc, pre_of, sel=None):
    """Post-hoc DT+dec and DT+ARB(test) on DT logits u at snapshot t. With sel=None: full grids (val);
    else only the selected (k, param) per arm."""
    info = {"iters": []}

    def fn(t, u, o, pa, S, G, endm, ec, optt, lo, hi, N):
        n = o.shape[1]; k = t // n
        p = torch.sigmoid(u).clamp(1e-6, 1 - 1e-6)
        grid_dec = DEC_C if sel is None else ([sel["DT+dec"]["c"]] if sel["DT+dec"]["k"] == k else [])
        for c in grid_dec:
            w = torch.where(o, -torch.log(p) + c, torch.zeros_like(p))
            on, ok = nets.shortest_path(w, o, S, G)
            exb, vpb, _, f1b, crb = maze_metrics(on, pa, endm, ec, optt)
            acc.put(f"{pre_of('DT+dec', k, c)}/exact", lo, hi, N, exb.cpu().numpy(), np.int8)
            acc.put(f"{pre_of('DT+dec', k, c)}/valid", lo, hi, N, vpb.cpu().numpy(), np.int8)
            acc.put(f"{pre_of('DT+dec', k, c)}/f1", lo, hi, N, f1b.cpu().numpy())
            acc.put(f"{pre_of('DT+dec', k, c)}/costratio", lo, hi, N, crb.cpu().numpy())
        grid_arb = ARBT_GRID if sel is None else ([tuple(sel["DT+ARB(test)"]["param"])] if sel["DT+ARB(test)"]["k"] == k else [])
        for tau, mode, cc in grid_arb:
            on, it = run_flow_bp(flow.unary(u, mode, cc), o, S, G, tau, 0.5, 20 * n, n)
            info["iters"].append(it)
            exb, vpb, _, f1b, crb = maze_metrics(on, pa, endm, ec, optt)
            key = pre_of("DT+ARB(test)", k, f"{tau}_{mode}_{cc}")
            acc.put(f"{key}/exact", lo, hi, N, exb.cpu().numpy(), np.int8)
            acc.put(f"{key}/valid", lo, hi, N, vpb.cpu().numpy(), np.int8)
            acc.put(f"{key}/f1", lo, hi, N, f1b.cpu().numpy())
            acc.put(f"{key}/costratio", lo, hi, N, crb.cpu().numpy())

    return fn, info


def eval_ckpt(a):
    dev = a.device
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    info = ck["info"]; cfg = info["cfg"]; name = os.path.basename(a.ckpt)[:-3]
    acc = Acc(); hist = {}; timing = {}
    sets = a.sets.split(",")
    if cfg.get("costnet"):
        model = nets.CostNet(info["w"]).to(dev); model.load_state_dict(ck["state"]); model.eval()
        for s in sets:
            t0 = time.time(); ts, L, opt = load_set(s, dev); n = ts["open"].shape[1]; N = len(L)
            for lo, hi in chunks(N, n, False):
                o = ts["open"][lo:hi]; S = ts["S"][lo:hi]; G = ts["G"][lo:hi]
                with torch.no_grad():
                    c = model(nets.make_input(o, S, G, ts["code"][lo:hi]))
                    on, ok = nets.shortest_path(nets.cell_to_edge_cost(c, o), o, S, G)
                endm = flow.endpoints(S, G, hi - lo, n, dev)
                exb, vpb, _, f1b, crb = maze_metrics(on, ts["path"][lo:hi], endm, flow.edge_cost_tensor(ts["cost"][lo:hi], o),
                                                     torch.tensor(opt[lo:hi], device=dev))
                acc.put(f"{s}/exact", lo, hi, N, exb.cpu().numpy(), np.int8); acc.put(f"{s}/valid", lo, hi, N, vpb.cpu().numpy(), np.int8)
                acc.put(f"{s}/f1", lo, hi, N, f1b.cpu().numpy()); acc.put(f"{s}/costratio", lo, hi, N, crb.cpu().numpy())
            timing[s] = round(time.time() - t0, 1)
        return name, info, acc, hist, timing, {}
    model = nets.DTNet(info["w"], fb=bool(cfg.get("fb")), extra=cfg.get("extra", 0)).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    is_dt = info["arm"] == "DT" and not a.no_post
    sel = {}
    # val first (post-hoc grids), then select per seed, then tests with the selected post-hoc settings
    order = [s for s in sets if s.startswith("val")] + [s for s in sets if not s.startswith("val")]
    for s in order:
        t0 = time.time(); ts, L, opt = load_set(s, dev); n = ts["open"].shape[1]
        post = None
        if is_dt:
            if s.startswith("val"):
                fn, pinfo = make_post(acc, lambda arm, k, prm, s=s: f"{s}/{arm}/k{k}/{prm}")
            else:
                fn, pinfo = make_post(acc, lambda arm, k, prm, s=s: f"{s}/{arm}", sel=sel)
            post = {"ts": {k * n for k in KS}, "fn": fn}
        run_loop_set(model, cfg, ts, L, opt, n, acc, s, dev, info["bp_tau"], info["bp_damp"], hist, post,
                     fw50=(s == "test_W33"), bp_theta=info.get("bp_theta", "logodds"), bp_c=info.get("bp_c", 0.0),
                     bp_kappa=info.get("bp_kappa", 5.0))
        if is_dt and s.startswith("val"):
            for arm, grid, fmt in (("DT+dec", DEC_C, lambda c: c), ("DT+ARB(test)", ARBT_GRID, lambda p: f"{p[0]}_{p[1]}_{p[2]}")):
                best = None
                for k in KS:
                    for prm in grid:
                        v = float(np.mean(acc.d[f"{s}/{arm}/k{k}/{fmt(prm)}/exact"]))
                        if best is None or v > best[0]:
                            best = (v, k, prm)
                sel[arm] = {"val_exact": best[0], "k": best[1], ("c" if arm == "DT+dec" else "param"): best[2]}
            timing["post_iters_val_mean"] = float(np.mean(pinfo["iters"])) if pinfo["iters"] else None
        timing[s] = round(time.time() - t0, 1)
    return name, info, acc, hist, timing, sel


def eval_solver(a):
    """Non-learned arms: A0-flow (theta = -true edge cost, max-sum; unit cost on U), A0-deg (undirected degree BP,
    U sets), ARB-kappa (theta = -kappa on every open edge, grid over (tau, kappa))."""
    dev = a.device; acc = Acc(); timing = {}
    for s in a.sets.split(","):
        t0 = time.time(); ts, L, opt = load_set(s, dev); n = ts["open"].shape[1]; N = len(L)
        for lo, hi in chunks(N, n, True):
            o = ts["open"][lo:hi]; S = ts["S"][lo:hi]; G = ts["G"][lo:hi]; pa = ts["path"][lo:hi]
            endm = flow.endpoints(S, G, hi - lo, n, dev); ec = flow.edge_cost_tensor(ts["cost"][lo:hi], o)
            optt = torch.tensor(opt[lo:hi], device=dev)
            if a.solver == "a0flow":
                grid = {"a0flow": (0.0, None)}
            elif a.solver == "arbk":
                grid = {f"{tau}_{kap}": (tau, kap) for tau, kap in ARBK_GRID}
            else:
                grid = {"a0deg": None}
            for key, prm in grid.items():
                if a.solver == "a0deg":
                    m = torch.zeros_like(ec); un = torch.where(o, -ec, torch.full_like(ec, -30.0))
                    for _ in range(20 * n):
                        m = flow.deg_bp_step(m, un, o, endm, damp=0.5)
                    on = flow.decode(flow.deg_beliefs(m, un, o), o); it = 20 * n
                else:
                    tau, kap = prm
                    th = -ec if kap is None else torch.where(o, torch.full_like(ec, -kap), torch.zeros_like(ec))
                    on, it = run_flow_bp(th, o, S, G, tau, 0.5, 20 * n, n)
                exb, vpb, syb, f1b, crb = maze_metrics(on, pa, endm, ec, optt)
                acc.put(f"{s}/{key}/exact", lo, hi, N, exb.cpu().numpy(), np.int8)
                acc.put(f"{s}/{key}/valid", lo, hi, N, vpb.cpu().numpy(), np.int8)
                acc.put(f"{s}/{key}/f1", lo, hi, N, f1b.cpu().numpy())
                acc.put(f"{s}/{key}/costratio", lo, hi, N, crb.cpu().numpy())
        timing[s] = round(time.time() - t0, 1)
    return a.solver, {"arm": a.solver}, acc, {}, timing, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--solver"); ap.add_argument("--sets", default=DEFAULT_SETS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_post", action="store_true"); ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global LIMIT
    LIMIT = a.limit
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    name, info, acc, hist, timing, sel = eval_solver(a) if a.solver else eval_ckpt(a)
    name = name + a.tag
    os.makedirs(os.path.join(HERE, "results", "eval"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "results", "eval", name + ".npz"), **acc.d)
    summ = {k: float(np.nanmean(v)) for k, v in acc.d.items() if v.dtype == np.int8 or "exact" in k}
    au = {k: auroc_from_hist(v["pos"], v["neg"]) for k, v in hist.items()}
    out = {"name": name, "info": info, "selected_posthoc": sel, "timing_s": timing, "total_s": round(time.time() - t0, 1),
           "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if a.device.startswith("cuda") else None,
           "cell_auroc": au, "means": summ}
    json.dump(out, open(os.path.join(HERE, "results", "eval", name + ".json"), "w"), indent=1, default=str)
    print("EVALDONE", name, json.dumps({k: v for k, v in out.items() if k in ("timing_s", "total_s", "peak_mem_MB", "selected_posthoc")}), flush=True)


if __name__ == "__main__":
    main()
