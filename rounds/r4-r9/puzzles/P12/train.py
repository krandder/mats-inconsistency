"""Train one P12 arm on the 9x9 W pool (or the U1 pool for the U1 anchor).

python train.py --arm DTxARB --seed 0 [--w 32] [--steps 12000] [--T 30] [--device cuda] [--smoke 50]
Batch order: for seed i, every arm draws batch indices, the per-group dihedral maps and the
progressive-loss (n, k) from the same torch.Generator(seed=1000+i), so seed i sees identical data in
every arm. Adam lr 1e-3 (200-step linear warmup, cosine to 10%), batch 64, grad clip 1.0, T=30,
progressive loss alpha=0.01 (Bansal et al.). Writes runs/<arm>_s<seed>.pt (state_dict + config + log).
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze, flow, nets

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "DT": {},
    "DT+fb": {"fb": 1, "extra": 8},
    "DT+pen": {"pen": 1.0},
    "DT-TD": {"td": 1},
    "DTxARB": {"fb": 1, "arb": "sg"},
    "DTxARB-noFB": {"arb": "sg"},
    "DTxARB-bp": {"fb": 1, "arb": "bp"},
    "U1-DT": {"pool": "U1_9"},
    "CostNet+BB": {"costnet": 1},
}


def t_dihedral(op, pa, code, S, G, k):
    """torch version of maze.dihedral for one map k on a batch."""
    n = op.shape[1]
    S = S.long().clone(); G = G.long().clone()
    if k & 4:
        op = op.transpose(1, 2)[..., [1, 0, 3, 2]]; pa = pa.transpose(1, 2)[..., [1, 0, 3, 2]]
        code = code.transpose(1, 2); S = S.flip(1); G = G.flip(1)
    if k & 2:
        op = op.flip(1)[..., [0, 3, 2, 1]]; pa = pa.flip(1)[..., [0, 3, 2, 1]]; code = code.flip(1)
        S = torch.stack([n - 1 - S[:, 0], S[:, 1]], 1); G = torch.stack([n - 1 - G[:, 0], G[:, 1]], 1)
    if k & 1:
        op = op.flip(2)[..., [2, 1, 0, 3]]; pa = pa.flip(2)[..., [2, 1, 0, 3]]; code = code.flip(2)
        S = torch.stack([S[:, 0], n - 1 - S[:, 1]], 1); G = torch.stack([G[:, 0], n - 1 - G[:, 1]], 1)
    return op.contiguous(), pa.contiguous(), code.contiguous(), S, G


def batch(pool, g, B, dev):
    idx = torch.randint(len(pool["L"]), (B,), generator=g)
    ks = torch.randint(8, (B // 8,), generator=g)
    parts = []
    for j in range(B // 8):
        sl = idx[j * 8:(j + 1) * 8].to(dev)
        parts.append(t_dihedral(pool["open"][sl], pool["path"][sl], pool["code"][sl], pool["S"][sl], pool["G"][sl], int(ks[j])))
    return [torch.cat([p[i] for p in parts]) for i in range(5)]


def load_pool(name, dev):
    d = maze.load(os.path.join(HERE, "data", f"pool_{name}.npz"))
    return {"open": torch.tensor(d["open"], device=dev), "path": torch.tensor(d["path"], device=dev),
            "code": torch.tensor(d["code"].astype(np.int64), device=dev), "S": torch.tensor(d["S"].astype(np.int64), device=dev),
            "G": torch.tensor(d["G"].astype(np.int64), device=dev), "L": d["L"]}


def _dup_running(out):
    """pid of another live train.py process with the same output path, else None."""
    mine, q = set(), os.getpid()   # self and ancestors (wrappers like timeout/nice carry the same cmdline)
    while q > 1 and q not in mine:
        mine.add(q)
        try:
            q = int(open(f"/proc/{q}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in mine:
            continue
        try:
            cl = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(x.endswith(b"train.py") for x in cl) and os.fsencode(out) in cl:
            return int(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=32); ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--T", type=int, default=30); ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bp_tau", type=float, default=0.0); ap.add_argument("--bp_damp", type=float, default=0.5)
    ap.add_argument("--bp_theta", default="logodds"); ap.add_argument("--bp_c", type=float, default=0.0)
    ap.add_argument("--bp_kappa", type=float, default=5.0)   # arbitrageur position limit (DTxARB arms)
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--stop_at", type=int, default=0)  # stop_at: resume test only
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = dict(ARMS[a.arm]); dev = a.device
    steps = a.smoke or a.steps
    out = a.out or os.path.join(HERE, "runs", f"{a.arm}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dup = _dup_running(out)
    if dup:   # a relaunch while the same run is still training (e.g. the gate retry loop) must not share its .part
        print(f"DUPLICATE: pid {dup} is already training {out}", flush=True); sys.exit(3)
    pool = load_pool(cfg.get("pool", "W9"), dev)
    torch.manual_seed(a.seed)
    if cfg.get("costnet"):
        model = nets.CostNet(a.w).to(dev)
    else:
        model = nets.DTNet(a.w, fb=bool(cfg.get("fb")), extra=cfg.get("extra", 0)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    warm = 200

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        q = (s - warm) / max(1, steps - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(1000 + a.seed)
    log = []; t0 = time.time(); run = {}; start = 0; prior_sec = 0.0
    part = out + ".part"   # resume point (spot preemption / VM age limit): written every 1000 steps
    if os.path.exists(part) and not a.smoke:
        ck = torch.load(part, map_location=dev, weights_only=False)
        ca = ck["args"]   # a checkpoint from before an argument existed matches if the run uses that argument's default
        if set(ca) <= set(vars(a)) and all(ca.get(k, ap.get_default(k)) == v for k, v in vars(a).items() if k != "stop_at"):
            model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            # map_location moves every tensor to dev; RNG states must be CPU ByteTensors (resume failed on CUDA)
            g.set_state(ck["g"].cpu()); torch.set_rng_state(ck["rng"].cpu())
            if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
            start = ck["step"]; log = ck["log"]; prior_sec = ck["sec"]
            print(json.dumps({"resumed_from_step": start}), flush=True)
    # Divergence rollback (added 2026-09-29 after DT+fb s1 w64 froze at loss ~8e18 with every update skipped):
    # at each 100-step row, a window with no skips and loss <= 2x the median of the last 10 row losses becomes the
    # in-memory snapshot; a window with >= 50% skipped updates restores the snapshot (model, Adam, LR schedule) and
    # replays from its step with fresh batches (the generator is not restored). At most MAX_RB rollbacks, then stop.
    # A run that never gets stuck is unchanged; .part checkpoints hold the snapshot.
    MAX_RB = 5
    snap_of = lambda st: {"state": copy.deepcopy(model.state_dict()), "opt": copy.deepcopy(opt.state_dict()),
                          "sched": copy.deepcopy(sched.state_dict()), "step": st, "log": list(log)}
    snap = snap_of(start); rollbacks = []; stopped_early = None
    s = start
    while s < steps:
        op, pa, code, S, G = batch(pool, g, a.B, dev)
        n_ng = int(torch.randint(0, a.T, (1,), generator=g)); k_gr = int(torch.randint(1, a.T - n_ng + 1, (1,), generator=g))
        if cfg.get("costnet"):
            loss, y = nets.costnet_loss(model, op, S, G, code, pa, lam=20.0)
            ex = {"exact": float(flow.exact(y.bool(), pa).float().mean())}
        else:
            lp = nets.Loop(model, cfg, op, S, G, code, bp_tau=a.bp_tau, bp_damp=a.bp_damp, bp_theta=a.bp_theta, bp_c=a.bp_c,
                           bp_kappa=a.bp_kappa)
            loss, ex = nets.train_loss(lp, pa, a.T, a.alpha, n_ng, k_gr)
        opt.zero_grad(set_to_none=True)
        skip = 0.0
        if torch.isfinite(loss):
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            else:
                skip = 1.0
        else:   # a non-finite loss or gradient skips the update (counted in the log as 'skipped')
            skip = 1.0; gn = torch.tensor(float("nan"))
        ex["skipped"] = skip
        sched.step()
        for k, v in {"loss": float(loss), "gn": float(gn), **ex}.items():
            run[k] = run.get(k, 0.0) + v
        if (s + 1) % 100 == 0 or s + 1 == steps:
            m = max(1, (s + 1) % 100 or 100)
            row = {"step": s + 1, "sec": round(prior_sec + time.time() - t0, 1), **{k: round(v / m, 5) for k, v in run.items()}}
            log.append(row); run = {}
            print(json.dumps(row), flush=True)
            prev = [r["loss"] for r in log[-11:-1] if "loss" in r and math.isfinite(r["loss"])]
            if row.get("skipped", 0) >= 0.5:
                if len(rollbacks) >= MAX_RB:
                    stopped_early = s + 1
                rollbacks.append({"at_step": s + 1, "to_step": snap["step"], "loss": row["loss"], "skipped": row["skipped"]})
                model.load_state_dict(snap["state"]); opt.load_state_dict(copy.deepcopy(snap["opt"]))
                sched.load_state_dict(snap["sched"]); log = list(snap["log"])
                print(json.dumps({"rollback": rollbacks[-1], "n": len(rollbacks)}), flush=True)
                if stopped_early:
                    print(json.dumps({"stopped_early": stopped_early, "model_step": snap["step"]}), flush=True); break
                s = snap["step"]; continue
            if row.get("skipped", 0) == 0 and math.isfinite(row["loss"]) and (not prev or row["loss"] <= 2 * float(np.median(prev))):
                snap = snap_of(s + 1)
        if (s + 1) % a.ckpt_every == 0 and s + 1 < steps and not a.smoke:
            torch.save({"args": vars(a), "state": snap["state"], "opt": snap["opt"], "sched": snap["sched"],
                        "g": g.get_state(), "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if dev.startswith("cuda") else None,
                        "step": snap["step"], "log": snap["log"], "sec": prior_sec + time.time() - t0}, part + ".tmp")
            os.replace(part + ".tmp", part)
            if a.stop_at and s + 1 >= a.stop_at:
                print("STOPPED_AT", s + 1, flush=True); return
        s += 1
    info = {"arm": a.arm, "seed": a.seed, "cfg": cfg, "w": a.w, "T": a.T, "steps": steps, "B": a.B, "lr": a.lr,
            "alpha": a.alpha, "bp_tau": a.bp_tau, "bp_damp": a.bp_damp, "bp_theta": a.bp_theta, "bp_c": a.bp_c, "bp_kappa": a.bp_kappa,
            "params": nets.n_params(model),
            "train_seconds": round(prior_sec + time.time() - t0, 1), "resumed_from_step": start,
            "rollbacks": rollbacks, "stopped_early": stopped_early,
            "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev.startswith("cuda") else None}
    torch.save({"state": model.state_dict(), "info": info, "log": log}, out)
    if os.path.exists(part):   # retire the resume point (*.tmp is not synced)
        os.replace(part, part + ".tmp")
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
