"""P1 pipelines (resume-safe with marker files in the output directory).

  python eval.py t1        --out results/t1           (CPU smoke test, pilot_Bds.pt)
  python eval.py t2m       --bb results/vm/t2m_bb_s0.pt --bseed 0 --out results/t2m
  python eval.py maze-repro | maze-heads | maze-size --size 59 | ...  --out results/maze
Each pool is written as obs_<pool>.npz (what rules may read) and eval_<pool>.npz
(labels-derived arrays; only scoring and the oracle read them).
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import (t2m_batch, MAZE_SPLITS, t_cap_maze, LabelledMazes, UnlabelledMazes,  # noqa: E402
                  write_generated, load_maze_arrays, split_indices)
from heads import ARMS_T2M, ARMS_MAZE, TT_ARMS, decision_times, head_macs_per_pos  # noqa: E402
from models import T2mNet, load_pilot, load_dtrecall, backbone_macs_per_pos  # noqa: E402
from traj import rollout, eval_arrays, first_cons  # noqa: E402
from train_heads import ArmSet, capacity_stats, capacity_report  # noqa: E402

NC = 200


def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def gpu_frac(device):
    """Cap this process's share of GPU memory (P1_GPU_FRAC): several jobs share one L4, and cuDNN
    otherwise picks conv algorithms with >20 GB workspaces (seen on odd lengths in PonderNet fits)."""
    f = float(os.environ.get("P1_GPU_FRAC", "0"))
    if f > 0 and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(f)


def c_grid(c_range):
    return np.geomspace(c_range[0], c_range[1], NC)


def marker(out, name):
    return os.path.exists(os.path.join(out, name + ".done"))


def mark(out, name, info=None):
    with open(os.path.join(out, name + ".done"), "w") as f:
        json.dump(info or {}, f)


# ---------------------------------------------------------------- partial checkpoints (spot preemption)
def armset_ckpt(A):
    return {"heads": {n: h.state_dict() for n, h in A.heads.items()},
            "opts": {n: o.state_dict() for n, o in A.opts.items()},
            "tgt": {n: t.state_dict() for n, t in A.tgt.items()},
            "norm": dict(A.norm), "gen": A.gen.get_state(), "nstep": A.nstep}


def armset_restore(A, S, device):
    for n, sd in S["heads"].items():
        A.heads[n].load_state_dict(sd)
    for n, sd in S["opts"].items():
        A.opts[n].load_state_dict(sd)
    for n, sd in S["tgt"].items():
        A.tgt[n].load_state_dict(sd)
    for n, v in S["norm"].items():
        A.norm[n] = None if v is None else (tuple(x.to(device) for x in v) if isinstance(v, tuple) else v.to(device))
    A.gen.set_state(S["gen"])
    A.nstep = S["nstep"]


def rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def rng_restore(s):
    torch.set_rng_state(s["cpu"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_partial(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- generic pool evaluation
def eval_pool(model, batches, Tc_of, head_times_of, armsets, arm_names, qlogc, device, emask_fn,
              with_labels=True, kgrid_of=None):
    """Roll out every batch once; heads of every seed evaluated at the head decision times.
    batches yields (x, y, meta dict of numpy arrays). Returns obs, ev, meta dicts (numpy)."""
    O, E, M = {}, {}, {}
    for x, y, meta in batches:
        x = x.to(device)
        Tc = Tc_of(x)
        ht = head_times_of(Tc)
        heads = {}
        for s, A in enumerate(armsets):
            for n in arm_names:
                if n in A.heads:
                    heads[f"h{s}_{n}"] = (A.heads[n], A.arms[n][0])
        R = rollout(model, x, Tc, head_times=ht, heads=heads, qact_logc=qlogc)
        o = {k: v.cpu().numpy().astype(np.float32) for k, v in R["obs"].items()}
        for k, v in R["hout"].items():
            if k.endswith("_bits"):
                o[k] = np.packbits(v.cpu().numpy(), axis=2)
            else:
                o[k] = v.cpu().numpy().astype(np.float16)
        for k, v in o.items():
            O.setdefault(k, []).append(v)
        if with_labels:
            y = y.to(device)
            ev = eval_arrays(R["Z"], y, emask_fn(x, y), Tc, kgrid_of(Tc) if kgrid_of else ht)
            for k, v in ev.items():
                v = v.cpu().numpy()
                E.setdefault(k, []).append(v.astype(np.float16) if k == "K" else v)
        fc = first_cons(R["Z"], Tc).cpu().numpy()
        M.setdefault("first_cons", []).append(fc)
        for k, v in meta.items():
            M.setdefault(k, []).append(np.asarray(v))
        del R
    cat = lambda D: {k: np.concatenate(v) for k, v in D.items()}  # noqa: E731
    return cat(O), cat(E), cat(M)


def save_pool(out, name, O, E, M, info):
    np.savez_compressed(os.path.join(out, f"obs_{name}.npz"), **O, **{"meta_" + k: v for k, v in M.items()},
                        info=json.dumps(info))
    if E:
        np.savez_compressed(os.path.join(out, f"eval_{name}.npz"), **E)


def save_heads(out, name, armsets):
    torch.save([A.state() for A in armsets], os.path.join(out, f"heads_{name}.pt"))


# ---------------------------------------------------------------- 1D tasks (T1 prefix, T2m)
def one_d_train(model, sampler, n_batches, armsets, names, ponder_names, device, logf, n_dec=32, part=None):
    g = torch.Generator().manual_seed(4242)
    i0 = 0
    if part and os.path.exists(part):   # resume after a preemption: heads, optimizers, generators, RNG
        S = torch.load(part, map_location="cpu", weights_only=False)
        for A, s in zip(armsets, S["sets"]):
            armset_restore(A, s, device)
        g.set_state(S["g"])
        rng_restore(S["rng"])
        i0 = S["i"]
        log(f"heads resumed at batch {i0}", logf)
    for i in range(i0, n_batches):
        x, y, L, Tc = sampler(i)
        R = rollout(model, x, Tc, keep_all_feats=True)
        D = decision_times(Tc - 1, n_dec, g, lo=1)
        emask = torch.ones_like(y, dtype=torch.bool)
        for A in armsets:
            ls = A.fit_batch(R["F"], R["Z"], D, Tc, y, emask, names, passes=2)
            if ponder_names:
                ls.update(A.fit_ponder(R["F"], R["Z"], Tc, y, ponder_names))
        if i % 20 == 0 or i == n_batches - 1:
            log(f"heads batch {i}/{n_batches} L={L} " + " ".join(f"{k}={v:.3f}" for k, v in ls.items()), logf)
        del R
        if part and (i + 1) % 10 == 0 and i + 1 < n_batches:
            save_partial(part, {"i": i + 1, "g": g.get_state(), "rng": rng_state(),
                                "sets": [armset_ckpt(A) for A in armsets]})


def one_d_capacity(model, sampler, n_batches, armsets, device):
    g = torch.Generator().manual_seed(999)
    reps = []
    for A in armsets:
        acc = {}
        for i in range(n_batches):
            x, y, L, Tc = sampler(10_000 + i)
            R = rollout(model, x, Tc, keep_all_feats=True)
            D = decision_times(Tc - 1, 32, g, lo=1)
            capacity_stats(A, [n for n in ("R4", "R7", "R7abl", "R8") if n in A.heads], R["F"], R["Z"], D, Tc, y,
                           torch.ones_like(y, dtype=torch.bool), acc)
            del R
        reps.append(capacity_report(acc))
    return reps


def run_1d(a, task):
    device = a.device
    os.makedirs(a.out, exist_ok=True)
    logf = os.path.join(a.out, "log.txt")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if task == "t1":
        model = load_pilot(a.pilot, device)
        w = model.w
        from mresnet import prefix_batch

        def sampler(i, n=a.bs):
            gg = torch.Generator().manual_seed(50_000 + i)
            L = int(torch.randint(8, 65, (1,), generator=gg))
            x, y, _ = prefix_batch(n, L, L, gg, device)
            return x, y, L, 2 * L
        seeds = list(range(a.head_seeds))
        pools = [(f"L{L}", L, 2 * L, 100) for L in (32, 48, 64)]
        c_range = (1e-3, 10.0)
    else:
        ck = torch.load(a.bb, map_location="cpu", weights_only=False)
        w = ck["args"]["width"]
        model = T2mNet(w=w).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        def sampler(i, n=a.bs):
            gg = torch.Generator().manual_seed(a.bseed * 1_000_000 + 50_000 + i)
            L = 64 if i % 2 == 0 else 128
            x, y, _, _ = t2m_batch(n, L, gg, device)
            return x, y, L, L
        seeds = list(range(a.head_seeds))
        pools = [("L64", 64, 64, 500), ("L128", 128, 128, 500), ("L256", 256, 256, 200)]
        c_range = (1e-3, 10.0)
    arms = {k: v for k, v in ARMS_T2M.items()}
    names = [n for n, v in arms.items() if not v[2].startswith("ponder")]
    ponder = [n for n, v in arms.items() if v[2].startswith("ponder")]
    cin = w + 5
    tag = f"b{a.bseed}"
    hpath = os.path.join(a.out, f"heads_{tag}.pt")
    armsets = [ArmSet(arms, cin, 1, s, device, c_range=c_range, delta=1) for s in seeds]
    if os.path.exists(hpath):
        st = torch.load(hpath, map_location="cpu", weights_only=False)
        for A, s in zip(armsets, st):
            for n, sd in s["heads"].items():
                A.heads[n].load_state_dict(sd)
        log(f"loaded heads {hpath}", logf)
    else:
        t0 = time.time()
        one_d_train(model, sampler, a.head_batches, armsets, names, ponder, device, logf, part=hpath + ".partial")
        save_heads(a.out, tag, armsets)
        if os.path.exists(hpath + ".partial"):
            os.remove(hpath + ".partial")
        log(f"heads trained in {time.time() - t0:.0f}s; params {armsets[0].params()}", logf)
    capp = os.path.join(a.out, f"capacity_{tag}.json")
    if not os.path.exists(capp):
        rep = one_d_capacity(model, sampler, a.cap_batches, armsets, device)
        json.dump(rep, open(capp, "w"), indent=1)
        log(f"capacity written {capp}", logf)
    qlogc = torch.log(torch.tensor(c_grid(c_range), dtype=torch.float32, device=device))
    all_names = names + ponder
    macs_bb = backbone_macs_per_pos("t2m", w)
    macs_head = {n: head_macs_per_pos(cin, v[1] if v[0] != "qact" else 2, 1) for n, v in arms.items()}
    info_base = {"task": task, "w": w, "bseed": a.bseed, "head_seeds": seeds, "c_grid": c_grid(c_range).tolist(),
                 "macs_backbone_per_pos": macs_bb, "macs_head_per_pos": macs_head, "grid": 1,
                 "arms": {n: list(v) for n, v in arms.items()}}
    # dev pool from the training distribution (knob ranges)
    dev_pools = [("dev64", 64, 64, 128), ("dev128", 128, 128, 128)] if task == "t2m" else \
        [(f"dev{L}", L, 2 * L, 32) for L in (16, 32, 48, 64)]
    for name, L, Tc, n in dev_pools + pools:
        n = max(8, int(n * a.pool_scale))
        full = f"{name}_{tag}"
        if marker(a.out, full):
            continue
        gg = torch.Generator().manual_seed(a.bseed * 1_000_000 + (777 if name.startswith("dev") else 888) * 1000 + L)

        def batches(n=n, L=L):
            bs = min(n, 250 if task == "t2m" else 100)
            for s in range(0, n, bs):
                if task == "t2m":
                    x, y, _, meta = t2m_batch(min(bs, n - s), L, gg, device)
                    meta = {k: v.cpu().numpy() for k, v in meta.items()}
                else:
                    from mresnet import prefix_batch
                    x, y, _ = prefix_batch(min(bs, n - s), L, L, gg, device)
                    meta = {"L": np.full(x.shape[0], L)}
                yield x, y, meta
        O, E, M = eval_pool(model, batches(), lambda x, Tc=Tc: Tc, lambda Tc: list(range(1, Tc + 1)), armsets,
                            all_names, qlogc, device, lambda x, y: torch.ones_like(y, dtype=torch.bool))
        save_pool(a.out, full, O, E, M, dict(info_base, pool=name, L=L, Tc=Tc, head_times=list(range(1, Tc + 1))))
        mark(a.out, full)
        log(f"pool {full} done", logf)


# ---------------------------------------------------------------- mazes
MAZE_BS = {9: 100, 13: 100, 19: 50, 25: 50, 33: 40, 59: 25, 99: 10}


def maze_emask(x, y):
    return x.max(1).values > 0


def maze_batches(ds, bs, device):
    for x, y, ii in ds.batches(bs):
        yield x, y, {"idx": ii, "size": np.full(len(ii), ds.n), "npix": np.full(len(ii), x.shape[-1] * x.shape[-2])}


def maze_setup(a):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = load_dtrecall(a.ckpt, a.device)
    return model


def run_maze_repro(a):
    """Reproduction gate: accuracy at fixed T on a split disjoint from dev/pool/eval."""
    os.makedirs(a.out, exist_ok=True)
    model = maze_setup(a)
    res = {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    for n, cnt, T in ((13, 600, 200), (59, 300, 1000)):
        x_all, y_all = load_maze_arrays(a.data, n)
        idx = np.arange(1500, 1500 + cnt)
        corr = np.zeros(T + 1)
        for s in range(0, cnt, MAZE_BS[n]):
            ii = idx[s:s + MAZE_BS[n]]
            x = torch.from_numpy(np.asarray(x_all[ii], dtype=np.float32)).to(a.device)
            y = torch.from_numpy(np.asarray(y_all[ii], dtype=np.float32)).to(a.device)
            em = maze_emask(x, y)
            with torch.no_grad():
                st = model.init_state(x)
                for t in range(1, T + 1):
                    st, logit, _ = model.step_once(st)
                    pred = ((logit > 0) & em).float()
                    corr[t] += (pred == y).flatten(1).all(1).float().sum().item()
        acc = corr / cnt
        res[str(n)] = {"n": cnt, "acc_at": {str(t): acc[t] for t in (30, 50, 70, 100, 156, 200, 300, 500, 708, 800, 984, 1000) if t <= T},
                       "best": float(acc.max()), "best_t": int(acc.argmax()),
                       "best_le_tcap": float(acc[:t_cap_maze(n) + 1].max()),
                       "curve_every10": acc[::10].tolist()}
        log(f"repro {n}: best {acc.max():.4f} at {acc.argmax()}, best<=Tcap {acc[:t_cap_maze(n)+1].max():.4f}")
    json.dump(res, open(os.path.join(a.out, "repro.json"), "w"), indent=1)


def maze_head_arms():
    return dict(ARMS_MAZE)


def run_maze_heads(a):
    """Base heads on fresh 9x9 mazes, rollouts to 160 steps, decision times <= 96 (even)."""
    os.makedirs(a.out, exist_ok=True)
    logf = os.path.join(a.out, "log.txt")
    model = maze_setup(a)
    d = os.path.join(a.data, "maze_data_train_fresh_9")
    if not os.path.exists(os.path.join(d, "solutions.npy")):
        from data import gen_mazes
        from multiprocessing import Pool
        jobs = [(250, 9, 97_000 + k) for k in range((a.fresh_n + a.cap_n + 249) // 250)]
        with Pool(4) as p:
            outs = p.starmap(gen_mazes, jobs)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "inputs.npy"), np.concatenate([o[0] for o in outs]))
        np.save(os.path.join(d, "solutions.npy"), np.concatenate([o[1] for o in outs]))
    X = np.load(os.path.join(d, "inputs.npy"), mmap_mode="r")
    Y = np.load(os.path.join(d, "solutions.npy"), mmap_mode="r")
    arms = maze_head_arms()
    cin = 128 + 5
    c_range = (1e-2, 1e3)
    armsets = [ArmSet(arms, cin, 2, s, a.device, c_range=c_range, delta=4) for s in range(a.head_seeds)]
    names = list(arms)
    g = torch.Generator().manual_seed(31337)
    Ttr, Dmax, bs = 160, 96, MAZE_BS[9]
    t0 = time.time()
    for bi, s in enumerate(range(0, a.fresh_n, bs)):
        x = torch.from_numpy(np.asarray(X[s:s + bs], dtype=np.float32)).to(a.device)
        y = torch.from_numpy(np.asarray(Y[s:s + bs], dtype=np.float32)).to(a.device)
        D = decision_times(Dmax, 32, g, even=True, lo=2)
        ft = sorted(set(D) | {t + 4 for t in D})
        R = rollout(model, x, Ttr, feat_times=ft)
        em = maze_emask(x, y)
        for A in armsets:
            ls = A.fit_batch(R["F"], R["Z"], D, Ttr, y, em, names, passes=2)
        if bi % 10 == 0:
            log(f"maze heads batch {bi} {time.time()-t0:.0f}s " + " ".join(f"{k}={v:.3f}" for k, v in ls.items()), logf)
        del R
    save_heads(a.out, "maze_base", armsets)
    # capacity gate on held-out fresh 9x9 mazes
    reps = []
    for A in armsets:
        acc = {}
        for s in range(a.fresh_n, a.fresh_n + a.cap_n, bs):
            x = torch.from_numpy(np.asarray(X[s:s + bs], dtype=np.float32)).to(a.device)
            y = torch.from_numpy(np.asarray(Y[s:s + bs], dtype=np.float32)).to(a.device)
            D = decision_times(Dmax, 32, g, even=True, lo=2)
            R = rollout(model, x, Ttr, feat_times=D)
            capacity_stats(A, ["R4", "R7", "R7abl", "R8"], R["F"], R["Z"], D, Ttr, y, maze_emask(x, y), acc)
            del R
        reps.append(capacity_report(acc))
    json.dump(reps, open(os.path.join(a.out, "capacity_maze.json"), "w"), indent=1)
    log(f"maze base heads done {time.time()-t0:.0f}s params {armsets[0].params()}", logf)


def load_armsets(a, name, arms, cin, dim, c_range, delta):
    st = torch.load(os.path.join(a.out, f"heads_{name}.pt"), map_location="cpu", weights_only=False)
    out = []
    for s, S in enumerate(st):
        A = ArmSet({}, cin, dim, s, a.device, c_range=c_range, delta=delta)
        for n, (kind, nout, settle) in S["arms"].items():
            A.add(n, kind, nout, settle, cin)
            A.heads[n].load_state_dict(S["heads"][n])
            if kind == "qact":
                A.tgt[n].load_state_dict(S["heads"][n])
            nv = S["norm"].get(n)
            if isinstance(nv, tuple):
                A.norm[n] = tuple(torch.tensor(v, device=a.device) for v in nv)
            elif nv is not None:
                A.norm[n] = nv.to(a.device)
        out.append(A)
    return out


def run_maze_size(a):
    """Per size: TT fit on the unlabelled pool (streaming, 2 passes per batch), then dev and
    (optionally) eval passes with every head of every seed."""
    n = a.size
    logf = os.path.join(a.out, "log.txt")
    model = maze_setup(a)
    cin, c_range = 133, (1e-2, 1e3)
    Tc = t_cap_maze(n)
    tag = f"n{n}"
    armsets = None
    hp = os.path.join(a.out, f"heads_tt_{tag}.pt")
    if os.path.exists(hp):
        armsets = load_armsets(a, f"tt_{tag}", {}, cin, 2, c_range, 4)
    else:
        armsets = load_armsets(a, "maze_base", {}, cin, 2, c_range, 4)
        for A in armsets:
            for dst, src in TT_ARMS.items():
                A.clone_arm(src, dst)
        pool = UnlabelledMazes(a.data, n, limit=a.limit)
        lab = LabelledMazes(a.data, n, "pool", limit=a.limit)          # labelled copy: read ONLY by R7TT
        g = torch.Generator().manual_seed(5000 + n)
        t0 = time.time()
        fcs = []
        part, k0 = hp + ".partial", 0
        if os.path.exists(part):   # resume after a preemption: TT heads, optimizers, generators, RNG
            S = torch.load(part, map_location="cpu", weights_only=False)
            for A, s in zip(armsets, S["sets"]):
                armset_restore(A, s, a.device)
            g.set_state(S["g"])
            rng_restore(S["rng"])
            fcs, k0 = S["fcs"], S["k"]
            log(f"TT {n} resumed at batch {k0}", logf)
        for j, ((x, ynone, ii), (_, ylab, jj)) in enumerate(zip(pool.batches(MAZE_BS[n]), lab.batches(MAZE_BS[n]))):
            if j < k0:
                continue
            assert ynone is None and (ii == jj).all()
            x = x.to(a.device)
            D = decision_times(Tc - 2, 32, g, even=True, lo=2)
            ft = sorted(set(D))
            R = rollout(model, x, Tc, feat_times=ft)
            fcs.append(first_cons(R["Z"], Tc).cpu().numpy())      # label-free, for R0L
            em = maze_emask(x, None)
            for A in armsets:
                A.fit_batch(R["F"], R["Z"], D, Tc, None, em, ["R9", "R9s", "R12TT"], passes=2)
                A.fit_batch(R["F"], R["Z"], D, Tc, ylab.to(a.device), em, ["R7TT"], passes=2)
            del R
            save_partial(part, {"k": j + 1, "g": g.get_state(), "rng": rng_state(), "fcs": fcs,
                                "sets": [armset_ckpt(A) for A in armsets]})
            log(f"TT {n} batch {j + 1} {time.time()-t0:.0f}s", logf)
        np.save(os.path.join(a.out, f"poolfc_n{n}.npy"), np.concatenate(fcs))
        save_heads(a.out, f"tt_{tag}", armsets)
        os.remove(part)
        log(f"TT {n} done {time.time()-t0:.0f}s", logf)
    names = list(armsets[0].heads)
    qlogc = torch.log(torch.tensor(c_grid(c_range), dtype=torch.float32, device=a.device))
    ht = lambda Tc_: list(range(4, Tc_ + 1, 4))  # noqa: E731
    macs_head = {nm: head_macs_per_pos(cin, v[1] if v[0] != "qact" else 2, 2) for nm, v in armsets[0].arms.items()}
    info = {"task": "maze", "size": n, "Tc": Tc, "head_times": ht(Tc), "grid": 4, "c_grid": c_grid(c_range).tolist(),
            "macs_backbone_per_pos": backbone_macs_per_pos("maze", 128), "macs_head_per_pos": macs_head,
            "head_seeds": list(range(len(armsets))), "arms": {k: list(v) for k, v in armsets[0].arms.items()},
            "tf32": True}
    for which in a.splits.split(","):
        if MAZE_SPLITS[n][{"dev": 0, "pool": 1, "eval": 2}[which]] == 0:
            continue
        full = f"{which}_{tag}"
        if marker(a.out, full):
            continue
        t0 = time.time()
        ds = LabelledMazes(a.data, n, which, limit=a.limit)
        idx_all, CH, parts = ds.idx, 4 * MAZE_BS[n], []
        for c in range(0, len(idx_all), CH):   # chunks of 4 batches, each saved (spot preemption)
            pp = os.path.join(a.out, f"part_{full}_{c:04d}.npz")
            if not os.path.exists(pp):
                ds.idx = idx_all[c:c + CH]
                O, E, M = eval_pool(model, maze_batches(ds, MAZE_BS[n], a.device), lambda x: Tc, ht, armsets, names,
                                    qlogc, a.device, maze_emask)
                with open(pp + ".tmp", "wb") as f:   # *.tmp is never synced
                    np.savez(f, **{"O_" + k: v for k, v in O.items()},
                             **{"E_" + k: v for k, v in E.items()}, **{"M_" + k: v for k, v in M.items()})
                os.replace(pp + ".tmp", pp)
                log(f"{full} chunk {c} done {time.time()-t0:.0f}s", logf)
            parts.append(pp)
        Zs = [dict(np.load(p)) for p in parts]
        O, E, M = ({k[2:]: np.concatenate([z[k] for z in Zs]) for k in Zs[0] if k.startswith(pre)}
                   for pre in ("O_", "E_", "M_"))
        save_pool(a.out, full, O, E, M, dict(info, pool=which))
        mark(a.out, full)
        for p in parts:
            os.remove(p)
        log(f"{full} done {time.time()-t0:.0f}s", logf)


def run_maze_devsimple(a):
    """Dev trajectories without heads (simple rules only) for the headroom gate."""
    n = a.size
    full = f"devs_n{n}"
    if marker(a.out, full):
        return
    model = maze_setup(a)
    Tc = t_cap_maze(n)
    t0 = time.time()
    ds = LabelledMazes(a.data, n, "dev", limit=a.limit)
    ht = list(range(4, Tc + 1, 4))
    O, E, M = eval_pool(model, maze_batches(ds, MAZE_BS[n], a.device), lambda x: Tc, lambda T_: ht, [], [],
                        None, a.device, maze_emask)
    save_pool(a.out, full, O, E, M, {"task": "maze", "size": n, "Tc": Tc, "head_times": ht, "grid": 4,
                                     "c_grid": [], "pool": "dev-simple"})
    mark(a.out, full)
    log(f"{full} done {time.time()-t0:.0f}s", os.path.join(a.out, "log.txt"))


def run_maze_gate(a):
    from acore import load_pool, maze_headroom
    res = {}
    for n in (33, 59):
        O, E, M, info = load_pool(a.out, f"devs_n{n}")
        res[str(n)] = maze_headroom(O, E, info)
    res["threshold"] = 0.30
    res["pass"] = bool(any(res[k]["headroom"] >= 0.30 for k in ("33", "59")))
    json.dump(res, open(os.path.join(a.out, "headroom_gate.json"), "w"), indent=1)
    log(f"HEADROOM GATE {json.dumps(res)}", os.path.join(a.out, "log.txt"))
    print("GATE_PASS" if res["pass"] else "GATE_FAIL")


def run_maze_poolstats(a):
    """Label-free pool statistics for R0L (first step where thresholded b_t = thresholded b_cap),
    computed from the unlabelled pool without heads."""
    n = a.size
    model = maze_setup(a)
    Tc = t_cap_maze(n)
    out = []
    for x, ynone, ii in UnlabelledMazes(a.data, n, limit=a.limit).batches(MAZE_BS[n]):
        assert ynone is None
        with torch.no_grad():
            R = rollout(model, x.to(a.device), Tc)
            out.append(first_cons(R["Z"], Tc).cpu().numpy())
        del R
    np.save(os.path.join(a.out, f"poolfc_n{n}.npy"), np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--out", default="results/t2m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bb", default=None)
    ap.add_argument("--bseed", type=int, default=0)
    ap.add_argument("--pilot", default=os.path.join(HERE, "..", "..", "market-resnet", "results", "pilot", "pilot_Bds.pt"))
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--head_seeds", type=int, default=3)
    ap.add_argument("--head_batches", type=int, default=125)
    ap.add_argument("--cap_batches", type=int, default=4)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "data", "dtrecall_model_best.pth"))
    ap.add_argument("--fresh_n", type=int, default=8000)
    ap.add_argument("--cap_n", type=int, default=300)
    ap.add_argument("--size", type=int, default=59)
    ap.add_argument("--splits", default="dev,eval")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap maze pools")
    ap.add_argument("--pool_scale", type=float, default=1.0, help="smoke: scale 1D pool sizes")
    a = ap.parse_args()
    gpu_frac(a.device)
    if a.cmd in ("t1", "t2m"):
        run_1d(a, a.cmd)
    elif a.cmd == "maze-repro":
        run_maze_repro(a)
    elif a.cmd == "maze-heads":
        run_maze_heads(a)
    elif a.cmd == "maze-size":
        run_maze_size(a)
    elif a.cmd == "maze-devsimple":
        run_maze_devsimple(a)
    elif a.cmd == "maze-gate":
        run_maze_gate(a)
    elif a.cmd == "maze-poolstats":
        run_maze_poolstats(a)
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
