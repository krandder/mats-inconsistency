"""Bansal et al. 2022 DT-recall for P15, ported with Bansal's module names so published checkpoints load
strictly. Also E-stacked (grouped-conv) DT-recall for the P-eq arm on our cell grids, the P-pub prefix
retrain (Bansal's recipe), and greedy multi-T evaluation.
The market-resnet port (../market-resnet/dtrecall.py, sha in SNAPSHOT-SHA256.txt) was read, not imported."""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p15  # noqa: E402

DEV = p15.DEV


class Block(nn.Module):
    def __init__(self, dim, w, E):
        super().__init__()
        C = nn.Conv1d if dim == 1 else nn.Conv2d
        self.conv1 = C(E * w, E * w, 3, 1, 1, bias=False, groups=E)
        self.conv2 = C(E * w, E * w, 3, 1, 1, bias=False, groups=E)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


class DT(nn.Module):
    """dim 1: Bansal dt_net_recall_1d (ReLU after the recall conv, head w->w->w/2->2).
    dim 2: dt_net_recall_2d (no ReLU after the recall conv, head w->32->8->2). E models stacked by groups."""

    def __init__(self, dim, width, cin, E=1):
        super().__init__()
        C = nn.Conv1d if dim == 1 else nn.Conv2d
        self.dim, self.w, self.cin, self.E = dim, width, cin, E
        self.projection = nn.Sequential(C(E * cin, E * width, 3, 1, 1, bias=False, groups=E), nn.ReLU())
        rec = [C(E * (width + cin), E * width, 3, 1, 1, bias=False, groups=E)]
        if dim == 1:
            rec.append(nn.ReLU())
        rec.append(nn.Sequential(Block(dim, width, E), Block(dim, width, E)))
        self.recur_block = nn.Sequential(*rec)
        h1, h2 = (width, width // 2) if dim == 1 else (32, 8)
        self.head = nn.Sequential(C(E * width, E * h1, 3, 1, 1, bias=False, groups=E), nn.ReLU(),
                                  C(E * h1, E * h2, 3, 1, 1, bias=False, groups=E), nn.ReLU(),
                                  C(E * h2, E * 2, 3, 1, 1, bias=False, groups=E))

    def cat(self, h, x):
        if self.E == 1:
            return torch.cat([h, x], 1)
        B, sp = h.shape[0], h.shape[2:]
        return torch.cat([h.view(B, self.E, self.w, *sp), x.view(B, self.E, self.cin, *sp)], 2).view(B, -1, *sp)

    def step(self, h, x):
        return self.recur_block(self.cat(h, x))

    def out(self, h):
        o = self.head(h)
        return o.view(o.shape[0], self.E, 2, *o.shape[2:])            # (B,E,2,...)


def load_pub(path, dim):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("net", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    w = sd["projection.0.weight"].shape[0]
    cin = sd["projection.0.weight"].shape[1]
    net = DT(dim, w, cin)
    net.load_state_dict(sd, strict=True)
    # Bansal's PrefixSumDataset feeds bits centred (x - 0.5); his maze images are raw 0/1.  Verified 2026-09-29
    # on the published prefix net: raw bits -> chance (0.0 exact), centred -> 1.0 exact at 32 and at 512 bits (T=300).
    net.xshift = 0.5 if dim == 1 else 0.0
    return net


# ------------------------------------------------------------------ flops (per element, analytic)
def dt_flops(dim, w, cin, cells, iters_fwd, iters_grad, head_evals):
    k = 3 if dim == 1 else 9
    f_proj = 2 * k * cin * w * cells + w * cells
    f_it = 2 * k * (w + cin) * w * cells + 4 * 2 * k * w * w * cells + 7 * w * cells
    h1, h2 = (w, w // 2) if dim == 1 else (32, 8)
    f_head = 2 * k * (w * h1 + h1 * h2 + h2 * 2) * cells + (h1 + h2) * cells + 8 * cells
    fwd_only = f_proj + iters_fwd * f_it
    with_grad = 3 * (f_proj + iters_grad * f_it + head_evals * f_head)
    return fwd_only + with_grad


# ------------------------------------------------------------------ P-eq training on our cell grids
def cell_tensor(task, ds, E):
    """per-element numpy batches -> x (B, E*Cx, L) or (B, E*6, n, n); y (B,E,N); mask (B,E,N)."""
    x = np.stack([d["x"] for d in ds], 1)                               # (B,E,N,Cx)
    B, _, N, Cx = x.shape
    if task.kind == "prefix":
        xt = torch.tensor(x, dtype=torch.float32).permute(0, 1, 3, 2).reshape(B, E * Cx, N)
    else:
        n = int(round(N ** 0.5))
        xt = torch.tensor(x, dtype=torch.float32).view(B, E, n, n, Cx).permute(0, 1, 4, 2, 3).reshape(B, E * Cx, n, n)
    y = torch.tensor(np.stack([d["y"] for d in ds], 1))
    m = torch.tensor(np.stack([d["valid"] for d in ds], 1), dtype=torch.float32)
    return xt.to(DEV), y.to(DEV), m.to(DEV)


def dt_loss(net, x, y, m, iters):
    h = net.projection(x)
    for _ in range(iters):
        h = net.step(h, x)
    o = net.out(h).flatten(3)                                           # (B,E,2,N)
    ce = F.cross_entropy(o.permute(0, 1, 3, 2).reshape(-1, 2), y.reshape(-1), reduction="none").view(y.shape)
    return ((ce * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1)).sum()


def peq_job(job, log=print):
    """job: {id, kind, D, seeds, widths:[w], frac, lr}. Trains E = widths x seeds? no: one width per job,
    E = seeds, Bansal progressive loss (alpha 1 prefix / 0.01 maze), Adam + L2 2e-4, clip 1 per element,
    exponential warmup over 5% of the budget, x0.1 at 40% and 67% of the budget (Bansal's 60/100 of 150)."""
    task = p15.Task(job["kind"], job["D"])
    seeds, w = job["seeds"], job["width"]
    E = len(seeds)
    D = task.D
    N = D if task.kind == "prefix" else task.n ** 2
    dim = 1 if task.kind == "prefix" else 2
    alpha = 1.0 if task.kind == "prefix" else 0.01
    outf = os.path.join(p15.RESDIR, job["id"] + ".json")
    ck = os.path.join(p15.CKDIR, job["id"] + ".pt")
    if os.path.exists(outf) and not job.get("force"):
        return json.load(open(outf))
    torch.manual_seed(1234 + seeds[0])
    nets = []
    for s in seeds:  # each element's init from its own seed, then stacked into grouped convs
        torch.manual_seed(5000 + s)
        nets.append(DT(dim, w, task.Cx))
    net = DT(dim, w, task.Cx, E).to(DEV)
    with torch.no_grad():
        for (name, p) in net.named_parameters():
            p.copy_(torch.cat([dict(nn_.named_parameters())[name] for nn_ in nets], 0))
    opt = torch.optim.Adam(net.parameters(), lr=job.get("lr", 1e-3), weight_decay=2e-4)
    Btot = p15.budget(task) * job.get("frac", 1.0)
    rngs = [np.random.default_rng([1000, s]) for s in seeds]
    srng = np.random.default_rng([3100, D])
    st = {"step": 0, "flops": 0.0, "curve": [], "wall": 0.0}
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        st = s["st"]
        for r, bs in zip(rngs, s["rngs"]):
            r.bit_generator.state = bs
        srng.bit_generator.state = s["srng"]
    marks = [Btot * (i + 1) / 10 for i in range(10)]
    ind = task.eval_set("indist")
    t0, last = time.time(), time.time()

    def save():
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "st": st,
                    "rngs": [r.bit_generator.state for r in rngs], "srng": srng.bit_generator.state}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    groups = [list(range(i * 1, (i + 1) * 1)) for i in range(E)]
    params = list(net.parameters())
    while len(st["curve"]) < 10:
        fr = st["flops"] / Btot
        lr = job.get("lr", 1e-3) * (1 - math.exp(-(fr / 0.05))) * (0.1 if fr > 0.4 else 1) * (0.1 if fr > 0.67 else 1)
        for g in opt.param_groups:
            g["lr"] = lr
        ds = [task.train_batch(r, 32) for r in rngs]
        x, y, m = cell_tensor(task, ds, E)
        n_ = int(srng.integers(0, D))
        k_ = int(srng.integers(1, D - n_ + 1))
        opt.zero_grad(set_to_none=True)
        h = net.projection(x)
        if n_ > 0:
            with torch.no_grad():
                for _ in range(n_):
                    h = net.step(h, x)
            h = h.detach()
        for _ in range(k_):
            h = net.step(h, x)
        o = net.out(h).flatten(3)
        ce = F.cross_entropy(o.permute(0, 1, 3, 2).reshape(-1, 2), y.reshape(-1), reduction="none").view(y.shape)
        loss = alpha * ((ce * m).sum((0, 2)) / m.sum((0, 2))).sum()
        fl = dt_flops(dim, w, task.Cx, 32 * N, n_, k_, 1)
        if alpha < 1:
            loss = loss + (1 - alpha) * dt_loss(net, x, y, m, D)
            fl += dt_flops(dim, w, task.Cx, 32 * N, 0, D, 1)
        loss.backward()
        with torch.no_grad():  # per-element grad-norm clip at 1 (Bansal clip=1 per model)
            sq = torch.zeros(E, device=DEV)
            for p in params:
                if p.grad is not None:
                    sq += p.grad.view(E, -1).pow(2).sum(1)
            sc = (1.0 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in params:
                if p.grad is not None:
                    p.grad.view(E, -1).mul_(sc[:, None])
        opt.step()
        st["step"] += 1
        st["flops"] += fl
        if st["flops"] >= marks[len(st["curve"])]:
            r = dt_indist(net, task, ind)
            r.update({"step": st["step"], "flops": st["flops"], "wall": st["wall"] + time.time() - t0})
            st["curve"].append(r)
            log(f"[{job['id']}] step {st['step']} exact {np.round(r['exact'], 2).tolist()} ll {np.round(r['logloss'], 3).tolist()}")
        if time.time() - last > 600:
            st["wall"] += time.time() - t0
            t0 = time.time()
            save()
            last = time.time()
    st["wall"] += time.time() - t0
    save()
    res = {"job": job, "steps": st["step"], "flops": st["flops"], "B_D": p15.budget(task), "wall_train": st["wall"],
           "curve": st["curve"], "elements": [{"seed": s, "width": w} for s in seeds]}
    if job.get("final", True):
        t0 = time.time()
        res["indist"] = dt_indist(net, task, ind)
        for which in job.get("evals", ("val", "test")):
            rs = []
            for size, num in p15.eval_sizes(task, which):
                d = task.eval_set(which, size, num)
                Ts = p15.t_grid(task.size_T(size), p15.CS)
                g = dt_greedy(net, task.kind, d, Ts, cap=job.get("cap", 3000) if task.kind == "maze" else None)
                rs.append({"size": size, "num": num, "Ts": g["Ts"], **{k: v for k, v in g.items() if k != "Ts"}})
                log(f"[{job['id']}] {which} {size}: {np.round(g['exact'], 2).tolist()}")
            res[which] = rs
        res["wall_eval"] = time.time() - t0
    json.dump(res, open(outf + ".tmp", "w"))
    os.replace(outf + ".tmp", outf)
    return res


@torch.no_grad()
def dt_indist(net, task, d):
    E = net.E
    M, N = d["y"].shape
    out = {"exact": np.zeros(E), "logloss": np.zeros(E)}
    bs = 250
    for m0 in range(0, M, bs):
        sl = slice(m0, min(M, m0 + bs))
        dd = [{k: d[k][sl] for k in ("x", "y", "valid")}] * E
        x, y, m = cell_tensor(task, dd, E)
        x = x - getattr(net, "xshift", 0.0)
        h = net.projection(x)
        for _ in range(task.D):
            h = net.step(h, x)
        o = net.out(h).flatten(3)                                       # (B,E,2,N)
        lp = F.log_softmax(o, 2).clamp(min=math.log(1e-4), max=math.log(1 - 1e-4))
        ll = -lp.gather(2, y[:, :, None]).squeeze(2)
        out["logloss"] += ((ll * m).sum(2) / m.sum(2)).sum(0).cpu().numpy()
        ok = ((o.argmax(2) == y) | (m == 0)).all(2)
        out["exact"] += ok.float().sum(0).cpu().numpy()
    return {k: (v / M).tolist() for k, v in out.items()}


@torch.no_grad()
def dt_greedy(net, kind, d, Ts, cap=None, pixels=False, bs=None, amp=False):
    """Greedy exact accuracy at each T (capped). kind prefix: d x (M,L,1) cells; maze on cells (6 ch) unless
    pixels=True: then d holds 'img' (M,3,P,P) and 'sol' (M,P,P) and Bansal's pixel metric is used.
    Also records per-cell log loss and mean max |logit| (runaway check)."""
    E = net.E
    Ts = sorted(set(min(t, cap) if cap else t for t in Ts))
    if pixels:
        M = d["img"].shape[0]
    else:
        M, N = d["y"].shape
    res = {"exact": np.zeros((E, len(Ts))), "logloss": np.zeros((E, len(Ts))), "maxlogit": np.zeros((E, len(Ts))),
           "Ts": Ts}
    cellmode = pixels and "ycell" in d          # also score the cell centres against the cell labels
    if cellmode:
        res["exact_cells"] = np.zeros((E, len(Ts)))
    if bs is None:
        cells = (d["img"].shape[2] ** 2) if pixels else N
        bs = max(1, min(M, int(8e6 // (cells * net.w * max(E, 1) / 32))))
    for m0 in range(0, M, bs):
        sl = slice(m0, min(M, m0 + bs))
        if pixels:
            x = torch.tensor(d["img"][sl], device=DEV)
            y = torch.tensor(d["sol"][sl], device=DEV).long().flatten(1)[:, None]      # (B,1,P*P)
            m = (x.max(1)[0] > 0).float().flatten(1)[:, None]
        else:
            dd = [{k: d[k][sl] for k in ("x", "y", "valid")}] * E
            x, y, m = cell_tensor(p15.Task(kind, 8 if kind == "prefix" else 8), dd, E)
        x = x - getattr(net, "xshift", 0.0)
        h = net.projection(x)
        ti = 0
        for it in range(1, Ts[-1] + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and DEV == "cuda"):
                h = net.step(h, x)
            if it == Ts[ti]:
                o = net.out(h.float()).flatten(3)                          # (B,E,2,N)
                pred = o.argmax(2)
                if pixels:
                    pred = pred * m.long()
                    ok = (pred == y).all(2)
                    if cellmode:
                        P_ = x.shape[-1]
                        pc = pred.view(pred.shape[0], P_, P_)[:, 3:P_ - 3:4, 3:P_ - 3:4].flatten(1)
                        yc = torch.tensor(d["ycell"][sl], device=DEV).long()
                        res["exact_cells"][:, ti] += (pc == yc).all(1).float().sum().cpu().numpy()
                else:
                    ok = ((pred == y) | (m == 0)).all(2)
                lp = F.log_softmax(o, 2).clamp(min=math.log(1e-4), max=math.log(1 - 1e-4))
                ll = -lp.gather(2, y[:, :, None]).squeeze(2)
                mm = m.expand(-1, E, -1) if pixels else m
                res["exact"][:, ti] += ok.float().sum(0).cpu().numpy()
                res["logloss"][:, ti] += ((ll * mm).sum(2) / mm.sum(2)).sum(0).cpu().numpy()
                res["maxlogit"][:, ti] += (((o[:, :, 1] - o[:, :, 0]).abs() * mm).sum(2) / mm.sum(2)).sum(0).cpu().numpy()
                ti += 1
                if ti >= len(Ts):
                    break
    for k in ("exact", "logloss", "maxlogit", "exact_cells"):
        if k in res:
            res[k] = (res[k] / M).tolist()
    return res


# ------------------------------------------------------------------ P-pub retrain (Bansal prefix recipe)
def pub_retrain(seed, data_dir, out_dir, epochs=150, width=400, log=print, centre=True):
    """Bansal prefix-sums recipe: 32-bit training data (80% of the 10k file), batch 100, Adam lr 1e-3
    (L2 2e-4), exponential warmup 10 epochs, x0.1 at epochs 60 and 100, clip 1, alpha 1, max_iters 30.
    centre=True subtracts 0.5 from the bits as Bansal's prefix-sums loader does (added 2026-09-29 after the
    uncentred retrain scored 0.0 at 512 bits); centre=False is the earlier uncentred run, kept as an ablation."""
    xs = 0.5 if centre else 0.0
    ck = os.path.join(out_dir, f"ppub_c_s{seed}.pt" if centre else f"ppub_s{seed}.pt")
    X = torch.load(os.path.join(data_dir, "32_data.pth"), weights_only=False).float()
    Y = torch.load(os.path.join(data_dir, "32_targets.pth"), weights_only=False).long()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    tr = perm[: int(0.8 * len(X))]
    torch.manual_seed(seed)
    net = DT(1, width, 1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=2e-4)
    ep0 = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        ep0 = s["epoch"]
        g.set_state(s["g"].cpu())  # map_location=DEV moves the RNG ByteTensor to the GPU
    rs = np.random.default_rng([seed, 77])
    for ep in range(ep0, epochs):
        lr = 1e-3 * (1 - math.exp(-(ep + 1) / 10)) * (0.1 if ep >= 60 else 1) * (0.1 if ep >= 100 else 1)
        for gg in opt.param_groups:
            gg["lr"] = lr
        order = tr[torch.randperm(len(tr), generator=g)]
        tot = 0.0
        for b0 in range(0, len(order), 100):
            idx = order[b0:b0 + 100]
            x = X[idx][:, None].to(DEV) - xs
            y = Y[idx].to(DEV)
            n_ = int(rs.integers(0, 30))
            k_ = int(rs.integers(1, 30 - n_ + 1))
            opt.zero_grad(set_to_none=True)
            h = net.projection(x)
            if n_ > 0:
                with torch.no_grad():
                    for _ in range(n_):
                        h = net.step(h, x)
                h = h.detach()
            for _ in range(k_):
                h = net.step(h, x)
            o = net.out(h)[:, 0]
            loss = F.cross_entropy(o, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "g": g.get_state()}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"[ppub s{seed}] epoch {ep + 1} loss {tot / (len(order) / 100):.4f}")
    net.xshift = xs
    return net


def ref_logit_check(net_path, dim, ref_dir="/tmp/p15dt"):
    """Our port vs Bansal's own DTNet classes on the same weights (fp64, CPU, small input)."""
    import types
    mod = types.ModuleType("dtref")
    src_b = open(os.path.join(ref_dir, "blocks.py")).read()
    exec(src_b, mod.__dict__)
    src = open(os.path.join(ref_dir, "dt_net_1d.py" if dim == 1 else "dt_net_2d.py")).read()
    src = src.replace("from .blocks import", "from dtref_blocks import")
    sys.modules["dtref_blocks"] = mod
    m2 = types.ModuleType("dtref2")
    exec(src, m2.__dict__)
    sd = torch.load(net_path, map_location="cpu", weights_only=False)
    sd = sd.get("net", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    w = sd["projection.0.weight"].shape[0]
    if dim == 1:
        ref = m2.DTNet1D(mod.BasicBlock1D, [2], w, recall=True)
        x = torch.randint(0, 2, (4, 1, 48)).double()
    else:
        ref = m2.DTNet(mod.BasicBlock2D, [2], w, in_channels=3, recall=True)
        img = p15.cells_to_pixels(p15.gen_mazes(2, 7, 5)["x"], 7)
        x = torch.tensor(img).double()
    ref.load_state_dict(sd, strict=True)
    ref = ref.double().eval()
    ours = load_pub(net_path, dim).double().eval()
    with torch.no_grad():
        a = ref(x, iters_to_do=20)[:, -1]
        h = ours.projection(x)
        for _ in range(20):
            h = ours.step(h, x)
        b = ours.out(h)[:, 0]
    return float((a - b).abs().max())
