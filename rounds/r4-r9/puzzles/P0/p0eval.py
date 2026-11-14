"""P0 harness: one frozen evaluation for DT-Recall, DT-L and MarketResNet.

evalsuite(adapter, sets, out_dir, ...) streams init -> step -> readout (never a stack of all
outputs), in strict fp32, and records per instance:
  - exact match at every iteration (bit-packed), so any stopping rule is scored post hoc;
  - at grid points: per-cell NLL, min-over-cells confidence, KL(b_t || b_{t-1}) (mean over open
    cells / real bits), latent RMS, maze checkers (ours and Knutson's), the running
    max-confidence pick (deep-thinking test_max_conf) and the stop-probe probability;
  - first iteration where KL < eps for each eps; the max-confidence patience-10 stop;
  - end state at T_cap: fixed point / 2-cycle / other / nonfinite (relative tolerance 0.01);
  - pooled ECE histograms at grid points; declared FLOPs checked with FlopCounterMode.
Iteration indices are int32 (0 = never), the rest float16.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import check, knutson, expand  # noqa: E402

GRID = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000,
        5000, 10000, 20000, 30000]
EPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
PATIENCE = 10
TOL = 0.01


def strict_fp32():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ------------------------------------------------------------------ data
def load_frozen(frozen_dir):
    man = json.load(open(os.path.join(frozen_dir, "manifest.json")))
    cache, sets = {}, {}
    for name, meta in man["sets"].items():
        f = meta["file"]
        if f not in cache:
            cache[f] = np.load(os.path.join(frozen_dir, f))
        z = cache[f]
        keys = [k.split("/", 1)[1] for k in z.files if k.startswith(name + "/")]
        sets[name] = {"meta": meta, **{k: z[f"{name}/{k}"] for k in keys}}
    return sets, man


def verify_frozen(sets):
    import hashlib
    bad = []
    for name, d in sets.items():
        for k, h in d["meta"].get("sha256", {}).items():
            if hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() != h:
                bad.append(f"{name}/{k}")
    return bad


def subset(d, idx, name=None):
    """Slice every per-instance array of a frozen set (never regenerate)."""
    out = {"meta": dict(d["meta"])}
    for k, v in d.items():
        if k == "meta":
            continue
        out[k] = v[idx] if (isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == d["meta"]["n"]) else v
    out["meta"]["n"] = len(out["code"] if "code" in out else out["x"])
    if name:
        out["meta"]["name"] = name
    return out


def make_batch(d, idx, device):
    kind = d["meta"]["kind"]
    if kind == "maze":
        code = torch.as_tensor(d["code"][idx])
        n = code.shape[-1]
        sol = torch.as_tensor(np.unpackbits(d["sol"][idx], axis=-1, count=n))
        x, s = expand(code, sol)
        x, s = x.to(device), s.to(device)
        mask = x.max(1)[0] > 0
        return {"x": x, "y": s.flatten(1), "mask": mask.flatten(1), "sol_px": s, "code": code.to(device),
                "sp_len": torch.as_tensor(d["sp_len"][idx]).to(device), "HW": tuple(s.shape[1:])}
    tok = torch.as_tensor(d["x"][idx]).long().to(device)
    mask = tok != 2
    y = torch.where(mask, torch.cumsum(torch.where(mask, tok, 0), 1) % 2, 0)
    return {"tok": tok, "y": y.float(), "mask": mask}


# ------------------------------------------------------------------ one batch
@torch.no_grad()
def run_batch(ad, b, Tc, grid, probe=None, want_feat=False, kind="maze"):
    x = ad.prep(b)
    s = ad.init(x)
    m = b["mask"]
    mf = m.float()
    nm = mf.sum(1).clamp(min=1)
    y = b["y"] > 0.5
    B = m.shape[0]
    dev = m.device
    G = len(grid)
    gi = {t: i for i, t in enumerate(grid)}
    correct = torch.zeros(B, Tc, dtype=torch.bool, device=dev)
    g = {k: torch.zeros(B, G, device=dev) for k in ("nll", "minconf", "kl", "hrms", "probe")}
    g["best_idx"] = torch.zeros(B, G, dtype=torch.long, device=dev)
    if kind == "maze":
        g["check"] = torch.zeros(B, G, dtype=torch.bool, device=dev)
        g["knutson"] = torch.zeros(B, G, dtype=torch.bool, device=dev)
    ece = torch.zeros(G, 10, 3, dtype=torch.float64, device=dev)
    kl_first = torch.zeros(B, len(EPS), dtype=torch.long, device=dev)
    eps_t = torch.tensor(EPS, device=dev)
    best_conf = torch.full((B,), -1.0, device=dev)
    best_idx = torch.zeros(B, dtype=torch.long, device=dev)
    since = torch.zeros(B, dtype=torch.long, device=dev)
    mc_stop = torch.zeros(B, dtype=torch.long, device=dev)
    mc_pick = torch.zeros(B, dtype=torch.long, device=dev)
    feats = []
    z = ad.readout(s)
    lp_prev, l1p_prev = F.logsigmoid(z), F.logsigmoid(-z)
    tail = []
    for t in range(1, Tc + 1):
        s = ad.step(s, x)
        z = ad.readout(s)
        lp, l1p = F.logsigmoid(z), F.logsigmoid(-z)
        pred = z > 0
        fin = torch.isfinite(z).all(1)
        ok = ((pred == y) | ~m).all(1) & fin
        correct[:, t - 1] = ok
        p = lp.exp()
        conf = torch.maximum(p, 1 - p)
        csum = torch.nan_to_num((conf * mf).sum(1), nan=-1.0)
        imp = csum >= best_conf               # ties move the pick to the later iteration, as torch.cummax
        best_conf = torch.where(imp, csum, best_conf)
        best_idx = torch.where(imp, torch.full_like(best_idx, t), best_idx)
        since = torch.where(imp, torch.zeros_like(since), since + 1)
        hit = (since >= PATIENCE) & (mc_stop == 0)
        mc_stop = torch.where(hit, torch.full_like(mc_stop, t), mc_stop)
        mc_pick = torch.where(hit, best_idx, mc_pick)
        kl = torch.nan_to_num(((p * (lp - lp_prev) + (1 - p) * (l1p - l1p_prev)) * mf).sum(1) / nm, nan=1e9)
        cond = (kl[:, None] < eps_t[None]) & (kl_first == 0)
        kl_first = torch.where(cond, torch.full_like(kl_first, t), kl_first)
        lp_prev, l1p_prev = lp, l1p
        if t in gi:
            i = gi[t]
            nll = -(torch.where(y, lp, l1p) * mf).sum(1) / nm
            g["nll"][:, i] = nll
            g["minconf"][:, i] = torch.where(m, conf, torch.ones_like(conf)).amin(1)
            g["kl"][:, i] = kl
            lat = ad.latent(s)
            g["hrms"][:, i] = lat.flatten(1).pow(2).mean(1).sqrt()
            g["best_idx"][:, i] = best_idx
            cc = ((pred == y) & m)
            bins = ((conf - 0.5) * 20).clamp(0, 9.999).long()
            for k in range(10):
                sel = m & (bins == k)
                ece[i, k, 0] += sel.sum()
                ece[i, k, 1] += (conf * sel).sum()
                ece[i, k, 2] += (cc & sel).sum()
            if kind == "maze":
                ppx = (pred & m).view(B, *b["HW"]).long()
                g["check"][:, i] = check(ppx, b["code"], b["sp_len"]) & fin
                g["knutson"][:, i] = knutson(b["x"], ppx, b["sol_px"]) & fin
            if probe is not None or want_feat:
                mm = mf[:, None, :] if lat.shape[-1] == mf.shape[-1] else None
                fmean = (lat * mm).sum(2) / nm[:, None]
                fmax = (lat + (mm - 1) * 1e4).amax(2)
                f = torch.cat([fmean, fmax], 1)
                if want_feat:
                    feats.append(f.float().cpu())
                if probe is not None:
                    fz = (f - probe["mu"]) / probe["sd"]
                    g["probe"][:, i] = torch.sigmoid(fz @ probe["w"] + probe["b"])
        if t >= Tc - 2:
            tail.append(ad.latent(s).flatten(1).clone())
    hT, h1, h2 = tail[-1], tail[-2], tail[-3]
    nT = hT.norm(dim=1).clamp(min=1e-12)
    r1 = (hT - h1).norm(dim=1) / nT
    r2 = (hT - h2).norm(dim=1) / nT
    finite = torch.isfinite(hT).all(1) & torch.isfinite(r1) & torch.isfinite(r2)
    end = torch.where(~finite, 3, torch.where(r1 < TOL, 0, torch.where(r2 < TOL, 1, 2)))
    out = {"correct": correct.cpu().numpy(), "kl_first": kl_first.int().cpu().numpy(),
           "mc_stop": mc_stop.int().cpu().numpy(), "mc_pick": mc_pick.int().cpu().numpy(),
           "end_state": end.to(torch.int8).cpu().numpy(),
           "end_r1": r1.half().cpu().numpy(), "end_r2": r2.half().cpu().numpy()}
    for k, v in g.items():
        if v.dtype == torch.bool:
            out[k] = v.cpu().numpy()
        elif v.dtype == torch.long:
            out[k] = v.int().cpu().numpy()
        else:
            out[k] = v.clamp(-6e4, 6e4).half().cpu().numpy()
    out["ece"] = ece.cpu().numpy()
    if want_feat:
        out["feat"] = torch.stack(feats, 1).numpy()   # (B, G, F), not saved
    return out


def batch_size(d, width):
    if d["meta"]["kind"] == "maze":
        P = d["meta"]["pixels"] ** 2
    else:
        P = d["x"].shape[1]
    return int(max(1, min(d["meta"]["n"], 64e6 // (P * width))))


def flop_check(ad, d, device):
    from torch.utils.flop_counter import FlopCounterMode
    b = make_batch(d, np.arange(2), device)
    x = ad.prep(b)
    fn = ad.flop_probe(x)
    with FlopCounterMode(display=False) as fc:
        fn()
    P = b["mask"].shape[1]
    measured = fc.get_total_flops() / (2 * P)
    return {"declared_per_pos": ad.flops_per_pos, "measured_per_pos": measured,
            "ratio": measured / ad.flops_per_pos, "ok": abs(measured / ad.flops_per_pos - 1) <= 0.10}


def eval_set(ad, d, device, width, probe=None, want_feat=False, Tc=None):
    Tc = Tc or d["meta"]["T_cap"]
    grid = [t for t in GRID if t <= Tc]
    if Tc not in grid:
        grid.append(Tc)
    n = d["meta"]["n"]
    bs = batch_size(d, width)
    parts = []
    for a in range(0, n, bs):
        idx = np.arange(a, min(n, a + bs))
        parts.append(run_batch(ad, make_batch(d, idx, device), Tc, grid, probe, want_feat, d["meta"]["kind"]))
    out = {}
    for k in parts[0]:
        out[k] = parts[0][k] + sum(p[k] for p in parts[1:]) if k == "ece" else np.concatenate([p[k] for p in parts])
    out["grid"] = np.array(grid, np.int32)
    return out


def summarize(out, d):
    c = out["correct"]
    grid = out["grid"]
    B, Tc = c.shape
    ever = c.any(1)
    first = np.where(ever, c.argmax(1) + 1, 0)
    wrong = ~c
    last_wrong = np.where(wrong.any(1), Tc - wrong[:, ::-1].argmax(1), 0)
    stable = np.where(c[:, -1], last_wrong + 1, 0)
    acc = c[:, grid - 1].mean(0)
    rows = np.arange(B)
    mc_at = np.array([c[rows, np.maximum(out["best_idx"][:, i], 1) - 1].mean() for i in range(len(grid))])
    e = out["ece"]
    ece = [float(np.abs(e[i, :, 1] - e[i, :, 2]).sum() / max(e[i, :, 0].sum(), 1)) for i in range(len(grid))]
    s = {"n": int(B), "T_cap": int(Tc), "grid": grid.tolist(), "acc": acc.round(5).tolist(),
         "maxconf_acc": mc_at.round(5).tolist(),
         "nll_cell": out["nll"].astype(np.float32).mean(0).round(5).tolist(),
         "ece": np.round(ece, 5).tolist(),
         "minconf": out["minconf"].astype(np.float32).mean(0).round(4).tolist(),
         "hrms": out["hrms"].astype(np.float32).mean(0).round(4).tolist(),
         "end_state": {k: int((out["end_state"] == i).sum()) for i, k in enumerate(["fixed", "2cycle", "other", "nonfinite"])},
         "acc_at_cap_nonfinite_excluded": float(c[:, -1].mean()),
         "ever_correct": float(ever.mean()), "stably_correct": float((stable > 0).mean()),
         "first_correct_median": float(np.median(first[ever])) if ever.any() else None,
         "first_stable_median": float(np.median(stable[stable > 0])) if (stable > 0).any() else None,
         "kl_never_frac": {str(eps): float((out["kl_first"][:, j] == 0).mean()) for j, eps in enumerate(EPS)},
         "mc_patience": {"mean_iters": float(np.where(out["mc_stop"] > 0, out["mc_stop"], Tc).mean()),
                         "acc": float(c[rows, np.where(out["mc_stop"] > 0, out["mc_pick"], out["best_idx"][:, -1]) - 1].mean())}}
    if "check" in out:
        s["check_acc"] = out["check"].mean(0).round(5).tolist()
        s["knutson_acc"] = out["knutson"].mean(0).round(5).tolist()
        ex = c[:, grid - 1]
        s["check_vs_exact_disagree"] = int((out["check"] != ex).sum())
        s["check_minus_exact_pos"] = int((out["check"] & ~ex).sum())
    return s, first, stable


def fit_probe(feat, correct, grid, device, l2=1e-3, iters=200):
    """Logistic probe on pooled latent -> 'stably correct at this grid point' (training-size validation set)."""
    B, G, Fd = feat.shape
    c = correct
    wrong = ~c
    Tc = c.shape[1]
    last_wrong = np.where(wrong.any(1), Tc - wrong[:, ::-1].argmax(1), 0)
    stable = np.where(c[:, -1], last_wrong + 1, Tc + 1)
    lab = (np.array(grid)[None, :] >= stable[:, None]).astype(np.float32)
    X = torch.tensor(feat.reshape(-1, Fd), device=device)
    Y = torch.tensor(lab.reshape(-1), device=device)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    Xz = (X - mu) / sd
    w = torch.zeros(Fd, device=device, requires_grad=True)
    b0 = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xz @ w + b0, Y) + l2 * (w * w).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        acc = (((Xz @ w + b0) > 0).float() == Y).float().mean().item()
    return {"mu": mu, "sd": sd, "w": w.detach(), "b": b0.detach(), "train_acc": acc, "pos_frac": float(lab.mean())}


def evalsuite(ad, sets, out_dir, device, width, order, probe_set=None, repeat_check=None, tcap_override=None):
    """Evaluate `ad` on the named sets (in order). If probe_set is given it is evaluated first with
    pooled features, a stop probe is fitted, and later sets carry its probabilities."""
    os.makedirs(out_dir, exist_ok=True)
    summ = {"flops": None, "sets": {}}
    first_set = sets[order[0] if probe_set is None else probe_set]
    summ["flops"] = flop_check(ad, first_set, device)
    probe = None
    if probe_set is not None:
        t0 = time.time()
        out = eval_set(ad, sets[probe_set], device, width, want_feat=True)
        feat = out.pop("feat")
        probe = fit_probe(feat, out["correct"], out["grid"], device)
        summ["probe"] = {"set": probe_set, "train_acc": probe["train_acc"], "pos_frac": probe["pos_frac"]}
        s, first, stable = summarize(out, sets[probe_set])
        s["sec"] = round(time.time() - t0, 1)
        save_out(out, first, stable, os.path.join(out_dir, probe_set + ".npz"))
        summ["sets"][probe_set] = s
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in probe.items()},
                   os.path.join(out_dir, "probe.pt"))
        print(json.dumps({"set": probe_set, "acc_last": s["acc"][-1], "sec": s["sec"]}), flush=True)
    for name in order:
        if name == probe_set:
            continue
        t0 = time.time()
        tc = (tcap_override or {}).get(name)
        out = eval_set(ad, sets[name], device, width, probe=probe, Tc=tc)
        s, first, stable = summarize(out, sets[name])
        s["sec"] = round(time.time() - t0, 1)
        save_out(out, first, stable, os.path.join(out_dir, name + ".npz"))
        summ["sets"][name] = s
        print(json.dumps({"set": name, "grid_acc": dict(zip(s["grid"][-6:], s["acc"][-6:])), "sec": s["sec"]}), flush=True)
    if repeat_check:
        out2 = eval_set(ad, sets[repeat_check], device, width, probe=probe)
        a = np.load(os.path.join(out_dir, repeat_check + ".npz"))
        same = all(np.array_equal(a[k], out2[k]) for k in out2 if k in a.files)
        summ["bitwise_repeat"] = {"set": repeat_check, "identical": bool(same)}
    return summ


def save_out(out, first, stable, path):
    o = dict(out)
    c = o.pop("correct")
    o["correct_bits"] = np.packbits(c, axis=1)
    o["T_cap"] = np.int32(c.shape[1])
    o["first_correct"] = first.astype(np.int32)
    o["first_stable"] = stable.astype(np.int32)
    np.savez_compressed(path, **o)


def load_correct(npz):
    z = np.load(npz)
    return np.unpackbits(z["correct_bits"], axis=1, count=int(z["T_cap"])).astype(bool)


# ------------------------------------------------------------------ models
def load_baseline(arm, seed=None, device="cpu", ckpt_dir=None, which="final"):
    """Adapter for a P0 model: 'M-ref', a trained arm (e.g. 'M1', 'P1'), or 'MR-<arm>' market-resnet file."""
    from dtnet import DTRecall, load_mref
    from dtl import DTL1D
    from mr_adapter import DTAdapter, MRAdapter, load_mr
    ckpt_dir = ckpt_dir or os.path.join(HERE, "ckpt")
    if arm == "M-ref":
        path = os.environ.get("P0_MREF", os.path.join(HERE, "src/dt_net.pth"))
        net = load_mref(path, device)
        return DTAdapter(net, "maze"), 128
    if arm.startswith("MRfile:"):
        m, a = load_mr(arm[7:], device)
        return MRAdapter(m), m.w
    suffix = "" if which == "final" else "_best"
    s = torch.load(os.path.join(ckpt_dir, f"{arm}_s{seed}{suffix}.pt"), map_location=device, weights_only=False)
    ar = s["arch"]
    if ar["kind"] == "mr":
        from mresnet import MarketResNet
        m = MarketResNet(w=ar["width"]).to(device).eval()
        m.load_state_dict(s["model"])
        return MRAdapter(m), ar["width"]
    if ar["kind"] == "dtl":
        net = DTL1D(ar["width"])
    else:
        net = DTRecall(ar["width"], dim=ar["dim"], in_channels=3)
    net.load_state_dict(s["model"])
    net = net.to(device).eval()
    return DTAdapter(net, "maze" if ar.get("dim", 1) == 2 else "prefix"), ar["width"]
