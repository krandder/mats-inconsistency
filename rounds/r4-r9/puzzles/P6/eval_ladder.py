"""P6 test ladder. One JSONL line per (model, variant, cell); finished cells are skipped on rerun.

variants: auction (econ, variant a; PASS permanent, so also variant c = halting),
          single  (econ, richest trader alone to D=8192; for EMA-clone the top-EMA trader),
          plain   (R0 / DT-L).
cells: L in 32..8192 on the depth grid D ~ 2^(j/4) up to DCAP[L], plus path independence (pi) at
       L in {256, 1024}: n=64, 4 random initial states, D = 8L.
Stated deviations from the spec (GPU budget; the spec's full ladder is ~47G position-steps per plain model):
  n = 256 (L <= 512), 128 (1024-2048), 64 (4096), 32 (8192, spec 64);
  depth runs to 32768 only for L <= 256; DCAP = 16384 (512), 8192 (1024, 2048), 16384 (4096), 32768 (8192),
  so D4 = min(4L, 32768) is always on the grid and overthinking retention is read at DCAP[L];
  path independence at L in {256, 1024} (spec {256, 2048}).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, math, time
import torch
import p6_entry as P
from mresnet import prefix_batch

LS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DMAX = 32768


DCAP = {32: 32768, 64: 32768, 128: 32768, 256: 32768, 512: 16384, 1024: 8192, 2048: 8192, 4096: 16384, 8192: 32768}


def n_for(L):
    return 256 if L <= 512 else (128 if L <= 2048 else (64 if L <= 4096 else 32))


def grid(L, dmax):
    g = {max(4, 4 * round(2 ** (j / 4) / 4)) for j in range(8, 61)}
    g |= {min(L, DMAX), min(4 * L, DMAX)}
    return sorted(d for d in g if d <= dmax)


def test_set(L, n, dev):
    g = torch.Generator().manual_seed(777_000 + L)
    return prefix_batch(n, L, L, g, dev)


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    ns = argparse.Namespace(**ck["args"])
    if arm in ("R0", "DTL"):
        M = P.Plain(arm, 0, dev)
        M.net.load_state_dict(ck["net"])
    else:
        M = P.Econ(arm, 0, dev, ns)
        with torch.no_grad():
            for k in M.P:
                M.P[k].copy_(ck["P"][k].to(dev))
        M.wealth, M.ema, M.age, M.founder = list(ck["wealth"]), list(ck["ema"]), list(ck["age"]), list(ck["founder"])
        M.N = len(M.wealth)
    return arm, M


def richest(arm, M):
    key = M.ema if arm == "EMA" else M.wealth
    el = M.eligible()
    cand = [n for n in range(M.N) if el[n]] or list(range(M.N))
    return max(cand, key=lambda n: (key[n], -n))


def logits_at(arm, M, x, y, m, depths, variant, slot=None, h0=None, b0=None, want_h=False):
    if arm in ("R0", "DTL"):
        out = M.rollout(x, m, y, depths, h0=h0, want_h=want_h)
        info = M.last_info
        if want_h:
            info = dict(info, h={D: v[1] for D, v in out.items()})
        return {D: v[0] for D, v in out.items()}, info
    out, info = M.rollout(x, m, y, depths, variant=variant, slot=slot, h0=h0, b0=b0, want_h=want_h)
    if want_h:
        info = dict(info, h={D: torch.cat([v[0].unsqueeze(1), v[1]], 1) for D, v in out.items()})
    return {D: v[0] for D, v in out.items()}, info


def per_example(logit, y, m):
    ok = (((logit > 0).float() == y) | (m == 0)).all(1)
    nats = -P.score_bits(logit, y, m)
    return ok, nats


def eval_cell(arm, M, L, variant, dev, slot):
    n = n_for(L)
    dmax = min(DMAX, 8192 if variant == "single" else DMAX, DCAP.get(L, DMAX))
    depths = grid(L, dmax)
    if arm not in ("R0", "DTL"):
        depths = [d for d in depths if d % 4 == 0]
    x, y, m = test_set(L, n, dev)
    t0 = time.time()
    lg, info = logits_at(arm, M, x, y, m, depths, variant, slot)
    rows = []
    for D in depths:
        ok, nats = per_example(lg[D], y, m)
        berr = (((lg[D] > 0).float() != y).float() * m).sum() / m.sum()
        rows.append([D, int(ok.sum()), round(float(nats.mean()), 6), round(float(berr), 6)])
    d4 = min(4 * L, dmax, DMAX)
    k4 = [r for r in rows if r[0] == d4][0]
    rec = dict(cell=f"L{L}", L=L, n=n, variant=variant, grid=rows, D4=d4, exact4=k4[1] / n, nats4=k4[2],
               wilson_lo4=wilson_lo(k4[1], n), sec=round(time.time() - t0, 1),
               resid_final=float(info["resid"].mean()), resid_final_max=float(info["resid"].max()),
               fixed_at=info.get("fixed_at"))
    if arm not in ("R0", "DTL"):
        hs = info["halt_step"]
        halted = hs >= 0
        ok_final, _ = per_example(lg[depths[-1]], y, m)
        rec.update(frac_halted=float(halted.float().mean()),
                   halt_step_median=float(hs[halted].float().median()) if halted.any() else None,
                   halt_over_L_mean=float((hs[halted].float() / L).mean()) if halted.any() else None,
                   exact_given_halted=float(ok_final[halted].float().mean()) if halted.any() else None,
                   exact_given_running=float(ok_final[~halted].float().mean()) if (~halted).any() else None,
                   steps_applied_mean=float(info["applied"].mean()),
                   s_small_frac=info["s_small"] / max(1, info["trades"]), wins_by_band=info["wins"],
                   founder=list(M.founder))
    return rec


def eval_pi(arm, M, L, variant, dev, slot):
    n, D = 64, min(8 * L, DMAX)
    x, y, m = test_set(L, n, dev)
    g = torch.Generator().manual_seed(555_000 + L)
    preds, oks, nats, dist = [], [], [], []
    t0 = time.time()
    lg_ref, inf_ref = logits_at(arm, M, x, y, m, [D], variant, slot=slot, want_h=True)
    href = inf_ref["h"][D].flatten(1)
    for r in range(4):
        if arm in ("R0", "DTL"):
            h0 = torch.randn(n, P.W, L, generator=g).to(dev)
            lg, inf = logits_at(arm, M, x, y, m, [D], variant, h0=h0, want_h=True)
        else:
            h0 = torch.randn(n, P.H, L, generator=g).to(dev)
            b0 = (2.0 * torch.randn(n, L, generator=g)).to(dev)
            lg, inf = logits_at(arm, M, x, y, m, [D], variant, slot=slot, h0=h0, b0=b0, want_h=True)
        ok, nt = per_example(lg[D], y, m)
        hr = inf["h"][D].flatten(1)
        dist.append(float(((hr - href).norm(dim=1) / (href.norm(dim=1) + 1e-12)).mean()))
        preds.append(((lg[D] > 0) | (m == 0)).cpu()); oks.append(ok.float().mean().item()); nats.append(nt.mean().item())
    agree = torch.stack([(p == preds[0]).all(1) for p in preds[1:]]).all(0).float().mean().item()
    ref_pred = ((lg_ref[D] > 0) | (m == 0)).cpu()
    agree_ref = torch.stack([(p == ref_pred).all(1) for p in preds]).all(0).float().mean().item()
    return dict(cell=f"pi{L}", L=L, n=n, D=D, variant=variant, exact_by_init=oks, nats_by_init=nats,
                exact_mean=sum(oks) / 4, all4_agree=agree, agree_with_std_init=agree_ref,
                rel_state_dist=dist, exact_std_init=per_example(lg_ref[D], y, m)[0].float().mean().item(),
                sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default=None, choices=["auction", "single", "plain"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--Ls", default=",".join(map(str, LS)))
    ap.add_argument("--pi", default="256,1024")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = a.device
    arm, M = load_model(a.model, dev)
    variant = a.variant or ("plain" if arm in ("R0", "DTL") else "auction")
    base = os.path.splitext(os.path.basename(a.model))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.model))), "eval", f"{base}_{variant}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)["cell"])
            except Exception:
                pass
    slot = richest(arm, M) if (variant == "single" and arm not in ("R0", "DTL")) else None
    cells = [("L", int(L)) for L in a.Ls.split(",") if L] + [("pi", int(L)) for L in a.pi.split(",") if L]
    if variant == "single":
        cells = [c for c in cells if c[0] == "L" and c[1] <= 8192]
    for kind, L in cells:
        cid = f"L{L}" if kind == "L" else f"pi{L}"
        if cid in done:
            continue
        rec = eval_cell(arm, M, L, variant, dev, slot) if kind == "L" else eval_pi(arm, M, L, variant, dev, slot)
        rec.update(model=base, arm=arm, slot=slot, device=dev)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in rec if k != "grid"}), flush=True)
    open(out.replace(".jsonl", ".done"), "w").write("done\n")
    print("EVALDONE", base, variant, flush=True)


if __name__ == "__main__":
    main()
