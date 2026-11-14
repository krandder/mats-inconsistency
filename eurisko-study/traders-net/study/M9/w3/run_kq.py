"""M9(c) w3: python3 run_kq.py ARM K HW SEED  (ARM A|B). See PREREG.md. Harness: M6/w1/m6layer.py (unchanged)."""
import json, math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
M6W1 = os.path.normpath(os.path.join(HERE, "..", "..", "M6", "w1"))
sys.path.insert(0, M6W1)
from m6layer import MarketNet, TraderLayer, masked_bce, make_opts  # noqa: E402
import run_w1  # noqa: E402  (metrics, constraints; FAM/LEARN)
import streams_ext  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

Q, W, T, B = 64, 32, 24032, 4
OUT = os.environ.get("KQ_OUT", os.path.join(HERE, "results"))
TMP = os.environ.get("KQ_TMP", "/tmp/traders-net/M9/w3")


def main():
    arm, K, hw, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    torch.manual_seed(seed); np.random.seed(seed)
    s = streams_ext.make_streams(Q, T, seed, heldout=0, regime_every=0, W=W)
    st = streams_ext.to_array_stream(s)
    fam, cons = np.asarray(s.fam0), run_w1.constraints(s)
    model = MarketNet(Q, W, arm=arm, n_heads=K)
    model.traders = TraderLayer(Q, 64, n_heads=K, hw=hw)
    o, om = make_opts(model)
    n_layer = sum(p.numel() for p in model.traders.parameters())
    n_net = sum(p.numel() for p in model.net_params())
    # FLOPs per training step (fwd + bwd), measured once
    win, y, m = st.window(np.arange(B)), st.label(np.arange(B)), st.label_mask(np.arange(B)).float()
    fc = FlopCounterMode(display=False)
    with fc:
        out = model(win); loss = masked_bce(out["z"], y, m) + masked_bce(out["l0"], y, m); loss.backward()
    flops_step = int(fc.get_total_flops()); model.zero_grad(set_to_none=True)
    fl = FlopCounterMode(display=False)
    with fl, torch.no_grad():
        model.traders(out["y"].detach(), out["l0"].detach())
    flops_layer_fwd = int(fl.get_total_flops())
    t0 = time.time()
    n_pos = st.n_pos
    Z = np.zeros((n_pos, Q), np.float32); Y = np.zeros_like(Z); M = np.zeros((n_pos, Q), bool)
    absd = np.zeros((Q, K)); nd = 0
    trained = np.full(n_pos, -1)
    for step in range(math.ceil(n_pos / B)):
        P = np.arange(step * B, min((step + 1) * B, n_pos))
        win, y, m = st.window(P), st.label(P), st.label_mask(P)
        model.eval()
        with torch.no_grad():
            out = model(win)
        assert np.all(trained[P] < 0), "prequential violation"
        Z[P], Y[P], M[P] = out["z"].numpy(), y.numpy(), m.numpy()
        if P[0] >= n_pos // 2:
            absd += out["d"].abs().sum(0).numpy(); nd += len(P)
        trained[P] = step
        model.train()
        out = model(win); mf = m.float()
        loss = masked_bce(out["z"], y, mf) + masked_bce(out["l0"], y, mf)
        o.zero_grad(set_to_none=True); om.zero_grad(set_to_none=True)
        loss.backward(); o.step(); om.step()
    sl = slice(n_pos // 2, n_pos)
    res = {"arm": arm, "K": K, "hw": hw, "seed": seed, "Q": Q, "T": T, "B": B, "steps": math.ceil(n_pos / B),
           "n_layer_params": n_layer, "n_net_params": n_net, "flops_step": flops_step,
           "flops_layer_fwd": flops_layer_fwd}
    res["score"] = run_w1.metrics(Z[sl], Y[sl], M[sl], fam, cons)
    # per-question Brier on the scored window
    p = 1 / (1 + np.exp(-Z.astype(np.float64)))
    se = (p - Y) ** 2
    mq = M[sl]
    res["q_brier"] = [float(se[sl][:, q][mq[:, q]].mean()) for q in range(Q)]
    res["q_fam"] = [run_w1.FAM[int(f)] for f in fam]
    # rolling Brier (learnable families) -> FLOPs to 0.1
    L = np.isin(fam, [run_w1.FAM.index(f) for f in run_w1.LEARN])
    per_pos = (se[:, L] * M[:, L]).sum(1) / np.maximum(M[:, L].sum(1), 1)
    w = 500; cs = np.cumsum(np.insert(per_pos, 0, 0)); roll = (cs[w:] - cs[:-w]) / w
    hit = np.nonzero(roll < 0.1)[0]
    res["pos_to_brier01"] = int(hit[0] + w) if len(hit) else None
    res["flops_to_brier01"] = (res["pos_to_brier01"] / B * flops_step) if len(hit) else None
    res["rolling_brier_final"] = float(roll[-1])
    res["fam_pos_to_brier01"] = {}
    for f in run_w1.FAM:
        Lf = fam == run_w1.FAM.index(f)
        pp = (se[:, Lf] * M[:, Lf]).sum(1) / np.maximum(M[:, Lf].sum(1), 1)
        c2 = np.cumsum(np.insert(pp, 0, 0)); r2 = (c2[w:] - c2[:-w]) / w
        h2 = np.nonzero(r2 < 0.1)[0]
        res["fam_pos_to_brier01"][f] = int(h2[0] + w) if len(h2) else None
    a = absd / max(nd, 1)
    pr = (a.sum(1) ** 2) / np.maximum((a ** 2).sum(1), 1e-12)
    res["eff_traders_per_q"] = float(pr.mean())
    res["wall_s"] = time.time() - t0
    tag = f"{arm}-K{K}-hw{hw}-s{seed}"
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    np.save(f"{TMP}/{tag}-roll.npy", roll[::50].astype(np.float32))
    print(tag, "done", round(res["wall_s"]), round(res["score"]["learn_bce"], 4), res["pos_to_brier01"], flush=True)


if __name__ == "__main__":
    main()
