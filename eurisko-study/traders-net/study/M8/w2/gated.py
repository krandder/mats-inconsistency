"""M8 worker 2: |a.g|-gated MLP neurons with rebirth vs controls, on the S0 regime-shift stream.
See PREREG.md. Usage: python3 gated.py ARM SEED [T]  -> results/run_{ARM}_s{SEED}.json (+ .npz in /tmp)
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
torch.set_num_threads(1)
from streams_ext import make_streams, to_array_stream  # noqa: E402
from model import Net  # noqa: E402

ARMS = ["absag", "taylor", "movement", "magnitude", "cbp", "random", "static128", "dense256"]
K, R, MATURE, YOUNG, NREB, HL = 128, 100, 500, 200, 2, 500
LAM = 0.5 ** (1 / HL)
B, LR = 4, 1e-3


def run(arm, seed, T=40000):
    assert arm in ARMS
    t0 = time.time()
    s = make_streams(Q=64, T=T, seed=seed, heldout=8, regime_every=10000, regime_frac=0.25)
    st = to_array_stream(s)
    torch.manual_seed(seed)
    net = Net(Q=st.Q, W=st.W, d=64, H=4, L=2, mlp=4)
    L, N = net.L, 4 * net.d
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(10_000 + seed)
    rng = np.random.default_rng(20_000 + seed)
    gated = arm != "dense256"
    active = torch.zeros(L, N)
    if arm == "random":
        for l in range(L):
            active[l, torch.randperm(N, generator=g)[:K]] = 1
    else:
        active[:, :K] = 1
    if not gated:
        active[:] = 1
    rebirth = arm not in ("static128", "dense256")
    score = torch.zeros(L, N)
    age = torch.full((L, N), MATURE, dtype=torch.long)   # all mature at start
    ones_h = torch.ones(L, net.H)
    n_pos = st.n_pos
    n_steps = math.ceil(n_pos / B)
    Z = np.full((n_pos, st.Q), np.nan, np.float32)
    Y = np.zeros((n_pos, st.Q), np.uint8)
    M = np.zeros((n_pos, st.Q), bool)
    births, ever_active_births = 0, 0
    born_flag = torch.zeros(L, N, dtype=torch.bool)       # reborn unit not yet active
    dead_trace = []
    zero_act = torch.zeros(L, N)                           # activation mass since last event
    for step in range(n_steps):
        lo, hi = step * B, min((step + 1) * B, n_pos)
        P = np.arange(lo, hi)
        win, y, m = st.window(P), st.label(P), st.label_mask(P)
        net.eval()
        with torch.no_grad():
            z = net(win, mask={"neuron": [active[l] for l in range(L)], "head": ones_h})
        Z[P] = z.numpy(); Y[P] = y.numpy().astype(np.uint8); M[P] = m.numpy()
        # train on the same block (delay 0: labels revealed right after the prediction)
        net.train()
        mk = [active[l].view(1, 1, N).expand(len(P), st.Q, N).clone().requires_grad_(True) for l in range(L)]
        rec = {}
        opt.zero_grad(set_to_none=True)
        zt = net(win, rec=rec, mask={"neuron": mk, "head": ones_h})
        mf = m.float()
        loss = (F.binary_cross_entropy_with_logits(zt, y, reduction="none") * mf).sum() / mf.sum().clamp_min(1)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for l in range(L):
                G = mk[l].grad   # per-token a.g = dL/dmask (shadow credit for inactive units)
                blk = net.blocks[l]
                a = F.relu(blk.fc1(blk.ln2(rec["cuts"][2 * l + 1])))            # pre-mask activation
                zero_act[l] += (a * active[l]).abs().sum((0, 1))
                if arm == "absag":
                    inc = G.abs().sum((0, 1))
                elif arm == "taylor":
                    inc = G.sum((0, 1)).abs()
                elif arm == "movement":
                    inc = -G.sum((0, 1))
                elif arm == "cbp":
                    inc = a.abs().mean((0, 1)) * blk.fc2.weight.abs().sum(0)
                else:
                    inc = None
                if inc is not None:
                    score[l] = LAM * score[l] + (1 - LAM) * inc
                if arm == "magnitude":
                    score[l] = blk.fc1.weight.norm(dim=1) * blk.fc2.weight.norm(dim=0)
        age += 1
        if gated and (step + 1) % R == 0:
            dead_trace.append(float(((zero_act == 0) & (active > 0)).sum() / active.sum()))
            zero_act.zero_()
            with torch.no_grad():
                for l in range(L):
                    blk = net.blocks[l]
                    if rebirth:
                        mat = (age[l] >= MATURE).nonzero().flatten()
                        if arm == "random":
                            pick = mat[torch.randperm(len(mat), generator=g)[:NREB]]
                        else:
                            pick = mat[torch.argsort(score[l, mat])[:NREB]]
                        for i in pick.tolist():
                            bd = 1 / math.sqrt(blk.fc1.in_features)
                            blk.fc1.weight[i].uniform_(-bd, bd, generator=g); blk.fc1.bias[i] = 0
                            bd2 = 0.1 / math.sqrt(blk.fc2.in_features)
                            blk.fc2.weight[:, i].uniform_(-bd2, bd2, generator=g)
                            for p, sl in ((blk.fc1.weight, (i,)), (blk.fc1.bias, (i,)), (blk.fc2.weight, (slice(None), i))):
                                stt = opt.state.get(p, {})
                                for key in ("exp_avg", "exp_avg_sq"):
                                    if key in stt:
                                        stt[key][sl] = 0
                            age[l, i] = 0; births += 1; born_flag[l, i] = True
                            others = torch.ones(N, dtype=torch.bool); others[i] = False
                            score[l, i] = score[l, others].median()
                    if arm != "random":
                        young = (age[l] < YOUNG)
                        new = torch.zeros(N)
                        new[young] = 1
                        rest = K - int(young.sum())
                        cand = (~young).nonzero().flatten()
                        top = cand[torch.argsort(score[l, cand], descending=True)[:max(rest, 0)]]
                        new[top] = 1
                        active[l] = new
                    # random arm: static active set; reborn active units restart in place
                    # a reborn unit "survives" if it is still active when its guaranteed slot ends
                    done = born_flag[l] & (age[l] >= YOUNG)
                    ever_active_births += int((done & (active[l] > 0)).sum())
                    born_flag[l] &= ~done
    # metrics
    bce = (np.logaddexp(0, Z.astype(np.float64)) - Y * Z).astype(np.float64)
    W = st.W
    tpos = np.arange(n_pos) + W                     # target time of each position
    live = M & np.isfinite(Z)
    sw = [e for e in s.events if e["kind"] == "switch"]
    sw_mask = np.zeros_like(live); ns_mask = np.zeros_like(live)
    per_shift = {}
    for b0 in sorted({e["t"] for e in sw}):
        qs = [e["q"] for e in sw if e["t"] == b0]
        rows = (tpos >= b0) & (tpos < b0 + 2000)
        qm = np.zeros(st.Q, bool); qm[qs] = True
        a_ = live & rows[:, None] & qm[None, :]; b_ = live & rows[:, None] & (~qm)[None, :]
        per_shift[str(b0)] = {"switched": float(bce[a_].mean()), "other": float(bce[b_].mean())}
        sw_mask |= a_; ns_mask |= b_
    last = live & (tpos >= T - 10000)[:, None]
    out = {"arm": arm, "seed": seed, "T": T, "K": K if gated else N, "R": R, "HL": HL,
           "bce_all": float(bce[live].mean()),
           "bce_postshift_switched": float(bce[sw_mask].mean()),
           "bce_postshift_other": float(bce[ns_mask].mean()),
           "bce_last10k": float(bce[last].mean()),
           "per_shift": per_shift, "births": births, "births_survived": ever_active_births,
           "dead_frac_mean": float(np.mean(dead_trace)) if dead_trace else None,
           "dead_frac_last": dead_trace[-1] if dead_trace else None,
           "wall_s": time.time() - t0, "torch": torch.__version__}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"run_{arm}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=1)
    os.makedirs("/tmp/traders-net/M8w2", exist_ok=True)
    np.savez_compressed(f"/tmp/traders-net/M8w2/z_{arm}_s{seed}.npz", Z=Z, Y=Y, M=M)
    print(json.dumps({k: v for k, v in out.items() if k != "per_shift"}))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40000)
