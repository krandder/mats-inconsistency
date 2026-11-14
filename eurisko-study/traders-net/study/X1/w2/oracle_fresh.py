"""X1 w2 R2: copy of M2/w1/oracle.py patched only in eval segment (OFF=8704), coalition RNG (+9001), zero ablation only.
Run: PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 oracle_fresh.py SEED --steps final --threads 1 --out /tmp/traders-net/X1w2/M2/w1/results_fresh"""
"""M2 ablation oracle: exact held-out Δloss of every neuron, head and readout, plus coalitions.

Read-only on wealth-net (weights, streams, model, snapshots, ablation.csv). For each seed and
checkpoint (8k, 20k, 30k, final) and each unit, replace the unit by 0 (zero) or by its mean over
1024 training positions (mean), and measure BCE/Brier on wealth-net ablate.py's fresh segment
(4096 positions after everything used in training). Δ = ablated − intact (positive: unit helps).
Then 200 random and 50 structured coalitions (see PREREG.md).

Outputs (under --out, default ./results):
  oracle_units_s{S}_step{K}.csv       one row per unit (all 4096 positions; A/B halves)
  oracle_coalitions_s{S}_step{K}.csv  one row per coalition (half A = even positions; singles on half A for the gap)
  oracle_perq_s{S}_step{K}.npz  per-question Δ for every unit ([U,Q], zero/mean) and unit ids
  oracle_checks_s{S}_step{K}.json     intact-vs-ablation.csv check, fast-vs-generic check, split-half rho
Usage: python3 oracle.py SEED [--threads 3] [--E 4096] [--steps 8000,20000,30000,final] [--out DIR]
"""
import argparse, csv, glob, json, math, os, re, sys, time
WN = "/home/agent/workspaces/eurisko-study/wealth-net"
sys.path.insert(0, WN)
sys.path.insert(0, os.path.join(WN, ".pylib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from streams import make_questions, FAMILIES  # noqa: E402
from model import Net  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RES = os.path.join(WN, "results")
MODES = ("zero",)


# ----------------------------------------------------------------------------- model pieces
def attn_hv(m, blk, x):
    B = x.shape[0]
    h = blk.ln1(x)
    q, k, v = blk.qkv(h).view(B, m.Q, 3, m.H, blk.dh).permute(2, 0, 3, 1, 4)
    A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
    return A @ v                                                 # [B,H,Q,dh]


def attn_out(m, blk, hv):
    B = hv.shape[0]
    return blk.o(hv.transpose(1, 2).reshape(B, m.Q, m.d))


def mlp_act(blk, x):
    return F.relu(blk.fc1(blk.ln2(x)))                           # [B,Q,N]


def readout(m, x):
    return (m.lnf(x) * m.rw).sum(-1) + m.rb                      # [B,Q]


def embed(m, win):
    return m.win(2.0 * win - 1.0) + m.qid.unsqueeze(0)


def forward_spec(m, win, spec, repl):
    """Generic forward with a coalition removed. spec: dict with bool tensors
    'neuron' [L,N], 'head' [L,H], 'readout' [Q]. repl: replacement dict (zeros or means)."""
    x = embed(m, win)
    for l, blk in enumerate(m.blocks):
        hv = attn_hv(m, blk, x)
        hm = spec["head"][l]
        if hm.any():
            hv = torch.where(hm.view(1, -1, 1, 1), repl["hv"][l].unsqueeze(0), hv)
        x = x + attn_out(m, blk, hv)
        act = mlp_act(blk, x)
        nm = spec["neuron"][l]
        if nm.any():
            act = torch.where(nm.view(1, 1, -1), repl["act"][l].unsqueeze(0), act)
        x = x + blk.fc2(act)
    z = readout(m, x)
    rm = spec["readout"]
    if rm.any():
        z = torch.where(rm.view(1, -1), repl["z"].unsqueeze(0), z)
    return z


def empty_spec(m):
    return {"neuron": torch.zeros(m.L, 4 * m.d, dtype=torch.bool), "head": torch.zeros(m.L, m.H, dtype=torch.bool),
            "readout": torch.zeros(m.Q, dtype=torch.bool)}


# ----------------------------------------------------------------------------- scoring
class Acc:
    """Per-question sums of BCE and Brier over halves A (even positions) and B (odd)."""

    def __init__(self, Q):
        self.bce = torch.zeros(2, Q, dtype=torch.float64)
        self.br = torch.zeros(2, Q, dtype=torch.float64)

    def add(self, z, y, par):
        # z,y: [B,Q]; par: [B] 0/1 half id
        b = F.binary_cross_entropy_with_logits(z, y, reduction="none").double()
        r = ((torch.sigmoid(z) - y) ** 2).double()
        for h in (0, 1):
            s = par == h
            self.bce[h] += b[s].sum(0); self.br[h] += r[s].sum(0)


def unit_ids(m):
    L, N, H, Q = m.L, 4 * m.d, m.H, m.Q
    ids = [("neuron", l, i, l * N + i, f"N{l}.{i}") for l in range(L) for i in range(N)]
    ids += [("head", l, h, l * H + h, f"H{l}.{h}") for l in range(L) for h in range(H)]
    ids += [("readout", -1, q, q, f"R{q}") for q in range(Q)]
    return ids


def spec_from_units(m, units):
    sp = empty_spec(m)
    N, H = 4 * m.d, m.H
    for u in units:
        t, rest = u[0], u[1:]
        if t == "N":
            l, i = map(int, rest.split(".")); sp["neuron"][l, i] = True
        elif t == "H":
            l, h = map(int, rest.split(".")); sp["head"][l, h] = True
        else:
            sp["readout"][int(rest)] = True
    return sp


# ----------------------------------------------------------------------------- main
def run(seed, steps_req, E, out, threads, chunk, no_coal=False):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    meta = json.load(open(os.path.join(RES, f"train-s{seed}.json")))
    tag, Q, W, T = meta["tag"], meta["Q"], meta["W"], meta["T"]
    OFF = 8704
    bits, fam, _ = make_questions(Q, T + OFF + 4096, seed)
    b = torch.tensor(bits, dtype=torch.float32)
    wins = b.unfold(1, W, 1)
    ev = torch.arange(T + OFF - W, T + OFF + 4096 - W)[:E]
    ev_win, ev_y = wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous()
    par = (torch.arange(len(ev)) % 2)
    tr = torch.randint(0, int((T - W) * 0.8), (1024,), generator=torch.Generator().manual_seed(seed))
    tr_win = wins[:, tr].permute(1, 0, 2).contiguous()
    famt = torch.tensor(fam)
    nA, nB = int((par == 0).sum()), int((par == 1).sum())

    ck = {}
    for f in glob.glob(os.path.join(RES, "weights", f"{tag}-step*.pt")):
        ck[int(re.search(r"step(\d+)", f).group(1))] = f
    ck[meta["steps_done"]] = os.path.join(RES, f"{tag}.pt")
    steps = [meta["steps_done"] if s == "final" else int(s) for s in steps_req]
    abl_rows = list(csv.DictReader(open(os.path.join(RES, "ablation.csv"))))

    assert meta["L"] == 2, "stage-skipping coalition forward assumes 2 layers"
    ufields = ["seed", "step", "final", "gran", "layer", "idx", "flat", "unit", "base_bce",
               "dloss_zero", "dloss_mean", "dloss_zero_A", "dloss_zero_B", "dloss_mean_A", "dloss_mean_B",
               "dbrier_zero", "dbrier_mean"] + [f"dloss_zero_{f}" for f in FAMILIES] + [f"dloss_mean_{f}" for f in FAMILIES]
    cfields = ["seed", "step", "final", "cid", "kind", "label", "size", "n_neuron", "n_head", "n_readout", "members",
               "dloss_zero", "dloss_mean",
               "sum_single_zero", "sum_single_mean", "gap_zero", "gap_mean", "gap_rel_zero", "gap_rel_mean"]
    for step in steps:
        t0 = time.time()
        checks = {"seed": seed, "step": step, "E": len(ev), "E_coalitions": int((par == 0).sum())}
        uf = open(os.path.join(out, f"oracle_units_s{seed}_step{step}.csv.part"), "w", newline="")
        uw = csv.DictWriter(uf, fieldnames=ufields, restval="nan"); uw.writeheader()
        cf = open(os.path.join(out, f"oracle_coalitions_s{seed}_step{step}.csv.part"), "w", newline="")
        cw = csv.DictWriter(cf, fieldnames=cfields, restval="nan"); cw.writeheader()
        m = Net(Q, W, meta["d"], meta["H"], meta["L"])
        m.load_state_dict(torch.load(ck[step], map_location="cpu")); m.eval()
        L, N, H, dh = m.L, 4 * m.d, m.H, m.d // m.H
        final = int(step == meta["steps_done"])
        with torch.no_grad():
            # replacement values (means over training positions, per question)
            rec = {}
            ztr = m(tr_win, rec=rec)
            repl = {"mean": {"act": [a.mean(0) for a in rec["act"]],                    # [Q,N]
                             "hv": [h.mean(0) for h in rec["hv"]],                       # [H,Q,dh]
                             "z": ztr.mean(0)},                                           # [Q]
                    "zero": {"act": [torch.zeros(Q, N) for _ in range(L)],
                             "hv": [torch.zeros(H, Q, dh) for _ in range(L)],
                             "z": torch.zeros(Q)}}
            # intact pass + cache
            base = Acc(Q)
            cache = []
            for s in range(0, len(ev), chunk):
                w, y, p = ev_win[s:s + chunk], ev_y[s:s + chunk], par[s:s + chunk]
                x = embed(m, w)
                c = {}
                for l, blk in enumerate(m.blocks):
                    x = x + attn_out(m, blk, attn_hv(m, blk, x))
                    act = mlp_act(blk, x)
                    c[f"pre{l}"] = x                  # resid before MLP l
                    c[f"act{l}"] = act
                    x = x + blk.fc2(act)
                c["x_fin"] = x
                c["attn1"] = c["pre1"] - (c["pre0"] + m.blocks[0].fc2(c["act0"]))
                z = readout(m, x)
                c["z"] = z
                base.add(z, y, p)
                cache.append(c)
            base_bce = float(base.bce.sum() / (len(ev) * Q))
            ref = [r for r in abl_rows if int(r["seed"]) == seed and int(r["step"]) == step and float(r["frac"]) == 0
                   and r["granularity"] == "neuron" and r["criterion"] == "wealth"]
            ref_bce = float(ref[0]["bce"]) if ref else None
            ck_ = {"base_bce": base_bce, "ablation_csv_bce": ref_bce,
                   "match_1e-4": (abs(base_bce - ref_bce) < 1e-4) if (ref_bce is not None and len(ev) == 4096) else None}

            units = unit_ids(m)
            U = len(units)
            per = {mode: {"bce": torch.zeros(U, 2, Q, dtype=torch.float64), "br": torch.zeros(U, 2, Q, dtype=torch.float64)}
                   for mode in MODES}

            def put(mode, uidx, acc):
                per[mode]["bce"][uidx] = acc.bce; per[mode]["br"][uidx] = acc.br

            W2 = [blk.fc2.weight for blk in m.blocks]                      # [d,N]
            for mode in MODES:
                R = repl[mode]
                # layer-1 (last) neurons: x_fin' = x_fin - (act - r) W2[:, i]
                lastl = L - 1
                accs = [Acc(Q) for _ in range(N)]
                for ci, (s, c) in enumerate(zip(range(0, len(ev), chunk), cache)):
                    y, p = ev_y[s:s + chunk], par[s:s + chunk]
                    dact = c[f"act{lastl}"] - R["act"][lastl].unsqueeze(0)                # [B,Q,N]
                    for g0 in range(0, N, 32):
                        idx = torch.arange(g0, min(N, g0 + 32))
                        xp = c["x_fin"].unsqueeze(0) - dact[..., idx].permute(2, 0, 1).unsqueeze(-1) * W2[lastl][:, idx].T.view(-1, 1, 1, m.d)
                        zp = readout(m, xp.reshape(-1, Q, m.d)).view(len(idx), -1, Q)
                        for j, i in enumerate(idx.tolist()):
                            accs[i].add(zp[j], y, p)
                for i in range(N):
                    put(mode, lastl * N + i, accs[i])
                # earlier-layer neurons: resid after MLP l changes by -(act - r) W2[:, i]; rerun later layers
                for l in range(L - 1):
                    accs = [Acc(Q) for _ in range(N)]
                    G = 16
                    for s, c in zip(range(0, len(ev), chunk), cache):
                        y, p = ev_y[s:s + chunk], par[s:s + chunk]
                        x_after = c[f"pre{l}"] + m.blocks[l].fc2(c[f"act{l}"])
                        dact = c[f"act{l}"] - R["act"][l].unsqueeze(0)
                        for g0 in range(0, N, G):
                            idx = torch.arange(g0, min(N, g0 + G))
                            xp = x_after.unsqueeze(0) - dact[..., idx].permute(2, 0, 1).unsqueeze(-1) * W2[l][:, idx].T.view(-1, 1, 1, m.d)
                            xp = xp.reshape(-1, Q, m.d)
                            for l2 in range(l + 1, L):
                                blk = m.blocks[l2]
                                xp = xp + attn_out(m, blk, attn_hv(m, blk, xp))
                                xp = xp + blk.fc2(mlp_act(blk, xp))
                            zp = readout(m, xp).view(len(idx), -1, Q)
                            for j, i in enumerate(idx.tolist()):
                                accs[i].add(zp[j], y, p)
                    for i in range(N):
                        put(mode, l * N + i, accs[i])
                # heads: generic forward
                for l in range(L):
                    for h in range(H):
                        sp = empty_spec(m); sp["head"][l, h] = True
                        a = Acc(Q)
                        for s in range(0, len(ev), chunk):
                            a.add(forward_spec(m, ev_win[s:s + chunk], sp, R), ev_y[s:s + chunk], par[s:s + chunk])
                        put(mode, L * N + l * H + h, a)
                # readouts: replace z_q only
                for q in range(Q):
                    a = Acc(Q)
                    for s, c in zip(range(0, len(ev), chunk), cache):
                        z = c["z"].clone(); z[:, q] = R["z"][q]
                        a.add(z, ev_y[s:s + chunk], par[s:s + chunk])
                    put(mode, L * N + L * H + q, a)
                print(f"s{seed} step{step} {mode} singles done {time.time() - t0:.0f}s", flush=True)

            # check 2: fast paths vs generic forward on random units
            rng = np.random.default_rng(seed * 1000 + step + 9001)
            fast_err = 0.0
            for mode in MODES:
                for uidx in list(rng.choice(L * N, 8, replace=False)) + [L * N + L * H + int(rng.integers(Q))]:
                    sp = spec_from_units(m, [units[uidx][4]])
                    a = Acc(Q)
                    for s in range(0, len(ev), chunk):
                        a.add(forward_spec(m, ev_win[s:s + chunk], sp, repl[mode]), ev_y[s:s + chunk], par[s:s + chunk])
                    fast = per[mode]["bce"][uidx].sum() / (len(ev) * Q); gen = a.bce.sum() / (len(ev) * Q)
                    fast_err = max(fast_err, float(abs(fast - gen)))
            ck_["fast_vs_generic_max_abs"] = fast_err
            ck_["fast_ok_1e-5"] = fast_err < 1e-5

            # write singles
            bq = base.bce.sum(0) / len(ev)                                      # [Q] intact
            bqh = base.bce / torch.tensor([nA, nB]).view(2, 1).double()
            brq = base.br.sum(0) / len(ev)
            dl = {}
            for mode in MODES:
                pq = per[mode]["bce"].sum(1) / len(ev) - bq                     # [U,Q]
                ph = per[mode]["bce"] / torch.tensor([nA, nB]).view(1, 2, 1).double() - bqh   # [U,2,Q]
                pbr = per[mode]["br"].sum(1) / len(ev) - brq
                dl[mode] = {"q": pq, "tot": pq.mean(1), "A": ph[:, 0].mean(1), "B": ph[:, 1].mean(1), "br": pbr.mean(1),
                            "fam": torch.stack([pq[:, famt == k].mean(1) for k in range(len(FAMILIES))], 1)}
            for k, u in enumerate(units):
                row = {"seed": seed, "step": step, "final": final, "gran": u[0], "layer": u[1], "idx": u[2], "flat": u[3],
                       "unit": u[4], "base_bce": round(base_bce, 7)}
                for mode in MODES:
                    row[f"dloss_{mode}"] = float(dl[mode]["tot"][k]); row[f"dloss_{mode}_A"] = float(dl[mode]["A"][k])
                    row[f"dloss_{mode}_B"] = float(dl[mode]["B"][k]); row[f"dbrier_{mode}"] = float(dl[mode]["br"][k])
                    for fi, fn in enumerate(FAMILIES):
                        row[f"dloss_{mode}_{fn}"] = float(dl[mode]["fam"][k, fi])
                uw.writerow(row)
            np.savez_compressed(os.path.join(out, f"oracle_perq_s{seed}_step{step}.npz"),
                                units=np.array([u[4] for u in units]), fam=fam,
                                dloss_zero=dl["zero"]["q"].float().numpy(), dloss_mean=dl["zero"]["q"].float().numpy(),
                                base_bce_q=bq.float().numpy())
            nz = dl["zero"]; nm_ = dl["zero"]
            sl = slice(0, L * N)
            sh = {}
            for mode in MODES:
                sh[mode] = {"neuron": float(spearmanr(dl[mode]["A"][sl].numpy(), dl[mode]["B"][sl].numpy())[0]),
                            "neuron_L0": float(spearmanr(dl[mode]["A"][:N].numpy(), dl[mode]["B"][:N].numpy())[0]),
                            "neuron_L1": float(spearmanr(dl[mode]["A"][N:2 * N].numpy(), dl[mode]["B"][N:2 * N].numpy())[0]),
                            "head": float(spearmanr(dl[mode]["A"][L * N:L * N + L * H].numpy(), dl[mode]["B"][L * N:L * N + L * H].numpy())[0]),
                            "readout": float(spearmanr(dl[mode]["A"][L * N + L * H:].numpy(), dl[mode]["B"][L * N + L * H:].numpy())[0])}
            ck_["split_half_spearman"] = sh
            ck_["zero_vs_mean_spearman"] = {g: float(spearmanr(nz["tot"][a:b_].numpy(), nm_["tot"][a:b_].numpy())[0])
                                            for g, a, b_ in (("neuron", 0, L * N), ("head", L * N, L * N + L * H), ("readout", L * N + L * H, U))}

            # ------------------------------------------------------------ coalitions
            singleA = {mode: {u[4]: float(dl[mode]["A"][k]) for k, u in enumerate(units)} for mode in MODES}
            neur = [u[4] for u in units if u[0] == "neuron"]
            heads = [u[4] for u in units if u[0] == "head"]
            coals = []
            for _ in range(100):
                k = int(round(math.exp(rng.uniform(math.log(2), math.log(64)))))
                coals.append(("random", "neurons", list(rng.choice(neur, k, replace=False))))
            for _ in range(40):
                k = int(rng.integers(2, 7))
                coals.append(("random", "heads", list(rng.choice(heads, k, replace=False))))
            for _ in range(30):
                l = int(rng.integers(L)); k = int(round(math.exp(rng.uniform(math.log(2), math.log(32)))))
                coals.append(("random", f"neurons_L{l}", [f"N{l}.{i}" for i in rng.choice(N, k, replace=False)]))
            for _ in range(30):
                kn = int(rng.integers(2, 21)); kh = int(rng.integers(1, 4))
                coals.append(("random", "mixed", list(rng.choice(neur, kn, replace=False)) + list(rng.choice(heads, kh, replace=False))))
            zt = nz["tot"][:L * N].numpy()
            order = np.argsort(-zt)
            for k in (2, 4, 8, 16, 32):
                coals.append(("structured", f"top{k}_neurons", [neur[i] for i in order[:k]]))
            top6 = [neur[i] for i in order[:6]]
            for a in range(6):
                for b_ in range(a + 1, 6):
                    coals.append(("structured", f"pair_top6_{a}{b_}", [top6[a], top6[b_]]))
            for l in range(L):
                coals.append(("structured", f"all_neurons_L{l}", [f"N{l}.{i}" for i in range(N)]))
                coals.append(("structured", f"heads_L{l}", [f"H{l}.{h}" for h in range(H)]))
            coals.append(("structured", "all_heads", heads))
            for fi, fn in enumerate(FAMILIES):
                fz = nz["fam"][:L * N, fi].numpy()
                coals.append(("structured", f"top8_{fn}", [neur[i] for i in np.argsort(-fz)[:8]]))
            for k in (8, 32):
                coals.append(("structured", f"bottom{k}_neurons", [neur[i] for i in order[::-1][:k]]))
            for l in range(L):
                zl = zt[l * N:(l + 1) * N]; ol = np.argsort(-zl)
                for k in (8, 16):
                    coals.append(("structured", f"top{k}_L{l}", [f"N{l}.{i}" for i in ol[:k]]))
            snf = os.path.join(RES, tag, "snapshots", f"step-{step:05d}.json")
            if os.path.exists(snf):
                wealth = np.array(json.load(open(snf))["wealth"]["neuron"]).ravel()
                wo = np.argsort(wealth, kind="stable")
                for k in (5, 26):
                    coals.append(("structured", f"poorest{k}_wealth", [neur[i] for i in wo[:k]]))
            for fi, fn in enumerate(FAMILIES):
                coals.append(("structured", f"readouts_{fn}", [f"R{q}" for q in range(Q) if fam[q] == fi]))
            ho = np.argsort(-nz["tot"][L * N:L * N + L * H].numpy())
            coals.append(("structured", "top2_heads", [heads[i] for i in ho[:2]]))
            def coal_eval(sp, R, generic=False):
                a = Acc(Q)
                for s, c in zip(range(0, len(ev), chunk), cache):
                    keep = par[s:s + chunk] == 0
                    y, p = ev_y[s:s + chunk][keep], par[s:s + chunk][keep]
                    if generic or sp["head"][0].any():
                        z = forward_spec(m, ev_win[s:s + chunk][keep], sp, R)
                    else:
                        b0, b1 = m.blocks
                        act0 = c["act0"][keep]
                        ch0 = bool(sp["neuron"][0].any())
                        if ch0:
                            act0 = torch.where(sp["neuron"][0].view(1, 1, -1), R["act"][0].unsqueeze(0), act0)
                        x2 = c["pre0"][keep] + b0.fc2(act0)
                        if ch0 or sp["head"][1].any():
                            hv = attn_hv(m, b1, x2)
                            if sp["head"][1].any():
                                hv = torch.where(sp["head"][1].view(1, -1, 1, 1), R["hv"][1].unsqueeze(0), hv)
                            x3 = x2 + attn_out(m, b1, hv)
                            act1 = mlp_act(b1, x3)
                        else:
                            x3 = x2 + c["attn1"][keep]
                            act1 = c["act1"][keep]
                        if sp["neuron"][1].any():
                            act1 = torch.where(sp["neuron"][1].view(1, 1, -1), R["act"][1].unsqueeze(0), act1)
                        z = readout(m, x3 + b1.fc2(act1))
                        if sp["readout"].any():
                            z = torch.where(sp["readout"].view(1, -1), R["z"].unsqueeze(0), z)
                    a.add(z, y, p)
                return float(a.bce[0].sum() / (nA * Q))

            baseA = float(base.bce[0].sum() / (nA * Q))
            cerr = 0.0
            for j in ((0, 105, 140, 170) if not no_coal else ()):                               # check: stage skipping == generic forward
                sp = spec_from_units(m, coals[j][2])
                cerr = max(cerr, abs(coal_eval(sp, repl["zero"]) - coal_eval(sp, repl["zero"], generic=True)))
            ck_["coalition_fast_vs_generic_max_abs"] = cerr
            if no_coal:
                coals = []
            for cid, (kind, label, mem) in enumerate(coals):
                sp = spec_from_units(m, mem)
                row = {"seed": seed, "step": step, "final": final, "cid": cid, "kind": kind, "label": label, "size": len(mem),
                       "n_neuron": sum(x[0] == "N" for x in mem), "n_head": sum(x[0] == "H" for x in mem),
                       "n_readout": sum(x[0] == "R" for x in mem), "members": ";".join(mem)}
                for mode in MODES:
                    d = coal_eval(sp, repl[mode]) - baseA              # half A only (even positions)
                    ss = sum(singleA[mode][x] for x in mem)            # singles on the same half
                    row.update({f"dloss_{mode}": d, f"sum_single_{mode}": ss,
                                f"gap_{mode}": d - ss, f"gap_rel_{mode}": (d - ss) / max(abs(d), abs(ss), 1e-9)})
                cw.writerow(row)
            ck_["n_coalitions"] = len(coals)
            ck_["seconds"] = time.time() - t0
            checks.update(ck_)
            uf.close(); cf.close()
            for f in (uf.name, cf.name):
                os.replace(f, f[:-5])
            json.dump(checks, open(os.path.join(out, f"oracle_checks_s{seed}_step{step}.json"), "w"), indent=1)
            print(f"s{seed} step{step} done {time.time() - t0:.0f}s {json.dumps(ck_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--E", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--steps", default="final,20000,30000,8000")
    ap.add_argument("--no-coalitions", action="store_true", help="singles only (PREREG amendment 1)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a.seed, a.steps.split(","), a.E, a.out, a.threads, a.chunk, a.no_coalitions)
