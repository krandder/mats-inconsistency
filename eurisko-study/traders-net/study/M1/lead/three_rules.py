"""M1 retarget (c): the 'three rules' settlement for attention, on wealth-net checkpoints.

Rules: LayerNorm denominators detached (exchange rate); linear nodes Euler (grad x input);
S = q.k^T/sqrt(dh) split half/half; softmax as an LMSR sub-market: values paid at the prior
pi = 1/Q, routing credit to the scores by the IG path from S = 0 (uniform prior), so that
  sum_j a_j u_j = sum_j pi_j u_j  (values)  +  sum_j (a_j - pi_j) u_j  (routing, via S)
with u_j = v_j . dR/dhv. Every other op as in M1/w4/leak.py. Relevance of a tensor = t . g_t
under the modified gradient of R = sum_q z_q s_q / B (s = y - p detached), exactly as w4.
Reuses w4's model, streams and held-out split (same eval positions), so w4's exact ablation
applies unit for unit. Writes lead/results/three-s{seed}-step{step}.json.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

W4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4")
sys.path.insert(0, W4)
from model import Net  # noqa: E402
from streams import make_questions  # noqa: E402
from leak import ln_apply, rel, ln_in_rel, WN  # noqa: E402


class LMSRSoftmaxAV(torch.autograd.Function):
    """hv = softmax(S) @ v (live forward). Backward: v gets pi^T g (prior price), S gets the
    path-averaged softmax gradient from 0 to S (IG, midpoint rule with `steps` points)."""

    @staticmethod
    def forward(ctx, S, v, steps):
        A = torch.softmax(S, dim=-1)
        ctx.save_for_backward(S, v)
        ctx.steps = steps
        return A @ v

    @staticmethod
    def backward(ctx, g):
        S, v = ctx.saved_tensors
        n = S.shape[-1]
        u = g @ v.transpose(-1, -2)                       # u[i,j] = g_i . v_j
        gv = g.sum(-2, keepdim=True).expand_as(v) / n     # pi_ij = 1/n
        gS = torch.zeros_like(S)
        for t in (np.arange(ctx.steps) + 0.5) / ctx.steps:
            a = torch.softmax(t * S, dim=-1)
            gS += a * (u - (a * u).sum(-1, keepdim=True))
        return gS / ctx.steps, gv, None


def forward(m, win, rule):
    B, Q, H = win.shape[0], m.Q, m.H
    rec = {"cuts": [], "L": []}
    x = m.win(2.0 * win - 1.0) + m.qid.unsqueeze(0).expand(B, -1, -1)
    rec["cuts"].append(x)
    for blk in m.blocks:
        r = {"x_in": x}
        h1 = ln_apply(blk.ln1, x, True)
        q, k, v = blk.qkv(h1).view(B, Q, 3, H, blk.dh).permute(2, 0, 3, 1, 4).unbind(0)
        if rule["kind"] == "three":
            S = (0.5 * (q @ k.detach().transpose(-1, -2)) + 0.5 * (q.detach() @ k.transpose(-1, -2))) / math.sqrt(blk.dh)
            A = torch.softmax(S, -1).detach()
            hv = LMSRSoftmaxAV.apply(S, v, rule["steps"])
        else:  # cp: LN and pattern detached
            S = q @ k.transpose(-1, -2) / math.sqrt(blk.dh)
            A = torch.softmax(S, -1).detach()
            hv = A @ v
        x = x + blk.o(hv.transpose(1, 2).reshape(B, Q, m.d))
        r.update(h1=h1, q=q, k=k, v=v, S=S, A=A, hv=hv, x_mid=x)
        rec["cuts"].append(x)
        h2 = ln_apply(blk.ln2, x, True)
        act = F.relu(blk.fc1(h2))
        x = x + blk.fc2(act)
        r.update(h2=h2, act=act)
        rec["cuts"].append(x)
        rec["L"].append(r)
    yf = ln_apply(m.lnf, x, True)
    z = (yf * m.rw).sum(-1) + m.rb
    rec.update(yf=yf, x_fin=x, z=z)
    return z, rec


def books(m, win, y, rule):
    B = win.shape[0]
    z, rec = forward(m, win, rule)
    if rule.get("l0") is None:      # tangent settlement at the trade price
        s = (y - torch.sigmoid(z)).detach()
    else:                           # exact LMSR from the prior l0: s = y - mean price on the 16-step path
        l0 = rule["l0"]; zd = z.detach()
        pbar = sum(torch.sigmoid(l0 + t * (zd - l0)) for t in (np.arange(16) + 0.5) / 16) / 16
        s = (y - pbar).detach()
    R = (z * s).sum() / B
    names = ["q", "k", "v", "S", "hv", "act"]
    tens = list(rec["cuts"]) + [r[n] for r in rec["L"] for n in names]
    bp = m.bias_params()
    grads = torch.autograd.grad(R, tens + [b for _, b, _ in bp], allow_unused=True)
    nc = len(rec["cuts"])
    gc = grads[:nc]
    gl = grads[nc:nc + len(names) * m.L]
    gb = grads[nc + len(names) * m.L:]
    G = [dict(zip(names, gl[i * len(names):(i + 1) * len(names)])) for i in range(m.L)]
    Rv = float(R.detach())
    brel = [(i, rel(b, g)) for (_, b, i), g in zip(bp, gb)]
    resid = [Rv - rel(rec["cuts"][k], gc[k]) - sum(r for i, r in brel if i >= k) for k in range(nc)]
    # g already carries the 1/B of R, so per-unit sums are per-example means (no second /B; w4/leak.py divides twice)
    ph = lambda t, g: np.zeros(t.shape[1]) if g is None else (t.detach().double() * g.double()).sum((0, 2, 3)).numpy()  # per head [H]
    gain = None
    if rule.get("l0") is not None:
        lp = lambda zz: (y * F.logsigmoid(zz) + (1 - y) * F.logsigmoid(-zz)).sum() / B
        gain = float(lp(z.detach()) - lp(rule["l0"].expand_as(z)))
    out = dict(R=Rv, logscore_gain=gain, maker=None if gain is None else Rv - gain, resid=resid, head=[], head_value=[], head_route=[], head_q=[], head_k=[], neuron=[],
               route_neg_query_frac=[], sm_defect=[])
    for r, g in zip(rec["L"], G):
        out["head"].append(ph(r["hv"], g["hv"]).tolist())
        out["head_value"].append(ph(r["v"], g["v"]).tolist())
        out["head_route"].append(ph(r["S"], g["S"]).tolist())
        out["head_q"].append(ph(r["q"], g["q"]).tolist())
        out["head_k"].append(ph(r["k"], g["k"]).tolist())
        out["neuron"].append((r["act"].detach().double() * g["act"].double()).sum((0, 1)).numpy().tolist())
        out["sm_defect"].append(rel(r["hv"], g["hv"]) - rel(r["S"], g["S"]) - rel(r["v"], g["v"]))
        # per (example, head, query) routing income sign: share of queries where routing loses
        rq = torch.zeros(r["S"].shape[:-1]) if g["S"] is None else (r["S"].detach() * g["S"]).sum(-1)   # [B,H,Q]
        out["route_neg_query_frac"].append((rq < 0).double().mean((0, 2)).numpy().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", default="8000,16000,24000,30000")
    ap.add_argument("--nev", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--settle", default="tangent", choices=["tangent", "lmsr"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    RULES = {"three16": dict(kind="three", steps=16), "three64": dict(kind="three", steps=64), "cp": dict(kind="cp")}
    Q, W, T = 64, 32, 24000
    for seed in map(int, a.seeds.split(",")):
        bits, fam, desc = make_questions(Q, T, seed)
        bits_t = torch.tensor(bits, dtype=torch.float32)
        wins = bits_t.unfold(1, W, 1)
        n_pos = T - W; split = int(n_pos * 0.8)
        ev_idx = torch.arange(split, n_pos, max(1, (n_pos - split) // a.nev))   # same as w4/leak.py
        ev_win = wins[:, ev_idx].permute(1, 0, 2).contiguous(); ev_y = bits_t[:, ev_idx + W].T.contiguous()
        l0 = None
        if a.settle == "lmsr":   # prior = per-question base rate of the training segment, (n1+1)/(n+2)
            tr = bits_t[:, W:split + W]
            pr = (tr.sum(1) + 1) / (tr.shape[1] + 2)
            l0 = torch.log(pr / (1 - pr))
        for rn in RULES:
            RULES[rn]["l0"] = l0
        tag = "three" if a.settle == "tangent" else "three-lmsr"
        for step in map(int, a.steps.split(",")):
            fn = os.path.join(a.out, f"{tag}-s{seed}-step{step}.json")
            if os.path.exists(fn):
                continue
            m = Net(Q, W, 64, 4, 2)
            m.load_state_dict(torch.load(os.path.join(WN, f"train-s{seed}-step{step}.pt"), map_location="cpu"))
            res = {"seed": seed, "step": step, "settle": a.settle, "n_eval": int(ev_win.shape[0]), "rules": {}}
            for rn, rule in RULES.items():
                acc = None
                for c in range(0, ev_win.shape[0], a.chunk):
                    bk = books(m, ev_win[c:c + a.chunk], ev_y[c:c + a.chunk], rule)
                    wgt = ev_win[c:c + a.chunk].shape[0] / ev_win.shape[0]
                    if acc is None:
                        acc = {k: np.asarray(0.0 if v is None else v, dtype=float) * wgt for k, v in bk.items()}
                    else:
                        for k, v in bk.items():
                            acc[k] = acc[k] + np.asarray(0.0 if v is None else v, dtype=float) * wgt
                res["rules"][rn] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in acc.items()}
            json.dump(res, open(fn, "w"))
            r = res["rules"]
            print(f"s{seed} step{step} " + " ".join(f"{rn}:R={r[rn]['R']:.3f} resid0/|R|={r[rn]['resid'][0] / abs(r[rn]['R']):+.2e}" for rn in r), flush=True)


if __name__ == "__main__":
    main()
