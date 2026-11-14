"""M1(d): split the books' conservation residual by operation, and AttnLRP-rule books.

Rules (see PREREG.md): live, v1 (LN detached), cp (LN + attention detached), attnlrp
(LN detached + uniform 1/2 rule on Q.K^T and A.V; softmax = grad x input on the modified gradient).
For each rule: relevance of every recorded tensor t is sum(t * g_t) with g_t the (modified)
gradient of R = sum_q z_q * s_q / B, s = y - p detached. Local defect of an op = relevance out
- relevance in (constant inputs, i.e. biases and LN shifts, count as inputs). All other ops are
exactly conservative, so the defects downstream of cut k must add up to the cut-k residual.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net
from streams import make_questions

WN = "/home/agent/workspaces/eurisko-study/wealth-net/results/weights"
RULES = {"live": dict(dln=False, dattn=False, half=False),
         "v1": dict(dln=True, dattn=False, half=False),
         "cp": dict(dln=True, dattn=True, half=False),
         "attnlrp": dict(dln=True, dattn=False, half=True)}


def ln_apply(ln, x, detach):
    xc = x - x.mean(-1, keepdim=True)
    s = torch.sqrt(xc.pow(2).mean(-1, keepdim=True) + ln.eps)
    if detach:
        s = s.detach()
    return xc / s * ln.g + ln.b


def bil(a, b, half):
    if half:
        return 0.5 * (a @ b.detach()) + 0.5 * (a.detach() @ b)
    return a @ b


def forward(m, win, rule):
    B, Q, H = win.shape[0], m.Q, m.H
    rec = {"cuts": [], "L": []}
    x = m.win(2.0 * win - 1.0) + m.qid.unsqueeze(0).expand(B, -1, -1)
    rec["cuts"].append(x)
    for blk in m.blocks:
        r = {"x_in": x}
        h1 = ln_apply(blk.ln1, x, rule["dln"])
        q, k, v = blk.qkv(h1).view(B, Q, 3, H, blk.dh).permute(2, 0, 3, 1, 4).unbind(0)
        S = bil(q, k.transpose(-1, -2), rule["half"]) / math.sqrt(blk.dh)
        A = torch.softmax(S, dim=-1)
        Au = A.detach() if rule["dattn"] else A
        hv = bil(Au, v, rule["half"])
        x = x + blk.o(hv.transpose(1, 2).reshape(B, Q, m.d))
        r.update(h1=h1, q=q, k=k, v=v, S=S, A=A, hv=hv, x_mid=x)
        rec["cuts"].append(x)
        h2 = ln_apply(blk.ln2, x, rule["dln"])
        act = F.relu(blk.fc1(h2))
        x = x + blk.fc2(act)
        r.update(h2=h2, act=act)
        rec["cuts"].append(x)
        rec["L"].append(r)
    yf = ln_apply(m.lnf, x, rule["dln"])
    z = (yf * m.rw).sum(-1) + m.rb
    rec.update(yf=yf, x_fin=x, z=z)
    return z, rec


def rel(t, g):
    return 0.0 if g is None else float((t.detach().double() * g.double()).sum())


def ln_in_rel(ln, x, g_out, detach):
    xr = x.detach().requires_grad_(True)
    (gi,) = torch.autograd.grad(ln_apply(ln, xr, detach), xr, g_out)
    return rel(xr, gi)


def books(m, win, y, rule):
    B = win.shape[0]
    z, rec = forward(m, win, rule)
    s = (y - torch.sigmoid(z)).detach()
    R = (z * s).sum() / B
    names = ["q", "k", "v", "S", "A", "hv", "h1", "h2", "act"]
    tens = list(rec["cuts"]) + [r[n] for r in rec["L"] for n in names] + [rec["yf"]]
    bp = m.bias_params()
    grads = torch.autograd.grad(R, tens + [b for _, b, _ in bp], allow_unused=True)
    nc = len(rec["cuts"])
    gc = grads[:nc]
    gl = grads[nc:nc + len(names) * m.L]
    g_yf = grads[nc + len(names) * m.L]
    gb = grads[nc + len(names) * m.L + 1:]
    G = [dict(zip(names, gl[i * len(names):(i + 1) * len(names)])) for i in range(m.L)]
    Rv = float(R.detach())
    brel = [(i, rel(b, g)) for (_, b, i), g in zip(bp, gb)]
    resid = [Rv - rel(rec["cuts"][k], gc[k]) - sum(r for i, r in brel if i >= k) for k in range(nc)]
    # local defects: ops listed with the cut index they sit after
    ops = []
    for l, (r, g) in enumerate(zip(rec["L"], G)):
        blk = m.blocks[l]
        d_ln1 = rel(r["h1"] - blk.ln1.b, g["h1"]) - ln_in_rel(blk.ln1, r["x_in"], g["h1"], rule["dln"])
        d_qk = rel(r["S"], g["S"]) - rel(r["q"], g["q"]) - rel(r["k"], g["k"])
        d_sm = rel(r["A"], g["A"]) - rel(r["S"], g["S"])
        d_av = rel(r["hv"], g["hv"]) - rel(r["A"], g["A"]) - rel(r["v"], g["v"])
        d_ln2 = rel(r["h2"] - blk.ln2.b, g["h2"]) - ln_in_rel(blk.ln2, r["x_mid"], g["h2"], rule["dln"])
        ops += [(f"L{l}.ln1", 2 * l, d_ln1), (f"L{l}.QK", 2 * l, d_qk), (f"L{l}.softmax", 2 * l, d_sm),
                (f"L{l}.AV", 2 * l, d_av), (f"L{l}.ln2", 2 * l + 1, d_ln2)]
    d_lnf = rel(rec["yf"] - m.lnf.b, g_yf) - ln_in_rel(m.lnf, rec["x_fin"], g_yf, rule["dln"])
    ops.append(("lnf", 2 * m.L, d_lnf))
    closure = [resid[k] - sum(d for _, i, d in ops if i >= k) for k in range(nc)]
    head = np.stack([(r["hv"].detach() * g["hv"]).sum((0, 2, 3)).numpy() / B for r, g in zip(rec["L"], G)])
    neuron = np.stack([(r["act"].detach() * g["act"]).sum((0, 1)).numpy() / B for r, g in zip(rec["L"], G)])
    return dict(R=Rv, resid=resid, ops={n: d for n, _, d in ops}, closure=closure, head=head, neuron=neuron)


def _block_from(m, x, l, stage):
    """Run the model from residual x entering block l at stage 'attn' or 'mlp'; return logits."""
    B, Q = x.shape[0], m.Q
    for li in range(l, m.L):
        blk = m.blocks[li]
        if not (li == l and stage == "mlp"):
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).view(B, Q, 3, m.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            x = x + blk.o((A @ v).transpose(1, 2).reshape(B, Q, m.d))
        x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
    return (m.lnf(x) * m.rw).sum(-1) + m.rb


def ablation(m, win, y):
    """Exact zero-ablation Delta BCE of every head and neuron, re-running only downstream layers."""
    bce = lambda z: float(F.binary_cross_entropy_with_logits(z, y))
    L, H, N = m.L, m.H, 4 * m.d
    with torch.no_grad():
        z0, rec = m(win, rec={}), {}
        m(win, rec=rec)
        b0 = bce(z0)
        dh = np.zeros((L, H)); dn = np.zeros((L, N))
        for l in range(L):
            blk = m.blocks[l]
            x_in = rec["cuts"][2 * l]; x_mid = rec["cuts"][2 * l + 1]
            hv = rec["hv"][l]
            for h in range(H):
                hv2 = hv.clone(); hv2[:, h] = 0
                xm = x_in + blk.o(hv2.transpose(1, 2).reshape(win.shape[0], m.Q, m.d))
                x2 = xm + blk.fc2(F.relu(blk.fc1(blk.ln2(xm))))
                dh[l, h] = bce(_block_from(m, x2, l + 1, "attn")) - b0
            act = rec["act"][l]; W2 = blk.fc2.weight
            x_out = rec["cuts"][2 * l + 2]
            for n in range(N):
                x2 = x_out - act[..., n:n + 1] * W2[:, n]
                dn[l, n] = bce(_block_from(m, x2, l + 1, "attn")) - b0
    return b0, dh, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", default="8000,16000,24000,30000")
    ap.add_argument("--nev", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--abl-stride", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    Q, W, T = 64, 32, 24000
    for seed in map(int, a.seeds.split(",")):
        bits, fam, desc = make_questions(Q, T, seed)
        bits_t = torch.tensor(bits, dtype=torch.float32)
        wins = bits_t.unfold(1, W, 1)
        n_pos = T - W; split = int(n_pos * 0.8)
        ev_idx = torch.arange(split, n_pos, max(1, (n_pos - split) // a.nev))
        ev_win = wins[:, ev_idx].permute(1, 0, 2).contiguous(); ev_y = bits_t[:, ev_idx + W].T.contiguous()
        for step in map(int, a.steps.split(",")):
            fn = os.path.join(a.out, f"s{seed}-step{step}.json")
            if os.path.exists(fn):
                continue
            m = Net(Q, W, 64, 4, 2)
            m.load_state_dict(torch.load(os.path.join(WN, f"train-s{seed}-step{step}.pt"), map_location="cpu"))
            res = {"seed": seed, "step": step, "n_eval": int(ev_win.shape[0]), "rules": {}}
            for rn, rule in RULES.items():
                acc = None
                for c in range(0, ev_win.shape[0], a.chunk):
                    bk = books(m, ev_win[c:c + a.chunk], ev_y[c:c + a.chunk], rule)
                    wgt = ev_win[c:c + a.chunk].shape[0] / ev_win.shape[0]
                    if acc is None:
                        acc = {k: (np.asarray(v, dtype=float) * wgt if not isinstance(v, dict) else {kk: vv * wgt for kk, vv in v.items()}) for k, v in bk.items()}
                    else:
                        for k, v in bk.items():
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    acc[k][kk] += vv * wgt
                            else:
                                acc[k] = acc[k] + np.asarray(v, dtype=float) * wgt
                res["rules"][rn] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in acc.items()}
            b0, dh, dn = ablation(m, ev_win[::a.abl_stride], ev_y[::a.abl_stride])
            res["ablation"] = {"n_eval": int(ev_win[::a.abl_stride].shape[0]), "bce": b0, "head": dh.tolist(), "neuron": dn.tolist()}
            json.dump(res, open(fn, "w"))
            r = res["rules"]
            print(f"s{seed} step{step} R={r['v1']['R']:.4f} " + " ".join(
                f"{rn}:resid0={r[rn]['resid'][0] / abs(r[rn]['R']):+.3f} clos={max(abs(c) for c in r[rn]['closure']) / abs(r[rn]['R']):.1e}"
                for rn in RULES), flush=True)


if __name__ == "__main__":
    main()
