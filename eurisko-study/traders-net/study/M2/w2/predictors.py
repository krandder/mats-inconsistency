"""M2 item 2: predictor scores for every unit the oracle (M2/w1) ablates. See PREREG.md.

Usage: python3 predictors.py SEED [--steps 8000,20000,30000,final] [--n 2048] [--threads 2]
Writes results/pred-sS.csv (long: seed,step,final,gran,flat,unit,predictor,value) and
results/checks-sS.json. Never touches the oracle's held-out segment (positions T .. T+4095).
"""
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WN = os.path.abspath(os.path.join(HERE, "../../../../wealth-net"))
M1 = os.path.abspath(os.path.join(HERE, "../../M1"))
sys.path.insert(0, WN); sys.path.insert(0, M1)
from streams import make_questions  # noqa: E402
from model import Net  # noqa: E402
from books import grad_norms  # noqa: E402
from currency import make_books, LOCK  # noqa: E402  (sha-checks books_v2.py)

RES = os.path.join(WN, "results")
E_ORACLE, E_VAL = 4096, 2048
VAL_OFF = 8192  # val starts at T+8192: after the oracle (T..T+4095) and w3 frontier (T+4096..T+8191) segments


def lp(z, y):
    return -F.softplus(-(2 * y - 1) * z)


class Rows:
    def __init__(self, seed):
        self.seed, self.rows = seed, []

    def add(self, step, final, gran, pred, vals):
        vals = np.asarray(vals, dtype=np.float64).ravel()
        for i, v in enumerate(vals):
            unit = {"neuron": f"N{i // 256}.{i % 256}", "head": f"H{i // 4}.{i % 4}", "readout": f"R{i}"}[gran]
            self.rows.append((self.seed, step, final, gran, i, unit, pred, float(v)))


def run_rest(m, x, start, mask_n=None, mask_h=None):
    """Forward from residual cut `start` (0 = before block0 attn, 1 = before mlp0, 2 = before attn1,
    3 = before mlp1, 4 = final) to logits. Returns z [B,Q]."""
    B, Q, d = x.shape
    for c in range(start, 4):
        l, part = c // 2, c % 2
        blk = m.blocks[l]
        if part == 0:
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).view(B, Q, 3, m.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            hv = A @ v
            if mask_h is not None:
                hv = hv * mask_h[l].view(1, -1, 1, 1)
            x = x + blk.o(hv.transpose(1, 2).reshape(B, Q, d))
        else:
            act = F.relu(blk.fc1(blk.ln2(x)))
            if mask_n is not None:
                act = act * mask_n[l]
            x = x + blk.fc2(act)
    return (m.lnf(x) * m.rw).sum(-1) + m.rb


def bce_sum(z, y):
    """Total BCE (sum over examples and questions) as float64."""
    return F.binary_cross_entropy_with_logits(z, y, reduction="sum").double()


@torch.no_grad()
def ablate_single(m, win, y, chunk=16, bsz=512):
    """Exact zero-ablation dloss (mean BCE per position x question) of every neuron, head, readout."""
    L, H, N = m.L, m.H, 4 * m.d
    n_tot = win.shape[0] * win.shape[1]
    dn = torch.zeros(L * N, dtype=torch.float64); dh = torch.zeros(L * H, dtype=torch.float64)
    dr = torch.zeros(m.Q, dtype=torch.float64); base = 0.0
    for b0 in range(0, win.shape[0], bsz):
        w, yy = win[b0:b0 + bsz], y[b0:b0 + bsz]
        rec = {}
        z = m(w, rec=rec)
        bl = bce_sum(z, yy); base += float(bl)
        per_q = F.binary_cross_entropy_with_logits(z, yy, reduction="none").sum(0).double()
        per_q0 = F.binary_cross_entropy_with_logits(torch.zeros_like(z), yy, reduction="none").sum(0).double()
        dr += per_q0 - per_q
        cuts = rec["cuts"]  # x0, after attn0, after mlp0, after attn1, after mlp1
        for l in range(L):
            act = rec["act"][l]; W2 = m.blocks[l].fc2.weight  # [d,N]
            post = cuts[2 * l + 2]
            for i0 in range(0, N, chunk):
                idx = torch.arange(i0, min(N, i0 + chunk))
                k = len(idx)
                xa = post.unsqueeze(2) - act[..., idx].unsqueeze(-1) * W2[:, idx].T.view(1, 1, k, -1)  # [B,Q,k,d]
                xa = xa.permute(2, 0, 1, 3).reshape(k * w.shape[0], m.Q, m.d)
                za = run_rest(m, xa, 2 * l + 2).view(k, w.shape[0], m.Q)
                la = F.binary_cross_entropy_with_logits(za, yy.unsqueeze(0).expand(k, -1, -1), reduction="none").sum((1, 2)).double()
                dn[l * N + idx] += la - bl
            for h in range(H):
                mh = torch.ones(L, H); mh[l, h] = 0
                za = run_rest(m, cuts[2 * l], 2 * l, mask_h=mh)
                dh[l * H + h] += bce_sum(za, yy) - bl
    return dn.numpy() / n_tot, dh.numpy() / n_tot, dr.numpy() / n_tot, base / n_tot


@torch.no_grad()
def head_table(m, win, y, bsz=512):
    """v(S) = -mean BCE with heads outside S zeroed, for all 256 subsets (bit j = flat head j)."""
    L, H = m.L, m.H
    n_tot = win.shape[0] * win.shape[1]
    v = np.zeros(2 ** (L * H))
    for b0 in range(0, win.shape[0], bsz):
        w, yy = win[b0:b0 + bsz], y[b0:b0 + bsz]
        x0 = m.win(2.0 * w - 1.0) + m.qid.unsqueeze(0)
        for s0 in range(2 ** H):  # layer-0 subset: run block 0 once
            mh = torch.ones(L, H)
            mh[0] = torch.tensor([(s0 >> j) & 1 for j in range(H)], dtype=torch.float32)
            B_, Q, d = x0.shape
            blk = m.blocks[0]
            h = blk.ln1(x0)
            q, k, vv = blk.qkv(h).view(B_, Q, 3, m.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            x1 = x0 + blk.o(((A @ vv) * mh[0].view(1, -1, 1, 1)).transpose(1, 2).reshape(B_, Q, d))
            x2 = x1 + blk.fc2(F.relu(blk.fc1(blk.ln2(x1))))
            for s1 in range(2 ** H):
                mh[1] = torch.tensor([(s1 >> j) & 1 for j in range(H)], dtype=torch.float32)
                z = run_rest(m, x2, 2, mask_h=mh)
                v[s0 | (s1 << H)] -= float(bce_sum(z, yy))
    return v / n_tot


def shapley_from_table(v, n, perms=None):
    phi = np.zeros(n)
    if perms is None:  # exact
        for S in range(2 ** n):
            k = bin(S).count("1")
            for j in range(n):
                if not (S >> j) & 1:
                    wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    phi[j] += wgt * (v[S | (1 << j)] - v[S])
        return phi
    for p in perms:
        S = 0
        for j in p:
            phi[j] += v[S | (1 << j)] - v[S]; S |= 1 << j
    return phi / len(perms)


def taylor(m, win, y, bsz=512):
    """Live-graph first-order scores: mean_b sum -a.dL/da (signed), mean_b |.| per example, grad norms."""
    L, H, N = m.L, m.H, 4 * m.d
    acc = {k: np.zeros(n) for k, n in (("n", L * N), ("h", L * H), ("r", m.Q))}
    acca = {k: np.zeros(n) for k, n in (("n", L * N), ("h", L * H), ("r", m.Q))}
    for b0 in range(0, win.shape[0], bsz):
        w, yy = win[b0:b0 + bsz], y[b0:b0 + bsz]
        rec = {}
        z = m(w, rec=rec)
        loss = F.binary_cross_entropy_with_logits(z, yy, reduction="sum")
        tg = rec["act"] + rec["hv"] + [z]
        gr = torch.autograd.grad(loss, tg)
        with torch.no_grad():
            tn = torch.cat([-(rec["act"][l] * gr[l]).sum(1) for l in range(L)], -1)            # [B, L*N]
            th = torch.cat([-(rec["hv"][l] * gr[L + l]).sum((2, 3)) for l in range(L)], -1)    # [B, L*H]
            tr = -(z * gr[-1])                                                              # [B, Q]
            for k, t in (("n", tn), ("h", th), ("r", tr)):
                acc[k] += t.sum(0).double().numpy(); acca[k] += t.abs().sum(0).double().numpy()
    B = win.shape[0]
    return {k: v / B for k, v in acc.items()}, {k: v / B for k, v in acca.items()}


def grad_norm_inst(m, win, y):
    m.zero_grad()
    z = m(win)
    F.binary_cross_entropy_with_logits(z, y).backward()
    gn = grad_norms(m)
    m.zero_grad()
    return gn["neuron"].numpy().ravel(), gn["head"].numpy().ravel(), gn["out"].numpy()


def conductance(m, win, y, steps=32, bsz=512):
    """Conductance of each unit for F = sum log p(y); input path from window 0.5 to the window
    (trapezoid on steps+1 grid points). Returns neuron, head, readout arrays and per-layer
    completeness info (sum over units vs F(x)-F(x'))."""
    L, H, N = m.L, m.H, 4 * m.d
    cn = np.zeros(L * N); ch = np.zeros(L * H); cr = np.zeros(m.Q); dF = 0.0
    al = torch.linspace(0, 1, steps + 1)
    for b0 in range(0, win.shape[0], bsz):
        w, yy = win[b0:b0 + bsz], y[b0:b0 + bsz]
        prev = None
        for k, a in enumerate(al):
            wa = 0.5 + a * (w - 0.5)
            rec = {}
            z = m(wa, rec=rec)
            Fv = lp(z, yy).sum()
            gr = torch.autograd.grad(Fv, rec["act"] + rec["hv"] + [z])
            cur = ([t.detach() for t in rec["act"]], [t.detach() for t in rec["hv"]], z.detach(), [g.detach() for g in gr], float(Fv))
            if prev is not None:
                pa, ph, pz, pg, pF = prev
                ca, chh, cz, cg, cF = cur
                for l in range(L):
                    g_ = 0.5 * (pg[l] + cg[l]); cn[l * N:(l + 1) * N] += (g_ * (ca[l] - pa[l])).sum((0, 1)).double().numpy()
                    g_ = 0.5 * (pg[L + l] + cg[L + l]); ch[l * H:(l + 1) * H] += (g_ * (chh[l] - ph[l])).sum((0, 2, 3)).double().numpy()
                g_ = 0.5 * (pg[-1] + cg[-1]); cr += (g_ * (cz - pz)).sum(0).double().numpy()
            else:
                F0 = float(Fv)
            prev = cur
        dF += prev[4] - F0
    B = win.shape[0]
    return cn / B, ch / B, cr / B, dF / B


def books_v2(m, fam, win, y, prior_n1, prior_n):
    bk = make_books(m, fam, 8, groups=[list(range(8))])
    bk.n1 = prior_n1.clone(); bk.n = prior_n.clone()
    for b0 in range(0, win.shape[0], 512):
        bk.observe(win[b0:b0 + 512], y[b0:b0 + 512], torch.ones_like(y[b0:b0 + 512]))
    gain = 0.0
    for b0 in range(0, win.shape[0], 512):
        acc = bk.account(win[b0:b0 + 512], y[b0:b0 + 512], None, freeze_attn=True)
        bk.add(acc)
        gain += float(acc["acct"]["gain"].sum()) if "acct" in acc else float("nan")
    out = {}
    for b in ("tangent", "lmsr", "meanbase"):
        out[b] = {g: bk.W[b][g].sum(-1).cpu().numpy().ravel() for g in ("neuron", "head", "out")}
    out["absag"] = {g: bk.W["absag"][g].cpu().numpy().ravel() for g in ("neuron", "head", "out")}
    ac = {k: float(v.sum()) for k, v in bk.W["acct"].items()}
    out["_c1"] = {"lmsr_out_sum": float(bk.W["lmsr"]["out"].sum()), **ac,
                  "rel_diff_lmsr_out_vs_gain": (float(bk.W["lmsr"]["out"].sum()) - ac["gain"]) / abs(ac["gain"])}
    return out


def snap(tag, step):
    return json.load(open(os.path.join(RES, tag, "snapshots", f"step-{step:05d}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--steps", default="8000,20000,30000,final")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--val-steps", default="final")  # steps that also get the _val sample ("all" = every step)
    ap.add_argument("--only", default="")          # comma list of blocks to run (debug)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    seed = a.seed; tag = f"train-s{seed}"
    meta = json.load(open(os.path.join(RES, f"{tag}.json")))
    Q, W, T = meta["Q"], meta["W"], meta["T"]
    final = meta["steps_done"]
    bits, fam, _ = make_questions(Q, T + VAL_OFF + E_VAL, seed)
    b = torch.tensor(bits, dtype=torch.float32)
    wins = b.unfold(1, W, 1)
    split = int((T - W) * 0.8)
    tr = torch.randint(0, split, (a.n,), generator=torch.Generator().manual_seed(seed + 77))
    va = torch.arange(T + VAL_OFF - W, T + VAL_OFF + E_VAL - W)[: a.n]
    assert va.min() + W >= T + 8192, "val must not overlap the oracle or w3 frontier segments"
    assert tr.max() + W < T, "train sample must lie in training data"
    data = {"": (wins[:, tr].permute(1, 0, 2).contiguous(), b[:, tr + W].T.contiguous()),
            "_val": (wins[:, va].permute(1, 0, 2).contiguous(), b[:, va + W].T.contiguous())}
    yb = b[:, W:split + W]
    prior_n1, prior_n = yb.sum(1), torch.full((Q,), float(yb.shape[1]))
    steps = [final if s == "final" else int(s) for s in a.steps.split(",")]
    only = set(a.only.split(",")) if a.only else None
    R = Rows(seed); checks = {"seed": seed, "lock_sha": LOCK["code"]["sha256"]}

    def load(step):
        m = Net(Q, W, meta["d"], meta["H"], meta["L"])
        wf = os.path.join(RES, f"{tag}.pt") if step == final else os.path.join(RES, "weights", f"{tag}-step{step}.pt")
        m.load_state_dict(torch.load(wf, map_location="cpu")); m.eval()
        for p in m.parameters():
            p.requires_grad_(True)
        return m

    def want(k):
        return only is None or k in only

    # --- books v2 at every saved checkpoint <= max step (cached per checkpoint) ---
    bv2 = {}
    if want("v2"):
        ck = sorted(set([s for s in range(8000, 30001, 2000)] + [final]))
        for s in ck:
            if s > max(steps):
                continue
            bv2[s] = books_v2(load(s), fam, *data[""], prior_n1, prior_n)
            checks.setdefault("C1_lmsr_out_vs_gain", {})[s] = bv2[s]["_c1"]
            print(seed, "v2", s, round(time.time() - t0), flush=True)

    # --- movement / live-grad cumulative from M1/w1 replay table ---
    mv = {}
    if want("move"):
        rows = list(csv.DictReader(open(os.path.join(M1, "w1", "results", f"units-s{seed}.csv"))))
        cols = ["X_move_in", "X_move_out", "X_grad_live_out", "X_grad_books_out"]
        for s in steps:
            wins_ = [(0, final)] if s == final else [(0, 8000)] + [(t, t + 2000) for t in range(8000, s, 2000)]
            agg = {}
            for r in rows:
                if (int(r["t0"]), int(r["t1"])) in wins_:
                    key = (r["gran"], int(r["unit"]))
                    v = agg.setdefault(key, np.zeros(len(cols)))
                    v += [float(r[c]) for c in cols]
            mv[s] = agg
            checks.setdefault("move_windows", {})[s] = len(wins_)

    val_steps = steps if a.val_steps == "all" else [final if v == "final" else int(v) for v in a.val_steps.split(",")]
    tagp = a.only.replace(",", "_") if a.only else "all"
    for s in steps:
        fin = int(s == final)
        fn_s = os.path.join(a.out, f"pred-s{seed}-{tagp}-step{s}.csv")
        if os.path.exists(fn_s):
            print(seed, s, "exists, skip", flush=True); continue
        R.rows = []
        m = load(s)
        # snapshots
        if want("snap"):
            sn = snap(tag, s); old = snap(tag, s - 2000)
            for g, gg in (("neuron", "neuron"), ("head", "head"), ("readout", "out")):
                w = np.array(sn["wealth"][gg], dtype=float).ravel()
                R.add(s, fin, g, "v1_wealth", w)
                R.add(s, fin, g, "v1_recent2k", w - np.array(old["wealth"][gg], dtype=float).ravel())
                R.add(s, fin, g, "gradnorm_cum", np.array(sn["gradnorm_cum"][gg], dtype=float).ravel())
            ab = {g: 0 for g in ("neuron", "head", "out")}
            for t in range(100, s + 1, 100):
                d = snap(tag, t)
                for g in ab:
                    ab[g] = ab[g] + np.abs(np.array(d["income_window"][g], dtype=float).ravel())
            for g, gg in (("neuron", "neuron"), ("head", "head"), ("readout", "out")):
                R.add(s, fin, g, "absag_acc", ab[gg])
            print(seed, s, "snap", round(time.time() - t0), flush=True)
        if want("v2") and bv2:
            ck = [c for c in bv2 if c <= s]
            for bk_ in ("tangent", "lmsr", "meanbase", "absag"):
                for g, gg in (("neuron", "neuron"), ("head", "head"), ("readout", "out")):
                    R.add(s, fin, g, f"v2_{bk_}_int" if bk_ != "absag" else "absag_int", sum(bv2[c][bk_][gg] for c in ck))
                    if bk_ != "absag":
                        R.add(s, fin, g, f"v2_{bk_}_inst", bv2[s][bk_][gg])
        if want("move") and s in mv:
            agg = mv[s]
            for g, gg in (("neuron", "neuron"), ("head", "head"), ("readout", "out")):
                keys = sorted(k for k in agg if k[0] == gg)
                assert [k[1] for k in keys] == list(range(len(keys))), (gg, len(keys))
                A_ = np.stack([agg[k] for k in keys])
                R.add(s, fin, g, "movement", A_[:, 0] + A_[:, 1])
                R.add(s, fin, g, "movement_out", A_[:, 1])
                R.add(s, fin, g, "grad_live_cum", A_[:, 2])
                R.add(s, fin, g, "v1_replay_books", A_[:, 3])
        if want("norm"):
            with torch.no_grad():
                wo_n = torch.cat([blk.fc2.weight.norm(dim=0) for blk in m.blocks]).numpy()
                wi_n = torch.cat([torch.sqrt(blk.fc1.weight.pow(2).sum(1) + blk.fc1.bias.pow(2)) for blk in m.blocks]).numpy()
                dh_ = m.d // m.H
                wo_h = torch.cat([blk.o.weight.view(m.d, m.H, dh_).pow(2).sum((0, 2)).sqrt() for blk in m.blocks]).numpy()
                wi_h = torch.cat([blk.qkv.weight.view(3, m.H, dh_, m.d).pow(2).sum((0, 2, 3)).sqrt() for blk in m.blocks]).numpy()
                wo_r = m.rw.norm(dim=1).numpy()
            R.add(s, fin, "neuron", "wout_norm", wo_n); R.add(s, fin, "neuron", "winout_norm", np.sqrt(wo_n ** 2 + wi_n ** 2))
            R.add(s, fin, "head", "wout_norm", wo_h); R.add(s, fin, "head", "winout_norm", np.sqrt(wo_h ** 2 + wi_h ** 2))
            R.add(s, fin, "readout", "wout_norm", wo_r); R.add(s, fin, "readout", "winout_norm", wo_r)
            gn, gh, gr_ = grad_norm_inst(m, *data[""])
            R.add(s, fin, "neuron", "gradnorm_inst", gn); R.add(s, fin, "head", "gradnorm_inst", gh); R.add(s, fin, "readout", "gradnorm_inst", gr_)
        for sfx, (w_, y_) in data.items():
            if sfx == "_val" and s not in val_steps:
                continue
            if want("taylor"):
                sg, ab_ = taylor(m, w_, y_)
                for k, g in (("n", "neuron"), ("h", "head"), ("r", "readout")):
                    R.add(s, fin, g, "taylor" + sfx, sg[k]); R.add(s, fin, g, "taylor_abs" + sfx, ab_[k])
                    R.add(s, fin, g, "taylor_sq" + sfx, sg[k] ** 2)
                print(seed, s, "taylor" + sfx, round(time.time() - t0), flush=True)
            if want("cond"):
                cn, ch, cr, dF = conductance(m, w_, y_)
                R.add(s, fin, "neuron", "cond" + sfx, cn); R.add(s, fin, "head", "cond" + sfx, ch); R.add(s, fin, "readout", "cond" + sfx, cr)
                checks.setdefault("C3_cond_completeness", {})[f"{s}{sfx}"] = {"readout_sum": float(cr.sum()), "dF": dF,
                    "mlp_l0": float(cn[:256].sum()), "mlp_l1": float(cn[256:].sum()), "heads": float(ch.sum())}
                print(seed, s, "cond" + sfx, round(time.time() - t0), flush=True)
            if want("abl"):
                dn, dh, dr, base = ablate_single(m, w_, y_)
                nm = "abl_train" if sfx == "" else "abl_val"
                R.add(s, fin, "neuron", nm, dn); R.add(s, fin, "head", nm, dh); R.add(s, fin, "readout", nm, dr)
                R.add(s, fin, "readout", "shap_readout" + sfx, dr)
                checks.setdefault("base_bce", {})[f"{s}{sfx}"] = base
                print(seed, s, nm, round(time.time() - t0), flush=True)
            if want("shap"):
                v = head_table(m, w_, y_)
                ex = shapley_from_table(v, 8)
                rng = np.random.default_rng(seed * 1000 + s)
                sm = shapley_from_table(v, 8, [rng.permutation(8) for _ in range(64)])
                R.add(s, fin, "head", "shap_head_exact" + sfx, ex); R.add(s, fin, "head", "shap_head_s64" + sfx, sm)
                checks.setdefault("C2_shap_efficiency", {})[f"{s}{sfx}"] = float(ex.sum() - (v[255] - v[0]))
                np.save(os.path.join(a.out, f"headtable-s{seed}-step{s}{sfx}.npy"), v)
                print(seed, s, "shap" + sfx, round(time.time() - t0), flush=True)
        with open(fn_s + ".part", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["seed", "step", "final", "gran", "flat", "unit", "predictor", "value"])
            wr.writerows(R.rows)
        os.replace(fn_s + ".part", fn_s)
        checks["seconds"] = round(time.time() - t0)
        json.dump(checks, open(os.path.join(a.out, f"checks-s{seed}-{tagp}.json"), "w"), indent=1, default=str)
        print("wrote", fn_s, len(R.rows), "rows", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
