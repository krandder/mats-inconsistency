"""P3 / T1: do Shapley-type pay rules make latent traders pick the output-natural factors?

Hidden cause z* = (b1, b2), uniform. 32 outputs in 4 groups of 8:
    y_i = u_i XOR f_g(i)(z*) XOR eps,  eps ~ Bern(0.05),  u_i a uniform side bit the reader sees.
Toy N: f = (b1, b2, b1 AND b2, b1 OR b2).  Toy M (mirrored): b2 replaced by b1 XOR b2.
Output-natural pair: (b1, b2) on N, (b1, b1^b2) on M.  Rotated factor: b1^b2 on N, b2 on M.

Inputs the traders see (never u):
  S : z* at the 4 vertices of a regular tetrahedron (edge 1.6) + N(0, 0.1^2); all three balanced
      splits are linear and equally learnable.
  B : b1 = parity of 3 bits, b2 = majority of 5 bits (+-1 coded), so b2 is the easy one.
  all: 8 distractor bits that copy a coin c with prob. 0.9 (+-1 coded).

Traders: n MLPs (in-64-64-K), straight-through hard codes (hard argmax forward, softmax backward).
Readers:
  tab : conditional market maker, one logit per output per product bucket, a_ib + w_ib u_i, written as
        x-only logit (a_i + w_i u_i, fitted by its own log loss) + a per-bucket offset fitted by the
        maker's own log loss.  In pay/mixture pricing, buckets not visited recently (EMA share < 1e-3)
        quote the x-only price.
  add : additive reader  logit = a_i + sum_k A_ik[z_k] + u_i (w_i + sum_k W_ik[z_k])  (no trader-trader terms)
  mob : tabular reader in Moebius form, L1 penalty on the interaction coefficients (|S| >= 2)
Arms (trader objective):
  loo   backprop on the per-output log score (= each trader's leave-one-out pay gradient)
  add   backprop through the additive reader
  mob   backprop through the Moebius reader (penalty only on the reader)
  shap  exact Shapley pay (8 coalitions, absent traders marginalised by detached batch marginals)
  banz  Banzhaf pay
  asig  Aumann-Shapley / IG pay, path q_j(t) = (1-t) marginal_j + t onehot_j, 20 midpoint steps,
        value = per-output log score
  asigj AS-IG on the joint (mixture-of-products) log score
Every pay is prequential: it uses the maker from before this batch's update.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P7")  # P7 common (read-only)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # on the VM, common.py is copied next to t1.py
from common import mutual_info, entropy  # noqa: E402

EPS = 0.05
O = 32
GROUP = torch.arange(O) // 8
H005 = -(EPS * math.log(EPS) + (1 - EPS) * math.log(1 - EPS))
BAYES = O * H005  # 6.35 nats/example
VIS_THR = 1e-3
TET = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32) * (1.6 / (2 * math.sqrt(2)))
TAB_ARMS = ("loo", "shap", "banz", "asig", "asigj", "mhard")


# ---------------------------------------------------------------- data
def make_batch(g, R, B, variant, toy, shuffle=False):
    ri = lambda *s: torch.randint(0, 2, s, generator=g)
    if variant == "S":
        b1, b2 = ri(R, B), ri(R, B)
        xs = TET[2 * b1 + b2] + 0.1 * torch.randn(R, B, 3, generator=g)
    else:
        b3, b5 = ri(R, B, 3), ri(R, B, 5)
        b1 = b3.sum(-1) % 2
        b2 = (b5.sum(-1) >= 3).long()
        xs = torch.cat([b3, b5], -1).float() * 2 - 1
    c = ri(R, B, 1)
    dis = c ^ (torch.rand(R, B, 8, generator=g) < 0.1).long()
    x = torch.cat([xs, dis.float() * 2 - 1], -1)
    zb1, zb2 = (ri(R, B), ri(R, B)) if shuffle else (b1, b2)
    a = zb1
    d = zb2 if toy == "N" else zb1 ^ zb2
    f = torch.stack([a, d, a & d, a | d], -1)[..., GROUP]  # R,B,O
    u = ri(R, B, O)
    e = (torch.rand(R, B, O, generator=g) < EPS).long()
    y = u ^ f ^ e
    qf = torch.where(f == 1, 1 - EPS, EPS)
    q = torch.where(u == 1, 1 - qf, qf)  # exact P(y=1 | z, u)
    return dict(x=x, y=y.float(), u=u.float(), q=q, b1=b1, b2=b2, f=f)


def factors(toy, b1, b2):
    """(natural factor 1, natural factor 2, rotated factor)"""
    return (b1, b2, b1 ^ b2) if toy == "N" else (b1, b1 ^ b2, b2)


# ---------------------------------------------------------------- traders
class Traders:
    def __init__(self, R, n, D, K, g, hid=64, frozen=False):
        def lin(i, o):
            bd = 1 / math.sqrt(i)
            return [(torch.rand(R, n, i, o, generator=g) * 2 - 1) * bd, (torch.rand(R, n, o, generator=g) * 2 - 1) * bd]
        self.params = lin(D, hid) + lin(hid, hid) + lin(hid, K)
        if not frozen:
            for p in self.params:
                p.requires_grad_(True)

    def logits(self, x):
        W1, c1, W2, c2, W3, c3 = self.params
        h = F.relu(torch.einsum("rbd,rndh->rnbh", x, W1) + c1[:, :, None])
        h = F.relu(torch.einsum("rnbh,rnhk->rnbk", h, W2) + c2[:, :, None])
        return torch.einsum("rnbh,rnhk->rnbk", h, W3) + c3[:, :, None]  # R,n,B,K


def st_codes(logits):
    p = logits.softmax(-1)
    hard = F.one_hot(p.argmax(-1), p.shape[-1]).float()
    return hard + p - p.detach(), hard


# ---------------------------------------------------------------- readers
class TabMaker:
    """x-only maker + per-bucket offsets.  NB = K^n buckets, bucket index = mixed radix of the codes."""

    def __init__(self, R, n, K):
        self.n, self.K, self.NB = n, K, K ** n
        z = lambda *s: torch.zeros(*s, requires_grad=True)
        self.xa, self.xw, self.da, self.dw = z(R, O), z(R, O), z(R, O, self.NB), z(R, O, self.NB)
        self.params = [self.xa, self.xw, self.da, self.dw]
        self.freq = torch.zeros(R, self.NB)

    def bucket(self, hard):  # hard R,n,B,K -> R,B
        idx = hard.argmax(-1)
        b = torch.zeros_like(idx[:, 0])
        for j in range(self.n):
            b = b * self.K + idx[:, j]
        return b

    def logits(self, u, masked=True):
        xo = self.xa[:, None] + self.xw[:, None] * u  # R,B,O
        off = self.da[:, None] + self.dw[:, None] * u[..., None]  # R,B,O,NB
        if masked:
            off = off * (self.freq > VIS_THR).float()[:, None, None]
        return xo, xo.detach()[..., None] + off

    def logprobs(self, y, u, masked=True):
        s = (2 * y - 1)
        xo, la = self.logits(u, masked)
        return F.logsigmoid(s * xo), F.logsigmoid(s[..., None] * la)  # R,B,O ; R,B,O,NB

    def own_loss(self, y, u, hard):
        s = 2 * y - 1
        xo = self.xa[:, None] + self.xw[:, None] * u
        b = self.bucket(hard)[:, None, :].expand(-1, O, -1)  # R,O,B
        lb = xo.detach() + self.da.gather(2, b).transpose(1, 2) + self.dw.gather(2, b).transpose(1, 2) * u
        return -(F.logsigmoid(s * xo).sum(-1) + F.logsigmoid(s * lb).sum(-1)).mean(-1).sum()

    def update_freq(self, hard, decay=0.9):
        b = self.bucket(hard)
        share = F.one_hot(b, self.NB).float().mean(1)
        self.freq = decay * self.freq + (1 - decay) * share


class AddReader:
    def __init__(self, R, n, K):
        z = lambda *s: torch.zeros(*s, requires_grad=True)
        self.a, self.w, self.A, self.W = z(R, O), z(R, O), z(R, O, n, K), z(R, O, n, K)
        self.params = [self.a, self.w, self.A, self.W]

    def logit(self, h, u):  # h R,n,B,K
        ma = torch.einsum("rnbk,ronk->rbo", h, self.A)
        mw = torch.einsum("rnbk,ronk->rbo", h, self.W)
        return self.a[:, None] + ma + u * (self.w[:, None] + mw)

    def penalty(self):
        return 0.0


class MobReader:
    def __init__(self, R, n, lam):
        self.subsets = [S for r in range(n + 1) for S in itertools.combinations(range(n), r)]
        z = lambda *s: torch.zeros(*s, requires_grad=True)
        self.C, self.W = z(R, O, len(self.subsets)), z(R, O, len(self.subsets))
        self.params = [self.C, self.W]
        self.inter = torch.tensor([len(S) >= 2 for S in self.subsets]).float()
        self.lam = lam

    def logit(self, h, u):
        c = h[..., 1]  # R,n,B  code bit (ST)
        mono = torch.stack([torch.ones_like(c[:, 0]) if len(S) == 0 else torch.stack([c[:, j] for j in S]).prod(0)
                            for S in self.subsets], -1)  # R,B,M
        return torch.einsum("rbm,rom->rbo", mono, self.C) + u * torch.einsum("rbm,rom->rbo", mono, self.W)

    def penalty(self):
        return self.lam * ((self.C.abs() + self.W.abs()) * self.inter).sum()


# ---------------------------------------------------------------- coalition values and pay rules
def contract(P, qs, skip=None):
    """P: R,B,O',K,...,K (one axis per trader). qs: list of R,B,K (or R,1,K) weights.
    Contract every trader axis except `skip` with its weights.  Returns R,B,O' or R,B,O',K."""
    n = len(qs)
    R, B = P.shape[:2]
    ax = "wxyz"[:n]
    ins = ["rbo" + ax] + ["rb" + ax[j] for j in range(n) if j != skip]
    ops = [P] + [qs[j].to(P.dtype).expand(R, B, qs[j].shape[-1]) for j in range(n) if j != skip]
    return torch.einsum(",".join(ins) + "->rbo" + (ax[skip] if skip is not None else ""), *ops)


def price_tensor(lx, la, n, K, joint):
    """Returns (P, off): the per-example value of weights qs is  sum_o' [log contract(P, qs) + off]."""
    R, B = la.shape[:2]
    if not joint:
        return la.exp().reshape(R, B, O, *[K] * n), -lx
    lj = la.sum(2)  # R,B,NB
    mx = lj.max(-1, keepdim=True).values
    return (lj - mx).exp().reshape(R, B, 1, *[K] * n), mx - lx.sum(-1, keepdim=True)


def value(P, off, qs):  # per-example, R,B
    return (torch.log(contract(P, qs)) + off).sum(2)


def pay_asig(P, off, h, m, steps=20):
    """Aumann-Shapley pay of each trader along q(t) = m + t (h - m); per example R,B.
    Only trader i's own code carries gradient in pay_i.  G_i(t), the contraction of the other traders'
    path weights, is a polynomial in t of degree n-1; its coefficients are computed once, then the
    midpoint rule is applied to  sum_o (d_i . G_i(t)) / (q_i(t) . G_i(t))."""
    n = len(h)
    pays = []
    tt = (torch.arange(steps, dtype=P.dtype) + 0.5) / steps  # midpoint rule
    tp = torch.stack([tt ** p for p in range(n)])  # n,S
    for i in range(n):
        others = [j for j in range(n) if j != i]
        A = [0.0] * n
        for r in range(n):
            for S in itertools.combinations(others, r):
                qs = [(h[j] - m[j]).detach() if j in S else m[j] for j in range(n)]
                A[r] = A[r] + contract(P, qs, skip=i)
        d = h[i] - m[i]
        A = torch.stack(A, -1)  # R,B,O,K,n (coefficients of t^p)
        dA = (A * d[:, :, None, :, None]).sum(3)  # R,B,O,n
        mA = (A * m[i][:, :, None, :, None].to(A.dtype)).sum(3)
        num = dA @ tp  # R,B,O,S
        den = mA @ tp + dA @ (tp * tt)
        pays.append((num / den).sum(2).mean(-1))
    return pays


def pay_semi(P, off, h, m, rule):
    """Semivalue pay (exact Shapley, Banzhaf or LOO), per example R,B; gradient only through trader i's code."""
    n = len(h)
    pays = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        tot = 0.0
        for r in range(n):
            for C in itertools.combinations(others, r):
                if rule == "shap":
                    w = math.factorial(r) * math.factorial(n - 1 - r) / math.factorial(n)
                elif rule == "banz":
                    w = 1 / 2 ** (n - 1)
                else:  # loo
                    w = 1.0 if r == n - 1 else 0.0
                if w == 0:
                    continue
                qs = [h[j].detach() if j in C else m[j] for j in range(n)]
                G = contract(P, qs, skip=i)
                vin = (torch.log((G * h[i][:, :, None]).sum(-1)) + off).sum(2)
                vout = (torch.log((G * m[i][:, :, None]).sum(-1)) + off).sum(2)
                tot = tot + w * (vin - vout)
        pays.append(tot)
    return pays


# ---------------------------------------------------------------- one run (R replicates batched)
def run(args):
    torch.set_num_threads(args.threads)
    R, n, K = args.reps, args.n, args.K
    g = torch.Generator().manual_seed(1000 + args.seed)
    D = 11 if args.variant == "S" else 16
    tr = Traders(R, n, D, K, g)
    decoy = Traders(R, 1, D, K, g, frozen=True) if args.decoy else None
    ntot = n + (1 if decoy else 0)
    arm = args.arm
    vq = None
    if arm in TAB_ARMS or arm == "msoft":
        reader = TabMaker(R, ntot, K)
    elif arm == "add":
        reader = AddReader(R, ntot, K)
    elif arm == "mob":
        reader = MobReader(R, ntot, args.lam)
    elif arm == "vqsup":
        tr = Traders(R, 1, D, 4, g)  # encoder to R^4
        vq = torch.randn(R, K, 4, generator=g).requires_grad_(True)
        reader = AddReader(R, 1, 4)  # linear in the quantised vector (4 points -> any table)
    opt_t = torch.optim.Adam(tr.params + ([vq] if vq is not None else []), lr=args.lr)
    opt_r = torch.optim.Adam(reader.params, lr=args.lr_maker)
    dg = torch.Generator().manual_seed(5000 + args.seed)
    xg = torch.Generator().manual_seed(7000 + args.seed)
    half = args.steps // 2
    pay_acc = torch.zeros(R, ntot)
    pay_cnt = 0
    traj = []
    t0 = time.time()
    for step in range(args.steps):
        bt = make_batch(dg, R, args.batch, args.variant, args.toy, args.shuffle)
        lg = tr.logits(bt["x"])
        if arm == "vqsup":
            ze = lg[:, 0]  # R,B,4
            dist = ((ze[:, :, None] - vq[:, None]) ** 2).sum(-1)
            idx = dist.argmin(-1)
            zq = torch.gather(vq, 1, idx[..., None].expand(-1, -1, 4))
            zst = ze + (zq - ze).detach()
            lo = reader.logit(zst[:, None], bt["u"])
            ll = -F.logsigmoid((2 * bt["y"] - 1) * lo).sum(-1).mean(-1).sum()
            loss = ll + ((zq - ze.detach()) ** 2).sum(-1).mean(-1).sum() + 0.25 * ((ze - zq.detach()) ** 2).sum(-1).mean(-1).sum()
            opt_t.zero_grad(); opt_r.zero_grad(); loss.backward(); opt_t.step(); opt_r.step()
            continue
        if arm == "msoft":
            h = lg.softmax(-1)
            hard = F.one_hot(h.argmax(-1), K).float()
        else:
            h, hard = st_codes(lg)
        if decoy is not None:
            with torch.no_grad():
                bt2 = bt if args.decoy_real else make_batch(dg, R, args.batch, args.variant, args.toy)
                dh = F.one_hot(decoy.logits(bt2["x"]).argmax(-1), K).float()
            h = torch.cat([h, dh], 1)
            hard = torch.cat([hard, dh], 1)
        hs = [h[:, j] for j in range(ntot)]
        ms = [hard[:, j].mean(1, keepdim=True).detach() for j in range(ntot)]  # R,1,K
        if isinstance(reader, TabMaker):
            with torch.no_grad():
                lx, la = reader.logprobs(bt["y"], bt["u"], masked=True)
            P, off = price_tensor(lx, la, ntot, K, joint=(arm == "asigj"))
            if arm == "loo":
                pays = pay_semi(P, off, hs, ms, "loo")  # gradient = backprop of v(N) (unit test 3)
                loss = -sum(p.mean(1).sum() for p in pays[:n])
            elif arm == "mhard":
                v = value(P, off, hs) - value(P, off, ms)
                loss = -v.mean(1).sum()
                pays = [v]
            elif arm == "msoft":
                v = value(P, off, hs)
                loss = -v.mean(1).sum()
                pays = [v]
            elif arm in ("shap", "banz"):
                pays = pay_semi(P, off, hs, ms, arm)
                loss = -sum(p.mean(1).sum() for p in pays[:n])
            elif arm in ("asig", "asigj"):
                pays = pay_asig(P, off, hs, ms, args.ig_steps)
                loss = -sum(p.mean(1).sum() for p in pays[:n])
            opt_t.zero_grad(); loss.backward(); opt_t.step()
            if step >= half:
                pay_acc += torch.stack([p.detach().mean(1) for p in pays], 1)
                pay_cnt += 1
            # maker update (after settlement: prequential)
            hx = hard.detach()
            if args.explore > 0:  # DIAGNOSTIC (not pre-registered): the maker sees each code re-drawn with prob. explore
                flip = torch.rand(hx.shape[:-1], generator=xg) < args.explore
                rnd = F.one_hot(torch.randint(0, K, hx.shape[:-1], generator=xg), K).float()
                hx = torch.where(flip[..., None], rnd, hx)
            reader.update_freq(hx)
            if arm == "msoft":
                lxm, lam_ = reader.logprobs(bt["y"], bt["u"], masked=False)
                Pm, offm = price_tensor(lxm, lam_, ntot, K, joint=False)
                rl = -(value(Pm, offm, [x.detach() for x in hs]) - offm.sum(2)).mean(1).sum() - lxm.sum(-1).mean(-1).sum()
            else:
                rl = reader.own_loss(bt["y"], bt["u"], hx)
            opt_r.zero_grad(); rl.backward(); opt_r.step()
        else:  # add / mob: plain backprop through the structured reader
            lo = reader.logit(h, bt["u"])
            ll = -F.logsigmoid((2 * bt["y"] - 1) * lo).sum(-1).mean(-1).sum()
            loss = ll + reader.penalty()
            opt_t.zero_grad(); opt_r.zero_grad(); loss.backward(); opt_t.step(); opt_r.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                fs = factors(args.toy, bt["b1"], bt["b2"])
                row = {"step": step, "loss": float(loss) / R}
                nm = np.zeros((R, 3))
                for r in range(R):
                    cd = hard[r, :n].argmax(-1).numpy()
                    for k in range(3):
                        fk = fs[k][r].numpy()
                        nm[r, k] = max(mutual_info(cd[j], fk) / max(entropy(fk), 1e-9) for j in range(n))
                row["max_nmi_nat1_nat2_rot"] = nm.mean(0).round(3).tolist()
                traj.append(row)
    train_sec = time.time() - t0
    out = dict(config=vars(args), train_sec=train_sec, traj=traj)
    out["prequential_pay"] = (pay_acc / max(pay_cnt, 1)).tolist() if pay_cnt else None
    out["reps"] = evaluate(args, tr, decoy, reader, vq, N=args.eval_n)
    return out


# ---------------------------------------------------------------- evaluation
CANON = {(0, 0, 0, 0): "const", (0, 0, 1, 1): "b1", (0, 1, 0, 1): "b2", (0, 1, 1, 0): "xor",
         (0, 1, 1, 1): "one00", (0, 1, 0, 0): "one01", (0, 0, 1, 0): "one10", (0, 0, 0, 1): "one11"}


def classify(code, z):
    """code: binary code per example, z = 2 b1 + b2.  Returns partition name or 'mixed'."""
    fr = np.array([code[z == k].mean() if (z == k).any() else 0.5 for k in range(4)])
    if not np.all((fr <= 0.05) | (fr >= 0.95)):
        return "mixed"
    pat = tuple(int(v) for v in np.round(fr))
    if pat[0] == 1:
        pat = tuple(1 - v for v in pat)
    return CANON[pat]


def suff_gap(z, bucket, toy):
    """Exact Bayes-refit loss of a tabular reader on the codes (population, from the empirical joint of z*
    and bucket), minus the Bayes loss.  The u flip is free for the refit reader, so only f matters."""
    Fz = np.array([[a, d, a & d, a | d] for a, d in
                   [((k >> 1), (k & 1) if toy == "N" else (k >> 1) ^ (k & 1)) for k in range(4)]])  # 4z x 4g
    qf = np.where(Fz == 1, 1 - EPS, EPS)
    tot = 0.0
    N = len(z)
    for b in np.unique(bucket):
        sel = bucket == b
        pz = np.bincount(z[sel], minlength=4) / sel.sum()
        p = pz @ qf  # 4g
        ce = -(qf * np.log(p) + (1 - qf) * np.log(1 - p))  # 4z x 4g
        tot += sel.sum() / N * (pz @ ce).sum() * (O // 4)
    return tot - BAYES


@torch.no_grad()
def evaluate(args, tr, decoy, reader, vq, N=100_000, chunk=2000):
    R, n, K = args.reps, args.n, args.K
    ntot = n + (1 if decoy else 0)
    eg = torch.Generator().manual_seed(99_000 + args.seed)
    codes, zs, b1s, b2s = [], [], [], []
    mloss = torch.zeros(R)
    ece_p = torch.zeros(R, 10)
    ece_q = torch.zeros(R, 10)
    ece_n = torch.zeros(R, 10)
    ledger = None
    for ci_ in range(N // chunk):
        bt = make_batch(eg, R, chunk, args.variant, args.toy)
        lg = tr.logits(bt["x"])
        if vq is not None:
            ze = lg[:, 0]
            idx = ((ze[:, :, None] - vq[:, None]) ** 2).sum(-1).argmin(-1)
            zq = torch.gather(vq, 1, idx[..., None].expand(-1, -1, 4))
            p = torch.sigmoid(reader.logit(zq[:, None], bt["u"]))
            hard = F.one_hot(idx, K).float()[:, None]
        else:
            hard = F.one_hot(lg.argmax(-1), K).float()
            if decoy is not None:
                bt2 = bt if args.decoy_real else make_batch(eg, R, chunk, args.variant, args.toy)
                hard = torch.cat([hard, F.one_hot(decoy.logits(bt2["x"]).argmax(-1), K).float()], 1)
            if isinstance(reader, TabMaker):
                xo, la = reader.logits(bt["u"], masked=True)
                b = reader.bucket(hard)
                lo = la.gather(-1, b[:, :, None, None].expand(-1, -1, O, 1))[..., 0]
                p = torch.sigmoid(lo)
            else:
                p = torch.sigmoid(reader.logit(hard, bt["u"]))
        q = bt["q"]
        p = p.clamp(1e-7, 1 - 1e-7)
        mloss += (-(q * p.log() + (1 - q) * (1 - p).log())).sum(-1).mean(-1) / (N // chunk)
        bi = (p * 10).long().clamp(max=9)
        oh = F.one_hot(bi, 10).float()
        ece_p += torch.einsum("rbok,rbo->rk", oh, p)
        ece_q += torch.einsum("rbok,rbo->rk", oh, q)
        ece_n += oh.sum((1, 2))
        codes.append(hard.argmax(-1))
        zs.append(2 * bt["b1"] + bt["b2"])
        b1s.append(bt["b1"]); b2s.append(bt["b2"])
        if ci_ == 0 and isinstance(reader, TabMaker) and ntot > 1:
            ledger = pay_ledger(args, reader, bt, hard, K, ntot)
    codes = torch.cat(codes, -1).numpy()  # R,ntot,N
    zs = torch.cat(zs, -1).numpy(); b1s = torch.cat(b1s, -1).numpy(); b2s = torch.cat(b2s, -1).numpy()
    ece = ((ece_p - ece_q).abs().sum(-1) / ece_n.sum(-1)).numpy()
    reps = []
    for r in range(R):
        fs = factors(args.toy, b1s[r], b2s[r])
        tr_rows = []
        for j in range(ntot):
            cd = codes[r, j]
            Hc = entropy(cd) / math.log(2)
            nmi = [mutual_info(cd, fk) / entropy(fk) for fk in fs]
            tr_rows.append(dict(H_bits=Hc, nmi_nat1=nmi[0], nmi_nat2=nmi[1], nmi_rot=nmi[2],
                                cls=classify(cd, zs[r]) if K == 2 else None,
                                decoy=(j >= n)))
        bucket = np.zeros_like(zs[r])
        for j in range(ntot):
            bucket = bucket * K + codes[r, j]
        live = [t for t in tr_rows[:n] if t["H_bits"] > 0.1]
        aligned = (any(t["nmi_nat1"] >= 0.8 for t in tr_rows[:n]) and any(t["nmi_nat2"] >= 0.8 for t in tr_rows[:n])
                   and not any(t["nmi_rot"] >= 0.5 for t in live))
        rep = dict(traders=tr_rows, aligned=bool(aligned), suff_gap=suff_gap(zs[r], bucket, args.toy),
                   model_gap=float(mloss[r]) - BAYES, ece=float(ece[r]),
                   profile="|".join(sorted(t["cls"] for t in tr_rows[:n])) if K == 2 else None,
                   rot_trader=any(t["nmi_rot"] >= 0.5 for t in live),
                   dup=int(sum(t["nmi_nat1"] >= 0.8 for t in tr_rows[:n]) >= 2 or sum(t["nmi_nat2"] >= 0.8 for t in tr_rows[:n]) >= 2))
        if ledger is not None:
            rep["ledger"] = {k: (v[r] if isinstance(v, np.ndarray) else v) for k, v in ledger.items()}
            rep["ledger"] = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in rep["ledger"].items()}
        reps.append(rep)
    return reps


@torch.no_grad()
def pay_ledger(args, reader, bt, hard, K, ntot):
    """All pay rules on one eval chunk with the final maker: per-trader mean pay (nats/example),
    and the AS-IG completeness error."""
    lx, la = reader.logprobs(bt["y"], bt["u"], masked=True)
    hs = [hard[:, j] for j in range(ntot)]
    ms = [hard[:, j].mean(1, keepdim=True) for j in range(ntot)]
    out = {}
    for joint in (False, True):
        P, off = price_tensor(lx, la, ntot, K, joint)
        tag = "joint_" if joint else ""
        a = pay_asig(P, off, hs, ms, args.ig_steps)
        tot = value(P, off, hs) - value(P, off, ms)  # R,B
        s = sum(a)
        out[tag + "asig"] = torch.stack([x.mean(1) for x in a], 1).numpy()
        out[tag + "total"] = tot.mean(1).numpy()
        out[tag + "as_err_abs_mean"] = (s - tot).mean(1).abs().numpy()
        out[tag + "as_err_mean_abs"] = (s - tot).abs().mean(1).numpy()
        for rule in ("shap", "banz", "loo"):
            out[tag + rule] = torch.stack([x.mean(1) for x in pay_semi(P, off, hs, ms, rule)], 1).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--variant", default="S")
    ap.add_argument("--toy", default="N")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_maker", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--decoy", action="store_true")
    ap.add_argument("--decoy_real", action="store_true", help="decoy reads the real x (default: an independent x draw)")
    ap.add_argument("--eval_n", type=int, default=100_000)
    ap.add_argument("--explore", type=float, default=0.0, help="DIAGNOSTIC: tabular maker trains on codes re-drawn with this prob.")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = run(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    reps = out["reps"]
    print("DONE", args.arm, args.variant, args.toy, "aligned %d/%d" % (sum(r["aligned"] for r in reps), len(reps)),
          "suff_gap %.4f" % np.mean([r["suff_gap"] for r in reps]), "model_gap %.4f" % np.mean([r["model_gap"] for r in reps]),
          "profiles", sorted(set(r["profile"] for r in reps if r["profile"])), "%.0fs" % out["train_sec"])


if __name__ == "__main__":
    main()
