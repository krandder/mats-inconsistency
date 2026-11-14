"""P10 S6 (6x6, 2x3 boxes, unique puzzles): headroom pilot, atom bank, and M2 learned events.

  python train.py s6pilot T SEED     B2-reg at 2T passes (headroom pilot)
  python train.py s6bank T SEED      atom-bank model: one random atom per example as a slot -> LOO gain and rate per atom
  python train.py m2 ARM T SEED      ARM in {auction, cvae, reg}

Unique puzzles: Y is a function of X, so every Shannon term is 0 and any gain is V-information.
Units: nats per cell = nats per example / 36.
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grids
from models import B2

torch.set_num_threads(int(os.environ.get("P10_THREADS", "2")))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P10_RES", os.path.join(HERE, "results"))
TMP = os.environ.get("P10_TMP", "/tmp/p10")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C, N = 36, 6
FEE = float(os.environ.get("P10_FEE", "0.005"))


def cosine_lr(opt, step, total, base, warm=200):
    lr = base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups:
        g["lr"] = lr


# ------------------------------------------------------------------ data
class S6:
    def __init__(self, seed, path=f"{TMP}/s6.npz"):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}
        sp = self.z["split"]
        self.idx = {"train": np.flatnonzero(sp == 0)[:20000], "val": np.flatnonzero(sp == 1)[:2000],
                    "test": np.flatnonzero(sp == 2)[:2000]}
        self.P = torch.as_tensor(grids.sym_perms(grids.G6).astype(np.int64))
        self.g = torch.Generator().manual_seed(seed)
        self.T = {k: torch.as_tensor(v.astype(np.int64)) for k, v in self.z.items() if k in ("puz", "sol", "force")}
        self.T["naked"] = torch.as_tensor(self.z["naked"].astype(np.int64))
        self.T["hidden"] = torch.as_tensor(self.z["hidden"].astype(np.int64))

    def batch(self, split, B=None, idx=None, aug=True):
        """Returns dict of tensors: x (B,C) tokens 0..6, y (B,C) 0..5, sol (B,C) 1..6, mask (B,C) bool (masked),
        force (B,C), naked/hidden (B,3,C)."""
        if idx is None:
            pool = self.idx[split]
            idx = pool[torch.randint(0, len(pool), (B,), generator=self.g).numpy()]
        idx = torch.as_tensor(idx)
        puz, sol = self.T["puz"][idx], self.T["sol"][idx]
        force, nk, hd = self.T["force"][idx], self.T["naked"][idx], self.T["hidden"][idx]
        if aug:
            Bn = len(idx)
            p = self.P[torch.randint(0, len(self.P), (Bn,), generator=self.g)]
            relab = torch.argsort(torch.rand(Bn, N, generator=self.g), 1) + 1
            relab = torch.cat([torch.zeros(Bn, 1, dtype=torch.long), relab], 1)
            puz = torch.gather(relab, 1, torch.gather(puz, 1, p))
            sol = torch.gather(relab, 1, torch.gather(sol, 1, p))
            force = torch.gather(force, 1, p)
            nk = torch.gather(nk, 2, p[:, None, :].expand(-1, 3, -1))
            hd = torch.gather(hd, 2, p[:, None, :].expand(-1, 3, -1))
        out = {"x": puz, "y": sol - 1, "sol": sol, "mask": puz == 0, "force": force, "naked": nk, "hidden": hd}
        return {k: v.to(DEV) for k, v in out.items()}


class Atoms:
    """Vectorised atom library on (augmented) batches."""
    def __init__(self):
        self.specs, self.ncell_start = grids.atom_specs(grids.G6)
        G = grids.G6
        A = len(self.specs)
        W = torch.zeros(A, C)
        dig = torch.zeros(A, dtype=torch.long)
        kind = []
        for a, s in enumerate(self.specs):
            kind.append(s[0])
            if s[0] == 'bl_row':
                _, b, d, r = s
                W[a, [i for i in range(C) if G.box[i] == b and G.row[i] == r]] = 1
                dig[a] = d
            elif s[0] == 'bl_col':
                _, b, d, c = s
                W[a, [i for i in range(C) if G.box[i] == b and G.col[i] == c]] = 1
                dig[a] = d
        self.W, self.dig, self.kind = W, dig, kind
        self.A = A
        self.is_bl = torch.tensor([k in ('bl_row', 'bl_col') for k in kind])
        self.cell = torch.tensor([s[1] if s[0] not in ('bl_row', 'bl_col') else 0 for s in self.specs])
        self.rnd = torch.tensor([s[2] if s[0] in ('naked', 'hidden', 'forced') else 0 for s in self.specs])
        self.eqd = torch.tensor([s[2] if s[0] == 'eq' else 0 for s in self.specs])
        tmap = {'bl_row': 0, 'bl_col': 1, 'naked': 2, 'hidden': 3, 'forced': 4, 'eq': 5, 'le3': 6}
        self.type_id = torch.tensor([tmap[k] for k in kind])
        # compositional descriptors for the bank embedding
        self.box_id = torch.tensor([s[1] if s[0] in ('bl_row', 'bl_col') else 0 for s in self.specs])
        self.line_id = torch.tensor([s[3] if s[0] in ('bl_row', 'bl_col') else 0 for s in self.specs])
        self.digit_id = torch.tensor([s[2] if s[0] in ('bl_row', 'bl_col', 'eq') else 0 for s in self.specs])
        for k in ("W", "dig", "is_bl", "cell", "rnd", "eqd", "type_id", "box_id", "line_id", "digit_id"):
            setattr(self, k, getattr(self, k).to(DEV))
        self.kmask = {k: torch.tensor([kk == k for kk in self.kind], device=DEV) for k in set(self.kind)}

    def values(self, bt, which=None):
        """(B, A') float atom values for atom ids `which` (default all)."""
        which = torch.arange(self.A) if which is None else torch.as_tensor(which).cpu()
        sol, force, nk, hd = bt["sol"], bt["force"], bt["naked"], bt["hidden"]
        oh = F.one_hot(sol - 1, N).float()  # (B,C,N)
        out = torch.zeros(len(sol), len(which), device=DEV)
        for j, a in enumerate(which.tolist()):
            k = self.kind[a]
            if k in ('bl_row', 'bl_col'):
                out[:, j] = ((oh[:, :, self.dig[a] - 1] * self.W[a]).sum(1) > 0).float()
            elif k == 'naked':
                out[:, j] = nk[:, self.rnd[a] - 1, self.cell[a]].float()
            elif k == 'hidden':
                out[:, j] = hd[:, self.rnd[a] - 1, self.cell[a]].float()
            elif k == 'forced':
                f = force[:, self.cell[a]]
                out[:, j] = ((f >= 1) & (f <= self.rnd[a])).float()
            elif k == 'eq':
                out[:, j] = (sol[:, self.cell[a]] == self.eqd[a]).float()
            else:
                out[:, j] = (sol[:, self.cell[a]] <= 3).float()
        return out

    def values_all(self, bt):
        """All atoms at once (faster path)."""
        sol, force, nk, hd = bt["sol"], bt["force"], bt["naked"], bt["hidden"]
        B = len(sol)
        oh = F.one_hot(sol - 1, N).float()  # (B,C,N)
        cnt = torch.einsum('bcn,ac->ban', oh, self.W)  # (B,A,N)
        bl = (cnt.gather(2, (self.dig.clamp_min(1) - 1)[None, :, None].expand(B, -1, 1)).squeeze(-1) > 0).float()
        out = torch.zeros(B, self.A, device=DEV)
        out[:, self.is_bl] = bl[:, self.is_bl]
        cellv = self.cell
        rnd = (self.rnd.clamp_min(1) - 1)
        for k, arr in (('naked', nk), ('hidden', hd)):
            m = self.kmask[k]
            out[:, m] = arr[:, rnd[m], cellv[m]].float()
        m = self.kmask['forced']
        f = force[:, cellv[m]]
        out[:, m] = ((f >= 1) & (f <= self.rnd[m][None])).float()
        m = self.kmask['eq']
        out[:, m] = (sol[:, cellv[m]] == self.eqd[m][None]).float()
        m = self.kmask['le3']
        out[:, m] = (sol[:, cellv[m]] <= 3).float()
        return out


def cell_nll(logits, y, mask):
    """Per-example summed NLL over masked cells (B,)."""
    l = F.cross_entropy(logits.reshape(-1, N), y.reshape(-1), reduction='none').view_as(mask)
    return (l * mask).sum(-1)


# ------------------------------------------------------------------ pilot: B2-reg headroom
def pilot(T, seed, steps=None):
    steps = steps or int(os.environ.get("P10_S6_STEPS", "3000"))
    torch.manual_seed(seed)
    d = S6(seed)
    model = B2(C, N, d=64, heads=4, nslot=8).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        cosine_lr(opt, step, steps, 2e-3)
        bt = d.batch("train", 128)
        logits, _, _ = model.run_slots(bt["x"], T, reg=True)
        loss = cell_nll(logits, bt["y"], bt["mask"]).mean() / C
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"[pilot T={T}] step {step} loss/cell {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    ev = eval_reg(model, d, T, "val")
    res = {"T": T, "seed": seed, "steps": steps, "time_s": time.time() - t0, **ev}
    print("pilot", res, flush=True)
    json.dump(res, open(f"{RES}/s6_pilot_T{T}_seed{seed}.json", "w"), indent=1)
    return res


@torch.no_grad()
def eval_reg(model, d, T, split):
    idx = d.idx[split]
    tot, tot_m, n_m, acc = 0.0, 0.0, 0, 0.0
    for i in range(0, len(idx), 500):
        bt = d.batch(split, idx=idx[i:i + 500], aug=False)
        logits, _, _ = model.run_slots(bt["x"], T, reg=True)
        nl = cell_nll(logits, bt["y"], bt["mask"])
        tot += float(nl.sum())
        tot_m += float((nl / bt["mask"].sum(-1)).sum())
        acc += float(((logits.argmax(-1) == bt["y"]) | ~bt["mask"]).all(-1).float().sum())
    n = len(idx)
    return {"nll_per_cell": tot / n / C, "nll_per_masked_cell": tot_m / n, "solve_rate_argmax": acc / n}


# ------------------------------------------------------------------ atom bank
class Bank(nn.Module):
    """B2 with one slot whose query/value embeddings are compositional in the atom descriptor."""
    def __init__(self, atoms, d=64):
        super().__init__()
        self.net = B2(C, N, d=d, heads=4, nslot=1)
        self.atoms = atoms
        self.type_q = nn.Embedding(8, d)  # 7 types + null
        self.box = nn.Embedding(6, d); self.line = nn.Embedding(6, d); self.digit = nn.Embedding(7, d)
        self.cell = nn.Embedding(36, d); self.rnd = nn.Embedding(4, d)
        self.val = nn.Embedding(2, d)
        self.pw = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
        self.null = nn.Parameter(torch.randn(d) * 0.02)

    def desc(self, a):
        at = self.atoms
        e = (self.type_q(at.type_id[a]) + self.box(at.box_id[a]) + self.line(at.line_id[a]) + self.digit(at.digit_id[a])
             + self.cell(at.cell[a]) + self.rnd(at.rnd[a]))
        return e

    def forward(self, x, a, ev, is_null):
        """a (B,) atom ids, ev (B,) values, is_null (B,) bool. Returns cell logits after 2T, prior logit (B,)."""
        net = self.net
        B = len(x)
        ec = net.embed(x)
        q = torch.where(is_null[:, None], self.null[None].expand(B, -1), self.desc(a))
        e1 = torch.cat([ec, (q + net.slot_pos[0])[:, None]], 1)
        h, _ = net.run(None, self.T, e=e1)
        pl = self.pw(self.ln(h[:, C])).squeeze(-1)
        v = torch.where(is_null[:, None], self.null[None].expand(B, -1), q + self.val(ev.long()))
        e2 = torch.cat([ec, (v + net.slot_pos[0])[:, None]], 1)
        _, logits = net.run(None, self.T, h=h, e=e2)
        return logits, pl


def bank(T, seed, steps=None):
    steps = steps or int(os.environ.get("P10_S6_STEPS", "3000"))
    torch.manual_seed(seed)
    d = S6(seed)
    at = Atoms()
    m = Bank(at).to(DEV)
    m.T = T
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        cosine_lr(opt, step, steps, 2e-3)
        bt = d.batch("train", 128)
        B = len(bt["x"])
        a = torch.randint(0, at.A, (B,), device=DEV)
        isn = torch.rand(B, device=DEV) < 0.2
        vals = at.values_all(bt)
        ev = vals[torch.arange(B, device=DEV), a]
        logits, pl = m(bt["x"], a, ev, isn)
        rate = F.binary_cross_entropy_with_logits(pl, ev, reduction='none') * (~isn)
        loss = (cell_nll(logits, bt["y"], bt["mask"]) + rate).mean() / C
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"[bank T={T}] step {step} loss/cell {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval()
    # LOO gain and rate per atom on val (subsample)
    idx = d.idx["val"][:int(os.environ.get("P10_BANK_EVAL", "600"))]
    bt = d.batch("val", idx=idx, aug=False)
    B = len(idx)
    vals = at.values_all(bt)
    with torch.no_grad():
        z0 = dict(device=DEV)
        l0, _ = m(bt["x"], torch.zeros(B, dtype=torch.long, **z0), torch.zeros(B, **z0), torch.ones(B, dtype=torch.bool, **z0))
        nl0 = cell_nll(l0, bt["y"], bt["mask"])
        gain, rate, Gk, p1 = [], [], [], []
        for a in range(at.A):
            aa = torch.full((B,), a, dtype=torch.long, device=DEV)
            ev = vals[:, a]
            la, pl = m(bt["x"], aa, ev, torch.zeros(B, dtype=torch.bool, **z0))
            nla = cell_nll(la, bt["y"], bt["mask"])
            # prior-resampled counterfactual: E_{e~p}[l(Y|X,e~)]
            l1, _ = m(bt["x"], aa, torch.ones(B, **z0), torch.zeros(B, dtype=torch.bool, **z0))
            l0v, _ = m(bt["x"], aa, torch.zeros(B, **z0), torch.zeros(B, dtype=torch.bool, **z0))
            pr = torch.sigmoid(pl)
            cf = pr * cell_nll(l1, bt["y"], bt["mask"]) + (1 - pr) * cell_nll(l0v, bt["y"], bt["mask"])
            r = F.binary_cross_entropy_with_logits(pl, ev, reduction='none')
            gain.append(float((nl0 - nla).mean()) / C)
            rate.append(float(r.mean()) / C)
            Gk.append(float((cf - nla - r).mean()) / C)
            p1.append(float(ev.mean()))
    res = {"T": T, "seed": seed, "steps": steps, "time_s": time.time() - t0, "nll_null_per_cell": float(nl0.mean()) / C,
           "atoms": [{"spec": list(map(lambda v: v if isinstance(v, str) else int(v), at.specs[a])), "cell_atom": a >= at.ncell_start,
                      "gain_vs_null": gain[a], "rate": rate[a], "G_resampled": Gk[a], "p1": p1[a]} for a in range(at.A)]}
    nc = [r for r in res["atoms"] if not r["cell_atom"] and 0.02 < r["p1"] < 0.98]
    best = max(nc, key=lambda r: r["G_resampled"])
    res["planted"] = best
    print("planted atom", best, flush=True)
    json.dump(res, open(f"{RES}/s6_bank_T{T}_seed{seed}.json", "w"), indent=1)
    return res


# ------------------------------------------------------------------ M2
SLOT_NAMES = ["learned0", "learned1", "learned2", "planted", "cellbit", "trivial", "random", "copyhash"]
K_LEARN = 3


class Phi(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C * (N + 1), hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, sol, mask):
        z = torch.cat([F.one_hot(sol - 1, N).float(), mask.float()[..., None]], -1).view(len(sol), -1)
        return self.net(z).squeeze(-1)


class M2(nn.Module):
    def __init__(self, T, planted_spec, atoms, hash_key_seed=12345, fixed_plant=None):
        super().__init__()
        self.T = T
        self.fixed_plant = fixed_plant  # positive control: planted slot = [y_c <= 3] for a fixed cell c
        self.net = B2(C, N, d=64, heads=4, nslot=len(SLOT_NAMES))
        self.phis = nn.ModuleList([Phi() for _ in range(K_LEARN)])
        self.atoms = atoms
        self.planted_id = atoms.specs.index(tuple(planted_spec))
        g = torch.Generator().manual_seed(hash_key_seed)
        self.register_buffer("hkey", torch.randint(0, 2, (C, N + 1), generator=g))
        self.register_buffer("listed", torch.ones(len(SLOT_NAMES), dtype=torch.bool))

    def events(self, bt, hard_only=False):
        """(B,K) straight-through event values (learned) / fixed values (references)."""
        sol, mask, force, x = bt["sol"], bt["mask"], bt["force"], bt["x"]
        B = len(sol)
        ev = []
        for k in range(K_LEARN):
            phi = self.phis[k](sol, mask)
            hard = (phi > 0).float()
            s = torch.sigmoid(phi)
            ev.append(hard if hard_only else hard + s - s.detach())
        if self.fixed_plant is not None:
            ev.append((sol[:, self.fixed_plant] <= 3).float())
        else:
            ev.append(self.atoms.values(bt, [self.planted_id])[:, 0])
        # cell bit: digit of the most deeply forced masked cell <= 3 (never-forced counts as deepest; ties -> lowest index)
        fd = torch.where(force < 0, torch.full_like(force, 99), force).float()
        ar = torch.arange(C, device=sol.device)
        fd = torch.where(mask, fd, torch.full_like(fd, -1.0)) - 1e-3 * ar[None]
        c = fd.argmax(-1)
        ev.append((sol[torch.arange(B, device=sol.device), c] <= 3).float())
        ev.append((x[:, 0] > 0).float())
        ev.append(torch.randint(0, 2, (B,), device=sol.device).float())
        ev.append((((self.hkey[ar[None].expand(B, -1), sol] * mask).sum(-1)) % 2).float())
        return torch.stack(ev, 1)

    def decode(self, x, ev, listed=None):
        """Phase 2 with value slots for listed slots, const for de-listed. Returns cell logits, prior logits."""
        net = self.net
        B = len(x)
        listed = self.listed if listed is None else listed
        ec = net.embed(x)
        e1 = torch.cat([ec, net.slot_inputs(B, 'query')], 1)
        h, _ = net.run(None, self.T, e=e1)
        pl = net.prior_logits(h)
        sv = net.slot_inputs(B, 'value', ev)
        sc = net.slot_inputs(B, 'const')
        s2 = torch.where(listed[None, :, None], sv, sc)
        e2 = torch.cat([ec, s2], 1)
        _, logits = net.run(None, self.T, h=h, e=e2)
        return logits, pl, (h, ec, sc)

    def decode_from(self, cache, ev, listed):
        net = self.net
        h, ec, sc = cache
        B = len(ec)
        sv = net.slot_inputs(B, 'value', ev)
        s2 = torch.where(listed[None, :, None], sv, sc)
        _, logits = net.run(None, self.T, h=h, e=torch.cat([ec, s2], 1))
        return logits


@torch.no_grad()
def royalties(model, bt):
    """Per-slot held-out royalty G_k = E_{e~_k ~ p(e_k|X)}[l(Y|X,e_-k,e~_k)] - l(Y|X,e) - l(e_k|X), nats per cell,
    plus its parts. Computed for every slot as if listed (de-listed slots are reported but do not change the others'
    context: the context e_-k uses the currently listed set)."""
    ev = model.events(bt, hard_only=True)
    logits, pl, cache = model.decode(bt["x"], ev)
    base = cell_nll(logits, bt["y"], bt["mask"])
    out = {}
    for k in range(len(SLOT_NAMES)):
        lst = model.listed.clone()
        lst[k] = True
        e1, e0 = ev.clone(), ev.clone()
        e1[:, k], e0[:, k] = 1.0, 0.0
        ek = ev.clone()
        lk = model.decode_from(cache, ek, lst)
        nl_true = cell_nll(lk, bt["y"], bt["mask"])
        nl1 = cell_nll(model.decode_from(cache, e1, lst), bt["y"], bt["mask"])
        nl0 = cell_nll(model.decode_from(cache, e0, lst), bt["y"], bt["mask"])
        p = torch.sigmoid(pl[:, k])
        cf = p * nl1 + (1 - p) * nl0
        rate = F.binary_cross_entropy_with_logits(pl[:, k], ev[:, k], reduction='none')
        lst_off = lst.clone(); lst_off[k] = False
        nl_off = cell_nll(model.decode_from(cache, ev, lst_off), bt["y"], bt["mask"])
        out[SLOT_NAMES[k]] = {"G": float((cf - nl_true - rate).mean()) / C,
                              "G_per_ex": ((cf - nl_true - rate) / C).cpu().numpy(),
                              "gain_resample": float((cf - nl_true).mean()) / C,
                              "gain_vs_const": float((nl_off - nl_true).mean()) / C,
                              "rate": float(rate.mean()) / C, "p1": float(ev[:, k].mean())}
    return out, float(base.mean()) / C


# ------------------------------------------------------------------ item 2 (batch-2 follow-up): init / warm-up fixes
WARM = int(os.environ.get("P10_WARM", "4000"))    # warm-up: no rate gradient into a learned encoder for WARM steps
RAMP = int(os.environ.get("P10_RAMP", "2000"))    # ... then its rate gradient ramps linearly to full over RAMP steps
CKPT = os.environ.get("P10_CKPT", f"{TMP}/ckpt")
CKPT_EVERY = int(os.environ.get("P10_CKPT_EVERY", "2500"))
EVAL_SPLIT = os.environ.get("P10_EVAL_SPLIT", "test")  # smoke/timing runs before the pre-registration use "val"


def parse_arm(arm):
    """'<base>[_gN][_fixC][_init][_warm][_both]', base in auction|cvae|reg|reg2. P10's arm names parse as before.
    init = centred encoder init (at start and at every respawn); warm = rate warm-up per learned slot (from its spawn);
    norate = diagnostic: the rate never sends gradient into a learned encoder (warm-up for the whole run)."""
    tok = arm.split("_")
    f = {"base": tok[0], "fix": None, "grace": 1000, "init": False, "warm": False, "norate": False}
    assert f["base"] in ("auction", "cvae", "reg", "reg2"), arm
    for t in tok[1:]:
        if t.startswith("fix"):
            f["fix"] = int(t[3:])
        elif t[0] == "g" and t[1:].isdigit():
            f["grace"] = int(t[1:])
        elif t in ("init", "warm", "norate"):
            f[t] = True
        elif t == "both":
            f["init"] = f["warm"] = True
        else:
            raise SystemExit(f"unknown arm token {t} in {arm}")
    return f


def cal_batch(d, key, B=1024):
    """Augmented train batch from its own generator, so the training data stream is the same as without it."""
    g0 = d.g
    d.g = torch.Generator().manual_seed(key)
    try:
        return d.batch("train", B)
    finally:
        d.g = g0


@torch.no_grad()
def centre_phi(phi, cal):
    """Data-dependent init of the encoder's last layer (as in LSUV / weight-norm init): phi <- (phi - median) / SD
    on a calibration batch, so the hard event (phi > 0) splits the batch in half at step 0."""
    z = phi(cal["sol"], cal["mask"])
    m, s = z.median(), z.std().clamp_min(1e-8)
    last = phi.net[-1]
    last.weight.div_(s)
    last.bias.sub_(m).div_(s)


def build_m2(arm, T, seed):
    f = parse_arm(arm)
    torch.manual_seed(seed)
    d = S6(seed)
    at = Atoms()
    bank_res = json.load(open(f"{RES}/s6_bank_T{T}_seed0.json"))
    planted = bank_res["planted"]["spec"]
    # "_fixC" = positive control (post-hoc in P10): the planted slot becomes [y_C <= 3] for fixed cell C
    model = M2(T, planted, at, fixed_plant=f["fix"]).to(DEV)
    if f["fix"] is not None:
        planted = ["fixed_cell_le3", f["fix"]]
    if f["init"]:
        for k in range(K_LEARN):
            centre_phi(model.phis[k], cal_batch(d, 1_000_003 * seed + 1009 * 0 + k))
    return f, d, at, model, planted


def m2init(arm, T, seed):
    """Gate G0 measurement (no training): P10's test of the learned encoders at step 0, p1 = mean hard event."""
    f, d, at, model, planted = build_m2(arm, T, seed)
    out = {"arm": arm, "T": T, "seed": seed, "flags": f}
    for split in ("val", EVAL_SPLIT) if EVAL_SPLIT != "val" else ("val",):
        bt = d.batch(split, idx=d.idx[split], aug=False)
        with torch.no_grad():
            z = [model.phis[k](bt["sol"], bt["mask"]) for k in range(K_LEARN)]
        out[split] = [{"p1": float((zk > 0).float().mean()), "phi_mean": float(zk.mean()), "phi_sd": float(zk.std())}
                      for zk in z]
    _dump(out, f"{RES}/m2init_{arm}_T{T}_seed{seed}.json")
    print("m2init", arm, seed, [round(r["p1"], 4) for r in out[EVAL_SPLIT]], flush=True)


def _dump(obj, path):
    json.dump(obj, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)


def m2(arm, T, seed, steps=None):
    """arm: 'auction' (fee, de-list, respawn), 'cvae' (no auction, prune at end), 'reg' (B2-reg, slots constant),
    plus the item-2 flags of parse_arm. Checkpoints every CKPT_EVERY steps and resumes from the checkpoint."""
    steps = steps or int(os.environ.get("P10_S6_STEPS", "3000"))
    f, d, at, model, planted = build_m2(arm, T, seed)
    fix, base_arm = f["fix"], f["base"]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    held = d.batch("val", idx=d.idx["val"][:512], aug=False)
    t0 = time.time()
    from auction import Auction
    # "_g5000" = the same auction with a 5000-step grace (post-hoc diagnostic in P10); with warm-up a learned slot is
    # judged only once its rate is fully on: its grace is grace + WARM + RAMP
    grace = f["grace"]
    is_auc = base_arm == "auction"
    auc = Auction(SLOT_NAMES, K_LEARN, FEE, ema=0.5, grace=grace, every=500,
                  grace_learn=grace + WARM + RAMP if f["warm"] else None)
    no_slots = torch.zeros(len(SLOT_NAMES), dtype=torch.bool, device=DEV)
    dead_phis = []
    init_phis = [{kk: v.clone() for kk, v in p.state_dict().items()} for p in model.phis]
    p1_trace = []
    start, elapsed0 = 0, 0.0
    os.makedirs(CKPT, exist_ok=True)
    ck_path = f"{CKPT}/m2_{arm}_T{T}_seed{seed}.pt"
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); auc.__dict__.update(ck["auc"])
        dead_phis, init_phis, p1_trace = ck["dead_phis"], ck["init_phis"], ck["p1_trace"]
        torch.set_rng_state(ck["rng"]); d.g.set_state(ck["rng_data"])
        start, elapsed0 = ck["step"] + 1, ck["elapsed"]
        print(f"[m2 {arm} T={T} s{seed}] resumed at step {start}", flush=True)
    stop_at = int(os.environ.get("P10_STOP_AT", "0"))  # test hook: exit after this step (to test resume)
    t0 -= elapsed0
    B = 128
    for step in range(start, steps):
        cosine_lr(opt, step, steps, 2e-3)
        bt = d.batch("train", B)
        if base_arm == "reg":
            logits, _, _ = model.net.run_slots(bt["x"], T, reg=True)
            loss = cell_nll(logits, bt["y"], bt["mask"]).mean() / C
        elif base_arm == "reg2":
            # B2-reg with distinct learned constants in phase 1 (query) and phase 2 (const): M2 with every slot de-listed
            logits, _, _ = model.decode(bt["x"], torch.zeros(len(bt["x"]), len(SLOT_NAMES), device=DEV), listed=no_slots)
            loss = cell_nll(logits, bt["y"], bt["mask"]).mean() / C
        else:
            ev = model.events(bt)
            logits, pl, _ = model.decode(bt["x"], ev)
            ev_r = ev
            if f["warm"] or f["norate"]:
                # same rate value; the gradient into learned encoder k is scaled by lam_k(age) (the prior always learns)
                lam = [0.0 if f["norate"] else min(1.0, max(0.0, (step - auc.born[SLOT_NAMES[k]] - WARM) / RAMP))
                       for k in range(K_LEARN)]
                lam = torch.tensor(lam + [1.0] * (len(SLOT_NAMES) - K_LEARN), device=DEV)
                ev_r = ev.detach() + lam[None] * (ev - ev.detach())
            rate = F.binary_cross_entropy_with_logits(pl, ev_r, reduction='none') * model.listed[None].float()
            loss = (cell_nll(logits, bt["y"], bt["mask"]) + rate.sum(-1)).mean() / C
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if base_arm in ("auction", "cvae") and (step % 500 == 0 or (step < 2000 and step % 50 == 0)):  # finer early: collapse is fast
            with torch.no_grad():
                p1_trace.append([step] + [float((model.phis[k](held["sol"], held["mask"]) > 0).float().mean())
                                          for k in range(K_LEARN)])
        if is_auc and auc.due(step):
            model.eval()
            roy, base = royalties(model, held)
            model.train()
            actions = auc.settle(step, {k: v["G"] for k, v in roy.items()},
                                 extra={"base_nll_per_cell": base, "rate": {k: v["rate"] for k, v in roy.items()},
                                        "gain": {k: v["gain_resample"] for k, v in roy.items()}})
            for act, k, name in actions:
                if act == "respawn":
                    dead_phis.append({"slot": name, "step": step,
                                      "state": {kk: v.clone() for kk, v in model.phis[k].state_dict().items()}})
                    fresh = Phi().to(DEV)
                    model.phis[k].load_state_dict(fresh.state_dict())
                    if f["init"]:
                        centre_phi(model.phis[k], cal_batch(d, 1_000_003 * seed + 1009 * step + k))
                    for p in model.phis[k].parameters():
                        opt.state.pop(p, None)
                    with torch.no_grad():
                        model.net.slot_val[k].normal_(0, 0.02)
                        model.net.prior_w[k].zero_(); model.net.prior_b[k].zero_()
                    for p in (model.net.slot_val, model.net.prior_w, model.net.prior_b):
                        st = opt.state.get(p)
                        if st:
                            for key in ("exp_avg", "exp_avg_sq"):
                                if key in st:
                                    st[key][k].zero_()
                else:
                    model.listed[k] = False
            print(f"[m2 {arm} T={T} s{seed}] step {step} base {base:.4f} listed {model.listed.int().tolist()} "
                  f"G {[round(roy[n]['G'], 4) for n in SLOT_NAMES]} ({time.time()-t0:.0f}s)", flush=True)
        elif step % 500 == 0:
            print(f"[m2 {arm} T={T} s{seed}] step {step} loss/cell {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % CKPT_EVERY == 0 and step + 1 < steps:
            torch.save({"step": step, "elapsed": time.time() - t0, "model": model.state_dict(), "opt": opt.state_dict(),
                        "auc": dict(auc.__dict__), "dead_phis": dead_phis, "init_phis": init_phis, "p1_trace": p1_trace,
                        "rng": torch.get_rng_state(), "rng_data": d.g.get_state()}, ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
        if stop_at and step == stop_at:
            raise SystemExit(f"stopped at {step} (P10_STOP_AT)")
    model.eval()
    train_time = time.time() - t0
    out = {"arm": arm, "T": T, "seed": seed, "steps": steps, "fee": FEE, "train_time_s": train_time, "planted": planted,
           "flags": f, "warm_steps": WARM, "ramp_steps": RAMP, "grace_learn": auc.grace_learn, "p1_trace_val": p1_trace,
           "resumed_from": start}
    test = d.batch(EVAL_SPLIT, idx=d.idx[EVAL_SPLIT], aug=False)
    out["eval_split"] = EVAL_SPLIT
    if base_arm in ("reg", "reg2"):
        with torch.no_grad():
            if base_arm == "reg":
                logits, _, _ = model.net.run_slots(test["x"], T, reg=True)
            else:
                logits, _, _ = model.decode(test["x"], torch.zeros(len(test["x"]), len(SLOT_NAMES), device=DEV),
                                            listed=no_slots)
        nl = cell_nll(logits, test["y"], test["mask"]) / C
        out["test_nll_per_cell"] = float(nl.mean())
        np.save(f"{RES}/m2_{arm}_T{T}_s{seed}_nll.npy", nl.cpu().numpy().astype(np.float32))
    else:
        if base_arm == "cvae":
            roy, _ = royalties(model, held)
            # prune slots whose held-out (val) rate exceeds their leave-one-out decoder gain
            keep = torch.tensor([roy[n]["gain_resample"] > roy[n]["rate"] for n in SLOT_NAMES])
            out["cvae_pruned"] = [n for n, kp in zip(SLOT_NAMES, keep.tolist()) if not kp]
            model.listed[:] = keep
        roy, base = royalties(model, test)
        out["test_royalty"] = {n: {kk: v for kk, v in r.items() if kk != "G_per_ex"} for n, r in roy.items()}
        # bootstrap CI of G_k over test examples
        from stats import boot_ci
        for n in SLOT_NAMES:
            out["test_royalty"][n]["G_ci"] = boot_ci(roy[n]["G_per_ex"])
        out["listed_final"] = dict(zip(SLOT_NAMES, model.listed.tolist()))
        out["surviving_G_total"] = float(sum(roy[n]["G"] for k, n in enumerate(SLOT_NAMES) if model.listed[k]))
        # end-to-end test: e_hat = argmax prior (listed slots)
        with torch.no_grad():
            ev_true = model.events(test, hard_only=True)
            _, pl, cache = model.decode(test["x"], ev_true)
            ehat = (pl > 0).float()
            lg = model.decode_from(cache, ehat, model.listed)
            nl = cell_nll(lg, test["y"], test["mask"]) / C
            lg_t = model.decode_from(cache, ev_true, model.listed)
            nl_t = cell_nll(lg_t, test["y"], test["mask"]) / C
        out["test_nll_ehat_per_cell"] = float(nl.mean())
        out["test_nll_teacher_per_cell"] = float(nl_t.mean())
        np.save(f"{RES}/m2_{arm}_T{T}_s{seed}_nll.npy", nl.cpu().numpy().astype(np.float32))
        # NMI of learned slots (live, dead, random init) with atoms on the test set
        vals = at.values_all(test).cpu().numpy().astype(np.int64)
        ncs = at.ncell_start

        def nmi_rec(phi_state):
            ph = Phi().to(DEV); ph.load_state_dict(phi_state)
            with torch.no_grad():
                e = (ph(test["sol"], test["mask"]) > 0).cpu().numpy().astype(np.int64)
            return nmi_max(e, vals, ncs)
        out["nmi"] = {"live": [], "dead": [], "init": []}
        for k in range(K_LEARN):
            r = nmi_rec(model.phis[k].state_dict()); r["slot"] = SLOT_NAMES[k]; r["listed"] = bool(model.listed[k])
            r["G"] = roy[SLOT_NAMES[k]]["G"]
            out["nmi"]["live"].append(r)
        for dp in dead_phis:
            r = nmi_rec(dp["state"]); r["slot"] = dp["slot"]; r["step"] = dp["step"]
            out["nmi"]["dead"].append(r)
        for s in init_phis:
            out["nmi"]["init"].append(nmi_rec(s))
    out["log"] = auc.log
    out["books"] = auc.books() if is_auc else None
    out["grace"] = grace
    _dump(out, f"{RES}/m2_{arm}_T{T}_seed{seed}.json")
    if os.path.exists(ck_path):
        os.unlink(ck_path)  # finished: the checkpoint is no longer needed
    print("done", arm, seed, round(time.time() - t0), "s", flush=True)
    return out


def _H(p):
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def nmi_bin(e, a):
    """NMI = I(e;a)/sqrt(H(e)H(a)) for binary arrays."""
    j = np.bincount(e * 2 + a, minlength=4).astype(float) / len(e)
    pe = np.array([j[0] + j[1], j[2] + j[3]]); pa = np.array([j[0] + j[2], j[1] + j[3]])
    He, Ha = _H(pe), _H(pa)
    if He < 1e-9 or Ha < 1e-9:
        return 0.0
    return max(0.0, (He + Ha - _H(j)) / math.sqrt(He * Ha))


def nmi_max(e, vals, ncs):
    nm = np.array([nmi_bin(e, vals[:, a]) for a in range(vals.shape[1])])
    return {"p1": float(e.mean()), "max_nmi_noncell": float(nm[:ncs].max()), "argmax_noncell": int(nm[:ncs].argmax()),
            "max_nmi_cell": float(nm[ncs:].max()), "argmax_cell": int(ncs + nm[ncs:].argmax())}


def main(argv):
    what = argv[0]
    if what == "s6pilot":
        pilot(int(argv[1]), int(argv[2]))
    elif what == "s6bank":
        bank(int(argv[1]), int(argv[2]))
    elif what == "m2":
        m2(argv[1], int(argv[2]), int(argv[3]))
    elif what == "m2init":
        m2init(argv[1], int(argv[2]), int(argv[3]))
