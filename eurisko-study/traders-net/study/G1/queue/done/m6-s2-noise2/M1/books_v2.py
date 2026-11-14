"""books_v2: three settlement books for the question transformer (traders-net M1 item c).

Worker 3 of M1, 2026-09-27. Model: S0 substrate model.Net (verbatim wealth-net copy). Units
(the "traders"): readout per question (out [Q]), attention head (head [L,H]), MLP neuron
(neuron [L,N]), residual component (comp [2+2L]: e_w, e_q, attn_l, mlp_l). Every book is split by
the question family whose settlement paid it (last axis F).

All books use one forward pass with LayerNorm denominators detached (freeze_ln, as wealth-net v1):
a unit's contribution to logit z_q is c = a . dz_q/da of this frozen-LN surrogate.

Books (per example, per unit; summed over the batch and divided by B):
  tangent   c . s_t,  s_t = y - sigmoid(z) at the TRADE price. Call `account` on a block BEFORE the
            optimiser has used its labels (prequential); calling it on trained data gives the
            in-sample v1 books. Mean zero at calibration (T1 lemma 1.4).
  lmsr      c . (y - p_bar): exact LMSR settlement of the move from the prior logit l0 to z
            (T1 lemma 2.2). l0 = logit of the running base rate of question q, (n1+1)/(n+2) over labels
            revealed so far (prequential; never includes the current block). p_bar = average execution
            price over a 16-step midpoint path l0 -> z; the closed form (sp(z)-sp(l0))/(z-l0) is
            carried as a check (max abs difference in `diag`). The readout book is the exact
            log-score gain log p_z(y) - log p_l0(y); the maker account -l0 (y - p_bar) plus the bias
            and first-order-residual accounts close the identity sum = gain.
  meanbase  (a - a_bar) . g_t: tangent gradient g_t times the activation's deviation from its running
            mean a_bar (per unit, per question token, per head dim; mean over all past booked
            examples; frozen before the current block). The constant part a_bar . g_t goes to a
            `meanbase_const` account (not a trader). Readout: (z - z_bar_q) s_t.
Extras: meanbase_lmsr ((a - a_bar) . g_lmsr), absag (sum over examples of |c . s_t| per unit,
family-summed, for M2/M3's |a.g| predictor), act_count.

Use:
  bk = BooksV2(model, fam, n_fam)
  acc = bk.account(win, y, mask)    # pure; returns dict of tensors
  bk.observe(win, y, mask)          # after booking: update base-rate prior and activation means
  bk.add(acc)                       # accumulate into bk.W[book][gran]
"""
import math
import torch
import torch.nn.functional as F

GRANS = ("out", "head", "neuron", "comp")
BOOKS = ("tangent", "lmsr", "meanbase", "meanbase_lmsr")
N_PATH = 16


def p_bar_path(l0, z, n=N_PATH):
    """Average execution price over the straight path l0 -> z, n-step midpoint rule."""
    t = (torch.arange(n, dtype=z.dtype, device=z.device) + 0.5) / n
    return torch.sigmoid(l0.unsqueeze(-1) + t * (z - l0).unsqueeze(-1)).mean(-1)


def p_bar_closed(l0, z):
    d = z - l0
    small = d.abs() < 1e-4
    pb = (F.softplus(z) - F.softplus(l0)) / torch.where(small, torch.ones_like(d), d)
    return torch.where(small, torch.sigmoid(0.5 * (z + l0)), pb)


def log_score_gain(l0, z, y):
    """log p_z(y) - log p_l0(y) with log p_l(y) = y l - softplus(l)."""
    return y * (z - l0) - F.softplus(z) + F.softplus(l0)


class BooksV2:
    def __init__(self, model, fam, n_fam, prior_a=1.0, prior_b=1.0, groups=None, batched=False):
        """groups: optional list of lists of family ids (a partition); the last axis of every book is
        then the group, not the family (fewer backward passes). Default: one group per family."""
        self.m = model
        dev = model.rw.device
        self.fam = torch.as_tensor(fam, dtype=torch.long, device=dev)
        foh = F.one_hot(self.fam, n_fam).float()                            # [Q,n_fam]
        if groups is None:
            groups = [[f] for f in range(n_fam)]
        self.groups = groups
        self.foh = torch.stack([foh[:, g].sum(1) for g in groups], 1)       # [Q,G]
        assert torch.all(self.foh.sum(1) == 1), "groups must partition the families"
        self.batched = batched
        n_fam = len(groups)
        self.n_fam = n_fam
        Q, L, H, d = model.Q, model.L, model.H, model.d
        N = model.blocks[0].fc1.out_features
        self.Q, self.L, self.H, self.N, self.dh = Q, L, H, N, d // H
        self.n1 = torch.zeros(Q, device=dev); self.n = torch.zeros(Q, device=dev)
        self.pa, self.pb = prior_a, prior_b
        # running means for the mean baseline (sum and count over booked examples)
        self.cnt = 0.0
        self.s_act = torch.zeros(L, Q, N, device=dev)
        self.s_hv = torch.zeros(L, H, Q, self.dh, device=dev)
        self.s_ew = torch.zeros(Q, d, device=dev)
        self.s_ao = torch.zeros(L, Q, d, device=dev)
        self.s_mo = torch.zeros(L, Q, d, device=dev)
        self.s_z = torch.zeros(Q, device=dev)
        shapes = {"out": (Q,), "head": (L, H), "neuron": (L, N), "comp": (2 + 2 * L,)}
        self.W = {b: {g: torch.zeros(*shapes[g], n_fam, device=dev) for g in GRANS} for b in BOOKS}
        self.W["absag"] = {g: torch.zeros(*shapes[g], device=dev) for g in GRANS}
        self.W["acct"] = {k: torch.zeros(n_fam, device=dev) for k in
                          ("gain", "R_tangent", "maker", "meanbase_const", "bias_tangent", "bias_lmsr")}
        self.steps = 0

    # ---- prior and running means (prequential state) ----
    def prior_logit(self):
        p = (self.n1 + self.pa) / (self.n + self.pa + self.pb)
        return torch.log(p) - torch.log1p(-p)

    def _means(self):
        c = max(self.cnt, 1.0)
        return (self.s_act / c, self.s_hv / c, self.s_ew / c, self.s_ao / c, self.s_mo / c, self.s_z / c)

    @torch.no_grad()
    def observe(self, win, y, mask, rec=None):
        """Update the base-rate prior with revealed labels and the activation means with this block.
        Call AFTER account() for the same block (so the block never sets its own prior/baseline)."""
        m = mask.float()
        self.n1 += (y * m).sum(0); self.n += m.sum(0)
        if rec is None:
            rec = {}
            z = self.m(win, rec=rec, freeze_ln=True)
        else:
            z = rec["z"]
        B = win.shape[0]
        self.cnt += B
        self.s_act += torch.stack([a.sum(0) for a in rec["act"]])
        self.s_hv += torch.stack([h.sum(0) for h in rec["hv"]])
        self.s_ew += rec["e_w"].sum(0)
        self.s_ao += torch.stack([a.sum(0) for a in rec["attn_out"]])
        self.s_mo += torch.stack([a.sum(0) for a in rec["mlp_out"]])
        self.s_z += z.sum(0)

    # ---- the books ----
    def account(self, win, y, mask=None, freeze_attn=False):
        m = self.m
        B, Q, L = win.shape[0], self.Q, self.L
        Fm = self.n_fam
        mk = torch.ones_like(y) if mask is None else mask.float()
        rec = {}
        z = m(win, rec=rec, freeze_ln=True, freeze_attn=freeze_attn)
        zd = z.detach()
        l0 = self.prior_logit().unsqueeze(0).expand(B, -1)
        s_t = (y - torch.sigmoid(zd)) * mk
        pb = p_bar_path(l0, zd)
        s_l = (y - pb) * mk
        cot = torch.cat([s_t.unsqueeze(0) * self.foh.T.unsqueeze(1),
                         s_l.unsqueeze(0) * self.foh.T.unsqueeze(1)])             # [2F,B,Q]
        biases = m.bias_params()
        tg = [rec["e_w"]] + list(rec["hv"]) + list(rec["act"]) + list(rec["attn_out"]) + list(rec["mlp_out"]) \
            + [b for _, b, _ in biases]
        if self.batched:
            g = torch.autograd.grad(z, tg, grad_outputs=cot, is_grads_batched=True)
        else:
            per = [torch.autograd.grad(z, tg, grad_outputs=cot[k], retain_graph=True) for k in range(cot.shape[0])]
            g = [torch.stack([pp[i] for pp in per]) for i in range(len(tg))]
        g_ew, g_hv, g_act = g[0], g[1:1 + L], g[1 + L:1 + 2 * L]
        g_ao, g_mo, g_b = g[1 + 2 * L:1 + 3 * L], g[1 + 3 * L:1 + 4 * L], g[1 + 4 * L:]
        # e_q is a parameter-only input; its grad is the grad at the first cut minus nothing: use cut0 = e_w + e_q,
        # dz/d e_q = dz/d e_w (same node sum), so reuse g_ew.
        a_act, a_hv, a_ew, a_ao, a_mo, a_z = self._means()
        e_q = rec["e_q"].detach()
        out = {}
        T_, Lm = slice(0, Fm), slice(Fm, 2 * Fm)

        def unit(a, gg, red, abar=None):
            a = a.detach()
            v = a.unsqueeze(0) * gg if abar is None else (a - abar).unsqueeze(0) * gg
            return v.sum(red) / B

        # granularity reductions: heads hv [2F,B,H,Q,dh] -> sum (1,3,4) -> [2F,H]
        head = torch.stack([unit(rec["hv"][l], g_hv[l], (1, 3, 4)) for l in range(L)])          # [L,2F,H]
        headm = torch.stack([unit(rec["hv"][l], g_hv[l], (1, 3, 4), a_hv[l]) for l in range(L)])
        neu = torch.stack([unit(rec["act"][l], g_act[l], (1, 2)) for l in range(L)])            # [L,2F,N]
        neum = torch.stack([unit(rec["act"][l], g_act[l], (1, 2), a_act[l]) for l in range(L)])
        comp = [unit(rec["e_w"], g_ew, (1, 2, 3)), unit(e_q, g_ew, (1, 2, 3))]
        compm = [unit(rec["e_w"], g_ew, (1, 2, 3), a_ew), torch.zeros_like(comp[1])]  # e_q constant: all baseline
        for l in range(L):
            comp += [unit(rec["attn_out"][l], g_ao[l], (1, 2, 3)), unit(rec["mlp_out"][l], g_mo[l], (1, 2, 3))]
            compm += [unit(rec["attn_out"][l], g_ao[l], (1, 2, 3), a_ao[l]), unit(rec["mlp_out"][l], g_mo[l], (1, 2, 3), a_mo[l])]
        comp, compm = torch.stack(comp), torch.stack(compm)                                    # [C,2F]
        cot_s = cot.sum(1)                                                                     # [2F,Q] per question
        outb = (zd.unsqueeze(0) * cot).sum(1) / B                                              # [2F,Q]
        outm = ((zd - a_z).unsqueeze(0) * cot).sum(1) / B
        gain = log_score_gain(l0, zd, y) * mk                                                  # [B,Q]
        for name, sl, src in (("tangent", T_, "a"), ("lmsr", Lm, "a"), ("meanbase", T_, "m"), ("meanbase_lmsr", Lm, "m")):
            if src == "a":
                out[name] = {"out": outb[sl].T, "head": head[:, sl].permute(0, 2, 1),
                             "neuron": neu[:, sl].permute(0, 2, 1), "comp": comp[:, sl]}
            else:
                out[name] = {"out": outm[sl].T, "head": headm[:, sl].permute(0, 2, 1),
                             "neuron": neum[:, sl].permute(0, 2, 1), "comp": compm[:, sl]}
        # lmsr readout book = exact log-score gain by construction (question-level): replace the
        # (z)(y-p_bar) readout book by (z - l0)(y - p_bar) = gain; the -l0 (y - p_bar) part is the maker.
        maker = -(l0 * s_l).unsqueeze(0) * self.foh.T.unsqueeze(1)                           # [F,B,Q]
        out["lmsr"]["out"] = out["lmsr"]["out"] + (maker.sum(1) / B).T
        # per-unit |a.g| (tangent, all families): per example, then abs, then sum over batch
        # the tangent family cotangents sum to s_t, so per-example tangent grads are sums over F
        gt = [x[T_].sum(0) for x in list(g_hv) + list(g_act) + [g_ew] + list(g_ao) + list(g_mo)]
        ab = {}
        ab["head"] = torch.stack([(rec["hv"][l].detach() * gt[l]).sum((2, 3)).abs().sum(0) for l in range(L)]) / B
        ab["neuron"] = torch.stack([(rec["act"][l].detach() * gt[L + l]).sum(1).abs().sum(0) for l in range(L)]) / B
        ab["out"] = (zd * s_t).abs().sum(0) / B
        cc = [(rec["e_w"].detach() * gt[2 * L]).sum((1, 2)), (e_q * gt[2 * L]).sum((1, 2))]
        for l in range(L):
            cc += [(rec["attn_out"][l].detach() * gt[2 * L + 1 + l]).sum((1, 2)),
                   (rec["mlp_out"][l].detach() * gt[3 * L + 1 + l]).sum((1, 2))]
        ab["comp"] = torch.stack(cc).abs().sum(1) / B
        out["absag"] = ab
        # accounts and diagnostics
        bias_rev = torch.stack([(b.detach().unsqueeze(0) * gb).reshape(2 * Fm, -1).sum(1) for (_, b, _), gb in zip(biases, g_b)]) / B  # [nb,2F]
        acct = {"gain": (gain.unsqueeze(0) * self.foh.T.unsqueeze(1)).sum((1, 2)) / B,
                "R_tangent": outb[T_].sum(1),
                "maker": maker.sum((1, 2)) / B,
                "meanbase_const": (out["tangent"]["comp"] - out["meanbase"]["comp"]).sum(0),
                "bias_tangent": bias_rev[:, T_].sum(0), "bias_lmsr": bias_rev[:, Lm].sum(0)}
        out["acct"] = acct
        # identity checks (family-summed): gain = out_lmsr; conservation at the input cut (first-order)
        cut0 = comp[0] + comp[1]                                                               # [2F]
        btot = bias_rev[torch.tensor([i >= 0 for _, _, i in biases], device=bias_rev.device)].sum(0)  # biases below cut 0
        R_l = outb[Lm].sum()
        diag = {"gain_minus_out_lmsr": float(acct["gain"].sum() - out["lmsr"]["out"].sum()),
                "pbar_path_vs_closed_maxabs": float((pb - p_bar_closed(l0, zd)).abs().max()),
                "cons_resid_rel_tangent": float((outb[T_].sum() - cut0[T_].sum() - btot[T_].sum()) /
                                                outb[T_].sum().abs().clamp_min(1e-12)),
                "cons_resid_rel_lmsr": float((R_l - cut0[Lm].sum() - btot[Lm].sum()) / R_l.abs().clamp_min(1e-12)),
                "loss": float(F.binary_cross_entropy_with_logits(zd, y, weight=mk, reduction="sum") / mk.sum().clamp_min(1)),
                "gain_mean": float(acct["gain"].sum()), "R_tangent": float(acct["R_tangent"].sum())}
        out["diag"] = diag
        out["_rec"] = {k: (v.detach() if torch.is_tensor(v) else [t.detach() for t in v]) for k, v in rec.items()}
        out["_rec"]["z"] = zd
        return out

    @torch.no_grad()
    def add(self, acc, scale=1.0):
        for b in BOOKS:
            for gname in GRANS:
                self.W[b][gname] += scale * acc[b][gname].detach()
        for gname in GRANS:
            self.W["absag"][gname] += scale * acc["absag"][gname].detach()
        for k, v in acc["acct"].items():
            self.W["acct"][k] += scale * v.detach()
        self.steps += 1

    def flat(self, book, fam=None):
        """Concatenate all granularities (family-summed or one family) as numpy, with slices."""
        import numpy as np
        parts, sl, o = [], {}, 0
        for gname in GRANS:
            x = self.W[book][gname]
            if book != "absag":
                x = x.sum(-1) if fam is None else x[..., fam]
            x = x.reshape(-1).cpu().numpy()
            sl[gname] = slice(o, o + len(x)); o += len(x); parts.append(x)
        return np.concatenate(parts), sl
