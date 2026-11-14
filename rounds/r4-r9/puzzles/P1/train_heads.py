"""Fit the head arms on detached rollouts of a frozen backbone.

ArmSet holds one head per arm (one head seed). fit_batch() trains every arm on the
decision points of one batch of rollouts (targets computed lazily from the stored
logits). Test-time fitting (TT) uses the same code with label-free targets only
(R7TT is the labelled reference and reads labels from a separate labelled copy).
"""
import copy
import math
import torch
import torch.nn.functional as F
from heads import (make_head, perpos_targets, logq, HORIZONS, KL_HORIZONS, horizon_index,
                   n_params)

LOG_HALF = math.log(0.5)


def _prob(Z, t):
    return torch.sigmoid(Z[t].float())


class ArmSet:
    def __init__(self, arms, cin, dim, seed, device, lr=1e-3, c_range=(1e-3, 10.0), delta=1):
        g = torch.Generator().manual_seed(1000 + seed)
        torch.manual_seed(1000 + seed)
        self.arms, self.dim, self.device, self.delta = dict(arms), dim, device, delta
        self.heads, self.opts, self.norm, self.tgt = {}, {}, {}, {}
        for n, (kind, nout, settle) in self.arms.items():
            self.add(n, kind, nout, settle, cin, lr)
        self.gen = g
        self.c_range = c_range
        self.nstep = 0

    def add(self, n, kind, nout, settle, cin, lr=1e-3):
        h = make_head(kind, nout, cin, self.dim, ablate=(n == "R7abl")).to(self.device)
        self.heads[n] = h
        self.arms[n] = (kind, nout, settle)
        self.opts[n] = torch.optim.Adam(h.parameters(), lr=lr)
        if kind == "qact":
            self.tgt[n] = copy.deepcopy(h).eval()

    def clone_arm(self, src, dst, settle=None, lr=5e-4):
        """TT arm = a copy of a base arm that continues fitting (new optimizer)."""
        kind, nout, s = self.arms[src]
        self.heads[dst] = copy.deepcopy(self.heads[src])
        self.arms[dst] = (kind, nout, settle or s)
        self.opts[dst] = torch.optim.Adam(self.heads[dst].parameters(), lr=lr)
        self.norm[dst] = self.norm.get(src)

    def params(self):
        return {n: n_params(h) for n, h in self.heads.items()}

    def modules(self, names=None):
        return {n: (self.heads[n], self.arms[n][0]) for n in (names or self.heads)
                if self.arms[n][2] not in ("ponder30", "ponder100") or True}

    # ------------------------------------------------------------ targets
    def targets_at(self, Z, t, Tc, y, yhat, emask, need):
        out = {}
        kinds = [k for k in ("G", "K", "P") if k in need]
        if (("G" in need) and y is None):
            raise ValueError("labelled target requested without labels")
        out.update(perpos_targets(Z, t, Tc, y=y, yhat=yhat, kinds=kinds))
        p = _prob(Z, t)
        if "exact" in need:
            pred = (p > 0.5) & emask
            out["exact"] = (pred.float() == y).flatten(1).all(1).float()
        if "cons" in need:
            out["cons"] = ((p > 0.5) == (yhat > 0.5)).flatten(1).all(1).float()
        if "s" in need:
            out["s"] = logq(p, y) - LOG_HALF
        return out

    @staticmethod
    def _need(settle):
        return {"G1": {"G"}, "G": {"G"}, "K": {"K"}, "P": {"P"}, "exact": {"exact"},
                "cons": {"cons"}, "qact": {"s"}}.get(settle, set())

    def _init_norm(self, n, tg, settle):
        if n in self.norm and self.norm[n] is not None:
            return
        if settle in ("G", "G1", "K", "P"):
            T = tg[{"G1": "G"}.get(settle, settle)]
            T = T[:, :1] if settle == "G1" else T
            sd = T.transpose(0, 1).flatten(1).float().std(1).clamp(min=1e-6)
            self.norm[n] = sd.view(1, -1, *([1] * self.dim))
        elif settle == "qact":
            s = tg["s"]
            self.norm[n] = (s.float().std().clamp(min=1e-6), s.flatten(1).sum(1).float().std().clamp(min=1e-3))

    # ------------------------------------------------------------ one decision point
    def step_arm(self, n, f, tg, Z=None, t=None, Tc=None, Fnext=None):
        kind, nout, settle = self.arms[n]
        h = self.heads[n]
        self._init_norm(n, tg, settle)
        if settle in ("G", "G1", "K", "P"):
            T = tg[{"G1": "G"}.get(settle, settle)]
            T = T[:, :1] if settle == "G1" else T
            o, _ = h(f)
            loss = (((o - T) / self.norm[n]) ** 2).mean()
        elif settle in ("exact", "cons"):
            logit, _ = h.totals(f)
            loss = F.binary_cross_entropy_with_logits(logit[:, 0], tg[settle])
        elif settle == "qact":
            sd_s, sd_U = self.norm[n]
            o, pooled = h(f)
            B = f.shape[0]
            npos = o[0, 0].numel()
            loss_h = (((o[:, 0] - tg["s"]) / sd_s) ** 2).mean()
            U = o[:, 0].flatten(1).sum(1)
            lo, hi = math.log(self.c_range[0]), math.log(self.c_range[1])
            logc = lo + (hi - lo) * torch.rand(B, device=f.device)
            c = logc.exp()
            A = o[:, 1].flatten(1).sum(1) + h.qm(torch.cat([pooled, logc[:, None]], 1)).squeeze(1) * npos
            with torch.no_grad():
                tn = self.tgt[n]
                ot, pt = tn(Fnext.float())
                Un = ot[:, 0].flatten(1).sum(1)
                if t + self.delta >= Tc:
                    y = Un - c * (Tc - t)
                else:
                    An = ot[:, 1].flatten(1).sum(1) + tn.qm(torch.cat([pt, logc[:, None]], 1)).squeeze(1) * npos
                    y = Un + torch.clamp(An, min=0) - c * self.delta
            loss_c = (((U.detach() + A - y) / sd_U) ** 2).mean()
            loss = loss_h + loss_c
        else:
            raise ValueError(settle)
        self.opts[n].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
        self.opts[n].step()
        return loss.item()

    def fit_batch(self, Fd, Z, D, Tc, y, emask, names, passes=2, sub=None):
        """Fd: dict t -> features; D: decision times; names: arms to fit. y may be None
        (label-free arms only). sub: optional index subset of the batch."""
        yhat = (_prob(Z, Tc) > 0.5).float()
        need = set()
        for n in names:
            need |= self._need(self.arms[n][2])
        losses = {n: [] for n in names}
        # target scales from all decision times of the first batch (not one random time, which can be degenerate)
        missing = [n for n in names if self.norm.get(n) is None and self.arms[n][2] in ("G", "G1", "K", "P", "qact")]
        if missing:
            tgs = [self.targets_at(Z, t, Tc, y, yhat, emask, need) for t in D]
            pooled = {k: torch.cat([tg[k] for tg in tgs], 0) for k in tgs[0]}
            for n in missing:
                self._init_norm(n, pooled, self.arms[n][2])
            del tgs, pooled
        for _ in range(passes):
            order = torch.randperm(len(D), generator=self.gen).tolist()
            for k in order:
                t = D[k]
                f = Fd[t].float()
                tg = self.targets_at(Z, t, Tc, y, yhat, emask, need)
                for n in names:
                    if self.arms[n][2] == "qact":
                        tn = min(t + self.delta, Tc)
                        losses[n].append(self.step_arm(n, f, tg, Z, t, Tc, Fd[tn]))
                    else:
                        losses[n].append(self.step_arm(n, f, tg))
                self.nstep += 1
                if self.nstep % 25 == 0:
                    for n in names:
                        if n in self.tgt:
                            self.tgt[n].load_state_dict(self.heads[n].state_dict())
        return {n: (sum(v) / max(1, len(v))) for n, v in losses.items()}

    def fit_ponder(self, Fall, Z, Tc, y, names, chunk=4096):
        """PonderNet on the frozen backbone: lambda_t = sigmoid(global logit), lambda_Tc = 1.
        loss = sum_t p_t L_t + beta KL(p || truncated geometric(lambda_p))."""
        B = y.shape[0]
        stack = torch.stack([Fall[t] for t in range(1, Tc + 1)], 0)       # (Tc, B, C, *S)
        Lt = torch.stack([-(logq(_prob(Z, t), y)).flatten(1).mean(1) for t in range(1, Tc + 1)], 0)  # (Tc,B)
        out = {}
        for n in names:
            lam_p = {"ponder30": 1 / 30, "ponder100": 1 / 100}[self.arms[n][2]]
            h = self.heads[n]
            flat = stack.flatten(0, 1).float()
            logits = torch.cat([h.totals(flat[i:i + chunk])[0][:, 0] for i in range(0, flat.shape[0], chunk)])
            lam = torch.sigmoid(logits).view(Tc, B)
            lam = torch.cat([lam[:-1], torch.ones(1, B, device=lam.device)], 0)
            surv = torch.cumprod(torch.cat([torch.ones(1, B, device=lam.device), 1 - lam[:-1]], 0), 0)
            p = lam * surv
            ts = torch.arange(1, Tc + 1, device=lam.device).float()
            g = lam_p * (1 - lam_p) ** (ts - 1)
            g = (g / g.sum()).view(Tc, 1)
            kl = (p * (torch.log(p.clamp(min=1e-10)) - torch.log(g))).sum(0)
            loss = ((p * Lt).sum(0) + 0.01 * kl).mean()
            self.opts[n].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            self.opts[n].step()
            out[n] = loss.item()
        return out

    def state(self):
        return {"heads": {n: h.state_dict() for n, h in self.heads.items()}, "arms": self.arms,
                "norm": {n: (v if not isinstance(v, tuple) else tuple(x.item() for x in v))
                         for n, v in self.norm.items()}}


# ---------------------------------------------------------------- capacity gate helpers
def spearman(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    ra = torch.empty_like(a); ra[a.argsort()] = torch.arange(len(a), dtype=torch.float64)
    rb = torch.empty_like(b); rb[b.argsort()] = torch.arange(len(b), dtype=torch.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


@torch.no_grad()
def capacity_stats(armset, names, Fd, Z, D, Tc, y, emask, acc):
    """Accumulate predicted vs realized totals at each horizon for label (G) and KL arms."""
    yhat = (_prob(Z, Tc) > 0.5).float()
    for t in D:
        f = Fd[t].float()
        tg = armset.targets_at(Z, t, Tc, y, yhat, emask, {"G", "K"})
        for n in names:
            settle = armset.arms[n][2]
            tot, _ = armset.heads[n].totals(f)
            real = tg["G" if settle in ("G", "G1") else "K"].flatten(2).sum(2)
            if settle == "G1":
                real = real[:, :1]
            acc.setdefault(n, {"pred": [], "real": []})
            acc[n]["pred"].append(tot.cpu())
            acc[n]["real"].append(real.cpu())
    return acc


def capacity_report(acc):
    rep = {}
    for n, d in acc.items():
        P = torch.cat(d["pred"]); R = torch.cat(d["real"])
        hs = HORIZONS if P.shape[1] == 8 else (KL_HORIZONS if P.shape[1] == 7 else [1])
        rep[n] = {str(H): {"spearman": spearman(P[:, j], R[:, j]),
                           "mean_ratio": float(P[:, j].mean() / (R[:, j].mean() + 1e-12)),
                           "pred_mean": float(P[:, j].mean()), "real_mean": float(R[:, j].mean())}
                  for j, H in enumerate(hs)}
    return rep
