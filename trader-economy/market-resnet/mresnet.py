"""Market ResNet: a shared-weight recurrent residual net read as a sequential
prediction market.

Three streams per example:
  x  raw input, embedded once (e) and re-injected at every step ("recall")
  b  belief = output logits, updated additively: b_t = b_{t-1} + db_t
  h  scratch/latent, updated additively: h_t = h_{t-1} + dh_t

Step t is a trader. Its payment is the log (Hanson LMSR) scoring-rule gain
  pay_t = S(b_t) - S(b_{t-1}),  S(b) = sum_i log p_b(y_i),
so payments telescope to S(b_T) - S(b_0).

Arms (see RESULTS.md):
  A    fixed T, loss = final log loss only
  B    A + per-step MSR terms (each step paid for its own trade, price before
       the trade treated as given: stop-grad on b_{t-1})
  Bds  A + standard deep supervision (mean_t logloss(b_t)); a check on B
  C    B + gain head (predicts next step's payment, trained by regression on
       detached features); inference halts when predicted gain <= cost
  D    PonderNet halting (Banino et al. 2021) on the same backbone
  E    random halting with C's step histogram (evaluation-time arm)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 2


# ---------------------------------------------------------------- tasks
def prefix_batch(n, lmin, lmax, gen, device="cpu"):
    """Prefix-sum parity (Schwarzschild et al. 2021): y_i = (sum_{j<=i} x_j) mod 2.
    Lengths uniform in [lmin, lmax]; right-padded with PAD; mask marks real bits."""
    L = torch.randint(lmin, lmax + 1, (n,), generator=gen)
    x = torch.randint(0, 2, (n, lmax), generator=gen)
    pos = torch.arange(lmax).unsqueeze(0)
    mask = pos < L.unsqueeze(1)
    y = torch.cumsum(x, 1) % 2
    x = torch.where(mask, x, torch.full_like(x, PAD))
    y = torch.where(mask, y, torch.zeros_like(y))
    return x.to(device), y.float().to(device), mask.float().to(device)


# ---------------------------------------------------------------- model
class ChanLN(nn.Module):
    """LayerNorm over channels at each position (length independent)."""
    def __init__(self, w):
        super().__init__()
        self.ln = nn.LayerNorm(w)

    def forward(self, h):
        return self.ln(h.transpose(1, 2)).transpose(1, 2)


class MarketResNet(nn.Module):
    def __init__(self, w=128, k=3, nin=3, nlayers=2):
        super().__init__()
        self.w = w
        self.nin = nin
        self.embed = nn.Conv1d(nin, w, k, padding=k // 2)
        self.norm = ChanLN(w)
        layers = [nn.Conv1d(2 * w + 1, w, k, padding=k // 2)]
        for _ in range(nlayers - 1):
            layers += [nn.Conv1d(w, w, k, padding=k // 2)]
        self.layers = nn.ModuleList(layers)
        self.bnorm = ChanLN(w)
        self.bhead = nn.Sequential(nn.Conv1d(w, w // 2, 1), nn.ReLU(), nn.Conv1d(w // 2, 1, 1))
        # price head: predicted payment of the NEXT step, per position (summed)
        self.ghead = nn.Sequential(nn.Conv1d(w + 2, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1))
        # PonderNet halting head (pooled mean+max -> lambda)
        self.phead = nn.Sequential(nn.Conv1d(w + 1, 64, 1), nn.ReLU())
        self.pout = nn.Linear(128, 1)

    def encode(self, x):
        oh = F.one_hot(x, self.nin).float().transpose(1, 2)
        return self.embed(oh)

    def step(self, h, b, e, mask):
        z = torch.cat([self.norm(h), e, torch.tanh(b / 4).unsqueeze(1)], 1)
        for i, conv in enumerate(self.layers):
            z = conv(F.relu(z) if i > 0 else z)
        h = (h + z) * mask.unsqueeze(1)
        db = self.bhead(F.relu(self.bnorm(h))).squeeze(1) * mask
        return h, b + db

    def gain_pred(self, h, b, mask):
        """Predicted payment of the next step (nats per example). Detached
        inputs: prices steer compute, not the representation."""
        f = torch.cat([self.norm(h).detach(), torch.tanh(b / 4).detach().unsqueeze(1),
                       torch.sigmoid(b).detach().unsqueeze(1).sub(0.5).abs()], 1)
        return (self.ghead(f).squeeze(1) * mask).sum(1)

    def halt_lambda(self, h, b, mask):
        f = self.phead(torch.cat([self.norm(h), torch.tanh(b / 4).unsqueeze(1)], 1))
        m = mask.unsqueeze(1)
        mean = (f * m).sum(2) / m.sum(2).clamp(min=1)
        mx = (f + (m - 1) * 1e4).amax(2)
        return torch.sigmoid(self.pout(torch.cat([mean, mx], 1)).squeeze(1))

    def forward(self, x, mask, T, alpha_h=None, alpha_b=None, want_h=False, halt_cost=None,
                explore=None, h0=None, b0=None):
        """Run T steps; return list of beliefs b_0..b_T, h states, gain preds
        (g_t predicts pay_{t+1} from state t), ponder lambdas (lambda_t at state t).
        alpha_h / alpha_b: optional (T,) scalings of the h and b writes, used by
        the Aumann-Shapley / integrated-gradients credit diagnostic."""
        B, L = x.shape
        e = self.encode(x)
        h = torch.zeros(B, self.w, L, device=x.device) if h0 is None else h0
        b = torch.zeros(B, L, device=x.device) if b0 is None else b0
        bs, gs, lams, hs = [b], [], [], []
        self.cf_pays = []  # (b_before, b_proposed) for counterfactual gain targets
        self.halt_info = []  # (executed, past_halt_decision) per step
        active = torch.ones(B, device=x.device)
        wanted = torch.ones(B, device=x.device)   # 0 once the halting rule said stop
        for t in range(T):
            gs.append(self.gain_pred(h, b, mask))
            if halt_cost is not None and t > 0:
                wanted = wanted * (gs[-1] > halt_cost).float()
                # epsilon-continuation: exploring examples ignore the halt decision
                active = wanted if explore is None else torch.maximum(wanted, explore)
            self.halt_info.append((active.clone(), 1 - wanted))
            hn, bn = self.step(h, b, e, mask)
            if halt_cost is not None:
                self.cf_pays.append((b, bn))
                a_ = active.view(B, 1)
                hn = h + a_.unsqueeze(2) * (hn - h)
                bn = b + a_ * (bn - b)
            if alpha_h is not None:
                hn = h + alpha_h[t] * (hn - h)
            if alpha_b is not None:
                bn = b + alpha_b[t] * (bn - b)
            h, b = hn, bn
            bs.append(b)
            lams.append(self.halt_lambda(h, b, mask))
            if want_h:
                hs.append(h)
        gs.append(self.gain_pred(h, b, mask))
        return bs, gs, lams, hs


def score(b, y, mask):
    """Log score S(b) = sum_i log p(y_i) over real positions (nats, per example)."""
    return -(F.binary_cross_entropy_with_logits(b, y, reduction="none") * mask).sum(1)


# ---------------------------------------------------------------- losses
def loss_fn(arm, model, x, y, mask, T, beta_ponder=0.01, lam_prior=None, halt_cost=None, eps=0.1):
    if arm == "P":
        # Bansal et al. 2022 progressive loss (alpha = 1 for prefix sums): n no-grad
        # steps from scratch, then k steps with grad from the detached state
        n = int(torch.randint(0, T, (1,)))
        k = int(torch.randint(1, T - n + 1, (1,)))
        with torch.no_grad():
            bs0, _, _, hs0 = model(x, mask, n, want_h=True) if n else ([None], None, None, [None])
        h0 = hs0[-1].detach() if n else None
        b0 = bs0[-1].detach() if n else None
        bs, _, _, _ = model(x, mask, k, h0=h0, b0=b0)
        loss = (-score(bs[-1], y, mask) / mask.sum(1)).mean()
        return loss, {"final": loss.item(), "loss": loss.item(), "n": n, "k": k}
    tt_arm = arm in ("Ctt", "Ctt_exec", "Ctt_eps", "Ctt_real")
    explore = None
    if arm == "Ctt_eps" and halt_cost is not None:
        explore = (torch.rand(x.shape[0], device=x.device) < eps).float()
    bs, gs, lams, _ = model(x, mask, T, halt_cost=halt_cost if tt_arm else None, explore=explore)
    nbits = mask.sum(1)
    S = [score(b, y, mask) for b in bs]
    final = (-S[-1] / nbits).mean()
    logs = {"final": final.item()}
    loss = final
    if arm in ("B", "C") or tt_arm:
        # per-step MSR terms: each step paid for its own trade at the given price
        msr = 0.0
        for t in range(1, T + 1):
            pre = bs[t - 1].detach()
            db = bs[t] - bs[t - 1]
            pay = score(pre + db, y, mask) - score(pre, y, mask)
            msr = msr + (-pay / nbits).mean()
        loss = loss + msr / T
    if arm == "Bds":
        loss = loss + torch.stack([(-s / nbits).mean() for s in S[1:]]).mean()
    if arm == "C" or tt_arm:
        wts = torch.ones(x.shape[0], T, device=x.device)
        if model.cf_pays and arm in ("Ctt_exec", "Ctt_eps"):
            ex = torch.stack([hi[0] for hi in model.halt_info], 1)       # executed
            past = torch.stack([hi[1] for hi in model.halt_info], 1)     # after a halt decision
            if arm == "Ctt_exec":   # buggy: settle only steps that ran
                wts = ex
            else:                   # eps: continued steps after a halt settle with weight 1/eps
                wts = ex * torch.where(past > 0, torch.full_like(past, 1.0 / eps), torch.ones_like(past))
        if model.cf_pays and arm == "Ctt_real":
            # buggy (self-fulfilling): settle every step against the REALIZED pay,
            # which is 0 after a halt, so a low prediction confirms itself
            pays = torch.stack([(S[t + 1] - S[t]).detach() for t in range(T)], 1)
        elif model.cf_pays:  # train-time halting: counterfactual payment of each step
            pays = torch.stack([(score(bn, y, mask) - score(bo, y, mask)).detach()
                                for bo, bn in model.cf_pays], 1)
        else:
            pays = torch.stack([(S[t + 1] - S[t]).detach() for t in range(T)], 1)  # (B,T)
        gp = torch.stack(gs[:T], 1)
        gl = (F.smooth_l1_loss(gp, pays, beta=1.0, reduction="none") * wts).sum() / wts.numel()
        loss = loss + gl
        logs["gain_mse"] = gl.item()
    if arm == "D":
        lam = torch.stack(lams, 1).clamp(1e-6, 1 - 1e-6)  # (B,T)
        lam = torch.cat([lam[:, :-1], torch.ones_like(lam[:, -1:])], 1)
        surv = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], 1), 1)
        p = lam * surv
        Ls = torch.stack([(-s / nbits) for s in S[1:]], 1)
        rec = (p * Ls).sum(1).mean()
        lp = lam_prior or 1.0 / (T / 2)
        tt = torch.arange(T, device=x.device, dtype=torch.float)
        prior = lp * (1 - lp) ** tt
        prior = prior / prior.sum()
        kl = (p * (torch.log(p + 1e-10) - torch.log(prior))).sum(1).mean()
        loss = rec + beta_ponder * kl
        logs["exp_steps"] = (p * (tt + 1)).sum(1).mean().item()
    logs["loss"] = loss.item()
    return loss, logs
