"""Prequential (test-then-train) single pass and multi-epoch stress mode for the S0 substrate.

Worker 1 of S0 (traders-net study, 2026-09-27). Model code: model.py (verbatim wealth-net copy).

Time convention. A stream has Q questions and bits[q, 0..T-1]. Position i in [0, n_pos),
n_pos = T - W, predicts target bit bits[q, i+W] (target time i+W) from a window of W past bits.

Stream interface (anything with these attributes/methods plugs in; see ArrayStream):
  Q, W, n_pos, fam [Q] int, delay [Q] int (positions; 0 = label known right after the
  prediction), window(idx)->float [n,Q,W], label(idx)->float [n,Q],
  label_mask(idx)->bool [n,Q] (False = question not live at that position: not scored, not
  trained; e.g. a held-out family before its release).

Prequential single pass (run_prequential). Step s owns the block of B consecutive positions
P_s = [sB, (s+1)B). 1) TEST: predict every (i in P_s, q) with the current weights and record
the logit (the "trade price"). 2) TRAIN: one optimizer step on every (i, q) whose label becomes
known inside this block, i.e. i + delay[q] in P_s. So each (i, q) is predicted exactly once,
before any update that used its label, and trained on exactly once. With delay 0 this is plain
test-then-train. The model never sees a label before it has been scored on it (asserted).

Multi-epoch stress (run_multiepoch). Positions [0, split) are the training set, visited for
`epochs` shuffled passes; positions [split, n_pos) are never trained and are the held-out set.
Train and held-out target indices are disjoint (asserted). Reports in-sample and held-out BCE per
family per epoch: the memorisation gap wealth-net's in-sample books pay for.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from streams import N_FAM


class ArrayStream:
    """Stream over a fixed bit array. For delay[q] = D > 0 the window of question q only shows
    bits already revealed at prediction time: window(i)[q] = bits[q, i-D : i+W-D] (0.5 before
    time 0). Label availability for training is handled by the loop via `delay`."""

    def __init__(self, bits, fam, W, delay=None, live=None, desc=None, n_fam=N_FAM):
        self.bits = np.asarray(bits, dtype=np.uint8)
        self.Q, self.T = self.bits.shape
        self.W = W
        self.n_pos = self.T - W
        self.fam = np.asarray(fam, dtype=np.int64)
        self.n_fam = n_fam
        self.delay = np.zeros(self.Q, dtype=np.int64) if delay is None else np.asarray(delay, dtype=np.int64)
        self.live = live  # optional bool [n_pos, Q]
        self.desc = desc
        pad = int(self.delay.max())
        # unknown (pre-time-0) window entries read 0.5, the same convention as streams_ext and
        # li_families (S0 lead merge, 2026-09-27; was 0).
        self._b = torch.tensor(np.concatenate([np.full((self.Q, pad), 0.5, np.float32),
                                               self.bits.astype(np.float32)], 1),
                               dtype=torch.float32)
        self._pad = pad
        self._qi = torch.arange(self.Q)

    def window(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        start = idx[:, None] - torch.as_tensor(self.delay)[None, :] + self._pad       # [n,Q]
        cols = start[..., None] + torch.arange(self.W)                                # [n,Q,W]
        return self._b[self._qi[None, :, None], cols]

    def label(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self._b[:, idx + self.W + self._pad].T.contiguous()

    def label_mask(self, idx):
        idx = np.asarray(idx)
        if self.live is None:
            return torch.ones(len(idx), self.Q, dtype=torch.bool)
        return torch.as_tensor(self.live[idx])


def _masked_bce(z, y, m):
    l = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1.0)


def run_prequential(model, stream, opt, B=4, steps=None, baseline=None, opt_b=None,
                    on_step=None, check=True):
    """Single prequential pass. Returns dict with z [n_pos,Q] (net logits at trade time),
    zb (baseline logits, if any), y, mask, pred_step and train_step [n_pos,Q] (-1 = never)."""
    n_pos, Q = stream.n_pos, stream.Q
    n_steps = math.ceil(n_pos / B) if steps is None else min(steps, math.ceil(n_pos / B))
    Z = np.full((n_pos, Q), np.nan, np.float32)
    ZB = np.full((n_pos, Q), np.nan, np.float32) if baseline is not None else None
    Y = np.zeros((n_pos, Q), np.uint8)
    M = np.zeros((n_pos, Q), bool)
    pred_step = np.full((n_pos, Q), -1, np.int32)
    train_step = np.full((n_pos, Q), -1, np.int32)
    delays = np.unique(stream.delay)
    dly = torch.as_tensor(stream.delay)
    for s in range(n_steps):
        lo, hi = s * B, min((s + 1) * B, n_pos)
        P = np.arange(lo, hi)
        # 1) TEST at the trade price
        win, y, m = stream.window(P), stream.label(P), stream.label_mask(P)
        model.eval()
        with torch.no_grad():
            z = model(win)
            Z[P] = z.numpy()
            if baseline is not None:
                ZB[P] = baseline(win).numpy()
        Y[P], M[P] = y.numpy().astype(np.uint8), m.numpy()
        pred_step[P] = s
        # 2) TRAIN on labels revealed inside this block
        parts = []
        for d in delays:
            Pd = np.arange(max(0, lo - d), max(0, hi - d))
            if len(Pd) == 0:
                continue
            qm = (dly == int(d))[None, :] & stream.label_mask(Pd)
            if not qm.any():
                continue
            parts.append((Pd, qm))
        if parts:
            idx = np.concatenate([p for p, _ in parts])
            tm = torch.cat([q for _, q in parts]).float()
            tw, ty = stream.window(idx), stream.label(idx)
            if check:
                tmb = tm.numpy() > 0
                ii, qq = np.nonzero(tmb)
                assert np.all(pred_step[idx[ii], qq] >= 0), "trained on a label before predicting it"
                assert np.all(train_step[idx[ii], qq] < 0), "label trained twice in a single pass"
                train_step[idx[ii], qq] = s
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = _masked_bce(model(tw), ty, tm)
            loss.backward()
            opt.step()
            if baseline is not None:
                opt_b.zero_grad(set_to_none=True)
                _masked_bce(baseline(tw), ty, tm).backward()
                opt_b.step()
        if on_step is not None:
            on_step(s, P, Z[P], Y[P], M[P])
    return {"z": Z, "zb": ZB, "y": Y, "mask": M, "pred_step": pred_step, "train_step": train_step,
            "steps": n_steps, "B": B}


def per_family(z, y, m, fam, n_fam=N_FAM):
    """Mean BCE and Brier per family over masked entries of [n,Q] arrays (numpy)."""
    z = z.astype(np.float64)
    bce = np.logaddexp(0, z) - y * z
    p = 1 / (1 + np.exp(-z))
    br = (p - y) ** 2
    out_b, out_r = np.full(n_fam, np.nan), np.full(n_fam, np.nan)
    for f in range(n_fam):
        mm = m & (fam[None, :] == f)
        if mm.any():
            out_b[f], out_r[f] = bce[mm].mean(), br[mm].mean()
    return out_b, out_r


def split_positions(n_pos, frac=0.8):
    return int(n_pos * frac)


def assert_disjoint(stream, train_pos, eval_pos):
    """Train and eval target indices never overlap, and no eval target is inside a train window
    or train target of the same question (index level)."""
    tr_t = set((np.asarray(train_pos) + stream.W).tolist())
    ev_t = set((np.asarray(eval_pos) + stream.W).tolist())
    assert not (tr_t & ev_t), "train and eval targets overlap"
    # latest bit index any training example touches (window or target)
    assert max(tr_t) < min(ev_t), "an eval target precedes a train target"
    return True


def run_multiepoch(model, stream, opt, epochs=20, B=128, frac=0.8, seed=0, baseline=None,
                   opt_b=None, n_eval=4096, on_epoch=None):
    """Multi-epoch in-sample training on [0, split) with held-out eval on [split, n_pos)."""
    split = split_positions(stream.n_pos, frac)
    tr = np.arange(0, split)
    ev = np.arange(split, stream.n_pos)
    assert_disjoint(stream, tr, ev)
    g = np.random.default_rng(seed + 7919)
    ev_s = ev[np.linspace(0, len(ev) - 1, min(n_eval, len(ev))).astype(int)]
    in_s = tr[np.linspace(0, len(tr) - 1, min(n_eval, len(tr))).astype(int)]
    rows = []

    def evaluate(P, mdl):
        with torch.no_grad():
            z = mdl(stream.window(P)).numpy()
        return per_family(z, stream.label(P).numpy(), stream.label_mask(P).numpy(), stream.fam, stream.n_fam)

    step = 0
    for e in range(epochs + 1):
        model.eval()
        rec = {"epoch": e, "step": step}
        rec["in_bce"], rec["in_brier"] = evaluate(in_s, model)
        rec["ho_bce"], rec["ho_brier"] = evaluate(ev_s, model)
        if baseline is not None:
            rec["in_bce_log"], _ = evaluate(in_s, baseline)
            rec["ho_bce_log"], _ = evaluate(ev_s, baseline)
        rows.append(rec)
        if on_epoch is not None:
            on_epoch(rec)
        if e == epochs:
            break
        perm = g.permutation(tr)
        model.train()
        for k in range(0, len(perm), B):
            P = perm[k:k + B]
            win, y, m = stream.window(P), stream.label(P), stream.label_mask(P).float()
            opt.zero_grad(set_to_none=True)
            _masked_bce(model(win), y, m).backward()
            opt.step()
            if baseline is not None:
                opt_b.zero_grad(set_to_none=True)
                _masked_bce(baseline(win), y, m).backward()
                opt_b.step()
            step += 1
    return rows, {"split": split, "n_train": len(tr), "n_eval": len(ev), "steps": step}


def overlap_audit(bits, fam, split, W, L=48, n_fam=N_FAM):
    """Content-level leakage audit: for each family, the fraction of held-out targets (target index
    >= split+W) whose preceding L-bit context + target (L+1 bits) appears anywhere in the
    training region (target index < split+W) of ANY question. Periodic questions are expected to
    repeat; a high value for a long-period family (LFSR, prime, pi, CA) means questions are
    shifted copies of each other (the old pi-offset overlap class)."""
    Q, T = bits.shape
    assert L + 1 <= 63
    w = (1 << np.arange(L + 1, dtype=np.uint64)).astype(np.uint64)

    def codes(row, a, b):  # codes of (L+1)-grams ending at target indices [a, b)
        a = max(a, L)
        if b <= a:
            return np.zeros(0, np.uint64)
        v = np.lib.stride_tricks.sliding_window_view(row.astype(np.uint64), L + 1)[a - L:b - L]
        return (v * w).sum(1)

    train_codes = np.unique(np.concatenate([codes(bits[q], 0, split + W) for q in range(Q)]))
    out = {}
    for f in range(n_fam):
        qs = np.nonzero(fam == f)[0]
        if len(qs) == 0:
            continue
        hits = n = 0
        for q in qs:
            c = codes(bits[q], split + W, T)
            hits += np.isin(c, train_codes).sum(); n += len(c)
        out[f] = hits / max(n, 1)
    return out
