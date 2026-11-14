"""Traders as blocks: backprop inside a block, market settlement at the block boundary.

Arms (same streams, model family and data order as wealth-net train.py):
  A  end-to-end: BCE of the final prediction, gradients flow everywhere
  B  block-local: block l's loss is the BCE of its logit-lens prediction (shared final LN +
     readout applied right after block l); the residual stream entering block l > 0 is
     detached, so no gradient crosses a block boundary. The final prediction is the last lens.
  C  block-local + market: as B, and each MLP neuron's output is scaled by m_i = N * w_i, its
     wealth share in the block. Each step w_i <- w_i * exp(eta * s_i) with s_i the exact
     leave-one-out log-score settlement at the block's own lens price; w is clipped and
     renormalised per block. Units with m_i below FLOOR are skipped (output zeroed) that step.
     Every BIRTH_EVERY steps each skipped unit is reborn as a noisy copy of a rich unit, which
     splits its wealth with the child (so the block's output barely changes).
Then, on the held-out segment: BCE per family, single-neuron ablation oracle, settlement vs
oracle Spearman, Gini, share negative, max single-unit damage, removal of the top 1%.
  R: control for C: same per-step skip count and birth rate (read from C's log for the seed), but the
     skipped and reborn units are chosen uniformly at random and there is no wealth scaling.
  W: control for C: wealth scaling only, no skipping and no births.
  L<alpha> (e.g. L0.3): arm B with a leaky boundary; the gradient crossing each block boundary
     is scaled by alpha (L0 = B; L1 = end-to-end on the sum of the block losses).
Usage: python3 blocklocal.py ARM SEED STEPS [out]
"""
import csv, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from streams import make_questions, FAMILIES
from model import Net

ETA, FLOOR, WMAX = float(os.environ.get("ETA", "1.0")), 0.1, 8.0
BIRTH_EVERY = int(os.environ.get("BIRTH_EVERY", "200"))
QUICK = os.environ.get("QUICK") == "1"          # smoke test: small eval


def lp(z, y):
    return -F.softplus(-(2 * y - 1) * z)


def lnstd(x, eps):
    xc = x - x.mean(-1, keepdim=True)
    return torch.sqrt(xc.pow(2).mean(-1) + eps)


def forward(m, win, local, scale=None, mask=None, keep=False, alpha=0.0):
    """Returns the lens logits after every block (last = prediction) and, with keep, the
    MLP activations (after scaling) and block outputs for settlement."""
    B, Q, H, d = win.shape[0], m.Q, m.H, m.d
    x = m.win(2.0 * win - 1.0) + m.qid
    lens, acts, outs = [], [], []
    for li, blk in enumerate(m.blocks):
        if local and li > 0:
            # leaky boundary: forward value unchanged, gradient crossing the boundary scaled by alpha
            x = x.detach() if alpha == 0 else x.detach() + alpha * (x - x.detach())
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).view(B, Q, 3, H, blk.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
        x = x + blk.o((A @ v).transpose(1, 2).reshape(B, Q, d))
        a = F.relu(blk.fc1(blk.ln2(x)))
        if scale is not None:
            a = a * scale[li]
        if mask is not None:
            a = a * mask[li]
        x = x + blk.fc2(a)
        lens.append((m.lnf(x) * m.rw).sum(-1) + m.rb)
        if keep:
            acts.append(a); outs.append(x)
    return lens, acts, outs


def settlement(m, lens, acts, outs, y, price):
    """Leave-one-out log-score settlement per neuron [L,N], summed over questions, mean over
    examples. price='block': each block's own lens; 'final': the final prediction."""
    L = m.L; res = []
    for li in range(L):
        blk = m.blocks[li]
        j = li if price == "block" else L - 1
        W2c = blk.fc2.weight - blk.fc2.weight.mean(0, keepdim=True)
        U = torch.einsum("dn,d,qd->qn", W2c, m.lnf.g, m.rw)
        c = acts[li] * U / lnstd(outs[j], m.lnf.eps).unsqueeze(-1)
        z = lens[j]
        s = lp(z, y).unsqueeze(-1) - lp(z.unsqueeze(-1) - c, y.unsqueeze(-1))
        res.append(s.sum(1).mean(0))
    return torch.stack(res)


def rebirth(blk, opt, i, p):
    """Unit i becomes a noisy copy of unit p (fc1 row and bias plus noise, fc2 column copied); its Adam state is reset."""
    for prm in (blk.fc1.weight, blk.fc1.bias):
        prm.data[i] = prm.data[p] + 0.01 * prm.data.std() * torch.randn_like(prm.data[p])
    blk.fc2.weight.data[:, i] = blk.fc2.weight.data[:, p]
    for prm, idx in ((blk.fc1.weight, i), (blk.fc1.bias, i)):
        st = opt.state.get(prm, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][idx] = 0
    st = opt.state.get(blk.fc2.weight, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            st[key][:, i] = 0


def gini_pos(x):
    x = np.sort(np.maximum(np.asarray(x, dtype=float), 0))
    if x.sum() <= 0:
        return 0.0
    n = len(x); return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main(arm, seed, steps, out="results/blocklocal"):
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    dev = torch.device(os.environ.get("DEVICE", "cpu"))
    os.makedirs(out, exist_ok=True)
    Q, W, T, Bsz, E = 64, 32, 24000, 128, (512 if QUICK else 4096)
    torch.manual_seed(seed); np.random.seed(seed)
    bits, fam, _ = make_questions(Q, T + E, seed)
    b = torch.tensor(bits, dtype=torch.float32, device=dev)
    wins = b.unfold(1, W, 1)
    n_pos = T - W; split = int(n_pos * 0.8)
    g = torch.Generator().manual_seed(seed + 1000 * 0)

    def batch(idx):
        idx = idx.to(dev)
        return wins[:, idx].permute(1, 0, 2).contiguous(), b[:, idx + W].T.contiguous()

    m = Net(Q, W).to(dev)
    L, N = m.L, 4 * m.d
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    local = arm[0] in "BCLRW"
    market = arm in ("C", "W")          # wealth scales contributions (W: no skipping, no births)
    dead = torch.zeros(L, N, dtype=torch.bool, device=dev)   # arm R: random skip set
    if arm == "R":                      # match C's skip count and birth rate over time (C log, same seed)
        ref = json.load(open(os.environ.get("REF_LOG", os.path.join(out, f"C-s{seed}-log.json"))))
        ref = ref["log"] if isinstance(ref, dict) else ref
        rs, rk, rb = [0] + [e["step"] for e in ref], [0] + [e["skipped"] for e in ref], [0] + [e["births"] for e in ref]
        rgen = torch.Generator().manual_seed(seed + 7)
    alpha = float(arm[1:]) if arm[0] == "L" else 0.0     # arm "L0.3": block-local with leak 0.3
    w = torch.full((L, N), 1.0 / N, device=dev)
    births, log = 0, []
    t0 = time.time()
    maxsec = 60 * float(os.environ.get("MAXMIN", "1e9"))
    for step in range(1, steps + 1):
        if time.time() - t0 > maxsec:
            steps = step - 1; print(arm, seed, "time cap at step", steps, flush=True); break
        win, y = batch(torch.randint(0, split, (Bsz,), generator=g))
        scale = (N * w).unsqueeze(1).unsqueeze(1) if market else None      # [L,1,1,N]
        mask = (N * w >= FLOOR).float().unsqueeze(1).unsqueeze(1) if arm == "C" else \
            (~dead).float().unsqueeze(1).unsqueeze(1) if arm == "R" else None
        lens, acts, outs = forward(m, win, local, scale=None if scale is None else [scale[l] for l in range(L)],
                                   mask=None if mask is None else [mask[l] for l in range(L)], keep=market, alpha=alpha)
        loss = sum(F.binary_cross_entropy_with_logits(z, y) for z in lens) if local else F.binary_cross_entropy_with_logits(lens[-1], y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if arm == "R" and step % BIRTH_EVERY == 0:
            with torch.no_grad():
                nb = int(round(np.interp(step, rs, rb) - np.interp(step - BIRTH_EVERY, rs, rb)))
                dl_ = torch.nonzero(dead.flatten()).flatten()
                for f in dl_[torch.randperm(len(dl_), generator=rgen)[:nb]].tolist():
                    li, i = divmod(f, N)
                    alive_ = torch.nonzero(~dead[li]).flatten()
                    p = int(alive_[torch.randint(len(alive_), (1,), generator=rgen)])
                    rebirth(m.blocks[li], opt, i, p); dead[li, i] = False; births += 1
                k = int(round(np.interp(step, rs, rk))) - int(dead.sum())
                if k > 0:
                    al_ = torch.nonzero(~dead.flatten()).flatten()
                    dead.view(-1)[al_[torch.randperm(len(al_), generator=rgen)[:k]]] = True
        if market:
            with torch.no_grad():
                s = settlement(m, [z.detach() for z in lens], [a.detach() for a in acts], [o.detach() for o in outs], y, "block")
                w = w * torch.exp((ETA * s).clamp(-0.5, 0.5))
                w = w / w.sum(1, keepdim=True)
                w = w.clamp(1e-3 / N, WMAX / N); w = w / w.sum(1, keepdim=True)
                if arm == "C" and step % BIRTH_EVERY == 0:
                    for li in range(L):
                        blk = m.blocks[li]
                        dead = torch.nonzero(N * w[li] < FLOOR).flatten().tolist()
                        for i in dead:
                            p = int(torch.multinomial(w[li], 1, generator=None))
                            if p in dead:
                                continue
                            rebirth(blk, opt, i, p)
                            w[li, i] = w[li, p] = w[li, p] / 2
                            births += 1
        if step % 1000 == 0 or step == steps or time.time() - t0 > maxsec:
            log.append(dict(step=step, loss=float(F.binary_cross_entropy_with_logits(lens[-1].detach(), y)),
                            births=births, skipped=int((N * w < FLOOR).sum()) if arm == "C" else int(dead.sum()),
                            wealth_gini=[gini_pos(w[l].cpu().numpy()) for l in range(L)] if market else None,
                            sec=round(time.time() - t0, 1)))
            print(arm, seed, log[-1], flush=True)
    torch.save({"state": m.state_dict(), "w": w.cpu()}, os.path.join(out, f"{arm}-s{seed}.pt"))

    # ---------------- evaluation (no training) ----------------
    m.eval()
    ev = torch.arange(T - W, T + E - W)
    ev_win, ev_y = wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous()
    famt = torch.tensor(fam, device=dev)
    sc = [(N * w[l]) for l in range(L)] if market else None
    base_mask = [(N * w[l] >= FLOOR).float() for l in range(L)] if arm == "C" else \
        [(~dead[l]).float() for l in range(L)] if arm == "R" else [torch.ones(N, device=dev) for _ in range(L)]

    def bce(win_, y_, mask):
        with torch.no_grad():
            z = torch.cat([forward(m, win_[i:i + 1024], False, sc, mask)[0][-1] for i in range(0, win_.shape[0], 1024)])
            return F.binary_cross_entropy_with_logits(z, y_, reduction="none").mean(0)

    full = bce(ev_win, ev_y, base_mask)
    sub = torch.arange(0, E, 8, device=dev)
    ow, oy = ev_win[sub], ev_y[sub]
    base_o = bce(ow, oy, base_mask).mean().item()
    dl = np.zeros(L * N)
    for li in range(L):
        for j in range(4 if QUICK else N):
            if base_mask[li][j] == 0:
                continue                                   # a skipped unit: removing it changes nothing
            mk = [x.clone() for x in base_mask]; mk[li][j] = 0
            dl[li * N + j] = bce(ow, oy, mk).mean().item() - base_o
    # settlement on 4096 training positions at the price each arm trains against
    tr = torch.randint(0, split, (E,), generator=torch.Generator().manual_seed(seed + 99))
    tw, ty = batch(tr)
    sett = {"block": torch.zeros(L, N, device=dev), "final": torch.zeros(L, N, device=dev)}
    with torch.no_grad():
        for i in range(0, E, 512):
            lens, acts, outs = forward(m, tw[i:i + 512], False, sc, base_mask, keep=True)
            for pr in sett:
                sett[pr] += settlement(m, lens, acts, outs, ty[i:i + 512], pr) / (E // 512)
    cur = (sett["final"] if arm == "A" else sett["block"]).cpu().numpy().ravel()
    alive = np.concatenate([x.cpu().numpy() for x in base_mask]) > 0
    top = np.argsort(dl)[::-1][:int(round(0.01 * L * N))]
    mk = [x.clone() for x in base_mask]
    for i in top:
        mk[i // N][i % N] = 0
    top1 = bce(ev_win, ev_y, mk).mean().item() - full.mean().item()
    rng = np.random.default_rng(seed)
    rnd = []
    for _ in range(3):
        mk = [x.clone() for x in base_mask]
        for i in rng.choice(np.nonzero(alive)[0], len(top), replace=False):
            mk[i // N][i % N] = 0
        rnd.append(bce(ev_win, ev_y, mk).mean().item() - full.mean().item())
    row = dict(arm=arm, seed=seed, steps=steps, bce=round(full.mean().item(), 6),
               **{f"bce_{f}": round(full[famt == i].mean().item(), 6) for i, f in enumerate(FAMILIES)},
               spearman_settlement_oracle=round(float(spearmanr(cur[alive], dl[alive]).correlation), 4),
               spearman_final_price_oracle=round(float(spearmanr(sett["final"].cpu().numpy().ravel()[alive], dl[alive]).correlation), 4),
               spearman_wealth_oracle=round(float(spearmanr(w.cpu().numpy().ravel()[alive], dl[alive]).correlation), 4) if market else "",
               gini_settlement=round(gini_pos(cur[alive]), 4), gini_wealth=round(gini_pos(w.cpu().numpy().ravel()), 4) if market else "",
               share_negative_settlement=round(float((cur[alive] < 0).mean()), 4), alive=int(alive.sum()), births=births,
               max_single_damage=round(float(dl.max()), 5), top1pct_removed_bce_increase=round(top1, 5),
               random1pct_removed_bce_increase=round(float(np.mean(rnd)), 5), train_seconds=log[-1]["sec"])
    print(row, flush=True)
    with open(os.path.join(out, f"{arm}-s{seed}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row)); wr.writeheader(); wr.writerow(row)
    json.dump(log, open(os.path.join(out, f"{arm}-s{seed}-log.json"), "w"))
    np.save(os.path.join(out, f"{arm}-s{seed}-oracle.npy"), dl)
    np.savez(os.path.join(out, f"{arm}-s{seed}-settlement.npz"), block=sett["block"].cpu().numpy(), final=sett["final"].cpu().numpy(), w=w.cpu().numpy())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]), a[3] if len(a) > 3 else "results/blocklocal")
