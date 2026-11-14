"""Kelvin's follow-up: is recent income a better currency than all-time wealth?

No training. From the per-100-step snapshots and saved weights, for MLP neurons:
  currencies  cum (all-time wealth), decayed wealth W_t = lam*W_{t-1} + income_t with
              half-lives 100..10000 steps, and income over the last 100/500/2000 steps
  oracle      single-neuron ablation: held-out BCE with that neuron zeroed minus intact BCE
              (final checkpoint; positive = the neuron helps)
  ablation    remove the lowest 1..20% by each currency (plus random and gradient-norm
              controls) at the final and 20k checkpoints; held-out BCE
The held-out segment is the one ablate.py uses (positions T..T+4095, never trained on);
to fit the CPU budget this uses every 4th position (1024) for the ablation curves and every
8th (512) for the oracle. Writes results/decay_*.csv.
Usage: python3 decay.py SEED [results]
"""
import csv, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from streams import make_questions, FAMILIES
from model import Net

HALF = [100, 300, 1000, 3000, 10000]
WIN = [100, 500, 2000]
FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
SNAP = 100


def currencies(res, tag, step):
    """Per-neuron currencies at `step` from cumulative-wealth snapshots every 100 steps."""
    ws = []
    for s in range(SNAP, step + 1, SNAP):
        d = json.load(open(os.path.join(res, tag, "snapshots", f"step-{s:05d}.json")))
        ws.append(np.array(d["wealth"]["neuron"], dtype=np.float64))
    ws = np.stack(ws)                                    # [S,512] cumulative
    inc = np.diff(np.vstack([np.zeros(ws.shape[1]), ws]), axis=0)   # income per 100-step window
    cur = {"cum": ws[-1]}
    for h in HALF:
        lam = 0.5 ** (SNAP / h); W = np.zeros(ws.shape[1])
        for x in inc:
            W = lam * W + x
        cur[f"hl{h}"] = W
    for w in WIN:
        cur[f"win{w}"] = inc[-(w // SNAP):].sum(0)
    return cur, d


def tail_fns(m):
    """Forward from the residual stream after layer-0 attention, with a neuron mask [L,N]."""
    L, H, Q, d = m.L, m.H, m.Q, m.d

    def attn(blk, x):
        B = x.shape[0]
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).view(B, Q, 3, H, blk.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
        return x + blk.o((A @ v).transpose(1, 2).reshape(B, Q, d))

    def mlp(blk, x, mk):
        return x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))) * mk)

    def readout(x):
        return (m.lnf(x) * m.rw).sum(-1) + m.rb

    def head(win):                                       # up to after layer-0 attention
        return attn(m.blocks[0], m.win(2.0 * win - 1.0) + m.qid)

    def tail(x1, mask):
        x = mlp(m.blocks[0], x1, mask[0])
        for li in range(1, L):
            x = mlp(m.blocks[li], attn(m.blocks[li], x), mask[li])
        return readout(x)
    return head, tail, attn, mlp, readout


def main(seed, res="results"):
    torch.set_num_threads(int(os.environ.get("THREADS", "5")))
    t0 = time.time()
    tag = f"train-s{seed}"
    meta = json.load(open(os.path.join(res, f"{tag}.json")))
    Q, W, T = meta["Q"], meta["W"], meta["T"]; E = 4096
    bits, fam, _ = make_questions(Q, T + E, seed)
    b = torch.tensor(bits, dtype=torch.float32)
    wins = b.unfold(1, W, 1)
    ev = torch.arange(T - W, T + E - W)
    ev_win, ev_y = wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous()
    famt = torch.tensor(fam)
    final = meta["steps_done"]
    ckpts = [(final, os.path.join(res, f"{tag}.pt")), (20000, os.path.join(res, "weights", f"{tag}-step20000.pt"))]
    abl_rows, cur_rows = [], []
    for step, wf in ckpts:
        step_s = step // SNAP * SNAP
        cur, sn = currencies(res, tag, step_s)
        gn = np.array(sn["gradnorm_cum"]["neuron"]).ravel()
        m = Net(Q, W, meta["d"], meta["H"], meta["L"])
        m.load_state_dict(torch.load(wf, map_location="cpu")); m.eval()
        L, N = m.L, 4 * m.d; n = L * N
        head, tail, attn, mlp, readout = tail_fns(m)
        ones = torch.ones(L, N)
        with torch.no_grad():
            assert torch.allclose(tail(head(ev_win[:64]), ones), m(ev_win[:64]), atol=1e-4)

            def bce_of(x1, y, mask):
                return F.binary_cross_entropy_with_logits(tail(x1, mask), y, reduction="none").mean(0)

            # ---- ablation curves by currency (every 4th held-out position)
            sub = torch.arange(0, E, 4)
            x1, y = head(ev_win[sub]), ev_y[sub]
            base = bce_of(x1, y, ones)
            abl_rows.append(dict(seed=seed, step=step, final=int(step == final), criterion="none", frac=0, k_removed=0,
                                 n_negative=0, bce=round(base.mean().item(), 6),
                                 **{f"bce_{f}": round(base[famt == i].mean().item(), 6) for i, f in enumerate(FAMILIES)}))
            rng = np.random.default_rng(seed * 7 + step)
            orders = {c: [np.argsort(v, kind="stable")] for c, v in cur.items()}
            orders["gradnorm"] = [np.argsort(gn, kind="stable")]
            orders["random"] = [rng.permutation(n) for _ in range(3)]
            for c, ods in orders.items():
                nneg = int((cur[c] < 0).sum()) if c in cur else 0
                for fr in FRACS:
                    k = int(round(fr * n)); vals = []
                    for od in ods:
                        mk = ones.clone().view(-1); mk[torch.tensor(od[:k])] = 0
                        vals.append(bce_of(x1, y, mk.view(L, N)))
                    v = torch.stack(vals).mean(0)
                    abl_rows.append(dict(seed=seed, step=step, final=int(step == final), criterion=c, frac=fr, k_removed=k,
                                         n_negative=nneg, bce=round(v.mean().item(), 6),
                                         **{f"bce_{f}": round(v[famt == i].mean().item(), 6) for i, f in enumerate(FAMILIES)}))
            print(seed, step, "ablation done", round(time.time() - t0), "s", flush=True)

            # ---- oracle: single-neuron ablation (final checkpoint only; every 8th position)
            if step != final:
                continue
            sub = torch.arange(0, E, 8)
            x1, y = head(ev_win[sub]), ev_y[sub]
            base_o = bce_of(x1, y, ones).mean().item()
            dl = np.zeros(n)
            # layer 0 neurons: rerun the tail; last-layer neurons: subtract their fc2 contribution
            for j in range(N):
                mk = ones.clone(); mk[0, j] = 0
                dl[j] = bce_of(x1, y, mk).mean().item() - base_o
            x = mlp(m.blocks[0], x1, ones[0])
            for li in range(1, L):
                blk = m.blocks[li]
                xa = attn(blk, x)
                act = F.relu(blk.fc1(blk.ln2(xa)))
                xo = xa + blk.fc2(act)
                for j in range(N):
                    xj = xo - act[..., j:j + 1] * blk.fc2.weight[:, j]
                    for lj in range(li + 1, L):
                        xj = mlp(m.blocks[lj], attn(m.blocks[lj], xj), ones[lj])
                    dl[li * N + j] = F.binary_cross_entropy_with_logits(readout(xj), y).item() - base_o
                x = xo
            print(seed, "oracle done", round(time.time() - t0), "s", flush=True)
            crit = dict(cur); crit["gradnorm"] = gn
            for c, v in crit.items():
                r = spearmanr(v, dl).correlation
                cur_rows.append(dict(seed=seed, step=step, currency=c, spearman_vs_oracle=round(float(r), 4),
                                     n_negative=int((v < 0).sum()) if c in cur else "",
                                     frac_negative=round(float((v < 0).mean()), 4) if c in cur else "",
                                     oracle_n_harmful=int((dl < 0).sum()), oracle_base_bce=round(base_o, 6)))
            np.save(os.path.join(res, f"decay_oracle_s{seed}.npy"), dl)
            np.savez(os.path.join(res, f"decay_currencies_s{seed}.npz"), **cur)
    for name, rows in (("decay_ablation", abl_rows), ("decay_spearman", cur_rows)):
        with open(os.path.join(res, f"{name}_s{seed}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(seed, "all done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "results")
