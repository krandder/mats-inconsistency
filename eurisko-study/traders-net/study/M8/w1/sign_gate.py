"""M8 worker 1: sign test for income-gated MLP neurons (see PREREG.md).
Usage: python3 sign_gate.py ARM SEED [T]   -> results/run-{ARM}-s{SEED}.json
ARM in none, pos0.1, pos1, neg0.1, neg1, shuf1."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
torch.set_num_threads(1)
import streams_ext
from model import Net

ARM, SEED = sys.argv[1], int(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
B, WARM, HL, ALPHA, HOLD = 4, 500, 1000, 1e-3, 512
SIGN = {"none": 0, "pos0.1": 1, "pos1": 1, "neg0.1": -1, "neg1": -1, "shuf1": 1}[ARM]
ETA = {"none": 0.0, "pos0.1": 0.1, "pos1": 1.0, "neg0.1": 0.1, "neg1": 1.0, "shuf1": 1.0}[ARM]
beta = 2 ** (-1.0 / HL)

torch.manual_seed(SEED); np.random.seed(SEED)
s = streams_ext.make_streams(Q=64, T=T, seed=SEED, heldout=8, regime_every=10000, regime_frac=0.25)
st = streams_ext.to_array_stream(s)
Qt, W = st.Q, st.W
net = Net(Qt, W, d=64, H=4, L=2, mlp=4)
opt = torch.optim.Adam(net.parameters(), 1e-3)
L, N = net.L, 4 * net.d
gate = torch.ones(L, N)
ema = torch.zeros(L, N); ema_abs = torch.zeros(L, N)
rng = np.random.default_rng(SEED + 1000)
perm = [torch.as_tensor(rng.permutation(N)) for _ in range(L)]

n_train = st.n_pos - HOLD
n_steps = math.ceil(n_train / B)
Z = np.full((n_train, Qt), np.nan, np.float32)
Y = np.zeros((n_train, Qt), np.uint8); M = np.zeros((n_train, Qt), bool)
ones_head = torch.ones(L, net.H)
t0 = time.time()
for step in range(n_steps):
    P = np.arange(step * B, min((step + 1) * B, n_train))
    win, y, m = st.window(P), st.label(P), st.label_mask(P)
    mask = {"neuron": gate, "head": ones_head}
    net.eval()
    with torch.no_grad():
        Z[P] = net(win, mask=mask).numpy()
    Y[P], M[P] = y.numpy().astype(np.uint8), m.numpy()
    # train on the same block (delay 0): each label scored above before being trained on
    net.train()
    rec = {}
    z = net(win, rec=rec, mask=mask)
    for a in rec["act"]:
        a.retain_grad()
    mf = m.float()
    l = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    loss = (l * mf).sum() / mf.sum().clamp_min(1.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        # act is pre-gate; act*dL/dact == h*dL/dh for h = gate*act
        c = torch.stack([-(a * a.grad).sum((0, 1)) for a in rec["act"]])       # [L,N]
        ema.mul_(beta).add_((1 - beta) * c)
        ema_abs.mul_(beta).add_((1 - beta) * c.abs())
        if SIGN != 0 and step >= WARM:
            zc = (ema - ema.mean(1, keepdim=True)) / (ema.std(1, keepdim=True) + 1e-12)
            w = N * ((1 - ALPHA) * torch.softmax(SIGN * ETA * zc, 1) + ALPHA / N)
            if ARM == "shuf1":
                if step % 1000 == 0:
                    perm = [torch.as_tensor(rng.permutation(N)) for _ in range(L)]
                w = torch.stack([w[i][perm[i]] for i in range(L)])
            gate = w
    if step % 2000 == 0:
        print(f"{ARM} s{SEED} step {step}/{n_steps} loss {loss.item():.4f} gmax {gate.max():.2f} {time.time()-t0:.0f}s", flush=True)

# ---- metrics
zz = Z.astype(np.float64)
bce = np.logaddexp(0, zz) - Y * zz
tt = np.arange(n_train) + W            # target time of each position
prim = float(bce[M].mean())
famt = s.fam_t[:, tt].T                  # [n_train, Qt] family at target time
fam_bce = {streams_ext.FAMILIES[f]: float(bce[M & (famt == f)].mean()) for f in range(len(streams_ext.FAMILIES)) if (M & (famt == f)).any()}
prim_nolfsr = float(bce[M & (famt != 1)].mean())
last = tt >= (tt.max() - 5000)
lfsr_last = M & (famt == 1) & last[:, None]
lfsr_learnt_q = int(sum((bce[:, q][lfsr_last[:, q]].mean() < 0.3) for q in range(Qt) if lfsr_last[:, q].any()))
lfsr_q = int(sum(lfsr_last[:, q].any() for q in range(Qt)))
sw = [e for e in s.events if e["kind"] == "switch"]
bounds = sorted({e["t"] for e in sw})
post, post_sw, pre = {}, {}, {}
for b in bounds:
    rows = (tt >= b) & (tt < b + 2000)
    rows_pre = (tt >= b - 2000) & (tt < b)
    qs = sorted({e["q"] for e in sw if e["t"] == b})
    mm = M & rows[:, None]
    post[b] = float(bce[mm].mean()) if mm.any() else None
    ms = np.zeros_like(M); ms[:, qs] = True; ms &= mm
    post_sw[b] = float(bce[ms].mean()) if ms.any() else None
    mp = M & rows_pre[:, None]
    pre[b] = float(bce[mp].mean()) if mp.any() else None

# ---- census on held-out segment (never trained), current gates in place
print("train done", time.time()-t0, flush=True)
Ph = np.arange(n_train, st.n_pos)[::4]   # 128 of the 512 held-out positions
with torch.no_grad():
    wh, yh, mh = st.window(Ph), st.label(Ph), st.label_mask(Ph).float()
    def hb(g):
        zh = net(wh, mask={"neuron": g, "head": ones_head})
        return float((F.binary_cross_entropy_with_logits(zh, yh, reduction="none") * mh).sum() / mh.sum())
    base = hb(gate)
    dl = torch.zeros(L, N)
    for li in range(L):
        for i in range(N):
            g2 = gate.clone(); g2[li, i] = 0.0
            dl[li, i] = hb(g2) - base
from scipy.stats import spearmanr
d_ = dl.flatten().numpy(); e_ = ema.flatten().numpy(); ea_ = ema_abs.flatten().numpy()
top = d_ >= np.quantile(d_, 0.9)
census = dict(heldout_bce=base, neg_share_all=float((e_ < 0).mean()), neg_share_top10=float((e_[top] < 0).mean()),
              spearman_income=float(spearmanr(e_, d_)[0]), spearman_absag=float(spearmanr(ea_, d_)[0]),
              frac_dead=float((d_ == 0).mean()))
out = dict(arm=ARM, seed=SEED, T=T, B=B, steps=n_steps, eta=ETA, sign=SIGN, alpha=ALPHA, half_life=HL,
           warmup=WARM, prequential_bce=prim, prequential_bce_no_lfsr16=prim_nolfsr, family_bce=fam_bce, lfsr16_learnt_q=lfsr_learnt_q, lfsr16_q=lfsr_q, post2k=post, post2k_switched=post_sw, pre2k=pre,
           gate_max=float(gate.max()), gate_min=float(gate.min()), census=census, wall_s=time.time() - t0)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", f"run-{ARM}-s{SEED}.json"), "w"), indent=1)
np.savez_compressed(os.path.join("/tmp/traders-net/M8", f"census-{ARM}-s{SEED}.npz"), dl=dl.numpy(), ema=ema.numpy(), ema_abs=ema_abs.numpy(), gate=gate.numpy())
print(json.dumps({k: out[k] for k in ("arm", "seed", "prequential_bce", "census", "wall_s")}))
