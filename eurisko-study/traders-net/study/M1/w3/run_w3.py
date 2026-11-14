"""M1 item (c) validation runs for books_v2 (see PREREG-w3.md). One seed per invocation."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "../../S0/substrate"))
torch.set_num_threads(1)
from model import Net
from books_v2 import BooksV2, BOOKS, GRANS
import streams_ext as se

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--p_T", type=int, default=32032); ap.add_argument("--p_B", type=int, default=16)
ap.add_argument("--me_T", type=int, default=16032); ap.add_argument("--me_epochs", type=int, default=40)
ap.add_argument("--me_B", type=int, default=128); ap.add_argument("--me_bk", type=int, default=32)
ap.add_argument("--max_min", type=float, default=90)
ap.add_argument("--skip_me", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
GROUPS = [[5], [0, 1, 2, 3, 4, 6, 7, 8]]      # prng, rest
t0 = time.time()


def dump(tag, bk, extra=None):
    d = {b: {g: bk.W[b][g].cpu().numpy() for g in GRANS} for b in list(BOOKS) + ["absag"]}
    arr = {f"{b}__{g}": v for b, dd in d.items() for g, v in dd.items()}
    arr.update({f"acct__{k}": v.cpu().numpy() for k, v in bk.W["acct"].items()})
    np.savez_compressed(os.path.join(a.out, f"{tag}-s{a.seed}.npz"), **arr, steps=bk.steps)


def mkmodel(seed):
    torch.manual_seed(seed)
    return Net(64, 32, d=64, H=4, L=2)


log = {"seed": a.seed, "args": vars(a), "diag": {}}


def diag_acc(name, acc):
    dd = log["diag"].setdefault(name, {})
    for k, v in acc["diag"].items():
        dd.setdefault(k, []).append(v)


# ---------------- P: prequential single pass ----------------
s = se.make_streams(Q=64, T=a.p_T, seed=a.seed, heldout=0, regime_every=0)
st = se.to_array_stream(s)
model = mkmodel(a.seed)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
pre = BooksV2(model, st.fam, st.n_fam, groups=GROUPS)
post = BooksV2(model, st.fam, st.n_fam, groups=GROUPS)
n_steps = st.n_pos // a.p_B
loss_pre = []
for k in range(n_steps):
    P = np.arange(k * a.p_B, (k + 1) * a.p_B)
    w, y, m = st.window(P), st.label(P), st.label_mask(P)
    model.eval()
    acc = pre.account(w, y, m); pre.add(acc); diag_acc("P_pre", acc)
    loss_pre.append(acc["diag"]["loss"])
    model.train(); opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(model(w), y).backward(); opt.step()
    acc2 = pre.account(w, y, m); post.add(acc2)        # same prior/means as pre: in-sample by one step
    pre.observe(w, y, m)
    if k % 200 == 0:
        print(f"[s{a.seed}] P {k}/{n_steps} loss {np.mean(loss_pre[-200:]):.4f} {time.time()-t0:.0f}s", flush=True)
dump("P_pre", pre); dump("P_post", post)
log["P_loss_by_quarter"] = [float(np.mean(x)) for x in np.array_split(loss_pre, 4)]
torch.save(model.state_dict(), f"/tmp/traders-net/M1/w3/P-s{a.seed}.pt")

# controls: frozen final weights and random init, no updates, every 4th block
for ctl in ("frozen", "random"):
    cm = model if ctl == "frozen" else mkmodel(a.seed + 1000)
    cm.eval()
    bk = BooksV2(cm, st.fam, st.n_fam, groups=GROUPS)
    for k in range(0, n_steps, 4):
        P = np.arange(k * a.p_B, (k + 1) * a.p_B)
        w, y, m = st.window(P), st.label(P), st.label_mask(P)
        acc = bk.account(w, y, m); bk.add(acc); diag_acc(f"P_{ctl}", acc); bk.observe(w, y, m)
    dump(f"P_{ctl}", bk)
print(f"[s{a.seed}] P done {time.time()-t0:.0f}s", flush=True)

# ---------------- ME: multi-epoch stress ----------------
if not a.skip_me:
    s = se.make_streams(Q=64, T=a.me_T, seed=a.seed, heldout=0, regime_every=0)
    st = se.to_array_stream(s)
    split = int(st.n_pos * 0.8)
    model = mkmodel(a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bin_ = BooksV2(model, st.fam, st.n_fam, groups=GROUPS)
    bho = BooksV2(model, st.fam, st.n_fam, groups=GROUPS)
    ytr = st.label(np.arange(split)); mtr = st.label_mask(np.arange(split)).float()
    bin_.n1 += (ytr * mtr).sum(0); bin_.n += mtr.sum(0)       # fixed train base-rate prior
    g = np.random.default_rng(a.seed + 7919)
    step, ep_log = 0, []
    for e in range(a.me_epochs):
        perm = g.permutation(split)
        for j in range(0, split - a.me_B + 1, a.me_B):
            P = perm[j:j + a.me_B]
            w, y, m = st.window(P), st.label(P), st.label_mask(P)
            if step % 2 == 0:
                model.eval()
                Pi = P[:a.me_bk]
                wi, yi, mi = st.window(Pi), st.label(Pi), st.label_mask(Pi)
                acc = bin_.account(wi, yi, mi); bin_.add(acc); diag_acc("ME_in", acc)
                Ph = g.integers(split, st.n_pos, a.me_bk)
                wh, yh, mh = st.window(Ph), st.label(Ph), st.label_mask(Ph)
                acch = bin_.account(wh, yh, mh); bho.add(acch); diag_acc("ME_ho", acch)
                bin_.observe(wi, yi, torch.zeros_like(mi))       # means only; prior stays fixed
            model.train(); opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(w), y).backward(); opt.step()
            step += 1
        # epoch-end BCE in-sample vs held-out (prng and rest), 1024 positions each
        model.eval()
        with torch.no_grad():
            r = {}
            for nm, PP in (("in", np.linspace(0, split - 1, 1024).astype(int)),
                           ("ho", np.linspace(split, st.n_pos - 1, 1024).astype(int))):
                l = F.binary_cross_entropy_with_logits(model(st.window(PP)), st.label(PP), reduction="none").mean(0).numpy()
                r[nm + "_prng"] = float(l[st.fam == 5].mean()); r[nm + "_rest"] = float(l[st.fam != 5].mean())
        r["epoch"] = e + 1; ep_log.append(r)
        print(f"[s{a.seed}] ME ep {e+1} {r} {time.time()-t0:.0f}s", flush=True)
        if (time.time() - t0) / 60 > a.max_min:
            log["ME_cut_at_epoch"] = e + 1
            break
    log["ME_epochs"] = ep_log
    dump("ME_in", bin_); dump("ME_ho", bho)

log["diag_mean"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in log["diag"].items()}
log["diag_absmean"] = {k: {kk: float(np.mean(np.abs(vv))) for kk, vv in v.items()} for k, v in log["diag"].items()}
del log["diag"]
log["seconds"] = time.time() - t0
json.dump(log, open(os.path.join(a.out, f"run-s{a.seed}.json"), "w"), indent=1)
print(f"[s{a.seed}] DONE {log['seconds']:.0f}s", flush=True)
