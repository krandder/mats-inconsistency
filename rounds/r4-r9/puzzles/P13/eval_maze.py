"""P13 evaluation: curves every 10 steps to T = mult * n for Trader2D (econ/bptt), DT-recall retrains and M-ref.

  python eval_maze.py --kind trader|dtr|mref --ckpt X.pt --sets e99,d99 --jopt none --out results/eval/NAME

Per record step t (every 10 steps, plus t_eq and 12n) and maze: exact (open-pixel mask), log loss in nats per open
pixel raw / clip c in {4,8,16} / temperature grid TAUS, max and mean |b| over open pixels, limit-cycle period
(smallest p in 1..8 with m_(t-k) = m_(t-k-p) for k < p, m = thresholded open-pixel mask; p >= 2 is a cycle, 1 a fixed
point, 0 none), per-level cov log loss (Trader2D). At t_eq and 12n: ECE histograms (15 bins, open pixels) and |b|
histograms of correct and wrong pixels (AUROC of error detection from -|b|). TF32 off. Saved as float16 npz.
Preemption resume (added 29 Sep 23:20Z): every --ck_every seconds, after a batch, the finished sets and the current set's
Recorder arrays and next batch offset are saved atomically to OUT.evck.npz; a rerun with the same arguments skips the
finished sets and batches. Batches are independent, so the result equals an uninterrupted run (only "sec" differs).
The resume file is renamed OUT.evck.used.npz at the end (small; kept for the audit).
"""
import argparse, json, math, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flops  # noqa: E402
from econ_maze import Model, pair_truth2d, soft_score, nlevels2d  # noqa: E402

LS = F.logsigmoid
CLIPS = [4.0, 8.0, 16.0]
TAUS = [2 ** (k / 2) for k in range(-2, 17)]          # 0.5 .. 256
NCAL = 1 + len(CLIPS) + len(TAUS)
ABS_EDGES = torch.cat([torch.zeros(1), torch.logspace(-3, 3, 120), torch.tensor([float("inf")])])
SETS_P0 = {"v9": "v9", "m13": "m13", "v33": "v33", "m59": "m59"}   # P0 frozen dev sets (m13 -> first 200)


def load_set(name, frozen_p0, frozen):
    from check import expand
    if name in SETS_P0:
        z = np.load(os.path.join(frozen_p0, "mazes_e2h.npz"))
        code, sol, sp = z[f"{name}/code"], z[f"{name}/sol"], z[f"{name}/sp_len"]
        if name == "m13":
            code, sol, sp = code[:200], sol[:200], sp[:200]
    else:
        z = np.load(os.path.join(frozen, "p13_eval.npz"))
        code, sol, sp = z[f"{name}/code"], z[f"{name}/sol"], z[f"{name}/sp_len"]
    n = code.shape[-1]
    s = np.unpackbits(sol, axis=-1, count=n)
    X, Y = expand(torch.as_tensor(code), torch.as_tensor(s))
    return X, Y, n, np.asarray(sp)


FLOOR = 0.0   # a model trained with --floor eps is scored with its fixed-share forecast (every log score)


def ll_all(b, y, m):
    """(B, NCAL) mean log loss per open pixel: raw, clips, temperatures"""
    from econ_maze import belief_score
    nb = m.sum((1, 2))
    outs = [b] + [b.clamp(-c, c) for c in CLIPS] + [b / t for t in TAUS]
    return torch.stack([(-(belief_score(z, y, FLOOR)) * m).sum((1, 2)) / nb for z in outs], 1)


class Recorder:
    def __init__(self, B, J, recs, calib_at, dev):
        self.recs, self.calib_at, self.dev = sorted(set(recs)), set(calib_at), dev
        R = len(self.recs)
        self.idx = {t: i for i, t in enumerate(self.recs)}
        self.exact = np.zeros((R, B), np.uint8)
        self.ll = np.zeros((R, B, NCAL), np.float16)
        self.maxb = np.zeros((R, B), np.float16)
        self.meanb = np.zeros((R, B), np.float16)
        self.cyc = np.zeros((R, B), np.uint8)
        self.cov = np.zeros((R, B, J), np.float16) if J else None
        self.ece = {t: np.zeros((15, 3)) for t in calib_at}           # count, sum conf, sum correct
        self.absh = {t: np.zeros((2, len(ABS_EDGES) - 1)) for t in calib_at}   # correct, wrong


def run_curves(step_fn, init_fn, x, y, mask, Tmax, rec, off, cov_fn=None):
    """step_fn(state, t) -> (state, b); stores into rec at batch offset off"""
    B = x.shape[0]
    hist = []            # last 16 thresholded masks
    state = init_fn()
    m = mask
    nb = m.sum((1, 2))
    for t in range(1, Tmax + 1):
        state, b = step_fn(state, t)
        th = (b > 0) & (m > 0)
        hist.append(th)
        if len(hist) > 16:
            hist.pop(0)
        if t in rec.idx:
            i = rec.idx[t]
            sl = slice(off, off + B)
            pred = th.float()
            rec.exact[i, sl] = ((pred == y) | (m == 0)).flatten(1).all(1).cpu().numpy()
            rec.ll[i, sl] = ll_all(b, y, m).cpu().numpy()
            ab = b.abs() * m
            rec.maxb[i, sl] = ab.flatten(1).max(1)[0].clamp(max=6e4).cpu().numpy()
            rec.meanb[i, sl] = (ab.sum((1, 2)) / nb).clamp(max=6e4).cpu().numpy()
            per = torch.zeros(B, dtype=torch.uint8, device=b.device)
            L = len(hist)
            for p in range(8, 0, -1):          # smallest period wins (loop from large to small)
                if L >= 2 * p:
                    ok = torch.ones(B, dtype=torch.bool, device=b.device)
                    for k in range(p):
                        ok &= (hist[L - 1 - k] == hist[L - 1 - k - p]).flatten(1).all(1)
                    per = torch.where(ok, torch.full_like(per, p), per)
            rec.cyc[i, sl] = per.cpu().numpy()
            if cov_fn is not None and rec.cov is not None:
                rec.cov[i, sl] = cov_fn(state, y, m).cpu().numpy()
            if t in rec.calib_at:
                p1 = torch.sigmoid(b)
                conf = torch.maximum(p1, 1 - p1)[m > 0]
                corr = (th.float() == y)[m > 0].float()
                bins = torch.clamp(((conf - 0.5) * 2 * 15).long(), 0, 14)
                e = rec.ece[t]
                e[:, 0] += torch.bincount(bins, minlength=15).cpu().numpy()
                e[:, 1] += torch.bincount(bins, conf, minlength=15).cpu().numpy()
                e[:, 2] += torch.bincount(bins, corr, minlength=15).cpu().numpy()
                a_ = b.abs()[m > 0].cpu()
                h = rec.absh[t]
                h[0] += torch.histogram(a_[corr.cpu() > 0].float(), ABS_EDGES)[0].numpy()
                h[1] += torch.histogram(a_[corr.cpu() == 0].float(), ABS_EDGES)[0].numpy()
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True)           # trader | dtr | mref
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--jopt", default="none")          # trader: none | scale | cap
    p.add_argument("--mult", type=float, default=12)  # T = mult * n
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--bs", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)    # first maze index (split one set across jobs)
    p.add_argument("--frozen_p0", default=os.path.join(HERE, "frozen_p0"))
    p.add_argument("--frozen", default=os.path.join(HERE, "frozen"))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ck_every", type=float, default=300)   # resume checkpoint period in seconds (0: every batch)
    p.add_argument("--kill_after", type=int, default=0)     # test only: exit(3) after this many batches of this process
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = a.device
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.kind == "trader":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        targs = SimpleNamespace(**ck["args"]); targs.device = dev
        M = Model(targs, dev)
        for t_, sd in zip(M.traders, ck["traders"]):
            t_.load_state_dict(sd)
        M.traders.eval()
        M.wealth = list(ck.get("wealth") or [targs.W0] * targs.N)
        width, insq = targs.width, targs.insq
        global FLOOR
        FLOOR = targs.floor
    else:
        from dtnet import DTRecall, load_mref
        if a.kind == "mref":
            net = load_mref(a.ckpt, dev)
        else:
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            net = DTRecall(ck["arch"]["width"], dim=2, in_channels=3)
            net.load_state_dict(ck["model"]); net = net.to(dev).eval()
    summary = {"kind": a.kind, "ckpt": a.ckpt, "jopt": a.jopt, "mult": a.mult, "sets": {}}
    arrays = {}
    ckp = a.out + ".evck.npz"
    sig = {k: getattr(a, k) for k in ["kind", "ckpt", "sets", "jopt", "mult", "every", "bs", "limit", "start"]}
    resume = None
    if os.path.exists(ckp):
        z = np.load(ckp, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        if meta["sig"] == sig:
            resume = {"meta": meta, "arr": {k: z[k] for k in z.files if k != "__meta__"}}
            summary["sets"] = meta["sets_done"]
            arrays = {k: v for k, v in resume["arr"].items() if not k.startswith("__cur__/")}
            print(json.dumps({"resume": ckp, "sets_done": list(meta["sets_done"]), "cur": meta["cur"],
                              "next_o": meta["next_o"]}), flush=True)
        else:
            print(json.dumps({"resume_ignored": ckp, "sig_saved": meta["sig"]}), flush=True)
    nbatch = [0]

    def save_ck(cur, next_o, rec, sec):
        d = dict(arrays)
        if cur is not None:
            for k in ["exact", "ll", "maxb", "meanb", "cyc"]:
                d[f"__cur__/{k}"] = getattr(rec, k)
            if rec.cov is not None:
                d["__cur__/cov"] = rec.cov
            for t in rec.calib_at:
                d[f"__cur__/ece@{t}"] = rec.ece[t]
                d[f"__cur__/absh@{t}"] = rec.absh[t]
        meta = {"sig": sig, "sets_done": summary["sets"], "cur": cur, "next_o": next_o, "sec": sec}
        d["__meta__"] = np.array(json.dumps(meta))
        np.savez(a.out + ".evck.tmp.npz", **d)
        os.replace(a.out + ".evck.tmp.npz", ckp)

    for name in a.sets.split(","):
        if name in summary["sets"]:
            continue
        X, Y, n, sp = load_set(name, a.frozen_p0, a.frozen)
        if a.start or a.limit:
            e = a.start + a.limit if a.limit else None
            X, Y, sp = X[a.start:e], Y[a.start:e], sp[a.start:e]
        N, side = X.shape[0], X.shape[-1]
        T12 = 12 * n
        Tmax = int(round(a.mult * n))
        if a.kind == "trader":
            J, jscale = flops.jlevels(side, a.jopt)
            per_step = flops.trader_macs(J, width, insq=insq)
            teq = int(flops.mref_test_macs(n) // per_step) if n in flops.SIDES else T12
        else:
            J, jscale, teq = 0, 1.0, T12
            per_step = flops.dtr_macs()[1]
        teq = max(1, min(teq, Tmax))
        recs = list(range(a.every, Tmax + 1, a.every)) + [teq, min(T12, Tmax), Tmax]
        rec = Recorder(N, J, recs, [teq, min(T12, Tmax)], dev)
        bs = a.bs or max(1, int(4e5 // (side * side)) if a.kind == "trader" else int(2e6 // (side * side)))
        o0, sec0 = 0, 0.0
        if resume and resume["meta"]["cur"] == name:
            ra = resume["arr"]
            for k in ["exact", "ll", "maxb", "meanb", "cyc"]:
                getattr(rec, k)[...] = ra[f"__cur__/{k}"]
            if rec.cov is not None:
                rec.cov[...] = ra["__cur__/cov"]
            for t in rec.calib_at:
                rec.ece[t][...] = ra[f"__cur__/ece@{t}"]
                rec.absh[t][...] = ra[f"__cur__/absh@{t}"]
            o0, sec0 = resume["meta"]["next_o"], resume["meta"]["sec"]
        t0 = time.time() - sec0
        tck = time.time()
        with torch.no_grad():
            for o in range(o0, N, bs):
                x, y = X[o:o + bs].to(dev), Y[o:o + bs].to(dev)
                m = (x.max(1)[0] > 0).float()
                B = x.shape[0]
                if a.kind == "trader":
                    b0 = torch.zeros(B, side, side, device=dev)
                    c0 = torch.zeros(B, J, 2, side, side, device=dev)
                    v0 = torch.zeros(B, 3, side, side, device=dev)
                    eq, meq = pair_truth2d(y, m, J)
                    nbm = m.sum((1, 2))

                    def step_fn(st, t, x=x, m=m):
                        return trader_step(M, st, t, x, m, jscale)

                    def cov_fn(st, y, m, eq=eq, meq=meq, nbm=nbm):
                        c = st[1]
                        return ((-soft_score(c, eq) * meq).sum((2, 3, 4)) / meq.sum((2, 3, 4)).clamp(min=1))

                    run_curves(step_fn, lambda b0=b0, c0=c0, v0=v0: (b0, c0, v0), x, y, m, Tmax, rec, o, cov_fn)
                else:
                    def step_fn(h, t, x=x):
                        h = net.step(h, x)
                        lo = net.readout(h)
                        return h, lo[:, 1] - lo[:, 0]
                    run_curves(step_fn, lambda x=x: net.init(x), x, y, m, Tmax, rec, o)
                print(json.dumps({"set": name, "done": o + B, "of": N, "sec": round(time.time() - t0, 1)}), flush=True)
                nbatch[0] += 1
                if o + B < N and time.time() - tck >= a.ck_every:
                    save_ck(name, o + B, rec, time.time() - t0)
                    tck = time.time()
                if a.kill_after and nbatch[0] >= a.kill_after:
                    print(json.dumps({"killed_after_batches": nbatch[0]}), flush=True)
                    os._exit(3)
        ti = {t: rec.idx[t] for t in [teq, min(T12, Tmax), Tmax]}
        s = {"n": n, "N": N, "side": side, "J": J, "jscale": jscale, "t_eq": teq, "T12": T12, "Tmax": Tmax,
             "step_macs_per_pix": per_step, "sec": round(time.time() - t0, 1)}
        for lab, t in [("t_eq", teq), ("12n", min(T12, Tmax)), ("Tmax", Tmax)]:
            i = rec.idx[t]
            s[f"exact@{lab}"] = float(rec.exact[i].mean())
            s[f"ll_raw@{lab}"] = float(rec.ll[i, :, 0].astype(np.float64).mean())
            s[f"cycle@{lab}"] = float((rec.cyc[i] >= 2).mean())
            s[f"maxb@{lab}"] = float(rec.maxb[i].astype(np.float64).max())
        s["best_over_t_exact"] = float(rec.exact.mean(1).max())
        for t in rec.calib_at:
            e = rec.ece[t]; cnt = e[:, 0].sum()
            ece = float(np.abs(e[:, 1] - e[:, 2]).sum() / max(cnt, 1))
            h = rec.absh[t]
            # AUROC of detecting wrong pixels by low |b|: P(|b|_wrong < |b|_correct) + 0.5 P(tie)
            cw, cc = h[1], h[0]
            nw, nc = cw.sum(), cc.sum()
            if nw and nc:
                cum_c = np.cumsum(cc[::-1])[::-1] - cc   # correct pixels in strictly higher bins
                auc = float(((cw * cum_c).sum() + 0.5 * (cw * cc).sum()) / (nw * nc))
            else:
                auc = None
            s[f"ece@{t}"], s[f"auroc_err@{t}"] = ece, auc
        summary["sets"][name] = s
        arrays.update({f"{name}/t": np.array(rec.recs), f"{name}/exact": rec.exact, f"{name}/ll": rec.ll,
                       f"{name}/maxb": rec.maxb, f"{name}/meanb": rec.meanb, f"{name}/cyc": rec.cyc,
                       f"{name}/sp_len": sp})
        if rec.cov is not None:
            arrays[f"{name}/cov"] = rec.cov
        for t in rec.calib_at:
            arrays[f"{name}/ece@{t}"] = rec.ece[t]; arrays[f"{name}/absh@{t}"] = rec.absh[t]
        print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}), flush=True)
        save_ck(None, 0, None, 0.0)
    np.savez_compressed(a.out + ".tmp.npz", **arrays)
    os.replace(a.out + ".tmp.npz", a.out + ".npz")
    summary["taus"], summary["clips"] = TAUS, CLIPS
    json.dump(summary, open(a.out + ".json.tmp", "w"), indent=1)
    os.replace(a.out + ".json.tmp", a.out + ".json")
    if os.path.exists(ckp):
        os.replace(ckp, a.out + ".evck.used.npz")


def trader_step(M, st, t, x, m, jscale=1.0):
    """one test step t (1-based), trader (t-1) mod N (round robin). An econ model trained with the wealth cap
    trades wealth-scaled with its final wealth, as econ2's evaluation does (Model.rollout, pick roundrobin)."""
    from econ_maze import liq_beta, worst_loss_per_pix
    b, c, v = st
    b = M.relax(b, t - 1)
    i = (t - 1) % len(M.traders)
    db, dc, dv, _ = M.traders[i](x, b, c, v, m, jscale)
    db = M.tdb(db, liq_beta(M.a.liq, t))
    if M.a.mode == "econ" and M.a.wealth_cap:
        s = torch.clamp(M.a.frac * max(M.wealth[i], 0.0) / (worst_loss_per_pix(b, db, m) + 1e-8), max=1.0)
        B = b.shape[0]
        b = b + s.view(B, 1, 1) * db; v = v + s.view(B, 1, 1, 1) * dv; c = c + s.view(B, 1, 1, 1, 1) * dc
    else:
        b = b + db; v = v + dv; c = c + dc
    b = M.bclamp(b)
    return (b, c, v), b


if __name__ == "__main__":
    main()
