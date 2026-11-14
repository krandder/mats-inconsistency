"""P18 evaluation: exact-maze curves every 10 steps to T = mult * n. Copied from P13 eval_maze.py with the Trader2D
parts removed and the K-block model added (round robin at test: step t uses block (t-1) mod K).

  python eval18.py --ckpt X.pt --sets v9,m13,v33,m59,e19,e33,e59 --out out/ev_NAME
  python eval18.py --kind mref --ckpt mref/dt_net.pth --sets m13,m59 --out out/ev_mref

Per record step t (every 10 steps, plus 12n and Tmax) and maze: exact (argmax prediction on the open-pixel mask must
equal the solution on every open pixel; b = logit_1 - logit_0 > 0), log loss per open pixel (raw, clips 4/8/16,
temperatures), max and mean |b|, limit-cycle period (<= 8). TF32 off. Unchanged from P13: the record steps, the
exact rule, the batch size int(2e6 // side^2), and the preemption resume file OUT.evck.npz (a rerun with the same
arguments skips finished sets and batches; batches are independent, so the output equals an uninterrupted run).
A checkpoint whose training diverged (non-finite loss) is evaluated as saved.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LS = F.logsigmoid
CLIPS = [4.0, 8.0, 16.0]
TAUS = [2 ** (k / 2) for k in range(-2, 17)]          # 0.5 .. 256
NCAL = 1 + len(CLIPS) + len(TAUS)
SETS_P0 = {"v9", "m13", "v33", "m59"}                  # P0 frozen dev sets (m13 -> first 200)


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


def ll_all(b, y, m):
    """(B, NCAL) mean log loss per open pixel: raw, clips, temperatures (binary log score of logit b)"""
    nb = m.sum((1, 2))
    yf = y.float()
    outs = [b] + [b.clamp(-c, c) for c in CLIPS] + [b / t for t in TAUS]
    return torch.stack([(-(yf * LS(z) + (1 - yf) * LS(-z)) * m).sum((1, 2)) / nb for z in outs], 1)


class Recorder:
    def __init__(self, B, recs):
        self.recs = sorted(set(recs))
        R = len(self.recs)
        self.idx = {t: i for i, t in enumerate(self.recs)}
        self.exact = np.zeros((R, B), np.uint8)
        self.ll = np.zeros((R, B, NCAL), np.float16)
        self.maxb = np.zeros((R, B), np.float16)
        self.meanb = np.zeros((R, B), np.float16)
        self.cyc = np.zeros((R, B), np.uint8)


def run_curves(step_fn, init_fn, x, y, mask, Tmax, rec, off):
    """step_fn(state, t) -> (state, b); stores into rec at batch offset off"""
    B = x.shape[0]
    hist = []
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
            for p in range(8, 0, -1):
                if L >= 2 * p:
                    ok = torch.ones(B, dtype=torch.bool, device=b.device)
                    for k in range(p):
                        ok &= (hist[L - 1 - k] == hist[L - 1 - k - p]).flatten(1).all(1)
                    per = torch.where(ok, torch.full_like(per, p), per)
            rec.cyc[i, sl] = per.cpu().numpy()
    return state


def load_model(kind, path, dev):
    from dtnet import DTRecall, load_mref
    from dtmod import DTRecallK
    if kind == "mref":
        return load_mref(path, dev), "dt2d", 1, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ar = ck["arch"]
    if ar["kind"] == "dt2d":
        net = DTRecall(ar["width"], dim=2, in_channels=3); K = 1
    else:
        net = DTRecallK(ar["width"], K=ar["K"]); K = ar["K"]
    net.load_state_dict(ck["model"])
    return net.to(dev).eval(), ar["kind"], K, ck.get("diverged")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="auto")          # auto (from the checkpoint's arch) | mref
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--mult", type=float, default=12)  # T = mult * n
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--bs", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frozen_p0", default=os.path.join(HERE, "frozen_p0"))
    p.add_argument("--frozen", default=os.path.join(HERE, "frozen"))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ck_every", type=float, default=300)
    p.add_argument("--kill_after", type=int, default=0)     # test only: exit(3) after this many batches of this process
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = a.device
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    net, akind, K, diverged = load_model(a.kind, a.ckpt, dev)
    summary = {"kind": a.kind, "arch_kind": akind, "K": K, "ckpt": a.ckpt, "mult": a.mult, "diverged": diverged, "sets": {}}
    arrays = {}
    ckp = a.out + ".evck.npz"
    sig = {k: getattr(a, k) for k in ["kind", "ckpt", "sets", "mult", "every", "bs", "limit", "start"]}
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
        recs = list(range(a.every, Tmax + 1, a.every)) + [min(T12, Tmax), Tmax, 30]
        rec = Recorder(N, [r for r in recs if r <= Tmax])
        bs = a.bs or int(2e6 // (side * side))
        o0, sec0 = 0, 0.0
        if resume and resume["meta"]["cur"] == name:
            ra = resume["arr"]
            for k in ["exact", "ll", "maxb", "meanb", "cyc"]:
                getattr(rec, k)[...] = ra[f"__cur__/{k}"]
            o0, sec0 = resume["meta"]["next_o"], resume["meta"]["sec"]
        t0 = time.time() - sec0
        tck = time.time()
        with torch.no_grad():
            for o in range(o0, N, bs):
                x, y = X[o:o + bs].to(dev), Y[o:o + bs].to(dev)
                m = (x.max(1)[0] > 0).float()
                B = x.shape[0]
                if akind == "dt2d":
                    def step_fn(h, t, x=x):
                        h = net.step(h, x)
                        lo = net.readout(h)
                        return h, lo[:, 1] - lo[:, 0]
                else:
                    def step_fn(h, t, x=x):
                        h = net.step(h, x, (t - 1) % K)
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
        s = {"n": n, "N": N, "side": side, "T12": T12, "Tmax": Tmax, "sec": round(time.time() - t0, 1)}
        for lab, t in [("30", 30), ("12n", min(T12, Tmax)), ("Tmax", Tmax)]:
            if t not in rec.idx:
                continue
            i = rec.idx[t]
            s[f"exact@{lab}"] = float(rec.exact[i].mean())
            s[f"ll_raw@{lab}"] = float(rec.ll[i, :, 0].astype(np.float64).mean())
            s[f"cycle@{lab}"] = float((rec.cyc[i] >= 2).mean())
            s[f"maxb@{lab}"] = float(rec.maxb[i].astype(np.float64).max())
        em = rec.exact.mean(1)
        s["best_over_t_exact"] = float(em.max())
        s["best_t"] = int(rec.recs[int(em.argmax())])
        summary["sets"][name] = s
        arrays.update({f"{name}/t": np.array(rec.recs), f"{name}/exact": rec.exact, f"{name}/ll": rec.ll,
                       f"{name}/maxb": rec.maxb, f"{name}/meanb": rec.meanb, f"{name}/cyc": rec.cyc,
                       f"{name}/sp_len": sp})
        print(json.dumps({"set": name, **{k: v for k, v in s.items()}}), flush=True)
        save_ck(None, 0, None, 0.0)
    np.savez_compressed(a.out + ".tmp.npz", **arrays)
    os.replace(a.out + ".tmp.npz", a.out + ".npz")
    summary["taus"], summary["clips"] = TAUS, CLIPS
    json.dump(summary, open(a.out + ".json.tmp", "w"), indent=1)
    os.replace(a.out + ".json.tmp", a.out + ".json")
    if os.path.exists(ckp):
        os.replace(ckp, a.out + ".evck.used.npz")


if __name__ == "__main__":
    main()
