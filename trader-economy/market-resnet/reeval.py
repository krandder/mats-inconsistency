"""Re-evaluation at 100 times the training length (L = 3200), for the metric and runaway questions.

For one trained checkpoint (econ2 econ/bptt or dtrecall), on a fresh test set of --n strings of
length 3200 (seed 777 + L, disjoint from eval_depths' seed 555 + L), round-robin order, at depths
320, 1600, 3200, reports:
  acc        exact-string accuracy (every one of the 3200 output bits right; the headline metric)
  bit_acc    fraction of output bits right
  nll_bit    mean log loss, nats per bit (as in RESULTS.md)
  nll_clip10 the same with every logit clipped to +-10 after the rollout (a cap on confidence)
  wrong-bit anatomy of the failed strings: wrong bits per failed string, contiguous runs of
             wrong bits, median |logit| on wrong and on right bits, share of wrong bits past |logit| 100
With --clamp C (econ2 models only) it also reruns the rollout with b, c and v clamped to +-C after
every step (traders read tanh(z/4), so values past about 20 are invisible to them) and reports the
same numbers: if the runaway fed back into the traders' behaviour, accuracy would change.
Per-string outcomes are saved so intervals can be computed over strings as well as seeds.
nll_bit is the model's own forecast (with --floor, the floored one); nll_raw ignores the floor.
With --deep 1 (default for reeval.DEEP_ARMS) it also runs extreme depths, without clamping: L = 32 at depths 24, 240,
2400, 24000 (n = 200) and L = 320 at depths 320, 3200, 32000 (n = 50).
"""
import argparse, json, os, re, types
import torch
from mresnet import prefix_batch
from econ2 import Economy, BPTT, soft_score, belief_score, nlevels

DEPTHS = [320, 1600, 3200]
DEEP = [(32, [24, 240, 2400, 24000], 200), (320, [320, 3200, 32000], 50)]
# the clamp-20 L3200 block (does the logit runaway cost accuracy?) only for these arms: the unregularized
# economy and its backprop controls (09-29 20:55 UTC: an n = 200 L3200 block takes ~40 min of shared GPU)
# (21:15 UTC: cut to the economy, its gradient twin and full BPTT; ~14 GPU-minutes per block; 21:45 UTC: the
# economy and full BPTT, the two arms of the runaway question)
CLAMP_ARMS = {"econ_cov_voc", "bptt_cov_dscov"}
# the extreme-depth evals (21:45 UTC) only for the arms whose depth behaviour is reported: the economy, full BPTT,
# the gradient twin, the regularizer and liquidity arms, the cap controls and DT-recall (not the order/VOC ablations)
DEEP_ARMS = {"econ_cov_voc", "bptt_cov_dscov", "bptt_cov_dscov_tb1_voc_rand", "econ_cov_voc_floor", "econ_cov_voc_l2b",
             "econ_cov_voc_bcap", "econ_cov_voc_dbcap", "econ_cov_voc_liqt", "econ_cov_voc_liqsqrt",
             "bptt_cov_dscov_tb1_voc_bcap_rand", "bptt_cov_dscov_bcap", "dtr_full", "dtr_eqflop"}


def arm_of(ckpt):
    return re.sub(r"_s\d+\.pt$", "", os.path.basename(ckpt))


def build(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "net" in ck:
        from dtrecall import DT
        a = types.SimpleNamespace(**ck["args"])
        M = DT(a, device); M.net.load_state_dict(ck["net"])
        return M, "dtr", a
    a = types.SimpleNamespace(**ck["args"])
    for k, v in dict(tbptt=0, train_pick="", clamp=0, floor=0, l2b=0, bcap=0, dbcap=0, liq="").items():
        if not hasattr(a, k):
            setattr(a, k, v)
    if a.mode == "econ":
        M = Economy(a, device)
        while len(M.traders) < len(ck["traders"]):
            M.add_trader()
        for t, sd in zip(M.traders, ck["traders"]):
            t[0].load_state_dict(sd)
        if ck.get("wealth"):
            M.wealth = list(ck["wealth"])
        else:   # phase-2 checkpoints kept no wealth: the last logged wealth (iteration 2999), as econ2.py --load does
            M.wealth = list(json.load(open(path[:-3] + ".json"))["log"][-1]["wealth"])
    else:
        M = BPTT(a, device); M.traders.load_state_dict(ck["traders"])
    return M, a.mode, a


def runs(b, ys):
    """number of maximal runs of wrong bits per string"""
    w = ((b > 0).float() != ys).int()
    starts = w[:, :1] + (w[:, 1:] - w[:, :-1]).clamp(min=0).sum(1, keepdim=True)
    return starts.squeeze(1)


@torch.no_grad()
def analyse(M, kind, L, n, chunk, device, clamp=0.0, depths=DEPTHS, part=None):
    """part (22:45 UTC, after many spot preemptions): a file that holds the finished chunks of this block, written
    after every chunk, so a relaunched VM resumes the block at its next chunk (the test strings are fixed by the
    seed, so the resumed block is identical to an uninterrupted one)."""
    fl = getattr(M.a, "floor", 0) or 0
    if kind != "dtr":
        M.a.clamp = clamp
    gen = torch.Generator().manual_seed(777 + L)
    x, y, m = prefix_batch(n, L, L, gen, device)
    per = {D: {"ok": [], "bit": [], "nll": [], "nllraw": [], "nll10": [], "nwrong": [], "runs": [], "wabs": [], "rabs": []}
           for D in depths}
    key = [L, n, chunk, clamp, list(depths)]
    start = 0
    if part and os.path.exists(part):
        try:
            z = json.load(open(part))
            if z["key"] == key:
                per = {int(D): v for D, v in z["per"].items()}; start = z["next"]
                print(f"resuming L{L} clamp {clamp} at string {start} of {n}", flush=True)
        except (OSError, ValueError, KeyError):
            pass
    for c0 in range(start, n, chunk):
        xs, ys, ms = x[c0:c0 + chunk], y[c0:c0 + chunk], m[c0:c0 + chunk]
        _, bs, _ = M.rollout(xs, ms, max(depths), pick="roundrobin", keep=False, record_at=set(depths))
        for D in depths:
            b = bs[D]; P = per[D]
            wrong = (b > 0).float() != ys
            P["ok"] += (~wrong).all(1).float().tolist()
            P["bit"] += (1 - wrong.float().mean(1)).tolist()
            P["nll"] += (-belief_score(b, ys, fl).mean(1)).tolist()
            P["nllraw"] += (-soft_score(b, ys).mean(1)).tolist()
            P["nll10"] += (-soft_score(b.clamp(-10, 10), ys).mean(1)).tolist()
            P["nwrong"] += wrong.sum(1).tolist()
            P["runs"] += runs(b, ys).tolist()
            wa = b.abs()[wrong]
            P["wabs"] += wa[::max(1, wa.numel() // 5000)].tolist()
            P["rabs"] += b.abs()[~wrong][::97].tolist()
        del bs
        if part and c0 + chunk < n:
            json.dump({"key": key, "next": c0 + chunk, "per": per}, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    if kind != "dtr":
        M.a.clamp = 0.0
    out = []
    for D in depths:
        P = per[D]
        ok = torch.tensor(P["ok"]); nw = torch.tensor(P["nwrong"], dtype=torch.float); rn = torch.tensor(P["runs"], dtype=torch.float)
        wa = torch.tensor(P["wabs"]) if P["wabs"] else torch.zeros(0); ra = torch.tensor(P["rabs"])
        fail = ok == 0
        out.append({"L": L, "D": D, "n": n, "clamp": clamp, "acc": ok.mean().item(),
                    "bit_acc": sum(P["bit"]) / n, "nll_bit": sum(P["nll"]) / n, "nll_raw": sum(P["nllraw"]) / n,
                    "nll_clip10": sum(P["nll10"]) / n, "abs_logit_median": torch.cat([wa, ra]).median().item(),
                    "n_fail": int(fail.sum()),
                    "wrong_bits_per_fail_median": nw[fail].median().item() if fail.any() else 0,
                    "wrong_bits_per_fail_mean": nw[fail].mean().item() if fail.any() else 0,
                    "wrong_runs_per_fail_median": rn[fail].median().item() if fail.any() else 0,
                    "wrong_abs_logit_median": wa.median().item() if len(wa) else None,
                    "wrong_share_abs_logit_gt100": (wa > 100).float().mean().item() if len(wa) else None,
                    "right_abs_logit_median": ra.median().item(),
                    "per_string_ok": P["ok"], "per_string_wrong_bits": P["nwrong"]})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--L", type=int, default=3200)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--clamp", type=float, default=-1.0)   # -1: 20 for CLAMP_ARMS, else no clamp block
    p.add_argument("--deep", type=int, default=-1)   # -1: 1 for DEEP_ARMS, else 0
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    M, kind, margs = build(a.ckpt, a.device)
    if a.clamp < 0:
        a.clamp = 20.0 if arm_of(a.ckpt) in CLAMP_ARMS else 0.0
    if a.deep < 0:
        a.deep = int(arm_of(a.ckpt) in DEEP_ARMS)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"ckpt": a.ckpt, "kind": kind, "rows": []}
    have = {(r["clamp"], r["L"]) for r in res["rows"]}
    jobs = [(a.L, DEPTHS, a.n, cl) for cl in ([0.0] + ([a.clamp] if a.clamp and kind != "dtr" else []))]
    jobs += [(L, ds, n, 0.0) for L, ds, n in DEEP] if a.deep else []
    for L, ds, n, cl in jobs:
        if (cl, L) in have:
            continue
        ch = min(n, a.chunk if kind != "dtr" else 20) if L > 320 else n
        rows = analyse(M, kind, L, n, ch, a.device, clamp=cl, depths=ds, part=a.out + ".part")
        res["rows"] += rows
        print(json.dumps([{k: v for k, v in r.items() if not k.startswith("per_")} for r in rows]), flush=True)
        json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
        if os.path.exists(a.out + ".part"):
            os.remove(a.out + ".part")   # the block is in the JSON; its chunk file is no longer needed
