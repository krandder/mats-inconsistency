"""P17 trajectory bank: run the released EqR Sudoku-Extreme checkpoint on a fixed puzzle sample, many restarts,
record checkable per-step signals for every (puzzle, restart, depth). Every allocation policy (market and baselines)
is then simulated offline from this bank with exact NFE accounting: restart k stopped at depth d costs d NFE, and a
trajectory's prefix does not depend on how far it is later run (noise is drawn per step), so any allocation inside
the staircase is exactly realisable.

Staircase (restart index k -> max depth): k 0-1: 1024, 2-3: 512, 4-7: 256, 8-15: 128, 16-31: 64, 32-63: 32,
64-127: 16  (8192 NFE per puzzle). One NFE = one outer EqR step (H_cycles x (L_cycles + 1) = 21 L-level calls).

Per step t (1-indexed) and row we store:
  delta  f16  mean over the 81 cells of ||logits_t - logits_{t-1}||_2   (EqR's convergence residual; NaN at t=1)
  viol   u8   checkable constraint violations of argmax grid: non-digit cells + duplicate digits in rows, cols,
              boxes (count-1 per digit per unit) + given-cell mismatches.  viol == 0  <=> grid is a valid completion
  corr   bool grid == label (used only for scoring, never by a policy)
  q      f16  EqR q_halt logit (learned; recorded, not used by the pre-registered market)
  cmean  f16  mean over cells of max softmax prob;  cmin f16 min over cells of max softmax prob
  h      i32  hash of the argmax grid (majority vote)

Usage: python bank.py --eqr EQR_REPO --ckpt CKPT --data DATA_DIR --out OUT --blocks 0-11 [--device cuda]
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

GROUPS = [  # (group id, first restart, n restarts, depth cap)
    (0, 0, 2, 1024), (1, 2, 2, 512), (2, 4, 4, 256), (3, 8, 8, 128), (4, 16, 16, 64), (5, 32, 32, 32), (6, 64, 64, 16)]
BLOCK = 256          # puzzles per block
N_VAL_BLOCKS = 4     # blocks 0-3 = val (1024 puzzles), blocks 4-11 = test (2048 puzzles)
SAMPLE_SEED = 17
MAX_ROWS = 4096


def sample_indices(n_total, n_blocks):
    rng = np.random.default_rng(SAMPLE_SEED)
    return rng.permutation(n_total)[: n_blocks * BLOCK]


def load_model(eqr_repo, ckpt_path, device, noise_scale=0.5, init_std=1.0):
    sys.path.insert(0, eqr_repo)
    from models.eqr import EqRConfig, InnerNetwork  # noqa
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck["config"]["arch"]
    cfg = EqRConfig(batch_size=1, seq_len=81 + int(arch["puzzle_emb_len"]), input_seq_len=81, num_puzzle_identifiers=1,
                    vocab_size=11, H_cycles=int(arch["H_cycles"]), L_cycles=int(arch["L_cycles"]),
                    H_layers=int(arch["H_layers"]), L_layers=int(arch["L_layers"]), hidden_size=int(arch["hidden_size"]),
                    expansion=float(arch["expansion"]), num_heads=int(arch["num_heads"]), pos_encodings="none",
                    halt_max_steps=64, halt_exploration_prob=0.1, forward_dtype="bfloat16", mlp_t=bool(arch["mlp_t"]),
                    puzzle_emb_len=int(arch["puzzle_emb_len"]), puzzle_emb_ndim=int(arch["puzzle_emb_ndim"]),
                    lambda_=float(arch["lambda_"]), noise_scale=noise_scale, H_init_std=init_std, L_init_std=init_std)
    net = InnerNetwork(cfg)
    # as utils/checkpoint.load_ema: model state first (puzzle_emb.weights is a buffer, not in the EMA), then the
    # EMA shadow overwrites every parameter
    pre = "_orig_mod.model.inner."
    sd = {k[len(pre):]: v for k, v in ck["model"].items() if k.startswith(pre)}
    sd.update({k[len(pre):]: v for k, v in ck["ema"]["shadow"].items() if k.startswith(pre)})
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith(("H_init", "L_init", "puzzle_emb.local", "rotary"))]
    assert not missing and not [u for u in unexpected if "local_" not in u], (missing, unexpected)
    net.eval().to(device)
    return net, cfg


def violations(preds, inputs):
    """preds, inputs: (R, 81) token ids (1 = blank/'0', 2..10 = digits 1..9). Returns (R,) int violation counts."""
    d = preds.long() - 1
    isdig = (d >= 1) & (d <= 9)
    oh = F.one_hot(d.clamp(0, 9), 10)[..., 1:] * isdig[..., None]           # (R, 81, 9)
    g = oh.view(-1, 9, 9, 9)                                                   # r, c, v
    dup = lambda cnt: (cnt - 1).clamp(min=0).sum(dim=(1, 2))
    rows, cols = g.sum(2), g.sum(1)
    boxes = g.view(-1, 3, 3, 3, 3, 9).sum(dim=(2, 4)).view(-1, 9, 9)
    given = inputs.long() > 1
    mism = (given & (preds.long() != inputs.long())).sum(1)
    return (~isdig).sum(1) + dup(rows) + dup(cols) + dup(boxes) + mism


@torch.inference_mode()
def run_rows(net, cfg, inputs, labels, depth, seed, device):
    """inputs/labels (R, 81) uint8 on device. Runs `depth` outer steps from a fresh random init."""
    torch.manual_seed(seed)
    R = inputs.shape[0]
    from models.common import trunc_normal_init_
    shape = (R, cfg.seq_len, cfg.hidden_size)
    zH = trunc_normal_init_(torch.empty(shape, dtype=torch.bfloat16, device=device), std=cfg.H_init_std)
    zL = trunc_normal_init_(torch.empty(shape, dtype=torch.bfloat16, device=device), std=cfg.L_init_std)
    from models.eqr import LatentCarry
    carry = LatentCarry(zH, zL)
    batch = {"inputs": inputs.int(), "puzzle_identifiers": torch.zeros(R, dtype=torch.int32, device=device)}
    hv = torch.randint(1, 2**62, (81,), generator=torch.Generator().manual_seed(12345)).to(device)
    out = {k: torch.empty((R, depth), dtype=dt, device=device) for k, dt in
           [("delta", torch.float16), ("viol", torch.uint8), ("corr", torch.bool), ("q", torch.float16),
            ("cmean", torch.float16), ("cmin", torch.float16), ("h", torch.int32)]}
    prev = None
    for t in range(depth):
        carry, logits, q = net(carry, batch)
        lg = logits.float()
        if prev is None:
            out["delta"][:, t] = float("nan")
        else:
            out["delta"][:, t] = (lg - prev).norm(dim=-1).mean(-1).half()
        prev = lg
        preds = lg.argmax(-1)
        pm = lg.softmax(-1).amax(-1)
        out["viol"][:, t] = violations(preds, inputs).clamp(max=255).to(torch.uint8)
        out["corr"][:, t] = (preds == labels.long()).all(-1)
        out["q"][:, t] = q.half()
        out["cmean"][:, t] = pm.mean(-1).half()
        out["cmin"][:, t] = pm.amin(-1).half()
        out["h"][:, t] = ((preds.long() * hv).sum(-1) & 0x7FFFFFFF).int()
    return {k: v.cpu().numpy() for k, v in out.items()}


def parse_blocks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqr", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0-11"); ap.add_argument("--groups", default="0-6")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--block_size", type=int, default=BLOCK)
    ap.add_argument("--depth_scale", type=float, default=1.0, help="smoke tests only: shrink the depth caps")
    ap.add_argument("--max_rows", type=int, default=MAX_ROWS)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--noise_scale", type=float, required=True,
                    help="EqR path-noise beta (PREREG amendment 1: 0.01, the paper's method setting)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    X = np.load(os.path.join(a.data, "test", "all__inputs.npy"))
    Y = np.load(os.path.join(a.data, "test", "all__labels.npy"))
    blocks, groups = parse_blocks(a.blocks), parse_blocks(a.groups)
    idx = sample_indices(len(X), max(blocks) + 1) if a.block_size == BLOCK else \
        np.random.default_rng(SAMPLE_SEED).permutation(len(X))[: (max(blocks) + 1) * a.block_size]
    net, cfg = load_model(a.eqr, a.ckpt, dev, noise_scale=a.noise_scale)
    if a.compile:
        net = torch.compile(net)
    meta = {"sample_seed": SAMPLE_SEED, "block": a.block_size, "groups": GROUPS, "depth_scale": a.depth_scale,
            "ckpt_sha256": hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest(), "noise_scale": cfg.noise_scale,
            "init_std": cfg.H_init_std, "lambda": cfg.lambda_, "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__}
    mp = os.path.join(a.out, "bank_meta.json")
    if os.path.exists(mp):  # resume: never mix units made with different settings in one bank
        old = json.load(open(mp))
        for k in ("noise_scale", "init_std", "lambda", "depth_scale", "ckpt_sha256", "block"):
            assert old.get(k) == meta.get(k), f"bank_meta mismatch on {k}: {old.get(k)} vs {meta.get(k)}"
    json.dump(meta, open(mp, "w"), indent=1)
    for b in blocks:
        pidx = idx[b * a.block_size:(b + 1) * a.block_size]
        np.save(os.path.join(a.out, f"b{b:02d}_idx.npy"), pidx)
        for gid, k0, K, D in [g for g in GROUPS if g[0] in groups]:
            D = max(4, int(round(D * a.depth_scale)))
            fn = os.path.join(a.out, f"b{b:02d}_g{gid}.npz")
            if os.path.exists(fn):
                continue
            t0 = time.time()
            P = len(pidx)
            per = max(1, a.max_rows // K)            # puzzles per sub-batch
            parts = []
            for s in range(0, P, per):
                sub = pidx[s:s + per]
                xi = torch.from_numpy(X[sub]).to(dev).repeat_interleave(K, 0)
                yi = torch.from_numpy(Y[sub]).to(dev).repeat_interleave(K, 0)
                seed = (b * 1000003 + gid * 7919 + s) % (2**31)
                r = run_rows(net, cfg, xi, yi, D, seed, dev)
                parts.append({k: v.reshape(len(sub), K, D) for k, v in r.items()})
            arr = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
            arr.update(k0=np.array(k0), D=np.array(D))
            tmp = fn[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **arr)
            os.replace(tmp, fn)
            dt = time.time() - t0
            rate = P * K * D / dt
            print(json.dumps({"block": b, "group": gid, "P": P, "K": K, "D": D, "sec": round(dt, 1),
                              "row_nfe_per_s": round(rate, 1),
                              "acc_last": float(arr["corr"][:, :, -1].mean()),
                              "valid_not_correct": int(((arr["viol"] == 0) & ~arr["corr"]).sum())}), flush=True)
    print("BANK_DONE", flush=True)


if __name__ == "__main__":
    main()
