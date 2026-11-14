"""M2/w4 stage 2 (PREREG.md): perturb the augmented tracr net, fine-tune with Adam, keep books.

Accumulated per-unit predictors over training: books_signed = sum_t -dL/dgate (signed a.g credit),
books_abs = sum_t |dL/dgate|, movement = sum_t sum_{theta in unit} -theta*dL/dtheta (Sanh et al. 2020).
End-state: oracle_zero (exact zero ablation), taylor, loo_direct, magnitude.
Run: PYTHONPATH=/tmp/traders-net/M2/w4/pylib python3 stage2.py --prog reverse --seed 1
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import argparse, csv, json, time
import numpy as np, jax, jax.numpy as jnp
import tracr_gt as T

BETA, STEPS, LR, PERT = 5.0, 2000, 1e-3, 0.05


def unit_param_score(P, gP, sl):
    """sum over each unit's own parameters of -theta*grad."""
    U = sl[-1][1][1]
    s = np.zeros(U)
    for l, (ly, gl) in enumerate(zip(P, gP)):
        (h0, h1), (n0, n1) = sl[l]
        m = {kk: -np.asarray(ly[kk]) * np.asarray(gl[kk]) for kk in ly}
        s[h0:h1] = sum(m[kk].reshape(h1 - h0, -1).sum(1) for kk in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo"))
        s[n0:n1] = m["W1"].sum(0) + m["b1"] + m["W2"].sum(1)
    return s


def main(prog, seed):
    t0 = time.time()
    obj = T.compile_cached(prog)
    P, x0, meta = T.augment(obj, seed, 0.15)
    # same noise-halving rule as stage 1
    nf = 0.15
    while True:
        f = T.make_fns(P, x0, obj["labels"], obj["out_dims"], BETA)
        acc = float(f["correct"](jnp.ones((1, f["U"])), jnp.zeros((f["U"], x0.shape[-1]))))
        if acc >= 1.0 or nf < 1e-3:
            break
        nf /= 2
        P, x0, meta = T.augment(obj, seed, nf)
    rng = np.random.default_rng(500 + seed)
    # perturb nonzero entries only (keeps Z/D/junk structure = ground-truth classes intact)
    for ly in P:
        for kk, v in ly.items():
            nz = v != 0
            if nz.any():
                ly[kk] = (v + nz * rng.normal(0, PERT * v[nz].std() + 1e-12, v.shape)).astype(np.float32)
    masks = [{kk: jnp.asarray(v != 0, jnp.float32) for kk, v in ly.items()} for ly in P]
    f = T.make_fns(P, x0, obj["labels"], obj["out_dims"], BETA)
    sl, U, fwd, ce = f["sl"], f["U"], f["forward"], f["ce"]
    X0, Y = f["C0"][1], f["C0"][2]
    Z0 = jnp.zeros((U, x0.shape[-1]))

    def loss_PG(Pj, G):
        C = (Pj, X0, Y)
        return ce(C, fwd(C, G, Z0)[0])

    vg = jax.jit(jax.value_and_grad(loss_PG, argnums=(0, 1)))
    Pj = jax.tree_util.tree_map(jnp.asarray, P)
    m1 = jax.tree_util.tree_map(jnp.zeros_like, Pj); m2 = jax.tree_util.tree_map(jnp.zeros_like, Pj)
    books_signed = np.zeros(U); books_abs = np.zeros(U); movement = np.zeros(U)
    ones = jnp.ones((1, U))
    L_start = None
    for t in range(1, STEPS + 1):
        L, (gP, gG) = vg(Pj, ones)
        gP = jax.tree_util.tree_map(lambda g, m: g * m, gP, masks)
        if L_start is None:
            L_start = float(L)
        gGn = np.asarray(gG)[0]
        books_signed += -gGn; books_abs += np.abs(gGn)
        movement += unit_param_score(Pj, gP, sl)
        m1 = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, m1, gP)
        m2 = jax.tree_util.tree_map(lambda m, g: 0.999 * m + 0.001 * g * g, m2, gP)
        Pj = jax.tree_util.tree_map(lambda p, a, b: p - LR * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8),
                                    Pj, m1, m2)
        if t % 500 == 0:
            print(f"{prog} s{seed} step {t} loss {float(L):.4g} t={time.time()-t0:.0f}s", flush=True)
    # end-state oracle and predictors
    Pf = jax.tree_util.tree_map(np.asarray, Pj)
    g = T.make_fns(Pf, x0, obj["labels"], obj["out_dims"], BETA)
    base = float(g["loss"](ones, Z0))
    acc_end = float(g["correct"](ones, Z0))
    oz = np.zeros(U)
    for i in range(U):
        G = np.ones((1, U), np.float32); G[0, i] = 0
        oz[i] = float(g["loss"](jnp.asarray(G), Z0)) - base
    taylor = -np.asarray(g["grad_G"](ones, Z0))[0]
    xfin, writes = g["full"](Z0)
    OD = obj["out_dims"]; Yj = jnp.asarray(obj["labels"])
    loo = np.zeros(U)
    for l, (w, a) in enumerate(writes):
        (h0, h1), (n0, n1) = sl[l]
        for rng_, wo in (((h0, h1), w[..., OD]), ((n0, n1), a[..., None] * jnp.asarray(Pf[l]["W2"][:, OD])[None, None])):
            logits = BETA * (xfin[:, 1:, None, :][..., OD] - wo[:, 1:])
            lp = jax.nn.log_softmax(logits, -1)
            nll = -jnp.take_along_axis(lp, Yj[:, :, None, None].repeat(wo.shape[2], 2), -1)[..., 0]
            loo[rng_[0]:rng_[1]] = np.asarray(nll.mean((0, 1))) - base
    mag = []
    for l, ly in enumerate(Pf):
        mag += [float(np.linalg.norm(ly["Wv"][h] @ ly["Wo"][h])) for h in range(ly["Wq"].shape[0])]
        mag += [float(np.linalg.norm(ly["W2"][j])) for j in range(ly["W1"].shape[1])]
    rows = [dict(prog=prog, seed=seed, uid=i, **{kk: m[kk] for kk in ("gran", "layer", "cls", "pair")},
                 oracle_zero=oz[i], taylor=taylor[i], loo_direct=loo[i], magnitude=mag[i],
                 books_signed=books_signed[i], books_abs=books_abs[i], movement=movement[i],
                 random=rng.random()) for i, m in enumerate(meta)]
    with open(f"{T.HERE}/results/s2units_{prog}_s{seed}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    info = dict(prog=prog, seed=seed, noise_frac=nf, loss_start=L_start, loss_end=base, acc_end=acc_end,
                steps=STEPS, lr=LR, pert=PERT, beta=BETA, U=U, wall_s=time.time() - t0)
    json.dump(info, open(f"{T.HERE}/results/s2meta_{prog}_s{seed}.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prog"); ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args(); STEPS = a.steps
    if a.prog == "dyck" and not os.environ.get("W4_S2_DYCK"):
        # amendment 3: dyck stage 2 dropped for cost (3 h cap already exceeded; stage-1 dyck took 53 min/seed)
        print("skip dyck stage 2 (amendment 3)"); raise SystemExit(0)
    main(a.prog, a.seed)
