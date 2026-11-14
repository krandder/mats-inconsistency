"""M2 worker 4: tracr ground truth for unit-importance predictors (see PREREG.md).

Run: PYTHONPATH=/tmp/traders-net/M2/w4/pylib python3 tracr_gt.py --prog reverse --seed 1
Writes results/units_<prog>_s<seed>.csv and results/pairs_<prog>_s<seed>.csv (+ meta json).
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import argparse, itertools, json, pickle, time, sys
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/traders-net/M2/w4/cache"
JUNK = 4

PROGS = {  # name: (vocab, length, distinct_only)
    "reverse": (["a", "b", "c", "d"], 6, False),
    "sort_unique": ([1, 2, 3, 4, 5, 6], 6, True),
    "hist": (["a", "b", "c", "d"], 6, False),
    "dyck": (["(", ")"], 8, False),
}


def make_prog(name):
    from tracr.rasp import rasp
    from tracr.compiler import lib
    return {"reverse": lambda: lib.make_reverse(rasp.tokens),
            "sort_unique": lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens),
            "hist": lambda: lib.make_hist(),
            "dyck": lambda: lib.make_shuffle_dyck(pairs=["()"])}[name]()


SUBSET = {"reverse": 1024}  # amendment 2: fixed random input subset (rng 0) for the costly program


def compile_cached(name):
    obj = _compile_cached(name)
    if name in SUBSET:
        n = len(obj["toks"])
        idx = np.sort(np.random.default_rng(0).choice(n, SUBSET[name], replace=False))
        pos = {int(j): i for i, j in enumerate(idx)}
        keep = [r for r, j in enumerate(obj["ref_idx"]) if int(j) in pos]
        obj = dict(obj, toks=obj["toks"][idx], labels=obj["labels"][idx],
                   ref_idx=np.array([pos[int(obj["ref_idx"][r])] for r in keep], dtype=int),
                   ref=obj["ref"][keep], subset=idx)
    return obj


def _compile_cached(name):
    """Compile once; return numpy params, inputs, labels, readout dims."""
    os.makedirs(CACHE, exist_ok=True)
    fn = f"{CACHE}/{name}.pkl"
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    from tracr.rasp import rasp
    from tracr.compiler import compiling
    vocab, L, distinct = PROGS[name]
    prog = make_prog(name)
    m = compiling.compile_rasp_to_model(prog, vocab=set(vocab), max_seq_len=L,
                                        compiler_bos="BOS", mlp_exactness=100)
    cfg = m.model_config
    H, k = cfg.num_heads, cfg.key_size
    p = {kk: {a: np.asarray(b) for a, b in v.items()} for kk, v in m.params.items()}
    D = p["token_embed"]["embeddings"].shape[1]
    layers = []
    for l in range(cfg.num_layers):
        a = lambda s: p[f"transformer/layer_{l}/attn/{s}"]
        ml = lambda s: p[f"transformer/layer_{l}/mlp/{s}"]
        layers.append(dict(
            Wq=a("query")["w"].reshape(D, H, k).transpose(1, 0, 2), bq=a("query")["b"].reshape(H, k),
            Wk=a("key")["w"].reshape(D, H, k).transpose(1, 0, 2), bk=a("key")["b"].reshape(H, k),
            Wv=a("value")["w"].reshape(D, H, k).transpose(1, 0, 2), bv=a("value")["b"].reshape(H, k),
            Wo=a("linear")["w"].reshape(H, k, D), bo=a("linear")["b"],
            W1=ml("linear_1")["w"], b1=ml("linear_1")["b"], W2=ml("linear_2")["w"], b2=ml("linear_2")["b"]))
    seqs = list(itertools.permutations(vocab, L)) if distinct else list(itertools.product(vocab, repeat=L))
    enc = m.input_encoder
    toks = np.array([enc.encode(["BOS"] + list(s)) for s in seqs], dtype=np.int32)
    labs_raw = [rasp.evaluate(prog, list(s)) for s in seqs]
    emap = m.output_encoder.encoding_map
    classes = sorted(emap, key=lambda v: emap[v])
    labels = np.array([[emap[v] for v in row] for row in labs_raw], dtype=np.int32)
    lab = prog.label
    res_labels = m.residual_labels
    out_dims = np.array([res_labels.index(f"{lab}:{c}") for c in classes])
    # reference decode from tracr for a correctness check of our forward
    ref_idx = np.random.default_rng(0).choice(len(seqs), size=min(64, len(seqs)), replace=False)
    ref = np.array([[emap[v] for v in m.apply(["BOS"] + list(seqs[i])).decoded[1:]] for i in ref_idx])
    obj = dict(layers=layers, tok=p["token_embed"]["embeddings"], pos=p["pos_embed"]["embeddings"],
               toks=toks, labels=labels, out_dims=out_dims, H=H, k=k, D=D,
               res_labels=res_labels, classes=[str(c) for c in classes], ref_idx=ref_idx, ref=ref)
    pickle.dump(obj, open(fn, "wb"))
    return obj


# ---------------------------------------------------------------- model with unit gates
def pad_junk(obj):
    """Append JUNK residual dims that nothing reads."""
    D = obj["D"]
    z = lambda shape: np.zeros(shape, dtype=np.float32)
    L = []
    for ly in obj["layers"]:
        n = {}
        for s in ("Wq", "Wk", "Wv"):
            n[s] = np.concatenate([ly[s], z((ly[s].shape[0], JUNK, ly[s].shape[2]))], 1).astype(np.float32)
        for s in ("bq", "bk", "bv"):
            n[s] = ly[s].astype(np.float32)
        n["Wo"] = np.concatenate([ly["Wo"], z(ly["Wo"].shape[:2] + (JUNK,))], 2).astype(np.float32)
        n["bo"] = np.concatenate([ly["bo"], z(JUNK)]).astype(np.float32)
        n["W1"] = np.concatenate([ly["W1"], z((JUNK, ly["W1"].shape[1]))], 0).astype(np.float32)
        n["b1"] = ly["b1"].astype(np.float32)
        n["W2"] = np.concatenate([ly["W2"], z((ly["W2"].shape[0], JUNK))], 1).astype(np.float32)
        n["b2"] = np.concatenate([ly["b2"], z(JUNK)]).astype(np.float32)
        L.append(n)
    x0 = obj["tok"][obj["toks"]] + obj["pos"][np.arange(obj["toks"].shape[1])][None]
    x0 = np.concatenate([x0, np.zeros(x0.shape[:2] + (JUNK,))], -1).astype(np.float32)
    return L, x0


def unit_slices(L):
    """Flat gate vector layout: per layer [heads..., neurons...]."""
    sl, o = [], 0
    for ly in L:
        H, h = ly["Wq"].shape[0], ly["W1"].shape[1]
        sl.append(((o, o + H), (o + H, o + H + h)))
        o += H + h
    return sl, o


def make_fns(L, x0, labels, out_dims, beta):
    sl, U = unit_slices(L)
    P = jax.tree_util.tree_map(jnp.asarray, L)
    X0, Y, OD = jnp.asarray(x0), jnp.asarray(labels), jnp.asarray(out_dims)
    k = L[0]["Wq"].shape[2]

    C0 = (P, X0, Y)

    def forward(C, G, M, dh=None, dn=None, collect=False):
        """G: (B or 1, U) gates; M: (U, D) mean write (head) / (U,) padded mean act in M[:,0] for neurons.
        Head gate acts on its residual write; neuron gate acts on its activation."""
        P, X0, _ = C
        x = X0
        writes = []
        for l, ly in enumerate(P):
            (h0, h1), (n0, n1) = sl[l]
            q = jnp.einsum("btd,hdk->bthk", x, ly["Wq"]) + ly["bq"]
            kk = jnp.einsum("btd,hdk->bthk", x, ly["Wk"]) + ly["bk"]
            v = jnp.einsum("btd,hdk->bthk", x, ly["Wv"]) + ly["bv"]
            att = jax.nn.softmax(jnp.einsum("bthk,bshk->bhts", q, kk) / np.sqrt(k), -1)
            z = jnp.einsum("bhts,bshk->bthk", att, v)
            if dh is not None:
                z = z + dh[l]
            w = jnp.einsum("bthk,hkd->bthd", z, ly["Wo"])  # per-head write
            g = G[:, h0:h1][:, None, :, None]
            w = g * w + (1 - g) * M[h0:h1][None, None]
            x = x + w.sum(2) + ly["bo"]
            a = jax.nn.relu(x @ ly["W1"] + ly["b1"])
            if dn is not None:
                a = a + dn[l]
            g = G[:, n0:n1][:, None, :]
            a = g * a + (1 - g) * M[n0:n1, 0][None, None]
            x = x + a @ ly["W2"] + ly["b2"]
            if collect:
                writes.append((w, a))
        return x, writes

    def ce(C, x, per_example=False):
        Y = C[2]
        logits = beta * x[:, 1:, :][..., OD]
        lp = jax.nn.log_softmax(logits, -1)
        nll = -jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]
        return nll.mean(1) if per_example else nll.mean()

    _loss = jax.jit(lambda C, G, M: ce(C, forward(C, G, M)[0]))
    _loss_pe = lambda C, G, M: ce(C, forward(C, G, M)[0], True).sum()
    _gG = jax.jit(jax.grad(lambda G, C, M: ce(C, forward(C, G, M)[0])))
    _gGpe = jax.jit(jax.grad(lambda G, C, M: _loss_pe(C, G, M)))
    _gd = jax.jit(jax.grad(lambda dh, dn, C, M: ce(C, forward(C, jnp.ones((1, U)), M, dh, dn)[0]), argnums=(0, 1)))
    _full = jax.jit(lambda C, M: forward(C, jnp.ones((1, U)), M, collect=True))
    _corr = jax.jit(lambda C, G, M: (jnp.argmax(forward(C, G, M)[0][:, 1:, :][..., OD], -1) == C[2]).mean())
    loss = lambda G, M: _loss(C0, G, M)
    grad_G = lambda G, M: _gG(G, C0, M)
    grad_G_pe = lambda G, M: _gGpe(G, C0, M)
    grad_delta = lambda dh, dn, M: _gd(dh, dn, C0, M)
    full = lambda M: _full(C0, M)
    correct = lambda G, M: _corr(C0, G, M)

    return dict(forward=forward, loss=loss, grad_G=grad_G, grad_G_pe=grad_G_pe, grad_delta=grad_delta,
                full=full, correct=correct, sl=sl, U=U, ce=ce, P=P, C0=C0)


# ---------------------------------------------------------------- augmentation
def augment(obj, seed, noise_frac=0.15):
    L, x0 = pad_junk(obj)
    D0 = obj["D"]
    rng = np.random.default_rng(seed)
    # classify compiled units with one forward (beta irrelevant)
    fns = make_fns(L, x0, obj["labels"], obj["out_dims"], 5.0)
    sl, U = fns["sl"], fns["U"]
    x, writes = fns["full"](jnp.zeros((U, x0.shape[-1])))
    units = []  # dicts: gran, layer, cls, pair
    for l, ly in enumerate(L):
        w, a = writes[l]
        w, a = np.asarray(w), np.asarray(a)
        if l == len(L) - 1:  # last layer: the BOS position cannot reach the loss
            w, a = w[:, 1:], a[:, 1:]
        for hh in range(ly["Wq"].shape[0]):
            ov = np.linalg.norm(np.einsum("dk,ke->de", ly["Wv"][hh], ly["Wo"][hh]))
            live = ov > 0 and np.abs(w[:, :, hh]).max() > 1e-6
            units.append(dict(gran="head", layer=l, cls="C" if live else "Z", pair=-1))
        for j in range(ly["W1"].shape[1]):
            live = np.linalg.norm(ly["W2"][j]) > 0 and a[:, :, j].max() > 1e-6
            units.append(dict(gran="neuron", layer=l, cls="C" if live else "Z", pair=-1))
    # rebuild per-layer with splits and additions
    pair_id = 0
    newL, newunits = [], []
    ptr = 0
    for l, ly in enumerate(L):
        H, h = ly["Wq"].shape[0], ly["W1"].shape[1]
        hu, nu = units[ptr:ptr + H], units[ptr + H:ptr + H + h]
        ptr += H + h
        ly = {kk: vv.copy() for kk, vv in ly.items()}
        hl = [dict(Wq=ly["Wq"][i], bq=ly["bq"][i], Wk=ly["Wk"][i], bk=ly["bk"][i], Wv=ly["Wv"][i],
                   bv=ly["bv"][i], Wo=ly["Wo"][i], **hu[i]) for i in range(H)]
        nl = [dict(W1=ly["W1"][:, j], b1=ly["b1"][j], W2=ly["W2"][j], **nu[j]) for j in range(h)]
        live_h = [i for i, u in enumerate(hl) if u["cls"] == "C"]
        live_n = [j for j, u in enumerate(nl) if u["cls"] == "C"]
        hnorm = np.mean([np.linalg.norm(hl[i]["Wv"] @ hl[i]["Wo"]) for i in live_h]) if live_h else None
        nnorm = np.mean([np.linalg.norm(nl[j]["W2"]) for j in live_n]) if live_n else None
        # R splits
        for i in (rng.choice(live_h, max(1, len(live_h) // 2), replace=False) if live_h else []):
            hl[i]["Wo"] = hl[i]["Wo"] / 2
            hl[i]["cls"], hl[i]["pair"] = "R", pair_id
            hl.append(dict(hl[i]))
            pair_id += 1
        for j in (rng.choice(live_n, max(1, len(live_n) // 4), replace=False) if live_n else []):
            nl[j]["W2"] = nl[j]["W2"] / 2
            nl[j]["cls"], nl[j]["pair"] = "R", pair_id
            nl.append(dict(nl[j]))
            pair_id += 1
        newL.append(dict(hl=hl, nl=nl, bo=ly["bo"], b2=ly["b2"], hnorm=hnorm, nnorm=nnorm))
    # global fallbacks for norms
    gh = np.mean([x["hnorm"] for x in newL if x["hnorm"] is not None])
    gn = np.mean([x["nnorm"] for x in newL if x["nnorm"] is not None])
    Dt = D0 + JUNK
    k = obj["k"]
    for l, x in enumerate(newL):
        hn = x["hnorm"] or gh
        nn_ = x["nnorm"] or gn
        nd = max(4, int(0.05 * sum(1 for u in x["nl"] if u["cls"] != "R" or True)))
        for kind in ("D", "N"):
            # heads: random QK/V on real dims
            Wq = np.zeros((Dt, k), np.float32); Wq[:D0] = rng.normal(0, 1, (D0, k))
            Wk = np.zeros((Dt, k), np.float32); Wk[:D0] = rng.normal(0, 1, (D0, k))
            Wv = np.zeros((Dt, k), np.float32); Wv[:D0] = rng.normal(0, 1, (D0, k))
            Wo = np.zeros((k, Dt), np.float32)
            if kind == "D":
                Wo[:, D0:] = rng.normal(0, 1, (k, JUNK))
                scale = hn
            else:
                Wo[:, :D0] = rng.normal(0, 1, (k, D0))
                scale = noise_frac * hn
            Wo *= scale / np.linalg.norm(Wv @ Wo)
            x["hl"].append(dict(Wq=Wq, bq=np.zeros(k, np.float32), Wk=Wk, bk=np.zeros(k, np.float32), Wv=Wv,
                                bv=np.zeros(k, np.float32), Wo=Wo, gran="head", layer=l, cls=kind, pair=-1))
            for _ in range(nd):
                W1 = np.zeros(Dt, np.float32); W1[:D0] = rng.normal(0, 1, D0)
                W2 = np.zeros(Dt, np.float32)
                if kind == "D":
                    W2[D0:] = rng.normal(0, 1, JUNK); W2 *= nn_ / np.linalg.norm(W2)
                else:
                    W2[:D0] = rng.normal(0, 1, D0); W2 *= noise_frac * nn_ / np.linalg.norm(W2)
                x["nl"].append(dict(W1=W1, b1=np.float32(0), W2=W2, gran="neuron", layer=l, cls=kind, pair=-1))
    # pack
    P, meta = [], []
    for x in newL:
        hl, nl = x["hl"], x["nl"]
        P.append(dict(Wq=np.stack([u["Wq"] for u in hl]).astype(np.float32), bq=np.stack([u["bq"] for u in hl]).astype(np.float32),
                      Wk=np.stack([u["Wk"] for u in hl]).astype(np.float32), bk=np.stack([u["bk"] for u in hl]).astype(np.float32),
                      Wv=np.stack([u["Wv"] for u in hl]).astype(np.float32), bv=np.stack([u["bv"] for u in hl]).astype(np.float32),
                      Wo=np.stack([u["Wo"] for u in hl]).astype(np.float32), bo=x["bo"],
                      W1=np.stack([u["W1"] for u in nl], 1).astype(np.float32), b1=np.array([u["b1"] for u in nl], np.float32),
                      W2=np.stack([u["W2"] for u in nl]).astype(np.float32), b2=x["b2"]))
        meta += [dict(gran=u["gran"], layer=u["layer"], cls=u["cls"], pair=u["pair"],
                      wnorm=float(np.linalg.norm(u["Wv"] @ u["Wo"])) if u["gran"] == "head" else float(np.linalg.norm(u["W2"])))
                 for u in hl + nl]
    return P, x0, meta


# ---------------------------------------------------------------- predictors
def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def run(prog, seed, betas=(5.0, 20.0), nperm=64, noise_frac=0.15):
    t0 = time.time()
    obj = compile_cached(prog)
    # check our forward reproduces tracr on the reference inputs
    L0, x00 = pad_junk(obj)
    f0 = make_fns(L0, x00, obj["labels"], obj["out_dims"], 5.0)
    xx, _ = f0["full"](jnp.zeros((f0["U"], x00.shape[-1])))
    pred = np.asarray(jnp.argmax(xx[:, 1:, :][..., obj["out_dims"]], -1))
    ref_ok = float((pred[obj["ref_idx"]] == obj["ref"]).mean()) if len(obj["ref_idx"]) else float("nan")
    acc_compiled = float((pred == obj["labels"]).mean())
    # augment; shrink noise until accuracy stays 100%
    while True:
        P, x0, meta = augment(obj, seed, noise_frac)
        fns = make_fns(P, x0, obj["labels"], obj["out_dims"], 5.0)
        acc = float(fns["correct"](jnp.ones((1, fns["U"])), jnp.zeros((fns["U"], x0.shape[-1]))))
        if acc >= 1.0 or noise_frac < 1e-3:
            break
        noise_frac /= 2
    U = fns["U"]
    rng = np.random.default_rng(1000 + seed)
    rows, pairs_out, info = [], [], dict(prog=prog, seed=seed, ref_match=ref_ok, acc_compiled=acc_compiled,
                                         acc_aug=acc, noise_frac=noise_frac, U=U, n_inputs=int(len(obj["toks"])))
    for beta in betas:
        fns = make_fns(P, x0, obj["labels"], obj["out_dims"], beta)
        sl, Dt = fns["sl"], x0.shape[-1]
        ones = jnp.ones((1, U))
        Z0 = jnp.zeros((U, Dt))
        # means for mean ablation
        _, writes = fns["full"](Z0)
        Mm = np.zeros((U, Dt), np.float32)
        for l, (w, a) in enumerate(writes):
            (h0, h1), (n0, n1) = sl[l]
            Mm[h0:h1] = np.asarray(w).mean((0, 1))
            Mm[n0:n1, 0] = np.asarray(a).mean((0, 1))
        Mm = jnp.asarray(Mm)
        base = float(fns["loss"](ones, Z0))
        base_m = float(fns["loss"](ones, Mm))

        def abl(offs, M=Z0):
            G = np.ones((1, U), np.float32); G[0, list(offs)] = 0
            return float(fns["loss"](jnp.asarray(G), M)) - base

        oz = np.array([abl([i]) for i in range(U)])
        om = np.array([abl([i], Mm) for i in range(U)])
        taylor = -np.asarray(fns["grad_G"](ones, Z0))[0]
        B = x0.shape[0]
        ge = np.asarray(fns["grad_G_pe"](jnp.ones((B, U)), Z0))  # (B,U), sum-loss grads
        abs_ag = np.abs(ge).sum(0) / B
        dh = [jnp.zeros((B, x0.shape[1]) + ly["Wq"].shape[0:1] + (ly["Wq"].shape[2],)) for ly in P]
        dn = [jnp.zeros((B, x0.shape[1], ly["W1"].shape[1])) for ly in P]
        gdh, gdn = fns["grad_delta"](dh, dn, Z0)
        gnorm = np.zeros(U)
        for l in range(len(P)):
            (h0, h1), (n0, n1) = sl[l]
            gnorm[h0:h1] = np.sqrt((np.asarray(gdh[l]) ** 2).sum((0, 1, 3)))
            gnorm[n0:n1] = np.sqrt((np.asarray(gdn[l]) ** 2).sum((0, 1)))
        mag = np.array([m["wnorm"] for m in meta])
        # conductance: per layer x granularity group, 16-step midpoint IG on gates
        cond = np.zeros(U); cond_resid = []; cond256 = np.zeros(U); cond256_resid = []
        groups = []
        for l in range(len(P)):
            (h0, h1), (n0, n1) = sl[l]
            groups += [list(range(h0, h1)), list(range(n0, n1))]
        cheap = beta != betas[0]  # amendment 2: secondary beta gets no conductance/Shapley
        if cheap:
            cond[:] = np.nan; cond256[:] = np.nan; cond_resid.append(np.nan); cond256_resid.append(np.nan)
        for grp in ([] if cheap else groups):
            acc_g = np.zeros(len(grp))
            for s in range(16):
                al = (s + 0.5) / 16
                G = np.ones((1, U), np.float32); G[0, grp] = al
                acc_g += np.asarray(fns["grad_G"](jnp.asarray(G), Z0))[0, grp]
            cond[grp] = -acc_g / 16
            cond_resid.append(abs(cond[grp].sum() - abl(grp)))
            acc_g = np.zeros(len(grp))
            for s in range(256):  # amendment 1: fine quadrature (tracr MLPs have slope ~100)
                G = np.ones((1, U), np.float32); G[0, grp] = (s + 0.5) / 256
                acc_g += np.asarray(fns["grad_G"](jnp.asarray(G), Z0))[0, grp]
            cond256[grp] = -acc_g / 256
            cond256_resid.append(abs(cond256[grp].sum() - abl(grp)))
        # direct-path LOO settlement at the output (committed-trader rule applied to every unit)
        xfin, writes = fns["full"](Z0)
        loo = np.zeros(U)
        OD = obj["out_dims"]
        Yj = jnp.asarray(obj["labels"])

        @jax.jit
        def loo_layer(xf, w_out):  # w_out (B,T,u,C) unit writes on readout dims
            logits = beta * (xf[:, 1:, None, :][..., OD] - w_out[:, 1:])
            lp = jax.nn.log_softmax(logits, -1)
            nll = -jnp.take_along_axis(lp, Yj[:, :, None, None].repeat(w_out.shape[2], 2), -1)[..., 0]
            return nll.mean((0, 1))

        for l, (w, a) in enumerate(writes):
            (h0, h1), (n0, n1) = sl[l]
            loo[h0:h1] = np.asarray(loo_layer(xfin, w[..., OD])) - base
            wa = a[..., None] * jnp.asarray(P[l]["W2"][:, OD])[None, None]
            loo[n0:n1] = np.asarray(loo_layer(xfin, wa)) - base
        # sampled Shapley of the zero-ablation game; players: heads of the whole net, neurons per layer
        shap = np.zeros(U)
        hgroup = [i for l in range(len(P)) for i in range(*sl[l][0])]
        sgroups = [hgroup] + [list(range(*sl[l][1])) for l in range(len(P))]
        zero_w = np.array([m["wnorm"] == 0 for m in meta])
        if cheap:
            shap[:] = np.nan
        for grp in ([] if cheap else sgroups):
            grp = [i for i in grp if not zero_w[i]]  # zero-write units: exactly 0 in every coalition
            if not grp:
                continue
            accs = np.zeros(len(grp))
            pos = {u: n for n, u in enumerate(grp)}
            for _ in range(nperm):
                G = np.ones((1, U), np.float32)
                prev = base
                for u in rng.permutation(grp):
                    G[0, u] = 0
                    cur = float(fns["loss"](jnp.asarray(G), Z0))
                    accs[pos[u]] += cur - prev
                    prev = cur
            shap[grp] = accs / nperm
        rnd = rng.random(U)
        for i, m in enumerate(meta):
            rows.append(dict(prog=prog, seed=seed, beta=beta, uid=i, **{kk: m[kk] for kk in ("gran", "layer", "cls", "pair")},
                             oracle_zero=oz[i], oracle_mean=om[i], taylor=taylor[i], abs_ag=abs_ag[i], gradnorm=gnorm[i],
                             magnitude=mag[i], conductance=cond[i], conductance256=cond256[i], shapley=shap[i], loo_direct=loo[i], random=rnd[i]))
        pids = sorted({m["pair"] for m in meta if m["pair"] >= 0})
        for pid in pids:
            mem = [i for i, m in enumerate(meta) if m["pair"] == pid]
            pairs_out.append(dict(prog=prog, seed=seed, beta=beta, pair=pid, gran=meta[mem[0]]["gran"],
                                  layer=meta[mem[0]]["layer"], u1=mem[0], u2=mem[1], d1=oz[mem[0]], d2=oz[mem[1]],
                                  dpair=abl(mem), dpair_mean=abl(mem, Mm), d1_mean=om[mem[0]], d2_mean=om[mem[1]]))
        info[f"beta{beta:g}"] = dict(base_loss=base, base_loss_meanM=base_m, cond_complete_resid_max=float(np.max(cond_resid)), cond256_complete_resid_max=float(np.max(cond256_resid)),
                                     taylor_vs_oz_spearman=spearman(taylor, oz))
        print(f"{prog} s{seed} beta={beta} U={U} base={base:.4g} cond_resid={np.max(cond_resid):.2e} t={time.time()-t0:.0f}s", flush=True)
    import csv
    os.makedirs(f"{HERE}/results", exist_ok=True)
    for fn, data in ((f"units_{prog}_s{seed}.csv", rows), (f"pairs_{prog}_s{seed}.csv", pairs_out)):
        with open(f"{HERE}/results/{fn}", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            wr.writeheader(); wr.writerows(data)
    info["wall_s"] = time.time() - t0
    json.dump(info, open(f"{HERE}/results/meta_{prog}_s{seed}.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nperm", type=int, default=64)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    if a.compile_only:
        o = compile_cached(a.prog); print(a.prog, o["toks"].shape, o["labels"].shape, o["classes"])
    else:
        run(a.prog, a.seed, nperm=a.nperm)
