"""Averaging vs structure, and the nature of the noise, for Qwen3-32B one-form-per-call answers (CPU only).

Input: <run>/log.jsonl from scripts/run_decomposition_sampling.py (samples t0, s0..s4,
w1..w3 per query), the vLLM Qwen3-32B v6_direct predictions, and <run>/subset.json.

Arms (all scored with clip 0.01, missing imputed 0.5, like scripts/score_verbalized_baseline):
  A0        temperature 0, one call per form (the baseline protocol on DashScope)
  A         one temperature-0.7 sample per form (averaged over the 5 choices of sample)
  B(k)      mean of k samples per form, in probability (B_prob) or log-odds (B_logit);
            every k-subset of the 5 samples, averaged
  C_<fit>   one sample per form, projected onto the nearest coherent 4-atom distribution
            (scripts/coherence_projection.fit: ls_prob, ls_logit, kl)
  D(k)      mean of k samples, then projected
  E         control: the same projection with the wrong logical structure.  Each stated value is
            moved to another form's slot (a random derangement, 5 per pair, seed fixed), the
            atoms are fitted to the moved values and each form reads back the implied value of
            the slot its value went to.  E_group permutes only within literals / ANDs / ORs
            (keeps each value's operator level, breaks which forms are logically tied);
            E_full permutes across all 12.  Same number of numbers pooled, same fit, same
            degrees of freedom; only the true relations are missing.  Permutations that are
            only a relabeling of the atoms (e.g. swap P<->notP and Q<->notQ everywhere) keep
            the true structure and are excluded.  E_shrink: shrink every
            log-odds toward 0 by one global factor chosen so the mean |change in p| equals C_kl's
            (label-free, structure-free shrinkage of the same size).
  F         rule ablations: F_lit (only P+notP=1, Q+notQ=1), F_comp (those two plus the four
            De Morgan complements AND/OR), full = C.  Closed forms: for kl and ls_prob the
            complement pair (x, y) becomes (m, 1-m) with m = (x + 1 - y) / 2; for ls_logit
            logit m = (logit x - logit y) / 2.
  W         wording average: mean over 4 wordings (original s0 + paraphrases w1..w3, one sample
            each), vs B(4) at the same cost; W projected vs D(4).

Noise decomposition (probability space, pairs with all 5 samples of all 12 forms, affine
coherent subspace = {A q : sum q = 1}; the truth vector lies in it):
  single-sample Brier = shared (coherent) error d + incoherence c + sampling noise a
  a = mean per-form variance across the 5 samples; c = |(I-Pi) mu|^2 / 12 and
  d = |Pi mu - y|^2 / 12 for the expected answer mu, both de-biased for the noise left in
  the 5-sample mean; b = variance across the 4 wordings minus a.  Averaging k samples
  removes (1 - 1/k) of a; projection removes c and the part of a outside the subspace.
Log-odds space: the same a, b and the ls_logit residual variance c (no truth in log-odds).
Correlations: 12x12 across pairs of (i) residuals mu - Pi mu (with the null pattern of
(I-Pi) for independent equal-variance form errors), (ii) errors mu - y, (iii) same-index
sampling noise (independent calls, so this should be ~0).

Paired bootstrap over pairs (2,000 resamples, seed 20260924), ratio of sums.

Usage: analyze_decomposition.py --run DIR --pairs subset|all --out-json docs/X.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coherence_projection import FORM_ATOMS, fit  # noqa: E402
from training.supervised_pairs import FORM_ORDER, relation_penalties  # noqa: E402

FORMS = list(FORM_ORDER)
A_MAT = np.array([FORM_ATOMS[f] for f in FORMS], dtype=float)
LO = 0.01
FITS = ("ls_prob", "ls_logit", "kl")
SAMPLES = ["s0", "s1", "s2", "s3", "s4"]
WORDS = ["s0", "w1", "w2", "w3"]
ANDOR = np.array(["AND" in f or "OR" in f for f in FORMS])
COMP_LIT = [("P", "NOT P"), ("Q", "NOT Q")]
COMP_DM = [("(P) AND (Q)", "(NOT P) OR (NOT Q)"), ("(P) AND (NOT Q)", "(NOT P) OR (Q)"),
           ("(NOT P) AND (Q)", "(P) OR (NOT Q)"), ("(NOT P) AND (NOT Q)", "(P) OR (Q)")]
GROUPS = [[i for i, f in enumerate(FORMS) if "AND" not in f and "OR" not in f],
          [i for i, f in enumerate(FORMS) if "AND" in f], [i for i, f in enumerate(FORMS) if "OR" in f]]
N_PERM = 5
VBASE = "/home/agent/data/qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl"
BOOT_SEED = 20260924


def lg(p):
    p = np.clip(p, LO, 1 - LO); return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-x))


def pmetrics(p: np.ndarray, y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """[LL all, n, LL obs, n obs, Brier all, Brier obs, LL and/or, n and/or, msv, |atoms sum-1|, |lit comp|]"""
    q = np.where(np.isnan(p), 0.5, p); q = np.clip(q, LO, 1 - LO)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q)); br = (q - y) ** 2
    d = dict(zip(FORMS, q.tolist()))
    msv = relation_penalties(d)["total"]
    atoms = abs(sum(d[f] for f in FORMS if "AND" in f) - 1)
    lit = (abs(d["P"] + d["NOT P"] - 1) + abs(d["Q"] + d["NOT Q"] - 1)) / 2
    return np.array([ll.sum(), 12, ll[obs].sum(), obs.sum(), br.sum(), br[obs].sum(), ll[ANDOR].sum(), ANDOR.sum(),
                     msv, atoms, lit])


def project(p: np.ndarray, method: str) -> np.ndarray:
    have = ~np.isnan(p)
    if not have.any():
        return p
    q = fit([f for f, h in zip(FORMS, have) if h], p[have], method)
    return A_MAT @ q


def project_perm(p: np.ndarray, perm: np.ndarray, method: str) -> np.ndarray:
    """Value of form i goes to slot perm[i]; fit on slots; form i reads slot perm[i]."""
    slot = np.full(12, np.nan); slot[perm] = p
    return project(slot, method)[perm]


def complements(p: np.ndarray, pairs, method: str) -> np.ndarray:
    out = p.copy(); idx = {f: i for i, f in enumerate(FORMS)}
    for a, b in pairs:
        i, j = idx[a], idx[b]
        x, y = out[i], out[j]
        if np.isnan(x) and np.isnan(y):
            continue
        if np.isnan(x):
            x = 1 - y
        if np.isnan(y):
            y = 1 - x
        if method == "ls_logit":
            m = float(sg((lg(x) - lg(y)) / 2))
        else:
            m = (min(1 - LO, max(LO, x)) + 1 - min(1 - LO, max(LO, y))) / 2
        out[i], out[j] = m, 1 - m
    return out


def mean_p(ps, mode):
    ps = np.array(ps)
    with np.errstate(all="ignore"):
        if mode == "prob":
            return np.nanmean(np.clip(ps, LO, 1 - LO), 0)
        return sg(np.nanmean(lg(ps), 0))


def derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


ATOM_PERMS = [np.array(s) for s in itertools.permutations(range(4))]


def is_symmetry(perm: np.ndarray) -> bool:
    """True if moving form i's value to slot perm[i] is just a relabeling of the atoms (then it is the true structure)."""
    return any(np.array_equal(A_MAT[perm], A_MAT[:, s]) for s in ATOM_PERMS)


def perms_for(pair_id: str):
    rng = np.random.default_rng([BOOT_SEED, int(hashlib.sha256(pair_id.encode()).hexdigest()[:12], 16)])
    grp, full = [], []
    for _ in range(N_PERM):
        g = np.arange(12)
        while True:
            for G in GROUPS:
                g[G] = np.array(G)[derangement(rng, len(G))]
            if not is_symmetry(g):
                break
        f = derangement(rng, 12)
        while is_symmetry(f):
            f = derangement(rng, 12)
        grp.append(g); full.append(f)
    return grp, full


def pair_arms(task):
    pid, S, y, obs, has_words = task
    out = defaultdict(list)
    add = lambda name, p: out[name].append(pmetrics(p, y, obs))
    add("A0", S["t0"])
    for m in FITS:
        add(f"A0_proj_{m}", project(S["t0"], m))
    if "vllm" in S:
        add("vllm_baseline", S["vllm"])
    singles = [S[s] for s in SAMPLES]
    grp, full = perms_for(pid)
    for p in singles:
        add("A", p)
        for m in FITS:
            add(f"C_{m}", project(p, m))
            add(f"F_lit_{m}", complements(p, COMP_LIT, m))
            add(f"F_comp_{m}", complements(p, COMP_LIT + COMP_DM, m))
        for g, f in zip(grp, full):
            add("E_group_kl", project_perm(p, g, "kl"))
            add("E_full_kl", project_perm(p, f, "kl"))
    for k in range(1, 6):
        for sub in itertools.combinations(range(5), k):
            for mode in ("prob", "logit"):
                mp = mean_p([singles[i] for i in sub], mode)
                add(f"B_{mode}_k{k}", mp)
                add(f"D_kl_{mode}_k{k}", project(mp, "kl"))
                if k == 5:
                    for m in ("ls_prob", "ls_logit"):
                        add(f"D_{m}_{mode}_k5", project(mp, m))
                    for g, f in zip(grp, full):
                        add(f"E_group_kl_on_B_{mode}_k5", project_perm(mp, g, "kl"))
                    add(f"F_comp_kl_on_B_{mode}_k5", complements(mp, COMP_LIT + COMP_DM, "kl"))
    if has_words:
        for mode in ("prob", "logit"):
            wm = mean_p([S[w] for w in WORDS], mode)
            add(f"W_{mode}_4wordings", wm)
            add(f"W_{mode}_4wordings_proj_kl", project(wm, "kl"))
        for w in ("w1", "w2", "w3"):
            add(f"A_wording_{w}", S[w])
    # movement of C_kl (for the magnitude-matched shrinkage control)
    mov = [np.nanmean(np.abs(project(p, "kl") - np.clip(p, LO, 1 - LO))) for p in singles]
    return pid, {k: np.mean(v, 0) for k, v in out.items()}, float(np.mean(mov))


def summary(pp: dict, pairs) -> dict:
    X = np.array([pp[p] for p in pairs]); s = X.sum(0)
    return {"all12_log_loss": s[0] / s[1], "observed_log_loss": s[2] / s[3], "andor_log_loss": s[6] / s[7],
            "all12_brier": s[4] / s[1], "observed_brier": s[5] / s[3],
            "coherence_msv": X[:, 8].mean(), "atoms_sum_abs_violation": X[:, 9].mean(),
            "literal_complement_abs_violation": X[:, 10].mean()}


def boot(x: dict, yy: dict, pairs, n=2000) -> dict:
    X = np.array([x[p] for p in pairs]); Y = np.array([yy[p] for p in pairs])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(pairs), size=(n, len(pairs)))
    res = {}
    for name, (num, den) in {"all12_log_loss": (0, 1), "observed_log_loss": (2, 3), "andor_log_loss": (6, 7),
                             "all12_brier": (4, 1)}.items():
        point = X[:, num].sum() / X[:, den].sum() - Y[:, num].sum() / Y[:, den].sum()
        d = X[idx, num].sum(1) / X[idx, den].sum(1) - Y[idx, num].sum(1) / Y[idx, den].sum(1)
        res[name] = {"diff": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


# ---------------- noise decomposition ----------------

def affine_proj():
    V = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float).T  # directions with sum 0
    M = A_MAT @ V
    Pi = M @ np.linalg.solve(M.T @ M, M.T)
    p0 = A_MAT @ np.full(4, 0.25)
    return Pi, p0


def decompose(S_all: dict, Y: dict, pairs, words: bool) -> dict:
    Pi, p0 = affine_proj(); R = np.eye(12) - Pi
    proj = lambda v: p0 + Pi @ (v - p0)
    keep = [p for p in pairs if all(not np.isnan(S_all[p][s]).any() for s in SAMPLES + (["w1", "w2", "w3"] if words else []))]
    out = {"pairs_used": len(keep)}
    Xs = np.array([[np.clip(S_all[p][s], LO, 1 - LO) for s in SAMPLES] for p in keep])  # n,5,12
    y = np.array([Y[p] for p in keep])
    mu = Xs.mean(1); var = Xs.var(1, ddof=1)  # n,12
    a = var.mean()
    raw_err = ((mu - y) ** 2).mean() - a / 5
    c = (np.einsum("ij,nj->ni", R, mu - p0) ** 2).sum(1).mean() / 12 - (np.diag(R) * var).sum(1).mean() / 5 / 12
    d = raw_err - c
    a_in = (np.diag(Pi) * var).sum(1).mean() / 12
    single = ((Xs - y[:, None, :]) ** 2).mean()
    prob = {"a_sampling": a, "c_incoherence": c, "d_shared_coherent_error": d,
            "sampling_inside_coherent_subspace": a_in, "sampling_outside_subspace": a - a_in,
            "check_single_sample_brier_measured": single, "check_single_sample_brier_a_plus_c_plus_d": a + c + d,
            "predicted": {"A": d + c + a, **{f"B_k{k}": d + c + a / k for k in range(1, 6)},
                          "C_affine": d + a_in, **{f"D_affine_k{k}": d + a_in / k for k in range(1, 6)}},
            "measured_affine": {"A": single,
                                "C_affine": float(np.mean([((proj(Xs[i, j]) - y[i]) ** 2).mean() for i in range(len(keep)) for j in range(5)])),
                                "B_k5": float(((mu - y) ** 2).mean()),
                                "D_affine_k5": float(np.mean([((proj(mu[i]) - y[i]) ** 2).mean() for i in range(len(keep))]))},
            "per_form_sampling_var": dict(zip(FORMS, var.mean(0).tolist())),
            "per_form_signed_residual_mean": dict(zip(FORMS, np.einsum("ij,nj->ni", R, mu - p0).mean(0).tolist()))}
    # log-odds space
    L = lg(Xs); lmu = L.mean(1); lvar = L.var(1, ddof=1)
    la = lvar.mean()
    lres = []
    for i in range(len(keep)):
        pr = project(sg(lmu[i]), "ls_logit")
        lres.append(lmu[i] - lg(pr))
    lres = np.array(lres)
    lc = (lres ** 2).mean() - 0.75 * la / 5
    between = lmu.var(0, ddof=1).mean()
    logit = {"a_sampling_var": la, "c_incoherence_var_ls_logit": lc, "between_pair_var_of_mean": between,
             "icc_prompt_fixed_share_single_sample": between / (between + la),
             "per_form_sampling_var": dict(zip(FORMS, lvar.mean(0).tolist()))}
    if words:
        W = np.array([[np.clip(S_all[p][w], LO, 1 - LO) for w in WORDS] for p in keep])  # n,4,12
        wv = W.var(1, ddof=1)
        prob["b_wording_raw_var_4wordings"] = wv.mean()
        prob["b_wording"] = wv.mean() - a
        dev = W - W.mean(1, keepdims=True)
        # split wording deviations into a fixed template effect (same shift for every pair, per form x wording)
        # and a pair-specific part
        fixed = dev.mean(0, keepdims=True)
        prob["b_wording_template_fixed_part"] = float((fixed ** 2).mean() * 4 / 3)
        prob["b_wording_pair_specific_part"] = float(((dev - fixed) ** 2).sum(1).mean() / 3) - a
        prob["wording_dev_share_outside_subspace"] = float((np.einsum("ij,nwj->nwi", R, dev) ** 2).sum() / (dev ** 2).sum())
        prob["template_shift_vs_s0_mean_signed"] = {w: float((W[:, j] - Xs[:, 0]).mean()) for j, w in enumerate(WORDS) if j}
        prob["template_shift_vs_s0_per_form"] = {w: dict(zip(FORMS, (W[:, j] - Xs[:, 0]).mean(0).tolist()))
                                                 for j, w in enumerate(WORDS) if j}
        LW = lg(W); lwv = LW.var(1, ddof=1)
        logit["b_wording_raw_var_4wordings"] = lwv.mean(); logit["b_wording_var"] = lwv.mean() - la
        logit["template_shift_vs_s0_mean_signed"] = {w: float((LW[:, j] - L[:, 0]).mean()) for j, w in enumerate(WORDS) if j}
        # each wording on its own (one sample each, so a full a of noise): its incoherence and shared error
        prob["per_wording"] = {}
        for j, w in enumerate(WORDS):
            v = W[:, j]
            cw = (np.einsum("ij,nj->ni", R, v - p0) ** 2).sum(1).mean() / 12 - (np.diag(R) * var).sum(1).mean() / 12
            rw = ((v - y) ** 2).mean() - a
            prob["per_wording"][w] = {"c_incoherence": cw, "d_shared_coherent_error": rw - cw, "brier": float(((v - y) ** 2).mean())}
        # wording variance without w3 (the negated rewrite)
        prob["b_wording_without_w3"] = float(W[:, :3].var(1, ddof=1).mean() - a)
        # the wording-averaged answer: its incoherence and shared error (noise: a/4 per form)
        wm = W.mean(1)
        wc = (np.einsum("ij,nj->ni", R, wm - p0) ** 2).sum(1).mean() / 12 - (np.diag(R) * var).sum(1).mean() / 4 / 12
        wr = ((wm - y) ** 2).mean() - a / 4
        prob["wording_mean"] = {"c_incoherence": wc, "d_shared_coherent_error": wr - wc, "measured_brier": float(((wm - y) ** 2).mean()),
                                "same_prompt_mean_of_4_expected_brier": d + c + a / 4}
    # correlations
    res = np.einsum("ij,nj->ni", R, mu - p0)
    err = mu - y
    noise = (Xs - mu[:, None, :]).reshape(-1, 12)
    null = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    cr = np.corrcoef(res.T); ce = np.corrcoef(err.T); cn = np.corrcoef(noise.T)
    cres_err = np.corrcoef(np.einsum("ij,nj->ni", Pi, mu - p0) + p0 - y, rowvar=False)
    off = ~np.eye(12, dtype=bool)
    excess = cr - null
    top = sorted(((float(excess[i, j]), FORMS[i], FORMS[j], float(cr[i, j]), float(null[i, j]))
                  for i in range(12) for j in range(i + 1, 12)), key=lambda t: -abs(t[0]))[:8]
    corr = {"forms": FORMS, "residual_corr": cr.tolist(), "residual_null_corr_independent_errors": null.tolist(),
            "error_vs_truth_corr": ce.tolist(), "coherent_part_error_corr": cres_err.tolist(),
            "sampling_noise_corr": cn.tolist(),
            "sampling_noise_mean_abs_offdiag": float(np.abs(cn[off]).mean()),
            "residual_minus_null_mean_abs_offdiag": float(np.abs(excess[off]).mean()),
            "residual_top_excess_pairs": [{"excess": t[0], "a": t[1], "b": t[2], "corr": t[3], "null": t[4]} for t in top],
            "residual_variance_explained_by_first_pc": float(np.linalg.eigvalsh(np.cov(res.T))[-1] / np.trace(np.cov(res.T)))}
    return {"probability": prob, "log_odds": logit, "correlation": corr}


def load(run: Path):
    S = defaultdict(dict); meta = {}
    for l in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(l)
        S[r["pair_id"]].setdefault(r["sample"], np.full(12, np.nan))[FORMS.index(r["form"])] = np.nan if r["p"] is None else r["p"]
        meta[r["query_id"]] = r
    Y, O = {}, {}
    for r in meta.values():
        Y.setdefault(r["pair_id"], np.zeros(12)); O.setdefault(r["pair_id"], np.zeros(12, bool))
        Y[r["pair_id"]][FORMS.index(r["form"])] = r["label"]; O[r["pair_id"]][FORMS.index(r["form"])] = r["observed"]
    for l in Path(VBASE).read_text().splitlines():
        r = json.loads(l)
        if r["pair_id"] in S:
            S[r["pair_id"]].setdefault("vllm", np.full(12, np.nan))[FORMS.index(r["form"])] = np.nan if r["p"] is None else r["p"]
    return S, Y, O, meta


def agreement(S, pairs) -> dict:
    a = np.concatenate([S[p]["t0"] for p in pairs]); b = np.concatenate([S[p]["vllm"] for p in pairs])
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    return {"n_forms": int(ok.sum()), "exact_same": float(np.mean(np.abs(a - b) < 1e-9)),
            "within_0.05": float(np.mean(np.abs(a - b) <= 0.05 + 1e-9)), "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "pearson_logodds": float(np.corrcoef(lg(a), lg(b))[0, 1]), "pearson_prob": float(np.corrcoef(a, b)[0, 1]),
            "mean_signed_diff_dashscope_minus_vllm": float(np.mean(a - b))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", choices=["subset", "all"], required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()
    run = Path(args.run)
    S, Y, O, meta = load(run)
    sub = json.loads((run / "subset.json").read_text())
    need = ["t0"] + SAMPLES
    pairs = sub["pair_ids"] if args.pairs == "subset" else sorted(S)
    pairs = [p for p in pairs if all(s in S[p] for s in need)]
    words = args.pairs == "subset" and all(all(w in S[p] for w in ("w1", "w2", "w3")) for p in pairs)
    tasks = [(p, S[p], Y[p], O[p], words) for p in pairs]
    with Pool(args.procs) as pool:
        res = pool.map(pair_arms, tasks, chunksize=4)
    pp = defaultdict(dict); mov = []
    for pid, arms, m in res:
        for k, v in arms.items():
            pp[k][pid] = v
        mov.append(m)
    # magnitude-matched structure-free shrinkage (E_shrink)
    target = float(np.mean(mov))
    singles = {p: [S[p][s] for s in SAMPLES] for p in pairs}

    def movement(lam):
        return np.nanmean([np.nanmean(np.abs(sg(lam * lg(x)) - np.clip(x, LO, 1 - LO))) for p in pairs for x in singles[p]])
    lo_, hi_ = 0.0, 1.0
    for _ in range(40):
        mid = (lo_ + hi_) / 2
        lo_, hi_ = (mid, hi_) if movement(mid) > target else (lo_, mid)
    lam = (lo_ + hi_) / 2
    for p in pairs:
        pp["E_shrink_matched"][p] = np.mean([pmetrics(sg(lam * lg(x)), Y[p], O[p]) for x in singles[p]], 0)
    arms = {k: summary(v, pairs) for k, v in pp.items()}
    cmp = {
        "B_prob_k5_vs_A": ("B_prob_k5", "A"), "B_logit_k5_vs_A": ("B_logit_k5", "A"),
        "C_kl_vs_A": ("C_kl", "A"), "C_ls_logit_vs_A": ("C_ls_logit", "A"), "C_ls_prob_vs_A": ("C_ls_prob", "A"),
        "D_kl_prob_k5_vs_B_prob_k5": ("D_kl_prob_k5", "B_prob_k5"), "D_kl_logit_k5_vs_B_logit_k5": ("D_kl_logit_k5", "B_logit_k5"),
        "C_kl_vs_B_logit_k5": ("C_kl", "B_logit_k5"), "D_kl_logit_k5_vs_C_kl": ("D_kl_logit_k5", "C_kl"),
        "E_group_kl_vs_C_kl": ("E_group_kl", "C_kl"), "E_full_kl_vs_C_kl": ("E_full_kl", "C_kl"),
        "E_group_kl_vs_A": ("E_group_kl", "A"), "E_shrink_matched_vs_A": ("E_shrink_matched", "A"),
        "E_shrink_matched_vs_C_kl": ("E_shrink_matched", "C_kl"),
        "F_comp_kl_vs_A": ("F_comp_kl", "A"), "F_lit_kl_vs_A": ("F_lit_kl", "A"), "C_kl_vs_F_comp_kl": ("C_kl", "F_comp_kl"),
        "A_vs_A0": ("A", "A0"), "A0_vs_vllm_baseline": ("A0", "vllm_baseline"), "A0_proj_kl_vs_A0": ("A0_proj_kl", "A0"),
    }
    if words:
        cmp.update({"W_logit_vs_B_logit_k4": ("W_logit_4wordings", "B_logit_k4"),
                    "W_prob_vs_B_prob_k4": ("W_prob_4wordings", "B_prob_k4"),
                    "W_logit_proj_vs_D_kl_logit_k4": ("W_logit_4wordings_proj_kl", "D_kl_logit_k4")})
    cis = {name: boot(pp[x], pp[y_], pairs) for name, (x, y_) in cmp.items()}
    curves = {name: [arms[f"{name}_k{k}"]["all12_log_loss"] for k in range(1, 6)]
              for name in ("B_prob", "B_logit", "D_kl_prob", "D_kl_logit")}
    calls = defaultdict(int); missing = 0
    recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    total_cost = sum(r["cost"] for r in recs)
    inpairs = set(pairs)
    recs = [r for r in recs if r["pair_id"] in inpairs]
    cost = sum(r["cost"] for r in recs)
    for r in recs:
        calls[r["sample"]] += 1; missing += r["p"] is None
    out = {"pairs": args.pairs, "n_pairs": len(pairs), "subset_seed": sub["seed"], "wordings": words,
           "e_shrink_lambda": lam, "c_kl_mean_abs_movement": target,
           "calls_per_arm_per_pair": {"A0": 12, "A": 12, "C": 12, "B_k": "12k", "D_k": "12k", "W": 48},
           "run": {"calls_by_sample": dict(calls), "missing_after_retry": missing, "retried": sum(r["retried"] for r in recs),
                   "cost_usd_list_price_these_pairs": cost, "cost_usd_list_price_whole_log": total_cost,
                   "served_models": sorted({r["served_model"] for r in recs if r["served_model"]})},
           "agreement_vs_vllm": agreement(S, pairs), "arms": arms, "curves_all12_log_loss_k1_to_5": curves,
           "bootstrap": cis, "noise": decompose(S, Y, pairs, words)}
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
    for k in sorted(arms):
        a = arms[k]
        print(f"{k:34s} all12 {a['all12_log_loss']:.4f} obs {a['observed_log_loss']:.4f} andor {a['andor_log_loss']:.4f} "
              f"brier {a['all12_brier']:.4f} msv {a['coherence_msv']:.4f}")
    for k, v in cis.items():
        print(f"{k:34s} {v['all12_log_loss']['diff']:+.4f} {np.round(v['all12_log_loss']['ci95'], 4).tolist()}")


if __name__ == "__main__":
    main()
