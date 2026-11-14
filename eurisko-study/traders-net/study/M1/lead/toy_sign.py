"""M1 retarget (a): a toy where gradient credit and market settlement provably disagree in sign.

Unit i enters the logit linearly: z = l_rest + w_i a_i. Let L(w) = E[BCE] as a function of w_i
(convex, since BCE is convex in z and z is affine in w_i).
  gradient credit (first-order Taylor / movement / books income) C = -w L'(w)
  ablation value (= expected exact LMSR settlement of a trader who commits w a_i at the
  price l_rest and is paid log p_new(y) - log p_old(y))     V = L(0) - L(w)
Convexity gives V >= C, C = 0 at the stationary point w*, C < 0 < V on (w*, w0) with
L(w0) = L(0). Part 1 checks this in closed form for x in {-1,1}, P(y=1|x) = sigmoid(beta x):
  C(w) = w (sigmoid(beta) - sigmoid(w)),  V(w) = log 2 - CE(sigmoid(beta), sigmoid(w)).
Part 2 simulates online training (Adam and SGD) of two redundant units carrying x plus one noise
unit, 3 seeds, and books per step (prequential, before the update): gradient credit a_i w_i (y - p)
and the committed-trader settlement log p(y) - log p_{-i}(y).
Writes lead/results/toy_closed_form.csv, toy_sim.csv, toy_summary.json.
"""
import csv, json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
sig = lambda t: 1.0 / (1.0 + np.exp(-t))


def closed_form(beta=1.0):
    pb = sig(beta)
    ws = np.round(np.arange(0.0, 4.01, 0.05), 4)
    rows = []
    for w in ws:
        C = w * (pb - sig(w))
        V = np.log(2) + pb * np.log(sig(w)) + (1 - pb) * np.log(1 - sig(w))
        rows.append(dict(w=w, credit=C, value=V))
    # w0: root of V beyond beta (bisection)
    V = lambda w: np.log(2) + pb * np.log(sig(w)) + (1 - pb) * np.log(1 - sig(w))
    lo, hi = beta, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if V(mid) > 0 else (lo, mid)
    w0 = 0.5 * (lo + hi)
    checks = dict(
        beta=beta, w0=w0,
        credit_at_beta=beta * (pb - sig(beta)), value_at_beta=V(beta),
        mutual_info_nats=np.log(2) - (-(pb * np.log(pb) + (1 - pb) * np.log(1 - pb))),
        disagree_interval=[beta, w0],
        convexity_V_ge_C_all=bool(all(r["value"] >= r["credit"] - 1e-12 for r in rows)),
        sign_disagree_grid_points=int(sum(1 for r in rows if r["credit"] < 0 < r["value"])),
        grid_points_in_interval=int(sum(1 for r in rows if beta < r["w"] < w0)),
    )
    return rows, checks


def simulate(seed, opt, steps=20000, beta=1.0, lr=0.05, batch=8, noise_sd=1.0):
    rng = np.random.default_rng(seed)
    w = np.zeros(3)  # units: 0 and 1 carry x (redundant), 2 carries pure noise
    m = np.zeros(3); v = np.zeros(3)
    credit = np.zeros(3); settle = np.zeros(3); g2 = np.zeros(3); tr = []
    for t in range(1, steps + 1):
        x = rng.choice([-1.0, 1.0], size=batch)
        y = (rng.random(batch) < sig(beta * x)).astype(float)
        a = np.stack([x, x, rng.normal(0, noise_sd, batch)], 1)   # [b,3]
        z = a @ w
        p = sig(z)
        lp = lambda zz: y * np.log(sig(zz)) + (1 - y) * np.log(1 - sig(zz))
        # prequential books, before the update
        credit += (a * w * (y - p)[:, None]).mean(0)
        settle += np.array([(lp(z) - lp(z - a[:, i] * w[i])).mean() for i in range(3)])
        g = -(a * (y - p)[:, None]).mean(0)
        if opt == "adam":
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        else:
            g2 += g * g
            w -= lr * g * 4  # plain SGD, step eta = 0.2
        if t % 1000 == 0:
            tr.append(dict(seed=seed, opt=opt, step=t, w0=w[0], w1=w[1], w2=w[2], wsum=w[0] + w[1],
                           credit0=credit[0], credit1=credit[1], credit2=credit[2],
                           settle0=settle[0], settle1=settle[1], settle2=settle[2]))
    # exact held-out ablation value of each unit at the final weights (closed form over x, y; noise unit by MC)
    pb = sig(beta)
    Lz = lambda wx: -(pb * np.log(sig(wx)) + (1 - pb) * np.log(1 - sig(wx)))  # x=+1 case; symmetric
    rng2 = np.random.default_rng(10_000 + seed)
    xs = rng2.choice([-1.0, 1.0], 200000); ys = (rng2.random(200000) < sig(beta * xs)).astype(float); ns = rng2.normal(0, noise_sd, 200000)
    def L(wv):
        zz = xs * (wv[0] + wv[1]) + ns * wv[2]
        return float(-(ys * np.log(sig(zz)) + (1 - ys) * np.log(1 - sig(zz))).mean())
    base = L(w)
    abl = [L(np.where(np.arange(3) == i, 0.0, w)) - base for i in range(3)]
    eta = lr * 4
    ident = (w * w / (2 * eta) - eta / 2 * g2) if opt == "sgd" else None   # credit = norm growth - (eta/2) sum g^2
    return tr, dict(seed=seed, opt=opt, sgd_identity_rhs=None if ident is None else ident.tolist(),
                    sgd_energy_term=None if ident is None else (eta / 2 * g2).tolist(), w=w.tolist(), credit=credit.tolist(), settle=settle.tolist(), ablation_dBCE=abl,
                    wsum_minus_beta=float(w[0] + w[1] - beta))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, checks = closed_form()
    with open(os.path.join(OUT, "toy_closed_form.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    trs, fins = [], []
    for opt in ("adam", "sgd"):
        for s in (1, 2, 3):
            tr, fin = simulate(s, opt); trs += tr; fins.append(fin)
    with open(os.path.join(OUT, "toy_sim.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(trs[0])); wr.writeheader(); wr.writerows(trs)
    summ = {}
    for opt in ("adam", "sgd"):
        F = [x for x in fins if x["opt"] == opt]
        red_credit = np.array([x["credit"][:2] for x in F]).ravel()      # 6 redundant-unit values
        red_settle = np.array([x["settle"][:2] for x in F]).ravel()
        red_abl = np.array([x["ablation_dBCE"][:2] for x in F]).ravel()
        summ[opt] = dict(
            redundant_units_n=int(red_credit.size),
            credit_mean=float(red_credit.mean()), credit_sd=float(red_credit.std(ddof=1)),
            credit_neg_frac=float((red_credit < 0).mean()),
            settle_mean=float(red_settle.mean()), settle_sd=float(red_settle.std(ddof=1)),
            settle_neg_frac=float((red_settle < 0).mean()),
            ablation_mean=float(red_abl.mean()), ablation_sd=float(red_abl.std(ddof=1)),
            credit_over_settle=float(red_credit.mean() / red_settle.mean()),
            noise_unit_credit=[x["credit"][2] for x in F], noise_unit_settle=[x["settle"][2] for x in F],
            wsum_minus_beta=[x["wsum_minus_beta"] for x in F], final=F)
        if opt == "sgd":
            summ[opt]["sgd_identity_max_abs_err"] = float(max(abs(c - r) for x in F for c, r in zip(x["credit"], x["sgd_identity_rhs"])))
    json.dump(dict(closed_form=checks, sim=summ), open(os.path.join(OUT, "toy_summary.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))
    for opt in summ:
        s = summ[opt]
        print(opt, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "final"})


if __name__ == "__main__":
    main()
