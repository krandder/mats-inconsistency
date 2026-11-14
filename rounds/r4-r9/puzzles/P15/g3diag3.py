"""G3 diagnostic: maze flips at softer level margins -> KL(q||p) and one Bl gradient step's direction."""
import json, sys, time
sys.path.insert(0, "/root/job")
import gates as G
import p15
Tm = G.Task("maze", 32)
dmi = Tm.eval_set("indist", num=64)
nbr_cache = [Tm.nbr(Tm.n)]
C, E_, S_, W_, N_ = p15.WIN_C, p15.WIN_E, p15.WIN_S, p15.WIN_W, p15.WIN_N


def m_sel(kind):
    def s(x, zw):
        xc = G.gather(x[None], nbr_cache[0])[0][..., C, :]
        fc, fe, fs, fw, fn = zw[..., C], zw[..., E_], zw[..., S_], zw[..., W_], zw[..., N_]
        notend = (xc[..., 4] == 0) & (xc[..., 5] == 0) & (fc == 0)
        if kind == "corridor_fill":
            return notend & (xc[..., 0] == 1) & (xc[..., 2] == 1) & (xc[..., 1] == 0) & (xc[..., 3] == 0) & (fe == 0) & (fw == 0)
        if kind == "deadend_keep":
            return notend & (xc[..., 0] == 1) & (xc[..., 1] == 0) & (xc[..., 2] == 0) & (xc[..., 3] == 0) & (fe == 0)
        return notend & (xc[..., 1] == 1) & (xc[..., 3] == 1) & (xc[..., 0] == 0) & (xc[..., 2] == 0) & (fs == 0) & (fn == 0)
    return s


def m_tgt(x, zw):
    xc = G.gather(x[None], nbr_cache[0])[0][..., p15.WIN_C, :]
    fc = zw[..., p15.WIN_C]
    cnt = sum(xc[..., d] * (1 - zw[..., w]) for d, w in enumerate((p15.WIN_E, p15.WIN_S, p15.WIN_W, p15.WIN_N)))
    end = (xc[..., 4] + xc[..., 5]) > 0
    return ((fc > 0) | (~end & (cnt <= 1))).long()


for M in (4.0, 8.0, 16.0):
    for kind in ("corridor_fill", "deadend_keep", "corridorNS_fill"):
        t0 = time.time()
        Pp = G.maze_rule(M=M, pert=kind)
        kl, _ = G.rollout_settle(Pp, Tm, dmi, 32)
        b0, cnt = G.pattern_logp(Pp, Tm, dmi, 32, m_sel(kind), m_tgt)
        a0, _ = G.pattern_logp(G.one_bl_step(Pp, Tm, dmi, 32, lr=1e-3, sgd=True), Tm, dmi, 32, m_sel(kind), m_tgt)
        r = {"M": M, "kind": kind, "kl": kl, "logp_before": b0, "logp_after": a0,
             "pass": bool(kl > 0.05 and a0 > b0), "n": cnt, "s": round(time.time() - t0)}
        print(json.dumps(r), flush=True)
