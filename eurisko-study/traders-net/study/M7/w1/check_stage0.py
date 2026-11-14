"""M7 Stage 0: the router gradient in the harness MoE layer is g_i (r_i - R), R = sum_S g r.

Checks, in float64 on the real harness layer (moe_harness.MoE) inside a real MoENet forward on an
S0 stream batch, 3 seeds x E in {8, 64}:
  C0a  renormalised softmax over S (the harness convention, arms none/switch/deepseek):
       -dL/dz_i = g_i (r_i - R) for i in S, 0 outside.      [T1 Lemma 7 (i)]
  C0b  same with a nonzero DeepSeek selection bias b_sel (enters selection only): still (i).
  C0c  expert choice (softmax over all E, column top-C, not renormalised per token):
       -dL/dz_i = s_i (r_i 1[i in S_n] - R_n).                [Lemma 7 (ii) form]
  C0d  zero-sum: sum_i dL/dz_i = 0 per token for both softmax conventions.
  C0e  Switch aux loss adds alpha*E*f_i*dP_i/dz: the router gradient departs from (i) by exactly
       that term (so the aux loss is a tax outside the revenue identity).
Revenue r_i = <E_i(x), -dL/dh> for every expert. Writes results/stage0.json.
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_harness as mh

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def one(E, seed, pol_cls, bias=False, pkw=None):
    torch.manual_seed(seed)
    s = mh.streams_ext.make_streams(Q=64, T=600, seed=seed, regime_every=0)
    st = mh.streams_ext.to_array_stream(s)
    net = mh.MoENet(st.Q, st.W, E, pol_cls, pkw=pkw)
    for p in net.parameters():
        p.data = p.data.double()
    net.qid.data.normal_(0, 1.0)
    for m in net.moe:
        m.router.weight.data.normal_(0, 0.5)   # spread gates away from uniform
        if bias:
            m.b_sel.data = torch.randn(E, dtype=torch.float64)
    P = np.arange(100, 104)
    win, y = st.window(P).double(), st.label(P).double()
    n = win.shape[0]
    net.ctx = dict(fam=torch.as_tensor(st.fam).repeat(n), qid=torch.arange(st.Q).repeat(n), step=0, train=True)
    zs = []
    for m in net.moe:          # capture router logits with grad
        def hook(mod, i, o):
            o.retain_grad(); zs.append(o)
        m.router.register_forward_hook(hook)
    logits = net(win)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    total = loss + net.aux
    total.backward()
    out = []
    for l, m in enumerate(net.moe):
        L_ = m.last
        z, G = zs[l], L_["G"].detach()
        u = -z.grad                                            # autograd -dL/dz
        r = -torch.einsum("ned,nd->ne", L_["Eo"].detach(), L_["out"].grad)
        R = (G * r).sum(-1, keepdim=True)
        if pol_cls is mh.ExpertChoice:
            sm = torch.softmax(z.detach(), -1)
            closed = sm * (r * (G > 0) - R)
        else:
            closed = G * (r - R)                                # zero outside S since G=0 there
        if pol_cls is mh.SwitchAux:                             # aux term: -d(aux)/dz
            zz = z.detach().clone().requires_grad_(True)
            aux = net.policy.aux_loss(m, zz, G, net.ctx)
            aux.backward()
            closed = closed - zz.grad
        err = (u - closed).abs().max().item() / max(u.abs().max().item(), 1e-300)
        zsum = u.sum(-1).abs().max().item() / max(u.abs().max().item(), 1e-300)
        out.append(dict(layer=l, rel_err=err, rel_row_sum=zsum, grad_scale=u.abs().max().item()))
    return out


res = {}
for name, cls, bias, pkw in [("C0a_renorm_topk", mh.Policy, False, None),
                             ("C0b_renorm_topk_with_bsel", mh.Policy, True, None),
                             ("C0c_expert_choice", mh.ExpertChoice, False, None),
                             ("C0e_switch_aux", mh.SwitchAux, False, {"alpha": 1e-2})]:
    rows = []
    for E in (8, 64):
        for seed in (1, 2, 3):
            for r in one(E, seed, cls, bias, pkw):
                r.update(E=E, seed=seed); rows.append(r)
    res[name] = dict(max_rel_err=max(r["rel_err"] for r in rows),
                     max_rel_row_sum=max(r["rel_row_sum"] for r in rows), rows=rows)
    print(name, "max rel err %.2e" % res[name]["max_rel_err"], "max |row sum|/scale %.2e" % res[name]["max_rel_row_sum"])
res["verdict"] = {k: bool(v["max_rel_err"] < 1e-10) for k, v in res.items()}
res["gate_convention"] = ("z = W_r x (no bias); S = top-k(z + b_sel), b_sel non-gradient (DeepSeek bias only); "
                          "g = softmax(z_S) renormalised over S; -dL/dz_i = g_i (r_i - R) on S, 0 off S")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", "stage0.json"), "w"), indent=1)
