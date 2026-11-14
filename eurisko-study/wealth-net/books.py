"""Wealth accounting by attribution flow.

Settlement per question and example: s = y - p (the negative loss gradient at the logit).
Output revenue R = sum_q z_q * s_q (s detached): each readout 'demands' its logit and is
settled like an LMSR position. Inside the network a unit with activation a earns
a * dR/da: one backward pass of R (never of the loss) gives every unit's first-order
attribution share. Consumers pay suppliers through the chain rule, so for any complete
cut of the graph the revenues add up to R up to first-order error (non-homogeneous parts:
attention softmax) plus the revenue of constant inputs (biases, LN shifts) below the cut.

The books use a separate forward pass with LayerNorm denominators detached (LN is
scale-invariant, so without this every inner cut would show ~0). torch.autograd.grad
returns gradients without touching any parameter .grad, so the loss gradients used by
the optimiser never mix with these.

All revenues are per example (divided by batch size) and split by the question family
whose settlement paid them (8 cotangents, one per family, batched with vmap).
"""
import torch

N_FAM = 8


def _targets(model, rec):
    L = model.L
    biases = model.bias_params()
    tg = list(rec["cuts"]) + list(rec["hv"]) + list(rec["act"]) + [b for _, b, _ in biases]
    return tg, biases


def account(model, win, y, fam_onehot, freeze_attn=False, batched=True):
    """Return dict of revenues (tensors, last axis = family) and conservation diagnostics.

    fam_onehot: [Q, N_FAM] float tensor.
    """
    B = win.shape[0]
    L, H = model.L, model.H
    rec = {}
    z = model(win, rec=rec, freeze_ln=True, freeze_attn=freeze_attn)
    s = (y - torch.sigmoid(z)).detach()                          # [B,Q]
    cot = s.unsqueeze(0) * fam_onehot.T.unsqueeze(1)             # [F,B,Q]
    tg, biases = _targets(model, rec)
    if batched:
        grads = torch.autograd.grad(z, tg, grad_outputs=cot, is_grads_batched=True)
    else:
        per = [torch.autograd.grad(z, tg, grad_outputs=cot[f], retain_graph=True) for f in range(N_FAM)]
        grads = [torch.stack([p[i] for p in per]) for i in range(len(tg))]
    nc = 2 * L + 1
    g_cut = grads[:nc]
    g_hv = grads[nc:nc + L]
    g_act = grads[nc + L:nc + 2 * L]
    g_b = grads[nc + 2 * L:]
    cuts = rec["cuts"]
    out = {}
    zd = z.detach()
    out["out"] = (zd.unsqueeze(0) * cot).sum(1).T / B              # [Q,F]
    out["R"] = out["out"].sum(0)                                   # [F]
    out["cut"] = torch.stack([(cuts[k].detach() * g_cut[k]).sum((1, 2, 3)) for k in range(nc)]) / B  # [nc,F]
    comp = [(rec["e_w"].detach() * g_cut[0]).sum((1, 2, 3)), (rec["e_q"].detach() * g_cut[0]).sum((1, 2, 3))]
    for l in range(L):
        comp.append((rec["attn_out"][l].detach() * g_cut[2 * l + 1]).sum((1, 2, 3)))
        comp.append((rec["mlp_out"][l].detach() * g_cut[2 * l + 2]).sum((1, 2, 3)))
    out["comp"] = torch.stack(comp) / B                            # [2+2L, F]
    out["head"] = torch.stack([(rec["hv"][l].detach() * g_hv[l]).sum((1, 3, 4)) for l in range(L)]).permute(0, 2, 1) / B  # [L,H,F]
    out["neuron"] = torch.stack([(rec["act"][l].detach() * g_act[l]).sum((1, 2)) for l in range(L)]).permute(0, 2, 1) / B  # [L,N,F]
    out["bias"] = torch.stack([(b.detach() * g).reshape(N_FAM, -1).sum(1) for (_, b, _), g in zip(biases, g_b)]) / B  # [nb,F]
    # Conservation: R = cut_k + revenue of biases downstream of cut k (+ residual).
    R = out["R"].sum()
    bidx = torch.tensor([i for _, _, i in biases], device=R.device)
    btot = out["bias"].sum(1)
    resid = []
    for k in range(nc):
        resid.append(R - out["cut"][k].sum() - btot[bidx >= k].sum())
    out["cons_resid"] = torch.stack(resid)                          # [nc]
    # Sum rules: component = sum of its units + its output bias.
    names = [n for n, _, _ in biases]
    sr = []
    for l in range(L):
        sr.append(out["comp"][2 + 2 * l].sum() - out["head"][l].sum() - btot[names.index(f"L{l}.o.bias")])
        sr.append(out["comp"][3 + 2 * l].sum() - out["neuron"][l].sum() - btot[names.index(f"L{l}.fc2.bias")])
    out["sum_resid"] = torch.stack(sr)                              # [2L]
    out["act_count"] = torch.stack([(a > 0).float().sum((0, 1)) for a in rec["act"]]) / B  # [L,N] per example
    out["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(zd, y)
    return {k: v.detach() for k, v in out.items()}


def grad_norms(model):
    """Per-unit L2 norm of the LOSS gradient (call after loss.backward())."""
    L, H, dh = model.L, model.H, model.d // model.H
    d = model.d
    gn = {}
    gn["out"] = torch.sqrt((model.rw.grad ** 2).sum(1) + model.rb.grad ** 2)
    heads, neurons, comp = [], [], []
    comp.append(torch.sqrt((model.win.weight.grad ** 2).sum() + (model.win.bias.grad ** 2).sum()))
    comp.append(torch.sqrt((model.qid.grad ** 2).sum()))
    for blk in model.blocks:
        wq = blk.qkv.weight.grad.view(3, H, dh, d)
        bq = blk.qkv.bias.grad.view(3, H, dh)
        wo = blk.o.weight.grad.view(d, H, dh)
        heads.append(torch.sqrt((wq ** 2).sum((0, 2, 3)) + (bq ** 2).sum((0, 2)) + (wo ** 2).sum((0, 2))))
        neurons.append(torch.sqrt((blk.fc1.weight.grad ** 2).sum(1) + blk.fc1.bias.grad ** 2 + (blk.fc2.weight.grad ** 2).sum(0)))
        pa = [blk.ln1.g, blk.ln1.b, blk.qkv.weight, blk.qkv.bias, blk.o.weight, blk.o.bias]
        pm = [blk.ln2.g, blk.ln2.b, blk.fc1.weight, blk.fc1.bias, blk.fc2.weight, blk.fc2.bias]
        comp.append(torch.sqrt(sum((p.grad ** 2).sum() for p in pa)))
        comp.append(torch.sqrt(sum((p.grad ** 2).sum() for p in pm)))
    gn["head"] = torch.stack(heads)
    gn["neuron"] = torch.stack(neurons)
    gn["comp"] = torch.stack(comp)
    return gn


def account_logistic(lm, win, y, fam_onehot):
    """Books for the logistic baseline: unit = one weight (question q, lag j), plus biases."""
    B = win.shape[0]
    with torch.no_grad():
        z = lm(win)
        s = y - torch.sigmoid(z)
        contrib = (2.0 * win - 1.0) * lm.w                         # [B,Q,W]
        lag = (contrib * s.unsqueeze(-1)).sum(0) / B               # [Q,W]
        bias = (lm.b * s).sum(0) / B                               # [Q]
        R = (z * s).sum() / B
    return {"lag": lag, "bias": bias, "cons_resid": R - lag.sum() - bias.sum()}
