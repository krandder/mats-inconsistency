"""M4 (scale-free-theory addendum, PREREG-SF.md): internal message size of the clean 2-layer nets.

After block 1, the residual stream of the 15 context tokens (all that block 2 reads from the context) is replaced by
c + (x - c) W_down W_up, a shared rank-k map (k = 0 keeps only c). The map is initialised from PCA of fit-episode
states and trained by distillation (KL(net || bottlenecked) on fresh episodes; net weights frozen). Score on a fixed
test set: mean KL(net || bottlenecked) and KL(Bayes || bottlenecked).
"""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/scale-free-theory')
from sf_net import load_clean, forward, episodes, head_logpdf, bayes_logpdf, kl_grid, YG

NETS = [f'w32_h2_L2_s{s}' for s in range(5)] + [f'w32_h2_L2_mix4_s{s}' for s in range(5)]
KS = [0, 1, 2, 3, 4, 6, 8]
FIT_SEED, TEST_SEED = 501, 502
STEPS, BATCH, LR = 600, 512, 3e-3
YT = torch.as_tensor(YG, dtype=torch.float32)


def logpdf_t(head):
    if len(head) == 2:
        mu, ls = head
        return -ls[:, None] - 0.5 * np.log(2 * np.pi) - 0.5 * ((YT[None] - mu[:, None]) / ls.exp()[:, None]) ** 2
    mu, ls, lg = head
    lw = torch.log_softmax(lg, -1)
    lp = lw[:, None, :] - ls[:, None, :] - 0.5 * np.log(2 * np.pi) - 0.5 * ((YT[None, :, None] - mu[:, None, :]) / ls.exp()[:, None, :]) ** 2
    return torch.logsumexp(lp, -1)


def kl_t(lp_p, lp_q):
    lp_p = torch.log_softmax(lp_p, -1); lp_q = torch.log_softmax(lp_q, -1)     # grid-normalised
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1).mean()


class Bottleneck(torch.nn.Module):
    def __init__(self, c, U, k):
        super().__init__()
        self.c = torch.nn.Parameter(c.clone())
        self.k = k
        if k > 0:
            self.down = torch.nn.Parameter(U[:, :k].clone()); self.up = torch.nn.Parameter(U[:, :k].T.clone())

    def forward(self, x):
        if self.k == 0:
            return self.c.expand_as(x)
        return self.c + (x - self.c) @ self.down @ self.up


def main():
    t0 = time.time()
    res = {}
    for name in (sys.argv[1:] or NETS):
        torch.manual_seed(0)
        model, meta = load_clean(name)
        for p in model.parameters():
            p.requires_grad_(False)
        dF, XF, mF = episodes(2048, FIT_SEED, meta['ds'])
        dT, XT, mT = episodes(4000, TEST_SEED, meta['ds'])
        with torch.no_grad():
            # context states after block 1 (PCA init)
            x = model.inp(XF)
            from sf_net import allow_mask, block_forward
            x1, _ = block_forward(model.blocks[0], x, allow_mask(mF))
            S = x1[:, :-1].reshape(-1, x1.shape[-1])
            c = S.mean(0)
            U = torch.linalg.svd(S - c, full_matrices=False)[2].T            # (d, d) principal directions
            headT, _ = forward(model, XT, mT)
            lpT_net = logpdf_t(headT)
        lp_bayes = bayes_logpdf(dT)
        rec = {'kl_bayes_net': float(kl_grid(lp_bayes, lpT_net.numpy().astype(np.float64)).mean())}
        for k in KS:
            bn = Bottleneck(c, U, k)
            opt = torch.optim.Adam(bn.parameters(), lr=LR)
            for step in range(STEPS):
                d_, Xb, mb = episodes(BATCH, 10_000 + step, meta['ds'])
                with torch.no_grad():
                    hn, _ = forward(model, Xb, mb)
                    lp_n = logpdf_t(hn)
                hb, _ = forward(model, Xb, mb, bottleneck=bn)
                loss = kl_t(lp_n, logpdf_t(hb))
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                hb, _ = forward(model, XT, mT, bottleneck=bn)
                lp_b = logpdf_t(hb).numpy().astype(np.float64)
            rec[f'k{k}'] = {'kl_net_bott': float(kl_grid(lpT_net.numpy().astype(np.float64), lp_b).mean()),
                            'kl_bayes_bott': float(kl_grid(lp_bayes, lp_b).mean()), 'final_train_loss': float(loss)}
            print(name, 'k', k, rec[f'k{k}'], round(time.time() - t0), 's', flush=True)
        res[name] = rec
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/m4_bottleneck_mix4.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
