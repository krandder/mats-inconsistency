import numpy as np
import torch
import harness as H
import looped as LP
import world3_chains as W


def test_looped_one_iter_equals_setpfn_and_ties_weights():
    torch.manual_seed(0)
    w = W.make_world()
    ep = w.sample(np.random.default_rng(0), 8, 3)
    X, M = torch.as_tensor(ep['X']), torch.as_tensor(ep['mask'])
    base = H.SetPFN(w.feat_dim, 2, 32, 4, 'cat')
    lp = LP.LoopedSetPFN(w.feat_dim, 2, 1, 32, 4, 'cat')
    lp.load_state_dict(base.state_dict())
    torch.testing.assert_close(lp(X, M), base(X, M))
    lp3 = LP.LoopedSetPFN(w.feat_dim, 2, 3, 32, 4, 'cat')
    assert sum(p.numel() for p in lp3.parameters()) == sum(p.numel() for p in base.parameters())
    # more iterations change the output; masked (pad) tokens do not affect the query
    lp3.load_state_dict(base.state_dict())
    assert not torch.allclose(lp3(X, M), base(X, M))
    X2 = X.clone(); X2[~M] = 5.0
    torch.testing.assert_close(lp3(X2, M), lp3(X, M))


def test_build_model_dispatch():
    w = W.make_world()
    a = dict(depth=4, width=32, heads=4, loop_block=2, loop_iters=3)
    assert isinstance(LP.build_model(w, a), LP.LoopedSetPFN)
    a['loop_iters'] = 0
    assert isinstance(LP.build_model(w, a), H.SetPFN)


def test_amp_wrapper_dispatch_and_cpu_identity():
    w = W.make_world()
    a = dict(depth=2, width=32, heads=4, loop_block=2, loop_iters=2, amp_bf16=True)
    m = LP.build_model(w, a)
    assert isinstance(m, LP.AmpWrap) and isinstance(m.m, LP.LoopedSetPFN) and m.kind == 'cat'
    ep = w.sample(np.random.default_rng(0), 4, 2)
    X, M = torch.as_tensor(ep['X']), torch.as_tensor(ep['mask'])
    torch.testing.assert_close(m(X, M), m.m(X, M))          # autocast is CUDA-only: identical on CPU
    o, qas = m(X, M, True)
    assert o.dtype == torch.float32 and len(qas) == 4
