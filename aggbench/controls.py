"""Controls for learned aggregators (PROTOCOL 4 and 9.1).

A learned aggregator is any object with
    .module                      a torch.nn.Module holding all learned weights
    .fit(view, truth)            training
    .predict(view) -> forecast   the full inference pipeline
and a factory `make(seed)` that builds it at initialization.

run_with_controls trains one copy and evaluates three arms through the same predict():
    "trained"                  the trained aggregator
    "control:untrained_init"   the same architecture at the same initialization, not trained
    "control:random_network"   the trained aggregator with every weight tensor replaced by random
                               values of matching mean and sd (scale-matched random weights)
A gain that the random-network control also shows is not evidence that learning did anything.
"""
import copy


def random_weights(agg, seed):
    """Deep copy of `agg` whose module weights are random, scale-matched per tensor."""
    import torch
    rnd = copy.deepcopy(agg)
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in rnd.module.parameters():
            if p.numel() > 1:
                mean, sd = p.mean(), p.std()
            else:
                mean, sd = torch.zeros_like(p).sum(), p.abs().sum()  # device follows p
            p.copy_(mean + sd * torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device))
    return rnd


def untrained_init(make, seed):
    """The aggregator exactly as `make(seed)` builds it, never fitted."""
    return make(seed)


def run_with_controls(make, train_view, train_truth, eval_view, seed=0):
    """Dict arm -> forecast on eval_view for trained, untrained-init and random-network arms."""
    trained = make(seed)
    trained.fit(train_view, train_truth)
    return {
        "trained": trained.predict(eval_view),
        "control:untrained_init": untrained_init(make, seed).predict(eval_view),
        "control:random_network": random_weights(trained, seed + 1).predict(eval_view),
    }
