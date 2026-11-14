"""Import the locked traders-net currency (reads CURRENCY.lock, checks the books_v2 hash).

    LOCK["headline"]: no bookkeeping currency for generic neurons; committed-trader settlement only.
    The currency is committed_settlement(); make_books/settle give DIAGNOSTIC BooksV2 books, not wealth.

    import sys; sys.path.insert(0, ".../traders-net/study/M1"); sys.path.insert(0, ".../traders-net/study/S0/substrate")
    from currency import make_books, settle
    bk = make_books(model, fam, n_fam)          # BooksV2 with the locked prior
    acc = settle(bk, win, y, mask)              # prequential: call BEFORE the optimiser step on this block
    wealth = bk.W[LOCK["wealth_book"]]          # {"out","head","neuron","comp"} -> [..., n_groups]
"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = json.load(open(os.path.join(HERE, "CURRENCY.lock")))
_sha = hashlib.sha256(open(os.path.join(HERE, "books_v2.py"), "rb").read()).hexdigest()
assert _sha == LOCK["code"]["sha256"], "books_v2.py changed since CURRENCY.lock; ask the M1 lead"
from books_v2 import BooksV2  # noqa: E402


def make_books(model, fam, n_fam, groups=None, batched=False):
    p = LOCK["prior"]
    return BooksV2(model, fam, n_fam, prior_a=p["prior_a"], prior_b=p["prior_b"], groups=groups, batched=batched)


def settle(bk, win, y, mask=None):
    """Book one block with the locked rule (cp, prequential), accumulate, then update prior/means.
    Must run before the optimiser trains on (win, y)."""
    acc = bk.account(win, y, mask, freeze_attn=True)
    bk.add(acc)
    import torch
    bk.observe(win, y, torch.ones_like(y) if mask is None else mask)
    return acc


def committed_settlement(p_trade, p_after, y, eps=1e-12):
    """Exact LMSR log-score payoff of a committed trader who moved the price of a binary question
    from p_trade (price at trade time) to p_after. Book at trade time, call only when y is revealed.
    Works on floats, numpy arrays or torch tensors (elementwise)."""
    try:
        import torch
        if torch.is_tensor(p_after):
            lg = lambda p: torch.log(torch.clamp(p, eps, 1.0))
        else:
            raise ImportError
    except ImportError:
        import numpy as np
        lg = lambda p: np.log(np.clip(p, eps, 1.0))
    import numpy as _np
    pa, pt = p_after, p_trade
    return y * (lg(pa) - lg(pt)) + (1 - y) * (lg(1 - pa) - lg(1 - pt))


if __name__ == "__main__":  # smoke test: telescoping, sign, lock headline
    import numpy as np
    assert LOCK["headline"] == "no bookkeeping currency for generic neurons; committed-trader settlement only"
    p = [0.5, 0.6, 0.8, 0.7]; y = 1
    tot = sum(committed_settlement(p[i], p[i + 1], y) for i in range(3))
    assert abs(tot - np.log(0.7 / 0.5)) < 1e-12, tot
    assert committed_settlement(0.5, 0.9, 1) > 0 > committed_settlement(0.5, 0.9, 0)
    assert abs(committed_settlement(0.3, 0.3, 1)) < 1e-15
    print("currency ok:", LOCK["headline"])
