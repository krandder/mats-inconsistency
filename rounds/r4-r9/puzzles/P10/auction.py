"""P10 slot auction (puzzles/P8 does not exist, so this is the local <=150-line version).

Every `every` training steps the caller measures each listed slot's held-out royalty
    G_k = E_{e~_k ~ p(e_k|X)}[l(Y|X, e_-k, e~_k)] - l(Y|X, e) - l(e_k|X)      (nats per cell)
and passes the dict {slot: G_k} to Auction.settle(step, G). The auction keeps an EMA of each slot's royalty
(the first reading initialises it) and, once a slot is older than `grace` steps, acts when the EMA is below the fee:
  - learned slots (index < k_learn) are respawned: the caller re-initialises the event function, its value
    embedding and its prior; the slot's age and EMA restart;
  - reference slots are de-listed for good (their value embedding is replaced by a learned constant).
The fee pot collects fee x listed slots at every settlement, and royalties are paid into a per-slot ledger, so the
books show what each slot earned. Nothing here touches the model; the caller applies the returned actions.
"""


class Auction:
    def __init__(self, slot_names, k_learn, fee, ema=0.5, grace=1000, every=500):
        self.names = list(slot_names)
        self.k_learn = k_learn
        self.fee = fee
        self.alpha = ema
        self.grace = grace
        self.every = every
        self.listed = {n: True for n in self.names}
        self.ema = {n: None for n in self.names}
        self.born = {n: 0 for n in self.names}
        self.earned = {n: 0.0 for n in self.names}
        self.fees_paid = {n: 0.0 for n in self.names}
        self.log = {"auctions": [], "respawns": [], "delist": []}

    def due(self, step):
        return step > 0 and step % self.every == 0

    def settle(self, step, G, extra=None):
        """G: {slot: royalty}. Returns a list of (action, slot_index, slot_name) with action in {'respawn', 'delist'}."""
        rec = {"step": step, "listed": [self.listed[n] for n in self.names], "G": dict(G)}
        if extra:
            rec.update(extra)
        actions = []
        for k, n in enumerate(self.names):
            if not self.listed[n]:
                continue
            g = float(G[n])
            self.earned[n] += g
            self.fees_paid[n] += self.fee
            self.ema[n] = g if self.ema[n] is None else self.alpha * self.ema[n] + (1 - self.alpha) * g
            if step - self.born[n] >= self.grace and self.ema[n] < self.fee:
                if k < self.k_learn:
                    actions.append(("respawn", k, n))
                    self.log["respawns"].append({"slot": n, "step": step, "ema": self.ema[n]})
                    self.born[n] = step
                    self.ema[n] = None
                else:
                    actions.append(("delist", k, n))
                    self.listed[n] = False
                    self.log["delist"].append({"slot": n, "step": step, "ema": self.ema[n]})
        rec["ema"] = dict(self.ema)
        self.log["auctions"].append(rec)
        return actions

    def books(self):
        return {"fee": self.fee, "earned": self.earned, "fees_paid": self.fees_paid,
                "fee_pot": sum(self.fees_paid.values()), "listed": self.listed}


if __name__ == "__main__":
    # self-test: a slot at 2x fee survives, a slot at 0 is de-listed after grace, a learned slot at 0 respawns
    a = Auction(["l0", "planted", "random"], k_learn=1, fee=0.005, grace=1000, every=500)
    acts = []
    for s in range(500, 3001, 500):
        acts += a.settle(s, {"l0": 0.0, "planted": 0.01, "random": 0.0})
    assert ("delist", 2, "random") in acts and a.listed["planted"] and not a.listed["random"]
    assert sum(1 for x in acts if x[0] == "respawn") >= 1
    print("auction.py self-test ok", a.log["respawns"], a.log["delist"])
