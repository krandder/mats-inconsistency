"""E2 leak arms without touching harness code: a RoleModel subclass with two extra routing modes.

leak = "none": harness behaviour (A0 fused / A1 protected / A4 dose).
leak = "P"   : hawks/E2.md channel P, the old gating="all": every adapter is computed with gate 1
               at every position of every pass (belief/forecast passes run the action adapter too).
leak = "K"   : hawks/E2.md channel K, the old gating="role" without the content mask: the goal text
               is put back into belief/forecast inputs (--p-goal-mode visible) and its positions are
               routed to the ACTION adapter (old ROLE_OF_TYPE: GOAL -> action), so belief/forecast
               claim tokens attend to K/V computed by the action adapter. Elsewhere role routing.
The reward phase still steps only the action adapter in A2/A3 (alpha 0); the trainer's phase masks
are unchanged and still assert it.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.environ.get("LIR_HARNESS", os.path.join(HERE, "..", "..", "harness"))
if H not in sys.path:
    sys.path.insert(0, H)

import torch  # noqa: E402
from model.roles import RoleModel  # noqa: E402

LEAK = os.environ.get("E2_LEAK", "none")


class LeakRoleModel(RoleModel):
    leak = LEAK

    def __init__(self, cfg, tokenizer=None):
        super().__init__(cfg, tokenizer)
        self._cur_ids = None
        self._goal_prefixes = []
        from env.shiphunt import PRESETS, ShipHunt
        from dataclasses import replace
        seen = set()
        for p in PRESETS.values():
            for g in ("hit", "avoid"):
                t = ShipHunt(replace(p, goal=g, note=False), 0).goal_text()
                ids = tuple(self.tok.encode("\nGoal: " + t + ".", add_special_tokens=False))
                if ids not in seen:
                    seen.add(ids)
                    self._goal_prefixes.append(torch.tensor(ids))

    def logits(self, role, ids, attn, self_pos=None):
        self._cur_ids = ids
        try:
            return super().logits(role, ids, attn, self_pos)
        finally:
            self._cur_ids = None

    def goal_mask(self, ids):
        B, L = ids.shape
        m = torch.zeros(B, L, dtype=torch.bool)
        for b in range(B):
            for pre in self._goal_prefixes:
                g = len(pre)
                if g <= L and torch.equal(ids[b, :g].cpu(), pre):
                    m[b, :g] = True
                    break
        return m

    def set_gates(self, role, B, L, self_pos=None):
        if self.cfg.fused or self.leak == "none":
            return super().set_gates(role, B, L, self_pos)
        if self.leak == "P":
            self._activate(self.adapters)
            for n in self.adapters:
                self._gates[n] = 1.0
            return
        if self.leak == "K" and role in ("belief", "forecast") and self._cur_ids is not None:
            gm = self.goal_mask(self._cur_ids)
            if self_pos is not None and role == "belief" and self.cfg.self_stack_action:
                gm = gm | self_pos.cpu().bool()
            if not bool(gm.any()):
                return super().set_gates(role, B, L, self_pos)
            self._activate([role, "action"])
            for n in self.adapters:
                self._gates[n] = 0.0
            self._gates[role] = 1.0
            self._gates["action"] = gm.float()[..., None].to(self.cfg.device)
            return
        return super().set_gates(role, B, L, self_pos)


def install(leak: str):
    """Make protocol.train build LeakRoleModel with the given leak mode."""
    LeakRoleModel.leak = leak
    import protocol.train as T
    T.RoleModel = LeakRoleModel
    return T
