"""E1 run wrapper: runs protocol.train.main unchanged, plus two instrumentation hooks and one
arm variant, all as monkeypatches from outside the harness (harness code is not modified).

  --e1-gating role|all  'all' = P-all: in belief and forecast passes the action adapter's LoRA delta
                        is also applied (gate 1 everywhere), so reward updates of the action adapter
                        reach belief/forecast outputs through the forward pass (hawks/E1.md 4.5).
                        Gradients: truth phases still step only their own adapter (harness _phase).
  R-phase probe         before and after every reward phase (phase "R"), the belief pass and the
                        forecast passes are re-run (no grad) on the current RL batch; logged per step:
                        e1_R_dbel_abs   mean |dE[q]| over cells (belief change caused by the reward step)
                        e1_R_dbel_note  signed dE[q] at the captain's-note cell (+ = toward agreeing)
                        e1_R_dbel_note_false  same, episodes whose note is false
                        e1_R_dfc_abs    mean |dE[P(success)]| over all candidates
                        e1_R_dfc_signed mean signed dE[P(success)] over all candidates (+ = optimism)
                        In P-role these must be exactly 0 (separate passes, own adapter only).
  calibration lag       around the truth phases: committed-plan forecast Brier (E[q] vs realised)
                        and signed error on the batch before the forecast phase (forecaster from the
                        previous step, data from the current policy) and after it:
                        e1_lag_brier_pre, e1_lag_brier_post, e1_lag_signed_pre, e1_lag_signed_post.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

E1_GATING = "role"
argv = []
it = iter(sys.argv[1:])
for a in it:
    if a == "--e1-gating":
        E1_GATING = next(it)
    else:
        argv.append(a)

from model import roles as R  # noqa: E402
from protocol import train as T  # noqa: E402
from protocol.rollout import belief_pass, forecast_pass, DEC  # noqa: E402
from protocol.losses import _seq  # noqa: E402

if E1_GATING == "all":
    def set_gates_all(self, role, B, L, self_pos=None):
        if self.cfg.fused:
            self._activate(self.adapters)
            for n in self.adapters:
                self._gates[n] = 1.0
            return
        act = [role] if role == "action" else [role, "action"]
        self._activate(act)
        for n in self.adapters:
            self._gates[n] = 1.0 if n in act else 0.0
    R.RoleModel.set_gates = set_gates_all
elif E1_GATING != "role":
    raise SystemExit(f"bad --e1-gating {E1_GATING}")

_LAST = {}
_orig_rollout = T.rollout


def _stash_rollout(*a, **k):
    recs = _orig_rollout(*a, **k)
    _LAST["recs"] = recs
    return recs


T.rollout = _stash_rollout


@torch.no_grad()
def _probe(tr, recs):
    m, V = tr.model, tr.V
    nc, L = tr.ecfg.n_cells, tr.ecfg.plan_len
    seqs = [_seq(r["b_ids"], r["b_q"]) for r in recs]
    cell_lp, _ = belief_pass(m, V, seqs, nc, [r["b_span"] for r in recs])
    q = (cell_lp.exp().float().cpu().numpy() * DEC).sum(-1)              # [n, nc]
    fseqs = [_seq(f["ids"], f["q"]) for r in recs for f in r["fc"]]
    s_lp, _, _ = forecast_pass(m, V, fseqs, L)
    fs = (s_lp.exp().float().cpu().numpy() * DEC).sum(-1)               # [n*K]
    return q, fs


_orig_phase = T.Trainer._phase


def _phase(self, name, groups_trainable, allowed, loss_fn, measure=None):
    recs = _LAST.get("recs")
    if name != "R" or recs is None or getattr(self, "phase_name", "rl") != "rl":
        return _orig_phase(self, name, groups_trainable, allowed, loss_fn, measure)
    q0, f0 = _probe(self, recs)
    st = _orig_phase(self, name, groups_trainable, allowed, loss_fn, measure)
    q1, f1 = _probe(self, recs)
    dq, df = q1 - q0, f1 - f0
    st["e1_R_dbel_abs"] = float(np.abs(dq).mean())
    st["e1_R_dfc_abs"] = float(np.abs(df).mean())
    st["e1_R_dfc_signed"] = float(df.mean())
    nn_ = [(i, r["note_cell"]) for i, r in enumerate(recs) if r["note_cell"] is not None]
    if nn_:
        st["e1_R_dbel_note"] = float(np.mean([dq[i, c] for i, c in nn_]))
        fl = [dq[i, c] for i, c in nn_ if recs[i]["occ"][c] == 0]
        if fl:
            st["e1_R_dbel_note_false"] = float(np.mean(fl))
    return st


T.Trainer._phase = _phase

_orig_truth = T.Trainer._truth_phases


@torch.no_grad()
def _committed_fc(tr, recs):
    L = tr.ecfg.plan_len
    seqs = [_seq(r["fc"][r["k"]]["ids"], r["fc"][r["k"]]["q"]) for r in recs]
    s_lp, _, _ = forecast_pass(tr.model, tr.V, seqs, L)
    q = (s_lp.exp().float().cpu().numpy() * DEC).sum(-1)
    y = np.array([float(r["outcomes"][r["k"]]["success"]) for r in recs])
    return float(((q - y) ** 2).mean()), float((q - y).mean())


def _truth_phases(self, recs, pc):
    if getattr(self, "phase_name", "rl") != "rl" or self.step == 0:
        return _orig_truth(self, recs, pc)
    b0, s0 = _committed_fc(self, recs)
    st = _orig_truth(self, recs, pc)
    b1, s1 = _committed_fc(self, recs)
    st.update(e1_lag_brier_pre=b0, e1_lag_brier_post=b1, e1_lag_signed_pre=s0, e1_lag_signed_post=s1)
    return st


T.Trainer._truth_phases = _truth_phases

if __name__ == "__main__":
    T.main(argv)
