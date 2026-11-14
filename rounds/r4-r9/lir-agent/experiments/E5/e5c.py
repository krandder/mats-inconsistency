"""E5c: does goal-blindness cost perception when relevance has work to do? (hawks/E5.md section 4)

Two-board ShipHunt (two independent `medium` boards, 5x4, ships 3+2, probe flip eps 0.15), >= 48
noisy probe lines per board drawn WITH replacement and interleaved at random, so ~100 evidence
lines precede the report. The goal names a board ("find a ship on board b"), drawn from its own RNG
stream, independent of both boards. A belief adapter (LoRA on frozen Qwen3) is trained on revealed
truth only (expected Brier of the decile claim, harness protocol.losses.exp_brier). Arms differ only
in the belief-pass input:
  V  goal visible (true board named) + report both boards
  Q  goal deleted; the board-naming QUERY is chosen by the action adapter (which sees the goal);
     report only the queried board. Training uses one query sequence per board (same labels as V/N/S).
  N  goal deleted, no query; report both boards
  S  goal shuffled: the goal line names the WRONG board; report both boards
Metric: on the goal board's cells, Brier and ECE of E[q] vs the exact Bayes posterior (and vs truth),
mean over a fixed eval set; curves at --eval-at steps. Format control: V and S checkpoints are also
evaluated with the other goal text (true vs wrong board) -> KL and Brier difference.
Harness code is imported, not modified.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import replace
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.environ.get("LIR_HARNESS", os.path.join(HERE, "..", "..", "harness"))
sys.path.insert(0, H)
from env.shiphunt import ShipHunt, PRESETS
from model.roles import RoleModel, ModelConfig
from protocol.render import Vocab, Seq, LETTERS
from protocol.rollout import pad_batch, DEC
from protocol.losses import exp_brier
from metrics.calib import ece, brier

CFG = replace(PRESETS["medium"], n_probes=0)
GOAL_TXT = "find a ship on board {b}: fire at most 3 shots, success = 1 hit(s); reward = 1 on success minus 0.1 per shot"


class TwoBoard:
    def __init__(self, seed: int, n_probes: int = 48):
        self.seed = seed
        self.boards = [ShipHunt(CFG, seed * 2 + 0 + 7), ShipHunt(CFG, seed * 2 + 1 + 7)]
        rng = np.random.default_rng([seed, 31337])
        lines = []
        for bi, b in enumerate(self.boards):
            cells = rng.integers(CFG.n_cells, size=n_probes)
            b.probes = []
            for c in cells:
                h = bool(b.occ[c]) ^ bool(rng.random() < CFG.eps)
                b.probes.append((int(c), h))
                lines.append(f"probe {bi + 1}{LETTERS[c]}: {'hit' if h else 'miss'}")
        order = rng.permutation(len(lines))
        self.lines = [lines[i] for i in order]
        self.goal = 1 if np.random.default_rng([seed, 104729]).random() < 0.5 else 2   # independent stream
        self.bayes = [b.bayes_cell_probs() for b in self.boards]
        self.occ = [b.occ.astype(float) for b in self.boards]

    def grid(self):
        b = self.boards[0]
        return b.grid_text()


def context(s: Seq, e: TwoBoard):
    s.text(f"Two grids {CFG.W}x{CFG.H} (board 1 and board 2), cells: {e.grid()}. Ships on each board: "
           + ", ".join(str(x) for x in CFG.ships) + ".\nEvidence: " + "; ".join(e.lines) + ".")


CELLFMT = "prefixed"   # prefixed: " 1C=" (same token pattern as the evidence "probe 1C:"); plain: " C="


def report(s: Seq, V: Vocab, board: int, tag: str):
    s.text(f" board {board}:")
    for c in range(CFG.n_cells):
        if CELLFMT == "prefixed":
            s.text(f" {board}{LETTERS[c]}").tok(V.eq)
        else:
            s.tok(V.cells[c]).tok(V.eq)
        s.q[f"{tag}{board}_{c}"] = len(s.ids) - 1


def belief_seq(V, e: TwoBoard, arm: str, query_board: int = None, goal_board: int = None):
    s = Seq(V)
    if arm in ("V", "S"):
        s.text("Goal: " + GOAL_TXT.format(b=goal_board) + ".\n")
    context(s, e)
    s.text("\nReport P(ship on cell) in tenths,")
    if arm == "Q":
        report(s, V, query_board, "c")
    else:
        report(s, V, 1, "c")
        s.text(";")
        report(s, V, 2, "c")
    return s


def action_seq(V, e: TwoBoard):
    s = Seq(V)
    s.text("Goal: " + GOAL_TXT.format(b=e.goal) + ".\n")
    s.text(f"Two grids {CFG.W}x{CFG.H} (board 1 and board 2), cells: {e.grid()}.")
    s.query("board", "\nQuery board", V.colon)
    return s


def belief_lp(model, V, seqs):
    """returns list of dict key -> logp[10] (torch)"""
    ids, attn = pad_batch([s.ids for s in seqs], V.pad)
    lg = model.logits("belief", ids, attn)
    out = []
    for b, s in enumerate(seqs):
        keys = list(s.q.keys())
        pos = [s.q[k] for k in keys]
        x = torch.stack([torch.log_softmax(lg[b, p, V.digits], -1) for p in pos])
        out.append((keys, x))
    return out


def train_targets(e: TwoBoard, keys):
    y = []
    for k in keys:
        bd, c = k[1:].split("_")
        y.append(e.occ[int(bd) - 1][int(c)])
    return torch.tensor(y, dtype=torch.float32)


def seqs_for(V, e, arm, qb=None, train=False):
    if arm == "MIX":   # shared pre-training: one format per episode, rotating over V/Q/N/S by episode seed
        return seqs_for(V, e, "VQNS"[e.seed % 4], qb, train)
    if arm == "Q":
        if train:
            return [belief_seq(V, e, "Q", query_board=1), belief_seq(V, e, "Q", query_board=2)]
        return [belief_seq(V, e, "Q", query_board=qb)]
    if arm == "V":
        return [belief_seq(V, e, "V", goal_board=e.goal)]
    if arm == "S":
        return [belief_seq(V, e, "S", goal_board=3 - e.goal)]
    return [belief_seq(V, e, "N")]


@torch.no_grad()
def choose_board(model, V, envs):
    seqs = [action_seq(V, e) for e in envs]
    ids, attn = pad_batch([s.ids for s in seqs], V.pad)
    lg = model.logits("action", ids, attn)
    out = []
    for b, s in enumerate(seqs):
        lp = torch.log_softmax(lg[b, s.q["board"], [V.digits[1], V.digits[2]]], -1)
        out.append(1 + int(lp.argmax()))
    return out


@torch.no_grad()
def evaluate(model, V, envs, arm, mb, qbs=None):
    """Returns per-episode arrays of E[q] and full decile dists for the goal board and the other board."""
    G, O, Gd = [], [], []
    for i in range(0, len(envs), mb):
        ch = envs[i:i + mb]
        seqs = []
        for j, e in enumerate(ch):
            seqs += seqs_for(V, e, arm, qb=(qbs[i + j] if qbs else None))
        res = belief_lp(model, V, seqs)
        for e, (keys, lp) in zip(ch, res):
            d = dict(zip(keys, lp.exp().float().cpu().numpy()))
            # a wrong query leaves the goal board unreported: score it as the uniform prior P(ship)=5/20
            prior = np.zeros(10); prior[2] = 0.5; prior[3] = 0.5      # E[q] = 0.25 = base rate
            g = np.stack([d.get(f"c{e.goal}_{c}", prior) for c in range(CFG.n_cells)])
            o = np.stack([d.get(f"c{3 - e.goal}_{c}", np.full(10, np.nan)) for c in range(CFG.n_cells)])
            Gd.append(g); G.append(g @ DEC); O.append(o @ DEC)
    return np.array(G), np.array(O), np.array(Gd)


def metrics(envs, G, O, qbs=None):
    pb = np.array([e.bayes[e.goal - 1] for e in envs])
    y = np.array([e.occ[e.goal - 1] for e in envs])
    out = {}
    if qbs is not None:   # query arm: cells reported only for the queried board
        ok = np.array([q == e.goal for q, e in zip(qbs, envs)])
        out["query_correct"] = float(ok.mean())
    m = ~np.isnan(G)
    p, pbm, ym = G[m], pb[m], y[m]
    out.update(brier_vs_bayes=float(np.mean((p - pbm) ** 2)), mae_vs_bayes=float(np.mean(np.abs(p - pbm))),
               brier_vs_truth=brier(p, ym), ece_vs_truth=ece(p, ym), ece_vs_bayes=ece(p, pbm),
               bayes_floor_brier=brier(pbm, ym), mean_p=float(p.mean()), n_cells_scored=int(m.sum()),
               coverage=float(m.mean()))
    out["per_episode_brier_vs_bayes"] = [float(np.nanmean((G[i] - pb[i]) ** 2)) for i in range(len(envs))]
    if not np.all(np.isnan(O)):
        po = np.array([e.bayes[2 - e.goal] for e in envs])
        out["other_board_brier_vs_bayes"] = float(np.nanmean((O - po) ** 2))
    return out


def kl_rows(P, Q):
    P = np.clip(P, 1e-9, 1); Q = np.clip(Q, 1e-9, 1)
    return float(np.nanmean((P * (np.log(P) - np.log(Q))).sum(-1)))


def train_belief(model, V, opt_b, a, arm, steps, seed_base, log, eval_at=(), do_eval=None):
    model.set_trainable(["belief"])
    for st in range(1, steps + 1):
        envs = [TwoBoard(seed_base + st * a.batch + j, a.n_probes) for j in range(a.batch)]
        seqs, tg = [], []
        for e in envs:
            for s in seqs_for(V, e, arm, train=True):
                seqs.append(s); tg.append(e)
        opt_b.zero_grad()
        tot = 0.0
        for i in range(0, len(seqs), a.mb):
            res = belief_lp(model, V, seqs[i:i + a.mb])
            ls = []
            for (keys, lp), e in zip(res, tg[i:i + a.mb]):
                y = train_targets(e, keys).to(lp.device)
                ls.append(exp_brier(lp.float(), y).mean())
            loss = torch.stack(ls).sum() / len(seqs)
            loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.group_params("belief"), 1.0)
        opt_b.step()
        log.write(json.dumps({"phase": "belief", "arm": arm, "step": st, "exp_brier": tot}) + "\n"); log.flush()
        if st in eval_at and do_eval is not None:
            model.set_trainable([])
            do_eval(st)
            model.set_trainable(["belief"])
    model.set_trainable([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=["V", "Q", "N", "S"])
    ap.add_argument("--arms", default=None, help="comma list: run these arms in one process, each from the same --pretrain-mix state")
    ap.add_argument("--pretrain-mix", type=int, default=0, help="shared truth-only pre-training steps on a V/Q/N/S format mixture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-seed", type=int, default=None, help="LoRA init + mixture-data seed (default = --seed)")
    ap.add_argument("--pretrain-until", type=float, default=0.0, help="extend mixture pre-training in 100-step blocks while the mean exp_brier of the last 20 steps exceeds this (arm-agnostic)")
    ap.add_argument("--pretrain-max", type=int, default=1200)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-probes", type=int, default=48)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--eval-at", default="0,50,100,150")
    ap.add_argument("--action-steps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cellfmt", default="prefixed", choices=["prefixed", "plain"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global CELLFMT
    CELLFMT = a.cellfmt
    arms = a.arms.split(",") if a.arms else [a.arm]
    torch.set_num_threads(a.threads)
    ps = a.seed if a.pretrain_seed is None else a.pretrain_seed
    torch.manual_seed(ps)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = RoleModel(ModelConfig(name=a.model, arm_adapters=("belief", "action"), dtype=a.dtype,
                                  device=a.device, seed=ps, self_stack_action=False))
    V = Vocab(model.tok)
    base0 = model.base_checksum()
    eval_envs = [TwoBoard(90_000_000 + i, a.n_probes) for i in range(a.eval_n)]
    eval_at = sorted(int(x) for x in a.eval_at.split(","))
    log = open(os.path.join(a.out, "train.jsonl"), "w")
    qbs = None
    # action pass (Q arm only): learn to name the goal board from the goal text (relevance = which query)
    if "Q" in arms:
        opt_a = torch.optim.AdamW(model.group_params("action"), lr=a.lr, weight_decay=0.0)
        model.set_trainable(["action"])
        for st in range(a.action_steps):
            envs = [TwoBoard(5_000_000 * (a.seed + 1) + st * 32 + j, a.n_probes) for j in range(32)]
            seqs = [action_seq(V, e) for e in envs]
            ids, attn = pad_batch([s.ids for s in seqs], V.pad)
            lg = model.logits("action", ids, attn)
            lp = torch.stack([torch.log_softmax(lg[b, s.q["board"], [V.digits[1], V.digits[2]]], -1)
                              for b, s in enumerate(seqs)])
            tgt = torch.tensor([e.goal - 1 for e in envs], device=lp.device)
            loss = torch.nn.functional.nll_loss(lp, tgt)
            opt_a.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.group_params("action"), 1.0); opt_a.step()
            log.write(json.dumps({"phase": "action", "step": st, "nll": loss.item()}) + "\n")
        model.set_trainable([])
        qbs = choose_board(model, V, eval_envs)
    if a.pretrain_mix:
        opt_p = torch.optim.AdamW(model.group_params("belief"), lr=a.lr, weight_decay=0.0)
        train_belief(model, V, opt_p, a, "MIX", a.pretrain_mix, 7_000_000 * (ps + 1), log)
        done = a.pretrain_mix
        hist = []
        while a.pretrain_until > 0 and done < a.pretrain_max:
            log.flush()
            hist = [json.loads(l)["exp_brier"] for l in open(os.path.join(a.out, "train.jsonl")) if '"MIX"' in l][-20:]
            if np.mean(hist) <= a.pretrain_until:
                break
            train_belief(model, V, opt_p, a, "MIX", 100, 7_000_000 * (ps + 1) + done * a.batch, log)
            done += 100
        json.dump({"pretrain_seed": ps, "pretrain_steps": done}, open(os.path.join(a.out, "pretrain.json"), "w"))
    torch.manual_seed(a.seed)
    state0 = model.adapter_state("belief")
    for arm in arms:
        model.load_adapter_state("belief", state0)
        out = a.out if len(arms) == 1 else os.path.join(a.out, f"{arm}-s{a.seed}")
        os.makedirs(out, exist_ok=True)
        evals = []

        def do_eval(step):
            G, O, Gd = evaluate(model, V, eval_envs, arm, a.mb, qbs if arm == "Q" else None)
            r = {"step": step, **metrics(eval_envs, G, O, qbs if arm == "Q" else None)}
            if arm in ("V", "S"):   # format control: same weights, the other goal text
                other = "S" if arm == "V" else "V"
                G2, O2, Gd2 = evaluate(model, V, eval_envs, other, a.mb)
                m2 = metrics(eval_envs, G2, O2)
                r["swapgoal_kl"] = kl_rows(Gd, Gd2)
                r["swapgoal_brier_vs_bayes"] = m2["brier_vs_bayes"]
                r["swapgoal_brier_vs_truth"] = m2["brier_vs_truth"]
            r["time"] = time.time() - t0
            evals.append(r)
            with open(os.path.join(out, "eval.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
            print(arm, json.dumps({k: v for k, v in r.items() if not k.startswith("per_")}), flush=True)

        if 0 in eval_at:
            do_eval(0)
        opt_b = torch.optim.AdamW(model.group_params("belief"), lr=a.lr, weight_decay=0.0)
        train_belief(model, V, opt_b, a, arm, a.steps, 1_000_000 * (a.seed + 1), log, eval_at, do_eval)
        assert model.base_checksum() == base0, "base moved"
        summ = {"arm": arm, "seed": a.seed, "model": a.model, "steps": a.steps, "pretrain_mix": a.pretrain_mix,
                "batch": a.batch, "n_probes_per_board": a.n_probes, "cellfmt": a.cellfmt, "lr": a.lr,
                "eval_n": a.eval_n, "secs": time.time() - t0, "base_checksum_unchanged": True,
                "final": {k: v for k, v in evals[-1].items() if not k.startswith("per_")},
                "seq_len_example": len(seqs_for(V, eval_envs[0], arm, qb=1)[0].ids)}
        json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
