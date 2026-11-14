"""Item 1a controls on the P12 weighted mazes (no learning), same solver and metrics as evaluate.py's CostNet path.

  oracle : the true hidden cell costs -> exact solver (solver/pipeline validity gate G1: exact >= 99% on every set)
  uniform: every cell cost 1 -> exact solver (shortest open path by hops; the no-learning floor)

  python controls.py [--sets ...]   -> $FU_EVAL_DIR/control_{oracle,uniform}.npz/.json
"""
import argparse, json, os, time
import numpy as np
import torch
import flow, nets
import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=E.DEFAULT_SETS); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cpu"
    for kind in ("oracle", "uniform"):
        acc = E.Acc(); t0 = time.time()
        for s in a.sets.split(","):
            ts, L, opt = E.load_set(s, dev); n = ts["open"].shape[1]; N = len(L)
            for lo, hi in E.chunks(N, n, False):
                o = ts["open"][lo:hi]; S = ts["S"][lo:hi]; G = ts["G"][lo:hi]
                c = ts["cost"][lo:hi].float() if kind == "oracle" else torch.ones_like(ts["cost"][lo:hi]).float()
                on, ok = nets.shortest_path(nets.cell_to_edge_cost(c, o), o, S, G)
                endm = flow.endpoints(S, G, hi - lo, n, dev)
                exb, vpb, _, f1b, crb = E.maze_metrics(on, ts["path"][lo:hi], endm, flow.edge_cost_tensor(ts["cost"][lo:hi], o),
                                                       torch.tensor(opt[lo:hi], device=dev))
                acc.put(f"{s}/exact", lo, hi, N, exb.cpu().numpy(), np.int8); acc.put(f"{s}/valid", lo, hi, N, vpb.cpu().numpy(), np.int8)
                acc.put(f"{s}/f1", lo, hi, N, f1b.cpu().numpy()); acc.put(f"{s}/costratio", lo, hi, N, crb.cpu().numpy())
            print(kind, s, "exact", float(np.mean(acc.d[f"{s}/exact"])), flush=True)
        os.makedirs(E.EVDIR, exist_ok=True)
        np.savez_compressed(os.path.join(E.EVDIR, f"control_{kind}.npz"), **acc.d)
        json.dump({"name": f"control_{kind}", "total_s": round(time.time() - t0, 1),
                   "means": {k: float(np.nanmean(v)) for k, v in acc.d.items()}},
                  open(os.path.join(E.EVDIR, f"control_{kind}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
