"""Tables for RESULTS.md from results/*-summary.json (validation) and results/test-*-final.json (TEST).

  python3 build_tables.py > /tmp/wm2/tables.md
Prints: the validation loop table per task (KL x1e3 per N, outcome dLL vs rbtd8, S_out, eligibility), compute per
board at every N (to the best checkpoint and in total, all phases), the WM diagnostics, and the arm-4 (flagged) rows.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("WM2_RESULTS", os.path.join(HERE, "results"))
NS = ["100", "300", "1000", "3000", "10000"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, k=2):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{k}f}"


def loop_table(task):
    s = load(os.path.join(R, f"{task}-val-summary.json"))
    if not s:
        return f"(no {task} validation summary)"
    summ = s["summary"]
    order = sorted(summ, key=lambda t: (summ[t]["S_out_partial"] is None, summ[t]["S_out_partial"] if summ[t]["S_out_partial"] is not None else 0))
    L = ["| tag | eligible | KL x1e3 at N=100 / 300 / 1k / 3k / 10k (ratio vs rbtd8) | dLL vs rbtd8 x1e4 at the 5 N | S_out |", "|---|---|---|---|---|"]
    for t in order:
        e = summ[t]
        kl, dl = [], []
        for n in NS:
            pn = e["per_n"].get(n)
            if not pn:
                kl.append("-")
                dl.append("-")
                continue
            r = pn.get("klratio_vs_rb")
            kl.append(f"{1e3 * pn['kl_mean']:.2f}" + (f" ({r:.2f})" if r and t != "B-rbtd8" else ""))
            d = pn.get("dll_vs_rb")
            dl.append(f"{1e4 * d:+.1f}" if d is not None else "-")
        why = "yes" if e["eligible"] else ("FLAGGED (arm 4)" if e["flag"] else ("not L0" if not e["l0"] else ("baseline" if t.startswith("B-") else "partial")))
        sc = e["S_out"] if e["S_out"] is not None else e["S_out_partial"]
        L.append(f"| {t} | {why} | " + " / ".join(kl) + " | " + " / ".join(dl) + f" | {f(sc, 3)}{'' if e['S_out'] is not None else '*'} |")
    L.append("")
    L.append("Champion by the pre-registered rule: **" + str(s.get("champion_by_S_out")) + "**. Gap k0 - rbtd8 in outcome log-loss x1e4 (the S_out scale): "
             + ", ".join(f"N={n}: {f(1e4 * g, 1) if g else '-'}" for n, g in s["gap_k0_minus_rb_logloss"].items()) + ".")
    return "\n".join(L)


def compute_table(task, tags):
    s = load(os.path.join(R, f"{task}-val-summary.json"))
    if not s:
        return ""
    summ = s["summary"]
    L = ["| tag | FLOPs per training board to the best checkpoint, all phases: N=100 / 300 / 1k / 3k / 10k | total FLOPs per board (incl. steps after the best) | x k0 at the same N (to best) |", "|---|---|---|---|"]
    k0 = summ.get("B-k0", {}).get("per_n", {})
    for t in tags:
        if t not in summ:
            continue
        e = summ[t]["per_n"]
        a = [f"{e[n]['flops_per_board_mean']:.2g}" if n in e else "-" for n in NS]
        b = [f"{e[n]['flops_per_board_total_mean']:.2g}" if n in e else "-" for n in NS]
        r = [f"{e[n]['flops_per_board_mean'] / k0[n]['flops_per_board_mean']:.0f}" if n in e and n in k0 and k0[n]['flops_per_board_mean'] else "-" for n in NS]
        L.append(f"| {t} | " + " / ".join(a) + " | " + " / ".join(b) + " | " + " / ".join(r) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    for task in ("t1", "t2"):
        print(f"### {task} validation loop\n")
        print(loop_table(task))
        print()
        wm = os.path.join(R, f"{task}-val-wm.md")
        if os.path.exists(wm):
            print(open(wm).read())
