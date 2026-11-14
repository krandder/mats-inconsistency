"""Pick the prompt variant by validation log loss (all 12 forms, missing imputed at 0.5).

Variants with more than 5% missing after retry are disqualified.  Writes
validation-selection.json next to the score files and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_MISSING = 0.05


def main() -> None:
    folder = Path(sys.argv[1])
    rows = []
    for f in sorted(folder.glob("scores-validation-v*.json")):
        if "thinking" in f.name or "pairs" in f.name:
            continue
        s = json.loads(f.read_text())
        a, o = s["all_forms"], s["observed"]
        rows.append({"variant": f.stem.removeprefix("scores-validation-"), "missing_rate": s["missing"]["missing_rate"],
                     "first_try_unparseable": s["missing"]["first_try_unparseable"],
                     "ll_all_imputed": a["imputed_0.5"]["log_loss"], "ll_all_excluded": a["missing_excluded"]["log_loss"],
                     "brier_all": a["imputed_0.5"]["brier"], "acc_all": a["imputed_0.5"]["accuracy"], "ece_all": a["imputed_0.5"]["ece"],
                     "ll_observed_imputed": o["imputed_0.5"]["log_loss"],
                     "consistency": s["consistency"]["missing_excluded"]["mean_squared_violation"],
                     "qualified": s["missing"]["missing_rate"] <= MAX_MISSING})
    ok = [r for r in rows if r["qualified"]]
    if not ok:
        raise SystemExit("ABORT: every variant exceeds 5% missing on validation")
    chosen = min(ok, key=lambda r: r["ll_all_imputed"])["variant"]
    (folder / "validation-selection.json").write_text(json.dumps({"rule": "min validation log loss, all 12 forms, missing imputed 0.5, missing <= 5%",
                                                                 "chosen": chosen, "variants": rows}, indent=1))
    print("| Variant | Missing | LL all (imp.) | LL all (excl.) | Brier | Acc | ECE | LL observed | Consistency |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mark = " **chosen**" if r["variant"] == chosen else ""
        print(f"| {r['variant']}{mark} | {r['missing_rate']:.2%} | {r['ll_all_imputed']:.4f} | {r['ll_all_excluded']:.4f} | {r['brier_all']:.4f} | "
              f"{r['acc_all']:.3f} | {r['ece_all']:.3f} | {r['ll_observed_imputed']:.4f} | {r['consistency']:.4f} |")
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
