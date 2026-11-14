"""Tests for scripts/summarize.py and scripts/matrix.sh --dry-run (make pytest)."""
import csv, importlib.util, io, json, math, os, subprocess, tempfile, unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("summarize", os.path.join(ROOT, "scripts", "summarize.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

HEAD = ("epoch,alive,wealth,wmax,ops,used,demands,births,deaths,volume,trader_pnl,brier_ppm,burned,"
        + ",".join(f"brier_{f}" for f in S.FAMS) + "," + ",".join(f"log_{f}" for f in S.FAMS)
        + ",coh_and_ppm,coh_or_ppm," + ",".join(f"sh_{s}" for s in S.SHADOWS)
        + ",gini_ppm,lineages,genotypes,active_ppm,cum_subsidy,cum_ops").split(",")


def write_run(d, n, seed, brier=lambda f, e: 250000, births=lambda e: 0, sh=lambda s, e: 0, conf=None):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEAD)
        for e in range(n):
            r = dict.fromkeys(HEAD, 0)
            r.update(epoch=e, births=births(e), coh_and_ppm=1000 * seed, coh_or_ppm=2000,
                     gini_ppm=500000, genotypes=100 + seed)
            for fam in S.FAMS:
                b = brier(fam, e); r[f"brier_{fam}"] = "nan" if b is None else b
            for s in S.SHADOWS:
                r[f"sh_{s}"] = sh(s, e)
            w.writerow([r[h] for h in HEAD])
    json.dump(conf or {"seed": seed, "logk": 8, "logq": 4}, open(os.path.join(d, "config.json"), "w"))
    open(os.path.join(d, "timing.csv"), "w").write("first_epoch,epochs,wall_s,instructions,instr_per_s\n0,10,1.5,0,0\n10,10,2.5,0,0\n")


def args(**kw):
    d = dict(window=50, takeoff=50, price=0.22, hold=50, slope=1000)
    d.update(kw)
    return type("A", (), d)


class Units(unittest.TestCase):
    def test_time_to_price(self):
        ep = list(range(100))
        br = [0.25] * 30 + [0.2] * 5 + [0.25] + [0.1] * 64  # priced from 36 on
        self.assertEqual(S.time_to_price(ep, br, 20, 0.22, 10), 16)
        self.assertTrue(math.isnan(S.time_to_price(ep, br, 20, 0.22, 65)))
        br[50] = math.nan  # nan breaks the streak
        self.assertEqual(S.time_to_price(ep, br, 20, 0.22, 20), 31)

    def test_takeoff_and_slope(self):
        self.assertEqual(S.takeoff([0, 1, 2, 3], [0, 49, 50, 99], 50), 2)
        self.assertTrue(math.isnan(S.takeoff([0, 1], [0, 1], 50)))
        self.assertAlmostEqual(S.slope([0, 1, 2, 3], [5, 7, 9, 11]), 2.0)

    def test_cell_names(self):
        self.assertEqual(S.cell_of("runs/m/k14-f12-s3-heldpi", {}), ("k14-f12-heldpi", 3))
        self.assertEqual(S.cell_of("runs/2026-09-27-calib-k12-f12-s2/", {}), ("k12-f12", 2))
        self.assertEqual(S.cell_of("runs/odd", {"logk": 9, "logq": 5, "seed": 4}), ("k9-f5", 4))


class Matrix(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as t:
            m = os.path.join(t, "m")
            for seed in (1, 2):
                # shadow c50 gains 65536 ticks (one share) per epoch; periodic Brier
                # 0.2 for seed 1, 0.3 for seed 2 over the final window
                write_run(os.path.join(m, f"k8-f4-s{seed}"), 200, seed,
                          brier=lambda f, e, s=seed: (200000 if s == 1 else 300000) if f == "periodic" else 250000,
                          births=lambda e, s=seed: 60 if e >= 10 * s else 0,
                          sh=lambda sh, e: 65536 * e if sh == "c50" else 0)
            conf = {"seed": 1, "logk": 8, "logq": 4, "held_out_family": "pi", "release": "100"}
            write_run(os.path.join(m, "k8-f4-s1-heldpi"), 200, 1, conf=conf,
                      brier=lambda f, e: None if (f == "pi" and e < 100) else (100000 if f == "pi" and e >= 130 else 250000))
            with redirect_stdout(io.StringIO()):
                S.main([m])
            rows = {r["cell"]: r for r in csv.DictReader(open(os.path.join(m, "summary.csv")))}
            base, held = rows["k8-f4"], rows["k8-f4-heldpi"]
            self.assertEqual(base["n"], "2")
            self.assertAlmostEqual(float(base["brier_periodic"]), 0.25)
            self.assertAlmostEqual(float(base["brier_periodic_sd"]), math.sqrt(0.005))
            self.assertAlmostEqual(float(base["slope_c50"]), 1.0)
            self.assertAlmostEqual(float(base["slope_mom"]), 0.0)
            self.assertAlmostEqual(float(base["takeoff"]), 15.0)
            self.assertAlmostEqual(float(base["coh_and"]), 0.0015)
            self.assertAlmostEqual(float(base["wall_s"]), 4.0)
            self.assertEqual(base["ttp"], "")
            self.assertAlmostEqual(float(held["ttp"]), 30.0)
            self.assertEqual(held["takeoff_n"], "0")
            md = open(os.path.join(m, "summary.md")).read()
            self.assertIn("| k8-f4 | 2 | 0.250±0.071 |", md)

    def test_matrix_dry_run(self):
        out = subprocess.run([os.path.join(ROOT, "scripts", "matrix.sh"), "--dry-run",
                              os.path.join(ROOT, "matrices", "phase3.json"), "--name", "dry-test"],
                             capture_output=True, text=True, check=True).stdout
        cmds = [l for l in out.splitlines() if l.startswith("scripts/run.sh")]
        self.assertEqual(len(cmds), 27 + 9)
        self.assertIn("scripts/run.sh --b 64 --free-income 0 --seed 3 --K 2^16 --F 2^12 --epochs 5000 "
                      "--chunk 100 --out runs/dry-test/k16-f12-s3", cmds)
        self.assertTrue(any("--held-out-family pi --release 2000" in c and c.endswith("k14-f12-s2-heldpi") for c in cmds))
        self.assertTrue(any("--visible-radius 1" in c and c.endswith("k14-f12-s1-r1") for c in cmds))
        self.assertEqual(out.count("# first 50 epochs at free income 4096"), 3)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "runs", "dry-test")))


if __name__ == "__main__":
    unittest.main()
