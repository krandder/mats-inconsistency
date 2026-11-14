"""No rentals: broker logic against a fake gcloud. Run: python3 infra/tests/test_infra.py"""
import os, pathlib, shutil, subprocess, tempfile, unittest

INFRA = pathlib.Path(__file__).resolve().parents[1]
FAKE = r'''
gcloud() {
  echo "$*" >> "$G1_ROOT/calls"
  if [[ "$*" == *"instances list"* ]]; then cat "$G1_ROOT/fake-vms" 2>/dev/null; return 0; fi
  if [[ "$*" == *"disks list"* ]]; then return 0; fi
  if [[ "$*" == *"instances create"* ]]; then touch "$G1_ROOT/created"; return 0; fi
  if [[ "$*" == *"instances delete"* ]]; then echo "$*" >> "$G1_ROOT/deletes"; return 0; fi
  if [[ "$*" == *"describe"* ]]; then echo [ip-removed]; return 0; fi
  if [[ "$*" == *"operations list"* ]]; then echo '[]'; return 0; fi
  return 0
}; export -f gcloud
post_status() { :; }
'''


class Broker(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(dir="/tmp")
        self.g1 = pathlib.Path(self.t.name) / "study" / "G1"
        shutil.copytree(INFRA, self.g1 / "infra", ignore=shutil.ignore_patterns("state", "tests", "__pycache__"))
        with (self.g1 / "infra/lib.sh").open("a") as f:
            f.write("\npost_status() { :; }\n")
        (self.g1.parent / "M7").mkdir()
        self.src = pathlib.Path(self.t.name) / "src"; self.src.mkdir()
        (self.src / "run.sh").write_text("echo hi > $OUT/x.csv\n")

    def tearDown(self):
        self.t.cleanup()

    def sh(self, s):
        env = dict(os.environ, G1_ROOT=str(self.g1))
        return subprocess.run(["bash", "-c", FAKE + s], cwd=self.g1, text=True, capture_output=True, env=env)

    def test_spend_prefers_audit_then_delete_then_reserve(self):
        (self.g1 / "GPU-LOG.md").write_text(
            "- t CREATE vm=a ws=M7 reserve_usd=5.00\n- t DELETE vm=a cost_usd=1.00\n- t COST-AUDIT vm=a ws=M7 cost_usd=0.50\n"
            "- t CREATE vm=b ws=M6 reserve_usd=2.00\n- t DELETE vm=b cost_usd=0.25\n- t PENDING-CREATE vm=c ws=M7 reserve_usd=3.00\n")
        r = self.sh("source infra/lib.sh; spend_usd; spend_usd M7; spend_usd M6")
        self.assertEqual(r.stdout, "3.75\n3.50\n0.25\n", r.stderr)

    def test_reserve_and_max_run(self):
        r = self.sh("source infra/lib.sh; max_run_min 1; reserve_usd 1; max_run_min 9; max_run_min 0.01")
        self.assertEqual(r.stdout.split(), ["80", "1.16", "360", "21"])

    def test_up_refuses_over_global_cap_without_creating(self):
        (self.g1 / "GPU-LOG.md").write_text("- t CREATE vm=a ws=M7 reserve_usd=44.50\n")
        r = self.sh("bash infra/gpu-up.sh --job m7-x --ws M7 --hours 1 --no-wait")
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("REFUSE", r.stderr)
        self.assertFalse((self.g1 / "created").exists())

    def test_up_refuses_over_ws_cap(self):
        r = self.sh("bash infra/gpu-up.sh --job m1-x --ws M1 --hours 4 --no-wait")
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("M1 cap", r.stderr)

    def test_up_waits_when_two_vms(self):
        (self.g1 / "fake-vms").write_text(
            "tn-gpu-a-1\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n"
            "tn-gpu-b-1\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n")
        r = self.sh("bash infra/gpu-up.sh --job m7-x --ws M7 --hours 1 --no-wait")
        self.assertEqual(r.returncode, 75, r.stderr); self.assertFalse((self.g1 / "created").exists())

    def test_up_creates_with_labels_and_ttl(self):
        r = self.sh("bash infra/gpu-up.sh --job m7-x --ws M7 --hours 1 --no-wait")
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = (self.g1 / "calls").read_text()
        self.assertIn("--max-run-duration=80m", calls); self.assertIn("traders-net=1,tn-job=m7-x,tn-ws=m7", calls)
        self.assertIn("--provisioning-model=SPOT", calls)
        self.assertIn("reserve_usd=1.16", (self.g1 / "GPU-LOG.md").read_text())

    def test_down_refuses_foreign_vm(self):
        r = self.sh("bash infra/gpu-down.sh --only soup-gpu-wealth-net-1")
        self.assertEqual(r.returncode, 2); self.assertFalse((self.g1 / "deletes").exists())

    def test_submit_and_runner_refuses_over_cap(self):
        r = self.sh(f"bash infra/submit.sh --ws M9 --name big --src {self.src} --hours 5")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "m9-big")
        r = self.sh("bash infra/queue-runner.sh --once")
        self.assertTrue((self.g1 / "queue/refused/m9-big/REFUSED.txt").exists(), r.stdout + r.stderr)

    def test_submit_rejects_results_outside_own_ws(self):
        r = self.sh(f"bash infra/submit.sh --ws M7 --name a --src {self.src} --hours 1 --results-to {self.g1.parent}/M6/x")
        self.assertEqual(r.returncode, 2)

    def test_runner_dry_run_starts_two(self):
        for n in ("a", "b", "c"):
            self.sh(f"bash infra/submit.sh --ws M7 --name {n} --src {self.src} --hours 1")
        r = self.sh("bash infra/queue-runner.sh --dry-run")
        self.assertEqual(r.stdout.count(" start "), 2, r.stdout); self.assertIn("waiting: m7-c", r.stdout)

    def test_watchdog_scoped(self):
        (self.g1 / "fake-vms").write_text("tn-gpu-old-1\tus-central1-a\t2020-01-01T00:00:00Z\tg2-standard-8\t1\tRUNNING\n")
        r = self.sh("bash infra/gpu-watchdog.sh --dry-run")
        self.assertIn("would delete tn-gpu-old-1", r.stdout, r.stderr)
        self.assertIn("labels.traders-net:* OR name ~ ^tn-gpu-", (self.g1 / "calls").read_text())

    def test_zero_vm_check(self):
        r = self.sh("bash infra/zero-vm-check.sh")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("ZERO", r.stdout)
        (self.g1 / "fake-vms").write_text("tn-gpu-x-1\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n")
        self.assertEqual(self.sh("bash infra/zero-vm-check.sh").returncode, 1)

    def test_cost_report(self):
        (self.g1 / "GPU-LOG.md").write_text("- 2026-09-27T22:00:00Z CREATE vm=tn-gpu-g1-smoke-1 ws=G1 job=g1-smoke reserve_usd=0.44\n"
                                            "- 2026-09-27T22:10:00Z COST-AUDIT vm=tn-gpu-g1-smoke-1 ws=G1 cost_usd=0.120000 basis=insert-to-delete-receipt\n")
        r = subprocess.run(["python3", str(self.g1 / "infra/cost-report.py"), "--no-live"], text=True, capture_output=True,
                           env=dict(os.environ, G1_ROOT=str(self.g1)))
        self.assertEqual(r.returncode, 0, r.stderr)
        md = (self.g1 / "COST.md").read_text()
        self.assertIn("Total (ceiling): $0.12 of $45.00", md); self.assertIn("| G1 | $0.12 |", md)


if __name__ == "__main__":
    unittest.main(verbosity=1)
