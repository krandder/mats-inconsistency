"""No rentals: exercise scoped cleanup and budget accounting against fake GCE."""
import json,os,pathlib,shutil,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Infra(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
  shutil.copytree(ROOT/'infra',self.root/'infra',ignore=shutil.ignore_patterns('state'))
  (self.root/'GPU-LOG.md').write_text('')
  with (self.root/'infra/lib.sh').open('a') as f:f.write('\npost_status() { :; }\n')
 def tearDown(self):self.tmp.cleanup()
 def runsh(self,s):return subprocess.run(['bash','-c',s],cwd=self.root,text=True,capture_output=True)
 def test_timeout_forms(self):
  for value in ['19500s','19800s','24m']:
   r=subprocess.run(['timeout',value,'true'],capture_output=True)
   self.assertEqual(r.returncode,0)
 def test_bad_timeout_before_rental(self):
  r=self.runsh("gcloud() { touch cloud-called; }; export -f gcloud; GPU_RUN_TIMEOUT=5h25m bash infra/gpu-run.sh --up -- true")
  self.assertNotEqual(r.returncode,0);self.assertFalse((self.root/'cloud-called').exists())
 def test_group_accounting(self):
  (self.root/'GPU-LOG.md').write_text('- x CREATE vm=a group=pool-sddpv reserve_usd=5.22\n- x CREATE vm=b group=other reserve_usd=22.14\n- x COST-AUDIT vm=a cost_usd=1.00\n- x CREATE vm=c group=pool-sddpv reserve_usd=5.22\n')
  r=self.runsh('source infra/lib.sh; spend_usd pool-sddpv; spend_usd');self.assertEqual(r.stdout,'6.22\n28.36\n')
 def test_scoped_cleanup(self):
  r=self.runsh('''
 gcloud() {
   if [[ "$*" == *"instances list"* ]]; then
     printf 'soup-gpu-one\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\nsoup-gpu-two\tus-central1-b\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n'
   elif [[ "$*" == *"instances delete"* ]]; then printf '%s\\n' "$*" >> deletes; fi
 }; export -f gcloud
 bash infra/gpu-down.sh --only soup-gpu-one --no-sync
 ''');self.assertEqual(r.returncode,0,r.stderr)
  lines=(self.root/'deletes').read_text();self.assertIn('delete soup-gpu-one',lines);self.assertNotIn('soup-gpu-two',lines)
 def test_unnamed_refuses_parallel(self):
  r=self.runsh('''gcloud() { printf 'soup-gpu-one\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n'; }; export -f gcloud; bash infra/gpu-up.sh --no-wait''')
  self.assertNotEqual(r.returncode,0);self.assertIn('Use --run-id',r.stderr)
 def test_unnamed_run_never_adopts_named(self):
  r=self.runsh("""gcloud() {
   if [[ "$*" == *"instances list"* ]]; then printf 'soup-gpu-owned\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n';
   elif [[ "$*" == *"instances describe"* ]]; then echo phase3-k16-f12-s1;
   else touch unexpected-cloud-action; fi
  }; export -f gcloud; bash infra/gpu-run.sh -- true""")
  self.assertNotEqual(r.returncode,0);self.assertFalse((self.root/'unexpected-cloud-action').exists())
 def test_plain_down_refuses_named(self):
  r=self.runsh("""gcloud() {
   if [[ "$*" == *"instances list"* ]]; then printf 'soup-gpu-owned\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n';
   elif [[ "$*" == *"instances describe"* ]]; then echo phase3-k16-f12-s1;
   else touch unexpected-cloud-action; fi
  }; export -f gcloud; bash infra/gpu-down.sh""")
  self.assertEqual(r.returncode,2);self.assertFalse((self.root/'unexpected-cloud-action').exists())
 def early_cleanup(self,interrupt=False):
  up=self.root/'infra/gpu-up.sh';up.write_text('#!/bin/bash\nprintf "%s\n" "$*" > created-args\n'+('kill -TERM "$PPID"\n' if interrupt else '')+'exit 0\n')
  down=self.root/'infra/gpu-down.sh';down.write_text('#!/bin/bash\nprintf "%s\n" "$*" > deleted-args\n')
  up.chmod(0o755);down.chmod(0o755)
  r=self.runsh('gcloud() { return 1; }; export -f gcloud; bash infra/gpu-run.sh --up --run-id owned -- true')
  self.assertNotEqual(r.returncode,0)
  created=(self.root/'created-args').read_text().split();name=created[created.index('--name')+1]
  self.assertIn('--only '+name,(self.root/'deleted-args').read_text())
 def test_failed_listing_after_create_cleans_known_name(self):self.early_cleanup()
 def test_term_during_create_cleans_known_name(self):self.early_cleanup(True)
 def test_cleanup_uses_intent_when_listing_fails(self):
  state=self.root/'infra/state';state.mkdir(exist_ok=True);(state/'intent-soup-gpu-owned').write_text('us-central1-b')
  r=self.runsh("""gcloud() {
   if [[ "$*" == *"instances list"* ]]; then return 1;
   elif [[ "$*" == *"instances delete"* ]]; then printf '%s\n' "$*" > deleted; fi
  }; export -f gcloud; bash infra/gpu-down.sh --only soup-gpu-owned --no-sync""")
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn('delete soup-gpu-owned --zone=us-central1-b',(self.root/'deleted').read_text())
 def test_named_cleanup_never_syncs(self):
  r=self.runsh("""gcloud() {
   if [[ "$*" == *"instances list"* ]]; then printf 'soup-gpu-owned\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n';
   elif [[ "$*" == *"instances describe"* ]]; then echo phase3-k16-f12-s1;
   elif [[ "$*" == *"instances delete"* ]]; then :;
   else touch unexpected-sync; fi
  }; export -f gcloud; bash infra/gpu-down.sh --only soup-gpu-owned""")
  self.assertEqual(r.returncode,0,r.stderr);self.assertFalse((self.root/'unexpected-sync').exists())
 def test_pending_reservation_counts_once(self):
  (self.root/'GPU-LOG.md').write_text('- x PENDING-CREATE vm=a group=pool-sddpv reserve_usd=5.22\n')
  self.assertEqual(self.runsh('source infra/lib.sh; spend_usd pool-sddpv').stdout,'5.22\n')
  with (self.root/'GPU-LOG.md').open('a') as f:f.write('- x CREATE vm=a group=pool-sddpv reserve_usd=5.22\n')
  self.assertEqual(self.runsh('source infra/lib.sh; spend_usd pool-sddpv').stdout,'5.22\n')
 def test_owner_lookup_failure_still_deletes(self):
  r=self.runsh("""gcloud() {
   if [[ "$*" == *"instances list"* ]]; then printf 'soup-gpu-owned\tus-central1-a\t2026-09-27T12:00:00Z\tg2-standard-8\t1\tRUNNING\n';
   elif [[ "$*" == *"instances describe"* ]]; then return 1;
   elif [[ "$*" == *"instances delete"* ]]; then touch deleted;
   else touch unexpected-sync; fi
  }; export -f gcloud; bash infra/gpu-down.sh --only soup-gpu-owned""")
  self.assertEqual(r.returncode,0,r.stderr);self.assertTrue((self.root/'deleted').exists());self.assertFalse((self.root/'unexpected-sync').exists())
 def test_audit_uses_machine_rate(self):
  binpath=self.root/'bin';binpath.mkdir()
  fake=binpath/'gcloud';fake.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps([dict(operationType='insert',insertTime='2026-09-27T00:00:00Z'),dict(operationType='delete',endTime='2026-09-27T01:00:00Z')] if 'operations' in sys.argv else []))\n");fake.chmod(0o755)
  env=dict(os.environ,PATH=str(binpath)+':'+os.environ['PATH'],GPU_LOG_PATH=str(self.root/'GPU-LOG.md'),GPU_CONTROL_ROOT=str(self.root))
  for machine,rate in [('g2-standard-8',.87),('a2-highgpu-1g',3.69)]:
   (self.root/'GPU-LOG.md').write_text(f'- x CREATE vm=soup-gpu-test type={machine} reserve_usd=22.14\n')
   out=self.root/machine
   r=subprocess.run(['python3',str(ROOT/'scripts/gpu-rental-audit.py'),'soup-gpu-test',str(out)],env=env,capture_output=True)
   self.assertEqual(r.returncode,0,r.stderr)
   self.assertEqual(json.loads((out/'cost-audit.json').read_text())['cost_ceiling_usd'],rate)
 def test_remote_workdir_is_inside_sudo_shell(self):
  r=self.runsh("""source infra/lib.sh
  vm_ssh() { printf '%s\n' "$*" > transport; cat > remote-script; }
  vm_root 127.0.0.1 env 'SOUP_CMD=build/soup --gpu on' scripts/run.sh --K '2^16'
  """)
  self.assertEqual(r.returncode,0,r.stderr)
  self.assertEqual((self.root/'transport').read_text(),'127.0.0.1 sudo bash -l -s\n')
  remote=(self.root/'remote-script').read_text();self.assertTrue(remote.startswith('cd /root/soup\n'))
  self.assertIn('SOUP_CMD=build/soup',remote)
 def test_two_no_progress_attempts_stop(self):
  scripts=self.root/'scripts';scripts.mkdir();shutil.copy(ROOT/'scripts/gpu-phase3.sh',scripts/'gpu-phase3.sh')
  (self.root/'source-commit.txt').write_text('fixture')
  control=self.root/'control';control.mkdir()
  (self.root/'infra/gpu-run.sh').write_text('#!/bin/bash\necho attempt >> "$GPU_CONTROL_ROOT/attempts"\nexit 255\n')
  (self.root/'infra/gpu-run.sh').chmod(0o755)
  r=self.runsh('sleep() { :; }; export -f sleep; GPU_CONTROL_ROOT="$PWD/control" GPU_PHASE3_FROZEN=1 bash scripts/gpu-phase3.sh 1')
  self.assertEqual(r.returncode,4,r.stderr);self.assertEqual(len((control/'attempts').read_text().splitlines()),2)
if __name__=='__main__':unittest.main()
