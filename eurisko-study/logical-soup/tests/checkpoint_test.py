import csv,json,pathlib,struct,subprocess,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Checkpoint(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.tmp.name)
  config=dict(logk=0,logq=2,seed=1,K=1,F=4,w0=1048576,free_income=0)
  (self.p/'config.json').write_text(json.dumps(config))
  blob=bytearray(28+76+4*52+120);struct.pack_into('<7I',blob,0,1347768147,1,0,2,6,1,1);struct.pack_into('<I',blob,28+64,1048570);struct.pack_into('<Q',blob,len(blob)-8,6);(self.p/'soup.dump').write_bytes(blob)
  (self.p/'metrics.csv').write_text('epoch,wealth,trader_pnl,used,burned,cum_ops\n0,1048570,0,6,0,6\n')
  (self.p/'timing.csv').write_text('first_epoch,epochs,wall_s,instructions,instr_per_s\n0,1,1,6,6\n')
 def tearDown(self):self.tmp.cleanup()
 def check(self,*args):return subprocess.run(['python3',str(ROOT/'scripts/verify-checkpoint.py'),str(self.p),*args],capture_output=True)
 def test_complete(self):self.assertEqual(self.check('1').returncode,0)
 def test_truncated_snapshot(self):
  f=self.p/'soup.dump';f.write_bytes(f.read_bytes()[:-1]);self.assertNotEqual(self.check().returncode,0)
 def test_snapshot_state_disagrees(self):
  f=self.p/'soup.dump';b=bytearray(f.read_bytes());struct.pack_into('<I',b,92,1048571);f.write_bytes(b);self.assertNotEqual(self.check().returncode,0)
 def test_interrupted_publication_repair(self):
  with (self.p/'metrics.csv').open('a') as f:f.write('1,1048564,0,6,0,12\n')
  with (self.p/'timing.csv').open('a') as f:f.write('1,1,1,6,6\n')
  self.assertNotEqual(self.check().returncode,0);self.assertEqual(self.check('--repair').returncode,0)
  self.assertEqual(len((self.p/'metrics.csv').read_text().splitlines()),2)
  self.assertEqual(len((self.p/'timing.csv').read_text().splitlines()),2)
if __name__=='__main__':unittest.main()
