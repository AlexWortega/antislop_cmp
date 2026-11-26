import json,tempfile,unittest
from pathlib import Path
from hf_ckpt import due,manifest,remote_mismatches,local_mismatches,scrub,sha256
class Due(unittest.TestCase):
 def test_first_checkpoint_always_uploads(self):self.assertTrue(due(21,None))
 def test_waits_for_next_tenth_step(self):
  self.assertFalse(due(29,21));self.assertTrue(due(30,21));self.assertFalse(due(30,30));self.assertTrue(due(41,30))
class Hashes(unittest.TestCase):
 def setUp(self):
  self.d=tempfile.TemporaryDirectory();self.root=Path(self.d.name);(self.root/'actor').mkdir();(self.root/'actor/a.pt').write_bytes(b'abc');(self.root/'b').write_bytes(b'x')
  (self.root/'.cache').mkdir();(self.root/'.cache/lock').write_bytes(b'')
  self.m=manifest(self.root)
 def tearDown(self):self.d.cleanup()
 def test_manifest_skips_hub_cache_and_itself(self):
  (self.root/'manifest.json').write_text('{}');self.assertEqual(sorted(manifest(self.root)),['actor/a.pt','b'])
 def test_remote_accepts_match_and_sizes_for_small_files(self):
  self.assertEqual(remote_mismatches(self.m,{'actor/a.pt':(3,self.m['actor/a.pt']['sha256']),'b':(1,None)}),[])
 def test_remote_flags_missing_short_or_wrong_hash(self):
  self.assertEqual(remote_mismatches(self.m,{'actor/a.pt':(3,'0'*64)}),['actor/a.pt','b'])
  self.assertEqual(remote_mismatches(self.m,{'actor/a.pt':(2,None),'b':(1,None)}),['actor/a.pt'])
 def test_local_detects_corruption_and_loss(self):
  (self.root/'manifest.json').write_text(json.dumps(self.m));self.assertEqual(local_mismatches(self.root),[])
  (self.root/'b').write_bytes(b'y');self.assertEqual(local_mismatches(self.root),['b'])
  (self.root/'b').unlink();self.assertEqual(local_mismatches(self.root),['b'])
 def test_scrub_hides_token(self):self.assertEqual(scrub('401 hf_abc',"hf_abc"),'401 [token]')
if __name__=='__main__':unittest.main()
