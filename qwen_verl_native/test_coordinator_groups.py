"""Group handling in process(): one broken trial or judge call must not crash the training step."""
import json,shutil,tempfile,threading,unittest
from pathlib import Path
from unittest import mock
import coordinator as c

def body(uid):return {'uid':uid,'task':'t','agent':'pi','split':'train','group':'g','policy':'p'}

class Groups(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp());self.addCleanup(shutil.rmtree,self.root)
  old=self.root/'old';old.mkdir()
  (old/'opus_budget.json').write_text(json.dumps({'limit':120.,'spent_with_margin':0.,'pending':{}}))
  (self.root/'instance.json').write_text(json.dumps({'deadline':9e12}))
  for p in (mock.patch.object(c,'ROOT',self.root),mock.patch.object(c,'OLD',old),mock.patch.object(c,'GROUP_SIZE',4),
            mock.patch.object(c,'CIRCUIT',self.root/'CIRCUIT_OPEN'),mock.patch.object(c,'GROUPS',{}),mock.patch.object(c,'RECENT',c.collections.deque(maxlen=64)),mock.patch.object(c,'GROUPS_RECENT',c.collections.deque(maxlen=16))):
   p.start();self.addCleanup(p.stop)
 def fake_trial(self,fail):
  def run(b):
   if b['uid'] in fail:raise fail[b['uid']]
   out=self.root/'rollouts/g'/b['uid'];out.mkdir(parents=True,exist_ok=True)
   return {'trial_id':b['uid'],'grade_outcome':{'valid':True}},out
  return run
 def fake_score(self,rewards=None,exc=None):
  def score(prepared,parent,out):
   if exc:raise exc
   out.mkdir(exist_ok=True)
   rows=json.loads(prepared.read_text())['trajectories']
   (out/'metrics.jsonl').write_text(''.join(json.dumps({'trial_id':r['trial_id'],'reward':rewards[r['trial_id']]})+'\n' for r in rows))
  return score
 def run_group(self,fail,score):
  results,errors={},{}
  def one(u):
   try:results[u]=c.process(body(u))
   except Exception as e:errors[u]=e
  with mock.patch.object(c,'run_trial',self.fake_trial(fail)),mock.patch.object(c,'score',score):
   threads=[threading.Thread(target=one,args=(u,)) for u in 'abcd']
   for t in threads:t.start()
   for t in threads:t.join(10)
  return results,errors
 def test_failed_trial_gets_mean_reward_and_mask_flag(self):
  res,err=self.run_group({'d':ValueError('Invalid trial grade: NonZeroAgentExitCodeError')},self.fake_score({'a':1.,'b':.5,'c':0.}))
  self.assertEqual(err,{});self.assertAlmostEqual(res['d']['reward'],.5);self.assertTrue(res['d']['infra_failed'])
  self.assertEqual(res['a']['reward'],1.);self.assertNotIn('infra_failed',res['a'])
 def test_judge_failure_skips_group_without_crash(self):
  res,err=self.run_group({},self.fake_score(exc=RuntimeError('Judge transport failed: ConnectionError')))
  self.assertEqual(err,{});self.assertTrue(all(r['infra_failed'] and r['reward']==0. for r in res.values()))
  self.assertTrue((self.root/'skipped_groups.jsonl').exists())
 def test_budget_exhaustion_still_stops(self):
  res,err=self.run_group({},self.fake_score(exc=RuntimeError('Judge HTTP 402; reservation retained')))
  self.assertEqual(len(err),4);self.assertIn('402',str(err['a']))
 def test_too_few_valid_trials_skip_group(self):
  fail={u:ValueError('boom') for u in 'bcd'}
  res,err=self.run_group(fail,self.fake_score({'a':1.}))
  self.assertEqual(err,{});self.assertTrue(all(r['infra_failed'] for r in res.values()))
 def test_circuit_opens_when_most_recent_trials_fail(self):
  for _ in range(20):c.note('trial',True,'x')
  for _ in range(12):c.note('trial',False)
  self.assertTrue((self.root/'CIRCUIT_OPEN').exists())
  with self.assertRaisesRegex(RuntimeError,'Circuit open'):c.process(body('z'))
 def test_circuit_stays_closed_on_occasional_failures(self):
  for i in range(64):c.note('trial',i%5==0)
  self.assertFalse((self.root/'CIRCUIT_OPEN').exists())
 def test_circuit_opens_when_most_groups_are_skipped(self):
  for _ in range(5):c.note_group(True)
  for _ in range(3):c.note_group(False)
  self.assertTrue((self.root/'CIRCUIT_OPEN').exists())
 def test_one_skipped_group_keeps_circuit_closed(self):
  for i in range(16):c.note_group(i==3)
  self.assertFalse((self.root/'CIRCUIT_OPEN').exists())
if __name__=='__main__':unittest.main()
