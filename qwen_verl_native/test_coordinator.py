"""run_trial retry and grading against a fake collector; no Docker, GPU or judge."""
import json,shutil,tempfile,unittest
from pathlib import Path
from unittest import mock
import coordinator as c
STARTUP="Hermes couldn't start the model connection: Gateway draining: update_paused. Your message was not sent."
def fake(script):
 """Each collector call writes the next (exception, reward, session, text) outcome."""
 calls=iter(script)
 def run(cmd,**kw):
  out=Path(cmd[cmd.index('--out')+1]);exc,reward,session,text=next(calls)
  agent=out/'trials/t/agent';agent.mkdir(parents=True)
  (agent/'hermes-session.jsonl').write_text(session);(agent/'hermes.txt').write_text(text)
  (out/'x-result.json').write_text(json.dumps({'exception_info':{'exception_type':exc},'verifier_result':{'rewards':{'reward':reward}}}))
  (out/'outcomes.jsonl').write_text(json.dumps({'trial_id':'t','grade_outcome':{'valid':False,'reason':exc}})+'\n')
 return run
BODY={'uid':'u','task':'t','agent':'hermes','split':'train','group':'g','policy':'p'}
class RunTrial(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp());self.addCleanup(shutil.rmtree,self.root)
  for p in (mock.patch.object(c,'ROOT',self.root),mock.patch.object(c,'STARTUP_BACKOFF',(0,0))):p.start();self.addCleanup(p.stop)
 def trial(self,script):
  with mock.patch.object(c.subprocess,'run',fake(script)):return c.run_trial(BODY)
 def test_startup_failure_retries_and_preserves_evidence(self):
  row,out=self.trial([('NonZeroAgentExitCodeError',0,'',STARTUP),('AgentTimeoutError',1,'{}\n','')])
  self.assertEqual(row['grade_outcome']['reward'],1.);self.assertTrue(out.is_dir())
  self.assertTrue((self.root/'rollouts_startup_failed/g/u-0/x-result.json').exists())
 def test_startup_retries_are_bounded(self):
  with self.assertRaisesRegex(ValueError,'startup failed'):self.trial([('NonZeroAgentExitCodeError',0,'',STARTUP)]*3)
 def test_empty_session_with_other_error_is_reported_not_indexerror(self):
  with self.assertRaisesRegex(ValueError,'Invalid trial grade'):self.trial([('NonZeroAgentExitCodeError',0,'','crash')])
 def test_timeout_keeps_verifier_zero(self):
  row,_=self.trial([('AgentTimeoutError',0,'','')]);self.assertEqual(row['grade_outcome']['reward'],0.)
if __name__=='__main__':unittest.main()
