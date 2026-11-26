import unittest
from cap_grade import confirmed_rate_limit_label_grade,docker_command_timed_out
from cap_grade import confirmed_cap_grade,confirmed_timeout_grade,hermes_startup_failed
class CapTests(unittest.TestCase):
 def example(self,reward=1):return ({'exception_info':{'exception_type':'NonZeroAgentExitCodeError'},'verifier_result':{'rewards':{'reward':reward}}},{'messages':[{'role':'user','content':"You've reached the maximum number of tool-calling iterations allowed."},{'role':'assistant','content':'Done'}]})
 def test_reads_real_verifier_including_zero(self):
  for n in [0,1]:self.assertEqual(confirmed_cap_grade(*self.example(n))['reward'],n)
 def test_never_relabels_missing_verifier(self):
  raw,session=self.example();raw.pop('verifier_result');self.assertIsNone(confirmed_cap_grade(raw,session))
 def test_unexpected_crash_stays_invalid(self):
  raw,session=self.example();raw['exception_info']['exception_type']='AgentTimeoutError';self.assertIsNone(confirmed_cap_grade(raw,session))
 def test_no_final_answer_stays_invalid(self):
  raw,session=self.example();session['messages'].pop();self.assertIsNone(confirmed_cap_grade(raw,session))
class TimeoutTests(unittest.TestCase):
 def raw(self,reward=0):return {'exception_info':{'exception_type':'AgentTimeoutError'},'verifier_result':{'rewards':{'reward':reward}}}
 def test_keeps_real_verifier_grade(self):
  for n in [0,1]:
   grade=confirmed_timeout_grade(self.raw(n));self.assertEqual(grade['reward'],n);self.assertTrue(grade['agent_timeout'])
 def test_missing_verifier_stays_invalid(self):
  raw=self.raw();raw.pop('verifier_result');self.assertIsNone(confirmed_timeout_grade(raw))
 def test_other_exception_stays_invalid(self):
  raw=self.raw();raw['exception_info']['exception_type']='NonZeroAgentExitCodeError';self.assertIsNone(confirmed_timeout_grade(raw))
class StartupTests(unittest.TestCase):
 RAW={'exception_info':{'exception_type':'NonZeroAgentExitCodeError'}}
 TEXT="Hermes couldn't start the model connection: Gateway draining: update_paused. \nYour message was not sent. Run `hermes doctor` to check the setup, or /model to \npick a different provider.\n"
 def test_detects_unsent_message(self):self.assertTrue(hermes_startup_failed(self.RAW,'',self.TEXT))
 def test_session_with_turns_is_not_startup(self):self.assertFalse(hermes_startup_failed(self.RAW,'{}\n',self.TEXT))
 def test_other_empty_exit_is_not_startup(self):self.assertFalse(hermes_startup_failed(self.RAW,'','Segmentation fault'))

class RateLimitLabel(unittest.TestCase):
 def raw(self,kind='ApiRateLimitError',reward=0.0):return {'exception_info':{'exception_type':kind},'verifier_result':{'rewards':{'reward':reward}}}
 def test_keeps_real_verifier_grade(self):
  self.assertEqual(confirmed_rate_limit_label_grade(self.raw())['reward'],0.0);self.assertEqual(confirmed_rate_limit_label_grade(self.raw(reward=1))['reward'],1.0)
 def test_needs_the_label_and_a_verifier_result(self):
  self.assertIsNone(confirmed_rate_limit_label_grade(self.raw(kind='AgentTimeoutError')));self.assertIsNone(confirmed_rate_limit_label_grade(self.raw(reward=None)))
  self.assertIsNone(confirmed_rate_limit_label_grade({'exception_info':{'exception_type':'ApiRateLimitError'}}))

class DockerTimeout(unittest.TestCase):
 def raw(self,msg='Command timed out after 10 seconds',kind='RuntimeError',verifier=None):return {'exception_info':{'exception_type':kind,'exception_message':msg},'verifier_result':verifier}
 def test_detects_stalled_docker(self):self.assertTrue(docker_command_timed_out(self.raw()))
 def test_ignores_other_failures(self):
  self.assertFalse(docker_command_timed_out(self.raw(msg='boom')));self.assertFalse(docker_command_timed_out(self.raw(kind='AgentTimeoutError')))
 def test_keeps_a_real_verifier_result(self):self.assertFalse(docker_command_timed_out(self.raw(verifier={'rewards':{'reward':0}})))
