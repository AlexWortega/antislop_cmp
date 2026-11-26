import unittest
from cloud.gpu_guard import decide as gpu_decide
from cloud.agent_guard import decide as host_decide
CFG={'deadline':1000,'heartbeat_timeout':100,'first_heartbeat_grace':300}
class GpuGuard(unittest.TestCase):
 def test_keeps_a_healthy_rental(self):self.assertIsNone(gpu_decide(500,CFG,0,450,False))
 def test_agent_stop_wins(self):self.assertEqual(gpu_decide(10,CFG,0,None,True),'agent_host_requested_stop')
 def test_deadline(self):self.assertEqual(gpu_decide(1000,CFG,0,999,False),'deadline')
 def test_stale_heartbeat(self):self.assertEqual(gpu_decide(600,CFG,0,450,False),'agent_host_heartbeat_stale')
 def test_grace_before_first_heartbeat(self):
  self.assertIsNone(gpu_decide(200,CFG,0,None,False));self.assertEqual(gpu_decide(301,CFG,0,None,False),'agent_host_never_connected')
GPU={'status':'created','deadline':10000};VM={'deadline':20000,'keep_after_run':1000}
class HostGuard(unittest.TestCase):
 def test_running(self):self.assertEqual(host_decide(100,GPU,VM,{},False),['heartbeat'])
 def test_asks_trainer_to_wrap_up_before_deadline(self):self.assertIn('request_stop',host_decide(8300,GPU,VM,{},False))
 def test_stops_gpu_only_after_runner_finished(self):
  self.assertIn('stop_gpu',host_decide(100,GPU,VM,{},True));self.assertNotIn('stop_gpu',host_decide(100,GPU,VM,{'stop_sent_at':50},True))
 def test_marks_gpu_gone_after_silence(self):
  self.assertIn('mark_gpu_gone',host_decide(800,GPU,VM,{'stop_sent_at':100,'last_ssh_ok':100},True))
  self.assertNotIn('mark_gpu_gone',host_decide(600,GPU,VM,{'stop_sent_at':100,'last_ssh_ok':100},True))
 def test_marks_gpu_gone_after_its_deadline(self):self.assertIn('mark_gpu_gone',host_decide(10700,GPU,VM,{},False))
 def test_stops_itself_after_keep_window_or_deadline(self):
  self.assertEqual(host_decide(2500,GPU,VM,{'gpu_gone_at':1000},False),['stop_self'])
  self.assertEqual(host_decide(1500,GPU,VM,{'gpu_gone_at':1000},False),[])
  self.assertIn('stop_self',host_decide(20001,GPU,VM,{},False))
if __name__=='__main__':unittest.main()
