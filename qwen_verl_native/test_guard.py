import unittest
from guard import parse,rules,LR
def line(step,lr=LR,norm=2.,reward=.6):return f"\x1b[36m(TaskRunner pid=1)\x1b[0m step:{step} - actor/lr:np.float64({lr}) - actor/grad_norm:np.float64({norm}) - critic/rewards/mean:{reward} - actor/entropy:0.3"
class Parse(unittest.TestCase):
 def test_reads_numpy_and_plain_values(self):
  m=parse([line(21)])[21];self.assertEqual(m['actor/lr'],LR);self.assertEqual(m['critic/rewards/mean'],.6)
 def test_reads_scientific_notation(self):self.assertEqual(parse(['step:3 - actor/lr:np.float64(1e-06)'])[3]['actor/lr'],1e-6)
 def test_merges_validation_line(self):
  self.assertEqual(parse(['step:20 - val-core/harbor/reward/mean@4:0.7'])[20]['val-core/harbor/reward/mean@4'],.7)
class Rules(unittest.TestCase):
 def stops(self,lines):return rules(parse(lines))[0]
 def test_healthy_run_passes(self):self.assertEqual(rules(parse([line(i) for i in range(21,50)])),([],[]))
 def test_old_lr_stops(self):self.assertIn('actor/lr',self.stops([line(21,lr=2e-7)])[0])
 def test_bad_gradient_stops(self):
  for norm in ('nan','inf',0.):self.assertIn('grad_norm',self.stops([line(21,norm=norm)])[0])
 def test_low_validation_stops(self):self.assertTrue(self.stops(['step:30 - val-core/harbor/reward/mean@4:0.5']))
 def test_reward_collapse_needs_a_full_window(self):
  self.assertEqual(self.stops([line(i,reward=.1) for i in range(19)]),[])
  self.assertTrue(self.stops([line(i,reward=.1) for i in range(20)]))
if __name__=='__main__':unittest.main()
