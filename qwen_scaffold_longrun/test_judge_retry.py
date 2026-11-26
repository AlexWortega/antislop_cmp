import unittest
from unittest.mock import patch
import json
from score_groups import checked_judge,JudgeBatchError

def response(ids):return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'results':[{'id':i} for i in ids]})}}]}
class RetryTests(unittest.TestCase):
 def test_recovers_without_relabelling(self):
  with patch('score_groups.budget.call',side_effect=[response([-1]),response([0,1])]) as call,patch('score_groups.time.sleep'):
   self.assertEqual(set(checked_judge({'messages':[]},[0,1])),{0,1})
   self.assertEqual(call.call_count,2)
   self.assertIn('Formatting retry',call.call_args.args[0]['messages'][-1]['content'])
 def test_exhausted_is_skippable(self):
  with patch('score_groups.budget.call',return_value=response([-1])) as call,patch('score_groups.time.sleep'):
   with self.assertRaises(JudgeBatchError):checked_judge({'messages':[]},[0,1])
   self.assertEqual(call.call_count,3)
 def test_non_object_results_are_retried(self):
  invalid={'choices':[{'finish_reason':'stop','message':{'content':'{"results":["bad"]}'}}]}
  with patch('score_groups.budget.call',side_effect=[invalid,response([0])]),patch('score_groups.time.sleep'):
   self.assertEqual(set(checked_judge({'messages':[]},[0])),{0})
 def test_budget_failure_is_fatal(self):
  with patch('score_groups.budget.call',side_effect=RuntimeError('budget exhausted')) as call:
   with self.assertRaisesRegex(RuntimeError,'budget exhausted'):checked_judge({'messages':[]},[0])
   self.assertEqual(call.call_count,1)
