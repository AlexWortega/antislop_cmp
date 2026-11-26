import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core import read_prompt, teacher_label, score

class Tests(unittest.TestCase):
    def test_runtime_modes_do_not_share_response_cache(self):
        from round3 import runtime
        def response(value):
            import json
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'results':[{'id':0,'slop':value}]})}}]}
        with tempfile.TemporaryDirectory() as d, patch('round3.runtime.ROOT',Path(d)), patch.dict('os.environ',{'OPENROUTER_API_KEY':'unit-test-only'}):
            (Path(d)/'cache').mkdir()
            with patch('round3.runtime.body_for',return_value={}), patch('round3.runtime.request',side_effect=[response(True),response(False)]) as request:
                self.assertTrue(runtime.predict_mode('prompt','text','default'))
                self.assertTrue(runtime.predict_mode('prompt','text','default'))
                self.assertFalse(runtime.predict_mode('prompt','text','medium'))
                self.assertEqual(request.call_count,2)

    def test_unavailable_predictions_are_errors_not_human_labels(self):
        from round2.evaluate_round import metrics
        rows=[{'slop':False},{'slop':False},{'slop':True},{'slop':True}]
        m=metrics(rows,[None,False,True,None])
        self.assertEqual(m['agreement'],0.5)
        self.assertEqual(m['balanced_accuracy'],0.5)
        self.assertEqual(m['unavailable'],2)
        self.assertEqual(m['fp']+m['fn'],0)

    def test_cached_provider_refusal_does_not_become_a_label(self):
        import core
        import json
        with tempfile.TemporaryDirectory() as d, patch('core.ROOT',Path(d)):
            cache=Path(d)/'cache';cache.mkdir()
            key=core.digest([core.MODEL,'strict-schema-batch-v2','prompt',['text']])
            (cache/('opus_batch_'+key+'.json')).write_text(json.dumps({'choices':[{'finish_reason':'content_filter','message':{'content':None}}]}))
            with self.assertRaises(core.PredictionUnavailable):core.predict_batch('prompt',['text'])

    def test_tiny_probability_is_not_slop(self):
        self.assertFalse(teacher_label({'fraction_ai': 1e-42, 'windows':[{'label':'Human Written'}]}))
        self.assertTrue(teacher_label({'windows':[{'label':'Human Written'},{'label':'AI-Assisted'}]}))
        with self.assertRaises(ValueError):
            teacher_label({'windows':[{'label':'unknown'}]})

    def test_candidate_cannot_execute_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.py'
            p.write_text('PROMPT = "A sufficiently long prompt string"')
            self.assertIn('prompt',read_prompt(p))
            p.write_text('import os\nPROMPT = "A sufficiently long prompt string"')
            with self.assertRaises(ValueError): read_prompt(p)

    def test_balanced_accuracy(self):
        rows=[{'text':str(i),'slop':i==3} for i in range(4)]
        with patch('core.predict_batch',side_effect=lambda prompt,texts: [False]*len(texts)):
            metrics,_=score('prompt',rows)
        self.assertEqual(metrics['agreement'],0.75)
        self.assertEqual(metrics['balanced_accuracy'],0.5)


class BudgetTests(unittest.TestCase):
    def test_resumed_settlement_does_not_charge_twice(self):
        import budget
        with tempfile.TemporaryDirectory() as d, patch('budget.PATH', Path(d)/'budget.json'):
            token=budget.reserve('pangram',5)
            budget.settle_once(token,1)
            budget.settle_once(token,1)
            with budget.ledger() as state:
                self.assertAlmostEqual(state['spent']['pangram'],1.2)
                self.assertNotIn(token,state['pending'])

    def test_reservation_prevents_overspend_and_settlement_releases_unused(self):
        import budget
        with tempfile.TemporaryDirectory() as d, patch('budget.PATH', Path(d)/'budget.json'):
            token=budget.reserve('openrouter',99)
            with self.assertRaises(RuntimeError): budget.reserve('openrouter',2)
            budget.settle(token,1)
            budget.reserve('openrouter',90)

class EvaluatorFailureTests(unittest.TestCase):
    def test_invalid_candidate_writes_failed_result(self):
        import subprocess
        import sys
        import json
        with tempfile.TemporaryDirectory() as d:
            candidate=Path(d)/'bad.py'
            candidate.write_text('import os')
            out=Path(d)/'result'
            evaluator=Path(__file__).parent/'evaluate.py'
            completed=subprocess.run([sys.executable,str(evaluator),'--program_path',str(candidate),'--results_dir',str(out)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertFalse(json.loads((out/'correct.json').read_text())['correct'])

if __name__ == '__main__': unittest.main()
