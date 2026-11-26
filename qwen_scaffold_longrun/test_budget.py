import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
import judge_budget as budget

class BudgetTests(unittest.TestCase):
    def test_refuses_request_before_exceeding_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=root/'budget.json'
            ledger.write_text(json.dumps({'limit':120.,'spent_with_margin':119.9,'pending':{},'requests':0}))
            with patch.object(budget,'LEDGER',ledger),patch.object(budget,'CACHE',root/'cache'),patch.object(budget.requests,'post') as post:
                with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                    budget.call({'model':'anthropic/claude-opus-5.5','max_tokens':8192})
                post.assert_not_called()

    def test_uncertain_failure_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=root/'budget.json'
            with patch.object(budget,'LEDGER',ledger),patch.object(budget,'CACHE',root/'cache'),patch.dict(budget.os.environ,{'OPENROUTER_API_KEY':'test-only'}),patch.object(budget.requests,'post',return_value=Mock(ok=False,status_code=500)):
                with self.assertRaisesRegex(RuntimeError,'reservation retained'):
                    budget.call({'model':'anthropic/claude-opus-5.5','max_tokens':100})
            self.assertGreater(sum(json.loads(ledger.read_text())['pending'].values()),0)

if __name__=='__main__':unittest.main()
