import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'repo/scaffold_training'))
from grpo_update import validated_groups,REVISION

class TrainingContractTests(unittest.TestCase):
    def build(self,directory,n=32,split='train'):
        root=Path(directory);protocol=root/'protocol.json'
        protocol.write_text(json.dumps({'tasks':[{'task':'x','split':'train'}]}))
        group={'task':'x','split':split,'base_revision':REVISION,'policy_id':'p',
               'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
               'trajectories':[{'outcome':{'valid':True,'reward':1.1 if i%2 else 0.},'turns':[{'prompt_ids':[1],'response_ids':[2],'old_logprobs':[-0.5]}]} for i in range(n)]}
        path=root/'groups.jsonl';path.write_text(json.dumps(group)+'\n')
        return path,protocol

    def test_32_and_bonus_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol=self.build(directory)
            groups,rejected=validated_groups(path,16384,protocol,'p',32)
            self.assertEqual(len(groups),1);self.assertFalse(rejected)

    def test_four_response_group_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol=self.build(directory,4)
            groups,rejected=validated_groups(path,16384,protocol,'p',4)
            self.assertEqual(len(groups),1);self.assertFalse(rejected)

    def test_small_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol=self.build(directory,31)
            with self.assertRaises(ValueError):validated_groups(path,16384,protocol,'p',32)

    def test_dev_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol=self.build(directory,split='dev')
            with self.assertRaises(ValueError):validated_groups(path,16384,protocol,'p',32)

if __name__=='__main__':unittest.main()
