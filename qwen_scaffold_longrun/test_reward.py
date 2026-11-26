import unittest
from unittest.mock import patch
from score_groups import reward_components, indexed_results, judge_batch

class RewardTests(unittest.TestCase):
    def test_diversity_cannot_rescue_incorrect_or_low_quality(self):
        self.assertEqual(reward_components(1,1,1,0,1)['reward'],0)
        self.assertEqual(reward_components(0.5,1,1,1,1)['diversity_bonus'],0)

    def test_slop_cannot_earn_diversity_bonus(self):
        result=reward_components(1,1,0,1,1)
        self.assertEqual(result['diversity_bonus'],0)
        self.assertEqual(result['reward'],0.7)

    def test_maximum_and_linear_quality(self):
        self.assertAlmostEqual(reward_components(1,1,1,1,1)['reward'],1.1)
        self.assertEqual(reward_components(0.8,1,1,1,0)['reward'],0.8)

    def test_invalid_judge_ids_fail_closed(self):
        data={'choices':[{'finish_reason':'stop','message':{'content':'{"results":[{"id":0},{"id":0}]}'}}]}
        with self.assertRaises(ValueError):indexed_results(data,[0,1])

    def test_identical_answers_share_one_judge_score(self):
        with patch('score_groups._judge_unique_batch',return_value=({0:{'id':0,'slop':False}},{0:{'id':0,'quality':4}})) as mock:
            labels,quality=judge_batch({},['same answer',' same answer ']*16)
        mock.assert_called_once_with({},['same answer'])
        self.assertEqual(len(quality),32)
        self.assertTrue(all(row['quality']==4 for row in quality.values()))

if __name__=='__main__':unittest.main()
