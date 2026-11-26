import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lab_core as core


def reply(rows, cost=0.01):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'results': rows})}}], 'usage': {'cost': cost}}


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        patcher = patch.object(core, 'STATE', Path(self.dir.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.dir.cleanup)
        self.cfg = copy.deepcopy(core.DEFAULT_CONFIG)
        self.cfg['judge']['api_key_env'] = ''
        # One fake judge returns both fields, as before the task judge was split out.
        self.cfg['judge']['fields'].append({'name': 'task', 'min': 0, 'max': 10, 'description': ''})
        self.cfg['reward']['formula'] = '0.5*style/10 + 0.3*task/10 + 0.2*length - penalty'
        self.cfg['task_judge'].update(enabled=False, api_key_env='')

    def judge_returns(self, by_text):
        """Fake endpoint: scores each candidate by a lookup on its text."""
        def post(url, headers, body, timeout=240):
            candidates = json.loads(body['messages'][1]['content'])['candidates']
            return reply([{'id': c['id'], **by_text[c['text']], 'reason': 'r'} for c in candidates])
        return patch.object(core, '_post', side_effect=post)


class Formula(unittest.TestCase):
    def test_arithmetic_and_functions(self):
        self.assertAlmostEqual(core.compile_formula('0.5*style/10 + max(0, task - 5)')({'style': 8, 'task': 7}), 2.4)

    def test_rejects_code(self):
        for bad in ('__import__("os")', 'style.real', 'x[0]', '"a"', '2**9', 'open("f")'):
            with self.assertRaises(ValueError):
                core.compile_formula(bad)({'style': 1, 'x': 1})

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            core.compile_formula('nope + 1')({})


class Flags(unittest.TestCase):
    def test_emoji_and_hashtag(self):
        found = core.flags('Готово ✅ #продуктивность', 'ru')
        self.assertEqual((found['emoji'], found['hashtag'], found['language_switch']), (1, 1, 0))

    def test_plain_prose_is_clean(self):
        self.assertEqual(sum(core.flags('We shipped it on Friday. Issue #12 is closed, C# works.', 'en').values()), 0)

    def test_markdown_only_when_not_asked(self):
        text = '## Plan\n- one\n- two'
        self.assertEqual(core.flags(text)['markdown'], 1)
        self.assertEqual(core.flags(text, allow_markdown=True)['markdown'], 0)

    def test_language_switch(self):
        self.assertEqual(core.flags('This is English text.', 'ru')['language_switch'], 1)
        self.assertEqual(core.flags('Это русский текст с одним словом API.', 'ru')['language_switch'], 0)

    def test_length_factor(self):
        self.assertEqual(core.length_factor(110, 100, 0.15, 0.35), 1.0)
        self.assertLess(core.length_factor(40, 100, 0.15, 0.35), 0.3)
        self.assertEqual(core.length_factor(40, None, 0.15, 0.35), 1.0)


class Reward(Sandbox):
    def test_style_dominates_and_penalties_apply(self):
        good, slop, emoji = 'A plain human note about Friday.', 'In today\'s fast-paced world, it is vital.', 'A plain note ✅ #win'
        with self.judge_returns({good: {'style': 9, 'task': 8}, slop: {'style': 2, 'task': 8}, emoji: {'style': 9, 'task': 8}}):
            out = core.score_group(self.cfg, 'write a note', [good, slop, emoji, '  '])
        r = out['rewards']
        self.assertAlmostEqual(r[0], 0.5 * 0.9 + 0.3 * 0.8 + 0.2)
        self.assertAlmostEqual(r[1], 0.5 * 0.2 + 0.3 * 0.8 + 0.2)
        self.assertAlmostEqual(r[2], r[0] - 0.3)
        self.assertEqual(r[3], 0.0)
        self.assertTrue(out['items'][3]['empty'])
        self.assertFalse(out['stats']['tied'])

    def test_group_mode_is_one_call_and_duplicates_judged_once(self):
        self.cfg['judge']['mode'] = 'group'
        with self.judge_returns({'a': {'style': 5, 'task': 5}, 'b': {'style': 7, 'task': 5}}) as post:
            out = core.score_group(self.cfg, 't', ['a', 'b', 'a'])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(out['rewards'][0], out['rewards'][2])

    def test_rank_normalize(self):
        self.cfg['reward']['rank_normalize'] = True
        with self.judge_returns({'a': {'style': 5, 'task': 5}, 'b': {'style': 7, 'task': 5}, 'c': {'style': 9, 'task': 5}}):
            out = core.score_group(self.cfg, 't', ['a', 'b', 'c'])
        self.assertEqual(out['rewards'], [0.0, 0.5, 1.0])

    def test_cache_prevents_second_charge(self):
        with self.judge_returns({'a': {'style': 5, 'task': 5}}) as post:
            first = core.score_group(self.cfg, 't', ['a'])
            second = core.score_group(self.cfg, 't', ['a'])
            third = core.score_group(self.cfg, 't', ['a'], repeat=1)
        self.assertEqual(post.call_count, 2)
        self.assertEqual((first['cost_usd'], second['cost_usd'], third['cost_usd']), (0.01, 0.0, 0.01))
        self.assertAlmostEqual(core.ledger()['spent_usd'], 0.02)

    def test_bad_judge_output_never_becomes_a_reward(self):
        with patch.object(core, '_post', return_value=reply([{'id': 0, 'style': 99, 'task': 5, 'reason': ''}])) as post:
            with self.assertRaises(core.JudgeError):
                core.score_group(self.cfg, 't', ['a'])
        self.assertEqual(post.call_count, 3)

    def test_budget_stops_calls(self):
        self.cfg['budget_usd'] = 0.005
        with self.judge_returns({'a': {'style': 5, 'task': 5}, 'b': {'style': 5, 'task': 5}}):
            core.score_group(self.cfg, 't', ['a'])
            with self.assertRaises(RuntimeError):
                core.score_group(self.cfg, 't', ['b'])

    def test_json_inside_prose_is_parsed(self):
        self.cfg['judge']['structured'] = False
        content = 'Here you go:\n{"results":[{"id":0,"style":6,"task":7,"reason":"ok"}]}'
        with patch.object(core, '_post', return_value={'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}):
            self.assertEqual(core.score_group(self.cfg, 't', ['a'])['items'][0]['scores'], {'style': 6.0, 'task': 7.0})


class ReferenceJudge(Sandbox):
    """Pairwise comparison with a real human answer ("beat the human")."""
    def setUp(self):
        super().setUp()
        self.cfg['ref_judge']['api_key_env'] = ''
        self.cfg['reward']['formula'] = 'has_ref*beat + (1 - has_ref)*style/10'

    def fake(self, *ranking):
        """Main judge: style 5 for everything. Reference judge: picks the side whose text comes first in ranking, else tie."""
        def post(url, headers, body, timeout=240):
            user = json.loads(body['messages'][1]['content'])
            if 'candidates' in user:
                return reply([{'id': c['id'], 'style': 5, 'task': 5, 'reason': 'r'} for c in user['candidates']])
            rank = {t: k for k, t in enumerate(ranking)}
            a, b = rank.get(user['A'], 99), rank.get(user['B'], 99)
            better = 'tie' if a == b else 'A' if a < b else 'B'
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'better': better, 'reason': 'x'})}}],
                    'usage': {'cost': 0.01}}
        return patch.object(core, '_post', side_effect=post)

    def test_beat_follows_the_pick_whatever_the_order(self):
        human = 'the human answer'
        with self.fake('good', human, 'bad'):
            out = core.score_group(self.cfg, 't', ['good', 'bad'], reference=human)
        self.assertEqual(out['rewards'], [1.0, 0.0])
        self.assertEqual([out['items'][0]['beat'], out['items'][1]['beat']], [1.0, 0.0])
        with self.fake():
            self.assertEqual(core.score_group(self.cfg, 't', ['good'], reference=human, repeat=1)['rewards'], [0.5])

    def test_order_is_mixed_and_stable(self):
        with self.fake('x'):
            flags = [core.ref_compare(self.cfg, 't', f'answer {i}', 'human')[0]['candidate_first'] for i in range(40)]
            again = [core.ref_compare(self.cfg, 't', f'answer {i}', 'human')[0]['candidate_first'] for i in range(40)]
        self.assertEqual(flags, again)
        self.assertTrue(10 < sum(flags) < 30)

    def test_no_reference_or_formula_without_beat_makes_no_pairwise_call(self):
        with self.fake('good') as post:
            out = core.score_group(self.cfg, 't', ['good'])
        self.assertEqual(post.call_count, 1)
        self.assertAlmostEqual(out['rewards'][0], 0.5)
        self.cfg['reward']['formula'] = 'style/10'
        with self.fake('good') as post:
            core.score_group(self.cfg, 't2', ['good'], reference='h')
        self.assertEqual(post.call_count, 1)

    def test_reference_widens_the_length_corridor(self):
        self.cfg['reward']['formula'] = 'length + 0*beat'
        text = ' '.join(['word'] * 130)
        with self.fake():
            plain = core.score_group(self.cfg, 't', [text], target_words=100)['rewards'][0]
            ref = core.score_group(self.cfg, 't', [text], target_words=100, reference='h')['rewards'][0]
        self.assertLess(plain, 1.0)
        self.assertEqual(ref, 1.0)
        with self.fake():
            wide = core.score_group(self.cfg, 't3', [text], target_words=100, length_tolerance=0.5)['rewards'][0]
        self.assertEqual(wide, 1.0)


class TwoJudges(Sandbox):
    def setUp(self):
        super().setUp()
        self.cfg['judge']['fields'].pop()
        self.cfg['task_judge']['enabled'] = True
        self.cfg['reward']['formula'] = '(0.7*style/10 + 0.3*length) * min(1, task/7) - penalty'

    def test_off_task_text_loses_its_style_reward(self):
        with self.judge_returns({'on': {'style': 8, 'task': 9}, 'off': {'style': 10, 'task': 1}}) as post:
            out = core.score_group(core.validate_config(self.cfg), 'write a note', ['on', 'off'])
        self.assertEqual(post.call_count, 5)       # the low task score is asked once more
        self.assertAlmostEqual(out['rewards'][0], 0.7 * 0.8 + 0.3)
        self.assertAlmostEqual(out['rewards'][1], 1.0 / 7)
        self.assertEqual(out['items'][1]['scores'], {'style': 10.0, 'task': 1.0})
        self.assertEqual(out['items'][0]['reason'], 'r | r')

    def test_same_field_in_both_judges_is_rejected(self):
        self.cfg['judge']['fields'].append({'name': 'task', 'min': 0, 'max': 10, 'description': ''})
        with self.assertRaises(ValueError):
            core.validate_config(self.cfg)

    def test_task_set_is_scored_by_the_task_judge_with_the_assignment(self):
        path = Path(self.dir.name) / 'set.jsonl'
        path.write_text(json.dumps({'task': 'write about cats', 'text': 'cats', 'ok': True, 'kind': 'good'}) + '\n'
                        + json.dumps({'task': 'write about cats', 'text': 'dogs', 'ok': False, 'kind': 'off_topic'}) + '\n')
        rows = core.load_labeled(path, 10, 0, 100, 0)
        seen = []
        def post(url, headers, body, timeout=240):
            user = json.loads(body['messages'][1]['content'])
            seen.append(user['assignment'])
            return reply([{'id': 0, 'task': 9 if user['candidates'][0]['text'] == 'cats' else 1, 'reason': 'r'}])
        with patch.object(core, '_post', side_effect=post):
            out = core.evaluate_labeled(core.task_view(self.cfg), rows)
        self.assertEqual(seen, ['write about cats'] * 2)
        self.assertEqual((out['agreement'], out['auroc'], out['by_source']['off_topic']['correct']), (1.0, 1.0, 1))


class TemplateJudge(Sandbox):
    def test_one_object_reply_with_lists_becomes_counts(self):
        self.cfg['judge'].update(input='template', structured=False, user_template='T:{task}\nX:{text}',
                                 fields=[{'name': 'human', 'min': 1, 'max': 10, 'description': ''},
                                         {'name': 'tells', 'min': 0, 'max': 500, 'description': ''}])
        self.cfg['reward']['formula'] = 'human/10 - 0.1*tells'
        self.cfg['calibration'] = {'ai_when': 'human < 8', 'score_field': 'human', 'higher_is_ai': False}
        sent = []
        def post(url, headers, body, timeout=240):
            sent.append(body)
            content = 'ok {"tells": [{"pattern": "Em Dashes", "quote": "x"}], "lost_facts": [], "human": 7, "task": 9}'
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
        with patch.object(core, '_post', side_effect=post):
            out = core.score_group(core.validate_config(self.cfg), 'note', ['a text'])
        self.assertEqual(sent[0]['messages'][1]['content'], 'T:note\nX:a text')
        self.assertNotIn('response_format', sent[0])
        self.assertEqual(out['items'][0]['scores'], {'human': 7.0, 'tells': 1.0})
        self.assertAlmostEqual(out['rewards'][0], 0.6)
        self.assertIn('Em Dashes', out['items'][0]['reason'])

    def test_opus_preset_matches_cmp_prompt(self):
        self.cfg['task_judge']['enabled'] = True
        cfg = core.validate_config(core.merged(self.cfg, core.opus_slop_preset()))
        body = core.judge_body(cfg, 'TASK', ['TEXT'])
        self.assertIn('<task>TASK</task>', body['messages'][1]['content'])
        self.assertIn('<text>TEXT</text>', body['messages'][1]['content'])
        self.assertIn('compare against the task', body['messages'][1]['content'])
        self.assertIn('<rulebook>', body['messages'][0]['content'][0]['text'])


class Calibration(Sandbox):
    def test_agreement_and_auroc(self):
        rows = [{'text': 'h1', 'slop': False, 'words': 1, 'source': ''}, {'text': 'h2', 'slop': False, 'words': 1, 'source': ''},
                {'text': 'a1', 'slop': True, 'words': 1, 'source': ''}, {'text': 'a2', 'slop': True, 'words': 1, 'source': ''}]
        scores = {'h1': {'style': 9, 'task': 5}, 'h2': {'style': 4, 'task': 5}, 'a1': {'style': 2, 'task': 5}, 'a2': {'style': 3, 'task': 5}}
        with self.judge_returns(scores):
            out = core.evaluate_labeled(self.cfg, rows, repeats=2)
        self.assertEqual((out['agreement'], out['human_recall'], out['ai_recall'], out['auroc']), (0.75, 0.5, 1.0, 1.0))
        self.assertEqual((out['flip_rate'], len(out['misses'])), (0.0, 1))

    def test_auroc_ties(self):
        self.assertEqual(core.auroc([1, 1], [1, 0]), 0.75)


class Config(unittest.TestCase):
    def test_default_is_valid(self):
        core.validate_config(copy.deepcopy(core.DEFAULT_CONFIG))

    def test_rejects_broken_formula_and_reserved_names(self):
        cfg = copy.deepcopy(core.DEFAULT_CONFIG)
        cfg['reward']['formula'] = 'style +'
        with self.assertRaises(ValueError):
            core.validate_config(cfg)
        cfg = copy.deepcopy(core.DEFAULT_CONFIG)
        cfg['judge']['fields'].append({'name': 'length', 'min': 0, 'max': 1, 'description': ''})
        with self.assertRaises(ValueError):
            core.validate_config(cfg)


if __name__ == '__main__':
    unittest.main()


class Blind(Sandbox):
    def setUp(self):
        super().setUp()
        import blind
        self.blind = blind
        contrast = Path(self.dir.name) / 'contrast.jsonl'
        contrast.write_text(''.join(json.dumps(r) + '\n' for r in [
            {'text': 'human one', 'slop': False, 'task': 'q1', 'source': 'human-eli5'},
            {'text': 'model one', 'slop': True, 'task': 'q1', 'source': 'gen-opus'},
            {'text': 'human two', 'slop': False, 'task': 'q2', 'source': 'human-eli5'},
            {'text': 'model two', 'slop': True, 'task': 'q2', 'source': 'gen-flash'}]))
        patcher = patch.object(blind, 'CONTRAST', contrast)
        patcher.start()
        self.addCleanup(patcher.stop)
        (Path(self.dir.name) / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
            {'source': 'train', 'step': 1, 'ts': 100, 'task': 'x', 'answers': ['a'], 'items': [{'reward': 0.1}]},
            {'source': 'eval', 'ts': 200, 'task': 't', 'agent': 'pi', 'answers': ['early'], 'items': [{'reward': 0.4}]},
            {'source': 'eval', 'ts': 300, 'task': 't', 'agent': 'pi', 'answers': ['late'], 'items': [{'reward': 0.8}]}]))

    def test_human_pairs_hide_labels_and_count_human_wins(self):
        seen = set()
        for seed in range(2):
            pair = self.blind.next_pair('human', seed)
            self.assertNotIn('label', json.dumps(pair))
            human_side = 'left' if pair['left'].startswith('human') else 'right'
            self.blind.vote(pair['id'], human_side)
            seen.add(pair['id'])
        self.assertEqual(len(seen), 2)
        self.assertTrue(self.blind.next_pair('human', 5)['done'])
        cell = self.blind.summary()['human']
        self.assertEqual((cell['n'], cell['human_win_rate']), (2, 1.0))

    def test_steps_pairs_track_later_wins_and_judge_agreement(self):
        pair = self.blind.next_pair('steps', 0)
        late_side = 'left' if pair['left'] == 'late' else 'right'
        record = self.blind.vote(pair['id'], late_side)
        self.assertTrue(record['later_wins'])
        self.assertTrue(record['judge_agrees'])      # the judge gave the late answer the higher reward
        cell = self.blind.summary()['steps']
        self.assertEqual((cell['later_win_rate'], cell['judge_agreement']), (1.0, 1.0))

    def test_vote_needs_a_pending_pair_and_a_known_choice(self):
        pair = self.blind.next_pair('human', 0)
        with self.assertRaises(ValueError):
            self.blind.vote(pair['id'], 'maybe')
        self.blind.vote(pair['id'], 'both_bad')
        with self.assertRaises(ValueError):
            self.blind.vote(pair['id'], 'left')
