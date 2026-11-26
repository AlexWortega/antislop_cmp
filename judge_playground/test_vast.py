import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lab_core as core
import vast_core as vast


def config():
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg['judge']['api_key_env'] = cfg['task_judge']['api_key_env'] = ''
    return cfg


class Offers(unittest.TestCase):
    def offer(self, **over):
        base = {'id': 1, 'num_gpus': 2, 'gpu_ram': 143000, 'gpu_name': 'H200', 'cuda_max_good': 13.4, 'reliability2': 0.99, 'inet_up': 900,
                'inet_down': 900, 'disk_space': 800, 'dph_total': 7.0, 'dph_base': 6.9, 'storage_cost': 0.2, 'cpu_ram': 500000}
        return {**base, **over}

    def test_filters_and_orders_by_price(self):
        gpu = config()['vast']['gpu']
        offers = [self.offer(id=1, dph_total=8.0), self.offer(id=2, dph_total=7.0), self.offer(id=3, num_gpus=1),
                  self.offer(id=4, gpu_name='A100'), self.offer(id=5, reliability2=0.9), self.offer(id=6, dph_total=20)]
        self.assertEqual([o['id'] for o in vast.pick_offers(offers, gpu)], [2, 1])

    def test_agent_host_needs_a_vm_with_cpus(self):
        agents = config()['vast']['agents']
        base = {'vms_enabled': True, 'cpu_cores_effective': 48, 'cpu_ram': 128000, 'reliability2': 0.99, 'disk_space': 500,
                'inet_up': 800, 'inet_down': 800, 'dph_total': 0.4, 'dph_base': 0.4, 'storage_cost': 0.1, 'machine_id': 7}
        offers = [{**base, 'id': 1}, {**base, 'id': 2, 'vms_enabled': False}, {**base, 'id': 3, 'cpu_cores_effective': 8},
                  {**base, 'id': 4, 'dph_total': 3.0}, {**base, 'id': 5, 'machine_id': 9, 'dph_total': 0.3, 'dph_base': 0.3}]
        self.assertEqual([o['id'] for o in vast.pick_host_offers(offers, agents)], [5, 1])
        self.assertEqual([o['id'] for o in vast.pick_host_offers(offers, agents, exclude={9})], [1])

    def test_endpoints_prefer_proxy_then_direct(self):
        info = {'ssh_host': 'ssh5.vast.ai', 'ssh_port': 1234, 'public_ipaddr': '1.2.3.4 ', 'ports': {'22/tcp': [{'HostPort': '40022'}]}}
        self.assertEqual(vast.endpoints(info), [('ssh5.vast.ai', 1234), ('1.2.3.4', 40022)])

    def test_price_bound_covers_disk_and_margin(self):
        self.assertAlmostEqual(vast.hourly({'dph_total': 3.0, 'dph_base': 3.0, 'storage_cost': 0.72}, 200), (3.0 + 0.2) * 1.1)


class Plan(unittest.TestCase):
    def setUp(self):
        self.rows = [{'task': f't{i}', 'split': 'train'} for i in range(20)] + [{'task': f'h{i}', 'split': 'heldout'} for i in range(5)]
        self.train = {**config()['vast']['train'], 'steps': 3, 'prompts_per_step': 4, 'eval_tasks': 3}

    def test_rows_follow_steps_and_never_touch_heldout(self):
        train, val = vast.data_rows(self.rows, self.train)
        self.assertEqual((len(train), len(val)), (12, 3))
        self.assertTrue(all(r['extra_info']['task'].startswith('t') and r['extra_info']['split'] == 'train' for r in train))
        self.assertTrue(all(r['extra_info']['task'].startswith('h') and r['extra_info']['split'] == 'dev' for r in val))
        self.assertEqual(len({r['extra_info']['task'] for r in train}), 12)
        self.assertTrue(train[4]['extra_info']['group'].startswith('train-s0002-'))
        self.assertEqual({r['extra_info']['agent'] for r in train}, {'hermes', 'pi'})

    def test_logprob_gap_guard_needs_three_bad_steps(self):
        runner = vast.Runner.__new__(vast.Runner)
        runner.v, runner.fatal = config()['vast'], None
        healthy = [{'step': i, 'actor/ppo_kl': k} for i, k in enumerate([0.0004, 0.001, 0.0022])]
        runner.check_logprob_gap(healthy + [{'step': 3, 'actor/ppo_kl': 0.15}])  # one spike: median still healthy
        self.assertIsNone(runner.fatal)
        runner.check_logprob_gap([{'step': i, 'actor/ppo_kl': k} for i, k in enumerate([0.14, 0.18, 0.07])])
        self.assertIn('Logprob mismatch', runner.fatal)
        self.assertIn('Logprob mismatch', vast.FATAL)

    def test_async_and_sync_use_different_verl_entrypoints(self):
        v = config()['vast']
        run = vast.scripts(v)['native/run_verl.sh']
        self.assertIn('fully_async_policy.fully_async_main', run)
        self.assertIn('actor_rollout_ref.model.lora.merge=True', run)
        self.assertIn('actor_rollout_ref.rollout.multi_turn.enable=True', run)
        self.assertIn('async_training.require_batches=2', run)
        self.assertIn('use_remove_padding=True', run)
        self.assertNotIn('mtp.enable', run)
        self.assertIn('actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768', run)  # packing on by default
        self.assertNotIn('actor.use_dynamic_bsz=False', run)
        v['train']['micro_tokens'] = 0
        single = vast.scripts(v)['native/run_verl.sh']
        self.assertIn('actor_rollout_ref.actor.use_dynamic_bsz=False', single)
        v['serve']['mtp_tokens'] = 2
        run = vast.scripts(v)['native/run_verl.sh']
        self.assertIn('actor_rollout_ref.model.mtp.enable_rollout=True', run)
        self.assertIn('actor_rollout_ref.model.mtp.num_speculative_tokens=2', run)
        self.assertIn('rollout.total_rollout_steps=1380', run)
        self.assertIn('trainer.total_training_steps=150', run)
        self.assertIn('trainer.n_gpus_per_node=1', run)
        v['train'].update(mode='sync', extra_overrides='trainer.seed=7')
        v['gpu']['num_gpus'] = 1
        v['serve']['gpu_util'] = 0.5
        run = vast.scripts(vast.validate_vast(v))['native/run_verl.sh']
        self.assertIn('verl.trainer.main_ppo', run)
        self.assertNotIn('async_training', run)
        self.assertNotIn('lora.merge', run)
        self.assertIn('trainer.total_training_steps=150', run)
        self.assertIn('trainer.seed=7', run)

    def test_convergence_knobs_reach_verl(self):
        v = config()['vast']
        run = vast.scripts(vast.validate_vast(v))['native/run_verl.sh']
        for part in ('ppo_mini_batch_size=4', 'use_kl_loss=False', 'norm_adv_by_std_in_grpo=False', 'bypass_mode=True',
                     'clip_ratio_high=0.28', 'rollout.temperature=1.0'):
            self.assertIn(part, run)
        v['train'].update(kl_coef=0.01, updates_per_step=1)
        run = vast.scripts(vast.validate_vast(v))['native/run_verl.sh']
        self.assertIn('use_kl_loss=True', run)
        self.assertIn('ppo_mini_batch_size=8', run)
        v['train']['updates_per_step'] = 3
        with self.assertRaises(ValueError):
            vast.validate_vast(v)

    def test_restore_offset_skips_trained_steps(self):
        train, _ = vast.data_rows(self.rows, self.train, offset=2)
        self.assertEqual([r['extra_info']['group'][:11] for r in train], ['train-s0003'] * 4)
        v = config()['vast']
        self.assertIn('rollout.total_rollout_steps=460', vast.scripts(v, offset=100)['native/run_verl.sh'])

    def test_scripts_cover_both_machines(self):
        compile(vast.GUARD, 'guard.py', 'exec')
        setup = vast.agent_setup_script()
        for line in ('GatewayPorts clientspecified', 'MaxSessions 256', 'antislop-pilot-hermes:local', 'import harbor', 'SETUP_DONE'):
            self.assertIn(line, setup)
        s = vast.scripts(config()['vast'])
        self.assertIn('-R "$BIND:18974:127.0.0.1:9001"', s['to_agents.sh'])
        self.assertIn('download retry', s['bootstrap.sh'])

    def test_model_card_names_base_and_judges(self):
        cfg = config()
        card = vast.model_card(cfg, {'groups': [{'source': 'eval', 'reward': 0.5}]}, 40, merged=False)
        self.assertIn('base_model: rawmodels/Qwen3.8-27B-antislop', card)
        self.assertIn('library_name: peft', card)
        self.assertIn('step 40', card)
        self.assertIn(cfg['judge']['model'], card)
        self.assertNotIn('library_name', vast.model_card(cfg, {}, 40, merged=True))

    def test_validation_catches_impossible_setups(self):
        v = config()['vast']
        v['gpu']['num_gpus'] = 1
        with self.assertRaises(ValueError):
            vast.validate_vast(v)
        v['train']['mode'] = 'sync'
        with self.assertRaises(ValueError):          # one shared GPU cannot give vLLM 85% of memory
            vast.validate_vast(v)
        v['serve']['gpu_util'] = 0.5
        vast.validate_vast(v)


class Helpers(unittest.TestCase):
    def test_log_lines_survive_split_multibyte_characters(self):
        text = 'строка один\nстрока два\nхвост'.encode()
        lines, rest = vast.split_log(b'', text[:9])
        more, rest = vast.split_log(rest, text[9:])
        self.assertEqual(lines + more, ['строка один', 'строка два'])
        self.assertEqual(rest.decode(), 'хвост')

    def test_known_errors_get_a_hint(self):
        self.assertIn('памяти GPU', vast.explain('torch.OutOfMemoryError: CUDA out of memory'))
        self.assertIn('бюджет судьи', vast.explain('Budget of $20.00 is spent'))
        self.assertEqual(vast.explain('something new'), '')

    def test_remote_commands_never_kill_by_their_own_path(self):
        """pkill -f <path> also matches the remote shell whose command line contains <path>, killing our own SSH session."""
        import inspect
        source = inspect.getsource(vast)
        self.assertNotIn('pgrep -f', source)
        for line in source.splitlines():
            if 'pkill' in line:
                self.assertTrue("'[V]LLM::'" in line or "'[h]ermes_collect.py'" in line, line)


class HfQuota(unittest.TestCase):
    """A full Hub deletes old backups and squashes history once, then gives up with exit code 3; other errors never delete."""

    def setUp(self):
        import hf_sync
        self.hf = hf_sync
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        token = Path(self.tmp.name) / 'token'
        token.write_text('t')
        patcher = patch.object(hf_sync, 'TOKEN_FILE', str(token))
        patcher.start()
        self.addCleanup(patcher.stop)

    def api(self, failures):
        class Entry:
            def __init__(self, path):
                self.path = path

        class Api:
            def __init__(self, token=None):
                self.calls, self.files = [], ['step_0010/a', 'step_0020/a', 'step_0030/a']

            def create_repo(self, *a, **k):
                pass

            def repo_info(self, repo):
                return type('Info', (), {'private': True})()

            def upload_folder(self, **k):
                self.calls.append('upload')
                if failures:
                    raise RuntimeError(failures.pop(0))
                self.files.append(k['path_in_repo'] + '/a')

            def upload_file(self, **k):
                pass

            def list_repo_tree(self, repo):
                return [Entry(f) for f in self.files]

            def delete_folder(self, path_in_repo, **k):
                self.calls.append('delete ' + path_in_repo)
                self.files = [f for f in self.files if not f.startswith(path_in_repo + '/')]

            def super_squash_history(self, **k):
                self.calls.append('squash')
        return Api

    def run_backup(self, failures):
        import types
        api = self.api(failures)
        holder = {}
        fake = types.SimpleNamespace(HfApi=lambda token=None: holder.setdefault('api', api(token)))
        with patch.dict('sys.modules', {'huggingface_hub': fake}):
            args = types.SimpleNamespace(repo='me/r', folder='f', name='step_0040', keep=2)
            try:
                self.hf.backup(args)
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, holder['api']

    def test_quota_frees_space_and_retries(self):
        code, api = self.run_backup(['HfHubHTTPError: 403 You have exceeded your private storage limit'])
        self.assertEqual(code, 0)
        self.assertEqual(api.calls[:5], ['upload', 'delete step_0010', 'delete step_0020', 'squash', 'upload'])
        self.assertEqual(sorted({f.split('/')[0] for f in api.files}), ['step_0030', 'step_0040'])

    def test_quota_twice_exits_3(self):
        code, api = self.run_backup(['storage limit', 'storage limit'])
        self.assertEqual(code, 3)

    def test_auth_error_never_deletes(self):
        with self.assertRaises(RuntimeError):
            self.run_backup(['401 Unauthorized: invalid token'])


class Rewards(unittest.TestCase):
    """The coordinator turns finished trials of one group into rewards through the lab judges."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for target in ((core, 'STATE', root / 'state'), (vast, 'RUNS', root / 'runs')):
            patcher = patch.object(*target)
            patcher.start()
            self.addCleanup(patcher.stop)
        (root / 'state').mkdir()
        tasks = root / 'rows.jsonl'
        tasks.write_text(json.dumps({'task': 'write_a', 'split': 'train', 'prompt': 'Write a note', 'target_words': 4, 'lang': 'en',
                                     'allow_markdown': False}) + '\n')
        cfg = config()
        cfg['vast']['local']['tasks'] = str(tasks)
        self.runner = vast.Runner(vast.new_run(cfg))

    def trial(self, uid, text):
        out = self.runner.dir / 'rollouts' / 'g' / uid
        (out / 'trials' / ('trial-' + uid) / 'verifier').mkdir(parents=True)
        (out / 'trials' / ('trial-' + uid) / 'verifier/answer.txt').write_text(text)
        return {'trial_id': 'trial-' + uid, 'grade_outcome': {'valid': True, 'reward': 1 if text else 0}}, out

    def test_group_is_scored_once_and_failed_trials_get_zero_advantage(self):
        coordinator = vast.Coordinator(self.runner)
        texts = {'u1': 'a plain human note', 'u2': 'delve into the tapestry', 'u3': None}

        def run_trial(body):
            if texts[body['uid']] is None:
                raise ValueError('container died')
            return self.trial(body['uid'], texts[body['uid']])

        def post(url, headers, body, timeout=240):
            user = json.loads(body['messages'][1]['content'])
            good = user['candidates'][0]['text'].startswith('a plain')
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
                {'results': [{'id': 0, 'style': 9 if good else 2, 'task': 10, 'reason': 'r'}]})}}], 'usage': {'cost': 0.01}}

        results = {}
        with patch.object(coordinator, 'run_trial', side_effect=run_trial), patch.object(core, '_post', side_effect=post) as judge_api:
            import threading
            threads = [threading.Thread(target=lambda u=u: results.update({u: coordinator.process(
                {'uid': u, 'group': 'train-s0003-hermes-write_a', 'task': 'write_a', 'agent': 'hermes', 'split': 'train', 'policy': 'lab', 'size': 3})}))
                for u in texts]
            [t.start() for t in threads]
            [t.join(20) for t in threads]
        self.assertGreater(results['u1']['reward'], results['u2']['reward'])
        self.assertTrue(results['u3']['infra_failed'])
        self.assertAlmostEqual(results['u3']['reward'], (results['u1']['reward'] + results['u2']['reward']) / 2)
        logged = [json.loads(x) for x in (core.STATE / 'rewards.jsonl').read_text().splitlines()]
        self.assertEqual((len(logged), logged[0]['step'], logged[0]['source']), (1, 3, 'train'))
        self.assertEqual(self.runner.state['groups'][0]['step'], 3)
        # Two answers x two judges: each judged once as its trial ended; scoring the group reused those calls (8 if not).
        self.assertEqual(judge_api.call_count, 4)

    def test_remote_trial_runs_detached_and_is_copied_back(self):
        self.runner.cfg['vast']['agents']['where'] = 'vast'
        self.runner.state['agents'] = {'id': 1, 'ssh_host': 'h', 'ssh_port': 22}
        coordinator = vast.Coordinator(self.runner)
        commands, polls = [], iter(['', '0'])

        def ssh(command, **kw):
            commands.append(command)
            return next(polls) if command.startswith('cat ') else ''

        def rsync(local, remote, upload=True, exclude=(), who='gpu'):
            self.assertEqual((upload, who), (False, 'agents'))
            Path(local).mkdir(parents=True, exist_ok=True)
            (Path(local) / 'outcomes.jsonl').write_text('{}\n')

        body = {'uid': 'u1', 'group': 'g', 'task': 'write_a', 'agent': 'pi', 'split': 'train', 'policy': 'lab', 'size': 2}
        out = self.runner.dir / 'rollouts' / 'g' / 'u1'
        with patch.object(self.runner, 'ssh', side_effect=ssh), patch.object(self.runner, 'rsync', side_effect=rsync), patch.object(vast.time, 'sleep'):
            self.assertEqual(coordinator.execute(body, out, 0), 0)
        self.assertIn('setsid nohup', commands[0])
        self.assertIn('http://host.docker.internal:18974/session/u1/v1', commands[0])
        self.assertIn('/root/lab/tasks/pi', commands[0])
        self.assertTrue((out / 'outcomes.jsonl').exists())

    def test_tunnel_does_not_share_the_multiplexed_connection(self):
        self.runner.state['instance'] = {'id': 1, 'ssh_host': 'h', 'ssh_port': 22}
        base, tunnel = self.runner.ssh_base(), self.runner.tunnel_base()
        self.assertIn('ControlMaster=auto', base)
        self.assertIn('ControlMaster=no', tunnel)
        self.assertNotIn('ControlMaster=auto', tunnel)
        self.assertEqual(base[-2:], tunnel[-2:])

    def test_restore_prefers_the_newest_saved_adapter(self):
        self.runner.state.update(instance={'id': 9, 'ssh_host': 'h', 'ssh_port': 22}, destroyed=[{'key': 'instance', 'id': 8}],
                                 last_downloaded=20, hf_backed=30, hf_repo='me/r')
        jobs = []
        with patch.object(self.runner, 'latest_checkpoint', return_value=0), patch.object(self.runner, 'ssh', return_value=''), \
                patch.object(self.runner, 'job', side_effect=lambda name, command, *a, **k: jobs.append(command)):
            self.runner.plan_restore()
        self.assertEqual((self.runner.offset(), self.runner.state['restore_path']), (30, '/workspace/lab/native/restore'))
        self.assertIn('fetch --repo me/r --name step_0030', jobs[0])

    def test_stop_request_refuses_new_trials(self):
        coordinator = vast.Coordinator(self.runner)
        (self.runner.dir / 'STOP').touch()
        with self.assertRaises(RuntimeError):
            coordinator.process({'uid': 'u', 'group': 'g', 'task': 'write_a', 'agent': 'hermes', 'split': 'train', 'policy': 'lab', 'size': 2})


if __name__ == '__main__':
    unittest.main()
