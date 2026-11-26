"""Runs on the agent host: prepare the GPU rental for a fresh run, then supervise training.

The laptop has already started gpu_guard on the GPU, so the GPU is never left without a way to stop
itself. Any failure here touches STOP, which makes agent_guard tell the GPU to destroy itself.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'qwen_scaffold_longrun'
sys.path.insert(0, str(OLD))
import run_loop as t  # noqa: E402

t.ROOT = ROOT
PYTHON = '/workspace/longrun/vendor/verl/.venv/bin/python'
EVENT = threading.Event()
PHASE = 'setup'
FILES = ('native_agent.py', 'native_compat.py', 'agent.yaml', 'run_verl.sh', 'reward_fn.py', 'serve.key',
         'train.parquet', 'val.parquet')


def phase(value):
    global PHASE
    PHASE = value
    temp = ROOT / 'progress.tmp'
    temp.write_text(json.dumps({'phase': value, 'updated_at': time.time()}))
    temp.replace(ROOT / 'progress.json')
    print(value, flush=True)


def heartbeat():
    while not EVENT.is_set():
        temp = ROOT / 'heartbeat.tmp'
        temp.write_text(json.dumps({'phase': PHASE, 'time': time.time(), 'pid': os.getpid()}))
        temp.replace(ROOT / 'heartbeat.json')
        EVENT.wait(20)


def preflight_checks(resolved):
    """Lines that must appear in the resolved trainer config before any GPU time is spent training."""
    lines = {line.strip() for line in resolved.splitlines()}
    expected = {'lr': f"lr: {float(os.environ['ANTISLOP_LR']):.1e}", 'rank': 'lora_rank: 256',
                # YAML quotes the key n, since a bare n reads as a boolean.
                'n': "'n': " + os.environ['ANTISLOP_GROUP_SIZE'], 'batch': 'train_batch_size: 8'}
    return [name for name, line in expected.items() if line not in lines]


def main():
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        phase('uploading_code')
        t.remote('mkdir -p /workspace/native/output/checkpoints /workspace/native/audit /workspace/longrun/output /workspace/longrun/vendor/verl')
        t.rsync('/workspace/longrun/vendor/verl/', str(ROOT.parent / 'qwen_verl_pilot/vendor/verl') + '/', upload=True,
                exclude=('.git', '.venv', '__pycache__'))
        t.rsync('/workspace/longrun/bootstrap_gpu.sh', OLD / 'bootstrap_gpu.sh', upload=True)
        for name in FILES:
            t.rsync('/workspace/native/' + name, ROOT / name, upload=True)
        # Validation groups have val_kwargs.n samples per prompt (run_verl.sh), not the training group size.
        t.remote('chmod 600 /workspace/native/serve.key; echo ' + os.environ['ANTISLOP_GROUP_SIZE'] + ' > /workspace/native/group_size; echo '
                 + os.environ.get('ANTISLOP_VAL_GROUP_SIZE', '4') + ' > /workspace/native/val_group_size')
        t.remote('nohup bash /workspace/longrun/bootstrap_gpu.sh > /workspace/longrun/output/bootstrap.log 2>&1 < /dev/null &')
        phase('waiting_bootstrap')
        for _ in range(360):
            if t.remote('test -f /workspace/longrun/BOOTSTRAP_DONE && echo ready || true').strip() == 'ready':
                break
            time.sleep(10)
        else:
            raise RuntimeError('GPU bootstrap deadline')
        phase('native_verl_preflight')
        t.remote('bash /workspace/native/run_verl.sh --cfg job > /workspace/native/output/resolved.yaml', timeout=300)
        resolved = t.remote('cat /workspace/native/output/resolved.yaml')
        (ROOT / 'cloud' / 'resolved.yaml').write_text(resolved)
        missing = preflight_checks(resolved)
        if missing:
            raise RuntimeError('Resolved config differs from the plan: ' + ', '.join(missing))
        EVENT.set()
        code = subprocess.run([sys.executable, '-u', str(ROOT / 'rejoin_runner.py')], env=os.environ).returncode
    except BaseException as exc:
        phase('failed')
        (ROOT / 'launch_failure.json').write_text(json.dumps({'error': type(exc).__name__ + ': ' + str(exc),
                                                              'phase': PHASE, 'time': time.time()}))
        code = 1
    finally:
        EVENT.set()
        (ROOT / 'STOP').touch()
    sys.exit(code)


if __name__ == '__main__':
    main()
