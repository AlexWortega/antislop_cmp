"""AReaL entry point for the probe: our Harbor rows as the dataset, HarborCoordinatorAgent as the workflow.

Usage (on the GPU box, from /workspace/areal with its venv):
  ANTISLOP_PROBE_ROWS=/workspace/lab/areal_probe/rows.json \
  python /workspace/lab/areal_probe/train.py --config /workspace/lab/areal_probe/config.yaml [key=value ...]

rows.json is {"train": [...], "val": [...]} written by make_rows.py (same rows vast_core.data_rows builds for verl).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import Dataset  # noqa: E402

from areal import PPOTrainer  # noqa: E402
from areal.api.cli_args import PPOConfig, load_expr_config  # noqa: E402


def flat(rows):
    # The agent reads task/agent/split/policy/group; messages is only a placeholder for AReaL's data path.
    return [{**r['extra_info'], 'messages': r['prompt']} for r in rows]


def main(argv):
    config, _ = load_expr_config(argv, PPOConfig)
    rows = json.loads(Path(os.environ['ANTISLOP_PROBE_ROWS']).read_text())
    train = Dataset.from_list(flat(rows['train']))
    valid = Dataset.from_list(flat(rows['val'])) if rows.get('val') else None
    kwargs = {'group_size': config.gconfig.n_samples,
              'eval_group_size': (config.eval_gconfig or config.gconfig).n_samples}
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        trainer.train(workflow='agent.HarborCoordinatorAgent', eval_workflow='agent.HarborCoordinatorAgent',
                      workflow_kwargs=kwargs, eval_workflow_kwargs=kwargs)


if __name__ == '__main__':
    main(sys.argv[1:])
