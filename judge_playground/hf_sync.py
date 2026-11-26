"""LoRA adapters out of verl checkpoints, private backups on Hugging Face, and publishing.

Runs on the GPU box with the verl environment, or here with any Python that has huggingface_hub (publishing an
adapter that is already exported). The token is read from a file, never from the command line.

    python hf_sync.py export  --checkpoint .../global_step_N/actor --out DIR --base-id ID --revision REV
    python hf_sync.py backup  --folder DIR --repo ID --name step_0010 --keep 2
    python hf_sync.py fetch   --repo ID --name step_0010 --out DIR
    python hf_sync.py publish --folder DIR --repo ID [--public] [--merged --base /workspace/model] --card README.md

Exit codes: 0 done, 3 the Hub refused for lack of storage (the caller pauses backups, training goes on), 1 other.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

TOKEN_FILE = os.environ.get('HF_TOKEN_FILE', '/root/.hf_token')
# Only messages about storage. An auth or rate-limit error must never trigger deleting old backups.
QUOTA = re.compile(r'storage|quota|exceeded your|too large|insufficient space|payload too large|\b413\b|\b507\b', re.I)


def token():
    path = Path(TOKEN_FILE).expanduser()
    if not path.exists():
        raise SystemExit(f'No Hugging Face token in {path}')
    return path.read_text().strip()


def say(**event):
    print(json.dumps({'hf': True, 'time': round(time.time()), **event}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- export

def export(args):
    """verl saves LoRA-only FSDP state dicts; turn one into a PEFT adapter that vLLM and PEFT load directly."""
    import torch
    from safetensors.torch import save_file
    actor = Path(args.checkpoint)
    shards = sorted(actor.glob('model_world_size_*_rank_*.pt'))
    if len(shards) != 1:
        raise SystemExit(f'Expected one model shard in {actor} (trainer on one GPU), found {len(shards)}')
    state = torch.load(shards[0], map_location='cpu', weights_only=False)
    weights, modules = {}, set()
    for name, value in state.items():
        if 'lora_' not in name:
            continue
        tensor = getattr(value, '_local_tensor', value)       # DTensor from FSDP2 on one rank holds the full tensor
        key = name.replace('.default.weight', '.weight')
        if not key.startswith('base_model.model.'):
            key = 'base_model.model.' + key
        weights[key] = tensor.detach().to(torch.bfloat16).contiguous()
        modules.add(key.split('.')[-3])
    if not weights:
        raise SystemExit('No LoRA tensors in the checkpoint')
    meta_path = actor / 'lora_train_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    any_a = next(v for k, v in weights.items() if 'lora_A' in k)
    rank = int(meta.get('r') or any_a.shape[0])
    if any_a.shape[0] != rank:
        raise SystemExit(f'LoRA rank in weights ({any_a.shape[0]}) differs from metadata ({rank})')
    out = Path(args.out)
    temp = out.with_name(out.name + '.tmp')
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    save_file(weights, str(temp / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    config = {'peft_type': 'LORA', 'task_type': meta.get('task_type') or 'CAUSAL_LM', 'r': rank, 'lora_alpha': int(meta.get('lora_alpha') or rank),
              'lora_dropout': 0.0, 'bias': 'none', 'target_modules': sorted(modules), 'base_model_name_or_path': args.base_id,
              'revision': args.revision, 'inference_mode': True, 'fan_in_fan_out': False}
    (temp / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.rmtree(out, ignore_errors=True)
    temp.rename(out)
    say(event='exported', tensors=len(weights), mib=round(sum(v.numel() * 2 for v in weights.values()) / 2 ** 20, 1), rank=rank,
        alpha=config['lora_alpha'], modules=sorted(modules))


# ---------------------------------------------------------------- backup

def free_space(api, repo, keep_names):
    """Delete old backups and squash history: deleted LFS files keep using storage until the history is gone."""
    names = sorted({entry.path.split('/')[0] for entry in api.list_repo_tree(repo) if entry.path.startswith('step_')})
    for name in names:
        if name not in keep_names:
            api.delete_folder(path_in_repo=name, repo_id=repo, commit_message=f'Drop old backup {name}')
            say(event='deleted', name=name)
    api.super_squash_history(repo_id=repo, commit_message='Squash backups to free storage')
    say(event='squashed', kept=sorted(set(names) & set(keep_names)))


def existing_backups(api, repo):
    return sorted({entry.path.split('/')[0] for entry in api.list_repo_tree(repo) if entry.path.startswith('step_')})


def backup(args):
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(args.repo, private=True, exist_ok=True)
    if not api.repo_info(args.repo).private:
        raise SystemExit(f'Refusing to back up into {args.repo}: the repository is public')
    attempt = 0
    while True:
        try:
            api.upload_folder(repo_id=args.repo, folder_path=args.folder, path_in_repo=args.name, commit_message=f'Backup {args.name}')
            break
        except Exception as exc:  # noqa: BLE001 - classified below
            text = f'{type(exc).__name__}: {exc}'
            if not QUOTA.search(text) or attempt:
                if QUOTA.search(text):
                    say(event='quota', error=text[:400])
                    sys.exit(3)
                raise
            # Out of storage: keep only the newest backup, squash, try once more.
            say(event='quota_retry', error=text[:300])
            older = existing_backups(api, args.repo)
            free_space(api, args.repo, set(older[-1:]))
            attempt += 1
    api.upload_file(repo_id=args.repo, path_or_fileobj=json.dumps({'name': args.name, 'time': time.time()}).encode(), path_in_repo='LATEST.json',
                    commit_message=f'Latest backup: {args.name}')
    names = existing_backups(api, args.repo)
    if len(names) > args.keep:
        free_space(api, args.repo, set(names[-args.keep:]))
    say(event='backed_up', repo=args.repo, name=args.name)


def fetch(args):
    """One backed-up adapter into --out, for a new GPU box to continue from."""
    from huggingface_hub import snapshot_download
    temp = Path(args.out + '.incoming')
    shutil.rmtree(temp, ignore_errors=True)
    snapshot_download(args.repo, allow_patterns=[args.name + '/*'], local_dir=str(temp), token=token())
    if not (temp / args.name / 'adapter_model.safetensors').exists():
        raise SystemExit(f'{args.repo} has no adapter in {args.name}')
    shutil.rmtree(args.out, ignore_errors=True)
    (temp / args.name).rename(args.out)
    shutil.rmtree(temp, ignore_errors=True)
    say(event='fetched', repo=args.repo, name=args.name)


# ---------------------------------------------------------------- publish

def merge(folder, base, out):
    """Base weights plus the adapter, on CPU so training on the GPUs is not disturbed."""
    import torch
    from peft import PeftModel
    import transformers
    loader = getattr(transformers, 'AutoModelForImageTextToText', None) or transformers.AutoModelForCausalLM
    say(event='merging', base=str(base))
    model = loader.from_pretrained(base, dtype=torch.bfloat16, device_map='cpu', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, folder).merge_and_unload()
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, max_shard_size='5GB', safe_serialization=True)
    for item in Path(base).iterdir():
        if item.is_file() and not item.name.endswith(('.safetensors', '.bin', '.pt')) and not (Path(out) / item.name).exists():
            shutil.copy2(item, Path(out) / item.name)
    say(event='merged', out=str(out))


def publish(args):
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    folder = Path(args.folder)
    if args.merged:
        if not args.base:
            raise SystemExit('Merged publishing needs --base, the local copy of the base model')
        target = folder.with_name(folder.name + '-merged')
        merge(folder, args.base, target)
        folder = target
    if args.card:
        shutil.copy2(args.card, folder / 'README.md')
    say(event='uploading', repo=args.repo, merged=args.merged, public=args.public)
    if args.merged:
        api.upload_large_folder(repo_id=args.repo, folder_path=str(folder), repo_type='model')
        shutil.rmtree(folder, ignore_errors=True)
    else:
        api.upload_folder(repo_id=args.repo, folder_path=str(folder), commit_message='Publish adapter')
    say(event='published', repo=args.repo, url=f'https://huggingface.co/{args.repo}', public=args.public)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('export')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--base-id', required=True)
    p.add_argument('--revision', default='')
    p = sub.add_parser('backup')
    p.add_argument('--folder', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--keep', type=int, default=2)
    p = sub.add_parser('fetch')
    p.add_argument('--repo', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('publish')
    p.add_argument('--folder', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--public', action='store_true')
    p.add_argument('--merged', action='store_true')
    p.add_argument('--base', default='')
    p.add_argument('--card', default='')
    args = parser.parse_args()
    {'export': export, 'backup': backup, 'fetch': fetch, 'publish': publish}[args.op](args)


if __name__ == '__main__':
    main()
