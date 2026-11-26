"""Private machine upload helper. Token arrives on stdin and is never saved."""
import sys
from pathlib import Path
from huggingface_hub import HfApi

if __name__=='__main__':
    token=sys.stdin.readline().strip()
    assert token.startswith('hf_')
    api=HfApi(token=token)
    repo='rawmodels/Qwen3.8-27B-antislop-rft1-GGUF'
    api.create_repo(repo,repo_type='model',private=False,exist_ok=True)
    root=Path('/home/alexw/antislop-space-20261012')
    info=api.upload_file(path_or_fileobj=root/'Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf',
        path_in_repo='Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf',repo_id=repo,
        commit_message='Add rft1 Q4_K_M GGUF with a rewrite importance matrix')
    print('MODEL_REVISION',info.oid,flush=True)
