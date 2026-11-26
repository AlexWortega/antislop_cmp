"""Replace the public alias with rft1 and its card in one HF commit.

Dry run by default. Uses existing Hugging Face authentication.
Run with the Python interpreter used by the installed `hf` command.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

ROOT = Path(__file__).resolve().parent
SOURCE = "rawmodels/Qwen3.8-27B-antislop-rft1"
DESTINATION = "rawmodels/Qwen3.8-27B-antislop"


def identity(file):
    return {
        "blob_id": file.blob_id,
        "size": file.size,
        "sha256": file.lfs.sha256 if file.lfs else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    api.auth_check(SOURCE, repo_type="model")
    api.auth_check(DESTINATION, repo_type="model", write=True)
    source = api.model_info(SOURCE, files_metadata=True)
    destination = api.model_info(DESTINATION, files_metadata=True)
    files = {s.rfilename: s for s in source.siblings if "/" not in s.rfilename and s.rfilename != "README.md"}
    index = json.loads(Path(hf_hub_download(SOURCE, "model.safetensors.index.json", revision=source.sha)).read_text())
    shards = set(index["weight_map"].values())
    if not shards or not shards <= files.keys():
        raise ValueError("Source is missing shards referenced by its weight index")
    if index["metadata"]["total_size"] < 40_000_000_000:
        raise ValueError("Expected a full merged 27B checkpoint")
    config = json.loads(Path(hf_hub_download(SOURCE, "config.json", revision=source.sha)).read_text())
    if "Qwen3_5ForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Unexpected source architecture")
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if name not in files:
            raise ValueError(f"Missing source file: {name}")

    # Replace root model assets; preserve unrelated files in the destination.
    obsolete = sorted(s.rfilename for s in destination.siblings if "/" not in s.rfilename
                      and s.rfilename not in files and s.rfilename != "README.md"
                      and (s.rfilename.startswith("model") or s.rfilename == "crc32.txt"))
    plan = {"source": SOURCE, "source_revision": source.sha,
            "destination": DESTINATION, "destination_revision": destination.sha,
            "backup_tag": "pre-rft1-" + destination.sha[:12],
            "source_files": {name: identity(file) for name, file in sorted(files.items())},
            "delete_obsolete": obsolete,
            "card_sha256": hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest()}
    (ROOT / "promotion_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: v for k, v in plan.items() if k != "source_files"}, indent=2))
    if not args.apply:
        return

    operations = [CommitOperationCopy(name, name, src_revision=source.sha,
                                      src_repo_id=SOURCE, src_repo_type="model") for name in files]
    operations += [CommitOperationDelete(name) for name in obsolete]
    operations += [CommitOperationAdd("README.md", ROOT / "README.md")]
    operations += [CommitOperationAdd("evaluations/" + p.name, p)
                   for p in sorted((ROOT / "evaluations").iterdir()) if p.is_file()]
    operations += [CommitOperationAdd("release_manifest.json", json.dumps(plan, indent=2).encode())]
    api.create_tag(DESTINATION, tag=plan["backup_tag"], revision=destination.sha, exist_ok=True)
    commit = api.create_commit(DESTINATION, operations=operations, parent_commit=destination.sha,
                               commit_message="Promote rft1; update model card and comparison tables")
    published = api.model_info(DESTINATION, revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in published.siblings}
    for name, file in files.items():
        if name not in remote or identity(remote[name]) != identity(file):
            raise RuntimeError(f"Published file differs from rft1: {name}; backup: {plan['backup_tag']}")
    card = Path(hf_hub_download(DESTINATION, "README.md", revision=commit.oid)).read_bytes()
    if hashlib.sha256(card).hexdigest() != plan["card_sha256"]:
        raise RuntimeError("Published card differs from the prepared card")
    receipt = {"commit": commit.oid, "url": commit.commit_url, "verified_source_files": len(files),
               "source_revision": source.sha, "backup_tag": plan["backup_tag"]}
    (ROOT / "published.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
