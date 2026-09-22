"""Populate the Hugging Face cache and write a commit-pinned manifest."""
import argparse
import json
from pathlib import Path
from pretraining_data.models import manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-output", type=Path, default=Path("models.lock.json"))
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    locked = {}
    for repo, entry in manifest().items():
        revision = api.model_info(repo, revision=entry["revision"]).sha
        if entry["kind"] == "fasttext":
            patterns = [entry["filename"], "README*", "LICENSE*"]
        elif entry["kind"] == "tokenizer":
            patterns = ["*.json", "*.txt", "*.model", "*.py", "*.tiktoken", "README*", "LICENSE*"]
        else:
            patterns = ["*.json", "*.txt", "*.model", "*.py", "*.safetensors", "pytorch_model*.bin", "README*", "LICENSE*"]
        print(f"Downloading {repo}@{revision}", flush=True)
        snapshot_download(repo, revision=revision, allow_patterns=patterns)
        locked[repo] = {**entry, "revision": revision}
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.lock_output.with_suffix(".tmp")
    temp.write_text(json.dumps(locked, indent=2) + "\n")
    temp.replace(args.lock_output)
    print(f"Saved {args.lock_output}; set PRETRAINING_MODEL_MANIFEST to this file")

if __name__ == "__main__":
    main()
