"""Model sources, with optional immutable revision manifests."""
import json
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def manifest():
    path = Path(os.environ.get("PRETRAINING_MODEL_MANIFEST", Path(__file__).with_name("models.json")))
    return json.loads(path.read_text())

def model_kwargs(repo_id):
    entry = manifest()[repo_id]
    return {"revision": entry["revision"]}
