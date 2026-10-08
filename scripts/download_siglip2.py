"""Download official weights with a pinned revision and record exact file hashes."""
import os, sys, json, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "data/cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
from huggingface_hub import HfApi, snapshot_download
model = "google/siglip2-base-patch32-256"
revision = HfApi().model_info(model).sha
for attempt in range(1, 5):
    try:
        path = Path(snapshot_download(model, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"], max_workers=2))
        break
    except OSError as error:
        print("Download interrupted, preserving partial file for resume:", attempt, str(error), flush=True)
        if attempt == 4:
            raise
        time.sleep(2)
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}
report = {"model": model, "revision": revision, "sha256": files, "python": sys.executable}
(ROOT / "reports/siglip2-download.json").write_text(json.dumps(report, indent=2))
print("SIGLIP2_DOWNLOAD_FINISHED", revision, flush=True)
