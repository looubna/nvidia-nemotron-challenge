"""Download adapter from Tinker and upload to Kaggle (no Modal needed)."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

from dotenv import load_dotenv

load_dotenv()

TINKER_PATH = "tinker://f687a1be-3c66-56f3-aada-edf905f74776:train:0/sampler_weights/final"
KAGGLE_INSTANCE = "loubnaenakhli/nemotron-adapter/Transformers/default"
ADAPTER_DIR = Path("/tmp/nemotron_adapter")


def download_from_tinker() -> Path:
    import tinker

    print(f"Getting download URL from Tinker for {TINKER_PATH}...")
    sc = tinker.ServiceClient()
    url = (
        sc.create_rest_client()
        .get_checkpoint_archive_url_from_tinker_path(TINKER_PATH)
        .result()
        .url
    )
    print(f"URL obtained. Downloading...")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = Path("/tmp/adapter.tar")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded/1e6:.0f} / {total/1e6:.0f} MB", end="\r")
    size_mb = tar_path.stat().st_size / 1e6
    print(f"\nDownloaded {size_mb:.1f} MB. Extracting...")

    # Extract to a temp dir first, then flatten to ADAPTER_DIR
    extract_dir = Path("/tmp/adapter_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir)
    tar_path.unlink()

    # Find and move adapter files to ADAPTER_DIR
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            src = Path(root) / fname
            dst = ADAPTER_DIR / fname
            shutil.move(str(src), str(dst))

    shutil.rmtree(extract_dir)

    print("Adapter files:")
    for f in sorted(ADAPTER_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

    return ADAPTER_DIR


def upload_to_kaggle(adapter_dir: Path) -> None:
    kaggle_key = os.environ.get("KAGGLE_API_KEY", "")
    if not kaggle_key:
        raise ValueError("KAGGLE_API_KEY not set in .env")

    # Set up kaggle credentials
    kaggle_cfg_dir = Path.home() / ".kaggle"
    kaggle_cfg_dir.mkdir(exist_ok=True)

    # Kaggle expects username:key format in access_token or kaggle.json
    # The KAGGLE_API_KEY in .env is the token string from the competition
    # Write kaggle.json format
    parts = kaggle_key.split("_", 1)  # KGAT_<key>
    if kaggle_key.startswith("KGAT_"):
        # Competition token format — write as access_token
        with open(kaggle_cfg_dir / "access_token", "w") as f:
            f.write(kaggle_key)
    else:
        # Standard kaggle.json format: {"username": "...", "key": "..."}
        with open(kaggle_cfg_dir / "kaggle.json", "w") as f:
            json.dump({"username": parts[0], "key": parts[1]}, f)
        (kaggle_cfg_dir / "kaggle.json").chmod(0o600)

    from kaggle.api.kaggle_api_extended import KaggleApi
    from requests.exceptions import HTTPError

    api = KaggleApi()
    api.authenticate()
    print("Kaggle authenticated")

    files = list(adapter_dir.iterdir())
    print(f"Uploading {len(files)} files to {KAGGLE_INSTANCE}...")

    def instance_exists() -> bool:
        try:
            api.model_instance_get(KAGGLE_INSTANCE)
            return True
        except HTTPError:
            return False

    if not instance_exists():
        parts = KAGGLE_INSTANCE.split("/")
        owner, model_slug, framework, instance_slug = parts
        upload_dir = tempfile.mkdtemp()
        for f in files:
            shutil.copy(str(f), upload_dir)
        metadata = {
            "ownerSlug": owner,
            "modelSlug": model_slug,
            "instanceSlug": instance_slug,
            "framework": framework,
            "licenseName": "Apache 2.0",
            "overview": "Nemotron-3-Nano-30B LoRA adapter",
        }
        with open(os.path.join(upload_dir, "model-instance-metadata.json"), "w") as f:
            json.dump(metadata, f)
        api.model_instance_create(upload_dir, dir_mode="skip")
        print("Instance created")
    else:
        print(f"Instance {KAGGLE_INSTANCE} already exists, uploading new version...")

    api.model_instance_version_create(KAGGLE_INSTANCE, str(adapter_dir), dir_mode="skip")
    print("Upload complete!")


if __name__ == "__main__":
    adapter_dir = download_from_tinker()
    upload_to_kaggle(adapter_dir)
    print(f"\nDone. Adapter uploaded to kaggle.com/models/{KAGGLE_INSTANCE}")
