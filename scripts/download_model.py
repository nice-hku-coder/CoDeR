"""
通过 https://hf-mirror.com/ 下载指定模型到 `CoDeR/models/` 目录下。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_HF_MIRROR = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model repo id, e.g. facebook/contriever")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Root directory for downloaded models",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch / tag / commit hash",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_HF_MIRROR,
        help="Hugging Face endpoint / mirror (default: https://hf-mirror.com)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token for gated repositories",
    )
    return parser.parse_args()

def set_local_name(model_id: str):
    return model_id.replace("/", "_")


def main() -> None:
    args = parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = set_local_name(args.model_id)
    local_dir = models_dir / model_name
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    downloaded_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=args.token
    )

    print(f"\nDownload completed: {downloaded_path}")


if __name__ == "__main__":
    main()
