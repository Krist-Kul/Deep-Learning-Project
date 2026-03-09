"""
Download the Chest X-ray Pneumonia dataset from Kaggle
and place it under data/chest_xray/ within this repository.

Usage:
    python data_download.py

Requirements:
    pip install kagglehub
    Kaggle API credentials configured (~/.kaggle/kaggle.json)
"""

import os
import shutil
from pathlib import Path

import kagglehub

# Destination: data/chest_xray/ inside the repo
DATA_DIR = Path(__file__).parent / "data"
DEST_PATH = DATA_DIR / "chest_xray"


def download_dataset() -> Path:
    """Download the dataset and symlink (or copy) it into data/chest_xray/."""
    if DEST_PATH.exists():
        print(f"Dataset already present at: {DEST_PATH}")
        return DEST_PATH

    print("Downloading dataset from Kaggle…")
    cache_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    src = Path(cache_path)
    # kagglehub may nest the actual split folders under chest_xray/
    if (src / "chest_xray").exists():
        src = src / "chest_xray"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer a symlink to avoid duplicating GBs on disk; fall back to copy
    try:
        os.symlink(src.resolve(), DEST_PATH)
        print(f"Symlinked dataset → {DEST_PATH}")
    except OSError:
        print("Symlink unavailable — copying dataset (this may take a while)…")
        shutil.copytree(src, DEST_PATH)
        print(f"Dataset copied → {DEST_PATH}")

    return DEST_PATH


def _print_summary(root: Path):
    print("\nDataset summary:")
    for split in ["train", "val", "test"]:
        split_path = root / split
        if not split_path.exists():
            continue
        counts = {
            cls_dir.name: len(list(cls_dir.iterdir()))
            for cls_dir in sorted(split_path.iterdir())
            if cls_dir.is_dir()
        }
        total = sum(counts.values())
        print(f"  {split:5s}: {total:5d} images  {counts}")


if __name__ == "__main__":
    dest = download_dataset()
    _print_summary(dest)
