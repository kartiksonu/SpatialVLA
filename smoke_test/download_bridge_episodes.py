#!/usr/bin/env python3
"""
Download Bridge V2 dataset episodes.

Downloads TFRecord files from the Bridge V2 dataset.
Each TFRecord file contains ~50 episodes.
"""

import os
import urllib.request
from pathlib import Path

# Base URL for Bridge V2 dataset
TFRECORD_BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"

# Directory to save files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_tfrecord(shard_number, total_shards=1024):
    """
    Download a specific TFRecord shard.

    Args:
        shard_number: Shard number (0-indexed)
        total_shards: Total number of shards (default 1024 for Bridge V2)
    """
    filename = f"bridge_dataset-train.tfrecord-{shard_number:05d}-of-{total_shards:05d}"
    url = TFRECORD_BASE_URL + filename
    filepath = DATA_DIR / filename

    if filepath.exists():
        print(f"✓ {filename} already exists ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Bridge V2 dataset episodes")
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 1],
                       help="Shard numbers to download (default: 0 1, which gives ~100 episodes)")
    parser.add_argument("--total_shards", type=int, default=1024,
                       help="Total number of shards (default: 1024)")
    args = parser.parse_args()

    print("="*70)
    print("Bridge V2 Dataset Downloader")
    print("="*70)
    print(f"Download directory: {DATA_DIR}")
    print(f"Shards to download: {args.shards}")
    print(f"Each shard contains ~50 episodes")
    print("="*70)

    success_count = 0
    for shard in args.shards:
        if download_tfrecord(shard, args.total_shards):
            success_count += 1

    print("\n" + "="*70)
    print(f"Downloaded {success_count}/{len(args.shards)} shards successfully")
    print(f"Files saved to: {DATA_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
