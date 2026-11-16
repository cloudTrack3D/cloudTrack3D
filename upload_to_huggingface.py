"""
Uploads the prepared CloudTrack-3D dataset (`../data/hf_cloudtrack3d_release`)
to Hugging Face Hub under `cloudTrack3D/CloudTrack-3D` (or a specified `--repo-id`).

Usage:
  python upload_to_huggingface.py --token hf_YOUR_WRITE_TOKEN
"""

from __future__ import annotations

import argparse
import os
from huggingface_hub import HfApi, create_repo

RELEASE_DIR = "../data/hf_cloudtrack3d_release"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload CloudTrack-3D dataset to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default="cloudTrack3D/CloudTrack-3D", help="Target HF dataset repo ID")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token (hf_...)")
    parser.add_argument("--private", action="store_true", help="Create as a private dataset repository")
    args = parser.parse_args()

    if not os.path.isdir(RELEASE_DIR):
        raise FileNotFoundError(f"Release directory not found: {RELEASE_DIR}. Run prepare_hf_dataset.py first.")

    api = HfApi(token=args.token)
    user_info = api.whoami()
    print(f"Authenticated as Hugging Face user: {user_info.get('name')}")

    repo_url = create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    print(f"Dataset repository ready: {repo_url}")

    print(f"Uploading {RELEASE_DIR} to {args.repo_id} ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=RELEASE_DIR,
        commit_message="Upload CloudTrack-3D benchmark dataset",
        ignore_patterns=[".cache*", ".git*"],
    )
    print(f"Successfully uploaded CloudTrack-3D to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
