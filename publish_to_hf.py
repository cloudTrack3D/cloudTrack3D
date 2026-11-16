#!/usr/bin/env python3
"""
Turnkey Packaging, Upload & Download Utility for CloudTrack-3D:
1. Hugging Face Datasets (<username>/CloudTrack-3D)
2. Hugging Face Model Hub (<username>/CloudTrack-3D-Models)

Usage Examples:
  # 1. Generate HF Dataset & Model Cards locally and inspect package statistics:
  python publish_to_hf.py --mode prepare-cards

  # 2. Authenticate with Hugging Face (or set HF_TOKEN env variable):
  huggingface-cli login

  # 3. Upload Dataset to Hugging Face Datasets:
  python publish_to_hf.py --mode upload-dataset --dataset-repo <hf-username>/CloudTrack-3D

  # 4. Upload all 16 Model Checkpoints + Evaluation JSONs to Hugging Face Model Hub:
  python publish_to_hf.py --mode upload-models --model-repo <hf-username>/CloudTrack-3D-Models

  # 5. Download Dataset or Models on a new machine:
  python publish_to_hf.py --mode download-dataset --dataset-repo <hf-username>/CloudTrack-3D --data-root ../data
  python publish_to_hf.py --mode download-models --model-repo <hf-username>/CloudTrack-3D-Models --output-dir ./outputs
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


MAIN_MODEL_DIRS = [
    # 1-Frame Still IR 10.5 um (490 Granules)
    "supervised_unet",
    "semisup_unet",
    "supervised_segvit",
    "semisup_segvit",
    "supervised_sam_1f",
    "semisup_sam_1f",
    "supervised_dinov3_1f",
    "semisup_dinov3_1f",
    # 3-Frame Temporal + Multi-Spectral IR 10.5 + VIS 0.6 um (408 Granules)
    "supervised_unet_3f_vis",
    "semisup_unet_3f_vis",
    "supervised_segvit_3f_vis",
    "semisup_segvit_3f_vis",
    "supervised_sam_3f_vis",
    "semisup_sam_3f_vis",
    "supervised_dinov3_3f_vis",
    "semisup_dinov3_3f_vis",
]


DATASET_CARD_TEMPLATE = """---
license: cc-by-4.0
task_categories:
  - image-segmentation
  - depth-estimation
  - other
tags:
  - remote-sensing
  - satellite-imagery
  - icesat-2
  - lidar
  - cloud-detection
  - cloud-height
  - meteorology
  - semi-supervised-learning
  - cvpr
pretty_name: CloudTrack-3D Multi-Temporal Satellite-Lidar Benchmark
size_categories:
  - 1M<n<10M
---

# CloudTrack-3D: A Multi-Temporal Satellite-Lidar Benchmark for Dense 3D Cloud Reconstruction

**CloudTrack-3D** is a multi-modal, multi-temporal benchmark dataset pairing passive geostationary satellite imagery from **EUMETSAT's Meteosat Third Generation Flexible Combined Imager (MTG-I1 FCI)** with co-located active 3-beam spaceborne photon-counting lidar vertical profiles from **NASA's ICESat-2 ATLAS (ATL09 L3A)**.

## Dataset Summary

- **Geographic Sector**: Eastern Mediterranean, Middle East, and North Africa (`Lon 24.0°E to 43.0°E`, `Lat 25.0°N to 40.0°N`, `EPSG:4326`, $4096 \\times 4096$ pixels at $\\sim 450\\text{ m}$ spatial resolution).
- **Temporal Coverage**: November 2024 – September 2025 (11-month cycle across all four seasons).
- **Matched Overpasses**:
  - **490 validated ICESat-2 3-beam granules** ($>6.1\\text{M}$ lidar profile shots; $\\sim 12,600$ labeled pixels per scene = $0.075\\%$ pixel density across `profile_1`, `profile_2`, `profile_3`).
  - **408 complete 30-minute dual-spectral sequences** ($t \\in \\{0, -15, -30\\}\\text{ min}$ across both $\\text{IR } 10.5\\ \\mu\\text{m}$ and $\\text{VIS } 0.6\\ \\mu\\text{m}$).
- **Target Annotations (per Lidar Pixel)**:
  1. `cloud_mask` ($\\in \\{0, 1\\}$): Binary cloud presence (`layer_attr == 1`).
  2. `cloud_top_km` ($\\in [0, 16]\\text{ km}$): Uppermost cloud top altitude above sea level.
  3. `cloud_base_km` ($\\in [0, h_{\\text{top}}]\\text{ km}$): Lowest valid cloud base altitude above sea level.
  4. `n_layers` ($\\in \\{0, \\dots, 10\\}$): Number of overlapping vertical cloud layers.
  5. `surface_height_km`: Ground/ocean surface elevation from ATL09.

## Directory Structure

```text
CloudTrack-3D/
├── manifests/
│   ├── matched_granules_cache.json          # 490 all-day 1-frame IR 10.5 um matched granules
│   └── matched_granules_3frames_cache.json  # 408 3-frame (t=0,-15,-30m) IR 10.5 + VIS 0.6 um sequences
├── vfms/                                    # 490 deduplicated ICESat-2 ATL09 L3A HDF5 (.h5) files
├── ir10.5/                                  # 4096x4096 MTG-I1 FCI IR 10.5 um frames (.png)
├── vis0.6/                                  # 4096x4096 MTG-I1 FCI VIS 0.6 um frames (.png)
└── figures/                                 # Overview and statistical characterization figures
```

## Quickstart Usage

```python
from huggingface_hub import snapshot_download

# Download dataset locally
data_path = snapshot_download(repo_id="{repo_id}", repo_type="dataset", local_dir="./data")
```
"""


MODEL_CARD_TEMPLATE = """---
license: mit
library_name: pytorch
tags:
  - remote-sensing
  - satellite-imagery
  - dinov3
  - sam2
  - segvit
  - unet
  - semi-supervised-learning
  - unimatch
  - cloud-height-estimation
  - cvpr
---

# CloudTrack-3D Pretrained Foundation & Baseline Models

This repository hosts all **16 main pretrained model checkpoints** (`Supervised` and `Semi-Supervised UniMatch` across `U-Net`, `SegViT`, `SAM 2.1`, and `DINOv3`) on the **CloudTrack-3D** benchmark, along with full evaluation metrics (`all_models_comparison.json` and `cvpr_all_results.json`).

## Model Zoo & Benchmark Performance

| Checkpoint Directory | Modality | Backbone | Training | Acc (%) | F1 | AUROC | Top MAE (km) | Top $R^2$ | Base MAE (km) | Base $R^2$ | Low-Cloud ($\\le 15\\text{kft}$) Base MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `supervised_unet` | 1f IR | U-Net (6.2M) | Supervised | 89.49 | 0.8034 | 0.9263 | 2.374 | 0.2244 | 2.196 | 0.1968 | 2.317 km |
| `semisup_unet` | 1f IR | U-Net (6.2M) | UniMatch | 89.73 | 0.8024 | 0.9303 | 2.303 | 0.2580 | 2.166 | 0.2093 | 2.275 km |
| `supervised_segvit` | 1f IR | SegViT (14.2M) | Supervised | 89.26 | 0.7945 | 0.9252 | 2.301 | 0.2290 | 2.152 | 0.1750 | 2.393 km |
| `semisup_segvit` | 1f IR | SegViT (14.2M) | UniMatch | 89.10 | 0.7949 | 0.9248 | 2.210 | 0.3187 | 2.075 | 0.2608 | 2.503 km |
| `supervised_dinov3_1f` | 1f IR | DINOv3 (88.9M) | Supervised | 88.81 | 0.7913 | 0.9239 | 2.326 | 0.2675 | 2.155 | 0.2006 | 2.369 km |
| `semisup_dinov3_1f` | 1f IR | DINOv3 (88.9M) | UniMatch | 88.82 | 0.7875 | 0.9249 | 2.054 | 0.3899 | 1.957 | 0.3230 | 2.214 km |
| `supervised_sam_1f` | 1f IR | SAM 2.1 (37.0M) | Supervised | 90.69 | 0.8226 | 0.9356 | 1.768 | 0.5195 | 1.776 | 0.4443 | 1.781 km |
| `semisup_sam_1f` | 1f IR | **SAM 2.1 (37.0M)** | **UniMatch** | **90.97** | **0.8314** | **0.9396** | **1.761** | **0.5060** | **1.783** | **0.4212** | **1.664 km** |
| `supervised_unet_3f_vis` | 3f IR+VIS | U-Net (6.3M) | Supervised | 89.82 | 0.7457 | 0.9078 | 1.732 | 0.5821 | 1.751 | 0.4936 | 1.061 km |
| `semisup_unet_3f_vis` | 3f IR+VIS | U-Net (6.3M) | UniMatch | 89.67 | 0.7467 | 0.9084 | 1.678 | 0.5869 | 1.724 | 0.4772 | 0.809 km |
| `supervised_segvit_3f_vis` | 3f IR+VIS | SegViT (14.2M) | Supervised | 89.65 | 0.7377 | 0.8987 | 1.839 | 0.5861 | 1.817 | 0.5167 | 1.126 km |
| `semisup_segvit_3f_vis` | 3f IR+VIS | SegViT (14.2M) | UniMatch | 89.87 | 0.7426 | 0.9024 | 1.956 | 0.5076 | 1.905 | 0.4388 | 1.385 km |
| `supervised_sam_3f_vis` | 3f IR+VIS | SAM 2.1 (37.0M) | Supervised | 89.03 | 0.7408 | 0.9213 | 1.535 | 0.6133 | 1.538 | 0.5458 | 0.957 km |
| `semisup_sam_3f_vis` | 3f IR+VIS | **SAM 2.1 (37.0M)** | **UniMatch** | 89.71 | 0.7530 | 0.9198 | 1.356 | 0.7232 | 1.361 | 0.6504 | 0.841 km |
| `supervised_dinov3_3f_vis` | 3f IR+VIS | DINOv3 (88.9M) | Supervised | 89.93 | 0.7587 | 0.9250 | 1.450 | 0.6496 | 1.494 | 0.5847 | 0.842 km |
| `semisup_dinov3_3f_vis` | 3f IR+VIS | **DINOv3 (88.9M)** | **UniMatch** | **90.14** | **0.7631** | **0.9281** | **1.346** | **0.7291** | **1.325** | **0.6844** | **0.728 km** |

## Loading a Pretrained Model in PyTorch

```python
import torch
from huggingface_hub import hf_hub_download
from models import build_model

ckpt_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="semisup_dinov3_3f_vis/best_semisup.pt"
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = build_model(arch="dinov3", in_channels=8, base_channels=64)
state_dict = ckpt.get("teacher_state_dict", ckpt.get("model_state_dict", ckpt))
model.load_state_dict(state_dict)
model.eval()
```
"""


def prepare_cards(staging_dir: Path, dataset_repo: str, model_repo: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    ds_card = staging_dir / "README_dataset.md"
    md_card = staging_dir / "README_models.md"
    ds_card.write_text(DATASET_CARD_TEMPLATE.replace("{repo_id}", dataset_repo))
    md_card.write_text(MODEL_CARD_TEMPLATE.replace("{repo_id}", model_repo))
    print(f"[OK] Wrote Hugging Face Dataset Card -> {ds_card}")
    print(f"[OK] Wrote Hugging Face Model Card   -> {md_card}")

    # Summarize matched dataset files
    c1_path = Path("outputs/matched_granules_cache.json")
    c3_path = Path("outputs/matched_granules_3frames_cache.json")
    if c1_path.exists() and c3_path.exists():
        from datetime import datetime, timedelta

        def _shift_name(name: str, delta_min: int) -> str:
            stem = name.replace(".png", "")
            dt = datetime.strptime(stem, "%Y-%m-%d_%H-%M") + timedelta(minutes=delta_min)
            return dt.strftime("%Y-%m-%d_%H-%M.png")

        c1 = json.loads(c1_path.read_text())
        c3_raw = json.loads(c3_path.read_text())
        c3 = c3_raw.get("items", c3_raw) if isinstance(c3_raw, dict) else c3_raw
        h5_files = {Path(g["h5_path"]).name for g in c1}
        ir_files = {Path(g["img_path"]).name for g in c1}
        vis_files = set()
        for g in c3:
            base_name = g["img_name"]
            for delta in (0, -15, -30):
                shifted = _shift_name(base_name, delta)
                ir_files.add(shifted)
                vis_files.add(shifted)
        print(
            f"[Dataset Manifest Summary] "
            f"{len(h5_files)} matched ICESat-2 .h5 granules | "
            f"{len(ir_files)} matched IR 10.5 frames | "
            f"{len(vis_files)} matched VIS 0.6 frames"
        )

    # Summarize model checkpoints
    found_ckpts = 0
    total_bytes = 0
    for d in MAIN_MODEL_DIRS:
        p = Path("outputs") / d
        for pt in p.glob("best_*.pt"):
            found_ckpts += 1
            total_bytes += pt.stat().st_size
    print(
        f"[Model Zoo Summary] Found {found_ckpts}/16 main checkpoints "
        f"({total_bytes / (1024**3):.2f} GiB raw, ~{total_bytes / (2 * 1024**3):.2f} GiB inference-only)"
    )


def stage_and_upload_dataset(
    dataset_repo: str,
    staging_dir: Path,
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=private, exist_ok=True)

    ds_stage = staging_dir / "dataset_upload"
    (ds_stage / "manifests").mkdir(parents=True, exist_ok=True)
    (ds_stage / "figures").mkdir(parents=True, exist_ok=True)

    # Write README.md
    (ds_stage / "README.md").write_text(DATASET_CARD_TEMPLATE.replace("{repo_id}", dataset_repo))

    # Copy manifests
    for m in ["matched_granules_cache.json", "matched_granules_3frames_cache.json"]:
        src = Path("outputs") / m
        if src.exists():
            shutil.copy2(src, ds_stage / "manifests" / m)

    # Copy overview figures
    for fig in ["fig1_dataset_overview.png", "fig2_physical_correlations.png"]:
        src = Path("paper/figures") / fig
        if src.exists():
            shutil.copy2(src, ds_stage / "figures" / fig)

    # Upload metadata & manifests first
    print(f"Uploading dataset card and manifests to {dataset_repo}...")
    api.upload_folder(
        repo_id=dataset_repo,
        repo_type="dataset",
        folder_path=str(ds_stage),
        commit_message="Upload CloudTrack-3D dataset card, manifests, and overview figures",
    )

    # Upload matched vfms, ir10.5, and vis0.6 folders
    for folder_name, local_rel in [
        ("vfms", Path("../data/vfms")),
        ("ir10.5", Path("../data/ir10.5")),
        ("vis0.6", Path("../data/vis0.6")),
    ]:
        if local_rel.exists():
            print(f"Uploading {folder_name} ({local_rel.resolve()}) to {dataset_repo}/{folder_name}...")
            api.upload_folder(
                repo_id=dataset_repo,
                repo_type="dataset",
                folder_path=str(local_rel.resolve()),
                path_in_repo=folder_name,
                commit_message=f"Upload {folder_name} data files",
            )
    print(f"[SUCCESS] Dataset uploaded to https://huggingface.co/datasets/{dataset_repo}")


def stage_and_upload_models(
    model_repo: str,
    staging_dir: Path,
    strip_optimizer: bool = True,
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=model_repo, repo_type="model", private=private, exist_ok=True)

    m_stage = staging_dir / "models_upload"
    m_stage.mkdir(parents=True, exist_ok=True)
    (m_stage / "README.md").write_text(MODEL_CARD_TEMPLATE.replace("{repo_id}", model_repo))

    # Copy evaluation summary JSONs
    for rel_json in [
        Path("outputs/comparison/all_models_comparison.json"),
        Path("outputs/cvpr_experiments/cvpr_all_results.json"),
        Path("results.md"),
    ]:
        if rel_json.exists():
            shutil.copy2(rel_json, m_stage / rel_json.name)

    # Stage all 16 model directories
    for d in MAIN_MODEL_DIRS:
        src_dir = Path("outputs") / d
        if not src_dir.exists():
            continue
        dst_dir = m_stage / d
        dst_dir.mkdir(parents=True, exist_ok=True)

        for meta_file in ["best_metrics.json", "metrics_history.json", "supervised_predictions.png", "semisup_predictions.png"]:
            f = src_dir / meta_file
            if f.exists():
                shutil.copy2(f, dst_dir / meta_file)

        for pt_file in src_dir.glob("best_*.pt"):
            dst_pt = dst_dir / pt_file.name
            if strip_optimizer:
                ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
                slim_ckpt = {
                    k: v
                    for k, v in ckpt.items()
                    if k in ("epoch", "model_state_dict", "teacher_state_dict", "best_val_loss", "metrics", "args")
                }
                torch.save(slim_ckpt, dst_pt)
            else:
                shutil.copy2(pt_file, dst_pt)
            print(f"Staged checkpoint: {dst_pt} ({dst_pt.stat().st_size / (1024**2):.1f} MiB)")

    print(f"Uploading all 16 models and metrics to https://huggingface.co/{model_repo} ...")
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(m_stage),
        commit_message="Upload all 16 CloudTrack-3D pretrained checkpoints and benchmark results",
    )
    print(f"[SUCCESS] Models uploaded to https://huggingface.co/{model_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudTrack-3D Hugging Face Dataset & Model Hub Publisher")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prepare-cards", "upload-dataset", "upload-models", "download-dataset", "download-models"],
    )
    parser.add_argument("--dataset-repo", type=str, default="cloudtrack3d/CloudTrack-3D")
    parser.add_argument("--model-repo", type=str, default="cloudtrack3d/CloudTrack-3D-Models")
    parser.add_argument("--staging-dir", type=str, default="./hf_staging")
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--private", action="store_true", help="Create private Hugging Face repository")
    parser.add_argument("--keep-optimizer", action="store_true", help="Keep optimizer states in uploaded checkpoints")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face write token")
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)

    if args.mode == "prepare-cards":
        prepare_cards(staging_dir, args.dataset_repo, args.model_repo)
    elif args.mode == "upload-dataset":
        stage_and_upload_dataset(args.dataset_repo, staging_dir, private=args.private, token=args.token)
    elif args.mode == "upload-models":
        stage_and_upload_models(
            args.model_repo,
            staging_dir,
            strip_optimizer=not args.keep_optimizer,
            private=args.private,
            token=args.token,
        )
    elif args.mode == "download-dataset":
        snapshot_download(repo_id=args.dataset_repo, repo_type="dataset", local_dir=args.data_root, token=args.token)
        print(f"[SUCCESS] Downloaded dataset to {args.data_root}")
    elif args.mode == "download-models":
        snapshot_download(repo_id=args.model_repo, repo_type="model", local_dir=args.output_dir, token=args.token)
        print(f"[SUCCESS] Downloaded models to {args.output_dir}")


if __name__ == "__main__":
    main()
