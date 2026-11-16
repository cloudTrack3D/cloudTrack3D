"""
Updates `cloudTrack3D/CloudTrack-3D` on Hugging Face Hub with:
  1. A clean, highly readable `README.md` Dataset Card (using Unicode instead of raw `$...$` LaTeX
     that fails to render in HF Markdown tables, plus embedded visual overview figures in `assets/`).
  2. Interactive Dataset Viewer images (`viewer/train/` and `viewer/test/` with `metadata.csv`
     in Hugging Face `imagefolder` format) so every row of the Dataset Viewer displays the
     co-registered IR 10.5 μm + VIS 0.6 μm satellite imagery directly in the browser.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from huggingface_hub import HfApi

RELEASE_DIR = "../data/hf_cloudtrack3d_release"
REPO_ID = "cloudTrack3D/CloudTrack-3D"


def build_clean_readme() -> str:
    return """---
license: cc-by-4.0
task_categories:
  - image-segmentation
  - depth-estimation
tags:
  - remote-sensing
  - satellite
  - clouds
  - lidar
  - icesat-2
  - meteosat
  - era5
  - meteorology
  - semi-supervised-learning
pretty_name: CloudTrack-3D
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: viewer/train/**
      - split: test
        path: viewer/test/**
---

# CloudTrack-3D: Multi-Temporal Satellite–Lidar–Reanalysis Benchmark for 3D Cloud Reconstruction

**CloudTrack-3D** is a multimodal satellite, spaceborne lidar, and atmospheric reanalysis benchmark for reconstructing dense 2D maps of **cloud presence**, **cloud top altitude (km)**, and **cloud base altitude (km)**.

It pairs **full-resolution geostationary multispectral sequences** (`4096 × 4096` EUMETSAT Meteosat **IR 10.5 μm** and **VIS 0.6 μm** at **15-minute** cadence) and **hourly ERA5 pressure-level thermodynamic profiles** with **collocated 3-beam NASA ICESat-2 ATLAS (`ATL09` L3A) lidar vertical curtains** over the Mediterranean, North Africa, and Europe (**0°–45°E, 25°–40°N**) across **13 months (November 2024 – November 2025)**.

<p align="center">
  <img src="https://huggingface.co/datasets/cloudTrack3D/CloudTrack-3D/resolve/main/assets/fig1_dataset_overview.png" width="96%" alt="CloudTrack-3D Dataset Overview"/>
</p>

---

## Visual Examples from the Benchmark

### 1. Co-Registered Satellite Imagery & 3-Beam ICESat-2 Lidar Curtain
Each granule pairs full-disk sector **IR 10.5 μm** brightness temperature and **VIS 0.6 μm** solar reflectance with 3 parallel ICESat-2 lidar ground tracks (`gt1l`, `gt2l`, `gt3l`) providing vertical cloud top and base boundaries up to 10 atmospheric layers:

<p align="center">
  <img src="https://huggingface.co/datasets/cloudTrack3D/CloudTrack-3D/resolve/main/assets/fig_vfm_data_example.png" width="96%" alt="ICESat-2 Lidar and Satellite Example"/>
</p>

### 2. Dense 2D Cloud Top & Base Reconstruction vs. Along-Track Lidar Ground Truth
Dense 2D predictions of cloud mask, cloud top height (`0–14 km`), and cloud base height (`0–14 km`) compared against the sparse ICESat-2 lidar curtain:

<p align="center">
  <img src="https://huggingface.co/datasets/cloudTrack3D/CloudTrack-3D/resolve/main/assets/fig_alpha_cloud_top_overlay.png" width="90%" alt="Alpha-Blended Cloud Top Height Overlay"/>
</p>

---

## Dataset Splits & Protocols

| Split Protocol | Total Granules | Train Granules | Val/Test Granules | Eval Patches (256 × 256) | Modalities Included |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`allday_490` (1f IR All-Day)** | 490 | 392 | 98 | 588 | IR 10.5 μm (t = 0) + Lat/Lon + ERA5 + ICESat-2 3-Beam |
| **`multispectral_3frames_408` (3f IR+VIS)** | 408 | 327 | 81 | 486 | IR 10.5 μm (t = 0, -15m, -30m) + VIS 0.6 μm (t = 0, -15m, -30m) + Lat/Lon + ERA5 + ICESat-2 |
| **`synoptic_block_408` (14-Day Block + 48h Buffer)** | 408 | 304 | 91 *(+13 buffer)* | 546 | Weather-regime-disjoint evaluation split (±48 h exclusion buffer between blocks) |

---

## Repository Files & Archives

| File / Directory | Size | Description |
| :--- | :---: | :--- |
| `cloudtrack3d_splits.json` | `3.1 MB` | Complete split manifest (`allday_490`, `multispectral_3frames_408`, `synoptic_block_408`) with exact `256 × 256` evaluation patch bounding boxes |
| `granule_manifest_train.csv` / `granule_manifest_test.csv` | `113 KB` | Flat CSV metadata tables of all 490 matched granules (timestamps, 3-frame filenames, ERA5 links, split flags, footprint counts) |
| `viewer/train/` & `viewer/test/` | `~18 MB` | Side-by-side (**IR 10.5 μm \| VIS 0.6 μm**) preview images rendered directly in the Hugging Face Dataset Viewer above |
| `vfms_490.tar` | `205 MB` | All 490 matched ICESat-2 `ATL09` L3A `.h5` lidar granules (`vfms/ATL09_*.h5`) |
| `era5_matched_490.npz` | `55 MB` | Pre-extracted ERA5 atmospheric profiles (`18 × 61 × 77` `float16`: Temperature and Relative Humidity across 9 pressure levels `200–900 hPa`) keyed by `granule_id` |
| `ir105_matched_1467.tar` | `1.65 GB` | 1,467 matched 3-frame (`t = 0, -15m, -30m`) full-resolution `4096 × 4096` **IR 10.5 μm** images (`ir10.5/YYYY-MM-DD_HH-MM.png`) |
| `vis06_matched_1252.tar` | `4.13 GB` | 1,252 matched 3-frame (`t = 0, -15m, -30m`) full-resolution `4096 × 4096` **VIS 0.6 μm** images (`vis0.6/YYYY-MM-DD_HH-MM.png`) |
| `METEOROLOGY.md` | `12 KB` | Physical documentation of the satellite channels, lidar profiles, and ERA5 pressure levels |

<p align="center">
  <img src="https://huggingface.co/datasets/cloudTrack3D/CloudTrack-3D/resolve/main/assets/fig_era5_vertical_curtain.png" width="94%" alt="ERA5 Thermodynamic Profile and Lidar Curtain"/>
</p>

---

## Quick Start: Downloading & Unpacking

```bash
# 1. Download the dataset repository
huggingface-cli download cloudTrack3D/CloudTrack-3D --repo-type dataset --local-dir ./data
cd ./data

# 2. Unpack the full-resolution (4096 x 4096) satellite and lidar archives
tar -xf vfms_490.tar
tar -xf ir105_matched_1467.tar
tar -xf vis06_matched_1252.tar
```

### Loading ERA5 Profiles & Split Manifest in Python
```python
import json
import numpy as np

# Load train/test splits and exact 256x256 evaluation patch coordinates
with open("data/cloudtrack3d_splits.json") as f:
    splits = json.load(f)

train_granules = splits["splits"]["multispectral_3frames_408"]["train"]
test_granules  = splits["splits"]["multispectral_3frames_408"]["test"]

# Load pre-extracted 18-channel ERA5 profiles (9 Temperature + 9 Relative Humidity levels, 200-900 hPa)
era5_npz = np.load("data/era5_matched_490.npz")
sample_profile = era5_npz[train_granules[0]["granule_id"]]  # shape: (18, 61, 77)
```
"""


def make_preview_image(img_name: str, out_path: str) -> None:
    if os.path.isfile(out_path):
        return
    ir_path = os.path.join("../ir10.5", img_name)
    vis_path = os.path.join("../vis0.6", img_name)

    target_sz = (384, 384)
    if os.path.isfile(ir_path):
        with Image.open(ir_path) as im:
            ir_small = im.convert("L").resize(target_sz, Image.Resampling.BILINEAR)
    else:
        ir_small = Image.new("L", target_sz, 0)

    if os.path.isfile(vis_path):
        with Image.open(vis_path) as im:
            vis_small = im.convert("L").resize(target_sz, Image.Resampling.BILINEAR)
    else:
        vis_small = Image.new("L", target_sz, 0)

    # Concatenate side-by-side: Left = IR 10.5 um, Right = VIS 0.6 um
    comp = Image.new("L", (target_sz[0] * 2 + 4, target_sz[1]), 255)
    comp.paste(ir_small, (0, 0))
    comp.paste(vis_small, (target_sz[0] + 4, 0))
    comp.save(out_path, format="JPEG", quality=82, optimize=True)


def main() -> None:
    t0 = time.time()
    os.makedirs(RELEASE_DIR, exist_ok=True)

    # 1. Write clean README.md without raw LaTeX
    readme_path = os.path.join(RELEASE_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(build_clean_readme())

    # 2. Copy visual overview PNGs into assets/
    assets_dir = os.path.join(RELEASE_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    for fig_name in [
        "fig1_dataset_overview.png",
        "fig_vfm_data_example.png",
        "fig_alpha_cloud_top_overlay.png",
        "fig_era5_vertical_curtain.png",
        "fig4_qualitative_comparison.png",
    ]:
        src = os.path.join("paper/figures", fig_name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(assets_dir, fig_name))

    # 3. Build viewer/train and viewer/test imagefolder previews + metadata.csv
    with open("../data/cloudtrack3d_splits.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    ms_408_ids = {
        g["granule_id"]: g["split"]
        for split_k in ("train", "test")
        for g in splits_data["splits"]["multispectral_3frames_408"][split_k]
    }
    syn_408_ids = {
        g["granule_id"]: split_k
        for split_k in ("train", "test", "buffer_excluded")
        for g in splits_data["splits"]["synoptic_block_408"][split_k]
    }

    fieldnames = [
        "file_name",
        "granule_id",
        "datetime_utc",
        "img_name",
        "frame_t15",
        "frame_t30",
        "h5_filename",
        "era5_hour_utc",
        "n_lidar_footprints_in_sector",
        "split_allday_490",
        "in_multispectral_408",
        "split_multispectral_408",
        "split_synoptic_408",
    ]

    preview_tasks = []
    for split_k in ("train", "test"):
        split_dir = os.path.join(RELEASE_DIR, "viewer", split_k)
        os.makedirs(split_dir, exist_ok=True)
        rows = splits_data["splits"]["allday_490"][split_k]
        with open(os.path.join(split_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                gid = r["granule_id"]
                f30 = r.get("frames_30min", [r["img_name"], "", ""])
                jpg_name = f"{gid}.jpg"
                jpg_path = os.path.join(split_dir, jpg_name)
                preview_tasks.append((r["img_name"], jpg_path))
                writer.writerow({
                    "file_name": jpg_name,
                    "granule_id": gid,
                    "datetime_utc": r["datetime_utc"],
                    "img_name": r["img_name"],
                    "frame_t15": f30[1] if len(f30) > 1 else "",
                    "frame_t30": f30[2] if len(f30) > 2 else "",
                    "h5_filename": os.path.basename(r["h5_path"]),
                    "era5_hour_utc": r.get("era5_hour_utc", ""),
                    "n_lidar_footprints_in_sector": r.get("n_lidar_footprints_in_sector", 0),
                    "split_allday_490": r["split"],
                    "in_multispectral_408": gid in ms_408_ids,
                    "split_multispectral_408": ms_408_ids.get(gid, "none"),
                    "split_synoptic_408": syn_408_ids.get(gid, "none"),
                })

    print(f"Generating {len(preview_tasks)} side-by-side (IR 10.5 | VIS 0.6) viewer preview images across 16 threads...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda args: make_preview_image(*args), preview_tasks))

    # Upload only README.md, assets/, and viewer/ so we don't re-upload the 6 GB .tar files
    api = HfApi()
    print("Uploading updated README.md, assets/, and viewer/ to Hugging Face Hub...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=RELEASE_DIR,
        allow_patterns=["README.md", "assets/**", "viewer/**"],
        commit_message="Update README readability (no raw LaTeX) and add interactive Dataset Viewer satellite previews",
    )
    print(f"Done in {time.time()-t0:.1f}s! Live at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
