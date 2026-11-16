"""
Prepares the CloudTrack-3D benchmark release directory (`../data/hf_cloudtrack3d_release`)
for upload to Hugging Face Hub (`cloudTrack/CloudTrack-3D`), including:
  - `README.md` (Hugging Face Dataset Card with YAML metadata & Dataset Viewer config)
  - `granule_manifest.csv` (490-row flat metadata table for HF Dataset Viewer)
  - `cloudtrack3d_splits.json` (Full train/test/synoptic split & patch coordinate manifest)
  - `METEOROLOGY.md` (Meteorological and physical documentation)
  - `vfms_490.tar` (490 matched ICESat-2 ATL09 L3A `.h5` lidar profiles, ~205 MB)
  - `era5_matched_490.npz` (490 matched 18-channel ERA5 pressure-level tensors, ~18 MB)
  - `ir105_matched_1467.tar` (1,467 matched 3-frame 4096x4096 IR 10.5 um PNGs, ~1.64 GB)
  - `vis06_matched_1252.tar` (1,252 matched 3-frame 4096x4096 VIS 0.6 um PNGs, ~4.13 GB)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
import time
import numpy as np

from dataset import discover_matched_granules, filter_granules_with_30min_window, get_3frame_filenames
from run_era5_experiments import preload_era5_for_granules

RELEASE_DIR = "../data/hf_cloudtrack3d_release"


def build_readme() -> str:
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
        path: granule_manifest_train.csv
      - split: test
        path: granule_manifest_test.csv
---

# CloudTrack-3D: Sparse Spaceborne Lidar to Dense 2D Cloud Top & Base Altitude Reconstruction

**CloudTrack-3D** is a multimodal satellite-lidar-reanalysis benchmark pairing **dense geostationary multispectral sequences** ($4096 \\times 4096$ Meteosat SEVIRI/FCI $\\text{IR } 10.5\\ \\mu\\text{m}$ and $\\text{VIS } 0.6\\ \\mu\\text{m}$ at $15\\text{-minute}$ cadence) and **hourly ERA5 pressure-level thermodynamic profiles** with **collocated 3-beam ICESat-2 ATLAS (`ATL09` L3A) lidar vertical curtains** over the Mediterranean, North Africa, and Europe ($0^\\circ\\text{--}45^\\circ\\text{E},\\ 25^\\circ\\text{--}40^\\circ\\text{N}$) across **13 months (November 2024 -- November 2025)**.

---

## Dataset Summary & Splits

| Split Protocol | Total Granules | Train Granules | Val/Test Granules | Eval Patches ($256 \\times 256$) | Modalities Included |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`allday_490` (1f IR All-Day)** | 490 | 392 | 98 | 588 | $\\text{IR } 10.5\\ \\mu\\text{m}(t_0)$ + Lat/Lon + ERA5 + ICESat-2 3-Beam |
| **`multispectral_3frames_408` (3f IR+VIS)** | 408 | 327 | 81 | 486 | $\\text{IR } 10.5\\ \\mu\\text{m}(t_0, t_{-15}, t_{-30})$ + $\\text{VIS } 0.6\\ \\mu\\text{m}(t_0, t_{-15}, t_{-30})$ + Lat/Lon + ERA5 + ICESat-2 |
| **`synoptic_block_408` (14-Day Block + 48h Buffer)** | 408 | 304 | 91 *(+13 buffer)* | 546 | Weather-regime-disjoint evaluation split ($\\pm 48\\text{ h}$ exclusion buffer) |

---

## Repository Contents

| File | Size | Description |
| :--- | :---: | :--- |
| `cloudtrack3d_splits.json` | `3.1 MB` | Complete split manifest (`allday_490`, `multispectral_3frames_408`, `synoptic_block_408`) with exact $256 \\times 256$ evaluation patch coordinates |
| `granule_manifest_train.csv` / `granule_manifest_test.csv` | `~90 KB` | Tabular metadata index of all 490 matched granules (timestamps, filenames, split flags, footprint counts) |
| `vfms_490.tar` | `205 MB` | All 490 matched ICESat-2 `ATL09` L3A `.h5` lidar granules (`vfms/ATL09_*.h5`) |
| `era5_matched_490.npz` | `~18 MB` | Pre-extracted ERA5 atmospheric profiles (`18 x 61 x 181` `float16`: $T$ and $RH$ at $200\\text{--}900\\text{ hPa}$) keyed by `img_name` |
| `ir105_matched_1467.tar` | `1.64 GB` | 1,467 matched 3-frame ($t_0, t_0-15\\text{m}, t_0-30\\text{m}$) $4096 \\times 4096$ $\\text{IR } 10.5\\ \\mu\\text{m}$ images (`ir10.5/YYYY-MM-DD_HH-MM.png`) |
| `vis06_matched_1252.tar` | `4.13 GB` | 1,252 matched 3-frame ($t_0, t_0-15\\text{m}, t_0-30\\text{m}$) $4096 \\times 4096$ $\\text{VIS } 0.6\\ \\mu\\text{m}$ images (`vis0.6/YYYY-MM-DD_HH-MM.png`) |
| `METEOROLOGY.md` | `12 KB` | Detailed physical documentation of the satellite channels, lidar profiles, and ERA5 pressure levels |

---

## Quick Start (Unpacking)

```bash
tar -xf vfms_490.tar
tar -xf ir105_matched_1467.tar
tar -xf vis06_matched_1252.tar
```
"""


def main() -> None:
    t0 = time.time()
    os.makedirs(RELEASE_DIR, exist_ok=True)

    # 1. Copy cloudtrack3d_splits.json and METEOROLOGY.md
    shutil.copy2("../data/cloudtrack3d_splits.json", os.path.join(RELEASE_DIR, "cloudtrack3d_splits.json"))
    shutil.copy2("../data/METEOROLOGY.md", os.path.join(RELEASE_DIR, "METEOROLOGY.md"))

    # 2. Write README.md
    with open(os.path.join(RELEASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_readme())

    # 3. Build granule_manifest_train.csv and granule_manifest_test.csv from cloudtrack3d_splits.json
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
        "granule_id",
        "datetime_utc",
        "timestamp_utc",
        "img_name",
        "frame_t15",
        "frame_t30",
        "h5_filename",
        "era5_nc",
        "era5_hour_utc",
        "n_lidar_footprints_in_sector",
        "split_allday_490",
        "in_multispectral_408",
        "split_multispectral_408",
        "split_synoptic_408",
    ]

    for split_k, out_csv in [("train", "granule_manifest_train.csv"), ("test", "granule_manifest_test.csv")]:
        rows = splits_data["splits"]["allday_490"][split_k]
        with open(os.path.join(RELEASE_DIR, out_csv), "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                gid = r["granule_id"]
                f30 = r.get("frames_30min", [r["img_name"], "", ""])
                writer.writerow({
                    "granule_id": gid,
                    "datetime_utc": r["datetime_utc"],
                    "timestamp_utc": r["timestamp_utc"],
                    "img_name": r["img_name"],
                    "frame_t15": f30[1] if len(f30) > 1 else "",
                    "frame_t30": f30[2] if len(f30) > 2 else "",
                    "h5_filename": os.path.basename(r["h5_path"]),
                    "era5_nc": r.get("era5_nc", ""),
                    "era5_hour_utc": r.get("era5_hour_utc", ""),
                    "n_lidar_footprints_in_sector": r.get("n_lidar_footprints_in_sector", 0),
                    "split_allday_490": r["split"],
                    "in_multispectral_408": gid in ms_408_ids,
                    "split_multispectral_408": ms_408_ids.get(gid, "none"),
                    "split_synoptic_408": syn_408_ids.get(gid, "none"),
                })

    # 4. Collect matched files for the 490 granules
    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    vfm_files = sorted({g.h5_path for g in granules_490})
    ir_files = set()
    vis_files = set()
    img_names_490 = sorted({g.img_name for g in granules_490})
    for g in granules_490:
        for fn in get_3frame_filenames(g.img_name):
            p_ir = os.path.join("../ir10.5", fn)
            if os.path.isfile(p_ir):
                ir_files.add(p_ir)
            p_vis = os.path.join("../vis0.6", fn)
            if os.path.isfile(p_vis):
                vis_files.add(p_vis)

    # 5. Build era5_matched_490.npz
    era5_npz_path = os.path.join(RELEASE_DIR, "era5_matched_490.npz")
    if not os.path.isfile(era5_npz_path):
        print(f"Building {era5_npz_path} for {len(granules_490)} granules...")
        era5_raw = preload_era5_for_granules(granules_490)
        era5_dict = {gid: arr.astype(np.float16) for gid, arr in era5_raw.items()}
        np.savez_compressed(era5_npz_path, **era5_dict)
        print(f"Saved {era5_npz_path} ({os.path.getsize(era5_npz_path)/1e6:.1f} MB)")

    def _anon_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = 0
        ti.gid = 0
        ti.uname = "root"
        ti.gname = "root"
        return ti

    # 6. Build vfms_490.tar
    vfm_tar_path = os.path.join(RELEASE_DIR, "vfms_490.tar")
    if not os.path.isfile(vfm_tar_path):
        print(f"Building {vfm_tar_path} ({len(vfm_files)} files)...")
        with tarfile.open(vfm_tar_path, "w") as tf:
            for p in vfm_files:
                tf.add(os.path.realpath(p), arcname=f"vfms/{os.path.basename(p)}", filter=_anon_filter)
        print(f"Saved {vfm_tar_path} ({os.path.getsize(vfm_tar_path)/1e6:.1f} MB)")

    # 7. Build ir105_matched_1467.tar
    ir_tar_path = os.path.join(RELEASE_DIR, "ir105_matched_1467.tar")
    if not os.path.isfile(ir_tar_path):
        print(f"Building {ir_tar_path} ({len(ir_files)} files)...")
        with tarfile.open(ir_tar_path, "w") as tf:
            for p in sorted(ir_files):
                tf.add(os.path.realpath(p), arcname=f"ir10.5/{os.path.basename(p)}", filter=_anon_filter)
        print(f"Saved {ir_tar_path} ({os.path.getsize(ir_tar_path)/1e6:.1f} MB)")

    # 8. Build vis06_matched_1252.tar
    vis_tar_path = os.path.join(RELEASE_DIR, "vis06_matched_1252.tar")
    if not os.path.isfile(vis_tar_path):
        print(f"Building {vis_tar_path} ({len(vis_files)} files)...")
        with tarfile.open(vis_tar_path, "w") as tf:
            for p in sorted(vis_files):
                tf.add(os.path.realpath(p), arcname=f"vis0.6/{os.path.basename(p)}", filter=_anon_filter)
        print(f"Saved {vis_tar_path} ({os.path.getsize(vis_tar_path)/1e6:.1f} MB)")

    print(f"Done preparing {RELEASE_DIR} in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
