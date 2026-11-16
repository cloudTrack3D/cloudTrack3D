#!/usr/bin/env python3
"""
15-Minute and 30-Minute Future Cloud Forecasting Sweep for CloudTrack-3D.

Evaluates predicting 3D cloud mask, cloud top height (km), and cloud base height (km)
at the ICESat-2 overpass time (t = 0) using ONLY past still satellite imagery from:
  - Lead Time 0 min  (Coincident: t = 0 min -> t = 0)
  - Lead Time +15 min (15 min into future: t = -15 min -> t = 0)
  - Lead Time +30 min (30 min into future: t = -30 min -> t = 0)

Across 4 still-frame modality combinations:
  1. Still IR (1f IR, without ERA5)
  2. Still IR + VIS (1f IR+VIS, without ERA5)
  3. Still IR + ERA5 (1f IR + ERA5 T & RH)
  4. Still IR + VIS + ERA5 (1f IR+VIS + ERA5 T & RH)

Covers:
  - Part A: Dedicated Forecast-Trained Tabular Models (HistGBDT trained per lead time)
  - Part B: Deep & Foundation Models (U-Net, SAM, DINOv3; Supervised & Semi-Supervised)
            evaluated across 0 min, +15 min, and +30 min forecast lead times.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from torch.utils.data import DataLoader

sys.path.insert(0, "../data")
from dataset import (
    CloudPatchDataset,
    MatchedGranule,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import evaluate_sparse_predictions
from models import build_model, unwrap_model
from run_cvpr_experiments import preload_all_images_and_tracks, save_json, set_seed
from run_era5_experiments import (
    eval_tabular_predictions,
    extract_tabular_features_from_dataset,
    make_era5_patch_dataset,
    preload_era5_for_granules,
)


def make_leadtime_dataset(
    granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    patches_per_granule: int,
    is_train: bool,
    use_vis06: bool,
    use_era5: bool,
    lead_time_min: int,
    seed: int = 42,
) -> CloudPatchDataset:
    ds = make_era5_patch_dataset(
        granules=granules,
        img_cache=img_cache,
        track_cache=track_cache,
        era5_cache=era5_cache,
        patches_per_granule=patches_per_granule,
        is_train=is_train,
        use_3frames=False,
        use_vis06=use_vis06,
        use_era5=use_era5,
        use_satellite=True,
        seed=seed,
    )
    ds.frame_offset_minutes = lead_time_min
    return ds


def main() -> None:
    set_seed(42)
    out_json = "outputs/cvpr_experiments/forecast_leadtime_results.json"
    results: dict[str, Any] = {}
    if os.path.isfile(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)

    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    img_cache, track_cache = preload_all_images_and_tracks(granules_408)
    era5_cache = preload_era5_for_granules(granules_408)

    lead_times = [0, 15, 30]
    modalities = [
        ("1f IR", False, False, 3),
        ("1f IR+VIS", True, False, 4),
        ("1f IR + ERA5", False, True, 21),
        ("1f IR+VIS + ERA5", True, True, 22),
    ]

    # =========================================================================
    # PART A: Dedicated Forecast-Trained Tabular Models (HistGBDT) across 0, +15m, +30m
    # =========================================================================
    results.setdefault("forecast_trained_gbdt", {})
    gbdt_res = results["forecast_trained_gbdt"]

    for mod_name, use_vis, use_era5, _ in modalities:
        for lt in lead_times:
            key = f"HistGBDT ({mod_name}) [+{lt}m]"
            if key in gbdt_res:
                continue
            t0 = time.time()
            tr_ds = make_leadtime_dataset(
                train_408, img_cache, track_cache, era5_cache, 6, False, use_vis, use_era5, lt, seed=42
            )
            va_ds = make_leadtime_dataset(
                val_408, img_cache, track_cache, era5_cache, 6, False, use_vis, use_era5, lt, seed=42
            )
            X_tr, yc_tr, yt_tr, yb_tr = extract_tabular_features_from_dataset(tr_ds, max_samples=100000)
            X_va, yc_va, yt_va, yb_va = extract_tabular_features_from_dataset(va_ds, max_samples=200000)
            cloudy_tr = yc_tr > 0.5

            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr, yc_tr.astype(np.int32)
            )
            reg_t = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr[cloudy_tr], yt_tr[cloudy_tr]
            )
            reg_b = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr[cloudy_tr], yb_tr[cloudy_tr]
            )
            prob = clf.predict_proba(X_va)[:, 1]
            pt = reg_t.predict(X_va)
            pb = reg_b.predict(X_va)
            m = eval_tabular_predictions(prob, yc_va, pt, pb, yt_va, yb_va)
            gbdt_res[key] = m
            save_json(results, out_json)
            print(
                f"[GBDT Forecast] {key:36s} ({time.time()-t0:.1f}s): "
                f"Top MAE={m['top_mae_km']:.3f} km | Bot MAE={m['bot_mae_km']:.3f} km | F1={m['cloud_f1']:.4f}"
            )

    # =========================================================================
    # PART B: Deep & Foundation Model Lead-Time Sweep across 0, +15m, +30m
    # =========================================================================
    results.setdefault("deep_foundation_leadtime_sweep", {})
    deep_res = results["deep_foundation_leadtime_sweep"]

    def _pick_best_gpu() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best_g = max(range(torch.cuda.device_count()), key=lambda g: torch.cuda.mem_get_info(g)[0])
        return torch.device(f"cuda:{best_g}")

    # Checkpoints to evaluate across lead times 0m, +15m, +30m
    ckpt_specs = [
        # 1f IR (without ERA5)
        ("Supervised SAM (1f IR)", "sam", 3, False, False, "outputs/cvpr_experiments/sup_sam_3ch_408/best_model.pt"),
        ("Semi-Sup SAM (1f IR)", "sam", 3, False, False, "outputs/cvpr_experiments/semisup_sam_3ch_408/best_semisup.pt"),
        ("Supervised DINOv3 (1f IR)", "dinov3", 3, False, False, "outputs/cvpr_experiments/sup_dinov3_3ch_408/best_model.pt"),
        ("Semi-Sup DINOv3 (1f IR)", "dinov3", 3, False, False, "outputs/cvpr_experiments/semisup_dinov3_3ch_408/best_semisup.pt"),
        # 1f IR+VIS (without ERA5)
        ("Supervised SAM (1f IR+VIS)", "sam", 4, True, False, "outputs/cvpr_experiments/sup_sam_4ch_408/best_model.pt"),
        ("Semi-Sup SAM (1f IR+VIS)", "sam", 4, True, False, "outputs/cvpr_experiments/semisup_sam_4ch_408/best_semisup.pt"),
        ("Supervised DINOv3 (1f IR+VIS)", "dinov3", 4, True, False, "outputs/cvpr_experiments/sup_dinov3_4ch_408/best_model.pt"),
        ("Semi-Sup DINOv3 (1f IR+VIS)", "dinov3", 4, True, False, "outputs/cvpr_experiments/semisup_dinov3_4ch_408/best_semisup.pt"),
        # 1f IR + ERA5
        ("Supervised U-Net (1f IR + ERA5)", "unet", 21, False, True, "outputs/cvpr_experiments/era5_sup_unet_1f_ir_era5/best_model.pt"),
        ("Semi-Sup U-Net (1f IR + ERA5)", "unet", 21, False, True, "outputs/cvpr_experiments/era5_semisup_unet_1f_ir_era5/best_semisup.pt"),
        ("Supervised SAM (1f IR + ERA5)", "sam", 21, False, True, "outputs/cvpr_experiments/era5_sup_sam_1f_ir_era5/best_model.pt"),
        ("Semi-Sup SAM (1f IR + ERA5)", "sam", 21, False, True, "outputs/cvpr_experiments/era5_semisup_sam_1f_ir_era5/best_semisup.pt"),
        ("Supervised DINOv3 (1f IR + ERA5)", "dinov3", 21, False, True, "outputs/cvpr_experiments/era5_sup_dinov3_1f_ir_era5/best_model.pt"),
        ("Semi-Sup DINOv3 (1f IR + ERA5)", "dinov3", 21, False, True, "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_era5/best_semisup.pt"),
        # 1f IR+VIS + ERA5
        ("Supervised U-Net (1f IR+VIS + ERA5)", "unet", 22, True, True, "outputs/cvpr_experiments/era5_sup_unet_1f_ir_vis_era5/best_model.pt"),
        ("Semi-Sup U-Net (1f IR+VIS + ERA5)", "unet", 22, True, True, "outputs/cvpr_experiments/era5_semisup_unet_1f_ir_vis_era5/best_semisup.pt"),
        ("Supervised SAM (1f IR+VIS + ERA5)", "sam", 22, True, True, "outputs/cvpr_experiments/era5_sup_sam_1f_ir_vis_era5/best_model.pt"),
        ("Semi-Sup SAM (1f IR+VIS + ERA5)", "sam", 22, True, True, "outputs/cvpr_experiments/era5_semisup_sam_1f_ir_vis_era5/best_semisup.pt"),
        ("Supervised DINOv3 (1f IR+VIS + ERA5)", "dinov3", 22, True, True, "outputs/cvpr_experiments/era5_sup_dinov3_1f_ir_vis_era5/best_model.pt"),
        ("Semi-Sup DINOv3 (1f IR+VIS + ERA5)", "dinov3", 22, True, True, "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_vis_era5/best_semisup.pt"),
    ]

    for label, arch, in_ch, use_vis, use_era5, ckpt_path in ckpt_specs:
        if not os.path.isfile(ckpt_path):
            print(f"[Skip] Checkpoint not found yet: {ckpt_path}")
            continue

        need_any = any(f"{label} [+{lt}m]" not in deep_res for lt in lead_times)
        if not need_any:
            continue

        device = _pick_best_gpu()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state")))
        base_ch = ckpt.get("args", {}).get("base_ch", 48)
        model = build_model(arch=arch, in_channels=in_ch, base_ch=base_ch, pretrained=False)
        unwrap_model(model).load_state_dict(state)
        model = model.to(device)
        model.eval()

        for lt in lead_times:
            key = f"{label} [+{lt}m]"
            if key in deep_res:
                continue
            t0 = time.time()
            va_ds = make_leadtime_dataset(
                val_408, img_cache, track_cache, era5_cache, 6, False, use_vis, use_era5, lt, seed=42
            )
            va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=0)
            m = evaluate_sparse_predictions(model, va_loader, device)
            deep_res[key] = m
            save_json(results, out_json)
            print(
                f"[Lead-Time Eval] {key:42s} ({time.time()-t0:.1f}s): "
                f"Top MAE={m['top_mae_km']:.3f} km | Bot MAE={m['bot_mae_km']:.3f} km | F1={m['cloud_f1']:.4f}"
            )

        del model
        torch.cuda.empty_cache()

    print("\n=== FORECAST LEAD-TIME SWEEP COMPLETED! ===")


if __name__ == "__main__":
    main()
