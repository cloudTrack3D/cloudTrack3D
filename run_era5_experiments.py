#!/usr/bin/env python3
"""
Comprehensive ERA5 Reanalysis Experiments for CloudTrack-3D (CVPR 2027).

Evaluates ERA5 atmospheric profiles (9 pressure levels [200..900 hPa] of
Temperature and Relative Humidity = 18 channels) across 5 input configurations:
  1. ERA5 Alone (no satellite imagery)               [20 channels]
  2. 1-Frame IR + ERA5                               [21 channels]
  3. 1-Frame IR + VIS + ERA5                         [22 channels]
  4. 3-Frame IR (t=0, -15m, -30m) + ERA5             [23 channels]
  5. 3-Frame IR + VIS (t=0, -15m, -30m) + ERA5       [26 channels]

Across classical tabular baselines (Ridge/Logistic, HistGBDT) and deep/foundation
architectures (U-Net, SAM-2, DINOv3) under both Supervised and Semi-Supervised
(EMA Teacher + Dual-Stream Consistency + Feature Perturbation) regimes.
"""

from __future__ import annotations

import datetime
import gc
import json
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "../data")
from era5_db import COMMON_9_PRESSURE_LEVELS, ERA5Database

from dataset import (
    DEFAULT_BBOX,
    DEFAULT_IMAGE_SHAPE,
    MAX_HEIGHT_KM,
    CloudPatchDataset,
    MatchedGranule,
    discover_matched_granules,
    filter_granules_with_30min_window,
    sample_era5_patch,
    split_granules_chronologically,
)
from losses import (
    KM_TO_KFT,
    THRESHOLD_5KM,
    THRESHOLD_15KFT_KM,
    _compute_subset_error_stats,
)
from models import build_model
from run_cvpr_experiments import (
    preload_all_images_and_tracks,
    run_jobs_across_4_gpus,
    save_json,
    set_seed,
    train_and_eval_semisup_single_gpu,
    train_and_eval_supervised_single_gpu,
)


def preload_era5_for_granules(
    granules: list[MatchedGranule],
    era5_root: str = "../data/era5_npy",
) -> dict[str, np.ndarray]:
    """
    Preload and normalize 18-channel ERA5 (9 Temperature + 9 Relative Humidity levels)
    grids of shape (18, 61, 77) for every granule timestamp.
    """
    t0 = time.time()
    db = ERA5Database(era5_root)
    era5_cache: dict[str, np.ndarray] = {}
    for g in granules:
        dt = datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc)
        prof_t = db.load_profile(dt, "temperature", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
        prof_rh = db.load_profile(dt, "relative_humidity", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
        norm_t = np.clip((prof_t - 200.0) / 120.0, 0.0, 1.0)
        norm_rh = np.clip(prof_rh / 100.0, 0.0, 1.0)
        era5_cache[g.granule_id] = np.concatenate([norm_t, norm_rh], axis=0).astype(np.float32)
    print(f"[Preload ERA5] Loaded {len(era5_cache)} (18, 61, 77) profiles in {time.time() - t0:.2f}s.")
    return era5_cache


def make_era5_patch_dataset(
    granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    patches_per_granule: int,
    is_train: bool,
    use_3frames: bool,
    use_vis06: bool,
    use_era5: bool = True,
    use_satellite: bool = True,
    seed: int = 42,
) -> CloudPatchDataset:
    ds = CloudPatchDataset.__new__(CloudPatchDataset)
    ds.granules = granules
    ds.patch_size = 256
    ds.patches_per_granule = patches_per_granule
    ds.bbox = DEFAULT_BBOX
    ds.cloud_attr_mode = "cloud_only"
    ds.is_train = is_train
    ds.rng = np.random.default_rng(seed)
    ds.max_cached_images = 100000
    ds.use_3frames = use_3frames
    ds.use_vis06 = use_vis06
    ds.use_era5 = use_era5
    ds.use_satellite = use_satellite
    ds.ir_dir = "../ir10.5"
    ds.vis_dir = "../vis0.6"
    ds._era5_cache = era5_cache
    ds.allowed_beams = (1, 2, 3)
    ds._img_cache = img_cache

    ds.track_cache = []
    ds.valid_granule_indices = []
    for idx, g in enumerate(granules):
        tr = track_cache[g.granule_id]
        ds.track_cache.append(tr)
        if len(tr["row"]) > 0:
            ds.valid_granule_indices.append(idx)
    return ds


def extract_tabular_features_from_dataset(
    ds: CloudPatchDataset,
    max_samples: int = 120000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pixel-level features X and targets (y_cls, y_top_km, y_bot_km) at valid lidar pixels."""
    xs, y_cls, y_top, y_bot = [], [], [], []
    for i in range(len(ds)):
        item = ds[i]
        vm = item["valid_mask"].numpy() > 0.5
        if not np.any(vm):
            continue
        img = item["image"].numpy()  # (C, H, W)
        pts = img[:, vm].T           # (N, C)
        xs.append(pts)
        y_cls.append(item["cloud_mask"].numpy()[vm])
        y_top.append(item["cloud_top"].numpy()[vm] * MAX_HEIGHT_KM)
        y_bot.append(item["cloud_bot"].numpy()[vm] * MAX_HEIGHT_KM)

    X = np.concatenate(xs, axis=0)
    yc = np.concatenate(y_cls, axis=0)
    yt = np.concatenate(y_top, axis=0)
    yb = np.concatenate(y_bot, axis=0)

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, yc, yt, yb = X[idx], yc[idx], yt[idx], yb[idx]
    return X, yc, yt, yb


def eval_tabular_predictions(
    prob: np.ndarray,
    yc: np.ndarray,
    pred_top: np.ndarray,
    pred_bot: np.ndarray,
    yt: np.ndarray,
    yb: np.ndarray,
) -> dict[str, float]:
    pred_cls = (prob >= 0.5).astype(np.float32)
    tp = float(np.sum((pred_cls == 1) & (yc == 1)))
    fp = float(np.sum((pred_cls == 1) & (yc == 0)))
    fn = float(np.sum((pred_cls == 0) & (yc == 1)))
    tn = float(np.sum((pred_cls == 0) & (yc == 0)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1.0)
    auroc = float(roc_auc_score(yc.astype(np.int32), prob)) if len(np.unique(yc)) > 1 else 0.5

    cloudy = yc > 0.5
    pt = np.clip(pred_top[cloudy], 0.0, MAX_HEIGHT_KM)
    pb = np.clip(np.minimum(pred_bot[cloudy], pt), 0.0, MAX_HEIGHT_KM)
    gt_t = yt[cloudy]
    gt_b = yb[cloudy]

    res: dict[str, float] = {
        "cloud_acc": float(acc),
        "cloud_precision": float(prec),
        "cloud_recall": float(rec),
        "cloud_f1": float(f1),
        "cloud_iou": float(iou),
        "cloud_auroc": float(auroc),
    }
    res.update(_compute_subset_error_stats(pt, gt_t, "top"))
    res.update(_compute_subset_error_stats(pb, gt_b, "bot"))
    res.update(_compute_subset_error_stats(pt[gt_t <= THRESHOLD_5KM], gt_t[gt_t <= THRESHOLD_5KM], "top_le5km"))
    res.update(_compute_subset_error_stats(pb[gt_b <= THRESHOLD_5KM], gt_b[gt_b <= THRESHOLD_5KM], "bot_le5km"))
    res.update(_compute_subset_error_stats(pt[gt_t <= THRESHOLD_15KFT_KM], gt_t[gt_t <= THRESHOLD_15KFT_KM], "top_le15kft"))
    res.update(_compute_subset_error_stats(pb[gt_b <= THRESHOLD_15KFT_KM], gt_b[gt_b <= THRESHOLD_15KFT_KM], "bot_le15kft"))
    return res


def run_tabular_era5_baselines(
    train_408: list[MatchedGranule],
    val_408: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    era5_res: dict[str, Any],
    out_json: str,
    all_res: dict[str, Any],
) -> None:
    configs = [
        ("ERA5 Only", False, False, True, False),
        ("1f IR + ERA5", False, False, True, True),
        ("1f IR+VIS + ERA5", False, True, True, True),
        ("3f IR + ERA5", True, False, True, True),
        ("3f IR+VIS + ERA5", True, True, True, True),
    ]
    for tag, u3, uv, ue, us in configs:
        gbdt_key = f"HistGBDT ({tag})"
        lin_key = f"Linear/Ridge ({tag})"
        if gbdt_key in era5_res and (tag != "ERA5 Only" or lin_key in era5_res):
            continue

        print(f"[Tabular ERA5] Extracting features for {tag}...")
        tr_ds = make_era5_patch_dataset(train_408, img_cache, track_cache, era5_cache, 6, False, u3, uv, ue, us, seed=42)
        va_ds = make_era5_patch_dataset(val_408, img_cache, track_cache, era5_cache, 6, False, u3, uv, ue, us, seed=42)
        X_tr, yc_tr, yt_tr, yb_tr = extract_tabular_features_from_dataset(tr_ds, max_samples=100000)
        X_va, yc_va, yt_va, yb_va = extract_tabular_features_from_dataset(va_ds, max_samples=200000)
        cloudy_tr = yc_tr > 0.5

        if tag == "ERA5 Only" and lin_key not in era5_res:
            clf_lin = LogisticRegression(max_iter=300, C=1.0).fit(X_tr, yc_tr.astype(np.int32))
            reg_t_lin = Ridge(alpha=1.0).fit(X_tr[cloudy_tr], yt_tr[cloudy_tr])
            reg_b_lin = Ridge(alpha=1.0).fit(X_tr[cloudy_tr], yb_tr[cloudy_tr])
            prob_lin = clf_lin.predict_proba(X_va)[:, 1]
            pt_lin = reg_t_lin.predict(X_va)
            pb_lin = reg_b_lin.predict(X_va)
            era5_res[lin_key] = eval_tabular_predictions(prob_lin, yc_va, pt_lin, pb_lin, yt_va, yb_va)
            print(
                f"  -> {lin_key}: Top MAE={era5_res[lin_key]['top_mae_km']:.3f} km | "
                f"Bot MAE={era5_res[lin_key]['bot_mae_km']:.3f} km | F1={era5_res[lin_key]['cloud_f1']:.4f}"
            )
            save_json(all_res, out_json)

        if gbdt_key not in era5_res:
            clf_gb = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr, yc_tr.astype(np.int32)
            )
            reg_t_gb = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr[cloudy_tr], yt_tr[cloudy_tr]
            )
            reg_b_gb = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42).fit(
                X_tr[cloudy_tr], yb_tr[cloudy_tr]
            )
            prob_gb = clf_gb.predict_proba(X_va)[:, 1]
            pt_gb = reg_t_gb.predict(X_va)
            pb_gb = reg_b_gb.predict(X_va)
            era5_res[gbdt_key] = eval_tabular_predictions(prob_gb, yc_va, pt_gb, pb_gb, yt_va, yb_va)
            print(
                f"  -> {gbdt_key}: Top MAE={era5_res[gbdt_key]['top_mae_km']:.3f} km | "
                f"Bot MAE={era5_res[gbdt_key]['bot_mae_km']:.3f} km | F1={era5_res[gbdt_key]['cloud_f1']:.4f}"
            )
            save_json(all_res, out_json)


def main() -> None:
    set_seed(42)
    out_json = "outputs/cvpr_experiments/cvpr_all_results.json"
    all_res: dict[str, Any] = {}
    if os.path.isfile(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            all_res = json.load(f)

    all_res.setdefault("era5_experiments", {})
    era5_res = all_res["era5_experiments"]

    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    img_cache, track_cache = preload_all_images_and_tracks(granules_408)
    era5_cache = preload_era5_for_granules(granules_408)

    # 1. Run Classical Tabular Baselines with ERA5
    run_tabular_era5_baselines(
        train_408, val_408, img_cache, track_cache, era5_cache, era5_res, out_json, all_res
    )

    # 2. Define the 5 ERA5 Modality / Temporal Settings
    # (tag, short_id, use_3frames, use_vis06, use_era5, use_satellite, in_channels)
    era5_settings = [
        ("ERA5 Only", "era5_only", False, False, True, False, 20),
        ("1f IR + ERA5", "1f_ir_era5", False, False, True, True, 21),
        ("1f IR+VIS + ERA5", "1f_ir_vis_era5", False, True, True, True, 22),
        ("3f IR + ERA5", "3f_ir_era5", True, False, True, True, 23),
        ("3f IR+VIS + ERA5", "3f_ir_vis_era5", True, True, True, True, 26),
    ]
    architectures = [
        ("U-Net", "unet"),
        ("SAM", "sam"),
        ("DINOv3", "dinov3"),
    ]

    # =========================================================================
    # WAVE 1: Supervised ERA5 Models across 4 GPUs
    # =========================================================================
    sup_jobs: list[Callable[[int], None]] = []
    for tag, short_id, u3, uv, ue, us, in_ch in era5_settings:
        for arch_label, arch_code in architectures:
            sup_key = f"Supervised {arch_label} ({tag})"
            sup_dir = f"outputs/cvpr_experiments/era5_sup_{arch_code}_{short_id}"
            if sup_key not in era5_res:
                def _make_sup_job(
                    _k=sup_key,
                    _ac=arch_code,
                    _ic=in_ch,
                    _u3=u3,
                    _uv=uv,
                    _ue=ue,
                    _us=us,
                    _sd=sup_dir,
                ) -> Callable[[int], None]:
                    def _run(gpu_id: int) -> None:
                        tr_ds = make_era5_patch_dataset(
                            train_408, img_cache, track_cache, era5_cache, 6, True, _u3, _uv, _ue, _us, seed=42 + gpu_id
                        )
                        va_ds = make_era5_patch_dataset(
                            val_408, img_cache, track_cache, era5_cache, 6, False, _u3, _uv, _ue, _us, seed=42
                        )
                        res = train_and_eval_supervised_single_gpu(
                            _k,
                            lambda: build_model(arch=_ac, in_channels=_ic, base_ch=48, pretrained=True),
                            tr_ds,
                            va_ds,
                            _sd,
                            gpu_id=gpu_id,
                            epochs=10,
                            batch_size=16,
                        )
                        era5_res[_k] = res
                        save_json(all_res, out_json)
                    return _run
                sup_jobs.append(_make_sup_job())

    print(f"\n=== ERA5 WAVE 1: Launching {len(sup_jobs)} Supervised Jobs across 4 GPUs ===")
    run_jobs_across_4_gpus(sup_jobs)

    # =========================================================================
    # WAVE 2: Semi-Supervised ERA5 Models across 4 GPUs
    # =========================================================================
    semi_jobs: list[Callable[[int], None]] = []
    for tag, short_id, u3, uv, ue, us, in_ch in era5_settings:
        for arch_label, arch_code in architectures:
            semi_key = f"Semi-Sup {arch_label} ({tag})"
            sup_dir = f"outputs/cvpr_experiments/era5_sup_{arch_code}_{short_id}"
            semi_dir = f"outputs/cvpr_experiments/era5_semisup_{arch_code}_{short_id}"
            if semi_key not in era5_res:
                def _make_semi_job(
                    _k=semi_key,
                    _ac=arch_code,
                    _ic=in_ch,
                    _u3=u3,
                    _uv=uv,
                    _ue=ue,
                    _us=us,
                    _sd=sup_dir,
                    _md=semi_dir,
                ) -> Callable[[int], None]:
                    def _run(gpu_id: int) -> None:
                        tr_ds = make_era5_patch_dataset(
                            train_408, img_cache, track_cache, era5_cache, 6, True, _u3, _uv, _ue, _us, seed=142 + gpu_id
                        )
                        va_ds = make_era5_patch_dataset(
                            val_408, img_cache, track_cache, era5_cache, 6, False, _u3, _uv, _ue, _us, seed=42
                        )
                        res = train_and_eval_semisup_single_gpu(
                            _k,
                            lambda: build_model(arch=_ac, in_channels=_ic, base_ch=48, pretrained=False),
                            os.path.join(_sd, "best_model.pt"),
                            tr_ds,
                            va_ds,
                            _md,
                            gpu_id=gpu_id,
                            epochs=10,
                            batch_size=12,
                        )
                        era5_res[_k] = res
                        save_json(all_res, out_json)
                    return _run
                semi_jobs.append(_make_semi_job())

    print(f"\n=== ERA5 WAVE 2: Launching {len(semi_jobs)} Semi-Supervised Jobs across 4 GPUs ===")
    run_jobs_across_4_gpus(semi_jobs)

    print("\n=== ALL ERA5 EXPERIMENTS COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
