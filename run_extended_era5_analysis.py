#!/usr/bin/env python3
"""
Extended Experimental & Analytical Suite for CloudTrack-3D (CVPR 2027).

Executes five experimental tracks in exact priority order:
  Track 1: Direct Physical Matching Between ERA5 & Satellite Imagery
           - 1A: Physics-based ERA5 Thermal/Moisture Profile Matching baseline
           - 1B: Spatio-Temporal Co-Registration & Perturbation Sweep
                 (dt in {0h, -1h, -3h, -6h, -12h, -24h, shuffled}, d_deg in {0, 0.25, 0.5, 1.0, 2.0})
  Track 2: Disentangling ERA5 Physical Variables (T vs RH) & Vertical Pressure Slabs
           - 2A: Temperature Only (9 ch) vs Relative Humidity Only (9 ch) vs Full T+RH (18 ch)
           - 2B: Low Boundary-Layer (800-900 hPa) vs Mid (500-700 hPa) vs High (200-400 hPa)
  Track 3: Full 490-Granule All-Day (Day vs Deep-Night) Benchmark with 1f IR + ERA5
           - Evaluates whether ERA5 substitutes for missing VIS 0.6 at night across all 490 granules
  Track 4: Multi-Frame Motion Nowcasting ([t=-30m, t=-15m] -> t=0m with & without ERA5)
           - Compares 2-frame past motion trajectory nowcasting (+15m ahead) against single-frame
  Track 5: Satellite-Sequence-to-ERA5 Contrastive Alignment (SatERA5-CLIP)
           - Dual-encoder CLIP contrastive learning (global + dense spatial InfoNCE) between
             multi-temporal satellite sequences and 9-level ERA5 thermodynamic profiles,
             cross-modal retrieval (R@1, R@5, R@10), virtual ERA5 profile prediction, and
             CLIP-distilled satellite-only 3D cloud height reconstruction.
"""

from __future__ import annotations

import copy
import datetime
import gc
import json
import os
import sys
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import DataLoader

sys.path.insert(0, "../data")
from era5_db import COMMON_9_PRESSURE_LEVELS, ERA5Database

from dataset import (
    DEFAULT_BBOX,
    MAX_HEIGHT_KM,
    CloudPatchDataset,
    MatchedGranule,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import evaluate_sparse_predictions
from models import RadiometricContextStem, build_model, unwrap_model
from run_cvpr_experiments import (
    preload_all_images_and_tracks,
    run_jobs_across_4_gpus,
    save_json,
    set_seed,
    train_and_eval_semisup_single_gpu,
    train_and_eval_supervised_single_gpu,
)
from run_era5_experiments import (
    eval_tabular_predictions,
    extract_tabular_features_from_dataset,
    make_era5_patch_dataset,
    preload_era5_for_granules,
)

# Standard ISA geometric altitudes (km) for COMMON_9_PRESSURE_LEVELS:
# [200, 300, 400, 500, 600, 700, 800, 850, 900] hPa
ISA_ALTITUDES_KM = np.array([11.78, 9.16, 7.19, 5.57, 4.21, 3.01, 1.95, 1.46, 0.99], dtype=np.float32)

OUT_JSON_PATH = "outputs/cvpr_experiments/extended_era5_analysis_results.json"
SAVE_LOCK = threading.Lock()


def load_or_init_results() -> dict[str, Any]:
    if os.path.exists(OUT_JSON_PATH):
        try:
            with open(OUT_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_incremental(results: dict[str, Any]) -> None:
    with SAVE_LOCK:
        os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
        save_json(results, OUT_JSON_PATH)


def fit_and_eval_histgbdt(
    ds_tr: CloudPatchDataset,
    ds_va: CloudPatchDataset,
    label: str = "",
) -> dict[str, float]:
    X_tr, yc_tr, yt_tr, yb_tr = extract_tabular_features_from_dataset(ds_tr, max_samples=100000)
    X_va, yc_va, yt_va, yb_va = extract_tabular_features_from_dataset(ds_va, max_samples=200000)
    cloudy_tr = yc_tr > 0.5
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
    res = eval_tabular_predictions(prob_gb, yc_va, pt_gb, pb_gb, yt_va, yb_va)
    print(
        f"  [HistGBDT {label}] Top MAE={res['top_mae_km']:.3f} km | "
        f"Bot MAE={res['bot_mae_km']:.3f} km | F1={res['cloud_f1']:.4f}"
    )
    return res


def load_ckpt_into_model(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get(
        "model_state_dict",
        ckpt.get("teacher_state", ckpt.get("model_state", ckpt.get("student_state", ckpt))),
    )
    unwrap_model(model).load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ==============================================================================
# TRACK 1: DIRECT PHYSICAL MATCHING & SPATIO-TEMPORAL PERTURBATION SWEEP
# ==============================================================================

def run_track1_physical_matching(
    train_granules: list[MatchedGranule],
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    split_label: str = "408_split",
) -> dict[str, Any]:
    """
    1A. Physics-Based ERA5-Satellite Level Matching Baseline:
        Matches satellite IR 10.5 um thermal coldness (and VIS 0.6 um optical thickness)
        against the co-located 9-level ERA5 Temperature profile T(z_k) and Relative
        Humidity profile RH(z_k) at z_k in [0.99 .. 11.78 km].
    """
    print(f"\n--- [Track 1A] Running Physics-Based ERA5-Satellite Profile Matching ({split_label}) ---")
    has_vis = (split_label == "408_split")
    ds_train = make_era5_patch_dataset(
        train_granules, img_cache, track_cache, era5_cache,
        patches_per_granule=4, is_train=False, use_3frames=False, use_vis06=has_vis,
        use_era5=True, use_satellite=True, seed=101,
    )
    ds_val = make_era5_patch_dataset(
        val_granules, img_cache, track_cache, era5_cache,
        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=has_vis,
        use_era5=True, use_satellite=True, seed=102,
    )

    X_tr, y_cls_tr, y_top_tr, y_bot_tr = extract_tabular_features_from_dataset(ds_train, max_samples=100000)
    X_va, y_cls_va, y_top_va, y_bot_va = extract_tabular_features_from_dataset(ds_val, max_samples=200000)

    # Channel layout of X:
    # If has_vis: [IR_0, VIS_0, Lat, Lon, ERA5_T(9), ERA5_RH(9)] -> ERA5 starts at idx 4
    # If not has_vis: [IR_0, Lat, Lon, ERA5_T(9), ERA5_RH(9)] -> ERA5 starts at idx 3
    era5_start = 4 if has_vis else 3
    ir_tr = X_tr[:, 0]
    ir_va = X_va[:, 0]
    vis_tr = X_tr[:, 1] if has_vis else np.zeros_like(ir_tr)
    vis_va = X_va[:, 1] if has_vis else np.zeros_like(ir_va)

    t_prof_tr = X_tr[:, era5_start:era5_start + 9] * 120.0 + 200.0      # Kelvin [200..900 hPa]
    rh_prof_tr = X_tr[:, era5_start + 9:era5_start + 18] * 100.0        # % [200..900 hPa]
    t_prof_va = X_va[:, era5_start:era5_start + 9] * 120.0 + 200.0      # Kelvin [200..900 hPa]
    rh_prof_va = X_va[:, era5_start + 9:era5_start + 18] * 100.0        # % [200..900 hPa]

    cld_tr = y_cls_tr > 0.5
    z_asc = ISA_ALTITUDES_KM[::-1]  # [0.99, 1.46, 1.95, 3.01, 4.21, 5.57, 7.19, 9.16, 11.78]
    t_asc_tr = t_prof_tr[:, ::-1]
    true_t_at_top_tr = np.array([
        np.interp(y_top_tr[i], z_asc, t_asc_tr[i]) for i in np.where(cld_tr)[0]
    ], dtype=np.float32)

    # Fit 3-parameter physical brightness temperature calibration:
    # T_b_eff = T_900 - (c_ir * IR_0 + c_vis * VIS_0 + c_0)
    t_900_tr = t_prof_tr[cld_tr, 8]
    delta_t_tr = t_900_tr - true_t_at_top_tr
    A_cal = np.column_stack([ir_tr[cld_tr], vis_tr[cld_tr], np.ones(np.sum(cld_tr))])
    cal_coef, _, _, _ = np.linalg.lstsq(A_cal, delta_t_tr, rcond=None)
    c_ir, c_vis, c_0 = float(cal_coef[0]), float(cal_coef[1]), float(cal_coef[2])
    print(f"  Calibrated T_b depression: dT = {c_ir:.2f}*IR + {c_vis:.2f}*VIS + {c_0:.2f} K")

    t_900_va = t_prof_va[:, 8]
    tb_va = t_900_va - (c_ir * ir_va + c_vis * vis_va + c_0)

    N_va = len(ir_va)
    pred_top_thermal = np.zeros(N_va, dtype=np.float32)
    pred_bot_thermal = np.zeros(N_va, dtype=np.float32)
    pred_top_coupled = np.zeros(N_va, dtype=np.float32)
    pred_bot_coupled = np.zeros(N_va, dtype=np.float32)

    mean_thick_tr = float(np.mean(np.maximum(y_top_tr[cld_tr] - y_bot_tr[cld_tr], 0.2)))

    for i in range(N_va):
        t_col = t_prof_va[i]     # [200..900 hPa], T increases with index (200 hPa is coldest, 900 hPa is warmest)
        rh_col = rh_prof_va[i]   # [200..900 hPa]
        t_mono = np.maximum.accumulate(t_col)
        h_thermal = float(np.interp(tb_va[i], t_mono, ISA_ALTITUDES_KM))
        h_thermal = float(np.clip(h_thermal, 0.5, 15.0))
        pred_top_thermal[i] = h_thermal
        pred_bot_thermal[i] = max(0.2, h_thermal - mean_thick_tr)

        # Coupled ERA5 Thermal + RH Saturation Matching:
        thermal_w = np.exp(-np.abs(t_col - tb_va[i]) / 12.0)
        moist_w = np.power(np.clip(rh_col / 100.0, 0.05, 1.0), 1.5)
        joint_w = thermal_w * moist_w
        joint_w /= (np.sum(joint_w) + 1e-8)
        h_rh_weighted = float(np.sum(joint_w * ISA_ALTITUDES_KM))
        h_top_c = 0.55 * h_thermal + 0.45 * h_rh_weighted
        pred_top_coupled[i] = h_top_c

        moist_levels = np.where((rh_col >= 62.0) & (ISA_ALTITUDES_KM <= h_top_c + 0.5))[0]
        if len(moist_levels) > 0:
            lowest_moist_idx = moist_levels[-1]
            h_bot_rh = float(ISA_ALTITUDES_KM[lowest_moist_idx])
            pred_bot_coupled[i] = float(np.clip(0.5 * h_bot_rh + 0.5 * max(0.2, h_top_c - mean_thick_tr), 0.1, h_top_c))
        else:
            pred_bot_coupled[i] = max(0.2, h_top_c - mean_thick_tr)

    max_rh_va = np.max(rh_prof_va, axis=1) / 100.0
    if has_vis:
        score_va = np.clip(0.45 * (ir_va / 0.45) + 0.30 * (vis_va / 0.45) + 0.25 * max_rh_va, 0.0, 1.0)
    else:
        score_va = np.clip(0.65 * (ir_va / 0.45) + 0.35 * max_rh_va, 0.0, 1.0)
    prob_va = 1.0 / (1.0 + np.exp(-5.0 * (score_va - 0.52)))

    res_thermal = eval_tabular_predictions(prob_va, y_cls_va, pred_top_thermal, pred_bot_thermal, y_top_va, y_bot_va)
    res_coupled = eval_tabular_predictions(prob_va, y_cls_va, pred_top_coupled, pred_bot_coupled, y_top_va, y_bot_va)

    print(
        f"  [{split_label}] ERA5 Thermal Profile Match -> Top MAE={res_thermal['top_mae_km']:.3f} km | "
        f"Bot MAE={res_thermal['bot_mae_km']:.3f} km | F1={res_thermal['cloud_f1']:.4f}"
    )
    print(
        f"  [{split_label}] ERA5 Thermal+RH Coupled   -> Top MAE={res_coupled['top_mae_km']:.3f} km | "
        f"Bot MAE={res_coupled['bot_mae_km']:.3f} km | F1={res_coupled['cloud_f1']:.4f}"
    )
    return {
        "calibration": {"c_ir": c_ir, "c_vis": c_vis, "c_0": c_0, "mean_thick_km": mean_thick_tr},
        "era5_thermal_profile_inversion": res_thermal,
        "era5_thermal_rh_coupled_matching": res_coupled,
    }


def build_perturbed_era5_cache(
    granules: list[MatchedGranule],
    base_era5_cache: dict[str, np.ndarray],
    dt_hours: float = 0.0,
    shift_deg: float = 0.0,
    shuffle_dates: bool = False,
    era5_root: str = "../data/era5_npy",
) -> dict[str, np.ndarray]:
    if shuffle_dates:
        rng = np.random.default_rng(999)
        keys = list(base_era5_cache.keys())
        shuffled_keys = list(rng.permutation(keys))
        return {k: base_era5_cache[sk].copy() for k, sk in zip(keys, shuffled_keys)}

    out_cache: dict[str, np.ndarray] = {}
    if abs(dt_hours) > 1e-5:
        db = ERA5Database(era5_root)
        for g in granules:
            dt = datetime.datetime.fromtimestamp(
                g.timestamp_utc + dt_hours * 3600.0, tz=datetime.timezone.utc
            )
            try:
                prof_t = db.load_profile(dt, "temperature", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
                prof_rh = db.load_profile(dt, "relative_humidity", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
                norm_t = np.clip((prof_t - 200.0) / 120.0, 0.0, 1.0)
                norm_rh = np.clip(prof_rh / 100.0, 0.0, 1.0)
                arr = np.concatenate([norm_t, norm_rh], axis=0).astype(np.float32)
            except Exception:
                arr = base_era5_cache[g.granule_id].copy()
            out_cache[g.granule_id] = arr
    else:
        for g in granules:
            out_cache[g.granule_id] = base_era5_cache[g.granule_id].copy()

    if abs(shift_deg) > 1e-5:
        shift_cells = int(round(shift_deg / 0.25))
        for gid, arr in out_cache.items():
            shifted = np.roll(np.roll(arr, shift_cells, axis=1), shift_cells, axis=2)
            out_cache[gid] = shifted

    return out_cache


def run_track1_spatiotemporal_perturbation_sweep(
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    base_era5_cache: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    """
    1B. Spatio-Temporal Co-Registration & Perturbation Sweep:
        Evaluates both physical correlation (IR/VIS vs ERA5) and trained Satellite+ERA5
        foundation models across temporal lags (0h, -1h, -3h, -6h, -12h, -24h, shuffled)
        and spatial shifts (0.0, 0.25, 0.50, 1.00, 2.00 deg).
    """
    print("\n--- [Track 1B] Running Spatio-Temporal ERA5 <-> Satellite Perturbation Sweep ---")
    perturbations = [
        {"name": "exact_0h_0deg", "dt_hours": 0.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "temporal_lag_1h", "dt_hours": -1.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "temporal_lag_3h", "dt_hours": -3.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "temporal_lag_6h", "dt_hours": -6.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "temporal_lag_12h", "dt_hours": -12.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "temporal_lag_24h", "dt_hours": -24.0, "shift_deg": 0.0, "shuffle": False},
        {"name": "spatial_shift_0.25deg_28km", "dt_hours": 0.0, "shift_deg": 0.25, "shuffle": False},
        {"name": "spatial_shift_0.50deg_56km", "dt_hours": 0.0, "shift_deg": 0.50, "shuffle": False},
        {"name": "spatial_shift_1.00deg_111km", "dt_hours": 0.0, "shift_deg": 1.00, "shuffle": False},
        {"name": "spatial_shift_2.00deg_222km", "dt_hours": 0.0, "shift_deg": 2.00, "shuffle": False},
        {"name": "shuffled_random_date", "dt_hours": 0.0, "shift_deg": 0.0, "shuffle": True},
    ]

    ckpts_to_test = [
        ("sup_sam_1f_ir_vis_era5", "sam", 22, False, True, "outputs/cvpr_experiments/era5_sup_sam_1f_ir_vis_era5/best_model.pt"),
        ("semi_dinov3_1f_ir_vis_era5", "dinov3", 22, False, True, "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_vis_era5/best_semisup.pt"),
        ("semi_dinov3_1f_ir_era5", "dinov3", 21, False, False, "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_era5/best_semisup.pt"),
    ]
    loaded_models: dict[str, tuple[nn.Module, bool, bool]] = {}
    for tag, arch, in_ch, use_3f, use_vis, path in ckpts_to_test:
        if os.path.exists(path):
            m = build_model(arch=arch, in_channels=in_ch, base_ch=48, pretrained=False)
            m = load_ckpt_into_model(m, path, device)
            loaded_models[tag] = (m, use_3f, use_vis)

    sweep_results: dict[str, Any] = {}
    for p in perturbations:
        pname = p["name"]
        pert_cache = build_perturbed_era5_cache(
            val_granules, base_era5_cache,
            dt_hours=p["dt_hours"], shift_deg=p["shift_deg"], shuffle_dates=p["shuffle"]
        )
        ds_val_vis = make_era5_patch_dataset(
            val_granules, img_cache, track_cache, pert_cache,
            patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=True,
            use_era5=True, use_satellite=True, seed=42,
        )
        X_va, y_cls_va, _, _ = extract_tabular_features_from_dataset(ds_val_vis, max_samples=150000)
        ir_v = X_va[:, 0]
        vis_v = X_va[:, 1]
        rh_prof = X_va[:, 13:22]
        col_max_rh = np.max(rh_prof, axis=1)
        rh_300 = rh_prof[:, 1]   # 300 hPa
        rh_850 = rh_prof[:, 7]   # 850 hPa
        r_ir_maxrh = float(np.corrcoef(ir_v, col_max_rh)[0, 1])
        r_ir_rh300 = float(np.corrcoef(ir_v, rh_300)[0, 1])
        r_vis_rh850 = float(np.corrcoef(vis_v, rh_850)[0, 1])
        r_cld_maxrh = float(np.corrcoef(y_cls_va, col_max_rh)[0, 1])

        entry: dict[str, Any] = {
            "dt_hours": p["dt_hours"],
            "shift_deg": p["shift_deg"],
            "shuffle": p["shuffle"],
            "physical_correlations": {
                "r_ir_vs_era5_max_rh": r_ir_maxrh,
                "r_ir_vs_era5_rh300hPa": r_ir_rh300,
                "r_vis_vs_era5_rh850hPa": r_vis_rh850,
                "r_lidar_cloud_vs_era5_max_rh": r_cld_maxrh,
            },
            "models": {},
        }

        for tag, (model, use_3f, use_vis) in loaded_models.items():
            ds_eval = ds_val_vis if use_vis else make_era5_patch_dataset(
                val_granules, img_cache, track_cache, pert_cache,
                patches_per_granule=6, is_train=False, use_3frames=use_3f, use_vis06=use_vis,
                use_era5=True, use_satellite=True, seed=42,
            )
            loader = DataLoader(ds_eval, batch_size=16, shuffle=False, num_workers=0)
            metrics = evaluate_sparse_predictions(model, loader, device)
            entry["models"][tag] = {
                "cloud_f1": metrics["cloud_f1"],
                "cloud_auroc": metrics["cloud_auroc"],
                "top_mae_km": metrics["top_mae_km"],
                "bot_mae_km": metrics["bot_mae_km"],
                "top_r2": metrics["top_r2"],
                "bot_le15kft_mae_km": metrics.get("bot_le15kft_mae_km", 0.0),
            }
        print(
            f"  [{pname:28s}] r(IR,MaxRH)={r_ir_maxrh:+.4f} | "
            f"SAM(1fIR+VIS+ERA5) Top={entry['models']['sup_sam_1f_ir_vis_era5']['top_mae_km']:.3f} | "
            f"DINOv3(1fIR+ERA5) Top={entry['models']['semi_dinov3_1f_ir_era5']['top_mae_km']:.3f}"
        )
        sweep_results[pname] = entry

    del loaded_models
    torch.cuda.empty_cache()
    return sweep_results


# ==============================================================================
# TRACK 2: DISENTANGLING ERA5 PHYSICAL VARIABLES (T vs RH) & PRESSURE SLABS
# ==============================================================================

def mask_era5_cache_channels(
    base_era5_cache: dict[str, np.ndarray],
    active_channel_indices: list[int],
) -> dict[str, np.ndarray]:
    active_set = set(active_channel_indices)
    inactive = [c for c in range(18) if c not in active_set]
    masked_cache: dict[str, np.ndarray] = {}
    for gid, arr in base_era5_cache.items():
        copy_arr = arr.copy()
        copy_arr[inactive, :, :] = 0.45
        masked_cache[gid] = copy_arr
    return masked_cache


def run_track2_parameter_and_slab_ablation(
    train_granules: list[MatchedGranule],
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    base_era5_cache: dict[str, np.ndarray],
    results: dict[str, Any],
) -> None:
    print("\n--- [Track 2] Disentangling ERA5 Physical Parameters (T vs RH) & Vertical Slabs ---")
    t2_res = results.setdefault("track2_parameter_and_slab_ablation", {})
    subsets = {
        "era5_temp_only_9levels": list(range(0, 9)),
        "era5_rh_only_9levels": list(range(9, 18)),
        "era5_low_slab_800_850_900hPa": [6, 7, 8, 15, 16, 17],
        "era5_mid_slab_500_600_700hPa": [3, 4, 5, 12, 13, 14],
        "era5_high_slab_200_300_400hPa": [0, 1, 2, 9, 10, 11],
    }

    masked_caches: dict[str, dict[str, np.ndarray]] = {}
    for sname, active_idx in subsets.items():
        mcache = mask_era5_cache_channels(base_era5_cache, active_idx)
        masked_caches[sname] = mcache
        hkey = f"{sname}_histgbdt"
        if hkey not in t2_res:
            ds_tr = make_era5_patch_dataset(
                train_granules, img_cache, track_cache, mcache,
                patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=True,
                use_era5=True, use_satellite=True, seed=42,
            )
            ds_va = make_era5_patch_dataset(
                val_granules, img_cache, track_cache, mcache,
                patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=True,
                use_era5=True, use_satellite=True, seed=42,
            )
            t2_res[hkey] = fit_and_eval_histgbdt(ds_tr, ds_va, label=sname)
            save_incremental(results)

    jobs: list[Callable[[int], None]] = []
    for sname in subsets.keys():
        mcache = masked_caches[sname]
        for arch in ["sam", "dinov3"]:
            job_key = f"{sname}_sup_{arch}"
            out_dir = f"outputs/cvpr_experiments/track2_{sname}_{arch}"
            if job_key not in t2_res:
                def _make_job(
                    _k: str = job_key,
                    _arch: str = arch,
                    _mcache: dict[str, np.ndarray] = mcache,
                    _od: str = out_dir,
                ) -> Callable[[int], None]:
                    def _run(gpu_id: int) -> None:
                        ds_tr = make_era5_patch_dataset(
                            train_granules, img_cache, track_cache, _mcache,
                            patches_per_granule=6, is_train=True, use_3frames=False, use_vis06=True,
                            use_era5=True, use_satellite=True, seed=42 + gpu_id,
                        )
                        ds_va = make_era5_patch_dataset(
                            val_granules, img_cache, track_cache, _mcache,
                            patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=True,
                            use_era5=True, use_satellite=True, seed=42,
                        )
                        res = train_and_eval_supervised_single_gpu(
                            _k,
                            lambda: build_model(arch=_arch, in_channels=22, base_ch=48, pretrained=True),
                            ds_tr,
                            ds_va,
                            _od,
                            gpu_id=gpu_id,
                            epochs=10,
                            batch_size=16,
                        )
                        t2_res[_k] = res
                        save_incremental(results)
                    return _run
                jobs.append(_make_job())

    if jobs:
        run_jobs_across_4_gpus(jobs)


# ==============================================================================
# TRACK 3: FULL 490-GRANULE ALL-DAY (DAY VS DEEP-NIGHT) 1f IR + ERA5 BENCHMARK
# ==============================================================================

def run_track3_allday_490_ir_era5(
    all_490_granules: list[MatchedGranule],
    img_cache_490: dict[str, np.ndarray],
    track_cache_490: dict[str, dict[str, np.ndarray]],
    era5_cache_490: dict[str, np.ndarray],
    results: dict[str, Any],
) -> None:
    print("\n--- [Track 3] Running Full 490-Granule All-Day 1f IR + ERA5 Benchmark ---")
    t3_res = results.setdefault("track3_allday_490_ir_era5", {})
    train_490, val_490 = split_granules_chronologically(all_490_granules, val_ratio=0.2)

    if "allday_490_1f_ir_era5_histgbdt" not in t3_res:
        ds_tr = make_era5_patch_dataset(
            train_490, img_cache_490, track_cache_490, era5_cache_490,
            patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
            use_era5=True, use_satellite=True, seed=42,
        )
        ds_va = make_era5_patch_dataset(
            val_490, img_cache_490, track_cache_490, era5_cache_490,
            patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
            use_era5=True, use_satellite=True, seed=42,
        )
        t3_res["allday_490_1f_ir_era5_histgbdt"] = fit_and_eval_histgbdt(ds_tr, ds_va, label="allday_490_1f_ir_era5")
        save_incremental(results)

    day_val = [
        g for g in val_490
        if 6 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 15
    ]
    night_val = [
        g for g in val_490
        if not (6 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 15)
    ]
    print(f"  490-Granule Val Split: Total={len(val_490)} | Day={len(day_val)} | Night/Twilight={len(night_val)}")

    # Wave 1: Supervised U-Net, SAM 2.1, DINOv3 on 490 Granules
    sup_jobs: list[Callable[[int], None]] = []
    for arch in ["unet", "sam", "dinov3"]:
        sup_key = f"allday_490_1f_ir_era5_sup_{arch}"
        sup_dir = f"outputs/cvpr_experiments/allday490_sup_{arch}_1f_ir_era5"
        if sup_key not in t3_res:
            def _make_sup(
                _k: str = sup_key,
                _arch: str = arch,
                _sd: str = sup_dir,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_era5_patch_dataset(
                        train_490, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=True, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42 + gpu_id,
                    )
                    va_ds = make_era5_patch_dataset(
                        val_490, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    res = train_and_eval_supervised_single_gpu(
                        _k,
                        lambda: build_model(arch=_arch, in_channels=21, base_ch=48, pretrained=True),
                        tr_ds,
                        va_ds,
                        _sd,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=16,
                    )
                    device = torch.device(f"cuda:{gpu_id}")
                    m = build_model(arch=_arch, in_channels=21, base_ch=48, pretrained=False)
                    m = load_ckpt_into_model(m, os.path.join(_sd, "best_model.pt"), device)
                    ds_day = make_era5_patch_dataset(
                        day_val, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    ds_night = make_era5_patch_dataset(
                        night_val, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    res["daytime_41_granules"] = evaluate_sparse_predictions(
                        m, DataLoader(ds_day, batch_size=16, shuffle=False, num_workers=0), device
                    )
                    res["nighttime_57_granules"] = evaluate_sparse_predictions(
                        m, DataLoader(ds_night, batch_size=16, shuffle=False, num_workers=0), device
                    )
                    del m
                    torch.cuda.empty_cache()
                    t3_res[_k] = res
                    save_incremental(results)
                return _run
            sup_jobs.append(_make_sup())

    if sup_jobs:
        run_jobs_across_4_gpus(sup_jobs)

    # Wave 2: Semi-Supervised U-Net, SAM 2.1, DINOv3 on 490 Granules
    semi_jobs: list[Callable[[int], None]] = []
    for arch in ["unet", "sam", "dinov3"]:
        semi_key = f"allday_490_1f_ir_era5_semi_{arch}"
        sup_dir = f"outputs/cvpr_experiments/allday490_sup_{arch}_1f_ir_era5"
        semi_dir = f"outputs/cvpr_experiments/allday490_semisup_{arch}_1f_ir_era5"
        if semi_key not in t3_res:
            def _make_semi(
                _k: str = semi_key,
                _arch: str = arch,
                _sd: str = sup_dir,
                _md: str = semi_dir,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = make_era5_patch_dataset(
                        train_490, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=True, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=142 + gpu_id,
                    )
                    va_ds = make_era5_patch_dataset(
                        val_490, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    res = train_and_eval_semisup_single_gpu(
                        _k,
                        lambda: build_model(arch=_arch, in_channels=21, base_ch=48, pretrained=False),
                        os.path.join(_sd, "best_model.pt"),
                        tr_ds,
                        va_ds,
                        _md,
                        gpu_id=gpu_id,
                        epochs=10,
                        batch_size=12,
                    )
                    device = torch.device(f"cuda:{gpu_id}")
                    m = build_model(arch=_arch, in_channels=21, base_ch=48, pretrained=False)
                    m = load_ckpt_into_model(m, os.path.join(_md, "best_semisup.pt"), device)
                    ds_day = make_era5_patch_dataset(
                        day_val, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    ds_night = make_era5_patch_dataset(
                        night_val, img_cache_490, track_cache_490, era5_cache_490,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    res["daytime_41_granules"] = evaluate_sparse_predictions(
                        m, DataLoader(ds_day, batch_size=16, shuffle=False, num_workers=0), device
                    )
                    res["nighttime_57_granules"] = evaluate_sparse_predictions(
                        m, DataLoader(ds_night, batch_size=16, shuffle=False, num_workers=0), device
                    )
                    del m
                    torch.cuda.empty_cache()
                    t3_res[_k] = res
                    save_incremental(results)
                return _run
            semi_jobs.append(_make_semi())

    if semi_jobs:
        run_jobs_across_4_gpus(semi_jobs)


# ==============================================================================
# TRACK 4: MULTI-FRAME MOTION NOWCASTING ([t=-30m, t=-15m] -> t=0m)
# ==============================================================================

class TwoFramePastForecastDataset(CloudPatchDataset):
    """
    Wraps a 3-frame CloudPatchDataset and shifts the temporal frames so the model
    observes ONLY past frames [t=-15m, t=-30m] (slot 0 <- -15m, slot 1 <- -30m, slot 2 <- -30m)
    to forecast the t=0m ICESat-2 3D cloud profile (+15 min lead time).
    """

    def __init__(self, base_ds: CloudPatchDataset, use_vis06: bool) -> None:
        self.base_ds = base_ds
        self.use_vis06 = use_vis06

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self.base_ds[idx])
        img = item["image"].clone()
        img[0:3] = img[[1, 2, 2]]
        if self.use_vis06:
            img[3:6] = img[[4, 5, 5]]
        item["image"] = img
        return item


def make_2frame_past_forecast_dataset(
    granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    patches_per_granule: int,
    is_train: bool,
    use_vis06: bool,
    use_era5: bool,
    seed: int = 42,
) -> CloudPatchDataset:
    ds = make_era5_patch_dataset(
        granules, img_cache, track_cache, era5_cache,
        patches_per_granule=patches_per_granule, is_train=is_train,
        use_3frames=True, use_vis06=use_vis06, use_era5=use_era5, use_satellite=True, seed=seed,
    )
    return TwoFramePastForecastDataset(ds, use_vis06=use_vis06)


def run_track4_multiframe_motion_nowcasting(
    train_granules: list[MatchedGranule],
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    print("\n--- [Track 4] Multi-Frame Past Motion Nowcasting ([t=-30m, t=-15m] -> t=0m, +15m Lead Time) ---")
    configs = [
        ("2f_ir_past_to_0m", False, False, 5, [
            ("sup_sam", "sam", "outputs/cvpr_experiments/sup_sam_5ch_408/best_model.pt"),
            ("semi_sam", "sam", "outputs/cvpr_experiments/semisup_sam_5ch_408/best_semisup.pt"),
            ("sup_dinov3", "dinov3", "outputs/cvpr_experiments/sup_dinov3_5ch_408/best_model.pt"),
            ("semi_dinov3", "dinov3", "outputs/cvpr_experiments/semisup_dinov3_5ch_408/best_semisup.pt"),
        ]),
        ("2f_ir_vis_past_to_0m", True, False, 8, [
            ("sup_unet", "unet", "outputs/supervised_unet_3f_vis/best_model.pt"),
            ("semi_unet", "unet", "outputs/semisup_unet_3f_vis/best_semisup.pt"),
            ("sup_sam", "sam", "outputs/supervised_sam_3f_vis/best_model.pt"),
            ("semi_sam", "sam", "outputs/semisup_sam_3f_vis/best_semisup.pt"),
            ("sup_dinov3", "dinov3", "outputs/supervised_dinov3_3f_vis/best_model.pt"),
            ("semi_dinov3", "dinov3", "outputs/semisup_dinov3_3f_vis/best_semisup.pt"),
        ]),
        ("2f_ir_era5_past_to_0m", False, True, 23, [
            ("sup_unet", "unet", "outputs/cvpr_experiments/era5_sup_unet_3f_ir_era5/best_model.pt"),
            ("semi_unet", "unet", "outputs/cvpr_experiments/era5_semisup_unet_3f_ir_era5/best_semisup.pt"),
            ("sup_sam", "sam", "outputs/cvpr_experiments/era5_sup_sam_3f_ir_era5/best_model.pt"),
            ("semi_sam", "sam", "outputs/cvpr_experiments/era5_semisup_sam_3f_ir_era5/best_semisup.pt"),
            ("sup_dinov3", "dinov3", "outputs/cvpr_experiments/era5_sup_dinov3_3f_ir_era5/best_model.pt"),
            ("semi_dinov3", "dinov3", "outputs/cvpr_experiments/era5_semisup_dinov3_3f_ir_era5/best_semisup.pt"),
        ]),
        ("2f_ir_vis_era5_past_to_0m", True, True, 26, [
            ("sup_unet", "unet", "outputs/cvpr_experiments/era5_sup_unet_3f_ir_vis_era5/best_model.pt"),
            ("semi_unet", "unet", "outputs/cvpr_experiments/era5_semisup_unet_3f_ir_vis_era5/best_semisup.pt"),
            ("sup_sam", "sam", "outputs/cvpr_experiments/era5_sup_sam_3f_ir_vis_era5/best_model.pt"),
            ("semi_sam", "sam", "outputs/cvpr_experiments/era5_semisup_sam_3f_ir_vis_era5/best_semisup.pt"),
            ("sup_dinov3", "dinov3", "outputs/cvpr_experiments/era5_sup_dinov3_3f_ir_vis_era5/best_model.pt"),
            ("semi_dinov3", "dinov3", "outputs/cvpr_experiments/era5_semisup_dinov3_3f_ir_vis_era5/best_semisup.pt"),
        ]),
    ]

    out: dict[str, Any] = {}
    for cfg_name, use_vis, use_era5, in_ch, ckpt_list in configs:
        ds_tr = make_2frame_past_forecast_dataset(
            train_granules, img_cache, track_cache, era5_cache,
            patches_per_granule=6, is_train=False, use_vis06=use_vis, use_era5=use_era5, seed=42,
        )
        ds_va = make_2frame_past_forecast_dataset(
            val_granules, img_cache, track_cache, era5_cache,
            patches_per_granule=6, is_train=False, use_vis06=use_vis, use_era5=use_era5, seed=42,
        )
        out[f"{cfg_name}_histgbdt"] = fit_and_eval_histgbdt(ds_tr, ds_va, label=cfg_name)

        loader = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=0)
        for mtag, arch, ckpt_path in ckpt_list:
            if not os.path.exists(ckpt_path):
                print(f"  [WARN] Missing checkpoint {ckpt_path}, skipping...")
                continue
            model = build_model(arch=arch, in_channels=in_ch, base_ch=48, pretrained=False)
            model = load_ckpt_into_model(model, ckpt_path, device)
            metrics = evaluate_sparse_predictions(model, loader, device)
            out[f"{cfg_name}_{mtag}"] = metrics
            print(
                f"  [{cfg_name}_{mtag:12s}] +15m Motion Forecast -> Top MAE={metrics['top_mae_km']:.3f} km | "
                f"Bot MAE={metrics['bot_mae_km']:.3f} km | F1={metrics['cloud_f1']:.4f}"
            )
            del model
            torch.cuda.empty_cache()

    return out


# ==============================================================================
# TRACK 5: SATELLITE-SEQUENCE-TO-ERA5 CONTRASTIVE ALIGNMENT (SatERA5-CLIP)
# ==============================================================================

class SatERA5ClipModel(nn.Module):
    """
    Dual-Encoder Contrastive Alignment & Cross-Modal Distillation Model (SatERA5-CLIP).
    Maps a satellite imagery sequence X_sat (C_sat, H, W) and a co-located 9-level ERA5
    thermodynamic tensor X_era5 (18, H, W) into a shared unit-hypersphere embedding space:
      - Global patch embeddings: z_sat, z_era5 in S^(D-1)  (D = 256)
      - Dense 16x16 spatial grid tokens: Z_sat, Z_era5 in R^(B, D, 16, 16)
      - Cross-modal Virtual ERA5 Decoder: predicts the 18-channel ERA5 (T, RH) vertical
        profile directly from the satellite embedding Z_sat.
    """

    def __init__(self, sat_in_channels: int = 8, embed_dim: int = 256) -> None:
        super().__init__()
        self.sat_stem = RadiometricContextStem(in_channels=sat_in_channels)
        c_stem = self.sat_stem.out_channels

        self.sat_encoder = nn.Sequential(
            nn.Conv2d(c_stem, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.sat_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.sat_grid_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.era5_encoder = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.era5_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.era5_grid_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.virtual_era5_head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 18, kernel_size=1),
            nn.Sigmoid(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

    def encode_satellite(self, x_sat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_stem = self.sat_stem(x_sat)
        grid = self.sat_encoder(feat_stem)
        mean_pool = grid.mean(dim=(2, 3))
        std_pool = grid.std(dim=(2, 3))
        z_global = F.normalize(self.sat_proj(torch.cat([mean_pool, std_pool], dim=1)), dim=-1)
        z_grid = F.normalize(self.sat_grid_proj(grid), dim=1)
        pred_era5_16x16 = self.virtual_era5_head(grid)
        return z_global, z_grid, pred_era5_16x16

    def encode_era5(self, x_era5_with_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.era5_encoder(x_era5_with_coords)
        mean_pool = grid.mean(dim=(2, 3))
        std_pool = grid.std(dim=(2, 3))
        z_global = F.normalize(self.era5_proj(torch.cat([mean_pool, std_pool], dim=1)), dim=-1)
        z_grid = F.normalize(self.era5_grid_proj(grid), dim=1)
        return z_global, z_grid


def split_combined_tensor(
    x_combined: torch.Tensor,
    use_3frames: bool,
    use_vis06: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_sat_raw = (3 if use_3frames else 1) * (2 if use_vis06 else 1)
    x_sat = x_combined[:, :n_sat_raw + 2]
    coords_2 = x_combined[:, n_sat_raw:n_sat_raw + 2]
    era5_18 = x_combined[:, n_sat_raw + 2:n_sat_raw + 20]
    x_era5_coords = torch.cat([era5_18, coords_2], dim=1)
    return x_sat, x_era5_coords, era5_18


def train_and_evaluate_satera5_clip_single_modality(
    gpu_id: int,
    modality_name: str,
    use_3frames: bool,
    use_vis06: bool,
    train_granules: list[MatchedGranule],
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    epochs: int = 20,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    set_seed(42 + gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    ds_tr = make_era5_patch_dataset(
        train_granules, img_cache, track_cache, era5_cache,
        patches_per_granule=6, is_train=True, use_3frames=use_3frames, use_vis06=use_vis06,
        use_era5=True, use_satellite=True, seed=42 + gpu_id,
    )
    ds_va = make_era5_patch_dataset(
        val_granules, img_cache, track_cache, era5_cache,
        patches_per_granule=4, is_train=False, use_3frames=use_3frames, use_vis06=use_vis06,
        use_era5=True, use_satellite=True, seed=42,
    )

    sat_in_ch = (3 if use_3frames else 1) * (2 if use_vis06 else 1) + 2
    model = SatERA5ClipModel(sat_in_channels=sat_in_ch, embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tr_loader = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=0)

    for ep in range(epochs):
        model.train()
        for batch in tr_loader:
            x_comb = batch["image"].to(device)
            x_sat, x_era5_c, era5_18 = split_combined_tensor(x_comb, use_3frames, use_vis06)
            B = x_sat.shape[0]

            z_sat, grid_sat, pred_era5_16 = model.encode_satellite(x_sat)
            z_era5, grid_era5 = model.encode_era5(x_era5_c)

            logit_scale = model.logit_scale.exp().clamp(max=100.0)
            logits_per_sat = logit_scale * (z_sat @ z_era5.t())
            targets = torch.arange(B, device=device)
            loss_global = 0.5 * (
                F.cross_entropy(logits_per_sat, targets) + F.cross_entropy(logits_per_sat.t(), targets)
            )

            gs = grid_sat.permute(2, 3, 0, 1).reshape(-1, B, 256)
            ge = grid_era5.permute(2, 3, 0, 1).reshape(-1, B, 256)
            sim_local = logit_scale * torch.bmm(gs, ge.transpose(1, 2))
            local_targets = targets.unsqueeze(0).expand(sim_local.shape[0], B)
            loss_local = 0.5 * (
                F.cross_entropy(sim_local.reshape(-1, B), local_targets.reshape(-1))
                + F.cross_entropy(sim_local.transpose(1, 2).reshape(-1, B), local_targets.reshape(-1))
            )

            era5_target_16 = F.adaptive_avg_pool2d(era5_18, (16, 16))
            loss_recon = F.l1_loss(pred_era5_16, era5_target_16) + F.mse_loss(pred_era5_16, era5_target_16)

            loss = loss_global + 0.5 * loss_local + 4.0 * loss_recon
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    ckpt_path = f"outputs/cvpr_experiments/satera5_clip_{modality_name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    all_z_sat, all_z_era5 = [], []
    matched_local_cos, mismatched_local_cos = [], []
    t_abs_err_k, t_sq_err_k = [], []
    rh_abs_err_pct, rh_sq_err_pct = [], []

    with torch.no_grad():
        for batch in va_loader:
            x_comb = batch["image"].to(device)
            x_sat, x_era5_c, era5_18 = split_combined_tensor(x_comb, use_3frames, use_vis06)
            z_sat, grid_sat, pred_era5_16 = model.encode_satellite(x_sat)
            z_era5, grid_era5 = model.encode_era5(x_era5_c)

            all_z_sat.append(z_sat.cpu())
            all_z_era5.append(z_era5.cpu())

            cos_match = (grid_sat * grid_era5).sum(dim=1).mean().item()
            cos_mismatch = (grid_sat * torch.roll(grid_era5, shifts=1, dims=0)).sum(dim=1).mean().item()
            matched_local_cos.append(cos_match)
            mismatched_local_cos.append(cos_mismatch)

            gt_16 = F.adaptive_avg_pool2d(era5_18, (16, 16))
            diff_t_k = (pred_era5_16[:, :9] - gt_16[:, :9]).abs() * 120.0
            diff_rh_pct = (pred_era5_16[:, 9:18] - gt_16[:, 9:18]).abs() * 100.0
            t_abs_err_k.append(diff_t_k.mean().item())
            t_sq_err_k.append((diff_t_k ** 2).mean().item())
            rh_abs_err_pct.append(diff_rh_pct.mean().item())
            rh_sq_err_pct.append((diff_rh_pct ** 2).mean().item())

    Z_sat = torch.cat(all_z_sat, dim=0).numpy()
    Z_era5 = torch.cat(all_z_era5, dim=0).numpy()
    sim_matrix = Z_sat @ Z_era5.T
    N = sim_matrix.shape[0]

    ranks_s2e, ranks_e2s = [], []
    for i in range(N):
        order_s2e = np.argsort(-sim_matrix[i])
        ranks_s2e.append(int(np.where(order_s2e == i)[0][0]) + 1)
        order_e2s = np.argsort(-sim_matrix[:, i])
        ranks_e2s.append(int(np.where(order_e2s == i)[0][0]) + 1)

    ranks_s2e_arr = np.array(ranks_s2e)
    ranks_e2s_arr = np.array(ranks_e2s)

    res = {
        "num_val_patches": N,
        "sat_to_era5_R1": float(np.mean(ranks_s2e_arr <= 1) * 100.0),
        "sat_to_era5_R5": float(np.mean(ranks_s2e_arr <= 5) * 100.0),
        "sat_to_era5_R10": float(np.mean(ranks_s2e_arr <= 10) * 100.0),
        "sat_to_era5_median_rank": float(np.median(ranks_s2e_arr)),
        "era5_to_sat_R1": float(np.mean(ranks_e2s_arr <= 1) * 100.0),
        "era5_to_sat_R5": float(np.mean(ranks_e2s_arr <= 5) * 100.0),
        "era5_to_sat_R10": float(np.mean(ranks_e2s_arr <= 10) * 100.0),
        "era5_to_sat_median_rank": float(np.median(ranks_e2s_arr)),
        "mean_matched_cosine_sim": float(np.mean(np.diag(sim_matrix))),
        "mean_unmatched_cosine_sim": float((np.sum(sim_matrix) - np.trace(sim_matrix)) / max(1, N * (N - 1))),
        "local_grid_matched_cosine_sim": float(np.mean(matched_local_cos)),
        "local_grid_mismatched_cosine_sim": float(np.mean(mismatched_local_cos)),
        "virtual_era5_temp_mae_K": float(np.mean(t_abs_err_k)),
        "virtual_era5_temp_rmse_K": float(np.sqrt(np.mean(t_sq_err_k))),
        "virtual_era5_rh_mae_pct": float(np.mean(rh_abs_err_pct)),
        "virtual_era5_rh_rmse_pct": float(np.sqrt(np.mean(rh_sq_err_pct))),
        "checkpoint": ckpt_path,
    }
    print(
        f"  [SatERA5-CLIP {modality_name:10s}] Sat->ERA5 R@1={res['sat_to_era5_R1']:.1f}% "
        f"R@5={res['sat_to_era5_R5']:.1f}% R@10={res['sat_to_era5_R10']:.1f}% (MedRank={res['sat_to_era5_median_rank']:.1f}) | "
        f"Virtual ERA5 T MAE={res['virtual_era5_temp_mae_K']:.2f} K, RH MAE={res['virtual_era5_rh_mae_pct']:.2f}%"
    )
    del model
    torch.cuda.empty_cache()
    return res


class VirtualERA5PatchDataset(CloudPatchDataset):
    """
    Wraps a satellite-only dataset and uses a frozen SatERA5-CLIP model (on CPU to avoid
    any GPU contention inside DataLoader) to synthesize the 18-channel Virtual ERA5
    profile on-the-fly from satellite imagery alone (requiring ZERO ERA5 reanalysis
    input at inference time!).
    """

    def __init__(
        self,
        base_sat_ds: CloudPatchDataset,
        clip_model_cpu: SatERA5ClipModel,
    ) -> None:
        self.base_sat_ds = base_sat_ds
        self.clip_model_cpu = clip_model_cpu

    def __len__(self) -> int:
        return len(self.base_sat_ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base_sat_ds[idx]
        x_sat = item["image"]  # (C_sat, H, W), [spec_patches..., Lat, Lon]
        H, W = x_sat.shape[1], x_sat.shape[2]
        with torch.no_grad():
            _, _, pred_era5_16 = self.clip_model_cpu.encode_satellite(x_sat.unsqueeze(0))
            virtual_era5 = F.interpolate(
                pred_era5_16, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
        item_out = dict(item)
        item_out["image"] = torch.cat([x_sat, virtual_era5], dim=0)
        return item_out


def run_track5_satera5_clip(
    train_granules: list[MatchedGranule],
    val_granules: list[MatchedGranule],
    img_cache: dict[str, np.ndarray],
    track_cache: dict[str, dict[str, np.ndarray]],
    era5_cache: dict[str, np.ndarray],
    results: dict[str, Any],
) -> None:
    print("\n--- [Track 5] Training & Evaluating SatERA5-CLIP Contrastive Alignment Models ---")
    t5_res = results.setdefault("track5_satera5_clip", {})
    clip_configs = [
        ("1f_ir", False, False),
        ("1f_ir_vis", False, True),
        ("3f_ir", True, False),
        ("3f_ir_vis", True, True),
    ]
    clip_jobs: list[Callable[[int], None]] = []
    for mod_name, use_3f, use_vis in clip_configs:
        ckey = f"satera5_clip_{mod_name}"
        if ckey not in t5_res:
            def _make_clip_job(
                _k: str = ckey,
                _mname: str = mod_name,
                _u3f: bool = use_3f,
                _uvis: bool = use_vis,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    res = train_and_evaluate_satera5_clip_single_modality(
                        gpu_id=gpu_id,
                        modality_name=_mname,
                        use_3frames=_u3f,
                        use_vis06=_uvis,
                        train_granules=train_granules,
                        val_granules=val_granules,
                        img_cache=img_cache,
                        track_cache=track_cache,
                        era5_cache=era5_cache,
                        epochs=20,
                    )
                    t5_res[_k] = res
                    save_incremental(results)
                return _run
            clip_jobs.append(_make_clip_job())

    if clip_jobs:
        run_jobs_across_4_gpus(clip_jobs)

    # Part 5B: Evaluate Zero-ERA5 Test-Time Reconstruction using Virtual ERA5 profiles
    # predicted by SatERA5-CLIP fed directly into our trained Satellite+ERA5 foundation models,
    # plus fine-tuning with Virtual ERA5!
    print("\n--- [Track 5B] Evaluating SatERA5-CLIP Virtual-ERA5 3D Cloud Reconstruction (Satellite-Only Test Time) ---")
    virt_res = t5_res.setdefault("virtual_era5_test_time_eval", {})
    virt_configs = [
        ("1f_ir_virtual_era5_dinov3", "1f_ir", False, False, 21, "dinov3",
         "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_era5/best_semisup.pt"),
        ("1f_ir_virtual_era5_sam", "1f_ir", False, False, 21, "sam",
         "outputs/cvpr_experiments/era5_semisup_sam_1f_ir_era5/best_semisup.pt"),
        ("1f_ir_vis_virtual_era5_sam", "1f_ir_vis", False, True, 22, "sam",
         "outputs/cvpr_experiments/era5_sup_sam_1f_ir_vis_era5/best_model.pt"),
        ("3f_ir_vis_virtual_era5_dinov3", "3f_ir_vis", True, True, 26, "dinov3",
         "outputs/cvpr_experiments/era5_semisup_dinov3_3f_ir_vis_era5/best_semisup.pt"),
    ]
    virt_jobs: list[Callable[[int], None]] = []
    for dkey, mname, u3f, uvis, in_ch, arch, init_ckpt in virt_configs:
        if dkey not in virt_res:
            def _make_virt_job(
                _dkey: str = dkey,
                _mname: str = mname,
                _u3f: bool = u3f,
                _uvis: bool = uvis,
                _in_ch: int = in_ch,
                _arch: str = arch,
                _init: str = init_ckpt,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    sat_in_ch = (3 if _u3f else 1) * (2 if _uvis else 1) + 2
                    clip_cpu = SatERA5ClipModel(sat_in_channels=sat_in_ch, embed_dim=256).cpu()
                    clip_ckpt = f"outputs/cvpr_experiments/satera5_clip_{_mname}.pt"
                    clip_cpu.load_state_dict(torch.load(clip_ckpt, map_location="cpu", weights_only=False))
                    clip_cpu.eval()

                    base_tr = make_era5_patch_dataset(
                        train_granules, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=True, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=False, use_satellite=True, seed=42 + gpu_id,
                    )
                    base_va = make_era5_patch_dataset(
                        val_granules, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=False, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=False, use_satellite=True, seed=42,
                    )
                    virt_tr = VirtualERA5PatchDataset(base_tr, clip_cpu)
                    virt_va = VirtualERA5PatchDataset(base_va, clip_cpu)

                    out_dir = f"outputs/cvpr_experiments/track5_{_dkey}"
                    res = train_and_eval_semisup_single_gpu(
                        _dkey,
                        lambda: build_model(arch=_arch, in_channels=_in_ch, base_ch=48, pretrained=False),
                        _init,
                        virt_tr,
                        virt_va,
                        out_dir,
                        gpu_id=gpu_id,
                        epochs=8,
                        batch_size=12,
                    )
                    virt_res[_dkey] = res
                    save_incremental(results)
                return _run
            virt_jobs.append(_make_virt_job())

    if virt_jobs:
        run_jobs_across_4_gpus(virt_jobs)


def main() -> None:
    t_start = time.time()
    set_seed(42)
    results = load_or_init_results()

    print("=== Discovering 490 All-Day and 408 3-Frame Granules ===")
    all_490_granules = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(all_490_granules, "../ir10.5", "../vis0.6", use_vis06=True)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)
    train_490, val_490 = split_granules_chronologically(all_490_granules, val_ratio=0.2)

    print("=== Preloading Satellite Imagery, Lidar Tracks, and 9-Level ERA5 Database ===")
    img_cache_490, track_cache_490 = preload_all_images_and_tracks(all_490_granules)
    era5_cache_490 = preload_era5_for_granules(all_490_granules)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # TRACK 1: Direct Physical Matching & Spatio-Temporal Perturbation
    # ------------------------------------------------------------------
    if "track1_physical_matching_and_perturbation" not in results:
        t1_408 = run_track1_physical_matching(
            train_408, val_408, img_cache_490, track_cache_490, era5_cache_490, split_label="408_split"
        )
        t1_490 = run_track1_physical_matching(
            train_490, val_490, img_cache_490, track_cache_490, era5_cache_490, split_label="490_split"
        )
        t1_pert = run_track1_spatiotemporal_perturbation_sweep(
            val_408, img_cache_490, track_cache_490, era5_cache_490, device=device
        )
        results["track1_physical_matching_and_perturbation"] = {
            "physical_matching_408_split": t1_408,
            "physical_matching_490_split": t1_490,
            "spatiotemporal_perturbation_sweep": t1_pert,
        }
        save_incremental(results)

    # ------------------------------------------------------------------
    # TRACK 2: Disentangling ERA5 Physical Variables (T vs RH) & Slabs
    # ------------------------------------------------------------------
    run_track2_parameter_and_slab_ablation(
        train_408, val_408, img_cache_490, track_cache_490, era5_cache_490, results
    )

    # ------------------------------------------------------------------
    # TRACK 3: Full 490-Granule All-Day (Day vs Deep-Night) 1f IR + ERA5
    # ------------------------------------------------------------------
    run_track3_allday_490_ir_era5(
        all_490_granules, img_cache_490, track_cache_490, era5_cache_490, results
    )

    # ------------------------------------------------------------------
    # TRACK 4: Multi-Frame Motion Nowcasting ([t=-30m, t=-15m] -> t=0m)
    # ------------------------------------------------------------------
    if "track4_multiframe_motion_nowcasting" not in results:
        results["track4_multiframe_motion_nowcasting"] = run_track4_multiframe_motion_nowcasting(
            train_408, val_408, img_cache_490, track_cache_490, era5_cache_490, device=device
        )
        save_incremental(results)

    # ------------------------------------------------------------------
    # TRACK 5: Satellite Sequence <-> ERA5 Contrastive Alignment (SatERA5-CLIP)
    # ------------------------------------------------------------------
    run_track5_satera5_clip(
        train_408, val_408, img_cache_490, track_cache_490, era5_cache_490, results
    )

    print(f"\n=== ALL 5 EXTENDED TRACKS COMPLETED CLEANLY IN {(time.time() - t_start)/60.0:.2f} MIN ===")


if __name__ == "__main__":
    main()
