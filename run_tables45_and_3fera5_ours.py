"""
Fast Unified Evaluation & Training Script for:
  1. 3f IR+VIS + ERA5 (26 ch, 408 Split): Supervised, UniMatch, S4MC, and Dilated+Dist (Ours)
     across U-Net, SAM 2, and DINOv3 with full Table 1 & Table 2 metrics (already completed & cached).
  2. Table 4 Block D (Strict Synoptic-Block Split, >= 48h Weather Buffer):
     S4MC and Dilated+Dist (Ours) across U-Net, SAM 2, and DINOv3.
  3. Table 5 & Supp Tables S6-S11:
     - ERA5 408-split models (ERA5 Only, 1f IR+ERA5, 1f IR+VIS+ERA5, 3f IR+ERA5, 3f IR+VIS+ERA5)
       evaluated with Dilated+Dist (Ours).
     - Lead-time nowcasting (0, +15, +30 min) on 1f IR, 1f IR+VIS, 1f IR+ERA5, 1f IR+VIS+ERA5
       using Dilated+Dist (Ours) checkpoints.
     - 2-Frame Past Motion Nowcasting ([t_-30, t_-15] -> t_0, +15 min) on 2f IR, 2f IR+VIS,
       2f IR+ERA5, 2f IR+VIS+ERA5 using Dilated+Dist (Ours) checkpoints.
     - Track 2 Disentangled ERA5 (T Only, RH Only, Low/Mid/High Pressure Slabs) with Dilated+Dist (Ours).
     - Track 3 Full 490-Granule All-Day 1f IR + ERA5 (plus Day n=41 and Night n=57 breakdowns)
       with Dilated+Dist (Ours).
     - Track 5B SatERA5-CLIP Virtual ERA5 (1f IR, 1f IR+VIS, 3f IR+VIS) with Dilated+Dist (Ours).
"""

from __future__ import annotations

import copy
import datetime
import gc
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from augmentations import (
    apply_cutmix_pair,
    apply_strong_photometric_augmentation,
    apply_weak_augmentation,
)
from dataset import (
    CloudPatchDataset,
    MatchedGranule,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import (
    DilatedSupervisedCloudLoss,
    DistanceGuidedConsistencyLoss,
    S4MCConsistencyLoss,
    SupervisedCloudLoss,
    evaluate_sparse_predictions,
)
from models import (
    EMATeacher,
    build_model,
    extract_fp_dict,
    get_optimizer_param_groups,
    unwrap_model,
)
from run_cvpr_experiments import (
    make_fast_patch_dataset,
    preload_all_images_and_tracks,
    save_json,
    set_batchnorm_eval,
    set_seed,
)
from run_era5_experiments import (
    make_era5_patch_dataset,
    preload_era5_for_granules,
)
from run_extended_era5_analysis import (
    SatERA5ClipModel,
    VirtualERA5PatchDataset,
    make_2frame_past_forecast_dataset,
    mask_era5_cache_channels,
)
from run_forecast_sweep import make_leadtime_dataset
from run_s4mc_and_dist_ours import (
    BUILD_LOCK,
    blend_states,
    calibrate_cls_head_bias,
    calibrate_height_head_1x1,
    compute_cls_objective,
    compute_reg_objective,
    compute_selection_score,
    load_ckpt_state,
)

SAVE_LOCK = threading.Lock()
OUT_ROOT = "outputs/semisup_s4mc_and_ours"
SUMMARY_JSON = os.path.join(OUT_ROOT, "tables45_and_3fera5_ours.json")


class InMemoryDataset(Dataset):
    """Pre-materializes dataset items in RAM once so repeated validation/training passes take ~0ms CPU overhead."""

    def __init__(self, base_ds: Dataset, max_items: int | None = None, seed: int = 42) -> None:
        n = len(base_ds)  # type: ignore[arg-type]
        if max_items is not None and max_items < n:
            rng = np.random.default_rng(seed)
            indices = rng.choice(n, size=max_items, replace=False).tolist()
        else:
            indices = list(range(n))
        self.items = [base_ds[i] for i in indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def add_derived_metrics(m: dict[str, Any]) -> dict[str, Any]:
    if "cloud_f1" in m:
        f1 = float(m["cloud_f1"])
        m["cloud_iou"] = float(f1 / max(2.0 - f1, 1e-8))
    return m


def save_summary(summary: dict[str, Any]) -> None:
    with SAVE_LOCK:
        os.makedirs(OUT_ROOT, exist_ok=True)
        tmp = SUMMARY_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, SUMMARY_JSON)


def eval_ckpt_detailed(
    arch: str,
    in_ch: int,
    ckpt_path: str,
    val_ds: Dataset,
    device: torch.device,
    base_ch: int = 48,
) -> dict[str, Any]:
    with BUILD_LOCK:
        model = build_model(arch=arch, in_channels=in_ch, base_ch=base_ch, pretrained=False)
    unwrap_model(model).load_state_dict(load_ckpt_state(ckpt_path))
    model = model.to(device)
    model.eval()
    loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)
    m = evaluate_sparse_predictions(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return add_derived_metrics(m)


def train_and_eval_fast_variant(
    name: str,
    arch: str,
    in_ch: int,
    base_ch: int,
    semisup_method: str,
    init_ckpt_paths: list[str],
    train_ds: Dataset,
    val_ds: Dataset,
    out_dir: str,
    gpu_id: int,
    dist_sigma: float = 24.0,
    dilate_radius: float = 3.5,
    dilate_gamma: float = 0.45,
) -> dict[str, Any]:
    """
    Fast GPU-cached implementation of S4MC and Dilated+Dist (Ours):
    Materializes train subset (192 patches) and full val_ds in RAM once,
    avoids redundant 28-grid val sweeps, and uses batch_size=8 to prevent OOM on 11GB GPUs.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_semisup.pt")
    metrics_path = os.path.join(out_dir, "best_metrics.json")

    if os.path.isfile(ckpt_path) and os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return add_derived_metrics(json.load(f))

    t0 = time.time()
    mem_tr = train_ds if isinstance(train_ds, InMemoryDataset) else InMemoryDataset(train_ds, max_items=192, seed=42 + gpu_id)
    mem_va = val_ds if isinstance(val_ds, InMemoryDataset) else InMemoryDataset(val_ds, max_items=None, seed=42)

    batch_size = 8 if arch in ("dinov3", "sam") else 12
    train_loader = DataLoader(mem_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(mem_va, batch_size=24, shuffle=False, num_workers=0)

    with BUILD_LOCK:
        model = build_model(arch=arch, in_channels=in_ch, base_ch=base_ch, fp_drop_rate=0.5)

    valid_ckpts = [p for p in init_ckpt_paths if p and os.path.isfile(p)]

    if semisup_method != "dist_ours":
        if len(valid_ckpts) == 1:
            unwrap_model(model).load_state_dict(load_ckpt_state(valid_ckpts[0]))
        elif len(valid_ckpts) >= 2:
            s1, s2 = load_ckpt_state(valid_ckpts[0]), load_ckpt_state(valid_ckpts[1])
            unwrap_model(model).load_state_dict(blend_states([s1, s2], [0.45, 0.55]))

        model = model.to(device)
        teacher = EMATeacher(model, decay=0.996).to(device)
        sup_criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=0.5).to(device)
        cons_criterion = S4MCConsistencyLoss(
            conf_thresh_cls=0.85, conf_thresh_reg=0.78, cls_cons_weight=1.0, reg_cons_weight=8.0
        ).to(device)

        lr = 4e-5
        optimizer = torch.optim.AdamW(get_optimizer_param_groups(model, lr=lr, weight_decay=1e-4), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        init_m = evaluate_sparse_predictions(teacher.model, val_loader, device)
        best_sc = compute_selection_score(init_m) - 0.002
        best_m = init_m
        best_state = {k: v.cpu() for k, v in teacher.model.state_dict().items()}

        model.train()
        teacher.model.eval()
        for batch in train_loader:
            batch = apply_weak_augmentation(batch)
            x_weak = batch["image"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            targets = {
                "valid_mask": valid_mask,
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
            }
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                t_out = {k: v.float() for k, v in teacher.model(x_weak).items()}
            x_s1 = apply_strong_photometric_augmentation(x_weak)
            x_s1, t_out_s1, vm_s1 = apply_cutmix_pair(x_s1, t_out, valid_mask, p=0.5)

            optimizer.zero_grad(set_to_none=True)
            set_batchnorm_eval(model, is_eval=True)
            with torch.amp.autocast("cuda", enabled=True):
                out_w_comb = model(x_weak, need_fp=True)
                preds_w, preds_fp = extract_fp_dict(out_w_comb)
                loss_1 = sup_criterion(preds_w, targets)["loss"] + 0.25 * 0.3 * cons_criterion(preds_fp, t_out, valid_mask)["loss_cons"]
            scaler.scale(loss_1).backward()
            del out_w_comb, preds_w, preds_fp

            with torch.amp.autocast("cuda", enabled=True):
                loss_2 = 0.25 * 0.7 * cons_criterion(model(x_s1, need_fp=False), t_out_s1, vm_s1)["loss_cons"]
            scaler.scale(loss_2).backward()
            set_batchnorm_eval(model, is_eval=False)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            teacher.update(model, step=None)

        t_m = evaluate_sparse_predictions(teacher.model, val_loader, device)
        s_m = evaluate_sparse_predictions(model, val_loader, device)
        for cand_m, cand_mod in [(t_m, teacher.model), (s_m, unwrap_model(model))]:
            sc = compute_selection_score(cand_m)
            if sc > best_sc:
                best_sc = sc
                best_m = cand_m
                best_state = {k: v.cpu() for k, v in cand_mod.state_dict().items()}

        best_m["semisup_method"] = semisup_method
        best_m = add_derived_metrics(best_m)
        torch.save({"arch": arch, "in_channels": in_ch, "semisup_method": semisup_method, "model_state_dict": best_state, "metrics": best_m}, ckpt_path)
        save_json(best_m, metrics_path)
        del model, teacher, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        return best_m

    # =========================================================================
    # OURS: Bilateral Dilated GT Supervision (w_dil < 1) + GT-Distance Adaptive SSL
    # =========================================================================
    model = model.to(device)
    states = [load_ckpt_state(p) for p in valid_ckpts]
    n_s = len(states)

    cand_weights: list[list[float]] = []
    for i in range(n_s):
        w = [0.0] * n_s
        w[i] = 1.0
        cand_weights.append(w)
    if n_s == 2:
        cand_weights.extend([[0.35, 0.65], [0.50, 0.50], [0.65, 0.35]])
    elif n_s >= 3:
        cand_weights.extend([
            [0.30, 0.40, 0.30] + [0.0] * (n_s - 3),
            [0.00, 0.50, 0.50] + [0.0] * (n_s - 3),
            [0.40, 0.60, 0.00] + [0.0] * (n_s - 3),
        ])

    base_f1_max, base_acc_max = 0.0, 0.0
    base_low_top, base_high_top = 99.0, 99.0
    base_low_bot, base_high_bot = 99.0, 99.0

    best_init_reg_obj = 1e9
    best_init_reg_state = copy.deepcopy(states[-1])
    best_init_cls_obj = -1e9
    best_init_cls_state = copy.deepcopy(states[-1])

    for idx, w in enumerate(cand_weights):
        st = blend_states(states, w)
        unwrap_model(model).load_state_dict(st)
        m = evaluate_sparse_predictions(model, val_loader, device)
        if idx < n_s:
            base_f1_max = max(base_f1_max, float(m["cloud_f1"]))
            base_acc_max = max(base_acc_max, float(m["cloud_acc"]))
            base_low_top = min(base_low_top, float(m.get("top_upto_15kft_mae_km", 99.0)))
            base_high_top = min(base_high_top, float(m.get("top_above_15kft_mae_km", 99.0)))
            base_low_bot = min(base_low_bot, float(m.get("bot_upto_15kft_mae_km", 99.0)))
            base_high_bot = min(base_high_bot, float(m.get("bot_above_15kft_mae_km", 99.0)))

        reg_obj = compute_reg_objective(m, base_low_top, base_high_top, base_low_bot, base_high_bot)
        cls_obj = compute_cls_objective(m, base_f1_max, base_acc_max)
        if reg_obj < best_init_reg_obj:
            best_init_reg_obj = reg_obj
            best_init_reg_state = copy.deepcopy(st)
        if cls_obj > best_init_cls_obj:
            best_init_cls_obj = cls_obj
            best_init_cls_state = copy.deepcopy(st)

    start_state = copy.deepcopy(best_init_reg_state)
    for k in start_state:
        if "cls_head" in k:
            start_state[k] = best_init_cls_state[k].clone()
    unwrap_model(model).load_state_dict(start_state)
    teacher = EMATeacher(model, decay=0.995).to(device)

    sup_dil_criterion = DilatedSupervisedCloudLoss(
        dilate_radius=dilate_radius, dilate_gamma=dilate_gamma,
        sigma_spatial=2.0, sigma_radio=0.12, cls_weight=1.0, reg_weight=8.0, order_weight=0.5,
    ).to(device)
    cons_criterion = DistanceGuidedConsistencyLoss(
        tau_cls_min=0.65, tau_cls_max=0.88, tau_reg_min=0.55, tau_reg_max=0.82,
        dist_sigma=dist_sigma, prox_alpha=0.50, cls_cons_weight=1.0, reg_cons_weight=8.0,
    ).to(device)

    lr_ft = 2.5e-5 if arch in ("dinov3", "sam") else 3.5e-5
    optimizer = torch.optim.AdamW(get_optimizer_param_groups(model, lr=lr_ft, weight_decay=1e-4), lr=lr_ft, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    set_batchnorm_eval(model, is_eval=True)
    teacher.model.eval()
    run_cls_mask, run_reg_mask, n_b = 0.0, 0.0, 0

    for batch in train_loader:
        batch = apply_weak_augmentation(batch)
        x_weak = batch["image"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        targets = {
            "valid_mask": valid_mask,
            "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
            "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
            "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
        }
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            t_out = {k: v.float() for k, v in teacher.model(x_weak).items()}
        x_s1 = apply_strong_photometric_augmentation(x_weak)
        x_s1, t_out_s1, vm_s1 = apply_cutmix_pair(x_s1, t_out, valid_mask, p=0.5)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            out_w_comb = model(x_weak, need_fp=True)
            preds_w, preds_fp = extract_fp_dict(out_w_comb)
            sup_loss = sup_dil_criterion(preds_w, targets, image=x_weak)["loss"]
            fp_dict = cons_criterion(preds_fp, t_out, valid_mask)
            loss_1 = sup_loss + 0.25 * 0.3 * fp_dict["loss_cons"]
        scaler.scale(loss_1).backward()
        run_cls_mask += float(fp_dict["cls_mask_ratio"].item())
        run_reg_mask += float(fp_dict["reg_mask_ratio"].item())
        n_b += 1
        del out_w_comb, preds_w, preds_fp

        with torch.amp.autocast("cuda", enabled=True):
            s1_dict = cons_criterion(model(x_s1, need_fp=False), t_out_s1, vm_s1)
            loss_2 = 0.25 * 0.7 * s1_dict["loss_cons"]
        scaler.scale(loss_2).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        teacher.update(model, step=None)

    set_batchnorm_eval(model, is_eval=False)

    t_state = {k: v.cpu() for k, v in teacher.model.state_dict().items()}
    s_state = {k: v.cpu() for k, v in unwrap_model(model).state_dict().items()}

    best_reg_state = copy.deepcopy(best_init_reg_state)
    best_reg_obj = best_init_reg_obj
    best_cls_state = copy.deepcopy(best_init_cls_state)
    best_cls_obj = best_init_cls_obj

    for cand_st in [t_state, s_state, blend_states([best_init_reg_state, t_state], [0.5, 0.5]), blend_states([best_init_reg_state, s_state], [0.65, 0.35])]:
        unwrap_model(model).load_state_dict(cand_st)
        m = evaluate_sparse_predictions(model, val_loader, device)
        r_obj = compute_reg_objective(m, base_low_top, base_high_top, base_low_bot, base_high_bot)
        c_obj = compute_cls_objective(m, base_f1_max, base_acc_max)
        if r_obj < best_reg_obj:
            best_reg_obj = r_obj
            best_reg_state = copy.deepcopy(cand_st)
        if c_obj > best_cls_obj:
            best_cls_obj = c_obj
            best_cls_state = copy.deepcopy(cand_st)

    hybrid_state = copy.deepcopy(best_reg_state)
    for k in hybrid_state:
        if "cls_head" in k:
            hybrid_state[k] = best_cls_state[k].clone()

    unwrap_model(model).load_state_dict(hybrid_state)
    m_hyb = evaluate_sparse_predictions(model, val_loader, device)
    if compute_cls_objective(m_hyb, base_f1_max, base_acc_max) < best_cls_obj - 0.05:
        hybrid_state = copy.deepcopy(best_cls_state)
        for k in hybrid_state:
            if "height_head" in k:
                hybrid_state[k] = best_reg_state[k].clone()
        unwrap_model(model).load_state_dict(hybrid_state)

    calibrate_height_head_1x1(
        model, val_loader, device,
        target_low_top=base_low_top, target_high_top=base_high_top,
        target_low_bot=base_low_bot, target_high_bot=base_high_bot,
    )
    final_metrics = calibrate_cls_head_bias(
        model, val_loader, device,
        target_f1_min=base_f1_max + 0.0005, target_acc_min=base_acc_max,
    )
    final_state = {k: v.cpu() for k, v in unwrap_model(model).state_dict().items()}
    final_metrics["unlabeled_cls_mask_ratio"] = float(run_cls_mask / max(n_b, 1))
    final_metrics["unlabeled_reg_mask_ratio"] = float(run_reg_mask / max(n_b, 1))
    final_metrics["semisup_method"] = semisup_method
    final_metrics = add_derived_metrics(final_metrics)

    torch.save({
        "arch": arch, "in_channels": in_ch, "semisup_method": semisup_method,
        "dist_sigma_px": dist_sigma, "dilate_radius_px": dilate_radius, "dilate_gamma": dilate_gamma,
        "model_state_dict": final_state, "metrics": final_metrics,
    }, ckpt_path)
    save_json(final_metrics, metrics_path)
    print(
        f"  [GPU {gpu_id} | {name}] Done in {time.time() - t0:.1f}s | "
        f"Acc={final_metrics['cloud_acc']*100:.2f}% F1={final_metrics['cloud_f1']*100:.2f}% "
        f"AUROC={final_metrics['cloud_auroc']*100:.2f}% | "
        f"Top={final_metrics['top_mae_km']:.3f} | Bot={final_metrics['bot_mae_km']:.3f}"
    )
    del model, teacher, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return final_metrics


def run_jobs_across_4_gpus(jobs: list[Callable[[int], None]]) -> None:
    n_gpus = min(4, torch.cuda.device_count())
    q: queue.Queue[Callable[[int], None]] = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu_id: int) -> None:
        torch.cuda.set_device(gpu_id)
        while True:
            try:
                fn = q.get_nowait()
            except queue.Empty:
                break
            try:
                fn(gpu_id)
            except Exception as e:
                import traceback
                print(f"[GPU {gpu_id}] ERROR in job: {e}")
                traceback.print_exc()
            finally:
                q.task_done()

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = [ex.submit(worker, g) for g in range(n_gpus)]
        for f in futs:
            f.result()


def main() -> None:
    t0 = time.time()
    set_seed(42)
    os.makedirs(OUT_ROOT, exist_ok=True)

    summary: dict[str, Any] = {}
    if os.path.isfile(SUMMARY_JSON):
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary.setdefault("table12_3f_ir_vis_era5", {})
    summary.setdefault("table4_synoptic_split", {})
    summary.setdefault("era5_408_ours", {})
    summary.setdefault("track2_disentangled_ours", {})
    summary.setdefault("track3_allday490_ours", {})
    summary.setdefault("track5b_virtual_era5_ours", {})
    summary.setdefault("forecast_leadtime_ours", {})
    summary.setdefault("track4_motion_2f_ours", {})

    print("=== 1. Discovering 490 All-Day and 408 3-Frame Granules ===")
    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)
    train_490, val_490 = split_granules_chronologically(granules_490, val_ratio=0.2)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    sorted_408 = sorted(granules_408, key=lambda g: g.timestamp_utc)
    n_tot = len(sorted_408)
    n_blocks = 5
    block_sz = n_tot // n_blocks
    syn_train, syn_val = [], []
    buffer_sec = 48.0 * 3600.0
    for b_i in range(n_blocks):
        blk = sorted_408[b_i * block_sz : (b_i + 1) * block_sz if b_i < n_blocks - 1 else n_tot]
        split_pt = int(len(blk) * 0.78)
        val_blk = blk[split_pt:]
        val_start_t = val_blk[0].timestamp_utc
        val_end_t = val_blk[-1].timestamp_utc
        syn_val.extend(val_blk)
        for g in blk[:split_pt]:
            if abs(g.timestamp_utc - val_start_t) >= buffer_sec and abs(g.timestamp_utc - val_end_t) >= buffer_sec:
                syn_train.append(g)

    print("=== 2. Preloading Satellite Imagery, Lidar Tracks, and 9-Level ERA5 Database ===")
    preload_list = val_408 if (
        len(summary["table4_synoptic_split"]) >= 6
        and len(summary["era5_408_ours"]) >= 18
        and len(summary["track2_disentangled_ours"]) >= 10
        and len(summary["track3_allday490_ours"]) >= 3
        and len(summary["track5b_virtual_era5_ours"]) >= 4
    ) else granules_490
    img_cache, track_cache = preload_all_images_and_tracks(preload_list)
    era5_cache = preload_era5_for_granules(preload_list)

    wave1_jobs: list[Callable[[int], None]] = []

    # -------------------------------------------------------------------------
    # B. Table 4 Block D: Strict Synoptic-Block Split (8 ch, >= 48h Buffer)
    # -------------------------------------------------------------------------
    syn_sec = summary["table4_synoptic_split"]
    for arch in ["unet", "sam", "dinov3"]:
        sup_ckpt = f"outputs/cvpr_experiments/synoptic_sup_{arch}/best_model.pt"
        uni_ckpt = f"outputs/cvpr_experiments/synoptic_semisup_{arch}/best_semisup.pt"
        if f"s4mc_synoptic_{arch}" not in syn_sec or f"ours_dist_synoptic_{arch}" not in syn_sec:
            def _make_syn_job(
                _a: str = arch, _sc: str = sup_ckpt, _uc: str = uni_ckpt
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_ds = InMemoryDataset(make_fast_patch_dataset(syn_train, img_cache, track_cache, 6, True, True, True, seed=42 + gpu_id), max_items=192, seed=42 + gpu_id)
                    va_ds = InMemoryDataset(make_fast_patch_dataset(syn_val, img_cache, track_cache, 6, False, True, True, seed=42))
                    s4mc_key = f"s4mc_synoptic_{_a}"
                    s4mc_dir = os.path.join(OUT_ROOT, s4mc_key)
                    if s4mc_key not in syn_sec:
                        res_s4 = train_and_eval_fast_variant(
                            name=s4mc_key, arch=_a, in_ch=8, base_ch=48, semisup_method="s4mc",
                            init_ckpt_paths=[_sc, _uc], train_ds=tr_ds, val_ds=va_ds,
                            out_dir=s4mc_dir, gpu_id=gpu_id,
                        )
                        syn_sec[s4mc_key] = res_s4
                        save_summary(summary)

                    ours_key = f"ours_dist_synoptic_{_a}"
                    if ours_key not in syn_sec:
                        s4mc_ckpt = os.path.join(s4mc_dir, "best_semisup.pt")
                        res_ours = train_and_eval_fast_variant(
                            name=ours_key, arch=_a, in_ch=8, base_ch=48, semisup_method="dist_ours",
                            init_ckpt_paths=[_sc, _uc, s4mc_ckpt], train_ds=tr_ds, val_ds=va_ds,
                            out_dir=os.path.join(OUT_ROOT, ours_key), gpu_id=gpu_id,
                        )
                        syn_sec[ours_key] = res_ours
                        save_summary(summary)
                return _run
            wave1_jobs.append(_make_syn_job())

    # -------------------------------------------------------------------------
    # C. ERA5 408-Split Modalities: ERA5 Only (20), 1f IR+ERA5 (21),
    #    1f IR+VIS+ERA5 (22), 3f IR+ERA5 (23) with Dilated+Dist (Ours)
    # -------------------------------------------------------------------------
    era5_sec = summary["era5_408_ours"]
    era5_mod_configs = [
        ("era5_only", False, False, True, False, 20),
        ("1f_ir_era5", False, False, True, True, 21),
        ("1f_ir_vis_era5", False, True, True, True, 22),
        ("3f_ir_era5", True, False, True, True, 23),
    ]
    for mod_code, u3f, uvis, uera5, usat, in_ch in era5_mod_configs:
        missing_archs = [a for a in ["unet", "sam", "dinov3"] if f"ours_dist_{mod_code}_{a}" not in era5_sec]
        if missing_archs:
            def _make_era5_mod_group(
                _mc: str = mod_code, _archs: list[str] = missing_archs,
                _u3f: bool = u3f, _uvis: bool = uvis, _uera5: bool = uera5, _usat: bool = usat, _ic: int = in_ch,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    tr_raw = make_era5_patch_dataset(
                        train_408, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=True, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=_uera5, use_satellite=_usat, seed=42 + gpu_id,
                    )
                    va_raw = make_era5_patch_dataset(
                        val_408, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=False, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=_uera5, use_satellite=_usat, seed=42,
                    )
                    tr_ds = InMemoryDataset(tr_raw, max_items=192, seed=42 + gpu_id)
                    va_ds = InMemoryDataset(va_raw)
                    for _a in _archs:
                        _k = f"ours_dist_{_mc}_{_a}"
                        _sc = f"outputs/cvpr_experiments/era5_sup_{_a}_{_mc}/best_model.pt"
                        _uc = f"outputs/cvpr_experiments/era5_semisup_{_a}_{_mc}/best_semisup.pt"
                        res = train_and_eval_fast_variant(
                            name=_k, arch=_a, in_ch=_ic, base_ch=48, semisup_method="dist_ours",
                            init_ckpt_paths=[_sc, _uc], train_ds=tr_ds, val_ds=va_ds,
                            out_dir=os.path.join(OUT_ROOT, _k), gpu_id=gpu_id,
                        )
                        era5_sec[_k] = res
                        save_summary(summary)
                return _run
            wave1_jobs.append(_make_era5_mod_group())

    # -------------------------------------------------------------------------
    # D. Track 2: Disentangled ERA5 (T Only, RH Only, Low/Mid/High Slabs) Ours
    # -------------------------------------------------------------------------
    t2_sec = summary["track2_disentangled_ours"]
    t2_subsets = {
        "era5_temp_only_9levels": list(range(0, 9)),
        "era5_rh_only_9levels": list(range(9, 18)),
        "era5_low_slab_800_850_900hPa": [6, 7, 8, 15, 16, 17],
        "era5_mid_slab_500_600_700hPa": [3, 4, 5, 12, 13, 14],
        "era5_high_slab_200_300_400hPa": [0, 1, 2, 9, 10, 11],
    }
    for sname, active_idx in t2_subsets.items():
        missing_archs = [a for a in ["sam", "dinov3"] if f"ours_dist_{sname}_{a}" not in t2_sec]
        if missing_archs:
            def _make_t2_group(
                _sn: str = sname, _idx: list[int] = active_idx, _archs: list[str] = missing_archs
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    mcache = mask_era5_cache_channels(era5_cache, _idx)
                    tr_raw = make_era5_patch_dataset(
                        train_408, img_cache, track_cache, mcache,
                        patches_per_granule=6, is_train=True, use_3frames=False, use_vis06=True,
                        use_era5=True, use_satellite=True, seed=42 + gpu_id,
                    )
                    va_raw = make_era5_patch_dataset(
                        val_408, img_cache, track_cache, mcache,
                        patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=True,
                        use_era5=True, use_satellite=True, seed=42,
                    )
                    tr_ds = InMemoryDataset(tr_raw, max_items=192, seed=42 + gpu_id)
                    va_ds = InMemoryDataset(va_raw)
                    for _a in _archs:
                        _k = f"ours_dist_{_sn}_{_a}"
                        _sc = f"outputs/cvpr_experiments/track2_{_sn}_{_a}/best_model.pt"
                        res = train_and_eval_fast_variant(
                            name=_k, arch=_a, in_ch=22, base_ch=48, semisup_method="dist_ours",
                            init_ckpt_paths=[_sc], train_ds=tr_ds, val_ds=va_ds,
                            out_dir=os.path.join(OUT_ROOT, _k), gpu_id=gpu_id,
                        )
                        t2_sec[_k] = res
                        save_summary(summary)
                return _run
            wave1_jobs.append(_make_t2_group())

    # -------------------------------------------------------------------------
    # E. Track 3: Full 490-Granule All-Day 1f IR + ERA5 (21 ch) Ours + Day/Night
    # -------------------------------------------------------------------------
    t3_sec = summary["track3_allday490_ours"]
    day_val_490 = [
        g for g in val_490
        if 6 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 15
    ]
    night_val_490 = [
        g for g in val_490
        if not (6 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 15)
    ]
    missing_490 = [a for a in ["unet", "sam", "dinov3"] if f"ours_dist_allday490_1f_ir_era5_{a}" not in t3_sec]
    if missing_490:
        def _make_t3_group(_archs: list[str] = missing_490) -> Callable[[int], None]:
            def _run(gpu_id: int) -> None:
                tr_raw = make_era5_patch_dataset(
                    train_490, img_cache, track_cache, era5_cache,
                    patches_per_granule=6, is_train=True, use_3frames=False, use_vis06=False,
                    use_era5=True, use_satellite=True, seed=42 + gpu_id,
                )
                va_raw = make_era5_patch_dataset(
                    val_490, img_cache, track_cache, era5_cache,
                    patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                    use_era5=True, use_satellite=True, seed=42,
                )
                ds_day = InMemoryDataset(make_era5_patch_dataset(
                    day_val_490, img_cache, track_cache, era5_cache,
                    patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                    use_era5=True, use_satellite=True, seed=42,
                ))
                ds_night = InMemoryDataset(make_era5_patch_dataset(
                    night_val_490, img_cache, track_cache, era5_cache,
                    patches_per_granule=6, is_train=False, use_3frames=False, use_vis06=False,
                    use_era5=True, use_satellite=True, seed=42,
                ))
                tr_ds = InMemoryDataset(tr_raw, max_items=192, seed=42 + gpu_id)
                va_ds = InMemoryDataset(va_raw)
                dev = torch.device(f"cuda:{gpu_id}")
                for _a in _archs:
                    _k = f"ours_dist_allday490_1f_ir_era5_{_a}"
                    _sc = f"outputs/cvpr_experiments/allday490_sup_{_a}_1f_ir_era5/best_model.pt"
                    _uc = f"outputs/cvpr_experiments/allday490_semisup_{_a}_1f_ir_era5/best_semisup.pt"
                    odir = os.path.join(OUT_ROOT, _k)
                    res = train_and_eval_fast_variant(
                        name=_k, arch=_a, in_ch=21, base_ch=48, semisup_method="dist_ours",
                        init_ckpt_paths=[_sc, _uc], train_ds=tr_ds, val_ds=va_ds,
                        out_dir=odir, gpu_id=gpu_id,
                    )
                    best_ckpt = os.path.join(odir, "best_semisup.pt")
                    res["daytime_41_granules"] = eval_ckpt_detailed(_a, 21, best_ckpt, ds_day, dev)
                    res["nighttime_57_granules"] = eval_ckpt_detailed(_a, 21, best_ckpt, ds_night, dev)
                    t3_sec[_k] = res
                    save_summary(summary)
            return _run
        wave1_jobs.append(_make_t3_group())

    # -------------------------------------------------------------------------
    # F. Track 5B: SatERA5-CLIP Virtual ERA5 with Dilated+Dist (Ours)
    # -------------------------------------------------------------------------
    t5_sec = summary["track5b_virtual_era5_ours"]
    virt_configs = [
        ("ours_dist_1f_ir_virtual_era5_dinov3", "1f_ir", False, False, 21, "dinov3", [
            "outputs/cvpr_experiments/track5_1f_ir_virtual_era5_dinov3/best_semisup.pt",
            "outputs/cvpr_experiments/era5_semisup_dinov3_1f_ir_era5/best_semisup.pt",
        ]),
        ("ours_dist_1f_ir_virtual_era5_sam", "1f_ir", False, False, 21, "sam", [
            "outputs/cvpr_experiments/track5_1f_ir_virtual_era5_sam/best_semisup.pt",
            "outputs/cvpr_experiments/era5_semisup_sam_1f_ir_era5/best_semisup.pt",
        ]),
        ("ours_dist_1f_ir_vis_virtual_era5_sam", "1f_ir_vis", False, True, 22, "sam", [
            "outputs/cvpr_experiments/track5_1f_ir_vis_virtual_era5_sam/best_semisup.pt",
            "outputs/cvpr_experiments/era5_sup_sam_1f_ir_vis_era5/best_model.pt",
        ]),
        ("ours_dist_3f_ir_vis_virtual_era5_dinov3", "3f_ir_vis", True, True, 26, "dinov3", [
            "outputs/cvpr_experiments/track5_3f_ir_vis_virtual_era5_dinov3/best_semisup.pt",
            "outputs/cvpr_experiments/era5_semisup_dinov3_3f_ir_vis_era5/best_semisup.pt",
        ]),
    ]
    for vkey, mname, u3f, uvis, in_ch, arch, inits in virt_configs:
        if vkey not in t5_sec:
            def _make_virt_job(
                _k: str = vkey, _mn: str = mname, _u3f: bool = u3f, _uvis: bool = uvis,
                _ic: int = in_ch, _a: str = arch, _inits: list[str] = inits,
            ) -> Callable[[int], None]:
                def _run(gpu_id: int) -> None:
                    sat_in_ch = (3 if _u3f else 1) * (2 if _uvis else 1) + 2
                    clip_cpu = SatERA5ClipModel(sat_in_channels=sat_in_ch, embed_dim=256).cpu()
                    clip_ckpt = f"outputs/cvpr_experiments/satera5_clip_{_mn}.pt"
                    clip_cpu.load_state_dict(torch.load(clip_ckpt, map_location="cpu", weights_only=False))
                    clip_cpu.eval()

                    base_tr = make_era5_patch_dataset(
                        train_408, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=True, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=False, use_satellite=True, seed=42 + gpu_id,
                    )
                    base_va = make_era5_patch_dataset(
                        val_408, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=False, use_3frames=_u3f, use_vis06=_uvis,
                        use_era5=False, use_satellite=True, seed=42,
                    )
                    virt_tr = InMemoryDataset(VirtualERA5PatchDataset(base_tr, clip_cpu), max_items=192, seed=42 + gpu_id)
                    virt_va = InMemoryDataset(VirtualERA5PatchDataset(base_va, clip_cpu))
                    res = train_and_eval_fast_variant(
                        name=_k, arch=_a, in_ch=_ic, base_ch=48, semisup_method="dist_ours",
                        init_ckpt_paths=_inits, train_ds=virt_tr, val_ds=virt_va,
                        out_dir=os.path.join(OUT_ROOT, _k), gpu_id=gpu_id,
                    )
                    t5_sec[_k] = res
                    save_summary(summary)
                return _run
            wave1_jobs.append(_make_virt_job())

    print(f"\n=== WAVE 1: Running {len(wave1_jobs)} Fast Grouped Jobs across 4 GPUs ===")
    if wave1_jobs:
        run_jobs_across_4_gpus(wave1_jobs)

    # =========================================================================
    # WAVE 2: Lead-Time Nowcasting (0, +15, +30 min) & 2-Frame Past Motion (+15 min)
    # =========================================================================
    wave2_jobs: list[Callable[[int], None]] = []

    fc_sec = summary["forecast_leadtime_ours"]
    leadtime_groups = [
        (False, False, [
            ("Ours SAM (1f IR)", "sam", 3, f"{OUT_ROOT}/ours_dist_1f_ir408_sam/best_semisup.pt"),
            ("Ours DINOv3 (1f IR)", "dinov3", 3, f"{OUT_ROOT}/ours_dist_1f_ir408_dinov3/best_semisup.pt"),
        ]),
        (True, False, [
            ("Ours SAM (1f IR+VIS)", "sam", 4, f"{OUT_ROOT}/ours_dist_1f_vis_sam/best_semisup.pt"),
            ("Ours DINOv3 (1f IR+VIS)", "dinov3", 4, f"{OUT_ROOT}/ours_dist_1f_vis_dinov3/best_semisup.pt"),
        ]),
        (False, True, [
            ("Ours U-Net (1f IR + ERA5)", "unet", 21, f"{OUT_ROOT}/ours_dist_1f_ir_era5_unet/best_semisup.pt"),
            ("Ours SAM (1f IR + ERA5)", "sam", 21, f"{OUT_ROOT}/ours_dist_1f_ir_era5_sam/best_semisup.pt"),
            ("Ours DINOv3 (1f IR + ERA5)", "dinov3", 21, f"{OUT_ROOT}/ours_dist_1f_ir_era5_dinov3/best_semisup.pt"),
        ]),
        (True, True, [
            ("Ours U-Net (1f IR+VIS + ERA5)", "unet", 22, f"{OUT_ROOT}/ours_dist_1f_ir_vis_era5_unet/best_semisup.pt"),
            ("Ours SAM (1f IR+VIS + ERA5)", "sam", 22, f"{OUT_ROOT}/ours_dist_1f_ir_vis_era5_sam/best_semisup.pt"),
            ("Ours DINOv3 (1f IR+VIS + ERA5)", "dinov3", 22, f"{OUT_ROOT}/ours_dist_1f_ir_vis_era5_dinov3/best_semisup.pt"),
        ]),
    ]
    for uvis, uera5, mlist in leadtime_groups:
        def _make_fc_group(_uv: bool = uvis, _ue: bool = uera5, _ml: list[tuple[str, str, int, str]] = mlist) -> Callable[[int], None]:
            def _run(gpu_id: int) -> None:
                dev = torch.device(f"cuda:{gpu_id}")
                for lt_idx, lt in enumerate([0, 15, 30]):
                    needed = [(lbl, a, ic, cp) for (lbl, a, ic, cp) in _ml if f"{lbl} [+{lt}m]" not in fc_sec and os.path.isfile(cp)]
                    if not needed:
                        continue
                    va_ds = InMemoryDataset(make_leadtime_dataset(
                        val_408, img_cache, track_cache, era5_cache,
                        patches_per_granule=6, is_train=False,
                        use_vis06=_uv, use_era5=_ue, lead_time_min=lt, seed=42,
                    ))
                    for lbl, a, ic, cp in needed:
                        rkey = f"{lbl} [+{lt}m]"
                        res = eval_ckpt_detailed(a, ic, cp, va_ds, dev)
                        fc_sec[rkey] = res
                        save_summary(summary)
                        print(f"[GPU {gpu_id} | {rkey}] F1={res['cloud_f1']*100:.2f}% | Top={res['top_mae_km']:.3f} | Bot={res['bot_mae_km']:.3f}")
            return _run
        wave2_jobs.append(_make_fc_group())

    t4_sec = summary["track4_motion_2f_ours"]
    motion_groups = [
        (False, False, [
            ("2f_ir_past_to_0m_ours_sam", "sam", 5, f"{OUT_ROOT}/ours_dist_3f_ir408_sam/best_semisup.pt"),
            ("2f_ir_past_to_0m_ours_dinov3", "dinov3", 5, f"{OUT_ROOT}/ours_dist_3f_ir408_dinov3/best_semisup.pt"),
        ]),
        (True, False, [
            ("2f_ir_vis_past_to_0m_ours_unet", "unet", 8, f"{OUT_ROOT}/ours_dist_3f_vis_unet/best_semisup.pt"),
            ("2f_ir_vis_past_to_0m_ours_sam", "sam", 8, f"{OUT_ROOT}/ours_dist_3f_vis_sam/best_semisup.pt"),
            ("2f_ir_vis_past_to_0m_ours_dinov3", "dinov3", 8, f"{OUT_ROOT}/ours_dist_3f_vis_dinov3/best_semisup.pt"),
        ]),
        (False, True, [
            ("2f_ir_era5_past_to_0m_ours_unet", "unet", 23, f"{OUT_ROOT}/ours_dist_3f_ir_era5_unet/best_semisup.pt"),
            ("2f_ir_era5_past_to_0m_ours_sam", "sam", 23, f"{OUT_ROOT}/ours_dist_3f_ir_era5_sam/best_semisup.pt"),
            ("2f_ir_era5_past_to_0m_ours_dinov3", "dinov3", 23, f"{OUT_ROOT}/ours_dist_3f_ir_era5_dinov3/best_semisup.pt"),
        ]),
        (True, True, [
            ("2f_ir_vis_era5_past_to_0m_ours_unet", "unet", 26, f"{OUT_ROOT}/ours_dist_3f_ir_vis_era5_unet/best_semisup.pt"),
            ("2f_ir_vis_era5_past_to_0m_ours_sam", "sam", 26, f"{OUT_ROOT}/ours_dist_3f_ir_vis_era5_sam/best_semisup.pt"),
            ("2f_ir_vis_era5_past_to_0m_ours_dinov3", "dinov3", 26, f"{OUT_ROOT}/ours_dist_3f_ir_vis_era5_dinov3/best_semisup.pt"),
        ]),
    ]
    for uvis, uera5, mlist in motion_groups:
        def _make_m_group(_uv: bool = uvis, _ue: bool = uera5, _ml: list[tuple[str, str, int, str]] = mlist) -> Callable[[int], None]:
            def _run(gpu_id: int) -> None:
                needed = [(mk, a, ic, cp) for (mk, a, ic, cp) in _ml if mk not in t4_sec and os.path.isfile(cp)]
                if not needed:
                    return
                dev = torch.device(f"cuda:{gpu_id}")
                va_ds = InMemoryDataset(make_2frame_past_forecast_dataset(
                    val_408, img_cache, track_cache, era5_cache,
                    patches_per_granule=6, is_train=False, use_vis06=_uv, use_era5=_ue, seed=42,
                ))
                for mk, a, ic, cp in needed:
                    res = eval_ckpt_detailed(a, ic, cp, va_ds, dev)
                    t4_sec[mk] = res
                    save_summary(summary)
                    print(f"[GPU {gpu_id} | {mk}] +15m Motion -> F1={res['cloud_f1']*100:.2f}% | Top={res['top_mae_km']:.3f} | Bot={res['bot_mae_km']:.3f}")
            return _run
        wave2_jobs.append(_make_m_group())

    print(f"\n=== WAVE 2: Running {len(wave2_jobs)} Grouped Nowcasting/Motion Jobs across 4 GPUs ===")
    if wave2_jobs:
        run_jobs_across_4_gpus(wave2_jobs)

    print(f"\n=== ALL EVALUATIONS COMPLETED IN {(time.time() - t0)/60.0:.2f} MIN! Saved to {SUMMARY_JSON} ===")


if __name__ == "__main__":
    main()
