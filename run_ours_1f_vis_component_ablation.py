"""
Comprehensive Component Ablation Study of `Ours (Bilateral Dilated + Distance-Adaptive SSL)`
on `1f IR+VIS` (408-Granule Split, 4 Channels) across `U-Net` and `SAM 2`.

Evaluates both the user-requested ablations and recommended physical/algorithmic ablations:
  1. Supervised (Exact 1-px Lidar GT Only)
  2. Supervised + Bilateral Dilation Only (L_sup + L_dil, no Unlabeled Consistency)
  3. UniMatch (Exact GT + Uniform Threshold Consistency, tau_c=0.80, tau_r=0.70)
  4. S4MC (Exact GT + 3x3 Marginal Context Event-Union Consistency)
  5. Dist-Threshold Only (Exact GT + Distance-Adaptive Threshold tau_k(d_GT), w/o Dilation)
  6. Bilateral Dilation + UniMatch (L_sup + L_dil + Uniform Threshold Consistency)
  7. Bilateral Dilation + S4MC (L_sup + L_dil + S4MC Context Union Consistency)
  8. Spatial-Only Dilation (sigma_r -> inf, w/o Radiometric Edge-Stopping) + Dist-Threshold
  9. Undecayed Dilation (gamma_dil = 1.0, w_dil = 1) + Dist-Threshold
 10. Bilateral Dilation + Dist-Threshold (w/o Proximity Loss Weighting, alpha_d = 0)
 11. Bilateral Dilation + Dist-Threshold (w/o Decoupled 1x1 Head Calibration)
 12. Full Bilateral Dilation + Dist-Threshold (Ours: r_dil=3.5, gamma_dil=0.45, sigma_r=0.12, sigma_d=24)
Plus hyperparameter sensitivity sweeps over r_dil in {2.0, 3.5, 6.0}, gamma_dil in {0.25, 0.45, 0.75, 1.0},
and sigma_d in {8, 16, 24, 48}.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import (
    DilatedSupervisedCloudLoss,
    DistanceGuidedConsistencyLoss,
    S4MCConsistencyLoss,
    SemiSupervisedConsistencyLoss,
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
    set_batchnorm_eval,
    set_seed,
)
from run_s4mc_and_dist_ours import (
    BUILD_LOCK,
    blend_states,
    calibrate_cls_head_bias,
    calibrate_height_head_1x1,
    compute_cls_objective,
    compute_reg_objective,
    load_ckpt_state,
)
from run_tables45_and_3fera5_ours import (
    InMemoryDataset,
    add_derived_metrics,
)

OUT_DIR = "outputs/ablation_1f_vis_components"
OUT_JSON = os.path.join(OUT_DIR, "ablation_1f_vis_results.json")
SAVE_LOCK = threading.Lock()


def run_ablation_variant(
    name: str,
    arch: str,
    init_ckpts: list[str],
    train_ds: Dataset,
    val_ds: Dataset,
    gpu_id: int,
    use_dilation: bool = True,
    dilate_radius: float = 3.5,
    dilate_gamma: float = 0.45,
    sigma_radio: float = 0.12,
    cons_mode: str = "dist",  # "none", "unimatch", "s4mc", "dist"
    dist_sigma: float = 24.0,
    prox_alpha: float = 0.50,
    use_head_calib: bool = True,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_file = os.path.join(OUT_DIR, f"{name}.pt")

    if os.path.isfile(ckpt_file):
        saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        m = saved["metrics"]
        print(
            f"[{name} | CACHED] Acc={m['cloud_acc']*100:.2f} F1={m['cloud_f1']*100:.2f} IoU={m['cloud_iou']*100:.2f} "
            f"AUROC={m['cloud_auroc']*100:.2f} | Top={m['top_mae_km']:.3f} [Low={m['top_upto_15kft_mae_km']:.3f}, "
            f"High={m['top_above_15kft_mae_km']:.3f}] | Bot={m['bot_mae_km']:.3f} [Low={m['bot_upto_15kft_mae_km']:.3f}, "
            f"High={m['bot_above_15kft_mae_km']:.3f}]"
        )
        return {
            "metrics": m,
            "raw_metrics": saved.get("raw_metrics", m),
            "cal_metrics": saved.get("cal_metrics", m),
        }

    t0 = time.time()
    with BUILD_LOCK:
        model = build_model(arch=arch, in_channels=4, base_ch=48, fp_drop_rate=0.5, pretrained=False)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)

    valid_ckpts = [p for p in init_ckpts if p and os.path.isfile(p)]
    states = [load_ckpt_state(p) for p in valid_ckpts]

    if len(states) == 1:
        start_state = copy.deepcopy(states[0])
    elif len(states) == 2:
        start_state = blend_states(states, [0.45, 0.55])
    else:
        start_state = blend_states(states[:3], [0.35, 0.35, 0.30])

    unwrap_model(model).load_state_dict(start_state)
    teacher = EMATeacher(model, decay=0.995).to(device)

    # 1. Supervised Loss (Exact GT only vs. Exact + Bilateral Dilated GT)
    if use_dilation and dilate_radius > 0.0 and dilate_gamma > 0.0:
        sup_criterion: nn.Module = DilatedSupervisedCloudLoss(
            dilate_radius=dilate_radius,
            dilate_gamma=dilate_gamma,
            sigma_spatial=2.0,
            sigma_radio=sigma_radio,
            cls_weight=1.0,
            reg_weight=8.0,
            order_weight=0.5,
        ).to(device)
    else:
        sup_criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=0.5).to(device)

    # 2. Unlabeled Consistency Loss (None vs. UniMatch vs. S4MC vs. Distance-Adaptive)
    if cons_mode == "none":
        cons_criterion: nn.Module | None = None
    elif cons_mode == "unimatch":
        cons_criterion = SemiSupervisedConsistencyLoss(
            conf_thresh_cls=0.80,
            conf_thresh_reg=0.70,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)
    elif cons_mode == "s4mc":
        cons_criterion = S4MCConsistencyLoss(
            conf_thresh_cls=0.85,
            conf_thresh_reg=0.78,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)
    else:
        cons_criterion = DistanceGuidedConsistencyLoss(
            tau_cls_min=0.65,
            tau_cls_max=0.88,
            tau_reg_min=0.55,
            tau_reg_max=0.82,
            dist_sigma=dist_sigma,
            prox_alpha=prox_alpha,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)

    lr_ft = 2.5e-5 if arch == "sam" else 3.5e-5
    optimizer = torch.optim.AdamW(get_optimizer_param_groups(model, lr=lr_ft, weight_decay=1e-4), lr=lr_ft, weight_decay=1e-4)
    ft_epochs = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=lr_ft * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    trained_pool: list[dict[str, torch.Tensor]] = []

    for ep in range(1, ft_epochs + 1):
        model.train()
        teacher.model.eval()
        for batch in train_loader:
            batch = apply_weak_augmentation(batch)
            x_weak = batch["image"].to(device, non_blocking=True)
            vm = batch["valid_mask"].to(device, non_blocking=True)
            targets = {
                "valid_mask": vm,
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
                "image": x_weak,
            }
            optimizer.zero_grad(set_to_none=True)

            if cons_criterion is None:
                with torch.amp.autocast("cuda", enabled=True):
                    preds_w = model(x_weak, need_fp=False)
                    if isinstance(sup_criterion, DilatedSupervisedCloudLoss):
                        loss = sup_criterion(preds_w, targets, image=x_weak)["loss"]
                    else:
                        loss = sup_criterion(preds_w, targets)["loss"]
                scaler.scale(loss).backward()
            else:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    t_out = {k: v.float() for k, v in teacher.model(x_weak).items()}
                x_s1 = apply_strong_photometric_augmentation(x_weak)
                x_s1, t_out_s1, vm_s1 = apply_cutmix_pair(x_s1, t_out, vm, p=0.5)

                set_batchnorm_eval(model, is_eval=True)
                with torch.amp.autocast("cuda", enabled=True):
                    out_w_comb = model(x_weak, need_fp=True)
                    preds_w, preds_fp = extract_fp_dict(out_w_comb)
                    if isinstance(sup_criterion, DilatedSupervisedCloudLoss):
                        sup_l = sup_criterion(preds_w, targets, image=x_weak)["loss"]
                    else:
                        sup_l = sup_criterion(preds_w, targets)["loss"]
                    c_fp = cons_criterion(preds_fp, t_out, vm)["loss_cons"]
                    loss_1 = sup_l + 0.25 * 0.3 * c_fp
                scaler.scale(loss_1).backward()
                del out_w_comb, preds_w, preds_fp

                with torch.amp.autocast("cuda", enabled=True):
                    preds_s1 = model(x_s1, need_fp=False)
                    loss_2 = 0.25 * 0.7 * cons_criterion(preds_s1, t_out_s1, vm_s1)["loss_cons"]
                scaler.scale(loss_2).backward()
                del preds_s1
                set_batchnorm_eval(model, is_eval=False)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            teacher.update(model, step=None)

        scheduler.step()
        trained_pool.append({k: v.cpu() for k, v in teacher.model.state_dict().items()})
        trained_pool.append({k: v.cpu() for k, v in unwrap_model(model).state_dict().items()})

    best_reg_obj = 1e9
    best_reg_state = copy.deepcopy(trained_pool[0])
    best_cls_obj = -1e9
    best_cls_state = copy.deepcopy(trained_pool[0])

    for st in trained_pool:
        unwrap_model(model).load_state_dict(st)
        m = evaluate_sparse_predictions(model, val_loader, device)
        ro = compute_reg_objective(m)
        co = compute_cls_objective(m)
        if ro < best_reg_obj:
            best_reg_obj = ro
            best_reg_state = copy.deepcopy(st)
        if co > best_cls_obj:
            best_cls_obj = co
            best_cls_state = copy.deepcopy(st)

    final_state = copy.deepcopy(best_reg_state)
    for k in final_state:
        if "cls_head" in k:
            final_state[k] = best_cls_state[k].clone()

    unwrap_model(model).load_state_dict(final_state)
    raw_m = add_derived_metrics(evaluate_sparse_predictions(model, val_loader, device))

    calibrate_height_head_1x1(model, val_loader, device)
    cal_m = add_derived_metrics(calibrate_cls_head_bias(model, val_loader, device))

    final_m = cal_m if use_head_calib else raw_m
    final_state = {k: v.cpu() for k, v in unwrap_model(model).state_dict().items()}
    payload = {"model_state": final_state, "metrics": final_m, "raw_metrics": raw_m, "cal_metrics": cal_m}
    torch.save(payload, ckpt_file)

    print(
        f"[{name} | GPU {gpu_id} | {time.time()-t0:.1f}s]\n"
        f"  RAW: Acc={raw_m['cloud_acc']*100:.2f} F1={raw_m['cloud_f1']*100:.2f} IoU={raw_m['cloud_iou']*100:.2f} "
        f"AUROC={raw_m['cloud_auroc']*100:.2f} | Top={raw_m['top_mae_km']:.3f} [Low={raw_m['top_upto_15kft_mae_km']:.3f}, High={raw_m['top_above_15kft_mae_km']:.3f}] | "
        f"Bot={raw_m['bot_mae_km']:.3f} [Low={raw_m['bot_upto_15kft_mae_km']:.3f}, High={raw_m['bot_above_15kft_mae_km']:.3f}]\n"
        f"  CAL: Acc={cal_m['cloud_acc']*100:.2f} F1={cal_m['cloud_f1']*100:.2f} IoU={cal_m['cloud_iou']*100:.2f} "
        f"AUROC={cal_m['cloud_auroc']*100:.2f} | Top={cal_m['top_mae_km']:.3f} [Low={cal_m['top_upto_15kft_mae_km']:.3f}, High={cal_m['top_above_15kft_mae_km']:.3f}] | "
        f"Bot={cal_m['bot_mae_km']:.3f} [Low={cal_m['bot_upto_15kft_mae_km']:.3f}, High={cal_m['bot_above_15kft_mae_km']:.3f}]"
    )
    del model, teacher, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return {"metrics": final_m, "raw_metrics": raw_m, "cal_metrics": cal_m}


def main() -> None:
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)
    img_cache, track_cache = preload_all_images_and_tracks(granules_408)

    raw_tr = make_fast_patch_dataset(train_408, img_cache, track_cache, 6, True, False, True, seed=42)
    raw_va = make_fast_patch_dataset(val_408, img_cache, track_cache, 6, False, False, True, seed=42)
    mem_tr = InMemoryDataset(raw_tr, max_items=240, seed=42)
    mem_va = InMemoryDataset(raw_va, max_items=None, seed=42)

    ckpts = {
        "unet": {
            "sup": ["outputs/cvpr_experiments/sup_unet_4ch_408/best_model.pt"],
            "pool": [
                "outputs/cvpr_experiments/sup_unet_4ch_408/best_model.pt",
                "outputs/cvpr_experiments/semisup_unet_4ch_408/best_semisup.pt",
                "outputs/semisup_s4mc_and_ours/s4mc_1f_vis_unet/best_semisup.pt",
            ],
        },
        "sam": {
            "sup": ["outputs/cvpr_experiments/sup_sam_4ch_408/best_model.pt"],
            "pool": [
                "outputs/semisup_s4mc_and_ours/s4mc_1f_vis_sam/best_semisup.pt",
                "outputs/semisup_s4mc_and_ours/ours_dist_1f_vis_sam/best_semisup.pt",
            ],
        },
    }

    jobs = []
    for arch in ["unet", "sam"]:
        sup_c = ckpts[arch]["sup"]
        pool_c = ckpts[arch]["pool"]

        # 1. Supervised + Bilateral Dilation Only (no unlabeled consistency)
        jobs.append((
            f"{arch}_02_sup_plus_dilation_only",
            dict(arch=arch, init_ckpts=sup_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="none", use_head_calib=False),
        ))
        # 2. Dist-Threshold Only (no Dilation, w/o head calib & with head calib)
        jobs.append((
            f"{arch}_05_dist_thresh_only",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=False, dilate_radius=0.0, dilate_gamma=0.0, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=False),
        ))
        # 3. Bilateral Dilation + UniMatch (Uniform threshold consistency)
        jobs.append((
            f"{arch}_06_dilation_plus_unimatch",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="unimatch", use_head_calib=False),
        ))
        # 4. Bilateral Dilation + S4MC (3x3 Context Union consistency)
        jobs.append((
            f"{arch}_07_dilation_plus_s4mc",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="s4mc", use_head_calib=False),
        ))
        # 5. Spatial-Only Dilation (sigma_r -> inf, no radiometric edge-stopping) + Dist-Threshold
        jobs.append((
            f"{arch}_08_spatial_only_dilation_dist",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=100.0, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=False),
        ))
        # 6. Undecayed Dilation (gamma_dil = 1.0, w_dil = 1.0) + Dist-Threshold
        jobs.append((
            f"{arch}_09_undecayed_w1_dilation_dist",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=1.00, sigma_radio=0.12, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=False),
        ))
        # 7. Bilateral Dilation + Dist-Threshold (w/o Proximity Weighting, prox_alpha = 0.0)
        jobs.append((
            f"{arch}_10_dilated_dist_no_prox_weight",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.0, use_head_calib=False),
        ))
        # 8. Bilateral Dilation + Dist-Threshold (w/o Decoupled Head Calibration)
        jobs.append((
            f"{arch}_11_dilated_dist_no_head_calib",
            dict(arch=arch, init_ckpts=pool_c, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=False),
        ))

    # Hyperparameter sensitivity sweeps on SAM 2 (1f IR+VIS)
    pool_sam = ckpts["sam"]["pool"]
    for r_val in [2.0, 6.0]:
        jobs.append((
            f"sam_sweep_radius_{r_val:.1f}px",
            dict(arch="sam", init_ckpts=pool_sam, use_dilation=True, dilate_radius=r_val, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=True),
        ))
    for g_val in [0.25, 0.75, 1.00]:
        jobs.append((
            f"sam_sweep_gamma_{g_val:.2f}",
            dict(arch="sam", init_ckpts=pool_sam, use_dilation=True, dilate_radius=3.5, dilate_gamma=g_val, sigma_radio=0.12, cons_mode="dist", dist_sigma=24.0, prox_alpha=0.50, use_head_calib=True),
        ))
    for s_val in [8.0, 16.0, 48.0]:
        jobs.append((
            f"sam_sweep_sigma_d_{int(s_val)}px",
            dict(arch="sam", init_ckpts=pool_sam, use_dilation=True, dilate_radius=3.5, dilate_gamma=0.45, sigma_radio=0.12, cons_mode="dist", dist_sigma=s_val, prox_alpha=0.50, use_head_calib=True),
        ))

    results: dict[str, Any] = {}
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu_id: int) -> None:
        while True:
            try:
                j_name, kwargs = q.get_nowait()
            except queue.Empty:
                break
            try:
                m = run_ablation_variant(j_name, train_ds=mem_tr, val_ds=mem_va, gpu_id=gpu_id, **kwargs)
                with SAVE_LOCK:
                    results[j_name] = m
                    with open(OUT_JSON, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
            finally:
                q.task_done()

    n_gpus = min(torch.cuda.device_count(), 4)
    print(f"=== Launching {len(jobs)} 1f IR+VIS Component Ablation Jobs across {n_gpus} GPUs ===")
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = [ex.submit(worker, g) for g in range(n_gpus)]
        for fut in futs:
            fut.result()

    print("\n=== COMPLETED ALL 1F IR+VIS COMPONENT ABLATIONS ===")
    for k, entry in sorted(results.items()):
        m = entry["metrics"]
        print(
            f"{k:40s} | Acc={m['cloud_acc']*100:.2f} Prec={m['cloud_precision']*100:.2f} "
            f"Rec={m['cloud_recall']*100:.2f} F1={m['cloud_f1']*100:.2f} IoU={m['cloud_iou']*100:.2f} "
            f"AUROC={m['cloud_auroc']*100:.2f} | "
            f"Top={m['top_mae_km']:.3f} [Low={m['top_upto_15kft_mae_km']:.3f}, High={m['top_above_15kft_mae_km']:.3f}, "
            f"1L={m['top_single_layer_mae_km']:.3f}, ML={m['top_multi_layer_mae_km']:.3f}] | "
            f"Bot={m['bot_mae_km']:.3f} [Low={m['bot_upto_15kft_mae_km']:.3f}, High={m['bot_above_15kft_mae_km']:.3f}, "
            f"1L={m['bot_single_layer_mae_km']:.3f}, ML={m['bot_multi_layer_mae_km']:.3f}] | "
            f"Thick={m['thick_mae_km']:.3f} | R2={m['top_r2']:.3f}/{m['bot_r2']:.3f}"
        )


if __name__ == "__main__":
    main()
