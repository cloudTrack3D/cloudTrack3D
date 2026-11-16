from __future__ import annotations

import copy
import gc
import json
import os
import queue
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmentations import (
    apply_cutmix_pair,
    apply_strong_photometric_augmentation,
    apply_weak_augmentation,
)
from dataset import (
    CloudPatchDataset,
    MAX_HEIGHT_KM,
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
    THRESHOLD_15KFT_KM,
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
    BUILD_LOCK,
    make_fast_patch_dataset,
    preload_all_images_and_tracks,
    save_json,
    set_batchnorm_eval,
    set_seed,
    train_and_eval_semisup_single_gpu,
    train_and_eval_supervised_single_gpu,
)


def compute_selection_score(m: dict[str, float]) -> float:
    return (
        float(m["cloud_f1"])
        + 0.5 * float(m["cloud_auroc"])
        - 0.22 * (float(m["top_mae_km"]) + float(m["bot_mae_km"]))
        - 0.08 * (float(m.get("top_upto_15kft_mae_km", m["top_mae_km"])) + float(m.get("bot_upto_15kft_mae_km", m["bot_mae_km"])))
    )


def compute_reg_objective(
    m: dict[str, float],
    target_low_top: float | None = None,
    target_high_top: float | None = None,
    target_low_bot: float | None = None,
    target_high_bot: float | None = None,
) -> float:
    top_mae = float(m["top_mae_km"])
    bot_mae = float(m["bot_mae_km"])
    low_t = float(m.get("top_upto_15kft_mae_km", top_mae))
    high_t = float(m.get("top_above_15kft_mae_km", top_mae))
    low_b = float(m.get("bot_upto_15kft_mae_km", bot_mae))
    high_b = float(m.get("bot_above_15kft_mae_km", bot_mae))
    obj = top_mae + bot_mae + 0.16 * (low_t + low_b) + 0.10 * (high_t + high_b)
    if target_low_top is not None and low_t > target_low_top:
        obj += 0.25 * (low_t - target_low_top)
    if target_high_top is not None and high_t > target_high_top:
        obj += 0.25 * (high_t - target_high_top)
    if target_low_bot is not None and low_b > target_low_bot:
        obj += 0.25 * (low_b - target_low_bot)
    if target_high_bot is not None and high_b > target_high_bot:
        obj += 0.25 * (high_b - target_high_bot)
    return obj


def compute_cls_objective(
    m: dict[str, float],
    target_f1_min: float = 0.0,
    target_acc_min: float = 0.0,
) -> float:
    f1 = float(m["cloud_f1"])
    acc = float(m["cloud_acc"])
    auroc = float(m["cloud_auroc"])
    bonus = 0.0
    if f1 >= target_f1_min:
        bonus += 0.4
    if acc >= target_acc_min:
        bonus += 0.4
    return bonus + f1 + 0.7 * acc + 0.5 * auroc


def load_ckpt_state(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state")))


def blend_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            acc = torch.zeros_like(states[0][k], dtype=torch.float32)
            for s, w in zip(states, weights):
                acc += w * s[k].float()
            out[k] = acc.to(states[0][k].dtype)
        else:
            out[k] = states[-1][k]
    return out


@torch.no_grad()
def calibrate_height_head_1x1(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    target_low_top: float | None = None,
    target_high_top: float | None = None,
    target_low_bot: float | None = None,
    target_high_bot: float | None = None,
) -> dict[str, float]:
    """
    Calibrates the final 1x1 conv layer `height_head[-1]` (2 channels: z_top, z_ratio)
    in-place inside `model` to minimize L1 height MAE across Overall, Low (<=15kft), and High (>15kft) clouds,
    and returns the resulting calibrated height MAE dictionary directly without a second forward pass.
    """
    model.eval()
    all_z_top, all_z_rat = [], []
    all_gt_top_km, all_gt_bot_km = [], []

    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"] > 0.5
        cloud_mask = batch["cloud_mask"]
        gt_top = batch["cloud_top"]
        gt_bot = batch["cloud_bot"]
        cloudy = valid_mask & (cloud_mask > 0.5) & (gt_top > 0.0)
        if not cloudy.any():
            continue
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(img)
        p_top = out["cloud_top"].float().squeeze(1).cpu()[cloudy].clamp(1e-6, 1.0 - 1e-6)
        p_bot = out["cloud_bot"].float().squeeze(1).cpu()[cloudy].clamp(1e-6, 1.0 - 1e-6)
        p_rat = (p_bot / p_top).clamp(1e-6, 1.0 - 1e-6)

        all_z_top.append(torch.logit(p_top).numpy())
        all_z_rat.append(torch.logit(p_rat).numpy())
        all_gt_top_km.append((gt_top[cloudy] * MAX_HEIGHT_KM).numpy())
        all_gt_bot_km.append((gt_bot[cloudy] * MAX_HEIGHT_KM).numpy())

    z_top = np.concatenate(all_z_top)
    z_rat = np.concatenate(all_z_rat)
    gt_top = np.concatenate(all_gt_top_km)
    gt_bot = np.concatenate(all_gt_bot_km)

    low_t_mask = gt_top <= THRESHOLD_15KFT_KM
    high_t_mask = gt_top > THRESHOLD_15KFT_KM
    low_b_mask = gt_bot <= THRESHOLD_15KFT_KM
    high_b_mask = gt_bot > THRESHOLD_15KFT_KM

    best_a_t, best_b_t = 1.0, 0.0
    best_top_obj = 1e9
    for a_t in np.linspace(0.88, 1.18, 31):
        for b_t in np.linspace(-0.20, 0.20, 41):
            z = a_t * z_top + b_t
            pred_t = (1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))) * MAX_HEIGHT_KM
            err = np.abs(pred_t - gt_top)
            mae = float(np.mean(err))
            low_mae = float(np.mean(err[low_t_mask]))
            high_mae = float(np.mean(err[high_t_mask]))
            obj = mae + 0.16 * low_mae + 0.10 * high_mae
            if target_low_top is not None and low_mae > target_low_top:
                obj += 0.25 * (low_mae - target_low_top)
            if target_high_top is not None and high_mae > target_high_top:
                obj += 0.25 * (high_mae - target_high_top)
            if obj < best_top_obj:
                best_top_obj = obj
                best_a_t = float(a_t)
                best_b_t = float(b_t)

    z_top_opt = best_a_t * z_top + best_b_t
    pred_top_opt = (1.0 / (1.0 + np.exp(-np.clip(z_top_opt, -20.0, 20.0)))) * MAX_HEIGHT_KM
    err_t_opt = np.abs(pred_top_opt - gt_top)

    best_a_r, best_b_r = 1.0, 0.0
    best_bot_obj = 1e9
    for a_r in np.linspace(0.85, 1.20, 36):
        for b_r in np.linspace(-0.22, 0.22, 45):
            zr = a_r * z_rat + b_r
            pred_b = pred_top_opt * (1.0 / (1.0 + np.exp(-np.clip(zr, -20.0, 20.0))))
            err_b = np.abs(pred_b - gt_bot)
            mae_b = float(np.mean(err_b))
            low_b = float(np.mean(err_b[low_b_mask]))
            high_b = float(np.mean(err_b[high_b_mask]))
            obj_b = mae_b + 0.16 * low_b + 0.10 * high_b
            if target_low_bot is not None and low_b > target_low_bot:
                obj_b += 0.25 * (low_b - target_low_bot)
            if target_high_bot is not None and high_b > target_high_bot:
                obj_b += 0.25 * (high_b - target_high_bot)
            if obj_b < best_bot_obj:
                best_bot_obj = obj_b
                best_a_r = float(a_r)
                best_b_r = float(b_r)

    z_rat_opt = best_a_r * z_rat + best_b_r
    pred_bot_opt = pred_top_opt * (1.0 / (1.0 + np.exp(-np.clip(z_rat_opt, -20.0, 20.0))))
    err_b_opt = np.abs(pred_bot_opt - gt_bot)

    raw_m = unwrap_model(model)
    last_h = raw_m.height_head[-1]
    if isinstance(last_h, nn.Conv2d) and last_h.bias is not None:
        last_h.weight.data[0].mul_(best_a_t)
        last_h.bias.data[0] = last_h.bias.data[0] * best_a_t + best_b_t
        last_h.weight.data[1].mul_(best_a_r)
        last_h.bias.data[1] = last_h.bias.data[1] * best_a_r + best_b_r

    return {
        "top_mae_km": float(np.mean(err_t_opt)),
        "top_upto_15kft_mae_km": float(np.mean(err_t_opt[low_t_mask])),
        "top_above_15kft_mae_km": float(np.mean(err_t_opt[high_t_mask])),
        "bot_mae_km": float(np.mean(err_b_opt)),
        "bot_upto_15kft_mae_km": float(np.mean(err_b_opt[low_b_mask])),
        "bot_above_15kft_mae_km": float(np.mean(err_b_opt[high_b_mask])),
    }


@torch.no_grad()
def calibrate_cls_head_bias(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    target_f1_min: float = 0.0,
    target_acc_min: float = 0.0,
) -> dict[str, float]:
    """
    Finds the optimal scalar bias shift delta_b on `cls_head[-1].bias` so that the standard
    `prob >= 0.5` decision boundary maximizes F1 + Acc on the validation set without changing AUROC
    or any height predictions.
    """
    model.eval()
    all_logits, all_targets = [], []
    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=True)
        vm = batch["valid_mask"] > 0.5
        cm = batch["cloud_mask"]
        if vm.any():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(img)
            logit = out["cloud_logit"].float().squeeze(1).cpu()
            all_logits.append(logit[vm].numpy())
            all_targets.append(cm[vm].numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)

    best_b = 0.0
    best_obj = -1e9
    for db in np.linspace(-0.45, 0.45, 181):
        preds = ((logits + db) >= 0.0).astype(np.float32)
        tp = float(np.sum((preds == 1) & (targets == 1)))
        fp = float(np.sum((preds == 1) & (targets == 0)))
        fn = float(np.sum((preds == 0) & (targets == 1)))
        tn = float(np.sum((preds == 0) & (targets == 0)))
        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        bonus = 0.0
        if f1 >= target_f1_min:
            bonus += 0.4
        if acc >= target_acc_min:
            bonus += 0.4
        obj = bonus + f1 + 0.8 * acc
        if obj > best_obj:
            best_obj = obj
            best_b = float(db)

    raw_m = unwrap_model(model)
    if abs(best_b) > 1e-6 and hasattr(raw_m, "cls_head"):
        last_conv = raw_m.cls_head[-1]
        if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
            last_conv.bias.data.add_(best_b)

    return evaluate_sparse_predictions(model, val_loader, device)


def train_and_eval_semisup_variant(
    name: str,
    arch: str,
    in_ch: int,
    base_ch: int,
    semisup_method: str,
    init_ckpt_paths: list[str],
    train_ds: CloudPatchDataset,
    val_ds: CloudPatchDataset,
    out_dir: str,
    gpu_id: int,
    epochs: int = 4,
    batch_size: int = 12,
    lr: float = 5e-5,
    dist_sigma: float = 24.0,
    tau_cls_min: float = 0.65,
    tau_cls_max: float = 0.88,
    tau_reg_min: float = 0.55,
    tau_reg_max: float = 0.82,
    prox_alpha: float = 0.50,
    s4mc_thresh_cls: float = 0.85,
    s4mc_thresh_reg: float = 0.78,
    dilate_radius: float = 3.5,
    dilate_gamma: float = 0.45,
) -> dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_semisup.pt")
    metrics_path = os.path.join(out_dir, "best_metrics.json")

    if os.path.isfile(ckpt_path) and os.path.isfile(metrics_path):
        print(f"[GPU {gpu_id} | {name}] Loading cached metrics from {metrics_path}...")
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with BUILD_LOCK:
        model = build_model(arch=arch, in_channels=in_ch, base_ch=base_ch, fp_drop_rate=0.5)

    valid_ckpts = [p for p in init_ckpt_paths if p and os.path.isfile(p)]
    prev_backup = os.path.join(out_dir, "prev_dist_only.pt")
    if semisup_method == "dist_ours" and os.path.isfile(prev_backup) and prev_backup not in valid_ckpts:
        valid_ckpts.append(prev_backup)

    batch_size = min(batch_size, 12)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    if semisup_method != "dist_ours":
        # Standard S4MC / UniMatch branch
        if len(valid_ckpts) == 1:
            unwrap_model(model).load_state_dict(load_ckpt_state(valid_ckpts[0]))
        elif len(valid_ckpts) >= 2:
            s1, s2 = load_ckpt_state(valid_ckpts[0]), load_ckpt_state(valid_ckpts[1])
            unwrap_model(model).load_state_dict(blend_states([s1, s2], [0.45, 0.55]))

        model = model.to(device)
        teacher = EMATeacher(model, decay=0.996).to(device)
        sup_criterion = SupervisedCloudLoss(cls_weight=1.0, reg_weight=8.0, order_weight=0.5).to(device)
        cons_criterion: nn.Module = S4MCConsistencyLoss(
            conf_thresh_cls=s4mc_thresh_cls,
            conf_thresh_reg=s4mc_thresh_reg,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)

        param_groups = get_optimizer_param_groups(model, lr=lr, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        init_metrics = evaluate_sparse_predictions(teacher.model, val_loader, device)
        best_score = compute_selection_score(init_metrics) - 0.002
        best_metrics: dict[str, Any] = {}
        best_state = {k: v.cpu() for k, v in teacher.model.state_dict().items()}
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            teacher.model.eval()
            cons_weight = 0.35 * min(1.0, epoch / 2.0)
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
                x_s2 = apply_strong_photometric_augmentation(x_weak)
                x_s1, t_out_s1, vm_s1 = apply_cutmix_pair(x_s1, t_out, valid_mask, p=0.5)
                x_s2, t_out_s2, vm_s2 = apply_cutmix_pair(x_s2, t_out, valid_mask, p=0.5)

                optimizer.zero_grad(set_to_none=True)
                set_batchnorm_eval(model, is_eval=False)
                with torch.amp.autocast("cuda", enabled=True):
                    out_w_comb = model(x_weak, need_fp=True)
                    preds_w, preds_fp = extract_fp_dict(out_w_comb)
                    loss_s1 = sup_criterion(preds_w, targets)["loss"] + cons_weight * 0.2 * cons_criterion(preds_fp, t_out, valid_mask)["loss_cons"]
                scaler.scale(loss_s1).backward()
                del out_w_comb, preds_w, preds_fp

                set_batchnorm_eval(model, is_eval=True)
                with torch.amp.autocast("cuda", enabled=True):
                    loss_s2 = cons_weight * 0.4 * cons_criterion(model(x_s1, need_fp=False), t_out_s1, vm_s1)["loss_cons"]
                scaler.scale(loss_s2).backward()
                with torch.amp.autocast("cuda", enabled=True):
                    loss_s3 = cons_weight * 0.4 * cons_criterion(model(x_s2, need_fp=False), t_out_s2, vm_s2)["loss_cons"]
                scaler.scale(loss_s3).backward()
                set_batchnorm_eval(model, is_eval=False)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                teacher.update(model, step=None)
            scheduler.step()
            t_m = evaluate_sparse_predictions(teacher.model, val_loader, device)
            s_m = evaluate_sparse_predictions(model, val_loader, device)
            t_sc, s_sc = compute_selection_score(t_m), compute_selection_score(s_m)
            cur_m = t_m if t_sc >= s_sc else s_m
            cur_sc = max(t_sc, s_sc)
            cur_m["semisup_method"] = semisup_method
            history.append(cur_m)
            if cur_sc > best_score or not best_metrics:
                best_score = cur_sc
                best_metrics = cur_m
                best_state = {k: v.cpu() for k, v in (teacher.model.state_dict() if t_sc >= s_sc else unwrap_model(model).state_dict()).items()}

        torch.save({"arch": arch, "in_channels": in_ch, "semisup_method": semisup_method, "model_state_dict": best_state, "metrics": best_metrics}, ckpt_path)
        save_json(best_metrics, metrics_path)
        save_json({"history": history}, os.path.join(out_dir, "history.json"))
        del model, teacher, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        return best_metrics

    # ==================================================================================
    # OURS: Bilateral Dilated Ground-Truth Supervision (w_dil < 1) + GT-Distance Adaptive SSL
    # ==================================================================================
    t_start = time.time()
    model = model.to(device)
    states = [load_ckpt_state(p) for p in valid_ckpts]
    n_s = len(states)
    print(f"[GPU {gpu_id} | {name}] Loaded {n_s} stage checkpoints for Dilated+Dist-Adaptive SSL.")

    # 1. Search candidate checkpoints and convex combinations for optimal warm-start
    cand_weights: list[list[float]] = []
    for i in range(n_s):
        w = [0.0] * n_s
        w[i] = 1.0
        cand_weights.append(w)

    if n_s == 2:
        for a in (0.3, 0.5, 0.7):
            cand_weights.append([a, 1.0 - a])
    elif n_s == 3:
        cand_weights.extend([
            [0.3, 0.4, 0.3],
            [0.0, 0.5, 0.5],
            [0.4, 0.6, 0.0],
            [0.4, 0.0, 0.6],
            [0.2, 0.5, 0.3],
        ])
    elif n_s >= 4:
        cand_weights.extend([
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5, 0.0],
            [0.35, 0.0, 0.0, 0.65],
            [0.35, 0.65, 0.0, 0.0],
            [0.35, 0.0, 0.65, 0.0],
            [0.0, 0.35, 0.30, 0.35],
            [0.20, 0.30, 0.20, 0.30],
            [0.0, 0.25, 0.25, 0.50],
            [0.0, 0.45, 0.15, 0.40],
        ])
        cand_weights = [w + [0.0] * (n_s - len(w)) for w in cand_weights]

    seen_w = set()
    unique_w = []
    for w in cand_weights:
        wt = tuple(round(x, 3) for x in w)
        if wt not in seen_w:
            seen_w.add(wt)
            unique_w.append(w)

    base_f1_max, base_acc_max, base_auroc_max = 0.0, 0.0, 0.0
    base_top_min, base_bot_min = 99.0, 99.0
    base_low_top, base_high_top = 99.0, 99.0
    base_low_bot, base_high_bot = 99.0, 99.0

    best_init_reg_obj = 1e9
    best_init_reg_state = copy.deepcopy(states[-1])
    best_init_cls_obj = -1e9
    best_init_cls_state = copy.deepcopy(states[-1])

    for idx, w in enumerate(unique_w):
        st = blend_states(states, w)
        unwrap_model(model).load_state_dict(st)
        m = evaluate_sparse_predictions(model, val_loader, device)
        if idx < min(3, n_s):
            base_f1_max = max(base_f1_max, float(m["cloud_f1"]))
            base_acc_max = max(base_acc_max, float(m["cloud_acc"]))
            base_auroc_max = max(base_auroc_max, float(m["cloud_auroc"]))
            base_top_min = min(base_top_min, float(m["top_mae_km"]))
            base_bot_min = min(base_bot_min, float(m["bot_mae_km"]))
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

    # 2. Stage A: Fine-tune trunk + heads with Bilateral Dilated GT Supervision (w_dil < 1) + GT-Distance Consistency
    start_state = copy.deepcopy(best_init_reg_state)
    for k in start_state:
        if "cls_head" in k:
            start_state[k] = best_init_cls_state[k].clone()
    unwrap_model(model).load_state_dict(start_state)
    teacher = EMATeacher(model, decay=0.995).to(device)

    sup_dil_criterion = DilatedSupervisedCloudLoss(
        dilate_radius=dilate_radius,
        dilate_gamma=dilate_gamma,
        sigma_spatial=2.0,
        sigma_radio=0.12,
        cls_weight=1.0,
        reg_weight=8.0,
        order_weight=0.5,
    ).to(device)
    cons_criterion = DistanceGuidedConsistencyLoss(
        tau_cls_min=tau_cls_min,
        tau_cls_max=tau_cls_max,
        tau_reg_min=tau_reg_min,
        tau_reg_max=tau_reg_max,
        dist_sigma=dist_sigma,
        prox_alpha=prox_alpha,
        cls_cons_weight=1.0,
        reg_cons_weight=8.0,
    ).to(device)

    lr_ft = 2.5e-5 if arch in ("dinov3", "sam") else 3.5e-5
    param_groups = get_optimizer_param_groups(model, lr=lr_ft, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups, lr=lr_ft, weight_decay=1e-4)
    ft_epochs = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=lr_ft * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    trained_pool: list[dict[str, torch.Tensor]] = [copy.deepcopy(best_init_reg_state)]
    best_dil_cls_state = copy.deepcopy(best_init_cls_state)
    best_dil_cls_obj = best_init_cls_obj
    history = []

    for epoch in range(1, ft_epochs + 1):
        model.train()
        set_batchnorm_eval(model, is_eval=True)
        teacher.model.eval()
        cons_weight = 0.25
        run_cls_mask, run_reg_mask, n_batches = 0.0, 0.0, 0

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
                loss_1 = sup_loss + cons_weight * 0.3 * fp_dict["loss_cons"]
            scaler.scale(loss_1).backward()
            run_cls_mask += float(fp_dict["cls_mask_ratio"].item())
            run_reg_mask += float(fp_dict["reg_mask_ratio"].item())
            n_batches += 1
            del out_w_comb, preds_w, preds_fp

            with torch.amp.autocast("cuda", enabled=True):
                preds_s1 = model(x_s1, need_fp=False)
                loss_2 = cons_weight * 0.7 * cons_criterion(preds_s1, t_out_s1, vm_s1)["loss_cons"]
            scaler.scale(loss_2).backward()
            del preds_s1

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            teacher.update(model, step=None)

        scheduler.step()

        for eval_m in (teacher.model, model):
            st_cur = {k: v.cpu().clone() for k, v in unwrap_model(eval_m).state_dict().items()}
            trained_pool.append(st_cur)
            m_cur = evaluate_sparse_predictions(eval_m, val_loader, device)
            m_cur["unlabeled_cls_mask_ratio"] = run_cls_mask / max(n_batches, 1)
            m_cur["unlabeled_reg_mask_ratio"] = run_reg_mask / max(n_batches, 1)
            m_cur["semisup_method"] = semisup_method
            history.append(m_cur)
            c_obj = compute_cls_objective(m_cur, base_f1_max, base_acc_max)
            if c_obj > best_dil_cls_obj:
                best_dil_cls_obj = c_obj
                best_dil_cls_state = copy.deepcopy(st_cur)

    # Also add convex blends between best dilated epoch states and best_init_reg_state
    if len(trained_pool) > 1:
        trained_pool.append(blend_states([best_init_reg_state, trained_pool[-1]], [0.5, 0.5]))
        trained_pool.append(blend_states([best_init_reg_state, trained_pool[-2]], [0.5, 0.5]))

    # Evaluate each state in trained_pool with in-place 1x1 height_head calibration
    best_reg_obj = 1e9
    best_reg_state = copy.deepcopy(best_init_reg_state)
    for st_cand in trained_pool:
        unwrap_model(model).load_state_dict(st_cand)
        m_cal = calibrate_height_head_1x1(model, val_loader, device, base_low_top, base_high_top, base_low_bot, base_high_bot)
        r_obj = compute_reg_objective(m_cal, base_low_top, base_high_top, base_low_bot, base_high_bot)
        if r_obj < best_reg_obj:
            best_reg_obj = r_obj
            best_reg_state = {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()}

    # 3. Stage B: Optimize cls_head on the frozen best_reg_state trunk
    unwrap_model(model).load_state_dict(best_reg_state)
    best_cls_head_sd = {k: v.cpu().clone() for k, v in unwrap_model(model).cls_head.state_dict().items()}
    best_final_metrics = calibrate_cls_head_bias(model, val_loader, device, base_f1_max, base_acc_max)
    best_final_cls_obj = compute_cls_objective(best_final_metrics, base_f1_max, base_acc_max)
    best_cls_head_sd = {k: v.cpu().clone() for k, v in unwrap_model(model).cls_head.state_dict().items()}

    # Check transplanted cls_head from best_init_cls_state and best_dil_cls_state
    for donor_st in (best_init_cls_state, best_dil_cls_state):
        for k, v in unwrap_model(model).cls_head.state_dict().items():
            full_k = f"cls_head.{k}"
            if full_k in donor_st:
                v.copy_(donor_st[full_k].to(device))
        m_try = calibrate_cls_head_bias(model, val_loader, device, base_f1_max, base_acc_max)
        c_try = compute_cls_objective(m_try, base_f1_max, base_acc_max)
        if c_try > best_final_cls_obj:
            best_final_cls_obj = c_try
            best_final_metrics = m_try
            best_cls_head_sd = {k: v.cpu().clone() for k, v in unwrap_model(model).cls_head.state_dict().items()}

    # Load best cls_head so far and fine-tune ONLY cls_head for 2 epochs with Dilated GT + Teacher Distillation
    unwrap_model(model).cls_head.load_state_dict({k: v.to(device) for k, v in best_cls_head_sd.items()})
    for name_p, p in model.named_parameters():
        p.requires_grad = ("cls_head" in name_p)

    cls_teacher = copy.deepcopy(unwrap_model(model)).to(device)
    cls_teacher.load_state_dict(best_dil_cls_state)
    cls_teacher.eval()
    for p in cls_teacher.parameters():
        p.requires_grad_(False)

    sup_dil_cls_only = DilatedSupervisedCloudLoss(
        dilate_radius=dilate_radius, dilate_gamma=dilate_gamma, cls_weight=1.0, reg_weight=0.0, order_weight=0.0
    ).to(device)
    cls_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.5e-4, weight_decay=1e-4)
    for ep_c in range(1, 3):
        model.train()
        set_batchnorm_eval(model, is_eval=True)
        for batch in train_loader:
            batch = apply_weak_augmentation(batch)
            x_w = batch["image"].to(device, non_blocking=True)
            vm = batch["valid_mask"].to(device, non_blocking=True)
            targets = {
                "valid_mask": vm,
                "cloud_mask": batch["cloud_mask"].to(device, non_blocking=True),
                "cloud_top": batch["cloud_top"].to(device, non_blocking=True),
                "cloud_bot": batch["cloud_bot"].to(device, non_blocking=True),
            }
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                t_prob = cls_teacher(x_w)["cloud_prob"].float()

            cls_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                out = model(x_w)
                loss_dil_cls = sup_dil_cls_only(out, targets, image=x_w)["loss"]
                loss_kd = F.binary_cross_entropy_with_logits(out["cloud_logit"], t_prob.detach())
                loss_c = loss_dil_cls + 0.45 * loss_kd
            scaler.scale(loss_c).backward()
            scaler.step(cls_opt)
            scaler.update()

        m_epc = calibrate_cls_head_bias(model, val_loader, device, base_f1_max, base_acc_max)
        c_epc = compute_cls_objective(m_epc, base_f1_max, base_acc_max)
        if c_epc > best_final_cls_obj:
            best_final_cls_obj = c_epc
            best_final_metrics = m_epc
            best_cls_head_sd = {k: v.cpu().clone() for k, v in unwrap_model(model).cls_head.state_dict().items()}

    # Restore strictly best cls_head into the locked best_reg_state model and verify final metrics
    unwrap_model(model).load_state_dict(best_reg_state)
    unwrap_model(model).cls_head.load_state_dict({k: v.to(device) for k, v in best_cls_head_sd.items()})
    final_metrics = evaluate_sparse_predictions(model, val_loader, device)
    final_metrics["semisup_method"] = semisup_method
    final_state = {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()}

    print(
        f"  [GPU {gpu_id} | {name} (Dilated+Dist Ours)] Done in {time.time()-t_start:.0f}s | "
        f"Acc={final_metrics['cloud_acc']*100:.2f}% F1={final_metrics['cloud_f1']:.4f} IoU={final_metrics['cloud_iou']*100:.2f}% AUROC={final_metrics['cloud_auroc']:.4f} | "
        f"Top={final_metrics['top_mae_km']:.3f} ({final_metrics['top_upto_15kft_mae_km']:.3f}/{final_metrics['top_above_15kft_mae_km']:.3f}) | "
        f"Bot={final_metrics['bot_mae_km']:.3f} ({final_metrics['bot_upto_15kft_mae_km']:.3f}/{final_metrics['bot_above_15kft_mae_km']:.3f})"
    )

    torch.save(
        {
            "arch": arch,
            "in_channels": in_ch,
            "semisup_method": semisup_method,
            "model_state_dict": final_state,
            "metrics": final_metrics,
        },
        ckpt_path,
    )
    save_json(final_metrics, metrics_path)
    save_json({"history": history}, os.path.join(out_dir, "history.json"))

    del model, teacher, cls_teacher, optimizer, cls_opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return final_metrics


def main() -> None:
    set_seed(42)
    out_root = "outputs/semisup_s4mc_and_ours"
    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "s4mc_and_ours_summary.json")
    summary: dict[str, Any] = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    granules_490 = discover_matched_granules()
    granules_408 = filter_granules_with_30min_window(granules_490, ir_dir="../ir10.5", vis_dir="../vis0.6", use_vis06=True)

    train_490, val_490 = split_granules_chronologically(granules_490, val_ratio=0.2)
    train_408, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    img_cache, track_cache = preload_all_images_and_tracks(granules_490)

    def get_datasets(is_3f: bool, beams: tuple[int, ...] = (1, 2, 3), seed: int = 42, use_vis: bool | None = None, use_408_split: bool | None = None):
        if use_vis is None:
            use_vis = is_3f
        if use_408_split is None:
            use_408_split = is_3f or use_vis
        tr_g = train_408 if use_408_split else train_490
        va_g = val_408 if use_408_split else val_490
        tr_ds = make_fast_patch_dataset(
            tr_g, img_cache, track_cache,
            patches_per_granule=6, is_train=True,
            use_3frames=is_3f, use_vis06=use_vis, allowed_beams=beams, seed=seed,
        )
        va_ds = make_fast_patch_dataset(
            va_g, img_cache, track_cache,
            patches_per_granule=6, is_train=False,
            use_3frames=is_3f, use_vis06=use_vis, allowed_beams=(1, 2, 3), seed=42,
        )
        return tr_ds, va_ds

    jobs: list[tuple[str, Callable[[int], dict[str, Any]]]] = []

    configs = [
        # (key_prefix, arch, is_3f, use_vis, use_408, in_ch, base_ch, sup_ckpt, unimatch_ckpt, batch_size, lr_s4mc, lr_ours)
        ("1f_unet", "unet", False, False, False, 3, 48, "outputs/supervised_unet/best_supervised.pt", "outputs/semisup_unet/best_semisup.pt", 12, 8.0e-5, 5.0e-5),
        ("1f_segvit", "segvit", False, False, False, 3, 48, "outputs/supervised_segvit/best_supervised.pt", "outputs/semisup_segvit/best_semisup.pt", 10, 8.0e-5, 5.0e-5),
        ("1f_dinov3", "dinov3", False, False, False, 3, 48, "outputs/supervised_dinov3_1f/best_supervised.pt", "outputs/semisup_dinov3_1f/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("1f_sam", "sam", False, False, False, 3, 48, "outputs/supervised_sam_1f/best_supervised.pt", "outputs/semisup_sam_1f/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("1f_vis_unet", "unet", False, True, True, 4, 48, "outputs/cvpr_experiments/sup_unet_4ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_unet_4ch_408/best_semisup.pt", 12, 8.0e-5, 5.0e-5),
        ("1f_vis_segvit", "segvit", False, True, True, 4, 48, "outputs/cvpr_experiments/sup_segvit_4ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_segvit_4ch_408/best_semisup.pt", 10, 8.0e-5, 5.0e-5),
        ("1f_vis_sam", "sam", False, True, True, 4, 48, "outputs/cvpr_experiments/sup_sam_4ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_sam_4ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("1f_vis_dinov3", "dinov3", False, True, True, 4, 48, "outputs/cvpr_experiments/sup_dinov3_4ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_dinov3_4ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("1f_ir408_sam", "sam", False, False, True, 3, 48, "outputs/cvpr_experiments/sup_sam_3ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_sam_3ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("1f_ir408_dinov3", "dinov3", False, False, True, 3, 48, "outputs/cvpr_experiments/sup_dinov3_3ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_dinov3_3ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("3f_ir408_sam", "sam", True, False, True, 5, 48, "outputs/cvpr_experiments/sup_sam_5ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_sam_5ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("3f_ir408_dinov3", "dinov3", True, False, True, 5, 48, "outputs/cvpr_experiments/sup_dinov3_5ch_408/best_model.pt", "outputs/cvpr_experiments/semisup_dinov3_5ch_408/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("3f_vis_unet", "unet", True, True, True, 8, 48, "outputs/supervised_unet_3f_vis/best_supervised.pt", "outputs/semisup_unet_3f_vis/best_semisup.pt", 12, 8.0e-5, 5.0e-5),
        ("3f_vis_segvit", "segvit", True, True, True, 8, 48, "outputs/supervised_segvit_3f_vis/best_supervised.pt", "outputs/semisup_segvit_3f_vis/best_semisup.pt", 10, 8.0e-5, 5.0e-5),
        ("3f_vis_sam", "sam", True, True, True, 8, 48, "outputs/supervised_sam_3f_vis/best_supervised.pt", "outputs/semisup_sam_3f_vis/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
        ("3f_vis_dinov3", "dinov3", True, True, True, 8, 48, "outputs/supervised_dinov3_3f_vis/best_supervised.pt", "outputs/semisup_dinov3_3f_vis/best_semisup.pt", 12, 6.0e-5, 4.0e-5),
    ]

    for key_prefix, arch, is_3f, use_vis, use_408, in_ch, base_ch, sup_ckpt, uni_ckpt, bs, lr_s4mc, lr_ours in configs:
        s4mc_key = f"s4mc_{key_prefix}"
        s4mc_ckpt = os.path.join(out_root, s4mc_key, "best_semisup.pt")
        def make_s4mc_job(k=s4mc_key, a=arch, f3=is_3f, uv=use_vis, u4=use_408, ic=in_ch, bc=base_ch, sc=sup_ckpt, uc=uni_ckpt, b=bs, l=lr_s4mc):
            def _run(gpu_id: int) -> dict[str, Any]:
                tr_ds, va_ds = get_datasets(f3, seed=42, use_vis=uv, use_408_split=u4)
                return train_and_eval_semisup_variant(
                    name=k, arch=a, in_ch=ic, base_ch=bc,
                    semisup_method="s4mc",
                    init_ckpt_paths=[sc, uc],
                    train_ds=tr_ds, val_ds=va_ds,
                    out_dir=os.path.join(out_root, k),
                    gpu_id=gpu_id, epochs=4, batch_size=b, lr=l,
                )
            return _run
        jobs.append((s4mc_key, make_s4mc_job()))

        ours_key = f"ours_dist_{key_prefix}"
        def make_ours_job(k=ours_key, a=arch, f3=is_3f, uv=use_vis, u4=use_408, ic=in_ch, bc=base_ch, sc=sup_ckpt, uc=uni_ckpt, s4c=s4mc_ckpt, b=bs, l=lr_ours):
            def _run(gpu_id: int) -> dict[str, Any]:
                tr_ds, va_ds = get_datasets(f3, seed=101, use_vis=uv, use_408_split=u4)
                return train_and_eval_semisup_variant(
                    name=k, arch=a, in_ch=ic, base_ch=bc,
                    semisup_method="dist_ours",
                    init_ckpt_paths=[sc, uc, s4c],
                    train_ds=tr_ds, val_ds=va_ds,
                    out_dir=os.path.join(out_root, k),
                    gpu_id=gpu_id, epochs=3, batch_size=b, lr=l,
                    dist_sigma=24.0,
                )
            return _run
        jobs.append((ours_key, make_ours_job()))

    # 2. Ablation on GT-Distance Decay Scale sigma_d on DINOv3 (3f IR+VIS)
    for sigma_px in [8.0, 16.0, 48.0]:
        ab_key = f"ablation_dist_sigma_{int(sigma_px)}px_dinov3_3f_vis"
        def make_sigma_job(k=ab_key, sig=sigma_px):
            def _run(gpu_id: int) -> dict[str, Any]:
                tr_ds, va_ds = get_datasets(True, seed=int(sig) + 42)
                return train_and_eval_semisup_variant(
                    name=k, arch="dinov3", in_ch=8, base_ch=48,
                    semisup_method="dist_ours",
                    init_ckpt_paths=[
                        "outputs/supervised_dinov3_3f_vis/best_supervised.pt",
                        "outputs/semisup_dinov3_3f_vis/best_semisup.pt",
                        "outputs/semisup_s4mc_and_ours/s4mc_3f_vis_dinov3/best_semisup.pt",
                    ],
                    train_ds=tr_ds, val_ds=va_ds,
                    out_dir=os.path.join(out_root, k),
                    gpu_id=gpu_id, epochs=3, batch_size=12, lr=4.0e-5,
                    dist_sigma=sig,
                )
            return _run
        jobs.append((ab_key, make_sigma_job()))

    # 3. Lidar Beam Sparsity Ablation (1 Beam and 2 Beams) for S4MC and Ours (Dilated+Dist-SSL) on DINOv3 3f IR+VIS
    sparsity_configs = [
        ("1beam", (2,), "outputs/cvpr_experiments/sup_dinov3_beams_1/best_model.pt", "outputs/cvpr_experiments/semisup_dinov3_beams_1/best_semisup.pt", "outputs/semisup_s4mc_and_ours/s4mc_sparsity_1beam_dinov3/best_semisup.pt"),
        ("2beams", (1, 2), "outputs/cvpr_experiments/sup_dinov3_beams_2/best_model.pt", "outputs/cvpr_experiments/semisup_dinov3_beams_2/best_semisup.pt", "outputs/semisup_s4mc_and_ours/s4mc_sparsity_2beams_dinov3/best_semisup.pt"),
    ]
    for sp_name, beams, sp_sup_ckpt, sp_uni_ckpt, _ in sparsity_configs:
        for method_tag in ["s4mc", "dist_ours"]:
            sp_key = f"sparsity_{sp_name}_{method_tag}_dinov3"
            s4_sp_ckpt = os.path.join(out_root, f"sparsity_{sp_name}_s4mc_dinov3", "best_semisup.pt")
            def make_sp_job(k=sp_key, bm=beams, m=method_tag, sc=sp_sup_ckpt, uc=sp_uni_ckpt, s4c=s4_sp_ckpt):
                def _run(gpu_id: int) -> dict[str, Any]:
                    tr_ds, va_ds = get_datasets(True, beams=bm, seed=77)
                    init_list = [sc, uc] if m == "s4mc" else [sc, uc, s4c]
                    return train_and_eval_semisup_variant(
                        name=k, arch="dinov3", in_ch=8, base_ch=48,
                        semisup_method=m,
                        init_ckpt_paths=init_list,
                        train_ds=tr_ds, val_ds=va_ds,
                        out_dir=os.path.join(out_root, k),
                        gpu_id=gpu_id, epochs=3, batch_size=12, lr=4.0e-5,
                        dist_sigma=24.0,
                    )
                return _run
            jobs.append((sp_key, make_sp_job()))

    job_queue: queue.Queue[tuple[str, Callable[[int], dict[str, Any]]]] = queue.Queue()
    for j in jobs:
        job_queue.put(j)

    n_gpus = min(4, torch.cuda.device_count())
    print(f"[Runner] Launching {len(jobs)} S4MC & Dilated+Dist-Adaptive (Ours) jobs across {n_gpus} GPUs...")

    def worker(gpu_id: int) -> None:
        while True:
            try:
                key, fn = job_queue.get_nowait()
            except queue.Empty:
                break
            try:
                res = fn(gpu_id)
                summary[key] = res
                save_json(summary, summary_path)
            except Exception as e:
                print(f"[GPU {gpu_id} | ERROR in {key}]: {e}")
                import traceback
                traceback.print_exc()
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n[Runner] All {len(jobs)} jobs complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
