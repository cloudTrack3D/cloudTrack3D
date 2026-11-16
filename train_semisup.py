from __future__ import annotations

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentations import (
    apply_cutmix_pair,
    apply_strong_photometric_augmentation,
    apply_weak_augmentation,
)
from dataset import (
    CloudPatchDataset,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import (
    DistanceGuidedConsistencyLoss,
    S4MCConsistencyLoss,
    SemiSupervisedConsistencyLoss,
    SupervisedCloudLoss,
    evaluate_sparse_predictions,
)
from models import EMATeacher, build_model, extract_fp_dict, get_optimizer_param_groups, unwrap_model
from train_supervised import plot_qualitative_predictions


def cosine_rampup(current: int, rampup_length: int) -> float:
    if rampup_length == 0:
        return 1.0
    current = max(0, min(current, rampup_length))
    return float(0.5 * (1.0 - math.cos(math.pi * current / rampup_length)))


def set_batchnorm_eval(model: nn.Module, is_eval: bool = True) -> None:
    """Freeze BatchNorm running stats during strong photometric augmentation streams."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if is_eval:
                m.eval()
            else:
                m.train()



def train_semisupervised(args: argparse.Namespace) -> dict:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if (device.type == "cuda" and args.multi_gpu) else 1
    print(f"[Semi-Supervised ({args.arch.upper()} + UniMatch + EMA)] Primary device: {device} | Active GPUs: {n_gpus}")

    granules = discover_matched_granules()
    if args.use_3frames or args.use_vis06:
        granules = filter_granules_with_30min_window(
            granules,
            ir_dir="../ir10.5",
            vis_dir="../vis0.6",
            use_vis06=args.use_vis06,
        )
        print(f"[Semi-Supervised ({args.arch.upper()})] Granules with complete 30-min (0, -15, -30m) window: {len(granules)}")
    if args.max_granules > 0:
        granules = granules[: args.max_granules]

    train_granules, val_granules = split_granules_chronologically(granules, val_ratio=0.2)
    print(f"[Semi-Supervised ({args.arch.upper()})] Train granules: {len(train_granules)}, Val granules: {len(val_granules)}")

    train_ds = CloudPatchDataset(
        train_granules,
        patch_size=args.patch_size,
        patches_per_granule=args.patches_per_granule,
        is_train=True,
        seed=args.seed,
        max_cached_images=3000,
        use_3frames=args.use_3frames,
        use_vis06=args.use_vis06,
    )
    val_ds = CloudPatchDataset(
        val_granules,
        patch_size=args.patch_size,
        patches_per_granule=args.val_patches_per_granule,
        is_train=False,
        seed=args.seed,
        max_cached_images=1000,
        use_3frames=args.use_3frames,
        use_vis06=args.use_vis06,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    in_ch = train_ds.in_channels
    student = build_model(arch=args.arch, in_channels=in_ch, base_ch=args.base_ch, fp_drop_rate=0.5).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Semi-Supervised ({args.arch.upper()})] Input channels: {in_ch} | Trainable parameters: {n_params:,}")

    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        print(f"[Semi-Supervised] Initializing Student & EMA Teacher from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])

    ema_teacher = EMATeacher(student, decay=args.ema_decay).to(device)

    if n_gpus > 1:
        student = nn.DataParallel(student, device_ids=list(range(n_gpus)))
        teacher_dp = nn.DataParallel(ema_teacher.model, device_ids=list(range(n_gpus)))
    else:
        teacher_dp = ema_teacher.model

    sup_criterion = SupervisedCloudLoss()
    semisup_method = getattr(args, "semisup_method", "unimatch")
    if semisup_method == "s4mc":
        cons_criterion = S4MCConsistencyLoss(
            conf_thresh_cls=getattr(args, "s4mc_thresh_cls", 0.85),
            conf_thresh_reg=getattr(args, "s4mc_thresh_reg", 0.78),
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)
    elif semisup_method == "dist_ours":
        cons_criterion = DistanceGuidedConsistencyLoss(
            tau_cls_min=getattr(args, "tau_cls_min", 0.65),
            tau_cls_max=getattr(args, "tau_cls_max", 0.88),
            tau_reg_min=getattr(args, "tau_reg_min", 0.55),
            tau_reg_max=getattr(args, "tau_reg_max", 0.82),
            dist_sigma=getattr(args, "dist_sigma", 24.0),
            prox_alpha=getattr(args, "prox_alpha", 0.50),
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)
    else:
        cons_criterion = SemiSupervisedConsistencyLoss(
            conf_thresh_cls=args.conf_thresh_cls,
            conf_thresh_reg=args.conf_thresh_reg,
            cls_cons_weight=1.0,
            reg_cons_weight=8.0,
        ).to(device)

    param_groups = get_optimizer_param_groups(student, lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_score = -1e9
    best_metrics = {}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        teacher_dp.eval()
        ramp = cosine_rampup(epoch, args.rampup_epochs)
        cons_weight = args.cons_weight * ramp

        run_total, run_sup, run_cons, run_cls_mask, run_reg_mask = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # 1. Weak augmentation applied identically to image & sparse labels
            batch_w = apply_weak_augmentation(batch)
            x_w = batch_w["image"]
            valid_w = batch_w["valid_mask"]

            # 2. EMA Teacher produces dense pseudo-labels on weakly augmented view
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                teacher_preds = teacher_dp(x_w, need_fp=False)
                teacher_preds = {k: v.float() for k, v in teacher_preds.items()}

            # 3. Create two strongly augmented views (photometric jitter + spatial CutMix)
            x_s1_raw = apply_strong_photometric_augmentation(x_w)
            x_s2_raw = apply_strong_photometric_augmentation(x_w)
            x_s1, t_preds_s1, valid_s1 = apply_cutmix_pair(x_s1_raw, teacher_preds, valid_w, p=0.5)
            x_s2, t_preds_s2, valid_s2 = apply_cutmix_pair(x_s2_raw, teacher_preds, valid_w, p=0.5)

            optimizer.zero_grad(set_to_none=True)

            # Stream 1: Student forward on weak view + UniMatch feature perturbation (fp)
            # Backpropagate immediately so autograd activations are freed before strong streams
            set_batchnorm_eval(student, is_eval=False)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out_w_fp = student(x_w, need_fp=True)
                preds_w, preds_fp = extract_fp_dict(out_w_fp)
                sup_dict = sup_criterion(preds_w, batch_w)
                loss_sup = sup_dict["loss"]
                cons_fp_dict = cons_criterion(preds_fp, teacher_preds, valid_w)
                loss_stream1 = loss_sup + cons_weight * 0.5 * cons_fp_dict["loss_cons"]
            scaler.scale(loss_stream1).backward()
            del out_w_fp, preds_w, preds_fp

            # Freeze BatchNorm running stats on strongly jittered views so running mean/var stay clean
            set_batchnorm_eval(student, is_eval=True)

            # Stream 2: Student forward on Strong View 1 (photometric + CutMix)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds_s1 = student(x_s1, need_fp=False)
                cons_s1_dict = cons_criterion(preds_s1, t_preds_s1, valid_s1)
                loss_stream2 = cons_weight * 0.25 * cons_s1_dict["loss_cons"]
            scaler.scale(loss_stream2).backward()
            del preds_s1

            # Stream 3: Student forward on Strong View 2 (photometric + CutMix)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds_s2 = student(x_s2, need_fp=False)
                cons_s2_dict = cons_criterion(preds_s2, t_preds_s2, valid_s2)
                loss_stream3 = cons_weight * 0.25 * cons_s2_dict["loss_cons"]
            scaler.scale(loss_stream3).backward()
            del preds_s2
            set_batchnorm_eval(student, is_eval=False)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_cons_val = float(
                (
                    0.5 * cons_fp_dict["loss_cons"]
                    + 0.25 * cons_s1_dict["loss_cons"]
                    + 0.25 * cons_s2_dict["loss_cons"]
                ).item()
            )
            loss_sup_val = float(loss_sup.item())
            loss_total_val = loss_sup_val + cons_weight * loss_cons_val

            global_step += 1
            ema_teacher.update(student, step=global_step if not args.pretrained_ckpt else None)

            run_total += loss_total_val
            run_sup += loss_sup_val
            run_cons += loss_cons_val
            run_cls_mask += float(cons_fp_dict["cls_mask_ratio"].item())
            run_reg_mask += float(cons_fp_dict["reg_mask_ratio"].item())
            n_batches += 1

        scheduler.step()

        student_metrics = evaluate_sparse_predictions(student, val_loader, device)
        teacher_metrics = evaluate_sparse_predictions(teacher_dp, val_loader, device)

        s_score = (
            student_metrics["cloud_f1"]
            + student_metrics["cloud_auroc"]
            - 0.15 * (student_metrics["top_mae_km"] + student_metrics["bot_mae_km"])
        )
        t_score = (
            teacher_metrics["cloud_f1"]
            + teacher_metrics["cloud_auroc"]
            - 0.15 * (teacher_metrics["top_mae_km"] + teacher_metrics["bot_mae_km"])
        )

        eval_metrics = teacher_metrics if t_score >= s_score else student_metrics
        eval_source = "EMA_Teacher" if t_score >= s_score else "Student"
        epoch_score = max(t_score, s_score)

        record = {
            "epoch": epoch,
            "arch": args.arch,
            "n_params": n_params,
            "eval_source": eval_source,
            "train_loss_total": run_total / max(n_batches, 1),
            "train_loss_sup": run_sup / max(n_batches, 1),
            "train_loss_cons": run_cons / max(n_batches, 1),
            "cons_weight": cons_weight,
            "unlabeled_cls_mask_ratio": run_cls_mask / max(n_batches, 1),
            "unlabeled_reg_mask_ratio": run_reg_mask / max(n_batches, 1),
            "elapsed_sec": round(time.time() - t0, 2),
            "student_val": student_metrics,
            "teacher_val": teacher_metrics,
            **eval_metrics,
        }
        history.append(record)

        print(
            f"[SemiSup-{args.arch.upper()} Ep {epoch:02d}/{args.epochs:02d} ({eval_source})] "
            f"sup={record['train_loss_sup']:.4f} cons={record['train_loss_cons']:.4f} (mask={record['unlabeled_cls_mask_ratio']:.2f}) | "
            f"F1={eval_metrics['cloud_f1']:.4f} AUROC={eval_metrics['cloud_auroc']:.4f} | "
            f"Top MAE={eval_metrics['top_mae_km']:.3f}km (R2={eval_metrics['top_r2']:.3f}, r={eval_metrics['top_pearson_r']:.3f}) | "
            f"Bot MAE={eval_metrics['bot_mae_km']:.3f}km (R2={eval_metrics['bot_r2']:.3f}, r={eval_metrics['bot_pearson_r']:.3f}) | "
            f"{record['elapsed_sec']:.1f}s"
        )

        if epoch_score > best_score:
            best_score = epoch_score
            best_metrics = record.copy()
            best_model_state = (
                ema_teacher.model.state_dict()
                if t_score >= s_score
                else unwrap_model(student).state_dict()
            )
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "in_channels": in_ch,
                    "eval_source": eval_source,
                    "model_state_dict": best_model_state,
                    "student_state_dict": unwrap_model(student).state_dict(),
                    "teacher_state_dict": ema_teacher.model.state_dict(),
                    "metrics": best_metrics,
                    "args": vars(args),
                },
                os.path.join(args.out_dir, "best_semisup.pt"),
            )

    with open(os.path.join(args.out_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    ckpt = torch.load(os.path.join(args.out_dir, "best_semisup.pt"), map_location=device, weights_only=False)
    unwrap_model(student).load_state_dict(ckpt["model_state_dict"])
    plot_qualitative_predictions(
        model=student,
        val_dataset=val_ds,
        device=device,
        out_path=os.path.join(args.out_dir, "semisup_predictions.png"),
        title_prefix=f"Semi-Supervised {args.arch.upper()} (UniMatch+EMA)",
    )
    summary_clean = {k: v for k, v in best_metrics.items() if not isinstance(v, dict)}
    print(f"\n[Semi-Supervised ({args.arch.upper()})] Best Validation Metrics:\n{json.dumps(summary_clean, indent=2)}")
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-Supervised Cloud Detection & Height Estimation (UniMatch + EMA)")
    parser.add_argument("--arch", type=str, default="segvit", choices=["unet", "segvit", "dinov3", "sam"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-ch", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-granule", type=int, default=6)
    parser.add_argument("--val-patches-per-granule", type=int, default=6)
    parser.add_argument("--max-granules", type=int, default=490)
    parser.add_argument("--use-3frames", action="store_true", default=False)
    parser.add_argument("--use-vis06", action="store_true", default=False)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--cons-weight", type=float, default=0.35)
    parser.add_argument("--rampup-epochs", type=int, default=5)
    parser.add_argument("--conf-thresh-cls", type=float, default=0.80)
    parser.add_argument("--conf-thresh-reg", type=float, default=0.70)
    parser.add_argument("--semisup-method", type=str, default="dist_ours", choices=["unimatch", "s4mc", "dist_ours"])
    parser.add_argument("--s4mc-thresh-cls", type=float, default=0.85)
    parser.add_argument("--s4mc-thresh-reg", type=float, default=0.78)
    parser.add_argument("--tau-cls-min", type=float, default=0.65)
    parser.add_argument("--tau-cls-max", type=float, default=0.88)
    parser.add_argument("--tau-reg-min", type=float, default=0.55)
    parser.add_argument("--tau-reg-max", type=float, default=0.82)
    parser.add_argument("--dist-sigma", type=float, default=24.0)
    parser.add_argument("--prox-alpha", type=float, default=0.50)
    parser.add_argument("--pretrained-ckpt", type=str, default="outputs/supervised_segvit/best_supervised.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--multi-gpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="outputs/semisup_segvit")
    return parser.parse_args()


if __name__ == "__main__":
    train_semisupervised(parse_args())
