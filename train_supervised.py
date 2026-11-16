from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentations import apply_weak_augmentation
from dataset import (
     MAX_HEIGHT_KM,
    CloudPatchDataset,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import SupervisedCloudLoss, evaluate_sparse_predictions
from models import build_model, get_optimizer_param_groups, unwrap_model


def plot_qualitative_predictions(
    model: torch.nn.Module,
    val_dataset: CloudPatchDataset,
    device: torch.device,
    out_path: str,
    title_prefix: str = "Supervised",
) -> None:
    model.eval()
    chosen_samples = []
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        vm = sample["valid_mask"] > 0.5
        cm = sample["cloud_mask"] > 0.5
        if (vm & cm).sum() >= 30 and (vm & ~cm).sum() >= 20:
            chosen_samples.append(sample)
        if len(chosen_samples) >= 3:
            break
    if not chosen_samples:
        chosen_samples = [val_dataset[i] for i in range(min(3, len(val_dataset)))]

    n_rows = len(chosen_samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(21, 4.2 * n_rows), dpi=140)
    if n_rows == 1:
        axes = axes[None, :]

    base_model = unwrap_model(model)
    with torch.no_grad():
        for row_i, sample in enumerate(chosen_samples):
            img_t = sample["image"].unsqueeze(0).to(device)
            out = base_model(img_t)

            ir = sample["image"][0].numpy()
            valid_mask = sample["valid_mask"].numpy() > 0.5
            gt_cloud = sample["cloud_mask"].numpy()
            gt_top_km = sample["cloud_top"].numpy() * MAX_HEIGHT_KM
            gt_bot_km = sample["cloud_bot"].numpy() * MAX_HEIGHT_KM

            pred_prob = out["cloud_prob"][0, 0].cpu().numpy()
            pred_top_km = out["cloud_top"][0, 0].cpu().numpy() * MAX_HEIGHT_KM
            pred_bot_km = np.minimum(out["cloud_bot"][0, 0].cpu().numpy() * MAX_HEIGHT_KM, pred_top_km)

            vr, vc = np.where(valid_mask)
            is_c = gt_cloud[vr, vc] > 0.5

            # 1. Input IR + Sparse ICESat-2 GT
            ax = axes[row_i, 0]
            ax.imshow(ir, cmap="gray", vmin=0, vmax=1)
            if np.any(~is_c):
                ax.scatter(vc[~is_c], vr[~is_c], s=3, c="deepskyblue", alpha=0.7, label="Clear GT")
            if np.any(is_c):
                sc = ax.scatter(
                    vc[is_c], vr[is_c], s=5, c=gt_top_km[vr[is_c], vc[is_c]], cmap="plasma", vmin=0, vmax=14, label="Cloud Top GT"
                )
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="km")
            ax.set_title(f"{title_prefix}: IR 10.5 µm + Sparse GT")
            ax.legend(loc="upper right", fontsize=7, markerscale=2)

            # 2. Dense Predicted Cloud Probability
            ax = axes[row_i, 1]
            im_p = ax.imshow(pred_prob, cmap="RdYlBu_r", vmin=0, vmax=1)
            plt.colorbar(im_p, ax=ax, fraction=0.046, pad=0.04, label="P(Cloud)")
            ax.set_title("Dense Predicted P(Cloud)")

            # 3. Dense Predicted Cloud Top Height (masked by P(Cloud) >= 0.50)
            ax = axes[row_i, 2]
            ax.imshow(ir, cmap="gray", vmin=0, vmax=1, alpha=0.45)
            masked_top = np.where(pred_prob >= 0.50, pred_top_km, np.nan)
            im_t = ax.imshow(masked_top, cmap="plasma", vmin=0, vmax=14)
            plt.colorbar(im_t, ax=ax, fraction=0.046, pad=0.04, label="km")
            ax.set_title("Dense Predicted Cloud Top (km)")

            # 4. Dense Predicted Cloud Base Height
            ax = axes[row_i, 3]
            ax.imshow(ir, cmap="gray", vmin=0, vmax=1, alpha=0.45)
            masked_bot = np.where(pred_prob >= 0.50, pred_bot_km, np.nan)
            im_b = ax.imshow(masked_bot, cmap="viridis", vmin=0, vmax=12)
            plt.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04, label="km")
            ax.set_title("Dense Predicted Cloud Base (km)")

            # 5. Along-Track Vertical Cross Section (GT vs Prediction on Valid Pixels)
            ax = axes[row_i, 4]
            order = np.argsort(vr)
            vr_s, vc_s = vr[order], vc[order]
            x_idx = np.arange(len(vr_s))
            gt_c_s = gt_cloud[vr_s, vc_s] > 0.5
            gt_top_s = np.where(gt_c_s, gt_top_km[vr_s, vc_s], np.nan)
            gt_bot_s = np.where(gt_c_s, gt_bot_km[vr_s, vc_s], np.nan)

            pred_c_s = pred_prob[vr_s, vc_s] >= 0.50
            p_top_s = np.where(pred_c_s, pred_top_km[vr_s, vc_s], np.nan)
            p_bot_s = np.where(pred_c_s, pred_bot_km[vr_s, vc_s], np.nan)

            ax.plot(x_idx, gt_top_s, "r.", markersize=3, label="GT Top")
            ax.plot(x_idx, gt_bot_s, "b.", markersize=3, label="GT Base")
            ax.plot(x_idx, p_top_s, "-", color="darkorange", linewidth=1.5, label="Pred Top")
            ax.plot(x_idx, p_bot_s, "-", color="teal", linewidth=1.5, label="Pred Base")
            ax.set_ylim(0, 15)
            ax.set_xlabel("Along-Track Pixel Index")
            ax.set_ylabel("Altitude (km)")
            ax.set_title("Along-Track Vertical Profile")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_supervised(args: argparse.Namespace) -> dict:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if (device.type == "cuda" and args.multi_gpu) else 1
    print(f"[Supervised ({args.arch.upper()})] Primary device: {device} | Active GPUs: {n_gpus}")

    granules = discover_matched_granules()
    if args.use_3frames or args.use_vis06:
        granules = filter_granules_with_30min_window(
            granules,
            ir_dir="../ir10.5",
            vis_dir="../vis0.6",
            use_vis06=args.use_vis06,
        )
        print(f"[Supervised ({args.arch.upper()})] Granules with complete 30-min (0, -15, -30m) window: {len(granules)}")
    if args.max_granules > 0:
        granules = granules[: args.max_granules]

    train_granules, val_granules = split_granules_chronologically(granules, val_ratio=0.2)
    print(f"[Supervised ({args.arch.upper()})] Train granules: {len(train_granules)}, Val granules: {len(val_granules)}")

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
    model = build_model(arch=args.arch, in_channels=in_ch, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Supervised ({args.arch.upper()})] Input channels: {in_ch} | Trainable parameters: {n_params:,}")

    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))

    criterion = SupervisedCloudLoss()
    param_groups = get_optimizer_param_groups(model, lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_score = -1e9
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_cls, running_reg = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch = apply_weak_augmentation(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(batch["image"], need_fp=False)
                loss_dict = criterion(preds, batch)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss_dict["loss"].item())
            running_cls += float(loss_dict["loss_cls"].item())
            running_reg += float(loss_dict["loss_reg"].item())
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate_sparse_predictions(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["arch"] = args.arch
        val_metrics["n_params"] = n_params
        val_metrics["train_loss"] = running_loss / max(n_batches, 1)
        val_metrics["train_loss_cls"] = running_cls / max(n_batches, 1)
        val_metrics["train_loss_reg"] = running_reg / max(n_batches, 1)
        val_metrics["elapsed_sec"] = round(time.time() - t0, 2)
        history.append(val_metrics)

        score = (
            val_metrics["cloud_f1"]
            + val_metrics["cloud_auroc"]
            - 0.15 * (val_metrics["top_mae_km"] + val_metrics["bot_mae_km"])
        )
        print(
            f"[Sup-{args.arch.upper()} Ep {epoch:02d}/{args.epochs:02d}] "
            f"loss={val_metrics['train_loss']:.4f} | "
            f"F1={val_metrics['cloud_f1']:.4f} AUROC={val_metrics['cloud_auroc']:.4f} | "
            f"Top MAE={val_metrics['top_mae_km']:.3f}km (R2={val_metrics['top_r2']:.3f}, r={val_metrics['top_pearson_r']:.3f}) | "
            f"Bot MAE={val_metrics['bot_mae_km']:.3f}km (R2={val_metrics['bot_r2']:.3f}, r={val_metrics['bot_pearson_r']:.3f}) | "
            f"{val_metrics['elapsed_sec']:.1f}s"
        )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "in_channels": in_ch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "metrics": best_metrics,
                    "args": vars(args),
                },
                os.path.join(args.out_dir, "best_supervised.pt"),
            )

    with open(os.path.join(args.out_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    ckpt = torch.load(os.path.join(args.out_dir, "best_supervised.pt"), map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    plot_qualitative_predictions(
        model=model,
        val_dataset=val_ds,
        device=device,
        out_path=os.path.join(args.out_dir, "supervised_predictions.png"),
        title_prefix=f"Supervised {args.arch.upper()}",
    )
    print(f"\n[Supervised ({args.arch.upper()})] Best Validation Metrics:\n{json.dumps(best_metrics, indent=2)}")
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised Cloud Detection & Height Estimation")
    parser.add_argument("--arch", type=str, default="segvit", choices=["unet", "segvit", "dinov3", "sam"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-ch", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-granule", type=int, default=6)
    parser.add_argument("--val-patches-per-granule", type=int, default=6)
    parser.add_argument("--max-granules", type=int, default=490)
    parser.add_argument("--use-3frames", action="store_true", default=False)
    parser.add_argument("--use-vis06", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--multi-gpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="outputs/supervised_segvit")
    return parser.parse_args()


if __name__ == "__main__":
    train_supervised(parse_args())
