from __future__ import annotations

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

from dataset import (
    DEFAULT_BBOX,
    DEFAULT_IMAGE_SHAPE,
    MAX_HEIGHT_KM,
    CloudPatchDataset,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import THRESHOLD_15KFT_KM
from models import build_model, unwrap_model


def make_rgb_satellite_underlay(ir0: np.ndarray, vis0: np.ndarray | None = None) -> np.ndarray:
    """
    Create a crisp, high-contrast 3-channel satellite background in [0, 1] suitable for
    alpha-blending colored cloud top height predictions on top.
    Uses enhanced grayscale / subtle false-color tint so the colored height overlay pops cleanly.
    """
    ir_norm = np.clip((ir0 - np.percentile(ir0, 2)) / max(np.percentile(ir0, 98) - np.percentile(ir0, 2), 1e-4), 0.0, 1.0)
    if vis0 is not None:
        vis_norm = np.clip((vis0 - np.percentile(vis0, 2)) / max(np.percentile(vis0, 98) - np.percentile(vis0, 2), 1e-4), 0.0, 1.0)
        base_lum = 0.55 * vis_norm + 0.45 * ir_norm
    else:
        base_lum = ir_norm

    # Apply gentle gamma and map to a dark-ocean / slate-terrain tones in [0.10, 0.82]
    # so both low-cloud (blue/cyan/green) and high-cloud (yellow/orange/red) alpha overlays are vivid
    lum = np.power(base_lum, 0.90)
    rgb = np.stack(
        [
            0.12 + 0.68 * lum,
            0.15 + 0.68 * lum,
            0.20 + 0.66 * lum,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def blend_cloud_top_over_satellite(
    sat_rgb: np.ndarray,
    cloud_prob: np.ndarray,
    pred_top_km: np.ndarray,
    cmap_name: str = "turbo",
    prob_thresh: float = 0.35,
    max_alpha: float = 0.62,
    vmin_km: float = 0.5,
    vmax_km: float = 13.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alpha-blends predicted cloud top height (colored by `cmap_name`) onto `sat_rgb`
    where `cloud_prob >= prob_thresh`, with smooth opacity ramp:
        alpha(r, c) = max_alpha * clip((cloud_prob - prob_thresh) / (1.0 - prob_thresh), 0, 1)
    Returns:
        blended_rgb: (H, W, 3) float array in [0, 1]
        alpha_map: (H, W) float array in [0, max_alpha]
    """
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin_km, vmax=vmax_km, clip=True)
    height_rgba = cmap(norm(pred_top_km))  # (H, W, 4)
    height_rgb = height_rgba[..., :3]

    # Smooth opacity mask modulated by predicted cloud probability
    conf_ramp = np.clip((cloud_prob - prob_thresh) / max(1.0 - prob_thresh, 1e-4), 0.0, 1.0)
    # Soft sigmoid-like transition at cloud boundaries for natural visual blending
    alpha_map = max_alpha * (conf_ramp ** 0.75) * (cloud_prob >= prob_thresh).astype(np.float32)

    alpha_3d = alpha_map[..., None]
    blended_rgb = (1.0 - alpha_3d) * sat_rgb + alpha_3d * height_rgb
    return np.clip(blended_rgb, 0.0, 1.0), alpha_map


@torch.no_grad()
def main() -> None:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        for g in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(g)
            if free_mem > 1800 * 1024 * 1024:
                device = torch.device(f"cuda:{g}")
                break
    granules_all = discover_matched_granules()
    granules_408 = filter_granules_with_30min_window(granules_all, ir_dir="../ir10.5", vis_dir="../vis0.6", use_vis06=True)
    _, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)

    val_ds = CloudPatchDataset(
        val_408,
        patch_size=256,
        patches_per_granule=6,
        is_train=False,
        seed=42,
        max_cached_images=16,
        use_3frames=True,
        use_vis06=True,
    )

    # Load our best Distance-Guided Semi-Supervised DINOv3 (Ours) checkpoint (fallback to semisup_dinov3_3f_vis)
    ckpt_candidates = [
        "outputs/semisup_s4mc_and_ours/ours_dist_3f_vis_dinov3/best_semisup.pt",
        "outputs/semisup_dinov3_3f_vis/best_semisup.pt",
    ]
    ckpt_path = next(p for p in ckpt_candidates if os.path.isfile(p))
    print(f"[AlphaOverlay] Loading model from {ckpt_path} on {device}...")

    model = build_model(arch="dinov3", in_channels=8, base_ch=48, fp_drop_rate=0.5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state")))
    unwrap_model(model).load_state_dict(state)
    model.eval()

    # Pick 3 scenes spanning all three vertical cloud regimes:
    # Scene 1 (idx 172): High/Mid Frontal System (6 - 11 km, green/yellow/orange/red)
    # Scene 2 (idx 192): Mid/Low Broken Cloud Deck (2 - 4.5 km, cyan/blue)
    # Scene 3 (idx 144): Low Marine Stratocumulus (1 - 2.5 km, deep blue/cyan, Track MAE 0.29 km)
    selected_indices = [172, 192, 144]
    print(f"[AlphaOverlay] Selected validation patch indices: {selected_indices}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
    })

    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.0, 1.0, 1.0, 1.18],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.14,
        hspace=0.24,
        left=0.04,
        right=0.92,
        top=0.96,
        bottom=0.06,
    )

    vmin_km, vmax_km = 0.5, 13.5
    cmap_height = "turbo"
    cmap_prob = "Blues"

    scene_labels = [
        "Scene A: Multi-Layer Frontal System (High Anvil + Boundary-Layer Stratus)",
        "Scene B: Broken Mesoscale Cellular Convection & Mid-Level Cloud Deck",
        "Scene C: Coastal Low-Cloud Deck (<= 4.57 km) & Upper-Tropospheric Cirrus",
    ]

    for row_i, ds_idx in enumerate(selected_indices):
        sample = val_ds[ds_idx]
        g_idx = val_ds.valid_granule_indices[ds_idx // val_ds.patches_per_granule]
        granule = val_ds.granules[g_idx]

        img_t = sample["image"].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(img_t)

        prob_map = out["cloud_prob"].float().squeeze().cpu().numpy()
        top_map_km = out["cloud_top"].float().squeeze().cpu().numpy() * MAX_HEIGHT_KM
        bot_map_km = np.minimum(out["cloud_bot"].float().squeeze().cpu().numpy() * MAX_HEIGHT_KM, top_map_km)

        img_np = sample["image"].numpy()
        ir0 = img_np[0]   # t=0 IR 10.5 um
        vis0 = img_np[3]  # t=0 VIS 0.6 um
        sat_rgb = make_rgb_satellite_underlay(ir0, vis0)

        blended_top_rgb, alpha_map = blend_cloud_top_over_satellite(
            sat_rgb=sat_rgb,
            cloud_prob=prob_map,
            pred_top_km=top_map_km,
            cmap_name=cmap_height,
            prob_thresh=0.35,
            max_alpha=0.62,
            vmin_km=vmin_km,
            vmax_km=vmax_km,
        )

        vm = sample["valid_mask"].numpy() > 0.5
        cm = sample["cloud_mask"].numpy() > 0.5
        gt_top_km = sample["cloud_top"].numpy() * MAX_HEIGHT_KM
        gt_bot_km = sample["cloud_bot"].numpy() * MAX_HEIGHT_KM

        # Col 0: Underlying Satellite Image (VIS 0.6 + IR 10.5) + Sparse ICESat-2 Track
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(sat_rgb, interpolation="bilinear")
        wy, wx = np.where(vm & (~cm))
        cy, cx = np.where(vm & cm)
        if len(wx) > 0:
            ax0.scatter(wx, wy, c="#00e5ff", s=2.2, alpha=0.75, label="ICESat-2 Clear", rasterized=True)
        if len(cx) > 0:
            ax0.scatter(cx, cy, c=gt_top_km[cy, cx], cmap=cmap_height, vmin=vmin_km, vmax=vmax_km, s=5.0, edgecolors="none", label="ICESat-2 GT Top", rasterized=True)
        if row_i == 0:
            ax0.set_title("(a) Satellite Image + Sparse GT Track\n(MTG-I1 VIS 0.6 + IR 10.5 $\\mu$m, $0.075\\%$ Lidar)", fontweight="bold")
        import datetime
        ts_human = datetime.datetime.fromtimestamp(float(granule.timestamp_utc), tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ax0.set_ylabel(f"Scene {row_i + 1}\n({ts_human})", fontweight="bold", fontsize=8.5)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: Predicted Cloud Probability Alpha Overlay over Satellite Image
        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(sat_rgb, interpolation="bilinear")
        # Show predicted cloud mask probability as a clean translucent cyan/white veil over satellite
        cloud_veil = np.zeros((256, 256, 4), dtype=np.float32)
        cloud_veil[..., 0] = 0.15
        cloud_veil[..., 1] = 0.88
        cloud_veil[..., 2] = 1.00
        cloud_veil[..., 3] = 0.58 * np.clip((prob_map - 0.30) / 0.70, 0.0, 1.0)
        ax1.imshow(cloud_veil, interpolation="bilinear")
        ax1.contour(prob_map, levels=[0.50], colors=["#ffffff"], linewidths=0.85, alpha=0.9)
        if row_i == 0:
            ax1.set_title("(b) Predicted Cloud Mask Overlay ($\\alpha$-Blended)\n($\\hat{p}_{\\mathrm{cloud}}$ with $p=0.5$ Boundary Contour)", fontweight="bold")
        cov_pct = 100.0 * float((prob_map >= 0.5).mean())
        ax1.text(
            0.03, 0.91, f"Pred Cloud Cover: {cov_pct:.1f}%",
            transform=ax1.transAxes, color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.68, edgecolor="none"),
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Col 2: Predicted Cloud + Cloud Top Height (km) Alpha-Overlaid on Satellite Image
        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(blended_top_rgb, interpolation="bilinear")
        # Draw contour separating Low Clouds (<= 4.57 km) vs High Clouds (> 4.57 km) inside cloudy regions
        try:
            ax2.contour(
                np.where(prob_map >= 0.45, top_map_km, 0.0),
                levels=[THRESHOLD_15KFT_KM],
                colors=["#ffffff"],
                linewidths=0.9,
                linestyles="--",
                alpha=0.85,
            )
        except Exception:
            pass
        # Overlay sparse GT lidar cloudy points with black-rimmed markers colored by true height
        if len(cx) > 0:
            ax2.scatter(
                cx[::3], cy[::3],
                c=gt_top_km[cy[::3], cx[::3]],
                cmap=cmap_height, vmin=vmin_km, vmax=vmax_km,
                s=14.0, edgecolors="black", linewidths=0.45, zorder=5,
                label="GT Lidar Height", rasterized=True,
            )
        if row_i == 0:
            ax2.set_title("(c) Predicted Cloud Top Height $\\hat{h}_{\\mathrm{top}}$ ($\\alpha=0.62\\,\\hat{p}$)\nOver Satellite Image + GT Lidar Track", fontweight="bold")
        mean_top = float(top_map_km[prob_map >= 0.5].mean()) if (prob_map >= 0.5).any() else 0.0
        mae_patch = float(np.mean(np.abs(top_map_km[cy, cx] - gt_top_km[cy, cx]))) if len(cx) > 0 else 0.0
        ax2.text(
            0.03, 0.91,
            f"Mean $\\hat{{h}}_{{\\mathrm{{top}}}}$: {mean_top:.2f} km | Track MAE: {mae_patch:.2f} km",
            transform=ax2.transAxes, color="white", fontsize=7.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.68, edgecolor="none"),
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Col 3: Along-Track Vertical Cross-Section (Predicted Top & Base Envelope vs ICESat-2 GT)
        ax3 = fig.add_subplot(gs[row_i, 3])
        # Sort valid track pixels along the primary orbit direction (row coordinate)
        vy, vx = np.where(vm)
        order = np.argsort(vy)
        vy, vx = vy[order], vx[order]
        dist_km = np.linspace(0.0, 256 * 0.445, len(vy))

        gt_cld = cm[vy, vx]
        gt_t = np.where(gt_cld, gt_top_km[vy, vx], np.nan)
        gt_b = np.where(gt_cld, gt_bot_km[vy, vx], np.nan)

        pr_p = prob_map[vy, vx]
        pr_t = np.where(pr_p >= 0.35, top_map_km[vy, vx], np.nan)
        pr_b = np.where(pr_p >= 0.35, bot_map_km[vy, vx], np.nan)

        ax3.fill_between(dist_km, gt_b, gt_t, color="#90a4ae", alpha=0.42, label="ICESat-2 GT Cloud Slab")
        ax3.plot(dist_km, gt_t, "k.", markersize=2.8, alpha=0.80, label="GT Top ($h_{\\mathrm{top}}$)")
        ax3.plot(dist_km, gt_b, color="#37474f", linestyle="none", marker=".", markersize=2.2, alpha=0.65, label="GT Base ($h_{\\mathrm{bot}}$)")

        ax3.fill_between(dist_km, pr_b, pr_t, color="#ff7043", alpha=0.30, label="Pred Cloud Slab (Ours)")
        ax3.plot(dist_km, pr_t, color="#d32f2f", linewidth=1.6, label="Pred Top ($\\hat{h}_{\\mathrm{top}}$)")
        ax3.plot(dist_km, pr_b, color="#1976d2", linewidth=1.4, linestyle="-", label="Pred Base ($\\hat{h}_{\\mathrm{bot}}$)")

        ax3.axhline(THRESHOLD_15KFT_KM, color="#2e7d32", linestyle="--", linewidth=0.9, alpha=0.8, label="Low/High Split (4.57 km)")
        ax3.set_ylim(0.0, 14.5)
        ax3.set_xlim(0.0, max(float(dist_km[-1]), 1.0))
        ax3.set_ylabel("Altitude (km)", fontsize=8)
        if row_i == 2:
            ax3.set_xlabel("Along-Track Patch Distance (km)", fontsize=8.5)
        if row_i == 0:
            ax3.set_title("(d) Along-Track 3D Cloud Envelope\n(Pred $\\hat{h}_{\\mathrm{top}}, \\hat{h}_{\\mathrm{bot}}$ vs. ICESat-2 Lidar)", fontweight="bold")
            ax3.legend(loc="upper right", fontsize=6.5, ncol=2, framealpha=0.85)
        ax3.grid(True, linestyle=":", alpha=0.45)

    # Shared Colorbar for Cloud Top Height (km)
    cbar_ax = fig.add_axes([0.935, 0.12, 0.014, 0.74])
    sm = ScalarMappable(norm=Normalize(vmin=vmin_km, vmax=vmax_km), cmap=cmap_height)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cloud Top Height $\\hat{h}_{\\mathrm{top}}$ (km)  [Alpha $\\alpha = 0.62 \\cdot \\hat{p}_{\\mathrm{cloud}}$]", fontweight="bold", fontsize=9.5, labelpad=10)
    cbar.ax.axhline(THRESHOLD_15KFT_KM, color="white", linestyle="--", linewidth=1.4)
    cbar.ax.text(
        0.5, THRESHOLD_15KFT_KM, "4.57 km",
        transform=cbar.ax.get_yaxis_transform(),
        va="center", ha="center", fontsize=6.5, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.72, edgecolor="white", linewidth=0.5),
    )

    os.makedirs("paper/figures", exist_ok=True)
    pdf_path = "paper/figures/fig_alpha_cloud_top_overlay.pdf"
    png_path = "paper/figures/fig_alpha_cloud_top_overlay.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[AlphaOverlay] Saved {pdf_path} and {png_path}!")


if __name__ == "__main__":
    main()
