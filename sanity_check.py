from __future__ import annotations

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from dataset import (
    ALT_BBOX_44,
    DEFAULT_BBOX,
    FullImageCloudDataset,
    build_2d_cloud_map,
    discover_matched_granules,
    extract_granule_tracks,
    load_ir_image,
)


def run_sanity_check(
    max_granules: int = 120,
    out_dir: str = "outputs/sanity_check",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    granules = discover_matched_granules()
    print(f"[Sanity Check] Discovered {len(granules)} unique matched granules.")

    # 1. Verify 2D Map Construction at full 4096 x 4096 resolution
    sample_ds = FullImageCloudDataset(granules[:1])
    sample_item = sample_ds[0]
    img_shape = tuple(sample_item["image"].shape)
    valid_px = int(sample_item["valid_mask"].sum().item())
    cloudy_px = int((sample_item["valid_mask"] * sample_item["cloud_mask"]).sum().item())
    print(
        f"[2D Map Check] Granule {sample_item['granule_id']} -> Image shape: {img_shape}, "
        f"2D valid_mask shape: {tuple(sample_item['valid_mask'].shape)}, "
        f"Labeled pixels: {valid_px} ({100.0 * valid_px / (4096 * 4096):.4f}% of image), "
        f"Cloudy pixels: {cloudy_px} ({100.0 * cloudy_px / max(valid_px, 1):.1f}% of labeled)"
    )

    # Save visual overlay of 2D map on full-resolution image & zoomed crop
    plot_sample_2d_map(granules[0], os.path.join(out_dir, "sample_2d_map_overlay.png"))

    # 2. Collect pixel-to-cloud correlation statistics across evenly spaced granules
    step = max(len(granules) // max_granules, 1)
    sampled_granules = granules[::step][:max_granules]

    ir_exact_43, ir_smooth_43, ir_exact_44 = [], [], []
    has_cloud_c1, top_c1, bot_c1, bot_low_c1 = [], [], [], []
    has_any_layer, top_any, bot_any = [], [], []
    cf_asr_all, lat_all = [], []

    for g in tqdm(sampled_granules, desc="Evaluating pixel-cloud correlation"):
        gray = load_ir_image(g.img_path)
        H, W = gray.shape

        tr43 = extract_granule_tracks(g.h5_path, bbox=DEFAULT_BBOX, img_shape=(H, W), cloud_attr_mode="cloud_only")
        tr44 = extract_granule_tracks(g.h5_path, bbox=ALT_BBOX_44, img_shape=(H, W), cloud_attr_mode="cloud_only")
        tr_any = extract_granule_tracks(g.h5_path, bbox=DEFAULT_BBOX, img_shape=(H, W), cloud_attr_mode="all_layers")

        if len(tr43["row"]) == 0 or len(tr43["row"]) != len(tr44["row"]):
            continue

        r43, c43 = tr43["row"], tr43["col"]
        r44, c44 = tr44["row"], tr44["col"]

        ir_exact_43.append(gray[r43, c43].astype(np.float32))
        ir_exact_44.append(gray[r44, c44].astype(np.float32))

        # Local 5x5 neighborhood average around (r43, c43)
        acc = np.zeros(len(r43), dtype=np.float32)
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                acc += gray[np.clip(r43 + dr, 0, H - 1), np.clip(c43 + dc, 0, W - 1)]
        ir_smooth_43.append(acc / 25.0)

        has_cloud_c1.append(tr43["has_cloud"])
        top_c1.append(tr43["cloud_top_km"])
        bot_c1.append(tr43["cloud_bot_km"])
        bot_low_c1.append(tr43["cloud_bot_lowest_km"])

        has_any_layer.append(tr_any["has_cloud"])
        top_any.append(tr_any["cloud_top_km"])
        bot_any.append(tr_any["cloud_bot_km"])

        cf_asr_all.append(tr43["cloud_flag_asr"].astype(np.float32))
        lat_all.append(tr43["lat"])

    ir43 = np.concatenate(ir_exact_43)
    ir43_sm = np.concatenate(ir_smooth_43)
    ir44 = np.concatenate(ir_exact_44)
    hc1 = np.concatenate(has_cloud_c1)
    tc1 = np.concatenate(top_c1)
    bc1 = np.concatenate(bot_c1)
    bc1_low = np.concatenate(bot_low_c1)
    hany = np.concatenate(has_any_layer)
    tany = np.concatenate(top_any)
    bany = np.concatenate(bot_any)
    cf_asr = np.concatenate(cf_asr_all)
    lats = np.concatenate(lat_all)

    vc1 = np.isfinite(tc1) & np.isfinite(bc1)
    vany = np.isfinite(tany) & np.isfinite(bany)

    results = {
        "n_granules_sampled": len(sampled_granules),
        "n_profile_points": int(len(ir43)),
        "cloud_prevalence_attr1": float(hc1.mean()),
        "any_layer_prevalence": float(hany.mean()),
        "mean_ir_clear_attr1": float(ir43[hc1 == 0].mean()),
        "std_ir_clear_attr1": float(ir43[hc1 == 0].std()),
        "mean_ir_cloudy_attr1": float(ir43[hc1 == 1].mean()),
        "std_ir_cloudy_attr1": float(ir43[hc1 == 1].std()),
        "bbox43_vs_bbox44": {
            "corr_ir_cloud_bbox43_pearson": float(stats.pearsonr(ir43, hc1)[0]),
            "corr_ir_cloud_bbox44_pearson": float(stats.pearsonr(ir44, hc1)[0]),
            "corr_ir_top_any_bbox43_pearson": float(stats.pearsonr(ir43[vany], tany[vany])[0]),
            "corr_ir_top_any_bbox44_pearson": float(stats.pearsonr(ir44[vany], tany[vany])[0]),
        },
        "correlations_bbox43": {
            "ir_vs_cloud_presence_attr1_pearson": float(stats.pearsonr(ir43, hc1)[0]),
            "ir_vs_cloud_presence_attr1_spearman": float(stats.spearmanr(ir43, hc1)[0]),
            "ir_smooth5x5_vs_cloud_presence_attr1_pearson": float(stats.pearsonr(ir43_sm, hc1)[0]),
            "ir_vs_cloud_top_attr1_pearson": float(stats.pearsonr(ir43[vc1], tc1[vc1])[0]),
            "ir_vs_cloud_top_attr1_spearman": float(stats.spearmanr(ir43[vc1], tc1[vc1])[0]),
            "ir_vs_cloud_base_attr1_pearson": float(stats.pearsonr(ir43[vc1], bc1[vc1])[0]),
            "ir_vs_cloud_base_lowest_attr1_pearson": float(stats.pearsonr(ir43[vc1], bc1_low[vc1])[0]),
            "ir_vs_any_layer_top_pearson": float(stats.pearsonr(ir43[vany], tany[vany])[0]),
            "ir_vs_any_layer_top_spearman": float(stats.spearmanr(ir43[vany], tany[vany])[0]),
            "ir_vs_any_layer_base_pearson": float(stats.pearsonr(ir43[vany], bany[vany])[0]),
            "ir_vs_cloud_flag_asr_pearson": float(stats.pearsonr(ir43, cf_asr)[0]),
        },
    }

    print("\n=== SANITY CHECK CORRELATION SUMMARY ===")
    print(json.dumps(results, indent=2))

    with open(os.path.join(out_dir, "sanity_check_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_correlation_diagnostics(
        ir43=ir43,
        hc1=hc1,
        tc1=tc1,
        bc1=bc1,
        tany=tany,
        results=results,
        out_path=os.path.join(out_dir, "sanity_check_plots.png"),
    )
    return results


def plot_sample_2d_map(granule, out_path: str) -> None:
    gray = load_ir_image(granule.img_path)
    maps = build_2d_cloud_map(granule.h5_path, bbox=DEFAULT_BBOX, img_shape=gray.shape)

    valid_r, valid_c = np.where(maps["valid_mask"] > 0)
    is_cloud = maps["cloud_mask"][valid_r, valid_c] > 0.5
    tops = maps["cloud_top"][valid_r, valid_c]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=140)

    # Panel 1: Full 4096x4096 IR image with 3 ICESat-2 beams overlaid
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].scatter(valid_c[~is_cloud], valid_r[~is_cloud], s=0.4, c="deepskyblue", alpha=0.5, label="Clear (ICESat-2)")
    sc = axes[0].scatter(
        valid_c[is_cloud],
        valid_r[is_cloud],
        s=0.8,
        c=tops[is_cloud],
        cmap="plasma",
        vmin=0,
        vmax=14,
        label="Cloud Top (km)",
    )
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04, label="Cloud Top Height (km)")
    axes[0].set_title(f"Full 4096x4096 IR 10.5 µm + 2D Label Map\n({granule.img_name})")
    axes[0].legend(loc="upper right", markerscale=8)

    # Find a cloudy region along the track for zoomed-in 256x256 inspection
    if np.any(is_cloud):
        cloudy_indices = np.where(is_cloud)[0]
        mid_idx = cloudy_indices[len(cloudy_indices) // 2]
        cr, cc = int(valid_r[mid_idx]), int(valid_c[mid_idx])
    else:
        cr, cc = int(valid_r[len(valid_r) // 2]), int(valid_c[len(valid_c) // 2])

    r0 = int(np.clip(cr - 128, 0, gray.shape[0] - 256))
    c0 = int(np.clip(cc - 128, 0, gray.shape[1] - 256))

    crop_ir = gray[r0 : r0 + 256, c0 : c0 + 256]
    crop_valid = maps["valid_mask"][r0 : r0 + 256, c0 : c0 + 256]
    crop_top = maps["cloud_top"][r0 : r0 + 256, c0 : c0 + 256]

    axes[1].imshow(crop_ir, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Zoomed 256x256 Patch IR 10.5 µm\n(Rows {r0}..{r0+256}, Cols {c0}..{c0+256})")

    axes[2].imshow(crop_ir, cmap="gray", vmin=0, vmax=255, alpha=0.55)
    vr, vc = np.where(crop_valid > 0)
    vt = crop_top[vr, vc]
    sc2 = axes[2].scatter(vc, vr, c=vt, cmap="plasma", s=10, vmin=0, vmax=14)
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04, label="Cloud Top Height (km)")
    axes[2].set_title("Zoomed 256x256 3-Beam Sparse Label Map\n(3 Parallel Beams ~7.5 px Apart)")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_diagnostics(
    ir43: np.ndarray,
    hc1: np.ndarray,
    tc1: np.ndarray,
    bc1: np.ndarray,
    tany: np.ndarray,
    results: dict,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=140)

    # 1. Distribution of IR Pixel Values for Clear vs Cloudy
    ax = axes[0, 0]
    clear_ir = ir43[hc1 == 0]
    cloud_ir = ir43[hc1 == 1]
    bins = np.linspace(0, 255, 52)
    ax.hist(clear_ir, bins=bins, density=True, alpha=0.6, color="steelblue", label=f"Clear (mean={clear_ir.mean():.1f})")
    ax.hist(cloud_ir, bins=bins, density=True, alpha=0.6, color="crimson", label=f"Cloudy (mean={cloud_ir.mean():.1f})")
    r_cls = results["correlations_bbox43"]["ir_vs_cloud_presence_attr1_pearson"]
    ax.set_title(f"IR 10.5 µm Brightness: Clear vs. Cloudy (r = {r_cls:.3f})")
    ax.set_xlabel("IR 10.5 µm Pixel Brightness (0=Warm/Dark, 255=Cold/Bright)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)

    # 2. Cloud Probability & Mean Top Height vs IR Brightness Bins
    ax = axes[0, 1]
    bin_edges = np.linspace(10, 250, 17)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cloud_probs = []
    mean_tops = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (ir43 >= lo) & (ir43 < hi)
        cloud_probs.append(hc1[m].mean() if np.any(m) else np.nan)
        m_top = m & np.isfinite(tany)
        mean_tops.append(tany[m_top].mean() if np.any(m_top) else np.nan)

    color1 = "crimson"
    ax.plot(bin_centers, cloud_probs, "o-", color=color1, linewidth=2, label="Cloud Fraction (attr=1)")
    ax.set_xlabel("IR 10.5 µm Pixel Brightness")
    ax.set_ylabel("Empirical Cloud Probability", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    color2 = "darkslateblue"
    ax2.plot(bin_centers, mean_tops, "s--", color=color2, linewidth=2, label="Mean Layer Top (km)")
    ax2.set_ylabel("Mean Layer Top Height (km)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax.set_title("Monotonic Increase of Cloud Fraction & Height with IR Brightness")

    # 3. 2D Density of IR Brightness vs Cloud Top Height (Cloudy pixels)
    ax = axes[1, 0]
    vc1 = np.isfinite(tc1)
    h = ax.hist2d(ir43[vc1], tc1[vc1], bins=[45, 45], cmap="magma", cmin=5)
    plt.colorbar(h[3], ax=ax, label="Shot Count")
    r_top = results["correlations_bbox43"]["ir_vs_cloud_top_attr1_pearson"]
    r_any = results["correlations_bbox43"]["ir_vs_any_layer_top_pearson"]
    ax.set_title(f"IR Brightness vs. Cloud Top Height\n(Cloud-only r = {r_top:.3f}, All-layer r = {r_any:.3f})")
    ax.set_xlabel("IR 10.5 µm Pixel Brightness")
    ax.set_ylabel("ICESat-2 Cloud Top Height (km)")
    ax.grid(alpha=0.25)

    # 4. Cloud Top vs Cloud Base Height colored by IR Brightness
    ax = axes[1, 1]
    sub = np.where(vc1)[0]
    if len(sub) > 15000:
        rng = np.random.default_rng(42)
        sub = rng.choice(sub, size=15000, replace=False)
    sc = ax.scatter(tc1[sub], bc1[sub], c=ir43[sub], cmap="viridis", s=4, alpha=0.5, vmin=20, vmax=220)
    plt.colorbar(sc, ax=ax, label="IR 10.5 µm Brightness")
    ax.plot([0, 15], [0, 15], "r--", linewidth=1, label="Top = Base")
    r_bot = results["correlations_bbox43"]["ir_vs_cloud_base_attr1_pearson"]
    ax.set_title(f"Cloud Top vs. Cloud Base Height (Base-IR r = {r_bot:.3f})")
    ax.set_xlabel("Cloud Top Height (km)")
    ax.set_ylabel("Cloud Base Height (km)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_sanity_check()
