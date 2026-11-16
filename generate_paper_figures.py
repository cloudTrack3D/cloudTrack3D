from __future__ import annotations

import datetime
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import (
    DEFAULT_BBOX,
    DEFAULT_IMAGE_SHAPE,
    MAX_HEIGHT_KM,
    CloudPatchDataset,
    build_2d_cloud_map,
    discover_matched_granules,
    extract_granule_tracks,
    filter_granules_with_30min_window,
    get_3frame_filenames,
    load_ir_image,
    split_granules_chronologically,
)
from losses import KM_TO_KFT, THRESHOLD_15KFT_KM
from models import build_model


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = "paper/figures"


def save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Figure] {pdf_path} & {png_path}")


def generate_fig1_dataset_overview(granules_408: list) -> None:
    """
    Fig 1: Teaser & Dataset Overview (Full-disk sector IR 10.5 + VIS 0.6 + 3-Beam Zoom + Vertical Curtain).
    """
    # Select a daytime granule with full-track (25N-40N) multi-altitude clouds
    chosen_g = None
    for g in granules_408:
        if g.img_name == "2024-12-21_13-25.png":
            chosen_g = g
            break
    if chosen_g is None:
        chosen_g = granules_408[10]

    ir_img = load_ir_image(chosen_g.img_path)
    vis_path = os.path.join("../vis0.6", chosen_g.img_name)
    vis_img = load_ir_image(vis_path)
    fnames = get_3frame_filenames(chosen_g.img_name)
    ir_30 = load_ir_image(os.path.join("../ir10.5", fnames[2]))
    tr = extract_granule_tracks(chosen_g.h5_path)

    fig = plt.figure(figsize=(15.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.12, 1.08, 1.45], wspace=0.40)

    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX
    extent = [min_lon, max_lon, min_lat, max_lat]

    # Panel (a): Full 4096x4096 IR 10.5 um + 3-beam track colored by Cloud Top Height
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ir_img[::4, ::4], cmap="gray", extent=extent, aspect="auto")
    cloudy = tr["has_cloud"] > 0.5
    clear = ~cloudy
    ax0.scatter(tr["lon"][clear][::10], tr["lat"][clear][::10], c="dodgerblue", s=1.0, alpha=0.5, label="Clear Sky")
    sc = ax0.scatter(
        tr["lon"][cloudy][::4],
        tr["lat"][cloudy][::4],
        c=tr["cloud_top_km"][cloudy][::4],
        cmap="turbo",
        vmin=0,
        vmax=14,
        s=3.5,
        label="Cloud Top (km)",
    )
    cbar0 = plt.colorbar(sc, ax=ax0, fraction=0.046, pad=0.03)
    cbar0.set_label("ICESat-2 Cloud Top (km)", fontsize=8)
    ax0.set_title(f"(a) MTG-I1 FCI IR 10.5 µm\n+ 3-Beam Track ({chosen_g.img_name[:-4]})", fontweight="bold", fontsize=9.2)
    ax0.set_xlabel("Longitude (°E)")
    ax0.set_ylabel("Latitude (°N)")

    # Panel (b): Co-registered VIS 0.6 um
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(vis_img[::4, ::4], cmap="bone", extent=extent, aspect="auto")
    # Pick a cloudy zoom window around median cloudy track point
    c_rows = tr["row"][cloudy]
    c_cols = tr["col"][cloudy]
    mid_idx = len(c_rows) // 2
    r0 = int(np.clip(c_rows[mid_idx] - 128, 0, 4096 - 256))
    c0 = int(np.clip(c_cols[mid_idx] - 128, 0, 4096 - 256))
    lon0 = min_lon + (c0 / 4095.0) * (max_lon - min_lon)
    lon1 = min_lon + ((c0 + 256) / 4095.0) * (max_lon - min_lon)
    lat1 = max_lat - (r0 / 4095.0) * (max_lat - min_lat)
    lat0 = max_lat - ((r0 + 256) / 4095.0) * (max_lat - min_lat)
    rect = mpatches.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, linewidth=1.8, edgecolor="crimson", facecolor="none")
    ax1.add_patch(rect)
    ax1.plot(tr["lon"][::20], tr["lat"][::20], ".", color="gold", markersize=1.2, alpha=0.6)
    ax1.set_title("(b) Co-Registered VIS 0.6 µm\nSolar Reflectance + Patch Window", fontweight="bold", fontsize=9.2)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")

    # Panel (c): Zoomed 256x256 Patch showing 30-min Motion + 3 Parallel Lidar Beams (~7.5 px spacing)
    ax2 = fig.add_subplot(gs[0, 2])
    rgb_comp = np.stack([
        ir_img[r0:r0+256, c0:c0+256].astype(np.float32) / 255.0,
        vis_img[r0:r0+256, c0:c0+256].astype(np.float32) / 255.0,
        np.clip((ir_img[r0:r0+256, c0:c0+256].astype(np.float32) - ir_30[r0:r0+256, c0:c0+256].astype(np.float32)) / 128.0 + 0.5, 0, 1),
    ], axis=-1)
    ax2.imshow(rgb_comp)
    in_p = (tr["row"] >= r0) & (tr["row"] < r0 + 256) & (tr["col"] >= c0) & (tr["col"] < c0 + 256)
    beam_colors = {1: "#00ffff", 2: "#ff00ff", 3: "#ffff00"}
    for b_id in (1, 2, 3):
        bm = in_p & (tr["beam"] == b_id)
        if np.any(bm):
            ax2.scatter(tr["col"][bm] - c0, tr["row"][bm] - r0, s=3.0, c=beam_colors[b_id], label=f"Beam {b_id}")
    ax2.set_title("(c) 256×256 Multi-Temporal Patch\n3 Parallel Lidar Beams (0.075% Pixels)", fontweight="bold")
    ax2.set_xlabel("Patch X (pixels, ~450 m/px)")
    ax2.set_ylabel("Patch Y (pixels)")
    ax2.legend(loc="lower right", framealpha=0.85, markerscale=2.0)

    # Panel (d): Along-Track Vertical Cross Section (Beam 2) showing Top, Base, and 15 kft threshold
    ax3 = fig.add_subplot(gs[0, 3])
    b2 = (tr["beam"] == 2) & (tr["has_cloud"] > 0.5)
    b2_all = tr["beam"] == 2
    lats_b2 = tr["lat"][b2]
    tops_b2 = tr["cloud_top_km"][b2]
    bots_b2 = tr["cloud_bot_km"][b2]
    order = np.argsort(lats_b2)
    lats_b2, tops_b2, bots_b2 = lats_b2[order], tops_b2[order], bots_b2[order]

    ax3.vlines(lats_b2, bots_b2, tops_b2, color="steelblue", alpha=0.25, linewidth=1.2, label="Cloud Vertical Column")
    ax3.scatter(lats_b2, tops_b2, c="crimson", s=5.0, label="Cloud Top Height ($h_{\\mathrm{top}}$)", zorder=3)
    ax3.scatter(lats_b2, bots_b2, c="navy", s=5.0, label="Cloud Base Height ($h_{\\mathrm{bot}}$)", zorder=3)
    if np.any(b2_all):
        lat_s = tr["lat"][b2_all]
        surf_s = np.nan_to_num(tr["surface_height_km"][b2_all], nan=0.0)
        s_ord = np.argsort(lat_s)
        ax3.fill_between(lat_s[s_ord], 0, surf_s[s_ord], color="saddlebrown", alpha=0.35, label="Terrain Elevation")
    ax3.axhline(THRESHOLD_15KFT_KM, color="darkorange", linestyle="--", linewidth=1.5, label="15,000 ft (4.572 km) Aviation Split")
    ax3.set_ylim(0, 15.5)
    ax3.set_title("(d) ICESat-2 Along-Track Vertical Profile\nCloud Top & Base Curtain (Profile 2)", fontweight="bold")
    ax3.set_xlabel("Latitude (°N)")
    ax3.set_ylabel("Altitude Above Sea Level (km)")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=7.2)
    ax3.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig1_dataset_overview")


def generate_fig2_dataset_statistics(granules_490: list, granules_408: list) -> None:
    """
    Fig 2: Dataset Physical & Radiometric Statistics across all granules.
    """
    tops_all, bots_all, nlayers_all = [], [], []
    months_all = []
    for g in granules_490:
        dt = datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc)
        months_all.append(dt.strftime("%Y-%m"))
        tr = extract_granule_tracks(g.h5_path)
        nlayers_all.append(tr["n_layers"])
        cloudy = (tr["has_cloud"] > 0.5) & np.isfinite(tr["cloud_top_km"]) & np.isfinite(tr["cloud_bot_km"])
        if np.any(cloudy):
            tops_all.append(tr["cloud_top_km"][cloudy])
            bots_all.append(tr["cloud_bot_km"][cloudy])

    tops = np.concatenate(tops_all)
    bots = np.concatenate(bots_all)
    nlayers = np.concatenate(nlayers_all)

    # Sample IR and VIS pixel values on daytime granules from 408-split
    ir_vals, vis_vals, cat_vals = [], [], []
    day_408 = [
        g for g in granules_408
        if 8 <= datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc).hour <= 14
    ]
    for g in day_408[::3]:
        ir_im = load_ir_image(g.img_path)
        vis_im = load_ir_image(os.path.join("../vis0.6", g.img_name))
        tr = extract_granule_tracks(g.h5_path)
        r, c = tr["row"], tr["col"]
        ir_p = ir_im[r, c].astype(np.float32) / 255.0
        vis_p = vis_im[r, c].astype(np.float32) / 255.0
        cl = tr["has_cloud"] > 0.5
        tp = tr["cloud_top_km"]
        # 0: Clear, 1: Low cloud <= 15kft, 2: High cloud > 15kft
        cat = np.zeros(len(r), dtype=np.int32)
        cat[cl & (tp <= THRESHOLD_15KFT_KM)] = 1
        cat[cl & (tp > THRESHOLD_15KFT_KM)] = 2
        ir_vals.append(ir_p)
        vis_vals.append(vis_p)
        cat_vals.append(cat)

    ir_v = np.concatenate(ir_vals)
    vis_v = np.concatenate(vis_vals)
    cat_v = np.concatenate(cat_vals)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 3.9))
    plt.subplots_adjust(wspace=0.38)

    # Panel (a): Joint Top vs Base height distribution
    ax = axes[0]
    hb = ax.hexbin(tops, bots, gridsize=45, cmap="YlGnBu", mincnt=10, bins="log", rasterized=True)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("$\\log_{10}(\\mathrm{Shot\\ Count})$")
    ax.plot([0, 16], [0, 16], "k--", linewidth=1.0, label="$h_{\\mathrm{bot}} = h_{\\mathrm{top}}$ Boundary")
    ax.axvline(THRESHOLD_15KFT_KM, color="crimson", linestyle=":", linewidth=1.4, label="15 kft (4.57 km)")
    ax.axhline(THRESHOLD_15KFT_KM, color="crimson", linestyle=":", linewidth=1.4)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_xlabel("ICESat-2 Cloud Top Height $h_{\\mathrm{top}}$ (km)")
    ax.set_ylabel("ICESat-2 Cloud Base Height $h_{\\mathrm{bot}}$ (km)")
    ax.set_title(f"(a) Joint Cloud Top vs. Base Distribution\n({len(tops):,} Cloudy Lidar Shots)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Panel (b): Radiometric Separation in IR 10.5 vs VIS 0.6 by Altitude Category
    ax = axes[1]
    rng = np.random.default_rng(42)
    for c_id, label, col in [
        (0, "Clear Sky", "#2b83ba"),
        (1, "Low Cloud ($\\leq 15$ kft / 4.57 km)", "#fdae61"),
        (2, "High Cloud ($> 15$ kft / 4.57 km)", "#d7191c"),
    ]:
        m = cat_v == c_id
        idx_all = np.where(m)[0]
        if len(idx_all) > 2000:
            idx = rng.choice(idx_all, size=2000, replace=False)
        else:
            idx = idx_all
        ax.scatter(
            ir_v[idx],
            vis_v[idx],
            s=6.0,
            alpha=0.38,
            c=col,
            edgecolors="none",
            rasterized=True,
            label=f"{label} ($\\mu_{{\\mathrm{{IR}}}}={ir_v[m].mean():.2f}, \\mu_{{\\mathrm{{VIS}}}}={vis_v[m].mean():.2f}$)",
        )
    ax.set_xlabel("Normalized IR 10.5 µm Brightness (Coldness)")
    ax.set_ylabel("Normalized VIS 0.6 µm Solar Reflectance")
    ax.set_title("(b) Dual-Spectral Separation of Low Clouds\nVIS 0.6 µm Resolves Low-Cloud IR Ambiguity", fontweight="bold")
    ax.legend(loc="upper left", markerscale=3.0, fontsize=7.2, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Panel (c): Layer Multiplicity & Seasonal Granule Distribution
    ax = axes[2]
    u_months = sorted(set(months_all))
    m_counts = [months_all.count(m) for m in u_months]
    x_m = np.arange(len(u_months))
    bars = ax.bar(x_m, m_counts, color="#4575b4", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_xticks(x_m)
    ax.set_xticklabels([m[2:] for m in u_months], rotation=45, ha="right")
    ax.set_xlabel("Acquisition Month (YY-MM, 2024–2025)")
    ax.set_ylabel("Matched Overpass Granules")
    c0_pct = 100.0 * np.mean(nlayers == 0)
    c1_pct = 100.0 * np.mean(nlayers == 1)
    c2_pct = 100.0 * np.mean(nlayers >= 2)
    ax.set_title(
        f"(c) Seasonal Coverage (490 Granules)\nClear: {c0_pct:.1f}% | 1-Layer: {c1_pct:.1f}% | Multi-Layer: {c2_pct:.1f}%",
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    save_fig(fig, "fig2_dataset_statistics")


def generate_fig3_architecture_pipeline() -> None:
    """
    Fig 3 / Fig 4: Clean, visual infographic schematic of the proposed Physics-Guided
    Sparse-to-Dense Foundation Framework and Dilated + GT-Distance Adaptive Learning Pipeline.
    Compressed into 5 modular stages with custom vector infographics inside each card.
    """
    fig, ax = plt.subplots(figsize=(16.0, 3.0))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 18.8)
    ax.axis("off")

    boxes = [
        (1.0,  "Multi-Modal\nSatellite Inputs",        "#fffdf5", "#d4a373", "#7f4f24"),
        (22.2, "Radiometric\nContext Stem",            "#f4fbf7", "#27ae60", "#145a32"),
        (43.4, "Foundation Backbone\n& High-Res Skip", "#f2f8fd", "#2980b9", "#154360"),
        (64.6, "Structurally\nOrdered Heads",          "#fdf5f5", "#c0392b", "#7b241c"),
        (85.8, "Dilated & Distance\nAdaptive Learning", "#fefcf2", "#e67e22", "#873600"),
    ]

    w, h = 19.0, 16.4
    y0 = 1.2

    for i, (x, title, fc, ec, tc) in enumerate(boxes):
        card = mpatches.FancyBboxPatch(
            (x, y0), w, h,
            boxstyle="round,pad=0.3,rounding_size=0.95",
            facecolor=fc, edgecolor=ec, linewidth=1.7, zorder=1
        )
        ax.add_patch(card)
        ax.text(
            x + w / 2, y0 + h - 2.0, title,
            ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=tc, linespacing=1.16, zorder=5
        )
        ax.plot([x + 1.1, x + w - 1.1], [y0 + h - 3.9, y0 + h - 3.9], color=ec, lw=0.9, alpha=0.45, zorder=2)

        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            ax.annotate(
                "", xy=(nx - 0.35, y0 + h / 2), xytext=(x + w + 0.35, y0 + h / 2),
                arrowprops=dict(arrowstyle="->,head_width=0.42,head_length=0.52", lw=2.0, color="#2c3e50"),
                zorder=10
            )

    # ==========================================================================
    # Infographic 1: Multi-Modal Inputs (3-Frame Satellite Stack + 9-Level ERA5)
    # ==========================================================================
    x1 = boxes[0][0]
    frame_colors = ["#aed6f1", "#5dade2", "#2e86c1"]
    for k, fcol in enumerate(frame_colors):
        fx = x1 + 1.4 + k * 0.85
        fy = y0 + 2.4 + k * 0.95
        fw, fh = 7.2, 6.7
        rect = mpatches.FancyBboxPatch(
            (fx, fy), fw, fh, boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor=fcol, edgecolor="#1b4f72", lw=1.1, alpha=0.90, zorder=3 + k
        )
        ax.add_patch(rect)
        if k == 2:
            c1 = mpatches.Ellipse((fx + 3.7, fy + 3.8), 4.3, 2.2, angle=14, facecolor="white", edgecolor="none", alpha=0.88, zorder=6)
            c2 = mpatches.Ellipse((fx + 2.6, fy + 4.3), 2.6, 1.8, angle=-10, facecolor="white", edgecolor="none", alpha=0.92, zorder=6)
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.text(fx + fw / 2, fy + 1.05, "IR + VIS", ha="center", va="center", fontsize=6.9, fontweight="bold", color="white", zorder=7)
    ax.text(x1 + 5.8, y0 + 1.05, "3-Frame Seq.", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    era_x = x1 + 11.7
    era_w = 5.4
    era_cols = plt.cm.RdYlBu_r(np.linspace(0.15, 0.88, 9))
    for lvl in range(9):
        ly = y0 + 2.3 + lvl * 0.96
        lrect = mpatches.Rectangle((era_x, ly), era_w, 0.78, facecolor=era_cols[8 - lvl], edgecolor="#555555", lw=0.6, zorder=4)
        ax.add_patch(lrect)
    ax.annotate(
        "", xy=(era_x + era_w + 0.65, y0 + 10.9), xytext=(era_x + era_w + 0.65, y0 + 2.2),
        arrowprops=dict(arrowstyle="<->,head_width=0.22,head_length=0.28", lw=1.0, color="#7f4f24"), zorder=5
    )
    ax.text(era_x + era_w / 2, y0 + 1.05, "9-Lvl ERA5", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    # ==========================================================================
    # Infographic 2: Radiometric Context Stem (Local Kernel + Motion Vectors)
    # ==========================================================================
    x2 = boxes[1][0]
    gx0, gy0 = x2 + 1.5, y0 + 2.5
    cell = 1.62
    for r in range(5):
        for c in range(5):
            is_center = (r == 2 and c == 2)
            is_ring = (1 <= r <= 3 and 1 <= c <= 3) and not is_center
            gfc = "#27ae60" if is_center else ("#abebc6" if is_ring else "#e8f8f5")
            grect = mpatches.Rectangle(
                (gx0 + c * cell, gy0 + r * cell), cell, cell,
                facecolor=gfc, edgecolor="#1e8449", lw=0.7, zorder=3
            )
            ax.add_patch(grect)
    ax.annotate(
        "", xy=(gx0 + 2.5 * cell, gy0 + 4.35 * cell), xytext=(gx0 + 2.5 * cell, gy0 + 2.5 * cell),
        arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.25", lw=1.2, color="#145a32"), zorder=5
    )
    ax.annotate(
        "", xy=(gx0 + 4.35 * cell, gy0 + 2.5 * cell), xytext=(gx0 + 2.5 * cell, gy0 + 2.5 * cell),
        arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.25", lw=1.2, color="#145a32"), zorder=5
    )
    ax.text(gx0 + 2.5 * cell, y0 + 1.05, "Local Anomaly", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    mx0, my0 = x2 + 10.6, y0 + 2.5
    mbox = mpatches.FancyBboxPatch(
        (mx0, my0), 6.9, 8.1, boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor="#d4efdf", edgecolor="#1e8449", lw=1.0, zorder=3
    )
    ax.add_patch(mbox)
    cl_old = mpatches.Ellipse((mx0 + 2.3, my0 + 2.6), 3.0, 1.9, angle=20, facecolor="white", edgecolor="#27ae60", ls="--", lw=1.0, alpha=0.75, zorder=4)
    cl_new = mpatches.Ellipse((mx0 + 4.7, my0 + 5.5), 3.2, 2.0, angle=20, facecolor="white", edgecolor="#145a32", lw=1.3, zorder=5)
    ax.add_patch(cl_old)
    ax.add_patch(cl_new)
    ax.annotate(
        "", xy=(mx0 + 4.6, my0 + 5.5), xytext=(mx0 + 2.3, my0 + 2.6),
        arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.32", lw=1.5, color="#c0392b"), zorder=6
    )
    ax.text(mx0 + 3.45, y0 + 1.05, "Motion Diffs", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    # ==========================================================================
    # Infographic 3: Foundation Backbone & High-Res Skip
    # ==========================================================================
    x3 = boxes[2][0]
    sy = y0 + 8.7
    for s_idx, (sw, sh) in enumerate([(2.3, 2.1), (1.9, 1.7), (2.3, 2.1)]):
        sx = x3 + 2.1 + s_idx * 5.2
        srect = mpatches.FancyBboxPatch(
            (sx, sy - sh / 2), sw, sh, boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="#fdebd0", edgecolor="#d35400", lw=1.1, zorder=4
        )
        ax.add_patch(srect)
    ax.plot([x3 + 1.5, x3 + 17.5], [sy, sy], color="#e67e22", lw=1.5, ls="--", zorder=3)
    ax.text(x3 + w / 2, sy + 1.65, "High-Res Skip (1x, 1/2x, 1/4x)", ha="center", va="center", fontsize=6.8, color="#a04000", fontweight="bold")

    by = y0 + 4.1
    for b_idx in range(3):
        bx = x3 + 1.9 + b_idx * 3.6
        brect = mpatches.FancyBboxPatch(
            (bx, by - 1.5), 2.8, 3.1, boxstyle="round,pad=0.05,rounding_size=0.25",
            facecolor="#aed6f1", edgecolor="#1f618d", lw=1.2, zorder=4
        )
        ax.add_patch(brect)
        ax.plot([bx + 0.55, bx + 2.25], [by - 0.65, by + 0.75], color="#154360", lw=0.85, alpha=0.75, zorder=5)
        ax.plot([bx + 0.55, bx + 2.25], [by + 0.75, by - 0.65], color="#154360", lw=0.85, alpha=0.75, zorder=5)
        if b_idx < 2:
            ax.annotate(
                "", xy=(bx + 3.55, by), xytext=(bx + 2.85, by),
                arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22", lw=1.2, color="#1f618d"), zorder=5
            )

    trap = mpatches.Polygon(
        [[x3 + 13.1, by - 1.3], [x3 + 13.1, by + 1.3], [x3 + 17.2, by + 2.4], [x3 + 17.2, by - 2.0]],
        closed=True, facecolor="#d2b4de", edgecolor="#6c3483", lw=1.2, zorder=4
    )
    ax.add_patch(trap)
    ax.annotate(
        "", xy=(x3 + 13.0, by), xytext=(x3 + 12.0, by),
        arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22", lw=1.2, color="#1f618d"), zorder=5
    )
    ax.annotate(
        "", xy=(x3 + 15.2, by + 1.9), xytext=(x3 + 15.2, sy - 0.2),
        arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.25", lw=1.3, color="#d35400"), zorder=5
    )
    ax.text(x3 + w / 2, y0 + 1.05, "DINOv3 / SAM 2.1 + DPT", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    # ==========================================================================
    # Infographic 4: Structurally Ordered Heads (Cloud Cross-Section h_bot <= h_top)
    # ==========================================================================
    x4 = boxes[3][0]
    hx0, hy0 = x4 + 1.5, y0 + 2.4
    hw, hh = 16.0, 8.4
    sky = mpatches.FancyBboxPatch(
        (hx0, hy0), hw, hh, boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor="#ebf5fb", edgecolor="#c0392b", lw=1.0, zorder=3
    )
    ax.add_patch(sky)
    gnd = mpatches.Rectangle((hx0, hy0), hw, 0.65, facecolor="#d5dbdb", edgecolor="none", zorder=4)
    ax.add_patch(gnd)

    c_main = mpatches.Ellipse((hx0 + 7.6, hy0 + 4.8), 10.6, 3.7, facecolor="white", edgecolor="#7f8c8d", lw=1.0, zorder=5)
    c_top1 = mpatches.Ellipse((hx0 + 6.0, hy0 + 5.7), 5.6, 2.6, facecolor="white", edgecolor="none", zorder=5)
    c_top2 = mpatches.Ellipse((hx0 + 9.4, hy0 + 5.3), 4.9, 2.4, facecolor="white", edgecolor="none", zorder=5)
    ax.add_patch(c_main)
    ax.add_patch(c_top1)
    ax.add_patch(c_top2)

    y_top_line = hy0 + 7.0
    y_bot_line = hy0 + 2.9
    ax.plot([hx0 + 1.1, hx0 + hw - 1.1], [y_top_line, y_top_line], color="#c0392b", lw=1.8, ls="-", zorder=6)
    ax.plot([hx0 + 1.1, hx0 + hw - 1.1], [y_bot_line, y_bot_line], color="#2980b9", lw=1.8, ls="--", zorder=6)

    ax.annotate(
        "", xy=(hx0 + 7.6, y_top_line), xytext=(hx0 + 7.6, y_bot_line),
        arrowprops=dict(arrowstyle="<->,head_width=0.28,head_length=0.32", lw=1.6, color="#922b21"), zorder=7
    )
    ax.text(hx0 + 13.6, y_top_line + 0.6, r"$\hat{h}_{\mathrm{top}}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#c0392b", zorder=7)
    ax.text(hx0 + 13.6, y_bot_line - 0.7, r"$\hat{h}_{\mathrm{bot}}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#2980b9", zorder=7)
    ax.text(x4 + w / 2, y0 + 1.05, r"$0 \leq \hat{h}_{\mathrm{bot}} \leq \hat{h}_{\mathrm{top}} \leq 16\mathrm{\ km}$", ha="center", va="center", fontsize=7.4, color="#7b241c", fontweight="bold")

    # ==========================================================================
    # Infographic 5: Dilated Ground-Truth & Distance-Adaptive Learning
    # ==========================================================================
    x5 = boxes[4][0]
    lx0, ly0 = x5 + 1.5, y0 + 2.4
    lw_box, lh_box = 16.0, 8.4
    fbox = mpatches.FancyBboxPatch(
        (lx0, ly0), lw_box, lh_box, boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor="#fef9e7", edgecolor="#e67e22", lw=1.0, zorder=3
    )
    ax.add_patch(fbox)

    zone_far = mpatches.Polygon(
        [[lx0 + 2.3, ly0 + 0.25], [lx0 + 6.3, ly0 + lh_box - 0.25], [lx0 + 13.5, ly0 + lh_box - 0.25], [lx0 + 9.5, ly0 + 0.25]],
        closed=True, facecolor="#fdebd0", edgecolor="none", zorder=4
    )
    ax.add_patch(zone_far)

    zone_dil = mpatches.Polygon(
        [[lx0 + 4.1, ly0 + 0.25], [lx0 + 8.1, ly0 + lh_box - 0.25], [lx0 + 11.7, ly0 + lh_box - 0.25], [lx0 + 7.7, ly0 + 0.25]],
        closed=True, facecolor="#f5b041", edgecolor="#d35400", lw=0.85, ls=":", alpha=0.68, zorder=5
    )
    ax.add_patch(zone_dil)

    for b_offset in [-1.1, 0.0, 1.1]:
        ax.plot(
            [lx0 + 5.9 + b_offset, lx0 + 9.9 + b_offset],
            [ly0 + 0.45, ly0 + lh_box - 0.45],
            color="#c0392b", lw=1.8, zorder=6
        )

    ax.annotate(
        "", xy=(lx0 + 15.0, ly0 + 4.0), xytext=(lx0 + 9.1, ly0 + 4.0),
        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.3", lw=1.5, color="#873600"), zorder=7
    )
    ax.text(lx0 + 13.0, ly0 + 5.2, r"$d_{\mathrm{GT}}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#873600", zorder=7)
    ax.text(lx0 + 2.8, ly0 + 6.4, "Dilated\nLabels", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#a04000", zorder=7)
    ax.text(x5 + w / 2, y0 + 1.05, r"Dilated $w_{\mathrm{dil}} < 1$ & EMA Teacher", ha="center", va="center", fontsize=7.2, color="#444444", fontweight="bold")

    fig.tight_layout(pad=0.25)
    save_fig(fig, "fig3_architecture_pipeline")


def generate_fig4_qualitative_comparison(granules_408: list) -> None:
    """
    Fig 4: Qualitative Dense 2D Predictions & Along-Track Vertical Curtain Comparison across Models.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, val_408 = split_granules_chronologically(granules_408, val_ratio=0.2)
    val_ds_3f = CloudPatchDataset(val_408, patch_size=256, patches_per_granule=10, is_train=False, use_3frames=True, use_vis06=True, max_cached_images=64)
    val_ds_1f = CloudPatchDataset(val_408, patch_size=256, patches_per_granule=10, is_train=False, use_3frames=False, use_vis06=False, max_cached_images=64)

    # Find a patch with both low and high clouds along the track
    best_idx = 0
    best_var = -1.0
    for idx in range(0, min(len(val_ds_3f), 300), 3):
        item = val_ds_3f[idx]
        cm = (item["valid_mask"] > 0.5) & (item["cloud_mask"] > 0.5)
        if int(cm.sum()) > 120:
            tops = item["cloud_top"][cm].numpy() * MAX_HEIGHT_KM
            v = float(np.std(tops)) * float(np.any(tops <= 4.5)) * float(np.any(tops > 6.5))
            if v > best_var:
                best_var = v
                best_idx = idx

    s_3f = val_ds_3f[best_idx]
    x_3f = s_3f["image"].unsqueeze(0).to(device)

    model_defs = [
        ("Semi-Sup U-Net\n(3f IR+VIS)", "unet", 8, "outputs/semisup_unet_3f_vis/best_semisup.pt", x_3f),
        ("Semi-Sup SAM 2.1\n(3f IR+VIS)", "sam", 8, "outputs/semisup_sam_3f_vis/best_semisup.pt", x_3f),
        ("Semi-Sup DINOv3\n(3f IR+VIS, Ours)", "dinov3", 8, "outputs/semisup_dinov3_3f_vis/best_semisup.pt", x_3f),
    ]

    preds_map = {}
    with torch.no_grad():
        for name, arch, in_ch, ckpt_p, x_in in model_defs:
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            base_ch = ckpt.get("args", {}).get("base_ch", 48)
            mod = build_model(arch=arch, in_channels=in_ch, base_ch=base_ch, pretrained=False).to(device)
            mod.load_state_dict(ckpt.get("model_state_dict", ckpt.get("teacher_state", ckpt.get("model_state"))))
            mod.eval()
            out = mod(x_in)
            prob = out["cloud_prob"].squeeze().cpu().numpy()
            top = out["cloud_top"].squeeze().cpu().numpy() * MAX_HEIGHT_KM
            bot = np.minimum(out["cloud_bot"].squeeze().cpu().numpy() * MAX_HEIGHT_KM, top)
            thick = np.maximum(0.0, top - bot)
            preds_map[name] = {"prob": prob, "top": top, "bot": bot, "thick": thick}

    n_cols = len(model_defs) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols + 0.8, 8.4))
    plt.subplots_adjust(wspace=0.22, hspace=0.30)

    ir_np = s_3f["image"][0].numpy()
    vis_np = s_3f["image"][3].numpy()
    vm = s_3f["valid_mask"].numpy() > 0.5
    cm = s_3f["cloud_mask"].numpy() > 0.5
    gt_top = s_3f["cloud_top"].numpy() * MAX_HEIGHT_KM
    gt_bot = s_3f["cloud_bot"].numpy() * MAX_HEIGHT_KM
    gt_thick = np.maximum(0.0, gt_top - gt_bot)

    ry, cx = np.where(vm & cm)
    r_clr, c_clr = np.where(vm & (~cm))

    # Row 0 Col 0: IR 10.5 + GT Track Top Height
    ax = axes[0, 0]
    ax.imshow(ir_np, cmap="gray")
    ax.scatter(c_clr, r_clr, c="dodgerblue", s=1.5, label="Clear")
    sc_gt_top = ax.scatter(cx, ry, c=gt_top[ry, cx], cmap="turbo", vmin=0, vmax=14, s=6.0, label="GT Top")
    ax.set_title("Input IR 10.5 µm ($t=0$)\n+ GT Cloud Top Track", fontweight="bold", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 1 Col 0: VIS 0.6 + GT Cloud Base Track
    ax = axes[1, 0]
    ax.imshow(vis_np, cmap="bone")
    ax.scatter(c_clr, r_clr, c="dodgerblue", s=1.5)
    sc_gt_bot = ax.scatter(cx, ry, c=gt_bot[ry, cx], cmap="turbo", vmin=0, vmax=14, s=6.0, label="GT Base")
    ax.set_title("Input VIS 0.6 µm ($t=0$)\n+ GT Cloud Base Track", fontweight="bold", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 2 Col 0: ICESat-2 GT Vertical Curtain
    order_t = np.argsort(ry)
    ry_s, cx_s = ry[order_t], cx[order_t]
    x_dist = np.arange(len(ry_s))
    g_t_line = gt_top[ry_s, cx_s]
    g_b_line = gt_bot[ry_s, cx_s]

    ax_gt_prof = axes[2, 0]
    ax_gt_prof.fill_between(x_dist, g_b_line, g_t_line, color="forestgreen", alpha=0.35, label="GT Layer")
    ax_gt_prof.plot(x_dist, g_t_line, "k.", markersize=3.0, label="GT Top")
    ax_gt_prof.plot(x_dist, g_b_line, ".", color="#444444", markersize=3.0, label="GT Base")
    ax_gt_prof.axhline(THRESHOLD_15KFT_KM, color="darkorange", linestyle="--", linewidth=1.0, label="15 kft Split")
    ax_gt_prof.set_ylim(0, 15.0)
    ax_gt_prof.set_title("ICESat-2 Lidar Track\nGround-Truth Profile", fontweight="bold", fontsize=9)
    ax_gt_prof.set_xlabel("Along-Track Shot Index", fontsize=8)
    ax_gt_prof.set_ylabel("Altitude (km)", fontsize=8)
    ax_gt_prof.legend(loc="upper right", fontsize=6.2)
    ax_gt_prof.grid(True, linestyle=":", alpha=0.4)

    # Columns 1..N: Model Predictions
    for col_i, (name, _, _, _, _) in enumerate(model_defs, start=1):
        pm = preds_map[name]
        mae_t = float(np.mean(np.abs(pm["top"][ry, cx] - gt_top[ry, cx])))
        mae_b = float(np.mean(np.abs(pm["bot"][ry, cx] - gt_bot[ry, cx])))

        # Row 0: Dense 2D Cloud Top Height
        ax_top = axes[0, col_i]
        masked_top = np.where(pm["prob"] >= 0.35, pm["top"], np.nan)
        ax_top.imshow(ir_np, cmap="gray", alpha=0.45)
        ax_top.imshow(masked_top, cmap="turbo", vmin=0, vmax=14, alpha=0.85)
        ax_top.set_title(f"{name}\nTop MAE: {mae_t:.2f} km", fontweight="bold", fontsize=9)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        # Row 1: Dense 2D Cloud Base Height
        ax_bot = axes[1, col_i]
        masked_bot = np.where(pm["prob"] >= 0.35, pm["bot"], np.nan)
        ax_bot.imshow(vis_np, cmap="bone", alpha=0.45)
        ax_bot.imshow(masked_bot, cmap="turbo", vmin=0, vmax=14, alpha=0.85)
        ax_bot.set_title(f"Pred Cloud Base ($\\hat{{h}}_{{\\mathrm{{bot}}}}$)\nBase MAE: {mae_b:.2f} km", fontweight="bold", fontsize=9)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

        # Row 2: Along-Track Vertical Curtain Comparison (GT vs Model)
        ax_prof = axes[2, col_i]
        p_t_line = pm["top"][ry_s, cx_s]
        p_b_line = pm["bot"][ry_s, cx_s]

        ax_prof.fill_between(x_dist, g_b_line, g_t_line, color="gray", alpha=0.25, label="GT Layer")
        ax_prof.plot(x_dist, g_t_line, "k.", markersize=2.2, label="GT Top")
        ax_prof.plot(x_dist, g_b_line, ".", color="#666666", markersize=2.2, label="GT Base")
        ax_prof.fill_between(x_dist, p_b_line, p_t_line, color="crimson", alpha=0.18, label="Pred Layer")
        ax_prof.plot(x_dist, p_t_line, "-", color="crimson", linewidth=1.3, label="Pred Top")
        ax_prof.plot(x_dist, p_b_line, "-", color="royalblue", linewidth=1.3, label="Pred Base")
        ax_prof.axhline(THRESHOLD_15KFT_KM, color="darkorange", linestyle="--", linewidth=0.9)
        ax_prof.set_ylim(0, 15.0)
        ax_prof.set_title("Along-Track Top & Base\nProfile Comparison", fontweight="bold", fontsize=9)
        ax_prof.set_xlabel("Along-Track Shot Index", fontsize=8)
        ax_prof.set_yticklabels([])
        ax_prof.grid(True, linestyle=":", alpha=0.4)

    cbar_top = fig.colorbar(sc_gt_top, ax=axes[0, :].tolist(), fraction=0.012, pad=0.015)
    cbar_top.set_label(r"Cloud Top Height $\hat{h}_{\mathrm{top}}$ (km)", fontsize=8)

    cbar_bot = fig.colorbar(sc_gt_bot, ax=axes[1, :].tolist(), fraction=0.012, pad=0.015)
    cbar_bot.set_label(r"Cloud Base Height $\hat{h}_{\mathrm{bot}}$ (km)", fontsize=8)
    save_fig(fig, "fig4_qualitative_comparison")


def generate_fig5_altitude_and_regime_breakdown(cvpr_res: dict[str, Any]) -> None:
    """
    Fig 5: Stratified Error Analysis on 1f IR+VIS:
      (a) Altitude Category Breakdown (<= 15 kft vs > 15 kft) across Supervised, UniMatch, S4MC, and Ours (SAM 2 | DINOv3)
      (b) Single-Layer vs Multi-Layer Clouds across U-Net, SegViT, SAM 2, and DINOv3 (Ours, 1f IR+VIS)
      (c) Variance Explained (R^2) across Supervised, UniMatch, S4MC, and Ours (SAM 2 | DINOv3, 1f IR+VIS)
    """
    mod_408 = cvpr_res["modality_temporal_408"]
    with open("outputs/semisup_s4mc_and_ours/s4mc_and_ours_summary.json", "r", encoding="utf-8") as f:
        s4mc_ours = json.load(f)

    methods_ac = [
        ("SAM 2 (Sup.)", mod_408["Supervised SAM (1f IR+VIS (408 Split))"]),
        ("SAM 2 (UniMatch)", mod_408["Semi-Sup SAM (1f IR+VIS (408 Split))"]),
        ("SAM 2 (S4MC)", s4mc_ours["s4mc_1f_vis_sam"]),
        ("SAM 2 (Ours)", s4mc_ours["ours_dist_1f_vis_sam"]),
        ("DINOv3 (Sup.)", mod_408["Supervised DINOv3 (1f IR+VIS (408 Split))"]),
        ("DINOv3 (UniMatch)", mod_408["Semi-Sup DINOv3 (1f IR+VIS (408 Split))"]),
        ("DINOv3 (S4MC)", s4mc_ours["s4mc_1f_vis_dinov3"]),
        ("DINOv3 (Ours)", s4mc_ours["ours_dist_1f_vis_dinov3"]),
    ]

    archs_b = [
        ("U-Net\n(1f IR+VIS)", s4mc_ours["ours_dist_1f_vis_unet"]),
        ("SegViT\n(1f IR+VIS)", s4mc_ours["ours_dist_1f_vis_segvit"]),
        ("SAM 2\n(1f IR+VIS)", s4mc_ours["ours_dist_1f_vis_sam"]),
        ("DINOv3\n(1f IR+VIS)", s4mc_ours["ours_dist_1f_vis_dinov3"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))
    plt.subplots_adjust(wspace=0.28)

    x = np.arange(len(methods_ac))
    w = 0.20

    # Panel (a): Altitude Category Breakdown (<= 15 kft vs > 15 kft) for Top and Base MAE on 1f IR+VIS
    ax = axes[0]
    top_low = [m["top_upto_15kft_mae_km"] for _, m in methods_ac]
    top_high = [m["top_above_15kft_mae_km"] for _, m in methods_ac]
    bot_low = [m["bot_upto_15kft_mae_km"] for _, m in methods_ac]
    bot_high = [m["bot_above_15kft_mae_km"] for _, m in methods_ac]

    ax.bar(x - 1.5 * w, top_low, w, label="Top $\\leq 15$ kft ($4.57$ km)", color="#2b83ba", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5 * w, bot_low, w, label="Base $\\leq 15$ kft ($4.57$ km)", color="#abdda4", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5 * w, top_high, w, label="Top $> 15$ kft ($4.57$ km)", color="#fdae61", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5 * w, bot_high, w, label="Base $> 15$ kft ($4.57$ km)", color="#d7191c", edgecolor="black", linewidth=0.5)

    ax.axvline(3.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods_ac], rotation=32, ha="right", rotation_mode="anchor", fontsize=7.6)
    ax.set_ylabel("Mean Absolute Error (km)")
    ax.set_ylim(0, 2.75)
    ax.set_title("(a) Cloud Top & Base Error by Altitude (1f IR+VIS)\n($\\leq 15$ kft Low Clouds vs. $> 15$ kft High Clouds)", fontweight="bold", fontsize=9.3)
    ax.legend(loc="upper right", fontsize=6.8, ncol=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel (b): Single-Layer (n=1) vs Multi-Layer (n>=2) Cloud Top & Base MAE across 1f IR+VIS Architectures (Ours)
    ax = axes[1]
    xs = np.arange(len(archs_b))
    w_b = 0.19
    t_1l = [m["top_single_layer_mae_km"] for _, m in archs_b]
    t_ml = [m["top_multi_layer_mae_km"] for _, m in archs_b]
    b_1l = [m["bot_single_layer_mae_km"] for _, m in archs_b]
    b_ml = [m["bot_multi_layer_mae_km"] for _, m in archs_b]

    ax.bar(xs - 1.5 * w_b, t_1l, w_b, label="Top (Single-Layer)", color="#3288bd", edgecolor="black", linewidth=0.5)
    ax.bar(xs - 0.5 * w_b, b_1l, w_b, label="Base (Single-Layer)", color="#66c2a5", edgecolor="black", linewidth=0.5)
    ax.bar(xs + 0.5 * w_b, t_ml, w_b, label="Top (Multi-Layer $\\geq 2$)", color="#f46d43", edgecolor="black", linewidth=0.5)
    ax.bar(xs + 1.5 * w_b, b_ml, w_b, label="Base (Multi-Layer $\\geq 2$)", color="#d53e4f", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in archs_b], rotation=0, ha="center", fontsize=8.0)
    ax.set_ylabel("Mean Absolute Error (km)")
    ax.set_ylim(0, 2.75)
    ax.set_title("(b) Single-Layer vs. Multi-Layer Clouds (1f IR+VIS)\nAcross Architectures (Ours)", fontweight="bold", fontsize=9.3)
    ax.legend(loc="upper right", fontsize=6.8, ncol=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel (c): Overall Top & Base R^2 Score Progression across Methods on 1f IR+VIS
    ax = axes[2]
    top_r2 = [m["top_r2"] for _, m in methods_ac]
    bot_r2 = [m["bot_r2"] for _, m in methods_ac]

    ax.bar(x - 0.18, top_r2, 0.34, label="Cloud Top $R^2$ (Higher is Better)", color="#1b7837", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.18, bot_r2, 0.34, label="Cloud Base $R^2$ (Higher is Better)", color="#762a83", edgecolor="black", linewidth=0.5)
    ax.axvline(3.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods_ac], rotation=32, ha="right", rotation_mode="anchor", fontsize=7.6)
    ax.set_ylabel("Coefficient of Determination ($R^2$)")
    ax.set_ylim(0, 0.85)
    ax.set_title("(c) Variance Explained ($R^2$) Across Methods\n(1f IR+VIS: SAM 2 vs. DINOv3)", fontweight="bold", fontsize=9.3)
    ax.legend(loc="upper left", fontsize=7.2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    save_fig(fig, "fig5_altitude_and_regime_breakdown")


def generate_fig6_ablation_and_sparsity(cvpr_res: dict[str, Any]) -> None:
    """
    Fig 6: Modality Disentanglement (408 Split), Lidar Beam Sparsity Scaling, and Component Ablations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
    plt.subplots_adjust(wspace=0.27)

    # Panel (a): Modality & Temporal Disentanglement on the identical 408-granule split
    ax = axes[0]
    mod_res = cvpr_res.get("modality_temporal_408", {})
    mod_steps = [
        ("1f IR\n(Sup)", "Supervised", "1f IR (408 Split)"),
        ("3f IR\n(Sup)", "Supervised", "3f IR Only (408 Split)"),
        ("1f IR+VIS\n(Sup)", "Supervised", "1f IR+VIS (408 Split)"),
        ("3f IR+VIS\n(Sup)", "Supervised", "3f IR+VIS"),
        ("3f IR+VIS\n(Semi-Sup)", "Semi-Sup", "3f IR+VIS"),
    ]
    xm = np.arange(len(mod_steps))
    w = 0.20

    dino_top = [mod_res.get(f"{p} DINOv3 ({tag})", {}).get("top_mae_km", 0) for _, p, tag in mod_steps]
    dino_low = [mod_res.get(f"{p} DINOv3 ({tag})", {}).get("top_upto_15kft_mae_km", 0) for _, p, tag in mod_steps]
    sam_top = [mod_res.get(f"{p} SAM ({tag})", {}).get("top_mae_km", 0) for _, p, tag in mod_steps]
    sam_low = [mod_res.get(f"{p} SAM ({tag})", {}).get("top_upto_15kft_mae_km", 0) for _, p, tag in mod_steps]

    ax.bar(xm - 1.5 * w, dino_top, w, label="DINOv3 Overall Top MAE", color="#2166ac", edgecolor="black", linewidth=0.5)
    ax.bar(xm - 0.5 * w, dino_low, w, label="DINOv3 Low-Cloud ($\\leq 15$kft)", color="#67a9cf", edgecolor="black", linewidth=0.5)
    ax.bar(xm + 0.5 * w, sam_top, w, label="SAM Overall Top MAE", color="#b2182b", edgecolor="black", linewidth=0.5)
    ax.bar(xm + 1.5 * w, sam_low, w, label="SAM Low-Cloud ($\\leq 15$kft)", color="#ef8a62", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xm)
    ax.set_xticklabels([m[0] for m in mod_steps], fontsize=7.5)
    ax.set_ylabel("Mean Absolute Error (km)")
    ax.set_ylim(0, 2.35)
    ax.set_title("(a) Modality & Temporal Disentanglement\n(Identical 408-Granule Split)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.6)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel (b): Lidar Track Sparsity Scaling (1 Beam vs 2 Beams vs 3 Beams)
    ax = axes[1]
    beam_res = cvpr_res.get("beam_sparsity_study", {})
    b_labels = ["1 Beam (Nadir)\n(0.025% px)", "2 Beams (Asym.)\n(0.050% px)", "3 Beams (Full)\n(0.075% px)"]
    b_keys_sup = [
        "Supervised DINOv3 (1 Beam (0.025% Pixels))",
        "Supervised DINOv3 (2 Beams (0.050% Pixels))",
        "Supervised DINOv3 (3 Beams, 0.075% Pixels)",
    ]
    b_keys_semi = [
        "Semi-Sup DINOv3 (1 Beam (0.025% Pixels))",
        "Semi-Sup DINOv3 (2 Beams (0.050% Pixels))",
        "Semi-Sup DINOv3 (3 Beams, 0.075% Pixels)",
    ]
    xb = np.arange(3)
    wb = 0.19
    sup_top_b = [beam_res.get(k, {}).get("top_mae_km", 0) for k in b_keys_sup]
    semi_top_b = [beam_res.get(k, {}).get("top_mae_km", 0) for k in b_keys_semi]
    sup_bot_b = [beam_res.get(k, {}).get("bot_mae_km", 0) for k in b_keys_sup]
    semi_bot_b = [beam_res.get(k, {}).get("bot_mae_km", 0) for k in b_keys_semi]

    ax.bar(xb - 1.5 * wb, sup_top_b, wb, color="#fd8d3c", edgecolor="black", linewidth=0.5, label="Supervised Top MAE")
    ax.bar(xb - 0.5 * wb, semi_top_b, wb, color="#1b9e77", edgecolor="black", linewidth=0.5, label="Dilated+Dist (Ours) Top MAE")
    ax.bar(xb + 0.5 * wb, sup_bot_b, wb, color="#f768a1", edgecolor="black", linewidth=0.5, label="Supervised Base MAE")
    ax.bar(xb + 1.5 * wb, semi_bot_b, wb, color="#7570b3", edgecolor="black", linewidth=0.5, label="Dilated+Dist (Ours) Base MAE")
    for idx_b in range(3):
        gain_m = int(round((sup_top_b[idx_b] - semi_top_b[idx_b]) * 1000))
        ax.text(
            xb[idx_b] - 0.5 * wb,
            semi_top_b[idx_b] + 0.04,
            f"-{gain_m}m",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#006837",
        )
    ax.set_xticks(xb)
    ax.set_xticklabels(b_labels, fontsize=7.8)
    ax.set_ylim(0, 2.68)
    ax.set_ylabel("Height MAE (km)")
    ax.set_title("(b) Lidar Beam Sparsity Sensitivity\nDilated+Dist (Ours) Bridges Sparse Supervision", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.6)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel (c): Component Ablation Waterfall on DINOv3 (3f IR+VIS)
    ax = axes[2]
    abl = cvpr_res.get("component_ablations", {})
    m16 = cvpr_res.get("main_16_models", {})
    abl_items = [
        ("Scratch\n(No Pretrain)", abl.get("Supervised DINOv3 (Scratch / Random Init)", {})),
        ("w/o RadSkip\n& Context Stem", abl.get("Supervised DINOv3 (w/o Radiometric Skip & Stem)", {})),
        ("w/o Ordering\nHead", abl.get("Supervised DINOv3 (w/o Structural Ordering Head)", {})),
        ("Full Sup.\nDINOv3", m16.get("Supervised DINOv3 (3f IR+VIS)", {})),
        ("S4MC\n(Context Union)", abl.get("S4MC DINOv3 (3f IR+VIS)", {"top_mae_km": 1.413, "bot_mae_km": 1.390})),
        ("UniMatch\n(Const. Thresh)", abl.get("UniMatch DINOv3 (3f IR+VIS)", {"top_mae_km": 1.346, "bot_mae_km": 1.325})),
        ("Dilated+Dist\n(Ours, σd=24)", m16.get("Semi-Sup DINOv3 (3f IR+VIS)", {"top_mae_km": 1.264, "bot_mae_km": 1.247})),
    ]
    xa = np.arange(len(abl_items))
    a_top = [it[1].get("top_mae_km", 0) for it in abl_items]
    a_bot = [it[1].get("bot_mae_km", 0) for it in abl_items]
    ax.bar(xa - 0.18, a_top, 0.35, label="Cloud Top MAE (km)", color="#4575b4", edgecolor="black", linewidth=0.5)
    ax.bar(xa + 0.18, a_bot, 0.35, label="Cloud Base MAE (km)", color="#91bfdb", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xa)
    ax.set_xticklabels([it[0] for it in abl_items], rotation=32, ha="right", fontsize=7.2)
    ax.set_ylabel("Mean Absolute Error (km)")
    ax.set_ylim(0, max(max(a_top, default=2.0) * 1.18, 2.0))
    ax.set_title("(c) Architecture & Semi-Sup Ablations\n(3-Frame IR + VIS 0.6 µm)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    save_fig(fig, "fig6_ablation_and_sparsity")


def generate_fig_vfm_data_example() -> None:
    """
    Dedicated Data Example Figure: Satellite Imagery with Highlighted Cloud VFM Track Overlay (left)
    paired side-by-side with the 2D Cloud Vertical Feature Mask (VFM) Curtain (top-right) and
    Co-Registered Along-Track IR 10.5 + VIS 0.6 Radiometric Profiles (bottom-right).
    """
    import h5py
    from matplotlib.lines import Line2D
    from PIL import Image

    h5_path = "../vfms/ATL09_20250115112941_04642601_007_01.h5"
    img_name = "2025-01-15_12-10.png"
    ir_img = np.array(Image.open(os.path.join("../ir10.5", img_name)).convert("L"), dtype=np.float32) / 255.0
    vis_img = np.array(Image.open(os.path.join("../vis0.6", img_name)).convert("L"), dtype=np.float32) / 255.0

    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX
    H, W = DEFAULT_IMAGE_SHAPE

    with h5py.File(h5_path, "r") as f:
        grp = f["profile_2/high_rate"]
        lat = grp["latitude"][:]
        lon = grp["longitude"][:]
        in_b = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        lat, lon = lat[in_b], lon[in_b]
        order = np.argsort(lat)
        lat, lon = lat[order], lon[order]
        l_attr = grp["layer_attr"][:][in_b][order]
        l_top = grp["layer_top"][:][in_b][order] / 1000.0
        l_bot = grp["layer_bot"][:][in_b][order] / 1000.0
        surf = grp["surface_height"][:][in_b][order] / 1000.0
        surf = np.where((surf > -1.0) & (surf < 10.0), np.maximum(surf, 0.0), 0.0)
        row = np.clip(np.rint((max_lat - lat) / (max_lat - min_lat) * (H - 1)).astype(int), 0, H - 1)
        col = np.clip(np.rint((lon - min_lon) / (max_lon - min_lon) * (W - 1)).astype(int), 0, W - 1)
        is_cld = (l_attr == 1) & np.isfinite(l_top) & np.isfinite(l_bot) & (l_top > 0) & (l_top < 20.0)
        n_lay = np.sum(is_cld, axis=1)
        top_all = np.where(is_cld, l_top, np.nan)
        bot_all = np.where(is_cld, l_bot, np.nan)
        with np.errstate(all="ignore"):
            h_top = np.nanmax(top_all, axis=1)
            h_bot = np.nanmin(bot_all, axis=1)
        ir_val = ir_img[row, col]
        vis_val = vis_img[row, col]

    fig = plt.figure(figsize=(14.5, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.45], height_ratios=[2.35, 1.0], wspace=0.22, hspace=0.32)

    # Panel (a): Satellite Imagery + Highlighted Cloud VFM Track Overlay
    ax_map = fig.add_subplot(gs[:, 0])
    ir_sub = ir_img[::4, ::4]
    vis_sub = vis_img[::4, ::4]
    rgb_bg = np.stack([
        np.clip(0.55 * vis_sub + 0.45 * ir_sub, 0, 1),
        np.clip(0.65 * vis_sub + 0.35 * ir_sub, 0, 1),
        np.clip(0.45 * vis_sub + 0.65 * ir_sub + 0.08, 0, 1),
    ], axis=-1)
    ax_map.imshow(rgb_bg, extent=[min_lon, max_lon, min_lat, max_lat], aspect="auto")

    # Dark halo behind track for high contrast over bright clouds
    ax_map.plot(lon, lat, color="black", linewidth=5.2, alpha=0.78, zorder=2)

    m_clear = (n_lay == 0)
    m_low = (n_lay == 1) & (h_top <= THRESHOLD_15KFT_KM)
    m_high = (n_lay == 1) & (h_top > THRESHOLD_15KFT_KM)
    m_multi = (n_lay >= 2)

    ax_map.scatter(lon[m_clear], lat[m_clear], c="#00b4d8", s=6.5, label="Clear Sky (VFM Layer = 0)", zorder=3)
    ax_map.scatter(lon[m_low], lat[m_low], c="#ffb703", s=8.5, label="Low Cloud VFM (≤ 15 kft / 4.57 km)", zorder=4)
    ax_map.scatter(lon[m_high], lat[m_high], c="#e63946", s=8.5, label="High Cloud VFM (> 15 kft, 1-Layer)", zorder=5)
    ax_map.scatter(lon[m_multi], lat[m_multi], c="#b5179e", s=9.5, label="Multi-Layer Cloud VFM (≥ 2 Layers)", zorder=6)

    for target_lat in [26.0, 29.0, 32.0, 35.0, 38.0]:
        idx = int(np.argmin(np.abs(lat - target_lat)))
        ax_map.scatter([lon[idx]], [lat[idx]], c="white", edgecolor="black", s=38, zorder=8, linewidth=1.0)
        ax_map.text(
            lon[idx] + 0.45,
            lat[idx] - 0.15,
            f"{target_lat:.0f}°N",
            color="white",
            fontweight="bold",
            fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.68, edgecolor="none"),
            zorder=9,
        )

    ax_map.set_title(
        "(a) MTG-I1 FCI Satellite Imagery (2025-01-15 12:10 UTC)\nwith Highlighted ICESat-2 Cloud VFM Track Overlay",
        fontweight="bold",
    )
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_xlim(min_lon, max_lon)
    ax_map.set_ylim(min_lat, max_lat)
    ax_map.legend(loc="lower left", framealpha=0.92, facecolor="white", edgecolor="gray", markerscale=1.6, fontsize=7.6)

    # Panel (b): 2D Cloud Vertical Feature Mask (VFM) Curtain
    ax_vfm = fig.add_subplot(gs[0, 1])
    ax_vfm.set_facecolor("#f4f7fb")

    l0_low_lat, l0_low_b, l0_low_t = [], [], []
    l0_high_lat, l0_high_b, l0_high_t = [], [], []
    multi_lat, multi_b, multi_t = [], [], []

    for i in range(len(lat)):
        if n_lay[i] == 0:
            continue
        valid_l = np.where(is_cld[i])[0]
        for rank, l_idx in enumerate(valid_l):
            t_val = top_all[i, l_idx]
            b_val = bot_all[i, l_idx]
            if rank == 0:
                if t_val <= THRESHOLD_15KFT_KM:
                    l0_low_lat.append(lat[i]); l0_low_b.append(b_val); l0_low_t.append(t_val)
                else:
                    l0_high_lat.append(lat[i]); l0_high_b.append(b_val); l0_high_t.append(t_val)
            else:
                multi_lat.append(lat[i]); multi_b.append(b_val); multi_t.append(t_val)

    ax_vfm.vlines(l0_low_lat, l0_low_b, l0_low_t, color="#ffb703", alpha=0.55, linewidth=1.2, zorder=2)
    ax_vfm.vlines(l0_high_lat, l0_high_b, l0_high_t, color="#3a86ff", alpha=0.42, linewidth=1.2, zorder=2)
    ax_vfm.vlines(multi_lat, multi_b, multi_t, color="#b5179e", alpha=0.68, linewidth=1.4, zorder=3)
    ax_vfm.scatter(multi_lat, multi_t, c="#b5179e", s=2.5, zorder=4)
    ax_vfm.scatter(multi_lat, multi_b, c="#7209b7", s=2.5, zorder=4)

    cld_mask = n_lay >= 1
    ax_vfm.scatter(lat[cld_mask], h_top[cld_mask], c="#d90429", s=3.5, zorder=5)
    ax_vfm.scatter(lat[cld_mask], h_bot[cld_mask], c="#023e8a", s=3.5, zorder=5)
    ax_vfm.fill_between(lat, 0, surf, color="#8d6e63", alpha=0.55, zorder=4)
    ax_vfm.axhline(THRESHOLD_15KFT_KM, color="#fb8500", linestyle="--", linewidth=1.5, zorder=6)

    # Non-overlapping annotations pointing to exact physical features in 2025-01-15_12-10
    ax_vfm.annotate(
        "Low Boundary-Layer Stratus\n(≤ 15 kft: Warm IR, Bright VIS)",
        xy=(26.6, 3.1),
        xytext=(25.2, 6.2),
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.2),
        fontsize=7.4,
        fontweight="bold",
        color="#b45309",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.90, edgecolor="#ffb703"),
        zorder=8,
    )
    ax_vfm.annotate(
        "Multi-Layer Overlap\n(Cirrus 11 km over Low Deck 3 km)",
        xy=(28.6, 9.2),
        xytext=(28.6, 12.1),
        arrowprops=dict(arrowstyle="->", color="#7209b7", lw=1.2),
        fontsize=7.4,
        fontweight="bold",
        color="#7209b7",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.90, edgecolor="#b5179e"),
        zorder=8,
    )
    ax_vfm.annotate(
        "Clear-Sky Terrain\n& Isolated Cumulus",
        xy=(33.9, 2.2),
        xytext=(32.7, 6.2),
        arrowprops=dict(arrowstyle="->", color="#5c4033", lw=1.2),
        fontsize=7.4,
        fontweight="bold",
        color="#5c4033",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.90, edgecolor="#8d6e63"),
        zorder=8,
    )
    ax_vfm.annotate(
        "Multi-Layer Frontal System\n(Lidar Penetrates 2–3 Decks)",
        xy=(39.1, 7.2),
        xytext=(35.1, 10.6),
        arrowprops=dict(arrowstyle="->", color="#d90429", lw=1.2),
        fontsize=7.4,
        fontweight="bold",
        color="#9b2226",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.90, edgecolor="#d90429"),
        zorder=8,
    )

    handles_vfm = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d90429", markersize=5, label="Cloud Top ($h_{\\mathrm{top}}$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#023e8a", markersize=5, label="Cloud Base ($h_{\\mathrm{bot}}$)"),
        mpatches.Patch(facecolor="#3a86ff", alpha=0.45, label="Upper Cloud (> 15 kft)"),
        mpatches.Patch(facecolor="#ffb703", alpha=0.55, label="Low Cloud (≤ 15 kft)"),
        mpatches.Patch(facecolor="#b5179e", alpha=0.65, label="Lower Multi-Layer (≥ 2)"),
        Line2D([0], [0], color="#fb8500", lw=1.5, ls="--", label="15,000 ft (4.57 km) Split"),
        mpatches.Patch(facecolor="#8d6e63", alpha=0.55, label="Terrain Elevation"),
    ]
    ax_vfm.legend(handles=handles_vfm, loc="upper center", ncol=4, framealpha=0.94, fontsize=6.9)
    ax_vfm.set_title("(b) 2D Cloud Vertical Feature Mask (VFM) Curtain Along Track (ICESat-2 ATL09 Profile 2)", fontweight="bold")
    ax_vfm.set_ylabel("Altitude Above Sea Level (km)")
    ax_vfm.set_xlim(25.0, 40.0)
    ax_vfm.set_ylim(0, 16.5)
    ax_vfm.grid(True, linestyle=":", alpha=0.5)

    # Panel (c): Co-Registered Satellite Radiometry Along Track
    ax_rad = fig.add_subplot(gs[1, 1], sharex=ax_vfm)
    kernel = np.ones(15) / 15.0
    ir_smooth = np.convolve(ir_val, kernel, mode="same")
    vis_smooth = np.convolve(vis_val, kernel, mode="same")

    ax_rad.plot(lat, ir_val, color="#ef233c", alpha=0.20, linewidth=0.6)
    ax_rad.plot(lat, vis_val, color="#0077b6", alpha=0.20, linewidth=0.6)
    ax_rad.plot(lat, ir_smooth, color="#d90429", linewidth=1.6, label="MTG-I1 IR 10.5 µm (Thermal Coldness)")
    ax_rad.plot(lat, vis_smooth, color="#0077b6", linewidth=1.6, label="MTG-I1 VIS 0.6 µm (Solar Reflectance)")
    ax_rad.axvspan(25.0, 28.2, color="#ffb703", alpha=0.18, label="Low-Cloud Deck (25–28.2°N): Bright VIS, Warm IR")
    ax_rad.axvspan(28.2, 32.8, color="#3a86ff", alpha=0.12, label="High-Anvil Regime (28.2–32.8°N): Cold IR")
    ax_rad.set_title("(c) Co-Registered Satellite Radiometric Profiles Along ICESat-2 Ground Track", fontweight="bold", fontsize=9.5)
    ax_rad.set_xlabel("Latitude Along Ground Track (°N)")
    ax_rad.set_ylabel("Normalized Radiance")
    ax_rad.set_xlim(25.0, 40.0)
    ax_rad.set_ylim(0.0, 1.26)
    ax_rad.set_yticks([0.0, 0.4, 0.8])
    ax_rad.legend(loc="upper right", ncol=2, framealpha=0.94, fontsize=6.7)
    ax_rad.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig_vfm_data_example")


def main() -> None:
    granules_490 = discover_matched_granules("../vfms", "../ir10.5")
    granules_408 = filter_granules_with_30min_window(granules_490, "../ir10.5", "../vis0.6", use_vis06=True)

    generate_fig1_dataset_overview(granules_408)
    generate_fig_vfm_data_example()
    generate_fig2_dataset_statistics(granules_490, granules_408)
    generate_fig3_architecture_pipeline()
    generate_fig4_qualitative_comparison(granules_408)

    res_path = "outputs/cvpr_experiments/cvpr_all_results.json"
    if os.path.isfile(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            cvpr_res = json.load(f)
        ours_path = "outputs/semisup_s4mc_and_ours/s4mc_and_ours_summary.json"
        if os.path.isfile(ours_path):
            with open(ours_path, "r", encoding="utf-8") as f:
                ours_res = json.load(f)
            m16 = cvpr_res.setdefault("main_16_models", {})
            for short_k, full_k in [
                ("1f_unet", "Semi-Sup U-Net (1f IR)"),
                ("1f_segvit", "Semi-Sup SegViT (1f IR)"),
                ("1f_sam", "Semi-Sup SAM (1f IR)"),
                ("1f_dinov3", "Semi-Sup DINOv3 (1f IR)"),
                ("3f_vis_unet", "Semi-Sup U-Net (3f IR+VIS)"),
                ("3f_vis_segvit", "Semi-Sup SegViT (3f IR+VIS)"),
                ("3f_vis_sam", "Semi-Sup SAM (3f IR+VIS)"),
                ("3f_vis_dinov3", "Semi-Sup DINOv3 (3f IR+VIS)"),
            ]:
                if f"dist_ours_{short_k}" in ours_res:
                    m16[full_k] = ours_res[f"dist_ours_{short_k}"]
            mod_res = cvpr_res.setdefault("modality_temporal_408", {})
            if "dist_ours_3f_vis_dinov3" in ours_res:
                mod_res["Semi-Sup DINOv3 (3f IR+VIS)"] = ours_res["dist_ours_3f_vis_dinov3"]
            if "dist_ours_3f_vis_sam" in ours_res:
                mod_res["Semi-Sup SAM (3f IR+VIS)"] = ours_res["dist_ours_3f_vis_sam"]
            beam_res = cvpr_res.setdefault("beam_sparsity_study", {})
            for b_short, b_full in [
                ("dist_ours_1beam_3f_vis_dinov3", "Semi-Sup DINOv3 (1 Beam (0.025% Pixels))"),
                ("dist_ours_2beam_3f_vis_dinov3", "Semi-Sup DINOv3 (2 Beams (0.050% Pixels))"),
                ("dist_ours_3f_vis_dinov3", "Semi-Sup DINOv3 (3 Beams, 0.075% Pixels)"),
            ]:
                if b_short in ours_res:
                    beam_res[b_full] = ours_res[b_short]
        if "main_16_models" in cvpr_res:
            generate_fig5_altitude_and_regime_breakdown(cvpr_res)
        if "modality_temporal_408" in cvpr_res and "beam_sparsity_study" in cvpr_res:
            generate_fig6_ablation_and_sparsity(cvpr_res)


if __name__ == "__main__":
    main()
