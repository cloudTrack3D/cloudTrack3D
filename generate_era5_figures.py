#!/usr/bin/env python3
"""
Generate publication-quality figures explaining and visualizing the ERA5
atmospheric reanalysis dataset (Temperature and Relative Humidity across the
9 pressure levels: 200, 300, 400, 500, 600, 700, 800, 850, 900 hPa),
shown both alone and co-registered with MTG-I1 satellite imagery and
ICESat-2 ATL09 Cloud Vertical Feature Mask (VFM) labels.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

sys.path.insert(0, "../data")
from era5_db import COMMON_9_PRESSURE_LEVELS, ERA5Database
from dataset import DEFAULT_BBOX, DEFAULT_IMAGE_SHAPE
from losses import THRESHOLD_15KFT_KM

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.2,
    "figure.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Standard barometric altitude approximation (km ASL) for the 9 ERA5 pressure levels (hPa)
# [200, 300, 400, 500, 600, 700, 800, 850, 900]
PRESSURE_HPA = np.array(COMMON_9_PRESSURE_LEVELS, dtype=np.float32)
PRESSURE_ALT_KM = np.array([11.78, 9.16, 7.19, 5.57, 4.21, 3.01, 1.95, 1.46, 0.99], dtype=np.float32)


def _save_both(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


def _load_example_scene():
    """Load the 2025-01-15 12:10 UTC scene: ERA5 (9 levels T & RH), MTG-I1 IR+VIS, and ICESat-2 VFM."""
    db = ERA5Database("../data/era5_npy")
    dt = datetime.datetime(2025, 1, 15, 12, 10, tzinfo=datetime.timezone.utc)
    prof_t = db.load_profile(dt, "temperature", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)      # (9, 61, 77) K
    prof_rh = np.clip(
        db.load_profile(dt, "relative_humidity", COMMON_9_PRESSURE_LEVELS, interpolate_time=True),    # (9, 61, 77) %
        0.0,
        100.0,
    )

    h5_path = "../data/vfms/ATL09_20250115112941_04642601_007_01.h5"
    img_name = "2025-01-15_12-10.png"
    ir_img = np.array(Image.open(f"../data/ir10.5/{img_name}").convert("L"), dtype=np.float32) / 255.0
    vis_img = np.array(Image.open(f"../data/vis0.6/{img_name}").convert("L"), dtype=np.float32) / 255.0

    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX

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
        is_cld = (l_attr == 1) & np.isfinite(l_top) & np.isfinite(l_bot) & (l_top > 0) & (l_top < 20.0)
        n_lay = np.sum(is_cld, axis=1)
        top_all = np.where(is_cld, l_top, np.nan)
        bot_all = np.where(is_cld, l_bot, np.nan)
        with np.errstate(all="ignore"):
            h_top = np.nanmax(top_all, axis=1)
            h_bot = np.nanmin(bot_all, axis=1)

    # Bilinearly interpolate ERA5 (9, 61, 77) along the exact ICESat-2 (lat, lon) ground track
    y = (40.0 - lat) / 15.0 * 60.0
    x = (lon - 24.0) / 19.0 * 76.0
    y0 = np.clip(np.floor(y).astype(int), 0, 59)
    x0 = np.clip(np.floor(x).astype(int), 0, 75)
    wy = (y - y0)[None, :]
    wx = (x - x0)[None, :]

    track_t = (
        (1 - wy) * (1 - wx) * prof_t[:, y0, x0]
        + (1 - wy) * wx * prof_t[:, y0, x0 + 1]
        + wy * (1 - wx) * prof_t[:, y0 + 1, x0]
        + wy * wx * prof_t[:, y0 + 1, x0 + 1]
    )  # (9, N_track)
    track_rh = (
        (1 - wy) * (1 - wx) * prof_rh[:, y0, x0]
        + (1 - wy) * wx * prof_rh[:, y0, x0 + 1]
        + wy * (1 - wx) * prof_rh[:, y0 + 1, x0]
        + wy * wx * prof_rh[:, y0 + 1, x0 + 1]
    )  # (9, N_track)

    return {
        "db_lats": db.latitudes,
        "db_lons": db.longitudes,
        "prof_t": prof_t,
        "prof_rh": prof_rh,
        "ir_img": ir_img,
        "vis_img": vis_img,
        "lat": lat,
        "lon": lon,
        "surf": surf,
        "is_cld": is_cld,
        "n_lay": n_lay,
        "top_all": top_all,
        "bot_all": bot_all,
        "h_top": h_top,
        "h_bot": h_bot,
        "track_t": track_t,
        "track_rh": track_rh,
    }


def generate_fig_era5_vertical_curtain(data: dict) -> None:
    """
    Figure 1: Comprehensive 5-panel visualization of ERA5 Temperature & Relative Humidity
    with and without MTG-I1 satellite imagery and ICESat-2 Cloud VFM labels.
    """
    db_lats = data["db_lats"]
    db_lons = data["db_lons"]
    prof_t = data["prof_t"]
    prof_rh = data["prof_rh"]
    ir_img = data["ir_img"]
    vis_img = data["vis_img"]
    lat = data["lat"]
    lon = data["lon"]
    surf = data["surf"]
    is_cld = data["is_cld"]
    n_lay = data["n_lay"]
    top_all = data["top_all"]
    bot_all = data["bot_all"]
    h_top = data["h_top"]
    h_bot = data["h_bot"]
    track_t = data["track_t"]
    track_rh = data["track_rh"]

    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX
    lon_grid, lat_grid = np.meshgrid(db_lons, db_lats)

    fig = plt.figure(figsize=(15.6, 8.5))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.18], hspace=0.36)
    gs_top = gs_main[0].subgridspec(1, 3, width_ratios=[1.0, 1.08, 1.02], wspace=0.24)
    gs_bot = gs_main[1].subgridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.25)

    # --- Panel (a): ERA5 Alone (850 hPa Low-Level RH + Temperature Isotherms) ---
    ax_a = fig.add_subplot(gs_top[0, 0])
    idx_850 = COMMON_9_PRESSURE_LEVELS.index(850)
    ax_a.pcolormesh(lon_grid, lat_grid, prof_rh[idx_850], cmap="YlGnBu", vmin=0, vmax=100, shading="auto")
    cs_a = ax_a.contour(
        lon_grid,
        lat_grid,
        prof_t[idx_850] - 273.15,
        levels=np.arange(-4, 18, 3),
        colors="#b91c1c",
        linewidths=0.9,
        linestyles="--",
    )
    ax_a.clabel(cs_a, inline=True, fontsize=6.8, fmt="%d°C")
    ax_a.set_title("(a) ERA5 Alone: 850 hPa (~1.5 km)\nRelative Humidity (%) & Temp. Isotherms (°C)", fontweight="bold")
    ax_a.set_xlabel("Longitude (°E)")
    ax_a.set_ylabel("Latitude (°N)")
    ax_a.set_xlim(min_lon, max_lon)
    ax_a.set_ylim(min_lat, max_lat)

    # --- Panel (b): ERA5 Alone (300 hPa Upper-Level RH + Temperature Isotherms) ---
    ax_b = fig.add_subplot(gs_top[0, 1])
    idx_300 = COMMON_9_PRESSURE_LEVELS.index(300)
    im_b = ax_b.pcolormesh(lon_grid, lat_grid, prof_rh[idx_300], cmap="YlGnBu", vmin=0, vmax=100, shading="auto")
    cs_b = ax_b.contour(
        lon_grid,
        lat_grid,
        prof_t[idx_300] - 273.15,
        levels=np.arange(-50, -30, 3),
        colors="#b91c1c",
        linewidths=0.9,
        linestyles="--",
    )
    ax_b.clabel(cs_b, inline=True, fontsize=6.8, fmt="%d°C")
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.044, pad=0.025)
    cb_b.set_label("ERA5 Relative Humidity (%)", fontsize=7.6)
    ax_b.set_title("(b) ERA5 Alone: 300 hPa (~9.2 km)\nRelative Humidity (%) & Temp. Isotherms (°C)", fontweight="bold")
    ax_b.set_xlabel("Longitude (°E)")
    ax_b.set_xlim(min_lon, max_lon)
    ax_b.set_ylim(min_lat, max_lat)

    # --- Panel (c): Combined Satellite + ERA5 Moisture Contours + ICESat-2 Cloud Labels ---
    ax_c = fig.add_subplot(gs_top[0, 2])
    ir_sub = ir_img[::4, ::4]
    vis_sub = vis_img[::4, ::4]
    rgb_bg = np.stack([
        np.clip(0.55 * vis_sub + 0.45 * ir_sub, 0, 1),
        np.clip(0.65 * vis_sub + 0.35 * ir_sub, 0, 1),
        np.clip(0.45 * vis_sub + 0.65 * ir_sub + 0.08, 0, 1),
    ], axis=-1)
    ax_c.imshow(rgb_bg, extent=[min_lon, max_lon, min_lat, max_lat], aspect="auto")

    idx_700 = COMMON_9_PRESSURE_LEVELS.index(700)
    cs_c = ax_c.contour(
        lon_grid,
        lat_grid,
        prof_rh[idx_700],
        levels=[70.0, 85.0],
        colors=["#38bdf8", "#0284c7"],
        linewidths=[1.0, 1.4],
        linestyles=["--", "-"],
    )
    ax_c.clabel(cs_c, inline=True, fontsize=6.5, fmt="RH %d%%")

    ax_c.plot(lon, lat, color="black", linewidth=4.2, alpha=0.78, zorder=3)
    m_clear = n_lay == 0
    m_low = (n_lay == 1) & (h_top <= THRESHOLD_15KFT_KM)
    m_high = (n_lay == 1) & (h_top > THRESHOLD_15KFT_KM)
    m_multi = n_lay >= 2

    ax_c.scatter(lon[m_clear], lat[m_clear], c="#00b4d8", s=4.5, label="Clear Sky", zorder=4)
    ax_c.scatter(lon[m_low], lat[m_low], c="#ffb703", s=6.0, label="Low Cloud (≤ 15 kft)", zorder=5)
    ax_c.scatter(lon[m_high], lat[m_high], c="#e63946", s=6.0, label="High Cloud (> 15 kft)", zorder=6)
    ax_c.scatter(lon[m_multi], lat[m_multi], c="#b5179e", s=7.0, label="Multi-Layer (≥ 2)", zorder=7)

    for pt_lbl, t_lat in [("P1", 26.2), ("P2", 28.8), ("P3", 38.8)]:
        idx_p = int(np.argmin(np.abs(lat - t_lat)))
        ax_c.scatter([lon[idx_p]], [lat[idx_p]], c="yellow", edgecolor="black", s=42, zorder=9, linewidth=1.1)
        ax_c.text(
            lon[idx_p] + 0.45,
            lat[idx_p] - 0.2,
            f"{pt_lbl} ({t_lat:.1f}°N)",
            color="yellow",
            fontweight="bold",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.72, edgecolor="none"),
            zorder=10,
        )

    ax_c.set_title("(c) Satellite Imagery + ERA5 700 hPa RH\n+ ICESat-2 Cloud VFM Track Overlay", fontweight="bold")
    ax_c.set_xlabel("Longitude (°E)")
    ax_c.set_xlim(min_lon, max_lon)
    ax_c.set_ylim(min_lat, max_lat)
    ax_c.legend(loc="lower left", framealpha=0.90, facecolor="white", markerscale=1.4, fontsize=6.6)

    # --- Panel (d): ERA5-Only Vertical Cross-Section Along Track (No Cloud Labels) ---
    ax_d = fig.add_subplot(gs_bot[0, 0])
    lat_2d, alt_2d = np.meshgrid(lat, PRESSURE_ALT_KM)
    ax_d.contourf(lat_2d, alt_2d, track_rh, levels=np.linspace(0, 100, 21), cmap="YlGnBu", vmin=0, vmax=100)
    cs_d = ax_d.contour(
        lat_2d,
        alt_2d,
        track_t - 273.15,
        levels=[-45, -30, -15, 0, 10],
        colors=["#7f1d1d", "#991b1b", "#b91c1c", "#dc2626", "#ef4444"],
        linewidths=[1.0, 1.0, 1.0, 1.6, 1.0],
        linestyles=["--", "--", "--", "-", "--"],
    )
    ax_d.clabel(cs_d, inline=True, fontsize=7.0, fmt="%d°C (ERA5 T)")
    for alt_lvl, p_lvl in zip(PRESSURE_ALT_KM, PRESSURE_HPA):
        ax_d.axhline(alt_lvl, color="gray", linestyle=":", linewidth=0.5, alpha=0.55)
        ax_d.text(40.12, alt_lvl, f"{int(p_lvl)} hPa", va="center", ha="left", fontsize=6.6, color="#334155")

    ax_d.fill_between(lat, 0, surf, color="#8d6e63", alpha=0.65, label="Terrain Elevation")
    ax_d.set_title(
        "(d) ERA5 Alone Along Orbit Track (Without Satellite or Lidar)\n9-Level Vertical RH (%) & Temperature (°C) Cross-Section",
        fontweight="bold",
    )
    ax_d.set_xlabel("Latitude Along ICESat-2 Ground Track (°N)")
    ax_d.set_ylabel("Standard Pressure Altitude (km ASL)")
    ax_d.set_xlim(25.0, 40.0)
    ax_d.set_ylim(0.0, 13.0)

    # --- Panel (e): ERA5 Vertical Cross-Section + ICESat-2 Cloud VFM Labels Overlay ---
    ax_e = fig.add_subplot(gs_bot[0, 1])
    cf_e = ax_e.contourf(
        lat_2d, alt_2d, track_rh, levels=np.linspace(0, 100, 21), cmap="YlGnBu", vmin=0, vmax=100, alpha=0.82
    )
    cs_e = ax_e.contour(
        lat_2d,
        alt_2d,
        track_t - 273.15,
        levels=[-45, -30, -15, 0, 10],
        colors="#991b1b",
        linewidths=0.9,
        linestyles="--",
        alpha=0.75,
    )
    ax_e.clabel(cs_e, inline=True, fontsize=6.6, fmt="%d°C")

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

    ax_e.vlines(l0_low_lat, l0_low_b, l0_low_t, color="#ffb703", alpha=0.70, linewidth=1.3, zorder=3)
    ax_e.vlines(l0_high_lat, l0_high_b, l0_high_t, color="#f97316", alpha=0.60, linewidth=1.3, zorder=3)
    ax_e.vlines(multi_lat, multi_b, multi_t, color="#d946ef", alpha=0.78, linewidth=1.4, zorder=4)

    cld_mask = n_lay >= 1
    ax_e.scatter(lat[cld_mask], h_top[cld_mask], c="#dc2626", s=4.0, zorder=5)
    ax_e.scatter(lat[cld_mask], h_bot[cld_mask], c="#1e3a8a", s=4.0, zorder=5)
    ax_e.fill_between(lat, 0, surf, color="#8d6e63", alpha=0.75, zorder=4)
    ax_e.axhline(THRESHOLD_15KFT_KM, color="#ea580c", linestyle="-.", linewidth=1.4, zorder=6)

    ax_e.annotate(
        "ERA5 Low-Level RH > 85%\n+ Dry Cap Constrains Stratus ≤ 3.5 km",
        xy=(26.2, 2.8),
        xytext=(25.2, 5.7),
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2),
        fontsize=6.8,
        fontweight="bold",
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.92, edgecolor="#f59e0b"),
        zorder=8,
    )
    ax_e.annotate(
        "Deep Saturation (850–250 hPa)\nMatches Multi-Layer Frontal Deck",
        xy=(38.8, 7.2),
        xytext=(33.4, 9.6),
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
        fontsize=6.8,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.92, edgecolor="#dc2626"),
        zorder=8,
    )

    handles_e = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=5, label="Lidar Cloud Top ($h_{\\mathrm{top}}$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1e3a8a", markersize=5, label="Lidar Cloud Base ($h_{\\mathrm{bot}}$)"),
        mpatches.Patch(facecolor="#ffb703", alpha=0.7, label="Low Cloud VFM (≤ 15 kft)"),
        mpatches.Patch(facecolor="#f97316", alpha=0.6, label="Upper Cloud VFM (> 15 kft)"),
        mpatches.Patch(facecolor="#d946ef", alpha=0.75, label="Multi-Layer VFM (≥ 2)"),
        Line2D([0], [0], color="#991b1b", lw=1.0, ls="--", label="ERA5 Temp. Isotherms (°C)"),
    ]
    ax_e.legend(handles=handles_e, loc="upper left", ncol=2, framealpha=0.92, fontsize=6.4)
    cb_e = fig.colorbar(cf_e, ax=ax_e, fraction=0.038, pad=0.02)
    cb_e.set_label("ERA5 Relative Humidity (%)", fontsize=7.8)
    ax_e.set_title(
        "(e) ERA5 Cross-Section + ICESat-2 Cloud VFM Labels Overlay\nThermodynamic Moisture & Isotherm Alignment with Cloud Layers",
        fontweight="bold",
    )
    ax_e.set_xlabel("Latitude Along ICESat-2 Ground Track (°N)")
    ax_e.set_ylabel("Altitude Above Sea Level (km)")
    ax_e.set_xlim(25.0, 40.0)
    ax_e.set_ylim(0.0, 13.0)

    _save_both(fig, "fig_era5_vertical_curtain")


def generate_fig_era5_9levels_and_soundings(data: dict) -> None:
    """
    Figure 2: Multi-Level Atlas & Vertical Atmospheric Soundings (P1, P2, P3, and Clear-Sky/Regime Mean).
    """
    db_lats = data["db_lats"]
    db_lons = data["db_lons"]
    prof_t = data["prof_t"]
    prof_rh = data["prof_rh"]
    lat = data["lat"]
    is_cld = data["is_cld"]
    top_all = data["top_all"]
    bot_all = data["bot_all"]
    track_t = data["track_t"]
    track_rh = data["track_rh"]

    lon_grid, lat_grid = np.meshgrid(db_lons, db_lats)
    shown_levels = [900, 700, 500, 300, 200]

    fig = plt.figure(figsize=(15.6, 8.8))
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.38], wspace=0.26)
    gs_left = gs_main[0].subgridspec(5, 2, hspace=0.46, wspace=0.32)
    gs_right = gs_main[1].subgridspec(2, 2, hspace=0.58, wspace=0.38)

    # Left 5x2 Grid: 5 pressure levels of Relative Humidity (%) and Temperature (°C)
    for row_i, p_lvl in enumerate(shown_levels):
        idx_l = COMMON_9_PRESSURE_LEVELS.index(p_lvl)
        alt_km = PRESSURE_ALT_KM[idx_l]

        ax_rh = fig.add_subplot(gs_left[row_i, 0])
        im_rh = ax_rh.pcolormesh(lon_grid, lat_grid, prof_rh[idx_l], cmap="YlGnBu", vmin=0, vmax=100, shading="auto")
        ax_rh.plot(data["lon"], data["lat"], color="#dc2626", lw=1.2, alpha=0.85)
        ax_rh.set_title(f"ERA5 RH @ {p_lvl} hPa (~{alt_km:.1f} km)", fontweight="bold", fontsize=7.9)
        if row_i == 4:
            ax_rh.set_xlabel("Longitude (°E)", fontsize=7.5)
        else:
            ax_rh.set_xticklabels([])
        ax_rh.set_ylabel("Lat (°N)", fontsize=7.5)
        cb1 = fig.colorbar(im_rh, ax=ax_rh, fraction=0.045, pad=0.02)
        cb1.ax.tick_params(labelsize=6.4)

        ax_t = fig.add_subplot(gs_left[row_i, 1])
        t_c = prof_t[idx_l] - 273.15
        im_t = ax_t.pcolormesh(lon_grid, lat_grid, t_c, cmap="coolwarm", shading="auto")
        ax_t.plot(data["lon"], data["lat"], color="black", lw=1.2, alpha=0.85)
        ax_t.set_title(f"ERA5 Temp @ {p_lvl} hPa ({t_c.min():.0f}..{t_c.max():.0f}°C)", fontweight="bold", fontsize=7.9)
        if row_i == 4:
            ax_t.set_xlabel("Longitude (°E)", fontsize=7.5)
        else:
            ax_t.set_xticklabels([])
        ax_t.set_yticklabels([])
        cb2 = fig.colorbar(im_t, ax=ax_t, fraction=0.045, pad=0.02)
        cb2.ax.tick_params(labelsize=6.4)

    # Right 2x2 Grid: 3 Vertical Soundings (P1, P2, P3) + Clear-Sky vs Cloud Regime Comparison
    sounding_specs = [
        ("P1 (25.6°N, 34.9°E): Low Stratus Beneath Dry Cap", 25.6, gs_right[0, 0], "#ffb703"),
        ("P2 (28.8°N, 35.3°E): High Cirrus Over Low Deck", 28.8, gs_right[0, 1], "#8b5cf6"),
        ("P3 (38.8°N, 36.4°E): Deep Multi-Layer Frontal System", 38.8, gs_right[1, 0], "#ef4444"),
    ]

    for title_str, target_lat, sub_spec, shade_col in sounding_specs:
        ax_s = fig.add_subplot(sub_spec)
        near_idx = np.where(np.abs(lat - target_lat) <= 0.12)[0]
        center_i = int(np.argmin(np.abs(lat - target_lat)))

        rh_prof = track_rh[:, center_i]
        t_prof_c = track_t[:, center_i] - 273.15

        ax_s.plot(rh_prof, PRESSURE_ALT_KM, "-o", color="#0284c7", linewidth=2.0, markersize=4.5, label="ERA5 RH (%)")
        ax_s.axvline(75.0, color="#0284c7", linestyle=":", linewidth=1.0, alpha=0.7, label="75% RH Threshold")

        ax_t2 = ax_s.twiny()
        ax_t2.plot(
            t_prof_c, PRESSURE_ALT_KM, "--s", color="#dc2626", linewidth=1.8, markersize=4.0, label="ERA5 Temp (°C)"
        )
        ax_t2.set_xlabel("ERA5 Temperature (°C)", color="#dc2626", fontsize=7.4)
        ax_t2.tick_params(axis="x", colors="#dc2626", labelsize=6.8)
        ax_t2.set_xlim(-60, 25)

        intervals = []
        for idx_shot in near_idx:
            for l_i in np.where(is_cld[idx_shot])[0]:
                intervals.append((bot_all[idx_shot, l_i], top_all[idx_shot, l_i]))
        if intervals:
            intervals.sort()
            merged = [list(intervals[0])]
            for b_v, t_v in intervals[1:]:
                if b_v <= merged[-1][1] + 0.25:
                    merged[-1][1] = max(merged[-1][1], t_v)
                else:
                    merged.append([b_v, t_v])
            for m_idx, (b_v, t_v) in enumerate(merged):
                lbl = "ICESat-2 Cloud Layer ($[h_{\\mathrm{bot}}, h_{\\mathrm{top}}]$)" if m_idx == 0 else None
                ax_s.axhspan(b_v, t_v, color=shade_col, alpha=0.30, label=lbl)
                ax_s.axhline(t_v, color="#b91c1c", linestyle="-", linewidth=1.1, alpha=0.8)
                ax_s.axhline(b_v, color="#1e3a8a", linestyle="-", linewidth=1.1, alpha=0.8)

        for alt_lvl, p_lvl in zip(PRESSURE_ALT_KM, PRESSURE_HPA):
            ax_s.text(101.5, alt_lvl, f"{int(p_lvl)}", va="center", ha="left", fontsize=6.1, color="#475569")

        ax_s.set_title(title_str, fontweight="bold", fontsize=8.0, pad=22)
        ax_s.set_xlabel("ERA5 Relative Humidity (%)", color="#0284c7", fontsize=7.5)
        ax_s.set_ylabel("Altitude (km ASL) / hPa", fontsize=7.5)
        ax_s.set_xlim(0, 100)
        ax_s.set_ylim(0, 12.8)
        ax_s.grid(True, linestyle=":", alpha=0.5)
        ax_s.legend(loc="lower left", fontsize=6.3, framealpha=0.92)

    # Bottom-Right Panel (gs_right[1, 1]): Regime Mean Vertical Profiles
    ax_stat = fig.add_subplot(gs_right[1, 1])
    clear_shots = (data["n_lay"] == 0) & (lat >= 31.0) & (lat <= 35.0)
    low_stratus_shots = (data["n_lay"] == 1) & (lat <= 26.8)
    deep_frontal_shots = (data["n_lay"] >= 2) & (lat >= 37.5)

    rh_clear_mean = track_rh[:, clear_shots].mean(axis=1)
    rh_low_mean = track_rh[:, low_stratus_shots].mean(axis=1)
    rh_high_mean = track_rh[:, deep_frontal_shots].mean(axis=1)

    ax_stat.plot(
        rh_clear_mean, PRESSURE_ALT_KM, "-o", color="#00b4d8", lw=2.0, label="Clear-Sky Subsidence (31–35°N)"
    )
    ax_stat.plot(
        rh_low_mean, PRESSURE_ALT_KM, "-s", color="#f59e0b", lw=2.0, label="Low Marine Stratus (≤ 15 kft, 25–26.8°N)"
    )
    ax_stat.plot(
        rh_high_mean, PRESSURE_ALT_KM, "-^", color="#e11d48", lw=2.0, label="Deep Frontal Multi-Layer (37.5–40°N)"
    )
    ax_stat.axhline(THRESHOLD_15KFT_KM, color="#ea580c", ls="--", lw=1.3, label="15,000 ft (4.57 km) Split")
    for alt_lvl, p_lvl in zip(PRESSURE_ALT_KM, PRESSURE_HPA):
        ax_stat.axhline(alt_lvl, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax_stat.text(101.5, alt_lvl, f"{int(p_lvl)} hPa", va="center", ha="left", fontsize=6.1, color="#334155")

    ax_stat.set_title(
        "Mean 9-Level ERA5 Relative Humidity Profile\nAcross Distinct Cloud Regimes Along Track",
        fontweight="bold",
        fontsize=8.0,
        pad=10,
    )
    ax_stat.set_xlabel("Mean ERA5 Relative Humidity (%)", fontsize=7.5)
    ax_stat.set_ylabel("Altitude (km ASL)", fontsize=7.5)
    ax_stat.set_xlim(0, 100)
    ax_stat.set_ylim(0, 12.8)
    ax_stat.grid(True, linestyle=":", alpha=0.5)
    ax_stat.legend(loc="upper right", fontsize=6.3, framealpha=0.92)

    _save_both(fig, "fig_era5_9levels_atlas")


if __name__ == "__main__":
    scene_data = _load_example_scene()
    generate_fig_era5_vertical_curtain(scene_data)
    generate_fig_era5_9levels_and_soundings(scene_data)
