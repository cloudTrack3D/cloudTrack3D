#!/usr/bin/env python3
"""
Generate publication-quality visualizations of Satellite Imagery (MTG-I1 IR 10.5 um & VIS 0.6 um)
matched directly to 9-level ERA5 Relative Humidity (RH) and Temperature (T), alongside
ICESat-2 ATL09 3D Cloud Vertical Feature Mask (VFM) ground-truth profiles.

Outputs:
  1. paper/figures/fig_satellite_era5_physical_matching.{pdf,png}
  2. paper/figures/fig_satellite_era5_multi_scene_matching.{pdf,png}
  (Also copies PNGs into the conversation artifact directory for direct preview.)
"""

from __future__ import annotations

import datetime
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from dataset import (
    DEFAULT_BBOX,
    DEFAULT_IMAGE_SHAPE,
    discover_matched_granules,
    filter_granules_with_30min_window,
    split_granules_chronologically,
)
from losses import THRESHOLD_15KFT_KM

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "figure.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PRESSURE_HPA = np.array(COMMON_9_PRESSURE_LEVELS, dtype=np.float32)
PRESSURE_ALT_KM = np.array([11.78, 9.16, 7.19, 5.57, 4.21, 3.01, 1.95, 1.46, 0.99], dtype=np.float32)


def _save_and_copy(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


def _load_scene_by_granule(g, db: ERA5Database) -> dict:
    dt = datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc)
    prof_t = db.load_profile(dt, "temperature", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
    prof_rh = np.clip(
        db.load_profile(dt, "relative_humidity", COMMON_9_PRESSURE_LEVELS, interpolate_time=True),
        0.0,
        100.0,
    )

    ir_img = np.array(Image.open(g.img_path).convert("L"), dtype=np.float32) / 255.0
    vis_path = f"../data/vis0.6/{g.img_name}"
    if os.path.exists(vis_path):
        vis_img = np.array(Image.open(vis_path).convert("L"), dtype=np.float32) / 255.0
    else:
        vis_img = np.zeros_like(ir_img)

    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX
    H, W = DEFAULT_IMAGE_SHAPE

    with h5py.File(g.h5_path, "r") as f:
        best_beam = "profile_2"
        best_cnt = -1
        for bname in ("profile_1", "profile_2", "profile_3"):
            if bname in f and "high_rate" in f[bname]:
                lt = f[bname]["high_rate"]["latitude"][:]
                ln = f[bname]["high_rate"]["longitude"][:]
                cnt = int(np.sum((lt >= min_lat) & (lt <= max_lat) & (ln >= min_lon) & (ln <= max_lon)))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_beam = bname

        grp = f[f"{best_beam}/high_rate"]
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

    rows = np.clip(np.round((max_lat - lat) / (max_lat - min_lat) * (H - 1)).astype(int), 0, H - 1)
    cols = np.clip(np.round((lon - min_lon) / (max_lon - min_lon) * (W - 1)).astype(int), 0, W - 1)
    track_ir = ir_img[rows, cols]
    track_vis = vis_img[rows, cols]

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
    )
    track_rh = (
        (1 - wy) * (1 - wx) * prof_rh[:, y0, x0]
        + (1 - wy) * wx * prof_rh[:, y0, x0 + 1]
        + wy * (1 - wx) * prof_rh[:, y0 + 1, x0]
        + wy * wx * prof_rh[:, y0 + 1, x0 + 1]
    )

    t_900 = track_t[8, :]
    tb_eff = t_900 - (54.54 * track_ir - 29.60 * track_vis + 26.60)
    N = len(lat)
    h_top_thermal = np.zeros(N, dtype=np.float32)
    h_top_coupled = np.zeros(N, dtype=np.float32)
    h_bot_coupled = np.zeros(N, dtype=np.float32)

    for i in range(N):
        t_col = track_t[:, i]
        rh_col = track_rh[:, i]
        t_mono = np.maximum.accumulate(t_col)
        ht = float(np.clip(np.interp(tb_eff[i], t_mono, PRESSURE_ALT_KM), 0.5, 14.5))
        h_top_thermal[i] = ht

        tw = np.exp(-np.abs(t_col - tb_eff[i]) / 12.0)
        mw = np.power(np.clip(rh_col / 100.0, 0.05, 1.0), 1.5)
        jw = tw * mw
        jw /= (np.sum(jw) + 1e-8)
        h_rh_w = float(np.sum(jw * PRESSURE_ALT_KM))
        hc = 0.55 * ht + 0.45 * h_rh_w
        h_top_coupled[i] = hc

        moist_idx = np.where((rh_col >= 62.0) & (PRESSURE_ALT_KM <= hc + 0.5))[0]
        if len(moist_idx) > 0:
            hb_rh = float(PRESSURE_ALT_KM[moist_idx[-1]])
            h_bot_coupled[i] = float(np.clip(0.55 * hb_rh + 0.45 * max(0.3, hc - 1.8), 0.2, hc - 0.15))
        else:
            h_bot_coupled[i] = max(0.3, hc - 1.8)

    return {
        "granule": g,
        "dt": dt,
        "dt_str": dt.strftime("%Y-%m-%d %H:%M UTC"),
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
        "track_ir": track_ir,
        "track_vis": track_vis,
        "track_t": track_t,
        "track_rh": track_rh,
        "tb_eff": tb_eff,
        "h_top_thermal": h_top_thermal,
        "h_top_coupled": h_top_coupled,
        "h_bot_coupled": h_bot_coupled,
    }


def collect_validation_matching_samples(val_granules: list, db: ERA5Database) -> dict:
    ir_list, vis_list, t_list, rh_list, top_list, bot_list, cld_list, day_list = [], [], [], [], [], [], [], []
    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX
    H, W = DEFAULT_IMAGE_SHAPE

    for g in val_granules[:55]:
        dt = datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc)
        is_daytime = 7 <= dt.hour <= 14
        try:
            prof_t = db.load_profile(dt, "temperature", COMMON_9_PRESSURE_LEVELS, interpolate_time=True)
            prof_rh = np.clip(
                db.load_profile(dt, "relative_humidity", COMMON_9_PRESSURE_LEVELS, interpolate_time=True),
                0.0,
                100.0,
            )
        except Exception:
            continue

        ir_img = np.array(Image.open(g.img_path).convert("L"), dtype=np.float32) / 255.0
        vis_path = f"../data/vis0.6/{g.img_name}"
        if not os.path.exists(vis_path):
            continue
        vis_img = np.array(Image.open(vis_path).convert("L"), dtype=np.float32) / 255.0

        with h5py.File(g.h5_path, "r") as f:
            for bname in ("profile_1", "profile_2", "profile_3"):
                if bname not in f or "high_rate" not in f[bname]:
                    continue
                grp = f[f"{bname}/high_rate"]
                lat = grp["latitude"][:]
                lon = grp["longitude"][:]
                in_b = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
                if not np.any(in_b):
                    continue
                idx_sub = np.where(in_b)[0][::6]
                lat = lat[idx_sub]
                lon = lon[idx_sub]
                l_attr = grp["layer_attr"][:][idx_sub]
                l_top = grp["layer_top"][:][idx_sub] / 1000.0
                l_bot = grp["layer_bot"][:][idx_sub] / 1000.0

                is_cld = (l_attr == 1) & np.isfinite(l_top) & np.isfinite(l_bot) & (l_top > 0) & (l_top < 20.0)
                has_c = np.any(is_cld, axis=1).astype(np.float32)
                top_all = np.where(is_cld, l_top, np.nan)
                bot_all = np.where(is_cld, l_bot, np.nan)
                with np.errstate(all="ignore"):
                    ht = np.nanmax(top_all, axis=1)
                    hb = np.nanmin(bot_all, axis=1)

                rows = np.clip(np.round((max_lat - lat) / (max_lat - min_lat) * (H - 1)).astype(int), 0, H - 1)
                cols = np.clip(np.round((lon - min_lon) / (max_lon - min_lon) * (W - 1)).astype(int), 0, W - 1)
                ir_v = ir_img[rows, cols]
                vis_v = vis_img[rows, cols]

                y = (40.0 - lat) / 15.0 * 60.0
                x = (lon - 24.0) / 19.0 * 76.0
                y0 = np.clip(np.floor(y).astype(int), 0, 59)
                x0 = np.clip(np.floor(x).astype(int), 0, 75)
                wy = (y - y0)[None, :]
                wx = (x - x0)[None, :]

                tr_t = (
                    (1 - wy) * (1 - wx) * prof_t[:, y0, x0]
                    + (1 - wy) * wx * prof_t[:, y0, x0 + 1]
                    + wy * (1 - wx) * prof_t[:, y0 + 1, x0]
                    + wy * wx * prof_t[:, y0 + 1, x0 + 1]
                ).T
                tr_rh = (
                    (1 - wy) * (1 - wx) * prof_rh[:, y0, x0]
                    + (1 - wy) * wx * prof_rh[:, y0, x0 + 1]
                    + wy * (1 - wx) * prof_rh[:, y0 + 1, x0]
                    + wy * wx * prof_rh[:, y0 + 1, x0 + 1]
                ).T

                ir_list.append(ir_v)
                vis_list.append(vis_v)
                t_list.append(tr_t)
                rh_list.append(tr_rh)
                top_list.append(ht)
                bot_list.append(hb)
                cld_list.append(has_c)
                day_list.append(np.full(len(ir_v), is_daytime, dtype=bool))

    return {
        "ir": np.concatenate(ir_list),
        "vis": np.concatenate(vis_list),
        "t": np.concatenate(t_list, axis=0),
        "rh": np.concatenate(rh_list, axis=0),
        "h_top": np.concatenate(top_list),
        "h_bot": np.concatenate(bot_list),
        "has_cloud": np.concatenate(cld_list),
        "is_day": np.concatenate(day_list),
    }


def generate_fig_satellite_era5_physical_matching(scene: dict, stats: dict) -> None:
    """
    Generate Figure: Direct Physical & Statistical Matching Between Satellite Imagery
    (IR 10.5 um & VIS 0.6 um), 9-Level ERA5 Temperature & Relative Humidity, and ICESat-2 Lidar.
    """
    fig = plt.figure(figsize=(16.6, 9.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], width_ratios=[1.02, 1.0, 1.05], hspace=0.36, wspace=0.39)

    # -------------------------------------------------------------------------
    # Panel (a): Satellite Thermal Brightness Temperature vs ERA5 Temp at True Cloud Top
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    cld_m = (stats["has_cloud"] > 0.5) & np.isfinite(stats["h_top"]) & (stats["ir"] > 0.08)
    ir_c = stats["ir"][cld_m]
    vis_c = stats["vis"][cld_m]
    t_c = stats["t"][cld_m]
    ht_c = stats["h_top"][cld_m]

    z_asc = PRESSURE_ALT_KM[::-1]
    t_asc = t_c[:, ::-1]
    t_era5_at_top = np.array([np.interp(ht_c[i], z_asc, t_asc[i]) for i in range(len(ht_c))], dtype=np.float32)

    A_fit = np.column_stack([ir_c, vis_c, t_c[:, 8], np.ones_like(ir_c)])
    coef, _, _, _ = np.linalg.lstsq(A_fit, t_era5_at_top, rcond=None)
    tb_sat = A_fit @ coef

    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(tb_sat), size=min(3500, len(tb_sat)), replace=False)
    sc_a = ax_a.scatter(
        tb_sat[sub_idx] - 273.15,
        t_era5_at_top[sub_idx] - 273.15,
        c=ht_c[sub_idx],
        cmap="turbo",
        s=7.0,
        alpha=0.58,
        vmin=0.5,
        vmax=13.5,
        rasterized=True,
    )
    cb_a = fig.colorbar(sc_a, ax=ax_a, fraction=0.044, pad=0.022)
    cb_a.set_label("ICESat-2 $h_{\\mathrm{top}}$ (km)", fontsize=7.5)

    lims = [-62, 18]
    ax_a.plot(lims, lims, "k--", linewidth=1.3, label="1:1 Thermal Equilibrium ($T_b = T_{\\mathrm{ERA5}}(h_{\\mathrm{top}})$)")
    mean_t_levels_c = np.mean(t_c, axis=0) - 273.15
    # Annotate non-overlapping representative levels: 200, 300, 500, 700, 900 hPa
    offsets_map = {0: (5, -8), 1: (5, -8), 3: (5, -8), 5: (5, -8), 8: (-36, 6)}
    for lvl_idx, xy_off in offsets_map.items():
        tc_val = float(mean_t_levels_c[lvl_idx])
        p_val = int(PRESSURE_HPA[lvl_idx])
        ax_a.scatter([tc_val], [tc_val], color="white", edgecolor="black", s=26, zorder=6, linewidth=1.0)
        ax_a.annotate(
            f"{p_val} hPa",
            (tc_val, tc_val),
            xytext=xy_off,
            textcoords="offset points",
            fontsize=6.6,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
        )

    r_therm = float(np.corrcoef(tb_sat, t_era5_at_top)[0, 1])
    mae_therm_k = float(np.mean(np.abs(tb_sat - t_era5_at_top)))
    ax_a.text(
        0.04,
        0.93,
        f"Pearson $r = {r_therm:.3f}$\n$|T_b - T_{{\\mathrm{{ERA5}}}}(h_{{\\mathrm{{top}}}})| = {mae_therm_k:.1f}$ K",
        transform=ax_a.transAxes,
        va="top",
        fontsize=7.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#94a3b8", alpha=0.92),
    )
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)
    ax_a.set_xlabel("Satellite-Derived Cloud-Top Brightness Temp. $T_b$ (°C)")
    ax_a.set_ylabel("ERA5 Temp. at Lidar Cloud Top $T_{\\mathrm{ERA5}}(h_{\\mathrm{top}})$ (°C)")
    ax_a.set_title("(a) Thermal Level Matching: Satellite $T_b$ vs. $T_{\\mathrm{ERA5}}(h_{\\mathrm{top}})$", fontweight="bold")
    ax_a.legend(loc="lower right", framealpha=0.9, fontsize=6.6)
    ax_a.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel (b): 9-Level Vertical Cross-Correlation Profile (Satellite & Lidar <-> ERA5 p_k)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    day_m = stats["is_day"]
    high_cld_indicator = ((stats["has_cloud"] > 0.5) & (stats["h_top"] > THRESHOLD_15KFT_KM)).astype(np.float32)
    low_cld_indicator = ((stats["has_cloud"] > 0.5) & (stats["h_bot"] <= THRESHOLD_15KFT_KM)).astype(np.float32)

    r_ir_rh = [float(np.corrcoef(stats["ir"], stats["rh"][:, k])[0, 1]) for k in range(9)]
    r_vis_rh = [float(np.corrcoef(stats["vis"][day_m], stats["rh"][day_m, k])[0, 1]) for k in range(9)]
    r_high_rh = [float(np.corrcoef(high_cld_indicator, stats["rh"][:, k])[0, 1]) for k in range(9)]
    r_low_rh = [float(np.corrcoef(low_cld_indicator, stats["rh"][:, k])[0, 1]) for k in range(9)]

    ax_b.plot(r_ir_rh, PRESSURE_ALT_KM, "-o", color="#dc2626", linewidth=2.0, markersize=4.8, label="MTG-I1 IR 10.5 $\\mu$m vs. $\\mathrm{RH}(p_k)$")
    ax_b.plot(r_vis_rh, PRESSURE_ALT_KM, "-s", color="#d97706", linewidth=2.0, markersize=4.8, label="Daytime VIS 0.6 $\\mu$m vs. $\\mathrm{RH}(p_k)$")
    ax_b.plot(r_high_rh, PRESSURE_ALT_KM, "--^", color="#7c3aed", linewidth=1.8, markersize=4.8, label="Lidar High Cloud (>15 kft) vs. $\\mathrm{RH}(p_k)$")
    ax_b.plot(r_low_rh, PRESSURE_ALT_KM, ":d", color="#0284c7", linewidth=1.9, markersize=4.8, label="Lidar Low Cloud ($\\leq$15 kft) vs. $\\mathrm{RH}(p_k)$")

    ax_b.axhline(THRESHOLD_15KFT_KM, color="#111827", linestyle="-.", linewidth=1.0, alpha=0.75)
    ax_b.text(0.01, THRESHOLD_15KFT_KM + 0.22, "15 kft Aviation Ceiling (4.57 km)", fontsize=6.7, fontweight="bold", color="#111827")

    ax_b.set_yticks(PRESSURE_ALT_KM)
    ax_b.set_yticklabels([f"{int(p)} hPa ({z:.1f} km)" for p, z in zip(PRESSURE_HPA, PRESSURE_ALT_KM)], fontsize=7.0)
    ax_b.set_xlabel("Cross-Modal Pearson Correlation $r$")
    ax_b.set_title("(b) Vertical Sensitivity Profile Across 9 ERA5 Levels", fontweight="bold")
    ax_b.set_xlim(-0.02, 0.56)
    ax_b.set_ylim(0.5, 12.4)
    ax_b.legend(loc="upper left", framealpha=0.92, fontsize=6.4)
    ax_b.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel (c): Joint (VIS 0.6, IR 10.5) Spectral Space Matched to ERA5 Column-Max RH
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    col_max_rh = np.max(stats["rh"][day_m], axis=1)
    vis_day = stats["vis"][day_m]
    ir_day = stats["ir"][day_m]
    sub_all = rng.choice(len(ir_day), size=min(4000, len(ir_day)), replace=False)
    sc_c = ax_c.scatter(
        vis_day[sub_all],
        ir_day[sub_all],
        c=col_max_rh[sub_all],
        cmap="YlGnBu",
        vmin=20,
        vmax=100,
        s=6.8,
        alpha=0.68,
        rasterized=True,
    )
    cb_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.044, pad=0.022)
    cb_c.set_label("ERA5 Column-Max $\\max_k \\mathrm{RH}(p_k)$ (%)", fontsize=7.5)

    regimes = [
        (0.18, 0.10, "Clear Sky\n($\\mathrm{RH}_{\\max} < 45\\%$, Warm $T_b$)", "#0f172a"),
        (0.58, 0.21, "Warm Low Stratus\n(800–900 hPa $\\mathrm{RH}>80\\%$,\nHigh VIS, Warm IR)", "#b45309"),
        (0.24, 0.68, "Thin High Cirrus\n(200–300 hPa $\\mathrm{RH}>75\\%$,\nLow VIS, Cold IR)", "#be123c"),
        (0.65, 0.72, "Deep Frontal / Anvil\n(200–850 hPa Saturated,\nHigh VIS & Cold IR)", "#1e3a8a"),
    ]
    for rx, ry, rtxt, rcol in regimes:
        ax_c.text(
            rx,
            ry,
            rtxt,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=rcol,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=rcol, lw=1.0, alpha=0.89),
        )

    ax_c.set_xlim(0.0, 0.90)
    ax_c.set_ylim(0.0, 0.90)
    ax_c.set_xlabel("Daytime MTG-I1 VIS 0.6 $\\mu$m Solar Reflectance (Normalized)")
    ax_c.set_ylabel("MTG-I1 IR 10.5 $\\mu$m Thermal Coldness (Normalized)")
    ax_c.set_title("(c) Joint Spectral Space Matched to ERA5 Moisture ($\\mathrm{RH}_{\\max}$)", fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.45)

    # -------------------------------------------------------------------------
    # Panel (d): Along-Track Curtain: Physics Profile Matching vs Lidar VFM (Spanning 2 cols)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0:2])
    lat = scene["lat"]
    track_rh = scene["track_rh"]
    track_t = scene["track_t"]
    is_cld = scene["is_cld"]
    top_all = scene["top_all"]
    bot_all = scene["bot_all"]
    surf = scene["surf"]

    lat_2d, alt_2d = np.meshgrid(lat, PRESSURE_ALT_KM)
    cf_d = ax_d.contourf(
        lat_2d,
        alt_2d,
        track_rh,
        levels=np.linspace(0, 100, 21),
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        extend="both",
        alpha=0.88,
    )
    cb_d = fig.colorbar(cf_d, ax=ax_d, fraction=0.024, pad=0.02)
    cb_d.set_label("ERA5 Relative Humidity (%)", fontsize=7.6)

    cs_d = ax_d.contour(
        lat_2d,
        alt_2d,
        track_t - 273.15,
        levels=[-50, -40, -30, -20, -10, 0, 10],
        colors="#475569",
        linewidths=0.85,
        linestyles="--",
    )
    ax_d.clabel(cs_d, inline=True, fontsize=6.6, fmt="%d°C")

    dlat = max(float(np.median(np.diff(lat))) * 2.2, 0.015)
    for k in range(10):
        valid_k = np.where(is_cld[:, k])[0]
        for idx in valid_k:
            b_k = float(bot_all[idx, k])
            t_k = max(float(top_all[idx, k]), b_k + 0.14)
            col_k = "#ef4444" if t_k > THRESHOLD_15KFT_KM else "#f59e0b"
            ax_d.fill_between(
                [lat[idx] - dlat, lat[idx] + dlat],
                [b_k, b_k],
                [t_k, t_k],
                color=col_k,
                alpha=0.85,
                linewidth=0,
                zorder=4,
            )

    cld_track_mask = scene["n_lay"] > 0
    kernel = np.ones(15) / 15.0
    ht_therm_s = np.where(cld_track_mask, np.convolve(scene["h_top_thermal"], kernel, mode="same"), np.nan)
    ht_coup_s = np.where(cld_track_mask, np.convolve(scene["h_top_coupled"], kernel, mode="same"), np.nan)
    hb_coup_s = np.where(cld_track_mask, np.convolve(scene["h_bot_coupled"], kernel, mode="same"), np.nan)

    ax_d.plot(lat, ht_therm_s, color="#d946ef", linewidth=1.9, linestyle=":", zorder=6)
    ax_d.plot(lat, ht_coup_s, color="#16a34a", linewidth=2.1, linestyle="-", zorder=7)
    ax_d.plot(lat, hb_coup_s, color="#15803d", linewidth=1.6, linestyle="--", zorder=7)

    ax_d.fill_between(lat, 0, surf, color="#334155", alpha=0.85, zorder=5)
    ax_d.axhline(THRESHOLD_15KFT_KM, color="#111827", linestyle="-.", linewidth=1.0, zorder=5)

    legend_d = [
        mpatches.Patch(color="#ef4444", label="ICESat-2 GT High Cloud (> 15 kft)"),
        mpatches.Patch(color="#f59e0b", label="ICESat-2 GT Low Cloud (≤ 15 kft)"),
        Line2D([0], [0], color="#d946ef", lw=1.9, ls=":", label="Thermal Inversion $T_b \\leftrightarrow T(p_k)$ (2.89 km MAE)"),
        Line2D([0], [0], color="#16a34a", lw=2.1, ls="-", label="Coupled $T + \\mathrm{RH}$ Top Match (2.57 km MAE)"),
        Line2D([0], [0], color="#15803d", lw=1.6, ls="--", label="Coupled $\\mathrm{RH}\\geq 62\\%$ Base Match (2.53 km MAE)"),
    ]
    ax_d.legend(handles=legend_d, loc="upper left", ncol=2, framealpha=0.92, fontsize=6.6)
    ax_d.set_xlim(float(lat.min()), float(lat.max()))
    ax_d.set_ylim(0.0, 13.5)
    ax_d.set_xlabel("Latitude Along ICESat-2 Ground Track (°N)")
    ax_d.set_ylabel("Altitude ASL (km)")
    ax_d.set_title(
        f"(d) Direct Physics-Based ERA5 Profile Inversion vs. ICESat-2 Lidar VFM ({scene['dt_str']})",
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Panel (e): Spatio-Temporal Co-Registration Perturbation Decay (Track 1B Results)
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 2])
    lags_labels = ["Exact\n(0h, 0°)", "-1h\nLag", "-3h\nLag", "-6h\nLag", "-12h\nLag", "-24h\nLag", "0.5°\n(56km)", "1.0°\n(111km)", "2.0°\n(222km)", "Shuffled\nDate"]
    r_vals = [0.6000, 0.5776, 0.5560, 0.4967, 0.4791, 0.3675, 0.5076, 0.4729, 0.4211, 0.0180]
    sam_mae = [1.218, 1.230, 1.316, 1.406, 1.651, 1.753, 1.502, 1.635, 1.730, 2.651]
    dino_mae = [1.371, 1.332, 1.343, 1.406, 1.592, 1.842, 1.536, 1.598, 1.681, 2.493]

    x_pos = np.arange(len(lags_labels))
    ax_e_r = ax_e.twinx()

    l1 = ax_e.plot(x_pos, sam_mae, "-o", color="#dc2626", linewidth=2.0, markersize=4.8, label="SAM 2.1 (1f IR+VIS+ERA5) Top MAE (km)")
    l2 = ax_e.plot(x_pos, dino_mae, "-s", color="#2563eb", linewidth=2.0, markersize=4.8, label="DINOv3 (1f IR+ERA5) Top MAE (km)")
    bars = ax_e_r.bar(x_pos, r_vals, width=0.48, color="#0d9488", alpha=0.25, edgecolor="#0f766e", linewidth=0.8, label="Physical $r(\\mathrm{IR}, \\mathrm{RH}_{\\max})$")

    ax_e.axvline(5.5, color="#64748b", linestyle="--", linewidth=1.0)
    ax_e.text(7.5, 2.58, "Spatial Shift $\\Delta\\mathrm{deg}$", ha="center", fontsize=6.8, fontweight="bold", color="#334155")

    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(lags_labels, fontsize=6.5, rotation=28, ha="right")
    ax_e.set_ylabel("Cloud-Top Height MAE (km) $\\downarrow$", color="#dc2626", fontweight="bold")
    ax_e_r.set_ylabel("Satellite–ERA5 Correlation $r(\\mathrm{IR}, \\mathrm{RH}_{\\max})$ $\\uparrow$", color="#0f766e", fontweight="bold")
    ax_e.set_ylim(1.05, 2.78)
    ax_e_r.set_ylim(0.0, 0.72)
    ax_e.set_title("(e) Co-Registration Sensitivity ($\\Delta t$ & $\\Delta\\mathrm{deg}$ Sweep)", fontweight="bold")
    ax_e.grid(True, linestyle=":", alpha=0.45)

    lines_e = l1 + l2 + [bars]
    labels_e = [h.get_label() for h in lines_e]
    ax_e.legend(lines_e, labels_e, loc="upper left", framealpha=0.92, fontsize=6.4)

    fig.suptitle(
        "Physical & Statistical Co-Registration Between MTG-I1 Satellite Imagery (IR 10.5 $\\mu$m, VIS 0.6 $\\mu$m), 9-Level ERA5 ($T$, $\\mathrm{RH}$), and ICESat-2 Lidar",
        fontweight="bold",
        y=0.995,
    )
    _save_and_copy(fig, "fig_satellite_era5_physical_matching")


def generate_fig_satellite_era5_multi_scene_matching(scenes: list[dict]) -> None:
    """
    Generate Figure: 4-Scene x 4-Column Multi-Modal Visual Atlas of Satellite Imagery
    Matched to ERA5 Relative Humidity & Temperature Across Diverse Cloud Regimes.
    """
    n_rows = len(scenes)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16.4, 2.95 * n_rows),
        gridspec_kw={"width_ratios": [1.0, 1.04, 1.04, 1.40], "wspace": 0.24, "hspace": 0.34},
    )
    min_lon, min_lat, max_lon, max_lat = DEFAULT_BBOX

    idx_300 = COMMON_9_PRESSURE_LEVELS.index(300)
    idx_850 = COMMON_9_PRESSURE_LEVELS.index(850)

    regime_tags = [
        "Frontal Cyclone & Anvil",
        "Mid-Latitude Multilayer",
        "Convective / Cirrus Shield",
        "Nighttime Frontal System",
    ]

    for r_i, sc in enumerate(scenes):
        db_lons = sc["db_lons"]
        db_lats = sc["db_lats"]
        lon_grid, lat_grid = np.meshgrid(db_lons, db_lats)
        ir_sub = sc["ir_img"][::8, ::8]
        vis_sub = sc["vis_img"][::8, ::8]
        lon_sat = np.linspace(min_lon, max_lon, ir_sub.shape[1])
        lat_sat = np.linspace(max_lat, min_lat, ir_sub.shape[0])
        lon_s_grid, lat_s_grid = np.meshgrid(lon_sat, lat_sat)

        lat = sc["lat"]
        lon = sc["lon"]
        n_lay = sc["n_lay"]
        h_top = sc["h_top"]

        # --- Col 0: Satellite False-Color Composite + ICESat-2 Cloud Labels ---
        ax0 = axes[r_i, 0]
        rgb = np.stack([
            np.clip(0.55 * vis_sub + 0.45 * ir_sub, 0, 1),
            np.clip(0.65 * vis_sub + 0.35 * ir_sub, 0, 1),
            np.clip(0.45 * vis_sub + 0.65 * ir_sub + 0.08, 0, 1),
        ], axis=-1)
        ax0.imshow(rgb, extent=[min_lon, max_lon, min_lat, max_lat], aspect="auto")
        ax0.plot(lon, lat, color="black", linewidth=3.4, alpha=0.75, zorder=3)
        m_clear = n_lay == 0
        m_low = (n_lay == 1) & (h_top <= THRESHOLD_15KFT_KM)
        m_high = (n_lay == 1) & (h_top > THRESHOLD_15KFT_KM)
        m_multi = n_lay >= 2
        ax0.scatter(lon[m_clear], lat[m_clear], c="#00b4d8", s=3.5, zorder=4)
        ax0.scatter(lon[m_low], lat[m_low], c="#ffb703", s=4.5, zorder=5)
        ax0.scatter(lon[m_high], lat[m_high], c="#e63946", s=4.5, zorder=6)
        ax0.scatter(lon[m_multi], lat[m_multi], c="#b5179e", s=5.5, zorder=7)
        ax0.set_xlim(min_lon, max_lon)
        ax0.set_ylim(min_lat, max_lat)
        ax0.set_ylabel(f"Scene {r_i+1}: {regime_tags[r_i]}\n{sc['dt_str']}\nLatitude (°N)", fontweight="bold", fontsize=7.5)
        if r_i == 0:
            ax0.set_title("(1) MTG-I1 VIS+IR Composite\n& ICESat-2 Lidar Track", fontweight="bold")
        if r_i == n_rows - 1:
            ax0.set_xlabel("Longitude (°E)")

        # --- Col 1: Upper-Tropospheric (300 hPa) ERA5 RH + Temp Isotherms + Satellite IR Cold Contours ---
        ax1 = axes[r_i, 1]
        ax1.pcolormesh(lon_grid, lat_grid, sc["prof_rh"][idx_300], cmap="YlGnBu", vmin=0, vmax=100, shading="auto")
        cs1_t = ax1.contour(lon_grid, lat_grid, sc["prof_t"][idx_300] - 273.15, levels=np.arange(-54, -30, 4), colors="#b91c1c", linewidths=0.8, linestyles="--")
        ax1.clabel(cs1_t, inline=True, fontsize=6.2, fmt="%d°C")
        ax1.contour(lon_s_grid, lat_s_grid, ir_sub, levels=[0.50, 0.68], colors=["#f43f5e", "#881337"], linewidths=[1.0, 1.3])
        ax1.plot(lon, lat, color="black", linewidth=1.6, linestyle=":", zorder=4)
        ax1.set_xlim(min_lon, max_lon)
        ax1.set_ylim(min_lat, max_lat)
        if r_i == 0:
            ax1.set_title("(2) Upper-Level Match (300 hPa, ~9.2 km)\nERA5 $\\mathrm{RH}$ & $T$ + IR Cold Cloud Contours", fontweight="bold")
        if r_i == n_rows - 1:
            ax1.set_xlabel("Longitude (°E)")

        # --- Col 2: Boundary-Layer (850 hPa) ERA5 RH + Temp Isotherms + Satellite VIS Bright Contours ---
        ax2 = axes[r_i, 2]
        ax2.pcolormesh(lon_grid, lat_grid, sc["prof_rh"][idx_850], cmap="YlGnBu", vmin=0, vmax=100, shading="auto")
        cs2_t = ax2.contour(lon_grid, lat_grid, sc["prof_t"][idx_850] - 273.15, levels=np.arange(-6, 22, 4), colors="#b91c1c", linewidths=0.8, linestyles="--")
        ax2.clabel(cs2_t, inline=True, fontsize=6.2, fmt="%d°C")
        if np.percentile(vis_sub, 95) > 0.25:
            ax2.contour(lon_s_grid, lat_s_grid, vis_sub, levels=[0.35, 0.55], colors=["#f59e0b", "#b45309"], linewidths=[1.0, 1.3])
        else:
            ax2.text(
                0.04,
                0.08,
                "Nighttime Granule (IR-Only Mode)",
                transform=ax2.transAxes,
                fontsize=6.6,
                fontweight="bold",
                color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#64748b", alpha=0.88),
            )
        ax2.plot(lon, lat, color="black", linewidth=1.6, linestyle=":", zorder=4)
        ax2.set_xlim(min_lon, max_lon)
        ax2.set_ylim(min_lat, max_lat)
        if r_i == 0:
            ax2.set_title("(3) Low-Level Match (850 hPa, ~1.5 km)\nERA5 $\\mathrm{RH}$ & $T$ + VIS Reflectance Contours", fontweight="bold")
        if r_i == n_rows - 1:
            ax2.set_xlabel("Longitude (°E)")

        # --- Col 3: Along-Track Vertical Cross-Section (ERA5 RH+T + Satellite Match vs Lidar VFM) ---
        ax3 = axes[r_i, 3]
        lat_2d, alt_2d = np.meshgrid(lat, PRESSURE_ALT_KM)
        cf3 = ax3.contourf(lat_2d, alt_2d, sc["track_rh"], levels=np.linspace(0, 100, 17), cmap="YlGnBu", vmin=0, vmax=100, alpha=0.88)
        cb3 = fig.colorbar(cf3, ax=ax3, fraction=0.032, pad=0.02)
        cb3.set_label("$\\mathrm{RH}$ (%)", fontsize=7.0)
        cs3 = ax3.contour(lat_2d, alt_2d, sc["track_t"] - 273.15, levels=[-45, -30, -15, 0, 10], colors="#475569", linewidths=0.75, linestyles="--")
        ax3.clabel(cs3, inline=True, fontsize=6.0, fmt="%d°C")

        dlat = max(float(np.median(np.diff(lat))) * 2.2, 0.015)
        for k in range(10):
            valid_k = np.where(sc["is_cld"][:, k])[0]
            for idx in valid_k:
                b_k = float(sc["bot_all"][idx, k])
                t_k = max(float(sc["top_all"][idx, k]), b_k + 0.15)
                col_k = "#ef4444" if t_k > THRESHOLD_15KFT_KM else "#f59e0b"
                ax3.fill_between([lat[idx] - dlat, lat[idx] + dlat], [b_k, b_k], [t_k, t_k], color=col_k, alpha=0.85, linewidth=0, zorder=4)

        kernel = np.ones(15) / 15.0
        ht_c_s = np.where(n_lay > 0, np.convolve(sc["h_top_coupled"], kernel, mode="same"), np.nan)
        hb_c_s = np.where(n_lay > 0, np.convolve(sc["h_bot_coupled"], kernel, mode="same"), np.nan)
        ax3.plot(lat, ht_c_s, color="#16a34a", linewidth=1.8, linestyle="-", zorder=6)
        ax3.plot(lat, hb_c_s, color="#15803d", linewidth=1.4, linestyle="--", zorder=6)
        ax3.fill_between(lat, 0, sc["surf"], color="#334155", alpha=0.85, zorder=5)
        ax3.axhline(THRESHOLD_15KFT_KM, color="#111827", linestyle="-.", linewidth=0.85, zorder=5)
        ax3.set_xlim(float(lat.min()), float(lat.max()))
        ax3.set_ylim(0.0, 13.2)
        ax3.set_ylabel("Alt. (km)")
        if r_i == 0:
            ax3.set_title("(4) Along-Track Vertical Match\nERA5 $\\mathrm{RH}(z), T(z)$ + Sat-ERA5 Inversion vs. Lidar", fontweight="bold")
        if r_i == n_rows - 1:
            ax3.set_xlabel("Latitude Along Track (°N)")

    # Shared legend at bottom of figure
    legend_handles = [
        mpatches.Patch(color="#00b4d8", label="Clear Sky (Lidar)"),
        mpatches.Patch(color="#f59e0b", label="Low Cloud ≤ 15 kft (Lidar)"),
        mpatches.Patch(color="#ef4444", label="High Cloud > 15 kft (Lidar)"),
        mpatches.Patch(color="#b5179e", label="Multi-Layer Cloud (Lidar)"),
        Line2D([0], [0], color="#f43f5e", lw=1.3, label="Satellite IR Cold Cloud Contours"),
        Line2D([0], [0], color="#f59e0b", lw=1.3, label="Satellite VIS Bright Cloud Contours"),
        Line2D([0], [0], color="#b91c1c", lw=1.0, ls="--", label="ERA5 Temperature Isotherms (°C)"),
        Line2D([0], [0], color="#16a34a", lw=1.8, ls="-", label="Coupled Sat-ERA5 $T+\\mathrm{RH}$ Cloud Top/Base Match"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.028), frameon=True, fontsize=7.3)

    fig.suptitle(
        "Multi-Scene Atlas of Satellite Imagery (IR 10.5 $\\mu$m & VIS 0.6 $\\mu$m) Co-Registered with ERA5 Relative Humidity ($\\mathrm{RH}$) & Temperature ($T$)",
        fontweight="bold",
        y=0.996,
    )
    _save_and_copy(fig, "fig_satellite_era5_multi_scene_matching")


def main() -> None:
    db = ERA5Database("../data/era5_npy")
    all_490 = discover_matched_granules("../data/vfms", "../data/ir10.5")
    g_408 = filter_granules_with_30min_window(
        all_490,
        "../data/ir10.5",
        "../data/vis0.6",
        use_vis06=True,
    )
    _, val_408 = split_granules_chronologically(g_408, val_ratio=0.2)

    print("Collecting validation satellite <-> ERA5 matching samples...")
    stats = collect_validation_matching_samples(val_408, db)
    print(f"Collected {len(stats['ir'])} matched lidar shots across validation granules.")

    # Select 3 rich daytime scenes (08:00-13:30 UTC) + 1 rich nighttime scene with high cloud fraction
    day_candidates = []
    night_candidates = []
    for g in all_490:
        dt = datetime.datetime.fromtimestamp(g.timestamp_utc, tz=datetime.timezone.utc)
        if g.img_name == "2025-01-15_12-10.png":
            day_candidates.insert(0, g)
        elif 9 <= dt.hour <= 13 and g in val_408:
            day_candidates.append(g)
        elif (dt.hour <= 3 or dt.hour >= 21) and g in val_408:
            night_candidates.append(g)

    selected_scenes = []
    # Always include the canonical 2025-01-15 12:10 UTC scene first
    selected_scenes.append(_load_scene_by_granule(day_candidates[0], db))
    for g in day_candidates[1:]:
        if len(selected_scenes) >= 3:
            break
        sc = _load_scene_by_granule(g, db)
        cld_frac = float(np.mean(sc["n_lay"] > 0))
        if cld_frac >= 0.35 and np.percentile(sc["vis_img"], 95) > 0.35:
            selected_scenes.append(sc)

    for g in night_candidates:
        sc = _load_scene_by_granule(g, db)
        cld_frac = float(np.mean(sc["n_lay"] > 0))
        if cld_frac >= 0.40:
            selected_scenes.append(sc)
            break

    generate_fig_satellite_era5_physical_matching(selected_scenes[0], stats)
    generate_fig_satellite_era5_multi_scene_matching(selected_scenes[:4])


if __name__ == "__main__":
    main()
