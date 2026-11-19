from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset

# ICESat-2 ATLAS SDP GPS Epoch: 2018-01-01 00:00:00 UTC
ATLAS_SDP_EPOCH = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc).timestamp()

# Default sector bounding box (min_lon, min_lat, max_lon, max_lat)
# Note: download_sat.py & download_vfms.ipynb use (24.0, 25.0, 43.0, 40.0).
# We also support (24.0, 25.0, 44.0, 40.0) for comparison in sanity checks.
DEFAULT_BBOX = (24.0, 25.0, 43.0, 40.0)
ALT_BBOX_44 = (24.0, 25.0, 44.0, 40.0)
DEFAULT_IMAGE_SHAPE = (4096, 4096)  # (height, width)
MAX_HEIGHT_KM = 16.0  # Normalization constant for cloud heights (in km)


@dataclass
class MatchedGranule:
    granule_id: str
    h5_path: str
    img_path: str
    img_name: str
    timestamp_utc: float
    version: str
    split: str = "train"
    synoptic_split: str = "train"


def load_ir_image(img_path: str) -> np.ndarray:
    """
    Load an EUMETSAT IR 10.5 um PNG image and return physical grayscale values (uint8, HxW).
    Handles 8-bit indexed palette ('P') PNGs produced by EUMETSAT GeoServer WMS.
    Higher values correspond to brighter pixels (colder brightness temperatures / higher clouds).
    """
    with Image.open(img_path) as img:
        if img.mode == "P":
            pal = np.array(img.getpalette(), dtype=np.uint8).reshape(-1, 3)[:, 0]
            idx = np.array(img, dtype=np.uint8)
            return pal[idx]
        return np.array(img.convert("L"), dtype=np.uint8)


def _resolve_data_dir(d: str) -> str:
    if os.path.isdir(d):
        return d
    alt = os.path.join("../data", os.path.basename(d.rstrip("/")))
    return alt if os.path.isdir(alt) else d


def discover_matched_granules(
    vfm_dir: str = "../vfms",
    ir_dir: str = "../ir10.5",
    deduplicate_versions: bool = True,
    cache_path: str | None = "outputs/matched_granules_cache.json",
) -> list[MatchedGranule]:
    """
    Scan `vfm_dir` for ICESat-2 ATL09 .h5 files, compute their rounded 5-minute UTC
    overpass timestamp, and match them with `ir_dir` satellite images.
    Deduplicates Release 006 and Release 007 files of the same orbit granule (preferring 007).
    """
    vfm_dir = _resolve_data_dir(vfm_dir)
    ir_dir = _resolve_data_dir(ir_dir)
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached and os.path.isfile(cached[0]["h5_path"]) and os.path.isfile(cached[0]["img_path"]):
            return [MatchedGranule(**item) for item in cached]

    vfm_files = sorted(f for f in os.listdir(vfm_dir) if f.endswith(".h5"))
    ir_files = set(f for f in os.listdir(ir_dir) if f.endswith(".png"))

    by_prefix: dict[str, list[str]] = {}
    for fname in vfm_files:
        parts = fname.split("_")
        prefix = "_".join(parts[:3]) if len(parts) >= 3 else fname
        by_prefix.setdefault(prefix, []).append(fname)

    matched: list[MatchedGranule] = []
    for prefix, fnames in sorted(by_prefix.items()):
        # Sort descending so version 007 is tried before 006
        candidates = sorted(fnames, reverse=True)
        for fname in candidates:
            h5_path = os.path.join(vfm_dir, fname)
            try:
                with h5py.File(h5_path, "r") as f:
                    dts = []
                    for prof in ("profile_1", "profile_2", "profile_3"):
                        if prof not in f or "high_rate" not in f[prof]:
                            dts = []
                            break
                        dt = f[prof]["high_rate"]["delta_time"][:]
                        dt = dt[~np.isnan(dt)]
                        if len(dt) > 0:
                            dts.append(float(dt.mean()))
                    if not dts:
                        continue
                    mean_time = ATLAS_SDP_EPOCH + float(np.mean(dts))
                    rounded_time = round(mean_time / 300.0) * 300.0
                    dt_utc = datetime.datetime.fromtimestamp(rounded_time, tz=datetime.timezone.utc)
                    img_name = dt_utc.strftime("%Y-%m-%d_%H-%M") + ".png"
                    if img_name in ir_files:
                        img_full_path = os.path.join(ir_dir, img_name)
                        try:
                            with Image.open(img_full_path) as im_check:
                                if im_check.size != DEFAULT_IMAGE_SHAPE:
                                    continue
                                im_check.load()
                            # Verify all required datasets in all 3 profiles can be decompressed
                            for prof in ("profile_1", "profile_2", "profile_3"):
                                grp = f[prof]["high_rate"]
                                for req_key in (
                                    "latitude",
                                    "longitude",
                                    "layer_attr",
                                    "layer_top",
                                    "layer_bot",
                                    "cloud_flag_asr",
                                    "cloud_flag_atm",
                                    "layer_flag",
                                    "surface_height",
                                ):
                                    _ = grp[req_key][:]
                        except Exception:
                            continue
                        parts = fname.split("_")
                        ver = parts[3] if len(parts) > 3 else "unknown"
                        matched.append(
                            MatchedGranule(
                                granule_id=prefix,
                                h5_path=h5_path,
                                img_path=img_full_path,
                                img_name=img_name,
                                timestamp_utc=rounded_time,
                                version=ver,
                            )
                        )
                        if deduplicate_versions:
                            break
            except Exception:
                continue

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in matched], f, indent=2)

    return matched


def get_3frame_filenames(img_name: str) -> list[str]:
    """
    Given a base image filename `YYYY-MM-DD_HH-MM.png` at t=0,
    return the 3 filenames for [t=0, t=-15 min, t=-30 min].
    """
    base_str = img_name.replace(".png", "")
    dt0 = datetime.datetime.strptime(base_str, "%Y-%m-%d_%H-%M").replace(tzinfo=datetime.timezone.utc)
    return [
        (dt0 - datetime.timedelta(minutes=off)).strftime("%Y-%m-%d_%H-%M.png")
        for off in (0, 15, 30)
    ]


def _is_valid_png_fast(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def filter_granules_with_30min_window(
    granules: list[MatchedGranule],
    ir_dir: str = "../ir10.5",
    vis_dir: str = "../vis0.6",
    use_vis06: bool = True,
    cache_path: str | None = "outputs/matched_granules_3frames_cache.json",
) -> list[MatchedGranule]:
    """
    Filter `granules` to only keep those that have valid PNG files for the full 30-minute
    3-frame window (t=0, t=-15 min, t=-30 min) in `ir_dir` (and `vis_dir` if `use_vis06=True`).
    """
    ir_dir = _resolve_data_dir(ir_dir)
    vis_dir = _resolve_data_dir(vis_dir)
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached and cached.get("use_vis06") == use_vis06 and len(cached.get("items", [])) > 0:
            first_fnames = get_3frame_filenames(cached["items"][0]["img_name"])
            if all(os.path.isfile(os.path.join(ir_dir, fn)) for fn in first_fnames):
                if not use_vis06 or all(os.path.isfile(os.path.join(vis_dir, fn)) for fn in first_fnames):
                    return [MatchedGranule(**item) for item in cached["items"]]

    valid_list: list[MatchedGranule] = []
    for g in granules:
        fnames = get_3frame_filenames(g.img_name)
        ir_ok = all(_is_valid_png_fast(os.path.join(ir_dir, fn)) for fn in fnames)
        if not ir_ok:
            continue
        if use_vis06:
            vis_ok = all(_is_valid_png_fast(os.path.join(vis_dir, fn)) for fn in fnames)
            if not vis_ok:
                continue
        valid_list.append(g)

    if cache_path and valid_list:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"use_vis06": use_vis06, "items": [m.__dict__ for m in valid_list]},
                f,
                indent=2,
            )
    return valid_list


def latlon_to_pixel(
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    img_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (lat, lon) arrays to integer (row, col) pixel indices in an EPSG:4326 WMS image.
    Row 0 corresponds to max_lat (North), Row H-1 corresponds to min_lat (South).
    Col 0 corresponds to min_lon (West), Col W-1 corresponds to max_lon (East).
    Returns (row, col, in_bounds_mask).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    H, W = img_shape

    in_bounds = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= min_lat)
        & (lat <= max_lat)
        & (lon >= min_lon)
        & (lon <= max_lon)
    )
    row_f = (max_lat - lat) / (max_lat - min_lat) * (H - 1)
    col_f = (lon - min_lon) / (max_lon - min_lon) * (W - 1)

    row = np.clip(np.rint(row_f).astype(np.int32), 0, H - 1)
    col = np.clip(np.rint(col_f).astype(np.int32), 0, W - 1)
    return row, col, in_bounds


def extract_granule_tracks(
    h5_path: str,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    img_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    cloud_attr_mode: str = "cloud_only",
) -> dict[str, np.ndarray]:
    """
    Extract 1D track points across all 3 strong beams ('profile_1', 'profile_2', 'profile_3')
    from an ICESat-2 ATL09 HDF5 file.

    Args:
        h5_path: Path to the ATL09 .h5 file.
        bbox: (min_lon, min_lat, max_lon, max_lat).
        img_shape: (H, W) of the satellite image.
        cloud_attr_mode:
            - "cloud_only": Only count layers where `layer_attr == 1` (Cloud).
            - "cloud_or_unknown": Count layers where `layer_attr in (1, 3)`.
            - "all_layers": Count all valid atmospheric layers (`layer_attr in 1..7`, excluding folded artifact 11).

    Returns:
        Dictionary of concatenated 1D/2D numpy arrays for all valid profile shots.
    """
    rows_list, cols_list, beams_list = [], [], []
    lat_list, lon_list = [], []
    has_cloud_list, n_layers_list = [], []
    top_km_list, bot_high_km_list, bot_low_km_list = [], [], []
    cf_asr_list, cf_atm_list, l_flag_list, surf_km_list = [], [], [], []

    with h5py.File(h5_path, "r") as f:
        for beam_idx, prof_name in enumerate(("profile_1", "profile_2", "profile_3"), start=1):
            if prof_name not in f or "high_rate" not in f[prof_name]:
                continue
            grp = f[prof_name]["high_rate"]
            lat = grp["latitude"][:]
            lon = grp["longitude"][:]
            row, col, in_bounds = latlon_to_pixel(lat, lon, bbox=bbox, img_shape=img_shape)
            if not np.any(in_bounds):
                continue

            lat = lat[in_bounds]
            lon = lon[in_bounds]
            row = row[in_bounds]
            col = col[in_bounds]

            l_attr = grp["layer_attr"][:][in_bounds]  # (N, 10)
            l_top = grp["layer_top"][:][in_bounds]    # (N, 10) meters
            l_bot = grp["layer_bot"][:][in_bounds]    # (N, 10) meters
            cf_asr = grp["cloud_flag_asr"][:][in_bounds]
            cf_atm = grp["cloud_flag_atm"][:][in_bounds]
            l_flag = grp["layer_flag"][:][in_bounds]
            surf_h = grp["surface_height"][:][in_bounds]

            valid_height = (l_top < 1e30) & (l_bot < 1e30) & np.isfinite(l_top) & np.isfinite(l_bot)
            if cloud_attr_mode == "cloud_only":
                is_target_layer = valid_height & (l_attr == 1)
            elif cloud_attr_mode == "cloud_or_unknown":
                is_target_layer = valid_height & ((l_attr == 1) | (l_attr == 3))
            elif cloud_attr_mode == "all_layers":
                is_target_layer = valid_height & (l_attr >= 1) & (l_attr <= 7)
            else:
                raise ValueError(f"Unknown cloud_attr_mode: {cloud_attr_mode}")

            has_cloud = np.any(is_target_layer, axis=1)
            n_layers = np.sum(is_target_layer, axis=1).astype(np.uint8)

            # Convert meters to kilometers; mask out non-target layers with NaN
            top_km_all = np.where(is_target_layer, l_top / 1000.0, np.nan)
            bot_km_all = np.where(is_target_layer, l_bot / 1000.0, np.nan)

            n_pts = len(row)
            top_km = np.full(n_pts, np.nan, dtype=np.float32)
            bot_low_km = np.full(n_pts, np.nan, dtype=np.float32)
            bot_high_km = np.full(n_pts, np.nan, dtype=np.float32)

            if np.any(has_cloud):
                top_km[has_cloud] = np.nanmax(top_km_all[has_cloud], axis=1)
                bot_low_km[has_cloud] = np.nanmin(bot_km_all[has_cloud], axis=1)
                # Base of the highest cloud layer (first True index along axis 1)
                first_idx = np.argmax(is_target_layer[has_cloud], axis=1)
                bot_high_km[has_cloud] = bot_km_all[has_cloud, first_idx]

            rows_list.append(row)
            cols_list.append(col)
            beams_list.append(np.full(n_pts, beam_idx, dtype=np.uint8))
            lat_list.append(lat.astype(np.float32))
            lon_list.append(lon.astype(np.float32))
            has_cloud_list.append(has_cloud.astype(np.float32))
            n_layers_list.append(n_layers)
            top_km_list.append(top_km)
            bot_high_km_list.append(bot_high_km)
            bot_low_km_list.append(bot_low_km)
            cf_asr_list.append(cf_asr.astype(np.int8))
            cf_atm_list.append(cf_atm.astype(np.int8))
            l_flag_list.append(l_flag.astype(np.int8))
            surf_km_list.append(np.where(surf_h < 1e30, surf_h / 1000.0, np.nan).astype(np.float32))

    if not rows_list:
        empty_f = np.zeros(0, dtype=np.float32)
        empty_i = np.zeros(0, dtype=np.int32)
        return {
            "row": empty_i,
            "col": empty_i,
            "beam": empty_i.astype(np.uint8),
            "lat": empty_f,
            "lon": empty_f,
            "has_cloud": empty_f,
            "n_layers": empty_i.astype(np.uint8),
            "cloud_top_km": empty_f,
            "cloud_bot_km": empty_f,
            "cloud_bot_lowest_km": empty_f,
            "cloud_flag_asr": empty_i.astype(np.int8),
            "cloud_flag_atm": empty_i.astype(np.int8),
            "layer_flag": empty_i.astype(np.int8),
            "surface_height_km": empty_f,
        }

    return {
        "row": np.concatenate(rows_list),
        "col": np.concatenate(cols_list),
        "beam": np.concatenate(beams_list),
        "lat": np.concatenate(lat_list),
        "lon": np.concatenate(lon_list),
        "has_cloud": np.concatenate(has_cloud_list),
        "n_layers": np.concatenate(n_layers_list),
        "cloud_top_km": np.concatenate(top_km_list),
        "cloud_bot_km": np.concatenate(bot_high_km_list),
        "cloud_bot_lowest_km": np.concatenate(bot_low_km_list),
        "cloud_flag_asr": np.concatenate(cf_asr_list),
        "cloud_flag_atm": np.concatenate(cf_atm_list),
        "layer_flag": np.concatenate(l_flag_list),
        "surface_height_km": np.concatenate(surf_km_list),
    }


def build_2d_cloud_map(
    h5_path: str,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    img_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    cloud_attr_mode: str = "cloud_only",
) -> dict[str, np.ndarray]:
    """
    Project ICESat-2 3-beam cloud information into a full-resolution 2D map of shape `img_shape` (4096 x 4096).
    Pixels without ICESat-2 measurements have `valid_mask == 0`.
    Pixels with ICESat-2 measurements have `valid_mask == 1`, `cloud_mask` in {0.0, 1.0},
    and `cloud_top` / `cloud_bot` in km (0.0 where clear or unlabeled, with `valid_cloud_mask == 1` where cloudy).
    """
    H, W = img_shape
    tracks = extract_granule_tracks(
        h5_path=h5_path,
        bbox=bbox,
        img_shape=img_shape,
        cloud_attr_mode=cloud_attr_mode,
    )

    valid_mask = np.zeros((H, W), dtype=np.uint8)
    cloud_mask = np.zeros((H, W), dtype=np.float32)
    cloud_top = np.zeros((H, W), dtype=np.float32)
    cloud_bot = np.zeros((H, W), dtype=np.float32)
    n_layers_map = np.zeros((H, W), dtype=np.uint8)
    beam_map = np.zeros((H, W), dtype=np.uint8)

    rows = tracks["row"]
    cols = tracks["col"]
    if len(rows) == 0:
        return {
            "valid_mask": valid_mask,
            "cloud_mask": cloud_mask,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
            "n_layers": n_layers_map,
            "beam_map": beam_map,
        }

    # First write clear-sky valid points, then write cloudy points so cloudy takes precedence
    # if two adjacent 25 Hz shots fall into the same ~400m pixel.
    is_cloudy = tracks["has_cloud"] > 0.5
    clear_idx = np.where(~is_cloudy)[0]
    cloudy_idx = np.where(is_cloudy)[0]

    if len(clear_idx) > 0:
        r_c, c_c = rows[clear_idx], cols[clear_idx]
        valid_mask[r_c, c_c] = 1
        cloud_mask[r_c, c_c] = 0.0
        beam_map[r_c, c_c] = tracks["beam"][clear_idx]

    if len(cloudy_idx) > 0:
        r_y, c_y = rows[cloudy_idx], cols[cloudy_idx]
        valid_mask[r_y, c_y] = 1
        cloud_mask[r_y, c_y] = 1.0
        cloud_top[r_y, c_y] = np.clip(tracks["cloud_top_km"][cloudy_idx], 0.0, MAX_HEIGHT_KM)
        cloud_bot[r_y, c_y] = np.clip(tracks["cloud_bot_km"][cloudy_idx], 0.0, MAX_HEIGHT_KM)
        n_layers_map[r_y, c_y] = tracks["n_layers"][cloudy_idx]
        beam_map[r_y, c_y] = tracks["beam"][cloudy_idx]

    return {
        "valid_mask": valid_mask,
        "cloud_mask": cloud_mask,
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
        "n_layers": n_layers_map,
        "beam_map": beam_map,
    }


class FullImageCloudDataset(Dataset):
    """
    Dataset yielding full-resolution (4096 x 4096) satellite IR 10.5 um images
    matched with full-resolution (4096 x 4096) 2D ICESat-2 cloud label maps.
    """

    def __init__(
        self,
        granules: list[MatchedGranule],
        bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
        cloud_attr_mode: str = "cloud_only",
    ) -> None:
        self.granules = granules
        self.bbox = bbox
        self.cloud_attr_mode = cloud_attr_mode

    def __len__(self) -> int:
        return len(self.granules)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self.granules[idx]
        ir_gray = load_ir_image(g.img_path)
        maps_2d = build_2d_cloud_map(
            h5_path=g.h5_path,
            bbox=self.bbox,
            img_shape=ir_gray.shape,
            cloud_attr_mode=self.cloud_attr_mode,
        )
        return {
            "granule_id": g.granule_id,
            "img_name": g.img_name,
            "image": torch.from_numpy(ir_gray.astype(np.float32) / 255.0).unsqueeze(0),  # (1, H, W)
            "valid_mask": torch.from_numpy(maps_2d["valid_mask"].astype(np.float32)),    # (H, W)
            "cloud_mask": torch.from_numpy(maps_2d["cloud_mask"]),                       # (H, W)
            "cloud_top": torch.from_numpy(maps_2d["cloud_top"] / MAX_HEIGHT_KM),         # (H, W) normalized [0,1]
            "cloud_bot": torch.from_numpy(maps_2d["cloud_bot"] / MAX_HEIGHT_KM),         # (H, W) normalized [0,1]
            "n_layers": torch.from_numpy(maps_2d["n_layers"]),                           # (H, W)
        }


def sample_era5_patch(
    era5_18_61_77: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    img_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> np.ndarray:
    """
    Fast separable 2D bilinear interpolation of an `(18, 61, 77)` ERA5 grid
    (covering Lat 40.0N->25.0N, Lon 24.0E->43.0E) onto a `[r0:r1, c0:c1]` patch
    of the `img_shape` (4096 x 4096) satellite grid.
    Returns shape `(18, r1 - r0, c1 - c0)` in `float32`.
    """
    H, W = img_shape
    rows = np.arange(r0, r1, dtype=np.float32)
    cols = np.arange(c0, c1, dtype=np.float32)

    y = rows * (60.0 / (H - 1))
    y0 = np.clip(np.floor(y).astype(np.int32), 0, 59)
    y1 = y0 + 1
    wy = (y - y0)[None, :, None]

    x = cols * (76.0 / (W - 1))
    x0 = np.clip(np.floor(x).astype(np.int32), 0, 75)
    x1 = x0 + 1
    wx = (x - x0)[None, None, :]

    row_interp = (1.0 - wy) * era5_18_61_77[:, y0, :] + wy * era5_18_61_77[:, y1, :]
    return ((1.0 - wx) * row_interp[:, :, x0] + wx * row_interp[:, :, x1]).astype(np.float32)


class CloudPatchDataset(Dataset):
    """
    Patch-based dataset for training and evaluating pixel-wise cloud detection and height estimation.
    Each item crops a `patch_size x patch_size` window from the full 4096 x 4096 image and its
    corresponding 2D label map along the ICESat-2 ground tracks.

    Supports:
      - Satellite-only (`use_satellite=True, use_era5=False`): 3, 4, 5, or 8 channels
      - ERA5-only (`use_satellite=False, use_era5=True`): 20 channels ([T(9), RH(9), lat, lon])
      - Satellite + ERA5 (`use_satellite=True, use_era5=True`): 21, 22, 23, or 26 channels
    """

    def __init__(
        self,
        granules: list[MatchedGranule],
        patch_size: int = 256,
        patches_per_granule: int = 8,
        bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
        cloud_attr_mode: str = "cloud_only",
        is_train: bool = True,
        seed: int = 42,
        max_cached_images: int = 64,
        use_3frames: bool = False,
        use_vis06: bool = False,
        use_era5: bool = False,
        use_satellite: bool = True,
        ir_dir: str = "../ir10.5",
        vis_dir: str = "../vis0.6",
        era5_cache: dict[str, np.ndarray] | None = None,
        allowed_beams: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        self.granules = granules
        self.patch_size = patch_size
        self.patches_per_granule = patches_per_granule
        self.bbox = bbox
        self.cloud_attr_mode = cloud_attr_mode
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.max_cached_images = max_cached_images
        self.use_3frames = use_3frames
        self.use_vis06 = use_vis06
        self.use_era5 = use_era5
        self.use_satellite = use_satellite
        self.ir_dir = _resolve_data_dir(ir_dir)
        self.vis_dir = _resolve_data_dir(vis_dir)
        self._era5_cache = era5_cache or {}
        self.allowed_beams = tuple(allowed_beams)

        # Pre-load sparse track arrays in RAM (very compact: ~50 KB per granule, ~25 MB total)
        self.track_cache: list[dict[str, np.ndarray]] = []
        self.valid_granule_indices: list[int] = []
        for idx, g in enumerate(self.granules):
            tr = extract_granule_tracks(
                h5_path=g.h5_path,
                bbox=self.bbox,
                img_shape=DEFAULT_IMAGE_SHAPE,
                cloud_attr_mode=self.cloud_attr_mode,
            )
            if self.allowed_beams != (1, 2, 3) and len(tr["row"]) > 0:
                beam_mask = np.isin(tr["beam"], self.allowed_beams)
                tr = {k: v[beam_mask] for k, v in tr.items()}
            self.track_cache.append(tr)
            if len(tr["row"]) > 0:
                self.valid_granule_indices.append(idx)

        self._img_cache: dict[str, np.ndarray] = {}
        if self.use_satellite and self.max_cached_images >= 1500:
            self._preload_images_parallel(max_workers=16)

    def _preload_images_parallel(self, max_workers: int = 16) -> None:
        from concurrent.futures import ThreadPoolExecutor

        paths_to_load: set[str] = set()
        for idx in self.valid_granule_indices:
            g = self.granules[idx]
            fnames = get_3frame_filenames(g.img_name) if self.use_3frames else [g.img_name]
            for fn in fnames:
                paths_to_load.add(g.img_path if fn == g.img_name else os.path.join(self.ir_dir, fn))
                if self.use_vis06:
                    paths_to_load.add(os.path.join(self.vis_dir, fn))

        sorted_paths = sorted(paths_to_load)[: self.max_cached_images]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(load_ir_image, sorted_paths))
        for p, arr in zip(sorted_paths, results):
            self._img_cache[p] = arr

    @property
    def in_channels(self) -> int:
        if not self.use_satellite:
            return 20  # 18 ERA5 (9 T + 9 RH) + 2 (lat, lon)
        n_frames = 3 if self.use_3frames else 1
        n_spec = 2 if self.use_vis06 else 1
        base = n_frames * n_spec + 2
        return base + 18 if self.use_era5 else base

    def _get_image(self, img_path: str) -> np.ndarray:
        if img_path not in self._img_cache:
            if len(self._img_cache) >= self.max_cached_images:
                # Evict oldest cached image
                first_key = next(iter(self._img_cache))
                del self._img_cache[first_key]
            self._img_cache[img_path] = load_ir_image(img_path)
        return self._img_cache[img_path]

    def __len__(self) -> int:
        return len(self.valid_granule_indices) * self.patches_per_granule

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        g_idx = self.valid_granule_indices[index // self.patches_per_granule]
        patch_idx = index % self.patches_per_granule
        g = self.granules[g_idx]
        tr = self.track_cache[g_idx]

        if self.use_satellite:
            img = self._get_image(g.img_path)
            H, W = img.shape
        else:
            img = None
            H, W = DEFAULT_IMAGE_SHAPE
        ps = self.patch_size
        half = ps // 2

        rows = tr["row"]
        cols = tr["col"]
        n_pts = len(rows)

        if self.is_train:
            pt_i = int(self.rng.integers(0, n_pts))
            jitter_r = int(self.rng.integers(-ps // 3, ps // 3 + 1))
            jitter_c = int(self.rng.integers(-ps // 3, ps // 3 + 1))
            center_r = int(rows[pt_i]) + jitter_r
            center_c = int(cols[pt_i]) + jitter_c
        else:
            # Deterministic evenly spaced patches along the track for reproducible evaluation
            pt_i = int((patch_idx + 0.5) * n_pts / self.patches_per_granule)
            pt_i = min(max(pt_i, 0), n_pts - 1)
            center_r = int(rows[pt_i])
            center_c = int(cols[pt_i])

        r0 = int(np.clip(center_r - half, 0, H - ps))
        c0 = int(np.clip(center_c - half, 0, W - ps))
        r1 = r0 + ps
        c1 = c0 + ps

        spec_patches: list[np.ndarray] = []
        if self.use_satellite:
            all_3f = get_3frame_filenames(g.img_name)
            if self.use_3frames:
                fnames = all_3f
            else:
                off_min = getattr(self, "frame_offset_minutes", 0)
                off_idx = {0: 0, 15: 1, 30: 2}.get(off_min, 0)
                fnames = [all_3f[off_idx]]
            # IR 10.5 um channels (1 frame at t=0/-15m/-30m or 3 frames: t=0, t=-15m, t=-30m)
            for fn in fnames:
                ir_full = img if fn == g.img_name else self._get_image(os.path.join(self.ir_dir, fn))
                spec_patches.append(ir_full[r0:r1, c0:c1].astype(np.float32) / 255.0)

            # VIS 0.6 um channels (1 frame or 3 frames: t=0, t=-15m, t=-30m)
            if self.use_vis06:
                for fn in fnames:
                    vis_full = self._get_image(os.path.join(self.vis_dir, fn))
                    spec_patches.append(vis_full[r0:r1, c0:c1].astype(np.float32) / 255.0)

        # Normalized latitude and longitude coordinate channels in [0, 1]
        r_coords = np.linspace(1.0 - r0 / (H - 1), 1.0 - (r1 - 1) / (H - 1), ps, dtype=np.float32)
        c_coords = np.linspace(c0 / (W - 1), (c1 - 1) / (W - 1), ps, dtype=np.float32)
        lat_patch = np.broadcast_to(r_coords[:, None], (ps, ps)).copy()
        lon_patch = np.broadcast_to(c_coords[None, :], (ps, ps)).copy()

        # Construct 2D label maps for the patch
        valid_mask = np.zeros((ps, ps), dtype=np.float32)
        cloud_mask = np.zeros((ps, ps), dtype=np.float32)
        cloud_top = np.zeros((ps, ps), dtype=np.float32)
        cloud_bot = np.zeros((ps, ps), dtype=np.float32)
        n_layers = np.zeros((ps, ps), dtype=np.uint8)
        surface_height = np.zeros((ps, ps), dtype=np.float32)

        in_patch = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
        if np.any(in_patch):
            pr = rows[in_patch] - r0
            pc = cols[in_patch] - c0
            p_cloud = tr["has_cloud"][in_patch] > 0.5
            p_top = tr["cloud_top_km"][in_patch]
            p_bot = tr["cloud_bot_km"][in_patch]
            p_nl = tr["n_layers"][in_patch]
            p_surf = np.nan_to_num(tr["surface_height_km"][in_patch], nan=0.0)

            clear_m = ~p_cloud
            if np.any(clear_m):
                valid_mask[pr[clear_m], pc[clear_m]] = 1.0
                cloud_mask[pr[clear_m], pc[clear_m]] = 0.0
                surface_height[pr[clear_m], pc[clear_m]] = p_surf[clear_m]

            if np.any(p_cloud):
                valid_mask[pr[p_cloud], pc[p_cloud]] = 1.0
                cloud_mask[pr[p_cloud], pc[p_cloud]] = 1.0
                cloud_top[pr[p_cloud], pc[p_cloud]] = np.clip(p_top[p_cloud], 0.0, MAX_HEIGHT_KM) / MAX_HEIGHT_KM
                cloud_bot[pr[p_cloud], pc[p_cloud]] = np.clip(p_bot[p_cloud], 0.0, MAX_HEIGHT_KM) / MAX_HEIGHT_KM
                n_layers[pr[p_cloud], pc[p_cloud]] = p_nl[p_cloud]
                surface_height[pr[p_cloud], pc[p_cloud]] = p_surf[p_cloud]

        if not self.use_satellite:
            era5_patch = sample_era5_patch(self._era5_cache[g.granule_id], r0, r1, c0, c1, img_shape=(H, W))
            input_tensor = np.concatenate([era5_patch, np.stack([lat_patch, lon_patch], axis=0)], axis=0)
        elif self.use_era5:
            sat_coord = np.stack(spec_patches + [lat_patch, lon_patch], axis=0)
            era5_patch = sample_era5_patch(self._era5_cache[g.granule_id], r0, r1, c0, c1, img_shape=(H, W))
            input_tensor = np.concatenate([sat_coord, era5_patch], axis=0)
        else:
            input_tensor = np.stack(spec_patches + [lat_patch, lon_patch], axis=0)

        return {
            "image": torch.from_numpy(input_tensor),        # (C, ps, ps)
            "valid_mask": torch.from_numpy(valid_mask),     # (ps, ps)
            "cloud_mask": torch.from_numpy(cloud_mask),     # (ps, ps)
            "cloud_top": torch.from_numpy(cloud_top),       # (ps, ps) normalized by MAX_HEIGHT_KM
            "cloud_bot": torch.from_numpy(cloud_bot),       # (ps, ps) normalized by MAX_HEIGHT_KM
            "n_layers": torch.from_numpy(n_layers),         # (ps, ps) uint8
            "surface_height": torch.from_numpy(surface_height),  # (ps, ps) km
        }


def split_granules_chronologically(
    granules: list[MatchedGranule],
    val_ratio: float = 0.2,
) -> tuple[list[MatchedGranule], list[MatchedGranule]]:
    """
    Return the fixed chronological train and test partitions without temporal/granule leakage.
    Every 5th granule in chronological order is assigned to the held-out test split (`split="test"`),
    ensuring both train and test cover the full annual cycle.
    """
    sorted_g = sorted(granules, key=lambda g: (g.timestamp_utc, g.granule_id))
    stride = max(int(round(1.0 / val_ratio)), 2)
    train_g, val_g = [], []
    for i, g in enumerate(sorted_g):
        if (i % stride) == (stride - 1):
            g.split = "test"
            val_g.append(g)
        else:
            g.split = "train"
            train_g.append(g)
    return train_g, val_g


def get_synoptic_block_split(
    granules: list[MatchedGranule],
    n_blocks: int = 5,
    train_frac: float = 0.78,
    buffer_hours: float = 48.0,
) -> tuple[list[MatchedGranule], list[MatchedGranule], list[MatchedGranule]]:
    """
    Partition granules into 5 contiguous seasonal blocks with a strict >= 48-hour synoptic
    weather buffer between train and test blocks. Returns (syn_train, syn_test, buffer_excluded).
    """
    sorted_g = sorted(granules, key=lambda g: g.timestamp_utc)
    n_tot = len(sorted_g)
    block_sz = n_tot // n_blocks
    syn_train: list[MatchedGranule] = []
    syn_test: list[MatchedGranule] = []
    syn_excluded: list[MatchedGranule] = []
    buffer_sec = buffer_hours * 3600.0
    for b_i in range(n_blocks):
        blk = sorted_g[b_i * block_sz : (b_i + 1) * block_sz if b_i < n_blocks - 1 else n_tot]
        split_pt = int(len(blk) * train_frac)
        val_blk = blk[split_pt:]
        val_start_t = val_blk[0].timestamp_utc
        val_end_t = val_blk[-1].timestamp_utc
        for g in val_blk:
            g.synoptic_split = "test"
            syn_test.append(g)
        for g in blk[:split_pt]:
            if abs(g.timestamp_utc - val_start_t) >= buffer_sec and abs(g.timestamp_utc - val_end_t) >= buffer_sec:
                g.synoptic_split = "train"
                syn_train.append(g)
            else:
                g.synoptic_split = "buffer_excluded"
                syn_excluded.append(g)
    return syn_train, syn_test, syn_excluded


def load_fixed_dataset_split(
    split_name: str = "multispectral_3frames_408",
    manifest_path: str = "outputs/cloudtrack3d_splits.json",
) -> tuple[list[MatchedGranule], list[MatchedGranule]]:
    """
    Load a permanently fixed CloudTrack-3D dataset train/test split directly from the
    canonical `cloudtrack3d_splits.json` dataset manifest.
    Supported `split_name` values:
      - "allday_490" (392 train, 98 test)
      - "multispectral_3frames_408" (327 train, 81 test)
      - "synoptic_block_408" (304 train, 91 test; 13 buffer_excluded omitted)
    """
    if not os.path.isfile(manifest_path):
        alt_path = "../data/cloudtrack3d_splits.json"
        if os.path.isfile(alt_path):
            manifest_path = alt_path
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sec = manifest["splits"][split_name]
    fields = ("granule_id", "h5_path", "img_path", "img_name", "timestamp_utc", "version", "split", "synoptic_split")
    train_list = [MatchedGranule(**{k: item[k] for k in fields if k in item}) for item in sec["train"]]
    test_list = [MatchedGranule(**{k: item[k] for k in fields if k in item}) for item in sec["test"]]
    return train_list, test_list


class InMemoryDataset(Dataset):
    """Pre-materializes dataset items in RAM once for fast multi-pass validation/training."""

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
