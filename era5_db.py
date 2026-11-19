"""
Reader and temporal interpolator for CloudTrack-3D ERA5 pressure-level profiles.

Supports both:
  1. The pre-extracted Hugging Face archive (`../data/era5_matched_490.npz`), and
  2. The raw directory hierarchy (`../data/era5_npy/<YYYY-MM>/<DD>/<HH>/<parameter>/<level>.npy`).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# 9 pressure levels (hPa) common across the full CloudTrack-3D benchmark
COMMON_9_PRESSURE_LEVELS: List[int] = [200, 300, 400, 500, 600, 700, 800, 850, 900]

ALL_16_PRESSURE_LEVELS: List[int] = [
    100, 150, 200, 250, 300, 400, 500, 600,
    700, 750, 800, 850, 900, 925, 950, 1000,
]


class ERA5Database:
    """Fast reader and temporal interpolator for the ERA5 NumPy database."""

    def __init__(self, root_dir: Union[str, Path] = "../data/era5_npy"):
        self.root = Path(root_dir)
        lat_path = self.root / "latitudes.npy"
        lon_path = self.root / "longitudes.npy"
        self.latitudes = np.load(lat_path) if lat_path.exists() else np.linspace(40.0, 25.0, 61, dtype=np.float32)
        self.longitudes = np.load(lon_path) if lon_path.exists() else np.linspace(24.0, 43.0, 77, dtype=np.float32)
        meta_path = self.root / "metadata.json"
        self.metadata: Dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def get_path(
        self,
        month: str,
        day: Union[int, str],
        hour: Union[int, str],
        parameter: str,
        level: Union[int, str] = "surface",
    ) -> Path:
        dd = f"{int(day):02d}" if isinstance(day, int) or str(day).isdigit() else str(day)
        hh = f"{int(hour):02d}" if isinstance(hour, int) or str(hour).isdigit() else str(hour)
        return self.root / month / dd / hh / parameter / f"{level}.npy"

    def load_slice(
        self,
        month: str,
        day: Union[int, str],
        hour: Union[int, str],
        parameter: str,
        level: Union[int, str] = "surface",
    ) -> np.ndarray:
        path = self.get_path(month, day, hour, parameter, level)
        if not path.exists():
            raise FileNotFoundError(f"Missing ERA5 slice: {path}")
        return np.load(path)

    def load_profile(
        self,
        dt: Union[datetime, str],
        parameter: str = "temperature",
        levels: Sequence[Union[int, str]] = COMMON_9_PRESSURE_LEVELS,
        interpolate_time: bool = True,
    ) -> np.ndarray:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)

        exact_dt = dt.replace(minute=0, second=0, microsecond=0)
        frac_hour = (dt - exact_dt).total_seconds() / 3600.0

        def _try_load_at(t: datetime) -> Optional[np.ndarray]:
            m_str = f"{t.year:04d}-{t.month:02d}"
            d_str = f"{t.day:02d}"
            h_str = f"{t.hour:02d}"
            slices = []
            for lvl in levels:
                p = self.get_path(m_str, d_str, h_str, parameter, lvl)
                if not p.exists():
                    return None
                slices.append(np.load(p))
            return np.stack(slices, axis=0)

        if frac_hour == 0.0:
            arr = _try_load_at(exact_dt)
            if arr is not None:
                return arr

        if not interpolate_time:
            snapped_hour = int(round((dt.hour + dt.minute / 60.0) / 3.0) * 3)
            snapped_dt = exact_dt.replace(hour=0) + timedelta(hours=snapped_hour)
            arr = _try_load_at(snapped_dt)
            if arr is not None:
                return arr
            raise FileNotFoundError(f"No ERA5 profile for {parameter} near {dt}")

        t_prev, arr_prev = None, None
        for dh in range(0, 7):
            cand = exact_dt - timedelta(hours=dh)
            arr = _try_load_at(cand)
            if arr is not None:
                t_prev, arr_prev = cand, arr
                break

        t_next, arr_next = None, None
        for dh in range(1 if frac_hour > 0 else 0, 7):
            cand = exact_dt + timedelta(hours=dh)
            arr = _try_load_at(cand)
            if arr is not None:
                t_next, arr_next = cand, arr
                break

        if arr_prev is not None and arr_next is not None:
            total_sec = (t_next - t_prev).total_seconds()
            if total_sec == 0:
                return arr_prev
            w = float((dt - t_prev).total_seconds() / total_sec)
            return ((1.0 - w) * arr_prev + w * arr_next).astype(np.float32)
        if arr_prev is not None:
            return arr_prev
        if arr_next is not None:
            return arr_next
        raise FileNotFoundError(f"No ERA5 profile found within +/-6h of {dt} for {parameter}")
