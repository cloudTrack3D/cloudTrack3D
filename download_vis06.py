from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.abspath(".."))
import download_sat


def create_resilient_session(pool_maxsize: int = 24) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.6,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_channel_timestamp(
    session: requests.Session,
    layer_name: str,
    fname: str,
    out_dir: str,
) -> tuple[str, bool, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000 and download_sat.validate_png(out_path):
        return fname, True, "cached"

    dt = datetime.strptime(fname.replace(".png", ""), "%Y-%m-%d_%H-%M").replace(tzinfo=timezone.utc)
    params = download_sat.build_wms_params(layer=layer_name, t=dt)
    # Use image/png8 (8-bit indexed palette PNG, matching ../ir10.5) for 4x faster GeoServer rendering
    params["FORMAT"] = "image/png8"

    tmp_path = out_path + ".part"
    for attempt in range(3):
        try:
            r = session.get(download_sat.WMS_ENDPOINT, params=params, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                if download_sat.validate_png(tmp_path):
                    os.replace(tmp_path, out_path)
                    return fname, True, f"downloaded ({len(r.content)//1024}KB)"
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                return fname, False, f"error: {type(e).__name__}"
            time.sleep(0.5 * (attempt + 1))

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return fname, False, "failed_validation"


def main() -> None:
    with open("outputs/matched_granules_cache.json", "r", encoding="utf-8") as f:
        granules = json.load(f)

    # Collect all timestamps for offsets [0, -15, -30] minutes across all 490 granules
    offsets = [0, -15, -30]
    required_fnames: set[str] = set()
    for g in granules:
        base_str = g["img_name"].replace(".png", "")
        dt0 = datetime.strptime(base_str, "%Y-%m-%d_%H-%M").replace(tzinfo=timezone.utc)
        for off in offsets:
            dt_off = dt0 + timedelta(minutes=off)
            required_fnames.add(dt_off.strftime("%Y-%m-%d_%H-%M.png"))

    sorted_fnames = sorted(required_fnames)
    vis_dir = os.path.abspath("../vis0.6")
    ir_dir = os.path.abspath("../ir10.5")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[Download] Total unique 3-frame timestamps (0, -15, -30 min) across {len(granules)} granules: {len(sorted_fnames)}")

    max_workers = 16
    session = create_resilient_session(pool_maxsize=max_workers + 4)

    # First ensure any missing IR 10.5 frame in the 30-min window is also repaired
    missing_ir = [
        fn for fn in sorted_fnames
        if not (os.path.isfile(os.path.join(ir_dir, fn)) and download_sat.validate_png(os.path.join(ir_dir, fn)))
    ]
    if missing_ir:
        print(f"[Download] Repairing {len(missing_ir)} missing/corrupt IR 10.5 files...")
        for fn in missing_ir:
            _, ok, msg = download_channel_timestamp(session, download_sat.CHANNELS["ir10.5"], fn, ir_dir)
            print(f"  [IR 10.5] {fn}: ok={ok} ({msg})")

    # Now download all VIS 0.6 files for the 3-frame windows
    t0 = time.time()
    n_total = len(sorted_fnames)
    n_done = 0
    n_ok = 0
    n_cached = 0
    n_fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_channel_timestamp, session, download_sat.CHANNELS["vis0.6"], fn, vis_dir): fn
            for fn in sorted_fnames
        }
        for fut in as_completed(futures):
            fn, ok, msg = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
                if msg == "cached":
                    n_cached += 1
            else:
                n_fail += 1
            if n_done % 100 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6)
                print(
                    f"[VIS 0.6 Download] {n_done:4d}/{n_total} | ok={n_ok} (cached={n_cached}) fail={n_fail} | "
                    f"rate={rate:.2f} img/s | elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
                )

    session.close()
    print(f"[Download Complete] VIS 0.6 valid files: {n_ok}/{n_total} (failed: {n_fail}) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
