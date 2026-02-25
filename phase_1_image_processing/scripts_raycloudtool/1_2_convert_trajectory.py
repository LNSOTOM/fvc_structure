#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a trajectory with SOW (seconds-of-week), Latitude, Longitude, Altitude
into a rayimport-ready text file: "time x y z", guaranteeing time overlap with
the LAS gps_time range and projecting to the LAS CRS.

Key features
- Reads LAS gps_time with laspy (fast). PDAL is optional fallback.
- Trajectory: text (with/without header) or binary SBET-like (17 doubles).
- Auto-detects degrees vs radians (or you can force via --latlon-in-radians).
- Absolute GPS time = GPS_week*604800 + SOW (week from column, inferred from LAS,
  or remapped by order into LAS time window).
- Auto-fix ±1e9 second offset (“Adjusted Standard GPS Time”) when applicable.
- Projects WGS84 3D (EPSG:4979) → target EPSG (meters) via pyproj.
"""

import argparse
import math
import re
import struct
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# Optional deps (handled gracefully)
try:
    import laspy
except Exception:
    laspy = None

try:
    from pyproj import Transformer
except Exception as e:
    raise SystemExit("pyproj is required: pip install pyproj") from e

WEEK = 604800.0  # seconds per GPS week

# ---------------- LAS time readers ----------------
def extract_gps_time_range_with_laspy(las_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if laspy is None:
        print("   ⚠️  laspy not available; skipping laspy time extraction.")
        return None, None
    try:
        las = laspy.read(str(las_path))
        if "gps_time" not in las.point_format.dimension_names:
            print("   ⚠️  LAS has no gps_time dimension.")
            return None, None
        t = las.gps_time
        t = np.asarray(t, dtype=float)
        tmin = float(np.nanmin(t)); tmax = float(np.nanmax(t))
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            print("   ⚠️  Invalid LAS gps_time min/max.")
            return None, None
        print(f"   ✅ Extracted GPS time via laspy: {tmin} - {tmax}")
        return tmin, tmax
    except Exception as e:
        print(f"   ⚠️  laspy extraction failed: {e}")
        return None, None

def extract_gps_time_range_with_pdal(las_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if shutil.which("pdal") is None:
        return None, None
    try:
        import subprocess, json as _json
        res = subprocess.run(["pdal", "info", "--stats", str(las_path)],
                             capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return None, None
        info = _json.loads(res.stdout)
        stats = info.get("stats", {})
        for stat in stats.get("statistic", []):
            if stat.get("name", "").lower() == "gpstime":
                mn = stat.get("minimum"); mx = stat.get("maximum")
                if mn is not None and mx is not None:
                    print(f"   ✅ Extracted GPS time via PDAL: {mn} - {mx}")
                    return float(mn), float(mx)
    except Exception:
        pass
    return None, None

# ---------------- Trajectory readers ----------------
def read_text_trajectory(path: Path) -> pd.DataFrame:
    # robust CSV/TXT: supports comma/semicolon/whitespace; ignores '#' comments
    df = pd.read_csv(path, engine="python", sep=r"[,\s;]+", comment="#", header=0)
    # Drop typical stray columns
    df = df.drop(columns=["%"], errors="ignore")
    return df

def read_binary_sbet(path: Path) -> Optional[np.ndarray]:
    """Attempt to read SBET-like binary (17 doubles). Returns ndarray [N,17] or None."""
    rec_size = 17 * 8
    try:
        data = path.read_bytes()
        if len(data) % rec_size != 0:
            return None
        n = len(data) // rec_size
        for endian in ("<", ">"):
            try:
                arr = np.frombuffer(data, dtype=endian + "f8")
                if arr.size != n * 17:  # sanity
                    continue
                arr = arr.reshape(n, 17)
                t, lat, lon, alt = arr[0, 0], arr[0, 1], arr[0, 2], arr[0, 3]
                if not (0 <= t <= WEEK):
                    continue
                if not (-math.pi <= lat <= math.pi) or not (-math.pi <= lon <= math.pi):
                    continue
                return arr
            except Exception:
                continue
    except Exception:
        return None
    return None

# ---------------- Helpers ----------------
def pick_col(df: pd.DataFrame, names: List[Optional[str]], required=True) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if not name:
            continue
        if name in df.columns:
            return name
        nl = name.lower()
        if nl in lower:
            return lower[nl]
    if required:
        raise RuntimeError(f"Could not find any of {names} in columns {list(df.columns)}")
    return None

def infer_week_from_las(sow_first: float, las_tmin: float) -> int:
    return int(round((las_tmin - sow_first) / WEEK))

def remap_times_to_las_range(n: int, las_tmin: float, las_tmax: float, pad_frac: float = 0.01) -> np.ndarray:
    if n <= 1:
        return np.array([las_tmin], dtype=np.float64)
    r = np.linspace(0.0, 1.0, n)
    pad = pad_frac * (las_tmax - las_tmin)
    return las_tmin - pad + r * ((las_tmax - las_tmin) + 2 * pad)

def looks_like_radians(lat: np.ndarray, lon: np.ndarray) -> bool:
    # typical ranges: lat ~ [-1.57, 1.57], lon ~ [-3.14, 3.14]
    try:
        return (np.nanmax(np.abs(lat)) <= 1.7) and (np.nanmax(np.abs(lon)) <= 3.3)
    except ValueError:
        return False

# ---------------- Core conversion ----------------
def convert_trajectory_to_rayimport(
    traj_path: Path,
    out_path: Path,
    las_path: Optional[Path] = None,
    target_epsg: int = 7854,
    latlon_in_radians: Optional[bool] = None,   # None => auto-detect
    week_col: Optional[str] = None,
    sow_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    alt_col: Optional[str] = None,
    force_remap: bool = False,
    time_offset: Optional[float] = None,
) -> None:
    print(f"\n🔄 Converting trajectory: {traj_path.name}")

    # LAS time range
    las_tmin = las_tmax = None
    if las_path and las_path.exists():
        las_tmin, las_tmax = extract_gps_time_range_with_laspy(las_path)
        if las_tmin is None:
            las_tmin, las_tmax = extract_gps_time_range_with_pdal(las_path)
        if las_tmin is not None:
            print(f"   ⏰ LAS gps_time range : {las_tmin:.6f} .. {las_tmax:.6f}")

    # Try binary SBET first
    sbet = read_binary_sbet(traj_path)
    using_text = sbet is None

    if not using_text:
        sow = sbet[:, 0]
        lat = sbet[:, 1]
        lon = sbet[:, 2]
        alt = sbet[:, 3]
        # SBET lat/lon are radians
        is_radians = True
        print(f"   📁 Read {len(sbet)} records from binary SBET")
    else:
        df = read_text_trajectory(traj_path)
        print(f"   🧭 Columns: {list(df.columns)}")

        # Auto-pick columns if not provided
        lat_col = pick_col(df, [lat_col, "Latitude", "lat", "Latitude (radians)"], required=True)
        lon_col = pick_col(df, [lon_col, "Longitude", "lon", "Longitude (radians)"], required=True)
        alt_col = pick_col(df, [alt_col, "Altitude", "Z", "height", "Altitude (meters)"], required=True)
        sow_col = pick_col(df, [sow_col, "Time", "SOW", "gps_sow", "gps_time_sow"], required=True)
        week_col = pick_col(df, [week_col, "GPS_week", "GPS Week", "Week"], required=False)

        # Coerce numerics and drop rows with bad lat/lon/alt
        for c in [lat_col, lon_col, alt_col, sow_col] + ([week_col] if week_col else []):
            if c:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        before = len(df)
        df = df.dropna(subset=[lat_col, lon_col, alt_col])  # allow sow to be constant/NaN; we may remap
        dropped = before - len(df)
        if dropped:
            print(f"   ⚠️  Dropped {dropped} rows with non-numeric/blank lat/lon/alt")

        lat = df[lat_col].to_numpy(dtype=float)
        lon = df[lon_col].to_numpy(dtype=float)
        alt = df[alt_col].to_numpy(dtype=float)
        sow = df[sow_col].to_numpy(dtype=float)

        # Auto-detect radians vs degrees unless forced
        if latlon_in_radians is None:
            is_radians = looks_like_radians(lat, lon)
            if is_radians:
                print("   ℹ️  Detected lat/lon in radians → converting to degrees.")
        else:
            is_radians = bool(latlon_in_radians)

        print(f"   📊 SOW range: {np.nanmin(sow):.6g} - {np.nanmax(sow):.6g}")

    # Convert radians → degrees if needed
    if is_radians:
        lat = np.degrees(lat)
        lon = np.degrees(lon)

    # Build absolute GPS time
    if using_text and (week_col is not None) and (week_col in df.columns) and not force_remap:
        week = df[week_col].to_numpy(dtype=int)
        t_abs = week * WEEK + sow
        print(f"   [TIME] Using provided GPS week column '{week_col}'")
    else:
        if las_tmin is not None and np.isfinite(las_tmin) and np.isfinite(las_tmax):
            # If sow looks invalid/constant, remap by order
            if (not np.isfinite(np.nanmin(sow))) or (np.nanmax(sow) - np.nanmin(sow) < 1.0):
                print("   [TIME] SOW invalid or near-constant → remapping by order to LAS range.")
                t_abs = remap_times_to_las_range(len(sow), las_tmin, las_tmax)
            else:
                w = infer_week_from_las(float(sow[0]), las_tmin)
                t_try = w * WEEK + sow
                if force_remap or (t_try.max() < las_tmin) or (t_try.min() > las_tmax):
                    print("   [TIME] Remapping trajectory times by order to LAS range.")
                    t_abs = remap_times_to_las_range(len(sow), las_tmin, las_tmax)
                else:
                    t_abs = t_try
                    print(f"   [TIME] Inferred GPS week {w} from LAS.")
        else:
            print("   [TIME] No LAS provided; generating relative monotonic times.")
            base = 1_000_000_000.0
            t_abs = base + np.linspace(0, 1, len(sow))

    # Manual offset
    if time_offset:
        print(f"   🔧 Applying user time offset: {time_offset:+.0f} s")
        t_abs = t_abs + float(time_offset)

    # DJI adjusted time auto-fix (+/- 1e9) relative to LAS
    if (las_tmin is not None) and (las_tmax is not None) and not force_remap:
        dt = float(np.nanmin(t_abs) - las_tmin)
        if abs(abs(dt) - 1_000_000_000.0) < 5_000.0:
            shift = -1_000_000_000.0 if dt > 0 else +1_000_000_000.0
            print(f"   🔁 Auto time shift by {shift:+.0f} s (DJI adjusted time fix).")
            t_abs = t_abs + shift

        if not (np.nanmax(t_abs) >= las_tmin and np.nanmin(t_abs) <= las_tmax):
            print("   ⚠️  Trajectory times still do not overlap LAS. Consider --force-remap.")

    # Project (lon,lat,alt) EPSG:4979 → target EPSG
    transformer = Transformer.from_crs(4979, target_epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)
    z = alt.astype(float)

    # Validate finite outputs
    bad = (~np.isfinite(x)) | (~np.isfinite(y)) | (~np.isfinite(z)) | (~np.isfinite(t_abs))
    if np.any(bad):
        nbad = int(np.sum(bad))
        print(f"   ❌ Non-finite values after projection: {nbad} rows (e.g., inf/NaN). "
              f"Check lat/lon units and EPSG.")
        # Try fallback: assume degrees if we thought radians (or vice versa)
        # Only attempt once (simple heuristic)
        if is_radians:
            print("   ↩️  Retrying as degrees (fallback)...")
            lon2, lat2 = np.degrees(lon) if looks_like_radians(lat, lon) else (lon, lat)
        else:
            print("   ↩️  Retrying as radians (fallback)...")
            lon2, lat2 = np.radians(lon), np.radians(lat)
            lon2, lat2 = np.degrees(lon2), np.degrees(lat2)  # ensure degrees for transformer
        x, y = transformer.transform(lon2, lat2)
        bad = (~np.isfinite(x)) | (~np.isfinite(y)) | (~np.isfinite(z)) | (~np.isfinite(t_abs))
        if np.any(bad):
            raise SystemExit("Projection still produced non-finite values. Aborting to avoid bad rayimport.")

    # Write rayimport file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ti, xi, yi, zi in zip(t_abs, x, y, z):
            f.write(f"{ti:.6f} {xi:.6f} {yi:.6f} {zi:.6f}\n")

    print(f"   ✅ Wrote rayimport trajectory -> {out_path}")
    print(f"   ⏰ Trajectory time range: {float(np.nanmin(t_abs)):.6f} .. {float(np.nanmax(t_abs)):.6f}")
    if las_tmin is not None:
        print(f"   ⏰ LAS gps_time range   : {las_tmin:.6f} .. {las_tmax:.6f}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Convert SOW/Lat/Lon/Alt trajectory to rayimport 'time x y z'.")
    ap.add_argument("--trajectory", "--traj", required=True, type=Path, help="Trajectory file (text CSV/TXT or SBET-like binary)")
    ap.add_argument("--lidar", "--las", required=False, type=Path, help="Reference LAS with gps_time (for overlap/alignment)")
    ap.add_argument("--output", "--out", required=True, type=Path, help="Output path (space-separated 'time x y z')")
    ap.add_argument("--target-epsg", required=True, type=int, help="Projected EPSG of LAS (e.g., 7854)")
    ap.add_argument("--latlon-in-radians", action="store_true", help="Force: trajectory lat/lon are in radians (disables auto-detect)")

    # Optional explicit column names for text trajectories
    ap.add_argument("--sow-col", default=None, help="Seconds-of-week column name (default: auto)")
    ap.add_argument("--week-col", default=None, help="GPS week column name (default: auto if present)")
    ap.add_argument("--lat-col", default=None, help="Latitude column name (default: auto)")
    ap.add_argument("--lon-col", default=None, help="Longitude column name (default: auto)")
    ap.add_argument("--alt-col", default=None, help="Altitude/height/Z column name (default: auto)")

    # Time handling
    ap.add_argument("--force-remap", action="store_true", help="Remap trajectory times by order into LAS range")
    ap.add_argument("--time-offset", type=float, default=None, help="Additive offset (seconds) to all traj times")

    args = ap.parse_args()

    convert_trajectory_to_rayimport(
        traj_path=args.trajectory,
        out_path=args.output,
        las_path=args.lidar,
        target_epsg=args.target_epsg,
        latlon_in_radians=True if args.latlon_in_radians else None,  # None => auto-detect
        week_col=args.week_col,
        sow_col=args.sow_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        alt_col=args.alt_col,
        force_remap=args.force_remap,
        time_offset=args.time_offset,
    )

if __name__ == "__main__":
    main()
