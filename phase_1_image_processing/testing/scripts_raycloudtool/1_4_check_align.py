#!/usr/bin/env python3
import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

def pdal_bounds(path: Path) -> Optional[Tuple[float,float,float,float,float,float]]:
    """Return (minx, miny, minz, maxx, maxy, maxz) via `pdal info --summary`."""
    if shutil.which("pdal") is None:
        return None
    try:
        out = subprocess.run(
            ["pdal", "info", "--summary", str(path)],
            capture_output=True, text=True, timeout=120
        )
        out.check_returncode()
        js = json.loads(out.stdout)
        b = js["summary"]["bounds"]
        # PDAL can return strings like: {"minx":"...", "maxx":"..."} or nested dicts
        def as_float(v): return float(v) if not isinstance(v, dict) else float(v.get("minimum", 0))
        minx = as_float(b["minx"]); miny = as_float(b["miny"]); minz = as_float(b["minz"])
        maxx = as_float(b["maxx"]); maxy = as_float(b["maxy"]); maxz = as_float(b["maxz"])
        return (minx, miny, minz, maxx, maxy, maxz)
    except Exception as e:
        print(f"⚠️  PDAL bounds failed for {path}: {e}")
        return None

def ply_centroid_with_plyfile(path: Path) -> Optional[Tuple[float,float,float]]:
    """Fallback: use plyfile to compute centroid for PLY (requires `pip install plyfile`)."""
    try:
        from plyfile import PlyData
        pd = PlyData.read(str(path))
        if "vertex" not in pd:
            return None
        vx = pd["vertex"]
        import numpy as np
        x = vx["x"]; y = vx["y"]; z = vx["z"]
        return (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
    except Exception as e:
        print(f"⚠️  plyfile fallback failed for {path}: {e}")
        return None

def centroid_from_bounds(b):
    minx, miny, minz, maxx, maxy, maxz = b
    cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy); cz = 0.5*(minz+maxz)
    return (cx, cy, cz)

def get_centroid(path: Path) -> Optional[Tuple[float,float,float]]:
    ext = path.suffix.lower()
    # Try PDAL first (works for LAS/LAZ/PLY)
    b = pdal_bounds(path)
    if b is not None:
        return centroid_from_bounds(b)
    # Fallback only for PLY if PDAL missing
    if ext == ".ply":
        return ply_centroid_with_plyfile(path)
    return None

def pretty_delta(a, b):
    return (b[0]-a[0], b[1]-a[1], b[2]-a[2])

def main():
    # Reference (L1) and moving (L2)
    L1 = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.ply")
    # L2 = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c_coarse_aligned.ply")
    L2 = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c_aligned_shifted.ply")  # unaligned version

    for p in (L1, L2):
        if not p.exists():
            print(f"❌ Missing: {p}")
            return

    c1 = get_centroid(L1)
    c2 = get_centroid(L2)

    if c1 is None or c2 is None:
        print("❌ Could not compute centroids (check PDAL is installed or install `plyfile`).")
        return

    dx, dy, dz = pretty_delta(c1, c2)
    print(f"L1 centroid: {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}")
    print(f"L2 centroid: {c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}")
    print(f"Δ (L2 - L1): dx={dx:.3f} m, dy={dy:.3f} m, dz={dz:.3f} m")

    # Also print bounds so you can eyeball overlap
    b1 = pdal_bounds(L1); b2 = pdal_bounds(L2)
    if b1 and b2:
        print("\nL1 bounds (minx,miny,minz,maxx,maxy,maxz):")
        print("  ", ", ".join(f"{v:.3f}" for v in b1))
        print("L2 bounds (minx,miny,minz,maxx,maxy,maxz):")
        print("  ", ", ".join(f"{v:.3f}" for v in b2))

if __name__ == "__main__":
    main()
