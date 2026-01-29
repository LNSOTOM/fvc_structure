#!/usr/bin/env micromamba run -n fvc_structure python
"""
Simple 3D structure pipeline (Option A):
- voxel merge
- mallee-only watershed
- treeID only for mallee canopy
"""

from pathlib import Path
from fvc3dstruct.pipeline import process_tile


# Base directory for this step
BASE = Path("/home/nvidia/projects/fvc_structure/phase_4_3D_lidar_prep/step_3_structure_class")

# ========== Site 1 (DD0001) ==========
z_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/")
s_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/")
o_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/site1_simple3d/")

# ========== Site 11 (DD0011) - toggle if needed ==========
# z_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf/")
# s_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp/")
# o_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid/site2_simple3d/")

o_dir.mkdir(parents=True, exist_ok=True)


def main():
    z_files = sorted(z_dir.glob("*.las"))

    print(f"Found {len(z_files)} Z_norm tiles")

    for zf in z_files:
        tid = zf.stem.split('.')[-1]

        sf = s_dir / f"mask_tiles_multispectral.{tid}.las"
        if not sf.exists():
            print(f"⚠ Missing structure file for: {tid}")
            continue

        out_path = o_dir / f"struct3d_{tid}.las"
        print(f"▶ Processing tile {tid} → {out_path.name}")

        process_tile(zf, sf, out_path)

    print("✅ Done.")


if __name__ == "__main__":
    main()
