"""Complete 3D fusion pipeline matching merge_znorm approach."""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import laspy

from .voxel_merge import cartesian_voxel_merge
from .watershed import get_mallee_centroids_watershed
from .watershed import auto_watershed_params
from .polar_refine import refine_with_polar_coords
from .height_stats import (
    compute_class_height_thresholds,
    auto_mallee_min_height,
)
from .io_las import ensure_las14_pf6
from .config import DEFAULT_CONFIG


def process_tile(
    z_path,
    s_path,
    out_path,
    config: Optional[Dict] = None,
):
    """
    Full 3D structure pipeline (Option A: mallee canopy treeID only).
    
    Steps:
    1. Cartesian voxel merge (majority vote)
    2. Watershed crown detection
    3. Polar refinement with height priors
    4. Height sanity rules
    5. TreeID only for mallee crowns above threshold
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    
    # Load inputs
    las_z = laspy.read(str(z_path))
    las_s = laspy.read(str(s_path))
    
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError(f"Z_norm LAS missing 'Z_norm' dimension: {z_path}")
    
    zn_arr = np.asarray(las_z["Z_norm"], dtype=float)
    
    print(f"📂 Processing {z_path.stem}")
    print(f"  Z_norm pts: {len(las_z.points):,}, range: {zn_arr.min():.2f}–{zn_arr.max():.2f} m")
    print(f"  Struct pts: {len(las_s.points):,}")
    
    # Step 1: Data-driven height thresholds
    print("\n🔍 Step 1: Computing class height thresholds...")
    height_ranges = compute_class_height_thresholds(las_s, las_z)
    mallee_min = auto_mallee_min_height(height_ranges)
    print(f"  → Mallee crown min height: {mallee_min:.2f} m")
    
    # Step 2: Cartesian voxel merge
    print("\n📦 Step 2: Cartesian voxel merge...")
    classification = cartesian_voxel_merge(
        las_z,
        las_s,
        voxel_xy=cfg["voxel_size_xy"],
        voxel_z=cfg["voxel_size_z"],
    )
    
    # Step 3: Watershed crown detection
    print("\n🌳 Step 3: Detecting mallee crowns...")

    # Auto-derive parameters if requested
    mt = cfg.get("min_tree_dist", 1.5)
    sg = cfg.get("smoothing_sigma", 1.0)
    use_auto_mt = isinstance(mt, str) and mt.lower() == "auto"
    use_auto_sg = isinstance(sg, str) and sg.lower() == "auto"
    if use_auto_mt or use_auto_sg:
        est_mt, est_sg = auto_watershed_params(las_s, las_z, grid_res=0.10, fallback_min_height=mallee_min)
        if use_auto_mt:
            mt = est_mt
        if use_auto_sg:
            sg = est_sg
        print(f"  → Auto params: min_tree_dist={mt:.2f} m, smoothing_sigma={sg:.2f} px")
    
    label_grid, centroids, grid_min_x, grid_min_y = get_mallee_centroids_watershed(
        las_s,
        grid_res=0.10,
        min_tree_dist=mt,
        smoothing_sigma=sg,
        fallback_points=(las_z.x, las_z.y, zn_arr),
        fallback_min_height=mallee_min,
    )
    
    # Step 4: Compute crown IDs for ALL points (before filtering)
    print("\n🏷️  Step 4: Computing crown IDs from watershed...")
    instance_ids = np.zeros(len(las_z.points), dtype=np.uint16)
    
    if label_grid.max() > 0:
        ix = np.clip(((las_z.x - grid_min_x) / 0.10).astype(int), 0, label_grid.shape[1] - 1)
        iy = np.clip(((las_z.y - grid_min_y) / 0.10).astype(int), 0, label_grid.shape[0] - 1)
        instance_ids = label_grid[iy, ix].astype(np.uint16)
        n_in_crowns = np.count_nonzero(instance_ids)
        print(f"  → {n_in_crowns:,} points fall within {label_grid.max()} crowns")
    
    # Step 5: Polar refinement (if crowns detected)
    if label_grid.max() > 0:
        print("\n🔄 Step 5: Polar refinement around crowns...")
        classification = refine_with_polar_coords(
            las_z,
            las_s,
            classification,
            zn_arr,
            label_grid,
            centroids,
            grid_min_x,
            grid_min_y,
            grid_res=0.10,
            height_ranges=height_ranges,
            r_bin=cfg["r_bin"],
            theta_bin_deg=cfg["theta_bin_deg"],
        )
    else:
        print("\n⚠️  No crowns detected, skipping polar refinement")
    
    # Step 6: Merge woody debris → mallee, apply height rules
    print("\n✂️  Step 6: Final height sanity rules...")
    
    # Merge high woody debris (65) to mallee (68)
    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    mask_65_high = mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)
    mask_65_low = mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM)
    classification[mask_65_high] = 68
    
    # Remove low woody debris (likely noise)
    keep_mask_base = ~mask_65_low
    
    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]
    instance_ids_kept = instance_ids[keep_mask_base]
    
    # Step 7: Filter class-0 and promote high unclassified points
    print("\n🧹 Step 7: Filtering class-0 and promoting high points...")
    HIGH_Z_KEEP = 1.3
    
    # Keep non-zero classes OR high class-0 points
    keep_final = (classification_kept != 0) | (zn_kept >= HIGH_Z_KEEP)
    
    classification_final = classification_kept[keep_final]
    zn_final = zn_kept[keep_final]
    instance_ids_final = instance_ids_kept[keep_final]
    
    # Promote high class-0 to woody debris, then merge to mallee
    promote = (classification_final == 0) & (zn_final >= HIGH_Z_KEEP)
    classification_final[promote] = 65
    classification_final[classification_final == 65] = 68  # Merge all 65 → 68
    
    n_kept = len(classification_final)
    n_original = len(las_z.points)
    pct_kept = 100.0 * n_kept / n_original
    print(f"  → Kept {n_kept:,}/{n_original:,} points ({pct_kept:.1f}%)")
    
    # Step 8: Spatial majority voting within crowns
    print("\n🗳️  Step 8: Spatial majority voting within crowns...")
    
    # For points within crowns (instance_ids > 0), apply majority voting
    # Reclassify ALL non-mallee classes to 68 if within a crown that's predominantly 68
    if instance_ids_final.max() > 0:
        n_reclassified = 0
        for crown_id in range(1, int(instance_ids_final.max()) + 1):
            in_crown = instance_ids_final == crown_id
            if in_crown.sum() == 0:
                continue
            
            crown_classes = classification_final[in_crown]
            # Count mallee (68) vs other classes
            n_mallee = (crown_classes == 68).sum()
            n_total = len(crown_classes)
            
            # If >50% of crown is already mallee, convert ALL non-mallee → 68
            if n_mallee / n_total > 0.5:
                is_non_mallee_in_crown = in_crown & (classification_final != 68)
                n_converted = is_non_mallee_in_crown.sum()
                if n_converted > 0:
                    classification_final[is_non_mallee_in_crown] = 68
                    n_reclassified += n_converted
        
        if n_reclassified > 0:
            print(f"  → Reclassified {n_reclassified:,} points to mallee (68) via majority vote")
            print(f"     (Converted classes 64, 66, 67 within mallee-dominant crowns)")

        # Height-based refinement: within crown, promote high points to mallee
        # Rationale: Crown definition uses a mallee_min height; points above this
        # threshold that lie inside crowns should be mallee (68).
        n_reclassified_height = 0
        in_any_crown = instance_ids_final > 0
        high_in_crown = in_any_crown & (zn_final >= mallee_min) & (classification_final != 68)
        if high_in_crown.any():
            n_reclassified_height = high_in_crown.sum()
            classification_final[high_in_crown] = 68
            print(f"  → Reclassified {n_reclassified_height:,} high crown points → 68 (height rule)")
    
    # Step 9: Finalize treeIDs (mallee crowns only)
    print("\n🌳 Step 9: Finalizing treeIDs (mallee crowns only)...")
    treeID = instance_ids_final.copy()
    
    # Only keep treeID for mallee canopy points above threshold
    is_mallee_crown = (classification_final == 68) & (zn_final >= mallee_min)
    treeID[~is_mallee_crown] = 0
    
    unique_trees = len(np.unique(treeID[treeID > 0]))
    print(f"  → Assigned {unique_trees} unique treeIDs to mallee crowns")
    
    # Build final keep mask for writing
    keep_mask = np.zeros(len(las_z.points), dtype=bool)
    keep_indices = np.where(keep_mask_base)[0][keep_final]
    keep_mask[keep_indices] = True
    
    # Step 10: Write output
    print("\n💾 Step 10: Writing output LAS...")
    las_base = ensure_las14_pf6(las_z)
    subset_pts = las_base.points[keep_mask].copy()
    
    las_out = laspy.LasData(las_base.header)
    las_out.points = subset_pts
    
    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    
    las_out.classification = classification_final
    las_out["Z_norm"] = zn_final.astype(np.float32)
    las_out["treeID"] = treeID
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))
    
    print(f"✅ Wrote: {Path(out_path).name}")
    print(f"   {len(las_out.points):,} points, {unique_trees} trees\n")


def process_folder(
    z_dir: Path,
    s_dir: Path,
    out_dir: Path,
    site_name: str = "",
    config: Optional[Dict] = None,
):
    """Batch process all matching LAS tile pairs in the given folders.

    Matching pattern:
      - Z_norm:  z_norm_csf_tiles_multispectral.{tile_id}.las
      - Struct:  mask_tiles_multispectral.{tile_id}.las

    Outputs will be written to `out_dir` with filenames:
      {site_name_}tile{tile_id}_struct_polar.las

    Returns a list of output file paths processed successfully.
    """
    z_dir = Path(z_dir)
    s_dir = Path(s_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_files = sorted(z_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    outputs = []
    skipped = []

    if not z_files:
        print(f"⚠️  No Z_norm tiles found in {z_dir}")
        return outputs

    print(f"📚 Found {len(z_files)} candidate Z_norm tiles in {z_dir}")

    for z_path in z_files:
        parts = z_path.name.split(".")
        # Expected: ["z_norm_csf_tiles_multispectral", "{tile_id}", "las"]
        if len(parts) < 3:
            skipped.append((z_path, "unexpected filename pattern"))
            continue
        tile_id = parts[-2]
        s_path = s_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not s_path.exists():
            skipped.append((z_path, "missing matching structure LAS"))
            continue

        out_name_prefix = f"{site_name}_" if site_name else ""
        out_path = out_dir / f"{out_name_prefix}tile{tile_id}_struct_polar.las"
        print("-" * 60)
        print(f"▶️  Processing tile {tile_id} → {out_path.name}")
        try:
            process_tile(z_path, s_path, out_path, config=config)
            outputs.append(out_path)
        except Exception as e:
            skipped.append((z_path, str(e)))
            print(f"❌ Skipped {z_path.name}: {e}")

    print("=" * 60)
    print(f"✅ Completed batch: {len(outputs)} processed, {len(skipped)} skipped")
    if skipped:
        print("Skipped details:")
        for zp, reason in skipped:
            print(f"  - {zp.name}: {reason}")

    return outputs
