"""Complete 3D fusion pipeline matching the Table-3 workflow."""

from pathlib import Path
from typing import Dict, Optional

import laspy
import numpy as np

from .voxel_merge import cartesian_voxel_merge
from .watershed import get_mallee_centroids_watershed
from .watershed import auto_watershed_params
from .polar_refine import refine_with_polar_coords
from .height_stats import (
    compute_class_height_thresholds,
    auto_mallee_min_height,
)
from .io_las import write_stage_output, write_las_with_fields
from .config import DEFAULT_CONFIG
from .treeid import assign_treeID_optionA


def process_tile(
    z_path,
    s_path,
    out_path,
    config: Optional[Dict] = None,
):
    """
        Occlusion-aware 3D vegetation-structure label fusion workflow.

        Stages (matching Table 3):
            1) Class height thresholding
            2) Cartesian voxel-based label transfer
            3) Mallee crown detection and instance assignment (treeIDs)
            4) Crown-centric polar refinement with height priors
            5) Height-based consolidation and crown-level consistency enforcement
            6) Output generation

        Notes:
            - `out_path` is the vegetation-only (non-zero) LAS product.
            - Optionally writes a full fused LAS (same points as `z_path`) when
                `write_full_fused=True` in config.
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
    
    print(f"📂 Processing {Path(z_path).stem}")
    print(f"  Z_norm pts: {len(las_z.points):,}, range: {zn_arr.min():.2f}–{zn_arr.max():.2f} m")
    print(f"  Struct pts: {len(las_s.points):,}")
    
    # Stage 1: Class height thresholding
    print("\n1️⃣  Stage 1: Class height thresholding (robust priors)...")
    height_ranges = compute_class_height_thresholds(
        las_s,
        las_z,
        percentile_low=15.0,
        percentile_high=85.0,
        max_match_dist=0.30,
    )
    mallee_min = auto_mallee_min_height(height_ranges)
    mallee_min_override = cfg.get("mallee_min_znorm", None)
    if mallee_min_override is not None:
        mallee_min = float(mallee_min_override)
        print(f"  → Mallee crown min height: {mallee_min:.2f} m (override)")
    else:
        print(f"  → Mallee crown min height: {mallee_min:.2f} m")
    
    # Stage 2: Cartesian voxel-based label transfer
    print("\n2️⃣  Stage 2: Cartesian voxel-based label transfer (baseline)...")
    classification_baseline, voxel_stats = cartesian_voxel_merge(
        las_z,
        las_s,
        voxel_xy=cfg["voxel_size_xy"],
        voxel_z=cfg["voxel_size_z"],
        return_stats=True,
    )

    save_stages = bool(cfg.get("save_stage_outputs", False))
    stage_dir = cfg.get("stage_out_dir")
    if save_stages and stage_dir:
        stage_dir = Path(stage_dir)
        tile = Path(z_path).stem
        write_stage_output(
            las_template=las_z,
            classification=classification_baseline,
            output_path=stage_dir / f"{tile}_stage2_voxel_baseline.las",
            zn_values=zn_arr,
            stage_label="Stage 2 (voxel baseline)",
        )
    
    # Stage 3: Mallee crown detection + per-point instance assignment
    print("\n3️⃣  Stage 3: Mallee crown detection + instance assignment...")

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

    instance_ids_all = assign_treeID_optionA(
        las_z,
        label_grid=label_grid,
        xmin=grid_min_x,
        ymin=grid_min_y,
        grid_res=0.10,
    )
    if label_grid.max() > 0:
        n_in_crowns = int(np.count_nonzero(instance_ids_all))
        print(f"  → {n_in_crowns:,} points fall within {int(label_grid.max())} crowns")
    else:
        print("  ⚠️ No crowns detected (label_grid max == 0)")

    if save_stages and stage_dir:
        tile = Path(z_path).stem
        write_stage_output(
            las_template=las_z,
            classification=classification_baseline,
            output_path=stage_dir / f"{tile}_stage3_treeid_instances.las",
            zn_values=zn_arr,
            stage_label="Stage 3 (treeID instances)",
            tree_ids=instance_ids_all,
        )
    
    # Stage 4: Crown-centric polar refinement with height priors
    classification_refined = classification_baseline.copy()
    stage4_changed = 0
    stage4_changed_pct = 0.0
    if label_grid.max() > 0:
        print("\n4️⃣  Stage 4: Crown-centric polar refinement with height priors...")
        classification_refined = refine_with_polar_coords(
            las_z,
            las_s,
            classification_refined,
            zn_arr,
            label_grid,
            centroids,
            grid_min_x,
            grid_min_y,
            grid_res=0.10,
            height_ranges=height_ranges,
            r_bin=cfg["r_bin"],
            theta_bin_deg=cfg["theta_bin_deg"],
            min_conf=cfg["min_conf"],
            min_flip_ratio=cfg["min_flip_ratio"],
        )
        stage4_changed = int(np.count_nonzero(classification_refined != classification_baseline))
        stage4_changed_pct = (
            100.0 * stage4_changed / len(classification_refined)
            if len(classification_refined) > 0
            else 0.0
        )
        print(
            f"  → Stage-4 label changes: {stage4_changed:,}/{len(classification_refined):,} "
            f"({stage4_changed_pct:.1f}%)"
        )
    else:
        print("\n4️⃣  Stage 4: Skipping polar refinement (no crowns)")

    if save_stages and stage_dir:
        tile = Path(z_path).stem
        write_stage_output(
            las_template=las_z,
            classification=classification_refined,
            output_path=stage_dir / f"{tile}_stage4_polar_refined.las",
            zn_values=zn_arr,
            stage_label="Stage 4 (polar refined)",
            tree_ids=instance_ids_all,
        )
    
    # Stage 5: Height-based consolidation + crown-level consistency
    print("\n5️⃣  Stage 5: Height-based consolidation + crown consistency...")

    classification_final_all = classification_refined.copy()
    merge_65_to_68 = float(cfg.get("merge_65_to_68_znorm", 0.5))
    promote_unlabelled = float(cfg.get("promote_unlabelled_znorm", 1.3))

    # Merge woody debris (65) → mallee (68) if tall enough; remove low woody debris.
    mask_65 = classification_final_all == 65
    classification_final_all[mask_65 & (zn_arr > merge_65_to_68)] = 68
    classification_final_all[mask_65 & (zn_arr <= merge_65_to_68)] = 0

    # Promote unlabelled points above threshold (gap-aware fill)
    promote_mask = (classification_final_all == 0) & (zn_arr >= promote_unlabelled)
    promoted_points = int(np.count_nonzero(promote_mask))
    if promoted_points:
        classification_final_all[promote_mask] = 68
        print(f"  → Promoted {promoted_points:,} high unlabelled points → 68")

    # Crown-level majority voting: if crown is >50% mallee, convert all points in crown → 68.
    n_reclassified = 0
    if instance_ids_all.max() > 0:
        max_id = int(instance_ids_all.max())
        for crown_id in range(1, max_id + 1):
            in_crown = instance_ids_all == crown_id
            if not np.any(in_crown):
                continue

            crown_classes = classification_final_all[in_crown]
            n_mallee = int(np.count_nonzero(crown_classes == 68))
            n_total = int(crown_classes.size)
            if n_total == 0:
                continue

            if (n_mallee / n_total) > 0.5:
                non_mallee = in_crown & (classification_final_all != 68)
                n_conv = int(np.count_nonzero(non_mallee))
                if n_conv:
                    classification_final_all[non_mallee] = 68
                    n_reclassified += n_conv

        if n_reclassified:
            print(f"  → Reclassified {n_reclassified:,} crown points → 68 via majority vote")

    # Optional: force mallee inside detected crowns above a chosen height.
    # This prevents tall points in crowns being kept as shrub (67) when you
    # want a strict Z_norm cutoff (e.g., 1.8 m).
    force_thr = cfg.get("mallee_force_min_znorm", None)
    n_forced = 0
    if force_thr is not None and instance_ids_all.max() > 0:
        force_thr = float(force_thr)
        force_mask = (instance_ids_all > 0) & (zn_arr >= force_thr)
        n_forced = int(np.count_nonzero(force_mask & (classification_final_all != 68)))
        if n_forced:
            classification_final_all[force_mask] = 68
            print(f"  → Forced {n_forced:,} crown points → 68 for Z_norm >= {force_thr:.2f} m")

    # Finalise valid treeIDs (Option A canopy only)
    treeID_all = instance_ids_all.copy().astype(np.uint16)
    is_valid_mallee_canopy = (classification_final_all == 68) & (zn_arr >= mallee_min)
    treeID_all[~is_valid_mallee_canopy] = 0

    # Safety: any point that keeps a treeID must be class 68 in the written output.
    if treeID_all.max() > 0:
        classification_final_all[treeID_all > 0] = 68

    # Optional: collapse all non-zero IDs to a single ID=1 in exported products.
    if bool(cfg.get("collapse_treeid_to_one", False)):
        treeID_all[treeID_all > 0] = 1

    unique_trees = int(len(np.unique(treeID_all[treeID_all > 0])))
    print(f"  → Valid mallee treeIDs: {unique_trees}")

    if save_stages and stage_dir:
        tile = Path(z_path).stem
        write_stage_output(
            las_template=las_z,
            classification=classification_final_all,
            output_path=stage_dir / f"{tile}_stage5_final_fullcloud.las",
            zn_values=zn_arr,
            stage_label="Stage 5 (final full-cloud)",
            tree_ids=treeID_all,
        )

    # Stage 6: Output generation
    print("\n6️⃣  Stage 6: Output generation...")

    # Vegetation-only product (non-zero classes)
    keep_veg = classification_final_all != 0
    n_kept = int(np.count_nonzero(keep_veg))
    n_original = int(len(las_z.points))
    pct_kept = 100.0 * n_kept / n_original if n_original else 0.0
    print(f"  → Vegetation-only: {n_kept:,}/{n_original:,} points ({pct_kept:.1f}%)")

    write_outputs = bool(cfg.get("write_outputs", True))
    if write_outputs:
        write_las_with_fields(
            las_template=las_z,
            out_path=out_path,
            classification=classification_final_all,
            zn_values=zn_arr,
            tree_ids=treeID_all,
            keep_mask=keep_veg,
        )
        print(f"✅ Wrote veg-only LAS: {Path(out_path).name}")
    else:
        print("ℹ️  Skipped LAS writing (write_outputs=False)")

    # Full fused point cloud (gap-aware) – optional
    if write_outputs and bool(cfg.get("write_full_fused", False)):
        out_full = Path(out_path)
        out_full = out_full.with_name(out_full.stem + "_full_fused" + out_full.suffix)
        write_las_with_fields(
            las_template=las_z,
            out_path=out_full,
            classification=classification_final_all,
            zn_values=zn_arr,
            tree_ids=treeID_all,
            keep_mask=None,
        )
        print(f"✅ Wrote full fused LAS: {out_full.name}")

    # Vegetation-only mask product – optional (boolean mask aligned to full Z_norm LAS)
    if write_outputs and bool(cfg.get("write_veg_mask_npy", False)):
        out_mask = Path(out_path)
        out_mask = out_mask.with_name(out_mask.stem + "_veg_mask.npy")
        np.save(out_mask, keep_veg)
        print(f"✅ Wrote veg mask: {out_mask.name}")

    print(f"   TreeIDs (unique, non-zero): {unique_trees}\n")

    n_in_crowns = int(np.count_nonzero(instance_ids_all))
    return {
        "tile": Path(z_path).stem,
        "voxel_size_xy": float(cfg["voxel_size_xy"]),
        "voxel_size_z": float(cfg["voxel_size_z"]),
        "z_points": int(len(las_z.points)),
        "structure_points": int(len(las_s.points)),
        "mallee_min_znorm": float(mallee_min),
        "stage2_nonzero_structure_voxels": int(voxel_stats["nonzero_structure_voxels"]),
        "stage2_labeled_points": int(voxel_stats["labeled_points"]),
        "stage2_labeled_pct": float(voxel_stats["labeled_pct"]),
        "stage3_watershed_crowns": int(label_grid.max()),
        "stage3_points_in_crowns": n_in_crowns,
        "stage4_changed_points": stage4_changed,
        "stage4_changed_pct": float(stage4_changed_pct),
        "stage5_promoted_points": promoted_points,
        "stage5_majority_reclassified_points": n_reclassified,
        "stage5_forced_points": n_forced,
        "final_veg_points": n_kept,
        "final_veg_pct": float(pct_kept),
        "final_treeids": unique_trees,
        "write_outputs": write_outputs,
    }


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
