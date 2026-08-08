# Default configuration matching merge_znorm pipeline
DEFAULT_CONFIG = {
    "voxel_size_xy": 0.10,
    "voxel_size_z": 0.10,
    "ground_thresh": 0.15,
    "r_bin": 0.10,
    "theta_bin_deg": 10.0,
    "min_conf": 0.3,
    "min_flip_ratio": 1.5,
    "max_fill_height": 2.0,
    "enable_radial_fill": False,
    # Stage 5 consolidation thresholds
    "merge_65_to_68_znorm": 0.5,
    "promote_unlabelled_znorm": 1.3,
    # Stage-output / product generation
    "write_outputs": True,
    "write_full_fused": False,
    "write_veg_mask_npy": False,
    # If True, include the full per-point Stage-2 baseline and final classification
    # arrays in process_tile()'s return dict (for class-transition/confusion analyses).
    "return_classification_arrays": False,
    # If True, collapse any non-zero treeID to 1 in the written outputs.
    # (This does not change crown detection; it only changes the exported IDs.)
    "collapse_treeid_to_one": False,
    "save_stage_outputs": False,
    "stage_out_dir": None,
    # Option A (mallee canopy) height controls
    # If set (meters), overrides the auto-derived mallee crown min height used for:
    #   - watershed fallback canopy signal threshold
    #   - final treeID validity mask (treeIDs only kept for Z_norm >= mallee_min_znorm)
    "mallee_min_znorm": None,
    # If set (meters), forces points *inside detected crowns* with Z_norm >= threshold
    # to be classified as mallee (68). Useful when tall shrub labels leak into crowns.
    "mallee_force_min_znorm": None,
    # Watershed parameters
    "min_tree_dist": 1.5,      # Minimum distance between tree peaks (m)
    "smoothing_sigma": 1.0,      # Gaussian smoothing for CHM
}
