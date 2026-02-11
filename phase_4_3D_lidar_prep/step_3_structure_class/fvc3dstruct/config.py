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
    "write_full_fused": False,
    "write_veg_mask_npy": False,
    "save_stage_outputs": False,
    "stage_out_dir": None,
    # Watershed parameters
    "min_tree_dist": 1.5,      # Minimum distance between tree peaks (m)
    "smoothing_sigma": 1.0,      # Gaussian smoothing for CHM
}
