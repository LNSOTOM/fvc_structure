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
    # Watershed parameters
    "min_tree_dist": 1.5,      # Minimum distance between tree peaks (m)
    "smoothing_sigma": 1.0,      # Gaussian smoothing for CHM
}
