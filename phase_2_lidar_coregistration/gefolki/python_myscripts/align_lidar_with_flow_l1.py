
# %%
##test 2
import sys
import numpy as np
import rasterio
import laspy
from pathlib import Path
from scipy.ndimage import map_coordinates, gaussian_filter
import time
from tqdm import tqdm

from phase2_config import load_config

# --- CONFIGURATION (align section of the job config) ---
align_cfg = load_config()["align"]

# --- KEY PARAMETER: Control the direction of the correction ---
# Use 1 to apply the flow as-is (e.g., X_new = X_old + dX).
# Use -1 to invert the flow if visual checks show the correction is backwards.
CORRECTION_DIRECTION = int(align_cfg.get("correction_direction", -1))

# --- KEY PARAMETER: Control smoothing of the flow field ---
# This is the most important parameter for fixing holes.
# Start with a value like 2.0.
# Increase it (e.g., 3.0, 4.0) if holes persist.
# Decrease it (e.g., 1.0) if the correction seems "blurry" or "damped".
GAUSSIAN_SIGMA = float(align_cfg.get("gaussian_sigma", 3.0))

# --- INPUT FILE PATHS ---
# The original, full-density LAS/LAZ file to be corrected
LAS_INPUT_PATH = Path(align_cfg["las_input"])

# The raster grid that defines the coordinate system for the flow fields.
# This MUST have the exact same dimensions as the flow field arrays.
REFERENCE_GRID_PATH = Path(align_cfg["reference_grid"])

# The displacement fields calculated by gefolki
FLOW_U_PATH = Path(align_cfg["flow_u"])
FLOW_V_PATH = Path(align_cfg["flow_v"])

# --- OUTPUT FILE PATH ---
# Where the new, fully coregistered LAS file will be saved
LAS_OUTPUT_PATH = (
    Path(align_cfg["las_output"])
    if align_cfg.get("las_output")
    else LAS_INPUT_PATH.with_name(LAS_INPUT_PATH.stem + "_coreg.las")
)


def apply_full_flow_to_las(las_in_path, las_out_path, grid_path, flow_u, flow_v, direction=1):
    """Applies the full, dense optical flow displacement field to a LAS point cloud."""
    print(f"--- Applying Full Non-Linear Correction (Direction: {direction}) ---")
    start_time = time.time()

    # 1. Get the coordinate transformation "dictionary" from the reference raster.
    # This allows us to translate between World coordinates (meters) and Pixel coordinates.
    with rasterio.open(grid_path) as src:
        transform = src.transform  # Translates from Pixel -> World (meters)
        H, W = src.height, src.width
        inv = ~transform           # The inverse: translates from World (meters) -> Pixel

    # Sanity check to ensure our map (flow field) and globe (reference grid) match
    if flow_u.shape != (H, W) or flow_v.shape != (H, W):
        raise ValueError(f"Flow shape {flow_u.shape} does not match raster size {(H, W)}")

    # 2. Open the input and output LAS files, streaming the data in chunks.
    with laspy.open(las_in_path) as reader:
        header = reader.header
        pbar = tqdm(total=header.point_count, desc="Applying Full Flow", unit="pts")
        
        with laspy.open(las_out_path, mode="w", header=header) as writer:
            for pts in reader.chunk_iterator(2_000_000): # Process 2 million points at a time
                
                # 3. For each point, find its address (row, col) on the flow field grid.
                col, row = inv * (pts.x, pts.y)
                
                # 4. Keep only the points that actually fall within the correction grid.
                #    We clip to W-1 and H-1 because interpolation needs valid neighbors.
                valid_mask = (col >= 0) & (col < W - 1) & (row >= 0) & (row < H - 1)
                if not np.any(valid_mask):
                    pbar.update(len(pts))
                    continue # Skip this chunk if no points overlap

                valid_pts = pts[valid_mask]
                coords = np.vstack((row[valid_mask], col[valid_mask]))

                # 5. Look up the precise correction (in pixels) for each point's address.
                #    `map_coordinates` interpolates the value for sub-pixel locations.
                #
                #    --- IMPROVEMENT ---
                #    Using order=3 (bicubic) gives a much smoother interpolation
                #    than order=1 (bilinear) or order=2 (quadratic),
                #    which further helps prevent artifacts.
                # du_pix = map_coordinates(flow_u, coords, order=3, mode="reflect")
                # dv_pix = map_coordinates(flow_v, coords, order=3, mode="reflect")
                du_pix = map_coordinates(flow_u, coords, order=2, mode="nearest")
                dv_pix = map_coordinates(flow_v, coords, order=2, mode="nearest")
                # du_pix = map_coordinates(flow_u, coords, order=3, mode="nearest")
                # dv_pix = map_coordinates(flow_v, coords, order=3, mode="nearest")


                # 6. Translate the correction from pixels back to meters.
                dx_world = du_pix * transform.a  # transform.a = pixel width in meters
                dy_world = dv_pix * transform.e  # transform.e = pixel height in meters

                # 7. Apply the final, specific correction to each point.
                valid_pts.x += direction * dx_world
                valid_pts.y += direction * dy_world

                # 8. Write the corrected points to the new file.
                writer.write_points(valid_pts)
                pbar.update(len(pts))
        pbar.close()

    end_time = time.time()
    print(f"\n✅ Full coregistration complete in {end_time - start_time:.2f} seconds.")
    print(f"   Saved final point cloud to: {las_out_path}")


if __name__ == '__main__':
    # --- Pre-flight check for all necessary input files ---
    input_files = [LAS_INPUT_PATH, REFERENCE_GRID_PATH, FLOW_U_PATH, FLOW_V_PATH]
    for f in input_files:
        if not f.exists():
            print(f"FATAL ERROR: Input file not found at: {f}")
            sys.exit(1)

    print("All input files found. Loading flow fields...")
    u_raw = np.load(FLOW_U_PATH)
    v_raw = np.load(FLOW_V_PATH)
    print("Flow fields loaded successfully.")

    # --- NEW: Smooth the flow fields ---
    # This is the key step to prevent holes.
    # Noisy, pixel-to-pixel variations in the flow field can "rip"
    # the dense point cloud apart, creating gaps. Smoothing
    # ensures the displacement is locally consistent.
    print(f"Smoothing flow fields with Gaussian filter (sigma={GAUSSIAN_SIGMA})...")
    u_smooth = gaussian_filter(u_raw, sigma=GAUSSIAN_SIGMA)
    v_smooth = gaussian_filter(v_raw, sigma=GAUSSIAN_SIGMA)
    print("Smoothing complete.")

    # --- Run the main function ---
    apply_full_flow_to_las(
        las_in_path=LAS_INPUT_PATH,
        las_out_path=LAS_OUTPUT_PATH,
        grid_path=REFERENCE_GRID_PATH,
        flow_u=u_smooth,  # Pass the SMOOTHED u-flow
        flow_v=v_smooth,  # Pass the SMOOTHED v-flow
        direction=CORRECTION_DIRECTION
    )
# %%
