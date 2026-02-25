#%%
import sys
import numpy as np
import rasterio
import laspy
from pathlib import Path
from tqdm import tqdm
import time

# --- CONFIGURATION: INPUT FILE PATHS ---

# 1. The original, full-density LAS/LAZ file that needs to be shifted
LAS_INPUT_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")

# 2. The displacement fields calculated by the previous gefolki script
FLOW_U_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_03.npy")
FLOW_V_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_03.npy")

# 3. The reference grid that MUST have the same dimensions as the flow fields.
#    This is used to convert the flow fields from pixels to meters.
REFERENCE_GRID_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03_coreg.tif")

# --- CONFIGURATION: OUTPUT FILE PATH ---
# Where the new, globally translated LAS file will be saved
LAS_OUTPUT_PATH = LAS_INPUT_PATH.with_name(LAS_INPUT_PATH.stem + "_coreg.las")


def calculate_median_translation(flow_u_path: Path, flow_v_path: Path, grid_path: Path) -> tuple[float, float]:
    """
    Analyzes flow fields to calculate the median X and Y translation in meters.

    Args:
        flow_u_path (Path): Path to the U-component of the flow field (.npy).
        flow_v_path (Path): Path to the V-component of the flow field (.npy).
        grid_path (Path): Path to the reference raster that defines the flow grid.

    Returns:
        A tuple containing the median X and Y translation in meters (dx, dy).
    """
    print("--- Step 1/2: Calculating Median Translation from Flow Fields ---")
    
    # Load the flow fields
    flow_u = np.load(flow_u_path)
    flow_v = np.load(flow_v_path)

    # Open the reference grid to get the pixel-to-meter conversion factor
    with rasterio.open(grid_path) as src:
        T = src.transform
        pixel_width = T.a   # Pixel width in meters (typically > 0)
        pixel_height = T.e  # Pixel height in meters (typically < 0 for north-up)

    # Convert the entire flow field from pixel displacements to meter displacements
    dx_m = flow_u * pixel_width
    dy_m = flow_v * pixel_height

    # Create a mask to exclude invalid (e.g., NaN) and near-zero values for robust statistics
    magnitude_m = np.hypot(dx_m, dy_m)
    valid_mask = np.isfinite(magnitude_m) & (magnitude_m > 0.0001) # Filter shifts < 0.1mm
    
    if not np.any(valid_mask):
        print("Warning: No significant displacement values found in flow fields.")
        return 0.0, 0.0

    # Calculate the median of the valid displacement values
    median_x = np.median(dx_m[valid_mask])
    median_y = np.median(dy_m[valid_mask])
    
    print(f"Calculation complete.")
    return median_x, median_y


def apply_global_translation(las_in_path: Path, las_out_path: Path, dx: float, dy: float):
    """
    Applies a constant X, Y translation to all points in a LAS file.
    """
    print("\n--- Step 2/2: Applying Global Translation to Point Cloud ---")
    print(f"Input file:  {las_in_path.name}")
    print(f"Output file: {las_out_path.name}")
    print(f"Applying Translation (dx, dy): ({dx:+.4f} m, {dy:+.4f} m)")

    with laspy.open(las_in_path) as reader:
        header = reader.header
        pbar = tqdm(total=header.point_count, desc="Translating Points", unit="pts")

        with laspy.open(las_out_path, mode="w", header=header) as writer:
            for pts in reader.chunk_iterator(2_000_000):
                # Apply the constant shift to the X and Y coordinates
                pts.x += dx
                pts.y += dy
                # Z coordinate is unchanged
                
                writer.write_points(pts)
                pbar.update(len(pts))
        pbar.close()

    print("\n✅ Translation complete.")


if __name__ == '__main__':
    # --- Pre-flight check for all necessary input files ---
    input_files = [LAS_INPUT_PATH, FLOW_U_PATH, FLOW_V_PATH, REFERENCE_GRID_PATH]
    all_files_found = True
    for f in input_files:
        if not f.exists():
            print(f"FATAL ERROR: Input file not found at: {f}")
            all_files_found = False
    
    if not all_files_found:
        sys.exit(1)
        
    print("All input files found. Starting process...")

    # --- Run the automated workflow ---
    # Step 1: Calculate the median shift
    dx_median, dy_median = calculate_median_translation(
        flow_u_path=FLOW_U_PATH,
        flow_v_path=FLOW_V_PATH,
        grid_path=REFERENCE_GRID_PATH
    )

    # Step 2: Apply the calculated shift to the LAS file
    apply_global_translation(
        las_in_path=LAS_INPUT_PATH,
        las_out_path=LAS_OUTPUT_PATH,
        dx=dx_median,
        dy=dy_median
    )



# %%
##test 2
import sys
import numpy as np
import rasterio
import laspy
from pathlib import Path
from scipy.ndimage import map_coordinates
import time
from tqdm import tqdm

# --- CONFIGURATION ---

# Add the path to your gefolki python scripts if not in a standard location
GEFOLKI_PYTHON_PATH = '/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/gefolki/python'
if GEFOLKI_PYTHON_PATH not in sys.path:
    sys.path.append(GEFOLKI_PYTHON_PATH)

# --- KEY PARAMETER: Control the direction of the correction ---
# Use 1 to apply the flow as-is (e.g., X_new = X_old + dX).
# Use -1 to invert the flow if visual checks show the correction is backwards.
CORRECTION_DIRECTION = -1

# --- INPUT FILE PATHS ---
# The original, full-density LAS/LAZ file to be corrected
LAS_INPUT_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")

# The raster grid that defines the coordinate system for the flow fields.
# This MUST have the exact same dimensions as the flow field arrays.
REFERENCE_GRID_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03_coreg_simple.tif")

# The displacement fields calculated by gefolki
FLOW_U_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_03.npy")
FLOW_V_PATH = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_03.npy")

# --- OUTPUT FILE PATH ---
# Where the new, fully coregistered LAS file will be saved
LAS_OUTPUT_PATH = LAS_INPUT_PATH.with_name(LAS_INPUT_PATH.stem + "_coreg.las")


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
                valid_mask = (col >= 0) & (col < W - 1) & (row >= 0) & (row < H - 1)
                if not np.any(valid_mask):
                    pbar.update(len(pts))
                    continue # Skip this chunk if no points overlap

                valid_pts = pts[valid_mask]
                coords = np.vstack((row[valid_mask], col[valid_mask]))

                # 5. Look up the precise correction (in pixels) for each point's address.
                #    `map_coordinates` interpolates the value for sub-pixel locations.
                du_pix = map_coordinates(flow_u, coords, order=1, mode="nearest")
                dv_pix = map_coordinates(flow_v, coords, order=1, mode="nearest")

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
    u = np.load(FLOW_U_PATH)
    v = np.load(FLOW_V_PATH)
    print("Flow fields loaded successfully.")

    # --- Run the main function ---
    apply_full_flow_to_las(
        las_in_path=LAS_INPUT_PATH,
        las_out_path=LAS_OUTPUT_PATH,
        grid_path=REFERENCE_GRID_PATH,
        flow_u=u,
        flow_v=v,
        direction=CORRECTION_DIRECTION
    )

# %%
