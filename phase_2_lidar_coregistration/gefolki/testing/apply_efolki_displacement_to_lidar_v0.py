import numpy as np
import laspy
import rasterio
from scipy.ndimage import map_coordinates
from pathlib import Path
import argparse
import time

# === Input paths ===
las_input_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
aligned_dem_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_05_coreg_to_rgb.tif")  # normalized to [0,1]


flow_u = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u.npy")
flow_v = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v.npy")

# === Load geotransform from aligned DEM ===
with rasterio.open(aligned_dem_path) as src:
    transform = src.transform
    width, height = src.width, src.height
    pixel_size_x = transform[0]
    pixel_size_y = -transform[4]  # Y pixel size is usually negative
    origin_x, origin_y = transform[2], transform[5]

# === Load LiDAR point cloud ===
las = laspy.read(str(las_input_path))
x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)

# === Convert world (x, y) to raster (row, col) coordinates ===
col = ((x - origin_x) / pixel_size_x).astype(np.float32)
row = ((origin_y - y) / pixel_size_y).astype(np.float32)  # flip Y

# === Filter points inside raster bounds ===
valid = (
    (col >= 0) & (col < width - 1) &
    (row >= 0) & (row < height - 1)
)

# === Interpolate optical flow at valid positions ===
coords = np.vstack((row[valid], col[valid]))
dx_pix = map_coordinates(flow_u, coords, order=1, mode='nearest')
dy_pix = map_coordinates(flow_v, coords, order=1, mode='nearest')

# === Convert pixel flow to meters ===
dx_m = dx_pix * pixel_size_x
dy_m = -dy_pix * pixel_size_y  # Flip Y axis

# === Apply displacement ===
x_new = x.copy()
y_new = y.copy()
x_new[valid] += dx_m
y_new[valid] += dy_m

# === Save displaced LiDAR to new LAS file ===
las.x = x_new
las.y = y_new
output_path = las_input_path.with_name(las_input_path.stem + "_displaced_by_efolki.las")
las.write(str(output_path))

print(f"✅ Displaced LiDAR saved to: {output_path}")

#%%
##### test 2
import numpy as np
import laspy
import rasterio
from scipy.ndimage import map_coordinates
from pathlib import Path
import argparse
import time

def warp_las_with_flow(las_input_path, reference_dem_path, flow_u_path, flow_v_path, las_output_path):
    """
    Applies a 2D GeFolki optical flow field to a 3D LAS/LAZ point cloud.

    Args:
        las_input_path (str): Path to the input .las or .laz file.
        reference_dem_path (str): Path to the georeferenced raster (e.g., the aligned DEM)
                                  that matches the flow fields' coordinate system.
        flow_u_path (str): Path to the flow_u.npy file (X displacement in pixels).
        flow_v_path (str): Path to the flow_v.npy file (Y displacement in pixels).
        las_output_path (str): Path for the output warped .las or .laz file.
    """
    print("--- Starting Point Cloud Warp ---")
    start_time = time.time()

    # === 1. Load flow fields and georeferencing info from the reference DEM ===
    print(f"Loading flow fields from {flow_u_path} and {flow_v_path}")
    flow_u = np.load(flow_u_path)
    flow_v = np.load(flow_v_path)

    print(f"Loading georeference information from {reference_dem_path}")
    with rasterio.open(reference_dem_path) as src:
        transform = src.transform
        width, height = src.width, src.height
        pixel_size_x = transform.a
        pixel_size_y = transform.e  # This is typically negative
        origin_x = transform.c
        origin_y = transform.f

    # === 2. Load the original LiDAR point cloud ===
    print(f"Loading LiDAR data from {las_input_path}...")
    las = laspy.read(las_input_path)
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
    print(f"Loaded {len(las.points)} points.")

    # === 3. Convert LiDAR world coordinates (meters) to raster pixel coordinates ===
    # The affine transform provides the formula to map pixel to world. We use its inverse.
    # col = (x - origin_x) / pixel_size_x
    # row = (y - origin_y) / pixel_size_y
    print("Projecting 3D points into 2D raster space...")
    col = (x - origin_x) / pixel_size_x
    row = (y - origin_y) / pixel_size_y
    
    # === 4. Filter for points that fall within the raster's bounds ===
    # We can only find a displacement vector for points inside the DEM area.
    valid_mask = (col >= 0) & (col < width - 1) & (row >= 0) & (row < height - 1)
    
    # Coordinates of valid points in the raster's pixel space
    valid_coords = np.vstack((row[valid_mask], col[valid_mask]))
    print(f"Found {len(valid_coords[0])} points within the DEM bounds to warp.")

    # === 5. Interpolate the flow fields to get displacement at each point's location ===
    # map_coordinates bilinearly interpolates the flow grid (u, v) at each point's (row, col).
    print("Interpolating displacement vectors for each point...")
    dx_pix = map_coordinates(flow_u, valid_coords, order=1, mode='nearest')
    dy_pix = map_coordinates(flow_v, valid_coords, order=1, mode='nearest')

    # === 6. Convert the displacement from pixels back to world units (meters) ===
    dx_meters = dx_pix * pixel_size_x
    dy_meters = dy_pix * pixel_size_y
    
    # === 7. Apply the displacement to the valid points ===
    print("Applying displacement to point coordinates...")
    x_new, y_new = x.copy(), y.copy()
    x_new[valid_mask] += dx_meters
    y_new[valid_mask] += dy_meters

    # === 8. Create a new LAS file with the warped coordinates ===
    print(f"Saving warped point cloud to {las_output_path}...")
    new_las = laspy.LasData(las.header)
    new_las.points = las.points.copy() # Copy all point data (classification, intensity, etc.)
    new_las.x = x_new
    new_las.y = y_new
    # Z-coordinates are not changed in this 2D warp
    
    new_las.write(las_output_path)
    
    end_time = time.time()
    print("\n--- Warp Complete ---")
    print(f"✅ Displaced LiDAR saved to: {las_output_path}")
    print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a 2D GeFolki warp field (optical flow) to a 3D LAS point cloud.")
    parser.add_argument("--las-input", required=True, help="Path to the source .las or .laz file to be warped.")
    parser.add_argument("--dem-georef", required=True, help="Path to the georeferenced raster that corresponds to the flow fields (e.g., the aligned DEM).")
    parser.add_argument("--flow-u", required=True, help="Path to the 'flow_u.npy' file (X displacement).")
    parser.add_argument("--flow-v", required=True, help="Path to the 'flow_v.npy' file (Y displacement).")
    parser.add_argument("--las-output", required=True, help="Path for the new, warped .las or .laz output file.")

    args = parser.parse_args()

    warp_las_with_flow(
        las_input_path=args.las_input,
        reference_dem_path=args.dem_georef,
        flow_u_path=args.flow_u,
        flow_v_path=args.flow_v,
        las_output_path=args.las_output
    )