#%%
"""
Processes a LAS file and a 2D mask to generate 3D NIfTI volumes 
for training with nnU-Net.

This script generates two files:
1.  A 3D Label Volume (e.g., "tile_109.nii.gz"):
    - A 3D array where each voxel contains its class ID (0, 65, 66, 67, 68).
2.  A 4D Feature Volume (e.g., "tile_109_0000.nii.gz"):
    - A 4D array (Channels, Z, Y, X) containing the input features.
    - Channel 0-4: 5 bands from the multispectral image (extruded)
    - Channel 5: Max Z_norm height (extruded from the 2D raster)
    - Channel 6: Voxel PointCount / Density (a true 3D feature)
"""
import laspy
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from tqdm import tqdm # Although not used, kept as it was in your script
from pathlib import Path
import nibabel as nib # For saving NIfTI files

def create_3d_training_data(
    las_path: str,
    mask_path: str,
    multispectral_path: str,
    output_dir: str,
    tile_id: str,
    voxel_size: float = 0.05,
    valid_labels: tuple = (65, 66, 67, 68)
):
    """
    Computes 3D voxel features and labels and saves them as nnU-Net
    compatible NIfTI files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Read Mask and Multispectral Rasters ---
    print(f"🖼️ Reading mask: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask_2d = src.read(1)
        transform = src.transform
        profile = src.profile.copy()
        height, width = mask_2d.shape
        # Get the real-world coordinates for the raster bounds
        xmin, ymin, xmax, ymax = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

    print(f"🌈 Reading multispectral: {multispectral_path}")
    with rasterio.open(multispectral_path) as src:
        # Assuming multispectral image has 5 bands and matches the mask
        if src.count != 5:
            print(f"Warning: Multispectral TIF has {src.count} bands, expected 5.")
        
        # Read all bands
        multispectral_2d = src.read() # Shape (5, H, W)

    # --- 2. Read and Filter LAS Data ---
    print(f"📥 Reading LAS: {las_path}")
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    # --- Extract Z_norm correctly ---
    if "Z_norm" in las.point_format.dimension_names:
        z = np.asarray(las["Z_norm"], dtype=np.float64)
    elif "attribute0" in las.point_format.dimension_names:
        z = np.asarray(las["attribute0"], dtype=np.float64)
    else:
        raise AttributeError(f"Missing 'Z_norm' or 'attribute0'. Found: {las.point_format.dimension_names}")

    # Map X,Y to raster row,col
    row, col = rowcol(transform, x, y)
    row, col = row.astype(int), col.astype(int)

    # Filter points to be within the raster bounds
    valid_mask = (
        (row >= 0) & (row < height) &
        (col >= 0) & (col < width)
    )
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    row, col = row[valid_mask], col[valid_mask]

    # Get class labels for each point from the 2D mask
    class_vals = mask_2d[row, col]

    # Separate labeled points from unlabeled points
    label_mask = np.isin(class_vals, valid_labels)
    
    labeled_x, labeled_y, labeled_z = x[label_mask], y[label_mask], z[label_mask]
    labeled_class = class_vals[label_mask]
    
    # We also keep the *unlabeled* points (where mask is 0)
    # This is important for creating the full 3D feature set
    unlabeled_x, unlabeled_y, unlabeled_z = x[~label_mask], y[~label_mask], z[~label_mask]

    print(f"✅ {len(labeled_z):,} labeled points matched to {valid_labels}")
    print(f"✅ {len(unlabeled_z):,} unlabeled points (background)")
    
    if len(z) == 0:
        print(f"🚨 ERROR: No points found within the raster bounds. Check file alignment and CRS.")
        return

    # --- 3. Define 3D Voxel Grid ---
    
    # We define the grid based on the rasters, not the points,
    # to ensure perfect alignment.
    # Note: Voxel Z dimension is based on point Z values
    z_min, z_max = z.min(), z.max()

    # Calculate voxel indices for ALL points
    # We must normalize all indices to start from 0
    ix = ((x - xmin) / voxel_size).astype(int)
    iy = ((ymax - y) / voxel_size).astype(int) # Y is inverted in rasters
    iz = ((z - z_min) / voxel_size).astype(int)
    
    # Get grid dimensions from voxel indices
    # This ensures the grid is as large as the raster
    grid_shape_x = width 
    grid_shape_y = height
    # Add a buffer for Z dimension
    grid_shape_z = int((z_max - z_min) / voxel_size) + 2 

    # nnU-Net expects (X, Y, Z) or (Z, Y, X)
    # Let's use (Z, Y, X) as it's common in medical imaging
    grid_shape = (grid_shape_z, grid_shape_y, grid_shape_x)
    print(f"🧊 3D Grid Shape (Z, Y, X): {grid_shape}")

    # --- 4. Create 3D Label Volume ---
    print("Creating 3D Label Volume...")
    label_volume = np.zeros(grid_shape, dtype=np.uint8)

    if len(labeled_z) > 0:
        # Get voxel indices for *labeled* points only
        labeled_ix = ((labeled_x - xmin) / voxel_size).astype(int)
        labeled_iy = ((ymax - labeled_y) / voxel_size).astype(int)
        labeled_iz = ((labeled_z - z_min) / voxel_size).astype(int)

        # Clip indices to be within our grid
        labeled_ix = np.clip(labeled_ix, 0, grid_shape[2] - 1)
        labeled_iy = np.clip(labeled_iy, 0, grid_shape[1] - 1)
        labeled_iz = np.clip(labeled_iz, 0, grid_shape[0] - 1)

        # "Burn" labels into the 3D volume
        # This assigns the 2D mask label to the specific 3D voxel
        label_volume[labeled_iz, labeled_iy, labeled_ix] = labeled_class.astype(np.uint8)
    else:
        print("⚠️ Warning: No labeled points found. Label volume will be empty.")

    # --- 5. Create 4D Feature Volume ---
    print("Creating 4D Feature Volume...")
    
    # 5 bands (multispectral) + 1 (Z_max) + 1 (PointCount) = 7 channels
    num_features = 7
    feature_volume = np.zeros((num_features, *grid_shape), dtype=np.float32)

    # --- Channels 0-4: Extruded Multispectral ---
    print("  ... Channels 0-4: Extruding Multispectral bands")
    for i in range(5):
        # "Extrude" the 2D band vertically through the Z dimension
        feature_volume[i, :, :, :] = multispectral_2d[i, np.newaxis, :, :]

    # --- Channel 5: Extruded Max Z Height ---
    print("  ... Channel 5: Calculating and Extruding Max Z")
    z_raster_max = np.full((height, width), -9999.0, dtype=np.float32)
    # We use ALL points for this, not just labeled ones
    all_rows, all_cols = rowcol(transform, x, y)
    all_rows = np.clip(all_rows.astype(int), 0, height - 1)
    all_cols = np.clip(all_cols.astype(int), 0, width - 1)

    for r, c, zv in zip(all_rows, all_cols, z):
        if zv > z_raster_max[r, c]:
            z_raster_max[r, c] = zv
            
    z_raster_max[z_raster_max == -9999.0] = 0 # Use 0 for no-data
    
    # Extrude this 2D feature
    feature_volume[5, :, :, :] = z_raster_max[np.newaxis, :, :]
    
    # --- Channel 6: Voxel Point Count (Density) ---
    print("  ... Channel 6: Calculating Voxel Point Count (Density)")
    point_count_volume = np.zeros(grid_shape, dtype=np.uint32)
    
    # Clip all point indices
    all_ix = np.clip(ix, 0, grid_shape[2] - 1)
    all_iy = np.clip(iy, 0, grid_shape[1] - 1)
    all_iz = np.clip(iz, 0, grid_shape[0] - 1)
    
    # Use np.add.at for efficient counting
    np.add.at(point_count_volume, (all_iz, all_iy, all_ix), 1)
    
    # Add to feature volume (normalize if desired, but raw counts are fine)
    feature_volume[6, :, :, :] = point_count_volume.astype(np.float32)

    # --- 6. Save NIfTI Files ---
    # nnU-Net requires a specific naming convention:
    # features: {tile_id}_0000.nii.gz
    # labels:   {tile_id}.nii.gz
    
    # Create a simple affine matrix (identity matrix)
    # Voxel size is in meters.
    z_spacing = voxel_size
    y_spacing = voxel_size 
    x_spacing = voxel_size

    affine = np.array([
        [-x_spacing, 0, 0, xmin],
        [0, -y_spacing, 0, ymax], # Y is negative
        [0, 0, z_spacing, z_min],
        [0, 0, 0, 1]
    ])

    # --- Save Label File ---
    label_path = output_dir / f"{tile_id}.nii.gz"
    label_img = nib.Nifti1Image(label_volume, affine)
    # Set spacing (pixdim)
    label_img.header['pixdim'][1:4] = [x_spacing, y_spacing, z_spacing]
    nib.save(label_img, str(label_path))
    print(f"✅ Saved Label Volume: {label_path}")

    # --- Save Feature File ---
    feature_path = output_dir / f"{tile_id}_0000.nii.gz"
    feature_img = nib.Nifti1Image(feature_volume, affine)
    # Set spacing (pixdim)
    feature_img.header['pixdim'][1:4] = [x_spacing, y_spacing, z_spacing]
    nib.save(feature_img, str(feature_path))
    print(f"✅ Saved Feature Volume: {feature_path} (Shape: {feature_volume.shape})")
    print("--- Done ---")


# === Example Usage ===
if __name__ == "__main__":
    
    # --- Set paths for your data ---
    las_file = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
    mask_file = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.109.tif"
    multi_file = "/media/laura/Laura 102/uas_data/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral/stacked/tiles_multispectral.109.tif" 
    tile_id = "109"
    
    output_nnunet_dir = "./nnUNet_input_data"
    
    print(f"\n--- Running script for Tile ID: {tile_id} ---")
    print(f"LAS: {las_file}")
    print(f"Mask: {mask_file}")
    print(f"Multi: {multi_file}")
    print(f"Output: {output_nnunet_dir}")
    
    try:
        create_3d_training_data(
            las_path=las_file,
            mask_path=mask_file,
            multispectral_path=multi_file,
            output_dir=output_nnunet_dir,
            tile_id=tile_id,
            voxel_size=0.05,
            valid_labels=(65, 66, 67, 68),
        )
    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# %%
