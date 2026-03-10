
#%%
import rasterio
import numpy as np
import pylab as pl
from pathlib import Path

def analyze_coregistration_difference(base_dsm_path: Path, coregistered_dsm_path: Path, output_diff_path: Path):
    """
    Calculates and analyzes the difference between a base DSM and a coregistered DSM.

    Args:
        base_dsm_path (Path): The reference raster (e.g., imagery DSM).
        coregistered_dsm_path (Path): The raster that was aligned (e.g., LiDAR DSM).
        output_diff_path (Path): Path to save the difference map GeoTIFF.
    """
    print("--- Verifying Coregistration Quality by Analyzing Difference ---")
    print(f"Base (Reference): {base_dsm_path}")
    print(f"Coregistered:     {coregistered_dsm_path}")

    with rasterio.open(base_dsm_path) as base_src, rasterio.open(coregistered_dsm_path) as coreg_src:
        # --- Sanity Checks ---
        if base_src.shape != coreg_src.shape:
            print(f"ERROR: Raster shapes do not match!")
            print(f"Base shape: {base_src.shape}, Coregistered shape: {coreg_src.shape}")
            return
        
        if base_src.crs != coreg_src.crs:
            print(f"WARNING: Coordinate Reference Systems (CRS) do not match.")

        # --- Read Data ---
        base_dsm = base_src.read(1).astype(np.float32)
        coreg_dsm = coreg_src.read(1).astype(np.float32)
        profile = base_src.profile

        # --- Create a shared mask for valid comparison ---
        # Ignore areas where either DSM has a nodata value.
        mask_base = (base_dsm != base_src.nodata) & np.isfinite(base_dsm)
        mask_coreg = (coreg_dsm != coreg_src.nodata) & np.isfinite(coreg_dsm)
        valid_mask = mask_base & mask_coreg
        
        if np.sum(valid_mask) == 0:
            print("ERROR: No overlapping valid data found between the two rasters.")
            return

        # --- Calculate Difference ---
        # Initialize difference array with a nodata value
        difference = np.full(base_dsm.shape, profile.get('nodata', -9999), dtype=np.float32)
        
        # Calculate difference only in valid areas
        difference[valid_mask] = base_dsm[valid_mask] - coreg_dsm[valid_mask]
        
        valid_diff = difference[valid_mask]

        # --- Calculate Statistics ---
        mean_diff = np.mean(valid_diff)
        median_diff = np.median(valid_diff)
        std_dev = np.std(valid_diff)
        # NMAD is a robust measure of spread, like standard deviation but less sensitive to outliers
        nmad = 1.4826 * np.median(np.abs(valid_diff - median_diff))
        
        print("\n--- Vertical Difference Statistics (Base - Coregistered) ---")
        print(f"Mean Difference:      {mean_diff:+.4f} meters")
        print(f"Median Difference:    {median_diff:+.4f} meters (robust measure of vertical bias)")
        print(f"Standard Deviation:   {std_dev:.4f} meters")
        print(f"NMAD:                 {nmad:.4f} meters (robust measure of spread/precision)")

        # --- Save Difference Map ---
        output_diff_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(dtype='float32', nodata=-9999)
        with rasterio.open(output_diff_path, 'w', **profile) as dst:
            dst.write(difference, 1)
        print(f"\n✅ Difference map saved to: {output_diff_path}")

        # --- Plot Difference Map ---
        pl.figure(figsize=(10, 10))
        # Use robust percentile clipping for better color stretch
        v_min, v_max = np.percentile(valid_diff, [2, 98])
        im = pl.imshow(difference, cmap='coolwarm_r', vmin=v_min, vmax=v_max)
        pl.title("DSM of Difference (Base - Coregistered)")
        cbar = pl.colorbar(im, extend='both')
        cbar.set_label("Elevation Difference (meters)")
        pl.show()


if __name__ == "__main__":
    # The file you just created
    COREGISTERED_LIDAR_DSM = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03_coreg.tif")
    
    # The original, stable reference DSM
    BASE_IMAGERY_DSM = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif")

    # Where to save the output analysis file
    OUTPUT_DIFFERENCE_MAP = COREGISTERED_LIDAR_DSM.with_name(COREGISTERED_LIDAR_DSM.stem + "_vs_base_DoD.tif")

    analyze_coregistration_difference(
        base_dsm_path=BASE_IMAGERY_DSM,
        coregistered_dsm_path=COREGISTERED_LIDAR_DSM,
        output_diff_path=OUTPUT_DIFFERENCE_MAP
    )

# %%
