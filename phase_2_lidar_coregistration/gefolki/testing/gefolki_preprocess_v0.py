#%%
import rasterio

#medium
with rasterio.open("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05.tif") as msrc, \
     rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_05.tif") as lsrc:
    
    
#low
# with rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif") as msrc, \
#      rasterio.open("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif") as lsrc:
    
    print("📏 MicaSense DSM shape:", msrc.height, "x", msrc.width)
    print("📏 LiDAR DEM shape    :", lsrc.height, "x", lsrc.width)
    print("🧭 MicaSense bounds   :", msrc.bounds)
    print("🧭 LiDAR bounds       :", lsrc.bounds)
    print("📐 MicaSense transform:", msrc.transform)
    print("📐 LiDAR transform    :", lsrc.transform)


#%%
import rasterio
## check if the two rasters have the same shape and CRS
#medium
with rasterio.open("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as msrc, \
     rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif") as lsrc:
   
    print("📏 MicaSense DSM shape:", msrc.height, "x", msrc.width)
    print("📏 LiDAR DEM shape    :", lsrc.height, "x", lsrc.width)
    print("🧭 MicaSense bounds   :", msrc.bounds)
    print("🧭 LiDAR bounds       :", lsrc.bounds)
    print("📐 MicaSense transform:", msrc.transform)
    print("📐 LiDAR transform    :", lsrc.transform)


#%%
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def resample_lidar_to_dsm_shape_preserve_crs(src_path, ref_path, dst_path):
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        dst_profile = src.profile.copy()  # Preserve LiDAR CRS
        dst_profile.update(
            height=ref.height,
            width=ref.width,
            transform=ref.transform,
            dtype='float32'
        )

        # Empty array with shape of DSM
        resampled = np.empty((ref.height, ref.width), dtype='float32')

        # Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=src.crs,  # Keep original LiDAR CRS
            resampling=Resampling.bilinear
        )

        with rasterio.open(dst_path, 'w', **dst_profile) as dst:
            dst.write(resampled, 1)

        print(f"✅ Resampled LiDAR DEM (with original CRS) saved to: {dst_path}")




#%%
resample_lidar_to_dsm_shape_preserve_crs(
    src_path="/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif",
    ref_path="/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif",
    dst_path="/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif"
)

# sfm_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05.tif"
# lidar_dem_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_05.tif"  # normalized to [0,1]

#########################

# %%
### GOOD: better resampling
import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def create_raster_with_reference_shape(
    source_data_path: str, 
    template_grid_path: str, 
    destination_path: str, 
    resampling_method=Resampling.bilinear
):
    """
    Creates a new raster file with the exact shape and grid of a template raster.

    This function reads the pixel values from the source raster and resamples them
    to fit perfectly onto the grid defined by the template raster. The output file
    will have the exact same dimensions (height/width), CRS, and extent as the
    template file. Both input files remain unchanged.

    Args:
        source_data_path (str): Path to the raster providing the pixel values (e.g., LiDAR DSM).
        template_grid_path (str): Path to the raster providing the target grid/shape (e.g., imagery DSM).
        destination_path (str): Path where the new, aligned raster will be saved.
        resampling_method (Resampling, optional): The algorithm for resampling.
            Defaults to Resampling.bilinear.
    """
    print("--- Creating New Raster with Template Shape ---")
    print(f"Source of Data:   '{source_data_path}'")
    print(f"Template for Grid:  '{template_grid_path}'")
    print(f"New Output File:    '{destination_path}'")

    # Ensure the output directory exists.
    output_dir = os.path.dirname(destination_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. Open the template file to get its grid definition (profile).
    # This file is ONLY READ FROM, never written to.
    with rasterio.open(template_grid_path) as template_ds:
        template_profile = template_ds.profile
        
    # 2. Open the source data file.
    # This file is also ONLY READ FROM.
    with rasterio.open(source_data_path) as source_ds:
        
        # 3. Define the metadata for our NEW output file.
        # It's a copy of the template's grid, but we'll use the
        # data type (e.g., float32) and nodata value from our source data.
        output_profile = template_profile.copy()
        output_profile.update({
            'dtype': source_ds.profile.get('dtype', 'float32'),
            'nodata': source_ds.nodata
        })

        # 4. Create an empty placeholder array. This array has the exact
        # dimensions of our template raster.
        destination_array = np.empty(
            (template_profile['height'], template_profile['width']), 
            dtype=output_profile['dtype']
        )

        # 5. This is the core step. 'reproject' takes the data from the source,
        # transforms it in memory, and fills our placeholder array.
        reproject(
            source=rasterio.band(source_ds, 1),
            destination=destination_array,
            src_transform=source_ds.transform,
            src_crs=source_ds.crs,
            dst_transform=template_profile['transform'],
            dst_crs=template_profile['crs'],
            resampling=resampling_method
        )
        
        # 6. Save the new array to the destination file with the correct profile.
        with rasterio.open(destination_path, 'w', **output_profile) as dst:
            dst.write(destination_array, 1)

    print(f"\n✅ Success! New raster created at '{destination_path}' with the exact shape of the reference file.")


if __name__ == "__main__":
    # --- Define file paths ---
    lidar_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    imagery_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif"
    
    # The final output file path.
    output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif"

    # --- Run the process ---
    if not os.path.exists(lidar_dsm_path):
        print(f"Error: Source data file not found at: {lidar_dsm_path}")
    elif not os.path.exists(imagery_dsm_path):
        print(f"Error: Template grid file not found at: {imagery_dsm_path}")
    else:
        create_raster_with_reference_shape(
            source_data_path=lidar_dsm_path,
            template_grid_path=imagery_dsm_path,
            destination_path=output_path
        )

# %%
### it does not give same shape as reference
import os
import rasterio
import rasterio.mask
from shapely.geometry import box

def clip_raster_to_reference_extent(source_path: str, reference_path: str, destination_path: str):
    """
    Clips a source raster to the exact bounding box (extent) of a reference raster.

    This is the ideal preprocessing step for coregistration workflows (e.g., with Gefolki),
    as it preserves the source raster's original resolution and pixel alignment
    within the clipped area.

    Args:
        source_path (str): Path to the raster to be clipped (e.g., LiDAR DSM).
        reference_path (str): Path to the raster providing the clipping extent.
        destination_path (str): Path where the clipped output raster will be saved.
    """
    print("--- Starting Raster Clipping by Extent (for Coregistration Prep) ---")
    print(f"Source (to be clipped): {source_path}")
    print(f"Reference (for extent): {reference_path}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(destination_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. Get the extent (bounding box) from the reference raster.
    with rasterio.open(reference_path) as ref:
        cropping_extent = ref.bounds

    # 2. Create a geometry object (a polygon) from the extent.
    cropping_geometry = box(*cropping_extent)

    # 3. Open the source raster and clip it using the geometry.
    with rasterio.open(source_path) as src:
        try:
            # The 'mask' function clips the data. 'crop=True' is essential.
            out_image, out_transform = rasterio.mask.mask(
                dataset=src,
                shapes=[cropping_geometry],
                crop=True
            )
        except ValueError as e:
            print(f"\n--- ERROR ---")
            print(f"Could not clip the raster. This often means the rasters do not overlap.")
            print(f"Original error: {e}")
            return

        # Copy the metadata from the original source raster.
        out_meta = src.profile.copy()

    # 4. Update the metadata to reflect the new (smaller) dimensions and transform.
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    })

    # 5. Write the clipped data to the new file.
    with rasterio.open(destination_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"\n✅ Clipping complete. Output saved to: {destination_path}")

if __name__ == "__main__":
    # --- Define file paths ---
    lidar_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    imagery_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif"
    
    # A more descriptive output filename
    output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03_clipped_to_rgb.tif"

    # --- Run the clipping ---
    if not os.path.exists(lidar_dsm_path):
        print(f"Error: LiDAR DSM input not found at: {lidar_dsm_path}")
    elif not os.path.exists(imagery_dsm_path):
        print(f"Error: Imagery DSM reference not found at: {imagery_dsm_path}")
    else:
        clip_raster_to_reference_extent(
            source_path=lidar_dsm_path,
            reference_path=imagery_dsm_path,
            destination_path=output_path
        )

# %%
