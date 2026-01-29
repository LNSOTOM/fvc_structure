#%%
import rasterio

#medium
##L1
# with rasterio.open("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as msrc, \
#      rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif") as lsrc:

# with rasterio.open("/data/calperumResearch/site1_1_DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as msrc, \
#      rasterio.open("/data/calperumResearch/site1_1_DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif") as lsrc:
    
##L2
# with rasterio.open("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as msrc, \
#      rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif") as lsrc:

# with rasterio.open("/data/calperumResearch/site1_1_DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as msrc, \
#      rasterio.open("/data/calperumResearch/site1_1_DD0001/lidar_l2/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif") as lsrc:
   
    
#low
# with rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif") as msrc, \
#      rasterio.open("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif") as lsrc:

#L1
# with rasterio.open("/data/calperumResearch/site2_1_DD0011/imagery/dsm/SASMDD0011_rgb_dsm_03.tif") as msrc, \
#      rasterio.open("/data/calperumResearch/site2_1_DD0011/lidar/dsm/crop_to_rgb/SASMDD0011_l1_dsm_03.tif") as lsrc:

#L2  
with rasterio.open("/data/calperumResearch/site2_1_DD0011/imagery/dsm/SASMDD0011_rgb_dsm_03.tif") as msrc, \
     rasterio.open("/data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/crop_to_rgb/SASMDD0011_l2_dsm_03.tif") as lsrc:
    
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
    #  rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif") as lsrc:
    rasterio.open("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif") as lsrc:

    print("📏 MicaSense DSM shape:", msrc.height, "x", msrc.width)
    print("📏 LiDAR DEM shape    :", lsrc.height, "x", lsrc.width)
    print("🧭 MicaSense bounds   :", msrc.bounds)
    print("🧭 LiDAR bounds       :", lsrc.bounds)
    print("📐 MicaSense transform:", msrc.transform)
    print("📐 LiDAR transform    :", lsrc.transform)


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
    ##site 1
    #l1
    # lidar_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    # lidar_dsm_path = "/data/calperumResearch/site1_1_DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    #l2
    # lidar_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/SASMDD0001_l2_dsm_03.tif"
    # lidar_dsm_path = "/data/calperumResearch/site1_1_DD0001/lidar_l2/dsm/SASMDD0001_l2_dsm_03.tif"
    
    # imagery_dsm_path = "/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif"
    # imagery_dsm_path = "/data/calperumResearch/site1_1_DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif"
    
    ##site 11
    #l1
    # lidar_dsm_path = "/data/calperumResearch/site2_1_DD0011/lidar/dsm/SASMDD0011_l1_dsm_03.tif"
    
    #L2
    lidar_dsm_path = "/data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/SASMDD0011_l2_dsm_03.tif"
    
    imagery_dsm_path = "/data/calperumResearch/site2_1_DD0011/imagery/dsm/SASMDD0011_rgb_dsm_03.tif"
      
    # The final output file path.
    ##site 1
    # output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif"
    # output_path = "/data/calperumResearch/site1_1_DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif"
    
    # output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/crop_to_l1/SASMDD0001_l2_dsm_03.tif"
    # output_path = "/data/calperumResearch/site1_1_DD0001/lidar_l2/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif"
    
    ##site 11
    # output_path = "/data/calperumResearch/site2_1_DD0011/lidar/dsm/crop_to_rgb/SASMDD0011_l1_dsm_03.tif"
    output_path = "/data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/crop_to_rgb/SASMDD0011_l2_dsm_03.tif"


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
