#%%
##one efile
#%%
# ###### Rasterised as mask based on structure class features - shp to raster for mask
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import os
import numpy as np

# Structure classification mapping from 2_1_structure_2D_labels.py
STRUCTURE_ID_MAPPING = {
    # Forbs -> ID 0 (closest to ground level)
    'forb': 0,
    
    # Woody debris/stem -> ID 1  
    'branch_stem_dead': 1,
    'branch_stem_standing': 1,
    'litter': 1,
    'cwd': 1,  # Coarse Woody Debris
    
    # Grass types -> ID 2
    'grass_hummock': 2,
    'grass_hummock_senescent': 2,
    
    # ALL shrub types -> ID 3
    'shrub_chenopod': 3,
    'shrub_chenopod_senescent': 3,
    'shrub': 3,
    
    # Mallee -> ID 4
    'mallee': 4
}

def shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path):
    """
    Convert a shapefile to a raster based on structure class attributes,
    using a reference raster for spatial extent and resolution.
    """
    # Read the shapefile
    shapes = gpd.read_file(shapefile_path)
    
    # Check if struct_id column exists (created by 2_1_structure_2D_labels.py)
    if 'struct_id' not in shapes.columns:
        print(f"Error: 'struct_id' column not found in {shapefile_path}")
        print(f"Available columns: {list(shapes.columns)}")
        print("Please run 2_1_structure_2D_labels.py first to create structure-classified shapefiles.")
        return
    
    # Report structure ID distribution
    print(f"Structure ID distribution in {os.path.basename(shapefile_path)}:")
    structure_counts = shapes['struct_id'].value_counts().sort_index()
    for struct_id, count in structure_counts.items():
        structure_name = {0: 'Forb', 1: 'Woody debris/stem', 2: 'Grass', 3: 'Shrub', 4: 'Mallee'}.get(struct_id, 'Unknown')
        print(f"  ID {struct_id} ({structure_name}): {count} polygons")
    
    with rasterio.open(reference_raster_path) as ref:
        # Define the output profile based on the reference raster
        profile = ref.profile.copy()
        profile.update(
            driver='GTiff',
            count=1,
            compress='lzw',
            nodata=255,  # Use 255 as nodata (outside valid range 0-4)
            dtype=rasterio.uint8,  # Use uint8 since values are 0-4
            tiled=True,
            blockxsize=3072,
            blockysize=3072
        )

        # Rasterize using struct_id values directly
        rasterized = rasterize(
            ((geometry, value) for geometry, value in zip(shapes.geometry, shapes['struct_id'])),
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=255,  # Fill with nodata value for areas not covered by any polygon
            all_touched=True,
            dtype=rasterio.uint8
        )

        # Write the rasterized output
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized, 1)
    
    print(f"✅ Structure raster saved to: {output_raster_path}")
    print(f"📊 Raster values: 0=Forb, 1=Woody debris, 2=Grass, 3=Shrub, 4=Mallee, 255=NoData")

# low site
# shapefile_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotations_shp/mask_fvc_3072.16.shp'
# output_raster_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.16.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.16.tif'

# medium site
shapefile_path = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.66.shp'
output_raster_path = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.66.tif'
reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.66.tif'

# dense site
# shapefile_path = '/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_structure_3072.128.shp'
# output_raster_path = '/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.128.tif'
# reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/tiles_multispectral/tiles_multispectral.128.tif'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)

# Convert shapefile to raster
shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path)


#%%
## for multiple files
# ###### Rasterised as mask based on structure class features - shp to raster for mask
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import os
import numpy as np

# Structure classification mapping from 2_1_structure_2D_labels.py
STRUCTURE_ID_MAPPING = {
    # Forbs -> ID 0 (closest to ground level)
    'forb': 0,
    
    # Woody debris/stem -> ID 1  
    'branch_stem_dead': 1,
    'branch_stem_standing': 1,
    'litter': 1,
    'cwd': 1,  # Coarse Woody Debris
    
    # Grass types -> ID 2
    'grass_hummock': 2,
    'grass_hummock_senescent': 2,
    
    # ALL shrub types -> ID 3
    'shrub_chenopod': 3,
    'shrub_chenopod_senescent': 3,
    'shrub': 3,
    
    # Mallee -> ID 4
    'mallee': 4
}

def shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path):
    """
    Convert a shapefile to a raster based on structure class attributes,
    using a reference raster for spatial extent and resolution.
    """
    # Read the shapefile
    shapes = gpd.read_file(shapefile_path)
    
    # Check if struct_id column exists (created by 2_1_structure_2D_labels.py)
    if 'struct_id' not in shapes.columns:
        print(f"Error: 'struct_id' column not found in {shapefile_path}")
        print(f"Available columns: {list(shapes.columns)}")
        print("Please run 2_1_structure_2D_labels.py first to create structure-classified shapefiles.")
        return False
    
    # Report structure ID distribution
    print(f"Structure ID distribution in {os.path.basename(shapefile_path)}:")
    structure_counts = shapes['struct_id'].value_counts().sort_index()
    for struct_id, count in structure_counts.items():
        structure_name = {0: 'Forb', 1: 'Woody debris/stem', 2: 'Grass', 3: 'Shrub', 4: 'Mallee'}.get(struct_id, 'Unknown')
        print(f"  ID {struct_id} ({structure_name}): {count} polygons")
    
    with rasterio.open(reference_raster_path) as ref:
        # Define the output profile based on the reference raster
        profile = ref.profile.copy()
        profile.update(
            driver='GTiff',
            count=1,
            compress='lzw',
            nodata=255,  # Use 255 as nodata (outside valid range 0-4)
            dtype=rasterio.uint8,  # Use uint8 since values are 0-4
            tiled=True,
            blockxsize=3072,
            blockysize=3072
        )

        # Rasterize using struct_id values directly
        rasterized = rasterize(
            ((geometry, value) for geometry, value in zip(shapes.geometry, shapes['struct_id'])),
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=255,  # Fill with nodata value for areas not covered by any polygon
            all_touched=True,
            dtype=rasterio.uint8
        )

        # Write the rasterized output
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized, 1)
    
    print(f"✅ Structure raster saved to: {output_raster_path}")
    return True

def process_all_shapefiles(shapefile_dir, output_dir, reference_dir):
    """
    Process all shapefiles in a directory and convert them to rasters
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all shapefiles in the directory
    shapefiles = [f for f in os.listdir(shapefile_dir) if f.endswith('.shp')]
    
    if not shapefiles:
        print(f"No shapefiles found in {shapefile_dir}")
        return
    
    print(f"Found {len(shapefiles)} shapefiles to process")
    print("=" * 60)
    
    processed_count = 0
    failed_count = 0
    
    for shapefile_name in sorted(shapefiles):
        print(f"\n🔄 Processing: {shapefile_name}")
        
        # Full paths
        shapefile_path = os.path.join(shapefile_dir, shapefile_name)
        
        # Create output filename (change mask_fvc to mask_structure, .shp to .tif)
        output_name = shapefile_name.replace('mask_fvc_', 'mask_structure_').replace('.shp', '.tif')
        output_raster_path = os.path.join(output_dir, output_name)
        
        # Find corresponding reference raster
        # Extract tile number from shapefile name (e.g., "66" from "mask_fvc_3072.66.shp")
        try:
            tile_number = shapefile_name.split('.')[-2]  # Get the part before .shp
            reference_name = f"tiles_multispectral.{tile_number}.tif"
            reference_raster_path = os.path.join(reference_dir, reference_name)
        except:
            print(f"❌ Could not extract tile number from {shapefile_name}")
            failed_count += 1
            continue
        
        # Check if reference raster exists
        if not os.path.exists(reference_raster_path):
            print(f"❌ Reference raster not found: {reference_raster_path}")
            failed_count += 1
            continue
        
        # Process the shapefile
        success = shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path)
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {processed_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📂 Output directory: {output_dir}")
    print(f"📊 Raster values: 0=Forb, 1=Woody debris, 2=Grass, 3=Shrub, 4=Mallee, 255=NoData")

# Configuration for different sites
# low site
shapefile_dir = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'
output_dir = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster'
reference_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/raw/tiles_multispectral'

# medium site
# shapefile_dir = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotations_shp'
# output_dir = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster'
# reference_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/raw/tiles_multispectral'

# dense site
# shapefile_dir = '/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'
# output_dir = '/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster'
# reference_dir = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/predictors/tiles_3072/raw/tiles_multispectral'

# Process all shapefiles in the directory
process_all_shapefiles(shapefile_dir, output_dir, reference_dir)
# %%
