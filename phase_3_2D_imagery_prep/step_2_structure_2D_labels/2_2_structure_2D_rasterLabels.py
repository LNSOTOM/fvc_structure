#%%
##one efile
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
    # 'litter': 1,
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

# medium site
shapefile_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.26.shp'
output_raster_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.26.tif'
reference_raster_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif'

# low site
# shapefile_path = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.66.shp'
# output_raster_path = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.66.tif'
# reference_raster_path = '/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral_66.tif'

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
import rasterio
from rasterio.features import rasterize
from pathlib import Path
import os
import numpy as np

# Mapping remains the same
STRUCTURE_ID_MAPPING = {
    'forb': 0,
    'branch_stem_dead': 1,
    'branch_stem_standing': 1,
    'cwd': 1,
    'grass_hummock': 2,
    'grass_hummock_senescent': 2,
    'shrub_chenopod': 3,
    'shrub_chenopod_senescent': 3,
    'shrub': 3,
    'mallee': 4
}

def shapefile_to_raster(shapefile_path, output_raster_path, reference_raster_path):
    """Convert a shapefile to a raster based on structure class attributes."""
    shapes = gpd.read_file(shapefile_path)

    if 'struct_id' not in shapes.columns:
        print(f"⚠️  'struct_id' column not found in {os.path.basename(shapefile_path)}")
        return

    with rasterio.open(reference_raster_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            driver='GTiff',
            count=1,
            compress='lzw',
            nodata=255,
            dtype=rasterio.uint8,
            tiled=True,
            blockxsize=3072,
            blockysize=3072
        )

        rasterized = rasterize(
            ((geom, val) for geom, val in zip(shapes.geometry, shapes['struct_id'])),
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=255,
            all_touched=True,
            dtype=rasterio.uint8
        )

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized, 1)

    print(f"✅ Saved raster: {output_raster_path}")


def batch_rasterize_shapefiles(shp_dir, ref_dir, out_dir, ref_suffix="tiles_multispectral"):
    """
    Rasterize all shapefiles in a folder.
    Each shapefile is matched to its reference raster by matching the numeric tile index.
    """
    shp_dir, ref_dir, out_dir = map(Path, [shp_dir, ref_dir, out_dir])
    os.makedirs(out_dir, exist_ok=True)

    shapefiles = sorted(shp_dir.glob("*.shp"))
    print(f"🔍 Found {len(shapefiles)} shapefiles in {shp_dir}")

    for shp_path in shapefiles:
        tile_id = shp_path.stem.split('.')[-1]  # Extract tile index (e.g., 26 from mask_fvc_3072.26)
        ref_path = ref_dir / f"{ref_suffix}.{tile_id}.tif"
        out_path = out_dir / f"mask_structure_3072.{tile_id}.tif"

        if not ref_path.exists():
            print(f"⚠️  No matching reference raster found for tile {tile_id}")
            continue

        shapefile_to_raster(shp_path, out_path, ref_path)


# Example usage for one site (you can repeat for other sites)
batch_rasterize_shapefiles(
    # medium site
    # shp_dir="/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp",
    # ref_dir="/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked",
    # out_dir="/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster",
    # ref_suffix="tiles_multispectral"
    
    #low site
    shp_dir="/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp",
    ref_dir="/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked",
    out_dir="/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster",
    ref_suffix="tiles_multispectral"
)

# %%
