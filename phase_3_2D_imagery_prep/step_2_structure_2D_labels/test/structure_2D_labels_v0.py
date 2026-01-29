#%%
import geopandas as gpd
import os
from datetime import datetime

# Structure classification mapping
STRUCTURE_ID_MAPPING = {
    # Grass types -> ID 2
    'grass_hummock': 2,
    'grass_hummock_senescent': 2,
    
    # Forbs -> ID 2
    'forb': 2,
    
    # Shrubs types -> ID 2
    'shrub_chenopod': 2,
    'shrub_chenopod_senescent': 2,
    
    # Woody debris/stem -> ID 1  
    'branch_stem_dead': 1,
    'branch_stem_standing': 1,
    'litter': 1,
    'cwd': 1,  # Coarse Woody Debris
    
    # Shrub types -> ID 3
    'shrub': 3,
    
    # Mallee -> ID 4
    'mallee': 4
}

# Classes to exclude (delete)
EXCLUDE_CLASSES = ['bare_ground', 'shadow']

# low site
directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
output_directory_path = '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'

# medium site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
# output_directory_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotations_shp'

# dense site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
# output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'

# Ensure the output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Log file location - save in the script directory
log_directory = '/media/laura/laura_usb/code/fvc_structure/phase_2_labeling_workflow/step_1_structure_2D_labels'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_directory, f"structure_classification_log_{timestamp}.txt")

def log_info(message, print_console=True):
    """Log information to both console and file"""
    if print_console:
        print(message)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')

def classify_vegetation_structure(gdf, filename):
    """
    Classify vegetation structure and assign structure IDs based on vegetation type.
    
    Args:
        gdf: GeoDataFrame with vegetation polygons
        filename: Name of the file being processed
        
    Returns:
        GeoDataFrame with structure classifications and filtered geometries
    """
    # Use the 'structure' column as the primary source for vegetation type
    type_columns = ['structure', 'class', 'type', 'label', 'vegetation', 'class_name', 'classname', 'CLASS', 'TYPE']
    type_column = None
    
    for col in type_columns:
        if col in gdf.columns:
            type_column = col
            break
    
    if type_column is None:
        log_info(f"Warning: No vegetation type column found in {gdf.columns.tolist()}")
        log_info(f"Available columns: {list(gdf.columns)}")
        # If no type column found, assign default ID of 1 (woody debris/stem)
        gdf['struct_id'] = 1
        return gdf
    
    log_info(f"Using vegetation type column: '{type_column}'")
    
    # Handle null/None values first
    null_mask = gdf[type_column].isnull()
    if null_mask.sum() > 0:
        log_info(f"Found {null_mask.sum()} rows with null/missing values in '{type_column}' column")
        # Fill null values with 'unknown' to handle them consistently
        gdf[type_column] = gdf[type_column].fillna('unknown')
    
    # Convert to string to ensure consistent data type
    gdf[type_column] = gdf[type_column].astype(str)
    
    unique_types = sorted(gdf[type_column].unique())
    log_info(f"Unique vegetation types found: {unique_types}")
    
    # Filter out excluded classes (bare_ground and shadow)
    log_info(f"\nBefore filtering: {len(gdf)} polygons")
    exclude_mask = gdf[type_column].isin(EXCLUDE_CLASSES)
    excluded_count = exclude_mask.sum()
    
    if excluded_count > 0:
        excluded_types = gdf[exclude_mask][type_column].value_counts()
        log_info(f"Excluding {excluded_count} polygons:")
        for exc_type, count in excluded_types.items():
            log_info(f"  - {exc_type}: {count} polygons")
    
    # Keep only non-excluded polygons
    gdf_filtered = gdf[~exclude_mask].copy()
    log_info(f"After filtering: {len(gdf_filtered)} polygons")
    
    if len(gdf_filtered) == 0:
        log_info("Warning: All polygons were excluded!")
        return gdf_filtered
    
    # Initialize struct_id column (shorter name to avoid truncation)
    gdf_filtered['struct_id'] = -1  # Default value for unclassified
    
    # Assign structure IDs based on vegetation type
    log_info(f"\nAssigning structure IDs:")
    for veg_type, structure_id in STRUCTURE_ID_MAPPING.items():
        mask = gdf_filtered[type_column] == veg_type
        if mask.sum() > 0:
            gdf_filtered.loc[mask, 'struct_id'] = structure_id
            log_info(f"  - {veg_type} -> ID {structure_id}: {mask.sum()} polygons")
    
    # Check for unclassified vegetation types
    unclassified_mask = gdf_filtered['struct_id'] == -1
    if unclassified_mask.sum() > 0:
        unclassified_types = gdf_filtered[unclassified_mask][type_column].unique()
        log_info(f"\nWarning: {unclassified_mask.sum()} polygons with unrecognized types: {list(unclassified_types)}")
        log_info("These will be assigned default struct_id = 1 (woody debris/stem)")
        gdf_filtered.loc[unclassified_mask, 'struct_id'] = 1
    
    return gdf_filtered

# Start logging
log_info("=" * 80)
log_info(f"STRUCTURE CLASSIFICATION LOG")
log_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_info("=" * 80)
log_info(f"\nCONFIGURATION:")
log_info(f"Input directory: {directory_path}")
log_info(f"Output directory: {output_directory_path}")
log_info(f"Log directory: {log_directory}")
log_info(f"\nStructure ID Mapping:")
for veg_type, structure_id in STRUCTURE_ID_MAPPING.items():
    log_info(f"  {veg_type} -> ID {structure_id}")
log_info(f"\nExcluded classes: {EXCLUDE_CLASSES}")
log_info("=" * 80)

# Initialize summary statistics
total_files_processed = 0
total_polygons_original = 0
total_polygons_final = 0
file_summary = []

# Loop through each file in the directory
log_info(f"\nProcessing shapefiles from: {directory_path}")
log_info(f"Output directory: {output_directory_path}")
log_info("=" * 60)

for filename in os.listdir(directory_path):
    # Check if the file is a shapefile
    if filename.endswith(".shp"):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        log_info(f"\n📁 Processing shapefile: {filename}")
        
        try:
            # Load the shapefile as a GeoDataFrame
            gdf = gpd.read_file(file_path)
            original_count = len(gdf)
            total_polygons_original += original_count
            
            # Apply structure classification
            gdf_classified = classify_vegetation_structure(gdf, filename)
            
            if len(gdf_classified) == 0:
                log_info(f"⚠️  Skipping {filename} - no polygons remain after filtering")
                file_summary.append(f"{filename}: SKIPPED (no polygons after filtering)")
                continue
            
            # Calculate the area of each polygon and add it as a new column
            gdf_classified['area'] = gdf_classified.geometry.area.astype(float)
            
            final_count = len(gdf_classified)
            total_polygons_final += final_count
            
            log_info(f"\n📊 Final structure ID distribution:")
            structure_counts = gdf_classified['struct_id'].value_counts().sort_index()
            structure_names = {1: 'Woody debris/stem', 2: 'Grass', 3: 'Shrub', 4: 'Mallee'}
            
            file_stats = []
            for structure_id, count in structure_counts.items():
                structure_name = structure_names.get(structure_id, 'Unknown')
                total_area = gdf_classified[gdf_classified['struct_id'] == structure_id]['area'].sum()
                stat_line = f"  ID {structure_id} ({structure_name}): {count} polygons, {total_area:.2f} m² total area"
                log_info(stat_line)
                file_stats.append(f"ID{structure_id}: {count}poly/{total_area:.1f}m²")

            # Construct the output file path
            output_file_path = os.path.join(output_directory_path, filename)
            
            # Save the updated GeoDataFrame back to a new shapefile
            gdf_classified.to_file(output_file_path)
            log_info(f"✅ Saved to: {output_file_path}")
            
            total_files_processed += 1
            file_summary.append(f"{filename}: {original_count}→{final_count} polygons ({', '.join(file_stats)})")
            
        except Exception as e:
            error_msg = f"❌ Error processing {filename}: {str(e)}"
            log_info(error_msg)
            file_summary.append(f"{filename}: ERROR - {str(e)}")

# Write summary statistics
log_info("\n" + "=" * 80)
log_info("PROCESSING SUMMARY")
log_info("=" * 80)
log_info(f"Total files processed: {total_files_processed}")
log_info(f"Total polygons (original): {total_polygons_original}")
log_info(f"Total polygons (final): {total_polygons_final}")
log_info(f"Polygons filtered out: {total_polygons_original - total_polygons_final}")
log_info(f"Reduction percentage: {((total_polygons_original - total_polygons_final) / total_polygons_original * 100):.1f}%" if total_polygons_original > 0 else "0.0%")

log_info(f"\nFILE-BY-FILE SUMMARY:")
for summary in file_summary:
    log_info(f"  {summary}")

log_info(f"\n🎉 Processing complete! Structure-classified shapefiles saved to:")
log_info(f"   {output_directory_path}")
log_info(f"\n📄 Log file saved to: {log_file_path}")

print(f"\n📄 Detailed log saved to: {log_file_path}")
# %%
