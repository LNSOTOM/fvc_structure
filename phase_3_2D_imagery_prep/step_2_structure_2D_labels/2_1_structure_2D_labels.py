#%%
import geopandas as gpd
import os
from datetime import datetime

# Structure classification mapping - REVISED SCHEME
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

# Classes to exclude (delete) - added 'be' back to exclusion list
EXCLUDE_CLASSES = ['bare_ground', 'shadow', 'be', 'unknowm']

# Sites configuration
SITES_CONFIG = {
    'low': {
        'directory_path': '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp',
        'output_directory_path': '/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'
    },
    'medium': {
        'directory_path': '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp',
        'output_directory_path': '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'
    },
    'dense': {
        'directory_path': '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp',
        'output_directory_path': '/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp'
    }
}

# *** CHANGE THIS TO PROCESS DIFFERENT SITES ***
CURRENT_SITE = 'medium'  # Change this to 'low', 'medium', or 'dense'
# *** TO PROCESS LOW SITE: CURRENT_SITE = 'low' ***
# *** TO PROCESS MEDIUM SITE: CURRENT_SITE = 'medium' ***
# *** TO PROCESS DENSE SITE: CURRENT_SITE = 'dense' ***

directory_path = SITES_CONFIG[CURRENT_SITE]['directory_path']
output_directory_path = SITES_CONFIG[CURRENT_SITE]['output_directory_path']

# Ensure the output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Log file location - save in the script directory
log_directory = '/media/laura/laura_usb/code/fvc_structure/phase_2_2D_imagery_prep/step_2_structure_2D_labels'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_directory, f"structure_classification_log_{CURRENT_SITE}_{timestamp}.txt")

def log_info(message, print_console=True):
    """Log information to both console and file"""
    if print_console:
        print(message)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')

def get_structure_name(structure_id, found_types):
    """Get appropriate structure name based on what types were actually found"""
    if structure_id == 0:
        return 'Forb'
    elif structure_id == 1:
        return 'Woody debris/stem'
    elif structure_id == 2:
        return 'Grass'
    elif structure_id == 3:
        # Check which shrub types are actually present
        shrub_types = []
        if any('chenopod' in t for t in found_types):
            shrub_types.append('Chenopod')
        if any(t == 'shrub' for t in found_types):
            shrub_types.append('Shrub')
        
        if shrub_types:
            return 'Shrub (' + '/'.join(shrub_types) + ')'
        else:
            return 'Shrub'
    elif structure_id == 4:
        return 'Mallee'
    else:
        return 'Unknown'

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
    
    # Filter out excluded classes FIRST - before any struct_id assignment
    log_info(f"\nBefore filtering: {len(gdf)} polygons")
    exclude_mask = gdf[type_column].str.lower().str.replace(' ', '_').isin([cls.lower() for cls in EXCLUDE_CLASSES])
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
    
    # Assign structure IDs based on vegetation type - ONLY for known types
    log_info(f"\nAssigning structure IDs:")
    for veg_type, structure_id in STRUCTURE_ID_MAPPING.items():
        mask = gdf_filtered[type_column] == veg_type
        if mask.sum() > 0:
            gdf_filtered.loc[mask, 'struct_id'] = structure_id
            log_info(f"  - {veg_type} -> ID {structure_id}: {mask.sum()} polygons")
    
    # Check for unclassified vegetation types - REMOVE THESE INSTEAD OF ASSIGNING ID 1
    unclassified_mask = gdf_filtered['struct_id'] == -1
    if unclassified_mask.sum() > 0:
        unclassified_types = gdf_filtered[unclassified_mask][type_column].unique()
        log_info(f"\nRemoving {unclassified_mask.sum()} polygons with unrecognized types:")
        for unclass_type in sorted(unclassified_types):
            count = (gdf_filtered[type_column] == unclass_type).sum()
            log_info(f"  - '{unclass_type}': {count} polygons")
        
        # REMOVE unclassified polygons instead of assigning them ID 1
        gdf_filtered = gdf_filtered[~unclassified_mask].copy()
        log_info(f"After removing unclassified: {len(gdf_filtered)} polygons")
    
    return gdf_filtered

# Start logging
log_info("=" * 80)
log_info(f"STRUCTURE CLASSIFICATION LOG - {CURRENT_SITE.upper()} SITE")
log_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_info("=" * 80)
log_info(f"\nCONFIGURATION:")
log_info(f"Current site: {CURRENT_SITE}")
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
all_found_types = set()
all_missing_types = set()

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
            
            # Track found types for global summary
            if len(gdf) > 0:
                type_col = None
                for col in ['structure', 'class', 'type', 'label', 'vegetation', 'class_name', 'classname', 'CLASS', 'TYPE']:
                    if col in gdf.columns:
                        type_col = col
                        break
                if type_col:
                    file_types = set(gdf[type_col].astype(str).unique())
                    all_found_types.update(file_types)
            
            log_info(f"\n📊 Final structure ID distribution:")
            structure_counts = gdf_classified['struct_id'].value_counts().sort_index()
            
            file_stats = []
            for structure_id, count in structure_counts.items():
                structure_name = get_structure_name(structure_id, all_found_types)
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

# Find missing types
known_types = set(STRUCTURE_ID_MAPPING.keys())
exclude_types = set(EXCLUDE_CLASSES)
all_missing_types = all_found_types - known_types - exclude_types

# Write summary statistics
log_info("\n" + "=" * 80)
log_info("PROCESSING SUMMARY")
log_info("=" * 80)
log_info(f"Total files processed: {total_files_processed}")
log_info(f"Total polygons (original): {total_polygons_original}")
log_info(f"Total polygons (final): {total_polygons_final}")
log_info(f"Polygons filtered out: {total_polygons_original - total_polygons_final}")
log_info(f"Reduction percentage: {((total_polygons_original - total_polygons_final) / total_polygons_original * 100):.1f}%" if total_polygons_original > 0 else "0.0%")

log_info(f"\nSTRUCTURE ID SUMMARY BY ACTUAL TYPES FOUND:")
for structure_id in sorted(set(STRUCTURE_ID_MAPPING.values())):
    matching_found = [t for t in all_found_types if t in STRUCTURE_ID_MAPPING and STRUCTURE_ID_MAPPING[t] == structure_id]
    if matching_found:
        structure_name = get_structure_name(structure_id, all_found_types)
        log_info(f"  ID {structure_id} ({structure_name}): {', '.join(sorted(matching_found))}")

log_info(f"\nCLASS COVERAGE ANALYSIS:")
log_info(f"Total unique types found across all files: {len(all_found_types)}")
log_info(f"Known/mapped types: {len(all_found_types & known_types)}")
log_info(f"Excluded types: {len(all_found_types & exclude_types)}")
log_info(f"Unknown/unmapped types: {len(all_missing_types)}")

if all_missing_types:
    log_info(f"\nUNKNOWN TYPES FOUND (removed - not in STRUCTURE_ID_MAPPING or EXCLUDE_CLASSES):")
    for unknown_type in sorted(all_missing_types):
        log_info(f"  - '{unknown_type}'")

log_info(f"\nFILE-BY-FILE SUMMARY:")
for summary in file_summary:
    log_info(f"  {summary}")

log_info(f"\n🎉 Processing complete! Structure-classified shapefiles saved to:")
log_info(f"   {output_directory_path}")
log_info(f"\n📄 Log file saved to: {log_file_path}")

print(f"\n📄 Detailed log saved to: {log_file_path}")
# %%