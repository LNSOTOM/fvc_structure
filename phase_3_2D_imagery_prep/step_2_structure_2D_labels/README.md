# Structure Classification for 2D Labels

This script processes shapefiles containing vegetation annotations and applies a structure-based classification system.

## Overview

The script reads shapefiles from the FVC (Fractional Vegetation Cover) classification pipeline and creates new structure-classified shapefiles with the following mapping:

### Structure ID Classification

- **ID 1 (Woody debris/stem)**: `branch_stem_standing`, `litter`, `cwd` (Coarse Woody Debris)
- **ID 2 (Grass)**: `grass_hummock`, `grass_hummock_senescent`
- **ID 3 (Shrub)**: `shrub`
- **ID 4 (Mallee)**: `mallee`

### Excluded Classes

The following vegetation types are **filtered out** and not included in the output:

- `bare_ground`
- `shadow`

## Input Requirements

- Shapefiles should contain a column indicating vegetation type (common column names: `class`, `type`, `label`, `vegetation`, `class_name`)
- Input shapefiles are expected to be in the configured `directory_path`

## Output

For each input shapefile, the script creates:

- A new shapefile with the same name in the `output_directory_path`
- Additional columns:
  - `structure_id`: Integer ID (1-4) based on the classification mapping
  - `area`: Calculated area of each polygon (float)

## Configuration

Edit the script to set appropriate paths for your site:

```python
# Choose your site by uncommenting the appropriate lines:

# low site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
# output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'

# medium site (currently active)
directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
output_directory_path = '/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotations_shp'

# dense site
# directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/preprocess/'
# output_directory_path = '/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp'
```

## Usage

1. Ensure your conda environment is active: `conda activate fvc_structure`
2. Run the script: `python structure_2D_labels.py`

## Output Example

The script will provide detailed logging:

```text
Processing shapefile: /path/to/input.shp
Original shapefile has 100 polygons
Using vegetation type column: 'class'
Unique vegetation types found: ['grass_hummock' 'shrub' 'bare_ground' 'mallee' 'shadow']
Excluding 25 polygons with classes: ['bare_ground', 'shadow']
Assigned ID 2 to 15 polygons of type 'grass_hummock'
Assigned ID 3 to 35 polygons of type 'shrub'
Assigned ID 4 to 25 polygons of type 'mallee'
Final shapefile has 75 polygons
Structure ID distribution:
  ID 2 (Grass): 15 polygons
  ID 3 (Shrub): 35 polygons
  ID 4 (Mallee): 25 polygons
Updated shapefile saved to: /path/to/output.shp
```

## Dependencies

- geopandas
- os (built-in)

The required packages are available in the `fvc_structure` conda environment.

## Notes

- If no vegetation type column is found, all polygons will be assigned default `structure_id = 1` (Woody debris/stem)
- Unrecognized vegetation types will also be assigned `structure_id = 1` with a warning
- The script automatically creates the output directory if it doesn't exist
- All calculations preserve the original coordinate system and geometry precision
