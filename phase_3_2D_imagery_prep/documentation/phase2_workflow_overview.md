# Phase 2 Labeling Workflow Overview

## Completed: Step 1 - Structure 2D Labels
✅ **2_0_merge_shapefiles.py** - Merge individual annotation shapefiles
✅ **2_1_structure_2D_labels.py** - Convert FVC classes to structure classes (0-4)
✅ **2_2_structure_2D_rasterLabels.py** - Convert structure shapefiles to raster format
✅ **2_3_downsampling_labels.py** - Downsample rasters to 5cm resolution

## Step 2 - Structure 3D Analysis
🔄 **2_1_voxel_point_extraction.py** - Extract LiDAR points with structure labels
🔄 **2_2_height_distribution_analysis.py** - Analyze height patterns by structure type
📋 **2_3_voxel_grid_analysis.py** - Create 3D voxel grids for spatial analysis
📋 **2_4_height_threshold_optimization.py** - Determine optimal height thresholds

## Step 3 - Structure-Height Integration
📋 **3_1_combined_classification.py** - Integrate 2D structure + height information
📋 **3_2_validation_metrics.py** - Validate combined classifications
📋 **3_3_label_refinement.py** - Refine labels based on height constraints

## Step 4 - Training Data Preparation
📋 **4_1_feature_extraction.py** - Extract spectral and spatial features
📋 **4_2_train_test_split.py** - Split data for training/validation
📋 **4_3_data_augmentation.py** - Augment training dataset if needed

## Step 5 - Model Training & Validation
📋 **5_1_model_training.py** - Train structure classification models
📋 **5_2_cross_validation.py** - Validate model performance
📋 **5_3_prediction_mapping.py** - Generate structure prediction maps

## Structure Classification Scheme
- **ID 0: Forb** - Herbaceous vegetation close to ground
- **ID 1: Woody debris/stem** - Dead wood, litter, coarse woody debris
- **ID 2: Grass** - Grass hummocks (live and senescent)
- **ID 3: Shrub** - All shrub types including chenopods
- **ID 4: Mallee** - Mallee vegetation (tallest structure)

## Next Immediate Tasks
1. Run voxel point extraction for all sites (Step 2.1)
2. Analyze height distributions by structure (Step 2.2)
3. Create 3D voxel analysis framework (Step 2.3)
4. Determine height-based classification rules (Step 2.4)