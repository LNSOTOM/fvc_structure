
#!/bin/bash

# Create the organized folder structure for Phase 2 workflow
BASE_DIR="/media/laura/laura_usb/code/fvc_structure/phase_2_labeling_workflow"

# Step 2: 3D Structure Analysis (Height-based classification)
mkdir -p "$BASE_DIR/step_2_structure_3D_analysis"
mkdir -p "$BASE_DIR/step_2_structure_3D_analysis/scripts"
mkdir -p "$BASE_DIR/step_2_structure_3D_analysis/outputs"
mkdir -p "$BASE_DIR/step_2_structure_3D_analysis/outputs/height_stats"
mkdir -p "$BASE_DIR/step_2_structure_3D_analysis/outputs/voxel_analysis"

# Step 3: Structure-Height Integration
mkdir -p "$BASE_DIR/step_3_structure_height_integration"
mkdir -p "$BASE_DIR/step_3_structure_height_integration/scripts"
mkdir -p "$BASE_DIR/step_3_structure_height_integration/outputs"
mkdir -p "$BASE_DIR/step_3_structure_height_integration/outputs/combined_labels"
mkdir -p "$BASE_DIR/step_3_structure_height_integration/outputs/validation"

# Step 4: Training Data Preparation
mkdir -p "$BASE_DIR/step_4_training_data_prep"
mkdir -p "$BASE_DIR/step_4_training_data_prep/scripts"
mkdir -p "$BASE_DIR/step_4_training_data_prep/outputs"
mkdir -p "$BASE_DIR/step_4_training_data_prep/outputs/train_test_split"
mkdir -p "$BASE_DIR/step_4_training_data_prep/outputs/feature_extraction"

# Step 5: Model Training & Validation
mkdir -p "$BASE_DIR/step_5_model_training"
mkdir -p "$BASE_DIR/step_5_model_training/scripts"
mkdir -p "$BASE_DIR/step_5_model_training/models"
mkdir -p "$BASE_DIR/step_5_model_training/results"
mkdir -p "$BASE_DIR/step_5_model_training/validation"

# Documentation and logs
mkdir -p "$BASE_DIR/documentation"
mkdir -p "$BASE_DIR/logs"

echo "✅ Phase 2 folder structure created successfully!"