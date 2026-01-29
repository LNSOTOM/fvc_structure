#!/bin/bash
# Test script to run the improved fvc3dstruct pipeline

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate fvc_structure

# Run the main pipeline script
cd /home/nvidia/projects/fvc_structure/phase_4_3D_lidar_prep/step_3_structure_class
python main_run.py
