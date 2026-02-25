#!/bin/bash
# run_full_lidar_alignment.sh
# Complete workflow: LAS to PLY, trajectory conversion, rayimport, and alignment

set -e

# Directories
SCRIPT_DIR="/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool"
OUT_DIR="/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output"
RAYCLOUD_BUILD="/media/laura/laura_usb/code/fvc_structure/raycloudtools/build"

# 1. Convert LAS to rayimport-compatible PLY
python3 "$SCRIPT_DIR/1_1_convert_las_to_ply-raycloud.py"

# 2. Convert trajectory files
python3 "$SCRIPT_DIR/1_2_convert_trajectory.py"

# 3. Import to raycloud format
cd "$RAYCLOUD_BUILD"
L1_PLY="$OUT_DIR/rayimport_samples/lidar_l1_rayimport.ply"
L2_PLY="$OUT_DIR/rayimport_samples/lidar_l2_rayimport.ply"
L1_TRAJ="$OUT_DIR/rayimport_samples/trajectory_L1_20220519_rayimport.txt"
L2_TRAJ="$OUT_DIR/rayimport_samples/trajectory_L2_20240312_rayimport.txt"

if [[ -f "$L1_PLY" && -f "$L1_TRAJ" ]]; then
    ./bin/rayimport-0.0.1 "$L1_PLY" "$L1_TRAJ"
else
    echo "❌ L1 PLY or trajectory missing!"
    exit 1
fi

if [[ -f "$L2_PLY" && -f "$L2_TRAJ" ]]; then
    ./bin/rayimport-0.0.1 "$L2_PLY" "$L2_TRAJ"
else
    echo "❌ L2 PLY or trajectory missing!"
    exit 1
fi

# 4. Align the two raycloud files (find most recent for each)
L1_RAYCLOUD=$(ls *raycloud*.ply | grep -i l1 | tail -1)
L2_RAYCLOUD=$(ls *raycloud*.ply | grep -i l2 | tail -1)

if [[ -f "$L1_RAYCLOUD" && -f "$L2_RAYCLOUD" ]]; then
    ./bin/rayalign-0.0.1 "$L1_RAYCLOUD" "$L2_RAYCLOUD" --verbose
    echo "🎉 Workflow complete! Aligned file created."
else
    echo "❌ Raycloud files for alignment missing!"
    exit 1
fi
