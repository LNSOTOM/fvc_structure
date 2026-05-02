#!/bin/bash
# filepath: /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool/1_2_rayimport_convert_las.sh

# no "set -e" so a single failure doesn't stop the whole flow
set -u

echo "🚁 Rayimport with Two LiDAR Datasets + Their Trajectories"
echo "=================================================="

SCRIPT_DIR="/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool"
OUT_DIR="/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output"
RAYIMP_DIR="$OUT_DIR/rayimport_samples"
TRJ_SRC_DIR="/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/trayectory"

BUILD_DIR="/media/laura/laura_usb/code/fvc_structure/raycloudtools/build"
RAYIMPORT_BIN="$BUILD_DIR/bin/rayimport-0.0.1"
RAYALIGN_BIN="$BUILD_DIR/bin/rayalign-0.0.1"

# Inputs
LIDAR_L1_2022="/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las"
LIDAR_L2_2024="/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las"

L1_TRJ_IN="$TRJ_SRC_DIR/DJI_20220519102349_0001_Zenmuse-L1-mission_sbet.txt"
L2_TRJ_IN="$TRJ_SRC_DIR/DJI_20240312130506_0002_L_sbet.txt"

L1_TRJ_OUT="$RAYIMP_DIR/trajectory_L1_20220519_rayimport.txt"
L2_TRJ_OUT="$RAYIMP_DIR/trajectory_L2_20240312_rayimport.txt"

mkdir -p "$RAYIMP_DIR"

# ------------------------------------------------------------------
# Manual commands (exact one-liners you asked to add for reference):
#
# L1
# python /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool/1_2_convert_trajectory.py \
#   --trajectory /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/trayectory/DJI_20220519102349_0001_Zenmuse-L1-mission_sbet.txt \
#   --lidar      /media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las \
#   --output     /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/rayimport_samples/trajectory_L1_20220519_rayimport.txt \
#   --target-epsg 7854 \
#   --sow-col Time --lat-col Latitude --lon-col Longitude --alt-col Altitude \
#   --force-remap
#
# L2
# python /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool/1_2_convert_trajectory.py \
#   --trajectory /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/trayectory/DJI_20240312130506_0002_L_sbet.txt \
#   --lidar      /media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las \
#   --output     /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/rayimport_samples/trajectory_L2_20240312_rayimport.txt \
#   --target-epsg 7854 \
#   --sow-col Time --lat-col Latitude --lon-col Longitude --alt-col Altitude \
#   --force-remap
# ------------------------------------------------------------------

echo "🔧 Step 1: Converting trajectory files using LAS time ranges..."

convert_traj () {
  local IN_TRAJ="$1"
  local IN_LAS="$2"
  local OUT_TRAJ="$3"
  local TAG="$4"

  if [[ -f "$IN_TRAJ" && -f "$IN_LAS" ]]; then
    echo "   ▶ $TAG: $(basename "$IN_TRAJ")  +  $(basename "$IN_LAS")"
    python3 "$SCRIPT_DIR/1_2_convert_trajectory.py" \
      --trajectory "$IN_TRAJ" \
      --lidar "$IN_LAS" \
      --output "$OUT_TRAJ" \
      --target-epsg 7854 \
      --sow-col Time --lat-col Latitude --lon-col Longitude --alt-col Altitude \
      --force-remap | tee "$RAYIMP_DIR/convert_${TAG}.log"
    if [[ ! -s "$OUT_TRAJ" ]]; then
      echo "   ❌ $TAG conversion did not produce $OUT_TRAJ"
    else
      echo "   ✅ $TAG trajectory written: $OUT_TRAJ"
    fi
  else
    echo "   ❌ Missing $TAG inputs"
    [[ ! -f "$IN_TRAJ" ]] && echo "      TRJ=$IN_TRAJ (missing)"
    [[ ! -f "$IN_LAS"  ]] && echo "      LAS=$IN_LAS (missing)"
  fi
}

convert_traj "$L1_TRJ_IN" "$LIDAR_L1_2022" "$L1_TRJ_OUT" "L1"
convert_traj "$L2_TRJ_IN" "$LIDAR_L2_2024" "$L2_TRJ_OUT" "L2"

echo ""
echo "🔄 Step 2: Running rayimport..."

if [[ ! -x "$RAYIMPORT_BIN" ]]; then
  echo "❌ rayimport binary not found at $RAYIMPORT_BIN"
  echo "   Build it in $BUILD_DIR first."
  exit 1
fi

cd "$BUILD_DIR" || { echo "❌ Cannot cd to $BUILD_DIR"; exit 1; }

# Clean old raycloud files from previous runs (only those with our tag)
echo ""
echo "🧹 Cleaning old raycloud files (tag _runply_)..."
rm -f *_runply_*.ply

import_pair () {
  local LAS="$1"
  local TRAJ="$2"
  local TAG="$3"

  if [[ ! -f "$LAS" ]]; then echo "   ❌ LAS missing: $LAS"; return; fi
  if [[ ! -f "$TRAJ" ]]; then echo "   ❌ TRAJ missing: $TRAJ"; return; fi

  local LAS_DIR TRAJ_DIR LAS_BASE
  LAS_DIR="$(dirname "$LAS")"
  TRAJ_DIR="$(dirname "$TRAJ")"
  LAS_BASE="$(basename "$LAS" .las)"

  # mark time before running rayimport (markers in all candidate dirs)
  local TS; TS="$(date +%s)"
  local MARK_BUILD="$RAYIMP_DIR/.marker_${TAG}_${TS}"
  local MARK_LAS="$LAS_DIR/.marker_${TAG}_${TS}"
  local MARK_TRJ="$TRAJ_DIR/.marker_${TAG}_${TS}"
  : > "$MARK_BUILD"; : > "$MARK_LAS"; : > "$MARK_TRJ"

  echo ""
  echo "📊 Importing $TAG..."
  echo "   LAS : $(basename "$LAS")"
  echo "   TRAJ: $(basename "$TRAJ")"
  "$RAYIMPORT_BIN" "$LAS" "$TRAJ" | tee "$RAYIMP_DIR/rayimport_${TAG}.log"

  # find newest PLY created after our markers in BUILD, LAS, and TRAJ dirs
  local NEWPLY=""
  NEWPLY="$(
    { find "$PWD"     -maxdepth 1 -type f -name "*.ply" -newer "$MARK_BUILD" -printf "%T@ %p\n" 2>/dev/null
      find "$LAS_DIR" -maxdepth 1 -type f -name "*.ply" -newer "$MARK_LAS"   -printf "%T@ %p\n" 2>/dev/null
      find "$TRAJ_DIR" -maxdepth 1 -type f -name "*.ply" -newer "$MARK_TRJ"  -printf "%T@ %p\n" 2>/dev/null; } \
    | sort -n | tail -1 | awk '{print $2}'
  )"

  # fallback to common filenames some builds use
  if [[ -z "$NEWPLY" || ! -f "$NEWPLY" ]]; then
    for cand in \
      "$TRAJ_DIR/lidar_${TAG,,}_rayimport.ply" \
      "$TRAJ_DIR/${LAS_BASE}_rayimport.ply" \
      "$TRAJ_DIR/${LAS_BASE}_raycloud.ply" \
      "$LAS_DIR/${LAS_BASE}_rayimport.ply" \
      "$LAS_DIR/${LAS_BASE}_raycloud.ply" \
      "$PWD/${LAS_BASE}_rayimport.ply" \
      "$PWD/${LAS_BASE}_raycloud.ply"
    do
      [[ -f "$cand" ]] && { NEWPLY="$cand"; break; }
    done
  fi

  if [[ -n "$NEWPLY" && -f "$NEWPLY" ]]; then
    local OUTPLY="raycloud_${TAG}_runply_$(date +%s).ply"
    mv -f "$NEWPLY" "$OUTPLY"
    echo "   ✅ $TAG raycloud: $OUTPLY  (from $(basename "$NEWPLY"))"
  else
    echo "   ❌ $TAG raycloud not found after import"
    echo "      Looked in: $PWD , $LAS_DIR , $TRAJ_DIR"
    echo "      Tip: open log → $RAYIMP_DIR/rayimport_${TAG}.log"
  fi

  rm -f "$MARK_BUILD" "$MARK_LAS" "$MARK_TRJ"
}



import_pair "$LIDAR_L1_2022" "$L1_TRJ_OUT" "L1"
import_pair "$LIDAR_L2_2024" "$L2_TRJ_OUT" "L2"

echo ""
echo "📁 Checking all created raycloud files..."
if ls *_runply_*.ply 1> /dev/null 2>&1; then
  echo "✅ Raycloud files created:"
  ls -la *_runply_*.ply

  RAYCLOUD_COUNT=$(ls *_runply_*.ply 2>/dev/null | wc -l)
  echo ""
  echo "📊 Summary: $RAYCLOUD_COUNT raycloud files created"

  if [[ "$RAYCLOUD_COUNT" -ge 2 ]]; then
    # pick deterministic order: fix L1, move L2
    FIX="raycloud_L1_runply_"*
    MOV="raycloud_L2_runply_"*
    echo ""
    echo "🎯 Ready for alignment:"
    echo "   $RAYALIGN_BIN '$FIX' '$MOV' --verbose"
  elif [[ "$RAYCLOUD_COUNT" -eq 1 ]]; then
    echo "   ⚠️ Only 1 raycloud created - need 2 for alignment"
  else
    echo "   ⚠️ Need at least 2 raycloud files for alignment"
  fi
else
  echo "❌ No raycloud files created"
  echo ""
  echo "🔧 Troubleshooting:"
  echo "   Check inputs exist:"
  echo "   - L1 LAS: $([[ -f "$LIDAR_L1_2022" ]] && echo EXISTS || echo MISSING)"
  echo "   - L2 LAS: $([[ -f "$LIDAR_L2_2024" ]] && echo EXISTS || echo MISSING)"
  echo "   - L1 TRJ: $([[ -f "$L1_TRJ_OUT" ]] && echo EXISTS || echo MISSING)"
  echo "   - L2 TRJ: $([[ -f "$L2_TRJ_OUT" ]] && echo EXISTS || echo MISSING)"
  [[ -f "$L1_TRJ_OUT" ]] && { echo ""; echo "   L1 trajectory first line:"; head -1 "$L1_TRJ_OUT"; }
  [[ -f "$L2_TRJ_OUT" ]] && { echo "   L2 trajectory first line:"; head -1 "$L2_TRJ_OUT"; }
fi

echo ""
echo "🎉 Script completed!"
echo "   Expected outputs:"
echo "   - L1 raycloud: From 2022 LiDAR + 2022 trajectory"
echo "   - L2 raycloud: From 2024 LiDAR + 2024 trajectory"

exit 0



## AFTERT AGAINA
# pdal translate \
#   "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c_aligned.ply" \
#   "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c_aligned_shifted.ply" \
#   filters.transformation \
#   --filters.transformation.matrix="1 0 0 15.25  0 1 0 840.75  0 0 1 -8.984  0 0 0 1"