#%%
import subprocess

def run_pdal_info(las_path):
    print(f"\n🔍 Inspecting: {las_path}")
    subprocess.run(["pdal", "info", "--metadata", las_path])

# Replace with your LAS file paths
source_las = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.26.las")
target_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26.las")

run_pdal_info(source_las)
run_pdal_info(target_las)

#%%
#test1
import json
import subprocess
from pathlib import Path

def read_las_metadata(las_path):
    """
    Uses `pdal info --metadata` to extract scale, offset, and SRS from a LAS file.
    Returns: scale_x, scale_y, scale_z, offset_x, offset_y, offset_z, srs
    """
    las_path = str(Path(las_path).resolve())
    
    cmd = ["pdal", "info", "--metadata", las_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        metadata = json.loads(result.stdout)

        reader = metadata["metadata"]["readers.las"]
        scale_x = reader["scale_x"]
        scale_y = reader["scale_y"]
        scale_z = reader["scale_z"]
        offset_x = reader["offset_x"]
        offset_y = reader["offset_y"]
        offset_z = reader["offset_z"]
        srs = reader.get("comp_spatialreference", None)

        return {
            "scale": (scale_x, scale_y, scale_z),
            "offset": (offset_x, offset_y, offset_z),
            "srs": srs
        }
    
    except subprocess.CalledProcessError as e:
        print("❌ Error running pdal info:")
        print(e.stderr)
    except KeyError as ke:
        print(f"❌ Key missing in metadata: {ke}")
    except Exception as ex:
        print(f"❌ Unexpected error: {ex}")
    
    return None

source_metadata = read_las_metadata("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.26.las")
target_metadata = read_las_metadata("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26_icp.las")

# source_path = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.26.las")
# target_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26_icp.las")
# output_pcd_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26_aligned.las")
# matrix_out_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/icp_transformation_matrix.txt")


if source_metadata:
    print("🔹 Source scale:", source_metadata["scale"])
    print("🔹 Source offset:", source_metadata["offset"])
    print("🔹 Source SRS:", source_metadata["srs"])



# %%
##test 2
# Import the necessary library
import cloudComPy as cc
import os 

# --- 1. Load Point Clouds (LAS/LAZ Setup) ---
# NOTE: Replace these placeholder paths with your actual LAS/LAZ file paths.
# Based on the GUI, the input clouds are:
# Data (To be aligned, RED): tiles_multispectral - Cloud
# Model (Reference, YELLOW): tiles_multispectral - imagery
data_cloud_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26.las" 
model_cloud_path = "/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.26.las"

try:
    if not os.path.exists(data_cloud_path) or not os.path.exists(model_cloud_path):
        raise FileNotFoundError("Please update the file paths for your LAS/LAZ point clouds.")

    # Load the point clouds
    data_cloud = cc.loadPointCloud(data_cloud_path)
    model_cloud = cc.loadPointCloud(model_cloud_path)

    if not data_cloud or not model_cloud:
        raise TypeError("Failed to load one or both point clouds. Check file formats and paths.")
        
    print(f"LAS files loaded: Data ({data_cloud.size()} pts), Model ({model_cloud.size()} pts)")

except (FileNotFoundError, TypeError) as e:
    print(f"Error during file loading: {e}")
    exit()

# --- 2. Set ICP Parameters from Screenshots ---
# Parameters from the 'Parameters' tab (Screenshot 100942):
MAX_ITERATIONS = 20
MIN_RMS_DECREASE = 1.0e-5 # Corresponds to "RMS difference"
FINAL_OVERLAP_RATIO = 1.0  # Corresponds to "Final overlap 100%"
# 'adjust scale' is unchecked -> ADJUST_SCALE = False
# 'max thread count 8/8' (set to 0 for maximum/auto) -> MAX_THREADS = 0
# 'Normals Ignored' -> NORMAL_MATCHING = NO_NORMAL

# Parameters from the 'Research' tab (Screenshot 100950):
RANDOM_SAMPLING_LIMIT = 50000 
REMOVE_FARTHEST = False # True=Corresponds to "Enable farthest points removal"

# Transformation Filters: All Rotations (XYZ) and Translations (Tx, Ty, Tz) are enabled.
# The default setting, SKIP_NONE (0), allows all 6 degrees of freedom.
TRANSFORMATION_FILTERS = cc.TRANSFORMATION_FILTERS.SKIP_NONE


# --- 3. Run the ICP Algorithm ---
print("\nStarting ICP registration with CloudCompare GUI settings...")

icp_result = cc.ICP(
    data=data_cloud,
    model=model_cloud,
    minRMSDecrease=MIN_RMS_DECREASE,
    maxIterationCount=MAX_ITERATIONS,
    randomSamplingLimit=RANDOM_SAMPLING_LIMIT,
    removeFarthestPoints=REMOVE_FARTHEST,
    method=cc.CONVERGENCE_TYPE.MAX_ITER_CONVERGENCE,
    adjustScale=False, # Unchecked in GUI
    finalOverlapRatio=FINAL_OVERLAP_RATIO,
    transformationFilters=TRANSFORMATION_FILTERS, # Skip none = All 6DoF enabled (XYZ, TxTyTz)
    maxThreadCount=8, # Auto/Max threads=0
    normalMatching=cc.normalMatching.NO_NORMAL, # Normals Ignored
    robustC2MSignedDistances=True # Default for robustness
)


# --- 4. Process and Save Results ---
if icp_result and icp_result.transMat:
    final_transformation = icp_result.transMat
    final_rms = icp_result.finalRMS
    
    print("ICP finished successfully! 🎉")
    print(f"Final RMS error: {final_rms:.8f}")
    
    # Print the resulting 4x4 rigid transformation matrix
    print("\nFinal Transformation Matrix (6 Degrees of Freedom used):")
    print(final_transformation)

    # Apply the calculated transformation to the original data cloud
    data_cloud.applyRigidTransform(final_transformation)
    print("\nTransformation applied to the data cloud.")

    # Save the aligned data cloud back into a new LAS file
    output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/icp/icp_tiles_multispectral.26.las"
    cc.SavePointCloud(data_cloud, output_path)
    print(f"Aligned cloud saved to: {output_path}")

else:
    print("ICP registration failed. Check initial alignment and point cloud health.")
# %%
