#%% [markdown]
# # Phase 1 & 2: Full Co-registration Pipeline
# 
# This notebook runs the complete 2-phase workflow:
# 
# 1.  **PHASE 1 (Single GPU):**
#     - Loads the two DSMs (RGB and LiDAR).
#     - Runs the `EFolki` algorithm on **one GPU** (using CuPy) to 
#       calculate the non-linear flow fields (u, v).
#     - Saves the .npy files.
# 
# 2.  **PHASE 2 (8-GPU CLUSTER):**
#     - Starts an 8-GPU Dask-CUDA cluster (8x A100 80GB).
#     - Loads the .npy flow files just created.
#     - Scatters the flow fields to the 8 GPUs.
#     - Reads the massive .las file in chunks.
#     - Sends each chunk to a GPU for parallel warping.
#     - Saves the final, corrected .las file.

#%% [markdown]
# ## Cell 1: Imports
# Import all libraries for both phases.

#%%
import sys
import os
import numpy as np
import rasterio
import laspy
from pathlib import Path
import time
from tqdm import tqdm
import pylab as pl

# === UCX/NCCL (recommended for multi-GPU on a single node) ===
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")      # single node, no IB
os.environ.setdefault("UCX_TLS", "sm,cuda_copy,cuda_ipc,tcp")
os.environ.setdefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp")

# === GeFolki dependencies === NVIDIA
sys.path.append('/home/nvidia/projects/fvc_structure/phase_2_lidar_coregistration/gefolki/python')
from python.algorithm import EFolki
from python.tools import wrapData

# === GPU & Parallel Processing Imports ===
import cupy as cp
# Use cupyx.scipy for map_coordinates
from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates 

from dask_cuda import LocalCUDACluster
from dask.distributed import Client, as_completed

#%% [markdown]
# ## Cell 2: File & Path Configuration
# All file paths for both phases are defined here.

#%%
# === Base Directory ===
BASE_DIR = Path("/data/calperumResearch/site1_1_DD0001")

# --- Phase 1 (EFolki) Inputs ---
BASE_IMAGE_PATH = BASE_DIR / "imagery" / "dsm" / "SASMDD0001_rgb_dsm_03.tif"
MOVING_IMAGE_PATH = BASE_DIR / "lidar" / "dsm" / "crop_to_rgb" / "SASMDD0001_l1_dsm_03.tif"

# --- Phase 1 (EFolki) Outputs ---
# These are the .npy files that Phase 2 will use as inputs
FLOW_U_OUTPUT_PATH = BASE_DIR / "imagery" / "dsm" / "flow_u_03_l1.npy"
FLOW_V_OUTPUT_PATH = BASE_DIR / "imagery" / "dsm" / "flow_v_03_l1.npy"
ALIGNED_TIF_PATH = MOVING_IMAGE_PATH.with_name(MOVING_IMAGE_PATH.stem + "_coreg.tif")

# --- Phase 2 (Warping) Inputs ---
LAS_INPUT_PATH = BASE_DIR / "lidar" / "20220519_SASMDD0001_triple_AusGeoid.las"
# Use the TIF from Phase 1 as the grid for mapping points
REFERENCE_GRID_PATH = ALIGNED_TIF_PATH  

# --- Phase 2 (Warping) Output ---
LAS_OUTPUT_PATH = LAS_INPUT_PATH.with_name(LAS_INPUT_PATH.stem + "_coreg_gpu.las")

# === Processing Parameters ===
BASE_BAND_IDX = 0
CORRECTION_DIRECTION = -1
GPUS_TO_USE = "0,1,2,3,4,5"  # Specify the 6 GPUs for Phase 2
PHASE_1_GPU_ID = 0           # Specify the single GPU for Phase 1
LAS_CHUNK_SIZE = 4_000_000   # 4 million points per job

#%% [markdown]
# ## Cell 3: Helper Function (CPU)
# Define the raster loading function for Phase 1.

#%%
def load_data_and_mask(path, band_idx):
    """
    (CPU Function) Loads raster data and, crucially, also returns a boolean 'mask'
    that perfectly preserves the footprint of where the valid data is.
    """
    with rasterio.open(path) as src:
        if band_idx >= src.count:
            raise ValueError(f"Band {band_idx} is out of bounds.")
        
        # Get a True/False mask of the valid data footprint (255=valid, 0=nodata)
        valid_mask = src.read_masks(band_idx + 1) > 0
        
        # Read the data and set nodata areas to 0 for processing
        arr = src.read(band_idx + 1).astype(np.float32)
        arr[~valid_mask] = 0
        return arr, valid_mask

#%% [markdown]
# # --- PHASE 1 (Single GPU) ---
# ## Cell 4: Load and Plot Initial DSMs

#%%
# === Load base and moving images, capturing their masks ===
print("Loading Phase 1 data...")
base_data, base_mask = load_data_and_mask(BASE_IMAGE_PATH, BASE_BAND_IDX)
moving_data, moving_mask = load_data_and_mask(MOVING_IMAGE_PATH, 0)
print("Data loaded.")

# === Plot initial data ===
pl.figure(figsize=(8,8)); pl.imshow(base_data, cmap='gray'); pl.title("Base Image (RGB DSM)")
pl.figure(figsize=(8,8)); pl.imshow(moving_data, cmap='gray'); pl.title("Moving Image (LiDAR DSM)")
pl.show()

#%% [markdown]
# ## Cell 5: Define and Plot Processing Area

#%%
# === Define processing area ===
# The area to process is where BOTH images have valid data
processing_mask = base_mask & moving_mask
base_data[~processing_mask] = 0
moving_data[~processing_mask] = 0

pl.figure(figsize=(8,8)); pl.imshow(moving_data, cmap='gray'); pl.title("Processing Area (Intersection)")
pl.show()

#%% [markdown]
# ## Cell 6: Calculate Optical Flow (Single GPU)
# Runs EFolki with CPU multi-threading, while pinning a single CUDA device for Phase 1.
# (EFolki in this Python build is CPU-based; GPU is used in Phase 2.)

#%%
# === Calculate optical flow ===
print("--- Running EFolki on CPU with Multi-threading ---")
print(f"Using 64 CPU threads for parallel NumPy operations")
print("Note: The GeFolki Python implementation is CPU-based.")
print("GPU acceleration will be used in Phase 2 for point cloud warping.")

try:
    print("Calculating flow (this may take several minutes)...")
    t0 = time.perf_counter()

    # --- Run EFolki on CPU data (NumPy arrays) ---
    # The algorithm will automatically use the multi-threading we configured
    u, v = EFolki(base_data, moving_data, iteration=8, radius=[16, 8, 4], rank=4, levels=6)

    t1 = time.perf_counter()
    print(f"✅ Flow calculated on CPU in {(t1 - t0)/60:.3f} minutes")
    print(f"   Average: {(t1 - t0):.1f} seconds")

except Exception as e:
    print(f"FATAL ERROR during Phase 1 processing: {e}")
    raise

# --- Mask and Plot ---
u[~processing_mask] = 0
v[~processing_mask] = 0

N = np.sqrt(u**2 + v**2)
pl.figure(figsize=(8,8)); pl.imshow(N, cmap='viridis'); pl.title("Norm of Optical Flow"); pl.colorbar()
pl.show()


#%% [markdown]
# ## Cell 7: Register (Warp) TIF Image
# Warp the LiDAR TIF to match the RGB TIF.

#%%
# === Register the LiDAR data AND its footprint ===
print("Registering TIF image...\n")
# 1. Warp the LiDAR elevation data (uses numpy/CPU)
imag_reg = wrapData(moving_data, u, v)

# 2. Warp the LiDAR's original footprint (its mask)
warped_mask = wrapData(moving_mask.astype('float32'), u, v) > 0.5

pl.figure(figsize=(8,8)); pl.imshow(imag_reg); pl.title("Registered TIF Image")
pl.show()

#%% [markdown]
# ## Cell 8: Save Phase 1 Results
# Save the warped TIF and the critical `.npy` flow fields.

#%%
# === Save flow and the correctly registered image ===
print("\n--- Saving Phase 1 Results ---")

# 1. Get metadata from the base image for the output grid
with rasterio.open(BASE_IMAGE_PATH) as src:
    profile = src.profile
    nodata_value = src.nodata if src.nodata is not None else -9999

# 2. Apply the warped mask to set nodata values
imag_reg[~warped_mask] = nodata_value

# 3. Update profile and write the TIF
profile.update(dtype='float32', nodata=nodata_value)
with rasterio.open(ALIGNED_TIF_PATH, "w", **profile) as dst:
    dst.write(imag_reg.astype('float32'), 1)

print(f"✅ Registered TIF saved to: {ALIGNED_TIF_PATH}")

# --- Save the flow fields (INPUT FOR PHASE 2) ---
np.save(FLOW_U_OUTPUT_PATH, u)
np.save(FLOW_V_OUTPUT_PATH, v)
print(f"💾 Flow fields saved to: {FLOW_U_OUTPUT_PATH.parent}")
print("\n--- PHASE 1 COMPLETE ---")


#%% [markdown]
# # --- PHASE 2 (8-GPU CLUSTER) ---
# ## Cell 9: Define GPU Worker Function
# This function will be sent to all 8 GPUs.

#%%
def process_chunk_on_gpu(pts, transform_a, transform_e, inv_transform, H, W, flow_u_gpu, flow_v_gpu):
    """
    (GPU Function) This function is sent to a Dask-CUDA worker.
    It receives a 'pts' object and the scattered flow fields.
    """
    try:
        # 1. Find point address on the flow grid
        col, row = inv_transform * (pts.x, pts.y)
        
        # 2. Keep only points within the grid
        valid_mask = (col >= 0) & (col < W - 1) & (row >= 0) & (row < H - 1)
        if not np.any(valid_mask):
            return None

        valid_pts = pts[valid_mask]
        
        # 3. MOVE DATA TO GPU
        coords_gpu = cp.vstack((row[valid_mask], col[valid_mask]))

        # 4. COMPUTE ON GPU (CuPy interpolation)
        du_pix_gpu = cupy_map_coordinates(flow_u_gpu, coords_gpu, order=2, mode="nearest")
        dv_pix_gpu = cupy_map_coordinates(flow_v_gpu, coords_gpu, order=2, mode="nearest")

        # 5. Translate correction to meters (on GPU)
        dx_world_gpu = du_pix_gpu * transform_a
        dy_world_gpu = dv_pix_gpu * transform_e

        # 6. GET RESULTS FROM GPU
        dx_world = cp.asnumpy(dx_world_gpu)
        dy_world = cp.asnumpy(dy_world_gpu)

        # 7. Apply correction (on CPU)
        valid_pts.x += CORRECTION_DIRECTION * dx_world
        valid_pts.y += CORRECTION_DIRECTION * dy_world

        # 8. Return corrected points
        return valid_pts
        
    except Exception as e:
        print(f"Error in worker process: {e}")
        return None

#%% [markdown]
# ## Cell 10: Start 6-GPU Cluster & Scatter Data
# This cell initializes the cluster and pre-loads the data.

#%%
print("\n--- Starting Phase 2: Parallel GPU Point Cloud Warping ---")
phase2_start_time = time.time()

# --- 1. Load reference data on the main process ---
print(f"Loading data for Phase 2...")
print(f"Reference Grid: {REFERENCE_GRID_PATH}")
print(f"Flow U: {FLOW_U_OUTPUT_PATH}")
print(f"Flow V: {FLOW_V_OUTPUT_PATH}")

try:
    with rasterio.open(REFERENCE_GRID_PATH) as src:
        transform = src.transform 
        inv_transform = ~transform
        H, W = src.height, src.width
        transform_a = transform.a
        transform_e = transform.e
    
    # Load flow fields (created in Cell 8) into CPU RAM
    flow_u_cpu = np.load(FLOW_U_OUTPUT_PATH)
    flow_v_cpu = np.load(FLOW_V_OUTPUT_PATH)
    print("Metadata and flow fields loaded.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load Phase 1 outputs. Did you run Cell 8?")
    print(e)
    raise

# --- 2. Start the 6-GPU Dask Cluster ---
print(f"Starting Dask CUDA Cluster for 6 A100 GPUs ({GPUS_TO_USE})...")

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES=GPUS_TO_USE,   # "0,1,2,3,4,5"
    n_workers=6,                        # one worker per GPU
    threads_per_worker=20,              # ~120 CPU threads total; tune as needed
    rmm_pool_size="70GB",               # preallocate VRAM to avoid fragmentation
    rmm_async=True,                     # cudaMallocAsync
    enable_nvlink=True,                 # fast inter-GPU
    protocol="ucx",
    enable_tcp_over_ucx=True,
    enable_infiniband=False,
)

client = Client(cluster)
print(f"Cluster ready. Dashboard: {client.dashboard_link}")

num_workers = len(client.scheduler_info()["workers"])
print(f"Found {num_workers} workers.")
assert num_workers == 6, f"Expected 6 workers, found {num_workers}"

# --- 3. Scatter read-only data to all GPUs ---
print("Scattering flow fields to all GPU workers (VRAM)...")
flow_u_future = client.scatter(cp.asarray(flow_u_cpu), broadcast=True)
flow_v_future = client.scatter(cp.asarray(flow_v_cpu), broadcast=True)
print("Scatter complete.")


#%% [markdown]
# ## Cell 11: Run Parallel Warping & Shutdown
# This cell reads the LAS, submits all jobs, gathers results, and shuts down.

#%%
print(f"Opening {LAS_INPUT_PATH} for chunked reading...")
try:
    with laspy.open(LAS_INPUT_PATH) as reader:
        header = reader.header
        total_points = header.point_count
        
        print(f"Output file will be: {LAS_OUTPUT_PATH}")

        with laspy.open(LAS_OUTPUT_PATH, mode="w", header=header) as writer:
            
            futures = [] # To hold our list of "pending" jobs
            
            # --- 5. Submit all chunks as jobs to the cluster ---
            print("Submitting all processing jobs to the cluster...")
            pbar_submit = tqdm(total=total_points, desc="Submitting jobs", unit="pts")
            for pts in reader.chunk_iterator(LAS_CHUNK_SIZE):
                future = client.submit(
                    process_chunk_on_gpu,
                    pts,
                    transform_a, transform_e,
                    inv_transform,
                    H, W,
                    flow_u_future,
                    flow_v_future
                )
                futures.append(future)
                pbar_submit.update(len(pts))
            pbar_submit.close()
            print("All jobs submitted.")

            # --- 6. Gather results and write to file ---
            print("Gathering and writing results as they complete...")
            pbar_write = tqdm(total=total_points, desc="Writing points", unit="pts")
            for future in as_completed(futures):
                try:
                    corrected_pts = future.result()
                    if corrected_pts is not None:
                        writer.write_points(corrected_pts)
                        pbar_write.update(len(corrected_pts))
                except Exception as e:
                    print(f"A job failed to complete: {e}")
            pbar_write.close()

except FileNotFoundError:
    print(f"FATAL ERROR: Input LAS file not found at {LAS_INPUT_PATH}")
except Exception as e:
    print(f"An error occurred during LAS processing: {e}")
finally:
    # --- 7. Clean up ---
    print("Shutting down cluster.")
    client.close()
    cluster.close()
    
    phase2_end_time = time.time()
    print("\n--- PHASE 2 COMPLETE ---")
    print(f"✅ Saved final point cloud to: {LAS_OUTPUT_PATH}")
    print(f"Phase 2 (GPU) time: {(phase2_end_time - phase2_start_time) / 60:.2f} minutes")