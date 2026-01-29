#%%
import sys
import numpy as np
import rasterio
import laspy
from pathlib import Path
import time
from tqdm import tqdm
import cupy as cp  # Import CuPy
from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates # GPU map_coordinates

# Dask and Dask-CUDA imports
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, as_completed

# --- CONFIGURATION (from your script) ---
LAS_INPUT_PATH = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
REFERENCE_GRID_PATH = Path("/data/calperumResearch/site1_1_DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03_coreg.tif")

FLOW_U_PATH = Path("/data/calperumResearch/site1_1_DD0001/imagery/dsm/flow_u_03_l1.npy")
FLOW_V_PATH = Path("/data/calperumResearch/site1_1_DD0001/imagery/dsm/flow_v_03_l1.npy")
LAS_OUTPUT_PATH = LAS_INPUT_PATH.with_name(LAS_INPUT_PATH.stem + "_coreg_dask.las")
CORRECTION_DIRECTION = -1
CHUNK_SIZE = 4_000_000 # 4 million points per chunk

# --- THIS FUNCTION WILL RUN ON YOUR GPUs ---
def process_chunk_on_gpu(x_coords):
    """Minimal debug task: just count points in the chunk."""
    return len(x_coords)

# --- THIS IS YOUR MAIN SCRIPT (RUNS ON CPU) ---
# It acts as the "manager" that feeds jobs to the GPU cluster.
if __name__ == '__main__':
    
    # --- 1. Load reference data on the main process ---
    print("Loading flow fields and raster metadata...")
    with rasterio.open(REFERENCE_GRID_PATH) as src:
        transform = src.transform 
        inv_transform = ~transform
        H, W = src.height, src.width
        transform_a = transform.a
        transform_e = transform.e
    
    # Load flow fields into CPU RAM
    flow_u_cpu = np.load(FLOW_U_PATH)
    flow_v_cpu = np.load(FLOW_V_PATH)
    print(f"Flow field shape: {flow_u_cpu.shape}")
    
    # --- 2. Start the 8-GPU Dask Cluster ---
    print("Starting 8-GPU Dask CUDA Cluster...")
    
    # Configure cluster with explicit settings for stability
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        n_workers=8,
        threads_per_worker=1,
        memory_limit='70GB'  # Leave some headroom on 80GB GPUs
    )
    
    client = Client(cluster)
    
    # Wait for all workers to be ready
    print("Waiting for workers to initialize...")
    client.wait_for_workers(n_workers=8, timeout=120)
    
    print(f"✅ Cluster ready. Dashboard: {client.dashboard_link}")
    print(f"✅ Workers: {len(client.scheduler_info()['workers'])}")

    # --- 3. Scatter read-only data to all GPUs ---
    # This copies the large flow fields to each GPU's VRAM *once*.
    print("Scattering flow fields to all GPU workers...")
    # Scatter as NumPy arrays - workers will convert to CuPy
    flow_u_future = client.scatter(flow_u_cpu, broadcast=True)
    flow_v_future = client.scatter(flow_v_cpu, broadcast=True)
    print("✅ Scatter complete.")

    # --- 4. Open LAS files for streaming ---
    print(f"Opening {LAS_INPUT_PATH} for chunked reading...")
    with laspy.open(LAS_INPUT_PATH) as reader:
        header = reader.header
        total_points = header.point_count
        
        # Deconstruct the inverse transform into its basic components for serialization
        inv_trans_comps = (inv_transform.a, inv_transform.b, inv_transform.c, 
                           inv_transform.d, inv_transform.e, inv_transform.f)

        # For debug purposes, skip writing output and just ensure workers complete jobs
        futures = []
        submitted_points = 0

        print("Submitting trivial debug jobs to the cluster...")
        pbar_submit = tqdm(total=total_points, desc="Submitting jobs", unit="pts")
        for pts in reader.chunk_iterator(CHUNK_SIZE):
            future = client.submit(process_chunk_on_gpu, pts.x)
            futures.append(future)
            points_in_chunk = len(pts.x)
            submitted_points += points_in_chunk
            pbar_submit.update(points_in_chunk)
        pbar_submit.close()
        print(f"All jobs submitted. Total points queued: {submitted_points:,}")

        print("Gathering debug results as they finish...")
        processed_points = 0
        pbar_results = tqdm(total=submitted_points, desc="Completed points", unit="pts")
        for fut, chunk_count in as_completed(futures, with_results=True):
            if isinstance(chunk_count, Exception):
                print(f"A debug job failed: {chunk_count}")
                continue
            processed_points += int(chunk_count)
            pbar_results.update(int(chunk_count))
        pbar_results.close()
        print(f"All debug jobs finished. Total processed points: {processed_points:,}")

    print("\n✅ Debug run complete (no LAS output written).")

    # --- 7. Clean up ---
    print("Shutting down cluster.")
    client.close()
    cluster.close()
# %%
