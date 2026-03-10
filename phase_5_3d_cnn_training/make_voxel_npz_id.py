#%%
import laspy
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
VOXEL_SIZE = 0.10   # 10 cm (Matches your Cartesian grid size from Step 1)
OUT_DIR = Path("voxel_data_ready")
OUT_DIR.mkdir(exist_ok=True)

# Mapping LAS classes to Model Classes (0, 1, 2...)
# Adjust this based on what you want the model to learn.
# Example: 64->0, 65->1, 66->2, 67->3, 68->4
CLASS_REMAP = {
    64: 0,  # Forb
    65: 1,  # Woody Debris
    66: 2,  # Hummock
    67: 3,  # Shrub
    68: 4,  # Mallee
    0: -100 # Ignore class (standard in PyTorch)
}

def process_las(path):
    print(f"Processing: {path.name}")
    las = laspy.read(path)

    # 1. Coordinates (Shift to local to prevent huge integers)
    # Deep learning models need coordinates near (0,0,0)
    raw_xyz = np.vstack([las.x, las.y, las.z]).T
    offset = raw_xyz.min(axis=0)
    xyz_local = raw_xyz - offset
    
    # Voxelize (Discrete Coordinates for Sparse Conv)
    coords = np.floor(xyz_local / VOXEL_SIZE).astype(np.int32)

    # 2. Features (Input to the Neural Net)
    # Usually: [R, G, B, Z_norm, Intensity]
    # Since we might not have RGB on this file, we use: [x_norm, y_norm, z_norm, intensity]
    
    # Intensity (Normalize to 0-1)
    intensity = las.intensity.astype(np.float32)
    if intensity.max() > 0:
        intensity /= 65535.0  # Assuming 16-bit
        
    # Z_norm (Check if exists, else calculate)
    if "Z_norm" in las.point_format.dimension_names:
        z_norm = las["Z_norm"].astype(np.float32)
    else:
        # Fallback: approximate relative height
        z_norm = (raw_xyz[:, 2] - raw_xyz[:, 2].min()).astype(np.float32)

    # Stack Features
    # Feature vector shape: (N, 5) -> [x_local, y_local, z_local, z_norm, intensity]
    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)

    # 3. Semantic Labels (Remapped)
    raw_labels = las.classification.astype(np.int32)
    sem_labels = np.full_like(raw_labels, -100) # Default to ignore
    
    for las_cls, model_cls in CLASS_REMAP.items():
        sem_labels[raw_labels == las_cls] = model_cls

    # 4. Instance Labels (treeID) -- CRITICAL FOR INSTANCE SEGMENTATION
    if "treeID" in las.point_format.dimension_names:
        inst_labels = las["treeID"].astype(np.int32)
    else:
        print(f"⚠️ Warning: 'treeID' not found in {path.name}. Instances will be 0.")
        inst_labels = np.zeros(len(raw_labels), dtype=np.int32)

    # 5. Save compressed NPZ
    out_npz = OUT_DIR / (path.stem + ".npz")
    np.savez_compressed(
        out_npz,
        coords=coords,       # Int32: Voxel indices (N, 3)
        feats=feats,         # Float32: Input features (N, C)
        semantic_label=sem_labels, # Int32: Class ID (N,)
        instance_label=inst_labels # Int32: Tree ID (N,)
    )

    return out_npz

def main():
    # Point this to the folder containing your files with 'treeID'
    input_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_obj_polar_id")
    
    # Get all LAS files
    files = sorted(input_dir.glob("*.las"))
    
    metadata_list = []
    
    for p in files:
        try:
            out_file = process_las(p)
            metadata_list.append(str(out_file))
        except Exception as e:
            print(f"❌ Error processing {p.name}: {e}")

    # Write dataset list (train.txt / val.txt structure)
    with open("dataset_list.txt", "w") as f:
        f.write("\n".join(metadata_list))
    
    print(f"✅ Done. Processed {len(metadata_list)} files.")

if __name__ == "__main__":
    main()
# %%
##tets 2
import laspy
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# 1. Output Directory (Where .npz tensors will go)
OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/voxel_tensors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Input Directory (Where your LAS files with 'treeID' are located)
INPUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id")

# 3. Voxel Size (Must match what you plan to use in training config, e.g. 5cm or 10cm)
VOXEL_SIZE = 0.10

# 4. Class Remapping
# Maps your LAS classes (64-68) to continuous integers (0-4) for the AI
CLASS_REMAP = {
    64: 0,  # Forb
    65: 1,  # Woody Debris
    66: 2,  # Hummock
    67: 3,  # Shrub
    68: 4,  # Mallee
    0: -100 # Ignore class (Background/Unclassified)
}

def process_las(path):
    print(f"Processing: {path.name}")
    las = laspy.read(path)

    # --- 1. Coordinates & Voxelization ---
    # We shift coordinates to local origin (min_x, min_y, min_z) 
    # to avoid floating point errors with large UTM coordinates.
    raw_xyz = np.vstack([las.x, las.y, las.z]).T
    offset = raw_xyz.min(axis=0)
    xyz_local = raw_xyz - offset
    
    # Discretize to integers
    coords = np.floor(xyz_local / VOXEL_SIZE).astype(np.int32)

    # --- 2. Input Features ---
    # Standard features: [Center_Normalized_Coordinates, Z_norm, Intensity]
    # Normalize Intensity (assuming 16-bit source)
    intensity = las.intensity.astype(np.float32)
    if intensity.max() > 0:
        intensity /= 65535.0
        
    # Get Z_norm (Relative Height)
    if "Z_norm" in las.point_format.dimension_names:
        z_norm = las["Z_norm"].astype(np.float32)
    else:
        # Fallback if missing
        z_norm = (raw_xyz[:, 2] - raw_xyz[:, 2].min()).astype(np.float32)

    # Stack them: (N, 5)
    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)

    # --- 3. Semantic Labels ---
    raw_labels = las.classification.astype(np.int32)
    sem_labels = np.full_like(raw_labels, -100) # Initialize with "Ignore" value
    
    for las_cls, model_cls in CLASS_REMAP.items():
        sem_labels[raw_labels == las_cls] = model_cls

    # --- 4. Instance Labels (Tree ID) ---
    # This is required for "Group" prediction in ForestFormer3D / PointGroup
    if "treeID" in las.point_format.dimension_names:
        inst_labels = las["treeID"].astype(np.int32)
    else:
        print(f"⚠️ Warning: 'treeID' dimension not found in {path.name}. Setting instances to 0.")
        inst_labels = np.zeros(len(raw_labels), dtype=np.int32)

    # --- 5. Save ---
    out_npz = OUT_DIR / (path.stem + ".npz")
    np.savez_compressed(
        out_npz,
        coords=coords,       
        feats=feats,         
        semantic_label=sem_labels, 
        instance_label=inst_labels 
    )

    return out_npz

def main():
    if not INPUT_DIR.exists():
        print(f"❌ Error: Input directory does not exist: {INPUT_DIR}")
        return

    files = sorted(INPUT_DIR.glob("*.las"))
    if not files:
        print(f"❌ Error: No .las files found in {INPUT_DIR}")
        return

    metadata_list = []
    
    print(f"🚀 Starting generation of {len(files)} tensors...")
    print(f"📂 Output: {OUT_DIR}")

    for p in files:
        try:
            out_file = process_las(p)
            # Store absolute path for the metadata list
            metadata_list.append(str(out_file.absolute()))
        except Exception as e:
            print(f"❌ Failed {p.name}: {e}")

    # Create the train/val list file (often needed by data loaders)
    list_file = OUT_DIR / "dataset_file_list.txt"
    with open(list_file, "w") as f:
        f.write("\n".join(metadata_list))
    
    print(f"✅ Done! Processed {len(metadata_list)} files.")
    print(f"📄 File list saved to: {list_file}")

if __name__ == "__main__":
    main()