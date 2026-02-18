
#%%
import numpy as np

# Read a single NPZ file
data = np.load('/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/z_norm_struct_109_obj_polar.npz')
# Expected: ['coords', 'feats', 'semantic_label', 'instance_label']


# Access the arrays
coords = data['coords']              # (N, 3) int32 - voxel coordinates or integer voxel coordinates
feats = data['feats']                # (N, 5) float32 - [x, y, z, z_norm, intensity]
semantic_label = data['semantic_label']  # (N,) int32 - classID or class labels (0-4, or -100 for ignore)
instance_label = data['instance_label']  # (N,) int32 - tree IDs
offset = data['offset']              # (3,) float64 - coordinate offset for original position

print(f"Points: {coords.shape[0]}")
print(f"Features shape: {feats.shape}")
print(f"Unique classes: {np.unique(semantic_label)}")
print(f"Unique tree IDs: {np.unique(instance_label)}")

# %%
import numpy as np
import pandas as pd

# Read the NPZ list file
with open('/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/npz_list.txt', 'r') as f:
    npz_files = [line.strip() for line in f]

print(f"Found {len(npz_files)} NPZ files")
print("First 5 files:")
for f in npz_files[:5]:
    print(f"  {f}")

# Load and inspect each file with detailed DataFrame output
for npz_path in npz_files:
    data = np.load(npz_path)
    
    # 1. Check keys in file
    print(f"\n{'='*60}")
    print(f"Keys in file: {data.files}")
    
    coords = data['coords']              # (N, 3) -> Integer Voxel Indices
    feats = data['feats']                # (N, 5) -> [x, y, z, z_norm, intensity]
    semantic_label = data['semantic_label']  # (N,) -> Class ID (0-4, or -100)
    instance_label = data['instance_label']  # (N,) -> Tree ID
    offset = data['offset']              # (3,) -> coordinate offset
    
    # 2. Combine into DataFrame
    combined_data = np.hstack([
        coords, 
        feats, 
        semantic_label.reshape(-1, 1), 
        instance_label.reshape(-1, 1)
    ])
    
    col_names = [
        "vox_x", "vox_y", "vox_z",        # coords
        "loc_x", "loc_y", "loc_z", "z_norm", "intensity", # feats
        "class_id",                       # semantic_label
        "tree_id"                         # instance_label
    ]
    
    df = pd.DataFrame(combined_data, columns=col_names)
    
    # 3. Display statistics
    print(f"\nTotal Points: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check unique Tree IDs
    unique_trees = df["tree_id"].unique()
    print(f"\nUnique Tree IDs found: {len(unique_trees)}")
    print(f"Sample IDs: {unique_trees[:10]}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(df["class_id"].value_counts().sort_index())
    
# %%
