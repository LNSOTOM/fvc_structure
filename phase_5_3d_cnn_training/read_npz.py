
#%%
import numpy as np
import pandas as pd

# data = np.load("/home/nvidia/projects/voxel_scenes/z_norm_mask_tiles_multispectral.55.npz")
data = np.load("/home/nvidia/projects/fvc_structure/phase_5_3d_cnn_training/voxel_data_ready/z_norm_struct_109_obj_polar_no0.npz")

coords = data["coords"]      # (N, 3)
feats  = data["feats"]       # (N, C)
labels = data["labels"]      # (N,)

# Combine
df = pd.DataFrame(
    np.hstack([coords, feats, labels.reshape(-1, 1)]),
    columns=[f"coord_{i}" for i in range(coords.shape[1])] +
            [f"feat_{j}" for j in range(feats.shape[1])] +
            ["label"]
)

print(df.head())

# %%
##test 2
import numpy as np
import pandas as pd

# Load the file
file_path = "/home/nvidia/projects/fvc_structure/phase_5_3d_cnn_training/voxel_data_ready/z_norm_struct_109_obj_polar_no0.npz"
data = np.load(file_path)

# 1. Check what keys are actually in the file
print("Keys in file:", data.files)
# Expected: ['coords', 'feats', 'semantic_label', 'instance_label']

# 2. Extract Data
coords = data["coords"]          # (N, 3) -> Integer Voxel Indices
feats  = data["feats"]           # (N, 5) -> [x, y, z, z_norm, intensity]
sem_labels = data["semantic_label"] # (N,) -> Class ID (0-4)
inst_labels = data["instance_label"] # (N,) -> Tree ID

# 3. Combine into DataFrame
# We reshape labels to (N, 1) so they stack correctly
combined_data = np.hstack([
    coords, 
    feats, 
    sem_labels.reshape(-1, 1), 
    inst_labels.reshape(-1, 1)
])

# Define column names based on how we saved them
col_names = [
    "vox_x", "vox_y", "vox_z",        # coords
    "loc_x", "loc_y", "loc_z", "z_norm", "intensity", # feats
    "class_id",                       # semantic_label
    "tree_id"                         # instance_label
]

df = pd.DataFrame(combined_data, columns=col_names)

# 4. Inspect
print(f"\nTotal Points: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

# Check unique Tree IDs found
unique_trees = df["tree_id"].unique()
print(f"\nUnique Tree IDs found: {len(unique_trees)}")
print(f"Sample IDs: {unique_trees[:10]}")
# %%
