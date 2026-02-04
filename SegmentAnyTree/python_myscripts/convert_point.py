
#%%
import numpy as np
import pandas as pd
from plyfile import PlyData
from pathlib import Path

ply_path = Path("/data/calperumResearch/segmentanytree/output/PUT_YOUR_INSTANCE_OUTPUT.ply")

# Choose voxel size (meters). 0.05 = 5 cm
voxel = 0.05

# Set this to the field name you found in the PLY header scan:
INSTANCE_FIELD = "pred_instance"   # e.g. "instance", "instance_id", etc.

ply = PlyData.read(str(ply_path))
v = ply["vertex"].data
fields = v.dtype.names
print("Fields:", fields)

# Coordinates
xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)

# Instance ids
ins = np.asarray(v[INSTANCE_FIELD]).astype(np.int64)

# Convert to voxel indices (integer grid)
origin = xyz.min(axis=0)  # local origin for indexing; save if you need consistency across files
ijk = np.floor((xyz - origin) / voxel).astype(np.int32)

# Create a key for grouping voxels
# (pack i,j,k into one 64-bit key)
key = (ijk[:, 0].astype(np.int64) << 42) | (ijk[:, 1].astype(np.int64) << 21) | ijk[:, 2].astype(np.int64)

df = pd.DataFrame({
    "key": key,
    "i": ijk[:, 0],
    "j": ijk[:, 1],
    "k": ijk[:, 2],
    "instance_id": ins
})

# Majority vote per voxel
# If you want "any point in voxel belongs to instance" use mode.
voxel_mode = (
    df.groupby("key", sort=False)["instance_id"]
      .agg(lambda s: s.value_counts().idxmax())
      .reset_index()
)

# Restore ijk for each key (take first occurrence)
ijk_first = df.groupby("key", sort=False)[["i","j","k"]].first().reset_index()
vox = voxel_mode.merge(ijk_first, on="key", how="left")

# Add voxel center coordinates
vox["x"] = origin[0] + (vox["i"] + 0.5) * voxel
vox["y"] = origin[1] + (vox["j"] + 0.5) * voxel
vox["z"] = origin[2] + (vox["k"] + 0.5) * voxel

print(vox.head())
print("Num voxels:", len(vox))

# Save sparse voxel labels
out_parquet = ply_path.with_suffix("").as_posix() + f"_vox{voxel:.2f}m.parquet"
vox.to_parquet(out_parquet, index=False)
print("Saved:", out_parquet)

# %%
