
#%%
import pandas as pd
from pyntcloud import PyntCloud

# Read a single PLY file
cloud = PyntCloud.from_file('/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply/z_norm_struct_109_obj_polar.ply')

df = cloud.points

# Access the columns
x = df['x'].values           # float - absolute X coordinate
y = df['y'].values           # float - absolute Y coordinate  
z = df['z'].values           # float - absolute Z coordinate
semantic = df['semantic_seg'].values  # int32 - class labels (0-4, or -100)
tree_id = df['treeID'].values        # int32 - instance IDs
z_norm = df['z_norm'].values         # float32 - height above ground
intensity = df['intensity'].values   # float32 - normalized intensity [0-1]

print(f"Points: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
# %%
