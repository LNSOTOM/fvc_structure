
# pip install open3d matplotlib laspy
import numpy as np
import laspy
from pathlib import Path
import open3d as o3d
from matplotlib import cm

IN_LAZ = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG_hdbscan_Znorm.laz")
OUT_PLY = IN_LAZ.with_suffix(".ply")

COLOR_MODE = "cluster"   # "cluster" or "znorm"
USE_Z_NORM_AS_GEOMETRY = False  # True -> use Z_norm as z coords, else absolute z
VOXEL_DOWNSAMPLE = 0.05  # meters; set 0/None to disable
NOISE_COLOR = np.array([0.6, 0.6, 0.6])  # grey for noise
CMAP = "viridis"  # used when COLOR_MODE="znorm"

las = laspy.read(str(IN_LAZ))
x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)
z_abs = np.asarray(las.z, dtype=np.float32)

# choose geometry Z
if "Z_norm" in las.point_format.dimension_names:
    z_norm = np.asarray(las["Z_norm"], dtype=np.float32)
elif "HAG" in las.point_format.dimension_names:
    z_norm = np.asarray(las["HAG"], dtype=np.float32)
else:
    z_norm = None

z = z_norm if (USE_Z_NORM_AS_GEOMETRY and z_norm is not None) else z_abs
pts = np.c_[x, y, z]

# colors
if COLOR_MODE == "cluster" and "cluster_id" in las.point_format.dimension_names:
    labels = np.asarray(las["cluster_id"], dtype=np.int32)
    k = labels.max()
    # palette: noise=grey, clusters colored
    if k > 0:
        base = cm.tab20(np.linspace(0, 1, min(20, k)))[:, :3]
        if k > 20:
            extra = cm.hsv(np.linspace(0, 1, k - 20 + 1))[:-1, :3]
            palette = np.vstack([base, extra])
        else:
            palette = base[:k]
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        colors[labels == 0] = NOISE_COLOR
        pos = labels > 0
        colors[pos] = palette[labels[pos] - 1]
    else:
        colors = np.tile(NOISE_COLOR, (len(labels), 1))
else:
    # color by Z_norm (continuous)
    if z_norm is None:
        raise RuntimeError("No Z_norm/HAG found for COLOR_MODE='znorm'.")
    lo, hi = np.percentile(z_norm, [1, 99.9])
    t = np.clip((z_norm - lo) / max(hi - lo, 1e-9), 0, 1)
    colors = cm.get_cmap(CMAP)(t)[:, :3].astype(np.float32)

# build cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# optional downsample
if VOXEL_DOWNSAMPLE and VOXEL_DOWNSAMPLE > 0:
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_DOWNSAMPLE)

# save + (optionally) display
o3d.io.write_point_cloud(str(OUT_PLY), pcd, write_ascii=False)
print(f"✅ Wrote PLY: {OUT_PLY}")
try:
    o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print(f"(viewer skipped: {e})")

# %%
##test 2
# pip install plyfile matplotlib laspy open3d  (open3d optional for viewer)
from pathlib import Path
import numpy as np, laspy
from plyfile import PlyData, PlyElement
from matplotlib import cm

IN_LAZ = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG_hdbscan_Znorm_COLOR.laz")
OUT_PLY = IN_LAZ.with_suffix(".ply")
COLOR_BY = "cluster"   # "cluster" or "znorm"
NOISE_RGB = (160,160,160)

las = laspy.read(str(IN_LAZ))

# geometry
x = np.asarray(las.x, np.float32)
y = np.asarray(las.y, np.float32)
z = np.asarray(las.z, np.float32)

# cluster ids (0 = noise)
cid = las["cluster_id"].astype(np.int32) if "cluster_id" in las.point_format.dimension_names else np.zeros(len(x), np.int32)

# colors
if COLOR_BY == "cluster":
    k = cid.max()
    if k > 0:
        base = cm.tab20(np.linspace(0,1,min(20,k)))[:,:3]
        extra = cm.hsv(np.linspace(0,1,max(0, k-20)+1))[:-1,:3] if k>20 else np.empty((0,3))
        pal = np.vstack([base, extra])
        rgb8 = np.zeros((len(x),3), np.uint8)
        rgb8[cid==0] = NOISE_RGB
        pos = cid>0
        rgb8[pos] = (pal[cid[pos]-1]*255 + 0.5).astype(np.uint8)
    else:
        rgb8 = np.tile(np.array(NOISE_RGB,np.uint8),(len(x),1))
else:
    # color by Z_norm/HAG
    if "Z_norm" in las.point_format.dimension_names:
        zn = np.asarray(las["Z_norm"], np.float32)
    elif "HAG" in las.point_format.dimension_names:
        zn = np.asarray(las["HAG"], np.float32)
    else:
        raise RuntimeError("No Z_norm/HAG to color by.")
    lo, hi = np.percentile(zn, [1, 99.9])
    t = np.clip((zn - lo)/max(hi-lo,1e-9), 0, 1)
    rgb8 = (cm.viridis(t)[:,:3]*255 + 0.5).astype(np.uint8)

# build PLY vertex array
vertex = np.empty(len(x), dtype=[
    ("x","f4"),("y","f4"),("z","f4"),
    ("red","u1"),("green","u1"),("blue","u1"),
    ("cluster_id","i4")
])
vertex["x"],vertex["y"],vertex["z"] = x,y,z
vertex["red"],vertex["green"],vertex["blue"] = rgb8[:,0],rgb8[:,1],rgb8[:,2]
vertex["cluster_id"] = cid

ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False)
ply.write(str(OUT_PLY))
print(f"✅ Wrote PLY: {OUT_PLY}")

# (optional) quick viewer
try:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(OUT_PLY))
    o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print(f"(viewer skipped: {e})")

# %%
