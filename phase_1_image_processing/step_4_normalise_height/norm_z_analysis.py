
#%%
from pathlib import Path
import numpy as np
import laspy
from copy import deepcopy

in_las  = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las")
out_las = in_las.with_name(in_las.stem + "_veg_only.las")

las = laspy.read(in_las)

# Boolean mask (e.g., HAG >= 0.10 m)
hag = las.Z_norm
veg_mask = hag >= 0.10
print("Vegetation pts:", int(veg_mask.sum()), "/", len(hag))

# ---- build a header compatible with the source ----
try:
    # If deepcopy works in your env, it preserves everything (scales/offsets/VLR placeholders)
    hdr = deepcopy(las.header)
except Exception:
    # Manual rebuild fallback
    hdr = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    hdr.scales  = las.header.scales
    hdr.offsets = las.header.offsets
    # copy CRS if available
    try:
        crs = las.header.parse_crs()
        if crs:
            hdr.add_crs(crs)
    except Exception:
        pass

# ---- create output LAS and assign masked points in one shot ----
las_veg = laspy.LasData(hdr)
las_veg.points = las.points[veg_mask]   # << no broadcasting; resizes correctly

# (optional) carry over VLRs/EVLRs (e.g., GeoTIFF keys for CRS)
try:
    las_veg.vlrs  = list(las.vlrs)
    las_veg.evlrs = list(las.evlrs)
except Exception:
    pass

# (optional) enforce a vegetation class code if you want
# las_veg.classification[:] = 3  # ASPRS 3=low veg, 4=medium, 5=high (choose as needed)

las_veg.write(out_las)
print("Saved vegetation-only LAS:", out_las)

# Quick QA
chk = laspy.read(out_las)
print("Out point count:", chk.header.point_count)
print("Z_norm stats:", float(chk.Z_norm.min()), float(np.median(chk.Z_norm)), float(chk.Z_norm.max()))



# %%
