
#%%
import numpy as np
import laspy
import rasterio
from pathlib import Path
from scipy.ndimage import map_coordinates

las_in  = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# Use the aligned DSM as reference grid
grid    = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05_aligned_to_lidar.tif")
# Use inverted flow to align LAS points to the aligned DSM
flow_u  = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_inverted.npy")
flow_v  = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_inverted.npy")

# --- Raster grid info (for world<->pixel transforms) ---
with rasterio.open(grid) as src:
    transform = src.transform
    H, W = src.height, src.width

# Sanity: flow arrays should match raster dimensions
if flow_u.shape != (H, W) or flow_v.shape != (H, W):
    raise ValueError(f"Flow shape {flow_u.shape} / {flow_v.shape} does not match raster size {(H, W)}")

inv = ~transform       # affine inverse for (x,y) -> (row,col)
eps = 1e-3             # small epsilon to keep coords inside bounds

out = las_in.with_name(las_in.stem + "_gefolki_coreg.las")
total_in, total_written = 0, 0

# Stream through the LAS in chunks to avoid huge RAM usage
with laspy.open(las_in) as reader, laspy.open(out, mode="w", header=reader.header) as writer:
    total_in = total_written = 0
    for pts in reader.chunk_iterator(5_000_000):
        total_in += len(pts)
        x, y = pts.x, pts.y
        row, col = inv * (x, y)

        valid = (col >= 0) & (col < W-1) & (row >= 0) & (row < H-1)
        if not np.any(valid):
            continue

        rowc = np.clip(row[valid], 0, H-1-eps)
        colc = np.clip(col[valid], 0, W-1-eps)
        coords = np.vstack((rowc, colc))

        dx_pix = map_coordinates(flow_u, coords, order=1, mode="nearest")
        dy_pix = map_coordinates(flow_v, coords, order=1, mode="nearest")

        dx = dx_pix * transform.a     # meters
        dy = dy_pix * transform.e     # meters (note: e is negative)

        sub = pts[valid]
        xv, yv = sub.x.copy(), sub.y.copy()
        xv += dx;  yv += dy
        sub.x = xv; sub.y = yv
        writer.write_points(sub)
        total_written += len(sub)

print(f"✅ Wrote {out}")
print(f"   kept {total_written} / {total_in} points ({total_written/total_in:.1%})")

# ✅ Wrote /media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_gefolki.las
#    kept 278194578 / 285786198 points (97.3%) that overlapped the DSM

#%%
import numpy as np, laspy, rasterio
from pathlib import Path

las_in = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
grid   = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05_aligned_to_lidar.tif")
u = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_inverted.npy")
v = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_inverted.npy")

with rasterio.open(grid) as src:
    a, e = src.transform.a, src.transform.e

# robust constant translation
TX = np.median(u[np.isfinite(u)]) * a
TY = np.median(v[np.isfinite(v)]) * e

out = las_in.with_name(las_in.stem + "_gefolki_coreg.las")
with laspy.open(las_in) as reader, laspy.open(out, mode="w", header=reader.header) as writer:
    for pts in reader.chunk_iterator(5_000_000):
        pts.x = pts.x + TX
        pts.y = pts.y + TY
        # Z unchanged
        writer.write_points(pts)

print(f"✅ Wrote {out} with constant shift TX={TX:.3f} m, TY={TY:.3f} m")


# %%
##test
import numpy as np, laspy, rasterio
from pathlib import Path

las_in = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
grid   = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05_aligned_to_lidar.tif")
u = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_inverted.npy")
v = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_inverted.npy")

with rasterio.open(grid) as src:
    T = src.transform
    a, e = T.a, T.e            # a>0 pixel width; e<0 pixel height (north-up rasters)
    H, W = src.height, src.width
    dsm = src.read(1).astype(np.float32)
    nd  = src.nodata if src.nodata is not None else -9999.0
    mask = (dsm != nd)

# 1) sanity: flow must match grid
assert u.shape == (H, W) and v.shape == (H, W), f"flow {u.shape}/{v.shape} vs grid {(H,W)}"

# 2) robust constant translation from valid pixels
valid = np.isfinite(u) & np.isfinite(v) & mask
TX = np.median(u[valid]) * a          # meters (X)
TY = np.median(v[valid]) * e          # meters (Y)  e is negative → correct sign

print(f"Use these in CloudCompare → Apply transformation → Translation:")
print(f"X = {TX:.4f}  Y = {TY:.4f}  Z = 0.0   (all rotations 0)")

# 3) apply same constant shift to LAS (chunked)
out = las_in.with_name(las_in.stem + "_gefolki_coreg.las")
with laspy.open(las_in) as reader, laspy.open(out, mode="w", header=reader.header) as writer:
    for pts in reader.chunk_iterator(5_000_000):
        pts.x = pts.x + TX
        pts.y = pts.y + TY
        writer.write_points(pts)      # Z unchanged

print(f"✅ Wrote {out}  (TX={TX:.3f} m, TY={TY:.3f} m)")

# %%
