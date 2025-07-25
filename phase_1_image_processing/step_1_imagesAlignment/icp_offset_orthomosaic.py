
#%%
##OPTION 1
import json
from pathlib import Path
import rasterio
from rasterio.transform import Affine

# === INPUT FILES ===
orthomosaic_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned_p1/icp_metadata.json")

# === OUTPUT DIRECTORY AND FILE ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1")
aligned_dir.mkdir(parents=True, exist_ok=True)
aligned_orthomosaic_path = aligned_dir / (orthomosaic_path.stem + "_aligned.tif")

# === LOAD ICP TRANSFORM ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

offset_x = matrix[3]
offset_y = matrix[7]
offset_z = matrix[11]

print(f"📐 Applying offset: X = {offset_x:.3f} m, Y = {offset_y:.3f} m, Z = {offset_z:.3f} m")

# === READ AND ALIGN ORTHOMOSAIC ===
with rasterio.open(orthomosaic_path) as src:
    original_transform = src.transform
    shifted_transform = Affine(
        original_transform.a,
        original_transform.b,
        original_transform.c + offset_x,  # Shift X
        original_transform.d,
        original_transform.e,
        original_transform.f + offset_y   # Shift Y
    )

    profile = src.profile
    profile.update(transform=shifted_transform)

    with rasterio.open(aligned_orthomosaic_path, "w", **profile) as dst:
        for i in range(1, src.count + 1):
            dst.write(src.read(i), i)

print(f"✅ Aligned orthomosaic saved to:\n{aligned_orthomosaic_path}")



# %%
##OPTION 2 - full affine transformation extracted from the ICP metadata using GDAL Warp
#%%
from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned/icp_metadata.json")

# === CUSTOM OUTPUT DIRECTORY ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned")
aligned_dir.mkdir(parents=True, exist_ok=True)

# === OUTPUT FILE ===
aligned_orthomosaic_path = aligned_dir / (orthomosaic_path.stem + "_affine_aligned.tif")

# === Extract ICP Transform Matrix ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

# Extract 2D affine matrix (from top-left 2x2 + translation)
a = matrix[0]   # R11
b = matrix[1]   # R12
d = matrix[4]   # R21
e = matrix[5]   # R22
xoff = matrix[3]  # T1 (X translation)
yoff = matrix[7]  # T2 (Y translation)

# GDAL expects a 3x3 warp matrix (flattened row-major)
warp_matrix = [a, b, 0, d, e, 0, xoff, yoff, 1]

print("📐 Applying full affine ICP transform:")
print(f"Matrix:\n [{a:.6f}, {b:.6f}, {xoff:.3f}]\n [{d:.6f}, {e:.6f}, {yoff:.3f}]")

# === Apply GDAL warp with full affine matrix ===
gdal.Warp(
    str(aligned_orthomosaic_path),
    str(orthomosaic_path),
    transformerOptions=["SRC_METHOD=NO_GEOTRANSFORM"],
    warpOptions=["INIT_DEST=NO_DATA"],
    options=["NUM_THREADS=ALL_CPUS"],
    transformer={'Geotransform': warp_matrix}
)

print(f"✅ Orthomosaic fully aligned using full affine transform:\n{aligned_orthomosaic_path}")

#%%
################
from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/home/laura/Documents/uas_data/Calperum/site4_DD0011/othomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned_p1/icp_metadata.json")

# === OUTPUT DIRECTORY AND FILE ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1")
aligned_dir.mkdir(parents=True, exist_ok=True)
aligned_orthomosaic_path = aligned_dir / f"{orthomosaic_path.stem}_affine_aligned.tif"

# === LOAD ICP TRANSFORM ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

# === Extract affine components ===
a, b = matrix[0], matrix[1]
d, e = matrix[4], matrix[5]
xoff, yoff = matrix[3], matrix[7]

print("📐 Applying affine ICP transform:")
print(f"Matrix:\n [{a:.6f}, {b:.6f}, {xoff:.3f}]\n [{d:.6f}, {e:.6f}, {yoff:.3f}]")

# === Open source orthomosaic ===
src_ds = gdal.Open(str(orthomosaic_path), gdal.GA_ReadOnly)
if src_ds is None:
    raise RuntimeError(f"❌ Could not open orthomosaic: {orthomosaic_path}")

old_gt = src_ds.GetGeoTransform()
proj = src_ds.GetProjection()
cols, rows = src_ds.RasterXSize, src_ds.RasterYSize
bands = src_ds.RasterCount
dtype = src_ds.GetRasterBand(1).DataType

# === Compute new affine geotransform ===
new_gt = [
    old_gt[0] * a + old_gt[1] * d + xoff,
    old_gt[1] * a + old_gt[2] * d,
    old_gt[1] * b + old_gt[2] * e,
    old_gt[3] * a + old_gt[4] * d + yoff,
    old_gt[4] * b + old_gt[5] * e,
    old_gt[4] * b + old_gt[5] * e
]

# === Create destination dataset ===
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(
    str(aligned_orthomosaic_path),
    cols,
    rows,
    bands,
    dtype,
    options=["TILED=YES", "COMPRESS=LZW"]
)

if dst_ds is None:
    raise RuntimeError(f"❌ Failed to create output: {aligned_orthomosaic_path}")

dst_ds.SetGeoTransform(new_gt)
dst_ds.SetProjection(proj)

# === Copy each band safely ===
for i in range(1, bands + 1):
    src_band = src_ds.GetRasterBand(i)
    data = src_band.ReadAsArray()

    if data is None or data.ndim != 2:
        raise ValueError(f"❌ Invalid data in band {i}, shape: {data.shape if data is not None else 'None'}")

    dst_band = dst_ds.GetRasterBand(i)
    dst_band.WriteArray(data)

    # Preserve NoData value if available
    nodata = src_band.GetNoDataValue()
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)

# === Finalize and cleanup ===
dst_ds.FlushCache()
del dst_ds
del src_ds

print(f"✅ Saved affine-aligned orthomosaic to:\n{aligned_orthomosaic_path}")

# %%
###############tetsing
from osgeo import gdal

path = "/home/laura/Documents/uas_data/Calperum/site4_DD0011/othomosaic/20220517_SASMDD0011_p1_ortho_01.tif"
ds = gdal.Open(path)
if ds is None:
    print("❌ Could not open image")

print("Bands:", ds.RasterCount)
for i in range(1, ds.RasterCount + 1):
    band = ds.GetRasterBand(i)
    arr = band.ReadAsArray()
    print(f"Band {i} shape:", arr.shape if arr is not None else "None")

# %%
#### proof
from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/home/laura/Documents/uas_data/Calperum/site4_DD0011/othomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned_p1/icp_metadata.json")

# === OUTPUT DIRECTORY AND FILE ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1")
aligned_dir.mkdir(parents=True, exist_ok=True)
aligned_orthomosaic_path = aligned_dir / f"{orthomosaic_path.stem}_affine_aligned.tif"

# === LOAD ICP TRANSFORM ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

# === Extract 2D affine matrix components ===
a, b = matrix[0], matrix[1]
d, e = matrix[4], matrix[5]
xoff, yoff = matrix[3], matrix[7]

print("📐 Applying 2D affine ICP transform:")
print(f"Matrix:\n [{a:.6f}, {b:.6f}, {xoff:.3f}]\n [{d:.6f}, {e:.6f}, {yoff:.3f}]")

# === Open source orthomosaic ===
src_ds = gdal.Open(str(orthomosaic_path), gdal.GA_ReadOnly)
if src_ds is None:
    raise RuntimeError(f"❌ Could not open orthomosaic: {orthomosaic_path}")

old_gt = src_ds.GetGeoTransform()
proj = src_ds.GetProjection()
cols, rows = src_ds.RasterXSize, src_ds.RasterYSize
dtype = src_ds.GetRasterBand(1).DataType

# === Force copy only first 3 bands (RGB), skip alpha ===
bands = 3
if src_ds.RasterCount < bands:
    raise RuntimeError(f"❌ Source image has fewer than 3 bands (found {src_ds.RasterCount}). Cannot extract RGB.")

# === Construct affine transform ===
pixel_size_x = old_gt[1]
pixel_size_y = old_gt[5]

new_gt = [
    old_gt[0] + xoff,
    a * pixel_size_x,
    b * pixel_size_x,
    old_gt[3] + yoff,
    d * pixel_size_y,
    e * pixel_size_y
]

# === Create destination dataset ===
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(
    str(aligned_orthomosaic_path),
    cols,
    rows,
    bands,
    dtype,
    options=["TILED=YES", "COMPRESS=LZW", "ALPHA=NO"]
)

if dst_ds is None:
    raise RuntimeError(f"❌ Failed to create output: {aligned_orthomosaic_path}")

dst_ds.SetGeoTransform(new_gt)
dst_ds.SetProjection(proj)

# === Copy RGB bands (1 to 3 only) ===
for i in range(1, 4):  # Band 1 to 3
    src_band = src_ds.GetRasterBand(i)
    data = src_band.ReadAsArray()
    if data is None or data.ndim != 2:
        raise ValueError(f"❌ Invalid data in band {i}, shape: {data.shape if data is not None else 'None'}")

    dst_band = dst_ds.GetRasterBand(i)
    dst_band.WriteArray(data)

    nodata = src_band.GetNoDataValue()
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)

# === Finalize ===
dst_ds.FlushCache()
del dst_ds
del src_ds

print(f"✅ Saved RGB-only affine-aligned orthomosaic to:\n{aligned_orthomosaic_path}")

# %%
#  gdalinfo "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1/20220517_SASMDD0011_p1_ortho_01_affine_aligned_stacked_rgb.tif"

from osgeo import gdal

ds = gdal.Open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1/20220517_SASMDD0011_p1_ortho_01_affine_aligned_stacked_rgb.tif")
print("Band count:", ds.RasterCount)
for i in range(1, ds.RasterCount + 1):
    band = ds.GetRasterBand(i)
    print(f"Band {i} - NoData: {band.GetNoDataValue()}")


# %%
import rasterio
import os
from pathlib import Path
import numpy as np

# === Input orthomosaic directory and output target ===
input_folder = '/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned_p1'
output_folder_3b = '/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_rgb_only'

# Ensure output folder exists
os.makedirs(output_folder_3b, exist_ok=True)

# Process each .tif file in the input folder
for input_filename in os.listdir(input_folder):
    if not input_filename.endswith('.tif'):
        continue

    input_path = os.path.join(input_folder, input_filename)
    output_path = os.path.join(output_folder_3b, f"{Path(input_filename).stem}_rgb_only.tif")

    try:
        with rasterio.open(input_path) as src:
            if src.count < 3:
                print(f"Skipping {input_filename}: fewer than 3 bands.")
                continue

            # Read first 3 bands (RGB)
            data_rgb = [src.read(i) for i in range(1, 4)]

            # Clip to uint8 range just in case
            data_rgb = [np.clip(band, 0, 255).astype(np.uint8) for band in data_rgb]

            # Prepare new profile
            profile = src.profile.copy()
            profile.update({
                'count': 3,
                'dtype': 'uint8',
                'compress': 'LZW'
            })

        # Write new RGB-only file
        with rasterio.open(output_path, 'w', **profile) as dst:
            for idx, band in enumerate(data_rgb, start=1):
                dst.write(band, idx)

        print(f"✅ Saved RGB-only file: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {input_filename}: {e}")

print("🎉 All eligible orthomosaics processed to RGB-only.")

# %%
