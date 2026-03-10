import laspy
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from tqdm import tqdm
from pathlib import Path



# --- INPUT FILES ---
las_file = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_mask_tiles_multispectral.109.las")                      # Your Z-normalised LAS file
mask_tif = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.109.tif")                       # Labeled raster mask (same CRS)
voxel_size = 0.05                                      # in meters (e.g., 5 cm voxels)
output_csv = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/voxel_stats/z_voxel_stats_by_class.109.csv")

# --- LOAD LAS DATA ---
print("Loading LAS...")
with laspy.open(las_file) as f:
    las = f.read()
    x, y, z = las.x, las.y, las.z

print(f"Total points: {len(z)}")

# --- LOAD RASTER MASK TO GET CLASS LABELS ---
print("Loading mask...")
with rasterio.open(mask_tif) as src:
    mask = src.read(1)
    transform = src.transform
    mask_nodata = src.nodata

# --- MATCH POINTS TO MASK PIXELS TO ASSIGN CLASS ---
print("Assigning class labels from raster...")
rows, cols = rowcol(transform, x, y)
valid_mask = (
    (rows >= 0) & (rows < mask.shape[0]) &
    (cols >= 0) & (cols < mask.shape[1])
)

rows = rows[valid_mask]
cols = cols[valid_mask]
x = x[valid_mask]
y = y[valid_mask]
z = z[valid_mask]

labels = mask[rows, cols]
valid_label_mask = labels != mask_nodata

x, y, z, labels = x[valid_label_mask], y[valid_label_mask], z[valid_label_mask], labels[valid_label_mask]

print(f"Remaining points with valid labels: {len(z)}")

# --- CREATE VOXEL KEY AND ACCUMULATE STATS ---
print("Computing voxel stats...")
df = pd.DataFrame({'Z': z, 'class_label': labels})
df['voxel_z'] = (df['Z'] / voxel_size).astype(int)

# --- PER CLASS STATS ---
results = []

for class_id, group in tqdm(df.groupby('class_label')):
    z_min = group['Z'].min()
    z_mean = group['Z'].mean()
    z_max = group['Z'].max()
    voxel_count = group['voxel_z'].nunique()
    
    results.append({
        'class_label': int(class_id),
        'Z_min': round(z_min, 3),
        'Z_mean': round(z_mean, 3),
        'Z_max': round(z_max, 3),
        'voxel_count': int(voxel_count)
    })

# --- SAVE ---
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n✅ Done! Saved to {output_csv}")



#%%
##test 2
#!/usr/bin/env python3
"""
Compute voxel occupancy with X, Y, Z, Label, and PointCount from LAS + raster mask.
Voxel size controls the 3D aggregation (e.g., 0.05 m = 5 cm cubes).
"""

def compute_voxel_stats_xyz(
    las_path,
    mask_path,
    voxel_size=0.05,
    output_path="voxel_stats_xyz.csv",
    valid_labels=(65, 66, 67, 68),
    attr_index=0  # Z_norm
):
    print(f"📥 Reading LAS: {las_path}")
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    if "Z_norm" in las.point_format.dimension_names:
        z = np.asarray(las["Z_norm"], dtype=np.float64)
    else:
        extra_dims = {dim.name: dim for dim in las.point_format.extra_dimensions}
        if "Z_norm" in extra_dims:
            z = np.asarray(las["Z_norm"], dtype=np.float64)
        elif "attribute0" in extra_dims:
            z = np.asarray(las["attribute0"], dtype=np.float64)
        else:
            raise AttributeError(
                "Could not find 'Z_norm' extra dimension. Available extras: "
                f"{list(extra_dims.keys())}"
            )

    classif = np.asarray(las.classification, dtype=np.int16)

    print(f"🖼️ Reading mask: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        row, col = rowcol(transform, x, y)
        row, col = row.astype(int), col.astype(int)

        valid_mask = (
            (row >= 0) & (row < mask.shape[0]) &
            (col >= 0) & (col < mask.shape[1])
        )

        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        row, col = row[valid_mask], col[valid_mask]
        class_vals = mask[row, col]
        classif = classif[valid_mask]

        label_mask = np.isin(class_vals, valid_labels)
        x, y, z, class_vals, classif = (
            x[label_mask],
            y[label_mask],
            z[label_mask],
            class_vals[label_mask],
            classif[label_mask],
        )

    print(f"✅ {len(z):,} matched points to mask labels {valid_labels}")

    # Bin spatially by voxel size
    class_vals = np.asarray(class_vals, dtype=np.int32)

    ix = (x / voxel_size).astype(int)
    iy = (y / voxel_size).astype(int)
    iz = (z / voxel_size).astype(int)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "ix": ix,
        "iy": iy,
        "iz": iz,
        "Label": class_vals.astype(np.int32),
        "PointCount": 1
    })

    grouped = df.groupby(["ix", "iy", "iz", "Label"], as_index=False).agg({
        "x": "mean",
        "y": "mean",
        "z": "mean",
        "PointCount": "sum"
    })

    grouped = grouped.sort_values(["Label", "z"])

    # Save
    grouped.to_csv(output_path, index=False)
    print(f"\n📁 Saved: {output_path}")
    print(f"🧱 Total voxel bins: {len(grouped):,}")

    return grouped



#%%
# === Example usage ===
if __name__ == "__main__":
    las_file = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_mask_tiles_multispectral.26.las"
    mask_file = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif"
    out_csv = "/media/laura/laura_usb/code/fvc_structure/phase_4_3d_cnn_training/outputs/voxel_occupancy.csv"

    compute_voxel_stats_xyz(
        las_path=las_file,
        mask_path=mask_file,
        voxel_size=0.05,
        # attr_index=0,  # 'Z_norm'
        output_path=out_csv,
        # return_classification=True
    )

# %%
###test 3
#!/usr/bin/env python3
"""
Compute voxel occupancy (3D) + max Z_norm height raster (2D)
from LAS + raster mask.
"""
import laspy
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from tqdm import tqdm
from pathlib import Path


def compute_voxel_stats_xyz(
    las_path,
    mask_path,
    voxel_size=0.05,
    output_csv="voxel_occupancy.csv",
    output_tif="z_height_structure_max.tif",
    valid_labels=(65, 66, 67, 68)
):
    print(f"📥 Reading LAS: {las_path}")
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    # --- Extract Z_norm correctly ---
    if "Z_norm" in las.point_format.dimension_names:
        z = np.asarray(las["Z_norm"], dtype=np.float64)
    elif "attribute0" in las.point_format.dimension_names:
        z = np.asarray(las["attribute0"], dtype=np.float64)
    else:
        raise AttributeError(f"Missing 'Z_norm' or 'attribute0'. Found: {las.point_format.dimension_names}")

    print(f"🖼️ Reading mask: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
        height, width = mask.shape

        # Map X,Y to raster row,col
        row, col = rowcol(transform, x, y)
        row, col = row.astype(int), col.astype(int)

        valid_mask = (
            (row >= 0) & (row < height) &
            (col >= 0) & (col < width)
        )

        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        row, col = row[valid_mask], col[valid_mask]
        class_vals = mask[row, col]

        label_mask = np.isin(class_vals, valid_labels)
        x, y, z, class_vals = (
            x[label_mask],
            y[label_mask],
            z[label_mask],
            class_vals[label_mask],
        )
        row, col = row[label_mask], col[label_mask]

    print(f"✅ {len(z):,} matched points to mask labels {valid_labels}")

    # --- Compute voxel occupancy ---
    ix = (x / voxel_size).astype(int)
    iy = (y / voxel_size).astype(int)
    iz = (z / voxel_size).astype(int)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "ix": ix,
        "iy": iy,
        "iz": iz,
        "Label": class_vals.astype(np.int32),
        "PointCount": 1
    })

    grouped = (
        df.groupby(["ix", "iy", "iz", "Label"], as_index=False)
        .agg({"x": "mean", "y": "mean", "z": "mean", "PointCount": "sum"})
        .sort_values(["Label", "z"])
    )

    grouped.to_csv(output_csv, index=False)
    print(f"📁 Saved voxel stats CSV: {output_csv}")

    # --- Build 2D raster of max Z_norm ---
    z_raster = np.full((height, width), np.nan, dtype=np.float32)
    for r, c, zv in zip(row, col, z):
        if np.isnan(z_raster[r, c]) or zv > z_raster[r, c]:
            z_raster[r, c] = zv

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(z_raster, 1)

    print(f"🗺️ Saved max Z_norm raster: {output_tif}")
    print(f"🧱 Total voxel bins: {len(grouped):,}")

    return grouped, z_raster


# === Example Usage ===
if __name__ == "__main__":
    las_file = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
    mask_file = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.109.tif"
    out_csv = "/media/laura/laura_usb/code/fvc_structure/phase_4_3d_cnn_training/outputs/voxel_occupancy.109.csv"
    out_tif = "/media/laura/laura_usb/code/fvc_structure/phase_4_3d_cnn_training/outputs/z_height_structure_max.109.tif"

    compute_voxel_stats_xyz(
        las_path=las_file,
        mask_path=mask_file,
        voxel_size=0.05,
        output_csv=out_csv,
        output_tif=out_tif,
        valid_labels=(65, 66, 67, 68),
    )

# %%
