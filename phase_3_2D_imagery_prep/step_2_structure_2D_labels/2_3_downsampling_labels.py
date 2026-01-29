
#%%
import os
import subprocess
from osgeo import gdal
import rasterio

def gdal_downsample_structure(
    input_raster,
    output_raster,
    target_resolution,
    crs,
    grid_feature=None,
    nodata=255,
    keep_dims=True,
    remap_values=None,
):
    """
    Downsample a structure classification raster using GDAL while preserving the same extent and shape.
    Works for integer-class rasters (0–4 or 64–68) with nodata=255.
    """
    output_dir = os.path.dirname(output_raster)
    os.makedirs(output_dir, exist_ok=True)

    # --- Optional clipping ---
    if grid_feature is not None:
        geom = grid_feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()
        projWin = [minX, maxY, maxX, minY]
        clipped_raster = os.path.join(output_dir, "clipped_temp.tif")
        gdal.Translate(
            clipped_raster,
            input_raster,
            options=gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Byte, projWin=projWin),
        )
        input_for_warp = clipped_raster
    else:
        input_for_warp = input_raster

    # --- Get raster properties ---
    ds = gdal.Open(input_for_warp)
    if ds is None:
        raise FileNotFoundError(f"❌ Could not open {input_for_warp}")

    width, height = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    orig_left, orig_top = gt[0], gt[3]
    orig_right = orig_left + width * gt[1]
    orig_bottom = orig_top + height * gt[5]
    orig_extent = (orig_left, orig_bottom, orig_right, orig_top)
    print(f"📐 Input extent: {orig_extent}")

    # --- Keep same extent as input ---
    new_extent = orig_extent

    # --- gdalwarp command ---
    gdal_command = [
        "gdalwarp",
        "-overwrite",
        "-q",
        "-s_srs", crs,                # ensure input CRS is known
        "-t_srs", crs,
        "-tr", str(target_resolution), str(target_resolution),
        "-r", "nearest",
        # IMPORTANT: GDAL expects top (maxY) before bottom (minY)
        "-te", str(new_extent[0]), str(new_extent[3]), str(new_extent[2]), str(new_extent[1]),
        "-ot", "Byte",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=3072",
        "-co", "BLOCKYSIZE=3072",
        "-dstnodata", str(nodata),
        input_for_warp,
        output_raster,
    ]

    # --- Run warp with error capture ---
    try:
        result = subprocess.run(gdal_command, check=True, capture_output=True, text=True)
        print(f"✅ Downsampled raster saved to: {output_raster}")
    except subprocess.CalledProcessError as e:
        print("❌ gdalwarp failed")
        print("STDERR:\n", e.stderr)
        raise

    # --- Remap pixel values ---
    if remap_values:
        with rasterio.open(output_raster, "r+") as dst:
            band = dst.read(1)
            for src_val, dst_val in remap_values.items():
                band[band == src_val] = dst_val
            dst.write(band, 1)
        print(f"Applied remap: {remap_values}")

    if grid_feature is not None and os.path.exists(clipped_raster):
        os.remove(clipped_raster)
        print("🧹 Removed temporary clip file.")

# --- Configuration ---
input_raster = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.26.tif"
output_raster = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif"

target_resolution = 0.05
crs = "EPSG:7854"
grid_feature = None
value_remap = {0:64, 1:65, 2:66, 3:67, 4:68}


# --- Run downsample for one file ---
gdal_downsample_structure(
    input_raster,
    output_raster,
    target_resolution,
    crs,
    grid_feature=grid_feature,
    keep_dims=True,
    remap_values=value_remap,
)

#%%
### 🔍 Validate output raster
with rasterio.open(output_raster) as src:
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"CRS: {src.crs}")
    print(f"NoData: {src.nodata}")
    unique_vals = sorted(set(src.read(1).flatten()))
    print(f"Unique values: {unique_vals}")
    print("Legend: 64=Forb, 65=Woody debris/stem, 66=Grass, 67=Shrub, 68=Mallee, 255=NoData")




#%%
## all files (medium site)

"""
Batch downsamples and remaps structure classification rasters.

This script correctly downsamples 3072x3072 (1cm) rasters to
614x614 (5cm) rasters, which is the correct resolution for the
nnU-Net pipeline configured with voxel_size=0.05.
"""
import os
import subprocess
from osgeo import gdal
import rasterio
import re 
from pathlib import Path 

def gdal_downsample_structure(
    input_raster,
    output_raster,
    target_resolution,
    crs,
    grid_feature=None,
    nodata=255,
    remap_values=None,
):
    """
    Downsample a SINGLE structure classification raster using GDAL.
    Uses -tr to set a new target resolution.
    """
    output_dir = os.path.dirname(output_raster)
    os.makedirs(output_dir, exist_ok=True)

    if grid_feature is not None:
        geom = grid_feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()
        projWin = [minX, maxY, maxX, minY]
        clipped_raster = os.path.join(output_dir, "clipped_temp.tif")
        gdal.Translate(
            clipped_raster,
            input_raster,
            options=gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Byte, projWin=projWin),
        )
        input_for_warp = clipped_raster
    else:
        input_for_warp = input_raster

    ds = gdal.Open(input_for_warp)
    if ds is None:
        raise FileNotFoundError(f"❌ Could not open {input_for_warp}")

    gt = ds.GetGeoTransform()
    orig_left, orig_top = gt[0], gt[3]
    orig_right = orig_left + ds.RasterXSize * gt[1]
    orig_bottom = orig_top + ds.RasterYSize * gt[5]
    ds = None # Close

    gdal_command = [
        "gdalwarp",
        "-overwrite",
        "-q",
        "-s_srs", crs,
        "-t_srs", crs,
        
        # --- THIS IS THE KEY ---
        # Use Target Resolution (-tr) to create 5cm pixels.
        # This will correctly result in a 614x614 image.
        "-tr", str(target_resolution), str(target_resolution),
        # --- END KEY ---

        "-r", "nearest", # 'nearest' is correct for classification
        "-te", str(orig_left), str(orig_bottom), str(orig_right), str(orig_top),
        "-ot", "Byte",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=3072",
        "-co", "BLOCKYSIZE=3072",
        "-srcnodata", str(nodata), 
        "-dstnodata", str(nodata),
        input_for_warp,
        output_raster,
    ]

    try:
        result = subprocess.run(gdal_command, check=True, capture_output=True, text=True)
        # Get output shape for logging
        with rasterio.open(output_raster) as out_ds:
            out_shape = (out_ds.height, out_ds.width)
        print(f"  ✅ Downsampled raster saved to: {output_raster}")
        print(f"  📐 Output dimensions: {out_shape}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ gdalwarp failed for {input_raster}")
        print("  STDERR:\n", e.stderr)
        raise

    if remap_values:
        try:
            with rasterio.open(output_raster, "r+") as dst:
                band = dst.read(1)
                band_remapped = band.copy()
                for src_val, dst_val in remap_values.items():
                    band_remapped[band == src_val] = dst_val
                band_remapped[band == nodata] = nodata # Ensure nodata stays nodata
                dst.write(band_remapped, 1)
                dst.nodata = nodata
            print(f"  ✅ Applied remap: {remap_values}")
        except Exception as e:
            print(f"  ❌ Failed to remap values for {output_raster}: {e}")
            raise

    if grid_feature is not None and os.path.exists(clipped_raster):
        os.remove(clipped_raster)
        print("  🧹 Removed temporary clip file.")


def batch_downsample_and_remap(
    input_dir,
    output_dir,
    target_resolution,
    crs,
    remap_values,
    grid_feature=None
):
    """
    Finds all mask files in input_dir and runs gdal_downsample_structure on them.
    """
    print(f"Scanning for masks in: {input_dir}")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = list(input_dir.glob("mask_structure_3072.*.tif"))
    
    if not mask_files:
        print(f"🚨 No mask files found matching 'mask_structure_3072.*.tif' in {input_dir}")
        return

    print(f"Found {len(mask_files)} mask files to process.")
    print("=" * 60)

    tile_id_re = re.compile(r"mask_structure_3072\.(\d+)\.tif")
    processed_count = 0
    failed_count = 0

    for input_path in sorted(mask_files):
        match = tile_id_re.search(input_path.name)
        if not match:
            print(f"⚠️ Skipping: Could not extract tile ID from {input_path.name}")
            failed_count += 1
            continue
        
        tile_id = match.group(1)
        print(f"\n🔄 Processing Tile ID: {tile_id} (File: {input_path.name})")

        output_path = output_dir / input_path.name

        try:
            gdal_downsample_structure(
                input_raster=str(input_path),
                output_raster=str(output_path),
                target_resolution=target_resolution,
                crs=crs,
                grid_feature=grid_feature,
                nodata=255, 
                remap_values=remap_values
            )
            processed_count += 1
        except Exception as e:
            print(f"❌ FAILED for Tile ID: {tile_id}")
            print(f"   Error: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"📊 Batch Processing Summary:")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📂 Output directory: {output_dir}")
    print(f"   📊 Raster values: {remap_values}, 255=NoData")

# --- Configuration ---

# == Medium site (DD0001) ==
# SITE_NAME = "DD0001_Medium"
# input_dir = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
# output_dir = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"

# == Low site (DD0011) ==
SITE_NAME = "DD0011_Low"
input_dir = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
output_dir = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"

# == Dense site (DD0012) ==
# SITE_NAME = "DD0012_Dense"
# input_dir = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
# output_dir = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"


# --- Shared Settings ---
target_resolution = 0.05 # This is 5cm
crs = "EPSG:7854" # GDA2020 / MGA zone 54
grid_feature = None # No clipping
value_remap = {0:64, 1:65, 2:66, 3:67, 4:68}


# --- Run Batch Processing ---
print(f"--- Starting Batch Job for Site: {SITE_NAME} ---")
batch_downsample_and_remap(
    input_dir,
    output_dir,
    target_resolution,
    crs,
    remap_values=value_remap,
    grid_feature=grid_feature
)

#%%
### 🔍 Validate output (optional)
print("\n--- Validating first output file (if exists) ---")
try:
    first_output = next(Path(output_dir).glob("mask_structure_3072.*.tif"))
    print(f"Checking: {first_output}")
    with rasterio.open(first_output) as src:
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"CRS: {src.crs}")
        print(f"NoData: {src.nodata}")
        unique_vals = sorted(list(set(src.read(1).flatten())))
        print(f"Unique values: {unique_vals}")
        print("Legend: 64=Forb, 65=Woody debris/stem, 66=Grass, 67=Shrub, 68=Mallee, 255=NoData")
except StopIteration:
    print("No output files found to validate.")
except Exception as e:
    print(f"Could not validate output file: {e}")

