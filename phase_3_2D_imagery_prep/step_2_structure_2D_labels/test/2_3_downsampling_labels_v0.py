#%%
## one file
import os
import subprocess
from osgeo import gdal
import rasterio

def gdal_downsample_structure(input_raster, output_raster, target_resolution, crs, grid_feature=None, nodata=255, keep_dims=True):
    """
    Downsample a structure classification raster using GDAL while preserving the same pixel dimensions as the input raster.
    
    Adapted for structure classification data with integer values 0-4 and nodata=255.
    
    Args:
        input_raster: Path to the input structure raster.
        output_raster: Path to save the downsampled raster.
        target_resolution: Target resolution in meters (e.g., 0.05).
        crs: Coordinate Reference System (e.g., 'EPSG:7854').
        grid_feature: (Optional) An OGR feature whose geometry will be used to clip the raster.
        nodata: NoData value for structure rasters (default: 255).
        keep_dims: If True, the output will have the same pixel dimensions as the input.
    """
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_raster)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if grid_feature is not None:
        # Use grid feature envelope to clip raster first
        geom = grid_feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()
        # For GDAL Translate, projWin must be [minX, maxY, maxX, minY]
        projWin = [minX, maxY, maxX, minY]
        print(f"Grid feature envelope: minX={minX}, maxX={maxX}, minY={minY}, maxY={maxY}")

        # Clip the input raster using gdal.Translate
        translate_options = gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Byte, projWin=projWin)
        clipped_raster = os.path.join(output_dir, "clipped_temp.tif")
        gdal.Translate(clipped_raster, input_raster, options=translate_options)
        print(f"Clipped raster saved to temporary file: {clipped_raster}")
        input_for_warp = clipped_raster
    else:
        input_for_warp = input_raster

    # Open the (clipped) input raster to get dimensions and geotransform
    ds = gdal.Open(input_for_warp)
    if ds is None:
        raise FileNotFoundError(f"Could not open {input_for_warp}")
    width, height = ds.RasterXSize, ds.RasterYSize
    print(f"Input raster dimensions: width={width}, height={height}")

    gt = ds.GetGeoTransform()
    if not gt:
        raise ValueError("Could not retrieve geotransform from the raster.")
    
    # Original extent computed from geotransform
    orig_left = gt[0]
    orig_top = gt[3]
    orig_right = orig_left + width * gt[1]
    orig_bottom = orig_top + height * gt[5]  # Note: gt[5] is usually negative
    orig_extent = (orig_left, orig_bottom, orig_right, orig_top)
    print(f"Original extent from raster: {orig_extent}")

    if keep_dims:
        # Recompute extent so that output pixel dimensions remain the same as input
        # New width in meters = target_resolution * width, new height = target_resolution * height.
        new_right = orig_left + target_resolution * width
        new_bottom = orig_top - target_resolution * height
        new_extent = (orig_left, new_bottom, new_right, orig_top)
        print(f"Recomputed extent for target resolution {target_resolution}: {new_extent}")
    else:
        # Use the original extent (which would change the pixel count if target_resolution differs)
        new_extent = orig_extent
        print(f"Using original extent without dimension preservation: {new_extent}")
    
    # Set a chunk size (currently for reference or further processing)
    chunk_size = 3072
    print(f"Using chunk size: {chunk_size}")

    # Define the gdalwarp command with the computed extent
    # Use "nearest" resampling for categorical data (structure classes)
    gdal_command = [
        "gdalwarp",
        "-overwrite",                          # Overwrite existing output raster
        "-s_srs", crs,                         # Source CRS
        "-t_srs", crs,                         # Target CRS
        "-tr", str(target_resolution), str(target_resolution),  # Target resolution
        "-r", "nearest",                       # Nearest neighbor resampling for categorical data
        "-te", str(new_extent[0]), str(new_extent[1]), str(new_extent[2]), str(new_extent[3]),  # Extent: left, bottom, right, top
        "-ot", "Byte",                         # Output data type (uint8 for values 0-255)
        "-of", "GTiff",                        # Output format
        "-co", "COMPRESS=LZW",                 # Compression
        "-co", "PREDICTOR=2",                  # Compression predictor
        "-co", "BIGTIFF=YES",                  # Allow big TIFF files
        "-co", "TILED=YES",                    # Tiled format for better performance
        "-co", "BLOCKXSIZE=3072",              # Block size matching your tiles
        "-co", "BLOCKYSIZE=3072",              # Block size matching your tiles
        "-dstnodata", str(nodata),             # NoData value
        input_for_warp,
        output_raster,
    ]

    # Execute the GDAL warp command
    subprocess.run(gdal_command, check=True)
    print(f"Downsampled structure raster saved to {output_raster}")
    
    # Clean up temporary file if created
    if grid_feature is not None and os.path.exists(clipped_raster):
        os.remove(clipped_raster)
        print(f"Cleaned up temporary file: {clipped_raster}")

# Configuration for structure classification rasters
# low site
# input_raster = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.66.tif"
# output_raster = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.66.tif"

# medium site
input_raster = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.16.tif"
output_raster = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.16.tif"

# dense site
# input_raster = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/mask_structure_3072.128.tif"
# output_raster = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.128.tif"

target_resolution = 0.05  # New ground resolution (meters per pixel)
crs = "EPSG:7854"         # Target CRS

# Set grid_feature to None if not using a clipping geometry
grid_feature = None

# Process the structure raster
gdal_downsample_structure(input_raster, output_raster, target_resolution, crs, grid_feature=grid_feature, keep_dims=True)

#%%
### Check the output raster
with rasterio.open(output_raster) as src:
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Data Type: {src.dtypes[0]}")
    print(f"CRS: {src.crs}")
    print(f"Extent: {src.bounds}")
    print(f"NoData Value: {src.nodata}")
    
    # Check unique values in a sample
    sample_data = src.read(1, window=((0, 1000), (0, 1000)))  # Read first 1000x1000 pixels
    unique_values = sorted(set(sample_data.flatten()))
    print(f"Sample unique values: {unique_values}")
    print(f"Structure Legend: 0=Forb, 1=Woody debris/stem, 2=Grass, 3=Shrub, 4=Mallee, 255=NoData")



#%%
## all files
import os
import subprocess
from osgeo import gdal
import rasterio

def gdal_downsample_structure(input_raster, output_raster, target_resolution, crs, grid_feature=None, nodata=255, keep_dims=True):
    """
    Downsample a structure classification raster using GDAL while preserving the same pixel dimensions as the input raster.
    
    Adapted for structure classification data with integer values 0-4 and nodata=255.
    
    Args:
        input_raster: Path to the input structure raster.
        output_raster: Path to save the downsampled raster.
        target_resolution: Target resolution in meters (e.g., 0.05).
        crs: Coordinate Reference System (e.g., 'EPSG:7854').
        grid_feature: (Optional) An OGR feature whose geometry will be used to clip the raster.
        nodata: NoData value for structure rasters (default: 255).
        keep_dims: If True, the output will have the same pixel dimensions as the input.
    """
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_raster)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if grid_feature is not None:
        # Use grid feature envelope to clip raster first
        geom = grid_feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()
        # For GDAL Translate, projWin must be [minX, maxY, maxX, minY]
        projWin = [minX, maxY, maxX, minY]
        print(f"Grid feature envelope: minX={minX}, maxX={maxX}, minY={minY}, maxY={maxY}")

        # Clip the input raster using gdal.Translate
        translate_options = gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Byte, projWin=projWin)
        clipped_raster = os.path.join(output_dir, "clipped_temp.tif")
        gdal.Translate(clipped_raster, input_raster, options=translate_options)
        print(f"Clipped raster saved to temporary file: {clipped_raster}")
        input_for_warp = clipped_raster
    else:
        input_for_warp = input_raster

    # Open the (clipped) input raster to get dimensions and geotransform
    ds = gdal.Open(input_for_warp)
    if ds is None:
        raise FileNotFoundError(f"Could not open {input_for_warp}")
    width, height = ds.RasterXSize, ds.RasterYSize
    print(f"Input raster dimensions: width={width}, height={height}")

    gt = ds.GetGeoTransform()
    if not gt:
        raise ValueError("Could not retrieve geotransform from the raster.")
    
    # Original extent computed from geotransform
    orig_left = gt[0]
    orig_top = gt[3]
    orig_right = orig_left + width * gt[1]
    orig_bottom = orig_top + height * gt[5]  # Note: gt[5] is usually negative
    orig_extent = (orig_left, orig_bottom, orig_right, orig_top)
    print(f"Original extent from raster: {orig_extent}")

    if keep_dims:
        # Recompute extent so that output pixel dimensions remain the same as input
        # New width in meters = target_resolution * width, new height = target_resolution * height.
        new_right = orig_left + target_resolution * width
        new_bottom = orig_top - target_resolution * height
        new_extent = (orig_left, new_bottom, new_right, orig_top)
        print(f"Recomputed extent for target resolution {target_resolution}: {new_extent}")
    else:
        # Use the original extent (which would change the pixel count if target_resolution differs)
        new_extent = orig_extent
        print(f"Using original extent without dimension preservation: {new_extent}")
    
    # Set a chunk size (currently for reference or further processing)
    chunk_size = 3072
    print(f"Using chunk size: {chunk_size}")

    # Define the gdalwarp command with the computed extent
    # Use "nearest" resampling for categorical data (structure classes)
    gdal_command = [
        "gdalwarp",
        "-overwrite",                          # Overwrite existing output raster
        "-s_srs", crs,                         # Source CRS
        "-t_srs", crs,                         # Target CRS
        "-tr", str(target_resolution), str(target_resolution),  # Target resolution
        "-r", "nearest",                       # Nearest neighbor resampling for categorical data
        "-te", str(new_extent[0]), str(new_extent[1]), str(new_extent[2]), str(new_extent[3]),  # Extent: left, bottom, right, top
        "-ot", "Byte",                         # Output data type (uint8 for values 0-255)
        "-of", "GTiff",                        # Output format
        "-co", "COMPRESS=LZW",                 # Compression
        "-co", "PREDICTOR=2",                  # Compression predictor
        "-co", "BIGTIFF=YES",                  # Allow big TIFF files
        "-co", "TILED=YES",                    # Tiled format for better performance
        "-co", "BLOCKXSIZE=3072",              # Block size matching your tiles
        "-co", "BLOCKYSIZE=3072",              # Block size matching your tiles
        "-dstnodata", str(nodata),             # NoData value
        input_for_warp,
        output_raster,
    ]

    # Execute the GDAL warp command
    subprocess.run(gdal_command, check=True)
    print(f"✅ Downsampled structure raster saved to {output_raster}")
    
    # Clean up temporary file if created
    if grid_feature is not None and os.path.exists(clipped_raster):
        os.remove(clipped_raster)
        print(f"Cleaned up temporary file: {clipped_raster}")
    
    return True

def process_all_structure_rasters(input_dir, output_dir, target_resolution, crs, grid_feature=None, nodata=255, keep_dims=True):
    """
    Process all structure rasters in a directory for downsampling
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all TIF files in the input directory
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and 'mask_structure_3072' in f]
    
    if not tif_files:
        print(f"No structure raster files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} structure raster files to process")
    print("=" * 60)
    
    processed_count = 0
    failed_count = 0
    
    for tif_file in sorted(tif_files):
        print(f"\n🔄 Processing: {tif_file}")
        
        input_raster = os.path.join(input_dir, tif_file)
        output_raster = os.path.join(output_dir, tif_file)
        
        try:
            success = gdal_downsample_structure(
                input_raster, 
                output_raster, 
                target_resolution, 
                crs, 
                grid_feature=grid_feature, 
                nodata=nodata, 
                keep_dims=keep_dims
            )
            
            if success:
                processed_count += 1
                
                # Quick validation of output
                with rasterio.open(output_raster) as src:
                    print(f"  📊 Output: {src.width}x{src.height}, {src.dtypes[0]}, NoData={src.nodata}")
            
        except Exception as e:
            print(f"❌ Failed to process {tif_file}: {str(e)}")
            failed_count += 1
            continue
    
    print("\n" + "=" * 60)
    print(f"📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {processed_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📂 Output directory: {output_dir}")
    print(f"  🎯 Target resolution: {target_resolution}m")
    print(f"📊 Structure Legend: 0=Forb, 1=Woody debris/stem, 2=Grass, 3=Shrub, 4=Mallee, 255=NoData")

# Configuration for structure classification rasters
# low site
# input_dir = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
# output_dir = "/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"

# medium site
input_dir = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
output_dir = "/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"

# dense site
# input_dir = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
# output_dir = "/media/laura/laura_usb/uas_data/DD0012/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"

target_resolution = 0.05  # New ground resolution (meters per pixel)
crs = "EPSG:7854"         # Target CRS

# Set grid_feature to None if not using a clipping geometry
grid_feature = None

# Process all structure rasters in the directory
process_all_structure_rasters(input_dir, output_dir, target_resolution, crs, grid_feature=grid_feature, keep_dims=True)

#%%
### Check one output raster as example
sample_output = os.path.join(output_dir, [f for f in os.listdir(output_dir) if f.endswith('.tif')][0])
print(f"\n🔍 Checking sample output: {os.path.basename(sample_output)}")

with rasterio.open(sample_output) as src:
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Data Type: {src.dtypes[0]}")
    print(f"CRS: {src.crs}")
    print(f"Extent: {src.bounds}")
    print(f"NoData Value: {src.nodata}")
    
    # Check unique values in a sample
    sample_data = src.read(1, window=((0, min(1000, src.height)), (0, min(1000, src.width))))
    unique_values = sorted(set(sample_data.flatten()))
    print(f"Sample unique values: {unique_values}")
    print(f"Structure Legend: 0=Forb, 1=Woody debris/stem, 2=Grass, 3=Shrub, 4=Mallee, 255=NoData")

# %%