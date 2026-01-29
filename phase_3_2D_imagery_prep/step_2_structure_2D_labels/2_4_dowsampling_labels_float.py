
#%%
import os
import subprocess
from osgeo import gdal
import rasterio

def gdal_downsample(input_raster, output_raster, target_resolution, crs, grid_feature=None, nodata=-32767, keep_dims=True):
    """
    Downsample a raster using GDAL while preserving the same pixel dimensions as the input raster.
    
    If keep_dims is True, the output raster will have the same number of pixels as the input.
    To achieve that, the output extent is recomputed based on the target resolution and input dimensions.
    
    Optionally, if a grid feature is provided, its geometry is used to clip the input raster
    before the downsampling.
    
    Args:
        input_raster: Path to the input raster.
        output_raster: Path to save the downsampled raster.
        target_resolution: Target resolution in meters (e.g., 0.05).
        crs: Coordinate Reference System (e.g., 'EPSG:7854').
        grid_feature: (Optional) An OGR feature whose geometry will be used to clip the raster.
        nodata: NoData value for the output raster (default: -32767).
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
        translate_options = gdal.TranslateOptions(format="GTiff", outputType=gdal.GDT_Float32, projWin=projWin)
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
    gdal_command = [
        "gdalwarp",
        "-overwrite",                          # Overwrite existing output raster
        "-s_srs", crs,                         # Source CRS
        "-t_srs", crs,                         # Target CRS
        "-tr", str(target_resolution), str(target_resolution),  # Target resolution
        "-r", "bilinear",                      # Bilinear resampling - bilinear / nearest
        "-te", str(new_extent[0]), str(new_extent[1]), str(new_extent[2]), str(new_extent[3]),  # Extent: left, bottom, right, top
        "-ot", "Float32",                      # Output data type
        "-of", "GTiff",                        # Output format
        "-co", "COMPRESS=LZW",                 # Compression
        "-co", "PREDICTOR=2",                  # Compression predictor
        "-co", "BIGTIFF=YES",                  # Allow big TIFF files
        "-dstnodata", str(nodata),             # NoData value
        input_for_warp,
        output_raster,
    ]

    # Execute the GDAL warp command
    subprocess.run(gdal_command, check=True)
    print(f"Downsampled raster saved to {output_raster}")

# Example usage:
input_raster = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif"
# output_raster = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/dowsample_5cm_nearest/mask_fvc_3072.26.tif"
output_raster = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/dowsample_5cm_bilinear/mask_fvc_3072.26.tif"

target_resolution = 0.05  # New ground resolution (meters per pixel)
crs = "EPSG:7854"         # Target CRS

# Set grid_feature to None if not using a clipping geometry.
grid_feature = None

gdal_downsample(input_raster, output_raster, target_resolution, crs, grid_feature=grid_feature, keep_dims=True)



#%%
### Check the output raster

with rasterio.open(output_raster) as src:
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"Data Type: {src.dtypes[0]}")
    print(f"CRS: {src.crs}")
    print(f"Extent: {src.bounds}")


# %%
