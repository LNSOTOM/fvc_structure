

import Metashape
import os

# === SETUP ===
project_path = "E:\phd\uas_data\calperumResearch\SASMDD0011\20220517\proc\metashape/project.psx"  # Update this
export_path = "/path/to/export/DSM_rgb_2cm.tif"  # Output DSM GeoTIFF

# Resolution (in meters per pixel)
resolution = 0.0207363  # ~2.07 cm

# === OPEN PROJECT ===
doc = Metashape.Document()
doc.open(project_path)
chunk = doc.chunk

# === SET OUTPUT CRS ===
chunk.crs = Metashape.CoordinateSystem("EPSG::7854")  # GDA2020 MGA Zone 54

# === BUILD DEM FROM DENSE CLOUD ===
chunk.buildDem(
    source_data=Metashape.DataSource.DenseCloudData,
    interpolation=Metashape.Interpolation.EnabledInterpolation,
    projection=chunk.crs,
    region=chunk.region,
    resolution=resolution
)

# === EXPORT DEM ===
chunk.exportDem(
    path=export_path,
    image_format=Metashape.ImageFormatTIFF,
    format=Metashape.RasterFormat.RasterFormatTiles,
    projection=chunk.crs,
    nodata=-9999,
    write_kml=False
)

print(f"✅ RGB-derived DSM saved to: {export_path}")
