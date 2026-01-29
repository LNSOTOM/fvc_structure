
#%%
#!/usr/bin/env python3
from pathlib import Path
import json
import laspy
import rasterio

# ======= INPUTS =======
DSM = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/dsm/SASMDD0011-dsm_smoothmesh_3cm.tif")
LAS_FILES = [
    Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las"),
    Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las"),
]
OUT_DIR = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_DSM_validated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======= HELPERS =======
def las_header_info(path: Path):
    with laspy.open(str(path)) as fh:
        h = fh.header
        (xmin, ymin, zmin) = h.mins
        (xmax, ymax, zmax) = h.maxs
        crs = h.parse_crs()
        epsg = crs.to_epsg() if crs else None
        return {
            "path": path,
            "xmin": float(xmin), "xmax": float(xmax),
            "ymin": float(ymin), "ymax": float(ymax),
            "npts": int(h.point_count),
            "crs": crs, "epsg": epsg
        }

def area(b):
    return max(0.0, (b["xmax"] - b["xmin"])) * max(0.0, (b["ymax"] - b["ymin"]))

def bounds_str(xmin, xmax, ymin, ymax):
    return f"X[{xmin:.3f},{xmax:.3f}] Y[{ymin:.3f},{ymax:.3f}]"

def intersect(a, b):
    xmin = max(a["xmin"], b["xmin"])
    xmax = min(a["xmax"], b["xmax"])
    ymin = max(a["ymin"], b["ymin"])
    ymax = min(a["ymax"], b["ymax"])
    if (xmax <= xmin) or (ymax <= ymin):
        return None
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

def pdal_crop_pipeline(in_las: Path, out_las: Path, xmin, xmax, ymin, ymax, a_srs=None):
    pipe = [
        {"type": "readers.las", "filename": str(in_las)},
        {"type": "filters.crop", "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"},
        {"type": "writers.las", "filename": str(out_las)}
    ]
    if a_srs:
        pipe[-1]["a_srs"] = a_srs
    return {"pipeline": pipe}

# ======= READ DSM BOUNDS/CRS =======
with rasterio.open(DSM) as src:
    dsm_crs = src.crs
    dsm_epsg = dsm_crs.to_epsg() if dsm_crs else None
    dsm_bounds = {
        "xmin": float(src.bounds.left),
        "xmax": float(src.bounds.right),
        "ymin": float(src.bounds.bottom),
        "ymax": float(src.bounds.top),
    }
print("— DSM —")
print(f"Path : {DSM}")
print(f"CRS  : {dsm_crs.to_string() if dsm_crs else 'None'}")
print("Ext  :", bounds_str(**dsm_bounds))

# ======= READ ALL LAS HEADERS =======
infos = [las_header_info(p) for p in LAS_FILES]
for i, info in enumerate(infos, 1):
    crs_desc = ("None (no CRS in header)" if info["crs"] is None
                else (f"EPSG:{info['epsg']}" if info["epsg"] else "WKT (no EPSG code)"))
    print(f"\n— LAS {i} —")
    print("Path :", info["path"])
    print("CRS  :", crs_desc)
    print("Ext  :", bounds_str(info["xmin"], info["xmax"], info["ymin"], info["ymax"]))
    print(f"Area : {area(info):.3f} m^2  Points: {info['npts']:,}")

# Warn if any LAS CRS != DSM CRS (best-effort via EPSG)
if dsm_epsg is not None:
    mismatched = [info for info in infos if info["epsg"] != dsm_epsg]
    if mismatched:
        print("\n[WARN] Some LAS files have EPSG ≠ DSM EPSG "
              f"(DSM EPSG:{dsm_epsg}). Outputs will be stamped with DSM EPSG.")
else:
    print("\n[WARN] DSM has no EPSG code; outputs will inherit smallest LAS CRS if present.")

# ======= PICK SMALLEST LAS BY AREA =======
small = min(infos, key=area)
print("\n=> Smallest LAS by area:", small["path"].name)

# ======= INTERSECT SMALLEST WITH DSM =======
inter = intersect(small, dsm_bounds)
if inter is None:
    raise SystemExit("ERROR: Smallest LAS does not intersect DSM extent. Check inputs/CRS.")
print("DSM-validated extent (smallest LAS ∩ DSM):", bounds_str(**inter))

# ======= BUILD PDAL PIPELINES FOR ALL LAS =======
# Prefer DSM EPSG for a_srs; else smallest LAS; else WKT fallback
a_srs = (f"EPSG:{dsm_epsg}" if dsm_epsg else
         (f"EPSG:{small['epsg']}" if small["epsg"] else (small["crs"].to_wkt() if small["crs"] else None)))

written = []
for info in infos:
    suffix = "_crop_DSM_intersection"
    if info is small:
        out = OUT_DIR / f"{info['path'].stem}{suffix}.las"
    else:
        out = OUT_DIR / f"{info['path'].stem}_to_{small['path'].stem}{suffix}.las"
    pipe = pdal_crop_pipeline(info["path"], out, inter["xmin"], inter["xmax"], inter["ymin"], inter["ymax"], a_srs=a_srs)
    fname = OUT_DIR / f"crop_{info['path'].stem}.json"
    with open(fname, "w") as f:
        json.dump(pipe, f, indent=2)
    written.append((fname, out))

print("\nWritten pipelines:")
for j in written:
    print("  -", j[0])

print("\nRun:")
for j in written:
    print(f"  pdal pipeline \"{j[0]}\"")

# ======= OPTIONAL: Print DSM crop command (no resampling, just cut to extent) =======
print("\n(Optional) Crop DSM to the same extent (keeps native resolution):")
print("gdalwarp -overwrite -te "
      f"{inter['xmin']} {inter['ymin']} {inter['xmax']} {inter['ymax']} "
      f"-te_srs {dsm_crs.to_string() if dsm_crs else 'EPSG:7854'} -of GTiff "
      f"\"{DSM}\" \"{OUT_DIR / (DSM.stem + '_crop_DSM_intersection.tif')}\"")

# %%
