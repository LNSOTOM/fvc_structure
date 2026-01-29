

#%%
from pathlib import Path
import subprocess, shutil, sys

input_file  = "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las"
output_file = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181.laz"

candidates = [
    "/home/laura/lastools/bin/laszip",
    "/home/laura/lastools/bin/laszip64",
    shutil.which("laszip"),
]
laszip = next((p for p in candidates if p and Path(p).exists()), None)
if not laszip:
    sys.exit("laszip not found. Check your LAStools install/path.")

Path(output_file).parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["chmod", "+x", laszip], check=True)
subprocess.run([laszip, "-i", input_file, "-o", output_file], check=True)
print(f"✅ Wrote {output_file}")

#%%
#!/usr/bin/env python3
import subprocess
import shutil
import sys
import os
from pathlib import Path
import json

# ========= USER INPUTS =========
INPUT_PATH = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
OUT_DIR    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar")
SAVE_PDAL_JSON = True  # set False if you don't want PDAL JSON dumps

# ========= LAStools binaries (edit if needed) =========
LASZIP_CANDIDATES  = ["/home/laura/lastools/bin/laszip", "/home/laura/lastools/bin/laszip64", shutil.which("laszip")]
LASINFO_CANDIDATES = ["/home/laura/lastools/bin/lasinfo", "/home/laura/lastools/bin/lasinfo64", shutil.which("lasinfo")]

# ========= ENV (if you need a specific lib path) =========
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = ENV.get("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")

# ========= HELPERS =========
def find_bin(candidates, name):
    for c in candidates:
        if c and Path(c).exists():
            return c
    print(f"⚠️  {name} not found via candidates. Will try to continue without it.")
    return None

def run_logged(cmd, check=True):
    print("•", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check, env=ENV)

def run_capture(cmd, check=True):
    print("•", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check, env=ENV, capture_output=True, text=True)

# ========= MAIN =========
def main():
    if not INPUT_PATH.exists():
        sys.exit(f"⛔ Input file not found: {INPUT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    laszip = find_bin(LASZIP_CANDIDATES, "laszip")
    lasinfo = find_bin(LASINFO_CANDIDATES, "lasinfo")

    # Decide output LAZ path
    if INPUT_PATH.suffix.lower() == ".laz":
        laz_path = OUT_DIR / INPUT_PATH.name
        if INPUT_PATH != laz_path:
            print("📁 Input is already .laz — copying to output dir…")
            laz_path.write_bytes(INPUT_PATH.read_bytes())
        else:
            print("📁 Input is already .laz in output dir — reusing.")
    else:
        # Compress LAS → LAZ with laszip or PDAL fallback
        laz_path = OUT_DIR / (INPUT_PATH.stem + ".laz")
        if laszip:
            print("🛠️ Compressing with laszip…")
            run_logged([laszip, "-i", str(INPUT_PATH), "-o", str(laz_path)])
        else:
            print("ℹ️ laszip not found. Falling back to PDAL for compression…")
            # PDAL: writers.las + compression=laszip
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(INPUT_PATH)},
                    {
                        "type": "writers.las",
                        "filename": str(laz_path),
                        "compression": "laszip",
                        "minor_version": 4
                    }
                ]
            }
            try:
                run_logged(["pdal", "pipeline", "-s", json.dumps(pipeline)])
            except FileNotFoundError:
                sys.exit("⛔ Neither laszip nor pdal is available for compression.")

        print(f"✅ Compression complete: {laz_path}")

    # lasinfo → write to .txt directly (native -o)
    info_txt = laz_path.with_suffix(".txt")
    if lasinfo:
        print("📊 Extracting metadata with lasinfo…")
        # `-o` writes directly to file (more reliable than capturing stdout/stderr)
        run_logged([lasinfo, "-i", str(laz_path), "-o", str(info_txt)])
        print(f"✅ Metadata saved to: {info_txt}")
    else:
        print("ℹ️ lasinfo not found. Skipping .txt; consider PDAL JSON below.")

    # Optional PDAL JSON dumps (very handy for scripting)
    if SAVE_PDAL_JSON:
        try:
            print("🧭 PDAL --metadata …")
            meta_json = laz_path.with_suffix(".metadata.json")
            meta = run_capture(["pdal", "info", "--metadata", str(laz_path)]).stdout
            meta_json.write_text(meta)

            print("🧮 PDAL --summary …")
            summary_json = laz_path.with_suffix(".summary.json")
            summary = run_capture(["pdal", "info", "--summary", str(laz_path)]).stdout
            summary_json.write_text(summary)

            print(f"✅ PDAL JSON written:\n   {meta_json}\n   {summary_json}")
        except FileNotFoundError:
            print("ℹ️ PDAL not found. Skipping PDAL JSON dumps.")

    print("🎉 Done.")

if __name__ == "__main__":
    main()


#%%
from pathlib import Path
import rasterio
import laspy

# ==== INPUT FILES ====
# input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/cloud2453cf90763afd91_thinned_3cm_gda2020.las")
aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# ==== LAS FILE ====
las = laspy.read(input_las)
las_bounds = (
    float(las.header.mins[0]),
    float(las.header.mins[1]),
    float(las.header.maxs[0]),
    float(las.header.maxs[1])
)

print("\n--- LAS Information ---")
print(f"Bounds (minX, minY, maxX, maxY): {las_bounds}")
# LAS CRS often needs to be checked from metadata or PDAL; laspy doesn't store CRS directly.
print("CRS: Not stored directly in LAS header (check with PDAL or external metadata)")

# ==== RASTER FILE ====
with rasterio.open(aoi_raster) as src:
    raster_bounds = src.bounds
    raster_crs = src.crs

print("\n--- Raster Information ---")
print(f"Bounds: {raster_bounds}")
print(f"CRS: {raster_crs}")

# ==== COMPARISON ====
print("\n--- Comparison ---")
print(f"X overlap: {not (las_bounds[2] < raster_bounds.left or las_bounds[0] > raster_bounds.right)}")
print(f"Y overlap: {not (las_bounds[3] < raster_bounds.bottom or las_bounds[1] > raster_bounds.top)}")
print("⚠️ Ensure both datasets are in the same CRS before comparing coordinates.")


# --- LAS Information ---
# Bounds (minX, minY, maxX, maxY): (461661.65966392256, 6237294.878825261, 462058.63800661237, 6237772.835600468)
# CRS: Not stored directly in LAS header (check with PDAL or external metadata)

# --- Raster Information ---
# Bounds: BoundingBox(left=461986.60469617933, bottom=6237690.407523299, right=462017.3246961793, top=6237721.127523298)
# CRS: EPSG:7854

# --- Comparison ---
# X overlap: True
# Y overlap: True
# ⚠️ Ensure both datasets are in the same CRS before comparing coordinates.

# %%
# Compare LAS CRS vs Raster CRS --> GOOD
import json, subprocess
from pathlib import Path
import rasterio as rio

# las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
# tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm/cloud2453cf90763afd91_cropped_only.las")
# tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")

# las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c.las")
# tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")

# las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c_aligned.las")
# tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")

#LOW
las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/trim/cloud2453cf90763afd91_clip_raster_extent.las")
tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")


#MEDIUM
# las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/trim/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.109.las")
# tif_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.109.tif") 



# --- Raster CRS ---
with rio.open(tif_path) as src:
    raster_crs = src.crs
print(f"Raster CRS: {raster_crs}")

# --- LAS CRS via PDAL ---
cmd = ["pdal", "info", "--metadata", str(las_path)]
meta = json.loads(subprocess.check_output(cmd))
las_srs = meta["metadata"].get("spatialreference", "").strip()
print("LAS spatialreference (raw):")
print(las_srs if las_srs else "⛔ None found")

# Very rough check for EPSG code presence
is_epsg7854 = "EPSG:7854" in las_srs or "7854" in las_srs and "GDA2020" in las_srs
print(f"LAS looks like EPSG:7854? {is_epsg7854}")

# --- Optional: Containment check you already ran, now explicit ---
las_minx = meta["metadata"]["minx"]; las_maxx = meta["metadata"]["maxx"]
las_miny = meta["metadata"]["miny"]; las_maxy = meta["metadata"]["maxy"]
rb = rio.open(tif_path).bounds
inside = (rb.left >= las_minx and rb.right <= las_maxx and
          rb.bottom >= las_miny and rb.top <= las_maxy)
print(f"Raster extent inside LAS extent? {inside}")


# Raster CRS: EPSG:7854
# LAS spatialreference (raw):
# PROJCS["GDA2020 / MGA zone 54",GEOGCS["GDA2020",DATUM["Geocentric_Datum_of_Australia_2020",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","1168"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","7844"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",141],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","7854"]]
# LAS looks like EPSG:7854? True
# Raster extent inside LAS extent? True


# %%
import json, subprocess
from pathlib import Path
import rasterio as rio

def check_las_raster_alignment(las_path, tif_path):
    las_path = Path(las_path); tif_path = Path(tif_path)

    # --- Raster: bounds + CRS
    with rio.open(tif_path) as src:
        r_bounds = src.bounds
        r_crs = src.crs

    # --- LAS: bounds + SRS via PDAL
    meta = json.loads(subprocess.check_output(
        ["pdal", "info", "--metadata", str(las_path)]
    ))
    m = meta["metadata"]
    l_minx, l_maxx = m["minx"], m["maxx"]
    l_miny, l_maxy = m["miny"], m["maxy"]
    l_srs_raw = (m.get("spatialreference") or "").strip()

    # Basic CRS check
    crs_match = "EPSG:7854" in l_srs_raw or ("GDA2020" in l_srs_raw and "zone 54" in l_srs_raw)
    print(f"Raster CRS: {r_crs}")
    print(f"LAS CRS (parsed): {'EPSG:7854 (GDA2020 / MGA zone 54)' if crs_match else 'Unknown/Other'}")
    if not crs_match:
        print("⚠️ LAS CRS did not look like EPSG:7854. Reproject or set SRS before comparing.")
        return

    # Containment + margins
    inside_x = (r_bounds.left >= l_minx) and (r_bounds.right <= l_maxx)
    inside_y = (r_bounds.bottom >= l_miny) and (r_bounds.top <= l_maxy)
    print(f"X inside LAS bbox? {inside_x}")
    print(f"Y inside LAS bbox? {inside_y}")

    # Margins (meters)
    margin_left   = r_bounds.left   - l_minx
    margin_right  = l_maxx          - r_bounds.right
    margin_bottom = r_bounds.bottom - l_miny
    margin_top    = l_maxy          - r_bounds.top
    print("Margins (meters):")
    print(f"  left: {margin_left:.3f}, right: {margin_right:.3f}, "
          f"bottom: {margin_bottom:.3f}, top: {margin_top:.3f}")

    # Overlap % (rectangle intersection)
    ix_left   = max(l_minx, r_bounds.left)
    ix_right  = min(l_maxx, r_bounds.right)
    ix_bottom = max(l_miny, r_bounds.bottom)
    ix_top    = min(l_maxy, r_bounds.top)
    inter_w = max(0.0, ix_right - ix_left)
    inter_h = max(0.0, ix_top - ix_bottom)
    inter_area = inter_w * inter_h
    r_area = (r_bounds.right - r_bounds.left) * (r_bounds.top - r_bounds.bottom)
    overlap_pct = 100.0 * inter_area / r_area if r_area > 0 else 0.0
    print(f"Raster–LAS overlap: {overlap_pct:.2f}% of raster area")

    # Edge proximity warning (optional threshold)
    edge_thresh = 0.5  # meters
    close_edges = []
    if margin_left < edge_thresh:   close_edges.append("left")
    if margin_right < edge_thresh:  close_edges.append("right")
    if margin_bottom < edge_thresh: close_edges.append("bottom")
    if margin_top < edge_thresh:    close_edges.append("top")
    if close_edges:
        print(f"⚠️ Raster is within {edge_thresh} m of LAS edge on: {', '.join(close_edges)}")

# === Use it ===
check_las_raster_alignment(
    "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las",
    "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif"
)

# Raster CRS: EPSG:7854
# LAS CRS (parsed): EPSG:7854 (GDA2020 / MGA zone 54)
# X inside LAS bbox? True
# Y inside LAS bbox? True
# Margins (meters):
#   left: 324.945, right: 41.313, bottom: 395.529, top: 51.708
# Raster–LAS overlap: 100.00% of raster area

# %%
import json, subprocess
from pathlib import Path
import rasterio as rio

las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
tif = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
out = las.with_name(las.stem + "_tiles_multispectral.26.laz")

buf = 0.5  # meters
with rio.open(tif) as src:
    b = src.bounds
bounds = f"([{b.left-buf},{b.right+buf}],[{b.bottom-buf},{b.top+buf}])"

pipeline = {
  "pipeline": [
    str(las),
    {"type": "filters.crop", "bounds": bounds},
    {"type": "writers.las", "filename": str(out)}
  ]
}
subprocess.check_call(["pdal", "pipeline", "-s", json.dumps(pipeline)])
print(f"✅ wrote {out}")

# %%
#!/usr/bin/env python3
"""
Crop a LAS/LAZ to a raster tile's extent (with optional buffer),
reading bounds automatically from the GeoTIFF.

Requirements: rasterio, PDAL (CLI in PATH)
"""
#!/usr/bin/env python3
import subprocess, sys, json
from pathlib import Path
import rasterio as rio

# ========= USER INPUTS =========
LAS_IN  = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
TIF_IN  = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
OUT_DIR = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim")
BUF_M   = 0.50          # meters buffer around tile
EXPECTED_EPSG = "EPSG:7854"
FAST_MODE = True        # set False to do CRS checks + detailed summary

OUT_DIR.mkdir(parents=True, exist_ok=True)
# Auto-name from raster filename: e.g., tiles_multispectral.26 → _tiles_multispectral.26.laz
tile_tag = TIF_IN.stem
OUT_LAZ = OUT_DIR / f"{LAS_IN.stem}_{tile_tag}.laz"

def run(cmd, capture=True):
    print("•", " ".join(map(str, cmd)))
    if capture:
        return subprocess.check_output(cmd)
    return subprocess.check_call(cmd)

def raster_bounds_crs(tif_path: Path):
    with rio.open(tif_path) as src:
        b = src.bounds
        crs = src.crs
    if crs is None:
        sys.exit("⛔ Raster has no CRS. Assign a CRS first.")
    return b, crs

def build_bounds(b, buf):
    xmin = b.left   - buf
    xmax = b.right  + buf
    ymin = b.bottom - buf
    ymax = b.top    + buf
    return f"([{xmin},{xmax}],[{ymin},{ymax}])"

def las_crs_contains_epsg7854(las_path: Path):
    meta = json.loads(run(["pdal", "info", "--metadata", str(las_path)]))
    srs = (meta.get("metadata", {}) or {}).get("spatialreference", "")
    return ("EPSG:7854" in srs) or ("GDA2020" in srs and "zone 54" in srs)

def quick_summary(path: Path):
    # Single call for bbox & count
    js = json.loads(run(["pdal", "info", "--summary", str(path)]))
    s  = js.get("summary", {})
    bbox = s.get("bbox", {}).get("native", {})
    count = s.get("num_points", "unknown")
    if bbox:
        minx = bbox.get("minx"); maxx = bbox.get("maxx")
        miny = bbox.get("miny"); maxy = bbox.get("maxy")
        print(f"   points: {count}")
        if minx is not None:
            print(f"   X: {minx:.3f} .. {maxx:.3f}")
            print(f"   Y: {miny:.3f} .. {maxy:.3f}")
    else:
        print(f"   points: {count}")

if __name__ == "__main__":
    # 1) Bounds + CRS from raster
    r_bounds, r_crs = raster_bounds_crs(TIF_IN)
    print(f"Raster CRS: {r_crs}, bounds: {r_bounds}")

    # 2) Optional: verify LAS CRS matches EPSG:7854
    if not FAST_MODE:
        if not las_crs_contains_epsg7854(LAS_IN):
            print("⚠️ LAS CRS is not clearly EPSG:7854. Consider setting/reprojecting before crop.")
            # You can sys.exit(1) here if you want to enforce.
    else:
        print("FAST_MODE=True: skipping LAS CRS check for speed.")

    # 3) Build buffered PDAL bounds string
    bounds_str = build_bounds(r_bounds, BUF_M)
    print("Crop bounds:", bounds_str)

    # 4) Fast crop with pdal translate (+ streaming)
    cmd = [
        "pdal", "translate",
        str(LAS_IN),
        str(OUT_LAZ),
        "crop",
        f"--filters.crop.bounds={bounds_str}",
        "--stream"
    ]
    run(cmd, capture=False)
    print(f"✅ Wrote {OUT_LAZ}")

    # 5) Optional quick summary (single call)
    if not FAST_MODE:
        print("→ Post-clip summary:")
        quick_summary(OUT_LAZ)

    print("✅ Done.")

# %%
### OTHER
#!/usr/bin/env python3
# Reproject ONLY the problem LAS to the raster's CRS (EPSG:7854) and write that CRS in the header.

from pathlib import Path
import json
import laspy
import rasterio
import pdal

# ===== Paths =====
REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_06.tif")
GOOD_LAS   = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")   # untouched
FIX_ME_LAS = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c.las")

OUT_DIR    = FIX_ME_LAS.parent / "reprojected_to_raster_crs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LAS    = OUT_DIR / f"{FIX_ME_LAS.stem}_EPSG7854.las"

# ===== Helpers =====
def pdal_run(pipe_dict):
    p = pdal.Pipeline(json.dumps(pipe_dict))
    p.execute()

def extent_xy(path: Path):
    las = laspy.read(str(path))
    x, y = las.x, las.y
    return (float(x.min()), float(x.max()), float(y.min()), float(y.max()), len(x))

def crs_str(path: Path) -> str:
    crs = laspy.read(str(path)).header.parse_crs()
    if crs is None:
        return "None"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()

# ===== Get raster CRS =====
with rasterio.open(REF_RASTER) as src:
    raster_srs = src.crs.to_string()  # e.g. 'EPSG:7854'
    left, bottom, right, top = src.bounds

print(f"Raster CRS: {raster_srs}")
print(f"Raster bounds: X[{left:.3f},{right:.3f}]  Y[{bottom:.3f},{top:.3f}]")

# ===== Report before =====
print("\n— BEFORE —")
print(f"GOOD LAS   : {GOOD_LAS}")
print(f"  CRS  : {crs_str(GOOD_LAS)}")
gx0, gx1, gy0, gy1, gn = extent_xy(GOOD_LAS)
print(f"  Extent: X[{gx0:.3f},{gx1:.3f}]  Y[{gy0:.3f},{gy1:.3f}]  n={gn:,}")

print(f"\nFIX-ME LAS: {FIX_ME_LAS}")
print(f"  CRS  : {crs_str(FIX_ME_LAS)}")
fx0, fx1, fy0, fy1, fn = extent_xy(FIX_ME_LAS)
print(f"  Extent: X[{fx0:.3f},{fx1:.3f}]  Y[{fy0:.3f},{fy1:.3f}]  n={fn:,}")

# ===== Reproject the FIX-ME LAS to raster CRS (drops vertical component) =====
pipe = {
    "pipeline": [
        {"type": "readers.las", "filename": str(FIX_ME_LAS)},
        {"type": "filters.reprojection", "out_srs": raster_srs},
        {"type": "writers.las", "filename": str(OUT_LAS), "a_srs": raster_srs}
    ]
}
pdal_run(pipe)

# ===== Report after =====
print("\n— AFTER  —")
print(f"Output LAS : {OUT_LAS}")
print(f"  CRS  : {crs_str(OUT_LAS)}")
ox0, ox1, oy0, oy1, on = extent_xy(OUT_LAS)
print(f"  Extent: X[{ox0:.3f},{ox1:.3f}]  Y[{oy0:.3f},{oy1:.3f}]  n={on:,}")

# ===== Quick overlap sanity =====
def overlaps(r_left, r_right, r_bottom, r_top, x0, x1, y0, y1):
    return (x1 >= r_left) and (x0 <= r_right) and (y1 >= r_bottom) and (y0 <= r_top)

print("\n— OVERLAP CHECKS —")
print(f"Raster within GOOD LAS?   {overlaps(left, right, bottom, top, gx0, gx1, gy0, gy1)}")
print(f"Raster within FIXED LAS?  {overlaps(left, right, bottom, top, ox0, ox1, oy0, oy1)}")

print("\nDone. Use the *_EPSG7854.las as input to your crop→sample thinning pipeline.")

# %%
##align
import laspy
import numpy as np

good = laspy.read("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
bad  = laspy.read("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/reprojected_to_raster_crs/cloud2d765c57b8471e2c_EPSG7854.las")

# centroids
cx_good, cy_good = np.mean(good.x), np.mean(good.y)
cx_bad,  cy_bad  = np.mean(bad.x), np.mean(bad.y)

dx, dy = cx_good - cx_bad, cy_good - cy_bad
print(f"ΔX={dx:.3f}, ΔY={dy:.3f}")

# apply shift
bad.x += dx
bad.y += dy

# save corrected LAS
out_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c_aligned.las"
bad.write(out_path)
print(f"Saved aligned LAS → {out_path}")

# ΔX=11377.886, ΔY=-1771.307
# %%
# 1) Quick overlap + density sanity check
from pathlib import Path
import laspy, numpy as np

GOOD = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
ALGN = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c_aligned.las")

def extent_xy(p):
    las = laspy.read(str(p))
    x, y = las.x, las.y
    return float(x.min()), float(x.max()), float(y.min()), float(y.max()), len(x)

def print_extent(name, p):
    x0,x1,y0,y1,n = extent_xy(p)
    print(f"{name}: X[{x0:.3f},{x1:.3f}]  Y[{y0:.3f},{y1:.3f}]  n={n:,}")

def bbox_overlaps(a, b):
    ax0,ax1,ay0,ay1,_ = extent_xy(a)
    bx0,bx1,by0,by1,_ = extent_xy(b)
    return (ax1 >= bx0) and (ax0 <= bx1) and (ay1 >= by0) and (ay0 <= by1)

print_extent("GOOD", GOOD)
print_extent("ALIGNED", ALGN)
print("BBoxes overlap? ", bbox_overlaps(GOOD, ALGN))

# %%
## CHECK CRS AND EXTENTS
from pathlib import Path
import laspy
import rasterio

# ===== Paths =====
# #low : DD0011
# REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_06.tif")
# REF_DSM    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/dsm/SASMDD0011-dsm_smoothmesh_3cm.tif")

# LAS_L1     = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las")
# LAS_L2     = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las")

#low_2 : DD0013
REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_2_DD0013/orthomosaic/20240310_SASMDD0013_multispec_ortho_05_cog.tif")
REF_DSM    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/dsm/SASMDD0011-dsm_smoothmesh_3cm.tif")

LAS_L1     = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las")
LAS_L2     = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las")


##medium : DD0001
# REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/orthomosaic/20220519_SASMDD0001_dual_ortho_05.tif")
# REF_DSM    = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/orthomosaic/dsm/SASMDD0001-dsm_smoothmesh_3cm.tif")

# LAS_L1     = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# LAS_L2     = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las")


# ===== Helpers =====
def las_crs_string(path: Path) -> str:
    with laspy.open(str(path)) as fh:
        crs = fh.header.parse_crs()
    if crs is None:
        return "None (no CRS in LAS header)"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()

def las_extent_from_header(path: Path):
    # Uses header mins/maxs (no full point load)
    with laspy.open(str(path)) as fh:
        h = fh.header
        (xmin, ymin, zmin) = h.mins
        (xmax, ymax, zmax) = h.maxs
        n_points = h.point_count
    return float(xmin), float(xmax), float(ymin), float(ymax), int(n_points)

def overlaps(r_left, r_right, r_bottom, r_top, x0, x1, y0, y1):
    return (x1 >= r_left) and (x0 <= r_right) and (y1 >= r_bottom) and (y0 <= r_top)

def crs_equal_raster_las(raster_crs, las_path: Path) -> bool:
    with laspy.open(str(las_path)) as fh:
        las_crs = fh.header.parse_crs()
    if (raster_crs is None) or (las_crs is None):
        return False
    r_epsg, l_epsg = raster_crs.to_epsg(), las_crs.to_epsg()
    if r_epsg and l_epsg:
        return r_epsg == l_epsg
    return raster_crs.to_wkt() == las_crs.to_wkt()

def print_raster_info(label, path: Path):
    with rasterio.open(path) as src:
        crs = src.crs
        left, bottom, right, top = src.bounds
        print(f"\n— {label} —")
        print(f"Path  : {path}")
        print(f"CRS   : {crs.to_string() if crs else 'None'}")
        print(f"Bounds: X[{left:.3f}, {right:.3f}]  Y[{bottom:.3f}, {top:.3f}]")
        return crs, left, bottom, right, top

# ===== Raster CRS & bounds =====
raster_crs, left, bottom, right, top = print_raster_info("ORTHO", REF_RASTER)
dsm_crs, dsm_left, dsm_bottom, dsm_right, dsm_top = print_raster_info("DSM (3 cm)", REF_DSM)

# ===== LAS reports =====
for label, las_path in [("LAS L1", LAS_L1), ("LAS L2", LAS_L2)]:
    x0, x1, y0, y1, n = las_extent_from_header(las_path)
    print(f"\n— {label} —")
    print(f"Path   : {las_path}")
    print(f"CRS    : {las_crs_string(las_path)}")
    print(f"Extent : X[{x0:.3f}, {x1:.3f}]  Y[{y0:.3f}, {y1:.3f}]  n={n:,}")
    print(f"Overlaps ORTHO? {overlaps(left, right, bottom, top, x0, x1, y0, y1)}")
    print(f"Overlaps DSM?   {overlaps(dsm_left, dsm_right, dsm_bottom, dsm_top, x0, x1, y0, y1)}")

# ===== CRS equality checks (best-effort) =====
print("\n— CRS MATCH (best-effort) —")
print(f"ORTHO vs LAS L1 : {crs_equal_raster_las(raster_crs, LAS_L1)}")
print(f"ORTHO vs LAS L2 : {crs_equal_raster_las(raster_crs, LAS_L2)}")
print(f"DSM   vs ORTHO  : {dsm_crs == raster_crs}")


# %%
