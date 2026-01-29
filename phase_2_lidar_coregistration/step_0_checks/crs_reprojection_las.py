
#%%
#!/usr/bin/env python3
"""
Re-write LAS spatial reference to EPSG:7854 (2D only).
This does NOT reproject coordinates, it only updates the SRS metadata in the header.
"""

import subprocess
from pathlib import Path

def rewrite_srs_to_7854(input_las: Path):
    """
    Update LAS header to EPSG:7854 (2D GDA2020 / MGA Zone 54).
    Coordinates are not changed, only the SRS tag.
    """
    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS not found: {input_las}")

    # Create output filename
    output_las = input_las.with_name(input_las.stem + "_7854.las")

    cmd = [
        "pdal", "translate",
        str(input_las),
        str(output_las),
        "--writers.las.a_srs=EPSG:7854"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Rewritten SRS to EPSG:7854 → {output_las}")

    return output_las

if __name__ == "__main__":
    las_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")
    rewrite_srs_to_7854(las_path)

# %%
#!/usr/bin/env python3
"""
Reproject LAS with PDAL from EPSG:4326+5711 → EPSG:7854+5711
"""

import subprocess
from pathlib import Path

def reproject_to_gda2020_zone54(input_las: Path):
    """
    Reproject LAS from EPSG:4326+5711 (WGS84 + AHD height)
    to EPSG:7854+5711 (GDA2020 / MGA Zone 54 + AHD height).
    """
    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS not found: {input_las}")

    # Output filename with suffix
    output_las = input_las.with_name(input_las.stem + "_gda2020z54.las")

    cmd = [
        "pdal", "translate",
        str(input_las),
        str(output_las),
        "reprojection",
        '--filters.reprojection.in_srs=EPSG:4326+5711',
        '--filters.reprojection.out_srs=EPSG:7854+5711'
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Reprojected → {output_las} with out_srs=EPSG:7854+5711")

    return output_las


if __name__ == "__main__":
    las_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")
    reproject_to_gda2020_zone54(las_path)

# %%
##test
#!/usr/bin/env python3
import os, subprocess
from pathlib import Path

# === Input LAS/LAZ ===
lidar_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")

# === Output report ===
output_info_txt = lidar_path.with_suffix(".lasinfo.txt")

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

print("Running lasinfo64…")

# 1) Ask lasinfo64 to write the report itself (-o)
#    (This avoids relying on stdout capture.)
cmd_file = ["/home/laura/lastools/bin/lasinfo64",
            "-i", str(lidar_path),
            "-o", str(output_info_txt)]
p1 = subprocess.run(cmd_file, capture_output=True, text=True, env=env)

# 2) If the file is empty or missing, try capturing stdout/stderr and write that
if (p1.returncode != 0) or (not output_info_txt.exists()) or (output_info_txt.stat().st_size == 0):
    print("Report empty via -o; retrying by capturing output…")
    cmd_cap = ["/home/laura/lastools/bin/lasinfo64", "-i", str(lidar_path)]
    p2 = subprocess.run(cmd_cap, capture_output=True, text=True, env=env)
    if p2.returncode != 0:
        print("lasinfo64 failed!\n" + p2.stderr)
        raise SystemExit(1)
    # Prefer stdout; if empty, write stderr (some builds log to stderr)
    data = p2.stdout if p2.stdout.strip() else p2.stderr
    output_info_txt.parent.mkdir(parents=True, exist_ok=True)
    output_info_txt.write_text(data)

print(f"✅ Report saved: {output_info_txt}  ({output_info_txt.stat().st_size} bytes)")

# --- Print a concise summary from the saved report ---
crs_line = None
min_line = None
max_line = None
for line in output_info_txt.read_text().splitlines():
    low = line.lower().strip()
    if (crs_line is None) and ("coordinate system" in low):
        crs_line = line.strip()
    if (min_line is None) and low.startswith("min x y z"):
        min_line = line.strip()
    if (max_line is None) and low.startswith("max x y z"):
        max_line = line.strip()

print("\n# ===== LAS Summary (lasinfo64) =====")
print("File :", lidar_path)
print("CRS  :", crs_line or "<no coordinate system line found>")
if min_line: print(min_line)
if max_line: print(max_line)
print("# ==================================")

# %%
##add crs
#!/usr/bin/env python3
"""
LAStools-only workflow:
1) Read LAS with lasinfo64 (save report)
2) Write CRS into header with las2las64 (-epsg 7854) -> new LAS
3) Verify new LAS with lasinfo64 (save report + print concise summary)

Notes:
- This sets HORIZONTAL CRS to EPSG:7854 (GDA2020 / MGA Zone 54).
- Vertical AHD tagging (+5711) is not added here (LAStools mainly supports horizontal EPSG tagging).
- Coordinates are NOT changed; only metadata is written.
"""

import os
import subprocess
from pathlib import Path

# === Input LAS ===
in_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")

# === Outputs ===
in_report = in_las.with_suffix(".lasinfo.txt")
out_las = in_las.with_name(in_las.stem + "_crs7854.las")
out_report = out_las.with_suffix(".lasinfo.txt")

# === LAStools paths/env ===
LASINFO = "/home/laura/lastools/bin/lasinfo64"
LAS2LAS = "/home/laura/lastools/bin/las2las64"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=env)

# 1) Base lasinfo (original file)
print("📊 lasinfo64 (input)…")
r1 = run([LASINFO, "-i", str(in_las)])
if r1.returncode != 0:
    print("❌ lasinfo64 failed on input:\n" + r1.stderr)
    raise SystemExit(1)
in_report.parent.mkdir(parents=True, exist_ok=True)
in_report.write_text(r1.stdout)
print(f"✅ Saved input report: {in_report}  ({in_report.stat().st_size} bytes)")

# 2) Write CRS with las2las (-epsg 7854)
print("🛠️ Writing CRS with las2las64 (-epsg 7854)…")
# -keep-xyzi ensures all standard dimensions are preserved; adjust if you use extras
r2 = run([LAS2LAS, "-i", str(in_las), "-o", str(out_las), "-epsg", "7854"])
if r2.returncode != 0:
    print("❌ las2las64 failed:\n" + r2.stderr)
    raise SystemExit(1)
print(f"✅ Wrote CRS to: {out_las}")

# 3) lasinfo on the new file (verify)
print("📊 lasinfo64 (output)…")
r3 = run([LASINFO, "-i", str(out_las)])
if r3.returncode != 0:
    print("❌ lasinfo64 failed on output:\n" + r3.stderr)
    raise SystemExit(1)
out_report.write_text(r3.stdout)
print(f"✅ Saved output report: {out_report}  ({out_report.stat().st_size} bytes)")

# 4) Concise summary from the new report
crs_line = None
min_line = None
max_line = None
for line in r3.stdout.splitlines():
    low = line.lower().strip()
    if crs_line is None and "coordinate system" in low:
        crs_line = line.strip()
    if min_line is None and low.startswith("min x y z"):
        min_line = line.strip()
    if max_line is None and low.startswith("max x y z"):
        max_line = line.strip()

print("\n# ===== LAS Summary (after tagging) =====")
print("File :", out_las)
print("CRS  :", crs_line or "<no coordinate system line found>")
if min_line: print(min_line)
if max_line: print(max_line)
print("# =======================================")

# %%
###trial 2 for previous
#!/usr/bin/env python3
"""
Force a clean 2D CRS (EPSG:7854) in a LAS/LAZ header using a PDAL pipeline.
- No reprojection (coords are unchanged) – only the SRS metadata is written.
- Strips any CompoundCRS/AHD tagging; result is pure PROJCS EPSG:7854.
- Saves the pipeline JSON and verifies with `pdal info --metadata`.
"""

import json
import shutil
import subprocess
from pathlib import Path

# ==== EDIT THIS INPUT PATH IF NEEDED ====
INPUT = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")

# Outputs
OUT = INPUT.with_name(INPUT.stem + "_2d_7854.las")
PIPE = INPUT.with_name(INPUT.stem + "_force_7854.pipeline.json")

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

# Primary pipeline: write EPSG:7854 and forward header/VLRs if supported
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": str(INPUT)},
        {
            "type": "writers.las",
            "filename": str(OUT),
            "a_srs": "EPSG:7854",
            # Try to forward existing header/VLRs without touching coords
            "forward": "all"
        }
    ]
}

# Save and run
PIPE.write_text(json.dumps(pipeline, indent=2))
print(f"📄 Pipeline saved: {PIPE}")

print("🛠️ Running PDAL pipeline…")
p = run(["pdal", "pipeline", str(PIPE)])
if p.returncode != 0:
    # Fallback for older PDAL builds that don't support "forward"
    print("⚠️ Pipeline failed (likely due to 'forward'). Retrying without it…")
    pipeline["pipeline"][1].pop("forward", None)
    PIPE.write_text(json.dumps(pipeline, indent=2))
    p = run(["pdal", "pipeline", str(PIPE)])
    if p.returncode != 0:
        print("❌ PDAL pipeline failed:\n" + p.stderr)
        raise SystemExit(1)

print(f"✅ Wrote: {OUT}")

# Verify SRS (no jq required)
print("\n🔎 Verifying SRS in output metadata:")
info = run(["pdal", "info", str(OUT), "--metadata"])
if info.returncode != 0:
    print("⚠️ pdal info failed:\n" + info.stderr)
    raise SystemExit(1)

# Print the "srs" block crudely
lines = info.stdout.splitlines()
printing = False
for line in lines:
    if '"srs"' in line:
        printing = True
    if printing:
        print(line)
    if printing and line.strip() == "}":
        break

# Optional: also show via LAStools if available
lasinfo = shutil.which("/home/laura/lastools/bin/lasinfo64")
if lasinfo:
    print("\n📊 lasinfo64 check (WKT/PROJ.4):")
    v = run([lasinfo, "-i", str(OUT), "-report_wkt", "-report_proj4"])
    print(v.stdout if v.returncode == 0 else v.stderr)

# %%
