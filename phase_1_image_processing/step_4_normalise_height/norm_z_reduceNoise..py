#%%
#!/usr/bin/env python3
import subprocess
import os

# Paths
input_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.109_HAG_delaunay.las"
output_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/triple_AusGeoid_tiles_multispectral.109_HAG_delaunay_noiseout.laz"

# Set environment (needed if LAStools uses shared libs)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

# Build command
cmd = [
    "/home/laura/lastools/bin/las2las64",  # full path to binary
    "-i", input_las,
    "-o", output_las,
    "-keep_class", "0"
]

print("🚀 Running:", " ".join(cmd))

res = subprocess.run(cmd, text=True, capture_output=True, env=env)

# Check success
if res.returncode != 0:
    print("----- STDOUT -----\n", res.stdout)
    print("----- STDERR -----\n", res.stderr)
    raise SystemExit(res.returncode)

print("✅ Wrote noise-reduced LAS:", output_las)

# %%
