# %%
from pathlib import Path
import numpy as np
import laspy

# las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")  # or .laz
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/lidar/clipped/cloudf25e6b1a8f18f0e.las")

with laspy.open(las_path) as reader:
    hdr = reader.header
    dims_available = set(hdr.point_format.dimension_names)

    print(f"File: {las_path.name}")
    print(f"Point format: {hdr.point_format}")
    print(f"Count points (from header): {hdr.point_count}")
    print(f"Has RGB? { {'red','green','blue'}.issubset(dims_available) }")

    # Required fields for discrete-return checks
    required = {"X", "Y", "Z", "return_number", "number_of_returns"}
    missing = required - dims_available
    if missing:
        print(f"❌ Missing required dims: {sorted(missing)}")
    else:
        print("✅ Discrete-return fields present")

    # Accumulate unique values across the whole file in chunks
    rn_uniques, nr_uniques = set(), set()
    total = 0
    first_scaled_xyz = None

    for i, pts in enumerate(reader.chunk_iterator(5_000_000)):  # adjust chunk size as needed
        total += len(pts)

        # Unique return stats
        if "return_number" in pts.point_format.dimension_names:
            rn_uniques.update(np.unique(pts["return_number"]).tolist())
        if "number_of_returns" in pts.point_format.dimension_names:
            nr_uniques.update(np.unique(pts["number_of_returns"]).tolist())

        # Grab a few scaled XYZ examples from the first chunk only
        if first_scaled_xyz is None:
            # Convert integer coords to scaled world coords using header scale/offset
            X = pts.X.astype(np.float64) * hdr.scales[0] + hdr.offsets[0]
            Y = pts.Y.astype(np.float64) * hdr.scales[1] + hdr.offsets[1]
            Z = pts.Z.astype(np.float64) * hdr.scales[2] + hdr.offsets[2]
            n_show = min(5, len(pts))
            first_scaled_xyz = list(zip(X[:n_show], Y[:n_show], Z[:n_show]))

    print(f"Total points (iterated): {total}")
    if rn_uniques:
        print("Unique return_number:", sorted(rn_uniques))
    if nr_uniques:
        print("Unique number_of_returns:", sorted(nr_uniques))

    has_multi = any(v > 1 for v in nr_uniques)
    print("Has multi-return pulses?", has_multi)

    if first_scaled_xyz is not None:
        print("Example scaled XYZ (first 5 from first chunk):")
        for row in first_scaled_xyz:
            print(row)


# %%
from pathlib import Path
import numpy as np
import laspy

# === Config ===
# las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/lidar/clipped/cloudf25e6b1a8f18f0e.las")
voxel = 0.10  # 10 cm grid

# Provide your grid bounds (match your model cube)
xmin, ymin, zmin = 473000.0, 6235600.0, 25.0
xmax, ymax, zmax = 473400.0, 6236000.0, 55.0

nx = int(np.ceil((xmax - xmin) / voxel))
ny = int(np.ceil((ymax - ymin) / voxel))
nz = int(np.ceil((zmax - zmin) / voxel))

hits   = np.zeros((nx, ny, nz), dtype=np.uint32)
misses = np.zeros((nx, ny, nz), dtype=np.uint32)
# occluded = np.zeros((nx, ny, nz), dtype=np.uint32)  # optional

def in_bounds(v):
    ix, iy, iz = v
    return (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz)

def world_to_idx(p):
    ix = int((p[0] - xmin) / voxel)
    iy = int((p[1] - ymin) / voxel)
    iz = int((p[2] - zmin) / voxel)
    return (ix, iy, iz)

def traverse_voxels(origin, end):
    """
    Amanatides–Woo DDA over a segment origin->end.
    Yields voxel indices crossed before the end voxel.
    Implemented minimally; consider numba for speed.
    """
    o = np.asarray(origin, float)
    e = np.asarray(end, float)
    d = e - o
    if np.allclose(d, 0):
        return

    ix, iy, iz = world_to_idx(o)
    ex, ey, ez = world_to_idx(e)
    if (ix,iy,iz) == (ex,ey,ez):
        return

    step_x = 1 if d[0] > 0 else (-1 if d[0] < 0 else 0)
    step_y = 1 if d[1] > 0 else (-1 if d[1] < 0 else 0)
    step_z = 1 if d[2] > 0 else (-1 if d[2] < 0 else 0)

    def boundary(minv, i, step):
        return np.inf if step == 0 else (minv + (i + (1 if step > 0 else 0)) * voxel)

    # Initial tMax and tDelta per axis
    if step_x != 0:
        bx = boundary(xmin, ix, step_x)
        tMaxX = (bx - o[0]) / d[0]
        tDeltaX = abs(voxel / d[0])
    else:
        tMaxX, tDeltaX = np.inf, np.inf

    if step_y != 0:
        by = boundary(ymin, iy, step_y)
        tMaxY = (by - o[1]) / d[1]
        tDeltaY = abs(voxel / d[1])
    else:
        tMaxY, tDeltaY = np.inf, np.inf

    if step_z != 0:
        bz = boundary(zmin, iz, step_z)
        tMaxZ = (bz - o[2]) / d[2]
        tDeltaZ = abs(voxel / d[2])
    else:
        tMaxZ, tDeltaZ = np.inf, np.inf

    while (ix,iy,iz) != (ex,ey,ez) and in_bounds((ix,iy,iz)):
        yield (ix,iy,iz)
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            ix += step_x; tMaxX += tDeltaX
        elif tMaxY <= tMaxX and tMaxY <= tMaxZ:
            iy += step_y; tMaxY += tDeltaY
        else:
            iz += step_z; tMaxZ += tDeltaZ

# Stub: supply sensor origins. Replace with your GNSS/INS lookup.
def get_origin_for_gpstime(gps_time_array):
    """
    Return an (N,3) array of sensor origins matched to each point’s gps_time.
    MVP: return a fixed origin or nearest-neighbour from a pose track.
    """
    # TODO: implement properly. For now, use a dummy fixed origin above the area.
    o = np.array([ (xmin+xmax)/2, (ymin+ymax)/2, zmax + 5.0 ], dtype=float)
    return np.repeat(o[None, :], len(gps_time_array), axis=0)

# === Scan the LAS in chunks and accumulate ray stats ===
with laspy.open(las_path) as reader:
    hdr = reader.header
    for pts in reader.chunk_iterator(5_000_000):
        # required dims exist (checked earlier)
        rn = np.asarray(pts["return_number"])
        nr = np.asarray(pts["number_of_returns"])
        gps = np.asarray(pts["gps_time"])  # used to fetch origins

        # Scale to world coords
        X = pts.X.astype(np.float64) * hdr.scales[0] + hdr.offsets[0]
        Y = pts.Y.astype(np.float64) * hdr.scales[1] + hdr.offsets[1]
        Z = pts.Z.astype(np.float64) * hdr.scales[2] + hdr.offsets[2]

        # terminal hits (last returns): return_number == number_of_returns
        mask_last = (rn == nr)
        if not np.any(mask_last):
            continue

        H = np.vstack([X[mask_last], Y[mask_last], Z[mask_last]]).T  # (M,3)
        O = get_origin_for_gpstime(gps[mask_last])                    # (M,3)

        # Accumulate misses and hits
        for o, h in zip(O, H):
            end_idx = world_to_idx(h)
            if not in_bounds(end_idx):
                continue
            for v in traverse_voxels(o, h):  # voxels before the hit voxel
                if in_bounds(v):
                    misses[v] += 1
            hits[end_idx] += 1

            # OPTIONAL: mark occluded voxels beyond last return (skipped in MVP)
            # for v in traverse_beyond(h, o): occluded[v] += 1

# === Gap fraction ===
den = hits + misses
gap_fraction = np.where(den > 0, misses / den, 0.0).astype(np.float32)
observed_mask = (den > 0).astype(np.uint8)

print("Done. Arrays:")
print(" hits:", hits.shape, hits.dtype)
print(" misses:", misses.shape, misses.dtype)
print(" gap_fraction:", gap_fraction.shape, gap_fraction.dtype)
print(" observed_mask:", observed_mask.shape, observed_mask.dtype)

# Save if you like
# np.save("/path/to/gap_fraction_10cm.npy", gap_fraction)
# np.save("/path/to/observed_mask_10cm.npy", observed_mask)

# %%
from pathlib import Path
import numpy as np
import laspy

# ========== CONFIG ==========
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/lidar/clipped/cloudf25e6b1a8f18f0e.las")
voxel = 0.10  # 10 cm
USE_HEADER_BOUNDS = True  # set False to use manual bounds below

# Manual bounds (used only if USE_HEADER_BOUNDS=False)
manual_bounds = dict(
    xmin=473000.0, ymin=6235600.0, zmin=25.0,
    xmax=473400.0, ymax=6236000.0, zmax=55.0
)

# ========== GRID SETUP ==========
if USE_HEADER_BOUNDS:
    with laspy.open(las_path) as r:
        h = r.header
        xmin, ymin, zmin = h.mins
        xmax, ymax, zmax = h.maxs
else:
    xmin, ymin, zmin = manual_bounds["xmin"], manual_bounds["ymin"], manual_bounds["zmin"]
    xmax, ymax, zmax = manual_bounds["xmax"], manual_bounds["ymax"], manual_bounds["zmax"]

# Small padding if you want (0.0 keeps exact LAS extents)
pad = 0.0
xmin, ymin, zmin = xmin - pad, ymin - pad, zmin - pad
xmax, ymax, zmax = xmax + pad, ymax + pad, zmax + pad

nx = int(np.ceil((xmax - xmin) / voxel))
ny = int(np.ceil((ymax - ymin) / voxel))
nz = int(np.ceil((zmax - zmin) / voxel))

hits   = np.zeros((nx, ny, nz), dtype=np.uint32)
misses = np.zeros((nx, ny, nz), dtype=np.uint32)
# occluded = np.zeros((nx, ny, nz), dtype=np.uint32)  # optional feature (not used in MVP)

# Precompute AABB arrays
bmin = np.array([xmin, ymin, zmin], dtype=float)
bmax = np.array([xmax, ymax, zmax], dtype=float)

# ========== HELPERS ==========
def in_bounds(v):
    x, y, z = v
    return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)

def world_to_idx(p):
    return (
        int((p[0] - xmin) / voxel),
        int((p[1] - ymin) / voxel),
        int((p[2] - zmin) / voxel),
    )

def segment_aabb_entry_exit(o, e, bmin, bmax):
    """
    Slab test for segment o->e against axis-aligned box [bmin, bmax].
    Returns (t_enter, t_exit) in [0,1], or (None, None) if no intersection.
    """
    d = e - o
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if d[i] == 0.0:
            if o[i] < bmin[i] or o[i] > bmax[i]:
                return None, None
        else:
            inv = 1.0 / d[i]
            t1 = (bmin[i] - o[i]) * inv
            t2 = (bmax[i] - o[i]) * inv
            t_near = t1 if t1 < t2 else t2
            t_far  = t2 if t1 < t2 else t1
            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)
            if tmin > tmax:
                return None, None
    return tmin, tmax

def traverse_voxels(origin, end):
    """
    Amanatides–Woo voxel traversal for segment origin->end.
    Yields voxel indices crossed BEFORE the end voxel.
    """
    o = np.asarray(origin, float)
    e = np.asarray(end, float)
    d = e - o
    if np.allclose(d, 0.0):
        return

    ix, iy, iz = world_to_idx(o)
    ex, ey, ez = world_to_idx(e)
    if (ix, iy, iz) == (ex, ey, ez):
        return

    sx = 1 if d[0] > 0 else (-1 if d[0] < 0 else 0)
    sy = 1 if d[1] > 0 else (-1 if d[1] < 0 else 0)
    sz = 1 if d[2] > 0 else (-1 if d[2] < 0 else 0)

    def boundary(minv, i, s):
        return np.inf if s == 0 else (minv + (i + (1 if s > 0 else 0)) * voxel)

    if sx != 0:
        bx = boundary(xmin, ix, sx)
        tMaxX = (bx - o[0]) / d[0]
        tDeltaX = abs(voxel / d[0])
    else:
        tMaxX = tDeltaX = np.inf

    if sy != 0:
        by = boundary(ymin, iy, sy)
        tMaxY = (by - o[1]) / d[1]
        tDeltaY = abs(voxel / d[1])
    else:
        tMaxY = tDeltaY = np.inf

    if sz != 0:
        bz = boundary(zmin, iz, sz)
        tMaxZ = (bz - o[2]) / d[2]
        tDeltaZ = abs(voxel / d[2])
    else:
        tMaxZ = tDeltaZ = np.inf

    while (ix, iy, iz) != (ex, ey, ez) and in_bounds((ix, iy, iz)):
        yield (ix, iy, iz)
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            ix += sx; tMaxX += tDeltaX
        elif tMaxY <= tMaxX and tMaxY <= tMaxZ:
            iy += sy; tMaxY += tDeltaY
        else:
            iz += sz; tMaxZ += tDeltaZ

# Stub: replace with your GNSS/INS nearest-neighbour pose lookup (+ lever-arm/boresight)
def get_origin_for_gpstime(gps_time_array):
    """
    Returns (N,3) sensor origins matched to each point's gps_time.
    MVP stub: fixed origin above scene center (replace with real pose).
    """
    o = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, zmax + 5.0], float)
    return np.repeat(o[None, :], len(gps_time_array), axis=0)

# ========== MAIN PASS ==========
used_beams = 0
skipped_no_intersect = 0
skipped_end_outside = 0

with laspy.open(las_path) as reader:
    hdr = reader.header
    for pts in reader.chunk_iterator(1_000_000):
        rn  = np.asarray(pts["return_number"])
        nr  = np.asarray(pts["number_of_returns"])
        gps = np.asarray(pts["gps_time"])

        # scale integer coordinates -> world coords
        X = pts.X.astype(np.float64) * hdr.scales[0] + hdr.offsets[0]
        Y = pts.Y.astype(np.float64) * hdr.scales[1] + hdr.offsets[1]
        Z = pts.Z.astype(np.float64) * hdr.scales[2] + hdr.offsets[2]

        # last returns per pulse
        mask_last = (rn == nr)
        if not np.any(mask_last):
            continue

        H = np.stack([X[mask_last], Y[mask_last], Z[mask_last]], axis=1)  # (M,3)
        O = get_origin_for_gpstime(gps[mask_last])                        # (M,3)

        for o, hpt in zip(O, H):
            # Require last-return voxel inside the grid (MVP rule)
            end_idx = world_to_idx(hpt)
            if not in_bounds(end_idx):
                skipped_end_outside += 1
                continue

            # Intersect segment with grid AABB; start at the entry point
            t_enter, t_exit = segment_aabb_entry_exit(o, hpt, bmin, bmax)
            if t_enter is None:
                skipped_no_intersect += 1
                continue

            eps = 1e-6
            start = o + (t_enter + eps) * (hpt - o)

            # March until the end voxel; accumulate misses, then hit
            for v in traverse_voxels(start, hpt):
                if in_bounds(v):
                    misses[v] += 1
            hits[end_idx] += 1
            used_beams += 1

print(f"Beams used: {used_beams}")
print(f"Skipped (no box intersection): {skipped_no_intersect}")
print(f"Skipped (end outside grid): {skipped_end_outside}")

# ========== GAP FRACTION ==========
den = hits + misses
gap_fraction = np.zeros_like(den, dtype=np.float32)
np.divide(misses, den, out=gap_fraction, where=(den > 0))  # safe divide, no warnings
observed_mask = (den > 0).astype(np.uint8)

print("Done. Arrays:")
print(" hits:", hits.shape, hits.dtype, "sum:", int(hits.sum()))
print(" misses:", misses.shape, misses.dtype, "sum:", int(misses.sum()))
print(" gap_fraction:", gap_fraction.shape, gap_fraction.dtype, "range:", float(gap_fraction.min()), float(gap_fraction.max()))
print(" observed_mask:", observed_mask.shape, observed_mask.dtype, "count:", int(observed_mask.sum()))

# Optional: save outputs
np.save("/media/laura/laura_usb/uas_data/DD0012/gap_fraction_10cm.npy", gap_fraction)
np.save("/media/laura/laura_usb/uas_data/DD0012/observed_mask_10cm.npy", observed_mask)

# %%
# 1) Show the middle Y-slice (X–Z cross-section)
import numpy as np
import matplotlib.pyplot as plt

gap = np.load("/media/laura/laura_usb/uas_data/DD0012/gap_fraction_10cm.npy")         # (nx, ny, nz)
obs = np.load("/media/laura/laura_usb/uas_data/DD0012/observed_mask_10cm.npy").astype(bool)

nx, ny, nz = gap.shape
y_mid = ny // 2

slice_xz = gap[:, y_mid, :].T                  # (nz, nx) → plot Z vs X
slice_obs = obs[:, y_mid, :].T

plt.figure(figsize=(8,5))
plt.imshow(np.where(slice_obs, slice_xz, np.nan), 
           origin="lower", aspect="auto")
plt.colorbar(label="Gap fraction (0=blocked, 1=open)")
plt.xlabel("X index")
plt.ylabel("Z index (height)")
plt.title(f"Vertical cross-section at Y={y_mid} (X–Z plane)")
plt.tight_layout()
plt.show()

# %%
# 2) Show the middle X-slice (Y–Z cross-section)
x_mid = nx // 2
slice_yz = gap[x_mid, :, :].T                  # (nz, ny)
slice_obs = obs[x_mid, :, :].T

plt.figure(figsize=(8,5))
plt.imshow(np.where(slice_obs, slice_yz, np.nan),
           origin="lower", aspect="auto")
plt.colorbar(label="Gap fraction")
plt.xlabel("Y index")
plt.ylabel("Z index (height)")
plt.title(f"Vertical cross-section at X={x_mid} (Y–Z plane)")
plt.tight_layout()
plt.show()

# %%
# 3) Average vertical profile (mean gap by height)
# Mean over X,Y only where observed
num = np.sum(gap, axis=(0,1), where=obs, initial=0.0)
den = np.sum(obs, axis=(0,1))                 # count of observed voxels per Z
mean_gap_z = np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den>0)

plt.figure(figsize=(5,6))
plt.plot(mean_gap_z, np.arange(gap.shape[2]))
plt.gca().invert_yaxis()  # optional: put ground at bottom, higher Z up
plt.xlabel("Mean gap fraction")
plt.ylabel("Z index (height)")
plt.title("Vertical mean gap fraction profile")
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# --- inputs ---
voxel_m = 0.10          # 10 cm voxels
zmin_m  = 0.0           # <-- put your grid's zmin here (meters, same CRS units)

gap = np.load("/media/laura/laura_usb/uas_data/DD0012/gap_fraction_10cm.npy")         # (nx, ny, nz)
obs = np.load("/media/laura/laura_usb/uas_data/DD0012/observed_mask_10cm.npy").astype(bool)


# Mean gap per height slice (Z)
num = np.sum(gap, axis=(0,1), where=obs, initial=0.0)
den = np.sum(obs, axis=(0,1))
mean_gap_z = np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den>0)

# Convert Z index -> height in meters
z_idx = np.arange(gap.shape[2])
height_m = zmin_m + z_idx * voxel_m

# Plot with height increasing upward
plt.figure(figsize=(5,6))
plt.plot(mean_gap_z, height_m)
plt.xlabel("Mean gap fraction (0=blocked, 1=open)")
plt.ylabel("Height (m)")
plt.title("Vertical mean gap fraction profile")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
# foliage density profile
import numpy as np
import matplotlib.pyplot as plt

# --- inputs ---
voxel_m = 0.10
zmin_m  = 0.0  # set to your grid zmin if you want absolute elevation


gap = np.load("/media/laura/laura_usb/uas_data/DD0012/gap_fraction_10cm.npy")         # (nx, ny, nz)
obs = np.load("/media/laura/laura_usb/uas_data/DD0012/observed_mask_10cm.npy").astype(bool)


# Mean gap per height (only over observed voxels)
num = np.sum(gap, axis=(0,1), where=obs, initial=0.0)
den = np.sum(obs, axis=(0,1))
mean_gap_z = np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den>0)

# Foliage "density" proxy
foliage_density = 1.0 - mean_gap_z

# Height axis in meters
z_idx = np.arange(gap.shape[2])
height_m = zmin_m + z_idx * voxel_m

# Optional: light smoothing (uncomment to use)
# from scipy.ndimage import gaussian_filter1d
# foliage_density = gaussian_filter1d(foliage_density, sigma=1.0)

plt.figure(figsize=(5,6))
plt.plot(foliage_density, height_m)
plt.xlabel("Foliage density proxy (1 - gap fraction)")
plt.ylabel("Height (m)")
plt.title("Vertical foliage density profile")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
# Step 3 — Voxelisation grid → compute gap_fraction on the Cartesian grid
# step3_gap_fraction.py
from pathlib import Path
import numpy as np
import laspy

# === config ===
LAS_PATH = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site3_1_DD0012/lidar/clipped/cloudf25e6b1a8f18f0e.las")
VOXEL = 0.10  # 10 cm

# --- helpers ---
def world_to_idx(p, xmin, ymin, zmin, voxel):  # p: (3,)
    return (int((p[0]-xmin)/voxel), int((p[1]-ymin)/voxel), int((p[2]-zmin)/voxel))

def in_bounds(v, nx, ny, nz):
    x,y,z=v; return 0<=x<nx and 0<=y<ny and 0<=z<nz

def segment_aabb_entry_exit(o, e, bmin, bmax):
    d = e - o; tmin, tmax = 0.0, 1.0
    for i in range(3):
        if d[i] == 0.0:
            if o[i] < bmin[i] or o[i] > bmax[i]: return None, None
        else:
            inv = 1.0/d[i]; t1 = (bmin[i]-o[i])*inv; t2 = (bmax[i]-o[i])*inv
            t_near, t_far = (t1, t2) if t1 < t2 else (t2, t1)
            tmin, tmax = max(tmin, t_near), min(tmax, t_far)
            if tmin > tmax: return None, None
    return tmin, tmax

def traverse_voxels(o, e, xmin, ymin, zmin, voxel, nx, ny, nz):
    o = np.asarray(o, float); e = np.asarray(e, float); d = e - o
    if np.allclose(d, 0.0): return
    ix,iy,iz = world_to_idx(o, xmin, ymin, zmin, voxel)
    ex,ey,ez = world_to_idx(e, xmin, ymin, zmin, voxel)
    if (ix,iy,iz) == (ex,ey,ez): return
    sx = 1 if d[0]>0 else (-1 if d[0]<0 else 0)
    sy = 1 if d[1]>0 else (-1 if d[1]<0 else 0)
    sz = 1 if d[2]>0 else (-1 if d[2]<0 else 0)
    def b(minv,i,s): return np.inf if s==0 else (minv + (i + (1 if s>0 else 0))*voxel)
    if sx: bx=b(xmin,ix,sx); tMaxX=(bx-o[0])/d[0]; tDeltaX=abs(voxel/d[0])
    else:  tMaxX=tDeltaX=np.inf
    if sy: by=b(ymin,iy,sy); tMaxY=(by-o[1])/d[1]; tDeltaY=abs(voxel/d[1])
    else:  tMaxY=tDeltaY=np.inf
    if sz: bz=b(zmin,iz,sz); tMaxZ=(bz-o[2])/d[2]; tDeltaZ=abs(voxel/d[2])
    else:  tMaxZ=tDeltaZ=np.inf
    while (ix,iy,iz)!=(ex,ey,ez) and in_bounds((ix,iy,iz), nx,ny,nz):
        yield (ix,iy,iz)
        if tMaxX<=tMaxY and tMaxX<=tMaxZ: ix+=sx; tMaxX+=tDeltaX
        elif tMaxY<=tMaxX and tMaxY<=tMaxZ: iy+=sy; tMaxY+=tDeltaY
        else: iz+=sz; tMaxZ+=tDeltaZ

def get_origin_for_gpstime(gps_time_array, xmin, ymin, zmax):  # stub; replace with UAV pose
    o = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, zmax+5.0], float)
    return np.repeat(o[None,:], len(gps_time_array), axis=0)

# --- main ---
with laspy.open(LAS_PATH) as r:
    h = r.header
    xmin, ymin, zmin = h.mins; xmax, ymax, zmax = h.maxs
    nx = int(np.ceil((xmax-xmin)/VOXEL)); ny = int(np.ceil((ymax-ymin)/VOXEL)); nz = int(np.ceil((zmax-zmin)/VOXEL))
    hits   = np.zeros((nx,ny,nz), np.uint32)
    misses = np.zeros((nx,ny,nz), np.uint32)
    bmin = np.array([xmin,ymin,zmin], float); bmax = np.array([xmax,ymax,zmax], float)

    used = 0
    for pts in r.chunk_iterator(1_000_000):
        rn = np.asarray(pts["return_number"]); nr = np.asarray(pts["number_of_returns"]); gps = np.asarray(pts["gps_time"])
        X = pts.X.astype(np.float64)*h.scales[0] + h.offsets[0]
        Y = pts.Y.astype(np.float64)*h.scales[1] + h.offsets[1]
        Z = pts.Z.astype(np.float64)*h.scales[2] + h.offsets[2]
        mask_last = (rn == nr)
        if not np.any(mask_last): continue
        H = np.stack([X[mask_last], Y[mask_last], Z[mask_last]], axis=1)
        O = get_origin_for_gpstime(gps[mask_last], xmin, ymin, zmax)

        for o, hpt in zip(O, H):
            end_idx = world_to_idx(hpt, xmin, ymin, zmin, VOXEL)
            if not in_bounds(end_idx, nx,ny,nz): continue
            t_enter, t_exit = segment_aabb_entry_exit(o, hpt, bmin, bmax)
            if t_enter is None: continue
            start = o + (t_enter + 1e-6) * (hpt - o)
            for v in traverse_voxels(start, hpt, xmin, ymin, zmin, VOXEL, nx,ny,nz):
                misses[v] += 1
            hits[end_idx] += 1
            used += 1

den = hits + misses
gap_fraction = np.zeros_like(den, dtype=np.float32)
np.divide(misses, den, out=gap_fraction, where=(den > 0))
observed_mask = (den > 0).astype(np.uint8)

np.save("gap_fraction_10cm.npy", gap_fraction)
np.save("observed_mask_10cm.npy", observed_mask)
print("Beams used:", used, "| gap saved:", gap_fraction.shape)

# %%
# step4_structure_feats.py
import numpy as np

GAP_PATH = "gap_fraction_10cm.npy"
OBS_PATH = "observed_mask_10cm.npy"
VOXEL_M  = 0.10  # 10 cm

gap = np.load(GAP_PATH)              # (nx, ny, nz)
obs = np.load(OBS_PATH).astype(bool) # same shape

foliage = 1.0 - gap                  # 0..1, per-voxel foliage density proxy

# --- per-column H95 (95% cumulative foliage height) ---
nx, ny, nz = gap.shape
z = np.arange(nz) * VOXEL_M          # height in meters relative to zmin

H95 = np.full((nx, ny), np.nan, dtype=np.float32)
HmaxD = np.full((nx, ny), np.nan, dtype=np.float32)

for i in range(nx):
    col_f = foliage[i]               # (ny, nz)
    col_o = obs[i]                   # (ny, nz)
    # mask unobserved
    col_f = np.where(col_o, col_f, 0.0)

    # cumulative by height
    csum = col_f.cumsum(axis=1)      # (ny, nz)
    total = csum[:, -1]              # (ny,)
    # avoid zero-columns
    valid = total > 0

    # H95
    targets = 0.95 * total[valid]    # (ny_valid,)
    idxs = np.argmax(csum[valid] >= targets[:, None], axis=1)  # arg where cum>=95%
    H95[i, valid] = z[idxs]

    # HmaxD: height of maximum density (argmax along z)
    idxs2 = np.argmax(col_f[valid], axis=1)
    HmaxD[i, valid] = z[idxs2]

# --- bandwise densities (broadcast along Z so they’re 3D channels) ---
bands_m = [(0.0, 0.5), (0.5, 2.0), (2.0, 99.0)]
band_channels = []
for lo, hi in bands_m:
    mask = (z >= lo) & (z < hi)
    # mean foliage where observed, per (x,y)
    num = foliage[:, :, mask].sum(axis=2)
    den = obs[:, :, mask].sum(axis=2)
    mean_band = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)  # (nx, ny)
    # broadcast to Z for easy stacking with 3D inputs
    band_channels.append(np.repeat(mean_band[:, :, None], nz, axis=2).astype(np.float32))

# save
np.save("foliage_density.npy", foliage.astype(np.float32))  # 1-gap
np.save("H95_map.npy", H95)        # (nx, ny)
np.save("HmaxD_map.npy", HmaxD)    # (nx, ny)
np.save("band_density_stack.npy", np.stack(band_channels, axis=0))  # (3, nx, ny, nz)
print("Saved: foliage, H95/HmaxD (2D), band densities (3, nx, ny, nz)")


#%%
# step5_tensor_prep.py
import numpy as np
import torch
import torch.nn.functional as F

# --- load your existing inputs ---
# X_spec: (5, nx, ny, nz) spectral channels (already scaled)
# X_height: (1, nx, ny, nz) mean normalized height per voxel (0..1)
X_spec   = np.load("X_spectral.npy")        # <- your file
X_height = np.load("X_mean_norm_height.npy")

# --- new occlusion features ---
gap      = np.load("gap_fraction_10cm.npy")             # (nx, ny, nz)
obs      = np.load("observed_mask_10cm.npy").astype(bool)
foliage  = np.load("foliage_density.npy")               # 1-gap, (nx, ny, nz)
bands    = np.load("band_density_stack.npy")            # (3, nx, ny, nz)

# If your model grid differs (e.g., 5 cm), resample here:
def trilinear_to(shape_out, arr):  # arr: (nx, ny, nz)
    ten = torch.from_numpy(arr[None, None]).float()     # (1,1,X,Y,Z)
    out = F.interpolate(ten, size=shape_out, mode="trilinear", align_corners=False)
    return out[0,0].numpy()

target_shape = X_spec.shape[1:]  # (nx, ny, nz)
if gap.shape != target_shape:
    gap     = trilinear_to(target_shape, gap)
    foliage = trilinear_to(target_shape, foliage)
    bands   = np.stack([trilinear_to(target_shape, bands[k]) for k in range(bands.shape[0])], axis=0)
    obs     = (trilinear_to(target_shape, obs.astype(np.float32)) > 0.5)

# --- stack channels ---
X = np.concatenate([
        X_spec.astype(np.float32),               # (5, X, Y, Z)
        X_height.astype(np.float32),            # (1, X, Y, Z)
        gap[None, ...].astype(np.float32),      # (1, X, Y, Z)   <-- occlusion MVP
        # Optional extras:
        # foliage[None, ...].astype(np.float32), # (+1)
        # bands.astype(np.float32),              # (+3)
    ], axis=0)                                   # => (C, X, Y, Z)

# --- to torch ---
X_t   = torch.from_numpy(X[None]).float()       # (B=1, C, X, Y, Z)
obs_t = torch.from_numpy(obs[None, None].astype(np.float32))  # (B,1,X,Y,Z)

# --- example forward + loss masking ---
# logits = model(X_t)                            # (B, K, X, Y, Z)
# labels = ...                                   # (B, X, Y, Z) long
# criterion = torch.nn.CrossEntropyLoss(reduction="none")
# per_voxel = criterion(logits, labels)          # (B, X, Y, Z)
# masked = (per_voxel * obs_t.squeeze(1))        # ignore unobserved voxels
# loss = masked.sum() / (obs_t.sum() + 1e-6)
# loss.backward()

np.save("X_input_for_model.npy", X.astype(np.float32))
print("Prepared input tensor with", X.shape[0], "channels → saved as X_input_for_model.npy")

# %%
