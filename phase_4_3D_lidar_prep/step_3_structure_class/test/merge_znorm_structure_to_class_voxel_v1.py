import laspy
import numpy as np
from pathlib import Path
from collections import Counter


def merge_znorm_structure_by_voxels(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    structure_field: str = "classification",
    voxel_size_xy: float = 0.03,   # 3 cm voxels in X/
    voxel_size_z: float = 0.03     # 3 cm voxels in Z (or whatever matches your Z_norm resolution)
):
    """
    Merge structure classes into a Z_norm LAS using a shared voxel grid.

    Workflow:
      1. Build a voxel grid that covers BOTH point clouds.
      2. For the structure LAS, compute majority class per voxel (65–68 → 1–4).
      3. For the Z_norm LAS, assign each point the class of its voxel.

    This avoids losing points due to imperfect XYZ coordinate matches and
    is more robust to small misalignments and sampling differences.
    """

    # --- Load LAS files ---
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)

    print(f"🔎 Z_norm points: {len(las_z.points):,}, Structure points: {len(las_s.points):,}")

    # --- Validate required fields ---
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")

    if structure_field not in las_s.point_format.dimension_names:
        raise ValueError(f"'{structure_field}' not found in structure LAS.")

    # --- Define voxel grid origin (covers both clouds) ---
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())

    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    print(f"🧱 Voxel grid origin (x_min, y_min, z_min): {origin}")

    # --- Helper to compute voxel indices ---
    def compute_voxel_indices(x, y, z, origin, vs_xy, vs_z):
        ix = np.floor((x - origin[0]) / vs_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / vs_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / vs_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    # === 1️⃣ Structure LAS → majority class per voxel ===
    print("📦 Building voxel → structure-class map...")

    vox_s = compute_voxel_indices(las_s.x, las_s.y, las_s.z, origin, voxel_size_xy, voxel_size_z)
    raw_struct = np.asarray(las_s[structure_field], dtype=np.uint16)

    # Remap structure labels (65–68 → compact 1–4)
    structure_map = {
        64: 0,  # Forb (assuming 0 means unmatched or no class)
        65: 1,  # hummock
        66: 2,  # shrub
        67: 3,  # woody debris
        68: 4   # mallee
    }
    labels_s = np.array([structure_map.get(lbl, 0) for lbl in raw_struct], dtype=np.uint8)

    # Unique voxels in structure cloud
    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)

    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint8)

    for k in range(len(uniq_vox_s)):
        class_vals = labels_s[inv_s == k]
        if class_vals.size == 0:
            continue
        # Majority vote, ignoring 0
        counts = np.bincount(class_vals, minlength=5)  # 0..4
        counts[0] = 0  # ignore "Unmatched" when picking majority
        best = np.argmax(counts)
        voxel_class[k] = best

    # Build dict: voxel_index → class
    voxel_struct_map = {
        (ix, iy, iz): c for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class) if c != 0
    }

    print(f"✅ Structure voxels with non-zero class: {len(voxel_struct_map):,}")

    # === 2️⃣ Z_norm LAS → assign class from its voxel ===
    print("🧩 Assigning voxel-based classes to Z_norm points...")

    vox_z = compute_voxel_indices(las_z.x, las_z.y, las_z.z, origin, voxel_size_xy, voxel_size_z)

    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    matched = 0

    # Simple loop; can be optimised later if needed
    for i, (ix, iy, iz) in enumerate(vox_z):
        cls = voxel_struct_map.get((ix, iy, iz), 0)
        classification[i] = cls
        if cls != 0:
            matched += 1

    print(f"✅ Matched structure labels via voxels: {matched:,} / {len(las_z.points):,}")

    # === 3️⃣ Prepare output LAS ===
    header = las_z.header
    las_out = laspy.LasData(header)
    las_out.points = las_z.points.copy()

    # Ensure Z_norm is present and float32
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    # Assign new classification
    las_out.classification = classification

    # Ensure software_id is ASCII-compliant
    las_out.header.software_id = las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")

    # Save
    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # Stats
    counts = Counter(classification)
    print(f"✅ Wrote merged LAS: {output_las}")
    print("✅ Class distribution:", dict(counts))
    print("💡 Classes: 0=Unmatched, 1=Hummock, 2=Shrub, 3=WoodyDebris, 4=Mallee")
    print(f"💡 Voxel sizes: {voxel_size_xy} m (XY), {voxel_size_z} m (Z)")
    
#%%
## teste 2
import laspy
import numpy as np
from pathlib import Path
from collections import Counter


def merge_znorm_structure_by_voxels(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    structure_field: str = "classification",
    voxel_size_xy: float = 0.03,   # 3 cm in X/Y
    voxel_size_z: float = 0.03     # 3 cm in Z
):
    """
    Merge structure classes into a Z_norm LAS using a shared voxel grid.

    Classes (compact internal encoding):
      0 = Unmatched
      1 = Forb        (orig. 64)
      2 = Hummock     (orig. 65)
      3 = Shrub       (orig. 66)
      4 = WoodyDebris (orig. 67)
      5 = Mallee      (orig. 68)
    """

    # --- Load LAS files ---
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)

    print(f"🔎 Z_norm points: {len(las_z.points):,}, Structure points: {len(las_s.points):,}")

    # --- Validate required fields ---
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")

    if structure_field not in las_s.point_format.dimension_names:
        raise ValueError(f"'{structure_field}' not found in structure LAS.")

    # --- Define voxel grid origin (covers both clouds) ---
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())

    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    print(f"🧱 Voxel grid origin (x_min, y_min, z_min): {origin}")

    # --- Helper to compute voxel indices ---
    def compute_voxel_indices(x, y, z, origin, vs_xy, vs_z):
        ix = np.floor((x - origin[0]) / vs_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / vs_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / vs_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    # === 1️⃣ Structure LAS → majority class per voxel ===
    print("📦 Building voxel → structure-class map...")

    vox_s = compute_voxel_indices(las_s.x, las_s.y, las_s.z, origin, voxel_size_xy, voxel_size_z)
    raw_struct = np.asarray(las_s[structure_field], dtype=np.uint16)

    # 🔁 Remap structure labels (64–68 → compact 1–5)
    structure_map = {
        64: 1,  # Forb
        65: 2,  # Hummock
        66: 3,  # Shrub
        67: 4,  # Woody Debris
        68: 5   # Mallee
    }
    labels_s = np.array([structure_map.get(lbl, 0) for lbl in raw_struct], dtype=np.uint8)

    # Unique voxels in structure cloud
    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)

    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint8)

    for k in range(len(uniq_vox_s)):
        class_vals = labels_s[inv_s == k]
        if class_vals.size == 0:
            continue
        # Majority vote, ignoring 0
        counts = np.bincount(class_vals, minlength=6)  # 0..5
        counts[0] = 0  # ignore "Unmatched" when picking majority
        best = np.argmax(counts)
        voxel_class[k] = best

    # Build dict: voxel_index → class (only non-zero)
    voxel_struct_map = {
        (ix, iy, iz): c for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class) if c != 0
    }

    print(f"✅ Structure voxels with non-zero class: {len(voxel_struct_map):,}")

    # === 2️⃣ Z_norm LAS → assign class from its voxel ===
    print("🧩 Assigning voxel-based classes to Z_norm points...")

    vox_z = compute_voxel_indices(las_z.x, las_z.y, las_z.z, origin, voxel_size_xy, voxel_size_z)

    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    matched = 0

    for i, (ix, iy, iz) in enumerate(vox_z):
        cls = voxel_struct_map.get((ix, iy, iz), 0)
        classification[i] = cls
        if cls != 0:
            matched += 1

    print(f"✅ Matched structure labels via voxels: {matched:,} / {len(las_z.points):,}")

    # === 3️⃣ Prepare output LAS ===
    header = las_z.header
    las_out = laspy.LasData(header)
    las_out.points = las_z.points.copy()

    # Ensure Z_norm is present and float32
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    # Assign new classification
    las_out.classification = classification

    # Ensure software_id is ASCII-compliant
    las_out.header.software_id = las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")

    # Save
    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # Stats
    counts = Counter(classification)
    print(f"✅ Wrote merged LAS: {output_las}")
    print("✅ Class distribution:", dict(counts))
    print("💡 Classes: 0=Unmatched, 1=Forb, 2=Hummock, 3=Shrub, 4=WoodyDebris, 5=Mallee")
    print(f"💡 Voxel sizes: {voxel_size_xy} m (XY), {voxel_size_z} m (Z)")

