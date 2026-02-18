
#%%
def merge_znorm_structure_column_height(
    z_norm_las,
    structure_las,
    output_las,
    output_las_no0=None,        # 🔹 back in the signature
    voxel_size_xy=0.05,
    voxel_size_z=0.05,
    min_conf=0.0
):
    # ---------------------
    # Load LAS files
    # ---------------------
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS missing 'Z_norm' extra dimension")

    # ---------------------
    # Define voxel grid
    # ---------------------
    x0 = min(las_z.x.min(), las_s.x.min())
    y0 = min(las_z.y.min(), las_s.y.min())
    z0 = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x0, y0, z0], float)

    def vox_xy(x, y):
        ix = np.floor((x - origin[0]) / voxel_size_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / voxel_size_xy).astype(np.int32)
        return ix, iy

    # Structure XY voxel indices
    s_ix, s_iy = vox_xy(las_s.x, las_s.y)

    # -------------------------------
    # 1) Build PROBABILITIES per XY column
    # -------------------------------
    mask = (las_s.classification >= 64) & (las_s.classification <= 68)

    df = pd.DataFrame({
        "ix": s_ix[mask],
        "iy": s_iy[mask],
        "cls": las_s.classification[mask]
    })

    if df.empty:
        raise RuntimeError("No structure classes found!")

    xy_counts = df.groupby(["ix", "iy", "cls"]).size().unstack(fill_value=0)
    xy_probs = xy_counts.div(xy_counts.sum(axis=1), axis=0)

    # Filter uncertain columns
    best_conf = xy_probs.max(axis=1)
    xy_probs = xy_probs[best_conf >= min_conf]

    # Convert to dict for fast lookup
    column_prob_map = {idx: xy_probs.loc[idx] for idx in xy_probs.index}

    # -------------------------------
    # 2) CLASSIFY Z_norm points
    # -------------------------------
    z_ix, z_iy = vox_xy(las_z.x, las_z.y)
    zn = las_z["Z_norm"].astype(float)

    out_class = np.zeros(len(las_z.points), dtype=np.uint8)

    dfz = pd.DataFrame({
        "ix": z_ix,
        "iy": z_iy,
        "idx": np.arange(len(las_z.points)),
        "z": zn
    })

    for (ix, iy), grp in dfz.groupby(["ix", "iy"]):
        key = (ix, iy)
        if key not in column_prob_map:
            continue  # no evidence → stay 0

        base = column_prob_map[key]
        classes = base.index.to_numpy()
        p_base = base.to_numpy()

        for z_val, pt_idx in zip(grp["z"].values, grp["idx"].values):
            posterior = np.array([
                p_c * height_prior(z_val, cls)
                for p_c, cls in zip(p_base, classes)
            ])

            if posterior.sum() <= 0:
                continue

            out_class[pt_idx] = int(classes[np.argmax(posterior)])

    # -------------------------------
    # 3) Write refined LAS (ALL points)
    # -------------------------------
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()
    las_out.classification = out_class
    las_out.header.software_id = "Column+HeightFusion"

    las_out = ensure_las14_pf6(las_out)
    Path(output_las).parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    # -------------------------------
    # 4) Optional: write no0 version
    # (subset of points, *does not* affect main file)
    # -------------------------------
    if output_las_no0 is not None:
        keep = out_class != 0
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        Path(output_las_no0).parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))
        print(f"✅ Wrote no0 subset: {output_las_no0}")

    # summary
    cnt = Counter(out_class)
    print("\nCLASS DISTRIBUTION:")
    for cid in sorted(cnt):
        print(f"{cnt[cid]:>10,}  {CLASS_NAMES.get(cid,'Unknown')} ({cid})")

    print(f"\nWritten: {output_las}")


# %%
merge_znorm_structure_column_height(
    z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.71.las",
    structure_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.71.las",
    output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_refined/z_norm_struct_refined_71.las",
    output_las_no0="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_refined/z_norm_struct_refined_no0_71.las",
    voxel_size_xy=0.05,
    voxel_size_z=0.05,
    min_conf=0.4
)


# %%
