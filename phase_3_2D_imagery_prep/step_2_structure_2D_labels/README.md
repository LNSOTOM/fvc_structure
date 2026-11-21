# Structure Classification for 2D Labels

The scripts in this directory classify vegetation polygons, rasterize them and
downsample the categorical labels. These original scripts are site-specific
interactive research scripts with hardcoded paths. For the config-driven
workflow, see the [Phase 3 handoff](../REPRODUCIBILITY.md).

`2_1_structure_2D_labels.py` writes `struct_id` and polygon `area`. Its mapping is:

| `struct_id` | Input types |
|---|---|
| 0 | `forb` |
| 1 | `branch_stem_dead`, `branch_stem_standing`, `cwd` |
| 2 | `grass_hummock`, `grass_hummock_senescent` |
| 3 | `shrub_chenopod`, `shrub_chenopod_senescent`, `shrub` |
| 4 | `mallee` |

The literal exclusion list is `bare_ground`, `shadow`, `be` and `unknowm`.
With a type column present, other unrecognized types are removed; matching
mapped types is case-sensitive. If no type column is found, the script assigns
all polygons class 1. Check the input field, class counts and log before using
the results. See the [label QA guide](../step_3_label_quality/README.md) for a
known discrepancy involving low-site `unknown` polygons.

`2_2_structure_2D_rasterLabels.py` uses a supplied reference raster's grid,
`all_touched=True`, class values 0–4 and NoData 255. For the documented
historical workflow that reference must be the 1 cm tile, not a coarser
predictor. `2_3_downsampling_labels.py` includes a batch example that uses
nearest-neighbour resampling and remaps classes 0–4 to 64–68; inspect
alignment and output values afterwards.
