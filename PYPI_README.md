# fvc3d

Transfer 2D vegetation growth-form labels into georeferenced 3D UAS LiDAR
point clouds.

`fvc3d` implements the label-transfer method of Sotomayor et al. (2026),
[*From 2D labels to 3D structure: Scalable label transfer and benchmarking of
3D vegetation models in rangeland ecosystems*](https://doi.org/10.31223/X5948M):
class height priors, voxel label transfer, watershed crown detection,
crown-centric polar refinement and consolidation, in one call.

## Install

```bash
pip install fvc3d
```

Python 3.10 or newer.

## Quick start

```python
from fvc3d import transfer_labels

result = transfer_labels(
    labels="growth_forms.tif",        # 2D label raster (GeoTIFF)
    point_cloud="lidar.laz",          # co-registered LAS/LAZ, same CRS
    output="labelled_vegetation.laz",
    validate_alignment=True,          # CRS, overlap and canopy-offset check
)
print(result.summary())
```

The output LAS/LAZ holds each point's `classification`, its height above
ground (`Z_norm`) and a `treeID` for each mallee crown.

## Classes

The defaults are the paper's five growth forms:

| Growth form | Raster ID | LAS code | Name in `fvc3d` |
|---|---|---|---|
| Forb | 0 | 64 | `forb` |
| Woody debris | 1 | 65 | `woody_debris` |
| Grass hummock | 2 | 66 | `hummock` |
| Shrub | 3 | 67 | `shrub` |
| Mallee | 4 | 68 | `mallee` |

- `class_map="paper"` reads the published rasters, which use the raster IDs.
- `classes="my_classes.yaml"` uses your own classes; see the
  [template](https://github.com/LNSOTOM/fvc_structure/blob/main/configs/classes.yaml).
- `classes="FOR-instanceV2"` writes FOR-instanceV2's ground, wood and leaf
  classes, as used for ForestFormer3D.

## Try it on the paper's data

The [workflow guide](https://lnsotom.github.io/fvc_structure/#api-try) shows
how to download one tile from the paper's
[Zenodo record](https://doi.org/10.5281/zenodo.24431573), run it, and compare
the result with the published labels.

## Links

- Guide: https://lnsotom.github.io/fvc_structure/
- Source and research workflow: https://github.com/LNSOTOM/fvc_structure
- Paper (preprint): https://doi.org/10.31223/X5948M
- Data and models: https://doi.org/10.5281/zenodo.24431573 ·
  UAS imagery and labels: https://doi.org/10.5281/zenodo.15036860

## Citation

If you use `fvc3d`, please cite the paper:

> Sotomayor, L.N., Lucieer, A., Turner, D., Barthelemy, J., Kattenborn, T.
> (2026). From 2D labels to 3D structure: Scalable label transfer and
> benchmarking of 3D vegetation models in rangeland ecosystems. EarthArXiv
> (preprint). https://doi.org/10.31223/X5948M

## Licence

MIT.
