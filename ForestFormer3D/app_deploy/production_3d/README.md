# FVC Structure Production 3D Pipeline

This folder contains the production COPC/STAC deployment helpers.

Split workflow:

- Development summary viewer: `ForestFormer3D/tools/qualitative_results_app.py`
- Production 3D viewer: `/fvc_structure-viewer/3d` in the website app, driven by STAC and COPC

The development viewer remains Plotly-based and is appropriate for local QA, debugging, and small-to-medium scenes.
The production 3D route should use public COPC assets and STAC metadata.

## Recommended R2 Layout

Use a separate production layout inside the bucket so the existing `.ply` development uploads stay untouched:

- `terrascientia/fvc_structure-viewer/lidar/raw/<experiment>/<scene>.ply` for the current development summary viewer
- `terrascientia/fvc_structure-viewer/lidar/copc/low/<scene>.copc.laz` for low scenes
- `terrascientia/fvc_structure-viewer/lidar/copc/medium/<scene>.copc.laz` for medium scenes
- `terrascientia/fvc_structure-viewer/lidar/stac/catalog.json`
- `terrascientia/fvc_structure-viewer/lidar/stac/collections/<collection>/collection.json`
- `terrascientia/fvc_structure-viewer/lidar/stac/items/<collection>/<item>.json`
- `terrascientia/fvc_structure-viewer/lidar/stac/summaries/<collection>/<item>-summary.json`
- `terrascientia/fvc_structure-viewer/imagery/rgb/...` for future RGB imagery products
- `terrascientia/fvc_structure-viewer/imagery/multispec/...` for future multispectral imagery products
- `terrascientia/fvc_structure-viewer/imagery/hyspec/...` for future hyperspectral imagery products

## 1. Convert Development Scenes to COPC

The conversion helper preserves experiment folders and writes `.copc.laz` outputs.

```bash
python ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root /path/to/dev-scenes \
  --output-root /path/to/prod-bundle/copc \
  --a-srs EPSG:7854
```

Requirements:

- PDAL must be installed and available as `pdal`
- The PDAL build must include `writers.copc`
- The converter writes `extra_dims=all` so PLY attributes such as predictions, scores, and GT labels are preserved as LAS extra dimensions in the COPC output
- Use `--a-srs` when the source files do not already carry CRS metadata and you need the COPC output stamped explicitly, for example `--a-srs EPSG:7854`

For the current Calperum production bundle, use `EPSG:7854` unless you later decide to write a compound horizontal+vertical CRS explicitly.

## 2. Generate STAC Metadata

Create the production STAC catalog from the converted COPC directory.

Recommended: generate relative COPC hrefs first. That keeps the bundle portable and avoids baking in the final public bucket domain during the local build step.

```bash
python ForestFormer3D/app_deploy/production_3d/generate_stac_for_copc.py \
  --copc-root /path/to/prod-bundle/copc \
  --stac-root /path/to/prod-bundle/stac
```

If you already know the final public bucket URL and want absolute COPC hrefs in the STAC items, use:

```bash
python ForestFormer3D/app_deploy/production_3d/generate_stac_for_copc.py \
  --copc-root /path/to/prod-bundle/copc \
  --stac-root /path/to/prod-bundle/stac \
  --asset-href-mode absolute \
  --public-base-url https://<public-r2-domain>/fvc_structure-viewer/lidar
```

This writes:

- `catalog.json`
- one collection per experiment folder
- one item per COPC scene
- one summary JSON per scene

With absolute href mode, the generated item assets point to:

- `https://<public-r2-domain>/fvc_structure-viewer/lidar/copc/medium/<scene>.copc.laz`
- `https://<public-r2-domain>/fvc_structure-viewer/lidar/copc/low/<scene>.copc.laz`

## 3. Upload the Production Bundle to R2

Use the provided sync script after configuring AWS-compatible credentials for Cloudflare R2.

```bash
export R2_ACCOUNT_ID=<cloudflare-account-id>
export AWS_ACCESS_KEY_ID=<r2-access-key-id>
export AWS_SECRET_ACCESS_KEY=<r2-secret-access-key>

bash ForestFormer3D/app_deploy/production_3d/sync_copc_to_r2.sh /path/to/prod-bundle terrascientia fvc_structure-viewer/lidar
```

To sync the raw scenes for the development viewer into `lidar/raw`, use:

```bash
bash ForestFormer3D/app_deploy/production_3d/sync_raw_to_r2.sh /path/to/dev-scenes terrascientia fvc_structure-viewer/lidar/raw
```

## 4. Make Production Assets Public

For the `/fvc_structure-viewer/3d` route, the STAC catalog and COPC assets must be reachable by the browser.
Use one of these:

- Cloudflare R2 Public Development URL
- a custom domain mapped to the bucket

## 5. Point the Website to the Production Catalog

Set this in the website deployment:

```bash
FVC_STRUCTURE_STAC_CATALOG_URL=https://<public-r2-domain>/fvc_structure-viewer/lidar/stac/catalog.json
```

Then `/fvc_structure-viewer/3d` will open the production STAC catalog and hand the selected COPC asset to the dedicated viewer.

## 6. Site 1 CSF Bundle

The repo now contains a regenerated Site 1 bundle built from:

```bash
/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf
```

Local bundle path:

```bash
ForestFormer3D/app_deploy/production_3d/generated/site1_znorm_csf
```

Expected public URLs after R2 sync:

```bash
https://<public-r2-domain>/fvc_structure-viewer/lidar/stac/catalog.json
https://<public-r2-domain>/fvc_structure-viewer/lidar/copc/z_norm_csf/z_norm_csf_tiles_multispectral.106.copc.laz
```

Deploy this bundle to Cloudflare R2 and update Heroku in one step:

```bash
export R2_ACCOUNT_ID=<cloudflare-account-id>
export R2_ACCESS_KEY_ID=<r2-access-key-id>
export R2_SECRET_ACCESS_KEY=<r2-secret-access-key>

bash ForestFormer3D/app_deploy/production_3d/deploy_site1_znorm_csf.sh \
  https://<public-r2-domain> \
  lauransotomayor
```

That helper will:

- regenerate STAC with absolute asset hrefs
- sync `copc/medium`, `copc/low`, and `stac/` to `terrascientia/fvc_structure-viewer/lidar` on R2
- set `FVC_STRUCTURE_STAC_CATALOG_URL` on the Heroku app if the `heroku` CLI is installed

## 7. Scene Header Probe

Use the lightweight header probe when you want to inspect LAS, LAZ, or COPC metadata without relying on `laspy`:

```bash
bash ForestFormer3D/app_deploy/production_3d/probe_scene_headers.sh \
  /path/to/file.las \
  /path/to/file.copc.laz
```

It prints the `pdal info --summary` and `pdal info --schema` output for each file and only depends on `pdal`.
