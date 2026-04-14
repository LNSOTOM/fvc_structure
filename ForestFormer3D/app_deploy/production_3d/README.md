# FVC Structure Production 3D Pipeline

This folder contains the production COPC/STAC deployment helpers.

Split workflow:

- Development summary viewer: `ForestFormer3D/tools/qualitative_results_app.py`
- Production 3D viewer: `/fvc_structure-viewer/3d` in the website app, driven by STAC and COPC

The development viewer remains Plotly-based and is appropriate for local QA, debugging, and small-to-medium scenes.
The production 3D route should use public COPC assets and STAC metadata.

## Recommended R2 Layout

Use a separate production layout inside the bucket so the existing `.ply` development uploads stay untouched:

- `terrascientia/fvc_structure-viewer/<experiment>/<scene>.ply` for the current development summary viewer
- `terrascientia/fvc_structure-viewer/copc/<experiment>/<scene>.copc.laz` for the production viewer
- `terrascientia/fvc_structure-viewer/stac/catalog.json`
- `terrascientia/fvc_structure-viewer/stac/collections/<collection>/collection.json`
- `terrascientia/fvc_structure-viewer/stac/items/<collection>/<item>.json`
- `terrascientia/fvc_structure-viewer/stac/summaries/<collection>/<item>-summary.json`

## 1. Convert Development Scenes to COPC

The conversion helper preserves experiment folders and writes `.copc.laz` outputs.

```bash
python ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root /path/to/dev-scenes \
  --output-root /path/to/prod-bundle/copc
```

Requirements:

- PDAL must be installed and available as `pdal`
- The PDAL build must include `writers.copc`

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
  --public-base-url https://<public-r2-domain>/fvc_structure-viewer
```

This writes:

- `catalog.json`
- one collection per experiment folder
- one item per COPC scene
- one summary JSON per scene

With absolute href mode, the generated item assets point to:

- `https://<public-r2-domain>/fvc_structure-viewer/copc/<experiment>/<scene>.copc.laz`

## 3. Upload the Production Bundle to R2

Use the provided sync script after configuring AWS-compatible credentials for Cloudflare R2.

```bash
export R2_ACCOUNT_ID=<cloudflare-account-id>
export AWS_ACCESS_KEY_ID=<r2-access-key-id>
export AWS_SECRET_ACCESS_KEY=<r2-secret-access-key>

bash ForestFormer3D/app_deploy/production_3d/sync_copc_to_r2.sh /path/to/prod-bundle terrascientia fvc_structure-viewer
```

## 4. Make Production Assets Public

For the `/fvc_structure-viewer/3d` route, the STAC catalog and COPC assets must be reachable by the browser.
Use one of these:

- Cloudflare R2 Public Development URL
- a custom domain mapped to the bucket

## 5. Point the Website to the Production Catalog

Set this in the website deployment:

```bash
FVC_STRUCTURE_STAC_CATALOG_URL=https://<public-r2-domain>/fvc_structure-viewer/stac/catalog.json
```

Then `/fvc_structure-viewer/3d` will open the production STAC catalog and hand the selected COPC asset to the dedicated viewer.
