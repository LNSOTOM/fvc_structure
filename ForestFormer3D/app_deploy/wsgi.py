import json
import os
import sys
from html import escape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

WORK_DIRS_ROOT = Path(os.getenv("WORK_DIRS_ROOT", "/app/data/work_dirs"))
NORMALIZED_HEIGHT_ROOT = Path(
    os.getenv(
        "NORMALIZED_HEIGHT_ROOT",
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf",
    )
)
MAX_POINTS = int(os.getenv("MAX_POINTS", "120000"))
SEED = int(os.getenv("SEED", "0"))
MARKER_SIZE = float(os.getenv("MARKER_SIZE", "2.0"))

segment_json = os.getenv("SEGMENT_ANY_TREE_EXPERIMENTS_JSON", "{}")
segment_map_raw = json.loads(segment_json)
segment_dirs = {k: Path(v) for k, v in segment_map_raw.items()}

experiment_families = {}

production_catalog_url = os.getenv("FVC_STRUCTURE_STAC_CATALOG_URL", "").strip()
import importlib

from dash import Dash, html

qualitative_results_app = importlib.import_module("tools.qualitative_results_app")
create_app = qualitative_results_app.create_app
scan_experiments = qualitative_results_app.scan_experiments
scan_named_experiments = qualitative_results_app.scan_named_experiments

sat_experiments = scan_named_experiments(segment_dirs) if segment_dirs else {}
ff_experiments = scan_experiments(WORK_DIRS_ROOT) if WORK_DIRS_ROOT.exists() else {}

if sat_experiments:
    experiment_families["SegmentAnyTree"] = sat_experiments
if ff_experiments:
    experiment_families["ForestFormer3D"] = ff_experiments

if not experiment_families:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("ForestFormer3D viewer"),
            html.P("The app is deployed, but no experiment scene files are available yet."),
            html.P("Expected files: .ply, .las, or .laz inside experiment folders."),
            html.Pre(
                "\n".join(
                    [
                        f"WORK_DIRS_ROOT={WORK_DIRS_ROOT}",
                        "SEGMENT_ANY_TREE_EXPERIMENTS_JSON=<optional JSON mapping>",
                    ]
                )
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "800px",
            "margin": "48px auto",
            "padding": "24px",
            "lineHeight": 1.5,
        },
    )
else:
    app = create_app(
        experiment_families=experiment_families,
        max_points=MAX_POINTS,
        seed=SEED,
        marker_size=MARKER_SIZE,
        normalized_height_root=NORMALIZED_HEIGHT_ROOT,
    )

server = app.server

if production_catalog_url:
    production_stac_base_url = production_catalog_url.rsplit("/", 1)[0] + "/"
    production_catalog_path = "/fvc_structure-viewer/api/stac/catalog.json"
    production_viewer_path = "/fvc_structure-viewer/"
    production_copc_viewer_base = "https://viewer.copc.io/?copc="

    def fetch_remote_bytes(url: str) -> tuple[bytes, str, list[tuple[str, str]]]:
        request = Request(
            url,
            headers={
                "User-Agent": "ForestFormer3D/production-viewer",
                "Accept": "*/*",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                headers = [("Content-Type", response.headers.get("Content-Type", "application/json"))]
                if response.headers.get("ETag"):
                    headers.append(("ETag", response.headers["ETag"]))
                if response.headers.get("Last-Modified"):
                    headers.append(("Last-Modified", response.headers["Last-Modified"]))
                return response.read(), "200 OK", headers
        except HTTPError as exc:
            return exc.read(), f"{exc.code} {exc.reason}", [("Content-Type", exc.headers.get("Content-Type", "text/plain; charset=utf-8"))]
        except URLError as exc:
            message = f"Failed to fetch {url}: {exc.reason}".encode("utf-8")
            return message, "502 Bad Gateway", [("Content-Type", "text/plain; charset=utf-8")]

    @server.get(production_catalog_path)
    def production_catalog_proxy():
        remote_url = production_catalog_url
        body, status, headers = fetch_remote_bytes(remote_url)
        headers = headers + [("Access-Control-Allow-Origin", "*"), ("Content-Length", str(len(body)))]
        return app.response_class(body, status=status, headers=headers)

    @server.get("/fvc_structure-viewer/api/stac/<path:subpath>")
    def production_stac_proxy(subpath: str):
        remote_url = urljoin(production_stac_base_url, subpath)
        body, status, headers = fetch_remote_bytes(remote_url)
        headers = headers + [("Access-Control-Allow-Origin", "*"), ("Content-Length", str(len(body)))]
        return app.response_class(body, status=status, headers=headers)

    @server.get(production_viewer_path)
    @server.get("/fvc_structure-viewer")
    @server.get("/fvc_structure-viewer/3d")
    def production_viewer_page():
        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ForestFormer3D production viewer</title>
  <style>
    :root {{
      --bg: #f4efe4;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5b6470;
      --border: #d9d1c3;
      --accent: #1f5c5b;
      --accent-dark: #15403f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 20px 24px; border-bottom: 1px solid var(--border); background: var(--panel); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 20px 24px 32px; }}
    .card {{ border: 1px solid var(--border); border-radius: 14px; background: var(--panel); padding: 16px; margin-bottom: 16px; box-shadow: 0 10px 28px rgba(31, 41, 51, 0.06); }}
    .list {{ max-height: 220px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; background: white; }}
    .list a {{ display: block; padding: 10px 12px; color: var(--ink); text-decoration: none; border-bottom: 1px solid #eee; }}
    .list a:hover {{ background: #f0f7f6; }}
    .viewer-shell {{ border: 1px solid var(--border); border-radius: 14px; overflow: hidden; background: white; box-shadow: 0 10px 28px rgba(31, 41, 51, 0.06); }}
    .viewer-bar {{ padding: 12px 16px; border-bottom: 1px solid var(--border); background: #eee7da; color: var(--muted); }}
    iframe {{ width: 100%; height: 74vh; border: 0; background: #0f172a; display: block; }}
    code {{ background: #f4efe4; padding: 2px 6px; border-radius: 8px; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>ForestFormer3D production viewer</h1>
    <p class="muted">Route mounted at <code>/fvc_structure-viewer/</code>. Catalog proxy: <code>{escape(production_catalog_path)}</code></p>
  </header>
  <div class="wrap">
    <div class="card">
      <strong>Open the catalog</strong>
      <p><a href="{escape(production_catalog_path)}">STAC catalog JSON</a></p>
      <p class="muted">This keeps the same vertical flow as the existing viewer and places the COPC viewer underneath.</p>
    </div>
    <div class="card">
      <strong>Scenes</strong>
      <div id="sceneList" class="list"><div style="padding:12px" class="muted">Loading...</div></div>
    </div>
    <div class="viewer-shell">
      <div class="viewer-bar">Embedded COPC viewer</div>
      <iframe id="viewer" title="COPC viewer"></iframe>
    </div>
  </div>
  <script>
    const catalogUrl = {json.dumps(production_catalog_path)};
    const viewerBase = {json.dumps(production_copc_viewer_base)};
    const viewer = document.getElementById("viewer");
    const sceneList = document.getElementById("sceneList");
    const scenes = [];

    async function load() {{
      const response = await fetch(catalogUrl);
      const catalog = await response.json();
      const links = Array.isArray(catalog.links) ? catalog.links : [];
      const discovered = [];
      for (const link of links) {{
        if (!link || !link.href || link.rel !== "child") continue;
        const child = await fetch(new URL(link.href, catalogUrl).href).then((r) => r.json());
        const itemLinks = Array.isArray(child.links) ? child.links : [];
        for (const itemLink of itemLinks) {{
          if (itemLink.rel !== "item") continue;
          const item = await fetch(new URL(itemLink.href, new URL(link.href, catalogUrl).href).href).then((r) => r.json());
          const href = item.assets && item.assets.copc && item.assets.copc.href;
          if (href) discovered.push({{ title: item.title || item.id, href: new URL(href, new URL(itemLink.href, new URL(link.href, catalogUrl).href).href).href }});
        }}
      }}
      if (!discovered.length) {{
        sceneList.innerHTML = "<div style='padding:12px' class='muted'>No COPC scenes found in STAC.</div>";
        return;
      }}
      sceneList.innerHTML = "";
      discovered.forEach((scene, index) => {{
        const link = document.createElement("a");
        link.href = "#";
        link.textContent = scene.title;
        link.addEventListener("click", (event) => {{
          event.preventDefault();
          viewer.src = viewerBase + encodeURIComponent(scene.href);
        }});
        if (index === 0) viewer.src = viewerBase + encodeURIComponent(scene.href);
        sceneList.appendChild(link);
      }});
    }}

    load().catch((error) => {{
      sceneList.innerHTML = "<div style='padding:12px;color:#8c2f39'>" + error + "</div>";
    }});
  </script>
</body>
</html>"""
        return body
