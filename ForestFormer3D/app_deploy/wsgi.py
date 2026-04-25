import json
import importlib
import os
import sys
from html import escape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dash import Dash, html

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

qualitative_results_app = importlib.import_module("tools.qualitative_results_app")
create_app = qualitative_results_app.create_app
scan_experiments = qualitative_results_app.scan_experiments
scan_named_experiments = qualitative_results_app.scan_named_experiments

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

production_catalog_url = os.getenv("FVC_STRUCTURE_STAC_CATALOG_URL", "").strip()

if production_catalog_url:
    production_stac_base_url = production_catalog_url.rsplit("/", 1)[0] + "/"
    production_viewer_path = "/fvc_structure-viewer/"
    production_copc_viewer_base = "https://viewer.copc.io/?copc="

    def fetch_remote_bytes(url: str) -> tuple[bytes, str, str]:
        request = Request(
            url,
            headers={
                "User-Agent": "ForestFormer3D/production-viewer",
                "Accept": "application/json, text/plain, */*",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "application/json")
                return response.read(), "200 OK", content_type
        except HTTPError as exc:
            return exc.read(), f"{exc.code} {exc.reason}", exc.headers.get("Content-Type", "text/plain; charset=utf-8")
        except URLError as exc:
            body = f"Failed to fetch {url}: {exc.reason}".encode("utf-8")
            return body, "502 Bad Gateway", "text/plain; charset=utf-8"

    @server.get("/fvc_structure-viewer/api/stac/catalog.json")
    def production_catalog_proxy():
        body, status, content_type = fetch_remote_bytes(production_catalog_url)
        response = server.response_class(body, status=status, mimetype=content_type)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    @server.get("/fvc_structure-viewer/api/stac/<path:subpath>")
    def production_stac_proxy(subpath: str):
        remote_url = urljoin(production_stac_base_url, subpath)
        body, status, content_type = fetch_remote_bytes(remote_url)
        response = server.response_class(body, status=status, mimetype=content_type)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    @server.get("/fvc_structure-viewer")
    @server.get("/fvc_structure-viewer/")
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
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: radial-gradient(circle at top left, #f8f3ea 0, var(--bg) 40%, #ebe4d7 100%);
      color: var(--ink);
    }}
    header {{
      padding: 20px 28px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 253, 248, 0.9);
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 22px;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px 24px 32px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 32px rgba(31, 41, 51, 0.06);
    }}
    .list {{
      max-height: 220px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: white;
    }}
    .list button {{
      width: 100%;
      border: 0;
      border-bottom: 1px solid #eee;
      background: white;
      color: var(--ink);
      padding: 10px 12px;
      text-align: left;
      font: inherit;
      cursor: pointer;
    }}
    .list button:hover {{
      background: #f0f7f6;
    }}
    .list button[aria-pressed="true"] {{
      background: #eaf3f2;
      color: var(--accent-dark);
      font-weight: 700;
    }}
    .viewer-shell {{
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      background: white;
      box-shadow: 0 10px 32px rgba(31, 41, 51, 0.06);
    }}
    .viewer-bar {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: #eee7da;
      color: var(--muted);
    }}
    iframe {{
      width: 100%;
      height: 74vh;
      border: 0;
      display: block;
      background: #0f172a;
    }}
    code {{
      background: #f4efe4;
      padding: 2px 6px;
      border-radius: 8px;
    }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>ForestFormer3D production viewer</h1>
    <p>Cloudflare-backed route at <code>/fvc_structure-viewer/</code></p>
  </header>
  <div class="wrap">
    <div class="card">
      <strong>STAC catalog</strong>
      <p><a href="/fvc_structure-viewer/api/stac/catalog.json">Open catalog JSON</a></p>
      <p class="muted">This route reads the published Cloudflare R2 catalog and opens COPC scenes in-browser.</p>
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
    const catalogUrl = "/fvc_structure-viewer/api/stac/catalog.json";
    const viewerBase = {json.dumps(production_copc_viewer_base)};
    const viewer = document.getElementById("viewer");
    const sceneList = document.getElementById("sceneList");

    async function fetchJson(url) {{
      const response = await fetch(url);
      if (!response.ok) {{
        throw new Error(`Failed to fetch ${{url}} (${{response.status}} ${{response.statusText}})`);
      }}
      return await response.json();
    }}

    async function loadScenes() {{
      const catalog = await fetchJson(catalogUrl);
      const queue = [{{ url: catalogUrl, doc: catalog }}];
      const seen = new Set([catalogUrl]);
      const scenes = [];

      while (queue.length) {{
        const {{ url, doc }} = queue.shift();
        const links = Array.isArray(doc.links) ? doc.links : [];
        for (const link of links) {{
          if (!link || !link.href) continue;
          if (link.rel !== "child" && link.rel !== "item") continue;
          const nextUrl = new URL(link.href, url).href;
          if (seen.has(nextUrl)) continue;
          seen.add(nextUrl);
          const nextDoc = await fetchJson(nextUrl);
          queue.push({{ url: nextUrl, doc: nextDoc }});
        }}
        if (doc.assets && doc.assets.copc && doc.assets.copc.href) {{
          scenes.push({{
            title: doc.title || doc.id || "COPC scene",
            href: new URL(doc.assets.copc.href, url).href,
          }});
        }}
      }}

      if (!scenes.length) {{
        sceneList.innerHTML = "<div style='padding:12px' class='muted'>No COPC scenes found in STAC.</div>";
        return;
      }}

      sceneList.innerHTML = "";
      scenes.forEach((scene, index) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = scene.title;
        button.setAttribute("aria-pressed", index === 0 ? "true" : "false");
        button.addEventListener("click", () => {{
          Array.from(sceneList.querySelectorAll("button")).forEach((el) => el.setAttribute("aria-pressed", "false"));
          button.setAttribute("aria-pressed", "true");
          viewer.src = viewerBase + encodeURIComponent(scene.href);
        }});
        sceneList.appendChild(button);
        if (index === 0) {{
          viewer.src = viewerBase + encodeURIComponent(scene.href);
        }}
      }});
    }}

    loadScenes().catch((error) => {{
      sceneList.innerHTML = "<div style='padding:12px;color:#8c2f39'>" + error + "</div>";
    }});
  </script>
</body>
</html>"""
        return body
