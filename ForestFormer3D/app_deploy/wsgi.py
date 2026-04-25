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

if production_catalog_url:
    production_stac_base_url = production_catalog_url.rsplit("/", 1)[0] + "/"
    production_catalog_path = "/fvc_structure-viewer/api/stac/catalog.json"
    production_viewer_url = "/fvc_structure-viewer/"
    production_copc_viewer_base = "https://viewer.copc.io/?copc="

    def fetch_remote_bytes(url: str) -> tuple[bytes, str, list[tuple[str, str]], int]:
        request = Request(
            url,
            headers={
                "User-Agent": "ForestFormer3D/production-viewer",
                "Accept": "*/*",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                content_type = response.headers.get_content_type()
                headers = [
                    ("Content-Type", response.headers.get("Content-Type", content_type)),
                    ("Cache-Control", "public, max-age=60"),
                ]
                if response.headers.get("ETag"):
                    headers.append(("ETag", response.headers["ETag"]))
                if response.headers.get("Last-Modified"):
                    headers.append(("Last-Modified", response.headers["Last-Modified"]))
                return response.read(), "200 OK", headers, response.status
        except HTTPError as exc:
            return exc.read(), f"{exc.code} {exc.reason}", [("Content-Type", exc.headers.get("Content-Type", "text/plain; charset=utf-8"))], exc.code
        except URLError as exc:
            message = f"Failed to fetch {url}: {exc.reason}".encode("utf-8")
            return message, "502 Bad Gateway", [("Content-Type", "text/plain; charset=utf-8")], 502

    def render_viewer_html() -> bytes:
        catalog_proxy_url = production_catalog_path
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
      background: rgba(255, 253, 248, 0.86);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 22px;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: calc(100vh - 72px);
    }}
    .sidebar {{
      padding: 20px;
      border-right: 1px solid var(--border);
      background: rgba(255, 253, 248, 0.92);
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 32px rgba(31, 41, 51, 0.06);
    }}
    label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    select, input, button {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 10px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }}
    button {{
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: white;
      font-weight: 700;
      border: none;
    }}
    button.secondary {{
      background: white;
      color: var(--accent-dark);
      border: 1px solid var(--border);
    }}
    .muted {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
    .list {{
      max-height: 38vh;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: white;
    }}
    .list button {{
      border: 0;
      border-bottom: 1px solid #eee;
      background: white;
      color: var(--ink);
      border-radius: 0;
      text-align: left;
      font-weight: 600;
    }}
    .list button[aria-pressed="true"] {{
      background: #eaf3f2;
      color: var(--accent-dark);
    }}
    .viewer {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 253, 248, 0.86);
      align-items: center;
      flex-wrap: wrap;
    }}
    .toolbar code {{
      background: #f4efe4;
      padding: 4px 8px;
      border-radius: 8px;
    }}
    iframe {{
      width: 100%;
      height: 100%;
      border: 0;
      background: #0f172a;
    }}
    .status {{
      font-size: 13px;
      color: var(--muted);
      margin-top: 10px;
      white-space: pre-wrap;
    }}
    .error {{
      color: #8c2f39;
    }}
    @media (max-width: 980px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--border); }}
      .list {{ max-height: 26vh; }}
      iframe {{ min-height: 68vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ForestFormer3D production viewer</h1>
    <p>STAC catalog proxy: <code>{escape(catalog_proxy_url)}</code></p>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <div class="card">
        <label for="collection">Collection</label>
        <select id="collection"></select>
      </div>
      <div class="card">
        <label for="scene">Scene</label>
        <select id="scene"></select>
        <div style="height:12px"></div>
        <button id="openScene">Open scene in viewer</button>
        <div style="height:8px"></div>
        <a id="openExternal" href="#" target="_blank" rel="noopener noreferrer">
          Open selected scene in a new tab
        </a>
      </div>
      <div class="card">
        <label>Scenes</label>
        <div id="sceneList" class="list"></div>
        <div id="status" class="status">Loading STAC catalog...</div>
      </div>
    </aside>
    <main class="viewer">
      <div class="toolbar">
        <span>Embedded viewer:</span>
        <code id="copcUrl">waiting...</code>
      </div>
      <iframe id="viewerFrame" title="COPC viewer" src="about:blank"></iframe>
    </main>
  </div>
  <script>
    const CATALOG_URL = {json.dumps(catalog_proxy_url)};
    const COPC_VIEWER_BASE = {json.dumps(production_copc_viewer_base)};
    const viewerFrame = document.getElementById("viewerFrame");
    const copcUrlLabel = document.getElementById("copcUrl");
    const statusEl = document.getElementById("status");
    const collectionEl = document.getElementById("collection");
    const sceneEl = document.getElementById("scene");
    const sceneListEl = document.getElementById("sceneList");
    const openExternal = document.getElementById("openExternal");
    const openSceneButton = document.getElementById("openScene");

    let scenes = [];
    let collections = [];
    let selectedSceneIndex = 0;

    function setStatus(message, isError = false) {{
      statusEl.textContent = message;
      statusEl.classList.toggle("error", isError);
    }}

    function resolveUrl(href, base) {{
      return new URL(href, base).href;
    }}

    async function fetchJson(url) {{
      const response = await fetch(url, {{ credentials: "same-origin" }});
      if (!response.ok) {{
        throw new Error(`Failed to fetch ${{url}} (${{response.status}} ${{response.statusText}})`);
      }}
      return await response.json();
    }}

    function inferCollectionName(doc) {{
      return doc?.title || doc?.id || "collection";
    }}

    function sceneLabel(scene) {{
      return scene.title || scene.id || scene.href;
    }}

    async function loadCatalog() {{
      const root = await fetchJson(CATALOG_URL);
      const queue = [{{ url: CATALOG_URL, doc: root }}];
      const seen = new Set([CATALOG_URL]);
      const foundScenes = [];
      const foundCollections = [];

      while (queue.length) {{
        const {{ url, doc }} = queue.shift();
        const links = Array.isArray(doc.links) ? doc.links : [];
        const collectionName = inferCollectionName(doc);
        if (doc.type === "Collection" || doc.links?.some((link) => link.rel === "item")) {{
          foundCollections.push({{ name: collectionName, href: url }});
        }}

        if (doc.assets && doc.assets.copc && doc.assets.copc.href) {{
          foundScenes.push({{
            id: doc.id || collectionName,
            title: doc.title || doc.id || collectionName,
            href: resolveUrl(doc.assets.copc.href, url),
            collection: doc.collection || collectionName,
            summary: doc.description || "",
          }});
        }}

        for (const link of links) {{
          if (!link || !link.href) continue;
          if (link.rel !== "child" && link.rel !== "item") continue;
          const nextUrl = resolveUrl(link.href, url);
          if (seen.has(nextUrl)) continue;
          seen.add(nextUrl);
          queue.push({{ url: nextUrl, doc: await fetchJson(nextUrl) }});
        }}
      }}

      collections = foundCollections.length ? foundCollections : [{{ name: root.title || root.id || "root", href: CATALOG_URL }}];
      scenes = foundScenes;
    }}

    function renderCollections() {{
      collectionEl.innerHTML = "";
      collections.forEach((collection, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = collection.name;
        collectionEl.appendChild(option);
      }});
    }}

    function scenesForCollection(collectionName) {{
      return scenes.filter((scene) => scene.collection === collectionName || collectionName === "All scenes");
    }}

    function renderScenes() {{
      const currentCollection = collections[Number(collectionEl.value)]?.name || "All scenes";
      const filtered = currentCollection === "All scenes" ? scenes : scenesForCollection(currentCollection);
      sceneEl.innerHTML = "";
      sceneListEl.innerHTML = "";

      filtered.forEach((scene, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = sceneLabel(scene);
        sceneEl.appendChild(option);

        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `${{sceneLabel(scene)}} · ${{scene.collection}}`;
        button.setAttribute("aria-pressed", index === selectedSceneIndex ? "true" : "false");
        button.addEventListener("click", () => {{
          selectedSceneIndex = index;
          sceneEl.value = String(index);
          renderSceneSelection(filtered[index]);
          renderSceneButtons(filtered);
        }});
        sceneListEl.appendChild(button);
      }});

      if (!filtered.length) {{
        sceneListEl.innerHTML = "<p style='padding:12px;margin:0;color:#5b6470'>No scenes found.</p>";
        return;
      }}

      const initialIndex = Math.min(selectedSceneIndex, filtered.length - 1);
      selectedSceneIndex = initialIndex;
      sceneEl.value = String(initialIndex);
      renderSceneSelection(filtered[initialIndex]);
      renderSceneButtons(filtered);
    }}

    function renderSceneButtons(filtered) {{
      const buttons = Array.from(sceneListEl.querySelectorAll("button"));
      buttons.forEach((button, index) => {{
        button.setAttribute("aria-pressed", index === selectedSceneIndex ? "true" : "false");
      }});
    }}

    function renderSceneSelection(scene) {{
      if (!scene) return;
      const viewerUrl = `${{COPC_VIEWER_BASE}}${{encodeURIComponent(scene.href)}}`;
      copcUrlLabel.textContent = scene.href;
      viewerFrame.src = viewerUrl;
      openExternal.href = viewerUrl;
      setStatus(`${{scene.title}}\\n${{scene.href}}\\nCollection: ${{scene.collection}}`);
    }}

    collectionEl.addEventListener("change", () => {{
      selectedSceneIndex = 0;
      renderScenes();
    }});

    sceneEl.addEventListener("change", () => {{
      const currentCollection = collections[Number(collectionEl.value)]?.name || "All scenes";
      const filtered = currentCollection === "All scenes" ? scenes : scenesForCollection(currentCollection);
      const scene = filtered[Number(sceneEl.value)];
      selectedSceneIndex = Number(sceneEl.value);
      renderSceneSelection(scene);
      renderSceneButtons(filtered);
    }});

    openSceneButton.addEventListener("click", () => {{
      const currentCollection = collections[Number(collectionEl.value)]?.name || "All scenes";
      const filtered = currentCollection === "All scenes" ? scenes : scenesForCollection(currentCollection);
      const scene = filtered[selectedSceneIndex];
      if (scene) renderSceneSelection(scene);
    }});

    (async () => {{
      try {{
        setStatus("Loading STAC catalog...");
        await loadCatalog();
        if (!collections.length) {{
          collections = [{{ name: "All scenes", href: CATALOG_URL }}];
        }}
        if (!collections.some((item) => item.name === "All scenes")) {{
          collections.unshift({{ name: "All scenes", href: CATALOG_URL }});
        }}
        renderCollections();
        collectionEl.value = "0";
        renderScenes();
        setStatus(`Loaded ${{scenes.length}} COPC scene(s) from STAC.`);
      }} catch (error) {{
        setStatus(String(error && error.message ? error.message : error), true);
        copcUrlLabel.textContent = "failed to load catalog";
      }}
    }})();
  </script>
</body>
</html>
"""
        return body.encode("utf-8")

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        viewer_paths = {
            "/",
            "/fvc_structure-viewer",
            "/fvc_structure-viewer/",
            "/fvc_structure-viewer/3d",
        }
        if path == "/health":
            body = json.dumps({"status": "ok", "mode": "production-viewer"}).encode("utf-8")
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                ],
            )
            return [body]

        if path.startswith("/fvc_structure-viewer/api/stac/"):
            remote_path = path.removeprefix("/fvc_structure-viewer/api/stac/")
            remote_url = urljoin(production_stac_base_url, remote_path)
            body, status, headers, _ = fetch_remote_bytes(remote_url)
            headers = headers + [("Access-Control-Allow-Origin", "*")]
            headers.append(("Content-Length", str(len(body))))
            start_response(status, headers)
            return [body]

        if path in viewer_paths:
            body = render_viewer_html()
            start_response(
                "200 OK",
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                ],
            )
            return [body]

        start_response(
            "302 FOUND",
            [
                ("Location", production_viewer_url),
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", "0"),
            ],
        )
        return [b""]
else:
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

server = app if production_catalog_url else app.server
