from pathlib import Path
import importlib
import sys


BASE_DIR = Path(__file__).resolve().parent / "ForestFormer3D"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

qualitative_results_app = importlib.import_module("app_deploy.wsgi")
server = qualitative_results_app.server
