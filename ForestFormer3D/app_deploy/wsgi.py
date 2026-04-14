import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.qualitative_results_app import (
    create_app,
    scan_experiments,
    scan_named_experiments,
)

WORK_DIRS_ROOT = Path(os.getenv("WORK_DIRS_ROOT", "/app/data/work_dirs"))
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
    raise RuntimeError(
        "No experiments found. Set WORK_DIRS_ROOT and/or SEGMENT_ANY_TREE_EXPERIMENTS_JSON."
    )

app = create_app(
    experiment_families=experiment_families,
    max_points=MAX_POINTS,
    seed=SEED,
    marker_size=MARKER_SIZE,
)
server = app.server
