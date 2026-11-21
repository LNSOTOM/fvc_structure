"""Shared configuration loading for the Phase 2 co-registration scripts.

Resolution order for the config file:
1. Explicit ``--config`` command line argument.
2. ``FVC_PHASE2_CONFIG`` environment variable.
3. ``phase_2_lidar_coregistration/configs/coregistration_job.local.yml``.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# phase_2_lidar_coregistration/
PHASE_DIR = Path(__file__).resolve().parents[2]
# gefolki/ (added to sys.path so `from python.algorithm import EFolki` works)
GEFOLKI_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = PHASE_DIR / "configs" / "coregistration_job.local.yml"


def add_gefolki_to_path() -> None:
    """Make the vendored GeFolki library importable as `python.*`."""
    for path in (GEFOLKI_DIR, GEFOLKI_DIR / "python"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the phase 2 YAML job config."""
    if path is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", default=None)
        args, _ = parser.parse_known_args()
        path = args.config or os.environ.get("FVC_PHASE2_CONFIG") or DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Phase 2 config not found: {cfg_path}\n"
            "Copy configs/coregistration_job.example.yml to "
            "configs/coregistration_job.local.yml (or pass --config / set "
            "FVC_PHASE2_CONFIG) and edit the paths."
        )
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)
