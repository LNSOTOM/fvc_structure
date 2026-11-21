"""Shared configuration loading for the Phase 3 imagery preparation scripts.

Resolution order for the config file:
1. Explicit ``--config`` command line argument.
2. ``FVC_PHASE3_CONFIG`` environment variable.
3. ``phase_3_2D_imagery_prep/configs/imagery_prep_job.local.yml``.
"""

import argparse
import os
from pathlib import Path

import yaml

PHASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PHASE_DIR / "configs" / "imagery_prep_job.local.yml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the phase 3 YAML job config."""
    if path is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", default=None)
        args, _ = parser.parse_known_args()
        path = args.config or os.environ.get("FVC_PHASE3_CONFIG") or DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Phase 3 config not found: {cfg_path}\n"
            "Copy configs/imagery_prep_job.example.yml to "
            "configs/imagery_prep_job.local.yml (or pass --config / set "
            "FVC_PHASE3_CONFIG) and edit the paths."
        )
    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Phase 3 config must contain a YAML mapping: {cfg_path}")
    return config
