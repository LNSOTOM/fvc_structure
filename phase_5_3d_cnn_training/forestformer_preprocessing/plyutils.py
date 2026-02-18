"""Small utility module expected by ForestFormer preprocessing.

ForestFormer3D's data export code imports `read_ply` from a top-level `plyutils` module.
This implementation reads PLY files produced by our pipeline and returns a dict of numpy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
from pyntcloud import PyntCloud


def read_ply(filename: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a PLY file and return a dict of column -> numpy array.

    Args:
        filename: Path to a .ply file

    Returns:
        Dict mapping each PLY property name to a numpy array.
    """
    ply_path = Path(filename)
    cloud = PyntCloud.from_file(str(ply_path))
    points_df = cloud.points

    result: Dict[str, np.ndarray] = {}
    for col in points_df.columns:
        result[col] = points_df[col].to_numpy()

    return result
