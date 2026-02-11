import numpy as np


def assign_treeID_optionA(las_z, label_grid, xmin, ymin, grid_res, classification=None):
    """
    Assign treeID for all points based on watershed labels.

    Returns array treeID where:
      - all points get crown label from the grid
      - unlabeled areas = 0
    """

    treeID = np.zeros(len(las_z.points), dtype=np.uint16)

    if label_grid is None:
        return treeID

    ix = ((las_z.x - xmin) / grid_res).astype(int)
    iy = ((las_z.y - ymin) / grid_res).astype(int)

    h, w = label_grid.shape
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    treeID = label_grid[iy, ix].astype(np.uint16)

    return treeID
