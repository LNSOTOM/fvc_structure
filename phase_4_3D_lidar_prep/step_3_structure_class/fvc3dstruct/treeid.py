import numpy as np


def assign_treeID_optionA(las_z, classification, label_grid,
                          xmin, ymin, grid_res):
    """
    Assign treeID only for points classified as MALLEE (68).

    Returns array treeID where:
      - mallee canopy points get crown label
      - everything else = 0
    """

    treeID = np.zeros(len(classification), dtype=np.uint16)

    if label_grid is None:
        return treeID

    ix = ((las_z.x - xmin) / grid_res).astype(int)
    iy = ((las_z.y - ymin) / grid_res).astype(int)

    h, w = label_grid.shape
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    mask_mallee = classification == 68
    treeID[mask_mallee] = label_grid[iy[mask_mallee], ix[mask_mallee]]

    return treeID
