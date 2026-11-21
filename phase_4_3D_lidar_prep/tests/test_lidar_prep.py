import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np
import rasterio
import yaml
from pyproj import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lidar_prep


class LidarPrepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.add_crs(CRS.from_epsg(7854))
        las = laspy.LasData(header)
        las.x = [100.25, 100.75]
        las.y = [200.25, 200.75]
        las.z = [1, 2]
        self.source = {"icp_las": root / "input.las",
                       "predictor_raster": root / "predictor.tif",
                       "mask_raster": root / "mask.tif"}
        las.write(self.source["icp_las"])
        for key in ("predictor_raster", "mask_raster"):
            with rasterio.open(self.source[key], "w", driver="GTiff", width=2, height=2,
                               count=1, dtype="uint8", crs="EPSG:7854",
                               transform=from_origin(100, 201, 0.5, 0.5),
                               nodata=255) as dst:
                dst.write(np.array([[64, 65], [68, 255]], dtype="uint8"), 1)

    def test_inputs_accept_matching_crs_and_mask(self):
        lidar_prep.check_inputs(self.source)

    def test_inputs_reject_wrong_crs_and_mask_codes(self):
        with rasterio.open(self.source["mask_raster"], "r+") as mask:
            mask.crs = "EPSG:7855"
        with self.assertRaisesRegex(ValueError, "mask raster CRS differ"):
            lidar_prep.check_inputs(self.source)
        with rasterio.open(self.source["mask_raster"], "r+") as mask:
            mask.crs = "EPSG:7854"
            mask.write(np.array([[64, 3], [68, 255]], dtype="uint8"), 1)
        with self.assertRaisesRegex(ValueError, "Mask must contain structure codes"):
            lidar_prep.check_inputs(self.source)

    def test_intermediate_requires_matching_normalized_points(self):
        las = laspy.read(self.source["icp_las"])
        las.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
        normalized = Path(self.tmp.name) / "normalized.las"
        las.write(normalized)
        lidar_prep.check_intermediate(normalized, self.source["icp_las"], normalized=True)
        las.points = las.points[:1]
        las.write(normalized)
        with self.assertRaisesRegex(ValueError, "all input points"):
            lidar_prep.check_intermediate(normalized, self.source["icp_las"], normalized=True)

    def test_all_checks_later_outputs_before_running_any_stage(self):
        root = Path(self.tmp.name)
        target = root / "fused.las"
        target.write_bytes(b"existing")
        job = {"inputs": {key: str(path) for key, path in self.source.items()},
               "outputs": {"masked_las": str(root / "masked.las"),
                           "ground_las": str(root / "ground.las"),
                           "normalized_las": str(root / "normalized.las"),
                           "fused_las": str(target)},
               "csf": {"resolution": 0.5, "iterations": 500, "threshold": 0.01,
                       "rigidness": 2}, "normalization": {"k": 3}}
        config = root / "job.yml"
        config.write_text(yaml.safe_dump(job), encoding="utf-8")
        with patch.object(sys, "argv", ["lidar_prep", "--config", str(config), "--stage", "all"]):
            with patch.object(lidar_prep, "classify_single") as classify:
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    lidar_prep.main()
                classify.assert_not_called()
        self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
