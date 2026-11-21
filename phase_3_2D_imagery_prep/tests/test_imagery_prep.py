import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box


SCRIPT = Path(__file__).resolve().parents[1] / "imagery_prep.py"


class ImageryPrepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        for folder in ("annotations", "reference"):
            (self.root / folder).mkdir()
        self.raster(self.root / "orthomosaic.tif", 2, 0.05, 10)
        self.raster(self.root / "reference" / "tile_26.tif", 10, 0.01, 1)
        gpd.GeoDataFrame(
            {"id": [26], "annotated": [1]},
            geometry=[box(0, 0, 0.1, 0.1)], crs="EPSG:7854",
        ).to_file(self.root / "grid.shp")
        gpd.GeoDataFrame(
            {"structure": ["forb", "grass_hummock", "unknown"]},
            geometry=[box(0, 0, 0.03, 0.1), box(0.06, 0, 0.1, 0.1), box(0.04, 0, 0.05, 0.1)],
            crs="EPSG:7854",
        ).to_file(self.root / "annotations" / "mask.26.shp")
        self.config = {
            "tile": {"orthomosaic": str(self.root / "orthomosaic.tif"),
                     "grid": str(self.root / "grid.shp"),
                     "output_dir": str(self.root / "raw")},
            "stack": {"input_dir": str(self.root / "raw"),
                      "output_dir": str(self.root / "stacked")},
            "classify": {"input_dir": str(self.root / "annotations"),
                         "output_dir": str(self.root / "classified")},
            "rasterize": {"input_dir": str(self.root / "classified"),
                          "reference_dir": str(self.root / "reference"),
                          "reference_pattern": "tile_{id}.tif",
                          "output_dir": str(self.root / "labels1")},
            "downsample": {"input_dir": str(self.root / "labels1"),
                           "output_dir": str(self.root / "labels5")},
        }

    def raster(self, path, size, resolution, bands):
        with rasterio.open(
            path, "w", driver="GTiff", height=size, width=size,
            count=bands, dtype="uint8", crs="EPSG:7854",
            transform=from_origin(0, 0.1, resolution, resolution),
        ) as dst:
            for band in range(1, bands + 1):
                dst.write(np.full((size, size), band, dtype=np.uint8), band)

    def run_stage(self, stage):
        config_path = self.root / "config.yml"
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(config_path), "--stage", stage],
            env=env, capture_output=True, text=True,
        )

    def test_complete_historical_workflow_and_overwrite_guard(self):
        result = self.run_stage("all")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("removing unrecognized", result.stdout)
        with rasterio.open(self.root / "stacked" / "tiles_multispectral.26.tif") as src:
            self.assertEqual(src.count, 5)
            self.assertEqual(src.read()[:, 0, 0].tolist(), [2, 4, 6, 8, 10])
        classified = gpd.read_file(self.root / "classified" / "mask.26.shp")
        self.assertEqual(classified["struct_id"].tolist(), [0, 2])
        with rasterio.open(self.root / "labels1" / "mask_structure_3072.26.tif") as src:
            self.assertEqual((src.width, src.height), (10, 10))
            self.assertEqual(src.res, (0.01, 0.01))
            self.assertEqual(set(np.unique(src.read(1))), {0, 2, 255})
        with rasterio.open(self.root / "labels5" / "mask_structure_3072.26.tif") as src:
            self.assertEqual((src.width, src.height), (2, 2))
            self.assertEqual(src.res, (0.05, 0.05))
            self.assertEqual(set(np.unique(src.read(1))), {64, 66})
            self.assertEqual(src.nodata, 255)
        result = self.run_stage("all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)

    def test_rejects_missing_type_field(self):
        gpd.GeoDataFrame(
            {"not_type": ["grass"]}, geometry=[box(0, 0, 0.1, 0.1)], crs="EPSG:7854",
        ).to_file(self.root / "annotations" / "mask.26.shp")
        result = self.run_stage("classify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No vegetation type field", result.stderr)

    def test_rejects_coarse_reference(self):
        self.raster(self.root / "reference" / "tile_26.tif", 2, 0.05, 1)
        result = self.run_stage("classify")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_stage("rasterize")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference must have 1 cm pixels", result.stderr)

    def test_downsample_3072_tile_matches_legacy_batch_grid(self):
        labels = self.root / "labels1"
        labels.mkdir()
        input_path = labels / "mask_structure_3072.26.tif"
        transform = from_origin(461771.464696, 6237536.887523, 0.01, 0.01)
        data = np.full((3072, 3072), 255, dtype=np.uint8)
        data[:1536, :] = 0
        data[1536:, :1536] = 4
        with rasterio.open(
            input_path, "w", driver="GTiff", height=3072, width=3072,
            count=1, dtype="uint8", crs="EPSG:7854", transform=transform, nodata=255,
        ) as dst:
            dst.write(data, 1)
            bounds = dst.bounds
        result = self.run_stage("downsample")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        legacy = self.root / "legacy.tif"
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        subprocess.run(
            ["gdalwarp", "-overwrite", "-q", "-s_srs", "EPSG:7854",
             "-t_srs", "EPSG:7854", "-tr", "0.05", "0.05", "-r", "nearest",
             "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
             "-ot", "Byte", "-of", "GTiff", "-srcnodata", "255", "-dstnodata", "255",
             str(input_path), str(legacy)],
            env=env, check=True,
        )
        with rasterio.open(self.root / "labels5" / input_path.name) as actual, rasterio.open(legacy) as expected:
            self.assertEqual((actual.width, actual.height), (614, 614))
            self.assertEqual(actual.transform, expected.transform)
            expected_data = expected.read(1)
            expected_data[expected_data == 0] = 64
            expected_data[expected_data == 4] = 68
            np.testing.assert_array_equal(actual.read(1), expected_data)


if __name__ == "__main__":
    unittest.main()
