import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from pyntcloud import PyntCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_lidar

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ForestFormer3D" / "data" / "ForAINetV2"))
from load_forainetv2_data import export


class PrepareLidarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "tile.las"
        self.output = self.root / "output"
        self.job = {"input_las": str(self.input), "output_dir": str(self.output),
                    "class_profile": "three_class_hsm"}

    def write_las(self, classes, include_tree=True):
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
        if include_tree:
            header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))
        las = laspy.LasData(header)
        n = len(classes)
        las.x = 461000.0 + np.arange(n) * 0.1
        las.y = 6237000.0 + np.arange(n) * 0.1
        las.z = 60.0 + np.arange(n) * 0.1
        las.classification = classes
        las["Z_norm"] = np.arange(n, dtype=np.float32) * 0.1
        if include_tree:
            las["treeID"] = np.arange(n, dtype=np.uint32)
        las.write(self.input)
        return las

    def test_three_class_labels_and_coordinate_reconstruction(self):
        source = self.write_las([0, 66, 67, 68])
        prepare_lidar.prepare(self.job)
        points = PyntCloud.from_file(str(self.output / "tile.ply")).points
        self.assertEqual(points["semantic_seg"].tolist(), [3, 4, 5])
        self.assertEqual(points["treeID"].tolist(), [1, 2, 3])
        restored = points[["x", "y", "z"]].to_numpy() + np.load(self.output / "tile_offset.npy")
        original = np.column_stack((source.x, source.y, source.z))[1:]
        np.testing.assert_allclose(restored, original, rtol=0, atol=1e-4)
        _, labels, _, _, _, _, _ = export(str(self.output / "tile.ply"))
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_three_class_rejects_forb_and_woody_debris(self):
        for label in (64, 65):
            with self.subTest(label=label):
                self.write_las([label, 68])
                with self.assertRaisesRegex(ValueError, "rejects forb"):
                    prepare_lidar.prepare(self.job)
                self.assertFalse(self.output.exists())

    def test_historical_profile_keeps_five_classes(self):
        self.write_las([64, 65, 66, 67, 68])
        self.job["class_profile"] = "historical_five_class"
        prepare_lidar.prepare(self.job)
        points = PyntCloud.from_file(str(self.output / "tile.ply")).points
        self.assertEqual(points["semantic_seg"].tolist(), [1, 2, 3, 4, 5])

    def test_missing_tree_ids_fail_instead_of_defaulting(self):
        self.write_las([68], include_tree=False)
        with self.assertRaisesRegex(ValueError, "treeID"):
            prepare_lidar.prepare(self.job)

    def test_existing_outputs_are_not_overwritten(self):
        self.write_las([68])
        self.output.mkdir()
        output = self.output / "tile_offset.npy"
        output.write_bytes(b"existing")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            prepare_lidar.prepare(self.job)
        self.assertEqual(output.read_bytes(), b"existing")
        self.assertFalse((self.output / "tile.ply").exists())


if __name__ == "__main__":
    unittest.main()
