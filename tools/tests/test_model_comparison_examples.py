import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_model_examples import build_scene


class ModelComparisonExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "znorm_3Dlabels_m_L1_93.ply"

    def write_points(self, points):
        PlyData([PlyElement.describe(points, "vertex")]).write(self.path)

    def test_preserves_labels_counts_and_valid_prediction_zero(self):
        dtype = [(name, "f4" if name in ("x", "y", "z") else "i4")
                 for name in ("x", "y", "z", "semantic_gt", "semantic_pred", "instance_gt", "instance_pred")]
        points = np.array([(1, 2, 3, 0, 1, 0, 0), (4, 5, 6, 1, 2, 7, -1),
                           (7, 8, 9, 2, 2, 7, 3)], dtype=dtype)
        self.write_points(points)
        result = build_scene(self.path, "medium-93", "medium", "93", 3)
        self.assertEqual(result["semantic_counts"], [[1, 1, 1], [0, 1, 2]])
        self.assertEqual((result["reference_instances"], result["predicted_instances"]), (1, 2))
        self.assertIn([1.0, 2.0, 3.0, 0, 1, 0, 0], result["points"])
        self.assertEqual(result["points"], build_scene(self.path, "medium-93", "medium", "93", 3)["points"])

    def test_rejects_missing_or_unsupported_labels(self):
        self.write_points(np.zeros(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]))
        with self.assertRaisesRegex(ValueError, "missing fields"):
            build_scene(self.path, "medium-93", "medium", "93", 3)
        dtype = [(name, "f4" if name in ("x", "y", "z") else "i4")
                 for name in ("x", "y", "z", "semantic_gt", "semantic_pred", "instance_gt", "instance_pred")]
        points = np.zeros(1, dtype=dtype)
        points["semantic_pred"] = 4
        self.write_points(points)
        with self.assertRaisesRegex(ValueError, "unsupported semantic_pred"):
            build_scene(self.path, "medium-93", "medium", "93", 3)

    def test_command_includes_forest_test_scene(self):
        dtype = [(name, "f4" if name in ("x", "y", "z") else "i4")
                 for name in ("x", "y", "z", "semantic_gt", "semantic_pred", "instance_gt", "instance_pred")]
        points = np.zeros(2, dtype=dtype)
        paths = {
            "medium": ("znorm_3Dlabels_m_L1_93.ply", "znorm_3Dlabels_m_L1_192.ply"),
            "low": ("znorm_3Dlabels_l_L1_43.ply", "znorm_3Dlabels_l_L1_75.ply"),
            "forest": ("SCION_SCION_plot_31_annotated_test.ply", "SCION_SCION_plot_61_annotated_test.ply"),
        }
        for site, names in paths.items():
            directory = self.root / site
            directory.mkdir()
            for name in names:
                PlyData([PlyElement.describe(points, "vertex")]).write(directory / name)
        output = self.root / "examples.js"
        script = Path(__file__).resolve().parents[1] / "build_model_examples.py"
        subprocess.run(
            [sys.executable, str(script), "--medium-dir", str(self.root / "medium"),
             "--low-dir", str(self.root / "low"), "--forest-dir", str(self.root / "forest"),
             "--output", str(output)],
            check=True, capture_output=True, text=True,
        )
        data = output.read_text(encoding="utf-8").removeprefix("window.FVC_MODEL_COMPARISONS = ").removesuffix(";\n")
        scenes = json.loads(data)
        forest = next(scene for scene in scenes if scene["id"] == "forest-scion-61")
        self.assertEqual((forest["site"], forest["source"], forest["total"]),
                         ("forest", paths["forest"][1], 2))
        forest31 = next(scene for scene in scenes if scene["id"] == "forest-scion-31")
        self.assertEqual((forest31["site"], forest31["source"], forest31["total"]),
                         ("forest", paths["forest"][0], 2))


if __name__ == "__main__":
    unittest.main()
