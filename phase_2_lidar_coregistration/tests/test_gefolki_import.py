import subprocess
import sys
import unittest
from pathlib import Path


class GeFolkiImportTest(unittest.TestCase):
    def test_vendored_library_imports_from_another_directory(self):
        scripts = Path(__file__).resolve().parents[1] / "gefolki" / "python_myscripts"
        code = (
            f"import sys; sys.path.insert(0, {str(scripts)!r}); "
            "from phase2_config import add_gefolki_to_path; "
            "add_gefolki_to_path(); "
            "from python.algorithm import EFolki; "
            "from python.tools import wrapData"
        )
        subprocess.run([sys.executable, "-c", code], cwd=scripts.parent, check=True)


if __name__ == "__main__":
    unittest.main()
