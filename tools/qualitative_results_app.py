from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parents[1]
    / "ForestFormer3D"
    / "tools"
    / "qualitative_results_app.py"
)

if not TARGET.exists():
    raise FileNotFoundError(f"Missing target script: {TARGET}")

runpy.run_path(str(TARGET), run_name="__main__")