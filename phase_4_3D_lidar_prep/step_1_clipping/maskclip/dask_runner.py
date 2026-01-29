import time
from pathlib import Path
from typing import List, Tuple

from dask.distributed import Client, as_completed
from tqdm import tqdm

from .core import classify_single

def _task_for_tile(las_file: Path, raster_dir: Path, mask_dir: Path, out_dir: Path) -> Tuple[Path, Path, Path, Path]:
    tile_id = las_file.stem.split(".")[-1]
    raster = raster_dir / f"tiles_multispectral.{tile_id}.tif"
    mask = mask_dir / f"mask_structure_3072.{tile_id}.tif"
    out_path = out_dir / f"mask_{las_file.stem}.las"
    return las_file, raster, mask, out_path

def classify_many_dask(las_dir: Path, raster_dir: Path, mask_dir: Path, out_dir: Path, skip_existing: bool = True, pattern: str = "tiles_multispectral*.las", write_signature: bool = True) -> List[Path]:
    las_dir = Path(las_dir)
    raster_dir = Path(raster_dir)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(las_dir.glob(pattern))
    if not las_files:
        raise SystemExit(f"No LAS tiles found in {las_dir} with pattern {pattern}")

    tasks = []
    for las in las_files:
        las_f, raster, mask, out_path = _task_for_tile(las, raster_dir, mask_dir, out_dir)
        if not raster.exists() or not mask.exists():
            continue
        if skip_existing and out_path.exists():
            continue
        tasks.append((las_f, raster, mask, out_path))

    if not tasks:
        return []

    start = time.time()
    written = []

    with Client(processes=False) as client:
        print(f"Dask dashboard: {client.dashboard_link}")
        futures = [client.submit(classify_single, las, raster, mask, out_path) for (las, raster, mask, out_path) in tasks]
        successful = failed = 0
        with tqdm(total=len(futures), desc="Mask → LAS", unit="tile") as pbar:
            for fut, (las, raster, mask, out_path) in zip(as_completed(futures), tasks):
                try:
                    pts = fut.result()
                    successful += 1
                    written.append(out_path)
                    print(f"✅ {las.name} → {out_path.name} ({pts:,} pts)")
                except Exception as e:
                    failed += 1
                    print(f"❌ Error {las.name}: {e}")
                pbar.update(1)

    elapsed = time.time() - start
    if write_signature:
        sig = out_dir / "mask_signature.txt"
        with sig.open("w") as f:
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LAS dir: {las_dir}\n")
            f.write(f"Raster dir: {raster_dir}\n")
            f.write(f"Mask dir: {mask_dir}\n")
            f.write(f"Pattern: {pattern}\n")
            f.write(f"Output dir: {out_dir}\n")
            f.write(f"Total LAS: {len(las_files)}\n")
            f.write(f"Successful: {len(written)}\n")
            f.write(f"Failed: {len(las_files) - len(written)}\n")
            f.write(f"Elapsed (s): {elapsed:.2f}\n")
        print(f"Signature written: {sig}")

    print(f"✅ Done classifying. Successful: {len(written)}, Failed: {len(las_files) - len(written)}, Elapsed: {elapsed:.2f}s")
    return written
