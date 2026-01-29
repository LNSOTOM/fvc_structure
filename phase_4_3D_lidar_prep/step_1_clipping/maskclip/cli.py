import argparse
from pathlib import Path

from .dask_runner import classify_many_dask

def main():
    parser = argparse.ArgumentParser(description="Mask-based LAS classification (Dask)")
    parser.add_argument("las_dir", type=Path, help="Directory of clipped LAS tiles")
    parser.add_argument("raster_dir", type=Path, help="Directory of predictor raster tiles")
    parser.add_argument("mask_dir", type=Path, help="Directory of mask raster tiles")
    parser.add_argument("out_dir", type=Path, help="Output directory for classified LAS")
    parser.add_argument("--pattern", default="tiles_multispectral*.las", help="Pattern for LAS tiles")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing outputs")
    parser.add_argument("--no-signature", action="store_true", help="Do not write signature file")

    args = parser.parse_args()
    outs = classify_many_dask(
        args.las_dir,
        args.raster_dir,
        args.mask_dir,
        args.out_dir,
        skip_existing=not args.no_skip,
        pattern=args.pattern,
        write_signature=not args.no_signature,
    )
    print(f"✅ Wrote {len(outs)} classified LAS tiles to {args.out_dir}")

if __name__ == "__main__":
    main()
