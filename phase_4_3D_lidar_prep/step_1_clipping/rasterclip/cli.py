import argparse
from pathlib import Path

from .core import clip_single, clip_many
from .dask_runner import clip_many_dask


def main():
    p = argparse.ArgumentParser(description="Clip LAS to raster tile bounds and reproject to raster CRS")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("single", help="Clip a single raster tile")
    s1.add_argument("input_las", type=Path)
    s1.add_argument("aoi_raster", type=Path)
    s1.add_argument("out_path", type=Path)

    s2 = sub.add_parser("many", help="Clip many tiles sequentially")
    s2.add_argument("input_las", type=Path)
    s2.add_argument("aoi_raster_dir", type=Path)
    s2.add_argument("out_dir", type=Path)
    s2.add_argument("--pattern", default="tiles_multispectral*.tif")
    s2.add_argument("--no-skip", action="store_true", help="Do not skip existing outputs")

    s3 = sub.add_parser("dask", help="Clip many tiles in parallel using Dask")
    s3.add_argument("input_las", type=Path)
    s3.add_argument("aoi_raster_dir", type=Path)
    s3.add_argument("out_dir", type=Path)
    s3.add_argument("--pattern", default="tiles_multispectral*.tif")
    s3.add_argument("--no-skip", action="store_true", help="Do not skip existing outputs")
    s3.add_argument("--no-signature", action="store_true", help="Do not write signature file")

    args = p.parse_args()

    if args.cmd == "single":
        points = clip_single(args.input_las, args.aoi_raster, args.out_path)
        print(f"✅ PDAL points processed: {points}")
        print(f"✅ Bounds-only clip written: {args.out_path}")
    elif args.cmd == "many":
        outputs = clip_many(args.input_las, args.aoi_raster_dir, args.out_dir, pattern=args.pattern, skip_existing=not args.no_skip)
        print(f"✅ Wrote {len(outputs)} LAS tiles to {args.out_dir}")
    else:
        outputs = clip_many_dask(
            args.input_las,
            args.aoi_raster_dir,
            args.out_dir,
            pattern=args.pattern,
            skip_existing=not args.no_skip,
            write_signature=not args.no_signature,
        )
        print(f"✅ Wrote {len(outputs)} LAS tiles to {args.out_dir} (Dask)")


if __name__ == "__main__":
    main()
