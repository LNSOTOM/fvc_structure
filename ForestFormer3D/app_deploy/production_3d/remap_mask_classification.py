"""Copy LAS/LAZ files and remap mask classification codes to ASPRS-compatible ones.

Default mapping (medium/maskCoreg → ASPRS-ish so maplibre-gl-lidar's
Classification palette renders):
    65 (Woody debris) -> 3
    66 (Hummock)      -> 0
    67 (Shrub)        -> 1
    68 (Mallee)       -> 2

Override with --remap "old:new,old:new,...".

RGB and all other dimensions are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np

DEFAULT_REMAP = {65: 3, 66: 0, 67: 1, 68: 2}


def parse_remap(spec: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        old, new = pair.split(":")
        out[int(old)] = int(new)
    return out


def remap_file(src: Path, dst: Path, remap: dict[int, int]) -> None:
    las = laspy.read(str(src))
    cls = np.asarray(las.classification, dtype=np.int32)
    new_cls = cls.copy()
    for old, new in remap.items():
        new_cls[cls == old] = new
    las.classification = new_cls.astype(las.classification.dtype)
    dst.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument(
        "--remap",
        default=None,
        help='Comma-separated "old:new" pairs, e.g. "64:0,67:1,68:2,65:3". Defaults to medium mapping.',
    )
    args = ap.parse_args()

    remap = parse_remap(args.remap) if args.remap else DEFAULT_REMAP
    print(f"using remap: {remap}")

    files = [p for p in args.input_root.iterdir() if p.suffix.lower() in {".las", ".laz"}]
    if not files:
        raise SystemExit(f"No LAS/LAZ files found in {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(files):
        dst = args.output_root / src.name
        print(f"remap {src} -> {dst}")
        remap_file(src, dst, remap)


if __name__ == "__main__":
    main()
