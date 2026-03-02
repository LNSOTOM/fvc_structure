#!/usr/bin/env python

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def _compile(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if not pattern:
        return None
    return re.compile(pattern)


def _filter_items(
    items: Iterable[Dict[str, Any]],
    field: str,
    include: Optional[re.Pattern[str]],
    exclude: Optional[re.Pattern[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            value = _get_nested(it, field)
        except KeyError:
            # If the field is missing, drop the item (explicit and safe).
            continue

        value_str = str(value)
        if include and not include.search(value_str):
            continue
        if exclude and exclude.search(value_str):
            continue
        out.append(it)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a OneFormer3D infos PKL (dict with metainfo/data_list) into a split PKL. "
            "Filtering is done by regex over a chosen field (default: lidar_points.lidar_path)."
        )
    )
    parser.add_argument('--in-pkl', required=True, help='Input infos PKL path')
    parser.add_argument('--out-pkl', required=True, help='Output infos PKL path')
    parser.add_argument(
        '--field',
        default='lidar_points.lidar_path',
        help='Dotted field to match against (e.g., lidar_points.lidar_path, pts_semantic_mask_path)',
    )
    parser.add_argument(
        '--include-regex',
        default=None,
        help='Keep items where field matches this regex (if omitted, keeps all).',
    )
    parser.add_argument(
        '--exclude-regex',
        default=None,
        help='Drop items where field matches this regex (optional).',
    )
    args = parser.parse_args()

    in_path = Path(args.in_pkl)
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = pickle.loads(in_path.read_bytes())
    if not isinstance(payload, dict) or 'data_list' not in payload:
        raise ValueError(
            f"Unexpected PKL schema in {in_path}. Expected a dict with key 'data_list'."
        )

    items = payload['data_list']
    if not isinstance(items, list):
        raise ValueError(
            f"Unexpected PKL schema in {in_path}. 'data_list' must be a list."
        )

    include = _compile(args.include_regex)
    exclude = _compile(args.exclude_regex)
    filtered = _filter_items(items, field=args.field, include=include, exclude=exclude)

    out_payload = dict(payload)
    out_payload['data_list'] = filtered
    out_path.write_bytes(pickle.dumps(out_payload))

    print(f"Input:  {in_path} ({len(items)} items)")
    print(f"Output: {out_path} ({len(filtered)} items)")
    print(f"Field: {args.field}")
    print(f"Include: {args.include_regex!r}")
    print(f"Exclude: {args.exclude_regex!r}")


if __name__ == '__main__':
    main()
