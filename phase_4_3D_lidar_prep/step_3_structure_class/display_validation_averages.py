#!/usr/bin/env python3
"""Load average validation CSVs and print a combined dataframe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Preferred metric column order in the wide-format output.
_METRIC_ORDER = [
    # Table 5
    "OA (%)",
    "IoU Forb",
    "IoU Woody debris",
    "IoU Hummock",
    "IoU Shrub",
    "IoU Mallee",
    "Common occupied cells",
    "Projected occupied cells",
    "Refined occupied cells",
    # Table 6
    "Configured min Znorm (m)",
    "Configured max Znorm (m)",
    "Median Znorm (m)",
    "IQR (m)",
    "Outside configured range (%)",
    "Points",
    # TreeID
    "Watershed crowns (Step 4)",
    "Points in crowns (Step 4)",
    "Final non-zero treeIDs (Step 9)",
    "Mallee points with treeID",
    "Crown count delta (Step 9 - Step 4)",
    "Crown retention (%)",
    "Mean points per watershed crown",
    "Mean points per treeID",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display combined average validation results from a validation folder."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Validation results directory containing *_average.csv files.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save the combined dataframe to a CSV.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Display the long-format dataframe instead of the default wide manuscript-style table.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any expected CSV is missing (default: warn and skip).",
    )
    return parser.parse_args()


def _load_csv(path: Path, strict: bool) -> pd.DataFrame | None:
    if not path.exists():
        msg = f"Missing results file: {path}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"Warning: {msg}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def build_summary_dataframe(results_dir: Path, strict: bool = False) -> pd.DataFrame:
    table5 = _load_csv(results_dir / "table5_label_agreement_average.csv", strict)
    table6 = _load_csv(results_dir / "table6_height_plausibility_average.csv", strict)
    treeid = _load_csv(results_dir / "treeid_instance_summary_average.csv", strict)

    rows: list[dict[str, object]] = []

    if table5 is not None and not table5.empty:
        row = table5.iloc[0].to_dict()
        for metric, value in row.items():
            if metric in {"Tile", "Site condition"}:
                continue
            rows.append({"Source": "Table 5", "Group": "All tiles", "Metric": metric, "Value": value})

    if table6 is not None:
        for _, row in table6.iterrows():
            class_name = row.get("Class", "Unknown")
            for metric, value in row.to_dict().items():
                if metric in {"Tile", "Site condition", "Class"}:
                    continue
                rows.append({"Source": "Table 6", "Group": class_name, "Metric": metric, "Value": value})

    if treeid is not None and not treeid.empty:
        row = treeid.iloc[0].to_dict()
        for metric, value in row.items():
            if metric in {"Tile", "Site condition"}:
                continue
            rows.append({"Source": "TreeID summary", "Group": "All tiles", "Metric": metric, "Value": value})

    return pd.DataFrame(rows, columns=["Source", "Group", "Metric", "Value"])


def build_wide_summary_dataframe(long_df: pd.DataFrame) -> pd.DataFrame:
    """Single combined wide table — used for CSV export only."""
    if long_df.empty:
        return pd.DataFrame(columns=["Source", "Group"])
    wide_df = (
        long_df.pivot_table(
            index=["Source", "Group"],
            columns="Metric",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_df.columns.name = None

    fixed = ["Source", "Group"]
    known = [c for c in _METRIC_ORDER if c in wide_df.columns]
    rest = sorted(c for c in wide_df.columns if c not in fixed and c not in known)
    wide_df = wide_df[fixed + known + rest]
    return wide_df


def _pivot_source(long_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Pivot one source group into a wide table containing only its own metrics."""
    subset = long_df[long_df["Source"] == source].copy()
    if subset.empty:
        return pd.DataFrame()
    wide = (
        subset.pivot_table(index="Group", columns="Metric", values="Value", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.rename(columns={"Group": "Class / Group"})

    # Reorder columns in domain order.
    fixed = ["Class / Group"]
    known = [c for c in _METRIC_ORDER if c in wide.columns]
    rest = sorted(c for c in wide.columns if c not in fixed and c not in known)
    return wide[fixed + known + rest]


def print_wide_by_source(long_df: pd.DataFrame) -> None:
    """Print each source as its own sub-table so no row appears empty."""
    for source in ["Table 5", "Table 6", "TreeID summary"]:
        sub = _pivot_source(long_df, source)
        if sub.empty:
            print(f"\n── {source} ── (no data)\n")
            continue
        print(f"\n{'─' * 6} {source} {'─' * (50 - len(source))}")
        print(sub.to_string(index=False))
        print()


def _fmt(value: object) -> str:
    """Round floats for cleaner terminal display."""
    try:
        f = float(value)  # type: ignore[arg-type]
        if f != f:
            return "nan"
        if f == int(f):
            return str(int(f))
        return f"{f:.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_long = build_summary_dataframe(results_dir, strict=args.strict)

    if summary_long.empty:
        print("No data loaded — check that the results directory contains *_average.csv files.")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.float_format", _fmt)

    if args.long:
        print(summary_long.to_string(index=False))
    else:
        print_wide_by_source(summary_long)

    if args.save:
        if args.long:
            out_name = "validation_average_summary_long.csv"
            summary_long.to_csv(results_dir / out_name, index=False)
        else:
            out_name = "validation_average_summary_wide.csv"
            build_wide_summary_dataframe(summary_long).to_csv(results_dir / out_name, index=False)
        out_path = results_dir / out_name
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
