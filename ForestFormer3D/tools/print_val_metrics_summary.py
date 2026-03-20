#!/usr/bin/env python3
import argparse
import glob
import re
from typing import Optional


def to_percent(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    value = float(raw)
    return value * 100 if value <= 1 else value


def fmt_percent(value: Optional[float]) -> str:
    return f"{value:7.2f}%" if value is not None else "   N/A"


def parse_line(line: str):
    if "Epoch(val)" not in line:
        return None

    epoch_match = re.search(r"Epoch\(val\) \[(\d+)\]", line)
    if not epoch_match:
        return None

    def extract(pattern: str) -> Optional[float]:
        match = re.search(pattern, line)
        return to_percent(match.group(1)) if match else None

    return {
        "epoch": int(epoch_match.group(1)),
        "prec": extract(r"(?:precision|Precision): ([0-9.]+)"),
        "rec": extract(r"(?:recall|Recall): ([0-9.]+)"),
        "f1": extract(r"F1: ([0-9.]+)"),
        "ground": extract(r"IoU_?ground: ([0-9.]+)"),
        "wood": extract(r"IoU_?wood: ([0-9.]+)"),
        "leaf": extract(r"IoU_?leaf: ([0-9.]+)"),
        "miou": extract(r"mIoU: ([0-9.]+)"),
        "harmonic": extract(r"(?:harmonic|Harmonic): ([0-9.]+)"),
    }


def print_block(title: str):
    print("╔════════════════════════════════════════════════════════════════════╗")
    print(f"║ {title:<66} ║")
    print("╚════════════════════════════════════════════════════════════════════╝")


def print_metrics(row):
    print(f"Epoch:  {row['epoch']}")
    print(f"  Prec:     {fmt_percent(row['prec'])}")
    print(f"  Rec:      {fmt_percent(row['rec'])}")
    print(f"  F1:       {fmt_percent(row['f1'])}")
    print(f"  Ground:   {fmt_percent(row['ground'])}")
    print(f"  Wood:     {fmt_percent(row['wood'])}")
    print(f"  Leaf:     {fmt_percent(row['leaf'])}")
    print(f"  mIoU:     {fmt_percent(row['miou'])}")
    print(f"  Harmonic: {fmt_percent(row['harmonic'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Print validation summary: best harmonic, epoch 3000, and top-k by mIoU."
    )
    parser.add_argument(
        "--log-glob",
        default="/workspace/work_dirs/exp2A_epoch3000_train/*/*.log",
        help="Glob pattern to find MMEngine training logs.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3000,
        help="Epoch to print in the dedicated epoch section.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top epochs to print by mIoU.",
    )
    args = parser.parse_args()

    rows = []
    for log_path in sorted(glob.glob(args.log_glob)):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = parse_line(line)
                if parsed is not None:
                    rows.append(parsed)

    if not rows:
        print(f"No validation rows found for: {args.log_glob}")
        return

    print_block("BEST HARMONIC VALIDATION METRICS")
    harmonic_rows = [row for row in rows if row["harmonic"] is not None]
    if harmonic_rows:
        best_harmonic = max(harmonic_rows, key=lambda row: row["harmonic"])
        print_metrics(best_harmonic)
    else:
        print("No harmonic metrics found in logs")

    print()
    print_block(f"EPOCH {args.epoch} VALIDATION METRICS")
    target_rows = [row for row in rows if row["epoch"] == args.epoch]
    if target_rows:
        print_metrics(target_rows[0])
    else:
        print(f"Epoch {args.epoch} not found in logs")

    print()
    print_block(f"TOP {args.top_k} EPOCHS BY mIoU")
    sorted_rows = sorted(
        [row for row in rows if row["miou"] is not None],
        key=lambda row: row["miou"],
        reverse=True,
    )
    for idx, row in enumerate(sorted_rows[: args.top_k], start=1):
        print(
            f"{idx}. Epoch {row['epoch']:4d}: "
            f"mIoU={fmt_percent(row['miou']).strip()} | "
            f"F1={fmt_percent(row['f1']).strip()} | "
            f"Harmonic={fmt_percent(row['harmonic']).strip()}"
        )
        print(
            "   └─ "
            f"Prec={fmt_percent(row['prec']).strip()} | "
            f"Rec={fmt_percent(row['rec']).strip()} | "
            f"Ground={fmt_percent(row['ground']).strip()} | "
            f"Wood={fmt_percent(row['wood']).strip()} | "
            f"Leaf={fmt_percent(row['leaf']).strip()}"
        )


if __name__ == "__main__":
    main()
