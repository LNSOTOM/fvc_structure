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
    if "Epoch(test)" not in line:
        return None

    epoch_match = re.search(r"Epoch\(test\) \[(\d+)\]", line)

    def extract(pattern: str) -> Optional[float]:
        match = re.search(pattern, line)
        return to_percent(match.group(1)) if match else None

    return {
        "test_epoch": int(epoch_match.group(1)) if epoch_match else None,
        "prec": extract(r"(?:precision|Precision|Prec): ([0-9.]+)"),
        "rec": extract(r"(?:recall|Recall|Rec): ([0-9.]+)"),
        "f1": extract(r"F1: ([0-9.]+)"),
        "ground": extract(r"IoU_?ground: ([0-9.]+)"),
        "wood": extract(r"IoU_?wood: ([0-9.]+)"),
        "leaf": extract(r"IoU_?leaf: ([0-9.]+)"),
        "miou": extract(r"mIoU: ([0-9.]+)"),
        "harmonic": extract(r"(?:harmonic|Harmonic): ([0-9.]+)"),
    }


def extract_checkpoint_from_log_text(text: str) -> Optional[str]:
    # Common MMEngine patterns
    match = re.search(r"Load checkpoint from\s+([^\s]+\.pth)", text)
    if match:
        return match.group(1)

    # Config dumps often include: load_from = '/path/to/file.pth'
    match = re.search(r"load_from\s*=\s*'([^']+\.pth)'", text)
    if match:
        return match.group(1)

    # Fallback: any .pth path in text (first hit)
    match = re.search(r"(/[^\s'\"]+\.pth)", text)
    if match:
        return match.group(1)
    return None


def print_block(title: str):
    print("╔════════════════════════════════════════════════════════════════════╗")
    print(f"║ {title:<66} ║")
    print("╚════════════════════════════════════════════════════════════════════╝")


def print_metrics(row):
    print(f"Run:    {row['source']}")
    if row.get("checkpoint"):
        print(f"Ckpt:   {row['checkpoint']}")
    if row["test_epoch"] is not None:
        print(f"T-Epoch:{row['test_epoch']}")
    print(f"  Prec:     {fmt_percent(row['prec'])}")
    print(f"  Rec:      {fmt_percent(row['rec'])}")
    print(f"  F1:       {fmt_percent(row['f1'])}")
    print(f"  Ground:   {fmt_percent(row['ground'])}")
    print(f"  Wood:     {fmt_percent(row['wood'])}")
    print(f"  Leaf:     {fmt_percent(row['leaf'])}")
    print(f"  mIoU:     {fmt_percent(row['miou'])}")
    print(f"  Harmonic: {fmt_percent(row['harmonic'])}")


def infer_split_label(log_glob: str, sources: list[str]) -> str:
    candidates = []
    haystacks = [log_glob] + sources
    for text in haystacks:
        t = text.lower()
        if "_eval_low_" in t or "low_l1" in t:
            candidates.append("LOW")
        if "_eval_medium_" in t or "medium_l1" in t:
            candidates.append("MEDIUM")
        if "_eval_low_l2" in t or "low_l2" in t:
            candidates.append("LOW_L2")
        if "_eval_medium_l2" in t or "medium_l2" in t:
            candidates.append("MEDIUM_L2")

    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "MIXED"
    return "TEST"


def main():
    parser = argparse.ArgumentParser(
        description="Print test summary: best harmonic, target run, and top-k by mIoU from Epoch(test) logs."
    )
    parser.add_argument(
        "--log-glob",
        default="/workspace/work_dirs/exp2A*_eval_medium_L1*/*/*.log",
        help="Glob pattern to find MMEngine test logs.",
    )
    parser.add_argument(
        "--target-substring",
        default="best_harmonic",
        help="Substring to select a specific run section (e.g., best_harmonic, epoch3000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top runs to print by mIoU.",
    )
    args = parser.parse_args()

    rows = []
    for log_path in sorted(glob.glob(args.log_glob)):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        checkpoint = extract_checkpoint_from_log_text(text)
        for line in text.splitlines():
            parsed = parse_line(line)
            if parsed is None:
                continue
            parsed["source"] = log_path
            parsed["checkpoint"] = checkpoint
            rows.append(parsed)

    if not rows:
        print(f"No test rows found for: {args.log_glob}")
        return

    split_label = infer_split_label(args.log_glob, [row["source"] for row in rows if row.get("source")])
    print_block(f"BEST HARMONIC TEST METRICS ({split_label})")
    harmonic_rows = [row for row in rows if row["harmonic"] is not None]
    if harmonic_rows:
        best_harmonic = max(harmonic_rows, key=lambda row: row["harmonic"])
        print_metrics(best_harmonic)
    else:
        print("No harmonic metrics found in logs")

    print()
    print_block(f"TARGET RUN METRICS ({args.target_substring})")
    target_rows = [
        row
        for row in rows
        if (args.target_substring in row["source"])
        or (row.get("checkpoint") is not None and args.target_substring in row["checkpoint"])
    ]
    if target_rows:
        best_target = max(
            target_rows,
            key=lambda row: row["miou"] if row["miou"] is not None else float("-inf"),
        )
        print_metrics(best_target)
    else:
        print(f"No run matched target substring: {args.target_substring}")

    print()
    print_block(f"TOP {args.top_k} TEST RUNS BY mIoU")
    sorted_rows = sorted(
        [row for row in rows if row["miou"] is not None],
        key=lambda row: row["miou"],
        reverse=True,
    )
    for idx, row in enumerate(sorted_rows[: args.top_k], start=1):
        source_short = row["source"].replace("/workspace/", "")
        ckpt_short = None
        if row.get("checkpoint"):
            ckpt_short = row["checkpoint"].replace("/workspace/", "")
        print(
            f"{idx}. {source_short}: "
            f"mIoU={fmt_percent(row['miou']).strip()} | "
            f"F1={fmt_percent(row['f1']).strip()} | "
            f"Harmonic={fmt_percent(row['harmonic']).strip()}"
        )
        if ckpt_short:
            print(f"   ├─ ckpt={ckpt_short}")
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
