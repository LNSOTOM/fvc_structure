#!/usr/bin/env python3
import argparse
import glob
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainRow:
    source: str
    epoch: int
    it: Optional[int]
    it_total: Optional[int]
    metrics: dict[str, float]


NUMBER_RE = r"[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?"


def parse_train_line(line: str) -> Optional[tuple[int, Optional[int], Optional[int], dict[str, float]]]:
    if "Epoch(train)" not in line:
        return None

    epoch_match = re.search(r"Epoch\(train\)\s*\[(\d+)\]", line)
    if not epoch_match:
        return None
    epoch = int(epoch_match.group(1))

    it = None
    it_total = None
    it_match = re.search(r"Epoch\(train\)\s*\[\d+\]\[(\d+)/(\d+)\]", line)
    if it_match:
        it = int(it_match.group(1))
        it_total = int(it_match.group(2))

    metrics: dict[str, float] = {}
    # Only capture *fully numeric* values. This avoids mis-parsing time-like values
    # such as: eta: 5:16:52 (we must not capture the leading "5").
    for key, raw in re.findall(
        rf"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*({NUMBER_RE})(?!:)(?=\s|$)",
        line,
    ):
        try:
            metrics[key] = float(raw)
        except ValueError:
            continue

    # Drop the epoch number if it was captured as a metric (shouldn't happen, but safe)
    metrics.pop("Epoch", None)
    return epoch, it, it_total, metrics


def print_block(title: str):
    print("╔════════════════════════════════════════════════════════════════════╗")
    print(f"║ {title:<66} ║")
    print("╚════════════════════════════════════════════════════════════════════╝")


def fmt_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print a compact per-epoch training summary from MMEngine logs (Epoch(train) lines). "
            "Collapses multiple iterations per epoch to the last seen line for that epoch."
        )
    )
    parser.add_argument(
        "--log-glob",
        default="/workspace/work_dirs/exp2A_epoch3000_train/*/*.log",
        help="Glob pattern to find MMEngine training logs.",
    )
    parser.add_argument(
        "--target-substring",
        default=None,
        help=(
            "Optional substring to select a specific run when multiple logs match. "
            "Matches against the log path."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3000,
        help="Epoch to print explicitly.",
    )
    parser.add_argument(
        "--metric",
        default="loss",
        help="Metric used for ranking best epochs (default: loss, lower is better).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many best epochs to show by the chosen metric.",
    )
    args = parser.parse_args()

    log_paths = sorted(glob.glob(args.log_glob))
    if args.target_substring:
        log_paths = [p for p in log_paths if args.target_substring in p]

    if not log_paths:
        print(f"No training logs found for: {args.log_glob}")
        if args.target_substring:
            print(f"(after filtering by target substring: {args.target_substring})")
        return

    # If multiple logs match, pick the newest-looking one by path sort (timestamped dirs)
    target_log = log_paths[-1]

    per_epoch_last: dict[int, TrainRow] = {}
    with open(target_log, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed = parse_train_line(line)
            if parsed is None:
                continue
            epoch, it, it_total, metrics = parsed
            per_epoch_last[epoch] = TrainRow(
                source=target_log,
                epoch=epoch,
                it=it,
                it_total=it_total,
                metrics=metrics,
            )

    if not per_epoch_last:
        print(f"No Epoch(train) rows found in: {target_log}")
        return

    epochs_sorted = sorted(per_epoch_last)
    last_epoch = epochs_sorted[-1]

    def metric_value(row: TrainRow, key: str) -> Optional[float]:
        return row.metrics.get(key)

    def print_row(label: str, row: TrainRow):
        print_block(label)
        print(f"Run:    {row.source}")
        it_str = ""
        if row.it is not None and row.it_total is not None:
            it_str = f"  iter={row.it}/{row.it_total}"
        print(f"Epoch:  {row.epoch}{it_str}")

        keys_to_show = [
            "lr",
            "loss",
            "discriminative_loss",
            "semantic_loss_bi",
            "grad_norm",
            "time",
            "data_time",
            "memory",
        ]
        shown = set()
        for key in keys_to_show:
            val = metric_value(row, key)
            if val is None:
                continue
            shown.add(key)
            print(f"  {key:<18} {fmt_value(val)}")

        # Print any remaining metrics (stable order)
        remaining = sorted(k for k in row.metrics.keys() if k not in shown)
        for key in remaining:
            print(f"  {key:<18} {fmt_value(row.metrics[key])}")

    # Epoch-specific
    if args.epoch in per_epoch_last:
        print_row(f"TRAIN METRICS (EPOCH {args.epoch})", per_epoch_last[args.epoch])
    else:
        print_block(f"TRAIN METRICS (EPOCH {args.epoch})")
        print(f"Run:    {target_log}")
        print(f"No Epoch(train) line found for epoch={args.epoch}")

    print()
    print_row("TRAIN METRICS (FINAL EPOCH)", per_epoch_last[last_epoch])

    print()
    print_block(f"TOP {args.top_k} EPOCHS BY {args.metric} (LOWEST)")
    rankable = []
    for ep in epochs_sorted:
        row = per_epoch_last[ep]
        val = metric_value(row, args.metric)
        if val is None:
            continue
        rankable.append((val, row))

    if not rankable:
        print(f"No rows had metric: {args.metric}")
        return

    rankable.sort(key=lambda x: x[0])
    for idx, (val, row) in enumerate(rankable[: args.top_k], start=1):
        parts = [f"{idx}. epoch={row.epoch:4d}", f"{args.metric}={fmt_value(val)}"]
        if args.metric != "loss":
            parts.append(f"loss={fmt_value(row.metrics.get('loss'))}")
        parts.append(f"lr={fmt_value(row.metrics.get('lr'))}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
