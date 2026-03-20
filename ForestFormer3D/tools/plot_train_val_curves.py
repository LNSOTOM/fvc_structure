#!/usr/bin/env python3
import argparse
import glob
import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


NUMBER_RE = r"[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class CurvePoint:
    epoch: int
    metrics: dict[str, float]


def extract_epoch(line: str, phase: str) -> Optional[int]:
    match = re.search(rf"Epoch\({re.escape(phase)}\)\s*\[(\d+)\]", line)
    if not match:
        return None
    return int(match.group(1))


def extract_kv_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    # Capture only fully numeric values; avoid time-like values like eta: 5:16:52
    for key, raw in re.findall(
        rf"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*({NUMBER_RE})(?!:)(?=\s|$)",
        line,
    ):
        try:
            metrics[key] = float(raw)
        except ValueError:
            continue

    return metrics


def to_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100 if value <= 1 else value


def read_curves(log_path: str) -> tuple[dict[int, CurvePoint], dict[int, CurvePoint]]:
    train_by_epoch: dict[int, CurvePoint] = {}
    val_by_epoch: dict[int, CurvePoint] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Epoch(train)" in line:
                epoch = extract_epoch(line, "train")
                if epoch is None:
                    continue
                metrics = extract_kv_metrics(line)
                train_by_epoch[epoch] = CurvePoint(epoch=epoch, metrics=metrics)
            elif "Epoch(val)" in line:
                epoch = extract_epoch(line, "val")
                if epoch is None:
                    continue
                metrics = extract_kv_metrics(line)
                # mIoU often appears on val lines, but sometimes line breaks occur; still fine.
                val_by_epoch[epoch] = CurvePoint(epoch=epoch, metrics=metrics)

    return train_by_epoch, val_by_epoch


def newest_log_from_glob(log_glob: str) -> Optional[str]:
    paths = sorted(glob.glob(log_glob))
    if not paths:
        return None
    return paths[-1]


def ensure_writable_out_dir(requested: Path, log_path: str) -> Path:
    requested = Path(requested)
    try:
        requested.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        requested = None  # handled below

    if requested is not None:
        # Directory may exist but still be unwritable (e.g., root-owned work_dirs on host).
        if os.access(str(requested), os.W_OK):
            return requested

    fallback = Path.cwd() / f"plots_{Path(log_path).stem}"
    fallback.mkdir(parents=True, exist_ok=True)
    if requested is not None:
        print(f"Output dir not writable: {requested}")
    else:
        print("Permission denied creating requested output dir")
    print(f"Falling back to: {fallback}")
    return fallback


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves from a single MMEngine log. "
            "Always plots train loss (from Epoch(train) lines). "
            "Plots val loss only if a loss-like field exists in Epoch(val) lines; "
            "also plots val mIoU/F1/harmonic when present."
        )
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path to a single MMEngine .log file.",
    )
    parser.add_argument(
        "--log-glob",
        default="/workspace/work_dirs/exp2A_epoch3000_train/*/*.log",
        help="Glob to select a training log; newest match is used.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write PNG outputs.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title prefix.",
    )
    args = parser.parse_args()

    log_path = args.log_path or newest_log_from_glob(args.log_glob)
    if not log_path:
        print(f"No logs found for: {args.log_glob}")
        return

    out_dir = ensure_writable_out_dir(Path(args.out_dir), log_path)

    train_by_epoch, val_by_epoch = read_curves(log_path)
    if not train_by_epoch and not val_by_epoch:
        print(f"No Epoch(train)/Epoch(val) lines found in: {log_path}")
        return

    # Lazy import so the script can still be used for parsing-only if matplotlib is missing.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib is required for plotting: {exc}")
        return

    title_prefix = args.title or Path(log_path).name

    # ----------------------------
    # Train loss curve
    # ----------------------------
    if train_by_epoch:
        epochs = sorted(train_by_epoch)
        train_loss = [train_by_epoch[e].metrics.get("loss") for e in epochs]

        plt.figure(figsize=(10, 4))
        plt.plot(epochs, train_loss, linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.title(f"{title_prefix} — Train loss")
        plt.grid(True, alpha=0.3)
        out_path = out_dir / "train_loss.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")
    else:
        print("No train curve found (no Epoch(train) lines).")

    # ----------------------------
    # Val metrics curve(s)
    # ----------------------------
    if val_by_epoch:
        val_epochs = sorted(val_by_epoch)

        def series(key: str, percent: bool = True):
            vals = [val_by_epoch[e].metrics.get(key) for e in val_epochs]
            if percent:
                return [to_percent(v) for v in vals]
            return vals

        val_miou = series("mIoU", percent=True)
        val_f1 = series("F1", percent=True)
        val_harm = series("harmonic", percent=True)

        # Validation loss (if present)
        # Prefer explicit val_loss keys if used; fallback to plain loss.
        val_loss = [
            val_by_epoch[e].metrics.get("val_loss")
            if val_by_epoch[e].metrics.get("val_loss") is not None
            else val_by_epoch[e].metrics.get("loss")
            for e in val_epochs
        ]
        has_val_loss = any(v is not None for v in val_loss)

        # Plot val metrics that exist
        any_metric = any(v is not None for v in val_miou + val_f1 + val_harm)
        if any_metric:
            plt.figure(figsize=(10, 4))
            if any(v is not None for v in val_miou):
                plt.plot(val_epochs, val_miou, label="mIoU", linewidth=1.5)
            if any(v is not None for v in val_f1):
                plt.plot(val_epochs, val_f1, label="F1", linewidth=1.5)
            if any(v is not None for v in val_harm):
                plt.plot(val_epochs, val_harm, label="Harmonic", linewidth=1.5)
            plt.xlabel("Epoch")
            plt.ylabel("Value (%)")
            plt.title(f"{title_prefix} — Validation metrics")
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_path = out_dir / "val_metrics.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved: {out_path}")
        else:
            print("No recognizable val metrics found (expected mIoU/F1/harmonic keys).")

        if has_val_loss:
            plt.figure(figsize=(10, 4))
            plt.plot(val_epochs, val_loss, linewidth=1.5)
            plt.xlabel("Epoch")
            plt.ylabel("Val loss")
            plt.title(f"{title_prefix} — Val loss")
            plt.grid(True, alpha=0.3)
            out_path = out_dir / "val_loss.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved: {out_path}")
        else:
            print("Val loss not found in Epoch(val) lines; no val_loss.png generated.")

    else:
        print("No val curve found (no Epoch(val) lines).")


if __name__ == "__main__":
    main()
