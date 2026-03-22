import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_float(pattern: str, text: str):
    # Many logs contain intermediate tqdm progress lines like `test_mIPre=0.0`.
    # We want the final metric summary at the end of the log, so take the LAST match.
    matches = re.findall(pattern, text, re.MULTILINE)
    if not matches:
        return None

    value = matches[-1]
    # If findall returns tuples (multiple groups), take the first group.
    if isinstance(value, tuple):
        value = value[0]

    if isinstance(value, str) and value.lower() == "nan":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_segmentanytree_metrics(log_file: str):
    """
    Parse SegmentAnyTree evaluation logs.

    Keeps only instance-style metrics by default because SegmentAnyTree
    does not provide semantic segmentation outputs in the same sense as
    ForestFormer3D / OneFormer3D tables.

    Returns a dict with values in [0, 1] where available.
    """
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    metrics = {
        # Instance / detection-style metrics
        "precision": None,
        "recall": None,
        "f1": None,
        "cov": None,
        "map": None,
        "ap_tree": None,
        # Optional extras if present in logs
        "instance_acc": None,
    }

    # Flexible patterns to tolerate slightly different logger layouts
    patterns = {
        "precision": [
            r"test_\s*\n\s*mIPre\s*=\s*([0-9nan.]+)",
            r"mIPre\s*=\s*([0-9nan.]+)",
            r"test_\s*\nmIPre\s*=\s*([0-9nan.]+)",
            r"test_mIPre\s*=\s*([0-9nan.]+)",
            r"Precision\s*[:=]\s*([0-9nan.]+)",
        ],
        "recall": [
            r"test_\s*\n\s*mIRec\s*=\s*([0-9nan.]+)",
            r"mIRec\s*=\s*([0-9nan.]+)",
            r"test_\s*\nmIRec\s*=\s*([0-9nan.]+)",
            r"test_mIRec\s*=\s*([0-9nan.]+)",
            r"Recall\s*[:=]\s*([0-9nan.]+)",
        ],
        "f1": [
            r"test_\s*\n\s*F1\s*=\s*([0-9nan.]+)",
            r"F1\s*=\s*([0-9nan.]+)",
            r"test_\s*\nF1\s*=\s*([0-9nan.]+)",
            r"test_F1\s*=\s*([0-9nan.]+)",
        ],
        "cov": [
            r"test_\s*\n\s*cov\s*=\s*([0-9nan.]+)",
            r"cov\s*=\s*([0-9nan.]+)",
            r"test_\s*\ncov\s*=\s*([0-9nan.]+)",
            r"test_cov\s*=\s*([0-9nan.]+)",
            r"Coverage\s*[:=]\s*([0-9nan.]+)",
        ],
        "map": [
            r"test_\s*\n\s*map\s*=\s*([0-9nan.]+)",
            r"map\s*=\s*([0-9nan.]+)",
            r"test_\s*\nmap\s*=\s*([0-9nan.]+)",
            r"test_map\s*=\s*([0-9nan.]+)",
            r"mAP\s*[:=]\s*([0-9nan.]+)",
        ],
        "ap_tree": [
            r"class_ap\s*=\s*\{1:\s*([0-9nan.]+),?\}",
            r"test_class_ap\s*=\s*\{1:\s*([0-9nan.]+),?\}",
            r"AP.*class.*1.*?([0-9nan.]+)",
        ],
        "instance_acc": [
            r"test_\s*\n\s*Iacc\s*=\s*([0-9nan.]+)",
            r"Iacc\s*=\s*([0-9nan.]+)",
            r"test_\s*\nIacc\s*=\s*([0-9nan.]+)",
            r"test_Iacc\s*=\s*([0-9nan.]+)",
        ],
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            value = extract_float(pattern, content)
            if value is not None:
                metrics[key] = value
                break

    return metrics


def safe_mean(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def plot_radar_chart(metrics, labels, title, filename):
    """
    Plot a radar chart for the given metrics.
    """
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    values = [metrics.get(label.lower().replace(' ', '_'), 0) for label in labels]
    values += values[:1]  # Close the loop

    ax.fill(angles, values, 'b', alpha=0.1)
    ax.plot(angles, values, 'o-', linewidth=2, label='Average Metrics')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    ax.grid(True)

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def fmt_pct(value):
    return f"{value * 100:.2f}" if value is not None else "--"


def fmt_raw(value):
    return f"{value:.4f}" if value is not None else "--"


# ---------------------------------------------------------------------
# Collect metrics across GPU logs
# ---------------------------------------------------------------------
run_dir = Path("/data/calperumResearch/segmentanytree/output/ForAINetV2/runs/20260310_052457")

gpu_metrics = {}
for gpu in range(1, 8):
    log_file = run_dir / f"gpu{gpu}.log"
    if log_file.exists():
        gpu_metrics[f"GPU {gpu}"] = parse_segmentanytree_metrics(str(log_file))
    else:
        gpu_metrics[f"GPU {gpu}"] = {}

# ---------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------
print("SegmentAnyTree Evaluation Metrics Summary (instance metrics only)")
print("=" * 110)
print(
    f"{'GPU':<8}"
    f"{'Prec (%)':<12}"
    f"{'Rec (%)':<12}"
    f"{'F1 (%)':<12}"
    f"{'Cov (%)':<12}"
    f"{'mAP':<12}"
    f"{'AP Tree':<12}"
    f"{'IAcc':<12}"
)
print("-" * 110)

for gpu, metrics in gpu_metrics.items():
    print(
        f"{gpu:<8}"
        f"{fmt_pct(metrics.get('precision')):<12}"
        f"{fmt_pct(metrics.get('recall')):<12}"
        f"{fmt_pct(metrics.get('f1')):<12}"
        f"{fmt_pct(metrics.get('cov')):<12}"
        f"{fmt_raw(metrics.get('map')):<12}"
        f"{fmt_raw(metrics.get('ap_tree')):<12}"
        f"{fmt_raw(metrics.get('instance_acc')):<12}"
    )

avg_precision = safe_mean([m.get("precision") for m in gpu_metrics.values()])
avg_recall = safe_mean([m.get("recall") for m in gpu_metrics.values()])
avg_f1 = safe_mean([m.get("f1") for m in gpu_metrics.values()])
avg_cov = safe_mean([m.get("cov") for m in gpu_metrics.values()])
avg_map = safe_mean([m.get("map") for m in gpu_metrics.values()])
avg_ap_tree = safe_mean([m.get("ap_tree") for m in gpu_metrics.values()])
avg_iacc = safe_mean([m.get("instance_acc") for m in gpu_metrics.values()])

print("-" * 110)
print(
    f"{'Avg':<8}"
    f"{fmt_pct(avg_precision):<12}"
    f"{fmt_pct(avg_recall):<12}"
    f"{fmt_pct(avg_f1):<12}"
    f"{fmt_pct(avg_cov):<12}"
    f"{fmt_raw(avg_map):<12}"
    f"{fmt_raw(avg_ap_tree):<12}"
    f"{fmt_raw(avg_iacc):<12}"
)

# ---------------------------------------------------------------------
# Plot instance metrics only
# ---------------------------------------------------------------------
gpus = list(gpu_metrics.keys())
precision_vals = [gpu_metrics[g].get("precision") if gpu_metrics[g].get("precision") is not None else np.nan for g in gpus]
recall_vals = [gpu_metrics[g].get("recall") if gpu_metrics[g].get("recall") is not None else np.nan for g in gpus]
f1_vals = [gpu_metrics[g].get("f1") if gpu_metrics[g].get("f1") is not None else np.nan for g in gpus]
cov_vals = [gpu_metrics[g].get("cov") if gpu_metrics[g].get("cov") is not None else np.nan for g in gpus]
ap_tree_vals = [gpu_metrics[g].get("ap_tree") if gpu_metrics[g].get("ap_tree") is not None else np.nan for g in gpus]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].bar(gpus, precision_vals)
axs[0].set_title("Precision per GPU")
axs[0].set_ylabel("Score")
axs[0].set_xlabel("GPU")
axs[0].set_ylim(0, 1)
axs[0].tick_params(axis="x", rotation=45)

axs[1].bar(gpus, recall_vals)
axs[1].set_title("Recall per GPU")
axs[1].set_ylabel("Score")
axs[1].set_xlabel("GPU")
axs[1].set_ylim(0, 1)
axs[1].tick_params(axis="x", rotation=45)

axs[2].bar(gpus, f1_vals)
axs[2].set_title("F1 per GPU")
axs[2].set_ylabel("Score")
axs[2].set_xlabel("GPU")
axs[2].set_ylim(0, 1)
axs[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
out_plot_1 = "/home/nvidia/projects/fvc_structure/segmentanytree_instance_metrics_main.png"
plt.savefig(out_plot_1, dpi=200)
print(f"\nSaved: {out_plot_1}")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].bar(gpus, cov_vals)
axs[0].set_title("Coverage per GPU")
axs[0].set_ylabel("Score")
axs[0].set_xlabel("GPU")
axs[0].set_ylim(0, 1)
axs[0].tick_params(axis="x", rotation=45)

axs[1].bar(gpus, ap_tree_vals)
axs[1].set_title("AP Tree per GPU")
axs[1].set_ylabel("Score")
axs[1].set_xlabel("GPU")
axs[1].set_ylim(0, 1)
axs[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
out_plot_2 = "/home/nvidia/projects/fvc_structure/segmentanytree_instance_metrics_aux.png"
plt.savefig(out_plot_2, dpi=200)
print(f"Saved: {out_plot_2}")

# ---------------------------------------------------------------------
# Radar chart for average metrics
# ---------------------------------------------------------------------
avg_metrics = {
    "precision": avg_precision,
    "recall": avg_recall,
    "f1": avg_f1,
    "coverage": avg_cov,
    "ap_tree": avg_ap_tree,
}
radar_labels = ["Precision", "Recall", "F1", "Coverage", "AP Tree"]
if all(avg_metrics[k] is not None for k in ["precision", "recall", "f1", "coverage", "ap_tree"]):
    plot_radar_chart(
        avg_metrics,
        radar_labels,
        "Average Instance Metrics Radar Chart",
        "/home/nvidia/projects/fvc_structure/segmentanytree_radar_chart.png",
    )
    print("Saved: /home/nvidia/projects/fvc_structure/segmentanytree_radar_chart.png")
else:
    print("Skipped radar chart: some average metrics are undefined (nan).")

print(
    "\nNote: SegmentAnyTree is treated here as an instance-segmentation method. "
    "Semantic metrics such as Ground/Wood/Leaf IoU and mIoU are intentionally omitted."
)