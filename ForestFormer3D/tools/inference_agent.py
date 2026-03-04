"""
Inference Agent: Iteratively improves test inference for label-mismatched data.

Problem: pretrained model uses FOREST3 labels (ground/wood/leaf),
         test data uses RAW4 labels (forb/hummock/shrub/mallee).

The agent tunes:
  - inst_score_thr  (instance confidence cutoff)
  - sp_score_thr    (superpoint seed quality)
  - npoint_thr      (minimum instance size)
  - pan_score_thr   (panoptic score threshold)

Usage (single GPU):
    python tools/inference_agent.py \
        --config configs/oneformer3d_qs_radius16_qp300_2many.py \
        --checkpoint work_dirs/pre_trained/trained_fix.pth \
        --output_dir work_dirs/agent_runs \
        --n_trials 30

Usage (multi-GPU distributed):
    python tools/inference_agent.py \
        --config configs/oneformer3d_qs_radius16_qp300_2many.py \
        --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
        --output_dir work_dirs/agent_runs \
        --n_trials 30 \
        --dist \
        --gpu_ids 0,1,2,3,4,6,7

Usage (inside Docker container, multi-GPU):
    python tools/inference_agent.py \
        --config configs/oneformer3d_qs_radius16_qp300_2many.py \
        --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
        --output_dir work_dirs/agent_runs \
        --n_trials 30 \
        --dist \
        --gpu_ids 0,1,2,3,4,6,7 \
        --docker \
        --container forestformer3d-container \
        --container_workdir /workspace
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Parameter space
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceParams:
    inst_score_thr: float = 0.1
    sp_score_thr: float = 0.25
    npoint_thr: int = 50
    pan_score_thr: float = 0.0
    topk_insts: int = 150
    obj_normalization_thr: float = 0.01

    def to_cfg_overrides(self) -> list[str]:
        """Convert to mmdet3d cfg-options format."""
        return [
            f"model.test_cfg.inst_score_thr={self.inst_score_thr}",
            f"model.test_cfg.sp_score_thr={self.sp_score_thr}",
            f"model.test_cfg.npoint_thr={self.npoint_thr}",
            f"model.test_cfg.pan_score_thr={self.pan_score_thr}",
            f"model.test_cfg.topk_insts={self.topk_insts}",
            f"model.test_cfg.obj_normalization_thr={self.obj_normalization_thr}",
        ]


def run_subprocess(cmd: list[str], env: dict, cwd, label: str = "") -> tuple[str, int]:
    """
    Run a subprocess and stream its stdout/stderr live to the terminal
    while also capturing the full output for metric parsing.
    Returns (combined_log, returncode).
    """
    if label:
        print(f"[Agent] {label}")

    process = subprocess.Popen(
        cmd, env=env, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )

    log_lines: list[str] = []
    lock = threading.Lock()

    def stream(pipe, prefix: str):
        for line in pipe:
            with lock:
                log_lines.append(line)
            print(f"{prefix}{line}", end="", flush=True)

    t_out = threading.Thread(target=stream, args=(process.stdout, ""), daemon=True)
    t_err = threading.Thread(target=stream, args=(process.stderr, ""), daemon=True)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    process.wait()

    return "".join(log_lines), process.returncode




def docker_wrap(cmd: list[str], container: str, workdir: str,
                env_vars: dict | None = None) -> list[str]:
    """
    Wrap a command list for execution inside a running Docker container:

        sudo docker exec [-e KEY=VAL ...] -w <workdir> <container> <cmd...>

    NOTE: -i (not -t) is used so stdout/stderr are captured correctly
          by subprocess.run(capture_output=True).
    """
    prefix = ["sudo", "docker", "exec", "-i"]
    if env_vars:
        for key, val in env_vars.items():
            prefix += ["-e", f"{key}={val}"]
    prefix += ["-w", workdir, container]
    return prefix + cmd


# ─────────────────────────────────────────────────────────────────────────────
# Metric parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_metrics_from_log(log_text: str) -> dict:
    """Extract key metrics from mmdet3d test output."""
    metrics = {}
    for line in log_text.splitlines():
        # Panoptic Quality
        if "PQ" in line and "AP" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p in ("PQ", "AP", "AR") and i + 1 < len(parts):
                    try:
                        metrics[p] = float(parts[i + 1])
                    except ValueError:
                        pass
        # mAP style output
        if "mAP" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if "mAP" in p and i + 1 < len(parts):
                    try:
                        metrics["mAP"] = float(parts[i + 1].strip(","))
                    except ValueError:
                        pass
    return metrics


def score_run(metrics: dict) -> float:
    """
    Composite score that balances precision and recall.
    Prioritise PQ (panoptic quality) then mAP, then AR.
    """
    pq = metrics.get("PQ", 0.0)
    ap = metrics.get("AP", metrics.get("mAP", 0.0))
    ar = metrics.get("AR", 0.0)
    # Harmonic-mean style: reward balanced AP/AR, weight PQ heavily
    if ap + ar > 0:
        f1 = 2 * ap * ar / (ap + ar)
    else:
        f1 = 0.0
    return 0.6 * pq + 0.4 * f1


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(config: str, checkpoint: str, params: InferenceParams,
             work_dir: str, gpu_id: int = 0,
             docker: bool = False, container: str = "forestformer3d-container",
             container_workdir: str = "/workspace") -> tuple[dict, str]:
    """Run mmdet3d single-GPU test and return parsed metrics + raw log."""
    inner_cmd = [
        sys.executable,
        "tools/test.py",
        config,
        checkpoint,
        "--cfg-options", *params.to_cfg_overrides(),
        "--work-dir", work_dir,
    ]

    if docker:
        cmd = docker_wrap(inner_cmd, container, container_workdir,
                          env_vars={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
        cwd = None  # docker exec handles the workdir
        env = os.environ.copy()
        print(f"\n[Agent] Running in container '{container}':")
        print(f"  CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(inner_cmd)}")
    else:
        cmd = inner_cmd
        cwd = str(Path(__file__).parent.parent)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"\n[Agent] Running: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")

    log, returncode = run_subprocess(cmd, env, cwd)
    if returncode != 0:
        print(f"[Agent] WARNING: test exited with code {returncode}")
    metrics = parse_metrics_from_log(log)
    return metrics, log


def run_test_dist(config: str, checkpoint: str, params: InferenceParams,
                  work_dir: str, gpu_ids: str = "0,1,2,3,4,6,7",
                  docker: bool = False, container: str = "forestformer3d-container",
                  container_workdir: str = "/workspace") -> tuple[dict, str]:
    """
    Run distributed mmdet3d test via dist_test.sh.

    Equivalent to:
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 \
        bash tools/dist_test.sh \
            <config> <checkpoint> <num_gpus> \
            --cfg-options ...

    With --docker, runs inside the container:
        sudo docker exec -e CUDA_VISIBLE_DEVICES=... -w /workspace \
            forestformer3d-container \
            bash tools/dist_test.sh ...
    """
    num_gpus = len(gpu_ids.split(","))

    inner_cmd = [
        "bash",
        "tools/dist_test.sh",
        config,
        checkpoint,
        str(num_gpus),
        "--cfg-options", *params.to_cfg_overrides(),
        "--work-dir", work_dir,
    ]

    print(f"\n[Agent] Running (dist, {num_gpus} GPUs):")
    print(f"  CUDA_VISIBLE_DEVICES={gpu_ids} \\")
    print(f"  bash tools/dist_test.sh \\")
    print(f"    {config} \\")
    print(f"    {checkpoint} \\")
    print(f"    {num_gpus}")

    if docker:
        cmd = docker_wrap(inner_cmd, container, container_workdir,
                          env_vars={"CUDA_VISIBLE_DEVICES": gpu_ids})
        cwd = None
        env = os.environ.copy()
        print(f"  [in container: {container}]")
    else:
        cmd = inner_cmd
        cwd = str(Path(__file__).parent.parent)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    log, returncode = run_subprocess(cmd, env, cwd)
    if returncode != 0:
        print(f"[Agent] WARNING: dist test exited with code {returncode}")
    metrics = parse_metrics_from_log(log)
    return metrics, log


# ─────────────────────────────────────────────────────────────────────────────
# Search strategy: simple random + greedy hill-climb
# ─────────────────────────────────────────────────────────────────────────────

import random
import copy


def neighbour(params: InferenceParams, strength: float = 0.15) -> InferenceParams:
    """Return a randomly perturbed neighbour of params."""
    p = copy.copy(params)
    choice = random.choice([
        "inst_score_thr", "sp_score_thr", "npoint_thr", "pan_score_thr"
    ])
    if choice == "inst_score_thr":
        p.inst_score_thr = float(
            max(0.01, min(0.5, p.inst_score_thr + random.gauss(0, strength * 0.2))))
    elif choice == "sp_score_thr":
        p.sp_score_thr = float(
            max(0.05, min(0.6, p.sp_score_thr + random.gauss(0, strength * 0.15))))
    elif choice == "npoint_thr":
        p.npoint_thr = int(
            max(10, min(200, p.npoint_thr + int(random.gauss(0, strength * 30)))))
    elif choice == "pan_score_thr":
        p.pan_score_thr = float(
            max(0.0, min(0.3, p.pan_score_thr + random.gauss(0, strength * 0.05))))
    return p


def run_agent(config: str, checkpoint: str, output_dir: str,
              n_trials: int = 20, gpu_id: int = 0,
              dist: bool = False, gpu_ids: str = "0,1,2,3,4,6,7",
              docker: bool = False, container: str = "forestformer3d-container",
              container_workdir: str = "/workspace",
              seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "agent_results.jsonl"

    best_params = InferenceParams()   # start from config defaults
    best_score = -1.0
    history = []

    for trial in range(n_trials):
        trial_dir = str(output_path / f"trial_{trial:03d}")

        # Explore: perturb best params (simulated annealing-lite)
        temperature = max(0.05, 1.0 - trial / n_trials)
        if trial == 0:
            params = best_params          # first run: use defaults
        else:
            params = neighbour(best_params, strength=temperature)

        print(f"\n{'='*60}")
        print(f"[Agent] Trial {trial+1}/{n_trials}  (temp={temperature:.2f})")
        print(f"[Agent] Params: {asdict(params)}")

        if dist:
            metrics, log = run_test_dist(
                config, checkpoint, params, trial_dir, gpu_ids,
                docker=docker, container=container, container_workdir=container_workdir)
        else:
            metrics, log = run_test(
                config, checkpoint, params, trial_dir, gpu_id,
                docker=docker, container=container, container_workdir=container_workdir)
        score = score_run(metrics)

        record = {
            "trial": trial,
            "params": asdict(params),
            "metrics": metrics,
            "score": score,
        }
        history.append(record)

        with open(results_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[Agent] Metrics: {metrics}  Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            print(f"[Agent] *** New best! score={best_score:.4f} ***")
            # Save best params
            with open(output_path / "best_params.json", "w") as f:
                json.dump({"score": best_score, "params": asdict(best_params)}, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Agent] Search complete.  Best score: {best_score:.4f}")
    print(f"[Agent] Best params:\n{json.dumps(asdict(best_params), indent=2)}")
    print(f"[Agent] Full results saved to: {results_file}")

    # Emit a ready-to-paste cfg snippet
    print("\n[Agent] Paste these into your config's test_cfg:")
    print(f"    inst_score_thr={best_params.inst_score_thr},")
    print(f"    sp_score_thr={best_params.sp_score_thr},")
    print(f"    npoint_thr={best_params.npoint_thr},")
    print(f"    pan_score_thr={best_params.pan_score_thr},")
    return best_params, history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference agent for label-mismatched test data")
    parser.add_argument("--config", required=True,
                        help="Path to mmdet3d config file")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--output_dir", default="work_dirs/agent_runs",
                        help="Directory to store trial results")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of search trials")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index for single-GPU mode")
    parser.add_argument("--dist", action="store_true",
                        help="Use distributed testing via dist_test.sh")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,6,7",
                        help="Comma-separated GPU IDs for distributed mode (e.g. 0,1,2,3,4,6,7)")
    parser.add_argument("--docker", action="store_true",
                        help="Run commands inside a Docker container via 'sudo docker exec'")
    parser.add_argument("--container", type=str, default="forestformer3d-container",
                        help="Docker container name (default: forestformer3d-container)")
    parser.add_argument("--container_workdir", type=str, default="/workspace",
                        help="Working directory inside the container (default: /workspace)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    run_agent(
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        gpu_id=args.gpu_id,
        dist=args.dist,
        gpu_ids=args.gpu_ids,
        docker=args.docker,
        container=args.container,
        container_workdir=args.container_workdir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
