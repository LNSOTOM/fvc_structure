import argparse
import os
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_dataroot_from_overrides(checkpoint_dir):
    overrides_path = os.path.join(checkpoint_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(overrides_path):
        return None
    data = load_yaml(overrides_path)
    for item in data:
        if isinstance(item, str) and item.startswith("data.dataroot="):
            return item.split("=", 1)[1].strip()
    return None


def get_dataroot_from_eval(eval_cfg, eval_yaml_path):
    # 1) Try fold paths from current eval.yaml (best source for inference run)
    folds = eval_cfg.get("data", {}).get("fold", [])
    if folds and isinstance(folds, list):
        first = folds[0]
        if isinstance(first, str) and first:
            # fold entries are .../utm2local/<scene>_out.ply
            return os.path.dirname(first)

    # 2) Fallback: DEST_DIR/utm2local, derived from eval.yaml location
    dest_dir = os.path.dirname(os.path.abspath(eval_yaml_path))
    return os.path.join(dest_dir, "utm2local")


def main():
    parser = argparse.ArgumentParser(description="Clear cached processed data")
    parser.add_argument("--eval_yaml", type=str, required=True, help="Path to eval.yaml")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.eval_yaml)

    # Prefer current eval run dataroot
    dataroot = get_dataroot_from_eval(eval_cfg, args.eval_yaml)

    # Optional fallback to checkpoint overrides if current one is missing
    if not os.path.exists(dataroot):
        ckpt_dir = eval_cfg.get("checkpoint_dir", "")
        fallback = get_dataroot_from_overrides(ckpt_dir) if ckpt_dir else None
        if fallback:
            dataroot = fallback

    # historical cache location expected by pipeline
    cache_path = os.path.join(dataroot, "treeinsfused", "processed_0.2_test")
    print(f"Clearing: {cache_path}")

    if os.path.exists(cache_path):
        for filename in os.listdir(cache_path):
            filepath = os.path.join(cache_path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Removed: {filepath}")
    else:
        print(f"Path does not exist: {cache_path}")


if __name__ == "__main__":
    main()
