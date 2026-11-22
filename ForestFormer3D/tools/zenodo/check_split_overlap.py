#!/usr/bin/env python3
"""Check that no scene is shared between training/validation and held-out test lists.

Reads every ``train_list_*.txt``, ``val_list_*.txt`` and ``test_list_*.txt`` in a
ForAINetV2 ``meta_data/`` directory and reports:

* ERROR  a scene in a train or val list that is also in a strict test list;
* ERROR  a scene in both the train and the val list of the same experiment;
* ERROR  a scene listed twice in the same file;
* INFO   overlaps with the L1 *pool* lists (``test_list_medium_L1.txt`` and
  ``test_list_low_L1.txt``). These hold every L1 tile, and the Experiment 2/3
  train, val and hold-out lists are drawn from them, so overlap is expected.

Strict test lists are all ``test_list_*.txt`` files except the pool lists and
lists that are empty (``test_list_exp2_mediumL1_lowL1.txt`` is deliberately
empty; see the experiment runbook).

Exit status is 1 if any ERROR is found, so the script can gate packaging.

Example::

    python tools/zenodo/check_split_overlap.py \
        --meta-data /workspace/data/ForAINetV2/meta_data
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

POOL_LISTS = {"test_list_medium_L1.txt", "test_list_low_L1.txt"}


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def suffix(name: str, prefix: str) -> str:
    return name[len(prefix):-len(".txt")]


def check(meta_data: Path, verbose: bool = True) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Return (errors, infos, lists) for the list files in ``meta_data``."""
    if not meta_data.is_dir():
        raise FileNotFoundError(f"meta_data directory not found: {meta_data}")

    lists = {p.name: read_ids(p) for p in sorted(meta_data.glob("*_list*.txt"))
             if p.name.startswith(("train_list", "val_list", "test_list"))}
    if not lists:
        raise FileNotFoundError(f"No train/val/test list files in {meta_data}")

    errors: list[str] = []
    infos: list[str] = []

    for name, ids in lists.items():
        dupes = sorted(k for k, n in Counter(ids).items() if n > 1)
        if dupes:
            errors.append(f"{name}: {len(dupes)} duplicated scene(s): {', '.join(dupes[:10])}")

    fit = {n: set(v) for n, v in lists.items() if n.startswith(("train_list", "val_list"))}
    strict = {n: set(v) for n, v in lists.items()
              if n.startswith("test_list") and n not in POOL_LISTS and v}
    pools = {n: set(v) for n, v in lists.items() if n in POOL_LISTS}

    for fname, fids in fit.items():
        for tname, tids in strict.items():
            shared = sorted(fids & tids)
            if shared:
                errors.append(f"{fname} ∩ {tname}: {len(shared)} scene(s): {', '.join(shared[:10])}")
        for pname, pids in pools.items():
            n = len(fids & pids)
            if n:
                infos.append(f"{fname} draws {n}/{len(fids)} scene(s) from pool {pname}")

    for tname, tids in fit.items():
        if not tname.startswith("train_list"):
            continue
        vname = "val_list" + tname[len("train_list"):]
        if vname in fit:
            shared = sorted(fit[tname] & fit[vname])
            if shared:
                errors.append(f"{tname} ∩ {vname}: {len(shared)} scene(s): {', '.join(shared[:10])}")
        else:
            infos.append(f"{tname} has no matching {vname}")

    if verbose:
        print(f"Checked {len(lists)} list file(s) in {meta_data}")
        for name, ids in lists.items():
            kind = "pool" if name in POOL_LISTS else ("empty" if not ids else "")
            print(f"  {len(ids):5d}  {name}" + (f"  ({kind})" if kind else ""))
        print(f"\nStrict test lists: {', '.join(sorted(strict)) or 'none'}")
        for line in infos:
            print(f"INFO   {line}")
        for line in errors:
            print(f"ERROR  {line}")
        print("\nOK: no train/val scene appears in a held-out test list." if not errors
              else f"\nFAILED: {len(errors)} problem(s) found.")
    return errors, infos, lists


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta-data", type=Path, required=True, help="ForAINetV2 meta_data/ directory")
    args = p.parse_args()
    errors, _, _ = check(args.meta_data)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
