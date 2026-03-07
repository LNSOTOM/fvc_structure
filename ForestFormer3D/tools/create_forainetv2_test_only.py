#!/usr/bin/env python
"""Create a *test-only* info pkl for ForAINetV2-style datasets.

This is intended for extra experiments (e.g., rangeland/CALP zero-shot)
without modifying the original train/val/test PKLs.

It assumes you already prepared `forainetv2_instance_data/*_vert.npy`,
`*_sem_label.npy`, `*_ins_label.npy`, and bbox/matrix npys under `root_path`.

Example (inside container):
  python tools/create_forainetv2_test_only.py \
    --root-path data/ForAINetV2_calp \
    --out-dir data/ForAINetV2_calp \
    --test-list data/ForAINetV2/meta_data/test_list_calp.txt \
    --extra-tag forainetv2 \
    --workers 4

This will write:
  <out-dir>/<extra-tag>_oneformer3d_infos_test.pkl
and then convert it to V2 format via update_infos_to_v2.
"""

import argparse
from os import path as osp

import mmengine

from forainetv2_data_utils import ForAINetV2Data
from update_infos_to_v2 import update_pkl_infos


def parse_args():
  parser = argparse.ArgumentParser(description='Create ForAINetV2 test-only info pkl')
  parser.add_argument(
    '--root-path',
    type=str,
    required=True,
    help='Dataset root containing forainetv2_instance_data/')
  parser.add_argument(
    '--out-dir',
    type=str,
    required=True,
    help='Where to write the generated pkl')
  parser.add_argument(
    '--extra-tag',
    type=str,
    default='forainetv2',
    help='Filename prefix for the output pkl')
  parser.add_argument(
    '--test-list',
    type=str,
    required=True,
    help='Path to a text file listing scene ids (one per line)')
  parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Overwrite the output pkl if it already exists (default: refuse to overwrite).')
  parser.add_argument('--workers', type=int, default=4)
  return parser.parse_args()


def main():
    args = parse_args()

    mmengine.mkdir_or_exist(args.out_dir)

    # Load explicit scene ids
    mmengine.check_file_exist(args.test_list)
    sample_id_list = [s.strip() for s in mmengine.list_from_file(args.test_list) if s.strip()]

    # Build dataset helper (needs meta_data/test_list.txt to exist, but we override sample_id_list)
    dataset = ForAINetV2Data(root_path=args.root_path, split='test')

    infos_test = dataset.get_infos(num_workers=args.workers, has_label=True, sample_id_list=sample_id_list)

    test_filename = osp.join(args.out_dir, f'{args.extra_tag}_oneformer3d_infos_test.pkl')

    if osp.exists(test_filename) and not args.overwrite:
        raise FileExistsError(
            f'Output already exists: {test_filename}\n'
            'Refusing to overwrite. Pass --overwrite to replace it, or choose a different --extra-tag/--out-dir.'
        )
    mmengine.dump(infos_test, test_filename, 'pkl')
    print(f'Test-only info file saved to {test_filename}')

    # Convert to v2 format (dataset type remains forainetv2)
    update_pkl_infos('forainetv2', out_dir=args.out_dir, pkl_path=test_filename)


if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules

    register_all_modules()
    main()
