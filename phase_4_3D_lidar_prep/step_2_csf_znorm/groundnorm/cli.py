"""
Command-line interface for groundnorm package.
"""

import argparse
import sys
from pathlib import Path
from .core import run_csf_ground_filter, run_smrf_ground_filter, normalize_las_z
from .dask_runner import process_batch_csf, process_batch_smrf


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ground classification and Z normalization for LAS files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file CSF
  groundnorm csf input.las output.las
  
  # Single file SMRF
  groundnorm smrf input.las output.las
  
  # Normalize Z
  groundnorm normalize input.las ground.las output.las
  
  # Batch processing
  groundnorm batch-csf /path/to/tiles /path/to/output
  groundnorm batch-smrf /path/to/tiles /path/to/output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # CSF command
    csf_parser = subparsers.add_parser('csf', help='Run CSF ground filter on single file')
    csf_parser.add_argument('input_las', type=Path, help='Input LAS file')
    csf_parser.add_argument('output_las', type=Path, help='Output ground LAS file')
    csf_parser.add_argument('--resolution', type=float, default=0.5, help='Cloth resolution')
    csf_parser.add_argument('--iterations', type=int, default=500, help='Max iterations')
    csf_parser.add_argument('--threshold', type=float, default=0.01, help='Classification threshold')
    csf_parser.add_argument('--rigidness', type=int, default=2, choices=[1,2,3], help='Rigidness (1-3)')
    
    # SMRF command
    smrf_parser = subparsers.add_parser('smrf', help='Run SMRF ground filter on single file')
    smrf_parser.add_argument('input_las', type=Path, help='Input LAS file')
    smrf_parser.add_argument('output_las', type=Path, help='Output ground LAS file')
    smrf_parser.add_argument('--slope', type=float, default=0.15, help='Slope parameter')
    smrf_parser.add_argument('--window', type=float, default=3.5, help='Window size')
    smrf_parser.add_argument('--threshold', type=float, default=0.25, help='Elevation threshold')
    smrf_parser.add_argument('--scalar', type=float, default=1.35, help='Scalar parameter')
    
    # Normalize command
    norm_parser = subparsers.add_parser('normalize', help='Normalize Z values (HAG)')
    norm_parser.add_argument('input_las', type=Path, help='Input LAS file')
    norm_parser.add_argument('ground_las', type=Path, help='Ground LAS file')
    norm_parser.add_argument('output_las', type=Path, help='Output normalized LAS file')
    norm_parser.add_argument('--k', type=int, default=3, help='Number of neighbors for interpolation')
    norm_parser.add_argument('--dim-name', type=str, default='Z_norm', help='Name of extra dimension')
    
    # Batch CSF
    batch_csf_parser = subparsers.add_parser('batch-csf', help='Run CSF on multiple files')
    batch_csf_parser.add_argument('input_dir', type=Path, help='Input directory with LAS files')
    batch_csf_parser.add_argument('output_dir', type=Path, help='Output directory')
    batch_csf_parser.add_argument('--pattern', type=str, default='*.las', help='File pattern')
    batch_csf_parser.add_argument('--normalize', action='store_true', help='Also normalize Z')
    batch_csf_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # Batch SMRF
    batch_smrf_parser = subparsers.add_parser('batch-smrf', help='Run SMRF on multiple files')
    batch_smrf_parser.add_argument('input_dir', type=Path, help='Input directory with LAS files')
    batch_smrf_parser.add_argument('output_dir', type=Path, help='Output directory')
    batch_smrf_parser.add_argument('--pattern', type=str, default='*.las', help='File pattern')
    batch_smrf_parser.add_argument('--normalize', action='store_true', help='Also normalize Z')
    batch_smrf_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'csf':
            run_csf_ground_filter(
                args.input_las,
                args.output_las,
                resolution=args.resolution,
                iterations=args.iterations,
                threshold=args.threshold,
                rigidness=args.rigidness,
            )
        
        elif args.command == 'smrf':
            run_smrf_ground_filter(
                args.input_las,
                args.output_las,
                slope=args.slope,
                window=args.window,
                threshold=args.threshold,
                scalar=args.scalar,
            )
        
        elif args.command == 'normalize':
            normalize_las_z(
                args.input_las,
                args.ground_las,
                args.output_las,
                k=args.k,
                extra_dim_name=args.dim_name,
            )
        
        elif args.command == 'batch-csf':
            process_batch_csf(
                args.input_dir,
                args.output_dir,
                pattern=args.pattern,
                normalize=args.normalize,
                n_workers=args.workers,
            )
        
        elif args.command == 'batch-smrf':
            process_batch_smrf(
                args.input_dir,
                args.output_dir,
                pattern=args.pattern,
                normalize=args.normalize,
                n_workers=args.workers,
            )
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
