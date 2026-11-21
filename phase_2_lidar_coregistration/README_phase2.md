# Phase 2: LiDAR-RGB Co-registration

Align LiDAR point clouds using GeFolki optical flow.

## Method

Compute a displacement field between DSMs and apply it to LiDAR points. Validate the resulting alignment for each acquisition.

The workflow was inspired by: https://github.com/RS-Haynes/efolki_coreg/blob/main/efolki_coreg_example.ipynb (loads hyperspectral images in ENVI format, extracts matching wavelengths, then calculates optical flow)

**Workflow:**
1. Preprocess DSMs (histogram matching, filtering)
2. Compute optical flow between temporal DSM pairs
3. Apply displacement field to LiDAR point clouds
4. Validate alignment accuracy

## Contents

```
phase_2_lidar_coregistration/
├── REPRODUCIBILITY.md                 # Step-by-step handoff checklist
├── configs/
│   └── coregistration_job.example.yml # Copy-and-edit job template
├── gefolki/
│   ├── python_myscripts/              # Main processing scripts (config-driven)
│   │   ├── phase2_config.py           # Shared YAML config loader
│   │   ├── gefolki_preprocess.py      # DSM preprocessing
│   │   ├── gefolki_flow_l1.py         # Flow computation (L1 sensor)
│   │   ├── gefolki_flow_l2.py         # Flow computation (L2 sensor)
│   │   ├── align_lidar_with_flow_l1.py # Apply flow to points (L1)
│   │   └── align_lidar_with_flow_l2.py # Apply flow to points (L2)
│   ├── notebooks/                     # Interactive workflows
│   │   ├── 1_gefolki_preprocess.ipynb
│   │   ├── 2_gefolki_flow_l1-site01.ipynb
│   │   ├── 2_gefolki_flow_l1-site11.ipynb
│   │   ├── 2_gefolki_flow_l2.ipynb
│   │   ├── 3_align_lidar_with_flow_l2.ipynb
│   │   └── stats_align_lidar_with_flow.py.ipynb # Statistical validation
│   ├── python/                        # GeFolki core library (Python)
│   ├── matlab/                        # GeFolki core library (MATLAB)
│   └── datasets/                      # Example datasets for testing
└── step_0_checks/                     # CRS validation tools
```

## Usage

### Notebooks (Recommended)
```bash
# Step 1: Preprocess DSMs
jupyter notebook gefolki/notebooks/1_gefolki_preprocess.ipynb

# Step 2: Compute optical flow
jupyter notebook gefolki/notebooks/2_gefolki_flow_l1-site01.ipynb

# Step 3: Apply flow to LiDAR
jupyter notebook gefolki/notebooks/3_align_lidar_with_flow_l2.ipynb

# Step 4: Statistical validation (optional)
jupyter notebook gefolki/stats_align_lidar_with_flow.py.ipynb
```

### Scripts
```bash
# Copy the config template and edit paths for your site/pass
cp configs/coregistration_job.example.yml configs/coregistration_job.local.yml

# Run sequentially (see REPRODUCIBILITY.md for details):
cd gefolki/python_myscripts
python gefolki_preprocess.py --config ../../configs/coregistration_job.local.yml
python gefolki_flow_l1.py    --config ../../configs/coregistration_job.local.yml
python align_lidar_with_flow_l1.py --config ../../configs/coregistration_job.local.yml
```

## Dependencies

Install the shared Python environment with `uv sync`; see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup and usage.

## GeFolki Algorithm

GeFolki (Geoscience Extended Flow Optical Lucas-Kanade Iterative) is an optical flow method designed for geoscience applications. It computes dense displacement fields between image pairs using:
- Lucas-Kanade iterative approach
- Multi-scale pyramid processing
- Robust correlation metrics

**References:**
- Original implementation: `gefolki/python/` (adapted from ONERA)
- Documentation: `gefolki/manual_gefolki_english.pdf`
- Additional implementation: https://github.com/RS-Haynes/efolki_coreg
