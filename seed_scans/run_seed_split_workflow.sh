#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plantcv

python ~/seed_scans/plantcv_workflow_miller_seeds-grid.py "$@"
