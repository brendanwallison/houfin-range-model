"""Checkpoint-aware lane dispatcher for existing A100 allocations."""
import os
from rolling_dev_dispatch import R,run_lane
run_lane(R,os.environ['SLURM_JOB_ID'],int(os.environ.get('SLURM_LOCALID','0')),int(os.environ['SLURM_JOB_END_TIME']))