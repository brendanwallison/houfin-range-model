#!/usr/bin/env bash
# Read-only live host inventory. Deliberately never prints the full environment.
set -euo pipefail
date -u +%FT%TZ
printf '\nCPU and RAM\n'
nproc
free -b
printf '\nFilesystems\n'
df -h / /mnt/c /mnt/e
printf '\nGPU\n'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
printf '\nPython experiment processes\n'
experiment_pattern='[p]ython.*(invasion_sandbox|local_takeover|local_joint_fit|local_structural|local_model|structural_fit_readout|structural_campaign_collect|field[-_]geometry)'
ps -eo pid,ppid,etimes,pcpu,pmem,args --sort=-pcpu | awk -v pattern="$experiment_pattern" 'NR == 1 || $0 ~ pattern'
printf '\nExperiment CPU affinity and whitelisted runtime flags\n'
for pid in $(pgrep -f "$experiment_pattern" || true); do
    taskset -pc "$pid" || true
    tr '\0' '\n' < "/proc/$pid/environ" | sed -n '/^JAX_PLATFORMS=/p; /^CUDA_VISIBLE_DEVICES=/p; /^XLA_FLAGS=/p; /^SBX_/p; /^AGE_MODEL_CONFIG=/p; /^PYTHONPATH=/p; /^OMP_NUM_THREADS=/p'
done
