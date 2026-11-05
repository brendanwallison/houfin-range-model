#!/bin/bash
# Submit the twin's Stage E chain once the twin DESK cube AND its EMA cube exist (run on a TACC login node).
# e1 inner grid -> e2 field fits (raw D, EMA D) -> e3 scoring + pre-registered rules. Development queue, CPU.
set -euo pipefail
F=/work/07980/bwa386/ls6/houfin/research/field54; S=$F/code_bc/research/field54
E=$F/runs/field54_eval
for f in $E/models/backcast40/interp_s1/CUBE_DONE $E/cubes/backcast40/desk_interp_s1_ema/EMA_DONE.json; do
  test -e $f || { echo "not ready: $f"; exit 3; }; done
a=$(sbatch $S/e1_inner.slurm | grep 'Submitted batch job' | awk '{print $4}')
b=$(sbatch -d afterok:$a $S/e2_fits.slurm | grep 'Submitted batch job' | awk '{print $4}')
c=$(sbatch -d afterok:$b $S/e3_score.slurm | grep 'Submitted batch job' | awk '{print $4}')
echo "stageE e1=$a e2=$b e3=$c $(date)" | tee -a $F/logs/bc_jobs.txt
