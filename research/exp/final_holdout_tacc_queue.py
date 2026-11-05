"""Install fixed final-evaluation tasks in the EXISTING age-worker queue2, atomically."""
from pathlib import Path

ROOT = Path("/work/07980/bwa386/ls6/houfin/research/final_holdout_20261025")
QUEUE = Path("/work/07980/bwa386/ls6/houfin/research/age_model/queue2")
TASKS = [(arm, 1966, 1986, "1", 1) for arm in ["v1_r64", "v2_r64", "v2_r128", "v2_s64"]]
TASKS += [(arm, lo, hi, "1", 1) for lo, hi in [(1966, 1975), (1976, 1985), (1986, 1995)]
          for arm in ["v1_r64", "v2_r64", "v2_r128", "v2_s64", "v2_w64"]]
TASKS += [(arm, 1966, 1986, "0.03", seed) for seed in [1, 2] for arm in ["v1_r64", "v2_r64", "v2_r128"]]

for i, (arm, lo, hi, thin, seed) in enumerate(TASKS, 9910):
    name = f"{i}_final_{arm}_e{lo}_{hi}_f{thin}_s{seed}.sh"
    if any((QUEUE / status / name).exists() for status in ["pending", "running", "done", "failed"]):
        print("already registered", name)
        continue
    script = ("#!/bin/bash\nset -euo pipefail\n"
              f"export FINAL_HOLDOUT_ROOT={ROOT}\n"
              "export FINAL_PYTHON=/work/07980/bwa386/ls6/houfin/venv/bin/python\n"
              f"export RESEARCH_SEALED_READ_LOG={ROOT}/sealed_reads.csv\n"
              f"bash {ROOT}/code_score/research/bin/final_holdout_summer.sh {arm} {lo} {hi} {thin} {seed}\n")
    temp = QUEUE / "pending" / (name + ".tmp")
    temp.write_text(script)
    temp.replace(QUEUE / "pending" / name)
    print("queued", name)
