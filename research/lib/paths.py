"""Where everything lives, in one place. Every research/ script imports this; nothing hardcodes a path.

The data tree mirrors TACC's: ``/work/07980/bwa386/ls6`` is a symlink to ``~/houfin/work`` (created once,
with sudo), so ``HOUFIN_PROCESSED`` is byte-for-byte the string the checkpoints recorded on TACC and
``model_arch.check_basis_matches`` accepts them without edits. ``SCRATCH`` is local, so ``HOUFIN_DATA``
differs from TACC's -- nothing compares that one as a string.

Run everything inside WSL (the Windows interpreter cannot import ``desk_training``: ``import resource``).
"""
import os
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = "/mnt/c/Dev/houfin-range-model"            # the Windows checkout the IDE edits, seen from WSL
TACC_WORK = "/work/07980/bwa386/ls6"              # symlink -> ~/houfin/work
WORK_REAL = HOME / "houfin" / "work"
SCRATCH = HOME / "houfin" / "scratch"
HOUFIN_DATA = f"{SCRATCH}/houfin/data"
HOUFIN_PROCESSED = f"{TACC_WORK}/houfin/processed"
VENV_PY = str(HOME / "houfin" / "venv" / "bin" / "python")
PUEUE = str(HOME / ".local" / "bin" / "pueue")    # `wsl -e` is not a login shell: ~/.local/bin is not on PATH
CODE_SNAPSHOTS = HOME / "houfin" / "runs" / "code"
OUT_ROOT = WORK_REAL / "houfin" / "research"      # experiment outputs (real path: works before the symlink)
CACHE_ROOT = WORK_REAL / "houfin" / "research_cache"
RESEARCH = Path(REPO) / "research"                # state files, versioned

#: Threads per job by queue group. 4 CPU slots x 6 + 1 GPU slot x 8 slightly oversubscribes 32 threads,
#: which is fine for BLAS-bound work that rarely runs all slots at once.
THREADS = {"cpu": 6, "gpu": 8, "default": 8}


def job_env(group="default", code_dir=None, overlay=None, extra=None):
    """The environment a research job runs in. Pure apart from reading os.environ."""
    env = dict(os.environ)
    n = str(THREADS.get(group, 8))
    env.update({
        "WORK": TACC_WORK, "SCRATCH": str(SCRATCH),
        "HOUFIN_DATA": HOUFIN_DATA, "HOUFIN_PROCESSED": HOUFIN_PROCESSED,
        "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
    })
    if code_dir:
        env["PYTHONPATH"] = f"{code_dir}:{code_dir}/src"
    if group == "cpu":
        # gp_kernels.knn_union defaults to CUDA; a CPU-slot job must not contend for the one GPU.
        env["CUDA_VISIBLE_DEVICES"] = ""
    if overlay:
        env["ESK_DESK_CONFIG"] = overlay if os.path.isabs(overlay) else f"{code_dir}/{overlay}"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
