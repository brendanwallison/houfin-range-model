#!/usr/bin/env python3
"""Thin launcher for the climr climate acquire step.

Logic lives in ``src/data/acquire/climatena.py`` (drives climate_climr.R via
Rscript). Example:
    python scripts/climate_climr.py --centroids <dir>/cell_centroids.csv --out <dir> --dry-run
Equivalent: ``python -m src.data.acquire.climatena ...``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.climatena import main

if __name__ == "__main__":
    main()
