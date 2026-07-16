#!/usr/bin/env python3
"""Thin launcher: derive the 4km ocean mask from the aggregated BUI raster.

Logic lives in ``src/data/preprocess/ocean_mask.py``. Equivalent: `python -m src.data.deprecated.preprocess.ocean_mask`.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.deprecated.preprocess.ocean_mask import main

if __name__ == "__main__":
    main()
