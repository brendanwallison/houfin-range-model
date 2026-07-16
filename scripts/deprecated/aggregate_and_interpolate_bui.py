#!/usr/bin/env python3
"""Thin launcher: aggregate the 250 m BUI series to the model grid.

Logic lives in ``src/data/preprocess/bui.py``. Equivalent:
``python -m src.data.deprecated.preprocess.bui [--interpolate]``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.deprecated.preprocess.bui import main

if __name__ == "__main__":
    main()
