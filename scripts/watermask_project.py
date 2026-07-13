#!/usr/bin/env python3
"""Thin launcher: reproject the water mask into the BUI grid.

Logic lives in ``src/data/preprocess/watermask.py``. Equivalent: `python -m src.data.preprocess.watermask`.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.preprocess.watermask import main

if __name__ == "__main__":
    main()
