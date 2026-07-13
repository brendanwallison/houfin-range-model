#!/usr/bin/env python3
"""Thin launcher: reproject PRISM monthly climate onto the model grid.

Logic lives in ``src/data/preprocess/prism.py``. Equivalent: `python -m src.data.preprocess.prism`.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.preprocess.prism import main

if __name__ == "__main__":
    main()
