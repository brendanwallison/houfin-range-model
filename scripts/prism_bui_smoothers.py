#!/usr/bin/env python3
"""Thin launcher: build EMA-smoothed yearly PRISM+BUI state grids (combine stage).

Logic lives in ``src/data/combine/states.py``. Equivalent:
``python -m src.data.combine.states``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.combine.states import main

if __name__ == "__main__":
    main()
