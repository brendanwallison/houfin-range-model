#!/usr/bin/env python3
"""Thin launcher for the AVONET species-selection pipeline.

Logic lives in ``src/data/identify/avonet.py`` (the 'identify' stage: decide
which species enter the eBird reference community). Equivalent:
``python -m src.data.identify.avonet``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.identify.avonet import main

if __name__ == "__main__":
    main()
