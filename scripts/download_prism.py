#!/usr/bin/env python3
"""Thin launcher for the PRISM monthly-climate downloader.

Logic lives in ``src/data/acquire/prism.py``. Equivalent:
``python -m src.data.acquire.prism [--scan-only|--verify|--resume]``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.prism import main

if __name__ == "__main__":
    main()
