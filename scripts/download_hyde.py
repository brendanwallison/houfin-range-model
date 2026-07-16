#!/usr/bin/env python3
"""Thin launcher for the HYDE 3.3 downloader.

Logic lives in ``src/data/acquire/hyde.py``. Example:
    python scripts/download_hyde.py --list
Equivalent: ``python -m src.data.acquire.hyde ...``.
Note: set data_config.json hyde.base_url to the confirmed HYDE 3.3 location.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.hyde import main

if __name__ == "__main__":
    main()
