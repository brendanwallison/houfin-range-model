#!/usr/bin/env python3
"""Thin launcher for BBS preprocessing.

Logic lives in ``src/data/preprocess/bbs.py`` (grids the raw BBS release into
model-ready observation records + core/margin initialization). Equivalent:
``python -m src.data.preprocess.bbs``.

Note: this is the *preprocess* step and assumes the raw BBS 2026 release CSVs
already sit under ``{datasets_root}/bbs_2026_release`` -- fetched by the *acquire*
step ``scripts/download_bbs.py`` (wired into ``download_all.sh``).
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.preprocess.bbs import main

if __name__ == "__main__":
    main()
