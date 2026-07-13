#!/usr/bin/env python3
"""Thin launcher: build age-model inputs from Z/Z_disp + BBS (combine stage).

Logic lives in ``src/data/combine/model_inputs.py``. Equivalent:
``python -m src.data.combine.model_inputs``.

Consumes Z/Z_disp, the BBS grid, and the ocean mask already at the model
resolution (grid.target_res_m); there is no AGG_FACTOR downsample.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.combine.model_inputs import main

if __name__ == "__main__":
    main()
