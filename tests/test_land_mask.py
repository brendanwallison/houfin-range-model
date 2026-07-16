"""Tests for src/data/preprocess/land_mask.py — coastline + gated snapping.

Runs standalone (``python tests/test_land_mask.py``) or under pytest. NumPy only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess import land_mask as lm


def test_land_fraction_and_threshold_de_dilates():
    # A 4x4 fine grid -> one 4km... one target cell (block 4). 6 of 16 land.
    fine = np.zeros((4, 4)); fine[:2, :3] = 1  # 6 land subpixels
    frac = lm.compute_land_fraction(fine, block=4)
    assert frac.shape == (1, 1)
    assert abs(frac[0, 0] - 6 / 16) < 1e-9
    # Old "any land -> land" rule would call this cell land; a tau=0.5 threshold
    # de-dilates it (6/16 = 0.375 < 0.5 -> ocean).
    assert lm.land_mask_from_fraction(frac, tau=0.5)[0, 0] == False
    assert lm.land_mask_from_fraction(frac, tau=0.3)[0, 0] == True


def test_snap_gated_by_radius():
    # Land on the left half; ocean on the right.
    land = np.zeros((1, 10), dtype=bool)
    land[0, :5] = True
    rows = np.array([0, 0, 0, 0])
    cols = np.array([3, 5, 6, 9])   # on-land, 1-off, 2-off, 4-off coast (coast at col 4)
    sr, sc, keep = lm.snap_to_nearest_land(rows, cols, land, max_cells=1)
    # col 3 on land -> unchanged, kept
    assert keep[0] and (sr[0], sc[0]) == (0, 3)
    # col 5 is 1 cell off the coast (col 4) -> snapped to col 4, kept
    assert keep[1] and sc[1] == 4
    # col 6 is 2 cells off -> beyond max_cells=1 -> dropped
    assert not keep[2]
    # col 9 far offshore -> dropped
    assert not keep[3]
    # a larger radius keeps the 2-off point
    _, _, keep2 = lm.snap_to_nearest_land(rows, cols, land, max_cells=2)
    assert keep2[2]


if __name__ == "__main__":
    test_land_fraction_and_threshold_de_dilates()
    print("[land-fraction] block-mean fraction + tau threshold de-dilates OK")
    test_snap_gated_by_radius()
    print("[snap] on-land kept, near-coast snapped, offshore dropped, radius honored OK")
    print("\nALL LAND-MASK / COASTLINE CHECKS PASSED")
