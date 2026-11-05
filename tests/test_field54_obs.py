import numpy as np

from src.processing.coarsen_model_grid import parent_cells
from research.exp.field54_obs import block_any, coarse_masks


def test_parent_cells_snaps_off_land_parents_to_nearest_land():
    land_c = np.array([[True, False], [False, False]])
    rr, rc, off, keep, dist = parent_cells(np.array([0, 1, 3]), np.array([0, 3, 3]), land_c, 2)
    assert list(zip(rr, rc)) == [(0, 0), (0, 0), (0, 0)]
    assert list(off) == [False, True, True] and keep.all()
    assert dist[0] == 0 and dist[1] == 1 and np.isclose(dist[2], np.sqrt(2))
    rr, rc, off, keep, _ = parent_cells(np.array([0, 1]), np.array([0, 3]), land_c, 2, coastal='drop')
    assert list(keep) == [True, False]


def test_block_any_pads_odd_edges():
    m = np.zeros((3, 3), bool); m[2, 2] = True
    out = block_any(m, (2, 2))
    assert out.tolist() == [[False, False], [False, True]]


def test_coarse_masks_any_child_holdout_wins_and_snap_destinations_are_withheld():
    h = np.zeros((4, 4), bool); b = np.zeros((4, 4), bool)
    h[0, 0] = True; b[0, 1] = True        # same parent (0,0): holdout wins
    b[2, 2] = True                        # parent (1,1)
    h[3, 0] = True                        # parent (1,0) is water -> its observations snap to (0,1)
    land = np.array([[True, True], [False, True]])
    src = np.array([[0, 0], [3, 0], [2, 2]])
    dst = np.array([[0, 0], [0, 1], [1, 1]])
    ho, bf, info = coarse_masks(h, b, land, src, dst)
    assert ho.tolist() == [[True, True], [False, False]]
    assert bf.tolist() == [[False, False], [False, True]]
    assert info['holdout_cells_added_by_snap'] == 1
