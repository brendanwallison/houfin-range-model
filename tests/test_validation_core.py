"""The shared validation primitives: one definition each, used by every validation arm."""
import numpy as np

from src.community_encoder.train_DESK.validation_core import (
    abba_halves, load_holdout_masks, row_splits, split_half_groups)


def test_rebuilt_buffer_matches_the_trainers_ring(tmp_path):
    # A checkpoint without buffer_cells.npy must get the ring the trainer drew, not an empty one:
    # the old fallback let bars and fits read buffer cells the model never trained on.
    from src.community_encoder.train_DESK.augment import blocked_holdout
    valid = np.ones((30, 30), bool)
    val, ring = blocked_holdout(valid, block_cells=6, holdout_frac=0.3, buffer_cells=2, seed=3)
    np.save(tmp_path / "holdout_cells.npy", val)
    np.savez(tmp_path / "desk_meta.npz", spatial_kernel=5)
    ho, bf, note = load_holdout_masks(str(tmp_path))
    assert note.startswith("rebuilt") and np.array_equal(bf, ring)
    np.save(tmp_path / "buffer_cells.npy", ring)
    assert load_holdout_masks(str(tmp_path))[2] == "saved buffer_cells.npy"


def test_buffer_floor_widens_the_rebuilt_ring(tmp_path):
    ho = np.zeros((20, 20), bool)
    ho[8:12, 8:12] = True
    np.save(tmp_path / "holdout_cells.npy", ho)
    np.savez(tmp_path / "desk_meta.npz", spatial_kernel=1)          # kernel alone: width 0
    assert not load_holdout_masks(str(tmp_path))[1].any()
    _, bf, _ = load_holdout_masks(str(tmp_path), buffer_floor=2)
    assert bf[6, 8] and not bf[5, 8]


def test_missing_holdout_is_reported_not_invented(tmp_path):
    ho, bf, note = load_holdout_masks(str(tmp_path))
    assert ho is None and bf is None and "no " in note


def test_split_half_groups_are_year_balanced_and_disjoint():
    years = np.array([2001, 1990, 1995, 1992, 1999, 1993, 1991, 1997])
    a, b, ok = split_half_groups([tuple(range(8)), (0,)], years=years)
    assert list(ok) == [True, False]
    assert set(a[0]).isdisjoint(b[0]) and len(a[0]) == len(b[0]) == 4
    assert years[list(a[0])].mean() == years[list(b[0])].mean() or \
        abs(years[list(a[0])].mean() - years[list(b[0])].mean()) < 1.5
    # sorted by YEAR, not by row index: rows given in reverse year order still balance exactly
    rows = np.arange(8)
    yrs = 2007 - rows                                   # row 0 is the LATEST year
    ha, hb = abba_halves(rows, yrs)
    assert yrs[ha].mean() == yrs[hb].mean()


def test_row_splits_exclude_withheld_years_from_training():
    ho = np.zeros((12, 12), bool)
    ho[0:6, 6:12] = True
    keys = np.array([[8, 1, 1970], [8, 1, 2000], [1, 7, 1970], [1, 7, 2000]])
    tr, grp, _ = row_splits(keys, ho, None, 6, withheld_years=[1970])
    assert tr.tolist() == [False, True, False, False]
    assert grp.tolist() == [2, 0, 3, 1]
