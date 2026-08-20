"""Multi-epoch direction panel: the pure cores.

The panel's whole purpose is to decide a 0.03 difference that 36 cells could not, so a quiet
error in WHICH cell-year each side reads would be invisible in the output and would corrupt the
verdict. These pin the selection rules.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.validate_epoch_directions import (
    DEFAULT_EPOCHS, _idw_at, nearest_survey)


def test_each_cell_uses_its_own_nearest_actual_survey():
    """No averaging and no interpolation in time: the model is read at the same REAL year the
    survey happened, so a cell surveyed in 1986 is compared at 1986, not at 1985."""
    pip = np.array([[0, 0, 1986], [0, 0, 1991], [1, 1, 1984], [2, 2, 1999]], dtype=np.int32)
    near = nearest_survey(pip, None, (1985,), tol=2)
    assert near[1985] == {(0, 0): 1986, (1, 1): 1984}          # 1999 is outside +/-2
    assert (2, 2) not in near[1985]


def test_ties_break_to_the_earlier_year_not_to_row_order():
    """1983 and 1987 are both 2 from 1985. Breaking by whichever row came first would make the
    panel depend on the point set's ordering."""
    a = nearest_survey(np.array([[0, 0, 1983], [0, 0, 1987]], np.int32), None, (1985,), 2)
    b = nearest_survey(np.array([[0, 0, 1987], [0, 0, 1983]], np.int32), None, (1985,), 2)
    assert a[1985][(0, 0)] == b[1985][(0, 0)] == 1983


def test_tolerance_is_respected_exactly_at_the_boundary():
    pip = np.array([[0, 0, 1967], [1, 1, 1969], [2, 2, 1970]], dtype=np.int32)
    near = nearest_survey(pip, None, (1967,), tol=2)
    assert set(near[1967]) == {(0, 0), (1, 1)}                 # 1970 is 3 away
    assert nearest_survey(pip, None, (1967,), tol=3)[1967][(2, 2)] == 1970


def test_unsupervised_rows_never_supply_an_epoch():
    """Duplicate cell-years exist for the ESK basis only. If one supplied the epoch value the
    same cell-year could enter twice, or a non-target row could stand in for a target."""
    pip = np.array([[0, 0, 1985], [1, 1, 1985]], dtype=np.int32)
    near = nearest_survey(pip, np.array([True, False]), (1985,), 2)
    assert set(near[1985]) == {(0, 0)}


def test_the_default_epochs_are_spaced_for_the_learned_ema():
    """~19 years is about two output-EMA half-lives at the learned 10.5 y -- the shortest
    interval over which the EMA is not dominating the predicted change."""
    gaps = np.diff(DEFAULT_EPOCHS)
    assert gaps.min() >= 18 and gaps.max() <= 20, gaps


def test_idw_reads_each_training_cell_at_its_own_epoch_year():
    """The bar must be built the same way the model side is, or the comparison is rigged: a
    training cell contributes its value at ITS nearest year to the epoch."""
    train = {(0, c): (1984 if c % 2 == 0 else 1986) for c in range(10)}
    seen = []

    def z_of(cell, year):
        seen.append((cell, year))
        return np.array([float(year), 1.0])

    out = _idw_at([(5, 5)], train, z_of, k=8)
    assert out.shape == (1, 2)
    assert set(seen) == {(c, y) for c, y in train.items()}
    assert 1984.0 <= out[0, 0] <= 1986.0


def test_idw_declines_when_there_are_too_few_training_cells():
    """Fewer than k neighbours would silently reuse the same cell k times and read as a strong
    baseline built from one point."""
    assert _idw_at([(0, 0)], {(1, 1): 2000, (2, 2): 2000}, lambda c, y: np.zeros(2), k=8) is None
    assert _idw_at([], {(i, i): 2000 for i in range(20)}, lambda c, y: np.zeros(2)) is None


def test_a_constant_field_interpolates_to_that_constant():
    out = _idw_at([(4, 4)], {(i, 0): 2000 for i in range(12)},
                  lambda c, y: np.array([3.0, -1.0]), k=8)
    assert np.allclose(out[0], [3.0, -1.0])
