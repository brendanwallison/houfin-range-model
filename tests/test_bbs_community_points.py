"""Pure cores of the raw-BBS community target builder.

The subtle pieces are the ones with no loud failure mode: a weight vector misaligned against
X_points attaches the wrong weight to the wrong cell-year and nothing downstream notices, and
a non-gap-aware EMA smooths a 12-year gap as hard as a 1-year step.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_community_points import (
    align_to_keys, cell_year_weights, temporal_ema, write_points,
)

_RK = ["CountryNum", "StateNum", "Route"]


def _route_cells(rows):
    """rows = [(route, row, col), ...]"""
    return pd.DataFrame([{"CountryNum": 840, "StateNum": 2, "Route": r, "row": rr, "col": cc}
                         for r, rr, cc in rows])


def _fy(rows):
    """rows = [(route, year, first_year), ...]"""
    return pd.DataFrame([{"CountryNum": 840, "StateNum": 2, "Route": r, "Year": y,
                          "first_year": f} for r, y, f in rows])


def _cov(rows):
    """rows = [(row, col, year, n_routes), ...]"""
    return pd.DataFrame([{"row": rr, "col": cc, "year": y, "n_routes": n}
                         for rr, cc, y, n in rows])


# ----------------------------- weights -----------------------------

def test_single_route_cell_year_is_either_full_or_downweighted():
    """The common case: 79% of route-bearing cells hold exactly one route, so the weight is
    either 1.0 or first_year_weight with nothing in between."""
    w = cell_year_weights(
        cov_df=_cov([(5, 6, 1990, 1), (5, 6, 1991, 1)]),
        route_cells=_route_cells([(1, 5, 6)]),
        fy_flags=_fy([(1, 1990, 1), (1, 1991, 0)]),
        first_year_weight=0.5)
    got = {(int(r.row), int(r.col), int(r.year)): float(r.weight) for r in w.itertuples()}
    assert got[(5, 6, 1990)] == 0.5      # observer's first year on this route
    assert got[(5, 6, 1991)] == 1.0


def test_multi_route_cell_year_downweights_by_the_flagged_fraction():
    """Two routes in one cell, one of them a first-year: the cell-year is half contaminated,
    so it takes half the penalty rather than all or none of it."""
    w = cell_year_weights(
        cov_df=_cov([(5, 6, 1990, 2)]),
        route_cells=_route_cells([(1, 5, 6), (2, 5, 6)]),
        fy_flags=_fy([(1, 1990, 1), (2, 1990, 0)]),
        first_year_weight=0.5)
    row = w.iloc[0]
    assert abs(float(row.fy_frac) - 0.5) < 1e-9
    assert abs(float(row.weight) - 0.75) < 1e-9      # 1 - 0.5*(1-0.5)


def test_weight_one_disables_the_downweight_entirely():
    w = cell_year_weights(_cov([(0, 0, 1990, 1)]), _route_cells([(1, 0, 0)]),
                          _fy([(1, 1990, 1)]), first_year_weight=1.0)
    assert float(w.iloc[0].weight) == 1.0


def test_unflagged_cell_year_keeps_full_weight():
    """A coverage row with no matching flag must not silently become 0."""
    w = cell_year_weights(_cov([(9, 9, 2000, 1)]), _route_cells([(1, 5, 6)]),
                          _fy([(1, 1990, 1)]), first_year_weight=0.5)
    assert float(w.iloc[0].weight) == 1.0 and float(w.iloc[0].fy_frac) == 0.0


def test_weight_out_of_range_is_refused():
    for bad in (-0.1, 1.5):
        try:
            cell_year_weights(_cov([(0, 0, 1, 1)]), _route_cells([(1, 0, 0)]),
                              _fy([(1, 1, 0)]), first_year_weight=bad)
        except ValueError:
            continue
        raise AssertionError(f"first_year_weight={bad} was accepted")


# ----------------------------- alignment -----------------------------

def test_align_to_keys_follows_key_order_not_frame_order():
    """keys is the authoritative row order for EVERY per-point artifact. A weight vector in
    the frame's order instead would attach weights to the wrong cell-years silently."""
    keys = np.array([[1, 1, 2000], [0, 0, 1990], [2, 2, 2010]], dtype="int32")
    df = pd.DataFrame({"row": [0, 2, 1], "col": [0, 2, 1],
                       "year": [1990, 2010, 2000], "weight": [0.1, 0.3, 0.2]})
    vals, n_missing = align_to_keys(keys, df, "weight")
    assert n_missing == 0
    assert np.allclose(vals, [0.2, 0.1, 0.3])        # keys order, not frame order


def test_align_to_keys_reports_missing_rather_than_hiding_them():
    keys = np.array([[0, 0, 1990], [7, 7, 1995]], dtype="int32")
    df = pd.DataFrame({"row": [0], "col": [0], "year": [1990], "weight": [0.5]})
    vals, n_missing = align_to_keys(keys, df, "weight", default=1.0)
    assert n_missing == 1
    assert vals[0] == 0.5 and vals[1] == 1.0


# ----------------------------- temporal EMA -----------------------------

def test_ema_off_by_default_is_the_identity():
    X = np.array([[1.0], [5.0]], dtype="float32")
    keys = np.array([[0, 0, 1990], [0, 0, 1991]], dtype="int32")
    assert temporal_ema(X, keys, 0) is X
    assert np.allclose(temporal_ema(X, keys, -1), X)


def test_ema_is_gap_aware():
    """A 12-year gap must smooth far less than a 1-year step. With a fixed alpha both would
    pull the same amount, which is not an EMA of elapsed time -- and BBS cell-years are only
    ~50% dense, with 2020 missing everywhere."""
    X = np.array([[0.0], [10.0], [0.0], [10.0]], dtype="float32")
    close = temporal_ema(X, np.array([[0, 0, 1990], [0, 0, 1991]], "int32"), tau=2.0)
    far = temporal_ema(X[2:], np.array([[0, 0, 1990], [0, 0, 2002]], "int32"), tau=2.0)
    # both start at 0 then see 10; the near step retains more history (so lands lower)
    assert close[1, 0] < far[1, 0]
    assert abs(float(far[1, 0]) - 10.0) < 0.1        # a 12-yr gap is nearly no smoothing


def test_ema_is_causal_and_never_looks_forward():
    """The first year of a cell is untouched, and a later spike cannot alter an earlier row."""
    keys = np.array([[0, 0, 1990], [0, 0, 1991], [0, 0, 1992]], dtype="int32")
    X = np.array([[1.0], [1.0], [100.0]], dtype="float32")
    out = temporal_ema(X, keys, tau=2.0)
    assert out[0, 0] == 1.0                          # first year = raw
    assert abs(out[1, 0] - 1.0) < 1e-6               # unchanged by the 1992 spike
    assert out[2, 0] < 100.0                         # the spike itself is damped


def test_ema_keeps_cells_independent():
    """Smoothing must not leak between cells -- that would be spatial smoothing, which is the
    thing this whole target exists to avoid."""
    keys = np.array([[0, 0, 1990], [9, 9, 1991]], dtype="int32")
    X = np.array([[0.0], [10.0]], dtype="float32")
    out = temporal_ema(X, keys, tau=2.0)
    assert np.allclose(out, X), "a different cell's value bled across"


def test_ema_sorts_by_year_regardless_of_row_order():
    keys = np.array([[0, 0, 1992], [0, 0, 1990], [0, 0, 1991]], dtype="int32")
    X = np.array([[9.0], [0.0], [0.0]], dtype="float32")
    out = temporal_ema(X, keys, tau=1.0)
    assert out[1, 0] == 0.0                          # 1990 is the cell's first year
    assert out[2, 0] < out[0, 0]                     # 1991 smoothed before 1992


# ----------------------------- artifact writing -----------------------------

def test_write_points_refuses_ragged_artifacts(tmp_path):
    """X, keys and weights are three parallel arrays in one row order. A length mismatch is a
    silent misalignment of every per-point value, so it must fail at write time."""
    X = np.zeros((3, 2), "float32"); keys = np.zeros((3, 3), "int32")
    try:
        write_points(str(tmp_path), X, keys, np.ones(2, "float32"), {})
    except ValueError as exc:
        assert "ragged" in str(exc)
    else:
        raise AssertionError("a short weight vector was written")


def test_write_points_roundtrips(tmp_path):
    X = np.arange(6, dtype="float32").reshape(3, 2)
    keys = np.array([[0, 0, 1990], [1, 1, 1991], [2, 2, 1992]], dtype="int32")
    w = np.array([1.0, 0.5, 1.0], dtype="float32")
    write_points(str(tmp_path), X, keys, w, {"target_source": "bbs_raw"})
    assert np.allclose(np.load(tmp_path / "X_points.npy"), X)
    assert np.array_equal(np.load(tmp_path / "point_index.npy"), keys)
    assert np.allclose(np.load(tmp_path / "point_weights.npy"), w)
    import json
    assert json.load(open(tmp_path / "points_meta.json"))["target_source"] == "bbs_raw"
