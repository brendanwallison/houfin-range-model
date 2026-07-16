"""Tests for src/temporal.py — the canonical model-timeline contract.

Runs standalone (``python tests/test_temporal.py``) or under pytest.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import temporal


def test_bio_year_months():
    # Bio-year T = Aug(T-1) .. Jul(T): 12 pairs, correct span, no post-count leak.
    m = temporal.bio_year_months(1902, start_month=8)
    assert len(m) == 12
    assert m[0] == (1901, 8) and m[-1] == (1902, 7)
    # every month is <= Jul of T (nothing after the ~June count year)
    assert all((y, mo) <= (1902, 7) for y, mo in m)
    # a different start month still yields a 12-month contiguous window
    assert len(temporal.bio_year_months(2000, start_month=10)) == 12


def test_year_to_index_gapsafe():
    years = list(range(1902, 1911))  # 1902..1910
    assert temporal.year_to_index(years, 1902) == 0
    assert temporal.year_to_index(years, 1910) == 8
    got = temporal.year_to_index(years, np.array([1903, 1902, 1910]))
    assert list(got) == [1, 0, 8]
    # a year outside the timeline raises (surfaces desync, not a silent offset)
    for bad in (lambda: temporal.year_to_index(years, 1899),
                lambda: temporal.year_to_index(years, np.array([1902, 1950]))):
        try:
            bad(); raise AssertionError("expected KeyError")
        except KeyError:
            pass


def test_year_to_index_beats_subtraction_on_gaps():
    # With a gap (1905 missing), bare subtraction desyncs; the lookup stays right.
    years = [1902, 1903, 1904, 1906, 1907]  # 1905 absent
    assert temporal.year_to_index(years, 1906) == 3          # correct scan index
    assert 1906 - years[0] == 4 != 3                          # subtraction is wrong
    try:
        temporal.assert_contiguous(years); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    temporal.assert_contiguous([1902, 1903, 1904])            # contiguous ok


def test_invasion_timestep_derived():
    tl = {"first_year": 1902, "end_year": 2025, "invasion_year": 1940, "bio_year_start_month": 8}
    assert temporal.invasion_timestep(tl) == 38               # 1940 - 1902
    assert temporal.invasion_timestep(tl, first_year=1900) == 40  # old start -> 1940 still
    # invariant: first_year + inv_timestep == invasion_year (always calendar 1940)
    assert tl["first_year"] + temporal.invasion_timestep(tl) == tl["invasion_year"]


def test_load_timeline_defaults_and_config():
    tl = temporal.load_timeline()  # from config/data_config.json
    assert tl["first_year"] == 1902 and tl["end_year"] == 2025
    assert tl["invasion_year"] == 1940 and tl["bio_year_start_month"] == 8
    ys = temporal.model_years(tl)
    assert ys[0] == 1902 and ys[-1] == 2025 and len(ys) == 2025 - 1902 + 1
    temporal.assert_contiguous(ys)


if __name__ == "__main__":
    test_bio_year_months(); print("[bio-year] Aug(T-1)->Jul(T), 12 months, no leak OK")
    test_year_to_index_gapsafe(); print("[year->index] scalar/array + out-of-range raise OK")
    test_year_to_index_beats_subtraction_on_gaps(); print("[gap] lookup correct where subtraction fails OK")
    test_invasion_timestep_derived(); print("[invasion] derived = invasion_year - first_year OK")
    test_load_timeline_defaults_and_config(); print("[config] timeline loads: 1902..2025, inv 1940 OK")
    print("\nALL TEMPORAL CONTRACT CHECKS PASSED")
