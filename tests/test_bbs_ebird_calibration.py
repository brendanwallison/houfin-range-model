"""eBird -> BBS scale calibration: the estimator choice and the fallback ladder.

The two things with silent failure modes: an OLS-style estimator compresses the variance of
what it produces (so calibrated eBird rows would look more like each other than real BBS rows
do, and the kernel would learn source membership), and a negative-correlation species would
get an abundance-inverting slope.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_ebird_calibration import (
    apply_calibration, calibration_meta, fit_calibration, ols_fit, rma_fit, scale_only_fit,
)


# ----------------------------- estimators -----------------------------

def test_rma_recovers_an_exact_linear_relation():
    x = np.linspace(0.0, 5.0, 50)
    a, b, r = rma_fit(x, 3.0 + 2.0 * x)
    assert abs(b - 2.0) < 1e-9 and abs(a - 3.0) < 1e-9 and abs(r - 1.0) < 1e-9


def test_rma_preserves_variance_where_ols_compresses_it():
    """The reason RMA is the default. With noise, OLS's slope is attenuated so its
    predictions have less spread than the target scale; RMA's do not. Calibrated eBird rows
    with compressed variance would be systematically more self-similar than BBS rows, making
    the data source itself a signal in the kernel."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    y = 2.0 * x + rng.normal(0, 1.5, 4000)              # substantial noise
    a_r, b_r, _ = rma_fit(x, y)
    a_o, b_o, _ = ols_fit(x, y)
    sd_target = y.std()
    sd_rma = (a_r + b_r * x).std()
    sd_ols = (a_o + b_o * x).std()
    assert abs(sd_rma / sd_target - 1.0) < 0.02, "RMA should preserve the target's spread"
    assert sd_ols < 0.9 * sd_target, "OLS should visibly compress it"
    assert b_o < b_r


def test_estimators_return_nan_on_a_constant_input():
    for f in (rma_fit, ols_fit):
        _, b, _ = f(np.ones(10), np.arange(10.0))
        assert not np.isfinite(b)


def test_scale_only_is_a_pure_offset():
    x = np.array([1.0, 2.0, 3.0])
    a, b, _ = scale_only_fit(x, x + 0.7)
    assert b == 1.0 and abs(a - 0.7) < 1e-9


# ----------------------------- the fallback ladder -----------------------------

def _pair(n, slope=2.0, intercept=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 4.0, n)
    y = intercept + slope * x + rng.normal(0, noise, n)
    return x, y


def test_rung1_species_with_enough_overlap_get_their_own_fit():
    cal = fit_calibration({0: _pair(200), 1: _pair(200, slope=0.5, intercept=0.2)},
                          n_species=2, min_overlap_points=50, verbose=False)
    assert list(cal["rung"]) == ["rma", "rma"]
    assert abs(cal["b"][0] - 2.0) < 0.05 and abs(cal["b"][1] - 0.5) < 0.05


def test_rung2_thin_species_fall_back_to_the_pooled_fit():
    """A species with too few overlapping cell-years must not get a fit from 5 points."""
    cal = fit_calibration({0: _pair(500), 1: _pair(5, seed=1)},
                          n_species=2, min_overlap_points=50, verbose=False)
    assert cal["rung"][0] == "rma"
    assert cal["rung"][1] == "pooled_rma"
    assert cal["b"][1] == cal["pooled"]["b"]
    assert cal["n"][1] == 5, "the thin count is still recorded for the audit trail"


def test_a_species_absent_from_the_overlap_still_gets_a_usable_calibration():
    """Species index 2 never appears in the overlap at all. It must still be calibrated (via
    pooled) rather than left with a nan that would poison its whole column."""
    cal = fit_calibration({0: _pair(200), 1: _pair(200)}, n_species=3,
                          min_overlap_points=50, verbose=False)
    assert cal["rung"][2] == "pooled_rma" and np.isfinite(cal["b"][2])
    assert cal["n"][2] == 0


def test_negative_correlation_species_is_refused_and_falls_back():
    """A negative slope would invert that species' abundance -- calibrating so that more
    eBird detections mean fewer BBS birds. That is products disagreeing, not a calibration."""
    x, y = _pair(300)
    anti = (x, -y)                                       # perfectly anti-correlated
    cal = fit_calibration({0: _pair(300), 1: anti}, n_species=2,
                          min_overlap_points=50, verbose=False)
    assert cal["rung"][1] == "pooled_rma", "an inverting slope was accepted"
    assert cal["b"][1] > 0.0


def test_rung3_scale_only_when_nothing_can_be_fitted():
    """Every overlap is constant in x, so no slope exists anywhere. The pooled rung has to
    degrade to a pure offset rather than emitting nan."""
    cal = fit_calibration({0: (np.ones(100), np.ones(100) * 3.0)}, n_species=1,
                          min_overlap_points=10, verbose=False)
    assert cal["pooled"]["rung"] == "pooled_scale_only"
    assert cal["b"][0] == 1.0 and abs(cal["a"][0] - 2.0) < 1e-9


def test_empty_overlap_does_not_crash_and_yields_an_identity():
    cal = fit_calibration({}, n_species=3, verbose=False)
    assert np.isfinite(cal["b"]).all() and (cal["b"] == 1.0).all()
    assert (cal["a"] == 0.0).all()


def test_unknown_form_is_refused():
    try:
        fit_calibration({}, n_species=1, form="magic", verbose=False)
    except ValueError as exc:
        assert "magic" in str(exc)
    else:
        raise AssertionError("an unknown calibration form was accepted")


# ----------------------------- application -----------------------------

def test_apply_calibration_is_per_species_and_clipped_at_zero():
    """Negative output is outside log1p-Ruzicka's domain: sum(min)/sum(max) with a negative
    term can exceed 1 or flip the denominator's sign."""
    cal = {"a": np.array([1.0, -5.0]), "b": np.array([2.0, 1.0])}
    out = apply_calibration(np.array([[1.0, 1.0], [2.0, 10.0]]), cal)
    assert np.allclose(out[:, 0], [3.0, 5.0])            # a + b*x per species
    assert out[0, 1] == 0.0                              # -5 + 1 clipped
    assert np.allclose(out[1, 1], 5.0)


def test_apply_calibration_refuses_a_species_count_mismatch():
    cal = {"a": np.zeros(3), "b": np.ones(3)}
    try:
        apply_calibration(np.zeros((2, 5)), cal)
    except ValueError as exc:
        assert "species" in str(exc)
    else:
        raise AssertionError("a species-count mismatch was accepted")


def test_calibration_meta_is_json_safe():
    import json
    cal = fit_calibration({0: _pair(100)}, n_species=2, min_overlap_points=50, verbose=False)
    meta = calibration_meta(cal, ["houspa", "amegfi"])
    json.dumps(meta)                                     # must not raise
    assert set(meta["per_species"]) == {"houspa", "amegfi"}
    assert meta["per_species"]["houspa"]["rung"] == "rma"
    # a nan correlation must serialize as null, not the string "nan"
    assert meta["per_species"]["amegfi"]["r"] is None or isinstance(
        meta["per_species"]["amegfi"]["r"], float)


def test_min_r_rejects_a_weak_but_positive_correlation():
    """A near-zero correlation is the subtle failure, and it matters BECAUSE of RMA: the slope
    is sign(r)*sd(y)/sd(x), which stays large even at r ~ 0, so a species with no real
    agreement would still get a confident-looking variance-matching slope on a random sign.
    Surfaced by a synthetic-grid run where 53 species took per-species fits at median r=0.044."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 4000)
    weak = 0.05 * x + rng.normal(0, 1, 4000)             # r ~ 0.05
    strong = 2.0 * x + rng.normal(0, 0.3, 4000)          # r ~ 0.99
    cal = fit_calibration({0: (x, strong), 1: (x, weak)}, n_species=2,
                          min_overlap_points=50, min_r=0.2, verbose=False)
    assert cal["rung"][0] == "rma", "a strong correlation should get its own fit"
    assert cal["rung"][1] == "pooled_rma", "a weak correlation must fall back"
    # ...and with the guard relaxed it would have been accepted, which is the point
    loose = fit_calibration({1: (x, weak)}, n_species=2, min_overlap_points=50,
                            min_r=0.0, verbose=False)
    assert loose["rung"][1] == "rma"


def test_min_r_is_recorded_for_the_audit_trail():
    cal = fit_calibration({}, n_species=1, min_r=0.35, verbose=False)
    assert cal["min_r"] == 0.35
    assert calibration_meta(cal, ["x"])["min_overlap_points"] == 50


def test_absent_entries_must_be_masked_to_zero_not_left_at_the_intercept():
    """The invariant behind build_ebird_window_rows' ``have`` mask. A species outside its own
    trend window has no data, but log1p(0)=0 and the calibration is affine, so a + b*0 = a
    would hand it the intercept as a measured abundance. Caught by a synthetic grid where a
    species with a 2014-2016 window came out at 0.373 in 2012."""
    cal = {"a": np.array([0.37, 1.0]), "b": np.array([2.0, 2.0])}
    X_log = np.zeros((2, 2))                             # no data anywhere
    calibrated = apply_calibration(X_log, cal)
    assert (calibrated > 0).any(), "precondition: the intercept does leak without a mask"
    have = np.array([[False, True], [False, True]])
    masked = np.where(have, calibrated, 0.0)
    assert (masked[:, 0] == 0.0).all(), "absent species must be exactly 0"
    assert (masked[:, 1] > 0.0).all()
