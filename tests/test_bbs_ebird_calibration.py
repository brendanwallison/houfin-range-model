"""Calibration of BBS onto the eBird scale: a power law through the origin, partially pooled.

Two things carry the design. The form is `log1p(ebird) = b_s * log1p(bbs)` with NO intercept, so
a zero maps to a zero and no measured absence can become a fabricated presence. And b_s is
shrunk toward a population value in proportion to the species' own evidence, so nothing needs a
threshold: 49 overlapping observations behave almost identically to 51.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_ebird_calibration import (
    apply_calibration, calibration_meta, fit_hierarchical_calibration,
)


def _pairs(n, slope, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 4.0, n)
    return x, slope * x + rng.normal(0, noise, n)


# ----------------------------- the form -----------------------------

def test_a_zero_stays_a_zero():
    """The property the whole form exists for. An entry of 0 is a surveyed cell-year where the
    species was not recorded; with an intercept it would become a small presence, and absences
    are 83% of the real BBS matrix -- so every cell-year would share a floor in every species,
    which is exactly what Ruzicka cannot see past."""
    out = apply_calibration(np.array([[0.0, 2.0], [1.0, 0.0]]),
                            {"b": np.array([0.5, 0.7])})
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    assert out[0, 1] > 0.0 and out[1, 0] > 0.0


def test_occupancy_is_preserved_exactly_across_a_whole_matrix():
    rng = np.random.default_rng(0)
    X = np.where(rng.random((300, 12)) < 0.83, 0.0, rng.uniform(0.5, 4.0, (300, 12)))
    out = apply_calibration(X, {"b": rng.uniform(0.3, 1.4, 12)})
    assert ((out == 0.0) == (X == 0.0)).all()
    assert abs((out > 0).mean() - (X > 0).mean()) < 1e-12


def test_apply_refuses_a_species_count_mismatch():
    try:
        apply_calibration(np.zeros((2, 5)), {"b": np.ones(3)})
    except ValueError as exc:
        assert "species" in str(exc)
    else:
        raise AssertionError("a species-count mismatch was accepted")


# ----------------------------- the fit -----------------------------

def test_a_species_with_ample_data_recovers_its_own_slope():
    cal = fit_hierarchical_calibration({0: _pairs(3000, 2.5)}, n_species=1, verbose=False)
    assert abs(cal["b"][0] - 2.5) < 0.05, cal["b"][0]
    assert cal["shrinkage"][0] < 0.01


def test_species_with_genuinely_different_slopes_keep_them():
    """Partial pooling must not flatten real between-species differences."""
    cal = fit_hierarchical_calibration({0: _pairs(3000, 0.4, seed=1),
                                        1: _pairs(3000, 2.8, seed=2)},
                                       n_species=2, verbose=False)
    assert abs(cal["b"][0] - 0.4) < 0.06 and abs(cal["b"][1] - 2.8) < 0.15


def test_a_species_with_no_overlap_lands_exactly_on_the_population():
    """No special case, no flag: with no data the estimate IS the population estimate."""
    cal = fit_hierarchical_calibration({0: _pairs(2000, 3.0)}, n_species=2, verbose=False)
    assert cal["n"][1] == 0
    assert abs(cal["b"][1] - cal["mu_b"]) < 1e-9
    assert cal["shrinkage"][1] == 1.0


def test_shrinkage_increases_smoothly_as_evidence_thins():
    """The property the old hard cutoffs could not have. More data means less pull toward the
    population, continuously -- there is no count at which a species' treatment jumps."""
    shr = []
    for n in (15, 60, 300, 2000):
        cal = fit_hierarchical_calibration(
            {0: _pairs(n, 3.0, seed=1), 1: _pairs(3000, 1.1, seed=2),
             2: _pairs(3000, 1.3, seed=3)}, n_species=3, verbose=False)
        shr.append(cal["shrinkage"][0])
    assert all(shr[i] > shr[i+1] for i in range(len(shr)-1)), shr
    assert shr[0] > shr[-1] * 5


def test_every_slope_is_positive_even_when_the_data_says_otherwise():
    """exp(beta) makes inversion unreachable. A negative slope would calibrate a species so
    that more BBS birds mean less eBird abundance, which is never a calibration."""
    x, y = _pairs(2000, 2.0, seed=20)
    cal = fit_hierarchical_calibration({0: (x, -y), 1: _pairs(2000, 2.0, seed=21)},
                                       n_species=2, verbose=False)
    assert (cal["b"] > 0).all(), cal["b"]


def test_the_prior_is_centred_on_the_products_corresponding():
    """A slope of 1 means the two products differ by a pure scale factor, which is the honest
    prior for two measurements of the same thing. With no data at all, that is where every
    species sits."""
    cal = fit_hierarchical_calibration({}, n_species=3, verbose=False)
    assert np.allclose(cal["b"], 1.0)
    assert abs(cal["mu_beta"]) < 1e-9


def test_evidence_can_move_the_population_off_the_prior():
    pairs = {i: _pairs(2000, 0.35, seed=i) for i in range(8)}
    cal = fit_hierarchical_calibration(pairs, n_species=8, verbose=False)
    assert cal["mu_b"] < 0.6, cal["mu_b"]


def test_fit_is_deterministic():
    p = {0: _pairs(500, 2.0), 1: _pairs(40, 1.5, seed=9)}
    a = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    b = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    assert np.allclose(a["b"], b["b"])


def test_meta_is_json_safe_and_records_the_form():
    import json
    cal = fit_hierarchical_calibration({0: _pairs(500, 2.0)}, n_species=2, verbose=False)
    m = calibration_meta(cal, ["houspa", "amegfi"])
    json.dumps(m)
    assert m["direction"] == "bbs_to_ebird"
    assert "b_s * log1p(bbs)" in m["form"]              # records that there is no intercept
    assert m["per_species"]["amegfi"]["shrinkage"] == 1.0
    assert "a" not in m["per_species"]["houspa"]
