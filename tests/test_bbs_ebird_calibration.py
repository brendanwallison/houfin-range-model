"""Partially pooled calibration of BBS onto the eBird scale.

The behaviour worth pinning is the shrinkage itself: a species with plenty of overlapping data
should be fitted by that data, and a species with little or with no real relationship should be
pulled to the population estimate SMOOTHLY -- in proportion to its evidence, not by crossing a
threshold. An earlier version used hard cutoffs on overlap count and correlation, which made 49
paired observations behave completely differently from 51.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_ebird_calibration import (
    apply_calibration, calibration_meta, fit_hierarchical_calibration,
)


def _pairs(n, slope, intercept=0.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 4.0, n)
    return x, intercept + slope * x + rng.normal(0, noise, n)


# ----------------------------- the fit itself -----------------------------

def test_a_species_with_ample_data_is_fitted_by_its_own_data():
    cal = fit_hierarchical_calibration({0: _pairs(3000, slope=2.5, intercept=0.4)},
                                       n_species=1, verbose=False)
    assert abs(cal["b"][0] - 2.5) < 0.05, cal["b"][0]
    assert abs(cal["a"][0] - 0.4) < 0.1
    assert cal["shrinkage"][0] < 0.05, "ample data should barely shrink"


def test_a_species_with_no_overlap_lands_exactly_on_the_population():
    """No special case, no flag: with no data the estimate IS the population estimate."""
    cal = fit_hierarchical_calibration({0: _pairs(2000, slope=3.0)}, n_species=2,
                                       verbose=False)
    assert cal["n"][1] == 0
    assert abs(cal["b"][1] - cal["mu_b"]) < 1e-9
    assert abs(cal["a"][1] - cal["mu_a"]) < 1e-9
    assert cal["shrinkage"][1] == 1.0


def test_shrinkage_increases_smoothly_as_evidence_thins():
    """The property the old thresholds could not have. More data means less pull toward the
    population, continuously -- there is no point at which a species' treatment jumps."""
    shr = []
    for n in (20, 100, 500, 2500):
        cal = fit_hierarchical_calibration(
            {0: _pairs(n, slope=3.0, seed=1), 1: _pairs(3000, slope=1.2, seed=2),
             2: _pairs(3000, slope=1.4, seed=3)}, n_species=3, verbose=False)
        shr.append(cal["shrinkage"][0])
    assert all(shr[i] > shr[i + 1] for i in range(len(shr) - 1)), shr
    assert shr[0] > shr[-1] * 3, "thin data should be pulled substantially harder"


def test_a_species_whose_products_disagree_cannot_be_flattened_or_inverted():
    """Replaces the old correlation cutoff, and fixes a real flaw in the first version of this
    model. A species whose two products carry no relationship has data genuinely saying "slope
    zero" -- so a directly-fitted slope lands near 0, which flattens that species to a constant
    in every cell and destroys whatever BBS knew about it, or below 0, which inverts it. Both
    are reachable with a Gaussian prior on the slope and neither is a calibration.

    Fitting exp(beta) makes both unreachable: the slope is positive for every beta, and the
    prior pulling beta toward 0 actively resists collapse rather than merely discouraging it.
    No threshold is involved, and nothing is dropped."""
    rng = np.random.default_rng(7)
    agree = [_pairs(3000, slope=2.0, seed=s) for s in (10, 11, 12)]
    noise_x = rng.uniform(0.5, 4.0, 400)
    noise_y = rng.normal(2.0, 1.5, 400)                   # unrelated to x
    pairs = {i: p for i, p in enumerate(agree)}
    pairs[3] = (noise_x, noise_y)
    cal = fit_hierarchical_calibration(pairs, n_species=4, verbose=False)
    assert cal["b"][3] > 0.0, "a calibration slope must be positive"
    assert cal["b"][3] < cal["b"][0], "its own data should still pull it away from the rest"
    # Worth being precise about what this does and does not guarantee. exp(beta) rules out
    # inversion and exact zero. It does NOT stop a species with 400 observations genuinely
    # saying "these two products do not covary" from receiving a small slope -- that is the
    # data speaking, and it means BBS's variation for that species carries no information
    # about the shared scale. The slope RANGE is logged so such a species is visible rather
    # than silently contributing a near-constant column.


def test_every_slope_is_positive_even_when_the_data_says_otherwise():
    """The structural guarantee. Anti-correlated data would give a negative slope under a
    direct fit -- calibrating so that more BBS birds mean less eBird abundance."""
    x, y = _pairs(2000, slope=2.0, seed=20)
    cal = fit_hierarchical_calibration({0: (x, -y), 1: _pairs(2000, slope=2.0, seed=21)},
                                       n_species=2, verbose=False)
    assert (cal["b"] > 0).all(), cal["b"]


def test_the_population_prior_is_centred_on_the_products_corresponding():
    """In log space a slope of 1 means the two products differ by a pure scale factor, which is
    the honest prior: they are two measurements of the same thing. With no data at all every
    species sits there."""
    cal = fit_hierarchical_calibration({}, n_species=3, verbose=False)
    assert np.allclose(cal["b"], 1.0) and np.allclose(cal["a"], 0.0)
    assert cal["prior"]["slope"] == 1.0 and cal["prior"]["intercept"] == 0.0
    assert abs(cal["mu_beta"]) < 1e-9, "log-slope 0 is slope 1"


def test_evidence_can_move_the_population_off_the_prior():
    """The prior is a starting belief, not a constraint: enough species agreeing on a different
    relationship must move the population estimate."""
    pairs = {i: _pairs(2000, slope=3.5, seed=i) for i in range(8)}
    cal = fit_hierarchical_calibration(pairs, n_species=8, verbose=False)
    assert cal["mu_b"] > 3.0, cal["mu_b"]


def test_species_with_genuinely_different_slopes_keep_them():
    """Partial pooling must not flatten real between-species differences."""
    pairs = {0: _pairs(3000, slope=1.0, seed=4), 1: _pairs(3000, slope=4.0, seed=5)}
    cal = fit_hierarchical_calibration(pairs, n_species=2, verbose=False)
    assert abs(cal["b"][0] - 1.0) < 0.15 and abs(cal["b"][1] - 4.0) < 0.15


def test_fit_is_deterministic():
    p = {0: _pairs(500, slope=2.0), 1: _pairs(60, slope=1.5, seed=9)}
    a = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    b = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    assert np.allclose(a["b"], b["b"]) and np.allclose(a["a"], b["a"])


# ----------------------------- application -----------------------------

def test_apply_is_per_species_and_clipped_at_zero():
    """Negative output is outside log1p-Ruzicka's domain: sum(min)/sum(max) with a negative
    term can exceed 1 or flip the denominator's sign."""
    cal = {"a": np.array([1.0, -5.0]), "b": np.array([2.0, 1.0])}
    out = apply_calibration(np.array([[1.0, 1.0], [2.0, 10.0]]), cal)
    assert np.allclose(out[:, 0], [3.0, 5.0])
    assert out[0, 1] == 0.0
    assert np.allclose(out[1, 1], 5.0)


def test_apply_refuses_a_species_count_mismatch():
    try:
        apply_calibration(np.zeros((2, 5)), {"a": np.zeros(3), "b": np.ones(3)})
    except ValueError as exc:
        assert "species" in str(exc)
    else:
        raise AssertionError("a species-count mismatch was accepted")


def test_meta_is_json_safe_and_records_direction_and_shrinkage():
    import json
    cal = fit_hierarchical_calibration({0: _pairs(500, slope=2.0)}, n_species=2, verbose=False)
    m = calibration_meta(cal, ["houspa", "amegfi"])
    json.dumps(m)
    assert m["direction"] == "bbs_to_ebird" and m["method"] == "hierarchical_map"
    assert set(m["per_species"]) == {"houspa", "amegfi"}
    # shrinkage is the field that replaces a pass/fail flag, so it must be present per species
    assert 0.0 <= m["per_species"]["houspa"]["shrinkage"] <= 1.0
    assert m["per_species"]["amegfi"]["shrinkage"] == 1.0        # no overlap


def test_report_zero_effect_surfaces_a_positive_intercept():
    """The instrument that decides whether zeros need handling at all. It cannot be run
    off-cluster on real data, so what is tested here is that it would flag the case."""
    from src.community_encoder.train_DESK.bbs_ebird_calibration import report_zero_effect
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    r = report_zero_effect(X, {"a": np.array([0.21, 0.30]), "b": np.array([0.5, 0.5])},
                           verbose=False)
    assert r["n_positive_intercepts"] == 2
    assert r["occupancy_before"] < r["occupancy_after"] == 1.0
    assert r["floor_birds_median"] > 0.0


def test_report_zero_effect_is_quiet_when_intercepts_are_negative():
    """With a negative intercept the existing clip already sends absences to 0, so there is
    nothing to fix. In a simulation with realistic BBS detection most species landed here."""
    from src.community_encoder.train_DESK.bbs_ebird_calibration import report_zero_effect
    X = np.array([[0.0, 2.0], [1.0, 0.0]])
    r = report_zero_effect(X, {"a": np.array([-0.2, -0.1]), "b": np.array([0.5, 0.5])},
                           verbose=False)
    assert r["n_positive_intercepts"] == 0
    assert r["occupancy_after"] <= r["occupancy_before"]


