"""Calibration of BBS onto the eBird scale: E = k_s * B^d_s, partially pooled.

Three properties carry the design and each has a test that would fail loudly without it:
a zero maps to a zero (so no measured absence becomes a fabricated presence), d = 1 with k free
expresses a pure unit conversion (which an earlier log1p form could not), and per-species values
shrink toward a LEARNED population value continuously, with no threshold.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_ebird_calibration import (
    apply_calibration, calibration_meta, fit_hierarchical_calibration,
)


def _pairs(n, k, d, noise=0.02, seed=0):
    """Overlapping cell-years obeying E = k*B^d, returned as log1p (the caller's units)."""
    rng = np.random.default_rng(seed)
    B = rng.uniform(0.5, 200.0, n)
    E = k * B ** d * np.exp(rng.normal(0, noise, n))
    return np.log1p(B), np.log1p(E)


# ----------------------------- the form -----------------------------

def test_a_zero_stays_a_zero_even_though_the_scale_is_free():
    """The property the whole form exists for, and the one an additive intercept breaks. k is
    multiplicative, so a free scale cannot create a floor."""
    out = apply_calibration(np.array([[0.0, np.log1p(4.0)], [np.log1p(2.0), 0.0]]),
                            {"k": np.array([5.0, 0.2]), "d": np.array([1.0, 0.8])})
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    assert out[0, 1] > 0.0 and out[1, 0] > 0.0


def test_occupancy_is_preserved_exactly():
    rng = np.random.default_rng(0)
    X = np.where(rng.random((300, 12)) < 0.83, 0.0, np.log1p(rng.uniform(0.5, 50, (300, 12))))
    out = apply_calibration(X, {"k": rng.uniform(0.05, 3.0, 12), "d": rng.uniform(0.4, 1.3, 12)})
    assert ((out == 0.0) == (X == 0.0)).all()


def test_a_pure_unit_conversion_is_representable_and_recovered():
    """The failure that motivated this form. If eBird is simply 0.2x BBS the model must
    reproduce it exactly. The previous log1p(E)=b*log1p(B) could not express it for any b --
    the implied b slid from 0.235 at low abundance to 0.720 at high.

    Checked through the PREDICTION rather than the raw parameters, because k is anchored at a
    typical BBS count rather than at one bird, so its numeric value depends on that anchor."""
    cal = fit_hierarchical_calibration({0: _pairs(3000, k=0.2, d=1.0, noise=0.0)},
                                       n_species=1, verbose=False)
    assert abs(cal["d"][0] - 1.0) < 0.02, cal["d"][0]
    B = np.array([[1.0], [10.0], [100.0]])
    out = np.expm1(apply_calibration(np.log1p(B), {"k": cal["k"][:1], "d": cal["d"][:1],
                                                   "B0": cal["B0"]}))
    assert np.allclose(out.ravel(), 0.2 * B.ravel(), rtol=0.05), out.ravel()


def test_apply_refuses_a_species_count_mismatch():
    try:
        apply_calibration(np.zeros((2, 5)), {"k": np.ones(3), "d": np.ones(3)})
    except ValueError as exc:
        assert "species" in str(exc)
    else:
        raise AssertionError("a species-count mismatch was accepted")


# ----------------------------- the fit -----------------------------

def test_a_species_with_ample_data_recovers_its_own_exponent_and_scale():
    cal = fit_hierarchical_calibration({0: _pairs(3000, k=0.5, d=0.7)}, n_species=1,
                                       verbose=False)
    # pulled slightly toward 1 by the linear preference, but its own data still decides
    assert 0.70 <= cal["d"][0] <= 0.78, cal["d"][0]
    assert cal["shrinkage"][0] < 0.20      # a real pull toward linear, not a constraint


def test_species_with_genuinely_different_exponents_keep_them():
    cal = fit_hierarchical_calibration({0: _pairs(3000, 0.5, 0.6, seed=1),
                                        1: _pairs(3000, 2.0, 1.2, seed=2)},
                                       n_species=2, verbose=False)
    assert abs(cal["d"][0] - 0.6) < 0.05 and abs(cal["d"][1] - 1.2) < 0.08


def test_a_species_with_no_overlap_lands_exactly_on_the_population():
    cal = fit_hierarchical_calibration({0: _pairs(2000, 0.4, 0.8)}, n_species=2, verbose=False)
    assert cal["n"][1] == 0
    assert abs(cal["d"][1] - cal["mu_d"]) < 1e-9
    assert abs(cal["k"][1] - cal["mu_k"]) < 1e-9
    assert cal["shrinkage"][1] == 1.0


def test_shrinkage_increases_smoothly_as_evidence_thins():
    """What the old hard cutoffs could not do. More data means less pull toward the population,
    continuously -- there is no count at which a species' treatment jumps."""
    shr = []
    for n in (15, 60, 300, 2000):
        cal = fit_hierarchical_calibration(
            {0: _pairs(n, 0.5, 1.4, seed=1), 1: _pairs(3000, 0.5, 0.8, seed=2),
             2: _pairs(3000, 0.5, 0.9, seed=3)}, n_species=3, verbose=False)
        shr.append(cal["shrinkage"][0])
    assert all(shr[i] > shr[i+1] for i in range(len(shr)-1)), shr
    assert shr[0] > shr[-1] * 5


def test_every_exponent_is_positive():
    """A negative exponent would invert a species: more BBS birds meaning less eBird
    abundance. exp() makes that unreachable."""
    x, y = _pairs(2000, 0.5, 0.8, seed=20)
    cal = fit_hierarchical_calibration({0: (x, y[::-1]), 1: _pairs(2000, 0.5, 0.8, seed=21)},
                                       n_species=2, verbose=False)
    assert (cal["d"] > 0).all()


def test_the_linear_preference_is_a_preference_and_the_data_wins():
    """The exponent's population location is nudged toward 1, not held there. With every
    species truly at 1.5 and 2000 observations each, the fit lands near 1.4 -- a small pull
    toward linear that plenty of evidence overrides. Holding it harder means taking the
    population prior well below 0.05, because that location is estimated from all species at
    once and the prior competes with every one of them."""
    pairs = {i: _pairs(2000, k=0.05, d=1.5, seed=i) for i in range(8)}
    cal = fit_hierarchical_calibration(pairs, n_species=8, verbose=False)
    assert cal["mu_d"] < 1.5, cal["mu_d"]               # pulled toward linear
    assert cal["mu_d"] > 1.2, cal["mu_d"]               # but nowhere near pinned
    assert 1.30 < cal["d"].mean() < 1.50, cal["d"]


def test_genuinely_linear_data_is_left_alone():
    """The other side: when the data agrees with the preference, nothing is distorted."""
    pairs = {i: _pairs(2000, k=0.05, d=1.0, seed=i) for i in range(8)}
    cal = fit_hierarchical_calibration(pairs, n_species=8, verbose=False)
    assert abs(cal["mu_d"] - 1.0) < 0.02
    assert np.abs(cal["d"] - 1.0).max() < 0.02


def test_the_prior_is_a_pure_unit_conversion():
    """d = 1 with k free. With no data at all, that is where every species sits."""
    cal = fit_hierarchical_calibration({}, n_species=3, verbose=False)
    assert np.allclose(cal["d"], 1.0)
    assert cal["prior"]["exponent"] == 1.0


def test_fit_is_deterministic():
    p = {0: _pairs(500, 0.5, 0.9), 1: _pairs(40, 0.5, 1.1, seed=9)}
    a = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    b = fit_hierarchical_calibration(p, n_species=2, verbose=False)
    assert np.allclose(a["d"], b["d"]) and np.allclose(a["k"], b["k"])


def test_meta_is_json_safe_and_records_the_form():
    import json
    cal = fit_hierarchical_calibration({0: _pairs(500, 0.5, 0.9)}, n_species=2, verbose=False)
    m = calibration_meta(cal, ["houspa", "amegfi"])
    json.dumps(m)
    assert m["direction"] == "bbs_to_ebird"
    assert "(bbs/B0)**d_s" in m["form"]
    assert m["B0_typical_bbs_count"] is not None
    assert m["per_species"]["amegfi"]["shrinkage"] == 1.0


def test_ample_data_still_overrides_the_tight_spread_prior():
    """The spreads are tight (0.10, so 95% of species within a 1.49x span of the population).
    Tight must not mean rigid: a species with plenty of overlapping data still moves toward its
    own value, because the prior is a belief about the POPULATION spread, not a constraint."""
    truth = [0.75, 0.90, 1.00, 1.15, 1.30]
    p = {i: _pairs(4000, k=0.3, d=d, noise=0.10, seed=i) for i, d in enumerate(truth)}
    cal = fit_hierarchical_calibration(p, len(truth), verbose=False)
    assert list(np.argsort(cal["d"])) == list(range(len(truth)))     # ordering preserved
    assert cal["d"].max() - cal["d"].min() > 0.25, cal["d"]          # real separation survives
    assert np.median(cal["shrinkage"]) < 0.15


def test_the_tight_exponent_prior_pulls_thin_species_together():
    """The other half of the same property. With 40 observations a species should not be
    trusted to claim an exponent far from its peers, and the tighter prior is what enforces
    that -- continuously, without a cutoff."""
    truth = [0.60, 0.80, 1.00, 1.25, 1.50]
    p = {i: _pairs(40, k=0.3, d=d, noise=0.15, seed=i) for i, d in enumerate(truth)}
    cal = fit_hierarchical_calibration(p, len(truth), verbose=False)
    spread_fitted = cal["d"].max() - cal["d"].min()
    assert spread_fitted < (max(truth) - min(truth)) * 0.6, spread_fitted
    assert np.median(cal["shrinkage"]) > 0.3
    # ordering is still preserved -- shrinkage pulls together, it does not scramble
    assert list(np.argsort(cal["d"])) == list(range(len(truth)))


def test_the_two_parameters_get_different_prior_structures():
    """The exponent's population location carries a prior at 1, because a linear relationship
    between the two surveys is what we expect. The scale's is effectively flat, because the
    units conversion between a route count and a modelled abundance index is genuinely unknown
    and the data should decide it. Its SPREAD is still tight, because species should not differ
    wildly in detectability BETWEEN the two surveys."""
    pr = fit_hierarchical_calibration({}, n_species=2, verbose=False)["prior"]
    assert pr["population_log_exponent_sd"] <= 0.1, "the exponent's location is unconstrained"
    assert pr["population_log_scale_sd"] > 10 * pr["log_scale_sd"], "the scale is not free"
    assert np.exp(4 * pr["log_scale_sd"]) < 2.0, "detectability ratios too loose"


def test_the_population_scale_really_does_follow_the_data():
    """A free location is only free if the data can actually move it. Every species here has a
    ratio far from 1; the population must follow rather than anchoring at the prior."""
    p = {i: _pairs(2000, k=0.02, d=1.0, seed=i) for i in range(6)}
    cal = fit_hierarchical_calibration(p, 6, verbose=False)
    expected = 0.02 * cal["B0"]                        # d = 1, so eBird at B0 is 0.02*B0
    assert abs(np.log(cal["mu_k"] / expected)) < 0.3, (cal["mu_k"], expected)


def test_the_four_priors_are_separate_knobs():
    """Between-species spread and confidence in the population location are different beliefs,
    for each of the two parameters, so there are four values and not two. They may happen to
    coincide numerically -- both exponent knobs are 0.05 -- but they are set independently."""
    pr = fit_hierarchical_calibration({}, n_species=2, verbose=False)["prior"]
    for key in ("log_exponent_sd", "log_scale_sd",
                "population_log_exponent_sd", "population_log_scale_sd"):
        assert key in pr, key
    assert pr["log_scale_sd"] != pr["population_log_scale_sd"]
