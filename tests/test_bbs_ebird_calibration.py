"""Latent-abundance calibration of BBS onto the eBird scale (SHELVED; see the module docstring).

Fixtures generate from the model itself -- one true abundance per cell-year, seen by BBS as a
POISSON count and by eBird as a continuous index. That matters: the model derives BBS's noise
from the counts rather than fitting it, so testing it against Gaussian noise on the log scale
would be testing a different model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.bbs_ebird_calibration import (
    TRUST_RATIO, apply_calibration, calibration_meta, corrected_slope,
    fit_hierarchical_calibration, species_moments,
)


def _world(S=30, n=2000, d=1.0, k=0.3, sy=0.15, mean_log=2.2, seed=0, per_species_d=None):
    """Draw from the model: latent abundance, Poisson BBS count, continuous eBird index."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in range(S):
        ds = d if per_species_d is None else per_species_d[s]
        t = rng.normal(mean_log, 0.8, n)                 # true log abundance
        B = rng.poisson(np.exp(t))                       # BBS: a count
        keep = B > 0                                     # only positive pairs reach the fit
        E = k * np.exp(ds * (t[keep] - mean_log)) * np.exp(rng.normal(0, sy, keep.sum()))
        out[s] = (np.log1p(B[keep]), np.log1p(E))
    return out


def _naive_slope(pairs, s):
    """What an ordinary regression of eBird on BBS would say -- the attenuated answer."""
    x = np.log(np.expm1(pairs[s][0])); y = np.log(np.expm1(pairs[s][1]))
    return float(np.cov(x, y)[0, 1] / x.var())


# ----------------------------- the point of the model -----------------------------

def test_the_poisson_correction_recovers_a_slope_an_ordinary_regression_misses():
    """The reason for the form. Regressing eBird on BBS treats the count as exact; a Poisson
    count is not, so the slope comes back shrunk. Subtracting the count's own variance from the
    denominator recovers it."""
    p = _world(d=1.0, mean_log=1.2, seed=0)              # smallish counts -> real Poisson noise
    naive = np.median([_naive_slope(p, s) for s in p])
    cal = fit_hierarchical_calibration(p, 30, verbose=False)
    assert naive < 0.92, naive                           # attenuated
    assert abs(np.median(cal["d"]) - 1.0) < 0.10, np.median(cal["d"])
    assert np.median(cal["d"]) > naive                   # and corrected in the right direction


def test_the_bbs_noise_is_derived_from_the_counts_not_fitted():
    """A Poisson count with mean B carries log-variance about 1/B, so rarer species are noisier
    automatically. Nothing is searched for and there is no free noise parameter."""
    common = _world(S=6, mean_log=3.5, seed=1)           # ~33 birds
    rare = _world(S=6, mean_log=0.7, seed=2)             # ~2 birds
    mc = species_moments(common, 6); mr = species_moments(rare, 6)
    assert np.median(mr[6]) > 5 * np.median(mc[6]), (np.median(mr[6]), np.median(mc[6]))


def test_corrected_slope_reduces_to_the_plain_ratio_when_there_is_no_noise():
    assert abs(corrected_slope(2.0, 8.0, 1.0, 0.0) - 0.5) < 1e-12
    # and steepens as the noise taken out grows
    assert corrected_slope(2.0, 8.0, 1.0, 1.0) > corrected_slope(2.0, 8.0, 1.0, 0.0)


def test_ebird_noise_is_reported_not_assumed():
    """Supplying eBird's noise as well would over-determine the model -- on the real products
    the two assumptions are incompatible. It falls out of the fit and TRUST_RATIO is the check."""
    cal = fit_hierarchical_calibration(_world(seed=3), 30, verbose=False)
    assert TRUST_RATIO == 0.1
    assert cal["trust_ratio"] == TRUST_RATIO
    assert (np.asarray(cal["ebird_noise_var"]) >= 0).all()


# ----------------------------- the form -----------------------------

def test_a_zero_stays_a_zero_even_though_the_scale_is_free():
    """A surveyed cell-year where the species went unrecorded must not become a presence.
    k is multiplicative, so unlike an additive intercept it cannot create a floor."""
    out = apply_calibration(np.array([[0.0, np.log1p(4.0)], [np.log1p(2.0), 0.0]]),
                            {"k": np.array([5.0, 0.2]), "d": np.array([1.0, 0.8]), "B0": 20.0})
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    assert out[0, 1] > 0.0 and out[1, 0] > 0.0


def test_occupancy_is_preserved_exactly():
    rng = np.random.default_rng(0)
    X = np.where(rng.random((300, 12)) < 0.83, 0.0, np.log1p(rng.uniform(0.5, 50, (300, 12))))
    out = apply_calibration(X, {"k": rng.uniform(0.05, 3.0, 12),
                                "d": rng.uniform(0.4, 1.3, 12), "B0": 20.0})
    assert ((out == 0.0) == (X == 0.0)).all()


def test_apply_refuses_a_species_count_mismatch():
    try:
        apply_calibration(np.zeros((2, 5)), {"k": np.ones(3), "d": np.ones(3), "B0": 1.0})
    except ValueError as exc:
        assert "species" in str(exc)
    else:
        raise AssertionError("a species-count mismatch was accepted")


# ----------------------------- pooling -----------------------------

def test_thin_species_pool_toward_the_population():
    shr = [float(np.median(fit_hierarchical_calibration(
        _world(S=30, n=nn, seed=7), 30, verbose=False)["shrinkage"]))
        for nn in (60, 300, 2000)]
    assert all(shr[i] > shr[i + 1] for i in range(len(shr) - 1)), shr


def test_a_species_with_no_overlap_lands_on_the_population():
    p = _world(S=3, seed=4); del p[2]
    cal = fit_hierarchical_calibration(p, 3, verbose=False)
    assert cal["n"][2] == 0
    assert abs(cal["d"][2] - cal["mu_d"]) < 1e-9 and cal["shrinkage"][2] == 1.0


def test_species_with_genuinely_different_exponents_stay_ordered():
    truth = np.array([0.7, 0.9, 1.0, 1.15, 1.3])
    cal = fit_hierarchical_calibration(
        _world(S=5, n=4000, per_species_d=truth, mean_log=2.5, seed=5), 5, verbose=False)
    assert list(np.argsort(cal["d"])) == list(range(5)), cal["d"]


def test_every_exponent_is_positive():
    cal = fit_hierarchical_calibration(_world(S=10, seed=6), 10, verbose=False)
    assert (cal["d"] > 0).all()


# ----------------------------- priors -----------------------------

def test_the_linear_preference_pulls_the_exponent_toward_one():
    cal = fit_hierarchical_calibration(_world(S=30, d=1.4, seed=8), 30, verbose=False)
    assert cal["mu_d"] < 1.35, cal["mu_d"]
    assert cal["mu_d"] > 0.95, cal["mu_d"]


def test_the_scale_population_follows_the_data():
    """Nothing holds it: the units conversion between a route count and a modelled abundance
    index is genuinely unknown."""
    lo = fit_hierarchical_calibration(_world(S=20, k=0.02, seed=10), 20, verbose=False)["mu_k"]
    hi = fit_hierarchical_calibration(_world(S=20, k=2.00, seed=11), 20, verbose=False)["mu_k"]
    assert hi > 10 * lo, (lo, hi)


def test_the_four_priors_are_separate_knobs():
    pr = fit_hierarchical_calibration({}, n_species=2, verbose=False)["prior"]
    for key in ("log_exponent_sd", "log_scale_sd",
                "population_log_exponent_sd", "population_log_scale_sd", "trust_ratio"):
        assert key in pr, key
    assert pr["population_log_scale_sd"] > 10 * pr["log_scale_sd"]
    assert pr["population_log_exponent_sd"] < pr["log_exponent_sd"]


# ----------------------------- plumbing -----------------------------

def test_moments_return_the_poisson_variance_alongside_the_rest():
    n, mx, my, vx, vy, cxy, pvx, log_B0 = species_moments(_world(S=4, n=800, seed=12), 4)
    assert vx.shape == (4,) and pvx.shape == (4,)
    assert (pvx[n > 0] > 0).all()
    assert np.isfinite([mx, my, vx, vy, cxy]).all() and log_B0 > 0


def test_fit_is_deterministic():
    p = _world(S=6, n=600, seed=13)
    a = fit_hierarchical_calibration(p, 6, verbose=False)
    b = fit_hierarchical_calibration(p, 6, verbose=False)
    assert np.allclose(a["d"], b["d"]) and np.allclose(a["k"], b["k"])


def test_meta_is_json_safe_and_records_the_model():
    import json
    cal = fit_hierarchical_calibration(_world(S=3, n=600, seed=14), 3, verbose=False)
    m = calibration_meta(cal, ["a", "b", "c"])
    json.dumps(m)
    assert m["method"] == "latent_abundance_deming_hierarchical_map"
    assert "(bbs/B0)**d_s" in m["form"] and m["trust_ratio"] == TRUST_RATIO
