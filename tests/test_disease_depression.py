"""The mycoplasmal-conjunctivitis effect on carrying capacity.

These tests exist because the previous formulation failed silently: an unbounded
penalty subtracted from K's pre-softplus argument annihilated carrying capacity
across the entire eastern US (softplus is effectively exp there, so the penalty was
an unbounded MULTIPLICATIVE one), and nothing in the test suite could have caught
it. The properties asserted here are the ones that make the structured form
trustworthy: it is bounded, it is off before the front arrives, it recovers, and
its priors mean what the config comments claim.
"""
import copy

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.handlers import seed, trace

from src.config_utils import load_age_model_config
from src.data.combine.model_inputs import generate_spatial_basis
from src.model.age_fields import (
    disease_k_fraction,
    disease_onset_timestep,
    disease_severity,
)
from src.model.age_priors import sample_priors

N_LAND = 60
DIS_T0 = 91  # 1993 - 1902, the epizootic window start


def _basis(n_freq=3, n_land=N_LAND, seed_val=0):
    rng = np.random.default_rng(seed_val)
    rows = rng.integers(0, 40, n_land)
    cols = rng.integers(0, 60, n_land)
    return generate_spatial_basis(40, 60, rows, cols, n_freq)


def _disease(mu_sev=0.0, b_late=0.0, rec=0.4, tau_rec=12.0, tau=1.5,
             lag0=0.0, w_scale=0.0, seed_val=0):
    sev_b, lag_b = _basis(3, seed_val=seed_val), _basis(3, seed_val=seed_val)
    rng = np.random.default_rng(seed_val + 1)
    onset = jnp.array(np.linspace(92.0, 105.0, N_LAND))  # arrival, timestep units
    return {
        "sev_basis": jnp.array(sev_b),
        "lag_basis": jnp.array(lag_b),
        "onset": onset,
        "onset_decades": (onset - onset.mean()) / 10.0,
        "mu_sev": mu_sev, "b_late": b_late,
        "w_sev": jnp.array(rng.normal(0, w_scale, sev_b.shape[0])),
        "lag0": lag0,
        "w_lag": jnp.array(rng.normal(0, w_scale, lag_b.shape[0])),
        "tau": tau, "rec": rec, "tau_rec": tau_rec,
    }


# --------------------------------------------------------------- boundedness

@pytest.mark.parametrize("mu_sev", [-8.0, -1.0, 0.0, 1.0, 8.0])
@pytest.mark.parametrize("w_scale", [0.0, 0.5, 3.0])
def test_removed_fraction_stays_in_unit_interval(mu_sev, w_scale):
    """K can be rescaled but never annihilated or amplified, for ANY draw.

    This is the property the old additive-inside-softplus form lacked: it could
    multiply K by exp(-3.5) = 0.03. Extreme parameter values are deliberate here --
    the bound must come from the parameterization, not from the priors.
    """
    d = _disease(mu_sev=mu_sev, w_scale=w_scale)
    for t in (DIS_T0, 100, 110, 123):
        frac = np.asarray(disease_k_fraction(d, float(t)))
        assert np.isfinite(frac).all()
        assert (frac >= 0.0).all(), "the effect must never ADD capacity"
        assert (frac < 1.0).all(), "the effect must never remove ALL capacity"


def test_severity_is_a_fraction_regardless_of_the_field():
    d = _disease(mu_sev=4.0, b_late=2.0, w_scale=5.0)
    sev = np.asarray(disease_severity(d))
    assert ((sev > 0.0) & (sev < 1.0)).all()


# ------------------------------------------------------------------ the gate

def test_no_effect_before_the_front_arrives():
    """Cells the front has not reached must be untouched.

    This is what pins alpha_k on data the disease term cannot reach -- the
    1966-1993 BBS record plus every not-yet-invaded cell -- and it is why the
    Jensen objection to a one-sided link does not bind.
    """
    d = _disease(tau=0.5)
    onset_t = np.asarray(disease_onset_timestep(d))
    # 15 years before each cell's own arrival.
    frac = np.asarray(disease_k_fraction(d, float(onset_t.min() - 15.0)))
    assert (frac < 1e-4).all()


def test_effect_switches_on_across_the_front():
    d = _disease(tau=1.5, rec=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    early = np.asarray(disease_k_fraction(d, float(onset_t.mean() - 6)))
    late = np.asarray(disease_k_fraction(d, float(onset_t.mean() + 6)))
    assert (late > early).all()


def test_onset_ordering_follows_the_arrival_map():
    """At a fixed year, later-arriving cells must be less affected."""
    d = _disease(tau=1.0, rec=0.0, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    frac = np.asarray(disease_k_fraction(d, float(np.median(onset_t))))
    order = np.argsort(onset_t)
    assert np.all(np.diff(frac[order]) <= 1e-6)


def test_lag_shifts_the_front_in_years():
    d0 = _disease(lag0=0.0, tau=0.75, rec=0.0)
    d5 = _disease(lag0=5.0, tau=0.75, rec=0.0)
    t = float(np.asarray(disease_onset_timestep(d0)).mean() + 2)
    assert (np.asarray(disease_k_fraction(d5, t))
            < np.asarray(disease_k_fraction(d0, t))).all()


# -------------------------------------------------------------- the recovery

def test_recovery_relaxes_the_hit_toward_its_asymptote():
    """Slowly increasing resilience: the hit decays to severity*(1-rec)."""
    rec, tau_rec = 0.5, 10.0
    d = _disease(rec=rec, tau_rec=tau_rec, tau=0.5, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    sev = np.asarray(disease_severity(d))
    peak = np.asarray(disease_k_fraction(d, float(onset_t.max() + 2)))
    late = np.asarray(disease_k_fraction(d, float(onset_t.max() + 8 * tau_rec)))
    assert (late < peak).all()
    np.testing.assert_allclose(late, sev * (1.0 - rec), rtol=1e-3)


def test_no_recovery_holds_the_hit_at_peak_severity():
    d = _disease(rec=0.0, tau=0.5, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    sev = np.asarray(disease_severity(d))
    held = np.asarray(disease_k_fraction(d, float(onset_t.max() + 40)))
    np.testing.assert_allclose(held, sev, rtol=1e-4)


def test_recovery_is_monotone_after_arrival():
    d = _disease(rec=0.6, tau_rec=8.0, tau=0.5)
    onset_t = float(np.asarray(disease_onset_timestep(d)).max())
    series = [np.asarray(disease_k_fraction(d, onset_t + a)).mean()
              for a in range(3, 40, 3)]
    assert np.all(np.diff(series) < 1e-9)


# ------------------------------------------------------- the late-arrival term

def test_late_arrival_coefficient_makes_later_populations_milder():
    """The western-genetic-diversity hypothesis, as one coefficient."""
    d = _disease(b_late=-0.5, w_scale=0.0)
    sev = np.asarray(disease_severity(d))
    onset = np.asarray(d["onset"])
    order = np.argsort(onset)
    assert sev[order][-1] < sev[order][0]
    # ... and the sign is free, so a positive coefficient reverses it.
    d_pos = _disease(b_late=+0.5, w_scale=0.0)
    sev_pos = np.asarray(disease_severity(d_pos))[order]
    assert sev_pos[-1] > sev_pos[0]


# --------------------------------------------------------- basis conditioning

@pytest.mark.parametrize("n_freq", [2, 3, 4, 6])
def test_spatial_basis_is_land_centered(n_freq):
    """Land-centering is what splits "level" from "regional deviation".

    Without it the field's coefficients can shift the continental severity, which
    then trades off against alpha_k and makes both unreportable.
    """
    basis = _basis(n_freq)
    assert basis.shape == ((n_freq + 1) ** 2 - 1, N_LAND)
    assert np.abs(basis.mean(axis=1)).max() < 1e-5


def test_basis_field_cannot_shift_the_continental_level():
    """A pure field perturbation leaves the mean logit unchanged."""
    base = _disease(w_scale=0.0)
    perturbed = _disease(w_scale=1.0)
    logit = lambda s: np.log(s / (1.0 - s))
    m0 = logit(np.asarray(disease_severity(base))).mean()
    m1 = logit(np.asarray(disease_severity(perturbed))).mean()
    assert abs(m1 - m0) < 1e-4


# ------------------------------------------------------------ prior predictive

def _prior_draws(n=4000, prior_scale=1.0):
    sev_b = _basis(4)
    out = []
    for i in range(n):
        with seed(rng_seed=i):
            tr = trace(sample_priors).get_trace(
                prior_scale=prior_scale, M_features=4, time=124,
                N_sev_basis=sev_b.shape[0], N_lag_basis=sev_b.shape[0])
        out.append({k: np.asarray(v["value"]) for k, v in tr.items()
                    if k.startswith("disease")})
    return out


def test_prior_median_peak_severity_is_about_fifty_percent():
    """The config claims a prior median of ~50% removal. Verify it.

    This is the check that the stated belief is what the model actually encodes --
    the numbers in age_model_config.json's _disease_prior_comment are otherwise
    unverified prose.
    """
    draws = _prior_draws(2000)
    # Severity at a cell with average arrival year and zero field contribution.
    sev = np.array([1.0 / (1.0 + np.exp(-d["disease_mu_sev"])) for d in draws])
    assert 0.45 < np.median(sev) < 0.55
    lo, hi = np.percentile(sev, [5, 95])
    assert 0.25 < lo < 0.40, f"5th percentile {lo:.2f} outside the documented ~0.31"
    assert 0.60 < hi < 0.78, f"95th percentile {hi:.2f} outside the documented ~0.69"


def test_prior_allows_but_does_not_assume_recovery():
    draws = _prior_draws(2000)
    rec = np.array([d["disease_rec"] for d in draws])
    assert 0.25 < np.median(rec) < 0.50          # config claims ~38%
    assert (rec > 0.05).mean() > 0.8             # recovery is plausible
    assert (rec < 0.2).mean() > 0.1              # "barely any" stays cheap
    tau_rec = np.array([d["disease_tau_rec"] for d in draws])
    assert 9.0 < np.median(tau_rec) < 16.0       # config claims ~12 yr
    assert (tau_rec > 0).all()


def test_prior_scale_tightens_the_disease_priors():
    """Continuation must actually constrain this term early in the fit."""
    wide = np.array([d["disease_mu_sev"] for d in _prior_draws(400, prior_scale=1.0)])
    tight = np.array([d["disease_mu_sev"] for d in _prior_draws(400, prior_scale=0.1)])
    assert tight.std() < 0.3 * wide.std()


def test_k_field_is_rescaled_never_annihilated_end_to_end():
    """The whole field function: K vs the same run with the disease switched off.

    This is the regression test for the actual production failure. The previous
    formulation subtracted an unbounded penalty from K's pre-softplus argument, and
    because fitted K sits where softplus is effectively exp(), that multiplied K by
    exp(-d) -- driving eastern carrying capacity to ~3% of baseline. Here the same
    comparison must show a bounded rescale, exact equality before the epizootic
    window, and no cell ever raised above baseline.
    """
    from src.model.age_fields import project_and_scatter_age_structured as project

    T, M, K_kern = 124, 3, 2
    rng = np.random.default_rng(0)
    Z = jnp.array(rng.normal(size=(T, N_LAND, M)))
    Zd = jnp.array(rng.normal(size=(T, N_LAND, K_kern, M)))
    idx = jnp.arange(N_LAND)
    rates = (jnp.ones(M) * .1, jnp.ones(M) * .1, 0.5, 0.3, -0.5, 0.4, 2.0, 0.3, 0.5, 0.3)

    d = _disease(mu_sev=0.3, b_late=-0.3, rec=0.4, w_scale=0.3)
    K_dis = np.asarray(project(T, 40, 60, idx, idx, Z, Zd, DIS_T0, d, *rates)[3])
    # mu_sev very negative => severity ~ 0 => K_base
    K_off = np.asarray(project(T, 40, 60, idx, idx, Z, Zd, DIS_T0,
                               dict(d, mu_sev=-30.0), *rates)[3])

    assert np.array_equal(K_dis[:DIS_T0], K_off[:DIS_T0]), \
        "pre-1993 K must be untouched EXACTLY -- this is what pins alpha_k"
    assert (K_dis <= K_off + 1e-9).all(), "the disease term must never raise K"
    ratio = K_dis / K_off
    assert ratio.min() > 0.05, f"K annihilated (min ratio {ratio.min():.3f})"
    assert (K_dis > 0).all() and np.isfinite(K_dis).all()


def test_prior_predictive_fraction_never_annihilates_capacity():
    """End-to-end: no prior draw can remove more than ~90% of K."""
    sev_b, lag_b = _basis(4), _basis(4)
    worst = 0.0
    for i, p in enumerate(_prior_draws(300)):
        d = _disease()
        d.update({"mu_sev": p["disease_mu_sev"], "b_late": p["disease_b_late"],
                  "w_sev": jnp.array(p["disease_w_sev"]),
                  "lag0": p["disease_lag0"], "w_lag": jnp.array(p["disease_w_lag"]),
                  "tau": p["disease_tau"], "rec": p["disease_rec"],
                  "tau_rec": p["disease_tau_rec"],
                  "sev_basis": jnp.array(sev_b), "lag_basis": jnp.array(lag_b)})
        worst = max(worst, float(np.asarray(disease_k_fraction(d, 123.0)).max()))
    assert worst < 0.95, f"prior admits a {worst:.0%} capacity wipeout"
