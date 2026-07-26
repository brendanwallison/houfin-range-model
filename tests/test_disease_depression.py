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
# A dense-population K_base (well above the k_half default) so the density term is
# near saturation and the tests below exercise the gate/recovery/field logic rather
# than the density ramp. Density dependence gets its own tests.
K_BASE = jnp.full(N_LAND, 4.0)


def _basis(n_freq=3, n_land=N_LAND, seed_val=0):
    rng = np.random.default_rng(seed_val)
    rows = rng.integers(0, 40, n_land)
    cols = rng.integers(0, 60, n_land)
    return generate_spatial_basis(40, 60, rows, cols, n_freq)


def _disease(mu_sev=0.0, b_late=0.0, rec=0.4, tau_rec=12.0, tau=1.5,
             lag0=0.0, w_scale=0.0, seed_val=0, ceiling=1.0,
             k_half=0.4, hill_n=1.5):
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
        # Density dependence: k_half in DENSITY units here (tests pass K_base
        # directly). hill_n is the fitted steepness.
        "k_half": k_half, "hill_n": hill_n,
        # ceiling=1.0 in most tests so the sigmoid's own bound is what gets
        # exercised; the configured production ceiling is asserted separately.
        "ceiling": ceiling,
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
        frac = np.asarray(disease_k_fraction(d, float(t), K_BASE))
        assert np.isfinite(frac).all()
        assert (frac >= 0.0).all(), "the effect must never ADD capacity"
        assert (frac < 1.0).all(), "the effect must never remove ALL capacity"


def test_severity_is_a_fraction_regardless_of_the_field():
    d = _disease(mu_sev=4.0, b_late=2.0, w_scale=5.0)
    sev = np.asarray(disease_severity(d, K_BASE))
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
    frac = np.asarray(disease_k_fraction(d, float(onset_t.min() - 15.0), K_BASE))
    assert (frac < 1e-4).all()


def test_effect_switches_on_across_the_front():
    d = _disease(tau=1.5, rec=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    early = np.asarray(disease_k_fraction(d, float(onset_t.mean() - 6), K_BASE))
    late = np.asarray(disease_k_fraction(d, float(onset_t.mean() + 6), K_BASE))
    assert (late > early).all()


def test_onset_ordering_follows_the_arrival_map():
    """At a fixed year, later-arriving cells must be less affected."""
    d = _disease(tau=1.0, rec=0.0, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    frac = np.asarray(disease_k_fraction(d, float(np.median(onset_t)), K_BASE))
    order = np.argsort(onset_t)
    assert np.all(np.diff(frac[order]) <= 1e-6)


def test_lag_shifts_the_front_in_years():
    d0 = _disease(lag0=0.0, tau=0.75, rec=0.0)
    d5 = _disease(lag0=5.0, tau=0.75, rec=0.0)
    t = float(np.asarray(disease_onset_timestep(d0)).mean() + 2)
    assert (np.asarray(disease_k_fraction(d5, t, K_BASE))
            < np.asarray(disease_k_fraction(d0, t, K_BASE))).all()


# -------------------------------------------------------------- the recovery

def test_recovery_relaxes_the_hit_toward_its_asymptote():
    """Slowly increasing resilience: the hit decays to severity*(1-rec)."""
    rec, tau_rec = 0.5, 10.0
    d = _disease(rec=rec, tau_rec=tau_rec, tau=0.5, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    sev = np.asarray(disease_severity(d, K_BASE))
    peak = np.asarray(disease_k_fraction(d, float(onset_t.max() + 2), K_BASE))
    late = np.asarray(disease_k_fraction(d, float(onset_t.max() + 8 * tau_rec), K_BASE))
    assert (late < peak).all()
    np.testing.assert_allclose(late, sev * (1.0 - rec), rtol=1e-3)


def test_no_recovery_holds_the_hit_at_peak_severity():
    d = _disease(rec=0.0, tau=0.5, w_scale=0.0)
    onset_t = np.asarray(disease_onset_timestep(d))
    sev = np.asarray(disease_severity(d, K_BASE))
    held = np.asarray(disease_k_fraction(d, float(onset_t.max() + 40), K_BASE))
    np.testing.assert_allclose(held, sev, rtol=1e-4)


def test_recovery_is_monotone_after_arrival():
    d = _disease(rec=0.6, tau_rec=8.0, tau=0.5)
    onset_t = float(np.asarray(disease_onset_timestep(d)).max())
    series = [np.asarray(disease_k_fraction(d, onset_t + a, K_BASE)).mean()
              for a in range(3, 40, 3)]
    assert np.all(np.diff(series) < 1e-9)


# ------------------------------------------------------- the late-arrival term

def test_late_arrival_coefficient_makes_later_populations_milder():
    """The western-genetic-diversity hypothesis, as one coefficient."""
    d = _disease(b_late=-0.5, w_scale=0.0)
    sev = np.asarray(disease_severity(d, K_BASE))
    onset = np.asarray(d["onset"])
    order = np.argsort(onset)
    assert sev[order][-1] < sev[order][0]
    # ... and the sign is free, so a positive coefficient reverses it.
    d_pos = _disease(b_late=+0.5, w_scale=0.0)
    sev_pos = np.asarray(disease_severity(d_pos, K_BASE))[order]
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
    """A pure field perturbation leaves the mean logit unchanged.

    Land-centering guarantees this, and it is what keeps disease_mu_sev the sole
    owner of the continental severity level. It also catches a subtler bug: any
    CLIP applied to the modifier (an earlier version used min(1, 2*sigmoid)) breaks
    it, because clipping is asymmetric and a zero-mean logit perturbation then
    shifts the mean downward.
    """
    base, perturbed = _disease(w_scale=0.0), _disease(w_scale=1.0)
    d = _disease(w_scale=0.0)
    hill = float(K_BASE[0]) ** d["hill_n"] / (
        float(K_BASE[0]) ** d["hill_n"] + d["k_half"] ** d["hill_n"])
    def mod_logit(dd):
        sev = np.asarray(disease_severity(dd, K_BASE)) / (dd["ceiling"] * hill)
        return np.log(sev / (1.0 - sev))
    assert abs(mod_logit(perturbed).mean() - mod_logit(base).mean()) < 1e-4


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


def test_prior_severity_in_dense_populations():
    """What the prior actually says about severity where the epizootic can spread.

    Severity is ``ceiling * sigmoid(mu + ...) * hill(K_base)``. In a DENSE population
    the Hill factor approaches 1, so the prior median severity is
    ``ceiling * median(modifier)``. The config claims ~40% with a 90% CI of roughly
    [31%, 46%] -- verify it, because those numbers are otherwise unverified prose.
    Deliberately below the documented ~50-60% eastern decline: the ceiling is a
    strict bound needed for monotonicity, not the expected value.
    """
    ceiling = float(load_age_model_config()["population_model"]
                    ["disease_prior"]["severity_ceiling"])
    draws = _prior_draws(2000)
    sev = np.array([ceiling / (1.0 + np.exp(-d["disease_mu_sev"])) for d in draws])
    assert 0.36 < np.median(sev) < 0.44, f"median {np.median(sev):.3f}"
    lo, hi = np.percentile(sev, [5, 95])
    assert 0.27 < lo < 0.35, f"5th pct {lo:.3f} outside the documented ~0.31"
    assert 0.42 < hi < 0.49, f"95th pct {hi:.3f} outside the documented ~0.46"
    assert (sev < ceiling).all(), "the ceiling must be a strict bound"


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
    # beta_s, beta_r, beta_k (capacity has its own manifold), then the continental
    # K trend basis + weights, then the alpha/gamma pairs.
    k_trend = jnp.array(np.cos(np.pi * np.linspace(0, 1, T))[None, :] - 0.0)
    rates = (jnp.ones(M) * .1, jnp.ones(M) * .1, jnp.ones(M) * .1,
             k_trend, jnp.zeros(1),
             0.5, 0.3, -0.5, 0.4, 2.0, 0.3,
             -2.128295, 0.3)   # alpha_k (density space), gamma_k

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
                  "sev_basis": jnp.array(sev_b), "lag_basis": jnp.array(lag_b),
                  "ceiling": 1.0})
        worst = max(worst, float(np.asarray(disease_k_fraction(d, 123.0, K_BASE)).max()))
    assert worst < 0.95, f"prior admits a {worst:.0%} capacity wipeout"


# ------------------------------------------------------- density dependence

def test_severity_vanishes_in_sparse_populations():
    """Epidemics need hosts: severity -> 0 as capacity -> 0.

    This is the property that stops the Allee effect from turning a small ABSOLUTE
    capacity loss into a local extinction in exactly the sparse cells where a
    density-dependent pathogen would not have established in the first place.
    """
    d = _disease(k_half=0.4, hill_n=1.5, w_scale=0.0)
    for K in (1e-4, 1e-3, 1e-2):
        sev = np.asarray(disease_severity(d, jnp.full(N_LAND, K)))
        assert (sev < 0.02).all(), f"K={K} still gets severity {sev.max():.3f}"


def test_severity_saturates_in_dense_populations():
    d = _disease(k_half=0.4, hill_n=1.5, w_scale=0.0, mu_sev=10.0)
    sev = np.asarray(disease_severity(d, jnp.full(N_LAND, 100.0)))
    np.testing.assert_allclose(sev, d["ceiling"], rtol=2e-2)


def test_severity_is_monotone_increasing_in_capacity():
    d = _disease(k_half=0.4, hill_n=2.0, w_scale=0.0)
    Ks = np.geomspace(1e-3, 100.0, 60)
    sev = np.array([float(np.asarray(disease_severity(d, jnp.full(1, k)))[0]) for k in Ks])
    assert np.all(np.diff(sev) > -1e-9)


@pytest.mark.parametrize("hill_n", [0.5, 1.0, 1.5, 3.0, 5.1])
def test_realized_capacity_stays_monotone_in_baseline_capacity(hill_n):
    """K = K_base*(1 - severity(K_base)) must be invertible.

    If it is not, two habitat qualities map to the SAME realized capacity and the
    covariates driving K become unidentifiable. At the worst point (K_base = k_half)
    the slope is 1 - ceiling*(1/2 + n/4), so the requirement is n < 4/ceiling - 2.
    age_priors derives the steepness bound from the ceiling for exactly this reason;
    5.9 here is just inside the limit at ceiling 0.5.
    """
    ceiling = float(load_age_model_config()["population_model"]
                    ["disease_prior"]["severity_ceiling"])
    # 0.85 margin: the closed form (n < 4/ceiling - 2) is optimistic because the
    # true worst point sits just below k_half. age_priors applies the same margin.
    assert hill_n <= 0.85 * (4.0 / ceiling - 2.0), "test case violates the bound"
    n_cells = 4000
    Kb = np.linspace(1e-3, 20.0, n_cells)
    d = _disease(k_half=0.4, hill_n=hill_n, w_scale=0.0, mu_sev=20.0, ceiling=ceiling)
    # Size the (unused, zero-weight) field to the sweep so shapes broadcast.
    d["sev_basis"] = jnp.zeros((1, n_cells))
    d["w_sev"] = jnp.zeros(1)
    d["onset_decades"] = jnp.zeros(n_cells)
    K = jnp.asarray(Kb) * (1.0 - disease_severity(d, jnp.asarray(Kb)))
    assert np.all(np.diff(np.asarray(K)) > 0), \
        f"K(K_base) is non-monotone at n={hill_n}, ceiling={ceiling}"


def test_steepness_bound_is_derived_from_the_ceiling():
    """The fitted steepness can never reach a non-monotone value.

    Structural, not a prior hope: n = n_min + (n_max - n_min)*sigmoid(raw) with
    n_max = 4/ceiling - 2. Raising the ceiling automatically tightens the allowed
    steepness, so the two cannot drift into an unidentifiable pair.
    """
    ceiling = float(load_age_model_config()["population_model"]
                    ["disease_prior"]["severity_ceiling"])
    n_max = 0.85 * (4.0 / ceiling - 2.0)
    ns = []
    for i in range(400):
        with seed(rng_seed=i):
            tr = trace(sample_priors).get_trace(
                prior_scale=1.0, M_features=4, time=124,
                N_sev_basis=8, N_lag_basis=8)
        ns.append(float(np.asarray(tr["disease_hill_n"]["value"])))
    ns = np.array(ns)
    assert (ns > 0.5).all() and (ns < n_max).all(), \
        f"steepness escaped (0.5, {n_max}): [{ns.min():.2f}, {ns.max():.2f}]"
    assert 1.0 < np.median(ns) < 2.2, f"prior median steepness {np.median(ns):.2f}"


# ------------------------------------------------------------ gauge invariance

def test_likelihood_scale_is_a_free_gauge():
    """Absolute-scale priors must be declared in route counts, not density units.

    The likelihood is exactly invariant under (K -> cK, initpop -> c*initpop,
    inv_pop -> c*inv_pop, allee_gamma -> allee_gamma/c, pop_scalar -> pop_scalar/c),
    because K enters the forward sim only as N/K and the Allee term only as
    allee_gamma*N. So any prior stated in DENSITY units is silently gauge-dependent
    and changes meaning whenever the gauge changes -- which is how alpha_k came to
    assert a capacity of ~205 route counts without anyone noticing.

    This test pins the invariance for the one boundary the model owns: a quantity
    declared in counts must convert to the same OBSERVABLE regardless of the gauge.
    """
    from src.model import age_priors as AP

    counts = 2.1
    real_gauge = AP._POP_SCALAR
    try:
        observables = []
        for gauge in (20.0, 60.0, 210.0):
            AP._POP_SCALAR = gauge
            density = float(AP.counts_to_relative(counts))
            observables.append(density * gauge)      # back to route counts
        np.testing.assert_allclose(observables, counts, rtol=1e-12)
    finally:
        AP._POP_SCALAR = real_gauge


def test_capacity_level_prior_is_declared_in_route_counts():
    """The level prior must sit near the typical OCCUPIED cell (~2 counts).

    Empirically (BBS >2000, per-cell means, occupied cells): median 2.12 counts,
    geometric mean 1.87. Empty cells are expected to be empty because lambda < 1,
    not because capacity is low there.

    Asserted on the POST-TRANSFORMATION level so the test survives a change of
    link: under exp the raw value is a log-median, under softplus a near-linear
    intercept, and only the transformed quantity is comparable across the two.
    """
    cfg = load_age_model_config()["population_model"]["capacity_level_prior"]
    sp = lambda x: np.log1p(np.exp(-abs(x))) + max(x, 0.0)
    if "target_level_mean_route_counts" in cfg:       # softplus link, density space
        level = float(cfg["target_level_mean_route_counts"])
    else:                                             # exp / LogNormal link
        level = float(cfg["median_route_counts"])
    assert 1.0 < level < 8.0, f"level prior {level:.2f} counts is implausible"
