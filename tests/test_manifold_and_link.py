"""The run_19 structural switches: exchangeable manifold prior, K link, dispersal_random.

* the exchangeable prior gives EVERY pair of fields correlation rho and each field variance
  w_scale^2 -- the uncentered-Ruzicka GP contract the rank-2 form also satisfies;
* the switches resolve from an overlay and change exactly the sites they should;
* the capacity level's prior MEDIAN stays at the measured 2.6183 route counts under both links;
* the visualization math applies the same K link as the model.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pytest.importorskip("numpyro")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpyro                                                    # noqa: E402
from numpyro import handlers                                      # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_exchangeable_prior_has_one_correlation_and_the_contract_variance(monkeypatch):
    import src.model.age_priors as ap
    monkeypatch.setitem(ap._MANIFOLD_PRIOR, "coupling_target", 0.85)
    monkeypatch.setitem(ap._MANIFOLD_PRIOR, "coupling_logit_scale", 0.15)
    w_scale = jnp.array([0.7, 2.0, 1.3, 0.4])
    rho = 0.6                                       # conditioned, away from the prior target
    M = 40000
    model = handlers.condition(
        lambda: ap._exchangeable_w_env(M, w_scale, 1.0),
        {"manifold_coupling_logit": jnp.log(rho / (1 - rho))})
    w_env, corr = handlers.seed(model, 0)()
    w = np.asarray(w_env)
    assert w.shape == (M, 4)
    emp = np.corrcoef(w.T)
    off = emp[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off, rho, atol=0.02)
    np.testing.assert_allclose(w.var(0), np.asarray(w_scale) ** 2, rtol=0.03)
    np.testing.assert_allclose(np.asarray(corr)[~np.eye(4, dtype=bool)], rho, rtol=1e-6)


def test_k_link_switch():
    import src.model.age_priors as ap
    x = jnp.array([-3.0, 0.0, 2.0])
    np.testing.assert_allclose(ap.k_link(x, "exp"), np.exp(x), rtol=1e-6)
    np.testing.assert_allclose(ap.k_link(x, "softplus"), np.log1p(np.exp(x)), rtol=1e-6)


def test_viz_capacity_uses_the_model_link(monkeypatch):
    import src.model.age_priors as ap
    from src.vis.age_model_math import rates_from_manifolds
    p = {"alpha_a": 0.0, "gamma_a": 1.0, "alpha_j": 0.0, "gamma_j": 1.0,
         "alpha_f": 0.0, "gamma_f": 1.0, "alpha_k": -2.0, "gamma_k": 1.0}
    h = np.array([3.0])
    monkeypatch.setattr(ap, "_K_LINK", "exp")
    assert rates_from_manifolds(p, h, h, h, h)["K"][0] == pytest.approx(np.exp(1.0))
    monkeypatch.setattr(ap, "_K_LINK", "softplus")
    assert rates_from_manifolds(p, h, h, h, h)["K"][0] == pytest.approx(np.log1p(np.exp(1.0)))


_PROBE = """
import json, src.model.age_priors as a
from numpyro import handlers
tr = handlers.trace(handlers.seed(a.sample_priors, 0)).get_trace(
    prior_scale=1.0, M_features=24, time=10, N_sev_basis=24, N_lag_basis=24)
s = sorted(k for k, v in tr.items() if v['type'] == 'sample')
print(json.dumps({"form": a._MANIFOLD_FORM, "link": a._K_LINK, "disp": a._DISPERSAL_RANDOM,
                  "level_counts": float(a.k_link(a._ALPHA_K_LOC)) * a._POP_SCALAR,
                  "sites": s, "factor_shape": list(tr['manifold_factor']['value'].shape)}))
"""


def _resolve(overlay):
    env = dict(os.environ)
    env.pop("AGE_MODEL_CONFIG", None)
    if overlay:
        env["AGE_MODEL_CONFIG"] = os.path.join(REPO, "config", "overlays", overlay)
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("overlay,form,link,disp", [
    (None, "rank2", "softplus", True),
    ("map_run19_exchangeable.json", "exchangeable", "softplus", False),
    ("map_run19_exchangeable_explink.json", "exchangeable", "exp", False),
])
def test_switches_resolve_from_overlays(overlay, form, link, disp):
    r = _resolve(overlay)
    assert (r["form"], r["link"], r["disp"]) == (form, link, disp)
    assert r["level_counts"] == pytest.approx(2.6183, rel=1e-6)
    assert ("dispersal_random" in r["sites"]) == disp
    assert ("manifold_angle" in r["sites"]) == (form == "rank2")
    assert ("manifold_coupling_logit" in r["sites"]) == (form == "exchangeable")
    assert r["factor_shape"] == ([24, 2] if form == "rank2" else [24, 1])


def test_crowding_capacity_is_growth_derived_and_bounded():
    """K = kappa * (R0-1)_+: zero for sinks, the Beverton-Holt term reduces to 1/kappa, and the
    habitat modifier can shift K by at most e^B."""
    import src.model.age_priors as ap
    Sa = jnp.array([0.5, 0.5, 0.5, 0.5]); Sj = jnp.array([0.4, 0.4, 0.4, 0.4])
    Fmax = jnp.array([1.0, 2.5, 5.0, 10.0])              # R0 = Fmax*Sa*Sj/(1-Sa) = 0.4, 1, 2, 4
    R0 = np.asarray(Fmax * Sa * Sj / (1 - Sa))
    K = np.asarray(ap.crowding_capacity(0.0, jnp.zeros(4), 0.0, Sa, Sj, Fmax, bound=0.0, softness=0.02))
    assert K[0] < 1e-6 and K[1] == pytest.approx(0.02 * np.log(2), rel=1e-3)   # sink; R0 = 1
    np.testing.assert_allclose(K[2:], R0[2:] - 1, rtol=1e-4)                  # kappa0 = 1
    c = np.maximum(R0 - 1, 0)
    np.testing.assert_allclose(c[2:] / K[2:], 1.0, rtol=1e-4)                  # c/K = 1/kappa
    B = np.log(4)
    Kmod = np.asarray(ap.crowding_capacity(0.0, jnp.full(4, 50.0), 0.0, Sa, Sj, Fmax, bound=B, softness=0.02))
    np.testing.assert_allclose(Kmod[2:] / K[2:], 4.0, rtol=1e-6)               # saturates at e^B


@pytest.mark.parametrize("overlay,bound", [("map_run20a_crowding.json", 0.0),
                                           ("map_run20b_crowding_modifier.json", 1.3863)])
def test_crowding_overlays_resolve(overlay, bound):
    probe = """
import json, src.model.age_priors as a
print(json.dumps({"form": a._K_FORM, "bound": a._CROWDING["modifier_bound"],
  "level": float(__import__("jax").numpy.exp(a._ALPHA_K_LOC)) * (a._CROWDING["reference_R0"] - 1) * a._POP_SCALAR,
  "rho": a._MANIFOLD_PRIOR["coupling_target"]}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", overlay))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert r["form"] == "crowding" and r["bound"] == pytest.approx(bound) and r["rho"] == 0.6
    assert r["level"] == pytest.approx(2.6183, rel=1e-6)


def test_single_field_has_one_pattern_and_no_private_parts(monkeypatch):
    import src.model.age_priors as ap
    monkeypatch.setitem(ap._MANIFOLD_PRIOR, "fixed_coupling", 1.0)
    w_scale = jnp.array([0.7, 2.0, 1.3, 0.4])
    tr = handlers.trace(handlers.seed(lambda: ap._exchangeable_w_env(30, w_scale, 1.0), 0)).get_trace()
    sites = {k for k, v in tr.items() if v["type"] == "sample"}
    assert sites == {"manifold_factor"}
    w = np.asarray(tr["w_env"]["value"])
    np.testing.assert_allclose(w / np.asarray(w_scale), np.repeat(w[:, :1] / 0.7, 4, axis=1), rtol=1e-6)


def test_occupied_habitat_center_counts_each_occupied_cell_year_once():
    from src.model.data_loading import occupied_habitat_center
    Z = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)      # (time, N_land, M)
    meta = {"Z_gathered": Z, "Ny": 2, "Nx": 2,
            "land_rows": np.array([0, 0, 1]), "land_cols": np.array([0, 1, 1]),
            # two routes in (t=0, land 1), one zero count, one occupied (t=1, land 2)
            "obs_time_indices": np.array([0, 0, 0, 1]), "obs_rows": np.array([0, 0, 0, 1]),
            "obs_cols": np.array([1, 1, 0, 1]), "observed_results": np.array([3, 1, 0, 2])}
    np.testing.assert_allclose(occupied_habitat_center(meta), (Z[0, 1] + Z[1, 2]) / 2)


def test_viz_rates_at_the_centre_equal_the_intercepts():
    """With centred fields, every rate evaluated at z_center is link(alpha): the
    visualization's intercept shift must reproduce the model's centring exactly."""
    from src.vis.age_model_math import demographic_params, rates_from_manifolds
    rng = np.random.default_rng(0)
    w_env = rng.normal(size=(5, 4)); z_c = rng.normal(size=5)
    lat = {"w_env": w_env, "alpha_a": 0.3, "alpha_j": -0.4, "alpha_f": 1.2, "alpha_k": -2.0,
           "gamma_a": 1.0, "gamma_j": 1.0, "gamma_f": 1.0, "gamma_k": 1.0,
           "habitat_center_offsets": z_c @ w_env}
    p = demographic_params(lat)
    H = z_c @ w_env                                   # the UNCENTERED z.beta callers compute
    r = rates_from_manifolds(p, H[0], H[3], H[1], H[2])
    sig = lambda x: 1 / (1 + np.exp(-x))              # noqa: E731
    assert r["Sa"] == pytest.approx(sig(0.3)) and r["Sj"] == pytest.approx(sig(-0.4))
    assert r["Fmax"] == pytest.approx(np.log1p(np.exp(1.2)))
    assert r["K"] == pytest.approx(np.log1p(np.exp(-2.0)))


def test_single_field_overlay_resolves():
    probe = """
import json, src.model.age_priors as a
from numpyro import handlers
tr = handlers.trace(handlers.seed(a.sample_priors, 0)).get_trace(
    prior_scale=1.0, M_features=24, time=10, N_sev_basis=24, N_lag_basis=24)
print(json.dumps({"centering": a._HABITAT_CENTERING, "fixed": a._MANIFOLD_PRIOR.get("fixed_coupling"),
  "level": float(a.k_link(a._ALPHA_K_LOC)) * a._POP_SCALAR, "scale": a._CAPACITY_LEVEL["alpha_k_scale"],
  "sites": sorted(k for k, v in tr.items() if v["type"] == "sample")}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", "map_run21_single_field.json"))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert r["centering"] == "occupied" and r["fixed"] == 1.0 and r["scale"] == 1.0
    assert r["level"] == pytest.approx(3.0, rel=1e-6)
    assert "manifold_idio" not in r["sites"] and "manifold_coupling_logit" not in r["sites"]


def test_relative_allee_is_invariant_to_capacity_scale(monkeypatch):
    """Under allee_prior.form = relative, scaling K and N together scales the next year's
    population by the same factor -- capacity cannot change whether a cell persists. The
    absolute form is NOT invariant (the property that made Allee-dead cells possible)."""
    from src.model.age_forward import reproduction_age_structured as rep
    import src.model.age_priors as ap
    from src.vis.age_model_math import allee_viability
    Na, Nj_s, Nj_a = jnp.array([0.02]), jnp.array([0.01]), jnp.array([0.005])
    args = lambda s: (Na * s, Nj_s * s, Nj_a * s, 0.6, 0.4, 3.0, 0.5 * s, 1.0)  # noqa: E731
    g = 14.0
    for rel, invariant in ((True, True), (False, False)):
        a1, j1 = rep(*args(1.0), g, allee_relative=rel)
        a10, j10 = rep(*args(10.0), g, allee_relative=rel)
        assert bool(np.allclose(np.asarray(j10) / 10, np.asarray(j1), rtol=1e-5)) == invariant
    monkeypatch.setattr(ap, "ALLEE_RELATIVE", True)
    _, F_at_K_small, _, _ = ap.equilibrium_age_quantities(0.6, 0.4, 3.0, 0.01, g)
    _, F_at_K_big, _, _ = ap.equilibrium_age_quantities(0.6, 0.4, 3.0, 100.0, g)
    assert float(F_at_K_small) == pytest.approx(float(F_at_K_big))
    v_small = allee_viability(0.6, 0.4, 3.0, 0.001, g)
    v_big = allee_viability(0.6, 0.4, 3.0, 1000.0, g)
    assert np.asarray(v_small["viable"]).item() == np.asarray(v_big["viable"]).item()
    np.testing.assert_allclose(np.asarray(v_small["F_peak"]), np.asarray(v_big["F_peak"]), rtol=1e-6)


def test_relative_allee_overlay_resolves():
    probe = """
import json, math, src.model.age_priors as a
print(json.dumps({"form": a._ALLEE_FORM, "rel": a.ALLEE_RELATIVE,
  "n50_median": math.log1p(math.exp(a._ALLEE_PRIOR["n50_raw_loc"])),
  "fixed": a._MANIFOLD_PRIOR.get("fixed_coupling"), "centering": a._HABITAT_CENTERING}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", "map_run22_relative_allee.json"))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert r["form"] == "relative" and r["rel"] and r["fixed"] == 1.0 and r["centering"] == "occupied"
    assert r["n50_median"] == pytest.approx(0.05, rel=1e-3)


def test_relative_density_is_finite_in_value_and_gradient_at_zero_capacity():
    """Run_22's MAP died on NaN densities in the forward scan; N/K at K ~ 0 is the culprit
    the guard targets. Value and gradient must stay finite for K = 0, underflowed K and
    empty cells, and the ratio must be exact where K is ordinary."""
    from src.model.age_forward import relative_density, reproduction_age_structured as rep
    # last two: FFT round-off negatives at a small and an ordinary K (the run_22 overflow)
    N = jnp.array([0.0, 0.3, 0.3, 0.0, 0.2, -1e-9, -1e-9])
    K = jnp.array([0.0, 0.0, 1e-30, 1e-30, 0.5, 1e-12, 0.5])
    r = np.asarray(relative_density(N, K))
    assert np.all(np.isfinite(r)) and r[0] == 0 and r[3] == 0 and r[1] == r[2] == 1e4
    assert r[4] == pytest.approx(0.4) and r[5] == 0 and r[6] == 0
    def total(n, k):
        a, j = rep(n * 0.5, n * 0.3, n * 0.2, 0.6, 0.4, 3.0, k, 1.0, 14.0, allee_relative=True)
        return (a + j).sum()
    gN, gK = jax.grad(total, argnums=(0, 1))(N, K)
    assert np.all(np.isfinite(np.asarray(gN))) and np.all(np.isfinite(np.asarray(gK)))


def test_juvenile_ratio_keeps_juveniles_below_adults(monkeypatch):
    import src.model.age_priors as ap
    from src.vis.age_model_math import rates_from_manifolds
    p = {"alpha_a": 0.0, "gamma_a": 1.0, "alpha_j": 3.0, "gamma_j": 2.0,
         "alpha_f": 1.0, "gamma_f": 1.0, "alpha_k": -2.0, "gamma_k": 1.0}
    H = np.linspace(-3, 3, 13)
    monkeypatch.setattr(ap, "JUVENILE_RATIO", True)
    r = rates_from_manifolds(p, H, H, H, H)
    assert np.all((r["Sj"] > 0) & (r["Sj"] < r["Sa"]) & (r["Sa"] < 1))
    monkeypatch.setattr(ap, "JUVENILE_RATIO", False)
    assert np.any(rates_from_manifolds(p, H, H, H, H)["Sj"] > r["Sa"])   # the old form can cross


def test_run23a_overlay_resolves():
    overlay = "map_run23a_weak_allee_juv_ratio.json"
    probe = """
import json, math, src.model.age_priors as a
print(json.dumps({"juv": a.JUVENILE_RATIO, "n50": math.log1p(math.exp(a._ALLEE_PRIOR["n50_raw_loc"])),
  "n50_scale": a._ALLEE_PRIOR["n50_raw_scale"], "allee": a._ALLEE_FORM,
  "fixed": a._MANIFOLD_PRIOR.get("fixed_coupling"), "centering": a._HABITAT_CENTERING}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", overlay))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert r["juv"] and r["allee"] == "absolute" and r["n50_scale"] == 0.25
    assert r["n50"] == pytest.approx(0.03, rel=1e-3) and r["fixed"] == 1.0 and r["centering"] == "occupied"


@pytest.mark.parametrize("overlay,pinned", [
    ("map_run24a_pin_allee.json", {"n50_raw"}),
    ("map_run24b_pin_demography.json", {"alpha_a", "alpha_j", "alpha_f"}),
    ("map_run24c_pin_both.json", {"alpha_a", "alpha_j", "alpha_f", "n50_raw"}),
])
def test_pinned_sites_are_held_fixed_not_fitted(overlay, pinned):
    """pinned_sites are conditioned: observed in the trace at the pinned value, so AutoDelta
    never fits them; every other site stays latent."""
    probe = """
import json, numpy as np, src.model.age_priors as a
from numpyro import handlers
f = handlers.condition(a.sample_priors, data={k: np.asarray(v) for k, v in a.PINNED.items()})
tr = handlers.trace(handlers.seed(f, 0)).get_trace(prior_scale=1.0, M_features=24, time=10,
                                                   N_sev_basis=24, N_lag_basis=24)
print(json.dumps({"pinned": a.PINNED,
  "observed": sorted(k for k, v in tr.items() if v["type"] == "sample" and v["is_observed"]),
  "values": {k: float(tr[k]["value"]) for k in a.PINNED}}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", overlay))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert set(r["pinned"]) == pinned and set(r["observed"]) == pinned
    assert r["values"] == pytest.approx(r["pinned"])


def test_ideal_free_settlement_conserves_mass_and_follows_capacity():
    from src.model.age_forward import ideal_free_settle
    rng = np.random.default_rng(1)
    K = jnp.asarray(rng.uniform(0.0, 2.0, (7, 10)))
    K = K.at[0, :].set(0.0)                               # a row with no capacity (ocean)
    N = jnp.asarray(rng.uniform(0.0, 1.0, (7, 10)))
    for t in (0, 1):                                      # both block offsets
        (out,) = ideal_free_settle((N,), K, 3, t)
        out = np.asarray(out)
        assert out.sum() == pytest.approx(float(N.sum()), rel=1e-6)
        assert np.all(out >= 0)
    # within one interior block (offset 0: rows 3-5, cols 3-5), out is proportional to K
    (out,) = ideal_free_settle((N,), K, 3, 0)
    blk = (slice(3, 6), slice(3, 6))
    r = np.asarray(out)[blk] / np.asarray(K)[blk]
    np.testing.assert_allclose(r, r.mean(), rtol=1e-6)
    assert np.asarray(out)[blk].sum() == pytest.approx(float(N[blk].sum()), rel=1e-6)


def test_adult_departure_offset_only_moves_adults():
    from src.model.age_forward import dispersal_step_age_structured
    rng = np.random.default_rng(2)
    Ny, Nx, nk = 8, 8, 2
    N_a = jnp.asarray(rng.uniform(0.1, 1.0, (Ny, Nx)))
    N_j = jnp.asarray(rng.uniform(0.1, 1.0, (Ny, Nx)))
    K = jnp.ones((Ny, Nx))
    ker = jnp.fft.fft2(jnp.zeros((2 * Ny, 2 * Nx)).at[0, 1].set(1.0))    # shift one column
    args = (N_a, N_j, K, 0.5, -2.0, 0.8, jnp.ones((Ny, Nx)), jnp.ones((nk, Ny, Nx)),
            ker, jnp.stack([ker] * nk), jnp.ones((nk, Ny, Nx)))
    a0, js0, ja0 = dispersal_step_age_structured(*args)
    a1, js1, ja1 = dispersal_step_age_structured(*args, adult_logit_offset=0.0)
    np.testing.assert_allclose(a0, a1)                    # offset 0 == shared (default unchanged)
    a2, js2, ja2 = dispersal_step_age_structured(*args, adult_logit_offset=-3.0)
    np.testing.assert_allclose(js2, js0)                  # juveniles untouched
    np.testing.assert_allclose(ja2, ja0)
    # fewer adults leave: the adult field moves less under the shift
    assert np.abs(np.asarray(a2) - np.asarray(N_a)).sum() < np.abs(np.asarray(a0) - np.asarray(N_a)).sum()


def test_crowding_arm_is_off_below_capacity_and_pulses_above():
    from src.model.age_forward import departure_logit
    x = jnp.asarray([0.0, 0.5, 0.9, 1.0, 1.5, 2.0, 3.0])
    base = departure_logit(x, -4.5, -8.0, 0.8)
    np.testing.assert_allclose(departure_logit(x, -4.5, -8.0, 0.8, None), base)   # None = one slope
    np.testing.assert_allclose(departure_logit(x, -4.5, -8.0, 0.8, 0.0), base)    # beta 0 = one slope
    z = departure_logit(x, -4.5, -8.0, 0.8, 12.0, 5.0)
    gain = np.asarray(z - base)
    assert np.all(gain >= 0) and gain[0] < 0.05 and gain[1] < 0.2       # ~nothing well below K
    assert np.all(np.diff(gain) > 0)                                     # rises with crowding
    assert float(z[-1]) > float(z[3])                                    # U-shape: 3K departs more than K


def _rates(shape, rng, Sa=0.55, Sj=0.4, F=6.0):
    Sa = jnp.full(shape, Sa); Sj = jnp.full(shape, Sj); F = jnp.full(shape, F)
    c = jnp.maximum(F * Sa * Sj / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    return Sa, Sj, F, c


def test_fitness_settlement_nests_capacity_weighting_under_equal_rates():
    from src.model.age_forward import fitness_settle, ideal_free_settle
    rng = np.random.default_rng(3)
    K = jnp.asarray(rng.uniform(0.0, 2.0, (7, 10))).at[0, :].set(0.0)
    N1 = jnp.asarray(rng.uniform(0.0, 1.0, (7, 10)))
    N2 = jnp.asarray(rng.uniform(0.0, 1.0, (7, 10)))
    Sa, Sj, F, c = _rates(K.shape, rng)
    for t in (0, 1):
        a = fitness_settle((N1, N2), K, Sa, Sj, F, c, 3, t)
        b = ideal_free_settle((N1, N2), K, 3, t)
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-9)
        assert float(sum(x.sum() for x in a)) == pytest.approx(float(N1.sum() + N2.sum()), rel=1e-6)


def test_fitness_settlement_leaves_poor_cells_empty_until_good_ones_crowd():
    from src.model.age_forward import fitness_settle
    K = jnp.ones((3, 3))
    Sa, Sj, F, c = _rates(K.shape, None)
    F = F.at[1, 1].set(1.0)                         # lambda0 < 1 there: a sink
    c = jnp.maximum(F * Sa * Sj / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    N = jnp.full((3, 3), 8.0 / 9.0)                 # block total = K of the 8 good cells
    (out,) = fitness_settle((N,), K, Sa, Sj, F, c, 3, 0)
    out = np.asarray(out)
    assert out[1, 1] == pytest.approx(0.0, abs=1e-6)
    good = np.ones((3, 3), bool); good[1, 1] = False
    np.testing.assert_allclose(out[good], 1.0, rtol=1e-4)   # w* = 1: every viable cell at K
    # the sink's empty-cell lambda0 is ~0.82: at 1.2x the good cells still do better (lambda ~0.96)
    # and it stays empty; at 3x they are crowded down to ~0.79 and it takes the spillover
    (mild,) = fitness_settle((N * 1.2,), K, Sa, Sj, F, c, 3, 0)
    assert float(np.asarray(mild)[1, 1]) == pytest.approx(0.0, abs=1e-6)
    (crowded,) = fitness_settle((N * 3.0,), K, Sa, Sj, F, c, 3, 0)
    assert float(np.asarray(crowded)[1, 1]) > 0.1


def test_fitness_settlement_gradient_matches_finite_difference():
    from jax.experimental import enable_x64
    with enable_x64():
        _fitness_gradient_check()


def _fitness_gradient_check():
    from src.model.age_forward import fitness_settle
    rng = np.random.default_rng(4)
    K = jnp.asarray(rng.uniform(0.2, 2.0, (6, 6)), dtype=jnp.float64)
    N = jnp.asarray(rng.uniform(0.0, 1.0, (6, 6)), dtype=jnp.float64)
    Sa, Sj, _, _ = _rates(K.shape, rng)
    Sa, Sj = Sa.astype(jnp.float64), Sj.astype(jnp.float64)
    probe = jnp.asarray(rng.uniform(0.0, 1.0, (6, 6)), dtype=jnp.float64)
    F0 = jnp.asarray(rng.uniform(3.0, 9.0, (6, 6)), dtype=jnp.float64)

    def f(scale):
        F = F0 * scale
        c = jnp.maximum(F * Sa * Sj / (1.0 - Sa + 1e-6) - 1.0, 0.0)
        (out,) = fitness_settle((N,), K, Sa, Sj, F, c, 3, 0)
        return jnp.sum(out * probe)

    g = jax.grad(f)(jnp.float64(1.0))
    h = 1e-6
    fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
    assert np.isfinite(float(g))
    assert float(g) == pytest.approx(float(fd), rel=1e-3, abs=1e-7)


def test_transient_mask_drops_arrivals_and_keeps_left_censored_cells():
    from src.model.data_loading import transient_observation_weight
    years = np.arange(1966, 2000)
    # cell A: surveyed from 1966, first positive 1980 (an arrival); cell B: positive from 1967 (censored)
    t = np.concatenate([np.arange(34), np.arange(34)])
    rows = np.array([1] * 34 + [2] * 34); cols = np.zeros(68, int)
    y = np.concatenate([(years >= 1980) * 3.0, (years >= 1967) * 5.0])
    meta = {"years": years, "obs_time_indices": t, "obs_rows": rows, "obs_cols": cols,
            "observed_results": y, "obs_quality": np.zeros(68, int)}
    w = transient_observation_weight(meta, window=5)
    a, b = w[:34], w[34:]
    assert np.all(a[(years >= 1975) & (years <= 1985)] == 0) and np.all(a[(years < 1975) | (years > 1985)] == 1)
    assert np.all(b == 1)


def _subcell_case(seed=5, S=6, Ny=4, Nx=5, scale=1.0):
    rng = np.random.default_rng(seed)
    pools = tuple(jnp.asarray(rng.uniform(0.0, 0.3, (Ny, Nx)) * scale) for _ in range(3))
    Sa = jnp.asarray(rng.uniform(0.4, 0.6, (Ny, Nx))); Sj = Sa * 0.6
    F = jnp.asarray(rng.uniform(3.0, 9.0, (Ny, Nx))); K = jnp.asarray(rng.uniform(0.5, 2.0, (Ny, Nx)))
    u = jnp.asarray(np.sort(rng.normal(0, 1, (S, Ny, Nx)), axis=0))
    a = jnp.asarray([0.15, 0.225, 0.25, 0.2, 0.12, 0.055])
    return pools, Sa, Sj, F, K, u, a


def test_subcell_reproduction_nests_the_one_slice_model():
    from src.model.age_forward import reproduction_age_structured, subcell_reproduction
    pools, Sa, Sj, F, K, u, a = _subcell_case()
    c = jnp.maximum(F * Sa * Sj / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    ref = reproduction_age_structured(*pools, Sa, Sj, F, K, c, 3.0)
    for bk, df, uu in ((0.0, 0.0, u), (1.5, 0.8, jnp.zeros_like(u))):   # no tilt / homogeneous cell
        out = subcell_reproduction(pools, Sa, Sj, F, K, 3.0, uu, a, bk, df)
        for x, y in zip(out, ref):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-5, atol=1e-9)


def test_subcell_clustering_relieves_the_allee_cost_in_sparse_cells():
    from src.model.age_forward import reproduction_age_structured, subcell_reproduction
    pools, Sa, Sj, F, K, u, a = _subcell_case(scale=0.05)                # sparse cells
    c = jnp.maximum(F * Sa * Sj / (1.0 - Sa + 1e-6) - 1.0, 0.0)
    gamma = 20.0                                                         # strong mate-finding limit
    _, j_one = reproduction_age_structured(*pools, Sa, Sj, F, K, c, gamma)
    _, j_sub = subcell_reproduction(pools, Sa, Sj, F, K, gamma, u, a, 1.0, 0.5)
    assert float(j_sub.sum()) > 1.2 * float(j_one.sum())                 # clustered birds breed more


def test_subcell_gradient_matches_finite_difference():
    from jax.experimental import enable_x64
    from src.model.age_forward import subcell_reproduction
    with enable_x64():
        pools, Sa, Sj, F, K, u, a = _subcell_case()
        cast = lambda x: jnp.asarray(x, jnp.float64)
        pools = tuple(cast(p) for p in pools); Sa, Sj, F, K, u, a = map(cast, (Sa, Sj, F, K, u, a))

        def f(th):
            na, nj = subcell_reproduction(pools, Sa, Sj, F, K, 3.0, u, a, th[0], th[1])
            return jnp.sum(na) + 2.0 * jnp.sum(nj)

        th = jnp.asarray([0.7, 0.3])
        g = np.asarray(jax.grad(f)(th))
        h = 1e-6
        fd = [(f(th.at[i].add(h)) - f(th.at[i].add(-h))) / (2 * h) for i in range(2)]
        np.testing.assert_allclose(g, np.asarray(fd), rtol=1e-4, atol=1e-8)


def test_single_release_puts_all_founders_at_the_nearest_site_in_one_year():
    overlay = "map_run30_single_release.json"
    probe = r"""
import json, sys
sys.path.insert(0, ".")
from numpyro import handlers
from tests.test_full_model_smoke import synthetic_inputs
from src.model.age_priors import build_model_2d
d = synthetic_inputs()
tr = handlers.trace(handlers.seed(build_model_2d, 0)).get_trace(data=d, prior_scale=1.0)
inv = tr["inv_pop_relative"]["value"]
print(json.dumps({"shape": list(inv.shape), "nonzero": int((inv > 0).sum()),
                  "has_grid_site": "log_inv_pulse_fraction" in tr, "has_total": "log_inv_total" in tr}))
"""
    env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays", overlay))
    r = json.loads(subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip().splitlines()[-1])
    assert r["shape"][0] == 1 and r["nonzero"] == 1
    assert r["has_total"] and not r["has_grid_site"]


def test_disease_mortality_lowers_equilibrium_and_density_dependence_eases_it():
    """forward_sim with disease_m_amp: zero amplitude == no disease; endemic mortality settles the
    population BELOW K (c is from disease-free rates); mortality is weaker at low density."""
    from src.model.age_forward import forward_sim_age_structured
    import tests.test_full_model_smoke as sm
    d = sm.synthetic_inputs(T=40)
    n = int(d["N_land"]); T = int(d["time"])
    Sa = jnp.full((T, n), 0.55); Sj = jnp.full((T, n), 0.3); F = jnp.full((T, n), 5.0)
    K = jnp.full((T, n), 1.0); c = F * Sa * Sj / (1 - Sa + 1e-6) - 1.0
    Q = jnp.ones((T, n, d["juvenile_fft_kernel_stack"].shape[0]))

    def run(amp, seed_level=1.0, kh=0.5):
        tot, _, _ = forward_sim_age_structured(
            Sa, Sj, F, K, c, Q, d["land_rows"], d["land_cols"], d["land_mask"],
            d["adult_fft_kernel"], d["juvenile_fft_kernel_stack"],
            d["adult_edge_correction"], d["juvenile_edge_correction_stack"],
            d["land_mask"] * seed_level, jnp.zeros(T), jnp.zeros((2, 10)), T, d["inv_locations"], 12,
            -3.0, 0.0, 200.0, disease_m_amp=amp, disease_k_half=kh, disease_hill_n=2.0)  # Allee ~off
        return np.asarray(tot)

    base = run(None)
    np.testing.assert_allclose(run(jnp.zeros((T, n))), base, rtol=1e-6, atol=1e-9)
    amp = jnp.where(jnp.arange(T)[:, None] >= 20, 0.4, 0.0) * jnp.ones((T, n))
    sick = run(amp)
    land = np.asarray(d["land_mask"]) > 0
    assert sick[-1][land].mean() < 0.9 * base[-1][land].mean()        # equilibrium lowered
    assert np.allclose(sick[:20], base[:20])                           # nothing before onset
    # density dependence: a sparse population loses a smaller FRACTION in the first sick year
    lo_b, lo_s = run(None, 0.02, kh=0.5), run(amp, 0.02, kh=0.5)
    hi_frac = 1 - sick[20][land].sum() / base[20][land].sum()
    lo_frac = 1 - lo_s[20][land].sum() / lo_b[20][land].sum()
    assert lo_frac < hi_frac
