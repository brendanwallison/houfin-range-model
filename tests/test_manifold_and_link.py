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
