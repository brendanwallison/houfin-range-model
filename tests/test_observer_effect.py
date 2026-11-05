"""BBS observer-route random effect + first-year effect (observation.observer_effect; bbs.py ingest).

The likelihood is exercised through the FULL build_model_2d on test_full_model_smoke's synthetic
inputs (switched on by patching age_priors._OBSERVER_SPEC, which is read at call time), plus a toy
NB model for the float64 finite-difference gradient check.
"""
import importlib.util
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density, potential_energy

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
from src.model import age_priors as AP  # noqa: E402

# The commit before the observer effect existed: "switch off" must reproduce its log density.
PRE_CHANGE_COMMIT = "ac521d8"


# ----------------------------------------------------------------- ingest (bbs.py)

def _toy_obs():
    # US route 1 (observers 7 then 9), US route 2 (observer 7), a route-year with no Weather
    # observer row, and a Mexico route (tier 1, no ObsN).
    rows = [(840, 2, 1, 1990, 0), (840, 2, 1, 1991, 0), (840, 2, 1, 1995, 0), (840, 2, 1, 1996, 0),
            (840, 2, 2, 1993, 0), (840, 2, 2, 1994, 0),
            (840, 2, 3, 2000, 0),
            (484, 5, 1, 2010, 1), (484, 5, 1, 2011, 1)]
    return pd.DataFrame(rows, columns=["CountryNum", "StateNum", "Route", "Year", "quality_tier"])


def _toy_runs():
    from src.data.preprocess.bbs import first_year_flags
    runs = pd.DataFrame([(840, 2, 1, 1990, 7), (840, 2, 1, 1991, 7), (840, 2, 1, 1995, 9),
                         (840, 2, 1, 1996, 9), (840, 2, 2, 1993, 7), (840, 2, 2, 1994, 7)],
                        columns=["CountryNum", "StateNum", "Route", "Year", "ObsN"])
    return first_year_flags(runs)


def test_observer_route_indices_toy_table():
    from src.data.preprocess.bbs import observer_route_indices
    obs = _toy_obs()
    r, o, fy, info = observer_route_indices(obs, _toy_runs())
    assert len(r) == len(o) == len(fy) == len(obs)
    assert sorted(set(r)) == [0, 1, 2, 3]                       # 3 US routes + 1 Mexico route
    assert r[0] == r[1] == r[2] == r[3] and r[4] == r[5] and r[7] == r[8]
    # observer x route: (1,7), (1,9), (2,7) distinct -- observer 7 on route 2 is a NEW id
    assert o[0] == o[1] and o[2] == o[3] and o[0] != o[2] and o[4] == o[5] and o[4] != o[0]
    assert o[7] == o[8]                                         # Mexico: one id per route
    assert (o == o[6]).sum() == 1                               # unmatched: own singleton id
    assert sorted(set(o)) == list(range(5))
    assert list(fy) == [1, 0, 1, 0, 1, 0, 0, 0, 0]              # first year per observer x route
    assert info["n_unmatched_observer"] == 1 and info["n_mexico"] == 2


def test_observer_route_indices_is_order_aligned():
    """Shuffling the observations permutes the outputs identically (alignment, not re-sorting)."""
    from src.data.preprocess.bbs import observer_route_indices
    obs = _toy_obs()
    perm = np.random.default_rng(0).permutation(len(obs))
    r, o, fy, _ = observer_route_indices(obs, _toy_runs())
    r2, o2, fy2, _ = observer_route_indices(obs.iloc[perm], _toy_runs())
    np.testing.assert_array_equal(fy[perm], fy2)
    np.testing.assert_array_equal(r[perm], r2)
    np.testing.assert_array_equal(o[perm], o2)


def test_dense_ids_keeps_pseudo_rows():
    from src.data.combine.model_inputs import dense_ids
    np.testing.assert_array_equal(dense_ids(np.array([-1, -1, 5, 9, 5, 12])), [-1, -1, 0, 1, 0, 2])


def test_rebuilt_npz_alignment():
    path = os.path.expandvars("${HOUFIN_DATA}/bbs_2026_release/bbs_data_obsroute_for_python.npz")
    if not os.path.exists(path):
        pytest.skip("rebuilt npz not present")
    z = np.load(path)
    n, n_tot = int(z["N_pseudo"]), len(z["observed_results"])
    for k in ("obs_route_idx", "obs_obsroute_idx", "obs_first_year"):
        assert len(z[k]) == n_tot
    assert (z["obs_route_idx"][:n] == -1).all() and (z["obs_obsroute_idx"][:n] == -1).all()
    assert (z["obs_first_year"][:n] == 0).all()
    o, q, fy = z["obs_obsroute_idx"][n:], z["obs_quality"][n:], z["obs_first_year"][n:]
    assert o.min() == 0 and np.unique(o).size == o.max() + 1
    assert fy[q == 1].sum() == 0                                         # Mexico: never first-year
    assert fy[q == 0].sum() == np.unique(o[q == 0]).size                 # one first year per obs-route


# ----------------------------------------------------------------- full model

def _inputs(n_or=25, seed=0):
    from test_full_model_smoke import synthetic_inputs
    data = dict(synthetic_inputs())
    rng = np.random.default_rng(seed)
    nobs = int(np.asarray(data["observed_results"]).shape[0])
    o = np.where(rng.random(nobs) < 0.2, -1, rng.integers(0, n_or, nobs))
    o[np.flatnonzero(o >= 0)[:n_or]] = np.arange(n_or)                 # every id used
    data["obs_obsroute_idx"] = jnp.asarray(o)
    data["obs_first_year"] = jnp.asarray(np.where(o >= 0, rng.integers(0, 2, nobs), 0))
    data["n_obsroute"] = n_or
    return data


def _point(model, data):
    """One prior draw of every latent site (constrained values), to evaluate log densities at."""
    with handlers.seed(rng_seed=3):
        tr = handlers.trace(model).get_trace(data, prior_scale=0.1)
    return {k: v["value"] for k, v in tr.items()
            if v["type"] == "sample" and not v["is_observed"]}


@pytest.fixture
def observer_on(monkeypatch):
    def _set(**kw):
        monkeypatch.setattr(AP, "_OBSERVER_SPEC", {"enabled": True, **kw})
    return _set


def _pre_change_module():
    try:
        src = subprocess.run(["git", "show", f"{PRE_CHANGE_COMMIT}:src/model/age_priors.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"commit {PRE_CHANGE_COMMIT} not available")
    path = os.path.join(os.environ.get("PYTEST_TMPDIR", "/tmp"), "age_priors_pre_observer.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("age_priors_pre_observer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_switch_off_equals_pre_change_model_exactly():
    old = _pre_change_module()
    data = _inputs()
    params = _point(AP.build_model_2d, data)
    assert not any(k.startswith("observer_") for k in params)           # shipped config: off
    lp_new, _ = log_density(AP.build_model_2d, (data,), {"prior_scale": 0.1}, params)
    lp_old, _ = log_density(old.build_model_2d, (data,), {"prior_scale": 0.1}, params)
    assert float(lp_new) == float(lp_old)


def test_sigma_zero_eta_zero_equals_off(observer_on):
    data = _inputs()
    params = _point(AP.build_model_2d, data)
    lp_off, tr_off = log_density(AP.build_model_2d, (data,), {"prior_scale": 0.1}, params)
    observer_on(sigma_u=0.0)
    lp_on, tr_on = log_density(AP.build_model_2d, (data,), {"prior_scale": 0.1},
                               {**params, "observer_first_year_eta": jnp.float32(0.0)})
    assert "observer_route_eps" not in tr_on                             # sigma 0: no eps site
    np.testing.assert_array_equal(np.asarray(tr_on["expected_obs"]["value"]),
                                  np.asarray(tr_off["expected_obs"]["value"]))
    eta_lp = float(dist.Normal(0.0, 1.0).log_prob(0.0))
    np.testing.assert_allclose(float(lp_on), float(lp_off) + eta_lp, rtol=1e-6)


def test_effect_multiplies_real_obs_and_leaves_pseudo_zeros(observer_on):
    observer_on(sigma_u=0.7)
    data = _inputs()
    params = _point(AP.build_model_2d, data)
    eps, eta = np.asarray(params["observer_route_eps"]), float(params["observer_first_year_eta"])
    assert abs(eps.sum()) < 1e-4                                          # zero-sum draw
    _, tr = log_density(AP.build_model_2d, (data,), {"prior_scale": 0.1}, params)
    before = np.asarray(tr["expected_obs_before_observer"]["value"])
    mean = np.asarray(tr["expected_obs"]["value"])
    o, fy = np.asarray(data["obs_obsroute_idx"]), np.asarray(data["obs_first_year"])
    real = o >= 0
    np.testing.assert_array_equal(mean[~real], before[~real])
    np.testing.assert_allclose(mean[real], before[real] * np.exp(eta * fy[real] + 0.7 * eps[o[real]]),
                               rtol=1e-5)


def test_fixed_sigma_is_not_an_optimizable_coordinate(observer_on):
    """AutoDelta (age_run_map) must not see a scale site in fixed mode; prior mode adds one."""
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    data = _inputs()
    for mode, expect in (("fixed", False), ("prior", True)):
        observer_on(sigma_u=0.5, sigma_mode=mode)
        svi = SVI(AP.build_model_2d, AutoDelta(AP.build_model_2d), numpyro.optim.Adam(1e-3), Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(0), data=data, prior_scale=0.1)
        state, loss = svi.update(state, data=data, prior_scale=0.1)
        params = svi.get_params(state)
        assert np.isfinite(float(loss))
        assert ("observer_route_sigma_auto_loc" in params) is expect, sorted(params)
        assert "observer_route_eps_auto_loc" in params and "observer_first_year_eta_auto_loc" in params


def test_full_model_gradient_is_finite(observer_on):
    observer_on(sigma_u=0.8)
    data = _inputs()
    params = _point(AP.build_model_2d, data)
    g = jax.grad(lambda p: log_density(AP.build_model_2d, (data,), {"prior_scale": 0.1}, p)[0])(params)
    for k in ("observer_first_year_eta", "observer_route_eps", "concentration"):
        assert np.all(np.isfinite(np.asarray(g[k]))), k
    assert float(jnp.abs(g["observer_route_eps"]).sum()) > 0


def test_missing_arrays_raise(observer_on):
    observer_on(sigma_u=0.5)
    data = _inputs()
    data.pop("obs_obsroute_idx")
    with pytest.raises(ValueError, match="obs_obsroute_idx"):
        _point(AP.build_model_2d, data)


def test_spec_validation_and_shipped_default():
    assert AP.observer_effect_spec()["enabled"] is False or os.environ.get("AGE_MODEL_CONFIG")
    with pytest.raises(ValueError):
        AP.observer_effect_spec({"enabled": True, "sigma_mode": "joint"})
    with pytest.raises(ValueError):
        AP.observer_effect_spec({"enabled": True, "sigma_u": -1.0})


# ----------------------------------------------------------------- centering, gradient

def test_zero_sum_keeps_median_multiplier():
    """Zero-sum eps: the geometric mean of exp(u) is exactly 1 and the median ~1, so the median-anchored
    capacity gauge (docs/PRIORS.md) does not move. A -sigma^2/2 correction would put it at exp(-sigma^2/2)."""
    sigma, n = 0.9, 4001
    eps = dist.ZeroSumNormal(1.0, event_shape=(n,)).sample(jax.random.PRNGKey(0))
    u = sigma * np.asarray(eps, np.float64)
    assert abs(u.mean()) < 1e-5
    assert abs(np.median(np.exp(u)) - 1.0) < 0.05
    assert np.exp(-sigma ** 2 / 2) < 0.7


def _toy_nb(dens, data, spec):
    mu = dens * jnp.exp(AP.observer_log_offset(data, spec))
    conc = numpyro.sample("concentration", dist.Exponential(1.0))
    numpyro.sample("obs", dist.NegativeBinomial2(mean=mu, concentration=conc), obs=data["y"])


def test_gradient_matches_finite_differences_float64():
    with jax.enable_x64(True):
        rng = np.random.default_rng(1)
        n, n_or = 60, 7
        o = np.concatenate([np.full(6, -1), rng.integers(0, n_or, n - 6)])
        o[6:6 + n_or] = np.arange(n_or)
        data = {"obs_obsroute_idx": jnp.asarray(o), "n_obsroute": n_or,
                "obs_first_year": jnp.asarray(np.where(o >= 0, rng.integers(0, 2, n), 0)),
                "y": jnp.asarray(rng.poisson(6, n).astype(np.float64))}
        dens = jnp.asarray(rng.uniform(1.0, 10.0, n))
        for spec in ({"enabled": True, "sigma_u": 0.6}, {"enabled": True, "sigma_mode": "prior"}):
            spec = AP.observer_effect_spec(spec)
            z = {"concentration": jnp.float64(0.4), "observer_first_year_eta": jnp.float64(-0.15),
                 "observer_route_eps": jnp.asarray(rng.normal(size=n_or - 1))}
            if spec["sigma_mode"] == "prior":
                z["observer_route_sigma"] = jnp.float64(-0.5)
            pe = jax.jit(lambda zz: potential_energy(_toy_nb, (dens, data, spec), {}, zz))
            g = jax.grad(pe)(z)
            h = 1e-6
            for k, v in z.items():
                flat = np.atleast_1d(np.asarray(v, np.float64))
                for j in range(flat.size):
                    up, dn = flat.copy(), flat.copy()
                    up[j] += h
                    dn[j] -= h
                    fd = (float(pe({**z, k: jnp.asarray(up.reshape(np.shape(v)))}))
                          - float(pe({**z, k: jnp.asarray(dn.reshape(np.shape(v)))}))) / (2 * h)
                    np.testing.assert_allclose(np.atleast_1d(np.asarray(g[k]))[j], fd, rtol=1e-5, atol=1e-6)
