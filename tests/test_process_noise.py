"""population_model.process_noise (E070) and the observation_mask holdouts.

Zero noise reproduces the base model; noise in the final year only rescales that year's abundance by exactly
exp(eps) (multiplicative, applied at the end of the step); the basis has unit land-RMS and is switched off from the
first held-out forecast year; the holdout mask is deterministic in its seed."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OV = os.path.join(REPO, "config", "overlays")


def _child(mode, path):
    import jax
    from numpyro import handlers
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_full_model_smoke import synthetic_inputs
    from src.model.age_priors import build_model_2d, _PN
    jax.config.update("jax_enable_x64", True)
    data = {k: (jax.numpy.asarray(v, dtype=jax.numpy.float64) if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in synthetic_inputs().items()}
    if mode == "base":
        tr = handlers.trace(handlers.seed(build_model_2d, 3)).get_trace(data=data, prior_scale=1.0)
        lat = {k: np.asarray(v["value"]) for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
        np.savez(path, dens=np.asarray(tr["simulated_density"]["value"]), **{"lat_" + k: v for k, v in lat.items()})
        return
    z = np.load(path)
    sub = {k[4:]: z[k] for k in z.files if k.startswith("lat_")}
    T, nb = data["pn_active"].shape[0], data["pn_S"].shape[-1]
    run = lambda: np.asarray(handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data=sub)).get_trace(
        data=data, prior_scale=1.0)["simulated_density"]["value"])
    sub["process_noise_u"] = np.zeros((T, nb))
    np.testing.assert_allclose(run(), z["dens"], rtol=1e-9, atol=1e-12)
    # final year only: that year's density is the base density times exp(eps), every earlier year untouched
    u = np.zeros((T, nb)); u[-1] = np.linspace(-2, 2, nb); sub["process_noise_u"] = u
    assert float(data["pn_active"][-1]) == 1.0
    eps = float(_PN["sigma"]) * np.asarray(data["pn_S"]) @ u[-1]
    d = run()
    rows, cols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
    np.testing.assert_allclose(d[:-1], z["dens"][:-1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(d[-1][rows, cols], z["dens"][-1][rows, cols] * np.exp(eps), rtol=1e-9, atol=1e-12)
    # an earlier pulse propagates forward
    u = np.zeros((T, nb)); u[T // 2, 0] = 2.0; sub["process_noise_u"] = u
    d = run()
    np.testing.assert_allclose(d[:T // 2], z["dens"][:T // 2], rtol=1e-9, atol=1e-12)
    assert not np.allclose(d[T // 2 + 1:], z["dens"][T // 2 + 1:])
    print("PN_OK")


def test_process_noise_nests_and_is_multiplicative():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "base.npz")
        env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(OV, "map_run69a_b1_nofloor.json"))
        r = subprocess.run([sys.executable, __file__, "base", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        # no holdout here: the synthetic years end in the holdout window, which would switch the final year off
        cfg = os.path.join(td, "pn.json")
        with open(cfg, "w") as f:
            json.dump({"_extends": os.path.join(OV, "map_run69a_b1_nofloor.json"),
                       "population_model": {"process_noise": {"enabled": True, "sigma": 0.05, "n_freq": 2,
                                                              "window_start": 1902}}}, f)
        env["AGE_MODEL_CONFIG"] = cfg
        r = subprocess.run([sys.executable, __file__, "pn", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert "PN_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


def _meta(seed=0):
    rng = np.random.default_rng(seed)
    Ny, Nx = 30, 40
    rows, cols = np.nonzero(np.ones((Ny, Nx)))
    years = np.arange(1960, 2026)
    n = 5000
    k = rng.integers(0, rows.size, n)
    return dict(Ny=Ny, Nx=Nx, land_rows=rows, land_cols=cols, years=years,
                obs_time_indices=rng.integers(6, years.size, n), obs_rows=rows[k], obs_cols=cols[k])


def test_holdout_mask_is_deterministic_and_sized():
    from src.model.data_loading import holdout_observation_mask
    m = _meta()
    spec = {"holdout_years": [2021, 2022], "holdout_blocks": {"block_cells": 5, "fraction": 0.2, "seed": 70}}
    a, b = holdout_observation_mask(m, spec), holdout_observation_mask(m, spec)
    assert np.array_equal(a, b)
    yrs = m["years"][m["obs_time_indices"]]
    assert a[np.isin(yrs, [2021, 2022])].all()
    sp = holdout_observation_mask(m, {"holdout_blocks": spec["holdout_blocks"]})
    assert 0.1 < sp.mean() < 0.3                                       # ~20% of 48 blocks
    blk = (m["obs_rows"] // 5) * 100000 + m["obs_cols"] // 5           # whole blocks, all years
    for bid in np.unique(blk[sp]):
        assert sp[blk == bid].all()


def test_process_noise_basis_scale_and_window(monkeypatch):
    import src.model.data_loading as dl
    m = _meta()
    monkeypatch.setattr(dl, "_observation_mask_spec", lambda: None)
    S, active = dl.process_noise_basis(m, {"n_freq": 3, "window_start": 1966})
    assert S.shape == (m["land_rows"].size, 15)
    np.testing.assert_allclose(np.mean(np.sum(S ** 2, axis=1)), 1.0, rtol=1e-10)   # sigma = land-RMS of eps
    assert active[m["years"] < 1966].sum() == 0 and active[m["years"] >= 1966].all()
    monkeypatch.setattr(dl, "_observation_mask_spec", lambda: {"holdout_years": [2021, 2023]})
    _, active = dl.process_noise_basis(m, {"n_freq": 3, "window_start": 1966})
    assert active[m["years"] >= 2021].sum() == 0 and active[(m["years"] >= 1966) & (m["years"] < 2021)].all()


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _child(sys.argv[1], sys.argv[2])


def test_holdout_score_matches_scipy_nb(monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts", "viz"))
    import map_diagnostics as md
    import src.model.data_loading as dl
    from scipy.stats import nbinom
    m = _meta(1)
    spec = {"holdout_years": [2024, 2025], "holdout_blocks": {"block_cells": 5, "fraction": 0.2, "seed": 1}}
    monkeypatch.setattr(dl, "_observation_mask_spec", lambda: spec)
    rng = np.random.default_rng(2)
    n = m["obs_rows"].size
    mu, phi = rng.gamma(2.0, 1.0, n), np.full(n, 1.7)
    y = rng.poisson(mu)
    data = dict(m, observed_results=y, obs_holdout=dl.holdout_observation_mask(m, spec))
    out = md._holdout_metrics(data, {"expected_obs": mu, "obs_concentration": phi})
    lp = nbinom.logpmf(y, phi, phi / (phi + mu))
    t = np.isin(m["years"][m["obs_time_indices"]], [2024, 2025])
    np.testing.assert_allclose(out["temporal"]["mean_lpd"], lp[t].mean(), rtol=1e-10)
    assert out["temporal"]["n"] + out["spatial"]["n"] + out["fitted"]["n"] == n


def test_holdout_score_matches_scipy_zinb(monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts", "viz"))
    import map_diagnostics as md
    import src.model.data_loading as dl
    from scipy.stats import nbinom
    m = _meta(3)
    spec = {"holdout_years": [2024, 2025]}
    monkeypatch.setattr(dl, "_observation_mask_spec", lambda: spec)
    rng = np.random.default_rng(4)
    n = m["obs_rows"].size
    mu, phi = rng.gamma(1.0, 1.0, n), np.full(n, 1.3)
    gate = np.exp(-mu / 0.4)
    y = rng.poisson(mu)
    data = dict(m, observed_results=y, obs_holdout=dl.holdout_observation_mask(m, spec))
    out = md._holdout_metrics(data, {"expected_obs": mu, "obs_concentration": phi, "obs_zero_gate": gate,
                                     "route_occupancy_log_m": np.log(0.4)})
    mo = mu / (1 - gate)
    lp = np.log1p(-gate) + nbinom.logpmf(y, phi, phi / (phi + mo))
    lp = np.where(y == 0, np.log(gate + np.exp(lp)), lp)
    t = np.isin(m["years"][m["obs_time_indices"]], [2024, 2025])
    np.testing.assert_allclose(out["temporal"]["mean_lpd"], lp[t].mean(), rtol=1e-10)
    np.testing.assert_allclose(out["route_occupancy_m"], 0.4)
