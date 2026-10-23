"""readout.within_tied: beta_dev = c * beta_level. c = 1 must reproduce the unsplit readout exactly (Z_ref cancels)."""
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
    from src.model.age_priors import build_model_2d
    jax.config.update("jax_enable_x64", True)
    data = {k: (jax.numpy.asarray(v, dtype=jax.numpy.float64) if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in synthetic_inputs().items()}
    if mode == "base":
        tr = handlers.trace(handlers.seed(build_model_2d, 7)).get_trace(data=data, prior_scale=1.0)
        lat = {k: np.asarray(v["value"]) for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
        np.savez(path, dens=np.asarray(tr["simulated_density"]["value"]), **{"lat_" + k: v for k, v in lat.items()})
        return
    z = np.load(path)
    sub = {k[4:]: z[k] for k in z.files if k.startswith("lat_")}
    tr = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 7), data=sub)).get_trace(data=data, prior_scale=1.0)
    np.testing.assert_allclose(np.asarray(tr["simulated_density"]["value"]), z["dens"], rtol=1e-8, atol=1e-12)
    print("TIED_OK")


def _cfg(td, readout):
    p = os.path.join(td, f"c{len(os.listdir(td))}.json")
    with open(p, "w") as f:
        json.dump({"_extends": os.path.join(OV, "map_run69a_b1_nofloor.json"),
                   "population_model": {"readout": readout, "k_trend": {"enabled": False}}}, f)
    return p


def test_tied_c1_equals_unsplit():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "base.npz")
        env = dict(os.environ, AGE_MODEL_CONFIG=_cfg(td, {"split": False}))
        r = subprocess.run([sys.executable, __file__, "base", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        env["AGE_MODEL_CONFIG"] = _cfg(td, {"split": True, "ref_window": [1996, 2023], "rank": 24,
                                            "within_tied": True, "within_ratio_fixed": 1.0})
        r = subprocess.run([sys.executable, __file__, "tied", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert "TIED_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]




def _corr_child():
    import jax
    from numpyro import handlers
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_full_model_smoke import synthetic_inputs
    from src.model.age_priors import build_model_2d
    jax.config.update("jax_enable_x64", True)
    data = {k: (jax.numpy.asarray(v, dtype=jax.numpy.float64) if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in synthetic_inputs().items()}
    tr = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data={"within_level_corr": np.ones(4)})
                        ).get_trace(data=data, prior_scale=1.0)
    np.testing.assert_allclose(np.asarray(tr["w_env_within"]["value"]), 0.14 * np.asarray(tr["w_env"]["value"]),
                               rtol=1e-9, atol=1e-12)
    tr0 = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data={"within_level_corr": np.zeros(4)})
                         ).get_trace(data=data, prior_scale=1.0)
    assert not np.allclose(np.asarray(tr0["w_env_within"]["value"]), 0.14 * np.asarray(tr0["w_env"]["value"]))
    print("CORR_OK")


def test_correlated_r1_equals_tied():
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, AGE_MODEL_CONFIG=_cfg(td, {"split": True, "ref_window": [1996, 2023], "rank": 24,
                                                          "within_ratio_fixed": 0.14, "within_corr": {"a": 8, "b": 2}}))
        r = subprocess.run([sys.executable, __file__, "corr", "-"], cwd=REPO, env=env, capture_output=True, text=True)
        assert "CORR_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _corr_child() if sys.argv[1] == "corr" else _child(sys.argv[1], sys.argv[2])
