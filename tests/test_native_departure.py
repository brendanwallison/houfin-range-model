"""population_model.native_departure (E077): r = 0 reproduces the base model; r > 0 keeps more birds at home."""
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
        tr = handlers.trace(handlers.seed(build_model_2d, 3)).get_trace(data=data, prior_scale=1.0)
        lat = {k: np.asarray(v["value"]) for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
        np.savez(path, dens=np.asarray(tr["simulated_density"]["value"]), **{"lat_" + k: v for k, v in lat.items()})
        return
    z = np.load(path)
    sub = {k[4:]: z[k] for k in z.files if k.startswith("lat_")}
    run = lambda: np.asarray(handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data=sub)).get_trace(
        data=data, prior_scale=1.0)["simulated_density"]["value"])
    sub["native_departure_r"] = np.array(1e-12)
    np.testing.assert_allclose(run(), z["dens"], rtol=1e-8, atol=1e-12)
    sub["native_departure_r"] = np.array(0.9)
    d = run()
    assert not np.allclose(d, z["dens"])
    print("ND_OK")


def test_native_departure_nests_and_acts():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "base.npz")
        env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(OV, "map_run69a_b1_nofloor.json"))
        r = subprocess.run([sys.executable, __file__, "base", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        cfg = os.path.join(td, "nd.json")
        with open(cfg, "w") as f:
            json.dump({"_extends": os.path.join(OV, "map_run69a_b1_nofloor.json"),
                       "population_model": {"native_departure": {"enabled": True}}}, f)
        env["AGE_MODEL_CONFIG"] = cfg
        r = subprocess.run([sys.executable, __file__, "nd", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert "ND_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _child(sys.argv[1], sys.argv[2])
