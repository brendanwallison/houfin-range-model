"""population_model.readout.split: with the within-block weights equal to the level block's, the split
readout reproduces the single-amplitude model exactly (z_ref.b + (z - z_ref).b = z.b)."""
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OV = os.path.join(REPO, "config", "overlays")


def _child(mode, path):
    import jax
    import numpyro
    from numpyro import handlers
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_full_model_smoke import synthetic_inputs
    from src.model.age_priors import build_model_2d
    jax.config.update("jax_enable_x64", True)    # test the algebra, not float32 rounding amplified by the dynamics
    data = {k: (jax.numpy.asarray(v, dtype=jax.numpy.float64) if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in synthetic_inputs().items()}
    if mode == "single":
        tr = handlers.trace(handlers.seed(build_model_2d, 7)).get_trace(data=data, prior_scale=1.0)
        lat = {k: np.asarray(v["value"]) for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
        np.savez(path, dens=np.asarray(tr["simulated_density"]["value"]), **{"lat_" + k: v for k, v in lat.items()})
    else:
        z = np.load(path)
        sub = {k[4:]: z[k] for k in z.files if k.startswith("lat_")}
        sub["w_scale_within_raw"] = sub["w_scale_raw"]
        sub["manifold_factor_within"] = sub["manifold_factor"]
        sub["manifold_idio_within"] = sub["manifold_idio"]
        tr = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 7), data=sub)).get_trace(
            data=data, prior_scale=1.0)
        np.testing.assert_allclose(np.asarray(tr["simulated_density"]["value"]), z["dens"], rtol=1e-9, atol=1e-12)
        print("NESTS_OK")


def test_split_with_equal_weights_nests_single_amplitude():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "single.npz")
        env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(OV, "map_run62b_numerics.json"))
        r = subprocess.run([sys.executable, __file__, "single", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        env["AGE_MODEL_CONFIG"] = os.path.join(OV, "map_run66a_readout.json")
        r = subprocess.run([sys.executable, __file__, "split", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert "NESTS_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _child(sys.argv[1], sys.argv[2])
