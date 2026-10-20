"""population_model.st_correction (E067): zero weights reproduce the base model; nonzero weights move the fields,
and the path habitat receives the correction's path image (not the local correction)."""
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
        np.savez(path, dens=np.asarray(tr["simulated_density"]["value"]), Q=np.asarray(tr["Q_flat"]["value"]),
                 **{"lat_" + k: v for k, v in lat.items()})
        return
    z = np.load(path)
    sub = {k[4:]: z[k] for k in z.files if k.startswith("lat_")}
    nb, nt = data["st_S"].shape[-1], data["st_g"].shape[0]
    sub["st_correction_w"] = np.zeros((nb, nt))
    tr = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data=sub)).get_trace(data=data, prior_scale=1.0)
    np.testing.assert_allclose(np.asarray(tr["simulated_density"]["value"]), z["dens"], rtol=1e-9, atol=1e-12)
    sub["st_correction_w"] = np.full((nb, nt), 0.3)
    tr2 = handlers.trace(handlers.substitute(handlers.seed(build_model_2d, 3), data=sub)).get_trace(data=data, prior_scale=1.0)
    assert not np.allclose(np.asarray(tr2["simulated_density"]["value"]), z["dens"])
    assert not np.allclose(np.asarray(tr2["Q_flat"]["value"]), z["Q"])     # path habitat feels the correction
    print("ST_OK")


def test_st_correction_nests_and_acts():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "base.npz")
        env = dict(os.environ, AGE_MODEL_CONFIG=os.path.join(OV, "map_run62b_numerics.json"))
        r = subprocess.run([sys.executable, __file__, "base", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        env["AGE_MODEL_CONFIG"] = os.path.join(OV, "map_run67a_st_correction.json")
        r = subprocess.run([sys.executable, __file__, "st", path], cwd=REPO, env=env, capture_output=True, text=True)
        assert "ST_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _child(sys.argv[1], sys.argv[2])
