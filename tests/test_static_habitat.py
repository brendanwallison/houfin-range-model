"""E080: readout.within_ratio_fixed = 0 with k_trend disabled gives habitat fields constant in time."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OV = os.path.join(REPO, "config", "overlays")


def _child():
    import jax
    from numpyro import handlers
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_full_model_smoke import synthetic_inputs
    from src.model.age_priors import build_model_2d
    jax.config.update("jax_enable_x64", True)
    data = {k: (jax.numpy.asarray(v, dtype=jax.numpy.float64) if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in synthetic_inputs().items()}
    tr = handlers.trace(handlers.seed(build_model_2d, 5)).get_trace(data=data, prior_scale=1.0)
    for site in ("Fmax_flat", "Sa_flat", "Sj_flat"):
        v = np.asarray(tr[site]["value"])
        assert np.allclose(v, v[:1], rtol=1e-10, atol=1e-12), site      # no temporal change
    assert np.allclose(np.asarray(tr["w_k_trend"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["w_scale_within"]["value"]), 0.0)
    print("STATIC_OK")


def test_static_habitat_is_time_invariant():
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "static.json")
        with open(cfg, "w") as f:
            json.dump({"_extends": os.path.join(OV, "map_run69a_b1_nofloor.json"),
                       "population_model": {"readout": {"split": True, "ref_window": [1996, 2023], "rank": 24,
                                                        "within_ratio_fixed": 0.0},
                                            "k_trend": {"enabled": False}}}, f)
        env = dict(os.environ, AGE_MODEL_CONFIG=cfg)
        r = subprocess.run([sys.executable, __file__], cwd=REPO, env=env, capture_output=True, text=True)
        assert "STATIC_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _child()
