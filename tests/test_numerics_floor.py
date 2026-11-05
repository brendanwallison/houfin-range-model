"""population_model.numerics: zero FFT padding + sub-resolution floor removal keeps gradients finite.

The model has true zeros everywhere off the occupied range once the legacy 1e-9 pad and the float32
round-off floor are gone; this runs 50 MAP steps under JAX_DEBUG_NANS in both precisions to check
nothing divides by or logs a zero density.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OVERLAY = os.path.join(REPO, "config", "overlays", "map_run62a_numerics.json")


def _run_steps():
    import jax
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_full_model_smoke import synthetic_inputs
    from src.model import age_forward
    from src.model.age_priors import build_model_2d
    assert age_forward.FFT_PAD_VALUE == 0.0 and age_forward.FFT_FLOOR_ULPS == 10.0
    data = synthetic_inputs()
    svi = SVI(build_model_2d, AutoDelta(build_model_2d), numpyro.optim.Adam(1e-2), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(3), data=data, prior_scale=1.0)
    for _ in range(50):
        state, loss = svi.update(state, data=data, prior_scale=1.0)
    assert np.isfinite(float(loss))
    print("NUMERICS_OK", float(loss))


@pytest.mark.parametrize("x64", ["0", "1"])
def test_zero_floor_gradients_finite(x64):
    env = dict(os.environ, JAX_DEBUG_NANS="True", AGE_MODEL_CONFIG=OVERLAY, JAX_ENABLE_X64=x64)
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert "NUMERICS_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _run_steps()
