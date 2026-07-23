import numpy as np

from src.model.checkpoints import auto_delta_params_to_latents
from src.model.age_run_map import _prior_scale_for_step


def test_auto_delta_parameter_names_map_exactly_to_sample_sites():
    params = {
        "alpha_a_auto_loc": np.array(1.0),
        "inv_eta_auto_loc": np.zeros(10),
    }
    out = auto_delta_params_to_latents(params)
    assert set(out) == {"alpha_a", "inv_eta"}


def test_map_prior_continuation_uses_absolute_steps():
    assert _prior_scale_for_step(0) == 0.1
    assert _prior_scale_for_step(299) == 0.1
    assert _prior_scale_for_step(300) == 0.5
    assert _prior_scale_for_step(600) == 1.0
    assert _prior_scale_for_step(50_000) == 1.0
