"""population_model.covariate_drift basis: held before the window, observation-weighted centring."""
import numpy as np

from src.model.data_loading import covariate_drift_basis


def test_basis_held_before_window_and_centred():
    years = np.arange(1902, 2024)
    rng = np.random.default_rng(0)
    obs_t = rng.integers(64, years.size, 5000)            # observations 1966+
    obs_t = np.concatenate([obs_t, np.full(300, 40)])     # plus pre-window (pseudo-zero) observations
    B = covariate_drift_basis(years, obs_t, 3, 1966)
    assert B.shape == (3, years.size)
    pre = years < 1966
    assert np.allclose(B[:, pre], B[:, [np.argmax(years == 1966)]])        # constant before the window
    w = np.bincount(obs_t, minlength=years.size).astype(float); w[pre] = 0.0; w /= w.sum()
    assert np.allclose(B @ w, 0.0, atol=1e-12)                             # centred on the observed window
    assert np.all(np.ptp(B[:, ~pre], axis=1) > 1.0)                        # and actually varies inside it
