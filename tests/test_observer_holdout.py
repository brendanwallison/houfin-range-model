"""map_diagnostics._observer_holdout_scores: integrated held-out scores for the observer x route effect."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "viz"))
from map_diagnostics import _observer_holdout_scores  # noqa: E402


def _case(sigma, seed=0, n=4000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 400, n)
    fy = (rng.random(n) < .15).astype(int)
    holdout = idx >= 300                       # observer-routes 300+ never fitted -> unseen
    holdout |= rng.random(n) < .1              # plus some held-out obs of seen observer-routes
    eps = rng.normal(size=400)
    eps[300:] = 0.0                            # unseen: prior mean at the mode
    eta = -.2
    mu0 = np.exp(rng.normal(1, .5, n))
    mu = mu0 * np.exp(eta * fy + sigma * eps[idx])
    phi = 2.0
    y = rng.negative_binomial(phi, phi / (phi + mu0 * np.exp(eta * fy + sigma * rng.normal(size=400)[idx]))).astype(float)
    data = {"obs_obsroute_idx": idx, "obs_first_year": fy, "obs_holdout": holdout,
            "obs_time_indices": np.zeros(n, int)}
    sim = {"expected_obs": mu, "expected_obs_before_observer": mu0,
           "latents": {"observer_route_sigma": sigma, "observer_first_year_eta": eta}}
    return data, sim, y, phi, holdout, mu, mu0, eta, fy


def test_sigma_zero_equals_plugin():
    data, sim, y, phi, h, mu, *_ = _case(1e-12)
    out = _observer_holdout_scores(data, sim, y, phi, None, h, np.zeros_like(h))
    plug = nbinom.logpmf(y[h], phi, phi / (phi + mu[h])).mean()
    assert abs(out["temporal"]["marginal_mean_lpd"] - plug) < 1e-9
    assert abs(out["temporal"]["conditional_mean_lpd"] - plug) < 1e-9


def test_marginal_matches_monte_carlo():
    data, sim, y, phi, h, mu, mu0, eta, fy = _case(.97)
    out = _observer_holdout_scores(data, sim, y, phi, None, h, np.zeros_like(h))
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)[:, None]
    base = (mu0 * np.exp(eta * fy))[h]
    lp = nbinom.logpmf(y[h], phi, phi / (phi + base * np.exp(.97 * x)))
    mc = (np.log(np.exp(lp - lp.max(0)).mean(0)) + lp.max(0)).mean()
    assert abs(out["temporal"]["marginal_mean_lpd"] - mc) < 5e-3
    unseen = data["obs_obsroute_idx"][h] >= 300
    assert abs(out["temporal"]["unseen_obsroute_share"] - unseen.mean()) < 1e-12
