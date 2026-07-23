import numpy as np

from scripts.viz.map_diagnostics import local_growth_lambda
from scripts.viz.map_diagnostics import plot_fit_diagnostics, plot_modern_niche, plot_modern_rate_maps


def test_local_growth_lambda_matches_forward_census_order():
    """At the correct F(lambda=1), the local projection has unit growth."""
    sa = np.array([0.60, 0.72])
    sj = np.array([0.40, 0.55])
    f_unit = (1.0 - sa) / (sa * sj)
    assert np.allclose(local_growth_lambda(sa, sj, f_unit), 1.0)
    assert np.allclose(local_growth_lambda(sa, sj, np.zeros_like(sa)), sa)


def test_local_growth_lambda_increases_with_fecundity():
    lam_low = local_growth_lambda(.65, .45, .5)
    lam_high = local_growth_lambda(.65, .45, 2.0)
    assert lam_high > lam_low > .65


def test_core_map_diagnostic_figures_render(tmp_path):
    years = np.array([2000, 2005, 2010])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    sa = np.array([[.5, .6, .7, .65], [.55, .62, .72, .7], [.6, .64, .75, .72]])
    sj = np.full_like(sa, .5)
    fmax = np.array([[.6, .9, 1.2, 1.0], [.8, 1.0, 1.3, 1.1], [1.0, 1.1, 1.4, 1.2]])
    sim = {"Sa_flat": sa, "Sj_flat": sj, "Fmax_flat": fmax, "K_flat": np.full_like(sa, 2.0),
           "expected_obs": np.array([1.0, 2.0, 3.0, 4.0])}
    lam = local_growth_lambda(sa, sj, fmax)
    _, _, transition = plot_modern_niche(lam, years, rows, cols, (2, 2), tmp_path / "niche.png", 2)
    assert np.isfinite(transition).all()
    plot_modern_rate_maps(sim, years, rows, cols, (2, 2), tmp_path / "rates.png", 2)
    fit = plot_fit_diagnostics(sim, {"observed_results": np.array([1., 2., 3., 4.]),
                                     "obs_time_indices": np.array([0, 1, 1, 2])}, years,
                               tmp_path / "fit.png")
    assert fit["n_observations"] == 4
    assert all((tmp_path / name).stat().st_size > 10_000 for name in ("niche.png", "rates.png", "fit.png"))
