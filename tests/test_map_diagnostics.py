from pathlib import Path

import numpy as np
import pytest

from scripts.viz.map_diagnostics import allee_viability, local_growth_lambda
from scripts.viz.map_diagnostics import plot_fit_diagnostics, plot_modern_niche, plot_modern_rate_maps
from scripts.viz.map_diagnostics import (
    modern_viability, plot_counterfactual_attribution,
    simulate_no_invasion_counterfactual,
)


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


def test_allee_viability_agrees_with_brute_force_scan():
    """The log-spaced scan must find the same fold as a dense linear scan."""
    sa, sj, fmax, g = 0.6, 0.4, 3.0, 69.31
    k = np.array([0.125, 0.025])          # 2.5 and 0.5 route counts at gauge 20
    got = allee_viability(sa, sj, fmax, k, g)
    c = fmax * sa * sj / (1.0 - sa + 1e-6) - 1.0
    for i, k_i in enumerate(k):
        n = np.linspace(1e-9, 6 * k_i, 400001)
        f_brute = fmax / (1.0 + c * n / k_i) * (1.0 - np.exp(-g * n))
        assert np.isclose(got["F_peak"][i], f_brute.max(), rtol=1e-4)
        assert bool(got["viable"][i]) == bool(f_brute.max() > (1 - sa) / (sa * sj))
    assert bool(got["viable"][0]) and not bool(got["viable"][1])


def test_allee_viability_roots_match_a_root_finder():
    """N_crit / N_star must be the actual roots of F(N) = F_replacement.

    N_crit is the Allee threshold a propagule must exceed, so it is the denominator of
    the invasion-pinning ratio; a scan artifact there would silently rescale every
    propagule-pressure number.
    """
    from scipy.optimize import brentq

    sa, sj, fmax, g = 0.6, 0.4, 3.0, 69.31
    thr = (1 - sa) / (sa * sj)
    c = fmax * sa * sj / (1.0 - sa + 1e-6) - 1.0
    k = np.array([0.125, 0.075, 0.05])
    got = allee_viability(sa, sj, fmax, k, g)
    assert got["viable"].all()
    for i, k_i in enumerate(k):
        f = lambda n: fmax / (1.0 + c * n / k_i) * (1.0 - np.exp(-g * n)) - thr
        peak = got["N_peak"][i]
        assert np.isclose(got["N_crit"][i], brentq(f, 1e-9, peak), rtol=1e-3)
        assert np.isclose(got["N_star"][i], brentq(f, peak, 20 * k_i), rtol=1e-3)
    # Ordering, and N_star strictly BELOW K: c pins lambda(K) = 1 only when the Allee
    # factor saturates, and it never fully does, so the stable equilibrium sits under K.
    assert np.all(got["N_crit"] < got["N_peak"])
    assert np.all(got["N_peak"] < got["N_star"])
    assert np.all(got["N_star"] < k)


def test_allee_viability_roots_are_nan_when_not_viable():
    """No fold => no equilibria. NaN, not a silent zero that would inflate pressure."""
    got = allee_viability(0.6, 0.4, 3.0, np.array([0.025]), 69.31)
    assert not got["viable"][0]
    assert np.isnan(got["N_crit"][0]) and np.isnan(got["N_star"][0])


def test_allee_viability_is_monotone_in_capacity():
    """Shrinking K can only ever remove viability, never create it."""
    k = np.logspace(-4, 0, 60)
    viable = allee_viability(0.6, 0.4, 3.0, k, 69.31)["viable"]
    assert not viable[0] and viable[-1]
    assert np.all(np.diff(viable.astype(int)) >= 0)  # one crossing, upward


def test_allee_viability_requires_fundamental_suitability():
    """c == 0 (lambda_fundamental <= 1) is never viable, at any K."""
    got = allee_viability(0.6, 0.4, 0.5, np.array([1e-3, 1.0, 1e3]), 69.31)
    assert not got["fundamental_viable"].any() and not got["viable"].any()


def test_allee_viability_does_not_depend_on_the_c_regularizer():
    """The old lambda(N=K) > 1 test flipped with the 1e-6 guard; the fold must not.

    Locks in the reason for the change: perturbing the guard by two orders of
    magnitude moves the old contour by ~35% in K and leaves the fold untouched.
    """
    sa, sj, fmax, g = 0.6, 0.4, 3.0, 69.31

    def fold_threshold(eps):
        k = np.logspace(-4, 0, 4000)
        c = np.maximum(fmax * sa * sj / (1.0 - sa + eps) - 1.0, 0.0)
        n = np.logspace(-5, 1, 512)[:, None] * k
        f = fmax / (1.0 + c * (n / k)) * (1.0 - np.exp(-g * n))
        return k[np.argmax(f.max(axis=0) > (1 - sa) / (sa * sj))]

    def lam_at_k_threshold(eps):
        k = np.logspace(-4, 0, 4000)
        c = np.maximum(fmax * sa * sj / (1.0 - sa + eps) - 1.0, 0.0)
        f_at_k = fmax / (1.0 + c) * (1.0 - np.exp(-g * k))
        lam = (sa + np.sqrt(sa ** 2 + 4.0 * f_at_k * sa * sj)) / 2.0
        return k[np.argmax(lam > 1.0)]

    assert np.isclose(fold_threshold(1e-6), fold_threshold(1e-8), rtol=1e-3)
    assert lam_at_k_threshold(1e-8) > 1.3 * lam_at_k_threshold(1e-6)


def _counterfactual_fixture(T=40, Ny=14, Nx=20, disp_int=0.5, k_mult=8.0):
    """Minimal but REAL forward-model setup: actual kernels, actual simulator."""
    import jax.numpy as jnp
    from src.model.build_kernels import build_simulation_struct
    from src.model.age_forward import forward_sim_age_structured
    from src.vis.age_model_math import realized_equilibrium

    land = np.ones((Ny, Nx)); land[:, :2] = 0.0
    ss = build_simulation_struct(jnp.asarray(land), 27.0, 100.0, 330.0, 0.468, 0.468,
                                [0.0, 132.6, 330.7, 1e9])
    nk = ss["juvenile_fft_kernel_stack"].shape[0]
    rows, cols = np.nonzero(land); n = rows.size
    pop = 20.0; g = np.log(2) / (0.20 / pop); x = cols / Nx
    sa = np.full((T, n), 0.62); sj = np.full((T, n), 0.42)
    fmax = np.tile(1.55 + 2.2 * np.exp(-((x - 0.8) / 0.2) ** 2)
                   + 1.5 * np.exp(-((x - 0.1) / 0.09) ** 2), (T, 1))
    kb = k_mult * np.tile(0.02 + 0.10 * np.exp(-((x - 0.8) / 0.28) ** 2), (T, 1))
    kd = kb.copy(); kd[:, x > 0.55] *= 0.55
    q = np.ones((T, n, nk))
    seed = np.zeros((Ny, Nx)); seed[Ny // 2 - 2:Ny // 2 + 2, 2:5] = 0.08
    # Two candidate release sites (not one), each with its own 8-year pulse vector --
    # matches the production (n_sites, n_years) shape.
    inv_locations = np.array([[Ny // 2, Nx - 4], [Ny // 2 + 1, Nx - 4]])
    inv = np.zeros((2, 8)) + 0.05
    data = dict(land_rows=jnp.asarray(rows), land_cols=jnp.asarray(cols),
                land_mask=jnp.asarray(land), adult_fft_kernel=ss["adult_fft_kernel"],
                juvenile_fft_kernel_stack=ss["juvenile_fft_kernel_stack"],
                adult_edge_correction=ss["adult_edge_correction"],
                juvenile_edge_correction_stack=ss["juvenile_edge_correction_stack"],
                time=T, inv_locations=inv_locations, inv_timestep=18,
                dispersal_target_fraction=0.8, pop_scalar=pop)
    lat = dict(dispersal_random=np.zeros(T), dispersal_logit_intercept=disp_int,
               dispersal_logit_slope=4.0)
    sim = dict(Sa_flat=sa, Sj_flat=sj, Fmax_flat=fmax, K_flat=kd, K_base_flat=kb,
               Q_flat=q, allee_gamma=np.float64(g), initpop_seeded=seed,
               inv_pop_relative=inv, latents=lat)
    c, _, _, _ = realized_equilibrium(sa, sj, fmax, kd, g)
    act, _, _ = forward_sim_age_structured(
        *(jnp.asarray(a) for a in (sa, sj, fmax, kd, c, q)),
        data["land_rows"], data["land_cols"], data["land_mask"],
        data["adult_fft_kernel"], data["juvenile_fft_kernel_stack"],
        data["adult_edge_correction"], data["juvenile_edge_correction_stack"],
        jnp.asarray(seed), jnp.zeros(T), jnp.asarray(inv), T, data["inv_locations"],
        data["inv_timestep"], disp_int, 4.0, jnp.asarray(g), target_fraction=0.8)
    sim["simulated_density"] = np.asarray(act)
    return sim, data, rows, cols, (Ny, Nx), np.asarray(land), T, pop


def test_counterfactual_deleting_the_pulse_cannot_add_birds():
    """With K held fixed, removing the release can only lower density."""
    sim, data, *_ = _counterfactual_fixture()
    b = simulate_no_invasion_counterfactual(sim, data, drop_disease=False)
    a = sim["simulated_density"]
    # Not exactly <= pointwise (dispersal redistributes), but the totals must not rise
    # and no cell may gain materially. This is the invariant the attribution map's
    # sign depends on.
    assert b.sum() <= a.sum() * (1.0 + 1e-9)
    assert np.max(b - a) <= 1e-6 * max(float(a.max()), 1e-12)


def test_counterfactual_dropping_disease_raises_capacity_and_population():
    """The no-epizootic arm must carry at least as much population as the disease arm."""
    sim, data, *_ = _counterfactual_fixture()
    b = simulate_no_invasion_counterfactual(sim, data, drop_disease=False)
    c = simulate_no_invasion_counterfactual(sim, data, drop_disease=True)
    assert c.sum() > b.sum()


def test_counterfactual_figure_and_metrics_survive_an_extinct_landscape():
    """All-NaN reductions must not crash the run -- nanmedian/nanpercentile raised.

    A degenerate or very early checkpoint can simulate a landscape with nothing on
    it; that must yield NaN metrics and a rendered figure, not an exception that
    takes down every other diagnostic (and, in a sweep, every point's metrics.json).
    Densities are zeroed directly rather than by tuning the fixture into extinction,
    so the guard is exercised deterministically.
    """
    import tempfile
    sim, data, rows, cols, shape, land, T, pop = _counterfactual_fixture()
    zero = np.zeros_like(sim["simulated_density"])
    sim = {**sim, "simulated_density": zero}
    years = np.arange(1902, 1902 + T)
    era = (int(years[-10]), int(years[-1]))
    _, viab = modern_viability(sim, years, era, k_key="K_base_flat")
    out = Path(tempfile.mkdtemp()) / "15.png"
    m = plot_counterfactual_attribution(sim, zero.copy(), zero.copy(), viab,
                                        years, rows, cols, shape,
                                        land, out, era, pop_scalar=pop)
    assert out.exists()
    assert np.isnan(m["median_attribution_where_occupied"])
    assert np.isnan(m["max_attribution"])
    assert m["modern_occupied_fraction_actual"] == 0.0


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
    _, _, transition = plot_modern_niche(lam, years, rows, cols, (2, 2),
                                         tmp_path / "niche.png", (2005, 2010), (2000, 2005))
    assert np.isfinite(transition).all()
    plot_modern_rate_maps(sim, years, rows, cols, (2, 2), tmp_path / "rates.png", (2005, 2010))
    fit = plot_fit_diagnostics(sim, {"observed_results": np.array([1., 2., 3., 4.]),
                                     "obs_time_indices": np.array([0, 1, 1, 2])}, years,
                               tmp_path / "fit.png")
    assert fit["n_observations"] == 4
    assert all((tmp_path / name).stat().st_size > 10_000 for name in ("niche.png", "rates.png", "fit.png"))


# ------------------------------------------------------------------------ eras

def test_named_eras_resolve_to_the_intended_spans():
    """The three comparison eras are the spans the figures claim in their titles."""
    from src.vis.age_model_math import ERAS, era_mean, era_span

    years = np.arange(1902, 2026)
    assert era_span("early", years)[2] == (1902, 1915)
    assert era_span("invasion", years)[2] == (1940, 1955)
    assert era_span("modern", years)[2] == (2010, 2025)
    # era_mean averages exactly those rows and reports the span it really used.
    v = np.arange(len(years), dtype=float)[:, None]
    for name, (lo, hi) in ERAS.items():
        mean, span, n = era_mean(v, years, name)
        assert span == (lo, hi) and n == hi - lo + 1
        assert np.isclose(mean[0], np.mean(np.arange(lo - 1902, hi - 1902 + 1)))


def test_era_clips_to_the_timeline_and_refuses_a_disjoint_span():
    from src.vis.age_model_math import era_span

    short = np.arange(1902, 1950)
    # "modern" does not exist on a truncated timeline; silently sliding the window
    # would put a false year range in a figure title.
    with pytest.raises(ValueError, match="does not overlap"):
        era_span("modern", short)
    # A partially-covered era clips, and reports the CLIPPED span.
    assert era_span("invasion", short)[2] == (1940, 1949)


def test_window_years_override_reproduces_the_pre_era_windows():
    """--window-years must still give the trailing/anchored windows it used to."""
    from src.vis.age_model_math import eras_from_window

    years = np.arange(1902, 2026)
    got = eras_from_window(years, 10)
    assert got["modern"] == (2016, 2025)      # trailing 10
    assert got["early"] == (1902, 1911)       # leading 10
    assert got["invasion"] == (1940, 1949)    # 10 anchored at the release


# ----------------------------------------------- single-feature lambda attribution

def _attribution_fixture(beta_s, beta_r, T=40, N=12):
    """Z with a per-feature linear trend, so every feature has a known delta."""
    M = len(beta_s)
    years = np.arange(1990, 1990 + T)
    ramp = np.linspace(0.0, 1.0, T)[:, None, None]
    base = np.linspace(-1.0, 1.0, N)[None, :, None] * np.ones((1, 1, M))
    Z = base + ramp * np.arange(1, M + 1)[None, None, :] * 0.1
    latents = {"w_env": np.stack([beta_s, beta_r, np.zeros(M)], axis=1),
               "alpha_a": 0.3, "alpha_j": -0.4, "alpha_f": 0.2, "alpha_k": 0.1,
               "gamma_a": 1.0, "gamma_j_diff": 0.1, "gamma_f": 1.0, "gamma_k": 1.0}
    rows, cols = np.arange(N), np.zeros(N, dtype=int)
    return ({"Z_gathered": Z}, {"latents": latents}, years, rows, cols, (N, 1))


def test_zero_weight_feature_is_attributed_exactly_zero_delta_lambda(tmp_path):
    """A feature the model does not use cannot be credited with moving lambda."""
    from scripts.viz.map_diagnostics import plot_z_feature_attribution

    beta_s = np.array([0.8, 0.0, -0.5])
    beta_r = np.array([0.3, 0.0, 0.6])       # feature 1 is unused on BOTH manifolds
    data, sim, years, rows, cols, shape = _attribution_fixture(beta_s, beta_r)
    m = plot_z_feature_attribution(data, sim, years, rows, cols, shape,
                                   tmp_path / "05c.png", (2020, 2029), (1990, 1999),
                                   top_n=3)
    by_name = {f["name"]: f["mean_abs_delta_lambda"] for f in m["top_features"]}
    assert by_name["Z_1"] == 0.0
    assert by_name["Z_0"] > 0.0 and by_name["Z_2"] > 0.0


def test_single_active_feature_attribution_equals_the_total_change(tmp_path):
    """With only one feature carrying weight there is no interaction to leak into."""
    from scripts.viz.map_diagnostics import plot_z_feature_attribution

    beta_s = np.array([0.0, 0.7, 0.0])
    beta_r = np.array([0.0, 0.4, 0.0])
    data, sim, years, rows, cols, shape = _attribution_fixture(beta_s, beta_r)
    m = plot_z_feature_attribution(data, sim, years, rows, cols, shape,
                                   tmp_path / "05c.png", (2020, 2029), (1990, 1999),
                                   top_n=3)
    # The decomposition is exact here, so the unattributed residual must vanish.
    assert m["mean_abs_unattributed_residual"] < 1e-12
    assert m["mean_abs_total_delta_lambda"] > 0.0
    assert m["top_features"][0]["name"] == "Z_1"


def test_attribution_reports_a_nonzero_residual_when_features_interact(tmp_path):
    """Nonlinear links mean per-feature deltas do NOT sum to the total; say so."""
    from scripts.viz.map_diagnostics import plot_z_feature_attribution

    beta_s = np.array([1.2, -1.1, 0.9])
    beta_r = np.array([0.9, 1.0, -1.2])
    data, sim, years, rows, cols, shape = _attribution_fixture(beta_s, beta_r)
    m = plot_z_feature_attribution(data, sim, years, rows, cols, shape,
                                   tmp_path / "05c.png", (2020, 2029), (1990, 1999))
    assert m["mean_abs_unattributed_residual"] > 0.0
    assert 0.0 < m["residual_fraction_of_total"] < 1.0


def test_w_env_ranking_orders_by_total_magnitude_and_keeps_signs(tmp_path):
    from scripts.viz.map_diagnostics import plot_w_env_ranking

    w = np.array([[0.1, -0.1, 0.0], [-2.0, 1.0, 0.5], [0.4, 0.4, -0.4]])
    m = plot_w_env_ranking({"latents": {"w_env": w}}, tmp_path / "05b.png")
    ranking = m["w_env_ranking"]
    assert [r["index"] for r in ranking] == [1, 2, 0]     # descending sum|w|
    assert ranking[0]["beta_s"] == -2.0                   # sign preserved
    assert (tmp_path / "05b.png").stat().st_size > 10_000
