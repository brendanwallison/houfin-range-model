from pathlib import Path

import numpy as np

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
    _, viab = modern_viability(sim, 10, k_key="K_base_flat")
    out = Path(tempfile.mkdtemp()) / "15.png"
    m = plot_counterfactual_attribution(sim, zero.copy(), zero.copy(), viab,
                                        np.arange(1902, 1902 + T), rows, cols, shape,
                                        land, out, 10, pop_scalar=pop)
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
    _, _, transition = plot_modern_niche(lam, years, rows, cols, (2, 2), tmp_path / "niche.png", 2)
    assert np.isfinite(transition).all()
    plot_modern_rate_maps(sim, years, rows, cols, (2, 2), tmp_path / "rates.png", 2)
    fit = plot_fit_diagnostics(sim, {"observed_results": np.array([1., 2., 3., 4.]),
                                     "obs_time_indices": np.array([0, 1, 1, 2])}, years,
                               tmp_path / "fit.png")
    assert fit["n_observations"] == 4
    assert all((tmp_path / name).stat().st_size > 10_000 for name in ("niche.png", "rates.png", "fit.png"))
